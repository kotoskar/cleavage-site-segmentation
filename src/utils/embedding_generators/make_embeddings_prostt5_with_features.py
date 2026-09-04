"""
Generate ProstT5 embeddings (per residue) via HuggingFace Transformers and save one .pt file per sequence.
Filename is md5 hash of the amino-acid sequence.

Output tensor shape per sequence: [L, D] where L = protein length, D = model hidden size (1024 for ProstT5).
"""

from hashlib import md5
import os
import argparse
import pathlib
from typing import Iterator, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel


def hash_aa_string(string: str) -> str:
    return md5(string.encode()).hexdigest()


def iter_fasta(path: pathlib.Path) -> Iterator[Tuple[str, str]]:
    """Tiny FASTA reader: yields (label, sequence)."""
    label = None
    seq_parts = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if label is not None:
                    yield label, "".join(seq_parts)
                label = line[1:].strip() or "seq"
                seq_parts = []
            else:
                seq_parts.append(line.replace(" ", "").upper())
    if label is not None:
        yield label, "".join(seq_parts)


# --- Cheap deterministic per-residue features ---

# Kyte-Doolittle hydropathy index (common canonical values)
_HYDRO_KD = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

# Simple charge proxy near neutral pH
# (Very cheap; if you want pH-dependent charge, do it in downstream code.)
_CHARGE7 = {"K": 1.0, "R": 1.0, "D": -1.0, "E": -1.0, "H": 0.1}

# Sets for cheap boolean features
_AROMATIC = {"F", "W", "Y"}
_PRO = {"P"}
_GLY = {"G"}

# "Polar" here means polar OR charged (helps many segmentation tasks)
_POLAR = {"S", "T", "N", "Q", "C", "Y", "D", "E", "K", "R", "H"}


def _sliding_mean_1d(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Edge-aware sliding mean using variable window near ends.

    x: [L] float tensor
    radius: window radius, so window size = 2*radius+1
    returns: [L]
    """
    L = int(x.numel())
    if L == 0:
        return x
    # cumulative sum with leading 0
    cs = torch.zeros(L + 1, dtype=x.dtype, device=x.device)
    cs[1:] = torch.cumsum(x, dim=0)

    idx = torch.arange(L, device=x.device)
    start = torch.clamp(idx - radius, min=0)
    end = torch.clamp(idx + radius + 1, max=L)

    # sum over [start, end)
    s = cs[end] - cs[start]
    denom = (end - start).to(x.dtype)
    return s / denom


def _sliding_sum_1d(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Edge-aware sliding sum using variable window near ends."""
    L = int(x.numel())
    if L == 0:
        return x
    cs = torch.zeros(L + 1, dtype=x.dtype, device=x.device)
    cs[1:] = torch.cumsum(x, dim=0)

    idx = torch.arange(L, device=x.device)
    start = torch.clamp(idx - radius, min=0)
    end = torch.clamp(idx + radius + 1, max=L)
    return cs[end] - cs[start]


def compute_features(seq: str, device) -> torch.Tensor:
    """Compute per-residue features from AA sequence.

    Returns: float32 tensor of shape [L, K] (K=10)
    """
    L = len(seq)
    if L == 0:
        return torch.zeros((0, 10), dtype=torch.float32, device=device)

    # Basic per-residue scalars
    # Map char -> value with defaults
    hydro = torch.tensor([_HYDRO_KD.get(a, 0.0) for a in seq], dtype=torch.float32, device=device)
    charge = torch.tensor([_CHARGE7.get(a, 0.0) for a in seq], dtype=torch.float32, device=device)

    is_polar = torch.tensor([1.0 if a in _POLAR else 0.0 for a in seq], dtype=torch.float32, device=device)
    is_arom = torch.tensor([1.0 if a in _AROMATIC else 0.0 for a in seq], dtype=torch.float32, device=device)
    is_pro = torch.tensor([1.0 if a in _PRO else 0.0 for a in seq], dtype=torch.float32, device=device)
    is_gly = torch.tensor([1.0 if a in _GLY else 0.0 for a in seq], dtype=torch.float32, device=device)

    # Positional features
    if L == 1:
        pos = torch.zeros(L, dtype=torch.float32, device=device)
    else:
        pos = torch.arange(L, dtype=torch.float32, device=device) / float(L - 1)
    rev_pos = 1.0 - pos

    # Windowed features
    mean_hydro_w7 = _sliding_mean_1d(hydro, radius=3)  # window 7
    net_charge_w9 = _sliding_sum_1d(charge, radius=4)  # window 9

    # Stack in fixed order
    feats = torch.stack(
        [
            pos,
            rev_pos,
            hydro,
            charge,
            is_polar,
            is_arom,
            is_pro,
            is_gly,
            mean_hydro_w7,
            net_charge_w9,
        ],
        dim=-1,
    )
    return feats


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _prep_for_prostt5(seq: str) -> str:
    """Prepare AA sequence for (Prost)T5 tokenizer.

    ProstT5 expects space-separated amino acids and typically replaces non-standard residues.
    """
    seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
    # Leading space matches common ProstT5 embedding scripts and keeps slicing stable.
    return " " + " ".join(list(seq))


def generate_prostt5_embeddings(
    fasta_file: pathlib.Path,
    embeddings_dir: pathlib.Path,
    model_name: str = "Rostlab/ProstT5_fp16",
    repr_layer: int = -1,  # -1 => last encoder layer
    max_tokens: int = 0,  # 0 => auto (default 1024 tokens incl. special tokens)
    add_features: bool = True,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()

    device = _pick_device()
    model = model.to(device)

    # Safety: if someone uses fp16 weights on CPU, cast to float32 to avoid unsupported ops.
    if device == "cpu" and str(getattr(model, "dtype", "")) == "torch.float16":
        model = model.float()

    # Auto max tokens from config (includes special tokens like EOS)
    if max_tokens <= 0:
        # T5 configs often show n_positions=512, but ProstT5 is commonly used with longer sequences.
        # We pick a practical default (1024 incl. special tokens) and let you override via --max_tokens.
        max_tokens = max(int(getattr(tokenizer, "model_max_length", 0)) or 512, 1024)

    # Validate repr_layer for this model
    n_layers = int(getattr(model.config, "num_layers", 0))
    if repr_layer == -1:
        repr_layer = n_layers  # last transformer layer
    if n_layers and not (0 <= repr_layer <= n_layers):
        raise ValueError(
            f"repr_layer={repr_layer} is invalid for this model (num_layers={n_layers}). "
            f"Use 1..{n_layers} for transformer layers (or -1 for last)."
        )

    os.makedirs(embeddings_dir, exist_ok=True)

    skipped_cached = 0
    skipped_long = 0

    items = list(iter_fasta(fasta_file))
    pbar = tqdm(items, desc="Embeddings", dynamic_ncols=True)

    # We'll infer base embedding dim D once we process the first saved sequence.
    base_D = None  # type: int | None
    K = 10 if add_features else 0

    with torch.no_grad():
        for label, seq in pbar:
            out_path = os.path.join(str(embeddings_dir), f"{hash_aa_string(seq)}.pt")

            if os.path.isfile(out_path):
                skipped_cached += 1
                pbar.set_postfix(cached=skipped_cached, long=skipped_long)
                continue

            seq_len = len(seq)
            # Tokenize (Prost)T5 expects spaced tokens
            seq_for_model = _prep_for_prostt5(seq)
            inputs = tokenizer(seq_for_model, return_tensors="pt", add_special_tokens=True)

            input_ids = inputs["input_ids"]
            if input_ids.size(1) > max_tokens:
                skipped_long += 1
                pbar.set_postfix(cached=skipped_cached, long=skipped_long)
                continue

            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            out = model(**inputs, output_hidden_states=True, return_dict=True)

            # hidden_states: tuple len = 1 + num_layers
            hs = out.hidden_states
            rep = hs[repr_layer][0]  # [T, D]

            # Trim special tokens to get per-residue [L, D]
            # ProstT5/T5 tokenization typically adds an EOS token at the end.
            T = int(rep.size(0))
            L = int(seq_len)

            if T == L + 1:
                # likely: residues + EOS
                res = rep[:L, :]
            elif T == L + 2:
                # likely: (extra first token) + residues + EOS
                res = rep[1 : L + 1, :]
            else:
                # Fallback: use attention_mask to estimate real token span, then slice best-effort
                attn = inputs.get("attention_mask")
                ids = inputs.get("input_ids")
                if attn is not None:
                    # Number of non-pad tokens
                    t_eff = int(attn[0].sum().item())

                    # Drop EOS if it is present at the end
                    eos_id = int(getattr(model.config, "eos_token_id", 1))
                    if ids is not None and t_eff > 0 and int(ids[0, t_eff - 1].item()) == eos_id:
                        t_eff_noeos = t_eff - 1
                    else:
                        t_eff_noeos = t_eff

                    # Some tokenizers add an extra leading token; compute offset to recover L residues.
                    extra = max(0, t_eff_noeos - L)
                    start = extra
                    res = rep[start : start + L, :]
                else:
                    res = rep[:L, :]

            if base_D is None:
                base_D = int(res.size(-1))

            if add_features:
                feats = compute_features(seq, device=res.device)  # [L, 10]
                if feats.size(0) != res.size(0):
                    feats = feats[: res.size(0), :]
                res = torch.cat([res, feats.to(res.dtype)], dim=-1)

            torch.save(res.float().cpu(), out_path)

    if base_D is None:
        base_D = int(getattr(model.config, "d_model", 0) or getattr(model.config, "hidden_size", 0) or 0)

    total_D = int(base_D) + K
    print(f"Done. Base D={base_D}. Added K={K}. Output D_total={total_D}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta_file", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--model", default="Rostlab/ProstT5_fp16")
    parser.add_argument(
        "--repr_layer",
        type=int,
        default=-1,
        help="Which encoder layer to export: -1 = last (recommended). For ProstT5 num_layers=24, so last is 24.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=0,
        help="Max tokens including special tokens. 0 = auto from model config (n_positions) or tokenizer.",
    )
    parser.add_argument(
        "--no_features",
        action="store_true",
        help="Disable concatenation of cheap AA-derived features (output stays [L, D_model]).",
    )
    args = parser.parse_args()

    generate_prostt5_embeddings(
        fasta_file=args.fasta_file,
        embeddings_dir=args.output_dir,
        model_name=args.model,
        repr_layer=args.repr_layer,
        max_tokens=args.max_tokens,
        add_features=not args.no_features,
    )


if __name__ == "__main__":
    main()