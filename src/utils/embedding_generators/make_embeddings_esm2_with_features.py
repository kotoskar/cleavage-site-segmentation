"""Generate ESM2 embeddings (per residue) and optionally append cheap per-residue features.

This script is intentionally kept very close in structure to make_embeddings.py:
  - reads a FASTA
  - runs HuggingFace ESM2
  - saves one .pt per sequence (filename = md5(sequence))

If --add_features is enabled (default), we concatenate a small set of deterministic
features derived ONLY from the AA sequence. Output tensor shape becomes:
  [L, D_total] where D_total = D_esm + K.

The default feature set is designed to be useful for per-residue segmentation tasks
(e.g., cleavage peptide segmentation) while remaining extremely cheap at inference.

K (default) = 10 features per residue:
  1) pos_norm            in [0,1]
  2) rev_pos_norm        in [0,1]
  3) hydro_kd            (Kyte-Doolittle hydropathy)
  4) charge7             (K/R=+1, D/E=-1, H=+0.1)
  5) is_polar            (polar or charged AA)
  6) is_aromatic         (F/W/Y)
  7) is_pro              (P)
  8) is_gly              (G)
  9) mean_hydro_w7       (local mean hydro, window 7)
 10) net_charge_w9       (local sum charge, window 9)

Notes:
  - Non-standard residues (X,B,Z,U,O, etc.) are mapped to 0 for numeric features,
    and to 0 for boolean features.
  - Features are concatenated to the RIGHT of the ESM2 embedding.
"""

from __future__ import annotations

from hashlib import md5
import os
import argparse
import pathlib
from typing import Iterator, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


def hash_aa_string(string: str) -> str:
    return md5(string.encode()).hexdigest()


def iter_fasta(path: pathlib.Path) -> Iterator[Tuple[str, str]]:
    """Tiny FASTA reader: yields (label, sequence)."""
    label = None
    seq_parts: list[str] = []
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


def compute_features(seq: str, device: str | torch.device) -> torch.Tensor:
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


def generate_esm2_embeddings(
    fasta_file: pathlib.Path,
    esm_embeddings_dir: pathlib.Path,
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    repr_layer: int = 33,
    max_tokens: int = 0,  # 0 => auto from model.config.max_position_embeddings
    add_features: bool = True,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Auto max tokens from config (includes special tokens)
    if max_tokens <= 0:
        max_tokens = int(getattr(model.config, "max_position_embeddings", 0)) or 1026

    # Validate repr_layer for this model
    n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if n_layers and not (0 <= repr_layer <= n_layers):
        raise ValueError(
            f"repr_layer={repr_layer} is invalid for this model (num_hidden_layers={n_layers}). "
            f"Use 1..{n_layers} for transformer layers."
        )

    os.makedirs(esm_embeddings_dir, exist_ok=True)

    skipped_cached = 0
    skipped_long = 0

    items = list(iter_fasta(fasta_file))
    pbar = tqdm(items, desc="Embeddings", dynamic_ncols=True)

    # We'll infer base D once we process the first sequence.
    base_D: int | None = None
    K = 10 if add_features else 0

    with torch.no_grad():
        for label, seq in pbar:
            out_path = os.path.join(str(esm_embeddings_dir), f"{hash_aa_string(seq)}.pt")

            if os.path.isfile(out_path):
                skipped_cached += 1
                pbar.set_postfix(cached=skipped_cached, long=skipped_long)
                continue

            # Tokenize one sequence (adds BOS/EOS-like special tokens)
            inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
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

            # Trim special tokens if present (typical: T == L+2)
            if rep.size(0) == len(seq) + 2:
                res = rep[1:-1, :]
            else:
                res = rep[: len(seq), :]

            if base_D is None:
                base_D = int(res.size(-1))

            if add_features:
                feats = compute_features(seq, device=res.device)  # [L, K]
                # Safety: if any weird tokenizer behavior changed lengths
                if feats.size(0) != res.size(0):
                    feats = feats[: res.size(0), :]
                res = torch.cat([res, feats.to(res.dtype)], dim=-1)

            # Save as float32 on CPU
            torch.save(res.float().cpu(), out_path)

    if base_D is None:
        base_D = int(getattr(model.config, "hidden_size", 0)) or 0

    total_D = base_D + K
    print(f"Done. Base D={base_D}. Added K={K}. Output D_total={total_D}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta_file", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--repr_layer", type=int, default=33)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=0,
        help="Max tokens including special tokens. 0 = auto from model config.",
    )
    parser.add_argument(
        "--no_features",
        action="store_true",
        help="Disable concatenation of cheap AA-derived features (output stays [L, D_esm]).",
    )
    args = parser.parse_args()

    generate_esm2_embeddings(
        fasta_file=args.fasta_file,
        esm_embeddings_dir=args.output_dir,
        model_name=args.model,
        repr_layer=args.repr_layer,
        max_tokens=args.max_tokens,
        add_features=not args.no_features,
    )


if __name__ == "__main__":
    main()
