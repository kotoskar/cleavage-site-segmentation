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

            torch.save(res.float().cpu(), out_path)


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
    args = parser.parse_args()

    generate_prostt5_embeddings(
        fasta_file=args.fasta_file,
        embeddings_dir=args.output_dir,
        model_name=args.model,
        repr_layer=args.repr_layer,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
