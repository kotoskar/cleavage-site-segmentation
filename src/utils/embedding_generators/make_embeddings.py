"""
Generate ESM2 embeddings (per residue) via HuggingFace Transformers and save one .pt file per sequence.
Filename is md5 hash of the amino-acid sequence.
"""

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


def generate_esm2_embeddings(
    fasta_file: pathlib.Path,
    esm_embeddings_dir: pathlib.Path,
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    repr_layer: int = 33,
    max_tokens: int = 0,  # 0 => auto from model.config.max_position_embeddings
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
        raise ValueError(f"repr_layer={repr_layer} is invalid for this model (num_hidden_layers={n_layers}). "
                         f"Use 1..{n_layers} for transformer layers (33 is last for esm2_t33).")

    os.makedirs(esm_embeddings_dir, exist_ok=True)

    skipped_cached = 0
    skipped_long = 0

    items = list(iter_fasta(fasta_file))
    pbar = tqdm(items, desc="Embeddings", dynamic_ncols=True)

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
            # hidden_states[0] = embedding output, hidden_states[1..n_layers] = transformer layers
            hs = out.hidden_states
            rep = hs[repr_layer][0]  # [T, D]

            # Trim special tokens if present (typical: T == L+2)
            if rep.size(0) == len(seq) + 2:
                res = rep[1:-1, :]
            else:
                # fallback: keep as much as possible, but still aim for per-residue
                # (some tokenizers may behave differently; this avoids silent shape corruption)
                res = rep[: len(seq), :]

            torch.save(res.float().cpu(), out_path)


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
    args = parser.parse_args()

    generate_esm2_embeddings(
        fasta_file=args.fasta_file,
        esm_embeddings_dir=args.output_dir,
        model_name=args.model,
        repr_layer=args.repr_layer,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
