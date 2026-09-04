"""
Generate ESM-C embeddings (per residue) and save one .pt file per sequence.
Filename is md5 hash of the amino-acid sequence.
"""

from hashlib import md5
import os
import argparse
import pathlib

import torch
from tqdm.auto import tqdm

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


def hash_aa_string(string: str) -> str:
    return md5(string.encode()).hexdigest()


def generate_esmc_embeddings(
    fasta_file: pathlib.Path,
    esm_embeddings_dir: pathlib.Path,
    model_name: str = "esmc_600m",
    repr_layer: int = 36,
    max_tokens: int = 2048,  # ESM-C context (we treat as max residues here)
) -> None:
    esm_model = ESMC.from_pretrained(model_name)
    esm_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm_model = esm_model.to(device)

    os.makedirs(esm_embeddings_dir, exist_ok=True)

    skipped_cached = 0
    skipped_long = 0

    # Ask for embeddings; hidden states are optional depending on installed esm version
    cfg_kwargs = {"sequence": True, "return_embeddings": True}
    ann = getattr(LogitsConfig, "__annotations__", {}) or {}
    if "return_hidden_states" in ann:
        cfg_kwargs["return_hidden_states"] = True
    cfg = LogitsConfig(**cfg_kwargs)

    # Minimal FASTA reading (streaming)
    def iter_fasta(path: pathlib.Path):
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

    items = iter_fasta(fasta_file)
    pbar = tqdm(items, desc="Embeddings", dynamic_ncols=True)

    with torch.no_grad():
        for label, seq in pbar:
            out_path = os.path.join(str(esm_embeddings_dir), f"{hash_aa_string(seq)}.pt")

            if os.path.isfile(out_path):
                skipped_cached += 1
                pbar.set_postfix(cached=skipped_cached, long=skipped_long)
                continue

            # Here we limit by residues length (not token length)
            if len(seq) > max_tokens:
                skipped_long += 1
                pbar.set_postfix(cached=skipped_cached, long=skipped_long)
                continue

            protein = ESMProtein(sequence=seq)
            protein_tensor = esm_model.encode(protein)

            # Some versions return an object/tensor already on device, but this is safe:
            if hasattr(protein_tensor, "to"):
                protein_tensor = protein_tensor.to(device)

            out = esm_model.logits(protein_tensor, cfg)

            # Prefer specific layer if hidden_states exist; otherwise out.embeddings (usually last layer)
            hs = getattr(out, "hidden_states", None)
            if hs is not None and isinstance(hs, (list, tuple)) and len(hs) > 0:
                rep = hs[repr_layer] if 0 <= repr_layer < len(hs) else hs[-1]
            else:
                rep = out.embeddings

            # rep: [1, T, D] or [T, D]
            if rep.dim() == 3:
                rep = rep[0]  # [T, D]

            # Trim BOS/EOS if present: typically T == L+2
            if rep.size(0) == len(seq) + 2:
                res = rep[1:-1, :]
            else:
                res = rep[: len(seq), :]

            torch.save(res.float().cpu(), out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta_file", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--model", default="esmc_600m")
    parser.add_argument("--repr_layer", type=int, default=36)
    parser.add_argument("--max_tokens", type=int, default=2048)
    args = parser.parse_args()

    generate_esmc_embeddings(
        fasta_file=args.fasta_file,
        esm_embeddings_dir=args.output_dir,
        model_name=args.model,
        repr_layer=args.repr_layer,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
