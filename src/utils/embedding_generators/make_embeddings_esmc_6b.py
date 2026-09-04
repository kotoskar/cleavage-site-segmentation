"""
Generate ESM-C 6B embeddings (per residue) via Forge API and save one .pt file per sequence.
Filename is md5 hash of the amino-acid sequence.

If daily credits are exhausted -> prints message and exits (already computed files stay cached).
"""

from hashlib import md5
import os
import argparse
import pathlib

import torch
from tqdm.auto import tqdm

from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk.api import ESMProtein, LogitsConfig, ESMProteinError
from tenacity import RetryError


def hash_aa_string(string: str) -> str:
    return md5(string.encode()).hexdigest()


def generate_esmc6b_embeddings(
    fasta_file: pathlib.Path,
    esm_embeddings_dir: pathlib.Path,
    token: str,
    model_name: str = "esmc-6b-2024-12",
    forge_url: str = "https://forge.evolutionaryscale.ai",
    max_residues: int = 2048,
) -> None:
    if not token:
        raise ValueError("No Forge token. Pass --token or set FORGE_TOKEN env var.")

    client = ESM3ForgeInferenceClient(model=model_name, url=forge_url, token=token)

    os.makedirs(esm_embeddings_dir, exist_ok=True)

    skipped_cached = 0
    skipped_long = 0
    skipped_err = 0

    # Minimal FASTA iterator (streaming)
    def fasta_iter(path: pathlib.Path):
        label = None
        seq_parts = []
        with open(path, "r", encoding="utf-8") as f:
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

    pbar = tqdm(fasta_iter(fasta_file), desc="Embeddings", dynamic_ncols=True)

    with torch.no_grad():
        for label, seq in pbar:
            out_path = os.path.join(str(esm_embeddings_dir), f"{hash_aa_string(seq)}.pt")

            if os.path.isfile(out_path):
                skipped_cached += 1
                pbar.set_postfix(cached=skipped_cached, long=skipped_long, err=skipped_err)
                continue

            if len(seq) > max_residues:
                skipped_long += 1
                pbar.set_postfix(cached=skipped_cached, long=skipped_long, err=skipped_err)
                continue

            protein = ESMProtein(sequence=seq)

            # encode can also return ESMProteinError
            protein_tensor = client.encode(protein)
            if isinstance(protein_tensor, ESMProteinError):
                skipped_err += 1
                pbar.set_postfix(cached=skipped_cached, long=skipped_long, err=skipped_err)
                continue

            try:
                out = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            except RetryError as e:
                # Forge SDK retries internally; when credits are over it ends here.
                msg = str(e)
                try:
                    last = e.last_attempt
                    if last is not None:
                        r = last.result()
                        if isinstance(r, ESMProteinError):
                            msg = getattr(r, "message", None) or getattr(r, "detail", None) or str(r)
                except Exception:
                    pass

                # Stop cleanly if daily credits are exhausted
                if "daily credit limit" in msg.lower() or "exceeded your daily credit limit" in msg.lower():
                    print(f"\n[STOP] Forge credits exhausted: {msg}\n"
                          f"Already computed embeddings are cached in: {esm_embeddings_dir}\n"
                          f"Rerun later to continue from where you stopped.")
                    return

                # Otherwise treat as a regular error and continue
                skipped_err += 1
                pbar.set_postfix(cached=skipped_cached, long=skipped_long, err=skipped_err)
                continue

            # Some versions might return ESMProteinError directly (just in case)
            if isinstance(out, ESMProteinError):
                msg = getattr(out, "message", None) or getattr(out, "detail", None) or str(out)
                if "daily credit limit" in msg.lower() or "exceeded your daily credit limit" in msg.lower():
                    print(f"\n[STOP] Forge credits exhausted: {msg}\n"
                          f"Already computed embeddings are cached in: {esm_embeddings_dir}\n"
                          f"Rerun later to continue from where you stopped.")
                    return
                skipped_err += 1
                pbar.set_postfix(cached=skipped_cached, long=skipped_long, err=skipped_err)
                continue

            rep = out.embeddings  # expected per-token/per-residue embeddings
            if not isinstance(rep, torch.Tensor):
                rep = torch.as_tensor(rep)

            # rep: [1, T, D] or [T, D]
            if rep.dim() == 3:
                rep = rep[0]

            # Trim BOS/EOS if present
            if rep.size(0) == len(seq) + 2:
                res = rep[1:-1, :]
            else:
                res = rep[: len(seq), :]

            torch.save(res.float().cpu(), out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta_file", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--token", default=os.environ.get("FORGE_TOKEN", "***REMOVED***"))
    parser.add_argument("--model", default="esmc-6b-2024-12")
    parser.add_argument("--forge_url", default="https://forge.evolutionaryscale.ai")
    parser.add_argument("--max_residues", type=int, default=2048)
    args = parser.parse_args()

    generate_esmc6b_embeddings(
        fasta_file=args.fasta_file,
        esm_embeddings_dir=args.output_dir,
        token=args.token,
        model_name=args.model,
        forge_url=args.forge_url,
        max_residues=args.max_residues,
    )


if __name__ == "__main__":
    main()
