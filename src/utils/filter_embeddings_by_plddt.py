"""
Filter AFT embedding files by mean per-residue pLDDT.

Expected tensor layout: [L, D] where the last column is pLDDT on a 0–100 scale
(produced by make_embeddings_aft_worker.py with default --features including plddt).

Usage:
    python filter_embeddings_by_plddt.py \
        --in_dir data/embeddings_aft \
        --out_dir data/embeddings_aft_filtered \
        --threshold 70

Output directory name is auto-suffixed with the threshold if --out_dir is not given:
    data/embeddings_aft_plddt70
"""

import argparse
import shutil
from pathlib import Path

import torch


def mean_plddt(path: Path) -> float:
    x = torch.load(path, map_location="cpu", weights_only=True)
    return float(x[:, -1].mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=Path, required=True,
                        help="Directory with AFT .pt embeddings")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Output directory (default: <in_dir>_plddt<threshold>)")
    parser.add_argument("--threshold", type=float, default=70.0,
                        help="Minimum mean pLDDT to keep a protein (0–100, default 70)")
    parser.add_argument("--plddt_col", type=int, default=-1,
                        help="Column index of the pLDDT feature (default: -1, last column)")
    args = parser.parse_args()

    if args.out_dir is None:
        suffix = f"_plddt{int(args.threshold)}" if args.threshold == int(args.threshold) else f"_plddt{args.threshold}"
        args.out_dir = args.in_dir.parent / (args.in_dir.name + suffix)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.in_dir.glob("*.pt"))
    if not files:
        print(f"No .pt files found in {args.in_dir}")
        return

    kept = skipped = errors = 0
    for path in files:
        try:
            x = torch.load(path, map_location="cpu", weights_only=True)
            score = float(x[:, args.plddt_col].mean().item())
            if score >= args.threshold:
                shutil.copy2(path, args.out_dir / path.name)
                kept += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"[ERR] {path.name}: {e}")
            errors += 1

    total = kept + skipped
    print(f"Threshold: {args.threshold}  |  kept: {kept}/{total}  |  skipped: {skipped}  |  errors: {errors}")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
