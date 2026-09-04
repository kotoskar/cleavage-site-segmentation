#!/usr/bin/env python3
"""
Merge multiple per-residue embedding directories into one concatenated directory.

Intended location:
    src/utils/merge_embeddings_generic.py

Run from repository root, for example:
    python src/utils/merge_embeddings_generic.py \
      --component esm2:data/embeddings_esm2:1280 \
      --component aho:data/embeddings_aho:86 \
      --out_dir data/embeddings_esm2_aho \
      --overwrite

Every component must contain .pt tensors named identically, one per protein. Tensors
may be stored as [L, C] or [C, L]; this script normalizes them to [L, C] and then
concatenates on C. Output tensors are [L, sum(C)].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch


def load_tensor(path: Path) -> torch.Tensor:
    x = torch.load(path, map_location="cpu")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{path} does not contain a torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"{path} must be 2D, got shape {tuple(x.shape)}")
    return x.float()


def ensure_last_dim(x: torch.Tensor, expected_dim: int, path: Path) -> torch.Tensor:
    """Ensure tensor is shaped [L, expected_dim]. If [expected_dim, L], transpose it."""
    if x.shape[1] == expected_dim:
        return x.contiguous()
    if x.shape[0] == expected_dim:
        return x.transpose(0, 1).contiguous()
    raise ValueError(
        f"{path} has incompatible shape {tuple(x.shape)}; expected [L, {expected_dim}] or [{expected_dim}, L]"
    )


def parse_components(values: List[str]) -> List[Tuple[str, Path, int]]:
    comps = []
    for raw in values:
        try:
            name, directory, dim = raw.split(":", 2)
        except ValueError:
            raise ValueError(
                f"Bad --component {raw!r}. Expected NAME:DIR:DIM, e.g. esm2:data/embeddings_esm2:1280"
            )
        comps.append((name, Path(directory), int(dim)))
    return comps


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate per-residue .pt embedding directories.")
    parser.add_argument(
        "--component",
        action="append",
        required=True,
        help="Component spec NAME:DIR:DIM. Repeat flag for each component. First component defines file list.",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Skip proteins missing in a non-base component. Default: count as error and skip that protein.",
    )
    args = parser.parse_args()

    components = parse_components(args.component)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _, base_dir, _ = components[0]
    base_files = sorted(base_dir.glob("*.pt"))
    if not base_files:
        raise RuntimeError(f"No .pt files found in base component directory: {base_dir}")

    merged = skipped = missing = errors = 0
    total_dim = sum(dim for _, _, dim in components)

    print("Components:")
    for name, directory, dim in components:
        print(f"  - {name}: dir={directory}, dim={dim}")
    print(f"Target merged dim: {total_dim}\n")

    for base_path in base_files:
        out_path = args.out_dir / base_path.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        tensors = []
        comp_shapes = []
        ref_len = None
        failed = False

        for name, directory, dim in components:
            path = directory / base_path.name
            if not path.exists():
                print(f"[MISS] {base_path.name}: no file for component {name!r} in {directory}")
                missing += 1
                failed = True
                break
            try:
                x = ensure_last_dim(load_tensor(path), dim, path)
                if ref_len is None:
                    ref_len = x.shape[0]
                elif x.shape[0] != ref_len:
                    raise ValueError(
                        f"Length mismatch for {base_path.name}: component {name!r} has L={x.shape[0]}, expected {ref_len}"
                    )
                tensors.append(x)
                comp_shapes.append(f"{name}:{tuple(x.shape)}")
            except Exception as e:
                print(f"[ERR] {base_path.name}: {e}")
                errors += 1
                failed = True
                break

        if failed:
            continue

        merged_tensor = torch.cat(tensors, dim=1)
        torch.save(merged_tensor, out_path)
        print(f"[OK] {base_path.name}: {' + '.join(comp_shapes)} -> {tuple(merged_tensor.shape)}")
        merged += 1

    meta = {
        "components": [{"name": n, "dir": str(d), "dim": dim} for n, d, dim in components],
        "total_dim": total_dim,
        "merged": merged,
        "skipped": skipped,
        "missing": missing,
        "errors": errors,
    }
    with open(args.out_dir / "merge_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")
    print(f"Merged:  {merged}")
    print(f"Skipped: {skipped}")
    print(f"Missing: {missing}")
    print(f"Errors:  {errors}")
    print(f"Total merged dimension: {total_dim}")


if __name__ == "__main__":
    main()
