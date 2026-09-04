import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch


DEFAULT_COMPONENTS = [
    ("single", 0, 384),
    ("pair", 384, 512),
    ("lddt_logits", 512, 562),
    ("plddt", 562, 563),
]


def load_tensor(path: Path) -> torch.Tensor:
    x = torch.load(path, map_location="cpu")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{path} does not contain a torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"{path} must be 2D, got shape {tuple(x.shape)}")
    return x


def parse_component(spec: str) -> Tuple[str, int, int]:
    """
    Parse NAME:START:END, where END is exclusive.
    Example: single:0:384
    """
    try:
        name, start, end = spec.split(":")
        start_i = int(start)
        end_i = int(end)
    except Exception as e:
        raise ValueError(
            f"Bad component spec '{spec}'. Expected NAME:START:END, e.g. single:0:384"
        ) from e

    if start_i < 0 or end_i <= start_i:
        raise ValueError(f"Bad slice for '{spec}': start must be >= 0 and end > start")

    return name, start_i, end_i


def build_component_map(user_specs: List[str]) -> Dict[str, Tuple[int, int]]:
    specs = [parse_component(s) for s in user_specs] if user_specs else DEFAULT_COMPONENTS
    comp_map: Dict[str, Tuple[int, int]] = {}
    for name, start, end in specs:
        if name in comp_map:
            raise ValueError(f"Duplicate component name: {name}")
        comp_map[name] = (start, end)
    return comp_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slice precomputed [L, D] embeddings into selected named components."
    )
    parser.add_argument(
        "--in_dir",
        type=Path,
        default=Path("data/embeddings_aft"),
        help="Directory with input .pt tensors of shape [L, D]",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory for output sliced .pt tensors",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        required=True,
        help="Names of components to keep, in output order. Example: --parts single plddt",
    )
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        help=(
            "Component definition NAME:START:END (END exclusive). "
            "Repeatable. If omitted, defaults to AFT layout: "
            "single:0:384, pair:384:512, lddt_logits:512:562, plddt:562:563"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    comp_map = build_component_map(args.component)

    unknown = [p for p in args.parts if p not in comp_map]
    if unknown:
        raise ValueError(
            f"Unknown parts: {unknown}. Available parts: {list(comp_map.keys())}"
        )

    files = sorted(args.in_dir.glob("*.pt"))
    if not files:
        raise RuntimeError(f"No .pt files found in {args.in_dir}")

    component_dims = {name: comp_map[name][1] - comp_map[name][0] for name in args.parts}
    total_dim = sum(component_dims.values())

    print("Selected parts:")
    for name in args.parts:
        start, end = comp_map[name]
        print(f"  - {name}: cols [{start}:{end}] -> dim {end - start}")
    print(f"Output dim: {total_dim}")

    written = 0
    skipped = 0
    errors = 0

    for path in files:
        out_path = args.out_dir / path.name

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            x = load_tensor(path)
            L, D = x.shape

            chunks = []
            for name in args.parts:
                start, end = comp_map[name]
                if end > D:
                    raise ValueError(
                        f"{path.name} has shape {tuple(x.shape)}, but part '{name}' "
                        f"needs cols [{start}:{end}]"
                    )
                chunks.append(x[:, start:end])

            y = torch.cat(chunks, dim=1)
            torch.save(y.contiguous(), out_path)
            print(f"[OK] {path.name}: {tuple(x.shape)} -> {tuple(y.shape)}")
            written += 1

        except Exception as e:
            print(f"[ERR] {path.name}: {e}")
            errors += 1

    print("\nDone.")
    print(f"Written: {written}")
    print(f"Skipped: {skipped}")
    print(f"Errors:  {errors}")
    print("\nFinal output layout:")
    for name in args.parts:
        print(f"  {name}: {component_dims[name]}")
    print(f"  total: {total_dim}")


if __name__ == "__main__":
    main()
