#!/usr/bin/env python3
"""
Build a normalized Aho-Corasick dictionary table from the DeepPeptide/UniProt 2022 files.

Expected project layout:

    <repo>/
      data/
        preprocessing/
          preprocess_uniprot_2022_for_aho.py   # this script
        uniprot_2022/
          labeled_sequences.csv
          graphpart_assignments.csv
        aho_train/
          uniprot_2022.tsv                     # generated

Output format is occurrence-level, not deduplicated by default:

    sequence    label    protein_id    start    end    length    fold    organism

`source` is intentionally not written as a column: downstream code can infer it from
`data/aho_train/uniprot_2022.tsv`.

Coordinates in DeepPeptide are UniProt-style: 1-based, inclusive.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
CANONICAL_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

# DeepPeptide column names. Keep aliases to make the script robust to small fork changes.
PROTEIN_ID_COLUMNS = ("protein_id", "AC", "accession", "Entry", "entry", "id")
SEQUENCE_COLUMNS = ("sequence", "Sequence", "protein_sequence", "seq")
PEPTIDE_COORD_COLUMNS = ("coordinates", "peptide_coordinates", "pep_coordinates")
PROPEPTIDE_COORD_COLUMNS = ("propeptide_coordinates", "propep_coordinates")
ORGANISM_COLUMNS = ("organism", "Organism")
FOLD_COLUMNS = ("cluster", "fold", "partition")
GRAPH_ID_COLUMNS = ("AC", "protein_id", "accession", "Entry", "entry", "id")


class FormatError(ValueError):
    """Raised when an input file does not look like the expected DeepPeptide format."""


@dataclass(frozen=True)
class Config:
    input_dir: Path
    output: Path
    labeled_file: str
    graphpart_file: str
    min_len: int
    max_len: int
    canonical_only: bool
    keep_outside_graphpart: bool
    write_unique: bool


def find_column(df: pd.DataFrame, candidates: Iterable[str], *, required: bool = True) -> str | None:
    """Find a column by exact name, then by lower-case match."""
    columns = list(df.columns)

    for name in candidates:
        if name in df.columns:
            return name

    by_lower = {str(col).lower(): str(col) for col in columns}
    for name in candidates:
        found = by_lower.get(name.lower())
        if found is not None:
            return found

    if required:
        raise FormatError(f"Missing column among {tuple(candidates)!r}. Available columns: {columns!r}")
    return None


def parse_coordinate_string(value: object) -> list[tuple[int, int]]:
    """
    Parse DeepPeptide coordinate strings without merging overlaps.

    Examples:
        "23-41,88-101"       -> [(23, 41), (88, 101)]
        "(23-41),(38-55)"    -> [(23, 41), (38, 55)]
        "" / NaN             -> []
    """
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return []

    intervals: list[tuple[int, int]] = []
    for raw_token in text.split(","):
        token = raw_token.strip().lstrip("(").rstrip(")").strip()
        if not token:
            continue

        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if match is None:
            raise FormatError(f"Cannot parse coordinate token {token!r} in {text!r}")

        start = int(match.group(1))
        end = int(match.group(2))
        if start <= 0 or end <= 0:
            raise FormatError(f"Coordinates must be positive 1-based integers: {token!r}")
        if end < start:
            raise FormatError(f"Coordinate end is smaller than start: {token!r}")
        intervals.append((start, end))

    # Deterministic order. Do not merge overlaps: Aho-only needs all annotated occurrences.
    return sorted(intervals, key=lambda x: (x[0], x[1]))


def read_labeled_sequences(path: Path) -> tuple[pd.DataFrame, str, str, str, str | None]:
    if not path.exists():
        raise FileNotFoundError(f"Missing labeled_sequences file: {path}")

    df = pd.read_csv(path)
    protein_id_col = find_column(df, PROTEIN_ID_COLUMNS)
    sequence_col = find_column(df, SEQUENCE_COLUMNS)
    peptide_coord_col = find_column(df, PEPTIDE_COORD_COLUMNS)
    propeptide_coord_col = find_column(df, PROPEPTIDE_COORD_COLUMNS)
    organism_col = find_column(df, ORGANISM_COLUMNS, required=False)

    df[protein_id_col] = df[protein_id_col].astype(str)
    df = df.set_index(protein_id_col, drop=False)
    df.index.name = "protein_id"

    return df, sequence_col, peptide_coord_col, propeptide_coord_col, organism_col


def read_graphpart(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing graphpart assignments file: {path}")

    gp = pd.read_csv(path)
    protein_id_col = find_column(gp, GRAPH_ID_COLUMNS)
    fold_col = find_column(gp, FOLD_COLUMNS)

    out = gp[[protein_id_col, fold_col]].copy()
    out.columns = ["protein_id", "fold"]
    out["protein_id"] = out["protein_id"].astype(str)
    return out.set_index("protein_id")


def is_canonical_sequence(seq: str) -> bool:
    return bool(CANONICAL_RE.fullmatch(seq))


def extract_segments(cfg: Config) -> pd.DataFrame:
    labeled_path = cfg.input_dir / cfg.labeled_file
    graphpart_path = cfg.input_dir / cfg.graphpart_file

    labeled, sequence_col, peptide_coord_col, propeptide_coord_col, organism_col = read_labeled_sequences(labeled_path)
    graphpart = read_graphpart(graphpart_path)

    graphpart_ids = set(graphpart.index.astype(str))
    rows: list[dict[str, object]] = []

    for protein_id, row in labeled.iterrows():
        protein_id = str(protein_id)

        if protein_id not in graphpart_ids and not cfg.keep_outside_graphpart:
            continue

        protein_seq = str(row[sequence_col]).strip().upper()
        if protein_seq == "" or protein_seq.lower() == "nan":
            continue

        fold = graphpart.loc[protein_id, "fold"] if protein_id in graphpart.index else pd.NA
        organism = row[organism_col] if organism_col is not None else pd.NA

        label_to_coord_col = {
            "pep": peptide_coord_col,
            "propep": propeptide_coord_col,
        }

        for label, coord_col in label_to_coord_col.items():
            for start, end in parse_coordinate_string(row.get(coord_col, "")):
                if end > len(protein_seq):
                    raise FormatError(
                        f"Coordinate outside protein bounds: {protein_id}:{start}-{end}, "
                        f"protein length={len(protein_seq)}"
                    )

                peptide_seq = protein_seq[start - 1 : end]
                length = len(peptide_seq)

                if length < cfg.min_len or length > cfg.max_len:
                    continue
                if cfg.canonical_only and not is_canonical_sequence(peptide_seq):
                    continue

                rows.append(
                    {
                        "sequence": peptide_seq,
                        "label": label,
                        "protein_id": protein_id,
                        "start": start,
                        "end": end,
                        "length": length,
                        "fold": fold,
                        "organism": organism,
                    }
                )

    columns = ["sequence", "label", "protein_id", "start", "end", "length", "fold", "organism"]
    return pd.DataFrame(rows, columns=columns)


def write_unique(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Optional sequence-level table for quick dictionary inspection.

    Keep the main occurrence-level file for fold-aware Aho-only experiments.
    """
    if df.empty:
        unique = pd.DataFrame(columns=["sequence", "label", "n_occurrences", "labels", "folds"])
    else:
        unique = (
            df.groupby("sequence", as_index=False)
            .agg(
                labels=("label", lambda x: ";".join(sorted(set(map(str, x))))),
                n_occurrences=("sequence", "size"),
                folds=("fold", lambda x: ";".join(sorted(set(map(str, x.dropna()))))),
            )
            .sort_values("sequence")
        )
        unique["label"] = unique["labels"].map(lambda s: s if ";" not in s else "ambiguous")
        unique = unique[["sequence", "label", "n_occurrences", "labels", "folds"]]

    unique_path = output_path.with_suffix(".unique.tsv")
    unique.to_csv(unique_path, sep="\t", index=False)
    return unique_path


def parse_args() -> Config:
    # This script is intended to live in <repo>/data/preprocessing/.
    script_path = Path(__file__).resolve()
    data_dir = script_path.parents[1]

    parser = argparse.ArgumentParser(description="Convert DeepPeptide UniProt 2022 data to data/aho_train format.")
    parser.add_argument("--input-dir", type=Path, default=data_dir / "uniprot_2022")
    parser.add_argument("--labeled-file", default="labeled_sequences.csv")
    parser.add_argument("--graphpart-file", default="graphpart_assignments.csv")
    parser.add_argument("--output", type=Path, default=data_dir / "aho_train" / "uniprot_2022.tsv")
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument(
        "--allow-noncanonical",
        action="store_true",
        help="Keep sequences containing non-canonical residues. Default: drop them.",
    )
    parser.add_argument(
        "--keep-outside-graphpart",
        action="store_true",
        help="Keep proteins absent from graphpart_assignments.csv. Default: drop them.",
    )
    parser.add_argument(
        "--write-unique",
        action="store_true",
        help="Also write data/aho_train/uniprot_2022.unique.tsv.",
    )
    args = parser.parse_args()

    if args.min_len <= 0:
        raise ValueError("--min-len must be positive")
    if args.max_len < args.min_len:
        raise ValueError("--max-len must be >= --min-len")

    return Config(
        input_dir=args.input_dir,
        output=args.output,
        labeled_file=args.labeled_file,
        graphpart_file=args.graphpart_file,
        min_len=args.min_len,
        max_len=args.max_len,
        canonical_only=not args.allow_noncanonical,
        keep_outside_graphpart=args.keep_outside_graphpart,
        write_unique=args.write_unique,
    )


def main() -> None:
    cfg = parse_args()

    df = extract_segments(cfg)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output, sep="\t", index=False)

    print(f"Wrote {len(df):,} segment occurrences to {cfg.output}")
    if not df.empty:
        print("Label counts:")
        for label, n in df["label"].value_counts().sort_index().items():
            print(f"  {label}: {n:,}")
        print(f"Unique sequences: {df['sequence'].nunique():,}")
        print("Fold counts:")
        for fold, n in df["fold"].value_counts(dropna=False).sort_index().items():
            print(f"  {fold}: {n:,}")

    if cfg.write_unique:
        unique_path = write_unique(df, cfg.output)
        print(f"Wrote unique dictionary table to {unique_path}")


if __name__ == "__main__":
    main()
