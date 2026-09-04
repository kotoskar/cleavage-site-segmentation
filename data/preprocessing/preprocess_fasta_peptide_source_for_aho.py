#!/usr/bin/env python3
"""
Normalize external peptide sequence sources into data/aho_train/*.tsv for Aho experiments.

Intended project layout:

    <repo>/
      data/
        raw/<source>/*.{fasta,fa,faa,txt,tsv,csv,xlsx}
        aho_train/<source>.tsv
        preprocessing/preprocess_fasta_peptide_source_for_aho.py

Output format:

    sequence    label    original_id    length    description

`source` is intentionally not written as a column; downstream code infers it from
output filename, e.g. data/aho_train/ampdb.tsv -> source='ampdb'.

Default filters are conservative for mature peptide dictionaries:
    - label=pep
    - 5 <= length <= 50
    - canonical amino acids only: ACDEFGHIKLMNPQRSTVWY
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

CANONICAL_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
SEQUENCE_COLUMNS = (
    "sequence", "Sequence", "seq", "Seq", "peptide", "Peptide",
    "peptide_sequence", "Peptide Sequence", "amino_acid_sequence", "AA Sequence",
    "protein_sequence", "Protein Sequence",
)
ID_COLUMNS = ("id", "ID", "accession", "Accession", "entry", "Entry", "name", "Name")
DESCRIPTION_COLUMNS = ("description", "Description", "info", "Info", "comment", "Comment")


class FormatError(ValueError):
    pass


def looks_like_html(path: Path, n: int = 512) -> bool:
    head = path.read_bytes()[:n].decode("utf-8", "ignore").lower()
    return "<html" in head or "<!doctype html" in head or "<body" in head or "bad gateway" in head


def clean_sequence(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip().upper()
    # Common formatting artifacts in peptide DB exports.
    s = s.replace(" ", "").replace("\t", "").replace("\r", "").replace("\n", "")
    s = s.replace("-", "").replace(".", "")
    # Remove terminal markers sometimes used in peptide tables.
    s = s.replace("NH2", "").replace("COOH", "")
    return s


def passes_filters(seq: str, min_len: int, max_len: int, canonical_only: bool) -> bool:
    if len(seq) < min_len or len(seq) > max_len:
        return False
    if canonical_only and not CANONICAL_RE.fullmatch(seq):
        return False
    return True


def parse_fasta(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    current_header: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_header, current_lines
        if current_header is None:
            return
        seq = clean_sequence("".join(current_lines))
        header = current_header.strip()
        original_id = header.split()[0] if header else ""
        records.append({"sequence": seq, "original_id": original_id, "description": header})
        current_header = None
        current_lines = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current_header = line[1:]
                current_lines = []
            else:
                current_lines.append(line.strip())
        flush()

    return records


def sniff_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="replace")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        return dialect.delimiter
    except Exception:
        return "\t" if "\t" in sample else ","


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols = list(df.columns)
    for c in candidates:
        if c in df.columns:
            return c
    by_lower = {str(c).lower().strip(): c for c in cols}
    for c in candidates:
        if c.lower().strip() in by_lower:
            return by_lower[c.lower().strip()]
    return None


def parse_table(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        sep = sniff_delimiter(path)
        df = pd.read_csv(path, sep=sep, engine="python")

    seq_col = find_column(df, SEQUENCE_COLUMNS)
    if seq_col is None:
        # Fallback: if the file is a one-column list of sequences.
        if len(df.columns) == 1:
            seq_col = df.columns[0]
        else:
            raise FormatError(f"Cannot find sequence column in {path}. Columns={list(df.columns)!r}")

    id_col = find_column(df, ID_COLUMNS)
    desc_col = find_column(df, DESCRIPTION_COLUMNS)

    out: list[dict[str, object]] = []
    for idx, row in df.iterrows():
        seq = clean_sequence(row[seq_col])
        original_id = str(row[id_col]) if id_col is not None and not pd.isna(row[id_col]) else str(idx)
        description = str(row[desc_col]) if desc_col is not None and not pd.isna(row[desc_col]) else ""
        out.append({"sequence": seq, "original_id": original_id, "description": description})
    return out


def read_source_file(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if looks_like_html(path):
        raise FormatError(
            f"{path} looks like HTML/error page, not data. Open the file/head to check the download URL."
        )

    suffix = path.suffix.lower()
    text_head = path.read_text(encoding="utf-8", errors="replace")[:2048]
    if suffix in {".fasta", ".fa", ".faa", ".fas"} or text_head.lstrip().startswith(">"):
        return parse_fasta(path)
    return parse_table(path)


def normalize_files(
    files: list[Path],
    label: str,
    min_len: int,
    max_len: int,
    canonical_only: bool,
    deduplicate: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in files:
        records = read_source_file(path)
        for rec in records:
            seq = clean_sequence(rec.get("sequence", ""))
            if not passes_filters(seq, min_len, max_len, canonical_only):
                continue
            rows.append(
                {
                    "sequence": seq,
                    "label": label,
                    "original_id": rec.get("original_id", ""),
                    "length": len(seq),
                    "description": rec.get("description", ""),
                }
            )

    df = pd.DataFrame(rows, columns=["sequence", "label", "original_id", "length", "description"])
    if deduplicate and not df.empty:
        df = (
            df.sort_values(["sequence", "original_id"])
            .drop_duplicates(subset=["sequence", "label"], keep="first")
            .reset_index(drop=True)
        )
    return df


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    data_dir = script_path.parents[1] if script_path.parent.name == "preprocessing" else Path("data").resolve()

    p = argparse.ArgumentParser(description="Convert external peptide FASTA/table files to Aho TSV format.")
    p.add_argument("--files", nargs="+", type=Path, required=True, help="Input FASTA/table files.")
    p.add_argument("--output", type=Path, required=True, help="Output TSV, e.g. data/aho_train/ampdb.tsv")
    p.add_argument("--label", choices=["pep", "propep"], default="pep")
    p.add_argument("--min-len", type=int, default=5)
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--allow-noncanonical", action="store_true")
    p.add_argument("--deduplicate", action="store_true", help="Deduplicate by (sequence,label). Recommended.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_len <= 0:
        raise ValueError("--min-len must be positive")
    if args.max_len < args.min_len:
        raise ValueError("--max-len must be >= --min-len")

    df = normalize_files(
        files=args.files,
        label=args.label,
        min_len=args.min_len,
        max_len=args.max_len,
        canonical_only=not args.allow_noncanonical,
        deduplicate=args.deduplicate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep="\t", index=False)

    print(f"Wrote {len(df):,} rows to {args.output}")
    if not df.empty:
        print(f"Unique sequences: {df['sequence'].nunique():,}")
        print("Length range:", int(df["length"].min()), int(df["length"].max()))


if __name__ == "__main__":
    main()
