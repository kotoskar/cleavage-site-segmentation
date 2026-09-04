#!/usr/bin/env python3
"""
Convert DRAMP peptide sequence downloads to the normalized data/aho_train format.

Expected project layout:

    <repo>/
      data/
        preprocessing/
          preprocess_dramp_for_aho.py   # this script
        raw/
          dramp/
            general_amps.fasta
            natural_amps.fasta
            # optionally .txt/.tsv/.csv/.xlsx exports
        aho_train/
          dramp_general.tsv
          dramp_natural.tsv

Output format is occurrence-level by default:

    sequence    label    original_id    length    description

`source` is intentionally not written as a column: train_loop_aho.py infers it from
file names, e.g. data/aho_train/dramp_natural.tsv -> source == "dramp_natural".

All DRAMP rows are written as label == "pep". DRAMP is a peptide/AMP source, not a
propeptide source.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

CANONICAL_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
SEQUENCE_COLUMN_CANDIDATES = (
    "sequence",
    "Sequence",
    "seq",
    "Seq",
    "peptide_sequence",
    "Peptide_sequence",
    "Peptide Sequence",
    "Peptide_Sequence",
    "Amino acid sequence",
    "Amino Acid Sequence",
    "AA sequence",
    "AA_sequence",
    "AASequence",
)
ID_COLUMN_CANDIDATES = (
    "DRAMP_ID",
    "DRAMP id",
    "DRAMP ID",
    "ID",
    "id",
    "Peptide_ID",
    "Peptide ID",
    "Name",
    "name",
    "Entry",
    "entry",
)
DESCRIPTION_COLUMN_CANDIDATES = (
    "description",
    "Description",
    "name",
    "Name",
    "Peptide name",
    "Peptide Name",
    "Source",
    "source",
)


class FormatError(ValueError):
    """Raised when an input file does not look like a usable DRAMP export."""


@dataclass(frozen=True)
class Config:
    input_dir: Path
    output_dir: Path
    files: tuple[Path, ...]
    min_len: int
    max_len: int
    canonical_only: bool
    deduplicate: bool
    recursive: bool


@dataclass(frozen=True)
class PeptideRow:
    sequence: str
    label: str
    original_id: str | None
    length: int
    description: str | None


def is_probably_html(path: Path) -> bool:
    """Detect failed downloads that saved an HTML error/login page as .fasta/.txt."""
    try:
        text = path.read_text(errors="ignore")[:4096].lower()
    except UnicodeDecodeError:
        return False
    stripped = text.lstrip()
    return (
        stripped.startswith("<!doctype html")
        or stripped.startswith("<html")
        or "<html" in text
        or "</html>" in text
    )


def clean_sequence(value: object) -> str:
    """
    Normalize a peptide sequence.

    DRAMP/peptide databases may include spaces, hyphens, terminal markers, or FASTA
    line breaks. This function removes whitespace/hyphens and uppercases the result.
    Non-canonical residues are filtered later unless --allow-noncanonical is used.
    """
    if value is None or pd.isna(value):
        return ""
    seq = str(value).strip().upper()
    seq = re.sub(r"\s+", "", seq)
    seq = seq.replace("-", "")
    seq = seq.strip("*.")
    return seq


def is_canonical_sequence(seq: str) -> bool:
    return bool(CANONICAL_RE.fullmatch(seq))


def find_column(df: pd.DataFrame, candidates: Iterable[str], *, required: bool = True) -> str | None:
    columns = list(df.columns)
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    by_lower = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        found = by_lower.get(candidate.strip().lower())
        if found is not None:
            return found

    if required:
        raise FormatError(f"Missing column among {tuple(candidates)!r}. Available columns: {columns!r}")
    return None


def normalize_output_stem(input_path: Path) -> str:
    """Map DRAMP file names to stable source names used by train_loop_aho.py."""
    stem = input_path.stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")

    # Common DRAMP download names.
    if stem in {"general_amps", "general_amp", "dramp_general_amps"}:
        return "dramp_general"
    if stem in {"natural_amps", "natural_amp", "dramp_natural_amps"}:
        return "dramp_natural"
    if stem in {"clinical_amps", "clinical_amp"}:
        return "dramp_clinical"
    if stem in {"patent_amps", "patent_amp"}:
        return "dramp_patent"
    if stem in {"specific_amps", "specific_amp"}:
        return "dramp_specific"

    return stem if stem.startswith("dramp_") else f"dramp_{stem}"


def iter_fasta(path: Path) -> Iterator[PeptideRow]:
    if is_probably_html(path):
        raise FormatError(
            f"{path} looks like HTML, not a DRAMP FASTA/TXT export. "
            "The download probably failed; re-download the file and check `head` first."
        )

    current_header: str | None = None
    seq_parts: list[str] = []
    saw_header = False

    def emit(header: str | None, parts: list[str]) -> PeptideRow | None:
        if header is None:
            return None
        seq = clean_sequence("".join(parts))
        header_text = header.strip()
        original_id = header_text.split()[0] if header_text else None
        description = header_text if header_text else None
        return PeptideRow(sequence=seq, label="pep", original_id=original_id, length=len(seq), description=description)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                saw_header = True
                row = emit(current_header, seq_parts)
                if row is not None:
                    yield row
                current_header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())

    row = emit(current_header, seq_parts)
    if row is not None:
        yield row

    if not saw_header:
        raise FormatError(f"{path} does not look like FASTA: no header lines starting with '>' were found.")


def read_table(path: Path) -> pd.DataFrame:
    if is_probably_html(path):
        raise FormatError(
            f"{path} looks like HTML, not a DRAMP table. "
            "The download probably failed; re-download the file and check `head` first."
        )

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)

    # .txt can be TSV, CSV, semicolon-separated, or sometimes whitespace-ish.
    sample = path.read_text(errors="replace")[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        return pd.read_csv(path, sep=dialect.delimiter)
    except Exception:
        # Last attempt: tab-separated is common for database exports.
        return pd.read_csv(path, sep="\t")


def iter_table(path: Path) -> Iterator[PeptideRow]:
    df = read_table(path)
    sequence_col = find_column(df, SEQUENCE_COLUMN_CANDIDATES)
    id_col = find_column(df, ID_COLUMN_CANDIDATES, required=False)
    description_col = find_column(df, DESCRIPTION_COLUMN_CANDIDATES, required=False)

    for _, row in df.iterrows():
        seq = clean_sequence(row[sequence_col])
        original_id = None if id_col is None or pd.isna(row[id_col]) else str(row[id_col]).strip()
        description = None if description_col is None or pd.isna(row[description_col]) else str(row[description_col]).strip()
        yield PeptideRow(sequence=seq, label="pep", original_id=original_id, length=len(seq), description=description)


def iter_input_rows(path: Path) -> Iterator[PeptideRow]:
    suffix = path.suffix.lower()
    if suffix in {".fa", ".faa", ".fasta", ".fas"}:
        yield from iter_fasta(path)
        return

    # Some DRAMP files may be named .txt but still contain FASTA.
    if suffix == ".txt":
        if is_probably_html(path):
            raise FormatError(
                f"{path} looks like HTML, not DRAMP data. The download probably failed."
            )
        first_nonempty = ""
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    first_nonempty = line.strip()
                    break
        if first_nonempty.startswith(">"):
            yield from iter_fasta(path)
            return

    yield from iter_table(path)


def filter_rows(rows: Iterable[PeptideRow], cfg: Config) -> tuple[pd.DataFrame, dict[str, int]]:
    kept: list[dict[str, object]] = []
    stats = {
        "loaded": 0,
        "empty": 0,
        "too_short": 0,
        "too_long": 0,
        "noncanonical": 0,
        "duplicate_removed": 0,
        "kept": 0,
    }
    seen: set[str] = set()

    for row in rows:
        stats["loaded"] += 1
        seq = clean_sequence(row.sequence)
        if not seq:
            stats["empty"] += 1
            continue
        if len(seq) < cfg.min_len:
            stats["too_short"] += 1
            continue
        if len(seq) > cfg.max_len:
            stats["too_long"] += 1
            continue
        if cfg.canonical_only and not is_canonical_sequence(seq):
            stats["noncanonical"] += 1
            continue
        if cfg.deduplicate and seq in seen:
            stats["duplicate_removed"] += 1
            continue
        seen.add(seq)
        kept.append(
            {
                "sequence": seq,
                "label": "pep",
                "original_id": row.original_id,
                "length": len(seq),
                "description": row.description,
            }
        )

    stats["kept"] = len(kept)
    return pd.DataFrame(kept, columns=["sequence", "label", "original_id", "length", "description"]), stats


def discover_files(input_dir: Path, recursive: bool) -> tuple[Path, ...]:
    patterns = ("*.fasta", "*.fa", "*.faa", "*.fas", "*.txt", "*.tsv", "*.csv", "*.xlsx", "*.xls")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
    files = sorted({p.resolve() for p in files if p.is_file() and not p.name.startswith(".")})
    if not files:
        raise FileNotFoundError(f"No DRAMP input files found in {input_dir}")
    return tuple(files)


def parse_args() -> Config:
    # This script is intended to live in <repo>/data/preprocessing/.
    script_path = Path(__file__).resolve()
    data_dir = script_path.parents[1]

    parser = argparse.ArgumentParser(description="Convert DRAMP sequence downloads to data/aho_train/*.tsv.")
    parser.add_argument("--input-dir", type=Path, default=data_dir / "raw" / "dramp")
    parser.add_argument("--output-dir", type=Path, default=data_dir / "aho_train")
    parser.add_argument(
        "--files",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit input files. If omitted, scans --input-dir for FASTA/TXT/CSV/TSV/XLSX files.",
    )
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument(
        "--allow-noncanonical",
        action="store_true",
        help="Keep sequences containing non-canonical residues. Default: drop them.",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Remove duplicate sequences within each output file. Default: keep occurrence-level rows.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan --input-dir when --files is omitted.",
    )
    args = parser.parse_args()

    if args.min_len <= 0:
        raise ValueError("--min-len must be positive")
    if args.max_len < args.min_len:
        raise ValueError("--max-len must be >= --min-len")

    files = tuple(p.resolve() for p in args.files) if args.files else discover_files(args.input_dir, args.recursive)
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)

    return Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        files=files,
        min_len=args.min_len,
        max_len=args.max_len,
        canonical_only=not args.allow_noncanonical,
        deduplicate=args.deduplicate,
        recursive=args.recursive,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(cfg.files)} DRAMP input file(s).")
    for path in cfg.files:
        source_stem = normalize_output_stem(path)
        output_path = cfg.output_dir / f"{source_stem}.tsv"

        rows = list(iter_input_rows(path))
        df, stats = filter_rows(rows, cfg)
        df.to_csv(output_path, sep="\t", index=False)

        print(f"\n{path}")
        print(f"  -> {output_path}")
        for key in ("loaded", "kept", "too_short", "too_long", "noncanonical", "empty", "duplicate_removed"):
            print(f"  {key}: {stats[key]:,}")
        if not df.empty:
            print(f"  unique sequences: {df['sequence'].nunique():,}")
            print(f"  min/max length: {df['length'].min()} / {df['length'].max()}")


if __name__ == "__main__":
    main()
