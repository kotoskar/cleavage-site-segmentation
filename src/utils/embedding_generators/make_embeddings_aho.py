#!/usr/bin/env python3
"""
Generate per-residue Aho-Corasick feature tensors for DeepPeptide-style proteins.

Intended location:
    src/utils/embedding_generators/make_embeddings_aho.py

Run from repository root. It reads normalized Aho dictionary files from data/aho_train
and writes one .pt tensor per protein with shape [L, D]. These tensors can be
concatenated with ESM2 embeddings by src/utils/merge_embeddings_generic.py.

Leakage policy:
  * Rows with a non-null `fold` column are treated as fold-aware / internal.
    Only rows whose fold is in --dict_folds are loaded.
  * Rows without fold are treated as external and are loaded for all splits.
  * Optional --drop_eval_exact_folds removes external exact gold sequences from
    selected evaluation folds for strict leakage checks.
  * Optional --exclude_self_folded suppresses a protein's own folded rows while
    generating its features, reducing train-time self-lookup leakage.

Output:
  out_dir/<filename_stem>.pt        torch.float32 tensor [L, D]
  out_dir/feature_names.json       ordered feature names
  out_dir/config.json              run config
  out_dir/summary.json             source/filter summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import numpy as np
import pandas as pd
import torch

try:
    import ahocorasick  # pip install pyahocorasick
    HAS_PYAHOCORASICK = True
except Exception:  # pragma: no cover - optional backend
    ahocorasick = None
    HAS_PYAHOCORASICK = False

CANONICAL_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
LABEL_ALIASES = {
    "pep": "pep",
    "peptide": "pep",
    "peptides": "pep",
    "PEPTIDE": "pep",
    "propep": "propep",
    "propeptide": "propep",
    "propeptides": "propep",
    "PROPEP": "propep",
}
PROTEIN_ID_COLUMNS = ("protein_id", "AC", "accession", "Entry", "entry", "id")
SEQUENCE_COLUMNS = ("sequence", "Sequence", "protein_sequence", "seq")
PEPTIDE_COORD_COLUMNS = ("coordinates", "peptide_coordinates", "pep_coordinates")
PROPEPTIDE_COORD_COLUMNS = ("propeptide_coordinates", "propep_coordinates")
GRAPH_ID_COLUMNS = ("AC", "protein_id", "accession", "Entry", "entry", "id")
FOLD_COLUMNS = ("cluster", "fold", "partition")


class FormatError(ValueError):
    pass


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str
    fold: int | None
    filename_stem: str


@dataclass(frozen=True)
class DictRow:
    sequence: str
    label: Literal["pep", "propep"]
    source: str
    source_type: Literal["folded", "external"]
    fold: int | None
    protein_id: str | None


@dataclass
class DictEntry:
    sequence: str
    label: Literal["pep", "propep"]
    rows: list[DictRow]


def find_column(df: pd.DataFrame, candidates: Iterable[str], *, required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        found = lower.get(c.lower())
        if found is not None:
            return found
    if required:
        raise FormatError(f"Missing column among {tuple(candidates)!r}. Available: {list(df.columns)!r}")
    return None


def normalize_label(value: Any) -> Literal["pep", "propep"] | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return LABEL_ALIASES.get(text, LABEL_ALIASES.get(text.lower()))


def clean_sequence(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    seq = str(value).strip().upper()
    seq = re.sub(r"\s+", "", seq)
    seq = seq.replace("-", "")
    return seq


def is_canonical(seq: str) -> bool:
    return bool(CANONICAL_RE.fullmatch(seq))


def parse_int_list(value: str | None) -> set[int]:
    if value is None or str(value).strip() == "":
        return set()
    return {int(x.strip()) for x in str(value).split(",") if x.strip()}


def parse_coordinate_string(value: Any) -> list[tuple[int, int]]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    out = [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*-\s*(\d+)", text)]
    if not out:
        # Fallback for strings like "[(23, 41), (88, 101)]".
        out = [(int(a), int(b)) for a, b in re.findall(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", text)]
    for a, b in out:
        if a <= 0 or b < a:
            raise FormatError(f"Bad coordinate interval {a}-{b} in {text!r}")
    return sorted(out, key=lambda x: (x[0], x[1]))


def read_table_auto(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    # Try TSV first, then CSV.
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.read_csv(path)


def md5_sequence(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()


def read_proteins(
    labeled_file: Path,
    partitioning_file: Path,
    filename_mode: str = "md5_sequence",
    filename_column: str | None = None,
) -> tuple[dict[str, ProteinRecord], pd.DataFrame, str, str, str]:
    labeled = pd.read_csv(labeled_file)
    pid_col = find_column(labeled, PROTEIN_ID_COLUMNS)
    seq_col = find_column(labeled, SEQUENCE_COLUMNS)
    pep_col = find_column(labeled, PEPTIDE_COORD_COLUMNS)
    pro_col = find_column(labeled, PROPEPTIDE_COORD_COLUMNS)

    if filename_mode == "column":
        if not filename_column:
            raise ValueError("--filename_mode column requires --filename_column")
        filename_column = find_column(labeled, (filename_column,), required=True)

    labeled[pid_col] = labeled[pid_col].astype(str)

    graph = pd.read_csv(partitioning_file)
    g_pid = find_column(graph, GRAPH_ID_COLUMNS)
    g_fold = find_column(graph, FOLD_COLUMNS)
    graph = graph[[g_pid, g_fold]].copy()
    graph.columns = ["protein_id", "fold"]
    graph["protein_id"] = graph["protein_id"].astype(str)
    graph["fold"] = graph["fold"].astype(int)
    fold_by_pid = dict(zip(graph["protein_id"], graph["fold"]))

    proteins: dict[str, ProteinRecord] = {}
    for _, row in labeled.iterrows():
        pid = str(row[pid_col])
        seq = clean_sequence(row[seq_col])
        if not seq:
            continue
        if filename_mode == "protein_id":
            filename_stem = pid
        elif filename_mode == "md5_sequence":
            filename_stem = md5_sequence(seq)
        elif filename_mode == "column":
            raw_name = row[filename_column]  # type: ignore[index]
            if raw_name is None or pd.isna(raw_name) or str(raw_name).strip() == "":
                raise FormatError(f"Missing filename value in column {filename_column!r} for protein {pid}")
            filename_stem = str(raw_name).strip()
            if filename_stem.endswith(".pt"):
                filename_stem = filename_stem[:-3]
        else:
            raise ValueError(f"Unknown filename_mode: {filename_mode}")

        proteins[pid] = ProteinRecord(pid, seq, int(fold_by_pid[pid]) if pid in fold_by_pid else None, filename_stem)

    return proteins, labeled.set_index(pid_col, drop=False), seq_col, pep_col, pro_col


def collect_gold_sequences_for_folds(
    labeled: pd.DataFrame,
    seq_col: str,
    pep_col: str,
    pro_col: str,
    folds_by_pid: dict[str, int | None],
    eval_folds: set[int],
    min_len: int,
    max_len: int,
    canonical_only: bool,
) -> set[tuple[str, str]]:
    gold: set[tuple[str, str]] = set()
    if not eval_folds:
        return gold
    for pid, row in labeled.iterrows():
        fold = folds_by_pid.get(str(pid))
        if fold not in eval_folds:
            continue
        prot = clean_sequence(row[seq_col])
        for label, col in (("pep", pep_col), ("propep", pro_col)):
            for start, end in parse_coordinate_string(row.get(col, "")):
                if end > len(prot):
                    continue
                seq = prot[start - 1 : end]
                if not (min_len <= len(seq) <= max_len):
                    continue
                if canonical_only and not is_canonical(seq):
                    continue
                gold.add((seq, label))
    return gold


def discover_dictionary_files(aho_dir: Path, sources: list[str] | None) -> list[Path]:
    files = []
    for ext in ("*.tsv", "*.csv", "*.xlsx", "*.xls"):
        files.extend(aho_dir.glob(ext))
    files = sorted(set(files))
    if sources:
        wanted = set(sources)
        files = [p for p in files if p.stem in wanted]
        missing = wanted - {p.stem for p in files}
        if missing:
            raise FileNotFoundError(f"Missing requested source files in {aho_dir}: {sorted(missing)}")
    return files


def load_dictionary(
    aho_dir: Path,
    sources: list[str] | None,
    dict_folds: set[int],
    eval_gold_to_drop: set[tuple[str, str]],
    min_len: int,
    max_len: int,
    canonical_only: bool,
) -> tuple[dict[tuple[str, str], DictEntry], dict[str, list[DictEntry]], dict[str, Any]]:
    files = discover_dictionary_files(aho_dir, sources)
    entries: dict[tuple[str, str], DictEntry] = {}
    summary: dict[str, Any] = {"files": {}, "rows_loaded": 0, "rows_kept": 0, "rows_dropped_eval_exact": 0}

    for path in files:
        source = path.stem
        df = read_table_auto(path)
        seq_col = find_column(df, ("sequence", "Sequence", "seq", "peptide", "Peptide"))
        label_col = find_column(df, ("label", "type", "class"), required=False)
        fold_col = find_column(df, FOLD_COLUMNS, required=False)
        pid_col = find_column(df, PROTEIN_ID_COLUMNS, required=False)

        loaded = len(df)
        kept = 0
        dropped_eval = 0
        for _, row in df.iterrows():
            seq = clean_sequence(row[seq_col])
            label = normalize_label(row[label_col]) if label_col else "pep"
            if label is None:
                continue
            if not (min_len <= len(seq) <= max_len):
                continue
            if canonical_only and not is_canonical(seq):
                continue

            fold: int | None = None
            source_type: Literal["folded", "external"] = "external"
            if fold_col is not None and not pd.isna(row.get(fold_col)):
                try:
                    fold = int(row[fold_col])
                    source_type = "folded"
                except Exception:
                    fold = None
                    source_type = "external"

            if source_type == "folded" and dict_folds and fold not in dict_folds:
                continue

            # Strict mode only drops external exact eval gold. Folded rows are handled by fold filtering.
            if source_type == "external" and (seq, label) in eval_gold_to_drop:
                dropped_eval += 1
                continue

            protein_id = None
            if pid_col is not None and not pd.isna(row.get(pid_col)):
                protein_id = str(row[pid_col])

            drow = DictRow(seq, label, source, source_type, fold, protein_id)
            key = (seq, label)
            if key not in entries:
                entries[key] = DictEntry(seq, label, [])
            entries[key].rows.append(drow)
            kept += 1

        summary["files"][source] = {"path": str(path), "rows_loaded": loaded, "rows_kept": kept, "rows_dropped_eval_exact": dropped_eval}
        summary["rows_loaded"] += loaded
        summary["rows_kept"] += kept
        summary["rows_dropped_eval_exact"] += dropped_eval

    by_seq: dict[str, list[DictEntry]] = defaultdict(list)
    for entry in entries.values():
        by_seq[entry.sequence].append(entry)

    summary["n_entries"] = len(entries)
    summary["n_patterns"] = len(by_seq)
    summary["sources"] = sorted({r.source for e in entries.values() for r in e.rows})
    return entries, dict(by_seq), summary


class AhoMatcher:
    def __init__(self, patterns: Iterable[str]) -> None:
        self.patterns = sorted(set(patterns), key=lambda s: (len(s), s))
        self.backend = "pyahocorasick" if HAS_PYAHOCORASICK else "pure_python"
        self.automaton = None
        if HAS_PYAHOCORASICK:
            automaton = ahocorasick.Automaton()
            for p in self.patterns:
                automaton.add_word(p, p)
            automaton.make_automaton()
            self.automaton = automaton

    def iter_hits(self, sequence: str) -> Iterator[tuple[int, int, str]]:
        """Yield (start, end, pattern), 1-based inclusive."""
        if self.automaton is not None:
            for end0, pat in self.automaton.iter(sequence):
                start0 = end0 - len(pat) + 1
                yield start0 + 1, end0 + 1, pat
        else:
            for pat in self.patterns:
                start = sequence.find(pat)
                while start != -1:
                    yield start + 1, start + len(pat), pat
                    start = sequence.find(pat, start + 1)


def build_feature_names(sources: list[str]) -> list[str]:
    labels = ["pep", "propep"]
    agg = [
        "inside",
        "start",
        "end",
        "count_log",
        "max_len_norm",
        "rel_from_start",
        "rel_to_end",
        "source_count_log",
        "multi_source",
        "start_window3",
        "end_window3",
        "start_decay",
        "end_decay",
    ]
    per_source = ["inside", "start", "end", "count_log", "max_len_norm"]
    names = []
    for label in labels:
        for feat in agg:
            names.append(f"{label}.{feat}")
    for src in sources:
        for label in labels:
            for feat in per_source:
                names.append(f"{src}.{label}.{feat}")
    return names


def add_hit_features(
    x: np.ndarray,
    name_to_idx: dict[str, int],
    start: int,
    end: int,
    label: Literal["pep", "propep"],
    rows: list[DictRow],
    max_len: int,
    window_radius: int,
    decay_radius: int,
    decay_sigma: float,
) -> None:
    L = x.shape[0]
    s0 = max(0, start - 1)
    e0 = min(L - 1, end - 1)
    if s0 > e0:
        return
    hit_len = e0 - s0 + 1
    len_norm = min(1.0, hit_len / float(max_len))

    source_counts = Counter(r.source for r in rows)
    n_occ = sum(source_counts.values())
    n_src = len(source_counts)

    # Aggregate channels.
    pref = label
    x[s0 : e0 + 1, name_to_idx[f"{pref}.inside"]] = 1.0
    x[s0, name_to_idx[f"{pref}.start"]] = 1.0
    x[e0, name_to_idx[f"{pref}.end"]] = 1.0
    x[s0 : e0 + 1, name_to_idx[f"{pref}.count_log"]] += float(n_occ)
    x[s0 : e0 + 1, name_to_idx[f"{pref}.source_count_log"]] += float(n_src)
    if n_src >= 2:
        x[s0 : e0 + 1, name_to_idx[f"{pref}.multi_source"]] = 1.0

    max_idx = name_to_idx[f"{pref}.max_len_norm"]
    rel_s_idx = name_to_idx[f"{pref}.rel_from_start"]
    rel_e_idx = name_to_idx[f"{pref}.rel_to_end"]
    current = x[s0 : e0 + 1, max_idx]
    mask_longer = len_norm > current
    if np.any(mask_longer):
        rel = np.linspace(0.0, 1.0, hit_len, dtype=np.float32) if hit_len > 1 else np.array([0.0], dtype=np.float32)
        positions = np.arange(s0, e0 + 1)[mask_longer]
        x[positions, max_idx] = len_norm
        x[positions, rel_s_idx] = rel[mask_longer]
        x[positions, rel_e_idx] = 1.0 - rel[mask_longer]

    # Boundary windows and decay around start/end.
    sw0 = max(0, s0 - window_radius)
    sw1 = min(L - 1, s0 + window_radius)
    ew0 = max(0, e0 - window_radius)
    ew1 = min(L - 1, e0 + window_radius)
    x[sw0 : sw1 + 1, name_to_idx[f"{pref}.start_window3"]] = 1.0
    x[ew0 : ew1 + 1, name_to_idx[f"{pref}.end_window3"]] = 1.0

    ds0 = max(0, s0 - decay_radius)
    ds1 = min(L - 1, s0 + decay_radius)
    de0 = max(0, e0 - decay_radius)
    de1 = min(L - 1, e0 + decay_radius)
    pos = np.arange(ds0, ds1 + 1)
    vals = np.exp(-np.abs(pos - s0) / max(1e-6, decay_sigma)).astype(np.float32)
    idx = name_to_idx[f"{pref}.start_decay"]
    x[pos, idx] = np.maximum(x[pos, idx], vals)
    pos = np.arange(de0, de1 + 1)
    vals = np.exp(-np.abs(pos - e0) / max(1e-6, decay_sigma)).astype(np.float32)
    idx = name_to_idx[f"{pref}.end_decay"]
    x[pos, idx] = np.maximum(x[pos, idx], vals)

    # Source-specific channels.
    for src, count in source_counts.items():
        base = f"{src}.{label}"
        if f"{base}.inside" not in name_to_idx:
            continue
        x[s0 : e0 + 1, name_to_idx[f"{base}.inside"]] = 1.0
        x[s0, name_to_idx[f"{base}.start"]] = 1.0
        x[e0, name_to_idx[f"{base}.end"]] = 1.0
        x[s0 : e0 + 1, name_to_idx[f"{base}.count_log"]] += float(count)
        mi = name_to_idx[f"{base}.max_len_norm"]
        x[s0 : e0 + 1, mi] = np.maximum(x[s0 : e0 + 1, mi], len_norm)


def make_features_for_protein(
    protein: ProteinRecord,
    matcher: AhoMatcher,
    entries_by_sequence: dict[str, list[DictEntry]],
    feature_names: list[str],
    exclude_self_folded: bool,
    max_len: int,
    window_radius: int,
    decay_radius: int,
    decay_sigma: float,
) -> np.ndarray:
    L = len(protein.sequence)
    x = np.zeros((L, len(feature_names)), dtype=np.float32)
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    for start, end, pattern in matcher.iter_hits(protein.sequence):
        for entry in entries_by_sequence.get(pattern, []):
            rows = []
            for r in entry.rows:
                if exclude_self_folded and r.source_type == "folded" and r.protein_id == protein.protein_id:
                    continue
                rows.append(r)
            if not rows:
                continue
            add_hit_features(
                x,
                name_to_idx,
                start,
                end,
                entry.label,
                rows,
                max_len=max_len,
                window_radius=window_radius,
                decay_radius=decay_radius,
                decay_sigma=decay_sigma,
            )

    # Convert raw count channels to log1p-scaled counts.
    count_indices = [i for i, n in enumerate(feature_names) if n.endswith("count_log") or n.endswith("source_count_log")]
    if count_indices:
        x[:, count_indices] = np.log1p(x[:, count_indices])
    return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create per-residue Aho feature tensors for CRF training.")
    p.add_argument("--data_file", type=Path, default=Path("data/uniprot_2022/labeled_sequences.csv"))
    p.add_argument("--partitioning_file", type=Path, default=Path("data/uniprot_2022/graphpart_assignments.csv"))
    p.add_argument("--aho_dir", type=Path, default=Path("data/aho_train"))
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--sources", type=str, default=None, help="Comma-separated source stems, e.g. uniprot_2022,dramp_general,dbamp_3")
    p.add_argument("--dict_folds", type=str, default="0,1,2", help="Fold-aware dictionary rows to include.")
    p.add_argument("--protein_folds", type=str, default=None, help="Optional comma-separated folds to emit. Default: all proteins.")
    p.add_argument("--drop_eval_exact_folds", type=str, default=None, help="Strict mode: drop external exact gold sequences from these folds, e.g. 3,4")
    p.add_argument("--exclude_self_folded", action="store_true", help="Do not use a protein's own folded rows when generating its features.")
    p.add_argument("--min_len", type=int, default=5)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--allow_noncanonical", action="store_true")
    p.add_argument("--window_radius", type=int, default=3)
    p.add_argument("--decay_radius", type=int, default=10)
    p.add_argument("--decay_sigma", type=float, default=3.0)
    p.add_argument(
        "--filename_mode",
        choices=["md5_sequence", "protein_id", "column"],
        default="md5_sequence",
        help=(
            "How to name output .pt files. Use md5_sequence to match DeepPeptide "
            "precomputed embeddings named by md5(sequence); protein_id for AC names; "
            "column to use --filename_column."
        ),
    )
    p.add_argument(
        "--filename_column",
        default=None,
        help="Column to use as output filename stem when --filename_mode column.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
    dict_folds = parse_int_list(args.dict_folds)
    protein_folds = parse_int_list(args.protein_folds)
    eval_folds = parse_int_list(args.drop_eval_exact_folds)
    canonical_only = not args.allow_noncanonical

    proteins, labeled, seq_col, pep_col, pro_col = read_proteins(
        args.data_file,
        args.partitioning_file,
        filename_mode=args.filename_mode,
        filename_column=args.filename_column,
    )
    folds_by_pid = {pid: rec.fold for pid, rec in proteins.items()}
    eval_gold = collect_gold_sequences_for_folds(
        labeled,
        seq_col,
        pep_col,
        pro_col,
        folds_by_pid,
        eval_folds,
        args.min_len,
        args.max_len,
        canonical_only,
    )

    entries, entries_by_seq, summary = load_dictionary(
        args.aho_dir,
        sources,
        dict_folds,
        eval_gold,
        args.min_len,
        args.max_len,
        canonical_only,
    )
    source_names = summary["sources"]
    feature_names = build_feature_names(source_names)

    matcher = AhoMatcher(entries_by_seq.keys())
    print(f"[Aho backend] {matcher.backend}")
    print(f"Loaded dictionary: {summary['n_entries']} entries, {summary['n_patterns']} unique patterns, sources={tuple(source_names)}")
    print(f"Aho feature dim: {len(feature_names)}")

    emitted = 0
    skipped = 0
    for protein in proteins.values():
        if protein_folds and protein.fold not in protein_folds:
            continue
        out_path = args.out_dir / f"{protein.filename_stem}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        feats = make_features_for_protein(
            protein,
            matcher,
            entries_by_seq,
            feature_names,
            exclude_self_folded=args.exclude_self_folded,
            max_len=args.max_len,
            window_radius=args.window_radius,
            decay_radius=args.decay_radius,
            decay_sigma=args.decay_sigma,
        )
        torch.save(torch.from_numpy(feats), out_path)
        emitted += 1

    config = vars(args).copy()
    for key in ("data_file", "partitioning_file", "aho_dir", "out_dir"):
        config[key] = str(config[key])
    config["aho_backend"] = matcher.backend
    config["feature_dim"] = len(feature_names)
    config["feature_names"] = feature_names

    filename_map = {pid: rec.filename_stem for pid, rec in proteins.items()}

    with open(args.out_dir / "protein_id_to_filename_stem.json", "w") as f:
        json.dump(filename_map, f, indent=2)
    with open(args.out_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(args.out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary | {"emitted": emitted, "skipped": skipped, "feature_dim": len(feature_names)}, f, indent=2)

    print(f"Done. Emitted={emitted}, skipped={skipped}, out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
