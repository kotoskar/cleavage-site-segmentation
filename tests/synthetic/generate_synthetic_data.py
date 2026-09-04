#!/usr/bin/env python3
"""Generate a tiny, fully synthetic dataset for smoke-testing the training/eval
pipeline end to end, without needing any real UniProt data or a real protein
language model.

This produces, under --out_dir (default: tests/synthetic/):
  - labeled_sequences.csv     Fake protein sequences with fake peptide /
                              propeptide segment annotations, in the exact
                              column schema `src/utils/dataset.py` expects
                              (matches data/uniprot_2022/labeled_sequences.csv
                              in the original project).
  - graphpart_assignments.csv A fake 5-fold partition assignment file, in the
                              schema GraphPart produces (columns: AC, priority,
                              label-val, between_connectivity, cluster).
  - embeddings/<md5(seq)>.pt  One random tensor per sequence, shape
                              (seq_len, embedding_dim), dtype float32 --
                              stands in for precomputed frozen pLM embeddings.
                              Keyed by md5(sequence), exactly like the loader
                              in PrecomputedCSVForOverlapCRFDataset expects
                              (see src/utils/dataset.py:make_hashes /
                              PrecomputedCSVForOverlapCRFDataset.__getitem__).

None of the sequences, organisms, or annotations are real biological data --
everything is randomly generated. This exists only so `run.py` / `infer.py` /
`src/infer_and_score.py` can be exercised on CPU without any external data
dependency.

The generated peptide/propeptide segments are constructed to always form a
*valid* path through the project's own extended-state CRF grammar
(src/models/crf_models.py:CRFBaseModel.get_crf_constraints): segment lengths
are kept in [min_len, max_len] = [5, 50], segments never overlap each other,
and every sequence keeps at least one flanking "background" residue before
the first and after the last segment. The script double-checks this directly
against the project's own label-encoding and CRF-constraint code before
writing anything to disk (see `_validate_all` below) -- an invalid gold path
would silently trip the training loop's NaN-guard (loss becomes non-finite,
the optimizer step is skipped, and the run would finish looking like a
"successful" 0-progress training), which would be very easy to mistake for a
passing smoke test.

Usage:
    python3 tests/synthetic/generate_synthetic_data.py
    python3 tests/synthetic/generate_synthetic_data.py --n_sequences 40 --embedding_dim 1280 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Make the project's own label/grammar code importable regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.crf_label_utils import peptide_list_to_label_sequence  # noqa: E402
from src.models.crf_models import CRFBaseModel  # noqa: E402

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")  # standard 20, no ambiguity codes
FAKE_ORGANISMS = [
    "Ficticia synthetica (synthetic test organism)",
    "Exampla fabricata (synthetic test organism)",
    "Testium imaginarium (synthetic test organism)",
    "Placeholderia ficta (synthetic test organism)",
]

MIN_LEN, MAX_LEN = 5, 50  # must match CRFBaseModel(max_len=50, min_len=5)


def make_hash(sequence: str) -> str:
    return hashlib.md5(sequence.encode()).hexdigest()


def sample_non_overlapping_segments(seq_len: int, n_peptides: int, n_propeptides: int, rng: random.Random):
    """Place `n_peptides` + `n_propeptides` non-overlapping segments (1-based,
    inclusive coordinates) inside [2, seq_len-1], each length in [MIN_LEN, MAX_LEN],
    leaving residue 1 and residue seq_len as background so the first/last state
    of the sequence is always a valid CRF start/end state (0)."""
    n_total = n_peptides + n_propeptides
    placed = []
    attempts = 0
    while len(placed) < n_total and attempts < 200:
        attempts += 1
        length = rng.randint(MIN_LEN, min(MAX_LEN, max(MIN_LEN, (seq_len - 2) // max(1, n_total))))
        start = rng.randint(2, seq_len - length)  # leave >=1 background residue at the start
        end = start + length - 1
        if end >= seq_len:  # leave >=1 background residue at the end
            continue
        if any(not (end < s - 1 or start > e + 1) for s, e in placed):  # require a 1-residue gap
            continue
        placed.append((start, end))
    rng.shuffle(placed)
    peptides = placed[:n_peptides]
    propeptides = placed[n_peptides:n_peptides + n_propeptides]
    return sorted(peptides), sorted(propeptides)


def coords_to_string(coords) -> str:
    return ",".join(f"({s}-{e})" for s, e in coords)


def bits_from_coords(coords, seq_len: int) -> str:
    bits = [0] * seq_len
    for s, e in coords:
        for i in range(s - 1, e):
            bits[i] = 1
    return "".join(str(b) for b in bits)


def _validate_all(peptides, propeptides, seq_len: int) -> None:
    """Assert the generated (peptides, propeptides) pair is a valid path
    through the project's own CRF grammar, using the project's own code
    (not a reimplementation) so this check can't silently drift from
    src/models/crf_models.py or src/utils/crf_label_utils.py."""
    allowed, starts, ends = CRFBaseModel.get_crf_constraints(MAX_LEN, MIN_LEN, n_branches=2)
    allowed = set(allowed)

    pep_label = peptide_list_to_label_sequence(peptides, seq_len, start_state=1, max_len=MAX_LEN, min_len=MIN_LEN)
    propep_label = peptide_list_to_label_sequence(propeptides, seq_len, start_state=51, max_len=MAX_LEN, min_len=MIN_LEN)
    label = (pep_label + propep_label).astype(int)

    assert label[0] in starts, f"invalid start state {label[0]}"
    assert label[-1] in ends, f"invalid end state {label[-1]}"
    for prev, cur in zip(label[:-1], label[1:]):
        assert (int(prev), int(cur)) in allowed, f"illegal transition {prev}->{cur}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent))
    ap.add_argument("--n_sequences", type=int, default=40)
    ap.add_argument("--embedding_dim", type=int, default=1280, help="Must match --embedding_dim passed to training.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_seq_len", type=int, default=90)
    ap.add_argument("--max_seq_len", type=int, default=180)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    emb_dir = out_dir / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    partitions = []

    for i in range(args.n_sequences):
        seq_len = rng.randint(args.min_seq_len, args.max_seq_len)
        sequence = "".join(rng.choice(AMINO_ACIDS) for _ in range(seq_len))

        n_peptides = rng.choice([0, 1, 1, 2])       # mostly 1, sometimes 0 or 2
        n_propeptides = rng.choice([0, 0, 1])       # mostly 0, sometimes 1
        if n_peptides == 0 and n_propeptides == 0:
            # The real curated dataset (data/uniprot_2022/labeled_sequences.csv) only
            # ever contains annotated precursor proteins: empirically, 0/8449 real rows
            # have both 'coordinates' and 'propeptide_coordinates' empty. Match that
            # invariant here. Without it, PrecomputedCSVForOverlapCRFDataset.
            # _sample_from_overlapping_peptides(src/utils/dataset.py) crashes on a
            # zero-segment protein with `ValueError: not enough values to unpack` from
            # `zip(*[])` -- a real, pre-existing edge-case bug in the shipped research
            # code that this generator deliberately avoids triggering rather than
            # papering over (see README "Known issues" for details).
            n_peptides = 1
        peptides, propeptides = sample_non_overlapping_segments(seq_len, n_peptides, n_propeptides, rng)

        _validate_all(peptides, propeptides, seq_len)  # fail loudly if the grammar is violated

        protein_id = f"SYN{i:05d}"
        seq_hash = make_hash(sequence)

        rows.append({
            "protein_name": f"SYNPROT_{i:03d}",
            "sequence": sequence,
            "organism": rng.choice(FAKE_ORGANISMS),
            "is_peptide": bits_from_coords(peptides, seq_len),
            "coordinates": coords_to_string(peptides),
            "is_propeptide": bits_from_coords(propeptides, seq_len),
            "propeptide_coordinates": coords_to_string(propeptides),
            "protein_id": protein_id,
        })
        partitions.append({
            "AC": protein_id,
            "priority": False,
            "label-val": int(np_rng.integers(1, 30)),
            "between_connectivity": 0,
            "cluster": float(i % args.n_folds),
        })

        # Random per-residue embedding, standing in for a precomputed frozen
        # pLM embedding of shape (seq_len, embedding_dim). torch.save(...) writes
        # a plain tensor, which PrecomputedCSVForOverlapCRFDataset.__getitem__
        # loads via torch.load(..., weights_only=False) and uses directly
        # (see src/utils/dataset.py).
        embedding = torch.randn(seq_len, args.embedding_dim, dtype=torch.float32)
        torch.save(embedding, emb_dir / f"{seq_hash}.pt")

    data_file = out_dir / "labeled_sequences.csv"
    with open(data_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "protein_name", "sequence", "organism", "is_peptide",
            "coordinates", "is_propeptide", "propeptide_coordinates", "protein_id",
        ])
        writer.writeheader()
        writer.writerows(rows)

    partitioning_file = out_dir / "graphpart_assignments.csv"
    with open(partitioning_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["AC", "priority", "label-val", "between_connectivity", "cluster"])
        writer.writeheader()
        writer.writerows(partitions)

    print(f"Wrote {len(rows)} synthetic sequences.")
    print(f"  data_file:         {data_file}")
    print(f"  partitioning_file: {partitioning_file}")
    print(f"  embeddings_dir:    {emb_dir} ({len(rows)} .pt files, dim={args.embedding_dim})")
    print("All generated label sequences validated against CRFBaseModel.get_crf_constraints().")


if __name__ == "__main__":
    main()
