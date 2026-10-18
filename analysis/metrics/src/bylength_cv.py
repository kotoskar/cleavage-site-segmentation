#!/usr/bin/env python3
"""F1 / precision / recall by segment length, under the 5x4 nested CV.

The single-split version of this breakdown lives in
`analysis/metrics/src/generators/bylength_fig.py` (figure 8 of the paper) and is
computed from the per-segment CSVs of the earlier protocol. This script answers
the same question on the nested-CV grid instead, reading the span dumps
runs/<model>/outer{o}_inner{k}/segments.json.gz written by tolerance_sweep_cv.py.
No GPU, no re-inference.

Definitions are the ones the single-split figure uses, so the two are readable
side by side:

  recall    over TRUE segment groups, binned by the group's length; a group
            counts as found when the paper's corrected +-3 matcher matches it.
  precision over PREDICTED segments, binned by the PREDICTED length; a
            prediction is a false positive when the same matcher leaves it
            unmatched.
  F1        the harmonic mean of the two, micro-averaged inside a cell.

Grouping and matching are delegated to `analysis.errors.src.error_analysis.
match_protein`, the authoritative +-3 matcher behind every number in the tables,
so a group's length is the length of its longest member exactly as there.

Aggregation is the paper's: micro inside a cell, mean over the 4 inner cells of
an outer fold, then mean +- std across the 5 outer folds. The 4 inner cells of
one outer fold share a test partition, so they are 4 models scored on the same
proteins -- averaging their F1 is right and pooling their counts would count
each protein four times.

Usage:
  env/bin/python analysis/metrics/src/bylength_cv.py \
      [--models 5cv_baseline_esm2 ...] [--tolerance 3] [--min-true 25]
      [--out analysis/metrics/bylength_cv.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / "run.py").exists())
sys.path.insert(0, str(REPO))

from analysis.errors.src.error_analysis import match_protein  # noqa: E402

TASKS = ("peptides", "propeptides")
DEFAULT_MODELS = ["5cv_baseline_esm2_fp32", "5cv_esm2_boundary", "5cv_esm2_adapter_only",
                  "5cv_esm2_full", "5cv_esmc6b_plain"]
EDGES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 51]
BINS = [f"{EDGES[i]}-{EDGES[i + 1] - 1}" for i in range(len(EDGES) - 1)]


def length_bin(n):
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= n < EDGES[i + 1]:
            return BINS[i]
    return None


def prf(tp, fn, fp):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return f, p, r


def score_cell(records, tol):
    """({bin: [tp, fn, fp]}, n_fp_outside_bins) for one cell, both types pooled.

    A false positive longer than 50 residues belongs to no length bin, so it is
    counted separately rather than dropped silently: it is about 1-2% of the
    unmatched predictions and it is why pooling the bins gives an F1 a couple of
    thousandths above the published +-3 number. Every true group is in range.
    """
    acc = defaultdict(lambda: [0, 0, 0])
    overflow = 0
    for rec in records:
        for task in TASKS:
            true = [tuple(map(int, x)) for x in rec[task]["true"]]
            pred = [tuple(map(int, x)) for x in rec[task]["pred"]]
            if not true and not pred:
                continue
            groups, pred_matched = match_protein(true, pred, tol)
            for g in groups:
                b = length_bin(g["length"])
                if b is None:
                    continue
                acc[b][0 if g["matched"] else 1] += 1
            for (ps, pe), matched in zip(pred, pred_matched):
                if matched:
                    continue                       # true positives are counted above
                b = length_bin(pe - ps + 1)
                if b is None:
                    overflow += 1
                else:
                    acc[b][2] += 1
    return acc, overflow


def load_cells(model):
    cells = {}
    for d in sorted((Path("runs") / model).glob("outer*_inner*")):
        f = d / "segments.json.gz"
        if not f.exists():
            continue
        outer = int(d.name.split("outer")[1].split("_")[0])
        inner = int(d.name.split("inner")[1])
        with gzip.open(f, "rt") as fh:
            cells[(outer, inner)] = json.load(fh)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--tolerance", type=int, default=3)
    ap.add_argument("--min-true", type=int, default=25,
                    help="drop a length bin whose mean true-group count per cell is below this")
    ap.add_argument("--out", default="analysis/metrics/bylength_cv.json")
    args = ap.parse_args()

    out = {"tolerance": args.tolerance, "bins": BINS, "min_true": args.min_true, "models": {}}
    counts = {}
    for model in args.models:
        cells = load_cells(model)
        if not cells:
            print(f"[skip] {model}: no segment dumps")
            continue
        rows = []
        for (outer, inner), recs in cells.items():
            acc, overflow = score_cell(recs, args.tolerance)
            row = {"outer": outer, "inner": inner, "fp_outside_bins": overflow}
            for b in BINS:
                tp, fn, fp = acc[b]
                f, p, r = prf(tp, fn, fp)
                row[f"{b}_f1"], row[f"{b}_precision"], row[f"{b}_recall"] = f, p, r
                row[f"{b}_n_true"] = tp + fn
                row[f"{b}_n_pred"] = tp + fp
            rows.append(row)
        df = pd.DataFrame(rows)
        per_outer = df.groupby("outer").mean(numeric_only=True)
        m = {}
        for col in per_outer.columns:
            if col == "inner":
                continue
            v = per_outer[col]
            m[col] = {"mean": round(float(v.mean()), 6),
                      "std": round(float(v.std()), 6),
                      "per_outer": [round(float(x), 6) for x in v]}
        out["models"][model] = {"n_cells": len(rows), "metrics": m}
        counts[model] = {b: m[f"{b}_n_true"]["mean"] for b in BINS}
        line = "  ".join(f"{b}:{m[f'{b}_f1']['mean']:.3f}" for b in BINS)
        print(f"[ok] {model:24s} {len(rows)} cells  F1 by length  {line}")
        print(f"{'':29s} false positives longer than 50 residues, outside every bin: "
              f"{m['fp_outside_bins']['mean']:.1f} per cell")

    # A bin is kept only if every model has enough true segments in it, so the
    # panels compare the same x range.
    keep = [b for b in BINS
            if all(counts[m][b] >= args.min_true for m in counts)] if counts else []
    dropped = [b for b in BINS if b not in keep]
    out["kept_bins"] = keep
    out["dropped_bins"] = dropped
    if dropped:
        print(f"[cut] dropped bins (mean true groups/cell < {args.min_true}): {dropped}")
    print("[n]  mean true groups per cell, baseline: "
          + "  ".join(f"{b}:{counts[args.models[0]][b]:.0f}" for b in BINS))

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[json] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
