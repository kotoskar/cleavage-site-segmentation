#!/usr/bin/env python3
"""Recompute the paper's numbers from the released per-cell scores.

Nothing here touches a GPU or a model: every figure in the paper's tables and
Results section is an aggregate over the small JSON files in this directory, and
this script rebuilds them so a reader can check the arithmetic rather than take
it on trust.

    python3 results/reproduce_paper_numbers.py            # all sections
    python3 results/reproduce_paper_numbers.py --table1   # just Table 1

What is here, per configuration:
  <run>/nested_cv_tolerance.json          F1/precision/recall at each tolerance,
                                          means and per-outer-fold values
  <run>/nested_cv_summary_corrected.json  the same F1 under both matchers
  <run>/outer<o>_inner<k>/...             the individual cells those means are
                                          built from, both matchers
  suites/                                 the tolerance-free IoU sweep and the
                                          cleavage-site scores

Aggregation, everywhere: micro-average inside a cell, mean over the four inner
cells of an outer fold, then mean and sample standard deviation (ddof=1) over the
five outer folds. Spreads are across outer folds and not across all 20 cells,
because four models sharing an outer fold share a test set.
"""
from __future__ import annotations
import argparse, glob, json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
BASE2, BASEC = "5cv_baseline_esm2_fp32", "5cv_esmc6b_plain"

ROWS = [
    ("ESM-2", "none",          BASE2),
    ("ESM-2", "boundary head", "5cv_esm2_boundary"),
    ("ESM-2", "adapter",       "5cv_esm2_adapter_only"),
    ("ESM-2", "both",          "5cv_esm2_full"),
    ("ESM-C 6B", "none",          BASEC),
    ("ESM-C 6B", "boundary head", "5cv_esmc6b_boundary"),
    ("ESM-C 6B", "adapter",       "5cv_esmc6b_adapter_only"),
    ("ESM-C 6B", "both",          "5cv_esmc6b_full"),
]


def load(run, name):
    with open(os.path.join(HERE, run, name)) as fh:
        return json.load(fh)


def per_outer(run, tol, task="all"):
    """The five outer-fold F1 values behind one cell of Table 1.

    Only F1 is stored per fold; precision and recall are stored as the mean and
    spread they are printed with, since nothing in the paper pairs them by fold.
    """
    d = load(run, "nested_cv_tolerance.json")[f"per_outer_tol{tol}_{task}_f1"]
    return [d[k] for k in sorted(d, key=int)] if isinstance(d, dict) else list(d)


def stored(run, tol, metric, task="all"):
    """The mean and spread as the sweep recorded them."""
    d = load(run, "nested_cv_tolerance.json")
    return d[f"cv_tol{tol}_{task}_{metric}_mean"], d[f"cv_tol{tol}_{task}_{metric}_std"]


def ms(values):
    return st.mean(values), st.stdev(values)


def table1():
    print("Table 1 — 5x4 nested cross-validation, corrected matcher")
    print(f"{'embedding':10s} {'additions':14s} {'P (+-3)':>15s} {'R (+-3)':>15s}"
          f" {'F1 +-3':>15s} {'F1 +-2':>15s} {'F1 +-1':>15s} {'F1 exact':>15s} {'growth':>15s}")
    for emb, add, run in ROWS:
        cells = []
        for metric, tol in (("precision", 3), ("recall", 3), ("f1", 3), ("f1", 2), ("f1", 1), ("f1", 0)):
            m, s = ms(per_outer(run, tol)) if metric == "f1" else stored(run, tol, metric)
            cells.append(f"{m:.3f}+-{s:.3f}")
        # growth: how much the gap to the ESM-2 base widens between +-3 and exact,
        # paired by outer fold so the spread means something
        if run == BASE2:
            cells.append("—".rjust(13))
        else:
            b3, b0 = per_outer(BASE2, 3), per_outer(BASE2, 0)
            g3 = [a - b for a, b in zip(per_outer(run, 3), b3)]
            g0 = [a - b for a, b in zip(per_outer(run, 0), b0)]
            m, s = ms([a - b for a, b in zip(g0, g3)])
            cells.append(f"{m:+.3f}+-{s:.3f}")
        print(f"{emb:10s} {add:14s} " + " ".join(f"{c:>15s}" for c in cells))


def effects():
    print("\nEffects, paired by outer fold against each embedding's own base")
    for base, label in ((BASE2, "ESM-2"), (BASEC, "ESM-C 6B")):
        b = per_outer(base, 3)
        for _, add, run in [r for r in ROWS if r[0].startswith(label[:5]) and r[2] != base]:
            d = [a - c for a, c in zip(per_outer(run, 3), b)]
            m, s = ms(d)
            print(f"  {label:9s} {add:14s} {m:+.4f} +- {s:.4f}   positive on {sum(1 for x in d if x > 0)}/5 folds")
    swap = [a - c for a, c in zip(per_outer(BASEC, 3), per_outer(BASE2, 3))]
    m, s = ms(swap)
    print(f"  embedding swap, base architecture: {m:+.4f} +- {s:.4f}   {sum(1 for x in swap if x > 0)}/5")
    inter = {}
    for lbl, base, head, ad, both in (
            ("ESM-2", BASE2, "5cv_esm2_boundary", "5cv_esm2_adapter_only", "5cv_esm2_full"),
            ("ESM-C 6B", BASEC, "5cv_esmc6b_boundary", "5cv_esmc6b_adapter_only", "5cv_esmc6b_full")):
        d = [f - h - a + b for f, h, a, b in
             zip(per_outer(both, 3), per_outer(head, 3), per_outer(ad, 3), per_outer(base, 3))]
        inter[lbl] = ms(d)
        print(f"  interaction (both - head - adapter + base), {lbl}: "
              f"{inter[lbl][0]:+.4f} +- {inter[lbl][1]:.4f}")
    print("  (the paper says the two are 'close to the sum', which is what these say)")


def matcher():
    print("\nWhat the corrected matcher changes, per run")
    shifts = []
    for _, add, run in ROWS:
        p = os.path.join(HERE, run, "nested_cv_summary_corrected.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        c = d.get("cv_f1_all_mean_corrected", d.get("cv_corr_all_f1_mean"))
        o = d.get("cv_f1_all_mean_original", d.get("cv_orig_all_f1_mean"))
        if c is None or o is None:
            continue
        shifts.append(c - o)
        print(f"  {run:26s} corrected {c:.4f}  original {o:.4f}  shift {c - o:+.4f}")
    if shifts:
        print(f"  range over the {len(shifts)} runs whose every cell carries both matchers: "
              f"{min(shifts):+.4f} to {max(shifts):+.4f}")
        print("  (the ESM-C adapter run is absent here: only some of its cells were scored "
              "under the original matcher, so it has no comparable aggregate)")

    # the per-cell agreement check: the rescored original matcher against the F1
    # each cell recorded at training time
    diffs = []
    for run in [r[2] for r in ROWS] + ["5cv_baseline_esm2_2022"]:
        for cell in sorted(glob.glob(os.path.join(HERE, run, "outer*_inner*"))):
            rec = os.path.join(cell, "cell_result_corrected.json")
            alt = os.path.join(cell, "corrected_metrics.json")
            src = rec if os.path.exists(rec) else (alt if os.path.exists(alt) else None)
            if not src:
                continue
            s = json.load(open(src))
            if run == "5cv_esmc6b_adapter_only":
                continue          # incomplete original-matcher records, see above
            a = s.get("test_orig_all_f1", s.get("orig_all_f1"))
            rec = os.path.join(cell, "cell_result.json")
            b = json.load(open(rec)).get("test_f1_all") if os.path.exists(rec) else None
            if a is not None and b is not None:
                diffs.append(abs(a - b))
    if diffs:
        print(f"  rescored original matcher against the F1 each cell recorded at training "
              f"time, over {len(diffs)} cells: worst {max(diffs):.1e}, median "
              f"{st.median(diffs):.1e}")


def folds():
    print("\nWhy the folds are not exchangeable")
    v = per_outer(BASE2, 3)
    print("  base architecture, outer folds 0 to 4: " + ", ".join(f"{x:.3f}" for x in v))
    print(f"  range {max(v) - min(v):.3f}, against a boundary-head effect of "
          f"{ms([a - b for a, b in zip(per_outer('5cv_esm2_boundary', 3), v)])[0]:+.3f}")


def reproduction():
    print("\nThe 2022-release reproduction")
    for run, label in ((BASE2, "2026 rebuild"), ("5cv_baseline_esm2_2022", "2022 release")):
        p = os.path.join(HERE, run, "nested_cv_summary_corrected.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        c = d.get("cv_f1_all_mean_corrected", d.get("cv_corr_all_f1_mean"))
        s = d.get("cv_f1_all_std_corrected", d.get("cv_corr_all_f1_std"))
        o = d.get("cv_f1_all_mean_original", d.get("cv_orig_all_f1_mean"))
        print(f"  {label:14s} corrected {c:.4f} +- {s:.4f}   original {o:.4f}")
    print("  published DeepPeptide: precision 0.68, recall 0.49 at +-3, implying F1",
          f"{2 * 0.68 * 0.49 / (0.68 + 0.49):.3f}")


def main():
    ap = argparse.ArgumentParser()
    for name in ("table1", "effects", "matcher", "folds", "reproduction"):
        ap.add_argument(f"--{name}", action="store_true")
    args = ap.parse_args()
    chosen = [n for n in ("table1", "effects", "matcher", "folds", "reproduction")
              if getattr(args, n)]
    for name in (chosen or ["table1", "effects", "matcher", "folds", "reproduction"]):
        globals()[name]()


if __name__ == "__main__":
    main()
