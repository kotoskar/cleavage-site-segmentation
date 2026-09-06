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
    """What the corrected matcher changes, per run, and how well it checks out.

    Each cell ships both readings of its own test fold: tolerance_metrics.json
    from the corrected matcher and original_metrics.json from the upstream one,
    both computed over the same decoded segments. A cell that ran to the end of
    training also recorded an original-matcher F1 at the time, which is what the
    reconstruction is checked against.
    """
    print("\nWhat the corrected matcher changes, per run")
    shifts = []
    for _, _add, run in ROWS + [("", "", "5cv_baseline_esm2_2022")]:
        by_outer = {}
        for cell in sorted(glob.glob(os.path.join(HERE, run, "outer*_inner*"))):
            f = os.path.join(cell, "original_metrics.json")
            if not os.path.exists(f):
                continue
            outer = int(os.path.basename(cell).split("_")[0][len("outer"):])
            by_outer.setdefault(outer, []).append(json.load(open(f))["tol3_all_f1"])
        if len(by_outer) != 5:
            continue
        o = st.mean([st.mean(v) for _, v in sorted(by_outer.items())])
        c = stored(run, 3, "f1")[0]
        shifts.append(c - o)
        print(f"  {run:26s} corrected {c:.4f}  original {o:.4f}  shift {c - o:+.4f}")
    if shifts:
        print(f"  range over the {len(shifts)} runs: {min(shifts):+.4f} to {max(shifts):+.4f}")

    # every cell that recorded an original-matcher F1 at training time, against
    # the same matcher run over the segments that cell saved
    diffs, outliers, no_record = [], [], 0
    for run in [r[2] for r in ROWS] + ["5cv_baseline_esm2_2022"]:
        for cell in sorted(glob.glob(os.path.join(HERE, run, "outer*_inner*"))):
            f = os.path.join(cell, "original_metrics.json")
            rec = os.path.join(cell, "cell_result.json")
            if not os.path.exists(f):
                continue
            got = json.load(open(f))["tol3_all_f1"]
            recorded = json.load(open(rec)).get("test_f1_all") if os.path.exists(rec) else None
            if recorded is None:
                no_record += 1
                continue
            d = abs(got - recorded)
            diffs.append(d)
            if d > 1e-2:
                outliers.append((run, os.path.basename(cell), got, recorded, d))
    ok = [d for d in diffs if d <= 1e-2]
    print(f"  reconstruction against the F1 each cell recorded at training time: "
          f"{len(diffs)} cells, {len(ok)} agree to {max(ok):.1e} at worst and "
          f"{st.median(ok):.1e} at the median")
    for run, name, got, recorded, d in outliers:
        print(f"    {run} {name} disagrees: {got:.4f} against {recorded:.4f}, {d:.4f}. "
              f"Its saved checkpoint does not reproduce what its training run recorded.")
    if no_record:
        print(f"  {no_record} cells never recorded one and are scored from their segments only")


def cells():
    """Rebuild each row of Table 1 from the individual cells behind it.

    Every cell carries tolerance_metrics.json, which holds the corrected
    matcher's counts on that cell's own test fold at each tolerance. Averaging
    the four inner cells of an outer fold and then the five outer folds has to
    give back what the run's aggregate file publishes. This covers all 20 cells
    of every run, including the ESM-C adapter run, whose GPU rescoring under the
    original matcher is only partial.
    """
    print("\nTable 1 rebuilt from the individual cells")
    worst, n_cells, n_runs, short = 0.0, 0, 0, []
    for _, _add, run in ROWS + [("", "", "5cv_baseline_esm2_2022")]:
        by_outer = {}
        for cell in sorted(glob.glob(os.path.join(HERE, run, "outer*_inner*"))):
            f = os.path.join(cell, "tolerance_metrics.json")
            if not os.path.exists(f):
                continue
            outer = int(os.path.basename(cell).split("_")[0][len("outer"):])
            by_outer.setdefault(outer, []).append(json.load(open(f))["tol3_all_f1"])
            n_cells += 1
        if len(by_outer) != 5 or any(len(v) != 4 for v in by_outer.values()):
            short.append(run)
            continue
        rebuilt = st.mean([st.mean(v) for _, v in sorted(by_outer.items())])
        worst = max(worst, abs(rebuilt - stored(run, 3, "f1")[0]))
        n_runs += 1
    print(f"  {n_cells} cells over {n_runs} runs, worst disagreement with the published "
          f"aggregate: {worst:.1e}")
    if short:
        print(f"  not rebuilt, fewer than 5x4 cells present: {', '.join(short)}")


def epochs():
    """Which cells stopped before the end of the schedule, and what that cost.

    A cell that ran to the end records its selected epoch in cell_result.json.
    Cells that stopped early carry no such record and are scored from the
    checkpoint their validation curve had already selected, which
    valid_metrics.json reports; this prints each one against the latest
    selection among the cells of the same run that did finish. Every cell of
    every run released here now carries a record, so this section is empty, and
    it stays in the script so that the check is visible rather than assumed.
    """
    print("\nEarly-stopped cells, and where selection had settled")
    any_short = False
    for _, _add, run in ROWS + [("", "", "5cv_baseline_esm2_2022")]:
        finished, short = [], []
        for cell in sorted(glob.glob(os.path.join(HERE, run, "outer*_inner*"))):
            v = os.path.join(cell, "valid_metrics.json")
            epoch = json.load(open(v)).get("epoch") if os.path.exists(v) else None
            if os.path.exists(os.path.join(cell, "cell_result.json")):
                finished.append(epoch)
            else:
                short.append((os.path.basename(cell), epoch))
        if not short:
            continue
        any_short = True
        latest = max(e for e in finished if e is not None)
        print(f"  {run}: {len(short)} of {len(short) + len(finished)} cells stopped early")
        for name, epoch in short:
            print(f"    {name:16s} checkpoint selected at epoch {epoch}")
        print(f"    the {len(finished)} cells of this run that finished selected at "
              f"epoch {min(e for e in finished if e is not None)}-{latest}, so no finished "
              f"cell of it ever improved after epoch {latest}")
    if not any_short:
        print("  none: every cell of every run ran to the end of its schedule")


def tolerance():
    """Gaps to the ESM-2 base at every tolerance, and how they grow.

    The headline number is the +-3 one, but a design loop reads the other end of
    this curve, so both ends are printed with the paired spread.
    """
    print("\nGap to the ESM-2 base at each tolerance, paired by outer fold")
    print(f"{'configuration':34s} {'+-3':>16s} {'+-2':>16s} {'+-1':>16s} {'exact':>16s}")
    for emb, add, run in ROWS:
        if run == BASE2:
            continue
        cells = []
        for tol in (3, 2, 1, 0):
            d = [a - b for a, b in zip(per_outer(run, tol), per_outer(BASE2, tol))]
            m, sd = ms(d)
            cells.append(f"{m:+.4f}+-{sd:.4f}")
        print(f"{emb + ', ' + add:34s} " + " ".join(f"{c:>16s}" for c in cells))
    print("  growth is the last column minus the first; both papers quote it")


def overlap():
    """The tolerance-free suite: what survives when the acceptance window goes."""
    path = os.path.join(HERE, "suites", "segment_quality_cv_anyoverlap.json")
    if not os.path.exists(path):
        print("\n(no tolerance-free suite in this release)")
        return
    d = json.load(open(path))
    base = d["baseline"]
    print(f"\nSegment F1 under IoU matching, against {base}")
    thr = [t for t in d["iou_thresholds"]]
    head = " ".join(f"{('any' if t < 1e-6 else t):>9}" for t in thr)
    print(f"{'model':34s} {head}")
    for name, entry in d["models"].items():
        m = entry.get("metrics", {})
        row = []
        for t in thr:
            key = f"all_f1_iou{'1e-09' if t < 1e-6 else t}"
            row.append(f"{m[key]['mean']:9.4f}" if key in m else "        -")
        print(f"{name:34s} " + " ".join(row))
    print("  the leftmost column asks only for one shared residue: that is the")
    print("  detection control, and it is where the additions buy nothing")


def sites():
    """Which end of a segment each block fixes."""
    path = os.path.join(HERE, "suites", "boundary_error_cv_fp32base.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    print(f"\nPaired placement, against {d['baseline']}")
    for name, v in d["variants"].items():
        p_, l = v["paired"], v["localisation"]
        print(f"  {name:26s} paired instances {p_['n']:6d}   exact share "
              f"{p_['exact_base']:.4f} -> {p_['exact_variant']:.4f}")
        print(f"  {'':26s} base localises {l['n_localised_base']}, this run "
              f"{l['n_localised_variant']} ({l['n_localised_variant'] - l['n_localised_base']:+d})")
    print("  counts are segment-instances: each true segment is scored by the four")
    print("  cells that share its outer fold, so divide by four for distinct segments")


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
    for name in ("table1", "effects", "tolerance", "overlap", "sites", "matcher",
                 "cells", "epochs", "folds", "reproduction"):
        ap.add_argument(f"--{name}", action="store_true")
    args = ap.parse_args()
    order = ["table1", "effects", "tolerance", "overlap", "sites", "matcher",
             "cells", "epochs", "folds", "reproduction"]
    chosen = [n for n in order if getattr(args, n, False)]
    for name in (chosen or order):
        globals()[name]()


if __name__ == "__main__":
    main()
