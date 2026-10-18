#!/usr/bin/env python3
"""Is the tolerance curve a parallel translation of the baseline, or is the
boundary genuinely sharper?

The retained-F1 fraction alone cannot answer that: a model that simply finds
MORE segments scores higher at every tolerance, and its retention ratio moves
even if each individual boundary is placed no better. This script separates
the two effects using the segments dumped by tolerance_sweep_cv.py.

Three views, in increasing order of how well detection is controlled:

  A. ABSOLUTE GAP.  Delta F1 vs the baseline at each tolerance. If the curves
     are a parallel translation, this gap is flat in tolerance. If the boundary
     is sharper, the gap widens as the tolerance tightens.

  B. LOCALISATION ERROR.  For every TRUE segment, the error of the best
     available prediction, err = min over preds of max(|dstart|, |dend|).
     This is the quantity the +-tau criterion thresholds, so the whole
     tolerance curve is a function of its distribution. Reported over the
     true segments the model localises at all (err <= cap).

  C. PAIRED, DETECTION-CONTROLLED.  Restricted to the true segments that BOTH
     the baseline and the variant localise within +-3, in the SAME cell, the
     same protein, the same segment: the paired distribution of err_base vs
     err_variant. Detection is held fixed by construction, so any shift here
     is localisation and nothing else. This is the decisive test.

Cells are paired across models by (outer, inner), which share a test partition
by construction of the nested-CV grid.

Usage:
  env/bin/python analysis/metrics/src/boundary_error_cv.py \
      [--baseline 5cv_baseline_esm2] [--variants NAME ...] [--out DIR]
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents
                            if (p / "run.py").exists())))

import numpy as np

TASKS = ("peptides", "propeptides")
CAP = 50  # errors above this are "not localised at all"


def seg_error(true_seg, preds):
    """min over predicted segments of max(|dstart|, |dend|); CAP if none."""
    ts, te = true_seg
    best = CAP
    for ps, pe in preds:
        e = max(abs(ps - ts), abs(pe - te))
        if e < best:
            best = e
    return best


def cell_errors(path: pathlib.Path):
    """{(protein, task, true_start, true_end): err} for one cell."""
    with gzip.open(path, "rt") as fh:
        segments = json.load(fh)
    out = {}
    for rec in segments:
        for task in TASKS:
            preds = [tuple(x) for x in rec[task]["pred"]]
            for t in rec[task]["true"]:
                out[(rec["name"], task, int(t[0]), int(t[1]))] = seg_error(tuple(t), preds)
    return out


def load_model(name: str):
    """{(outer,inner): {seg_key: err}} for every cell that has a dump."""
    root = pathlib.Path("runs") / name
    cells = {}
    for d in sorted(root.glob("outer*_inner*")):
        p = d / "segments.json.gz"
        if not p.exists():
            continue
        o, i = d.name.replace("outer", "").split("_inner")
        cells[(int(o), int(i))] = cell_errors(p)
    return cells


def f1_at(errs_true, n_pred, tol):
    """Recall side from true-segment errors; precision needs pred counts."""
    tp = int((errs_true <= tol).sum())
    fn = len(errs_true) - tp
    fp = max(n_pred - tp, 0)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="5cv_baseline_esm2_fp32")
    ap.add_argument("--variants", nargs="*",
                    default=["5cv_esm2_boundary", "5cv_esm2_adapter_only", "5cv_esm2_full"])
    ap.add_argument("--tolerances", nargs="*", type=int, default=[3, 2, 1, 0])
    ap.add_argument("--out", default="analysis/metrics")
    args = ap.parse_args()

    base = load_model(args.baseline)
    if not base:
        print(f"no dumps for baseline {args.baseline}; run tolerance_sweep_cv.py first")
        return 1
    print(f"baseline {args.baseline}: {len(base)} cells with dumps")

    report = {"baseline": args.baseline, "cap": CAP, "variants": {}}

    # ---- A. absolute gap, straight from the sweep summaries -------------
    def tol_summary(name):
        p = pathlib.Path("runs") / name / "nested_cv_tolerance.json"
        return json.load(open(p)) if p.exists() else None

    b_sum = tol_summary(args.baseline)
    print("\n=== A. absolute F1 gap vs baseline, by tolerance ===")
    if b_sum:
        head = "  ".join(f"tol{t}" for t in args.tolerances)
        print(f"{'model':34s} {head}")
        print(f"{args.baseline:34s} " +
              "  ".join(f"{b_sum[f'cv_tol{t}_all_f1_mean']:.3f}" for t in args.tolerances))
        for v in args.variants:
            vs = tol_summary(v)
            if not vs:
                print(f"{v:34s} (sweep not finished)")
                continue
            gaps = [vs[f"cv_tol{t}_all_f1_mean"] - b_sum[f"cv_tol{t}_all_f1_mean"]
                    for t in args.tolerances]
            print(f"{v:34s} " + "  ".join(f"{g:+.3f}" for g in gaps) +
                  f"   spread={max(gaps)-min(gaps):+.3f}")
            report["variants"].setdefault(v, {})["abs_gap"] = dict(
                zip(map(str, args.tolerances), [round(g, 4) for g in gaps]))
    print("  a flat row = parallel translation; a row growing to the right = sharper boundary")

    # ---- B / C. localisation -------------------------------------------
    print("\n=== B. localisation error over true segments (median / mean / <=1 / <=0) ===")
    print("=== C. paired on segments BOTH models localise within +-3          ===")
    for v in args.variants:
        var = load_model(v)
        shared = sorted(set(base) & set(var))
        if not shared:
            print(f"\n{v}: no overlapping cells with dumps yet")
            continue

        b_all, v_all = [], []
        pair_b, pair_v = [], []
        per_cell_delta = defaultdict(list)   # outer -> paired err deltas
        per_cell_exact = defaultdict(list)   # outer -> paired exact-hit deltas
        for key in shared:
            bc, vc = base[key], var[key]
            common = bc.keys() & vc.keys()
            for k in common:
                eb, ev = bc[k], vc[k]
                b_all.append(eb)
                v_all.append(ev)
                if eb <= 3 and ev <= 3:
                    pair_b.append(eb)
                    pair_v.append(ev)
                    per_cell_delta[key[0]].append(ev - eb)
                    per_cell_exact[key[0]].append((ev == 0) - (eb == 0))

        b_all = np.array(b_all); v_all = np.array(v_all)
        pb = np.array(pair_b); pv = np.array(pair_v)
        loc_b = b_all[b_all <= 3]; loc_v = v_all[v_all <= 3]

        print(f"\n{v}   ({len(shared)} cells, {len(b_all)} true segments)")
        print(f"  B  baseline  localised={len(loc_b)}  median={np.median(loc_b):.1f}  "
              f"mean={loc_b.mean():.2f}  exact={np.mean(loc_b == 0):.3f}  <=1={np.mean(loc_b <= 1):.3f}")
        print(f"  B  variant   localised={len(loc_v)}  median={np.median(loc_v):.1f}  "
              f"mean={loc_v.mean():.2f}  exact={np.mean(loc_v == 0):.3f}  <=1={np.mean(loc_v <= 1):.3f}")

        if len(pb):
            d = pv - pb
            better = float(np.mean(d < 0)); worse = float(np.mean(d > 0))
            # paired bootstrap over outer folds
            outers = sorted(per_cell_delta)
            fold_means = np.array([np.mean(per_cell_delta[o]) for o in outers])
            print(f"  C  n={len(pb)} paired segments   "
                  f"mean err {pb.mean():.3f} -> {pv.mean():.3f}  (delta {d.mean():+.3f} residues)")
            print(f"  C  variant tighter on {better:.1%}, looser on {worse:.1%}, equal on "
                  f"{1-better-worse:.1%}")
            print(f"  C  exact-match share {np.mean(pb == 0):.3f} -> {np.mean(pv == 0):.3f}  "
                  f"({np.mean(pv == 0) - np.mean(pb == 0):+.3f})")
            print(f"  C  per-outer-fold delta(mean err): " +
                  " ".join(f"{m:+.3f}" for m in fold_means) +
                  (f"   (mean {fold_means.mean():+.3f} +- {fold_means.std(ddof=1):.3f})"
                   if len(fold_means) > 1 else ""))
            ex_means = np.array([np.mean(per_cell_exact[o]) for o in outers])
            print(f"  C  per-outer-fold delta(exact share): " +
                  " ".join(f"{m:+.3f}" for m in ex_means) +
                  (f"   (mean {ex_means.mean():+.3f} +- {ex_means.std(ddof=1):.3f}, "
                   f"{int((ex_means > 0).sum())}/{len(ex_means)} folds positive)"
                   if len(ex_means) > 1 else ""))
            report["variants"].setdefault(v, {})["paired"] = {
                "n": len(pb),
                "mean_err_base": round(float(pb.mean()), 4),
                "mean_err_variant": round(float(pv.mean()), 4),
                "delta": round(float(d.mean()), 4),
                "frac_tighter": round(better, 4),
                "frac_looser": round(worse, 4),
                "exact_base": round(float(np.mean(pb == 0)), 4),
                "exact_variant": round(float(np.mean(pv == 0)), 4),
                "per_outer_delta": [round(float(m), 4) for m in fold_means],
            }
            report["variants"][v]["localisation"] = {
                "base_mean": round(float(loc_b.mean()), 4),
                "variant_mean": round(float(loc_v.mean()), 4),
                "base_exact": round(float(np.mean(loc_b == 0)), 4),
                "variant_exact": round(float(np.mean(loc_v == 0)), 4),
                "n_localised_base": int(len(loc_b)),
                "n_localised_variant": int(len(loc_v)),
            }

    out = pathlib.Path(args.out) / "boundary_error_cv.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
