#!/usr/bin/env python3
"""Boundary-sharpness metrics for the nested-CV runs that do NOT depend on the
+-3 acceptance window.

Why
---
The headline metric of the paper accepts a predicted segment when both of its
ends land within +-3 residues of the annotation. Three is our own choice, and
the window is absolute, so it is relatively forgiving on a 5-residue peptide
(a 3-residue slip is 60% of its length) and relatively strict on a 45-residue
propeptide. A reviewer can reasonably ask whether the effect survives a
criterion we did not pick. These metrics answer that:

  * IoU (Jaccard) between a predicted and an annotated span, and segment F1 at
    an IoU threshold -- the standard detection criterion, scale-free.
  * The displacement of each end separately, signed and absolute, which the
    symmetric max(|dstart|, |dend|) error of boundary_error_cv.py hides.
  * Cleavage-site (event) precision/recall: every annotated boundary is one
    event, scored by distance, with no segment matching at all.
  * Residue-level precision/recall/F1/MCC/Jaccard: no matching whatsoever, so
    no matcher can be blamed for the result.
  * Structure: over-/under-segmentation, splits, merges, length ratio.
  * A paired, detection-controlled IoU comparison against the baseline, on the
    segments BOTH models find, which is the decisive test for "sharper" as
    opposed to "finds more".

Everything is computed from runs/<model>/outer{o}_inner{k}/segments.json.gz,
the span dumps written by tolerance_sweep_cv.py. No GPU, no re-inference; the
whole suite runs in a few seconds.

Aggregation follows the rest of the paper: a metric is pooled (micro) inside a
cell, averaged over the 4 inner cells of an outer fold, then reported as
mean +- std across the 5 outer folds.

Matching note: the IoU matching here is one-to-one and greedy in IoU, the
detection convention. The paper's +-3 matcher instead groups overlapping true
segments and lets a group count as found if any member is hit; the two are not
interchangeable, which is why the +-3 numbers are recomputed here as a
cross-check rather than reused.

Usage:
  env/bin/python analysis/metrics/src/segment_quality_cv.py \
      [--models 5cv_baseline_esm2 ...] [--baseline 5cv_baseline_esm2] \
      [--out analysis/metrics/segment_quality_cv.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / "run.py").exists())
sys.path.insert(0, str(REPO))

TASKS = ("peptides", "propeptides")
DEFAULT_MODELS = ["5cv_baseline_esm2_fp32", "5cv_esm2_boundary", "5cv_esm2_adapter_only",
                  "5cv_esm2_full", "5cv_esmc6b_plain"]
IOU_THRESHOLDS = (1e-9, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95)
SITE_TOLERANCES = (0, 1, 2, 3)
LENGTH_BINS = [(5, 9), (10, 14), (15, 19), (20, 24), (25, 29),
               (30, 34), (35, 39), (40, 44), (45, 50)]
# "not localised at all" ceiling for the symmetric boundary error, matching
# boundary_error_cv.py so the two scripts' detection gates agree.
CAP_ERR = 50


# --------------------------------------------------------------- geometry ---

def iou(a, b):
    """Jaccard of two inclusive 1-based residue spans."""
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union


def greedy_iou_match(true, pred):
    """One-to-one matching, greedy in IoU. -> [(t_idx, p_idx, iou)] with iou > 0."""
    if not true or not pred:
        return []
    cand = []
    for i, t in enumerate(true):
        for j, p in enumerate(pred):
            v = iou(t, p)
            if v > 0:
                cand.append((v, i, j))
    # Ties broken by index so the result does not depend on dict/list order.
    cand.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_t, used_p, out = set(), set(), []
    for v, i, j in cand:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        out.append((i, j, v))
    return out


def greedy_site_match(true_sites, pred_sites, tol):
    """Count annotated boundary positions recovered within `tol`, one-to-one."""
    if not true_sites or not pred_sites:
        return 0
    cand = []
    for i, t in enumerate(true_sites):
        for j, p in enumerate(pred_sites):
            d = abs(p - t)
            if d <= tol:
                cand.append((d, i, j))
    cand.sort()
    used_t, used_p, tp = set(), set(), 0
    for d, i, j in cand:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        tp += 1
    return tp


def prf(tp, fn, fp):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def mcc(tp, tn, fp, fn):
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn - fp * fn) / den) if den else 0.0


def length_bin(n):
    for lo, hi in LENGTH_BINS:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return None


# ------------------------------------------------------------ per-cell run ---

def score_cell(records, lengths, keep=None):
    """All metrics for one nested-CV cell, plus the per-segment IoU/error map
    used later for the paired comparison against the baseline.

    `keep`, when given, is the set of protein names every model has in this
    cell; the dumps do not agree exactly (the ESM-C pipeline carries 123 long
    proteins the ESM-2 one drops), and an unpaired table must not compare rows
    scored on different test sets.
    """
    if keep is not None:
        records = [r for r in records if r["name"] in keep]
    acc = {t: defaultdict(float) for t in TASKS}
    ious = {t: [] for t in TASKS}
    dstart = {t: [] for t in TASKS}
    dend = {t: [] for t in TASKS}
    lenratio = {t: [] for t in TASKS}
    bin_iou = defaultdict(list)                       # (task, bin) -> [iou], matched only
    bin_true = defaultdict(int)                       # bin -> n TRUE segments
    bin_best = defaultdict(float)                     # bin -> sum of best IoU over ALL true
    dstart_in = {t: [] for t in TASKS}                # interior boundaries only
    dend_in = {t: [] for t in TASKS}
    anchored = {t: defaultdict(int) for t in TASKS}   # task -> counts of terminal boundaries
    tp_at = {t: defaultdict(int) for t in TASKS}      # task -> {threshold: tp}
    site_tp = {t: defaultdict(int) for t in TASKS}    # task -> {(end, tol): tp}
    site_n = {t: defaultdict(int) for t in TASKS}     # task -> {("true"/"pred", end): n}
    res = {t: defaultdict(int) for t in TASKS}        # residue-level counts, per task
    res_any = defaultdict(int)                        # "in any segment" -- see below
    seen_prot = set()
    struct = {t: defaultdict(int) for t in TASKS}
    per_segment = {}                                  # (name, task, ts, te) -> dict

    for rec in records:
        name = rec["name"]
        L = lengths.get(name)
        # The combined residue view is a SINGLE binary task ("this residue is inside
        # some annotated segment"), not the sum of the two per-task confusion
        # matrices: summing them counts every non-segment residue as a true
        # negative once per task, i.e. twice, which inflates MCC.
        T_any, P_any = set(), set()
        for task in TASKS:
            for a, b in rec[task]["true"]:
                T_any.update(range(int(a), int(b) + 1))
            for a, b in rec[task]["pred"]:
                P_any.update(range(int(a), int(b) + 1))
        if name not in seen_prot:
            seen_prot.add(name)
            tp_any = len(T_any & P_any)
            res_any["tp"] += tp_any
            res_any["fp"] += len(P_any) - tp_any
            res_any["fn"] += len(T_any) - tp_any
            if L is not None:
                res_any["tn"] += max(0, L - len(T_any | P_any))
                res_any["n_prot"] += 1
        for task in TASKS:
            true = [(int(a), int(b)) for a, b in rec[task]["true"]]
            pred = [(int(a), int(b)) for a, b in rec[task]["pred"]]
            if not true and not pred:
                # No segment of this type, annotated or predicted: it contributes
                # nothing anywhere except true negatives, and dropping those was
                # deflating MCC by ~0.019 (45% of the negative residues).
                if L is not None:
                    res[task]["tn"] += L
                    res[task]["n_prot_with_len"] += 1
                continue
            acc[task]["n_true"] += len(true)
            acc[task]["n_pred"] += len(pred)

            # ---- IoU matching -------------------------------------------------
            matches = greedy_iou_match(true, pred)
            matched_iou = {i: (j, v) for i, j, v in matches}
            for i, j, v in matches:
                ts, te = true[i]
                ps, pe = pred[j]
                ious[task].append(v)
                dstart[task].append(ps - ts)
                dend[task].append(pe - te)
                lenratio[task].append((pe - ps + 1) / (te - ts + 1))
                b = length_bin(te - ts + 1)
                if b:
                    bin_iou[(task, b)].append(v)
                # A boundary that IS the chain terminus cannot be overshot, so its
                # error is half-truncated by construction; ~29% of annotated peptide
                # ends are the C-terminus and they are placed almost for free.
                # The filter is PER END: a segment whose end is the C-terminus still
                # has an informative start, and dropping it from the start
                # distribution as well would compare two different populations --
                # which is the very confound this control exists to remove.
                if ts == 1:
                    anchored[task]["start_at_1"] += 1
                else:
                    dstart_in[task].append(ps - ts)
                if L is not None and te == L:
                    anchored[task]["end_at_terminus"] += 1
                else:
                    dend_in[task].append(pe - te)
                for th in IOU_THRESHOLDS:
                    if v >= th:
                        tp_at[task][th] += 1
            for i, (ts, te) in enumerate(true):
                j, v = matched_iou.get(i, (None, 0.0))
                b = length_bin(te - ts + 1)
                if b:
                    # Every true segment enters its bin, matched or not. Scoring a
                    # bin only over the segments a model happened to find compares
                    # each model on its own, self-selected subset.
                    bin_true[b] += 1
                    bin_best[b] += max((iou((ts, te), q) for q in pred), default=0.0)
                per_segment[(name, task, ts, te)] = {
                    "iou": v,
                    "dstart": (pred[j][0] - ts) if j is not None else None,
                    "dend": (pred[j][1] - te) if j is not None else None,
                    # min over preds of max(|dstart|, |dend|): the quantity the
                    # +-tau criterion thresholds, for the detection gates below.
                    "err": min((max(abs(q[0] - ts), abs(q[1] - te)) for q in pred),
                               default=CAP_ERR),
                }
            acc[task]["iou_sum_over_true"] += sum(v for _, _, v in matches)

            # ---- cleavage-site events ----------------------------------------
            for end, idx in (("start", 0), ("end", 1)):
                ts_sites = [s[idx] for s in true]
                ps_sites = [s[idx] for s in pred]
                site_n[task][("true", end)] += len(ts_sites)
                site_n[task][("pred", end)] += len(ps_sites)
                for tol in SITE_TOLERANCES:
                    site_tp[task][(end, tol)] += greedy_site_match(ts_sites, ps_sites, tol)

            # ---- residue level -------------------------------------------------
            T = set()
            for s, e in true:
                T.update(range(s, e + 1))
            P = set()
            for s, e in pred:
                P.update(range(s, e + 1))
            rtp = len(T & P)
            res[task]["tp"] += rtp
            res[task]["fp"] += len(P) - rtp
            res[task]["fn"] += len(T) - rtp
            if L is not None:
                res[task]["tn"] += max(0, L - len(T | P))
                res[task]["n_prot_with_len"] += 1

            # ---- structure ------------------------------------------------------
            for s, e in true:
                overlapping = sum(1 for ps, pe in pred if min(e, pe) >= max(s, ps))
                if overlapping == 0:
                    struct[task]["true_missed"] += 1
                elif overlapping > 1:
                    struct[task]["true_split"] += 1
            for ps, pe in pred:
                overlapping = sum(1 for s, e in true if min(e, pe) >= max(s, ps))
                if overlapping == 0:
                    struct[task]["pred_spurious"] += 1
                elif overlapping > 1:
                    struct[task]["pred_merged"] += 1

    # ------------------------------------------------------------- reduce ---
    out = {}

    def put(task, key, value):
        out[f"{task}_{key}"] = value

    combined = defaultdict(float)
    for task in TASKS:
        n_true = acc[task]["n_true"]
        n_pred = acc[task]["n_pred"]
        combined["n_true"] += n_true
        combined["n_pred"] += n_pred
        for th in IOU_THRESHOLDS:
            tp = tp_at[task][th]
            combined[f"tp{th}"] += tp
            p, r, f = prf(tp, n_true - tp, n_pred - tp)
            put(task, f"f1_iou{th}", f)
            put(task, f"precision_iou{th}", p)
            put(task, f"recall_iou{th}", r)
        put(task, "mean_iou_matched", float(np.mean(ious[task])) if ious[task] else float("nan"))
        put(task, "mean_iou_over_true",
            acc[task]["iou_sum_over_true"] / n_true if n_true else float("nan"))
        combined["iou_sum"] += acc[task]["iou_sum_over_true"]
        combined["n_matched"] += len(ious[task])
        combined["iou_matched_sum"] += float(np.sum(ious[task])) if ious[task] else 0.0

        for name, arr in (("dstart", dstart[task]), ("dend", dend[task])):
            a = np.asarray(arr, dtype=float)
            put(task, f"{name}_signed_mean", float(a.mean()) if a.size else float("nan"))
            put(task, f"{name}_abs_mean", float(np.abs(a).mean()) if a.size else float("nan"))
            put(task, f"{name}_abs_median", float(np.median(np.abs(a))) if a.size else float("nan"))
            put(task, f"{name}_exact", float((a == 0).mean()) if a.size else float("nan"))
            put(task, f"{name}_within1", float((np.abs(a) <= 1).mean()) if a.size else float("nan"))
        both = np.asarray([1.0 if (s == 0 and e == 0) else 0.0
                           for s, e in zip(dstart[task], dend[task])])
        put(task, "both_ends_exact", float(both.mean()) if both.size else float("nan"))
        # Interior-only copies: the start-vs-end contrast is only meaningful on
        # boundaries that are not the chain terminus.
        for name_, arr in (("dstart_interior", dstart_in[task]),
                           ("dend_interior", dend_in[task])):
            a = np.asarray(arr, dtype=float)
            put(task, f"{name_}_abs_mean", float(np.abs(a).mean()) if a.size else float("nan"))
            put(task, f"{name_}_exact", float((a == 0).mean()) if a.size else float("nan"))
        put(task, "n_matched_interior", len(dstart_in[task]))
        nm = len(ious[task])
        put(task, "frac_start_at_residue_1",
            anchored[task]["start_at_1"] / nm if nm else float("nan"))
        put(task, "frac_end_at_c_terminus",
            anchored[task]["end_at_terminus"] / nm if nm else float("nan"))
        lr = np.asarray(lenratio[task], dtype=float)
        put(task, "length_ratio_mean", float(lr.mean()) if lr.size else float("nan"))

        for end in ("start", "end"):
            nt = site_n[task][("true", end)]
            npd = site_n[task][("pred", end)]
            for tol in SITE_TOLERANCES:
                tp = site_tp[task][(end, tol)]
                combined[f"site_tp_{end}_{tol}"] += tp
                p, r, f = prf(tp, nt - tp, npd - tp)
                put(task, f"site_{end}_f1_tol{tol}", f)
                put(task, f"site_{end}_recall_tol{tol}", r)
            combined[f"site_n_true_{end}"] += nt
            combined[f"site_n_pred_{end}"] += npd

        rtp, rfp, rfn, rtn = (res[task]["tp"], res[task]["fp"],
                              res[task]["fn"], res[task]["tn"])
        p, r, f = prf(rtp, rfn, rfp)
        put(task, "residue_precision", p)
        put(task, "residue_recall", r)
        put(task, "residue_f1", f)
        put(task, "residue_jaccard", rtp / (rtp + rfp + rfn) if (rtp + rfp + rfn) else float("nan"))
        put(task, "residue_mcc", mcc(rtp, rtn, rfp, rfn))
        put(task, "residue_n_total", rtp + rfp + rfn + rtn)
        put(task, "residue_n_proteins_with_length", res[task]["n_prot_with_len"])
        for k in ("tp", "fp", "fn", "tn"):
            combined[f"res_{k}"] += res[task][k]

        put(task, "split_rate", struct[task]["true_split"] / n_true if n_true else float("nan"))
        put(task, "merge_rate", struct[task]["pred_merged"] / n_pred if n_pred else float("nan"))
        put(task, "pred_per_true", n_pred / n_true if n_true else float("nan"))
        put(task, "n_true", n_true)
        put(task, "n_pred", n_pred)

    # combined ("all") view
    for th in IOU_THRESHOLDS:
        tp = combined[f"tp{th}"]
        p, r, f = prf(tp, combined["n_true"] - tp, combined["n_pred"] - tp)
        out[f"all_f1_iou{th}"] = f
        out[f"all_precision_iou{th}"] = p
        out[f"all_recall_iou{th}"] = r
    out["all_mean_iou_matched"] = (combined["iou_matched_sum"] / combined["n_matched"]
                                   if combined["n_matched"] else float("nan"))
    out["all_mean_iou_over_true"] = (combined["iou_sum"] / combined["n_true"]
                                     if combined["n_true"] else float("nan"))
    for end in ("start", "end"):
        for tol in SITE_TOLERANCES:
            tp = combined[f"site_tp_{end}_{tol}"]
            p, r, f = prf(tp, combined[f"site_n_true_{end}"] - tp,
                          combined[f"site_n_pred_{end}"] - tp)
            out[f"all_site_{end}_f1_tol{tol}"] = f
    # Two residue views, kept apart because they answer different questions.
    #
    # `all_*` sums the two per-task confusion matrices, so calling a peptide
    # residue a propeptide residue is an error. Precision/recall/F1/Jaccard do not
    # touch TN and are exact under this reading. MCC is deliberately NOT reported
    # here: TN would be each protein's non-segment residues counted once per task,
    # i.e. twice, which inflates it.
    stp, sfp, sfn = combined["res_tp"], combined["res_fp"], combined["res_fn"]
    p, r, f = prf(stp, sfn, sfp)
    out["all_residue_precision"] = p
    out["all_residue_recall"] = r
    out["all_residue_f1"] = f
    out["all_residue_jaccard"] = stp / (stp + sfp + sfn) if (stp + sfp + sfn) else float("nan")
    #
    # `any_*` is one binary task -- "this residue is inside some annotated
    # segment" -- with every residue counted exactly once, so TN is well defined
    # and MCC means what it says. Type confusion is invisible here by design.
    rtp, rfp, rfn, rtn = (res_any["tp"], res_any["fp"], res_any["fn"], res_any["tn"])
    p, r, f = prf(rtp, rfn, rfp)
    out["any_residue_precision"] = p
    out["any_residue_recall"] = r
    out["any_residue_f1"] = f
    out["any_residue_jaccard"] = rtp / (rtp + rfp + rfn) if (rtp + rfp + rfn) else float("nan")
    out["any_residue_mcc"] = mcc(rtp, rtn, rfp, rfn)
    out["any_residue_n_total"] = rtp + rfp + rfn + rtn
    out["any_residue_n_proteins"] = res_any["n_prot"]
    out["all_n_true"] = combined["n_true"]
    out["all_n_pred"] = combined["n_pred"]

    # Length-stratified IoU, pooled over both segment types.
    #
    # `iou_len<bin>` averages the best available IoU over EVERY true segment in
    # the bin, scoring an undetected segment 0. That is the number to quote: the
    # matched-only variant next to it conditions on detection, and the match rate
    # is itself a model property (in the 45-50 bin the boundary head matches 37
    # of 87 true segments against the baseline's 48), so comparing matched-only
    # means compares five different, self-selected subsets.
    per_bin = defaultdict(list)
    for (task, b), vals in bin_iou.items():
        per_bin[b].extend(vals)
    for lo, hi in LENGTH_BINS:
        b = f"{lo}-{hi}"
        nt = bin_true[b]
        out[f"iou_len{b}"] = bin_best[b] / nt if nt else float("nan")
        out[f"iou_len{b}_matched_only"] = (float(np.mean(per_bin[b]))
                                           if per_bin[b] else float("nan"))
        out[f"n_true_len{b}"] = nt
        out[f"n_matched_len{b}"] = len(per_bin[b])
        out[f"match_rate_len{b}"] = len(per_bin[b]) / nt if nt else float("nan")

    return out, per_segment


# ------------------------------------------------------------- aggregation ---

def load_cells(model):
    root = Path("runs") / model
    cells = {}
    for d in sorted(root.glob("outer*_inner*")):
        f = d / "segments.json.gz"
        if not f.exists():
            continue
        outer = int(d.name.split("outer")[1].split("_")[0])
        inner = int(d.name.split("inner")[1])
        with gzip.open(f, "rt") as fh:
            cells[(outer, inner)] = json.load(fh)
    return cells


def aggregate(rows):
    """rows: list of dicts carrying 'outer'. -> {metric: {mean, std, per_outer}}"""
    df = pd.DataFrame(rows)
    per_outer = df.groupby("outer").mean(numeric_only=True)
    out = {}
    for col in per_outer.columns:
        if col == "inner":
            continue
        vals = per_outer[col]
        out[col] = {
            "mean": round(float(vals.mean()), 6),
            "std": round(float(vals.std()), 6),
            "per_outer": [round(float(v), 6) for v in vals],
        }
    return out


# Gates for the paired comparison, from loosest to strictest. The loose one is
# NOT a detection control: admitting a pair on a single residue of overlap lets
# segments that one model essentially misses dominate the mean (for the boundary
# head, 78% of its loose-gate delta comes from pairs where one of the two models
# is outside +-3, which is the very detection gain the headline F1 already
# counts). Only the strict gates isolate localisation.
GATES = {
    # name          predicate on (baseline record, variant record)
    "overlap":  lambda b, v: b["iou"] > 0 and v["iou"] > 0,
    "iou50":    lambda b, v: b["iou"] >= 0.5 and v["iou"] >= 0.5,
    "tol3":     lambda b, v: b["err"] <= 3 and v["err"] <= 3,
}
PRIMARY_GATE = "iou50"   # strict, and does not smuggle the +-3 window back in


def paired_vs_baseline(base_seg, var_seg, gate=PRIMARY_GATE):
    """Paired localisation delta on the true segments both models place well.

    `gate` decides what "both models found it" means; see GATES. The pairing is
    by (protein, task, true_start, true_end) inside one cell, and the 4 inner
    cells of an outer fold share a test partition, so pairing is exact.
    """
    keep = GATES[gate]
    rows = []
    for (outer, inner), bseg in base_seg.items():
        vseg = var_seg.get((outer, inner))
        if vseg is None:
            continue
        d_iou, d_start, d_end, tighter, looser = [], [], [], 0, 0
        for key, b in bseg.items():
            v = vseg.get(key)
            # Both models must have an assigned prediction for the pair to have a
            # displacement at all; err <= 3 alone can hold while the one-to-one
            # match went elsewhere, so the gates sit on top of that requirement.
            if v is None or b["dstart"] is None or v["dstart"] is None:
                continue
            if not keep(b, v):
                continue
            d_iou.append(v["iou"] - b["iou"])
            d_start.append(abs(v["dstart"]) - abs(b["dstart"]))
            d_end.append(abs(v["dend"]) - abs(b["dend"]))
            tighter += v["iou"] > b["iou"]
            looser += v["iou"] < b["iou"]
        if not d_iou:
            continue
        n = len(d_iou)
        rows.append({
            "outer": outer, "inner": inner, "n_paired": n,
            "d_iou_mean": float(np.mean(d_iou)),
            "d_iou_median": float(np.median(d_iou)),
            "d_abs_dstart_mean": float(np.mean(d_start)),
            "d_abs_dend_mean": float(np.mean(d_end)),
            "frac_tighter": tighter / n,
            "frac_looser": looser / n,
        })
    if not rows:
        return {}
    agg = aggregate(rows)
    # A mean +- std over 5 folds hides how consistent the sign is; a 5/5 sign
    # count is p = 0.031 one-sided where 1.4 sigma reads as noise.
    # "Improved" means a larger IoU but a SMALLER absolute displacement, so the
    # good direction is not the same sign for the three metrics. Reporting a bare
    # "n_outer_positive" for all of them inverts two of the three.
    GOOD_IF_POSITIVE = {"d_iou_mean": True,
                        "d_abs_dstart_mean": False,
                        "d_abs_dend_mean": False}
    for k, good_pos in GOOD_IF_POSITIVE.items():
        if k in agg:
            po = agg[k]["per_outer"]
            agg[k]["n_outer_improved"] = int(sum((x > 0) == good_pos for x in po))
            agg[k]["n_outer"] = len(po)
    return agg


# --------------------------------------------------------------------- cli ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--baseline", default="5cv_baseline_esm2_fp32")
    ap.add_argument("--out", default="analysis/metrics/segment_quality_cv.json")
    ap.add_argument("--sequences", default="data/uniprot_2026/labeled_sequences.csv")
    args = ap.parse_args()

    seqs = pd.read_csv(args.sequences, usecols=["protein_id", "sequence"])
    lengths = {p: len(s) for p, s in zip(seqs.protein_id, seqs.sequence)}
    print(f"[data] {len(lengths)} protein lengths")

    # The dumps do not agree on the protein set: the ESM-C pipeline carries 123
    # long proteins (min 1026 aa) that the ESM-2 one drops, and lacks 21 others.
    # Unpaired tables would then compare rows scored on different test sets, so
    # every model is restricted to the proteins all of them have in that cell.
    cellsets = {}
    for model in args.models:
        cells = load_cells(model)
        if not cells:
            print(f"[skip] {model}: no segment dumps")
            continue
        cellsets[model] = cells
    if not cellsets:
        print("nothing to score")
        return 1
    common_cells = sorted(set.intersection(*(set(c) for c in cellsets.values())))
    shared = {}
    for cell in common_cells:
        names = None
        for model in cellsets:
            got = {r["name"] for r in cellsets[model][cell]}
            names = got if names is None else (names & got)
        shared[cell] = names
    for model in cellsets:
        dropped = sum(len({r["name"] for r in cellsets[model][c]} - shared[c])
                      for c in common_cells)
        total = sum(len(cellsets[model][c]) for c in common_cells)
        if dropped:
            print(f"[set] {model:24s} {dropped} of {total} protein-cells dropped "
                  f"to the set shared by all models")
    print(f"[set] scoring {sum(len(v) for v in shared.values())} protein-cells "
          f"across {len(common_cells)} cells, identical for every model")

    summary, segments = {}, {}
    for model, cells in cellsets.items():
        rows, segments[model] = [], {}
        for cell in common_cells:
            m, per_seg = score_cell(cells[cell], lengths, keep=shared[cell])
            m.update({"outer": cell[0], "inner": cell[1]})
            rows.append(m)
            segments[model][cell] = per_seg
        summary[model] = {"n_cells": len(rows), "metrics": aggregate(rows)}
        a = summary[model]["metrics"]
        print(f"[ok] {model:24s} {len(rows)} cells  "
              f"IoU/true={a['all_mean_iou_over_true']['mean']:.4f}  "
              f"F1@0.5={a['all_f1_iou0.5']['mean']:.4f}  "
              f"F1@0.9={a['all_f1_iou0.9']['mean']:.4f}  "
              f"resF1={a['all_residue_f1']['mean']:.4f}")

    base = args.baseline
    if base in segments:
        for model in segments:
            if model == base:
                continue
            # All three gates are reported: the loose one is what a naive
            # "both models found it" filter gives, and the gap between it and
            # the strict ones is exactly the detection component.
            summary[model]["paired_vs_baseline"] = {
                g: paired_vs_baseline(segments[base], segments[model], gate=g)
                for g in GATES}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"baseline": base,
                               "iou_thresholds": list(IOU_THRESHOLDS),
                               "paired_gates": list(GATES),
                               "primary_gate": PRIMARY_GATE,
                               "models": summary}, indent=2) + "\n")
    print(f"[json] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
