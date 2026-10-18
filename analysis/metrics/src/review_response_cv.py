#!/usr/bin/env python3
"""Paired effects with intervals and their breakdowns, computed without a GPU.

Everything is read from runs/<model>/outer{o}_inner{k}/segments.json.gz (the span dumps
of the 5x4 nested CV) and from data/uniprot_2026/processed_data/features_master.parquet
(annotation evidence codes). Nothing is re-inferred.

Sections of the output JSON
---------------------------
  check        every Table 1 F1 (+-3 and exact) rebuilt from the dumps, with the paper value.
  paired       paired effect of each addition against its own base, at +-3 and at exact
               match, for both segment types pooled and for peptides / propeptides alone:
               mean of the five paired outer-fold differences (the paper's estimator), the
               sample SD across folds, the number of positive folds, and two bootstrap
               intervals (protein resampling; genus-cluster resampling).
  untouched    the same effects restricted to test proteins that were never scored in the
               screening comparison (folds 2 and 5 of the seven-fold screening split).
  evidence     recall of annotated segments split by the evidence code of the annotation.
  bylength     paired effects per true-segment-length bin, denominators printed.
  matcher      the same paired effects under the original and the corrected matcher, from the
               per-cell corrected_metrics.json files that already hold both readings.

Estimator, as in the paper: F1 is pooled inside a cell, averaged over the four inner cells of
an outer fold (they share a test set), then averaged over the five outer folds.

Bootstrap: within each outer fold, resample units (proteins, or genera as a proxy for
homology clusters) with replacement, the SAME resample for every configuration and every inner
cell, so the interval is for the paired difference. It covers the sampling of test proteins
only: not the choice of training proteins, and not the training seed.

Usage:
  python analysis/metrics/src/review_response_cv.py [--boot 4000] [--out PATH] [--runs-dir DIR] [--data-dir DIR]
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

RUNS = REPO / "runs"                      # overridden by --runs-dir
DATA = REPO / "data/uniprot_2026"         # overridden by --data-dir

from analysis.errors.src.error_analysis import match_protein, _group  # noqa: E402

TASKS = ("peptides", "propeptides")
TAUS = (0, 3)
FOLDS = range(5)
CONFIGS = {
    "ESM-2": {"base": "5cv_baseline_esm2_fp32", "head": "5cv_esm2_boundary",
              "adapter": "5cv_esm2_adapter_only", "both": "5cv_esm2_full"},
    "ESM-C 6B": {"base": "5cv_esmc6b_plain", "head": "5cv_esmc6b_boundary",
                 "adapter": "5cv_esmc6b_adapter_only", "both": "5cv_esmc6b_full"},
}
ADDITIONS = ("head", "adapter", "both")
# Table 1 of the paper: (F1 at +-3, F1 at exact match).
PAPER = {
    ("ESM-2", "base"): (0.576, 0.354), ("ESM-2", "head"): (0.597, 0.384),
    ("ESM-2", "adapter"): (0.602, 0.396), ("ESM-2", "both"): (0.630, 0.432),
    ("ESM-C 6B", "base"): (0.588, 0.371), ("ESM-C 6B", "head"): (0.641, 0.429),
    ("ESM-C 6B", "adapter"): (0.604, 0.395), ("ESM-C 6B", "both"): (0.666, 0.466),
}
EDGES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 51]
BINS = [f"{EDGES[i]}-{EDGES[i + 1] - 1}" for i in range(len(EDGES) - 1)]
EVIDENCE_CLASSES = ("experimental", "similarity", "other inferred", "none listed")
SCREEN_FOLDS = (2, 5)  # model_select and sealed test of the seven-fold screening split


def cells(o):
    return [k for k in FOLDS if k != o]


def length_bin(n):
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= n < EDGES[i + 1]:
            return i
    return None


def evidence_class(codes):
    codes = set(codes) if codes is not None else set()
    if "ECO:0000269" in codes:
        return 0
    if "ECO:0000250" in codes:
        return 1
    if codes:
        return 2
    return 3


def load_evidence():
    f = pd.read_parquet(DATA / "processed_data/features_master.parquet")
    v = f[f.is_valid_for_training]
    return {(a, l, int(s), int(e)): evidence_class(c)
            for a, l, s, e, c in zip(v.accession, v.label, v.start, v.end, v.evidence_codes)}


def load_genus():
    d = pd.read_csv(DATA / "labeled_sequences.csv", usecols=["protein_id", "organism"])
    return {a: (str(o).split()[0] if isinstance(o, str) and o.split() else "unknown")
            for a, o in zip(d.protein_id, d.organism)}


def screening_proteins():
    """Proteins scored in the screening comparison: union over the seven-fold files."""
    out = set()
    for p in sorted(DATA.glob("graphpart_assignments.*covered.csv")):
        d = pd.read_csv(p)
        out |= set(d.loc[d["cluster"].astype(int).isin(SCREEN_FOLDS), "AC"])
    return out


def score_cell(path, evidence):
    """Per-protein counts plus the per-class and per-length aggregates of one cell."""
    recs = json.load(gzip.open(path))
    recs.sort(key=lambda r: r["name"])
    names = [r["name"] for r in recs]
    counts = np.zeros((len(recs), len(TAUS), len(TASKS), 3), dtype=np.int32)
    ev = np.zeros((len(EVIDENCE_CLASSES), len(TAUS), 2), dtype=np.int64)  # n groups, n matched
    # Same definition as bylength_cv.py (and Fig. bylength): a true group is a TP or FN in the bin
    # of its own length, an unmatched prediction is a FP in the bin of its predicted length.
    ln = np.zeros((len(BINS), 3), dtype=np.int64)  # tp, fn, fp
    for i, r in enumerate(recs):
        for t, (task, label) in enumerate((("peptides", "PEPTIDE"), ("propeptides", "PROPEP"))):
            true = [tuple(map(int, x)) for x in r[task]["true"]]
            pred = [tuple(map(int, x)) for x in r[task]["pred"]]
            groups = _group(true) if true else []
            for a, tau in enumerate(TAUS):
                grec, pm = match_protein(true, pred, tau)
                tp = sum(g["matched"] for g in grec)
                counts[i, a, t] = (tp, len(grec) - tp, sum(not m for m in pm))
                for g, rec in zip(groups, grec):
                    # A group takes the strongest evidence among its members.
                    c = min(evidence.get((r["name"], label, s, e), 3) for s, e in g)
                    ev[c, a, 0] += 1
                    ev[c, a, 1] += int(rec["matched"])
                if tau == 3:
                    for g, rec in zip(groups, grec):
                        b = length_bin(rec["length"])
                        if b is not None:
                            ln[b, 0 if rec["matched"] else 1] += 1
                    for (ps, pe), m in zip(pred, pm):
                        b = length_bin(pe - ps + 1)
                        if b is not None and not m:
                            ln[b, 2] += 1
    return names, counts, ev, ln


def f1_from(tp, fn, fp):
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    return np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)


def scope_f1(c):
    """c[..., tau, task, 3] summed over proteins -> F1[..., tau, scope] for (all, pep, pro)."""
    tp, fn, fp = c[..., 0], c[..., 1], c[..., 2]
    allf = f1_from(tp.sum(-1), fn.sum(-1), fp.sum(-1))
    return np.stack([allf, f1_from(tp[..., 0], fn[..., 0], fp[..., 0]),
                     f1_from(tp[..., 1], fn[..., 1], fp[..., 1])], axis=-1)


def main():
    global RUNS, DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out", default=str(REPO / "analysis/metrics/review_response_cv.json"))
    ap.add_argument("--runs-dir", default=str(RUNS), help="holds <run>/outer*_inner*/segments.json.gz and original_metrics.json")
    ap.add_argument("--data-dir", default=str(DATA), help="the processed 2026 data: labeled_sequences.csv, graphpart_assignments*.csv, processed_data/features_master.parquet")
    args = ap.parse_args()
    RUNS, DATA = Path(args.runs_dir), Path(args.data_dir)

    evidence, genus, screen = load_evidence(), load_genus(), screening_proteins()
    data = {}  # (emb, cfg) -> dict(o -> (names, counts[4,P,tau,task,3], ev[4,...], ln[4,...]))
    for emb, cfgs in CONFIGS.items():
        for cfg, run in cfgs.items():
            per = {}
            for o in FOLDS:
                parts = [score_cell(RUNS / run / f"outer{o}_inner{k}" / "segments.json.gz", evidence)
                         for k in cells(o)]
                names = parts[0][0]
                assert all(p[0] == names for p in parts), (run, o, "inner cells disagree on proteins")
                per[o] = (names, np.stack([p[1] for p in parts]), np.stack([p[2] for p in parts]),
                          np.stack([p[3] for p in parts]))
            data[(emb, cfg)] = per
        for cfg in cfgs:
            for o in FOLDS:
                assert data[(emb, cfg)][o][0] == data[(emb, "base")][o][0], (emb, cfg, o, "protein sets differ")

    def fold_f1(emb, cfg, mask_fn=None):
        """[fold, tau, scope] mean-over-inner-cells F1."""
        out = np.zeros((5, len(TAUS), 3))
        for o in FOLDS:
            names, c, _, _ = data[(emb, cfg)][o]
            m = np.ones(len(names), bool) if mask_fn is None else np.array([mask_fn(n) for n in names])
            out[o] = scope_f1(c[:, m].sum(1)).mean(0)
        return out

    res = {"check": [], "paired": {}, "untouched": {}, "evidence": {}, "bylength": {}, "matcher": {}}

    # ---- check against Table 1 ------------------------------------------------------------
    worst = 0.0
    for (emb, cfg), (p3, p0) in PAPER.items():
        f = fold_f1(emb, cfg).mean(0)
        got3, got0 = f[TAUS.index(3), 0], f[TAUS.index(0), 0]
        worst = max(worst, abs(got3 - p3), abs(got0 - p0))
        res["check"].append({"embedding": emb, "config": cfg, "f1_pm3": got3, "paper_pm3": p3,
                             "f1_exact": got0, "paper_exact": p0})
    res["check_max_abs_diff"] = worst
    print(f"Table 1 rebuilt from dumps, worst |diff| = {worst:.4f} (rounding is 0.0005)")

    # ---- bootstrap machinery --------------------------------------------------------------
    rng = np.random.default_rng(20261017)

    def boot_delta(emb, cfg, unit, mask_fn=None):
        """[B, tau, scope] bootstrap distribution of F1(cfg) - F1(base), mean over folds."""
        acc = None
        for o in FOLDS:
            names, cb, _, _ = data[(emb, "base")][o]
            _, cv, _, _ = data[(emb, cfg)][o]
            keep = np.ones(len(names), bool) if mask_fn is None else np.array([mask_fn(n) for n in names])
            keys = [n if unit == "protein" else genus.get(n, "unknown") for n in names]
            uniq = {}
            ids = np.array([uniq.setdefault(k, len(uniq)) for k in keys])
            K = len(uniq)
            # Resample units among those present in the evaluated subset.
            present = np.unique(ids[keep])
            Kp = len(present)
            remap = {u: j for j, u in enumerate(present)}
            ucounts = {}
            for tag, c in (("b", cb), ("v", cv)):
                u = np.zeros((4, Kp, len(TAUS), len(TASKS), 3))
                for j in np.nonzero(keep)[0]:
                    u[:, remap[ids[j]]] += c[:, j]
                ucounts[tag] = u.reshape(4, Kp, -1)
            M = rng.multinomial(Kp, np.full(Kp, 1 / Kp), size=args.boot).astype(float)
            diffs = []
            for tag in ("v", "b"):
                tot = np.einsum("bk,ckf->bcf", M, ucounts[tag]).reshape(args.boot, 4, len(TAUS), len(TASKS), 3)
                diffs.append(scope_f1(tot).mean(1))  # mean over inner cells
            d = diffs[0] - diffs[1]
            acc = d if acc is None else acc + d
        return acc / 5

    # NOTE: the multinomial draw must be shared between base and variant for the pairing,
    # which holds because both are resampled through the same M inside one call.

    def effect_row(emb, cfg, mask_fn=None, boot=True):
        fv, fb = fold_f1(emb, cfg, mask_fn), fold_f1(emb, "base", mask_fn)
        d = fv - fb  # [fold, tau, scope]
        row = {}
        for a, tau in enumerate(TAUS):
            for s, scope in enumerate(("all", "peptides", "propeptides")):
                x = d[:, a, s]
                row[f"tau{tau}_{scope}"] = {
                    "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                    "pos_folds": int((x > 0).sum()), "per_fold": [round(float(v), 4) for v in x],
                    "base": float(fb[:, a, s].mean()), "variant": float(fv[:, a, s].mean())}
        if boot:
            for unit in ("protein", "genus"):
                b = boot_delta(emb, cfg, unit, mask_fn)
                for a, tau in enumerate(TAUS):
                    for s, scope in enumerate(("all", "peptides", "propeptides")):
                        lo, hi = np.percentile(b[:, a, s], [2.5, 97.5])
                        row[f"tau{tau}_{scope}"][f"ci95_{unit}"] = [float(lo), float(hi)]
        return row

    for emb in CONFIGS:
        res["paired"][emb] = {cfg: effect_row(emb, cfg) for cfg in ADDITIONS}
    print("paired effects done")

    # ---- untouched proteins ---------------------------------------------------------------
    for emb in CONFIGS:
        names_all = [n for o in FOLDS for n in data[(emb, "base")][o][0]]
        unseen = [n for n in names_all if n not in screen]
        res["untouched"][emb] = {
            "n_test_proteins": len(names_all), "n_unseen": len(unseen),
            "n_screened": len(names_all) - len(unseen),
            **{cfg: effect_row(emb, cfg, mask_fn=lambda n: n not in screen) for cfg in ADDITIONS}}
    print("untouched subset done")

    # ---- evidence quality -----------------------------------------------------------------
    for emb in CONFIGS:
        out = {}
        for ci, cname in enumerate(EVIDENCE_CLASSES):
            row = {"n_groups_per_fold": float(np.mean([
                data[(emb, "base")][o][2][:, ci, 0, 0].mean() for o in FOLDS]))}
            for a, tau in enumerate(TAUS):
                rec = {}
                for cfg in CONFIGS[emb]:
                    r = np.array([(data[(emb, cfg)][o][2][:, ci, a, 1] /
                                   np.maximum(data[(emb, cfg)][o][2][:, ci, a, 0], 1)).mean() for o in FOLDS])
                    rec[cfg] = r
                row[f"tau{tau}"] = {"recall_base": float(rec["base"].mean())}
                for cfg in ADDITIONS:
                    d = rec[cfg] - rec["base"]
                    row[f"tau{tau}"][cfg] = {"recall": float(rec[cfg].mean()), "delta": float(d.mean()),
                                             "sd": float(d.std(ddof=1)), "pos_folds": int((d > 0).sum())}
            out[cname] = row
        res["evidence"][emb] = out
    print("evidence done")

    # ---- by length ------------------------------------------------------------------------
    for emb in CONFIGS:
        out = {}
        for bi, b in enumerate(BINS):
            per_cfg = {}
            for cfg in CONFIGS[emb]:
                f = []
                for o in FOLDS:
                    ln = data[(emb, cfg)][o][3][:, bi]  # [cell, 3]
                    den = 2 * ln[:, 0] + ln[:, 1] + ln[:, 2]
                    f.append((2 * ln[:, 0] / np.maximum(den, 1)).mean())
                per_cfg[cfg] = np.array(f)
            row = {"n_true_per_fold": float(np.mean([(data[(emb, "base")][o][3][:, bi, 0] + data[(emb, "base")][o][3][:, bi, 1]).mean() for o in FOLDS])),
                   "base": float(per_cfg["base"].mean())}
            for cfg in ADDITIONS:
                d = per_cfg[cfg] - per_cfg["base"]
                row[cfg] = {"f1": float(per_cfg[cfg].mean()), "delta": float(d.mean()),
                            "sd": float(d.std(ddof=1)), "pos_folds": int((d > 0).sum())}
            out[b] = row
        res["bylength"][emb] = out
    print("bylength done")

    # ---- matcher: original vs corrected --------------------------------------------------
    # Original reading: original_metrics.json of every cell (the upstream matcher on the same
    # decoded segments). Corrected reading: rebuilt above from the span dumps.
    for emb, cfgs in CONFIGS.items():
        fold = {}
        for cfg, run in cfgs.items():
            o_f = np.zeros(5)
            for o in FOLDS:
                o_f[o] = np.mean([json.load(open(RUNS / run / f"outer{o}_inner{k}" / "original_metrics.json"))
                                  ["tol3_all_f1"] for k in cells(o)])
            fold[cfg] = (o_f, fold_f1(emb, cfg)[:, TAUS.index(3), 0])
        row = {"base": {"original": float(fold["base"][0].mean()), "corrected": float(fold["base"][1].mean())}}
        for cfg in ADDITIONS:
            do, dc = fold[cfg][0] - fold["base"][0], fold[cfg][1] - fold["base"][1]
            row[cfg] = {"original_mean": float(do.mean()), "original_sd": float(do.std(ddof=1)),
                        "original_pos": int((do > 0).sum()),
                        "corrected_mean": float(dc.mean()), "corrected_sd": float(dc.std(ddof=1)),
                        "corrected_pos": int((dc > 0).sum())}
        res["matcher"][emb] = row
    print("matcher done")

    # ---- smallest attainable two-sided p of an exact test on five folds ---------------------
    from scipy.stats import wilcoxon
    res["min_p_wilcoxon_n5"] = float(wilcoxon([1, 2, 3, 4, 5]).pvalue)

    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
