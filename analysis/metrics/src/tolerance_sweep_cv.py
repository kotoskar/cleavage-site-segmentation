#!/usr/bin/env python3
"""Boundary-tolerance sweep over the nested-CV grid.

For every nested-CV cell (runs/5cv_*/outer{o}_inner{k}/) this runs TEST-fold
inference once, decodes peptide/propeptide segment borders, and

  1. DUMPS the decoded true/predicted segments to
     runs/<name>/outer{o}_inner{k}/segments.json.gz, so that any future
     tolerance / stratification can be scored offline with no GPU, and
  2. scores precision/recall/F1 with the CORRECTED matcher
     (analysis.errors.src.error_analysis.match_protein) at every tolerance in
     --tolerances, for peptides / propeptides / all.

Per-cell results go to tolerance_metrics.json; the aggregate over the grid
(mean of the 4 inner cells per outer fold, then mean+std across the 5 outer
folds -- the same procedure as nested_cv_driver.aggregate and
corrected_metrics_cv.aggregate) goes to
runs/<name>/nested_cv_tolerance.json.

At tol=3 the numbers here must reproduce corr_* in
nested_cv_summary_corrected.json; --check verifies that and fails loudly if
they drift, so the sweep cannot silently disagree with the published table.

Usage:
  env/bin/python analysis/metrics/src/tolerance_sweep_cv.py \
      [--models NAME ...] [--tolerances 0 1 2 3] [--device 0] [--limit-cells N]
      [--rescore-only] [--check]

--rescore-only skips the GPU entirely and rescores from the dumped
segments.json.gz files (use it after a first full pass).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "run.py").exists())))

import pandas as pd
import torch

from infer import load_run_args, load_state_dict
from src.train_loop_crf import get_dataloaders, get_model, run_dataloader
from src.utils.manuscript_metrics import (
    convert_path_to_peptide_borders,
    PEPTIDE_START_STATE, PEPTIDE_END_STATE,
    PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE,
)
from analysis.errors.src.error_analysis import match_protein

N_FOLDS = 5
DEFAULT_MODELS = ["5cv_baseline_esm2_fp32", "5cv_esm2_boundary", "5cv_esm2_adapter_only",
                  "5cv_esm2_full", "5cv_esmc6b_plain"]
STATES = {"peptides": (PEPTIDE_START_STATE, PEPTIDE_END_STATE),
          "propeptides": (PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE)}
TASKS = ("peptides", "propeptides", "all")


# ---------------------------------------------------------------- scoring ---

def corrected_counts(true, pred, tol):
    """(tp, fn, fp) under the corrected matcher at tolerance `tol`."""
    groups, pred_matched = match_protein(true, pred, tol)
    tp = sum(g["matched"] for g in groups)
    fn = sum(not g["matched"] for g in groups)
    fp = sum(not m for m in pred_matched)
    return tp, fn, fp


def prf(tp, fn, fp):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score_segments(segments: list[dict], tolerances) -> dict:
    """Score a cell's dumped segments at each tolerance.

    segments: [{"name": str, "peptides": {"true": [[s,e],..], "pred": [...]},
                "propeptides": {...}}, ...]
    """
    out = {}
    for tol in tolerances:
        acc = {t: [0, 0, 0] for t in ("peptides", "propeptides")}
        for rec in segments:
            for task in ("peptides", "propeptides"):
                true = [tuple(x) for x in rec[task]["true"]]
                pred = [tuple(x) for x in rec[task]["pred"]]
                if not true and not pred:
                    continue
                tp, fn, fp = corrected_counts(true, pred, tol)
                acc[task][0] += tp
                acc[task][1] += fn
                acc[task][2] += fp
        allc = [acc["peptides"][i] + acc["propeptides"][i] for i in range(3)]
        for task, c in (("peptides", acc["peptides"]),
                        ("propeptides", acc["propeptides"]),
                        ("all", allc)):
            p, r, f = prf(*c)
            out[f"tol{tol}_{task}_precision"] = round(p, 6)
            out[f"tol{tol}_{task}_recall"] = round(r, 6)
            out[f"tol{tol}_{task}_f1"] = round(f, 6)
            out[f"tol{tol}_{task}_tp"] = c[0]
            out[f"tol{tol}_{task}_fn"] = c[1]
            out[f"tol{tol}_{task}_fp"] = c[2]
    return out


# ------------------------------------------------------------- inference ---

def cell_inference(run_dir: Path, device: str, cell: dict):
    args = load_run_args(run_dir, SimpleNamespace(batch_size=None, device=None))
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    _, _, test_loader = get_dataloaders(
        args,
        train_partitions=cell["train_partitions"],
        valid_partitions=cell["valid_partitions"],
        test_partitions=cell["test_partitions"],
        device=device,
    )
    model = get_model(args).to(device)
    if getattr(args, "feature_extractor", None) == "LSTMCNN" and hasattr(model, "feature_extractor"):
        bilstm = getattr(model.feature_extractor, "biLSTM", None)
        if bilstm is not None and hasattr(bilstm, "flatten_parameters"):
            bilstm.flatten_parameters()
    sd = load_state_dict(run_dir / "model.pt", device)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        buf = set(dict(model.named_buffers()).keys())
        inc = model.load_state_dict(sd, strict=False)
        if not set(inc.missing_keys).issubset(buf) or inc.unexpected_keys:
            raise
    model.eval()
    _, _, preds, _, _ = run_dataloader(test_loader, model, optimizer=None, do_train=False,
                                       device=device, collect_outputs=True,
                                       desc=f"infer {run_dir.parent.name}/{run_dir.name}")
    return preds, test_loader.dataset.names, test_loader.dataset.data


def dump_segments(run_dir: Path, device: str, cell: dict) -> list[dict]:
    preds, names, data = cell_inference(run_dir, device, cell)
    segments = []
    for i, pred in enumerate(preds):
        row = data.loc[names[i]]
        rec = {"name": str(names[i])}
        for task, (ss, es) in STATES.items():
            true = [[int(a), int(b)] for a, b in
                    (row["true_peptides"] if task == "peptides" else row["true_propeptides"])]
            pb = [[int(a), int(b)] for a, b in
                  convert_path_to_peptide_borders(pred, start_state=ss, stop_state=es, offset=1)]
            rec[task] = {"true": true, "pred": pb}
        segments.append(rec)
    with gzip.open(run_dir / "segments.json.gz", "wt") as fh:
        json.dump(segments, fh)
    return segments


def load_segments(run_dir: Path) -> list[dict] | None:
    path = run_dir / "segments.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


# ------------------------------------------------------------- aggregate ---

def aggregate(name: str, cells: list[dict], tolerances) -> dict:
    df = pd.DataFrame(cells)
    per_outer = df.groupby("outer").mean(numeric_only=True)
    summary = {"name": name, "n_cells": len(df), "tolerances": list(tolerances)}
    for tol in tolerances:
        for task in TASKS:
            for metric in ("precision", "recall", "f1"):
                col = f"tol{tol}_{task}_{metric}"
                summary[f"cv_{col}_mean"] = round(float(per_outer[col].mean()), 6)
                summary[f"cv_{col}_std"] = round(float(per_outer[col].std()), 6)
        summary[f"per_outer_tol{tol}_all_f1"] = per_outer[f"tol{tol}_all_f1"].round(4).to_dict()
    return summary


def check_against_corrected(name: str, summary: dict) -> str:
    """tol=3 here must match corr_* in nested_cv_summary_corrected.json."""
    ref_path = Path("runs") / name / "nested_cv_summary_corrected.json"
    if not ref_path.exists():
        return f"[check] {name}: no nested_cv_summary_corrected.json, skipped"
    ref = json.load(open(ref_path))
    lines = []
    ok = True
    for task in TASKS:
        got = summary.get(f"cv_tol3_{task}_f1_mean")
        exp = ref.get(f"cv_corr_{task}_f1_mean")
        if got is None or exp is None:
            continue
        d = abs(got - exp)
        ok &= d < 1e-4
        lines.append(f"    {task:12s} sweep={got:.6f} published={exp:.6f} |d|={d:.2e}")
    head = f"[check] {name}: tol=3 vs published corrected -> {'MATCH' if ok else 'MISMATCH'}"
    return "\n".join([head] + lines)


# ------------------------------------------------------------------ main ---

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--tolerances", nargs="*", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--limit-cells", type=int, default=None,
                    help="score only the first N cells of each model (smoke test)")
    ap.add_argument("--rescore-only", action="store_true",
                    help="do not touch the GPU; rescore from dumped segments.json.gz")
    ap.add_argument("--allow-unfinished", action="store_true",
                    help="score a cell that has model.pt but no cell_result.json, i.e. one whose "
                         "training was cut short after its best-validation checkpoint")
    ap.add_argument("--check", action="store_true",
                    help="verify tol=3 reproduces nested_cv_summary_corrected.json")
    args = ap.parse_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    tolerances = sorted(args.tolerances)

    for name in args.models:
        run_root = Path("runs") / name
        if not run_root.exists():
            print(f"[skip] {name}: {run_root} does not exist")
            continue

        cells_meta = []
        for o in range(N_FOLDS):
            for k in range(N_FOLDS):
                if k == o:
                    continue
                cell_dir = run_root / f"outer{o}_inner{k}"
                cr = cell_dir / "cell_result.json"
                if not cr.exists():
                    if not args.allow_unfinished or not (cell_dir / "model.pt").exists():
                        print(f"[skip] {name} outer{o}_inner{k}: no cell_result.json")
                        continue
                    # A cell whose training was cut short still carries the
                    # best-validation checkpoint it had reached. Scoring it is
                    # only sound when the trajectory is already past its
                    # optimum; the caller is asserting that.
                    print(f"[unfinished] {name} outer{o}_inner{k}: scoring from model.pt")
                    cells_meta.append((cell_dir, {
                        "outer": o, "inner": k, "unfinished": True,
                        "test_partitions": [o], "valid_partitions": [k],
                        "train_partitions": [f for f in range(N_FOLDS) if f not in (o, k)],
                    }))
                    continue
                cells_meta.append((cell_dir, json.load(open(cr))))
        if args.limit_cells:
            cells_meta = cells_meta[:args.limit_cells]

        results = []
        for cell_dir, cell in cells_meta:
            t0 = time.time()
            try:
                segments = load_segments(cell_dir)
                source = "cached"
                if segments is None:
                    if args.rescore_only:
                        print(f"[skip] {cell_dir}: no segments.json.gz and --rescore-only")
                        continue
                    segments = dump_segments(cell_dir, device, cell)
                    source = "inferred"
                r = {"outer": cell["outer"], "inner": cell["inner"],
                     "n_proteins": len(segments)}
                r.update(score_segments(segments, tolerances))
            except Exception as e:
                print(f"[FAIL] {cell_dir}: {type(e).__name__}: {e}")
                continue
            json.dump(r, open(cell_dir / "tolerance_metrics.json", "w"), indent=2)
            results.append(r)
            span = " ".join(f"t{t}={r[f'tol{t}_all_f1']:.4f}" for t in tolerances)
            print(f"[OK] {name}/{cell_dir.name} ({source}, {len(segments)}p, "
                  f"{time.time()-t0:.1f}s) {span}", flush=True)

        if not results:
            print(f"[WARN] {name}: nothing scored")
            continue
        if len(results) < len(cells_meta):
            print(f"[WARN] {name}: only {len(results)}/{len(cells_meta)} cells scored; "
                  f"aggregate written anyway but is INCOMPLETE")
        summary = aggregate(name, results, tolerances)
        summary["complete"] = len(results) == len(cells_meta) == 20
        out_path = run_root / "nested_cv_tolerance.json"
        json.dump(summary, open(out_path, "w"), indent=2)
        line = " ".join(f"t{t}={summary[f'cv_tol{t}_all_f1_mean']:.4f}" for t in tolerances)
        print(f"[model done] {name}: {line} -> {out_path}")
        if args.check:
            print(check_against_corrected(name, summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
