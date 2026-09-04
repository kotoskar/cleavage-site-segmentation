#!/usr/bin/env python3
"""Re-score completed 5-fold nested-CV cells with the corrected +/-3 matcher.

WHY THIS EXISTS
----------------
The live training/eval pipeline (src/train_loop_crf.py -> compute_all_metrics ->
compute_peptide_finding_metrics -> get_counts_for_protein, in
src/utils/manuscript_metrics.py) has a variable-shadowing bug: the outer loop over
true segments and the inner loop over predicted segments both reuse the loop
variable `idx`, so a match's "matched" flag gets written to the wrong true-segment
row. This deflates recall/F1, more so for proteins with more predicted than true
segments. It is present in the upstream DeepPeptide code, not something introduced
here (see paper appendix).

cell_result.json for every 5x4 nested-CV cell (runs/5cv_*/outer{o}_inner{k}/) was
produced by that buggy matcher. A separately implemented corrected matcher
(analysis/errors/src/error_analysis.py:match_protein) already exists and was used
to re-score the OLDER single-split runs (analysis/metrics/src/corrected_metrics.py)
-- but never against the nested-CV cells, because that requires re-running TEST-
partition inference from each cell's saved checkpoint, which only exist on the
machine that did the training (they are not committed to git / not in this clone
by default).

WHAT THIS SCRIPT DOES
----------------------
For each nested-CV cell:
  1. Reads train/valid/test partitions from that cell's own cell_result.json
     (NOT from config.json, which does not store them for nested cells, and NOT
     from any function's hard-coded defaults -- both are wrong for nested-CV
     cells other than by accident).
  2. Rebuilds the TEST-partition dataloader and reloads the saved best checkpoint
     (runs/<name>/outer{o}_inner{k}/model.pt), reusing the exact loading code
     used elsewhere in this repo (infer.load_run_args/load_state_dict,
     src.train_loop_crf.get_dataloaders/get_model/run_dataloader).
  3. Runs inference once, then scores the SAME predictions two ways: the original
     (buggy) matcher and the corrected one, side by side -- exactly the
     dual-reporting approach already used in corrected_metrics.py.
  4. Writes cell_result_corrected.json next to cell_result.json (does not
     overwrite anything).
  5. Re-aggregates exactly like nested_cv_driver.py's aggregate(): mean over the
     4 inner cells per outer fold, then mean +/- std ACROSS the 5 outer folds
     (not across all 20 cells -- the 4 inner cells per outer fold share a test
     set and are not independent).

SELF-VALIDATION GATE
---------------------
Each cell's reconstructed "original-matcher" F1 is compared against the F1
already published in cell_result.json. If they don't match (within float
tolerance), the partition/checkpoint reconstruction for that cell is wrong, and
its corrected number should NOT be trusted -- the aggregate step prints a warning
and flags this instead of silently proceeding.

CAVEAT THIS SCRIPT DOES NOT FIX
---------------------------------
This only re-scores the FINAL test-set evaluation. It cannot retroactively change
which epoch was selected as "best" during training -- that early-stopping
selection was itself based on buggy validation F1 (compute_all_metrics on the
validation split, same bug). A run whose true best epoch differs under the
corrected metric will not be captured by re-scoring alone. See the paper
appendix for how we describe this.

USAGE
-----
Must run on the machine that holds the actual checkpoints (runs/<name>/outer*_inner*/model.pt):

  env/bin/python analysis/experiments/rescore_nested_cv_corrected.py \\
      --names 5cv_baseline_esm2 5cv_esm2_boundary 5cv_esm2_adapter_only 5cv_esm2_full 5cv_esmc6b_plain \\
      --device 0

Re-aggregate only (safe to run anywhere once cell_result_corrected.json files
exist, e.g. after copying them to a machine with no GPU):

  env/bin/python analysis/experiments/rescore_nested_cv_corrected.py --aggregate-only \\
      --names 5cv_baseline_esm2 5cv_esm2_boundary 5cv_esm2_adapter_only 5cv_esm2_full 5cv_esmc6b_plain
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pandas as pd

sys.path.insert(0, str(next(
    p for p in Path(__file__).resolve().parents if (p / "run.py").exists()
)))  # locate the repo root by its entry point rather than a .git dir, since
# this anonymized release is deliberately not a git repository

from src.utils.manuscript_metrics import (
    convert_path_to_peptide_borders,
    get_counts_for_protein,
    PEPTIDE_START_STATE,
    PEPTIDE_END_STATE,
    PROPEPTIDE_START_STATE,
    PROPEPTIDE_END_STATE,
)
from analysis.errors.src.error_analysis import match_protein

N_FOLDS = 5
SANITY_TOL = 1e-4


def cell_pairs():
    return [(o, k) for o in range(N_FOLDS) for k in range(N_FOLDS) if k != o]


def corrected_counts(true, pred, tol=3):
    g, pm = match_protein(true, pred, tol)
    return sum(x["matched"] for x in g), sum(not x["matched"] for x in g), sum(not m for m in pm)


def prf(tp, fn, fp):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score_cell(run_dir: Path, device: str) -> dict:
    """Re-run TEST-partition inference for one nested-CV cell; score both ways."""
    from infer import load_run_args, load_state_dict
    from src.train_loop_crf import get_dataloaders, get_model, run_dataloader

    cell_result = json.load(open(run_dir / "cell_result.json"))
    train_p = cell_result["train_partitions"]
    valid_p = cell_result["valid_partitions"]
    test_p = cell_result["test_partitions"]

    cli_args = SimpleNamespace(batch_size=None, device=None)
    args = load_run_args(run_dir, cli_args)

    _, _, test_loader = get_dataloaders(
        args, train_partitions=train_p, valid_partitions=valid_p, test_partitions=test_p, device=device
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

    _, _, preds, _, _ = run_dataloader(
        test_loader, model, optimizer=None, do_train=False, device=device,
        collect_outputs=True, desc=f"rescore {run_dir.name}",
    )
    data = test_loader.dataset.data
    names = test_loader.dataset.names

    orig = {"peptides": [0, 0, 0], "propeptides": [0, 0, 0]}
    corr = {"peptides": [0, 0, 0], "propeptides": [0, 0, 0]}
    states = {
        "peptides": (PEPTIDE_START_STATE, PEPTIDE_END_STATE),
        "propeptides": (PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE),
    }
    for i, pred in enumerate(preds):
        row = data.loc[names[i]]
        for task, (ss, es) in states.items():
            true = [
                (int(a), int(b))
                for a, b in (row["true_peptides"] if task == "peptides" else row["true_propeptides"])
            ]
            pb = convert_path_to_peptide_borders(pred, start_state=ss, stop_state=es, offset=1)
            otp, ofn, ofp = get_counts_for_protein(true, pb, 3) if (true or pb) else (0, 0, 0)
            ctp, cfn, cfp = corrected_counts(true, pb, 3)
            orig[task][0] += otp
            orig[task][1] += ofn
            orig[task][2] += ofp
            corr[task][0] += ctp
            corr[task][1] += cfn
            corr[task][2] += cfp

    out = {
        "outer": cell_result["outer"],
        "inner": cell_result["inner"],
        "best_epoch": cell_result["best_epoch"],
    }
    for kind, acc in (("orig", orig), ("corr", corr)):
        allc = [acc["peptides"][k] + acc["propeptides"][k] for k in range(3)]
        for task, c in (("peptides", acc["peptides"]), ("propeptides", acc["propeptides"]), ("all", allc)):
            p, r, f = prf(*c)
            out[f"test_{kind}_{task}_precision"] = round(p, 6)
            out[f"test_{kind}_{task}_recall"] = round(r, 6)
            out[f"test_{kind}_{task}_f1"] = round(f, 6)

    published = cell_result.get("test_f1_all")
    out["_published_test_f1_all"] = published
    out["_sanity_matches_published"] = bool(
        published is not None and abs(out["test_orig_all_f1"] - published) < SANITY_TOL
    )
    return out


def aggregate_corrected(name: str) -> Optional[dict]:
    cells = []
    for o, k in cell_pairs():
        p = Path("runs") / name / f"outer{o}_inner{k}" / "cell_result_corrected.json"
        if p.exists():
            cells.append(json.load(open(p)))
    if len(cells) < len(cell_pairs()):
        print(f"[{name}] only {len(cells)}/{len(cell_pairs())} corrected cells present, skipping aggregate")
        return None
    df = pd.DataFrame(cells)
    if not df["_sanity_matches_published"].all():
        bad = df[~df["_sanity_matches_published"]][["outer", "inner"]].to_dict("records")
        print(
            f"[{name}] WARNING: sanity check failed for cells {bad} -- reconstructed "
            f"'orig' F1 does not match the already-published cell_result.json for those "
            f"cells. Do not trust this aggregate (or those specific cells) until resolved."
        )
    per_outer = df.groupby("outer").mean(numeric_only=True)
    summary = {
        "name": name,
        "n_cells": len(df),
        "per_outer_f1_all_corrected": per_outer["test_corr_all_f1"].round(4).to_dict(),
        "cv_f1_all_mean_corrected": float(per_outer["test_corr_all_f1"].mean()),
        "cv_f1_all_std_corrected": float(per_outer["test_corr_all_f1"].std()),
        "cv_f1_all_mean_original": float(per_outer["test_orig_all_f1"].mean()),
        "cv_f1_all_std_original": float(per_outer["test_orig_all_f1"].std()),
        "all_cells_sanity_ok": bool(df["_sanity_matches_published"].all()),
    }
    json.dump(summary, open(Path("runs") / name / "nested_cv_summary_corrected.json", "w"), indent=2)
    print(
        f"[{name}] corrected: {summary['cv_f1_all_mean_corrected']:.4f} +/- "
        f"{summary['cv_f1_all_std_corrected']:.4f}   (original: "
        f"{summary['cv_f1_all_mean_original']:.4f} +/- {summary['cv_f1_all_std_original']:.4f})"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True, help="e.g. 5cv_baseline_esm2 5cv_esm2_boundary ...")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--aggregate-only", action="store_true",
        help="skip inference, just re-aggregate existing cell_result_corrected.json files",
    )
    args = ap.parse_args()

    device = "cpu"
    if not args.aggregate_only:
        import torch
        device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    for name in args.names:
        if not args.aggregate_only:
            for o, k in cell_pairs():
                run_dir = Path("runs") / name / f"outer{o}_inner{k}"
                out_path = run_dir / "cell_result_corrected.json"
                if not (run_dir / "model.pt").exists():
                    print(f"[skip] {run_dir}: no model.pt (not on this machine?)")
                    continue
                if out_path.exists():
                    print(f"[skip done] {run_dir}")
                    continue
                try:
                    result = score_cell(run_dir, device)
                except Exception as e:  # noqa: BLE001 -- keep sweeping other cells on one bad cell
                    print(f"[FAIL] {run_dir}: {e}")
                    continue
                json.dump(result, open(out_path, "w"), indent=2)
                tag = "OK" if result["_sanity_matches_published"] else "SANITY-MISMATCH"
                print(
                    f"[{tag}] {run_dir}: orig_f1={result['test_orig_all_f1']:.4f} "
                    f"corr_f1={result['test_corr_all_f1']:.4f} "
                    f"(published={result['_published_test_f1_all']})"
                )
        aggregate_corrected(name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
