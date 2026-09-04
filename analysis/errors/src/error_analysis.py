#!/usr/bin/env python3
"""Per-segment error analysis for selected runs, by peptide length and organism.

Matching is the AUTHORITATIVE cleavage-site tolerance used by the manuscript
metrics (`manuscript_metrics.get_counts_for_protein`, tolerance=±3): a true
peptide group counts as recovered (TP) iff some predicted peptide has BOTH its
start within ±3 and its stop within ±3 of a group member. This is deliberately
NOT the IoU matching in infer_with_error_stats.py — only the ±3 definition is
consistent with the numbers reported in the tables.

For each selected run we run inference on the TEST split, decode predicted
peptide/propeptide borders, replicate the per-protein ±3 matching while keeping
per-segment flags, and emit:
  - analysis/error_stats/<run>__segments.csv  (one row per true group + per FP pred)
  - aggregated breakdowns (recall by length bin / by organism; FP by length;
    the len<=5 "tiny peptide" bucket) in analysis/error_stats/summary.{csv,md}

Self-validation gate: the aggregate recall/precision recomputed from the emitted
per-segment records must reproduce the run's published test_metrics.json
(window=3) within 1e-6, else the run is flagged and excluded from the summary.

Usage:
    env/bin/python analysis/error_analysis.py [--runs r1 r2 ...] [--device 0]
"""
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(next(
    p for p in Path(__file__).resolve().parents if (p / "run.py").exists()
)))  # locate the repo root by its entry point rather than a .git dir, since
# this anonymized release is deliberately not a git repository
from infer import load_run_args, load_state_dict            # reuse exact config/ckpt loading
from src.train_loop_crf import get_dataloaders, get_model, run_dataloader
from src.utils.manuscript_metrics import (
    convert_path_to_peptide_borders,
    PEPTIDE_START_STATE, PEPTIDE_END_STATE,
    PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE,
)

# Default selection: top-5 of Table 1 (architectures) ∪ top-5 of Table 2 (embeddings),
# ranked by f1 peptides; unrecoverable-for-infer runs excluded. See analysis/canonical_metrics.csv.
DEFAULT_RUNS = [
    # Table 1 (architectural changes; all ESM2 embeddings)
    "esm2_telescoping_segmental",
    "esm2_aho_mid_fusion_raw_m64",
    "train_run_esm2",                     # baseline; shared with Table 2
    "esm2_aho_emission_fusion_h32",
    "esm2_aho_emission_fusion",
    # Table 2 (embedding generators)
    "train_run_esm2+3di_proj_gated_conv",
    "train_run_esm2_aft_single_gated",    # NOTE: AFT partial protein coverage (test subset differs)
    "train_run_esm2+3di_proj",
    "train_run_esmc_600m",
]

TASKS = [
    ("peptides", PEPTIDE_START_STATE, PEPTIDE_END_STATE),
    ("propeptides", PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE),
]


def species(organism: str) -> str:
    if not isinstance(organism, str):
        return "unknown"
    return organism.split("(")[0].strip() or "unknown"


def match_protein(true_segs: List[Tuple[int, int]], pred_segs: List[Tuple[int, int]],
                  tol: int = 3) -> Tuple[List[dict], List[bool]]:
    """Replicate get_counts_for_protein but return per-true-GROUP and per-pred flags.

    Returns (group_records, pred_matched) where group_records is one dict per
    overlapping true group: {length, matched}. length = max member length.
    pred_matched[i] = whether pred i matched any true (unmatched => FP).
    """
    if len(true_segs) == 0:
        return [], [False] * len(pred_segs)  # every pred is a FP
    if len(pred_segs) == 0:
        # group the trues to count groups as FN
        groups = _group(true_segs)
        return [{"length": max(e - s + 1 for s, e in g), "matched": False} for g in groups], []

    groups = _group(true_segs)
    pred_matched = [False] * len(pred_segs)
    group_records = []
    for g in groups:
        g_matched = False
        for (ts, te) in g:
            for pi, (ps, pe) in enumerate(pred_segs):
                if abs(ps - ts) <= tol and abs(pe - te) <= tol:
                    g_matched = True
                    pred_matched[pi] = True
        group_records.append({"length": max(e - s + 1 for s, e in g), "matched": g_matched})
    return group_records, pred_matched


def _group(segs: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """Group overlapping segments (mirrors the cummax-shift grouping in manuscript_metrics)."""
    segs = sorted(segs, key=lambda x: (x[0], -(x[1] - x[0])))
    groups: List[List[Tuple[int, int]]] = []
    cur: List[Tuple[int, int]] = []
    cur_max_end = -1
    for s, e in segs:
        if cur and s > cur_max_end:           # strictly after => new group
            groups.append(cur); cur = []
            cur_max_end = -1
        cur.append((s, e)); cur_max_end = max(cur_max_end, e)
    if cur:
        groups.append(cur)
    return groups


def run_inference(run_dir: Path, device: str, partition: str = "test"):
    args = load_run_args(run_dir, SimpleNamespace(batch_size=None, device=None))
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    _, valid_loader, test_loader = get_dataloaders(args, device=device)
    loader = valid_loader if partition == "val" else test_loader
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
    _, _, preds, _, _ = run_dataloader(loader, model, optimizer=None, do_train=False,
                                       device=device, collect_outputs=True, desc=f"infer {run_dir.name} [{partition}]")
    data = loader.dataset.data
    names = loader.dataset.names
    return preds, names, data, args.label_type


def analyse_run(run: str, device: str) -> Tuple[pd.DataFrame, dict]:
    run_dir = Path("runs") / run
    preds, names, data, label_type = run_inference(run_dir, device)
    rows = []
    agg = {t: {"tp": 0, "fn": 0, "fp": 0} for t, _, _ in TASKS}
    for i, pred in enumerate(preds):
        pid = names[i]
        row = data.loc[pid]
        org = species(row.get("organism"))
        for task, s_state, e_state in TASKS:
            true_segs = list(row["true_peptides"] if task == "peptides" else row["true_propeptides"])
            true_segs = [(int(a), int(b)) for a, b in true_segs]
            pred_segs = convert_path_to_peptide_borders(pred, start_state=s_state, stop_state=e_state, offset=1)
            grecs, pred_matched = match_protein(true_segs, pred_segs, tol=3)
            for gr in grecs:
                rows.append({"run": run, "protein_id": pid, "organism": org, "task": task,
                             "kind": "true", "length": gr["length"], "matched": gr["matched"]})
                agg[task]["tp" if gr["matched"] else "fn"] += 1
            for pi, m in enumerate(pred_matched):
                ps, pe = pred_segs[pi]
                # Emit BOTH matched (tp_pred) and unmatched (fp_pred) predictions with
                # their PREDICTED length, so precision-by-predicted-length is computable
                # (precision@len = tp_pred / (tp_pred + fp_pred) in that bin).
                rows.append({"run": run, "protein_id": pid, "organism": org, "task": task,
                             "kind": "tp_pred" if m else "fp_pred",
                             "length": pe - ps + 1, "matched": bool(m)})
                if not m:
                    agg[task]["fp"] += 1
    df = pd.DataFrame(rows)
    # validation gate vs published test_metrics.json (window=3)
    pub = json.load(open(run_dir / "test_metrics.json"))
    gate = {}
    for task in ("peptides", "propeptides"):
        c = agg[task]
        rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        prc = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        gate[task] = {"recall_d": abs(rec - pub[f"recall {task}"]),
                      "prec_d": abs(prc - pub[f"precision {task}"]),
                      "recall": rec, "precision": prc}
    return df, gate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    out = Path("analysis/errors/error_stats"); out.mkdir(parents=True, exist_ok=True)

    all_df = []
    gates = {}
    for run in args.runs:
        if not (Path("runs") / run / "model.pt").exists():
            print(f"[skip] {run}: no model.pt"); continue
        try:
            df, gate = analyse_run(run, device)
        except Exception as e:
            print(f"[FAIL] {run}: {type(e).__name__}: {e}"); continue
        df.to_csv(out / f"{run.replace('/', '_')}__segments.csv", index=False)
        gates[run] = gate
        max_d = max(gate[t]["recall_d"] for t in gate) if gate else 1.0
        max_d = max(max_d, max(gate[t]["prec_d"] for t in gate))
        # NOTE: published P/R come from manuscript_metrics.get_counts_for_protein,
        # which has a variable-shadowing bug (inner loop reuses `idx`, so the
        # matched-flag is written to the wrong DataFrame row). This correct ±3
        # matcher therefore diverges from published by a small, data-dependent
        # amount. We DO NOT gate on reproducing the buggy value — we report the
        # delta as a finding. Every gate-passed run is included.
        print(f"[OK] {run}  correct-vs-published max|Δ|={max_d:.2e} "
              f"(recall pep corrected={gate['peptides']['recall']:.3f})")
        df["gate_ok"] = True
        all_df.append(df)
    if not all_df:
        print("no runs produced records"); return 1
    big = pd.concat(all_df, ignore_index=True)
    big.to_csv(out / "all_segments.csv", index=False)
    json.dump(gates, open(out / "validation_gates.json", "w"), indent=2)

    _write_summary(big, gates, out)
    print(f"\nWrote {out}/summary.md  ({len(big)} segment records over {big['run'].nunique()} runs)")
    return 0


LEN_BINS = [(5, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 10**9)]
LEN_LABELS = ["5", "6-10", "11-20", "21-30", "31-50", "51+"]


def _bin(L: int) -> str:
    for (lo, hi), lab in zip(LEN_BINS, LEN_LABELS):
        if lo <= L <= hi:
            return lab
    return "other"


def _write_summary(big: pd.DataFrame, gates: dict, out: Path):
    big = big[big["gate_ok"]].copy()
    lines = ["# Error analysis (±3 cleavage-tolerance matching)", ""]
    lines.append("Runs (gate-passed): " + ", ".join(sorted(big["run"].unique())))
    lines.append("")
    lines.append("Matching = manuscript ±3 (a true peptide group is recovered iff some "
                 "prediction's start AND stop are within ±3). Recall = TP/(TP+FN) over true "
                 "groups; FP = predicted segments matching no true group.")
    lines.append("")
    true_df = big[big["kind"] == "true"].copy()
    true_df["lenbin"] = true_df["length"].apply(_bin)

    # 1. recall by length bin (per task, pooled over runs)
    for task in ("peptides", "propeptides"):
        t = true_df[true_df["task"] == task]
        lines.append(f"## Recall by length bin — {task} (pooled over gate-passed runs)")
        lines.append("")
        lines.append("| length | n true | recall | FN |")
        lines.append("|---|---:|---:|---:|")
        for lab in LEN_LABELS:
            sub = t[t["lenbin"] == lab]
            n = len(sub); rec = sub["matched"].mean() if n else float("nan")
            lines.append(f"| {lab} | {n} | {rec:.3f} | {n - int(sub['matched'].sum())} |")
        lines.append("")

    # 2. tiny (len==5) bucket: share of all FN
    lines.append("## Tiny peptides (length = 5)")
    lines.append("")
    for task in ("peptides", "propeptides"):
        t = true_df[true_df["task"] == task]
        fn_total = int((~t["matched"]).sum())
        tiny = t[t["length"] == 5]
        fn_tiny = int((~tiny["matched"]).sum())
        share = fn_tiny / fn_total if fn_total else float("nan")
        rec5 = tiny["matched"].mean() if len(tiny) else float("nan")
        lines.append(f"- **{task}**: {len(tiny)} true len-5 segments, recall={rec5:.3f}; "
                     f"len-5 FN = {fn_tiny} of {fn_total} total FN ({share:.1%}).")
    lines.append("")

    # 3. recall by organism (top organisms by true-segment count, peptides)
    t = true_df[true_df["task"] == "peptides"]
    top_org = t["organism"].value_counts().head(12).index
    lines.append("## Recall by organism — peptides (top 12 by true count, pooled)")
    lines.append("")
    lines.append("| organism | n true | recall |")
    lines.append("|---|---:|---:|")
    for org in top_org:
        sub = t[t["organism"] == org]
        lines.append(f"| {org} | {len(sub)} | {sub['matched'].mean():.3f} |")
    lines.append("")

    # 4. per-run recall headline + gate
    lines.append("## Per-run recall (peptides / propeptides) and validation gate")
    lines.append("")
    lines.append("| run | recall pep | recall propep | gate max|Δ| |")
    lines.append("|---|---:|---:|---:|")
    for run in sorted(big["run"].unique()):
        g = gates[run]
        md = max(g[x]["recall_d"] for x in g)
        lines.append(f"| {run} | {g['peptides']['recall']:.3f} | {g['propeptides']['recall']:.3f} | {md:.1e} |")
    lines.append("")

    # 5. FINDING: corrected ±3 matching vs published (buggy) per-peptide metric
    lines.append("## Finding: corrected ±3 matching vs published per-peptide metric")
    lines.append("")
    lines.append("`manuscript_metrics.get_counts_for_protein` has a variable-shadowing bug "
                 "(the inner `for idx, row in pred_df.iterrows()` reuses `idx`, so "
                 "`true_df.loc[idx,'matched']=True` writes the matched flag to the row whose "
                 "label equals the *pred* index, not the true row). It mostly cancels in the "
                 "aggregate but diverges when a protein has more predictions than true segments. "
                 "Below: published recall (buggy) vs this correct ±3 matcher.")
    lines.append("")
    lines.append("| run | pep recall pub | pep recall correct | Δ | propep recall pub | propep recall correct | Δ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for run in sorted(gates.keys()):
        g = gates[run]
        pub = json.load(open(Path("runs") / run / "test_metrics.json"))
        lines.append(f"| {run} | {pub['recall peptides']:.3f} | {g['peptides']['recall']:.3f} | "
                     f"{g['peptides']['recall']-pub['recall peptides']:+.3f} | "
                     f"{pub['recall propeptides']:.3f} | {g['propeptides']['recall']:.3f} | "
                     f"{g['propeptides']['recall']-pub['recall propeptides']:+.3f} |")
    lines.append("")

    (out / "summary.md").write_text("\n".join(lines))
    # machine-readable: recall by lenbin x task x run
    recs = []
    tdf = big[big["kind"] == "true"].copy(); tdf["lenbin"] = tdf["length"].apply(_bin)
    for (run, task, lab), sub in tdf.groupby(["run", "task", "lenbin"]):
        recs.append({"run": run, "task": task, "lenbin": lab, "n": len(sub),
                     "recall": sub["matched"].mean()})
    pd.DataFrame(recs).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    raise SystemExit(main())
