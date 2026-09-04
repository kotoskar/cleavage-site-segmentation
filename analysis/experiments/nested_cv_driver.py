"""Sequential driver for 5-fold nested CV (5 outer x 4 inner = 20 trainings/model).

Reproduces the original DeepPeptide-paper CV protocol on the reconstructed 5-fold
flanking-motif-balanced split: for each
outer fold, 4 independently early-stopped models (one per remaining inner fold used
as the validation/early-stop fold) are trained on the other 3 folds and evaluated on
the untouched outer fold. Averaging the 4 inner epoch/metric values gives the
outer-fold estimate; averaging the 5 outer-fold estimates gives the final CV number.
Report mean +/- std ACROSS THE 5 OUTER FOLDS as the honest uncertainty -- that is the
actual independent-sample axis (the 4 inner cells per outer fold share the same held-out
test set, so they are not independent samples of test performance).

After each cell, runs a fp32 infer.py pass (--train/valid/test_partitions matching the
cell) to also collect residue MCC/AUC per task (peptides/propeptides/all) almost for free.

Reboot-safe: skips any cell whose cell_result.json already exists. Sequential (single
GPU, never co-run with anything else -- a parked second job will starve this one).

Run from repo root, PYTHONPATH=.:
  env/bin/python analysis/experiments/nested_cv_driver.py \
      analysis/experiments/nested_cv_queue.jsonl analysis/experiments/nested_cv_status.json

Queue file = one JSON object per line:
  {"name": "cv_esmc6b_boundary", "base": "configs/esmc6b_boundary.json",
   "emb": "data/embeddings/esmc6b",
   "split": "data/graphpart_5fold.esmc6b.csv",
   "set": {}}
"""
import json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(".").resolve()
QUEUE = Path(sys.argv[1])
STATUS = Path(sys.argv[2])
HERE = Path(__file__).parent
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": "."}
N_FOLDS = 5


def st(**kw):
    kw["ts"] = time.strftime("%Y-%m-%d_%H:%M:%S")
    json.dump(kw, open(STATUS, "w"), indent=2)


def cell_pairs():
    return [(o, k) for o in range(N_FOLDS) for k in range(N_FOLDS) if k != o]


def aggregate(name):
    cells = []
    for o, k in cell_pairs():
        p = ROOT / "runs" / name / f"outer{o}_inner{k}" / "cell_result.json"
        if p.exists():
            cells.append(json.load(open(p)))
    if len(cells) < len(cell_pairs()):
        return None
    df = pd.DataFrame(cells)
    per_outer = df.groupby("outer").mean(numeric_only=True)
    summary = {
        "name": name,
        "n_cells": len(df),
        "mean_epoch": float(df["best_epoch"].mean()),
        "std_epoch": float(df["best_epoch"].std()),
        "per_outer_f1_all": per_outer["test_f1_all"].round(4).to_dict(),
        "cv_f1_all_mean": float(per_outer["test_f1_all"].mean()),
        "cv_f1_all_std": float(per_outer["test_f1_all"].std()),
        "cv_f1_peptides_mean": float(per_outer["test_f1_peptides"].mean()),
        "cv_f1_peptides_std": float(per_outer["test_f1_peptides"].std()),
        "cv_f1_propeptides_mean": float(per_outer["test_f1_propeptides"].mean()),
        "cv_f1_propeptides_std": float(per_outer["test_f1_propeptides"].std()),
    }
    json.dump(summary, open(ROOT / "runs" / name / "nested_cv_summary.json", "w"), indent=2)
    return summary


def run_infer(name, o, k):
    """fp32 infer for one cell -> adds MCC/AUC per task into outer{o}_inner{k}/test_metrics_infer.json."""
    cmd = [PY, "infer.py", "--runs_dir", f"runs/{name}", "--only", f"outer{o}_inner{k}",
           "--device", "0", "--train_partitions", *map(str, [f for f in range(N_FOLDS) if f not in (o, k)]),
           "--valid_partitions", str(k), "--test_partitions", str(o)]
    subprocess.run(cmd, env=ENV, cwd=str(ROOT))


def main():
    items = [json.loads(l) for l in open(QUEUE) if l.strip()]
    pairs = cell_pairs()
    for i, it in enumerate(items):
        name = it["name"]
        for ci, (o, k) in enumerate(pairs):
            done = ROOT / "runs" / name / f"outer{o}_inner{k}" / "cell_result.json"
            if done.exists():
                print(f"[skip done] {name} outer{o}_inner{k}", flush=True)
                continue
            st(stage="training", model=name, model_idx=i, model_total=len(items),
               cell=f"outer{o}_inner{k}", cell_idx=ci, cell_total=len(pairs))
            cmd = [PY, str(HERE / "train_nested_cell.py"),
                   "--base", it["base"], "--emb", it["emb"], "--split", it["split"],
                   "--out", name, "--outer", str(o), "--inner", str(k), "--n_folds", str(N_FOLDS)]
            for sk, sv in it.get("set", {}).items():
                sv_str = sv if isinstance(sv, str) else json.dumps(sv)
                cmd += ["--set", f"{sk}={sv_str}"]
            print(f"[cell {ci + 1}/{len(pairs)}] {name} outer={o} inner={k}", flush=True)
            rc = subprocess.run(cmd, env=ENV, cwd=str(ROOT)).returncode
            if not done.exists():
                st(stage="FAILED", model=name, cell=f"outer{o}_inner{k}", rc=rc)
                print(f"[FAILED] {name} outer{o}_inner{k} rc={rc} -- stopping queue", flush=True)
                return
            st(stage="infer", model=name, cell=f"outer{o}_inner{k}")
            run_infer(name, o, k)
        summary = aggregate(name)
        st(stage="model_done", model=name, model_idx=i, model_total=len(items), summary=summary)
        print(f"[model done] {name}: {summary}", flush=True)
    st(stage="QUEUE_COMPLETE")
    print("[QUEUE_COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
