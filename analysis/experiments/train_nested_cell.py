"""Train ONE (outer, inner) cell of 5-fold nested CV.

For outer fold `o` and inner fold `i` (i != o): trains on the remaining 3
folds, uses `i` as the early-stopping/validation fold, and evaluates on `o`
(never seen in training or validation for this cell). This mirrors the
original DeepPeptide paper's nested-CV protocol (inner loop picks the
stopping epoch, outer fold reports quality) on the reconstructed 5-fold
flanking-motif-balanced split.

Usage (from repo root, PYTHONPATH=.):
  python3 analysis/experiments/train_nested_cell.py \
      --base configs/esm2_base.json --emb <embeddings_dir> --split <covered_split.csv> \
      --out <model_name> --outer 0 --inner 1 --n_folds 5 [--data <labeled_sequences.csv>] [--set k=v ...]
"""
import argparse, json, os, sys


def main():
    sys.path.insert(0, os.getcwd())
    from argparse import Namespace

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to a base runs/<x>/config.json (or configs/esm2_base.json) to clone architecture/hparams from")
    ap.add_argument("--emb", required=True, help="embeddings_dir override")
    ap.add_argument("--split", required=True, help="partitioning_file override (5-fold covered split)")
    ap.add_argument("--out", required=True, help="model name; cell goes to runs/<out>/outer{o}_inner{i}")
    ap.add_argument("--outer", type=int, required=True)
    ap.add_argument("--inner", type=int, required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--data", default=None,
        help=(
            "data_file override (labeled_sequences.csv). Defaults to whatever 'data_file' "
            "is already set in --base, so it does not have to be repeated on every call. "
            "Portability note: the original script hardcoded this to a specific dataset "
            "revision (data/uniprot_2026/labeled_sequences.csv) that is not part of this "
            "release; it is now an explicit/derived override instead."
        ),
    )
    ap.add_argument("--set", nargs="*", default=[], help="extra config overrides k=v (JSON-decoded if possible)")
    a = ap.parse_args()

    assert a.outer != a.inner, "outer and inner fold must differ"
    folds = list(range(a.n_folds))
    train_partitions = [f for f in folds if f not in (a.outer, a.inner)]
    valid_partitions = [a.inner]
    test_partitions = [a.outer]

    cell_name = f"{a.out}/outer{a.outer}_inner{a.inner}"
    out_dir = f"runs/{cell_name}"
    os.makedirs(out_dir, exist_ok=True)

    cfg = json.load(open(a.base))
    cfg.update({
        "embeddings_dir": a.emb,
        "data_file": a.data if a.data is not None else cfg["data_file"],
        "partitioning_file": a.split,
        "out_dir": out_dir,
        "checkpoints_dir": f"{out_dir}/checkpoints",
        "seed": a.seed,
        "resume": True,
    })
    # configs/*.json in this release carry architecture/hparams only; the
    # machine-specific knobs are not part of them, so supply the defaults that
    # src/train_loop_crf.py expects (it reads args.device / args.port directly).
    cfg.setdefault("device", 0)
    cfg.setdefault("port", 12355)

    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    json.dump(cfg, open(f"{out_dir}/config.json", "w"), indent=3)

    print(f"[nested_cell] {cell_name}: train={train_partitions} valid(inner)={valid_partitions} "
          f"test(outer)={test_partitions}", flush=True)

    from src.train_loop_crf import train
    try:
        from aim import Run
        run = Run()
    except Exception:
        class _D(dict):
            def track(self, *x, **k): pass
        run = _D()

    best_val_metrics, test_metrics = train(
        Namespace(**cfg),
        train_partitions=train_partitions,
        valid_partitions=valid_partitions,
        test_partitions=test_partitions,
        run=run,
    )

    result = {
        "outer": a.outer, "inner": a.inner,
        "train_partitions": train_partitions, "valid_partitions": valid_partitions, "test_partitions": test_partitions,
        "best_epoch": best_val_metrics.get("epoch"),
        "val_f1_all": best_val_metrics.get("f1 all"),
        "val_f1_peptides": best_val_metrics.get("f1 peptides"),
        "val_f1_propeptides": best_val_metrics.get("f1 propeptides"),
        "test_f1_all": test_metrics.get("f1 all"),
        "test_f1_peptides": test_metrics.get("f1 peptides"),
        "test_f1_propeptides": test_metrics.get("f1 propeptides"),
        "test_precision_all": test_metrics.get("precision all"),
        "test_recall_all": test_metrics.get("recall all"),
    }
    json.dump(result, open(f"{out_dir}/cell_result.json", "w"), indent=2)
    print(f"[nested_cell] {cell_name}: DONE epoch={result['best_epoch']} test_f1_all={result['test_f1_all']}", flush=True)


if __name__ == "__main__":
    main()
