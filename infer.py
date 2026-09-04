#!/usr/bin/env python3
"""Batch inference over saved runs/ experiments.

This script is meant to live next to run.py. It scans subdirectories in runs/,
loads config.json + model.pt for each experiment, rebuilds the model in the same
way training does, runs evaluation on the test split, and saves metrics.

It also computes extra residue-level metrics (MCC and ROC AUC) on top of the
existing manuscript metrics.

Usage:
    python infer.py
    python infer.py --runs_dir runs --device 0
    python infer.py --runs_dir runs --batch_size 32 --only my_run another_run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.train_loop_crf import get_dataloaders, get_model, run_dataloader
from src.utils.seeding import set_seed
from src.utils.manuscript_metrics import (
    PEPTIDE_END_STATE,
    PEPTIDE_START_STATE,
    PROPEPTIDE_END_STATE,
    PROPEPTIDE_START_STATE,
    compute_all_metrics,
)


TRAIN_DEFAULTS: Dict[str, Any] = {
    # Keep these defaults aligned with train_loop_crf.parse_arguments().
    # They are only used when an older config.json does not contain a key.
    "embeddings_dir": "data/embeddings/",
    "checkpoints_dir": "./checkpoints",
    "data_file": "data/labeled_sequences.csv",
    "partitioning_file": "data/graphpart_assignments.csv",
    "embedding": "precomputed",
    "embedding_dim": 1280,
    "esm2_model_name": "esm2_t33_650M_UR50D",
    "esm2_repr_layer": -1,
    "esm2_max_sequence_length": 1022,
    "esm2_lora_rank": 8,
    "esm2_lora_alpha": 16.0,
    "esm2_lora_dropout": 0.0,
    "esm2_lora_target_modules": "q_proj,k_proj,v_proj,out_proj,fc1,fc2",
    "esm2_lora_layers": "all",
    "esm2_lora_train_layer_norm": False,
    "seq_input_size": 1280,
    "struct_input_size": 20,
    "seq_proj_size": 256,
    "residue_proj_size": 16,
    "struct_proj_size": 64,
    "projector_dropout": 0.4,
    "struct_conv_kernel": 5,
    "multiscale_kernels": "3,7,15",
    "multiscale_dropout": 0.1,
    "residue_input_size": 10,
    "aho_hidden_size": 0,
    "aho_mid_hidden_size": 64,
    "aho_dropout": 0.1,
    "aho_scale": 1.0,
    "aho_branch_dropout": 0.0,
    "aho_no_zero_init": False,
    "aho_none_scale": 1.0,
    "aho_pep_scale": 1.0,
    "aho_propep_scale": 1.0,
    "aho_feature_names_file": None,
    "aho_state_boundary_feature": "binary",
    "aho_state_scale": 1.0,
    "aho_state_branch_dropout": 0.0,
    "aho_state_bias_trainable": False,
    "aho_state_pep_inside_bias": 0.0,
    "aho_state_pep_start_bias": 0.0,
    "aho_state_pep_end_bias": 0.0,
    "aho_state_propep_inside_bias": 0.0,
    "aho_state_propep_start_bias": 0.0,
    "aho_state_propep_end_bias": 0.0,
    "aho_state_pep_to_propep_inside_bias": 0.0,
    "aho_state_pep_to_propep_start_bias": 0.0,
    "aho_state_pep_to_propep_end_bias": 0.0,
    "boundary_state_hidden_size": 64,
    "boundary_state_dropout": 0.1,
    "boundary_state_scale": 1.0,
    "boundary_state_no_zero_init": False,
    "bond_loss_lambda": 0.02,
    "bond_soft_window": 5,
    "bond_soft_tau": 1.5,
    "bond_soft_mode": "exp",
    "bond_positive_weight": 10.0,
    "bond_hidden_size": 64,
    "bond_dropout": 0.1,
    "bond_zero_init": False,
    "gated_residual_scale": 0.2,
    # Compatibility alias used by get_model() for gated 3Di variants.
    "gated_gate_bias": -2.5,
    "residue_residual_scale": 0.05,
    "struct_residual_scale": 0.10,
    "residue_branch_dropout": 0.2,
    "struct_branch_dropout": 0.3,
    "residue_gate_bias": -2.5,
    "struct_gate_bias": -2.5,
    "model": "lstmcnncrf",
    "out_dir": "runs/train_run",
    "epochs": 100,
    "batch_size": 48,
    "lr": 1e-4,
    "dropout": 0.1,
    "conv_dropout": 0.1,
    "kernel_size": 3,
    "num_filters": 32,
    "hidden_size": 64,
    "device": 0,
    "port": 12355,
    "feature_extractor": "LSTMCNN",
    "homo_only": False,
    "K": 10,
    "amp": False,
    "amp_dtype": "bf16",
    "label_type": "multistate_with_propeptides",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs", help="Directory with experiment subfolders.")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Override device id from configs. If CUDA is unavailable, CPU is used automatically.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from configs for inference only.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of run folder names to evaluate. By default evaluates all subfolders in runs_dir.",
    )
    parser.add_argument(
        "--test_partitions",
        type=int,
        nargs="*",
        default=[4],
        help="Test partitions to use. Default mirrors training: partition 4.",
    )
    parser.add_argument(
        "--valid_partitions",
        type=int,
        nargs="*",
        default=[3],
        help="Validation partitions. Kept only to mirror training API defaults.",
    )
    parser.add_argument(
        "--train_partitions",
        type=int,
        nargs="*",
        default=[0, 1, 2],
        help="Train partitions. Kept only to mirror training API defaults.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_infer",
        help="Suffix for generated files inside each run dir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed for deterministic, reproducible inference.",
    )
    return parser.parse_args()


def resolve_device(config_device: int, override_device: Optional[int]) -> str:
    if torch.cuda.is_available():
        device_idx = config_device if override_device is None else override_device
        return f"cuda:{device_idx}"
    return "cpu"


def load_run_args(run_dir: Path, cli_args: argparse.Namespace) -> SimpleNamespace:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    merged = dict(TRAIN_DEFAULTS)
    merged.update(config)

    # Force the current run directory as output dir. This matters because train.py
    # saved model.pt to args.out_dir/model.pt and we want inference artifacts there too.
    merged["out_dir"] = str(run_dir)

    if cli_args.batch_size is not None:
        merged["batch_size"] = cli_args.batch_size
    if cli_args.device is not None:
        merged["device"] = cli_args.device

    # Older configs may have only one of these equivalent gate-bias names.
    # Use the value from config.json when it exists, not just the fallback default.
    if "gated_gate_bias" not in config and "struct_gate_bias" in config:
        merged["gated_gate_bias"] = config["struct_gate_bias"]
    if "struct_gate_bias" not in config and "gated_gate_bias" in config:
        merged["struct_gate_bias"] = config["gated_gate_bias"]

    return SimpleNamespace(**merged)


def squeeze_prob_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim >= 1 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    return arr


def state_in_ranges(values: np.ndarray, ranges: Sequence[Tuple[int, int]]) -> np.ndarray:
    values = np.asarray(values)
    mask = np.zeros(values.shape, dtype=bool)
    for start, end in ranges:
        mask |= (values >= start) & (values <= end)
    return mask


PEPTIDE_RANGES = [(PEPTIDE_START_STATE, PEPTIDE_END_STATE)]
PROPEPTIDE_RANGES = [(PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE)]
ALL_POSITIVE_RANGES = PEPTIDE_RANGES + PROPEPTIDE_RANGES


class TaskSpec(Tuple[str, Sequence[Tuple[int, int]], str]):
    pass


TASKS: List[Tuple[str, Sequence[Tuple[int, int]], str]] = [
    ("peptides", PEPTIDE_RANGES, "peptide"),
    ("propeptides", PROPEPTIDE_RANGES, "propeptide"),
    ("all", ALL_POSITIVE_RANGES, "all"),
]


def get_task_scores(prob_seq: np.ndarray, task_name: str, label_type: str,
                    positive_ranges: Optional[Sequence[Tuple[int, int]]] = None) -> Optional[np.ndarray]:
    arr = squeeze_prob_array(np.asarray(prob_seq))

    # Most likely cases:
    #   (L,)   -> one positive class probability (binary peptide case)
    #   (L, 2) -> [peptide_prob, propeptide_prob]
    #   (L, C) -> positive channels only or logits-like positive channels.
    if arr.ndim == 1:
        if task_name in {"peptides", "all"}:
            return arr.astype(float)
        return None

    if arr.ndim != 2:
        return None

    num_cols = arr.shape[1]
    if num_cols == 0:
        return None

    # STATE-MARGINAL case (the real inference path): the CRF returns per-residue
    # marginal probabilities over ALL num_states states (e.g. 101 = None + 50 peptide
    # + 50 propeptide), NOT a 3-class coarse vector. The per-residue probability that
    # the residue belongs to a task is the SUM of marginals over that task's state
    # range. Previously this block was absent and the (L,101) array fell through to the
    # coarse-column logic below, which used column 1 (peptide START state only) as the
    # peptide score and column 2 (a peptide state!) as the propeptide score — making
    # residue ROC-AUC meaningless (propeptide AUC ~0.5 / <0.5). Summing over ranges fixes it.
    if positive_ranges is not None and num_cols > 3:
        score = np.zeros(arr.shape[0], dtype=np.float64)
        for lo, hi in positive_ranges:
            score += arr[:, lo:hi + 1].sum(axis=1)
        return np.clip(score, 0.0, 1.0)

    if num_cols == 1:
        if task_name in {"peptides", "all"}:
            return arr[:, 0].astype(float)
        return None

    with_propeptides = "with_propeptides" in label_type

    if with_propeptides:
        if num_cols >= 3:
            peptide_col = 1
            propeptide_col = 2
        else:
            peptide_col = 0
            propeptide_col = 1

        if task_name == "peptides":
            return arr[:, peptide_col].astype(float)
        if task_name == "propeptides":
            return arr[:, propeptide_col].astype(float)
        if task_name == "all":
            return np.clip(arr[:, [peptide_col, propeptide_col]].sum(axis=1), 0.0, 1.0).astype(float)
        return None

    # Binary peptide-only case. If two columns are present, assume [background, peptide].
    peptide_col = 1 if num_cols >= 2 else 0
    if task_name in {"peptides", "all"}:
        return arr[:, peptide_col].astype(float)
    return None


def binary_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return float("nan")
    return float((tp * tn - fp * fn) / denom)


try:
    from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
except Exception:  # pragma: no cover
    sklearn_roc_auc_score = None


def binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    finite_mask = np.isfinite(y_score)
    y_true = y_true[finite_mask]
    y_score = y_score[finite_mask]

    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")

    if sklearn_roc_auc_score is not None:
        return float(sklearn_roc_auc_score(y_true, y_score))

    # Fallback: rank-based AUC with average ranks for ties.
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_score[order]

    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j

    pos = y_true == 1
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(~pos))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    rank_sum_pos = float(np.sum(ranks[pos]))
    u = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def compute_residue_level_metrics(
    probs: Sequence[np.ndarray],
    preds: Sequence[Sequence[int]],
    labels: Sequence[np.ndarray],
    label_type: str,
) -> Dict[str, float]:
    pred_cursor = 0
    collected: Dict[str, Dict[str, List[float]]] = {
        task_name: {"y_true": [], "y_pred": [], "y_score": []}
        for task_name, _, _ in TASKS
    }

    for batch_probs, batch_labels in zip(probs, labels):
        batch_probs = np.asarray(batch_probs)
        batch_labels = np.asarray(batch_labels)
        batch_size = batch_labels.shape[0]

        for row_idx in range(batch_size):
            if pred_cursor >= len(preds):
                raise RuntimeError("Mismatch between preds and labels while reconstructing sequences.")

            pred_seq = np.asarray(preds[pred_cursor])
            pred_cursor += 1
            seq_len = len(pred_seq)

            true_seq = np.asarray(batch_labels[row_idx])[:seq_len]
            prob_seq = np.asarray(batch_probs[row_idx])[:seq_len]

            for task_name, positive_ranges, _ in TASKS:
                y_true = state_in_ranges(true_seq, positive_ranges).astype(np.int64)
                y_pred = state_in_ranges(pred_seq, positive_ranges).astype(np.int64)
                y_score = get_task_scores(prob_seq, task_name, label_type, positive_ranges)

                collected[task_name]["y_true"].append(y_true)
                collected[task_name]["y_pred"].append(y_pred)
                if y_score is not None:
                    collected[task_name]["y_score"].append(np.asarray(y_score, dtype=np.float64))

    if pred_cursor != len(preds):
        raise RuntimeError("Not all decoded predictions were consumed while reconstructing sequences.")

    metrics: Dict[str, float] = {}
    for task_name, _, _ in TASKS:
        y_true_chunks = collected[task_name]["y_true"]
        y_pred_chunks = collected[task_name]["y_pred"]
        y_score_chunks = collected[task_name]["y_score"]

        if not y_true_chunks:
            metrics[f"residue mcc {task_name}"] = float("nan")
            metrics[f"residue roc_auc {task_name}"] = float("nan")
            metrics[f"residue positives {task_name}"] = 0
            metrics[f"residue total {task_name}"] = 0
            continue

        y_true = np.concatenate(y_true_chunks)
        y_pred = np.concatenate(y_pred_chunks)
        metrics[f"residue mcc {task_name}"] = binary_mcc(y_true, y_pred)
        metrics[f"residue positives {task_name}"] = int(y_true.sum())
        metrics[f"residue total {task_name}"] = int(y_true.size)

        if y_score_chunks and sum(len(chunk) for chunk in y_score_chunks) == y_true.size:
            y_score = np.concatenate(y_score_chunks)
            metrics[f"residue roc_auc {task_name}"] = binary_roc_auc(y_true, y_score)
        else:
            metrics[f"residue roc_auc {task_name}"] = float("nan")

    return metrics


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def load_state_dict(model_path: Path, device: str) -> Dict[str, Any]:
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise ValueError(f"Unexpected checkpoint format in {model_path}")


def evaluate_loader(
    loader,
    model: torch.nn.Module,
    device: str,
    args: SimpleNamespace,
) -> Tuple[float, Dict[str, Any], Tuple[Any, Any, Any, Any]]:
    # IMPORTANT: evaluate in fp32, NOT AMP — even when config.amp is True.
    # The training-time TEST metrics (the published P/R/F1 in the tables) were
    # computed in fp32: train_loop_crf.train()'s final `run_dataloader(test_loader,
    # ..., desc='Test')` call omits use_amp (defaults to False). Running eval in
    # bf16 AMP here changes the numerics enough to flip borderline predictions of
    # low-confidence models (e.g. the AFT runs: precision shifts ~0.05-0.13),
    # which made their fresh metrics fail to reproduce train-time. fp32 matches
    # training and reproduces train-time P/R/F1 exactly.
    use_amp = False
    amp_dtype = torch.bfloat16

    loss, probs, preds, _, labels = run_dataloader(
        loader,
        model,
        optimizer=None,
        do_train=False,
        device=device,
        scaler=None,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        collect_outputs=True,
    )

    metrics = compute_all_metrics(
        probs,
        preds,
        labels,
        loader.dataset.names,
        loader.dataset.data,
        windows=[3],
    )[0]
    metrics.update(compute_residue_level_metrics(probs, preds, labels, args.label_type))
    metrics["loss"] = float(loss)

    outputs = (probs, preds, labels, loader.dataset.names)
    return loss, metrics, outputs


def evaluate_single_run(
    run_dir: Path,
    cli_args: argparse.Namespace,
) -> Dict[str, Any]:
    args = load_run_args(run_dir, cli_args)
    device = resolve_device(args.device, cli_args.device)

    # Reproducibility: deterministic eval so fresh inference is repeatable
    # (see memory note deeppeptide-infer-divergence). seed comes from CLI.
    set_seed(getattr(cli_args, "seed", 42))

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.pt in {run_dir}")

    if args.homo_only:
        _, _, test_loader = get_dataloaders(
            args,
            train_partitions=cli_args.train_partitions,
            valid_partitions=cli_args.valid_partitions,
            test_partitions=cli_args.test_partitions,
            device=device,
            restrict=_load_homo_ids(args),
        )
        homo_test_loader = None
    else:
        _, _, test_loader = get_dataloaders(
            args,
            train_partitions=cli_args.train_partitions,
            valid_partitions=cli_args.valid_partitions,
            test_partitions=cli_args.test_partitions,
            device=device,
        )
        _, _, homo_test_loader = get_dataloaders(
            args,
            train_partitions=cli_args.train_partitions,
            valid_partitions=cli_args.valid_partitions,
            test_partitions=cli_args.test_partitions,
            restrict=_load_homo_ids(args),
            device=device,
        )

    model = get_model(args).to(device)
    if getattr(args, "feature_extractor", None) == "LSTMCNN" and hasattr(model, "feature_extractor"):
        bilstm = getattr(model.feature_extractor, "biLSTM", None)
        if bilstm is not None and hasattr(bilstm, "flatten_parameters"):
            bilstm.flatten_parameters()

    state_dict = load_state_dict(model_path, device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Some older checkpoints predate non-learned buffers that were added to
        # the model later (e.g. aho_label_scales, a no-op identity buffer).
        # Allow strict=False ONLY when every missing key is a buffer (never a
        # learned parameter) and nothing is unexpected, so we never silently
        # init-fill a trained weight.
        buffer_names = set(dict(model.named_buffers()).keys())
        incompat = model.load_state_dict(state_dict, strict=False)
        missing = set(incompat.missing_keys)
        unexpected = set(incompat.unexpected_keys)
        if not missing.issubset(buffer_names) or unexpected:
            raise
        print(f"[warn] {run_dir.name}: loaded with strict=False; "
              f"defaulted missing buffer(s) {sorted(missing)}")
    model.eval()

    test_loss, test_metrics, test_outputs = evaluate_loader(test_loader, model, device, args)
    result: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "status": "ok",
        "device": device,
        "test_metrics": test_metrics,
    }

    suffix = cli_args.output_suffix
    with (run_dir / f"test_metrics{suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(test_metrics), f, indent=2, ensure_ascii=False)
    # with (run_dir / f"test_outputs{suffix}.pickle").open("wb") as f:
        # pickle.dump(test_outputs, f)

    if homo_test_loader is not None:
        homo_test_loss, homo_test_metrics, homo_test_outputs = evaluate_loader(homo_test_loader, model, device, args)
        result["homo_test_metrics"] = homo_test_metrics
        with (run_dir / f"homo_test_metrics{suffix}.json").open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(homo_test_metrics), f, indent=2, ensure_ascii=False)
        # with (run_dir / f"homo_test_outputs{suffix}.pickle").open("wb") as f:
            # pickle.dump(homo_test_outputs, f)

    return result


_HOMO_IDS_CACHE: Dict[str, List[str]] = {}


def _load_homo_ids(args: SimpleNamespace) -> List[str]:
    """Load the homo subset id list the same way train_loop_crf.train() does."""
    homo_path = (Path(args.embeddings_dir) / ".." / ".." / "protein_id_homo.txt").resolve()
    cache_key = str(homo_path)
    if cache_key not in _HOMO_IDS_CACHE:
        if homo_path.exists():
            with homo_path.open("r", encoding="utf-8") as f:
                _HOMO_IDS_CACHE[cache_key] = [line.strip() for line in f if line.strip()]
        else:
            _HOMO_IDS_CACHE[cache_key] = []
    return _HOMO_IDS_CACHE[cache_key]


def iter_run_dirs(runs_dir: Path, only: Optional[Sequence[str]]) -> Iterable[Path]:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    only_set = set(only) if only else None
    for path in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if only_set is not None and path.name not in only_set:
            continue
        yield path


def flatten_summary_rows(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in results:
        row: Dict[str, Any] = {
            "run_name": Path(item["run_dir"]).name,
            "status": item["status"],
            "device": item.get("device"),
            "error": item.get("error"),
        }

        if item.get("status") == "ok":
            for prefix, metrics_key in [("test", "test_metrics"), ("homo_test", "homo_test_metrics")]:
                metrics = item.get(metrics_key)
                if not metrics:
                    continue
                for k, v in metrics.items():
                    row[f"{prefix}:{k}"] = v
        rows.append(row)
    return rows


def write_summary_files(runs_dir: Path, results: Sequence[Dict[str, Any]], suffix: str) -> None:
    summary_json = runs_dir / f"infer_summary{suffix}.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(list(results)), f, indent=2, ensure_ascii=False)

    rows = flatten_summary_rows(results)
    if not rows:
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    summary_csv = runs_dir / f"infer_summary{suffix}.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(to_jsonable(rows))


def main() -> int:
    cli_args = parse_args()
    runs_dir = Path(cli_args.runs_dir)
    results: List[Dict[str, Any]] = []

    for run_dir in iter_run_dirs(runs_dir, cli_args.only):
        print(f"\n=== [{run_dir.name}] ===")
        try:
            result = evaluate_single_run(run_dir, cli_args)
            results.append(result)
            print(f"[OK] {run_dir.name}")
        except Exception as exc:
            tb = traceback.format_exc()
            failure = {
                "run_dir": str(run_dir),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": tb,
            }
            results.append(failure)
            print(f"[FAILED] {run_dir.name}: {failure['error']}")
            print(tb)

    write_summary_files(runs_dir, results, cli_args.output_suffix)

    total = len(results)
    failed = sum(item["status"] != "ok" for item in results)
    ok = total - failed
    print("\n=== Summary ===")
    print(f"Total runs: {total}")
    print(f"Succeeded:  {ok}")
    print(f"Failed:     {failed}")
    print(f"Summary JSON: {runs_dir / f'infer_summary{cli_args.output_suffix}.json'}")
    print(f"Summary CSV:  {runs_dir / f'infer_summary{cli_args.output_suffix}.csv'}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
