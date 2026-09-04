# src/infer_and_score.py
"""
Inference + metrics for DeepPeptide CRF models.

What it does:
- Loads a checkpoint (.pth) from ./checkpoints (or custom path)
- Runs inference on a chosen split (train/valid/test partitions)
- Saves raw outputs (probs/preds/labels/names) as a pickle
- Recomputes manuscript_metrics.compute_all_metrics() for requested tolerances (windows)
- Writes metrics json (+ optional pretty print)

Example:
python3 -m src.infer_and_score \
  --data_file data/labeled_sequences.csv \
  --partitioning_file data/graphpart_assignments.csv \
  --embeddings_dir data/embeddings \
  --checkpoint checkpoints/model_99.pth \
  --out_dir runs/infer_model_99 \
  --split test \
  --batch_size 16 \
  --device 0 \
  --windows 0 1 2 3
"""
import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import LSTMCNNCRF, SimpleLSTMCNNCRF, SelfAttentionCRF
from src.models.crf_models import (
    LSTMCNNCRFBoundaryBondLoss,
    LSTMCNNCRFGated3DiBoundary,
)
from src.utils.dataset import PrecomputedCSVForOverlapCRFDataset
from src.utils.manuscript_metrics import compute_all_metrics

import re
import csv
from glob import glob

# -------------------------
# Helpers
# -------------------------

def _parse_windows(values: List[str]) -> List[int]:
    if not values:
        return [3]
    return [int(x) for x in values]


def _split_to_partitions(split: str) -> Tuple[List[int], List[int], List[int]]:
    """
    Matches your training defaults:
      train: [0,1,2], valid:[3], test:[4]
    But we want to infer on just one split at a time.
    """
    if split == "train":
        return [0, 1, 2], [], []
    if split == "valid":
        return [], [3], []
    if split == "test":
        return [], [], [4]
    raise ValueError(f"Unknown split: {split}")


def build_dataloader(
    embeddings_dir: str,
    data_file: str,
    partitioning_file: str,
    partitions: List[int],
    label_type: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    ds = PrecomputedCSVForOverlapCRFDataset(
        embeddings_dir=embeddings_dir,
        data_file=data_file,
        partitioning_file=partitioning_file,
        partitions=partitions,
        label_type=label_type,
        restrict=None,
        device=None,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ds.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return dl


def _load_state_dict(path: str) -> dict:
    """Load a checkpoint saved in either of the two formats this codebase writes.

    Training writes the best epoch as a bare state dict (`<run>/model.pt`) and the
    periodic resume checkpoints as `{'model_state_dict', 'optimizer_state_dict',
    'epoch'}` (`<run>/checkpoints/model_<N>.pth`). Accept both, so that pointing
    at the wrong one is not silently wrong.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def _assert_weights_loaded(missing, unexpected, model, ckpt_path, allow: bool = False) -> None:
    """Fail when the checkpoint did not fully populate the model.

    `load_state_dict(..., strict=False)` leaves any unmatched tensor at its
    random initialisation and returns quietly, so a mismatched architecture
    yields meaningless metrics instead of an error. For a rescoring tool that
    is the worst possible failure mode, and it is not hypothetical: the two
    variant architectures here share most of their tensor names with the base
    model, so a wrong `--model` leaves only a handful of tensors missing and
    would otherwise slip through.

    Any missing tensor is therefore an error unless explicitly allowed.
    """
    if allow or not missing:
        return
    total = len(list(model.state_dict().keys()))
    raise RuntimeError(
        f"Checkpoint {ckpt_path} did not fully populate the model: "
        f"{len(missing)}/{total} tensors missing, {len(unexpected)} unexpected. "
        f"Those tensors are still randomly initialised, so any metric computed "
        f"now would be meaningless. The architecture flags most likely differ "
        f"from the ones this checkpoint was trained with -- pass "
        f"--config <run>/config.json. Use --allow_partial_load to override.\n"
        f"  missing (up to 10): {missing[:10]}"
    )


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "lstmcnncrf":
        model = LSTMCNNCRF(
            input_size=args.embedding_dim,
            num_labels=3 if "with_propeptides" in args.label_type else 2,
            dropout_input=args.dropout,
            num_states=101 if "with_propeptides" in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_conv1=args.conv_dropout,
            feature_extractor=args.feature_extractor,
        )
    elif args.model == "lstmcnncrf_simple":
        model = SimpleLSTMCNNCRF(
            input_size=args.embedding_dim,
            num_labels=3 if args.label_type == "simple_with_propeptides" else 2,
            dropout_input=args.dropout,
            num_states=3 if args.label_type == "simple_with_propeptides" else 2,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_conv1=args.conv_dropout,
        )
    elif args.model == "selfattentioncrf":
        model = SelfAttentionCRF(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_labels=3 if "with_propeptides" in args.label_type else 2,
            dropout_input=args.dropout,
            num_states=121 if "with_propeptides" in args.label_type else 61,
            n_heads=args.num_filters,
            attn_dropout=args.conv_dropout,
        )
    elif args.model == "lstmcnncrf_boundary_bond_loss":
        # Base encoder + boundary-state emission head. The bond head is built
        # for checkpoint compatibility but plays no part in inference; with
        # bond_loss_lambda = 0 it also plays no part in training.
        model = LSTMCNNCRFBoundaryBondLoss(
            input_size=args.embedding_dim,
            num_labels=3 if "with_propeptides" in args.label_type else 2,
            num_states=101 if "with_propeptides" in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
            feature_extractor=args.feature_extractor,
            boundary_state_hidden_size=args.boundary_state_hidden_size,
            boundary_state_dropout=args.boundary_state_dropout,
            boundary_state_scale=args.boundary_state_scale,
            boundary_state_zero_init=not args.boundary_state_no_zero_init,
            boundary_window=getattr(args, "boundary_window", 1),
            bond_loss_lambda=getattr(args, "bond_loss_lambda", 0.0),
            bond_soft_window=getattr(args, "bond_soft_window", 5),
            bond_soft_tau=getattr(args, "bond_soft_tau", 1.5),
            bond_soft_mode=getattr(args, "bond_soft_mode", "exp"),
            bond_positive_weight=getattr(args, "bond_positive_weight", 10.0),
            bond_hidden_size=getattr(args, "bond_hidden_size", 64),
            bond_dropout=getattr(args, "bond_dropout", 0.1),
            bond_zero_init=getattr(args, "bond_zero_init", False),
        )
    elif args.model == "lstmcnncrf_gated3di_boundary":
        # Input adapter (seq_only=True) + optional boundary head. With
        # seq_only=True no structural branch is constructed at all; with
        # boundary_state_scale = 0 the boundary head contributes exactly zero
        # to the emissions.
        model = LSTMCNNCRFGated3DiBoundary(
            input_size=args.embedding_dim,
            seq_input_size=getattr(args, "seq_input_size", args.embedding_dim),
            struct_input_size=getattr(args, "struct_input_size", 20),
            seq_proj_size=getattr(args, "seq_proj_size", 256),
            struct_proj_size=getattr(args, "struct_proj_size", 32),
            dropout_projector=getattr(args, "projector_dropout", 0.4),
            residual_scale=getattr(args, "gated_residual_scale", 0.1),
            struct_branch_dropout=getattr(args, "struct_branch_dropout", 0.3),
            gate_bias=getattr(args, "gated_gate_bias", -2.5),
            struct_conv_kernel=getattr(args, "struct_conv_kernel", 5),
            boundary_state_hidden_size=args.boundary_state_hidden_size,
            boundary_state_dropout=args.boundary_state_dropout,
            boundary_state_scale=args.boundary_state_scale,
            boundary_state_zero_init=not args.boundary_state_no_zero_init,
            boundary_window=getattr(args, "boundary_window", 1),
            seq_only=getattr(args, "gated_seq_only", False),
            num_labels=3 if "with_propeptides" in args.label_type else 2,
            num_states=101 if "with_propeptides" in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
        )
    else:
        raise NotImplementedError(args.model)

    return model


@torch.no_grad()
def infer_one_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    top_k: int,
) -> Tuple[List[np.ndarray], List[List[int]], List[np.ndarray], List[str]]:
    """
    Returns:
      probs_list: list of np arrays (B, L, num_states or num_labels) per batch
      preds_list: list of viterbi paths (list[int]) per sequence
      labels_list: list of np arrays (B, L) per batch (padded)
      names: list of protein ids aligned with preds
    """
    model.eval()

    probs_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    preds_list: List[List[int]] = []
    names: List[str] = []

    # dataset names are in loader.dataset.names in the same order as DataLoader when shuffle=False
    # We'll append names per batch by slicing.
    ds_names = loader.dataset.names
    seen = 0

    for batch in tqdm(loader, desc="Infer", dynamic_ncols=True):
        embeddings, mask, label, peptides = batch
        bsz = embeddings.size(0)

        # move
        embeddings = embeddings.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True).long()

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            # CRFBaseModel forward supports top_k; SimpleLSTMCNNCRF does not.
            try:
                probs, viterbi_paths, _ = model(embeddings, mask, targets=None, skip_marginals=False, top_k=top_k)
            except TypeError:
                probs, viterbi_paths = model(embeddings, mask, targets=None, skip_marginals=False)

        probs_list.append(probs.detach().float().cpu().numpy())
        labels_list.append(label.detach().cpu().numpy())
        preds_list.extend(viterbi_paths)

        names.extend(ds_names[seen:seen + bsz])
        seen += bsz

    return probs_list, preds_list, labels_list, names


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_dir", type=str, required=True)
    p.add_argument("--data_file", "-df", type=str, required=True)
    p.add_argument("--partitioning_file", "-pf", type=str, required=True)

    p.add_argument("--checkpoint", type=str, default=None,
               help="Path to a single checkpoint (.pt/.pth). Optional if --sweep_dir is provided.")
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--split", type=str, choices=["train", "valid", "test"], default="test")

    # model config (must match training)
    p.add_argument("--model", type=str, default="lstmcnncrf", choices=[
        "lstmcnncrf", "lstmcnncrf_simple", "selfattentioncrf",
        "lstmcnncrf_boundary_bond_loss", "lstmcnncrf_gated3di_boundary",
    ])
    p.add_argument("--feature_extractor", type=str, default="LSTMCNN")  # or anything you used
    p.add_argument("--label_type", type=str, default="multistate_with_propeptides")

    # Architecture options for the boundary-head / adapter variants. Rather than
    # repeating these on the command line, pass --config <run>/config.json and
    # they are taken from the config the cell was trained with.
    p.add_argument("--config", type=str, default=None,
                   help="Path to the run's config.json; any key matching an argument below "
                        "overrides its default (explicit command-line flags still win).")
    p.add_argument("--boundary_state_hidden_size", type=int, default=64)
    p.add_argument("--boundary_state_dropout", type=float, default=0.1)
    p.add_argument("--boundary_state_scale", type=float, default=1.0)
    p.add_argument("--boundary_state_no_zero_init", action="store_true")
    p.add_argument("--boundary_window", type=int, default=1)
    p.add_argument("--bond_loss_lambda", type=float, default=0.0)
    p.add_argument("--seq_input_size", type=int, default=None)
    p.add_argument("--struct_input_size", type=int, default=20)
    p.add_argument("--seq_proj_size", type=int, default=256)
    p.add_argument("--struct_proj_size", type=int, default=32)
    p.add_argument("--projector_dropout", type=float, default=0.4)
    p.add_argument("--gated_residual_scale", type=float, default=0.1)
    p.add_argument("--struct_branch_dropout", type=float, default=0.3)
    p.add_argument("--gated_gate_bias", type=float, default=-2.5)
    p.add_argument("--struct_conv_kernel", type=int, default=5)
    p.add_argument("--gated_seq_only", action="store_true")
    p.add_argument("--embedding_dim", type=int, default=1280)

    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--conv_dropout", type=float, default=0.1)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--num_filters", type=int, default=32)
    p.add_argument("--hidden_size", type=int, default=64)

    # runtime
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--no_persistent_workers", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=2)

    # amp
    p.add_argument("--amp", action="store_true", help="Enable AMP autocast")
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--top_k", type=int, default=1, help="CRF decode top_k (only for CRFBaseModel models)")

    # metrics
    p.add_argument("--windows", nargs="*", type=int, default=[0, 1, 2, 3],
               help="tolerance windows, e.g. --windows 0 1 2 3")

    # sweep
    p.add_argument("--sweep_dir", type=str, default=None, help="Directory with checkpoints, e.g. checkpoints/")
    p.add_argument("--sweep_glob", type=str, default="model_*.pth", help="Glob inside sweep_dir")
    p.add_argument("--select_tol", type=int, default=3, help="Tolerance window used for best checkpoint selection")
    p.add_argument("--select_criterion", type=str, default="avg_f1_pep_pro",
                   choices=["avg_f1_pep_pro", "f1_all", "f1_peptides", "f1_propeptides"])
    p.add_argument("--allow_partial_load", action="store_true",
                   help="Permit a checkpoint that does not populate every tensor. "
                        "Off by default: unmatched tensors stay randomly initialised.")
    p.add_argument("--strict_sweep", action="store_true",
                   help="If set, fail sweep when checkpoint has missing/unexpected keys")

    
    args = p.parse_args()

    # Merge a training config.json, if given. Only keys the user did not pass
    # explicitly are taken from it, so command-line flags always win.
    if args.config is not None:
        with open(args.config) as fh:
            cfg = json.load(fh)
        # Split on "=" first: --flag=value and --flag value must both register
        # as explicitly given, or the config would override what the user typed.
        explicit = {a.lstrip("-").split("=", 1)[0].replace("-", "_")
                    for a in sys.argv[1:] if a.startswith("--")}
        for key, value in cfg.items():
            if hasattr(args, key) and key not in explicit and key != "config":
                setattr(args, key, value)

    # The adapter path defaults to "the whole embedding is the sequence channel".
    if args.seq_input_size is None:
        args.seq_input_size = args.embedding_dim

    if args.checkpoint is None and args.sweep_dir is None:
        p.error("Provide either --checkpoint (single model) or --sweep_dir (sweep over checkpoints).")

    windows = _parse_windows(args.windows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Portability fix for the anonymized release: the original script hard-required
    # CUDA. Mirror the same CPU fallback already used by infer.py's resolve_device()
    # so this script also runs on machines without a GPU (e.g. for the smoke test).
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    use_amp = bool(args.amp)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    # dataloader split
    train_p, valid_p, test_p = _split_to_partitions(args.split)
    partitions = train_p or valid_p or test_p
    if not partitions:
        raise RuntimeError("No partitions selected.")

    loader = build_dataloader(
        embeddings_dir=args.embeddings_dir,
        data_file=args.data_file,
        partitioning_file=args.partitioning_file,
        partitions=partitions,
        label_type=args.label_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        persistent_workers=not bool(args.no_persistent_workers),
        prefetch_factor=args.prefetch_factor,
    )

    print(f"Loaded {len(loader.dataset)} sequences for split='{args.split}' partitions={partitions}")

    # If sweep enabled: ignore --checkpoint and run all
    if args.sweep_dir is not None:
        summary = sweep_checkpoints(
            args=args,
            loader=loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            windows=windows,
        )
        _write_sweep_outputs(summary, out_dir)

        best = summary["best"]
        print("\nBEST CHECKPOINT")
        print(f"  checkpoint: {best['checkpoint']}")
        print(f"  epoch:      {best['epoch']}")
        print(f"  score:      {best['score']:.6f}")
        print(f"  criterion:  {summary['select_criterion']} @ tol=±{summary['select_tol']}")
        return

    # model
    model = build_model(args).to(device)
    if args.feature_extractor == "LSTMCNN" and hasattr(model, "feature_extractor") and hasattr(model.feature_extractor, "biLSTM"):
        try:
            model.feature_extractor.biLSTM.flatten_parameters()
        except Exception:
            pass

    # load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    state = _load_state_dict(str(ckpt_path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict strict=False")
        if missing:
            print(f"  missing keys: {len(missing)} (showing up to 20): {missing[:20]}")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)} (showing up to 20): {unexpected[:20]}")
    _assert_weights_loaded(missing, unexpected, model, ckpt_path,
                           allow=getattr(args, 'allow_partial_load', False))

    # inference
    probs, preds, labels, names = infer_one_loader(
        model=model,
        loader=loader,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        top_k=args.top_k,
    )

    # save raw outputs
    outputs_path = out_dir / f"outputs_{args.split}.pickle"
    with open(outputs_path, "wb") as f:
        pickle.dump((probs, preds, labels, names), f)
    print(f"Saved raw outputs: {outputs_path}")

    # compute_all_metrics expects:
    #  probs: list/np arrays OK (your training passes list of arrays too)
    #  preds: list of paths
    #  labels: list/np arrays
    #  names: list[str]
    #  true_df: loader.dataset.data must include true_peptides/true_propeptides (it does in your dataset)
    metrics_list = compute_all_metrics(
        probs=probs,
        preds=preds,
        labels=labels,
        names=np.array(names),
        true_df=loader.dataset.data,
        windows=windows,
    )

    metrics_path = out_dir / f"metrics_{args.split}.json"
    with open(metrics_path, "w") as f:
        json.dump({"split": args.split, "checkpoint": str(ckpt_path), "windows": windows, "metrics": metrics_list}, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # pretty print
    for tol, m in zip(windows, metrics_list):
        print(
            f"[tol=±{tol}] "
            f"pep f1={m['f1 peptides']:.4f} (p={m['precision peptides']:.4f} r={m['recall peptides']:.4f}) | "
            f"pro f1={m['f1 propeptides']:.4f} (p={m['precision propeptides']:.4f} r={m['recall propeptides']:.4f}) | "
            f"all f1={m['f1 all']:.4f}"
        )

def _extract_epoch_from_name(path: str) -> int:
    # checkpoints/model_99.pth -> 99, else -1
    m = re.search(r"model_(\d+)\.pth$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _select_metric(metrics_list: List[Dict[str, float]], windows: List[int], tol: int, criterion: str) -> float:
    """
    metrics_list: list of dicts returned by compute_all_metrics, aligned with windows.
    criterion:
      - "avg_f1_pep_pro" (default): (f1 peptides + f1 propeptides)/2
      - "f1_all"
      - "f1_peptides"
      - "f1_propeptides"
    """
    if tol not in windows:
        raise ValueError(f"Requested tol={tol} not present in windows={windows}")
    i = windows.index(tol)
    m = metrics_list[i]

    if criterion == "avg_f1_pep_pro":
        return 0.5 * (m["f1 peptides"] + m["f1 propeptides"])
    if criterion == "f1_all":
        return m["f1 all"]
    if criterion == "f1_peptides":
        return m["f1 peptides"]
    if criterion == "f1_propeptides":
        return m["f1 propeptides"]
    raise ValueError(f"Unknown criterion: {criterion}")


def sweep_checkpoints(
    args: argparse.Namespace,
    loader: DataLoader,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    windows: List[int],
) -> Dict[str, Any]:
    """
    Runs inference+metrics for all checkpoints in args.sweep_dir (glob args.sweep_glob),
    returns a summary dict with best checkpoint and per-checkpoint scores.
    """
    pattern = os.path.join(args.sweep_dir, args.sweep_glob)
    ckpts = sorted(glob(pattern), key=_extract_epoch_from_name)

    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found by pattern: {pattern}")

    print(f"Sweep: found {len(ckpts)} checkpoints")

    results = []
    best = None

    # Build the model skeleton once; from here on we only load state_dicts into it.
    model = build_model(args).to(device)
    if args.feature_extractor == "LSTMCNN" and hasattr(model, "feature_extractor") and hasattr(model.feature_extractor, "biLSTM"):
        try:
            model.feature_extractor.biLSTM.flatten_parameters()
        except Exception:
            pass

    for ckpt in ckpts:
        state = _load_state_dict(str(ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        _assert_weights_loaded(missing, unexpected, model, ckpt,
                               allow=getattr(args, 'allow_partial_load', False))

        if (missing or unexpected) and args.strict_sweep:
            raise RuntimeError(
                f"Checkpoint {ckpt} has missing/unexpected keys.\n"
                f"missing({len(missing)}): {missing[:10]}\n"
                f"unexpected({len(unexpected)}): {unexpected[:10]}"
            )

        probs, preds, labels, names = infer_one_loader(
            model=model,
            loader=loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            top_k=args.top_k,
        )

        metrics_list = compute_all_metrics(
            probs=probs,
            preds=preds,
            labels=labels,
            names=np.array(names),
            true_df=loader.dataset.data,
            windows=windows,
        )

        score = _select_metric(metrics_list, windows, tol=args.select_tol, criterion=args.select_criterion)

        row = {
            "checkpoint": ckpt,
            "epoch": _extract_epoch_from_name(ckpt),
            "score": float(score),
            "metrics": metrics_list,
        }
        results.append(row)

        if best is None or row["score"] > best["score"]:
            best = row

        print(f"[sweep] {os.path.basename(ckpt)} epoch={row['epoch']} score={row['score']:.6f}")

    return {
        "split": args.split,
        "windows": windows,
        "select_tol": args.select_tol,
        "select_criterion": args.select_criterion,
        "num_checkpoints": len(results),
        "best": best,
        "results": results,
    }


def _write_sweep_outputs(summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / "sweep_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved sweep summary: {json_path}")

    # CSV (only the main columns)
    csv_path = out_dir / "sweep_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "checkpoint", "score"])
        for r in summary["results"]:
            w.writerow([r["epoch"], r["checkpoint"], r["score"]])
    print(f"Saved sweep CSV: {csv_path}")


if __name__ == "__main__":
    main()
