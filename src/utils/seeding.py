"""Reproducibility helpers: global seeding + determinism toggles.

Used by both training (`src/train_loop_crf.py`) and inference (`infer.py`) so a
run can be reproduced bit-for-bit on the same hardware. Without this, a full
eval pass over the CRF/LSTM was observed to drift run-to-run (even ground-truth
residue counts moved), which is why fresh inference diverged from train-time
metrics.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed python/numpy/torch and (optionally) force deterministic algorithms.

    `deterministic=True` sets cuDNN deterministic mode and
    `torch.use_deterministic_algorithms(True, warn_only=True)`. We keep
    `warn_only=True` so ops without a deterministic implementation degrade to a
    warning instead of crashing a long run. CUBLAS_WORKSPACE_CONFIG is required
    for deterministic cuBLAS matmuls; set it here as a best effort (it only
    takes effect if set before the first CUDA context, so also export it in the
    launch environment for full determinism).
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch without warn_only kwarg; fall back to best effort.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


def seeded_generator(seed: int = 42) -> torch.Generator:
    """A torch.Generator for DataLoader(shuffle=True) so batch order is fixed."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:  # pragma: no cover - used by DataLoader
    """worker_init_fn so each DataLoader worker is deterministically seeded."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
