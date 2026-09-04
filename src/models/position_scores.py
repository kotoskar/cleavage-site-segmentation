"""Relative-position scoring functions for telescoping segmental CRF emissions.

The scores returned here are log-potentials, not probabilities.
"""
from __future__ import annotations

import torch


def neg_abs_position_score(r_pred: torch.Tensor, t_rel: torch.Tensor, tau: float = 0.25) -> torch.Tensor:
    """Negative absolute-distance position potential.

    Args:
        r_pred: Predicted preferred relative position, usually in [0, 1].
        t_rel: Candidate relative position for the same residue.
        tau: Positive temperature. Smaller values make mismatch penalties sharper.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return -torch.abs(t_rel - r_pred) / float(tau)


def neg_squared_position_score(r_pred: torch.Tensor, t_rel: torch.Tensor, tau: float = 0.25) -> torch.Tensor:
    """Negative squared-distance position potential."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    tau = float(tau)
    return -((t_rel - r_pred) ** 2) / (2.0 * tau * tau)


def compute_position_score(
    r_pred: torch.Tensor,
    t_rel: torch.Tensor,
    mode: str = "neg_abs",
    tau: float = 0.25,
    **kwargs,
) -> torch.Tensor:
    """Compute a relative-position log-potential.

    Supported modes:
        - "neg_abs":     -abs(t_rel - r_pred) / tau
        - "neg_squared": -((t_rel - r_pred)^2) / (2 tau^2)
        - "none":        zero potential
    """
    if mode == "neg_abs":
        return neg_abs_position_score(r_pred, t_rel, tau=tau)
    if mode == "neg_squared":
        return neg_squared_position_score(r_pred, t_rel, tau=tau)
    if mode == "none":
        return torch.zeros_like(r_pred + t_rel)
    raise ValueError(f"Unknown position score mode: {mode!r}")
