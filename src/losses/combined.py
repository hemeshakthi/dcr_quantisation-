"""CombinedLoss: alpha*Dice + beta*(dCR | generic boundary)   [ENGINEERING, master §8]

Also hosts `calibrate_beta_by_gradient_norm()`, which exists to remove a confound that would
otherwise sink prediction P5.

THE PROBLEM IT SOLVES. B' and C differ only in their boundary term, but the curvature-weighted
term and the flat term do not naturally produce gradients of the same magnitude. If B' were
run with the same numeric beta as C, the two models would differ in BOTH "curvature vs flat"
AND "how strongly the boundary term pulls" -- and a C-over-B' win could be nothing more than C
having a larger effective boundary weight. Matching the gradient-norm contribution at equal
alpha makes the boundary term equally influential in both, so the only remaining difference is
WHERE that gradient is placed. That is exactly the comparison P5 needs.
"""
from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """Weighted sum of a region term and (optionally) a boundary term.

    Args:
        region_loss: the Dice term, shared by every model.
        boundary_loss: DCRLoss, GenericBoundaryLoss, or None (Vanilla and B use None).
        alpha: weight on the region term.
        beta: weight on the boundary term. For B' this should come from
            `calibrate_beta_by_gradient_norm`, not be set by hand.

    Returns the scalar total; `last_components` exposes the per-term values so the training
    loop can log them separately (master §10 logs per-epoch gradient norms).
    """

    def __init__(self,
                 region_loss: nn.Module,
                 boundary_loss: nn.Module | None = None,
                 alpha: float = 1.0,
                 beta: float = 1.0) -> None:
        super().__init__()
        self.region_loss = region_loss
        self.boundary_loss = boundary_loss
        self.alpha = alpha
        self.beta = beta
        self.last_components: dict[str, float] = {}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args: logits (B,1,H,W); target (B,1,H,W) in {0,1}. Returns: scalar total loss."""
        raise NotImplementedError


def gradient_norm_contribution(loss_fn: nn.Module,
                               model: nn.Module,
                               batches: list[dict[str, Any]],
                               device: str = "cpu") -> float:
    """Mean L2 norm of the gradient that `loss_fn` alone induces on `model`'s parameters.

    Runs backward from the loss term in isolation over a fixed set of batches and returns the
    mean gradient norm. Deterministic given the same batches and model state -- the caller must
    pass the SAME batches for both losses, or the calibration compares noise.
    """
    raise NotImplementedError


def calibrate_beta_by_gradient_norm(reference_loss: nn.Module,
                                    target_loss: nn.Module,
                                    model: nn.Module,
                                    batches: list[dict[str, Any]],
                                    reference_beta: float = 1.0,
                                    device: str = "cpu") -> float:
    """Find the beta making `target_loss` match `reference_loss`'s gradient contribution.

    Returns:
        beta such that beta * ||grad(target_loss)|| == reference_beta * ||grad(reference_loss)||.

    Usage (master §8): dCR is the reference at its configured beta, and B's generic boundary
    loss is the target, so B' receives the beta that equalises their influence at equal alpha.

    The calibration is done ONCE, on the untrained (or shared pruned) initial weights, with a
    fixed batch list, and the resulting value is written into the experiment log -- it must not
    be re-estimated mid-training, which would make B' a moving target.
    """
    raise NotImplementedError


def build_loss_from_config(cfg: dict[str, Any]) -> CombinedLoss:
    """Construct the CombinedLoss for a run from its composed config.

    Enforces the shared-band invariant: when both a dCR and a generic boundary term are
    constructible from the config, they receive the SAME band_width (master §8).
    """
    raise NotImplementedError
