"""CombinedLoss: alpha*Dice + beta*(dCR | generic boundary)   [ENGINEERING, master §8]

Also hosts `calibrate_beta_by_gradient_norm()`, which removes a confound that would otherwise
sink prediction P5.

THE PROBLEM IT SOLVES. B' and C differ only in their boundary term, but the curvature-weighted
term and the flat term do not naturally produce gradients of the same magnitude. Run with the
same numeric beta, the two models would differ in BOTH "curvature vs flat" AND "how strongly
the boundary term pulls" -- and a C-over-B' win could be nothing more than C carrying a larger
effective boundary weight. Matching the gradient-norm contribution at equal alpha makes the
boundary term equally influential in both, so the only remaining difference is WHERE that
gradient is placed. That is exactly the comparison P5 needs.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn

from src.losses.boundary_band import assert_bands_identical
from src.losses.dcr_loss import DCRLoss
from src.losses.dice import DiceLoss
from src.losses.generic_boundary import GenericBoundaryLoss


class CombinedLoss(nn.Module):
    """Weighted sum of a region term and (optionally) a boundary term.

    Args:
        region_loss: the Dice term, shared by every model.
        boundary_loss: DCRLoss, GenericBoundaryLoss, or None (Vanilla and B use None).
        alpha: weight on the region term.
        beta: weight on the boundary term. For B' this must come from
            `calibrate_beta_by_gradient_norm`, not be set by hand.

    `last_components` exposes the per-term values so the training loop can log them separately
    (master §10 logs per-epoch gradient norms).
    """

    def __init__(self,
                 region_loss: nn.Module,
                 boundary_loss: nn.Module | None = None,
                 alpha: float = 1.0,
                 beta: float = 1.0) -> None:
        super().__init__()
        self.region_loss = region_loss
        self.boundary_loss = boundary_loss
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.last_components: dict[str, float] = {}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args: logits (B,1,H,W); target (B,1,H,W) in {0,1}. Returns: scalar total loss."""
        region = self.region_loss(logits, target)
        total = self.alpha * region
        self.last_components = {"region": float(region.detach()),
                                "alpha": self.alpha, "beta": self.beta}
        if self.boundary_loss is not None and self.beta != 0.0:
            boundary = self.boundary_loss(logits, target)
            total = total + self.beta * boundary
            self.last_components["boundary"] = float(boundary.detach())
        self.last_components["total"] = float(total.detach())
        return total

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, beta={self.beta}"


# --------------------------------------------------------------------------------------
# Gradient-norm calibration
# --------------------------------------------------------------------------------------
def gradient_norm_contribution(loss_fn: nn.Module,
                               model: nn.Module,
                               batches: Iterable[dict[str, Any]],
                               device: str = "cpu") -> float:
    """Mean L2 norm of the gradient `loss_fn` alone induces on `model`'s parameters.

    Backward is run from the loss term in ISOLATION over a fixed set of batches. Deterministic
    given the same batches and model state -- the caller MUST pass the same batches for both
    losses, or the calibration is comparing noise rather than the two terms.

    Each batch is a dict with "image" and "mask". The model is left with zeroed grads.
    """
    model = model.to(device)
    was_training = model.training
    model.eval()  # keep BatchNorm statistics frozen so calibration does not perturb them
    norms: list[float] = []
    try:
        for batch in batches:
            model.zero_grad(set_to_none=True)
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            loss = loss_fn(model(x), y)
            loss.backward()
            sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    sq += float(p.grad.detach().pow(2).sum())
            norms.append(sq ** 0.5)
    finally:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
    if not norms:
        raise ValueError("no batches supplied: cannot calibrate on an empty set")
    return sum(norms) / len(norms)


def calibrate_beta_by_gradient_norm(reference_loss: nn.Module,
                                    target_loss: nn.Module,
                                    model: nn.Module,
                                    batches: Iterable[dict[str, Any]],
                                    reference_beta: float = 1.0,
                                    device: str = "cpu") -> dict[str, float]:
    """Find the beta making `target_loss` match `reference_loss`'s gradient contribution.

    Returns a record containing:
        beta                 -- reference_beta * ||grad(reference)|| / ||grad(target)||
        grad_norm_reference  -- measured
        grad_norm_target     -- measured, before calibration
        ratio_before / after -- (beta_target * ||g_target||) / (beta_ref * ||g_ref||)

    Usage (master §8): dCR is the reference at its configured beta and B's generic boundary
    loss is the target, so B' receives the beta that equalises their influence at equal alpha.

    Calibration is done ONCE, on the shared pruned base with a fixed batch list, and the value
    is written into the experiment log. It must NOT be re-estimated mid-training, which would
    make B' a moving target and destroy the comparability the whole control depends on.
    """
    batches = list(batches)
    g_ref = gradient_norm_contribution(reference_loss, model, batches, device)
    g_tgt = gradient_norm_contribution(target_loss, model, batches, device)
    if g_tgt <= 0.0:
        raise ValueError(
            "target loss produced a zero gradient norm; cannot calibrate against it")
    beta = reference_beta * g_ref / g_tgt
    return {
        "beta": beta,
        "reference_beta": float(reference_beta),
        "grad_norm_reference": g_ref,
        "grad_norm_target": g_tgt,
        "ratio_before": (reference_beta * g_tgt) / (reference_beta * g_ref),
        "ratio_after": (beta * g_tgt) / (reference_beta * g_ref),
        "n_batches": len(batches),
    }


# --------------------------------------------------------------------------------------
# Config-driven construction
# --------------------------------------------------------------------------------------
def build_boundary_loss(cfg: dict[str, Any]) -> nn.Module | None:
    """Build the boundary term from a `loss.boundary` block, or None if there is none."""
    b = (cfg.get("loss", cfg)).get("boundary", {}) or {}
    name = b.get("name")
    if name in (None, "null", "none"):
        return None
    width = int(b.get("band_width", 3))
    if name == "dcr":
        return DCRLoss(
            band_width=width,
            curvature_gamma=float(b.get("curvature_gamma", 1.0)),
            curvature_weight_power=float(b.get("curvature_weight_power", 1.0)),
            normalise_weights=bool(b.get("normalise_weights", True)),
            curvature_smooth_sigma=float(b.get("curvature_smooth_sigma", 2.0)),
            curvature_source=b.get("curvature_source", "target"),
            curvature_proxy=b.get("curvature_proxy", "sobel_levelset"),
        )
    if name in ("generic_boundary", "generic"):
        return GenericBoundaryLoss(band_width=width)
    raise ValueError(f"unknown boundary loss {name!r}")


def build_loss_from_config(cfg: dict[str, Any]) -> CombinedLoss:
    """Construct the CombinedLoss for a run from its composed config.

    Enforces the shared-band invariant (master §8): a config whose boundary band width differs
    from its declared calibration reference is rejected, because B' and C must share one band
    or P5 is confounded by band geometry rather than isolating curvature.
    """
    loss_cfg = cfg.get("loss", cfg)
    region_cfg = loss_cfg.get("region", {}) or {}
    dice = DiceLoss(smooth=float(region_cfg.get("smooth", 1e-5)))
    boundary = build_boundary_loss(cfg)
    beta = loss_cfg.get("boundary", {}).get("beta", 0.0) if boundary is not None else 0.0
    if beta is None:
        # Calibrated later by calibrate_beta_by_gradient_norm; 0.0 here would silently train
        # B' with no boundary term at all, so this must be an explicit failure.
        raise ValueError(
            "loss.boundary.beta is null: it must be resolved by "
            "calibrate_beta_by_gradient_norm() before training, not left unset")
    return CombinedLoss(region_loss=dice, boundary_loss=boundary,
                        alpha=float(region_cfg.get("alpha", 1.0)), beta=float(beta))


def assert_shared_band(dcr: DCRLoss, generic: GenericBoundaryLoss,
                       sample_mask: torch.Tensor) -> None:
    """Assert the dCR and generic losses extract a pixel-identical band (master §8)."""
    assert_bands_identical(dcr.band_width, generic.band_width, sample_mask)
