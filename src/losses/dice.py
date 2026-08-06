"""Soft Dice loss (master §8, smooth=1e-5) [PAPER].

The region term shared by every model in the study (Vanilla, A, B, B', C). Master §6 treats
Dice as the FOIL: pruning is predicted to leave it largely intact (P1) while damaging the
boundary metrics, so it must be computed identically everywhere or that contrast is meaningless.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def soft_dice_coefficient(probs: torch.Tensor, target: torch.Tensor,
                          smooth: float = 1e-5) -> torch.Tensor:
    """Per-sample soft Dice coefficient. Returns shape (B,).

    Separate from the loss so evaluation can reuse the identical formula and the reported Dice
    can never drift from the trained objective.

    `smooth` on both numerator and denominator makes the empty-vs-empty case evaluate to 1.0
    (a correct prediction of "nothing here") rather than 0/0.
    """
    if probs.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(probs.shape)} vs {tuple(target.shape)}")
    dims = tuple(range(1, probs.dim()))
    intersection = (probs * target).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + target.sum(dim=dims)
    return (2.0 * intersection + smooth) / (cardinality + smooth)


class DiceLoss(nn.Module):
    """Soft Dice loss on raw logits.

    Args:
        smooth: numerical stabiliser, FROZEN at 1e-5 by master §8. Exposed so it is recorded in
            the config rather than buried as a literal, but it is not a tuning knob.
        from_logits: if True (default) apply sigmoid internally; the U-Net head emits raw
            logits with no activation (master §7).

    Foreground is a small minority class here -- measured at ~3.2% of pixels in Phase 2 -- which
    is precisely why Dice rather than plain BCE is the region term.
    """

    def __init__(self, smooth: float = 1e-5, from_logits: bool = True) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.from_logits = bool(from_logits)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args: logits (B,1,H,W); target (B,1,H,W) in {0,1}. Returns: scalar loss."""
        probs = torch.sigmoid(logits) if self.from_logits else logits
        return 1.0 - soft_dice_coefficient(probs, target, self.smooth).mean()

    def extra_repr(self) -> str:
        return f"smooth={self.smooth}, from_logits={self.from_logits}"
