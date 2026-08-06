"""Soft Dice loss (master §8, smooth=1e-5) [PAPER].

The region term shared by every model in the study (Vanilla, A, B, B', C). Master §6 treats
Dice as the FOIL: pruning is predicted to leave it largely intact (P1) while damaging the
boundary metrics, so it must be computed identically everywhere or that contrast is
meaningless.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss on raw logits.

    Args:
        smooth: numerical stabiliser, FROZEN at 1e-5 by master §8. Exposed as an argument so
            it is recorded in the config rather than buried as a literal, but it is not a
            tuning knob.
        from_logits: if True (default) apply sigmoid internally; the U-Net head emits raw
            logits with no activation (master §7).

    Foreground is a small minority class here -- measured at ~3.2% of pixels (Phase 2) -- which
    is precisely why Dice rather than plain BCE is the region term.
    """

    def __init__(self, smooth: float = 1e-5, from_logits: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args: logits (B,1,H,W); target (B,1,H,W) in {0,1}. Returns: scalar loss."""
        raise NotImplementedError


def soft_dice_coefficient(probs: torch.Tensor, target: torch.Tensor,
                          smooth: float = 1e-5) -> torch.Tensor:
    """Per-sample soft Dice coefficient in [0, 1]. Returns shape (B,).

    Separate from the loss so evaluation can reuse the identical formula and the reported
    Dice can never drift from the trained objective.
    """
    raise NotImplementedError
