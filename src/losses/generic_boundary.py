"""Generic boundary loss -- the control that isolates curvature-specificity (Model B').

[LITERATURE] Drawn from the boundary-loss family the dCR paper itself cites (Kervadec ref 15,
Active Boundary Loss ref 17, Conditional Boundary Loss ref 18, Boundary DoU ref 19).

THE ENTIRE POINT of this class is to be identical to DCRLoss except for one thing: the band is
weighted FLAT instead of by curvature. It therefore:
  * calls the SAME `extract_boundary_band()` (master §8, FROZEN)
  * takes the SAME `band_width`
  * applies the SAME BCE on the band
  * differs ONLY in that every band pixel gets weight 1

If C beats B', the improvement is attributable to curvature-awareness rather than to
boundary-awareness in general -- that is prediction P5, the study's strongest claim. Any
divergence between this class and DCRLoss beyond the weighting silently destroys that
inference, so changes here must be mirrored in dcr_loss.py.

Formally this loss is exactly `DCRLoss(curvature_gamma=0)`; verification asserts that identity
numerically, which is a stronger guarantee than the two implementations merely looking alike.

Equal band and equal formula are still not sufficient on their own: the two terms must also
enter the total loss at matched gradient magnitude, which is
`CombinedLoss.calibrate_beta_by_gradient_norm()`'s job.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.boundary_band import extract_boundary_band


class GenericBoundaryLoss(nn.Module):
    """Flat-weighted boundary-band BCE.

    Args:
        band_width: MUST equal the DCRLoss band_width for the same experiment.
        from_logits: the head emits raw logits (master §7).
    """

    def __init__(self, band_width: int = 3, eps: float = 1e-6,
                 from_logits: bool = True) -> None:
        super().__init__()
        self.band_width = int(band_width)
        self.eps = float(eps)
        self.from_logits = bool(from_logits)

    def weight_map(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-pixel weights actually applied: the band itself, uniformly 1.

        Same signature as DCRLoss.weight_map so the two can be plotted side by side and the
        difference in WHERE gradient lands can be shown rather than asserted.
        """
        return extract_boundary_band(target, self.band_width)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args: logits (B,1,H,W); target (B,1,H,W) in {0,1}. Returns: scalar loss."""
        if logits.shape != target.shape:
            raise ValueError(f"shape mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}")
        weights = self.weight_map(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none") \
            if self.from_logits else \
            F.binary_cross_entropy(logits.clamp(1e-7, 1 - 1e-7), target, reduction="none")
        num = (weights * bce).sum()
        den = weights.sum()
        # Empty band (no foreground): keep the graph connected rather than returning a
        # detached constant -- identical handling to DCRLoss, deliberately.
        if float(den) <= 0.0:
            return logits.sum() * 0.0
        return num / (den + self.eps)

    def extra_repr(self) -> str:
        return f"band_width={self.band_width}, from_logits={self.from_logits}"
