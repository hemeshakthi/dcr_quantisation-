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
boundary-awareness in general -- that is prediction P5, and it is the study's strongest claim.
Any divergence between this class and DCRLoss beyond the weighting silently destroys that
inference, so changes here must be mirrored in dcr.py or the comparison is invalid.

Note the two losses must also be gradient-matched, which is a separate concern handled by
`CombinedLoss.calibrate_beta_by_gradient_norm()` -- equal band and equal formula are not
enough if the two terms enter the total loss at different magnitudes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GenericBoundaryLoss(nn.Module):
    """Flat-weighted boundary-band BCE.

    Args:
        band_width: MUST equal the DCRLoss band_width for the same experiment.
        from_logits: the head emits raw logits (master §7).
    """

    def __init__(self, band_width: int = 3, from_logits: bool = True) -> None:
        super().__init__()
        self.band_width = band_width
        self.from_logits = from_logits

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args: logits (B,1,H,W); target (B,1,H,W) in {0,1}. Returns: scalar loss."""
        raise NotImplementedError

    def weight_map(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-pixel weights actually applied: the band itself, uniformly 1.

        Same signature as DCRLoss.weight_map so the two can be plotted side by side and the
        difference in where gradient lands can be shown rather than asserted.
        """
        raise NotImplementedError


def assert_bands_identical(dcr_loss, generic_loss, sample_mask: torch.Tensor) -> None:
    """Assert the two losses extract a PIXEL-IDENTICAL band from the same mask.

    Master §8 requires this. It is checked as a test rather than assumed, because a silent
    divergence would confound P5 without producing any visible error.

    Raises:
        AssertionError: if the bands differ anywhere, or the band widths differ.
    """
    raise NotImplementedError
