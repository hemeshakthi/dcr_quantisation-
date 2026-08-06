"""Shared boundary-band extraction -- the single most safety-critical utility in the study.

master §8 (FROZEN): "dCR and GenericBoundaryLoss MUST share extract_boundary_band() --
pixel-identical except weighting."

This is what makes prediction P5 testable. B' (generic boundary loss) and C (dCR) are meant
to differ ONLY in whether the boundary band is weighted by curvature. If the two losses
extracted their bands even slightly differently, a C-vs-B' difference could be caused by the
band geometry rather than by curvature-awareness, and P5 -- the experiment's whole claim to
curvature-SPECIFICITY -- would be confounded beyond repair.

Therefore: both losses call this function, with the same arguments, and neither may implement
its own band. Any future change here changes both losses identically, by construction.

Everything is differentiable and stays on-device: no numpy in the forward path (master §8).
"""
from __future__ import annotations

import torch


def extract_boundary_band(mask: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Soft boundary band of a binary mask.

    Implemented with differentiable morphology (max-pool dilation and its erosion dual) so
    the band can be recomputed from a PREDICTION as well as from ground truth without
    breaking the graph.

    Args:
        mask: (B, 1, H, W) in [0, 1] -- probabilities or a hard mask.
        width: band thickness in pixels; the dilation/erosion structuring element is
            (2*width+1) square. Configurable, never hardcoded at the call site.

    Returns:
        (B, 1, H, W) band membership in [0, 1], 1 on the boundary and falling to 0 away from
        it. Symmetric about the boundary: it covers `width` px inside and `width` px outside.
    """
    raise NotImplementedError


def signed_distance_band(mask: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Alternative band via a signed distance transform, for ablation only.

    NOT used by either training loss -- `extract_boundary_band` is the frozen one. Provided
    so a reviewer can check that the morphological band is not an artefact of its
    implementation, and kept clearly separate so it can never be silently substituted.
    """
    raise NotImplementedError


def boundary_band_from_logits(logits: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Convenience wrapper: sigmoid the logits, then extract the band.

    Kept as a named function so both losses take the identical path from logits to band.
    """
    raise NotImplementedError
