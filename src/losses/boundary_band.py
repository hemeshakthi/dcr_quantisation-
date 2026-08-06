"""Shared boundary-band extraction -- the most safety-critical utility in the study.

master §8 (FROZEN): "dCR and GenericBoundaryLoss MUST share extract_boundary_band() --
pixel-identical except weighting."

This is what makes prediction P5 testable. B' (generic boundary loss) and C (dCR) are meant to
differ ONLY in whether the boundary band is weighted by curvature. If the two losses extracted
their bands even slightly differently, a C-vs-B' difference could be caused by band geometry
rather than by curvature-awareness, and P5 -- the experiment's whole claim to curvature
SPECIFICITY -- would be confounded beyond repair.

Therefore both losses call this one function, and neither implements its own band. Any future
change here changes both losses identically, by construction.

Everything is differentiable and stays on-device: no numpy in the forward path (master §8).
The band is a morphological gradient built from max-pooling, whose subgradients let the band be
recomputed from a PREDICTION as well as from ground truth without breaking the graph.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def extract_boundary_band(mask: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Soft boundary band of a mask, via differentiable morphology.

    band = dilate(mask, w) - erode(mask, w), i.e. the morphological gradient. For a binary
    mask this is 1 within `width` px either side of the boundary and 0 elsewhere; the band is
    SYMMETRIC, covering `width` px inside and `width` px outside.

    Erosion is the max-pool dual, -maxpool(-x). PyTorch pads max-pool with -inf, so border
    windows simply ignore the padding rather than inventing background there -- which would
    otherwise paint a spurious band around the whole image edge.

    Args:
        mask: (B, 1, H, W) in [0, 1] -- hard mask or probabilities.
        width: band half-thickness in pixels. Configurable, never hardcoded at the call site,
            because B' and C must be given the SAME value.

    Returns:
        (B, 1, H, W) band membership in [0, 1].
    """
    if mask.dim() != 4:
        raise ValueError(f"expected (B,1,H,W), got {tuple(mask.shape)}")
    if width < 1:
        raise ValueError(f"band width must be >= 1, got {width}")
    k = 2 * int(width) + 1
    p = int(width)
    dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=p)
    eroded = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=p)
    return (dilated - eroded).clamp(0.0, 1.0)


def boundary_band_from_logits(logits: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Sigmoid the logits, then extract the band.

    A named function so both losses take the IDENTICAL path from logits to band -- the
    conversion is part of the shared contract, not a per-loss detail.
    """
    return extract_boundary_band(torch.sigmoid(logits), width=width)


def band_mean(values: torch.Tensor, band: torch.Tensor,
              eps: float = 1e-8) -> torch.Tensor:
    """Per-image mean of `values` over the band. Returns (B, 1, 1, 1) for broadcasting.

    Used for the scale-free normalisation in DCRLoss. Guarded against an empty band (an image
    with no foreground at all), which would otherwise be 0/0.
    """
    num = (values * band).sum(dim=(1, 2, 3), keepdim=True)
    den = band.sum(dim=(1, 2, 3), keepdim=True)
    return num / (den + eps)


def band_quantile(values: torch.Tensor, band: torch.Tensor, q: float = 0.95,
                  eps: float = 1e-8) -> torch.Tensor:
    """Per-image quantile of `values` restricted to the band. Returns (B,1,1,1).

    Used to normalise curvature into a bounded range. A quantile rather than the max because a
    single pixel of numerical noise would otherwise set the scale for the whole image; images
    with an empty band fall back to `eps`.
    """
    out = []
    for b in range(values.shape[0]):
        sel = values[b][band[b] > 0]
        out.append(torch.quantile(sel, q) if sel.numel() > 0
                   else values.new_tensor(eps))
    return torch.stack(out).view(-1, 1, 1, 1)


def assert_bands_identical(width_a: int, width_b: int,
                           sample_mask: torch.Tensor,
                           atol: float = 0.0) -> None:
    """Assert two configured band widths produce a PIXEL-IDENTICAL band.

    Master §8 requires it. Checked rather than assumed, because a silent divergence would
    confound P5 without raising any error.

    Raises:
        AssertionError: if the widths differ or the extracted bands differ anywhere.
    """
    if width_a != width_b:
        raise AssertionError(
            f"band widths differ ({width_a} vs {width_b}): B' and C must share one band, "
            f"or the C-vs-B' comparison (P5) is confounded by band geometry")
    a = extract_boundary_band(sample_mask, width_a)
    b = extract_boundary_band(sample_mask, width_b)
    if not torch.allclose(a, b, atol=atol, rtol=0.0):
        raise AssertionError(
            f"bands differ despite equal width {width_a}: max abs diff "
            f"{(a - b).abs().max().item()}")
