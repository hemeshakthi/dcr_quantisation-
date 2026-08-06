"""dCR -- discriminative Curvature Regularization loss.

*** THIS IS A SURROGATE, NOT THE PAPER'S EQUATION. See implementation_assumptions.md A06. ***

The dCR paper (Asha & Gopakumar 2026) is confirmed closed-access, with no preprint and no
public code (docs/source_review.md §§1-2), so the exact "curvature embedding space"
formulation could not be recovered. Master §8 specifies the closest defensible stand-in: a
boundary-band BCE weighted by a differentiable curvature proxy.

Every result produced with this class is a result for "a curvature-weighted boundary loss in
the spirit of dCR", NOT a reproduction of the paper's method, and must be reported as such.

HOW TO REPLACE IT WITH THE REAL EQUATION (one function)
-------------------------------------------------------
The curvature weighting is isolated behind a registry. To adopt the professor's formulation:
    1. write `def dcr_official(mask, eps): ...` returning a (B,1,H,W) non-negative map
    2. `register_curvature_proxy("official", dcr_official)`
    3. set `loss.boundary.curvature_proxy: official` in the model YAML
Nothing else changes -- not this class, not CombinedLoss, not the configs' structure. Then
re-label A06 [ASSUMPTION] -> [PAPER]/[OFFICIAL] and re-run every affected phase
(Golden Rule 8).

WHAT THIS SURROGATE IS NOT
--------------------------
  * NOT bending loss (Wang 2020, paper ref 30). Bending loss penalises the integrated squared
    curvature of the predicted contour -- an unconditional smoothness prior that pushes ALL
    boundaries toward straightness. dCR does the opposite in spirit: it does not penalise
    curvature, it uses curvature to decide WHERE the segmentation error matters most. Applying
    bending loss here would actively fight P4, since it would suppress exactly the
    high-curvature detail the study is trying to recover.
  * NOT the OCT curvature loss (Xing 2022, paper ref 29). That loss encodes a strong domain
    prior -- retinal layers are near-horizontal, smoothly varying surfaces -- as a penalty on
    vertical curvature along a single-valued boundary per column. Cell boundaries are closed,
    multi-valued contours with genuine corners, so the prior is not merely inapplicable but
    wrong-signed for this data.
Both are cited ANCESTORS that the paper explicitly differentiates itself from, so copying
either would misattribute someone else's loss to dCR.

WHY THE PROXY IS LEVEL-SET CURVATURE, NOT RAW |grad p|
-----------------------------------------------------
Master §8 words the proxy as "Sobel-gradient magnitude on prediction". Taken literally as
|grad p|, that quantity is roughly CONSTANT along a boundary -- it measures edge strength, not
shape -- so it is near-identical at a sharp corner and along a straight edge. Weighting by it
would make dCR a flat boundary loss up to a scale factor, i.e. numerically almost the same
objective as GenericBoundaryLoss, and the B'-vs-C comparison (P5) would be testing nothing.
The measured evidence is in audit/verify_phase5.py: raw |grad p| gives a corner/edge weight
ratio of ~1.0, while level-set curvature gives a strongly separated ratio.

We therefore build the proxy FROM Sobel derivatives, as master §8 directs, but form the
textbook level-set curvature
        kappa = div( grad p / |grad p| )
which is large at corners and near zero along straight edges. `sobel_magnitude` is retained in
the registry so the literal reading remains available and auditable, but it is NOT the default.
Recorded in A06.

Fully differentiable, no numpy in forward (master §8).
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.boundary_band import (band_mean, band_quantile,
                                      extract_boundary_band)

# --------------------------------------------------------------------------------------
# Sobel kernels, registered as buffers by the module that uses them so they follow .to(device).
# --------------------------------------------------------------------------------------
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0],
                         [-2.0, 0.0, 2.0],
                         [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0],
                         [0.0, 0.0, 0.0],
                         [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0


def _sobel_grads(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sobel dx, dy with replicate padding (so the image border adds no artificial edge)."""
    kx = _SOBEL_X.to(device=x.device, dtype=x.dtype)
    ky = _SOBEL_Y.to(device=x.device, dtype=x.dtype)
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(xp, kx), F.conv2d(xp, ky)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur. Differentiable, torch-only, no numpy (master §8).

    Required before differentiating a HARD binary mask. Curvature of an unsmoothed binary mask
    is the second derivative of a step function: it reports the pixel staircase, not the shape.
    MEASURED on a synthetic square (audit/verify_phase5.py): at sigma=0 a corner scores kappa
    0.317 against 0.333 on a straight edge -- a ratio of 0.95, i.e. no discrimination at all --
    while at sigma=2 the straight edge falls to EXACTLY 0.000 and only the corners survive.
    This is the same effect A27 documented for the §11 contour statistic, arriving here through
    a different route.
    """
    if sigma <= 0:
        return x
    radius = max(1, int(3.0 * sigma))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(t * t) / (2.0 * sigma * sigma))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), k.view(1, 1, -1, 1))
    return x


def sobel_magnitude(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Raw Sobel gradient magnitude |grad p| -- master §8's LITERAL wording.

    Retained for auditability and ablation. NOT the default: it measures edge STRENGTH, which
    is near-constant along a boundary, so it cannot discriminate a corner from a straight edge
    (see the module docstring).
    """
    gx, gy = _sobel_grads(mask)
    return torch.sqrt(gx * gx + gy * gy + eps * eps)


def sobel_levelset_curvature(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """|kappa| = |div(grad p / |grad p|)|, the level-set curvature, from Sobel derivatives.

    Large at corners and junctions, near zero along straight edges -- which is the property the
    weighting needs. `eps` inside the magnitude keeps the normalised field ~0 in flat regions
    instead of dividing by zero, so the divergence there is ~0 rather than NaN.

    Returns: (B,1,H,W) non-negative curvature magnitude, in 1/px.
    """
    gx, gy = _sobel_grads(mask)
    mag = torch.sqrt(gx * gx + gy * gy + eps * eps)
    nx, ny = gx / mag, gy / mag
    nxx, _ = _sobel_grads(nx)
    _, nyy = _sobel_grads(ny)
    return (nxx + nyy).abs()


# --------------------------------------------------------------------------------------
# Swappable registry -- the single seam where the real dCR equation drops in.
# --------------------------------------------------------------------------------------
CurvatureProxy = Callable[[torch.Tensor, float], torch.Tensor]

_CURVATURE_PROXIES: dict[str, CurvatureProxy] = {
    "sobel_levelset": sobel_levelset_curvature,   # DEFAULT (A06 surrogate)
    "sobel_magnitude": sobel_magnitude,           # master §8's literal reading; ablation only
}


def register_curvature_proxy(name: str, fn: CurvatureProxy, overwrite: bool = False) -> None:
    """Register a curvature proxy. THIS IS THE HOOK FOR THE PAPER'S REAL EQUATION."""
    if name in _CURVATURE_PROXIES and not overwrite:
        raise ValueError(f"curvature proxy {name!r} already registered; pass overwrite=True")
    _CURVATURE_PROXIES[name] = fn


def get_curvature_proxy(name: str) -> CurvatureProxy:
    if name not in _CURVATURE_PROXIES:
        raise KeyError(f"unknown curvature proxy {name!r}; "
                       f"registered: {sorted(_CURVATURE_PROXIES)}")
    return _CURVATURE_PROXIES[name]


def available_curvature_proxies() -> list[str]:
    return sorted(_CURVATURE_PROXIES)


class DCRLoss(nn.Module):
    """Curvature-weighted boundary loss (SURROGATE for dCR -- A06).

    BCE on the shared boundary band, weighted per pixel by
        w = 1 + gamma * (kappa / mean_band(kappa)) ** power
    so gradient concentrates on high-curvature boundary pixels -- the mechanism master §2
    predicts will recover pruning damage.

    Note the deliberate structure: at `gamma = 0` the weights collapse to 1 and this loss
    becomes EXACTLY GenericBoundaryLoss. That is not a coincidence but a design property -- it
    makes the control an exact special case of the treatment, so the B'-vs-C comparison isolates
    the curvature term alone. Verification asserts this identity numerically.

    Args:
        band_width: band half-thickness. MUST match GenericBoundaryLoss's, or P5 is confounded;
            `build_loss_from_config` enforces it.
        curvature_gamma: weighting strength. A maximally-curved band pixel receives at most
            (1 + gamma) times the weight of a straight-edge pixel, so gamma is directly
            interpretable. 0 reduces EXACTLY to the flat loss.
        curvature_weight_power: exponent on the normalised curvature.
        curvature_smooth_sigma: Gaussian sigma applied to the mask BEFORE differentiating.
            Non-negotiable for hard binary targets -- see `gaussian_blur`. Defaults to 2.0,
            matching the A27 override for the same underlying reason. Set 0 only if a
            replacement proxy does its own smoothing.
        normalise_weights: rescale so the mean weight over the band is 1. Without it the loss
            magnitude would scale with how curved the image happens to be, letting curved
            images dominate the batch gradient through the weighting alone -- which would be
            indistinguishable from the effect under test.
        curvature_source: "target" | "prediction" | "both". "target" keeps the weights constant
            w.r.t. the prediction and is the default; "prediction" lets the model lower its own
            loss by FLATTENING boundaries, a degenerate solution that would work against P4.
        curvature_proxy: registry key. Swap to the official equation here.
        from_logits: the head emits raw logits (master §7).
    """

    def __init__(self,
                 band_width: int = 3,
                 curvature_gamma: float = 1.0,
                 curvature_weight_power: float = 1.0,
                 curvature_smooth_sigma: float = 2.0,
                 curvature_quantile: float = 0.95,
                 normalise_weights: bool = True,
                 curvature_source: str = "target",
                 curvature_proxy: str = "sobel_levelset",
                 eps: float = 1e-6,
                 from_logits: bool = True) -> None:
        super().__init__()
        if curvature_source not in ("target", "prediction", "both"):
            raise ValueError(f"curvature_source must be target|prediction|both, "
                             f"got {curvature_source!r}")
        self.band_width = int(band_width)
        self.curvature_gamma = float(curvature_gamma)
        self.curvature_weight_power = float(curvature_weight_power)
        self.curvature_smooth_sigma = float(curvature_smooth_sigma)
        self.curvature_quantile = float(curvature_quantile)
        self.normalise_weights = bool(normalise_weights)
        self.curvature_source = curvature_source
        self.curvature_proxy_name = curvature_proxy
        self.curvature_proxy = get_curvature_proxy(curvature_proxy)
        self.eps = float(eps)
        self.from_logits = bool(from_logits)

    # ------------------------------------------------------------------ internals
    def _curvature(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Smooth, then apply the registered proxy. The proxy always receives a SMOOTHED field."""
        sig = self.curvature_smooth_sigma
        if self.curvature_source == "target":
            return self.curvature_proxy(gaussian_blur(target, sig), self.eps)
        if self.curvature_source == "prediction":
            return self.curvature_proxy(gaussian_blur(probs, sig), self.eps)
        return 0.5 * (self.curvature_proxy(gaussian_blur(target, sig), self.eps)
                      + self.curvature_proxy(gaussian_blur(probs, sig), self.eps))

    def weight_map(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """The per-pixel weight actually applied (band included).

        Exposed so figures and audits can SHOW where dCR puts its gradient rather than assert it.
        """
        probs = torch.sigmoid(logits) if self.from_logits else logits
        band = extract_boundary_band(target, self.band_width)
        kappa = self._curvature(probs, target)
        # Normalise by a high quantile OVER THE BAND and clamp to [0,1], so the weight is
        # bounded in [1, 1+gamma]. Normalising by the mean instead would let a few very sharp
        # corners push weights to arbitrary multiples and effectively switch the straight-edge
        # supervision off; a quantile rather than the max keeps one noisy pixel from setting
        # the scale for the whole image.
        scale = band_quantile(kappa.detach(), band, self.curvature_quantile, self.eps)
        k_rel = (kappa / (scale + self.eps)).clamp(0.0, 1.0)
        w = 1.0 + self.curvature_gamma * k_rel ** self.curvature_weight_power
        if self.normalise_weights:
            w = w / (band_mean(w, band, self.eps) + self.eps)
        return w * band

    # ------------------------------------------------------------------ forward
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
        # An image with no foreground at all has an empty band. Returning a plain 0.0 would
        # detach the graph; scaling the logits by zero keeps it connected and differentiable.
        if float(den) <= 0.0:
            return logits.sum() * 0.0
        return num / (den + self.eps)

    def extra_repr(self) -> str:
        return (f"band_width={self.band_width}, gamma={self.curvature_gamma}, "
                f"power={self.curvature_weight_power}, sigma={self.curvature_smooth_sigma}, "
                f"source={self.curvature_source}, proxy={self.curvature_proxy_name}, "
                f"normalise={self.normalise_weights}")
