"""dCR -- discriminative Curvature Regularization loss.

*** THIS IS A SURROGATE, NOT THE PAPER'S EQUATION. See implementation_assumptions.md A06. ***

The dCR paper (Asha & Gopakumar 2026) is confirmed closed-access, with no preprint and no
public code (docs/source_review.md §§1-2), so the exact "curvature embedding space"
formulation could not be recovered. Master §8 therefore specifies the closest defensible
stand-in: a boundary-band term weighted by a differentiable curvature proxy.

What this surrogate is NOT:
  * NOT bending loss (Wang 2020, paper ref 30)
  * NOT the OCT curvature loss (Xing 2022, paper ref 29)
Both are cited ancestors that the paper explicitly differentiates itself from, so copying
either would misattribute someone else's loss to dCR.

Consequently every result produced with this class is a result for "a curvature-weighted
boundary loss in the spirit of dCR", NOT a reproduction of the paper's method, and must be
reported as such. REPLACE THIS THE MOMENT THE PROFESSOR SUPPLIES THE REAL EQUATION OR CODE
(master Golden Rule 8), then re-run every affected phase.

Fully differentiable, no numpy in forward (master §8).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def curvature_proxy(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable per-pixel curvature proxy.

    Master §8 specifies a Sobel-gradient-magnitude proxy on the prediction. The divergence of
    the normalised gradient field, div(grad p / |grad p|), is the textbook level-set mean
    curvature and is the intended interpretation; the exact discretisation is recorded in
    A06 and must be stated in the paper.

    Args:
        probs: (B,1,H,W) in [0,1].
        eps: guards the normalisation where the gradient vanishes (flat regions), preventing
            a divide-by-zero that would otherwise produce NaN gradients over most of the image.

    Returns:
        (B,1,H,W) non-negative curvature magnitude, in 1/px.
    """
    raise NotImplementedError


class DCRLoss(nn.Module):
    """Curvature-weighted boundary loss (surrogate for dCR).

    Computes BCE on the boundary band, weighted per-pixel by the curvature proxy, so gradient
    is concentrated on high-curvature boundary pixels -- the mechanism master §2 predicts will
    recover pruning damage.

    Args:
        band_width: boundary band thickness in px. MUST match the value given to
            GenericBoundaryLoss, or the B'-vs-C comparison (P5) is confounded; the CombinedLoss
            builder is responsible for passing one shared value.
        curvature_weight_power: exponent on the curvature proxy. 1.0 = linear weighting.
        normalise_weights: divide weights by their per-image mean so the loss magnitude does
            not scale with how curved the image happens to be. Without this, high-curvature
            images would dominate the batch gradient purely through the weighting -- which
            would be indistinguishable from the effect under test.
        curvature_source: "prediction" | "target" | "both". Weighting by the TARGET's
            curvature keeps the weights fixed and is the more stable choice; weighting by the
            PREDICTION lets the model reduce its own loss by flattening boundaries, which is a
            degenerate solution worth guarding against. Recorded in the config, not chosen here.
        from_logits: the head emits raw logits (master §7).
    """

    def __init__(self,
                 band_width: int = 3,
                 curvature_weight_power: float = 1.0,
                 normalise_weights: bool = True,
                 curvature_source: str = "target",
                 from_logits: bool = True) -> None:
        super().__init__()
        self.band_width = band_width
        self.curvature_weight_power = curvature_weight_power
        self.normalise_weights = normalise_weights
        self.curvature_source = curvature_source
        self.from_logits = from_logits

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args: logits (B,1,H,W); target (B,1,H,W) in {0,1}. Returns: scalar loss."""
        raise NotImplementedError

    def weight_map(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the per-pixel weight map actually applied, for figures and auditing.

        Exposed so the visual gate and the paper can SHOW where dCR puts its gradient, rather
        than asserting it.
        """
        raise NotImplementedError
