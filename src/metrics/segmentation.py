"""The three FROZEN metrics: Dice, Boundary F1, HD95 (master §6).

No others. Master §6 explicitly excludes clDice, Betti numbers, and IoU-as-a-separate-metric.

Their roles differ and must not be conflated:
  * Dice        [PAPER]      -- the FOIL. P1 predicts pruning leaves it largely intact while
                                damaging the boundary metrics. It is tested for EQUIVALENCE via
                                TOST (+/-0.02), not superiority: "B and C are equally good by
                                Dice" is the finding, and a superiority test cannot show that.
  * Boundary F1 [LITERATURE] -- 2 px AND 5 px tolerance, reported separately.
  * HD95        [LITERATURE] -- 95th-percentile Hausdorff, standardised morphological-gradient
                                boundary extraction, integer coordinates.

Tolerance in context (Phase 2 measurement): median cell radius is ~10 px at 1024x1024, so 2 px
is ~20% of a radius and 5 px ~50%. This is why A19 rejected 512 px, where 5 px would have
equalled the entire radius and the metric would have stopped discriminating.

Every function returns PER-IMAGE values. Master §6 requires per-image storage for the paired
Wilcoxon tests; aggregating early would destroy the pairing the whole statistical design rests on.
"""
from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Dice coefficient for one image.

    Args:
        pred: (H, W) binary prediction.
        target: (H, W) binary ground truth.

    Returns:
        Dice in [0, 1]. When both masks are empty returns 1.0 -- a perfect prediction of
        "nothing here" -- rather than 0/0.
    """
    raise NotImplementedError


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Standardised 1-px boundary via morphological gradient (master §6).

    ONE implementation, shared by Boundary F1 and HD95, so the two metrics can never disagree
    about where the boundary is. Uses a 3x3 structuring element and integer coordinates.
    """
    raise NotImplementedError


def boundary_f1(pred: np.ndarray, target: np.ndarray,
                tolerance_px: float = 2.0) -> dict[str, float]:
    """Boundary F1 at a given pixel tolerance.

    A predicted boundary pixel counts as matched if a ground-truth boundary pixel lies within
    `tolerance_px` (Euclidean, via distance transform), and vice versa.

    Returns:
        {"boundary_f1", "boundary_precision", "boundary_recall", "tolerance_px"}

    Precision and recall are returned alongside F1 because they fail differently and the
    failure taxonomy (§12) needs to tell them apart: boundary_smoothing depresses recall on
    high-curvature detail, whereas spurious contours depress precision.
    """
    raise NotImplementedError


def hd95(pred: np.ndarray, target: np.ndarray, spacing: float = 1.0) -> float:
    """95th-percentile symmetric Hausdorff distance, in pixels.

    The 95th percentile rather than the maximum, so a single outlier pixel cannot dominate --
    the standard choice in medical segmentation.

    Returns:
        HD95 in px, or NaN if either mask is empty (undefined, and must NOT be silently
        recorded as 0, which would read as a perfect score).
    """
    raise NotImplementedError


def compute_all_metrics(pred: np.ndarray, target: np.ndarray,
                        tolerances: tuple[float, ...] = (2.0, 5.0)) -> dict[str, float]:
    """All frozen metrics for one image, as a flat dict ready for a per-image CSV row."""
    raise NotImplementedError
