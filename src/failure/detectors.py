"""The six failure detectors (master §12).

corner_collapse, boundary_smoothing, cell_merging, cell_splitting, thin_structure_loss,
curvature_flattening.

Two rules from master §12 that shape this module:
  1. THRESHOLDS ARE LOCKED BEFORE INSPECTION. They live in configs/evaluation.yaml and are
     fixed before any model output is examined. Tuning a threshold after seeing which model it
     favours would manufacture the P6 result.
  2. DETECTORS ARE BLIND TO THE MODEL. Every function takes only (prediction, ground truth)
     and never the model name, so it cannot treat C differently from B.

P6 is then the ratio
    (reduction in curvature-related failures) / (reduction in non-curvature-related failures)
for B -> C. Which detectors count as curvature-related is declared in the config BEFORE
results exist, not chosen afterwards to suit the numbers.

cell_merging and cell_splitting require INSTANCE masks, not binary ones: two touching cells
share one binary blob, so a merge is invisible to a binary-only comparison. This is why Phase 2
preserved instance IDs.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Declared BEFORE any result exists (master §12). Used to compute the P6 ratio.
CURVATURE_RELATED = ("corner_collapse", "boundary_smoothing", "curvature_flattening")
NON_CURVATURE_RELATED = ("cell_merging", "cell_splitting", "thin_structure_loss")


def corner_collapse(pred: np.ndarray, target: np.ndarray,
                    curvature_percentile: float = 90.0,
                    min_drop: float = 0.3) -> dict[str, Any]:
    """High-curvature contour points rounded off in the prediction.

    Compares point curvature at matched boundary locations; a corner has "collapsed" when the
    prediction's local curvature falls by more than `min_drop` relative to ground truth.

    Returns: {"count": int, "sites": [(y, x), ...], "severity": float}
    """
    raise NotImplementedError


def boundary_smoothing(pred: np.ndarray, target: np.ndarray,
                       min_ratio: float = 0.9) -> dict[str, Any]:
    """Global loss of boundary detail: predicted perimeter shorter than ground truth's.

    A smoothed boundary is shorter than the true one at equal area, so the perimeter ratio
    detects it without needing point correspondence.
    """
    raise NotImplementedError


def cell_merging(pred_instances: np.ndarray, gt_instances: np.ndarray,
                 iou_threshold: float = 0.5) -> dict[str, Any]:
    """Two or more GT instances covered by a single predicted component.

    Requires INSTANCE masks: touching cells share a binary blob, so this is undetectable from
    binary masks alone.
    """
    raise NotImplementedError


def cell_splitting(pred_instances: np.ndarray, gt_instances: np.ndarray,
                   iou_threshold: float = 0.5) -> dict[str, Any]:
    """One GT instance broken into several predicted components. The dual of cell_merging."""
    raise NotImplementedError


def thin_structure_loss(pred: np.ndarray, target: np.ndarray,
                        max_width_px: float = 3.0) -> dict[str, Any]:
    """Thin protrusions present in GT but missing from the prediction.

    Relevant here specifically: the high-curvature bin is populated by migrating T cells with
    pointed uropods (Phase 2 spot check), which is exactly the structure most easily lost.
    """
    raise NotImplementedError


def curvature_flattening(pred: np.ndarray, target: np.ndarray,
                         percentile: float = 90.0,
                         min_drop: float = 0.15) -> dict[str, Any]:
    """Drop in the image-level 90th-percentile boundary curvature.

    Master §12 permits merging this with corner_collapse if they prove redundant, PROVIDED the
    merge is documented. They differ in scope: corner_collapse is per-site and local,
    curvature_flattening is the image-level aggregate that mirrors the §11 stratification
    statistic. Redundancy is an empirical question to settle on real outputs, and the answer
    must be recorded either way.
    """
    raise NotImplementedError


def run_all_detectors(pred: np.ndarray, target: np.ndarray,
                      pred_instances: np.ndarray | None = None,
                      gt_instances: np.ndarray | None = None,
                      thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run all six detectors on one image.

    Args:
        thresholds: from configs/evaluation.yaml, locked before inspection. Passing None uses
            the signature defaults, which is acceptable only for unit tests.

    Returns:
        One flat dict of per-detector counts and severities, ready for a
        results/failure_counts/ row.
    """
    raise NotImplementedError


def p6_ratio(counts_b: dict[str, int], counts_c: dict[str, int]) -> dict[str, float]:
    """The P6 statistic: curvature-related vs non-curvature-related reduction, B -> C.

    Returns the two reductions and their ratio. A ratio > 1 supports P6 (dCR preferentially
    fixes curvature-related failures). Returns NaN for the ratio when the non-curvature
    reduction is zero -- an undefined ratio must never be reported as a large finite number.
    """
    raise NotImplementedError
