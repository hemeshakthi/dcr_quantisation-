"""HD95 -- 95th-percentile symmetric Hausdorff distance (master §6) [LITERATURE].

The **95th percentile**, NOT the maximum: a single outlier boundary pixel must not dominate the
score (verified by the `percentile: float = 95.0` default and the max-of-two-95th-percentiles
formula below -- there is no code path that returns a raw maximum).

Uses the SAME boundary definition as `boundary_f1.py::extract_boundary` (standardised
morphological-gradient extraction, integer coordinates) -- sharing one boundary definition is
what makes Boundary F1 and HD95 measurements of the same object, not two different ones.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from src.metrics.boundary_f1 import extract_boundary


def hd95(pred: np.ndarray, target: np.ndarray, percentile: float = 95.0) -> float:
    """95th-percentile symmetric Hausdorff distance, in pixels.

    Symmetric: computed both pred->target and target->pred (via two distance transforms), then
    the max of the two 95th percentiles is taken, so damage in either direction is caught.

    Returns NaN when either mask is empty: the distance is undefined there and must NOT be
    silently recorded as 0, which would read as a perfect score.
    """
    bp, bt = extract_boundary(pred), extract_boundary(target)
    if not bp.any() or not bt.any():
        return float("nan")
    d_pt = ndimage.distance_transform_edt(~bt)[bp]
    d_tp = ndimage.distance_transform_edt(~bp)[bt]
    return float(max(np.percentile(d_pt, percentile), np.percentile(d_tp, percentile)))
