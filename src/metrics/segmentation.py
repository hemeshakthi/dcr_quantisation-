"""The three FROZEN metrics: Dice, Boundary F1, HD95 (master §6).

No others. Master §6 explicitly excludes clDice, Betti numbers, and IoU-as-a-separate-metric.

Their roles differ and must not be conflated:
  * Dice        [PAPER]      -- the FOIL. P1 predicts pruning leaves it largely intact while
                                damaging the boundary metrics. It is tested for EQUIVALENCE via
                                TOST (+/-0.02), not superiority: a failed superiority test
                                cannot establish that Dice is preserved.
  * Boundary F1 [LITERATURE] -- 2 px AND 5 px tolerance, reported separately.
  * HD95        [LITERATURE] -- 95th-percentile symmetric Hausdorff, standardised
                                morphological-gradient boundary extraction, integer coordinates.

Tolerance in context (Phase 2 measurement): median cell radius is ~10 px at 1024x1024, so 2 px
is ~20% of a radius and 5 px ~50%. A19 rejected 512 px precisely because 5 px would then have
equalled a whole radius and the metric would have stopped discriminating.

Every function returns PER-IMAGE values. Master §6 requires per-image storage for the paired
Wilcoxon tests; aggregating early would destroy the pairing the statistical design rests on.

scipy.ndimage is used rather than cv2 so that ONE boundary definition serves both Boundary F1
and HD95 -- if they disagreed about where the boundary is, the two metrics would not be
measuring the same object.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# 8-connectivity, matching the connected-component convention used elsewhere in the study.
_STRUCT = np.ones((3, 3), dtype=bool)


def _as_bool(x: np.ndarray) -> np.ndarray:
    return np.asarray(x) > 0


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Dice coefficient for one image.

    Both masks empty returns 1.0 -- a correct prediction of "nothing here" -- rather than 0/0.
    """
    p, t = _as_bool(pred), _as_bool(target)
    inter = float(np.logical_and(p, t).sum())
    card = float(p.sum() + t.sum())
    return (2.0 * inter + smooth) / (card + smooth)


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Standardised 1-px boundary via morphological gradient (master §6).

    ONE implementation, shared by Boundary F1 and HD95. Uses the INNER boundary
    (mask minus its erosion) so boundary pixels always lie inside the object and integer
    coordinates are preserved.
    """
    m = _as_bool(mask)
    if not m.any():
        return np.zeros_like(m)
    eroded = ndimage.binary_erosion(m, structure=_STRUCT, border_value=0)
    return m & ~eroded


def boundary_f1(pred: np.ndarray, target: np.ndarray,
                tolerance_px: float = 2.0) -> dict[str, float]:
    """Boundary F1 at a given pixel tolerance.

    A predicted boundary pixel counts as matched if a ground-truth boundary pixel lies within
    `tolerance_px` (Euclidean, via distance transform), and vice versa.

    Precision and recall are returned alongside F1 because they fail differently and the §12
    taxonomy needs to tell them apart: boundary_smoothing depresses recall on high-curvature
    detail, whereas spurious contours depress precision.
    """
    bp, bt = extract_boundary(pred), extract_boundary(target)
    np_, nt = int(bp.sum()), int(bt.sum())
    out = {"tolerance_px": float(tolerance_px),
           "n_boundary_pred": np_, "n_boundary_gt": nt}

    if np_ == 0 and nt == 0:
        out.update(boundary_f1=1.0, boundary_precision=1.0, boundary_recall=1.0)
        return out
    if np_ == 0 or nt == 0:
        out.update(boundary_f1=0.0,
                   boundary_precision=0.0 if np_ else 1.0,
                   boundary_recall=0.0 if nt else 1.0)
        return out

    dist_to_gt = ndimage.distance_transform_edt(~bt)
    dist_to_pred = ndimage.distance_transform_edt(~bp)
    precision = float((dist_to_gt[bp] <= tolerance_px).mean())
    recall = float((dist_to_pred[bt] <= tolerance_px).mean())
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    out.update(boundary_f1=f1, boundary_precision=precision, boundary_recall=recall)
    return out


def hd95(pred: np.ndarray, target: np.ndarray, percentile: float = 95.0) -> float:
    """95th-percentile symmetric Hausdorff distance, in pixels.

    The 95th percentile rather than the maximum, so a single outlier pixel cannot dominate.

    Returns NaN when either mask is empty: the distance is undefined there and must NOT be
    silently recorded as 0, which would read as a perfect score.
    """
    bp, bt = extract_boundary(pred), extract_boundary(target)
    if not bp.any() or not bt.any():
        return float("nan")
    d_pt = ndimage.distance_transform_edt(~bt)[bp]
    d_tp = ndimage.distance_transform_edt(~bp)[bt]
    return float(max(np.percentile(d_pt, percentile), np.percentile(d_tp, percentile)))


def compute_all_metrics(pred: np.ndarray, target: np.ndarray,
                        tolerances: tuple[float, ...] = (2.0, 5.0),
                        percentile: float = 95.0) -> dict[str, float]:
    """All frozen metrics for one image, flat, ready for a per-image CSV row."""
    out: dict[str, float] = {"dice": dice_score(pred, target)}
    for tol in tolerances:
        r = boundary_f1(pred, target, tol)
        tag = f"{int(tol)}px"
        out[f"boundary_f1_{tag}"] = r["boundary_f1"]
        out[f"boundary_precision_{tag}"] = r["boundary_precision"]
        out[f"boundary_recall_{tag}"] = r["boundary_recall"]
    out["hd95"] = hd95(pred, target, percentile)
    out["n_instances_pred"] = int(ndimage.label(_as_bool(pred), structure=_STRUCT)[1])
    out["n_instances_gt"] = int(ndimage.label(_as_bool(target), structure=_STRUCT)[1])
    return out
