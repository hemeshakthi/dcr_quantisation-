"""Boundary F1 at 2px and 5px tolerance (master §6) [LITERATURE].

Tolerance in context (Phase 2 measurement): median cell radius is ~10 px at 1024x1024, so 2 px
is ~20% of a radius and 5 px ~50%. A19 rejected 512 px precisely because 5 px would then have
equalled a whole radius and the metric would have stopped discriminating.

`extract_boundary` is the ONE boundary definition shared with `hausdorff.py::hd95` -- if the two
metrics disagreed about where the boundary is, they would not be measuring the same object.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# 8-connectivity, matching the connected-component convention used elsewhere in the study.
_STRUCT = np.ones((3, 3), dtype=bool)


def _as_bool(x: np.ndarray) -> np.ndarray:
    return np.asarray(x) > 0


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
