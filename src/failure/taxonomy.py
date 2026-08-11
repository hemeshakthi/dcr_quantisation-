"""The six failure detectors (master §12).

corner_collapse, boundary_smoothing, cell_merging, cell_splitting, thin_structure_loss,
curvature_flattening.

Two rules from master §12 that shape this module:
  1. THRESHOLDS ARE LOCKED BEFORE INSPECTION. They live in `configs/evaluation.yaml`
     (`failure_detectors:`) and are fixed before any model output is examined -- every default
     argument below is that config's value, read once at call time, never tuned after seeing
     which model it favours (that would manufacture the P6 result).
  2. DETECTORS ARE BLIND TO THE MODEL. Every function takes only (prediction, ground truth) [+
     instance masks for the two instance-level detectors] and never the model name, so a
     detector cannot treat C differently from B.

P6 is the ratio
    (reduction in curvature-related failures) / (reduction in non-curvature-related failures)
for B -> C (`p6_ratio` below). Which detectors count as curvature-related is declared in the
config BEFORE results exist (`CURVATURE_RELATED`/`NON_CURVATURE_RELATED`), not chosen afterwards
to suit the numbers.

`cell_merging` and `cell_splitting` require INSTANCE masks, not binary ones: two touching cells
share one binary blob, so a merge is invisible to a binary-only comparison -- this is why Phase 2
preserved instance IDs. The predicted side has no "instance ID" of its own (the model emits one
binary channel, master §7); a predicted instance is read off via connected-component labelling
of the binary prediction, which is the only thing "a predicted cell" can mean here.

[ENGINEERING, Phase 9 -- A45] Implemented at `src/failure/taxonomy.py`, replacing a Phase 3
scaffold (`src/failure/detectors.py`, never implemented, zero real callers) whose function
signatures and docstrings this module reuses almost verbatim -- see A45.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from src.datasets.curvature_stratification import contour_curvature, image_curvature_stats

# Declared BEFORE any result exists (master §12). Used to compute the P6 ratio.
CURVATURE_RELATED = ("corner_collapse", "boundary_smoothing", "curvature_flattening")
NON_CURVATURE_RELATED = ("cell_merging", "cell_splitting", "thin_structure_loss")

_STRUCT8 = np.ones((3, 3), dtype=bool)


def _as_binary_u8(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x) > 0).astype(np.uint8)


def _contours(binary: np.ndarray, min_points: int = 8) -> list[np.ndarray]:
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1, 2) for c in cnts if len(c) >= min_points]


# ------------------------------------------------------------------------------ curvature-related
def corner_collapse(pred: np.ndarray, target: np.ndarray,
                    curvature_percentile: float = 90.0,
                    min_drop: float = 0.30) -> dict[str, Any]:
    """High-curvature contour points rounded off in the prediction.

    Identifies GT boundary points at or above the `curvature_percentile`-th percentile of GT's
    OWN curvature distribution (i.e. GT's genuine corners), matches each to the NEAREST point on
    the predicted boundary (Euclidean, via a KD-tree -- no correspondence is assumed a priori),
    and flags a collapse where the matched predicted point's local curvature has fallen by more
    than `min_drop` (a fraction of the GT value) relative to the ground-truth corner.

    Returns: {"count": int, "sites": [(y, x), ...], "severity": float, "n_gt_corners": int}
    `severity` is the mean fractional curvature drop across collapsed sites (0 if none).
    """
    t_bin, p_bin = _as_binary_u8(target), _as_binary_u8(pred)
    t_contours = _contours(t_bin)
    p_contours = _contours(p_bin)
    if not t_contours or not p_contours:
        return {"count": 0, "sites": [], "severity": 0.0, "n_gt_corners": 0}

    gt_pts = np.concatenate(t_contours, axis=0).astype(np.float64)
    gt_curv = np.concatenate([contour_curvature(c) for c in t_contours])
    pred_pts = np.concatenate(p_contours, axis=0).astype(np.float64)
    pred_curv = np.concatenate([contour_curvature(c) for c in p_contours])

    corner_thresh = float(np.percentile(gt_curv, curvature_percentile))
    is_corner = gt_curv >= corner_thresh
    n_gt_corners = int(is_corner.sum())
    if n_gt_corners == 0:
        return {"count": 0, "sites": [], "severity": 0.0, "n_gt_corners": 0}

    tree = cKDTree(pred_pts)
    _, nearest_idx = tree.query(gt_pts[is_corner])
    matched_pred_curv = pred_curv[nearest_idx]
    matched_gt_curv = gt_curv[is_corner]

    with np.errstate(divide="ignore", invalid="ignore"):
        frac_drop = np.where(matched_gt_curv > 0,
                             1.0 - matched_pred_curv / matched_gt_curv, 0.0)
    collapsed = frac_drop >= min_drop
    sites = [(int(y), int(x)) for x, y in gt_pts[is_corner][collapsed]]
    severity = float(frac_drop[collapsed].mean()) if collapsed.any() else 0.0
    return {"count": int(collapsed.sum()), "sites": sites, "severity": severity,
           "n_gt_corners": n_gt_corners}


def boundary_smoothing(pred: np.ndarray, target: np.ndarray,
                       min_ratio: float = 0.90) -> dict[str, Any]:
    """Global loss of boundary detail: predicted perimeter shorter than ground truth's.

    A smoothed boundary is shorter than the true one at comparable area, so the total-perimeter
    ratio (summed across every connected component) detects it without needing point
    correspondence. Flagged when `pred_perimeter / target_perimeter < min_ratio`.
    """
    t_bin, p_bin = _as_binary_u8(target), _as_binary_u8(pred)
    t_perimeter = sum(cv2.arcLength(c.reshape(-1, 1, 2).astype(np.int32), closed=True)
                      for c in _contours(t_bin, min_points=3))
    p_perimeter = sum(cv2.arcLength(c.reshape(-1, 1, 2).astype(np.int32), closed=True)
                      for c in _contours(p_bin, min_points=3))
    ratio = float(p_perimeter / t_perimeter) if t_perimeter > 0 else float("nan")
    detected = bool(np.isfinite(ratio) and ratio < min_ratio)
    return {"detected": detected, "pred_perimeter": float(p_perimeter),
           "target_perimeter": float(t_perimeter), "ratio": ratio, "min_ratio": min_ratio}


def curvature_flattening(pred: np.ndarray, target: np.ndarray,
                         percentile: float = 90.0,
                         min_drop: float = 0.15) -> dict[str, Any]:
    """Drop in the image-level 90th-percentile boundary curvature.

    Reuses `curvature_stratification.image_curvature_stats` (the SAME statistic and sigma the
    §11 stratification itself uses) on both masks, and flags a drop of more than `min_drop`
    (a fraction of GT's value) in the aggregate. Differs from `corner_collapse` in scope:
    corner_collapse is per-site and local; this is the whole-image aggregate.
    """
    t_bin, p_bin = _as_binary_u8(target), _as_binary_u8(pred)
    t_stats = image_curvature_stats(t_bin)
    p_stats = image_curvature_stats(p_bin)
    t_score, p_score = t_stats["curvature_score"], p_stats["curvature_score"]
    if not np.isfinite(t_score) or t_score <= 0:
        return {"detected": False, "target_score": t_score, "pred_score": p_score,
               "frac_drop": float("nan"), "min_drop": min_drop}
    frac_drop = float(1.0 - p_score / t_score) if np.isfinite(p_score) else 1.0
    detected = bool(frac_drop >= min_drop)
    return {"detected": detected, "target_score": float(t_score),
           "pred_score": float(p_score) if np.isfinite(p_score) else None,
           "frac_drop": frac_drop, "min_drop": min_drop}


# -------------------------------------------------------------------------- non-curvature-related
# [ENGINEERING A45] Merge/split detection uses an ASYMMETRIC coverage measure, not symmetric IoU,
# despite configs/evaluation.yaml naming the field "iou_threshold". This is a necessity, not a
# convenience: for a predicted blob P formed by merging GT instances G1, G2 (P = G1 union G2),
# symmetric IoU(P, G1) = |G1| / |G1 union G2| is PROVABLY <= 0.5 whenever G1 and G2 are disjoint
# or share at most G1's own overlap -- so a >0.5 IoU threshold can never flag both G1 and G2
# against the same merged P, making merge/split undetectable at any threshold >= 0.5 by
# construction. Using RECALL (intersection / GT area) for merging and PRECISION
# (intersection / predicted-component area) for splitting is the standard fix in the instance-
# segmentation literature and is what "iou_threshold" is read as here: "what fraction of the
# smaller/explained region is covered".
def _recall_matches(pred_labeled: np.ndarray, gt_instances: np.ndarray,
                    threshold: float) -> dict[int, list[int]]:
    """{pred_component_id: [gt_instance_ids >= `threshold` COVERED BY this one component]}.

    recall = |pred_component ∩ gt_instance| / |gt_instance| -- "how much of this GT cell does
    this one predicted blob explain". Used by `cell_merging`: a component explaining >=2 GT
    instances at high recall has swallowed more than one cell.
    """
    matches: dict[int, list[int]] = {}
    pred_ids = [i for i in np.unique(pred_labeled) if i > 0]
    for pid in pred_ids:
        pred_mask = pred_labeled == pid
        gt_ids_here = np.unique(gt_instances[pred_mask])
        gt_ids_here = gt_ids_here[gt_ids_here > 0]
        hits = []
        for gid in gt_ids_here:
            gt_mask = gt_instances == gid
            gt_area = int(gt_mask.sum())
            if gt_area == 0:
                continue
            inter = int(np.logical_and(pred_mask, gt_mask).sum())
            if (inter / gt_area) >= threshold:
                hits.append(int(gid))
        matches[int(pid)] = hits
    return matches


def cell_merging(pred_instances: np.ndarray, gt_instances: np.ndarray,
                 iou_threshold: float = 0.5) -> dict[str, Any]:
    """Two or more GT instances covered by a single predicted component.

    `pred_instances` may be a raw BINARY prediction (connected-component-labelled here) or an
    already-labelled array -- either way it is re-labelled by connectivity, since "a predicted
    cell" only ever means one connected blob of the model's single output channel (master §7).

    Requires INSTANCE ground truth: touching cells share a binary blob, so this is undetectable
    from a binary-only comparison. See the module-level note above on why this uses RECALL
    (coverage of each GT instance), not symmetric IoU.
    """
    pred_labeled, n_pred = ndimage.label(_as_binary_u8(pred_instances) > 0, structure=_STRUCT8)
    matches = _recall_matches(pred_labeled, gt_instances, iou_threshold)
    merges = {pid: gids for pid, gids in matches.items() if len(gids) >= 2}
    return {"count": len(merges), "merged_components": merges, "n_pred_components": int(n_pred)}


def cell_splitting(pred_instances: np.ndarray, gt_instances: np.ndarray,
                   iou_threshold: float = 0.5) -> dict[str, Any]:
    """One GT instance broken into several predicted components. The dual of cell_merging.

    precision = |pred_component ∩ gt_instance| / |pred_component| -- "how much of this
    predicted fragment consists of this one GT cell". A GT instance explaining >=2 predicted
    fragments at high precision has been broken up. See the module-level note above on why this
    uses PRECISION (coverage of each predicted fragment), not symmetric IoU.
    """
    pred_labeled, n_pred = ndimage.label(_as_binary_u8(pred_instances) > 0, structure=_STRUCT8)
    gt_ids = [i for i in np.unique(gt_instances) if i > 0]
    splits: dict[int, list[int]] = {}
    for gid in gt_ids:
        gt_mask = gt_instances == gid
        pred_ids_here = np.unique(pred_labeled[gt_mask])
        pred_ids_here = pred_ids_here[pred_ids_here > 0]
        hits = []
        for pid in pred_ids_here:
            pred_mask = pred_labeled == pid
            pred_area = int(pred_mask.sum())
            if pred_area == 0:
                continue
            inter = int(np.logical_and(pred_mask, gt_mask).sum())
            if (inter / pred_area) >= iou_threshold:
                hits.append(int(pid))
        if len(hits) >= 2:
            splits[int(gid)] = hits
    return {"count": len(splits), "split_instances": splits, "n_pred_components": int(n_pred)}


def thin_structure_loss(pred: np.ndarray, target: np.ndarray,
                        max_width_px: float = 3.0,
                        min_loss_fraction: float = 0.5) -> dict[str, Any]:
    """Thin protrusions present in GT but missing from the prediction.

    A morphological OPENING with a disk of radius `max_width_px/2` removes any structure
    narrower than `max_width_px` while leaving wider structures intact (standard technique); the
    pixels erased by the opening ARE the thin structures. Flagged when more than
    `min_loss_fraction` of those thin-structure pixels are absent from the prediction.

    Relevant here specifically: the high-curvature bin is populated by migrating T cells with
    pointed uropods (Phase 2 spot check), which is exactly the structure most easily lost.

    [ASSUMPTION A45] `min_loss_fraction=0.5` is not in `configs/evaluation.yaml` (only
    `max_width_px` is locked there) -- "the thin structure is gone" needs a majority-loss
    threshold, and 0.5 (a genuine majority) is the least arbitrary choice available; logged
    since master §12 requires every detector threshold to be recorded.
    """
    t_bin, p_bin = _as_binary_u8(target).astype(bool), _as_binary_u8(pred).astype(bool)
    radius = max(1, int(round(max_width_px / 2)))
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = (x * x + y * y) <= radius * radius
    opened = ndimage.binary_opening(t_bin, structure=disk)
    thin = t_bin & ~opened
    n_thin = int(thin.sum())
    if n_thin == 0:
        return {"detected": False, "n_thin_px": 0, "n_missing_px": 0, "loss_fraction": 0.0,
               "max_width_px": max_width_px}
    missing = thin & ~p_bin
    n_missing = int(missing.sum())
    loss_fraction = n_missing / n_thin
    detected = bool(loss_fraction >= min_loss_fraction)
    return {"detected": detected, "n_thin_px": n_thin, "n_missing_px": n_missing,
           "loss_fraction": float(loss_fraction), "max_width_px": max_width_px}


# ---------------------------------------------------------------------------------- orchestration
def run_all_detectors(pred: np.ndarray, target: np.ndarray,
                      pred_instances: np.ndarray | None = None,
                      gt_instances: np.ndarray | None = None,
                      thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run all six detectors on one image.

    Args:
        pred_instances/gt_instances: required for cell_merging/cell_splitting. If omitted,
            `pred`/`target` (binary) are used for `pred_instances` and those two detectors are
            skipped for `gt_instances` (instance IDs cannot be recovered from a binary mask --
            silently connected-component-labelling the GT would be a bug, not a fallback).
        thresholds: the `failure_detectors:` block from `configs/evaluation.yaml`, locked before
            inspection (master §12). `None` falls back to each function's own signature default,
            which is acceptable only for unit tests -- a real evaluation run must always pass
            the locked config explicitly.

    Returns:
        One flat dict of per-detector results, ready for a `results/failure_counts/` row.
    """
    th = thresholds or {}
    cc = th.get("corner_collapse", {})
    bs = th.get("boundary_smoothing", {})
    cm = th.get("cell_merging", {})
    cs = th.get("cell_splitting", {})
    tsl = th.get("thin_structure_loss", {})
    cf = th.get("curvature_flattening", {})

    out: dict[str, Any] = {
        "corner_collapse": corner_collapse(
            pred, target,
            curvature_percentile=cc.get("curvature_percentile", 90.0),
            min_drop=cc.get("min_drop", 0.30)),
        "boundary_smoothing": boundary_smoothing(
            pred, target, min_ratio=bs.get("min_perimeter_ratio", 0.90)),
        "curvature_flattening": curvature_flattening(
            pred, target,
            percentile=cf.get("percentile", 90.0), min_drop=cf.get("min_drop", 0.15)),
        "thin_structure_loss": thin_structure_loss(
            pred, target, max_width_px=tsl.get("max_width_px", 3.0)),
    }
    if gt_instances is not None:
        pred_inst = pred_instances if pred_instances is not None else pred
        out["cell_merging"] = cell_merging(
            pred_inst, gt_instances, iou_threshold=cm.get("iou_threshold", 0.5))
        out["cell_splitting"] = cell_splitting(
            pred_inst, gt_instances, iou_threshold=cs.get("iou_threshold", 0.5))
    else:
        out["cell_merging"] = None
        out["cell_splitting"] = None
    return out


def p6_ratio(counts_b: dict[str, int], counts_c: dict[str, int]) -> dict[str, float]:
    """The P6 statistic: curvature-related vs non-curvature-related reduction, B -> C.

    master §12: ratio = (reduction in curvature-related failures) / (reduction in
    non-curvature-related failures), summed over each group's detectors. A ratio > 1 supports
    P6 (dCR preferentially fixes curvature-related failures over non-curvature-related ones).

    Args:
        counts_b, counts_c: {detector_name: total_count} aggregated over the test set (or a
            bin) for models B and C respectively -- e.g. summed `run_all_detectors(...)[name]
            ["count"]` (or `["detected"]` cast to int) across every image.

    Returns:
        {"curvature_reduction", "non_curvature_reduction", "ratio", "counts_b", "counts_c"}.
        `ratio` is NaN when the non-curvature reduction is zero -- an undefined ratio must never
        be reported as a large finite number (dividing by ~0 would manufacture an arbitrarily
        large "support" for P6 from noise).
    """
    missing_b = [d for d in (*CURVATURE_RELATED, *NON_CURVATURE_RELATED) if d not in counts_b]
    missing_c = [d for d in (*CURVATURE_RELATED, *NON_CURVATURE_RELATED) if d not in counts_c]
    if missing_b or missing_c:
        raise KeyError(f"counts missing detector(s): counts_b={missing_b}, counts_c={missing_c}")

    curvature_reduction = sum(counts_b[d] - counts_c[d] for d in CURVATURE_RELATED)
    non_curvature_reduction = sum(counts_b[d] - counts_c[d] for d in NON_CURVATURE_RELATED)
    if non_curvature_reduction == 0:
        ratio = float("nan")
    else:
        ratio = float(curvature_reduction / non_curvature_reduction)
    return {"curvature_reduction": float(curvature_reduction),
           "non_curvature_reduction": float(non_curvature_reduction), "ratio": ratio,
           "counts_b": dict(counts_b), "counts_c": dict(counts_c)}
