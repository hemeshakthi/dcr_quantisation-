"""The three FROZEN metrics: Dice, Boundary F1, HD95 (master §6) -- composition layer.

[ENGINEERING, Phase 8 -- A44] Phase 8 split the per-metric algorithms into dedicated modules
(`dice_metrics.py` / `boundary_f1.py` / `hausdorff.py`, each independently importable). This
module re-exports them under their original names for the callers that already depend on this
import path (`backend/services/metric_service.py`, `src/training/validator.py`, several
`audit/*` scripts -- all real, tested, verified end-to-end) and adds `compute_all_metrics()`,
which composes all three into one per-image row. There is exactly ONE implementation of each
metric; this file contains no metric logic of its own, only composition.

Their roles differ and must not be conflated:
  * Dice        [PAPER]      -- the FOIL. P1 predicts pruning leaves it largely intact while
                                damaging the boundary metrics. It is tested for EQUIVALENCE via
                                TOST (+/-0.02), not superiority: a failed superiority test
                                cannot establish that Dice is preserved.
  * Boundary F1 [LITERATURE] -- 2 px AND 5 px tolerance, reported separately.
  * HD95        [LITERATURE] -- 95th-percentile symmetric Hausdorff, standardised
                                morphological-gradient boundary extraction, integer coordinates.

Every function returns PER-IMAGE values. Master §6 requires per-image storage for the paired
Wilcoxon tests; aggregating early would destroy the pairing the statistical design rests on.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from src.metrics.boundary_f1 import boundary_f1, extract_boundary
from src.metrics.dice_metrics import dice_score
from src.metrics.hausdorff import hd95

__all__ = ["dice_score", "extract_boundary", "boundary_f1", "hd95", "compute_all_metrics"]

# 8-connectivity, matching the connected-component convention used elsewhere in the study.
_STRUCT = np.ones((3, 3), dtype=bool)


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
    out["n_instances_pred"] = int(ndimage.label(np.asarray(pred) > 0, structure=_STRUCT)[1])
    out["n_instances_gt"] = int(ndimage.label(np.asarray(target) > 0, structure=_STRUCT)[1])
    return out
