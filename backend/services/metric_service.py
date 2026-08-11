"""Wraps the frozen segmentation metrics (src/metrics/segmentation.py, master §6) for the API.

Only the four values docs/api_contract.md promises per model are surfaced here (dice,
boundary_f1_2px, boundary_f1_5px, hd95) -- computed via the EXACT SAME functions the study's own
per-image tables use, so a number shown in the live demo can never silently disagree with a
number in a results table (Golden Rule 3).
"""
from __future__ import annotations

import math

import numpy as np

from src.metrics.segmentation import boundary_f1, dice_score, hd95


def _json_safe(value: float) -> float | None:
    """NaN is a legitimate metric value here (hd95 on an empty mask) but is not valid JSON;
    docs/api_contract.md promises `null` for that case, never a fabricated 0."""
    return None if (value != value or math.isinf(value)) else float(value)


def compute_case_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    return {
        "dice": _json_safe(dice_score(pred, target)),
        "boundary_f1_2px": _json_safe(boundary_f1(pred, target, tolerance_px=2.0)["boundary_f1"]),
        "boundary_f1_5px": _json_safe(boundary_f1(pred, target, tolerance_px=5.0)["boundary_f1"]),
        "hd95": _json_safe(hd95(pred, target)),
    }
