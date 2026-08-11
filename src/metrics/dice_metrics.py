"""Dice coefficient (master §6) -- the FOIL metric.

P1 predicts pruning leaves Dice largely intact while damaging boundary geometry. Dice is
therefore tested for EQUIVALENCE via TOST (+/-0.02, `src/metrics/statistical_tests.py`), never
for superiority -- a failed superiority test cannot establish that a metric was PRESERVED.
"""
from __future__ import annotations

import numpy as np


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
