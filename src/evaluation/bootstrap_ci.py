"""Seeded bootstrap confidence intervals (master §6).

Percentile method: resample with replacement `n_boot` times, take the [alpha/2, 1-alpha/2]
percentiles of the resampled statistic. Seeded (A15: seed 42) so the reported interval is
reproducible run to run rather than a fresh random draw each time results are regenerated.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

_STATISTICS = {"mean": np.mean, "median": np.median}


def bootstrap_ci(values: Sequence[float], statistic: str = "mean",
                 n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> dict[str, Any]:
    """Seeded bootstrap confidence interval (master §6). Percentile method.

    Args:
        statistic: "mean" or "median".
        n_boot: [ASSUMPTION] master §6 does not fix a count; 10000 is configs/evaluation.yaml's
            value (`statistics.bootstrap.n_boot`).
        seed: [ASSUMPTION A15] fixed at 42 so the CI is reproducible.

    NaN values are dropped before resampling (e.g. HD95 on an empty-mask image) rather than
    propagating NaN into every bootstrap draw that happens to sample one.
    """
    stat_fn = _STATISTICS.get(statistic)
    if stat_fn is None:
        raise ValueError(f"unknown statistic {statistic!r}, expected one of {list(_STATISTICS)}")

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
               "n": 0, "n_boot": n_boot, "alpha": alpha, "seed": seed, "statistic": statistic}

    point = float(stat_fn(arr))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = stat_fn(arr[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi), "n": n,
           "n_boot": n_boot, "alpha": alpha, "seed": seed, "statistic": statistic}
