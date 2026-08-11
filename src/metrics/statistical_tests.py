"""Statistical hypothesis tests for the confirmatory analysis (master §6).

Master §6 makes the paired Wilcoxon signed-rank test PRIMARY (small, non-normal per-image
samples) and keeps the paper's own paired Student t-test as a reported SECONDARY. On top of
that: Holm-Bonferroni correction over the confirmatory family, TOST equivalence for Dice (the
metric predicted to be PRESERVED, not improved -- P1), rank-biserial effect sizes (reported for
every test, since a non-significant result at n=36 may simply be underpowered), and partial
correlation for the curvature-vs-boundary-error confound check (master §11).

Direction conventions differ by metric and getting them wrong silently inverts a conclusion:
Boundary F1 is better HIGHER, HD95 is better LOWER. This module reports raw statistics/p-values
and does not itself apply a direction convention -- that is `src/evaluation/curvature_stratified.py`'s
job, which knows which metric is which.

[ENGINEERING, Phase 8 -- A44] Consolidates what a Phase 3 scaffold called
`src/evaluation/statistics.py` (never implemented; nothing depended on it) under the module
location and function names this phase's instruction specifies. See
`docs/implementation_assumptions.md` A44.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def _paired_arrays(a: Sequence[float], b: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"paired arrays must have equal shape, got {a_arr.shape} vs {b_arr.shape}")
    return a_arr, b_arr


def paired_wilcoxon(a: Sequence[float], b: Sequence[float],
                    alternative: str = "two-sided") -> dict[str, Any]:
    """Wilcoxon signed-rank test on paired per-image values (PRIMARY, master §6).

    Returns statistic, p-value, n, and the number of zero differences (which the test
    discards, so a large count means the effective n is well below the nominal n and must be
    reported).
    """
    a_arr, b_arr = _paired_arrays(a, b)
    diff = a_arr - b_arr
    n = int(len(diff))
    n_zero = int(np.sum(diff == 0))
    n_eff = n - n_zero
    if n_eff == 0:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": n,
               "n_zero_diff": n_zero, "n_effective": 0, "alternative": alternative}
    statistic, p_value = stats.wilcoxon(a_arr, b_arr, alternative=alternative,
                                        zero_method="wilcox")
    return {"statistic": float(statistic), "p_value": float(p_value), "n": n,
           "n_zero_diff": n_zero, "n_effective": n_eff, "alternative": alternative}


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Paired Student t-test -- reported SECONDARY, since it is the paper's own test (ref 40)."""
    a_arr, b_arr = _paired_arrays(a, b)
    diff = a_arr - b_arr
    n = int(len(diff))
    if n < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": n,
               "mean_diff": float(diff.mean()) if n else float("nan"), "sd_diff": float("nan")}
    statistic, p_value = stats.ttest_rel(a_arr, b_arr)
    return {"statistic": float(statistic), "p_value": float(p_value), "n": n,
           "mean_diff": float(diff.mean()), "sd_diff": float(diff.std(ddof=1))}


def rank_biserial_effect_size(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Matched-pairs rank-biserial correlation, the effect size paired with Wilcoxon.

    r = (W+ - W-) / (W+ + W-), where W+/W- are the sums of ranks of the positive/negative
    signed differences. Returns the coefficient plus its Cohen-scale label (small/medium/large)
    -- reported for EVERY test, since with n=36 in the high bin a non-significant result may
    simply be underpowered, and only the effect size distinguishes "no effect" from "not enough
    data".
    """
    a_arr, b_arr = _paired_arrays(a, b)
    diff = a_arr - b_arr
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return {"r": float("nan"), "label": "undefined (all differences zero)", "n_effective": 0}
    ranks = stats.rankdata(np.abs(nonzero))
    w_pos = float(ranks[nonzero > 0].sum())
    w_neg = float(ranks[nonzero < 0].sum())
    r = (w_pos - w_neg) / (w_pos + w_neg)
    abs_r = abs(r)
    label = "small" if abs_r < 0.3 else "medium" if abs_r < 0.5 else "large"
    return {"r": float(r), "label": label, "n_effective": int(len(nonzero)),
           "w_pos": w_pos, "w_neg": w_neg}


def tost_equivalence_test(a: Sequence[float], b: Sequence[float], margin: float = 0.02,
                          metric_name: str | None = None) -> dict[str, Any]:
    """Two one-sided tests (TOST) for EQUIVALENCE (master §4: Dice within +/-0.02).

    Needed because "we found no significant difference in Dice" does not establish that Dice is
    preserved -- absence of evidence is not evidence of absence, and P1's claim that pruning
    spares Dice requires an equivalence test, not a failed superiority test.

    [ENGINEERING] `metric_name`, if given, is checked against "dice" -- master §4 restricts
    TOST to Dice specifically; every other confirmatory metric uses superiority testing
    (`paired_wilcoxon`), never equivalence. This is an orchestration-level guard (raises if
    misused), not a mathematical restriction -- TOST is well-defined for any bounded paired
    metric, but this study only ever intends to run it on one. See A44.
    """
    if metric_name is not None and metric_name != "dice":
        raise ValueError(
            f"TOST is restricted to Dice (master §4); got metric_name={metric_name!r}. "
            f"Every other confirmatory metric uses superiority testing (paired_wilcoxon).")
    a_arr, b_arr = _paired_arrays(a, b)
    diff = a_arr - b_arr
    n = int(len(diff))
    df = n - 1
    if df < 1:
        return {"equivalent": False, "p_lower": float("nan"), "p_upper": float("nan"),
               "p_tost": float("nan"), "mean_diff": float("nan"), "margin": margin, "n": n}
    mean_diff = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / np.sqrt(n)
    if se == 0:
        # zero variance: equivalent iff the (constant) difference is inside the margin
        equivalent = abs(mean_diff) < margin
        return {"equivalent": equivalent, "p_lower": 0.0 if equivalent else 1.0,
               "p_upper": 0.0 if equivalent else 1.0, "p_tost": 0.0 if equivalent else 1.0,
               "mean_diff": mean_diff, "margin": margin, "n": n}
    t_lower = (mean_diff - (-margin)) / se   # H0: true diff <= -margin
    t_upper = (mean_diff - margin) / se      # H0: true diff >= +margin
    p_lower = float(1 - stats.t.cdf(t_lower, df))
    p_upper = float(stats.t.cdf(t_upper, df))
    p_tost = max(p_lower, p_upper)
    return {"equivalent": p_tost < 0.05, "p_lower": p_lower, "p_upper": p_upper,
           "p_tost": p_tost, "mean_diff": mean_diff, "margin": margin, "n": n}


def holm_bonferroni_correction(p_values: dict[str, float], alpha: float = 0.05,
                               family_size: int | None = None) -> dict[str, Any]:
    """Holm-Bonferroni step-down correction over the confirmatory family.

    Args:
        p_values: {test_name: p}. Must contain ONLY confirmatory tests; passing
            descriptive-only mid-bin tests would dilute the correction (A29).
        family_size: defaults to len(p_values); pass the full planned family size (16, A29) to
            correct over the whole planned family even when only some tests have been run yet.

    Returns per-test rank, adjusted threshold and reject/retain decision. Step-down: sorted by
    p ascending, test i is rejected only if p_i <= alpha/(m-i+1) AND every smaller-p test in
    the family was also rejected (a single retained test stops the cascade for everything
    ranked after it).
    """
    m = family_size if family_size is not None else len(p_values)
    if m <= 0:
        raise ValueError("family_size must be positive")
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    results: dict[str, dict[str, Any]] = {}
    reject_so_far = True
    for rank, (name, p) in enumerate(ordered, start=1):
        threshold = alpha / (m - rank + 1)
        reject = reject_so_far and (p <= threshold)
        reject_so_far = reject
        results[name] = {"p_value": float(p), "rank": rank, "threshold": float(threshold),
                         "reject_null": bool(reject)}
    return {"family_size": m, "alpha": alpha, "tests": results,
           "n_rejected": sum(1 for r in results.values() if r["reject_null"])}


def partial_correlation(df: pd.DataFrame, x: str, y: str,
                        covariates: Sequence[str]) -> dict[str, Any]:
    """Partial correlation of x and y controlling for covariates (master §6).

    Residualises x and y on the covariates via OLS, then Pearson-correlates the residuals (the
    standard partial-correlation construction). The intended call is curvature vs boundary
    error, controlling for cell_size and contrast (master §11).

    Phase 2 measured the cell-size confound as near-zero after the sigma=2 revision (Spearman
    +0.09 / -0.05, A27), so this is now a confirmatory check rather than a load-bearing
    correction -- but it is still run and still reported, because that near-zero was measured
    on ground truth, not on model errors.
    """
    covariates = list(covariates)
    sub = df[[x, y, *covariates]].dropna()
    n = len(sub)
    k = len(covariates)
    if n <= k + 2:
        return {"r": float("nan"), "p_value": float("nan"), "n": n, "df": n - k - 2,
               "x": x, "y": y, "covariates": covariates}
    design = sm.add_constant(sub[covariates].to_numpy())
    resid_x = sm.OLS(sub[x].to_numpy(), design).fit().resid
    resid_y = sm.OLS(sub[y].to_numpy(), design).fit().resid
    r, p = stats.pearsonr(resid_x, resid_y)
    return {"r": float(r), "p_value": float(p), "n": n, "df": n - k - 2,
           "x": x, "y": y, "covariates": covariates}
