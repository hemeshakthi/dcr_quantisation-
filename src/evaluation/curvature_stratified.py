"""Curvature-stratified confirmatory analysis (master §6, A29).

Per-bin Wilcoxon restricted to HIGH vs LOW bins as the PRIMARY CONFIRMATORY tests (master §6);
the MEDIUM bin is DESCRIPTIVE-ONLY and excluded from the Holm-Bonferroni family (A29: measured
power 0.337 at d=0.5, n=12 -- underpowered by construction, cannot support or refute P1-P6).

Confirmatory family = 16 tests: 2 comparisons {B vs C, B' vs C} x 2 bins {low, high} x 2 ratios
{0.5, 0.7} x 2 boundary metrics {boundary_f1_2px, boundary_f1_5px}. Dice is NOT in this family
-- it is tested for equivalence via `statistical_tests.tost_equivalence_test`, never here.

Direction conventions differ by metric and getting them wrong silently inverts a conclusion:
Boundary F1 is better HIGHER, HD95 is better LOWER (recorded in `HIGHER_IS_BETTER` below and
used to compute which model a paired difference "favors").
"""
from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from src.metrics.statistical_tests import (holm_bonferroni_correction, paired_t_test,
                                           paired_wilcoxon, rank_biserial_effect_size,
                                           tost_equivalence_test)

# A29: the confirmatory family. Mid bin excluded (descriptive-only).
CONFIRMATORY_BINS = ("low", "high")
DESCRIPTIVE_ONLY_BINS = ("mid",)
CONFIRMATORY_FAMILY_SIZE = 16

HIGHER_IS_BETTER = {"dice": True, "boundary_f1_2px": True, "boundary_f1_5px": True,
                    "hd95": False}


def _paired_values(df: pd.DataFrame, model_a: str, model_b: str, ratio: float | None,
                   bin_name: str, metric: str) -> tuple[list[float], list[float]]:
    sub = df[df["curvature_bin"] == bin_name]
    if ratio is not None and "pruning_ratio" in sub.columns:
        sub = sub[sub["pruning_ratio"] == ratio]
    a = sub[sub["model"] == model_a].groupby("image_id")[metric].mean()
    b = sub[sub["model"] == model_b].groupby("image_id")[metric].mean()
    common = a.index.intersection(b.index)
    return a.loc[common].tolist(), b.loc[common].tolist()


def run_bin_comparison(df: pd.DataFrame, model_a: str, model_b: str, ratio: float | None,
                       bin_name: str, metric: str) -> dict[str, Any]:
    """One (comparison, ratio, bin, metric) cell: Wilcoxon (primary) + t-test (secondary) +
    rank-biserial effect size, all on the same paired sample."""
    a_vals, b_vals = _paired_values(df, model_a, model_b, ratio, bin_name, metric)
    wilcoxon = paired_wilcoxon(a_vals, b_vals)
    ttest = paired_t_test(a_vals, b_vals)
    effect = rank_biserial_effect_size(a_vals, b_vals)
    higher_better = HIGHER_IS_BETTER.get(metric, True)
    n = len(a_vals)
    mean_diff = (sum(a_vals) - sum(b_vals)) / n if n else float("nan")
    favors_a = bool((mean_diff > 0) == higher_better) if n else None
    return {"model_a": model_a, "model_b": model_b, "ratio": ratio, "bin": bin_name,
           "metric": metric, "n": n, "wilcoxon": wilcoxon, "t_test": ttest,
           "effect_size": effect, "mean_diff": mean_diff, "favors_a": favors_a,
           "confirmatory": bin_name in CONFIRMATORY_BINS}


def run_curvature_stratified_suite(df: pd.DataFrame,
                                   comparisons: Sequence[tuple[str, str]],
                                   ratios: Sequence[float],
                                   metrics: Sequence[str] = ("boundary_f1_2px", "boundary_f1_5px"),
                                   alpha: float = 0.05) -> dict[str, Any]:
    """Run every (comparison, ratio, bin, metric) cell across low/high/mid; Holm-correct only
    the low/high (confirmatory) cells over the FULL planned family size (A29), even if fewer
    than 16 tests actually ran (e.g. only one ratio available so far).

    Args:
        comparisons: (model_a, model_b) pairs, e.g. [("C", "B"), ("C", "B_prime")].
        ratios: pruning ratios to include, e.g. [0.5, 0.7].
        metrics: boundary metrics only -- Dice is never in this family (use
            `run_dice_tost` instead).
    """
    confirmatory: dict[str, dict[str, Any]] = {}
    descriptive: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {}

    for model_a, model_b in comparisons:
        for ratio in ratios:
            for metric in metrics:
                for bin_name in (*CONFIRMATORY_BINS, *DESCRIPTIVE_ONLY_BINS):
                    key = f"{model_a}_vs_{model_b}|ratio={ratio}|bin={bin_name}|{metric}"
                    result = run_bin_comparison(df, model_a, model_b, ratio, bin_name, metric)
                    if bin_name in CONFIRMATORY_BINS:
                        confirmatory[key] = result
                        p_values[key] = result["wilcoxon"]["p_value"]
                    else:
                        descriptive[key] = result

    holm = holm_bonferroni_correction(p_values, alpha=alpha,
                                      family_size=CONFIRMATORY_FAMILY_SIZE)

    return {"confirmatory": confirmatory, "descriptive_mid_bin": descriptive,
           "holm_bonferroni": holm, "confirmatory_family_size": CONFIRMATORY_FAMILY_SIZE,
           "n_confirmatory_tests_run": len(confirmatory)}


def run_dice_tost(df: pd.DataFrame, model_a: str, model_b: str, ratio: float | None,
                  bin_name: str, margin: float = 0.02) -> dict[str, Any]:
    """TOST equivalence on Dice ONLY (master §4) -- the sole call site in this codebase that
    invokes `tost_equivalence_test`, and it always passes `metric_name="dice"` so a misuse
    elsewhere is rejected rather than silently run."""
    a_vals, b_vals = _paired_values(df, model_a, model_b, ratio, bin_name, "dice")
    result = tost_equivalence_test(a_vals, b_vals, margin=margin, metric_name="dice")
    result.update(model_a=model_a, model_b=model_b, ratio=ratio, bin=bin_name)
    return result
