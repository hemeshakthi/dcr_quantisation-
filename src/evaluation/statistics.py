"""Statistical analysis (master §6; scope fixed by A29 and docs/success_criteria.md).

Master §6 keeps the paper's Student t-test as a reported SECONDARY and makes the paired
Wilcoxon signed-rank test PRIMARY, because the per-image samples here are small and not
normal. On top of that it adds Holm-Bonferroni correction, TOST on Dice, rank-biserial effect
sizes, bootstrap CIs, partial correlation, and power at the actual n.

SCOPE, fixed before any p-value exists (A29):
  * confirmatory family = 16 tests
    (2 comparisons {B vs C, B' vs C} x 2 bins {low, high} x 2 ratios x 2 boundary metrics)
  * the MID bin (test n=12) is DESCRIPTIVE-ONLY: measured power is 0.337 at d=0.5, so it
    cannot support or refute P1-P6, and it is excluded from the corrected family so it does
    not dilute correction on the tests that carry claims.
  * Dice is NOT in the superiority family -- it is tested for EQUIVALENCE via TOST (+/-0.02).

Direction conventions differ by metric and getting them wrong silently inverts a conclusion:
Boundary F1 is better HIGHER, HD95 is better LOWER.
"""
from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

# A29: the confirmatory family. Mid bin excluded (descriptive-only).
CONFIRMATORY_BINS = ("low", "high")
DESCRIPTIVE_ONLY_BINS = ("mid",)
CONFIRMATORY_FAMILY_SIZE = 16


def paired_wilcoxon(a: Sequence[float], b: Sequence[float],
                    alternative: str = "two-sided") -> dict[str, Any]:
    """Wilcoxon signed-rank test on paired per-image values (PRIMARY, master §6).

    Returns statistic, p-value, n, and the number of zero differences (which the test
    discards, so a large count means the effective n is well below the nominal n and must be
    reported).
    """
    raise NotImplementedError


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Paired Student t-test -- reported SECONDARY, since it is the paper's own test (ref 40)."""
    raise NotImplementedError


def rank_biserial_effect_size(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Matched-pairs rank-biserial correlation, the effect size paired with Wilcoxon.

    Returns the coefficient plus its Cohen-scale label (small/medium/large). An effect size is
    reported for every test: with n=36 in the high bin a non-significant result may simply be
    underpowered, and only the effect size distinguishes "no effect" from "not enough data".
    """
    raise NotImplementedError


def tost_equivalence(a: Sequence[float], b: Sequence[float],
                     margin: float = 0.02) -> dict[str, Any]:
    """Two one-sided tests for EQUIVALENCE (master §4: Dice within +/-0.02).

    Needed because "we found no significant difference in Dice" does not establish that Dice
    is preserved -- absence of evidence is not evidence of absence, and P1's claim that pruning
    spares Dice requires an equivalence test, not a failed superiority test.
    """
    raise NotImplementedError


def holm_bonferroni(p_values: dict[str, float],
                    alpha: float = 0.05,
                    family_size: int | None = None) -> dict[str, Any]:
    """Holm-Bonferroni step-down correction over the confirmatory family.

    Args:
        p_values: {test_name: p}. Must contain ONLY confirmatory tests; passing
            descriptive-only mid-bin tests would dilute the correction (A29).
        family_size: defaults to len(p_values); pass CONFIRMATORY_FAMILY_SIZE explicitly to
            correct over the full planned family even when only some tests have been run.

    Returns per-test adjusted thresholds, adjusted p-values and reject/retain decisions.
    """
    raise NotImplementedError


def bootstrap_ci(values: Sequence[float], statistic: str = "mean",
                 n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 42) -> dict[str, Any]:
    """Seeded bootstrap confidence interval (master §6). Percentile method."""
    raise NotImplementedError


def partial_correlation(df: pd.DataFrame, x: str, y: str,
                        covariates: Sequence[str]) -> dict[str, Any]:
    """Partial correlation of x and y controlling for covariates (master §6).

    The intended call is curvature vs boundary error, controlling for cell_size and contrast.

    Phase 2 measured the cell-size confound as near-zero after the sigma=2 revision
    (Spearman +0.09 / -0.05, A27), so this is now a confirmatory check rather than a
    load-bearing correction -- but it is still run and still reported, because that near-zero
    was measured on ground truth, not on model errors.
    """
    raise NotImplementedError


def run_confirmatory_suite(per_image_csv: str | Path,
                           output_dir: str | Path,
                           alpha: float = 0.05) -> dict[str, Any]:
    """Run the full planned analysis and write the results tables.

    Runs the 16 confirmatory tests with Holm correction, the descriptive-only mid-bin tests
    (labelled as such, with their power stated inline), TOST on Dice, effect sizes, bootstrap
    CIs and the partial correlation.

    Reports whatever it finds. Master Golden Rule 4: never fabricate a positive; a null result
    is reported honestly.
    """
    raise NotImplementedError
