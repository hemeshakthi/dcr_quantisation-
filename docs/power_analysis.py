"""Statistical power analysis at the ACTUAL per-bin test n (master §6).

Master §6 specifies: "power analysis at actual n (matches paper's Cohen power analysis
ref 42)". This script supplies it, using the real test-bin occupancy measured in Phase 2
(low=50, mid=12, high=36 -- see data/curvature/curvature_bins.csv).

DESIGN BEING POWERED
--------------------
The study's confirmatory comparisons are PAIRED: every model (B, B', C) is evaluated on the
SAME test images, so each image contributes one paired difference. Master §6 makes the paired
Wilcoxon signed-rank test PRIMARY (small, non-normal medical samples) and keeps the paired
t-test as a reported secondary.

  * Paired t-test power is computed exactly, from the noncentral t distribution.
  * Wilcoxon signed-rank power is obtained from the t-test power at an EFFECTIVE sample size
    n_eff = ARE * n, where ARE is the asymptotic relative efficiency of the signed-rank test
    against the t-test. ARE = 3/pi ~= 0.955 for normally distributed differences; it is
    bounded below by 0.864 for ANY continuous distribution and can EXCEED 1 for heavy-tailed
    ones (Hodges & Lehmann 1956). Using 0.955 is therefore conservative-but-realistic for the
    normal case and pessimistic for the heavy-tailed case we actually expect.

EFFECT SIZE
-----------
Effect sizes are Cohen's d for PAIRED differences, i.e. d = mean(diff) / sd(diff) -- NOT the
between-group d. This matters: d here is inflated relative to a two-sample d whenever the two
models' per-image scores are positively correlated across images, which they will be (the same
image is easy or hard for every model). A paired design is therefore substantially more
powerful than the raw n suggests, and the numbers below are a floor, not a ceiling.

We cannot convert master §4's absolute threshold (Boundary F1 C-B >= 0.04) into a d until the
sd of the paired differences is known, which requires trained models. So power is reported
ACROSS a range of d (Cohen's small/medium/large), plus the minimum detectable effect at 80%
and 90% power. Once Phase 3+ produces per-image metrics, rerun with --sd-diff to convert the
0.04 threshold into a concrete d.

ALPHA
-----
Two alphas are reported:
  * nominal            alpha = 0.05 (two-sided)
  * Holm worst case    alpha = 0.05 / m, where m is the confirmatory family size.
Holm-Bonferroni tests the smallest p-value against alpha/m and progressively relaxes, so
alpha/m is the MOST stringent threshold any test in the family faces -- a conservative
planning bound, not the typical one.

Usage:
    python docs/power_analysis.py --n_test 50      # low-curvature bin
    python docs/power_analysis.py --n_test 12      # mid-curvature bin
    python docs/power_analysis.py --n_test 36      # high-curvature bin
    python docs/power_analysis.py --all-bins       # all three, side by side
"""
from __future__ import annotations

import argparse

import numpy as np
from statsmodels.stats.power import TTestPower

# Asymptotic relative efficiency of the Wilcoxon signed-rank test vs the paired t-test,
# for normally distributed paired differences. Hodges & Lehmann (1956).
ARE_WILCOXON_NORMAL = 3.0 / np.pi  # ~= 0.9549

COHEN_LEVELS = [
    (0.20, "small"),
    (0.30, ""),
    (0.50, "medium"),
    (0.80, "large"),
    (1.00, ""),
]

# Confirmatory family size after excluding the descriptive-only mid bin (see
# docs/success_criteria.md "Statistical scope"). Default used for the Holm worst-case alpha.
DEFAULT_FAMILY_SIZE = 16


def t_power(d: float, n: int, alpha: float) -> float:
    """Exact paired t-test power (two-sided) from the noncentral t distribution."""
    if n < 2:
        return float("nan")
    return float(TTestPower().power(effect_size=d, nobs=n, alpha=alpha,
                                    alternative="two-sided"))


def wilcoxon_power(d: float, n: int, alpha: float,
                   are: float = ARE_WILCOXON_NORMAL) -> float:
    """Signed-rank power via the ARE-adjusted effective sample size."""
    n_eff = max(2.0, are * n)
    if n_eff < 2:
        return float("nan")
    return float(TTestPower().power(effect_size=d, nobs=n_eff, alpha=alpha,
                                    alternative="two-sided"))


def mde(n: int, alpha: float, power: float, are: float | None = None) -> float:
    """Minimum detectable paired effect size d at the requested power."""
    nobs = n if are is None else max(2.0, are * n)
    if nobs < 2:
        return float("nan")
    try:
        return float(TTestPower().solve_power(effect_size=None, nobs=nobs, alpha=alpha,
                                              power=power, alternative="two-sided"))
    except Exception:
        return float("nan")


def n_required(d: float, alpha: float, power: float,
               are: float = ARE_WILCOXON_NORMAL) -> float:
    """Paired n needed for the requested power at effect size d (Wilcoxon, ARE-adjusted)."""
    try:
        n_eff = float(TTestPower().solve_power(effect_size=d, nobs=None, alpha=alpha,
                                               power=power, alternative="two-sided"))
        return n_eff / are
    except Exception:
        return float("nan")


def report(n: int, label: str, family_size: int, sd_diff: float | None,
           target_delta: float) -> dict:
    alpha_nom = 0.05
    alpha_holm = 0.05 / family_size

    print("=" * 78)
    print(f"POWER ANALYSIS  --  n_test = {n}" + (f"   ({label}-curvature bin)" if label else ""))
    print("=" * 78)
    print(f"  design            : PAIRED (same test images across models)")
    print(f"  primary test      : Wilcoxon signed-rank (master §6), ARE = {ARE_WILCOXON_NORMAL:.4f}")
    print(f"  secondary test    : paired t-test (master §6, the paper's own test)")
    print(f"  alpha nominal     : {alpha_nom:.4f}  (two-sided)")
    print(f"  alpha Holm worst  : {alpha_holm:.6f}  (= 0.05 / {family_size}, most stringent "
          f"threshold in the family)")

    print(f"\n  {'Cohen d':>9}  {'':8}  {'Wilcoxon':>9} {'t-test':>9}   |  "
          f"{'Wilcoxon':>9} {'t-test':>9}")
    print(f"  {'':>9}  {'':8}  {'(a=0.05)':>9} {'(a=0.05)':>9}   |  "
          f"{'(Holm)':>9} {'(Holm)':>9}")
    print("  " + "-" * 72)
    rows = {}
    for d, tag in COHEN_LEVELS:
        wn = wilcoxon_power(d, n, alpha_nom)
        tn = t_power(d, n, alpha_nom)
        wh = wilcoxon_power(d, n, alpha_holm)
        th = t_power(d, n, alpha_holm)
        rows[d] = {"wilcoxon_nominal": wn, "t_nominal": tn,
                   "wilcoxon_holm": wh, "t_holm": th}
        print(f"  {d:>9.2f}  {tag:8}  {wn:>9.3f} {tn:>9.3f}   |  {wh:>9.3f} {th:>9.3f}")

    print(f"\n  MINIMUM DETECTABLE EFFECT (paired Cohen d)")
    mde80_nom = mde(n, alpha_nom, 0.80, ARE_WILCOXON_NORMAL)
    mde90_nom = mde(n, alpha_nom, 0.90, ARE_WILCOXON_NORMAL)
    mde80_holm = mde(n, alpha_holm, 0.80, ARE_WILCOXON_NORMAL)
    print(f"    at 80% power, alpha=0.05      : d >= {mde80_nom:.3f}")
    print(f"    at 90% power, alpha=0.05      : d >= {mde90_nom:.3f}")
    print(f"    at 80% power, alpha=Holm worst: d >= {mde80_holm:.3f}")

    print(f"\n  n REQUIRED (Wilcoxon, alpha=0.05, 80% power)")
    for d, tag in [(0.5, "medium"), (0.8, "large")]:
        print(f"    to detect d={d:.1f} ({tag:6}) : n >= {int(np.ceil(n_required(d, alpha_nom, 0.80)))}"
              f"   -> current n={n} is {'ADEQUATE' if n >= n_required(d, alpha_nom, 0.80) else 'UNDERPOWERED'}")

    out = {"n": n, "label": label, "rows": rows,
           "mde80_nominal": mde80_nom, "mde90_nominal": mde90_nom,
           "mde80_holm": mde80_holm}

    if sd_diff is not None and sd_diff > 0:
        d_target = target_delta / sd_diff
        print(f"\n  ANCHORED TO MASTER §4 (Boundary F1 C-B >= {target_delta})")
        print(f"    supplied sd of paired differences : {sd_diff}")
        print(f"    implied paired Cohen d            : {d_target:.3f}")
        print(f"    Wilcoxon power at alpha=0.05      : {wilcoxon_power(d_target, n, alpha_nom):.3f}")
        print(f"    Wilcoxon power at Holm worst-case : {wilcoxon_power(d_target, n, alpha_holm):.3f}")
        out["d_target"] = d_target
    else:
        print(f"\n  NOTE: master §4's threshold (Boundary F1 C-B >= {target_delta}) cannot be")
        print(f"        converted to a Cohen d until the sd of paired differences is known,")
        print(f"        which requires trained models. Rerun with --sd-diff after Phase 3+.")

    verdict = ("ADEQUATE for medium effects" if n >= n_required(0.5, alpha_nom, 0.80)
               else "ADEQUATE only for large effects" if n >= n_required(0.8, alpha_nom, 0.80)
               else "UNDERPOWERED even for large effects")
    print(f"\n  VERDICT: n={n} is {verdict}.")
    out["verdict"] = verdict
    print()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_test", type=int, default=None, help="test-set n for one bin")
    ap.add_argument("--label", type=str, default="", help="bin label for the header")
    ap.add_argument("--all-bins", action="store_true",
                    help="run the three measured Phase 2 bins (low=50, mid=12, high=36)")
    ap.add_argument("--family-size", type=int, default=DEFAULT_FAMILY_SIZE,
                    help=f"confirmatory family size for the Holm worst-case alpha "
                         f"(default {DEFAULT_FAMILY_SIZE})")
    ap.add_argument("--sd-diff", type=float, default=None,
                    help="sd of paired per-image differences, once known from real runs")
    ap.add_argument("--target-delta", type=float, default=0.04,
                    help="master §4 Boundary F1 threshold (default 0.04)")
    args = ap.parse_args()

    if args.all_bins:
        for n, lab in [(50, "low"), (12, "mid"), (36, "high")]:
            report(n, lab, args.family_size, args.sd_diff, args.target_delta)
        return 0
    if args.n_test is None:
        ap.error("provide --n_test or --all-bins")
    report(args.n_test, args.label, args.family_size, args.sd_diff, args.target_delta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
