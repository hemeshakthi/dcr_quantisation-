# Success Criteria

**LOCKED — do not modify after implementation begins.**

Source: PROJECT_MASTER.md §4. Locked before any training run. These thresholds define PASS/FAIL for the
compression study's central claim and must not be adjusted post-hoc to fit results (master Golden Rule
4: never fabricate a positive; report null honestly).

## Quantitative gate (ratio 0.5, high-curvature bin, mean over 3 seeds) [ENGINEERING]

| Criterion | Threshold |
|---|---|
| Boundary F1, C − B | ≥ 0.04 |
| HD95, C vs B | ≥ 10% relative reduction |
| Dice, B vs C | Equivalent via TOST within ±0.02 |
| Visual restoration | ≥ 6/10 panels judged restored (blind, two raters) |

## Outcome tree

- **Full positive** — all four quantitative/visual criteria above are met, and P5 (C > B′ in
  high-curvature bin) also holds → dCR recovery is real and curvature-specific.
- **Partial** — boundary metrics improve (C > B) but B′ ≈ C → recovery exists but is not
  curvature-specific; report as "generic boundary-awareness recovers most of the damage, not curvature
  per se."
- **Null** — C does not exceed B on boundary metrics → dCR does not recover pruning damage in this
  setup; report honestly, do not retrain-until-positive.
- **Confounded** — the visual gate (§13 of master) fails, or B/B′/C do not have identical
  params/FLOPs (master §9), or any other methodological gate fails → diagnose the confound before
  drawing any conclusion; do not auto-retry with different hyperparameters to chase significance.

## Governing rules

- These thresholds are fixed before training begins and are not tuned afterward.
- A null or partial result is a valid, reportable scientific outcome, not a failure of the project.
- The visual gate (master §13) is a hard STOP: two independent human raters, blind to numeric results,
  must confirm ≥6/10 restored panels before any quantitative claim of success is finalized.

---

## AMENDMENT 2026-08-06 (authorised addition to a LOCKED document)

Added before any training run, on explicit instruction, after Phase 2 measured the actual test-bin
occupancy. The thresholds and outcome tree above are UNCHANGED; this section only scopes which
tests are confirmatory.

### Statistical scope

Test-bin occupancy (measured, Phase 2): **low=50, mid=12, high=36**. The mid-curvature
bin is underpowered by construction (a consequence of A17's group-atomic leakage-free
split being determined independently of curvature stratification) and its Wilcoxon
results will be reported descriptively only, with power explicitly stated, never used
to support or refute P1–P6. The primary hypothesis tests (P2, P4, P5) are scoped to
the high-curvature bin (n=36) vs low-curvature bin (n=50) comparison, which was
always the theoretically central contrast. This is a disclosed limitation, not a
retroactive weakening of the hypothesis.

### Measured power at the actual n (`docs/power_analysis.py`, master §6)

Paired design; Wilcoxon signed-rank primary (ARE = 3/π ≈ 0.955 vs the paired t-test); Cohen's d
is for PAIRED differences. Nominal α = 0.05 two-sided; "Holm" = 0.05/16, the most stringent
threshold any test in the corrected confirmatory family faces.

| bin | n | power @ d=0.5 (α=.05) | power @ d=0.8 (α=.05) | MDE @ 80% power | verdict |
|---|---|---|---|---|---|
| low  | **50** | **0.923** | 1.000 | d ≥ 0.414 | adequate for medium effects |
| mid  | **12** | 0.337 | 0.689 | d ≥ 0.914 | **underpowered even for large effects** |
| high | **36** | **0.812** | 0.995 | d ≥ 0.492 | adequate for medium effects |

Under the Holm worst-case α, power at d=0.5 is 0.632 (low), 0.056 (mid), 0.415 (high). A medium
effect requires n ≥ 35 at 80% power, so **n=12 cannot reach 80% power for any effect below
d≈0.91** — hence descriptive-only. Note these are a floor: the paired design gains power from
the positive across-image correlation between models, which is not credited here.

### Confirmatory test family for Holm–Bonferroni correction (Phase 8)

Master §6 specifies Holm–Bonferroni "across ~24 tests". Enumerated, that original family is:

> 2 comparisons (B vs C, B′ vs C) × 3 bins × 2 pruning ratios (0.5, 0.7) × 2 boundary metrics
> (Boundary F1, HD95) = **24** — matching master §6's figure. Dice is excluded from this family:
> it is tested for *equivalence* via TOST (master §4), not superiority.

**Corrected family size: 24 → 16.** Excluding the mid bin, whose tests are descriptive-only:

> 2 comparisons × **2 bins (low, high)** × 2 ratios × 2 metrics = **16**

Excluding descriptive-only tests from the corrected family is required for correctness, not a
convenience: Holm controls the family-wise error rate over the tests that carry confirmatory
claims, and padding it with tests that can never support or refute a hypothesis would dilute
correction strength on the tests that do. The most stringent threshold relaxes from
α/24 = 0.002083 to α/16 = 0.003125 (1.50×), recovering ~5.3 points of power at d=0.5 in the high
bin (0.362 → 0.415) and lowering its 80%-power MDE from d ≥ 0.720 to d ≥ 0.693.

Mid-bin Wilcoxon results are still computed and reported — with their power stated inline — but
sit outside the corrected family and outside the P1–P6 evidence base.
