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
