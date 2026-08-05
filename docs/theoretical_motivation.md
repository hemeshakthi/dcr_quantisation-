# Theoretical Motivation

**LOCKED — do not modify after implementation begins.**

Source: PROJECT_MASTER.md §2. Reproduced here verbatim as the standing rationale for the compression
study; any change to this reasoning must be treated as a change to the study's frozen methodology
(master §0, STOP CONDITION) and escalated, not edited in place.

## Mechanism [ENGINEERING, grounded in [LITERATURE]]

Structured pruning reduces skip-connection bandwidth — the pathways carrying high-frequency boundary
detail from encoder to decoder. L1-norm pruning can't protect boundary-critical channels, so pruning
damages high-curvature boundary geometry while sparing Dice. dCR concentrates gradient on boundary-band
pixels weighted by curvature, recovering that damage. This is complementary to MedCore (Zhang 2026,
arXiv:2605.13688), which changes the pruning criterion; this study changes the fine-tuning loss instead.

## Six predictions (locked, master §2)

- **P1** — Pruning preserves Dice but degrades Boundary F1/HD95.
- **P2** — Degradation is larger in the high-curvature bin than in low/mid bins.
- **P3** — Model C outperforms Model B on boundary metrics.
- **P4** — Recovery (C over B) is concentrated in the high-curvature bin.
- **P5** — Model C outperforms Model B′ in the high-curvature bin specifically (i.e., the recovery is
  curvature-specific, not just "any boundary-aware loss helps").
- **P6** — dCR reduces curvature-related failure modes more than non-curvature-related ones (B→C).

## Relationship to the frozen research question

Research question (master §0b, FROZEN): *Can dCR loss recover the boundary geometry degradation
introduced by structured channel pruning in microscopy cell segmentation?*

P1–P2 establish that there is damage to recover and that it is curvature-concentrated. P3–P4 test
whether dCR recovers it. P5 is the critical falsification test: it isolates whether the recovery is
specifically due to curvature-awareness (dCR) rather than any generic boundary-band loss (B′). P6
provides an independent, failure-mode-level corroboration of P5 rather than relying on aggregate metrics
alone.

This document exists so that every later phase (training, evaluation, statistics, failure taxonomy) can
be checked against a single frozen statement of what the study is trying to show — no phase may
introduce a goal, metric, or comparison that isn't traceable back to P1–P6.
