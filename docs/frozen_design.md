# Frozen Design

**LOCKED — do not modify after implementation begins.**

Source: PROJECT_MASTER.md §§0b, 1, 7, 8, 9. This document consolidates every FROZEN decision that
functional code, once written, must not deviate from. Any change here is a change to the study's
methodology and must go through the master's STOP CONDITION (§0) — re-search, compare, document as
[ASSUMPTION], escalate — not a quiet edit.

## What this project is [ENGINEERING, master §0b]

Publication-quality medical AI research. Every result — especially in the UI — must come from real
trained models. Two identical model outputs across different models is always a bug to find and fix,
never an acceptable result to report.

**Research question (FROZEN):** Can dCR loss recover the boundary geometry degradation introduced by
structured channel pruning in microscopy cell segmentation?

## Four models (FROZEN)

| Model | Definition | Role |
|---|---|---|
| A | Uncompressed U-Net + Dice + dCR | Reference |
| B | Pruned U-Net + Dice | Compressed baseline (damage) |
| B′ | Pruned U-Net + Dice + Generic Boundary Loss | Control (isolates curvature-specificity) |
| C | Pruned U-Net + Dice + dCR | Proposed (recovery) |

Plus a Vanilla U-Net (Dice only, uncompressed) baseline, not counted among the four. Split 70/30
[ENGINEERING — professor-set].

## Out of scope (§1) — never introduce

- NO transformers, Attention U-Net, UNet++ (plain U-Net only, see Architecture below).
- NO quantization, knowledge distillation, deployment research (no TensorRT/Jetson; ONNX only as an
  optional stub).
- NO extra hypotheses or research models beyond Vanilla/A/B/B′/C.
- NO persistent homology, Betti loss, cross-stain evaluation, energy metrics, compression-ordering
  studies.
- NO placeholders, fake graphs, hardcoded metrics, mock backend, or fabricated UI — anywhere, at any
  phase.

## Architecture (FROZEN) — single fixed U-Net [PAPER-family, channel widths [ASSUMPTION] per A08]

- Family: U-Net (Falk 2019, ref 36 lineage). UNet++ (ref 43) / DeepLab (ref 44) are the *original
  paper's* comparison architectures — not used here.
- Plain U-Net: encoder [64,128,256,512,1024], bottleneck 1024, decoder [512,256,128,64], final
  `Conv2d(64,1,1)` with NO activation, input `(B,1,H,W)`, output `(B,1,H,W)` logits.
- **Why plain U-Net, not Attention U-Net or UNet++** [ENGINEERING]: skip connections are the mechanism
  under study. Attention U-Net's learned gating would adapt to pruning, confounding the B/B′/C
  comparison. UNet++'s dense skip connections give redundant boundary pathways that would artificially
  protect Model B and shrink the B-C gap, also a confound. A single fixed backbone keeps architecture
  out of the causal chain entirely — only the loss function varies across B/B′/C.

## Losses (FROZEN, master §8)

- **DiceLoss** [PAPER]: soft Dice, `smooth=1e-5`.
- **DCRLoss** [ASSUMPTION — see implementation_assumptions.md A06]: curvature regularization surrogate
  — boundary-band BCE weighted by a differentiable curvature proxy (Sobel-gradient magnitude on the
  prediction). Fully differentiable, no numpy in the forward pass. Distinguished from bending loss
  (Wang 2020) and OCT curvature loss (Xing 2022) per related_work.md. **Must be replaced with the real
  equation/code the instant the professor provides it — this is the highest-impact open assumption in
  the project.**
- **GenericBoundaryLoss (B′)** [LITERATURE, from the paper's boundary-loss family — e.g. Kervadec
  ref 15]: shares `extract_boundary_band()` pixel-for-pixel with DCRLoss, uses FLAT weighting, no
  curvature term. This shared extraction is what makes B′ vs C a clean curvature-specificity test
  (P5).
- **CombinedLoss** [ENGINEERING]: `α·Dice + β·(dcr|generic)`, with
  `calibrate_beta_by_gradient_norm()` so that B′'s β is calibrated to match dCR's gradient-norm
  contribution at equal α — otherwise a naive equal-β comparison would confound "curvature-specific"
  with "differently-scaled."

## Structured pruning (FROZEN protocol, master §9) — the study's actual new contribution

L1-norm structured channel pruning. Ratios: 0.5 (primary), 0.7 (robustness check). Protocol:

1. Train Model A.
2. Prune ONCE per ratio — save the per-layer kept-channel index mask (JSON) plus a hash of it.
3. B, B′, and C all load the SAME pruned base for a given ratio. None of them re-prunes independently.
4. Prune BatchNorm parameters in lockstep with their associated Conv layer.
5. Recalibrate BatchNorm via 1–2k frozen forward passes, using identical images and identical order
   across B/B′/C.
6. Fine-tune B, B′, C from that shared pruned+recalibrated base.

Compression accounting uses MEASURED params and FLOPs (via `thop`), not theoretical estimates. B, B′,
and C at a given ratio MUST have identical measured params/FLOPs — if they don't, that's a pruning-step
bug, not a reportable result (this is the pruning-side analogue of "two identical model outputs = bug").

## Why these choices cannot be silently changed

Every item above is either directly reproduced from the paper ([PAPER]), or is an explicit engineering
choice made to prevent a specific, named confound in the B/B′/C comparison. Changing any of them without
going back through master §0's STOP CONDITION would invalidate the causal interpretation of P1–P6
(see theoretical_motivation.md) even if it produced "better-looking" numbers.
