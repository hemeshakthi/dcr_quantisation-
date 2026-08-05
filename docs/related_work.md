# Related Work / Lineage

**LOCKED — do not modify after implementation begins.**

Source: PROJECT_MASTER.md §3, cross-referenced with §0a and docs/source_review.md. This document fixes
the study's positioning relative to prior work so that scope creep (master §1) can be checked against a
single reference list.

## Closest related work

- **MedCore (Zhang 2026, arXiv:2605.13688)** [LITERATURE] — changes the pruning *criterion* on
  MedSAM/ViT-family models. This study is complementary: it holds the pruning criterion fixed (L1-norm
  structured pruning, master §9) and instead changes the *fine-tuning loss* after pruning. The two
  approaches are not competitors and could in principle be combined in future work (out of scope here,
  master §1).

## dCR ancestors [PAPER references, per master §0a]

- **Bending loss** (Wang 2020, paper ref 30) — direct ancestor of dCR; penalizes contour bending
  energy for nuclei segmentation.
- **OCT curvature loss** (Xing 2022, paper ref 29) — direct ancestor of dCR; curvature-aware loss for
  OCT layer segmentation.
- dCR's claimed novelty (per the paper's abstract) is formulating curvature regularization in a
  "curvature embedding space," differentiating it from both ancestors. Neither ancestor loss has been
  tested under structured compression/pruning — that gap is exactly what this study addresses.
- Supporting curvature theory cited by the paper: total curvature (Goldluecke 2011, ref 27),
  morphological curvature evolution (Marquez-Neila 2013, ref 33), contour-curvature perception
  (Foster 1993, ref 35).

## Boundary-loss family (context for B′, the generic-boundary control) [PAPER references]

- Kervadec boundary loss (ref 15)
- Active Boundary Loss (ref 17)
- Conditional Boundary Loss (ref 18)
- Boundary DoU (ref 19)
- InverseForm (ref 10)

**B′ (Generic Boundary Loss) selection** [LITERATURE, per master §8]: drawn from this cited family
(e.g., Kervadec-style boundary-band loss), sharing the exact same `extract_boundary_band()` utility as
the dCR surrogate (master §8), differing ONLY in weighting (flat vs. curvature-weighted). This shared
extraction is what makes the B′ vs C comparison (P5) a clean test of curvature-specificity rather than a
confounded comparison between two unrelated boundary losses.

## Architecture lineage [PAPER references]

- U-Net for cell counting/morphometry (Falk 2019, ref 36) — the architecture family this study fixes on
  (master §7).
- UNet++ (ref 43) and DeepLab encoder-decoder (ref 44) — used by the original dCR paper as *comparison*
  architectures. Explicitly OUT OF SCOPE for this study (master §1) — this study uses a single fixed
  plain U-Net so that architecture is never a confounding variable in the B/B′/C comparison.

## Data lineage [PAPER references + LITERATURE corroboration, see docs/source_review.md §3]

- Authors' prior bright-field cell/blood segmentation work (refs 5, 6, 9), most notably the direct
  predecessor **SBU-net** (Asha, Gopakumar, Subrahmanyam, *Engineering Applications of Artificial
  Intelligence* 2023), which used two public bright-field datasets: T cells and pancreatic cancer
  cells — matching Zenodo 4034929 and 4034939 respectively (see source_review.md and
  implementation_assumptions.md entry A01).
- Zenodo StarDist/TrackMate flow-chamber dataset (ref 37, Zenodo 4034939).
- Zenodo StarDist/TrackMate T-cell dataset (ref 38, Zenodo 4034929).
- DIC reconstruction dataset (ref 39) — named in the paper's reference list but not corroborated as an
  actually-used dataset by any evidence found in this session; treated as lower-confidence than
  refs 37/38 (source_review.md §3).

## Statistics lineage [PAPER references]

- Student's t-test (ref 40), normal distribution assumption (ref 41), Cohen's power analysis (ref 42) —
  the paper's own statistical apparatus. This study keeps the t-test as a reported secondary statistic
  and adds a fuller nonparametric battery (paired Wilcoxon primary, Holm-Bonferroni correction, TOST,
  rank-biserial effect sizes, bootstrap CIs, partial correlation) per master §6, appropriate for the
  small non-normal per-image samples typical of medical image segmentation studies.
