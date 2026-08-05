# Implementation Assumptions Log

**LOCKED FORMAT — every entry below is an [ASSUMPTION] per PROJECT_MASTER.md §0.** No entry may be
silently made; every one must be logged here at the moment the assumption is introduced, using the
template below. When the professor or the paper PDF confirms a value, update the entry's status to
CONFIRMED, re-label it [PAPER]/[OFFICIAL] in the relevant code/doc, and note the confirmation date and
source here — do not delete the historical record.

## Template

```
### <short id> — <one-line title>
- Status: OPEN | CONFIRMED (date, source)
- Missing info:
- Sources searched:
- Why unrecoverable:
- Alternatives considered:
- Final decision:
- Scientific justification:
- Expected reproducibility impact:
```

---

### A01 — Exact dataset identity and source links
- Status: OPEN
- Missing info: Which dataset(s) the dCR paper actually trained/evaluated on, and their exact
  download links/DOIs.
- Sources searched: Springer article page (paywalled, redirects to login), Semantic Scholar API
  (abstract/references elided by publisher), Unpaywall API (confirmed closed access, no repository
  copy), predecessor paper by the same authors (SBU-net, EAAI 2023), Zenodo records 4034929 and
  4034939, author faculty page (amrita.edu/faculty/gopakumarg/), GitHub/arXiv searches for an official
  repo. See docs/source_review.md §3 for full trail.
- Why unrecoverable: Paper body confirmed paywalled (Unpaywall: is_oa=false, oa_status=closed,
  has_repository_copy=false); the only public reference-list evidence (refs 37/38) names Zenodo
  4034939 and 4034929 but the body text that would confirm actual usage is inaccessible.
- Alternatives considered: BCCD (bounding boxes only, no masks — rejected as primary), Cellpose cyto
  (non-commercial license, different modality/domain), DeepBacs E. coli (different cell morphology,
  bacterial not mammalian), generic HeLa bright-field sets (source ambiguous).
- Final decision: Working hypothesis = Zenodo 4034929 (T-cell) + Zenodo 4034939 (AsPC1 pancreatic
  flow-chamber), both StarDist/TrackMate/ZeroCostDL4Mic brightfield sets, CC-BY. This is the strongest
  defensible choice because the same author group's immediate predecessor paper (SBU-net, EAAI 2023)
  explicitly used "two publicly available bright-field datasets of T cells and pancreatic cancer
  cells," which matches these two Zenodo records' descriptions exactly (T cells on ICAM-1; AsPC1
  pancreatic cancer cells in a microfluidic chamber).
- Scientific justification: Same lab, same PhD lineage (Asha S. Benny under Gopakumar), same problem
  framing (bright-field cell boundary segmentation), continuity of dataset reuse across a research
  program is a common and defensible pattern.
- Expected reproducibility impact: HIGH if wrong — different datasets have different cell morphology,
  contrast, and image counts, which would shift Dice/curvature-stratification results materially. MUST
  be confirmed with the professor before Phase 1 (data pipeline) finalizes.

---

### A02 — Exact train/val/test split protocol
- Status: OPEN
- Missing info: Whether the dCR paper used a simple holdout split, k-fold cross-validation (the
  predecessor SBU-net paper used 5-fold), and the exact split ratios/seeds.
- Sources searched: Same as A01 (paywalled body).
- Why unrecoverable: Not present in any publicly accessible abstract/metadata.
- Alternatives considered: Match predecessor's 5-fold CV; simple 70/30 holdout; 80/10/10 train/val/test.
- Final decision: Use 70/30 split per PROJECT_MASTER.md §0b (explicitly professor-set for THIS
  compression-study follow-up, overriding the paper's own protocol since this is a new experiment, not
  a strict paper reproduction). Labeled [ENGINEERING] in master, not re-labeled here.
- Scientific justification: The compression study is a novel follow-up question, not a reproduction of
  the paper's exact numbers, so a fixed 70/30 split (professor directive) takes precedence over
  recovering the original paper's split.
- Expected reproducibility impact: LOW for the compression study itself (split is frozen by directive);
  MEDIUM if a direct paper-number reproduction is later requested.

---

### A03 — Image resolution, normalization, and augmentation pipeline
- Status: OPEN
- Missing info: Input image resolution, pixel normalization scheme, and augmentation policy (flips,
  rotations, elastic deform, intensity jitter, etc.) used in the dCR paper.
- Sources searched: Paper body (paywalled), predecessor SBU-net paper (abstract/snippets only, no
  preprocessing detail surfaced), Zenodo dataset pages (raw image specs only, not the paper's
  preprocessing choices).
- Why unrecoverable: Preprocessing detail lives in the paper's Methods section (paywalled).
- Alternatives considered: Deferred — no default chosen yet; will be set explicitly in Phase 1 (data
  pipeline) with its own [ASSUMPTION] entry citing standard microscopy-segmentation practice
  (min-max or z-score normalization; flip/rotate/elastic augmentation) if still unconfirmed at that
  point.
- Final decision: NOT YET SET. To be decided at Phase 1 start.
- Scientific justification: N/A pending Phase 1.
- Expected reproducibility impact: MEDIUM — affects absolute metric values but unlikely to change the
  compression study's directional conclusions (B vs C comparisons), since all models in the study share
  identical preprocessing regardless of which scheme is chosen.

---

### A04 — Optimizer, learning rate, scheduler, weight decay
- Status: OPEN
- Missing info: Optimizer type, initial LR, LR schedule, weight decay used in the paper's training.
- Sources searched: Paper body (paywalled); no official code found (docs/source_review.md §2).
- Why unrecoverable: Training hyperparameters live in the paper's Methods/Experiments section
  (paywalled); no official repo exists to inspect a config file.
- Alternatives considered: SGD+momentum (common in older U-Net literature) vs. Adam/AdamW (common in
  recent segmentation literature).
- Final decision: Default per master §10 — Adam, lr 1e-4, weight decay 1e-5, ReduceLROnPlateau. To be
  finalized/confirmed at Phase 2 (training pipeline) start.
- Scientific justification: Adam with a modest LR is the de facto default for U-Net-family medical
  image segmentation in 2020s literature; low risk of being a confound for the compression study since
  identical optimizer/LR is enforced across B/B′/C (master §10).
- Expected reproducibility impact: LOW for the compression study's internal comparisons (B vs B′ vs C
  share the same optimizer by design); MEDIUM if absolute Dice numbers are compared against the
  paper's reported values.

---

### A05 — Epoch count, batch size, stopping criterion, initialization
- Status: OPEN
- Missing info: Number of training epochs, batch size, early-stopping criterion, weight initialization
  scheme used in the paper.
- Sources searched: Paper body (paywalled); predecessor paper (5-fold CV mentioned, no epoch/batch
  detail surfaced in accessible snippets).
- Why unrecoverable: Training-loop detail lives in the paywalled Methods section.
- Alternatives considered: Fixed epoch budget vs. plateau-based early stopping.
- Final decision: Default per master §10 — batch 8, ≤100 epochs + plateau early stopping (patience 15
  on val Boundary F1), default framework initialization (Kaiming/He for conv layers). To be
  finalized/confirmed at Phase 2 start.
- Scientific justification: Plateau-based stopping is standard reproducible practice and avoids
  overfitting to an arbitrary fixed epoch count; identical across B/B′/C so it cannot confound the
  compression study's internal comparisons.
- Expected reproducibility impact: LOW for internal B/B′/C comparisons; MEDIUM for absolute
  paper-number reproduction.

---

### A06 — Exact dCR loss equation ("curvature embedding space")
- Status: OPEN
- Missing info: The precise mathematical formulation of the dCR loss — how "curvature embedding space"
  regularization is computed and combined with the segmentation loss.
- Sources searched: Paper body (paywalled), Semantic Scholar (abstract elided), arXiv (no preprint
  found), GitHub (no official implementation found — docs/source_review.md §2), ancestor papers (Wang
  2020 bending loss, Xing 2022 OCT curvature loss — related but explicitly NOT identical per the
  paper's own claimed novelty).
- Why unrecoverable: This is the paper's core technical contribution and lives entirely in the
  paywalled Methods/Equations section; no code release exists to reverse-engineer it from.
- Alternatives considered: (1) Directly reuse Wang 2020 bending loss verbatim — rejected, paper
  explicitly differentiates itself from this; (2) directly reuse Xing 2022 OCT curvature loss verbatim
  — rejected, same reason, also domain-specific to OCT; (3) boundary-band loss weighted by a
  differentiable curvature proxy — selected.
- Final decision: Implement a surrogate per master §8 — boundary-band BCE weighted by a differentiable
  curvature proxy (Sobel-gradient magnitude on the prediction), sharing `extract_boundary_band()` with
  `GenericBoundaryLoss` so the two losses differ ONLY in weighting (flat vs. curvature-weighted). This
  is explicitly a surrogate, not a reproduction of the paper's actual equation.
- Scientific justification: Preserves the paper's stated mechanism (curvature-weighted boundary
  gradient concentration) using a fully differentiable, standard image-processing curvature proxy,
  without fabricating an equation attributed to the authors. Keeps B′ vs C strictly comparable
  (identical boundary-band extraction, differing only in the curvature term) so the compression study's
  P5 prediction (curvature-specificity) remains testable even with a surrogate loss.
- Expected reproducibility impact: HIGH — this is the single most consequential open assumption in the
  entire project. MUST be replaced with the paper's real equation/code the moment the professor
  provides it (master §8, Golden Rule 8). Until then, all C-model results are results for "a
  curvature-weighted boundary loss in the spirit of dCR," not a reproduction of the paper's exact
  method.

---

### A07 — Post-processing of model outputs
- Status: OPEN
- Missing info: Whether the paper applies any post-processing (thresholding scheme, morphological
  cleanup, connected-component filtering, watershed splitting, etc.) before computing Dice/boundary
  metrics.
- Sources searched: Paper body (paywalled).
- Why unrecoverable: Lives in the paywalled Methods/Evaluation section.
- Alternatives considered: Raw sigmoid threshold at 0.5 vs. paper-specific post-processing.
- Final decision: NOT YET SET. Default to raw 0.5-threshold binarization with no post-processing until
  Phase 2/evaluation pipeline, at which point this will be finalized with its own justification if still
  unconfirmed.
- Scientific justification: N/A pending evaluation-pipeline phase.
- Expected reproducibility impact: MEDIUM — affects absolute metric values, particularly HD95 and
  Boundary F1, which are sensitive to small mask artifacts; identical post-processing across all five
  models (Vanilla/A/B/B′/C) keeps internal comparisons valid regardless of the specific choice.

---

### A08 — U-Net exact channel widths
- Status: OPEN (low priority — high-confidence default)
- Missing info: Whether the paper specifies non-standard encoder/decoder channel widths, or whether
  the standard [64,128,256,512,1024] U-Net configuration (Ronneberger et al. 2015 convention) applies.
- Sources searched: Paper body (paywalled); Falk 2019 (ref 36, U-Net for cell counting/morphometry —
  cited by the paper as architecture lineage, uses the conventional U-Net configuration).
- Why unrecoverable: Exact channel widths, if non-standard, would only appear in the paywalled body.
- Alternatives considered: None seriously — [64,128,256,512,1024] is the near-universal default for
  U-Net-family medical segmentation and is what the paper's own cited lineage (Falk 2019) uses.
- Final decision: Use standard [64,128,256,512,1024] per master §7.
- Scientific justification: This is the textbook U-Net configuration used by essentially all cited
  lineage papers; deviating without evidence would be a bigger assumption than following convention.
- Expected reproducibility impact: LOW — standard convention, low risk of being wrong; would only
  matter for exact FLOPs/param-count reproduction of the original paper's reported numbers (not
  relevant to the compression study's internal B/B′/C comparisons, which use measured params/FLOPs
  per master §9 regardless of the base width).

---

## Open questions requiring professor input (cross-ref docs/source_review.md §5)

1. Full paper PDF / paywall-free access.
2. Exact dataset(s) + links (A01).
3. Exact train/val/test split protocol (A02).
4. Exact dCR loss equation / code (A06).
5. Image resolution, normalization, augmentation (A03).
6. Optimizer/LR/scheduler/weight decay/batch/epochs/init/stopping (A04, A05).
7. Post-processing (A07).
8. Exact U-Net channel widths, if non-standard (A08).
9. Any unofficial/official code the professor may be aware of.

**No entry in this file may be marked CONFIRMED without a specific source (PDF page, professor message,
official repo commit) recorded in the Status line.**
