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

# Phase 1 additions (data acquisition, QC, splitting) — added 2026-08-06

### A09 — Working dataset set restricted to the two Zenodo StarDist/TrackMate records
- Status: OPEN
- Missing info: Which datasets the pipeline should actually operate on.
- Sources searched: See A01 and docs/source_review.md §3. Additionally, both Zenodo records
  were verified live this phase via the Zenodo REST API and by reading each archive's ZIP
  central directory over HTTP range requests (file names, sizes, and counts confirmed —
  not assumed).
- Why unrecoverable: Paper body paywalled; only refs 37/38 name datasets publicly.
- Alternatives considered: Adding BCCD/Cellpose/DeepBacs/HeLa as additional training data.
  Rejected for now — none is named anywhere in the paper's public record, so including them
  would silently invent the study's data basis. BCCD additionally has no usable masks (A16).
- Final decision: `PRIMARY_CANDIDATES = [stardist_tcell, stardist_flowchamber]`
  (Zenodo 4034929, 4034939). All other registry entries are MANUAL-only: the downloader
  refuses to fetch them and writes `status: NOT_ACQUIRED`.
- Scientific justification: Uses exactly the datasets the paper's own public reference list
  names, and no more. Any dataset outside this set would be a pure invention.
- Expected reproducibility impact: HIGH if the professor names different datasets — the split
  and every trained model would have to be regenerated. Verified native properties for the
  record: both are `.tif`, 1024×1024, brightfield; T cell = 645 nm/px (Zeiss Axiovert 200M,
  10× phase contrast), flow chamber = 650 nm/px (Zeiss Laser-TIRF 3, 10×), AsPC1 pancreatic.

### A10 — Image↔mask pairing convention
- Status: OPEN (low risk — convention verified against the real archives)
- Missing info: How images pair with masks inside the ZeroCostDL4Mic archive layout.
- Sources searched: Real ZIP central directories of both archives (verified, not guessed);
  ZeroCostDL4Mic conventions.
- Why unrecoverable: Not documented in the paper; but directly observable from the data.
- Alternatives considered: index-order pairing (rejected — fragile and silent on mismatch).
- Final decision: pair `Training dataset/Training_source/<f>` with
  `Training dataset/Training_target/<f>` by IDENTICAL filename, with a fallback that swaps the
  `Training_source`→`Training_target` filename prefix. Any image without a matching mask is
  reported as a PROBLEM and excluded — never silently dropped or paired by position.
- Scientific justification: Filename-identity pairing is the ZeroCostDL4Mic convention and is
  verifiable per-file; positional pairing could silently mis-pair every image.
- Expected reproducibility impact: LOW — deterministic and self-verifying.

### A11 — Grouped (leakage-safe) splitting instead of random per-image splitting
- Status: OPEN (methodological choice; ratio itself is FROZEN)
- Missing info: Whether the dCR paper's split was group-aware or naive per-frame random.
- Sources searched: Paper body (paywalled); Zenodo dataset descriptions (confirm time-lapse
  acquisition: T cell frames every 30 s over 10 min; flow-chamber filenames encode timepoints
  `t:1-…`, `t:301-…`, `t:601-…` from the SAME field of view).
- Why unrecoverable: Split protocol is in the paywalled Methods section.
- Alternatives considered: (1) naive random per-image 70/30 — REJECTED, it would place frame
  *t* in train and frame *t+1* in test, leaking near-identical cells and inflating every
  boundary metric; (2) filename-heuristic grouping alone — rejected as sole mechanism because
  the heuristic could be wrong; (3) near-duplicate-graph grouping alone — rejected because
  perceptually distinct frames of one movie can still be dependent.
- Final decision: atomic split unit = connected component of
  **(pHash near-duplicate graph) ∪ (filename series groups)**. The 70/30 ratio (master §0b,
  FROZEN) is hit by assigning whole groups. Leakage is then RE-VERIFIED independently after
  assignment, and the split is rejected if any image, near-duplicate pair, or series spans
  train/test.
- Scientific justification: The study's claim is specifically about fine boundary geometry;
  temporal leakage would inflate exactly the metrics under test (Boundary F1, HD95) and could
  manufacture a false positive for P3/P5. Group-aware splitting is standard practice for
  time-lapse microscopy. The union rule degrades safely: if the filename heuristic is wrong,
  grouping falls back to measured perceptual similarity rather than leaking silently.
- Expected reproducibility impact: MEDIUM on absolute metric values (a leaky split would score
  higher), HIGH on validity — this choice protects the study's central claim from a confound.

### A12 — Exclusion of the archives' `Test Data` / `Quality control` subsets
- Status: OPEN
- Missing info: Whether the paper used the archives' own train/test/QC partitions.
- Sources searched: Real ZIP central directories (verified).
- Why unrecoverable: Paper body paywalled.
- Alternatives considered: Using the archives' own `Test Data/` as the study's test set —
  REJECTED because it is a single large UNLABELLED time-lapse stack (T cell: one 57 MB `.tif`
  with no corresponding mask), so it cannot support any supervised metric.
- Final decision: build train/test ONLY from paired `Training dataset` (plus the tiny
  `Quality control dataset` pair where it exists and is properly paired), then apply the
  project's own FROZEN 70/30 grouped split (master §0b). `Test Data/` is excluded.
- Scientific justification: Every image used must have a ground-truth mask; unlabelled stacks
  cannot contribute to Dice/Boundary F1/HD95.
- Expected reproducibility impact: LOW–MEDIUM. Note the resulting corpus is small
  (~209 + ~57 pairs), which is consistent with the paper's stated focus on *small training
  sets*, but it does constrain statistical power — relevant to master §6's power analysis.

### A13 — Near-duplicate detection method and threshold
- Status: OPEN
- Missing info: No authoritative threshold exists; the paper does not discuss deduplication.
- Sources searched: Paper body (paywalled); standard practice in dataset-hygiene literature.
- Alternatives considered: exact-hash only (too weak — misses adjacent frames); SSIM/pixel
  correlation (more sensitive but O(n²) expensive and threshold-arbitrary too).
- Final decision: perceptual hash (`imagehash.phash`, 64-bit) with Hamming distance ≤ **8**
  (12.5% of bits) flagged as near-duplicate. Exposed as `--hamming` and recorded in
  `data/processed/<name>_nearduplicates.json` alongside the full edge list.
- Scientific justification: pHash at ≤8 bits is a widely used near-duplicate operating point;
  because the threshold only ever *merges* images into the same split group, a too-permissive
  threshold costs split granularity but can never cause leakage. Errs safe.
- Expected reproducibility impact: LOW — the full pair list is persisted, so any later
  threshold change is auditable and re-derivable without re-reading images.

### A14 — Binarisation of instance-label masks
- Status: OPEN
- Missing info: Whether the paper trained on binary semantic masks or instance labels.
- Sources searched: Paper body (paywalled). StarDist archives ship *instance-label* masks
  (each cell a distinct integer) because StarDist is an instance-segmentation method.
- Alternatives considered: instance segmentation (out of scope — master §7 fixes a plain
  U-Net with a single-channel logit output, i.e. binary semantic segmentation).
- Final decision: binarise masks as `mask > 0` for training and for Dice/Boundary F1/HD95.
  Retain the original instance labels for per-cell statistics (cell size, curvature
  stratification, cell-merging/splitting failure detectors — master §11, §12).
- Scientific justification: Master §7's frozen architecture emits one logit channel, which is
  binary semantic segmentation by construction. Keeping instance labels alongside preserves the
  per-cell information the curvature stratification and failure taxonomy need.
- Expected reproducibility impact: LOW for Dice; MEDIUM for the cell-merging/splitting failure
  detectors, which depend on instance identity being available — hence it is retained, not
  discarded.

### A15 — Split seed
- Status: OPEN
- Missing info: The paper's split seed (almost certainly unreported).
- Alternatives considered: any arbitrary integer.
- Final decision: seed **42**, matching the seed master §13 fixes for visual-gate
  pre-selection, so the data pipeline uses one auditable project-wide constant. Recorded in
  `data/splits/split_manifest.json`.
- Scientific justification: The seed only shuffles group assignment order; the leakage
  guarantee and the 70/30 ratio hold for any seed. Logging it makes the split exactly
  reproducible.
- Expected reproducibility impact: LOW, and fully mitigated by the write-once split artefacts.

### A16 — BCCD segmentation-mask source (BLOCKING if BCCD is in scope)
- Status: OPEN — **BLOCKING**
- Missing info: Where BCCD segmentation masks would come from at all.
- Sources searched: github.com/Shenggan/BCCD_Dataset — ships PASCAL VOC **bounding-box** XML
  annotations only, no masks.
- Why unrecoverable: They do not exist in the dataset as published.
- Alternatives considered: deriving masks from boxes — **REJECTED AND FORBIDDEN** by master §5
  ("do not fabricate masks from boxes"). Rectangular pseudo-masks have no meaningful curvature
  and would silently destroy curvature stratification (§11), boundary metrics (§6), and the
  failure taxonomy (§12) — i.e. they would corrupt the exact quantities this study measures.
- Final decision: BCCD is **excluded**. The pipeline will not generate BCCD masks under any
  flag. `download_datasets.py` and `quality_check.py` both print a blocking TODO banner.
- Scientific justification: Fabricated geometry cannot support a geometry claim (Golden Rule 1).
- Expected reproducibility impact: None while excluded. If the professor confirms BCCD is in
  scope, a genuine mask source must be supplied before any BCCD work begins.

### A17 — Split stratification level, and the resulting 70/30 deviation
- Status: OPEN — **DECISION REQUIRED FROM PROFESSOR**
- Missing info: Whether the FROZEN 70/30 ratio (master §0b) is to be hit exactly on the
  pooled corpus, or applied within each acquisition condition.
- Why this arose: measured, not hypothesised. The T-cell corpus contains only **2 atomic
  T-cell movie groups** (66 and 56 images) out of 210 images. Because acquisition series must
  stay atomic to prevent temporal leakage (A11), no group-atomic assignment can hit 70/30
  *within* the 'T cells' condition — the only options are 66/56 (54%) or 122/0 (100%).
- Alternatives considered and MEASURED (seed 42, leakage PASSED in every case):
  | Mode | Global train frac | Per-dataset | Composition |
  |---|---|---|---|
  | pooled across datasets | 0.730 | tcell 0.848 / flow 0.298 | T cells absent from test; flow-chamber ratio INVERTED |
  | `--no-stratify` (per dataset) | 0.712 | tcell 0.714 / flow 0.702 | T cells absent from TEST, VCAM absent from TRAIN |
  | `--stratify` (per condition, **DEFAULT**) | 0.633 | tcell 0.614 / flow 0.702 | every condition on BOTH sides |
- Final decision: **`--stratify` is the default.** The pooled mode was rejected outright as a
  bug (one dataset's ratio silently compensating for the other's). Between the remaining two,
  a test set containing ZERO T cells — 58% of that corpus — would confound compression damage
  with domain shift and could not support any claim about boundary geometry generalising.
  A ratio of 0.633 vs 0.700 is a **−0.067 deviation** recorded explicitly in
  `data/splits/split_manifest.json` as `train_fraction_deviation`.
- Scientific justification: The FROZEN quantity is a 70/30 *split*; both surviving modes
  honour that target, differing only in the population it is computed over. Applying it within
  each condition is the standard reading for a heterogeneous corpus. The residual deviation is
  forced by the data's group structure, not by a design change — it cannot be removed without
  either leaking (A11) or emptying a condition from one side.
- Expected reproducibility impact: MEDIUM. Absolute metrics shift slightly with less training
  data (169 vs 190 images); validity improves substantially. **If the professor requires an
  exact 70/30, re-run with `--no-stratify` and the covariate-shift caveat must then be stated
  explicitly in the paper.** Both modes are reproducible from the same seed.

### A18 — Image reading backend and host memory constraints (engineering, not scientific)
- Status: OPEN (informational — no scientific content, recorded for reproducibility)
- Missing info: n/a. Recorded because it changed how the code is written.
- What was measured: (1) the installed `tifffile` (2026.5.15) calls
  `ndarray.reshape(copy=False)`, which requires numpy ≥ 2.0, while this environment has numpy
  1.26.4 — so tifffile raises `TypeError` on **every** `.tif` in both datasets (420 + 554
  occurrences). (2) The host's Windows commit limit equals physical RAM (pagefile effectively
  disabled): `FreeVirtualMB ≈ 936`, causing `MemoryError` on allocations as small as 15 MiB.
- Final decision: (a) `read_image()` tries tifffile → PIL → cv2 and reports a file as corrupt
  ONLY if every backend fails, with per-backend failure counts printed, so a library
  incompatibility can never be silently misreported as data corruption; (b) normalisation uses
  float32 with strided-subsample percentiles; (c) QC overlays are rendered as uint8 arrays via
  cv2 instead of matplotlib, whose colormap path expands to float64 RGBA; (d) the downloader
  uses 256 KiB chunks with resume and 5 retries.
- Scientific justification: None of these touch the arrays used for metrics — they affect
  display normalisation and IO robustness only. Note the environment does NOT match
  `requirements.txt` (pinned numpy 1.24.3, torch 2.1.0; installed numpy 1.26.4, torch 2.3.1+cpu),
  and `tifffile` is not pinned at all. **`tifffile` should be added to requirements.txt with a
  numpy-compatible pin before training.**
- Expected reproducibility impact: LOW scientifically; HIGH operationally — this host cannot
  run the 24 training runs (master §17 already routes those to Kaggle).

---

## Open questions requiring professor input (cross-ref docs/source_review.md §5)

1. Full paper PDF / paywall-free access.
2. Exact dataset(s) + links (A01, A09) — **and confirmation of the working hypothesis
   Zenodo 4034929 + 4034939**.
3. Exact train/val/test split protocol (A02); was it group-aware given the time-lapse data (A11)?
4. Exact dCR loss equation / code (A06).
5. Image resolution, normalization, augmentation (A03).
6. Optimizer/LR/scheduler/weight decay/batch/epochs/init/stopping (A04, A05).
7. Post-processing (A07).
8. Exact U-Net channel widths, if non-standard (A08).
9. Any unofficial/official code the professor may be aware of.
10. Which archive subsets were used — `Training dataset` only, or also `Test Data`/`Quality
    control`? (A12)
11. Were masks binarised or used as instance labels? (A14)
12. **If BCCD is genuinely part of the study, what is the source of its segmentation masks?**
    (A16 — blocking)
13. **Split stratification (A17):** accept the condition-stratified split (0.633 train, every
    condition represented on both sides), or require an exact 70/30 at the cost of the test set
    containing no T cells at all? This is the one Phase 1 decision that materially affects what
    the study can claim.

**No entry in this file may be marked CONFIRMED without a specific source (PDF page, professor message,
official repo commit) recorded in the Status line.**
