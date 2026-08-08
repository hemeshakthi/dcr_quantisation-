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

#### A06 addendum — Phase 5 implementation, with the two measured deviations from §8's wording

**As implemented** (`src/losses/dcr_loss.py`), per band pixel:

    kappa  = proxy( gaussian_blur(mask, sigma) )
    k_rel  = clamp( kappa / quantile_95_over_band(kappa), 0, 1 )
    w      = 1 + gamma * k_rel ** power          # bounded in [1, 1+gamma]
    loss   = weighted-mean BCE over extract_boundary_band(target, band_width)

Defaults: `band_width=3`, `gamma=1.0`, `power=1.0`, `sigma=2.0`, `proxy=sobel_levelset`,
`curvature_source=target`, `normalise_weights=true`.

**Deviation 1 — the proxy is level-set curvature, not raw |∇p|.** Master §8 words the proxy as
"Sobel-gradient magnitude". Taken literally, |∇p| measures edge STRENGTH, which is near-constant
along a boundary and therefore near-identical at a sharp corner and along a straight edge.
MEASURED on a synthetic square: raw |∇p| gives a corner/edge weight ratio of **0.935** — it
weights corners slightly LESS than straight edges. Weighting by it would make dCR a flat
boundary loss up to a scale factor, i.e. numerically almost the same objective as
GenericBoundaryLoss, and **the B′-vs-C comparison (P5) would be testing nothing**. We therefore
build the proxy FROM Sobel derivatives as §8 directs, but form the textbook level-set curvature
κ = div(∇p/|∇p|), measured corner/edge ratio **1.661**. `sobel_magnitude` stays in the registry
so the literal reading remains available and auditable, but it is not the default.

**Deviation 2 — the mask must be Gaussian-smoothed before differentiating.** Curvature of an
unsmoothed binary mask is the second derivative of a step function: it reports the pixel
staircase, not the shape. MEASURED, level-set κ on a synthetic square:

| σ | corner κ | straight-edge κ | discriminates? |
|---|---|---|---|
| 0 (hard mask) | 0.3166 | 0.3333 | **no** — edge scores *higher* |
| 1.0 | 0.1877 | 0.00000 | yes |
| **2.0 (default)** | **0.1770** | **0.00000** | yes |

At σ=2 a straight edge has EXACTLY zero curvature, as it should, and only corners survive. This
is the same phenomenon A27 documented for the §11 contour statistic, reached independently
here — and σ=2 matches A27's authorised value, so the two places now agree.

**Bounded weighting.** Normalising by a 95th-percentile over the band and clamping to [0,1]
keeps `w ∈ [1, 1+gamma]`, so `gamma` is directly interpretable as "a maximally-curved pixel gets
at most (1+gamma)× the weight of a straight-edge pixel". Normalising by the mean instead would
let a few very sharp corners push weights to arbitrary multiples and effectively switch off
straight-edge supervision; a quantile rather than the max stops one noisy pixel setting the
scale.

**gamma=0 collapses dCR EXACTLY to GenericBoundaryLoss** — verified numerically to <1e-6, both
in the weight map and in the loss value. The control is therefore an exact special case of the
treatment, not a lookalike implementation, which is a stronger guarantee for P5 than two
separate code paths that merely look similar.

**Swap point for the real equation.** The proxy is behind a registry:
`register_curvature_proxy("official", fn)` then `curvature_proxy: official` in the model YAML.
No other code changes. Set `curvature_smooth_sigma: 0` if the official formulation does its own
smoothing.

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

# Phase 2 additions (preprocessing, augmentation, curvature stratification) — added 2026-08-06

**Supersedes A03's "NOT YET SET"** for resolution/normalisation/augmentation. A03 remains OPEN
as to what the PAPER did; A19/A20/A23 record what THIS study does and why.

### A19 — Target resolution 1024×1024
- Status: OPEN (paper's value unrecoverable); decision MEASURED, not conventional
- Missing info: The resolution the dCR paper trained at.
- Measured basis: median instance equivalent-radius is **9.9 px** (tcell, native 1024) and
  **15.2 px** (flowchamber, native 1400); p5 radii 7.8 / 11.7 px.
- Alternatives considered, with the measured consequence at each:
  | Target | tcell median Ø | flow median Ø | consequence |
  |---|---|---|---|
  | 256 | 4.9 px | 5.5 px | cells ~5 px ACROSS — curvature destroyed; rejected outright |
  | 512 | 9.9 px | 11.1 px | radius ~5 px, so master §6's **5 px Boundary-F1 tolerance ≈ the entire cell radius** — the study's primary metric stops discriminating |
  | **1024** | **19.8 px** | **22.2 px** | 2 px tol = 20% of radius, 5 px = 50%; workable |
- Final decision: **1024×1024.** It is the NATIVE size of stardist_tcell, so **210 of 267
  images (79%) are never resampled at all** — no interpolation of the majority of the study's
  ground-truth boundaries. Only flowchamber is rescaled, by 0.731×. 1024 is divisible by 16 as
  the frozen 4-pooling U-Net requires; native 1400 is not (87.5), so flowchamber could not have
  been left at native size regardless.
- Scientific justification: this study measures boundary geometry; downsampling is precisely
  the operation that destroys it, and the metric tolerances are fixed in pixels.
- Expected reproducibility impact: HIGH on absolute metrics. **Operational note:** at 1024²
  a U-Net [64…1024] will not fit batch 8 on a 16 GB T4; batch size (itself [ASSUMPTION] A05)
  will need to drop to ~2–4, or random-crop training used. Flag before Phase 3.

### A20 — Per-image percentile normalisation (p1 / p99.8), clip=False  **[LITERATURE]**
- Status: **RESOLVED 2026-08-06 — relabelled [ASSUMPTION] → [LITERATURE].** This is not a
  guess made in the absence of evidence: it is the published CSBDeep/StarDist `normalize()`
  convention (Weigert et al., *Content-aware image restoration*, Nat Methods 2018; Schmidt
  et al., *Cell Detection with Star-convex Polygons*, MICCAI 2018), and specifically the
  convention used by the **ZeroCostDL4Mic/StarDist toolchain that generated the ground-truth
  masks for both of these datasets**. Following the annotation toolchain's own normalisation is
  the best-supported available choice, independent of what the dCR paper did.
  What remains unknown — and stays open under A03 — is only whether the dCR *paper* used this
  same scheme; that does not affect the standing of the choice itself.
- Measured basis: the two datasets occupy **completely different raw uint16 ranges** —
  tcell p1..p99.8 ≈ 2477..3021 (global extremes 1300..3803); flowchamber ≈ 8692..16647
  (global 925..28383) — against a 65535 full scale. Neither uses more than ~6% of the range.
- Alternatives considered: fixed `/65535` scaling — **rejected on measurement**: it would map
  tcell into ≈0.038..0.046, i.e. almost no contrast, and make the two datasets mutually
  unnormalisable. `mean=0.5/std=0.5` — rejected, it presumes a full-range 8-bit source that
  does not exist here.
- Final decision: CSBDeep/StarDist `normalize(x, 1, 99.8)` per image, `clip=False`, float32.
- Scientific justification: per-image percentile normalisation is *mandatory* given the ranges
  above, and this specific convention is the one used by the ZeroCostDL4Mic/StarDist pipeline
  that produced these datasets' ground truth. `clip=False` (CSBDeep default) keeps faint
  boundary contrast distinguishable rather than flattening outliers.
- Expected reproducibility impact: MEDIUM absolute, LOW directional (identical for all models).

### A21 — Resampling kernels
- Status: OPEN (low risk)
- Final decision: images **INTER_AREA** when downscaling (correct area-average decimation,
  avoids aliasing), INTER_LINEAR if ever upscaling; **both mask types INTER_NEAREST**.
- Scientific justification: nearest is the only kernel that cannot blend two instance IDs into
  a third nonexistent label, or soften a boundary this study then measures.
- Verified: instance counts are **exactly preserved** through the 1400→1024 resize —
  3678 → 3678 (flowchamber) and 24608 → 24608 (tcell, unresampled). Zero instances lost.

### A22 — Alignment-test tolerance (±5%)
- Status: OPEN (methodological, engineering)
- What happened: three alignment statistics were tried; the first two were REJECTED on measured
  evidence and are recorded so they are not reintroduced.
  1. *foreground centroid vs darkest-pixel centroid* — defeated by strong illumination
     vignetting; reported ~90 px offsets on demonstrably well-aligned pairs.
  2. *mean intensity under the mask, minimised at zero shift* — defeated by the phase-contrast
     donut profile. Measured radial profile: 1–2 px inside the boundary = **0.123** (dark rim),
     6–20 px inside = **0.656** (bright core, *brighter than* the 0.496 background). Flagged
     152/267 correctly-aligned pairs.
  3. **boundary Sobel-gradient under the 1 px contour, maximised at zero shift** — adopted.
     Measured on a representative image: 1.068 unshifted vs 0.94–1.01 at ±4 px and 0.63–0.69
     at ±12 px.
- Final decision: pass if the aligned boundary gradient is within **−5%** of the best shifted
  one. Measured: 6/267 images tie strictly (margins −0.0013..−0.0114 against a corpus median of
  **+0.087**); those 6 carry ~186 instances vs a corpus median of 100, i.e. they are the most
  crowded fields where a shift lands on a *neighbouring* cell's edge. Visual inspection of the
  worst case confirmed correct registration.
- Result: **all 267 images pass alignment**; 0 problems.

### A23 — Augmentation implemented directly on cv2/numpy, not albumentations
- Status: OPEN
- Missing info: the paper's augmentation policy (A03, paywalled).
- Why not albumentations: master §16 pins **1.3.1**; the environment has **2.0.8**, whose
  rewrite changed transform signatures and internal RNG handling. Master §10 demands full
  determinism with an augmentation seed SHARED across B/B′/C, and the stream must be identical
  between this Windows host and Kaggle's Linux runners — a version-dependent RNG would silently
  desynchronise them and break the B/B′/C comparison at its foundation.
- Final decision: explicit implementation driven by `np.random.default_rng([seed, epoch, index])`.
  Policy: flips + rot90 (exact label-preserving symmetries), mild affine (±15°, ±10% scale,
  ±5% shift), **mild+infrequent elastic** (p=0.25, α=12 px, σ=16), light brightness/contrast
  (±10%) and Gaussian noise (σ=0.02).
- Scientific justification: elastic deformation is deliberately mild because this study
  **measures curvature** — a strong elastic field would manufacture curvature and contaminate
  the very signal being binned (master §11). No cutout/erasing: it would fabricate false
  boundaries.
- Verified: 8/8 self-tests pass — repeatable per (seed, epoch, index), reproducible from a
  fresh instance, varies with index and seed, binary mask stays exactly the instance support,
  **no invented instance IDs**, shapes and dtypes preserved.

### A24 — Minimum contour length of 16 points
- Status: OPEN
- Final decision: contours with <16 points are excluded from the curvature statistic.
- Scientific justification: discrete curvature on a handful of integer-coordinate points
  reports pixelation, not shape. 16 points ≈ the perimeter of a 3 px-radius object, far below
  the measured median cell radius (9.9–15.2 px), so no real cell is excluded.

### A25 — `contrast` defined as normalised boundary gradient
- Status: OPEN
- Final decision: `contrast` = mean Sobel-gradient magnitude along the GT boundary ÷ image
  intensity std. `contrast_michelson` retained alongside for reference.
- Scientific justification: the paper targets "weak or low **contrast boundaries**", so the
  confound to control for (master §6 partial correlation) is boundary *visibility*. A plain
  foreground-vs-background intensity difference is actively misleading here — the measured
  dark-rim/bright-core profile (A22) makes the two means partially cancel.

### A26 — "3 bins by thirds" read as equal-count tertiles
- Status: OPEN
- Final decision: cut points at the 33rd/67th percentiles of the curvature score, applied
  **per dataset** (`--bin-scope per_dataset`).
- Scientific justification for equal counts: keeps the per-bin paired tests (master §6)
  balanced; equal-width thirds of the range would leave the high bin nearly empty given the
  long right tail, destroying its statistical power. Result: **89/89/89**.
- Scientific justification for per-dataset scope: MEASURED — only **30.5%** of tcell images
  fall in the range shared with flowchamber. Global binning would therefore have largely
  encoded "is a flowchamber image" as "high curvature", confounding curvature with dataset.
- Verified non-degenerate: all bins non-empty, high-bin median (0.2612) > low-bin (0.2512),
  263 distinct scores, and **both train and test contain all three bins**.

### A27 — Contour smoothing σ: master §11's 1 px **OVERRIDDEN to 2 px** [ENGINEERING]
- Status: **RESOLVED / APPLIED 2026-08-06** — authorised override of master §11, on user
  instruction, on the measured basis below. Source label upgraded [ASSUMPTION] → **[ENGINEERING]**:
  this is no longer a guess in the absence of evidence, it is a decision *driven by* measurement.
  `data/curvature/curvature_bins.csv` was DELETED and regenerated at σ=2; master §11 carries an
  amendment pointer to this entry.
- What was measured (24-image sweep) that motivated the override:
  | σ | mean 90th-pct κ | implied radius | between-image CV |
  |---|---|---|---|
  | **1.0 (master §11)** | 0.2654 | **3.8 px** | 4.99% |
  | 2.0 | 0.1783 | 5.6 px | 7.81% |
  | 3.0 | 0.1630 | 6.1 px | 11.01% |
  | 4.0 | 0.1608 | 6.2 px | 12.78% |
- Why this matters: κ=0.2654 implies a **3.8 px radius of curvature — smaller than any real
  cell here** (measured median cell radius 9.9–15.2 px). At σ=1 the statistic is therefore
  substantially reporting the integer-coordinate staircase of the contour rather than cell
  shape. The mean stabilises from σ≥2 (0.178 → 0.163 → 0.161) while between-image
  discriminative spread **more than doubles**.
- Related measured confound: the σ=1 score correlates most strongly with **cell size**
  (Spearman −0.54 tcell / −0.33 flowchamber; −0.58/−0.41 against median cell size) and only
  weakly with cell touching (+0.13/+0.09). The "high-curvature bin" is therefore substantially
  a "small-cell bin" — κ≈1/r. This makes master §6's partial correlation controlling for cell
  size **essential rather than optional**, and means P2/P4/P5 could otherwise be explained by
  "damage is worse for small cells".
- Mitigating evidence at σ=1: the master-mandated 10-image spot check passed — high-bin images
  showed genuinely elongated, pointed/teardrop cells. So the σ=1 bins did track real shape, but
  noisily and confounded.

**FINAL DECISION: σ = 2 px.** Chosen as the *smallest* value past the knee in the sweep — i.e.
the least smoothing that removes the pixelation artefact. Deliberately NOT σ=3 or 4, which
would additionally suppress the genuine fine-scale boundary detail this study exists to measure
(master §6 evaluates Boundary F1 at 2 px and 5 px tolerance).

**MEASURED OUTCOME after regenerating all 267 images at σ=2:**

| quantity | σ=1 (old) | **σ=2 (applied)** |
|---|---|---|
| mean 90th-pct κ | 0.2654 | **0.1834** |
| implied radius of curvature | 3.8 px *(impossible: < any real cell)* | **5.5 px** *(plausible: sharpest 10% of boundary points on cells of radius 9.9–15.2 px)* |
| CV, pooled (n=267) | 4.99%\* | **6.91%** |
| CV, tcell / flowchamber | — | **5.72% / 5.87%** |
| **Spearman κ vs cell_size** | **−0.54 tcell / −0.33 flow** | **+0.09 tcell / −0.05 flow** |
| Spearman κ vs median cell size | −0.58 / −0.41 | +0.02 / −0.16 |
| bins (low/mid/high) | 89/89/89 | 89/89/89 |
| distinct scores | 263 | 264 |

\* the sweep figures were computed on a 24-image subsample; the CV column above for σ=2 is the
full-corpus value. See the CV note in the verification report — the σ=2 CV is **6.91%**, not the
~12.78% one might expect from the sweep table, because 12.78% was the σ=**4** row.

- **The override is validated by the confound collapsing.** The cell-size correlation that made
  the "high-curvature bin" substantially a "small-cell bin" at σ=1 (−0.54) drops to ≈0 at σ=2
  (+0.09). This is strong evidence that the σ=1 correlation was itself an artefact of pixelation
  — smaller cells have proportionally more staircase per unit boundary, inflating their apparent
  curvature — rather than a real geometric relationship. Master §6's partial-correlation control
  for cell size remains in place regardless (see docs/theoretical_motivation.md), but it is now a
  belt-and-braces check rather than load-bearing.
- Degeneracy + spot check re-run against the NEW σ=2 bins: all checks PASS; the 10 high-bin
  overlays show pointed/teardrop protrusions and touching-cell junctions.
- Residual note: the test split's bin occupancy is now low=50 / mid=12 / high=36 (train
  39/77/53). The **high** bin — the one master §4's success criteria are stated on — has n=36,
  which is workable; the mid bin at n=12 has limited power for any mid-bin-specific test. Bins
  are tertiles over the whole corpus and the split was fixed earlier (A11/A17), so occupancy is
  not tuneable without breaking one of those.

### A28 — Training batch size = 4 at 1024×1024 on a 16 GB Kaggle T4
- Status: OPEN — supersedes master §10's assumed batch of 8 for the chosen resolution (A19).
- Missing info: the paper's batch size (A05, paywalled).
- Why master §10's batch 8 cannot stand: A19 fixed the input at 1024×1024, four times the pixel
  count of a 512×512 input. For the frozen U-Net (master §7, encoder [64,128,256,512,1024]) the
  stored forward activations needed for backprop are dominated by the first level: at 1024²,
  64 channels × 1024² × 4 B ≈ 268 MB **per tensor per sample**, and the encoder/decoder stack
  roughly doubles that. That puts a single sample near ~2 GB of activations, so batch 8 ≈ 16 GB
  before parameters, gradients and Adam state — it will OOM on a 16 GB T4 or P100.
- Alternatives considered: (a) drop the resolution — rejected, A19 shows 512 collapses master
  §6's 5 px Boundary-F1 tolerance onto the whole cell radius; (b) mixed precision — rejected for
  final runs by master §10; (c) gradient checkpointing — adds complexity and slows training,
  held in reserve; (d) random-crop training at native scale — viable, retained as a fallback.
- Final decision: **batch_size = 4**, with gradient accumulation available if an effective
  larger batch is wanted. To be confirmed against real GPU memory on the first Kaggle run and
  reduced to 2 if it OOMs.
- Scientific justification: batch size is held **identical across Vanilla/A/B/B′/C** (master
  §10), so it cannot confound the B-vs-C comparison the study rests on; only wall-clock time and
  BatchNorm statistics are affected. BatchNorm at batch 4 is noisier than at 8, which is worth
  noting because master §9 requires a BatchNorm recalibration pass after pruning — that
  recalibration uses frozen forward passes and is therefore unaffected.
- Expected reproducibility impact: LOW directionally, MEDIUM on absolute values. **Verify on the
  first Kaggle run before committing to all 24 runs.**

### A06-ADDENDUM — the curvature-proxy choice was validated by MEASUREMENT, not by the paper
- Status: OPEN — records the *epistemic status* of the A06 surrogate. Added 2026-08-06.
- **This must be stated plainly in the paper's methods section, not just here.**
- What happened: Phase 5 chose the level-set formulation `div(∇p/|∇p|)` over the literal
  Sobel-gradient magnitude `|∇p|` that master §8 names, and chose σ=2 px smoothing over the
  unsmoothed alternative, on the basis of numerical experiments on synthetic shapes — a corner
  scored 0.3166 versus a straight edge's 0.3333 under the naive choice, i.e. the proxy failed to
  discriminate corners at all, which is the one property dCR requires of it.
- **What that does and does not establish.** It establishes that the implemented proxy is
  *numerically fit for purpose*: it ranks high-curvature boundary points above smooth ones, which
  a literal `|∇p|` demonstrably does not. It does **NOT** establish that it matches the dCR
  paper's actual equation. The paper remains closed-access with no preprint and no code
  (docs/source_review.md §§1–2), so the true "curvature embedding space" formulation is still
  unrecovered and unverified.
- Consequence for how results may be described: every result produced with this loss is a result
  for **"a curvature-weighted boundary loss in the spirit of dCR, with a numerically validated
  curvature proxy"** — NOT a reproduction of Asha & Gopakumar (2026). Any phrasing that implies
  the latter would misrepresent the work. Phase 8 must carry this caveat into every table and
  claim that depends on Models A and C.
- What would change this: the professor supplying the equation or code. The proxy is isolated
  behind a single swappable function precisely so that replacement is a one-function change
  (master Golden Rule 8), after which this entry is re-labelled and every affected phase re-run.

### A30 — Kaggle must read hyperparameters from configs/, never restate them [ENGINEERING]
- Status: **RESOLVED / ENFORCED 2026-08-06**
- Why this needed enforcing: `curvature_smooth_sigma` became a config value in Phase 5. If a
  Kaggle notebook cell restated it, that cell could drift from the repo and silently reintroduce
  the exact pixelation bug Phase 5 fixed — and the drift would be invisible, because both runs
  would "succeed".
- Enforcement, in layers rather than by convention:
  1. `experiments/multi_seed_train.py` is the single entry point; it composes every value from
     `configs/` via `src.config.compose_run_config`. The Kaggle notebook calls it and passes
     only a config path, a seed index and a device.
  2. `validate_config` REJECTS a dCR run whose `curvature_smooth_sigma` is missing or ≤ 0, so a
     drifted or defaulted value fails before training starts rather than after.
  3. The runner echoes the resolved sigma at startup, marked as coming from the YAML.
  4. `curvature_smooth_sigma`, `loss_band_width`, `loss_beta`, `curvature_proxy` and a SHA-256
     `config_fingerprint` are written into every experiment-log row, so a drift is detectable
     after the fact even if it somehow got past the front-end checks.
- Verified: `audit/verify_phase6.py` confirms both `model_a_dcr.yaml` and
  `model_c_pruned_dcr.yaml` declare σ=2.0, that the composed config carries it through, that
  σ=0 and a missing σ are both rejected, and that the runner prints the YAML-resolved value.

### A31 — Checkpoint/early-stopping monitor is `val_boundary_f1_2px` (which tolerance) [ASSUMPTION]
- Status: OPEN
- Missing info: master §10 says early stopping monitors "val Boundary F1", but master §6 defines
  **two** Boundary-F1 tolerances (2 px and 5 px). Which one drives selection is unspecified.
- Why it surfaced: the Phase 6 smoke test caught the configs monitoring a bare
  `val_boundary_f1`, a key the validator never emits. The lookup failed, the monitored value
  degraded to NaN, the tracker never improved and **no best checkpoint was ever written** —
  while the run still reported success. Left unfixed this would have produced 24 Kaggle runs
  with final-only checkpoints and no selectable best.
- Final decision: monitor **`val_boundary_f1_2px`**. The 2 px tolerance is the stricter of the
  two and therefore the more sensitive to exactly the fine boundary degradation P1/P2 predict;
  selecting on the looser 5 px would tolerate more of the damage under study. Both tolerances
  are still computed and reported (master §6) — this choice affects checkpoint selection only.
- Hardening: `monitored_value` now raises a KeyError naming the available keys instead of
  defaulting, and the Trainer deliberately does NOT catch it. A monitor typo now fails on epoch
  0 rather than silently wasting a full training run.
- Expected reproducibility impact: LOW — identical across all five models, so it cannot
  confound the B/B′/C comparison; it only determines which epoch's weights are kept.

### A32 — Train/val Boundary-F1 gap monitor: subsample size and evaluation design [ASSUMPTION]
- Status: **RESOLVED / APPLIED 2026-08-06**, on direct operator instruction for Phase 6-RUN:
  "Training set is small (~170 images) -- monitor train/val Boundary F1 gap every epoch; flag
  in run notes if it exceeds 0.15 at convergence."
- Missing info: no paper-derived guidance exists on overfitting monitoring for this study; the
  instruction specifies *what* to watch and the flag threshold, not *how* to compute a train-side
  metric cheaply every epoch.
- Design: `build_dataloaders` (src/datasets/dataset.py) additionally returns a `train_eval_loader`
  when `training.track_train_val_gap` is enabled — a FIXED, unaugmented, deterministic subsample
  of the training set (`sort_values("image_id").head(N)`), evaluated with the same
  `compute_all_metrics` used for validation. Deterministic and stable across epochs so the trend
  reflects the model changing, not the subsample changing.
- `train_metrics_subsample_size = 40` [ASSUMPTION]: large enough to average out per-image noise
  in the Boundary F1 estimate, small enough that adding a second metrics pass every epoch does
  not meaningfully slow training (bounded, independent of the ~170-image training set size).
- Alternatives considered: (a) evaluate on the FULL training set every epoch — rejected, doubles
  per-epoch cost for a monitoring signal, not a result that needs full-corpus precision;
  (b) re-sample the subset each epoch — rejected, would make the gap trend noisy for reasons
  unrelated to overfitting, confounding "gap is widening" with "subsample changed".
- Scientific justification: the gap is a DIAGNOSTIC, not one of the three frozen metrics
  (master §6) and not part of any confirmatory test — it exists solely to catch overfitting on
  a small corpus while it is happening, not to feed P1–P6.
- Expected reproducibility impact: NONE on the study's conclusions (diagnostic only); HIGH
  operationally — a flagged run signals its own results should be inspected before being trusted.

### A33 — Overfit-gap flagging: threshold, "at convergence" semantics, and a bug this design caught
- Status: **RESOLVED / APPLIED 2026-08-06**, threshold given verbatim by the operator instruction
  quoted in A32 ("exceeds 0.15 at convergence").
- "At convergence" is read as **the last epoch actually trained** — whichever epoch ends
  training, whether by early-stopping on plateau or by exhausting `max_epochs` — not necessarily
  the epoch that produced the best checkpoint (tracked separately as
  `train_val_gap_at_best_epoch`, reported alongside for context). This is the literal reading of
  the instruction's wording and is also the more useful one operationally: it reflects the
  model's state at the point training actually stopped.
- `overfit_gap_threshold = 0.15` [ENGINEERING, given verbatim] applies identically to every model
  (Vanilla/A/B/B′/C) via `training_shared.yaml`, so it cannot itself become a source of
  between-model difference.
- Gap sign convention: `gap = train_boundary_f1_2px − val_boundary_f1_2px` (positive = training
  outperforming validation, the classic overfitting signature). When flagged, `notes` records the
  exact gap value, the threshold, the epoch, and the training-set size, e.g.: *"WARNING:
  train/val boundary_f1_2px gap 0.8500 exceeds 0.15 at convergence (epoch 0, n_train=2) --
  possible overfitting on the small training set."* Unflagged runs leave `notes` empty, and a run
  with tracking disabled reports `""` (not `NaN`) in the gap columns, so an empty experiment-log
  cell is unambiguous.
- **A real bug this design caught, unrelated to the monitoring feature itself but found while
  building it:** `src.training.validator.validate(..., collect_rows=False)` — the mode
  `evaluate_train_subsample` uses for efficiency — only ever appended per-image metric dicts to
  `rows` INSIDE the `if collect_rows:` branch, so `summarise(rows)` received an empty list and
  silently returned `{}` for the ENTIRE summary, not merely omitting per-image detail. Every gap
  would have computed as `NaN` (a value that never exceeds any threshold), i.e. the overfit
  monitor would have run, logged a `train_val_gap` column, and never once fired — a second
  instance of the exact silent-degrade failure class A31 already found (a lookup that quietly
  produces nothing rather than erring). Fixed by always accumulating metrics into a
  `metrics_only` list regardless of `collect_rows`, and feeding THAT into `summarise()`;
  `collect_rows` now controls only whether the returned per-image `rows` (image_id/meta) are
  built, never whether metrics are computed. `audit/verify_phase6.py` gained a direct regression
  check: `collect_rows=True` and `collect_rows=False` must agree on the summary.
- Expected reproducibility impact: the bug fix has no effect on any of the three frozen metrics
  or on model weights (validate() is read-only); it only restores a monitoring signal that would
  otherwise have failed silently for all 24 Kaggle runs.

### A34 — Kaggle environment: do not reinstall torch/torchvision over the platform build
- Status: **RESOLVED / APPLIED 2026-08-06**, for Phase 6-RUN's notebook.
- Missing info: master §16 pins `torch==2.1.0`/`torchvision==0.16.0`; Kaggle's GPU images ship a
  specific torch build matched to their CUDA driver, which is very unlikely to be exactly 2.1.0.
- Why forcing the pin is the wrong call: `pip install torch==2.1.0` on Kaggle would pull a CPU or
  mismatched-CUDA wheel unless the exact `+cu121`-style local version is also pinned and happens
  to match the image's driver, and getting this wrong silently downgrades training to CPU or
  breaks CUDA entirely, for infrastructure reasons having nothing to do with the model. This is
  precisely the class of risk Phase 6 built portability against: `requirements-kaggle.txt`'s
  header already documents that the code targets a *behaviourally* compatible range
  (no `verbose=` on `ReduceLROnPlateau`, no version-specific `torch.amp` path, explicit
  `weights_only=` on `torch.load`), specifically so it tolerates whatever torch Kaggle actually
  provides.
- Final decision: the Kaggle notebook installs every `requirements-kaggle.txt` package EXCEPT
  `torch` and `torchvision`, which are left as Kaggle's preinstalled, GPU-matched build. The
  notebook prints the resolved `torch.__version__`, `torch.version.cuda`, and
  `torch.cuda.is_available()` at startup and writes them into the run's notes, so the actual
  environment is recorded rather than assumed to match the pin.
- Scientific justification: master §10's invariants (identical optimiser/LR/seeds/determinism
  across B/B′/C) do not depend on the exact torch patch version, only on internal consistency
  across the 24 runs — which running the SAME Kaggle image for all of them already guarantees.
- Expected reproducibility impact: LOW for the study's internal comparisons (all 24 runs share
  one Kaggle image); MEDIUM if exact numerical reproduction on a different torch version is later
  attempted — record the printed version alongside any such attempt.

### A29 — Confirmatory test family reduced 24 → 16 (mid bin descriptive-only) [ENGINEERING]
- Status: **RESOLVED / APPLIED 2026-08-06** (authorised, pre-training). Recorded because it is a
  deviation from master §6's stated "~24 tests".
- What forced it: Phase 2 measured the test-bin occupancy as low=50, **mid=12**, high=36. Bins are
  corpus-wide tertiles (A26) while the split is group-atomic and leakage-driven (A11/A17); the two
  were necessarily determined independently, so bin membership and split membership were never
  jointly optimised. The mid bin's n=12 is therefore underpowered **by construction** and cannot be
  fixed without breaking the leakage guarantee — which is not on the table.
- Measured power at n=12 (`docs/power_analysis.py`): 0.337 at d=0.5 and 0.689 at d=0.8 (α=0.05);
  0.056 at d=0.5 under the Holm worst-case α. Its 80%-power MDE is d ≥ 0.914, i.e. it can only
  detect effects larger than "large". Detecting a medium effect needs n ≥ 35.
- Final decision: mid-bin Wilcoxon results are **descriptive-only** — computed and reported with
  power stated inline, but never used to support or refute P1–P6, and **excluded from the
  Holm–Bonferroni confirmatory family**. Family size recounted:
  - original: 2 comparisons × 3 bins × 2 ratios × 2 boundary metrics = **24** (reproduces master §6)
  - corrected: 2 comparisons × **2 bins** × 2 ratios × 2 metrics = **16**
- Scientific justification: Holm controls the family-wise error rate across tests that carry
  confirmatory claims. Including tests that are declared incapable of supporting or refuting a
  hypothesis would dilute correction strength on the tests that matter, without protecting any
  real claim — so excluding them is the correct application of the procedure, not a relaxation of
  it. The exclusion is declared **before** any model is trained or any p-value exists, which is
  what distinguishes it from post-hoc family trimming. P2/P4/P5 were always framed on the
  high-curvature bin, so no hypothesis is narrowed retroactively.
- Effect of the correction: most stringent threshold α/24 = 0.002083 → α/16 = 0.003125 (1.50×);
  power at d=0.5 in the high bin 0.362 → 0.415; 80%-power MDE there d ≥ 0.720 → d ≥ 0.693.
- Expected reproducibility impact: LOW on estimates, MEDIUM on inference — it changes which
  comparisons are corrected and must be stated explicitly in the paper's statistics section.
- Recorded in: `docs/success_criteria.md` ("Statistical scope" + "Confirmatory test family").

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
14. ~~Curvature smoothing σ (A27)~~ — **RESOLVED 2026-08-06**: overridden to σ=2 px and all
    bins regenerated. Retained here only as a record that the change happened before any
    training. See A27 for the measured before/after.
15. **Batch size at 1024×1024 (A28).** Provisionally set to 4; confirm against real GPU memory
    on the first Kaggle run before committing to all 24 runs.

**No entry in this file may be marked CONFIRMED without a specific source (PDF page, professor message,
official repo commit) recorded in the Status line.**
