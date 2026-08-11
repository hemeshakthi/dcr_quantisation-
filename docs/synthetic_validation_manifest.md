# Synthetic Validation Manifest

**Purpose.** This is the single reference document for every file anywhere in this repository
that resulted from SYNTHETIC validation (Phases 7, 8, 9, 12 — author-only phases run before any
real Kaggle checkpoint existed). It exists so that nobody — a human, or a different AI agent
(Antigravity) picking this project up later — can mistake a synthetic-mechanics demonstration
for a real experimental result. Produced by the dedicated safety audit run 2026-08-10; see
`docs/implementation_assumptions.md` for the phase-by-phase engineering log this summarizes.

**The hard rule** (standing instruction, applied throughout Phases 7–12): every synthetic output
file either (a) has `SYNTHETIC` in its filename, or (b) lives under a path containing
`synthetic_validation`. Every file below satisfies at least one — most satisfy both.

**As of this audit:**
- `checkpoints/` — **genuinely empty** (0 files). No real training has happened.
- `results/tables/experiment_log.csv` — **header row only** (0 data rows, real or synthetic).
- `data/curvature/curvature_bins.csv` — sha256 `e7c09383fe7b7c4aad05eb4dbcf420ee87743970f6463cade3d8ca797242a242`,
  byte-identical to its Phase 2 commit (`git diff` against `HEAD` is empty).
- The 6 real training configs (`configs/model_*.yaml`, `configs/training_shared.yaml`) are
  byte-identical to their committed state — never modified by any synthetic test.

If any of the four facts above is ever no longer true, this manifest's "nothing real exists yet"
framing is stale and must be re-verified before being trusted.

---

## Phase 7 — Structured Pruning (mechanics validation)

Verifies pruning MECHANICS (channel selection, mask round-trip, measured params/FLOPs
reduction, BatchNorm recalibration determinism) using a FRESH, RANDOMLY-INITIALISED UNet
(quarter-width, real `UNet` class — see A43) as a stand-in for a trained checkpoint. Says
nothing about real model quality. Reproduce via `python audit/verify_phase7.py`.

| Path | What it demonstrates | Real? |
|---|---|---|
| `results/synthetic_validation/phase7/SYNTHETIC_standin_model_a/best.pt` | An UNTRAINED, randomly-initialised UNet checkpoint, used only as a fixture to prune | **NOT real** — random weights, never trained |
| `results/synthetic_validation/phase7/SYNTHETIC_phase7_channel_mask_ratio_0.5.json` | A kept-channel mask computed by pruning the stand-in at ratio 0.5 | **NOT real** — pruned from untrained weights |
| `results/synthetic_validation/phase7/SYNTHETIC_phase7_channel_mask_ratio_0.7.json` | Same, at ratio 0.7 | **NOT real** |
| `results/synthetic_validation/phase7/SYNTHETIC_phase7_record_ratio_0.5.json` | Measured params/FLOPs before/after pruning at ratio 0.5 (real thop measurement, on fake weights) | Measurement mechanism is real; the model measured is **NOT real** |
| `results/synthetic_validation/phase7/SYNTHETIC_phase7_record_ratio_0.7.json` | Same, at ratio 0.7 | Same caveat |
| `results/synthetic_validation/phase7/SYNTHETIC_phase7_validation_report.json` | Pass/fail summary of all Phase 7 mechanics checks | Report of a synthetic test run |

**Ephemeral, not persisted:** `audit/verify_backend_integration.py` also prunes/recalibrates/
fine-tunes a tiny 5-model fixture (real training, real backprop, real pruning, on tiny real
image crops) to prove the backend end-to-end — but it builds everything inside a
`tempfile.mkdtemp()` directory **outside this repository** and deletes it (`shutil.rmtree`) in
a `finally` block before the script exits. Nothing from that test ever appears anywhere in this
repo, including `checkpoints/channel_masks/` — confirmed by this audit's Check 4 (repo-wide
search for the real mask filename pattern found zero matches).

---

## Phase 8 — Evaluation + Statistics (metric/stats correctness validation)

Verifies Dice/Boundary-F1/HD95 and every statistical test (Wilcoxon, t-test, TOST,
Holm-Bonferroni, bootstrap CI, partial correlation) against synthetic mask pairs and synthetic
per-image data with KNOWN answers by construction (a perfect match, a disjoint case, a
constructed corner-notch case). Reproduce via `python audit/verify_phase8.py`.

| Path | What it demonstrates | Real? |
|---|---|---|
| `results/synthetic_validation/phase8/SYNTHETIC_phase8_validation_report.json` | Pass/fail summary; includes the 3 synthetic mask cases' computed metric values | Metric CODE is real; the 3 mask cases are **engineered, not real predictions** |

No image files are persisted for Phase 8 (its mask cases are generated and checked in-memory;
only the summary report is written). All of its statistical-test sanity checks (paired Wilcoxon,
TOST, bootstrap CI, partial correlation) run against seeded synthetic arrays constructed
in-process — none of those intermediate arrays are written to disk at all.

---

## Phase 9 — Failure Taxonomy (detector correctness validation)

Verifies all 6 failure detectors (`corner_collapse`, `boundary_smoothing`, `cell_merging`,
`cell_splitting`, `thin_structure_loss`, `curvature_flattening`) against synthetic mask pairs
engineered to trigger/not-trigger each one, using thresholds read from the real locked
`configs/evaluation.yaml`. The `cell_merging`/`cell_splitting` scene uses 30 synthetic cells,
sized inside the REAL measured Phase 1/2 object-density range (16–203/image) but the cells
themselves are synthetic circles, not real segmented cells. Reproduce via
`python audit/verify_phase9.py`.

| Path | What it demonstrates | Real? |
|---|---|---|
| `results/synthetic_validation/phase9/SYNTHETIC_phase9_corner_collapse.png` | Sharp square vs. the same square with corners chamfered off | **NOT real** — engineered geometry |
| `results/synthetic_validation/phase9/SYNTHETIC_phase9_boundary_smoothing.png` | Spiky star vs. a smoothed circle of similar area | **NOT real** |
| `results/synthetic_validation/phase9/SYNTHETIC_phase9_curvature_flattening.png` | Same star/circle construction, curvature-percentile framing | **NOT real** |
| `results/synthetic_validation/phase9/SYNTHETIC_phase9_cell_merging.png` | 30-cell synthetic scene, one engineered touching/merged pair | **NOT real** — synthetic circles |
| `results/synthetic_validation/phase9/SYNTHETIC_phase9_cell_splitting.png` | Same 30-cell scene, one GT cell engineered to fragment into 2 predicted blobs | **NOT real** |
| `results/synthetic_validation/phase9/SYNTHETIC_phase9_thin_structure_loss.png` | A cell body with a thin spike, present vs. removed | **NOT real** |
| `results/synthetic_validation/phase9/SYNTHETIC_phase9_validation_report.json` | Pass/fail summary + the `p6_ratio` worked example (hand-computed synthetic counts) | Report of a synthetic test run; `p6_ratio`'s example B/C failure counts are **fabricated for the arithmetic check only**, never presented as a real P6 result |

---

## Phase 12 — Final Figures (rendering-function demonstration)

Demonstrates all 3 new figure functions. **Figure 1's SELECTION rule runs for real** against the
real `data/curvature/curvature_bins.csv` (real image_ids, real curvature scores) — that part is
not synthetic. What IS synthetic, because no trained checkpoint exists, are the rendered
"model outputs": a real selected image's real ground-truth mask, plus two masks built by
morphologically opening+closing that SAME real GT mask at two different kernel radii (large
radius = simulated compression damage, small radius = simulated recovery) — see A46. Figure 3's
per-image metric VALUES are a seeded synthetic trend (real curvature scores, fabricated
Boundary-F1 values following a disclosed formula) since no real per-image evaluation exists yet.
Reproduce via `python audit/verify_phase12.py`.

| Path | What it demonstrates | Real? |
|---|---|---|
| `results/synthetic_validation/phase12/SYNTHETIC_phase12_figure1_selection.json` | The stratified-random + median panel selection, run against real curvature data | **Selection is real** (real image_ids chosen by a real, reproducible rule) |
| `results/synthetic_validation/phase12/SYNTHETIC_phase12_figure1_contour_crop_panel.png` | A rendered crop panel: real image + real GT, with 2 synthetic "model" contours | Base image/GT **real**; "B"/"C" contours **NOT real predictions** |
| `results/synthetic_validation/phase12/SYNTHETIC_phase12_figure2_boundary_band_error_B.png` | Boundary-band error map for the synthetic "damaged" (B) prediction | **NOT real** |
| `results/synthetic_validation/phase12/SYNTHETIC_phase12_figure2_boundary_band_error_C.png` | Boundary-band error map for the synthetic "recovered" (C) prediction | **NOT real** |
| `results/synthetic_validation/phase12/SYNTHETIC_phase12_figure3_input_data.csv` | Real curvature scores + a seeded SYNTHETIC Boundary-F1 trend per model | Curvature column **real**; `boundary_f1_2px` column **fabricated** (disclosed formula, seed=0) |
| `results/synthetic_validation/phase12/SYNTHETIC_phase12_figure3_curvature_percentile_curve.png` | The rendered curve from the synthetic data above | **NOT real** |
| `results/synthetic_validation/phase12/SYNTHETIC_phase12_validation_report.json` | Pass/fail summary of all Phase 12 checks | Report of a synthetic demonstration run |

---

## Other places checked and found clean (no action needed)

- **`docs/api_contract.md`** — contained one unlabeled illustrative JSON example
  (`"dice": 0.8123, ...`) showing the `/api/infer` response SHAPE, written before any model was
  trained. Fixed by this audit: an explicit "ILLUSTRATIVE PLACEHOLDER ... not a real measured
  result" label was added immediately before that code block.
- **`README.md`** — stale (still says "Phase 0 complete") but contains zero numeric results;
  out of this audit's scope (staleness is a documentation-freshness issue, not a
  synthetic-vs-real mislabeling risk).
- **Power-analysis tables** in `docs/success_criteria.md` and `docs/implementation_assumptions.md`
  (A29) — prospective statistical power/MDE calculations at standard textbook Cohen's-d
  benchmarks (0.5, 0.8), computed from the REAL measured Phase 2 test-bin sample sizes. These
  are pre-registration methodology, not experimental results, and are already unambiguously
  framed as such (section header: "Measured power at the actual n"). Not renamed or flagged.

---

## How to keep this manifest trustworthy

Whenever a future phase (Kaggle training, real evaluation, etc.) produces the first REAL
artifact in one of the four "as of this audit" locations listed at the top, **re-run this
audit's checks** (or at minimum re-verify `checkpoints/` is no longer expected to be empty,
`experiment_log.csv` now legitimately has real rows, etc.) before trusting this document's
"nothing real exists yet" framing. This manifest describes a snapshot, not an invariant.
