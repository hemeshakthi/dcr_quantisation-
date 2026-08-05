# PROJECT MASTER CONTEXT — dCR Compression Study (v2, source-labeled)

> **READ THIS ENTIRE FILE BEFORE DOING ANYTHING.** Single source of truth. Every decision below carries a source label: **[PAPER]** reproduced from the dCR paper, **[OFFICIAL]** from author repo/supplementary, **[LITERATURE]** supported by recent peer-reviewed work, **[ENGINEERING]** best practice for reproducibility, **[ASSUMPTION]** could not be recovered from any authoritative source, chosen as closest defensible approximation. Every [ASSUMPTION] must also be logged in `docs/implementation_assumptions.md`. Never silently assume anything.

---

## 0. AUTHORITATIVE REFERENCE POLICY (obey before every decision)

The primary reference is: **Asha, S.B. & Gopakumar, G. (2026). "Discriminative Curvature Regularization Loss for Boundary Segmentation in Microscopy Cell Images." SN Computer Science 7(1):64. DOI: 10.1007/s42979-025-04633-7.**

Before designing/implementing anything, search in this priority order: (1) published paper, (2) supplementary, (3) official author repo, (4) official implementation, (5) project page, (6) author GitHub, (7) institutional repo, (8) dataset links cited by the paper, (9) references cited inside the paper, (10) corrigendum/updates.

Whenever the paper specifies something, follow it EXACTLY. Do not replace the paper's methodology with a modern alternative. Never invent equations, preprocessing, datasets, or hyperparameters. If a detail cannot be recovered, search 2024–2026 literature (MICCAI, Medical Image Analysis, IEEE TMI, Nature BME, CVPR medical), pick the closest reproducible choice from high-quality work, explain why, and log it as [ASSUMPTION] in `docs/implementation_assumptions.md` with: missing info, sources searched, why unrecoverable, alternatives considered, final decision, scientific justification, expected reproducibility impact.

**STOP CONDITION:** If a decision would materially change the scientific methodology of the dCR paper, do not guess — re-search all authoritative sources, compare multiple peer-reviewed implementations, pick the closest reproducible approach, and document as [ASSUMPTION]. Never redesign the original research without explicit evidence.

---

## 0a. WHAT WAS RECOVERABLE FROM AUTHORITATIVE SOURCES (source review done)

**Recovered from the paper's public record (abstract, metadata, reference list):** [PAPER]
- Authors: S.B. Asha (lead), G. Gopakumar (supervisor, corresponding, gopakumarg@am.amrita.edu). Amrita School of Computing, Amritapuri. Received Oct 2024, accepted Dec 2025, published 07 Jan 2026. Vol 7, art 64.
- Contribution: curvature-aware loss formulating curvature-based regularization in a "curvature embedding space." +9.18% Dice over Jaccard, +6.21% over existing boundary-aware losses. Focus: touching cells, low-contrast boundaries, small training sets. "Minimal computational overhead."
- Metrics: Dice explicitly named; Jaccard used as a comparison loss.
- Statistics lineage: Student's t-test (ref 40), normal distribution (ref 41), Cohen's power analysis (ref 42).
- Architecture lineage: U-Net for cell counting/morphometry (Falk 2019, ref 36); UNet++ (ref 43) and DeepLab encoder-decoder (ref 44) as comparison architectures. U-Net-family encoder-decoder.
- dCR lineage: OCT curvature loss (Xing 2022, ref 29) and bending loss (Wang 2020, ref 30) are direct ancestors; total curvature (Goldluecke 2011, ref 27), morphological curvature evolution (Marquez-Neila 2013, ref 33), contour-curvature perception (Foster 1993, ref 35) underpin "perceptual curvature."
- Boundary-loss context: Kervadec boundary loss (ref 15), Active Boundary Loss (ref 17), Conditional Boundary Loss (ref 18), Boundary DoU (ref 19), InverseForm (ref 10).
- Data lineage: authors' prior bright-field cell/blood segmentation (refs 5, 6, 9); Zenodo StarDist/TrackMate datasets flow-chamber (ref 37, Zenodo 4034939) and T-cell (ref 38, Zenodo 4034929); DIC reconstruction (ref 39). These are the only datasets named in the PUBLIC reference list.
- Data availability: "publicly available and the repository links are given in the manuscript" — exact identities live in the paywalled body.

**NOT recoverable from public sources (paper body paywalled) — treat as [ASSUMPTION] until PDF or professor confirms:**
exact datasets + splits, image resolution, normalization, augmentation, optimizer, LR, scheduler, weight decay, epochs, batch size, initialization, stopping criteria, the exact dCR equation, post-processing.
**ACTION:** obtain the PDF / ask the professor for all of the above BEFORE Phase 2 finalizes. Every value used before confirmation is an [ASSUMPTION] logged in docs/implementation_assumptions.md.

---

## 0b. WHAT THIS PROJECT IS

Publication-quality medical AI research (MICCAI / IEEE TMI / Medical Image Analysis). NOT a toy. Every result — especially in the UI — comes from **real trained models**. A previous attempt failed because all models produced identical results. Two identical model outputs = a BUG to find and fix, never acceptable.

**Research question (FROZEN):** Can dCR loss recover the boundary geometry degradation introduced by structured channel pruning in microscopy cell segmentation? [ENGINEERING — professor-set follow-up; extends, does not alter, the paper]

**Four models (FROZEN):**
| Model | Definition | Role |
|---|---|---|
| A | Uncompressed U-Net + Dice + dCR | Reference |
| B | Pruned U-Net + Dice | Compressed baseline (damage) |
| B′ | Pruned U-Net + Dice + Generic Boundary Loss | Control (isolates curvature-specificity) |
| C | Pruned U-Net + Dice + dCR | Proposed (recovery) |

Plus a Vanilla U-Net (Dice only, uncompressed) baseline. Split 70/30 [ENGINEERING — professor-set]. Datasets: same as the paper [PAPER — identities to confirm from body].

---

## 1. OUT OF SCOPE (never introduce)
NO transformers, attention U-Net, UNet++ (plain U-Net only, §7). NO quantization, knowledge distillation, deployment research (no TensorRT/Jetson; ONNX only optional stub). NO extra hypotheses/research models. NO persistent homology, Betti loss, cross-stain, energy metrics, compression-ordering. NO placeholders/fake graphs/hardcoded metrics/mock backend/fabricated UI.

---

## 2. THEORETICAL MECHANISM (for the paper) [ENGINEERING, grounded in [LITERATURE]]
Structured pruning reduces skip-connection bandwidth — the pathways carrying high-frequency boundary detail from encoder to decoder. L1-norm pruning can't protect boundary-critical channels, so pruning damages high-curvature boundary geometry while sparing Dice. dCR concentrates gradient on boundary-band pixels weighted by curvature, recovering that damage. Complementary to MedCore (Zhang 2026, arXiv:2605.13688) which changes the pruning criterion; we change the fine-tuning loss.

Six predictions: **P1** pruning preserves Dice but degrades Boundary F1/HD95; **P2** degradation larger in high-curvature bin; **P3** C>B on boundary metrics; **P4** recovery concentrated in high-curvature bin; **P5** C>B′ in high-curvature bin (curvature-specific); **P6** dCR reduces curvature-related failures more than non-curvature ones.

---

## 3. RELATED WORK / LINEAGE [PAPER references]
- **MedCore (Zhang 2026)** — closest; pruning-criterion change on MedSAM/ViT; ours complementary (fine-tuning loss on U-Net). [LITERATURE]
- **dCR ancestors** — bending loss (Wang 2020, ref 30), OCT curvature loss (Xing 2022, ref 29); dCR differs via a curvature embedding space; neither tested under compression. [PAPER refs]
- **B′ generic boundary loss** — pick a simple established boundary-band loss from the paper's cited family (e.g. Kervadec ref 15). [PAPER refs]

---

## 4. SUCCESS CRITERIA (locked before training) [ENGINEERING]
High-curvature bin, ratio 0.5, mean over 3 seeds: Boundary F1 C−B ≥ 0.04; HD95 C vs B ≥ 10% relative reduction; Dice B vs C equivalent via TOST within ±0.02; visual ≥6/10 restoration. Outcome tree: full positive / partial (B′≈C) / null (C≯B) / confounded (gate fails → diagnose, don't auto-retry). Never fabricate positive; report null honestly.

---

## 5. DATASETS [PAPER identity to confirm]
Paper body (paywalled) has exact datasets + links. Only StarDist/TrackMate Zenodo sets (refs 37/38) are named in the PUBLIC reference list — HIGHEST-confidence candidates. Other candidates consistent with the abstract + authors' prior bright-field cell work:
| Candidate | Source | License | Status |
|---|---|---|---|
| StarDist/TrackMate flow-chamber | Zenodo 4034939 (ref 37) | CC-BY | [PAPER ref — strong] |
| StarDist/TrackMate T-cell | Zenodo 4034929 (ref 38) | CC-BY | [PAPER ref — strong] |
| BCCD | github.com/Shenggan/BCCD_Dataset | MIT | [ASSUMPTION — matches blood-cell lineage ref 6] |
| Cellpose cyto | cellpose.org/dataset | non-commercial | [ASSUMPTION] |
| DeepBacs E. coli | Zenodo | CC-BY 4.0 | [ASSUMPTION] |
| HeLa bright-field | confirm repo | dataset-dependent | [ASSUMPTION] |

**ACTION before Phase 1 finalizes (mandatory):** obtain PDF / ask professor for (a) exact datasets + links, (b) exact dCR equation/code, (c) preprocessing (resolution, normalization, augmentation), (d) optimizer/LR/scheduler/epochs/batch/init/stopping. Everything before confirmation = [ASSUMPTION].
**BCCD caveat** [LITERATURE]: BCCD ships as bounding boxes, not masks — do not fabricate masks from boxes; confirm mask source with professor.

---

## 6. METRICS
- **Dice** [PAPER] — foil; TOST equivalence (Dice only, ±0.02).
- **Boundary F1** [LITERATURE] — 2px + 5px tolerance, relative to mean cell diameter.
- **HD95** [LITERATURE] — 95th-percentile Hausdorff, standardized morphological-gradient extraction, integer coords.
No clDice/Betti/IoU-as-separate.
**Statistics:** paper uses t-test + Cohen power analysis [PAPER]. We ADD [ENGINEERING]: per-image storage; paired Wilcoxon (B vs C, B′ vs C) per bin/metric/ratio; Holm-Bonferroni across ~24 tests; TOST on Dice; rank-biserial effect sizes (Cohen scale); bootstrap 95% CIs; partial correlation (curvature vs boundary error controlling cell size + contrast); power analysis at actual n (matches paper's Cohen power analysis ref 42). Keep the paper's t-test as a reported secondary; Wilcoxon primary (small non-normal medical samples).

---

## 7. ARCHITECTURE (FROZEN) — single fixed U-Net
Family = U-Net (Falk 2019 ref 36; UNet++ ref 43 / DeepLab ref 44 are baselines the paper compares to) [PAPER-family]. Plain U-Net: encoder [64,128,256,512,1024], bottleneck 1024, decoder [512,256,128,64], final Conv2d(64,1,1) NO activation, input (B,1,H,W), output (B,1,H,W) logits. [Channel widths [ASSUMPTION] — standard U-Net; confirm if body specifies.]
**Why plain U-Net (state in paper) [ENGINEERING]:** skip connections are the mechanism under study; Attention U-Net's learned weighting adapts to pruning (confound); UNet++'s dense skips give redundant boundary pathways, artificially protecting B and shrinking the B-C gap (confound). Single fixed backbone → architecture never a variable.

---

## 8. LOSSES (FROZEN)
- **DiceLoss** [PAPER]: soft Dice, smooth=1e-5.
- **DCRLoss** [PAPER formulation to confirm]: curvature regularization in a curvature embedding space. Until the exact equation/code is confirmed, implement the closest defensible surrogate — boundary-band BCE weighted by a differentiable curvature proxy (Sobel-gradient magnitude on prediction) — labeled [ASSUMPTION], distinguished from bending loss (Wang 2020) and OCT curvature loss (Xing 2022). Fully differentiable, no numpy in forward. **Replace with the real equation/code the moment the professor provides it.**
- **GenericBoundaryLoss (B′)** [LITERATURE — from the paper's boundary-loss family, e.g. Kervadec ref 15]: IDENTICAL boundary-band extraction to dCR (shared util), FLAT weighting, no curvature term.
- **CombinedLoss** [ENGINEERING]: α·Dice + β·(dcr|generic) + calibrate_beta_by_gradient_norm() so B′'s β matches dCR's gradient-norm contribution at equal α.
dCR and GenericBoundaryLoss MUST share extract_boundary_band() — pixel-identical except weighting.

---

## 9. STRUCTURED PRUNING (FROZEN protocol) [ENGINEERING — the new contribution, not in the paper]
L1-norm structured channel pruning. Ratios 0.5 (primary), 0.7 (robustness). Protocol: train A → prune ONCE per ratio → save per-layer kept-channel index mask (JSON) + hash → B/B′/C load the SAME pruned base, never re-prune → prune BatchNorm with Conv → recalibrate BatchNorm (1–2k frozen forward passes, identical images/order) → fine-tune. Compression accounting uses MEASURED params+FLOPs (thop). B/B′/C at a ratio MUST have identical params/FLOPs.

---

## 10. TRAINING PIPELINE (publication-quality) [ENGINEERING; hyperparams [ASSUMPTION] until confirmed]
best+final checkpointing, resume, plateau early stopping (patience 15 on val Boundary F1) [ASSUMPTION], ReduceLROnPlateau [ASSUMPTION], TensorBoard+CSV logging, experiment-log per run, three separate seeds logged separately, full determinism, BatchNorm recalibration after pruning, mixed-precision OFF for final runs, GPU+Kaggle.
Fixed augmentation seed shared across B/B′/C; only model_init_seed varies across 3 seeds. Identical optimizer/LR across B/B′/C. Log per-epoch gradient norms.
Determinism block: torch/cuda/numpy/random seeds; cudnn.deterministic=True; benchmark=False; use_deterministic_algorithms(True).
**Optimizer/LR/epochs/batch [ASSUMPTION]** — defaults if unrecoverable: Adam, lr 1e-4, wd 1e-5, batch 8, ≤100 epochs + plateau. All logged as [ASSUMPTION].
Total runs: Vanilla(3)+A(3)+[B+B′+C]×2×3 (18) = 24.

---

## 11. CURVATURE STRATIFICATION (locked before training) [ENGINEERING]
Discrete curvature on Gaussian-smoothed (σ=1px) GT contours; per-image score = 90th-percentile point curvature. 3 bins by thirds. Write once to data/curvature/curvature_bins.csv (image_id, curvature_score, bin, cell_size, contrast) with write-once guard. Verify high bin has real corners/junctions (spot-check 10). Store cell size + contrast for partial-correlation confound check.

---

## 12. FAILURE TAXONOMY (6 detectors, thresholds locked before inspection) [ENGINEERING]
corner_collapse, boundary_smoothing, cell_merging, cell_splitting, thin_structure_loss, curvature_flattening (doc if merged). Coded rules, blind to model. Analytical: (curvature-related reduction)/(non-curvature-related reduction) for B→C = P6.

---

## 13. VISUAL GATE (hard STOP) [ENGINEERING]
Single-seed A/B/C → 10 four-column contour figures from pre-selected high-curvature images (seed=42, from GT before any output) → STOP → 2 people confirm ≥6/10, blind to numbers. Fail → diagnose, don't auto-retry.

---

## 14. UI / PROTOTYPE (must be REAL) [ENGINEERING — hard requirement]
Previous prototype failed with identical results for all models. Prevent it.
- LIVE inference on real checkpoints. No placeholders/hardcoded/fake/mock.
- Frontend↔backend connected. Backend loads 5 real checkpoints (Vanilla, A, B, B′, C at 0.5). Live metrics per image.
- Select test image (dropdown labeled by curvature bin; default high-curvature). Show original, GT, and per model: prediction + contour overlay, toggleable error map, Dice/BoundaryF1/HD95/model size/param count/curvature bin.
- Black & white ONLY. Contours as lines (distinct styles, no color). Error maps solid (FP) vs hatched (FN). Metrics as relative bars vs Model A + raw numbers.
- 60-second story: B damaged boundaries, C recovered them, Dice would hide it. If C doesn't beat B, show it honestly.
- Startup self-check: any two byte-identical model predictions on a probe → red "MODEL COLLAPSE DETECTED" banner.

---

## 15. REPO STRUCTURE
```
dcr-compression-study/
├── configs/  ├── data/{raw,processed,splits,curvature}
├── src/{datasets,models,losses,compression,metrics,failure,training,evaluation,visualization}
├── experiments/  ├── backend/  ├── frontend/  ├── docs/  ├── tests/  ├── audit/
├── checkpoints/  ├── results/{per_image_metrics,tables,figures,failure_counts}  ├── logs/
├── requirements.txt  ├── environment.yml  └── README.md
```

---

## 16. requirements.txt (pinned)
```
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3
opencv-python==4.8.1.78
albumentations==1.3.1
scikit-image==0.21.0
scipy==1.11.3
pyyaml==6.0.1
tensorboard==2.14.0
pillow==10.1.0
matplotlib==3.8.0
tqdm==4.66.1
pandas==2.1.1
statsmodels==0.14.0
fastapi==0.109.0
uvicorn==0.27.0
python-multipart==0.0.6
imagehash==4.3.1
thop==0.1.1.post2209072238
pytest==7.4.3
```

---

## 17. CLAUDE CODE vs KAGGLE
Authoring, verification, stats, figures, UI, backend, audit → **Claude Code**. Only the 24-run training + full-test inference → **Kaggle (T4/P100)**. Kaggle notebooks mount repo+processed data, write to /kaggle/working/, download back to checkpoints/ and logs/. ~30–90 min per run.

---

## 18. GOLDEN RULES
1. Real models, real metrics, real inference — never fake/placeholder/hardcoded.
2. Two identical model outputs = BUG.
3. Every UI number traces to a live inference call.
4. Never fabricate a positive; report null honestly.
5. Every phase ends with verification; nothing proceeds until it passes.
6. Nothing out of scope (§1); nothing deviating from FROZEN.
7. Every decision carries a source label; every [ASSUMPTION] logged in docs/implementation_assumptions.md.
8. The paper is authoritative — when its body/code is available, replace every [ASSUMPTION] with [PAPER]/[OFFICIAL] and re-verify.
9. When unsure, re-read this file and re-search authoritative sources.
