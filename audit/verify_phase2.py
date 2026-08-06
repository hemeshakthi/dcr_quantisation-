"""Phase 2 verification: preprocessing, augmentation, curvature stratification.

Recomputes from the artefacts actually on disk. Where a property was asserted by the pipeline,
this script re-derives it INDEPENDENTLY (alignment is re-measured from the saved arrays;
cell size is recomputed from the instance masks; determinism is re-tested from scratch) so a
bug in the pipeline cannot certify itself.
"""
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

failures, warnings = [], []


def check(label, cond, warn_only=False):
    ok = bool(cond)
    print(f"[{'PASS' if ok else ('WARN' if warn_only else 'FAIL')}] {label}")
    if not ok:
        (warnings if warn_only else failures).append(label)
    return ok


print("=" * 78)
print("PHASE 2 VERIFICATION")
print("=" * 78)

# ---- 1. scripts ------------------------------------------------------------------
for f in ["preprocess.py", "augmentations.py", "curvature_stratification.py"]:
    check(f"script exists: src/datasets/{f}", (ROOT / "src" / "datasets" / f).is_file())

# ---- 2. preprocessing manifest + config ------------------------------------------
man_p = ROOT / "data" / "processed" / "preprocess_manifest.csv"
cfg_p = ROOT / "data" / "processed" / "preprocess_config.json"
man = None
if check("preprocess_manifest.csv exists", man_p.is_file()):
    man = pd.read_csv(man_p)
    check(f"manifest covers all 267 split images (got {len(man)})", len(man) == 267)
    check("every processed image file exists",
          all((ROOT / p).is_file() for p in man["proc_image"]))
    check("every binary mask file exists",
          all((ROOT / p).is_file() for p in man["proc_binary"]))
    check("every INSTANCE mask file exists (instance IDs preserved)",
          all((ROOT / p).is_file() for p in man["proc_instances"]))
    check("both native resolutions represented",
          set(map(tuple, man[["native_h", "native_w"]].drop_duplicates().values.tolist()))
          == {(1024, 1024), (1400, 1400)})
    check("all outputs are 1024x1024",
          bool((man["out_h"] == 1024).all() and (man["out_w"] == 1024).all()))
    check("zero instances lost during resize",
          int(man["instances_lost"].sum()) == 0)
    check("shape_match true for every image", bool(man["shape_match"].all()))
    check("binary == instance support for every image",
          bool(man["binary_equals_instance_support"].all()))
    check("alignment OK for every image (post-resize)", bool(man["alignment_ok"].all()))
    check("both splits present in manifest", set(man["split"]) == {"train", "test"})

if check("preprocess_config.json exists", cfg_p.is_file()):
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    check("config records target size 1024", cfg.get("target_size") == 1024)
    check("config records percentile normalisation (not 8-bit mean/std)",
          "percentile" in str(cfg.get("normalisation", {}).get("method", "")).lower())
    check("config records pmin=1 / pmax=99.8",
          cfg["normalisation"]["pmin"] == 1.0 and cfg["normalisation"]["pmax"] == 99.8)
    check("config records INTER_NEAREST for masks",
          cfg["resampling"]["mask"] == "INTER_NEAREST")
    check("config records instance masks retained",
          cfg.get("instance_masks_retained") is True)
    check("config carries source labels", bool(cfg.get("source_labels")))

# ---- 3. INDEPENDENT re-verification on real saved arrays -------------------------
if man is not None and len(man):
    rng = np.random.default_rng(0)
    sample = man.sample(n=min(12, len(man)), random_state=0)
    bad_align, bad_binary, bad_dtype, bad_range = 0, 0, 0, 0
    for _, r in sample.iterrows():
        img = np.load(ROOT / r["proc_image"])
        inst = np.load(ROOT / r["proc_instances"])
        binm = cv2.imread(str(ROOT / r["proc_binary"]), cv2.IMREAD_UNCHANGED)
        if img.dtype != np.float32 or inst.dtype != np.uint16:
            bad_dtype += 1
        if not np.array_equal((binm > 0), (inst > 0)):
            bad_binary += 1
        if img.shape != (1024, 1024) or inst.shape != (1024, 1024):
            bad_range += 1
        # re-measure alignment from scratch
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        np.multiply(gx, gx, out=gx); np.multiply(gy, gy, out=gy)
        np.add(gx, gy, out=gx); np.sqrt(gx, out=gx)
        del gy
        fg = (inst > 0).astype(np.uint8)
        edge = cv2.morphologyEx(fg, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
        aligned = float(gx[edge].mean())
        best = max(float(gx[np.roll(edge, s * d, axis=a)].mean())
                   for d in (4, 12) for a in (0, 1) for s in (1, -1))
        if (aligned - best) / (best + 1e-8) <= -0.05:
            bad_align += 1
        del img, inst, binm, gx, fg, edge
        gc.collect()
    check(f"re-measured alignment holds on {len(sample)} sampled images "
          f"(failures={bad_align})", bad_align == 0)
    check(f"binary mask == instance support, re-derived (failures={bad_binary})",
          bad_binary == 0)
    check(f"dtypes float32/uint16 on disk (failures={bad_dtype})", bad_dtype == 0)
    check(f"shapes 1024x1024 on disk (failures={bad_range})", bad_range == 0)

# ---- 4. augmentation determinism (re-tested from scratch) ------------------------
from src.datasets.augmentations import Augmenter, self_test  # noqa: E402

print("  -- augmentation self-test --")
check("augmentation self-test passes (determinism + invariants)", self_test(verbose=False))

img = np.random.default_rng(1).random((64, 64), dtype=np.float32)
inst = np.zeros((64, 64), np.uint16)
inst[20:40, 20:40] = 7
binary = (inst > 0).astype(np.uint8)
o1 = Augmenter(seed=77)(img, binary, inst, epoch=2, index=3)
o2 = Augmenter(seed=77)(img, binary, inst, epoch=2, index=3)
check("two independent Augmenter instances agree byte-for-byte",
      all(np.array_equal(a, b) for a, b in zip(o1, o2)))
check("augmented instance IDs are a subset of the originals (no blending)",
      set(np.unique(o1[2]).tolist()).issubset({0, 7}))

# ---- 5. curvature stratification -------------------------------------------------
cb_p = ROOT / "data" / "curvature" / "curvature_bins.csv"
ccfg_p = ROOT / "data" / "curvature" / "curvature_config.json"
if check("curvature_bins.csv exists", cb_p.is_file()):
    cb = pd.read_csv(cb_p)
    for col in ["image_id", "curvature_score", "bin", "cell_size", "contrast"]:
        check(f"curvature_bins.csv has master-§11 column '{col}'", col in cb.columns)
    check(f"curvature_bins.csv covers all 267 images (got {len(cb)})", len(cb) == 267)
    check("no null in the master-§11 columns",
          not cb[["image_id", "curvature_score", "bin", "cell_size", "contrast"]]
          .isna().any().any())
    counts = cb["bin"].value_counts().to_dict()
    check(f"all three bins non-empty {counts}",
          all(counts.get(b, 0) > 0 for b in ("low", "mid", "high")))
    check("high-bin median curvature > low-bin median",
          cb[cb["bin"] == "high"]["curvature_score"].median()
          > cb[cb["bin"] == "low"]["curvature_score"].median())
    check("curvature scores not collapsed (>3 distinct values)",
          cb["curvature_score"].nunique() > 3)
    for sp in ("train", "test"):
        present = set(cb[cb["split"] == sp]["bin"])
        check(f"split '{sp}' contains all three bins",
              present >= {"low", "mid", "high"})

    # --- INDEPENDENT re-derivation of cell_size from the INSTANCE masks ---
    sub = cb.sample(n=min(8, len(cb)), random_state=1)
    mismatch, blob_would_differ = 0, 0
    for _, r in sub.iterrows():
        inst = np.load(ROOT / r["proc_instances"])
        ids, counts_ = np.unique(inst[inst > 0], return_counts=True)
        recomputed = float(counts_.mean())
        if abs(recomputed - float(r["cell_size"])) > 1e-3:
            mismatch += 1
        # what a (wrong) binary-blob measurement would have given
        nb, _ = cv2.connectedComponents((inst > 0).astype(np.uint8), connectivity=8)
        blob_mean = float((inst > 0).sum() / max(1, nb - 1))
        if abs(blob_mean - recomputed) > 1e-3:
            blob_would_differ += 1
        del inst
        gc.collect()
    check(f"cell_size re-derived from INSTANCE ids matches stored value "
          f"(mismatches={mismatch})", mismatch == 0)
    check(f"cell_size differs from a binary-blob measurement on {blob_would_differ}/{len(sub)} "
          f"sampled images (confirms instance IDs are genuinely being used, not blobs)",
          blob_would_differ > 0)
    check("instances_per_blob > 1 recorded (touching cells exist)",
          float(cb["instances_per_blob"].median()) > 1.0)

if check("curvature_config.json exists", ccfg_p.is_file()):
    cc = json.loads(ccfg_p.read_text(encoding="utf-8"))
    check("curvature config records sigma", cc.get("contour_sigma_px") is not None)
    check(f"curvature config sigma == 2.0 (post-override), got {cc.get('contour_sigma_px')}",
          float(cc.get("contour_sigma_px", 0)) == 2.0)
    check("curvature config records the override provenance",
          "AUTHORISED" in str(cc.get("contour_sigma_override", "")))
    check("curvature config records 90th-percentile score",
          cc.get("score_percentile") == 90.0)
    check("curvature source mask is the BINARISED mask",
          "BINARIS" in str(cc.get("curvature_source_mask", "")).upper())
    check("cell size source mask is INSTANCE ids",
          "INSTANCE" in str(cc.get("cell_size_source_mask", "")).upper())
    check("no degeneracy problems recorded", not cc.get("degeneracy_problems"))

spot = ROOT / "results" / "figures" / "curvature_spotcheck"
pngs = sorted(spot.glob("high_curv_*.png")) if spot.is_dir() else []
check(f"10 high-curvature spot-check overlays saved (found {len(pngs)})", len(pngs) == 10)
check("spot-check index.tsv exists (traceability)", (spot / "index.tsv").is_file())

# ---- 6. assumptions --------------------------------------------------------------
ass = (ROOT / "docs" / "implementation_assumptions.md").read_text(encoding="utf-8")
for aid in [f"A{i}" for i in range(19, 29)]:
    check(f"assumption {aid} logged", f"### {aid} " in ass)
check("resolution choice justified with measured cell radii",
      "9.9" in ass and "15.2" in ass)
check("normalisation choice justified with measured uint16 ranges",
      "2477" in ass and "8692" in ass)
check("A27 records the sigma 1px->2px override as APPLIED/RESOLVED",
      "OVERRIDDEN to 2 px" in ass and "RESOLVED / APPLIED" in ass)
check("A20 relabelled [ASSUMPTION] -> [LITERATURE]",
      "A20 — Per-image percentile normalisation (p1 / p99.8), clip=False  **[LITERATURE]**" in ass)
check("A28 logs batch_size=4 for 1024px on a 16GB T4",
      "A28" in ass and "batch_size = 4" in ass)
check("cell-size confound collapse after the override is recorded",
      "+0.09" in ass and "-0.54" in ass.replace("−", "-"))

tm = (ROOT / "docs" / "theoretical_motivation.md").read_text(encoding="utf-8")
# whitespace-normalised: the phrase legitimately wraps across a line in the source
tm_flat = " ".join(tm.split())
check("theoretical_motivation.md notes the cell-size confound + §6 partial-correlation control",
      "cell-size confound" in tm_flat and "partial correlation" in tm_flat)
check("theoretical_motivation.md amendment is marked as an authorised edit to a LOCKED doc",
      "AMENDMENT" in tm and "LOCKED" in tm)
pm = (ROOT / "PROJECT_MASTER.md").read_text(encoding="utf-8")
check("PROJECT_MASTER §11 carries the sigma amendment pointer (no silent contradiction)",
      "AMENDMENT" in pm and "A27" in pm)

print()
print("=" * 78)
if warnings:
    print(f"WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  - {w}")
if failures:
    print(f"RESULT: FAIL ({len(failures)} check(s) failed)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: PASS — Phase 2 verification complete.")
print("=" * 78)
print("A27 RESOLVED: curvature sigma overridden 1px -> 2px, bins regenerated pre-training.")
print("OPEN: A28 batch_size=4 must be confirmed against real GPU memory on the first Kaggle run.")
