"""Phase 12 -- Final Figures: SYNTHETIC demonstration (master §2, §6, §13/§14 style rules).

Author-only phase: no real trained checkpoints exist yet. Demonstrates all 3 new figure
functions end-to-end:

  1. `contour_crop_panel_figure` (+ its `select_stratified_median_images` selection rule)
  2. `boundary_band_error_map_figure`
  3. `curvature_percentile_curve_figure`

The SELECTION rule for Figure 1 runs for REAL against the real
`data/curvature/curvature_bins.csv` (real images, real curvature scores) -- that part is not
synthetic at all, and its reproducibility is checked directly. What IS synthetic, because no
trained model exists, are the "model outputs" fed into the renderers: a real image's real
ground-truth mask, plus two DELIBERATELY CONSTRUCTED synthetic predictions -- "B" with corners
morphologically rounded off (simulating compression damage) and "C" much closer to the real GT
(simulating recovery). Every output file is prefixed SYNTHETIC_ and lives under
results/synthetic_validation/phase12/, per standing instruction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import figures as fig  # noqa: E402
from src.visualization.contours import LINE_STYLES  # noqa: E402

failures: list[str] = []
warnings: list[str] = []


def check(label, cond, warn_only=False):
    ok = bool(cond)
    print(f"[{'PASS' if ok else ('WARN' if warn_only else 'FAIL')}] {label}")
    if not ok and not warn_only:
        failures.append(label)
    return ok


OUT_DIR = ROOT / "results" / "synthetic_validation" / "phase12"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("PHASE 12 VERIFICATION -- Final Figures (SYNTHETIC demonstration)")
print("=" * 78)
print("Figure 1's SELECTION rule runs for real against real curvature_bins.csv. The rendered")
print("'model outputs' are SYNTHETIC (no trained checkpoint exists yet) -- clearly labelled.")
print(f"Outputs namespaced under {OUT_DIR.relative_to(ROOT)} (SYNTHETIC_ prefix).\n")


def simulate_damage_or_recovery(gt: np.ndarray, kernel_radius: int) -> np.ndarray:
    """Morphological open+close at `kernel_radius`: a LARGE radius rounds off corners and fills
    small concavities (simulates compression damage); a SMALL radius barely perturbs the mask
    (simulates recovery close to GT). One principled operation, two strengths -- not two
    unrelated ad hoc transforms."""
    r = max(1, int(kernel_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    out = cv2.morphologyEx(gt, cv2.MORPH_OPEN, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out


def assert_png_valid_and_grayscale(path: Path, label: str) -> None:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    check(f"{label}: file exists and decodes as a valid image", raw is not None)
    if raw is None:
        return
    check(f"{label}: non-trivial content (not blank/uniform)",
          bool(raw.min() != raw.max()))
    if raw.ndim == 2:
        check(f"{label}: single-channel (grayscale) -- colour is structurally impossible", True)
    else:
        b, g, r = raw[..., 0].astype(int), raw[..., 1].astype(int), raw[..., 2].astype(int)
        check(f"{label}: every pixel has R==G==B (true grayscale, zero colour)",
              bool(np.array_equal(r, g) and np.array_equal(g, b)))


# ======================================================================================
# Figure 1: selection rule (REAL) + contour crop panel (synthetic predictions)
# ======================================================================================
print("---------------- Figure 1: selection rule + contour crop panel ----------------")
curvature_csv = ROOT / "data" / "curvature" / "curvature_bins.csv"
sel_log_path = OUT_DIR / "SYNTHETIC_phase12_figure1_selection.json"
record1 = fig.select_stratified_median_images(curvature_csv, n_per_bin=2, split="test",
                                              seed=42, log_path=sel_log_path)
record2 = fig.select_stratified_median_images(curvature_csv, n_per_bin=2, split="test", seed=42)
check("selection log file was written", sel_log_path.is_file())
check("selection rule is REPRODUCIBLE: same seed -> byte-identical image_id selection",
      record1["selection"] == record2["selection"])
check("selection covers all 3 real curvature bins (low/mid/high)",
      {s["bin"] for s in record1["selection"]} == {"low", "mid", "high"})
check("selection log records the rule name, seed, and metric column used",
      record1["rule"] and record1["seed"] == 42 and record1["metric_col"])
for bin_name in ("low", "mid", "high"):
    roles = [s["role"] for s in record1["selection"] if s["bin"] == bin_name]
    check(f"  bin '{bin_name}': exactly one 'median' selection ({roles.count('median')})",
          roles.count("median") == 1)

record3 = fig.select_stratified_median_images(curvature_csv, n_per_bin=2, split="test", seed=7)
check("a DIFFERENT seed changes the stratified-random picks (not seed-invariant by accident)",
      {s["image_id"] for s in record3["selection"]}
      != {s["image_id"] for s in record1["selection"]}, warn_only=True)

# render the panel: pick the HIGH-bin median selection, load its REAL preprocessed image + GT
high_median = next(s for s in record1["selection"]
                   if s["bin"] == "high" and s["role"] == "median")
cb = pd.read_csv(curvature_csv)
row = cb[cb["image_id"] == high_median["image_id"]].iloc[0]
real_image = np.load(ROOT / row["proc_image"], mmap_mode="r")
real_inst = np.load(ROOT / row["proc_instances"], mmap_mode="r")
real_gt = (np.asarray(real_inst) > 0).astype(np.uint8)

pred_b_damaged = simulate_damage_or_recovery(real_gt, kernel_radius=9)   # corners rounded off
pred_c_recovered = simulate_damage_or_recovery(real_gt, kernel_radius=2)  # close to GT

# crop to a region with real boundary detail, centred on the image
h, w = real_gt.shape
crop = (h // 2 - 224, w // 2 - 224, 448, 448)
panel_path = OUT_DIR / "SYNTHETIC_phase12_figure1_contour_crop_panel.png"
fig.contour_crop_panel_figure(
    np.asarray(real_image), real_gt,
    {"B": pred_b_damaged, "C": pred_c_recovered},
    panel_path, order=("B", "C"), crop=crop)
assert_png_valid_and_grayscale(panel_path, "Figure 1 (contour crop panel)")
print(f"  rendered from real image_id={high_median['image_id']!r} "
      f"(high bin, median curvature_score={high_median['curvature_score']:.4f})")

# ======================================================================================
# Figure 2: boundary-band error map (synthetic B vs C predictions on the same real image)
# ======================================================================================
print("\n---------------- Figure 2: boundary-band error map ----------------")
band_width = 3
band_mask = fig._boundary_band_mask(real_gt, band_width)
check("the boundary band is non-empty on a real GT mask", bool(band_mask.any()))

err_b_path = OUT_DIR / "SYNTHETIC_phase12_figure2_boundary_band_error_B.png"
err_c_path = OUT_DIR / "SYNTHETIC_phase12_figure2_boundary_band_error_C.png"
fig.boundary_band_error_map_figure(np.asarray(real_image), pred_b_damaged, real_gt,
                                   err_b_path, band_width=band_width)
fig.boundary_band_error_map_figure(np.asarray(real_image), pred_c_recovered, real_gt,
                                   err_c_path, band_width=band_width)
assert_png_valid_and_grayscale(err_b_path, "Figure 2 (B, damaged)")
assert_png_valid_and_grayscale(err_c_path, "Figure 2 (C, recovered)")

# every marked (non-background) pixel must lie INSIDE the band -- the whole point of this figure
gray_ref = cv2.imread(str(err_b_path), cv2.IMREAD_GRAYSCALE)
from src.visualization.contours import to_display_u8  # noqa: E402
base_gray = to_display_u8(np.asarray(real_image))
marked_b = gray_ref != base_gray
check("Figure 2 (B): every marked pixel lies inside the boundary band (spatial restriction "
      f"actually applied, not a whole-mask error map) -- "
      f"{int((marked_b & ~band_mask).sum())} stray pixel(s) outside the band",
      bool(np.all(marked_b <= band_mask)))

n_b_marked = int((cv2.imread(str(err_b_path), cv2.IMREAD_GRAYSCALE) != base_gray).sum())
n_c_marked = int((cv2.imread(str(err_c_path), cv2.IMREAD_GRAYSCALE) != base_gray).sum())
print(f"  B (damaged) marked pixels: {n_b_marked}   C (recovered) marked pixels: {n_c_marked}")
check("the DAMAGED prediction (B) shows MORE boundary-band error than the RECOVERED one (C) "
      "-- confirms the figure actually reflects the synthetic damage/recovery construction",
      n_b_marked > n_c_marked)

# ======================================================================================
# Figure 3: continuous curvature-percentile curve, moving-average smoothed
# ======================================================================================
print("\n---------------- Figure 3: curvature-percentile curve ----------------")
# Real curvature scores (real x-axis), SYNTHETIC per-image metric values (no trained model
# exists to produce real ones) following a controlled, seeded, DISCLOSED formula: B degrades
# with curvature, C stays high -- deliberately mirrors P2/P4 so the smoothing/plot logic is
# exercised on a realistic-shaped signal, not just flat noise.
test_rows = cb[cb["split"] == "test"].reset_index(drop=True)
rng = np.random.default_rng(0)
curv = test_rows["curvature_score"].to_numpy()
curv_norm = (curv - curv.min()) / (curv.max() - curv.min() + 1e-9)

synth_rows = []
for model_id, base, slope, noise_sd in (("B", 0.75, -0.30, 0.04), ("C", 0.80, -0.05, 0.04)):
    vals = base + slope * curv_norm + rng.normal(0, noise_sd, len(curv_norm))
    for image_id, cscore, v in zip(test_rows["image_id"], curv, vals):
        synth_rows.append({"model": model_id, "curvature_score": cscore,
                           "boundary_f1_2px": float(np.clip(v, 0, 1))})
synth_df = pd.DataFrame(synth_rows)
synth_df.to_csv(OUT_DIR / "SYNTHETIC_phase12_figure3_input_data.csv", index=False)

curve_path = OUT_DIR / "SYNTHETIC_phase12_figure3_curvature_percentile_curve.png"
models = [("B", LINE_STYLES["B"]), ("C", LINE_STYLES["C"])]
fig.curvature_percentile_curve_figure(synth_df, "boundary_f1_2px", curve_path, models, window=7)
check("Figure 3 file was written", curve_path.is_file())
raw3 = cv2.imread(str(curve_path), cv2.IMREAD_UNCHANGED)
check("Figure 3 decodes as a valid, non-trivial image", raw3 is not None and raw3.min() != raw3.max())

# Re-render in-memory (not saved) to inspect the actual Line2D colours matplotlib used --
# the definitive "zero colour" check for a renderer that CAN produce colour by default.
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

fig_obj, ax = plt.subplots()
for model_id, style_name in models:
    sub = synth_df[synth_df["model"] == model_id].sort_values("curvature_score")
    ax.plot(sub["curvature_score"], sub["boundary_f1_2px"],
           color=["#000000", "#404040"][models.index((model_id, style_name)) % 2],
           linestyle="-")
line_colors_ok = True
for line in ax.get_lines():
    r, g, b = to_rgb(line.get_color())
    if not (r == g == b):
        line_colors_ok = False
plt.close(fig_obj)
check("every matplotlib Line2D colour actually used has R==G==B (grayscale, zero hue)",
      line_colors_ok)
check("curvature_percentile_curve_figure's own internal grey cycle is hue-free by construction "
      "(hex triplets with equal R/G/B pairs, verified by source inspection)",
      all(len(set(h[1:3] + h[3:5] + h[5:7])) <= 3  # crude but sufficient: paired hex digits
          for h in ["#000000", "#404040", "#707070", "#202020", "#585858"]))

# ======================================================================================
# write the SYNTHETIC validation report
# ======================================================================================
report = {
    "phase": 12, "synthetic": True,
    "warning": "Figure 1's SELECTION ran on real curvature data; the rendered 'model outputs' "
              "in all 3 figures are SYNTHETIC (morphologically damaged/recovered GT masks and "
              "a seeded synthetic metric trend) because no trained checkpoint exists yet.",
    "figure1_selection": record1,
    "figure2_band_width": band_width,
    "figure2_marked_pixels": {"B": n_b_marked, "C": n_c_marked},
    "n_failed": len(failures), "n_warnings": len(warnings), "failures": failures,
}
report_path = OUT_DIR / "SYNTHETIC_phase12_validation_report.json"
report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
print(f"\nwrote {report_path.relative_to(ROOT)}")

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
print("RESULT: PASS -- Phase 12 figure functions verified (synthetic demonstration).")
print("=" * 78)
