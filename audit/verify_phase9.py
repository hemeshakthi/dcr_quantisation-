"""Phase 9 -- Failure Taxonomy: SYNTHETIC validation (master §12).

Author-only phase: no real checkpoints exist locally yet. For EACH of the six detectors, this
builds one synthetic image pair ENGINEERED to trigger it and one ENGINEERED NOT to, then
confirms each detector correctly distinguishes its own positive/negative case. Detector
thresholds are read from `configs/evaluation.yaml`'s `failure_detectors:` block -- locked before
inspection (master §12), never invented fresh here.

The two instance-level detectors (`cell_merging`/`cell_splitting`) use a synthetic multi-cell
scene sized to the REAL measured object-density range from Phase 1/2
(`data/processed/preprocess_manifest.csv`: stardist_tcell 88-203 objects/image,
stardist_flowchamber 16-134 objects/image) -- 30 synthetic cells, not a trivial 2-cell toy, so
the "not placeholders" requirement is concrete and checkable, not just asserted in prose.

Every persisted output lives under results/synthetic_validation/phase9/, prefixed SYNTHETIC_.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.failure import taxonomy as ft  # noqa: E402

failures: list[str] = []
warnings: list[str] = []


def check(label, cond, warn_only=False):
    ok = bool(cond)
    print(f"[{'PASS' if ok else ('WARN' if warn_only else 'FAIL')}] {label}")
    if not ok and not warn_only:
        failures.append(label)
    return ok


OUT_DIR = ROOT / "results" / "synthetic_validation" / "phase9"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("PHASE 9 VERIFICATION -- Failure Taxonomy (SYNTHETIC validation, master §12)")
print("=" * 78)
print("Every mask pair below is engineered with a KNOWN answer; no trained model involved.")
print(f"Outputs namespaced under {OUT_DIR.relative_to(ROOT)} (SYNTHETIC_ prefix).\n")

# ---- 0. thresholds come from the LOCKED config, not invented here -----------------
ev_cfg = yaml.safe_load((ROOT / "configs" / "evaluation.yaml").read_text(encoding="utf-8"))
FD = ev_cfg["failure_detectors"]
check("failure_detectors config block loaded from configs/evaluation.yaml (locked, master §12)",
      FD.get("enabled") is True)
check("CURVATURE_RELATED/NON_CURVATURE_RELATED in taxonomy.py match the locked config groups",
      list(ft.CURVATURE_RELATED) == FD["curvature_related"]
      and list(ft.NON_CURVATURE_RELATED) == FD["non_curvature_related"])

# ---- real Phase 1/2 object-density stats, re-derived (not trusted from prose) -----
print("\n---------------- real Phase 1/2 object-density grounding ----------------")
manifest = pd.read_csv(ROOT / "data" / "processed" / "preprocess_manifest.csv")
density_by_dataset = manifest.groupby("dataset")["n_instances_out"].agg(["min", "max"])
print(density_by_dataset)
tcell_range = (int(density_by_dataset.loc["stardist_tcell", "min"]),
              int(density_by_dataset.loc["stardist_tcell", "max"]))
flow_range = (int(density_by_dataset.loc["stardist_flowchamber", "min"]),
             int(density_by_dataset.loc["stardist_flowchamber", "max"]))
check(f"stardist_tcell density re-derived as {tcell_range} (expected 88-203)",
      tcell_range == (88, 203))
check(f"stardist_flowchamber density re-derived as {flow_range} (expected 16-134)",
      flow_range == (16, 134))
N_SYNTH_CELLS = 30
check(f"synthetic cell_merging/cell_splitting scenes use {N_SYNTH_CELLS} cells, inside the "
      f"REAL measured range {flow_range[0]}-{tcell_range[1]} (not a 2-cell toy)",
      flow_range[0] <= N_SYNTH_CELLS <= tcell_range[1])


def save_png(name: str, *panels: np.ndarray) -> None:
    imgs = [(p.astype(np.uint8) * 255) for p in panels]
    sep = np.full((imgs[0].shape[0], 4), 128, np.uint8)
    out = imgs[0]
    for im in imgs[1:]:
        out = np.hstack([out, sep, im])
    cv2.imwrite(str(OUT_DIR / f"SYNTHETIC_phase9_{name}.png"), out)


# ======================================================================================
# 1. corner_collapse -- sharp square corners vs the same square with corners chamfered off
# ======================================================================================
print("\n---------------- corner_collapse ----------------")
SIZE = 200


def make_square(canvas=SIZE):
    img = np.zeros((canvas, canvas), np.uint8)
    cv2.fillPoly(img, [np.array([[50, 50], [150, 50], [150, 150], [50, 150]])], 1)
    return img


def make_chamfered_square(canvas=SIZE, cut=25):
    img = np.zeros((canvas, canvas), np.uint8)
    pts = np.array([
        [50 + cut, 50], [150 - cut, 50], [150, 50 + cut], [150, 150 - cut],
        [150 - cut, 150], [50 + cut, 150], [50, 150 - cut], [50, 50 + cut],
    ])
    cv2.fillPoly(img, [pts], 1)
    return img


target_cc = make_square()
pred_cc_negative = target_cc.copy()               # identical -- corners preserved
pred_cc_positive = make_chamfered_square()         # all 4 corners rounded off
save_png("corner_collapse", target_cc, pred_cc_negative, pred_cc_positive)

cc_neg = ft.corner_collapse(pred_cc_negative, target_cc,
                            curvature_percentile=FD["corner_collapse"]["curvature_percentile"],
                            min_drop=FD["corner_collapse"]["min_drop"])
cc_pos = ft.corner_collapse(pred_cc_positive, target_cc,
                            curvature_percentile=FD["corner_collapse"]["curvature_percentile"],
                            min_drop=FD["corner_collapse"]["min_drop"])
print(f"  negative (identical): count={cc_neg['count']}  n_gt_corners={cc_neg['n_gt_corners']}")
print(f"  positive (chamfered): count={cc_pos['count']}  severity={cc_pos['severity']:.3f}")
check("corner_collapse: identical prediction -> count == 0 (negative case correctly clean)",
      cc_neg["count"] == 0)
check("corner_collapse: chamfered prediction -> count > 0 (positive case correctly flagged)",
      cc_pos["count"] > 0)
check("corner_collapse: positive case's severity (mean fractional curvature drop) is finite "
      f"and positive ({cc_pos['severity']:.3f})",
      np.isfinite(cc_pos["severity"]) and cc_pos["severity"] > 0)

# ======================================================================================
# 2. boundary_smoothing -- a spiky star vs a smoothed circle of similar area
# ======================================================================================
print("\n---------------- boundary_smoothing ----------------")


def make_star(canvas=SIZE, center=(100, 100), r_outer=70, r_inner=25, n_spikes=8):
    angles = np.linspace(0, 2 * np.pi, n_spikes * 2, endpoint=False)
    pts = []
    for i, ang in enumerate(angles):
        r = r_outer if i % 2 == 0 else r_inner
        pts.append([center[0] + r * np.cos(ang), center[1] + r * np.sin(ang)])
    img = np.zeros((canvas, canvas), np.uint8)
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 1)
    return img


def make_circle(canvas=SIZE, center=(100, 100), radius=45):
    img = np.zeros((canvas, canvas), np.uint8)
    cv2.circle(img, center, radius, 1, thickness=-1)
    return img


target_bs = make_star()
pred_bs_negative = target_bs.copy()    # identical -- no smoothing
pred_bs_positive = make_circle()       # smoothed to a near-circle, similar area
save_png("boundary_smoothing", target_bs, pred_bs_negative, pred_bs_positive)

bs_neg = ft.boundary_smoothing(pred_bs_negative, target_bs,
                               min_ratio=FD["boundary_smoothing"]["min_perimeter_ratio"])
bs_pos = ft.boundary_smoothing(pred_bs_positive, target_bs,
                               min_ratio=FD["boundary_smoothing"]["min_perimeter_ratio"])
print(f"  negative (identical): ratio={bs_neg['ratio']:.3f}  detected={bs_neg['detected']}")
print(f"  positive (smoothed):  ratio={bs_pos['ratio']:.3f}  detected={bs_pos['detected']}")
check("boundary_smoothing: identical star -> ratio == 1.0, NOT detected",
      bs_neg["detected"] is False and abs(bs_neg["ratio"] - 1.0) < 1e-6)
check("boundary_smoothing: star smoothed to a circle -> ratio well below min_ratio, DETECTED",
      bs_pos["detected"] is True and bs_pos["ratio"] < FD["boundary_smoothing"]["min_perimeter_ratio"])
area_ratio = pred_bs_positive.sum() / target_bs.sum()
check(f"boundary_smoothing positive case has COMPARABLE area (ratio={area_ratio:.2f}, not just "
      f"'smaller'), isolating perimeter shortening as the real signal",
      0.5 < area_ratio < 2.0)

# ======================================================================================
# 3. curvature_flattening -- same star/circle shapes, image-level 90th-pct curvature drop
# ======================================================================================
print("\n---------------- curvature_flattening ----------------")
target_cf = make_star(center=(100, 100), r_outer=70, r_inner=25, n_spikes=10)
pred_cf_negative = target_cf.copy()
pred_cf_positive = make_circle(center=(100, 100), radius=45)
save_png("curvature_flattening", target_cf, pred_cf_negative, pred_cf_positive)

cf_neg = ft.curvature_flattening(pred_cf_negative, target_cf,
                                 percentile=FD["curvature_flattening"]["percentile"],
                                 min_drop=FD["curvature_flattening"]["min_drop"])
cf_pos = ft.curvature_flattening(pred_cf_positive, target_cf,
                                 percentile=FD["curvature_flattening"]["percentile"],
                                 min_drop=FD["curvature_flattening"]["min_drop"])
print(f"  negative (identical): frac_drop={cf_neg['frac_drop']:.3f}  detected={cf_neg['detected']}")
print(f"  positive (rounded):   frac_drop={cf_pos['frac_drop']:.3f}  detected={cf_pos['detected']}")
check("curvature_flattening: identical star -> frac_drop ~ 0, NOT detected",
      cf_neg["detected"] is False and abs(cf_neg["frac_drop"]) < 1e-6)
check("curvature_flattening: star flattened to a circle -> real drop exceeding min_drop, DETECTED",
      cf_pos["detected"] is True and cf_pos["frac_drop"] >= FD["curvature_flattening"]["min_drop"])

# ======================================================================================
# 4/5. cell_merging / cell_splitting -- a 30-cell scene (real Phase 1/2 density range)
# ======================================================================================
print(f"\n---------------- cell_merging / cell_splitting ({N_SYNTH_CELLS}-cell scene) ----------------")
CANVAS = 460
N_ROWS, N_COLS = 5, 6
SPACING = 70
RADIUS = 16


def build_grid_instances(overlap_pair: bool) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """N_ROWS x N_COLS instance grid. If overlap_pair, cells #1 and #2 are moved to touch."""
    gt = np.zeros((CANVAS, CANVAS), np.int32)
    centers = []
    cid = 1
    for r in range(N_ROWS):
        for c in range(N_COLS):
            cy = 45 + r * SPACING
            cx = 45 + c * SPACING
            centers.append([cy, cx, cid])
            cid += 1
    if overlap_pair:
        # move cell #2's centre close to cell #1's so their disks touch/overlap
        centers[1][0] = centers[0][0]
        centers[1][1] = centers[0][1] + int(RADIUS * 1.2)
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS]
    for cy, cx, cid_ in centers:
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= RADIUS ** 2
        gt[mask] = cid_  # later (higher-id) circles win pixel ties in any true overlap
    return gt, [tuple(c) for c in centers]


def render_binary_union(centers: list[tuple[int, int, int]]) -> np.ndarray:
    """The prediction: plain union of FULL circles (not clipped by instance partitioning) --
    what a binary (non-instance-aware) model output actually looks like."""
    pred = np.zeros((CANVAS, CANVAS), np.uint8)
    for cy, cx, _ in centers:
        cv2.circle(pred, (cx, cy), RADIUS, 1, thickness=-1)
    return pred


# ---- cell_merging ----
gt_merge_neg, centers_neg = build_grid_instances(overlap_pair=False)
pred_merge_neg = render_binary_union(centers_neg)
gt_merge_pos, centers_pos = build_grid_instances(overlap_pair=True)
pred_merge_pos = render_binary_union(centers_pos)
save_png("cell_merging", (gt_merge_neg > 0).astype(np.uint8), pred_merge_neg,
        (gt_merge_pos > 0).astype(np.uint8), pred_merge_pos)

n_gt_instances = len(np.unique(gt_merge_neg)) - 1
check(f"synthetic scene has exactly {N_SYNTH_CELLS} GT instances (got {n_gt_instances})",
      n_gt_instances == N_SYNTH_CELLS)

cm_neg = ft.cell_merging(pred_merge_neg, gt_merge_neg,
                         iou_threshold=FD["cell_merging"]["iou_threshold"])
cm_pos = ft.cell_merging(pred_merge_pos, gt_merge_pos,
                         iou_threshold=FD["cell_merging"]["iou_threshold"])
print(f"  negative (all {N_SYNTH_CELLS} cells separated): count={cm_neg['count']} "
      f"n_pred_components={cm_neg['n_pred_components']}")
print(f"  positive (cells #1,#2 overlapping): count={cm_pos['count']} "
      f"merged={cm_pos['merged_components']}")
check(f"cell_merging: {N_SYNTH_CELLS} cleanly-separated cells -> count == 0 "
      f"(and n_pred_components == {N_SYNTH_CELLS})",
      cm_neg["count"] == 0 and cm_neg["n_pred_components"] == N_SYNTH_CELLS)
check("cell_merging: cells #1 and #2 overlapping into one predicted blob -> count == 1, "
      "and the flagged component explains exactly GT ids {1, 2}",
      cm_pos["count"] == 1
      and sorted(next(iter(cm_pos["merged_components"].values()))) == [1, 2])

# ---- cell_splitting ----
gt_split, centers_split = build_grid_instances(overlap_pair=False)
pred_split_negative = render_binary_union(centers_split)  # every cell -> its own matching blob
# positive: GT cell #1's predicted blob is broken into two disjoint half-disks entirely inside it
pred_split_positive = pred_split_negative.copy()
cy1, cx1, _ = centers_split[0]
pred_split_positive[max(0, cy1 - RADIUS):cy1 + RADIUS + 1,
                    max(0, cx1 - RADIUS):cx1 + RADIUS + 1] = 0
left_half = np.zeros_like(pred_split_positive)
right_half = np.zeros_like(pred_split_positive)
cv2.circle(left_half, (cx1 - RADIUS // 2, cy1), RADIUS // 3, 1, thickness=-1)
cv2.circle(right_half, (cx1 + RADIUS // 2, cy1), RADIUS // 3, 1, thickness=-1)
pred_split_positive = pred_split_positive | left_half | right_half
save_png("cell_splitting", (gt_split > 0).astype(np.uint8), pred_split_negative,
        pred_split_positive)

cs_neg = ft.cell_splitting(pred_split_negative, gt_split,
                           iou_threshold=FD["cell_splitting"]["iou_threshold"])
cs_pos = ft.cell_splitting(pred_split_positive, gt_split,
                           iou_threshold=FD["cell_splitting"]["iou_threshold"])
print(f"  negative (every cell matches 1:1): count={cs_neg['count']}")
print(f"  positive (cell #1 broken into 2 fragments): count={cs_pos['count']} "
      f"split={cs_pos['split_instances']}")
check(f"cell_splitting: {N_SYNTH_CELLS} cleanly-matched cells -> count == 0",
      cs_neg["count"] == 0)
check("cell_splitting: GT cell #1 predicted as two disjoint fragments -> count == 1, "
      "flagged for GT id 1", cs_pos["count"] == 1 and 1 in cs_pos["split_instances"])

# ======================================================================================
# 6. thin_structure_loss -- a cell body with a thin spike, present vs missing
# ======================================================================================
print("\n---------------- thin_structure_loss ----------------")


def make_body_with_spike(include_spike: bool) -> np.ndarray:
    img = np.zeros((SIZE, SIZE), np.uint8)
    cv2.circle(img, (100, 120), 30, 1, thickness=-1)   # the cell body
    if include_spike:
        img[60:90, 98:101] = 1                          # a 3px-wide spike (uropod) on top
    return img


target_tsl = make_body_with_spike(include_spike=True)
pred_tsl_negative = make_body_with_spike(include_spike=True)   # spike preserved
pred_tsl_positive = make_body_with_spike(include_spike=False)  # spike lost
save_png("thin_structure_loss", target_tsl, pred_tsl_negative, pred_tsl_positive)

tsl_neg = ft.thin_structure_loss(pred_tsl_negative, target_tsl,
                                 max_width_px=FD["thin_structure_loss"]["max_width_px"])
tsl_pos = ft.thin_structure_loss(pred_tsl_positive, target_tsl,
                                 max_width_px=FD["thin_structure_loss"]["max_width_px"])
print(f"  negative (spike preserved): n_thin_px={tsl_neg['n_thin_px']} "
      f"loss_fraction={tsl_neg['loss_fraction']:.3f} detected={tsl_neg['detected']}")
print(f"  positive (spike missing):   n_thin_px={tsl_pos['n_thin_px']} "
      f"loss_fraction={tsl_pos['loss_fraction']:.3f} detected={tsl_pos['detected']}")
check("thin_structure_loss: a real thin structure was found in GT (n_thin_px > 0)",
      tsl_neg["n_thin_px"] > 0)
check("thin_structure_loss: spike preserved in prediction -> NOT detected",
      tsl_neg["detected"] is False)
check("thin_structure_loss: spike entirely missing from prediction -> DETECTED "
      f"(loss_fraction={tsl_pos['loss_fraction']:.3f})",
      tsl_pos["detected"] is True and tsl_pos["loss_fraction"] > 0.9)

# ======================================================================================
# run_all_detectors + p6_ratio
# ======================================================================================
print("\n---------------- run_all_detectors + p6_ratio ----------------")
all_pos = ft.run_all_detectors(pred_cc_positive, target_cc,
                               pred_instances=pred_merge_pos, gt_instances=gt_merge_pos,
                               thresholds=FD)
check("run_all_detectors returns all 6 detector keys",
      set(all_pos) == {"corner_collapse", "boundary_smoothing", "cell_merging",
                       "cell_splitting", "thin_structure_loss", "curvature_flattening"})

# Known synthetic B/C failure counts: C has fewer curvature-related failures (dCR working) and
# ALSO fewer non-curvature ones, but proportionally less -- classic P6-supporting pattern.
counts_b = {"corner_collapse": 20, "boundary_smoothing": 15, "curvature_flattening": 18,
           "cell_merging": 10, "cell_splitting": 8, "thin_structure_loss": 6}
counts_c = {"corner_collapse": 8, "boundary_smoothing": 6, "curvature_flattening": 7,
           "cell_merging": 9, "cell_splitting": 7, "thin_structure_loss": 5}
p6 = ft.p6_ratio(counts_b, counts_c)
expected_curv_reduction = (20 - 8) + (15 - 6) + (18 - 7)     # = 32
expected_noncurv_reduction = (10 - 9) + (8 - 7) + (6 - 5)    # = 3
check(f"p6_ratio: curvature_reduction == {expected_curv_reduction} (hand-computed)",
      p6["curvature_reduction"] == expected_curv_reduction)
check(f"p6_ratio: non_curvature_reduction == {expected_noncurv_reduction} (hand-computed)",
      p6["non_curvature_reduction"] == expected_noncurv_reduction)
check(f"p6_ratio: ratio == {expected_curv_reduction}/{expected_noncurv_reduction} "
      f"(got {p6['ratio']:.4f}), ratio > 1 supports P6 in this synthetic example",
      abs(p6["ratio"] - expected_curv_reduction / expected_noncurv_reduction) < 1e-9
      and p6["ratio"] > 1)

# undefined-ratio guard: non-curvature reduction of exactly zero must give NaN, never a
# fabricated large number
counts_c_zero_noncurv = dict(counts_c)
counts_c_zero_noncurv["cell_merging"] = counts_b["cell_merging"]
counts_c_zero_noncurv["cell_splitting"] = counts_b["cell_splitting"]
counts_c_zero_noncurv["thin_structure_loss"] = counts_b["thin_structure_loss"]
p6_zero = ft.p6_ratio(counts_b, counts_c_zero_noncurv)
check("p6_ratio: zero non-curvature reduction gives ratio == NaN, not a fabricated large number",
      np.isnan(p6_zero["ratio"]))

try:
    ft.p6_ratio({"corner_collapse": 1}, counts_c)
    check("p6_ratio rejects counts missing a required detector key", False)
except KeyError:
    check("p6_ratio rejects counts missing a required detector key", True)

# ======================================================================================
# write the SYNTHETIC validation report
# ======================================================================================
report = {
    "phase": 9, "synthetic": True,
    "warning": "All mask pairs are SYNTHETIC, engineered with known-by-construction answers. "
              "This report verifies detector CORRECTNESS, not real model failure rates.",
    "n_synthetic_cells": N_SYNTH_CELLS,
    "real_density_range_tcell": tcell_range, "real_density_range_flowchamber": flow_range,
    "thresholds_used": FD,
    "p6_ratio_example": p6,
    "n_failed": len(failures), "n_warnings": len(warnings), "failures": failures,
}
report_path = OUT_DIR / "SYNTHETIC_phase9_validation_report.json"
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
print("RESULT: PASS -- Phase 9 failure taxonomy verified (synthetic, known-answer cases).")
print("=" * 78)
