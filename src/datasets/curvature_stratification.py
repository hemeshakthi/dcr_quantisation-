"""Phase 2 step 3: curvature stratification (master §11).

Per master §11:
  * discrete curvature on Gaussian-smoothed (sigma=1 px) GROUND-TRUTH contours
  * per-image score = 90th percentile of point curvature
  * 3 bins by thirds
  * write-once to data/curvature/curvature_bins.csv
    (image_id, curvature_score, bin, cell_size, contrast)
  * degeneracy check + 10-image spot check that the high bin has real corners/junctions

Which mask is used where -- this distinction matters and is deliberate:
  * CONTOURS / CURVATURE come from the BINARISED mask. That is what the model actually
    predicts (master §7: one logit channel), so the curvature the study bins on must be the
    curvature of the binary boundary -- including the concave junctions where two touching
    cells meet, which is precisely where dCR is expected to help.
  * CELL SIZE comes from the INSTANCE mask: the mean area PER INSTANCE ID. Touching cells
    share a single binary blob, so measuring size per binary connected component would
    silently merge them and inflate the apparent cell size exactly in the crowded images that
    matter most.

Usage:
    python src/datasets/curvature_stratification.py
    python src/datasets/curvature_stratification.py --bin-scope global --force
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.dataset_registry import DATA_PROCESSED, ROOT  # noqa: E402

CURV_DIR = ROOT / "data" / "curvature"
OUT_CSV = CURV_DIR / "curvature_bins.csv"

# master §11: Gaussian smoothing of the contour, sigma = 1 px. FROZEN -- kept as the default.
# MEASURED CAVEAT (A27), surfaced not silently overridden: sigma=1 px does not fully suppress
# the integer-coordinate staircase of a pixelated contour. Measured over 24 images:
#     sigma  mean 90th-pct kappa   implied radius   between-image CV
#       1.0        0.2654              3.8 px            4.99%
#       2.0        0.1783              5.6 px            7.81%
#       3.0        0.1630              6.1 px           11.01%
#       4.0        0.1608              6.2 px           12.78%
# A curvature of 0.2654 implies a 3.8 px radius of curvature, which is SMALLER than any real
# cell here (measured median cell radius 9.9-15.2 px) -- i.e. at sigma=1 the statistic is
# substantially reporting pixelation rather than cell shape. The mean stabilises from sigma>=2
# (0.178 -> 0.163 -> 0.161) while between-image discriminative spread more than doubles.
# This is a DECISION FOR THE PROFESSOR (see A27); --sigma exposes it without editing §11.
CONTOUR_SIGMA = 1.0
# master §11: per-image score is the 90th percentile of point curvature
SCORE_PERCENTILE = 90.0
# [ASSUMPTION A24] Contours shorter than this are dropped before curvature is computed.
# Discrete curvature on a handful of points is dominated by pixelation rather than shape, and
# a sigma=1px smoother needs a few points either side to be meaningful. 16 points ~= the
# perimeter of a 3px-radius object, well below the measured median cell radius (9.9-15.2 px).
MIN_CONTOUR_POINTS = 16
N_SPOTCHECK = 10
SPOTCHECK_SEED = 42  # master §13 fixes seed 42 for pre-selection; reused for auditability


def contour_curvature(contour_xy: np.ndarray, sigma: float = CONTOUR_SIGMA) -> np.ndarray:
    """Discrete curvature at each point of a CLOSED contour, in 1/px.

    kappa = |x' y'' - y' x''| / (x'^2 + y'^2)^{3/2}

    The contour is periodic, so both the Gaussian smoothing and the derivatives use wrap-around
    ('wrap' mode / np.roll central differences); treating it as an open sequence would create a
    spurious high-curvature artefact at the seam where the contour closes.
    """
    x = contour_xy[:, 0].astype(np.float64)
    y = contour_xy[:, 1].astype(np.float64)
    if sigma > 0:
        x = gaussian_filter1d(x, sigma=sigma, mode="wrap")
        y = gaussian_filter1d(y, sigma=sigma, mode="wrap")
    # periodic central differences
    x1 = (np.roll(x, -1) - np.roll(x, 1)) / 2.0
    y1 = (np.roll(y, -1) - np.roll(y, 1)) / 2.0
    x2 = np.roll(x, -1) - 2.0 * x + np.roll(x, 1)
    y2 = np.roll(y, -1) - 2.0 * y + np.roll(y, 1)
    denom = (x1 * x1 + y1 * y1) ** 1.5
    denom = np.maximum(denom, 1e-8)
    return np.abs(x1 * y2 - y1 * x2) / denom


def image_curvature_stats(binary: np.ndarray, sigma: float = CONTOUR_SIGMA) -> dict:
    """Curvature statistics for one image, from its BINARISED mask."""
    b = (binary > 0).astype(np.uint8)
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_k, n_used, n_dropped = [], 0, 0
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if len(pts) < MIN_CONTOUR_POINTS:
            n_dropped += 1
            continue
        all_k.append(contour_curvature(pts, sigma=sigma))
        n_used += 1
    if not all_k:
        return {"curvature_score": np.nan, "curvature_mean": np.nan,
                "curvature_max": np.nan, "n_contours_used": 0,
                "n_contours_dropped": n_dropped, "n_boundary_points": 0}
    k = np.concatenate(all_k)
    return {
        "curvature_score": float(np.percentile(k, SCORE_PERCENTILE)),
        "curvature_mean": float(k.mean()),
        "curvature_max": float(k.max()),
        "n_contours_used": n_used,
        "n_contours_dropped": n_dropped,
        "n_boundary_points": int(len(k)),
    }


def cell_size_from_instances(inst: np.ndarray) -> dict:
    """Per-cell size from INSTANCE IDs (never from binary blobs).

    Also reports how badly a binary-blob measurement would have differed, which quantifies how
    much cell touching there is in the image -- directly relevant to master §12's
    cell_merging / cell_splitting detectors.
    """
    ids, counts = np.unique(inst[inst > 0], return_counts=True)
    if len(ids) == 0:
        return {"cell_size": np.nan, "cell_size_median": np.nan, "n_instances": 0,
                "n_binary_blobs": 0, "instances_per_blob": np.nan}
    n_blobs, _ = cv2.connectedComponents((inst > 0).astype(np.uint8), connectivity=8)
    n_blobs -= 1  # drop background label
    return {
        "cell_size": float(counts.mean()),           # mean area per INSTANCE, px^2
        "cell_size_median": float(np.median(counts)),
        "n_instances": int(len(ids)),
        "n_binary_blobs": int(n_blobs),
        # >1 means touching cells are being merged by binarisation
        "instances_per_blob": float(len(ids) / n_blobs) if n_blobs else np.nan,
    }


def boundary_contrast(img: np.ndarray, binary: np.ndarray) -> dict:
    """Boundary visibility, for the partial-correlation confound check (master §6/§11).

    [ASSUMPTION A25] `contrast` is defined as the mean Sobel-gradient magnitude sampled along
    the GT boundary, normalised by the image's own intensity std. Rationale: the paper's stated
    target is "weak or low contrast BOUNDARIES", so the confound to control for is how visible
    the boundary is -- not global image contrast. A plain foreground-vs-background intensity
    difference is actively misleading on these datasets: measured radial profiles show cells
    have a DARK rim (0.123) and a BRIGHT core (0.656) against a 0.496 background, so the two
    means partially cancel. The Michelson figure is retained alongside for reference only.
    """
    # in-place magnitude into gx: this host's commit limit equals physical RAM (A18) and a
    # third full-size float32 buffer has been observed to fail even at 4 MiB.
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    # in-place magnitude: square, add, sqrt with out= so no third full-size buffer is
    # allocated (this host's commit limit equals physical RAM, A18). np.* is used rather than
    # cv2.magnitude's dst argument, which is positional-only and rejects dst=.
    np.multiply(gx, gx, out=gx)
    np.multiply(gy, gy, out=gy)
    np.add(gx, gy, out=gx)
    np.sqrt(gx, out=gx)
    del gy
    grad = gx
    edge = cv2.morphologyEx((binary > 0).astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8)).astype(bool)
    std = float(img.std()) + 1e-8
    out = {"contrast": float(grad[edge].mean() / std) if edge.any() else np.nan}
    # Background mean is derived arithmetically rather than by boolean-indexing the ~97% of
    # pixels that are background: `img[binary == 0]` materialises a ~4 MB copy, which this
    # host has been observed to refuse (A18). Foreground is only ~3% of pixels, so indexing
    # that side is cheap.
    fg = binary > 0
    n_fg = int(fg.sum())
    n_tot = int(img.size)
    if 0 < n_fg < n_tot:
        total = float(img.sum(dtype=np.float64))
        fg_sum = float(img[fg].sum(dtype=np.float64))
        mf = fg_sum / n_fg
        mb = (total - fg_sum) / (n_tot - n_fg)
        out["contrast_michelson"] = float(abs(mf - mb) / (abs(mf) + abs(mb) + 1e-8))
    else:
        out["contrast_michelson"] = np.nan
    del grad, edge
    return out


def assign_bins(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """3 bins by thirds (tertiles of the curvature score).

    [ASSUMPTION A26] "3 bins by thirds" is read as equal-COUNT tertiles (33rd/67th percentile
    cut points) rather than equal-width thirds of the range. Equal counts keep the per-bin
    paired tests (master §6) balanced and are robust to the long right tail of a
    90th-percentile curvature statistic; equal-width thirds would leave the high bin nearly
    empty and destroy its statistical power.

    `scope` controls whether the cut points are computed globally or per dataset -- see the
    printed report for the measured justification.
    """
    df = df.copy()
    df["bin"] = pd.NA
    df["bin_scope"] = scope

    def _cut(sub: pd.DataFrame) -> pd.Series:
        s = sub["curvature_score"]
        lo, hi = s.quantile(1 / 3), s.quantile(2 / 3)
        return pd.cut(s, bins=[-np.inf, lo, hi, np.inf],
                      labels=["low", "mid", "high"], include_lowest=True)

    if scope == "global":
        df["bin"] = _cut(df)
    else:
        for ds, sub in df.groupby("dataset"):
            df.loc[sub.index, "bin"] = _cut(sub)
    return df


def save_spotcheck(df: pd.DataFrame, out_dir: Path, n: int = N_SPOTCHECK) -> list[str]:
    """Save n HIGH-bin overlays so a human can confirm real corners/junctions exist."""
    out_dir.mkdir(parents=True, exist_ok=True)
    high = df[df["bin"] == "high"]
    if high.empty:
        return []
    rng = np.random.default_rng(SPOTCHECK_SEED)
    take = high.sample(n=min(n, len(high)), random_state=int(rng.integers(1 << 30)))
    lines = []
    for i, (_, r) in enumerate(take.iterrows(), 1):
        img = np.load(ROOT / r["proc_image"])
        inst = np.load(ROOT / r["proc_instances"])
        b = (inst > 0).astype(np.uint8)
        lo, hi = np.percentile(img[::2, ::2], 1), np.percentile(img[::2, ::2], 99)
        img8 = np.clip((img - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        edge = cv2.morphologyEx(b, cv2.MORPH_GRADIENT, k).astype(bool)
        halo = cv2.dilate(edge.astype(np.uint8), k, iterations=1).astype(bool) & ~edge
        ov = img8.copy()
        ov[halo] = 0
        ov[edge] = 255
        # a 448px crop centred on the highest-curvature contour point makes corners visible
        cy, cx = _peak_curvature_location(b)
        s = 448
        y0 = int(np.clip(cy - s // 2, 0, max(0, b.shape[0] - s)))
        x0 = int(np.clip(cx - s // 2, 0, max(0, b.shape[1] - s)))
        crop_img = img8[y0:y0 + s, x0:x0 + s]
        crop_ov = ov[y0:y0 + s, x0:x0 + s]
        sep = np.full((crop_img.shape[0], 4), 128, np.uint8)
        cv2.imwrite(str(out_dir / f"high_curv_{i:02d}.png"),
                    np.hstack([crop_img, sep, crop_ov]))
        lines.append(f"high_curv_{i:02d}.png\t{r['image_id']}\t"
                     f"curvature_score={r['curvature_score']:.4f}\t"
                     f"cell_size={r['cell_size']:.1f}\tcrop=({y0},{x0})")
        del img, inst, b, img8, edge, halo, ov
        gc.collect()
    (out_dir / "index.tsv").write_text(
        "file\timage_id\tcurvature_score\tcell_size\tcrop_origin\n" + "\n".join(lines) + "\n",
        encoding="utf-8")
    return lines


def _peak_curvature_location(binary: np.ndarray) -> tuple[int, int]:
    """(y, x) of the highest-curvature contour point, for centring the spot-check crop."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best, best_k = (binary.shape[0] // 2, binary.shape[1] // 2), -1.0
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if len(pts) < MIN_CONTOUR_POINTS:
            continue
        k = contour_curvature(pts)
        j = int(np.argmax(k))
        if k[j] > best_k:
            best_k = float(k[j])
            best = (int(pts[j, 1]), int(pts[j, 0]))
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin-scope", choices=["per_dataset", "global"], default="per_dataset")
    ap.add_argument("--sigma", type=float, default=CONTOUR_SIGMA,
                    help="contour Gaussian sigma in px (default 1.0, FROZEN by master §11; "
                         "see A27 for measured evidence that >=2 is less pixelation-driven)")
    ap.add_argument("--force", action="store_true",
                    help="override the WRITE-ONCE guard (invalidates any downstream analysis)")
    args = ap.parse_args()

    if OUT_CSV.exists() and not args.force:
        print(f"WRITE-ONCE GUARD: {OUT_CSV.relative_to(ROOT)} already exists.")
        print("Curvature bins are locked BEFORE training (master §11) and must not drift.")
        print("Re-run with --force only if you intend to invalidate downstream analysis.")
        return 1

    man_path = DATA_PROCESSED / "preprocess_manifest.csv"
    if not man_path.exists():
        print("ERROR: run src/datasets/preprocess.py first.")
        return 2
    man = pd.read_csv(man_path)

    print("=" * 78)
    print("PHASE 2 :: CURVATURE STRATIFICATION (master §11)")
    print("=" * 78)
    print(f"  contour smoothing sigma : {args.sigma} px"
          f"{'  (master §11 default)' if args.sigma == CONTOUR_SIGMA else '  (OVERRIDDEN)'}")
    print(f"  per-image score         : {SCORE_PERCENTILE}th percentile of point curvature")
    print(f"  min contour points      : {MIN_CONTOUR_POINTS}   [ASSUMPTION A24]")
    print(f"  bin scope               : {args.bin_scope}")
    print(f"  images                  : {len(man)}")

    rows = []
    for n, (_, r) in enumerate(man.iterrows(), 1):
        img = np.load(ROOT / r["proc_image"])
        inst = np.load(ROOT / r["proc_instances"])
        binary = (inst > 0).astype(np.uint8)

        rec = {"image_id": r["image_id"], "dataset": r["dataset"], "split": r["split"],
               "key": r["key"], "proc_image": r["proc_image"],
               "proc_instances": r["proc_instances"]}
        rec.update(image_curvature_stats(binary, sigma=args.sigma))
        rec.update(cell_size_from_instances(inst))
        rec.update(boundary_contrast(img, binary))
        rows.append(rec)
        del img, inst, binary
        gc.collect()
        if n % 50 == 0:
            print(f"    ...{n}/{len(man)}")

    df = pd.DataFrame(rows)
    n_nan = int(df["curvature_score"].isna().sum())
    if n_nan:
        print(f"  WARNING: {n_nan} image(s) produced no usable contour and are unbinned")
    df = assign_bins(df, args.bin_scope)

    # ---------------- distributions ----------------
    print("\n  ---------------- CURVATURE SCORE BY DATASET ----------------")
    for ds, sub in df.groupby("dataset"):
        s = sub["curvature_score"].dropna()
        print(f"    {ds:<24} n={len(s):<4} min={s.min():.4f} p33={s.quantile(1/3):.4f} "
              f"median={s.median():.4f} p67={s.quantile(2/3):.4f} max={s.max():.4f}")
    a = df[df.dataset == "stardist_tcell"]["curvature_score"].dropna()
    b = df[df.dataset == "stardist_flowchamber"]["curvature_score"].dropna()
    if len(a) and len(b):
        overlap_lo, overlap_hi = max(a.min(), b.min()), min(a.max(), b.max())
        frac_a = float(((a >= overlap_lo) & (a <= overlap_hi)).mean())
        frac_b = float(((b >= overlap_lo) & (b <= overlap_hi)).mean())
        print(f"    distribution overlap: tcell {100*frac_a:.1f}% / flowchamber "
              f"{100*frac_b:.1f}% of images fall in the shared range "
              f"[{overlap_lo:.4f}, {overlap_hi:.4f}]")
        print("    -> if overlap were poor, GLOBAL binning would confound 'high curvature'")
        print("       with 'is a T-cell image'; per_dataset binning avoids that by design.")

    print("\n  ---------------- BIN OCCUPANCY ----------------")
    print(f"    {'scope':<12}{'dataset':<24}{'bin':<7}{'n':>5}  "
          f"{'curv range':<24}{'mean cell_size px2':>19}")
    for (ds, bn), sub in df.groupby(["dataset", "bin"], observed=True):
        s = sub["curvature_score"]
        print(f"    {args.bin_scope:<12}{ds:<24}{str(bn):<7}{len(sub):>5}  "
              f"[{s.min():.4f}, {s.max():.4f}]      {sub['cell_size'].mean():>13.1f}")
    print("    ---- per split ----")
    for (sp, bn), sub in df.groupby(["split", "bin"], observed=True):
        print(f"    {'':<12}{sp:<24}{str(bn):<7}{len(sub):>5}")

    # ---------------- degeneracy checks ----------------
    print("\n  ---------------- DEGENERACY CHECKS ----------------")
    problems = []
    for bn in ("low", "mid", "high"):
        cnt = int((df["bin"] == bn).sum())
        ok = cnt > 0
        print(f"    [{'PASS' if ok else 'FAIL'}] bin '{bn}' non-empty (n={cnt})")
        if not ok:
            problems.append(f"bin {bn} empty")
    for (ds, _), sub in df.groupby(["dataset", "bin"], observed=True):
        pass
    lo_m = df[df["bin"] == "low"]["curvature_score"].median()
    hi_m = df[df["bin"] == "high"]["curvature_score"].median()
    sep_ok = bool(hi_m > lo_m)
    print(f"    [{'PASS' if sep_ok else 'FAIL'}] high-bin median curvature ({hi_m:.4f}) > "
          f"low-bin median ({lo_m:.4f})")
    if not sep_ok:
        problems.append("high bin not separated from low bin")
    uniq = df["curvature_score"].nunique()
    uok = uniq > 3
    print(f"    [{'PASS' if uok else 'FAIL'}] curvature scores are not collapsed "
          f"({uniq} distinct values)")
    if not uok:
        problems.append("curvature scores collapsed")
    for sp in ("train", "test"):
        missing = [b for b in ("low", "mid", "high")
                   if int(((df["split"] == sp) & (df["bin"] == b)).sum()) == 0]
        ok = not missing
        print(f"    [{'PASS' if ok else 'FAIL'}] split '{sp}' has all three bins"
              f"{'' if ok else f' (missing {missing})'}")
        if not ok:
            problems.append(f"split {sp} missing bins {missing}")
    touch = df["instances_per_blob"].median()
    print(f"    [INFO] median instances per binary blob = {touch:.3f} "
          f"(>1 confirms touching cells are merged by binarisation -- which is why "
          f"cell_size uses INSTANCE ids)")

    # ---------------- write ----------------
    CURV_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["image_id", "curvature_score", "bin", "cell_size", "contrast",
            "dataset", "split", "key", "bin_scope",
            "curvature_mean", "curvature_max", "n_boundary_points",
            "n_contours_used", "n_contours_dropped",
            "cell_size_median", "n_instances", "n_binary_blobs", "instances_per_blob",
            "contrast_michelson", "proc_image", "proc_instances"]
    df[cols].to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV.relative_to(ROOT)}  ({len(df)} rows)")

    spot_dir = ROOT / "results" / "figures" / "curvature_spotcheck"
    lines = save_spotcheck(df, spot_dir)
    print(f"  wrote {len(lines)} HIGH-bin spot-check overlays -> "
          f"{spot_dir.relative_to(ROOT)}")
    print("  >>> MANUAL STEP (master §11): inspect those 10 images and confirm the high bin")
    print("      shows real corners/junctions, not smooth circles.")

    meta = {
        "contour_sigma_px": args.sigma,
        "contour_sigma_is_master_default": bool(args.sigma == CONTOUR_SIGMA),
        "score_percentile": SCORE_PERCENTILE,
        "min_contour_points": MIN_CONTOUR_POINTS,
        "bin_scope": args.bin_scope,
        "bin_rule": "equal-count tertiles (33rd/67th percentile cut points)",
        "curvature_source_mask": "BINARISED (what the model predicts)",
        "cell_size_source_mask": "INSTANCE ids (mean area per instance, not per binary blob)",
        "contrast_definition": "mean Sobel gradient along GT boundary / image intensity std",
        "n_images": int(len(df)),
        "n_unbinned": n_nan,
        "degeneracy_problems": problems,
        "source_labels": {"min_contour_points": "[ASSUMPTION] A24",
                          "contrast_definition": "[ASSUMPTION] A25",
                          "bin_rule": "[ASSUMPTION] A26",
                          "sigma_pixelation_caveat": "[ASSUMPTION] A27",
                          "sigma/percentile/3-bins": "[ENGINEERING] master §11"},
    }
    (CURV_DIR / "curvature_config.json").write_text(json.dumps(meta, indent=2),
                                                    encoding="utf-8")
    print(f"  wrote {(CURV_DIR / 'curvature_config.json').relative_to(ROOT)}")

    if problems:
        print(f"\n  RESULT: DEGENERACY PROBLEMS ({len(problems)})")
        for p in problems:
            print(f"    ! {p}")
        return 1
    print("\n  RESULT: bins are non-degenerate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
