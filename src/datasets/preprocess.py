"""Phase 2 step 1: preprocess raw microscopy pairs into model-ready arrays.

Produces, per dataset, under data/processed/<name>/:
    images/<key>.npy          float32, per-image percentile-normalised   (model input)
    binary_masks/<key>.png    uint8 {0,255}, foreground/background       (training target)
    instance_masks/<key>.npy  uint16 instance IDs, PRESERVED             (§11 cell size, §12 merge/split)

Instance IDs are NEVER discarded: the binary mask is derived FROM the instance mask, and both
are written. Touching cells share one binary blob but remain distinct instances, which the
curvature stratification (§11) and the cell_merging/cell_splitting detectors (§12) depend on.

Every choice below is logged in docs/implementation_assumptions.md (A19 resolution,
A20 normalisation, A21 resampling).

Usage:
    python src/datasets/preprocess.py
    python src/datasets/preprocess.py --force        # rebuild existing outputs
    python src/datasets/preprocess.py --limit 5      # smoke test
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.dataset_registry import DATA_PROCESSED, DATA_SPLITS, ROOT  # noqa: E402
from src.datasets.quality_check import read_image  # noqa: E402

# ----------------------------------------------------------------------------------
# FROZEN PREPROCESSING CONSTANTS -- see implementation_assumptions.md A19/A20
# ----------------------------------------------------------------------------------
# [ASSUMPTION A19] Target resolution 1024x1024.
#   * It is the NATIVE resolution of stardist_tcell (210 of 267 images = 79% of the corpus),
#     so 79% of the data is NEVER resampled -- no mask interpolation on the majority of the
#     study's ground truth. Only stardist_flowchamber is rescaled, and only by 0.731x.
#   * MEASURED cell geometry drove this. Median instance equivalent-radius is 9.9 px (tcell)
#     and 15.2 px (flowchamber) at native scale. At 512 those fall to ~4.9/5.5 px, which would
#     make master §6's 5 px Boundary-F1 tolerance EQUAL TO THE WHOLE CELL RADIUS -- the primary
#     metric of this study would stop discriminating. At 256 cells are ~5 px ACROSS and
#     curvature is destroyed outright.
#   * 1024 is divisible by 16, required by the frozen 4-pooling U-Net (master §7). Native 1400
#     is not (1400/16 = 87.5), so the flow-chamber data could not be kept at native size anyway.
TARGET_SIZE = 1024

# [LITERATURE A20] Per-image percentile normalisation, pmin=1, pmax=99.8 (CSBDeep/StarDist
# `normalize()` convention -- the very pipeline that produced these datasets' ground truth).
# MEASURED justification: the two datasets occupy completely different raw uint16 ranges
# (tcell p1..p99.8 ~= 2477..3021; flowchamber ~= 8692..16647, global extremes 925..28383 of a
# 65535 full scale). Any fixed global scaling (e.g. /65535) would compress tcell into
# ~0.038..0.046 -- almost no contrast -- and the two datasets would be mutually
# unnormalisable. Per-image percentile normalisation is therefore mandatory, not stylistic.
# NOT the 8-bit mean=0.5/std=0.5 convention, which assumes a full-range 8-bit source.
# Label upgraded [ASSUMPTION] -> [LITERATURE] 2026-08-06: this is the published CSBDeep/StarDist
# convention (Weigert et al. Nat Methods 2018; Schmidt et al. MICCAI 2018) and specifically the
# one used by the ZeroCostDL4Mic/StarDist toolchain that produced these datasets' ground truth.
NORM_PMIN, NORM_PMAX = 1.0, 99.8
# CSBDeep's default is clip=False; retained so genuine bright outliers stay distinguishable
# rather than being flattened onto 1.0 (this study cares about faint boundary contrast).
NORM_CLIP = False

# [ASSUMPTION A21] Resampling kernels: INTER_AREA for image DOWN-scaling (correct area-average
# for decimation, avoids aliasing) and INTER_NEAREST for BOTH mask types (the only kernel that
# cannot invent label values or blend two instance IDs into a third).
IMG_DOWN_INTERP = cv2.INTER_AREA
MASK_INTERP = cv2.INTER_NEAREST

# [ASSUMPTION A22] Tolerance for the boundary-gradient alignment test. A shifted boundary
# scoring within 5% of the unshifted one is a TIE, not evidence of misalignment -- see the
# reasoning in verify_alignment(). Only a clear reversal (>5% stronger when shifted) fails.
ALIGNMENT_MARGIN_TOL = 0.05


def make_key(image_id: str) -> str:
    """Filesystem-safe, stable, collision-resistant key for an image_id."""
    h = hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:10]
    stem = Path(image_id).stem
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)[:60]
    return f"{safe}__{h}"


def normalize_percentile(img: np.ndarray) -> np.ndarray:
    """CSBDeep-style per-image percentile normalisation -> float32."""
    a = np.asarray(img, dtype=np.float32)
    # percentiles on a strided subsample: identical to within float noise, far cheaper on a
    # host whose commit limit equals physical RAM (A18)
    step = max(1, int(max(a.shape) // 512))
    s = a[::step, ::step]
    lo = float(np.percentile(s, NORM_PMIN))
    hi = float(np.percentile(s, NORM_PMAX))
    del s
    if hi <= lo:
        hi = float(a.max())
        lo = float(a.min())
    if hi <= lo:
        return np.zeros_like(a)
    a -= lo
    a /= (hi - lo)
    if NORM_CLIP:
        np.clip(a, 0.0, 1.0, out=a)
    return a


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    interp = IMG_DOWN_INTERP if img.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def resize_instances(inst: np.ndarray, size: int) -> np.ndarray:
    if inst.shape[0] == size and inst.shape[1] == size:
        return inst
    return cv2.resize(inst, (size, size), interpolation=MASK_INTERP)


def _flatten2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    while a.ndim > 2:
        a = a[0] if a.shape[0] < a.shape[-1] else a[..., 0]
    return a


def preprocess_one(image_path: Path, mask_path: Path, size: int) -> dict | None:
    """Preprocess a single pair. Returns per-image stats, or None if unreadable."""
    img = read_image(image_path)
    inst = read_image(mask_path)
    if img is None or inst is None:
        return None
    img = _flatten2d(img)
    inst = _flatten2d(inst)

    native_h, native_w = img.shape[:2]
    n_inst_before = int(len(np.unique(inst)) - (1 if 0 in inst else 0))

    # native byte order (source tif is big-endian '>u2'; cv2 needs native order)
    img = np.ascontiguousarray(img.astype(np.float32))
    inst = np.ascontiguousarray(inst.astype(np.int32))

    img_r = resize_image(img, size)
    del img
    inst_r = resize_instances(inst, size)
    del inst
    gc.collect()

    img_n = normalize_percentile(img_r)
    del img_r

    binary = (inst_r > 0).astype(np.uint8)
    n_inst_after = int(len(np.unique(inst_r)) - (1 if 0 in inst_r else 0))

    stats = {
        "native_h": native_h, "native_w": native_w,
        "out_h": int(img_n.shape[0]), "out_w": int(img_n.shape[1]),
        "resampled": bool((native_h, native_w) != (size, size)),
        "n_instances_native": n_inst_before,
        "n_instances_out": n_inst_after,
        "instances_lost": n_inst_before - n_inst_after,
        "fg_fraction": float(binary.mean()),
        "img_min": float(img_n.min()), "img_max": float(img_n.max()),
        "img_mean": float(img_n.mean()), "img_std": float(img_n.std()),
    }
    return {"image": img_n, "instances": inst_r.astype(np.uint16),
            "binary": binary, "stats": stats}


def verify_alignment(img: np.ndarray, binary: np.ndarray, inst: np.ndarray,
                     shifts: tuple[int, ...] = (4, 12)) -> dict:
    """Re-verify image/mask alignment AFTER resize, via BOUNDARY-GRADIENT shift sensitivity.

    Shape identity is necessary but not sufficient. The mask BOUNDARY should sit exactly on
    the cell edge, i.e. on the ridge of maximum image gradient. We therefore measure the mean
    Sobel-gradient magnitude sampled along the 1-px mask boundary, and repeat it with the
    boundary rolled by +/-4 and +/-12 px in each axis. Correct registration => the unshifted
    boundary scores the HIGHEST.

    Two earlier statistics were tried and REJECTED on measured evidence, recorded here so they
    are not reintroduced:
      1. centroid of foreground vs centroid of darkest pixels -- defeated by the strong
         illumination vignetting in both datasets (reported ~90 px offsets on pairs that are
         demonstrably well aligned).
      2. mean INTENSITY under the mask, minimised at zero shift -- defeated by the cells'
         phase-contrast donut profile. Measured radial profile of a representative T-cell
         image: 1-2 px inside the boundary = 0.123 (dark rim), 6-20 px inside = 0.656 (bright
         core, BRIGHTER than the 0.496 background). Mean intensity under the mask is thus a
         rim/core mixture with no minimum at zero shift, and flagged 152/267 well-aligned
         pairs as misaligned.
    Boundary gradient has no such failure mode and is the quantity this study actually cares
    about. Measured on the same image: 1.068 unshifted vs 0.94-1.01 at +/-4 px and 0.63-0.69
    at +/-12 px.
    """
    out = {
        "shape_match": bool(img.shape[:2] == binary.shape[:2] == inst.shape[:2]),
        "binary_equals_instance_support": bool(np.array_equal(binary > 0, inst > 0)),
    }
    fg = binary > 0
    if not fg.any():
        out.update({"fg_darker_than_bg": None, "boundary_grad_argmax_is_zero": None,
                    "alignment_margin": None})
        return out

    # background mean derived arithmetically -- `img[~fg]` would materialise a ~4 MB copy of
    # the ~97% background, which this host has been observed to refuse (A18)
    n_fg, n_tot = int(fg.sum()), int(img.size)
    total = float(img.sum(dtype=np.float64))
    fg_sum = float(img[fg].sum(dtype=np.float64))
    fg_mean = fg_sum / max(1, n_fg)
    bg_mean = (total - fg_sum) / max(1, n_tot - n_fg)
    out["fg_mean"] = round(fg_mean, 5)
    out["bg_mean"] = round(bg_mean, 5)
    out["fg_darker_than_bg"] = bool(fg_mean < bg_mean)

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
    edge = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8)).astype(bool)

    aligned_score = float(grad[edge].mean())
    shifted = []
    for d in shifts:
        for axis in (0, 1):
            for sgn in (+1, -1):
                rolled = np.roll(edge, sgn * d, axis=axis)
                shifted.append(float(grad[rolled].mean()))
                del rolled
    best_shifted = max(shifted)
    out["boundary_grad_aligned"] = round(aligned_score, 5)
    out["boundary_grad_best_shifted"] = round(best_shifted, 5)
    out["boundary_grad_argmax_is_zero"] = bool(aligned_score > best_shifted)
    # relative margin: how much stronger the aligned boundary gradient is than the best
    # misaligned one. >0 means correctly registered.
    margin = (aligned_score - best_shifted) / (best_shifted + 1e-8)
    out["alignment_margin"] = round(margin, 4)
    # A strict argmax is too sharp a criterion for these images. Both datasets are
    # phase-contrast: every cell has a dark rim AND a bright outer halo, so a small shift can
    # land the boundary on the halo's gradient ridge, which is nearly as strong. Crowded
    # fields compound this -- a shift lands on a NEIGHBOURING cell's edge. Measured: the 6
    # strictly-failing images have margins of only -0.0013..-0.0114 (a 0.1-1.1% tie) against a
    # corpus median of +0.087, and carry ~186 instances vs the corpus median of 100. Visual
    # inspection of the worst case confirmed the contours sit correctly on the cells.
    # A tie is therefore treated as aligned; only a CLEAR reversal counts as misalignment.
    out["alignment_ok"] = bool(margin > -ALIGNMENT_MARGIN_TOL)
    del fg, grad, edge
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=TARGET_SIZE)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="process only N images (smoke test)")
    args = ap.parse_args()

    train = pd.read_csv(DATA_SPLITS / "train.csv")
    test = pd.read_csv(DATA_SPLITS / "test.csv")
    train["split"], test["split"] = "train", "test"
    allf = pd.concat([train, test], ignore_index=True)
    if args.limit:
        allf = allf.groupby("dataset", group_keys=False).head(args.limit)

    print("=" * 78)
    print("PHASE 2 :: PREPROCESSING")
    print("=" * 78)
    print(f"  target resolution : {args.size}x{args.size}   [ASSUMPTION A19]")
    print(f"  normalisation     : per-image percentile p{NORM_PMIN}/p{NORM_PMAX}, "
          f"clip={NORM_CLIP}   [LITERATURE A20]")
    print(f"  resampling        : image={'INTER_AREA(down)'}, mask=INTER_NEAREST "
          f"  [ASSUMPTION A21]")
    print(f"  images to process : {len(allf)}")

    rows, align_problems = [], []
    for n, (_, r) in enumerate(allf.iterrows(), 1):
        ds = r["dataset"]
        key = make_key(r["image_id"])
        out_img = DATA_PROCESSED / ds / "images" / f"{key}.npy"
        out_bin = DATA_PROCESSED / ds / "binary_masks" / f"{key}.png"
        out_ins = DATA_PROCESSED / ds / "instance_masks" / f"{key}.npy"
        for p in (out_img, out_bin, out_ins):
            p.parent.mkdir(parents=True, exist_ok=True)

        if out_img.exists() and out_bin.exists() and out_ins.exists() and not args.force:
            continue

        res = preprocess_one(ROOT / r["image_path"], ROOT / r["mask_path"], args.size)
        if res is None:
            align_problems.append(f"UNREADABLE: {r['image_id']}")
            continue

        align = verify_alignment(res["image"], res["binary"], res["instances"])
        if not align["shape_match"]:
            align_problems.append(f"SHAPE MISMATCH after resize: {r['image_id']}")
        if not align["binary_equals_instance_support"]:
            align_problems.append(f"BINARY != INSTANCE SUPPORT: {r['image_id']}")
        if align.get("fg_darker_than_bg") is False:
            align_problems.append(
                f"FG NOT DARKER THAN BG (check polarity/alignment): {r['image_id']}")
        if align.get("alignment_ok") is False:
            align_problems.append(
                f"MISALIGNED (margin {align.get('alignment_margin')}): a shifted boundary sits "
                f"clearly stronger than the unshifted one: {r['image_id']}")

        np.save(out_img, res["image"])
        np.save(out_ins, res["instances"])
        cv2.imwrite(str(out_bin), res["binary"] * 255)

        rows.append({
            "image_id": r["image_id"], "key": key, "dataset": ds, "split": r["split"],
            "series_key": r.get("series_key"), "group_id": r.get("group_id"),
            "proc_image": str(out_img.relative_to(ROOT)).replace("\\", "/"),
            "proc_binary": str(out_bin.relative_to(ROOT)).replace("\\", "/"),
            "proc_instances": str(out_ins.relative_to(ROOT)).replace("\\", "/"),
            **res["stats"], **align,
        })
        del res
        gc.collect()
        if n % 25 == 0:
            print(f"    ...{n}/{len(allf)}")

    man_path = DATA_PROCESSED / "preprocess_manifest.csv"
    if rows:
        new = pd.DataFrame(rows)
        if man_path.exists() and not args.force:
            old = pd.read_csv(man_path)
            new = (pd.concat([old, new], ignore_index=True)
                     .drop_duplicates(subset="image_id", keep="last"))
        new.to_csv(man_path, index=False)
    if not man_path.exists():
        print("\n  nothing processed and no manifest present.")
        return 1
    man = pd.read_csv(man_path)

    # ---------------- reports ----------------
    print("\n  ---------------- RESAMPLING ----------------")
    for ds, sub in man.groupby("dataset"):
        native = sorted({(int(a), int(b)) for a, b in zip(sub["native_h"], sub["native_w"])})
        print(f"    {ds:<24} native={native} -> {args.size}x{args.size}   "
              f"resampled={int(sub['resampled'].sum())}/{len(sub)}")
        lost = int(sub["instances_lost"].sum())
        tot = int(sub["n_instances_native"].sum())
        pct = (100.0 * lost / tot) if tot else 0.0
        flag = "" if pct < 1.0 else "   <-- REVIEW"
        print(f"      instances: native={tot}  after_resize={int(sub['n_instances_out'].sum())}"
              f"  lost={lost} ({pct:.2f}%){flag}")

    print("\n  ---------------- ALIGNMENT (post-resize) ----------------")
    print(f"    shape_match all               : {bool(man['shape_match'].all())}")
    print(f"    binary == instance support all: {bool(man['binary_equals_instance_support'].all())}")
    print(f"    fg darker than bg (all)       : {bool(man['fg_darker_than_bg'].all())}")
    print(f"    boundary-gradient alignment OK (all): {bool(man['alignment_ok'].all())}"
          f"   [tol {ALIGNMENT_MARGIN_TOL:+.2f}; vs +/-4,+/-12px shifts]")
    strict = int((~man["boundary_grad_argmax_is_zero"]).sum())
    print(f"    strict argmax ties (informational)  : {strict}/{len(man)} "
          f"(all within tolerance -- phase-contrast halo / crowding, see A22)")
    marg = man["alignment_margin"].dropna()
    if len(marg):
        print(f"    alignment margin (img std)    : min={marg.min():.3f} "
              f"median={marg.median():.3f}  (>0 means correctly registered)")
    if align_problems:
        print(f"    PROBLEMS ({len(align_problems)}):")
        for p in align_problems[:20]:
            print(f"      ! {p}")
    else:
        print("    no alignment problems detected")

    print("\n  ---------------- INTENSITY AFTER NORMALISATION ----------------")
    for ds, sub in man.groupby("dataset"):
        print(f"    {ds:<24} mean={sub['img_mean'].mean():.4f} std={sub['img_std'].mean():.4f} "
              f"min={sub['img_min'].min():.3f} max={sub['img_max'].max():.3f}")

    print("\n  ---------------- CLASS BALANCE PER SPLIT ----------------")
    print(f"    {'dataset':<24} {'split':<7} {'n':>5} {'fg%':>8} {'fg% sd':>8} "
          f"{'inst/img':>9}")
    for (ds, sp), sub in man.groupby(["dataset", "split"]):
        print(f"    {ds:<24} {sp:<7} {len(sub):>5} {100*sub['fg_fraction'].mean():>7.3f}% "
              f"{100*sub['fg_fraction'].std():>7.3f}% {sub['n_instances_out'].mean():>9.1f}")
    for sp, sub in man.groupby("split"):
        print(f"    {'ALL':<24} {sp:<7} {len(sub):>5} {100*sub['fg_fraction'].mean():>7.3f}% "
              f"{100*sub['fg_fraction'].std():>7.3f}% {sub['n_instances_out'].mean():>9.1f}")
    print("    (foreground is a small minority class -- relevant to loss weighting; Dice is "
          "used as the region term, master §8)")

    cfg = {
        "target_size": args.size,
        "normalisation": {"method": "per-image percentile (CSBDeep/StarDist convention)",
                          "pmin": NORM_PMIN, "pmax": NORM_PMAX, "clip": NORM_CLIP},
        "resampling": {"image_down": "INTER_AREA", "image_up": "INTER_LINEAR",
                       "mask": "INTER_NEAREST"},
        "instance_masks_retained": True,
        "source_labels": {"resolution": "[ASSUMPTION] A19",
                          "normalisation": "[LITERATURE] A20",
                          "resampling": "[ASSUMPTION] A21",
                          "binarisation": "[ASSUMPTION] A14"},
        "n_processed": int(len(man)),
    }
    (DATA_PROCESSED / "preprocess_config.json").write_text(json.dumps(cfg, indent=2),
                                                           encoding="utf-8")
    print(f"\n  wrote {man_path.relative_to(ROOT)}  ({len(man)} rows)")
    print(f"  wrote {(DATA_PROCESSED / 'preprocess_config.json').relative_to(ROOT)}")
    return 1 if align_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
