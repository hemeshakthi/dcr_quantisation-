"""Phase 1 step 2: per-dataset quality control.

Checks, per dataset:
  * image count == mask count, and every image has exactly one matching mask
  * corruption / unreadable files / shape mismatch between image and its mask
  * mask semantics (binary vs instance labels), foreground coverage, object count
  * perceptual-hash NEAR-DUPLICATE detection across ALL image pairs (and across datasets)
  * SEQUENTIAL-SERIES detection -- the generalisation of the "BCCD same-slide" check.
    Both StarDist/TrackMate datasets are TIME-LAPSE movies, so this is not hypothetical:
    adjacent frames of the same field of view are near-identical and MUST NOT be split
    across train/test (master §5 "split by slide if found").
  * 10 image+mask overlay figures per dataset for manual review

Writes:
  data/processed/<name>_inventory.csv        one row per image/mask pair
  data/processed/<name>_nearduplicates.json  near-duplicate graph + series groups
  results/figures/qc/<name>/overlay_*.png    10 overlays for manual QC

Nothing here fabricates data. If a dataset is absent, it is reported as absent.

Usage:
    python src/datasets/quality_check.py
    python src/datasets/quality_check.py --datasets stardist_tcell --hamming 6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.dataset_registry import (  # noqa: E402
    DATA_PROCESSED,
    REGISTRY,
    PRIMARY_CANDIDATES,
    ROOT,
    DatasetSpec,
    print_bccd_todo,
    print_professor_banner,
)

# [ASSUMPTION A13] Perceptual-hash near-duplicate threshold. pHash at hash_size=8 gives a
# 64-bit hash; Hamming distance <= 8 (12.5% of bits) is a widely used "near duplicate"
# operating point. Recorded in implementation_assumptions.md; tune via --hamming.
DEFAULT_HAMMING = 8
N_OVERLAYS = 10
OVERLAY_SEED = 42  # master §13 uses seed 42 for pre-selection; reused here for QC sampling

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


# ----------------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------------
def _read_tifffile(path: Path) -> np.ndarray:
    import tifffile
    return tifffile.imread(str(path))


def _read_pil(path: Path) -> np.ndarray:
    from PIL import Image
    img = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(np.array(img))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames[0] if len(frames) == 1 else np.stack(frames)


def _read_cv2(path: Path) -> np.ndarray:
    import cv2
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise OSError("cv2 returned None")
    return arr


# Multiple backends: tifffile is version-sensitive (it calls reshape(copy=False), which
# needs numpy>=2.0 and raises TypeError under numpy 1.x), so we never depend on a single
# reader. The first backend that succeeds wins; only if ALL fail is the file reported
# corrupt. [ENGINEERING] robustness -- a library incompatibility must not be silently
# misreported as data corruption.
_READERS = (_read_tifffile, _read_pil, _read_cv2)
_reader_errors: dict[str, int] = {}


def read_image(path: Path) -> np.ndarray | None:
    """Read an image via any available backend; None only if every backend fails."""
    for fn in _READERS:
        try:
            arr = fn(path)
            if arr is not None:
                return np.asarray(arr)
        except Exception as e:  # noqa: BLE001 - we deliberately try the next backend
            key = f"{fn.__name__}:{type(e).__name__}"
            _reader_errors[key] = _reader_errors.get(key, 0) + 1
    return None


def to_uint8_2d(arr: np.ndarray) -> np.ndarray:
    """Collapse to a single 2-D uint8 plane for perceptual hashing / display only.

    Deliberately memory-frugal: percentiles are estimated from a strided SUBSAMPLE and the
    rescale runs in float32 in-place. This host's commit limit equals physical RAM (pagefile
    effectively disabled), where a float64 copy of a 1400x1400 plane already raised
    MemoryError. Subsampled percentiles are statistically ample for a display/hash
    normalisation and never touch the arrays used for metrics.
    """
    a = arr
    while a.ndim > 2:
        a = a[0] if a.shape[0] <= 4 and a.shape[0] < a.shape[-1] else a[..., 0]

    step = max(1, int(max(a.shape) // 512))
    sample = np.asarray(a[::step, ::step], dtype=np.float32)
    lo, hi = float(np.percentile(sample, 1)), float(np.percentile(sample, 99))
    if hi <= lo:
        lo, hi = float(sample.min()), float(sample.max())
    del sample
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)

    out = np.asarray(a, dtype=np.float32)
    if out is a:  # never mutate the caller's array
        out = out.copy()
    out -= lo
    out *= (255.0 / (hi - lo))
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


# ----------------------------------------------------------------------------------
# Series-key derivation (the "same-slide / same-movie" grouping)
# ----------------------------------------------------------------------------------
_UUID_SUFFIX = re.compile(
    r"\.tiff?_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_TIMEPOINT_PREFIX = re.compile(r"^t[_:]\d+-\d+\s*-\s*", re.I)
_MOVIE_FRAME_5DIGIT = re.compile(r"^(.*?)[\s_-]*(\d)(\d{4})$")
_TRAILING_DIGITS = re.compile(r"^(.*?)[\s_-]*\d+$")


def source_stem(name: str) -> str:
    """Recover the ORIGINAL source-image filename stem from a ZeroCostDL4Mic filename.

    ZeroCostDL4Mic emits MULTIPLE PATCHES per source image, each with the same original
    stem but a different appended UUID -- e.g. 'ICAM  2-2-1' appears 5 times and
    'T cells 10000' 3 times in the T-cell training set. Patches of one source image are
    literally crops of the same pixels and MUST never be split across train/test.
    """
    s = name
    for pref in ("Training_source_", "Training_target_"):
        if s.startswith(pref):
            s = s[len(pref):]
    s = _UUID_SUFFIX.sub("", s)
    s = re.sub(r"\.tiff?$", "", s, flags=re.I)
    return s.strip() or name


def derive_series_key(name: str) -> str:
    """Derive a 'same acquisition series' (movie / field-of-view) key from a filename.

    Transparent, documented rules -- the resulting group table is PRINTED in the QC report
    so a human can verify it rather than trusting the heuristic blindly:
      1. recover the original source stem (see source_stem) -- this alone groups patches
         of the same source image
      2. strip a leading timepoint token 't:<n>-<N> - '   (flow-chamber convention)
      3. if the stem ends in exactly 5 digits, treat it as <movie><frame:4>, keeping the
         movie digit:            'T cells 10006' -> 'T cells 1'
      4. otherwise strip the trailing frame number:
                                 'ICAM  2-2-1'   -> 'ICAM  2-2'
                                 'FOV3 B0000'    -> 'FOV3 B'
                                 'FOV1 B.czi #1' -> 'FOV1 B.czi #'

    [ASSUMPTION A11] Rules inferred from OBSERVED filenames, not from dataset
    documentation. This is deliberately CONSERVATIVE (coarse groups): over-grouping only
    costs split granularity, whereas under-grouping would leak temporally adjacent frames
    of the same migrating cells across train/test. make_split.py additionally unions these
    groups with the empirical near-duplicate graph, so a wrong heuristic degrades to
    "grouped by measured similarity" rather than silently leaking.
    """
    s = source_stem(name)
    s = _TIMEPOINT_PREFIX.sub("", s).strip()
    m = _MOVIE_FRAME_5DIGIT.match(s)
    if m:
        return f"{m.group(1).strip()} {m.group(2)}".strip()
    m = _TRAILING_DIGITS.match(s)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return s or name


# ----------------------------------------------------------------------------------
# Pair discovery
# ----------------------------------------------------------------------------------
def discover_pairs(spec: DatasetSpec) -> tuple[list[dict], list[str]]:
    """Find image/mask pairs. Returns (pairs, problems)."""
    pairs, problems = [], []
    if not spec.raw_dir.exists():
        return pairs, [f"raw dir missing: {spec.raw_dir}"]

    for src_rel, tgt_rel in spec.pair_dirs:
        src_dir, tgt_dir = spec.raw_dir / src_rel, spec.raw_dir / tgt_rel
        if not src_dir.is_dir() or not tgt_dir.is_dir():
            problems.append(f"pair dirs not found: {src_rel} / {tgt_rel}")
            continue
        imgs = sorted(p for p in src_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        masks = {p.name: p for p in tgt_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
        subset = src_rel.split("/")[-2] if "/" in src_rel else src_rel

        for img in imgs:
            mask = masks.get(img.name)
            if mask is None:
                # ZeroCostDL4Mic sometimes swaps the source/target filename prefix.
                alt = img.name.replace("Training_source", "Training_target")
                mask = masks.get(alt)
            if mask is None:
                problems.append(f"NO MASK for image: {src_rel}/{img.name}")
                continue
            pairs.append({
                "image_path": img, "mask_path": mask,
                "subset": subset, "src_rel": src_rel,
            })
        extra = set(masks) - {p["mask_path"].name for p in pairs}
        for name in sorted(extra):
            if not any(name == m for m in masks):
                continue
        n_extra = len(masks) - sum(1 for p in pairs if p["src_rel"] == src_rel)
        if n_extra > 0:
            problems.append(f"{n_extra} mask(s) in {tgt_rel} with no matching image")
    return pairs, problems


# ----------------------------------------------------------------------------------
# Main per-dataset QC
# ----------------------------------------------------------------------------------
def qc_dataset(spec: DatasetSpec, hamming: int) -> dict:
    import imagehash
    from PIL import Image

    print("\n" + "=" * 78)
    print(f"QC :: {spec.name} :: {spec.title}")
    print("=" * 78)
    print(f"  confidence: {spec.confidence}")
    print(f"  license   : {spec.license}")
    print(f"  raw dir   : {spec.raw_dir}")

    result = {"name": spec.name, "status": "ABSENT", "n_pairs": 0}

    pairs, problems = discover_pairs(spec)
    if not pairs:
        print("  STATUS: ABSENT / UNPAIRED -- no image+mask pairs found.")
        for p in problems:
            print(f"    - {p}")
        result["problems"] = problems
        return result

    print(f"\n  discovered {len(pairs)} candidate image/mask pair(s)")
    for p in problems:
        print(f"    PROBLEM: {p}")

    rows, corrupt = [], []
    for i, pr in enumerate(pairs, 1):
        img_p, mask_p = pr["image_path"], pr["mask_path"]
        img, mask = read_image(img_p), read_image(mask_p)
        if img is None or mask is None:
            corrupt.append(str(img_p if img is None else mask_p))
            continue

        img2d, mask2d = to_uint8_2d(img), mask
        while mask2d.ndim > 2:
            mask2d = mask2d[0] if mask2d.shape[0] < mask2d.shape[-1] else mask2d[..., 0]

        uniq = np.unique(mask2d)
        n_obj = int(len(uniq) - (1 if 0 in uniq else 0))
        fg = float((mask2d > 0).mean())
        shape_ok = img2d.shape[:2] == mask2d.shape[:2]
        if not shape_ok:
            problems.append(
                f"SHAPE MISMATCH {img_p.name}: image {img2d.shape[:2]} vs mask {mask2d.shape[:2]}")

        ph = imagehash.phash(Image.fromarray(img2d))
        stem = img_p.name
        rows.append({
            "image_id": f"{spec.name}/{pr['subset']}/{stem}",
            "dataset": spec.name,
            "subset": pr["subset"],
            "image_path": str(img_p.relative_to(ROOT)).replace("\\", "/"),
            "mask_path": str(mask_p.relative_to(ROOT)).replace("\\", "/"),
            "img_h": int(img2d.shape[0]), "img_w": int(img2d.shape[1]),
            "img_dtype": str(np.asarray(img).dtype),
            "img_ndim": int(np.asarray(img).ndim),
            "mask_dtype": str(np.asarray(mask).dtype),
            "mask_n_unique": int(len(uniq)),
            "mask_max": int(uniq.max()) if uniq.size else 0,
            "mask_is_binary": bool(set(uniq.tolist()) <= {0, 1} or set(uniq.tolist()) <= {0, 255}),
            "n_objects": n_obj,
            "fg_fraction": round(fg, 6),
            "shape_match": shape_ok,
            "source_stem": source_stem(stem),
            "series_key": derive_series_key(stem),
            "phash": str(ph),
        })
        if i % 50 == 0:
            print(f"    ...processed {i}/{len(pairs)}")

    if not rows:
        print("  STATUS: ALL PAIRS UNREADABLE")
        result["problems"] = problems + [f"corrupt: {c}" for c in corrupt]
        return result

    df = pd.DataFrame(rows)

    # ---- near-duplicate graph -----------------------------------------------------
    hashes = [imagehash.hex_to_hash(h) for h in df["phash"]]
    edges = []
    n = len(hashes)
    for a in range(n):
        for b in range(a + 1, n):
            d = hashes[a] - hashes[b]
            if d <= hamming:
                edges.append({"a": df.iloc[a]["image_id"], "b": df.iloc[b]["image_id"],
                              "hamming": int(d)})
    exact = [e for e in edges if e["hamming"] == 0]

    # ---- series groups ------------------------------------------------------------
    series = defaultdict(list)
    for _, r in df.iterrows():
        series[r["series_key"]].append(r["image_id"])

    # ---- report -------------------------------------------------------------------
    n_img = len(df)
    print("\n  ---------------- COUNTS ----------------")
    print(f"    image/mask pairs readable : {n_img}")
    print(f"    corrupt/unreadable files  : {len(corrupt)}")
    for c in corrupt:
        print(f"      ! {c}")
    print(f"    image count == mask count : {'YES' if not problems else 'SEE PROBLEMS'}")
    if _reader_errors:
        print("    reader-backend fallbacks  : (a backend failing is NOT data corruption "
              "so long as another succeeded)")
        for k, v in sorted(_reader_errors.items()):
            print(f"      {k}: {v}")
    print(f"    shape mismatches          : {int((~df['shape_match']).sum())}")
    by_subset = df.groupby("subset").size().to_dict()
    print(f"    per subset                : {by_subset}")

    print("\n  ---------------- IMAGE PROPERTIES ----------------")
    print(f"    resolutions : {sorted(set(zip(df['img_h'], df['img_w'])))}")
    print(f"    image dtypes: {sorted(df['img_dtype'].unique())}")
    print(f"    mask dtypes : {sorted(df['mask_dtype'].unique())}")

    print("\n  ---------------- MASK SEMANTICS ----------------")
    n_binary = int(df["mask_is_binary"].sum())
    print(f"    binary masks           : {n_binary}/{n_img}")
    print(f"    INSTANCE-LABEL masks   : {n_img - n_binary}/{n_img}")
    print(f"    objects per image      : min={df['n_objects'].min()}, "
          f"median={df['n_objects'].median()}, max={df['n_objects'].max()}")
    print(f"    foreground fraction    : min={df['fg_fraction'].min():.4f}, "
          f"median={df['fg_fraction'].median():.4f}, max={df['fg_fraction'].max():.4f}")
    if n_img - n_binary > 0:
        print("    NOTE: instance-label masks detected. The study's task is BINARY boundary")
        print("          segmentation, so masks must be binarised (>0). Instance labels are")
        print("          retained for per-cell size/curvature stats (master §11).")
        print("          Logged as [ASSUMPTION] A14.")

    print("\n  ---------------- NEAR-DUPLICATE DETECTION ----------------")
    print(f"    pHash, Hamming threshold <= {hamming}  ({n*(n-1)//2:,} pairs compared)")
    print(f"    EXACT-hash duplicate pairs : {len(exact)}")
    print(f"    near-duplicate pairs       : {len(edges)}")
    if edges:
        print("    examples (first 10):")
        for e in edges[:10]:
            print(f"      d={e['hamming']:2d}  {e['a']}  <->  {e['b']}")

    print("\n  ---------------- SEQUENTIAL-SERIES DETECTION ----------------")
    print("    (generalisation of the 'BCCD same-slide' check -- master §5)")
    print(f"    distinct series keys: {len(series)}")
    multi = {k: v for k, v in series.items() if len(v) > 1}
    print(f"    series with >1 frame: {len(multi)}")
    for k, v in sorted(series.items(), key=lambda kv: -len(kv[1]))[:15]:
        print(f"      {len(v):4d} frame(s)  series='{k}'")
    if multi:
        print("    => SEQUENTIAL FRAMES PRESENT. Splitting MUST be grouped by series")
        print("       (and by near-duplicate component). make_split.py enforces this.")

    # ---- write artefacts ----------------------------------------------------------
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    inv_path = DATA_PROCESSED / f"{spec.name}_inventory.csv"
    df.to_csv(inv_path, index=False)
    dup_path = DATA_PROCESSED / f"{spec.name}_nearduplicates.json"
    dup_path.write_text(json.dumps({
        "dataset": spec.name,
        "hamming_threshold": hamming,
        "n_images": n_img,
        "n_pairs_compared": n * (n - 1) // 2,
        "exact_duplicate_pairs": exact,
        "near_duplicate_pairs": edges,
        "series_groups": {k: v for k, v in series.items()},
        "series_key_rules": derive_series_key.__doc__,
    }, indent=2), encoding="utf-8")
    print(f"\n    wrote {inv_path.relative_to(ROOT)}")
    print(f"    wrote {dup_path.relative_to(ROOT)}")

    save_overlays(spec, df)

    result.update({
        "status": "OK", "n_pairs": n_img, "n_corrupt": len(corrupt),
        "n_exact_dup_pairs": len(exact), "n_near_dup_pairs": len(edges),
        "n_series": len(series), "n_multi_frame_series": len(multi),
        "problems": problems,
    })
    return result


def save_overlays(spec: DatasetSpec, df: pd.DataFrame) -> None:
    """Save N_OVERLAYS three-panel image+mask overlays for manual review.

    Rendered as plain uint8 arrays via cv2 rather than matplotlib: matplotlib's colormap
    path expands an image to a float64 RGBA LUT (a 700x700 panel alone needs 15 MiB), which
    this host's commit limit cannot satisfy. Staying in uint8 keeps a full three-panel
    figure under ~1 MB and, as a bonus, gives exact control over the contour rendering.

    Panels: [ image | binarised mask | image with boundary contour ].
    The contour is drawn as a white line with a black halo so it stays visible against both
    dark cells and bright background -- greyscale only, consistent with the project's B/W
    convention (master §14).
    """
    import cv2

    out_dir = ROOT / "results" / "figures" / "qc" / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(OVERLAY_SEED)
    idx = rng.choice(len(df), size=min(N_OVERLAYS, len(df)), replace=False)

    max_side = 512

    def _shrink(a: np.ndarray) -> np.ndarray:
        step = max(1, int(max(a.shape[:2]) // max_side))
        return np.ascontiguousarray(a[::step, ::step])

    def _flatten2d(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        while a.ndim > 2:
            a = a[0] if a.shape[0] < a.shape[-1] else a[..., 0]
        return a

    index_lines = []
    written = 0
    for k, i in enumerate(sorted(idx), 1):
        r = df.iloc[int(i)]
        img = read_image(ROOT / r["image_path"])
        mask = read_image(ROOT / r["mask_path"])
        if img is None or mask is None:
            continue
        img8 = _shrink(to_uint8_2d(img))
        m = (_shrink(_flatten2d(mask)) > 0).astype(np.uint8)
        del img, mask

        kernel = np.ones((3, 3), np.uint8)
        edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel).astype(bool)
        halo = cv2.dilate(edge.astype(np.uint8), kernel, iterations=1).astype(bool) & ~edge

        overlay = img8.copy()
        overlay[halo] = 0
        overlay[edge] = 255

        sep = np.full((img8.shape[0], 4), 128, np.uint8)
        panel = np.hstack([img8, sep, m * 255, sep, overlay])
        cv2.imwrite(str(out_dir / f"overlay_{k:02d}.png"), panel)

        index_lines.append(f"overlay_{k:02d}.png\t{r['image_id']}\tn_obj={r['n_objects']}\t"
                           f"fg={r['fg_fraction']}")
        written += 1
        del img8, m, edge, halo, overlay, panel

    (out_dir / "index.tsv").write_text(
        "file\timage_id\tn_objects\tfg_fraction\n" + "\n".join(index_lines) + "\n",
        encoding="utf-8")
    print(f"    wrote {written} overlay(s) [panels: image | mask | contour] -> "
          f"{out_dir.relative_to(ROOT)}")


def cross_dataset_duplicates(inventories: list[pd.DataFrame], hamming: int) -> None:
    import imagehash
    if len(inventories) < 2:
        return
    print("\n" + "=" * 78)
    print("CROSS-DATASET NEAR-DUPLICATE CHECK")
    print("=" * 78)
    allf = pd.concat(inventories, ignore_index=True)
    hashes = [imagehash.hex_to_hash(h) for h in allf["phash"]]
    hits = 0
    for a in range(len(hashes)):
        for b in range(a + 1, len(hashes)):
            if allf.iloc[a]["dataset"] == allf.iloc[b]["dataset"]:
                continue
            d = hashes[a] - hashes[b]
            if d <= hamming:
                hits += 1
                if hits <= 10:
                    print(f"  d={d:2d}  {allf.iloc[a]['image_id']} <-> {allf.iloc[b]['image_id']}")
    print(f"  cross-dataset near-duplicate pairs: {hits}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--hamming", type=int, default=DEFAULT_HAMMING)
    args = ap.parse_args()

    print_professor_banner()

    keys = args.datasets or (list(REGISTRY) if args.all else list(PRIMARY_CANDIDATES))
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        print(f"ERROR: unknown dataset key(s): {unknown}")
        return 2

    results, invs = [], []
    for key in keys:
        res = qc_dataset(REGISTRY[key], args.hamming)
        results.append(res)
        inv = DATA_PROCESSED / f"{key}_inventory.csv"
        if inv.exists():
            invs.append(pd.read_csv(inv))

    cross_dataset_duplicates(invs, args.hamming)

    print_bccd_todo()

    print("\n" + "=" * 78)
    print("QUALITY CHECK SUMMARY")
    print("=" * 78)
    any_data = False
    for r in results:
        if r["status"] == "OK":
            any_data = True
            print(f"  {r['name']:<24} OK      pairs={r['n_pairs']:<5} "
                  f"corrupt={r['n_corrupt']:<3} exact_dup={r['n_exact_dup_pairs']:<4} "
                  f"near_dup={r['n_near_dup_pairs']:<5} series={r['n_series']}")
        else:
            print(f"  {r['name']:<24} {r['status']}")
    if not any_data:
        print("\n  RESULT: PARTIAL -- no dataset present under data/raw/.")
        print("  Run: python src/datasets/download_datasets.py")
        return 1
    print("\n  Reminder: dataset identity remains UNCONFIRMED ([ASSUMPTION] A01).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
