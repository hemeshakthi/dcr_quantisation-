"""Test-split image lookup and loading (docs/api_contract.md §2, §3).

Single source of truth: `data/curvature/curvature_bins.csv` (Phase 2 output). It already carries
a `split` column, so `GET /api/images` and `POST /api/infer` both read from this ONE file rather
than joining it against `data/splits/test.csv` -- two files that are supposed to agree, but
joining them would be an extra place for silent drift versus reading the one file that already
has everything (image_id, curvature bin/score, cell_size, contrast, split, and the paths to the
Phase 2 preprocessed .npy arrays).

Restricted to the TEST split only, deliberately: this backend is a live demo/audit tool, not a
training-data browser, and master §6's "test split touched once, at final evaluation" intent
extends naturally to "the live UI shows test images, not train images the models saw".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.dataset_registry import ROOT

CURVATURE_CSV = ROOT / "data" / "curvature" / "curvature_bins.csv"

_KEEP_COLS = ["image_id", "dataset", "bin", "curvature_score", "cell_size", "contrast",
             "proc_image", "proc_instances"]


class ImageNotFoundError(Exception):
    """Not a KeyError: KeyError.__str__ re-reprs its message (`"'msg'"`), which would leak into
    the 404 body's `error` string verbatim (docs/api_contract.md §4)."""


def _load_frame() -> pd.DataFrame:
    """Read the curvature CSV fresh every call.

    Not cached: this file is write-once in normal operation (curvature_stratification.py's own
    guard), but re-reading a ~300-row CSV per request is cheap and means the backend never
    serves a stale image list if the CSV is ever legitimately regenerated (`--force`) while the
    server is running.
    """
    if not CURVATURE_CSV.exists():
        return pd.DataFrame(columns=_KEEP_COLS + ["split"])
    df = pd.read_csv(CURVATURE_CSV)
    return df[df["split"] == "test"].reset_index(drop=True)


def list_test_images() -> list[dict[str, Any]]:
    """All test-split images, labelled by curvature bin (docs/api_contract.md §2)."""
    df = _load_frame()
    out = []
    for _, r in df.iterrows():
        out.append({
            "image_id": r["image_id"],
            "dataset": r["dataset"],
            "curvature_bin": r["bin"],
            "curvature_score": float(r["curvature_score"]),
        })
    return out


def default_image_id(images: list[dict[str, Any]] | None = None) -> str | None:
    """Deterministic default: the first HIGH-bin image_id in ascending sort order.

    A HIGH-bin image is chosen because that is where boundary damage/recovery (the study's
    subject) is most visible on first load. "First in ascending image_id order" needs no RNG and
    is stable across restarts and platforms, which a shuffle-then-pick-index-0 approach would
    not obviously improve on for a single deterministic pick.
    """
    images = images if images is not None else list_test_images()
    high = sorted((im["image_id"] for im in images if im["curvature_bin"] == "high"))
    if high:
        return high[0]
    allids = sorted(im["image_id"] for im in images)
    return allids[0] if allids else None


def load_image_and_gt(image_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the Phase 2 preprocessed image and its binary ground-truth mask for a test image_id.

    Returns:
        (image: float32 (H,W), gt_binary: uint8 (H,W) in {0,1}, meta: row fields the caller
        needs -- dataset/curvature_bin/curvature_score/cell_size/contrast).

    Raises:
        ImageNotFoundError: `image_id` is not in the test split. Never falls back to train --
        see the module docstring.
    """
    df = _load_frame()
    hit = df[df["image_id"] == image_id]
    if hit.empty:
        raise ImageNotFoundError(f"image_id {image_id!r} not found in the test split")
    r = hit.iloc[0]
    img = np.array(np.load(ROOT / r["proc_image"], mmap_mode="r"), dtype=np.float32)
    inst = np.array(np.load(ROOT / r["proc_instances"], mmap_mode="r"))
    gt = (inst > 0).astype(np.uint8)
    meta = {
        "dataset": r["dataset"],
        "curvature_bin": r["bin"],
        "curvature_score": float(r["curvature_score"]),
        "cell_size": float(r["cell_size"]),
        "contrast": float(r["contrast"]),
    }
    return img, gt, meta
