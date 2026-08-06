"""Per-image metric storage (master §6).

Master §6 requires per-image storage so the paired Wilcoxon tests can pair by image. One row
per (run, model, ratio, seed, image_id), written to results/per_image_metrics/.

This table is the single source of truth for every number that reaches a table, a figure, or
the UI. Master §14 requires every UI number to trace to a live inference call; keeping one
canonical per-image table is how that traceability is maintained rather than asserted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PER_IMAGE_COLUMNS = [
    "run_id", "model", "pruning_ratio", "seed", "image_id", "dataset", "split",
    "curvature_bin", "curvature_score", "cell_size", "contrast",
    "dice", "boundary_f1_2px", "boundary_precision_2px", "boundary_recall_2px",
    "boundary_f1_5px", "boundary_precision_5px", "boundary_recall_5px", "hd95",
    "n_instances_gt", "n_instances_pred", "checkpoint_hash", "computed_at_utc",
]


def append_per_image_metrics(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Append rows to a per-image metrics CSV, creating it with the canonical header.

    Validates that every row carries the identifying keys; a row without run_id/model/seed/
    image_id cannot be paired later and is worthless.
    """
    raise NotImplementedError


def load_per_image_metrics(path: str | Path) -> pd.DataFrame:
    """Load one or more per-image metric CSVs into a single frame."""
    raise NotImplementedError


def join_curvature_bins(df: pd.DataFrame, curvature_csv: str | Path) -> pd.DataFrame:
    """Attach the write-once curvature bins to a metrics frame, joining on image_id.

    Raises if any metric row has no matching curvature row -- an unbinned image cannot be
    placed in the stratified analysis and must not be silently dropped.
    """
    raise NotImplementedError


def pair_for_comparison(df: pd.DataFrame, model_a: str, model_b: str,
                        ratio: float, bin_name: str, metric: str,
                        seed: int | None = None) -> pd.DataFrame:
    """Build the paired table for one Wilcoxon comparison.

    Returns one row per image_id with both models' values and their difference, keeping only
    images present for BOTH models -- an unpaired image would silently break the paired test's
    assumption.

    Args:
        seed: a specific seed, or None to average across the 3 seeds first (master §4 states
            its criteria on the mean over 3 seeds).
    """
    raise NotImplementedError


def assert_no_identical_predictions(df: pd.DataFrame) -> list[str]:
    """Flag model pairs whose per-image metrics are byte-identical across every image.

    Master Golden Rule 2: two identical model outputs is a BUG, never an acceptable result. A
    previous attempt at this project failed exactly this way, so the check runs automatically
    over the metrics table rather than waiting for someone to notice.

    Returns:
        Human-readable descriptions of any suspicious pairs; empty means all models differ.
    """
    raise NotImplementedError
