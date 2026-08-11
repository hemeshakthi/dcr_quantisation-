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
    image_id cannot be paired later and is worthless. Any PER_IMAGE_COLUMNS key a row omits is
    written as empty rather than silently reordering or dropping columns, so the header stays
    identical across every append.
    """
    if not rows:
        raise ValueError("no rows to append")
    required_keys = ("run_id", "model", "seed", "image_id")
    for i, r in enumerate(rows):
        missing = [k for k in required_keys if r.get(k) in (None, "")]
        if missing:
            raise ValueError(
                f"row {i} missing identifying key(s) {missing} -- cannot be paired later")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    for col in PER_IMAGE_COLUMNS:
        if col not in df_new.columns:
            df_new[col] = pd.NA
    df_new = df_new[PER_IMAGE_COLUMNS]
    write_header = not p.exists()
    df_new.to_csv(p, mode="a", header=write_header, index=False)
    return p


def load_per_image_metrics(path: str | Path) -> pd.DataFrame:
    """Load one or more per-image metric CSVs into a single frame."""
    if isinstance(path, (list, tuple)):
        frames = [pd.read_csv(p) for p in path]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PER_IMAGE_COLUMNS)
    return pd.read_csv(path)


def join_curvature_bins(df: pd.DataFrame, curvature_csv: str | Path) -> pd.DataFrame:
    """Attach the write-once curvature bins to a metrics frame, joining on image_id.

    Raises if any metric row has no matching curvature row -- an unbinned image cannot be
    placed in the stratified analysis and must not be silently dropped.
    """
    cb = pd.read_csv(curvature_csv)[["image_id", "bin", "curvature_score", "cell_size", "contrast"]]
    cb = cb.rename(columns={"bin": "curvature_bin"})
    base = df.drop(columns=["curvature_bin", "curvature_score", "cell_size", "contrast"],
                   errors="ignore")
    merged = base.merge(cb, on="image_id", how="left")
    unmatched = merged[merged["curvature_bin"].isna()]
    if len(unmatched):
        ids = sorted(unmatched["image_id"].unique())
        raise ValueError(
            f"{len(unmatched)} metric row(s) have no matching curvature row for image_id(s): "
            f"{ids[:5]}{'...' if len(ids) > 5 else ''} -- an unbinned image cannot be placed "
            f"in the stratified analysis")
    return merged


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
    sub = df[df["curvature_bin"] == bin_name]
    if "pruning_ratio" in sub.columns and ratio is not None:
        sub = sub[sub["pruning_ratio"] == ratio]

    def _one(model_name: str) -> pd.Series:
        s = sub[sub["model"] == model_name]
        if seed is not None:
            s = s[s["seed"] == seed]
        return s.groupby("image_id")[metric].mean()

    a = _one(model_a).rename(f"{metric}_{model_a}")
    b = _one(model_b).rename(f"{metric}_{model_b}")
    merged = pd.concat([a, b], axis=1, join="inner").reset_index()
    merged["diff"] = merged[f"{metric}_{model_a}"] - merged[f"{metric}_{model_b}"]
    return merged


def assert_no_identical_predictions(df: pd.DataFrame) -> list[str]:
    """Flag model pairs whose per-image metrics are byte-identical across every image.

    Master Golden Rule 2: two identical model outputs is a BUG, never an acceptable result. A
    previous attempt at this project failed exactly this way, so the check runs automatically
    over the metrics table rather than waiting for someone to notice.

    Returns:
        Human-readable descriptions of any suspicious pairs; empty means all models differ.
    """
    metric_cols = [c for c in ("dice", "boundary_f1_2px", "boundary_f1_5px", "hd95")
                  if c in df.columns]
    problems: list[str] = []
    if not metric_cols or "model" not in df.columns:
        return problems
    models = sorted(df["model"].dropna().unique())
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            ma, mb = models[i], models[j]
            a = df[df["model"] == ma].set_index("image_id")[metric_cols]
            b = df[df["model"] == mb].set_index("image_id")[metric_cols]
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            a_c, b_c = a.loc[common].sort_index(), b.loc[common].sort_index()
            if a_c.equals(b_c):
                problems.append(
                    f"{ma} and {mb} have BYTE-IDENTICAL metrics on all {len(common)} shared "
                    f"images -- Golden Rule 2 violation, investigate before trusting this run")
    return problems
