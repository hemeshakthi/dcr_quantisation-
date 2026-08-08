"""Validation: per-image metrics over a loader (master §6, §10).

Produces ONE ROW PER IMAGE, never a pre-aggregated summary. Master §6 requires per-image
storage so the paired Wilcoxon tests can pair by image_id; aggregating inside the validation
loop would throw away the pairing the entire statistical design depends on. The epoch-level
means this returns for checkpoint selection are derived FROM those rows, not instead of them.

Selection monitors Boundary F1, not Dice (master §10). P1 predicts Dice survives pruning while
boundary geometry degrades, so a Dice-selected checkpoint would be nearly blind to the quantity
the study exists to measure.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.metrics.segmentation import compute_all_metrics


@torch.no_grad()
def validate(model: torch.nn.Module,
             loader: Any,
             loss_fn: torch.nn.Module | None = None,
             device: str = "cpu",
             threshold: float = 0.5,
             tolerances: tuple[float, ...] = (2.0, 5.0),
             percentile: float = 95.0,
             collect_rows: bool = True,
             prefix: str = "val") -> dict[str, Any]:
    """Evaluate `model` over `loader`.

    Returns:
        {"rows": [per-image dicts], "summary": {epoch-level means}}

    HD95 is NaN for images where either mask is empty (undefined, see metrics). The summary uses
    nanmean so those images are excluded from the average rather than poisoning it -- and the
    count of valid HD95 images is reported so a summary computed from few images is visible.

    `prefix` names the summary keys ("val_boundary_f1_2px" by default). This function is also
    used to evaluate a TRAIN subsample for overfitting monitoring (Phase 6-RUN), where the
    caller passes prefix="train" so "train_boundary_f1_2px" and "val_boundary_f1_2px" can sit
    side by side in the same epoch row without colliding.

    `collect_rows=False` skips building the PER-IMAGE detail dicts (image_id, meta) that the
    Wilcoxon pairing needs later -- it does NOT skip computing metrics. The two are deliberately
    decoupled: an earlier version fed only the (conditionally built) `rows` list into
    `summarise()`, so collect_rows=False silently produced an EMPTY summary -- every value
    NaN -- rather than merely a smaller `rows` list. That is precisely the failure mode this
    project treats as a bug rather than an edge case (cf. the swallowed-KeyError bug this same
    function's monitor lookup was hardened against): a caller that only wants a summary (e.g.
    the train/val gap monitor's train-subsample pass) must still get a REAL summary.
    """
    was_training = model.training
    model.eval()
    model.to(device)

    rows: list[dict[str, Any]] = []
    metrics_only: list[dict[str, Any]] = []
    losses: list[float] = []
    try:
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            logits = model(x)
            if loss_fn is not None:
                losses.append(float(loss_fn(logits, y).detach()))

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).to(torch.uint8).cpu().numpy()
            gts = (y >= 0.5).to(torch.uint8).cpu().numpy()
            ids = batch.get("image_id", [None] * preds.shape[0])
            metas = batch.get("meta", [{}] * preds.shape[0])

            for i in range(preds.shape[0]):
                m = compute_all_metrics(preds[i, 0], gts[i, 0], tolerances, percentile)
                metrics_only.append(m)          # ALWAYS -- summary correctness never depends
                                                 # on collect_rows
                if collect_rows:
                    row = {"image_id": ids[i] if i < len(ids) else None}
                    meta = metas[i] if i < len(metas) else {}
                    if isinstance(meta, dict):
                        row.update({k: meta.get(k) for k in
                                    ("dataset", "split", "curvature_bin",
                                     "curvature_score", "cell_size", "contrast")})
                    row.update(m)
                    rows.append(row)
    finally:
        if was_training:
            model.train()

    summary = summarise(metrics_only, prefix=prefix)
    if losses:
        summary[f"{prefix}_loss"] = float(np.mean(losses))
    return {"rows": rows, "summary": summary}


def summarise(rows: list[dict[str, Any]], prefix: str = "val") -> dict[str, float]:
    """Epoch-level means from per-image rows. NaN-aware for HD95."""
    if not rows:
        return {}
    out: dict[str, float] = {"n_images": float(len(rows))}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float, np.floating))]
    for k in keys:
        vals = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        finite = np.isfinite(vals)
        out[f"{prefix}_{k}"] = float(vals[finite].mean()) if finite.any() else float("nan")
        if k == "hd95":
            # Reported so a summary averaged over only a handful of valid images is visible
            # rather than looking like a full-set result.
            out[f"{prefix}_hd95_n_valid"] = float(finite.sum())
    return out


def monitored_value(summary: dict[str, float], monitor: str) -> float:
    """Read the checkpoint/early-stopping monitor out of a summary.

    Raises KeyError rather than defaulting: silently falling back to another metric would let a
    run select checkpoints on the wrong quantity without any visible error.
    """
    if monitor in summary:
        return float(summary[monitor])
    if f"val_{monitor}" in summary:
        return float(summary[f"val_{monitor}"])
    raise KeyError(
        f"monitor {monitor!r} is not produced by the validator. Available: "
        f"{sorted(k for k in summary if k.startswith('val_'))}. "
        f"Note master §6 defines TWO Boundary-F1 tolerances, so the monitor must name one "
        f"explicitly (e.g. 'val_boundary_f1_2px'); a bare 'val_boundary_f1' matches nothing.")
