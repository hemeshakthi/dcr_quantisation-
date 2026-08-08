"""One row per run in results/tables/experiment_log.csv (master §10).

Master §10 requires the three seeds to be logged SEPARATELY: strictly one row per
(model, ratio, seed) -- 24 rows for the full study (Vanilla 3 + A 3 + [B + B' + C] x 2 ratios
x 3 seeds). Averaging seeds into one row would discard the between-seed variance that master
§4's "mean over 3 seeds" criteria depend on.

Every row carries the git commit hash and the composed-config fingerprint, so a result can be
tied to the exact code AND the exact configuration that produced it. The loss columns
(including `curvature_smooth_sigma`) exist so a value that drifted between local verification
and a Kaggle notebook is detectable in the log itself, rather than only showing up as an
inexplicable difference in results.
"""
from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "results" / "tables" / "experiment_log.csv"


def git_commit_hash(short: bool = False) -> str:
    """Current git commit, or 'UNKNOWN' if git is unavailable.

    Suffixed '-dirty' when the working tree has uncommitted changes: a result produced from a
    dirty tree is not reproducible from the recorded commit, and that must be visible in the
    log rather than discovered later.
    """
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD"]
        if short:
            args = ["git", "rev-parse", "--short", "HEAD"]
        h = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{h}-dirty" if dirty else h
    except Exception:
        return "UNKNOWN"


def read_header(path: str | Path = DEFAULT_LOG) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            return row
    return []


def append_run(record: dict[str, Any], path: str | Path = DEFAULT_LOG) -> Path:
    """Append one completed run.

    Validates the record against the existing header and REFUSES to write on unknown keys
    rather than silently producing a ragged table that later analysis would misread.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = read_header(p)
    if not header:
        header = list(record.keys())
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
    unknown = [k for k in record if k not in header]
    if unknown:
        raise ValueError(
            f"experiment_log record has keys not in the header: {unknown}. "
            f"Extend the header deliberately rather than writing a ragged row.")
    with open(p, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([record.get(k, "") for k in header])
    return p


def build_run_record(run_id: str,
                     cfg: dict[str, Any],
                     summary: dict[str, Any],
                     config_fingerprint: str = "",
                     seeds: dict[str, int] | None = None,
                     executed_on: str = "local") -> dict[str, Any]:
    """Assemble the row from a run's config and its Trainer summary."""
    seeds = seeds or {}
    boundary = (cfg.get("loss", {}) or {}).get("boundary", {}) or {}
    opt = cfg.get("optimizer", {}) or {}
    tr = cfg.get("training", {}) or {}
    es = cfg.get("early_stopping", {}) or {}
    pr = cfg.get("pruning", {}) or {}

    return {
        "run_id": run_id,
        "model": cfg.get("run", {}).get("name", ""),
        "pruning_ratio": pr.get("ratio") if pr.get("enabled") else "",
        "seed": seeds.get("model_init_seed", ""),
        "model_init_seed": seeds.get("model_init_seed", ""),
        "augmentation_seed": seeds.get("augmentation_seed", ""),
        "dataset": ",".join(cfg.get("data", {}).get("datasets", []) or []),
        "split_name": cfg.get("data", {}).get("train_split", ""),
        "optimizer": opt.get("name", ""),
        "learning_rate": opt.get("lr", ""),
        "weight_decay": opt.get("weight_decay", ""),
        "batch_size": tr.get("batch_size", ""),
        "epochs_planned": tr.get("max_epochs", ""),
        "epochs_trained": summary.get("epochs_trained", ""),
        "early_stopping_patience": es.get("patience", ""),
        "stopping_epoch": summary.get("stopping_epoch", ""),
        "best_epoch": summary.get("best_epoch", ""),
        "best_metric_name": summary.get("best_metric_name", ""),
        "best_metric_value": summary.get("best_metric_value", ""),
        "params_count": summary.get("params_count", ""),
        "flops": summary.get("flops", ""),
        "mixed_precision": tr.get("mixed_precision", ""),
        "deterministic": summary.get("deterministic", ""),
        "pruned_mask_hash": summary.get("pruned_mask_hash", ""),
        "checkpoint_best_path": summary.get("checkpoint_best_path", ""),
        "checkpoint_final_path": summary.get("checkpoint_final_path", ""),
        "train_loss_final": summary.get("train_loss_final", ""),
        "val_loss_final": summary.get("val_loss_final", ""),
        "val_dice_best": summary.get("val_dice_best", ""),
        "val_boundary_f1_2px_best": summary.get("val_boundary_f1_2px_best", ""),
        "val_boundary_f1_5px_best": summary.get("val_boundary_f1_5px_best", ""),
        "val_hd95_best": summary.get("val_hd95_best", ""),
        "test_dice": summary.get("test_dice", ""),
        "test_boundary_f1_2px": summary.get("test_boundary_f1_2px", ""),
        "test_boundary_f1_5px": summary.get("test_boundary_f1_5px", ""),
        "test_hd95": summary.get("test_hd95", ""),
        "mean_gradient_norm": summary.get("mean_gradient_norm", ""),
        "start_time": summary.get("start_time", ""),
        "end_time": summary.get("end_time", ""),
        "duration_minutes": summary.get("duration_minutes", ""),
        "git_commit_hash": git_commit_hash(),
        "executed_on": executed_on,
        "status": summary.get("status", ""),
        "notes": summary.get("notes", ""),
        # --- loss provenance: makes a config drift detectable in the log itself ---
        "loss_boundary_name": boundary.get("name") or "none",
        "loss_band_width": boundary.get("band_width", ""),
        "loss_beta": boundary.get("beta", ""),
        "curvature_proxy": boundary.get("curvature_proxy", ""),
        "curvature_smooth_sigma": boundary.get("curvature_smooth_sigma", ""),
        "config_fingerprint": config_fingerprint,
        # --- overfit monitor (Phase 6-RUN): small (~170-image) training corpus ---
        "train_val_gap_final": summary.get("train_val_gap_final", ""),
        "train_val_gap_at_best_epoch": summary.get("train_val_gap_at_best_epoch", ""),
        "overfit_gap_threshold": summary.get("overfit_gap_threshold", ""),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def expected_run_count(cfg: dict[str, Any] | None = None) -> int:
    """Vanilla(3) + A(3) + [B + B' + C] x 2 ratios x 3 seeds = 24 (master §10)."""
    return 3 + 3 + 3 * 2 * 3


def audit_completeness(path: str | Path = DEFAULT_LOG, expected: int = 24) -> dict[str, Any]:
    """Check the log for missing or duplicated (model, ratio, seed) combinations.

    A duplicate usually means a run was repeated after a change; the stale row must not silently
    enter the analysis.
    """
    p = Path(path)
    if not p.exists():
        return {"exists": False, "n_rows": 0, "expected": expected}
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keys = [(r.get("model"), r.get("pruning_ratio"), r.get("seed")) for r in rows]
    dupes = {k for k in keys if keys.count(k) > 1}
    return {"exists": True, "n_rows": len(rows), "expected": expected,
            "complete": len(rows) >= expected, "duplicates": sorted(map(str, dupes)),
            "dirty_commits": sorted({r.get("git_commit_hash", "") for r in rows
                                     if str(r.get("git_commit_hash", "")).endswith("-dirty")})}
