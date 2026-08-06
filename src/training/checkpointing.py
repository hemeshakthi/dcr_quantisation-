"""Checkpoint save/load/resume (master §10).

Master §10 requires best AND final checkpoints plus resume. `best` is selected on validation
Boundary F1 -- NOT Dice. That follows from P1: pruning is predicted to leave Dice intact while
damaging boundary geometry, so selecting on Dice would be nearly blind to the quantity under
study and could pick a checkpoint with visibly worse boundaries.

Every checkpoint stores the config, its fingerprint, the channel-mask hash (master §9) and the
FULL RNG state, so a resumed run continues the identical data order and augmentation stream
rather than silently reshuffling -- which is what makes "resume == run straight through"
verifiable at all.

torch.save/torch.load are called with an explicit `weights_only` argument, present since torch
1.13, so the same call works under both requirements-local.txt and requirements-kaggle.txt.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


def checkpoint_hash(path: str | Path) -> str:
    """SHA-256 of a checkpoint file, for provenance in metric rows and the UI (master §14)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(path: str | Path,
                    model: torch.nn.Module,
                    optimizer: Any = None,
                    scheduler: Any = None,
                    epoch: int = 0,
                    metrics: dict[str, float] | None = None,
                    config: dict[str, Any] | None = None,
                    config_fingerprint: str | None = None,
                    channel_mask_hash: str | None = None,
                    rng_state: dict[str, Any] | None = None,
                    extra: dict[str, Any] | None = None) -> str:
    """Write a checkpoint and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "metrics": metrics or {},
        "config": config or {},
        "config_fingerprint": config_fingerprint,
        "channel_mask_hash": channel_mask_hash,
        "rng_state": rng_state,
        "extra": extra or {},
    }
    torch.save(payload, path)
    return checkpoint_hash(path)


def load_checkpoint(path: str | Path,
                    model: torch.nn.Module | None = None,
                    optimizer: Any = None,
                    scheduler: Any = None,
                    map_location: str = "cpu",
                    strict: bool = True,
                    expected_channel_mask_hash: str | None = None) -> dict[str, Any]:
    """Load a checkpoint, optionally restoring model/optimizer/scheduler in place.

    Raises on a channel-mask hash mismatch: loading pruned weights into a differently-pruned
    skeleton would either error obscurely or, worse, silently mis-map channels and train a model
    that is not the one the experiment believes it is.

    `weights_only=False` is explicit because these payloads legitimately contain non-tensor
    state (config dicts, RNG state) and the default flipped in later torch versions.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    got = ckpt.get("channel_mask_hash")
    if expected_channel_mask_hash is not None and got != expected_channel_mask_hash:
        raise ValueError(
            f"channel mask hash mismatch loading {path}: checkpoint has {got!r}, "
            f"expected {expected_channel_mask_hash!r}. These are different networks "
            f"(master §9) and are not comparable.")
    if model is not None:
        model.load_state_dict(ckpt["model_state"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


class BestCheckpointTracker:
    """Track the best epoch on a monitored metric and handle plateau early stopping.

    Args:
        monitor: metric name; master §10 fixes this at validation Boundary F1.
        mode: "max" for F1/Dice, "min" for HD95 and losses.
        patience: [ASSUMPTION] master §10 value is 15.
        min_delta: improvement smaller than this does not count as improvement.
    """

    def __init__(self, monitor: str = "val_boundary_f1_2px", mode: str = "max",
                 patience: int = 15, min_delta: float = 0.0) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be max|min, got {mode!r}")
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.epochs_without_improvement = 0

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def update(self, epoch: int, value: float) -> bool:
        """Record an epoch. Returns True if it is a new best."""
        if value != value:      # NaN never counts as an improvement
            self.epochs_without_improvement += 1
            return False
        if self._is_better(value):
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    def should_stop(self) -> bool:
        """True once `patience` epochs have passed with no improvement."""
        return self.epochs_without_improvement >= self.patience

    def state_dict(self) -> dict[str, Any]:
        return {"monitor": self.monitor, "mode": self.mode, "patience": self.patience,
                "min_delta": self.min_delta, "best_value": self.best_value,
                "best_epoch": self.best_epoch,
                "epochs_without_improvement": self.epochs_without_improvement}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore on resume -- otherwise early stopping would forget its patience counter and
        a resumed run could train for a different number of epochs than an uninterrupted one."""
        self.best_value = state.get("best_value")
        self.best_epoch = state.get("best_epoch")
        self.epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
