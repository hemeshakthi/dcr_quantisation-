"""Checkpoint save/load/resume (master §10).

Master §10 requires best AND final checkpoints plus resume. `best` is selected on validation
Boundary F1 -- NOT on Dice. That choice follows directly from P1: pruning is predicted to leave
Dice intact while damaging boundary geometry, so selecting on Dice would be nearly blind to the
quantity the study is about, and could select a checkpoint with visibly worse boundaries.

Every checkpoint stores the config, the channel-mask hash (master §9) and the RNG state, so a
resumed run continues the identical data order rather than reshuffling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(path: str | Path,
                    model: Any,
                    optimizer: Any,
                    scheduler: Any,
                    epoch: int,
                    metrics: dict[str, float],
                    config: dict[str, Any],
                    channel_mask_hash: str | None = None,
                    rng_state: dict[str, Any] | None = None) -> str:
    """Write a checkpoint and return its SHA-256.

    The hash is recorded in the per-image metrics rows, so any number in a table or in the UI
    can be traced to the exact weights that produced it (master §14).
    """
    raise NotImplementedError


def load_checkpoint(path: str | Path, model: Any = None, optimizer: Any = None,
                    scheduler: Any = None, strict: bool = True) -> dict[str, Any]:
    """Load a checkpoint, optionally restoring model/optimizer/scheduler in place.

    Raises on a channel-mask hash mismatch when the checkpoint records one: loading pruned
    weights into a differently-pruned skeleton would either error obscurely or silently
    mis-map channels.
    """
    raise NotImplementedError


def checkpoint_hash(path: str | Path) -> str:
    """SHA-256 of a checkpoint file, for provenance in metrics rows and the UI."""
    raise NotImplementedError


class BestCheckpointTracker:
    """Track the best epoch on a monitored metric and handle plateau early stopping.

    Args:
        monitor: metric name; FROZEN as validation Boundary F1 (master §10).
        mode: "max" for F1/Dice, "min" for HD95 and losses.
        patience: master §10 [ASSUMPTION] value is 15.
        min_delta: improvement below this does not count as improvement.
    """

    def __init__(self, monitor: str = "val_boundary_f1", mode: str = "max",
                 patience: int = 15, min_delta: float = 0.0) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.epochs_without_improvement = 0

    def update(self, epoch: int, metrics: dict[str, float]) -> bool:
        """Record an epoch. Returns True if it is a new best."""
        raise NotImplementedError

    def should_stop(self) -> bool:
        """True once `patience` epochs have passed with no improvement."""
        raise NotImplementedError
