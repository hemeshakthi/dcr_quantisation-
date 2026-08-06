"""Training loop (master §10).

Publication-quality requirements from master §10, all of which this class owns:
  * best + final checkpointing, resume
  * plateau early stopping (patience 15 on val Boundary F1) [ASSUMPTION]
  * ReduceLROnPlateau [ASSUMPTION]
  * TensorBoard + CSV logging, one experiment-log row per run
  * three seeds logged SEPARATELY (never averaged into a single run record)
  * full determinism
  * BatchNorm recalibration after pruning
  * mixed precision OFF for final runs
  * per-epoch gradient norms logged

Mixed precision is off deliberately: fp16 changes numerics enough to perturb small boundary
differences, and the effect sizes this study is chasing (Boundary F1 delta >= 0.04) are small
enough that an fp16-induced difference could be mistaken for a real one.

Identical optimiser and LR across B/B'/C is a hard invariant (master §10), enforced by
composing every run from the same training_shared.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class Trainer:
    """Trains one model for one (model, ratio, seed) combination.

    Args:
        model: the network, already pruned and BN-recalibrated if this is B/B'/C.
        loss_fn: a CombinedLoss.
        train_loader / val_loader: built by src.datasets.dataset.
        config: the composed run config.
        run_id: unique identifier; also the checkpoint and log directory name.
        device: "cuda" or "cpu".
        resume_from: checkpoint path to resume, or None.
    """

    def __init__(self,
                 model: Any,
                 loss_fn: Any,
                 train_loader: Any,
                 val_loader: Any,
                 config: dict[str, Any],
                 run_id: str,
                 device: str = "cuda",
                 resume_from: str | Path | None = None) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.run_id = run_id
        self.device = device
        self.resume_from = resume_from

    def build_optimizer(self) -> Any:
        """Adam by default (master §10 [ASSUMPTION] A04: lr 1e-4, weight decay 1e-5).

        Read entirely from config so B, B' and C provably share it.
        """
        raise NotImplementedError

    def build_scheduler(self, optimizer: Any) -> Any:
        """ReduceLROnPlateau on validation Boundary F1 (master §10 [ASSUMPTION])."""
        raise NotImplementedError

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """One training epoch.

        Calls `dataset.set_epoch(epoch)` so augmentation varies per epoch while staying
        reproducible, and records the per-epoch gradient norm required by master §10 --
        which is also the diagnostic that would reveal a boundary term contributing far more
        or less gradient than intended (see CombinedLoss beta calibration).
        """
        raise NotImplementedError

    def validate(self, epoch: int) -> dict[str, float]:
        """Evaluate on the validation set, returning Dice, Boundary F1 (2 px/5 px) and HD95.

        Boundary F1 drives both checkpoint selection and the LR schedule.
        """
        raise NotImplementedError

    def fit(self) -> dict[str, Any]:
        """Run the full training loop.

        Returns a summary for the experiment log: best epoch and metrics, stopping reason,
        wall-clock, measured params/FLOPs, checkpoint hashes and the channel-mask hash.
        """
        raise NotImplementedError
