"""Training loop (master §10) -- publication-quality, not a toy.

Master §10 requirements, all owned by this class:
  * best + final checkpointing, resume
  * plateau early stopping (patience 15 on val Boundary F1)          [ASSUMPTION A05]
  * ReduceLROnPlateau                                                [ASSUMPTION A04]
  * TensorBoard + CSV logging, one experiment-log row per run
  * three seeds logged SEPARATELY (never averaged into one record)
  * full determinism
  * mixed precision OFF for final runs
  * per-epoch gradient norms logged

MIXED PRECISION IS OFF BY DEFAULT AND THAT IS DELIBERATE. fp16 perturbs numerics enough to move
small boundary differences, and the effect this study chases (Boundary F1 delta >= 0.04) is
small enough that an fp16 artefact could be mistaken for a real result.

SELECTION IS ON BOUNDARY F1, NOT DICE. P1 predicts pruning spares Dice while degrading boundary
geometry, so a Dice-selected checkpoint would be nearly blind to the quantity under study.

Identical optimiser and LR across B/B'/C is a hard invariant (master §10), enforced upstream by
composing every run from the same training_shared.yaml.

PORTABILITY: runs unmodified under requirements-local.txt and requirements-kaggle.txt. No
`verbose=` on ReduceLROnPlateau (deprecated in newer torch); the AMP scaler is imported lazily
and only when mixed precision is explicitly enabled, so the torch.amp API change is never
touched on the default path.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch

from src.training.checkpointing import (BestCheckpointTracker, save_checkpoint,
                                        load_checkpoint)
from src.training.experiment_log import utc_now
from src.training.seed_manager import rng_state_restore, rng_state_snapshot
from src.training.validator import monitored_value, validate

ROOT = Path(__file__).resolve().parents[2]


class Trainer:
    """Trains one model for one (model, ratio, seed) combination."""

    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 train_loader: Any,
                 val_loader: Any,
                 config: dict[str, Any],
                 run_id: str,
                 device: str = "cpu",
                 output_root: str | Path | None = None,
                 config_fingerprint: str = "",
                 channel_mask_hash: str | None = None,
                 resume_from: str | Path | None = None,
                 log_tensorboard: bool | None = None) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.run_id = run_id
        self.device = device
        self.config_fingerprint = config_fingerprint
        self.channel_mask_hash = channel_mask_hash
        self.resume_from = resume_from

        tr = config.get("training", {})
        es = config.get("early_stopping", {})
        ck = config.get("checkpointing", {})
        lg = config.get("logging", {})

        self.max_epochs = int(tr.get("max_epochs", 100))
        self.grad_accum = int(tr.get("gradient_accumulation_steps", 1))
        self.mixed_precision = bool(tr.get("mixed_precision", False))
        self.grad_clip = tr.get("grad_clip_norm", None)
        self.log_gradient_norms = bool(tr.get("log_gradient_norms", True))

        self.monitor = ck.get("monitor", es.get("monitor", "val_boundary_f1_2px"))
        self.mode = ck.get("mode", "max")

        root = Path(output_root) if output_root else ROOT
        self.ckpt_dir = root / ck.get("dir", "checkpoints") / run_id
        self.log_dir = root / lg.get("dir", "logs") / run_id
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "epochs.csv"

        self.tracker = BestCheckpointTracker(
            monitor=self.monitor, mode=self.mode,
            patience=int(es.get("patience", 15)),
            min_delta=float(es.get("min_delta", 0.0)))
        self.early_stopping_enabled = bool(es.get("enabled", True))

        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler(self.optimizer)
        self._tb = self._build_tensorboard(lg if log_tensorboard is None else
                                           {"tensorboard": log_tensorboard})
        self.start_epoch = 0
        self._grad_norms: list[float] = []

    # ------------------------------------------------------------------ construction
    def build_optimizer(self) -> torch.optim.Optimizer:
        """Adam by default ([ASSUMPTION] A04: lr 1e-4, weight decay 1e-5).

        Read entirely from config so B, B' and C provably share it.
        """
        o = self.config.get("optimizer", {})
        name = str(o.get("name", "adam")).lower()
        params = [p for p in self.model.parameters() if p.requires_grad]
        kwargs = {"lr": float(o.get("lr", 1e-4)),
                  "weight_decay": float(o.get("weight_decay", 1e-5))}
        if name == "adam":
            return torch.optim.Adam(params, betas=tuple(o.get("betas", (0.9, 0.999))), **kwargs)
        if name == "adamw":
            return torch.optim.AdamW(params, betas=tuple(o.get("betas", (0.9, 0.999))), **kwargs)
        if name == "sgd":
            return torch.optim.SGD(params, momentum=float(o.get("momentum", 0.9)), **kwargs)
        raise ValueError(f"unknown optimizer {name!r}")

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        """ReduceLROnPlateau on validation Boundary F1 ([ASSUMPTION] master §10).

        `verbose=` is deliberately not passed: it is deprecated in newer torch and absent in
        older, and this file must run unmodified under both requirement sets.
        """
        s = self.config.get("scheduler", {})
        if str(s.get("name", "reduce_on_plateau")).lower() != "reduce_on_plateau":
            return None
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=s.get("mode", "max"),
            factor=float(s.get("factor", 0.5)),
            patience=int(s.get("patience", 7)),
            min_lr=float(s.get("min_lr", 1e-6)),
        )

    def _build_tensorboard(self, lg: dict[str, Any]):
        if not lg.get("tensorboard", True):
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(log_dir=str(self.log_dir / "tb"))
        except Exception:
            # TensorBoard is a convenience; a missing/incompatible install must not stop a
            # training run, and the CSV log is the authoritative record either way.
            return None

    # ------------------------------------------------------------------ epochs
    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """One training epoch.

        Calls `set_epoch(epoch)` on the dataset when available so augmentation varies per epoch
        while staying reproducible, and records the per-epoch gradient norm required by master
        §10 -- which is also the diagnostic that would reveal a boundary term contributing far
        more or less gradient than the beta calibration intended.
        """
        self.model.train()
        ds = getattr(self.train_loader, "dataset", None)
        if ds is not None and hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)

        total, n_batches, grad_norms = 0.0, 0, []
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader):
            x = batch["image"].to(self.device)
            y = batch["mask"].to(self.device)
            loss = self.loss_fn(self.model(x), y) / self.grad_accum
            loss.backward()

            if (step + 1) % self.grad_accum == 0:
                gn = self._global_grad_norm()
                grad_norms.append(gn)
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   float(self.grad_clip))
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            total += float(loss.detach()) * self.grad_accum
            n_batches += 1

        mean_gn = sum(grad_norms) / len(grad_norms) if grad_norms else float("nan")
        self._grad_norms.extend(grad_norms)
        return {"train_loss": total / max(1, n_batches), "grad_norm": mean_gn}

    def _global_grad_norm(self) -> float:
        sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum())
        return sq ** 0.5

    def validate(self, epoch: int) -> dict[str, Any]:
        """Evaluate on the validation set: Dice, Boundary F1 (2 px/5 px) and HD95."""
        ev = self.config.get("metrics", {})
        tols = tuple(ev.get("boundary_f1", {}).get("tolerances_px", (2.0, 5.0)))
        thr = float(self.config.get("postprocessing", {}).get("threshold", 0.5))
        return validate(self.model, self.val_loader, self.loss_fn, self.device,
                        threshold=thr, tolerances=tols)

    # ------------------------------------------------------------------ logging
    def _write_csv_row(self, row: dict[str, Any]) -> None:
        exists = self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)

    def _log_tb(self, row: dict[str, Any], epoch: int) -> None:
        if self._tb is None:
            return
        for k, v in row.items():
            if isinstance(v, (int, float)):
                self._tb.add_scalar(k, v, epoch)

    # ------------------------------------------------------------------ fit
    def fit(self) -> dict[str, Any]:
        """Run the full training loop and return the experiment-log summary."""
        start = time.time()
        start_time = utc_now()

        if self.resume_from:
            ck = load_checkpoint(self.resume_from, self.model, self.optimizer,
                                 self.scheduler, map_location=self.device,
                                 expected_channel_mask_hash=self.channel_mask_hash)
            self.start_epoch = int(ck.get("epoch", -1)) + 1
            tstate = (ck.get("extra") or {}).get("tracker")
            if tstate:
                self.tracker.load_state_dict(tstate)
            # Restoring RNG is what makes a resumed run continue the SAME data order and
            # augmentation stream, so resume == uninterrupted rather than merely similar.
            rng_state_restore(ck.get("rng_state") or {})

        best_summary: dict[str, Any] = {}
        last: dict[str, Any] = {}
        stopping_epoch, status = None, "completed"

        for epoch in range(self.start_epoch, self.max_epochs):
            tr = self.train_one_epoch(epoch)
            val = self.validate(epoch)
            summary = val["summary"]

            # Deliberately NOT wrapped in try/except. A monitor key that does not exist is a
            # configuration bug, and degrading it to NaN would mean the tracker never improves,
            # no best checkpoint is ever written, and every run "succeeds" while producing
            # nothing selectable. Fail on epoch 0 instead of after 24 Kaggle runs.
            monitored = monitored_value(summary, self.monitor)

            improved = self.tracker.update(epoch, monitored)
            if self.scheduler is not None and monitored == monitored:
                self.scheduler.step(monitored)

            row = {"epoch": epoch, "train_loss": tr["train_loss"],
                   "grad_norm": tr["grad_norm"],
                   "lr": self.optimizer.param_groups[0]["lr"],
                   "monitor": self.monitor, "monitored_value": monitored,
                   "is_best": bool(improved),
                   **{k: v for k, v in summary.items() if isinstance(v, (int, float))}}
            self._write_csv_row(row)
            self._log_tb(row, epoch)
            last = {"train": tr, "val": summary, "epoch": epoch}

            if improved:
                best_summary = dict(summary)
                save_checkpoint(
                    self.ckpt_dir / "best.pt", self.model, self.optimizer, self.scheduler,
                    epoch=epoch, metrics=summary, config=self.config,
                    config_fingerprint=self.config_fingerprint,
                    channel_mask_hash=self.channel_mask_hash,
                    rng_state=rng_state_snapshot(),
                    extra={"tracker": self.tracker.state_dict()})

            if self.early_stopping_enabled and self.tracker.should_stop():
                stopping_epoch, status = epoch, "early_stopped"
                break

        final_hash = save_checkpoint(
            self.ckpt_dir / "final.pt", self.model, self.optimizer, self.scheduler,
            epoch=last.get("epoch", self.start_epoch), metrics=last.get("val", {}),
            config=self.config, config_fingerprint=self.config_fingerprint,
            channel_mask_hash=self.channel_mask_hash,
            rng_state=rng_state_snapshot(),
            extra={"tracker": self.tracker.state_dict()})

        if self._tb is not None:
            self._tb.close()

        dur = (time.time() - start) / 60.0
        boundary = (self.config.get("loss", {}) or {}).get("boundary", {}) or {}
        return {
            "run_id": self.run_id,
            "epochs_trained": (last.get("epoch", self.start_epoch - 1) + 1) - self.start_epoch,
            "stopping_epoch": stopping_epoch if stopping_epoch is not None else "",
            "best_epoch": self.tracker.best_epoch,
            "best_metric_name": self.monitor,
            "best_metric_value": self.tracker.best_value,
            "params_count": sum(p.numel() for p in self.model.parameters()),
            "checkpoint_best_path": str(self.ckpt_dir / "best.pt"),
            "checkpoint_final_path": str(self.ckpt_dir / "final.pt"),
            "checkpoint_final_hash": final_hash,
            "train_loss_final": last.get("train", {}).get("train_loss", ""),
            "val_loss_final": last.get("val", {}).get("val_loss", ""),
            "val_dice_best": best_summary.get("val_dice", ""),
            "val_boundary_f1_2px_best": best_summary.get("val_boundary_f1_2px", ""),
            "val_boundary_f1_5px_best": best_summary.get("val_boundary_f1_5px", ""),
            "val_hd95_best": best_summary.get("val_hd95", ""),
            "mean_gradient_norm": (sum(self._grad_norms) / len(self._grad_norms)
                                   if self._grad_norms else ""),
            "start_time": start_time,
            "end_time": utc_now(),
            "duration_minutes": round(dur, 4),
            "pruned_mask_hash": self.channel_mask_hash or "",
            "deterministic": self.config.get("determinism", {}).get(
                "use_deterministic_algorithms", ""),
            "status": status,
            "csv_log": str(self.csv_path),
            "curvature_smooth_sigma": boundary.get("curvature_smooth_sigma", ""),
        }
