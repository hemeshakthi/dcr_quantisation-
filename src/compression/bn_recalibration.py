"""BatchNorm recalibration after pruning (master §9).

Removing channels invalidates every BatchNorm's running_mean/running_var: the surviving
channels now see a different input distribution. Fine-tuning without recalibrating starts from
badly-scaled activations, which depresses the pruned model's accuracy for a reason that has
nothing to do with capacity -- and would inflate the apparent "pruning damage" that P1 and P3
are trying to measure.

Master §9 fixes the protocol: 1-2k frozen forward passes over IDENTICAL images IN IDENTICAL
ORDER for B, B' and C. Identical order matters because BatchNorm's running statistics are an
order-dependent exponential moving average -- same images in a different order gives different
statistics, and the three models would no longer start from the same point.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def reset_bn_statistics(model: nn.Module) -> None:
    """Zero every BatchNorm's running estimates before recalibration.

    Without a reset the stale pre-pruning statistics are only partially washed out by the
    recalibration passes, leaving a dependence on the unpruned model that differs with the
    number of passes.
    """
    raise NotImplementedError


def recalibrate_batchnorm(model: nn.Module,
                          batches: Iterable[dict],
                          num_batches: int = 1000,
                          device: str = "cpu",
                          momentum: float | None = None) -> dict:
    """Re-estimate BatchNorm statistics with frozen weights.

    Runs forward-only passes with the model in train() mode (so BN updates its running stats)
    but under `torch.no_grad()` and with no optimiser step, so no weight changes.

    Args:
        batches: a REPRODUCIBLE iterable -- the same images in the same order for B, B' and C.
            A shuffled DataLoader must not be passed here.
        num_batches: master §9 specifies 1-2k passes.
        momentum: None uses cumulative averaging (momentum=None in torch), which is
            order-insensitive in the limit and therefore the safer choice; an EMA momentum
            leaves the result weighted toward the last batches seen.

    Returns:
        A record with the number of passes, the batch fingerprint, and per-layer statistic
        summaries so the audit can prove all three models were recalibrated identically.
    """
    raise NotImplementedError


def batch_order_fingerprint(batches: Iterable[dict]) -> str:
    """Hash of the image_ids in order.

    Written into each run's experiment log so the audit can verify B, B' and C consumed the
    identical recalibration stream, rather than trusting that they did.
    """
    raise NotImplementedError
