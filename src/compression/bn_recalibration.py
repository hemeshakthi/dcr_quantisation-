"""BatchNorm recalibration after pruning (master §9).

Removing channels invalidates every BatchNorm's running_mean/running_var: the surviving
channels now see a different input distribution. Fine-tuning without recalibrating starts from
badly-scaled activations, which depresses the pruned model's accuracy for a reason that has
nothing to do with capacity -- and would inflate the apparent "pruning damage" that P1 and P3
are trying to measure.

Master §9 fixes the protocol: 1-2k frozen forward passes over IDENTICAL images IN IDENTICAL
ORDER for B, B' and C. Identical order matters because BatchNorm's running statistics are an
order-dependent moving average -- the same images in a different order give different
statistics, and the three models would no longer start from the same point. `momentum=None`
(cumulative averaging) is used by default, which makes the result order-INSENSITIVE and removes
that hazard entirely; the fingerprint is still recorded so the audit can prove the streams
matched.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

import torch
import torch.nn as nn


def reset_bn_statistics(model: nn.Module) -> int:
    """Zero every BatchNorm's running estimates. Returns how many layers were reset.

    Without a reset the stale pre-pruning statistics are only partially washed out by the
    recalibration passes, leaving a dependence on the unpruned model that varies with the
    number of passes -- i.e. a hidden hyperparameter.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()
            n += 1
    return n


def batch_order_fingerprint(batches: Iterable[dict[str, Any]]) -> str:
    """SHA-256 over the batches' image_ids IN ORDER (falls back to tensor bytes).

    Written into each run's experiment log so the audit can VERIFY that B, B' and C consumed an
    identical recalibration stream rather than trusting that they did.
    """
    h = hashlib.sha256()
    for b in batches:
        ids = b.get("image_id")
        if ids is not None:
            for i in (ids if isinstance(ids, (list, tuple)) else [ids]):
                h.update(str(i).encode("utf-8"))
        else:
            t = b["image"]
            h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def recalibrate_batchnorm(model: nn.Module,
                          batches: Iterable[dict[str, Any]],
                          num_batches: int = 1500,
                          device: str = "cpu",
                          momentum: float | None = None,
                          reset_first: bool = True) -> dict[str, Any]:
    """Re-estimate BatchNorm statistics with frozen weights.

    Forward-only passes with the model in train() mode -- so BatchNorm updates its running
    statistics -- but under no_grad and with no optimiser, so no weight changes at all.

    Args:
        batches: a REPRODUCIBLE iterable: the same images in the same order for B, B' and C.
            A shuffled DataLoader must not be passed here.
        num_batches: master §9 specifies 1-2k passes.
        momentum: None uses cumulative averaging (torch's momentum=None), which converges to
            the true dataset statistics and is order-insensitive. An EMA momentum would leave
            the result weighted toward whichever batches happened to come last.
        reset_first: zero the running estimates before accumulating.

    Returns a record with the pass count, the batch fingerprint and per-layer statistic
    summaries, so the audit can prove all three models were recalibrated identically.
    """
    model = model.to(device)
    bn_layers = [m for m in model.modules()
                 if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    if not bn_layers:
        return {"n_bn_layers": 0, "n_batches": 0, "skipped": "no BatchNorm layers"}

    n_reset = reset_bn_statistics(model) if reset_first else 0
    saved_momentum = [m.momentum for m in bn_layers]
    for m in bn_layers:
        m.momentum = momentum   # None => cumulative moving average

    was_training = model.training
    model.train()
    consumed: list[dict[str, Any]] = []
    try:
        n = 0
        for batch in batches:
            if n >= num_batches:
                break
            model(batch["image"].to(device))
            consumed.append(batch)
            n += 1
    finally:
        for m, mom in zip(bn_layers, saved_momentum):
            m.momentum = mom
        if not was_training:
            model.eval()

    return {
        "n_bn_layers": len(bn_layers),
        "n_reset": n_reset,
        "n_batches": n,
        "momentum": momentum,
        "batch_order_fingerprint": batch_order_fingerprint(consumed),
        "running_mean_abs_mean": float(
            torch.stack([m.running_mean.abs().mean() for m in bn_layers]).mean()),
        "running_var_mean": float(
            torch.stack([m.running_var.mean() for m in bn_layers]).mean()),
    }


def assert_identical_recalibration(records: dict[str, dict[str, Any]]) -> None:
    """Assert B, B' and C consumed the identical recalibration stream (master §9).

    Raises AssertionError naming the divergent runs -- a silent divergence here would give the
    three models different starting points and confound the loss comparison.
    """
    prints = {k: v.get("batch_order_fingerprint") for k, v in records.items()}
    if len({v for v in prints.values() if v is not None}) > 1:
        raise AssertionError(
            f"BatchNorm recalibration streams differ across runs: {prints}. "
            f"B/B'/C must see identical images in identical order (master §9).")
    counts = {k: v.get("n_batches") for k, v in records.items()}
    if len(set(counts.values())) > 1:
        raise AssertionError(f"recalibration batch counts differ across runs: {counts}")
