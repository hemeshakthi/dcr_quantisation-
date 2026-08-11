"""Measured model complexity -- params and FLOPs via thop (master §9).

Master §9 is explicit that compression accounting uses MEASURED params and FLOPs, not
theoretical estimates, and that B, B' and C at a given ratio MUST have identical values.

That equality is a CORRECTNESS CHECK, not a reporting nicety. The three models differ only in
loss; if their measured complexity differs at all, the pruning step has diverged and the
comparison is invalid -- the pruned-model analogue of master's "two identical model outputs =
a bug" rule, running in the opposite direction.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _strip_thop_buffers(model: nn.Module) -> None:
    """Remove the `total_ops`/`total_params` buffers thop's hooks leave on every submodule.

    [ENGINEERING, Phase 7 -- A43] thop.profile does NOT clean these up itself (verified: after
    profiling, `named_buffers()` carries 32 extra `total_ops`/`total_params` entries on this
    study's UNet). Left in place they would silently widen the model's `state_dict()` with
    accounting artefacts that have no business in a saved checkpoint.
    """
    for m in model.modules():
        for key in ("total_ops", "total_params"):
            if key in m._buffers:
                del m._buffers[key]


def measure_complexity(model: nn.Module,
                       input_shape: tuple[int, int, int, int] = (1, 1, 1024, 1024),
                       device: str = "cpu") -> dict[str, Any]:
    """Measure params and FLOPs with thop.

    Args:
        input_shape: (B, C, H, W); defaults to the study's 1024x1024 input (A19).

    Returns:
        {"params": int, "flops": int, "input_shape": tuple, "thop_version": str}

    Counts are taken on the ACTUAL module, so a "pruned" model that was only masked with zeros
    rather than physically narrowed would show unchanged params -- which is exactly the failure
    this measurement is meant to catch.

    Profiles `model` DIRECTLY, not a `copy.deepcopy` of it. [ENGINEERING, Phase 7 -- A43,
    superseding the original deepcopy design]: `copy.deepcopy` immediately followed by
    `thop.profile` on the real 31M-parameter frozen architecture at inputs larger than a small
    smoke-test size (measured: reproducible at 256x256, not at 64x64) was found to SEGFAULT the
    interpreter on this host -- a genuine, reproducible native-level crash, not a resource
    fluke (isolated by bisecting deepcopy vs. no-deepcopy crossed with input size; either alone
    is safe, the combination at scale is not). Profiling the model directly and explicitly
    stripping thop's added buffers afterward (`_strip_thop_buffers`) achieves the same
    "caller's module is left clean" guarantee the deepcopy existed for, without the crash --
    verified: `state_dict()` keys are byte-identical before and after.
    """
    import thop

    was_training = model.training
    model.eval()
    dummy = torch.zeros(*input_shape, dtype=torch.float32)
    try:
        with torch.no_grad():
            flops, params = thop.profile(model, inputs=(dummy,), verbose=False)
    finally:
        _strip_thop_buffers(model)
        if was_training:
            model.train()
    return {
        "params": int(params),
        "flops": int(flops),
        "input_shape": tuple(input_shape),
        "thop_version": getattr(thop, "__version__", "unknown"),
    }


def compression_summary(baseline: dict[str, Any], pruned: dict[str, Any]) -> dict[str, Any]:
    """Ratios of pruned to baseline complexity.

    Returns params/FLOPs reduction factors and percentages for the paper's compression table.
    """
    def ratio_and_pct(base: int, small: int) -> tuple[float, float]:
        if base <= 0:
            return float("nan"), float("nan")
        r = small / base
        return r, (1.0 - r) * 100.0

    p_ratio, p_pct = ratio_and_pct(baseline["params"], pruned["params"])
    f_ratio, f_pct = ratio_and_pct(baseline["flops"], pruned["flops"])
    return {
        "params_baseline": baseline["params"], "params_pruned": pruned["params"],
        "params_ratio": p_ratio, "params_reduction_pct": p_pct,
        "flops_baseline": baseline["flops"], "flops_pruned": pruned["flops"],
        "flops_ratio": f_ratio, "flops_reduction_pct": f_pct,
    }


def assert_identical_complexity(records: dict[str, dict[str, Any]],
                                tolerance: int = 0) -> None:
    """Assert every model in `records` has identical measured params and FLOPs.

    Args:
        records: {run_name: complexity_record} for B, B' and C at one ratio.
        tolerance: allowed absolute difference; FROZEN at 0 -- these should be bit-identical
            because all three are built from the same channel mask.

    Raises:
        AssertionError: naming the divergent runs and their counts.
    """
    if len(records) < 2:
        return
    names = list(records)
    ref = records[names[0]]
    bad = []
    for name in names[1:]:
        r = records[name]
        if (abs(r["params"] - ref["params"]) > tolerance
                or abs(r["flops"] - ref["flops"]) > tolerance):
            bad.append((name, r["params"], r["flops"]))
    if bad:
        raise AssertionError(
            f"measured complexity differs across models that share one channel mask "
            f"(master §9): reference {names[0]}={ref['params']}p/{ref['flops']}f, "
            f"divergent={bad}")
