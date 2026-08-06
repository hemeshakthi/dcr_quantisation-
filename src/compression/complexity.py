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
    """
    raise NotImplementedError


def compression_summary(baseline: dict[str, Any], pruned: dict[str, Any]) -> dict[str, Any]:
    """Ratios of pruned to baseline complexity.

    Returns params/FLOPs reduction factors and percentages for the paper's compression table.
    """
    raise NotImplementedError


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
    raise NotImplementedError
