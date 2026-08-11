"""L1-norm structured channel pruning (master §9) [ENGINEERING -- the study's new contribution].

Ratios 0.5 (primary) and 0.7 (robustness). Structured, not unstructured: whole output channels
are removed so the network genuinely shrinks in measured params and FLOPs, rather than
acquiring zeros that a dense kernel still multiplies.

BatchNorm is pruned in lockstep with its Conv (master §9). A BatchNorm's affine and running
statistics are per-channel, so dropping conv output channel k without dropping BN channel k
would silently misalign every downstream channel -- a bug that trains without error and
produces garbage.

Theoretical framing (master §2): L1 magnitude cannot know which channels carry boundary
detail, so pruning is expected to damage high-curvature boundary geometry while sparing Dice.
That expectation is what P1/P2 test; this module must NOT be "improved" with a
boundary-aware criterion, which is MedCore's contribution and explicitly complementary
(master §3), not ours.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def channel_l1_norms(conv: nn.Conv2d) -> torch.Tensor:
    """Per-output-channel L1 norm of a conv's weights. Returns shape (out_channels,)."""
    raise NotImplementedError


def select_kept_channels(norms: torch.Tensor, ratio: float,
                         min_keep: int = 1) -> list[int]:
    """Choose which output channels survive.

    Args:
        norms: per-channel L1 norms.
        ratio: fraction to REMOVE (0.5 removes half).
        min_keep: never reduce a layer below this many channels; a layer pruned to zero would
            sever the network.

    Returns:
        Sorted kept indices. Ties are broken by ascending index so the selection is fully
        deterministic -- master §10 requires determinism, and an arbitrary tie-break would make
        the mask irreproducible across runs and platforms.
    """
    raise NotImplementedError


def build_channel_mask(model: nn.Module, ratio: float,
                       min_keep: int = 1) -> dict[str, list[int]]:
    """Compute the kept-channel mask for every prunable conv in the model.

    Skips the 1x1 output head (pruning it would change the number of classes). Skip
    connections constrain which channels may be dropped -- an encoder stage's output is
    concatenated into the decoder, so the two must agree on the surviving index set; this
    function is responsible for keeping those consistent rather than pruning each layer
    independently.
    """
    raise NotImplementedError


def apply_channel_mask(model: nn.Module, mask: dict[str, list[int]]) -> nn.Module:
    """Physically narrow the model to the masked channels.

    Slices Conv2d weight/bias on the output axis, the following Conv2d on its input axis, and
    the interposed BatchNorm's weight/bias/running_mean/running_var identically.

    Returns a NEW module; the input model is left untouched so Model A's weights stay
    available for the other ratio.
    """
    raise NotImplementedError


def prune_model(model: nn.Module, ratio: float, mask_path: str | None = None,
                min_keep: int = 1) -> tuple[nn.Module, dict[str, Any]]:
    """Prune once and record the mask (master §9).

    If `mask_path` exists, the mask is LOADED rather than recomputed -- this is the path B, B'
    and C take, and it is what guarantees all three share one architecture.

    Returns:
        (pruned_model, record) where record carries the mask, its hash, the ratio and the
        measured params/FLOPs before and after.
    """
    raise NotImplementedError
