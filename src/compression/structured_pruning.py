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

GRANULARITY [ASSUMPTION A37]: master §9 fixes the CRITERION (L1 norm) and the PROTOCOL (prune
once, recalibrate BN, fine-tune) but not the granularity of a "channel". `UNet.prunable_conv_names()`
(master §7/Phase 4) lists every non-1x1 Conv2d individually -- both convs of every DoubleConv.
This module prunes at the DoubleConv BLOCK level: since every block's constructor defaults
`mid_channels = out_channels` (true for all five model configs, none override it), a block's
first and second conv already share one width by construction, so this module selects ONE
kept-index set per block (via the SECOND conv's L1 norms -- the one whose output actually
propagates to the next stage / a skip connection) and applies it to BOTH convs. This is
documented, not silent: `build_channel_mask` records the same list under both conv names, so
`prunable_conv_names()`'s per-conv listing stays accurate for `mask_io.py`/`compression_accounting.py`,
while the actual freedom of choice is one selection per block, matching the network's real
degrees of freedom under the mid==out convention.
"""
from __future__ import annotations

import json
from typing import Any

import torch
import torch.nn as nn

# The stage order pruning walks, and how each stage's INPUT is produced. Built once against
# UNet's actual module tree (src/models/unet.py) rather than assumed, so a change to the
# network's naming would surface here as a KeyError rather than a silent mismatch.
_ENCODER_PREFIX = "encoders.{i}.block"
_BOTTLENECK_PREFIX = "bottleneck.block"
_DECODER_PREFIX = "decoders.{i}.conv.block"
_HEAD_NAME = "head"


def channel_l1_norms(conv: nn.Conv2d) -> torch.Tensor:
    """Per-output-channel L1 norm of a conv's weights. Returns shape (out_channels,)."""
    return conv.weight.detach().abs().sum(dim=(1, 2, 3))


def select_kept_channels(norms: torch.Tensor, ratio: float,
                         min_keep: int = 1) -> list[int]:
    """Choose which output channels survive.

    Args:
        norms: per-channel L1 norms.
        ratio: fraction to REMOVE (0.5 removes half).
        min_keep: never reduce a layer below this many channels; a layer pruned to zero would
            sever the network.

    Returns:
        Sorted kept indices. Ties are broken by ascending index (stable sort on `-norm` keeps
        the ORIGINAL index order among equal norms) so the selection is fully deterministic --
        master §10 requires determinism, and an arbitrary tie-break would make the mask
        irreproducible across runs and platforms.
    """
    n = norms.shape[0]
    n_keep = max(int(min_keep), n - int(round(n * ratio)))
    n_keep = min(n_keep, n)
    order = sorted(range(n), key=lambda i: (-float(norms[i]), i))  # stable: ties -> ascending i
    kept = sorted(order[:n_keep])
    return kept


def _module_by_name(model: nn.Module, name: str) -> nn.Module:
    return dict(model.named_modules())[name]


def _block_conv_names(model: nn.Module, block_prefix: str) -> tuple[str, str]:
    """The (conv1, conv2) Conv2d names inside one DoubleConv's `nn.Sequential`.

    DoubleConv.block is `Sequential(conv, BN, ReLU, conv, BN, ReLU)` (src/models/unet.py) --
    indices 0 and 3 are the two Conv2d layers.
    """
    return f"{block_prefix}.0", f"{block_prefix}.3"


def _block_bn_names(block_prefix: str) -> tuple[str, str]:
    return f"{block_prefix}.1", f"{block_prefix}.4"


def _stage_prefixes(model) -> tuple[list[str], str, list[str]]:
    enc = [_ENCODER_PREFIX.format(i=i) for i in range(len(model.encoders))]
    bott = _BOTTLENECK_PREFIX
    dec = [_DECODER_PREFIX.format(i=i) for i in range(len(model.decoders))]
    return enc, bott, dec


def build_channel_mask(model: nn.Module, ratio: float,
                       min_keep: int = 1) -> dict[str, list[int]]:
    """Compute the kept-channel mask for every prunable conv in the model.

    Skips the 1x1 output head (pruning it would change the number of classes). One kept-index
    set is chosen per DoubleConv BLOCK (A37 above) via its second conv's L1 norms, and recorded
    under BOTH that block's conv names -- `prunable_conv_names()` stays the authoritative name
    list, this function just makes the two names agree.
    """
    enc_prefixes, bott_prefix, dec_prefixes = _stage_prefixes(model)
    mask: dict[str, list[int]] = {}
    for prefix in [*enc_prefixes, bott_prefix, *dec_prefixes]:
        c1, c2 = _block_conv_names(model, prefix)
        conv2 = _module_by_name(model, c2)
        norms = channel_l1_norms(conv2)
        kept = select_kept_channels(norms, ratio, min_keep)
        mask[c1] = kept
        mask[c2] = kept
    known = set(model.prunable_conv_names())
    produced = set(mask)
    if produced != known:
        raise RuntimeError(
            f"internal error: pruning mask names {produced ^ known} do not match "
            f"UNet.prunable_conv_names() -- the block-name convention in structured_pruning.py "
            f"has drifted from src/models/unet.py's actual module tree")
    return mask


def apply_channel_mask(model: nn.Module, mask: dict[str, list[int]]) -> nn.Module:
    """Physically narrow the model to the masked channels.

    Builds a NEW, narrower `UNet` (via its own constructor, so every layer's shape -- including
    ConvTranspose2d/BatchNorm/head, none of which appear in `mask` -- is computed correctly from
    the reduced per-stage widths) and transplants weights wherever a clean slice is well-defined:

      * encoder chain and bottleneck: fully sliceable (a strict producer -> consumer chain), so
        every weight is a real slice of Model A's trained weights, not a fresh initialisation.
      * decoder DoubleConvs: input is `concat(upsampled, skip)`. The SKIP half of the input is
        sliced from Model A (the producing encoder block's kept indices are well-defined); the
        UPSAMPLE half has no such producer -- the preceding `ConvTranspose2d`'s output identity
        is not "a subset of Model A's channels" in any well-defined sense once its own INPUT
        width has changed -- so that half of the weight, and the ConvTranspose2d itself, are
        left at the new module's own fresh initialisation. [ENGINEERING, A37] This is why
        master §9's protocol mandates BatchNorm recalibration AND fine-tuning after pruning,
        not a frozen transplant: the freshly-initialised portions are exactly what that
        fine-tuning step exists to adapt.
      * head: input-sliced from the last decoder block's kept indices (well-defined, no
        upsample-mixing ambiguity here).

    Returns a NEW module; the input model is left untouched so Model A's weights stay available
    for the other ratio.
    """
    from src.models.unet import UNet  # local import: avoids a compression<->models cycle

    enc_prefixes, bott_prefix, dec_prefixes = _stage_prefixes(model)

    def kept_width(prefix: str) -> int:
        _, c2 = _block_conv_names(model, prefix)
        return len(mask[c2])

    new_encoder = [kept_width(p) for p in enc_prefixes]
    new_bottleneck = kept_width(bott_prefix)
    new_decoder = [kept_width(p) for p in dec_prefixes]

    pruned = UNet(
        in_channels=model.in_channels,
        out_channels=model.out_channels,
        encoder_channels=new_encoder,
        bottleneck_channels=new_bottleneck,
        decoder_channels=new_decoder,
        bilinear_upsample=model.bilinear_upsample,
    )

    def slice_conv(src: nn.Conv2d, dst: nn.Conv2d,
                   out_idx: list[int] | None, in_idx: list[int] | None,
                   in_offset: int = 0) -> None:
        """Copy src -> dst, output-sliced by out_idx and/or input-sliced by in_idx.

        `in_offset` places a partial (e.g. skip-only) input slice at the right column offset
        inside dst's wider input dimension (used for decoder DoubleConvs, where columns
        [0:up_out] are the fresh-init upsample half and [up_out:] are the transplanted skip
        half).
        """
        w = src.weight.detach()
        if out_idx is not None:
            w = w[out_idx]
        with torch.no_grad():
            if in_idx is not None:
                w_in = w[:, in_idx]
                dst.weight[:, in_offset:in_offset + w_in.shape[1]] = w_in
            else:
                dst.weight[:, in_offset:in_offset + w.shape[1]] = w
            if src.bias is not None and dst.bias is not None:
                b = src.bias.detach()
                dst.bias.copy_(b[out_idx] if out_idx is not None else b)

    def slice_bn(src: nn.BatchNorm2d, dst: nn.BatchNorm2d, idx: list[int]) -> None:
        with torch.no_grad():
            dst.weight.copy_(src.weight.detach()[idx])
            dst.bias.copy_(src.bias.detach()[idx])
            dst.running_mean.copy_(src.running_mean.detach()[idx])
            dst.running_var.copy_(src.running_var.detach()[idx])
            dst.num_batches_tracked.copy_(src.num_batches_tracked)

    # ---- encoder chain: strictly sliceable end to end ----
    prev_kept: list[int] | None = None  # None => stage 0's input is the raw image, unpruned
    for prefix in enc_prefixes:
        c1, c2 = _block_conv_names(model, prefix)
        b1, b2 = _block_bn_names(prefix)
        kept = mask[c2]
        slice_conv(_module_by_name(model, c1), _module_by_name(pruned, c1),
                  out_idx=kept, in_idx=prev_kept)
        slice_bn(_module_by_name(model, b1), _module_by_name(pruned, b1), kept)
        slice_conv(_module_by_name(model, c2), _module_by_name(pruned, c2),
                  out_idx=kept, in_idx=kept)
        slice_bn(_module_by_name(model, b2), _module_by_name(pruned, b2), kept)
        prev_kept = kept

    # ---- bottleneck: consumes the last encoder stage's kept channels ----
    c1, c2 = _block_conv_names(model, bott_prefix)
    b1, b2 = _block_bn_names(bott_prefix)
    bott_kept = mask[c2]
    slice_conv(_module_by_name(model, c1), _module_by_name(pruned, c1),
              out_idx=bott_kept, in_idx=prev_kept)
    slice_bn(_module_by_name(model, b1), _module_by_name(pruned, b1), bott_kept)
    slice_conv(_module_by_name(model, c2), _module_by_name(pruned, c2),
              out_idx=bott_kept, in_idx=bott_kept)
    slice_bn(_module_by_name(model, b2), _module_by_name(pruned, b2), bott_kept)

    # ---- decoder: skip half transplanted, upsample half left at fresh init ----
    encoder_kepts = []  # rebuild the per-encoder-stage kept lists, deepest first
    pk = None
    for prefix in enc_prefixes:
        _, c2 = _block_conv_names(model, prefix)
        encoder_kepts.append(mask[c2])
    skip_kepts_deepest_first = list(reversed(encoder_kepts))

    for dec_prefix, skip_kept in zip(dec_prefixes, skip_kepts_deepest_first):
        c1, c2 = _block_conv_names(model, dec_prefix)
        b1, b2 = _block_bn_names(dec_prefix)
        kept = mask[c2]
        dst_conv1 = _module_by_name(pruned, c1)
        up_width = dst_conv1.in_channels - len(skip_kept)  # the fresh-init upsample portion
        slice_conv(_module_by_name(model, c1), dst_conv1,
                  out_idx=kept, in_idx=skip_kept, in_offset=up_width)
        slice_bn(_module_by_name(model, b1), _module_by_name(pruned, b1), kept)
        slice_conv(_module_by_name(model, c2), _module_by_name(pruned, c2),
                  out_idx=kept, in_idx=kept)
        slice_bn(_module_by_name(model, b2), _module_by_name(pruned, b2), kept)

    # ---- head: input-sliced by the final decoder block's kept indices ----
    last_dec_kept = mask[_block_conv_names(model, dec_prefixes[-1])[1]]
    slice_conv(_module_by_name(model, _HEAD_NAME), _module_by_name(pruned, _HEAD_NAME),
              out_idx=None, in_idx=last_dec_kept)

    return pruned


def prune_model(model: nn.Module, ratio: float, mask_path: str | None = None,
                min_keep: int = 1,
                input_shape: tuple[int, int, int, int] = (1, 1, 1024, 1024)
                ) -> tuple[nn.Module, dict[str, Any]]:
    """Prune once and record the mask (master §9).

    If `mask_path` exists, the mask is LOADED rather than recomputed -- this is the path B, B'
    and C take, and it is what guarantees all three share one architecture.

    Args:
        input_shape: passed through to `measure_complexity`'s before/after calls. Defaults to
            the study's real 1024x1024 input (A19) -- the shape the real Kaggle GPU run
            measures at. [ENGINEERING, Phase 7 -- A43] Overridable because thop-profiling the
            full 31M-parameter frozen architecture at 1024x1024 was found to SEGFAULT on this
            RAM-constrained authoring host (A18) even after removing the deepcopy that caused a
            separate, smaller-scale crash; local verification passes a smaller real shape
            (still 16-divisible) here, while the default stays 1024x1024 for real usage on
            Kaggle's much larger GPU memory.

    Returns:
        (pruned_model, record) where record carries the mask, its hash, the ratio and the
        measured params/FLOPs before and after (at `input_shape`).
    """
    from src.compression.compression_accounting import measure_complexity
    from src.compression.mask_io import (compute_mask_hash, load_channel_mask,
                                         save_channel_mask)

    before = measure_complexity(model, input_shape=input_shape)

    if mask_path and __import__("pathlib").Path(mask_path).exists():
        loaded = load_channel_mask(mask_path)
        mask = {k: v for k, v in loaded["mask"].items()}
        mask_hash = loaded["hash"]
    else:
        mask = build_channel_mask(model, ratio, min_keep)
        mask_hash = compute_mask_hash(mask)
        if mask_path:
            save_channel_mask(mask, mask_path, metadata={"ratio": ratio})

    pruned = apply_channel_mask(model, mask)
    after = measure_complexity(pruned, input_shape=input_shape)

    record = {
        "ratio": ratio,
        "mask_hash": mask_hash,
        "mask_path": str(mask_path) if mask_path else None,
        "params_before": before["params"], "params_after": after["params"],
        "flops_before": before["flops"], "flops_after": after["flops"],
    }
    return pruned, record


class StructuredPruner:
    """OO facade over this module's L1-norm structured-pruning functions (master §9).

    Every method here delegates to the tested module-level functions above -- this class adds
    no new pruning logic of its own, only the object-oriented entry point (`compute_mask`,
    `apply`, `save_pruning_mask`, `prune`) so a caller can hold one configured pruner (criterion,
    tie-break) and reuse it across ratios/models without re-passing those two arguments each call.

    Args:
        criterion: pruning criterion. FROZEN at "l1_norm" (master §9) -- L1 magnitude cannot
            know which channels carry boundary detail, which is the mechanism P1/P2 test; a
            boundary-aware criterion is MedCore's contribution (master §3), not this study's.
        tie_break: FROZEN at "ascending_index" (master §10 determinism) -- ties in
            `select_kept_channels` break to the ORIGINAL index order, never at random.
    """

    def __init__(self, criterion: str = "l1_norm", tie_break: str = "ascending_index") -> None:
        if criterion != "l1_norm":
            raise ValueError(
                f"criterion is FROZEN at 'l1_norm' (master §9); got {criterion!r}. A "
                f"boundary-aware criterion is MedCore's contribution, not this study's.")
        if tie_break != "ascending_index":
            raise ValueError(
                f"tie_break is FROZEN at 'ascending_index' (master §10 determinism); "
                f"got {tie_break!r}.")
        self.criterion = criterion
        self.tie_break = tie_break

    def compute_channel_norms(self, conv: nn.Module) -> torch.Tensor:
        """Per-output-channel L1 norm of one Conv2d's weights."""
        return channel_l1_norms(conv)

    def compute_mask(self, model: nn.Module, ratio: float,
                     min_keep: int = 1) -> dict[str, list[int]]:
        """The kept-channel mask for every prunable block in `model` at `ratio`."""
        return build_channel_mask(model, ratio, min_keep)

    def apply(self, model: nn.Module, mask: dict[str, list[int]]) -> nn.Module:
        """Physically narrow `model` to `mask`, transplanting weights (see `apply_channel_mask`)."""
        return apply_channel_mask(model, mask)

    def save_pruning_mask(self, mask: dict[str, list[int]], path: str,
                          metadata: dict[str, Any] | None = None) -> str:
        """Write `mask` as per-layer kept-channel-index JSON (write-once) and return its hash."""
        from src.compression.mask_io import save_channel_mask
        return save_channel_mask(mask, path, metadata)

    def prune(self, model: nn.Module, ratio: float, mask_path: str | None = None,
             min_keep: int = 1,
             input_shape: tuple[int, int, int, int] = (1, 1, 1024, 1024)
             ) -> tuple[nn.Module, dict[str, Any]]:
        """Prune once and record the mask -- loads `mask_path` instead of recomputing if it
        already exists, which is the path B, B' and C take (master §9: prune once, load same).
        `input_shape` is forwarded to the params/FLOPs measurement -- see `prune_model`."""
        return prune_model(model, ratio, mask_path, min_keep, input_shape)
