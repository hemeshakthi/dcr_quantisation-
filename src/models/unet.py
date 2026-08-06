"""Plain U-Net -- the single FROZEN backbone for the whole study (master §7).

ARCHITECTURE (master §7)
------------------------
    input       (B, 1, H, W)
    encoder     64 -> 128 -> 256 -> 512      each: DoubleConv, keep skip, MaxPool2
    bottleneck  1024                          DoubleConv
    decoder     512 -> 256 -> 128 -> 64      each: up, concat skip, DoubleConv
    head        Conv2d(64, 1, kernel_size=1)  NO activation
    output      (B, 1, H, W) raw logits

Master §7 writes the encoder as [64,128,256,512,1024] and the decoder as [512,256,128,64].
Those two are only mutually consistent if the trailing 1024 IS the bottleneck: four decoder
stages mirror four encoder stages, so there are four pooling steps, not five. This class
therefore accepts EITHER spelling -- pass `encoder_channels=(64,128,256,512,1024)` exactly as
master §7 writes it and the last element is taken as the bottleneck, or pass
`encoder_channels=(64,128,256,512)` with `bottleneck_channels=1024`. Both produce the identical
network; see `_normalise_widths`.

Four pooling stages means H and W must be divisible by 16. The 1024x1024 preprocessing (A19)
satisfies this; a non-conforming input raises immediately rather than failing obscurely deep in
the decoder.

[ASSUMPTION A08] The channel widths are the standard U-Net configuration. The dCR paper's body
is paywalled (docs/source_review.md), so whether it used non-standard widths could not be
recovered. Its own cited architecture lineage (Falk 2019, ref 36) uses this configuration, and
deviating from convention without evidence would be a larger assumption than following it.
CONFIRM AND RE-LABEL IF THE PAPER BODY SPECIFIES OTHERWISE.

WHY PLAIN U-NET -- AND WHY IT MAY NOT BE SWAPPED (master §7, to be stated in the paper)
--------------------------------------------------------------------------------------
Skip connections are the mechanism under study. Master §2's proposed mechanism is that
structured pruning reduces skip-connection bandwidth -- the pathways carrying high-frequency
boundary detail from encoder to decoder -- and that dCR recovers the resulting damage. The
backbone must therefore leave those pathways plain and unmodified:

  * Attention U-Net is excluded because its learned gating ADAPTS to pruning. Attention weights
    would re-balance after channels are removed, partially compensating for the damage the study
    is trying to measure. Any observed B-to-C difference would then confound "dCR recovered the
    boundary" with "attention re-weighted around the missing channels".

  * UNet++ is excluded because its dense nested skips provide REDUNDANT boundary pathways. That
    redundancy would artificially protect Model B -- the compressed baseline whose degradation is
    the entire premise of P1/P2 -- and shrink the B-to-C gap the study exists to detect. A
    smaller measured effect would reflect the architecture, not the loss.

  * Transformers, pretrained encoders and student-teacher schemes are all out of scope
    (master §1). Pretrained weights would additionally break the "identical initialisation
    except model_init_seed" invariant that makes the three seeds comparable.

A single fixed backbone means architecture is NEVER a variable: Vanilla, A, B, B' and C differ
only in loss and in whether they were pruned. This one class serves all five.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Four MaxPool2 stages, so spatial dimensions must be divisible by 2**4.
DOWNSAMPLE_FACTOR = 16


@contextmanager
def _scoped_rng(seed: int | None):
    """Seed torch's RNG for the enclosed block, then restore the caller's stream exactly.

    A no-op when `seed` is None. Covers the CUDA generators too, since a model may be built
    after CUDA has been initialised and those streams must be left equally undisturbed.
    """
    if seed is None:
        yield
        return
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2, the standard U-Net block.

    BatchNorm is required by the frozen pruning protocol: master §9 prunes each BatchNorm in
    lockstep with its Conv and recalibrates it afterwards.

    `mid_channels` is separate from `out_channels` so a pruned network can narrow the block's
    internal width independently of the width it hands to the skip connection -- which is
    exactly what per-layer channel masks produce.

    Convolutions are bias-free: the following BatchNorm has its own affine shift, so a conv bias
    would be redundant and immediately cancelled by BN's mean subtraction.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 mid_channels: int | None = None) -> None:
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """One decoder stage: upsample, concatenate the skip, then DoubleConv.

    Args:
        in_channels: channels arriving from the layer below.
        skip_channels: channels of the encoder skip being concatenated.
        out_channels: channels this stage emits.
        bilinear: bilinear upsampling + 1x1 channel projection instead of ConvTranspose2d.

    The DoubleConv's input width is computed from the ACTUAL upsampled and skip widths rather
    than assumed symmetric, so a pruned network whose encoder and decoder were narrowed by
    different amounts still wires up correctly.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 bilinear: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
            up_out = out_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            up_out = out_channels

        self.conv = DoubleConv(up_out + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against an odd spatial size somewhere in the chain. With inputs divisible by 16
        # this is a no-op; it prevents a silent shape error if that ever stops holding.
        if x.shape[-2:] != skip.shape[-2:]:
            dy = skip.shape[-2] - x.shape[-2]
            dx = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Plain U-Net producing single-channel logits. One class for all five model variants.

    Args:
        in_channels: 1 for these greyscale bright-field images.
        out_channels: FROZEN at 1 -- binary segmentation (master §7).
        encoder_channels: per-stage encoder widths, shallowest first. Master §7's literal
            5-element form (…, 1024) is accepted, with the last element read as the bottleneck.
        bottleneck_channels: width at the lowest resolution. Ignored if `encoder_channels`
            already carries it as its last element.
        decoder_channels: per-stage decoder widths, deepest first. Must mirror the encoder.
        bilinear_upsample: bilinear + 1x1 conv instead of ConvTranspose2d.
        init_seed: seed applied to weight initialisation. Master §10 varies ONLY this across
            the three seeds, while the augmentation seed stays shared across B/B'/C.

    Every width is a constructor argument rather than a module-level constant, so
    `src.compression` can rebuild a narrowed network from a saved channel mask (master §9)
    without touching this file.
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 encoder_channels: Sequence[int] = (64, 128, 256, 512),
                 bottleneck_channels: int = 1024,
                 decoder_channels: Sequence[int] = (512, 256, 128, 64),
                 bilinear_upsample: bool = False,
                 init_seed: int | None = None) -> None:
        super().__init__()

        enc, bottleneck = self._normalise_widths(encoder_channels, bottleneck_channels)
        dec = tuple(int(c) for c in decoder_channels)
        if len(dec) != len(enc):
            raise ValueError(
                f"decoder_channels must mirror encoder_channels: got {len(dec)} decoder "
                f"stages for {len(enc)} encoder stages ({dec} vs {enc})")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.encoder_channels = enc
        self.bottleneck_channels = int(bottleneck)
        self.decoder_channels = dec
        self.bilinear_upsample = bool(bilinear_upsample)
        self.init_seed = init_seed

        # Seeded construction must be HERMETIC. Building nn.Conv2d/nn.BatchNorm2d draws from
        # the global RNG during their own __init__, so seeding only around _initialise_weights
        # would still leave the caller's stream advanced -- and by a different amount per
        # init_seed. Master §10 varies model_init_seed across the three seeds while B/B'/C share
        # one augmentation seed; if constructing the model perturbed the global stream, those
        # three runs would silently diverge in data order too, confounding "different loss" with
        # "different data". So the ENTIRE build is wrapped, and the prior state is restored.
        with _scoped_rng(init_seed):
            # ---- encoder: DoubleConv then MaxPool, keeping the pre-pool output as skip ----
            self.encoders = nn.ModuleList()
            prev = self.in_channels
            for width in enc:
                self.encoders.append(DoubleConv(prev, width))
                prev = width
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # ---- bottleneck ----
            self.bottleneck = DoubleConv(prev, self.bottleneck_channels)

            # ---- decoder: mirror the encoder, consuming skips deepest-first ----
            self.decoders = nn.ModuleList()
            prev = self.bottleneck_channels
            for i, width in enumerate(dec):
                skip_width = enc[len(enc) - 1 - i]
                self.decoders.append(
                    Up(prev, skip_width, width, bilinear=self.bilinear_upsample))
                prev = width

            # ---- head: 1x1 conv, NO activation (master §7) ----
            # The losses apply their own numerically-stable sigmoid, so emitting raw logits
            # here is required, not merely conventional.
            self.head = nn.Conv2d(prev, self.out_channels, kernel_size=1)

            self._initialise_weights()

    # ------------------------------------------------------------------ construction helpers
    @staticmethod
    def _normalise_widths(encoder_channels: Sequence[int],
                          bottleneck_channels: int) -> tuple[tuple[int, ...], int]:
        """Accept master §7's literal encoder list, which includes the bottleneck.

        `(64,128,256,512,1024)` -> encoder `(64,128,256,512)`, bottleneck `1024`
        `(64,128,256,512)`      -> encoder unchanged, bottleneck from the argument
        """
        enc = tuple(int(c) for c in encoder_channels)
        if len(enc) < 2:
            raise ValueError(f"encoder_channels needs at least 2 stages, got {enc}")
        # A trailing element wider than the rest is master §7's bottleneck written inline.
        if len(enc) >= 3 and enc[-1] == bottleneck_channels:
            return enc[:-1], int(bottleneck_channels)
        return enc, int(bottleneck_channels)

    def _initialise_weights(self) -> None:
        """Kaiming/He initialisation for convs, unit-affine for BatchNorm ([ASSUMPTION] A05).

        Called inside `_scoped_rng(init_seed)` from __init__, which owns the seeding and the
        restoration of the caller's RNG stream -- so the three seeds of master §10 differ in
        initialisation and in nothing else.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (B, in_channels, H, W). Returns: (B, out_channels, H, W) logits.

        The returned tensor is UNACTIVATED and will contain negative values; that is the
        contract every loss and every evaluation path depends on.
        """
        self._check_input(x)
        skips: list[torch.Tensor] = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        return self.head(x)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(f"expected a 4-D (B,C,H,W) tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channel(s), got {x.shape[1]}")
        h, w = x.shape[-2:]
        if h % DOWNSAMPLE_FACTOR or w % DOWNSAMPLE_FACTOR:
            raise ValueError(
                f"input H and W must be divisible by {DOWNSAMPLE_FACTOR} "
                f"({len(self.encoders)} pooling stages), got {h}x{w}")

    # -------------------------------------------------------------------------- introspection
    def trace_shapes(self, x: torch.Tensor) -> list[dict]:
        """Record the tensor shape at every stage, including each skip connection.

        Exists so the skip-connection wiring can be VERIFIED rather than assumed -- a decoder
        stage silently concatenating the wrong skip would still train, just worse, which is the
        kind of bug that would be blamed on the loss.
        """
        self._check_input(x)
        trace: list[dict] = [{"stage": "input", "shape": tuple(x.shape)}]
        skips: list[torch.Tensor] = []
        with torch.no_grad():
            for i, enc in enumerate(self.encoders, 1):
                x = enc(x)
                skips.append(x)
                trace.append({"stage": f"encoder{i}", "shape": tuple(x.shape),
                              "skip": True})
                x = self.pool(x)
                trace.append({"stage": f"pool{i}", "shape": tuple(x.shape)})
            x = self.bottleneck(x)
            trace.append({"stage": "bottleneck", "shape": tuple(x.shape)})
            for i, (dec, skip) in enumerate(zip(self.decoders, reversed(skips)), 1):
                x = dec(x, skip)
                trace.append({"stage": f"decoder{i}", "shape": tuple(x.shape),
                              "skip_from": tuple(skip.shape)})
            x = self.head(x)
            trace.append({"stage": "head", "shape": tuple(x.shape)})
        return trace

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Total parameter count."""
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not trainable_only)

    def prunable_conv_names(self) -> list[str]:
        """Names of Conv2d layers eligible for structured OUTPUT-channel pruning (master §9).

        Excludes `head`: pruning it would change the number of output channels, i.e. the task,
        rather than the network's capacity. Also excludes ConvTranspose2d upsamplers, whose
        output width is tied to the decoder stage that follows.

        The names returned here are the canonical keys used by the channel-mask JSON, so this
        method and `src.compression.channel_mask` share one naming convention.
        """
        names = []
        for name, module in self.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if name == "head" or name.endswith(".head"):
                continue
            if module.kernel_size == (1, 1):
                continue  # 1x1 projections in bilinear upsampling are tied to their stage
            names.append(name)
        return names

    def architecture_summary(self) -> dict:
        """Compact description for the experiment log and the paper's methods section."""
        return {
            "architecture": "unet",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "encoder_channels": list(self.encoder_channels),
            "bottleneck_channels": self.bottleneck_channels,
            "decoder_channels": list(self.decoder_channels),
            "bilinear_upsample": self.bilinear_upsample,
            "final_activation": None,
            "n_pooling_stages": len(self.encoders),
            "input_divisibility": DOWNSAMPLE_FACTOR,
            "total_parameters": self.count_parameters(trainable_only=False),
            "channel_widths_source_label": "[ASSUMPTION] A08 -- standard U-Net; "
                                           "confirm if the paper body specifies",
        }

    # ------------------------------------------------------------------------- constructors
    @classmethod
    def from_config(cls, cfg: dict, init_seed: int | None = None) -> "UNet":
        """Build from the `model` block of a composed run config (see src/config.py).

        Reads `model.*` if given the full config, or the block directly.
        """
        model_cfg = cfg.get("model", cfg)
        if model_cfg.get("architecture", "unet") != "unet":
            raise ValueError(
                f"architecture is FROZEN as plain 'unet' (master §7); "
                f"got {model_cfg.get('architecture')!r}")
        if model_cfg.get("final_activation", None) is not None:
            raise ValueError(
                "final_activation must be null: master §7 freezes the head as an "
                "unactivated Conv2d, and the losses apply their own sigmoid")
        return cls(
            in_channels=model_cfg.get("in_channels", 1),
            out_channels=model_cfg.get("out_channels", 1),
            encoder_channels=model_cfg.get("encoder_channels", (64, 128, 256, 512)),
            bottleneck_channels=model_cfg.get("bottleneck_channels", 1024),
            decoder_channels=model_cfg.get("decoder_channels", (512, 256, 128, 64)),
            bilinear_upsample=model_cfg.get("bilinear_upsample", False),
            init_seed=init_seed,
        )

    @classmethod
    def from_channel_mask(cls, cfg: dict, mask: dict) -> "UNet":
        """Build a NARROWED network matching a saved kept-channel mask (master §9).

        Contract: `mask` maps each name from `prunable_conv_names()` to the list of kept output
        channel indices; the resulting per-block widths are the lengths of those lists.

        B, B' and C all call this with the SAME mask for a given ratio, which is what makes
        their parameter and FLOP counts identical.

        NOT YET IMPLEMENTED -- it lands in Phase 5 alongside `src.compression.l1_structured`,
        which produces the mask. Implementing it now would mean inventing the mask's exact
        per-layer semantics before the pruning code that defines them exists.
        """
        raise NotImplementedError(
            "UNet.from_channel_mask lands in Phase 5 with src.compression.l1_structured")
