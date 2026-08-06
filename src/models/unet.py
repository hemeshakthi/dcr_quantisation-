"""Plain U-Net -- the single FROZEN backbone for the whole study (master §7).

Architecture (master §7, channel widths [ASSUMPTION] A08):
    input      (B, 1, H, W)
    encoder    64 -> 128 -> 256 -> 512        (4 stages, each 2x(Conv3x3+BN+ReLU) then MaxPool2)
    bottleneck 1024
    decoder    512 -> 256 -> 128 -> 64        (4 stages, upsample + skip concat + 2x conv)
    head       Conv2d(64, 1, kernel_size=1)   NO activation
    output     (B, 1, H, W) raw logits

Master §7 writes the encoder as [64,128,256,512,1024]; the trailing 1024 is the bottleneck,
which is why the decoder is [512,256,128,64]. Four pooling stages means inputs must be
divisible by 16 -- satisfied by the 1024x1024 preprocessing (A19).

WHY PLAIN U-NET, AND WHY IT MAY NOT BE SWAPPED (master §1, §7): skip connections are the
mechanism under study. Attention U-Net's learned gating would adapt to pruning, and UNet++'s
dense skips would give redundant boundary pathways that artificially protect Model B and
shrink the B-C gap. Both are confounds, and both are explicitly out of scope.

The head emits raw logits with no activation so the losses (master §8) can apply their own
numerically-stable sigmoid.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2, the standard U-Net block.

    BatchNorm is required by the frozen pruning protocol: master §9 prunes BatchNorm in
    lockstep with its Conv and then recalibrates it after pruning.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UNet(nn.Module):
    """Plain U-Net producing single-channel logits.

    Args:
        in_channels: input channels; 1 for these greyscale microscopy images.
        out_channels: output channels; FROZEN at 1 (binary segmentation, master §7).
        encoder_channels: per-stage encoder widths, shallowest first.
        bottleneck_channels: width at the lowest resolution.
        decoder_channels: per-stage decoder widths, deepest first. Must mirror the encoder.
        bilinear_upsample: if True use bilinear upsampling + conv, else ConvTranspose2d.

    All widths are constructor arguments rather than constants so that
    `src.compression` can rebuild a narrowed network from a saved kept-channel mask
    (master §9) without any code change.
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 encoder_channels: Sequence[int] = (64, 128, 256, 512),
                 bottleneck_channels: int = 1024,
                 decoder_channels: Sequence[int] = (512, 256, 128, 64),
                 bilinear_upsample: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channels = tuple(encoder_channels)
        self.bottleneck_channels = bottleneck_channels
        self.decoder_channels = tuple(decoder_channels)
        self.bilinear_upsample = bilinear_upsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (B, in_channels, H, W). Returns: (B, 1, H, W) raw logits."""
        raise NotImplementedError

    def prunable_conv_names(self) -> list[str]:
        """Names of Conv2d layers eligible for structured output-channel pruning.

        Excludes the 1x1 output head: pruning it would change the number of output classes
        rather than the network's capacity.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg: dict) -> "UNet":
        """Build from the `model` block of a composed run config (see src/config.py)."""
        raise NotImplementedError

    @classmethod
    def from_channel_mask(cls, cfg: dict, mask: dict) -> "UNet":
        """Build a NARROWED network matching a saved kept-channel mask (master §9).

        B, B' and C all call this with the SAME mask for a given ratio, which is what
        guarantees they have identical parameter and FLOP counts.
        """
        raise NotImplementedError
