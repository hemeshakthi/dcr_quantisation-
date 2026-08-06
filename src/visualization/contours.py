"""Contour and error-map rendering -- BLACK AND WHITE ONLY (master §14).

Master §14 is strict, and for a reason: the paper and the prototype must read correctly in
greyscale print and for colour-blind readers. So models are distinguished by LINE STYLE, never
by colour, and error maps by FILL PATTERN (solid = false positive, hatched = false negative).

Rendering here is intentionally low-level (uint8 arrays via cv2) rather than matplotlib
colormaps: the host's commit limit equals physical RAM (A18) and matplotlib's colormap path
expands images to float64 RGBA, which was measured to fail at 1024x1024 during Phase 2.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Distinct line styles, since colour is not available (master §14).
LINE_STYLES = {
    "ground_truth": "solid",
    "vanilla": "dotted",
    "A": "dashed",
    "B": "dashdot",
    "B_prime": "longdash",
    "C": "solid_thick",
}


def draw_contour(image: np.ndarray, mask: np.ndarray,
                 style: str = "solid", thickness: int = 1,
                 halo: bool = True) -> np.ndarray:
    """Draw a mask's boundary onto a greyscale image.

    Args:
        halo: draw a black outline around the white contour. Necessary here rather than
            decorative -- these bright-field cells have a bright core and a dark rim
            (Phase 2 radial profile), so a plain white line vanishes against the core and a
            plain black line vanishes against the rim.

    Returns: (H, W) uint8.
    """
    raise NotImplementedError


def overlay_multiple_contours(image: np.ndarray,
                              masks: dict[str, np.ndarray],
                              styles: dict[str, str] | None = None) -> np.ndarray:
    """Overlay several models' contours on one image, distinguished by line style alone."""
    raise NotImplementedError


def error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """False positives as SOLID fill, false negatives as HATCHED fill (master §14).

    The two error types must remain visually distinguishable without colour, because they mean
    opposite things: FP is over-segmentation, FN is missed structure.
    """
    raise NotImplementedError


def four_column_figure(image: np.ndarray,
                       ground_truth: np.ndarray,
                       predictions: dict[str, np.ndarray],
                       order: Sequence[str] = ("A", "B", "C"),
                       crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """The visual-gate panel: [image | GT | A | B | C] (master §13).

    Args:
        crop: (y, x, h, w) to zoom on a high-curvature region -- at 1024x1024 with a ~10 px
            median cell radius, a whole-image view is far too small to judge boundary geometry.
    """
    raise NotImplementedError
