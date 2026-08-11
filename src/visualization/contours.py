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

import cv2
import numpy as np

# Distinct line styles, since colour is not available (master §14). Names match
# frontend/app.js's DASH_ARRAY exactly (cross-checked by audit/verify_frontend.py) so a model's
# line style can never drift between the server-rendered raster path and the client-rendered
# vector path.
LINE_STYLES = {
    "ground_truth": "solid",
    "vanilla": "dotted",
    "A": "dashed",
    "B": "dashdot",
    "B_prime": "longdash",
    "C": "solid_thick",
}

# [ASSUMPTION A38] Concrete raster parameters for each named style: (thickness, dash_on,
# dash_off). Not specified by master beyond "distinct styles, no color"; chosen so every style
# remains visually distinguishable from every other at 1024px, verified by
# audit/verify_backend.py rendering all six onto one test image.
_STYLE_PARAMS = {
    "solid": (1, None, None),
    "solid_thick": (3, None, None),
    "dotted": (1, 2, 4),
    "dashed": (1, 10, 6),
    "dashdot": (1, 10, 4),   # approximated as a dash; a true dash-dot needs a 4-segment cycle,
                            # not worth the added complexity for a B&W thickness/period cue
    "longdash": (2, 18, 8),
}


def _to_binary(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    while m.ndim > 2:
        m = m[0] if m.shape[0] < m.shape[-1] else m[..., 0]
    return (m > 0).astype(np.uint8)


def to_display_u8(image: np.ndarray) -> np.ndarray:
    """Public min/max-normalised uint8 conversion, shared with backend/ so the served
    'original' image and the one baked into every overlay/error-map PNG use one definition."""
    return _to_gray_u8(image)


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    while img.ndim > 2:
        img = img[0] if img.shape[0] < img.shape[-1] else img[..., 0]
    if img.dtype == np.uint8:
        return img
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros(img.shape, np.uint8)
    return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _extract_polylines(binary: np.ndarray) -> list[np.ndarray]:
    """cv2.findContours -> a list of (N,2) int32 point arrays, one per connected boundary."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2) for c in contours if len(c) >= 3]


def _draw_dashed_polyline(canvas: np.ndarray, pts: np.ndarray, color: int,
                          thickness: int, dash_on: int | None, dash_off: int | None) -> None:
    """Draw one closed polyline, solid or dashed, by walking its perimeter in arc-length."""
    n = len(pts)
    if n < 2:
        return
    if dash_on is None:
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], isClosed=True,
                      color=color, thickness=thickness, lineType=cv2.LINE_AA)
        return
    period = dash_on + dash_off
    dist = 0.0
    for i in range(n):
        p0, p1 = pts[i], pts[(i + 1) % n]
        seg_len = float(np.hypot(*(p1 - p0)))
        if seg_len == 0:
            continue
        steps = max(1, int(seg_len))
        for s in range(steps):
            t0, t1 = s / steps, (s + 1) / steps
            if (dist % period) < dash_on:
                a = (p0 + (p1 - p0) * t0).astype(int)
                b = (p0 + (p1 - p0) * t1).astype(int)
                cv2.line(canvas, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)
            dist += seg_len / steps


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
    canvas = _to_gray_u8(image).copy()
    binary = _to_binary(mask)
    polys = _extract_polylines(binary)
    style_thickness, dash_on, dash_off = _STYLE_PARAMS.get(style, _STYLE_PARAMS["solid"])
    t = max(thickness, style_thickness)

    if halo:
        for pts in polys:
            _draw_dashed_polyline(canvas, pts, color=0, thickness=t + 2,
                                  dash_on=dash_on, dash_off=dash_off)
    for pts in polys:
        _draw_dashed_polyline(canvas, pts, color=255, thickness=t,
                              dash_on=dash_on, dash_off=dash_off)
    return canvas


def overlay_multiple_contours(image: np.ndarray,
                              masks: dict[str, np.ndarray],
                              styles: dict[str, str] | None = None) -> np.ndarray:
    """Overlay several models' contours on one image, distinguished by line style alone."""
    styles = styles or LINE_STYLES
    canvas = _to_gray_u8(image).copy()
    for name, mask in masks.items():
        style = styles.get(name, "solid")
        binary = _to_binary(mask)
        polys = _extract_polylines(binary)
        style_thickness, dash_on, dash_off = _STYLE_PARAMS.get(style, _STYLE_PARAMS["solid"])
        for pts in polys:
            _draw_dashed_polyline(canvas, pts, color=0, thickness=style_thickness + 2,
                                  dash_on=dash_on, dash_off=dash_off)
        for pts in polys:
            _draw_dashed_polyline(canvas, pts, color=255, thickness=style_thickness,
                                  dash_on=dash_on, dash_off=dash_off)
    return canvas


def error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """False positives as SOLID fill, false negatives as HATCHED fill (master §14).

    The two error types must remain visually distinguishable without colour, because they mean
    opposite things: FP is over-segmentation, FN is missed structure.

    Returns: (H, W) uint8, white(255)=background, with FP painted solid black and FN painted as
    a black diagonal hatch (only pixels where (x+y) % 8 < 3 are painted -- matching
    frontend/app.js's client-side hatch exactly, so a server-rendered and a client-rendered
    error map for the same input are pixel-equivalent).
    """
    p, t = _to_binary(pred).astype(bool), _to_binary(target).astype(bool)
    fp = p & ~t
    fn = ~p & t
    h, w = p.shape
    yy, xx = np.mgrid[0:h, 0:w]
    hatch = ((xx + yy) % 8) < 3

    canvas = np.full((h, w), 255, np.uint8)
    canvas[fp] = 0
    canvas[fn & hatch] = 0
    return canvas


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
    gray = _to_gray_u8(image)

    def maybe_crop(a: np.ndarray) -> np.ndarray:
        if crop is None:
            return a
        y, x, ch, cw = crop
        return a[y:y + ch, x:x + cw]

    panels = [maybe_crop(gray)]
    panels.append(maybe_crop(draw_contour(gray, ground_truth, style="solid")))
    for name in order:
        if name not in predictions:
            continue
        panels.append(maybe_crop(draw_contour(gray, predictions[name],
                                              style=LINE_STYLES.get(name, "solid"))))
    sep = np.full((panels[0].shape[0], 4), 128, np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.hstack([out, sep, p])
    return out
