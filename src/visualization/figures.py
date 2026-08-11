"""Paper figures (master §13 style rules: black and white only).

All figures are generated FROM the per-image metrics table and real checkpoints. Master §1
forbids placeholders and fabricated graphs, and Golden Rule 3 requires every number to trace
to a live computation -- so no figure here accepts hardcoded values.

Where a figure would show a null or negative result, it shows it (Golden Rule 4).

[ENGINEERING, Phase 12] Three functions below (`select_stratified_median_images`,
`contour_crop_panel_figure`, `boundary_band_error_map_figure`,
`curvature_percentile_curve_figure`) are REAL, implemented this phase. The other five
(`metric_by_bin_figure`, `dice_vs_boundary_figure`, `failure_taxonomy_figure`,
`compression_table`, `generate_all_figures`) remain Phase-3 scaffolding for a LATER phase --
Phase 12's own instruction scoped this work to exactly three figure functions, not this whole
module, so they are left as `NotImplementedError` rather than filled in speculatively.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd


def metric_by_bin_figure(per_image_csv: str | Path, metric: str,
                         output_path: str | Path,
                         ratio: float = 0.5,
                         models: Sequence[str] = ("A", "B", "B_prime", "C")) -> Path:
    """Metric by curvature bin and model, with bootstrap CIs.

    The mid bin is drawn but explicitly annotated as descriptive-only (n=12, power 0.337 at
    d=0.5, A29) so a reader cannot mistake it for confirmatory evidence.
    """
    raise NotImplementedError


def dice_vs_boundary_figure(per_image_csv: str | Path,
                            output_path: str | Path) -> Path:
    """The study's headline contrast: Dice preserved while boundary metrics degrade (P1).

    Plots both on a shared model axis, which is what makes the point that Dice alone would
    have hidden the damage -- the 60-second story master §14 asks the prototype to tell.
    """
    raise NotImplementedError


def curvature_response_figure(per_image_csv: str | Path,
                              output_path: str | Path) -> Path:
    """Boundary error against continuous curvature score, per model (P2/P4).

    Uses the continuous score rather than the 3 bins, so it does not inherit the mid bin's
    small n; the bins remain the unit of the confirmatory tests.
    """
    raise NotImplementedError


def failure_taxonomy_figure(failure_counts_csv: str | Path,
                            output_path: str | Path) -> Path:
    """Curvature-related vs non-curvature-related failure counts per model (P6)."""
    raise NotImplementedError


def compression_table(experiment_log_csv: str | Path,
                      output_path: str | Path) -> Path:
    """Measured params/FLOPs per model and ratio (master §9).

    Renders the B/B'/C identity check as a visible column: if the three do not match exactly,
    the table shows it rather than hiding a pruning bug behind an average.
    """
    raise NotImplementedError


def generate_all_figures(per_image_csv: str | Path,
                         experiment_log_csv: str | Path,
                         failure_counts_csv: str | Path,
                         output_dir: str | Path) -> list[Path]:
    """Generate every paper figure from real result tables."""
    raise NotImplementedError


# ==========================================================================================
# Phase 12 -- REAL implementations
# ==========================================================================================
def select_stratified_median_images(curvature_csv: str | Path,
                                    n_per_bin: int = 2,
                                    split: str = "test",
                                    metric_col: str = "curvature_score",
                                    seed: int = 42,
                                    log_path: str | Path | None = None) -> dict[str, Any]:
    """Figure-1 panel selection: stratified-random + median, per curvature bin.

    Per bin: ONE image at the MEDIAN of `metric_col` within that bin -- a representative,
    "typical" case, never a cherry-picked best/worst one -- plus `n_per_bin - 1` additional
    images drawn by SEEDED random sampling from the bin's remaining images (adds coverage
    beyond the single median case). This is deliberately a different rule from the Visual Gate's
    (master §13, `visual_gate.py`): the gate pre-selects the TOP high-curvature images for a
    blind pass/fail judgement; this selects REPRESENTATIVE examples across all three bins for
    the paper's illustrative figure -- different purposes, different rules, not to be confused.

    "Median" is read as MEDIAN BY RANK (the middle element of the bin sorted by `metric_col`,
    `floor(n/2)`-th of a 0-indexed sort) -- a simple, exactly-reproducible definition that needs
    no interpolation choice for even-sized bins.

    Deterministic given `seed`: the exact same `(bin, role, image_id)` selection is produced on
    every call against the same `curvature_csv`. The full record -- rule, seed, and every chosen
    image_id -- is written to `log_path` when given, so the selection is auditable after the
    fact (master §12's own "declared before results exist" principle, applied to figure
    selection: this function is called against curvature bins alone, before any prediction is
    rendered).
    """
    df = pd.read_csv(curvature_csv)
    df = df[df["split"] == split]
    if df.empty:
        raise ValueError(f"no rows for split={split!r} in {curvature_csv}")
    rng = np.random.default_rng(seed)

    selection: list[dict[str, Any]] = []
    for bin_name, sub in df.groupby("bin", observed=True):
        sub_sorted = sub.sort_values(metric_col).reset_index(drop=True)
        median_idx = len(sub_sorted) // 2
        median_row = sub_sorted.iloc[median_idx]
        selection.append({"bin": str(bin_name), "role": "median",
                          "image_id": median_row["image_id"],
                          metric_col: float(median_row[metric_col])})

        remaining = sub_sorted.drop(index=median_idx)
        n_random = max(0, n_per_bin - 1)
        if n_random and len(remaining):
            picks = remaining.sample(n=min(n_random, len(remaining)),
                                     random_state=int(rng.integers(0, 2**31 - 1)))
            for _, r in picks.iterrows():
                selection.append({"bin": str(bin_name), "role": "stratified_random",
                                  "image_id": r["image_id"], metric_col: float(r[metric_col])})

    record = {"rule": "stratified-random + median per curvature bin", "seed": seed,
             "metric_col": metric_col, "n_per_bin": n_per_bin, "split": split,
             "curvature_csv": str(curvature_csv), "selection": selection}
    if log_path is not None:
        lp = Path(log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def contour_crop_panel_figure(image: np.ndarray, ground_truth: np.ndarray,
                              predictions: dict[str, np.ndarray],
                              output_path: str | Path,
                              order: Sequence[str] = ("A", "B", "C"),
                              crop: tuple[int, int, int, int] | None = None) -> Path:
    """One [image | GT | model contours...] crop panel (master §13/§14 style: B&W, line style
    per model, no colour). A thin wrapper around
    `src.visualization.contours.four_column_figure` -- the exact rendering function Phase 13's
    Visual Gate panels and the live backend's contour overlays already use, so a paper figure
    panel can never visually disagree with those. `select_stratified_median_images` supplies
    WHICH images to render; this function only renders one.
    """
    from src.visualization.contours import four_column_figure

    panel = four_column_figure(image, ground_truth, predictions, order=order, crop=crop)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), panel)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed writing {out}")
    return out


def _boundary_band_mask(mask: np.ndarray, width: int) -> np.ndarray:
    """The band around `mask`'s boundary, via `src.losses.boundary_band.extract_boundary_band`
    (master §8's ONE shared band definition) -- reused via a numpy<->torch round trip rather
    than a second, independently-drifting band implementation, so this figure's band matches
    EXACTLY what the dCR/generic boundary losses actually optimise over."""
    import torch

    from src.losses.boundary_band import extract_boundary_band

    t = torch.from_numpy((np.asarray(mask) > 0).astype(np.float32))[None, None]
    band = extract_boundary_band(t, width=int(width))[0, 0].numpy()
    return band > 0.5


def boundary_band_error_map_figure(image: np.ndarray, pred: np.ndarray, target: np.ndarray,
                                   output_path: str | Path, band_width: int = 3) -> Path:
    """Error map RESTRICTED to the boundary band, solid (FP) vs hatched (FN), no colour.

    Differs from `src.visualization.contours.error_map` (which marks errors over the WHOLE
    mask): here only errors that fall INSIDE the boundary band are drawn, which is the region
    the boundary losses (and this whole study) actually care about -- an interior false positive
    far from any boundary is a different failure mode from a boundary-band one, and this figure
    isolates the latter. Uses the SAME solid/hatch convention as `contours.error_map`
    (`(x+y) % 8 < 3` for the FN hatch) so the two are visually consistent.
    """
    from src.visualization.contours import to_display_u8

    band = _boundary_band_mask(target, band_width)
    p, t = np.asarray(pred) > 0, np.asarray(target) > 0
    fp = p & ~t & band
    fn = ~p & t & band

    gray = to_display_u8(image)
    h, w = gray.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    hatch = ((xx + yy) % 8) < 3

    canvas = gray.copy()
    canvas[fp] = 0
    canvas[fn & hatch] = 0

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), canvas)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed writing {out}")
    return out


def curvature_percentile_curve_figure(df: pd.DataFrame, metric_col: str,
                                      output_path: str | Path,
                                      models: Sequence[tuple[str, str]],
                                      window: int = 5,
                                      curvature_col: str = "curvature_score") -> Path:
    """`metric_col` against CONTINUOUS curvature PERCENTILE (not the 3 bins), per model, with
    moving-average smoothing (P2/P4). One line per model, distinguished ONLY by line style
    (`src.visualization.contours.LINE_STYLES` names) and never by colour -- every line is drawn
    in black/grey.

    Args:
        df: per-image rows with at least [model, curvature_col, metric_col].
        models: [(model_id, line_style_name), ...] -- line_style_name must be a key of
            `src.visualization.contours.LINE_STYLES`'s VALUES (e.g. "dotted", "dashed").
        window: moving-average window, in RANK-ordered images (not curvature units) -- the
            x-axis is a percentile RANK, so a rank-based window keeps the smoothing well
            defined even where images are unevenly spaced in raw curvature score.

    Percentile is RANK-based: the i-th image (0-indexed, sorted ascending by `curvature_col`)
    within a model's data is plotted at `100 * (i + 0.5) / n` -- the midpoint convention, so the
    first and last points do not sit exactly on the 0/100 axis lines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL_STYLE = {
        "solid": {"linestyle": "-", "linewidth": 1.4},
        "solid_thick": {"linestyle": "-", "linewidth": 2.8},
        "dotted": {"linestyle": ":", "linewidth": 1.6},
        "dashed": {"linestyle": "--", "linewidth": 1.4},
        "dashdot": {"linestyle": "-.", "linewidth": 1.4},
        "longdash": {"linestyle": (0, (8, 4)), "linewidth": 1.8},
    }
    # Distinct greys as a SECOND cue on top of line style (still zero hue: R==G==B always).
    _GREY_CYCLE = ["#000000", "#404040", "#707070", "#202020", "#585858"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor="white")
    ax.set_facecolor("white")

    for i, (model_id, style_name) in enumerate(models):
        sub = df[df["model"] == model_id].dropna(subset=[curvature_col, metric_col])
        if sub.empty:
            continue
        sub = sub.sort_values(curvature_col).reset_index(drop=True)
        n = len(sub)
        percentile = 100.0 * (np.arange(n) + 0.5) / n
        smoothed = sub[metric_col].rolling(window=min(window, n), center=True,
                                           min_periods=1).mean()
        style = _MPL_STYLE.get(style_name, _MPL_STYLE["solid"])
        ax.plot(percentile, smoothed, color=_GREY_CYCLE[i % len(_GREY_CYCLE)],
               label=f"{model_id}", **style)

    ax.set_xlabel("Curvature percentile (rank-based, within-model)")
    ax.set_ylabel(metric_col)
    ax.set_xlim(0, 100)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
