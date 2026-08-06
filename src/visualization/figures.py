"""Paper figures (master §14 style rules: black and white only).

All figures are generated FROM the per-image metrics table and real checkpoints. Master §1
forbids placeholders and fabricated graphs, and Golden Rule 3 requires every number to trace
to a live computation -- so no figure here accepts hardcoded values.

Where a figure would show a null or negative result, it shows it (Golden Rule 4).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


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
