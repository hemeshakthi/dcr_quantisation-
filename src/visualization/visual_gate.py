"""The visual gate -- a HARD STOP (master §13).

Protocol, in order, and the order is the point:
  1. Pre-select 10 high-curvature test images with seed=42, FROM GROUND TRUTH, BEFORE any
     model output is looked at.
  2. Render four-column contour figures for a single seed of A / B / C.
  3. STOP. Two people, BLIND TO THE NUMBERS, independently judge each panel.
  4. Require >= 6/10 restored to proceed.
  5. On failure: DIAGNOSE. Do not auto-retry with different hyperparameters (master §4).

Selecting the images from ground truth beforehand is what stops the gate from becoming a
formality: choosing panels after seeing the predictions would let the most flattering examples
be picked, and the gate would confirm whatever it was shown.

This module prepares the material and records the verdicts. It never scores the panels itself
-- master §13 requires two human raters, and an automated stand-in would defeat the purpose.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

SELECTION_SEED = 42
N_PANELS = 10
REQUIRED_RESTORED = 6


def preselect_images(curvature_csv: str | Path,
                     n: int = N_PANELS,
                     seed: int = SELECTION_SEED,
                     split: str = "test",
                     bin_name: str = "high") -> list[str]:
    """Pick the panel images from GROUND-TRUTH curvature only, before any inference.

    Write-once: the selection is saved so it cannot be silently re-rolled after the figures
    have been seen.

    Returns: the chosen image_ids.
    """
    raise NotImplementedError


def render_gate_figures(image_ids: list[str],
                        checkpoints: dict[str, str | Path],
                        config: dict[str, Any],
                        output_dir: str | Path,
                        device: str = "cpu") -> list[Path]:
    """Render the four-column figures for the pre-selected images.

    Figures carry NO metric values and NO model-performance ordering -- raters must be blind to
    the numbers (master §13). Columns are labelled with neutral identifiers.
    """
    raise NotImplementedError


def record_verdicts(verdicts: dict[str, dict[str, bool]],
                    output_path: str | Path) -> dict[str, Any]:
    """Record both raters' per-panel judgements and compute the outcome.

    Args:
        verdicts: {rater_id: {image_id: restored_bool}}; exactly two raters (master §13).

    Returns:
        Per-rater counts, agreement, and pass/fail against the >= 6/10 threshold. Also
        reports inter-rater agreement, since a 6/10 pass with poor agreement is weak evidence
        and should be visible as such.
    """
    raise NotImplementedError


def gate_status(record_path: str | Path) -> dict[str, Any]:
    """Read the recorded gate outcome.

    Downstream phases call this and must refuse to proceed while the status is not PASSED --
    the gate is a hard stop, not advisory.
    """
    raise NotImplementedError
