"""Test-set inference and per-image metric computation (master §6).

Runs a trained checkpoint over the test split, writes one per-image metric row per image, and
runs the six failure detectors (§12).

Two safeguards that are not optional here:
  * every metric row records the checkpoint hash, so any number can be traced to the weights
    that produced it (master §14);
  * `detect_identical_outputs` compares raw predictions across models on a probe image.
    Master Golden Rule 2 makes two identical model outputs a BUG; a previous attempt at this
    project failed exactly that way, so it is checked automatically rather than assumed.

Post-processing is deliberately minimal -- a 0.5 threshold on the sigmoid, no morphological
cleanup ([ASSUMPTION] A07) -- and, whatever it is, it is IDENTICAL for all five models. A
post-processing step applied unevenly would be indistinguishable from a real effect.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_model_for_eval(checkpoint_path: str | Path, config: dict[str, Any],
                        device: str = "cpu") -> Any:
    """Rebuild the network (narrowed to its channel mask if pruned) and load its weights.

    Puts the model in eval() mode, which for the pruned models means BatchNorm uses the
    recalibrated running statistics rather than batch statistics.
    """
    raise NotImplementedError


def predict_image(model: Any, image: Any, device: str = "cpu",
                  threshold: float = 0.5) -> dict[str, Any]:
    """Run inference on one image.

    Returns:
        {"logits", "probs", "binary"} -- the raw logits are kept so the UI can show
        confidence and so identical-output detection can compare pre-threshold values, where
        two genuinely different models are far less likely to coincide by chance.
    """
    raise NotImplementedError


def evaluate_run(checkpoint_path: str | Path,
                 config: dict[str, Any],
                 run_id: str,
                 split: str = "test",
                 device: str = "cpu",
                 output_dir: str | Path | None = None) -> dict[str, Any]:
    """Evaluate one run over a split and write its per-image metrics.

    Returns aggregate summaries per curvature bin -- but the per-image CSV, not this summary,
    is what the statistics consume (master §6 requires pairing by image).
    """
    raise NotImplementedError


def detect_identical_outputs(predictions: dict[str, Any],
                             atol: float = 0.0) -> list[str]:
    """Compare model predictions pairwise on the same input (Golden Rule 2).

    Args:
        predictions: {model_name: probs array} for one probe image.
        atol: 0.0 means byte-identical. Non-zero tolerance flags suspiciously-similar models
            too, which catches the subtler failure where two runs loaded the same checkpoint.

    Returns:
        Descriptions of any offending pairs; empty means all models genuinely differ.
    """
    raise NotImplementedError


def aggregate_by_bin(per_image_csv: str | Path) -> Any:
    """Aggregate per-image metrics by curvature bin, model and ratio, for the paper's tables.

    Mid-bin rows are marked descriptive-only (A29) so a reader cannot mistake them for
    confirmatory evidence.
    """
    raise NotImplementedError
