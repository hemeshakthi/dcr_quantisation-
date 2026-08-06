"""Configuration loading and composition.

Every value used by the pipeline comes from a YAML in configs/ -- nothing is hardcoded in
code (PROJECT_MASTER.md Phase 3 requirement). A run config is composed as:

    training_shared.yaml  +  pruning.yaml  +  model_<x>.yaml   ->  RunConfig

with the model config taking precedence, so B/B'/C provably share identical optimiser, LR,
seeds and augmentation (master §10) while differing ONLY in the loss and pruning fields.

Each YAML annotates its values with a source label ([PAPER]/[OFFICIAL]/[LITERATURE]/
[ENGINEERING]/[ASSUMPTION]); `source_labels` blocks carry those into the composed config so
they can be written into the experiment log and never drift from the code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file into a dict.

    Args:
        path: absolute path, or a name relative to configs/.

    Returns:
        The parsed mapping.

    Raises:
        FileNotFoundError: if the file does not exist. Never returns a default -- a missing
            config is a bug, not something to paper over.
    """
    raise NotImplementedError


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict.

    Mappings merge key-wise; every other type (including lists) is replaced wholesale, so a
    model config that sets `loss.terms` fully replaces the shared default rather than
    appending to it.
    """
    raise NotImplementedError


def compose_run_config(model_config: str | Path,
                       shared: str | Path = "training_shared.yaml",
                       pruning: str | Path | None = "pruning.yaml") -> dict[str, Any]:
    """Compose the full config for one run.

    Precedence (lowest to highest): shared -> pruning -> model.

    Args:
        model_config: one of configs/model_*.yaml.
        shared: shared training config.
        pruning: pruning config; may be None for uncompressed runs, though the frozen
            protocol (master §9) still records `pruning.enabled: false` explicitly rather
            than omitting the section.

    Returns:
        The merged config dict.
    """
    raise NotImplementedError


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Check a composed config for internal consistency.

    Enforces the FROZEN invariants that later phases depend on, including:
      * `model.out_channels == 1` and no output activation (master §7)
      * `training.mixed_precision is False` for final runs (master §10)
      * `pruning.ratio` in the frozen set {0.5, 0.7} when pruning is enabled (master §9)
      * B/B'/C configs agreeing on optimiser/LR/seeds (checked pairwise by the caller)
      * every non-trivial value carrying a source label

    Returns:
        A list of human-readable problems; empty means valid. Never raises -- the caller
        decides whether to treat problems as fatal.
    """
    raise NotImplementedError


def resolve_source_labels(cfg: dict[str, Any]) -> dict[str, str]:
    """Flatten the config's `source_labels` blocks into {dotted.key: label}.

    Used by the experiment log so every recorded hyperparameter carries the provenance label
    it had in the YAML (master §0, Golden Rule 7).
    """
    raise NotImplementedError
