"""Configuration loading and composition.

Every value used by the pipeline comes from a YAML in configs/ -- nothing is hardcoded in code
(Phase 3 requirement). A run config is composed as:

    training_shared.yaml  +  pruning.yaml  +  model_<x>.yaml   ->  run config

with the model config taking precedence, so B/B'/C provably share optimiser, LR, seeds and
augmentation (master §10) while differing ONLY in the loss and pruning fields.

WHY COMPOSITION MATTERS FOR KAGGLE (see also docs/kaggle_run_contract.md): the Kaggle notebook
must call into this loader rather than restating any hyperparameter in a notebook cell. A value
retyped in a cell can drift from the repo, and `curvature_smooth_sigma` in particular would
silently reintroduce the exact pixelation bug Phase 5 fixed. `config_fingerprint()` is written
into every experiment-log row so a drift is detectable after the fact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file. `path` may be absolute or a name relative to configs/.

    Raises FileNotFoundError rather than returning a default -- a missing config is a bug, not
    something to paper over.
    """
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = CONFIG_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"config not found: {path} (resolved to {p})")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config {p} did not parse to a mapping")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict.

    Mappings merge key-wise; every other type (including lists) is replaced wholesale, so a
    model config that sets a list fully replaces the shared default rather than appending.
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def compose_run_config(model_config: str | Path,
                       shared: str | Path = "training_shared.yaml",
                       pruning: str | Path | None = "pruning.yaml") -> dict[str, Any]:
    """Compose the full config for one run. Precedence: shared -> pruning -> model."""
    cfg = load_yaml(shared)
    if pruning is not None:
        cfg = deep_merge(cfg, load_yaml(pruning))
    cfg = deep_merge(cfg, load_yaml(model_config))
    cfg["_sources"] = {
        "shared": str(shared),
        "pruning": str(pruning) if pruning else None,
        "model": str(model_config),
    }
    return cfg


def config_fingerprint(cfg: dict[str, Any]) -> str:
    """SHA-256 over the composed config, excluding bookkeeping keys.

    Recorded in every experiment-log row so two runs that were supposed to share a
    configuration can be PROVEN to have done so, rather than assumed.
    """
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Check a composed config for the FROZEN invariants later phases depend on.

    Returns a list of human-readable problems; empty means valid. Never raises -- the caller
    decides whether problems are fatal.
    """
    p: list[str] = []
    model = cfg.get("model", {})
    if model.get("architecture") != "unet":
        p.append(f"model.architecture must be 'unet' (master §7 FROZEN), "
                 f"got {model.get('architecture')!r}")
    if model.get("out_channels") != 1:
        p.append(f"model.out_channels must be 1 (master §7), got {model.get('out_channels')!r}")
    if model.get("final_activation", "MISSING") is not None:
        p.append("model.final_activation must be null (master §7): the head emits raw logits")
    if cfg.get("training", {}).get("mixed_precision") is not False:
        p.append("training.mixed_precision must be false for final runs (master §10)")
    det = cfg.get("determinism", {})
    for key, want in (("cudnn_deterministic", True), ("cudnn_benchmark", False),
                      ("use_deterministic_algorithms", True)):
        if det.get(key) is not want:
            p.append(f"determinism.{key} must be {want} (master §10)")
    seeds = cfg.get("seeds", {}).get("model_init_seeds", [])
    if len(seeds) != 3:
        p.append(f"seeds.model_init_seeds must contain exactly 3 seeds (master §10), got {seeds}")
    pr = cfg.get("pruning", {})
    if pr.get("enabled"):
        if pr.get("ratio") not in (0.5, 0.7):
            p.append(f"pruning.ratio must be in the frozen set {{0.5, 0.7}} (master §9), "
                     f"got {pr.get('ratio')!r}")
        if pr.get("reprune") is not False:
            p.append("pruning.reprune must be false: prune ONCE per ratio (master §9)")

    boundary = cfg.get("loss", {}).get("boundary", {}) or {}
    if boundary.get("name") == "dcr":
        # Phase 5 measured that curvature of an UNSMOOTHED binary mask reports pixel staircase
        # rather than shape (corner 0.3166 vs straight edge 0.3333). Training with sigma=0
        # would silently reintroduce that bug, so it is rejected here rather than trusted.
        sig = boundary.get("curvature_smooth_sigma")
        if sig is None:
            p.append("loss.boundary.curvature_smooth_sigma missing for a dCR run (A06): it "
                     "must come from the YAML, never a default or a notebook cell")
        elif float(sig) <= 0:
            p.append(f"loss.boundary.curvature_smooth_sigma must be > 0 for a dCR run "
                     f"(A06 measured sigma=0 fails to discriminate corners), got {sig}")
    return p


def resolve_source_labels(cfg: dict[str, Any]) -> dict[str, str]:
    """Flatten the config's `source_labels` blocks into {dotted.key: label}.

    Used by the experiment log so every recorded hyperparameter carries the provenance label it
    had in the YAML (master §0, Golden Rule 7).
    """
    return dict(cfg.get("source_labels", {}) or {})
