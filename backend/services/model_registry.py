"""Loads and caches the five real checkpoints; runs the model-collapse probe (Golden Rule 2).

Lazy, cached loading -- see docs/api_contract.md §0: checkpoints are loaded on first request,
not blocking process startup, since this host may have zero checkpoints present at boot (the
CURRENT real state -- Kaggle training has not yet run) and startup must not hang or crash on
that. Degraded mode (some/all checkpoints missing) is reported honestly via
`models_unavailable`/`unavailable_reasons`, never papered over with a fabricated fifth column
(Golden Rule 1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import load_yaml
from src.datasets.dataset_registry import ROOT
from src.models.unet import UNet
from src.training.checkpointing import checkpoint_hash
from src.visualization.contours import LINE_STYLES

from backend.services.image_service import default_image_id, list_test_images, load_image_and_gt

SEED = 1  # A35 (docs/implementation_assumptions.md): every UI model column shows seed 1.

# model_id (matches src.visualization.contours.LINE_STYLES keys) -> run.name (matches
# configs/model_*.yaml and the checkpoints/<run_name>_seed<seed>/ directory convention, A35).
MODEL_SPECS: list[dict[str, Any]] = [
    {"model_id": "vanilla", "run_name": "vanilla", "label": "Vanilla",
     "is_reference": False, "pruning_ratio": None},
    {"model_id": "A", "run_name": "A_dcr", "label": "A (uncompressed + dCR, reference)",
     "is_reference": True, "pruning_ratio": None},
    {"model_id": "B", "run_name": "B_pruned", "label": "B (pruned, Dice only -- the damage)",
     "is_reference": False, "pruning_ratio": 0.5},
    {"model_id": "B_prime", "run_name": "B_prime_generic",
     "label": "B′ (pruned, generic boundary -- the control)",
     "is_reference": False, "pruning_ratio": 0.5},
    {"model_id": "C", "run_name": "C_pruned_dcr",
     "label": "C (pruned + dCR -- the proposed recovery)",
     "is_reference": False, "pruning_ratio": 0.5},
]


# build_model_from_state_dict moved to UNet.from_state_dict_shapes (src/models/unet.py, A40) so
# src/evaluation/eval_runner.py can share it without depending on backend/ -- see A44.
build_model_from_state_dict = UNet.from_state_dict_shapes


def predict_prob(model: torch.nn.Module, image_hw: np.ndarray) -> np.ndarray:
    """Pre-threshold sigmoid probability map for one image, shape (H, W) float32."""
    model.eval()
    t = torch.from_numpy(np.ascontiguousarray(image_hw, dtype=np.float32))
    t = t.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(t))[0, 0].numpy()
    return prob


def _relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


class ModelRegistry:
    """Lazily loads and caches the five real checkpoints; runs the collapse probe once."""

    def __init__(self, checkpoints_root: str | Path | None = None, seed: int = SEED) -> None:
        self.checkpoints_root = Path(checkpoints_root) if checkpoints_root else ROOT / "checkpoints"
        self.seed = int(seed)
        self._models: dict[str, torch.nn.Module] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._unavailable: dict[str, str] = {}
        self._collapse_cache: dict[frozenset, dict[str, Any]] = {}
        threshold_cfg = load_yaml("evaluation.yaml").get("postprocessing", {})
        self.threshold = float(threshold_cfg.get("threshold", 0.5))

    def ensure_loaded(self) -> None:
        """Load every checkpoint not yet attempted.

        Safe to call every request: once a model has either loaded or been recorded
        unavailable, this is a no-op for it, so repeated `/api/health` calls do not repeatedly
        retry a checkpoint that simply does not exist.
        """
        for spec in MODEL_SPECS:
            mid = spec["model_id"]
            if mid in self._models or mid in self._unavailable:
                continue
            self._load_one(spec)

    def _load_one(self, spec: dict[str, Any]) -> None:
        mid = spec["model_id"]
        path = self.checkpoints_root / f"{spec['run_name']}_seed{self.seed}" / "best.pt"
        if not path.exists():
            self._unavailable[mid] = f"checkpoint not found at {_relpath(path)}"
            return
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = payload["model_state"]
            model = build_model_from_state_dict(state_dict)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as exc:  # noqa: BLE001 -- reported honestly, never swallowed silently
            self._unavailable[mid] = f"failed to load checkpoint {path.name}: {exc}"
            return
        self._models[mid] = model
        self._meta[mid] = {
            "model_id": mid,
            "label": spec["label"],
            "is_reference": spec["is_reference"],
            "line_style": LINE_STYLES.get(mid, "solid"),
            "pruning_ratio": spec["pruning_ratio"],
            "params_count": int(sum(p.numel() for p in model.parameters())),
            "model_size_bytes": path.stat().st_size,
            "checkpoint_hash": checkpoint_hash(path),
            "checkpoint_path": _relpath(path),
            "seed": self.seed,
        }

    def loaded_ids(self) -> list[str]:
        return [s["model_id"] for s in MODEL_SPECS if s["model_id"] in self._models]

    def unavailable(self) -> dict[str, str]:
        return dict(self._unavailable)

    def static_metadata(self) -> list[dict[str, Any]]:
        return [self._meta[mid] for mid in self.loaded_ids()]

    def get_model(self, model_id: str) -> torch.nn.Module:
        return self._models[model_id]

    def meta_for(self, model_id: str) -> dict[str, Any]:
        return self._meta[model_id]

    def run_collapse_check(self) -> dict[str, Any]:
        """Pairwise identical-output probe (Golden Rule 2), cached per loaded-model-set.

        Compares PRE-THRESHOLD sigmoid probabilities at atol=0, matching
        configs/evaluation.yaml's `identical_output_check` (compare_pre_threshold, atol=0.0).
        """
        loaded = self.loaded_ids()
        key = frozenset(loaded)
        if key in self._collapse_cache:
            return self._collapse_cache[key]

        images = list_test_images()
        probe_id = default_image_id(images) if images else None
        result: dict[str, Any] = {"probe_image_id": probe_id, "collapsed": False, "pairs": []}
        if probe_id is not None and len(loaded) >= 2:
            img, _gt, _meta = load_image_and_gt(probe_id)
            probs = {mid: predict_prob(self._models[mid], img) for mid in loaded}
            pairs = []
            for i in range(len(loaded)):
                for j in range(i + 1, len(loaded)):
                    a, b = loaded[i], loaded[j]
                    identical = bool(np.array_equal(probs[a], probs[b]))
                    pairs.append({
                        "model_a": a, "model_b": b, "identical": identical,
                        "mean_abs_diff": float(np.abs(probs[a] - probs[b]).mean()),
                    })
            result["pairs"] = pairs
            result["collapsed"] = any(p["identical"] for p in pairs)
        self._collapse_cache[key] = result
        return result


_default_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """FastAPI dependency: one process-lifetime ModelRegistry by default.

    Overridable per-app via `app.dependency_overrides[get_registry] = lambda: other_registry`
    -- how the integration test points the backend at a tiny fixture checkpoints/ directory
    instead of the real (currently empty) one, without monkeypatching this module's global.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry()
    return _default_registry
