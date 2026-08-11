"""GET /api/health -- docs/api_contract.md §1.

Triggers lazy loading of every expected checkpoint and runs the collapse probe (once, cached).
Merges the earlier draft's separate /api/health + /api/selfcheck + /api/models into one call.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.services.image_service import list_test_images
from backend.services.model_registry import ModelRegistry, get_registry

router = APIRouter()


@router.get("/health")
def health(registry: ModelRegistry = Depends(get_registry)) -> dict:
    registry.ensure_loaded()
    unavailable = registry.unavailable()
    collapse = registry.run_collapse_check()
    status = "ok" if (not unavailable and not collapse["collapsed"]) else "degraded"
    return {
        "status": status,
        "models_loaded": registry.loaded_ids(),
        "models_unavailable": sorted(unavailable),
        "unavailable_reasons": unavailable,
        "models": registry.static_metadata(),
        "collapse_check": collapse,
        "test_images_available": len(list_test_images()),
    }
