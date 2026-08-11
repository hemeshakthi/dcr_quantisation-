"""POST /api/infer -- docs/api_contract.md §3. Live inference on every loaded model."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.services.image_service import ImageNotFoundError
from backend.services.inference_service import build_case_response
from backend.services.model_registry import ModelRegistry, get_registry

router = APIRouter()


class InferRequest(BaseModel):
    image_id: str


@router.post("/infer")
def infer(body: InferRequest, registry: ModelRegistry = Depends(get_registry)) -> dict:
    registry.ensure_loaded()
    try:
        return build_case_response(body.image_id, registry)
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 -- surfaced to the caller, never swallowed
        raise HTTPException(status_code=500, detail=str(exc)) from exc
