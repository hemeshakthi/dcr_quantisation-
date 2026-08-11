"""GET /api/images -- docs/api_contract.md §2. Test-split only, labelled by curvature bin."""
from __future__ import annotations

from fastapi import APIRouter

from backend.services.image_service import default_image_id, list_test_images

router = APIRouter()


@router.get("/images")
def images() -> dict:
    imgs = list_test_images()
    return {"default_image_id": default_image_id(imgs), "images": imgs}
