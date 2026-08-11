"""Orchestrates one POST /api/infer response: live inference + rendering + metrics for every
loaded model on one test image (docs/api_contract.md §3).
"""
from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from src.visualization.contours import LINE_STYLES, draw_contour, error_map, to_display_u8

from backend.services.image_service import load_image_and_gt
from backend.services.metric_service import compute_case_metrics
from backend.services.model_registry import ModelRegistry, predict_prob


def _png_base64(arr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("cv2.imencode('.png', ...) failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_case_response(image_id: str, registry: ModelRegistry) -> dict[str, Any]:
    """Live inference on every currently-loaded model (docs/api_contract.md §3).

    Raises `backend.services.image_service.ImageNotFoundError` for an unknown image_id -- left
    to propagate so the router can turn it into a 404, per §4.
    """
    img, gt, meta = load_image_and_gt(image_id)
    gray = to_display_u8(img)

    models_out = []
    for model_id in registry.loaded_ids():
        model = registry.get_model(model_id)
        prob = predict_prob(model, img)
        pred = (prob > registry.threshold).astype(np.uint8)

        style = LINE_STYLES.get(model_id, "solid")
        overlay = draw_contour(gray, pred, style=style)
        err = error_map(pred, gt)
        model_meta = registry.meta_for(model_id)

        models_out.append({
            "model_id": model_id,
            "prediction_mask_png_base64": _png_base64(pred * 255),
            "overlay_png_base64": _png_base64(overlay),
            "error_map_png_base64": _png_base64(err),
            "metrics": compute_case_metrics(pred, gt),
            "model_size_bytes": model_meta["model_size_bytes"],
            "params_count": model_meta["params_count"],
            "pruning_ratio": model_meta["pruning_ratio"],
        })

    gt_overlay = draw_contour(gray, gt, style=LINE_STYLES["ground_truth"])

    return {
        "image_id": image_id,
        "dataset": meta["dataset"],
        "curvature_bin": meta["curvature_bin"],
        "curvature_score": meta["curvature_score"],
        "cell_size": meta["cell_size"],
        "contrast": meta["contrast"],
        "original_png_base64": _png_base64(gray),
        "ground_truth": {
            "mask_png_base64": _png_base64(gt * 255),
            "overlay_png_base64": _png_base64(gt_overlay),
        },
        "models": models_out,
    }
