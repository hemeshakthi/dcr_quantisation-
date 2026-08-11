"""Test-set inference and per-image metric computation (master §6).

Runs a trained checkpoint over the test split, writes one per-image metric row per image. The
six failure detectors (§12) are a separate module (`src/failure/detectors.py`) called by
whoever orchestrates a full evaluation pass, not by this one.

Two safeguards that are not optional here:
  * every metric row records the checkpoint hash, so any number can be traced to the weights
    that produced it (master §14);
  * `detect_identical_outputs` compares raw predictions across models on a probe image.
    Master Golden Rule 2 makes two identical model outputs a BUG; a previous attempt at this
    project failed exactly that way, so it is checked automatically rather than assumed.

Post-processing is deliberately minimal -- a 0.5 threshold on the sigmoid, no morphological
cleanup ([ASSUMPTION] A07) -- and, whatever it is, it is IDENTICAL for all five models. A
post-processing step applied unevenly would be indistinguishable from a real effect.

[ENGINEERING, Phase 8 -- A44] Consolidates what a Phase 3 scaffold called
`src/evaluation/evaluate.py` (never implemented; nothing depended on it) under the name Phase 8
specifies. Model reconstruction reuses `backend/services/model_registry.py::build_model_from_state_dict`
(A40): reading a checkpoint's own tensor shapes is more robust than reconstructing from a
possibly-absent channel-mask file, and it is exactly as valid for offline test-set evaluation as
it is for the live backend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.metrics.per_image import PER_IMAGE_COLUMNS, append_per_image_metrics
from src.metrics.segmentation import compute_all_metrics
from src.models.unet import UNet
from src.training.checkpointing import checkpoint_hash
from src.training.experiment_log import utc_now


def load_model_for_eval(checkpoint_path: str | Path, config: dict[str, Any],
                        device: str = "cpu") -> Any:
    """Rebuild the network (narrowed to its channel mask if pruned) and load its weights.

    Puts the model in eval() mode, which for the pruned models means BatchNorm uses the
    recalibrated running statistics rather than batch statistics.

    Widths are read directly off the checkpoint's own saved tensors
    (`UNet.from_state_dict_shapes`, A40), not re-derived from `config`/a channel-mask file -- a
    checkpoint's weights are fully self-describing and this cannot silently disagree with what
    is actually being loaded. `config` is accepted for API compatibility with
    `experiments/multi_seed_train.py`'s config-driven callers but is not required to build the
    correct shape.
    """
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet.from_state_dict_shapes(payload["model_state"])
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


def predict_image(model: Any, image: Any, device: str = "cpu",
                  threshold: float = 0.5) -> dict[str, Any]:
    """Run inference on one image.

    Returns:
        {"logits", "probs", "binary"} -- the raw logits are kept so the UI can show
        confidence and so identical-output detection can compare pre-threshold values, where
        two genuinely different models are far less likely to coincide by chance.
    """
    t = image if isinstance(image, torch.Tensor) else torch.from_numpy(
        np.asarray(image, dtype=np.float32))
    t = t.float()
    while t.dim() < 4:
        t = t.unsqueeze(0)
    t = t.to(device)
    with torch.no_grad():
        logits = model(t)
        probs = torch.sigmoid(logits)
        binary = (probs >= threshold).to(torch.uint8)
    return {"logits": logits, "probs": probs, "binary": binary}


def detect_identical_outputs(predictions: dict[str, Any], atol: float = 0.0) -> list[str]:
    """Compare model predictions pairwise on the same input (Golden Rule 2).

    Args:
        predictions: {model_name: probs array/tensor} for one probe image.
        atol: 0.0 means byte-identical. Non-zero tolerance flags suspiciously-similar models
            too, which catches the subtler failure where two runs loaded the same checkpoint.

    Returns:
        Descriptions of any offending pairs; empty means all models genuinely differ.
    """
    names = list(predictions)
    problems: list[str] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = predictions[names[i]], predictions[names[j]]
            a_np = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
            b_np = b.detach().cpu().numpy() if hasattr(b, "detach") else np.asarray(b)
            if a_np.shape != b_np.shape:
                continue
            if np.allclose(a_np, b_np, atol=atol, rtol=0):
                problems.append(
                    f"{names[i]} and {names[j]} produced IDENTICAL outputs (atol={atol}) on "
                    f"the probe image -- Golden Rule 2 violation, investigate before trusting "
                    f"either checkpoint")
    return problems


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
    from src.datasets.dataset import CellSegmentationDataset, collate_batch
    from src.datasets.dataset_registry import DATA_PROCESSED
    from torch.utils.data import DataLoader

    man = pd.read_csv(DATA_PROCESSED / "preprocess_manifest.csv")
    split_df = man[man["split"] == split].reset_index(drop=True)
    if len(split_df) == 0:
        raise ValueError(f"no rows for split={split!r} in the preprocess manifest")
    ds = CellSegmentationDataset(split_df, augmenter=None, return_instances=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_batch)

    model = load_model_for_eval(checkpoint_path, config, device=device)
    ckpt_hash = checkpoint_hash(checkpoint_path)
    threshold = float(config.get("postprocessing", {}).get("threshold", 0.5))
    model_name = config.get("run", {}).get("name", "")
    pruning_cfg = config.get("pruning", {}) or {}
    ratio = pruning_cfg.get("ratio") if pruning_cfg.get("enabled") else None
    seed = config.get("_seed", "")

    rows: list[dict[str, Any]] = []
    for batch in loader:
        out = predict_image(model, batch["image"][0, 0].numpy(), device=device,
                            threshold=threshold)
        pred = out["binary"][0, 0].cpu().numpy()
        target = batch["mask"][0, 0].numpy() >= 0.5
        m = compute_all_metrics(pred, target)
        meta = batch["meta"][0] if batch["meta"] else {}
        rows.append({
            "run_id": run_id, "model": model_name, "pruning_ratio": ratio, "seed": seed,
            "image_id": batch["image_id"][0], "dataset": meta.get("dataset"), "split": split,
            "curvature_bin": meta.get("curvature_bin"),
            "curvature_score": meta.get("curvature_score"),
            "cell_size": meta.get("cell_size"), "contrast": meta.get("contrast"),
            "dice": m["dice"],
            "boundary_f1_2px": m["boundary_f1_2px"],
            "boundary_precision_2px": m["boundary_precision_2px"],
            "boundary_recall_2px": m["boundary_recall_2px"],
            "boundary_f1_5px": m["boundary_f1_5px"],
            "boundary_precision_5px": m["boundary_precision_5px"],
            "boundary_recall_5px": m["boundary_recall_5px"],
            "hd95": m["hd95"],
            "n_instances_gt": m["n_instances_gt"], "n_instances_pred": m["n_instances_pred"],
            "checkpoint_hash": ckpt_hash, "computed_at_utc": utc_now(),
        })

    if output_dir is not None:
        out_path = Path(output_dir) / f"{run_id}.csv"
        append_per_image_metrics(rows, out_path)

    df = pd.DataFrame(rows, columns=PER_IMAGE_COLUMNS)
    summary: dict[str, Any] = {}
    for bin_name, sub in df.groupby("curvature_bin", dropna=True):
        hd95_valid = sub["hd95"].dropna()
        summary[bin_name] = {
            "n": int(len(sub)), "dice_mean": float(sub["dice"].mean()),
            "boundary_f1_2px_mean": float(sub["boundary_f1_2px"].mean()),
            "boundary_f1_5px_mean": float(sub["boundary_f1_5px"].mean()),
            "hd95_mean": float(hd95_valid.mean()) if len(hd95_valid) else float("nan"),
            "hd95_n_valid": int(len(hd95_valid)),
        }
    return {"run_id": run_id, "n_images": len(rows), "checkpoint_hash": ckpt_hash,
           "summary_by_bin": summary, "rows": rows}


def aggregate_by_bin(per_image_csv: str | Path) -> pd.DataFrame:
    """Aggregate per-image metrics by curvature bin, model and ratio, for the paper's tables.

    Mid-bin rows are marked descriptive-only (A29) so a reader cannot mistake them for
    confirmatory evidence.
    """
    df = pd.read_csv(per_image_csv)
    agg = df.groupby(["model", "pruning_ratio", "curvature_bin"], dropna=False).agg(
        n=("image_id", "count"),
        dice_mean=("dice", "mean"),
        boundary_f1_2px_mean=("boundary_f1_2px", "mean"),
        boundary_f1_5px_mean=("boundary_f1_5px", "mean"),
        hd95_mean=("hd95", "mean"),
    ).reset_index()
    agg["descriptive_only"] = agg["curvature_bin"] == "mid"
    return agg
