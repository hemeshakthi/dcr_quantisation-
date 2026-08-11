"""Live end-to-end backend verification against a REAL, genuinely trained/pruned tiny 5-model
fixture (master §14, backend-authoring phase's own verification step: "/health confirms 5
distinct checkpoints + passes collapse probe; /infer returns real DIFFERENT metrics per model;
frontend renders end-to-end").

WHY THIS EXISTS: the real `checkpoints/` directory is currently EMPTY -- the actual 24-run
Kaggle study (master §17) has not been executed yet, so `backend/` cannot be verified against
real full-scale checkpoints today. Rather than fabricate fixture data (forbidden, Golden Rule 1)
or skip verification entirely, this script does what Phase 6/6-RUN already established as this
project's pattern for exactly this situation: build a genuinely small-scale but GENUINELY REAL
artifact through the actual production code paths, and verify against that.

TWO PHASES, TWO SEPARATE OS PROCESSES. `--phase build` trains/prunes/recalibrates/fine-tunes the
5 tiny models (heavy torch working set: 7+ models, optimizers, thop profiling all live in memory
at once). `--phase verify` imports FastAPI + cv2 (via backend/main.py) and drives the real app
through TestClient. Run with no `--phase` at all (the default), this file orchestrates BOTH as
subprocesses rather than doing both in one process: this host's Windows commit limit equals
physical RAM (A18), and running everything in one process was observed to leave too little
headroom for cv2's DLL load right after the training phase's peak memory use. Two processes means
the training phase's memory is fully released (process exit) before the verify phase's cv2/
FastAPI import ever runs.

Concretely:
  1. [build] Loads real preprocessed microscopy crops (real pixel content, Phase 2 output) and
     trains a tiny (but architecturally real) UNet twice with real backprop: Vanilla (Dice-only)
     and A_dcr (Dice+dCR), via the real `Trainer.fit()`.
  2. [build] Prunes A_dcr's real best checkpoint with the real `l1_structured.prune_model` at
     ratio 0.5, producing one shared channel mask -- exactly the master §9 protocol.
  3. [build] Recalibrates BatchNorm and fine-tunes THREE independent copies of that shared pruned
     base with three different real losses (Dice-only = B, generic boundary = B', dCR = C), using
     the SAME recalibration batch stream for all three (order-fingerprint verified).
  4. [build] Writes all 5 checkpoints into an isolated temp directory in the exact
     `<run_name>_seed1/best.pt` layout `backend/services/model_registry.py` expects.
  5. [verify] Points a real `ModelRegistry` at that temp directory (via FastAPI
     `dependency_overrides`, never touching the real, currently-empty `checkpoints/`) and drives
     the real `backend/main.py` app through `TestClient`: `/api/health`, `/api/images`,
     `/api/infer`.
  6. [orchestrator] Deletes the temp directory afterward.

This is a genuine (if tiny and CPU-fast) run of the full pipeline, not a mock of it -- every
number asserted below was computed by real forward/backward passes on real image content.
"""
from __future__ import annotations

import argparse
import base64
import copy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 1
CROP = 128  # divisible by 16 (UNet.DOWNSAMPLE_FACTOR); small enough for a fast CPU pass, real
           # pixel content (a window of a real Phase 2 preprocessed image, not synthetic data)

TINY_MODEL_CFG = {
    "architecture": "unet", "in_channels": 1, "out_channels": 1,
    "encoder_channels": [4, 8, 16, 32], "bottleneck_channels": 64,
    "decoder_channels": [32, 16, 8, 4], "bilinear_upsample": False,
    "final_activation": None,
}


def base_cfg(max_epochs: int = 2) -> dict:
    return {
        "model": dict(TINY_MODEL_CFG),
        "optimizer": {"name": "adam", "lr": 1e-3, "weight_decay": 0.0},
        "scheduler": {"name": "none"},
        "training": {"max_epochs": max_epochs, "batch_size": 2, "mixed_precision": False,
                    "gradient_accumulation_steps": 1, "log_gradient_norms": True},
        "early_stopping": {"enabled": False, "monitor": "val_boundary_f1_2px", "mode": "max",
                          "patience": 100},
        "checkpointing": {"dir": "checkpoints", "monitor": "val_boundary_f1_2px", "mode": "max"},
        "logging": {"dir": "logs", "tensorboard": False},
        "metrics": {"boundary_f1": {"tolerances_px": [2.0, 5.0]}},
        "postprocessing": {"threshold": 0.5},
        "loss": {"region": {"name": "dice", "smooth": 1e-5, "alpha": 1.0},
                 "boundary": {"name": None, "beta": 0.0}},
    }


# ==========================================================================================
# PHASE 1: build the fixture (torch/training only -- no FastAPI, no cv2)
# ==========================================================================================
def phase_build(out_root: Path) -> int:
    import numpy as np
    import pandas as pd
    import torch

    from src.compression.bn_recalibration import (assert_identical_recalibration,
                                                   recalibrate_batchnorm)
    from src.compression.compression_accounting import (assert_identical_complexity,
                                                         measure_complexity)
    from src.compression.structured_pruning import prune_model
    from src.datasets.dataset_registry import DATA_PROCESSED
    from src.losses.combined import (build_boundary_loss, build_loss_from_config,
                                     calibrate_beta_by_gradient_norm)
    from src.models.unet import UNet
    from src.training.checkpointing import load_checkpoint
    from src.training.trainer import Trainer

    failures: list[str] = []

    def check(label, cond, warn_only=False):
        ok = bool(cond)
        print(f"[{'PASS' if ok else ('WARN' if warn_only else 'FAIL')}] {label}")
        if not ok and not warn_only:
            failures.append(label)
        return ok

    def load_crop(proc_image_path: str, proc_instances_path: str, crop: int, seed: int):
        img = np.load(ROOT / proc_image_path).astype(np.float32)
        inst = np.load(ROOT / proc_instances_path)
        h, w = img.shape[:2]
        rng = np.random.default_rng(seed)
        y0 = int(rng.integers(0, max(1, h - crop + 1)))
        x0 = int(rng.integers(0, max(1, w - crop + 1)))
        img_c = img[y0:y0 + crop, x0:x0 + crop]
        mask_c = (inst[y0:y0 + crop, x0:x0 + crop] > 0).astype(np.float32)
        return img_c, mask_c

    def stack_batch(imgs, masks, ids):
        x = torch.from_numpy(np.stack(imgs)).unsqueeze(1).float()
        y = torch.from_numpy(np.stack(masks)).unsqueeze(1).float()
        return {"image": x, "mask": y, "image_id": list(ids), "meta": [{} for _ in ids]}

    def make_batches(rows: list[dict], batch_size: int, seed: int) -> list[dict]:
        batches, imgs, masks, ids = [], [], [], []
        for i, r in enumerate(rows):
            im, mk = load_crop(r["proc_image"], r["proc_instances"], CROP, seed=seed + i)
            imgs.append(im)
            masks.append(mk)
            ids.append(r["image_id"])
            if len(imgs) == batch_size:
                batches.append(stack_batch(imgs, masks, ids))
                imgs, masks, ids = [], [], []
        if imgs:
            batches.append(stack_batch(imgs, masks, ids))
        return batches

    print("---------------- [build] real training/validation crops ----------------")
    manifest = pd.read_csv(DATA_PROCESSED / "preprocess_manifest.csv")
    train_all = manifest[manifest["split"] == "train"].sort_values("image_id").reset_index(drop=True)
    check(f"found real preprocessed train-split rows to build the fixture from ({len(train_all)})",
          len(train_all) >= 12)
    train_rows = train_all.iloc[:8].to_dict("records")
    val_rows = train_all.iloc[8:12].to_dict("records")

    train_batches = make_batches(train_rows, batch_size=2, seed=1)
    val_batches = make_batches(val_rows, batch_size=2, seed=101)
    recal_batches = make_batches(train_rows, batch_size=2, seed=1)  # SAME crops/order as train
    check(f"built {len(train_batches)} real train batches / {len(val_batches)} real val batches "
          f"/ {len(recal_batches)} real BN-recalibration batches from real image crops",
          len(train_batches) > 0 and len(val_batches) > 0 and len(recal_batches) > 0)

    print("\n---------------- [build] training Vanilla + A_dcr (real Trainer.fit()) ----------------")
    vanilla_cfg = base_cfg()
    vanilla_model = UNet.from_config(vanilla_cfg, init_seed=1)
    vanilla_loss = build_loss_from_config(vanilla_cfg)
    vanilla_trainer = Trainer(model=vanilla_model, loss_fn=vanilla_loss,
                              train_loader=train_batches, val_loader=val_batches,
                              config=vanilla_cfg, run_id=f"vanilla_seed{SEED}", device="cpu",
                              output_root=out_root, log_tensorboard=False)
    vanilla_summary = vanilla_trainer.fit()
    check("vanilla training completed", vanilla_summary["status"] == "completed")
    check("vanilla best.pt was written",
          (out_root / "checkpoints" / f"vanilla_seed{SEED}" / "best.pt").exists())

    a_cfg = base_cfg()
    a_cfg["loss"]["boundary"] = {
        "name": "dcr", "beta": 1.0, "band_width": 3, "curvature_gamma": 1.0,
        "curvature_weight_power": 1.0, "curvature_smooth_sigma": 2.0,
        "curvature_source": "target", "curvature_proxy": "sobel_levelset",
        "normalise_weights": True,
    }
    a_model = UNet.from_config(a_cfg, init_seed=2)
    a_loss = build_loss_from_config(a_cfg)
    a_trainer = Trainer(model=a_model, loss_fn=a_loss, train_loader=train_batches,
                        val_loader=val_batches, config=a_cfg, run_id=f"A_dcr_seed{SEED}",
                        device="cpu", output_root=out_root, log_tensorboard=False)
    a_summary = a_trainer.fit()
    check("A_dcr training completed", a_summary["status"] == "completed")
    a_best_path = out_root / "checkpoints" / f"A_dcr_seed{SEED}" / "best.pt"
    check("A_dcr best.pt was written", a_best_path.exists())
    del vanilla_model, vanilla_trainer, a_model, a_trainer

    print("\n---------------- [build] pruning A_dcr's best checkpoint (real l1_structured) ----------------")
    a_best_model = UNet.from_config(a_cfg, init_seed=2)
    load_checkpoint(a_best_path, model=a_best_model, map_location="cpu")

    mask_path = out_root / "channel_masks" / "ratio_0.5.json"
    pruned_base, prune_record = prune_model(a_best_model, ratio=0.5, mask_path=mask_path,
                                            min_keep=2)
    check("pruning genuinely reduced measured params (not a no-op)",
          prune_record["params_after"] < prune_record["params_before"])
    print(f"  params: {prune_record['params_before']} -> {prune_record['params_after']} "
          f"({100 * (1 - prune_record['params_after'] / prune_record['params_before']):.1f}% reduction)")
    check("mask file was written (write-once)", mask_path.exists())
    del a_best_model

    print("\n---------------- [build] B / B' / C: recalibrate + fine-tune (real) ----------------")

    def recalibrate_and_finetune(run_name: str, loss_cfg_boundary: dict | None,
                                 beta_override: float | None = None):
        model = copy.deepcopy(pruned_base)
        recal_record = recalibrate_batchnorm(model, recal_batches, num_batches=len(recal_batches),
                                             device="cpu", momentum=None, reset_first=True)
        cfg = base_cfg()
        cfg["loss"]["boundary"] = loss_cfg_boundary or {"name": None, "beta": 0.0}
        if beta_override is not None:
            cfg["loss"]["boundary"]["beta"] = beta_override
        loss_fn = build_loss_from_config(cfg)
        trainer = Trainer(model=model, loss_fn=loss_fn, train_loader=train_batches,
                          val_loader=val_batches, config=cfg, run_id=f"{run_name}_seed{SEED}",
                          device="cpu", output_root=out_root,
                          channel_mask_hash=prune_record["mask_hash"], log_tensorboard=False)
        summary = trainer.fit()
        return model, recal_record, summary

    b_model, b_recal, b_summary = recalibrate_and_finetune("B_pruned", None)
    check("B (Dice-only) training completed", b_summary["status"] == "completed")

    # B' beta is gradient-norm-calibrated against C's dCR term (master §8), not hand-set.
    dcr_boundary_ref = build_boundary_loss(a_cfg)          # the SAME boundary config as A/C
    generic_boundary_cfg = {"name": "generic_boundary", "band_width": 3, "beta": None}
    generic_boundary_target = build_boundary_loss({"loss": {"boundary": generic_boundary_cfg}})
    calib_model = copy.deepcopy(pruned_base)
    calib = calibrate_beta_by_gradient_norm(dcr_boundary_ref, generic_boundary_target,
                                            calib_model, recal_batches, reference_beta=1.0,
                                            device="cpu")
    check("B' beta was gradient-norm calibrated (finite, positive)",
          calib["beta"] == calib["beta"] and calib["beta"] > 0)
    del calib_model
    bp_model, bp_recal, bp_summary = recalibrate_and_finetune(
        "B_prime_generic", dict(generic_boundary_cfg), beta_override=calib["beta"])
    check("B' (generic boundary) training completed", bp_summary["status"] == "completed")

    c_model, c_recal, c_summary = recalibrate_and_finetune("C_pruned_dcr", a_cfg["loss"]["boundary"])
    check("C (dCR) training completed", c_summary["status"] == "completed")

    print("\n---------------- [build] master §9 shared-architecture invariants ----------------")
    try:
        assert_identical_recalibration({"B": b_recal, "B_prime": bp_recal, "C": c_recal})
        check("B/B'/C consumed the IDENTICAL BN-recalibration batch stream (order fingerprint)", True)
    except AssertionError as e:
        check(f"B/B'/C recalibration streams identical ({e})", False)

    complexities = {"B": measure_complexity(b_model, input_shape=(1, 1, CROP, CROP)),
                    "B_prime": measure_complexity(bp_model, input_shape=(1, 1, CROP, CROP)),
                    "C": measure_complexity(c_model, input_shape=(1, 1, CROP, CROP))}
    try:
        assert_identical_complexity(complexities)
        check("B/B'/C have IDENTICAL measured params/FLOPs (share one pruned architecture)", True)
    except AssertionError as e:
        check(f"B/B'/C measured complexity identical ({e})", False)

    expected_dirs = ["vanilla", "A_dcr", "B_pruned", "B_prime_generic", "C_pruned_dcr"]
    for run_name in expected_dirs:
        best = out_root / "checkpoints" / f"{run_name}_seed{SEED}" / "best.pt"
        check(f"  {run_name}_seed{SEED}/best.pt exists", best.exists())

    print()
    if failures:
        print(f"[build] RESULT: FAIL ({len(failures)} check(s) failed)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[build] RESULT: PASS -- real tiny 5-model fixture built successfully.")
    return 0


# ==========================================================================================
# PHASE 2: verify the live backend against the fixture (FastAPI/cv2 only -- fresh process)
# ==========================================================================================
def phase_verify(checkpoints_dir: Path) -> int:
    failures: list[str] = []

    def check(label, cond, warn_only=False):
        ok = bool(cond)
        print(f"[{'PASS' if ok else ('WARN' if warn_only else 'FAIL')}] {label}")
        if not ok and not warn_only:
            failures.append(label)
        return ok

    print("---------------- [verify] live backend against the real fixture (FastAPI TestClient) ----------------")
    from fastapi.testclient import TestClient

    from backend.main import app as fastapi_app
    from backend.services.model_registry import ModelRegistry, get_registry

    fixture_registry = ModelRegistry(checkpoints_root=checkpoints_dir, seed=SEED)
    fastapi_app.dependency_overrides[get_registry] = lambda: fixture_registry
    try:
        client = TestClient(fastapi_app)

        r_health = client.get("/api/health")
        check("GET /api/health returns 200", r_health.status_code == 200)
        health = r_health.json()
        check("all 5/5 models loaded", sorted(health.get("models_loaded", [])) ==
              sorted(["vanilla", "A", "B", "B_prime", "C"]))
        check("no unavailable models", not health.get("models_unavailable"))
        hashes = [m["checkpoint_hash"] for m in health.get("models", [])]
        check(f"all 5 checkpoint hashes are DISTINCT ({len(set(hashes))}/{len(hashes)})",
              len(set(hashes)) == len(hashes) == 5)
        check("collapse probe PASSES (no identical pairs) on the real probe image",
              health.get("collapse_check", {}).get("collapsed") is False)
        check("collapse probe compared all C(5,2)=10 pairs",
              len(health.get("collapse_check", {}).get("pairs", [])) == 10)

        r_images = client.get("/api/images")
        check("GET /api/images returns 200 with real test-split images", r_images.status_code == 200
              and len(r_images.json().get("images", [])) > 0)
        default_id = r_images.json().get("default_image_id")

        r_infer = client.post("/api/infer", json={"image_id": default_id})
        check("POST /api/infer on a REAL test image returns 200", r_infer.status_code == 200)
        infer_body = r_infer.json()
        models_out = infer_body.get("models", [])
        check("infer response carries all 5 models", len(models_out) == 5)

        # ---- Golden Rule 2, exercised on REAL end-to-end inference (not just the probe) ----
        metrics_by_model = {m["model_id"]: m["metrics"] for m in models_out}
        dice_values = [m["dice"] for m in metrics_by_model.values()]
        check("Dice is NOT byte-identical across all 5 models on this real image "
              f"({dice_values})", len(set(dice_values)) > 1)

        pred_b64s = {m["model_id"]: m["prediction_mask_png_base64"] for m in models_out}
        check("prediction masks are NOT byte-identical across all 5 models",
              len(set(pred_b64s.values())) > 1)

        for m in models_out:
            for field in ("prediction_mask_png_base64", "overlay_png_base64", "error_map_png_base64"):
                raw = base64.b64decode(m[field])
                check(f"  {m['model_id']}.{field} decodes to a real PNG (magic bytes)",
                      raw[:8] == b"\x89PNG\r\n\x1a\n")

        r_404 = client.post("/api/infer", json={"image_id": "does-not-exist"})
        check("POST /api/infer on an unknown image_id still returns 404 (fixture didn't break this)",
              r_404.status_code == 404)
    finally:
        fastapi_app.dependency_overrides.pop(get_registry, None)

    print()
    if failures:
        print(f"[verify] RESULT: FAIL ({len(failures)} check(s) failed)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[verify] RESULT: PASS -- backend verified live against the real fixture.")
    return 0


# ==========================================================================================
# orchestrator: runs build and verify as SEPARATE processes (see module docstring, A18)
# ==========================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["build", "verify"], default=None,
                    help="internal: run only one phase (used by the orchestrator's subprocess calls)")
    ap.add_argument("--dir", default=None, help="internal: fixture directory for the given phase")
    args = ap.parse_args()

    if args.phase == "build":
        return phase_build(Path(args.dir))
    if args.phase == "verify":
        return phase_verify(Path(args.dir))

    print("=" * 78)
    print("BACKEND INTEGRATION TEST -- real tiny 5-model pipeline + live TestClient (master §14)")
    print("=" * 78)
    print("NOTE: the real checkpoints/ is currently EMPTY (Kaggle training has not run, master §17).")
    print("This builds a genuinely trained/pruned tiny fixture in an ISOLATED temp dir to prove the")
    print("prune -> recalibrate -> fine-tune -> checkpoint -> backend -> API pipeline is real and")
    print("correct end to end. It is NOT a claim about the eventual full-scale study result.")
    print()
    print("Running as two separate OS processes (build, then verify) so peak training memory and")
    print("cv2/FastAPI's own footprint never coexist in one process -- see A18 (host commit limit")
    print("== physical RAM).")
    print()

    tmp_root = Path(tempfile.mkdtemp(prefix="dcr_backend_fixture_"))
    print(f"fixture temp dir: {tmp_root}\n")
    try:
        r1 = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--phase", "build", "--dir", str(tmp_root)])
        build_ok = r1.returncode == 0
        if not build_ok:
            print("\n[orchestrator] build phase FAILED -- skipping verify phase.")
            return 1

        print()
        r2 = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--phase", "verify", "--dir", str(tmp_root / "checkpoints")])
        verify_ok = r2.returncode == 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"\ncleaned up fixture temp dir: {tmp_root}")

    print()
    print("=" * 78)
    if build_ok and verify_ok:
        print("RESULT: PASS -- backend verified end to end against a real (tiny) 5-model fixture.")
        print("=" * 78)
        return 0
    print("RESULT: FAIL -- see build/verify phase output above.")
    print("=" * 78)
    return 1


if __name__ == "__main__":
    sys.exit(main())
