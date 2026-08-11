"""Phase 7 -- Structured Pruning: SYNTHETIC validation (master §9).

Author-only phase: no Kaggle, no real trained checkpoint exists yet. Verifies pruning
MECHANICS -- does the code correctly remove channels, does the mask round-trip correctly, do
measured params/FLOPs actually decrease, is BatchNorm recalibration deterministic -- using a
FRESH, RANDOMLY-INITIALISED UNet (real architecture, real UNet class, UNTRAINED weights) as a
stand-in for a real trained checkpoint. This says NOTHING about real model quality; it only
proves the pruning pipeline itself is correct, exactly as the phase instruction frames it.

Also exercises the ACTUAL Kaggle runner path added in this phase
(`experiments/multi_seed_train.py::_build_pruned_base`) against the same synthetic fixture --
not just the library functions in isolation -- since that is the code the real Phase 7-RUN
notebook's Cell 8 will call for B/B'/C.

TWO PHASES, PER-RATIO SUBPROCESSES for the thop calls. [ENGINEERING -- A43] Profiling the full
31M-parameter frozen architecture with thop (`measure_complexity`, called inside
`StructuredPruner.prune()`) was found to SEGFAULT this RAM-constrained authoring host (A18)
intermittently -- reproducible, but load-dependent rather than tied to one specific input size
or code path (bisection ruled out `copy.deepcopy` alone and 256x256 alone; the crash recurred
even after removing the deepcopy). Exactly like `audit/verify_backend_integration.py`'s cv2/
training memory conflict, the reliable fix is process isolation: each ratio's thop-touching
"prune + measure" step runs in its OWN short-lived subprocess (`--phase measure_ratio`), so its
peak memory is fully released before anything else runs. Every other check in this file (mask
reload, BN/Conv alignment, mask-reuse hashing, inference shape, BN recalibration determinism,
the runner path) never calls thop and runs safely in the main process.

Every file this script writes lives under results/synthetic_validation/phase7/ and is prefixed
SYNTHETIC_ -- per standing instruction: synthetic/smoke-validation artifacts must never be mixed
into the folder structure real results will occupy.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "synthetic_validation" / "phase7"
STANDIN_PATH = OUT_DIR / "SYNTHETIC_standin_model_a" / "best.pt"
LOCAL_INPUT_SHAPE = (1, 1, 128, 128)  # smaller but real, 16-divisible -- see A43
MEASURE_RATIO_MAX_ATTEMPTS = 3  # A43: the crash is intermittent even in a fresh subprocess
# A43: even with the input above and process isolation, the crash recurred with the full
# 31M-parameter frozen widths -- reproduced once inside a bare torch.save(), i.e. genuine host
# memory pressure (A18), not something tied to any one function. Quarter-width channels here
# (still the REAL `UNet` class, still 4 encoder stages + bottleneck + 4 decoder stages + skip
# connections -- every code path this phase tests is exercised identically) cut per-tensor
# memory ~16x and eliminated the crash in repeated local runs. This is a "mechanics only" phase
# by the instruction's own framing; the real 1024x1024, full-width measurement happens for real
# once Kaggle training produces an actual Model A checkpoint (Phase 7-RUN).
WIDTH_DIVISOR = 4


def _arch() -> dict:
    from src.config import load_yaml
    full = load_yaml("model_a_dcr.yaml")["model"]
    reduced = dict(full)
    reduced["encoder_channels"] = [max(2, c // WIDTH_DIVISOR) for c in full["encoder_channels"]]
    reduced["bottleneck_channels"] = max(2, full["bottleneck_channels"] // WIDTH_DIVISOR)
    reduced["decoder_channels"] = [max(2, c // WIDTH_DIVISOR) for c in full["decoder_channels"]]
    return reduced


def _mask_path(ratio: float) -> Path:
    return OUT_DIR / f"SYNTHETIC_phase7_channel_mask_ratio_{ratio}.json"


def _record_path(ratio: float) -> Path:
    return OUT_DIR / f"SYNTHETIC_phase7_record_ratio_{ratio}.json"


# ==========================================================================================
# subprocess phase: the ONLY place thop.profile is called
# ==========================================================================================
def phase_measure_ratio(ratio: float, min_keep: int) -> int:
    from src.compression.structured_pruning import StructuredPruner
    from src.models.unet import UNet
    from src.training.checkpointing import load_checkpoint

    arch = _arch()
    fresh = UNet.from_config({"model": arch})
    load_checkpoint(STANDIN_PATH, model=fresh, map_location="cpu")

    pruner = StructuredPruner()
    mask_path = _mask_path(ratio)
    _pruned, record = pruner.prune(fresh, ratio=ratio, mask_path=str(mask_path),
                                   min_keep=min_keep, input_shape=LOCAL_INPUT_SHAPE)
    _record_path(ratio).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[measure_ratio {ratio}] params {record['params_before']:,} -> "
          f"{record['params_after']:,}; flops {record['flops_before']:,} -> "
          f"{record['flops_after']:,}; mask_hash={record['mask_hash'][:16]}...")
    return 0


# ==========================================================================================
# orchestrator (default, no --phase)
# ==========================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["measure_ratio"], default=None,
                    help="internal: used by the orchestrator's subprocess calls")
    ap.add_argument("--ratio", type=float, default=None)
    ap.add_argument("--min-keep", type=int, default=None)
    args = ap.parse_args()

    if args.phase == "measure_ratio":
        return phase_measure_ratio(args.ratio, args.min_keep)

    import pandas as pd

    from src.compression.bn_recalibration import recalibrate_batchnorm
    from src.compression.mask_io import compute_mask_hash, load_channel_mask
    from src.compression.structured_pruning import StructuredPruner
    from src.config import compose_run_config, load_yaml
    from src.models.unet import UNet
    from src.training.checkpointing import load_checkpoint, save_checkpoint

    failures: list[str] = []
    warnings: list[str] = []

    def check(label, cond, warn_only=False):
        ok = bool(cond)
        print(f"[{'PASS' if ok else ('WARN' if warn_only else 'FAIL')}] {label}")
        if not ok and not warn_only:
            failures.append(label)
        return ok

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)  # write-once masks below would otherwise refuse a re-run
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PHASE 7 VERIFICATION -- Structured Pruning (SYNTHETIC validation, master §9)")
    print("=" * 78)
    print("Stand-in 'checkpoint': a FRESH, RANDOMLY-INITIALISED UNet (real architecture, real")
    print("UNet class, UNTRAINED weights). This tests pruning MECHANICS ONLY -- it says nothing")
    print("about real model quality.")
    print(f"Every output file below is namespaced under {OUT_DIR.relative_to(ROOT)} "
          f"(SYNTHETIC_ prefix), never mixed with real results.")
    print()

    # ---- 0. the real frozen architecture and pruning protocol (from configs/) -----
    ARCH = _arch()
    print(f"architecture: real UNet class, {WIDTH_DIVISOR}x-reduced widths from "
          f"configs/model_a_dcr.yaml (A43, host memory) -- "
          f"encoder={ARCH['encoder_channels']} bottleneck={ARCH['bottleneck_channels']} "
          f"decoder={ARCH['decoder_channels']}")

    pruning_cfg_yaml = load_yaml("pruning.yaml")["pruning"]
    RATIOS = pruning_cfg_yaml["ratios"]
    MIN_KEEP = max(2, int(pruning_cfg_yaml["min_channels_per_layer"]) // WIDTH_DIVISOR)
    check(f"pruning.yaml's frozen ratios are [0.5, 0.7] (master §9) -- got {RATIOS}",
          RATIOS == [0.5, 0.7])

    # ---- 1. build + save the SYNTHETIC stand-in "Model A" checkpoint --------------
    print("\n---------------- stand-in checkpoint ----------------")
    standin = UNet.from_config({"model": ARCH}, init_seed=42)
    ckpt_hash = save_checkpoint(
        STANDIN_PATH, standin, epoch=0, metrics={}, config={"model": ARCH},
        extra={"synthetic": True,
              "warning": "UNTRAINED random-init weights -- pruning MECHANICS test only, "
                         "not a real model. See audit/verify_phase7.py."})
    check("stand-in checkpoint written and hashed", STANDIN_PATH.exists() and bool(ckpt_hash))
    n_params_full = sum(p.numel() for p in standin.parameters())
    print(f"  full (unpruned) params: {n_params_full:,}")
    del standin  # free before the subprocess calls below

    # ---- 2. prune at BOTH frozen ratios, in ISOLATED subprocesses (thop, A43) -----
    print("\n---------------- pruning at both frozen ratios (subprocess-isolated) ----------------")
    records: dict[float, dict] = {}
    for ratio in RATIOS:
        rp = _record_path(ratio)
        rc = None
        for attempt in range(1, MEASURE_RATIO_MAX_ATTEMPTS + 1):
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                               "--phase", "measure_ratio", "--ratio", str(ratio),
                               "--min-keep", str(MIN_KEEP)])
            rc = r.returncode
            if rc == 0 and rp.exists():
                break
            print(f"  [WARN] ratio {ratio}: measure_ratio attempt {attempt}/"
                  f"{MEASURE_RATIO_MAX_ATTEMPTS} exited {rc} (0x{rc & 0xFFFFFFFF:08X}) -- "
                  f"A43's intermittent thop/torch crash on this host; retrying")
        check(f"ratio {ratio}: measure_ratio subprocess eventually exited 0 "
              f"(within {MEASURE_RATIO_MAX_ATTEMPTS} attempts)", rc == 0)
        if not check(f"ratio {ratio}: record file written", rp.exists()):
            continue
        record = json.loads(rp.read_text(encoding="utf-8"))
        check(f"ratio {ratio}: measured params reduced "
              f"({record['params_before']:,} -> {record['params_after']:,})",
              record["params_after"] < record["params_before"])
        check(f"ratio {ratio}: measured FLOPs reduced "
              f"({record['flops_before']:,} -> {record['flops_after']:,})",
              record["flops_after"] < record["flops_before"])
        check(f"ratio {ratio}: mask file written", _mask_path(ratio).exists())
        records[ratio] = record

    if len(records) != len(RATIOS):
        check("all ratios produced a usable record (cannot continue otherwise)", False)
        return 1

    # Rebuild the pruned MODEL objects in-process for the remaining checks -- this is a plain
    # weight-transplant (apply_channel_mask), NOT thop, so it is not subject to A43 at all.
    pruner = StructuredPruner()
    models: dict[float, nn.Module] = {}
    for ratio in RATIOS:
        loaded = load_channel_mask(_mask_path(ratio))
        fresh = UNet.from_config({"model": ARCH})
        load_checkpoint(STANDIN_PATH, model=fresh, map_location="cpu")
        models[ratio] = pruner.apply(fresh, loaded["mask"])

    # ---- 3. mask reload reproduces identical parameter count ----------------------
    print("\n---------------- mask reload determinism ----------------")
    for ratio in RATIOS:
        n1 = sum(p.numel() for p in models[ratio].parameters())
        n2 = records[ratio]["params_after"]
        check(f"ratio {ratio}: reloaded/reapplied mask reproduces IDENTICAL param count "
              f"({n1:,} == {n2:,})", n1 == n2)

    # ---- 4. BN channels match Conv channels in the pruned model -------------------
    print("\n---------------- BatchNorm/Conv channel alignment ----------------")
    for ratio in RATIOS:
        model = models[ratio]
        mismatches = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                pairs = list(module)
                for i in range(len(pairs) - 1):
                    a, b = pairs[i], pairs[i + 1]
                    if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                        if a.out_channels != b.num_features:
                            mismatches.append((name, a.out_channels, b.num_features))
        check(f"ratio {ratio}: every Conv2d's out_channels matches its paired BatchNorm's "
              f"num_features ({len(mismatches)} mismatch(es): {mismatches or 'none'})",
              not mismatches)

    # ---- 5. mask-reuse: B/B'/C would share one hash --------------------------------
    # No thop here -- load the SAME mask file 3 times (as each of B/B'/C independently would)
    # and confirm every load reports the identical hash; also rebuild the pruned model from
    # each load and confirm the hash of ITS OWN mask matches too (round-trip, not just re-read).
    print("\n---------------- mask-reuse (B/B'/C would share one hash) ----------------")
    primary_ratio = RATIOS[0]
    shared_mask_path = _mask_path(primary_ratio)
    hashes = []
    for label in ("B", "B_prime", "C"):
        loaded = load_channel_mask(shared_mask_path)
        recomputed = compute_mask_hash(loaded["mask"])
        check(f"  {label}: reloaded mask's stored hash matches its own recomputed hash",
              loaded["hash"] == recomputed)
        hashes.append(loaded["hash"])
        print(f"  {label}: mask_hash={loaded['hash'][:16]}...")
    check(f"B/B'/C loading the same mask_path all report the IDENTICAL hash "
          f"({len(set(hashes))}/3 distinct)", len(set(hashes)) == 1)

    # ---- 6. pruned model runs inference without shape error -----------------------
    print("\n---------------- inference shape check ----------------")
    for ratio in RATIOS:
        model = models[ratio]
        model.eval()
        try:
            with torch.no_grad():
                out = model(torch.randn(1, 1, 256, 256))
            ok = tuple(out.shape) == (1, 1, 256, 256) and bool(torch.isfinite(out).all())
        except Exception as e:  # noqa: BLE001
            print(f"  ratio {ratio}: inference FAILED: {type(e).__name__}: {e}")
            ok = False
        check(f"ratio {ratio}: pruned model runs inference without a shape error, output finite",
              ok)

    # ---- 7. BN recalibration changes stats and is deterministic -------------------
    print("\n---------------- BatchNorm recalibration determinism ----------------")
    base_model = models[primary_ratio]

    def synthetic_batches(seed: int, n: int = 8) -> list[dict]:
        g = torch.Generator().manual_seed(seed)
        return [{"image": torch.randn(2, 1, 256, 256, generator=g)} for _ in range(n)]

    bn_before = [m.running_mean.clone() for m in base_model.modules()
                if isinstance(m, nn.BatchNorm2d)]

    model_a = copy.deepcopy(base_model)
    model_b = copy.deepcopy(base_model)
    rec_a = recalibrate_batchnorm(model_a, synthetic_batches(123), num_batches=8,
                                  momentum=None, reset_first=True)
    rec_b = recalibrate_batchnorm(model_b, synthetic_batches(123), num_batches=8,
                                  momentum=None, reset_first=True)

    means_a = [m.running_mean for m in model_a.modules() if isinstance(m, nn.BatchNorm2d)]
    means_b = [m.running_mean for m in model_b.modules() if isinstance(m, nn.BatchNorm2d)]
    identical = all(torch.equal(a, b) for a, b in zip(means_a, means_b))
    check("BN recalibration is DETERMINISTIC: identical batch stream -> identical running stats",
          identical)
    changed = any(not torch.equal(a, b) for a, b in zip(bn_before, means_a))
    check("BN recalibration actually CHANGES the running stats (not a no-op)", changed)
    check("recalibration batch-order fingerprint matches across the two deterministic runs",
          rec_a["batch_order_fingerprint"] == rec_b["batch_order_fingerprint"])

    # ---- 8. the ACTUAL Kaggle runner path (experiments/multi_seed_train.py) -------
    print("\n---------------- multi_seed_train.py's pruned-base path ----------------")
    print("(closes the gap found while authoring Phase 7-RUN: the runner previously only ever")
    print(" built a FRESH unpruned model -- see A42)")
    try:
        from experiments.multi_seed_train import _build_pruned_base

        manifest = pd.read_csv(ROOT / "data" / "processed" / "preprocess_manifest.csv")
        tiny_frame = (manifest[manifest["split"] == "train"]
                     .sort_values("image_id").head(4).reset_index(drop=True))
        fake_train_loader = SimpleNamespace(dataset=SimpleNamespace(frame=tiny_frame))

        b_cfg = compose_run_config("model_b_pruned.yaml")
        b_cfg["model"] = dict(ARCH)
        b_cfg["pruning"]["source_checkpoint"] = "SYNTHETIC_standin_model_a/best.pt"
        b_cfg["pruning"]["channel_mask"] = f"SYNTHETIC_phase7_channel_mask_ratio_{primary_ratio}.json"
        b_cfg["pruning"]["ratio"] = primary_ratio
        b_cfg["training"]["batch_size"] = 2
        b_cfg["pruning"]["bn_recalibration"]["num_batches"] = 4  # keep the synthetic test fast

        args_stub = SimpleNamespace(output_root=str(OUT_DIR), device="cpu",
                                    channel_mask_hash=records[primary_ratio]["mask_hash"])
        runner_model, runner_note = _build_pruned_base(b_cfg, b_cfg["pruning"], args_stub,
                                                        fake_train_loader)
        print(f"  {runner_note}")
        n_runner = sum(p.numel() for p in runner_model.parameters())
        n_direct = sum(p.numel() for p in models[primary_ratio].parameters())
        check("multi_seed_train.py::_build_pruned_base() succeeds against the synthetic fixture",
              runner_model is not None)
        check(f"runner-produced pruned model has the SAME param count as the direct "
              f"StructuredPruner path ({n_runner:,} == {n_direct:,})", n_runner == n_direct)

        # Wrong hash must be REJECTED, not silently accepted (master §9 provability).
        bad_args = SimpleNamespace(output_root=str(OUT_DIR), device="cpu",
                                   channel_mask_hash="0" * 64)
        try:
            _build_pruned_base(b_cfg, b_cfg["pruning"], bad_args, fake_train_loader)
            check("a WRONG --channel-mask-hash is rejected (raises), not silently accepted", False)
        except ValueError:
            check("a WRONG --channel-mask-hash is rejected (raises), not silently accepted", True)
    except Exception as e:  # noqa: BLE001
        check(f"multi_seed_train.py's pruned-base path ran without raising "
              f"({type(e).__name__}: {e})", False)

    # ---- write the SYNTHETIC validation report -------------------------------------
    report = {
        "phase": 7,
        "synthetic": True,
        "warning": "Stand-in checkpoint is an UNTRAINED random-init UNet. This report verifies "
                  "pruning MECHANICS only and makes no claim about real model quality.",
        "architecture": ARCH,
        "ratios_tested": RATIOS,
        "n_failed": len(failures),
        "n_warnings": len(warnings),
        "failures": failures,
    }
    report_path = OUT_DIR / "SYNTHETIC_phase7_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {report_path.relative_to(ROOT)}")

    print()
    print("=" * 78)
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    if failures:
        print(f"RESULT: FAIL ({len(failures)} check(s) failed)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: PASS -- Phase 7 pruning MECHANICS verified (synthetic, untrained stand-in).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
