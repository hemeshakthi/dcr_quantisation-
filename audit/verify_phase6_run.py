"""Phase 6-RUN verification: run AFTER bringing checkpoints/logs back from Kaggle.

This does NOT run any training. It verifies the real artifacts produced by
experiments/kaggle_train_vanilla_a.ipynb once they have been downloaded and placed at the
repo's normal checkpoints/ and logs/ paths (docs/kaggle_run_contract.md §5).

Required checks (Phase 6-RUN prompt):
  * all checkpoints load
  * Vanilla vs A produce DIFFERENT outputs on a probe image
    (identical = the model-collapse bug that killed a previous attempt at this project --
    master Golden Rule 2 -- investigate, never accept)
  * experiment_log is complete

If checkpoints/ is empty, this reports PARTIAL with exact instructions, mirroring the pattern
used by every other "bring artifacts back" phase in this project.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

failures, warnings = [], []


def check(label, cond, warn_only=False):
    ok = bool(cond)
    print(f"[{'PASS' if ok else ('WARN' if warn_only else 'FAIL')}] {label}")
    if not ok:
        (warnings if warn_only else failures).append(label)
    return ok


print("=" * 78)
print("PHASE 6-RUN VERIFICATION -- real Kaggle artifacts (Vanilla x3, A_dcr x3)")
print("=" * 78)

from src.training.checkpointing import load_checkpoint  # noqa: E402
from src.training.experiment_log import audit_completeness  # noqa: E402
from src.models.unet import UNet  # noqa: E402
from src.datasets.dataset_registry import DATA_PROCESSED  # noqa: E402

EXPECTED_RUNS = [f"{m}_seed{s}" for m in ("vanilla", "A_dcr") for s in (1, 2, 3)]
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"

# ---- 0. presence -------------------------------------------------------------------
present = {r: (CKPT_DIR / r) for r in EXPECTED_RUNS if (CKPT_DIR / r).is_dir()}
if not present:
    print(f"\nRESULT: PARTIAL -- no run directories found under {CKPT_DIR}")
    print("\nExpected, after downloading phase6_run_artifacts.zip from Kaggle and unzipping:")
    for r in EXPECTED_RUNS:
        print(f"  checkpoints/{r}/best.pt")
        print(f"  checkpoints/{r}/final.pt")
        print(f"  logs/{r}/epochs.csv")
    print(f"  results/tables/experiment_log.csv  (merge the Kaggle rows into the repo's copy)")
    print("\nPlace the unzipped 'checkpoints/' and 'logs/' folders directly at the repo root,")
    print("then re-run this script.")
    sys.exit(1)

check(f"found {len(present)}/6 expected run directories under checkpoints/",
      len(present) == 6)
for r in EXPECTED_RUNS:
    check(f"  checkpoints/{r}/ present", r in present)

# ---- 1. every checkpoint loads ------------------------------------------------------
print("\n---------------- checkpoint loading ----------------")
loaded: dict[str, dict] = {}
for run_id, d in present.items():
    for kind in ("best", "final"):
        p = d / f"{kind}.pt"
        if not p.is_file():
            check(f"  {run_id}/{kind}.pt exists", False)
            continue
        try:
            ckpt = load_checkpoint(p, map_location="cpu")
            ok = "model_state" in ckpt and bool(ckpt["model_state"])
        except Exception as e:
            print(f"       {run_id}/{kind}.pt: {type(e).__name__}: {e}")
            ok = False
        check(f"  {run_id}/{kind}.pt loads (has model_state)", ok)
        if ok and kind == "best":
            loaded[run_id] = ckpt

# ---- 2. rebuild each model from its OWN stored config and run inference -----------
print("\n---------------- probe-image inference ----------------")
man_path = DATA_PROCESSED / "preprocess_manifest.csv"
probe_path = None
if man_path.exists():
    import pandas as pd
    man = pd.read_csv(man_path)
    test_rows = man[man["split"] == "test"]
    if len(test_rows):
        probe_path = ROOT / test_rows.iloc[0]["proc_image"]

if probe_path is None or not probe_path.exists():
    print("  no processed test image found locally -- using a fixed-seed random probe instead "
        "(still valid for the identical-output check, which only compares models against "
        "EACH OTHER on one shared input)")
    probe = torch.from_numpy(
        np.random.default_rng(42).standard_normal((1, 1, 1024, 1024)).astype("float32"))
else:
    arr = np.load(probe_path).astype("float32")
    probe = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    print(f"  probe image: {probe_path.relative_to(ROOT)}  shape={tuple(probe.shape)}")

outputs: dict[str, torch.Tensor] = {}
for run_id, ckpt in loaded.items():
    try:
        model = UNet.from_config(ckpt["config"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(probe))
        outputs[run_id] = probs
        ok = True
    except Exception as e:
        print(f"  {run_id}: inference FAILED: {type(e).__name__}: {e}")
        ok = False
    check(f"  {run_id}: rebuilds from its stored config and runs inference", ok)

# ---- 3. Golden Rule 2: Vanilla vs A must NOT be identical --------------------------
print("\n---------------- Vanilla vs A: identical-output check (Golden Rule 2) ----------------")
vanilla_runs = [r for r in outputs if r.startswith("vanilla_")]
a_runs = [r for r in outputs if r.startswith("A_dcr_")]
identical_pairs, near_identical_pairs = [], []
for v in vanilla_runs:
    for a in a_runs:
        d = float((outputs[v] - outputs[a]).abs().mean())
        if d == 0.0:
            identical_pairs.append((v, a, d))
        elif d < 1e-6:
            near_identical_pairs.append((v, a, d))
        print(f"  {v:<16} vs {a:<16}  mean|diff| = {d:.6e}")

check(f"NO byte-identical Vanilla/A output pairs (found {len(identical_pairs)})",
      not identical_pairs)
check(f"NO suspiciously-near-identical pairs, mean|diff|<1e-6 (found {len(near_identical_pairs)})",
      not near_identical_pairs, warn_only=True)
if identical_pairs or near_identical_pairs:
    print("\n  *** MODEL COLLAPSE SUSPECTED (master Golden Rule 2) ***")
    print("  Vanilla and A were trained with different losses; identical predictions mean")
    print("  either (a) the dCR loss term had zero effect on training, (b) a data/label bug")
    print("  made all runs see the same trivial target, or (c) checkpoints were mixed up.")
    print("  Do NOT proceed to Phase 7+ until this is diagnosed.")

# Same-model, different-seed: informational only (some similarity is EXPECTED, since only
# initialisation differs), but all-zero variance here would itself be a red flag.
same_model_diffs = []
for group, label in ((vanilla_runs, "vanilla"), (a_runs, "A_dcr")):
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            d = float((outputs[group[i]] - outputs[group[j]]).abs().mean())
            same_model_diffs.append(d)
            print(f"  [info] {group[i]:<16} vs {group[j]:<16}  mean|diff| = {d:.6e}  (same model, different seed)")
if same_model_diffs:
    check("different SEEDS of the same model are not all byte-identical to each other "
          "(would indicate seeding is not actually varying anything)",
          any(d > 0 for d in same_model_diffs), warn_only=True)

# ---- 4. experiment_log completeness -------------------------------------------------
print("\n---------------- experiment_log completeness ----------------")
audit = audit_completeness(ROOT / "results" / "tables" / "experiment_log.csv", expected=6)
print(f"  rows: {audit.get('n_rows')}  expected: 6  complete: {audit.get('complete')}")
check("experiment_log.csv has at least 6 rows", audit.get("n_rows", 0) >= 6)
check("no duplicate (model, ratio, seed) rows", not audit.get("duplicates"))
check("no rows carry a '-dirty' git commit hash (reproducibility)",
      not audit.get("dirty_commits"), warn_only=True)

import csv
log_path = ROOT / "results" / "tables" / "experiment_log.csv"
if log_path.exists():
    with open(log_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    got_ids = {r.get("run_id") for r in rows}
    missing = set(EXPECTED_RUNS) - got_ids
    check(f"all 6 expected run_ids present in experiment_log.csv (missing: {missing or 'none'})",
          not missing)
    for r in rows:
        if r.get("run_id") in EXPECTED_RUNS:
            if r["run_id"].startswith("A_dcr_"):
                check(f"  {r['run_id']}: curvature_smooth_sigma == 2.0 (A30 drift guard)",
                      r.get("curvature_smooth_sigma") in ("2.0", "2"))
            check(f"  {r['run_id']}: status is completed/early_stopped, not a crash",
                  r.get("status") in ("completed", "early_stopped"))
            if str(r.get("notes", "")).strip():
                print(f"  [note] {r['run_id']}: {r['notes']}")

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
    sys.exit(1)
print("RESULT: PASS -- Phase 6-RUN artifacts verified.")
print("=" * 78)
