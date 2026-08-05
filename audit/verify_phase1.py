"""Phase 1 verification: data acquisition, quality control, and splitting.

Verifies the REAL artefacts on disk. Nothing here is asserted from memory -- every count is
recomputed from the files the pipeline actually wrote, and the leakage check is re-derived
independently of make_split.py so a bug there cannot certify itself.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

failures, warnings = [], []


def check(label, condition, warn_only=False):
    ok = bool(condition)
    print(f"[{'PASS' if ok else ('WARN' if warn_only else 'FAIL')}] {label}")
    if not ok:
        (warnings if warn_only else failures).append(label)
    return ok


print("=" * 78)
print("PHASE 1 VERIFICATION")
print("=" * 78)

# ---- 1. scripts exist -------------------------------------------------------------
for f in ["dataset_registry.py", "download_datasets.py", "quality_check.py", "make_split.py"]:
    check(f"script exists: src/datasets/{f}", (ROOT / "src" / "datasets" / f).is_file())

# ---- 2. registry sanity -----------------------------------------------------------
from src.datasets.dataset_registry import (  # noqa: E402
    REGISTRY, PRIMARY_CANDIDATES, BCCD_MASK_TODO_BANNER, PROFESSOR_ACTION_BANNER,
)

check("registry defines the 2 paper-cited Zenodo datasets as primary candidates",
      set(PRIMARY_CANDIDATES) == {"stardist_tcell", "stardist_flowchamber"})
check("BCCD is present in the registry but MANUAL-only (never auto-downloaded)",
      REGISTRY["bccd"].acquisition == "MANUAL")
check("every non-Zenodo candidate is MANUAL-only",
      all(s.acquisition == "MANUAL" for k, s in REGISTRY.items()
          if k not in PRIMARY_CANDIDATES))
check("BCCD TODO banner exists and forbids fabricating masks from boxes",
      "NOT FABRICATE" in BCCD_MASK_TODO_BANNER.upper()
      and "BOUNDING-BOX" in BCCD_MASK_TODO_BANNER.upper())
check("professor action banner names both Zenodo records",
      "4034939" in PROFESSOR_ACTION_BANNER and "4034929" in PROFESSOR_ACTION_BANNER)
check("every registry entry carries a source label",
      all(any(t in s.confidence for t in ("[PAPER", "[ASSUMPTION", "[LITERATURE", "[OFFICIAL"))
          for s in REGISTRY.values()))

# ---- 3. raw data + _meta.json -----------------------------------------------------
acquired = []
for key in PRIMARY_CANDIDATES:
    spec = REGISTRY[key]
    mp = spec.meta_path
    if not check(f"_meta.json exists: data/raw/{key}/_meta.json", mp.is_file()):
        continue
    meta = json.loads(mp.read_text(encoding="utf-8"))
    check(f"  {key}: _meta.json records source_url", bool(meta.get("source_url")))
    check(f"  {key}: _meta.json records license", bool(meta.get("license")))
    check(f"  {key}: _meta.json records confidence label",
          bool(meta.get("confidence_label")))
    check(f"  {key}: professor confirmation flagged as NOT received",
          meta.get("professor_confirmation_received") is False)
    if meta.get("status") == "ACQUIRED":
        acquired.append(key)
    else:
        check(f"  {key}: status is ACQUIRED (got {meta.get('status')})", False, warn_only=True)

# ---- 4. quality-check artefacts ---------------------------------------------------
inventories = {}
for key in acquired:
    inv = ROOT / "data" / "processed" / f"{key}_inventory.csv"
    dup = ROOT / "data" / "processed" / f"{key}_nearduplicates.json"
    if not check(f"inventory exists: data/processed/{key}_inventory.csv", inv.is_file()):
        continue
    df = pd.read_csv(inv)
    inventories[key] = df
    check(f"  {key}: inventory non-empty ({len(df)} rows)", len(df) > 0)
    check(f"  {key}: image count == mask count (every row has both paths)",
          df["image_path"].notna().all() and df["mask_path"].notna().all())
    check(f"  {key}: no shape mismatches", bool(df["shape_match"].all()))
    check(f"  {key}: all referenced image files exist on disk",
          all((ROOT / p).is_file() for p in df["image_path"]))
    check(f"  {key}: all referenced mask files exist on disk",
          all((ROOT / p).is_file() for p in df["mask_path"]))
    check(f"  {key}: duplicate report exists", dup.is_file())
    if dup.is_file():
        info = json.loads(dup.read_text(encoding="utf-8"))
        check(f"  {key}: duplicate report records a real comparison count "
              f"({info.get('n_pairs_compared')} pairs)",
              int(info.get("n_pairs_compared", 0)) > 0)
        check(f"  {key}: duplicate report records series groups",
              len(info.get("series_groups", {})) > 0)

# ---- 5. overlays ------------------------------------------------------------------
for key in acquired:
    d = ROOT / "results" / "figures" / "qc" / key
    pngs = sorted(d.glob("overlay_*.png")) if d.is_dir() else []
    check(f"  {key}: 10 QC overlays saved (found {len(pngs)})", len(pngs) == 10)
    check(f"  {key}: overlay index.tsv exists (traceability)", (d / "index.tsv").is_file())
    check(f"  {key}: overlays are non-empty files",
          all(p.stat().st_size > 1000 for p in pngs))

# ---- 6. splits --------------------------------------------------------------------
train_csv = ROOT / "data" / "splits" / "train.csv"
test_csv = ROOT / "data" / "splits" / "test.csv"
man_path = ROOT / "data" / "splits" / "split_manifest.json"

if check("data/splits/train.csv exists", train_csv.is_file()) and \
        check("data/splits/test.csv exists", test_csv.is_file()):
    tr, te = pd.read_csv(train_csv), pd.read_csv(test_csv)
    check(f"train.csv non-empty ({len(tr)} rows)", len(tr) > 0)
    check(f"test.csv non-empty ({len(te)} rows)", len(te) > 0)

    # --- independent leakage re-derivation (does NOT trust make_split.py) ---
    overlap = set(tr["image_id"]) & set(te["image_id"])
    check(f"no image_id appears in both splits (overlap={len(overlap)})", not overlap)

    tr_paths = set(tr["image_path"]) | set(tr["mask_path"])
    te_paths = set(te["image_path"]) | set(te["mask_path"])
    check(f"no file path shared across splits (overlap={len(tr_paths & te_paths)})",
          not (tr_paths & te_paths))

    grp_overlap = set(tr["group_id"]) & set(te["group_id"])
    check(f"no atomic group spans splits (overlap={len(grp_overlap)})", not grp_overlap)

    ser_overlap = set(tr["series_key"]) & set(te["series_key"])
    check(f"no acquisition series spans splits (overlap={len(ser_overlap)})",
          not ser_overlap)

    # near-duplicate edges must not cross the split
    assign = {}
    for _, r in tr.iterrows():
        assign[r["image_id"]] = "train"
    for _, r in te.iterrows():
        assign[r["image_id"]] = "test"
    crossing = 0
    for key in acquired:
        dup = ROOT / "data" / "processed" / f"{key}_nearduplicates.json"
        if not dup.is_file():
            continue
        info = json.loads(dup.read_text(encoding="utf-8"))
        for e in info.get("near_duplicate_pairs", []):
            a, b = assign.get(e["a"]), assign.get(e["b"])
            if a and b and a != b:
                crossing += 1
    check(f"no near-duplicate pair spans splits (crossing={crossing})", crossing == 0)

    # every split row traces to the inventory
    if inventories:
        all_inv = pd.concat(inventories.values(), ignore_index=True)
        known = set(all_inv["image_id"])
        unknown = (set(tr["image_id"]) | set(te["image_id"])) - known
        check(f"every split row traces to a QC inventory row (untraceable={len(unknown)})",
              not unknown)

if check("data/splits/split_manifest.json exists", man_path.is_file()):
    man = json.loads(man_path.read_text(encoding="utf-8"))
    check("manifest logs the seed", man.get("seed") is not None)
    check("manifest logs the leakage check as PASSED", man.get("leakage_check") == "PASSED")
    check("manifest logs the grouping rule", bool(man.get("grouping_rule")))
    check("manifest logs the achieved-vs-target ratio deviation",
          man.get("train_fraction_deviation") is not None)
    check("manifest carries source labels", bool(man.get("source_labels")))
    check("manifest warns that dataset identity is UNCONFIRMED",
          "UNCONFIRMED" in str(man.get("warning", "")).upper())

# ---- 7. assumptions logged --------------------------------------------------------
ass = ROOT / "docs" / "implementation_assumptions.md"
text = ass.read_text(encoding="utf-8") if ass.is_file() else ""
for aid in [f"A{i:02d}" for i in range(9, 19)]:
    check(f"assumption {aid} logged", f"### {aid} " in text)
for topic in ["near-duplicate", "binaris", "stratif", "BCCD", "leakage"]:
    check(f"assumptions cover topic: {topic}", topic.lower() in text.lower())
check("BCCD mask fabrication explicitly forbidden in the assumptions log",
      "REJECTED AND FORBIDDEN" in text)

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
print("RESULT: PASS — Phase 1 verification complete.")
print("=" * 78)
print("REMINDER: dataset identity remains an [ASSUMPTION] (A01/A09) until the professor")
print("confirms it. A17 (split stratification) requires an explicit professor decision.")
