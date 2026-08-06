"""Phase 3 verification: module skeletons and configuration.

Beyond the stated requirements (every module imports, every YAML loads, no out-of-scope
files), this also cross-checks the config INVARIANTS that later phases silently depend on --
B/B'/C sharing a channel mask, B' and C sharing a band width, everything sharing optimiser and
seeds. Those are exactly the things that break without raising an error, so they are checked
here rather than discovered after 24 training runs.
"""
import importlib
import inspect
import re
import subprocess
import sys
from pathlib import Path

import yaml

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
print("PHASE 3 VERIFICATION")
print("=" * 78)

# ---- 1. every module imports ------------------------------------------------------
MODULES = [
    "src.config",
    "src.datasets.dataset_registry", "src.datasets.download_datasets",
    "src.datasets.quality_check", "src.datasets.make_split",
    "src.datasets.preprocess", "src.datasets.augmentations",
    "src.datasets.curvature_stratification", "src.datasets.dataset",
    "src.models.unet",
    "src.losses.boundary_band", "src.losses.dice", "src.losses.dcr",
    "src.losses.generic_boundary", "src.losses.combined",
    "src.compression.channel_mask", "src.compression.l1_structured",
    "src.compression.bn_recalibration", "src.compression.complexity",
    "src.metrics.segmentation", "src.metrics.per_image",
    "src.failure.detectors",
    "src.training.determinism", "src.training.checkpointing",
    "src.training.trainer", "src.training.experiment_log",
    "src.evaluation.evaluate", "src.evaluation.statistics",
    "src.visualization.contours", "src.visualization.figures",
    "src.visualization.visual_gate",
]
imported = {}
for m in MODULES:
    try:
        imported[m] = importlib.import_module(m)
        ok = True
    except Exception as e:
        print(f"       import error: {type(e).__name__}: {e}")
        ok = False
    check(f"imports: {m}", ok)

# ---- 2. skeletons fail loudly rather than returning plausible values --------------
# A stub that returned 0.0 or an empty dict would be a placeholder producing fake results
# (master §1, Golden Rule 1). Every STILL-UNIMPLEMENTED body must raise NotImplementedError.
# Modules receive real logic phase by phase; implemented ones are exempted here because they
# are covered by their own phase verifier instead.
IMPLEMENTED_MODULES = {
    "src.models.unet",                          # Phase 4 -> audit/verify_phase4.py
    "src.datasets.dataset_registry", "src.datasets.download_datasets",
    "src.datasets.quality_check", "src.datasets.make_split",
    "src.datasets.preprocess", "src.datasets.augmentations",
    "src.datasets.curvature_stratification",    # Phases 1-2 -> verify_phase1/2.py
}
stub_total, stub_raising = 0, 0
for name, mod in imported.items():
    if name in IMPLEMENTED_MODULES:
        continue
    if not name.startswith(("src.models", "src.losses", "src.compression", "src.metrics",
                            "src.failure", "src.training", "src.evaluation",
                            "src.visualization")) and name != "src.config":
        continue
    for _, obj in inspect.getmembers(mod):
        fns = []
        if inspect.isfunction(obj) and obj.__module__ == name:
            fns = [obj]
        elif inspect.isclass(obj) and obj.__module__ == name:
            fns = [f for _, f in inspect.getmembers(obj, inspect.isfunction)
                   if f.__module__ == name and f.__name__ != "__init__"]
        for fn in fns:
            try:
                src = inspect.getsource(fn)
            except OSError:
                continue
            body = src.split('"""')[-1] if '"""' in src else src
            stub_total += 1
            if "NotImplementedError" in body:
                stub_raising += 1
check(f"all {stub_total} skeleton callables raise NotImplementedError "
      f"(no silent placeholder returns) -- {stub_raising}/{stub_total}",
      stub_total > 0 and stub_raising == stub_total)

# ---- 3. every YAML loads ----------------------------------------------------------
EXPECTED_CONFIGS = [
    "model_vanilla.yaml", "model_a_dcr.yaml", "model_b_pruned.yaml",
    "model_b_prime_generic.yaml", "model_c_pruned_dcr.yaml",
    "training_shared.yaml", "pruning.yaml", "evaluation.yaml",
]
cfgs = {}
for name in EXPECTED_CONFIGS:
    p = ROOT / "configs" / name
    if not check(f"config exists: configs/{name}", p.is_file()):
        continue
    try:
        cfgs[name] = yaml.safe_load(p.read_text(encoding="utf-8"))
        ok = isinstance(cfgs[name], dict) and len(cfgs[name]) > 0
    except Exception as e:
        print(f"       YAML error: {type(e).__name__}: {e}")
        ok = False
    check(f"  loads and is a non-empty mapping: {name}", ok)

check(f"exactly the 8 required configs present (found {len(list((ROOT/'configs').glob('*.yaml')))})",
      len(list((ROOT / "configs").glob("*.yaml"))) == 8)

# ---- 4. source labels present -----------------------------------------------------
for name, cfg in cfgs.items():
    check(f"  {name} carries a source_labels block", bool(cfg.get("source_labels")))
    raw = (ROOT / "configs" / name).read_text(encoding="utf-8")
    n_labels = len(re.findall(r"\[(PAPER|OFFICIAL|LITERATURE|ENGINEERING|ASSUMPTION)", raw))
    check(f"  {name} annotates values with source labels ({n_labels} found)", n_labels >= 5)

# ---- 5. FROZEN invariants across the model configs --------------------------------
models = {k: v for k, v in cfgs.items() if k.startswith("model_")}
check("all 5 model configs loaded", len(models) == 5)

for name, cfg in models.items():
    m = cfg.get("model", {})
    check(f"  {name}: out_channels == 1 (master §7 FROZEN)", m.get("out_channels") == 1)
    check(f"  {name}: no output activation (master §7 FROZEN)",
          m.get("final_activation", "MISSING") is None)
    check(f"  {name}: architecture is plain unet (master §1/§7)", m.get("architecture") == "unet")
    check(f"  {name}: batchnorm enabled (required by the §9 protocol)", m.get("batchnorm") is True)

# B / B' / C must be the SAME NETWORK -- same mask, same ratio, same source checkpoint.
bbc = {k: cfgs[k] for k in ("model_b_pruned.yaml", "model_b_prime_generic.yaml",
                            "model_c_pruned_dcr.yaml") if k in cfgs}
if len(bbc) == 3:
    masks = {k: v["pruning"]["channel_mask"] for k, v in bbc.items()}
    check(f"B/B'/C load the IDENTICAL channel mask (master §9) -- {set(masks.values())}",
          len(set(masks.values())) == 1)
    check("B/B'/C use the identical pruning ratio",
          len({v["pruning"]["ratio"] for v in bbc.values()}) == 1)
    check("B/B'/C prune from the identical source checkpoint",
          len({v["pruning"]["source_checkpoint"] for v in bbc.values()}) == 1)
    check("B/B'/C all set reprune: false (prune ONCE per ratio, master §9)",
          all(v["pruning"]["reprune"] is False for v in bbc.values()))
    check("B/B'/C all recalibrate BatchNorm (master §9)",
          all(v["pruning"]["recalibrate_batchnorm"] is True for v in bbc.values()))
    check("B/B'/C share the identical region-loss alpha (only the boundary term may differ)",
          len({v["loss"]["region"]["alpha"] for v in bbc.values()}) == 1)
    check("B has no boundary term; B' and C do",
          bbc["model_b_pruned.yaml"]["loss"]["boundary"]["name"] is None
          and bbc["model_b_prime_generic.yaml"]["loss"]["boundary"]["name"] is not None
          and bbc["model_c_pruned_dcr.yaml"]["loss"]["boundary"]["name"] is not None)

    # P5 hinges on these two: same band, gradient-matched strength.
    bw_bp = bbc["model_b_prime_generic.yaml"]["loss"]["boundary"]["band_width"]
    bw_c = bbc["model_c_pruned_dcr.yaml"]["loss"]["boundary"]["band_width"]
    check(f"B' and C share band_width ({bw_bp} vs {bw_c}) -- master §8, or P5 is confounded",
          bw_bp == bw_c)
    check("B' beta is gradient-calibrated against C, not hand-set (master §8)",
          bbc["model_b_prime_generic.yaml"]["loss"]["boundary"]["beta"] is None
          and bbc["model_b_prime_generic.yaml"]["loss"]["boundary"]
              .get("calibrate_beta_by_gradient_norm") is True)
    check("B' and C differ in the boundary loss NAME (that is the whole control)",
          bbc["model_b_prime_generic.yaml"]["loss"]["boundary"]["name"]
          != bbc["model_c_pruned_dcr.yaml"]["loss"]["boundary"]["name"])

# Uncompressed models must not be pruned.
for k in ("model_vanilla.yaml", "model_a_dcr.yaml"):
    if k in cfgs:
        check(f"  {k}: pruning disabled (uncompressed)",
              cfgs[k]["pruning"]["enabled"] is False)

# ---- 6. shared training / pruning / evaluation invariants -------------------------
ts = cfgs.get("training_shared.yaml", {})
check("mixed precision OFF for final runs (master §10)",
      ts.get("training", {}).get("mixed_precision") is False)
check("determinism block fully enabled (master §10)",
      ts.get("determinism", {}).get("cudnn_deterministic") is True
      and ts.get("determinism", {}).get("cudnn_benchmark") is False
      and ts.get("determinism", {}).get("use_deterministic_algorithms") is True)
check("exactly 3 model-init seeds (master §10)",
      len(ts.get("seeds", {}).get("model_init_seeds", [])) == 3)
check("early stopping monitors val_boundary_f1, NOT dice (P1 rationale)",
      ts.get("early_stopping", {}).get("monitor") == "val_boundary_f1")
check("early stopping patience == 15 (master §10)",
      ts.get("early_stopping", {}).get("patience") == 15)
check("image_size == 1024 (A19)", ts.get("data", {}).get("image_size") == 1024)
check("normalisation is percentile p1/p99.8 (A20)",
      ts.get("normalization", {}).get("method") == "percentile"
      and ts.get("normalization", {}).get("pmax") == 99.8)
check("batch_size == 4 (A28, for 1024px on a 16GB T4)",
      ts.get("training", {}).get("batch_size") == 4)
check("augmentation seed is a single shared value (master §10)",
      isinstance(ts.get("augmentation", {}).get("seed"), int))

pr = cfgs.get("pruning.yaml", {}).get("pruning", {})
check("pruning ratios are the frozen [0.5, 0.7] (master §9)", pr.get("ratios") == [0.5, 0.7])
check("pruning criterion is l1_norm (master §9)", pr.get("criterion") == "l1_norm")
check("BatchNorm pruned with its Conv (master §9)",
      pr.get("prune_batchnorm_with_conv") is True)
check("BN recalibration does NOT shuffle (order-dependent statistics, master §9)",
      pr.get("bn_recalibration", {}).get("shuffle") is False)
check("BN recalibration batches within master §9's 1-2k range",
      1000 <= pr.get("bn_recalibration", {}).get("num_batches", 0) <= 2000)
check("complexity asserted identical across B/B'/C with tolerance 0 (master §9)",
      pr.get("complexity", {}).get("assert_identical_across_models") is True
      and pr.get("complexity", {}).get("tolerance") == 0)
check("deterministic tie-break for channel selection (master §10)",
      pr.get("tie_break") == "ascending_index")

ev = cfgs.get("evaluation.yaml", {})
check("Boundary F1 tolerances are 2px and 5px (master §6)",
      ev.get("metrics", {}).get("boundary_f1", {}).get("tolerances_px") == [2.0, 5.0])
check("HD95 percentile is 95 (master §6)",
      ev.get("metrics", {}).get("hd95", {}).get("percentile") == 95)
st = ev.get("statistics", {})
check("Wilcoxon primary, t-test secondary (master §6)",
      st.get("primary_test") == "wilcoxon_signed_rank"
      and st.get("secondary_test") == "paired_t_test")
check("confirmatory family size == 16 and mid bin is descriptive-only (A29)",
      st.get("confirmatory_family_size") == 16
      and st.get("descriptive_only_bins") == ["mid"]
      and st.get("confirmatory_bins") == ["low", "high"])
check("measured test-bin n recorded as low=50 mid=12 high=36 (Phase 2)",
      st.get("power_analysis", {}).get("test_bin_n") == {"low": 50, "mid": 12, "high": 36})
check("TOST on Dice with margin 0.02 (master §4)",
      st.get("tost", {}).get("metric") == "dice" and st.get("tost", {}).get("margin") == 0.02)
sc = ev.get("success_criteria", {})
check("success criteria match master §4 (0.04 / 10% / 0.02 / 6-of-10)",
      sc.get("boundary_f1_min_gain") == 0.04
      and sc.get("hd95_min_relative_reduction") == 0.10
      and sc.get("dice_equivalence_margin") == 0.02
      and sc.get("visual_gate_min_restored") == 6)
check("identical-output check enabled (Golden Rule 2)",
      ev.get("identical_output_check", {}).get("enabled") is True)
fd = ev.get("failure_detectors", {})
check("all six failure detectors configured (master §12)",
      all(k in fd for k in ("corner_collapse", "boundary_smoothing", "cell_merging",
                            "cell_splitting", "thin_structure_loss", "curvature_flattening")))
check("curvature-related vs non-curvature-related declared before results (P6, master §12)",
      len(fd.get("curvature_related", [])) == 3 and len(fd.get("non_curvature_related", [])) == 3)

# ---- 7. OUT-OF-SCOPE scan (master §1) ---------------------------------------------
FORBIDDEN = ["distill", "quant", "onnx", "tensorrt", "betti", "cross_stain",
             "crossstain", "transformer", "attention", "unetpp", "unet++",
             "persistent_homology", "cldice"]
scan_dirs = ["src", "configs", "audit", "tests", "backend", "frontend"]
name_hits, content_hits = [], []
for d in scan_dirs:
    base = ROOT / d
    if not base.exists():
        continue
    for p in base.rglob("*"):
        if not p.is_file() or p.suffix not in {".py", ".yaml", ".yml", ".json", ".ts",
                                               ".tsx", ".js", ".html"}:
            continue
        low = p.name.lower()
        for term in FORBIDDEN:
            if term in low:
                name_hits.append(f"{p.relative_to(ROOT)} (filename contains '{term}')")
        try:
            text = p.read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            continue
        for term in FORBIDDEN:
            for ln, line in enumerate(text.splitlines(), 1):
                if term not in line:
                    continue
                # Prose that NAMES an excluded technique in order to exclude it is required by
                # master §1; only an actual import/definition would be a scope violation.
                if re.search(rf"^\s*(import|from)\s+.*{term}", line) or \
                   re.search(rf"^\s*(class|def)\s+\w*{term}", line):
                    content_hits.append(f"{p.relative_to(ROOT)}:{ln}: {line.strip()[:80]}")

check(f"no out-of-scope FILENAMES (master §1) -- {name_hits or 'none'}", not name_hits)
check(f"no out-of-scope IMPORTS or DEFINITIONS (master §1) -- {content_hits or 'none'}",
      not content_hits)

# ---- 8. repo hygiene --------------------------------------------------------------
ui_files = [p for d in ("backend", "frontend")
            for p in (ROOT / d).rglob("*")
            if p.is_file() and p.suffix in {".py", ".ts", ".tsx", ".js"}]
check(f"no backend/ or frontend/ code yet (UI is master §14, a later phase) -- "
      f"found {len(ui_files)}", not ui_files)
r = subprocess.run([sys.executable, "-m", "compileall", "-q", str(ROOT / "src")],
                   capture_output=True, text=True)
check(f"all of src/ byte-compiles (exit {r.returncode})", r.returncode == 0)

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
print("RESULT: PASS — Phase 3 verification complete.")
print("=" * 78)
print("Skeletons only: every body raises NotImplementedError, so nothing here can")
print("produce a number. Logic lands in Phase 4+.")
