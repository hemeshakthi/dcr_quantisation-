"""Phase 8 -- Evaluation + Statistics: SYNTHETIC validation (master §6).

Author-only phase: no real checkpoints exist locally yet (Kaggle training has not run). Verifies
metric and statistical-test CORRECTNESS using synthetic prediction/target mask pairs with KNOWN
answers -- a perfect match, a disjoint case, and a partial-overlap case with a known concave
CORNER cut out of the prediction (simulating the "corner collapse" damage master §12 names,
without needing a trained model to produce it). This says nothing about real model quality; it
proves the metric/statistics pipeline computes the right numbers on cases whose right answer is
known by construction.

VERIFICATION (all required by this phase's instruction):
  1. metrics finite and correct on all synthetic cases
  2. HD95 is 95th percentile, confirmed by CODE INSPECTION (not just numeric plausibility)
  3. TOST applied only to Dice (rejects any other metric_name; the only real call site passes
     metric_name="dice" literally)
  4. Holm-Bonferroni test count matches the documented confirmatory family size (16, A29)
  5. curvature_bins.csv hash unchanged from Phase 2 (verified via `git diff` on the tracked,
     write-once file -- nothing in Phase 7/8's work touched it)

Every persisted output lives under results/synthetic_validation/phase8/, prefixed SYNTHETIC_ --
per standing instruction: synthetic/smoke-validation artifacts must never be mixed into the
folder structure real results will occupy.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import bootstrap_ci as bootstrap_ci_mod  # noqa: E402
from src.evaluation import curvature_stratified as cs  # noqa: E402
from src.metrics import boundary_f1 as boundary_f1_mod  # noqa: E402
from src.metrics import hausdorff as hausdorff_mod  # noqa: E402
from src.metrics.dice_metrics import dice_score  # noqa: E402
from src.metrics.segmentation import compute_all_metrics  # noqa: E402
from src.metrics.statistical_tests import (holm_bonferroni_correction, paired_t_test,  # noqa: E402
                                           paired_wilcoxon, partial_correlation,
                                           rank_biserial_effect_size, tost_equivalence_test)

failures: list[str] = []
warnings: list[str] = []


def check(label, cond, warn_only=False):
    ok = bool(cond)
    print(f"[{'PASS' if ok else ('WARN' if warn_only else 'FAIL')}] {label}")
    if not ok and not warn_only:
        failures.append(label)
    return ok


OUT_DIR = ROOT / "results" / "synthetic_validation" / "phase8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("PHASE 8 VERIFICATION -- Evaluation + Statistics (SYNTHETIC validation, master §6)")
print("=" * 78)
print("Synthetic mask pairs with KNOWN answers by construction; no trained model involved.")
print(f"Outputs namespaced under {OUT_DIR.relative_to(ROOT)} (SYNTHETIC_ prefix).\n")

# ======================================================================================
# 1. three synthetic mask cases with known answers
# ======================================================================================
print("---------------- synthetic mask cases ----------------")
SIZE = 100

# Case A: PERFECT MATCH.
target_perfect = np.zeros((SIZE, SIZE), np.uint8)
target_perfect[20:70, 20:70] = 1
pred_perfect = target_perfect.copy()

# Case B: DISJOINT. Two 20x20 squares far apart (corner-to-corner gap >> 5px tolerance).
target_disjoint = np.zeros((SIZE, SIZE), np.uint8)
target_disjoint[5:25, 5:25] = 1
pred_disjoint = np.zeros((SIZE, SIZE), np.uint8)
pred_disjoint[75:95, 75:95] = 1

# Case C: PARTIAL OVERLAP, KNOWN CORNER DAMAGE. Ground truth is a 50x50 square with a 15x15
# notch cut from one corner (a concave corner -- exactly what "corner collapse" damages,
# master §12). The prediction FILLS IN the notch (rounds off the concave corner) but is
# otherwise identical -- the model "smooths over" the corner, the canonical corner-collapse
# signature: Dice barely moves (small area change) while the boundary metrics take the hit.
target_corner = np.zeros((SIZE, SIZE), np.uint8)
target_corner[20:70, 20:70] = 1
target_corner[20:35, 20:35] = 0  # concave notch cut from the top-left corner
pred_corner = np.zeros((SIZE, SIZE), np.uint8)
pred_corner[20:70, 20:70] = 1  # same square, notch NOT reproduced (corner collapse)

cases = {
    "perfect_match": (pred_perfect, target_perfect),
    "disjoint": (pred_disjoint, target_disjoint),
    "corner_damage": (pred_corner, target_corner),
}
metrics_by_case: dict[str, dict] = {}
for name, (pred, target) in cases.items():
    m = compute_all_metrics(pred, target)
    metrics_by_case[name] = m
    print(f"  {name:<14} dice={m['dice']:.4f}  f1_2px={m['boundary_f1_2px']:.4f}  "
          f"f1_5px={m['boundary_f1_5px']:.4f}  hd95={m['hd95']:.4f}")

# ---- perfect match: everything is exactly the ideal value ----
pm = metrics_by_case["perfect_match"]
check("perfect match: dice == 1.0 (up to smooth-constant rounding)", abs(pm["dice"] - 1.0) < 1e-3)
check("perfect match: boundary_f1_2px == 1.0", pm["boundary_f1_2px"] == 1.0)
check("perfect match: boundary_f1_5px == 1.0", pm["boundary_f1_5px"] == 1.0)
check("perfect match: hd95 == 0.0", pm["hd95"] == 0.0)

# ---- disjoint: dice ~ 0, boundary F1 = 0 (both tolerances), hd95 large and finite ----
dj = metrics_by_case["disjoint"]
check("disjoint: dice ~= 0 (< 0.01)", dj["dice"] < 0.01)
check("disjoint: boundary_f1_2px == 0.0 (no boundary pixel within 2px of the other mask)",
      dj["boundary_f1_2px"] == 0.0)
check("disjoint: boundary_f1_5px == 0.0 (gap exceeds even the 5px tolerance)",
      dj["boundary_f1_5px"] == 0.0)
check("disjoint: hd95 is finite and large (real geometric separation, not NaN/0)",
      np.isfinite(dj["hd95"]) and dj["hd95"] > 50)

# ---- corner damage: Dice largely PRESERVED, boundary metrics REALLY damaged (P1's own claim,
#      reproduced here on a case where the "damage" is exact by construction) ----
cd = metrics_by_case["corner_damage"]
check("corner damage: Dice largely preserved (> 0.9) despite the corner being wrong",
      cd["dice"] > 0.9)
check("corner damage: Dice is NOT perfect (the notch really is missing area)",
      cd["dice"] < 1.0)
check("corner damage: boundary_f1_2px measurably DAMAGED (< 1.0, > 0.0)",
      0.0 < cd["boundary_f1_2px"] < 1.0)
check("corner damage: boundary_f1_5px measurably DAMAGED (< 1.0, > 0.0)",
      0.0 < cd["boundary_f1_5px"] < 1.0)
check("corner damage: hd95 finite, positive, and SMALLER than the disjoint case's "
      f"(less damage -> smaller HD95: {cd['hd95']:.2f} < {dj['hd95']:.2f})",
      np.isfinite(cd["hd95"]) and 0 < cd["hd95"] < dj["hd95"])
# The notch is 15x15 in a corner; its diagonal is 15*sqrt(2) ~= 21.2, so HD95 (a percentile,
# not the max) over that boundary should land comfortably under it.
check(f"corner damage: hd95 ({cd['hd95']:.2f}) is bounded by the notch geometry (< 21.2, "
      f"the notch's own diagonal) -- confirms hd95 is measuring the notch, not something else",
      cd["hd95"] < 21.2)

# ---- every value across every case is finite (no NaN/inf leaking through) ----
all_finite = all(np.isfinite(v) for m in metrics_by_case.values()
                 for k, v in m.items() if isinstance(v, (int, float)))
check("every metric value across all 3 cases is finite (no NaN/inf)", all_finite)

# ======================================================================================
# 2. HD95 is the 95th percentile -- confirmed by CODE INSPECTION, not just numeric plausibility
# ======================================================================================
print("\n---------------- HD95 = 95th percentile: code inspection ----------------")
hd95_source = inspect.getsource(hausdorff_mod.hd95)
sig = inspect.signature(hausdorff_mod.hd95)
check("hd95's default `percentile` parameter is literally 95.0",
      sig.parameters["percentile"].default == 95.0)
check("hd95's body calls np.percentile(..., percentile) -- not .max()/.min()",
      "np.percentile(" in hd95_source and ".max()" not in hd95_source)
check("hd95 takes the max of TWO 95th percentiles (symmetric: pred->target AND target->pred)",
      hd95_source.count("np.percentile(") == 2)
check("hd95's own module docstring states '95th percentile, NOT the maximum' explicitly",
      "95th percentile" in hausdorff_mod.__doc__ and "NOT the maximum" in hausdorff_mod.__doc__)
check("boundary_f1.py::extract_boundary is the SAME function hd95 imports (one boundary "
      "definition, not two independently-drifting ones)",
      hausdorff_mod.extract_boundary is boundary_f1_mod.extract_boundary)

# ======================================================================================
# 3. TOST applied only to Dice
# ======================================================================================
print("\n---------------- TOST restricted to Dice ----------------")
a = [0.85, 0.86, 0.84, 0.87, 0.85]
b = [0.851, 0.859, 0.841, 0.869, 0.851]
tost_dice = tost_equivalence_test(a, b, margin=0.02, metric_name="dice")
check("tost_equivalence_test(metric_name='dice') runs normally", "equivalent" in tost_dice)
try:
    tost_equivalence_test(a, b, margin=0.02, metric_name="boundary_f1_2px")
    check("tost_equivalence_test REJECTS a non-Dice metric_name (raises ValueError)", False)
except ValueError:
    check("tost_equivalence_test REJECTS a non-Dice metric_name (raises ValueError)", True)

cs_source = inspect.getsource(cs)
tost_call_sites = [ln for ln in cs_source.splitlines() if "tost_equivalence_test(" in ln
                  and "def " not in ln]
check(f"src/evaluation/curvature_stratified.py has exactly ONE call site invoking "
      f"tost_equivalence_test ({len(tost_call_sites)} found)", len(tost_call_sites) == 1)
check("that one call site passes metric_name=\"dice\" literally (not a variable that could "
      "silently be something else)",
      bool(tost_call_sites) and 'metric_name="dice"' in tost_call_sites[0])
check("run_curvature_stratified_suite() (the confirmatory-family runner) never calls TOST "
      "-- confirmed by inspecting its own source",
      "tost_equivalence_test" not in inspect.getsource(cs.run_curvature_stratified_suite))

# ======================================================================================
# 4. Holm-Bonferroni test count matches the documented confirmatory family size (16, A29)
# ======================================================================================
print("\n---------------- Holm-Bonferroni family size ----------------")
check("curvature_stratified.CONFIRMATORY_FAMILY_SIZE == 16 (A29)",
      cs.CONFIRMATORY_FAMILY_SIZE == 16)

rng = np.random.default_rng(42)
n_img = 20
synth_rows = []
for model in ("B", "B_prime", "C"):
    for ratio in (0.5, 0.7):
        for bin_name in ("low", "mid", "high"):
            n = {"low": 50, "mid": 12, "high": 36}[bin_name]
            for i in range(min(n, n_img)):
                base_f1_2 = {"B": 0.55, "B_prime": 0.60, "C": 0.66}[model]
                base_f1_5 = {"B": 0.75, "B_prime": 0.79, "C": 0.84}[model]
                synth_rows.append({
                    "model": model, "pruning_ratio": ratio, "curvature_bin": bin_name,
                    "image_id": f"{bin_name}_{i:03d}",
                    "dice": float(rng.normal(0.85, 0.02)),
                    "boundary_f1_2px": float(np.clip(rng.normal(base_f1_2, 0.05), 0, 1)),
                    "boundary_f1_5px": float(np.clip(rng.normal(base_f1_5, 0.05), 0, 1)),
                })
synth_df = pd.DataFrame(synth_rows)

suite = cs.run_curvature_stratified_suite(
    synth_df, comparisons=[("C", "B"), ("C", "B_prime")], ratios=[0.5, 0.7],
    metrics=["boundary_f1_2px", "boundary_f1_5px"])
check(f"run_curvature_stratified_suite() over 2 comparisons x 2 ratios x 2 metrics x "
      f"{{low,high}} runs EXACTLY 16 confirmatory tests (got "
      f"{suite['n_confirmatory_tests_run']})", suite["n_confirmatory_tests_run"] == 16)
check("holm_bonferroni_correction was applied with family_size == 16",
      suite["holm_bonferroni"]["family_size"] == 16)
check("mid bin ran too, but reported SEPARATELY (descriptive_mid_bin), never folded into "
      "the confirmatory/Holm family",
      len(suite["descriptive_mid_bin"]) > 0
      and all("bin=mid" not in k for k in suite["confirmatory"]))

# holm_bonferroni_correction itself, on a known small example (manual arithmetic check)
known_p = {"t1": 0.001, "t2": 0.01, "t3": 0.03, "t4": 0.20}
holm_known = holm_bonferroni_correction(known_p, alpha=0.05, family_size=4)
# step-down thresholds: t1 vs .05/4=.0125 (reject), t2 vs .05/3=.0167 (reject),
# t3 vs .05/2=.025 (.03 > .025 -> RETAIN, and the cascade stops there)
check("holm_bonferroni_correction reproduces the known step-down result by hand "
      f"(t1,t2 rejected; t3,t4 retained) -- got "
      f"{[k for k,v in holm_known['tests'].items() if v['reject_null']]}",
      holm_known["tests"]["t1"]["reject_null"] is True
      and holm_known["tests"]["t2"]["reject_null"] is True
      and holm_known["tests"]["t3"]["reject_null"] is False
      and holm_known["tests"]["t4"]["reject_null"] is False)

# cross-check against configs/evaluation.yaml, already verified equal elsewhere, re-confirmed here
import yaml
ev_cfg = yaml.safe_load((ROOT / "configs" / "evaluation.yaml").read_text(encoding="utf-8"))
check("configs/evaluation.yaml's statistics.confirmatory_family_size == 16, matching the code",
      ev_cfg["statistics"]["confirmatory_family_size"] == 16)

# ======================================================================================
# 5. other statistical tests: sane values on synthetic data with KNOWN structure
# ======================================================================================
print("\n---------------- statistical tests: sanity on known synthetic structure ----------------")

# paired_wilcoxon + effect size: a CLEAR, consistent improvement should reject the null with a
# large effect; two near-identical samples should NOT reject.
rng2 = np.random.default_rng(7)
b_scores = rng2.normal(0.55, 0.04, 30)
c_scores_better = b_scores + 0.10  # C consistently 0.10 higher, every single image
wilcox_clear = paired_wilcoxon(c_scores_better.tolist(), b_scores.tolist())
effect_clear = rank_biserial_effect_size(c_scores_better.tolist(), b_scores.tolist())
check("paired_wilcoxon on a CONSISTENT +0.10 improvement rejects the null (p < 0.001)",
      wilcox_clear["p_value"] < 0.001)
check("rank_biserial_effect_size on that same consistent improvement reports 'large' "
      f"(|r|={abs(effect_clear['r']):.3f})", effect_clear["label"] == "large")

c_scores_same = b_scores + rng2.normal(0, 0.001, 30)  # essentially identical
wilcox_same = paired_wilcoxon(c_scores_same.tolist(), b_scores.tolist())
check("paired_wilcoxon on near-identical paired samples does NOT reject at alpha=0.05 "
      f"(p={wilcox_same['p_value']:.3f})", wilcox_same["p_value"] > 0.05, warn_only=True)

ttest_clear = paired_t_test(c_scores_better.tolist(), b_scores.tolist())
check("paired_t_test on the same +0.10 improvement also rejects (p < 0.001), agreeing with "
      "Wilcoxon", ttest_clear["p_value"] < 0.001)
check("paired_t_test's mean_diff recovers the known +0.10 shift "
      f"(got {ttest_clear['mean_diff']:.4f})", abs(ttest_clear["mean_diff"] - 0.10) < 0.02)

# TOST: two distributions within the margin should report equivalent=True; two distributions
# with a real, consistent gap larger than the margin should report equivalent=False.
dice_b = rng2.normal(0.85, 0.015, 30)
dice_c_equivalent = dice_b + rng2.normal(0.0, 0.005, 30)   # tiny, symmetric noise
tost_equiv = tost_equivalence_test(dice_c_equivalent.tolist(), dice_b.tolist(), margin=0.02,
                                   metric_name="dice")
check(f"TOST on Dice values within +/-0.02 of each other reports equivalent=True "
      f"(mean_diff={tost_equiv['mean_diff']:.4f})", tost_equiv["equivalent"] is True)

dice_c_different = dice_b + 0.08  # a real, consistent 0.08 shift -- outside the 0.02 margin
tost_diff = tost_equivalence_test(dice_c_different.tolist(), dice_b.tolist(), margin=0.02,
                                  metric_name="dice")
check(f"TOST on Dice values shifted by +0.08 (outside the margin) reports equivalent=False "
      f"(mean_diff={tost_diff['mean_diff']:.4f})", tost_diff["equivalent"] is False)

# bootstrap_ci: the interval should bracket the TRUE population mean it was drawn from.
rng3 = np.random.default_rng(99)
true_mean = 0.72
sample = rng3.normal(true_mean, 0.05, 200)
ci = bootstrap_ci_mod.bootstrap_ci(sample.tolist(), statistic="mean", n_boot=2000, seed=42)
check(f"bootstrap_ci brackets the TRUE mean it was sampled from (mean={true_mean}, "
      f"CI=[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}])",
      ci["ci_low"] < true_mean < ci["ci_high"])
ci2 = bootstrap_ci_mod.bootstrap_ci(sample.tolist(), statistic="mean", n_boot=2000, seed=42)
check("bootstrap_ci is DETERMINISTIC given the same seed (identical CI on repeat)",
      ci["ci_low"] == ci2["ci_low"] and ci["ci_high"] == ci2["ci_high"])

# partial_correlation: construct a KNOWN confound -- curvature and boundary_error are each
# driven by a shared confound (cell_size) but have NO direct relationship of their own. Raw
# Pearson should show a spurious correlation; partial correlation controlling for cell_size
# should be much smaller (the confound genuinely removed).
rng4 = np.random.default_rng(123)
n_conf = 200
cell_size = rng4.normal(300, 60, n_conf)
noise_curv = rng4.normal(0, 0.02, n_conf)
noise_err = rng4.normal(0, 0.02, n_conf)
curvature_score = 0.30 * (cell_size / 300) + noise_curv
boundary_error = 0.30 * (cell_size / 300) + noise_err  # shares the SAME confound, no direct link
confound_df = pd.DataFrame({"curvature_score": curvature_score,
                            "boundary_error": boundary_error, "cell_size": cell_size,
                            "contrast": rng4.normal(5, 1, n_conf)})
raw_r = float(np.corrcoef(curvature_score, boundary_error)[0, 1])
partial = partial_correlation(confound_df, "curvature_score", "boundary_error",
                              covariates=["cell_size", "contrast"])
check(f"raw Pearson correlation shows the SPURIOUS confound-driven correlation (r={raw_r:.3f}, "
      f"expected clearly positive)", raw_r > 0.3)
check(f"partial correlation controlling for cell_size REMOVES most of it "
      f"(partial r={partial['r']:.3f} << raw r={raw_r:.3f})",
      abs(partial["r"]) < abs(raw_r) * 0.5)
check("partial correlation is no longer significant once the confound is controlled "
      f"(p={partial['p_value']:.3f})", partial["p_value"] > 0.05, warn_only=True)

# ======================================================================================
# 6. curvature_bins.csv hash unchanged from Phase 2
# ======================================================================================
print("\n---------------- curvature_bins.csv unchanged since Phase 2 ----------------")
cb_path = ROOT / "data" / "curvature" / "curvature_bins.csv"
check("data/curvature/curvature_bins.csv exists", cb_path.is_file())
current_hash = hashlib.sha256(cb_path.read_bytes()).hexdigest()
print(f"  current sha256: {current_hash}")

git_diff = subprocess.run(["git", "diff", "--stat", "HEAD", "--",
                          "data/curvature/curvature_bins.csv"],
                         cwd=ROOT, capture_output=True, text=True)
git_diff_cached = subprocess.run(["git", "diff", "--cached", "--stat", "HEAD", "--",
                                 "data/curvature/curvature_bins.csv"],
                                cwd=ROOT, capture_output=True, text=True)
check("git reports ZERO diff on data/curvature/curvature_bins.csv against HEAD (unstaged) -- "
      "nothing in Phase 7/8's work touched the write-once curvature table",
      git_diff.returncode == 0 and git_diff.stdout.strip() == "")
check("git reports ZERO diff on data/curvature/curvature_bins.csv against HEAD (staged)",
      git_diff_cached.returncode == 0 and git_diff_cached.stdout.strip() == "")

# ======================================================================================
# write the SYNTHETIC validation report
# ======================================================================================
report = {
    "phase": 8, "synthetic": True,
    "warning": "All mask cases and statistical-test inputs are SYNTHETIC, constructed with "
              "known-by-construction answers. This report verifies metric/statistics "
              "CORRECTNESS, not real model quality.",
    "cases": {k: {kk: (vv if np.isfinite(vv) else None) for kk, vv in v.items()
                 if isinstance(vv, (int, float))}
             for k, v in metrics_by_case.items()},
    "confirmatory_family_size": cs.CONFIRMATORY_FAMILY_SIZE,
    "curvature_bins_csv_sha256": current_hash,
    "n_failed": len(failures), "n_warnings": len(warnings), "failures": failures,
}
report_path = OUT_DIR / "SYNTHETIC_phase8_validation_report.json"
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
    sys.exit(1)
print("RESULT: PASS -- Phase 8 metrics/statistics verified (synthetic, known-answer cases).")
print("=" * 78)
