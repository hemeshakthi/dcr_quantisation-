"""Verification for the pre-Phase-3 statistical scope decision.

Checks that:
  * the power analysis exists and runs at the ACTUAL per-bin test n
  * those n values are re-derived from data/curvature/curvature_bins.csv, not trusted
  * the power numbers quoted in docs/success_criteria.md match a fresh computation
  * the mid bin is declared descriptive-only and excluded from the Holm family
  * the corrected family size is arithmetically consistent (24 -> 16)
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs"))

failures = []


def check(label, cond):
    ok = bool(cond)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)
    return ok


print("=" * 78)
print("STATISTICAL DESIGN VERIFICATION (pre-Phase-3)")
print("=" * 78)

check("docs/power_analysis.py exists", (ROOT / "docs" / "power_analysis.py").is_file())

# ---- 1. re-derive the bin n from the real curvature table -------------------------
cb = pd.read_csv(ROOT / "data" / "curvature" / "curvature_bins.csv")
test_counts = cb[cb["split"] == "test"]["bin"].value_counts().to_dict()
expected = {"low": 50, "mid": 12, "high": 36}
check(f"test-bin occupancy re-derived from curvature_bins.csv == {expected} "
      f"(got {test_counts})", test_counts == expected)

# ---- 2. the power analysis runs for each bin --------------------------------------
for n in (50, 12, 36):
    r = subprocess.run([sys.executable, str(ROOT / "docs" / "power_analysis.py"),
                        "--n_test", str(n)],
                       capture_output=True, text=True, cwd=ROOT)
    check(f"power_analysis.py runs for n={n} (exit {r.returncode})", r.returncode == 0)
    check(f"  n={n} output reports a verdict", "VERDICT:" in r.stdout)

# ---- 3. recompute the power numbers and compare with the documented table ---------
from power_analysis import wilcoxon_power, mde, n_required  # noqa: E402

sc = (ROOT / "docs" / "success_criteria.md").read_text(encoding="utf-8")

recomputed = {n: {"d05": wilcoxon_power(0.5, n, 0.05),
                  "d08": wilcoxon_power(0.8, n, 0.05),
                  "mde80": mde(n, 0.05, 0.80, 3.0 / 3.141592653589793),
                  "holm_d05": wilcoxon_power(0.5, n, 0.05 / 16)}
              for n in (50, 12, 36)}
for n, vals in recomputed.items():
    print(f"       n={n:>2}: d=0.5 -> {vals['d05']:.3f}, d=0.8 -> {vals['d08']:.3f}, "
          f"MDE80 -> {vals['mde80']:.3f}, Holm d=0.5 -> {vals['holm_d05']:.3f}")

for n, key, fmt in [(50, "d05", "0.923"), (12, "d05", "0.337"), (36, "d05", "0.812"),
                    (50, "mde80", "0.414"), (12, "mde80", "0.914"), (36, "mde80", "0.492")]:
    check(f"documented value {fmt} for n={n} ({key}) matches a fresh computation "
          f"({recomputed[n][key]:.3f})",
          abs(recomputed[n][key] - float(fmt)) < 0.0015 and fmt in sc)

check("mid bin is genuinely underpowered for a medium effect (recomputed)",
      recomputed[12]["d05"] < 0.5 and n_required(0.5, 0.05, 0.80) > 12)
check("high and low bins reach >=0.80 power at d=0.5 (recomputed)",
      recomputed[36]["d05"] >= 0.80 and recomputed[50]["d05"] >= 0.80)

# ---- 4. family-size arithmetic ----------------------------------------------------
original = 2 * 3 * 2 * 2   # comparisons x bins x ratios x metrics
corrected = 2 * 2 * 2 * 2  # mid bin dropped
check(f"original family enumerates to 24, matching master §6's '~24' (got {original})",
      original == 24)
check(f"corrected family enumerates to 16 (got {corrected})", corrected == 16)
check("success_criteria.md documents the corrected family size 16",
      "= **16**" in sc and "24 → 16" in sc)
check("success_criteria.md shows the Holm thresholds for both family sizes",
      "0.002083" in sc and "0.003125" in sc)

# ---- 5. scope note present and correctly worded -----------------------------------
flat = " ".join(sc.split())
for phrase in [
    "low=50, mid=12, high=36",
    "underpowered by construction",
    "reported descriptively only",
    "never used to support or refute P1–P6",
    "high-curvature bin (n=36) vs low-curvature bin (n=50)",
    "disclosed limitation, not a retroactive weakening",
]:
    check(f"scope note contains: '{phrase[:52]}...'", phrase in flat)

check("success_criteria.md amendment is marked as authorised on a LOCKED doc",
      "AMENDMENT" in sc and "LOCKED" in sc)
check("original thresholds table left intact (0.04 / 10% / ±0.02 / 6-of-10)",
      "≥ 0.04" in sc and "10% relative reduction" in sc and "±0.02" in sc)
check("A29 logs the family-size reduction",
      "### A29" in (ROOT / "docs" / "implementation_assumptions.md").read_text(encoding="utf-8"))

print()
print("=" * 78)
if failures:
    print(f"RESULT: FAIL ({len(failures)} check(s) failed)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: PASS — statistical scope verified.")
print("=" * 78)
print("Confirmatory family = 16 (mid bin descriptive-only). P2/P4/P5 scoped to high (n=36)")
print("vs low (n=50). Mid bin (n=12) reported descriptively with power stated inline.")
