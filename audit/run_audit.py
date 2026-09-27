"""audit/run_audit.py -- Unified audit harness (master §14).

Orchestrates every individual verify_phaseX.py script in this directory, producing
a single PASS / FAIL / PENDING TRAINING summary table.

PENDING TRAINING means: the check cannot be completed without real trained checkpoints
(Vanilla, A, B, B', C) which do not yet exist. This is NOT a failure -- it is a
deferred verification that will be re-run after Phase 6-RUN / Phase 7-RUN complete.

Exit codes:
  0  all checks are PASS or PENDING TRAINING
  1  at least one check is a genuine FAIL

Usage:
  python audit/run_audit.py
  python audit/run_audit.py --verbose   (show each script's stdout on failure)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import sys
from pathlib import Path

# Force UTF-8 output on Windows (default cp1252 cannot encode box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif sys.stdout.encoding not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent  # repo root

# ──────────────────────────────────────────────────────────────────────────────
# Registry: every individual verify script, in phase order.
# "pending_if_no_checkpoints": when True, a non-zero exit code is treated as
#   PENDING TRAINING (not FAIL) if checkpoints/ contains no real .pt files.
# ──────────────────────────────────────────────────────────────────────────────
SCRIPTS: list[dict] = [
    {"label": "Phase 0  — Foundation",          "script": "verify_phase0.py",              "pending_if_no_checkpoints": False},
    {"label": "Phase 1  — Data acquisition",    "script": "verify_phase1.py",              "pending_if_no_checkpoints": False},
    {"label": "Phase 2  — Preprocessing",       "script": "verify_phase2.py",              "pending_if_no_checkpoints": False},
    {"label": "Phase 3  — Imports/scope-grep",  "script": "verify_phase3.py",              "pending_if_no_checkpoints": False},
    {"label": "Phase 4  — Model architecture",  "script": "verify_phase4.py",              "pending_if_no_checkpoints": False},
    {"label": "Phase 5  — Loss functions",      "script": "verify_phase5.py",              "pending_if_no_checkpoints": False},
    {"label": "Phase 6  — Training infra",      "script": "verify_phase6.py",              "pending_if_no_checkpoints": False},
    {"label": "Phase 6-RUN — Kaggle notebook A","script": "verify_phase6_run.py",          "pending_if_no_checkpoints": True},
    {"label": "Phase 7  — Structured pruning",  "script": "verify_phase7.py",              "pending_if_no_checkpoints": False},
    {"label": "Phase 8  — Eval/statistics",     "script": "verify_phase8.py",              "pending_if_no_checkpoints": False},
    {"label": "Phase 9  — Failure taxonomy",    "script": "verify_phase9.py",              "pending_if_no_checkpoints": False},
    {"label": "Phase 12 — Figures",             "script": "verify_phase12.py",             "pending_if_no_checkpoints": False},
    {"label": "Stats design",                   "script": "verify_stats_design.py",        "pending_if_no_checkpoints": False},
    {"label": "Backend integration",            "script": "verify_backend_integration.py", "pending_if_no_checkpoints": True},
    {"label": "Frontend",                       "script": "verify_frontend.py",            "pending_if_no_checkpoints": False},
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
CURVATURE_CSV = ROOT / "data" / "curvature" / "curvature_bins.csv"
EXPECTED_HASH  = "e7c09383fe7b7c4aad05eb4dbcf420ee87743970f6463cade3d8ca797242a242"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest().lower()


def _real_checkpoints_exist() -> bool:
    """Return True if checkpoints/ holds at least one real .pt file.

    Synthetic stand-ins in results/synthetic_validation/ are excluded.
    The only files that count are those directly under checkpoints/.
    """
    ckpt_dir = ROOT / "checkpoints"
    if not ckpt_dir.exists():
        return False
    return any(ckpt_dir.rglob("*.pt"))


def _run_script(script_path: Path, verbose: bool) -> tuple[int, str]:
    """Run one verify script, return (returncode, combined_stdout_stderr)."""
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.returncode, result.stdout


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="print each script's output on FAIL")
    args = ap.parse_args()

    audit_dir = Path(__file__).resolve().parent
    has_real_checkpoints = _real_checkpoints_exist()

    print("=" * 78)
    print("dCR COMPRESSION STUDY — UNIFIED AUDIT HARNESS  (audit/run_audit.py)")
    print("=" * 78)
    print(f"repo root  : {ROOT}")
    print(f"audit dir  : {audit_dir}")
    print(f"real checkpoints exist: {has_real_checkpoints}")
    print()

    # ---- curvature_bins.csv integrity (the single most important file check) ----
    print("-" * 78)
    print("FILE INTEGRITY -- data/curvature/curvature_bins.csv")
    print("-" * 78)

    if CURVATURE_CSV.exists():
        actual_hash = _sha256(CURVATURE_CSV)
        hash_ok = actual_hash == EXPECTED_HASH
        status_str = "PASS" if hash_ok else "FAIL — HASH MISMATCH"
        print(f"  SHA-256 (actual)   : {actual_hash}")
        print(f"  SHA-256 (expected) : {EXPECTED_HASH}")
        print(f"  Status             : {status_str}")
        if not hash_ok:
            print("  *** CRITICAL: curvature_bins.csv has been modified after locking. ***")
            print("  *** All downstream analysis is invalidated. Re-run Phase 2 ONLY     ***")
            print("  *** after resolving this with the project PI. Do NOT continue.       ***")
    else:
        actual_hash = None
        print("  FAIL — curvature_bins.csv does not exist. Run Phase 2 first.")
    print()

    # ── experiment_log.csv row count ──
    exp_log = ROOT / "results" / "tables" / "experiment_log.csv"
    if exp_log.exists():
        lines = exp_log.read_text(encoding="utf-8").strip().splitlines()
        data_rows = len(lines) - 1  # subtract header
        print(f"experiment_log.csv   : {data_rows} data row(s) "
              f"({'header only — no training yet' if data_rows == 0 else 'HAS REAL TRAINING ROWS'})")
    else:
        print("experiment_log.csv   : does not exist")
    print()

    # ---- run each verify script ----
    print("-" * 78)
    print("PHASE-BY-PHASE CHECKS")
    print("-" * 78)


    results: list[dict] = []
    for entry in SCRIPTS:
        label  = entry["label"]
        script = audit_dir / entry["script"]
        pend   = entry["pending_if_no_checkpoints"]

        if not script.exists():
            status = "MISSING"
            reason = f"{entry['script']} not found in audit/"
            output = ""
        else:
            rc, output = _run_script(script, args.verbose)
            if rc == 0:
                status = "PASS"
                reason = ""
            else:
                if pend and not has_real_checkpoints:
                    status = "PENDING TRAINING"
                    reason = "requires real trained checkpoints (not yet run)"
                else:
                    status = "FAIL"
                    reason = f"exit code {rc}"

        pad = " " * max(0, 35 - len(label))
        print(f"  {label}{pad}  {status:<18}  {reason}")

        if args.verbose and status == "FAIL" and output:
            print()
            for line in output.strip().splitlines()[-20:]:
                print(f"    {line}")
            print()

        results.append({"label": label, "script": entry["script"],
                         "status": status, "reason": reason, "output": output})

    # ── summary ──
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    n_pass    = sum(1 for r in results if r["status"] == "PASS")
    n_fail    = sum(1 for r in results if r["status"] == "FAIL")
    n_pending = sum(1 for r in results if r["status"] == "PENDING TRAINING")
    n_missing = sum(1 for r in results if r["status"] == "MISSING")
    total     = len(results)

    print(f"  PASS            : {n_pass}/{total}")
    print(f"  FAIL            : {n_fail}/{total}")
    print(f"  PENDING TRAINING: {n_pending}/{total}")
    print(f"  MISSING script  : {n_missing}/{total}")
    print()

    if CURVATURE_CSV.exists():
        print(f"  curvature_bins.csv SHA-256: {actual_hash}")
        print(f"  hash {'MATCHES' if (actual_hash == EXPECTED_HASH) else 'DOES NOT MATCH'} "
              f"expected ({EXPECTED_HASH})")
    print()

    if n_fail == 0 and n_missing == 0:
        print("  RESULT: READY FOR KAGGLE TRAINING  "
              "(all checks PASS or PENDING TRAINING pending real checkpoints)")
        return 0
    else:
        print(f"  RESULT: {n_fail} genuine FAIL(s) and {n_missing} missing script(s).")
        print("  Resolve all FAILs before proceeding to Kaggle training.")
        for r in results:
            if r["status"] in ("FAIL", "MISSING"):
                print(f"    [{r['status']}] {r['label']}: {r['reason']}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
