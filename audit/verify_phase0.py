"""Phase 0 verification script. Checks scaffolding only, no pipeline logic. Re-run at any time."""
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
failures = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        failures.append(label)


# 1. Repo tree per master §15
expected_dirs = [
    "configs",
    "data/raw", "data/processed", "data/splits", "data/curvature",
    "src/datasets", "src/models", "src/losses", "src/compression", "src/metrics",
    "src/failure", "src/training", "src/evaluation", "src/visualization",
    "experiments", "backend", "frontend", "docs", "tests", "audit",
    "checkpoints",
    "results/per_image_metrics", "results/tables", "results/figures", "results/failure_counts",
    "logs",
]
for d in expected_dirs:
    check(f"dir exists: {d}", (ROOT / d).is_dir())

expected_init = [
    "src/__init__.py", "src/datasets/__init__.py", "src/models/__init__.py",
    "src/losses/__init__.py", "src/compression/__init__.py", "src/metrics/__init__.py",
    "src/failure/__init__.py", "src/training/__init__.py", "src/evaluation/__init__.py",
    "src/visualization/__init__.py",
]
for f in expected_init:
    check(f"__init__.py exists: {f}", (ROOT / f).is_file())

# 2. requirements.txt parses (pinned == format)
req_path = ROOT / "requirements.txt"
req_ok = False
if req_path.is_file():
    lines = [l.strip() for l in req_path.read_text().splitlines() if l.strip()]
    req_ok = len(lines) > 0 and all("==" in l for l in lines)
check("requirements.txt present and every line pinned with ==", req_ok)

# 3. environment.yml present and mentions python=3.10
env_path = ROOT / "environment.yml"
env_ok = env_path.is_file() and "python=3.10" in env_path.read_text()
check("environment.yml present with python=3.10", env_ok)

# 4. Docs present and non-empty
expected_docs = [
    "theoretical_motivation.md", "success_criteria.md", "related_work.md",
    "frozen_design.md", "source_review.md", "implementation_assumptions.md",
]
for doc in expected_docs:
    p = ROOT / "docs" / doc
    check(f"docs/{doc} present and non-empty", p.is_file() and p.stat().st_size > 0)

# 4b. Locked design docs carry the LOCKED header
locked_docs = ["theoretical_motivation.md", "success_criteria.md", "related_work.md", "frozen_design.md"]
for doc in locked_docs:
    p = ROOT / "docs" / doc
    ok = p.is_file() and "LOCKED — do not modify after implementation begins" in p.read_text(encoding="utf-8")
    check(f"docs/{doc} carries LOCKED header", ok)

# 5. experiment_log.csv header correct (non-empty header, no data rows yet)
log_path = ROOT / "results" / "tables" / "experiment_log.csv"
log_ok = False
if log_path.is_file():
    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    log_ok = len(rows) == 1 and len(rows[0]) > 1 and "run_id" in rows[0] and "model" in rows[0]
check("experiment_log.csv has header row only, includes run_id/model", log_ok)

# 6. .gitignore covers required entries
gi_path = ROOT / ".gitignore"
gi_required = ["checkpoints/", "data/raw/", "data/processed/", "logs/", "__pycache__", "*.pth"]
gi_text = gi_path.read_text() if gi_path.is_file() else ""
gi_ok = all(entry in gi_text for entry in gi_required)
check(".gitignore covers all required entries (master §15)", gi_ok)

# 7. git log shows first commit
git_ok = False
try:
    out = subprocess.run(["git", "log", "--oneline"], cwd=ROOT, capture_output=True, text=True, check=True)
    git_ok = len(out.stdout.strip().splitlines()) >= 1
except Exception:
    git_ok = False
check("git log shows at least one commit", git_ok)

# 8. implementation_assumptions.md logs every currently-unconfirmed value from master §0a
assumptions_path = ROOT / "docs" / "implementation_assumptions.md"
assumptions_text = assumptions_path.read_text(encoding="utf-8") if assumptions_path.is_file() else ""
required_topics = [
    "dataset", "split", "resolution", "normalization", "augmentation",
    "optimizer", "learning rate", "scheduler", "epoch", "batch", "initial",
    "stopping", "dCR loss equation", "post-processing", "channel widths",
]
missing_topics = [t for t in required_topics if t.lower() not in assumptions_text.lower()]
check(f"implementation_assumptions.md covers all master §0a missing items (missing: {missing_topics or 'none'})",
      not missing_topics)

# 9. No functional pipeline code present yet (Phase 0 discipline)
src_py_files = [p for p in (ROOT / "src").rglob("*.py") if p.name != "__init__.py"]
check("no functional code under src/ yet (only __init__.py)", len(src_py_files) == 0)

print()
if failures:
    print(f"RESULT: FAIL ({len(failures)} check(s) failed)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("RESULT: PASS — Phase 0 verification complete.")
