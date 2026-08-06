"""One row per run in results/tables/experiment_log.csv (master §10).

The header was fixed in Phase 0. Master §10 requires the three seeds to be logged SEPARATELY,
so this is strictly one row per (model, ratio, seed) -- 24 rows for the full study
(Vanilla 3 + A 3 + [B + B' + C] x 2 ratios x 3 seeds = 24). Averaging seeds into one row would
discard the between-seed variance that master §4's "mean over 3 seeds" criteria depend on.

Every logged hyperparameter carries the source label it had in its YAML, so the provenance
chain from PROJECT_MASTER.md through the config to the result is unbroken (Golden Rule 7).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def append_run(record: dict[str, Any], path: str | Path | None = None) -> None:
    """Append one completed run.

    Validates the record against the existing CSV header and refuses to write on a mismatch
    rather than silently producing a ragged table.
    """
    raise NotImplementedError


def build_run_record(run_id: str, config: dict[str, Any], summary: dict[str, Any],
                     source_labels: dict[str, str] | None = None) -> dict[str, Any]:
    """Assemble the row from a run's config and its Trainer summary.

    Captures the git commit hash so results can be tied to the exact code that produced them.
    """
    raise NotImplementedError


def expected_run_count(config: dict[str, Any]) -> int:
    """Total runs implied by the design: Vanilla(3) + A(3) + [B+B'+C] x 2 ratios x 3 seeds = 24."""
    raise NotImplementedError


def audit_completeness(path: str | Path, expected: int = 24) -> dict[str, Any]:
    """Check the log for missing or duplicated (model, ratio, seed) combinations.

    Returns which combinations are absent and which appear more than once -- a duplicate
    usually means a run was repeated after a change, and the stale row must not silently enter
    the analysis.
    """
    raise NotImplementedError
