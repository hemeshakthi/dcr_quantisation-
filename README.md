# dCR Compression Study

Research question (FROZEN): Can dCR loss recover the boundary geometry degradation introduced by
structured channel pruning in microscopy cell segmentation?

**Single source of truth:** [`PROJECT_MASTER.md`](PROJECT_MASTER.md). Read it in full before making any
change to this repository. Every design decision carries a source label
([PAPER]/[OFFICIAL]/[LITERATURE]/[ENGINEERING]/[ASSUMPTION]); every [ASSUMPTION] is logged in
[`docs/implementation_assumptions.md`](docs/implementation_assumptions.md).

## Status

Phase 0 (repository scaffolding + authoritative source review) complete. See
[`docs/source_review.md`](docs/source_review.md) for what was recoverable about the original dCR paper
from public sources, and [`docs/implementation_assumptions.md`](docs/implementation_assumptions.md) for
every value still awaiting professor/PDF confirmation before Phase 2 (training pipeline) finalizes.

## Repository structure

See PROJECT_MASTER.md §15. Locked design docs (do not modify after implementation begins):

- [`docs/theoretical_motivation.md`](docs/theoretical_motivation.md)
- [`docs/success_criteria.md`](docs/success_criteria.md)
- [`docs/related_work.md`](docs/related_work.md)
- [`docs/frozen_design.md`](docs/frozen_design.md)

## Setup

```
conda env create -f environment.yml
```

or

```
pip install -r requirements.txt
```

No functional code has been written yet (Phase 0 is scaffolding-only per master's phase discipline).
