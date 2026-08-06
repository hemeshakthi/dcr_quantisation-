"""Seed management and full determinism (master §10).

Master §10's determinism block: torch/cuda/numpy/random seeds; cudnn.deterministic=True;
cudnn.benchmark=False; torch.use_deterministic_algorithms(True).

Determinism is load-bearing here, not hygiene. The study's claim is that B, B' and C differ
ONLY in their loss. They share the pruned base, the augmentation seed and the data order; if
nondeterminism leaked in, an observed C-over-B difference could be run-to-run variance. Three
seeds bound that variance, but only if each seed is itself exactly reproducible.

THE TWO SEEDS ARE DELIBERATELY SEPARATE
---------------------------------------
  * `model_init_seed`   VARIES across the 3 runs -- it is the only intended source of
                        between-seed variation.
  * `augmentation_seed` is SHARED across B/B'/C and across all 3 seeds -- so the three models
                        see byte-identical augmented data.
Collapsing these into one global seed would make B, B' and C train on different data, and the
whole comparison would silently become "different loss AND different data".

Works identically under requirements-local.txt and requirements-kaggle.txt: no torch API used
here changed between 2.1 and 2.3.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class SeedSet:
    """The seeds for one run. `model_init_seed` is the only one that varies across runs."""
    model_init_seed: int
    augmentation_seed: int
    data_order_seed: int

    def as_dict(self) -> dict[str, int]:
        return {"model_init_seed": self.model_init_seed,
                "augmentation_seed": self.augmentation_seed,
                "data_order_seed": self.data_order_seed}


def seeds_for_run(cfg: dict[str, Any], seed_index: int) -> SeedSet:
    """Build the SeedSet for run `seed_index` (0, 1, 2) from a composed config.

    Only `model_init_seed` is indexed. The augmentation and data-order seeds are taken from the
    SHARED training config and are identical for every run, which is what master §10 requires.
    """
    init_seeds = cfg.get("seeds", {}).get("model_init_seeds", [1, 2, 3])
    if not 0 <= seed_index < len(init_seeds):
        raise IndexError(f"seed_index {seed_index} out of range for {init_seeds}")
    aug = int(cfg.get("augmentation", {}).get("seed", 1234))
    return SeedSet(model_init_seed=int(init_seeds[seed_index]),
                   augmentation_seed=aug,
                   data_order_seed=aug)


def set_determinism(seed: int, cudnn_deterministic: bool = True,
                    cudnn_benchmark: bool = False,
                    use_deterministic_algorithms: bool = True,
                    warn_only: bool = False) -> dict[str, Any]:
    """Seed every RNG and enable deterministic kernels. Returns a record for the log.

    CUBLAS_WORKSPACE_CONFIG must be set BEFORE the first cuBLAS call or CUDA >= 10.2 raises as
    soon as a deterministic matmul runs; setting it here rather than relying on the environment
    keeps a Kaggle notebook from having to remember it.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    deterministic_algorithms_set = False
    try:
        torch.use_deterministic_algorithms(bool(use_deterministic_algorithms),
                                           warn_only=bool(warn_only))
        deterministic_algorithms_set = True
    except Exception as exc:  # pragma: no cover - depends on build
        # Recorded rather than swallowed: a run that could NOT enable deterministic algorithms
        # must be visible in its log, not silently treated as reproducible.
        deterministic_algorithms_error = str(exc)
    else:
        deterministic_algorithms_error = None

    return {
        "seed": seed,
        "cudnn_deterministic": bool(cudnn_deterministic),
        "cudnn_benchmark": bool(cudnn_benchmark),
        "use_deterministic_algorithms": deterministic_algorithms_set,
        "deterministic_algorithms_error": deterministic_algorithms_error,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def apply_run_determinism(cfg: dict[str, Any], seeds: SeedSet) -> dict[str, Any]:
    """Apply the config's determinism block using this run's model_init_seed."""
    det = cfg.get("determinism", {})
    record = set_determinism(
        seed=seeds.model_init_seed,
        cudnn_deterministic=det.get("cudnn_deterministic", True),
        cudnn_benchmark=det.get("cudnn_benchmark", False),
        use_deterministic_algorithms=det.get("use_deterministic_algorithms", True),
        warn_only=det.get("warn_only", False),
    )
    record.update(seeds.as_dict())
    return record


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn giving each worker a distinct, reproducible seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Seeded torch.Generator for DataLoader shuffling, so data ORDER is reproducible."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def rng_state_snapshot() -> dict[str, Any]:
    """Capture every RNG state, for checkpoint/resume.

    Without this a resumed run would continue with a fresh RNG and therefore a different data
    order and different augmentations -- so "resume" would not reproduce "run straight through",
    and the resume path could not be verified.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def rng_state_restore(state: dict[str, Any]) -> None:
    """Restore states captured by `rng_state_snapshot`."""
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu")
                            else state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def determinism_report() -> dict[str, Any]:
    """Snapshot the determinism-relevant state, written into every experiment-log row."""
    return {
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_available": torch.cuda.is_available(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
