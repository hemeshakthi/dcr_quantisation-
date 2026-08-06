"""Full determinism (master §10).

Master §10's determinism block: torch/cuda/numpy/random seeds; cudnn.deterministic=True;
cudnn.benchmark=False; torch.use_deterministic_algorithms(True).

Determinism is load-bearing here, not hygiene. The study's claim rests on B, B' and C
differing ONLY in their loss. They share the pruned base, the augmentation seed and the data
order; if any nondeterminism leaked in, an observed C-over-B difference could be run-to-run
variance. Three seeds (master §10) bound that variance, but only if each seed is itself
reproducible.

Note the deliberate split: `model_init_seed` varies across the 3 seeds, while the augmentation
seed is SHARED across B/B'/C. Conflating them into one global seed would make the three models
see different augmented data and destroy the comparison.
"""
from __future__ import annotations


def set_determinism(seed: int, cudnn_deterministic: bool = True,
                    warn_only: bool = False) -> dict:
    """Seed every RNG and enable deterministic kernels.

    Args:
        seed: seeds python's `random`, numpy, torch CPU and all CUDA devices.
        warn_only: pass through to torch.use_deterministic_algorithms; True downgrades a
            missing deterministic kernel from an error to a warning. Should stay False for
            final runs so a nondeterministic op fails loudly instead of quietly.

    Returns:
        A record of what was set, for the experiment log.

    Also sets CUBLAS_WORKSPACE_CONFIG, which CUDA >= 10.2 requires before deterministic
    cuBLAS matmuls will work -- without it torch raises at the first matmul.
    """
    raise NotImplementedError


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn giving each worker a distinct, reproducible seed."""
    raise NotImplementedError


def determinism_report() -> dict:
    """Snapshot the current determinism-relevant state.

    Written into every run's experiment log so a run that was NOT deterministic is visible
    afterwards rather than assumed away.
    """
    raise NotImplementedError


def verify_reproducible(fn, *args, n_repeats: int = 2, **kwargs) -> bool:
    """Run `fn` repeatedly under the same seed and check the outputs are byte-identical.

    Used in tests to prove the pipeline is actually deterministic rather than merely
    configured to be.
    """
    raise NotImplementedError
