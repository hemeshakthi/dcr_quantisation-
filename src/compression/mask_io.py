"""Kept-channel index masks: save, load, hash, verify (master §9).

The frozen protocol is: train A -> prune ONCE per ratio -> save the per-layer kept-channel
index mask as JSON with a hash -> B, B' and C all LOAD THE SAME MASK and never re-prune.

Why the hash matters. If B, B' and C were each allowed to prune independently, they would
select different channels (L1 ties break differently, and any RNG or ordering difference
shifts the choice). They would then have different architectures, and every downstream
comparison -- the entire study -- would confound "different loss" with "different network".
The mask file plus its hash is what turns "they should be the same" into "they are provably
the same": each run records the hash it loaded, and the audit fails if they disagree.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def compute_mask_hash(mask: dict[str, list[int]]) -> str:
    """Stable SHA-256 over the mask contents.

    Serialises with sorted keys and canonical separators so the hash depends only on which
    channels are kept, not on dict ordering or JSON whitespace.
    """
    raise NotImplementedError


def save_channel_mask(mask: dict[str, list[int]],
                      path: str | Path,
                      metadata: dict[str, Any] | None = None) -> str:
    """Write the mask JSON and return its hash.

    The file records the kept indices per layer, the hash, the pruning ratio, the source
    checkpoint of Model A, and the measured params/FLOPs -- everything needed to reconstruct
    the identical narrowed network later.

    Refuses to overwrite an existing mask (write-once, like the curvature bins): a silently
    regenerated mask after some runs had already used the old one would split B/B'/C across
    two different architectures.
    """
    raise NotImplementedError


def load_channel_mask(path: str | Path, expected_hash: str | None = None) -> dict[str, Any]:
    """Load a mask and verify its integrity.

    Args:
        expected_hash: if given, the recomputed hash must match.

    Raises:
        ValueError: on hash mismatch -- the mask has been edited or regenerated, so anything
            trained against the old one is no longer comparable.
    """
    raise NotImplementedError


def assert_same_mask(run_records: list[dict[str, Any]]) -> None:
    """Assert every run in the list pruned from the identical mask.

    Called by the audit over B, B' and C at a given ratio. Master §9 requires them to have
    identical params and FLOPs; identical mask hash is the stronger, upstream guarantee.

    Raises:
        AssertionError: listing the differing runs and their hashes.
    """
    raise NotImplementedError
