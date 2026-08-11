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

import hashlib
import json
from pathlib import Path
from typing import Any


def compute_mask_hash(mask: dict[str, list[int]]) -> str:
    """Stable SHA-256 over the mask contents.

    Serialises with sorted keys and canonical separators so the hash depends only on which
    channels are kept, not on dict ordering or JSON whitespace.
    """
    canonical = {k: list(v) for k, v in sorted(mask.items())}
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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
    p = Path(path)
    if p.exists():
        raise FileExistsError(
            f"channel mask {p} already exists (write-once, master §9). Delete it explicitly "
            f"if you intend to invalidate every model pruned from it.")
    mask_hash = compute_mask_hash(mask)
    payload = {"hash": mask_hash, "mask": {k: list(v) for k, v in sorted(mask.items())},
              "metadata": metadata or {}}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return mask_hash


def load_channel_mask(path: str | Path, expected_hash: str | None = None) -> dict[str, Any]:
    """Load a mask and verify its integrity.

    Args:
        expected_hash: if given, the recomputed hash must match.

    Raises:
        ValueError: on hash mismatch -- the mask has been edited or regenerated, so anything
            trained against the old one is no longer comparable.
    """
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    mask = {k: list(v) for k, v in payload["mask"].items()}
    recomputed = compute_mask_hash(mask)
    if recomputed != payload["hash"]:
        raise ValueError(
            f"channel mask {p} is corrupted: stored hash {payload['hash']} does not match "
            f"its own contents ({recomputed})")
    if expected_hash is not None and recomputed != expected_hash:
        raise ValueError(
            f"channel mask {p} hash {recomputed} does not match expected {expected_hash} -- "
            f"this is a DIFFERENT mask than the one the caller was built against")
    return {"mask": mask, "hash": recomputed, "metadata": payload.get("metadata", {})}


def assert_same_mask(run_records: list[dict[str, Any]]) -> None:
    """Assert every run in the list pruned from the identical mask.

    Called by the audit over B, B' and C at a given ratio. Master §9 requires them to have
    identical params and FLOPs; identical mask hash is the stronger, upstream guarantee.

    Raises:
        AssertionError: listing the differing runs and their hashes.
    """
    hashes = {r.get("run_id", i): r.get("mask_hash") for i, r in enumerate(run_records)}
    distinct = set(hashes.values())
    if len(distinct) > 1:
        raise AssertionError(
            f"channel mask hashes differ across runs that must share one architecture "
            f"(master §9): {hashes}")


def apply_mask_to_checkpoint(mask_path: str | Path,
                             checkpoint_path: str | Path,
                             model_cfg: dict[str, Any],
                             expected_hash: str | None = None,
                             map_location: str = "cpu") -> tuple[Any, str]:
    """Load a saved mask, apply it to a FRESH checkpoint's weights, return (pruned_model, hash).

    This is the load-time counterpart B/B'/C actually use (master §9: "load the SAME pruned
    base"): `checkpoint_path` is Model A's trained checkpoint (a FRESH, pre-pruning-shape
    network state), `mask_path` is the ONE mask A was pruned to at this ratio. Verifying
    `expected_hash` here -- not just trusting the file on disk -- is what turns "B/B'/C should
    share a mask" into a runtime-enforced guarantee rather than an assumption about which file
    happens to be present.

    Returns:
        (pruned_model, mask_hash) -- a physically narrowed `UNet` with A's weights transplanted
        wherever a clean slice is well-defined (see `structured_pruning.apply_channel_mask`),
        and the verified hash of the mask that produced it.
    """
    from src.compression.structured_pruning import apply_channel_mask
    from src.models.unet import UNet
    from src.training.checkpointing import load_checkpoint

    loaded = load_channel_mask(mask_path, expected_hash=expected_hash)
    fresh = UNet.from_config(model_cfg)
    load_checkpoint(checkpoint_path, model=fresh, map_location=map_location)
    pruned = apply_channel_mask(fresh, loaded["mask"])
    return pruned, loaded["hash"]
