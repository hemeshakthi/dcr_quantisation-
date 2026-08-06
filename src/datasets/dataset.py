"""Torch Dataset / DataLoader over the Phase 2 preprocessed arrays.

Reads the static split CSVs (data/splits/{train,test}.csv) and the preprocess manifest, and
serves (image, binary_mask, instance_mask, metadata) tuples.

Instance masks are served ALONGSIDE the binary target, never instead of it: the binary mask
is the training target (master §7 emits one logit channel) while instance IDs are needed by
the curvature stratification (§11) and the cell_merging / cell_splitting detectors (§12).

Augmentation is applied through `src.datasets.augmentations.Augmenter`, which is seeded by
(seed, epoch, index). Master §10 requires the augmentation seed to be SHARED across B/B'/C
with only `model_init_seed` varying, so the augmentation seed is a dataset-level argument
deliberately separate from the model seed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.augmentations import Augmenter


class CellSegmentationDataset(Dataset):
    """Preprocessed microscopy pairs for one split.

    Args:
        split: "train" or "test".
        augmenter: seeded augmenter, or None to serve un-augmented data (evaluation).
        datasets: optional dataset-key filter (e.g. only stardist_tcell).
        curvature_bins: if given, attach each image's curvature bin to the returned metadata
            so evaluation can stratify without a second join.
        return_instances: serve the instance-ID mask as well as the binary mask.
    """

    def __init__(self,
                 split: str,
                 augmenter: Augmenter | None = None,
                 datasets: list[str] | None = None,
                 curvature_bins: str | Path | None = None,
                 return_instances: bool = True) -> None:
        self.split = split
        self.augmenter = augmenter
        self.datasets = datasets
        self.curvature_bins = curvature_bins
        self.return_instances = return_instances
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to derive augmentation randomness.

        Must be called every epoch. The augmenter keys off (seed, epoch, index), so without
        this the same augmentation would repeat each epoch.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Returns a dict with:
            image      (1, H, W) float32 tensor
            mask       (1, H, W) float32 tensor in {0., 1.}
            instances  (H, W) int32 tensor of instance IDs (if return_instances)
            image_id   str, the stable identifier used by every downstream table
            meta       dict with dataset, split, curvature bin, cell_size, contrast
        """
        raise NotImplementedError


def build_dataloader(split: str,
                     batch_size: int,
                     shuffle: bool,
                     num_workers: int = 0,
                     augmenter: Augmenter | None = None,
                     seed: int | None = None,
                     **dataset_kwargs: Any) -> DataLoader:
    """Construct a DataLoader with deterministic ordering.

    Uses a seeded `torch.Generator` and a worker_init_fn so shuffling and worker RNG are
    reproducible, as master §10's determinism block requires. `num_workers=0` is the safe
    default here because the host's commit limit equals physical RAM (A18); Kaggle can raise
    it.
    """
    raise NotImplementedError


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate that keeps per-image metadata as plain lists.

    The default collate would try to tensorise the metadata dicts and the string image_ids;
    those must survive intact so every metric row can be traced back to its image.
    """
    raise NotImplementedError
