"""Torch Dataset / DataLoader over the Phase 2 preprocessed arrays.

Reads the preprocess manifest and serves (image, binary mask, instance mask, metadata).

Instance masks are served ALONGSIDE the binary target, never instead of it: the binary mask is
the training target (master §7 emits one logit channel) while instance IDs are needed by the
curvature stratification (§11) and the cell_merging / cell_splitting detectors (§12).

Augmentation is applied through `src.datasets.augmentations.Augmenter`, seeded by
(seed, epoch, index). Master §10 requires the augmentation seed to be SHARED across B/B'/C with
only `model_init_seed` varying, so the augmentation seed is a dataset-level argument
deliberately separate from the model seed.

THE VALIDATION SPLIT IS CARVED OUT OF TRAIN ONLY. The test split is touched exactly once, at
final evaluation; using it for checkpoint selection would leak it into the reported result. The
carve-out is GROUP-AWARE, reusing the same atomic groups as Phase 1 (A11) -- otherwise patches
of one source frame would straddle train and val and the early-stopping signal would be
optimistic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.augmentations import Augmenter
from src.datasets.dataset_registry import DATA_PROCESSED, ROOT
from src.training.seed_manager import make_generator, seed_worker

MANIFEST = DATA_PROCESSED / "preprocess_manifest.csv"
CURVATURE = ROOT / "data" / "curvature" / "curvature_bins.csv"


def group_aware_val_split(df: pd.DataFrame, val_fraction: float,
                          seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a TRAIN frame into (train, val) keeping atomic groups intact (A11)."""
    if "group_id" not in df.columns:
        groups = df["image_id"]
    else:
        groups = df["group_id"]
    uniq = sorted(groups.dropna().unique().tolist())
    rng = np.random.default_rng(seed)
    order = list(uniq)
    rng.shuffle(order)
    n_val_target = val_fraction * len(df)
    val_groups, n = set(), 0
    for g in order:
        if n >= n_val_target:
            break
        val_groups.add(g)
        n += int((groups == g).sum())
    is_val = groups.isin(val_groups)
    return df[~is_val].copy(), df[is_val].copy()


class CellSegmentationDataset(Dataset):
    """Preprocessed microscopy pairs for one split."""

    def __init__(self,
                 frame: pd.DataFrame,
                 augmenter: Augmenter | None = None,
                 return_instances: bool = True,
                 curvature_csv: str | Path | None = CURVATURE) -> None:
        self.frame = frame.reset_index(drop=True)
        self.augmenter = augmenter
        self.return_instances = return_instances
        self.epoch = 0
        self._curv: dict[str, dict[str, Any]] = {}
        if curvature_csv and Path(curvature_csv).exists():
            cb = pd.read_csv(curvature_csv)
            self._curv = {r["image_id"]: {"curvature_bin": r.get("bin"),
                                          "curvature_score": r.get("curvature_score"),
                                          "cell_size": r.get("cell_size"),
                                          "contrast": r.get("contrast")}
                          for _, r in cb.iterrows()}

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to derive augmentation randomness.

        Must be called every epoch: the augmenter keys off (seed, epoch, index), so without
        this the identical augmentation would repeat each epoch.
        """
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        r = self.frame.iloc[index]
        # mmap: never materialise the full array twice (A18 -- this host's commit limit equals
        # physical RAM; harmless and slightly faster on Kaggle too)
        img = np.array(np.load(ROOT / r["proc_image"], mmap_mode="r"), dtype=np.float32)
        inst = np.array(np.load(ROOT / r["proc_instances"], mmap_mode="r"), dtype=np.uint16)
        binary = (inst > 0).astype(np.uint8)

        if self.augmenter is not None:
            img, binary, inst = self.augmenter(img, binary, inst,
                                               epoch=self.epoch, index=index)

        meta = {"dataset": r.get("dataset"), "split": r.get("split")}
        meta.update(self._curv.get(r["image_id"], {}))

        out = {
            "image": torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(binary)).unsqueeze(0).float(),
            "image_id": r["image_id"],
            "meta": meta,
        }
        if self.return_instances:
            out["instances"] = torch.from_numpy(
                np.ascontiguousarray(inst.astype(np.int32)))
        return out


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate that keeps per-image metadata as plain lists.

    The default collate would try to tensorise the metadata dicts and the string image_ids;
    those must survive intact so every metric row can be traced back to its image.
    """
    out: dict[str, Any] = {
        "image": torch.stack([i["image"] for i in items]),
        "mask": torch.stack([i["mask"] for i in items]),
        "image_id": [i["image_id"] for i in items],
        "meta": [i["meta"] for i in items],
    }
    if "instances" in items[0]:
        out["instances"] = torch.stack([i["instances"] for i in items])
    return out


def build_dataloaders(cfg: dict[str, Any], augmentation_seed: int,
                      data_order_seed: int
                      ) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Build the (train, val, train_eval) loaders for a run from a composed config.

    `train_eval` is a FIXED, unaugmented, deterministic subsample of the training set, used
    only to monitor the train/val Boundary F1 gap each epoch (Phase 6-RUN; the training corpus
    here is small, ~170 images, so overfitting risk is a real concern to watch, not a
    formality). It is None when `training.track_train_val_gap` is disabled. Selection is
    `head(N)` after sorting by image_id -- deterministic and stable across epochs and seeds, so
    the SAME subsample is compared epoch to epoch; a re-sampled subset each epoch would make the
    gap trend noisy for reasons unrelated to overfitting.
    """
    data = cfg.get("data", {})
    man = pd.read_csv(data.get("preprocess_manifest", MANIFEST))
    train_all = man[man["split"] == "train"]
    datasets = data.get("datasets")
    if datasets:
        train_all = train_all[train_all["dataset"].isin(datasets)]

    tr_df, val_df = group_aware_val_split(
        train_all, float(data.get("val_fraction", 0.15)),
        int(data.get("val_split_seed", 42)))

    aug_cfg = cfg.get("augmentation", {})
    augmenter = None
    if aug_cfg.get("enabled", True):
        from src.datasets.augmentations import AugmentationConfig
        fields = {k: v for k, v in aug_cfg.items()
                  if k in AugmentationConfig.__dataclass_fields__}
        augmenter = Augmenter(seed=augmentation_seed, config=AugmentationConfig(**fields))

    return_instances = bool(data.get("return_instances", True))
    tr_ds = CellSegmentationDataset(tr_df, augmenter, return_instances)
    va_ds = CellSegmentationDataset(val_df, None, return_instances)

    bs = int(cfg.get("training", {}).get("batch_size", 4))
    nw = int(data.get("num_workers", 0))
    common = dict(num_workers=nw, collate_fn=collate_batch,
                  pin_memory=bool(data.get("pin_memory", False)))
    if nw > 0:
        common["worker_init_fn"] = seed_worker

    train_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                              generator=make_generator(data_order_seed), **common)
    val_loader = DataLoader(va_ds, batch_size=bs, shuffle=False, **common)

    train_eval_loader = None
    tr_cfg = cfg.get("training", {})
    if bool(tr_cfg.get("track_train_val_gap", False)) and len(tr_df):
        n_sub = min(int(tr_cfg.get("train_metrics_subsample_size", 40)), len(tr_df))
        sub_df = tr_df.sort_values("image_id").head(n_sub)
        te_ds = CellSegmentationDataset(sub_df, None, return_instances)
        train_eval_loader = DataLoader(te_ds, batch_size=bs, shuffle=False, **common)

    return train_loader, val_loader, train_eval_loader
