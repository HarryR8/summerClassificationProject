"""DataLoader creation for SICM.

This module wires together:
  - SICMDataset  (model-agnostic; returns PIL.Image + metadata)
  - get_preprocess (model-specific: resize / augment / normalize / tensorize)
  - make_collate_fn (applies preprocess inside the DataLoader worker,
                     stacks tensors, and collects metadata lists)

Differs from the reference (busbra) loaders.py in one structural way:
create_dataloaders() here returns a 2-tuple (train, test), not a 3-tuple
(train, val, test) — this dataset's splits.csv only has "train"/"test"
rows. See src/sicm/data/prepare_data.py's module docstring for why there
is no val split (short version: too few independent sessions to keep
every split session-disjoint AND class-balanced with a three-way split).

Typical usage
-------------
    from sicm.data.loaders import create_dataloaders

    train_loader, test_loader = create_dataloaders(
        split_file="data/splits/splits.csv",
        images_dir="data/raw",
        model_key="imagenet_cnn",   # or "clip", "dinov2", "dinov3"
        batch_size=32,
        num_workers=4,
        size=224,
    )

    for batch in train_loader:
        images = batch["image"]     # (B, 3, H, W) float32
        labels = batch["label"]     # (B,)          int64
        cases  = batch["case"]      # list[str]
        ids    = batch["image_id"]  # list[str]
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from sicm.data.dataset import SICMDataset
from sicm.data.preprocessing import get_preprocess


# ---------------------------------------------------------------------------
# Collate function factory
# ---------------------------------------------------------------------------

class _CollateFn:
    """Picklable collate function that applies model-specific preprocessing.

    Using a class instead of a closure ensures this object can be pickled by
    Python 3.13+ multiprocessing (spawn/forkserver start methods), which cannot
    pickle local functions returned from factory functions.
    """

    def __init__(self, preprocess_fn: Callable) -> None:
        self.preprocess_fn = preprocess_fn

    def __call__(self, samples: list[dict]) -> dict:
        images = torch.stack([self.preprocess_fn(s["image"]) for s in samples])
        labels = torch.tensor([s["label"] for s in samples], dtype=torch.long)
        return {
            "image":    images,
            "label":    labels,
            "case":     [s["case"]     for s in samples],
            "image_id": [s["image_id"] for s in samples],
        }


def make_collate_fn(preprocess_fn: Callable) -> "_CollateFn":
    """Return a picklable collate function that applies `preprocess_fn` to each sample.

    Parameters
    ----------
    preprocess_fn : Callable[[PIL.Image], torch.FloatTensor]
        Model-specific transform returned by get_preprocess(…).

    Returns
    -------
    _CollateFn
        A collate callable compatible with torch DataLoader's
        ``collate_fn`` argument.  The returned batch dict has:
            "image"    : (B, 3, H, W)  float32 tensor
            "label"    : (B,)          int64 tensor
            "case"     : list[str]     length B
            "image_id" : list[str]     length B
    """
    return _CollateFn(preprocess_fn)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    split_file: str,
    images_dir: str,
    model_key: str = "imagenet_cnn",
    batch_size: int = 32,
    num_workers: int = 4,
    size: int = 224,
    masks_dir: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train / test DataLoaders (no val — see module docstring).

    Parameters
    ----------
    split_file : str
        Path to splits.csv produced by prepare_data.py.
    images_dir : str
        Directory containing <ID>.png image files.
    model_key : str
        Backbone key passed to get_preprocess.
        One of "imagenet_cnn", "clip", "dinov2", "dinov3".
    batch_size : int
        Mini-batch size.
    num_workers : int
        DataLoader worker processes.
    size : int
        Target image size (used by imagenet_cnn pipeline).
    masks_dir : str or None
        Unsupported for SICM data (no lesion-mask equivalent) — must be
        None. Kept as a parameter for interface parity with the reference
        pipeline's CLI flags; SICMDataset raises if it's set.

    Returns
    -------
    (train_loader, test_loader)
    """
    # Build model-specific preprocessors per split
    preprocess_train = get_preprocess(model_key, split="train", size=size)
    preprocess_test  = get_preprocess(model_key, split="test",  size=size)

    # Datasets return raw PIL.Image; no transform stored inside Dataset
    datasets = {
        split: SICMDataset(split_file, images_dir, split, masks_dir=masks_dir)
        for split in ["train", "test"]
    }

    # Weighted sampler on training set to compensate for class imbalance
    train_labels = datasets["train"].df["label"].values
    class_counts  = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(
        datasets["train"],
        sampler=sampler,
        drop_last=True,
        collate_fn=make_collate_fn(preprocess_train),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        datasets["test"],
        collate_fn=make_collate_fn(preprocess_test),
        **loader_kwargs,
    )

    return train_loader, test_loader
