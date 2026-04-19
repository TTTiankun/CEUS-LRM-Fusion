"""Data helpers for the neural LRM-Fusion branch."""

from __future__ import annotations

from typing import Dict, Tuple

from torch.utils.data import DataLoader

from ceus_lrm_fusion.ceus.data import TimeSeriesDataset, pad_collate_fn


def build_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader | None, TimeSeriesDataset]:
    train_dataset = TimeSeriesDataset(
        directory=config["train_dir"],
        augment=bool(config.get("use_augmentation", True)),
        aug_cfg=config.get("augmentations", {}),
        confidence_cfg=config.get("confidence_suppression", {}),
    )
    val_dataset = None
    if config.get("val_dir"):
        val_dataset = TimeSeriesDataset(
            directory=config["val_dir"],
            label_map=train_dataset.label_map,
            augment=False,
            confidence_cfg=config.get("confidence_suppression", {}),
        )

    batch_size = int(config.get("batch_size", 32))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pad_collate_fn,
        )
    return train_loader, val_loader, train_dataset
