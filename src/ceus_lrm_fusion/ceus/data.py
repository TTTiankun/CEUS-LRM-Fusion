"""Data loading utilities for CEUS temporal sequences."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def add_gaussian_noise(array: np.ndarray, std: float) -> np.ndarray:
    return array + np.random.normal(0.0, std, size=array.shape).astype(np.float32)


def apply_time_mask(array: np.ndarray, mask_pct: float, max_mask_len: int) -> np.ndarray:
    masked = array.copy()
    sequence_length = masked.shape[0]
    num_masks = int(sequence_length * mask_pct)
    for _ in range(num_masks):
        start = np.random.randint(0, max(sequence_length, 1))
        span = np.random.randint(1, max_mask_len + 1)
        masked[start : start + span] = 0.0
    return masked


def apply_feature_mask(array: np.ndarray, feature_mask_pct: float) -> np.ndarray:
    masked = array.copy()
    feature_dim = masked.shape[1]
    num_features = int(feature_dim * feature_mask_pct)
    if num_features <= 0:
        return masked
    feature_index = np.random.choice(feature_dim, num_features, replace=False)
    masked[:, feature_index] = 0.0
    return masked


def apply_temporal_jitter(array: np.ndarray, window_size: int, jitter_prob: float) -> np.ndarray:
    if array.shape[0] < window_size or window_size <= 1:
        return array
    jittered = array.copy()
    for start in range(0, array.shape[0] - window_size + 1, window_size):
        if random.random() < jitter_prob:
            window = jittered[start : start + window_size].copy()
            np.random.shuffle(window)
            jittered[start : start + window_size] = window
    return jittered


def apply_time_warp(array: np.ndarray, stretch_sigma: float, stretch_prob: float) -> np.ndarray:
    if random.random() > stretch_prob:
        return array
    factor = float(np.clip(np.random.normal(1.0, stretch_sigma), 0.5, 2.0))
    sequence_length, feature_dim = array.shape
    warped_length = max(1, int(sequence_length * factor))
    original_index = np.linspace(0, sequence_length - 1, num=sequence_length)
    warped_index = np.linspace(0, sequence_length - 1, num=warped_length)
    warped = np.zeros((warped_length, feature_dim), dtype=np.float32)
    for feature in range(feature_dim):
        warped[:, feature] = np.interp(warped_index, original_index, array[:, feature])
    return warped


def apply_augmentations(array: np.ndarray, config: Optional[Dict]) -> np.ndarray:
    """Apply optional sequence-level augmentation during training."""
    if not config:
        return array
    augmented = array.copy()
    gaussian = config.get("gaussian_noise", {})
    if gaussian.get("enable"):
        augmented = add_gaussian_noise(augmented, std=float(gaussian.get("noise_std", 0.02)))
    time_mask = config.get("time_mask", {})
    if time_mask.get("enable"):
        augmented = apply_time_mask(
            augmented,
            mask_pct=float(time_mask.get("mask_pct", 0.1)),
            max_mask_len=int(time_mask.get("max_mask_len", 3)),
        )
    feature_mask = config.get("feature_mask", {})
    if feature_mask.get("enable"):
        augmented = apply_feature_mask(
            augmented,
            feature_mask_pct=float(feature_mask.get("feat_mask_pct", 0.1)),
        )
    temporal_jitter = config.get("temporal_jitter", {})
    if temporal_jitter.get("enable"):
        augmented = apply_temporal_jitter(
            augmented,
            window_size=int(temporal_jitter.get("window_size", 3)),
            jitter_prob=float(temporal_jitter.get("jitter_prob", 0.2)),
        )
    time_warp = config.get("time_warp", {})
    if time_warp.get("enable"):
        augmented = apply_time_warp(
            augmented,
            stretch_sigma=float(time_warp.get("stretch_sigma", 0.1)),
            stretch_prob=float(time_warp.get("stretch_prob", 0.2)),
        )
    return augmented


class TimeSeriesDataset(Dataset):
    """Dataset for one temporal CEUS sequence per file."""

    def __init__(
        self,
        directory: str,
        label_map: Optional[Dict[str, int]] = None,
        augment: bool = False,
        aug_cfg: Optional[Dict] = None,
    ) -> None:
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"Sequence directory not found: {self.directory}")

        self.files = sorted(
            path for path in self.directory.iterdir() if path.suffix.lower() in {".npz", ".txt"}
        )
        if not self.files:
            raise RuntimeError(f"No .npz or .txt files found in {self.directory}")

        self.label_map = dict(label_map or {})
        self.labels: List[int] = []
        for file_path in self.files:
            label_name = file_path.name.split("_")[0]
            if label_name not in self.label_map:
                if label_map is not None:
                    raise ValueError(f"Unknown label {label_name!r} in {file_path.name}")
                self.label_map[label_name] = len(self.label_map)
            self.labels.append(self.label_map[label_name])

        first_sample = self._load_sequence(self.files[0])
        self.feature_dim = int(first_sample.shape[1])
        self.augment = augment
        self.aug_cfg = aug_cfg or {}

    def _load_npz(self, file_path: Path) -> np.ndarray:
        with np.load(file_path) as content:
            arrays = [content[key] for key in sorted(content.files)]
        flattened = []
        for array in arrays:
            if array.ndim == 1:
                flattened.append(array[None, :])
            elif array.ndim > 2:
                flattened.append(array.reshape(array.shape[0], -1))
            else:
                flattened.append(array)
        return np.concatenate(flattened, axis=0).astype(np.float32)

    def _load_probability_txt(self, file_path: Path) -> np.ndarray:
        sequence: List[List[float]] = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            tokens = line.strip().split()
            if len(tokens) != 4:
                continue
            class_a, prob_a, class_b, prob_b = tokens
            probabilities = {
                class_a: float(prob_a),
                class_b: float(prob_b),
            }
            ordered = [probabilities.get(label_name, 0.0) for label_name, _ in sorted(self.label_map.items(), key=lambda item: item[1])]
            sequence.append(ordered)
        if not sequence:
            raise ValueError(f"No valid sequence rows found in {file_path}")
        return np.asarray(sequence, dtype=np.float32)

    def _load_sequence(self, file_path: Path) -> np.ndarray:
        if file_path.suffix.lower() == ".npz":
            return self._load_npz(file_path)
        return self._load_probability_txt(file_path)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int, str]:
        file_path = self.files[index]
        array = self._load_sequence(file_path)
        if self.augment:
            array = apply_augmentations(array, self.aug_cfg)
        tensor = torch.tensor(array, dtype=torch.float32)
        sample_name = file_path.stem
        return tensor, self.labels[index], tensor.shape[0], sample_name


def pad_collate_fn(
    batch: Iterable[Tuple[torch.Tensor, int, int, str]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Pad a variable-length sequence batch."""
    inputs, labels, lengths, names = zip(*batch)
    padded = pad_sequence(inputs, batch_first=True)
    return (
        padded,
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        list(names),
    )
