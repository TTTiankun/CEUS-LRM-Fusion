"""Data utilities for the Clinical-LR branch."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_NAMES = [
    "age",
    "hepatitis_history",
    "non_hepatic_tumor",
    "afp_category",
    "afp_value",
    "ca199_category",
    "ca125_category",
    "hbsag",
    "hbcab",
    "hcsag",
]

NUMERIC_FEATURES = ["age", "afp_value"]
CATEGORICAL_FEATURES = [
    "hepatitis_history",
    "non_hepatic_tumor",
    "afp_category",
    "ca199_category",
    "ca125_category",
    "hbsag",
    "hbcab",
    "hcsag",
]


def parse_feature_file(file_path: Path) -> Dict[str, float]:
    raw_line = file_path.read_text(encoding="utf-8-sig").strip()
    values = [float(value) for value in raw_line.split()]
    if len(values) != len(FEATURE_NAMES):
        raise ValueError(f"{file_path} does not contain {len(FEATURE_NAMES)} features")
    return dict(zip(FEATURE_NAMES, values))


def build_preprocessor() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", encoder, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def augment_training_dataframe(
    frame: pd.DataFrame,
    labels: np.ndarray,
    sample_names: List[str],
    n_augment: int,
    noise_std: float,
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Augment continuous variables with mild Gaussian perturbation."""
    if n_augment <= 0:
        return frame, labels, sample_names
    rng = np.random.RandomState(seed)
    augmented_rows = []
    augmented_labels = []
    augmented_names = []
    for row_index, (_, row) in enumerate(frame.iterrows()):
        numeric_values = row[NUMERIC_FEATURES].to_numpy(dtype=float)
        categorical_values = row[CATEGORICAL_FEATURES].to_numpy(dtype=float)
        for copy_index in range(n_augment):
            noisy_numeric = numeric_values + rng.normal(0.0, noise_std, size=len(NUMERIC_FEATURES))
            augmented_rows.append(np.concatenate([noisy_numeric, categorical_values]))
            augmented_labels.append(labels[row_index])
            augmented_names.append(f"{sample_names[row_index]}__aug{copy_index+1}")
    combined = pd.concat(
        [
            frame,
            pd.DataFrame(augmented_rows, columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES),
        ],
        ignore_index=True,
    )
    combined_labels = np.concatenate([labels, np.asarray(augmented_labels)])
    combined_names = sample_names + augmented_names
    return combined, combined_labels, combined_names


def load_labeled_split(
    root_dir: str,
    positive_label: str = "HCC",
    augment: bool = False,
    augment_n: int = 0,
    augment_noise_std: float = 0.01,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    rows: List[Dict[str, float]] = []
    labels: List[int] = []
    sample_names: List[str] = []
    class_names = sorted([path.name for path in root.iterdir() if path.is_dir()])
    if len(class_names) != 2:
        raise ValueError(f"Clinical-LR expects exactly two classes, found: {class_names}")

    for class_name in class_names:
        class_dir = root / class_name
        for file_path in sorted(class_dir.glob("*.txt")):
            rows.append(parse_feature_file(file_path))
            labels.append(1 if class_name == positive_label else 0)
            sample_names.append(file_path.stem)

    frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
    label_array = np.asarray(labels, dtype=int)
    if augment:
        frame, label_array, sample_names = augment_training_dataframe(
            frame=frame,
            labels=label_array,
            sample_names=sample_names,
            n_augment=augment_n,
            noise_std=augment_noise_std,
            seed=seed,
        )
    ordered_class_names = [label for label in class_names if label != positive_label] + [positive_label]
    return frame, label_array, sample_names, ordered_class_names


def load_unlabeled_directory(input_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    rows = []
    sample_names = []
    for file_path in sorted(root.glob("*.txt")):
        rows.append(parse_feature_file(file_path))
        sample_names.append(file_path.stem)
    if not rows:
        raise RuntimeError(f"No .txt files found in {root}")
    return pd.DataFrame(rows, columns=FEATURE_NAMES), sample_names
