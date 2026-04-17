"""Data alignment utilities for case-level fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def _read_prediction_csv(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {"sample_name", "prob_hcc"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def load_fusion_table(
    ceus_csv: str,
    clinical_csv: str,
    label_csv: str | None = None,
) -> pd.DataFrame:
    """Join CEUS and clinical case-level probabilities on sample_name."""
    ceus_frame = _read_prediction_csv(ceus_csv)[["sample_name", "prob_hcc"]].rename(
        columns={"prob_hcc": "ceus_prob_hcc"}
    )
    clinical_frame = _read_prediction_csv(clinical_csv)[["sample_name", "prob_hcc"]].rename(
        columns={"prob_hcc": "clinical_prob_hcc"}
    )
    merged = ceus_frame.merge(clinical_frame, on="sample_name", how="inner")
    if label_csv:
        labels = pd.read_csv(label_csv)
        if not {"sample_name", "label"}.issubset(labels.columns):
            raise ValueError("Label CSV must contain sample_name and label columns")
        merged = merged.merge(labels[["sample_name", "label"]], on="sample_name", how="inner")
    return merged
