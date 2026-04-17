"""Metrics used by the CEUS-GRU branch."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Estimate expected calibration error for binary probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        if not np.any(mask):
            continue
        bin_accuracy = y_true[mask].mean()
        bin_confidence = y_prob[mask].mean()
        ece += np.abs(bin_accuracy - bin_confidence) * mask.mean()
    return float(ece)


def binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """Return the primary binary metrics used in the manuscript."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "ppv": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    if y_prob is None:
        metrics.update({"auc": None, "average_precision": None, "brier_score": None, "ece": None})
        return metrics

    y_prob = np.asarray(y_prob, dtype=float)
    metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    metrics["brier_score"] = float(np.mean((y_prob - y_true) ** 2))
    metrics["ece"] = expected_calibration_error(y_true, y_prob)
    return metrics


def optimal_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute the validation-set threshold that maximizes Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float))
    j_scores = tpr - fpr
    best_index = int(np.nanargmax(j_scores))
    return {
        "threshold": float(thresholds[best_index]),
        "youden_j": float(j_scores[best_index]),
        "best_index": best_index,
    }
