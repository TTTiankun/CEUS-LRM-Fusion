"""Plotting helpers for the Clinical-LR branch."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], class_names)
    ax.set_yticks([0, 1], class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout()
    return fig


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(fpr, tpr, linewidth=2)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    fig.tight_layout()
    return fig


def plot_pr(y_true: np.ndarray, y_prob: np.ndarray):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    fig.tight_layout()
    return fig


def plot_coefficients(coefficient_frame: pd.DataFrame, top_k: int = 20):
    ranking = coefficient_frame.reindex(
        coefficient_frame["coef_mean"].abs().sort_values(ascending=False).index
    ).head(top_k)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(ranking))))
    ax.barh(
        ranking["feature_name"],
        ranking["coef_mean"],
        xerr=[
            ranking["coef_mean"] - ranking["coef_ci_lower"],
            ranking["coef_ci_upper"] - ranking["coef_mean"],
        ],
        color="#3b82f6",
        alpha=0.9,
    )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Logistic coefficient")
    ax.set_title("Clinical-LR coefficients")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig
