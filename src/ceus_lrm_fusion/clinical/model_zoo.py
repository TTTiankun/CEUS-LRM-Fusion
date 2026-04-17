"""Clinical-LR model helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample


def train_logistic_pipeline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float = 1.0,
    penalty: str = "l2",
    seed: int = 42,
    max_iter: int = 5000,
) -> LogisticRegression:
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    model = LogisticRegression(
        C=c_value,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def bootstrap_logistic_coefficients(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    c_value: float = 1.0,
    penalty: str = "l2",
    seed: int = 42,
    max_iter: int = 5000,
    n_bootstraps: int = 200,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Estimate coefficient confidence intervals with bootstrap resampling."""
    rng = np.random.RandomState(seed)
    coefficients = np.zeros((n_bootstraps, x_train.shape[1]), dtype=float)
    for index in range(n_bootstraps):
        x_resampled, y_resampled = resample(
            x_train,
            y_train,
            replace=True,
            random_state=rng.randint(0, 2**31 - 1),
        )
        model = train_logistic_pipeline(
            x_train=x_resampled,
            y_train=y_resampled,
            c_value=c_value,
            penalty=penalty,
            seed=seed + index,
            max_iter=max_iter,
        )
        coefficients[index] = model.coef_[0]

    lower = np.percentile(coefficients, alpha / 2 * 100, axis=0)
    upper = np.percentile(coefficients, (1 - alpha / 2) * 100, axis=0)
    mean = coefficients.mean(axis=0)
    return pd.DataFrame(
        {
            "feature_name": feature_names,
            "coef_mean": mean,
            "coef_ci_lower": lower,
            "coef_ci_upper": upper,
        }
    ).sort_values("coef_mean", ascending=False)
