"""Early stopping helper."""

from __future__ import annotations


class EarlyStopping:
    """Track the best validation score and stop after patience is exhausted."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 0.0) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def step(self, value: float) -> bool:
        if self.best_score is None:
            self.best_score = value
            return False
        if self.mode == "max":
            improved = value > self.best_score + self.min_delta
        elif self.mode == "min":
            improved = value < self.best_score - self.min_delta
        else:
            raise ValueError("mode must be 'max' or 'min'")

        if improved:
            self.best_score = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
