"""Binary split-conformal prediction sets."""

from __future__ import annotations

import math

import numpy as np


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("At least one calibration score is required.")
    level = min(1.0, math.ceil((scores.size + 1) * (1 - alpha)) / scores.size)
    try:
        return float(np.quantile(scores, level, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(scores, level, interpolation="higher"))


class SplitConformalClassifier:
    """LAC, class-conditional LAC, or APS prediction sets for two classes."""

    def __init__(self, method: str = "lac", alpha: float = 0.1):
        if method not in {"lac", "mondrian_lac", "aps"}:
            raise ValueError(f"Unknown conformal method: {method}")
        self.method = method
        self.alpha = alpha
        self.quantiles: dict[int | str, float] = {}

    @staticmethod
    def _matrix(positive_probabilities: np.ndarray) -> np.ndarray:
        p1 = np.clip(np.asarray(positive_probabilities, dtype=float), 0, 1)
        return np.column_stack([1 - p1, p1])

    def fit(
        self, positive_probabilities: np.ndarray, y_true: np.ndarray
    ) -> "SplitConformalClassifier":
        probabilities = self._matrix(positive_probabilities)
        y_true = np.asarray(y_true, dtype=int)
        if self.method in {"lac", "mondrian_lac"}:
            scores = 1 - probabilities[np.arange(len(y_true)), y_true]
        else:
            predicted = np.argmax(probabilities, axis=1)
            scores = np.where(
                predicted == y_true,
                probabilities[np.arange(len(y_true)), predicted],
                1.0,
            )
        if self.method == "mondrian_lac":
            for label in (0, 1):
                self.quantiles[label] = _finite_sample_quantile(
                    scores[y_true == label], self.alpha
                )
        else:
            self.quantiles["all"] = _finite_sample_quantile(scores, self.alpha)
        return self

    def predict_sets(self, positive_probabilities: np.ndarray) -> np.ndarray:
        if not self.quantiles:
            raise RuntimeError("Conformal classifier must be fitted first.")
        probabilities = self._matrix(positive_probabilities)
        sets = np.zeros_like(probabilities, dtype=bool)
        if self.method == "lac":
            sets = (1 - probabilities) <= self.quantiles["all"]
        elif self.method == "mondrian_lac":
            for label in (0, 1):
                sets[:, label] = (
                    1 - probabilities[:, label] <= self.quantiles[label]
                )
        else:
            top = np.argmax(probabilities, axis=1)
            top_probability = probabilities[np.arange(len(top)), top]
            sets[np.arange(len(top)), top] = True
            include_other = top_probability < self.quantiles["all"]
            sets[np.arange(len(top))[include_other], 1 - top[include_other]] = True
        return sets
