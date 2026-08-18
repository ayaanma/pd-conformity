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
    """LAC, class-conditional LAC, or deterministic APS sets for two classes."""

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

    @staticmethod
    def _aps_scores(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Return deterministic APS ranked cumulative-mass scores."""

        order = np.argsort(-probabilities, axis=1, kind="mergesort")
        sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
        cumulative = np.cumsum(sorted_probabilities, axis=1)
        ranks = np.argmax(order == labels[:, None], axis=1)
        return cumulative[np.arange(len(labels)), ranks]

    def fit(
        self, positive_probabilities: np.ndarray, y_true: np.ndarray
    ) -> "SplitConformalClassifier":
        probabilities = self._matrix(positive_probabilities)
        y_true = np.asarray(y_true, dtype=int)
        if self.method in {"lac", "mondrian_lac"}:
            scores = 1 - probabilities[np.arange(len(y_true)), y_true]
        else:
            scores = self._aps_scores(probabilities, y_true)
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
            # Deterministic cumulative-mass APS based on Romano et al.: sort
            # labels by descending probability without randomized boundary mass,
            # then include labels while cumulative mass before the candidate is
            # below q_hat. This includes the first boundary label reaching it.
            order = np.argsort(-probabilities, axis=1, kind="mergesort")
            sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
            cumulative_before = np.cumsum(sorted_probabilities, axis=1) - sorted_probabilities
            include_sorted = cumulative_before < self.quantiles["all"]
            np.put_along_axis(sets, order, include_sorted, axis=1)
        return sets


def raw_conformal_prediction_sets(
    method: str,
    alpha: float,
    *,
    raw_calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    raw_test_probabilities: np.ndarray,
) -> np.ndarray:
    """Fit and apply conformal prediction using raw model probabilities only.

    The keyword-only API intentionally makes the probability source explicit
    and prevents probability calibration from being hidden inside this path.
    """

    conformal = SplitConformalClassifier(method, alpha).fit(
        raw_calibration_probabilities, calibration_labels
    )
    return conformal.predict_sets(raw_test_probabilities)
