"""Post-hoc probability calibration fitted on held-out subjects."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Identity, Platt/sigmoid, or isotonic calibration for binary scores."""

    def __init__(self, method: str = "sigmoid", random_state: int = 42):
        if method not in {"uncalibrated", "sigmoid", "isotonic"}:
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self.random_state = random_state
        self.model: LogisticRegression | IsotonicRegression | None = None

    @staticmethod
    def _clip(probabilities: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)

    @classmethod
    def _logits(cls, probabilities: np.ndarray) -> np.ndarray:
        probabilities = cls._clip(probabilities)
        return np.log(probabilities / (1 - probabilities)).reshape(-1, 1)

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        probabilities = self._clip(probabilities)
        y_true = np.asarray(y_true, dtype=int)
        if len(np.unique(y_true)) != 2:
            raise ValueError("Calibration requires examples from both classes.")
        if self.method == "sigmoid":
            self.model = LogisticRegression(
                solver="lbfgs", random_state=self.random_state
            ).fit(self._logits(probabilities), y_true)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(
                probabilities, y_true
            )
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = self._clip(probabilities)
        if self.method == "uncalibrated":
            return probabilities
        if self.model is None:
            raise RuntimeError("Calibrator must be fitted before predict().")
        if self.method == "sigmoid":
            calibrated = self.model.predict_proba(self._logits(probabilities))[:, 1]
        else:
            calibrated = self.model.predict(probabilities)
        return self._clip(calibrated)
