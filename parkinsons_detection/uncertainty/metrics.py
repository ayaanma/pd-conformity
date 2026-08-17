"""Patient-level classification, calibration, and prediction-set metrics."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            error += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(error)


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "ece": expected_calibration_error(y_true, probabilities),
    }


def conformal_metrics(y_true: np.ndarray, prediction_sets: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    sizes = prediction_sets.sum(axis=1)
    covered = prediction_sets[np.arange(len(y_true)), y_true]
    singleton = sizes == 1
    singleton_labels = np.argmax(prediction_sets[singleton], axis=1)
    selective_accuracy = (
        float(np.mean(singleton_labels == y_true[singleton])) if singleton.any() else np.nan
    )
    selective_balanced_accuracy = (
        float(balanced_accuracy_score(y_true[singleton], singleton_labels))
        if singleton.any() and len(np.unique(y_true[singleton])) == 2
        else np.nan
    )
    return {
        "coverage": float(covered.mean()),
        "average_set_size": float(sizes.mean()),
        "singleton_rate": float(singleton.mean()),
        "abstention_rate": float((~singleton).mean()),
        "ambiguous_rate": float((sizes == 2).mean()),
        "empty_rate": float((sizes == 0).mean()),
        "selective_accuracy": selective_accuracy,
        "selective_balanced_accuracy": selective_balanced_accuracy,
        "selected_subjects": int(singleton.sum()),
    }


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    radius = z * np.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return float(centre - radius), float(centre + radius)
