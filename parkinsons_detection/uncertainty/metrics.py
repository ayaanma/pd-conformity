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


def _percentile_interval(values: list[float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_classification_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    replicates: int = 2000,
    random_state: int = 42,
) -> dict[str, float]:
    """Subject-bootstrap 95% intervals for key classification metrics."""

    if replicates < 1:
        raise ValueError("replicates must be at least one.")
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0, 1)
    if len(y_true) != len(probabilities) or len(y_true) == 0:
        raise ValueError("Labels and probabilities must be non-empty and aligned.")
    rng = np.random.default_rng(random_state)
    samples: dict[str, list[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "roc_auc": [],
        "brier": [],
    }
    for _ in range(replicates):
        indices = rng.integers(0, len(y_true), len(y_true))
        labels = y_true[indices]
        scores = probabilities[indices]
        predicted = (scores >= 0.5).astype(int)
        samples["accuracy"].append(float(np.mean(predicted == labels)))
        samples["brier"].append(float(np.mean((scores - labels) ** 2)))
        if np.unique(labels).size == 2:
            samples["balanced_accuracy"].append(
                float(balanced_accuracy_score(labels, predicted))
            )
            samples["roc_auc"].append(float(roc_auc_score(labels, scores)))
    intervals: dict[str, float] = {}
    for metric, values in samples.items():
        low, high = _percentile_interval(values)
        intervals[f"{metric}_ci_low"] = low
        intervals[f"{metric}_ci_high"] = high
    return intervals


def bootstrap_conformal_intervals(
    y_true: np.ndarray,
    prediction_sets: np.ndarray,
    replicates: int = 2000,
    random_state: int = 42,
) -> dict[str, float]:
    """Subject-bootstrap 95% intervals for coverage and selective metrics."""

    if replicates < 1:
        raise ValueError("replicates must be at least one.")
    y_true = np.asarray(y_true, dtype=int)
    prediction_sets = np.asarray(prediction_sets, dtype=bool)
    if prediction_sets.shape != (len(y_true), 2) or len(y_true) == 0:
        raise ValueError("Prediction sets must have shape (subjects, 2).")
    rng = np.random.default_rng(random_state)
    samples: dict[str, list[float]] = {
        "coverage": [],
        "abstention_rate": [],
        "selective_accuracy": [],
        "selective_balanced_accuracy": [],
    }
    for _ in range(replicates):
        indices = rng.integers(0, len(y_true), len(y_true))
        metrics = conformal_metrics(y_true[indices], prediction_sets[indices])
        for metric in samples:
            value = float(metrics[metric])
            if np.isfinite(value):
                samples[metric].append(value)
    intervals: dict[str, float] = {}
    for metric, values in samples.items():
        low, high = _percentile_interval(values)
        intervals[f"{metric}_ci_low"] = low
        intervals[f"{metric}_ci_high"] = high
    return intervals


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
