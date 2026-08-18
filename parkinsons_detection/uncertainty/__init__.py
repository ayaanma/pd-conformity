"""Calibration, conformal prediction, and subject-level study helpers."""

from .calibration import ProbabilityCalibrator
from .conformal import SplitConformalClassifier, raw_conformal_prediction_sets
from .metrics import (
    bootstrap_classification_intervals,
    bootstrap_conformal_intervals,
    classification_metrics,
    conformal_metrics,
    wilson_interval,
)

__all__ = [
    "ProbabilityCalibrator",
    "SplitConformalClassifier",
    "raw_conformal_prediction_sets",
    "bootstrap_classification_intervals",
    "bootstrap_conformal_intervals",
    "classification_metrics",
    "conformal_metrics",
    "wilson_interval",
]
