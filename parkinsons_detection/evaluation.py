"""Leakage-safe subject splitting, aggregation, and model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .uncertainty.metrics import classification_metrics


@dataclass(frozen=True)
class StudySplit:
    train_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray
    train_subjects: np.ndarray
    calibration_subjects: np.ndarray
    test_subjects: np.ndarray


def subject_table(
    y: pd.Series, groups: pd.Series, demographics: pd.DataFrame | None = None
) -> pd.DataFrame:
    data: dict[str, object] = {"subject_id": groups, "class": y}
    if demographics is not None:
        data["gender"] = demographics["gender"]
        data["sex"] = demographics["sex"]
    table = pd.DataFrame(data).groupby("subject_id", as_index=False).first()
    return table.sort_values("subject_id", ignore_index=True)


def _strata(table: pd.DataFrame) -> pd.Series:
    if "gender" in table:
        combined = table["class"].astype(str) + "_" + table["gender"].astype(str)
        if combined.value_counts().min() >= 2:
            return combined
    return table["class"]


def make_study_split(
    y: pd.Series,
    groups: pd.Series,
    demographics: pd.DataFrame,
    calibration_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> StudySplit:
    """Create stratified, disjoint training/calibration/test subject sets."""

    if calibration_size <= 0 or test_size <= 0 or calibration_size + test_size >= 1:
        raise ValueError("Calibration/test sizes must be positive and sum to less than one.")
    subjects = subject_table(y, groups, demographics)
    train, heldout = train_test_split(
        subjects,
        test_size=calibration_size + test_size,
        stratify=_strata(subjects),
        random_state=random_state,
    )
    relative_test_size = test_size / (calibration_size + test_size)
    calibration, test = train_test_split(
        heldout,
        test_size=relative_test_size,
        stratify=_strata(heldout),
        random_state=random_state + 1,
    )
    train_subjects = train["subject_id"].to_numpy()
    calibration_subjects = calibration["subject_id"].to_numpy()
    test_subjects = test["subject_id"].to_numpy()
    return StudySplit(
        train_indices=np.flatnonzero(groups.isin(train_subjects).to_numpy()),
        calibration_indices=np.flatnonzero(groups.isin(calibration_subjects).to_numpy()),
        test_indices=np.flatnonzero(groups.isin(test_subjects).to_numpy()),
        train_subjects=train_subjects,
        calibration_subjects=calibration_subjects,
        test_subjects=test_subjects,
    )


def positive_class_scores(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        if hasattr(model, "predict_proba"):
            scores = np.asarray(model.predict_proba(X))[:, 1]
        elif hasattr(model, "decision_function"):
            raw = np.asarray(model.decision_function(X), dtype=float)
            scores = 1 / (1 + np.exp(-np.clip(raw, -30, 30)))
        else:
            raise TypeError(f"{type(model).__name__} cannot produce ranking scores.")
    if not np.isfinite(scores).all():
        raise ValueError(f"{type(model).__name__} produced non-finite scores.")
    return scores


def aggregate_probabilities(
    probabilities: np.ndarray,
    y: pd.Series,
    groups: pd.Series,
    demographics: pd.DataFrame | None = None,
    strategy: str = "mean_probability",
) -> pd.DataFrame:
    """Collapse recording predictions into one row per subject."""

    frame = pd.DataFrame(
        {
            "subject_id": groups.to_numpy(),
            "class": y.to_numpy(),
            "probability": probabilities,
        }
    )
    if demographics is not None:
        frame["gender"] = demographics["gender"].to_numpy()
        frame["sex"] = demographics["sex"].to_numpy()
    if strategy == "mean_probability":
        aggregate = frame.groupby("subject_id", as_index=False).agg(
            {"class": "first", "probability": "mean"}
        )
    elif strategy == "majority_vote":
        frame["vote"] = (frame["probability"] >= 0.5).astype(float)
        aggregate = frame.groupby("subject_id", as_index=False).agg(
            {"class": "first", "vote": "mean"}
        ).rename(columns={"vote": "probability"})
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")
    if demographics is not None:
        demo = frame.groupby("subject_id", as_index=False)[["gender", "sex"]].first()
        aggregate = aggregate.merge(demo, on="subject_id", validate="one_to_one")
    return aggregate.sort_values("subject_id", ignore_index=True)


def aggregate_features(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    demographics: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame | None]:
    frame = X.copy()
    frame["__subject_id"] = groups.to_numpy()
    mean_features = frame.groupby("__subject_id", sort=True).mean()
    labels = pd.DataFrame({"subject_id": groups, "class": y}).groupby(
        "subject_id"
    )["class"].first().reindex(mean_features.index)
    subject_ids = pd.Series(mean_features.index, name="subject_id").reset_index(drop=True)
    mean_features = mean_features.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    demo_result = None
    if demographics is not None:
        demo_frame = demographics.copy()
        demo_frame["subject_id"] = groups.to_numpy()
        demo_result = demo_frame.groupby("subject_id")[["gender", "sex"]].first()
        demo_result = demo_result.reindex(subject_ids).reset_index(drop=True)
    return mean_features, labels, subject_ids, demo_result


def compare_models_grouped_cv(
    models: dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    folds: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Select models from patient-level validation predictions only."""

    counts = subject_table(y, groups)["class"].value_counts()
    actual_folds = min(folds, int(counts.min()))
    if actual_folds < 2:
        raise ValueError("Grouped CV needs at least two subjects in each class.")
    splitter = StratifiedGroupKFold(
        n_splits=actual_folds, shuffle=True, random_state=random_state
    )
    fold_rows: list[dict[str, float | str | int]] = []
    for model_name, estimator in models.items():
        for fold, (fit_idx, validation_idx) in enumerate(
            splitter.split(X, y, groups), start=1
        ):
            started = perf_counter()
            model = clone(estimator).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            elapsed = perf_counter() - started
            recording_scores = positive_class_scores(model, X.iloc[validation_idx])
            patient = aggregate_probabilities(
                recording_scores, y.iloc[validation_idx], groups.iloc[validation_idx]
            )
            metrics = classification_metrics(
                patient["class"].to_numpy(), patient["probability"].to_numpy()
            )
            fold_rows.append(
                {"model": model_name, "fold": fold, "fit_time_seconds": elapsed, **metrics}
            )
    fold_results = pd.DataFrame(fold_rows)
    rows: list[dict[str, float | str]] = []
    metric_names = [
        "accuracy", "balanced_accuracy", "precision", "recall", "f1",
        "roc_auc", "brier", "log_loss", "ece",
    ]
    for model_name, values in fold_results.groupby("model", sort=False):
        row: dict[str, float | str] = {"model": model_name}
        for metric in metric_names:
            row[f"cv_{metric}_mean"] = float(values[metric].mean())
            row[f"cv_{metric}_std"] = float(values[metric].std(ddof=0))
        row["cv_fit_time_mean"] = float(values["fit_time_seconds"].mean())
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values(
        "cv_roc_auc_mean", ascending=False, ignore_index=True
    )
    return comparison, fold_results, actual_folds
