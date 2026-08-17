"""End-to-end uncertainty-aware Parkinson's voice study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .data import load_voice_dataset
from .evaluation import (
    aggregate_features,
    aggregate_probabilities,
    compare_models_grouped_cv,
    make_study_split,
    positive_class_scores,
    subject_table,
)
from .models import build_models
from .uncertainty.calibration import ProbabilityCalibrator
from .uncertainty.conformal import SplitConformalClassifier
from .uncertainty.metrics import classification_metrics, conformal_metrics, wilson_interval


@dataclass(frozen=True)
class ExperimentConfig:
    data_path: str | None = None
    output_dir: str = "artifacts"
    calibration_size: float = 0.2
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    selected_models: list[str] | None = None
    alpha_levels: tuple[float, ...] = (0.05, 0.1, 0.2)
    leakage_repeats: int = 5
    save_plots: bool = True
    save_paper_outputs: bool = True
    save_model: bool = True


def _fit_patient_probabilities(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_evaluation: pd.DataFrame,
    y_evaluation: pd.Series,
    groups_evaluation: pd.Series,
    demographics_evaluation: pd.DataFrame,
) -> tuple[BaseEstimator, pd.DataFrame]:
    model = clone(estimator).fit(X_train, y_train)
    scores = positive_class_scores(model, X_evaluation)
    patients = aggregate_probabilities(
        scores, y_evaluation, groups_evaluation, demographics_evaluation
    )
    return model, patients


def _leakage_sensitivity(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    repeats: int,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    indices = np.arange(len(y))
    for repeat in range(repeats):
        seed = random_state + 100 + repeat
        fit_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=y, random_state=seed
        )
        model = clone(estimator).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        metrics = classification_metrics(
            y.iloc[test_idx].to_numpy(), positive_class_scores(model, X.iloc[test_idx])
        )
        rows.append({"protocol": "random_recording", "repeat": repeat + 1, **metrics})

        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        fit_idx, test_idx = next(splitter.split(X, y, groups))
        model = clone(estimator).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        metrics = classification_metrics(
            y.iloc[test_idx].to_numpy(), positive_class_scores(model, X.iloc[test_idx])
        )
        rows.append({"protocol": "heldout_subjects", "repeat": repeat + 1, **metrics})
    return pd.DataFrame(rows)


def _print_summary(
    comparison: pd.DataFrame,
    calibration: pd.DataFrame,
    conformal: pd.DataFrame,
    aggregation: pd.DataFrame,
    best_model: str,
) -> None:
    display = comparison[["model", "cv_accuracy_mean", "cv_balanced_accuracy_mean", "cv_f1_mean", "cv_roc_auc_mean", "test_accuracy", "test_roc_auc"]].copy()
    for column in display.columns[1:]:
        display[column] = display[column].map(lambda value: f"{100 * value:.1f}%")
    print("=== SUBJECT-LEVEL MODEL COMPARISON ===")
    print(display.to_string(index=False))
    winner = comparison.iloc[0]
    margin = winner["cv_roc_auc_mean"] - comparison.iloc[1]["cv_roc_auc_mean"] if len(comparison) > 1 else 0
    print(f"\nBest model: {best_model.replace('_', ' ').title()}")
    print(f"Selection: grouped-CV ROC-AUC {winner['cv_roc_auc_mean']:.3f} (margin {margin:.3f})")
    print(f"Locked test: accuracy {winner['test_accuracy']:.3f}, ROC-AUC {winner['test_roc_auc']:.3f}")

    best_calibration = calibration[calibration["model"] == best_model][["method", "brier", "ece", "log_loss"]]
    print("\n=== PROBABILITY CALIBRATION ON LOCKED TEST SUBJECTS ===")
    print(best_calibration.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    primary = conformal[(conformal["method"] == "mondrian_lac") & np.isclose(conformal["alpha"], 0.1)].iloc[0]
    print("\n=== PRIMARY CONFORMAL RESULT (CLASS-CONDITIONAL LAC, 90% TARGET) ===")
    print(f"Coverage: {primary['coverage']:.3f} | healthy: {primary['healthy_coverage']:.3f} | PD: {primary['pd_coverage']:.3f}")
    print(f"Abstention: {primary['abstention_rate']:.3f} | selective accuracy: {primary['selective_accuracy']:.3f} | selective balanced accuracy: {primary['selective_balanced_accuracy']:.3f}")
    print("\n=== REPEATED-RECORDING AGGREGATION ===")
    print(aggregation[["strategy", "accuracy", "roc_auc", "brier", "coverage", "abstention_rate"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    dataset = load_voice_dataset(config.data_path)
    models = build_models(config.random_state, config.selected_models)
    split = make_study_split(
        dataset.target, dataset.groups, dataset.demographics,
        config.calibration_size, config.test_size, config.random_state,
    )
    train_idx, calibration_idx, test_idx = (
        split.train_indices, split.calibration_indices, split.test_indices
    )
    X_train, y_train, groups_train = (
        dataset.features.iloc[train_idx], dataset.target.iloc[train_idx], dataset.groups.iloc[train_idx]
    )

    comparison, fold_scores, actual_folds = compare_models_grouped_cv(
        models, X_train, y_train, groups_train, config.cv_folds, config.random_state
    )
    best_model = str(comparison.iloc[0]["model"])
    fitted: dict[str, BaseEstimator] = {}
    patient_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    test_metric_rows: list[dict[str, float | str]] = []
    calibration_rows: list[dict[str, float | str]] = []
    best_calibration_predictions: list[pd.DataFrame] = []

    for model_name, estimator in models.items():
        model, calibration_patients = _fit_patient_probabilities(
            estimator, X_train, y_train,
            dataset.features.iloc[calibration_idx], dataset.target.iloc[calibration_idx],
            dataset.groups.iloc[calibration_idx], dataset.demographics.iloc[calibration_idx],
        )
        test_scores = positive_class_scores(model, dataset.features.iloc[test_idx])
        test_patients = aggregate_probabilities(
            test_scores, dataset.target.iloc[test_idx], dataset.groups.iloc[test_idx],
            dataset.demographics.iloc[test_idx],
        )
        fitted[model_name] = model
        patient_frames[model_name] = (calibration_patients, test_patients)
        raw_metrics = classification_metrics(test_patients["class"], test_patients["probability"])
        test_metric_rows.append({"model": model_name, **{f"test_{key}": value for key, value in raw_metrics.items()}})
        for method in ("uncalibrated", "sigmoid", "isotonic"):
            calibrator = ProbabilityCalibrator(method, config.random_state).fit(
                calibration_patients["probability"], calibration_patients["class"]
            )
            probabilities = calibrator.predict(test_patients["probability"])
            metrics = classification_metrics(test_patients["class"], probabilities)
            calibration_rows.append({"model": model_name, "method": method, **metrics})
            if model_name == best_model:
                frame = test_patients[["subject_id", "class", "sex"]].copy()
                frame["method"] = method
                frame["probability"] = probabilities
                best_calibration_predictions.append(frame)

    comparison = comparison.merge(pd.DataFrame(test_metric_rows), on="model", validate="one_to_one")
    comparison = comparison.sort_values("cv_roc_auc_mean", ascending=False, ignore_index=True)
    calibration_results = pd.DataFrame(calibration_rows)
    calibration_predictions = pd.concat(best_calibration_predictions, ignore_index=True)

    calibration_patients, test_patients = patient_frames[best_model]
    primary_calibrator = ProbabilityCalibrator("sigmoid", config.random_state).fit(
        calibration_patients["probability"], calibration_patients["class"]
    )
    calibration_probabilities = primary_calibrator.predict(calibration_patients["probability"])
    test_probabilities = primary_calibrator.predict(test_patients["probability"])
    conformal_rows: list[dict[str, float | str]] = []
    primary_sets: np.ndarray | None = None
    for method in ("lac", "mondrian_lac", "aps"):
        for alpha in config.alpha_levels:
            conformal = SplitConformalClassifier(method, alpha).fit(
                calibration_probabilities, calibration_patients["class"].to_numpy()
            )
            prediction_sets = conformal.predict_sets(test_probabilities)
            labels = test_patients["class"].to_numpy()
            covered = prediction_sets[np.arange(len(labels)), labels]
            conformal_rows.append(
                {"method": method, "alpha": alpha, "target_coverage": 1 - alpha,
                 **conformal_metrics(labels, prediction_sets),
                 "healthy_coverage": float(covered[labels == 0].mean()),
                 "pd_coverage": float(covered[labels == 1].mean())}
            )
            if method == "mondrian_lac" and np.isclose(alpha, 0.1):
                primary_sets = prediction_sets
    conformal_results = pd.DataFrame(conformal_rows)
    if primary_sets is None:
        primary_conformal = SplitConformalClassifier("mondrian_lac", 0.1).fit(
            calibration_probabilities, calibration_patients["class"].to_numpy()
        )
        primary_sets = primary_conformal.predict_sets(test_probabilities)

    final_predictions = test_patients[["subject_id", "class", "gender", "sex"]].copy()
    final_predictions["probability"] = test_probabilities
    final_predictions["set_healthy"] = primary_sets[:, 0]
    final_predictions["set_pd"] = primary_sets[:, 1]
    final_predictions["set_size"] = primary_sets.sum(axis=1)

    selective_rows = []
    confidence = np.maximum(test_probabilities, 1 - test_probabilities)
    forced_labels = (test_probabilities >= 0.5).astype(int)
    for threshold in np.linspace(0.5, 0.95, 10):
        selected = confidence >= threshold
        selective_rows.append({
            "threshold": threshold,
            "coverage": float(selected.mean()),
            "selective_accuracy": float(np.mean(forced_labels[selected] == test_patients.loc[selected, "class"])) if selected.any() else np.nan,
        })
    selective_curve = pd.DataFrame(selective_rows).dropna()

    aggregation_rows: list[dict[str, float | str]] = []
    aggregation_details: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "mean_probability": (calibration_patients.copy(), test_patients.copy()),
        "majority_vote": (
            aggregate_probabilities(
                positive_class_scores(fitted[best_model], dataset.features.iloc[calibration_idx]),
                dataset.target.iloc[calibration_idx], dataset.groups.iloc[calibration_idx],
                dataset.demographics.iloc[calibration_idx], strategy="majority_vote",
            ),
            aggregate_probabilities(
                positive_class_scores(fitted[best_model], dataset.features.iloc[test_idx]),
                dataset.target.iloc[test_idx], dataset.groups.iloc[test_idx],
                dataset.demographics.iloc[test_idx], strategy="majority_vote",
            ),
        ),
    }
    train_mean = aggregate_features(X_train, y_train, groups_train)
    calibration_mean = aggregate_features(
        dataset.features.iloc[calibration_idx], dataset.target.iloc[calibration_idx],
        dataset.groups.iloc[calibration_idx], dataset.demographics.iloc[calibration_idx],
    )
    test_mean = aggregate_features(
        dataset.features.iloc[test_idx], dataset.target.iloc[test_idx],
        dataset.groups.iloc[test_idx], dataset.demographics.iloc[test_idx],
    )
    feature_model = clone(models[best_model]).fit(train_mean[0], train_mean[1])
    feature_calibration = pd.DataFrame({
        "subject_id": calibration_mean[2], "class": calibration_mean[1],
        "probability": positive_class_scores(feature_model, calibration_mean[0]),
        "gender": calibration_mean[3]["gender"], "sex": calibration_mean[3]["sex"],
    })
    feature_test = pd.DataFrame({
        "subject_id": test_mean[2], "class": test_mean[1],
        "probability": positive_class_scores(feature_model, test_mean[0]),
        "gender": test_mean[3]["gender"], "sex": test_mean[3]["sex"],
    })
    aggregation_details["feature_mean"] = (feature_calibration, feature_test)
    for strategy, (calibration_frame, test_frame) in aggregation_details.items():
        calibrator = ProbabilityCalibrator("sigmoid", config.random_state).fit(
            calibration_frame["probability"], calibration_frame["class"]
        )
        cal_probs = calibrator.predict(calibration_frame["probability"])
        probs = calibrator.predict(test_frame["probability"])
        conformal = SplitConformalClassifier("mondrian_lac", 0.1).fit(cal_probs, calibration_frame["class"])
        sets = conformal.predict_sets(probs)
        aggregation_rows.append({"strategy": strategy, **classification_metrics(test_frame["class"], probs), **conformal_metrics(test_frame["class"], sets)})
    aggregation_results = pd.DataFrame(aggregation_rows)

    subgroup_rows: list[dict[str, float | str | int]] = []
    subgroup_specs = [
        ("Sex", "sex", {"Female": "Female", "Male": "Male"}),
        ("Diagnosis", "class", {0: "Healthy", 1: "PD"}),
    ]
    for attribute, column, labels in subgroup_specs:
        for value, part in final_predictions.groupby(column):
            covered = np.where(part["class"].to_numpy() == 1, part["set_pd"], part["set_healthy"]).astype(bool)
            low, high = wilson_interval(int(covered.sum()), len(part))
            singleton = part["set_size"] == 1
            predicted = part.loc[singleton, "set_pd"].astype(int)
            subgroup_rows.append({
                "attribute": attribute, "subgroup": labels[value], "subjects": len(part),
                "coverage": float(covered.mean()), "coverage_ci_low": low,
                "coverage_ci_high": min(high, 1.0), "abstention_rate": float((~singleton).mean()),
                "selective_accuracy": float(np.mean(predicted == part.loc[singleton, "class"])) if singleton.any() else np.nan,
            })
    subgroup_results = pd.DataFrame(subgroup_rows)

    leakage_results = _leakage_sensitivity(
        models[best_model], dataset.features, dataset.target, dataset.groups,
        config.leakage_repeats, config.random_state,
    )

    split_counts = pd.DataFrame(
        {
            "subjects": [len(split.train_subjects), len(split.calibration_subjects), len(split.test_subjects)],
            "recordings": [len(train_idx), len(calibration_idx), len(test_idx)],
        }, index=["train", "calibration", "test"]
    )
    subjects = subject_table(dataset.target, dataset.groups, dataset.demographics)
    dataset_table = pd.DataFrame([
        {"cohort": "All", "subjects": len(subjects), "recordings": len(dataset.target), "pd_subjects": int(subjects["class"].sum()), "healthy_subjects": int((subjects["class"] == 0).sum())},
        *[{"cohort": name.title(), "subjects": int(row["subjects"]), "recordings": int(row["recordings"]), "pd_subjects": int(subjects[subjects["subject_id"].isin(getattr(split, f"{name}_subjects"))]["class"].sum()), "healthy_subjects": int((subjects[subjects["subject_id"].isin(getattr(split, f"{name}_subjects"))]["class"] == 0).sum())} for name, row in split_counts.iterrows()],
    ])

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model_comparison.csv": comparison,
        "cross_validation_fold_scores.csv": fold_scores,
        "calibration_results.csv": calibration_results,
        "conformal_results.csv": conformal_results,
        "aggregation_results.csv": aggregation_results,
        "subgroup_results.csv": subgroup_results,
        "leakage_sensitivity.csv": leakage_results,
        "test_subject_predictions.csv": final_predictions,
        "selective_risk_curve.csv": selective_curve,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset_source": dataset.source,
        "recordings": len(dataset.target), "subjects": int(dataset.groups.nunique()),
        "acoustic_features": dataset.features.shape[1], "selected_model": best_model,
        "selection_metric": "patient-level grouped-CV ROC-AUC", "cross_validation_folds": actual_folds,
        "split_counts": split_counts.to_dict(orient="index"), "config": asdict(config),
        "limitations": ["Single public dataset", "Sex is the only available demographic subgroup", "Research use only; no clinical validation"],
    }
    (output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if config.save_model:
        joblib.dump({"acoustic_model": fitted[best_model], "probability_calibrator": primary_calibrator, "feature_names": list(dataset.features.columns)}, output_dir / "best_uncertainty_model.joblib")
    if config.save_plots and config.save_paper_outputs:
        from .uncertainty_figures import generate_uncertainty_paper_outputs

        generate_uncertainty_paper_outputs(output_dir / "conformal_paper", {
            "split_counts": split_counts, "dataset_table": dataset_table,
            "model_comparison": comparison, "best_model": best_model,
            "calibration_results": calibration_results,
            "calibration_predictions": calibration_predictions,
            "conformal_results": conformal_results, "selective_curve": selective_curve,
            "aggregation_results": aggregation_results, "subgroup_results": subgroup_results,
            "final_predictions": final_predictions, "leakage_results": leakage_results,
        })
    _print_summary(comparison, calibration_results, conformal_results, aggregation_results, best_model)
    print(f"\nArtifacts: {output_dir}")
    return comparison
