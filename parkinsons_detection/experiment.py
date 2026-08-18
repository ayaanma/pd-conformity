"""End-to-end uncertainty-aware Parkinson's voice study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import shutil
import subprocess

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .data import CSV_MIRROR_URL, EXPECTED_CSV_SHA256, UCI_DATASET_PAGE, load_voice_dataset
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
from .uncertainty.conformal import raw_conformal_prediction_sets
from .uncertainty.metrics import (
    bootstrap_classification_intervals,
    bootstrap_conformal_intervals,
    classification_metrics,
    conformal_metrics,
    wilson_interval,
)


PAPER_SOFTWARE_VERSIONS = {
    "python": "3.9.6",
    "numpy": "2.0.2",
    "scipy": "1.13.1",
    "pandas": "2.3.3",
    "scikit-learn": "1.6.1",
    "matplotlib": "3.9.4",
    "joblib": "1.5.3",
}

GENERATED_ROOT_FILES = {
    "aggregation_results.csv",
    "artifact_manifest.json",
    "best_uncertainty_model.joblib",
    "calibration_results.csv",
    "conformal_results.csv",
    "cross_validation_fold_scores.csv",
    "dataset_and_split.csv",
    "experiment_metadata.json",
    "leakage_sensitivity.csv",
    "model_comparison.csv",
    "selective_risk_curve.csv",
    "subgroup_results.csv",
    "test_subject_predictions.csv",
}

LEGACY_ROOT_FILES = {
    "best_model.joblib",
    "best_model_confusion_matrix.png",
    "classification_reports.json",
    "feature_importance.csv",
    "feature_importance.png",
    "model_comparison.png",
    "roc_curves.png",
}


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
    primary_alpha: float = 0.1
    bootstrap_replicates: int = 2000
    leakage_repeats: int = 5
    results_dir: str = "results/paper_v1"
    save_plots: bool = True
    save_paper_outputs: bool = True
    save_model: bool = True
    enforce_paper_environment: bool = True
    paper_snapshot_tag: str = "paper-v1"


def _software_versions() -> dict[str, str]:
    """Return the exact runtime versions that can affect paper results."""

    observed = {"python": platform.python_version()}
    for package in (
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "joblib",
    ):
        try:
            observed[package] = version(package)
        except PackageNotFoundError:
            observed[package] = "not-installed"
    return observed


def _validate_paper_environment(observed: dict[str, str]) -> None:
    mismatches = {
        name: {"expected": expected, "observed": observed.get(name)}
        for name, expected in PAPER_SOFTWARE_VERSIONS.items()
        if observed.get(name) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{name}={values['observed']} (expected {values['expected']})"
            for name, values in mismatches.items()
        )
        raise RuntimeError(
            "The paper experiment requires the locked environment: "
            f"{details}. Install requirements-paper.txt, or pass "
            "--allow-unlocked-environment for non-paper exploratory runs."
        )


def _git_provenance(snapshot_tag: str) -> dict[str, object]:
    """Capture the immutable source revision and relevant worktree state."""

    repository = Path(__file__).resolve().parents[1]

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )

    commit = git("rev-parse", "HEAD")
    if commit.returncode != 0:
        return {
            "source_commit_sha": "unavailable",
            "paper_snapshot_tag": snapshot_tag,
            "source_tags_at_run": [],
            "worktree_clean": None,
            "analysis_paths_clean": None,
        }
    tags = git("tag", "--points-at", "HEAD")
    all_status = git("status", "--porcelain=v1", "--untracked-files=all")
    analysis_status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "main.py",
        "README.md",
        ".gitignore",
        "pyproject.toml",
        "requirements.txt",
        "requirements-paper.txt",
        ".python-version",
        "parkinsons_detection",
        "tests",
        "docs",
        "results/paper_v1",
    )
    return {
        "source_commit_sha": commit.stdout.strip(),
        "paper_snapshot_tag": snapshot_tag,
        "source_tags_at_run": sorted(tags.stdout.splitlines()) if tags.returncode == 0 else [],
        "worktree_clean": all_status.returncode == 0 and not all_status.stdout.strip(),
        "analysis_paths_clean": (
            analysis_status.returncode == 0 and not analysis_status.stdout.strip()
        ),
    }


def _prepare_output_directory(output_dir: Path) -> None:
    """Remove only files and directories owned by this artifact pipeline."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in GENERATED_ROOT_FILES | LEGACY_ROOT_FILES:
        path = output_dir / filename
        if path.is_file() or path.is_symlink():
            path.unlink()
    for directory_name in ("conformal_paper", "paper"):
        path = output_dir / directory_name
        if path.exists():
            shutil.rmtree(path)


def _write_artifact_manifest(output_dir: Path) -> None:
    managed_files = [
        filename
        for filename in sorted(GENERATED_ROOT_FILES - {"artifact_manifest.json"})
        if (output_dir / filename).is_file()
    ]
    paper_directory = output_dir / "conformal_paper"
    if paper_directory.exists():
        managed_files.extend(
            sorted(
                str(path.relative_to(output_dir))
                for path in paper_directory.rglob("*")
                if path.is_file()
            )
        )
    manifest = {
        "schema_version": 1,
        "purpose": "Files generated by the current uncertainty-aware paper pipeline",
        "managed_files": managed_files,
        "legacy_names_removed_on_each_run": sorted(LEGACY_ROOT_FILES),
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


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
    primary_alpha: float,
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
    print(f"Held-out development: accuracy {winner['test_accuracy']:.3f}, ROC-AUC {winner['test_roc_auc']:.3f}")

    best_calibration = calibration[calibration["model"] == best_model][["method", "brier", "ece", "log_loss"]]
    print("\n=== PROBABILITY CALIBRATION ON HELD-OUT DEVELOPMENT SUBJECTS ===")
    print(best_calibration.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    primary = conformal[(conformal["method"] == "mondrian_lac") & np.isclose(conformal["alpha"], primary_alpha)].iloc[0]
    print(f"\n=== PRIMARY CONFORMAL RESULT (CLASS-CONDITIONAL LAC, {100 * (1 - primary_alpha):.0f}% TARGET) ===")
    print(f"Coverage: {primary['coverage']:.3f} | healthy: {primary['healthy_coverage']:.3f} | PD: {primary['pd_coverage']:.3f}")
    print(f"Abstention: {primary['abstention_rate']:.3f} | selective accuracy: {primary['selective_accuracy']:.3f} | selective balanced accuracy: {primary['selective_balanced_accuracy']:.3f}")
    print("\n=== REPEATED-RECORDING AGGREGATION ===")
    print(aggregation[["strategy", "accuracy", "roc_auc", "brier", "coverage", "abstention_rate"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    if not any(np.isclose(config.primary_alpha, alpha) for alpha in config.alpha_levels):
        raise ValueError("primary_alpha must be included in alpha_levels.")
    software_versions = _software_versions()
    if config.enforce_paper_environment:
        _validate_paper_environment(software_versions)
    git_provenance = _git_provenance(config.paper_snapshot_tag)
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
        raw_intervals = bootstrap_classification_intervals(
            test_patients["class"],
            test_patients["probability"],
            config.bootstrap_replicates,
            config.random_state,
        )
        test_metric_rows.append({
            "model": model_name,
            **{f"test_{key}": value for key, value in raw_metrics.items()},
            **{f"test_{key}": value for key, value in raw_intervals.items()},
        })
        for method in ("uncalibrated", "sigmoid", "isotonic"):
            calibrator = ProbabilityCalibrator(method, config.random_state).fit(
                calibration_patients["probability"], calibration_patients["class"]
            )
            probabilities = calibrator.predict(test_patients["probability"])
            metrics = classification_metrics(test_patients["class"], probabilities)
            intervals = bootstrap_classification_intervals(
                test_patients["class"], probabilities,
                config.bootstrap_replicates, config.random_state,
            )
            calibration_rows.append({
                "model": model_name,
                "method": method,
                "probability_source": (
                    "raw_subject_mean_probability"
                    if method == "uncalibrated"
                    else f"{method}_calibrated_subject_probability"
                ),
                **metrics,
                **intervals,
            })
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
    # Conformal scores deliberately use raw subject-level model probabilities.
    # Probability calibration is a parallel analysis: its fitted outputs must
    # never enter the split-conformal score or quantile path.
    raw_conformal_calibration_probs = calibration_patients["probability"].to_numpy()
    raw_conformal_test_probs = test_patients["probability"].to_numpy()
    platt_calibration_probs = primary_calibrator.predict(
        raw_conformal_calibration_probs
    )
    platt_test_probs = primary_calibrator.predict(raw_conformal_test_probs)
    isotonic_calibrator = ProbabilityCalibrator("isotonic", config.random_state).fit(
        raw_conformal_calibration_probs, calibration_patients["class"]
    )
    isotonic_test_probs = isotonic_calibrator.predict(raw_conformal_test_probs)
    conformal_rows: list[dict[str, float | str]] = []
    primary_sets: np.ndarray | None = None
    for method in ("lac", "mondrian_lac", "aps"):
        for alpha in config.alpha_levels:
            prediction_sets = raw_conformal_prediction_sets(
                method,
                alpha,
                raw_calibration_probabilities=raw_conformal_calibration_probs,
                calibration_labels=calibration_patients["class"].to_numpy(),
                raw_test_probabilities=raw_conformal_test_probs,
            )
            labels = test_patients["class"].to_numpy()
            covered = prediction_sets[np.arange(len(labels)), labels]
            healthy_low, healthy_high = wilson_interval(
                int(covered[labels == 0].sum()), int((labels == 0).sum())
            )
            pd_low, pd_high = wilson_interval(
                int(covered[labels == 1].sum()), int((labels == 1).sum())
            )
            conformal_rows.append(
                {"method": method, "alpha": alpha,
                 "probability_source": "uncalibrated_subject_mean_probability",
                 "target_coverage": 1 - alpha,
                 **conformal_metrics(labels, prediction_sets),
                 **bootstrap_conformal_intervals(
                     labels, prediction_sets,
                     config.bootstrap_replicates, config.random_state,
                 ),
                 "healthy_coverage": float(covered[labels == 0].mean()),
                 "healthy_coverage_ci_low": healthy_low,
                 "healthy_coverage_ci_high": healthy_high,
                 "pd_coverage": float(covered[labels == 1].mean()),
                 "pd_coverage_ci_low": pd_low,
                 "pd_coverage_ci_high": pd_high}
            )
            if method == "mondrian_lac" and np.isclose(alpha, config.primary_alpha):
                primary_sets = prediction_sets
    conformal_results = pd.DataFrame(conformal_rows)
    conformal_leading = [
        "method", "alpha", "probability_source", "target_coverage",
        "coverage", "coverage_ci_low", "coverage_ci_high",
        "healthy_coverage", "healthy_coverage_ci_low", "healthy_coverage_ci_high",
        "pd_coverage", "pd_coverage_ci_low", "pd_coverage_ci_high",
    ]
    conformal_results = conformal_results[
        conformal_leading
        + [column for column in conformal_results if column not in conformal_leading]
    ]
    if primary_sets is None:
        primary_sets = raw_conformal_prediction_sets(
            "mondrian_lac",
            config.primary_alpha,
            raw_calibration_probabilities=raw_conformal_calibration_probs,
            calibration_labels=calibration_patients["class"].to_numpy(),
            raw_test_probabilities=raw_conformal_test_probs,
        )

    final_predictions = test_patients[["subject_id", "class", "gender", "sex"]].copy()
    final_predictions["raw_probability"] = raw_conformal_test_probs
    final_predictions["platt_probability"] = platt_test_probs
    final_predictions["isotonic_probability"] = isotonic_test_probs
    final_predictions["set_healthy"] = primary_sets[:, 0]
    final_predictions["set_pd"] = primary_sets[:, 1]
    final_predictions["set_size"] = primary_sets.sum(axis=1)

    selective_rows = []
    confidence = np.maximum(platt_test_probs, 1 - platt_test_probs)
    forced_labels = (platt_test_probs >= 0.5).astype(int)
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
        raw_strategy_conformal_calibration_probs = calibration_frame[
            "probability"
        ].to_numpy()
        raw_strategy_conformal_test_probs = test_frame["probability"].to_numpy()
        calibrator = ProbabilityCalibrator("sigmoid", config.random_state).fit(
            raw_strategy_conformal_calibration_probs, calibration_frame["class"]
        )
        platt_strategy_test_probs = calibrator.predict(
            raw_strategy_conformal_test_probs
        )
        sets = raw_conformal_prediction_sets(
            "mondrian_lac",
            config.primary_alpha,
            raw_calibration_probabilities=raw_strategy_conformal_calibration_probs,
            calibration_labels=calibration_frame["class"].to_numpy(),
            raw_test_probabilities=raw_strategy_conformal_test_probs,
        )
        strategy_labels = test_frame["class"].to_numpy(dtype=int)
        strategy_covered = sets[np.arange(len(strategy_labels)), strategy_labels]
        healthy_low, healthy_high = wilson_interval(
            int(strategy_covered[strategy_labels == 0].sum()),
            int((strategy_labels == 0).sum()),
        )
        pd_low, pd_high = wilson_interval(
            int(strategy_covered[strategy_labels == 1].sum()),
            int((strategy_labels == 1).sum()),
        )
        aggregation_rows.append(
            {
                "strategy": strategy,
                "classification_probability_source": "platt_scaled",
                "conformal_probability_source": "uncalibrated_strategy_subject_probability",
                **classification_metrics(
                    test_frame["class"], platt_strategy_test_probs
                ),
                **bootstrap_classification_intervals(
                    test_frame["class"], platt_strategy_test_probs,
                    config.bootstrap_replicates, config.random_state,
                ),
                **conformal_metrics(test_frame["class"], sets),
                **bootstrap_conformal_intervals(
                    test_frame["class"], sets,
                    config.bootstrap_replicates, config.random_state,
                ),
                "healthy_coverage": float(
                    strategy_covered[strategy_labels == 0].mean()
                ),
                "healthy_coverage_ci_low": healthy_low,
                "healthy_coverage_ci_high": healthy_high,
                "pd_coverage": float(strategy_covered[strategy_labels == 1].mean()),
                "pd_coverage_ci_low": pd_low,
                "pd_coverage_ci_high": pd_high,
            }
        )
    aggregation_results = pd.DataFrame(aggregation_rows)
    aggregation_coverage_columns = [
        "coverage", "coverage_ci_low", "coverage_ci_high",
        "healthy_coverage", "healthy_coverage_ci_low", "healthy_coverage_ci_high",
        "pd_coverage", "pd_coverage_ci_low", "pd_coverage_ci_high",
    ]
    aggregation_prefix = [
        "strategy", "classification_probability_source", "conformal_probability_source",
    ]
    aggregation_results = aggregation_results[
        aggregation_prefix
        + [
            column for column in aggregation_results
            if column not in aggregation_prefix + aggregation_coverage_columns
        ]
        + aggregation_coverage_columns
    ]

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
    _prepare_output_directory(output_dir)
    outputs = {
        "dataset_and_split.csv": dataset_table,
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
        "software_versions": software_versions,
        "paper_environment_lock": {
            "file": "requirements-paper.txt",
            "expected_versions": PAPER_SOFTWARE_VERSIONS,
            "matches_lock": software_versions == PAPER_SOFTWARE_VERSIONS,
            "enforced": config.enforce_paper_environment,
        },
        "git_provenance": git_provenance,
        "primary_conformal_method": "mondrian_lac",
        "primary_alpha": config.primary_alpha,
        "conformal_probability_source": "uncalibrated_subject_mean_probability",
        "probability_calibration_analysis": ["uncalibrated", "sigmoid", "isotonic"],
        "dataset_provenance": {
            "uci_dataset_page": UCI_DATASET_PAGE,
            "csv_mirror_url": CSV_MIRROR_URL,
            "observed_sha256": dataset.sha256,
            "verified_mirror_sha256": EXPECTED_CSV_SHA256,
            "checksum_matches_verified_mirror": dataset.sha256 == EXPECTED_CSV_SHA256,
            "structural_validation": "756 recordings, 252 subjects, required columns, finite acoustic features",
        },
        "recordings": len(dataset.target), "subjects": int(dataset.groups.nunique()),
        "acoustic_features": dataset.features.shape[1], "selected_model": best_model,
        "selection_metric": "patient-level grouped-CV ROC-AUC", "cross_validation_folds": actual_folds,
        "split_counts": split_counts.to_dict(orient="index"), "config": asdict(config),
        "analysis_definition": {
            "primary_conformal_method": "class-conditional (Mondrian) LAC",
            "primary_alpha": config.primary_alpha,
            "primary_target_coverage": 1 - config.primary_alpha,
            "conformal_probability_source": "uncalibrated subject-level aggregate probabilities",
            "calibration_methods": ["uncalibrated", "Platt scaling", "isotonic regression"],
            "classification_probability_source": "reported separately for raw, Platt-scaled, and isotonic probabilities",
            "aggregation_conformal_source": "each strategy's own uncalibrated subject-level probability",
            "bootstrap_unit": "subject",
            "bootstrap_replicates": config.bootstrap_replicates,
            "bootstrap_confidence_level": 0.95,
            "bootstrap_interval_method": "percentile",
            "bootstrap_random_state": config.random_state,
            "evaluation_status": "held-out development evaluation",
        },
        "probability_variables": {
            "raw_conformal_calibration_probs": "raw selected-model mean probability for each calibration subject",
            "raw_conformal_test_probs": "raw selected-model mean probability for each held-out development subject",
            "platt_calibration_probs": "parallel Platt-scaled calibration-subject probability; not used for conformal scores",
            "platt_test_probs": "parallel Platt-scaled development-subject probability used for calibrated classification summaries",
            "isotonic_test_probs": "parallel isotonic development-subject probability used only for calibrated classification summaries",
        },
        "limitations": [
            "Single public dataset",
            "Sex is the only individual-level demographic field in the distributed feature table; age-stratified auditing is not possible",
            "Research use only; no clinical validation",
        ],
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
            "primary_alpha": config.primary_alpha,
        })
    results_dir = Path(config.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(results_dir / filename, index=False)
    (results_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    result_index = {
        "version": "paper_v1",
        "purpose": "Compact, version-controlled source tables for every reported result",
        "generated_from": config.output_dir,
        "files": sorted([*outputs, "experiment_metadata.json"]),
        "row_lookup": {
            "model_selection": "model_comparison.csv: selected_model row",
            "calibration": "calibration_results.csv: selected_model and method rows",
            "primary_conformal": (
                "conformal_results.csv: method=mondrian_lac and "
                f"alpha={config.primary_alpha}"
            ),
            "aggregation": "aggregation_results.csv: one row per strategy",
            "subgroups": "subgroup_results.csv: one row per sex or diagnosis subgroup",
            "leakage": "leakage_sensitivity.csv: protocol and repeat rows",
        },
    }
    (results_dir / "RESULT_INDEX.json").write_text(
        json.dumps(result_index, indent=2), encoding="utf-8"
    )
    _write_artifact_manifest(output_dir)
    _print_summary(
        comparison, calibration_results, conformal_results, aggregation_results,
        best_model, config.primary_alpha,
    )
    print(f"\nArtifacts: {output_dir}")
    return comparison
