"""Publication-ready figures and tables for the uncertainty study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


COLORS = {"blue": "#176B87", "orange": "#E67E22", "green": "#2E8B57", "red": "#B23A48"}


def _save(fig: plt.Figure, directory: Path, stem: str) -> list[str]:
    fig.tight_layout()
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png.name, pdf.name]


def _display(name: str) -> str:
    return {
        "support_vector_machine": "RBF SVM",
        "logistic_regression": "Logistic regression",
        "random_forest": "Random forest",
        "extra_trees": "Extra Trees",
        "gradient_boosting": "Gradient boosting",
        "knn": "KNN",
        "mean_probability": "Mean probability",
        "majority_vote": "Majority vote",
        "feature_mean": "Mean acoustic features",
        "uncalibrated": "Uncalibrated",
        "sigmoid": "Platt scaling",
        "isotonic": "Isotonic",
        "mondrian_lac": "Class-conditional LAC",
        "lac": "LAC",
        "aps": "APS",
    }.get(name, name.replace("_", " ").title())


def _workflow_figure(split_counts: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    boxes = [
        (0.02, "756 recordings\n252 subjects\n752 acoustic features", COLORS["blue"]),
        (0.28, f"Training subjects\n{int(split_counts.loc['train', 'subjects'])}\nGrouped model selection", COLORS["orange"]),
        (0.54, f"Calibration subjects\n{int(split_counts.loc['calibration', 'subjects'])}\nCalibrators + conformal", COLORS["green"]),
        (0.80, f"Test subjects\n{int(split_counts.loc['test', 'subjects'])}\nLocked final evaluation", COLORS["red"]),
    ]
    for x, text, color in boxes:
        ax.text(x + 0.09, 0.52, text, ha="center", va="center", color="white",
                fontsize=11, weight="bold", transform=ax.transAxes,
                bbox={"boxstyle": "round,pad=0.8", "facecolor": color, "edgecolor": "none"})
    for x in (0.235, 0.495, 0.755):
        ax.annotate("", xy=(x + 0.035, 0.52), xytext=(x, 0.52), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "lw": 2, "color": "#444444"})
    ax.text(0.5, 0.1, "Subject IDs are disjoint at every stage; repeated recordings never cross a boundary.",
            transform=ax.transAxes, ha="center", fontsize=10)
    return fig


def _aggregation_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.axis("off")
    for y, label in zip((0.72, 0.5, 0.28), ("Recording 1", "Recording 2", "Recording 3")):
        ax.text(0.08, y, label, ha="center", va="center", transform=ax.transAxes,
                bbox={"boxstyle": "round", "facecolor": "#DCEFF4", "edgecolor": COLORS["blue"]})
        ax.annotate("", xy=(0.28, y), xytext=(0.17, y), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.text(0.36, 0.5, "Acoustic model\nper recording", ha="center", va="center", transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.7", "facecolor": COLORS["blue"], "edgecolor": "none"}, color="white")
    ax.annotate("", xy=(0.57, 0.5), xytext=(0.46, 0.5), xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "lw": 2})
    ax.text(0.67, 0.5, "Subject probability\nmean / vote / feature mean", ha="center", va="center",
            transform=ax.transAxes, bbox={"boxstyle": "round,pad=0.7", "facecolor": COLORS["orange"], "edgecolor": "none"}, color="white")
    ax.annotate("", xy=(0.86, 0.5), xytext=(0.78, 0.5), xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "lw": 2})
    ax.text(0.93, 0.5, "Calibrated\nprediction set", ha="center", va="center", transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.7", "facecolor": COLORS["green"], "edgecolor": "none"}, color="white")
    ax.text(0.5, 0.08, "A singleton set yields a diagnosis; {healthy, PD} triggers abstention.",
            transform=ax.transAxes, ha="center", fontsize=10)
    return fig


def generate_uncertainty_paper_outputs(directory: Path, study: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    assets: list[dict[str, object]] = []

    files = _save(_workflow_figure(study["split_counts"]), directory, "figure_1_subject_safe_workflow")
    assets.append({"figure": 1, "files": files, "caption": "Subject-disjoint training, calibration, and test workflow."})
    files = _save(_aggregation_figure(), directory, "figure_2_repeated_recording_aggregation")
    assets.append({"figure": 2, "files": files, "caption": "Three repeated recordings are aggregated before uncertainty reporting."})

    comparison = study["model_comparison"].sort_values("cv_roc_auc_mean")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(comparison))
    for offset, metric, label, color in [(-0.2, "roc_auc", "ROC-AUC", COLORS["blue"]),
                                          (0, "balanced_accuracy", "Balanced accuracy", COLORS["orange"]),
                                          (0.2, "f1", "F1", COLORS["green"])]:
        ax.errorbar(comparison[f"cv_{metric}_mean"], y + offset,
                    xerr=comparison[f"cv_{metric}_std"], fmt="o", capsize=3, label=label, color=color)
    ax.set_yticks(y, [_display(value) for value in comparison["model"]])
    ax.set_xlim(0.35, 1.02); ax.set_xlabel("Patient-level grouped CV score"); ax.legend(loc="lower left")
    files = _save(fig, directory, "figure_3_model_comparison")
    assets.append({"figure": 3, "files": files, "caption": "Patient-level grouped cross-validation performance (mean and standard deviation)."})

    calibration = study["calibration_results"]
    predictions = study["calibration_predictions"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    ax1.plot([0, 1], [0, 1], "--", color="#777777", label="Ideal")
    for method, color in zip(("uncalibrated", "sigmoid", "isotonic"), (COLORS["blue"], COLORS["orange"], COLORS["green"])):
        part = predictions[predictions["method"] == method]
        observed, predicted = calibration_curve(part["class"], part["probability"], n_bins=6, strategy="quantile")
        ax1.plot(predicted, observed, marker="o", label=_display(method), color=color)
    ax1.set(xlabel="Mean predicted PD probability", ylabel="Observed PD frequency", title="Reliability diagram")
    ax1.legend()
    best_cal = calibration[calibration["model"] == study["best_model"]]
    positions = np.arange(len(best_cal)); width = 0.36
    ax2.bar(positions - width / 2, best_cal["brier"], width, label="Brier score", color=COLORS["blue"])
    ax2.bar(positions + width / 2, best_cal["ece"], width, label="ECE", color=COLORS["orange"])
    ax2.set_xticks(positions, [_display(x) for x in best_cal["method"]], rotation=15)
    ax2.set(title="Calibration error (lower is better)", ylabel="Error"); ax2.legend()
    files = _save(fig, directory, "figure_4_probability_calibration")
    assets.append({"figure": 4, "files": files, "caption": "Reliability and calibration error for the selected model on test subjects."})

    conformal = study["conformal_results"]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    target = np.linspace(0.75, 1.0, 100); ax.plot(target, target, "--", color="#777777", label="Nominal")
    for method, part in conformal.groupby("method"):
        ax.plot(1 - part["alpha"], part["coverage"], marker="o", label=_display(method))
    ax.set(xlabel="Target coverage (1 - alpha)", ylabel="Empirical test coverage", xlim=(0.77, 0.97), ylim=(0.7, 1.02))
    ax.legend()
    files = _save(fig, directory, "figure_5_conformal_coverage")
    assets.append({"figure": 5, "files": files, "caption": "Empirical versus target subject-level conformal coverage."})

    selective = study["selective_curve"]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(selective["coverage"], 1 - selective["selective_accuracy"], marker="o", color=COLORS["red"])
    ax.set(xlabel="Fraction of subjects receiving a prediction", ylabel="Selective risk (1 - accuracy)", xlim=(0, 1.02))
    ax.set_ylim(bottom=0)
    files = _save(fig, directory, "figure_6_risk_coverage")
    assets.append({"figure": 6, "files": files, "caption": "Risk–coverage trade-off from confidence-based abstention."})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    for method, part in conformal.groupby("method"):
        target_coverage = 1 - part["alpha"]
        ax1.plot(target_coverage, part["average_set_size"], marker="o", label=_display(method))
        ax2.plot(target_coverage, part["abstention_rate"], marker="o", label=_display(method))
    ax1.set(xlabel="Target coverage", ylabel="Average prediction-set size")
    ax2.set(xlabel="Target coverage", ylabel="Abstention rate")
    ax1.legend()
    files = _save(fig, directory, "figure_7_conformal_efficiency")
    assets.append({"figure": 7, "files": files, "caption": "Prediction-set size and abstention across coverage targets."})

    aggregation = study["aggregation_results"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, metric, title in zip(axes, ("accuracy", "roc_auc", "brier"), ("Accuracy (higher)", "ROC-AUC (higher)", "Brier score (lower)")):
        ax.bar([_display(x) for x in aggregation["strategy"]], aggregation[metric], color=[COLORS["blue"], COLORS["orange"], COLORS["green"]])
        ax.set_title(title); ax.tick_params(axis="x", rotation=18, labelsize=9)
    files = _save(fig, directory, "figure_8_aggregation_comparison")
    assets.append({"figure": 8, "files": files, "caption": "Comparison of three repeated-recording aggregation strategies."})

    subgroup = study["subgroup_results"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, (attribute, part) in zip(axes, subgroup.groupby("attribute", sort=False)):
        x = np.arange(len(part)); values = part["coverage"].to_numpy()
        errors = np.vstack([values - part["coverage_ci_low"], part["coverage_ci_high"] - values])
        ax.bar(x, values, color=[COLORS["orange"], COLORS["blue"]], yerr=errors, capsize=5)
        ax.axhline(0.9, ls="--", color="#555555", label="90% target")
        ax.set_xticks(x, part["subgroup"]); ax.set(title=attribute, ylim=(0, 1.05))
    axes[0].legend(loc="upper left")
    axes[0].set_ylabel("Empirical coverage")
    files = _save(fig, directory, "figure_9_subgroup_reliability")
    assets.append({"figure": 9, "files": files, "caption": "Conformal coverage by sex with 95% Wilson intervals."})

    final_predictions = study["final_predictions"]
    forced = (final_predictions["probability"] >= 0.5).astype(int)
    selected = final_predictions["set_size"] == 1
    selective_labels = final_predictions.loc[selected, "set_pd"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    ConfusionMatrixDisplay(confusion_matrix(final_predictions["class"], forced, labels=[0, 1]), display_labels=["Healthy", "PD"]).plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Forced predictions (all subjects)")
    ConfusionMatrixDisplay(confusion_matrix(final_predictions.loc[selected, "class"], selective_labels, labels=[0, 1]), display_labels=["Healthy", "PD"]).plot(ax=axes[1], colorbar=False, cmap="Greens")
    axes[1].set_title(f"Singleton predictions (n={int(selected.sum())})")
    files = _save(fig, directory, "figure_10_forced_vs_abstaining")
    assets.append({"figure": 10, "files": files, "caption": "Forced classification versus class-conditional conformal singleton decisions at alpha=0.10."})

    leakage = study["leakage_results"]
    protocols = ["random_recording", "heldout_subjects"]
    summary = leakage.groupby("protocol")[["accuracy", "roc_auc"]].agg(["mean", "std"]).reindex(protocols)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(protocols)); width = 0.34
    for offset, metric, color in [(-width / 2, "accuracy", COLORS["blue"]), (width / 2, "roc_auc", COLORS["orange"])]:
        ax.bar(x + offset, summary[(metric, "mean")], width, yerr=summary[(metric, "std")], capsize=4, label=metric.replace("_", " ").upper(), color=color)
    ax.set_xticks(x, ["Random recording split\n(leaky)", "Held-out subjects\n(subject-safe)"])
    ax.set(ylabel="Score", ylim=(0, 1.05)); ax.legend()
    files = _save(fig, directory, "figure_11_leakage_sensitivity")
    assets.append({"figure": 11, "files": files, "caption": "Sensitivity of apparent performance to recording-level leakage."})

    tables = {
        "table_1_dataset_and_split": study["dataset_table"],
        "table_2_model_comparison": study["model_comparison"],
        "table_3_calibration": study["calibration_results"],
        "table_4_conformal": study["conformal_results"],
        "table_5_aggregation": study["aggregation_results"],
        "table_6_subgroups": study["subgroup_results"],
        "table_7_leakage_sensitivity": study["leakage_results"],
    }
    for stem, table in tables.items():
        table.to_csv(directory / f"{stem}.csv", index=False)
        (directory / f"{stem}.tex").write_text(
            table.to_latex(index=False, float_format="%.3f"), encoding="utf-8"
        )
    (directory / "asset_manifest.json").write_text(json.dumps(assets, indent=2), encoding="utf-8")
