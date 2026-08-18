"""Command-line interface."""

import argparse

from .experiment import ExperimentConfig, run_experiment
from .models import MODEL_BUILDERS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run subject-level Parkinson's voice model comparison, calibration, "
            "conformal prediction, and abstention experiments."
        )
    )
    parser.add_argument(
        "--data",
        help="Optional pd_speech_features.csv from UCI dataset 470.",
    )
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--calibration-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--alpha-levels", nargs="+", type=float, default=[0.05, 0.1, 0.2],
        help="Conformal error rates to evaluate (default: 0.05 0.1 0.2).",
    )
    parser.add_argument(
        "--primary-alpha", type=float, default=0.1,
        help="Primary conformal error rate; must be in --alpha-levels (default: 0.1).",
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=2000,
        help="Subject-bootstrap replicates for 95%% intervals (default: 2000).",
    )
    parser.add_argument(
        "--leakage-repeats", type=int, default=5,
        help="Repeated random-vs-subject split comparisons (default: 5).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_BUILDERS),
        help="Subset of models to compare (default: all).",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--no-paper-figures",
        action="store_true",
        help="Skip the publication figure and LaTeX table suite.",
    )
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument(
        "--allow-unlocked-environment",
        action="store_true",
        help=(
            "Allow an exploratory run outside the exact paper environment; "
            "the observed versions are still recorded in metadata."
        ),
    )
    parser.add_argument(
        "--results-dir", default="results/paper_v1",
        help="Versioned compact result tables (default: results/paper_v1).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        calibration_size=args.calibration_size,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        selected_models=args.models,
        alpha_levels=tuple(args.alpha_levels),
        primary_alpha=args.primary_alpha,
        bootstrap_replicates=args.bootstrap_replicates,
        leakage_repeats=args.leakage_repeats,
        results_dir=args.results_dir,
        save_plots=not args.no_plots,
        save_paper_outputs=not (args.no_plots or args.no_paper_figures),
        save_model=not args.no_save_model,
        enforce_paper_environment=not args.allow_unlocked_environment,
    )
    run_experiment(config)
