# Uncertainty-Aware Parkinson's Detection from Repeated Voice Measurements

This repository is a reproducible research pipeline for the proposed paper:

> **Knowing When Not to Diagnose: Subject-Level Conformal Prediction and
> Abstention for Parkinson's Detection from Repeated Voice Measurements**

It uses [UCI's Parkinson's Disease Classification dataset
470](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification):
756 recordings from 252 subjects, three sustained-vowel recordings per person,
and 752 acoustic predictors. The project is for research only, not clinical use.

## What the study tests

- Six classifiers: logistic regression, RBF SVM, KNN, random forest, Extra
  Trees, and histogram gradient boosting.
- Patient-level grouped model selection, followed by fully disjoint 60/20/20
  training/calibration/test subjects.
- Uncalibrated probabilities, Platt scaling, and isotonic regression.
- Class-conditional LAC as the primary method, with ordinary LAC and APS as
  comparators, at 80%, 90%, and 95% target coverage.
- Mean-probability, majority-vote, and mean-feature aggregation of each
  subject's three recordings.
- Risk-coverage behavior, abstention, classwise and sex subgroup reliability,
  and a repeated recording-level leakage sensitivity experiment.

Gender is intentionally excluded from model inputs and retained only for the
subgroup audit. Feature selection and scaling stay inside each fitted pipeline.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python main.py
```

The first run downloads and caches a validated CSV mirror at
`data/pd_speech_features.csv`. UCI distributes the original CSV inside a RAR;
you can instead extract it yourself and run:

```bash
python main.py --data path/to/pd_speech_features.csv
```

Useful options:

```bash
python main.py \
  --models logistic_regression support_vector_machine extra_trees \
  --calibration-size 0.20 \
  --test-size 0.20 \
  --alpha-levels 0.05 0.10 0.20 \
  --leakage-repeats 5 \
  --output-dir artifacts
```

Model selection never uses calibration or test subjects. Test results compare
models after selection for analysis; they do not retroactively change the
winner.

## Project structure

```text
.
├── main.py
├── parkinsons_detection/
│   ├── cli.py
│   ├── data.py
│   ├── evaluation.py
│   ├── experiment.py
│   ├── uncertainty_figures.py
│   ├── models/                 # one module per classifier
│   └── uncertainty/
│       ├── calibration.py
│       ├── conformal.py
│       └── metrics.py
├── docs/PAPER_PLAN.md
└── tests/
```

## Outputs

The console names the training-CV winner and prints locked-test discrimination,
calibration, conformal coverage, abstention, selective accuracy, selective
balanced accuracy, and aggregation results.

`artifacts/` contains machine-readable CSVs, experiment metadata, individual
test-subject prediction sets, the risk-coverage curve, and the fitted acoustic
model plus Platt calibrator. `artifacts/conformal_paper/` contains:

- 11 numbered figures as 300-DPI PNG and vector PDF;
- 7 paper tables as CSV and LaTeX;
- `asset_manifest.json` with draft captions.

The figure suite covers the subject-safe workflow, repeated-recording design,
model comparison, reliability diagrams, conformal coverage and efficiency,
risk-coverage behavior, aggregation, subgroup/classwise reliability, forced
versus abstaining confusion matrices, and leakage sensitivity.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Interpretation guardrails

The dataset is imbalanced (188 PD and 64 healthy subjects), comes from a single
study, and provides sex but not age. Overall accuracy and overall conformal
coverage can therefore hide minority-class behavior. Report balanced metrics,
healthy/PD coverage, and subgroup confidence intervals. Independent external
validation is required before any clinical claim.
