# Uncertainty-Aware Parkinson's Voice Classification from Repeated Measurements

This repository is a reproducible research pipeline for the proposed paper:

> **Knowing When the Model Should Abstain: Subject-Level Conformal Prediction and
> Abstention for Parkinson's Detection from Repeated Voice Measurements**

It uses [UCI's Parkinson's Disease Classification dataset
470](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification):
756 recordings from 252 subjects, three sustained-vowel recordings per person,
and 752 acoustic predictors. The project is for research only, not clinical use.

## What the study tests

- Six classifiers: logistic regression, RBF SVM, KNN, random forest, Extra
  Trees, and histogram gradient boosting.
- Patient-level grouped model selection, followed by fully disjoint 60/20/20
  training/calibration/development-holdout subjects.
- Uncalibrated probabilities, Platt scaling, and isotonic regression.
- Class-conditional LAC as the primary method, with ordinary LAC and a
  deterministic, non-randomized cumulative-mass APS comparator based on
  Romano et al., at 80%, 90%, and 95% target coverage.
- Raw subject probabilities for every conformal score; Platt and isotonic
  probabilities remain separate calibration analyses.
- Mean-probability, majority-vote, and mean-feature aggregation of each
  subject's three recordings.
- Risk-coverage behavior, abstention, classwise and sex subgroup reliability,
  and a repeated recording-level leakage sensitivity experiment.

Gender is intentionally excluded from model inputs and retained only for the
subgroup audit. Feature selection and scaling stay inside each fitted pipeline.

## Run

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-paper.txt
python -m pip install -e . --no-deps
python main.py
```

The paper environment is locked to Python 3.9.6 and exact NumPy, SciPy,
pandas, scikit-learn, matplotlib, and joblib versions. The pipeline refuses a
paper run when those versions differ. `--allow-unlocked-environment` is
available only for exploratory runs and records the mismatch in metadata.

The first run downloads and caches a checksum-verified CSV mirror at
`data/pd_speech_features.csv`. The expected SHA-256 is recorded in code and in
the experiment metadata. UCI distributes the original CSV inside a RAR;
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
  --primary-alpha 0.10 \
  --bootstrap-replicates 2000 \
  --leakage-repeats 5 \
  --output-dir artifacts \
  --results-dir results/paper_v1
```

Model selection never uses calibration or development-holdout subjects. Extra
Trees is selected by the grouped-CV ranking criterion; a small ranking margin
is not treated as evidence that it significantly outperforms another model.

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
├── docs/
│   ├── PAPER_PLAN.md
│   └── LLM_HANDOFF.md           # Complete paper/code context for another LLM
├── results/paper_v1/             # Versioned lightweight source-result tables
└── tests/
```

## Outputs

The console names the training-CV winner and prints held-out-development discrimination,
calibration, conformal coverage, abstention, selective accuracy, selective
balanced accuracy, and aggregation results.

`artifacts/` contains machine-readable CSVs, experiment metadata, individual
development-subject prediction sets, the risk-coverage curve, and the fitted acoustic
model plus Platt calibrator. `artifacts/conformal_paper/` contains:

- 11 numbered figures as 300-DPI PNG and vector PDF;
- 7 paper tables as CSV and LaTeX;
- `asset_manifest.json` with draft captions.

Each run cleans pipeline-owned current and legacy outputs before writing and
creates `artifact_manifest.json`, preventing older classifier reports or plots
from being mixed with the current result bundle.

The figure suite covers the subject-safe workflow, repeated-recording design,
model comparison, reliability diagrams, conformal coverage and efficiency,
risk-coverage behavior, aggregation, subgroup/classwise reliability, forced
versus abstaining confusion matrices, and leakage sensitivity.

`results/paper_v1/` contains the compact CSV and JSON sources intended for
version control. `RESULT_INDEX.json` maps each paper result to its source rows.
Reported accuracy, balanced accuracy, ROC-AUC, Brier score, coverage,
abstention, and selective-performance estimates include 95% intervals from
2,000 fixed-seed subject-level percentile-bootstrap resamples. Classwise and subgroup
coverage retain Wilson intervals.

## Tests

```bash
python -m unittest discover -s tests -v
```

For paper writing or transfer to another assistant, start with
[`docs/LLM_HANDOFF.md`](docs/LLM_HANDOFF.md). It records the exact findings,
figure meanings, source-of-truth files, interpretation rules, and unresolved
publication caveats.

## Interpretation guardrails

The dataset is imbalanced (188 PD and 64 healthy subjects), comes from a single
study, and provides sex in the distributed feature table. Individual-level age
is not included in that table, so age-stratified auditing cannot be performed;
the UCI description reports only aggregate cohort age ranges. Overall accuracy
and overall conformal coverage can hide minority-class behavior. Report
balanced metrics, healthy/PD coverage, and subgroup confidence intervals.
Independent external validation is required before any clinical claim.

For a clean release archive, commit the analysis snapshot, tag it `paper-v1`,
and run:

```bash
git archive --format=zip --output voicepd-conformal-paper-v1.zip paper-v1
```

A Git archive contains tracked files only, so `.venv`, `.DS_Store`,
`__MACOSX`, cached data, and generated `artifacts/` are excluded.
