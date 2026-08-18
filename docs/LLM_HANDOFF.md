# LLM handoff: `voicepd-conformal`

This file is the canonical context packet for an LLM helping with this
repository or its accompanying paper. Read it before drafting claims, choosing
figures, interpreting metrics, or changing the experiment.

## 1. Repository in one paragraph

`voicepd-conformal` is a research pipeline for uncertainty-aware Parkinson's
disease classification from repeated sustained-vowel voice measurements. It
compares six classical machine-learning models under subject-level separation,
aggregates three recordings into one subject prediction, evaluates probability
calibration, and constructs conformal prediction sets that can abstain instead
of forcing a binary model output. The intended paper is not another accuracy-only
classifier comparison. Its central question is whether uncertainty reporting,
class-conditional coverage, and abstention reveal weaknesses hidden by overall
accuracy in an imbalanced repeated-measure dataset.

Proposed title:

> **Knowing When the Model Should Abstain: Subject-Level Conformal Prediction and
> Abstention for Parkinson's Detection from Repeated Voice Measurements**

This is an educational research benchmark, not a clinical diagnostic system.
The paper has not been published, and this repository does not establish
clinical validity.

## 2. Current status and source of truth

The current artifacts were generated with seed 42 and verified on 2026-08-17.
The full pipeline and 19 unit tests pass. Generated outputs include 11 figures
in PNG/PDF and 7 tables in CSV/LaTeX.

Use files in this order when numbers disagree:

1. `results/paper_v1/*.csv` for version-controlled exact numerical results.
2. `results/paper_v1/RESULT_INDEX.json` to map a paper value to a source row.
3. `results/paper_v1/experiment_metadata.json` for dataset provenance, split,
   probability sources, bootstrap settings, and selected model information.
4. `artifacts/*.csv` and `artifacts/conformal_paper/` for the full regenerated
   output bundle, including figures and LaTeX.
5. `artifacts/test_subject_predictions.csv` for individual held-out-development
   predictions and conformal sets.
6. This handoff and `docs/PAPER_PLAN.md` for interpretation.
7. `README.md` for concise user-facing instructions.

Do not recover exact values by measuring bars or points in an image. Read the
corresponding CSV.

## 3. Dataset

Source: [UCI Parkinson's Disease Classification dataset
470](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification).

- 756 recordings from 252 subjects.
- Exactly three sustained `/a/` recordings per subject.
- 188 subjects with Parkinson's disease and 64 healthy controls.
- 752 acoustic predictors after removing subject ID, diagnosis, and gender.
- UCI's binary `gender` field is mapped in code as `0 = Female`, `1 = Male` and
  exposed as `sex` in paper outputs.
- Gender/sex is excluded from model inputs and used only for subgroup auditing.
- Individual-level age is not included in the distributed feature table, so
  age-stratified auditing cannot be performed. UCI's description reports
  aggregate cohort age ranges; do not turn those ranges into subject-level
  data or claim that this study evaluates age fairness.
- There are no missing acoustic values in the validated dataset used here.
- The cached convenience mirror is checksum verified. SHA-256:
  `9495d1100beaa24005ea951d4b6588186091de4c21cc45009b027f38f699b8ab`.
  UCI 470 remains the dataset citation; the mirror is only a reproducibility
  convenience.

Fixed subject split:

| Partition | Subjects | Recordings | PD | Healthy |
|---|---:|---:|---:|---:|
| Training | 151 | 453 | 112 | 39 |
| Calibration | 50 | 150 | 37 | 13 |
| Held-out development | 51 | 153 | 39 | 12 |

Subject IDs are disjoint across partitions. All three recordings from a person
remain in the same partition.

## 4. Research questions

The code is designed to answer six questions:

1. Which classical model performs best under subject-grouped validation?
2. How do uncalibrated, Platt-scaled, and isotonic probabilities compare?
3. What coverage/abstention trade-off is produced by LAC,
   class-conditional/Mondrian LAC, and deterministic APS prediction sets?
4. How should three recordings be combined: mean probability, majority vote,
   or mean acoustic features?
5. Does coverage differ by diagnosis or sex?
6. How much does a random recording split inflate apparent performance relative
   to holding out entire subjects?

## 5. Experimental flow

1. Load and validate UCI 470.
2. Remove ID, diagnosis, and gender from acoustic predictors.
3. Split subjects into 60% training, 20% calibration, and 20% development-holdout partitions,
   stratified by diagnosis and gender when feasible.
4. Compare six models with five-fold stratified grouped CV inside training
   subjects only.
5. Aggregate validation recording probabilities by subject before calculating
   selection metrics.
6. Select the model with the highest mean patient-level CV ROC-AUC.
7. Fit every model on all training recordings for descriptive holdout comparison;
   the winner remains determined only by training CV.
8. For the selected model, aggregate the three recording probabilities per
   subject. Fit Platt and isotonic mappings on calibration subjects and evaluate
   those mappings only as a parallel probability-calibration analysis.
9. Use the raw, uncalibrated calibration-subject probabilities to construct all
   conformal thresholds. Platt/isotonic outputs never enter conformal scores.
10. Evaluate classification, calibration, coverage, efficiency, abstention,
    subgroup behavior, and leakage sensitivity.

Feature selection is inside every model pipeline: a fold-local univariate
`SelectPercentile(f_classif, percentile=10)` screen is fitted only on that
fold's training data. Models that require scaling also fit scaling inside their
pipelines. There is no hyperparameter search; model configurations are fixed in
`parkinsons_detection/models/`.

Models:

- Logistic regression
- RBF support vector machine
- K-nearest neighbours
- Random forest
- Extra Trees
- Histogram gradient boosting

## 6. What “uncertain” means

The UCI dataset does **not** contain a ground-truth label saying that a subject
“should be uncertain.” It contains only diagnosis labels.

Uncertainty is an operational output of conformal prediction:

- `{Healthy}`: singleton healthy model prediction.
- `{PD}`: singleton PD model prediction.
- `{Healthy, PD}`: ambiguous set; abstain.
- `{}`: empty set; also abstain. No empty sets occurred in the current primary
  development-holdout result.

For LAC, the calibration nonconformity score is:

```text
1 - probability assigned to the true class
```

At error rate `alpha`, the code uses a finite-sample higher quantile of these
scores. Ordinary LAC uses one threshold. Class-conditional/Mondrian LAC learns
one threshold for healthy controls and another for PD subjects. At prediction time, a
candidate class is included when its candidate nonconformity score does not
exceed the relevant threshold.

The true holdout diagnosis is used only afterward to measure whether the set
covered the truth. It does not decide which cases are marked uncertain.

Figure 6 is related but distinct: its risk-coverage curve thresholds maximum
calibrated probability directly. It is a confidence-based selective prediction
analysis, not the conformal-set rule used for the primary result.

The APS comparator is a deterministic, non-randomized cumulative-mass
construction based on Romano et al. For each
calibration subject, its score is the sum of class probabilities in descending
order through the true class. At prediction time, labels are included in ranked
order through the boundary label that first reaches the finite-sample quantile.
The original Romano-Sesia-Candes APS construction explicitly uses
randomization. Do not describe this implementation as an exact implementation
of that randomized algorithm or attach its precise theorem unchanged.

## 7. Exact current findings

### Model selection

Extra Trees is the training-CV winner.

| Model | CV ROC-AUC, mean +/- SD | Holdout ROC-AUC | Holdout accuracy |
|---|---:|---:|---:|
| Extra Trees | 0.877 +/- 0.042 | 0.846 | 0.843 |
| Logistic regression | 0.870 +/- 0.055 | 0.806 | 0.706 |
| KNN | 0.867 +/- 0.067 | 0.812 | 0.784 |
| RBF SVM | 0.866 +/- 0.061 | 0.812 | 0.804 |
| Random forest | 0.844 +/- 0.047 | 0.846 | 0.824 |
| Gradient boosting | 0.841 +/- 0.060 | 0.844 | 0.765 |

The Extra Trees CV ROC-AUC margin over logistic regression is approximately
0.007. This is a narrow margin relative to fold variability; describe Extra
Trees as the selected model, not as conclusively superior.

The holdout metrics in `model_comparison.csv` use uncalibrated subject-mean
probabilities with a 0.5 threshold. Holdout results must not be used to redefine
the selected model.

For Extra Trees, the 2,000-replicate subject-bootstrap 95% intervals are:
accuracy 0.745-0.941, balanced accuracy 0.599-0.897, ROC-AUC 0.707-0.955,
and Brier score 0.082-0.183. These intervals quantify sampling variation in the
51-subject development holdout, not performance in a broad clinical population.

### Probability calibration for Extra Trees

| Method | Brier | ECE | Log loss | Accuracy |
|---|---:|---:|---:|---:|
| Uncalibrated | 0.129 | 0.161 | 0.414 | 0.843 |
| Platt/sigmoid | 0.132 | 0.069 | 0.415 | 0.784 |
| Isotonic | 0.134 | 0.105 | 0.617 | 0.824 |

Platt scaling substantially reduces ECE, but it does not improve Brier score or
log loss for the selected model on this development split. Say “improved measured
calibration alignment/ECE,” not “improved all calibration metrics.” ROC-AUC is
unchanged by monotonic Platt scaling.

The calibrated 0.5 decision threshold changes accuracy. This explains why the
Extra Trees accuracy is 0.843 in `model_comparison.csv` but 0.784 for the
Platt-scaled mean-probability strategy in `aggregation_results.csv`. These are
not contradictory results.

### Primary conformal result

The current primary method is class-conditional/Mondrian LAC with
`alpha = 0.10`, corresponding to 90% target class-conditional coverage.

| Metric | Result |
|---|---:|
| Overall empirical coverage | 0.941 |
| Overall 95% subject-bootstrap interval | 0.863-1.000 |
| Healthy coverage | 0.917 |
| PD coverage | 0.949 |
| Average prediction-set size | 1.569 |
| Singleton rate | 0.431 |
| Abstention/ambiguous rate | 0.569 |
| Abstention 95% subject-bootstrap interval | 0.431-0.706 |
| Empty-set rate | 0.000 |
| Selected subjects | 22 of 51 |
| Selective accuracy | 0.864 |
| Selective balanced accuracy | 0.854 |

The selective-accuracy bootstrap interval is 0.708-1.000 and the selective
balanced-accuracy interval is 0.657-1.000. Classwise coverage uses Wilson
intervals: 0.646-0.985 for healthy controls and 0.831-0.986 for PD.

This result trades substantial abstention for more balanced behavior. It should
be reported as a coverage-efficiency trade-off, not simply as 86.4% accuracy.

Useful comparator findings at the same 90% target:

- Ordinary LAC: 0.961 overall coverage, 0.333 abstention, 0.941 selective
  accuracy, and 0.800 selective balanced accuracy. Healthy coverage is 0.833
  versus PD coverage of 1.000. Class imbalance remains visible even though
  marginal coverage is above target.
- Deterministic APS: 1.000 coverage, 0.961 abstention, and only 2 singleton model
  predictions. Treat this as inefficiency on this small binary task, not as a
  useful clinical result. Selective balanced accuracy is undefined because the
  singleton subset does not contain both true classes.

### Repeated-recording aggregation

Classification, ROC-AUC, and Brier values below use strategy-specific Platt
probabilities. Conformal values use each strategy's raw subject score with
class-conditional LAC at `alpha = 0.10`.

| Strategy | Accuracy | ROC-AUC | Brier | Coverage | Abstention |
|---|---:|---:|---:|---:|---:|
| Mean recording probability | 0.784 | 0.846 | 0.132 | 0.941 | 0.569 |
| Majority vote | 0.824 | 0.740 | 0.139 | 1.000 | 1.000 |
| Mean acoustic features | 0.824 | 0.816 | 0.138 | 0.941 | 0.608 |

Mean probability gives the strongest discrimination and Brier score. Majority
vote has higher thresholded accuracy but destroys useful probability resolution
and causes the conformal method to abstain on every subject. Do not call
majority vote the best overall strategy based only on accuracy.

### Subgroup and classwise reliability

Primary class-conditional LAC results:

- Female: 25 subjects, coverage 0.880, abstention 0.560, selective accuracy
  0.727; 95% Wilson coverage interval approximately 0.700-0.958.
- Male: 26 subjects, coverage 1.000, abstention 0.577, selective accuracy 1.000;
  interval approximately 0.871-1.000.
- Healthy: 12 subjects, coverage 0.917, abstention 0.500, selective accuracy
  0.833; interval approximately 0.646-0.985.
- PD: 39 subjects, coverage 0.949, abstention 0.590, selective accuracy 0.875;
  interval approximately 0.831-0.986.

These samples are too small for strong fairness conclusions. Report them as an
audit with wide uncertainty, not evidence of demographic parity or disparity.
The conformal method is class-conditional by diagnosis, not conditional by sex;
there is no formal sex-specific coverage guarantee.

### Leakage sensitivity

Across five repetitions for the selected Extra Trees pipeline:

- Random recording split: mean accuracy approximately 0.866 and mean ROC-AUC
  approximately 0.930.
- Held-out-subject split: mean accuracy approximately 0.778 and mean ROC-AUC
  approximately 0.788.

This experiment evaluates recording-level metrics under both protocols so that
the comparison uses the same unit. Its purpose is to demonstrate optimism when
recordings from the same subject can occur in both training and test data.
Primary paper metrics remain subject-level.

### Confusion matrices shown in Figure 10

The forced-prediction panel uses Platt-scaled mean probabilities at threshold
0.5, not the uncalibrated model-comparison threshold:

```text
Forced, all 51 subjects
                 Predicted healthy   Predicted PD
True healthy             3                9
True PD                  2               37
```

The singleton panel contains only the 22 class-conditional conformal decisions:

```text
Singletons only
                 Predicted healthy   Predicted PD
True healthy             5                1
True PD                  2               14
```

The remaining 29 subjects are abstentions and do not appear in the singleton
confusion matrix.

## 8. Resolved blocker and remaining methodological cautions

These points must be resolved or disclosed before making formal claims.

### A. Resolved: probability calibration is separate from conformal calibration

All primary and comparator conformal scores now use raw subject-level model
probabilities from a classifier trained only on training subjects. The same rule
is enforced separately for mean probability, majority vote, and feature-mean
aggregation. Platt and isotonic mappings are parallel analyses used only for
probability-quality, thresholded-classification, reliability, and risk-coverage
outputs. The conformal, subgroup, aggregation, and singleton outputs were all
regenerated after this correction.

The primary empirical values remained numerically unchanged because applying a
strictly monotone transform preserves the within-class threshold ordering of
Mondrian LAC in this dataset. That numerical coincidence does not justify the
old path; metadata and explicit raw-variable names now protect the valid path.
The finite-sample claim still depends on exchangeability of subject-level
calibration and future examples and does not imply conditional or clinical
validity.

### B. The current evaluation split is a repeatedly inspected development holdout

The same deterministic development split was inspected repeatedly while figures
and the primary presentation were refined. Class-conditional LAC was designated
as the primary exploratory analysis during that iterative work. Therefore:

- Do not claim the development holdout was viewed exactly once.
- Treat current findings as exploratory/development results.
- For a confirmatory paper, freeze the pipeline and evaluate on an external
  cohort or a newly reserved split that is not used for further design choices.

### C. Single-split and small-sample uncertainty

- Main results use one seed and one 51-subject development split.
- Only 13 healthy subjects are available for calibration and 12 for testing.
- Class-conditional quantiles are therefore coarse, especially for controls.
- ECE with 51 subjects is noisy and bin-dependent.
- CV standard deviations describe selection-fold variability, not confidence
  intervals for a broad clinical population.

The result CSVs now include fixed-seed, 2,000-replicate percentile
subject-bootstrap 95%
intervals for accuracy, balanced accuracy, ROC-AUC, Brier score, coverage,
abstention, selective accuracy, and selective balanced accuracy. Classwise and
subgroup coverage retain Wilson intervals. These intervals improve reporting
but do not solve single-cohort generalizability.

Before submission, consider repeated nested subject-level splits and an
external dataset. Preserve a clear distinction between exploratory
development results and a future confirmatory evaluation.

### D. Scope limitations

- Single public dataset and collection protocol.
- Sustained vowel recordings, not conversational or home speech.
- No individual-level age in the distributed feature table, and therefore no
  age-stratified analysis; there is also no disease stage, medication state,
  recording-device, site, or ethnicity subgroup analysis.
- No comparison with clinician uncertainty or human disagreement.
- No ground-truth “uncertain” labels.
- No external or prospective clinical validation.
- No decision-curve, utility, or cost analysis establishing that abstention is
  clinically useful.
- Feature selection and model settings are fixed; there is no nested
  hyperparameter optimization.

## 9. Claims the paper may and may not make

Supported, with appropriate exploratory wording:

- Subject-level separation materially changes apparent performance relative to
  random recording splits.
- Class-conditional conformal sets expose a coverage/abstention trade-off that
  accuracy alone does not show.
- Ordinary marginal LAC can look strong by selective accuracy while remaining
  poor by selective balanced accuracy in an imbalanced dataset.
- Mean-probability aggregation preserves more discrimination than majority
  voting in this experiment; this is a comparative result, not a novel
  aggregation method.
- Deterministic APS is empirically inefficient in this binary setting and
  configuration.

Not supported:

- Clinical diagnostic readiness, safety, or superiority to clinicians.
- Conditional coverage for each individual, sex subgroup, clinic, or device;
  Mondrian LAC targets coverage by diagnosis class under exchangeability.
- Generalization to new hospitals, devices, languages, or natural speech.
- General demographic fairness.
- “First ever” or absolute novelty without an updated systematic literature
  search immediately before submission.
- Extra Trees is universally the best Parkinson's voice model.
- Abstained cases are objectively ambiguous in a clinical sense.

A defensible novelty sentence based on the search refreshed on 2026-08-17 is:

> “In our search updated through 17 August 2026, we found no prior study that
> combined this public repeated-voice cohort with class-conditional conformal
> prediction and explicit abstention.”

This gap is deliberately narrow. Yang et al. (2025) already compared repeated-
recording aggregation strategies on the same 756-recording, 252-subject cohort
([DOI 10.3390/data10010004](https://doi.org/10.3390/data10010004)). Briscoe et
al. (2026) already used subject-level nested CV and sex-specific analysis on
this cohort ([DOI 10.1016/j.bspc.2026.111014](https://doi.org/10.1016/j.bspc.2026.111014)).
Therefore aggregation, subject-safe evaluation, and sex auditing are not
individually novel contributions. Refresh the search again at submission.

## 10. Figure guide

All assets are under `artifacts/conformal_paper/` as 300-DPI PNG and vector PDF.
Use PDF for production typesetting when the venue accepts it; use PNG for review or
systems that rasterize uploads.

1. **`figure_1_subject_safe_workflow`** — Put in Methods near the data split.
   It establishes the subject-disjoint 151/50/51 design. Essential main-text
   figure.
2. **`figure_2_repeated_recording_aggregation`** — Put in Methods near the
   aggregation and prediction-set definition. It explicitly labels raw-score
   conformal sets and non-abstained model predictions.
3. **`figure_3_model_comparison`** — Put at the start of Results. Shows grouped
   CV mean +/- SD for ROC-AUC, balanced accuracy, and F1. It is a training-only
   model-selection figure, not held-out-development performance.
4. **`figure_4_probability_calibration`** — Use in calibration Results. The
   reliability curves and ECE/Brier bars are based on 51 development subjects;
   avoid overinterpreting individual bins.
5. **`figure_5_conformal_coverage`** — Use in the main uncertainty Results.
   Compare empirical and nominal coverage across alpha values. At alpha 0.10,
   deterministic APS's perfect coverage is paired with 96.1% abstention and is
   not evidence of utility.
6. **`figure_6_risk_coverage`** — Use in Results or supplement. This is a
   probability-confidence threshold curve, not the conformal method itself.
7. **`figure_7_conformal_efficiency`** — Pair with Figure 5 or place in the
   supplement. It explains coverage cost using set size and abstention.
8. **`figure_8_aggregation_comparison`** — Use for the repeated-recording
   analysis. It has six panels separating Platt-scaled classification metrics
   from raw-score conformal coverage/abstention, with 95% subject-bootstrap
   intervals.
9. **`figure_9_subgroup_reliability`** — Use for the audit. It shows sex and
   diagnosis coverage with 95% Wilson intervals. Wide intervals are part of the
   finding.
10. **`figure_10_forced_vs_abstaining`** — Strong discussion figure. It compares
    forced Platt-threshold predictions with the 22 singleton conformal decisions;
    29 abstentions are omitted from the right matrix and must be stated.
11. **`figure_11_leakage_sensitivity`** — Use in Methods validation or early
    Discussion. It shows why the subject, not the recording, must be the split
    unit. Both protocols are scored at recording level for a like-for-like
    sensitivity comparison; the bars summarize five repetitions.

Recommended compact main-text set if the venue restricts figures: 1, 3, 4, 5,
7, 9, 10, and 11. Move 2, 6, and 8 to supplementary material if necessary.

Current draft captions are in
`artifacts/conformal_paper/asset_manifest.json`. Captions should be expanded in
the manuscript to name the evaluation partition, sample size, alpha, error bars,
and whether probabilities are calibrated.

## 11. Table guide

- `table_1_dataset_and_split`: dataset and diagnosis counts by partition.
- `table_2_model_comparison`: grouped-CV and uncalibrated held-out metrics for
  all models. A paper may shorten this to primary metrics.
- `table_3_calibration`: all model/calibration combinations on development subjects.
- `table_4_conformal`: all methods and alpha levels, including classwise
  coverage and selective balanced accuracy.
- `table_5_aggregation`: aggregation classification, calibration, and conformal
  efficiency metrics.
- `table_6_subgroups`: sex and diagnosis coverage with Wilson intervals.
- `table_7_leakage_sensitivity`: each of five repeated protocol comparisons.

CSV files are the numerical source of truth. LaTeX files are generated for
convenience and may need column selection, rounding, and venue-specific styling.

## 12. Code map

- `main.py`: minimal executable entry point.
- `parkinsons_detection/cli.py`: command-line arguments and defaults.
- `parkinsons_detection/data.py`: UCI 470 download/cache, two-header CSV
  handling, SHA-256 verification, demographic separation, and feature construction.
- `parkinsons_detection/evaluation.py`: subject table, three-way split,
  probability/feature aggregation, positive-class scoring, and grouped CV.
- `parkinsons_detection/experiment.py`: full orchestration and artifact writing.
- `parkinsons_detection/models/`: one model builder per classifier.
- `parkinsons_detection/uncertainty/calibration.py`: identity, Platt/sigmoid,
  and isotonic probability mappings.
- `parkinsons_detection/uncertainty/conformal.py`: finite-sample quantiles and
  raw-score routing plus binary LAC, class-conditional LAC, and deterministic
  non-randomized cumulative-mass APS sets based on Romano et al.
- `parkinsons_detection/uncertainty/metrics.py`: classification, ECE, conformal,
  selective, subject-bootstrap, and Wilson-interval metrics.
- `parkinsons_detection/uncertainty_figures.py`: all paper figures and tables.
- `tests/`: data validation, subject isolation, grouped CV, calibration, and
  conformal unit tests.
- `docs/PAPER_PLAN.md`: shorter paper outline and related-work starting points.

## 13. Reproduction

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-paper.txt
python -m pip install -e . --no-deps
python main.py
```

The locked paper runtime is Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1,
pandas 2.3.3, scikit-learn 1.6.1, matplotlib 3.9.4, and joblib 1.5.3.
`experiment_metadata.json` records the observed versions, whether they match
the lock, the generating source commit SHA, and the intended `paper-v1` tag.
The pipeline rejects mismatched paper environments by default.

Use a local extracted UCI CSV when automatic download is unavailable:

```bash
python main.py \
  --data path/to/pd_speech_features.csv \
  --output-dir artifacts \
  --bootstrap-replicates 2000 \
  --results-dir results/paper_v1
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

Important defaults:

- Random seed: 42
- Training/calibration/development proportions: 0.60/0.20/0.20
- Grouped CV folds: 5
- Conformal alpha values: 0.05, 0.10, 0.20
- Subject-bootstrap replicates: 2,000
- Leakage repetitions: 5
- Primary aggregation: mean recording probability
- Current primary conformal analysis: class-conditional LAC at alpha 0.10

Changing a split, model, feature screen, calibration design, aggregation rule,
or primary alpha invalidates the current numeric handoff. Regenerate all
artifacts and update this file together.

## 14. Suggested manuscript narrative

The cleanest story is:

1. Repeated recordings make subject-level separation essential; demonstrate
   the leakage effect.
2. Select a conventional acoustic model using grouped training CV rather than
   development-holdout accuracy.
3. Show that probability quality and thresholded accuracy are different
   properties; calibration can improve ECE while changing 0.5-threshold errors.
4. Introduce prediction sets and abstention because a forced model output hides
   uncertainty.
5. Show why marginal coverage and selective accuracy alone remain insufficient
   under imbalance.
6. Present class-conditional coverage and selective balanced accuracy as the
   more informative evaluation.
7. Discuss the price: more than half of development subjects are abstained upon.
8. End with methodological and external-validation requirements, not a clinical
   deployment claim.

The most interesting finding is not that Extra Trees ranked first by a narrow
CV margin. It is that ordinary LAC achieved high marginal coverage while
healthy coverage remained 0.833 versus 1.000 for PD; class-conditional LAC made
classwise coverage more balanced (0.917 versus 0.949) at a substantial
abstention cost. That is the paper's strongest evidence for looking beyond a
single performance number.

## 15. Checklist for an LLM drafting the paper

Before returning prose, verify that it:

- Uses “subject” or “participant,” not “independent recording,” as the primary
  evaluation unit.
- Distinguishes training CV selection from held-out descriptive results.
- Distinguishes uncalibrated model metrics from Platt-scaled aggregation metrics.
- States that uncertainty has no human-provided ground-truth label.
- Reports abstention whenever reporting singleton/selective accuracy.
- Pairs overall coverage with healthy and PD coverage.
- Pairs selective accuracy with selective balanced accuracy.
- Describes deterministic APS's 100% coverage together with 96.1% abstention
  and only two singleton model predictions.
- Reports sample sizes and uncertainty intervals for subgroup claims.
- States that raw subject probabilities, not fitted Platt/isotonic outputs, feed
  every conformal score.
- Qualifies the finite-sample guarantee by subject-level exchangeability and
  does not imply individual-conditional or clinical validity.
- Does not describe the development holdout as pristine or blinded.
- Does not imply clinical readiness.
- Refreshes the literature search before making a novelty statement.
- Cites the original UCI dataset and method sources, not this handoff, for
  scientific claims.
