# Paper plan: knowing when the model should abstain

## Working title

**Knowing When the Model Should Abstain: Subject-Level Conformal Prediction and
Abstention for Parkinson's Detection from Repeated Voice Measurements**

## Contribution

Many Parkinson's voice papers compare classifier accuracy. The stronger and
less saturated question is whether a subject-level voice system can quantify
when its output is unreliable, preserve stated coverage after aggregating
repeated recordings, and expose failures that ordinary accuracy hides.

This is not a claim that conformal prediction makes the system clinically safe.
It is a reproducible evaluation of uncertainty and abstention on a public,
repeated-measure benchmark. The core contributions are:

1. subject-disjoint training, calibration, and held-out development testing;
2. model and calibration comparisons at the subject level;
3. three conformal-set methods and explicit abstention;
4. three ways to evaluate repeated-recording aggregation, as a comparison to
   prior aggregation work rather than a novel aggregation method;
5. classwise and sex subgroup reliability with confidence intervals; and
6. a leakage sensitivity analysis showing how the validation unit changes the
   apparent result.

## Dataset and protocol

- UCI dataset 470: 756 rows, 252 subjects, three sustained `/a/` recordings per
  subject, 752 acoustic inputs after removing ID, diagnosis, and gender.
- Subject diagnoses: 188 Parkinson's and 64 healthy controls.
- Gender is an audit field, not a predictor. Individual-level age is absent
  from the distributed feature table, so age-stratified auditing is not
  possible; UCI reports only aggregate cohort age ranges.
- Fixed seed 42 split: 151 training, 50 calibration, and 51 held-out development
  subjects, stratified by diagnosis and gender.
- Five-fold stratified grouped CV inside training subjects selects the model by
  mean patient-level ROC-AUC.
- Each fold fits feature screening and preprocessing from its own training
  partition only.
- Calibration subjects fit Platt/isotonic mappings for the separate probability
  calibration analysis. Their raw subject probabilities estimate
  split-conformal quantiles. The current development partition is a repeatedly
  inspected development holdout, not a blinded confirmatory cohort.

## Research questions

1. Which classical model discriminates PD best under subject-safe validation?
2. Does post-hoc calibration improve Brier score, log loss, and ECE?
3. Do split-conformal sets reach 80%, 90%, and 95% target coverage, and at what
   set-size/abstention cost?
4. Does mean probability, majority vote, or feature averaging use the three
   recordings most effectively?
5. Does reliability differ for healthy versus PD subjects or by sex?
6. How much do random recording splits change apparent accuracy and ROC-AUC?

## Models and methods

- Logistic regression, RBF SVM, KNN, random forest, Extra Trees, and histogram
  gradient boosting.
- Uncalibrated probabilities, Platt scaling, and isotonic regression.
- Class-conditional/Mondrian Least Ambiguous set-valued Classifier (LAC) is the
  primary exploratory analysis because diagnosis is imbalanced; ordinary LAC
  and deterministic, non-randomized cumulative-mass Adaptive Prediction Sets
  (APS) based on Romano et al. are comparators. Do not claim that this APS
  implementation is the original randomized construction or inherits its
  precise theorem unchanged.
- Every conformal method operates on raw subject-level aggregate probabilities.
  Platt scaling and isotonic regression are evaluated in parallel and are not
  inputs to the conformal score calculation.
- Patient-first feature averaging versus record-first mean probability and
  record-first majority vote.

## Metrics that must be reported

- Discrimination: ROC-AUC, accuracy, balanced accuracy, precision, recall, F1.
- Probability quality: Brier score, log loss, expected calibration error, and
  reliability diagrams.
- Conformal: marginal coverage, healthy coverage, PD coverage, average set size,
  singleton/ambiguous/empty rates, abstention, selective accuracy, and selective
  balanced accuracy.
- Reproducibility: fold standard deviations, fit time, exact subject counts,
  random seed, and all individual development-holdout predictions.
- Uncertainty intervals: fixed-seed 2,000-replicate subject bootstrap for
  accuracy, balanced accuracy, ROC-AUC, Brier score, marginal coverage,
  abstention, selective accuracy, and selective balanced accuracy; Wilson
  intervals for classwise and subgroup coverage.

High singleton accuracy must never be quoted alone. In an imbalanced holdout,
it can coexist with poor healthy-control performance. Pair it with selective
balanced accuracy and classwise coverage.

## Figure and table map

1. Subject-disjoint experimental workflow.
2. Repeated-recording aggregation and abstention design.
3. Grouped-CV model performance with fold dispersion.
4. Reliability diagrams and calibration errors.
5. Empirical versus target conformal coverage.
6. Selective risk versus prediction coverage.
7. Prediction-set size and abstention versus target coverage.
8. Repeated-recording aggregation comparison.
9. Conformal reliability by sex and diagnosis.
10. Forced versus singleton-only confusion matrices.
11. Random-recording versus held-out-subject sensitivity.

Tables cover dataset/split composition, model comparison, calibration,
conformal methods, aggregation, subgroup/classwise reliability, and leakage
sensitivity. Every figure is generated as PNG and PDF; every table as CSV and
LaTeX. Lightweight source tables and metadata are versioned under
`results/paper_v1/`, with a row-level result index.

## Suggested paper structure

1. Abstract: motivation, uncertainty gap, dataset/protocol, coverage and
   abstention headline, and limitations.
2. Introduction: why forced binary voice decisions are risky and why repeated
   recordings define the subject as the evaluation unit.
3. Related work: voice classifier comparisons, subject leakage, probability
   calibration, selective prediction, and conformal prediction in medicine.
4. Methods: dataset, split, feature handling, models, calibration, conformal
   scores/finite-sample quantile, aggregation, metrics, and subgroup analysis.
5. Results: model selection first, then calibration, conformal efficiency,
   aggregation, subgroups/classes, and leakage sensitivity.
6. Discussion: what abstention catches, class imbalance, finite calibration-set
   resolution, why APS can be inefficient for binary tasks, and implications.
7. Limitations: one dataset, no external cohort, only 50 calibration subjects,
   sex-only demographic audit, sustained vowel rather than natural speech, and
   no clinical deployment study.
8. Conclusion: uncertainty-aware reporting is more informative than another
   accuracy-only benchmark, without claiming clinical readiness.

## Closest prior work to cite

- [UCI Parkinson's Disease Classification dataset
  470](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification)
- [Performance of machine learning methods in diagnosing Parkinson's disease
  based on dysphonia measures](https://pmc.ncbi.nlm.nih.gov/articles/PMC6208554/)
- [Addressing voice recording replications for Parkinson's disease
  detection](https://www.sciencedirect.com/science/article/pii/S0957417415007381)
- [Harnessing Voice Analysis and Machine Learning for Early Diagnosis of
  Parkinson's Disease](https://pubmed.ncbi.nlm.nih.gov/38740529/)
- [SignSpeak](https://arxiv.org/abs/2407.12020) as an organizational reference,
  not as methodological precedent for conformal Parkinson's voice analysis.
- [Classification with Valid and Adaptive Coverage](https://arxiv.org/abs/2006.02544)
  as the source that motivated ranked cumulative-mass APS. Its original
  construction is randomized; this repository uses a deterministic comparator.
- [Yang et al. (2025), *Optimizing Parkinson's Disease Prediction: A
  Comparative Analysis of Data Aggregation Methods Using Multiple Voice
  Recordings*](https://doi.org/10.3390/data10010004), which already compares
  pre-mean and post-mean/min/max aggregation on this same 756-recording,
  252-subject UCI cohort. Aggregation itself is therefore not novel here.
- [Briscoe et al. (2026), *Interpretable Voice-Based Parkinson's Detection:
  Sex-Specific Acoustic Biomarkers with Subject-Level Nested
  Cross-Validation*](https://doi.org/10.1016/j.bspc.2026.111014), which already
  applies subject-level nested CV and sex-specific analyses to this cohort.
  Neither subject-safe evaluation nor a sex audit is individually novel here.

In a search refreshed on 2026-08-17, no prior work was found that combined this
specific repeated-voice cohort with class-conditional conformal prediction and
explicit abstention. Use only that narrow, dated, qualified gap; do not claim
that aggregation, subject-safe validation, or sex auditing is new, and do not
claim an absolute first.
