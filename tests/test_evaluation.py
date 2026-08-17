import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression

from parkinsons_detection.evaluation import compare_models_grouped_cv, make_study_split


def synthetic_subject_data(subjects: int = 40):
    groups = pd.Series([subject for subject in range(subjects) for _ in range(3)])
    subject_labels = [subject % 2 for subject in range(subjects)]
    subject_gender = [(subject // 2) % 2 for subject in range(subjects)]
    y = pd.Series([label for label in subject_labels for _ in range(3)])
    gender = pd.Series([value for value in subject_gender for _ in range(3)])
    demographics = pd.DataFrame({"gender": gender, "sex": gender.map({0: "Female", 1: "Male"})})
    X = pd.DataFrame({"signal": y + pd.Series([(i % 3) * 0.05 for i in range(len(y))]), "noise": [(i % 7) / 7 for i in range(len(y))]})
    return X, y, groups, demographics


class EvaluationTests(unittest.TestCase):
    def test_grouped_cv_returns_patient_level_metrics(self) -> None:
        X, y, groups, _ = synthetic_subject_data()
        summary, folds, actual = compare_models_grouped_cv(
            {"logistic": LogisticRegression()}, X, y, groups, folds=4
        )
        self.assertEqual(actual, 4)
        self.assertEqual(len(folds), 4)
        self.assertIn("brier", folds.columns)
        self.assertIn("cv_roc_auc_mean", summary.columns)

    def test_three_way_split_has_no_subject_overlap(self) -> None:
        _, y, groups, demographics = synthetic_subject_data()
        split = make_study_split(y, groups, demographics, 0.2, 0.2, 7)
        train, calibration, test = map(set, (split.train_subjects, split.calibration_subjects, split.test_subjects))
        self.assertFalse(train & calibration)
        self.assertFalse(train & test)
        self.assertFalse(calibration & test)
        self.assertEqual(train | calibration | test, set(groups.unique()))


if __name__ == "__main__":
    unittest.main()
