import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from parkinsons_detection.evaluation import (
    aggregate_features,
    aggregate_probabilities,
    compare_models_grouped_cv,
    make_study_split,
)


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
        X, y, groups, demographics = synthetic_subject_data()
        split = make_study_split(y, groups, demographics, 0.2, 0.2, 7)
        train, calibration, test = map(set, (split.train_subjects, split.calibration_subjects, split.test_subjects))
        self.assertFalse(train & calibration)
        self.assertFalse(train & test)
        self.assertFalse(calibration & test)
        self.assertEqual(train | calibration | test, set(groups.unique()))

        paths = {
            "train": split.train_indices,
            "calibration": split.calibration_indices,
            "test": split.test_indices,
        }
        index_sets = {name: set(indices) for name, indices in paths.items()}
        self.assertFalse(index_sets["train"] & index_sets["calibration"])
        self.assertFalse(index_sets["train"] & index_sets["test"])
        self.assertFalse(index_sets["calibration"] & index_sets["test"])
        self.assertEqual(set().union(*index_sets.values()), set(range(len(X))))
        for name, indices in paths.items():
            expected_subjects = set(getattr(split, f"{name}_subjects"))
            self.assertEqual(set(groups.iloc[indices]), expected_subjects)

    def test_seed_42_split_is_deterministic(self) -> None:
        _, y, groups, demographics = synthetic_subject_data()
        first = make_study_split(y, groups, demographics, 0.2, 0.2, 42)
        second = make_study_split(y, groups, demographics, 0.2, 0.2, 42)
        for field in (
            "train_indices",
            "calibration_indices",
            "test_indices",
            "train_subjects",
            "calibration_subjects",
            "test_subjects",
        ):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))

    def test_all_aggregation_paths_return_one_row_per_subject(self) -> None:
        X, y, groups, demographics = synthetic_subject_data()
        probabilities = np.linspace(0.01, 0.99, len(y))
        for strategy in ("mean_probability", "majority_vote"):
            aggregate = aggregate_probabilities(
                probabilities, y, groups, demographics, strategy=strategy
            )
            self.assertEqual(len(aggregate), groups.nunique())
            self.assertTrue(aggregate["subject_id"].is_unique)
            self.assertEqual(set(aggregate["subject_id"]), set(groups.unique()))

        mean_features, labels, subject_ids, mean_demographics = aggregate_features(
            X, y, groups, demographics
        )
        self.assertEqual(len(mean_features), groups.nunique())
        self.assertEqual(len(labels), groups.nunique())
        self.assertEqual(len(subject_ids), groups.nunique())
        self.assertEqual(len(mean_demographics), groups.nunique())
        self.assertTrue(subject_ids.is_unique)
        self.assertEqual(set(subject_ids), set(groups.unique()))


if __name__ == "__main__":
    unittest.main()
