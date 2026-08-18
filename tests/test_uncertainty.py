import unittest
import inspect

import numpy as np

from parkinsons_detection.uncertainty.calibration import ProbabilityCalibrator
from parkinsons_detection.uncertainty.conformal import (
    SplitConformalClassifier,
    _finite_sample_quantile,
    raw_conformal_prediction_sets,
)
from parkinsons_detection.uncertainty.metrics import (
    bootstrap_classification_intervals,
    bootstrap_conformal_intervals,
    conformal_metrics,
)


class UncertaintyTests(unittest.TestCase):
    def test_sigmoid_calibrator_produces_probabilities(self) -> None:
        probabilities = np.array([0.1, 0.2, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1])
        calibrated = ProbabilityCalibrator("sigmoid").fit(probabilities, labels).predict(probabilities)
        self.assertTrue(np.all((calibrated > 0) & (calibrated < 1)))
        self.assertGreater(calibrated[-1], calibrated[0])

    def test_lac_returns_binary_prediction_sets(self) -> None:
        calibration_probabilities = np.array([0.05, 0.2, 0.75, 0.9, 0.1, 0.8])
        labels = np.array([0, 0, 1, 1, 0, 1])
        conformal = SplitConformalClassifier("lac", alpha=0.2).fit(calibration_probabilities, labels)
        sets = conformal.predict_sets(np.array([0.1, 0.5, 0.9]))
        self.assertEqual(sets.shape, (3, 2))
        metrics = conformal_metrics(np.array([0, 1, 1]), sets)
        self.assertIn("coverage", metrics)
        self.assertIn("abstention_rate", metrics)

    def test_finite_sample_quantile_uses_higher_correction(self) -> None:
        scores = np.arange(0.1, 1.0, 0.1)
        self.assertAlmostEqual(_finite_sample_quantile(scores, alpha=0.5), 0.6)

    def test_mondrian_lac_estimates_one_threshold_per_class(self) -> None:
        probabilities = np.array([0.1, 0.2, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1])
        conformal = SplitConformalClassifier("mondrian_lac", alpha=0.5).fit(
            probabilities, labels
        )
        self.assertAlmostEqual(conformal.quantiles[0], 0.2)
        self.assertAlmostEqual(conformal.quantiles[1], 0.3)

    def test_alpha_levels_produce_weakly_nested_sets_for_all_methods(self) -> None:
        calibration = np.linspace(0.01, 0.99, 100)
        labels = (calibration >= 0.5).astype(int)
        evaluation = np.linspace(0.02, 0.98, 49)
        for method in ("lac", "mondrian_lac", "aps"):
            sizes = []
            for alpha in (0.05, 0.1, 0.2):
                sets = SplitConformalClassifier(method, alpha).fit(
                    calibration, labels
                ).predict_sets(evaluation)
                self.assertEqual(sets.shape, (49, 2))
                sizes.append(int(sets.sum()))
            self.assertGreaterEqual(sizes[0], sizes[1], method)
            self.assertGreaterEqual(sizes[1], sizes[2], method)

    def test_aps_scores_match_ranked_cumulative_mass_reference(self) -> None:
        probabilities = np.array([[0.2, 0.8], [0.4, 0.6]])
        labels = np.array([1, 0])
        scores = SplitConformalClassifier._aps_scores(probabilities, labels)
        np.testing.assert_allclose(scores, np.array([0.8, 1.0]))

    def test_aps_prediction_sets_include_the_boundary_label(self) -> None:
        conformal = SplitConformalClassifier("aps", alpha=0.1)
        conformal.quantiles = {"all": 0.75}
        actual = conformal.predict_sets(np.array([0.8, 0.55, 0.2]))
        expected = np.array(
            [[False, True], [True, True], [True, False]], dtype=bool
        )
        np.testing.assert_array_equal(actual, expected)

    def test_subject_bootstrap_is_deterministic_at_seed_42(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1])
        probabilities = np.array([0.1, 0.25, 0.6, 0.4, 0.75, 0.9])
        sets = np.array(
            [[1, 0], [1, 0], [1, 1], [1, 1], [0, 1], [0, 1]], dtype=bool
        )
        first = bootstrap_classification_intervals(
            labels, probabilities, replicates=100, random_state=42
        )
        second = bootstrap_classification_intervals(
            labels, probabilities, replicates=100, random_state=42
        )
        self.assertEqual(first, second)
        conformal = bootstrap_conformal_intervals(
            labels, sets, replicates=100, random_state=42
        )
        self.assertLessEqual(conformal["coverage_ci_low"], 1.0)
        self.assertGreaterEqual(conformal["abstention_rate_ci_high"], 0.0)

    def test_raw_conformal_api_exposes_only_raw_probability_inputs(self) -> None:
        parameters = inspect.signature(raw_conformal_prediction_sets).parameters
        self.assertIn("raw_calibration_probabilities", parameters)
        self.assertIn("raw_test_probabilities", parameters)
        self.assertNotIn("calibrated_probabilities", parameters)

        raw_calibration = np.array([0.05, 0.2, 0.75, 0.9, 0.1, 0.8])
        labels = np.array([0, 0, 1, 1, 0, 1])
        raw_test = np.array([0.1, 0.5, 0.9])
        expected = SplitConformalClassifier("lac", 0.2).fit(
            raw_calibration, labels
        ).predict_sets(raw_test)
        actual = raw_conformal_prediction_sets(
            "lac",
            0.2,
            raw_calibration_probabilities=raw_calibration,
            calibration_labels=labels,
            raw_test_probabilities=raw_test,
        )
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
