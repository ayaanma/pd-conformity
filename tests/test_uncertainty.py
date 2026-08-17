import unittest

import numpy as np

from parkinsons_detection.uncertainty.calibration import ProbabilityCalibrator
from parkinsons_detection.uncertainty.conformal import SplitConformalClassifier
from parkinsons_detection.uncertainty.metrics import conformal_metrics


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


if __name__ == "__main__":
    unittest.main()
