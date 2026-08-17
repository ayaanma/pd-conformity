import tempfile
import unittest
from pathlib import Path

import pandas as pd

from parkinsons_detection.data import load_voice_dataset


class DataLoadingTests(unittest.TestCase):
    def test_local_uci470_data_excludes_demographic_predictor(self) -> None:
        frame = pd.DataFrame({
            "id": [1, 1, 2, 2], "gender": [0, 0, 1, 1],
            "feature_a": [1.0, 1.1, 2.0, 2.1], "feature_b": [3.0, 3.1, 4.0, 4.1],
            "class": [0, 0, 1, 1],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            frame.to_csv(path, index=False)
            dataset = load_voice_dataset(path)
        self.assertEqual(dataset.features.columns.tolist(), ["feature_a", "feature_b"])
        self.assertNotIn("gender", dataset.features)
        self.assertEqual(dataset.groups.tolist(), [1, 1, 2, 2])
        self.assertEqual(dataset.demographics["sex"].tolist(), ["Female", "Female", "Male", "Male"])

    def test_inconsistent_subject_labels_are_rejected(self) -> None:
        frame = pd.DataFrame({"id": [1, 1], "gender": [0, 0], "feature": [1.0, 2.0], "class": [0, 1]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                load_voice_dataset(path)


if __name__ == "__main__":
    unittest.main()
