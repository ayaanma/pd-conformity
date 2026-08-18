import json
import tempfile
import unittest
from pathlib import Path

from parkinsons_detection.experiment import (
    LEGACY_ROOT_FILES,
    PAPER_SOFTWARE_VERSIONS,
    _prepare_output_directory,
    _software_versions,
    _validate_paper_environment,
    _write_artifact_manifest,
)


class ReproducibilityTests(unittest.TestCase):
    def test_active_environment_matches_paper_lock(self) -> None:
        observed = _software_versions()
        self.assertEqual(observed, PAPER_SOFTWARE_VERSIONS)
        _validate_paper_environment(observed)

    def test_mismatched_environment_is_rejected(self) -> None:
        observed = dict(PAPER_SOFTWARE_VERSIONS)
        observed["scikit-learn"] = "1.8.0"
        with self.assertRaisesRegex(RuntimeError, "scikit-learn=1.8.0"):
            _validate_paper_environment(observed)

    def test_output_cleanup_removes_generated_and_legacy_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "model_comparison.csv").write_text("old", encoding="utf-8")
            legacy = output / next(iter(LEGACY_ROOT_FILES))
            legacy.write_text("old", encoding="utf-8")
            (output / "paper").mkdir()
            (output / "paper" / "old.png").write_text("old", encoding="utf-8")
            unrelated = output / "notes.txt"
            unrelated.write_text("keep", encoding="utf-8")

            _prepare_output_directory(output)

            self.assertFalse((output / "model_comparison.csv").exists())
            self.assertFalse(legacy.exists())
            self.assertFalse((output / "paper").exists())
            self.assertTrue(unrelated.exists())

            _write_artifact_manifest(output)
            manifest = json.loads(
                (output / "artifact_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["managed_files"], [])


if __name__ == "__main__":
    unittest.main()
