"""Extremely randomized trees classifier."""

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline


def build(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("select", SelectPercentile(f_classif, percentile=10)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=250,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
