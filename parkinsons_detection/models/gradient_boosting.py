"""Histogram gradient-boosting classifier."""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline


def build(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("select", SelectPercentile(f_classif, percentile=10)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.06,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
