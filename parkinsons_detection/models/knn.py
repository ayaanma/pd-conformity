"""K-nearest-neighbours classifier."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build(random_state: int = 42) -> Pipeline:
    del random_state
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", SelectPercentile(f_classif, percentile=10)),
            ("model", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]
    )
