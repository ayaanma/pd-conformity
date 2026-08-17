"""Model registry for reproducible comparisons."""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator

from . import (
    extra_trees,
    gradient_boosting,
    knn,
    logistic_regression,
    random_forest,
    support_vector_machine,
)

ModelBuilder = Callable[[int], BaseEstimator]

MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "logistic_regression": logistic_regression.build,
    "support_vector_machine": support_vector_machine.build,
    "knn": knn.build,
    "random_forest": random_forest.build,
    "extra_trees": extra_trees.build,
    "gradient_boosting": gradient_boosting.build,
}


def build_models(
    random_state: int = 42, selected: list[str] | None = None
) -> dict[str, BaseEstimator]:
    """Instantiate all models, or a requested subset, from the registry."""

    names = selected or list(MODEL_BUILDERS)
    unknown = set(names).difference(MODEL_BUILDERS)
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(MODEL_BUILDERS)}"
        )
    return {name: MODEL_BUILDERS[name](random_state) for name in names}
