"""Models (prediction functions) and competition scoring.

A "model" is one prediction function. They live in the ``MODELS`` dict -- adding
one is a single entry; no registry, no decorators.

    def _xgb(random_state=42):
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=random_state)

    MODELS["xgb"] = _xgb
"""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Submission/probability column order. Everything downstream assumes this order.
OUTCOMES = ("home_win", "draw", "away_win")


def _logistic(random_state: int = 42):
    """Median-impute, standardize, multinomial logistic regression."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=random_state),
    )


def _tabpfn(random_state: int = 42):
    """PriorLabs TabPFN via the client (imported lazily so logistic stays offline)."""
    from tabpfn_client import TabPFNClassifier

    return TabPFNClassifier(
        ignore_pretraining_limits=True,
        random_state=random_state,
    )


# name -> factory(random_state) -> a classifier with fit / predict_proba
MODELS = {
    "logistic": _logistic,
    "tabpfn": _tabpfn,
}


def get_model(name: str, random_state: int = 42):
    """Build a model by name from ``MODELS``."""
    try:
        factory = MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown model {name!r}. Available: {', '.join(MODELS)}"
        ) from None
    return factory(random_state)


def predict_probabilities(classifier, features: np.ndarray) -> np.ndarray:
    """Return finite probabilities ordered as home win, draw, away win."""
    raw = np.asarray(classifier.predict_proba(features), dtype=float)
    classes = _model_classes(classifier)
    missing = sorted(set(OUTCOMES).difference(classes))
    if missing:
        raise ValueError(f"Model did not learn outcome classes: {', '.join(missing)}")
    indices = [classes.index(outcome) for outcome in OUTCOMES]
    probabilities = raw[:, indices]
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or np.any(row_sums <= 0):
        raise ValueError("Model returned invalid prediction probabilities")
    return probabilities / row_sums


def competition_log_loss(actual: np.ndarray, probabilities: np.ndarray) -> float:
    """The competition's mean negative log probability (THE metric to minimize)."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(actual), len(OUTCOMES)):
        raise ValueError(
            f"Expected probability shape {(len(actual), len(OUTCOMES))}, "
            f"got {probabilities.shape}"
        )
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise ValueError("Probabilities must be finite, non-negative, and sum to one")
    class_indices = {outcome: index for index, outcome in enumerate(OUTCOMES)}
    try:
        actual_indices = np.asarray([class_indices[str(value)] for value in actual])
    except KeyError as error:
        raise ValueError(f"Unknown outcome class: {error.args[0]}") from error
    actual_probability = probabilities[np.arange(len(actual_indices)), actual_indices]
    # Match standard multiclass log-loss behavior for a model that emits an
    # exact zero without letting log(0) make an otherwise valid run unusable.
    actual_probability = np.clip(actual_probability, np.finfo(float).eps, 1.0)
    return float(-np.log(actual_probability).mean())


def _model_classes(classifier) -> list[str]:
    classes = getattr(classifier, "classes_", None)
    if classes is None and hasattr(classifier, "named_steps"):
        final_step = list(classifier.named_steps.values())[-1]
        classes = getattr(final_step, "classes_", None)
    if classes is None:
        raise ValueError("Model does not expose fitted classes_")
    return [str(value) for value in classes]
