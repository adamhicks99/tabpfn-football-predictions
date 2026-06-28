"""Models (prediction functions) and the competition metric.

A model is one entry in ``MODELS``: a factory ``(seed) -> classifier`` exposing
fit / predict_proba over the home_win / draw / away_win classes. Add one = one entry.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTCOMES = ("home_win", "draw", "away_win")
PROBABILITY_COLUMNS = ("p_home_win", "p_draw", "p_away_win")


def _logistic(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=seed),
    )


def _tabpfn(seed: int) -> Any:
    from tabpfn_client import TabPFNClassifier

    return TabPFNClassifier(ignore_pretraining_limits=True, random_state=seed)


MODELS = {
    "logistic": _logistic,
    "tabpfn": _tabpfn,
}


def get_model(name: str, seed: int = 42) -> Any:
    try:
        return MODELS[name](seed)
    except KeyError:
        raise ValueError(f"Unknown model {name!r}. Available: {', '.join(MODELS)}") from None


def ordered_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    """Return probabilities ordered home/draw/away, normalized and finite."""
    raw = np.asarray(model.predict_proba(features), dtype=float)
    classes = _classes(model)
    missing = sorted(set(OUTCOMES).difference(classes))
    if missing:
        raise ValueError(f"Model did not learn outcome classes: {', '.join(missing)}")
    probabilities = raw[:, [classes.index(outcome) for outcome in OUTCOMES]]
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Model returned invalid probabilities")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Model returned a zero-sum probability row")
    return probabilities / row_sums


def log_loss(actual: np.ndarray, probabilities: np.ndarray) -> float:
    """Competition mean negative log-probability of the actual outcome."""
    if probabilities.shape != (len(actual), len(OUTCOMES)):
        raise ValueError("Probability matrix has the wrong shape")
    indices = {outcome: index for index, outcome in enumerate(OUTCOMES)}
    try:
        actual_indices = np.asarray([indices[str(value)] for value in actual])
    except KeyError as error:
        raise ValueError(f"Unknown outcome: {error.args[0]}") from error
    selected = probabilities[np.arange(len(actual)), actual_indices]
    return float(-np.log(np.clip(selected, np.finfo(float).eps, 1.0)).mean())


def _classes(model: Any) -> list[str]:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(list(model.named_steps.values())[-1], "classes_", None)
    if classes is None:
        raise ValueError("Fitted model does not expose classes_")
    return [str(value) for value in classes]
