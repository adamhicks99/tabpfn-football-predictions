from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from football_lab.catalog import Catalog, DatasetKind
from football_lab.models import (
    OUTCOMES,
    PROBABILITY_COLUMNS,
    create_model,
    log_loss,
    model_parameters,
    ordered_probabilities,
)
from football_lab.provenance import git_provenance


def run_evaluation(
    catalog: Catalog,
    *,
    model_name: str,
    training_reference: str,
    evaluation_reference: str,
    seed: int = 42,
    tags: dict[str, str] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    training_record = catalog.get_dataset(
        training_reference,
        expected_kind=DatasetKind.TRAINING,
    )
    evaluation_record = catalog.get_dataset(
        evaluation_reference,
        expected_kind=DatasetKind.EVALUATION,
    )
    code_commit, code_dirty = git_provenance()
    experiment_id = catalog.start_experiment(
        run_type="evaluation",
        model_name=model_name,
        model_params=model_parameters(model_name, seed),
        training_dataset_id=training_record.id,
        evaluation_dataset_id=evaluation_record.id,
        seed=seed,
        tags=tags or {},
        note=note,
        code_commit=code_commit,
        code_dirty=code_dirty,
    )
    try:
        training = catalog.load_frame(training_record)
        evaluation = catalog.load_frame(evaluation_record)
        feature_columns = _validate_compatible_datasets(
            training_record.metadata,
            evaluation_record.metadata,
            training,
            evaluation,
        )
        if training["date"].max() >= evaluation["date"].min():
            raise ValueError("Training data must end before evaluation data begins")

        model = create_model(model_name, seed)
        model.fit(
            training.loc[:, feature_columns].to_numpy(),
            training["outcome"].to_numpy(),
        )
        probabilities = ordered_probabilities(
            model,
            evaluation.loc[:, feature_columns].to_numpy(),
        )
        predicted = np.asarray(OUTCOMES)[probabilities.argmax(axis=1)]
        predictions = evaluation.loc[
            :, ["date", "home_team", "away_team", "outcome"]
        ].copy()
        predictions["predicted"] = predicted
        for index, column in enumerate(PROBABILITY_COLUMNS):
            predictions[column] = probabilities[:, index]

        metrics = {
            ("evaluation", "log_loss"): log_loss(
                evaluation["outcome"].to_numpy(),
                probabilities,
            ),
            ("evaluation", "accuracy"): float(
                accuracy_score(evaluation["outcome"], predicted)
            ),
            ("evaluation", "multiclass_brier"): _brier_score(
                evaluation["outcome"].to_numpy(),
                probabilities,
            ),
            ("evaluation", "matches"): float(len(evaluation)),
        }
        artifact_path = catalog.experiment_path(experiment_id) / "predictions.csv"
        _atomic_csv(predictions, artifact_path)
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        catalog.complete_experiment(
            experiment_id,
            metrics=metrics,
            artifacts=[
                (
                    "evaluation_predictions",
                    catalog.relative(artifact_path),
                    None,
                    digest,
                )
            ],
        )
        return {
            "experiment_id": experiment_id,
            "model": model_name,
            "training_dataset": training_record.id,
            "evaluation_dataset": evaluation_record.id,
            "metrics": {name: value for (_, name), value in metrics.items()},
            "predictions": str(artifact_path),
        }
    except Exception as error:
        catalog.fail_experiment(experiment_id, f"{type(error).__name__}: {error}")
        raise


def parse_tags(values: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for value in values:
        key, separator, tag_value = value.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"Tag {value!r} must use KEY=VALUE")
        tags[key.strip()] = tag_value.strip()
    return tags


def _validate_compatible_datasets(
    training_metadata: dict[str, Any],
    evaluation_metadata: dict[str, Any],
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> list[str]:
    training_features = list(training_metadata.get("feature_columns", []))
    evaluation_features = list(evaluation_metadata.get("feature_columns", []))
    if not training_features or training_features != evaluation_features:
        raise ValueError("Training and evaluation datasets use different feature schemas")
    required = {*training_features, "outcome", "date"}
    for label, frame in (("training", training), ("evaluation", evaluation)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label.title()} dataset is missing: {', '.join(missing)}")
    return training_features


def _brier_score(actual: np.ndarray, probabilities: np.ndarray) -> float:
    indices = {outcome: index for index, outcome in enumerate(OUTCOMES)}
    actual_indices = np.asarray([indices[str(value)] for value in actual])
    one_hot = np.eye(len(OUTCOMES))[actual_indices]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
