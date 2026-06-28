from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_lab.catalog import Catalog, DatasetKind, DatasetRecord
from football_lab.data import (
    file_sha256,
    load_results,
    neutral_value,
    tournament_importance,
)
from football_lab.features import build_features
from football_lab.models import (
    PROBABILITY_COLUMNS,
    create_model,
    model_parameters,
    ordered_probabilities,
)
from football_lab.provenance import git_provenance


RESULT_COLUMNS = (
    "date",
    "home_team",
    "away_team",
    "p_home_win",
    "p_draw",
    "p_away_win",
)
PLACEHOLDER_MARKERS = ("tbd", "winner", "runner-up", "third place", "group ")


def build_competition_result(
    catalog: Catalog,
    *,
    name: str,
    model_name: str,
    training_reference: str,
    history_source: str | Path,
    fixtures_source: str | Path,
    as_of: str,
    seed: int = 42,
    tags: dict[str, str] | None = None,
    note: str | None = None,
) -> tuple[DatasetRecord, str]:
    training_record = catalog.get_dataset(
        training_reference,
        expected_kind=DatasetKind.TRAINING,
    )
    cutoff = pd.Timestamp(as_of).normalize()
    code_commit, code_dirty = git_provenance()
    experiment_id = catalog.start_experiment(
        run_type="competition",
        model_name=model_name,
        model_params={
            **model_parameters(model_name, seed),
            "as_of": cutoff.date().isoformat(),
        },
        training_dataset_id=training_record.id,
        evaluation_dataset_id=None,
        seed=seed,
        tags=tags or {},
        note=note,
        code_commit=code_commit,
        code_dirty=code_dirty,
    )
    try:
        training = catalog.load_frame(training_record)
        feature_columns = list(training_record.metadata.get("feature_columns", []))
        if not feature_columns:
            raise ValueError("Training dataset does not declare feature columns")

        fixtures = _load_fixtures(fixtures_source)
        if fixtures["date"].min() < cutoff:
            raise ValueError("Competition fixtures cannot begin before the as-of date")
        if training["date"].max() >= cutoff:
            raise ValueError("Training dataset must end before the competition as-of date")

        results = load_results(history_source)
        known_teams = set(results["home_team"]) | set(results["away_team"])
        _validate_fixture_teams(fixtures, known_teams)
        history = results.loc[
            results["outcome"].notna() & results["date"].lt(cutoff)
        ].copy()
        history["_result_fixture"] = False

        fixture_rows = fixtures.copy()
        fixture_rows["home_score"] = np.nan
        fixture_rows["away_score"] = np.nan
        fixture_rows["tournament"] = fixture_rows.get(
            "tournament",
            "FIFA World Cup",
        )
        fixture_rows["neutral"] = fixture_rows.get("neutral", 1)
        fixture_rows["neutral"] = fixture_rows["neutral"].map(neutral_value)
        fixture_rows["outcome"] = np.nan
        fixture_rows["importance"] = fixture_rows["tournament"].map(
            tournament_importance
        )
        fixture_rows["_result_fixture"] = True
        combined = (
            pd.concat([history, fixture_rows], ignore_index=True, sort=False)
            .sort_values("date", kind="stable")
            .reset_index(drop=True)
        )
        featured = build_features(combined)
        prediction_rows = featured.loc[featured["_result_fixture"]].sort_values(
            "date",
            kind="stable",
        )
        missing_features = sorted(
            set(feature_columns).difference(prediction_rows.columns)
        )
        if missing_features:
            raise ValueError(
                f"Fixture feature frame is missing: {', '.join(missing_features)}"
            )

        model = create_model(model_name, seed)
        model.fit(
            training.loc[:, feature_columns].to_numpy(),
            training["outcome"].to_numpy(),
        )
        probabilities = ordered_probabilities(
            model,
            prediction_rows.loc[:, feature_columns].to_numpy(),
        )
        output = prediction_rows.loc[:, ["date", "home_team", "away_team"]].copy()
        output["date"] = output["date"].dt.strftime("%Y-%m-%d")
        for index, column in enumerate(PROBABILITY_COLUMNS):
            output[column] = probabilities[:, index]
        _validate_result(output)

        metadata: dict[str, Any] = {
            "dataset_schema": "prior-competition-result-v1",
            "model": model_name,
            "model_parameters": model_parameters(model_name, seed),
            "training_dataset_id": training_record.id,
            "history_source_name": Path(history_source).name,
            "history_source_sha256": file_sha256(history_source),
            "fixtures_source_name": Path(fixtures_source).name,
            "fixtures_source_sha256": file_sha256(fixtures_source),
            "as_of": cutoff.date().isoformat(),
            "feature_set": training_record.metadata.get("feature_set"),
            "code_commit": code_commit,
            "code_dirty": code_dirty,
            "output_columns": list(RESULT_COLUMNS),
        }
        result_record = catalog.store_dataset(
            kind=DatasetKind.RESULT,
            name=name,
            frame=output.loc[:, RESULT_COLUMNS],
            metadata=metadata,
        )
        catalog.complete_experiment(
            experiment_id,
            result_dataset_id=result_record.id,
            artifacts=[
                (
                    "competition_result",
                    catalog.relative(result_record.path / "data.csv"),
                    result_record.id,
                    result_record.content_sha256,
                )
            ],
        )
        return result_record, experiment_id
    except Exception as error:
        catalog.fail_experiment(experiment_id, f"{type(error).__name__}: {error}")
        raise


def export_result(
    catalog: Catalog,
    reference: str,
    destination: str | Path,
) -> Path:
    record = catalog.get_dataset(reference, expected_kind=DatasetKind.RESULT)
    frame = catalog.load_frame(record)
    _validate_result(frame)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(record.path / "data.csv", temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _load_fixtures(path: str | Path) -> pd.DataFrame:
    fixtures = pd.read_csv(path)
    required = {"date", "home_team", "away_team", "tournament", "status"}
    missing = sorted(required.difference(fixtures.columns))
    if missing:
        raise ValueError(f"Fixture source is missing columns: {', '.join(missing)}")
    if fixtures.empty:
        raise ValueError("Fixture source cannot be empty")
    fixtures = fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="raise").dt.normalize()
    if fixtures["date"].isna().any():
        raise ValueError("Fixture source contains a missing date")
    for column in ("home_team", "away_team", "tournament", "status"):
        if fixtures[column].isna().any():
            raise ValueError(f"Fixture source contains a missing {column}")
        fixtures[column] = fixtures[column].astype(str).str.strip()
        if fixtures[column].eq("").any():
            raise ValueError(f"Fixture source contains a blank {column}")
    if fixtures.duplicated(["date", "home_team", "away_team"]).any():
        raise ValueError("Fixture source contains duplicate matches")
    if fixtures["home_team"].eq(fixtures["away_team"]).any():
        raise ValueError("Fixture source contains a same-team match")
    if not fixtures["status"].str.lower().eq("confirmed").all():
        raise ValueError("Every fixture status must be confirmed")
    return fixtures


def _validate_fixture_teams(fixtures: pd.DataFrame, known_teams: set[str]) -> None:
    names = pd.concat([fixtures["home_team"], fixtures["away_team"]], ignore_index=True)
    placeholders = sorted(
        {
            name
            for name in names
            if any(marker in name.lower() for marker in PLACEHOLDER_MARKERS)
        }
    )
    if placeholders:
        raise ValueError(f"Fixture source contains unresolved teams: {', '.join(placeholders)}")
    unknown = sorted(set(names).difference(known_teams))
    if unknown:
        raise ValueError(
            f"Fixture teams do not match the history source: {', '.join(unknown)}"
        )


def _validate_result(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != RESULT_COLUMNS:
        raise ValueError(
            "Result dataset columns must be exactly: " + ", ".join(RESULT_COLUMNS)
        )
    probabilities = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    ):
        raise ValueError("Result probabilities must be finite and sum to one")
