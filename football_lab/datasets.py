from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_lab.catalog import Catalog, DatasetKind, DatasetRecord
from football_lab.data import file_sha256, load_results
from football_lab.features import FEATURE_COLUMNS, build_features


IDENTITY_COLUMNS = ("date", "home_team", "away_team", "tournament")
TARGET_COLUMN = "outcome"


def build_training_dataset(
    catalog: Catalog,
    *,
    name: str,
    source: str | Path,
    start: str,
    end: str,
    max_rows: int | None = None,
) -> DatasetRecord:
    return _build_supervised_dataset(
        catalog,
        kind=DatasetKind.TRAINING,
        name=name,
        source=source,
        start=start,
        end=end,
        max_rows=max_rows,
    )


def build_evaluation_dataset(
    catalog: Catalog,
    *,
    name: str,
    source: str | Path,
    start: str,
    end: str,
) -> DatasetRecord:
    return _build_supervised_dataset(
        catalog,
        kind=DatasetKind.EVALUATION,
        name=name,
        source=source,
        start=start,
        end=end,
        max_rows=None,
    )


def _build_supervised_dataset(
    catalog: Catalog,
    *,
    kind: DatasetKind,
    name: str,
    source: str | Path,
    start: str,
    end: str,
    max_rows: int | None,
) -> DatasetRecord:
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    if start_date >= end_date:
        raise ValueError("Dataset start must be earlier than end")

    matches = load_results(source)
    featured = build_features(matches)
    selected = featured.loc[
        featured["outcome"].notna()
        & featured["date"].ge(start_date)
        & featured["date"].lt(end_date)
    ]
    if selected.empty:
        raise ValueError(f"No played matches fall within [{start_date.date()}, {end_date.date()})")
    if max_rows is not None:
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        selected = selected.tail(max_rows)

    columns = [*IDENTITY_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN]
    output = selected.loc[:, columns].copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    metadata = {
        "dataset_schema": "supervised-match-v1",
        "source_name": Path(source).name,
        "source_sha256": file_sha256(source),
        "start_inclusive": start_date.date().isoformat(),
        "end_exclusive": end_date.date().isoformat(),
        "max_rows": max_rows,
        "feature_set": "base-v1",
        "feature_columns": list(FEATURE_COLUMNS),
        "identity_columns": list(IDENTITY_COLUMNS),
        "target_column": TARGET_COLUMN,
        "feature_history": "strictly-prior-matches",
    }
    return catalog.store_dataset(
        kind=kind,
        name=name,
        frame=output.reset_index(drop=True),
        metadata=metadata,
    )
