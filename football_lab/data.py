from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_RESULT_COLUMNS = (
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
)


def load_results(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Results source does not exist: {source}")
    raw = pd.read_csv(source)
    missing = sorted(set(REQUIRED_RESULT_COLUMNS).difference(raw.columns))
    if missing:
        raise ValueError(f"Results source is missing columns: {', '.join(missing)}")

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any():
        raise ValueError("Results source contains a missing date")
    for column in ("home_team", "away_team", "tournament"):
        if frame[column].isna().any():
            raise ValueError(f"Results source contains a missing {column}")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"Results source contains a blank {column}")
    if frame["home_team"].eq(frame["away_team"]).any():
        raise ValueError("Results source contains a same-team match")

    for column in ("home_score", "away_score"):
        source_scores = frame[column]
        converted = pd.to_numeric(source_scores, errors="coerce")
        invalid = source_scores.notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Results source contains an invalid {column}")
        frame[column] = converted
    partial_score = frame["home_score"].isna() ^ frame["away_score"].isna()
    if partial_score.any():
        row = frame.loc[partial_score].iloc[0]
        raise ValueError(
            f"Match has only one score: {row.home_team} vs {row.away_team} on "
            f"{row.date.date()}"
        )
    played_scores = frame.loc[
        frame["home_score"].notna(), ["home_score", "away_score"]
    ]
    if (played_scores.lt(0) | played_scores.mod(1).ne(0)).any().any():
        raise ValueError("Played match scores must be non-negative integers")

    frame["neutral"] = frame["neutral"].map(neutral_value)
    frame = frame.drop_duplicates(
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "neutral",
        ],
        keep="first",
    )
    frame["outcome"] = np.select(
        [
            frame["home_score"] > frame["away_score"],
            frame["home_score"] < frame["away_score"],
        ],
        ["home_win", "away_win"],
        default="draw",
    )
    frame.loc[frame["home_score"].isna(), "outcome"] = np.nan
    frame["importance"] = frame["tournament"].map(tournament_importance)
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def tournament_importance(tournament: str) -> float:
    name = str(tournament).lower()
    if "world cup" in name and "qual" not in name:
        return 60.0
    if "confederations" in name:
        return 50.0
    if any(
        value in name
        for value in (
            "uefa euro",
            "copa am",
            "african cup",
            "asian cup",
            "gold cup",
            "nations league",
            "oceania nations",
        )
    ):
        return 45.0
    if "qualif" in name:
        return 35.0
    if "friendly" in name:
        return 20.0
    return 30.0


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def neutral_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1
    if text in {"false", "0", "no"}:
        return 0
    raise ValueError(f"Invalid neutral value: {value!r}")
