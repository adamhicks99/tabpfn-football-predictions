"""Load match results and join the historical betting-odds snapshots."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_PATH = "results.csv"
ODDS_PATH = "data/odds/features.csv"

REQUIRED_RESULT_COLUMNS = (
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
)

# Odds columns (in data/odds/features.csv) exposed to the model as ``odds_*``.
ODDS_MERGE_COLUMNS = (
    "market_p_home",
    "market_p_draw",
    "market_p_away",
    "market_overround",
    "book_count",
)
ODDS_KEYS = ("date", "home_team", "away_team")


def load_results(path: str | Path = RESULTS_PATH) -> pd.DataFrame:
    """Load and clean the international results dataset (played + unplayed rows)."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Results source does not exist: {source}")
    raw = pd.read_csv(source)
    missing = sorted(set(REQUIRED_RESULT_COLUMNS).difference(raw.columns))
    if missing:
        raise ValueError(f"Results source is missing columns: {', '.join(missing)}")

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for column in ("home_team", "away_team", "tournament"):
        frame[column] = frame[column].astype(str).str.strip()
    frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
    frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
    frame["neutral"] = frame["neutral"].map(neutral_value)
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


def load_odds(path: str | Path = ODDS_PATH) -> pd.DataFrame:
    """Load the odds snapshots, one row per match (latest snapshot wins)."""
    source = Path(path)
    if not source.is_file():
        return pd.DataFrame(columns=[*ODDS_KEYS, *ODDS_MERGE_COLUMNS])
    odds = pd.read_csv(source)
    odds["date"] = pd.to_datetime(odds["date"], errors="raise").dt.normalize()
    for column in ("home_team", "away_team"):
        odds[column] = odds[column].astype(str).str.strip()
    keep = [*ODDS_KEYS, *(c for c in ODDS_MERGE_COLUMNS if c in odds.columns)]
    return odds.loc[:, keep].drop_duplicates(list(ODDS_KEYS), keep="last")


def merge_odds(matches: pd.DataFrame, path: str | Path = ODDS_PATH) -> pd.DataFrame:
    """Left-join odds onto matches by (date, home_team, away_team) as ``odds_*``."""
    odds = load_odds(path).rename(
        columns={column: f"odds_{column}" for column in ODDS_MERGE_COLUMNS}
    )
    merged = matches.copy()
    merged["_date"] = merged["date"].dt.normalize()
    odds = odds.rename(columns={"date": "_date"})
    merged = merged.merge(odds, on=["_date", "home_team", "away_team"], how="left")
    return merged.drop(columns="_date")


def odds_covered(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that have a de-vigged market probability."""
    return frame.loc[frame["odds_market_p_home"].notna()].copy()


def load_modeling_data(
    results_path: str | Path = RESULTS_PATH,
    odds_path: str | Path = ODDS_PATH,
) -> pd.DataFrame:
    """Results + base features + odds, ready for evaluate/predict."""
    from features import build_features

    matches = load_results(results_path)
    featured = build_features(matches)
    return merge_odds(featured, odds_path)


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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
