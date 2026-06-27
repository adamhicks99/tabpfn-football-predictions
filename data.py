"""Load and clean the international football results dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path("results.csv")
RAW_URL = (
    "https://raw.githubusercontent.com/"
    "martj42/international_results/master/results.csv"
)


def importance(tournament: str) -> float:
    """Map a tournament name to the weight used by the Elo update."""
    name = tournament.lower()
    if "world cup" in name and "qual" not in name:
        return 60.0
    if "confederations" in name:
        return 50.0
    if any(
        key in name
        for key in (
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


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw results and derive the three-class match outcome."""
    df = raw.copy()
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Results data is missing columns: {', '.join(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["outcome"] = np.select(
        [
            df["home_score"] > df["away_score"],
            df["home_score"] < df["away_score"],
        ],
        ["home_win", "away_win"],
        default="draw",
    )
    missing_score = df["home_score"].isna() | df["away_score"].isna()
    df.loc[missing_score, "outcome"] = np.nan
    df["importance"] = df["tournament"].apply(importance)
    return df


def load_data(
    refresh: bool = False,
    path: str | Path = DEFAULT_DATA_PATH,
    raw_url: str = RAW_URL,
) -> pd.DataFrame:
    """Load results, optionally refreshing without erasing newer local scores."""
    data_path = Path(path)
    if refresh:
        downloaded = pd.read_csv(raw_url)
        raw = (
            _merge_refreshed_results(downloaded, pd.read_csv(data_path))
            if data_path.exists()
            else downloaded
        )
        raw.to_csv(data_path, index=False)
    elif not data_path.exists():
        raw = pd.read_csv(raw_url)
        raw.to_csv(data_path, index=False)
    else:
        raw = pd.read_csv(data_path)
    return clean_data(raw)


def _merge_refreshed_results(
    downloaded: pd.DataFrame,
    existing: pd.DataFrame,
) -> pd.DataFrame:
    """Prefer downloaded scores, but retain local scores/rows absent upstream."""
    keys = ["date", "home_team", "away_team", "tournament"]
    scores = ["home_score", "away_score"]
    needed = set(keys + scores)
    if not needed.issubset(downloaded.columns) or not needed.issubset(existing.columns):
        return downloaded

    local_scores = existing.loc[:, keys + scores].drop_duplicates(keys, keep="last")
    merged = downloaded.merge(
        local_scores,
        on=keys,
        how="left",
        suffixes=("", "_local"),
        validate="many_to_one",
    )
    for score in scores:
        local_score = merged.pop(f"{score}_local")
        missing_score = merged[score].isna()
        merged.loc[missing_score, score] = local_score.loc[missing_score].to_numpy()

    downloaded_keys = pd.MultiIndex.from_frame(downloaded.loc[:, keys])
    existing_keys = pd.MultiIndex.from_frame(existing.loc[:, keys])
    local_only = existing.loc[~existing_keys.isin(downloaded_keys)]
    return pd.concat([merged, local_only], ignore_index=True, sort=False)
