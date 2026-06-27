"""Feature sets: the training data a model learns from.

A "feature set" is a function ``matches -> (frame, feature_columns)``. They live
in the ``FEATURE_SETS`` dict -- adding one is a single builder + entry; no
providers, pipelines, or registries.

    def _build_mine(matches, odds_csv=None):
        frame, cols = _build_base(matches)
        frame["my_feature"] = ...
        return frame, cols + ["my_feature"]

    FEATURE_SETS["mine"] = _build_mine
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HOME_ADVANTAGE = 65.0
DEFAULT_ODDS_CSV = "data/odds/features.csv"

BASE_FEATURES = (
    "elo_diff",
    "home_elo",
    "away_elo",
    "form5_diff",
    "form10_diff",
    "home_form5",
    "away_form5",
    "home_winrate",
    "away_winrate",
    "home_gf5",
    "away_gf5",
    "home_ga5",
    "away_ga5",
    "gd10_diff",
    "home_streak",
    "away_streak",
    "home_rest",
    "away_rest",
    "home_played",
    "away_played",
    "h2h_n",
    "h2h_home_winrate",
    "h2h_draw_rate",
    "h2h_gd",
    "neutral",
    "importance",
)

# Numeric odds columns produced by `odds.py` and consumed as features here.
ODDS_FEATURES = (
    "market_p_home",
    "market_p_draw",
    "market_p_away",
    "market_home_std",
    "market_draw_std",
    "market_away_std",
    "market_overround",
    "book_count",
)
# Columns the market-comparison baseline reads (de-vigged market probabilities).
MARKET_COLUMNS = ("odds_market_p_home", "odds_market_p_draw", "odds_market_p_away")


def build_features(
    matches: pd.DataFrame,
    feature_set: str = "base",
    odds_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a named feature set: returns (frame with features, feature columns)."""
    try:
        builder = FEATURE_SETS[feature_set]
    except KeyError:
        raise ValueError(
            f"Unknown feature set {feature_set!r}. Available: {', '.join(FEATURE_SETS)}"
        ) from None
    return builder(matches, odds_csv)


# --------------------------------------------------------------------------- #
# Feature-set builders
# --------------------------------------------------------------------------- #


def _build_base(
    matches: pd.DataFrame,
    odds_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Elo, recent form, rest days, and head-to-head (the 26 base features)."""
    frame = matches.join(_base_block(matches))
    return frame, list(BASE_FEATURES)


def _build_base_odds(
    matches: pd.DataFrame,
    odds_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Base features plus leakage-safe market-odds features."""
    frame, cols = _build_base(matches)
    odds = add_odds_features(matches, odds_csv or DEFAULT_ODDS_CSV)
    return frame.join(odds), cols + list(odds.columns)


FEATURE_SETS = {
    "base": _build_base,
    "base+odds": _build_base_odds,
}


# --------------------------------------------------------------------------- #
# Base block: one chronological pass, no leakage
# --------------------------------------------------------------------------- #


def _base_block(matches: pd.DataFrame) -> pd.DataFrame:
    """Build base features using only matches earlier in the input sequence."""
    elo = defaultdict(lambda: 1500.0)
    results = defaultdict(list)
    last_date: dict[str, pd.Timestamp] = {}
    head_to_head = defaultdict(list)

    def team_features(team: str) -> tuple[float, ...]:
        history = results[team]
        if not history:
            return (elo[team], 1.3, 1.3, 0.33, 1.0, 1.0, 0.0, 0.0, 0)
        last5, last10 = history[-5:], history[-10:]
        streak = 0
        for points, *_ in reversed(history):
            if points != 3:
                break
            streak += 1
        return (
            elo[team],
            np.mean([points for points, *_ in last5]),
            np.mean([points for points, *_ in last10]),
            np.mean([won for *_, won in last10]),
            np.mean([goals for _, goals, _, _ in last5]),
            np.mean([goals for _, _, goals, _ in last5]),
            np.mean(
                [gf - ga for _, gf, ga, _ in last10]
            ),
            streak,
            len(history),
        )

    def matchup_features(home: str, away: str) -> tuple[float, ...]:
        history = head_to_head[tuple(sorted((home, away)))]
        if not history:
            return 0, 0.5, 0.25, 0.0
        count = len(history)
        return (
            count,
            sum(winner == home for _, _, winner in history) / count,
            sum(winner == "draw" for _, _, winner in history) / count,
            np.mean(
                [
                    gd if hist_home == home else -gd
                    for hist_home, gd, _ in history
                ]
            ),
        )

    rows: list[dict[str, float]] = []
    for match in matches.itertuples():
        home, away = match.home_team, match.away_team
        adjustment = HOME_ADVANTAGE * (1 - match.neutral)
        (
            home_elo, home_form5, home_form10, home_winrate,
            home_gf5, home_ga5, home_gd10, home_streak, home_played,
        ) = team_features(home)
        (
            away_elo, away_form5, away_form10, away_winrate,
            away_gf5, away_ga5, away_gd10, away_streak, away_played,
        ) = team_features(away)
        h2h_n, h2h_home_winrate, h2h_draw_rate, h2h_gd = matchup_features(home, away)
        rows.append(
            {
                "elo_diff": home_elo + adjustment - away_elo,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "form5_diff": home_form5 - away_form5,
                "form10_diff": home_form10 - away_form10,
                "home_form5": home_form5,
                "away_form5": away_form5,
                "home_winrate": home_winrate,
                "away_winrate": away_winrate,
                "home_gf5": home_gf5,
                "away_gf5": away_gf5,
                "home_ga5": home_ga5,
                "away_ga5": away_ga5,
                "gd10_diff": home_gd10 - away_gd10,
                "home_streak": home_streak,
                "away_streak": away_streak,
                "home_rest": (
                    min((match.date - last_date[home]).days, 90)
                    if home in last_date
                    else 30
                ),
                "away_rest": (
                    min((match.date - last_date[away]).days, 90)
                    if away in last_date
                    else 30
                ),
                "home_played": home_played,
                "away_played": away_played,
                "h2h_n": h2h_n,
                "h2h_home_winrate": h2h_home_winrate,
                "h2h_draw_rate": h2h_draw_rate,
                "h2h_gd": h2h_gd,
            }
        )

        if pd.notna(match.home_score) and pd.notna(match.away_score):
            gd = match.home_score - match.away_score
            expected_home = 1 / (1 + 10 ** ((away_elo - home_elo - adjustment) / 400))
            actual_home = 1.0 if gd > 0 else (0.0 if gd < 0 else 0.5)
            margin = (
                1.0
                if abs(gd) <= 1
                else (1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8)
            )
            delta = match.importance * margin * (actual_home - expected_home)
            elo[home] += delta
            elo[away] -= delta
            results[home].append(
                (3 if gd > 0 else (1 if gd == 0 else 0), match.home_score, match.away_score, gd > 0)
            )
            results[away].append(
                (3 if gd < 0 else (1 if gd == 0 else 0), match.away_score, match.home_score, gd < 0)
            )
            last_date[home] = last_date[away] = match.date
            head_to_head[tuple(sorted((home, away)))].append(
                (home, gd, home if gd > 0 else (away if gd < 0 else "draw"))
            )

    # neutral and importance already live on the match table.
    generated = [c for c in BASE_FEATURES if c not in ("neutral", "importance")]
    return pd.DataFrame(rows, index=matches.index, columns=generated)


# --------------------------------------------------------------------------- #
# Odds block: leakage-safe point-in-time join to archived market snapshots
# --------------------------------------------------------------------------- #


def add_odds_features(
    matches: pd.DataFrame,
    odds_csv: str | Path = DEFAULT_ODDS_CSV,
) -> pd.DataFrame:
    """Join the latest market snapshot observed before each match's submission cutoff.

    Returns ``odds_``-prefixed numeric columns aligned to ``matches.index``.
    Missing markets stay NaN; ``odds_market_available`` flags coverage.
    """
    snapshots = pd.read_csv(odds_csv)
    keys = ["date", "home_team", "away_team"]
    needed = set(keys) | {"observed_at", "submission_cutoff", *ODDS_FEATURES}
    missing = sorted(needed.difference(snapshots.columns))
    if missing:
        raise ValueError(f"{odds_csv} is missing columns: {', '.join(missing)}")

    left = matches.loc[:, keys].copy()
    left["_match_index"] = matches.index
    right = snapshots.loc[:, [*keys, "observed_at", "submission_cutoff", *ODDS_FEATURES]].copy()
    for key in keys:
        if key == "date":
            left[key] = pd.to_datetime(left[key]).dt.date.astype(str)
            right[key] = pd.to_datetime(right[key]).dt.date.astype(str)
        else:
            left[key] = left[key].astype(str)
            right[key] = right[key].astype(str)

    cutoffs = right.loc[:, [*keys, "submission_cutoff"]].drop_duplicates()
    if cutoffs.duplicated(subset=keys, keep=False).any():
        raise ValueError(f"{odds_csv} has multiple submission_cutoff values for one match")
    left = left.merge(cutoffs, on=keys, how="left", validate="many_to_one")

    left["_cutoff"] = pd.to_datetime(left["submission_cutoff"], utc=True)
    right["_observed"] = pd.to_datetime(right["observed_at"], utc=True)
    if right["_observed"].isna().any():
        raise ValueError(f"{odds_csv} contains invalid observed_at values")

    eligible = left.loc[left["_cutoff"].notna()].sort_values("_cutoff", kind="stable")
    right = right.sort_values("_observed", kind="stable")
    if eligible.empty:
        joined = pd.DataFrame(
            index=matches.index,
            columns=[*ODDS_FEATURES, "_observed", "_cutoff"],
        )
    else:
        joined = pd.merge_asof(
            eligible,
            right,
            left_on="_cutoff",
            right_on="_observed",
            by=keys,
            direction="backward",
            allow_exact_matches=True,
        ).set_index("_match_index")
        joined = joined.reindex(matches.index)

    out = pd.DataFrame(index=matches.index)
    for column in ODDS_FEATURES:
        out[f"odds_{column}"] = pd.to_numeric(joined[column], errors="coerce")
    out["odds_snapshot_age_hours"] = (
        pd.to_datetime(joined["_cutoff"], utc=True)
        - pd.to_datetime(joined["_observed"], utc=True)
    ).dt.total_seconds() / 3_600
    out["odds_market_available"] = out["odds_book_count"].notna().astype(float)
    return out
