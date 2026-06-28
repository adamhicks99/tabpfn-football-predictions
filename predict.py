"""Predict upcoming fixtures from the odds feed (the authoritative fixture list).

Fixtures are the odds-feed games that have NOT yet kicked off (commence_time > now),
so completed games drop out automatically. Writes the model submission plus a
market-only fallback in the Prior schema:

    python odds.py upcoming --execute            # refresh the bracket + odds
    python predict.py --model tabpfn --features odds
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from data import ODDS_PATH, load_results, merge_odds, odds_covered, tournament_importance
from features import FEATURE_SETS, build_features, feature_columns
from models import MODELS, PROBABILITY_COLUMNS, get_model, ordered_probabilities

MARKET_COLUMNS = ("odds_market_p_home", "odds_market_p_draw", "odds_market_p_away")


def prepare(
    tournament: str = "FIFA World Cup",
    as_of: str | None = None,
    max_train_rows: int = 10_000,
    odds_path: str = ODDS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (odds-covered training rows, upcoming fixture rows) with features+odds.

    Upcoming = odds-feed games whose kickoff is still in the future, so games that
    have already started/finished are excluded.
    """
    results = load_results()
    cutoff = pd.Timestamp(as_of, tz="UTC") if as_of else pd.Timestamp.now(tz="UTC")

    odds = pd.read_csv(odds_path)
    odds["commence_time"] = pd.to_datetime(odds["commence_time"], utc=True, errors="coerce")
    upcoming = (
        odds.loc[odds["commence_time"].notna() & (odds["commence_time"] > cutoff)]
        .drop_duplicates(["date", "home_team", "away_team"], keep="last")
        .sort_values("commence_time", kind="stable")
    )
    if upcoming.empty:
        raise SystemExit(
            f"No upcoming odds fixtures with kickoff after {cutoff}.\n"
            "Refresh them:  python odds.py upcoming --execute"
        )

    fixtures = pd.DataFrame(
        {
            "date": pd.to_datetime(upcoming["date"].to_numpy()),
            "home_team": upcoming["home_team"].to_numpy(),
            "away_team": upcoming["away_team"].to_numpy(),
            "home_score": np.nan,
            "away_score": np.nan,
            "tournament": tournament,
            "neutral": 1,
            "outcome": np.nan,
        }
    )
    fixtures["importance"] = fixtures["tournament"].map(tournament_importance)
    fixtures["_fixture"] = True

    history_teams = set(results["home_team"]) | set(results["away_team"])
    missing = sorted((set(fixtures["home_team"]) | set(fixtures["away_team"])) - history_teams)
    if missing:
        print(f"WARNING: no results history for {', '.join(missing)} -> default Elo "
              "(add to FEED_TO_RESULTS in odds.py).")

    history = results.copy()
    history["_fixture"] = False
    combined = (
        pd.concat([history, fixtures], ignore_index=True, sort=False)
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    featured = merge_odds(build_features(combined))
    fixture_rows = featured.loc[featured["_fixture"]].sort_values("date", kind="stable")
    train = odds_covered(
        featured.loc[(~featured["_fixture"]) & featured["outcome"].notna()]
    ).tail(max_train_rows)
    return train, fixture_rows


def predict(
    model: str = "tabpfn",
    features: str = "odds",
    as_of: str | None = None,
    tournament: str = "FIFA World Cup",
    max_train_rows: int = 10_000,
    seed: int = 42,
    output: str = "submission.csv",
    market_output: str = "submission_market.csv",
) -> pd.DataFrame:
    """Train on odds-covered history; predict the upcoming odds-feed fixtures."""
    train, fixture_rows = prepare(tournament, as_of, max_train_rows)
    columns = feature_columns(features)
    classifier = get_model(model, seed)
    classifier.fit(train.loc[:, columns].to_numpy(), train["outcome"].to_numpy())
    probabilities = ordered_probabilities(classifier, fixture_rows.loc[:, columns].to_numpy())

    submission = _submission(fixture_rows, probabilities)
    submission.to_csv(output, index=False)

    market = fixture_rows.loc[:, list(MARKET_COLUMNS)].to_numpy(dtype=float)
    market = market / market.sum(axis=1, keepdims=True)
    _submission(fixture_rows, market).to_csv(market_output, index=False)

    print(f"\n{len(submission)} predictions (train {len(train)}) -> {output} | {market_output}\n")
    for row in submission.itertuples():
        print(
            f"  {row.date}  {row.home_team:>16} vs {row.away_team:<20}  "
            f"H {row.p_home_win:4.0%} | D {row.p_draw:4.0%} | A {row.p_away_win:4.0%}"
        )
    return submission


def _submission(fixtures: pd.DataFrame, probabilities) -> pd.DataFrame:
    output = fixtures.loc[:, ["date", "home_team", "away_team"]].copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    for index, column in enumerate(PROBABILITY_COLUMNS):
        output[column] = probabilities[:, index]
    return output.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="tabpfn")
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="odds")
    parser.add_argument("--as-of", default=None, help="Treat this instant as 'now' (default: now)")
    parser.add_argument("--tournament", default="FIFA World Cup")
    parser.add_argument("--max-train-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--market-output", default="submission_market.csv")
    args = parser.parse_args()
    predict(
        args.model, args.features, args.as_of, args.tournament,
        args.max_train_rows, args.seed, args.output, args.market_output,
    )


if __name__ == "__main__":
    main()
