"""Produce the competition submission: a TabPFN x market blend for upcoming games.

The production strategy is a weighted blend ``w*TabPFN(odds) + (1-w)*market`` on the
odds-feed fixtures that have not yet kicked off. ``w`` defaults to the holdout-validated
DEFAULT_BLEND_WEIGHT; any ``w > 0`` uses TabPFN, so the submission stays eligible.

    python odds.py upcoming --execute    # refresh the bracket + odds
    python predict.py                    # writes submission.csv (the blend)

Re-validate the weight any time with:  python evaluate.py --blend
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from data import ODDS_PATH, load_results, merge_odds, odds_covered, tournament_importance
from features import build_features, feature_columns
from models import MODELS, PROBABILITY_COLUMNS, get_model, ordered_probabilities

MARKET_COLUMNS = ("odds_market_p_home", "odds_market_p_draw", "odds_market_p_away")

# Weight on TabPFN in the blend; the rest is the de-vigged market.
# The market dominates the holdout, so this is kept small (a thin TabPFN blend
# that stays eligible). Best-eligible was w=0.15 on 204 training matches and
# w=0.05 on 381; 0.10 is the robust middle. Re-check with: python evaluate.py --blend
DEFAULT_BLEND_WEIGHT = 0.10


def prepare(
    tournament: str = "FIFA World Cup",
    as_of: str | None = None,
    max_train_rows: int = 10_000,
    odds_path: str = ODDS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (odds-covered training rows, upcoming fixture rows) with features + odds.

    Upcoming = odds-feed games whose kickoff is still in the future, so games that have
    already started/finished are excluded automatically.
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
    weight: float = DEFAULT_BLEND_WEIGHT,
    model: str = "tabpfn",
    features: str = "odds",
    as_of: str | None = None,
    tournament: str = "FIFA World Cup",
    max_train_rows: int = 10_000,
    seed: int = 42,
    output: str = "submission.csv",
) -> pd.DataFrame:
    """Write the blended submission: w*model(features) + (1-w)*market."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0 and 1")
    train, fixtures = prepare(tournament, as_of, max_train_rows)
    columns = feature_columns(features)
    classifier = get_model(model, seed)
    classifier.fit(train.loc[:, columns].to_numpy(), train["outcome"].to_numpy())
    model_probabilities = ordered_probabilities(classifier, fixtures.loc[:, columns].to_numpy())
    market = fixtures.loc[:, list(MARKET_COLUMNS)].to_numpy(dtype=float)
    market = market / market.sum(axis=1, keepdims=True)
    blended = weight * model_probabilities + (1 - weight) * market

    submission = _submission(fixtures, blended)
    submission.to_csv(output, index=False)

    eligibility = (
        f"uses {model} (eligible)" if weight > 0 else "market only -- NOT TabPFN-eligible"
    )
    print(
        f"\n{len(submission)} games -> {output}\n"
        f"blend: {weight:.2f}*{model}({features}) + {1 - weight:.2f}*market  [{eligibility}]\n"
    )
    for row in submission.itertuples():
        print(
            f"  {row.date}  {row.home_team:>16} vs {row.away_team:<22} "
            f"H {row.p_home_win:5.1%}  D {row.p_draw:5.1%}  A {row.p_away_win:5.1%}"
        )
    return submission


def _submission(fixtures: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    output = fixtures.loc[:, ["date", "home_team", "away_team"]].copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    for index, column in enumerate(PROBABILITY_COLUMNS):
        output[column] = probabilities[:, index]
    return output.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=float, default=DEFAULT_BLEND_WEIGHT,
                        help="TabPFN weight in the blend (0=market, 1=model)")
    parser.add_argument("--model", choices=list(MODELS), default="tabpfn")
    parser.add_argument("--features", default="odds")
    parser.add_argument("--as-of", default=None, help="Treat this instant as 'now' (default: now)")
    parser.add_argument("--tournament", default="FIFA World Cup")
    parser.add_argument("--max-train-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    predict(
        args.weight, args.model, args.features, args.as_of,
        args.tournament, args.max_train_rows, args.seed, args.output,
    )


if __name__ == "__main__":
    main()
