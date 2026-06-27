"""Train a (model, feature set) and predict an explicit fixture manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from data import importance, load_data
from features import FEATURE_SETS, build_features
from models import MODELS, OUTCOMES, competition_log_loss, get_model, predict_probabilities

TODAY = pd.Timestamp.now().normalize()
TRAIN_START = pd.Timestamp("2014-01-01")
MAX_TRAIN = 10_000
FIXTURE_COLUMNS = ("date", "home_team", "away_team")
PLACEHOLDER_MARKERS = ("tbd", "winner", "runner-up", "third place", "group ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="tabpfn")
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="base")
    parser.add_argument("--odds-csv", default=None, help="Odds snapshot CSV (for base+odds)")
    parser.add_argument("--refresh", action="store_true", help="Re-download the results dataset")
    parser.add_argument(
        "--fixtures",
        required=True,
        help="CSV containing the exact date, home_team, and away_team rows to predict",
    )
    parser.add_argument(
        "--expected-fixtures",
        type=int,
        default=None,
        help="Fail unless the fixture manifest has exactly this many rows",
    )
    parser.add_argument("--output", default=None, help="Output CSV path")
    return parser.parse_args()


def load_prediction_fixtures(
    path: str | Path,
    *,
    known_teams: set[str],
    earliest_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load and validate an explicit fixture manifest."""
    fixtures = pd.read_csv(path)
    missing = sorted(set(FIXTURE_COLUMNS).difference(fixtures.columns))
    if missing:
        raise ValueError(f"Fixture file is missing columns: {', '.join(missing)}")
    if fixtures.empty:
        raise ValueError("Fixture file contains no matches")

    fixtures = fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="raise").dt.normalize()
    for column in ("home_team", "away_team"):
        fixtures[column] = fixtures[column].astype(str).str.strip()

    duplicates = fixtures.duplicated(list(FIXTURE_COLUMNS), keep=False)
    if duplicates.any():
        raise ValueError("Fixture file contains duplicate matches")
    if fixtures["home_team"].eq(fixtures["away_team"]).any():
        raise ValueError("A fixture cannot contain the same home and away team")
    if (fixtures["date"] < pd.Timestamp(earliest_date).normalize()).any():
        old = fixtures.loc[fixtures["date"] < pd.Timestamp(earliest_date).normalize()].iloc[0]
        raise ValueError(
            f"Fixture file contains a past match: {old.home_team} vs {old.away_team} "
            f"on {old.date.date()}"
        )
    if "status" in fixtures and not fixtures["status"].astype(str).str.lower().eq("confirmed").all():
        raise ValueError("Every fixture status must be 'confirmed'")

    names = pd.concat([fixtures["home_team"], fixtures["away_team"]], ignore_index=True)
    placeholders = sorted(
        {
            name
            for name in names
            if any(marker in name.lower() for marker in PLACEHOLDER_MARKERS)
        }
    )
    if placeholders:
        raise ValueError(f"Fixture file contains unresolved teams: {', '.join(placeholders)}")
    unknown = sorted(set(names).difference(known_teams))
    if unknown:
        raise ValueError(
            "Fixture teams do not match the historical dataset: "
            f"{', '.join(unknown)}"
        )
    return fixtures


def append_prediction_fixtures(
    matches: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """Drop stale unplayed rows and append only the explicit prediction fixtures."""
    history = matches.loc[matches["outcome"].notna()].copy()
    history["_prediction_fixture"] = False

    fixture_rows = fixtures.copy()
    fixture_rows["home_score"] = np.nan
    fixture_rows["away_score"] = np.nan
    fixture_rows["tournament"] = fixture_rows.get("tournament", "FIFA World Cup")
    fixture_rows["neutral"] = 1
    fixture_rows["outcome"] = np.nan
    fixture_rows["importance"] = fixture_rows["tournament"].map(importance)
    fixture_rows["_prediction_fixture"] = True

    return (
        pd.concat([history, fixture_rows], ignore_index=True, sort=False)
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    matches = load_data(refresh=args.refresh)
    known_teams = set(matches["home_team"]) | set(matches["away_team"])
    fixtures = load_prediction_fixtures(
        args.fixtures,
        known_teams=known_teams,
        earliest_date=TODAY,
    )
    if args.expected_fixtures is not None and len(fixtures) != args.expected_fixtures:
        raise ValueError(
            f"Expected {args.expected_fixtures} fixtures, found {len(fixtures)} "
            f"in {args.fixtures}"
        )
    prediction_data = append_prediction_fixtures(matches, fixtures)
    frame, feature_cols = build_features(prediction_data, args.features, args.odds_csv)

    played = frame.loc[frame["outcome"].notna() & (frame["date"] >= TRAIN_START)]
    future = frame.loc[frame["_prediction_fixture"]].sort_values("date", kind="stable")
    if played.empty:
        raise ValueError(f"No played matches found on or after {TRAIN_START.date()}")

    latest_played = played["date"].max()
    print(f"Latest played match in dataset: {latest_played.date()}")
    print(f"Data freshness: {pd.Timestamp.now() - latest_played}")

    previous_month = TODAY.to_period("M") - 1
    test = played.loc[
        (played["date"] >= previous_month.start_time)
        & (played["date"] < (previous_month + 1).start_time)
    ]
    if not test.empty:
        training = played.loc[played["date"] < previous_month.start_time].tail(MAX_TRAIN)
        classifier = get_model(args.model)
        classifier.fit(training.loc[:, feature_cols].to_numpy(), training["outcome"].to_numpy())
        probabilities = predict_probabilities(classifier, test.loc[:, feature_cols].to_numpy())
        predicted = np.asarray(OUTCOMES)[probabilities.argmax(axis=1)]
        print(
            f"\nQuick check {previous_month} ({len(test)} matches): "
            f"accuracy {accuracy_score(test['outcome'], predicted):.0%}, "
            f"log-loss {competition_log_loss(test['outcome'].to_numpy(), probabilities):.3f}"
        )

    if future.empty:
        print("\nNo unplayed fixtures found for today or later.")
        return

    classifier = get_model(args.model)
    training = played.tail(MAX_TRAIN)
    classifier.fit(training.loc[:, feature_cols].to_numpy(), training["outcome"].to_numpy())
    probabilities = predict_probabilities(classifier, future.loc[:, feature_cols].to_numpy())
    predicted = np.asarray(OUTCOMES)[probabilities.argmax(axis=1)]

    output = future.loc[:, ["date", "home_team", "away_team"]].copy()
    output["p_home_win"] = probabilities[:, 0]
    output["p_draw"] = probabilities[:, 1]
    output["p_away_win"] = probabilities[:, 2]
    filename = args.output or f"predictions_{pd.Timestamp.now():%Y%m%d}.csv"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(filename, index=False)

    print(f"\n{len(output)} fixture predictions -> {filename}\n")
    for row, prediction in zip(output.itertuples(), predicted, strict=True):
        print(
            f"  {row.date.date()}  {row.home_team:>20} vs {row.away_team:<20}  "
            f"-> {prediction:<9}  "
            f"H {row.p_home_win:4.0%} | D {row.p_draw:4.0%} | A {row.p_away_win:4.0%}"
        )


if __name__ == "__main__":
    main()
