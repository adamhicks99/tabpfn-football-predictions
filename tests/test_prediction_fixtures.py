"""Tests for explicit live-prediction fixture manifests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data import clean_data, load_data
from predict import append_prediction_fixtures, load_prediction_fixtures


class PredictionFixtureTest(unittest.TestCase):
    def _matches(self) -> pd.DataFrame:
        return clean_data(
            pd.DataFrame(
                {
                    "date": ["2026-06-01", "2026-06-27"],
                    "home_team": ["A", "C"],
                    "away_team": ["B", "D"],
                    "home_score": [1, np.nan],
                    "away_score": [0, np.nan],
                    "tournament": ["Friendly", "FIFA World Cup"],
                    "neutral": ["TRUE", "TRUE"],
                }
            )
        )

    def test_only_explicit_fixtures_are_marked_for_prediction(self) -> None:
        fixtures = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-06-28"]),
                "home_team": ["B"],
                "away_team": ["A"],
                "status": ["confirmed"],
            }
        )
        combined = append_prediction_fixtures(self._matches(), fixtures)
        selected = combined.loc[combined["_prediction_fixture"]]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["home_team"], "B")
        self.assertNotIn("C", set(combined["home_team"]))

    def test_manifest_rejects_past_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixtures.csv"
            pd.DataFrame(
                {
                    "date": ["2026-06-26"],
                    "home_team": ["A"],
                    "away_team": ["B"],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "past match"):
                load_prediction_fixtures(
                    path,
                    known_teams={"A", "B"},
                    earliest_date=pd.Timestamp("2026-06-27"),
                )

    def test_manifest_rejects_placeholders_and_unknown_teams(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixtures.csv"
            pd.DataFrame(
                {
                    "date": ["2026-06-28"],
                    "home_team": ["Group L winner"],
                    "away_team": ["B"],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "unresolved teams"):
                load_prediction_fixtures(
                    path,
                    known_teams={"A", "B"},
                    earliest_date=pd.Timestamp("2026-06-27"),
                )

    def test_refresh_does_not_erase_newer_local_scores(self) -> None:
        columns = {
            "date": ["2026-06-26"],
            "home_team": ["A"],
            "away_team": ["B"],
            "tournament": ["FIFA World Cup"],
            "neutral": ["TRUE"],
        }
        existing = pd.DataFrame({**columns, "home_score": [2], "away_score": [1]})
        downloaded = pd.DataFrame(
            {**columns, "home_score": [np.nan], "away_score": [np.nan]}
        )
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "results.csv"
            remote_path = Path(directory) / "download.csv"
            existing.to_csv(local_path, index=False)
            downloaded.to_csv(remote_path, index=False)
            refreshed = load_data(
                refresh=True,
                path=local_path,
                raw_url=str(remote_path),
            )
        self.assertEqual(refreshed.loc[0, "home_score"], 2)
        self.assertEqual(refreshed.loc[0, "away_score"], 1)


if __name__ == "__main__":
    unittest.main()
