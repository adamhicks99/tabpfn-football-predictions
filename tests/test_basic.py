"""Small sanity tests for the flat odds-centric project."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data import merge_odds, odds_covered
from features import BASE_FEATURES, FEATURE_SETS, build_features, feature_columns
from models import OUTCOMES, log_loss, ordered_probabilities


class _FakeModel:
    """Minimal classifier with shuffled class order, to test reordering."""

    classes_ = np.asarray(["away_win", "draw", "home_win"])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile([0.1, 0.3, 0.6], (len(features), 1))


class MetricsTest(unittest.TestCase):
    def test_log_loss_matches_hand_value(self) -> None:
        actual = np.asarray(["home_win", "draw", "away_win"])
        probs = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        self.assertAlmostEqual(log_loss(actual, probs), -np.log(0.8))

    def test_ordered_probabilities_reorders_to_home_draw_away(self) -> None:
        probs = ordered_probabilities(_FakeModel(), np.zeros((2, 3)))
        # FakeModel classes are [away, draw, home] with [0.1, 0.3, 0.6];
        # reordered to home/draw/away -> [0.6, 0.3, 0.1].
        self.assertTrue(np.allclose(probs[0], [0.6, 0.3, 0.1]))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))


class FeatureTest(unittest.TestCase):
    def _matches(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
                "home_team": ["A", "B", "A", "C"],
                "away_team": ["B", "C", "C", "A"],
                "home_score": [1.0, 2.0, 0.0, 3.0],
                "away_score": [0.0, 2.0, 1.0, 1.0],
                "neutral": [0, 0, 0, 0],
                "importance": [30.0, 30.0, 30.0, 30.0],
                "outcome": ["home_win", "draw", "away_win", "home_win"],
                "tournament": ["Friendly"] * 4,
            }
        )

    def test_build_features_adds_base_columns(self) -> None:
        featured = build_features(self._matches())
        self.assertTrue(set(BASE_FEATURES).issubset(featured.columns))
        self.assertEqual(len(featured), 4)

    def test_feature_sets(self) -> None:
        self.assertEqual(set(FEATURE_SETS), {"base", "odds", "base+odds"})
        self.assertEqual(len(feature_columns("base")), 26)
        self.assertIn("odds_market_p_home", feature_columns("base+odds"))


class OddsMergeTest(unittest.TestCase):
    def test_merge_odds_and_coverage(self) -> None:
        matches = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-06-20", "2024-06-21"]),
                "home_team": ["Argentina", "Spain"],
                "away_team": ["Chile", "Italy"],
            }
        )
        odds = pd.DataFrame(
            {
                "date": ["2024-06-20"],
                "home_team": ["Argentina"],
                "away_team": ["Chile"],
                "market_p_home": [0.6],
                "market_p_draw": [0.25],
                "market_p_away": [0.15],
                "market_overround": [1.05],
                "book_count": [20],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odds.csv"
            odds.to_csv(path, index=False)
            merged = merge_odds(matches, path)
        self.assertAlmostEqual(merged.loc[0, "odds_market_p_home"], 0.6)
        self.assertTrue(pd.isna(merged.loc[1, "odds_market_p_home"]))
        self.assertEqual(len(odds_covered(merged)), 1)


if __name__ == "__main__":
    unittest.main()
