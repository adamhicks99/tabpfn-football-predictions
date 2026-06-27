"""Tests for the flat model / feature-set / certified-dataset layout."""

from __future__ import annotations

import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from backtest import (
    compare_with_market,
    make_folds,
    parse_key_value_tags,
    run_backtest,
)
from data import clean_data
from datasets import EVAL_DATASETS, EvalDataset
from features import FEATURE_SETS, ODDS_FEATURES, add_odds_features, build_features
from models import MODELS, competition_log_loss, get_model
from odds import (
    APIResponse,
    OddsAPIClient,
    OddsAPIError,
    Quota,
    consensus_h2h,
    load_api_key,
    load_raw_response,
    write_raw_once,
)


class FrequencyClassifier:
    """Deterministic class-frequency classifier, registered as a test model."""

    def fit(self, features: np.ndarray, target: np.ndarray):
        self.classes_ = np.asarray(["away_win", "draw", "home_win"])
        counts = {value: int((target == value).sum()) for value in self.classes_}
        total = sum(counts.values())
        self.probabilities_ = np.asarray([counts[v] / total for v in self.classes_])
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(self.probabilities_, (len(features), 1))


# Adding a model is just a dict entry -- exercise that here.
MODELS["frequency"] = lambda random_state=42: FrequencyClassifier()


class ScoringTest(unittest.TestCase):
    def test_competition_log_loss_uses_submission_column_order(self) -> None:
        actual = np.asarray(["home_win", "draw", "away_win"])
        probabilities = np.asarray(
            [[0.80, 0.10, 0.10], [0.10, 0.80, 0.10], [0.10, 0.10, 0.80]]
        )
        self.assertAlmostEqual(competition_log_loss(actual, probabilities), -np.log(0.80))

    def test_competition_log_loss_clips_exact_zero(self) -> None:
        actual = np.asarray(["home_win"])
        probabilities = np.asarray([[0.0, 0.5, 0.5]])
        self.assertTrue(np.isfinite(competition_log_loss(actual, probabilities)))

    def test_competition_log_loss_rejects_non_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to one"):
            competition_log_loss(
                np.asarray(["home_win"]),
                np.asarray([[0.8, 0.3, 0.1]]),
            )


class FoldTest(unittest.TestCase):
    def test_overlapping_test_windows_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlapping"):
            make_folds(
                start=pd.Timestamp("2025-01-01"),
                end=pd.Timestamp("2026-01-01"),
                train_start=pd.Timestamp("2020-01-01"),
                test_months=3,
                step_months=1,
                train_months=None,
            )


class BacktestTest(unittest.TestCase):
    def test_walk_forward_produces_out_of_fold_predictions(self) -> None:
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        outcomes = np.resize(np.asarray(["home_win", "draw", "away_win"]), len(dates))
        frame = pd.DataFrame(
            {
                "date": dates,
                "home_team": ["Home"] * len(dates),
                "away_team": ["Away"] * len(dates),
                "tournament": ["Test Cup"] * len(dates),
                "outcome": outcomes,
                "strength": 1.0,
            }
        )
        spec = EvalDataset(
            name="t",
            description="synthetic",
            start="2021-01-01",
            train_start="2020-01-01",
            end="2022-01-01",
            test_months=3,
            step_months=3,
            min_train_rows=6,
            max_train_rows=None,
        )
        result = run_backtest(frame, ["strength"], spec, "frequency", seed=0)

        self.assertEqual(result.summary["folds"], 4)
        self.assertEqual(result.summary["matches"], 12)
        self.assertEqual(result.predictions["source_index"].min(), 12)
        self.assertEqual(result.summary["model"], "frequency")
        self.assertTrue(
            np.allclose(result.predictions[list(OUTCOMES_COLS)].sum(axis=1), 1.0)
        )

    def test_blend_weight_uses_only_completed_folds(self) -> None:
        model_predictions = pd.DataFrame(
            {
                "fold": ["fold_1"] * 3 + ["fold_2"] * 3,
                "source_index": range(6),
                "actual": ["home_win"] * 6,
                "p_home_win": [0.4] * 6,
                "p_draw": [0.3] * 6,
                "p_away_win": [0.3] * 6,
            }
        )
        feature_frame = pd.DataFrame(
            {
                "odds_market_p_home": [0.8] * 6,
                "odds_market_p_draw": [0.1] * 6,
                "odds_market_p_away": [0.1] * 6,
            }
        )
        predictions, _ = compare_with_market(model_predictions, feature_frame, minimum_history=3)
        self.assertTrue((predictions.loc[:2, "market_weight"] == 0).all())
        self.assertTrue((predictions.loc[3:, "market_weight"] == 1).all())


OUTCOMES_COLS = ("p_home_win", "p_draw", "p_away_win")


class FeatureTest(unittest.TestCase):
    def _played(self) -> pd.DataFrame:
        raw = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
                "home_team": ["A", "B", "A", "C"],
                "away_team": ["B", "C", "C", "A"],
                "home_score": [1, 2, 0, 3],
                "away_score": [0, 2, 1, 1],
                "tournament": ["Friendly"] * 4,
                "neutral": ["FALSE"] * 4,
            }
        )
        return clean_data(raw)

    def test_build_base_features_returns_26_columns(self) -> None:
        frame, cols = build_features(self._played(), "base")
        self.assertEqual(len(cols), 26)
        self.assertIn("elo_diff", cols)
        self.assertTrue(set(cols).issubset(frame.columns))

    def test_odds_join_picks_latest_snapshot_before_cutoff(self) -> None:
        matches = pd.DataFrame(
            {"date": ["2026-06-28"], "home_team": ["Brazil"], "away_team": ["Japan"]}
        )
        base = {feature: [0.5, 0.9] for feature in ODDS_FEATURES}
        snapshots = pd.DataFrame(
            {
                "date": ["2026-06-28", "2026-06-28"],
                "home_team": ["Brazil", "Brazil"],
                "away_team": ["Japan", "Japan"],
                "observed_at": ["2026-06-27T10:00:00Z", "2026-06-27T20:00:00Z"],
                "submission_cutoff": ["2026-06-27T18:00:00Z", "2026-06-27T18:00:00Z"],
                **base,
            }
        )
        # Distinguish the two snapshots on market_p_home.
        snapshots["market_p_home"] = [0.60, 0.75]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odds.csv"
            snapshots.to_csv(path, index=False)
            features = add_odds_features(matches, path)
        self.assertAlmostEqual(features.loc[0, "odds_market_p_home"], 0.60)
        self.assertAlmostEqual(features.loc[0, "odds_snapshot_age_hours"], 8.0)
        self.assertEqual(features.loc[0, "odds_market_available"], 1.0)

    def test_odds_join_handles_empty_snapshot_file(self) -> None:
        matches = pd.DataFrame(
            {"date": ["2026-06-28"], "home_team": ["Brazil"], "away_team": ["Japan"]}
        )
        columns = [
            "date",
            "home_team",
            "away_team",
            "observed_at",
            "submission_cutoff",
            *ODDS_FEATURES,
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odds.csv"
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            features = add_odds_features(matches, path)
        self.assertTrue(np.isnan(features.loc[0, "odds_market_p_home"]))
        self.assertEqual(features.loc[0, "odds_market_available"], 0.0)

    def test_missing_either_score_has_no_outcome(self) -> None:
        raw = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01"],
                "home_team": ["A", "A"],
                "away_team": ["B", "B"],
                "home_score": [1, 2],
                "away_score": [np.nan, 0],
                "tournament": ["Friendly", "Friendly"],
                "neutral": ["FALSE", "FALSE"],
            }
        )
        cleaned = clean_data(raw)
        frame, _ = build_features(cleaned, "base")
        self.assertTrue(pd.isna(cleaned.loc[0, "outcome"]))
        self.assertEqual(frame.loc[1, "home_elo"], 1500.0)
        self.assertEqual(frame.loc[1, "away_elo"], 1500.0)


class RegistryDictTest(unittest.TestCase):
    def test_built_in_models(self) -> None:
        self.assertIn("logistic", MODELS)
        self.assertIn("tabpfn", MODELS)
        model = get_model("logistic", random_state=0)
        self.assertTrue(callable(getattr(model, "predict_proba", None)))

    def test_unknown_model_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown model 'nope'"):
            get_model("nope")

    def test_feature_sets_and_datasets_present(self) -> None:
        self.assertEqual(set(FEATURE_SETS), {"base", "base+odds"})
        self.assertIn("full_2018", EVAL_DATASETS)
        self.assertIsInstance(EVAL_DATASETS["full_2018"], EvalDataset)


class OddsTest(unittest.TestCase):
    def test_api_client_rejects_untrusted_paths_before_network_access(self) -> None:
        client = OddsAPIClient("test-secret")
        with self.assertRaisesRegex(ValueError, "Unsafe API path"):
            client.get("https://attacker.invalid/steal")

    def test_api_client_redacts_secret_from_network_errors(self) -> None:
        client = OddsAPIClient("test-secret")
        error = urllib.error.URLError("connection failed for test-secret")
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(OddsAPIError) as caught:
                client.get("sports")
        self.assertNotIn("test-secret", str(caught.exception))
        self.assertIn("[REDACTED]", str(caught.exception))

    def test_api_key_environment_variable_is_explicitly_configured(self) -> None:
        with patch.dict(os.environ, {"CUSTOM_ODDS_CREDENTIAL": "test-secret"}, clear=True):
            self.assertEqual(
                load_api_key(env_var="CUSTOM_ODDS_CREDENTIAL"),
                "test-secret",
            )

    def test_api_key_has_no_implicit_credential_location(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "No API credential"):
                load_api_key()

    def test_bookmaker_consensus_is_devigged_and_normalized(self) -> None:
        event = {
            "home_team": "Brazil",
            "away_team": "Japan",
            "bookmakers": [
                {"markets": [{"key": "h2h", "outcomes": [
                    {"name": "Brazil", "price": 1.5}, {"name": "Draw", "price": 4.0}, {"name": "Japan", "price": 8.0}]}]},
                {"markets": [{"key": "h2h", "outcomes": [
                    {"name": "Brazil", "price": 1.6}, {"name": "Draw", "price": 3.8}, {"name": "Japan", "price": 7.0}]}]},
            ],
        }
        consensus = consensus_h2h(event)
        self.assertEqual(consensus["book_count"], 2)
        self.assertAlmostEqual(
            consensus["market_p_home"] + consensus["market_p_draw"] + consensus["market_p_away"], 1.0
        )

    def test_raw_cache_round_trip_avoids_a_second_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parameters = {"date": "2022-12-18T14:00:00Z", "markets": ["h2h"], "regions": ["eu"]}
            write_raw_once(
                directory,
                kind="odds",
                sport_key="soccer_fifa_world_cup",
                timestamp="2022-12-18T14:00:00Z",
                parameters=parameters,
                body={"timestamp": "2022-12-18T13:55:39Z", "data": []},
                quota=Quota(remaining=19990, used=10, last=10),
            )
            network = Mock()
            cached = load_raw_response(
                directory,
                kind="odds",
                sport_key="soccer_fifa_world_cup",
                timestamp="2022-12-18T14:00:00Z",
                parameters=parameters,
            )
            response = cached if cached is not None else network()
        self.assertIsInstance(response, APIResponse)
        self.assertEqual(response.quota.last, 0)
        network.assert_not_called()

    def test_raw_cache_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Unsafe sport_key"):
                write_raw_once(
                    directory,
                    kind="odds",
                    sport_key="../outside",
                    timestamp="2026-06-28T00:00:00Z",
                    parameters={},
                    body=[],
                    quota=Quota(remaining=None, used=None, last=None),
                )


class TagTest(unittest.TestCase):
    def test_parse_key_value_tags(self) -> None:
        self.assertEqual(
            parse_key_value_tags(["hypothesis=add rest", "src=http://x?a=b"]),
            {"hypothesis": "add rest", "src": "http://x?a=b"},
        )

    def test_parse_key_value_tags_rejects_bad_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "KEY=VALUE"):
            parse_key_value_tags(["nope"])
        with self.assertRaisesRegex(ValueError, "KEY=VALUE"):
            parse_key_value_tags(["=value"])


if __name__ == "__main__":
    unittest.main()
