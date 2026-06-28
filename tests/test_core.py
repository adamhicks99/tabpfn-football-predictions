from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from football_lab.catalog import Catalog, DatasetKind
from football_lab.datasets import build_evaluation_dataset, build_training_dataset
from football_lab.experiments import run_evaluation
from football_lab.features import build_features
from football_lab.results import RESULT_COLUMNS, build_competition_result, export_result


class CoreWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "results.csv"
        self.fixtures = self.root / "fixtures.csv"
        self.workspace = self.root / "workspace"
        self._write_results()
        pd.DataFrame(
            {
                "date": ["2022-01-15", "2022-01-16"],
                "home_team": ["A", "C"],
                "away_team": ["B", "A"],
                "tournament": ["Test Cup", "Test Cup"],
                "status": ["confirmed", "confirmed"],
                "neutral": ["TRUE", "TRUE"],
            }
        ).to_csv(self.fixtures, index=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_datasets_are_content_versioned_and_idempotent(self) -> None:
        catalog = Catalog(self.workspace)
        first = build_training_dataset(
            catalog,
            name="baseline",
            source=self.source,
            start="2020-01-01",
            end="2021-01-01",
        )
        second = build_training_dataset(
            catalog,
            name="baseline",
            source=self.source,
            start="2020-01-01",
            end="2021-01-01",
        )
        changed = build_training_dataset(
            catalog,
            name="baseline",
            source=self.source,
            start="2020-01-01",
            end="2021-02-01",
        )

        self.assertEqual(first.id, second.id)
        self.assertNotEqual(first.id, changed.id)
        self.assertTrue((first.path / "data.csv").is_file())
        self.assertTrue((first.path / "manifest.json").is_file())
        self.assertEqual(len(catalog.list_datasets(DatasetKind.TRAINING)), 2)

    def test_dataset_integrity_is_verified_when_loaded(self) -> None:
        catalog = Catalog(self.workspace)
        training = self._training(catalog)
        with (training.path / "data.csv").open("a", encoding="utf-8") as handle:
            handle.write("\n")

        with self.assertRaisesRegex(RuntimeError, "integrity check"):
            catalog.load_frame(training)

    def test_evaluation_run_is_queryable_and_reproducible(self) -> None:
        catalog = Catalog(self.workspace)
        training = self._training(catalog)
        evaluation = build_evaluation_dataset(
            catalog,
            name="holdout",
            source=self.source,
            start="2021-01-01",
            end="2022-01-01",
        )
        result = run_evaluation(
            catalog,
            model_name="logistic",
            training_reference=training.id,
            evaluation_reference=evaluation.id,
            seed=7,
            tags={"candidate": "baseline"},
        )

        self.assertIn("log_loss", result["metrics"])
        self.assertTrue(Path(result["predictions"]).is_file())
        rows = catalog.query(
            """
            SELECT e.model_name, e.status, m.value AS log_loss
            FROM experiments e
            JOIN metrics m ON m.experiment_id = e.id
            WHERE m.name = 'log_loss'
            """
        )
        self.assertEqual(rows[0]["model_name"], "logistic")
        self.assertEqual(rows[0]["status"], "succeeded")
        self.assertGreater(rows[0]["log_loss"], 0)

    def test_same_day_matches_share_the_same_prior_state(self) -> None:
        matches = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2021-01-01", "2021-01-01", "2021-01-02"]
                ),
                "home_team": ["A", "A", "A"],
                "away_team": ["B", "C", "D"],
                "home_score": [1, 2, 0],
                "away_score": [0, 0, 0],
                "neutral": [1, 1, 1],
                "importance": [30.0, 30.0, 30.0],
            }
        )

        featured = build_features(matches)

        self.assertEqual(featured.loc[0, "home_played"], 0)
        self.assertEqual(featured.loc[1, "home_played"], 0)
        self.assertEqual(featured.loc[2, "home_played"], 2)

    def test_competition_result_has_exact_upload_schema_and_lineage(self) -> None:
        catalog = Catalog(self.workspace)
        training = self._training(catalog)
        result, experiment_id = build_competition_result(
            catalog,
            name="round-32",
            model_name="logistic",
            training_reference=training.id,
            history_source=self.source,
            fixtures_source=self.fixtures,
            as_of="2022-01-01",
            seed=11,
        )
        frame = catalog.load_frame(result)

        self.assertEqual(result.kind, DatasetKind.RESULT)
        self.assertEqual(tuple(frame.columns), RESULT_COLUMNS)
        self.assertTrue(
            np.allclose(frame.loc[:, RESULT_COLUMNS[3:]].sum(axis=1), 1.0)
        )
        experiment = catalog.query(
            "SELECT status, result_dataset_id FROM experiments WHERE id = ?",
            (experiment_id,),
        )[0]
        self.assertEqual(experiment["status"], "succeeded")
        self.assertEqual(experiment["result_dataset_id"], result.id)

        exported = export_result(catalog, result.id, self.root / "upload.csv")
        self.assertEqual(tuple(pd.read_csv(exported).columns), RESULT_COLUMNS)

    def test_catalog_rejects_mutating_queries(self) -> None:
        catalog = Catalog(self.workspace)
        with self.assertRaisesRegex(ValueError, "read-only"):
            catalog.query("DELETE FROM experiments")
        with self.assertRaisesRegex(ValueError, "query failed"):
            catalog.query(
                "WITH ids AS (SELECT id FROM experiments) "
                "DELETE FROM experiments WHERE id IN (SELECT id FROM ids)"
            )

    def test_failed_evaluation_remains_queryable(self) -> None:
        catalog = Catalog(self.workspace)
        training = self._training(catalog)
        overlapping = build_evaluation_dataset(
            catalog,
            name="overlap",
            source=self.source,
            start="2020-12-01",
            end="2021-06-01",
        )
        with self.assertRaisesRegex(ValueError, "must end before"):
            run_evaluation(
                catalog,
                model_name="logistic",
                training_reference=training.id,
                evaluation_reference=overlapping.id,
            )
        failed = catalog.query(
            "SELECT status, error FROM experiments WHERE status = 'failed'"
        )
        self.assertEqual(len(failed), 1)
        self.assertIn("must end before", failed[0]["error"])

    def _training(self, catalog: Catalog):
        return build_training_dataset(
            catalog,
            name="baseline",
            source=self.source,
            start="2020-01-01",
            end="2021-01-01",
        )

    def _write_results(self) -> None:
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        teams = [("A", "B"), ("B", "C"), ("C", "A")]
        scores = [(2, 0), (1, 1), (0, 2)]
        rows = []
        for index, date in enumerate(dates):
            home, away = teams[index % len(teams)]
            home_score, away_score = scores[index % len(scores)]
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "tournament": "Test Cup",
                    "city": "Test City",
                    "country": "Test Country",
                    "neutral": True,
                }
            )
        pd.DataFrame(rows).to_csv(self.source, index=False)


if __name__ == "__main__":
    unittest.main()
