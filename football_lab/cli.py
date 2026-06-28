from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from football_lab.catalog import Catalog, DatasetKind, DatasetRecord
from football_lab.datasets import build_evaluation_dataset, build_training_dataset
from football_lab.experiments import parse_tags, run_evaluation
from football_lab.models import MODEL_FACTORIES
from football_lab.results import build_competition_result, export_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="football-lab",
        description="Build versioned football datasets and track model experiments.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace"),
        help="Artifact store and SQLite catalog directory",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    dataset = commands.add_parser("dataset", help="Manage versioned datasets")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    training = dataset_commands.add_parser(
        "build-training",
        help="Build an immutable training dataset",
    )
    _dataset_build_arguments(training)
    training.add_argument(
        "--max-rows",
        type=int,
        help="Keep only the most recent N rows in the selected training window",
    )
    evaluation = dataset_commands.add_parser(
        "build-evaluation",
        help="Build an immutable evaluation dataset",
    )
    _dataset_build_arguments(evaluation)
    dataset_list = dataset_commands.add_parser("list", help="List dataset versions")
    dataset_list.add_argument("--kind", choices=[kind.value for kind in DatasetKind])
    dataset_show = dataset_commands.add_parser("show", help="Show a dataset manifest")
    dataset_show.add_argument("reference")

    experiment = commands.add_parser("experiment", help="Run and query experiments")
    experiment_commands = experiment.add_subparsers(
        dest="experiment_command",
        required=True,
    )
    experiment_run = experiment_commands.add_parser(
        "run",
        help="Train on one dataset and score one evaluation dataset",
    )
    experiment_run.add_argument("--model", choices=list(MODEL_FACTORIES), required=True)
    experiment_run.add_argument("--training", required=True)
    experiment_run.add_argument("--evaluation", required=True)
    experiment_run.add_argument("--seed", type=int, default=42)
    experiment_run.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE")
    experiment_run.add_argument("--note")
    experiment_list = experiment_commands.add_parser("list", help="List experiments")
    experiment_list.add_argument("--limit", type=int, default=50)
    experiment_query = experiment_commands.add_parser(
        "query",
        help="Run a read-only SQL query against the catalog",
    )
    experiment_query.add_argument("sql")

    result = commands.add_parser("result", help="Build Prior competition results")
    result_commands = result.add_subparsers(dest="result_command", required=True)
    result_build = result_commands.add_parser(
        "build",
        help="Train a model and create a versioned competition result",
    )
    result_build.add_argument("--name", required=True)
    result_build.add_argument("--model", choices=list(MODEL_FACTORIES), required=True)
    result_build.add_argument("--training", required=True)
    result_build.add_argument("--history", type=Path, required=True)
    result_build.add_argument("--fixtures", type=Path, required=True)
    result_build.add_argument("--as-of", required=True)
    result_build.add_argument("--seed", type=int, default=42)
    result_build.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE")
    result_build.add_argument("--note")
    result_export = result_commands.add_parser(
        "export",
        help="Export a result dataset for competition upload",
    )
    result_export.add_argument("reference")
    result_export.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    catalog = Catalog(args.workspace)
    if args.command == "dataset":
        if args.dataset_command == "build-training":
            record = build_training_dataset(
                catalog,
                name=args.name,
                source=args.source,
                start=args.start,
                end=args.end,
                max_rows=args.max_rows,
            )
            _print_dataset(record)
            return 0
        if args.dataset_command == "build-evaluation":
            record = build_evaluation_dataset(
                catalog,
                name=args.name,
                source=args.source,
                start=args.start,
                end=args.end,
            )
            _print_dataset(record)
            return 0
        if args.dataset_command == "list":
            kind = DatasetKind(args.kind) if args.kind else None
            _print_json([_dataset_summary(record) for record in catalog.list_datasets(kind)])
            return 0
        record = catalog.get_dataset(args.reference)
        _print_json(
            {
                **_dataset_summary(record),
                "columns": list(record.columns),
                "metadata": record.metadata,
                "path": str(record.path),
            }
        )
        return 0

    if args.command == "experiment":
        if args.experiment_command == "run":
            _print_json(
                run_evaluation(
                    catalog,
                    model_name=args.model,
                    training_reference=args.training,
                    evaluation_reference=args.evaluation,
                    seed=args.seed,
                    tags=parse_tags(args.tag),
                    note=args.note,
                )
            )
            return 0
        if args.experiment_command == "list":
            _print_json(catalog.list_experiments(args.limit))
            return 0
        _print_json(catalog.query(args.sql))
        return 0

    if args.result_command == "build":
        record, experiment_id = build_competition_result(
            catalog,
            name=args.name,
            model_name=args.model,
            training_reference=args.training,
            history_source=args.history,
            fixtures_source=args.fixtures,
            as_of=args.as_of,
            seed=args.seed,
            tags=parse_tags(args.tag),
            note=args.note,
        )
        _print_json(
            {
                **_dataset_summary(record),
                "experiment_id": experiment_id,
                "upload_file": str(record.path / "data.csv"),
            }
        )
        return 0
    output = export_result(catalog, args.reference, args.output)
    _print_json({"output": str(output)})
    return 0


def _dataset_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)


def _dataset_summary(record: DatasetRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "kind": record.kind.value,
        "name": record.name,
        "version": record.version,
        "rows": record.row_count,
        "content_sha256": record.content_sha256,
        "created_at": record.created_at,
    }


def _print_dataset(record: DatasetRecord) -> None:
    _print_json({**_dataset_summary(record), "path": str(record.path)})


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))
