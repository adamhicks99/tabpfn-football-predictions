"""Walk-forward backtesting and the one ``evaluate`` entry point.

``evaluate(model, feature_set, dataset)`` is the unit of work: it scores one
model on one feature set against one certified dataset and returns log-loss (the
metric to minimize) plus accuracy. Run it from the CLI:

    python backtest.py --model tabpfn --features base --dataset full_2018
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from data import DEFAULT_DATA_PATH, load_data
from datasets import EVAL_DATASETS, EvalDataset, get_dataset
from features import (
    DEFAULT_ODDS_CSV,
    FEATURE_SETS,
    MARKET_COLUMNS,
    add_odds_features,
    build_features,
)
from models import (
    MODELS,
    OUTCOMES,
    competition_log_loss,
    get_model,
    predict_probabilities,
)

PROBABILITY_COLUMNS = ("p_home_win", "p_draw", "p_away_win")
# MLflow defaults (kept here so argparse doesn't import mlflow just to show help).
DEFAULT_EXPERIMENT = "football-backtests"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


@dataclass(frozen=True)
class Fold:
    """One chronological train/test split."""

    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def name(self) -> str:
        return f"fold_{self.number:03d}_{self.test_start:%Y%m%d}_{self.test_end:%Y%m%d}"


@dataclass
class BacktestResult:
    """Out-of-fold predictions, per-fold metrics, calibration, and summary."""

    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    calibration: pd.DataFrame
    summary: dict[str, object]
    comparison_predictions: pd.DataFrame | None = None


# --------------------------------------------------------------------------- #
# The entry point
# --------------------------------------------------------------------------- #


def evaluate(
    model: str,
    feature_set: str = "base",
    dataset: str = "full_2018",
    *,
    seed: int = 42,
    odds_csv: str | Path | None = None,
    compare_market: bool = False,
    blend_min_history: int = 30,
    matches: pd.DataFrame | None = None,
    refresh: bool = False,
) -> BacktestResult:
    """Score one (model, feature_set) against one certified dataset -> log-loss."""
    spec = get_dataset(dataset)
    default_data = matches is None
    if matches is None:
        matches = load_data(refresh=refresh)
    frame, feature_cols = build_features(matches, feature_set, odds_csv)
    if compare_market and not all(c in frame.columns for c in MARKET_COLUMNS):
        # Join market odds for scoring only (not as model features).
        frame = frame.join(add_odds_features(matches, odds_csv or DEFAULT_ODDS_CSV))

    result = run_backtest(frame, feature_cols, spec, model, seed=seed)
    result.summary["feature_set"] = feature_set
    result.summary["eval_dataset"] = spec.name
    result.summary["dataset_spec"] = spec.as_params()
    result.summary["seed"] = seed
    result.summary["data"] = _data_fingerprint(
        matches,
        source_path=DEFAULT_DATA_PATH if default_data else None,
    )

    if compare_market:
        comparison, metrics = compare_with_market(
            result.predictions, frame, minimum_history=blend_min_history
        )
        result.comparison_predictions = comparison
        result.summary["market_comparison"] = metrics
    return result


# --------------------------------------------------------------------------- #
# Walk-forward engine
# --------------------------------------------------------------------------- #


def make_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_start: pd.Timestamp,
    test_months: int,
    step_months: int,
    train_months: int | None,
) -> list[Fold]:
    """Generate expanding or rolling calendar folds."""
    if start >= end:
        raise ValueError("Backtest start must be earlier than end")
    if train_start >= end:
        raise ValueError("Training start must be earlier than backtest end")
    if test_months < 1 or step_months < 1:
        raise ValueError("Fold and step lengths must be positive")
    if step_months < test_months:
        raise ValueError("step_months must be >= test_months (no overlapping test windows)")

    boundaries: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    test_start = start
    while test_start < end:
        test_end = min(test_start + pd.DateOffset(months=test_months), end)
        boundaries.append((test_start, test_end))
        test_start += pd.DateOffset(months=step_months)

    folds: list[Fold] = []
    for number, (fold_start, fold_end) in enumerate(boundaries, start=1):
        window_start = train_start
        if train_months is not None:
            window_start = max(window_start, fold_start - pd.DateOffset(months=train_months))
        folds.append(
            Fold(number, window_start, fold_start, fold_start, fold_end)
        )
    return folds


def run_backtest(
    frame: pd.DataFrame,
    feature_cols: list[str],
    spec: EvalDataset,
    model: str,
    *,
    seed: int = 42,
) -> BacktestResult:
    """Fit a fresh model per fold and collect true out-of-fold probabilities."""
    _validate_frame(frame, feature_cols)
    played = frame.loc[frame["outcome"].notna()].sort_values("date", kind="stable")
    latest_played = played["date"].max()
    end = pd.Timestamp(spec.end) if spec.end else latest_played.normalize() + pd.DateOffset(days=1)
    folds = make_folds(
        start=pd.Timestamp(spec.start),
        end=end,
        train_start=pd.Timestamp(spec.train_start),
        test_months=spec.test_months,
        step_months=spec.step_months,
        train_months=spec.train_months,
    )

    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for fold in folds:
        train = played.loc[(played["date"] >= fold.train_start) & (played["date"] < fold.train_end)]
        if spec.max_train_rows is not None:
            train = train.tail(spec.max_train_rows)
        if len(train) < spec.min_train_rows:
            continue
        test = played.loc[(played["date"] >= fold.test_start) & (played["date"] < fold.test_end)]
        if spec.tournament_regex:
            test = test.loc[
                test["tournament"].str.contains(
                    spec.tournament_regex, case=False, na=False, regex=True
                )
            ]
        if test.empty:
            continue

        classifier = get_model(model, seed)
        classifier.fit(train.loc[:, feature_cols].to_numpy(), train["outcome"].to_numpy())
        probabilities = predict_probabilities(classifier, test.loc[:, feature_cols].to_numpy())
        predictions = _prediction_frame(test, probabilities, fold)
        metric_rows.append(
            _metrics(
                predictions,
                train_rows=len(train),
                missing_feature_rate=float(test.loc[:, feature_cols].isna().to_numpy().mean()),
            )
        )
        prediction_parts.append(predictions)

    if not prediction_parts:
        raise ValueError("No folds produced predictions. Check dates, filters, and min_train_rows.")

    predictions = pd.concat(prediction_parts, ignore_index=True)
    fold_metrics = pd.DataFrame(metric_rows)
    aggregate = _metrics(
        predictions,
        train_rows=None,
        missing_feature_rate=float(
            frame.loc[predictions["source_index"], feature_cols].isna().to_numpy().mean()
        ),
    )
    summary: dict[str, object] = {
        "model": model,
        "matches": int(len(predictions)),
        "folds": int(len(fold_metrics)),
        "features": int(len(feature_cols)),
        "feature_columns": list(feature_cols),
        "accuracy": aggregate["accuracy"],
        "log_loss": aggregate["log_loss"],
        "uniform_log_loss": math.log(len(OUTCOMES)),
        "log_loss_improvement_vs_uniform": math.log(len(OUTCOMES)) - float(aggregate["log_loss"]),
        "multiclass_brier": aggregate["multiclass_brier"],
        "mean_probability_on_actual": aggregate["mean_probability_on_actual"],
        "missing_feature_rate": aggregate["missing_feature_rate"],
        "window": {
            "start": pd.Timestamp(spec.start).isoformat(),
            "end": end.isoformat(),
            "train_start": pd.Timestamp(spec.train_start).isoformat(),
        },
    }
    return BacktestResult(
        predictions=predictions,
        fold_metrics=fold_metrics,
        calibration=calibration_table(predictions),
        summary=summary,
    )


# --------------------------------------------------------------------------- #
# Metrics, calibration, market comparison, artifacts
# --------------------------------------------------------------------------- #


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Return one-vs-rest reliability bins for each outcome."""
    rows: list[dict[str, object]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    actual = predictions["actual"].to_numpy()
    for outcome, column in zip(OUTCOMES, PROBABILITY_COLUMNS, strict=True):
        probabilities = predictions[column].to_numpy()
        observed = (actual == outcome).astype(float)
        assignments = np.minimum(np.searchsorted(edges, probabilities, side="right") - 1, bins - 1)
        for bin_index in range(bins):
            mask = assignments == bin_index
            if not mask.any():
                continue
            rows.append(
                {
                    "outcome": outcome,
                    "bin_lower": edges[bin_index],
                    "bin_upper": edges[bin_index + 1],
                    "count": int(mask.sum()),
                    "mean_probability": float(probabilities[mask].mean()),
                    "observed_frequency": float(observed[mask].mean()),
                    "calibration_gap": float(observed[mask].mean() - probabilities[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def compare_with_market(
    model_predictions: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    minimum_history: int = 30,
    weight_step: float = 0.02,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    """Add market + strictly past-only blended probabilities, scored on matched rows."""
    missing = set(MARKET_COLUMNS).difference(feature_frame.columns)
    if missing:
        raise ValueError(f"Feature frame is missing market columns: {', '.join(sorted(missing))}")
    output = model_predictions.copy()
    market = feature_frame.loc[output["source_index"], list(MARKET_COLUMNS)].reset_index(drop=True)
    market.columns = [f"market_{name}" for name in PROBABILITY_COLUMNS]
    output = pd.concat([output.reset_index(drop=True), market], axis=1)
    output["market_available"] = market.notna().all(axis=1)
    for column in PROBABILITY_COLUMNS:
        output[f"blend_{column}"] = output[column]
    output["market_weight"] = 0.0

    history: list[int] = []
    for fold in output["fold"].drop_duplicates():
        eligible = output.index[output["fold"].eq(fold) & output["market_available"]].tolist()
        weight = _best_weight(output.loc[history], weight_step) if len(history) >= minimum_history else 0.0
        if eligible:
            model_p = output.loc[eligible, list(PROBABILITY_COLUMNS)].to_numpy()
            market_p = output.loc[eligible, [f"market_{n}" for n in PROBABILITY_COLUMNS]].to_numpy()
            output.loc[eligible, "market_weight"] = weight
            output.loc[eligible, [f"blend_{n}" for n in PROBABILITY_COLUMNS]] = (
                (1.0 - weight) * model_p + weight * market_p
            )
        history.extend(eligible)

    matched = output.loc[output["market_available"]]
    metrics: dict[str, float | int | None] = {
        "matched_matches": int(len(matched)),
        "coverage": float(output["market_available"].mean()),
        "model_log_loss_matched": _loss(matched, ""),
        "market_log_loss": _loss(matched, "market_"),
        "blend_log_loss": _loss(matched, "blend_"),
        "last_market_weight": float(matched["market_weight"].iloc[-1]) if len(matched) else None,
    }
    return output, metrics


def write_artifacts(result: BacktestResult, output_dir: str | Path) -> Path:
    """Persist predictions, fold metrics, calibration, summary, and any blend output."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(path / "predictions.csv", index=False)
    result.fold_metrics.to_csv(path / "fold_metrics.csv", index=False)
    result.calibration.to_csv(path / "calibration.csv", index=False)
    with (path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if result.comparison_predictions is not None:
        result.comparison_predictions.to_csv(path / "market_comparison_predictions.csv", index=False)
    return path


def parse_key_value_tags(pairs: list[str]) -> dict[str, str]:
    """Parse ``KEY=VALUE`` CLI tag strings into a dict, rejecting malformed input."""
    tags: dict[str, str] = {}
    for item in pairs:
        key, separator, value = item.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"Tag {item!r} must be formatted as KEY=VALUE")
        tags[key] = value.strip()
    return tags


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _prediction_frame(test: pd.DataFrame, probabilities: np.ndarray, fold: Fold) -> pd.DataFrame:
    predicted = np.asarray(OUTCOMES)[probabilities.argmax(axis=1)]
    columns = [c for c in ("date", "home_team", "away_team", "tournament", "neutral") if c in test.columns]
    output = test.loc[:, columns].copy()
    output.insert(0, "source_index", test.index)
    output.insert(0, "fold", fold.name)
    output["actual"] = test["outcome"].to_numpy()
    output["predicted"] = predicted
    output["p_home_win"] = probabilities[:, 0]
    output["p_draw"] = probabilities[:, 1]
    output["p_away_win"] = probabilities[:, 2]
    return output.reset_index(drop=True)


def _metrics(predictions: pd.DataFrame, train_rows: int | None, missing_feature_rate: float) -> dict[str, object]:
    probabilities = predictions.loc[:, list(PROBABILITY_COLUMNS)].to_numpy()
    actual = predictions["actual"].to_numpy()
    predicted = predictions["predicted"].to_numpy()
    class_indices = {outcome: index for index, outcome in enumerate(OUTCOMES)}
    actual_indices = np.asarray([class_indices[value] for value in actual])
    one_hot = np.eye(len(OUTCOMES))[actual_indices]
    actual_probability = probabilities[np.arange(len(actual)), actual_indices]
    row: dict[str, object] = {
        "matches": int(len(predictions)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "log_loss": competition_log_loss(actual, probabilities),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "mean_probability_on_actual": float(actual_probability.mean()),
        "missing_feature_rate": missing_feature_rate,
    }
    if train_rows is not None:
        row = {
            "fold": predictions["fold"].iloc[0],
            "test_start": predictions["date"].min(),
            "test_end": predictions["date"].max(),
            "train_rows": train_rows,
            **row,
        }
    return row


def _validate_frame(frame: pd.DataFrame, feature_cols: list[str]) -> None:
    required = {"date", "outcome", "home_team", "away_team"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Backtest frame is missing columns: {', '.join(missing)}")
    missing_features = sorted(set(feature_cols).difference(frame.columns))
    if missing_features:
        raise ValueError(f"Backtest frame is missing features: {', '.join(missing_features)}")
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValueError(f"Backtest features must be numeric: {', '.join(non_numeric)}")


def _best_weight(history: pd.DataFrame, step: float) -> float:
    weights = np.arange(0.0, 1.0 + step / 2, step)
    actual = history["actual"].to_numpy()
    model = history.loc[:, list(PROBABILITY_COLUMNS)].to_numpy()
    market = history.loc[:, [f"market_{n}" for n in PROBABILITY_COLUMNS]].to_numpy()
    losses = [competition_log_loss(actual, (1 - w) * model + w * market) for w in weights]
    return float(weights[int(np.argmin(losses))])


def _loss(frame: pd.DataFrame, prefix: str) -> float | None:
    if frame.empty:
        return None
    probabilities = frame.loc[:, [f"{prefix}{n}" for n in PROBABILITY_COLUMNS]].to_numpy()
    return competition_log_loss(frame["actual"].to_numpy(), probabilities)


def _data_fingerprint(
    matches: pd.DataFrame,
    source_path: str | Path | None = None,
) -> dict[str, object]:
    latest_played = matches.loc[matches["outcome"].notna(), "date"].max()
    fingerprint: dict[str, object] = {
        "path": str(source_path) if source_path is not None else "<provided DataFrame>",
        "rows": int(len(matches)),
        "latest_played": latest_played.isoformat(),
    }
    if source_path is not None and Path(source_path).exists():
        digest = hashlib.sha256()
        with Path(source_path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    else:
        digest = hashlib.sha256()
        digest.update("\0".join(map(str, matches.columns)).encode())
        digest.update(pd.util.hash_pandas_object(matches, index=True).values.tobytes())
    fingerprint["sha256"] = digest.hexdigest()
    return fingerprint


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward football backtest")
    parser.add_argument("--model", choices=list(MODELS), default="logistic")
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="base")
    parser.add_argument("--dataset", choices=list(EVAL_DATASETS), default="full_2018")
    parser.add_argument("--odds-csv", default=None, help="Override the odds snapshot CSV")
    parser.add_argument("--compare-market", action="store_true")
    parser.add_argument("--blend-min-history", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--tag", action="append", default=[], metavar="KEY=VALUE",
        help="Run tag for organizing iterations; repeat for multiple",
    )
    parser.add_argument("--note", default=None, help="Free-text run description")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_tags = parse_key_value_tags(args.tag)

    result = evaluate(
        args.model,
        feature_set=args.features,
        dataset=args.dataset,
        seed=args.seed,
        odds_csv=args.odds_csv,
        compare_market=args.compare_market,
        blend_min_history=args.blend_min_history,
        refresh=args.refresh,
    )
    summary = result.summary
    output_dir = args.output_dir or (
        Path("artifacts") / "backtests" / f"{args.dataset}__{args.model}__{args.features.replace('+', '_')}"
    )
    path = write_artifacts(result, output_dir)

    mlflow_run_id = None
    if not args.no_mlflow:
        from tracking import log_backtest

        mlflow_run_id, _ = log_backtest(
            result=result,
            artifact_dir=path,
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            tags=run_tags,
            note=args.note,
        )

    print(
        f"\n{args.model} / {args.features} / {args.dataset}  "
        f"({summary['folds']} folds, {summary['matches']} matches)"
    )
    print(f"  LOG-LOSS {summary['log_loss']:.4f}   (the number to minimize)")
    print(
        f"  accuracy {summary['accuracy']:.1%} | brier {summary['multiclass_brier']:.3f} | "
        f"uniform log-loss {summary['uniform_log_loss']:.3f}"
    )
    print(f"  artifacts: {path}")
    if mlflow_run_id:
        print(f"  mlflow run: {mlflow_run_id}")
    if result.comparison_predictions is not None:
        m = summary["market_comparison"]
        if m["matched_matches"]:
            print(
                f"  market rows {m['matched_matches']} ({m['coverage']:.1%}) | "
                f"model {m['model_log_loss_matched']:.3f} | market {m['market_log_loss']:.3f} | "
                f"blend {m['blend_log_loss']:.3f}"
            )
        else:
            print("  no out-of-fold rows had matching market odds")


if __name__ == "__main__":
    main()
