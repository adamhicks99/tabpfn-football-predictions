"""MLflow tracking: log each evaluation so it traces back to a certified dataset.

Every run records its model, feature set, eval dataset, and data SHA as tags, and
log-loss as the primary metric -- so scores are only ever compared within the same
(eval_dataset, data version).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mlflow
import pandas as pd

from backtest import BacktestResult, parse_key_value_tags  # re-exported for callers

__all__ = ["log_backtest", "parse_key_value_tags", "DEFAULT_EXPERIMENT", "DEFAULT_TRACKING_URI"]

DEFAULT_EXPERIMENT = "football-backtests"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
PRIMARY_METRIC = "log_loss"


def log_backtest(
    result: BacktestResult,
    artifact_dir: str | Path,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    note: str | None = None,
) -> tuple[str, str]:
    """Log one parent run (+ a child run per fold) with full traceability."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as parent:
        if tags:
            mlflow.set_tags(tags)
        _log_parent(result)
        _log_source_snapshot()
        if note:
            mlflow.set_tag("mlflow.note.content", note)
        mlflow.log_artifacts(str(artifact_dir), artifact_path="backtest")
        for step, fold in result.fold_metrics.iterrows():
            _log_fold(fold, step=int(step))
        return parent.info.run_id, parent.info.experiment_id


def _log_parent(result: BacktestResult) -> None:
    s = result.summary
    spec = s.get("dataset_spec", {})
    window = s.get("window", {})
    data = s.get("data", {})
    mlflow.log_params(
        {
            "model": s["model"],
            "feature_set": s.get("feature_set", "unknown"),
            "eval_dataset": s.get("eval_dataset", "unknown"),
            "seed": s.get("seed", "unknown"),
            "start": window.get("start"),
            "end": window.get("end"),
            "train_start": window.get("train_start"),
            "test_months": spec.get("test_months"),
            "step_months": spec.get("step_months"),
            "train_months": spec.get("train_months") or "expanding",
            "max_train_rows": spec.get("max_train_rows") or "unlimited",
            "min_train_rows": spec.get("min_train_rows"),
            "tournament_regex": spec.get("tournament_regex") or "all",
            "feature_count": s["features"],
            "fold_count": s["folds"],
            "data_rows": data.get("rows"),
            "data_latest_played": data.get("latest_played"),
            "data_sha256": data.get("sha256", "unknown"),
        }
    )
    mlflow.log_metrics(
        {
            "log_loss": float(s["log_loss"]),
            "accuracy": float(s["accuracy"]),
            "uniform_log_loss": float(s["uniform_log_loss"]),
            "log_loss_improvement_vs_uniform": float(s["log_loss_improvement_vs_uniform"]),
            "multiclass_brier": float(s["multiclass_brier"]),
            "mean_probability_on_actual": float(s["mean_probability_on_actual"]),
            "missing_feature_rate": float(s["missing_feature_rate"]),
            "matches": float(s["matches"]),
        }
    )
    _log_market_comparison(s.get("market_comparison"))
    mlflow.set_tags(
        {
            "run_type": "walk_forward_backtest",
            "primary_metric": PRIMARY_METRIC,
            "model": s["model"],
            "feature_set": s.get("feature_set", "unknown"),
            "eval_dataset": s.get("eval_dataset", "unknown"),
            "dataset_sha": str(data.get("sha256", "unknown"))[:12],
            "compared_to_market": str("market_comparison" in s).lower(),
        }
    )
    for step, fold in result.fold_metrics.iterrows():
        for metric in ("log_loss", "accuracy", "multiclass_brier"):
            mlflow.log_metric(f"fold_{metric}", float(fold[metric]), step=int(step))


def _log_market_comparison(comparison: dict[str, object] | None) -> None:
    if not comparison:
        return
    metrics = {
        f"comparison_{name}": float(value)
        for name, value in comparison.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if metrics:
        mlflow.log_metrics(metrics)


def _log_fold(fold: pd.Series, step: int) -> None:
    with mlflow.start_run(run_name=str(fold["fold"]), nested=True):
        mlflow.set_tag("run_type", "walk_forward_fold")
        mlflow.log_params(
            {
                "fold": fold["fold"],
                "test_start": _date_string(fold["test_start"]),
                "test_end": _date_string(fold["test_end"]),
                "train_rows": int(fold["train_rows"]),
            }
        )
        mlflow.log_metrics(
            {
                "log_loss": float(fold["log_loss"]),
                "accuracy": float(fold["accuracy"]),
                "multiclass_brier": float(fold["multiclass_brier"]),
                "mean_probability_on_actual": float(fold["mean_probability_on_actual"]),
                "missing_feature_rate": float(fold["missing_feature_rate"]),
                "matches": float(fold["matches"]),
                "fold_step": float(step),
            }
        )


def _date_string(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _log_source_snapshot() -> None:
    """Attach enough Git state to reproduce experiments between commits."""
    try:
        commit = _git("rev-parse", "HEAD").strip()
        status = _git("status", "--short")
        diff_stat = _git("diff", "--stat", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        mlflow.set_tag("source_snapshot", "unavailable")
        return

    mlflow.set_tags(
        {
            "mlflow.source.git.commit": commit,
            "git_dirty": str(bool(status.strip())).lower(),
        }
    )
    mlflow.log_text(status or "clean\n", "source/git-status.txt")
    mlflow.log_text(
        diff_stat or "No tracked-file changes.\n",
        "source/uncommitted-diff-stat.txt",
    )
    requirements = Path("requirements.txt")
    if requirements.exists():
        mlflow.log_artifact(str(requirements), artifact_path="source")


def _git(*arguments: str) -> str:
    completed = subprocess.run(("git", *arguments), check=True, capture_output=True, text=True)
    return completed.stdout
