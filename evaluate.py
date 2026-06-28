"""Score one (model, feature set) on the odds-covered holdout.

The number to lower is LOG-LOSS. Every run also prints the de-vigged market's
log-loss as the baseline to beat, and appends a row to experiments.csv.

    python evaluate.py --model tabpfn --features base+odds
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from data import load_modeling_data, odds_covered
from features import FEATURE_SETS, feature_columns
from models import MODELS, OUTCOMES, get_model, log_loss, ordered_probabilities

EXPERIMENTS_CSV = "experiments.csv"
MARKET_COLUMNS = ("odds_market_p_home", "odds_market_p_draw", "odds_market_p_away")


def evaluate(
    model: str = "tabpfn",
    features: str = "base+odds",
    cutoff: str = "2024-06-01",
    max_train_rows: int = 10_000,
    seed: int = 42,
    note: str = "",
) -> dict[str, object]:
    """Train on odds-covered matches before `cutoff`, score log-loss after it."""
    frame = odds_covered(load_modeling_data())
    played = frame.loc[frame["outcome"].notna()].sort_values("date", kind="stable")
    cutoff_ts = pd.Timestamp(cutoff)
    train = played.loc[played["date"] < cutoff_ts].tail(max_train_rows)
    test = played.loc[played["date"] >= cutoff_ts]
    if train.empty or test.empty:
        raise ValueError(
            f"Empty split at cutoff {cutoff}: train={len(train)}, test={len(test)} "
            f"(odds-covered played matches = {len(played)})"
        )

    columns = feature_columns(features)
    classifier = get_model(model, seed)
    classifier.fit(train.loc[:, columns].to_numpy(), train["outcome"].to_numpy())
    probabilities = ordered_probabilities(classifier, test.loc[:, columns].to_numpy())
    model_log_loss = log_loss(test["outcome"].to_numpy(), probabilities)
    predicted = np.asarray(OUTCOMES)[probabilities.argmax(axis=1)]
    accuracy = float(accuracy_score(test["outcome"], predicted))

    market = test.loc[:, list(MARKET_COLUMNS)].to_numpy(dtype=float)
    market = market / market.sum(axis=1, keepdims=True)
    market_log_loss = log_loss(test["outcome"].to_numpy(), market)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "features": features,
        "cutoff": cutoff,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "log_loss": round(model_log_loss, 4),
        "accuracy": round(accuracy, 4),
        "market_log_loss": round(market_log_loss, 4),
        "note": note,
    }
    _append_experiment(record)

    verdict = "BEATS market" if model_log_loss < market_log_loss else "worse than market"
    print(f"\n{model} / {features}  (train {len(train)}, test {len(test)} @ cutoff {cutoff})")
    print(f"  LOG-LOSS {model_log_loss:.4f}   [market baseline {market_log_loss:.4f} -> {verdict}]")
    print(f"  accuracy {accuracy:.1%}   logged -> {EXPERIMENTS_CSV}")
    return record


def _append_experiment(record: dict[str, object]) -> None:
    path = Path(EXPERIMENTS_CSV)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="tabpfn")
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="base+odds")
    parser.add_argument("--cutoff", default="2024-06-01", help="Train before / test on-or-after this date")
    parser.add_argument("--max-train-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    evaluate(args.model, args.features, args.cutoff, args.max_train_rows, args.seed, args.note)


if __name__ == "__main__":
    main()
