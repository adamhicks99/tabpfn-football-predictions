"""Validate on a time holdout. Two modes:

  python evaluate.py                       # score one (model, feature set) vs the market
  python evaluate.py --model tabpfn --features base+odds
  python evaluate.py --blend               # sweep TabPFN(odds) x market blend weights

The number to minimize is LOG-LOSS. Single-model runs append to experiments.csv.
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


def _holdout(cutoff: str, max_train_rows: int):
    """Return (train, test) odds-covered played frames split at `cutoff`."""
    frame = odds_covered(load_modeling_data())
    played = frame.loc[frame["outcome"].notna()].sort_values("date", kind="stable")
    train = played.loc[played["date"] < pd.Timestamp(cutoff)].tail(max_train_rows)
    test = played.loc[played["date"] >= pd.Timestamp(cutoff)]
    if train.empty or test.empty:
        raise ValueError(f"Empty split at {cutoff}: train={len(train)}, test={len(test)}")
    return train, test


def _market(test: pd.DataFrame) -> np.ndarray:
    market = test.loc[:, list(MARKET_COLUMNS)].to_numpy(dtype=float)
    return market / market.sum(axis=1, keepdims=True)


def evaluate(model="tabpfn", features="base+odds", cutoff="2024-06-01",
             max_train_rows=10_000, seed=42, note="") -> dict[str, object]:
    """Score one (model, feature set) vs the market baseline; log to experiments.csv."""
    train, test = _holdout(cutoff, max_train_rows)
    columns = feature_columns(features)
    classifier = get_model(model, seed)
    classifier.fit(train.loc[:, columns].to_numpy(), train["outcome"].to_numpy())
    probabilities = ordered_probabilities(classifier, test.loc[:, columns].to_numpy())
    model_log_loss = log_loss(test["outcome"].to_numpy(), probabilities)
    predicted = np.asarray(OUTCOMES)[probabilities.argmax(axis=1)]
    accuracy = float(accuracy_score(test["outcome"], predicted))
    market_log_loss = log_loss(test["outcome"].to_numpy(), _market(test))

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model, "features": features, "cutoff": cutoff,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "log_loss": round(model_log_loss, 4), "accuracy": round(accuracy, 4),
        "market_log_loss": round(market_log_loss, 4), "note": note,
    }
    _append_experiment(record)
    verdict = "BEATS market" if model_log_loss < market_log_loss else "worse than market"
    print(f"\n{model} / {features}  (train {len(train)}, test {len(test)} @ {cutoff})")
    print(f"  LOG-LOSS {model_log_loss:.4f}   [market {market_log_loss:.4f} -> {verdict}]")
    print(f"  accuracy {accuracy:.1%}   logged -> {EXPERIMENTS_CSV}")
    return record


def blend_sweep(cutoff="2024-06-01", max_train_rows=10_000, seed=42, step=0.05):
    """Sweep w in w*TabPFN(odds) + (1-w)*market on the holdout; report the best."""
    train, test = _holdout(cutoff, max_train_rows)
    columns = feature_columns("odds")
    classifier = get_model("tabpfn", seed)
    classifier.fit(train.loc[:, columns].to_numpy(), train["outcome"].to_numpy())
    tabpfn = ordered_probabilities(classifier, test.loc[:, columns].to_numpy())
    market = _market(test)
    actual = test["outcome"].to_numpy()

    weights = np.round(np.arange(0.0, 1.0 + 1e-9, step), 3)
    results = [(float(w), log_loss(actual, w * tabpfn + (1 - w) * market)) for w in weights]
    best_w, best_ll = min(results, key=lambda r: r[1])
    eligible_w, eligible_ll = min((r for r in results if r[0] > 0), key=lambda r: r[1])

    print(f"\nBlend sweep on holdout (test={len(actual)}, w = TabPFN weight):")
    for w, ll in results:
        print(f"  w={w:.2f}  log_loss={ll:.4f}" + ("  <- best" if w == best_w else ""))
    print(f"\npure market (w=0): {results[0][1]:.4f} | pure TabPFN (w=1): {results[-1][1]:.4f}")
    print(f"best ELIGIBLE (w>0): w={eligible_w:.2f} -> {eligible_ll:.4f}  "
          "(use this as predict.py --weight)")
    return eligible_w, eligible_ll


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
    parser.add_argument("--blend", action="store_true", help="Sweep TabPFN(odds) x market blend weights")
    parser.add_argument("--model", choices=list(MODELS), default="tabpfn")
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="base+odds")
    parser.add_argument("--cutoff", default="2024-06-01")
    parser.add_argument("--max-train-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=float, default=0.05, help="Blend sweep step")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    if args.blend:
        blend_sweep(args.cutoff, args.max_train_rows, args.seed, args.step)
    else:
        evaluate(args.model, args.features, args.cutoff, args.max_train_rows, args.seed, args.note)


if __name__ == "__main__":
    main()
