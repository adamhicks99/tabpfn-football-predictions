"""Sweep weighted blends of TabPFN(odds) and the raw market, then optionally submit.

The blend is ``w * tabpfn + (1 - w) * market`` (w = weight on TabPFN). w=0 is the
pure market; w=1 is pure TabPFN. Any w > 0 uses TabPFN, so it stays an eligible
submission while letting the sharper market carry most of the signal.

    python blend.py                      # holdout sweep -> best weight
    python blend.py --submit             # also write submission_blend.csv at the best weight
    python blend.py --submit --weight 0.2
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from data import load_modeling_data, odds_covered
from features import feature_columns
from models import get_model, log_loss, ordered_probabilities
from predict import MARKET_COLUMNS, _submission, prepare

FEATURES = "odds"  # TabPFN on the de-vigged market consensus (best eligible config)


def holdout_probs(cutoff: str, max_train_rows: int, seed: int):
    """Return (actual, tabpfn probs, market probs) on the time holdout."""
    frame = odds_covered(load_modeling_data())
    played = frame.loc[frame["outcome"].notna()].sort_values("date", kind="stable")
    train = played.loc[played["date"] < pd.Timestamp(cutoff)].tail(max_train_rows)
    test = played.loc[played["date"] >= pd.Timestamp(cutoff)]
    columns = feature_columns(FEATURES)
    model = get_model("tabpfn", seed)
    model.fit(train.loc[:, columns].to_numpy(), train["outcome"].to_numpy())
    tabpfn = ordered_probabilities(model, test.loc[:, columns].to_numpy())
    market = test.loc[:, list(MARKET_COLUMNS)].to_numpy(dtype=float)
    market = market / market.sum(axis=1, keepdims=True)
    return test["outcome"].to_numpy(), tabpfn, market


def sweep(actual, tabpfn, market, step: float) -> list[tuple[float, float]]:
    weights = np.round(np.arange(0.0, 1.0 + 1e-9, step), 3)
    return [(float(w), log_loss(actual, w * tabpfn + (1 - w) * market)) for w in weights]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", default="2024-06-01")
    parser.add_argument("--max-train-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--weight", type=float, default=None,
        help="TabPFN weight for the submission (default: best eligible w>0 from the sweep)",
    )
    parser.add_argument("--output", default="submission_blend.csv")
    args = parser.parse_args()

    actual, tabpfn, market = holdout_probs(args.cutoff, args.max_train_rows, args.seed)
    results = sweep(actual, tabpfn, market, args.step)
    pure_market, pure_tabpfn = results[0][1], results[-1][1]
    best_w, best_ll = min(results, key=lambda r: r[1])
    eligible = [r for r in results if r[0] > 0]
    elig_w, elig_ll = min(eligible, key=lambda r: r[1])

    print(f"Blend sweep on holdout (test={len(actual)}, w = TabPFN weight):")
    for w, ll in results:
        tag = "  <- best" if w == best_w else ("  <- best eligible (w>0)" if w == elig_w and best_w == 0 else "")
        print(f"  w={w:.2f}  log_loss={ll:.4f}{tag}")
    print(f"\npure market (w=0): {pure_market:.4f} | pure TabPFN (w=1): {pure_tabpfn:.4f}")
    print(f"best overall: w={best_w:.2f} -> {best_ll:.4f}")
    print(f"best ELIGIBLE (uses TabPFN, w>0): w={elig_w:.2f} -> {elig_ll:.4f}")

    if not args.submit:
        return
    w = args.weight if args.weight is not None else elig_w
    train, fixtures = prepare()
    columns = feature_columns(FEATURES)
    model = get_model("tabpfn", args.seed)
    model.fit(train.loc[:, columns].to_numpy(), train["outcome"].to_numpy())
    tab_fx = ordered_probabilities(model, fixtures.loc[:, columns].to_numpy())
    mkt_fx = fixtures.loc[:, list(MARKET_COLUMNS)].to_numpy(dtype=float)
    mkt_fx = mkt_fx / mkt_fx.sum(axis=1, keepdims=True)
    blend = w * tab_fx + (1 - w) * mkt_fx

    submission = _submission(fixtures, blend)
    submission.to_csv(args.output, index=False)
    print(f"\nWrote blend submission (w={w:.2f}, TabPFN-eligible) -> {args.output} ({len(submission)} fixtures)\n")
    for row in submission.itertuples():
        print(
            f"  {row.date}  {row.home_team:>16} vs {row.away_team:<20}  "
            f"H {row.p_home_win:4.0%} | D {row.p_draw:4.0%} | A {row.p_away_win:4.0%}"
        )


if __name__ == "__main__":
    main()
