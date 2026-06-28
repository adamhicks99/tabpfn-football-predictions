# World Cup win/draw/loss predictions

A small project to predict World Cup match outcome probabilities (home win / draw /
away win) and lower the competition **log-loss**. Betting odds are the backbone:
**training is restricted to matches that have odds**, and predictions need odds for
the fixtures being predicted.

## Files

| File | What it is |
|---|---|
| `data.py` | Load `results.csv`, join odds, filter to odds-covered matches |
| `features.py` | `build_features` (Elo/form/rest/h2h) + `FEATURE_SETS` (`base`, `odds`, `base+odds`) |
| `models.py` | `MODELS` dict (`logistic`, `tabpfn`) + log-loss |
| `evaluate.py` | Score a (model, feature set) on a holdout vs the market baseline |
| `predict.py` | Train and write the submission CSV for upcoming fixtures |
| `odds.py` | The Odds API tool (historical + upcoming). The only thing that costs money. |

## Iterate (the loop)

Try a feature set + model; the number to lower is **log-loss**. Every run also prints
the de-vigged market's log-loss (the baseline to beat) and appends to `experiments.csv`.

```bash
./.venv/bin/python evaluate.py --model tabpfn --features base+odds
./.venv/bin/python evaluate.py --model logistic --features base --note "no odds"
```

## Predict the next round

Odds are the backbone, so first fetch the upcoming fixtures' odds (paid), then predict:

```bash
./.venv/bin/python odds.py upcoming --execute          # fetch R32 odds (current)
./.venv/bin/python predict.py --model tabpfn --features base+odds
```

`predict.py` writes `submission.csv` (model) and `submission_market.csv` (market-only
fallback) in the Prior schema: `date,home_team,away_team,p_home_win,p_draw,p_away_win`.

## Betting odds (`odds.py`)

The Odds API key is read with `--api-key-env NAME` or `--keychain-service/--keychain-account`.
Raw responses cache under `data/odds/raw/`, so re-runs are free; only new requests cost credits.

```bash
./.venv/bin/python odds.py upcoming --dry-run          # show plan, no spend
./.venv/bin/python odds.py discover                    # find historical event IDs (quota-light)
./.venv/bin/python odds.py fetch-historical --execute --max-credits N
```

## Develop

```bash
./.venv/bin/python -m unittest discover -s tests -v
```
