# World Cup outcome predictions — TabPFN × market blend

Predict home/draw/away probabilities for World Cup knockout games and minimize the
competition **log-loss**. The production strategy is a **weighted blend of TabPFN and the
de-vigged betting market**:

```
submission = w · TabPFN(odds)  +  (1 − w) · market        # w = 0.15 (holdout-validated)
```

Any `w > 0` uses TabPFN, so the submission is eligible; the sharper market carries most of
the weight. On the holdout this beats both the raw market and pure TabPFN (see `artifacts/`).
Training is restricted to matches that have betting odds.

## Files

| File | Role |
|---|---|
| `data.py` | Load `results.csv`, join odds, `odds_covered` filter |
| `features.py` | `build_features` (Elo/form/rest/h2h) + `FEATURE_SETS` (`base`, `odds`, `base+odds`) |
| `models.py` | `MODELS` (`logistic`, `tabpfn`), `ordered_probabilities`, `log_loss` |
| `odds.py` | The Odds API tool — historical fetch + live `upcoming` feed (the only paid part) |
| `evaluate.py` | Holdout validation: score a model, or `--blend` to sweep blend weights |
| `predict.py` | **Produce the submission** (the blend) for upcoming fixtures |

## Make a submission

```bash
# 1. refresh the live bracket + odds (paid; ~1 credit)
python odds.py --keychain-service prior-labs-football-the-odds-api \
  --keychain-account adamhicks upcoming --execute

# 2. write submission.csv (the blend)
python predict.py
```

`predict.py` writes `submission.csv` in the Prior schema
(`date,home_team,away_team,p_home_win,p_draw,p_away_win`) for the odds-feed games that have
**not yet kicked off**. Override the blend with `--weight`, the model with `--model`, the
features with `--features`.

## Validate / tune

```bash
python evaluate.py --blend                       # sweep blend weights -> best w
python evaluate.py --model tabpfn --features odds # score one config vs the market
```

`evaluate.py` logs single-model runs to `experiments.csv` and prints log-loss vs the market.
Re-run `--blend` after the data changes and update `DEFAULT_BLEND_WEIGHT` in `predict.py`.

## Betting odds (`odds.py`)

The only component that spends money (The Odds API). Key is read via
`--api-key-env NAME` or `--keychain-service/--keychain-account` (global flags, before the
subcommand). Raw responses cache under `data/odds/raw/`, so re-runs are free; only new
requests cost credits.

```bash
python odds.py upcoming                          # dry run (no spend): show the planned call
python odds.py discover                          # find more historical event IDs (quota-light)
python odds.py fetch-historical --execute --max-credits N   # grow the training set (gated)
```

## Develop

```bash
python -m unittest discover -s tests -v
```

`artifacts/round_of_32/` holds the methodology and the validated submission for the record.
