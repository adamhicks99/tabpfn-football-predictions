# TabPFN Football Predictions

This repository is a template to participate in Prior Labs' [World Cup Game Outcome Prediction competition](https://ux.priorlabs.ai/worldcup). It has a basic script that outputs predictions with a standard prediction template. Use this template to generate predictions. The `predict.py` script should only be a source of inspiration, feel free to fork the repo and add your own ideas.

The script predicts international football match outcomes using [TabPFN](https://github.com/PriorLabs/TabPFN) using the [client repository](https://github.com/PriorLabs/tabpfn-client). It achieves ~59% accuracy and ~0.86 log-loss on held-out data. There is a good margin of progression. We look forward to your submission!

The model is trained on engineered features: ELO ratings, recent form, head-to-head record, rest days, and tournament importance. Data comes from [martj42/international_results](https://github.com/martj42/international_results).

This fork also includes named model/feature/dataset configurations, calendar
walk-forward backtesting, point-in-time odds features, and MLflow experiment
tracking. See [BACKTESTING.md](BACKTESTING.md).

## Setup

```bash
git clone https://github.com/eliott-kalfon/tabpfn-football-predictions.git
cd tabpfn-football-predictions
pip install -r requirements.txt
```

## Run

```bash
python predict.py \
  --fixtures round_of_32_fixtures.csv \
  --output round_of_32_predictions.csv
```

This will:

1. Download the full international results dataset (~47 000 matches) on first run
2. Validate the explicit fixture manifest and reject past, duplicate, unknown,
   or placeholder matchups
3. Discard stale unplayed rows from the historical results source
4. Build features with a single chronological pass (no leakage)
5. Run a quick backtest on the previous calendar month and print accuracy + log-loss
6. Train on up to 10 000 recent matches and predict only the supplied fixtures
7. Save competition-format probabilities and print them to the console

To refresh the dataset from source before predicting:

```bash
python predict.py \
  --refresh \
  --fixtures round_of_32_fixtures.csv
```

Run a tracked local baseline:

```bash
python backtest.py --model logistic --run-name logistic-baseline
```

This evaluates `logistic / base / full_2018`, writes reproducibility artifacts
under `artifacts/backtests/`, and records the run in the local MLflow database.
Use `--no-mlflow` for an untracked smoke test.

The prediction CLI uses TabPFN and the base feature set by default. Its model,
feature set, odds source, and output path are configurable:

```bash
python predict.py \
  --model logistic \
  --features base \
  --fixtures round_of_32_fixtures.csv \
  --output predictions.csv
```

For a final Round of 32 export, require all 16 confirmed fixtures:

```bash
python predict.py \
  --fixtures round_of_32_fixtures.csv \
  --expected-fixtures 16 \
  --output round_of_32_predictions.csv
```

The command fails rather than predicting unresolved bracket placeholders. The
fixture manifest should be updated from the
[official FIFA schedule](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums)
as the remaining pairings are confirmed.

## Output

```
Latest played match in dataset: 2026-06-25
Data freshness: 2 days 18:32:11

Quick check 2026-05 (87 matches): accuracy 59%, log-loss 0.861

9 fixture predictions -> round_of_32_predictions.csv

  2026-06-28        South Africa vs Canada                -> away_win   H  24% | D  28% | A  47%
  2026-06-29             Germany vs Paraguay              -> home_win   H  49% | D  29% | A  22%
  ...
```

## Features

| Feature | Description |
|---|---|
| `elo_diff` | ELO gap (home + home advantage - away) |
| `home_elo`, `away_elo` | Current ELO ratings |
| `form5_diff` | Difference in average points per game over last 5 matches |
| `form10_diff` | Same over last 10 matches |
| `home_winrate`, `away_winrate` | Win rate over last 10 matches |
| `home_gf5`, `away_gf5` | Goals scored per game over last 5 matches |
| `home_ga5`, `away_ga5` | Goals conceded per game over last 5 matches |
| `gd10_diff` | Difference in average goal difference over last 10 matches |
| `home_streak`, `away_streak` | Current win streak |
| `home_rest`, `away_rest` | Days since last match (capped at 90) |
| `home_played`, `away_played` | Total matches played in history |
| `h2h_n` | Number of head-to-head meetings |
| `h2h_home_winrate` | Home team win rate in head-to-head |
| `h2h_draw_rate` | Draw rate in head-to-head |
| `h2h_gd` | Average goal difference in head-to-head (from home team's perspective) |
| `neutral` | 1 if played at a neutral venue |
| `importance` | Tournament importance score (60 = World Cup, 20 = friendly) |

## Development

Run the offline test suite:

```bash
python -m unittest discover -s tests -v
```

The implementation is intentionally flat:

- `models.py` defines the model choices and competition scoring;
- `features.py` defines leakage-safe feature sets;
- `datasets.py` defines version-controlled evaluation protocols;
- `backtest.py` evaluates one model/feature-set/dataset combination;
- `odds.py` archives and audits historical market snapshots;
- `tracking.py` records completed evaluations in MLflow.

Before publishing or reporting a vulnerability, see
[SECURITY.md](SECURITY.md). Direct dependencies are pinned and the repository
does not contain default credential locations or raw secret values.
