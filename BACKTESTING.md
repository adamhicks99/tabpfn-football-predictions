# Backtesting and experiment tracking

The unit of evaluation is one named model, one named feature set, and one
certified dataset:

```python
from backtest import evaluate

result = evaluate("logistic", feature_set="base", dataset="full_2018")
print(result.summary["log_loss"])
```

The CLI exposes the same contract:

```bash
python backtest.py \
  --model logistic \
  --features base \
  --dataset full_2018 \
  --run-name logistic-baseline
```

`log_loss` is the primary metric and lower is better. Accuracy, multiclass
Brier score, calibration, and mean probability assigned to the actual outcome
are reported as diagnostics.

## Evaluation protocol

Each dataset in `datasets.py` fixes the evaluation start/end dates, training
window, test-window size, step size, row limits, and optional tournament
filter. For every calendar fold the backtester:

1. trains a fresh model using only matches before the test window;
2. predicts every eligible match in the test window;
3. records probabilities in the fixed order `home_win`, `draw`, `away_win`;
4. aggregates only out-of-fold predictions.

Test windows cannot overlap. Features are built in chronological order, and
state such as Elo, form, rest, and head-to-head is updated only after a played
match.

The built-in evaluation datasets are:

- `full_2018`: all internationals from 2018 through the latest result;
- `recent_3y`: all internationals from June 2023 onward;
- `tournaments_2022`: selected major tournaments from November 2022 through
  June 2025.

Scores are comparable only when the evaluation dataset and input-data
fingerprint match.

## Live prediction fixtures

Historical results and future prediction fixtures are deliberately separate.
Blank-score rows in `results.csv` are not trusted as a live schedule. Supply a
fixture manifest explicitly:

```bash
python predict.py \
  --model tabpfn \
  --features base \
  --fixtures round_of_32_fixtures.csv \
  --output round_of_32_predictions.csv
```

The fixture file must contain `date`, `home_team`, and `away_team`. Optional
`match_number`, `tournament`, and `status` columns are retained for auditability.
When `status` is present, every row must be `confirmed`.

The predictor rejects:

- fixtures before the current date;
- duplicate or same-team matchups;
- unresolved names such as `TBD`, `Group L winner`, or `third place`;
- team names that do not exactly match the historical dataset.

It then drops every unplayed row inherited from the results source and appends
only the validated manifest. This prevents stale group-stage fixtures from
leaking into a knockout-round output.

Use `--expected-fixtures 16` for the final Round of 32 export so an incomplete
manifest fails before model training.

## Artifacts and MLflow

Every CLI run writes:

- `predictions.csv`: out-of-fold probabilities and actual outcomes;
- `fold_metrics.csv`: metrics for each calendar fold;
- `calibration.csv`: one-vs-rest reliability bins;
- `summary.json`: configuration, feature columns, data fingerprint, and
  aggregate metrics;
- `market_comparison_predictions.csv`: model, market, and blended
  probabilities when market comparison is enabled.

The default location is:

```text
artifacts/backtests/<dataset>__<model>__<feature-set>/
```

Unless `--no-mlflow` is supplied, the same run is logged to the
`football-backtests` experiment in `sqlite:///mlflow.db`. The parent run holds
aggregate metrics and artifacts; nested runs hold per-fold metrics. It also
records the Git commit, dirty status, tracked-file diff summary, requirements
file, feature manifest, and data SHA-256. Patch contents are deliberately not
uploaded because uncommitted changes can contain credentials.

Start the local dashboard with:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Then open <http://127.0.0.1:5000>.

Use a short run name, an optional note, and stable tag keys to organize
iterations:

```bash
python backtest.py \
  --model logistic \
  --features base \
  --dataset recent_3y \
  --run-name rest-feature-baseline \
  --note "Baseline before changing rest-day handling" \
  --tag hypothesis=rest-days \
  --tag status=baseline
```

Model, feature-set, and dataset provenance tags are authoritative; custom tags
with those names cannot overwrite them.

## Models, feature sets, and datasets

The configuration is deliberately plain Python dictionaries rather than a
plugin framework.

Models are factories in `models.py`:

```python
def _xgb(random_state=42):
    from xgboost import XGBClassifier

    return XGBClassifier(random_state=random_state)


MODELS["xgb"] = _xgb
```

A model must expose `fit`, `predict_proba`, and fitted `classes_` containing
`home_win`, `draw`, and `away_win`.

Feature sets are builders in `features.py`:

```python
def _build_mine(matches, odds_csv=None):
    frame, columns = _build_base(matches)
    frame["my_feature"] = 0.0
    return frame, columns + ["my_feature"]


FEATURE_SETS["mine"] = _build_mine
```

Every returned feature must be numeric and aligned to the match-frame index.
The built-in choices are `base` and `base+odds`.

Certified datasets are `EvalDataset` entries in `datasets.py`. Add a new entry
when the evaluation protocol itself changes; do not silently move an existing
dataset's boundaries and compare the new score with old runs.

## Point-in-time odds

Market data is eligible only when its `observed_at` timestamp is at or before
the match's `submission_cutoff`. `features.py` selects the latest eligible
snapshot and leaves unmatched markets missing. It never backfills from a later
snapshot.

Credential locations are supplied at runtime; no environment-variable,
Keychain service, Keychain account, or secret value is hard-coded. The key is
not written to raw responses, logs, or MLflow.

To use an environment variable:

```bash
read -s THE_ODDS_API_KEY
export THE_ODDS_API_KEY
python odds.py --api-key-env THE_ODDS_API_KEY discover
```

Or reference an existing macOS Keychain item:

```bash
python odds.py \
  --keychain-service YOUR_SERVICE \
  --keychain-account YOUR_ACCOUNT \
  discover
```

Inspect the configured archive and conservative credit estimate:

```bash
python odds.py plan
```

Discover exact event IDs:

```bash
python odds.py --api-key-env THE_ODDS_API_KEY discover
```

Historical odds fetching is a dry run unless both `--execute` and an explicit
credit ceiling are provided:

```bash
python odds.py --api-key-env THE_ODDS_API_KEY fetch-historical

python odds.py \
  --api-key-env THE_ODDS_API_KEY \
  fetch-historical \
  --execute \
  --max-credits 10860
```

Responses are cached immediately under `data/odds/raw/` and never overwritten.
An interrupted fetch can therefore resume without repeating cached paid
requests. All generated odds data is ignored by Git.

Audit the derived feature file before modeling:

```bash
python odds.py audit
```

The audit rejects malformed probabilities, duplicate match rows, and snapshots
observed after the submission cutoff. Three-way decimal prices are converted
to implied probabilities, de-vigged per bookmaker, and combined with a median
consensus.

Use odds as model inputs:

```bash
python backtest.py \
  --model logistic \
  --features base+odds \
  --dataset tournaments_2022 \
  --odds-csv data/odds/features.csv
```

Or compare a base model with the raw market and a past-only blend on identical
out-of-fold rows:

```bash
python backtest.py \
  --model logistic \
  --features base \
  --dataset tournaments_2022 \
  --odds-csv data/odds/features.csv \
  --compare-market
```

For each fold, the blend weight is selected using matched predictions from
completed earlier folds only. Until `--blend-min-history` matched rows exist,
the model receives weight 1 and the market receives weight 0.

## Verification

Run the offline regression suite and a full local smoke test:

```bash
python -m unittest discover -s tests -v

python backtest.py \
  --model logistic \
  --features base \
  --dataset recent_3y \
  --output-dir /tmp/football-backtest-smoke \
  --no-mlflow
```
