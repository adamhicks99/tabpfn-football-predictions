# Football Lab

Football Lab builds immutable football datasets, evaluates models against fixed
holdouts, and produces versioned result files for Prior's World Cup prediction
competition.

The repository has four responsibilities:

1. build reusable training datasets;
2. build reusable evaluation datasets;
3. create competition result datasets for future World Cup rounds;
4. record every model run, metric, dataset version, and artifact in SQLite.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev --locked --no-editable
```

All generated datasets and experiment records are stored under `workspace/`,
which is excluded from Git.

## Build datasets

Training and evaluation datasets are immutable. Their versions are derived
from the data content, source hash, date boundaries, and feature schema.

```bash
football-lab dataset build-training \
  --name historical-baseline \
  --source results.csv \
  --start 2014-01-01 \
  --end 2025-01-01 \
  --max-rows 10000

football-lab dataset build-evaluation \
  --name holdout-2025 \
  --source results.csv \
  --start 2025-01-01 \
  --end 2026-01-01
```

Each command returns an exact reference such as:

```text
training/historical-baseline@a1b2c3d4e5f6
evaluation/holdout-2025@b2c3d4e5f6a1
```

`@latest` is accepted for interactive use. Experiment records always retain the
resolved immutable version.

## Evaluate a model

```bash
football-lab experiment run \
  --model logistic \
  --training training/historical-baseline@a1b2c3d4e5f6 \
  --evaluation evaluation/holdout-2025@b2c3d4e5f6a1 \
  --tag hypothesis=baseline \
  --note "Initial fixed-dataset baseline"
```

The run fits only the referenced training dataset. It writes out-of-sample
predictions, records log loss, accuracy, and multiclass Brier score, and links
all artifacts to the exact model and dataset versions.

## Produce a Prior result dataset

Create a fixture CSV for the confirmed matches in the next World Cup round:

```csv
date,home_team,away_team,tournament,status
2026-06-28,South Africa,Canada,FIFA World Cup,confirmed
```

Build a versioned result:

```bash
football-lab result build \
  --name world-cup-round-32 \
  --model tabpfn \
  --training training/world-cup@a1b2c3d4e5f6 \
  --history results.csv \
  --fixtures fixtures/round-32.csv \
  --as-of 2026-06-28
```

The result dataset contains exactly the Prior upload schema:

```text
date,home_team,away_team,p_home_win,p_draw,p_away_win
```

Export a stable upload file:

```bash
football-lab result export \
  result/world-cup-round-32@c3d4e5f6a1b2 \
  --output submissions/world-cup-round-32.csv
```

Unconfirmed placeholders, unknown teams, duplicate matches, train/cutoff
leakage, and invalid probability rows fail before a result is registered.

## Query experiments

List recent runs:

```bash
football-lab experiment list
```

Run a read-only SQL query:

```bash
football-lab experiment query "
SELECT
    e.model_name,
    e.training_dataset_id,
    e.evaluation_dataset_id,
    m.value AS log_loss
FROM experiments e
JOIN metrics m ON m.experiment_id = e.id
WHERE e.status = 'succeeded'
  AND m.name = 'log_loss'
ORDER BY m.value ASC
"
```

The catalog is a standard SQLite database at `workspace/catalog.sqlite3`.

## Development

```bash
./.venv/bin/ruff check .
./.venv/bin/python -m unittest discover -s tests -v
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for data contracts and storage
invariants, and [SECURITY.md](SECURITY.md) for release requirements.
