# Architecture

## Data model

Football Lab has three immutable dataset kinds.

| Kind | Purpose | Required contents |
|---|---|---|
| `training` | Fit a model | Match identity, numeric features, outcome |
| `evaluation` | Compare models on a fixed holdout | Same schema as training |
| `result` | Upload future-round predictions to Prior | Match identity and three probabilities |

Dataset references use `kind/name@version`. Versions are twelve-character
content identifiers derived from:

- canonical CSV bytes;
- dataset kind and name;
- source checksum and build boundaries;
- feature and schema metadata.

Rebuilding the same dataset is idempotent. Any input or recipe change produces
a different version.

## Storage

The workspace is self-contained:

```text
workspace/
├── catalog.sqlite3
├── datasets/
│   ├── training/<name>/<version>/
│   │   ├── data.csv
│   │   └── manifest.json
│   ├── evaluation/<name>/<version>/
│   └── result/<name>/<version>/
└── experiments/<experiment-id>/
    └── predictions.csv
```

Dataset directories are written atomically and never modified after creation.
Manifests record row count, column types, source checksum, build parameters,
content checksum, and creation time.

## Experiment catalog

`catalog.sqlite3` is the system of record.

- `dataset_versions` indexes immutable dataset artifacts and lineage.
- `experiments` records model, parameters, exact dataset IDs, code commit,
  tags, status, timing, and failure details.
- `metrics` stores normalized numeric metrics by run and split.
- `experiment_artifacts` links prediction files and result datasets to runs.

Experiment status transitions are `running` to either `succeeded` or `failed`.
Failures remain queryable. Catalog queries exposed through the CLI are
read-only.

## Evaluation

An evaluation run enforces:

- exact feature-schema equality between training and evaluation datasets;
- training dates strictly before evaluation dates;
- a fresh model instance for every run;
- outcome probabilities ordered as home win, draw, away win;
- immutable out-of-sample predictions and aggregate metrics.

Evaluation predictions are experiment artifacts. They are not competition
results.

## Competition results

A result run takes an immutable training dataset, a historical match snapshot,
an explicit confirmed-fixture file, and an `as_of` cutoff.

Only historical matches before the cutoff contribute feature state. The
training dataset must also end before the cutoff. Fixture rows never update
feature state.

The output contract is:

```text
date
home_team
away_team
p_home_win
p_draw
p_away_win
```

Every probability must be finite and between zero and one. Each row must sum to
one. The result manifest records model, seed, training dataset ID, source
checksums, cutoff, feature schema, and Git provenance.
