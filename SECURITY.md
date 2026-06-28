# Security

## Repository policy

Do not commit credentials, `.env` files, private keys, local model caches,
generated datasets, SQLite catalogs, or submission exports.

Football Lab does not accept API keys through its CLI and does not implement
credential storage. Authentication required by a model client is delegated to
that client.

The workspace and common credential formats are excluded by `.gitignore`.
Experiment provenance records the Git commit and dirty status, never patch
contents or environment values.

## Release checks

Before publishing:

```bash
.venv/bin/ruff check .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/bandit -r football_lab -ll
.venv/bin/pip-audit \
  --path .venv/lib/python3.12/site-packages \
  --progress-spinner off
git ls-files -z | xargs -0 .venv/bin/detect-secrets-hook
```

Run these commands after `uv sync --extra dev --locked --no-editable`.
Dependency versions are recorded in `uv.lock`, and Dependabot checks them
weekly.

If a secret is committed, rotate it immediately and remove it from repository
history before publishing. Deleting it in a later commit is insufficient.

## Reporting

Use GitHub private vulnerability reporting. Do not include credentials or
working exploits in a public issue.
