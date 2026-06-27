# Security

## Credentials

Never commit API keys, access tokens, Keychain identifiers, `.env` files,
private keys, or credential JSON files.

`odds.py` accepts only the *location* of an Odds API credential:

```bash
python odds.py --api-key-env YOUR_ENVIRONMENT_VARIABLE discover
```

or:

```bash
python odds.py \
  --keychain-service YOUR_SERVICE \
  --keychain-account YOUR_ACCOUNT \
  discover
```

The raw key is never accepted as a command-line argument because command-line
values can be exposed through shell history and process listings.

Generated odds responses, local aliases, MLflow databases, artifacts, and
common secret-file formats are excluded by `.gitignore`. MLflow records only a
Git diff summary, not patch contents.

If a secret is ever committed, removing it in a later commit is insufficient:
revoke or rotate it immediately and remove it from Git history before making
the repository public.

## Reporting a vulnerability

Report vulnerabilities privately through GitHub's private vulnerability
reporting feature when available. Do not open a public issue containing an
exploit or credential.
