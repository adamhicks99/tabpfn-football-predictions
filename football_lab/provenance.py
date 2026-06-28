from __future__ import annotations

import subprocess


def git_provenance() -> tuple[str | None, bool]:
    try:
        commit = _git("rev-parse", "HEAD").strip()
        dirty = bool(_git("status", "--short").strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, False


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout
