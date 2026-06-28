from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


class DatasetKind(StrEnum):
    TRAINING = "training"
    EVALUATION = "evaluation"
    RESULT = "result"


@dataclass(frozen=True)
class DatasetRecord:
    id: str
    kind: DatasetKind
    name: str
    version: str
    path: Path
    content_sha256: str
    row_count: int
    columns: tuple[str, ...]
    metadata: dict[str, Any]
    created_at: str


class Catalog:
    def __init__(self, root: str | Path = "workspace") -> None:
        self.root = Path(root).resolve()
        self.database_path = self.root / "catalog.sqlite3"
        self.root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL CHECK (kind IN ('training', 'evaluation', 'result')),
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    row_count INTEGER NOT NULL CHECK (row_count >= 0),
                    columns_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (kind, name, version)
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL CHECK (run_type IN ('evaluation', 'competition')),
                    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
                    model_name TEXT NOT NULL,
                    model_params_json TEXT NOT NULL,
                    training_dataset_id TEXT NOT NULL REFERENCES dataset_versions(id),
                    evaluation_dataset_id TEXT REFERENCES dataset_versions(id),
                    result_dataset_id TEXT REFERENCES dataset_versions(id),
                    seed INTEGER NOT NULL,
                    tags_json TEXT NOT NULL,
                    note TEXT,
                    code_commit TEXT,
                    code_dirty INTEGER NOT NULL CHECK (code_dirty IN (0, 1)),
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                    split TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY (experiment_id, split, name)
                );

                CREATE TABLE IF NOT EXISTS experiment_artifacts (
                    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    relative_path TEXT,
                    dataset_id TEXT REFERENCES dataset_versions(id),
                    content_sha256 TEXT,
                    PRIMARY KEY (experiment_id, name)
                );

                CREATE INDEX IF NOT EXISTS idx_datasets_lookup
                    ON dataset_versions(kind, name, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_experiments_lookup
                    ON experiments(status, model_name, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_metrics_lookup
                    ON metrics(name, value);
                """
            )

    def store_dataset(
        self,
        *,
        kind: DatasetKind,
        name: str,
        frame: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> DatasetRecord:
        _validate_name(name)
        if frame.empty:
            raise ValueError("Dataset cannot be empty")
        if frame.columns.duplicated().any():
            raise ValueError("Dataset columns must be unique")

        payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        content_sha256 = hashlib.sha256(payload).hexdigest()
        stable_metadata = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        version = hashlib.sha256(
            f"{kind.value}\0{name}\0{content_sha256}\0{stable_metadata}".encode()
        ).hexdigest()[:12]
        dataset_id = f"{kind.value}/{name}@{version}"
        relative_path = Path("datasets") / kind.value / name / version
        target = self.root / relative_path
        created_at = _now()
        manifest = {
            "schema_version": 1,
            "id": dataset_id,
            "kind": kind.value,
            "name": name,
            "version": version,
            "content_sha256": content_sha256,
            "row_count": int(len(frame)),
            "columns": [
                {"name": column, "dtype": str(frame[column].dtype)}
                for column in frame.columns
            ],
            "metadata": metadata,
            "created_at": created_at,
        }

        if target.exists():
            existing = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            if existing["content_sha256"] != content_sha256:
                raise RuntimeError(f"Immutable dataset path has conflicting content: {target}")
            created_at = str(existing["created_at"])
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{version}.", dir=target.parent))
            try:
                (temporary / "data.csv").write_bytes(payload)
                (temporary / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                try:
                    os.replace(temporary, target)
                except OSError:
                    if not target.exists():
                        raise
                    existing = json.loads(
                        (target / "manifest.json").read_text(encoding="utf-8")
                    )
                    if existing["content_sha256"] != content_sha256:
                        raise RuntimeError(
                            f"Concurrent write produced conflicting content: {target}"
                        ) from None
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO dataset_versions (
                    id, kind, name, version, relative_path, content_sha256,
                    row_count, columns_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    kind.value,
                    name,
                    version,
                    str(relative_path),
                    content_sha256,
                    len(frame),
                    json.dumps(list(frame.columns)),
                    stable_metadata,
                    created_at,
                ),
            )
        return self.get_dataset(dataset_id)

    def get_dataset(
        self,
        reference: str,
        expected_kind: DatasetKind | None = None,
    ) -> DatasetRecord:
        kind, name, version = _parse_reference(reference)
        if expected_kind is not None and kind != expected_kind:
            raise ValueError(f"Expected a {expected_kind.value} dataset, got {kind.value}")
        with self._connect() as connection:
            if version == "latest":
                row = connection.execute(
                    """
                    SELECT * FROM dataset_versions
                    WHERE kind = ? AND name = ?
                    ORDER BY created_at DESC, version DESC
                    LIMIT 1
                    """,
                    (kind.value, name),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM dataset_versions WHERE id = ?",
                    (f"{kind.value}/{name}@{version}",),
                ).fetchone()
        if row is None:
            raise KeyError(f"Unknown dataset reference: {reference}")
        return self._dataset_record(row)

    def list_datasets(self, kind: DatasetKind | None = None) -> list[DatasetRecord]:
        sql = "SELECT * FROM dataset_versions"
        parameters: tuple[str, ...] = ()
        if kind is not None:
            sql += " WHERE kind = ?"
            parameters = (kind.value,)
        sql += " ORDER BY created_at DESC, kind, name"
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._dataset_record(row) for row in rows]

    def load_frame(self, record: DatasetRecord) -> pd.DataFrame:
        path = record.path / "data.csv"
        if not path.is_file():
            raise RuntimeError(f"Dataset artifact is missing: {record.id}")
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != record.content_sha256:
            raise RuntimeError(f"Dataset artifact failed integrity check: {record.id}")
        frame = pd.read_csv(path)
        if "date" in frame:
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        return frame

    def start_experiment(
        self,
        *,
        run_type: str,
        model_name: str,
        model_params: dict[str, Any],
        training_dataset_id: str,
        evaluation_dataset_id: str | None,
        seed: int,
        tags: dict[str, str],
        note: str | None,
        code_commit: str | None,
        code_dirty: bool,
    ) -> str:
        experiment_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experiments (
                    id, run_type, status, model_name, model_params_json,
                    training_dataset_id, evaluation_dataset_id, seed, tags_json,
                    note, code_commit, code_dirty, started_at
                ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    run_type,
                    model_name,
                    json.dumps(model_params, sort_keys=True),
                    training_dataset_id,
                    evaluation_dataset_id,
                    seed,
                    json.dumps(tags, sort_keys=True),
                    note,
                    code_commit,
                    int(code_dirty),
                    _now(),
                ),
            )
        return experiment_id

    def complete_experiment(
        self,
        experiment_id: str,
        *,
        metrics: dict[tuple[str, str], float] | None = None,
        artifacts: Iterable[tuple[str, str | None, str | None, str | None]] = (),
        result_dataset_id: str | None = None,
    ) -> None:
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE experiments
                SET status = 'succeeded', completed_at = ?, result_dataset_id = ?
                WHERE id = ? AND status = 'running'
                """,
                (_now(), result_dataset_id, experiment_id),
            )
            if updated.rowcount != 1:
                raise RuntimeError(
                    f"Experiment is missing or no longer running: {experiment_id}"
                )
            for (split, name), value in (metrics or {}).items():
                connection.execute(
                    """
                    INSERT INTO metrics (experiment_id, split, name, value)
                    VALUES (?, ?, ?, ?)
                    """,
                    (experiment_id, split, name, float(value)),
                )
            for name, relative_path, dataset_id, content_sha256 in artifacts:
                connection.execute(
                    """
                    INSERT INTO experiment_artifacts (
                        experiment_id, name, relative_path, dataset_id, content_sha256
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (experiment_id, name, relative_path, dataset_id, content_sha256),
                )

    def fail_experiment(self, experiment_id: str, error: str) -> None:
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE experiments
                SET status = 'failed', completed_at = ?, error = ?
                WHERE id = ? AND status = 'running'
                """,
                (_now(), error[:4000], experiment_id),
            )
            if updated.rowcount != 1:
                raise RuntimeError(
                    f"Experiment is missing or no longer running: {experiment_id}"
                )

    def list_experiments(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.query(
            """
            SELECT
                e.id,
                e.run_type,
                e.status,
                e.model_name,
                e.training_dataset_id,
                e.evaluation_dataset_id,
                e.result_dataset_id,
                e.started_at,
                MAX(CASE WHEN m.name = 'log_loss' THEN m.value END) AS log_loss,
                MAX(CASE WHEN m.name = 'accuracy' THEN m.value END) AS accuracy
            FROM experiments e
            LEFT JOIN metrics m ON m.experiment_id = e.id
            GROUP BY e.id
            ORDER BY e.started_at DESC
            LIMIT ?
            """,
            (limit,),
        )

    def query(
        self,
        sql: str,
        parameters: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        statement = sql.lstrip().upper()
        if not statement.startswith(("SELECT", "WITH", "EXPLAIN")):
            raise ValueError("Catalog queries must be read-only SELECT, WITH, or EXPLAIN statements")
        with self._connect() as connection:
            connection.execute("PRAGMA query_only = ON")
            try:
                rows = connection.execute(sql, parameters).fetchall()
            except sqlite3.DatabaseError as error:
                raise ValueError(f"Catalog query failed: {error}") from error
        return [dict(row) for row in rows]

    def experiment_path(self, experiment_id: str) -> Path:
        path = self.root / "experiments" / experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root))

    def _dataset_record(self, row: sqlite3.Row) -> DatasetRecord:
        return DatasetRecord(
            id=str(row["id"]),
            kind=DatasetKind(row["kind"]),
            name=str(row["name"]),
            version=str(row["version"]),
            path=self.root / row["relative_path"],
            content_sha256=str(row["content_sha256"]),
            row_count=int(row["row_count"]),
            columns=tuple(json.loads(row["columns_json"])),
            metadata=json.loads(row["metadata_json"]),
            created_at=str(row["created_at"]),
        )


def _validate_name(name: str) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,63}", name):
        raise ValueError(
            "Dataset names must be 2-64 lowercase letters, numbers, '.', '_', or '-'"
        )


def _parse_reference(reference: str) -> tuple[DatasetKind, str, str]:
    match = re.fullmatch(
        r"(training|evaluation|result)/([a-z0-9][a-z0-9._-]{1,63})@([a-f0-9]{12}|latest)",
        reference,
    )
    if match is None:
        raise ValueError(
            "Dataset references must use kind/name@version, for example "
            "training/baseline@latest"
        )
    return DatasetKind(match.group(1)), match.group(2), match.group(3)


def _now() -> str:
    return datetime.now(UTC).isoformat()
