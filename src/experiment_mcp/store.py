"""Small SQLite store for experiment records."""

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from experiment_mcp.models import ExperimentRecord, ExperimentSummary, PrimaryMetric

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    primary_metric TEXT NOT NULL,
    offer_id TEXT,
    salt TEXT NOT NULL,
    traffic_split TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    closed_at TEXT,
    notes TEXT NOT NULL
)
"""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class ExperimentStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def create(
        self,
        *,
        name: str,
        hypothesis: str,
        primary_metric: PrimaryMetric,
        offer_id: str | None,
        salt: str,
        traffic_split: dict[str, float],
        notes: str,
    ) -> ExperimentRecord:
        record = ExperimentRecord(
            id=f"exp_{uuid.uuid4().hex[:12]}",
            name=name,
            hypothesis=hypothesis,
            primary_metric=primary_metric,
            offer_id=offer_id,
            salt=salt,
            traffic_split=traffic_split,
            status="running",
            created_at=_utcnow(),
            closed_at=None,
            notes=notes,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experiments (
                    id, name, hypothesis, primary_metric, offer_id, salt,
                    traffic_split, status, created_at, closed_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.name,
                    record.hypothesis,
                    record.primary_metric,
                    record.offer_id,
                    record.salt,
                    json.dumps(record.traffic_split),
                    record.status,
                    record.created_at,
                    record.closed_at,
                    record.notes,
                ),
            )
        return record

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()
        return None if row is None else _record(row)

    def list_experiments(self, status: str | None = None) -> list[ExperimentSummary]:
        query = "SELECT * FROM experiments"
        parameters: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY created_at"
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            ExperimentSummary(
                id=row["id"],
                name=row["name"],
                status=row["status"],
                primary_metric=row["primary_metric"],
                offer_id=row["offer_id"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def ids(self) -> list[str]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT id FROM experiments ORDER BY created_at").fetchall()
        return [str(row["id"]) for row in rows]

    def close(self, experiment_id: str) -> ExperimentRecord | None:
        closed_at = _utcnow()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE experiments
                SET status = 'closed', closed_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (closed_at, experiment_id),
            )
            changed = cursor.rowcount
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()
        if row is None:
            return None
        if changed == 0:
            return _record(row)
        return _record(row)


def _record(row: sqlite3.Row) -> ExperimentRecord:
    return ExperimentRecord(
        id=row["id"],
        name=row["name"],
        hypothesis=row["hypothesis"],
        primary_metric=row["primary_metric"],
        offer_id=row["offer_id"],
        salt=row["salt"],
        traffic_split=json.loads(row["traffic_split"]),
        status=row["status"],
        created_at=row["created_at"],
        closed_at=row["closed_at"],
        notes=row["notes"],
    )
