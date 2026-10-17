"""SQLite storage for workers (with face embeddings) and violations."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS workers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    email       TEXT UNIQUE,
    embedding   BLOB NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS violations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT NOT NULL,
    worker_id   INTEGER REFERENCES workers(id) ON DELETE SET NULL,
    worker_name TEXT NOT NULL,
    similarity  REAL,
    confidence  REAL NOT NULL,
    snapshot    TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_violations_created ON violations(created_at);
"""


@dataclass(frozen=True)
class Worker:
    id: int
    name: str
    email: str | None
    created_at: str


@dataclass(frozen=True)
class ViolationRecord:
    id: int
    camera_id: str
    worker_id: int | None
    worker_name: str
    similarity: float | None
    confidence: float
    snapshot: str | None
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    # Workers -----------------------------------------------------------------

    def add_worker(self, name: str, email: str | None, embedding: np.ndarray) -> Worker:
        blob = np.asarray(embedding, dtype=np.float32).tobytes()
        created = _now()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT INTO workers (name, email, embedding, created_at) VALUES (?, ?, ?, ?)",
                (name, email, blob, created),
            )
        return Worker(cursor.lastrowid, name, email, created)

    def list_workers(self) -> list[Worker]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, email, created_at FROM workers ORDER BY name").fetchall()
        return [Worker(**dict(row)) for row in rows]

    def delete_worker(self, worker_id: int) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM workers WHERE id = ?", (worker_id,))
        return cursor.rowcount > 0

    def worker_embeddings(self) -> list[tuple[int, str, np.ndarray]]:
        with self._lock:
            rows = self._conn.execute("SELECT id, name, embedding FROM workers").fetchall()
        return [(row["id"], row["name"], np.frombuffer(row["embedding"], dtype=np.float32))
                for row in rows]

    # Violations --------------------------------------------------------------

    def add_violation(self, camera_id: str, worker_id: int | None, worker_name: str,
                      similarity: float | None, confidence: float,
                      snapshot: str | None) -> ViolationRecord:
        created = _now()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT INTO violations (camera_id, worker_id, worker_name, similarity, confidence,"
                " snapshot, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (camera_id, worker_id, worker_name, similarity, confidence, snapshot, created),
            )
        return ViolationRecord(cursor.lastrowid, camera_id, worker_id, worker_name, similarity,
                               confidence, snapshot, created)

    def list_violations(self, limit: int = 50, worker_id: int | None = None) -> list[ViolationRecord]:
        query = "SELECT * FROM violations"
        params: tuple = ()
        if worker_id is not None:
            query += " WHERE worker_id = ?"
            params = (worker_id,)
        query += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(query, params + (limit,)).fetchall()
        return [ViolationRecord(**dict(row)) for row in rows]

    def violation_summary(self) -> dict:
        today = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
            today_count = self._conn.execute(
                "SELECT COUNT(*) FROM violations WHERE created_at >= ?", (today,)).fetchone()[0]
            by_worker = self._conn.execute(
                "SELECT worker_name, COUNT(*) AS count FROM violations"
                " GROUP BY worker_name ORDER BY count DESC LIMIT 10").fetchall()
        return {"total": total, "today": today_count,
                "by_worker": [dict(row) for row in by_worker]}
