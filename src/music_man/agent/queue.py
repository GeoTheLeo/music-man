"""
SQLite-backed approval queue for proposed royalty holds. Mirrors the
NorthStar Agentic Intervention Copilot's lib/db.ts pattern: propose_hold()
only ever inserts a pending row here - approving/rejecting (done by a
human in the Streamlit reviewer) is the only thing that has any real
effect.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from music_man.paths import QUEUE_DB_PATH


@dataclass
class Hold:
    id: int
    artist_id: str
    period: str
    rationale: str
    status: str
    created_at: str
    decided_at: str | None
    decided_by: str | None


def _connect() -> sqlite3.Connection:
    QUEUE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(QUEUE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS holds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            artist_id TEXT NOT NULL,
            period TEXT NOT NULL,
            rationale TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending_approval',
            created_at TEXT NOT NULL,
            decided_at TEXT,
            decided_by TEXT
        )
        """
    )
    return conn


def _row_to_hold(row: sqlite3.Row) -> Hold:
    return Hold(**{k: row[k] for k in row.keys()})


def has_hold_for_period(artist_id: str, period: str) -> bool:
    """True if a non-rejected hold already exists for this artist/period."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM holds WHERE artist_id = ? AND period = ? AND status != 'rejected'",
            (artist_id, period),
        ).fetchone()
        return row["n"] > 0


def create_hold(artist_id: str, period: str, rationale: str) -> Hold:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO holds (artist_id, period, rationale, created_at) VALUES (?, ?, ?, ?)",
            (artist_id, period, rationale, now),
        )
        conn.commit()
        return get_hold(cursor.lastrowid)


def get_hold(hold_id: int) -> Hold:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM holds WHERE id = ?", (hold_id,)).fetchone()
        return _row_to_hold(row)


def list_holds(status: str | None = None) -> list[Hold]:
    with _connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM holds WHERE status = ? ORDER BY created_at DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM holds ORDER BY created_at DESC").fetchall()
        return [_row_to_hold(r) for r in rows]


def decide_hold(hold_id: int, status: str, decided_by: str) -> Hold:
    if status not in ("executed", "rejected"):
        raise ValueError(f"invalid status: {status}")
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "UPDATE holds SET status = ?, decided_at = ?, decided_by = ? WHERE id = ?",
            (status, now, decided_by, hold_id),
        )
        conn.commit()
    return get_hold(hold_id)
