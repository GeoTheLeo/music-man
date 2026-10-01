"""
SQLite-backed approval queue for proposed royalty holds. Mirrors the
NorthStar Agentic Intervention Copilot's lib/db.ts pattern: propose_hold()
only ever inserts a pending row here - approving/rejecting (done by a
human in the Streamlit reviewer) is the only thing that has any real
effect.

Holds follow an explicit state machine:

    pending_approval --approve--> executed --release--> released
            |
            +-------reject------> rejected

Every transition is a compare-and-set (UPDATE ... WHERE status = <expected>),
so a double-clicked Approve or two reviewers racing can't apply a decision
twice, and every transition is recorded in hold_events. A partial unique
index allows at most one *active* (pending or executed) hold per
artist/period, which makes create_hold() idempotent under retries: a
second insert for the same artist/period returns the existing hold instead
of creating a duplicate.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from music_man.paths import QUEUE_DB_PATH

PENDING = "pending_approval"
EXECUTED = "executed"
REJECTED = "rejected"
RELEASED = "released"

ACTIVE_STATUSES = (PENDING, EXECUTED)

# from_status -> allowed to_statuses
TRANSITIONS: dict[str, frozenset[str]] = {
    PENDING: frozenset({EXECUTED, REJECTED}),
    EXECUTED: frozenset({RELEASED}),
    REJECTED: frozenset(),
    RELEASED: frozenset(),
}


class InvalidTransition(Exception):
    """Raised when a hold isn't in a state that allows the requested transition."""


@dataclass
class Hold:
    id: int
    artist_id: str
    period: str
    rationale: str
    status: str
    created_at: str
    decided_at: str | None = None
    decided_by: str | None = None
    run_id: str | None = None
    released_at: str | None = None
    released_by: str | None = None
    release_reason: str | None = None


# Columns added after the original schema; _connect() adds any that an
# existing queue.db is missing.
_MIGRATED_COLUMNS = {
    "run_id": "TEXT",
    "released_at": "TEXT",
    "released_by": "TEXT",
    "release_reason": "TEXT",
}


def db_path() -> Path:
    """Queue location; MUSIC_MAN_QUEUE_DB overrides it (tests and evals use a scratch DB)."""
    override = os.environ.get("MUSIC_MAN_QUEUE_DB")
    return Path(override) if override else QUEUE_DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
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
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(holds)")}
    for column, column_type in _MIGRATED_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE holds ADD COLUMN {column} {column_type}")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS one_active_hold_per_period
        ON holds (artist_id, period)
        WHERE status IN ('pending_approval', 'executed')
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hold_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hold_id INTEGER NOT NULL REFERENCES holds(id),
            from_status TEXT,
            to_status TEXT NOT NULL,
            actor TEXT NOT NULL,
            reason TEXT,
            at TEXT NOT NULL
        )
        """
    )
    return conn


def _row_to_hold(row: sqlite3.Row) -> Hold:
    return Hold(**{k: row[k] for k in row.keys()})


def _record_event(
    conn: sqlite3.Connection,
    hold_id: int,
    from_status: str | None,
    to_status: str,
    actor: str,
    reason: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO hold_events (hold_id, from_status, to_status, actor, reason, at) VALUES (?, ?, ?, ?, ?, ?)",
        (hold_id, from_status, to_status, actor, reason, _now()),
    )


def get_active_hold(artist_id: str, period: str) -> Hold | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM holds WHERE artist_id = ? AND period = ? AND status IN (?, ?)",
            (artist_id, period, *ACTIVE_STATUSES),
        ).fetchone()
        return _row_to_hold(row) if row else None


def has_hold_for_period(artist_id: str, period: str) -> bool:
    """True if an active (pending or executed) hold already exists for this artist/period."""
    return get_active_hold(artist_id, period) is not None


def create_hold(
    artist_id: str, period: str, rationale: str, run_id: str | None = None
) -> tuple[Hold, bool]:
    """Insert a pending hold. Returns (hold, created).

    Idempotent: if an active hold already exists for this artist/period
    (including one this same call created before a retry), the unique index
    rejects the insert and the existing hold is returned with created=False.
    """
    with _connect() as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO holds (artist_id, period, rationale, created_at, run_id) VALUES (?, ?, ?, ?, ?)",
                (artist_id, period, rationale, _now(), run_id),
            )
        except sqlite3.IntegrityError:
            existing = conn.execute(
                "SELECT * FROM holds WHERE artist_id = ? AND period = ? AND status IN (?, ?)",
                (artist_id, period, *ACTIVE_STATUSES),
            ).fetchone()
            return _row_to_hold(existing), False
        _record_event(conn, cursor.lastrowid, None, PENDING, f"agent:{run_id or 'unknown'}")
        conn.commit()
        hold_id = cursor.lastrowid
    return get_hold(hold_id), True


def get_hold(hold_id: int) -> Hold:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM holds WHERE id = ?", (hold_id,)).fetchone()
        if row is None:
            raise KeyError(f"hold {hold_id} not found")
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


def list_events(hold_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM hold_events WHERE hold_id = ? ORDER BY id", (hold_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def transition(hold_id: int, to_status: str, actor: str, reason: str | None = None) -> Hold:
    """Move a hold to to_status if the state machine allows it.

    Compare-and-set: the UPDATE only matches if the hold is still in the
    status we read, so a concurrent or repeated decision raises
    InvalidTransition instead of silently applying twice.
    """
    current = get_hold(hold_id)
    if to_status not in TRANSITIONS.get(current.status, frozenset()):
        raise InvalidTransition(f"hold {hold_id}: {current.status} -> {to_status} is not allowed")

    now = _now()
    with _connect() as conn:
        if to_status == RELEASED:
            cursor = conn.execute(
                "UPDATE holds SET status = ?, released_at = ?, released_by = ?, release_reason = ? "
                "WHERE id = ? AND status = ?",
                (to_status, now, actor, reason, hold_id, current.status),
            )
        else:
            cursor = conn.execute(
                "UPDATE holds SET status = ?, decided_at = ?, decided_by = ? WHERE id = ? AND status = ?",
                (to_status, now, actor, hold_id, current.status),
            )
        if cursor.rowcount == 0:
            raise InvalidTransition(f"hold {hold_id} changed state concurrently; nothing applied")
        _record_event(conn, hold_id, current.status, to_status, actor, reason)
        conn.commit()
    return get_hold(hold_id)


def decide_hold(hold_id: int, status: str, decided_by: str) -> Hold:
    """Approve (status='executed') or reject a pending hold."""
    if status not in (EXECUTED, REJECTED):
        raise ValueError(f"invalid status: {status}")
    return transition(hold_id, status, decided_by)


def release_hold(hold_id: int, released_by: str, reason: str) -> Hold:
    """Roll back an executed hold. The payout is released and the period is free for a new hold."""
    if not reason.strip():
        raise ValueError("a release reason is required")
    return transition(hold_id, RELEASED, released_by, reason)
