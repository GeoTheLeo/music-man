"""
One connection layer for SQLite (local, tests) and PostgreSQL (cloud).

Set MUSIC_MAN_DATABASE_URL=postgresql://... and the approval queue and the
webhook event log move to Postgres, so separately deployed services (MCP
server, webhook, review app) share one store. Unset, each falls back to its
own SQLite file, exactly as before.

Callers write SQL once with "?" placeholders; Connection translates them
for psycopg. The few real dialect differences (auto-increment keys, getting
an inserted id back, unique-violation errors, adding columns) are covered
by the helpers below rather than by branching in every caller.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

URL_ENV = "MUSIC_MAN_DATABASE_URL"


def database_url() -> str | None:
    return os.environ.get(URL_ENV) or None


def _unique_violation_types() -> tuple[type[Exception], ...]:
    types: tuple[type[Exception], ...] = (sqlite3.IntegrityError,)
    try:
        from psycopg import errors

        types += (errors.UniqueViolation,)
    except ImportError:  # psycopg is only needed when MUSIC_MAN_DATABASE_URL is set
        pass
    return types


UniqueViolation = _unique_violation_types()


class Connection:
    """A small DB-API wrapper: dict-like rows, "?" placeholders, works as a context manager."""

    def __init__(self, raw, postgres: bool) -> None:
        self.raw = raw
        self.postgres = postgres

    @property
    def autoincrement_pk(self) -> str:
        return "BIGSERIAL PRIMARY KEY" if self.postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"

    def execute(self, sql: str, params: tuple = ()):
        if self.postgres:
            sql = sql.replace("?", "%s")
        return self.raw.execute(sql, params)

    def insert_returning_id(self, sql: str, params: tuple) -> int:
        """INSERT and return the new row's id (RETURNING on Postgres, lastrowid on SQLite)."""
        if self.postgres:
            return self.execute(sql + " RETURNING id", params).fetchone()["id"]
        return self.execute(sql, params).lastrowid

    def try_insert(self, sql: str, params: tuple, returning_id: bool = False):
        """INSERT that may hit a unique constraint. Returns the id (or True), or None on a duplicate.

        On Postgres a failed statement aborts the whole transaction, so the
        INSERT runs inside a savepoint and only that savepoint is rolled back.
        """
        if self.postgres:
            self.raw.execute("SAVEPOINT try_insert")
        try:
            result = self.insert_returning_id(sql, params) if returning_id else (self.execute(sql, params) and True)
        except UniqueViolation:
            if self.postgres:
                self.raw.execute("ROLLBACK TO SAVEPOINT try_insert")
            return None
        if self.postgres:
            self.raw.execute("RELEASE SAVEPOINT try_insert")
        return result

    def columns(self, table: str) -> set[str]:
        if self.postgres:
            rows = self.execute(
                "SELECT column_name AS name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = ?",
                (table,),
            )
        else:
            rows = self.execute(f"PRAGMA table_info({table})")
        return {row["name"] for row in rows}

    def commit(self) -> None:
        self.raw.commit()

    def close(self) -> None:
        self.raw.close()

    def __enter__(self) -> Connection:
        return self

    def __exit__(self, exc_type, *exc) -> None:
        if exc_type is None:
            self.raw.commit()
        else:
            self.raw.rollback()
        self.close()


def connect(sqlite_path: Path) -> Connection:
    """Postgres if MUSIC_MAN_DATABASE_URL is set, otherwise the given SQLite file."""
    url = database_url()
    if url:
        import psycopg
        from psycopg.rows import dict_row

        return Connection(psycopg.connect(url, row_factory=dict_row), postgres=True)

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    raw = sqlite3.connect(sqlite_path)
    raw.row_factory = sqlite3.Row
    return Connection(raw, postgres=False)
