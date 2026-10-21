"""
Run the whole suite against PostgreSQL instead of SQLite:

    docker run -d --rm --name mm-pg -e POSTGRES_PASSWORD=test -p 55432:5432 postgres:16
    MUSIC_MAN_TEST_DATABASE_URL=postgresql://postgres:test@localhost:55432/postgres pytest

Each test gets a fresh schema, so tests stay as isolated as their SQLite
scratch files. Without MUSIC_MAN_TEST_DATABASE_URL nothing changes.
"""

from __future__ import annotations

import os
import uuid

import pytest

from music_man.agent import queue

TEST_URL = os.environ.get("MUSIC_MAN_TEST_DATABASE_URL")


@pytest.fixture(autouse=True)
def postgres_schema(monkeypatch):
    if not TEST_URL:
        yield
        return

    import psycopg

    schema = f"t_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(TEST_URL, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {schema}")

    separator = "&" if "?" in TEST_URL else "?"
    monkeypatch.setenv("MUSIC_MAN_DATABASE_URL", f"{TEST_URL}{separator}options=-csearch_path%3D{schema}")
    monkeypatch.setattr(queue, "_ready", set())
    yield

    with psycopg.connect(TEST_URL, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")
