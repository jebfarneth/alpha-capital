"""Guard tests for the live M4 launch scratch runner."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from alpha.db import engine as db_engine
from alpha.db.engine import create_all_tables
from alpha.jobs.run_m4_launch_scratch import (
    _parse_timestamp,
    _require_safe_database,
    _schema_name,
    _url_metadata,
)


def test_launch_scratch_schema_name_is_generated_and_namespaced():
    schema = _schema_name(datetime(2026, 5, 28, 12, 34, 56, tzinfo=timezone.utc))

    assert schema == "m4_live_scratch_20260528_123456"


def test_launch_scratch_url_metadata_redacts_credentials():
    meta = _url_metadata(
        "postgresql+psycopg://user:secret@db.example.supabase.co:5432/postgres"
        "?sslmode=require"
    )

    assert meta == {
        "scheme": "postgresql+psycopg",
        "hostname": "db.example.supabase.co",
        "port": 5432,
        "database": "postgres",
        "sslmode": "require",
        "host_class": "direct_supabase",
    }
    assert "secret" not in str(meta)
    assert "user" not in str(meta)


@pytest.mark.parametrize(
    ("url", "schema", "message"),
    [
        ("sqlite:///tmp.db", "m4_live_scratch_20260528_120000", "PostgreSQL"),
        ("postgresql+psycopg://u:p@localhost/db", "public", "non-public"),
        ("postgresql+psycopg://u:p@localhost/db", "manual_test", "m4_live_scratch_"),
    ],
)
def test_launch_scratch_refuses_unsafe_targets(url, schema, message):
    with pytest.raises(ValueError, match=message):
        _require_safe_database(url, schema)


def test_launch_scratch_requires_timezone_aware_run_timestamp():
    with pytest.raises(ValueError, match="timezone-aware"):
        _parse_timestamp("2026-05-28T12:00:00")

    parsed = _parse_timestamp("2026-05-28T12:00:00-04:00")
    assert parsed == datetime(2026, 5, 28, 16, 0, tzinfo=timezone.utc)


def test_create_all_tables_restricts_checkfirst_to_scratch_schema(monkeypatch):
    class FakeDialect:
        name = "postgresql"

    class FakeConnection:
        def __init__(self):
            self.statements = []

        def execute(self, statement):
            self.statements.append(str(statement))

    class FakeBegin:
        def __init__(self, connection):
            self.connection = connection

        def __enter__(self):
            return self.connection

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeEngine:
        dialect = FakeDialect()

        def __init__(self):
            self.connection = FakeConnection()

        def begin(self):
            return FakeBegin(self.connection)

    fake_engine = FakeEngine()
    create_calls = []
    monkeypatch.setenv("ALPHA_DB_SCHEMA", "m4_live_scratch_20260529_020000")
    monkeypatch.setattr(
        db_engine.Base.metadata,
        "create_all",
        lambda bind: create_calls.append(bind),
    )

    create_all_tables(fake_engine)

    assert fake_engine.connection.statements == [
        'SET LOCAL search_path TO "m4_live_scratch_20260529_020000"'
    ]
    assert create_calls == [fake_engine.connection]
