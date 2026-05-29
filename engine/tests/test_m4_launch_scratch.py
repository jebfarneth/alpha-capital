"""Guard tests for the live M4 launch scratch runner."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

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
