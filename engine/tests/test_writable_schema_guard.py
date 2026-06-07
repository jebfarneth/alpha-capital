"""Writable scratch-schema guard tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha.db import engine as db_engine
from alpha.db.engine import (
    SchemaTargetError,
    _validate_schema_name,
    prepare_writable_schema_target,
    schema_connect_args,
)
from alpha.jobs import (
    run_forward_context,
    run_forward_return,
    run_m1_daily,
    run_m4_daily,
    run_market_path_features,
    run_nasdaq_archive,
    run_universe,
)


class _FakeEngine:
    def __init__(self):
        self.disposed = False

    def dispose(self):
        self.disposed = True


def test_prepare_writable_schema_target_refuses_missing_schema(monkeypatch):
    created = []
    fake_engine = _FakeEngine()
    monkeypatch.setattr(db_engine, "create_engine", lambda *a, **k: fake_engine)
    monkeypatch.setattr(db_engine, "_schema_exists", lambda engine, schema: False)
    monkeypatch.setattr(
        db_engine,
        "_missing_schema_tables",
        lambda engine, schema, required: [],
    )

    with pytest.raises(SchemaTargetError, match="does not exist"):
        prepare_writable_schema_target(
            url="postgresql+psycopg://user:pass@example.com/db",
            schema="scratch_missing",
            create_tables=False,
        )

    assert created == []
    assert fake_engine.disposed is True


def test_prepare_writable_schema_target_refuses_public_schema():
    with pytest.raises(SchemaTargetError, match="must not be public"):
        prepare_writable_schema_target(
            url="postgresql+psycopg://user:pass@example.com/db",
            schema="public",
            create_tables=False,
        )
    with pytest.raises(SchemaTargetError, match="must not be public"):
        prepare_writable_schema_target(
            url="postgresql+psycopg://user:pass@example.com/db",
            schema="PUBLIC",
            create_tables=True,
        )


@pytest.mark.parametrize("schema", ["PUBLIC", "Public", "Scratch_Audit"])
def test_validate_schema_name_rejects_non_lowercase(schema):
    with pytest.raises(ValueError, match="lowercase"):
        _validate_schema_name(schema)


@pytest.mark.parametrize(
    "schema",
    ["public", "pg_catalog", "information_schema", "pg_temp_audit"],
)
def test_validate_schema_name_rejects_reserved_names(schema):
    with pytest.raises(ValueError, match="Reserved"):
        _validate_schema_name(schema)


def test_schema_connect_args_rejects_folded_public_variants():
    with pytest.raises(ValueError, match="lowercase"):
        schema_connect_args(
            "postgresql+psycopg://user:pass@example.com/db",
            "PUBLIC",
        )


def test_alembic_env_validates_schema_and_removes_public_fallback():
    env_py = Path("migrations/env.py").read_text()

    assert "_validate_schema_name(schema)" in env_py
    assert "schema_connect_args(url, schema)" in env_py
    assert 'SET search_path TO "{schema}", public' not in env_py
    assert 'SET search_path TO "{schema}"' in env_py


def test_prepare_writable_schema_target_refuses_incomplete_schema(monkeypatch):
    fake_engine = _FakeEngine()
    monkeypatch.setattr(db_engine, "create_engine", lambda *a, **k: fake_engine)
    monkeypatch.setattr(db_engine, "_schema_exists", lambda engine, schema: True)
    monkeypatch.setattr(
        db_engine,
        "_missing_schema_tables",
        lambda engine, schema, required: ["signal_registry"],
    )

    with pytest.raises(SchemaTargetError, match="missing required ORM tables"):
        prepare_writable_schema_target(
            url="postgresql+psycopg://user:pass@example.com/db",
            schema="scratch_incomplete",
            create_tables=False,
        )

    assert fake_engine.disposed is True


def test_prepare_writable_schema_target_create_tables_verifies_complete_schema(
    monkeypatch,
):
    engines = [_FakeEngine(), _FakeEngine()]
    create_engine_calls = []
    create_schema_calls = []
    create_all_calls = []

    def fake_create_engine(*args, **kwargs):
        create_engine_calls.append((args, kwargs))
        return engines[len(create_engine_calls) - 1]

    monkeypatch.setattr(db_engine, "create_engine", fake_create_engine)
    monkeypatch.setattr(
        db_engine,
        "create_schema_if_missing",
        lambda *, engine, schema: create_schema_calls.append((engine, schema)),
    )
    monkeypatch.setattr(
        db_engine,
        "create_all_tables",
        lambda *, engine, schema: create_all_calls.append((engine, schema)),
    )
    monkeypatch.setattr(
        db_engine,
        "_missing_schema_tables",
        lambda engine, schema, required: [],
    )

    prepare_writable_schema_target(
        url="postgresql+psycopg://user:pass@example.com/db",
        schema="scratch_ready",
        create_tables=True,
    )

    assert create_schema_calls == [(engines[0], "scratch_ready")]
    assert create_all_calls == [(engines[1], "scratch_ready")]
    assert create_engine_calls[1][1]["connect_args"]["options"] == (
        "-csearch_path=scratch_ready"
    )
    assert engines[0].disposed is True
    assert engines[1].disposed is True


@pytest.mark.parametrize(
    ("module", "argv"),
    [
        (
            run_forward_return,
            [
                "--live",
                "--schema",
                "scratch_missing",
                "--run-timestamp",
                "2026-06-03T21:00:00+00:00",
            ],
        ),
        (
            run_m4_daily,
            [
                "--live",
                "--schema",
                "scratch_missing",
                "--run-timestamp",
                "2026-06-03T21:00:00+00:00",
            ],
        ),
        (
            run_m1_daily,
            [
                "--live",
                "--schema",
                "scratch_missing",
                "--run-timestamp",
                "2026-06-03T21:00:00+00:00",
            ],
        ),
        (
            run_forward_context,
            [
                "--live",
                "--schema",
                "scratch_missing",
                "--run-timestamp",
                "2026-06-03T21:00:00+00:00",
            ],
        ),
        (
            run_universe,
            [
                "--live",
                "--schema",
                "scratch_missing",
                "--trading-date",
                "2026-06-03",
            ],
        ),
        (
            run_nasdaq_archive,
            [
                "--live",
                "--schema",
                "scratch_missing",
                "--run-timestamp",
                "2026-06-03T21:00:00+00:00",
            ],
        ),
    ],
)
def test_writable_schema_entrypoints_preflight_before_session(
    module,
    argv,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("FMP_API_KEY", "test-fmp")
    monkeypatch.setattr(module, "load_runtime_env", lambda: None)
    monkeypatch.setattr(
        module,
        "prepare_writable_schema_target",
        lambda **kwargs: (_ for _ in ()).throw(
            SchemaTargetError("schema 'scratch_missing' does not exist")
        ),
    )
    monkeypatch.setattr(
        module,
        "get_session",
        lambda *args, **kwargs: pytest.fail("session opened before schema guard"),
    )

    rc = module.main(argv)
    captured = capsys.readouterr()

    assert rc == 1
    assert "schema 'scratch_missing' does not exist" in captured.out


def test_market_path_schema_preflight_does_not_require_m3_tables(monkeypatch, capsys):
    captured_kwargs = {}

    monkeypatch.setattr(run_market_path_features, "load_runtime_env", lambda: None)
    monkeypatch.setattr(
        run_market_path_features,
        "prepare_writable_schema_target",
        lambda **kwargs: captured_kwargs.update(kwargs) or (_ for _ in ()).throw(
            SchemaTargetError("schema preflight stop")
        ),
    )
    monkeypatch.setattr(
        run_market_path_features,
        "get_session",
        lambda *args, **kwargs: pytest.fail("session opened before schema guard"),
    )

    rc = run_market_path_features.main([
        "--live",
        "--schema",
        "scratch_v3",
        "--signal-start-date",
        "2026-06-02",
        "--signal-end-date",
        "2026-06-02",
        "--through-date",
        "2026-06-05",
    ])
    out = capsys.readouterr().out

    assert rc == 1
    assert "schema preflight stop" in out
    assert captured_kwargs["required_tables"] == (
        "evidence_jobs",
        "evidence_job_runs",
        "data_lineage",
        "feature_snapshots",
        "signal_registry",
        "market_path_features",
    )


@pytest.mark.parametrize(
    ("module", "argv"),
    [
        (
            run_forward_return,
            [
                "--live",
                "--run-timestamp",
                "2026-06-03T21:00:00+00:00",
            ],
        ),
        (
            run_m4_daily,
            [
                "--live",
                "--run-timestamp",
                "2026-06-03T21:00:00+00:00",
            ],
        ),
        (
            run_m1_daily,
            [
                "--live",
                "--run-timestamp",
                "2026-06-03T21:00:00+00:00",
            ],
        ),
        (
            run_forward_context,
            [
                "--live",
                "--run-timestamp",
                "2026-06-03T21:00:00+00:00",
            ],
        ),
        (
            run_universe,
            [
                "--live",
                "--trading-date",
                "2026-06-03",
            ],
        ),
        (
            run_nasdaq_archive,
            [
                "--live",
                "--run-timestamp",
                "2026-06-03T21:00:00+00:00",
            ],
        ),
    ],
)
def test_writable_schema_entrypoints_preflight_env_only_schema_before_session(
    module,
    argv,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("FMP_API_KEY", "test-fmp")
    monkeypatch.setenv("ALPHA_DB_SCHEMA", "scratch_env_missing")
    monkeypatch.setattr(module, "load_runtime_env", lambda: None)
    monkeypatch.setattr(
        module,
        "prepare_writable_schema_target",
        lambda **kwargs: (_ for _ in ()).throw(
            SchemaTargetError("schema 'scratch_env_missing' does not exist")
        ),
    )
    monkeypatch.setattr(
        module,
        "get_session",
        lambda *args, **kwargs: pytest.fail("session opened before schema guard"),
    )

    rc = module.main(argv)
    captured = capsys.readouterr()

    assert rc == 1
    assert "schema 'scratch_env_missing' does not exist" in captured.out
