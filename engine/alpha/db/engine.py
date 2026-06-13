"""SQLAlchemy engine/session helpers with optional scratch-schema routing."""

from __future__ import annotations

import os
import re
from typing import Optional, Sequence

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from alpha.db.models import Base

_engine = None
_SessionLocal = None
_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_SCHEMAS = frozenset({"public", "pg_catalog", "information_schema"})


class SchemaTargetError(RuntimeError):
    """Raised when a scratch schema write target is absent or incomplete."""


def _validate_schema_name(schema: str) -> str:
    if not _SCHEMA_RE.match(schema):
        raise ValueError(f"Invalid database schema name: {schema!r}")
    if schema != schema.lower():
        raise ValueError(
            f"Invalid database schema name: {schema!r}; schema names must be lowercase"
        )
    if schema in _RESERVED_SCHEMAS or schema.startswith("pg_"):
        raise ValueError(f"Reserved database schema name: {schema!r}")
    return schema


def schema_connect_args(url: str, schema: str | None = None) -> dict:
    """Return SQLAlchemy kwargs that route PostgreSQL connections to schema.

    Schema-routed sessions intentionally use only the target schema. Keeping
    ``public`` in the search path lets a scratch typo or partial schema resolve
    unqualified ORM tables to canonical public tables, which can mutate the
    production corpus. PostgreSQL still searches ``pg_catalog`` implicitly;
    any legitimate non-table object outside the target schema must be
    schema-qualified by the caller.
    """
    if not schema:
        return {}
    schema = _validate_schema_name(schema)
    if not url.startswith("postgresql"):
        raise ValueError("ALPHA_DB_SCHEMA is only supported for PostgreSQL URLs")
    return {"connect_args": {"options": f"-csearch_path={schema}"}}


def get_engine(url: str | None = None, schema: str | None = None):
    """Return the process-global SQLAlchemy engine for the configured database."""

    global _engine
    if _engine is None:
        url = url or os.environ.get(
            "DATABASE_URL", "sqlite:///alpha_capital.db"
        )
        schema = schema or os.environ.get("ALPHA_DB_SCHEMA")
        _engine = create_engine(
            url,
            echo=False,
            pool_pre_ping=True,
            **schema_connect_args(url, schema),
        )
    return _engine


def get_session(url: str | None = None, schema: str | None = None) -> Session:
    """Return a SQLAlchemy session bound to the process-global engine."""

    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(url, schema=schema))
    return _SessionLocal()


def create_all_tables(engine=None, schema: str | None = None):
    """Create ORM tables for local smoke databases and isolated scratch schemas."""

    engine = engine or get_engine()
    schema = schema or os.environ.get("ALPHA_DB_SCHEMA")
    if schema and engine.dialect.name == "postgresql":
        schema = _validate_schema_name(schema)
        with engine.begin() as conn:
            # Restrict checkfirst to the scratch schema; otherwise SQLAlchemy
            # can see stale public tables and skip creating the scratch-local
            # table.
            conn.execute(text(f'SET search_path TO "{schema}"'))
            Base.metadata.create_all(conn)
        return
    Base.metadata.create_all(engine)


def create_schema_if_missing(engine=None, schema: str | None = None) -> None:
    """Create a PostgreSQL schema target when scratch setup requests it."""

    schema = schema or os.environ.get("ALPHA_DB_SCHEMA")
    if not schema:
        return
    schema = _validate_schema_name(schema)
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))


def prepare_writable_schema_target(
    *,
    url: str | None = None,
    schema: str | None = None,
    create_tables: bool = False,
    required_tables: Optional[Sequence[str]] = None,
) -> None:
    """Fail closed unless a scratch write target exists and has ORM tables.

    Writable jobs call this before opening a session. With ``create_tables``
    false, a missing schema or a partial schema is an operator error. With
    ``create_tables`` true, the helper creates the schema and ORM tables, then
    verifies table presence using ``information_schema``.
    """

    schema = schema or os.environ.get("ALPHA_DB_SCHEMA")
    if not schema:
        return
    if schema.strip().casefold() == "public":
        raise SchemaTargetError(
            "writable --schema targets must not be public; omit --schema for "
            "canonical writes"
        )
    schema = _validate_schema_name(schema)
    url = url or os.environ.get("DATABASE_URL", "sqlite:///alpha_capital.db")
    if not url.startswith("postgresql"):
        raise SchemaTargetError("scratch schema writes require a PostgreSQL DATABASE_URL")

    required = tuple(required_tables or Base.metadata.tables.keys())
    admin_engine = create_engine(url, echo=False)
    try:
        if create_tables:
            create_schema_if_missing(engine=admin_engine, schema=schema)
            target_engine = create_engine(
                url,
                echo=False,
                **schema_connect_args(url, schema),
            )
            try:
                create_all_tables(engine=target_engine, schema=schema)
            finally:
                target_engine.dispose()
        elif not _schema_exists(admin_engine, schema):
            raise SchemaTargetError(
                f"schema {schema!r} does not exist; pass --create-tables to "
                "create an isolated scratch schema"
            )

        missing = _missing_schema_tables(admin_engine, schema, required)
        if missing:
            raise SchemaTargetError(
                f"schema {schema!r} is missing required ORM tables: "
                f"{', '.join(missing[:20])}"
                f"{' ...' if len(missing) > 20 else ''}; pass --create-tables "
                "to create a complete isolated scratch schema"
            )
    finally:
        admin_engine.dispose()


def _schema_exists(engine, schema: str) -> bool:
    with engine.connect() as conn:
        return bool(
            conn.execute(
                text(
                    "SELECT 1 FROM information_schema.schemata "
                    "WHERE schema_name = :schema"
                ),
                {"schema": schema},
            ).scalar()
        )


def _missing_schema_tables(
    engine,
    schema: str,
    required_tables: Sequence[str],
) -> list[str]:
    if not required_tables:
        return []
    with engine.connect() as conn:
        present = {
            row[0]
            for row in conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = :schema"
                ),
                {"schema": schema},
            )
        }
    return sorted(set(required_tables) - present)


def reset_globals():
    """For test isolation."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None
