"""SQLAlchemy engine/session helpers with optional scratch-schema routing."""

from __future__ import annotations

import os
import re
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from alpha.db.models import Base

_engine = None
_SessionLocal = None
_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_schema_name(schema: str) -> str:
    if not _SCHEMA_RE.match(schema):
        raise ValueError(f"Invalid database schema name: {schema!r}")
    return schema


def schema_connect_args(url: str, schema: str | None = None) -> dict:
    """Return SQLAlchemy kwargs that route PostgreSQL connections to schema.

    The search path keeps ``public`` second so scratch runs can still resolve
    extensions or explicitly seeded fallback objects, while all tables created
    in the scratch schema shadow public for unqualified ORM reads/writes.
    """
    if not schema:
        return {}
    schema = _validate_schema_name(schema)
    if not url.startswith("postgresql"):
        raise ValueError("ALPHA_DB_SCHEMA is only supported for PostgreSQL URLs")
    return {"connect_args": {"options": f"-csearch_path={schema},public"}}


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
            **schema_connect_args(url, schema),
        )
    return _engine


def get_session(url: str | None = None, schema: str | None = None) -> Session:
    """Return a SQLAlchemy session bound to the process-global engine."""

    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(url, schema=schema))
    return _SessionLocal()


def create_all_tables(engine=None):
    """Create ORM tables for local smoke databases and isolated scratch schemas."""

    engine = engine or get_engine()
    schema = os.environ.get("ALPHA_DB_SCHEMA")
    if schema and engine.dialect.name == "postgresql":
        schema = _validate_schema_name(schema)
        with engine.begin() as conn:
            # During scratch runs the normal connection search_path is
            # ``scratch,public`` so reads can still resolve shared objects. For
            # metadata creation, restrict the path to the scratch schema only;
            # otherwise SQLAlchemy's checkfirst can see stale public tables and
            # skip creating the scratch-local table.
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


def reset_globals():
    """For test isolation."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None
