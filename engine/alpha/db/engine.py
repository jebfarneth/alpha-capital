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
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(url, schema=schema))
    return _SessionLocal()


def create_all_tables(engine=None):
    engine = engine or get_engine()
    Base.metadata.create_all(engine)


def create_schema_if_missing(engine=None, schema: str | None = None) -> None:
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
