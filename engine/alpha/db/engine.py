"""SQLAlchemy engine/session helpers with optional scratch-schema routing."""

from __future__ import annotations

import os
import re
from typing import Optional, Sequence
from urllib.parse import urlparse

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session

from alpha.db.models import Base

_engine = None
_SessionLocal = None
_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_SCHEMAS = frozenset({"public", "pg_catalog", "information_schema"})
DEFAULT_POSTGRES_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_POSTGRES_KEEPALIVES_IDLE_SECONDS = 30
DEFAULT_POSTGRES_KEEPALIVES_INTERVAL_SECONDS = 10
DEFAULT_POSTGRES_KEEPALIVES_COUNT = 5
DEFAULT_POSTGRES_TCP_USER_TIMEOUT_MS = 30_000
DEFAULT_POSTGRES_STATEMENT_TIMEOUT_MS = 300_000
DEFAULT_POSTGRES_IDLE_IN_TRANSACTION_TIMEOUT_MS = 300_000


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


def _postgres_timeout_options(
    schema: str | None,
    *,
    include_statement_timeouts: bool = True,
) -> str:
    options: list[str] = []
    if schema:
        options.append(f"-csearch_path={schema}")
    if not include_statement_timeouts:
        return " ".join(options)
    statement_timeout = int(
        os.environ.get(
            "ALPHA_DB_STATEMENT_TIMEOUT_MS",
            str(DEFAULT_POSTGRES_STATEMENT_TIMEOUT_MS),
        )
    )
    idle_timeout = int(
        os.environ.get(
            "ALPHA_DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS",
            str(DEFAULT_POSTGRES_IDLE_IN_TRANSACTION_TIMEOUT_MS),
        )
    )
    options.append(f"-cstatement_timeout={statement_timeout}")
    options.append(f"-cidle_in_transaction_session_timeout={idle_timeout}")
    return " ".join(options)


def _postgres_connect_args(
    schema: str | None = None,
    *,
    include_statement_timeouts: bool = True,
) -> dict:
    return {
        "connect_timeout": int(
            os.environ.get(
                "ALPHA_DB_CONNECT_TIMEOUT_SECONDS",
                str(DEFAULT_POSTGRES_CONNECT_TIMEOUT_SECONDS),
            )
        ),
        "keepalives": 1,
        "keepalives_idle": int(
            os.environ.get(
                "ALPHA_DB_KEEPALIVES_IDLE_SECONDS",
                str(DEFAULT_POSTGRES_KEEPALIVES_IDLE_SECONDS),
            )
        ),
        "keepalives_interval": int(
            os.environ.get(
                "ALPHA_DB_KEEPALIVES_INTERVAL_SECONDS",
                str(DEFAULT_POSTGRES_KEEPALIVES_INTERVAL_SECONDS),
            )
        ),
        "keepalives_count": int(
            os.environ.get(
                "ALPHA_DB_KEEPALIVES_COUNT",
                str(DEFAULT_POSTGRES_KEEPALIVES_COUNT),
            )
        ),
        "tcp_user_timeout": int(
            os.environ.get(
                "ALPHA_DB_TCP_USER_TIMEOUT_MS",
                str(DEFAULT_POSTGRES_TCP_USER_TIMEOUT_MS),
            )
        ),
        "options": _postgres_timeout_options(
            schema,
            include_statement_timeouts=include_statement_timeouts,
        ),
    }


def schema_connect_args(
    url: str,
    schema: str | None = None,
    *,
    include_statement_timeouts: bool = True,
) -> dict:
    """Return SQLAlchemy kwargs that route PostgreSQL connections to schema.

    Startup options also set statement timeouts so a stuck DB operation cannot
    pin a long-running shard forever. Some poolers ignore startup options, so
    ``get_engine`` also binds the live session search path with ``SET`` on
    connect.
    """
    if schema and not url.startswith("postgresql"):
        raise ValueError("ALPHA_DB_SCHEMA is only supported for PostgreSQL URLs")
    if not url.startswith("postgresql"):
        return {}
    if schema:
        schema = _validate_schema_name(schema)
    return {
        "connect_args": _postgres_connect_args(
            schema,
            include_statement_timeouts=include_statement_timeouts,
        )
    }


def _bind_schema_search_path(engine, url: str, schema: str | None) -> None:
    if not schema or not url.startswith("postgresql"):
        return
    schema = _validate_schema_name(schema)

    @event.listens_for(engine, "connect")
    def _set_search_path(dbapi_conn, _record):  # noqa: ANN001
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute(f'SET search_path TO "{schema}", public')
        finally:
            cursor.close()
        if not getattr(dbapi_conn, "autocommit", False):
            dbapi_conn.commit()


def _is_supabase_transaction_pooler(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.hostname is not None
        and parsed.hostname.endswith("pooler.supabase.com")
        and parsed.port == 6543
    )


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
        _bind_schema_search_path(_engine, url, schema)
    return _engine


def get_session(url: str | None = None, schema: str | None = None) -> Session:
    """Return a SQLAlchemy session bound to the process-global engine."""

    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(url, schema=schema))
    return _SessionLocal()


def assert_session_targets_schema(session: Session, schema: str | None) -> None:
    """Fail closed unless scratch is the first PostgreSQL search_path entry."""

    if not schema:
        return
    bind = session.get_bind()
    if getattr(getattr(bind, "dialect", None), "name", None) != "postgresql":
        return
    search_path = session.execute(text("SHOW search_path")).scalar() or ""
    parts = [part.strip().strip('"') for part in search_path.split(",")]
    if not parts or parts[0] != schema:
        raise SchemaTargetError(
            f"Refusing to write: requested schema {schema!r} but the live session "
            f"search_path is {search_path!r} (scratch schema is not first). "
            "Aborting to protect canonical data."
        )


def open_writable_session(
    url: str | None = None,
    schema: str | None = None,
) -> Session:
    """Open a writable session guaranteed not to reuse a stale global bind."""

    schema = schema or os.environ.get("ALPHA_DB_SCHEMA")
    url = url or os.environ.get("DATABASE_URL", "sqlite:///alpha_capital.db")
    if schema and _is_supabase_transaction_pooler(url):
        raise SchemaTargetError(
            "transaction pooler does not persist scratch-schema search_path; "
            "use the Supabase session pooler on port 5432 or a direct "
            "PostgreSQL connection for writable jobs"
        )
    reset_globals()
    session = get_session(url=url, schema=schema)
    try:
        assert_session_targets_schema(session, schema)
    except Exception:
        session.close()
        raise
    return session


def create_all_tables(
    engine=None,
    schema: str | None = None,
    table_names: Optional[Sequence[str]] = None,
):
    """Create ORM tables for local smoke databases and isolated scratch schemas."""

    engine = engine or get_engine()
    schema = schema or os.environ.get("ALPHA_DB_SCHEMA")
    tables = None
    if table_names is not None:
        missing = sorted(set(table_names) - set(Base.metadata.tables))
        if missing:
            raise SchemaTargetError(
                "unknown ORM tables requested for scratch schema creation: "
                f"{', '.join(missing)}"
            )
        tables = [Base.metadata.tables[name] for name in table_names]
    if schema and engine.dialect.name == "postgresql":
        schema = _validate_schema_name(schema)
        with engine.begin() as conn:
            # Restrict checkfirst to the scratch schema; otherwise SQLAlchemy
            # can see stale public tables and skip creating the scratch-local
            # table.
            conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))
            if tables is None:
                Base.metadata.create_all(conn)
            else:
                Base.metadata.create_all(conn, tables=tables)
        return
    if tables is None:
        Base.metadata.create_all(engine)
    else:
        Base.metadata.create_all(engine, tables=tables)


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
    admin_engine = create_engine(
        url,
        echo=False,
        **schema_connect_args(url, None),
    )
    try:
        if create_tables:
            create_schema_if_missing(engine=admin_engine, schema=schema)
            target_engine = create_engine(
                url,
                echo=False,
                **schema_connect_args(url, schema),
            )
            try:
                create_all_tables(
                    engine=target_engine,
                    schema=schema,
                    table_names=required,
                )
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
