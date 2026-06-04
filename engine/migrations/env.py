import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, text

# Make alpha package importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alpha.db.engine import _validate_schema_name, schema_connect_args
from alpha.db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _migration_url():
    env_url = os.environ.get("DATABASE_URL")
    url = env_url or config.get_main_option("sqlalchemy.url")
    allow_sqlite = os.environ.get("ALPHA_ALLOW_SQLITE_ALEMBIC") == "1"
    if not env_url and not allow_sqlite:
        raise RuntimeError(
            "DATABASE_URL is required for Alembic migrations; set "
            "ALPHA_ALLOW_SQLITE_ALEMBIC=1 only for local SQLite test runs"
        )
    if url.startswith("sqlite") and not allow_sqlite:
        raise RuntimeError(
            "Alembic migrations refuse SQLite unless "
            "ALPHA_ALLOW_SQLITE_ALEMBIC=1 is set for a local test run"
        )
    return url


def run_migrations_offline():
    url = _migration_url()
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    url = _migration_url()
    schema = os.environ.get("ALPHA_DB_SCHEMA")
    if schema:
        schema = _validate_schema_name(schema)
        admin = create_engine(url)
        try:
            with admin.begin() as connection:
                connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        finally:
            admin.dispose()
    connectable = create_engine(url, **schema_connect_args(url, schema))
    with connectable.connect() as connection:
        if schema:
            connection.exec_driver_sql(f'SET search_path TO "{schema}"')
            connection.commit()
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=schema,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
