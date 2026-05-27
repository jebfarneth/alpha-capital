import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, text

# Make alpha package importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alpha.db.engine import schema_connect_args
from alpha.db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    url = os.environ.get("DATABASE_URL", config.get_main_option("sqlalchemy.url"))
    schema = os.environ.get("ALPHA_DB_SCHEMA")
    connectable = create_engine(url, **schema_connect_args(url, schema))
    with connectable.connect() as connection:
        if schema:
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            connection.commit()
            connection.exec_driver_sql(f'SET search_path TO "{schema}", public')
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
