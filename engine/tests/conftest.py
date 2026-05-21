import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from alpha.db.models import Base


def _enable_sqlite_fks(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


@pytest.fixture()
def db_session():
    """In-memory SQLite session for test isolation with FK enforcement."""
    engine = create_engine("sqlite:///:memory:")
    event.listen(engine, "connect", _enable_sqlite_fks)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()
