from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from alpha.db.models import Base

_engine = None
_SessionLocal = None


def get_engine(url: str | None = None):
    global _engine
    if _engine is None:
        url = url or os.environ.get(
            "DATABASE_URL", "sqlite:///alpha_capital.db"
        )
        _engine = create_engine(url, echo=False)
    return _engine


def get_session(url: str | None = None) -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(url))
    return _SessionLocal()


def create_all_tables(engine=None):
    engine = engine or get_engine()
    Base.metadata.create_all(engine)


def reset_globals():
    """For test isolation."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None
