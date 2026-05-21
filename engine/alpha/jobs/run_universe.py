#!/usr/bin/env python3
"""
Dev entrypoint: run the universe builder with mock data.

Safe to run locally — no network, no secrets, uses SQLite.

Usage:
    cd engine
    .venv/bin/python -m alpha.jobs.run_universe
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from alpha.data.contracts import AdapterResponse, LineageMeta, stable_hash, utcnow
from alpha.data.fmp import FmpScreenerResult
from alpha.db.models import Base
from alpha.jobs.runner import run_job
from alpha.jobs.universe_builder import UniverseBuilderJob

MOCK_SCREENER_DATA = [
    FmpScreenerResult(symbol="ACME", company_name="Acme Corp", market_cap=75_000_000, price=5.25, sector="Technology", exchange="NASDAQ", country="US", is_etf=False, is_actively_trading=True),
    FmpScreenerResult(symbol="BETA", company_name="Beta Inc", market_cap=50_000_000, price=3.00, sector="Healthcare", exchange="NASDAQ", country="US", is_etf=False, is_actively_trading=True),
    FmpScreenerResult(symbol="GAMA", company_name="Gama Ltd", market_cap=120_000_000, price=8.50, sector="Industrials", exchange="NYSE", country="US", is_etf=False, is_actively_trading=True),
    FmpScreenerResult(symbol="DELT", company_name="Delta ETF", market_cap=90_000_000, price=25.00, sector="Financial", exchange="ARCA", country="US", is_etf=True, is_actively_trading=True),
    FmpScreenerResult(symbol="EPSI", company_name="Epsilon Micro", market_cap=10_000_000, price=0.80, sector="Technology", exchange="OTC", country="US", is_etf=False, is_actively_trading=True),
    FmpScreenerResult(symbol="ZETA", company_name="Zeta GmbH", market_cap=60_000_000, price=4.00, sector="Healthcare", exchange="XETRA", country="DE", is_etf=False, is_actively_trading=True),
    FmpScreenerResult(symbol="EETA", company_name="Eeta Dormant", market_cap=80_000_000, price=2.00, sector="Energy", exchange="NASDAQ", country="US", is_etf=False, is_actively_trading=False),
]


def _mock_screener_response():
    ts = utcnow()
    payload_text = str(MOCK_SCREENER_DATA)
    return AdapterResponse(
        data=MOCK_SCREENER_DATA,
        lineage=LineageMeta(
            provider="FMP",
            endpoint="/stable/company-screener",
            request_timestamp=ts,
            asof_timestamp=ts,
            raw_payload_hash=stable_hash(payload_text),
            source_authority="mock",
        ),
    )


def main():
    engine = create_engine("sqlite:///universe_dev.db")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    screener_resp = _mock_screener_response()
    job = UniverseBuilderJob(session=session, screener_response=screener_resp)

    result = run_job(session, job, params={"source": "mock"})

    print(f"Status:   {result.status}")
    print(f"Metrics:  {result.metrics}")
    print(f"Input H:  {result.input_hashes}")
    print(f"Output H: {result.output_hashes}")
    if result.errors:
        print(f"Errors:   {result.errors}")

    session.close()
    engine.dispose()

    import os
    os.remove("universe_dev.db")


if __name__ == "__main__":
    main()
