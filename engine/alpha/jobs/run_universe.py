#!/usr/bin/env python3
"""
Dev entrypoint: run the universe builder with mock or live FMP data.

Safe to run locally — mock mode uses no network/secrets, live mode
reads FMP_API_KEY from env.

Usage:
    cd engine

    # Mock mode (default, no network):
    .venv/bin/python -m alpha.jobs.run_universe

    # Live sliced FMP mode (requires FMP_API_KEY):
    FMP_API_KEY=your-key .venv/bin/python -m alpha.jobs.run_universe --live
"""

from __future__ import annotations

import os
import sys

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
    FmpScreenerResult(symbol="EETA", company_name="Eeta Dormant", market_cap=80_000_000, price=4.00, sector="Energy", exchange="NASDAQ", country="US", is_etf=False, is_actively_trading=False),
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


def _run_mock():
    engine = create_engine("sqlite:///universe_dev.db")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    screener_resp = _mock_screener_response()
    job = UniverseBuilderJob(session=session, screener_response=screener_resp)
    result = run_job(session, job, params={"source": "mock"})

    print(f"Status:   {result.status}")
    print(f"Metrics:  {result.metrics}")
    if result.errors:
        print(f"Errors:   {result.errors}")

    session.close()
    engine.dispose()
    os.remove("universe_dev.db")


def _run_live():
    from alpha.data.config import FmpConfig
    from alpha.data.fmp import FmpAdapter
    from alpha.data.universe import SlicedUniverseFetcher

    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        print("ERROR: FMP_API_KEY not set")
        sys.exit(1)

    config = FmpConfig.from_env()
    adapter = FmpAdapter(config)
    fetcher = SlicedUniverseFetcher(adapter)

    print("Fetching sliced universe from FMP...")
    sliced = fetcher.fetch()

    if not sliced.response.ok:
        print(f"FAILED: {sliced.response.error}")
        sys.exit(1)

    engine = create_engine("sqlite:///universe_dev.db")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    job = UniverseBuilderJob(
        session=session,
        screener_response=sliced.response,
        slice_diagnostics=sliced.slice_diagnostics,
    )
    result = run_job(session, job, params={"source": "fmp_sliced"})

    print(f"\nStatus:        {result.status}")
    print(f"Raw unique:    {sliced.unique_raw_count}")
    print(f"Included:      {result.metrics.get('included', 0)}")
    print(f"Excluded:      {result.metrics.get('excluded', 0)}")
    print(f"Slices:        {sliced.slice_count}")
    print(f"Slice limits:  {sliced.slice_limit_hits}")

    exclusion_counts = result.metrics.get("exclusion_counts", {})
    if exclusion_counts:
        print("\nTop exclusion reasons:")
        for reason, count in sorted(exclusion_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    session.close()
    engine.dispose()
    os.remove("universe_dev.db")


def main():
    if "--live" in sys.argv:
        _run_live()
    else:
        _run_mock()


if __name__ == "__main__":
    main()
