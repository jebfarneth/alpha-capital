#!/usr/bin/env python3
"""
Universe build entrypoint.

Live mode reads DATABASE_URL and FMP_API_KEY from the environment, fetches
the sliced FMP universe, refreshes required security profiles, then writes
a strict canonical universe scan.

Usage:
    cd engine
    .venv/bin/python -m alpha.jobs.run_universe --live --trading-date 2026-05-24
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from alpha.data.contracts import AdapterResponse, LineageMeta, stable_hash, utcnow
from alpha.data.fmp import FmpScreenerResult
from alpha.data.universe_config import MCAP_MAX, MCAP_MIN
from alpha.db.engine import create_all_tables, get_session, reset_globals
from alpha.db.models import Base
from alpha.jobs.runner import run_job
from alpha.jobs.security_type import SecurityTypeEnrichmentJob
from alpha.jobs.universe_builder import (
    UniverseBuilderJob,
    _dedupe_screener_rows,
    _requires_security_profile,
)

MOCK_SCREENER_DATA = [
    FmpScreenerResult(symbol="ACME", company_name="Acme Corp", market_cap=75_000_000, price=5.25, sector="Technology", exchange="NASDAQ", country="US", is_etf=False, is_actively_trading=True),
    FmpScreenerResult(symbol="BETA", company_name="Beta Inc", market_cap=50_000_000, price=3.00, sector="Healthcare", exchange="NASDAQ", country="US", is_etf=False, is_actively_trading=True),
    FmpScreenerResult(symbol="GAMA", company_name="Gama Ltd", market_cap=120_000_000, price=8.50, sector="Industrials", exchange="NYSE", country="US", is_etf=False, is_actively_trading=True),
    FmpScreenerResult(symbol="DELT", company_name="Delta ETF", market_cap=90_000_000, price=25.00, sector="Financial", exchange="ARCA", country="US", is_etf=True, is_actively_trading=True),
    FmpScreenerResult(symbol="EPSI", company_name="Epsilon Micro", market_cap=10_000_000, price=0.80, sector="Technology", exchange="OTC", country="US", is_etf=False, is_actively_trading=True),
    FmpScreenerResult(symbol="ZETA", company_name="Zeta GmbH", market_cap=60_000_000, price=4.00, sector="Healthcare", exchange="XETRA", country="DE", is_etf=False, is_actively_trading=True),
    FmpScreenerResult(symbol="EETA", company_name="Eeta Dormant", market_cap=80_000_000, price=4.00, sector="Energy", exchange="NASDAQ", country="US", is_etf=False, is_actively_trading=False),
]


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)


def _default_trading_date() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


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


def _required_profile_symbols(stocks: list[FmpScreenerResult]) -> list[str]:
    deduped, _ = _dedupe_screener_rows(stocks)
    return sorted({
        symbol
        for _, included, reason, symbol in deduped
        if _requires_security_profile(included, reason)
    })


def _print_safe_error(label: str, result) -> None:
    print(f"{label} status: {result.status}")
    if result.errors:
        for error in result.errors:
            print(f"  error_stage={error.get('stage')} message={error.get('message')}")


def _run_mock(args) -> int:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        screener_resp = _mock_screener_response()
        job = UniverseBuilderJob(session=session, screener_response=screener_resp)
        result = run_job(
            session, job,
            params={"source": "mock", "trading_date": args.trading_date},
        )
        print(f"Status:   {result.status}")
        print(f"Metrics:  {result.metrics}")
        return 0 if result.ok else 1
    finally:
        session.close()
        engine.dispose()


def _run_live(args) -> int:
    from alpha.data.config import FmpConfig
    from alpha.data.fmp import FmpAdapter
    from alpha.data.universe import SlicedUniverseFetcher

    _load_dotenv()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if not os.environ.get("FMP_API_KEY"):
        print("ERROR: FMP_API_KEY not set")
        return 1

    session = get_session()
    if args.create_tables:
        create_all_tables()

    adapter = FmpAdapter(FmpConfig.from_env())
    fetcher = SlicedUniverseFetcher(adapter)

    print(f"Fetching sliced universe from FMP: ${MCAP_MIN:,}-${MCAP_MAX:,}...")
    sliced = fetcher.fetch()
    if not sliced.response.ok:
        err = sliced.response.error
        print(
            "Fetch failed:",
            f"error_type={getattr(err, 'error_type', None)}",
            f"status={getattr(err, 'status_code', None)}",
            f"retryable={getattr(err, 'retryable', None)}",
        )
        session.close()
        return 1

    symbols = _required_profile_symbols(sliced.response.data or [])
    print(f"Refreshing required security profiles: {len(symbols)} symbols")
    enrichment_job = SecurityTypeEnrichmentJob(
        session=session,
        adapter=adapter,
        symbols=symbols,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )
    enrichment = run_job(
        session,
        enrichment_job,
        params={
            "source": "fmp_profile",
            "trading_date": args.trading_date,
            "required_symbol_count": len(symbols),
        },
    )
    _print_safe_error("Security enrichment", enrichment)
    print(f"Security enrichment metrics: {enrichment.metrics}")
    if not enrichment.ok:
        session.close()
        return 1

    universe_job = UniverseBuilderJob(
        session=session,
        screener_response=sliced.response,
        slice_diagnostics=sliced.slice_diagnostics,
        require_security_profile_cache=not args.allow_incomplete_security_cache,
        min_security_profile_coverage=args.min_security_profile_coverage,
    )
    universe = run_job(
        session,
        universe_job,
        params={
            "source": "fmp_sliced",
            "trading_date": args.trading_date,
            "mcap_min": MCAP_MIN,
            "mcap_max": MCAP_MAX,
        },
    )
    _print_safe_error("Universe build", universe)

    metrics = universe.metrics or {}
    print(f"Raw unique:    {sliced.unique_raw_count}")
    print(f"Included:      {metrics.get('included', 0)}")
    print(f"Excluded:      {metrics.get('excluded', 0)}")
    print(f"Coverage:      {metrics.get('security_profile_coverage_ratio')}")
    print(f"Slices:        {sliced.slice_count}")
    print(f"Slice limits:  {sliced.slice_limit_hits}")
    print(f"Security type exclusions: {metrics.get('security_type_exclusion_counts', {})}")

    exclusion_counts = metrics.get("exclusion_counts", {})
    if exclusion_counts:
        print("Top exclusion reasons:")
        for reason, count in sorted(exclusion_counts.items(), key=lambda item: -item[1])[:12]:
            print(f"  {reason}: {count}")

    session.close()
    return 0 if universe.ok else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the universe build workflow.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Run live FMP workflow")
    mode.add_argument("--mock", action="store_true", help="Run local mock workflow")
    parser.add_argument(
        "--trading-date",
        default=_default_trading_date(),
        help="Market trading date for the scan (YYYY-MM-DD). Defaults to US/Eastern date.",
    )
    parser.add_argument("--database-url", help="Override DATABASE_URL for this run")
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Create tables from SQLAlchemy metadata before running. Use only for local smoke DBs; production should use Alembic.",
    )
    parser.add_argument(
        "--allow-incomplete-security-cache",
        action="store_true",
        help="Disable the strict security-profile coverage gate.",
    )
    parser.add_argument(
        "--min-security-profile-coverage",
        type=float,
        default=1.0,
        help="Minimum required security-profile coverage for strict live builds.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=1.0,
        help="Initial backoff for retryable FMP profile errors.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if args.live:
        return _run_live(args)
    return _run_mock(args)


if __name__ == "__main__":
    raise SystemExit(main())
