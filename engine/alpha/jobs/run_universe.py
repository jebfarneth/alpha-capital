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
from contextlib import contextmanager
import os
import sys
from datetime import datetime
from typing import Iterator
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import sessionmaker

from alpha.data.contracts import AdapterResponse, LineageMeta, stable_hash, utcnow
from alpha.data.config import PolygonConfig
from alpha.data.fmp import FmpScreenerResult
from alpha.data.universe_config import MCAP_MAX, MCAP_MIN
from alpha.db.engine import (
    SchemaTargetError,
    create_all_tables,
    get_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.db.models import Base
from alpha.jobs.runner import run_job
from alpha.jobs.security_type import SecurityTypeEnrichmentJob
from alpha.jobs.security_type import profile_refresh_plan
from alpha.jobs.universe_builder import (
    PRICE_MIN,
    UniverseBuilderJob,
    _dedupe_screener_rows,
    _requires_security_profile,
    _screener_asof_error,
)
from alpha.runtime_env import load_runtime_env

MOCK_SCREENER_DATA = [
    FmpScreenerResult(symbol="ACME", company_name="Acme Corp", market_cap=75_000_000, price=5.25, sector="Technology", exchange="NASDAQ", country="US", is_etf=False, is_actively_trading=True),
    FmpScreenerResult(symbol="BETA", company_name="Beta Inc", market_cap=50_000_000, price=3.00, sector="Healthcare", exchange="NASDAQ", country="US", is_etf=False, is_actively_trading=True),
    FmpScreenerResult(symbol="GAMA", company_name="Gama Ltd", market_cap=120_000_000, price=8.50, sector="Industrials", exchange="NYSE", country="US", is_etf=False, is_actively_trading=True),
    FmpScreenerResult(symbol="DELT", company_name="Delta ETF", market_cap=90_000_000, price=25.00, sector="Financial", exchange="ARCA", country="US", is_etf=True, is_actively_trading=True),
    FmpScreenerResult(symbol="EPSI", company_name="Epsilon Micro", market_cap=10_000_000, price=0.80, sector="Technology", exchange="OTC", country="US", is_etf=False, is_actively_trading=True),
    FmpScreenerResult(symbol="ZETA", company_name="Zeta GmbH", market_cap=60_000_000, price=4.00, sector="Healthcare", exchange="XETRA", country="DE", is_etf=False, is_actively_trading=True),
    FmpScreenerResult(symbol="EETA", company_name="Eeta Dormant", market_cap=80_000_000, price=4.00, sector="Energy", exchange="NASDAQ", country="US", is_etf=False, is_actively_trading=False),
]

LIVE_UNIVERSE_ADVISORY_LOCK_KEY = 2026052601


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


@contextmanager
def _live_universe_lock(session) -> Iterator[bool]:
    """Hold one Postgres session-level lock across enrichment and build commits."""
    bind = session.get_bind()
    if bind.dialect.name != "postgresql":
        yield True
        return

    connection: Connection = bind.connect()
    acquired = False
    try:
        acquired = bool(
            connection.execute(
                text("SELECT pg_try_advisory_lock(:key)"),
                {"key": LIVE_UNIVERSE_ADVISORY_LOCK_KEY},
            ).scalar()
        )
        yield acquired
    finally:
        try:
            if acquired:
                connection.execute(
                    text("SELECT pg_advisory_unlock(:key)"),
                    {"key": LIVE_UNIVERSE_ADVISORY_LOCK_KEY},
                )
        finally:
            connection.close()


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

    load_runtime_env()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if args.schema:
        os.environ["ALPHA_DB_SCHEMA"] = args.schema
        reset_globals()
    target_schema = args.schema or os.environ.get("ALPHA_DB_SCHEMA")
    if target_schema is not None:
        try:
            prepare_writable_schema_target(
                schema=target_schema,
                create_tables=args.create_tables,
            )
        except (SchemaTargetError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1
    if not os.environ.get("FMP_API_KEY"):
        print("ERROR: FMP_API_KEY not set")
        return 1
    if args.skip_identity_enrichment and args.require_identity_enrichment:
        print("ERROR: --require-identity-enrichment cannot be combined with --skip-identity-enrichment")
        return 1
    if not 0.0 <= args.min_identity_coverage <= 1.0:
        print("ERROR: --min-identity-coverage must be between 0.0 and 1.0")
        return 1

    session = get_session()
    if args.create_tables and not target_schema:
        create_all_tables()

    config = FmpConfig.from_env()
    adapter = FmpAdapter(config)
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

    asof_error = _screener_asof_error(
        args.trading_date,
        sliced.response.lineage.asof_timestamp,
    )
    if asof_error is not None:
        print(f"Universe build refused: {asof_error}")
        session.close()
        return 1

    with _live_universe_lock(session) as lock_acquired:
        if not lock_acquired:
            print("ERROR: another live universe/enrichment run is already active")
            session.close()
            return 1

        symbols = _required_profile_symbols(sliced.response.data or [])
        refresh_symbols, refresh_metrics = profile_refresh_plan(
            session,
            symbols,
            asof=sliced.response.lineage.asof_timestamp,
        )
        print(
            "Refreshing required security profiles: "
            f"{len(refresh_symbols)} of {len(symbols)} symbols"
        )
        if refresh_metrics["fresh_cached_count"]:
            print(
                "Fresh cached profiles: "
                f"{refresh_metrics['fresh_cached_count']}"
            )
        enrichment_job = SecurityTypeEnrichmentJob(
            session=session,
            adapter=adapter,
            symbols=refresh_symbols,
            retry_backoff_seconds=args.retry_backoff_seconds,
            max_workers=args.profile_max_workers,
            max_profile_calls_per_minute=args.profile_rate_limit_per_minute,
            adapter_factory=lambda: FmpAdapter(config),
        )
        enrichment = run_job(
            session,
            enrichment_job,
            params={
                "source": "fmp_profile",
                "trading_date": args.trading_date,
                "schema": args.schema,
                "required_symbol_count": len(symbols),
                "refresh_symbol_count": len(refresh_symbols),
                "profile_refresh_plan": refresh_metrics,
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
                "schema": args.schema,
                "mcap_min": MCAP_MIN,
                "mcap_max": MCAP_MAX,
                "price_min": PRICE_MIN,
            },
        )
        _print_safe_error("Universe build", universe)

        metrics = universe.metrics or {}
        print(f"Raw unique:    {sliced.unique_raw_count}")
        print(f"Included:      {metrics.get('included', 0)}")
        print(f"Excluded:      {metrics.get('excluded', 0)}")
        print(f"Coverage:      {metrics.get('security_profile_coverage_ratio')}")
        print(
            "Coverage headroom: "
            f"{metrics.get('security_profile_coverage_headroom_count')} profiles"
        )
        print(f"Slices:        {sliced.slice_count}")
        print(f"Slice limits:  {sliced.slice_limit_hits}")
        print(f"Country rescues: {metrics.get('country_profile_rescue_count', 0)}")
        print(f"Cap buckets:   {metrics.get('included_market_cap_bucket_counts', {})}")
        print(f"Price buckets: {metrics.get('included_price_bucket_counts', {})}")
        print(f"Countries:     {metrics.get('included_country_counts', {})}")
        print(f"Shell exclusions: {metrics.get('shell_company_exclusion_count', 0)}")
        print(
            "Included shell-label names: "
            f"{metrics.get('included_shell_company_count', 0)}"
        )
        shell_review = metrics.get("shell_company_exclusion_review_records", [])
        spac_review = metrics.get("spac_pattern_exclusion_review_records", [])
        included_shell_review = metrics.get("included_shell_company_review_records", [])
        print(f"Shell review preview: {shell_review[:25]}")
        print(f"SPAC pattern review preview: {spac_review[:25]}")
        print(f"Included shell-label review preview: {included_shell_review[:25]}")
        print(f"Security type exclusions: {metrics.get('security_type_exclusion_counts', {})}")
        print(
            "Security type reasons: "
            f"{metrics.get('security_type_classification_reason_counts', {})}"
        )

        exclusion_counts = metrics.get("exclusion_counts", {})
        if exclusion_counts:
            print("Top exclusion reasons:")
            for reason, count in sorted(
                exclusion_counts.items(),
                key=lambda item: -item[1],
            )[:12]:
                print(f"  {reason}: {count}")

        # --- Polygon identity enrichment ---
        scan_id = metrics.get("scan_id")
        included_tickers = metrics.get("included_tickers", [])
        if scan_id and included_tickers and not args.skip_identity_enrichment:
            try:
                from alpha.data.polygon import PolygonAdapter
                from alpha.jobs.identity_enrichment import PolygonIdentityEnrichmentJob

                polygon_adapter = PolygonAdapter(PolygonConfig.from_env())
                identity_job = PolygonIdentityEnrichmentJob(
                    session=session,
                    adapter=polygon_adapter,
                    scan_id=scan_id,
                    tickers=included_tickers,
                    max_exception_lookups=args.identity_max_exception_lookups,
                    ticker_event_probes=_csv_arg(args.identity_ticker_event_probes),
                    bulk_limit=args.identity_bulk_limit,
                )
                identity_result = run_job(
                    session,
                    identity_job,
                    params={
                        "source": "polygon_identity",
                        "trading_date": args.trading_date,
                        "schema": args.schema,
                        "scan_id": scan_id,
                        "max_exception_lookups": args.identity_max_exception_lookups,
                        "ticker_event_probes": _csv_arg(args.identity_ticker_event_probes),
                        "bulk_limit": args.identity_bulk_limit,
                    },
                )
                id_metrics = identity_result.metrics or {}
                print(f"Polygon identity attempted: {id_metrics.get('identity_attempted_count')}")
                print(f"Polygon identity present:   {id_metrics.get('identity_present_count')}")
                print(f"Polygon identity no data:   {id_metrics.get('identity_no_data_count')}")
                print(f"Polygon identity errors:    {id_metrics.get('identity_error_count')}")
                print(f"Polygon exception lookups:  {id_metrics.get('identity_exception_lookup_count')}")
                print(f"Polygon ticker events tried:{id_metrics.get('ticker_event_attempted_count')}")
                print(f"Polygon ticker events rows: {id_metrics.get('ticker_event_present_count')}")
                print(f"Polygon CIK present:        {id_metrics.get('cik_present_count')}")
                print(f"Polygon composite FIGI:     {id_metrics.get('composite_figi_present_count')}")
                print(f"Polygon share-class FIGI:   {id_metrics.get('share_class_figi_present_count')}")
                print(f"Polygon coverage ratio:     {id_metrics.get('identity_coverage_ratio')}")
                print(f"Polygon bulk pages:         {id_metrics.get('bulk_pages_fetched')}")
                print(f"Polygon API calls:          {id_metrics.get('polygon_api_call_count')}")
                strict_error = _identity_strict_error(
                    args,
                    id_metrics,
                    included_tickers,
                    identity_result_ok=identity_result.ok,
                )
                if strict_error is not None:
                    print(f"ERROR: Polygon identity enrichment strict check failed: {strict_error}")
                    session.close()
                    return 1
            except Exception as exc:
                print(f"Polygon identity enrichment skipped: {exc}")
                if args.require_identity_enrichment:
                    print("ERROR: Polygon identity enrichment is required")
                    session.close()
                    return 1
        elif args.require_identity_enrichment:
            print(
                "ERROR: Polygon identity enrichment is required but the universe "
                "build did not produce a scan_id and included tickers"
            )
            session.close()
            return 1

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
        "--schema",
        help=(
            "PostgreSQL schema/search_path target for scratch audits. "
            "Requires the schema to exist or --create-tables to create it."
        ),
    )
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
    parser.add_argument(
        "--profile-max-workers",
        type=int,
        default=20,
        help="Maximum concurrent FMP profile fetch workers.",
    )
    parser.add_argument(
        "--profile-rate-limit-per-minute",
        type=int,
        default=2000,
        help="Global cap for FMP profile calls per minute; leave headroom below provider limit.",
    )
    parser.add_argument(
        "--skip-identity-enrichment",
        action="store_true",
        help="Skip Polygon identity enrichment step.",
    )
    parser.add_argument(
        "--require-identity-enrichment",
        action="store_true",
        help=(
            "Fail the live universe run if Polygon identity enrichment is skipped, "
            "errors before completion, or does not attempt every included ticker."
        ),
    )
    parser.add_argument(
        "--min-identity-coverage",
        type=float,
        default=0.0,
        help=(
            "Minimum required Polygon identity_present/attempted coverage when "
            "--require-identity-enrichment is set. Defaults to 0.0 so explicit "
            "no_data/provider_error rows remain allowed."
        ),
    )
    parser.add_argument(
        "--identity-max-exception-lookups",
        type=int,
        default=25,
        help=(
            "Maximum per-ticker Polygon detail lookups after bulk reference misses. "
            "Bulk reference remains the primary path."
        ),
    )
    parser.add_argument(
        "--identity-ticker-event-probes",
        default="",
        help=(
            "Comma-separated targeted tickers for Polygon ticker-events probes. "
            "Ticker events are not called for every universe symbol by default."
        ),
    )
    parser.add_argument(
        "--identity-bulk-limit",
        type=int,
        default=1000,
        help="Polygon /v3/reference/tickers page size, capped at provider max 1000.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for universe construction and security-type enrichment."""

    previous_env = {
        "DATABASE_URL": os.environ.get("DATABASE_URL"),
        "ALPHA_DB_SCHEMA": os.environ.get("ALPHA_DB_SCHEMA"),
    }
    try:
        args = _parse_args(argv or sys.argv[1:])
        if args.live:
            return _run_live(args)
        return _run_mock(args)
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_globals()


def _csv_arg(value: str) -> list[str]:
    """Parse a comma-separated CLI option into normalized symbols."""

    return [
        item.strip().upper()
        for item in str(value or "").split(",")
        if item.strip()
    ]


def _identity_strict_error(
    args: argparse.Namespace,
    metrics: dict,
    included_tickers: list[str],
    *,
    identity_result_ok: bool,
) -> str | None:
    """Return the strict-mode failure reason, or None when the run is acceptable."""

    if not getattr(args, "require_identity_enrichment", False):
        return None
    if not identity_result_ok:
        return "identity enrichment job returned a failed status"

    expected = len(included_tickers)
    attempted = int(metrics.get("identity_attempted_count") or 0)
    if attempted != expected:
        return f"attempted {attempted} of {expected} included tickers"

    min_coverage = float(getattr(args, "min_identity_coverage", 0.0) or 0.0)
    present = int(metrics.get("identity_present_count") or 0)
    coverage = present / attempted if attempted else 0.0
    if coverage < min_coverage:
        return (
            f"coverage {coverage:.4f} below required "
            f"{min_coverage:.4f}"
        )
    return None


if __name__ == "__main__":
    raise SystemExit(main())
