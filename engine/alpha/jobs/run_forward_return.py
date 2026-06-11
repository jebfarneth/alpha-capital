#!/usr/bin/env python3
"""Production forward-return entrypoint.

Usage:
    cd engine
    uv run python -m alpha.jobs.run_forward_return --live --run-timestamp 2026-06-16T17:00:00-04:00

Historical M4 backfill chunks:
    while true; do
      uv run python -m alpha.jobs.run_forward_return --live \
        --signal-source historical-m4-replay \
        --signal-start-date 2024-01-01 \
        --signal-end-date 2024-01-31 \
        --run-timestamp 2026-06-09T22:15:03.685171+00:00 \
        --limit 5000 \
        --prefetch-workers 16 \
        --prefetch-rate-limit-per-minute 1500
      # Stop when Remaining eligible prints 0.
    done
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from alpha.data.benzinga import BenzingaAdapter
from alpha.data.config import BenzingaConfig, FmpConfig, SecEdgarConfig
from alpha.data.edgar import SecEdgarAdapter
from alpha.data.fmp import FmpAdapter
from alpha.data.nasdaq import NasdaqTraderListingAdapter
from alpha.db.engine import (
    SchemaTargetError,
    create_all_tables,
    get_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.jobs.forward_return import (
    DEFAULT_FINALITY_LAG_SESSIONS,
    DEFAULT_REVISION_WINDOW_SESSIONS,
    PRICE_DRIFT_ABS_TOL,
    PRICE_DRIFT_REL_TOL,
    ForwardReturnJob,
)
from alpha.jobs.historical_m4_signal_selector import (
    SIGNAL_SOURCE_CHOICES,
    SIGNAL_SOURCE_LIVE,
)
from alpha.jobs.runner import run_job
from alpha.runtime_env import load_runtime_env

LIVE_RUN_TIMESTAMP_SKEW_TOLERANCE = timedelta(minutes=5)
NASDAQ_LISTING_AUTHORITY_ENV = "NASDAQ_LISTING_AUTHORITY_ENABLED"


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return date.fromisoformat(value)


def _live_timestamp_error(
    value: Optional[str],
    *,
    now: Optional[datetime] = None,
    tolerance: timedelta = LIVE_RUN_TIMESTAMP_SKEW_TOLERANCE,
) -> Optional[str]:
    if not value:
        return None
    try:
        parsed = _parse_timestamp(value)
    except ValueError:
        return f"invalid run_timestamp: {value}"
    if parsed is None:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return "live run_timestamp must be timezone-aware"
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    if parsed > now + tolerance:
        return (
            "live run_timestamp is in the future; use explicit audited "
            "historical/backfill mode instead of --live time travel"
        )
    return None


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _run_live(args: argparse.Namespace) -> int:
    load_runtime_env()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if args.schema:
        os.environ["ALPHA_DB_SCHEMA"] = args.schema
        reset_globals()
    target_schema = args.schema or os.environ.get("ALPHA_DB_SCHEMA")
    timestamp_error = _live_timestamp_error(args.run_timestamp)
    if timestamp_error:
        print(f"ERROR: {timestamp_error}")
        return 1
    try:
        signal_start_date = _parse_date(args.signal_start_date)
        signal_end_date = _parse_date(args.signal_end_date)
    except ValueError as exc:
        print(f"ERROR: invalid signal date: {exc}")
        return 1
    if (
        signal_start_date is not None
        and signal_end_date is not None
        and signal_start_date > signal_end_date
    ):
        print("ERROR: signal_start_date must be on or before signal_end_date")
        return 1
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

    session = get_session()
    if args.create_tables and not target_schema:
        create_all_tables()

    try:
        fmp_config = FmpConfig.from_env()
        adapter = FmpAdapter(fmp_config)
        survivorship_adapters = []
        if os.environ.get("SEC_USER_AGENT"):
            survivorship_adapters.append(SecEdgarAdapter(SecEdgarConfig.from_env()))
        if os.environ.get("BENZINGA_API_KEY") or os.environ.get("BENZINGA_TOKEN"):
            survivorship_adapters.append(BenzingaAdapter(BenzingaConfig.from_env()))
        listing_authority_adapter = None
        if _env_flag(NASDAQ_LISTING_AUTHORITY_ENV):
            listing_authority_adapter = NasdaqTraderListingAdapter()
        survivorship_source_names = ["fmp_delisted_companies"] + [
            "sec_edgar_survivorship_events"
            for _source in survivorship_adapters
            if isinstance(_source, SecEdgarAdapter)
        ] + [
            "benzinga_calendar_ma"
            for _source in survivorship_adapters
            if isinstance(_source, BenzingaAdapter)
        ]
        if listing_authority_adapter is not None:
            survivorship_source_names.append("nasdaq_listing_status")
        job = ForwardReturnJob(
            session=session,
            adapter=adapter,
            survivorship_adapters=survivorship_adapters,
            listing_authority_adapter=listing_authority_adapter,
            run_timestamp=_parse_timestamp(args.run_timestamp),
            max_attempts=args.max_attempts,
            pattern_id=args.pattern_id,
            finality_lag_sessions=args.finality_lag_sessions,
            reconcile_computed=args.reconcile_computed,
            revision_window_sessions=args.revision_window_sessions,
            price_drift_abs_tol=args.price_drift_abs_tol,
            price_drift_rel_tol=args.price_drift_rel_tol,
            signal_source=args.signal_source,
            signal_start_date=signal_start_date,
            signal_end_date=signal_end_date,
            limit=args.limit,
            prefetch_workers=args.prefetch_workers,
            prefetch_rate_limit_per_minute=args.prefetch_rate_limit_per_minute,
            adapter_factory=lambda: FmpAdapter(fmp_config),
        )
        result = run_job(
            session,
            job,
            params={
                "source": "fmp_full",
                "survivorship_sources": survivorship_source_names,
                "run_timestamp": args.run_timestamp,
                "max_attempts": args.max_attempts,
                "pattern_id": args.pattern_id,
                "finality_lag_sessions": args.finality_lag_sessions,
                "reconcile_computed": args.reconcile_computed,
                "revision_window_sessions": args.revision_window_sessions,
                "price_drift_abs_tol": args.price_drift_abs_tol,
                "price_drift_rel_tol": args.price_drift_rel_tol,
                "signal_source": args.signal_source,
                "signal_start_date": args.signal_start_date,
                "signal_end_date": args.signal_end_date,
                "limit": args.limit,
                "prefetch_workers": args.prefetch_workers,
                "prefetch_rate_limit_per_minute": (
                    args.prefetch_rate_limit_per_minute
                ),
                "schema": args.schema,
            },
        )
        metrics = result.metrics or {}
        print(f"Status:                  {result.status}")
        print(f"Mode:                    {metrics.get('mode')}")
        print(f"Pattern:                 {metrics.get('pattern_id')}")
        print(f"Signal source:           {metrics.get('signal_source')}")
        print(f"Signal start date:       {metrics.get('signal_start_date')}")
        print(f"Signal end date:         {metrics.get('signal_end_date')}")
        print(f"Schema:                  {args.schema or os.environ.get('ALPHA_DB_SCHEMA') or 'default'}")
        print(f"Survivorship sources:    {survivorship_source_names}")
        print(f"Eligible signals:        {metrics.get('total_eligible')}")
        print(f"Selected signals:        {metrics.get('selected_signals')}")
        print(f"Limit applied:           {metrics.get('limit_applied')}")
        print(f"Remaining eligible:      {metrics.get('remaining_eligible_count')}")
        print(f"Prefetch enabled:        {metrics.get('prefetch_enabled')}")
        print(f"Prefetch workers:        {metrics.get('prefetch_workers')}")
        print(f"Prefetch fetches:        {metrics.get('prefetch_fetches')}")
        print(
            "Prefetch peak calls/min: "
            f"{metrics.get('prefetch_peak_calls_per_minute')}"
        )
        print(f"Computed:                {metrics.get('computed')}")
        print(f"Pending:                 {metrics.get('pending')}")
        print(f"Finality pending:        {metrics.get('price_finality_pending')}")
        print(f"Price drift review:      {metrics.get('price_drift_review')}")
        print(f"Reconciliation passed:   {metrics.get('reconciliation_passed')}")
        print(f"Provider revision review:{metrics.get('provider_revision_review')}")
        print(f"Skipped outside window:  {metrics.get('skipped_outside_window')}")
        print(f"Retryable unavailable:   {metrics.get('retryable_unavailable')}")
        print(f"Terminal unavailable:    {metrics.get('terminal_unavailable')}")
        print(f"Observations upserted:   {metrics.get('observations_upserted')}")
        print(f"Events appended:         {metrics.get('events_appended')}")
        print(f"Fetch errors:            {metrics.get('fetch_error_count')}")
        if result.errors:
            print("Errors:")
            for error in result.errors[:20]:
                print(f"  {error}")
        return 0 if result.ok else 1
    finally:
        session.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run forward-return population.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Run live FMP workflow")
    parser.add_argument("--database-url", help="Override DATABASE_URL for this run")
    parser.add_argument(
        "--schema",
        help=(
            "PostgreSQL schema/search_path target for scratch audits. "
            "Requires the schema to exist or --create-tables to create it."
        ),
    )
    parser.add_argument(
        "--run-timestamp",
        help=(
            "Timezone-aware timestamp used for maturity/session resolution. "
            "Defaults to the evidence job run start time."
        ),
    )
    parser.add_argument(
        "--pattern-id",
        default="M4",
        help="Pattern id to price. First production slice supports M4.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Terminalize retryable unavailable outcomes at this attempt count.",
    )
    parser.add_argument(
        "--finality-lag-sessions",
        type=int,
        default=DEFAULT_FINALITY_LAG_SESSIONS,
        help=(
            "Regular U.S. equity sessions to wait after the exit session before "
            "finalizing M4 forward returns."
        ),
    )
    parser.add_argument(
        "--reconcile-computed",
        action="store_true",
        help=(
            "Explicit post-compute provider-revision sweep for recently "
            "computed M4 forward-return rows."
        ),
    )
    parser.add_argument(
        "--revision-window-sessions",
        type=int,
        default=DEFAULT_REVISION_WINDOW_SESSIONS,
        help=(
            "Regular U.S. equity sessions after finality during which computed "
            "rows are eligible for provider-revision reconciliation."
        ),
    )
    parser.add_argument(
        "--price-drift-abs-tol",
        type=float,
        default=PRICE_DRIFT_ABS_TOL,
        help="Absolute provider-row drift tolerance for price reconciliation.",
    )
    parser.add_argument(
        "--price-drift-rel-tol",
        type=float,
        default=PRICE_DRIFT_REL_TOL,
        help="Relative provider-row drift tolerance for price reconciliation.",
    )
    parser.add_argument(
        "--signal-source",
        choices=SIGNAL_SOURCE_CHOICES,
        default=SIGNAL_SOURCE_LIVE,
        help=(
            "Signal corpus to label. Use historical-m4-replay to exclude stale "
            "live M4 rows outside replay membership."
        ),
    )
    parser.add_argument(
        "--signal-start-date",
        help=(
            "Inclusive signal date lower bound, YYYY-MM-DD. Use with "
            "--signal-end-date for month-scoped historical backfills."
        ),
    )
    parser.add_argument(
        "--signal-end-date",
        help=(
            "Inclusive signal date upper bound, YYYY-MM-DD. Use with "
            "--signal-start-date for month-scoped historical backfills."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Process at most N eligible signals, then finish and commit normally. "
            "For historical backfills, rerun chunks such as --limit 5000 until "
            "Remaining eligible prints 0."
        ),
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=0,
        help=(
            "Opt-in concurrent FMP price prefetch workers. 0 or 1 preserves the "
            "current sequential fetch path; use about 16 for historical backfills."
        ),
    )
    parser.add_argument(
        "--prefetch-rate-limit-per-minute",
        type=int,
        default=1500,
        help=(
            "Client-side cap for opt-in FMP prefetch calls per minute. "
            "Default leaves at least 50%% headroom below the 3000/min plan."
        ),
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Create tables from SQLAlchemy metadata before running local smoke DBs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for forward-return population and reconciliation."""

    previous_env = {
        "DATABASE_URL": os.environ.get("DATABASE_URL"),
        "ALPHA_DB_SCHEMA": os.environ.get("ALPHA_DB_SCHEMA"),
    }
    try:
        args = _parse_args(argv or sys.argv[1:])
        if args.live:
            return _run_live(args)
        return 1
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_globals()


if __name__ == "__main__":
    raise SystemExit(main())
