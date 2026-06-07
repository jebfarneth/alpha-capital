#!/usr/bin/env python3
"""Market-path feature collector/backfill entrypoint.

Examples:
    cd engine
    uv run python -m alpha.jobs.run_market_path_features --live \
      --pattern-id M4 --signal-start-date 2026-06-02 --signal-end-date 2026-06-05 \
      --through-date 2026-06-05
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from typing import Optional

from alpha.data.config import ConfigError, FmpConfig
from alpha.data.fmp import FmpAdapter
from alpha.db.engine import (
    SchemaTargetError,
    create_all_tables,
    get_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.jobs.market_path_features import (
    DEFAULT_LOOKBACK_CALENDAR_DAYS,
    MarketPathFeatureJob,
)
from alpha.jobs.runner import run_job
from alpha.runtime_env import load_runtime_env


MARKET_PATH_REQUIRED_TABLES = (
    "evidence_jobs",
    "evidence_job_runs",
    "data_lineage",
    "feature_snapshots",
    "signal_registry",
    "market_path_features",
)


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return date.fromisoformat(value)


def _run_live(args: argparse.Namespace) -> int:
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
                required_tables=MARKET_PATH_REQUIRED_TABLES,
            )
        except (SchemaTargetError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1

    try:
        fmp_adapter = FmpAdapter(FmpConfig.from_env())
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 1

    session = get_session()
    if args.create_tables and not target_schema:
        create_all_tables()

    try:
        job = MarketPathFeatureJob(
            session=session,
            fmp_adapter=fmp_adapter,
            run_timestamp=_parse_timestamp(args.run_timestamp),
            pattern_ids=args.pattern_id,
            decision_date=_parse_date(args.decision_date),
            signal_start_date=_parse_date(args.signal_start_date),
            signal_end_date=_parse_date(args.signal_end_date),
            through_date=_parse_date(args.through_date),
            lookback_calendar_days=args.lookback_calendar_days,
            include_signal_session=args.include_signal_session,
            liquidity_min_dollar_volume_20d=args.liquidity_min_dollar_volume_20d,
            liquidity_min_price=args.liquidity_min_price,
        )
        result = run_job(
            session,
            job,
            params={
                "source": "market_path_features",
                "decision_date": args.decision_date,
                "run_timestamp": args.run_timestamp,
                "pattern_id": args.pattern_id,
                "signal_start_date": args.signal_start_date,
                "signal_end_date": args.signal_end_date,
                "through_date": args.through_date,
                "lookback_calendar_days": args.lookback_calendar_days,
                "include_signal_session": args.include_signal_session,
                "schema": args.schema,
            },
        )
        metrics = result.metrics or {}
        print(f"Status:                 {result.status}")
        print(f"Schema:                 {args.schema or os.environ.get('ALPHA_DB_SCHEMA') or 'default'}")
        print(f"Pattern ids:            {metrics.get('pattern_ids')}")
        print(f"Signal start date:      {metrics.get('signal_start_date')}")
        print(f"Signal end date:        {metrics.get('signal_end_date')}")
        print(f"Through date:           {metrics.get('through_date')}")
        print(f"Feature version:        {metrics.get('feature_version')}")
        print(f"Signals scanned:        {metrics.get('signals_scanned')}")
        print(f"Ticker fetches:         {metrics.get('ticker_fetch_count')}")
        print(f"Rows inserted:          {metrics.get('rows_inserted')}")
        print(f"Rows updated:           {metrics.get('rows_updated')}")
        print(f"Rows skipped:           {metrics.get('rows_skipped')}")
        print(f"Fetch errors:           {metrics.get('fetch_error_count')}")
        if metrics.get("no_op_reason"):
            print(f"No-op reason:           {metrics.get('no_op_reason')}")
        if result.errors:
            print("Errors:")
            for error in result.errors[:20]:
                print(f"  {error}")
        return 0 if result.ok else 1
    finally:
        session.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run market-path feature collection/backfill."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Run live feature capture")
    parser.add_argument("--database-url", help="Override DATABASE_URL for this run")
    parser.add_argument(
        "--schema",
        help=(
            "PostgreSQL schema/search_path target for scratch audits. "
            "Requires the schema to exist or --create-tables to create it."
        ),
    )
    parser.add_argument("--create-tables", action="store_true")
    parser.add_argument("--run-timestamp")
    parser.add_argument("--decision-date")
    parser.add_argument(
        "--pattern-id",
        action="append",
        default=[],
        help="Pattern id to enrich. Repeat for multiple patterns. Defaults to M4.",
    )
    parser.add_argument("--signal-start-date")
    parser.add_argument("--signal-end-date")
    parser.add_argument("--through-date")
    parser.add_argument(
        "--lookback-calendar-days",
        type=int,
        default=DEFAULT_LOOKBACK_CALENDAR_DAYS,
    )
    parser.add_argument("--include-signal-session", action="store_true")
    parser.add_argument("--liquidity-min-dollar-volume-20d", type=float, default=100_000.0)
    parser.add_argument("--liquidity-min-price", type=float, default=1.0)
    args = parser.parse_args(argv)
    if not args.pattern_id:
        args.pattern_id = ["M4"]
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if args.live:
        return _run_live(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
