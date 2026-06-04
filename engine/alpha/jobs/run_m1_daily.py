#!/usr/bin/env python3
"""Daily M1 production entrypoint."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

from alpha.data.config import FmpConfig
from alpha.data.fmp import FmpAdapter
from alpha.db.engine import (
    SchemaTargetError,
    create_all_tables,
    get_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.jobs.m1_daily import M1DailyAssemblyJob
from alpha.jobs.runner import run_job
from alpha.runtime_env import load_runtime_env


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _run_live(args: argparse.Namespace) -> int:
    load_runtime_env()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if args.schema:
        os.environ["ALPHA_DB_SCHEMA"] = args.schema
        reset_globals()
    target_schema = args.schema
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
        adapter = FmpAdapter(FmpConfig.from_env())
        job = M1DailyAssemblyJob(
            session=session,
            adapter=adapter,
            run_timestamp=_parse_timestamp(args.run_timestamp),
            earnings_window_sessions=args.earnings_window_sessions,
            next_earnings_calendar_days=args.next_earnings_calendar_days,
            price_lookback_calendar_days=args.price_lookback_calendar_days,
        )
        result = run_job(
            session,
            job,
            params={
                "source": "fmp_earnings",
                "run_timestamp": args.run_timestamp,
                "earnings_window_sessions": args.earnings_window_sessions,
                "next_earnings_calendar_days": args.next_earnings_calendar_days,
                "price_lookback_calendar_days": args.price_lookback_calendar_days,
                "schema": args.schema,
            },
        )
        metrics = result.metrics or {}
        assembly = metrics.get("assembly") or {}
        orchestration = metrics.get("orchestration") or {}
        print(f"Status:                    {result.status}")
        print(f"Decision date:             {metrics.get('decision_date')}")
        print(f"Evidence session date:     {metrics.get('evidence_session_date')}")
        print(f"Schema:                    {args.schema or os.environ.get('ALPHA_DB_SCHEMA') or 'default'}")
        print(f"Included universe:         {metrics.get('included_universe_size')}")
        print(f"Announcing universe rows:  {metrics.get('announcing_universe_event_count')}")
        print(f"Foster computed:           {metrics.get('foster_computed_count')}")
        print(f"Insufficient EPS history:  {metrics.get('foster_insufficient_history_count')}")
        print(f"Friction computed:         {metrics.get('friction_computed_count')}")
        print(f"Market factor:             {metrics.get('market_factor_symbol')}")
        print(f"Assembled M1 inputs:       {assembly.get('assembled_count')}")
        print(f"M1 signals persisted:      {orchestration.get('total_signals_persisted')}")
        if result.errors:
            print("Errors:")
            for error in result.errors[:20]:
                print(f"  {error}")
        return 0 if result.ok else 1
    finally:
        session.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily M1 PEAD production wiring.")
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
            "Timezone-aware timestamp used for market-session resolution. "
            "Defaults to the evidence job run start time."
        ),
    )
    parser.add_argument(
        "--earnings-window-sessions",
        type=int,
        default=15,
        help="Trailing sessions in the announcing cohort.",
    )
    parser.add_argument(
        "--next-earnings-calendar-days",
        type=int,
        default=140,
        help="Forward calendar days to search for next earnings dates.",
    )
    parser.add_argument(
        "--price-lookback-calendar-days",
        type=int,
        default=430,
        help="Calendar days of price history for D1/sigma_epsilon.",
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Create SQLAlchemy metadata tables before running. Production should use Alembic.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if args.live:
        return _run_live(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
