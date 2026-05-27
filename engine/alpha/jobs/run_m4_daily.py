#!/usr/bin/env python3
"""Daily M4 production entrypoint.

Usage:
    cd engine
    uv run python -m alpha.jobs.run_m4_daily --live --run-timestamp 2026-05-26T08:00:00+00:00
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

from alpha.data.config import FmpConfig
from alpha.data.fmp import FmpAdapter
from alpha.db.engine import create_all_tables, create_schema_if_missing, get_session, reset_globals
from alpha.jobs.m4_daily import M4DailyAssemblyJob
from alpha.jobs.runner import run_job


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


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _run_live(args: argparse.Namespace) -> int:
    _load_dotenv()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if args.schema:
        os.environ["ALPHA_DB_SCHEMA"] = args.schema
        reset_globals()
    if not os.environ.get("FMP_API_KEY"):
        print("ERROR: FMP_API_KEY not set")
        return 1

    session = get_session()
    if args.create_tables:
        create_schema_if_missing(schema=args.schema)
        create_all_tables()

    try:
        adapter = FmpAdapter(FmpConfig.from_env())
        job = M4DailyAssemblyJob(
            session=session,
            adapter=adapter,
            run_timestamp=_parse_timestamp(args.run_timestamp),
            lookback_calendar_days=args.lookback_calendar_days,
        )
        result = run_job(
            session,
            job,
            params={
                "source": "fmp_full",
                "run_timestamp": args.run_timestamp,
                "lookback_calendar_days": args.lookback_calendar_days,
                "schema": args.schema,
            },
        )
        metrics = result.metrics or {}
        orchestration = metrics.get("orchestration") or {}
        assembly = metrics.get("assembly") or {}
        print(f"Status:                 {result.status}")
        print(f"Decision date:          {metrics.get('decision_date')}")
        print(f"Evidence session date:  {metrics.get('evidence_session_date')}")
        print(f"Schema:                 {args.schema or os.environ.get('ALPHA_DB_SCHEMA') or 'default'}")
        print(f"Fetch through date:     {metrics.get('fetch_to_date')}")
        print(f"Included universe:      {metrics.get('included_universe_size')}")
        print(f"Fetched symbols:        {metrics.get('fetched_symbol_count')}")
        print(f"Fetched bars:           {metrics.get('fetched_bar_count')}")
        print(f"Fetch errors:           {metrics.get('fetch_error_count')}")
        print(f"Assembled M4 inputs:    {assembly.get('assembled_count')}")
        print(f"M4 signals persisted:   {orchestration.get('total_signals_persisted')}")
        if result.errors:
            print("Errors:")
            for error in result.errors[:20]:
                print(f"  {error}")
        return 0 if result.ok else 1
    finally:
        session.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily M4 production wiring.")
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
        "--lookback-calendar-days",
        type=int,
        default=430,
        help="Calendar days to request before the evidence session.",
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Create tables from SQLAlchemy metadata before running. Use only for local smoke DBs; production should use Alembic.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the production daily M4 signal path."""

    args = _parse_args(argv or sys.argv[1:])
    if args.live:
        return _run_live(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
