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

from alpha.data.benzinga import BenzingaAdapter
from alpha.data.config import BenzingaConfig, ConfigError, FmpConfig, PolygonConfig
from alpha.data.fmp import FmpAdapter
from alpha.data.polygon import PolygonAdapter
from alpha.db.engine import (
    SchemaTargetError,
    create_all_tables,
    get_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.jobs.m4_daily import M4DailyAssemblyJob
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

    session = get_session()
    if args.create_tables and not target_schema:
        create_all_tables()

    try:
        adapter = FmpAdapter(FmpConfig.from_env())
        polygon_adapter = _optional_polygon_adapter()
        benzinga_adapter = _optional_benzinga_adapter()
        job = M4DailyAssemblyJob(
            session=session,
            adapter=adapter,
            polygon_adapter=polygon_adapter,
            benzinga_adapter=benzinga_adapter,
            run_timestamp=_parse_timestamp(args.run_timestamp),
            lookback_calendar_days=args.lookback_calendar_days,
            signal_context_breakout_buffer=args.signal_context_breakout_buffer,
        )
        result = run_job(
            session,
            job,
            params={
                "source": "fmp_full",
                "run_timestamp": args.run_timestamp,
                "lookback_calendar_days": args.lookback_calendar_days,
                "signal_context_breakout_buffer": args.signal_context_breakout_buffer,
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
        context_metrics = metrics.get("signal_context") or {}
        print(f"Context candidates:     {context_metrics.get('context_prefilter_candidate_count')}")
        print(f"Context skipped:        {context_metrics.get('context_prefilter_skipped_count')}")
        print(f"Context attempts:       {context_metrics.get('source_attempt_count')}")
        print(f"Context provider errs:  {context_metrics.get('provider_error_count')}")
        print(f"Assembled M4 inputs:    {assembly.get('assembled_count')}")
        print(f"M4 signals persisted:   {orchestration.get('total_signals_persisted')}")
        if result.errors:
            print("Errors:")
            for error in result.errors[:20]:
                print(f"  {error}")
        return 0 if result.ok else 1
    finally:
        session.close()


def _optional_polygon_adapter() -> PolygonAdapter | None:
    try:
        return PolygonAdapter(PolygonConfig.from_env())
    except ConfigError:
        return None


def _optional_benzinga_adapter() -> BenzingaAdapter | None:
    try:
        return BenzingaAdapter(BenzingaConfig.from_env())
    except ConfigError:
        return None


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
        "--signal-context-breakout-buffer",
        type=float,
        default=0.02,
        help=(
            "M4 context prefilter buffer. Context is fetched only for inputs "
            "with price >= high_52w * (1 - buffer)."
        ),
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Create tables from SQLAlchemy metadata before running. Use only for local smoke DBs; production should use Alembic.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the production daily M4 signal path."""

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
