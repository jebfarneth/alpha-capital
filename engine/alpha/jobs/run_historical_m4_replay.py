#!/usr/bin/env python3
"""Historical M4 replay entrypoint.

Scratch-only by design. This CLI refuses public/default schema writes.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from typing import Optional

from alpha.data.config import ConfigError, FmpConfig, PolygonConfig
from alpha.data.fmp import FmpAdapter
from alpha.data.polygon import PolygonAdapter
from alpha.db.engine import (
    SchemaTargetError,
    get_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.jobs.historical_m4_replay import HistoricalM4ReplayJob
from alpha.jobs.runner import run_job
from alpha.market_calendar import is_us_equity_session
from alpha.runtime_env import load_runtime_env


HISTORICAL_M4_REPLAY_REQUIRED_TABLES = (
    "evidence_jobs",
    "evidence_job_runs",
    "data_lineage",
    "feature_snapshots",
    "signal_registry",
    "universe_scans",
    "canonical_universe_scans",
    "universe_snapshots",
    "historical_universe_reconstructions",
)


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")
    days: list[date] = []
    cursor = start
    while cursor <= end:
        if is_us_equity_session(cursor):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _run_live(args: argparse.Namespace) -> int:
    load_runtime_env()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if args.schema:
        os.environ["ALPHA_DB_SCHEMA"] = args.schema
        reset_globals()
    target_schema = args.schema or os.environ.get("ALPHA_DB_SCHEMA")
    if not target_schema:
        print("ERROR: historical M4 replay requires --schema")
        return 1
    if target_schema.strip().casefold() == "public":
        print("ERROR: historical M4 replay refuses the public schema")
        return 1
    try:
        prepare_writable_schema_target(
            schema=target_schema,
            create_tables=args.create_tables,
            required_tables=HISTORICAL_M4_REPLAY_REQUIRED_TABLES,
        )
    except (SchemaTargetError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        fmp_adapter = FmpAdapter(FmpConfig.from_env())
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 1
    polygon_adapter = _optional_polygon_adapter()

    if args.replay_date:
        replay_dates = [_parse_date(value) for value in args.replay_date]
    else:
        replay_dates = _date_range(_parse_date(args.start_date), _parse_date(args.end_date))

    session = get_session()
    try:
        job = HistoricalM4ReplayJob(
            session=session,
            fmp_adapter=fmp_adapter,
            polygon_adapter=polygon_adapter,
            replay_dates=replay_dates,
            run_timestamp=_parse_timestamp(args.run_timestamp),
            allow_partial_universe=args.allow_partial_universe,
            lookback_calendar_days=args.lookback_calendar_days,
        )
        result = run_job(
            session,
            job,
            params={
                "source": "historical_m4_replay",
                "schema": target_schema,
                "replay_dates": [day.isoformat() for day in replay_dates],
                "run_timestamp": args.run_timestamp,
                "allow_partial_universe": args.allow_partial_universe,
                "lookback_calendar_days": args.lookback_calendar_days,
                "polygon_fallback_configured": polygon_adapter is not None,
            },
        )
        metrics = result.metrics or {}
        print(f"Status:                 {result.status}")
        print(f"Schema:                 {target_schema}")
        print(f"Replay dates:           {metrics.get('replay_dates')}")
        print(f"Universe included:      {metrics.get('total_universe_included_count')}")
        print(f"Tickers with bars:      {metrics.get('total_tickers_with_bars')}")
        print(f"Tickers missing bars:   {metrics.get('total_tickers_missing_bars')}")
        print(f"Assembled:              {metrics.get('total_assembled_count')}")
        print(f"M4 signals inserted:    {metrics.get('total_fired_m4_signal_count')}")
        print(f"Rejected/no-fire:       {metrics.get('total_rejected_or_no_fire_count')}")
        print(f"Fetch errors:           {metrics.get('total_fetch_error_count')}")
        print(f"Rows inserted:          {metrics.get('total_rows_inserted')}")
        print(f"Rows reused:            {metrics.get('total_rows_reused')}")
        for date_result in metrics.get("date_results") or []:
            print(
                "Date result:            "
                f"{date_result.get('replay_date')} "
                f"included={date_result.get('universe_included_count')} "
                f"bars={date_result.get('tickers_with_bars')} "
                f"missing={date_result.get('tickers_missing_bars')} "
                f"assembled={date_result.get('assembled_count')} "
                f"signals={date_result.get('fired_m4_signal_count')}"
            )
        if result.errors:
            print("Errors:")
            for error in result.errors[:20]:
                print(f"  {error}")
        return 0 if result.ok else 1
    finally:
        session.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay historical M4 base-daily signals in a scratch schema."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--database-url")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--create-tables", action="store_true")
    parser.add_argument("--run-timestamp")
    parser.add_argument("--lookback-calendar-days", type=int, default=430)
    parser.add_argument("--allow-partial-universe", action="store_true")
    date_mode = parser.add_mutually_exclusive_group(required=True)
    date_mode.add_argument("--replay-date", action="append")
    date_mode.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args(argv)
    if args.start_date and not args.end_date:
        parser.error("--end-date is required with --start-date")
    return args


def _optional_polygon_adapter() -> PolygonAdapter | None:
    try:
        return PolygonAdapter(PolygonConfig.from_env())
    except ConfigError:
        return None


def main(argv: list[str] | None = None) -> int:
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
