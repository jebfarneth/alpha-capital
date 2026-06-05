#!/usr/bin/env python3
"""M3 sector-history backfill / forward-capture entrypoint."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

from alpha.data.config import FmpConfig, PolygonConfig
from alpha.data.fmp import FmpAdapter
from alpha.data.polygon import PolygonAdapter
from alpha.db.engine import (
    SchemaTargetError,
    create_all_tables,
    get_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.jobs.m3_sector_history import M3SectorHistoryJob
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
    if not os.environ.get("POLYGON_API_KEY"):
        print("ERROR: POLYGON_API_KEY not set")
        return 1
    if not args.no_fmp_fallback and not os.environ.get("FMP_API_KEY"):
        print("ERROR: FMP_API_KEY not set; pass --no-fmp-fallback to run Polygon-only")
        return 1

    session = get_session()
    if args.create_tables and not target_schema:
        create_all_tables()
    try:
        polygon_adapter = PolygonAdapter(PolygonConfig.from_env())
        fmp_adapter = None if args.no_fmp_fallback else FmpAdapter(FmpConfig.from_env())
        mode = "backfill" if args.backfill else "forward_capture"
        job = M3SectorHistoryJob(
            session,
            polygon_adapter=polygon_adapter,
            fmp_adapter=fmp_adapter,
            run_timestamp=_parse_timestamp(args.run_timestamp),
            mode=mode,
            lookback_years=args.lookback_years,
            sample_frequency_days=args.sample_frequency_days,
        )
        result = run_job(
            session,
            job,
            params={
                "source": "polygon_sic_asof",
                "run_timestamp": args.run_timestamp,
                "mode": mode,
                "lookback_years": args.lookback_years,
                "sample_frequency_days": args.sample_frequency_days,
                "schema": args.schema,
            },
        )
        metrics = result.metrics or {}
        print(f"Status:                    {result.status}")
        print(f"Mode:                      {metrics.get('mode')}")
        print(f"Decision date:             {metrics.get('decision_date')}")
        print(f"Ticker count:              {metrics.get('ticker_count')}")
        print(f"As-of date count:          {metrics.get('asof_date_count')}")
        print(f"Resolved assignments:      {metrics.get('resolved_assignment_count')}")
        print(f"Polygon SIC assignments:   {metrics.get('polygon_sic_assignment_count')}")
        print(f"FMP fallback assignments:  {metrics.get('fmp_fallback_assignment_count')}")
        print(f"Sector unknown:            {metrics.get('sector_unknown_count')}")
        print(f"Sector changes:            {metrics.get('sector_change_count')}")
        if result.errors:
            print("Errors:")
            for error in result.errors[:20]:
                print(f"  {error}")
        return 0 if result.ok else 1
    finally:
        session.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M3 sector history pipeline.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backfill", action="store_true")
    mode.add_argument("--forward-capture", action="store_true")
    parser.add_argument("--database-url")
    parser.add_argument("--schema")
    parser.add_argument("--run-timestamp")
    parser.add_argument("--lookback-years", type=float, default=3.0)
    parser.add_argument(
        "--sample-frequency-days",
        type=int,
        default=1,
        help="Backfill as-of sampling cadence; default 1 preserves exact daily intervals.",
    )
    parser.add_argument("--no-fmp-fallback", action="store_true")
    parser.add_argument("--create-tables", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    previous_env = {
        "DATABASE_URL": os.environ.get("DATABASE_URL"),
        "ALPHA_DB_SCHEMA": os.environ.get("ALPHA_DB_SCHEMA"),
    }
    try:
        args = _parse_args(argv or sys.argv[1:])
        return _run_live(args)
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_globals()


if __name__ == "__main__":
    raise SystemExit(main())
