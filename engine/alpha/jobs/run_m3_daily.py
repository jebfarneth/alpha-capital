#!/usr/bin/env python3
"""Daily M3 production entrypoint."""

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
from alpha.jobs.m3_daily import M3DailyAssemblyJob
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
    if not os.environ.get("FMP_API_KEY"):
        print("ERROR: FMP_API_KEY not set")
        return 1

    session = get_session()
    if args.create_tables and not target_schema:
        create_all_tables()
    try:
        job = M3DailyAssemblyJob(
            session,
            polygon_adapter=PolygonAdapter(PolygonConfig.from_env()),
            fmp_adapter=FmpAdapter(FmpConfig.from_env()),
            run_timestamp=_parse_timestamp(args.run_timestamp),
            sector_lookback_sessions=args.sector_lookback_sessions,
            refresh_sector_history=not args.skip_sector_refresh,
        )
        result = run_job(
            session,
            job,
            params={
                "source": "polygon_sic_fmp_sector_returns",
                "run_timestamp": args.run_timestamp,
                "sector_lookback_sessions": args.sector_lookback_sessions,
                "skip_sector_refresh": args.skip_sector_refresh,
                "schema": args.schema,
            },
        )
        metrics = result.metrics or {}
        assembly = metrics.get("assembly") or {}
        orchestration = metrics.get("orchestration") or {}
        refresh = metrics.get("sector_history_refresh") or {}
        print(f"Status:                    {result.status}")
        print(f"Decision date:             {metrics.get('decision_date')}")
        print(f"Evidence session date:     {metrics.get('evidence_session_date')}")
        print(f"Schema:                    {args.schema or os.environ.get('ALPHA_DB_SCHEMA') or 'default'}")
        print(f"Formation date:            {metrics.get('formation_date')}")
        print(f"Formation universe:        {metrics.get('formation_universe_size')}")
        print(f"Current sector assignments:{metrics.get('current_sector_assignment_count')}")
        print(f"Sector returns:            {metrics.get('sector_return_count')}")
        print(f"Sector refresh unknown:    {refresh.get('unknown')}")
        print(f"Assembled M3 inputs:       {assembly.get('assembled_count')}")
        print(f"M3 signals persisted:      {orchestration.get('total_signals_persisted')}")
        if result.errors:
            print("Errors:")
            for error in result.errors[:20]:
                print(f"  {error}")
        return 0 if result.ok else 1
    finally:
        session.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily M3 sector-rotation production wiring.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Run live Polygon/FMP workflow")
    parser.add_argument("--database-url")
    parser.add_argument("--schema")
    parser.add_argument("--run-timestamp")
    parser.add_argument("--sector-lookback-sessions", type=int, default=126)
    parser.add_argument("--skip-sector-refresh", action="store_true")
    parser.add_argument("--create-tables", action="store_true")
    return parser.parse_args(argv)


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
