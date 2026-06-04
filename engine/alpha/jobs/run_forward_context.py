#!/usr/bin/env python3
"""Production forward-context panel entrypoint.

Usage:
    cd engine
    uv run python -m alpha.jobs.run_forward_context --live --run-timestamp 2026-06-02T18:15:00-04:00
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

from alpha.data.benzinga import BenzingaAdapter
from alpha.data.config import BenzingaConfig, ConfigError, PolygonConfig
from alpha.data.polygon import PolygonAdapter
from alpha.db.engine import (
    SchemaTargetError,
    create_all_tables,
    get_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.jobs.forward_context import ForwardContextCollectorJob
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

    try:
        polygon_adapter = _required_polygon_adapter()
        benzinga_adapter = _required_benzinga_adapter()
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 1

    session = get_session()
    if args.create_tables and not target_schema:
        create_all_tables()

    try:
        job = ForwardContextCollectorJob(
            session=session,
            polygon_adapter=polygon_adapter,
            benzinga_adapter=benzinga_adapter,
            run_timestamp=_parse_timestamp(args.run_timestamp),
            pattern_id=args.pattern_id,
        )
        result = run_job(
            session,
            job,
            params={
                "source": "forward_context_panel",
                "run_timestamp": args.run_timestamp,
                "forward_session_date": args.forward_session_date,
                "pattern_id": args.pattern_id,
                "schema": args.schema,
            },
        )
        metrics = result.metrics or {}
        print(f"Status:                  {result.status}")
        print(f"Pattern:                 {metrics.get('pattern_id')}")
        print(f"Schema:                  {args.schema or os.environ.get('ALPHA_DB_SCHEMA') or 'default'}")
        print(f"Decision date:           {metrics.get('decision_date')}")
        print(f"Forward session date:    {metrics.get('forward_session_date')}")
        print(f"As-of timestamp:         {metrics.get('asof_timestamp')}")
        print(f"Active signals:          {metrics.get('active_signal_count')}")
        print(f"Eligible signals:        {metrics.get('eligible_signal_count')}")
        print(f"Rows inserted:           {metrics.get('rows_inserted')}")
        print(f"Rows already present:    {metrics.get('rows_existing')}")
        print(f"Ticker provider pulls:   {metrics.get('ticker_fetch_count')}")
        if metrics.get("no_op_reason"):
            print(f"No-op reason:            {metrics.get('no_op_reason')}")
        if result.errors:
            print("Errors:")
            for error in result.errors[:20]:
                print(f"  {error}")
        return 0 if result.ok else 1
    finally:
        session.close()


def _required_polygon_adapter() -> PolygonAdapter:
    return PolygonAdapter(PolygonConfig.from_env())


def _required_benzinga_adapter() -> BenzingaAdapter:
    return BenzingaAdapter(BenzingaConfig.from_env())


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run forward-context panel collection."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Run live context capture")
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
            "Timezone-aware timestamp used for session resolution. Defaults to "
            "the evidence job run start time."
        ),
    )
    parser.add_argument(
        "--forward-session-date",
        help=(
            "Optional audited session override (YYYY-MM-DD). Must not be in "
            "the future or past relative to the run timestamp's evidence session."
        ),
    )
    parser.add_argument(
        "--pattern-id",
        default="M4",
        help="Pattern id to collect. First production slice supports M4.",
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Create tables from SQLAlchemy metadata before running local smoke DBs.",
    )
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
