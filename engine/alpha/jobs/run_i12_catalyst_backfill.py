#!/usr/bin/env python3
"""Guarded runner for the I12 catalyst retag backfill."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from alpha.data.benzinga import BenzingaAdapter
from alpha.data.config import BenzingaConfig, ConfigError, PolygonConfig, SecEdgarConfig
from alpha.data.edgar import SecEdgarAdapter
from alpha.data.polygon import PolygonAdapter
from alpha.db.engine import (
    create_all_tables,
    open_writable_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.jobs.i12_catalyst_backfill import I12CatalystBackfillJob, JOB_NAME
from alpha.jobs.i12_catalysts import I12CatalystResolver
from alpha.jobs.i12_historical_corpus import DEFAULT_FETCH_DEADLINE_SECONDS
from alpha.jobs.runner import run_job
from alpha.jobs.watchdog import (
    DEFAULT_MAX_CONSECUTIVE_FETCH_TIMEOUTS,
    DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS,
    WatchdogState,
)
from alpha.runtime_env import load_runtime_env


I12_CATALYST_REQUIRED_TABLES = [
    "evidence_jobs",
    "evidence_job_runs",
    "feature_snapshots",
    "signal_registry",
    "intraday_event_details",
    "security_identity_snapshots",
]
DEFAULT_EDGAR_CACHE_DIR = Path.home() / ".cache" / "alpha-capital" / "i12_edgar_submissions"


def _run_live(args: argparse.Namespace) -> int:
    load_runtime_env()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if args.schema:
        os.environ["ALPHA_DB_SCHEMA"] = args.schema
        reset_globals()
    target_schema = args.schema or os.environ.get("ALPHA_DB_SCHEMA")
    try:
        _validate_write_target(
            schema=target_schema,
            confirm_live_write=args.confirm_live_write,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    if target_schema is not None:
        prepare_writable_schema_target(
            schema=target_schema,
            create_tables=args.create_tables,
            required_tables=I12_CATALYST_REQUIRED_TABLES,
        )
    elif args.create_tables:
        create_all_tables()

    try:
        edgar_adapter = SecEdgarAdapter(SecEdgarConfig.from_env())
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 1
    polygon_news_adapter = None
    if not args.disable_polygon_news:
        try:
            polygon_news_adapter = PolygonAdapter(PolygonConfig.from_env())
        except ConfigError:
            polygon_news_adapter = None
    benzinga_news_adapter = None
    if not args.disable_benzinga_news:
        try:
            benzinga_news_adapter = BenzingaAdapter(BenzingaConfig.from_env())
        except ConfigError:
            benzinga_news_adapter = None

    session = open_writable_session(schema=target_schema)
    watchdog = WatchdogState(
        max_outstanding_timeouts=args.max_outstanding_fetch_timeouts,
        max_consecutive_timeouts=args.max_consecutive_fetch_timeouts,
    )
    resolver = I12CatalystResolver(
        session=session,
        edgar_adapter=edgar_adapter,
        polygon_news_adapter=polygon_news_adapter,
        benzinga_news_adapter=benzinga_news_adapter,
        edgar_cache_dir=args.edgar_cache_dir,
        lookback_days=args.lookback_days,
        fetch_deadline_seconds=args.fetch_deadline_seconds,
        watchdog_state=watchdog,
    )
    job = I12CatalystBackfillJob(
        session=session,
        catalyst_resolver=resolver,
        start_date=_parse_date(args.start_date),
        end_date=_parse_date(args.end_date),
        skip_existing=args.skip_existing,
        batch_size=args.batch_size,
        progress_artifact=args.progress_artifact,
    )
    try:
        result = run_job(
            session,
            job,
            params={
                "source": JOB_NAME,
                "schema": target_schema,
                "confirm_live_write": args.confirm_live_write,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "skip_existing": args.skip_existing,
                "batch_size": args.batch_size,
                "edgar_cache_dir": args.edgar_cache_dir,
                "lookback_days": args.lookback_days,
                "fetch_deadline_seconds": args.fetch_deadline_seconds,
                "max_outstanding_fetch_timeouts": args.max_outstanding_fetch_timeouts,
                "max_consecutive_fetch_timeouts": args.max_consecutive_fetch_timeouts,
                "polygon_news_enabled": polygon_news_adapter is not None,
                "benzinga_news_enabled": benzinga_news_adapter is not None,
                "progress_artifact": args.progress_artifact,
            },
        )
    finally:
        session.close()
    print(f"I12 catalyst backfill {result.status}")
    print(result.metrics)
    if result.errors:
        print(result.errors)
    return 0 if result.status == "finished" else 1


def _validate_write_target(*, schema: str | None, confirm_live_write: bool) -> None:
    if schema is None and not confirm_live_write:
        raise ValueError(
            "public/default schema writes require --confirm-live-write; "
            "use --schema for scratch backfills"
        )


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retag existing I12 rows with PIT catalyst features.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--database-url")
    parser.add_argument("--schema")
    parser.add_argument("--create-tables", action="store_true")
    parser.add_argument("--confirm-live-write", action="store_true")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--edgar-cache-dir", default=str(DEFAULT_EDGAR_CACHE_DIR))
    parser.add_argument("--lookback-days", type=int, default=5)
    parser.add_argument(
        "--fetch-deadline-seconds",
        type=float,
        default=DEFAULT_FETCH_DEADLINE_SECONDS,
        help="Wall-clock deadline for one EDGAR submissions fetch.",
    )
    parser.add_argument(
        "--max-outstanding-fetch-timeouts",
        type=int,
        default=DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS,
    )
    parser.add_argument(
        "--max-consecutive-fetch-timeouts",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_FETCH_TIMEOUTS,
    )
    parser.add_argument("--disable-polygon-news", action="store_true")
    parser.add_argument("--disable-benzinga-news", action="store_true")
    parser.add_argument("--progress-artifact")
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
