#!/usr/bin/env python3
"""Guarded I12 historical corpus runner.

This runner is scratch-first. Writing to the public/default schema requires
``--confirm-live-write`` and is intentionally deferred until external audit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from alpha.data.config import ConfigError, FmpConfig, PolygonConfig
from alpha.data.fmp import FmpAdapter
from alpha.data.polygon import PolygonAdapter
from alpha.db.engine import (
    SchemaTargetError,
    create_all_tables,
    open_writable_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.jobs.i12_historical_corpus import (
    DEFAULT_FETCH_DEADLINE_SECONDS,
    DEFAULT_MAX_CONSECUTIVE_FETCH_TIMEOUTS,
    DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS,
    I12HistoricalCorpusJob,
    JOB_NAME,
)
from alpha.jobs.run_market_path_backfill import CachedHistoricalPriceFmpAdapter
from alpha.jobs.runner import run_job
from alpha.runtime_env import load_runtime_env


I12_CORPUS_REQUIRED_TABLES = [
    "evidence_datasets",
    "evidence_jobs",
    "evidence_job_runs",
    "evidence_snapshots",
    "data_lineage",
    "universe_scans",
    "universe_snapshots",
    "feature_snapshots",
    "signal_registry",
    "forward_return_observations",
    "intraday_event_details",
]
DEFAULT_MINUTE_CACHE_DIR = Path.home() / ".cache" / "alpha-capital" / "i12_polygon_minute_aggs"


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
        try:
            prepare_writable_schema_target(
                schema=target_schema,
                create_tables=args.create_tables,
                required_tables=I12_CORPUS_REQUIRED_TABLES,
            )
        except (SchemaTargetError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1
    elif args.create_tables:
        create_all_tables()

    try:
        fmp_adapter = CachedHistoricalPriceFmpAdapter(FmpAdapter(FmpConfig.from_env()))
        polygon_adapter = PolygonAdapter(PolygonConfig.from_env())
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 1

    progress_artifact = Path(args.progress_artifact) if args.progress_artifact else None
    artifact: dict[str, Any] = {
        "job": JOB_NAME,
        "started_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "events": [],
    }

    def progress(event: str, payload: Mapping[str, Any]) -> None:
        record = {
            "event": event,
            "payload": dict(payload),
            "at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        }
        print(
            "PROGRESS "
            + " ".join(
                f"{key}={value}"
                for key, value in {"event": event, **dict(payload)}.items()
                if key != "metrics"
            )
        )
        artifact["events"].append(record)
        artifact["last_event"] = record
        if "ticker" in record["payload"] and "trading_date" in record["payload"]:
            artifact["last_ticker_day_activity"] = {
                "event": event,
                "ticker": record["payload"]["ticker"],
                "trading_date": record["payload"]["trading_date"],
                "at": record["payload"].get("wall_clock_utc", record["at"]),
            }
        if progress_artifact is not None:
            progress_artifact.parent.mkdir(parents=True, exist_ok=True)
            progress_artifact.write_text(json.dumps(artifact, indent=2, default=str))

    session = open_writable_session(schema=target_schema)
    job = I12HistoricalCorpusJob(
        session=session,
        fmp_adapter=fmp_adapter,
        polygon_adapter=polygon_adapter,
        start_date=_parse_date(args.start_date),
        end_date=_parse_date(args.end_date),
        run_timestamp=_parse_timestamp(args.run_timestamp),
        batch_days=args.batch_days,
        minute_cache_dir=args.minute_cache_dir,
        polygon_rate_limit_per_minute=args.polygon_rate_limit_per_minute,
        skip_existing=args.skip_existing,
        max_db_retries=args.max_db_retries,
        db_retry_backoff_seconds=args.db_retry_backoff_seconds,
        fetch_deadline_seconds=args.fetch_deadline_seconds,
        max_outstanding_fetch_timeouts=args.max_outstanding_fetch_timeouts,
        max_consecutive_fetch_timeouts=args.max_consecutive_fetch_timeouts,
        progress_callback=progress,
    )
    try:
        result = run_job(
            session,
            job,
            params={
                "source": JOB_NAME,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "schema": target_schema,
                "confirm_live_write": args.confirm_live_write,
                "batch_days": args.batch_days,
                "minute_cache_dir": args.minute_cache_dir,
                "polygon_cache_dir": args.minute_cache_dir,
                "polygon_rate_limit_per_minute": args.polygon_rate_limit_per_minute,
                "skip_existing": args.skip_existing,
                "max_db_retries": args.max_db_retries,
                "db_retry_backoff_seconds": args.db_retry_backoff_seconds,
                "fetch_deadline_seconds": args.fetch_deadline_seconds,
                "max_outstanding_fetch_timeouts": args.max_outstanding_fetch_timeouts,
                "max_consecutive_fetch_timeouts": args.max_consecutive_fetch_timeouts,
                "progress_artifact": str(progress_artifact) if progress_artifact else None,
            },
        )
    finally:
        session.close()
    artifact["ended_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    artifact["result"] = {
        "status": result.status,
        "metrics": result.metrics,
        "errors": result.errors,
    }
    if progress_artifact is not None:
        progress_artifact.write_text(json.dumps(artifact, indent=2, default=str))
    print(json.dumps(result.metrics, indent=2, sort_keys=True, default=str))
    return 0 if result.ok else 1


def _validate_write_target(*, schema: str | None, confirm_live_write: bool) -> None:
    normalized = (schema or "").strip().lower()
    if (schema is None or normalized == "public") and not confirm_live_write:
        raise ValueError(
            "Refusing public/default I12 corpus write without --confirm-live-write"
        )


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the durable historical I12 corpus build.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--database-url")
    parser.add_argument("--schema")
    parser.add_argument("--create-tables", action="store_true")
    parser.add_argument("--confirm-live-write", action="store_true")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--run-timestamp")
    parser.add_argument("--batch-days", type=int, default=10)
    parser.add_argument(
        "--minute-cache-dir",
        "--polygon-cache-dir",
        dest="minute_cache_dir",
        default=str(DEFAULT_MINUTE_CACHE_DIR),
        help="Disk cache directory for Polygon adjusted minute bars.",
    )
    parser.add_argument(
        "--polygon-rate-limit-per-minute",
        type=int,
        default=300,
        help="Maximum uncached Polygon minute fetches per minute.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip ticker-days that already have an I12 intraday_event_details row.",
    )
    parser.add_argument("--max-db-retries", type=int, default=3)
    parser.add_argument("--db-retry-backoff-seconds", type=float, default=5.0)
    parser.add_argument(
        "--fetch-deadline-seconds",
        type=float,
        default=DEFAULT_FETCH_DEADLINE_SECONDS,
        help="Wall-clock deadline for one uncached Polygon minute fetch.",
    )
    parser.add_argument(
        "--max-outstanding-fetch-timeouts",
        type=int,
        default=DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS,
        help="Abort the shard when this many timed-out fetch workers remain outstanding.",
    )
    parser.add_argument(
        "--max-consecutive-fetch-timeouts",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_FETCH_TIMEOUTS,
        help="Abort the shard after this many consecutive provider watchdog timeouts.",
    )
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
