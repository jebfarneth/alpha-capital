#!/usr/bin/env python3
"""Guarded M4 pre-signal context backfill runner.

This runner is scratch-first. Public/default writes are hard-refused until the
scratch pilot, monthly cost rehearsal, and dual audit gates clear.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from alpha.data.config import ConfigError, FmpConfig
from alpha.data.fmp import FmpAdapter
from alpha.db.engine import (
    SchemaTargetError,
    create_all_tables,
    get_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.jobs.historical_m4_signal_selector import (
    SIGNAL_SOURCE_CHOICES,
)
from alpha.jobs.market_path_pre_signal_context import (
    DEFAULT_LOOKBACK_CALENDAR_DAYS,
    DEFAULT_PRE_SIGNAL_WINDOW,
    FEATURE_VERSION,
    JOB_NAME,
    MarketPathPreSignalContextJob,
)
from alpha.jobs.run_market_path_backfill import CachedHistoricalPriceFmpAdapter
from alpha.jobs.run_market_path_bulk_backfill import (
    DEFAULT_REQUEST_RETRIES,
    RetryingHistoricalPriceFmpAdapter,
    _TimeoutRequestsSession,
)
from alpha.jobs.runner import run_job
from alpha.runtime_env import load_runtime_env


PRE_SIGNAL_REQUIRED_TABLES = [
    "evidence_jobs",
    "evidence_job_runs",
    "data_lineage",
    "feature_snapshots",
    "signal_registry",
    "historical_universe_reconstructions",
    "market_path_pre_signal_contexts",
    "market_path_pre_signal_links",
]


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
        _validate_write_target(schema=target_schema, confirm_live_write=args.confirm_live_write)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    if target_schema is not None:
        try:
            prepare_writable_schema_target(
                schema=target_schema,
                create_tables=args.create_tables,
                required_tables=PRE_SIGNAL_REQUIRED_TABLES,
            )
        except (SchemaTargetError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1
    elif args.create_tables:
        create_all_tables()

    try:
        fmp_adapter = RetryingHistoricalPriceFmpAdapter(
            CachedHistoricalPriceFmpAdapter(
                FmpAdapter(
                    FmpConfig.from_env(),
                    session=_TimeoutRequestsSession(args.request_timeout_seconds),
                )
            ),
            max_retries=DEFAULT_REQUEST_RETRIES,
            request_timeout_seconds=args.request_timeout_seconds,
        )
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
        if progress_artifact is not None:
            progress_artifact.parent.mkdir(parents=True, exist_ok=True)
            progress_artifact.write_text(json.dumps(artifact, indent=2, default=str))

    session = get_session()
    job = MarketPathPreSignalContextJob(
        session=session,
        fmp_adapter=fmp_adapter,
        pattern_ids=args.pattern_id or ["M4"],
        signal_start_date=_parse_date(args.signal_start_date),
        signal_end_date=_parse_date(args.signal_end_date),
        run_timestamp=_parse_timestamp(args.run_timestamp),
        pre_signal_window=args.pre_signal_window,
        batch_days=args.batch_days,
        lookback_calendar_days=args.lookback_calendar_days,
        feature_version=args.feature_version,
        signal_source=args.signal_source,
        progress_callback=progress,
        progress_artifact=progress_artifact,
        progress_every=args.progress_every,
    )
    try:
        result = run_job(
            session,
            job,
            params={
                "source": JOB_NAME,
                "pattern_id": args.pattern_id or ["M4"],
                "signal_start_date": args.signal_start_date,
                "signal_end_date": args.signal_end_date,
                "schema": target_schema,
                "confirm_live_write": args.confirm_live_write,
                "pre_signal_window": args.pre_signal_window,
                "batch_days": args.batch_days,
                "lookback_calendar_days": args.lookback_calendar_days,
                "feature_version": args.feature_version,
                "signal_source": args.signal_source,
                "progress_artifact": str(progress_artifact) if progress_artifact else None,
                "request_timeout_seconds": args.request_timeout_seconds,
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
    if not normalized or normalized == "public":
        raise ValueError(
            "Refusing public/default pre-signal context write until scratch rehearsal and audit gates clear"
        )


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M4 pre-signal context backfill.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--database-url")
    parser.add_argument("--schema")
    parser.add_argument("--create-tables", action="store_true")
    parser.add_argument("--confirm-live-write", action="store_true")
    parser.add_argument("--run-timestamp")
    parser.add_argument(
        "--pattern-id",
        action="append",
        default=[],
        help="Pattern id to backfill. Repeat for multiple patterns.",
    )
    parser.add_argument("--signal-start-date", required=True)
    parser.add_argument("--signal-end-date", required=True)
    parser.add_argument("--pre-signal-window", type=int, default=DEFAULT_PRE_SIGNAL_WINDOW)
    parser.add_argument("--batch-days", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--lookback-calendar-days",
        type=int,
        default=DEFAULT_LOOKBACK_CALENDAR_DAYS,
    )
    parser.add_argument("--feature-version", default=FEATURE_VERSION)
    parser.add_argument(
        "--signal-source",
        choices=SIGNAL_SOURCE_CHOICES,
        required=True,
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
