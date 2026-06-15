#!/usr/bin/env python3
"""Guarded I11 historical corpus runner.

This runner is scratch-first. Public/default writes are hard-refused until
I12 public corpus completion plus external I11 pilot audit clears the
sequencing gate; ``--confirm-live-write`` is intentionally inert for public.
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
from alpha.jobs.i11_historical_corpus import I11HistoricalCorpusJob, JOB_NAME
from alpha.jobs.i12_historical_corpus import (
    DEFAULT_FETCH_DEADLINE_SECONDS,
    DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS,
)
from alpha.jobs.run_market_path_backfill import CachedHistoricalPriceFmpAdapter
from alpha.jobs.runner import run_job
from alpha.runtime_env import load_runtime_env


I11_CORPUS_REQUIRED_TABLES = [
    "evidence_jobs",
    "evidence_job_runs",
    "data_lineage",
    "feature_snapshots",
    "signal_registry",
    "intraday_event_details",
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
        _validate_write_target(
            schema=target_schema,
            confirm_live_write=args.confirm_live_write,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    if target_schema is not None:
        try:
            required_tables = list(I11_CORPUS_REQUIRED_TABLES)
            if args.at_open:
                required_tables.append("forward_return_observations")
            prepare_writable_schema_target(
                schema=target_schema,
                create_tables=args.create_tables,
                required_tables=required_tables,
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
    job = I11HistoricalCorpusJob(
        session=session,
        fmp_adapter=fmp_adapter,
        polygon_adapter=polygon_adapter,
        start_date=_parse_date(args.start_date),
        end_date=_parse_date(args.end_date),
        run_timestamp=_parse_timestamp(args.run_timestamp),
        batch_days=args.batch_days,
        minute_cache_dir=args.polygon_cache_dir,
        polygon_rate_limit_per_minute=args.polygon_rate_limit_per_minute,
        skip_existing=args.skip_existing,
        max_db_retries=args.max_db_retries,
        db_retry_backoff_seconds=args.db_retry_backoff_seconds,
        fetch_deadline_seconds=args.fetch_deadline_seconds,
        max_outstanding_fetch_timeouts=args.max_outstanding_fetch_timeouts,
        progress_callback=progress,
        catalyst_tags_by_ticker_date=_load_catalyst_tags_artifact(args.catalyst_tags_artifact),
        at_open=args.at_open,
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
                "polygon_cache_dir": args.polygon_cache_dir,
                "polygon_rate_limit_per_minute": args.polygon_rate_limit_per_minute,
                "skip_existing": args.skip_existing,
                "max_db_retries": args.max_db_retries,
                "db_retry_backoff_seconds": args.db_retry_backoff_seconds,
                "fetch_deadline_seconds": args.fetch_deadline_seconds,
                "max_outstanding_fetch_timeouts": args.max_outstanding_fetch_timeouts,
                "progress_artifact": str(progress_artifact) if progress_artifact else None,
                "catalyst_tags_artifact": args.catalyst_tags_artifact,
                "at_open": args.at_open,
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
    if schema is None or normalized == "public":
        raise ValueError(
            "Refusing public/default I11 corpus write until pilot audit and public sequencing gates clear"
        )


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _load_catalyst_tags_artifact(
    path: str | None,
) -> dict[tuple[str, date], list[str]]:
    if not path:
        return {}
    with Path(path).open("r") as f:
        raw = json.load(f)
    out: dict[tuple[str, date], list[str]] = {}
    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                raise ValueError("catalyst tag rows must be objects")
            ticker = str(row.get("ticker") or row.get("symbol") or "").upper()
            day_raw = row.get("trading_date") or row.get("date")
            tags = row.get("tags") or row.get("catalyst_tags") or []
            _add_catalyst_tags(out, ticker=ticker, day_raw=day_raw, tags=tags)
        return out
    if isinstance(raw, dict):
        for ticker, by_date in raw.items():
            if not isinstance(by_date, dict):
                raise ValueError("dict catalyst artifact values must be date maps")
            for day_raw, tags in by_date.items():
                _add_catalyst_tags(out, ticker=str(ticker).upper(), day_raw=day_raw, tags=tags)
        return out
    raise ValueError("catalyst tag artifact must be a list or dict")


def _add_catalyst_tags(
    out: dict[tuple[str, date], list[str]],
    *,
    ticker: str,
    day_raw: Any,
    tags: Any,
) -> None:
    if not ticker:
        raise ValueError("catalyst tag artifact row missing ticker")
    day = _parse_date(str(day_raw))
    if isinstance(tags, str):
        normalized = [tags]
    elif isinstance(tags, list):
        normalized = [str(tag) for tag in tags]
    else:
        raise ValueError("catalyst tags must be a string or list")
    out[(ticker.upper(), day)] = sorted({
        tag.strip()
        for tag in normalized
        if tag.strip()
    })


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the durable historical I11 corpus build.")
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
        "--polygon-cache-dir",
        default=".cache/i11_polygon_minute_aggs",
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
        help="Skip ticker-days that already have an I11 intraday_event_details row.",
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
        "--at-open",
        action="store_true",
        help="Build the widened I11 at-open early-entry corpus instead of the confirmed-entry corpus.",
    )
    parser.add_argument("--progress-artifact")
    parser.add_argument(
        "--catalyst-tags-artifact",
        help=(
            "Optional JSON artifact mapping ticker/date to premarket catalyst tags "
            "such as offering or NT-late-filer."
        ),
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
