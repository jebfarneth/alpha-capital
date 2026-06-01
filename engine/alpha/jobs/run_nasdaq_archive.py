#!/usr/bin/env python3
"""Production Nasdaq listing self-archive entrypoint.

Usage:
    cd engine
    uv run python -m alpha.jobs.run_nasdaq_archive --live --run-timestamp 2026-06-01T18:15:00-04:00
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from alpha.data.contracts import AdapterResponse, ProviderError
from alpha.data.nasdaq import (
    ARCHIVE_REQUIRED_SOURCE_TYPES,
    NasdaqArchiveCaptureResult,
    NasdaqTraderListingAdapter,
)
from alpha.db.engine import (
    create_all_tables,
    create_schema_if_missing,
    get_session,
    reset_globals,
)
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.runner import run_job
from alpha.market_calendar import (
    EASTERN_TZ,
    is_us_equity_session,
    us_equity_session_close_timestamp,
)
from alpha.runtime_env import load_runtime_env

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_SLEEP_SECONDS = 5.0


class NasdaqArchiveJob(BaseJob):
    """Capture current Nasdaq Trader public listing-status sources."""

    def __init__(
        self,
        *,
        session: Any,
        adapter: NasdaqTraderListingAdapter,
        asof_timestamp: datetime,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_sleep_seconds: float = DEFAULT_RETRY_SLEEP_SECONDS,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if asof_timestamp.tzinfo is None or asof_timestamp.utcoffset() is None:
            raise ValueError("asof_timestamp must be timezone-aware")
        self._session = session
        self._adapter = adapter
        self._asof_timestamp = asof_timestamp.astimezone(timezone.utc)
        self._max_attempts = max_attempts
        self._retry_sleep_seconds = retry_sleep_seconds
        self._sleeper = sleeper

    @property
    def job_name(self) -> str:
        return "nasdaq_listing_archive"

    @property
    def job_type(self) -> str:
        return "source_archive"

    def run(self, ctx: JobContext) -> JobResult:
        attempts = []
        total_inserted_snapshots = 0
        total_existing_snapshots = 0
        total_inserted_rows = 0
        last_response: Optional[AdapterResponse[NasdaqArchiveCaptureResult]] = None

        for attempt_number in range(1, self._max_attempts + 1):
            response = self._adapter.archive_current_snapshot(
                self._session,
                asof=self._asof_timestamp,
            )
            last_response = response
            summary = _attempt_summary(attempt_number, response)
            attempts.append(summary)
            data = response.data
            if data is not None:
                total_inserted_snapshots += data.inserted_snapshots
                total_existing_snapshots += data.existing_snapshots
                total_inserted_rows += data.inserted_rows
            if _capture_response_is_complete(response):
                return JobResult(
                    status="finished",
                    metrics=_metrics(
                        asof_timestamp=self._asof_timestamp,
                        attempts=attempts,
                        final_response=response,
                        max_attempts=self._max_attempts,
                        total_inserted_snapshots=total_inserted_snapshots,
                        total_existing_snapshots=total_existing_snapshots,
                        total_inserted_rows=total_inserted_rows,
                    ),
                    output_hashes=dict(data.raw_payload_hashes if data else {}),
                )
            if attempt_number < self._max_attempts:
                summary["will_retry"] = True
                if self._retry_sleep_seconds > 0:
                    self._sleeper(self._retry_sleep_seconds)

        return JobResult(
            status="failed",
            metrics=_metrics(
                asof_timestamp=self._asof_timestamp,
                attempts=attempts,
                final_response=last_response,
                max_attempts=self._max_attempts,
                total_inserted_snapshots=total_inserted_snapshots,
                total_existing_snapshots=total_existing_snapshots,
                total_inserted_rows=total_inserted_rows,
            ),
            errors=[_failure_error(last_response)],
        )


def _capture_response_is_complete(
    response: AdapterResponse[NasdaqArchiveCaptureResult],
) -> bool:
    if not response.ok or response.data is None or response.data.failed_sources:
        return False
    return set(response.data.captured_sources).issuperset(ARCHIVE_REQUIRED_SOURCE_TYPES)


def _attempt_summary(
    attempt_number: int,
    response: AdapterResponse[NasdaqArchiveCaptureResult],
) -> dict[str, Any]:
    data = response.data
    error = response.error
    return {
        "attempt": attempt_number,
        "ok": response.ok,
        "captured_sources": list(data.captured_sources) if data else [],
        "failed_sources": list(data.failed_sources) if data else [],
        "inserted_snapshots": data.inserted_snapshots if data else 0,
        "existing_snapshots": data.existing_snapshots if data else 0,
        "inserted_rows": data.inserted_rows if data else 0,
        "raw_payload_hashes": dict(data.raw_payload_hashes) if data else {},
        "error": _provider_error_payload(error),
        "will_retry": False,
    }


def _metrics(
    *,
    asof_timestamp: datetime,
    attempts: list[dict[str, Any]],
    final_response: Optional[AdapterResponse[NasdaqArchiveCaptureResult]],
    max_attempts: int,
    total_inserted_snapshots: int,
    total_existing_snapshots: int,
    total_inserted_rows: int,
) -> dict[str, Any]:
    final_data = final_response.data if final_response else None
    final_error = final_response.error if final_response else None
    failed_sources = list(final_data.failed_sources) if final_data else []
    captured_sources = list(final_data.captured_sources) if final_data else []
    return {
        "source": "nasdaq_trader_listing_archive",
        "asof_timestamp": asof_timestamp.isoformat(),
        "archive_session_date": asof_timestamp.astimezone(EASTERN_TZ).date().isoformat(),
        "attempt_count": len(attempts),
        "max_attempts": max_attempts,
        "attempts": attempts,
        "captured_sources": captured_sources,
        "captured_source_count": len(captured_sources),
        "failed_sources": failed_sources,
        "failed_source_count": len(failed_sources),
        "inserted_snapshots": final_data.inserted_snapshots if final_data else 0,
        "existing_snapshots": final_data.existing_snapshots if final_data else 0,
        "inserted_rows": final_data.inserted_rows if final_data else 0,
        "total_inserted_snapshots": total_inserted_snapshots,
        "total_existing_snapshots": total_existing_snapshots,
        "total_inserted_rows": total_inserted_rows,
        "raw_payload_hashes": dict(final_data.raw_payload_hashes) if final_data else {},
        "provider_error": _provider_error_payload(final_error),
    }


def _provider_error_payload(error: Optional[ProviderError]) -> Optional[dict[str, Any]]:
    if error is None:
        return None
    return {
        "provider": error.provider,
        "endpoint": error.endpoint,
        "status_code": error.status_code,
        "error_type": error.error_type,
        "message": error.message,
        "retryable": error.retryable,
    }


def _failure_error(
    response: Optional[AdapterResponse[NasdaqArchiveCaptureResult]],
) -> dict[str, Any]:
    if response is None:
        return {"message": "Nasdaq archive capture did not run"}
    if response.error is not None:
        return {"message": "Nasdaq archive capture failed", "provider_error": _provider_error_payload(response.error)}
    data = response.data
    return {
        "message": "Nasdaq archive capture incomplete after retries",
        "failed_sources": list(data.failed_sources) if data else [],
    }


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("run_timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _resolve_run_timestamp(value: Optional[str]) -> tuple[datetime, bool]:
    parsed = _parse_timestamp(value)
    if parsed is not None:
        return parsed, True
    return _utcnow(), False


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _run_live(args: argparse.Namespace) -> int:
    load_runtime_env()
    try:
        run_timestamp, explicit_run_timestamp = _resolve_run_timestamp(args.run_timestamp)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    session_day = run_timestamp.astimezone(EASTERN_TZ).date()
    if not is_us_equity_session(session_day):
        message = (
            f"run_timestamp resolves to non-trading session date "
            f"{session_day.isoformat()}"
        )
        if explicit_run_timestamp:
            print(f"ERROR: {message}")
            return 1
        print("Status:                  finished")
        print("No-op reason:            non_trading_day")
        print(f"Run timestamp:           {run_timestamp.isoformat()}")
        print(f"Session date:            {session_day.isoformat()}")
        return 0

    asof_timestamp = us_equity_session_close_timestamp(session_day)
    if run_timestamp < asof_timestamp:
        print("Status:                  finished")
        print("No-op reason:            pre_session_close_skip")
        print(f"Run timestamp:           {run_timestamp.isoformat()}")
        print(f"Session date:            {session_day.isoformat()}")
        print(f"Session close:           {asof_timestamp.isoformat()}")
        return 0

    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if args.schema:
        os.environ["ALPHA_DB_SCHEMA"] = args.schema
        reset_globals()

    session = get_session()
    if args.create_tables:
        create_schema_if_missing(schema=args.schema)
        create_all_tables()

    try:
        adapter = NasdaqTraderListingAdapter()
        job = NasdaqArchiveJob(
            session=session,
            adapter=adapter,
            asof_timestamp=asof_timestamp,
            max_attempts=args.max_attempts,
            retry_sleep_seconds=args.retry_sleep_seconds,
        )
        result = run_job(
            session,
            job,
            params={
                "source": "nasdaq_trader_listing_archive",
                "run_timestamp": run_timestamp.isoformat(),
                "asof_timestamp": asof_timestamp.isoformat(),
                "archive_session_date": session_day.isoformat(),
                "max_attempts": args.max_attempts,
                "retry_sleep_seconds": args.retry_sleep_seconds,
                "schema": args.schema,
            },
        )
        metrics = result.metrics or {}
        print(f"Status:                  {result.status}")
        print(f"Schema:                  {args.schema or os.environ.get('ALPHA_DB_SCHEMA') or 'default'}")
        print(f"Run timestamp:           {run_timestamp.isoformat()}")
        print(f"Archive session date:    {metrics.get('archive_session_date')}")
        print(f"As-of timestamp:         {metrics.get('asof_timestamp')}")
        print(f"Attempts:                {metrics.get('attempt_count')}/{metrics.get('max_attempts')}")
        print(f"Captured sources:        {metrics.get('captured_sources')}")
        print(f"Failed sources:          {metrics.get('failed_sources')}")
        print(f"Inserted snapshots:      {metrics.get('inserted_snapshots')}")
        print(f"Existing snapshots:      {metrics.get('existing_snapshots')}")
        print(f"Inserted rows:           {metrics.get('inserted_rows')}")
        print(f"Raw payload hashes:      {metrics.get('raw_payload_hashes')}")
        if result.errors:
            print("Errors:")
            for error in result.errors[:20]:
                print(f"  {error}")
        return 0 if result.ok else 1
    finally:
        session.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nasdaq Trader listing self-archive capture.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Run live Nasdaq archive capture")
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
        help="Timezone-aware timestamp used to resolve the archive session date.",
    )
    parser.add_argument(
        "--max-attempts",
        type=_positive_int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum archive attempts before surfacing failure.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=_nonnegative_float,
        default=DEFAULT_RETRY_SLEEP_SECONDS,
        help="Seconds to sleep between retry attempts.",
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Create tables from SQLAlchemy metadata before running local smoke DBs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.live:
        return _run_live(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
