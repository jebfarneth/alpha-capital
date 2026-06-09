#!/usr/bin/env python3
"""FMP delisted-company ingestion entrypoint.

Examples:
    cd engine
    uv run python -m alpha.jobs.run_delisted_companies --live \
      --schema scratch_delisted_20260606 --create-tables --max-pages 5
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from typing import Optional

from alpha.data.config import ConfigError, FmpConfig
from alpha.data.fmp import FmpAdapter
from alpha.db.engine import (
    SchemaTargetError,
    create_all_tables,
    get_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.jobs.fmp_delisted_companies import FmpDelistedCompaniesIngestionJob
from alpha.jobs.contracts import JobResult
from alpha.jobs.runner import run_job
from alpha.runtime_env import load_runtime_env


DELISTED_REQUIRED_TABLES = (
    "evidence_jobs",
    "evidence_job_runs",
    "data_lineage",
    "fmp_delisted_companies",
)


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return date.fromisoformat(value)


def _run_live(args: argparse.Namespace) -> int:
    load_runtime_env()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if args.schema:
        os.environ["ALPHA_DB_SCHEMA"] = args.schema
        reset_globals()
    target_schema = args.schema or os.environ.get("ALPHA_DB_SCHEMA")
    if not target_schema and not args.confirm_live_write:
        print(
            "ERROR: refusing default-schema delisted-company ingestion without "
            "--confirm-live-write; use --schema for scratch proof"
        )
        return 1
    if target_schema is not None:
        try:
            prepare_writable_schema_target(
                schema=target_schema,
                create_tables=args.create_tables,
                required_tables=DELISTED_REQUIRED_TABLES,
            )
        except (SchemaTargetError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1

    try:
        fmp_adapter = FmpAdapter(FmpConfig.from_env())
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 1

    session = get_session()
    if args.create_tables and not target_schema:
        create_all_tables()

    try:
        job = FmpDelistedCompaniesIngestionJob(
            session=session,
            fmp_adapter=fmp_adapter,
            run_timestamp=_parse_timestamp(args.run_timestamp),
            page_limit=args.page_limit,
            max_pages=args.max_pages,
            stop_after_delisted_before=_parse_date(args.stop_after_delisted_before),
        )
        result = run_job(
            session,
            job,
            params={
                "source": "fmp_delisted_companies",
                "run_timestamp": args.run_timestamp,
                "page_limit": args.page_limit,
                "max_pages": args.max_pages,
                "stop_after_delisted_before": args.stop_after_delisted_before,
                "schema": args.schema,
                "confirm_live_write": args.confirm_live_write,
                "allow_partial": args.allow_partial,
            },
        )
        metrics = result.metrics or {}
        print(f"Status:                    {result.status}")
        print(f"Schema:                    {target_schema or 'default'}")
        print(f"Pages fetched:             {metrics.get('pages_fetched')}")
        print(f"Pages with data:           {metrics.get('pages_with_data')}")
        print(f"Rows seen:                 {metrics.get('rows_seen')}")
        print(f"Rows inserted:             {metrics.get('rows_inserted')}")
        print(f"Rows updated:              {metrics.get('rows_updated')}")
        print(f"Rows skipped:              {metrics.get('rows_skipped')}")
        print(f"Malformed rows:            {metrics.get('malformed_rows')}")
        print(f"U.S.-listed rows:          {metrics.get('us_listed_rows')}")
        print(f"Fetch errors:              {metrics.get('fetch_error_count')}")
        print(f"Max pages reached:         {metrics.get('max_pages_reached')}")
        print(f"Date cutoff reached:       {metrics.get('date_cutoff_reached')}")
        print(f"Oldest delisted date seen: {metrics.get('oldest_delisted_date_seen')}")
        if result.errors:
            print("Errors:")
            for error in result.errors[:20]:
                print(f"  {error}")
        return _exit_code_for_result(result, allow_partial=args.allow_partial)
    finally:
        session.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest FMP /stable/delisted-companies into durable storage."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Run live FMP ingestion")
    parser.add_argument("--database-url", help="Override DATABASE_URL for this run")
    parser.add_argument(
        "--schema",
        help=(
            "PostgreSQL schema/search_path target for scratch audits. "
            "Requires the schema to exist or --create-tables to create it."
        ),
    )
    parser.add_argument("--create-tables", action="store_true")
    parser.add_argument(
        "--confirm-live-write",
        action="store_true",
        help="Allow default-schema writes. Not needed for scratch --schema runs.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Return exit 0 for max-page truncation only. The persisted run status "
            "and metrics still show partial_failed and max_pages_reached=True."
        ),
    )
    parser.add_argument("--run-timestamp")
    parser.add_argument("--page-limit", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=1000)
    parser.add_argument(
        "--stop-after-delisted-before",
        help=(
            "Stop once FMP's descending delisted-date directory has moved before "
            "this date. Use for bounded replay windows, e.g. 2026-01-01."
        ),
    )
    return parser.parse_args(argv)


def _exit_code_for_result(result: JobResult, *, allow_partial: bool = False) -> int:
    if result.ok:
        return 0
    metrics = result.metrics or {}
    if (
        allow_partial
        and result.status == "partial_failed"
        and metrics.get("max_pages_reached") is True
        and not metrics.get("fetch_error_count")
    ):
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if args.live:
        return _run_live(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
