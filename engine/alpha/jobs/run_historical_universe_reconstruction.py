#!/usr/bin/env python3
"""Historical PIT universe reconstruction entrypoint."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from typing import Optional

from alpha.db.engine import (
    SchemaTargetError,
    prepare_writable_schema_target,
    get_session,
    reset_globals,
)
from alpha.jobs.historical_universe_reconstruction import (
    HistoricalUniverseReconstructionJob,
)
from alpha.jobs.runner import run_job
from alpha.runtime_env import load_runtime_env


HISTORICAL_UNIVERSE_REQUIRED_TABLES = (
    "evidence_jobs",
    "evidence_job_runs",
    "data_lineage",
    "security_profiles",
    "universe_scans",
    "canonical_universe_scans",
    "universe_snapshots",
    "fmp_delisted_companies",
    "historical_universe_reconstructions",
)


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_date(value: str) -> date:
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
    if not target_schema:
        print("ERROR: historical universe reconstruction requires --schema")
        return 1
    if target_schema.strip().casefold() == "public":
        print("ERROR: historical universe reconstruction refuses the public schema")
        return 1
    try:
        prepare_writable_schema_target(
            schema=target_schema,
            create_tables=args.create_tables,
            required_tables=HISTORICAL_UNIVERSE_REQUIRED_TABLES,
        )
    except (SchemaTargetError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    session = get_session()
    try:
        job = HistoricalUniverseReconstructionJob(
            session=session,
            replay_date=_parse_date(args.replay_date),
            run_timestamp=_parse_timestamp(args.run_timestamp),
            allow_partial_delisted_source=args.allow_partial_delisted_source,
        )
        result = run_job(
            session,
            job,
            params={
                "source": "historical_universe_reconstruction",
                "replay_date": args.replay_date,
                "run_timestamp": args.run_timestamp,
                "schema": target_schema,
                "allow_partial_delisted_source": args.allow_partial_delisted_source,
            },
        )
        metrics = result.metrics or {}
        print(f"Status:                 {result.status}")
        print(f"Schema:                 {target_schema}")
        print(f"Replay date:            {metrics.get('replay_date')}")
        print(f"Candidates:             {metrics.get('candidate_count')}")
        print(f"Rows inserted:          {metrics.get('rows_inserted')}")
        print(f"Rows updated:           {metrics.get('rows_updated')}")
        print(f"Included:               {metrics.get('included_count')}")
        print(f"Excluded:               {metrics.get('excluded_count')}")
        print(f"Rejection reasons:      {metrics.get('rejection_reason_counts')}")
        print(f"Sources:                {metrics.get('source_counts')}")
        print(f"Delisted source complete: {metrics.get('delisted_source_complete')}")
        print(f"Delisted partial reason:  {metrics.get('delisted_source_partial_reason')}")
        if result.errors:
            print("Errors:")
            for error in result.errors[:20]:
                print(f"  {error}")
        return 0 if result.ok else 1
    finally:
        session.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct one historical PIT operating-universe date."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--database-url")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--create-tables", action="store_true")
    parser.add_argument("--replay-date", required=True)
    parser.add_argument("--run-timestamp")
    parser.add_argument(
        "--allow-partial-delisted-source",
        action="store_true",
        help=(
            "Permit reconstruction to exit successfully when the FMP delisted "
            "directory source is known partial; metrics still stamp the partial source."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if args.live:
        return _run_live(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
