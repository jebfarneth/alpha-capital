#!/usr/bin/env python3
"""Daily M2 production entrypoint."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

from alpha.data.config import FmpConfig, SecEdgarConfig
from alpha.data.edgar import SecEdgarAdapter
from alpha.data.fmp import FmpAdapter
from alpha.db.engine import (
    SchemaTargetError,
    create_all_tables,
    get_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.jobs.m2_daily import M2DailyAssemblyJob
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
    if not os.environ.get("SEC_USER_AGENT"):
        print("ERROR: SEC_USER_AGENT not set")
        return 1
    if not args.skip_fmp_enrichment and not os.environ.get("FMP_API_KEY"):
        print("ERROR: FMP_API_KEY not set; pass --skip-fmp-enrichment to run SEC-only")
        return 1

    session = get_session()
    if args.create_tables and not target_schema:
        create_all_tables()

    try:
        sec_adapter = SecEdgarAdapter(SecEdgarConfig.from_env())
        fmp_adapter = None if args.skip_fmp_enrichment else FmpAdapter(FmpConfig.from_env())
        job = M2DailyAssemblyJob(
            session=session,
            sec_adapter=sec_adapter,
            fmp_adapter=fmp_adapter,
            run_timestamp=_parse_timestamp(args.run_timestamp),
            form4_lookback_calendar_days=args.form4_lookback_calendar_days,
            fmp_page_limit=args.fmp_page_limit,
            skip_fmp_enrichment=args.skip_fmp_enrichment,
        )
        result = run_job(
            session,
            job,
            params={
                "source": "sec_form4_fmp_insider",
                "run_timestamp": args.run_timestamp,
                "form4_lookback_calendar_days": args.form4_lookback_calendar_days,
                "fmp_page_limit": args.fmp_page_limit,
                "skip_fmp_enrichment": args.skip_fmp_enrichment,
                "schema": args.schema,
            },
        )
        metrics = result.metrics or {}
        assembly = metrics.get("assembly") or {}
        orchestration = metrics.get("orchestration") or {}
        print(f"Status:                    {result.status}")
        print(f"Decision date:             {metrics.get('decision_date')}")
        print(f"Evidence session date:     {metrics.get('evidence_session_date')}")
        print(f"Schema:                    {args.schema or os.environ.get('ALPHA_DB_SCHEMA') or 'default'}")
        print(f"Included universe:         {metrics.get('included_universe_size')}")
        print(f"SEC Form 4 transactions:   {metrics.get('sec_transaction_count')}")
        print(f"FMP enrichment rows:       {metrics.get('fmp_enrichment_count')}")
        print(f"Unresolved issuer CIKs:    {metrics.get('unresolved_cik_count')}")
        print(f"M2 assembled inputs:       {(assembly.get('M2') or {}).get('assembled_count')}")
        print(f"M2U assembled inputs:      {(assembly.get('M2U') or {}).get('assembled_count')}")
        print(f"M2/M2U signals persisted:  {orchestration.get('total_signals_persisted')}")
        if result.errors:
            print("Errors:")
            for error in result.errors[:20]:
                print(f"  {error}")
        return 0 if result.ok else 1
    finally:
        session.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily M2 insider-cluster production wiring.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Run live SEC/FMP workflow")
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
            "Timezone-aware timestamp used for market-session resolution. "
            "Defaults to the evidence job run start time."
        ),
    )
    parser.add_argument(
        "--form4-lookback-calendar-days",
        type=int,
        default=1465,
        help="Calendar days of Form 4 history to fetch for CMP classification.",
    )
    parser.add_argument(
        "--fmp-page-limit",
        type=int,
        default=100,
        help="FMP insider-trading enrichment page size per ticker.",
    )
    parser.add_argument(
        "--skip-fmp-enrichment",
        action="store_true",
        help="Run SEC-only without FMP insider enrichment.",
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Create SQLAlchemy metadata tables before running. Production should use Alembic.",
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
