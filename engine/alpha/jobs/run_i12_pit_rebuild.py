#!/usr/bin/env python3
"""Guarded runner for the PIT-clean I12 research rebuild."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from alpha.data.alpaca import AlpacaAdapter
from alpha.data.config import AlpacaConfig, ConfigError, FmpConfig, PolygonConfig
from alpha.data.fmp import FmpAdapter
from alpha.data.polygon import PolygonAdapter
from alpha.db.engine import (
    SchemaTargetError,
    open_writable_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.jobs.i12_pit_rebuild import (
    DEFAULT_DECISION_TIMES,
    DEFAULT_INTENDED_ORDER_USD,
    DEFAULT_MAX_QUOTE_AGE_SECONDS,
    DEFAULT_MAX_SPREAD_BPS,
    DEFAULT_SLIPPAGE_BPS,
    I12PitRebuildJob,
    JOB_NAME,
    i12_pit_rebuild_report,
)
from alpha.jobs.runner import run_job
from alpha.runtime_env import load_runtime_env


I12_PIT_REBUILD_REQUIRED_TABLES = (
    "evidence_jobs",
    "evidence_job_runs",
    "i12_pit_candidates",
    "i12_pit_quote_replays",
    "i12_pit_cost_replays",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    load_runtime_env()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if args.schema:
        os.environ["ALPHA_DB_SCHEMA"] = args.schema
        reset_globals()

    target_schema = args.schema or os.environ.get("ALPHA_DB_SCHEMA")
    try:
        _validate_write_target(target_schema)
        prepare_writable_schema_target(
            schema=target_schema,
            create_tables=args.create_tables,
            required_tables=I12_PIT_REBUILD_REQUIRED_TABLES,
        )
    except (SchemaTargetError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    session = open_writable_session(schema=target_schema)
    try:
        if args.report_only:
            report = i12_pit_rebuild_report(
                session,
                source_hur_schema=args.source_hur_schema,
                decision_time_count=len(args.decision_time or list(DEFAULT_DECISION_TIMES)),
                start_date=_parse_date(args.start_date) if args.start_date else None,
                end_date=_parse_date(args.end_date) if args.end_date else None,
                job_run_id=args.job_run_id,
            )
            print(json.dumps(report, indent=2, sort_keys=True, default=str))
            _write_json_artifact(args.report_artifact, report)
            return 0
        try:
            fmp = FmpAdapter(FmpConfig.from_env())
            polygon = PolygonAdapter(PolygonConfig.from_env())
            alpaca = None if args.no_quote_replay else AlpacaAdapter(AlpacaConfig.from_env())
        except ConfigError as exc:
            print(f"ERROR: {exc}")
            return 1
        job = I12PitRebuildJob(
            session=session,
            fmp_adapter=fmp,
            polygon_adapter=polygon,
            alpaca_adapter=alpaca,
            start_date=_parse_date(args.start_date),
            end_date=_parse_date(args.end_date),
            decision_times=args.decision_time or list(DEFAULT_DECISION_TIMES),
            intended_order_usd=args.intended_order_usd,
            max_spread_bps=args.max_spread_bps,
            max_quote_age_seconds=args.max_quote_age_seconds,
            slippage_bps=args.slippage_bps,
            feed=args.feed,
            skip_existing=not args.replace_existing,
            replace_existing=args.replace_existing,
            quote_replay=not args.no_quote_replay,
            source_hur_schema=args.source_hur_schema,
            output_schema=target_schema,
            allow_source_hur_schema_matches_output=args.allow_source_hur_schema_matches_output,
            progress_artifact=args.progress_artifact,
        )
        result = run_job(
            session,
            job,
            params={
                "source": JOB_NAME,
                "schema": target_schema,
                "source_hur_schema": args.source_hur_schema,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "decision_times": args.decision_time or list(DEFAULT_DECISION_TIMES),
                "feed": args.feed,
                "quote_replay": not args.no_quote_replay,
                "intended_order_usd": args.intended_order_usd,
                "max_spread_bps": args.max_spread_bps,
                "max_quote_age_seconds": args.max_quote_age_seconds,
                "slippage_bps": args.slippage_bps,
                "replace_existing": args.replace_existing,
                "progress_artifact": args.progress_artifact,
            },
        )
        print(json.dumps(result.metrics or {}, indent=2, sort_keys=True, default=str))
        _write_json_artifact(args.report_artifact, result.metrics or {})
        return 0 if result.ok else 1
    finally:
        session.close()


def _validate_write_target(schema: str | None) -> None:
    normalized = (schema or "").strip().casefold()
    if not normalized:
        raise ValueError("I12 PIT rebuild requires --schema SCRATCH_SCHEMA")
    if normalized in {"public", "canonical", "main", "default"}:
        raise ValueError(f"I12 PIT rebuild refuses non-scratch schema: {schema}")


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _write_json_artifact(path_value: str | None, payload: dict) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PIT-clean I12 research candidates and scoped SIP quote replay."
    )
    parser.add_argument("--database-url")
    parser.add_argument("--schema", help="Named scratch schema for research writes.")
    parser.add_argument(
        "--source-hur-schema",
        default="public",
        help="Schema containing canonical historical_universe_reconstructions rows.",
    )
    parser.add_argument(
        "--allow-source-hur-schema-matches-output",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--create-tables", action="store_true")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--decision-time",
        action="append",
        default=None,
        help="Decision time in HH:MM ET; repeatable.",
    )
    parser.add_argument("--feed", default="sip", choices=("sip", "iex", "otc"))
    parser.add_argument("--intended-order-usd", type=float, default=DEFAULT_INTENDED_ORDER_USD)
    parser.add_argument("--max-spread-bps", type=float, default=DEFAULT_MAX_SPREAD_BPS)
    parser.add_argument("--max-quote-age-seconds", type=float, default=DEFAULT_MAX_QUOTE_AGE_SECONDS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Deliberately delete/replace matching PIT artifacts instead of reusing them.",
    )
    parser.add_argument("--no-quote-replay", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--job-run-id", help="Optional report-only filter.")
    parser.add_argument("--report-artifact")
    parser.add_argument("--progress-artifact")
    args = parser.parse_args(argv)
    if not args.report_only and (not args.start_date or not args.end_date):
        parser.error("--start-date and --end-date are required unless --report-only is set")
    if bool(args.start_date) != bool(args.end_date):
        parser.error("--start-date and --end-date must be provided together")
    if args.start_date and args.end_date and _parse_date(args.start_date) > _parse_date(args.end_date):
        parser.error("--start-date must be <= --end-date")
    if args.intended_order_usd <= 0:
        parser.error("--intended-order-usd must be positive")
    if args.max_spread_bps <= 0:
        parser.error("--max-spread-bps must be positive")
    if args.max_quote_age_seconds <= 0:
        parser.error("--max-quote-age-seconds must be positive")
    if args.slippage_bps < 0:
        parser.error("--slippage-bps must be >= 0")
    if (
        args.schema
        and args.source_hur_schema
        and args.schema.strip().casefold() == args.source_hur_schema.strip().casefold()
        and not args.allow_source_hur_schema_matches_output
    ):
        parser.error("--source-hur-schema must differ from --schema")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
