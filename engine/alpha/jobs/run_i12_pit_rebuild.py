#!/usr/bin/env python3
"""Guarded runner for the PIT-clean I12 research rebuild."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import inspect

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
    DEFAULT_FETCH_DEADLINE_SECONDS,
    DEFAULT_INTENDED_ORDER_USD,
    DEFAULT_MAX_QUOTE_AGE_SECONDS,
    DEFAULT_MAX_SPREAD_BPS,
    DEFAULT_SLIPPAGE_BPS,
    I12PitRebuildJob,
    JOB_NAME,
    MINUTE_PATH_MODES,
    STRICT_MINUTE_PATH_MODE,
    i12_pit_rebuild_report,
)
from alpha.jobs.runner import run_job
from alpha.jobs.watchdog import (
    DEFAULT_MAX_CONSECUTIVE_FETCH_TIMEOUTS,
    DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS,
)
from alpha.runtime_env import load_runtime_env


I12_PIT_REBUILD_REQUIRED_TABLES = (
    "evidence_jobs",
    "evidence_job_runs",
    "i12_pit_candidates",
    "i12_pit_quote_replays",
    "i12_pit_cost_replays",
)

I12_PIT_REBUILD_REQUIRED_COLUMNS = {
    "i12_pit_candidates": (
        "path_mode",
        "candidate_attempt_hash",
        "is_active",
        "superseded_at",
        "superseded_by_candidate_id",
        "candidate_identity_hash",
        "label_hash",
    ),
    "i12_pit_quote_replays": (
        "quote_replay_attempt_hash",
        "is_active",
        "superseded_at",
        "superseded_by_quote_replay_id",
        "bid_notional",
        "ask_notional",
        "executable_notional",
        "executable_side",
        "quote_size_basis",
    ),
    "i12_pit_cost_replays": (
        "cost_replay_attempt_hash",
        "is_active",
        "superseded_at",
        "superseded_by_cost_replay_id",
    ),
}

I12_PIT_REBUILD_REQUIRED_INDEXES = {
    "i12_pit_candidates": ("ux_i12_pit_candidates_active_attempt",),
    "i12_pit_quote_replays": ("ux_i12_pit_quote_replays_active_attempt",),
    "i12_pit_cost_replays": ("ux_i12_pit_cost_replays_active_attempt",),
}


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
        try:
            _assert_required_pit_columns(session, target_schema)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 1
        if args.preflight_only:
            print(json.dumps({
                "preflight": "ok",
                "schema": target_schema,
                "minute_path_mode": args.minute_path_mode,
            }, indent=2, sort_keys=True))
            return 0
        if args.report_only:
            report = i12_pit_rebuild_report(
                session,
                source_hur_schema=args.source_hur_schema,
                decision_time_count=len(args.decision_time or list(DEFAULT_DECISION_TIMES)),
                decision_time_labels=args.decision_time or list(DEFAULT_DECISION_TIMES),
                start_date=_parse_date(args.start_date) if args.start_date else None,
                end_date=_parse_date(args.end_date) if args.end_date else None,
                job_run_id=args.job_run_id,
                path_mode=None if args.compare_path_modes else args.minute_path_mode,
                compare_path_modes=args.compare_path_modes,
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
            minute_path_mode=args.minute_path_mode,
            skip_existing=not args.replace_existing,
            replace_existing=args.replace_existing,
            quote_replay=not args.no_quote_replay,
            source_hur_schema=args.source_hur_schema,
            output_schema=target_schema,
            allow_source_hur_schema_matches_output=args.allow_source_hur_schema_matches_output,
            progress_artifact=args.progress_artifact,
            fetch_deadline_seconds=args.fetch_deadline_seconds,
            max_outstanding_fetch_timeouts=args.max_outstanding_fetch_timeouts,
            max_consecutive_fetch_timeouts=args.max_consecutive_fetch_timeouts,
            max_no_progress_seconds=(
                args.max_no_progress_minutes * 60
                if args.max_no_progress_minutes > 0
                else None
            ),
            no_progress_exit_callback=(
                _exit_on_no_progress if args.max_no_progress_minutes > 0 else None
            ),
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
                "minute_path_mode": args.minute_path_mode,
                "quote_replay": not args.no_quote_replay,
                "intended_order_usd": args.intended_order_usd,
                "max_spread_bps": args.max_spread_bps,
                "max_quote_age_seconds": args.max_quote_age_seconds,
                "slippage_bps": args.slippage_bps,
                "replace_existing": args.replace_existing,
                "progress_artifact": args.progress_artifact,
                "fetch_deadline_seconds": args.fetch_deadline_seconds,
                "max_outstanding_fetch_timeouts": args.max_outstanding_fetch_timeouts,
                "max_consecutive_fetch_timeouts": args.max_consecutive_fetch_timeouts,
                "max_no_progress_minutes": args.max_no_progress_minutes,
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


def _assert_required_pit_columns(session, schema: str | None) -> None:
    inspector = inspect(session.get_bind())
    for table_name, required_columns in I12_PIT_REBUILD_REQUIRED_COLUMNS.items():
        try:
            columns = {
                column["name"]
                for column in inspector.get_columns(table_name, schema=schema)
            }
        except Exception:
            columns = {column["name"] for column in inspector.get_columns(table_name)}
        missing = [column for column in required_columns if column not in columns]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                f"schema {schema} has old {table_name} table without {missing_text}; "
                "create a fresh scratch schema or migrate it"
            )
    for table_name, required_indexes in I12_PIT_REBUILD_REQUIRED_INDEXES.items():
        try:
            indexes = {
                index["name"]
                for index in inspector.get_indexes(table_name, schema=schema)
            }
        except Exception:
            indexes = {index["name"] for index in inspector.get_indexes(table_name)}
        missing = [index for index in required_indexes if index not in indexes]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                f"schema {schema} has old {table_name} table without index "
                f"{missing_text}; create a fresh scratch schema or migrate it"
            )


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _write_json_artifact(path_value: str | None, payload: dict) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _exit_on_no_progress(payload: Mapping[str, Any]) -> None:
    print(
        "ERROR: I12 PIT rebuild no-progress watchdog fired: "
        + json.dumps(payload, sort_keys=True, default=str),
        file=sys.stderr,
        flush=True,
    )
    os._exit(70)


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
    parser.add_argument(
        "--minute-path-mode",
        default=STRICT_MINUTE_PATH_MODE,
        choices=MINUTE_PATH_MODES,
        help="Minute path policy for candidate construction.",
    )
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
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Create/verify scratch schema and PIT tables, then exit without providers.",
    )
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--compare-path-modes",
        action="store_true",
        help="Report strict and sparse path modes together with comparison finality.",
    )
    parser.add_argument("--job-run-id", help="Optional report-only filter.")
    parser.add_argument("--report-artifact")
    parser.add_argument("--progress-artifact")
    parser.add_argument(
        "--fetch-deadline-seconds",
        type=float,
        default=DEFAULT_FETCH_DEADLINE_SECONDS,
        help="Hard wall-clock deadline for one provider fetch.",
    )
    parser.add_argument(
        "--max-outstanding-fetch-timeouts",
        type=int,
        default=DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS,
        help="Open the provider outage circuit after this many abandoned timed-out fetch workers.",
    )
    parser.add_argument(
        "--max-consecutive-fetch-timeouts",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_FETCH_TIMEOUTS,
        help="Open the provider outage circuit after this many consecutive watchdog timeouts.",
    )
    parser.add_argument(
        "--max-no-progress-minutes",
        type=float,
        default=20.0,
        help=(
            "Exit nonzero if no progress artifact event occurs for this many "
            "minutes during normal rebuild runs; 0 disables the shard-level "
            "no-progress monitor. Report-only and preflight-only modes do not "
            "start the monitor."
        ),
    )
    args = parser.parse_args(argv)
    if (
        not args.report_only
        and not args.preflight_only
        and (not args.start_date or not args.end_date)
    ):
        parser.error(
            "--start-date and --end-date are required unless --report-only "
            "or --preflight-only is set"
        )
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
    if args.fetch_deadline_seconds <= 0:
        parser.error("--fetch-deadline-seconds must be > 0")
    if args.max_outstanding_fetch_timeouts < 1:
        parser.error("--max-outstanding-fetch-timeouts must be >= 1")
    if args.max_consecutive_fetch_timeouts < 1:
        parser.error("--max-consecutive-fetch-timeouts must be >= 1")
    if args.max_no_progress_minutes < 0:
        parser.error("--max-no-progress-minutes must be >= 0")
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
