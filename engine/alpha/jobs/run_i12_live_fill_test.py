#!/usr/bin/env python3
"""Run the read-only I12 Stage-0 live fill-test machine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime
from typing import Any, Optional

from alpha.data.alpaca import AlpacaAdapter
from alpha.data.config import AlpacaConfig, ConfigError
from alpha.db.engine import (
    SchemaTargetError,
    open_writable_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.jobs.i12_live_fill_test import (
    DEFAULT_INTENDED_ORDER_USD,
    DEFAULT_MAX_SPREAD_BPS,
    DEFAULT_TOP_K,
    I12LiveFillConfig,
    I12LiveFillTestJob,
    capture_i12_exit_quotes,
    i12_gate0_report,
)
from alpha.jobs.paper_execution import load_premarket_context_artifact
from alpha.jobs.runner import run_job
from alpha.runtime_env import load_runtime_env


I12_FILL_TEST_REQUIRED_TABLES = (
    "evidence_jobs",
    "evidence_job_runs",
    "feature_snapshots",
    "signal_registry",
    "signal_ml_scores",
    "i12_fill_log",
)


def _run(args: argparse.Namespace) -> int:
    load_runtime_env()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if args.schema:
        os.environ["ALPHA_DB_SCHEMA"] = args.schema
        reset_globals()

    try:
        alpaca = AlpacaAdapter(AlpacaConfig.from_env())
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.alpaca_probe_symbol:
        return _run_quote_probe(alpaca, args)

    try:
        schema = require_stage0_scratch_schema(args.schema or os.environ.get("ALPHA_DB_SCHEMA"))
    except SchemaTargetError as exc:
        print(f"ERROR: {exc}")
        return 1
    try:
        prepare_writable_schema_target(
            schema=schema,
            create_tables=args.create_tables,
            required_tables=I12_FILL_TEST_REQUIRED_TABLES,
        )
        session = open_writable_session(schema=schema)
    except (SchemaTargetError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        if args.exit_quotes:
            result = capture_i12_exit_quotes(session, alpaca, feed=args.feed)
            print(json.dumps(result, sort_keys=True))
            return 0 if "error" not in result else 1

        if args.gate0_report:
            report = i12_gate0_report(
                session,
                decision_date=_parse_date(args.trading_date) if args.trading_date else None,
                max_spread_bps=args.max_spread_bps,
                intended_order_usd=args.intended_order_usd,
            )
            print(json.dumps(report, sort_keys=True))
            return 0

        trading_date = _parse_date(args.trading_date) if args.trading_date else date.today()
        contexts = load_premarket_context_artifact(
            args.context_artifact,
            expected_date=trading_date,
        )
        config = I12LiveFillConfig(
            model_id=args.model_id,
            allow_latest_model=args.allow_latest_model,
            top_k=args.top_k,
            intended_order_usd=args.intended_order_usd,
            max_spread_bps=args.max_spread_bps,
            feed=args.feed,
            require_market_open=not args.allow_market_closed,
        )
        while True:
            job = I12LiveFillTestJob(
                session=session,
                alpaca_adapter=alpaca,
                contexts=contexts,
                config=config,
                asof=_parse_datetime(args.asof) if args.asof else None,
            )
            result = run_job(
                session,
                job,
                params={
                    "trading_date": trading_date.isoformat(),
                    "top_k": args.top_k,
                    "intended_order_usd": args.intended_order_usd,
                    "max_spread_bps": args.max_spread_bps,
                    "feed": args.feed,
                    "read_only": True,
                    "schema": schema,
                },
            )
            print(json.dumps(_job_result_payload(result), sort_keys=True))
            if args.once:
                return 0 if result.status == "finished" else 1
            time.sleep(args.poll_interval_seconds)
    finally:
        session.close()


def _run_quote_probe(alpaca: AlpacaAdapter, args: argparse.Namespace) -> int:
    response = alpaca.get_latest_quote(args.alpaca_probe_symbol, feed=args.feed)
    if not response.ok or response.data is None:
        print(json.dumps({"ok": False, "error": str(response.error)}, sort_keys=True))
        return 1
    quote = response.data
    print(
        json.dumps(
            {
                "ok": True,
                "symbol": quote.symbol,
                "bid": quote.bid_price,
                "ask": quote.ask_price,
                "bid_size": quote.bid_size,
                "ask_size": quote.ask_size,
                "timestamp": quote.timestamp,
                "read_only": True,
            },
            sort_keys=True,
        )
    )
    return 0


def require_stage0_scratch_schema(schema: str | None) -> str:
    if not schema:
        raise SchemaTargetError("Stage-0 fill-test persistence requires --schema SCRATCH_SCHEMA")
    if schema.strip().casefold() == "public":
        raise SchemaTargetError("Stage-0 fill-test persistence refuses public schema")
    return schema


def _job_result_payload(result: Any) -> dict[str, Any]:
    return {
        "status": result.status,
        "metrics": result.metrics or {},
        "errors": result.errors or [],
    }


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_datetime(value: str) -> datetime:
    raw = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("--asof must be timezone-aware")
    return parsed


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the scratch-bound, read-only I12 Stage-0 fill test."
    )
    parser.add_argument("--database-url")
    parser.add_argument("--schema", help="Required scratch schema for any DB write.")
    parser.add_argument("--create-tables", action="store_true")
    parser.add_argument("--context-artifact")
    parser.add_argument("--trading-date")
    parser.add_argument("--model-id")
    parser.add_argument("--allow-latest-model", action="store_true")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--intended-order-usd", type=float, default=DEFAULT_INTENDED_ORDER_USD)
    parser.add_argument("--max-spread-bps", type=float, default=DEFAULT_MAX_SPREAD_BPS)
    parser.add_argument("--feed", default="iex", choices=("iex", "sip", "otc"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=float, default=60.0)
    parser.add_argument("--allow-market-closed", action="store_true")
    parser.add_argument("--asof")
    parser.add_argument("--exit-quotes", action="store_true")
    parser.add_argument("--gate0-report", action="store_true")
    parser.add_argument(
        "--alpaca-probe-symbol",
        help="Read-only quote probe; does not require --schema or write to DB.",
    )
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.intended_order_usd <= 0:
        parser.error("--intended-order-usd must be positive")
    if args.max_spread_bps <= 0:
        parser.error("--max-spread-bps must be positive")
    if args.poll_interval_seconds <= 0:
        parser.error("--poll-interval-seconds must be positive")
    if args.alpaca_probe_symbol:
        return args
    if not args.exit_quotes and not args.gate0_report and not args.context_artifact:
        parser.error("--context-artifact is required unless --exit-quotes or --gate0-report")
    if not args.exit_quotes and not args.gate0_report:
        if not args.model_id and not args.allow_latest_model:
            parser.error("--model-id is required unless --allow-latest-model is set")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    return _run(_parse_args(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
