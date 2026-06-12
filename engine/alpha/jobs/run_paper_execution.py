#!/usr/bin/env python3
"""Run the pattern-agnostic paper-trading intraday loop.

The loop uses Polygon delayed snapshots for entry decisions and wall clock for
exit submission. A current-date premarket context artifact is required before
polling starts; this prevents accidental live runs with stale daily context.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from typing import Any, Callable, Mapping, Optional

from alpha.data.alpaca import AlpacaAdapter
from alpha.data.config import AlpacaConfig, ConfigError, PolygonConfig
from alpha.data.polygon import PolygonAdapter
from alpha.db.engine import create_all_tables, get_session, reset_globals
from alpha.jobs.paper_execution import (
    FatalBrokerAuthError,
    PaperExecutionConfig,
    PaperTradingLoop,
    PremarketContextBuilder,
    default_pattern_registry,
    load_premarket_context_artifact,
    validate_paper_base_url,
)
from alpha.runtime_env import load_runtime_env


def _run(args: argparse.Namespace) -> int:
    load_runtime_env()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if args.create_tables:
        create_all_tables()

    try:
        polygon = PolygonAdapter(PolygonConfig.from_env())
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 1

    trading_date = date.fromisoformat(args.trading_date) if args.trading_date else date.today()
    if args.build_context:
        builder = PremarketContextBuilder(polygon)
        contexts = builder.build(
            context_date=trading_date,
            output_path=args.context_artifact,
        )
        print(f"Context artifact: {args.context_artifact}")
        print(f"Context date:     {trading_date.isoformat()}")
        print(f"Tickers:          {len(contexts)}")
        return 0

    try:
        alpaca_config = AlpacaConfig.from_env()
        validate_paper_base_url(
            alpaca_config.base_url,
            confirm_live_trade=args.confirm_live_trade,
        )
    except (ConfigError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    contexts = load_premarket_context_artifact(
        args.context_artifact,
        expected_date=trading_date,
    )
    registry = default_pattern_registry()
    plugins = registry.selected(args.pattern_id)
    session = get_session()
    alpaca_adapter = AlpacaAdapter(alpaca_config)
    loop = PaperTradingLoop(
        session=session,
        alpaca_adapter=alpaca_adapter,
        config=PaperExecutionConfig(
            notional=args.notional,
            max_concurrent_positions=args.max_concurrent_positions,
            max_new_entries_per_day=args.max_new_entries_per_day,
            dry_run=not args.paper_trade,
            paper_trade=args.paper_trade,
            confirm_live_trade=args.confirm_live_trade,
        ),
        plugins=plugins,
    )
    try:
        loop.reconcile_startup()
        return _run_polling_loop(
            loop=loop,
            polygon=polygon,
            alpaca_adapter=alpaca_adapter,
            contexts=contexts,
            poll_interval_seconds=args.poll_interval_seconds,
            once=args.once,
            max_consecutive_snapshot_failures=args.max_consecutive_snapshot_failures,
        )
    except FatalBrokerAuthError as exc:
        print(f"ERROR: broker auth failed; stopping loop: {exc}")
        return 1
    finally:
        session.close()


def _run_polling_loop(
    *,
    loop: Any,
    polygon: Any,
    alpaca_adapter: Any,
    contexts: Mapping[str, Any],
    poll_interval_seconds: float,
    once: bool,
    max_consecutive_snapshot_failures: int,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    consecutive_snapshot_failures = 0
    while True:
        clock_resp = alpaca_adapter.get_clock()
        if clock_resp.error is not None and clock_resp.error.error_type == "auth":
            raise FatalBrokerAuthError(clock_resp.error.message)
        if not clock_resp.ok or clock_resp.data is None:
            print(f"WARNING: Alpaca clock unavailable; skipping entries: {clock_resp.error}")
            loop.submit_due_exits()
            if once:
                return 0
            sleep_fn(poll_interval_seconds)
            continue
        if not clock_resp.data.is_open:
            exits_submitted = loop.submit_due_exits()
            print(f"MARKET_CLOSED exits_submitted={exits_submitted}")
            if once:
                return 0
            sleep_fn(poll_interval_seconds)
            continue

        snapshot_resp = polygon.get_full_market_snapshot()
        if not snapshot_resp.ok:
            consecutive_snapshot_failures += 1
            print(
                "WARNING: Polygon snapshot failed "
                f"consecutive={consecutive_snapshot_failures}/"
                f"{max_consecutive_snapshot_failures}: {snapshot_resp.error}"
            )
            loop.submit_due_exits()
            if consecutive_snapshot_failures >= max_consecutive_snapshot_failures:
                loop.submit_due_exits()
                print("ERROR: Polygon snapshot failure threshold reached; stopping loop")
                return 1
            sleep_fn(poll_interval_seconds)
            continue

        consecutive_snapshot_failures = 0
        counters = loop.run_snapshot_poll(
            snapshots=snapshot_resp.data or [],
            contexts=contexts,
            lineage_hash=snapshot_resp.lineage.raw_payload_hash,
        )
        print(
            "POLL "
            f"snapshots={counters['snapshots']} "
            f"candidate_confirmed={counters['candidate_confirmed']} "
            f"orders_submitted={counters['orders_submitted']} "
            f"skipped={counters['skipped']}"
        )
        if once:
            return 0
        sleep_fn(poll_interval_seconds)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run I-track paper execution loop.")
    parser.add_argument("--database-url")
    parser.add_argument("--create-tables", action="store_true")
    parser.add_argument("--context-artifact", required=True)
    parser.add_argument("--trading-date")
    parser.add_argument("--build-context", action="store_true")
    parser.add_argument(
        "--pattern-id",
        action="append",
        default=[],
        help="Pattern id to enable. Repeat for multiple patterns. Defaults to I12 and I11.",
    )
    parser.add_argument("--paper-trade", action="store_true")
    parser.add_argument("--confirm-live-trade", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=float, default=60.0)
    parser.add_argument("--max-consecutive-snapshot-failures", type=int, default=10)
    parser.add_argument("--notional", type=float, default=250.0)
    parser.add_argument("--max-concurrent-positions", type=int, default=4)
    parser.add_argument("--max-new-entries-per-day", type=int, default=4)
    args = parser.parse_args(argv)
    if args.poll_interval_seconds <= 0:
        parser.error("--poll-interval-seconds must be positive")
    if args.max_consecutive_snapshot_failures < 1:
        parser.error("--max-consecutive-snapshot-failures must be >= 1")
    if args.paper_trade and args.build_context:
        parser.error("--paper-trade cannot be combined with --build-context")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    return _run(_parse_args(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
