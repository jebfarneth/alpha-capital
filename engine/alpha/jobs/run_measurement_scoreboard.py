#!/usr/bin/env python3
"""Read-only measurement scoreboard entrypoint.

Usage:
    cd engine
    uv run python -m alpha.jobs.run_measurement_scoreboard --live
    uv run python -m alpha.jobs.run_measurement_scoreboard --live --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from alpha.db.engine import get_session, reset_globals
from alpha.jobs.measurement_scoreboard import (
    DEFAULT_MFE_TAIL_THRESHOLD,
    ROLLUP_STATUS_BUCKETS,
    ScoreboardPartitionError,
    ScoreboardResult,
    build_measurement_scoreboard,
)
from alpha.jobs.forward_return import REQUIRED_FORWARD_RETURN_STATUSES
from alpha.runtime_env import load_runtime_env


def _run_live(args: argparse.Namespace) -> int:
    load_runtime_env()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if args.schema:
        os.environ["ALPHA_DB_SCHEMA"] = args.schema
        reset_globals()

    session = get_session()
    try:
        result = build_measurement_scoreboard(
            session,
            pattern_id=args.pattern_id,
            mfe_tail_threshold=args.mfe_tail_threshold,
        )
    except (ScoreboardPartitionError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        session.close()

    if args.json:
        print(json.dumps(result.to_dict(), sort_keys=True))
    else:
        _print_scoreboard(
            result,
            schema=args.schema or os.environ.get("ALPHA_DB_SCHEMA") or "default",
        )
    return 0


def _print_scoreboard(result: ScoreboardResult, *, schema: str) -> None:
    stats = result.computed_stats
    print("Status:                  finished")
    print(f"Schema:                  {schema}")
    print(f"Pattern:                 {result.pattern_id or 'all'}")
    print(f"MFE tail threshold:      {result.mfe_tail_threshold:.6g} (fraction)")
    print(f"Total firings:           {result.total_observations}")
    print("")
    print("Per-status counts:")
    for status in REQUIRED_FORWARD_RETURN_STATUSES:
        print(f"  {status}: {result.per_status_counts.get(status, 0)}")
    print("")
    print("Roll-up buckets:")
    for bucket in ROLLUP_STATUS_BUCKETS:
        print(f"  {bucket}: {result.rollup_counts.get(bucket, 0)}")
    print("")
    print("Anomalies:")
    print(f"  computed_missing_forward_return: {result.anomalies.computed_missing_forward_return}")
    print(f"  non_computed_with_forward_return: {result.anomalies.non_computed_with_forward_return}")
    print("")
    if stats.no_graded_firings:
        print("Graded stats:            no graded firings yet")
        return
    print("Graded stats:")
    print(f"  N:                     {stats.n}")
    print(f"  Expectancy:            {_format_optional(stats.expectancy)}")
    print(f"  Median:                {_format_optional(stats.median)}")
    print(f"  Best:                  {_format_optional(stats.best)}")
    print(f"  Worst:                 {_format_optional(stats.worst)}")
    print(f"  Win / flat / loss:     {stats.win_count} / {stats.flat_count} / {stats.loss_count}")
    print(f"  Avg win:               {_format_optional(stats.avg_win)}")
    print(f"  Avg loss:              {_format_optional(stats.avg_loss)}")
    print(f"  Win/loss ratio:        {_format_optional(stats.win_loss_ratio)}")
    print(f"  Hit T1:                {stats.hit_t1_count} ({_format_optional(stats.hit_t1_rate)})")
    print(f"  Hit T2:                {stats.hit_t2_count} ({_format_optional(stats.hit_t2_rate)})")
    print(f"  Hit T3:                {stats.hit_t3_count} ({_format_optional(stats.hit_t3_rate)})")
    print(f"  Hit stop:              {stats.hit_stop_count} ({_format_optional(stats.hit_stop_rate)})")
    print(f"  Same-day ambiguity:    {stats.same_day_barrier_ambiguity_count}")
    print(f"  MFE mean/median/max:   {_format_optional(stats.mfe_mean)} / {_format_optional(stats.mfe_median)} / {_format_optional(stats.mfe_max)}")
    print(f"  MAE mean/median/worst: {_format_optional(stats.mae_mean)} / {_format_optional(stats.mae_median)} / {_format_optional(stats.mae_worst)}")
    print(f"  Tail events:           {stats.tail_event_count} ({_format_optional(stats.tail_event_fraction)})")


def _format_optional(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print read-only measurement scoreboard over forward-return observations."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Read the configured database")
    parser.add_argument("--database-url", help="Override DATABASE_URL for this run")
    parser.add_argument(
        "--schema",
        help="PostgreSQL schema/search_path target for scratch read-only audits.",
    )
    parser.add_argument(
        "--pattern-id",
        default=None,
        help="Optional pattern id filter. Defaults to all patterns.",
    )
    parser.add_argument(
        "--mfe-tail-threshold",
        type=float,
        default=DEFAULT_MFE_TAIL_THRESHOLD,
        help="Tail-event MFE threshold as a fraction; 0.25 means +25%.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-formatted output.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if args.live:
        return _run_live(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
