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
from typing import Dict, List, Optional

from alpha.db.engine import get_session, reset_globals
from alpha.jobs.measurement_scoreboard import (
    DEFAULT_MFE_TAIL_THRESHOLD,
    ROLLUP_STATUS_BUCKETS,
    ScoreboardDirectionError,
    ScoreboardPatternIntegrityError,
    ScoreboardPartitionError,
    ScoreboardPoolingError,
    ScoreboardResult,
    ScoreboardSignalIntegrityError,
    ScoreboardWindowIntegrityError,
    build_measurement_scoreboard,
    build_measurement_scoreboard_by_horizon,
)
from alpha.jobs.forward_return import REQUIRED_FORWARD_RETURN_STATUSES
from alpha.runtime_env import load_runtime_env

TAIL_THRESHOLD_WARNING = (
    "WARNING: interpreting --mfe-tail-threshold as a fraction; "
    "25 == +2500% - pass 0.25 for +25%"
)


def _run_live(args: argparse.Namespace) -> int:
    previous_env = {
        "DATABASE_URL": os.environ.get("DATABASE_URL"),
        "ALPHA_DB_SCHEMA": os.environ.get("ALPHA_DB_SCHEMA"),
    }
    session = None
    load_runtime_env()
    try:
        if args.database_url:
            os.environ["DATABASE_URL"] = args.database_url
            reset_globals()
        if args.schema:
            os.environ["ALPHA_DB_SCHEMA"] = args.schema
            reset_globals()
        display_schema = args.schema or os.environ.get("ALPHA_DB_SCHEMA") or "default"
        warning = _tail_threshold_warning(args.mfe_tail_threshold)
        if warning:
            print(warning, file=sys.stderr)
        if args.group_by_horizon and args.signal_horizon is not None:
            raise ValueError("--group-by-horizon and --signal-horizon are mutually exclusive")

        session = get_session()
        if args.group_by_horizon:
            result = build_measurement_scoreboard_by_horizon(
                session,
                pattern_id=args.pattern_id,
                mfe_tail_threshold=args.mfe_tail_threshold,
            )
        else:
            result = build_measurement_scoreboard(
                session,
                pattern_id=args.pattern_id,
                signal_horizon=args.signal_horizon,
                mfe_tail_threshold=args.mfe_tail_threshold,
            )
    except (
        ScoreboardPartitionError,
        ScoreboardDirectionError,
        ScoreboardSignalIntegrityError,
        ScoreboardPatternIntegrityError,
        ScoreboardWindowIntegrityError,
        ScoreboardPoolingError,
        ValueError,
    ) as exc:
        if args.json:
            print(json.dumps(_error_payload(exc), sort_keys=True))
            return 1
        _print_error(exc)
        return 1
    finally:
        if session is not None:
            session.close()
        _restore_env(previous_env)
        reset_globals()

    if args.json:
        print(json.dumps(result.to_dict(), sort_keys=True))
    else:
        if args.group_by_horizon:
            _print_scoreboard_by_horizon(result, schema=display_schema)
        else:
            _print_scoreboard(result, schema=display_schema)
    return 0


def _print_scoreboard(result: ScoreboardResult, *, schema: str) -> None:
    stats = result.computed_stats
    print("Status:                  finished")
    print(f"Schema:                  {schema}")
    print(f"Pattern:                 {result.pattern_id or 'all'}")
    print(f"MFE tail threshold:      {result.mfe_tail_threshold:.6g} (fraction)")
    print(f"Total firings:           {result.total_observations}")
    print(f"Raw observation rows:    {result.raw_observation_rows}")
    print(f"Stale duplicate rows:    {result.stale_duplicate_observation_rows}")
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
    print(f"  graded_missing_excursion: {result.anomalies.graded_missing_excursion_count}")
    print(
        "  graded reconciliation: "
        f"{result.graded_rollup_reconciliation.graded_rollup_count} = "
        f"{result.graded_rollup_reconciliation.computed_sample_n} + "
        f"{result.graded_rollup_reconciliation.computed_missing_forward_return}"
    )
    print("")
    if stats.no_graded_firings:
        print("Graded stats:            no graded firings yet")
        return
    print("Graded stats:")
    print(f"  N:                     {stats.n}")
    print(f"  Graded firings:        {stats.total_firings}")
    print(f"  Distinct tickers:      {stats.distinct_tickers}")
    print(f"  Overlap firings:       {stats.overlapping_window_firings}")
    print(f"  Max same-ticker conc:  {stats.max_concurrent_same_ticker}")
    print(f"  Effective sample size: {_format_optional(stats.effective_sample_size)}")
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
    print(f"  Finite MFE / MAE N:    {stats.mfe_finite_count} / {stats.mae_finite_count}")
    print(f"  MFE mean/median/max:   {_format_optional(stats.mfe_mean)} / {_format_optional(stats.mfe_median)} / {_format_optional(stats.mfe_max)}")
    print(f"  MAE mean/median/worst: {_format_optional(stats.mae_mean)} / {_format_optional(stats.mae_median)} / {_format_optional(stats.mae_worst)}")
    print(f"  Tail denominator:      {stats.tail_event_denominator}")
    print(f"  Tail events:           {stats.tail_event_count} ({_format_optional(stats.tail_event_fraction)})")


def _format_optional(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _print_scoreboard_by_horizon(result, *, schema: str) -> None:
    print("Status:                  finished")
    print(f"Schema:                  {schema}")
    print(f"Pattern:                 {result.pattern_id or 'all'}")
    print(f"MFE tail threshold:      {result.mfe_tail_threshold:.6g} (fraction)")
    print("")
    print("Graded stats by horizon:")
    if not result.computed_stats_by_horizon:
        print("  no graded firings yet")
        return
    for horizon, stats in result.computed_stats_by_horizon.items():
        print(f"  {horizon}:")
        print(f"    N:                   {stats.n}")
        print(f"    Expectancy:          {_format_optional(stats.expectancy)}")
        print(f"    Finite MFE / MAE N:  {stats.mfe_finite_count} / {stats.mae_finite_count}")
        print(f"    Tail denominator:    {stats.tail_event_denominator}")
        print(f"    Tail events:         {stats.tail_event_count} ({_format_optional(stats.tail_event_fraction)})")


def _tail_threshold_warning(value: float) -> Optional[str]:
    if value > 1.0:
        return TAIL_THRESHOLD_WARNING
    return None


def _print_error(exc: Exception) -> None:
    print(f"ERROR: {exc}")
    if isinstance(exc, ScoreboardPoolingError):
        print(f"Pattern counts: {json.dumps(exc.pattern_counts, sort_keys=True)}")
        print(f"Pattern horizons: {json.dumps(exc.pattern_horizons, sort_keys=True)}")
        if exc.horizon_counts:
            print(f"Horizon counts: {json.dumps(exc.horizon_counts, sort_keys=True)}")
        if exc.pattern_horizon_counts:
            print(
                "Pattern horizon counts: "
                f"{json.dumps(exc.pattern_horizon_counts, sort_keys=True)}"
            )
    if isinstance(exc, ScoreboardSignalIntegrityError):
        print("Orphan observations:")
        for orphan in exc.orphans[:20]:
            print(f"  {json.dumps(orphan, sort_keys=True)}")
    if isinstance(exc, ScoreboardPatternIntegrityError):
        print("Pattern mismatches:")
        for mismatch in exc.mismatches[:20]:
            print(f"  {json.dumps(mismatch, sort_keys=True)}")
    if isinstance(exc, ScoreboardWindowIntegrityError):
        print("Window integrity errors:")
        for error in exc.window_errors[:20]:
            print(f"  {json.dumps(error, sort_keys=True)}")


def _error_payload(exc: Exception) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "status": "error",
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    if isinstance(exc, ScoreboardPartitionError):
        payload["unknown_status_counts"] = exc.unknown_status_counts
        payload["unknown_status_details"] = exc.unknown_status_details
    if isinstance(exc, ScoreboardDirectionError):
        payload["unsupported_direction_counts"] = exc.unsupported_direction_counts
        payload["unsupported_direction_details"] = exc.unsupported_direction_details
    if isinstance(exc, ScoreboardSignalIntegrityError):
        payload["orphans"] = exc.orphans
    if isinstance(exc, ScoreboardPatternIntegrityError):
        payload["mismatches"] = exc.mismatches
    if isinstance(exc, ScoreboardWindowIntegrityError):
        payload["window_errors"] = exc.window_errors
    if isinstance(exc, ScoreboardPoolingError):
        payload["pattern_counts"] = exc.pattern_counts
        payload["pattern_horizons"] = exc.pattern_horizons
        payload["horizon_counts"] = exc.horizon_counts
        payload["pattern_horizon_counts"] = exc.pattern_horizon_counts
    return payload


def _restore_env(previous_env: Dict[str, Optional[str]]) -> None:
    for key, value in previous_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


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
        "--signal-horizon",
        default=None,
        help="Optional signal horizon filter, such as 15d or 9d.",
    )
    parser.add_argument(
        "--group-by-horizon",
        action="store_true",
        help="Emit computed stats grouped by signal horizon instead of one headline block.",
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
