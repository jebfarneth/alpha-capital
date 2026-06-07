#!/usr/bin/env python3
"""Chunked market-path feature backfill entrypoint.

Runs the existing MarketPathFeatureJob in small, independently committed
pattern/date chunks. This keeps wide feature rows resumable without changing
feature formulas or detector behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from sqlalchemy.orm import Session

from alpha.data.contracts import AdapterResponse, LineageMeta, stable_hash
from alpha.data.config import ConfigError, FmpConfig
from alpha.data.fmp import FmpAdapter
from alpha.db.engine import (
    SchemaTargetError,
    create_all_tables,
    get_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.jobs.contracts import JobResult
from alpha.jobs.market_path_features import (
    DEFAULT_LOOKBACK_CALENDAR_DAYS,
    MarketPathFeatureJob,
)
from alpha.jobs.runner import run_job
from alpha.runtime_env import load_runtime_env


MARKET_PATH_REQUIRED_TABLES = (
    "evidence_jobs",
    "evidence_job_runs",
    "data_lineage",
    "feature_snapshots",
    "signal_registry",
    "market_path_features",
)


@dataclass(frozen=True)
class MarketPathBackfillChunk:
    pattern_id: str
    signal_start_date: date
    signal_end_date: date


@dataclass(frozen=True)
class BackfillRunConfig:
    through_date: date
    run_timestamp: datetime | None = None
    decision_date: date | None = None
    lookback_calendar_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS
    include_signal_session: bool = False
    liquidity_min_dollar_volume_20d: float = 100_000.0
    liquidity_min_price: float = 1.0
    schema: str | None = None


SessionFactory = Callable[[], Session]
JobFactory = Callable[..., MarketPathFeatureJob]
JobRunner = Callable[..., JobResult]
PrintFn = Callable[[str], None]


@dataclass
class _CachedHistoricalPrice:
    ticker: str
    from_date: date
    to_date: date
    asof_key: str | None
    kwargs_key: str
    response: AdapterResponse[Any]


class CachedHistoricalPriceFmpAdapter:
    """Run-scoped historical-price cache for adjacent backfill chunks."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self._historical_price_cache: list[_CachedHistoricalPrice] = []
        self.cache_hits = 0
        self.cache_misses = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def get_historical_price(
        self,
        ticker: str,
        from_date: date | None = None,
        to_date: date | None = None,
        asof: datetime | None = None,
        **kwargs: Any,
    ) -> AdapterResponse[Any]:
        requested_from = _coerce_date(from_date)
        requested_to = _coerce_date(to_date)
        if requested_from is None or requested_to is None:
            self.cache_misses += 1
            return self._wrapped.get_historical_price(
                ticker,
                from_date=from_date,
                to_date=to_date,
                asof=asof,
                **kwargs,
            )

        normalized_ticker = ticker.upper()
        asof_key = _historical_price_asof_key(asof)
        kwargs_key = _historical_price_kwargs_key(kwargs)
        for entry in self._historical_price_cache:
            if (
                entry.ticker == normalized_ticker
                and entry.asof_key == asof_key
                and entry.kwargs_key == kwargs_key
                and entry.from_date <= requested_from
                and entry.to_date >= requested_to
                and entry.response.ok
                and entry.response.data is not None
            ):
                self.cache_hits += 1
                data = _filter_cached_bars(
                    entry.response.data,
                    from_date=requested_from,
                    to_date=requested_to,
                )
                return AdapterResponse(
                    data=data,
                    lineage=LineageMeta(
                        provider=entry.response.lineage.provider,
                        endpoint=entry.response.lineage.endpoint,
                        request_timestamp=entry.response.lineage.request_timestamp,
                        asof_timestamp=asof or entry.response.lineage.asof_timestamp,
                        raw_payload_hash=stable_hash([
                            _cached_bar_payload(bar)
                            for bar in data
                        ]),
                        freshness_seconds=entry.response.lineage.freshness_seconds,
                        source_authority=entry.response.lineage.source_authority,
                        data_quality_flags={
                            **(entry.response.lineage.data_quality_flags or {}),
                            "market_path_backfill_cache_hit": True,
                        },
                    ),
                    rate_limit=entry.response.rate_limit,
                )

        self.cache_misses += 1
        response = self._wrapped.get_historical_price(
            ticker,
            from_date=from_date,
            to_date=to_date,
            asof=asof,
            **kwargs,
        )
        self._historical_price_cache.append(
            _CachedHistoricalPrice(
                ticker=normalized_ticker,
                from_date=requested_from,
                to_date=requested_to,
                asof_key=asof_key,
                kwargs_key=kwargs_key,
                response=response,
            )
        )
        return response


def plan_chunks(
    pattern_ids: Iterable[str],
    *,
    signal_start_date: date,
    signal_end_date: date,
    chunk_days: int = 1,
) -> list[MarketPathBackfillChunk]:
    """Split backfill work by pattern and inclusive signal-date chunks."""

    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")
    if signal_start_date > signal_end_date:
        raise ValueError("signal_start_date must be on or before signal_end_date")

    chunks: list[MarketPathBackfillChunk] = []
    for pattern_id in _unique_patterns(pattern_ids):
        cursor = signal_start_date
        while cursor <= signal_end_date:
            chunk_end = min(signal_end_date, cursor + timedelta(days=chunk_days - 1))
            chunks.append(
                MarketPathBackfillChunk(
                    pattern_id=pattern_id,
                    signal_start_date=cursor,
                    signal_end_date=chunk_end,
                )
            )
            cursor = chunk_end + timedelta(days=1)
    return chunks


def run_backfill_chunks(
    chunks: list[MarketPathBackfillChunk],
    *,
    session_factory: SessionFactory,
    fmp_adapter: Any,
    config: BackfillRunConfig,
    artifact_path: str | Path,
    job_factory: JobFactory = MarketPathFeatureJob,
    job_runner: JobRunner = run_job,
    print_fn: PrintFn = print,
) -> dict[str, Any]:
    """Run chunked backfill work, stopping on first failed chunk."""

    artifact = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": _config_payload(config),
        "chunks": [],
        "summary": {},
    }
    summary = {
        "chunks_total": len(chunks),
        "chunks_finished": 0,
        "chunks_failed": 0,
        "rows_inserted_total": 0,
        "rows_updated_total": 0,
        "fetch_error_count_total": 0,
        "artifact_path": str(artifact_path),
    }
    _write_artifact(artifact_path, artifact)

    for index, chunk in enumerate(chunks, start=1):
        print_fn(
            "START "
            f"chunk={index}/{len(chunks)} "
            f"pattern_id={chunk.pattern_id} "
            f"chunk_start={chunk.signal_start_date.isoformat()} "
            f"chunk_end={chunk.signal_end_date.isoformat()}"
        )
        chunk_record: dict[str, Any] = {
            "chunk_index": index,
            "pattern_id": chunk.pattern_id,
            "chunk_start": chunk.signal_start_date.isoformat(),
            "chunk_end": chunk.signal_end_date.isoformat(),
            "status": "running",
            "rows_inserted": 0,
            "rows_updated": 0,
            "fetch_error_count": 0,
        }
        artifact["chunks"].append(chunk_record)
        _write_artifact(artifact_path, artifact)

        started = time.monotonic()
        cache_before = _adapter_cache_stats(fmp_adapter)
        session = session_factory()
        try:
            job = job_factory(
                session=session,
                fmp_adapter=fmp_adapter,
                run_timestamp=config.run_timestamp,
                pattern_ids=(chunk.pattern_id,),
                decision_date=config.decision_date,
                signal_start_date=chunk.signal_start_date,
                signal_end_date=chunk.signal_end_date,
                through_date=config.through_date,
                lookback_calendar_days=config.lookback_calendar_days,
                include_signal_session=config.include_signal_session,
                liquidity_min_dollar_volume_20d=(
                    config.liquidity_min_dollar_volume_20d
                ),
                liquidity_min_price=config.liquidity_min_price,
            )
            result = job_runner(
                session,
                job,
                params={
                    "source": "market_path_backfill",
                    "pattern_id": [chunk.pattern_id],
                    "signal_start_date": chunk.signal_start_date.isoformat(),
                    "signal_end_date": chunk.signal_end_date.isoformat(),
                    "through_date": config.through_date.isoformat(),
                    "run_timestamp": (
                        config.run_timestamp.isoformat()
                        if config.run_timestamp is not None else None
                    ),
                    "lookback_calendar_days": config.lookback_calendar_days,
                    "include_signal_session": config.include_signal_session,
                    "schema": config.schema,
                },
            )
            metrics = result.metrics or {}
            status = result.status
            errors = result.errors or []
        except Exception as exc:  # pragma: no cover - defensive CLI boundary
            result = JobResult(
                status="failed",
                errors=[{"exception": str(exc)}],
            )
            metrics = {}
            status = "failed"
            errors = result.errors
        finally:
            session.close()

        elapsed = time.monotonic() - started
        metrics = dict(metrics or {})
        stage_timings = dict(metrics.get("stage_timing_seconds") or {})
        internal_elapsed = stage_timings.get("job_internal_total_seconds")
        if internal_elapsed is not None:
            stage_timings["runner_job_commit_overhead_seconds"] = round(
                max(0.0, elapsed - float(internal_elapsed)),
                6,
            )
            metrics["stage_timing_seconds"] = stage_timings
        cache_after = _adapter_cache_stats(fmp_adapter)
        if cache_after:
            metrics["fmp_historical_price_cache_hits"] = (
                cache_after["cache_hits"] - cache_before.get("cache_hits", 0)
            )
            metrics["fmp_historical_price_cache_misses"] = (
                cache_after["cache_misses"] - cache_before.get("cache_misses", 0)
            )
        dominant_stage, dominant_seconds = _dominant_stage(stage_timings)
        rows_inserted = int(metrics.get("rows_inserted") or 0)
        rows_updated = int(metrics.get("rows_updated") or 0)
        fetch_error_count = int(metrics.get("fetch_error_count") or 0)
        rc = 0 if result.ok else 1
        chunk_record.update(
            {
                "status": status,
                "rc": rc,
                "elapsed_seconds": round(elapsed, 3),
                "rows_inserted": rows_inserted,
                "rows_updated": rows_updated,
                "fetch_error_count": fetch_error_count,
                "metrics": metrics,
                "errors": errors,
                "dominant_stage": dominant_stage,
                "dominant_stage_seconds": dominant_seconds,
            }
        )
        print_fn(
            "FINISH "
            f"chunk={index}/{len(chunks)} "
            f"pattern_id={chunk.pattern_id} "
            f"chunk_start={chunk.signal_start_date.isoformat()} "
            f"chunk_end={chunk.signal_end_date.isoformat()} "
            f"elapsed_seconds={elapsed:.3f} "
            f"rc={rc} status={status} "
            f"rows_inserted={rows_inserted} "
            f"rows_updated={rows_updated} "
            f"fetch_error_count={fetch_error_count} "
            f"dominant_stage={dominant_stage or 'none'} "
            f"dominant_stage_seconds={dominant_seconds:.3f}"
        )

        if result.ok:
            summary["chunks_finished"] += 1
            summary["rows_inserted_total"] += rows_inserted
            summary["rows_updated_total"] += rows_updated
            summary["fetch_error_count_total"] += fetch_error_count
        else:
            summary["chunks_failed"] += 1
            summary["failed_pattern_id"] = chunk.pattern_id
            summary["failed_chunk_start"] = chunk.signal_start_date.isoformat()
            summary["failed_chunk_end"] = chunk.signal_end_date.isoformat()
            artifact["summary"] = summary
            artifact["ended_at"] = datetime.now(timezone.utc).isoformat()
            _write_artifact(artifact_path, artifact)
            break

        artifact["summary"] = summary
        _write_artifact(artifact_path, artifact)

    artifact["summary"] = summary
    artifact["ended_at"] = datetime.now(timezone.utc).isoformat()
    _write_artifact(artifact_path, artifact)
    print_fn(
        "SUMMARY "
        f"chunks_total={summary['chunks_total']} "
        f"chunks_finished={summary['chunks_finished']} "
        f"chunks_failed={summary['chunks_failed']} "
        f"rows_inserted_total={summary['rows_inserted_total']} "
        f"rows_updated_total={summary['rows_updated_total']} "
        f"fetch_error_count_total={summary['fetch_error_count_total']} "
        f"artifact_path={summary['artifact_path']}"
    )
    return summary


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
    if target_schema is not None:
        try:
            prepare_writable_schema_target(
                schema=target_schema,
                create_tables=args.create_tables,
                required_tables=MARKET_PATH_REQUIRED_TABLES,
            )
        except (SchemaTargetError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1

    if args.create_tables and target_schema is None:
        create_all_tables()

    try:
        fmp_adapter = CachedHistoricalPriceFmpAdapter(FmpAdapter(FmpConfig.from_env()))
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 1

    chunks = plan_chunks(
        args.pattern_id,
        signal_start_date=_parse_date(args.signal_start_date),
        signal_end_date=_parse_date(args.signal_end_date),
        chunk_days=args.chunk_days,
    )
    artifact_path = args.artifact_path or _default_artifact_path()
    config = BackfillRunConfig(
        through_date=_parse_date(args.through_date),
        run_timestamp=_parse_timestamp(args.run_timestamp),
        decision_date=_parse_date(args.decision_date) if args.decision_date else None,
        lookback_calendar_days=args.lookback_calendar_days,
        include_signal_session=args.include_signal_session,
        liquidity_min_dollar_volume_20d=args.liquidity_min_dollar_volume_20d,
        liquidity_min_price=args.liquidity_min_price,
        schema=target_schema,
    )
    summary = run_backfill_chunks(
        chunks,
        session_factory=get_session,
        fmp_adapter=fmp_adapter,
        config=config,
        artifact_path=artifact_path,
    )
    return 0 if summary["chunks_failed"] == 0 else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run market-path feature backfills in resumable chunks."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Run live backfill chunks")
    parser.add_argument("--database-url", help="Override DATABASE_URL for this run")
    parser.add_argument("--schema", help="Optional PostgreSQL schema/search_path target")
    parser.add_argument("--create-tables", action="store_true")
    parser.add_argument("--run-timestamp")
    parser.add_argument("--decision-date")
    parser.add_argument(
        "--pattern-id",
        action="append",
        default=[],
        help="Pattern id to backfill. Repeat for multiple patterns. Defaults to M4.",
    )
    parser.add_argument("--signal-start-date", required=True)
    parser.add_argument("--signal-end-date", required=True)
    parser.add_argument("--through-date", required=True)
    parser.add_argument("--chunk-days", type=int, default=1)
    parser.add_argument("--artifact-path")
    parser.add_argument(
        "--lookback-calendar-days",
        type=int,
        default=DEFAULT_LOOKBACK_CALENDAR_DAYS,
    )
    parser.add_argument("--include-signal-session", action="store_true")
    parser.add_argument("--liquidity-min-dollar-volume-20d", type=float, default=100_000.0)
    parser.add_argument("--liquidity-min-price", type=float, default=1.0)
    args = parser.parse_args(argv)
    if not args.pattern_id:
        args.pattern_id = ["M4"]
    return args


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


def _unique_patterns(pattern_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_pattern in pattern_ids:
        pattern = str(raw_pattern or "").strip().upper()
        if not pattern or pattern in seen:
            continue
        ordered.append(pattern)
        seen.add(pattern)
    if not ordered:
        raise ValueError("at least one pattern_id is required")
    return ordered


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _historical_price_kwargs_key(kwargs: dict[str, Any]) -> str:
    return json.dumps(kwargs, sort_keys=True, default=str)


def _historical_price_asof_key(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is not None and value.utcoffset() is not None:
        value = value.astimezone(timezone.utc)
    return value.isoformat()


def _filter_cached_bars(
    bars: Iterable[Any],
    *,
    from_date: date,
    to_date: date,
) -> list[Any]:
    filtered = []
    for bar in bars:
        bar_date = _coerce_date(getattr(bar, "date", None))
        if bar_date is None:
            continue
        if from_date <= bar_date <= to_date:
            filtered.append(bar)
    return filtered


def _cached_bar_payload(bar: Any) -> dict[str, Any]:
    return {
        "date": getattr(bar, "date", None),
        "open": getattr(bar, "open", None),
        "high": getattr(bar, "high", None),
        "low": getattr(bar, "low", None),
        "close": getattr(bar, "close", None),
        "volume": getattr(bar, "volume", None),
        "split_adjusted_close": getattr(bar, "split_adjusted_close", None),
        "adj_close": getattr(bar, "adj_close", None),
        "vwap": getattr(bar, "vwap", None),
    }


def _adapter_cache_stats(adapter: Any) -> dict[str, int]:
    if not hasattr(adapter, "cache_hits") or not hasattr(adapter, "cache_misses"):
        return {}
    return {
        "cache_hits": int(getattr(adapter, "cache_hits", 0)),
        "cache_misses": int(getattr(adapter, "cache_misses", 0)),
    }


def _config_payload(config: BackfillRunConfig) -> dict[str, Any]:
    return {
        "through_date": config.through_date.isoformat(),
        "run_timestamp": (
            config.run_timestamp.isoformat()
            if config.run_timestamp is not None else None
        ),
        "decision_date": (
            config.decision_date.isoformat()
            if config.decision_date is not None else None
        ),
        "lookback_calendar_days": config.lookback_calendar_days,
        "include_signal_session": config.include_signal_session,
        "liquidity_min_dollar_volume_20d": (
            config.liquidity_min_dollar_volume_20d
        ),
        "liquidity_min_price": config.liquidity_min_price,
        "schema": config.schema,
    }


def _write_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str))


def _dominant_stage(stage_timings: dict[str, Any]) -> tuple[str | None, float]:
    candidates: dict[str, float] = {}
    for key, value in stage_timings.items():
        if key == "job_internal_total_seconds":
            continue
        try:
            candidates[key] = float(value)
        except (TypeError, ValueError):
            continue
    if not candidates:
        return None, 0.0
    stage, seconds = max(candidates.items(), key=lambda item: item[1])
    return stage, seconds


def _default_artifact_path() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"/tmp/market_path_backfill_{stamp}.json"


if __name__ == "__main__":
    raise SystemExit(main())
