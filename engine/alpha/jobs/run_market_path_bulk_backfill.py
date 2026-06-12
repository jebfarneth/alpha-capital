#!/usr/bin/env python3
"""Bulk market-path feature backfill entrypoint.

This runner uses the existing MarketPathFeatureJob calculation path, then
persists computed rows through a stage/merge write path. It is intended for
larger historical ranges where many small ORM writes are too slow over remote
Postgres.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

import requests
from sqlalchemy import Column, MetaData, Table, bindparam, text
from sqlalchemy.orm import Session

from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError, stable_hash, utcnow
from alpha.data.config import ConfigError, FmpConfig
from alpha.data.fmp import FmpAdapter, HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.db.engine import (
    SchemaTargetError,
    create_all_tables,
    get_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.db.models import MarketPathFeature
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.market_path_features import (
    DEFAULT_LOOKBACK_CALENDAR_DAYS,
    FEATURE_VERSION,
    MarketPathFeatureJob,
    _bulk_upsert_market_path_features,
    sanitize_provider_error_message,
)
from alpha.jobs.historical_m4_signal_selector import (
    SIGNAL_SOURCE_CHOICES,
    SIGNAL_SOURCE_LIVE,
    normalize_signal_source,
)
from alpha.jobs.run_market_path_backfill import (
    CachedHistoricalPriceFmpAdapter,
    MARKET_PATH_REQUIRED_TABLES,
)
from alpha.jobs.runner import run_job
from alpha.runtime_env import load_runtime_env


JOB_NAME = "market_path_bulk_backfill"
DEFAULT_REQUEST_RETRIES = 2

PrintFn = Callable[[str], None]
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class BulkBackfillBatch:
    pattern_id: str
    signal_start_date: date
    signal_end_date: date


@dataclass
class BulkMergeResult:
    stage_table: str | None
    rows_staged: int
    rows_merged: int
    stage_load_seconds: float = 0.0
    merge_seconds: float = 0.0


class _TimeoutRequestsSession(requests.Session):
    """Requests session that lets the bulk runner own FMP request timeouts."""

    def __init__(self, timeout_seconds: float) -> None:
        super().__init__()
        self._timeout_seconds = timeout_seconds

    def get(self, url: str, **kwargs: Any) -> requests.Response:  # type: ignore[override]
        kwargs["timeout"] = self._timeout_seconds
        return super().get(url, **kwargs)


class RetryingHistoricalPriceFmpAdapter:
    """Bounded retry wrapper for FMP historical-price calls used by bulk backfill."""

    def __init__(
        self,
        wrapped: Any,
        *,
        max_retries: int = DEFAULT_REQUEST_RETRIES,
        retry_sleep_seconds: float = 1.0,
        request_timeout_seconds: float | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._wrapped = wrapped
        self._max_retries = max_retries
        self._retry_sleep_seconds = retry_sleep_seconds
        self._request_timeout_seconds = request_timeout_seconds

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    @property
    def cache_hits(self) -> int:
        return int(getattr(self._wrapped, "cache_hits", 0))

    @property
    def cache_misses(self) -> int:
        return int(getattr(self._wrapped, "cache_misses", 0))

    def get_historical_price(
        self,
        ticker: str,
        from_date: date | None = None,
        to_date: date | None = None,
        asof: datetime | None = None,
        **kwargs: Any,
    ) -> AdapterResponse[Any]:
        attempts: list[dict[str, Any]] = []
        last_response: AdapterResponse[Any] | None = None
        for attempt_index in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._wrapped.get_historical_price(
                    ticker,
                    from_date=from_date,
                    to_date=to_date,
                    asof=asof,
                    **kwargs,
                )
            except Exception as exc:
                response = _historical_price_exception_response(
                    ticker=ticker,
                    from_date=from_date,
                    to_date=to_date,
                    asof=asof,
                    adjusted=bool(kwargs.get("adjusted", False)),
                    exc=exc,
                )
            attempts.append(_attempt_payload(response, attempt_index + 1, started))
            last_response = response
            if response.ok:
                return _with_retry_lineage_flags(
                    response,
                    attempts=attempts,
                    retry_exhausted=False,
                    max_retries=self._max_retries,
                    request_timeout_seconds=self._request_timeout_seconds,
                )
            if response.error is None or not response.error.retryable:
                return _with_retry_lineage_flags(
                    response,
                    attempts=attempts,
                    retry_exhausted=False,
                    max_retries=self._max_retries,
                    request_timeout_seconds=self._request_timeout_seconds,
                )
            if attempt_index < self._max_retries and self._retry_sleep_seconds > 0:
                time.sleep(self._retry_sleep_seconds)

        assert last_response is not None
        return _with_retry_lineage_flags(
            last_response,
            attempts=attempts,
            retry_exhausted=True,
            max_retries=self._max_retries,
            request_timeout_seconds=self._request_timeout_seconds,
        )


class MarketPathBulkBackfillJob(BaseJob):
    """Compute market-path features in batches and bulk merge them."""

    def __init__(
        self,
        *,
        session: Session,
        fmp_adapter: Any,
        pattern_ids: Sequence[str],
        signal_start_date: date,
        signal_end_date: date,
        through_date: date,
        run_timestamp: datetime | None = None,
        batch_days: int = 20,
        include_signal_session: bool = False,
        lookback_calendar_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS,
        feature_version: str = FEATURE_VERSION,
        progress_artifact: str | Path | None = None,
        schema: str | None = None,
        progress_every: int = 10,
        request_timeout_seconds: float = 30.0,
        max_fetch_concurrency: int = 1,
        signal_source: str = SIGNAL_SOURCE_LIVE,
        print_fn: PrintFn = print,
    ) -> None:
        if batch_days < 1:
            raise ValueError("batch_days must be >= 1")
        if progress_every < 1:
            raise ValueError("progress_every must be >= 1")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be > 0")
        if max_fetch_concurrency < 1:
            raise ValueError("max_fetch_concurrency must be >= 1")
        self._session = session
        self._fmp = fmp_adapter
        self._pattern_ids = tuple(_unique_patterns(pattern_ids))
        self._signal_start_date = signal_start_date
        self._signal_end_date = signal_end_date
        self._through_date = through_date
        self._run_timestamp = run_timestamp
        self._batch_days = batch_days
        self._include_signal_session = include_signal_session
        self._lookback_calendar_days = lookback_calendar_days
        self._feature_version = feature_version
        self._progress_artifact = Path(progress_artifact) if progress_artifact else None
        self._schema = schema
        self._progress_every = progress_every
        self._request_timeout_seconds = request_timeout_seconds
        self._max_fetch_concurrency = max_fetch_concurrency
        self._signal_source = normalize_signal_source(signal_source)
        self._print_fn = print_fn

    @property
    def job_name(self) -> str:
        return JOB_NAME

    @property
    def job_type(self) -> str:
        return "feature_enrichment"

    def run(self, ctx: JobContext) -> JobResult:
        started_total = time.perf_counter()
        if self._signal_start_date > self._signal_end_date:
            return JobResult(
                status="failed",
                errors=[{"message": "signal_start_date must be on or before signal_end_date"}],
            )

        batches = plan_bulk_batches(
            self._pattern_ids,
            signal_start_date=self._signal_start_date,
            signal_end_date=self._signal_end_date,
            batch_days=self._batch_days,
        )
        artifact: dict[str, Any] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "schema": self._schema,
            "feature_version": self._feature_version,
            "request_timeout_seconds": self._request_timeout_seconds,
            "signal_source": self._signal_source,
            "max_fetch_concurrency_requested": self._max_fetch_concurrency,
            "max_fetch_concurrency_effective": 1,
            "batches": [],
            "summary": {},
        }
        _write_artifact(self._progress_artifact, artifact)

        signal_count = 0
        feature_row_count = 0
        rows_inserted = 0
        rows_updated = 0
        rows_unchanged = 0
        rows_skipped = 0
        rows_merged = 0
        lineages_recorded = 0
        non_session_bars_skipped = 0
        non_session_bar_skip_sample: list[dict[str, str]] = []
        fetch_errors: list[dict[str, Any]] = []
        stage_seconds = 0.0
        merge_seconds = 0.0
        compute_seconds = 0.0
        ticker_fetch_started_total = 0
        ticker_fetch_finished_total = 0
        ticker_fetch_error_total = 0
        max_stage_batch_size = 0
        stage_tables: list[str] = []

        for index, batch in enumerate(batches, start=1):
            batch_started = time.perf_counter()
            batch_record: dict[str, Any] = {
                "batch_index": index,
                "pattern_id": batch.pattern_id,
                "signal_start_date": batch.signal_start_date.isoformat(),
                "signal_end_date": batch.signal_end_date.isoformat(),
                "status": "running",
                "progress_events": [],
            }
            artifact["batches"].append(batch_record)
            self._append_batch_event(
                artifact,
                batch_record,
                "batch_start",
                {
                    "batch_index": index,
                    "batch_count": len(batches),
                    "pattern_id": batch.pattern_id,
                    "signal_start_date": batch.signal_start_date.isoformat(),
                    "signal_end_date": batch.signal_end_date.isoformat(),
                },
            )

            def batch_progress(event: str, payload: dict[str, Any]) -> None:
                self._append_batch_event(artifact, batch_record, event, payload)

            feature_job = MarketPathFeatureJob(
                session=self._session,
                fmp_adapter=self._fmp,
                run_timestamp=self._run_timestamp,
                pattern_ids=(batch.pattern_id,),
                signal_start_date=batch.signal_start_date,
                signal_end_date=batch.signal_end_date,
                through_date=self._through_date,
                lookback_calendar_days=self._lookback_calendar_days,
                feature_version=self._feature_version,
                include_signal_session=self._include_signal_session,
                progress_callback=batch_progress,
                progress_every=self._progress_every,
                max_fetch_concurrency=self._max_fetch_concurrency,
                signal_source=self._signal_source,
            )
            collect_started = time.perf_counter()
            collection = feature_job.collect_feature_rows(ctx)
            compute_seconds += time.perf_counter() - collect_started
            batch_progress(
                "collect_finish",
                {
                    "signals_loaded": collection.signals_scanned,
                    "tickers_planned": collection.ticker_planned_count,
                    "ticker_fetch_started_count": collection.ticker_fetch_started_count,
                    "ticker_fetch_finished_count": collection.ticker_fetch_finished_count,
                    "ticker_fetch_error_count": collection.ticker_fetch_error_count,
                    "feature_rows_generated": len(collection.pending_feature_rows or []),
                    "fetch_error_count": len(collection.fetch_errors or []),
                    "non_session_bars_skipped": collection.non_session_bars_skipped,
                    "non_session_bar_skip_sample": collection.non_session_bar_skip_sample or [],
                    "elapsed_seconds": round(time.perf_counter() - collect_started, 6),
                },
            )
            if collection.errors:
                fetch_errors.extend(collection.errors)
                batch_record["status"] = "failed"
                batch_record["errors"] = collection.errors
                _write_artifact(self._progress_artifact, artifact)
                break
            fetch_errors.extend(collection.fetch_errors or [])

            signal_count += collection.signals_scanned
            feature_row_count += collection.rows_inserted + collection.rows_updated
            ticker_fetch_started_total += collection.ticker_fetch_started_count
            ticker_fetch_finished_total += collection.ticker_fetch_finished_count
            ticker_fetch_error_total += collection.ticker_fetch_error_count
            rows_inserted += collection.rows_inserted
            rows_updated += max(0, collection.rows_updated - collection.rows_unchanged)
            rows_unchanged += collection.rows_unchanged
            rows_skipped += collection.rows_skipped
            lineages_recorded += collection.lineages_recorded
            non_session_bars_skipped += collection.non_session_bars_skipped
            for sample in collection.non_session_bar_skip_sample or []:
                if len(non_session_bar_skip_sample) >= 10:
                    break
                non_session_bar_skip_sample.append(sample)

            rows = collection.pending_feature_rows or []
            lineages = collection.pending_lineages or []
            batch_progress(
                "staging_write_start",
                {
                    "pending_lineage_count": len(lineages),
                    "pending_feature_row_count": len(rows),
                },
            )
            if rows and lineages:
                self._session.add_all(lineages)
                self._session.flush()

            merge_result = bulk_stage_merge_market_path_features(
                self._session,
                rows,
                progress_callback=batch_progress,
            )
            if rows:
                self._session.flush()
            self._session.commit()
            batch_progress(
                "staging_write_finish",
                {
                    "rows_staged": merge_result.rows_staged,
                    "rows_merged": merge_result.rows_merged,
                    "stage_load_seconds": merge_result.stage_load_seconds,
                    "merge_seconds": merge_result.merge_seconds,
                },
            )
            stage_seconds += merge_result.stage_load_seconds
            merge_seconds += merge_result.merge_seconds
            rows_merged += merge_result.rows_merged
            max_stage_batch_size = max(max_stage_batch_size, merge_result.rows_staged)
            if merge_result.stage_table:
                stage_tables.append(merge_result.stage_table)

            batch_record.update({
                "status": "partial_failed" if collection.fetch_errors else "finished",
                "signals_scanned": collection.signals_scanned,
                "tickers_planned": collection.ticker_planned_count,
                "ticker_fetch_started_count": collection.ticker_fetch_started_count,
                "ticker_fetch_finished_count": collection.ticker_fetch_finished_count,
                "ticker_fetch_error_count": collection.ticker_fetch_error_count,
                "feature_rows_scanned": collection.rows_inserted + collection.rows_updated,
                "rows_staged": merge_result.rows_staged,
                "rows_merged": merge_result.rows_merged,
                "lineages_recorded": collection.lineages_recorded,
                "non_session_bars_skipped": collection.non_session_bars_skipped,
                "non_session_bar_skip_sample": collection.non_session_bar_skip_sample or [],
                "fetch_error_count": len(collection.fetch_errors or []),
                "fetch_errors": collection.fetch_errors or [],
                "elapsed_seconds": round(time.perf_counter() - batch_started, 6),
                "stage_timing_seconds": collection.stage_timings or {},
            })
            batch_progress(
                "batch_finish",
                {
                    "status": batch_record["status"],
                    "elapsed_seconds": batch_record["elapsed_seconds"],
                    "fetch_error_count": batch_record["fetch_error_count"],
                    "rows_merged": merge_result.rows_merged,
                },
            )
            _write_artifact(self._progress_artifact, artifact)
            batch_progress(
                "batch_artifact_written",
                {"artifact_path": str(self._progress_artifact) if self._progress_artifact else None},
            )
            if collection.fetch_errors:
                break

        rank_rows_updated = 0
        rank_started = time.perf_counter()
        if not fetch_errors:
            self._append_run_event(
                artifact,
                "rank_pass_start",
                {
                    "signal_start_date": self._signal_start_date.isoformat(),
                    "through_date": self._through_date.isoformat(),
                    "pattern_ids": list(self._pattern_ids),
                },
            )
            rank_job = MarketPathFeatureJob(
                session=self._session,
                fmp_adapter=self._fmp,
                run_timestamp=self._run_timestamp,
                pattern_ids=self._pattern_ids,
                signal_start_date=self._signal_start_date,
                signal_end_date=self._signal_end_date,
                through_date=self._through_date,
                lookback_calendar_days=self._lookback_calendar_days,
                feature_version=self._feature_version,
                include_signal_session=self._include_signal_session,
                signal_source=self._signal_source,
            )
            rank_rows_updated = rank_job._populate_cross_sectional_ranks(
                start_date=self._signal_start_date,
                through_date=self._through_date,
                progress_callback=lambda event, payload: self._append_run_event(
                    artifact,
                    event,
                    payload,
                ),
                progress_every=self._progress_every,
            )
            if rank_rows_updated:
                self._session.flush()
            self._session.commit()
            self._append_run_event(
                artifact,
                "rank_pass_finish",
                {
                    "rank_rows_updated": rank_rows_updated,
                    "elapsed_seconds": round(time.perf_counter() - rank_started, 6),
                },
            )
        rank_seconds = time.perf_counter() - rank_started

        validation_started = time.perf_counter()
        self._append_run_event(artifact, "validation_start", {})
        validation = validate_market_path_bulk_backfill(
            self._session,
            pattern_ids=self._pattern_ids,
            signal_start_date=self._signal_start_date,
            signal_end_date=self._signal_end_date,
            feature_version=self._feature_version,
        )
        validation_seconds = time.perf_counter() - validation_started
        self._append_run_event(
            artifact,
            "validation_finish",
            {
                **validation,
                "elapsed_seconds": round(validation_seconds, 6),
            },
        )

        cache_stats = _adapter_cache_stats(self._fmp)
        stage_timings = {
            "compute_seconds": round(compute_seconds, 6),
            "stage_load_seconds": round(stage_seconds, 6),
            "merge_seconds": round(merge_seconds, 6),
            "rank_seconds": round(rank_seconds, 6),
            "validation_seconds": round(validation_seconds, 6),
            "total_seconds": round(time.perf_counter() - started_total, 6),
        }
        dominant_stage = max(stage_timings.items(), key=lambda item: item[1])[0]
        metrics = {
            "batch_count": len(batches),
            "signal_count": signal_count,
            "feature_row_count": feature_row_count,
            "fmp_fetch_count": int(cache_stats.get("cache_misses", 0)),
            "fmp_cache_hit_count": int(cache_stats.get("cache_hits", 0)),
            "fmp_cache_miss_count": int(cache_stats.get("cache_misses", 0)),
            "ticker_fetch_started_count": ticker_fetch_started_total,
            "ticker_fetch_finished_count": ticker_fetch_finished_total,
            "ticker_fetch_error_count": ticker_fetch_error_total,
            "request_timeout_seconds": self._request_timeout_seconds,
            "signal_source": self._signal_source,
            "max_fetch_concurrency_requested": self._max_fetch_concurrency,
            "max_fetch_concurrency_effective": 1,
            "rows_inserted": rows_inserted,
            "rows_updated": rows_updated,
            "rows_unchanged": rows_unchanged,
            "rows_skipped": rows_skipped,
            "rows_merged": rows_merged,
            "non_session_bars_skipped": non_session_bars_skipped,
            "non_session_bar_skip_sample": non_session_bar_skip_sample,
            "rank_rows_updated": rank_rows_updated,
            "rank_pass_count": 1 if not fetch_errors else 0,
            "lineages_recorded": lineages_recorded,
            "fetch_error_count": len(fetch_errors),
            "fetch_error_sample": _error_sample(fetch_errors),
            "max_stage_batch_size": max_stage_batch_size,
            "stage_tables": stage_tables,
            "stage_timing_seconds": stage_timings,
            "dominant_stage": dominant_stage,
            **validation,
        }
        artifact["summary"] = metrics
        artifact["ended_at"] = datetime.now(timezone.utc).isoformat()
        _write_artifact(self._progress_artifact, artifact)
        return JobResult(
            status="partial_failed" if fetch_errors else "finished",
            metrics=metrics,
            errors=fetch_errors,
        )

    def _append_batch_event(
        self,
        artifact: dict[str, Any],
        batch_record: dict[str, Any],
        event: str,
        payload: dict[str, Any],
    ) -> None:
        event_record = {
            "event": event,
            "at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        batch_record.setdefault("progress_events", []).append(event_record)
        batch_record["last_progress_event"] = event
        _write_artifact(self._progress_artifact, artifact)
        self._print_progress(event_record)

    def _append_run_event(
        self,
        artifact: dict[str, Any],
        event: str,
        payload: dict[str, Any],
    ) -> None:
        event_record = {
            "event": event,
            "at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        artifact.setdefault("progress_events", []).append(event_record)
        artifact["last_progress_event"] = event
        _write_artifact(self._progress_artifact, artifact)
        self._print_progress(event_record)

    def _print_progress(self, event_record: dict[str, Any]) -> None:
        event = event_record.get("event")
        pieces = [f"PROGRESS event={event}"]
        for key in (
            "batch_index",
            "batch_count",
            "pattern_id",
            "signal_start_date",
            "signal_end_date",
            "signals_loaded",
            "ticker_count",
            "ticker_fetch_started_count",
            "ticker_fetch_finished_count",
            "ticker_fetch_error_count",
            "feature_rows_generated",
            "rows_staged",
            "rows_merged",
            "rank_group_processed",
            "rank_group_total",
            "feature_session_date",
            "feature_version",
            "rank_rows_updated",
            "status",
            "elapsed_seconds",
        ):
            if key in event_record:
                pieces.append(f"{key}={event_record[key]}")
        self._print_fn(" ".join(pieces))


def plan_bulk_batches(
    pattern_ids: Iterable[str],
    *,
    signal_start_date: date,
    signal_end_date: date,
    batch_days: int,
) -> list[BulkBackfillBatch]:
    if batch_days < 1:
        raise ValueError("batch_days must be >= 1")
    batches: list[BulkBackfillBatch] = []
    for pattern_id in _unique_patterns(pattern_ids):
        cursor = signal_start_date
        while cursor <= signal_end_date:
            batch_end = min(signal_end_date, cursor + timedelta(days=batch_days - 1))
            batches.append(BulkBackfillBatch(pattern_id, cursor, batch_end))
            cursor = batch_end + timedelta(days=1)
    return batches


def bulk_stage_merge_market_path_features(
    session: Session,
    rows: Sequence[dict[str, Any]],
    *,
    progress_callback: ProgressCallback | None = None,
) -> BulkMergeResult:
    if not rows:
        if progress_callback is not None:
            progress_callback("stage_load_start", {"rows_to_stage": 0})
            progress_callback("stage_load_finish", {"rows_staged": 0, "elapsed_seconds": 0.0})
            progress_callback("merge_upsert_start", {"rows_to_merge": 0})
            progress_callback("merge_upsert_finish", {"rows_merged": 0, "elapsed_seconds": 0.0})
        return BulkMergeResult(stage_table=None, rows_staged=0, rows_merged=0)
    dialect_name = session.get_bind().dialect.name
    if dialect_name == "sqlite":
        if progress_callback is not None:
            progress_callback("stage_load_start", {"rows_to_stage": len(rows), "dialect": "sqlite"})
        merge_started = time.perf_counter()
        if progress_callback is not None:
            progress_callback("merge_upsert_start", {"rows_to_merge": len(rows), "dialect": "sqlite"})
        _bulk_upsert_market_path_features(session, rows)
        elapsed = round(time.perf_counter() - merge_started, 6)
        if progress_callback is not None:
            progress_callback("stage_load_finish", {"rows_staged": len(rows), "elapsed_seconds": 0.0})
            progress_callback("merge_upsert_finish", {"rows_merged": len(rows), "elapsed_seconds": elapsed})
        return BulkMergeResult(
            stage_table=None,
            rows_staged=len(rows),
            rows_merged=len(rows),
            stage_load_seconds=0.0,
            merge_seconds=elapsed,
        )

    stage_name = f"_tmp_market_path_features_{uuid4().hex}"
    quoted_stage = _quote_ident(stage_name)
    session.execute(text(
        f"CREATE TEMP TABLE {quoted_stage} "
        "(LIKE market_path_features INCLUDING DEFAULTS) ON COMMIT DROP"
    ))
    stage_table = _stage_table(stage_name)
    if progress_callback is not None:
        progress_callback(
            "stage_load_start",
            {"rows_to_stage": len(rows), "dialect": dialect_name, "stage_table": stage_name},
        )
    stage_started = time.perf_counter()
    session.execute(stage_table.insert(), list(rows))
    stage_load_seconds = time.perf_counter() - stage_started
    if progress_callback is not None:
        progress_callback(
            "stage_load_finish",
            {
                "rows_staged": len(rows),
                "elapsed_seconds": round(stage_load_seconds, 6),
                "stage_table": stage_name,
            },
        )
    target = MarketPathFeature.__table__
    columns = [column.name for column in target.columns]
    column_sql = ", ".join(_quote_ident(column) for column in columns)
    update_sql = ", ".join(
        f"{_quote_ident(column)} = EXCLUDED.{_quote_ident(column)}"
        for column in columns
        if column not in {"market_path_feature_id", "created_at"}
    )
    if progress_callback is not None:
        progress_callback("merge_upsert_start", {"rows_to_merge": len(rows), "stage_table": stage_name})
    merge_started = time.perf_counter()
    session.execute(text(
        "INSERT INTO market_path_features "
        f"({column_sql}) SELECT {column_sql} FROM {quoted_stage} "
        "ON CONFLICT (signal_id, feature_session_date, feature_version) "
        f"DO UPDATE SET {update_sql}"
    ))
    merge_seconds = time.perf_counter() - merge_started
    if progress_callback is not None:
        progress_callback(
            "merge_upsert_finish",
            {
                "rows_merged": len(rows),
                "elapsed_seconds": round(merge_seconds, 6),
                "stage_table": stage_name,
            },
        )
    return BulkMergeResult(
        stage_table=stage_name,
        rows_staged=len(rows),
        rows_merged=len(rows),
        stage_load_seconds=round(stage_load_seconds, 6),
        merge_seconds=round(merge_seconds, 6),
    )


def validate_market_path_bulk_backfill(
    session: Session,
    *,
    pattern_ids: Sequence[str],
    signal_start_date: date,
    signal_end_date: date,
    feature_version: str,
) -> dict[str, int]:
    params = {
        "patterns": tuple(pattern_ids),
        "signal_start": signal_start_date.isoformat(),
        "signal_end": signal_end_date.isoformat(),
        "feature_version": feature_version,
    }
    scoped_filter = (
        "pattern_id IN :patterns "
        "AND signal_date >= :signal_start "
        "AND signal_date <= :signal_end "
        "AND feature_version = :feature_version"
    )
    duplicate_groups = session.execute(
        text(
            "SELECT COUNT(*) FROM ("
            "SELECT signal_id, feature_session_date, feature_version, COUNT(*) "
            "FROM market_path_features "
            "GROUP BY signal_id, feature_session_date, feature_version "
            "HAVING COUNT(*) > 1"
            ") d"
        )
    ).scalar()
    row = session.execute(
        text(
            "SELECT "
            "COUNT(*) AS scoped_feature_rows, "
            "COUNT(*) FILTER (WHERE input_hash IS NULL OR output_hash IS NULL "
            "OR data_lineage_id IS NULL OR feature_json IS NULL) AS missing_lineage_hash_rows, "
            "COUNT(*) FILTER (WHERE prior_52w_high IS NOT NULL) AS prior_52w_high_rows, "
            "COUNT(*) FILTER (WHERE dollar_volume_rank IS NOT NULL) AS rank_populated_rows, "
            "COUNT(*) FILTER (WHERE feature_session_date < entry_session_date "
            "AND (return_from_entry_open IS NOT NULL "
            "OR return_from_entry_high IS NOT NULL "
            "OR return_from_entry_low IS NOT NULL "
            "OR return_from_entry_close IS NOT NULL)) AS pre_entry_leakage_rows "
            "FROM market_path_features WHERE "
            f"{scoped_filter}"
        ).bindparams(bindparam("patterns", expanding=True)),
        params,
    ).mappings().one()
    v1_rows = session.execute(
        text(
            "SELECT COUNT(*) FROM market_path_features "
            "WHERE pattern_id IN :patterns "
            "AND signal_date >= :signal_start "
            "AND signal_date <= :signal_end "
            "AND feature_version <> :feature_version"
        ).bindparams(bindparam("patterns", expanding=True)),
        params,
    ).scalar()
    return {
        "duplicate_groups": int(duplicate_groups or 0),
        "pre_entry_leakage_count": int(row["pre_entry_leakage_rows"] or 0),
        "missing_lineage_hash_count": int(row["missing_lineage_hash_rows"] or 0),
        "scoped_feature_row_count": int(row["scoped_feature_rows"] or 0),
        "rich_prior_52w_high_populated_count": int(row["prior_52w_high_rows"] or 0),
        "rank_populated_count": int(row["rank_populated_rows"] or 0),
        "coexisting_other_feature_version_count": int(v1_rows or 0),
    }


def _stage_table(stage_name: str) -> Table:
    metadata = MetaData()
    return Table(
        stage_name,
        metadata,
        *[
            Column(column.name, column.type)
            for column in MarketPathFeature.__table__.columns
        ],
    )


def _validate_write_target(*, schema: str | None, confirm_live_write: bool) -> None:
    if schema is None and not confirm_live_write:
        raise ValueError(
            "Refusing public/default market_path bulk backfill without "
            "--confirm-live-write"
        )


def _run_live(args: argparse.Namespace) -> int:
    load_runtime_env()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if args.schema:
        os.environ["ALPHA_DB_SCHEMA"] = args.schema
        reset_globals()
    target_schema = args.schema or os.environ.get("ALPHA_DB_SCHEMA")
    try:
        _validate_write_target(
            schema=target_schema,
            confirm_live_write=args.confirm_live_write,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

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
    elif args.create_tables:
        create_all_tables()

    try:
        fmp_adapter = RetryingHistoricalPriceFmpAdapter(
            CachedHistoricalPriceFmpAdapter(
                FmpAdapter(
                    FmpConfig.from_env(),
                    session=_TimeoutRequestsSession(args.request_timeout_seconds),
                )
            ),
            max_retries=DEFAULT_REQUEST_RETRIES,
            request_timeout_seconds=args.request_timeout_seconds,
        )
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 1

    artifact_path = args.progress_artifact or _default_artifact_path()
    job = MarketPathBulkBackfillJob(
        session=get_session(),
        fmp_adapter=fmp_adapter,
        pattern_ids=args.pattern_id or ["M4"],
        signal_start_date=_parse_date(args.signal_start_date),
        signal_end_date=_parse_date(args.signal_end_date),
        through_date=_parse_date(args.through_date),
        run_timestamp=_parse_timestamp(args.run_timestamp),
        batch_days=args.batch_days,
        include_signal_session=args.include_signal_session,
        lookback_calendar_days=args.lookback_calendar_days,
        progress_artifact=artifact_path,
        schema=target_schema,
        progress_every=args.progress_every,
        request_timeout_seconds=args.request_timeout_seconds,
        max_fetch_concurrency=args.max_fetch_concurrency,
        signal_source=args.signal_source,
    )
    try:
        result = run_job(
            job._session,
            job,
            params={
                "source": "market_path_bulk_backfill",
                "pattern_id": args.pattern_id or ["M4"],
                "signal_start_date": args.signal_start_date,
                "signal_end_date": args.signal_end_date,
                "through_date": args.through_date,
                "include_signal_session": args.include_signal_session,
                "batch_days": args.batch_days,
                "schema": target_schema,
                "confirm_live_write": args.confirm_live_write,
                "progress_artifact": str(artifact_path),
                "progress_every": args.progress_every,
                "request_timeout_seconds": args.request_timeout_seconds,
                "max_fetch_concurrency": args.max_fetch_concurrency,
                "signal_source": args.signal_source,
            },
        )
    finally:
        job._session.close()
    print(json.dumps(result.metrics, sort_keys=True, indent=2, default=str))
    return 0 if result.ok else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run market-path feature backfills through a bulk stage/merge path."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Run live bulk backfill")
    parser.add_argument("--database-url", help="Override DATABASE_URL")
    parser.add_argument("--schema", help="Optional PostgreSQL schema/search_path target")
    parser.add_argument("--create-tables", action="store_true")
    parser.add_argument("--confirm-live-write", action="store_true")
    parser.add_argument("--run-timestamp")
    parser.add_argument(
        "--pattern-id",
        action="append",
        default=[],
        help="Pattern id to backfill. Repeat for multiple patterns.",
    )
    parser.add_argument("--signal-start-date", required=True)
    parser.add_argument("--signal-end-date", required=True)
    parser.add_argument("--through-date", required=True)
    parser.add_argument("--include-signal-session", action="store_true")
    parser.add_argument("--batch-days", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-fetch-concurrency", type=int, default=1)
    parser.add_argument(
        "--signal-source",
        choices=SIGNAL_SOURCE_CHOICES,
        default=SIGNAL_SOURCE_LIVE,
        help=(
            "Signal corpus to consume. Use historical-m4-replay for ML/backfills "
            "that must exclude stale live M4 rows outside replay membership."
        ),
    )
    parser.add_argument("--progress-artifact")
    parser.add_argument(
        "--lookback-calendar-days",
        type=int,
        default=DEFAULT_LOOKBACK_CALENDAR_DAYS,
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


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


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


def _adapter_cache_stats(adapter: Any) -> dict[str, int]:
    if not hasattr(adapter, "cache_hits") or not hasattr(adapter, "cache_misses"):
        return {"cache_hits": 0, "cache_misses": 0}
    return {
        "cache_hits": int(getattr(adapter, "cache_hits", 0)),
        "cache_misses": int(getattr(adapter, "cache_misses", 0)),
    }


def _error_sample(errors: Sequence[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    return [_sanitize_error_payload(error) for error in errors[:limit]]


def _attempt_payload(
    response: AdapterResponse[Any],
    attempt_number: int,
    started: float,
) -> dict[str, Any]:
    error = response.error
    payload: dict[str, Any] = {
        "attempt": attempt_number,
        "ok": response.ok,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    if error is not None:
        payload.update({
            "provider": error.provider,
            "endpoint": error.endpoint,
            "status_code": error.status_code,
            "error_type": error.error_type,
            "message": sanitize_provider_error_message(error.message),
            "retryable": error.retryable,
        })
    return payload


def _historical_price_exception_response(
    *,
    ticker: str,
    from_date: date | None,
    to_date: date | None,
    asof: datetime | None,
    adjusted: bool,
    exc: Exception,
) -> AdapterResponse[Any]:
    request_ts = utcnow()
    endpoint = HISTORICAL_PRICE_FULL_ENDPOINT
    payload = {
        "ticker": ticker,
        "from_date": from_date.isoformat() if from_date else None,
        "to_date": to_date.isoformat() if to_date else None,
        "adjusted": adjusted,
        "exception_type": type(exc).__name__,
        "message": sanitize_provider_error_message(str(exc)),
    }
    sanitized_message = sanitize_provider_error_message(str(exc))
    return AdapterResponse(
        data=None,
        lineage=LineageMeta(
            provider="FMP",
            endpoint=endpoint,
            request_timestamp=request_ts,
            asof_timestamp=asof or request_ts,
            raw_payload_hash=stable_hash(payload),
            source_authority="FMP_Ultimate",
            data_quality_flags={
                "market_path_bulk_adapter_exception": True,
                "exception_type": type(exc).__name__,
            },
        ),
        error=ProviderError(
            provider="FMP",
            endpoint=endpoint,
            status_code=None,
            error_type="adapter_exception",
            message=sanitized_message,
            retryable=True,
        ),
    )


def _with_retry_lineage_flags(
    response: AdapterResponse[Any],
    *,
    attempts: list[dict[str, Any]],
    retry_exhausted: bool,
    max_retries: int,
    request_timeout_seconds: float | None,
) -> AdapterResponse[Any]:
    if len(attempts) <= 1 and not retry_exhausted and request_timeout_seconds is None:
        return response
    flags = dict(response.lineage.data_quality_flags or {})
    flags.update({
        "market_path_bulk_retry_attempt_count": len(attempts),
        "market_path_bulk_retry_max_retries": max_retries,
        "market_path_bulk_retry_exhausted": retry_exhausted,
        "market_path_bulk_retry_attempts": attempts,
    })
    if request_timeout_seconds is not None:
        flags["market_path_bulk_request_timeout_seconds"] = request_timeout_seconds
    flags = _sanitize_error_payload(flags)
    error = response.error
    if error is not None:
        sanitized_message = sanitize_provider_error_message(error.message)
        if sanitized_message != error.message:
            error = ProviderError(
                provider=error.provider,
                endpoint=error.endpoint,
                status_code=error.status_code,
                error_type=error.error_type,
                message=sanitized_message,
                retryable=error.retryable,
            )
    lineage = LineageMeta(
        provider=response.lineage.provider,
        endpoint=response.lineage.endpoint,
        request_timestamp=response.lineage.request_timestamp,
        asof_timestamp=response.lineage.asof_timestamp,
        raw_payload_hash=response.lineage.raw_payload_hash,
        freshness_seconds=response.lineage.freshness_seconds,
        source_authority=response.lineage.source_authority,
        data_quality_flags=flags,
    )
    return AdapterResponse(
        data=response.data,
        lineage=lineage,
        rate_limit=response.rate_limit,
        error=error,
    )


def _sanitize_error_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _sanitize_error_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_sanitize_error_payload(value) for value in payload]
    if isinstance(payload, str):
        return sanitize_provider_error_message(payload)
    return payload


def _write_artifact(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str))


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _default_artifact_path() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"/tmp/market_path_bulk_backfill_{stamp}.json"


if __name__ == "__main__":
    raise SystemExit(main())
