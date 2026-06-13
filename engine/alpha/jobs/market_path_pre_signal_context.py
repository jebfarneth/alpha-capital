"""M4 pre-signal setup context backfill.

This job builds ticker-date keyed predictor rows for the sessions before a
signal fires. Rows are conditional on a later signal by construction; the
future no-fire negative cohort is expected to reuse the same table without a
signal-link row.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import Column, MetaData, Table, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from alpha.data.contracts import stable_hash
from alpha.data.fmp import HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.db.models import (
    DataLineage,
    HistoricalUniverseReconstruction,
    MarketPathPreSignalContext,
    MarketPathPreSignalLink,
    SignalRegistry,
)
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.historical_m4_signal_selector import (
    SIGNAL_SOURCE_HISTORICAL_M4_REPLAY,
    apply_signal_source_filter,
    normalize_signal_source,
)
from alpha.jobs.market_path_features import (
    _bar_lineage_payload,
    _bar_payload,
    _build_data_lineage,
    _clean_bars,
    _lineage_quality_flags,
    _median,
    _price_basis,
    _safe_return,
    _sigma_close_to_close,
    sanitize_provider_error_message,
)
from alpha.market_calendar import (
    previous_us_equity_session,
    us_equity_session_close_timestamp,
)


JOB_NAME = "market_path_pre_signal_context_backfill"
FEATURE_ROLE = "pre_signal_context"
FEATURE_VERSION = "market_path_pre_signal_v1"
RECONSTRUCTION_METHOD = "m4_pre_signal_context_fmp_eod_v1"
DEFAULT_PRE_SIGNAL_WINDOW = 20
DEFAULT_LOOKBACK_CALENDAR_DAYS = 520
ROW_STATUS_COMPUTED = "computed"
ROW_STATUS_INSUFFICIENT_HISTORY = "insufficient_history"
ROW_STATUS_OUTSIDE_UNIVERSE = "outside_universe_coverage"
ROW_STATUS_FETCH_ERROR = "price_fetch_error"
RANK_STATUS_NOT_APPLICABLE = "not_applicable_predictor_row"
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class PreSignalBatch:
    pattern_id: str
    signal_start_date: date
    signal_end_date: date


@dataclass(frozen=True)
class _SignalWindow:
    signal: SignalRegistry
    signal_date: date
    feature_dates_by_relative_index: dict[int, date]


@dataclass
class _MergeResult:
    rows_staged: int = 0
    rows_merged: int = 0
    stage_load_seconds: float = 0.0
    merge_seconds: float = 0.0


class MarketPathPreSignalContextJob(BaseJob):
    """Build T-k daily predictor rows for fired M4 signals."""

    def __init__(
        self,
        *,
        session: Session,
        fmp_adapter: Any,
        pattern_ids: Sequence[str] = ("M4",),
        signal_start_date: date,
        signal_end_date: date,
        run_timestamp: datetime | None = None,
        pre_signal_window: int = DEFAULT_PRE_SIGNAL_WINDOW,
        batch_days: int = 5,
        lookback_calendar_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS,
        feature_role: str = FEATURE_ROLE,
        feature_version: str = FEATURE_VERSION,
        signal_source: str = SIGNAL_SOURCE_HISTORICAL_M4_REPLAY,
        progress_callback: ProgressCallback | None = None,
        progress_artifact: str | Path | None = None,
        progress_every: int = 10,
    ) -> None:
        if pre_signal_window < 1:
            raise ValueError("pre_signal_window must be >= 1")
        if batch_days < 1:
            raise ValueError("batch_days must be >= 1")
        if progress_every < 1:
            raise ValueError("progress_every must be >= 1")
        self._session = session
        self._fmp = fmp_adapter
        self._pattern_ids = tuple(_unique_patterns(pattern_ids))
        self._signal_start_date = signal_start_date
        self._signal_end_date = signal_end_date
        self._run_timestamp = run_timestamp
        self._pre_signal_window = pre_signal_window
        self._batch_days = batch_days
        self._lookback_calendar_days = lookback_calendar_days
        self._feature_role = feature_role
        self._feature_version = feature_version
        self._signal_source = normalize_signal_source(signal_source)
        self._progress_callback = progress_callback
        self._progress_artifact = Path(progress_artifact) if progress_artifact else None
        self._progress_every = progress_every
        self.partial_metrics: dict[str, Any] = {}

    @property
    def job_name(self) -> str:
        return JOB_NAME

    @property
    def job_type(self) -> str:
        return "feature_enrichment"

    def run(self, ctx: JobContext) -> JobResult:
        started = time.perf_counter()
        if self._signal_start_date > self._signal_end_date:
            return JobResult(
                status="failed",
                errors=[{"message": "signal_start_date must be on or before signal_end_date"}],
            )

        run_ts = _ensure_aware(self._run_timestamp or ctx.started_at)
        batches = plan_pre_signal_batches(
            self._pattern_ids,
            signal_start_date=self._signal_start_date,
            signal_end_date=self._signal_end_date,
            batch_days=self._batch_days,
        )
        artifact = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "job": JOB_NAME,
            "feature_role": self._feature_role,
            "feature_version": self._feature_version,
            "pre_signal_window": self._pre_signal_window,
            "signal_source": self._signal_source,
            "selection_bias": {
                "conditional_on_fire": True,
                "negative_no_fire_cohort_deferred_to_manifest_stage": True,
            },
            "batches": [],
            "summary": {},
        }
        _write_artifact(self._progress_artifact, artifact)

        totals: dict[str, Any] = defaultdict(int)
        fetch_errors: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(batches, start=1):
            batch_started = time.perf_counter()
            batch_record: dict[str, Any] = {
                "batch_index": batch_index,
                "batch_count": len(batches),
                "pattern_id": batch.pattern_id,
                "signal_start_date": batch.signal_start_date.isoformat(),
                "signal_end_date": batch.signal_end_date.isoformat(),
                "status": "running",
                "progress_events": [],
            }
            artifact["batches"].append(batch_record)
            self._emit(
                artifact,
                batch_record,
                "batch_start",
                {
                    "batch_index": batch_index,
                    "batch_count": len(batches),
                    "pattern_id": batch.pattern_id,
                    "signal_start_date": batch.signal_start_date.isoformat(),
                    "signal_end_date": batch.signal_end_date.isoformat(),
                },
            )
            result = self._run_batch(
                ctx,
                batch,
                run_ts=run_ts,
                artifact=artifact,
                batch_record=batch_record,
            )
            for key, value in result.items():
                if isinstance(value, int):
                    totals[key] += value
            fetch_errors.extend(result.get("fetch_errors", []))
            batch_record.update({
                **result,
                "status": "partial_failed" if result.get("fetch_errors") else "finished",
                "elapsed_seconds": round(time.perf_counter() - batch_started, 6),
            })
            self._emit(
                artifact,
                batch_record,
                "batch_finish",
                {
                    "status": batch_record["status"],
                    "context_rows_merged": result.get("context_rows_merged", 0),
                    "link_rows_merged": result.get("link_rows_merged", 0),
                    "elapsed_seconds": batch_record["elapsed_seconds"],
                },
            )
            _write_artifact(self._progress_artifact, artifact)

        metrics = {
            "pattern_ids": list(self._pattern_ids),
            "signal_source": self._signal_source,
            "signal_start_date": self._signal_start_date.isoformat(),
            "signal_end_date": self._signal_end_date.isoformat(),
            "feature_role": self._feature_role,
            "feature_version": self._feature_version,
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "pre_signal_window": self._pre_signal_window,
            "batch_count": len(batches),
            "conditional_on_fire": True,
            "selection_bias": {
                "row_population": "fired_signal_windows_only",
                "manifest_requires_no_fire_negative_cohort": True,
            },
            "rank_status": RANK_STATUS_NOT_APPLICABLE,
            "fetch_error_count": len(fetch_errors),
            "fetch_error_sample": fetch_errors[:10],
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            **dict(totals),
        }
        artifact["summary"] = metrics
        artifact["ended_at"] = datetime.now(timezone.utc).isoformat()
        _write_artifact(self._progress_artifact, artifact)
        self.partial_metrics = metrics
        return JobResult(
            status="partial_failed" if fetch_errors else "finished",
            metrics=metrics,
            errors=fetch_errors,
        )

    def _run_batch(
        self,
        ctx: JobContext,
        batch: PreSignalBatch,
        *,
        run_ts: datetime,
        artifact: dict[str, Any],
        batch_record: dict[str, Any],
    ) -> dict[str, Any]:
        signals = self._signals(batch.signal_start_date, batch.signal_end_date, (batch.pattern_id,))
        windows = [_signal_window(signal, self._pre_signal_window) for signal in signals]
        by_ticker: dict[str, list[_SignalWindow]] = defaultdict(list)
        for window in windows:
            by_ticker[window.signal.ticker.upper()].append(window)

        all_feature_dates = sorted({
            feature_date
            for window in windows
            for feature_date in window.feature_dates_by_relative_index.values()
        })
        coverage_start = (
            all_feature_dates[0] - timedelta(days=self._lookback_calendar_days)
            if all_feature_dates else None
        )
        coverage = (
            self._hur_coverage(set(by_ticker), coverage_start, all_feature_dates[-1])
            if by_ticker and all_feature_dates else set()
        )
        existing_contexts = self._existing_contexts(set(by_ticker), all_feature_dates)
        existing_links = self._existing_links([signal.signal_id for signal in signals])

        context_rows: list[dict[str, Any]] = []
        link_rows: list[dict[str, Any]] = []
        lineages: list[DataLineage] = []
        fetch_errors: list[dict[str, Any]] = []
        row_status_counts: dict[str, int] = defaultdict(int)
        context_inserted = context_material_updates = context_unchanged = 0
        link_inserted = link_updated = 0
        ticker_fetch_started = ticker_fetch_finished = ticker_fetch_errors = 0
        ticker_planned_count = len(by_ticker)

        for ticker, ticker_windows in sorted(by_ticker.items()):
            unique_feature_dates = sorted({
                feature_date
                for window in ticker_windows
                for feature_date in window.feature_dates_by_relative_index.values()
            })
            if not unique_feature_dates:
                continue
            ticker_hur_dates = {
                day
                for covered_ticker, day in coverage
                if covered_ticker == ticker
            }
            hur_dates = {day for day in unique_feature_dates if day in ticker_hur_dates}
            bars = []
            lineage: DataLineage | None = None
            if hur_dates:
                from_date = unique_feature_dates[0] - timedelta(days=self._lookback_calendar_days)
                to_date = unique_feature_dates[-1]
                ticker_fetch_started += 1
                self._emit(
                    artifact,
                    batch_record,
                    "ticker_fetch_start",
                    {
                        "ticker": ticker,
                        "ticker_count": ticker_planned_count,
                        "ticker_fetch_started_count": ticker_fetch_started,
                        "ticker_fetch_finished_count": ticker_fetch_finished,
                        "ticker_fetch_error_count": ticker_fetch_errors,
                        "from_date": from_date.isoformat(),
                        "through_date": to_date.isoformat(),
                    },
                )
                fetch_started = time.perf_counter()
                resp = self._fmp.get_historical_price(
                    ticker,
                    from_date=from_date,
                    to_date=to_date,
                    asof=run_ts,
                    adjusted=False,
                )
                ticker_fetch_finished += 1
                if not resp.ok or resp.data is None:
                    ticker_fetch_errors += 1
                    error = {
                        "ticker": ticker,
                        "stage": "fmp_historical_price",
                        "message": sanitize_provider_error_message(
                            getattr(resp.error, "message", "missing response")
                        ),
                        "error_type": getattr(resp.error, "error_type", None),
                        "retryable": getattr(resp.error, "retryable", None),
                        "status_code": getattr(resp.error, "status_code", None),
                        "provider": getattr(resp.error, "provider", None),
                    }
                    fetch_errors.append(error)
                    self._emit(
                        artifact,
                        batch_record,
                        "ticker_fetch_error",
                        {
                            **error,
                            "ticker_count": ticker_planned_count,
                            "ticker_fetch_started_count": ticker_fetch_started,
                            "ticker_fetch_finished_count": ticker_fetch_finished,
                            "ticker_fetch_error_count": ticker_fetch_errors,
                            "elapsed_seconds": round(time.perf_counter() - fetch_started, 6),
                        },
                    )
                else:
                    bars = _clean_bars(resp.data, ticker=ticker)
                    lineage = _build_data_lineage(
                        provider="FMP",
                        endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                        asof_timestamp=run_ts,
                        raw_payload=_bar_lineage_payload(
                            symbol=ticker,
                            from_date=from_date,
                            through_date=to_date,
                            bars=bars,
                            feature_version=self._feature_version,
                            symbol_field="ticker",
                            source_role="pre_signal_context",
                            reconstruction_method=RECONSTRUCTION_METHOD,
                        ),
                        source_authority="fmp_eod",
                        data_quality_flags=_lineage_quality_flags(
                            resp,
                            derived_feature_replay=True,
                            pre_signal_context=True,
                            lineage_payload_schema="compact_bar_digest_v1",
                            conditional_on_fire=True,
                            adapter_raw_payload_hash=resp.lineage.raw_payload_hash,
                        ),
                        job_run_id=ctx.job_run_id,
                    )
                    self._emit(
                        artifact,
                        batch_record,
                        "ticker_fetch_finish",
                        {
                            "ticker": ticker,
                            "ticker_count": ticker_planned_count,
                            "ticker_fetch_started_count": ticker_fetch_started,
                            "ticker_fetch_finished_count": ticker_fetch_finished,
                            "ticker_fetch_error_count": ticker_fetch_errors,
                            "bar_count": len(bars),
                            "elapsed_seconds": round(time.perf_counter() - fetch_started, 6),
                        },
                    )

            for feature_date in unique_feature_dates:
                row = _context_row(
                    ticker=ticker,
                    feature_date=feature_date,
                    bars=bars,
                    hur_included=(ticker, feature_date) in coverage,
                    hur_included_dates=ticker_hur_dates,
                    fetch_failed=bool(hur_dates and lineage is None),
                    data_lineage_id=lineage.data_lineage_id if lineage is not None else None,
                    job_run_id=ctx.job_run_id,
                    run_ts=run_ts,
                    feature_role=self._feature_role,
                    feature_version=self._feature_version,
                )
                row_status_counts[row["row_status"]] += 1
                existing = existing_contexts.get(
                    (ticker, feature_date, self._feature_role, self._feature_version)
                )
                if existing is None:
                    context_inserted += 1
                    context_rows.append(row)
                elif _context_materially_changed(existing, row):
                    context_material_updates += 1
                    context_rows.append(row)
                else:
                    context_unchanged += 1

            if lineage is not None and any(
                row.get("ticker") == ticker and row.get("data_lineage_id") == lineage.data_lineage_id
                for row in context_rows
            ):
                lineages.append(lineage)

            for window in ticker_windows:
                for relative_index, feature_date in window.feature_dates_by_relative_index.items():
                    link = _link_row(
                        signal=window.signal,
                        signal_date=window.signal_date,
                        ticker=ticker,
                        feature_date=feature_date,
                        relative_index=relative_index,
                        feature_role=self._feature_role,
                        feature_version=self._feature_version,
                        job_run_id=ctx.job_run_id,
                    )
                    key = (
                        window.signal.signal_id,
                        feature_date,
                        self._feature_role,
                        self._feature_version,
                    )
                    if key in existing_links:
                        link_updated += 1
                    else:
                        link_inserted += 1
                    link_rows.append(link)

            if ticker_fetch_finished and ticker_fetch_finished % self._progress_every == 0:
                self._emit(
                    artifact,
                    batch_record,
                    "context_rows_generated",
                    {
                        "ticker": ticker,
                        "pending_context_rows": len(context_rows),
                        "pending_link_rows": len(link_rows),
                        "context_inserted": context_inserted,
                        "context_material_updates": context_material_updates,
                        "context_unchanged": context_unchanged,
                    },
                )

        if lineages:
            self._session.add_all(lineages)
            self._session.flush()
        context_merge = bulk_stage_merge_pre_signal_contexts(self._session, context_rows)
        link_merge = bulk_stage_merge_pre_signal_links(self._session, link_rows)
        self._session.commit()
        return {
            "signals_scanned": len(signals),
            "ticker_planned_count": ticker_planned_count,
            "ticker_fetch_started_count": ticker_fetch_started,
            "ticker_fetch_finished_count": ticker_fetch_finished,
            "ticker_fetch_error_count": ticker_fetch_errors,
            "fmp_fetch_count": _adapter_cache_misses(self._fmp),
            "lineages_recorded": len(lineages),
            "context_rows_inserted": context_inserted,
            "context_rows_material_updates": context_material_updates,
            "context_rows_unchanged": context_unchanged,
            "context_rows_merged": context_merge.rows_merged,
            "link_rows_inserted": link_inserted,
            "link_rows_updated": link_updated,
            "link_rows_merged": link_merge.rows_merged,
            "row_status_counts": dict(row_status_counts),
            "computed_rows": row_status_counts.get(ROW_STATUS_COMPUTED, 0),
            "insufficient_history_rows": row_status_counts.get(ROW_STATUS_INSUFFICIENT_HISTORY, 0),
            "outside_universe_coverage_rows": row_status_counts.get(ROW_STATUS_OUTSIDE_UNIVERSE, 0),
            "fetch_error_rows": row_status_counts.get(ROW_STATUS_FETCH_ERROR, 0),
            "fetch_errors": fetch_errors,
            "stage_load_seconds": round(
                context_merge.stage_load_seconds + link_merge.stage_load_seconds,
                6,
            ),
            "merge_seconds": round(context_merge.merge_seconds + link_merge.merge_seconds, 6),
        }

    def _signals(
        self,
        start: date,
        end: date,
        pattern_ids: Sequence[str],
    ) -> list[SignalRegistry]:
        start_dt = datetime.combine(start, datetime.min.time(), timezone.utc)
        end_dt = datetime.combine(end + timedelta(days=1), datetime.min.time(), timezone.utc)
        query = self._session.query(SignalRegistry).filter(
            SignalRegistry.pattern_id.in_(tuple(pattern_ids)),
            SignalRegistry.signal_timestamp >= start_dt,
            SignalRegistry.signal_timestamp < end_dt,
        )
        query = apply_signal_source_filter(
            query,
            self._session,
            signal_source=self._signal_source,
            signal_start_date=start,
            signal_end_date=end,
        )
        return query.order_by(SignalRegistry.ticker, SignalRegistry.signal_timestamp).all()

    def _hur_coverage(
        self,
        tickers: set[str],
        start: date,
        end: date,
    ) -> set[tuple[str, date]]:
        if not tickers:
            return set()
        rows = (
            self._session.query(
                HistoricalUniverseReconstruction.normalized_symbol,
                HistoricalUniverseReconstruction.replay_date,
            )
            .filter(
                HistoricalUniverseReconstruction.normalized_symbol.in_(tuple(sorted(tickers))),
                HistoricalUniverseReconstruction.replay_date >= start,
                HistoricalUniverseReconstruction.replay_date <= end,
                HistoricalUniverseReconstruction.inclusion_status == "included",
            )
            .all()
        )
        return {(str(symbol).upper(), replay_date) for symbol, replay_date in rows}

    def _existing_contexts(
        self,
        tickers: set[str],
        feature_dates: Sequence[date],
    ) -> dict[tuple[str, date, str, str], MarketPathPreSignalContext]:
        if not tickers or not feature_dates:
            return {}
        rows = (
            self._session.query(MarketPathPreSignalContext)
            .filter(
                MarketPathPreSignalContext.ticker.in_(tuple(sorted(tickers))),
                MarketPathPreSignalContext.feature_session_date >= feature_dates[0],
                MarketPathPreSignalContext.feature_session_date <= feature_dates[-1],
                MarketPathPreSignalContext.feature_role == self._feature_role,
                MarketPathPreSignalContext.feature_version == self._feature_version,
            )
            .all()
        )
        return {
            (row.ticker, row.feature_session_date, row.feature_role, row.feature_version): row
            for row in rows
        }

    def _existing_links(
        self,
        signal_ids: Sequence[str],
    ) -> set[tuple[str, date, str, str]]:
        if not signal_ids:
            return set()
        rows = (
            self._session.query(
                MarketPathPreSignalLink.signal_id,
                MarketPathPreSignalLink.feature_session_date,
                MarketPathPreSignalLink.feature_role,
                MarketPathPreSignalLink.feature_version,
            )
            .filter(
                MarketPathPreSignalLink.signal_id.in_(tuple(signal_ids)),
                MarketPathPreSignalLink.feature_role == self._feature_role,
                MarketPathPreSignalLink.feature_version == self._feature_version,
            )
            .all()
        )
        return {(signal_id, feature_date, role, version) for signal_id, feature_date, role, version in rows}

    def _emit(
        self,
        artifact: dict[str, Any],
        batch_record: dict[str, Any] | None,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        record = {
            "event": event,
            "at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        if batch_record is not None:
            batch_record.setdefault("progress_events", []).append(record)
            batch_record["last_progress_event"] = event
        else:
            artifact.setdefault("progress_events", []).append(record)
            artifact["last_progress_event"] = event
        _write_artifact(self._progress_artifact, artifact)
        if self._progress_callback is not None:
            self._progress_callback(event, payload)


def plan_pre_signal_batches(
    pattern_ids: Iterable[str],
    *,
    signal_start_date: date,
    signal_end_date: date,
    batch_days: int,
) -> list[PreSignalBatch]:
    if batch_days < 1:
        raise ValueError("batch_days must be >= 1")
    batches: list[PreSignalBatch] = []
    for pattern_id in _unique_patterns(pattern_ids):
        cursor = signal_start_date
        while cursor <= signal_end_date:
            batch_end = min(signal_end_date, cursor + timedelta(days=batch_days - 1))
            batches.append(PreSignalBatch(pattern_id, cursor, batch_end))
            cursor = batch_end + timedelta(days=1)
    return batches


def bulk_stage_merge_pre_signal_contexts(
    session: Session,
    rows: Sequence[dict[str, Any]],
) -> _MergeResult:
    return _bulk_stage_merge(
        session,
        rows,
        table=MarketPathPreSignalContext.__table__,
        conflict_columns=("ticker", "feature_session_date", "feature_role", "feature_version"),
        stage_prefix="_tmp_market_path_pre_signal_contexts",
    )


def bulk_stage_merge_pre_signal_links(
    session: Session,
    rows: Sequence[dict[str, Any]],
) -> _MergeResult:
    return _bulk_stage_merge(
        session,
        rows,
        table=MarketPathPreSignalLink.__table__,
        conflict_columns=("signal_id", "feature_session_date", "feature_role", "feature_version"),
        stage_prefix="_tmp_market_path_pre_signal_links",
    )


def _bulk_stage_merge(
    session: Session,
    rows: Sequence[dict[str, Any]],
    *,
    table: Table,
    conflict_columns: Sequence[str],
    stage_prefix: str,
) -> _MergeResult:
    if not rows:
        return _MergeResult()
    dialect_name = session.get_bind().dialect.name
    if dialect_name == "sqlite":
        started = time.perf_counter()
        insert_factory = sqlite_insert
        for batch in _batched(rows, 100):
            stmt = insert_factory(table).values(list(batch))
            excluded = stmt.excluded
            update_columns = {
                column.name: getattr(excluded, column.name)
                for column in table.columns
                if column.name not in set(conflict_columns) | {"created_at"}
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=list(conflict_columns),
                set_=update_columns,
            )
            session.execute(stmt)
        return _MergeResult(
            rows_staged=len(rows),
            rows_merged=len(rows),
            merge_seconds=round(time.perf_counter() - started, 6),
        )

    stage_name = f"{stage_prefix}_{uuid4().hex}"
    quoted_stage = _quote_ident(stage_name)
    target_name = table.name
    session.execute(text(
        f"CREATE TEMP TABLE {quoted_stage} "
        f"(LIKE {_quote_ident(target_name)} INCLUDING DEFAULTS) ON COMMIT DROP"
    ))
    stage_table = _stage_table(stage_name, table)
    stage_started = time.perf_counter()
    session.execute(stage_table.insert(), list(rows))
    stage_seconds = time.perf_counter() - stage_started
    column_names = [column.name for column in table.columns]
    column_sql = ", ".join(_quote_ident(column) for column in column_names)
    update_sql = ", ".join(
        f"{_quote_ident(column)} = EXCLUDED.{_quote_ident(column)}"
        for column in column_names
        if column not in set(conflict_columns) | {"created_at"}
    )
    conflict_sql = ", ".join(_quote_ident(column) for column in conflict_columns)
    merge_started = time.perf_counter()
    session.execute(text(
        f"INSERT INTO {_quote_ident(target_name)} "
        f"({column_sql}) SELECT {column_sql} FROM {quoted_stage} "
        f"ON CONFLICT ({conflict_sql}) DO UPDATE SET {update_sql}"
    ))
    return _MergeResult(
        rows_staged=len(rows),
        rows_merged=len(rows),
        stage_load_seconds=round(stage_seconds, 6),
        merge_seconds=round(time.perf_counter() - merge_started, 6),
    )


def validate_pre_signal_context_backfill(
    session: Session,
    *,
    feature_start_date: date,
    feature_end_date: date,
    feature_role: str = FEATURE_ROLE,
    feature_version: str = FEATURE_VERSION,
) -> dict[str, int]:
    params = {
        "feature_start": feature_start_date,
        "feature_end": feature_end_date,
        "feature_role": feature_role,
        "feature_version": feature_version,
    }
    duplicate_groups = session.execute(
        text(
            "SELECT COUNT(*) FROM ("
            "SELECT ticker, feature_session_date, feature_role, feature_version, COUNT(*) "
            "FROM market_path_pre_signal_contexts "
            "WHERE feature_session_date >= :feature_start "
            "AND feature_session_date <= :feature_end "
            "AND feature_role = :feature_role "
            "AND feature_version = :feature_version "
            "GROUP BY ticker, feature_session_date, feature_role, feature_version "
            "HAVING COUNT(*) > 1"
            ") d"
        ),
        params,
    ).scalar()
    row = session.execute(
        text(
            "SELECT "
            "COUNT(*) AS scoped_context_rows, "
            "COUNT(*) FILTER (WHERE input_hash IS NULL OR output_hash IS NULL "
            "OR feature_json IS NULL) AS missing_hash_rows, "
            "COUNT(*) FILTER (WHERE rank_status <> :rank_status) AS ranked_rows "
            "FROM market_path_pre_signal_contexts "
            "WHERE feature_session_date >= :feature_start "
            "AND feature_session_date <= :feature_end "
            "AND feature_role = :feature_role "
            "AND feature_version = :feature_version"
        ),
        {**params, "rank_status": RANK_STATUS_NOT_APPLICABLE},
    ).mappings().one()
    return {
        "duplicate_context_groups": int(duplicate_groups or 0),
        "scoped_context_row_count": int(row["scoped_context_rows"] or 0),
        "missing_hash_count": int(row["missing_hash_rows"] or 0),
        "ranked_context_row_count": int(row["ranked_rows"] or 0),
    }


def _signal_window(signal: SignalRegistry, pre_signal_window: int) -> _SignalWindow:
    signal_date = signal.signal_timestamp.date()
    feature_dates: dict[int, date] = {}
    cursor = signal_date
    for relative_index in range(-1, -(pre_signal_window + 1), -1):
        cursor = previous_us_equity_session(cursor)
        feature_dates[relative_index] = cursor
    return _SignalWindow(
        signal=signal,
        signal_date=signal_date,
        feature_dates_by_relative_index=feature_dates,
    )


def _context_row(
    *,
    ticker: str,
    feature_date: date,
    bars: Sequence[Any],
    hur_included: bool,
    hur_included_dates: set[date] | None,
    fetch_failed: bool,
    data_lineage_id: str | None,
    job_run_id: str,
    run_ts: datetime,
    feature_role: str,
    feature_version: str,
) -> dict[str, Any]:
    hur_included_dates = hur_included_dates or set()
    by_date = {bar.date: bar for bar in bars}
    bar = by_date.get(feature_date)
    prior = [candidate for candidate in bars if candidate.date < feature_date]
    row_input_bars = [candidate for candidate in bars if candidate.date <= feature_date]
    if not hur_included:
        row_status = ROW_STATUS_OUTSIDE_UNIVERSE
        status_reason = "feature_session_outside_historical_universe_inclusion"
        row_input_bars = []
        bar = None
        prior = []
    elif fetch_failed:
        row_status = ROW_STATUS_FETCH_ERROR
        status_reason = "price_fetch_failed_for_hur_included_ticker"
        row_input_bars = []
        bar = None
        prior = []
    elif bar is None or len(prior) < 20:
        row_status = ROW_STATUS_INSUFFICIENT_HISTORY
        status_reason = "missing_feature_bar_or_less_than_20_prior_sessions"
    else:
        row_status = ROW_STATUS_COMPUTED
        status_reason = "computed"

    features = _empty_feature_values()
    status = {
        "row_status": row_status,
        "status_reason": status_reason,
        "hur_included_on_feature_date": hur_included,
        "rank_status": RANK_STATUS_NOT_APPLICABLE,
        "conditional_on_fire": True,
        "no_forward_labels": True,
    }
    if bar is not None:
        previous = prior[-1] if prior else None
        close_basis = _price_basis(bar)
        prev_close = _price_basis(previous) if previous is not None else None
        prior20 = prior[-20:]
        prior252_plus_current = [*prior[-251:], bar]
        trailing60_plus_current = [*prior[-59:], bar]
        range20 = _range_pct([*prior[-19:], bar])
        has_prior60 = len(trailing60_plus_current) >= 60
        has_prior252 = len(prior252_plus_current) >= 252
        boundary60 = (
            _long_window_identity_boundary(trailing60_plus_current, hur_included_dates)
            if has_prior60 else None
        )
        boundary252 = (
            _long_window_identity_boundary(prior252_plus_current, hur_included_dates)
            if has_prior252 else None
        )
        range60 = _range_pct(trailing60_plus_current) if has_prior60 and boundary60 is None else None
        low252 = (
            min((_price_basis(item) for item in prior252_plus_current), default=None)
            if has_prior252 and boundary252 is None else None
        )
        high252 = (
            max((_price_basis(item) for item in prior252_plus_current), default=None)
            if has_prior252 and boundary252 is None else None
        )
        features.update({
            "previous_close": prev_close,
            "open_price": bar.open,
            "high_price": bar.high,
            "low_price": bar.low,
            "close_price": close_basis,
            "volume": bar.volume,
            "split_adjusted_close": bar.split_adjusted_close,
            "adj_close": bar.adj_close,
            "dollar_volume": bar.dollar_volume,
            "sub_dollar": close_basis < 1.0,
            "median_volume_20d": _median([item.volume for item in prior20]),
            "median_dollar_volume_20d": _median([item.dollar_volume for item in prior20]),
            "return_1d": _safe_return(close_basis, prev_close),
            "return_5d": _safe_return(close_basis, _price_basis(prior[-5]) if len(prior) >= 5 else None),
            "return_20d": _safe_return(close_basis, _price_basis(prior[-20]) if len(prior) >= 20 else None),
            "sigma_20d": _sigma_close_to_close(prior20),
            "range_contraction_ratio_60d": _safe_ratio(range20, range60),
            "volume_trend_slope_60d": (
                _volume_trend_slope(trailing60_plus_current)
                if has_prior60 and boundary60 is None else None
            ),
            "base_depth_60d": (
                _base_depth(trailing60_plus_current)
                if has_prior60 and boundary60 is None else None
            ),
            "base_length_60d": (
                _base_length(trailing60_plus_current)
                if has_prior60 and boundary60 is None else None
            ),
            "off_low252": _safe_return(close_basis, low252),
            "dist_hi252": _safe_return(close_basis, high252),
        })
        features["volume_expansion_20d"] = _safe_ratio(
            features["volume"],
            features["median_volume_20d"],
        )
        status["prior_session_count"] = len(prior)
        status["prior252_available_count"] = len(prior252_plus_current)
        status["prior60_available_count"] = len(trailing60_plus_current)
        status["insufficient_history"] = {
            "prior20": len(prior) < 20,
            "prior60": not has_prior60,
            "prior252": not has_prior252,
        }
        boundary = _merge_identity_boundaries(boundary60, boundary252)
        if boundary is not None:
            status["long_window_identity_boundary"] = boundary
    else:
        status["prior_session_count"] = len(prior)

    bars_digest = stable_hash([_bar_payload(item) for item in row_input_bars])
    input_payload = {
        "ticker": ticker,
        "feature_session_date": feature_date.isoformat(),
        "feature_role": feature_role,
        "feature_version": feature_version,
        "reconstruction_method": RECONSTRUCTION_METHOD,
        "row_status": row_status,
        "hur_included": hur_included,
        "bars_through_feature_session_digest": bars_digest,
        "bars_through_feature_session_count": len(row_input_bars),
    }
    input_hash = stable_hash(input_payload)
    feature_json = {
        "schema_version": feature_version,
        "feature_role": feature_role,
        "reconstruction_method": RECONSTRUCTION_METHOD,
        "row_status": row_status,
        "row_input_hash": input_hash,
        "row_input_hash_schema": "bars_through_feature_session_digest_v1",
        "row_input_window_start": (
            row_input_bars[0].date.isoformat() if row_input_bars else None
        ),
        "row_input_window_end": feature_date.isoformat() if row_input_bars else None,
        "max_input_date": feature_date.isoformat(),
        "strict_no_lookahead": {
            "uses_only_bars_lte_feature_session_date": True,
            "signal_day_fields_structurally_absent": True,
            "forward_label_fields_structurally_absent": True,
        },
        "selection_bias": {
            "conditional_on_fire": True,
            "negative_no_fire_cohort_deferred_to_manifest_stage": True,
        },
        "rank_status": {
            "status": RANK_STATUS_NOT_APPLICABLE,
            "reason": "fired_cohort_pre_signal_ranks_are_circular",
        },
        "split_adjustment_caveats": {
            "retroactive_adjusted_price_level_caveat": True,
            "affected_price_level_fields": [
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "dollar_volume",
                "sub_dollar",
                "off_low252",
                "dist_hi252",
            ],
            "returns_and_sigma_are_split_invariant": True,
        },
        "long_window_aggregates": {
            "range_contraction_ratio_60d": features["range_contraction_ratio_60d"],
            "volume_trend_slope_60d": features["volume_trend_slope_60d"],
            "base_depth_60d": features["base_depth_60d"],
            "base_length_60d": features["base_length_60d"],
            "off_low252": features["off_low252"],
            "dist_hi252": features["dist_hi252"],
        },
        "status": status,
    }
    row = {
        "ticker": ticker,
        "feature_session_date": feature_date,
        "feature_role": feature_role,
        "feature_version": feature_version,
        "row_status": row_status,
        "asof_timestamp": us_equity_session_close_timestamp(feature_date),
        "reconstruction_method": RECONSTRUCTION_METHOD,
        **features,
        "rank_status": RANK_STATUS_NOT_APPLICABLE,
        "retroactive_adjustment_caveat": True,
        "conditional_on_fire": True,
        "feature_json": json.dumps(feature_json, sort_keys=True, default=str),
        "status_json": json.dumps(status, sort_keys=True, default=str),
        "source_provider": "FMP",
        "source_endpoint": HISTORICAL_PRICE_FULL_ENDPOINT,
        "data_lineage_id": data_lineage_id,
        "job_run_id": job_run_id,
        "input_hash": input_hash,
    }
    row["output_hash"] = stable_hash({
        key: value
        for key, value in row.items()
        if key not in {"data_lineage_id", "job_run_id"}
    })
    now = datetime.now(timezone.utc)
    row["created_at"] = now
    row["updated_at"] = now
    return row


def _link_row(
    *,
    signal: SignalRegistry,
    signal_date: date,
    ticker: str,
    feature_date: date,
    relative_index: int,
    feature_role: str,
    feature_version: str,
    job_run_id: str,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "signal_id": signal.signal_id,
        "ticker": ticker,
        "pattern_id": signal.pattern_id,
        "signal_date": signal_date,
        "feature_session_date": feature_date,
        "relative_session_index": relative_index,
        "feature_role": feature_role,
        "feature_version": feature_version,
        "job_run_id": job_run_id,
        "created_at": now,
        "updated_at": now,
    }


def _empty_feature_values() -> dict[str, Any]:
    return {
        "previous_close": None,
        "open_price": None,
        "high_price": None,
        "low_price": None,
        "close_price": None,
        "volume": None,
        "split_adjusted_close": None,
        "adj_close": None,
        "dollar_volume": None,
        "sub_dollar": None,
        "median_volume_20d": None,
        "median_dollar_volume_20d": None,
        "volume_expansion_20d": None,
        "return_1d": None,
        "return_5d": None,
        "return_20d": None,
        "sigma_20d": None,
        "range_contraction_ratio_60d": None,
        "volume_trend_slope_60d": None,
        "base_depth_60d": None,
        "base_length_60d": None,
        "off_low252": None,
        "dist_hi252": None,
    }


def _context_materially_changed(
    existing: MarketPathPreSignalContext,
    row: dict[str, Any],
) -> bool:
    return (
        existing.output_hash != row.get("output_hash")
        or existing.input_hash != row.get("input_hash")
        or existing.row_status != row.get("row_status")
        or existing.feature_json != row.get("feature_json")
    )


def _range_pct(bars: Sequence[Any]) -> float | None:
    if len(bars) < 2:
        return None
    last_close = _price_basis(bars[-1])
    if last_close <= 0:
        return None
    return (max(bar.high for bar in bars) - min(bar.low for bar in bars)) / last_close


def _volume_trend_slope(bars: Sequence[Any]) -> float | None:
    if len(bars) < 2:
        return None
    values = [math.log(max(float(bar.volume), 1.0)) for bar in bars]
    return _linear_slope(values)


def _linear_slope(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    xs = list(range(len(values)))
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(values)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0:
        return None
    return float(sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denom)


def _base_depth(bars: Sequence[Any]) -> float | None:
    if len(bars) < 2:
        return None
    peak: float | None = None
    max_drawdown = 0.0
    for bar in bars:
        close = _price_basis(bar)
        if peak is None or close > peak:
            peak = close
        if peak and peak > 0:
            max_drawdown = max(max_drawdown, 1.0 - close / peak)
    return max_drawdown


def _base_length(bars: Sequence[Any]) -> int | None:
    if len(bars) < 2:
        return None
    closes = [_price_basis(bar) for bar in bars]
    high = max(closes)
    high_index = max(index for index, value in enumerate(closes) if value == high)
    return len(closes) - high_index


def _safe_ratio(value: float | None, basis: float | None) -> float | None:
    if value is None or basis is None or basis == 0:
        return None
    return float(value) / float(basis)


def _long_window_identity_boundary(
    bars: Sequence[Any],
    hur_included_dates: set[date],
) -> dict[str, Any] | None:
    excluded = sorted({bar.date for bar in bars if bar.date not in hur_included_dates})
    if not excluded:
        return None
    return {
        "first_excluded_date": excluded[0].isoformat(),
        "excluded_count": len(excluded),
        "excluded_dates": [day.isoformat() for day in excluded],
    }


def _merge_identity_boundaries(
    boundary60: dict[str, Any] | None,
    boundary252: dict[str, Any] | None,
) -> dict[str, Any] | None:
    boundaries = [boundary for boundary in (boundary60, boundary252) if boundary is not None]
    if not boundaries:
        return None
    excluded_dates = sorted({
        str(day)
        for boundary in boundaries
        for day in boundary.get("excluded_dates", [])
    })
    first_excluded = excluded_dates[0] if excluded_dates else min(
        str(boundary["first_excluded_date"]) for boundary in boundaries
    )
    return {
        "first_excluded_date": first_excluded,
        "excluded_count": len(excluded_dates) if excluded_dates else sum(
            int(boundary["excluded_count"]) for boundary in boundaries
        ),
        "excluded_dates": excluded_dates,
        "windows": {
            key: boundary
            for key, boundary in (("prior60", boundary60), ("prior252", boundary252))
            if boundary is not None
        },
    }


def _stage_table(stage_name: str, source_table: Table) -> Table:
    metadata = MetaData()
    return Table(
        stage_name,
        metadata,
        *[Column(column.name, column.type) for column in source_table.columns],
    )


def _batched(rows: Sequence[dict[str, Any]], batch_size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for index in range(0, len(rows), batch_size):
        yield rows[index:index + batch_size]


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _unique_patterns(pattern_ids: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for pattern_id in pattern_ids:
        normalized = str(pattern_id).strip().upper()
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return tuple(out or ["M4"])


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _adapter_cache_misses(adapter: Any) -> int:
    return int(getattr(adapter, "cache_misses", 0))


def _write_artifact(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str))
