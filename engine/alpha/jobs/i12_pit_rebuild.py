"""PIT-clean I12 research rebuild with scoped historical quote replay.

This job is research/corpus construction only. It does not place orders and it
does not mutate the legacy deferred-PIT I12 corpus. Candidate features are built
at fixed historical decision timestamps using prior daily bars and minute bars
at or before the decision timestamp. Quote replay is scoped to persisted
candidate event windows.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading
import time as time_module
from concurrent.futures import TimeoutError as FuturesTimeoutError
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from sqlalchemy import cast, func, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from alpha.data.alpaca import AlpacaAdapter, AlpacaQuote
from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError, stable_hash
from alpha.data.fmp import FmpBar
from alpha.data.polygon import PolygonBar
from alpha.db.models import (
    HistoricalUniverseReconstruction,
    I12PitCandidate,
    I12PitCostReplay,
    I12PitQuoteReplay,
)
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.i12_historical_corpus import (
    MINUTE_BAR_ENDPOINT,
    _DailyBar,
    _MinuteBar,
    _clean_daily_bars,
    _clean_minute_bars,
    _safe_return,
    _sigma,
    _split_adjusted_low,
)
from alpha.jobs.i12_live_fill_test import (
    ALPACA_QUOTE_SIZE_BASIS,
    HALT_CONDITIONS,
)
from alpha.jobs.paper_execution import EASTERN
from alpha.jobs.watchdog import (
    DEFAULT_MAX_CONSECUTIVE_FETCH_TIMEOUTS,
    DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS,
    ProviderOutageCircuitBreaker,
    WatchdogState,
    call_with_daemon_deadline,
)
from alpha.market_calendar import (
    is_us_equity_session,
    next_us_equity_session,
    us_equity_session_close_time,
    us_equity_session_open_timestamp,
)


JOB_NAME = "i12_pit_rebuild"
I12_PATTERN_ID = "I12"
FEATURE_MANIFEST_VERSION = "i12_pit_decision_features_v1"
REBUILD_METHOD = "i12_pit_fixed_decision_minute_rebuild_v1"
DEFAULT_DECISION_TIMES = ("09:35", "09:40", "09:45", "10:00")
DEFAULT_MAX_QUOTE_AGE_SECONDS = 60.0
DEFAULT_QUOTE_WINDOW_BEFORE_SECONDS = 120.0
DEFAULT_QUOTE_WINDOW_AFTER_SECONDS = 5.0
DEFAULT_MAX_SPREAD_BPS = 200.0
DEFAULT_INTENDED_ORDER_USD = 250.0
DEFAULT_SLIPPAGE_BPS = 0.0
DEFAULT_VOLUME_PARTICIPATION_THRESHOLD = 0.05
VOLUME_PARTICIPATION_THRESHOLDS = (0.01, 0.025, 0.05, 0.10)
VOLUME_TRADEABILITY_OK_STATUS = "tradeable_volume"
VOLUME_TRADEABILITY_SKIP_STATUS = "skipped_cash"
VOLUME_WINDOW_BASIS_PRE_DECISION = "pre_decision_completed_minutes"
VOLUME_PRICE_BASIS_LAST_PRE_DECISION = "last_predecision_price_proxy"
SAME_DAY_EXIT_TIME = time(15, 55)
NEXT_OPEN_EXIT_OFFSET_MINUTES = 1
MIN_PRIOR_DAILY_SESSIONS = 20
REQUIRED_QUOTE_ROLES = ("entry", "same_day_exit", "next_open_exit")
QUOTE_REPLAY_DURABLE_COVERAGE_STATUSES = frozenset({"ok", "missing", "stale", "error"})
EXIT_ROLES = ("same_day_exit", "next_open_exit")
DEFAULT_TICKER_PROGRESS_INTERVAL = 250
DEFAULT_FETCH_DEADLINE_SECONDS = 120.0
CHILD_CANDIDATE_ID_QUERY_CHUNK_SIZE = 10000
PROGRESS_METRICS_STATEMENT_TIMEOUT_MS = 15_000
HUR_LOAD_OPERATIONAL_ERROR_RETRIES = 2
HUR_LOAD_RETRY_BACKOFF_SECONDS = 1.0
LEGACY_DAILY_SOURCE_HASHES_FLAG = "i12_pit_legacy_per_date_daily_source_hashes"
DAILY_SOURCE_HASH_BASIS_LEGACY = "legacy_per_date_raw_payload"
DAILY_SOURCE_HASH_BASIS_CLEAN_SLICE = "clean_slice_v1"
DAILY_SOURCE_HASH_BASIS_PROVIDER_ERROR = "provider_error"
DAILY_SOURCE_HASH_REUSE_FRESH = "fresh"
DAILY_SOURCE_HASH_REUSE_EXISTING_ACTIVE = "existing_active_attempt_reuse"
SOURCE_BARS_NON_IDENTITY_KEYS = frozenset({
    "daily_source_hash_basis",
    "daily_provider_fetch_hash",
    "daily_source_hash_reuse_status",
})
STRICT_MINUTE_PATH_MODE = "strict_contiguous"
SPARSE_ZERO_FILL_MINUTE_PATH_MODE = "sparse_zero_fill"
MINUTE_PATH_MODES = (
    STRICT_MINUTE_PATH_MODE,
    SPARSE_ZERO_FILL_MINUTE_PATH_MODE,
)
LEAKY_FEATURE_TOKENS = (
    "full_day",
    "same_day_close",
    "day_close",
    "session_close",
    "full_day_high",
    "full_day_low",
    "forward",
    "next_open",
    "exit",
    "label",
    "mfe",
    "mae",
)


@dataclass(frozen=True)
class PitCandidateResult:
    ticker: str
    decision_date: date
    decision_ts: datetime
    decision_time_label: str
    path_mode: str
    candidate_status: str
    coverage_status: str
    fail_reason: str | None
    feature_json: dict[str, Any]
    gate_values: dict[str, Any]
    leakage_guard: dict[str, Any]
    source_bars: dict[str, Any]
    label_json: dict[str, Any]
    feature_asof_ts: datetime | None
    candidate_attempt_hash: str
    input_hash: str
    candidate_identity_hash: str
    label_hash: str
    content_hash: str
    error_json: dict[str, Any] | None = None


@dataclass(frozen=True)
class HurSourceRow:
    ticker: str
    trading_date: date
    source_hur_identity_hash: str
    source_hur_payload: dict[str, Any]


@dataclass(frozen=True)
class _ReportCandidateAggregates:
    active_count: int
    active_decision_time_labels: list[str]
    candidate_status_counts: dict[str, int]
    candidate_coverage_status_counts: dict[str, int]
    candidate_counts_by_path_mode: dict[str, int]
    coverage_status_by_path_mode: dict[str, dict[str, int]]
    candidate_status_counts_by_path_mode: dict[str, dict[str, int]]
    daily_source_hash_basis_counts: dict[str, int]
    daily_source_hash_reuse_status_counts: dict[str, int]
    daily_source_hash_basis_counts_by_path_mode: dict[str, dict[str, int]]
    daily_source_hash_reuse_status_counts_by_path_mode: dict[str, dict[str, int]]
    decision_time_bucket_counts: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class QuoteWindow:
    quote_role: str
    target_ts: datetime
    window_start_ts: datetime
    window_end_ts: datetime


@dataclass(frozen=True)
class QuoteReplayResult:
    quote_role: str
    target_ts: datetime
    window_start_ts: datetime
    window_end_ts: datetime
    quote: AlpacaQuote | None
    quote_ts: datetime | None
    quote_age_seconds: float | None
    bid: float | None
    ask: float | None
    bid_size: float | None
    ask_size: float | None
    spread_bps: float | None
    top_of_book_notional: float | None
    bid_notional: float | None
    ask_notional: float | None
    executable_notional: float | None
    executable_side: str | None
    feed: str
    source: str
    coverage_status: str
    raw_json: dict[str, Any] | None
    error_json: dict[str, Any] | None


@dataclass(frozen=True)
class CostReplayResult:
    exit_role: str
    tradeability_status: str
    skipped_reason: str
    intended_order_usd: float
    max_spread_bps: float
    slippage_bps: float
    entry_ask: float | None
    exit_bid: float | None
    gross_return: float | None
    quote_cost_return: float | None
    slippage_return: float | None
    modeled_return: float


class _CleanCounters:
    def record_non_session(self, ticker: str, parsed_date: date) -> None:
        del ticker, parsed_date


class I12PitRebuildJob(BaseJob):
    """Build PIT-clean candidates, scoped SIP quote replay, and cost rows."""

    def __init__(
        self,
        *,
        session: Session,
        fmp_adapter: Any,
        polygon_adapter: Any,
        alpaca_adapter: AlpacaAdapter | None,
        start_date: date,
        end_date: date,
        decision_times: Sequence[str] = DEFAULT_DECISION_TIMES,
        intended_order_usd: float = DEFAULT_INTENDED_ORDER_USD,
        max_spread_bps: float = DEFAULT_MAX_SPREAD_BPS,
        max_quote_age_seconds: float = DEFAULT_MAX_QUOTE_AGE_SECONDS,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
        feed: str = "sip",
        minute_path_mode: str = STRICT_MINUTE_PATH_MODE,
        skip_existing: bool = True,
        replace_existing: bool = False,
        quote_replay: bool = True,
        source_hur_schema: str = "public",
        output_schema: str | None = None,
        allow_source_hur_schema_matches_output: bool = False,
        progress_artifact: str | Path | None = None,
        progress_interval_tickers: int = DEFAULT_TICKER_PROGRESS_INTERVAL,
        fetch_deadline_seconds: float = DEFAULT_FETCH_DEADLINE_SECONDS,
        max_outstanding_fetch_timeouts: int = DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS,
        max_consecutive_fetch_timeouts: int = DEFAULT_MAX_CONSECUTIVE_FETCH_TIMEOUTS,
        max_no_progress_seconds: float | None = None,
        no_progress_exit_callback: Callable[[Mapping[str, Any]], None] | None = None,
        progress_session_factory: Callable[[], Session] | None = None,
        skip_final_report: bool = False,
    ) -> None:
        if fetch_deadline_seconds <= 0:
            raise ValueError("fetch_deadline_seconds must be > 0")
        if max_outstanding_fetch_timeouts < 1:
            raise ValueError("max_outstanding_fetch_timeouts must be >= 1")
        if max_consecutive_fetch_timeouts < 1:
            raise ValueError("max_consecutive_fetch_timeouts must be >= 1")
        if max_no_progress_seconds is not None and max_no_progress_seconds <= 0:
            raise ValueError("max_no_progress_seconds must be > 0 when provided")
        self._session = session
        self._fmp = fmp_adapter
        self._polygon = polygon_adapter
        self._alpaca = alpaca_adapter
        self._start_date = start_date
        self._end_date = end_date
        self._decision_times = _normalize_decision_time_labels(decision_times)
        self._intended_order_usd = float(intended_order_usd)
        self._max_spread_bps = float(max_spread_bps)
        self._max_quote_age_seconds = float(max_quote_age_seconds)
        self._slippage_bps = float(slippage_bps)
        self._feed = feed
        self._minute_path_mode = _validate_minute_path_mode(minute_path_mode)
        self._skip_existing = skip_existing
        self._replace_existing = replace_existing
        self._quote_replay = quote_replay
        self._source_hur_schema = _validate_source_schema_name(source_hur_schema)
        self._output_schema = output_schema
        self._progress_artifact = Path(progress_artifact) if progress_artifact else None
        self._progress_interval_tickers = max(1, int(progress_interval_tickers))
        self._fetch_deadline_seconds = float(fetch_deadline_seconds)
        self._fetch_watchdog = WatchdogState(
            max_outstanding_timeouts=max_outstanding_fetch_timeouts,
            max_consecutive_timeouts=max_consecutive_fetch_timeouts,
        )
        self._daily_response_cache: dict[str, AdapterResponse[Any]] = {}
        self._max_no_progress_seconds = (
            float(max_no_progress_seconds)
            if max_no_progress_seconds is not None
            else None
        )
        self._no_progress_exit_callback = no_progress_exit_callback
        self._progress_state_lock = threading.Lock()
        self._skip_final_report = bool(skip_final_report)
        self._last_progress_monotonic = time_module.monotonic()
        self._last_progress_wall_clock_utc = datetime.now(timezone.utc)
        self._last_progress_event = "init"
        self._last_progress_payload: dict[str, Any] = {}
        self._no_progress_stop = threading.Event()
        self._no_progress_thread: threading.Thread | None = None
        self._progress_session_factory = progress_session_factory
        self.partial_metrics: dict[str, Any] = {}
        if (
            output_schema
            and self._source_hur_schema.casefold() == output_schema.casefold()
            and not allow_source_hur_schema_matches_output
        ):
            raise ValueError(
                "source_hur_schema must differ from the scratch output schema"
            )

    @property
    def job_name(self) -> str:
        return JOB_NAME

    @property
    def job_type(self) -> str:
        return "research_rebuild"

    def run(self, ctx: JobContext) -> JobResult:
        self._progress("start", {"job_run_id": ctx.job_run_id})
        self._start_no_progress_monitor(ctx.job_run_id)
        try:
            trading_dates = [
                self._start_date + timedelta(days=offset)
                for offset in range((self._end_date - self._start_date).days + 1)
                if is_us_equity_session(self._start_date + timedelta(days=offset))
            ]
            counters: Counter[str] = Counter()
            hur_rows_loaded = 0
            for trading_date in trading_dates:
                try:
                    hur_rows = self._load_hur_rows_with_retry(
                        trading_date,
                        job_run_id=ctx.job_run_id,
                    )
                except Exception:
                    if self._session.in_transaction():
                        self._session.rollback()
                    raise
                else:
                    if self._session.in_transaction():
                        self._session.rollback()
                hur_rows_loaded += len(hur_rows)
                self._progress(
                    "date_start",
                    self._heartbeat_payload(
                        counters=counters,
                        trading_date=trading_date,
                        hur_rows_for_date=len(hur_rows),
                        hur_rows_processed_for_date=0,
                        cumulative_hur_rows_loaded=hur_rows_loaded,
                        job_run_id=ctx.job_run_id,
                    ),
                )
                if not hur_rows:
                    counters["zero_hur_dates"] += 1
                for hur_index, hur_row in enumerate(hur_rows, start=1):
                    ticker = hur_row.ticker
                    daily_resp = self._fetch_fmp_daily_with_deadline(
                        ticker,
                        trading_date,
                        job_run_id=ctx.job_run_id,
                    )
                    if not daily_resp.ok:
                        daily_error = _provider_error_payload(daily_resp)
                        daily_source_hash = getattr(daily_resp.lineage, "raw_payload_hash", None)
                        daily_source_hash_basis = DAILY_SOURCE_HASH_BASIS_PROVIDER_ERROR
                        daily_provider_fetch_hash = daily_source_hash
                        for label in self._decision_times:
                            decision_ts = _decision_timestamp(trading_date, label)
                            result = _candidate_error(
                                ticker=ticker,
                                trading_date=trading_date,
                                decision_ts=decision_ts,
                                decision_time_label=label,
                                coverage_status="daily_fetch_error",
                                fail_reason="daily_fetch_error",
                                daily_source_hash=daily_source_hash,
                                daily_source_hash_basis=daily_source_hash_basis,
                                daily_provider_fetch_hash=daily_provider_fetch_hash,
                                minute_source_hash=None,
                                daily_error=daily_error,
                                minute_error=None,
                                source_hur_identity_hash=hur_row.source_hur_identity_hash,
                                source_hur=hur_row.source_hur_payload,
                                path_mode=self._minute_path_mode,
                            )
                            self._persist_candidate(result, ctx.job_run_id)
                            self._session.commit()
                            counters[f"candidate_{result.candidate_status}"] += 1
                            counters[f"coverage_{result.coverage_status}"] += 1
                        self._maybe_emit_ticker_progress(
                            counters=counters,
                            trading_date=trading_date,
                            hur_rows_for_date=len(hur_rows),
                            hur_rows_processed_for_date=hur_index,
                            cumulative_hur_rows_loaded=hur_rows_loaded,
                            current_ticker=ticker,
                            job_run_id=ctx.job_run_id,
                        )
                        continue
                    daily_bars = _daily_bars_for_trading_date(
                        ticker,
                        trading_date,
                        daily_resp.data or [],
                    )
                    (
                        daily_source_hash,
                        daily_source_hash_basis,
                        daily_provider_fetch_hash,
                    ) = _daily_source_hash_for_trading_date(
                        ticker,
                        trading_date,
                        daily_resp,
                        daily_bars,
                    )
                    prefilter_results = _daily_prefilter_results(
                        ticker=ticker,
                        trading_date=trading_date,
                        decision_time_labels=self._decision_times,
                        daily_bars=daily_bars,
                        daily_source_hash=daily_source_hash,
                        daily_source_hash_basis=daily_source_hash_basis,
                        daily_provider_fetch_hash=daily_provider_fetch_hash,
                        source_hur_identity_hash=hur_row.source_hur_identity_hash,
                        source_hur=hur_row.source_hur_payload,
                        path_mode=self._minute_path_mode,
                    )
                    if prefilter_results is not None:
                        for result in prefilter_results:
                            self._persist_candidate(result, ctx.job_run_id)
                            self._session.commit()
                            counters[f"candidate_{result.candidate_status}"] += 1
                            counters[f"coverage_{result.coverage_status}"] += 1
                        self._maybe_emit_ticker_progress(
                            counters=counters,
                            trading_date=trading_date,
                            hur_rows_for_date=len(hur_rows),
                            hur_rows_processed_for_date=hur_index,
                            cumulative_hur_rows_loaded=hur_rows_loaded,
                            current_ticker=ticker,
                            job_run_id=ctx.job_run_id,
                        )
                        continue
                    minute_resp = self._fetch_polygon_minutes_with_deadline(
                        ticker,
                        trading_date,
                        job_run_id=ctx.job_run_id,
                    )
                    if not minute_resp.ok:
                        minute_error = _provider_error_payload(minute_resp)
                        minute_source_hash = getattr(minute_resp.lineage, "raw_payload_hash", None)
                        for label in self._decision_times:
                            decision_ts = _decision_timestamp(trading_date, label)
                            result = _candidate_error(
                                ticker=ticker,
                                trading_date=trading_date,
                                decision_ts=decision_ts,
                                decision_time_label=label,
                                coverage_status="minute_fetch_error",
                                fail_reason="minute_fetch_error",
                                daily_source_hash=daily_source_hash,
                                daily_source_hash_basis=daily_source_hash_basis,
                                daily_provider_fetch_hash=daily_provider_fetch_hash,
                                minute_source_hash=minute_source_hash,
                                daily_error=None,
                                minute_error=minute_error,
                                source_hur_identity_hash=hur_row.source_hur_identity_hash,
                                source_hur=hur_row.source_hur_payload,
                                path_mode=self._minute_path_mode,
                            )
                            self._persist_candidate(result, ctx.job_run_id)
                            self._session.commit()
                            counters[f"candidate_{result.candidate_status}"] += 1
                            counters[f"coverage_{result.coverage_status}"] += 1
                        self._maybe_emit_ticker_progress(
                            counters=counters,
                            trading_date=trading_date,
                            hur_rows_for_date=len(hur_rows),
                            hur_rows_processed_for_date=hur_index,
                            cumulative_hur_rows_loaded=hur_rows_loaded,
                            current_ticker=ticker,
                            job_run_id=ctx.job_run_id,
                        )
                        continue
                    raw_minutes = minute_resp.data or []
                    minute_bars = _clean_minute_bars(trading_date, raw_minutes)
                    minute_source_hash = (
                        getattr(minute_resp.lineage, "raw_payload_hash", None)
                        if minute_resp.ok
                        else None
                    )
                    for label in self._decision_times:
                        decision_ts = _decision_timestamp(trading_date, label)
                        (
                            result_daily_source_hash,
                            result_minute_source_hash,
                            result_daily_source_hash_basis,
                            result_daily_provider_fetch_hash,
                            result_daily_source_hash_reuse_status,
                        ) = self._candidate_attempt_source_hashes(
                            ticker=ticker,
                            trading_date=trading_date,
                            decision_ts=decision_ts,
                            decision_time_label=label,
                            source_hur_identity_hash=hur_row.source_hur_identity_hash,
                            path_mode=self._minute_path_mode,
                            daily_source_hash=daily_source_hash,
                            minute_source_hash=minute_source_hash,
                            daily_source_hash_basis=daily_source_hash_basis,
                            daily_provider_fetch_hash=daily_provider_fetch_hash,
                        )
                        result = build_i12_pit_candidate(
                            ticker=ticker,
                            trading_date=trading_date,
                            decision_ts=decision_ts,
                            decision_time_label=label,
                            daily_bars=daily_bars,
                            minute_bars=minute_bars,
                            daily_source_hash=result_daily_source_hash,
                            daily_source_hash_basis=result_daily_source_hash_basis,
                            daily_provider_fetch_hash=result_daily_provider_fetch_hash,
                            daily_source_hash_reuse_status=(
                                result_daily_source_hash_reuse_status
                            ),
                            minute_source_hash=result_minute_source_hash,
                            minute_error=None if minute_resp.ok else _provider_error_payload(minute_resp),
                            source_hur_identity_hash=hur_row.source_hur_identity_hash,
                            source_hur=hur_row.source_hur_payload,
                            path_mode=self._minute_path_mode,
                        )
                        candidate = self._persist_candidate(result, ctx.job_run_id)
                        candidate_id = candidate.i12_pit_candidate_id
                        self._session.commit()
                        counters[f"candidate_{result.candidate_status}"] += 1
                        counters[f"coverage_{result.coverage_status}"] += 1
                        if (
                            self._quote_replay
                            and self._alpaca is not None
                            and result.candidate_status == "passed"
                        ):
                            quotes = self._replay_quotes(candidate_id, ctx.job_run_id)
                            candidate_for_costs = self._session.get(
                                I12PitCandidate,
                                candidate_id,
                            )
                            if candidate_for_costs is None:
                                raise RuntimeError(
                                    f"I12 PIT candidate disappeared before cost replay: "
                                    f"{candidate_id}"
                                )
                            self._persist_costs(candidate_for_costs, quotes, ctx.job_run_id)
                            self._session.commit()
                            counters["quote_replayed_candidates"] += 1
                    self._maybe_emit_ticker_progress(
                        counters=counters,
                        trading_date=trading_date,
                        hur_rows_for_date=len(hur_rows),
                        hur_rows_processed_for_date=hur_index,
                        cumulative_hur_rows_loaded=hur_rows_loaded,
                        current_ticker=ticker,
                        job_run_id=ctx.job_run_id,
                    )
                self._session.commit()
                date_finish_payload = self._heartbeat_payload(
                    counters=counters,
                    trading_date=trading_date,
                    hur_rows_for_date=len(hur_rows),
                    hur_rows_processed_for_date=len(hur_rows),
                    cumulative_hur_rows_loaded=hur_rows_loaded,
                    job_run_id=ctx.job_run_id,
                )
                self._progress("date_loop_complete", date_finish_payload)
                self._reset_provider_sessions_at_date_boundary()
                self._progress("date_finish_metrics_start", date_finish_payload)
                metrics_updated = self._update_partial_metrics(
                    counters=counters,
                    hur_rows_loaded=hur_rows_loaded,
                    event="date_finish",
                    trading_date=trading_date,
                    job_run_id=ctx.job_run_id,
                )
                if not metrics_updated:
                    self._progress(
                        "date_finish_minimal",
                        {
                            **date_finish_payload,
                            "progress_metrics_session": "isolated",
                            "progress_metrics_status": "failed",
                            "progress_error": self.partial_metrics.get("progress_error"),
                        },
                    )
            if self._skip_final_report:
                metrics = self._worker_shard_finish_metrics(
                    counters=counters,
                    hur_rows_loaded=hur_rows_loaded,
                    trading_dates=trading_dates,
                    job_run_id=ctx.job_run_id,
                )
                self.partial_metrics = metrics
                self._progress("finish", metrics)
                return JobResult(status="finished", metrics=metrics)
            report = i12_pit_rebuild_report(
                self._session,
                output_schema=self._output_schema,
                source_hur_schema=self._source_hur_schema,
                hur_rows_loaded=hur_rows_loaded,
                decision_time_count=len(self._decision_times),
                decision_time_labels=self._decision_times,
                start_date=self._start_date,
                end_date=self._end_date,
                path_mode=self._minute_path_mode,
            )
            metrics = {
                "counters": dict(counters),
                "fetch_watchdog": self._fetch_watchdog.snapshot(),
                **report,
            }
            self.partial_metrics = metrics
            self._progress("finish", metrics)
            return JobResult(status="finished", metrics=metrics)
        finally:
            self._stop_no_progress_monitor()

    def _worker_shard_finish_metrics(
        self,
        *,
        counters: Counter[str],
        hur_rows_loaded: int,
        trading_dates: Sequence[date],
        job_run_id: str | None,
    ) -> dict[str, Any]:
        final_partial_metrics = dict(self.partial_metrics or {})
        last_completed_trading_date = trading_dates[-1] if trading_dates else None
        return {
            "job_run_id": job_run_id,
            "source_hur_schema": self._source_hur_schema,
            "start_date": self._start_date.isoformat(),
            "end_date": self._end_date.isoformat(),
            "decision_time_labels": list(self._decision_times),
            "decision_time_count": len(self._decision_times),
            "minute_path_mode": self._minute_path_mode,
            "hur_rows_loaded": hur_rows_loaded,
            "last_completed_trading_date": (
                last_completed_trading_date.isoformat()
                if last_completed_trading_date else None
            ),
            "counters": dict(counters),
            "fetch_watchdog": self._fetch_watchdog.snapshot(),
            "final_report_skipped": True,
            "training_status": "worker_shard_complete_pending_report",
            "shard_status": "worker_shard_complete_pending_report",
            "final_partial_metrics": final_partial_metrics,
        }

    def _load_hur_rows_with_retry(
        self,
        trading_date: date,
        *,
        job_run_id: str | None,
    ) -> list[HurSourceRow]:
        max_attempts = HUR_LOAD_OPERATIONAL_ERROR_RETRIES + 1
        for attempt in range(1, max_attempts + 1):
            try:
                return self._load_hur_rows(trading_date)
            except OperationalError as exc:
                if self._session.in_transaction():
                    self._session.rollback()
                if attempt >= max_attempts:
                    raise
                self._progress(
                    "hur_load_retry",
                    {
                        "job_run_id": job_run_id,
                        "trading_date": trading_date.isoformat(),
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                time_module.sleep(HUR_LOAD_RETRY_BACKOFF_SECONDS)
        raise RuntimeError("unreachable HUR load retry state")

    def _fetch_fmp_daily_with_deadline(
        self,
        ticker: str,
        trading_date: date,
        *,
        job_run_id: str | None,
    ) -> AdapterResponse[Any]:
        cache_key = ticker.upper()
        cached = self._daily_response_cache.get(cache_key)
        if cached is not None:
            return cached
        from_date = self._start_date - timedelta(days=460)
        to_date = next_us_equity_session(self._end_date + timedelta(days=2))
        stage = "fmp_daily_fetch"
        self._progress_provider_fetch_start(
            stage=stage,
            provider="FMP",
            ticker=ticker,
            trading_date=trading_date,
            job_run_id=job_run_id,
        )
        self._rollback_before_provider_call()

        def _fetch() -> AdapterResponse[Any]:
            return self._fmp.get_historical_price(
                ticker,
                from_date=from_date,
                to_date=to_date,
                adjusted=False,
            )

        try:
            response = call_with_daemon_deadline(
                _fetch,
                timeout_seconds=self._fetch_deadline_seconds,
                thread_name="i12-pit-fmp-daily-fetch",
                state=self._fetch_watchdog,
                context={
                    "stage": stage,
                    "provider": "FMP",
                    "ticker": ticker.upper(),
                    "trading_date": trading_date.isoformat(),
                    "deadline_seconds": self._fetch_deadline_seconds,
                },
            )
            if response.ok:
                self._daily_response_cache[cache_key] = response
            return response
        except ProviderOutageCircuitBreaker as exc:
            self._progress("provider_outage_circuit_breaker", exc.payload)
            if not _breaker_opened_by_current_watchdog_timeout(exc):
                raise
            self._reset_provider_session(self._fmp)
            self._progress_provider_fetch_timeout(
                stage=stage,
                provider="FMP",
                ticker=ticker,
                trading_date=trading_date,
                job_run_id=job_run_id,
            )
            return _provider_watchdog_timeout_response(
                provider="FMP",
                endpoint="historical_price_full",
                ticker=ticker,
                stage=stage,
                trading_date=trading_date,
                deadline_seconds=self._fetch_deadline_seconds,
                watchdog_snapshot=self._fetch_watchdog.snapshot(),
                extra={"provider_outage_circuit_breaker": exc.payload},
            )
        except FuturesTimeoutError:
            self._reset_provider_session(self._fmp)
            self._progress_provider_fetch_timeout(
                stage=stage,
                provider="FMP",
                ticker=ticker,
                trading_date=trading_date,
                job_run_id=job_run_id,
            )
            return _provider_watchdog_timeout_response(
                provider="FMP",
                endpoint="historical_price_full",
                ticker=ticker,
                stage=stage,
                trading_date=trading_date,
                deadline_seconds=self._fetch_deadline_seconds,
                watchdog_snapshot=self._fetch_watchdog.snapshot(),
            )

    def _fetch_polygon_minutes_with_deadline(
        self,
        ticker: str,
        trading_date: date,
        *,
        job_run_id: str | None,
    ) -> AdapterResponse[Any]:
        stage = "polygon_minute_fetch"
        self._progress_provider_fetch_start(
            stage=stage,
            provider="Polygon",
            ticker=ticker,
            trading_date=trading_date,
            job_run_id=job_run_id,
        )
        self._rollback_before_provider_call()

        def _fetch() -> AdapterResponse[Any]:
            return self._polygon.get_minute_aggs(
                ticker,
                trading_date.isoformat(),
                trading_date.isoformat(),
                adjusted=True,
            )

        try:
            return call_with_daemon_deadline(
                _fetch,
                timeout_seconds=self._fetch_deadline_seconds,
                thread_name="i12-pit-polygon-minute-fetch",
                state=self._fetch_watchdog,
                context={
                    "stage": stage,
                    "provider": "Polygon",
                    "ticker": ticker.upper(),
                    "trading_date": trading_date.isoformat(),
                    "deadline_seconds": self._fetch_deadline_seconds,
                },
            )
        except ProviderOutageCircuitBreaker as exc:
            self._progress("provider_outage_circuit_breaker", exc.payload)
            if not _breaker_opened_by_current_watchdog_timeout(exc):
                raise
            self._reset_provider_session(self._polygon)
            self._progress_provider_fetch_timeout(
                stage=stage,
                provider="Polygon",
                ticker=ticker,
                trading_date=trading_date,
                job_run_id=job_run_id,
            )
            return _provider_watchdog_timeout_response(
                provider="Polygon",
                endpoint=MINUTE_BAR_ENDPOINT.format(
                    ticker=ticker.upper(),
                    date=trading_date.isoformat(),
                ),
                ticker=ticker,
                stage=stage,
                trading_date=trading_date,
                deadline_seconds=self._fetch_deadline_seconds,
                watchdog_snapshot=self._fetch_watchdog.snapshot(),
                extra={"provider_outage_circuit_breaker": exc.payload},
            )
        except FuturesTimeoutError:
            self._reset_provider_session(self._polygon)
            self._progress_provider_fetch_timeout(
                stage=stage,
                provider="Polygon",
                ticker=ticker,
                trading_date=trading_date,
                job_run_id=job_run_id,
            )
            return _provider_watchdog_timeout_response(
                provider="Polygon",
                endpoint=MINUTE_BAR_ENDPOINT.format(
                    ticker=ticker.upper(),
                    date=trading_date.isoformat(),
                ),
                ticker=ticker,
                stage=stage,
                trading_date=trading_date,
                deadline_seconds=self._fetch_deadline_seconds,
                watchdog_snapshot=self._fetch_watchdog.snapshot(),
            )

    def _load_hur_rows(self, trading_date: date) -> list[HurSourceRow]:
        if _should_schema_qualify_hur(self._session, self._source_hur_schema):
            rows = self._session.execute(
                text(
                    "SELECT * "
                    f"FROM {_quote_ident(self._source_hur_schema)}."
                    "historical_universe_reconstructions "
                    "WHERE replay_date = :trading_date "
                    "AND inclusion_status = 'included' "
                    "ORDER BY normalized_symbol"
                ),
                {"trading_date": trading_date},
            ).all()
            return [
                _hur_source_row_from_mapping(
                    row._mapping,
                    source_schema=self._source_hur_schema,
                    fallback_date=trading_date,
                )
                for row in rows
            ]
        rows = (
            self._session.query(HistoricalUniverseReconstruction)
            .filter(
                HistoricalUniverseReconstruction.replay_date == trading_date,
                HistoricalUniverseReconstruction.inclusion_status == "included",
            )
            .order_by(HistoricalUniverseReconstruction.normalized_symbol)
            .all()
        )
        return [
            _hur_source_row_from_model(
                row,
                source_schema=self._source_hur_schema,
            )
            for row in rows
        ]

    def _persist_candidate(
        self,
        result: PitCandidateResult,
        job_run_id: str | None,
    ) -> I12PitCandidate:
        active_attempt = (
            self._session.query(I12PitCandidate)
            .filter(
                I12PitCandidate.candidate_attempt_hash == result.candidate_attempt_hash,
                I12PitCandidate.is_active.is_(True),
            )
            .one_or_none()
        )
        if active_attempt is not None and self._replace_existing:
            self._session.delete(active_attempt)
            self._session.flush()
            active_attempt = None
        elif active_attempt is not None and active_attempt.content_hash == result.content_hash:
            active_attempt.job_run_id = job_run_id
            source_bars_json = _json_dumps(result.source_bars)
            if active_attempt.source_bars_json != source_bars_json:
                active_attempt.source_bars_json = source_bars_json
            if active_attempt.label_hash != result.label_hash:
                active_attempt.label_json = _json_dumps(result.label_json)
                active_attempt.label_hash = result.label_hash
            self._session.flush()
            return active_attempt
        elif active_attempt is not None:
            active_attempt.is_active = False
            superseded_at = datetime.now(timezone.utc)
            active_attempt.superseded_at = superseded_at
            self._inactivate_candidate_child_evidence(active_attempt, superseded_at)
            self._session.flush()

        existing_content = (
            self._session.query(I12PitCandidate)
            .filter(I12PitCandidate.content_hash == result.content_hash)
            .one_or_none()
        )
        if existing_content is not None and self._replace_existing:
            self._session.delete(existing_content)
            self._session.flush()
        elif existing_content is not None:
            existing_content.job_run_id = job_run_id
            existing_content.is_active = True
            existing_content.superseded_at = None
            existing_content.superseded_by_candidate_id = None
            source_bars_json = _json_dumps(result.source_bars)
            if existing_content.source_bars_json != source_bars_json:
                existing_content.source_bars_json = source_bars_json
            if existing_content.label_hash != result.label_hash:
                existing_content.label_json = _json_dumps(result.label_json)
                existing_content.label_hash = result.label_hash
            if active_attempt is not None:
                active_attempt.superseded_by_candidate_id = existing_content.i12_pit_candidate_id
                self._session.flush()
            return existing_content
        row = I12PitCandidate(
            job_run_id=job_run_id,
            ticker=result.ticker,
            decision_date=result.decision_date,
            decision_ts=result.decision_ts,
            decision_time_label=result.decision_time_label,
            path_mode=result.path_mode,
            feature_asof_ts=result.feature_asof_ts,
            candidate_status=result.candidate_status,
            coverage_status=result.coverage_status,
            fail_reason=result.fail_reason,
            feature_json=_json_dumps(result.feature_json),
            gate_values_json=_json_dumps(result.gate_values),
            leakage_guard_json=_json_dumps(result.leakage_guard),
            source_bars_json=_json_dumps(result.source_bars),
            label_json=_json_dumps(result.label_json),
            error_json=_json_dumps(result.error_json) if result.error_json else None,
            candidate_attempt_hash=result.candidate_attempt_hash,
            is_active=True,
            superseded_at=None,
            superseded_by_candidate_id=None,
            input_hash=result.input_hash,
            candidate_identity_hash=result.candidate_identity_hash,
            label_hash=result.label_hash,
            content_hash=result.content_hash,
        )
        self._session.add(row)
        self._session.flush()
        if active_attempt is not None:
            active_attempt.superseded_by_candidate_id = row.i12_pit_candidate_id
            self._session.flush()
        return row

    def _inactivate_candidate_child_evidence(
        self,
        candidate: I12PitCandidate,
        superseded_at: datetime,
    ) -> None:
        active_quotes = (
            self._session.query(I12PitQuoteReplay)
            .filter(
                I12PitQuoteReplay.i12_pit_candidate_id == candidate.i12_pit_candidate_id,
                I12PitQuoteReplay.is_active.is_(True),
            )
            .all()
        )
        for quote in active_quotes:
            quote.is_active = False
            quote.superseded_at = superseded_at
            quote.superseded_by_quote_replay_id = None
        active_costs = (
            self._session.query(I12PitCostReplay)
            .filter(
                I12PitCostReplay.i12_pit_candidate_id == candidate.i12_pit_candidate_id,
                I12PitCostReplay.is_active.is_(True),
            )
            .all()
        )
        for cost in active_costs:
            cost.is_active = False
            cost.superseded_at = superseded_at
            cost.superseded_by_cost_replay_id = None

    def _replay_quotes(
        self,
        candidate_id: str,
        job_run_id: str | None,
    ) -> dict[str, I12PitQuoteReplay]:
        assert self._alpaca is not None
        candidate = self._session.get(I12PitCandidate, candidate_id)
        if candidate is None:
            raise RuntimeError(f"I12 PIT candidate not found for quote replay: {candidate_id}")
        windows_and_attempts = [
            (
                window,
                _quote_replay_attempt_hash(
                    candidate,
                    window,
                    feed=self._feed,
                    max_quote_age_seconds=self._max_quote_age_seconds,
                ),
            )
            for window in quote_windows_for_candidate(candidate)
        ]
        ticker = candidate.ticker
        if self._session.in_transaction():
            self._session.rollback()
        rows: dict[str, I12PitQuoteReplay] = {}
        for window, attempt_hash in windows_and_attempts:
            active_quote = (
                self._session.query(I12PitQuoteReplay)
                .filter(
                    I12PitQuoteReplay.quote_replay_attempt_hash == attempt_hash,
                    I12PitQuoteReplay.is_active.is_(True),
                )
                .one_or_none()
            )
            if (
                active_quote is not None
                and active_quote.coverage_status == "ok"
                and not self._replace_existing
            ):
                active_quote.job_run_id = job_run_id
                self._session.flush()
                rows[window.quote_role] = active_quote
                self._session.commit()
                continue
            if self._session.in_transaction():
                self._session.rollback()
            resp = self._fetch_alpaca_quotes_with_deadline(
                ticker,
                start=window.window_start_ts,
                end=window.window_end_ts,
                feed=self._feed,
                quote_role=window.quote_role,
                job_run_id=job_run_id,
            )
            breaker_payload = _watchdog_timeout_breaker_payload(resp)
            result = replay_quote_window(
                ticker=ticker,
                window=window,
                response=resp,
                feed=self._feed,
                max_quote_age_seconds=self._max_quote_age_seconds,
            )
            candidate = self._session.get(I12PitCandidate, candidate_id)
            if candidate is None:
                raise RuntimeError(
                    f"I12 PIT candidate disappeared before quote persist: {candidate_id}"
                )
            row = self._persist_quote(
                candidate,
                result,
                job_run_id,
                attempt_hash=attempt_hash,
            )
            rows[result.quote_role] = row
            self._session.commit()
            if breaker_payload is not None:
                raise ProviderOutageCircuitBreaker(breaker_payload)
        return rows

    def _fetch_alpaca_quotes_with_deadline(
        self,
        ticker: str,
        *,
        start: datetime,
        end: datetime,
        feed: str,
        quote_role: str,
        job_run_id: str | None,
    ) -> AdapterResponse[Any]:
        assert self._alpaca is not None
        stage = f"alpaca_quote_fetch:{quote_role}"
        self._progress_provider_fetch_start(
            stage=stage,
            provider="Alpaca",
            ticker=ticker,
            trading_date=start.date(),
            job_run_id=job_run_id,
            extra={
                "quote_role": quote_role,
                "window_start_ts": start.isoformat(),
                "window_end_ts": end.isoformat(),
                "feed": feed,
            },
        )
        self._rollback_before_provider_call()

        def _fetch() -> AdapterResponse[Any]:
            return self._alpaca.get_historical_quotes(
                ticker,
                start=start,
                end=end,
                feed=feed,
            )

        try:
            return call_with_daemon_deadline(
                _fetch,
                timeout_seconds=self._fetch_deadline_seconds,
                thread_name="i12-pit-alpaca-quote-fetch",
                state=self._fetch_watchdog,
                context={
                    "stage": stage,
                    "provider": "Alpaca",
                    "ticker": ticker.upper(),
                    "quote_role": quote_role,
                    "deadline_seconds": self._fetch_deadline_seconds,
                },
            )
        except ProviderOutageCircuitBreaker as exc:
            self._progress("provider_outage_circuit_breaker", exc.payload)
            if not _breaker_opened_by_current_watchdog_timeout(exc):
                raise
            self._reset_provider_session(self._alpaca)
            self._progress_provider_fetch_timeout(
                stage=stage,
                provider="Alpaca",
                ticker=ticker,
                trading_date=start.date(),
                job_run_id=job_run_id,
                extra={
                    "quote_role": quote_role,
                    "window_start_ts": start.isoformat(),
                    "window_end_ts": end.isoformat(),
                    "feed": feed,
                },
            )
            return _provider_watchdog_timeout_response(
                provider="Alpaca",
                endpoint="/v2/stocks/{symbol}/quotes",
                ticker=ticker,
                stage=stage,
                trading_date=start.date(),
                deadline_seconds=self._fetch_deadline_seconds,
                watchdog_snapshot=self._fetch_watchdog.snapshot(),
                extra={
                    "quote_role": quote_role,
                    "window_start_ts": start.isoformat(),
                    "window_end_ts": end.isoformat(),
                    "feed": feed,
                    "provider_outage_circuit_breaker": exc.payload,
                },
            )
        except FuturesTimeoutError:
            self._reset_provider_session(self._alpaca)
            self._progress_provider_fetch_timeout(
                stage=stage,
                provider="Alpaca",
                ticker=ticker,
                trading_date=start.date(),
                job_run_id=job_run_id,
                extra={
                    "quote_role": quote_role,
                    "window_start_ts": start.isoformat(),
                    "window_end_ts": end.isoformat(),
                    "feed": feed,
                },
            )
            return _provider_watchdog_timeout_response(
                provider="Alpaca",
                endpoint="/v2/stocks/{symbol}/quotes",
                ticker=ticker,
                stage=stage,
                trading_date=start.date(),
                deadline_seconds=self._fetch_deadline_seconds,
                watchdog_snapshot=self._fetch_watchdog.snapshot(),
                extra={
                    "quote_role": quote_role,
                    "window_start_ts": start.isoformat(),
                    "window_end_ts": end.isoformat(),
                    "feed": feed,
                },
            )

    def _persist_quote(
        self,
        candidate: I12PitCandidate,
        result: QuoteReplayResult,
        job_run_id: str | None,
        *,
        attempt_hash: str,
    ) -> I12PitQuoteReplay:
        content_hash = stable_hash({
            "candidate_id": candidate.i12_pit_candidate_id,
            "candidate_attempt_hash": candidate.candidate_attempt_hash,
            "candidate_content_hash": candidate.content_hash,
            "quote_role": result.quote_role,
            "target_ts": result.target_ts.isoformat(),
            "window_start_ts": result.window_start_ts.isoformat(),
            "window_end_ts": result.window_end_ts.isoformat(),
            "quote_ts": result.quote_ts.isoformat() if result.quote_ts else None,
            "quote_age_seconds": result.quote_age_seconds,
            "coverage_status": result.coverage_status,
            "bid": result.bid,
            "ask": result.ask,
            "bid_size": result.bid_size,
            "ask_size": result.ask_size,
            "spread_bps": result.spread_bps,
            "bid_notional": result.bid_notional,
            "ask_notional": result.ask_notional,
            "executable_notional": result.executable_notional,
            "executable_side": result.executable_side,
            "feed": result.feed,
            "quote_size_basis": ALPACA_QUOTE_SIZE_BASIS,
            "raw_json": result.raw_json,
            "error_json": result.error_json,
        })
        now = datetime.now(timezone.utc)
        active_attempt = (
            self._session.query(I12PitQuoteReplay)
            .filter(
                I12PitQuoteReplay.quote_replay_attempt_hash == attempt_hash,
                I12PitQuoteReplay.is_active.is_(True),
            )
            .one_or_none()
        )
        if active_attempt is not None and active_attempt.content_hash == content_hash:
            active_attempt.job_run_id = job_run_id
            self._session.flush()
            return active_attempt
        if active_attempt is not None:
            active_attempt.is_active = False
            active_attempt.superseded_at = now
            active_attempt.superseded_by_quote_replay_id = None
            self._session.flush()

        existing_content = (
            self._session.query(I12PitQuoteReplay)
            .filter(
                I12PitQuoteReplay.content_hash == content_hash,
                I12PitQuoteReplay.i12_pit_candidate_id == candidate.i12_pit_candidate_id,
            )
            .one_or_none()
        )
        if existing_content is not None:
            existing_content.job_run_id = job_run_id
            existing_content.quote_replay_attempt_hash = attempt_hash
            existing_content.is_active = True
            existing_content.superseded_at = None
            existing_content.superseded_by_quote_replay_id = None
            if active_attempt is not None:
                active_attempt.superseded_by_quote_replay_id = (
                    existing_content.i12_pit_quote_replay_id
                )
            self._session.flush()
            return existing_content
        row = I12PitQuoteReplay(
            i12_pit_candidate_id=candidate.i12_pit_candidate_id,
            job_run_id=job_run_id,
            ticker=candidate.ticker,
            decision_date=candidate.decision_date,
            decision_ts=candidate.decision_ts,
            quote_role=result.quote_role,
            target_ts=result.target_ts,
            window_start_ts=result.window_start_ts,
            window_end_ts=result.window_end_ts,
            quote_ts=result.quote_ts,
            quote_age_seconds=result.quote_age_seconds,
            bid=result.bid,
            ask=result.ask,
            bid_size=result.bid_size,
            ask_size=result.ask_size,
            spread_bps=result.spread_bps,
            top_of_book_notional=result.top_of_book_notional,
            bid_notional=result.bid_notional,
            ask_notional=result.ask_notional,
            executable_notional=result.executable_notional,
            executable_side=result.executable_side,
            feed=result.feed,
            source=result.source,
            quote_size_basis=ALPACA_QUOTE_SIZE_BASIS,
            coverage_status=result.coverage_status,
            raw_json=_json_dumps(result.raw_json) if result.raw_json is not None else None,
            error_json=_json_dumps(result.error_json) if result.error_json is not None else None,
            quote_replay_attempt_hash=attempt_hash,
            is_active=True,
            superseded_at=None,
            superseded_by_quote_replay_id=None,
            content_hash=content_hash,
        )
        self._session.add(row)
        self._session.flush()
        if active_attempt is not None:
            active_attempt.superseded_by_quote_replay_id = row.i12_pit_quote_replay_id
            self._session.flush()
        return row

    def _persist_costs(
        self,
        candidate: I12PitCandidate,
        quotes: Mapping[str, I12PitQuoteReplay],
        job_run_id: str | None,
    ) -> None:
        for exit_role in ("same_day_exit", "next_open_exit"):
            result = evaluate_quote_cost_replay(
                entry_quote=quotes.get("entry"),
                exit_quote=quotes.get(exit_role),
                exit_role=exit_role,
                intended_order_usd=self._intended_order_usd,
                max_spread_bps=self._max_spread_bps,
                slippage_bps=self._slippage_bps,
            )
            self._persist_cost(
                candidate,
                quotes,
                exit_role,
                result,
                job_run_id,
            )

    def _persist_cost(
        self,
        candidate: I12PitCandidate,
        quotes: Mapping[str, I12PitQuoteReplay],
        exit_role: str,
        result: CostReplayResult,
        job_run_id: str | None,
    ) -> I12PitCostReplay:
        entry_quote = quotes.get("entry")
        exit_quote = quotes.get(exit_role)
        attempt_hash = _cost_replay_attempt_hash(
            candidate,
            exit_role=exit_role,
            intended_order_usd=self._intended_order_usd,
            max_spread_bps=self._max_spread_bps,
            slippage_bps=self._slippage_bps,
            entry_quote=entry_quote,
            exit_quote=exit_quote,
        )
        content_hash = stable_hash({
            "candidate_id": candidate.i12_pit_candidate_id,
            "candidate_attempt_hash": candidate.candidate_attempt_hash,
            "candidate_content_hash": candidate.content_hash,
            "exit_role": exit_role,
            "entry_quote": entry_quote.content_hash if entry_quote else None,
            "exit_quote": exit_quote.content_hash if exit_quote else None,
            "tradeability_status": result.tradeability_status,
            "skipped_reason": result.skipped_reason,
            "entry_ask": result.entry_ask,
            "exit_bid": result.exit_bid,
            "gross_return": result.gross_return,
            "quote_cost_return": result.quote_cost_return,
            "slippage_return": result.slippage_return,
            "modeled_return": result.modeled_return,
            "policy": {
                "intended_order_usd": self._intended_order_usd,
                "max_spread_bps": self._max_spread_bps,
                "slippage_bps": self._slippage_bps,
            },
        })
        now = datetime.now(timezone.utc)
        active_same_policy = (
            self._session.query(I12PitCostReplay)
            .filter(
                I12PitCostReplay.i12_pit_candidate_id == candidate.i12_pit_candidate_id,
                I12PitCostReplay.exit_role == exit_role,
                I12PitCostReplay.intended_order_usd == self._intended_order_usd,
                I12PitCostReplay.max_spread_bps == self._max_spread_bps,
                I12PitCostReplay.slippage_bps == self._slippage_bps,
                I12PitCostReplay.is_active.is_(True),
            )
            .all()
        )
        for active in active_same_policy:
            if active.content_hash == content_hash:
                active.job_run_id = job_run_id
                active.cost_replay_attempt_hash = attempt_hash
                self._session.flush()
                return active
            active.is_active = False
            active.superseded_at = now
            active.superseded_by_cost_replay_id = None
        if active_same_policy:
            self._session.flush()

        existing_content = (
            self._session.query(I12PitCostReplay)
            .filter(
                I12PitCostReplay.content_hash == content_hash,
                I12PitCostReplay.i12_pit_candidate_id == candidate.i12_pit_candidate_id,
            )
            .one_or_none()
        )
        if existing_content is not None:
            existing_content.job_run_id = job_run_id
            existing_content.cost_replay_attempt_hash = attempt_hash
            existing_content.is_active = True
            existing_content.superseded_at = None
            existing_content.superseded_by_cost_replay_id = None
            for active in active_same_policy:
                active.superseded_by_cost_replay_id = existing_content.i12_pit_cost_replay_id
            self._session.flush()
            return existing_content
        row = I12PitCostReplay(
            i12_pit_candidate_id=candidate.i12_pit_candidate_id,
            job_run_id=job_run_id,
            ticker=candidate.ticker,
            decision_date=candidate.decision_date,
            decision_ts=candidate.decision_ts,
            exit_role=exit_role,
            entry_quote_replay_id=(
                quotes.get("entry").i12_pit_quote_replay_id
                if quotes.get("entry") else None
            ),
            exit_quote_replay_id=(
                quotes.get(exit_role).i12_pit_quote_replay_id
                if quotes.get(exit_role) else None
            ),
            tradeability_status=result.tradeability_status,
            skipped_reason=result.skipped_reason,
            intended_order_usd=result.intended_order_usd,
            max_spread_bps=result.max_spread_bps,
            slippage_bps=result.slippage_bps,
            entry_ask=result.entry_ask,
            exit_bid=result.exit_bid,
            gross_return=result.gross_return,
            quote_cost_return=result.quote_cost_return,
            slippage_return=result.slippage_return,
            modeled_return=result.modeled_return,
            cost_replay_attempt_hash=attempt_hash,
            is_active=True,
            superseded_at=None,
            superseded_by_cost_replay_id=None,
            content_hash=content_hash,
        )
        self._session.add(row)
        self._session.flush()
        for active in active_same_policy:
            active.superseded_by_cost_replay_id = row.i12_pit_cost_replay_id
        if active_same_policy:
            self._session.flush()
        return row

    def _update_partial_metrics(
        self,
        *,
        counters: Counter[str],
        hur_rows_loaded: int,
        event: str,
        trading_date: date | None = None,
        job_run_id: str | None = None,
    ) -> bool:
        progress_session: Session | None = None
        try:
            if self._session.in_transaction():
                self._session.commit()
            if self._session.in_transaction():
                raise RuntimeError("main job session still has an open transaction")
            progress_session = self._open_progress_metrics_session()
            self._update_partial_metrics_inner(
                progress_session=progress_session,
                counters=counters,
                hur_rows_loaded=hur_rows_loaded,
                event=event,
                trading_date=trading_date,
                job_run_id=job_run_id,
            )
        except Exception as exc:
            self._close_progress_metrics_session(progress_session)
            progress_session = None
            if self._session.in_transaction():
                self._session.rollback()
            self.partial_metrics = {
                "last_event": event,
                "last_trading_date": trading_date.isoformat() if trading_date else None,
                "job_run_id": job_run_id,
                "fetch_watchdog": self._fetch_watchdog.snapshot(),
                "progress_metrics_session": "isolated",
                "progress_metrics_status": "failed",
                "progress_error": {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                "counters": dict(counters),
            }
            self._progress("progress_error", self.partial_metrics)
            return False
        else:
            self._close_progress_metrics_session(progress_session)
            progress_session = None
            self._progress(event, self.partial_metrics)
            return True
        finally:
            self._close_progress_metrics_session(progress_session)
            if self._session.in_transaction():
                self._session.rollback()

    def _update_partial_metrics_inner(
        self,
        *,
        progress_session: Session,
        counters: Counter[str],
        hur_rows_loaded: int,
        event: str,
        trading_date: date | None = None,
        job_run_id: str | None = None,
    ) -> None:
        self._set_local_progress_statement_timeout(progress_session)
        progress_start_date = self._start_date
        progress_end_date = trading_date if trading_date is not None else self._end_date
        if progress_end_date is not None and self._end_date is not None:
            progress_end_date = min(progress_end_date, self._end_date)
        active_candidate_query = progress_session.query(I12PitCandidate).filter(
            I12PitCandidate.is_active.is_(True),
            I12PitCandidate.path_mode == self._minute_path_mode,
        )
        historical_candidate_query = progress_session.query(I12PitCandidate).filter(
            I12PitCandidate.path_mode == self._minute_path_mode,
        )
        if self._start_date is not None:
            active_candidate_query = active_candidate_query.filter(
                I12PitCandidate.decision_date >= self._start_date
            )
            historical_candidate_query = historical_candidate_query.filter(
                I12PitCandidate.decision_date >= self._start_date
            )
        if progress_end_date is not None:
            active_candidate_query = active_candidate_query.filter(
                I12PitCandidate.decision_date <= progress_end_date
            )
            historical_candidate_query = historical_candidate_query.filter(
                I12PitCandidate.decision_date <= progress_end_date
            )
        active_candidate_query = active_candidate_query.filter(
            I12PitCandidate.decision_time_label.in_(self._decision_times)
        )
        historical_candidate_query = historical_candidate_query.filter(
            I12PitCandidate.decision_time_label.in_(self._decision_times)
        )
        active_candidate_count = active_candidate_query.count()
        historical_candidate_count = historical_candidate_query.count()
        active_candidate_ids = active_candidate_query.with_entities(
            I12PitCandidate.i12_pit_candidate_id
        ).subquery()
        historical_candidate_ids = historical_candidate_query.with_entities(
            I12PitCandidate.i12_pit_candidate_id
        ).subquery()
        quote_replay_row_count = _count_child_rows_for_candidate_subquery(
            progress_session,
            I12PitQuoteReplay,
            active_candidate_ids,
            active_only=True,
        )
        historical_quote_replay_row_count = _count_child_rows_for_candidate_subquery(
            progress_session,
            I12PitQuoteReplay,
            historical_candidate_ids,
        )
        cost_replay_row_count = _count_child_rows_for_candidate_subquery(
            progress_session,
            I12PitCostReplay,
            active_candidate_ids,
            active_only=True,
        )
        historical_cost_replay_row_count = _count_child_rows_for_candidate_subquery(
            progress_session,
            I12PitCostReplay,
            historical_candidate_ids,
        )
        expected_candidate_attempts = hur_rows_loaded * len(self._decision_times)
        missing_source_attempt_count = max(expected_candidate_attempts - active_candidate_count, 0)
        extra_source_attempt_count = max(active_candidate_count - expected_candidate_attempts, 0)
        source_attempt_count_exact = (
            missing_source_attempt_count == 0 and extra_source_attempt_count == 0
        )
        self.partial_metrics = {
            "source_hur_schema": self._source_hur_schema,
            "hur_rows_loaded": hur_rows_loaded,
            "progress_metrics_session": "isolated",
            "progress_metrics_status": "ok",
            "progress_source_start_date": (
                progress_start_date.isoformat() if progress_start_date else None
            ),
            "progress_source_end_date": (
                progress_end_date.isoformat() if progress_end_date else None
            ),
            "progress_source_scope": (
                "processed_through_trading_date"
                if trading_date is not None
                else "configured_full_window"
            ),
            "decision_time_count": len(self._decision_times),
            "minute_path_mode": self._minute_path_mode,
            "expected_candidate_attempts": expected_candidate_attempts,
            "candidate_row_count": active_candidate_count,
            "active_candidate_row_count_for_path_mode": active_candidate_count,
            "missing_source_attempt_count_for_path_mode": missing_source_attempt_count,
            "extra_source_attempt_count_for_path_mode": extra_source_attempt_count,
            "source_attempt_count_exact_for_path_mode": source_attempt_count_exact,
            "source_identity_denominator_known_for_path_mode": False,
            "source_identity_denominator_error_for_path_mode": (
                "progress_identity_audit_skipped_final_report_only"
            ),
            "missing_source_attempt_identity_count_for_path_mode": None,
            "extra_source_attempt_identity_count_for_path_mode": None,
            "source_attempt_identity_exact_for_path_mode": False,
            "historical_candidate_row_count": historical_candidate_count,
            "historical_candidate_row_count_for_path_mode": historical_candidate_count,
            "quote_replay_row_count": quote_replay_row_count,
            "quote_replay_row_count_for_path_mode": quote_replay_row_count,
            "historical_quote_replay_row_count": historical_quote_replay_row_count,
            "historical_quote_replay_row_count_for_path_mode": (
                historical_quote_replay_row_count
            ),
            "cost_replay_row_count": cost_replay_row_count,
            "cost_replay_row_count_for_path_mode": cost_replay_row_count,
            "historical_cost_replay_row_count": historical_cost_replay_row_count,
            "historical_cost_replay_row_count_for_path_mode": (
                historical_cost_replay_row_count
            ),
            "schema_total_active_candidate_row_count": progress_session.query(I12PitCandidate)
            .filter(I12PitCandidate.is_active.is_(True))
            .count(),
            "schema_total_historical_candidate_row_count": progress_session.query(
                I12PitCandidate
            ).count(),
            "schema_total_active_quote_replay_row_count": progress_session.query(
                I12PitQuoteReplay
            )
            .filter(I12PitQuoteReplay.is_active.is_(True))
            .count(),
            "schema_total_historical_quote_replay_row_count": progress_session.query(
                I12PitQuoteReplay
            ).count(),
            "schema_total_active_cost_replay_row_count": progress_session.query(
                I12PitCostReplay
            )
            .filter(I12PitCostReplay.is_active.is_(True))
            .count(),
            "schema_total_historical_cost_replay_row_count": progress_session.query(
                I12PitCostReplay
            ).count(),
            "daily_fetch_error_count": counters.get("coverage_daily_fetch_error", 0),
            "minute_fetch_error_count": counters.get("coverage_minute_fetch_error", 0),
            "counters": dict(counters),
            "last_event": event,
            "last_trading_date": trading_date.isoformat() if trading_date else None,
            "job_run_id": job_run_id,
        }

    def _maybe_emit_ticker_progress(
        self,
        *,
        counters: Counter[str],
        trading_date: date,
        hur_rows_for_date: int,
        hur_rows_processed_for_date: int,
        cumulative_hur_rows_loaded: int,
        current_ticker: str,
        job_run_id: str | None,
    ) -> None:
        if (
            hur_rows_processed_for_date % self._progress_interval_tickers != 0
            and hur_rows_processed_for_date != hur_rows_for_date
        ):
            return
        self._progress(
            "ticker_progress",
            self._heartbeat_payload(
                counters=counters,
                trading_date=trading_date,
                hur_rows_for_date=hur_rows_for_date,
                hur_rows_processed_for_date=hur_rows_processed_for_date,
                cumulative_hur_rows_loaded=cumulative_hur_rows_loaded,
                current_ticker=current_ticker,
                job_run_id=job_run_id,
            ),
        )

    def _heartbeat_payload(
        self,
        *,
        counters: Counter[str],
        trading_date: date,
        hur_rows_for_date: int,
        hur_rows_processed_for_date: int,
        cumulative_hur_rows_loaded: int,
        job_run_id: str | None,
        current_ticker: str | None = None,
    ) -> dict[str, Any]:
        return {
            "job_run_id": job_run_id,
            "trading_date": trading_date.isoformat(),
            "minute_path_mode": self._minute_path_mode,
            "decision_time_labels": list(self._decision_times),
            "hur_rows_for_date": hur_rows_for_date,
            "hur_rows_processed_for_date": hur_rows_processed_for_date,
            "cumulative_hur_rows_loaded": cumulative_hur_rows_loaded,
            "current_ticker": current_ticker,
            "candidate_passed": counters.get("candidate_passed", 0),
            "candidate_failed": counters.get("candidate_failed", 0),
            "daily_fetch_error_count": counters.get("coverage_daily_fetch_error", 0),
            "minute_fetch_error_count": counters.get("coverage_minute_fetch_error", 0),
            "counters": dict(counters),
            "fetch_watchdog": self._fetch_watchdog.snapshot(),
        }

    def _rollback_before_provider_call(self) -> None:
        if self._session.in_transaction():
            self._session.rollback()

    def _open_progress_metrics_session(self) -> Session:
        if self._progress_session_factory is not None:
            return self._progress_session_factory()
        return Session(bind=self._session.get_bind())

    @staticmethod
    def _close_progress_metrics_session(progress_session: Session | None) -> None:
        if progress_session is None:
            return
        try:
            if progress_session.in_transaction():
                progress_session.rollback()
        finally:
            progress_session.close()

    @staticmethod
    def _set_local_progress_statement_timeout(progress_session: Session) -> None:
        bind = progress_session.get_bind()
        dialect_name = getattr(getattr(bind, "dialect", None), "name", "")
        if dialect_name != "postgresql":
            return
        progress_session.execute(
            text(f"SET LOCAL statement_timeout = {PROGRESS_METRICS_STATEMENT_TIMEOUT_MS}")
        )

    def _progress_provider_fetch_start(
        self,
        *,
        stage: str,
        provider: str,
        ticker: str,
        trading_date: date,
        job_run_id: str | None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self._progress(
            "provider_fetch_start",
            {
                "job_run_id": job_run_id,
                "stage": stage,
                "provider": provider,
                "ticker": ticker.upper(),
                "trading_date": trading_date.isoformat(),
                "deadline_seconds": self._fetch_deadline_seconds,
                "minute_path_mode": self._minute_path_mode,
                "decision_time_labels": list(self._decision_times),
                "fetch_watchdog": self._fetch_watchdog.snapshot(),
                **dict(extra or {}),
            },
        )

    def _progress_provider_fetch_timeout(
        self,
        *,
        stage: str,
        provider: str,
        ticker: str,
        trading_date: date,
        job_run_id: str | None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self._progress(
            "provider_fetch_timeout",
            {
                "job_run_id": job_run_id,
                "stage": stage,
                "provider": provider,
                "ticker": ticker.upper(),
                "trading_date": trading_date.isoformat(),
                "deadline_seconds": self._fetch_deadline_seconds,
                "minute_path_mode": self._minute_path_mode,
                "decision_time_labels": list(self._decision_times),
                "fetch_watchdog": self._fetch_watchdog.snapshot(),
                **dict(extra or {}),
            },
        )

    @staticmethod
    def _reset_provider_session(adapter: Any) -> None:
        reset_session = getattr(adapter, "reset_session", None)
        if callable(reset_session):
            reset_session()

    def _reset_provider_sessions_at_date_boundary(self) -> None:
        for adapter in (self._fmp, self._polygon, self._alpaca):
            self._reset_provider_session(adapter)

    def _start_no_progress_monitor(self, job_run_id: str | None) -> None:
        if self._max_no_progress_seconds is None:
            return
        if self._no_progress_thread is not None:
            return
        self._no_progress_stop.clear()
        self._no_progress_thread = threading.Thread(
            target=self._no_progress_monitor_loop,
            args=(job_run_id,),
            name="i12-pit-no-progress-monitor",
            daemon=True,
        )
        self._no_progress_thread.start()

    def _stop_no_progress_monitor(self) -> None:
        thread = self._no_progress_thread
        if thread is None:
            return
        self._no_progress_stop.set()
        thread.join(timeout=1.0)
        self._no_progress_thread = None

    def _no_progress_monitor_loop(self, job_run_id: str | None) -> None:
        assert self._max_no_progress_seconds is not None
        sleep_seconds = min(5.0, max(0.1, self._max_no_progress_seconds / 4.0))
        while not self._no_progress_stop.wait(sleep_seconds):
            payload = self._no_progress_timeout_payload(job_run_id)
            if payload is None:
                continue
            try:
                self._progress("no_progress_timeout", payload)
            except Exception as exc:  # noqa: BLE001 - monitor must still terminate
                print(
                    "ERROR: failed to write no-progress artifact: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            callback = self._no_progress_exit_callback
            if callback is not None:
                callback(payload)
            return

    def _no_progress_timeout_payload(
        self,
        job_run_id: str | None,
    ) -> dict[str, Any] | None:
        if self._max_no_progress_seconds is None:
            return None
        now_monotonic = time_module.monotonic()
        with self._progress_state_lock:
            age_seconds = now_monotonic - self._last_progress_monotonic
            if age_seconds < self._max_no_progress_seconds:
                return None
            return {
                "job_run_id": job_run_id,
                "minute_path_mode": self._minute_path_mode,
                "decision_time_labels": list(self._decision_times),
                "max_no_progress_seconds": self._max_no_progress_seconds,
                "no_progress_age_seconds": age_seconds,
                "last_progress_event": self._last_progress_event,
                "last_progress_wall_clock_utc": (
                    self._last_progress_wall_clock_utc.isoformat()
                ),
                "last_progress_payload": dict(self._last_progress_payload),
                "fetch_watchdog": self._fetch_watchdog.snapshot(),
            }

    def _candidate_attempt_source_hashes(
        self,
        *,
        ticker: str,
        trading_date: date,
        decision_ts: datetime,
        decision_time_label: str,
        source_hur_identity_hash: str | None,
        path_mode: str,
        daily_source_hash: str | None,
        minute_source_hash: str | None,
        daily_source_hash_basis: str,
        daily_provider_fetch_hash: str | None,
    ) -> tuple[str | None, str | None, str, str | None, str]:
        # Daily caching can change the provider raw-payload hash basis from the
        # old per-date request to a wider per-ticker request. Preserve hashes
        # from an active natural attempt so unchanged reruns do not supersede
        # candidates or detach quote/cost evidence.
        attempt_hash = _candidate_attempt_hash(
            ticker=ticker,
            trading_date=trading_date,
            decision_ts=decision_ts,
            decision_time_label=decision_time_label,
            source_hur_identity_hash=source_hur_identity_hash,
            path_mode=path_mode,
        )
        existing = (
            self._session.query(I12PitCandidate)
            .filter(
                I12PitCandidate.candidate_attempt_hash == attempt_hash,
                I12PitCandidate.is_active.is_(True),
            )
            .one_or_none()
        )
        if existing is None:
            return (
                daily_source_hash,
                minute_source_hash,
                daily_source_hash_basis,
                daily_provider_fetch_hash,
                DAILY_SOURCE_HASH_REUSE_FRESH,
            )
        source_bars = _json_loads(existing.source_bars_json)
        existing_basis = source_bars.get("daily_source_hash_basis")
        if not isinstance(existing_basis, str) or not existing_basis:
            existing_basis = "unknown"
        return (
            source_bars.get("daily_source_hash") or daily_source_hash,
            source_bars.get("minute_source_hash") or minute_source_hash,
            existing_basis,
            source_bars.get("daily_provider_fetch_hash") or daily_provider_fetch_hash,
            DAILY_SOURCE_HASH_REUSE_EXISTING_ACTIVE,
        )

    def _mark_progress(self, event: str, payload: Mapping[str, Any]) -> None:
        with self._progress_state_lock:
            self._last_progress_monotonic = time_module.monotonic()
            self._last_progress_wall_clock_utc = datetime.now(timezone.utc)
            self._last_progress_event = event
            self._last_progress_payload = dict(payload)

    def _progress(self, event: str, payload: Mapping[str, Any]) -> None:
        self._mark_progress(event, payload)
        if self._progress_artifact is None:
            return
        record = {
            "event": event,
            "wall_clock_utc": datetime.now(timezone.utc).isoformat(),
            **dict(payload),
        }
        self._progress_artifact.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self._progress_artifact,
            json.dumps(record, indent=2, sort_keys=True, default=str),
        )


def _atomic_write_text(path: Path, text_value: str) -> None:
    temp_path: Path | None = None
    fd: int | None = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            text=True,
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            handle.write(text_value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
        _fsync_parent_directory(path.parent)
    finally:
        if fd is not None:
            os.close(fd)
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _fsync_parent_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        dir_fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _load_child_rows_for_candidate_ids(
    session: Session,
    model: Any,
    candidate_ids: Sequence[str],
) -> list[Any]:
    rows: list[Any] = []
    for chunk in _iter_chunks(candidate_ids, CHILD_CANDIDATE_ID_QUERY_CHUNK_SIZE):
        rows.extend(
            session.query(model)
            .filter(model.i12_pit_candidate_id.in_(chunk))
            .all()
        )
    return rows


def _count_child_rows_for_candidate_ids(
    session: Session,
    model: Any,
    candidate_ids: Sequence[str],
    *,
    active_only: bool = False,
) -> int:
    total = 0
    for chunk in _iter_chunks(candidate_ids, CHILD_CANDIDATE_ID_QUERY_CHUNK_SIZE):
        query = session.query(model).filter(model.i12_pit_candidate_id.in_(chunk))
        if active_only:
            query = query.filter(model.is_active.is_(True))
        total += query.count()
    return total


def _count_child_rows_for_candidate_subquery(
    session: Session,
    model: Any,
    candidate_ids_subquery: Any,
    *,
    active_only: bool = False,
) -> int:
    query = session.query(model).join(
        candidate_ids_subquery,
        model.i12_pit_candidate_id == candidate_ids_subquery.c.i12_pit_candidate_id,
    )
    if active_only:
        query = query.filter(model.is_active.is_(True))
    return query.count()


def _iter_chunks(values: Sequence[str], size: int) -> list[list[str]]:
    return [
        list(values[index : index + size])
        for index in range(0, len(values), size)
    ]


def _daily_bars_for_trading_date(
    ticker: str,
    trading_date: date,
    raw_bars: Sequence[FmpBar],
) -> list[_DailyBar]:
    from_date = trading_date - timedelta(days=460)
    to_date = next_us_equity_session(trading_date + timedelta(days=2))
    return [
        bar
        for bar in _clean_daily_bars(ticker, raw_bars, _CleanCounters())
        if from_date <= bar.date <= to_date
    ]


def _daily_source_window(trading_date: date) -> tuple[date, date]:
    return (
        trading_date - timedelta(days=460),
        next_us_equity_session(trading_date + timedelta(days=2)),
    )


def _legacy_daily_source_hash_key(ticker: str, trading_date: date) -> str:
    from_date, to_date = _daily_source_window(trading_date)
    return f"{ticker.upper()}|{from_date.isoformat()}|{to_date.isoformat()}|adjusted=false"


def _daily_source_hash_for_trading_date(
    ticker: str,
    trading_date: date,
    daily_resp: AdapterResponse[Any],
    daily_bars: Sequence[_DailyBar],
) -> tuple[str | None, str, str | None]:
    # Exact legacy raw-response parity is only available when the adapter
    # lineage explicitly carries the old per-date hash. Otherwise use a
    # deterministic clean-slice provenance hash instead of the wide fetch hash.
    flags = getattr(daily_resp.lineage, "data_quality_flags", None) or {}
    provider_fetch_hash = getattr(daily_resp.lineage, "raw_payload_hash", None)
    if isinstance(flags, Mapping):
        legacy_hashes = flags.get(LEGACY_DAILY_SOURCE_HASHES_FLAG)
        if isinstance(legacy_hashes, Mapping):
            for key in (
                _legacy_daily_source_hash_key(ticker, trading_date),
                trading_date.isoformat(),
            ):
                value = legacy_hashes.get(key)
                if isinstance(value, str) and value:
                    return (
                        value,
                        DAILY_SOURCE_HASH_BASIS_LEGACY,
                        provider_fetch_hash,
                    )
    return (
        _daily_slice_source_hash(ticker, trading_date, daily_bars),
        DAILY_SOURCE_HASH_BASIS_CLEAN_SLICE,
        provider_fetch_hash,
    )


def _daily_slice_source_hash(
    ticker: str,
    trading_date: date,
    daily_bars: Sequence[_DailyBar],
) -> str:
    from_date, to_date = _daily_source_window(trading_date)
    return stable_hash({
        "source": "fmp_daily_clean_slice_v1",
        "ticker": ticker.upper(),
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "bars": [
            {
                "date": bar.date.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "split_adjusted_close": bar.split_adjusted_close,
            }
            for bar in daily_bars
        ],
    })


def _source_bars_for_candidate_identity(
    source_bars: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in source_bars.items()
        if key not in SOURCE_BARS_NON_IDENTITY_KEYS
    }


def _daily_prefilter_results(
    *,
    ticker: str,
    trading_date: date,
    decision_time_labels: Sequence[str],
    daily_bars: Sequence[_DailyBar],
    daily_source_hash: str | None,
    daily_source_hash_basis: str,
    daily_provider_fetch_hash: str | None,
    source_hur_identity_hash: str | None,
    source_hur: Mapping[str, Any] | None,
    path_mode: str,
) -> list[PitCandidateResult] | None:
    try:
        prior_ctx = _prior_context(ticker.upper(), trading_date, daily_bars)
    except RuntimeError as exc:
        return [
            _candidate_error(
                ticker=ticker,
                trading_date=trading_date,
                decision_ts=_decision_timestamp(trading_date, label),
                decision_time_label=label,
                coverage_status="daily_context_error",
                fail_reason=str(exc),
                daily_source_hash=daily_source_hash,
                daily_source_hash_basis=daily_source_hash_basis,
                daily_provider_fetch_hash=daily_provider_fetch_hash,
                minute_source_hash=None,
                prior_ctx=None,
                source_hur_identity_hash=source_hur_identity_hash,
                source_hur=source_hur,
                path_mode=path_mode,
            )
            for label in decision_time_labels
        ]
    if prior_ctx["distance_from_max252"] <= -0.50:
        return None
    return [
        _candidate_error(
            ticker=ticker,
            trading_date=trading_date,
            decision_ts=_decision_timestamp(trading_date, label),
            decision_time_label=label,
            coverage_status="daily_prefilter_skip",
            fail_reason="drawdown",
            daily_source_hash=daily_source_hash,
            daily_source_hash_basis=daily_source_hash_basis,
            daily_provider_fetch_hash=daily_provider_fetch_hash,
            minute_source_hash=None,
            prior_ctx=prior_ctx,
            source_hur_identity_hash=source_hur_identity_hash,
            source_hur=source_hur,
            path_mode=path_mode,
        )
        for label in decision_time_labels
    ]


def build_i12_pit_candidate(
    *,
    ticker: str,
    trading_date: date,
    decision_ts: datetime,
    decision_time_label: str,
    daily_bars: Sequence[_DailyBar],
    minute_bars: Sequence[_MinuteBar],
    daily_source_hash: str | None,
    daily_source_hash_basis: str = "unknown",
    daily_provider_fetch_hash: str | None = None,
    daily_source_hash_reuse_status: str = DAILY_SOURCE_HASH_REUSE_FRESH,
    minute_source_hash: str | None = None,
    minute_error: dict[str, Any] | None = None,
    source_hur_identity_hash: str | None = None,
    source_hur: Mapping[str, Any] | None = None,
    path_mode: str = STRICT_MINUTE_PATH_MODE,
) -> PitCandidateResult:
    ticker = ticker.upper()
    path_mode = _validate_minute_path_mode(path_mode)
    try:
        prior_ctx = _prior_context(ticker, trading_date, daily_bars)
    except RuntimeError as exc:
        return _candidate_error(
            ticker=ticker,
            trading_date=trading_date,
            decision_ts=decision_ts,
            decision_time_label=decision_time_label,
            coverage_status="daily_context_error",
            fail_reason=str(exc),
            daily_source_hash=daily_source_hash,
            daily_source_hash_basis=daily_source_hash_basis,
            daily_provider_fetch_hash=daily_provider_fetch_hash,
            daily_source_hash_reuse_status=daily_source_hash_reuse_status,
            minute_source_hash=minute_source_hash,
            minute_error=minute_error,
            source_hur_identity_hash=source_hur_identity_hash,
            source_hur=source_hur,
            path_mode=path_mode,
        )
    eligible_minutes = sorted(
        (
            bar for bar in minute_bars
            if bar.timestamp < decision_ts
            and bar.timestamp.astimezone(EASTERN).date() == trading_date
        ),
        key=lambda bar: bar.timestamp,
    )
    if not eligible_minutes:
        path_diagnostics = _minute_path_diagnostics(
            trading_date=trading_date,
            decision_ts=decision_ts,
            minute_bars=eligible_minutes,
            prior_ctx=prior_ctx,
            path_mode=path_mode,
        )
        return _candidate_error(
            ticker=ticker,
            trading_date=trading_date,
            decision_ts=decision_ts,
            decision_time_label=decision_time_label,
            coverage_status="missing_minute_bars",
            fail_reason="missing_minute_bars_before_decision",
            daily_source_hash=daily_source_hash,
            daily_source_hash_basis=daily_source_hash_basis,
            daily_provider_fetch_hash=daily_provider_fetch_hash,
            daily_source_hash_reuse_status=daily_source_hash_reuse_status,
            minute_source_hash=minute_source_hash,
            minute_error=minute_error,
            prior_ctx=prior_ctx,
            source_hur_identity_hash=source_hur_identity_hash,
            source_hur=source_hur,
            path_mode=path_mode,
            path_diagnostics=path_diagnostics,
        )
    duplicate_minute_starts = _duplicate_minute_start_timestamps(eligible_minutes)
    if duplicate_minute_starts:
        path_diagnostics = _minute_path_diagnostics(
            trading_date=trading_date,
            decision_ts=decision_ts,
            minute_bars=eligible_minutes,
            prior_ctx=prior_ctx,
            path_mode=path_mode,
        )
        path_diagnostics["duplicate_minute_timestamps"] = [
            ts.isoformat() for ts in duplicate_minute_starts
        ]
        return _candidate_error(
            ticker=ticker,
            trading_date=trading_date,
            decision_ts=decision_ts,
            decision_time_label=decision_time_label,
            coverage_status="duplicate_minute_bars",
            fail_reason="duplicate_minute_bars_before_decision",
            daily_source_hash=daily_source_hash,
            daily_source_hash_basis=daily_source_hash_basis,
            daily_provider_fetch_hash=daily_provider_fetch_hash,
            daily_source_hash_reuse_status=daily_source_hash_reuse_status,
            minute_source_hash=minute_source_hash,
            minute_error=minute_error,
            prior_ctx=prior_ctx,
            source_hur_identity_hash=source_hur_identity_hash,
            source_hur=source_hur,
            path_mode=path_mode,
            path_diagnostics=path_diagnostics,
        )
    path_diagnostics = _minute_path_diagnostics(
        trading_date=trading_date,
        decision_ts=decision_ts,
        minute_bars=eligible_minutes,
        prior_ctx=prior_ctx,
        path_mode=path_mode,
    )
    minute_path_error = _minute_path_coverage_error(
        trading_date=trading_date,
        decision_ts=decision_ts,
        minute_bars=eligible_minutes,
    )
    if minute_path_error is not None and path_mode == STRICT_MINUTE_PATH_MODE:
        coverage_status, fail_reason = minute_path_error
        return _candidate_error(
            ticker=ticker,
            trading_date=trading_date,
            decision_ts=decision_ts,
            decision_time_label=decision_time_label,
            coverage_status=coverage_status,
            fail_reason=fail_reason,
            daily_source_hash=daily_source_hash,
            daily_source_hash_basis=daily_source_hash_basis,
            daily_provider_fetch_hash=daily_provider_fetch_hash,
            daily_source_hash_reuse_status=daily_source_hash_reuse_status,
            minute_source_hash=minute_source_hash,
            minute_error=minute_error,
            prior_ctx=prior_ctx,
            source_hur_identity_hash=source_hur_identity_hash,
            source_hur=source_hur,
            path_mode=path_mode,
            path_diagnostics=path_diagnostics,
        )
    if path_mode == SPARSE_ZERO_FILL_MINUTE_PATH_MODE:
        if minute_path_error is not None and minute_path_error[0] != "partial_minute_path":
            coverage_status, fail_reason = minute_path_error
            return _candidate_error(
                ticker=ticker,
                trading_date=trading_date,
                decision_ts=decision_ts,
                decision_time_label=decision_time_label,
                coverage_status=coverage_status,
                fail_reason=fail_reason,
                daily_source_hash=daily_source_hash,
                daily_source_hash_basis=daily_source_hash_basis,
                daily_provider_fetch_hash=daily_provider_fetch_hash,
                daily_source_hash_reuse_status=daily_source_hash_reuse_status,
                minute_source_hash=minute_source_hash,
                minute_error=minute_error,
                prior_ctx=prior_ctx,
                source_hur_identity_hash=source_hur_identity_hash,
                source_hur=source_hur,
                path_mode=path_mode,
                path_diagnostics=path_diagnostics,
            )
        eligible_minutes, imputation_diagnostics = _sparse_zero_fill_minutes(
            trading_date=trading_date,
            decision_ts=decision_ts,
            minute_bars=eligible_minutes,
        )
        path_diagnostics.update(imputation_diagnostics)

    last_minute = eligible_minutes[-1]
    computed_source_minute_bars_max_start_ts = _minute_floor_utc(last_minute.timestamp)
    source_minute_bars_max_start_ts = (
        _parse_optional_datetime(path_diagnostics.get("source_minute_bars_max_start_ts"))
        or computed_source_minute_bars_max_start_ts
    )
    completed_through_ts = _minute_floor_utc(_aware_utc(decision_ts))
    day_open = eligible_minutes[0].open
    session_minutes = _session_minutes(trading_date)
    completed_minutes = _completed_minutes_before_decision(trading_date, decision_ts)
    cumulative_volume = sum(bar.volume for bar in eligible_minutes)
    projected_volume = (
        cumulative_volume / completed_minutes * session_minutes
        if completed_minutes > 0 else None
    )
    projected_volume_ratio = (
        projected_volume / prior_ctx["avg20_volume"]
        if projected_volume is not None and prior_ctx["avg20_volume"] > 0 else None
    )
    early_high = max(bar.high for bar in eligible_minutes)
    early_low = min(bar.low for bar in eligible_minutes)
    gap = day_open / prior_ctx["prior_close"] - 1.0
    early_return = last_minute.close / day_open - 1.0
    candidate_passed = (
        prior_ctx["distance_from_max252"] <= -0.50
        and -0.05 <= gap < 0.05
        and projected_volume_ratio is not None
        and projected_volume_ratio >= 5.0
        and 5.0 <= completed_minutes <= 60.0
    )
    fail_reasons: list[str] = []
    if prior_ctx["distance_from_max252"] > -0.50:
        fail_reasons.append("drawdown")
    if not (-0.05 <= gap < 0.05):
        fail_reasons.append("gap")
    if projected_volume_ratio is None or projected_volume_ratio < 5.0:
        fail_reasons.append("projected_volume_ratio")
    if not (5.0 <= completed_minutes <= 60.0):
        fail_reasons.append("decision_elapsed")

    feature_json = {
        "feature_manifest_version": FEATURE_MANIFEST_VERSION,
        "reconstruction_method": REBUILD_METHOD,
        "path_mode": path_mode,
        "pattern_id": I12_PATTERN_ID,
        "ticker": ticker,
        "decision_date": trading_date.isoformat(),
        "decision_ts": decision_ts.isoformat(),
        "feature_asof_ts": completed_through_ts.isoformat(),
        "completed_through_ts": completed_through_ts.isoformat(),
        "source_minute_bars_max_start_ts": source_minute_bars_max_start_ts.isoformat(),
        "lookback_end": prior_ctx["lookback_end"],
        "prior_close": prior_ctx["prior_close"],
        "distance_from_max252": prior_ctx["distance_from_max252"],
        "drawdown_from_max252": prior_ctx["distance_from_max252"],
        "off_low252": prior_ctx["off_low252"],
        "mom20": prior_ctx["mom20"],
        "sigma20": prior_ctx["sigma20"],
        "prev_day_return": prior_ctx["prev_day_return"],
        "prev_day_green": prior_ctx["prev_day_green"],
        "gap": gap,
        "early_cumulative_volume": cumulative_volume,
        "projected_volume_at_decision": projected_volume,
        "projected_volume_ratio_at_decision": projected_volume_ratio,
        "early_return": early_return,
        "early_high_return": early_high / day_open - 1.0,
        "early_low_return": early_low / day_open - 1.0,
        "decision_elapsed_minutes": completed_minutes,
        "completed_minute_count": completed_minutes,
        "session_minutes": session_minutes,
        "expected_minute_count_before_decision": (
            path_diagnostics["expected_minute_count_before_decision"]
        ),
        "observed_minute_count_before_decision": (
            path_diagnostics["observed_minute_count_before_decision"]
        ),
        "missing_minute_count_before_decision": (
            path_diagnostics["missing_minute_count_before_decision"]
        ),
        "path_coverage_ratio": path_diagnostics["path_coverage_ratio"],
        "opening_bar_present": path_diagnostics["opening_bar_present"],
        "last_observed_before_decision_age_seconds": (
            path_diagnostics["last_observed_before_decision_age_seconds"]
        ),
        "observed_cumulative_volume_before_decision": (
            path_diagnostics["observed_cumulative_volume_before_decision"]
        ),
        "observed_open_to_decision_return": (
            path_diagnostics["observed_open_to_decision_return"]
        ),
        "zero_fill_projected_volume_ratio": (
            path_diagnostics["zero_fill_projected_volume_ratio"]
        ),
        "zero_fill_imputed_minute_count": path_diagnostics.get(
            "zero_fill_imputed_minute_count",
            0,
        ),
        "zero_fill_imputed_minute_ratio": path_diagnostics.get(
            "zero_fill_imputed_minute_ratio",
            0.0,
        ),
    }
    leakage_guard = {
        "decision_ts": decision_ts.isoformat(),
        "entry_quote_target_ts": decision_ts.isoformat(),
        "feature_asof_ts": completed_through_ts.isoformat(),
        "completed_through_ts": completed_through_ts.isoformat(),
        "source_minute_bars_max_start_ts": source_minute_bars_max_start_ts.isoformat(),
        "uses_full_day_volume": False,
        "uses_same_day_close": False,
        "uses_full_day_high_low": False,
        "uses_forward_bars": False,
        "decision_time_semantics": (
            "decision_after_prior_completed_minute_start_stamped_bars"
        ),
        "path_mode": path_mode,
        "predictor_time_basis": "prior_daily_bars_plus_minutes_lt_decision_ts",
        "minute_imputation_policy": (
            "zero_volume_forward_fill_prices_from_prior_observed_minute"
            if path_mode == SPARSE_ZERO_FILL_MINUTE_PATH_MODE
            else "none_strict_contiguous_required"
        ),
    }
    assert_i12_pit_feature_payload_leakage_clean(feature_json, leakage_guard)
    gate_values = {
        "distance_from_max252": prior_ctx["distance_from_max252"],
        "gap": gap,
        "projected_volume_ratio_at_decision": projected_volume_ratio,
        "decision_elapsed_minutes": completed_minutes,
        "completed_minute_count": completed_minutes,
        "session_minutes": session_minutes,
        "path_mode": path_mode,
        "candidate_passed": candidate_passed,
        "fail_reasons": fail_reasons,
    }
    labels = _label_payload(
        trading_date=trading_date,
        daily_bars=daily_bars,
        minute_bars=minute_bars,
        entry_price=last_minute.close,
    )
    source_bars = {
        "daily_source_hash": daily_source_hash,
        "daily_source_hash_basis": daily_source_hash_basis,
        "daily_provider_fetch_hash": daily_provider_fetch_hash,
        "daily_source_hash_reuse_status": daily_source_hash_reuse_status,
        "minute_source_hash": minute_source_hash,
        "source_hur_identity_hash": source_hur_identity_hash,
        "source_hur": dict(source_hur or {}),
        "prior_daily_sessions": prior_ctx["prior_count"],
        "minute_bar_count_before_decision": len(eligible_minutes),
        "path_mode": path_mode,
        "path_diagnostics": path_diagnostics,
        "completed_minute_count": completed_minutes,
        "expected_minute_bar_count_before_decision": _expected_minute_count_before_decision(
            trading_date,
            decision_ts,
        ),
        "minute_bar_first_ts": eligible_minutes[0].timestamp.isoformat(),
        "minute_bar_last_ts": source_minute_bars_max_start_ts.isoformat(),
        "source_minute_bars_max_start_ts": source_minute_bars_max_start_ts.isoformat(),
        "completed_through_ts": completed_through_ts.isoformat(),
    }
    input_hash = stable_hash({
        "ticker": ticker,
        "decision_date": trading_date.isoformat(),
        "decision_ts": decision_ts.isoformat(),
        "daily_source_hash": daily_source_hash,
        "minute_source_hash": minute_source_hash,
        "source_bars": _source_bars_for_candidate_identity(source_bars),
        "path_mode": path_mode,
        "reconstruction_method": REBUILD_METHOD,
    })
    candidate_attempt_hash = _candidate_attempt_hash(
        ticker=ticker,
        trading_date=trading_date,
        decision_ts=decision_ts,
        decision_time_label=decision_time_label,
        source_hur_identity_hash=source_hur_identity_hash,
        path_mode=path_mode,
    )
    candidate_identity_hash = stable_hash({
        "input_hash": input_hash,
        "feature_json": feature_json,
        "gate_values": gate_values,
        "leakage_guard": leakage_guard,
    })
    label_hash = stable_hash(labels)
    return PitCandidateResult(
        ticker=ticker,
        decision_date=trading_date,
        decision_ts=decision_ts,
        decision_time_label=decision_time_label,
        path_mode=path_mode,
        candidate_status="passed" if candidate_passed else "failed",
        coverage_status="ok",
        fail_reason=",".join(fail_reasons) if fail_reasons else None,
        feature_json=feature_json,
        gate_values=gate_values,
        leakage_guard=leakage_guard,
        source_bars=source_bars,
        label_json=labels,
        feature_asof_ts=completed_through_ts,
        candidate_attempt_hash=candidate_attempt_hash,
        input_hash=input_hash,
        candidate_identity_hash=candidate_identity_hash,
        label_hash=label_hash,
        content_hash=candidate_identity_hash,
    )


def assert_i12_pit_feature_payload_leakage_clean(
    feature_json: Mapping[str, Any],
    leakage_guard: Mapping[str, Any],
) -> None:
    if leakage_guard.get("uses_full_day_volume") is not False:
        raise RuntimeError("I12 PIT feature leakage guard requires uses_full_day_volume=false")
    if leakage_guard.get("uses_same_day_close") is not False:
        raise RuntimeError("I12 PIT feature leakage guard requires uses_same_day_close=false")
    if leakage_guard.get("uses_full_day_high_low") is not False:
        raise RuntimeError("I12 PIT feature leakage guard requires uses_full_day_high_low=false")
    if leakage_guard.get("uses_forward_bars") is not False:
        raise RuntimeError("I12 PIT feature leakage guard requires uses_forward_bars=false")
    for path, value in _walk_feature_paths(feature_json):
        lowered = path.lower()
        if any(token in lowered for token in LEAKY_FEATURE_TOKENS):
            raise RuntimeError(f"leaky I12 PIT feature path: {path}")
        if isinstance(value, datetime):
            raise RuntimeError(f"datetime object must be serialized explicitly: {path}")


def quote_windows_for_candidate(
    candidate: I12PitCandidate,
    *,
    before_seconds: float = DEFAULT_QUOTE_WINDOW_BEFORE_SECONDS,
    after_seconds: float = DEFAULT_QUOTE_WINDOW_AFTER_SECONDS,
) -> list[QuoteWindow]:
    same_day_exit = _eastern_timestamp(candidate.decision_date, SAME_DAY_EXIT_TIME)
    next_session = next_us_equity_session(candidate.decision_date + timedelta(days=1))
    next_open_exit = us_equity_session_open_timestamp(next_session) + timedelta(
        minutes=NEXT_OPEN_EXIT_OFFSET_MINUTES
    )
    return [
        _quote_window("entry", candidate.decision_ts, before_seconds, after_seconds),
        _quote_window("same_day_exit", same_day_exit, before_seconds, after_seconds),
        _quote_window("next_open_exit", next_open_exit, before_seconds, after_seconds),
    ]


def replay_quote_window(
    *,
    ticker: str,
    window: QuoteWindow,
    response: AdapterResponse[Any],
    feed: str,
    max_quote_age_seconds: float = DEFAULT_MAX_QUOTE_AGE_SECONDS,
) -> QuoteReplayResult:
    if not response.ok:
        return _quote_result(
            window=window,
            quote=None,
            feed=feed,
            coverage_status="error",
            error_json=_provider_error_payload(response),
        )
    quotes = list(response.data or [])
    selected = _latest_quote_at_or_before(quotes, window.target_ts)
    if selected is None:
        return _quote_result(
            window=window,
            quote=None,
            feed=feed,
            coverage_status="missing",
            error_json=None,
        )
    quote_ts = _parse_quote_ts(selected)
    age = (window.target_ts - quote_ts).total_seconds() if quote_ts else None
    status = "ok"
    if age is None or age < 0:
        status = "missing"
    elif age > max_quote_age_seconds:
        status = "stale"
    bid = _finite_positive(selected.bid_price)
    ask = _finite_positive(selected.ask_price)
    bid_size = _finite_nonnegative(selected.bid_size)
    ask_size = _finite_nonnegative(selected.ask_size)
    spread = _spread_bps(bid, ask)
    bid_notional = bid * bid_size if bid is not None and bid_size is not None else None
    ask_notional = ask * ask_size if ask is not None and ask_size is not None else None
    executable_side = "buy" if window.quote_role == "entry" else "sell"
    executable_notional = ask_notional if executable_side == "buy" else bid_notional
    return QuoteReplayResult(
        quote_role=window.quote_role,
        target_ts=window.target_ts,
        window_start_ts=window.window_start_ts,
        window_end_ts=window.window_end_ts,
        quote=selected,
        quote_ts=quote_ts,
        quote_age_seconds=age,
        bid=bid,
        ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        spread_bps=spread,
        top_of_book_notional=executable_notional,
        bid_notional=bid_notional,
        ask_notional=ask_notional,
        executable_notional=executable_notional,
        executable_side=executable_side,
        feed=feed,
        source="alpaca_historical_quotes",
        coverage_status=status,
        raw_json=selected.raw,
        error_json=None,
    )


def evaluate_quote_cost_replay(
    *,
    entry_quote: I12PitQuoteReplay | None,
    exit_quote: I12PitQuoteReplay | None,
    exit_role: str,
    intended_order_usd: float,
    max_spread_bps: float,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> CostReplayResult:
    skipped_reason = _entry_skip_reason(
        entry_quote,
        intended_order_usd=intended_order_usd,
        max_spread_bps=max_spread_bps,
    )
    if skipped_reason == "none":
        skipped_reason = _exit_skip_reason(
            exit_quote,
            intended_order_usd=intended_order_usd,
            max_spread_bps=max_spread_bps,
        )
    if skipped_reason != "none":
        return CostReplayResult(
            exit_role=exit_role,
            tradeability_status="skipped_cash",
            skipped_reason=skipped_reason,
            intended_order_usd=intended_order_usd,
            max_spread_bps=max_spread_bps,
            slippage_bps=slippage_bps,
            entry_ask=entry_quote.ask if entry_quote is not None else None,
            exit_bid=None,
            gross_return=None,
            quote_cost_return=None,
            slippage_return=None,
            modeled_return=0.0,
        )
    assert entry_quote is not None and exit_quote is not None
    entry_mid = (entry_quote.bid + entry_quote.ask) / 2.0
    exit_mid = (exit_quote.bid + exit_quote.ask) / 2.0
    gross_return = exit_mid / entry_mid - 1.0
    quote_cost_return = exit_quote.bid / entry_quote.ask - 1.0
    slip = slippage_bps / 10000.0
    slip_entry = entry_quote.ask * (1.0 + slip)
    slip_exit = exit_quote.bid * (1.0 - slip)
    slippage_return = slip_exit / slip_entry - 1.0
    return CostReplayResult(
        exit_role=exit_role,
        tradeability_status="tradeable",
        skipped_reason="none",
        intended_order_usd=intended_order_usd,
        max_spread_bps=max_spread_bps,
        slippage_bps=slippage_bps,
        entry_ask=entry_quote.ask,
        exit_bid=exit_quote.bid,
        gross_return=gross_return,
        quote_cost_return=quote_cost_return,
        slippage_return=slippage_return,
        modeled_return=slippage_return,
    )


def i12_pit_rebuild_report(
    session: Session,
    *,
    output_schema: str | None = None,
    source_hur_schema: str | None = None,
    hur_rows_loaded: int | None = None,
    decision_time_count: int | None = None,
    decision_time_labels: Sequence[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    job_run_id: str | None = None,
    path_mode: str | None = STRICT_MINUTE_PATH_MODE,
    compare_path_modes: bool = False,
) -> dict[str, Any]:
    if compare_path_modes:
        report_path_mode = None
        requested_path_modes = tuple(MINUTE_PATH_MODES)
    else:
        report_path_mode = _validate_minute_path_mode(path_mode or STRICT_MINUTE_PATH_MODE)
        requested_path_modes = (report_path_mode,)
    report_decision_time_labels = (
        _normalize_decision_time_labels(decision_time_labels)
        if decision_time_labels is not None
        else None
    )
    base_candidate_query = session.query(I12PitCandidate)
    candidate_query = base_candidate_query.filter(I12PitCandidate.is_active.is_(True))
    if start_date is not None:
        candidate_query = candidate_query.filter(I12PitCandidate.decision_date >= start_date)
        base_candidate_query = base_candidate_query.filter(I12PitCandidate.decision_date >= start_date)
    if end_date is not None:
        candidate_query = candidate_query.filter(I12PitCandidate.decision_date <= end_date)
        base_candidate_query = base_candidate_query.filter(I12PitCandidate.decision_date <= end_date)
    if job_run_id is not None:
        candidate_query = candidate_query.filter(I12PitCandidate.job_run_id == job_run_id)
        base_candidate_query = base_candidate_query.filter(I12PitCandidate.job_run_id == job_run_id)
    unfiltered_base_candidate_query = base_candidate_query
    available_path_modes = _distinct_candidate_values(
        unfiltered_base_candidate_query,
        I12PitCandidate.path_mode,
    )
    mixed_path_modes_present = len(available_path_modes) > 1
    available_decision_time_labels = _distinct_candidate_values(
        unfiltered_base_candidate_query,
        I12PitCandidate.decision_time_label,
    )
    mixed_decision_times_present = len(available_decision_time_labels) > 1
    if report_path_mode is not None:
        candidate_query = candidate_query.filter(I12PitCandidate.path_mode == report_path_mode)
        base_candidate_query = base_candidate_query.filter(
            I12PitCandidate.path_mode == report_path_mode
        )
    if report_decision_time_labels is not None:
        candidate_query = candidate_query.filter(
            I12PitCandidate.decision_time_label.in_(report_decision_time_labels)
        )
        base_candidate_query = base_candidate_query.filter(
            I12PitCandidate.decision_time_label.in_(report_decision_time_labels)
        )
    available_decision_time_labels_for_report_scope = _distinct_candidate_values(
        base_candidate_query,
        I12PitCandidate.decision_time_label,
    )
    historical_candidate_row_count = base_candidate_query.count()
    candidate_aggregates = _report_candidate_aggregates(candidate_query)
    active_candidate_count = candidate_aggregates.active_count
    scoped_candidate_ids_subquery = base_candidate_query.with_entities(
        I12PitCandidate.i12_pit_candidate_id
    ).subquery()
    inactive_scoped_candidate_ids_subquery = (
        base_candidate_query.filter(I12PitCandidate.is_active.is_(False))
        .with_entities(I12PitCandidate.i12_pit_candidate_id)
        .subquery()
    )
    active_nonpassed_scoped_candidate_ids_subquery = (
        candidate_query.filter(I12PitCandidate.candidate_status != "passed")
        .with_entities(I12PitCandidate.i12_pit_candidate_id)
        .subquery()
    )
    passed_query = candidate_query.filter(I12PitCandidate.candidate_status == "passed")
    passed = passed_query.all()
    passed_candidate_ids = [row.i12_pit_candidate_id for row in passed]
    loaded_quotes = _load_child_rows_for_candidate_ids(
        session,
        I12PitQuoteReplay,
        passed_candidate_ids,
    )
    loaded_costs = _load_child_rows_for_candidate_ids(
        session,
        I12PitCostReplay,
        passed_candidate_ids,
    )
    quotes = [
        row for row in loaded_quotes
        if row.is_active
    ]
    costs = [
        row for row in loaded_costs
        if row.is_active
    ]
    historical_quote_replay_row_count = _count_child_rows_for_candidate_subquery(
        session,
        I12PitQuoteReplay,
        scoped_candidate_ids_subquery,
    )
    historical_cost_replay_row_count = _count_child_rows_for_candidate_subquery(
        session,
        I12PitCostReplay,
        scoped_candidate_ids_subquery,
    )
    active_quote_rows_with_inactive_candidate_count = (
        _count_child_rows_for_candidate_subquery(
            session,
            I12PitQuoteReplay,
            inactive_scoped_candidate_ids_subquery,
            active_only=True,
        )
    )
    active_cost_rows_with_inactive_candidate_count = (
        _count_child_rows_for_candidate_subquery(
            session,
            I12PitCostReplay,
            inactive_scoped_candidate_ids_subquery,
            active_only=True,
        )
    )
    active_quote_rows_with_nonpassed_candidate_count = (
        _count_child_rows_for_candidate_subquery(
            session,
            I12PitQuoteReplay,
            active_nonpassed_scoped_candidate_ids_subquery,
            active_only=True,
        )
    )
    active_cost_rows_with_nonpassed_candidate_count = (
        _count_child_rows_for_candidate_subquery(
            session,
            I12PitCostReplay,
            active_nonpassed_scoped_candidate_ids_subquery,
            active_only=True,
        )
    )
    child_evidence_parent_inactive = (
        active_quote_rows_with_inactive_candidate_count > 0
        or active_cost_rows_with_inactive_candidate_count > 0
    )
    child_evidence_parent_nonpassed = (
        active_quote_rows_with_nonpassed_candidate_count > 0
        or active_cost_rows_with_nonpassed_candidate_count > 0
    )
    source_identity_denominator_error: str | None = None
    if hur_rows_loaded is None and start_date is not None and end_date is not None:
        try:
            hur_rows_loaded = _count_hur_rows(
                session,
                source_hur_schema or "public",
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            source_identity_denominator_error = f"{type(exc).__name__}: {exc}"
    if report_decision_time_labels is not None:
        decision_time_count = len(report_decision_time_labels)
    elif decision_time_count is None:
        decision_time_count = (
            len(candidate_aggregates.active_decision_time_labels)
            if active_candidate_count else None
        )
    candidate_coverage_counts = Counter(
        candidate_aggregates.candidate_coverage_status_counts
    )
    quote_status = Counter(row.coverage_status for row in quotes)
    skip_reasons = Counter(row.skipped_reason for row in costs)
    quote_audit = _quote_completeness_audit(passed, quotes)
    cost_audit = _cost_completeness_audit(passed, costs)
    by_exit: dict[str, dict[str, Any]] = {}
    quotes_by_id = {row.i12_pit_quote_replay_id: row for row in quotes}
    for exit_role in EXIT_ROLES:
        rows = [row for row in costs if row.exit_role == exit_role]
        modeled = [row.modeled_return for row in rows]
        tradeable = [row for row in rows if row.tradeability_status == "tradeable"]
        role_quotes = [
            row for row in quotes
            if row.quote_role == exit_role and row.coverage_status == "ok"
        ]
        suff_den = 0
        suff_count = 0
        for row in rows:
            quote = quotes_by_id.get(row.exit_quote_replay_id)
            if quote is None:
                continue
            suff_den += 1
            if (
                quote.executable_notional is not None
                and quote.executable_notional >= row.intended_order_usd
            ):
                suff_count += 1
        by_exit[exit_role] = {
            "candidates": len(passed),
            "row_count": len(rows),
            "tradeable_count": len(tradeable),
            "tradeable_rate": len(tradeable) / len(passed) if passed else None,
            "skipped_cash_count": len(rows) - len(tradeable),
            "skipped_cash_by_reason": dict(Counter(row.skipped_reason for row in rows)),
            "quote_coverage": quote_audit["quote_coverage_by_role"].get(exit_role, {}),
            "spread_bps": _numeric_summary(row.spread_bps for row in role_quotes),
            "executable_notional": _numeric_summary(
                row.executable_notional for row in role_quotes
            ),
            "top_of_book_sufficient_count": suff_count,
            "top_of_book_sufficient_denominator": suff_den,
            "top_of_book_sufficient_rate": (
                suff_count / suff_den if suff_den else None
            ),
            "mean_modeled_return_skips_as_cash": (
                sum(modeled) / len(modeled) if modeled else None
            ),
            "win_rate_skips_as_cash": (
                sum(1 for value in modeled if value > 0.0) / len(modeled)
                if modeled else None
            ),
            "mean_quote_cost_return_tradeable": _mean(
                row.quote_cost_return for row in tradeable
            ),
            "mean_slippage_return_tradeable": _mean(
                row.slippage_return for row in tradeable
            ),
        }
    predecision_volume_tradeability = _volume_tradeability_report(
        passed_candidates=passed,
        quotes=quotes,
        costs=costs,
        default_threshold=DEFAULT_VOLUME_PARTICIPATION_THRESHOLD,
        evidence_getter=_predecision_volume_evidence,
        window_prefix="predecision_window",
        evidence_metadata={
            "entry_volume_window_basis": VOLUME_WINDOW_BASIS_PRE_DECISION,
            "entry_volume_window_price_basis_preference": [
                "entry_quote_mid",
                VOLUME_PRICE_BASIS_LAST_PRE_DECISION,
            ],
            "entry_volume_window_description": (
                "completed start-stamped minute bars strictly before decision_ts; "
                "for 09:40 this is 09:30 through 09:39 ET, not the 09:40 bar"
            ),
        },
    )
    execution_window_volume_tradeability = _volume_tradeability_report(
        passed_candidates=passed,
        quotes=quotes,
        costs=costs,
        default_threshold=DEFAULT_VOLUME_PARTICIPATION_THRESHOLD,
        evidence_getter=_execution_window_volume_evidence,
        window_prefix="execution_window",
        evidence_metadata={
            "execution_window_basis": "persisted_execution_window_minute_bars",
            "execution_window_description": (
                "actual execution-window minute volume beginning at decision_ts; "
                "absent unless a scratch/cache backfill persisted this evidence"
            ),
        },
    )
    required_quote_rows = len(passed) * len(REQUIRED_QUOTE_ROLES)
    quote_ok_count = quote_status.get("ok", 0)
    quote_non_ok_count = len(quotes) - quote_ok_count
    quote_coverage_rate = (
        quote_audit["durable_quote_role_count"] / required_quote_rows
        if required_quote_rows else None
    )
    quote_ok_rate = (
        quote_audit["usable_quote_role_count"] / required_quote_rows
        if required_quote_rows else None
    )
    quote_complete = quote_audit["quote_replay_complete"]
    cost_complete = cost_audit["cost_replay_complete"]
    zero_pit_candidates = len(passed) == 0
    if zero_pit_candidates:
        quote_replay_status = "not_applicable"
        cost_replay_status = "not_applicable"
    else:
        quote_replay_status = "complete" if quote_complete else "incomplete"
        cost_replay_status = "complete" if cost_complete else "incomplete"
    expected_candidate_attempts_per_path_mode = (
        hur_rows_loaded * decision_time_count
        if hur_rows_loaded is not None and decision_time_count is not None
        else None
    )
    expected_candidate_attempts = (
        expected_candidate_attempts_per_path_mode * len(requested_path_modes)
        if expected_candidate_attempts_per_path_mode is not None
        else None
    )
    identity_audit = _source_attempt_identity_audit(
        session,
        source_hur_schema=source_hur_schema or "public",
        start_date=start_date,
        end_date=end_date,
        decision_time_labels=report_decision_time_labels,
        path_modes=requested_path_modes,
        candidate_query=candidate_query,
    )
    if source_identity_denominator_error is not None:
        identity_audit = _source_identity_error_audit(
            source_identity_denominator_error,
            requested_path_modes,
        )
    missing_source_attempt_count = (
        max(expected_candidate_attempts - active_candidate_count, 0)
        if expected_candidate_attempts is not None
        else None
    )
    extra_source_attempt_count = (
        max(active_candidate_count - expected_candidate_attempts, 0)
        if expected_candidate_attempts is not None
        else None
    )
    daily_fetch_error_count = candidate_coverage_counts.get("daily_fetch_error", 0)
    minute_fetch_error_count = candidate_coverage_counts.get("minute_fetch_error", 0)
    source_provider_error_count = daily_fetch_error_count + minute_fetch_error_count
    source_denominator_known = expected_candidate_attempts is not None
    identity_denominator_error = identity_audit.get("source_identity_denominator_error")
    aggregate_source_attempts_complete = (
        source_denominator_known
        and missing_source_attempt_count == 0
        and extra_source_attempt_count == 0
        and identity_audit["source_identity_denominator_known"]
        and identity_audit["missing_source_attempt_identity_count"] == 0
        and identity_audit["extra_source_attempt_identity_count"] == 0
        and source_provider_error_count == 0
    )
    path_mode_metrics = _path_mode_report_metrics(
        candidate_aggregates,
        passed,
        quotes,
        costs,
        requested_path_modes=requested_path_modes,
        expected_candidate_attempts=expected_candidate_attempts_per_path_mode,
        identity_audits_by_mode=identity_audit["path_mode_identity_audits"],
        source_denominator_known=source_denominator_known,
        zero_hur_source_blocked=hur_rows_loaded == 0 if hur_rows_loaded is not None else False,
        child_evidence_parent_inactive=child_evidence_parent_inactive,
        child_evidence_parent_nonpassed=child_evidence_parent_nonpassed,
        compare_path_modes=compare_path_modes,
    )
    strict_partial_rows_that_would_pass_sparse_zero_fill = (
        _strict_partial_rows_that_would_pass_sparse_zero_fill(candidate_query)
    )
    comparison_conclusions_final = (
        all(
            path_mode_metrics.get(mode, {}).get("conclusions_final") is True
            for mode in requested_path_modes
        )
        if compare_path_modes else None
    )
    path_mode_source_attempts_complete = (
        all(
            _path_mode_source_attempts_complete(path_mode_metrics.get(mode, {}))
            for mode in requested_path_modes
        )
        if compare_path_modes else True
    )
    source_attempts_complete = (
        aggregate_source_attempts_complete and path_mode_source_attempts_complete
    )
    zero_hur_source_blocked = hur_rows_loaded == 0 if hur_rows_loaded is not None else False
    data_integrity_passed = (
        source_attempts_complete
        and not zero_pit_candidates
        and quote_complete
        and cost_complete
        and not child_evidence_parent_inactive
        and not child_evidence_parent_nonpassed
        and not zero_hur_source_blocked
        and (comparison_conclusions_final is not False)
    )
    if child_evidence_parent_inactive:
        training_status = "blocked_child_evidence_parent_inactive"
    elif child_evidence_parent_nonpassed:
        training_status = "blocked_child_evidence_parent_nonpassed"
    elif identity_denominator_error:
        training_status = "blocked_source_identity_denominator_error"
    elif not source_denominator_known:
        training_status = "blocked_source_denominator_unknown"
    elif zero_hur_source_blocked:
        training_status = "blocked_zero_hur_source"
    elif source_denominator_known and not identity_audit["source_identity_denominator_known"]:
        training_status = "blocked_source_identity_denominator_unknown"
    elif missing_source_attempt_count not in {None, 0}:
        training_status = "blocked_source_replay_incomplete"
    elif extra_source_attempt_count not in {None, 0}:
        training_status = "blocked_source_replay_extra_active_attempts"
    elif (
        identity_audit["source_identity_denominator_known"]
        and (
            identity_audit["missing_source_attempt_identity_count"] > 0
            or identity_audit["extra_source_attempt_identity_count"] > 0
        )
    ):
        training_status = "blocked_source_identity_mismatch"
    elif compare_path_modes and not path_mode_source_attempts_complete:
        training_status = "blocked_path_mode_comparison_incomplete"
    elif source_provider_error_count > 0:
        training_status = "blocked_source_provider_errors"
    elif zero_pit_candidates:
        training_status = "blocked_zero_pit_candidates"
    elif quote_audit["unknown_quote_coverage_status_count"] > 0:
        training_status = "blocked_quote_replay_integrity"
    elif not quote_complete:
        training_status = "blocked_quote_replay_incomplete"
    elif not cost_complete:
        training_status = "blocked_cost_replay_incomplete"
    else:
        training_status = "eligible_for_retrain_evaluation"
    if (
        compare_path_modes
        and training_status == "eligible_for_retrain_evaluation"
        and comparison_conclusions_final is not True
    ):
        training_status = "blocked_path_mode_comparison_incomplete"
    return {
        "schema": output_schema,
        "output_schema": output_schema,
        "start_date": start_date.isoformat() if start_date is not None else None,
        "end_date": end_date.isoformat() if end_date is not None else None,
        "report_path_mode": report_path_mode,
        "minute_path_mode": report_path_mode,
        "compare_path_modes": compare_path_modes,
        "comparison_path_modes": list(requested_path_modes) if compare_path_modes else [],
        "available_path_modes": available_path_modes,
        "mixed_path_modes_present": mixed_path_modes_present,
        "report_decision_time_labels": list(report_decision_time_labels or []),
        "available_decision_time_labels": available_decision_time_labels,
        "available_decision_time_labels_for_report_scope": (
            available_decision_time_labels_for_report_scope
        ),
        "mixed_decision_times_present": mixed_decision_times_present,
        "comparison_conclusions_final": comparison_conclusions_final,
        "source_hur_schema": source_hur_schema or "public",
        "hur_rows_loaded": hur_rows_loaded,
        "zero_hur_source_blocked": zero_hur_source_blocked,
        "decision_time_count": decision_time_count,
        "expected_candidate_attempts": expected_candidate_attempts,
        "candidate_row_count": active_candidate_count,
        "actual_candidate_row_count": active_candidate_count,
        "active_candidate_row_count": active_candidate_count,
        "historical_candidate_row_count": historical_candidate_row_count,
        "missing_source_attempt_count": missing_source_attempt_count,
        "extra_source_attempt_count": extra_source_attempt_count,
        "source_identity_denominator_known": identity_audit[
            "source_identity_denominator_known"
        ],
        "source_identity_denominator_error": identity_denominator_error,
        "expected_source_attempt_identity_count": identity_audit[
            "expected_source_attempt_identity_count"
        ],
        "actual_source_attempt_identity_count": identity_audit[
            "actual_source_attempt_identity_count"
        ],
        "missing_source_attempt_identity_count": identity_audit[
            "missing_source_attempt_identity_count"
        ],
        "extra_source_attempt_identity_count": identity_audit[
            "extra_source_attempt_identity_count"
        ],
        "missing_source_attempt_identities_sample": identity_audit[
            "missing_source_attempt_identities_sample"
        ],
        "extra_source_attempt_identities_sample": identity_audit[
            "extra_source_attempt_identities_sample"
        ],
        "daily_fetch_error_count": daily_fetch_error_count,
        "minute_fetch_error_count": minute_fetch_error_count,
        "source_provider_error_count": source_provider_error_count,
        "source_denominator_known": source_denominator_known,
        "source_replay_complete": source_attempts_complete,
        "pit_candidate_count": len(passed),
        "candidate_status_counts": candidate_aggregates.candidate_status_counts,
        "candidate_coverage_status_counts": dict(candidate_coverage_counts),
        "daily_source_hash_basis_counts": (
            candidate_aggregates.daily_source_hash_basis_counts
        ),
        "daily_source_hash_reuse_status_counts": (
            candidate_aggregates.daily_source_hash_reuse_status_counts
        ),
        "candidate_counts_by_path_mode": candidate_aggregates.candidate_counts_by_path_mode,
        "coverage_status_by_path_mode": candidate_aggregates.coverage_status_by_path_mode,
        "passed_candidates_by_path_mode": dict(Counter(row.path_mode for row in passed)),
        "passed_candidates_by_path_mode_decision_time": (
            _passed_candidates_by_path_mode_decision_time(passed)
        ),
        "strict_partial_rows_that_would_pass_sparse_zero_fill": (
            strict_partial_rows_that_would_pass_sparse_zero_fill
        ),
        "sparse_imputation_distributions": _sparse_imputation_distributions(candidate_query),
        "path_mode_metrics": path_mode_metrics,
        "quote_replay_row_count": len(quotes),
        "historical_quote_replay_row_count": historical_quote_replay_row_count,
        "active_quote_rows_with_inactive_candidate_count": (
            active_quote_rows_with_inactive_candidate_count
        ),
        "active_quote_rows_with_nonpassed_candidate_count": (
            active_quote_rows_with_nonpassed_candidate_count
        ),
        "quote_coverage_status_counts": dict(quote_status),
        "quote_replay_status": quote_replay_status,
        "quote_coverage_rate": quote_coverage_rate,
        "quote_ok_count": quote_ok_count,
        "quote_non_ok_count": quote_non_ok_count,
        "quote_ok_rate": quote_ok_rate,
        "quote_usable_ok_count": quote_audit["usable_quote_role_count"],
        "unknown_quote_coverage_status_count": (
            quote_audit["unknown_quote_coverage_status_count"]
        ),
        "quote_coverage_by_role": quote_audit["quote_coverage_by_role"],
        "missing_quote_role_count": quote_audit["missing_quote_role_count"],
        "duplicate_quote_role_count": quote_audit["duplicate_quote_role_count"],
        "candidate_complete_quote_count": quote_audit["candidate_complete_quote_count"],
        "candidate_incomplete_quote_count": quote_audit["candidate_incomplete_quote_count"],
        "cost_replay_row_count": len(costs),
        "historical_cost_replay_row_count": historical_cost_replay_row_count,
        "active_cost_rows_with_inactive_candidate_count": (
            active_cost_rows_with_inactive_candidate_count
        ),
        "active_cost_rows_with_nonpassed_candidate_count": (
            active_cost_rows_with_nonpassed_candidate_count
        ),
        "cost_replay_status": cost_replay_status,
        "cost_replay_complete": cost_complete,
        "missing_cost_role_count": cost_audit["missing_cost_role_count"],
        "duplicate_cost_role_count": cost_audit["duplicate_cost_role_count"],
        "candidate_complete_cost_count": cost_audit["candidate_complete_cost_count"],
        "candidate_incomplete_cost_count": cost_audit["candidate_incomplete_cost_count"],
        "cost_coverage_by_exit_role": cost_audit["cost_coverage_by_exit_role"],
        "skip_reason_counts": dict(skip_reasons),
        "exit_metrics": by_exit,
        "candidate_semantics": {
            "candidate_status_passed": (
                "detector gate passed using PIT-safe decision-time features; "
                "tradeability is evaluated separately in the execution/cost layer"
            ),
            "displayed_tradeability_metrics": (
                "strict displayed top-of-book size model from persisted cost rows"
            ),
            "volume_tradeability_metrics": (
                "backward-compatible alias for pre-decision volume-participation "
                "metrics using persisted completed pre-decision minute volume"
            ),
            "predecision_volume_tradeability_metrics": (
                "diagnostic volume-participation model using persisted completed "
                "pre-decision minute volume and entry/exit quote costs"
            ),
            "execution_window_volume_tradeability_metrics": (
                "optional diagnostic volume-participation model using persisted "
                "actual execution-window minute volume when available"
            ),
        },
        "displayed_tradeability_metrics": by_exit,
        "displayed_size_tradeability_status_counts": dict(
            Counter(row.tradeability_status for row in costs)
        ),
        "predecision_volume_tradeability_metrics": predecision_volume_tradeability["metrics"],
        "predecision_volume_tradeability_sensitivity": (
            predecision_volume_tradeability["sensitivity"]
        ),
        "predecision_volume_tradeability_status_counts": (
            predecision_volume_tradeability["status_counts"]
        ),
        "predecision_volume_tradeability_skip_reason_counts": (
            predecision_volume_tradeability["skip_reason_counts"]
        ),
        "predecision_volume_tradeability_rate": (
            predecision_volume_tradeability["tradeability_rate"]
        ),
        "predecision_volume_tradeability_evidence": predecision_volume_tradeability[
            "evidence"
        ],
        "execution_window_volume_tradeability_metrics": (
            execution_window_volume_tradeability["metrics"]
        ),
        "execution_window_volume_tradeability_sensitivity": (
            execution_window_volume_tradeability["sensitivity"]
        ),
        "execution_window_volume_tradeability_status_counts": (
            execution_window_volume_tradeability["status_counts"]
        ),
        "execution_window_volume_tradeability_skip_reason_counts": (
            execution_window_volume_tradeability["skip_reason_counts"]
        ),
        "execution_window_volume_tradeability_rate": (
            execution_window_volume_tradeability["tradeability_rate"]
        ),
        "execution_window_volume_tradeability_evidence": (
            execution_window_volume_tradeability["evidence"]
        ),
        "volume_tradeability_metrics": predecision_volume_tradeability["metrics"],
        "volume_tradeability_sensitivity": predecision_volume_tradeability["sensitivity"],
        "volume_tradeability_status_counts": predecision_volume_tradeability["status_counts"],
        "volume_tradeability_skip_reason_counts": (
            predecision_volume_tradeability["skip_reason_counts"]
        ),
        "volume_tradeability_rate": predecision_volume_tradeability["tradeability_rate"],
        "volume_tradeability_evidence": predecision_volume_tradeability["evidence"],
        "decision_time_buckets": _decision_time_buckets(
            candidate_aggregates.decision_time_bucket_counts,
            passed,
            costs,
        ),
        "data_integrity_passed": data_integrity_passed,
        "quote_replay_complete": quote_complete,
        "training_status": training_status,
        "conclusions_final": data_integrity_passed,
        "ml_ranking_status": (
            "not_run_quote_layer_ready"
            if data_integrity_passed
            else (
                training_status
                if training_status.startswith("blocked_source")
                or training_status == "blocked_zero_hur_source"
                or training_status == "blocked_child_evidence_parent_inactive"
                or training_status == "blocked_child_evidence_parent_nonpassed"
                or training_status == "blocked_zero_pit_candidates"
                or training_status == "blocked_path_mode_comparison_incomplete"
                or training_status == "blocked_quote_replay_integrity"
                else "blocked_until_quote_replay_complete"
            )
        ),
    }


def _quote_completeness_audit(
    passed_candidates: Sequence[I12PitCandidate],
    quotes: Sequence[I12PitQuoteReplay],
) -> dict[str, Any]:
    quotes_by_candidate_role: dict[tuple[str, str], list[I12PitQuoteReplay]] = defaultdict(list)
    for quote in quotes:
        quotes_by_candidate_role[
            (quote.i12_pit_candidate_id, quote.quote_role)
        ].append(quote)

    coverage_by_role: dict[str, dict[str, Any]] = {
        role: {
            "required": len(passed_candidates),
            "present": 0,
            "usable_ok": 0,
            "missing": 0,
            "duplicate": 0,
            "unknown_status": 0,
            "coverage_status_counts": {},
        }
        for role in REQUIRED_QUOTE_ROLES
    }
    missing_quote_role_count = 0
    duplicate_quote_role_count = 0
    complete_candidates = 0
    usable_quote_role_count = 0
    durable_quote_role_count = 0
    unknown_quote_coverage_status_count = 0

    for candidate in passed_candidates:
        candidate_complete = True
        for role in REQUIRED_QUOTE_ROLES:
            rows = quotes_by_candidate_role.get(
                (candidate.i12_pit_candidate_id, role),
                [],
            )
            role_bucket = coverage_by_role[role]
            if not rows:
                missing_quote_role_count += 1
                role_bucket["missing"] += 1
                candidate_complete = False
                continue
            status_counts = Counter(row.coverage_status for row in rows)
            role_bucket["coverage_status_counts"] = dict(
                Counter(role_bucket["coverage_status_counts"]) + status_counts
            )
            unknown_status_count = sum(
                count
                for status, count in status_counts.items()
                if status not in QUOTE_REPLAY_DURABLE_COVERAGE_STATUSES
            )
            if unknown_status_count:
                role_bucket["unknown_status"] += unknown_status_count
                unknown_quote_coverage_status_count += unknown_status_count
                candidate_complete = False
            if len(rows) > 1:
                duplicate_quote_role_count += len(rows) - 1
                role_bucket["duplicate"] += len(rows) - 1
                candidate_complete = False
            elif len(rows) == 1:
                role_bucket["present"] += 1
                if rows[0].coverage_status in QUOTE_REPLAY_DURABLE_COVERAGE_STATUSES:
                    durable_quote_role_count += 1
                    if rows[0].coverage_status == "ok":
                        role_bucket["usable_ok"] += 1
                        usable_quote_role_count += 1
                else:
                    candidate_complete = False
        if candidate_complete:
            complete_candidates += 1

    incomplete_candidates = len(passed_candidates) - complete_candidates
    return {
        "quote_coverage_by_role": coverage_by_role,
        "missing_quote_role_count": missing_quote_role_count,
        "duplicate_quote_role_count": duplicate_quote_role_count,
        "candidate_complete_quote_count": complete_candidates,
        "candidate_incomplete_quote_count": incomplete_candidates,
        "usable_quote_role_count": usable_quote_role_count,
        "durable_quote_role_count": durable_quote_role_count,
        "unknown_quote_coverage_status_count": unknown_quote_coverage_status_count,
        "quote_replay_complete": bool(passed_candidates)
        and complete_candidates == len(passed_candidates)
        and missing_quote_role_count == 0
        and duplicate_quote_role_count == 0
        and unknown_quote_coverage_status_count == 0,
    }


def _cost_completeness_audit(
    passed_candidates: Sequence[I12PitCandidate],
    costs: Sequence[I12PitCostReplay],
) -> dict[str, Any]:
    costs_by_candidate_role: dict[tuple[str, str], list[I12PitCostReplay]] = defaultdict(list)
    for cost in costs:
        costs_by_candidate_role[(cost.i12_pit_candidate_id, cost.exit_role)].append(cost)

    coverage_by_role: dict[str, dict[str, Any]] = {
        role: {
            "required": len(passed_candidates),
            "present": 0,
            "missing": 0,
            "duplicate": 0,
            "tradeability_status_counts": {},
            "skipped_reason_counts": {},
        }
        for role in EXIT_ROLES
    }
    missing_cost_role_count = 0
    duplicate_cost_role_count = 0
    complete_candidates = 0

    for candidate in passed_candidates:
        candidate_complete = True
        for role in EXIT_ROLES:
            rows = costs_by_candidate_role.get(
                (candidate.i12_pit_candidate_id, role),
                [],
            )
            role_bucket = coverage_by_role[role]
            if not rows:
                missing_cost_role_count += 1
                role_bucket["missing"] += 1
                candidate_complete = False
                continue
            role_bucket["tradeability_status_counts"] = dict(
                Counter(role_bucket["tradeability_status_counts"])
                + Counter(row.tradeability_status for row in rows)
            )
            role_bucket["skipped_reason_counts"] = dict(
                Counter(role_bucket["skipped_reason_counts"])
                + Counter(row.skipped_reason for row in rows)
            )
            if len(rows) == 1:
                role_bucket["present"] += 1
            else:
                duplicate_cost_role_count += len(rows) - 1
                role_bucket["duplicate"] += len(rows) - 1
                candidate_complete = False
        if candidate_complete:
            complete_candidates += 1

    incomplete_candidates = len(passed_candidates) - complete_candidates
    return {
        "cost_coverage_by_exit_role": coverage_by_role,
        "missing_cost_role_count": missing_cost_role_count,
        "duplicate_cost_role_count": duplicate_cost_role_count,
        "candidate_complete_cost_count": complete_candidates,
        "candidate_incomplete_cost_count": incomplete_candidates,
        "cost_replay_complete": bool(passed_candidates)
        and complete_candidates == len(passed_candidates)
        and missing_cost_role_count == 0
        and duplicate_cost_role_count == 0,
    }


def _distinct_candidate_values(candidate_query: Any, column: Any) -> list[str]:
    return [
        row[0]
        for row in candidate_query.with_entities(column)
        .distinct()
        .order_by(column)
        .all()
        if row[0] is not None
    ]


def _candidate_group_counts(candidate_query: Any, column: Any) -> dict[str, int]:
    return {
        value: int(count or 0)
        for value, count in candidate_query.with_entities(
            column,
            func.count(I12PitCandidate.i12_pit_candidate_id),
        )
        .group_by(column)
        .all()
    }


def _candidate_nested_group_counts(
    candidate_query: Any,
    outer_column: Any,
    inner_column: Any,
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    rows = (
        candidate_query.with_entities(
            outer_column,
            inner_column,
            func.count(I12PitCandidate.i12_pit_candidate_id),
        )
        .group_by(outer_column, inner_column)
        .all()
    )
    for outer, inner, count in rows:
        out[outer][inner] = int(count or 0)
    return {key: dict(value) for key, value in sorted(out.items())}


def _candidate_decision_time_bucket_counts(
    candidate_query: Any,
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    rows = (
        candidate_query.with_entities(
            I12PitCandidate.decision_time_label,
            I12PitCandidate.candidate_status,
            I12PitCandidate.coverage_status,
            func.count(I12PitCandidate.i12_pit_candidate_id),
        )
        .group_by(
            I12PitCandidate.decision_time_label,
            I12PitCandidate.candidate_status,
            I12PitCandidate.coverage_status,
        )
        .all()
    )
    for label, candidate_status, coverage_status, count in rows:
        bucket = buckets.setdefault(
            label,
            {
                "candidate_count": 0,
                "passed_count": 0,
                "candidate_status_counts": Counter(),
                "coverage_status_counts": Counter(),
            },
        )
        row_count = int(count or 0)
        bucket["candidate_count"] += row_count
        if candidate_status == "passed":
            bucket["passed_count"] += row_count
        bucket["candidate_status_counts"][candidate_status] += row_count
        bucket["coverage_status_counts"][coverage_status] += row_count
    return {
        label: {
            "candidate_count": bucket["candidate_count"],
            "passed_count": bucket["passed_count"],
            "candidate_status_counts": dict(bucket["candidate_status_counts"]),
            "coverage_status_counts": dict(bucket["coverage_status_counts"]),
        }
        for label, bucket in sorted(buckets.items())
    }


def _daily_source_hash_counts_from_query(
    candidate_query: Any,
) -> tuple[
    dict[str, int],
    dict[str, int],
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
]:
    basis_counts: Counter[str] = Counter()
    reuse_counts: Counter[str] = Counter()
    basis_by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    reuse_by_mode: dict[str, Counter[str]] = defaultdict(Counter)

    dialect_name = getattr(candidate_query.session.get_bind().dialect, "name", "")
    basis_expr = _source_bars_json_text_expr(
        "daily_source_hash_basis",
        dialect_name=dialect_name,
    )
    reuse_expr = _source_bars_json_text_expr(
        "daily_source_hash_reuse_status",
        dialect_name=dialect_name,
    )
    basis_value = func.coalesce(func.nullif(basis_expr, ""), "unknown")
    reuse_value = func.coalesce(func.nullif(reuse_expr, ""), "unknown")
    rows = (
        candidate_query.with_entities(
            I12PitCandidate.path_mode,
            basis_value,
            reuse_value,
            func.count(I12PitCandidate.i12_pit_candidate_id),
        )
        .group_by(I12PitCandidate.path_mode, basis_value, reuse_value)
        .all()
    )
    for path_mode, basis, reuse_status, count in rows:
        row_count = int(count or 0)
        basis = str(basis or "unknown")
        reuse_status = str(reuse_status or "unknown")
        basis_counts[basis] += row_count
        reuse_counts[reuse_status] += row_count
        basis_by_mode[path_mode][basis] += row_count
        reuse_by_mode[path_mode][reuse_status] += row_count
    return (
        dict(basis_counts),
        dict(reuse_counts),
        {mode: dict(counter) for mode, counter in sorted(basis_by_mode.items())},
        {mode: dict(counter) for mode, counter in sorted(reuse_by_mode.items())},
    )


def _source_bars_json_text_expr(key: str, *, dialect_name: str) -> Any:
    if dialect_name == "postgresql":
        return func.jsonb_extract_path_text(
            cast(I12PitCandidate.source_bars_json, postgresql.JSONB),
            key,
        )
    if dialect_name == "sqlite":
        return func.json_extract(I12PitCandidate.source_bars_json, f"$.{key}")
    raise RuntimeError(
        f"I12 PIT report JSON aggregation is unsupported for dialect {dialect_name!r}"
    )


def _report_candidate_aggregates(candidate_query: Any) -> _ReportCandidateAggregates:
    (
        daily_source_hash_basis_counts,
        daily_source_hash_reuse_status_counts,
        daily_source_hash_basis_counts_by_path_mode,
        daily_source_hash_reuse_status_counts_by_path_mode,
    ) = _daily_source_hash_counts_from_query(candidate_query)
    return _ReportCandidateAggregates(
        active_count=candidate_query.count(),
        active_decision_time_labels=_distinct_candidate_values(
            candidate_query,
            I12PitCandidate.decision_time_label,
        ),
        candidate_status_counts=_candidate_group_counts(
            candidate_query,
            I12PitCandidate.candidate_status,
        ),
        candidate_coverage_status_counts=_candidate_group_counts(
            candidate_query,
            I12PitCandidate.coverage_status,
        ),
        candidate_counts_by_path_mode=_candidate_group_counts(
            candidate_query,
            I12PitCandidate.path_mode,
        ),
        coverage_status_by_path_mode=_candidate_nested_group_counts(
            candidate_query,
            I12PitCandidate.path_mode,
            I12PitCandidate.coverage_status,
        ),
        candidate_status_counts_by_path_mode=_candidate_nested_group_counts(
            candidate_query,
            I12PitCandidate.path_mode,
            I12PitCandidate.candidate_status,
        ),
        daily_source_hash_basis_counts=daily_source_hash_basis_counts,
        daily_source_hash_reuse_status_counts=daily_source_hash_reuse_status_counts,
        daily_source_hash_basis_counts_by_path_mode=(
            daily_source_hash_basis_counts_by_path_mode
        ),
        daily_source_hash_reuse_status_counts_by_path_mode=(
            daily_source_hash_reuse_status_counts_by_path_mode
        ),
        decision_time_bucket_counts=_candidate_decision_time_bucket_counts(
            candidate_query
        ),
    )


def _decision_time_buckets(
    candidate_bucket_counts: Mapping[str, Mapping[str, Any]],
    passed: Sequence[I12PitCandidate],
    costs: Sequence[I12PitCostReplay],
) -> dict[str, Any]:
    costs_by_candidate: dict[str, list[I12PitCostReplay]] = defaultdict(list)
    for row in costs:
        costs_by_candidate[row.i12_pit_candidate_id].append(row)
    passed_by_label: dict[str, list[I12PitCandidate]] = defaultdict(list)
    for row in passed:
        passed_by_label[row.decision_time_label].append(row)
    buckets: dict[str, Any] = {}
    for label in sorted(candidate_bucket_counts):
        counts = candidate_bucket_counts[label]
        label_passed = passed_by_label.get(label, [])
        bucket: dict[str, Any] = {
            "candidate_count": int(counts.get("candidate_count") or 0),
            "passed_count": int(counts.get("passed_count") or 0),
            "candidate_status_counts": dict(counts.get("candidate_status_counts") or {}),
            "coverage_status_counts": dict(counts.get("coverage_status_counts") or {}),
        }
        for exit_role in EXIT_ROLES:
            role_costs = [
                cost
                for candidate in label_passed
                for cost in costs_by_candidate.get(candidate.i12_pit_candidate_id, [])
                if cost.exit_role == exit_role
            ]
            tradeable = [
                cost for cost in role_costs
                if cost.tradeability_status == "tradeable"
            ]
            bucket[exit_role] = {
                "cost_row_count": len(role_costs),
                "tradeable_count": len(tradeable),
                "tradeable_rate": (
                    len(tradeable) / len(label_passed) if label_passed else None
                ),
                "skipped_cash_by_reason": dict(
                    Counter(cost.skipped_reason for cost in role_costs)
                ),
                "mean_modeled_return_skips_as_cash": _mean(
                    cost.modeled_return for cost in role_costs
                ),
            }
        buckets[label] = bucket
    return buckets


def _coverage_status_by_path_mode(
    candidates: Sequence[I12PitCandidate],
) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in candidates:
        out[candidate.path_mode][candidate.coverage_status] += 1
    return {mode: dict(counter) for mode, counter in sorted(out.items())}


def _daily_source_hash_basis_counts(
    candidates: Sequence[I12PitCandidate],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        source_bars = _json_loads(candidate.source_bars_json)
        basis = source_bars.get("daily_source_hash_basis")
        counts[str(basis or "unknown")] += 1
    return dict(counts)


def _daily_source_hash_reuse_status_counts(
    candidates: Sequence[I12PitCandidate],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        source_bars = _json_loads(candidate.source_bars_json)
        reuse_status = source_bars.get("daily_source_hash_reuse_status")
        counts[str(reuse_status or "unknown")] += 1
    return dict(counts)


def _passed_candidates_by_path_mode_decision_time(
    passed: Sequence[I12PitCandidate],
) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in passed:
        out[candidate.path_mode][candidate.decision_time_label] += 1
    return {mode: dict(counter) for mode, counter in sorted(out.items())}


def _source_attempt_identity_audit(
    session: Session,
    *,
    source_hur_schema: str,
    start_date: date | None,
    end_date: date | None,
    decision_time_labels: Sequence[str] | None,
    path_modes: Sequence[str],
    candidate_query: Any,
) -> dict[str, Any]:
    known = (
        bool(source_hur_schema)
        and start_date is not None
        and end_date is not None
        and bool(decision_time_labels)
        and bool(path_modes)
    )
    empty = _empty_source_identity_audit(known=False)
    if not known:
        empty["path_mode_identity_audits"] = {
            mode: _empty_source_identity_audit(known=False) for mode in path_modes
        }
        return empty

    try:
        expected_by_mode = {
            mode: _expected_source_attempt_hashes_for_mode(
                session,
                source_hur_schema=source_hur_schema,
                start_date=start_date,
                end_date=end_date,
                decision_time_labels=decision_time_labels or (),
                path_mode=mode,
            )
            for mode in path_modes
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return _source_identity_error_audit(error, path_modes)
    actual_by_mode: dict[str, set[str]] = {mode: set() for mode in path_modes}
    actual_rows = (
        candidate_query.filter(I12PitCandidate.path_mode.in_(path_modes))
        .with_entities(
            I12PitCandidate.path_mode,
            I12PitCandidate.candidate_attempt_hash,
        )
        .all()
    )
    for path_mode, candidate_attempt_hash in actual_rows:
        if path_mode in actual_by_mode and candidate_attempt_hash is not None:
            actual_by_mode[path_mode].add(candidate_attempt_hash)
    path_mode_identity_audits = {
        mode: _source_identity_set_diff(expected_by_mode[mode], actual_by_mode[mode])
        for mode in path_modes
    }
    expected_all = set().union(*expected_by_mode.values()) if expected_by_mode else set()
    actual_all = set().union(*actual_by_mode.values()) if actual_by_mode else set()
    out = _source_identity_set_diff(expected_all, actual_all)
    out["path_mode_identity_audits"] = path_mode_identity_audits
    return out


def _source_identity_error_audit(
    error: str,
    path_modes: Sequence[str],
) -> dict[str, Any]:
    out = _empty_source_identity_audit(known=False, error=error)
    out["path_mode_identity_audits"] = {
        mode: _empty_source_identity_audit(known=False, error=error)
        for mode in path_modes
    }
    return out


def _empty_source_identity_audit(
    *,
    known: bool,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "source_identity_denominator_known": known,
        "source_identity_denominator_error": error,
        "expected_source_attempt_identity_count": None,
        "actual_source_attempt_identity_count": None,
        "missing_source_attempt_identity_count": None,
        "extra_source_attempt_identity_count": None,
        "missing_source_attempt_identities_sample": [],
        "extra_source_attempt_identities_sample": [],
    }


def _source_identity_set_diff(
    expected: set[str],
    actual: set[str],
) -> dict[str, Any]:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return {
        "source_identity_denominator_known": True,
        "source_identity_denominator_error": None,
        "expected_source_attempt_identity_count": len(expected),
        "actual_source_attempt_identity_count": len(actual),
        "missing_source_attempt_identity_count": len(missing),
        "extra_source_attempt_identity_count": len(extra),
        "missing_source_attempt_identities_sample": missing[:20],
        "extra_source_attempt_identities_sample": extra[:20],
    }


def _expected_source_attempt_hashes_for_mode(
    session: Session,
    *,
    source_hur_schema: str,
    start_date: date,
    end_date: date,
    decision_time_labels: Sequence[str],
    path_mode: str,
) -> set[str]:
    hashes: set[str] = set()
    for trading_date in _session_dates(start_date, end_date):
        for hur_row in _load_hur_source_rows_for_report(
            session,
            source_hur_schema=source_hur_schema,
            trading_date=trading_date,
        ):
            for label in decision_time_labels:
                hashes.add(_candidate_attempt_hash(
                    ticker=hur_row.ticker,
                    trading_date=trading_date,
                    decision_ts=_decision_timestamp(trading_date, label),
                    decision_time_label=label,
                    source_hur_identity_hash=hur_row.source_hur_identity_hash,
                    path_mode=path_mode,
                ))
    return hashes


def _session_dates(start_date: date, end_date: date) -> list[date]:
    return [
        start_date + timedelta(days=offset)
        for offset in range((end_date - start_date).days + 1)
        if is_us_equity_session(start_date + timedelta(days=offset))
    ]


def _load_hur_source_rows_for_report(
    session: Session,
    *,
    source_hur_schema: str,
    trading_date: date,
) -> list[HurSourceRow]:
    if _should_schema_qualify_hur(session, source_hur_schema):
        rows = session.execute(
            text(
                "SELECT * "
                f"FROM {_quote_ident(source_hur_schema)}."
                "historical_universe_reconstructions "
                "WHERE replay_date = :trading_date "
                "AND inclusion_status = 'included' "
                "ORDER BY normalized_symbol"
            ),
            {"trading_date": trading_date},
        ).all()
        return [
            _hur_source_row_from_mapping(
                row._mapping,
                source_schema=source_hur_schema,
                fallback_date=trading_date,
            )
            for row in rows
        ]
    rows = (
        session.query(HistoricalUniverseReconstruction)
        .filter(
            HistoricalUniverseReconstruction.replay_date == trading_date,
            HistoricalUniverseReconstruction.inclusion_status == "included",
        )
        .order_by(HistoricalUniverseReconstruction.normalized_symbol)
        .all()
    )
    return [
        _hur_source_row_from_model(row, source_schema=source_hur_schema)
        for row in rows
    ]


def _strict_partial_rows_that_would_pass_sparse_zero_fill(
    candidate_query: Any,
) -> int:
    sparse_pass_rows = (
        candidate_query.filter(
            I12PitCandidate.path_mode == SPARSE_ZERO_FILL_MINUTE_PATH_MODE,
            I12PitCandidate.candidate_status == "passed",
        )
        .with_entities(
            I12PitCandidate.ticker,
            I12PitCandidate.decision_date,
            I12PitCandidate.decision_time_label,
            I12PitCandidate.source_bars_json,
        )
        .all()
    )
    sparse_pass_keys = {
        _candidate_attempt_comparison_key_from_values(
            ticker,
            decision_date,
            decision_time_label,
            source_bars_json,
        )
        for ticker, decision_date, decision_time_label, source_bars_json in sparse_pass_rows
    }
    if not sparse_pass_keys:
        return 0
    strict_partial_rows = (
        candidate_query.filter(
            I12PitCandidate.path_mode == STRICT_MINUTE_PATH_MODE,
            I12PitCandidate.coverage_status == "partial_minute_path",
        )
        .with_entities(
            I12PitCandidate.ticker,
            I12PitCandidate.decision_date,
            I12PitCandidate.decision_time_label,
            I12PitCandidate.source_bars_json,
        )
        .all()
    )
    return sum(
        1
        for ticker, decision_date, decision_time_label, source_bars_json in strict_partial_rows
        if _candidate_attempt_comparison_key_from_values(
            ticker,
            decision_date,
            decision_time_label,
            source_bars_json,
        )
        in sparse_pass_keys
    )


def _candidate_attempt_comparison_key(
    row: I12PitCandidate,
) -> tuple[str, date, str, str | None]:
    return _candidate_attempt_comparison_key_from_values(
        row.ticker,
        row.decision_date,
        row.decision_time_label,
        row.source_bars_json,
    )


def _candidate_attempt_comparison_key_from_values(
    ticker: str,
    decision_date: date,
    decision_time_label: str,
    source_bars_json: str | None,
) -> tuple[str, date, str, str | None]:
    return (
        ticker,
        decision_date,
        decision_time_label,
        _json_loads(source_bars_json).get("source_hur_identity_hash"),
    )


def _candidate_source_hur_identity_hash(row: I12PitCandidate) -> str | None:
    return _json_loads(row.source_bars_json).get("source_hur_identity_hash")


def _sparse_imputation_distributions(
    candidate_query: Any,
) -> dict[str, Any]:
    sparse_values = [
        _candidate_path_diagnostic_values(source_bars_json, feature_json)
        for source_bars_json, feature_json in candidate_query.filter(
            I12PitCandidate.path_mode == SPARSE_ZERO_FILL_MINUTE_PATH_MODE,
        )
        .with_entities(
            I12PitCandidate.source_bars_json,
            I12PitCandidate.feature_json,
        )
        .all()
    ]
    return {
        "candidate_count": len(sparse_values),
        "missing_minute_count": _numeric_summary(
            values.get("missing_minute_count_before_decision")
            for values in sparse_values
        ),
        "path_coverage_ratio": _numeric_summary(
            values.get("path_coverage_ratio") for values in sparse_values
        ),
        "last_observed_age_seconds": _numeric_summary(
            values.get("last_observed_before_decision_age_seconds")
            for values in sparse_values
        ),
        "observed_cumulative_volume": _numeric_summary(
            values.get("observed_cumulative_volume_before_decision")
            for values in sparse_values
        ),
        "projected_volume_ratio": _numeric_summary(
            values.get("zero_fill_projected_volume_ratio")
            for values in sparse_values
        ),
    }


def _path_mode_report_metrics(
    candidate_aggregates: _ReportCandidateAggregates,
    passed: Sequence[I12PitCandidate],
    quotes: Sequence[I12PitQuoteReplay],
    costs: Sequence[I12PitCostReplay],
    *,
    requested_path_modes: Sequence[str],
    expected_candidate_attempts: int | None,
    identity_audits_by_mode: Mapping[str, Mapping[str, Any]],
    source_denominator_known: bool,
    zero_hur_source_blocked: bool,
    child_evidence_parent_inactive: bool,
    child_evidence_parent_nonpassed: bool,
    compare_path_modes: bool,
) -> dict[str, Any]:
    modes = list(requested_path_modes)
    metrics: dict[str, Any] = {}
    for mode in modes:
        mode_candidate_count = candidate_aggregates.candidate_counts_by_path_mode.get(
            mode,
            0,
        )
        mode_passed = [row for row in passed if row.path_mode == mode]
        mode_candidate_ids = {row.i12_pit_candidate_id for row in mode_passed}
        mode_quotes = [row for row in quotes if row.i12_pit_candidate_id in mode_candidate_ids]
        mode_costs = [row for row in costs if row.i12_pit_candidate_id in mode_candidate_ids]
        quote_audit = _quote_completeness_audit(mode_passed, mode_quotes)
        cost_audit = _cost_completeness_audit(mode_passed, mode_costs)
        mode_not_run = (
            compare_path_modes
            and not mode_candidate_count
            and not source_denominator_known
        )
        mode_missing_attempts = (
            max(expected_candidate_attempts - mode_candidate_count, 0)
            if expected_candidate_attempts is not None
            else None
        )
        mode_extra_attempts = (
            max(mode_candidate_count - expected_candidate_attempts, 0)
            if expected_candidate_attempts is not None
            else None
        )
        coverage_counts = candidate_aggregates.coverage_status_by_path_mode.get(
            mode,
            {},
        )
        provider_errors = int(coverage_counts.get("daily_fetch_error") or 0) + int(
            coverage_counts.get("minute_fetch_error") or 0
        )
        identity_audit = dict(identity_audits_by_mode.get(mode, {}))
        identity_error = identity_audit.get("source_identity_denominator_error")
        identity_mismatch = (
            bool(identity_audit.get("source_identity_denominator_known"))
            and (
                int(identity_audit.get("missing_source_attempt_identity_count") or 0) > 0
                or int(identity_audit.get("extra_source_attempt_identity_count") or 0) > 0
            )
        )
        if mode_not_run:
            training_status = "not_run"
        elif child_evidence_parent_inactive:
            training_status = "blocked_child_evidence_parent_inactive"
        elif child_evidence_parent_nonpassed:
            training_status = "blocked_child_evidence_parent_nonpassed"
        elif identity_error:
            training_status = "blocked_source_identity_denominator_error"
        elif not source_denominator_known:
            training_status = "blocked_source_denominator_unknown"
        elif zero_hur_source_blocked:
            training_status = "blocked_zero_hur_source"
        elif source_denominator_known and not identity_audit.get(
            "source_identity_denominator_known",
            False,
        ):
            training_status = "blocked_source_identity_denominator_unknown"
        elif mode_missing_attempts not in {None, 0}:
            training_status = "blocked_source_replay_incomplete"
        elif mode_extra_attempts not in {None, 0}:
            training_status = "blocked_source_replay_extra_active_attempts"
        elif identity_mismatch:
            training_status = "blocked_source_identity_mismatch"
        elif provider_errors > 0:
            training_status = "blocked_source_provider_errors"
        elif not mode_passed:
            training_status = "blocked_zero_pit_candidates"
        elif quote_audit["unknown_quote_coverage_status_count"] > 0:
            training_status = "blocked_quote_replay_integrity"
        elif not quote_audit["quote_replay_complete"]:
            training_status = "blocked_quote_replay_incomplete"
        elif not cost_audit["cost_replay_complete"]:
            training_status = "blocked_cost_replay_incomplete"
        else:
            training_status = "eligible_for_retrain_evaluation"
        metrics[mode] = {
            "candidate_count": mode_candidate_count,
            "expected_candidate_attempts": expected_candidate_attempts,
            "missing_source_attempt_count": mode_missing_attempts,
            "extra_source_attempt_count": mode_extra_attempts,
            "source_identity_denominator_known": identity_audit.get(
                "source_identity_denominator_known",
                False,
            ),
            "source_identity_denominator_error": identity_error,
            "expected_source_attempt_identity_count": identity_audit.get(
                "expected_source_attempt_identity_count"
            ),
            "actual_source_attempt_identity_count": identity_audit.get(
                "actual_source_attempt_identity_count"
            ),
            "missing_source_attempt_identity_count": identity_audit.get(
                "missing_source_attempt_identity_count"
            ),
            "extra_source_attempt_identity_count": identity_audit.get(
                "extra_source_attempt_identity_count"
            ),
            "missing_source_attempt_identities_sample": identity_audit.get(
                "missing_source_attempt_identities_sample",
                [],
            ),
            "extra_source_attempt_identities_sample": identity_audit.get(
                "extra_source_attempt_identities_sample",
                [],
            ),
            "candidate_status_counts": (
                candidate_aggregates.candidate_status_counts_by_path_mode.get(mode, {})
            ),
            "coverage_status_counts": coverage_counts,
            "daily_source_hash_basis_counts": (
                candidate_aggregates.daily_source_hash_basis_counts_by_path_mode.get(
                    mode,
                    {},
                )
            ),
            "daily_source_hash_reuse_status_counts": (
                candidate_aggregates.daily_source_hash_reuse_status_counts_by_path_mode.get(
                    mode,
                    {},
                )
            ),
            "passed_candidate_count": len(mode_passed),
            "quote_replay_status": (
                "not_applicable"
                if mode_not_run or not mode_passed else (
                    "complete" if quote_audit["quote_replay_complete"] else "incomplete"
                )
            ),
            "quote_replay_complete": quote_audit["quote_replay_complete"],
            "unknown_quote_coverage_status_count": (
                quote_audit["unknown_quote_coverage_status_count"]
            ),
            "quote_ok_count": sum(
                1 for row in mode_quotes if row.coverage_status == "ok"
            ),
            "quote_non_ok_count": sum(
                1 for row in mode_quotes if row.coverage_status != "ok"
            ),
            "quote_usable_ok_count": quote_audit["usable_quote_role_count"],
            "cost_replay_status": (
                "not_applicable"
                if mode_not_run or not mode_passed else (
                    "complete" if cost_audit["cost_replay_complete"] else "incomplete"
                )
            ),
            "cost_replay_complete": cost_audit["cost_replay_complete"],
            "training_status": training_status,
            "conclusions_final": training_status == "eligible_for_retrain_evaluation",
            "exit_metrics": _exit_metrics_for_candidates(mode_passed, mode_costs),
        }
    return metrics


def _path_mode_source_attempts_complete(metrics: Mapping[str, Any]) -> bool:
    if not metrics:
        return False
    if metrics.get("training_status") == "not_run":
        return False
    if metrics.get("missing_source_attempt_count") not in {0, None}:
        return False
    if metrics.get("extra_source_attempt_count") not in {0, None}:
        return False
    coverage_counts = metrics.get("coverage_status_counts") or {}
    provider_errors = (
        int(coverage_counts.get("daily_fetch_error") or 0)
        + int(coverage_counts.get("minute_fetch_error") or 0)
    )
    if provider_errors:
        return False
    if metrics.get("source_identity_denominator_error"):
        return False
    if not metrics.get("source_identity_denominator_known"):
        return False
    if metrics.get("missing_source_attempt_identity_count") != 0:
        return False
    if metrics.get("extra_source_attempt_identity_count") != 0:
        return False
    return True


def _exit_metrics_for_candidates(
    passed: Sequence[I12PitCandidate],
    costs: Sequence[I12PitCostReplay],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for exit_role in EXIT_ROLES:
        rows = [row for row in costs if row.exit_role == exit_role]
        tradeable = [row for row in rows if row.tradeability_status == "tradeable"]
        modeled = [row.modeled_return for row in rows]
        out[exit_role] = {
            "candidates": len(passed),
            "row_count": len(rows),
            "tradeable_count": len(tradeable),
            "tradeable_rate": len(tradeable) / len(passed) if passed else None,
            "skipped_cash_count": len(rows) - len(tradeable),
            "skipped_cash_by_reason": dict(Counter(row.skipped_reason for row in rows)),
            "mean_modeled_return_skips_as_cash": _mean(modeled),
            "win_rate_skips_as_cash": (
                sum(1 for value in modeled if value > 0.0) / len(modeled)
                if modeled else None
            ),
        }
    return out


def _candidate_path_diagnostic_value(
    candidate: I12PitCandidate,
    key: str,
) -> float | None:
    return _candidate_path_diagnostic_values(
        candidate.source_bars_json,
        candidate.feature_json,
    ).get(key)


def _candidate_path_diagnostic_values(
    source_bars_json: str | None,
    feature_json: str | None,
) -> dict[str, float | None]:
    keys = (
        "missing_minute_count_before_decision",
        "path_coverage_ratio",
        "last_observed_before_decision_age_seconds",
        "observed_cumulative_volume_before_decision",
        "zero_fill_projected_volume_ratio",
    )
    return {
        key: _candidate_path_diagnostic_value_from_payloads(
            source_bars_json,
            feature_json,
            key,
        )
        for key in keys
    }


def _candidate_path_diagnostic_value_from_payloads(
    source_bars_json: str | None,
    feature_json: str | None,
    key: str,
) -> float | None:
    source_bars = _json_loads(source_bars_json)
    path_diagnostics = source_bars.get("path_diagnostics") or {}
    value = path_diagnostics.get(key)
    if value is None:
        value = _json_loads(feature_json).get(key)
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _prior_context(
    ticker: str,
    trading_date: date,
    daily_bars: Sequence[_DailyBar],
) -> dict[str, Any]:
    daily_by_date = {bar.date: bar for bar in daily_bars}
    if trading_date not in daily_by_date:
        raise RuntimeError("missing_day0_daily_open_bar")
    prior = [bar for bar in daily_bars if bar.date < trading_date]
    if len(prior) < MIN_PRIOR_DAILY_SESSIONS:
        raise RuntimeError("insufficient_prior_daily_history")
    prior20 = prior[-20:]
    prior252 = prior[-252:]
    prior_close = prior[-1].split_adjusted_close
    max_prior = max(bar.split_adjusted_close for bar in prior252)
    if prior_close <= 0 or max_prior <= 0:
        raise RuntimeError("invalid_prior_price_context")
    avg20_volume = sum(bar.volume for bar in prior20) / 20.0
    if avg20_volume <= 0:
        raise RuntimeError("invalid_prior_volume_context")
    adjusted_lows = [
        low for bar in prior252
        if (low := _split_adjusted_low(bar)) is not None
    ]
    low252 = min(adjusted_lows) if adjusted_lows else None
    return {
        "ticker": ticker,
        "prior_count": len(prior),
        "lookback_end": prior[-1].date.isoformat(),
        "prior_close": prior_close,
        "avg20_volume": avg20_volume,
        "max_prior_252_closes": max_prior,
        "distance_from_max252": prior_close / max_prior - 1.0,
        "off_low252": _safe_return(prior_close, low252) if low252 else None,
        "mom20": (
            prior_close / prior[-21].split_adjusted_close - 1.0
            if len(prior) >= 21 and prior[-21].split_adjusted_close > 0 else None
        ),
        "sigma20": _sigma(prior20),
        "prev_day_return": (
            prior[-1].split_adjusted_close / prior[-2].split_adjusted_close - 1.0
            if len(prior) >= 2 and prior[-2].split_adjusted_close > 0 else None
        ),
        "prev_day_green": prior[-1].close > prior[-1].open,
    }


def _candidate_error(
    *,
    ticker: str,
    trading_date: date,
    decision_ts: datetime,
    decision_time_label: str,
    coverage_status: str,
    fail_reason: str,
    daily_source_hash: str | None,
    daily_source_hash_basis: str = "unknown",
    daily_provider_fetch_hash: str | None = None,
    daily_source_hash_reuse_status: str = DAILY_SOURCE_HASH_REUSE_FRESH,
    minute_source_hash: str | None = None,
    daily_error: dict[str, Any] | None = None,
    minute_error: dict[str, Any] | None = None,
    prior_ctx: Mapping[str, Any] | None = None,
    source_hur_identity_hash: str | None = None,
    source_hur: Mapping[str, Any] | None = None,
    path_mode: str = STRICT_MINUTE_PATH_MODE,
    path_diagnostics: Mapping[str, Any] | None = None,
) -> PitCandidateResult:
    path_mode = _validate_minute_path_mode(path_mode)
    path_diagnostics = dict(path_diagnostics or {})
    feature_asof_value = _parse_optional_datetime(path_diagnostics.get("feature_asof_ts"))
    feature_json = {
        "feature_manifest_version": FEATURE_MANIFEST_VERSION,
        "reconstruction_method": REBUILD_METHOD,
        "path_mode": path_mode,
        "pattern_id": I12_PATTERN_ID,
        "ticker": ticker,
        "decision_date": trading_date.isoformat(),
        "decision_ts": decision_ts.isoformat(),
        "feature_asof_ts": None,
        "completed_through_ts": None,
        "source_minute_bars_max_start_ts": None,
        "coverage_status": coverage_status,
        "missing_minute_bars": coverage_status == "missing_minute_bars",
    }
    if path_diagnostics:
        feature_json.update({
            "feature_asof_ts": path_diagnostics.get("feature_asof_ts"),
            "completed_through_ts": path_diagnostics.get("completed_through_ts"),
            "source_minute_bars_max_start_ts": path_diagnostics.get(
                "source_minute_bars_max_start_ts"
            ),
            "expected_minute_count_before_decision": path_diagnostics.get(
                "expected_minute_count_before_decision"
            ),
            "observed_minute_count_before_decision": path_diagnostics.get(
                "observed_minute_count_before_decision"
            ),
            "missing_minute_count_before_decision": path_diagnostics.get(
                "missing_minute_count_before_decision"
            ),
            "path_coverage_ratio": path_diagnostics.get("path_coverage_ratio"),
            "opening_bar_present": path_diagnostics.get("opening_bar_present"),
            "first_observed_minute_ts": path_diagnostics.get("first_observed_minute_ts"),
            "last_observed_minute_ts": path_diagnostics.get("last_observed_minute_ts"),
            "last_observed_before_decision_age_seconds": path_diagnostics.get(
                "last_observed_before_decision_age_seconds"
            ),
            "observed_cumulative_volume_before_decision": path_diagnostics.get(
                "observed_cumulative_volume_before_decision"
            ),
            "observed_open_to_decision_return": path_diagnostics.get(
                "observed_open_to_decision_return"
            ),
            "zero_fill_projected_volume_ratio": path_diagnostics.get(
                "zero_fill_projected_volume_ratio"
            ),
        })
    if prior_ctx:
        feature_json.update({
            "lookback_end": prior_ctx.get("lookback_end"),
            "prior_close": prior_ctx.get("prior_close"),
            "distance_from_max252": prior_ctx.get("distance_from_max252"),
        })
    leakage_guard = {
        "decision_ts": decision_ts.isoformat(),
        "entry_quote_target_ts": decision_ts.isoformat(),
        "feature_asof_ts": None,
        "completed_through_ts": None,
        "source_minute_bars_max_start_ts": None,
        "uses_full_day_volume": False,
        "uses_same_day_close": False,
        "uses_full_day_high_low": False,
        "uses_forward_bars": False,
        "decision_time_semantics": (
            "decision_after_prior_completed_minute_start_stamped_bars"
        ),
        "path_mode": path_mode,
        "predictor_time_basis": "failed_before_feature_materialization",
    }
    if path_diagnostics:
        leakage_guard.update({
            "feature_asof_ts": path_diagnostics.get("feature_asof_ts"),
            "completed_through_ts": path_diagnostics.get("completed_through_ts"),
            "source_minute_bars_max_start_ts": path_diagnostics.get(
                "source_minute_bars_max_start_ts"
            ),
        })
    gate_values = {
        "candidate_passed": False,
        "fail_reasons": [fail_reason],
        "path_mode": path_mode,
    }
    source_bars = {
        "daily_source_hash": daily_source_hash,
        "daily_source_hash_basis": daily_source_hash_basis,
        "daily_provider_fetch_hash": daily_provider_fetch_hash,
        "daily_source_hash_reuse_status": daily_source_hash_reuse_status,
        "minute_source_hash": minute_source_hash,
        "source_hur_identity_hash": source_hur_identity_hash,
        "source_hur": dict(source_hur or {}),
        "daily_error": daily_error,
        "minute_error": minute_error,
        "path_mode": path_mode,
        "path_diagnostics": path_diagnostics,
    }
    source_errors = {
        key: value
        for key, value in {
            "daily_error": daily_error,
            "minute_error": minute_error,
        }.items()
        if value is not None
    }
    error_json = {"source_errors": source_errors} if source_errors else None
    input_hash = stable_hash({
        "ticker": ticker,
        "decision_date": trading_date.isoformat(),
        "decision_ts": decision_ts.isoformat(),
        "coverage_status": coverage_status,
        "source_bars": _source_bars_for_candidate_identity(source_bars),
        "path_mode": path_mode,
    })
    candidate_attempt_hash = _candidate_attempt_hash(
        ticker=ticker,
        trading_date=trading_date,
        decision_ts=decision_ts,
        decision_time_label=decision_time_label,
        source_hur_identity_hash=source_hur_identity_hash,
        path_mode=path_mode,
    )
    candidate_identity_hash = stable_hash({
        "input_hash": input_hash,
        "feature_json": feature_json,
        "gate_values": gate_values,
        "leakage_guard": leakage_guard,
    })
    label_hash = stable_hash({})
    return PitCandidateResult(
        ticker=ticker,
        decision_date=trading_date,
        decision_ts=decision_ts,
        decision_time_label=decision_time_label,
        path_mode=path_mode,
        candidate_status="failed",
        coverage_status=coverage_status,
        fail_reason=fail_reason,
        feature_json=feature_json,
        gate_values=gate_values,
        leakage_guard=leakage_guard,
        source_bars=source_bars,
        label_json={},
        feature_asof_ts=feature_asof_value,
        candidate_attempt_hash=candidate_attempt_hash,
        input_hash=input_hash,
        candidate_identity_hash=candidate_identity_hash,
        label_hash=label_hash,
        content_hash=candidate_identity_hash,
        error_json=error_json,
    )


def _label_payload(
    *,
    trading_date: date,
    daily_bars: Sequence[_DailyBar],
    minute_bars: Sequence[_MinuteBar],
    entry_price: float,
) -> dict[str, Any]:
    exit_target = _eastern_timestamp(trading_date, SAME_DAY_EXIT_TIME)
    same_day_bar = _latest_minute_at_or_before(minute_bars, exit_target)
    next_session = next_us_equity_session(trading_date + timedelta(days=1))
    daily_by_date = {bar.date: bar for bar in daily_bars}
    next_open = daily_by_date.get(next_session).open if next_session in daily_by_date else None
    return {
        "same_day_exit_role": "same_day_exit",
        "same_day_exit_ts": exit_target.isoformat(),
        "same_day_exit_price_proxy": same_day_bar.close if same_day_bar else None,
        "same_day_return_bar_proxy": (
            _safe_return(same_day_bar.close, entry_price) if same_day_bar else None
        ),
        "next_open_exit_role": "next_open_exit",
        "next_open_exit_date": next_session.isoformat(),
        "next_open_price_proxy": next_open,
        "next_open_return_bar_proxy": _safe_return(next_open, entry_price),
        "label_time_basis": "outcome_only_not_predictor",
    }


def _minute_path_coverage_error(
    *,
    trading_date: date,
    decision_ts: datetime,
    minute_bars: Sequence[_MinuteBar],
) -> tuple[str, str] | None:
    session_open = us_equity_session_open_timestamp(trading_date)
    session_close = _eastern_timestamp(
        trading_date,
        us_equity_session_close_time(trading_date),
    )
    decision_utc = _aware_utc(decision_ts)
    if decision_utc < session_open or decision_utc >= session_close:
        return "invalid_decision_time", "decision_ts_outside_regular_session"
    normalized = sorted({_minute_floor_utc(bar.timestamp) for bar in minute_bars})
    if not normalized:
        return "missing_minute_bars", "missing_minute_bars_at_or_before_decision"
    if normalized[0] != session_open:
        return "missing_open_bar", "missing_open_bar"
    expected = _expected_minute_timestamps_before_decision(trading_date, decision_utc)
    missing = [ts for ts in expected if ts not in normalized]
    if missing:
        return "partial_minute_path", f"missing_{len(missing)}_opening_path_minutes"
    return None


def _duplicate_minute_start_timestamps(
    minute_bars: Sequence[_MinuteBar],
) -> list[datetime]:
    counts = Counter(_minute_floor_utc(bar.timestamp) for bar in minute_bars)
    return sorted(ts for ts, count in counts.items() if count > 1)


def _expected_minute_count_before_decision(
    trading_date: date,
    decision_ts: datetime,
) -> int:
    return len(_expected_minute_timestamps_before_decision(trading_date, decision_ts))


def _completed_minutes_before_decision(
    trading_date: date,
    decision_ts: datetime,
) -> float:
    session_open = us_equity_session_open_timestamp(trading_date)
    decision_utc = _minute_floor_utc(_aware_utc(decision_ts))
    return max((decision_utc - session_open).total_seconds() / 60.0, 0.0)


def _expected_minute_timestamps_before_decision(
    trading_date: date,
    decision_ts: datetime,
) -> list[datetime]:
    session_open = us_equity_session_open_timestamp(trading_date)
    decision_utc = _minute_floor_utc(_aware_utc(decision_ts))
    count = int(_completed_minutes_before_decision(trading_date, decision_utc))
    return [session_open + timedelta(minutes=idx) for idx in range(max(count, 0))]


def _session_minutes(trading_date: date) -> int:
    close_time = us_equity_session_close_time(trading_date)
    open_minutes = 9 * 60 + 30
    close_minutes = close_time.hour * 60 + close_time.minute
    return close_minutes - open_minutes


def _minute_floor_utc(value: datetime) -> datetime:
    return _aware_utc(value).replace(second=0, microsecond=0)


def _latest_minute_at_or_before(
    minute_bars: Sequence[_MinuteBar],
    target_ts: datetime,
) -> _MinuteBar | None:
    eligible = [bar for bar in minute_bars if bar.timestamp <= target_ts]
    return eligible[-1] if eligible else None


def _quote_window(
    role: str,
    target_ts: datetime,
    before_seconds: float,
    after_seconds: float,
) -> QuoteWindow:
    target_utc = _coerce_persisted_utc(target_ts)
    return QuoteWindow(
        quote_role=role,
        target_ts=target_utc,
        window_start_ts=target_utc - timedelta(seconds=before_seconds),
        window_end_ts=target_utc + timedelta(seconds=after_seconds),
    )


def _quote_result(
    *,
    window: QuoteWindow,
    quote: AlpacaQuote | None,
    feed: str,
    coverage_status: str,
    error_json: dict[str, Any] | None,
) -> QuoteReplayResult:
    return QuoteReplayResult(
        quote_role=window.quote_role,
        target_ts=window.target_ts,
        window_start_ts=window.window_start_ts,
        window_end_ts=window.window_end_ts,
        quote=quote,
        quote_ts=None,
        quote_age_seconds=None,
        bid=None,
        ask=None,
        bid_size=None,
        ask_size=None,
        spread_bps=None,
        top_of_book_notional=None,
        bid_notional=None,
        ask_notional=None,
        executable_notional=None,
        executable_side=None,
        feed=feed,
        source="alpaca_historical_quotes",
        coverage_status=coverage_status,
        raw_json=None,
        error_json=error_json,
    )


def _latest_quote_at_or_before(
    quotes: Sequence[AlpacaQuote],
    target_ts: datetime,
) -> AlpacaQuote | None:
    parsed = [
        (quote_ts, quote)
        for quote in quotes
        if (quote_ts := _parse_quote_ts(quote)) is not None and quote_ts <= target_ts
    ]
    if not parsed:
        return None
    return max(parsed, key=lambda item: item[0])[1]


def _parse_quote_ts(quote: AlpacaQuote) -> datetime | None:
    raw = quote.timestamp
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return _aware_utc(parsed)


def _entry_skip_reason(
    quote: I12PitQuoteReplay | None,
    *,
    intended_order_usd: float,
    max_spread_bps: float,
) -> str:
    if quote is None:
        return "entry_quote_missing"
    if quote.coverage_status == "missing":
        return "entry_quote_missing"
    if quote.coverage_status == "stale":
        return "entry_quote_stale"
    if quote.coverage_status != "ok":
        return "entry_quote_error"
    raw = _json_loads(quote.raw_json)
    conditions = raw.get("c") or raw.get("conditions") or []
    if any(str(item).upper() in HALT_CONDITIONS for item in conditions):
        return "halt_or_condition_uncertain"
    if quote.bid is None or quote.ask is None or quote.bid <= 0 or quote.ask <= 0:
        return "entry_quote_invalid"
    if quote.ask < quote.bid:
        return "entry_quote_invalid"
    if quote.spread_bps is None or quote.spread_bps > max_spread_bps:
        return "spread"
    notional = quote.ask_notional if quote.ask_notional is not None else quote.top_of_book_notional
    if notional is None or notional < intended_order_usd:
        return "size"
    return "none"


def _exit_skip_reason(
    quote: I12PitQuoteReplay | None,
    *,
    intended_order_usd: float,
    max_spread_bps: float,
) -> str:
    if quote is None:
        return "exit_quote_missing"
    if quote.coverage_status == "missing":
        return "exit_quote_missing"
    if quote.coverage_status == "stale":
        return "exit_quote_stale"
    if quote.coverage_status != "ok":
        return "exit_quote_error"
    raw = _json_loads(quote.raw_json)
    conditions = raw.get("c") or raw.get("conditions") or []
    if any(str(item).upper() in HALT_CONDITIONS for item in conditions):
        return "halt_or_condition_uncertain"
    if quote.bid is None or quote.ask is None or quote.bid <= 0 or quote.ask <= 0:
        return "exit_quote_invalid"
    if quote.ask < quote.bid:
        return "exit_quote_invalid"
    if quote.spread_bps is None or quote.spread_bps > max_spread_bps:
        return "spread"
    notional = quote.bid_notional if quote.bid_notional is not None else quote.top_of_book_notional
    if notional is None or notional < intended_order_usd:
        return "size"
    return "none"


def _volume_tradeability_report(
    *,
    passed_candidates: Sequence[I12PitCandidate],
    quotes: Sequence[I12PitQuoteReplay],
    costs: Sequence[I12PitCostReplay],
    default_threshold: float,
    evidence_getter: Callable[[I12PitCandidate | None, I12PitQuoteReplay | None], dict[str, Any]],
    window_prefix: str,
    evidence_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_by_id = {
        row.i12_pit_candidate_id: row for row in passed_candidates
    }
    quotes_by_id = {row.i12_pit_quote_replay_id: row for row in quotes}
    quotes_by_candidate_role: dict[tuple[str, str], list[I12PitQuoteReplay]] = defaultdict(list)
    for quote in quotes:
        quotes_by_candidate_role[
            (quote.i12_pit_candidate_id, quote.quote_role)
        ].append(quote)

    metrics = {
        role: _volume_tradeability_metrics_for_role(
            role,
            passed_candidates,
            [row for row in costs if row.exit_role == role],
            candidate_by_id,
            quotes_by_id,
            quotes_by_candidate_role,
            default_threshold,
            evidence_getter,
            window_prefix,
        )
        for role in EXIT_ROLES
    }
    sensitivity = {
        _volume_threshold_label(threshold): {
            role: _volume_tradeability_metrics_for_role(
                role,
                passed_candidates,
                [row for row in costs if row.exit_role == role],
                candidate_by_id,
                quotes_by_id,
                quotes_by_candidate_role,
                threshold,
                evidence_getter,
                window_prefix,
            )
            for role in EXIT_ROLES
        }
        for threshold in VOLUME_PARTICIPATION_THRESHOLDS
    }
    status_counts: Counter[str] = Counter()
    skip_counts: Counter[str] = Counter()
    tradeable = 0
    total = 0
    evidence_missing = 0
    basis_counts: Counter[str] = Counter()
    denominator_basis_counts: Counter[str] = Counter()
    price_basis_counts: Counter[str] = Counter()
    for role_metrics in metrics.values():
        status_counts.update(role_metrics["volume_tradeability_status_counts"])
        skip_counts.update(role_metrics["volume_tradeability_skip_reason_counts"])
        tradeable += int(role_metrics["tradeable_volume_count"])
        total += int(role_metrics["row_count"])
        evidence_missing += int(role_metrics["volume_evidence_missing_count"])
        basis_counts.update(role_metrics[f"{window_prefix}_basis_counts"])
        denominator_basis_counts.update(
            role_metrics[f"{window_prefix}_denominator_basis_counts"]
        )
        price_basis_counts.update(role_metrics[f"{window_prefix}_price_basis_counts"])
    return {
        "metrics": metrics,
        "sensitivity": sensitivity,
        "status_counts": dict(status_counts),
        "skip_reason_counts": dict(skip_counts),
        "tradeability_rate": tradeable / total if total else None,
        "evidence": {
            **dict(evidence_metadata),
            "default_participation_threshold": default_threshold,
            "volume_evidence_missing_count": evidence_missing,
            f"{window_prefix}_basis_counts": dict(basis_counts),
            f"{window_prefix}_denominator_basis_counts": dict(denominator_basis_counts),
            f"{window_prefix}_price_basis_counts": dict(price_basis_counts),
        },
    }


def _volume_tradeability_metrics_for_role(
    exit_role: str,
    passed_candidates: Sequence[I12PitCandidate],
    rows: Sequence[I12PitCostReplay],
    candidate_by_id: Mapping[str, I12PitCandidate],
    quotes_by_id: Mapping[str, I12PitQuoteReplay],
    quotes_by_candidate_role: Mapping[tuple[str, str], Sequence[I12PitQuoteReplay]],
    threshold: float,
    evidence_getter: Callable[[I12PitCandidate | None, I12PitQuoteReplay | None], dict[str, Any]],
    window_prefix: str,
) -> dict[str, Any]:
    modeled_returns: list[float] = []
    tradeable_returns: list[float] = []
    participation_rates: list[float] = []
    share_volumes: list[float] = []
    share_participation_rates: list[float] = []
    dollar_volumes: list[float] = []
    status_counts: Counter[str] = Counter()
    skip_counts: Counter[str] = Counter()
    basis_counts: Counter[str] = Counter()
    denominator_basis_counts: Counter[str] = Counter()
    price_basis_counts: Counter[str] = Counter()
    evidence_missing = 0
    tradeable = 0

    for cost in rows:
        candidate = candidate_by_id.get(cost.i12_pit_candidate_id)
        entry_quote = _quote_for_cost_role(
            cost,
            "entry",
            quotes_by_id,
            quotes_by_candidate_role,
        )
        exit_quote = _quote_for_cost_role(
            cost,
            exit_role,
            quotes_by_id,
            quotes_by_candidate_role,
        )
        result = _volume_tradeability_for_cost(
            candidate=candidate,
            entry_quote=entry_quote,
            exit_quote=exit_quote,
            cost=cost,
            threshold=threshold,
            evidence_getter=evidence_getter,
        )
        status_counts[result["volume_tradeability_status"]] += 1
        if result["volume_tradeability_status"] == VOLUME_TRADEABILITY_OK_STATUS:
            tradeable += 1
            if result["modeled_return"] is not None:
                tradeable_returns.append(float(result["modeled_return"]))
        else:
            skip_counts[str(result["volume_skipped_reason"])] += 1
        modeled_returns.append(float(result["modeled_return"] or 0.0))
        if result["entry_window_dollar_volume"] is not None:
            dollar_volumes.append(float(result["entry_window_dollar_volume"]))
        if result["intended_order_participation_rate"] is not None:
            participation_rates.append(float(result["intended_order_participation_rate"]))
        if result["entry_window_share_volume"] is not None:
            share_volumes.append(float(result["entry_window_share_volume"]))
        if result["intended_order_share_participation_rate"] is not None:
            share_participation_rates.append(
                float(result["intended_order_share_participation_rate"])
            )
        if result["volume_evidence_available"] is not True:
            evidence_missing += 1
        basis_counts[str(result["window_basis"] or "unknown")] += 1
        denominator_basis_counts[str(result["denominator_basis"] or "unknown")] += 1
        price_basis_counts[str(result["window_price_basis"] or "unknown")] += 1

    dollar_summary = _numeric_summary(dollar_volumes)
    participation_summary = _numeric_summary(participation_rates)
    share_summary = _numeric_summary(share_volumes)
    share_participation_summary = _numeric_summary(share_participation_rates)

    return {
        "candidates": len(passed_candidates),
        "row_count": len(rows),
        "participation_threshold": threshold,
        "tradeable_volume_count": tradeable,
        "volume_tradeability_rate": tradeable / len(passed_candidates)
        if passed_candidates else None,
        "skipped_cash_count": len(rows) - tradeable,
        "volume_tradeability_status_counts": dict(status_counts),
        "volume_tradeability_skip_reason_counts": dict(skip_counts),
        "mean_modeled_return_volume_skips_as_cash": (
            sum(modeled_returns) / len(modeled_returns) if modeled_returns else None
        ),
        "win_rate_volume_skips_as_cash": (
            sum(1 for value in modeled_returns if value > 0.0) / len(modeled_returns)
            if modeled_returns else None
        ),
        "mean_modeled_return_volume_tradeable": _mean(tradeable_returns),
        "entry_window_dollar_volume": dollar_summary,
        "intended_order_participation_rate": participation_summary,
        "entry_window_share_volume": share_summary,
        "intended_order_share_participation_rate": share_participation_summary,
        f"{window_prefix}_dollar_volume": dollar_summary,
        f"{window_prefix}_share_volume": share_summary,
        f"{window_prefix}_share_participation_rate": share_participation_summary,
        "volume_evidence_missing_count": evidence_missing,
        f"{window_prefix}_basis_counts": dict(basis_counts),
        f"{window_prefix}_denominator_basis_counts": dict(denominator_basis_counts),
        f"{window_prefix}_price_basis_counts": dict(price_basis_counts),
        f"{window_prefix}_basis": next(iter(basis_counts), None),
        f"{window_prefix}_denominator_basis": next(
            iter(denominator_basis_counts), None
        ),
        f"{window_prefix}_price_basis": next(iter(price_basis_counts), None),
        "entry_volume_window_basis": next(iter(basis_counts), None),
    }


def _quote_for_cost_role(
    cost: I12PitCostReplay,
    role: str,
    quotes_by_id: Mapping[str, I12PitQuoteReplay],
    quotes_by_candidate_role: Mapping[tuple[str, str], Sequence[I12PitQuoteReplay]],
) -> I12PitQuoteReplay | None:
    quote_id = (
        cost.entry_quote_replay_id if role == "entry" else cost.exit_quote_replay_id
    )
    if quote_id:
        quote = quotes_by_id.get(quote_id)
        if quote is not None:
            return quote
    candidates = list(quotes_by_candidate_role.get((cost.i12_pit_candidate_id, role), ()))
    return candidates[0] if len(candidates) == 1 else None


def _volume_tradeability_for_cost(
    *,
    candidate: I12PitCandidate | None,
    entry_quote: I12PitQuoteReplay | None,
    exit_quote: I12PitQuoteReplay | None,
    cost: I12PitCostReplay,
    threshold: float,
    evidence_getter: Callable[[I12PitCandidate | None, I12PitQuoteReplay | None], dict[str, Any]],
) -> dict[str, Any]:
    evidence = evidence_getter(candidate, entry_quote)
    entry_reason = _quote_quality_skip_reason(
        entry_quote,
        missing_reason="entry_quote_missing",
        stale_reason="entry_quote_stale",
        error_reason="halt_or_bad_quote",
        max_spread_bps=cost.max_spread_bps,
    )
    participation_rate = None
    share_participation_rate = None
    observed_share_volume = evidence["entry_window_share_volume_denominator"]
    if entry_quote is not None and entry_quote.ask and entry_quote.ask > 0:
        if evidence["entry_window_dollar_volume"]:
            participation_rate = (
                cost.intended_order_usd / evidence["entry_window_dollar_volume"]
            )
        if observed_share_volume:
            intended_shares = cost.intended_order_usd / entry_quote.ask
            share_participation_rate = intended_shares / observed_share_volume
    if entry_reason == "none":
        if evidence["volume_evidence_available"] is not True:
            skipped_reason = "volume_missing"
        elif participation_rate is None:
            skipped_reason = "volume_missing"
        elif participation_rate > threshold:
            skipped_reason = "volume_too_thin"
        else:
            skipped_reason = _quote_quality_skip_reason(
                exit_quote,
                missing_reason=f"{cost.exit_role}_quote_missing",
                stale_reason=f"{cost.exit_role}_quote_stale",
                error_reason="halt_or_bad_quote",
                max_spread_bps=cost.max_spread_bps,
            )
    else:
        skipped_reason = entry_reason

    quote_cost_return = None
    slippage_return = None
    modeled_return = 0.0
    status = VOLUME_TRADEABILITY_SKIP_STATUS
    if skipped_reason == "none" and entry_quote is not None and exit_quote is not None:
        assert entry_quote.ask is not None and exit_quote.bid is not None
        quote_cost_return = exit_quote.bid / entry_quote.ask - 1.0
        slip = cost.slippage_bps / 10000.0
        slip_entry = entry_quote.ask * (1.0 + slip)
        slip_exit = exit_quote.bid * (1.0 - slip)
        slippage_return = slip_exit / slip_entry - 1.0
        modeled_return = slippage_return
        status = VOLUME_TRADEABILITY_OK_STATUS

    return {
        "volume_tradeability_status": status,
        "volume_skipped_reason": skipped_reason,
        "entry_window_share_volume": observed_share_volume,
        "intended_order_share_participation_rate": share_participation_rate,
        "entry_window_dollar_volume": evidence["entry_window_dollar_volume"],
        "intended_order_usd": cost.intended_order_usd,
        "intended_order_participation_rate": participation_rate,
        "quote_cost_return": quote_cost_return,
        "slippage_return": slippage_return,
        "modeled_return": modeled_return,
        "volume_evidence_available": evidence["volume_evidence_available"],
        "window_basis": evidence["window_basis"],
        "window_price_basis": evidence.get("window_price_basis"),
        "denominator_basis": evidence.get("denominator_basis"),
    }


def _predecision_volume_evidence(
    candidate: I12PitCandidate | None,
    entry_quote: I12PitQuoteReplay | None,
) -> dict[str, Any]:
    if candidate is None:
        return _missing_entry_volume_evidence()
    features = _json_loads(candidate.feature_json)
    source_bars = _json_loads(candidate.source_bars_json)
    observed_volume = _finite_float(features.get("observed_cumulative_volume_before_decision"))
    early_volume = _finite_float(features.get("early_cumulative_volume"))
    if observed_volume is not None:
        share_volume = observed_volume
        denominator_basis = "observed_cumulative_volume_before_decision"
    elif early_volume is not None:
        share_volume = early_volume
        denominator_basis = "early_cumulative_volume_fallback"
    else:
        share_volume = None
        denominator_basis = "missing"
    if share_volume is not None and share_volume > 0:
        unsafe_reason = _predecision_volume_timestamp_violation(
            candidate,
            features=features,
            source_bars=source_bars,
        )
        if unsafe_reason is not None:
            return _missing_entry_volume_evidence(
                window_basis="unsafe_predecision_timestamp",
                denominator_basis=unsafe_reason,
            )
    prior_close = _finite_float(features.get("prior_close"))
    gap = _finite_float(features.get("gap"))
    early_return = _finite_float(features.get("early_return"))
    price = None
    price_basis = "missing"
    if (
        entry_quote is not None
        and entry_quote.bid is not None
        and entry_quote.ask is not None
        and entry_quote.bid > 0
        and entry_quote.ask > 0
        and entry_quote.ask >= entry_quote.bid
    ):
        price = (entry_quote.bid + entry_quote.ask) / 2.0
        price_basis = "entry_quote_mid"
    if (
        price is None
        and prior_close is not None
        and prior_close > 0
        and gap is not None
        and early_return is not None
    ):
        day_open = prior_close * (1.0 + gap)
        price = day_open * (1.0 + early_return)
        price_basis = VOLUME_PRICE_BASIS_LAST_PRE_DECISION
    dollar_volume = (
        share_volume * price
        if share_volume is not None
        and share_volume > 0
        and price is not None
        and price > 0
        else None
    )
    return {
        "volume_evidence_available": dollar_volume is not None,
        "entry_window_dollar_volume": dollar_volume,
        "entry_window_share_volume_denominator": (
            share_volume if share_volume is not None and share_volume > 0 else None
        ),
        "window_basis": VOLUME_WINDOW_BASIS_PRE_DECISION,
        "window_price_basis": price_basis,
        "denominator_basis": denominator_basis,
        "source_minute_bars_max_start_ts": (
            source_bars.get("source_minute_bars_max_start_ts")
        ),
        "completed_through_ts": (
            features.get("completed_through_ts") or source_bars.get("completed_through_ts")
        ),
    }


def _missing_entry_volume_evidence(
    *,
    window_basis: str = "missing",
    denominator_basis: str = "missing",
) -> dict[str, Any]:
    return {
        "volume_evidence_available": False,
        "entry_window_dollar_volume": None,
        "entry_window_share_volume_denominator": None,
        "window_basis": window_basis,
        "window_price_basis": None,
        "denominator_basis": denominator_basis,
        "source_minute_bars_max_start_ts": None,
        "completed_through_ts": None,
    }


def _predecision_volume_timestamp_violation(
    candidate: I12PitCandidate,
    *,
    features: Mapping[str, Any],
    source_bars: Mapping[str, Any],
) -> str | None:
    decision_ts = _coerce_persisted_utc(candidate.decision_ts)
    source_max_ts, source_max_error = _select_predecision_timestamp_proof(
        features,
        source_bars,
        "source_minute_bars_max_start_ts",
    )
    if source_max_error is not None:
        return source_max_error
    if source_max_ts is not None and source_max_ts >= decision_ts:
        return "source_minute_bars_max_start_ts_at_or_after_decision_ts"
    completed_through_ts, completed_error = _select_predecision_timestamp_proof(
        features,
        source_bars,
        "completed_through_ts",
    )
    if completed_error is not None:
        return completed_error
    if completed_through_ts is not None and completed_through_ts > decision_ts:
        return "completed_through_ts_after_decision_ts"
    return None


def _select_predecision_timestamp_proof(
    features: Mapping[str, Any],
    source_bars: Mapping[str, Any],
    key: str,
) -> tuple[datetime | None, str | None]:
    saw_value = False
    for mapping in (features, source_bars):
        value = mapping.get(key)
        if _timestamp_value_missing(value):
            continue
        saw_value = True
        parsed = _parse_timestamp_candidate(value)
        if parsed is not None:
            return parsed, None
    if not saw_value:
        return None, "missing_predecision_timestamp_proof"
    return None, "malformed_predecision_timestamp_proof"


def _timestamp_value_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _parse_timestamp_candidate(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _coerce_persisted_utc(value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return _coerce_persisted_utc(parsed)


def _execution_window_volume_evidence(
    candidate: I12PitCandidate | None,
    entry_quote: I12PitQuoteReplay | None,
) -> dict[str, Any]:
    del entry_quote
    if candidate is None:
        return _missing_execution_window_volume_evidence()
    features = _json_loads(candidate.feature_json)
    source_bars = _json_loads(candidate.source_bars_json)
    payload = _first_mapping(
        source_bars.get("execution_window_volume_evidence"),
        source_bars.get("execution_window_volume"),
        features.get("execution_window_volume_evidence"),
        features.get("execution_window_volume"),
    )
    if not payload:
        return _missing_execution_window_volume_evidence()

    dollar_volume = _finite_float(
        payload.get("execution_window_dollar_volume"),
        payload.get("dollar_volume"),
    )
    share_volume = _finite_float(
        payload.get("execution_window_share_volume"),
        payload.get("share_volume"),
        payload.get("volume"),
    )
    if dollar_volume is None:
        price = _finite_float(
            payload.get("execution_window_vwap"),
            payload.get("vwap"),
            payload.get("close"),
        )
        if (
            share_volume is not None
            and share_volume > 0
            and price is not None
            and price > 0
        ):
            dollar_volume = share_volume * price
    if share_volume is None and dollar_volume is not None:
        price = _finite_float(
            payload.get("execution_window_vwap"),
            payload.get("vwap"),
            payload.get("close"),
        )
        if price is not None and price > 0:
            share_volume = dollar_volume / price

    return {
        "volume_evidence_available": (
            dollar_volume is not None and dollar_volume > 0
        ),
        "entry_window_dollar_volume": (
            dollar_volume if dollar_volume is not None and dollar_volume > 0 else None
        ),
        "entry_window_share_volume_denominator": (
            share_volume if share_volume is not None and share_volume > 0 else None
        ),
        "window_basis": str(
            payload.get("execution_window_basis")
            or payload.get("basis")
            or "persisted_execution_window_minute_bars"
        ),
        "window_price_basis": payload.get("price_basis") or payload.get("vwap_basis"),
        "denominator_basis": str(
            payload.get("denominator_basis")
            or "persisted_execution_window_volume"
        ),
        "execution_window_start_ts": payload.get("execution_window_start_ts"),
        "execution_window_end_ts": payload.get("execution_window_end_ts"),
        "execution_window_minutes": payload.get("execution_window_minutes"),
    }


def _missing_execution_window_volume_evidence() -> dict[str, Any]:
    return {
        "volume_evidence_available": False,
        "entry_window_dollar_volume": None,
        "entry_window_share_volume_denominator": None,
        "window_basis": "missing_execution_window_volume",
        "window_price_basis": None,
        "denominator_basis": "missing",
        "execution_window_start_ts": None,
        "execution_window_end_ts": None,
        "execution_window_minutes": None,
    }


def _first_mapping(*values: Any) -> Mapping[str, Any] | None:
    for value in values:
        if isinstance(value, Mapping):
            return value
    return None


def _quote_quality_skip_reason(
    quote: I12PitQuoteReplay | None,
    *,
    missing_reason: str,
    stale_reason: str,
    error_reason: str,
    max_spread_bps: float,
) -> str:
    if quote is None:
        return missing_reason
    if quote.coverage_status == "missing":
        return missing_reason
    if quote.coverage_status == "stale":
        return stale_reason
    if quote.coverage_status != "ok":
        return error_reason
    raw = _json_loads(quote.raw_json)
    conditions = raw.get("c") or raw.get("conditions") or []
    if any(str(item).upper() in HALT_CONDITIONS for item in conditions):
        return "halt_or_bad_quote"
    if quote.bid is None or quote.ask is None or quote.bid <= 0 or quote.ask <= 0:
        return "halt_or_bad_quote"
    if quote.ask < quote.bid:
        return "halt_or_bad_quote"
    if quote.spread_bps is None or quote.spread_bps > max_spread_bps:
        return "spread"
    return "none"


def _finite_float(*values: Any) -> float | None:
    for value in values:
        try:
            if value is None:
                continue
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _volume_threshold_label(threshold: float) -> str:
    pct = threshold * 100.0
    text = f"{pct:g}".replace(".", "_")
    return f"{text}pct"


def _provider_error_payload(resp: AdapterResponse[Any]) -> dict[str, Any] | None:
    if resp.error is None:
        return None
    payload = {
        "provider": resp.error.provider,
        "endpoint": resp.error.endpoint,
        "status_code": resp.error.status_code,
        "error_type": resp.error.error_type,
        "message": resp.error.message,
        "retryable": resp.error.retryable,
    }
    flags = getattr(resp.lineage, "data_quality_flags", None)
    if flags:
        payload["data_quality_flags"] = dict(flags)
    return payload


def _breaker_opened_by_current_watchdog_timeout(
    exc: ProviderOutageCircuitBreaker,
) -> bool:
    payload = exc.payload
    reason = str(payload.get("circuit_reason") or "")
    return (
        payload.get("breaker_opened_by_current_call") is True
        and reason.startswith("watchdog_timeout")
    )


def _watchdog_timeout_breaker_payload(resp: AdapterResponse[Any]) -> dict[str, Any] | None:
    if resp.error is None or resp.error.error_type != "watchdog_timeout":
        return None
    flags = getattr(resp.lineage, "data_quality_flags", None)
    if not isinstance(flags, Mapping):
        return None
    breaker = flags.get("provider_outage_circuit_breaker")
    if not isinstance(breaker, Mapping):
        return None
    reason = str(breaker.get("circuit_reason") or "")
    if (
        breaker.get("breaker_opened_by_current_call") is True
        and reason.startswith("watchdog_timeout")
    ):
        return dict(breaker)
    return None


def _provider_watchdog_timeout_response(
    *,
    provider: str,
    endpoint: str,
    ticker: str,
    stage: str,
    trading_date: date,
    deadline_seconds: float,
    watchdog_snapshot: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> AdapterResponse[Any]:
    normalized = ticker.upper()
    payload = {
        "provider": provider,
        "endpoint": endpoint,
        "ticker": normalized,
        "stage": stage,
        "trading_date": trading_date.isoformat(),
        "deadline_seconds": deadline_seconds,
        "watchdog": dict(watchdog_snapshot),
        **dict(extra or {}),
    }
    return AdapterResponse(
        data=None,
        lineage=LineageMeta(
            provider=provider,
            endpoint=endpoint,
            request_timestamp=datetime.now(timezone.utc),
            asof_timestamp=datetime.now(timezone.utc),
            raw_payload_hash=stable_hash(payload),
            source_authority=provider,
            data_quality_flags=payload,
        ),
        error=ProviderError(
            provider=provider,
            endpoint=endpoint,
            status_code=None,
            error_type="watchdog_timeout",
            message=(
                f"{stage} exceeded hard wall-clock deadline "
                f"of {deadline_seconds:g} seconds"
            ),
            retryable=True,
        ),
    )


def _validate_minute_path_mode(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in MINUTE_PATH_MODES:
        raise ValueError(
            f"minute_path_mode must be one of {', '.join(MINUTE_PATH_MODES)}"
        )
    return normalized


def _minute_path_diagnostics(
    *,
    trading_date: date,
    decision_ts: datetime,
    minute_bars: Sequence[_MinuteBar],
    prior_ctx: Mapping[str, Any] | None,
    path_mode: str,
) -> dict[str, Any]:
    expected = _expected_minute_timestamps_before_decision(trading_date, decision_ts)
    expected_set = set(expected)
    observed_by_ts = {
        _minute_floor_utc(bar.timestamp): bar
        for bar in minute_bars
        if _minute_floor_utc(bar.timestamp) in expected_set
    }
    observed_ts = sorted(observed_by_ts)
    missing = [ts for ts in expected if ts not in observed_by_ts]
    session_open = us_equity_session_open_timestamp(trading_date)
    completed_through_ts = _minute_floor_utc(_aware_utc(decision_ts))
    first = observed_by_ts[observed_ts[0]] if observed_ts else None
    last = observed_by_ts[observed_ts[-1]] if observed_ts else None
    cumulative_volume = sum(bar.volume for bar in observed_by_ts.values())
    completed_minutes = _completed_minutes_before_decision(trading_date, decision_ts)
    session_minutes = _session_minutes(trading_date)
    zero_fill_projected_volume = (
        cumulative_volume / completed_minutes * session_minutes
        if completed_minutes > 0 else None
    )
    avg20_volume = prior_ctx.get("avg20_volume") if prior_ctx else None
    zero_fill_projected_volume_ratio = (
        zero_fill_projected_volume / avg20_volume
        if zero_fill_projected_volume is not None
        and avg20_volume is not None
        and avg20_volume > 0
        else None
    )
    return {
        "path_mode": path_mode,
        "expected_minute_count_before_decision": len(expected),
        "observed_minute_count_before_decision": len(observed_ts),
        "missing_minute_count_before_decision": len(missing),
        "path_coverage_ratio": len(observed_ts) / len(expected) if expected else None,
        "missing_minute_offsets": [
            int((ts - session_open).total_seconds() // 60) for ts in missing
        ],
        "missing_minute_timestamps": [ts.isoformat() for ts in missing],
        "first_observed_minute_ts": observed_ts[0].isoformat() if observed_ts else None,
        "last_observed_minute_ts": observed_ts[-1].isoformat() if observed_ts else None,
        "opening_bar_present": bool(observed_ts and observed_ts[0] == session_open),
        "last_observed_before_decision_age_seconds": (
            (completed_through_ts - observed_ts[-1] - timedelta(minutes=1)).total_seconds()
            if observed_ts else None
        ),
        "observed_cumulative_volume_before_decision": cumulative_volume,
        "observed_open_to_decision_return": (
            _safe_return(last.close, first.open) if first and last else None
        ),
        "zero_fill_projected_volume_at_decision": zero_fill_projected_volume,
        "zero_fill_projected_volume_ratio": zero_fill_projected_volume_ratio,
        "source_minute_bars_max_start_ts": observed_ts[-1].isoformat()
        if observed_ts else None,
        "feature_asof_ts": completed_through_ts.isoformat(),
        "completed_through_ts": completed_through_ts.isoformat(),
    }


def _sparse_zero_fill_minutes(
    *,
    trading_date: date,
    decision_ts: datetime,
    minute_bars: Sequence[_MinuteBar],
) -> tuple[list[_MinuteBar], dict[str, Any]]:
    expected = _expected_minute_timestamps_before_decision(trading_date, decision_ts)
    by_ts = {_minute_floor_utc(bar.timestamp): bar for bar in minute_bars}
    session_open = us_equity_session_open_timestamp(trading_date)
    filled: list[_MinuteBar] = []
    last_observed_close: float | None = None
    imputed: list[datetime] = []
    for ts in expected:
        bar = by_ts.get(ts)
        if bar is not None:
            filled.append(bar)
            last_observed_close = bar.close
            continue
        if last_observed_close is None:
            raise RuntimeError("sparse_zero_fill_requires_prior_observed_trade")
        idx = int((ts - session_open).total_seconds() // 60)
        filled.append(_MinuteBar(
            timestamp=ts,
            minute_index=idx,
            open=last_observed_close,
            high=last_observed_close,
            low=last_observed_close,
            close=last_observed_close,
            volume=0.0,
        ))
        imputed.append(ts)
    return filled, {
        "zero_fill_imputed_minute_count": len(imputed),
        "zero_fill_imputed_minute_ratio": len(imputed) / len(expected) if expected else None,
        "zero_fill_imputed_minute_offsets": [
            int((ts - session_open).total_seconds() // 60) for ts in imputed
        ],
        "zero_fill_imputed_minute_timestamps": [ts.isoformat() for ts in imputed],
        "minute_imputation_policy": (
            "zero_volume_forward_fill_prices_from_prior_observed_minute"
        ),
    }


def _candidate_attempt_hash(
    *,
    ticker: str,
    trading_date: date,
    decision_ts: datetime,
    decision_time_label: str,
    source_hur_identity_hash: str | None,
    path_mode: str = STRICT_MINUTE_PATH_MODE,
) -> str:
    path_mode = _validate_minute_path_mode(path_mode)
    return stable_hash({
        "ticker": ticker.upper(),
        "decision_date": trading_date.isoformat(),
        "decision_ts": _aware_utc(decision_ts).isoformat(),
        "decision_time_label": decision_time_label,
        "path_mode": path_mode,
        "reconstruction_method": REBUILD_METHOD,
        "feature_manifest_version": FEATURE_MANIFEST_VERSION,
        "source_hur_identity_hash": source_hur_identity_hash,
        "decision_policy": _decision_policy_payload(decision_time_label),
    })


def _quote_replay_attempt_hash(
    candidate: I12PitCandidate,
    window: QuoteWindow,
    *,
    feed: str,
    max_quote_age_seconds: float,
) -> str:
    return stable_hash({
        "candidate_attempt_hash": candidate.candidate_attempt_hash,
        "candidate_id": candidate.i12_pit_candidate_id,
        "ticker": candidate.ticker.upper(),
        "decision_date": candidate.decision_date.isoformat(),
        "decision_ts": _coerce_persisted_utc(candidate.decision_ts).isoformat(),
        "quote_role": window.quote_role,
        "target_ts": _aware_utc(window.target_ts).isoformat(),
        "window_start_ts": _aware_utc(window.window_start_ts).isoformat(),
        "window_end_ts": _aware_utc(window.window_end_ts).isoformat(),
        "feed": feed,
        "max_quote_age_seconds": float(max_quote_age_seconds),
        "quote_size_basis": ALPACA_QUOTE_SIZE_BASIS,
        "source": "alpaca_historical_quotes",
    })


def _cost_replay_attempt_hash(
    candidate: I12PitCandidate,
    *,
    exit_role: str,
    intended_order_usd: float,
    max_spread_bps: float,
    slippage_bps: float,
    entry_quote: I12PitQuoteReplay | None,
    exit_quote: I12PitQuoteReplay | None,
) -> str:
    return stable_hash({
        "candidate_attempt_hash": candidate.candidate_attempt_hash,
        "candidate_id": candidate.i12_pit_candidate_id,
        "ticker": candidate.ticker.upper(),
        "decision_date": candidate.decision_date.isoformat(),
        "decision_ts": _coerce_persisted_utc(candidate.decision_ts).isoformat(),
        "exit_role": exit_role,
        "intended_order_usd": float(intended_order_usd),
        "max_spread_bps": float(max_spread_bps),
        "slippage_bps": float(slippage_bps),
        "entry_quote_replay_id": (
            entry_quote.i12_pit_quote_replay_id if entry_quote else None
        ),
        "entry_quote_attempt_hash": (
            entry_quote.quote_replay_attempt_hash if entry_quote else None
        ),
        "entry_quote_content_hash": entry_quote.content_hash if entry_quote else None,
        "exit_quote_replay_id": (
            exit_quote.i12_pit_quote_replay_id if exit_quote else None
        ),
        "exit_quote_attempt_hash": (
            exit_quote.quote_replay_attempt_hash if exit_quote else None
        ),
        "exit_quote_content_hash": exit_quote.content_hash if exit_quote else None,
    })


def _decision_policy_payload(decision_time_label: str) -> dict[str, Any]:
    return {
        "decision_time_label": decision_time_label,
        "decision_time_semantics": (
            "decision_after_prior_completed_minute_start_stamped_bars"
        ),
        "source_minute_predicate": "bar_start_lt_decision_ts",
        "drawdown_max252_threshold": -0.50,
        "gap_min": -0.05,
        "gap_max_exclusive": 0.05,
        "projected_volume_ratio_min": 5.0,
        "min_completed_minutes": 5.0,
        "max_completed_minutes": 60.0,
    }


def _hur_source_row_from_model(
    row: HistoricalUniverseReconstruction,
    *,
    source_schema: str,
) -> HurSourceRow:
    payload = {
        "source_hur_schema": source_schema,
        "ticker": str(row.normalized_symbol).upper(),
        "replay_date": row.replay_date.isoformat(),
        "input_hash": row.input_hash,
        "output_hash": row.output_hash,
        "reconstruction_method": row.reconstruction_method,
        "source": row.source,
        "pit_filter_status_json": row.pit_filter_status_json,
    }
    return HurSourceRow(
        ticker=str(row.normalized_symbol).upper(),
        trading_date=row.replay_date,
        source_hur_identity_hash=stable_hash(payload),
        source_hur_payload=payload,
    )


def _hur_source_row_from_mapping(
    mapping: Mapping[str, Any],
    *,
    source_schema: str,
    fallback_date: date,
) -> HurSourceRow:
    ticker = str(
        mapping.get("normalized_symbol")
        or mapping.get("ticker")
        or ""
    ).upper()
    replay_date = _coerce_date(mapping.get("replay_date"), fallback=fallback_date)
    payload = {
        "source_hur_schema": source_schema,
        "ticker": ticker,
        "replay_date": replay_date.isoformat(),
        "input_hash": mapping.get("input_hash"),
        "output_hash": mapping.get("output_hash"),
        "reconstruction_method": mapping.get("reconstruction_method"),
        "source": mapping.get("source"),
        "pit_filter_status_json": mapping.get("pit_filter_status_json"),
    }
    return HurSourceRow(
        ticker=ticker,
        trading_date=replay_date,
        source_hur_identity_hash=stable_hash(payload),
        source_hur_payload=payload,
    )


def _coerce_date(value: Any, *, fallback: date) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if value:
        return date.fromisoformat(str(value))
    return fallback


def _should_schema_qualify_hur(session: Session, source_schema: str) -> bool:
    dialect = getattr(getattr(session.get_bind(), "dialect", None), "name", None)
    if dialect == "postgresql":
        return True
    if dialect == "sqlite" and source_schema not in {"public", "main"}:
        return True
    return False


def _count_hur_rows(
    session: Session,
    source_schema: str,
    *,
    start_date: date,
    end_date: date,
) -> int:
    source_schema = _validate_source_schema_name(source_schema)
    if _should_schema_qualify_hur(session, source_schema):
        return int(session.execute(
            text(
                "SELECT count(*) "
                f"FROM {_quote_ident(source_schema)}."
                "historical_universe_reconstructions "
                "WHERE replay_date >= :start_date "
                "AND replay_date <= :end_date "
                "AND inclusion_status = 'included'"
            ),
            {
                "start_date": start_date,
                "end_date": end_date,
            },
        ).scalar() or 0)
    return int(
        session.query(HistoricalUniverseReconstruction)
        .filter(
            HistoricalUniverseReconstruction.replay_date >= start_date,
            HistoricalUniverseReconstruction.replay_date <= end_date,
            HistoricalUniverseReconstruction.inclusion_status == "included",
        )
        .count()
    )


def _validate_source_schema_name(schema: str | None) -> str:
    normalized = (schema or "").strip()
    if not normalized:
        raise ValueError("source_hur_schema is required")
    if normalized != normalized.lower():
        raise ValueError("source_hur_schema must be lowercase")
    if not normalized.replace("_", "a").isalnum() or normalized[0].isdigit():
        raise ValueError(f"invalid source_hur_schema: {schema!r}")
    if normalized.startswith("pg_") or normalized in {"pg_catalog", "information_schema"}:
        raise ValueError(f"reserved source_hur_schema: {schema!r}")
    return normalized


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _decision_timestamp(trading_date: date, label: str) -> datetime:
    hour, minute = [int(part) for part in _normalize_decision_time_label(label).split(":", 1)]
    return _eastern_timestamp(trading_date, time(hour, minute))


def _normalize_decision_time_labels(labels: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(_normalize_decision_time_label(label) for label in labels))
    if not normalized:
        raise ValueError("at least one decision time is required")
    return normalized


def _normalize_decision_time_label(label: str) -> str:
    raw = str(label or "").strip()
    try:
        hour_text, minute_text = raw.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except (TypeError, ValueError):
        raise ValueError(f"invalid decision time label: {label!r}") from None
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"invalid decision time label: {label!r}")
    return f"{hour:02d}:{minute:02d}"


def _eastern_timestamp(trading_date: date, value: time) -> datetime:
    return datetime.combine(trading_date, value, tzinfo=EASTERN).astimezone(timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _coerce_persisted_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_optional_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return _coerce_persisted_utc(value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return _coerce_persisted_utc(parsed)


def _finite_positive(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        return None
    return float(value)


def _finite_nonnegative(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)) or float(value) < 0:
        return None
    return float(value)


def _spread_bps(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid * 10000.0


def _walk_feature_paths(payload: Any, prefix: str = ""):
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_feature_paths(value, path)
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            yield from _walk_feature_paths(value, f"{prefix}[{idx}]")
    else:
        yield prefix, payload


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _mean(values) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _numeric_summary(values) -> dict[str, float | int | None]:
    finite = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p75": None,
            "p90": None,
        }
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "p50": _percentile_sorted(finite, 0.50),
        "p75": _percentile_sorted(finite, 0.75),
        "p90": _percentile_sorted(finite, 0.90),
    }


def _percentile_sorted(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight
