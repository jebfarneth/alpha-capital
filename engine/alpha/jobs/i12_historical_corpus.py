"""Durable historical I12 intraday corpus builder.

This is an I-track research corpus path, not the canonical detector stack.
Confirmed I12 entries are persisted to ``signal_registry``; all minute
lifecycle rows, including non-entry controls, are persisted to
``intraday_event_details``. Labels are computed here and are not delegated to
the forward-return clock. Persisted event rows are conditioned on a daily
candidate screen, including a full-day volume-ratio floor that is deliberately
not point-in-time. That keeps the historical control class tractable and
comparable to the validated research set, but confirmed registry rows and
feature snapshots therefore set ``point_in_time_passed=False`` while retaining
``lookahead_guard_passed=True`` for the ex-label feature payload itself.
Premarket poison gaps that pass this same volume floor are retained as
daily-only controls without Polygon minute fetches. Both caveats are stamped
into every feature payload.

Security-type classification intentionally reuses the M4-labeled exclusion
artifact, extended to cover all public HUR-included symbols. This job only
loads tickers from ``historical_universe_reconstructions`` rows with
``inclusion_status='included'``; a non-HUR ticker reaching classification is a
job invariant violation.
"""

from __future__ import annotations

import json
import math
import time as time_module
from concurrent.futures import TimeoutError as FuturesTimeoutError
from copy import deepcopy
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import Session

from alpha.data.contracts import AdapterResponse, stable_hash, utcnow
from alpha.data.fmp import FmpBar, HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.data.polygon import PolygonBar
from alpha.db.models import (
    DataLineage,
    FmpDelistedCompanyRecord,
    ForwardReturnObservation,
    HistoricalUniverseReconstruction,
    IntradayEventDetail,
    SignalRegistry,
)
from alpha.evidence.writer import record_data_lineage, record_feature_snapshot, record_signal
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.watchdog import (
    DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS,
    ProviderOutageCircuitBreaker,
    WatchdogState,
    call_with_daemon_deadline,
)
from alpha.jobs.paper_execution import (
    BOUNDARY_EPSILON,
    EASTERN,
    I12_PATTERN_ID,
    PremarketContext,
    PolygonSnapshotTicker,
    compute_shared_intraday_math,
    i12_entry_gate,
)
from alpha.market_calendar import (
    is_us_equity_session,
    next_us_equity_session,
    us_equity_session_close_timestamp,
    us_equity_session_open_timestamp,
)
from alpha.ml.security_type_exclusions import (
    ExclusionArtifactError,
    SecurityTypeClassification,
    load_classifications,
)


JOB_NAME = "i12_historical_corpus"
RECONSTRUCTION_METHOD = "historical_i12_replay_polygon_minute_fmp_eod_v1"
FEATURE_MANIFEST_VERSION = "i12_historical_corpus_v1"
I12_SIGNAL_HORIZON = "1d"
OUTCOME_CONFIRMED = "confirmed_filled"
OUTCOME_NEVER_CONFIRMED = "never_confirmed"
OUTCOME_POISON = "poison_blocked"
OUTCOME_POISON_PREMARKET = "poison_premarket"
OUTCOME_PARABOLIC = "parabolic_blocked"
OUTCOME_HALTED_UNFILLABLE = "halted_unfillable"
MINUTE_BAR_ENDPOINT = "/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}"
SPLIT_BASIS_OPEN_TOLERANCE_PCT = 0.02
I12_CONFIRMATION_MAX_MINUTE = 60
SESSION_EXIT_TIME = time(15, 55)
MIN_PRIOR_DAILY_SESSIONS = 20
CANDIDATE_FULL_DAY_VR_FLOOR = 2.0
ML_EXCLUSION_PRIMARY_LABEL_UNAVAILABLE = "primary_label_unavailable"
ML_EXCLUSION_SPLIT_BASIS_MISMATCH = "split_basis_mismatch"
CANDIDATE_SCREEN_STAMP = {
    "drawdown_max": -0.50,
    "gap_band": [-0.05, 0.05],
    "full_day_vr_floor": CANDIDATE_FULL_DAY_VR_FLOOR,
    "selection_uses_full_day_volume": True,
    "caveat": "full_day_vr_floor_is_candidate_conditioning_not_pit",
    "poison_gap_controls": "gap_below_-0.05_retained_as_poison_premarket_when_volume_screen_passes",
}
PROGRESS_HEARTBEAT_EVERY_TICKER_DAYS = 250
DEFAULT_FETCH_DEADLINE_SECONDS = 120.0
POLYGON_SESSION_RESET_TIMEOUT_INTERVAL = 3

ProgressCallback = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class _DailyBar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    split_adjusted_close: float


@dataclass(frozen=True)
class _MinuteBar:
    timestamp: datetime
    minute_index: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class _TickerDayInput:
    ticker: str
    trading_date: date
    daily_bars: tuple[_DailyBar, ...]
    minute_bars: tuple[_MinuteBar, ...]
    daily_lineage: DataLineage
    minute_lineage: DataLineage
    security_type: SecurityTypeClassification
    sessions_to_delist: int | None


@dataclass(frozen=True)
class _DailyTickerInput:
    ticker: str
    trading_date: date
    daily_bars: tuple[_DailyBar, ...]
    daily_lineage: DataLineage
    security_type: SecurityTypeClassification
    sessions_to_delist: int | None


@dataclass(frozen=True)
class _DailyContext:
    context: PremarketContext
    day_bar: _DailyBar
    prior_count: int
    distance_from_max252: float
    gap: float
    full_day_volume_ratio: float | None


@dataclass(frozen=True)
class _CandidateScreenResult:
    passed: bool
    reason: str | None
    daily_context: _DailyContext
    control_outcome: str | None = None


@dataclass(frozen=True)
class _I12Event:
    ticker: str
    trading_date: date
    outcome: str
    gate_values: dict[str, Any]
    feature_json: dict[str, Any]
    labels: dict[str, Any]
    artifact_flags: dict[str, Any]
    signal_timestamp: datetime | None
    confirmation_timestamp: datetime | None
    entry_timestamp: datetime | None
    exit_timestamp: datetime | None
    conf_minute: int | None
    entry_minute: int | None
    entry_price: float | None
    exit_price: float | None
    session_open_price: float | None
    session_close_price: float | None
    next_open_price: float | None
    projected_vol_at_conf: float | None
    projected_vol_ratio_at_conf: float | None
    full_day_volume_ratio: float | None
    chase_pct: float | None
    gap_pct: float | None
    distance_from_max252: float | None
    ret_conf: float | None
    ret_open_close: float | None
    ret_next_open: float | None
    mae_pct: float | None
    mfe_pct: float | None
    halted: bool
    sub_dollar_at_open: bool
    split_basis_mismatch: bool
    is_ml_excluded: bool
    ml_exclusion_reason: str
    security_type: str
    sessions_to_delist: int | None
    data_lineage_ids: tuple[str, ...]
    input_hash: str
    output_hash: str
    event_identity_hash: str
    signal_identity_hash: str | None


@dataclass
class _RunCounters:
    ticker_days_scanned: int = 0
    candidates: int = 0
    candidates_screened_out: int = 0
    confirmed: int = 0
    never_confirmed: int = 0
    poison_blocked: int = 0
    poison_premarket: int = 0
    parabolic_blocked: int = 0
    halted_unfillable: int = 0
    excluded_by_type: int = 0
    artifact_excluded: int = 0
    sub_dollar_included: int = 0
    non_session_bars_skipped: int = 0
    fetch_errors: int = 0
    watchdog_timeouts: int = 0
    quarantined: int = 0
    inserted_details: int = 0
    reused_details: int = 0
    inserted_signals: int = 0
    reused_signals: int = 0
    forward_return_observations_inserted: int = 0
    forward_return_observations_reused: int = 0
    minute_cache_hits: int = 0
    minute_cache_misses: int = 0
    skipped_existing: int = 0
    db_reconnect_retries: int = 0
    batches: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    non_session_bar_samples: list[dict[str, str]] = field(default_factory=list)
    candidate_screen_fail_reasons: Counter[str] = field(default_factory=Counter)
    outcomes: Counter[str] = field(default_factory=Counter)

    def record_non_session(self, ticker: str, day: date) -> None:
        self.non_session_bars_skipped += 1
        if len(self.non_session_bar_samples) < 10:
            self.non_session_bar_samples.append({
                "ticker": ticker.upper(),
                "date": day.isoformat(),
            })

    def merge(self, other: "_RunCounters") -> None:
        self.ticker_days_scanned += other.ticker_days_scanned
        self.candidates += other.candidates
        self.candidates_screened_out += other.candidates_screened_out
        self.confirmed += other.confirmed
        self.never_confirmed += other.never_confirmed
        self.poison_blocked += other.poison_blocked
        self.poison_premarket += other.poison_premarket
        self.parabolic_blocked += other.parabolic_blocked
        self.halted_unfillable += other.halted_unfillable
        self.excluded_by_type += other.excluded_by_type
        self.artifact_excluded += other.artifact_excluded
        self.sub_dollar_included += other.sub_dollar_included
        self.non_session_bars_skipped += other.non_session_bars_skipped
        self.fetch_errors += other.fetch_errors
        self.watchdog_timeouts += other.watchdog_timeouts
        self.quarantined += other.quarantined
        self.inserted_details += other.inserted_details
        self.reused_details += other.reused_details
        self.inserted_signals += other.inserted_signals
        self.reused_signals += other.reused_signals
        self.forward_return_observations_inserted += other.forward_return_observations_inserted
        self.forward_return_observations_reused += other.forward_return_observations_reused
        self.minute_cache_hits += other.minute_cache_hits
        self.minute_cache_misses += other.minute_cache_misses
        self.skipped_existing += other.skipped_existing
        self.db_reconnect_retries += other.db_reconnect_retries
        self.batches += other.batches
        self.errors.extend(other.errors)
        self.non_session_bar_samples.extend(
            other.non_session_bar_samples[: max(0, 10 - len(self.non_session_bar_samples))]
        )
        self.candidate_screen_fail_reasons.update(other.candidate_screen_fail_reasons)
        self.outcomes.update(other.outcomes)


class I12HistoricalCorpusJob(BaseJob):
    """Replay the frozen I12 gate over historical HUR membership."""

    @property
    def job_name(self) -> str:
        return JOB_NAME

    @property
    def job_type(self) -> str:
        return "historical_replay"

    @property
    def event_pattern_id(self) -> str:
        return I12_PATTERN_ID

    def __init__(
        self,
        *,
        session: Session,
        fmp_adapter: Any,
        polygon_adapter: Any,
        start_date: date,
        end_date: date,
        run_timestamp: datetime | None = None,
        batch_days: int = 10,
        classification_records: Mapping[str, SecurityTypeClassification] | None = None,
        minute_cache_dir: str | Path | None = None,
        polygon_rate_limit_per_minute: int | None = None,
        skip_existing: bool = False,
        max_db_retries: int = 3,
        db_retry_backoff_seconds: float = 5.0,
        fetch_deadline_seconds: float = DEFAULT_FETCH_DEADLINE_SECONDS,
        progress_callback: ProgressCallback | None = None,
        max_outstanding_fetch_timeouts: int = DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS,
    ) -> None:
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        if batch_days < 1:
            raise ValueError("batch_days must be >= 1")
        if max_db_retries < 0:
            raise ValueError("max_db_retries must be >= 0")
        if db_retry_backoff_seconds < 0:
            raise ValueError("db_retry_backoff_seconds must be >= 0")
        if fetch_deadline_seconds <= 0:
            raise ValueError("fetch_deadline_seconds must be > 0")
        if max_outstanding_fetch_timeouts < 1:
            raise ValueError("max_outstanding_fetch_timeouts must be >= 1")
        self._session = session
        self._fmp = fmp_adapter
        self._polygon = polygon_adapter
        self._start_date = start_date
        self._end_date = end_date
        self._run_timestamp = run_timestamp
        self._batch_days = batch_days
        self._classification_records = classification_records
        self._minute_cache_dir = Path(minute_cache_dir) if minute_cache_dir else None
        self._polygon_rate_limit_per_minute = polygon_rate_limit_per_minute
        self._skip_existing = skip_existing
        self._max_db_retries = max_db_retries
        self._db_retry_backoff_seconds = db_retry_backoff_seconds
        self._fetch_deadline_seconds = float(fetch_deadline_seconds)
        self._last_polygon_fetch_monotonic: float | None = None
        self._progress_callback = progress_callback
        self._latest_metrics: dict[str, Any] = {}
        self._fetch_watchdog = WatchdogState(
            max_outstanding_timeouts=max_outstanding_fetch_timeouts,
            max_consecutive_timeouts=max_outstanding_fetch_timeouts,
        )

    @property
    def partial_metrics(self) -> dict[str, Any]:
        return dict(self._latest_metrics)

    def run(self, ctx: JobContext) -> JobResult:
        counters = _RunCounters()
        trading_dates = _trading_dates(self._start_date, self._end_date)
        if not trading_dates:
            return JobResult(
                status="finished",
                metrics=self._metrics(counters, trading_dates=[]),
            )
        classifications = (
            self._classification_records
            if self._classification_records is not None
            else load_classifications()
        )
        if self._run_timestamp is None:
            self._run_timestamp = utcnow()

        daily_cache: dict[str, tuple[tuple[_DailyBar, ...], DataLineage]] = {}
        minute_cache: dict[tuple[str, date], tuple[tuple[_MinuteBar, ...], DataLineage]] = {}

        try:
            self._run_batches_with_retry(
                ctx,
                counters=counters,
                trading_dates=trading_dates,
                classifications=classifications,
                daily_cache=daily_cache,
                minute_cache=minute_cache,
                process_batch=self._run_batch_once,
            )
        except ProviderOutageCircuitBreaker as exc:
            counters.fetch_errors += 1
            counters.watchdog_timeouts = max(
                counters.watchdog_timeouts,
                self._fetch_watchdog.total_timeouts,
            )
            counters.errors.append(exc.payload)
            self._progress(
                "provider_outage_circuit_breaker",
                {
                    "metrics": self._metrics(counters, trading_dates=trading_dates),
                    **exc.payload,
                },
            )
            return JobResult(
                status="partial_failed",
                metrics=self._metrics(counters, trading_dates=trading_dates),
                errors=counters.errors,
            )

        return JobResult(
            status="finished",
            metrics=self._metrics(counters, trading_dates=trading_dates),
            errors=counters.errors,
        )

    def _run_batches_with_retry(
        self,
        ctx: JobContext,
        *,
        counters: Any,
        trading_dates: Sequence[date],
        classifications: Mapping[str, SecurityTypeClassification],
        daily_cache: dict[str, tuple[tuple[_DailyBar, ...], DataLineage]],
        minute_cache: dict[tuple[str, date], tuple[tuple[_MinuteBar, ...], DataLineage]],
        process_batch: Callable[..., Any],
    ) -> None:
        for batch_index, batch_dates in enumerate(_chunks(trading_dates, self._batch_days), start=1):
            attempt = 0
            while True:
                try:
                    batch_counters = process_batch(
                        ctx,
                        batch_index=batch_index,
                        batch_dates=batch_dates,
                        classifications=classifications,
                        trading_dates=trading_dates,
                        daily_cache=daily_cache,
                        minute_cache=minute_cache,
                        cumulative_counters=counters,
                    )
                    counters.merge(batch_counters)
                    self._progress("batch_finish", {
                        "batch_index": batch_index,
                        "metrics": self._metrics(counters, trading_dates=trading_dates),
                    })
                    break
                except Exception as exc:
                    if not _is_transient_db_error(exc) or attempt >= self._max_db_retries:
                        raise
                    attempt += 1
                    counters.db_reconnect_retries += 1
                    counters.errors.append({
                        "error": "transient_db_disconnect_retry",
                        "batch_index": batch_index,
                        "attempt": attempt,
                        "exception": _transient_error_summary(exc),
                    })
                    self._recover_after_transient_db_error()
                    daily_cache.clear()
                    minute_cache.clear()
                    self._progress("batch_retry", {
                        "batch_index": batch_index,
                        "attempt": attempt,
                        "max_attempts": self._max_db_retries,
                        "date_start": batch_dates[0].isoformat(),
                        "date_end": batch_dates[-1].isoformat(),
                        "metrics": self._metrics(counters, trading_dates=trading_dates),
                    })
                    if self._db_retry_backoff_seconds > 0:
                        time_module.sleep(self._db_retry_backoff_seconds * attempt)

    def _run_batch_once(
        self,
        ctx: JobContext,
        *,
        batch_index: int,
        batch_dates: Sequence[date],
        classifications: Mapping[str, SecurityTypeClassification],
        trading_dates: Sequence[date],
        daily_cache: dict[str, tuple[tuple[_DailyBar, ...], DataLineage]],
        minute_cache: dict[tuple[str, date], tuple[tuple[_MinuteBar, ...], DataLineage]],
        cumulative_counters: _RunCounters,
    ) -> _RunCounters:
        counters = _RunCounters(batches=1)
        self._progress("batch_start", {
            "batch_index": batch_index,
            "date_start": batch_dates[0].isoformat(),
            "date_end": batch_dates[-1].isoformat(),
        })
        hur_rows = self._load_hur_rows(batch_dates)
        missing_hur_dates = [
            day.isoformat()
            for day in batch_dates
            if day not in hur_rows
        ]
        if missing_hur_dates:
            counters.quarantined += len(missing_hur_dates)
            counters.errors.append({
                "error": "missing_hur_rows",
                "dates": missing_hur_dates[:20],
            })
            self._session.commit()
            return counters
        for trading_date in batch_dates:
            hur_members = set(hur_rows[trading_date])
            for ticker in hur_rows[trading_date]:
                counters.ticker_days_scanned += 1
                scanned_total = cumulative_counters.ticker_days_scanned + counters.ticker_days_scanned
                if scanned_total == 1 or scanned_total % PROGRESS_HEARTBEAT_EVERY_TICKER_DAYS == 0:
                    progress_counters = deepcopy(cumulative_counters)
                    progress_counters.merge(counters)
                    self._progress("ticker_day_progress", {
                        "ticker_days_scanned": scanned_total,
                        "trading_date": trading_date.isoformat(),
                        "ticker": ticker,
                        "metrics": self._metrics(progress_counters, trading_dates=trading_dates),
                    })
                if self._skip_existing and self._has_existing_event(ticker, trading_date):
                    counters.skipped_existing += 1
                    continue
                try:
                    sec = _classification_for_hur_ticker(
                        classifications,
                        ticker,
                        hur_included=ticker in hur_members,
                    )
                except ExclusionArtifactError:
                    raise
                if sec.ml_excluded:
                    counters.excluded_by_type += 1
                try:
                    daily_input = self._load_daily_ticker_input(
                        ticker=ticker,
                        trading_date=trading_date,
                        security_type=sec,
                        daily_cache=daily_cache,
                        counters=counters,
                        job_run_id=ctx.job_run_id,
                    )
                    screen = _screen_daily_candidate(daily_input)
                    if screen.control_outcome == OUTCOME_POISON_PREMARKET:
                        event = _evaluate_i12_poison_premarket(daily_input, screen.daily_context)
                        counters.outcomes[event.outcome] += 1
                        counters.poison_premarket += 1
                        if event.split_basis_mismatch:
                            counters.artifact_excluded += 1
                        if event.sub_dollar_at_open:
                            counters.sub_dollar_included += 1
                        persisted = self._persist_event(event, ctx.job_run_id)
                        counters.inserted_details += int(persisted["inserted_detail"])
                        counters.reused_details += int(not persisted["inserted_detail"])
                        counters.inserted_signals += int(persisted["inserted_signal"])
                        counters.reused_signals += int(persisted["reused_signal"])
                        counters.forward_return_observations_inserted += int(
                            persisted.get("inserted_forward_return_observation", False)
                        )
                        counters.forward_return_observations_reused += int(
                            persisted.get("reused_forward_return_observation", False)
                        )
                        continue
                    if not screen.passed:
                        counters.candidates_screened_out += 1
                        counters.candidate_screen_fail_reasons[screen.reason or "unknown"] += 1
                        continue
                    counters.candidates += 1
                    inp = self._load_ticker_day_input(
                        daily_input=daily_input,
                        minute_cache=minute_cache,
                        counters=counters,
                        job_run_id=ctx.job_run_id,
                    )
                except _Quarantine as exc:
                    counters.quarantined += 1
                    counters.errors.append(exc.payload)
                    continue
                event = _evaluate_i12_event(inp)
                counters.outcomes[event.outcome] += 1
                if event.outcome == OUTCOME_CONFIRMED:
                    counters.confirmed += 1
                elif event.outcome == OUTCOME_NEVER_CONFIRMED:
                    counters.never_confirmed += 1
                elif event.outcome == OUTCOME_POISON:
                    counters.poison_blocked += 1
                elif event.outcome == OUTCOME_PARABOLIC:
                    counters.parabolic_blocked += 1
                elif event.outcome == OUTCOME_HALTED_UNFILLABLE:
                    counters.halted_unfillable += 1
                if event.split_basis_mismatch:
                    counters.artifact_excluded += 1
                if event.sub_dollar_at_open:
                    counters.sub_dollar_included += 1
                persisted = self._persist_event(event, ctx.job_run_id)
                counters.inserted_details += int(persisted["inserted_detail"])
                counters.reused_details += int(not persisted["inserted_detail"])
                counters.inserted_signals += int(persisted["inserted_signal"])
                counters.reused_signals += int(persisted["reused_signal"])
                counters.forward_return_observations_inserted += int(
                    persisted.get("inserted_forward_return_observation", False)
                )
                counters.forward_return_observations_reused += int(
                    persisted.get("reused_forward_return_observation", False)
                )
        self._session.commit()
        return counters

    def _load_hur_rows(self, trading_dates: Sequence[date]) -> dict[date, list[str]]:
        rows = (
            self._session.query(
                HistoricalUniverseReconstruction.replay_date,
                HistoricalUniverseReconstruction.normalized_symbol,
            )
            .filter(
                HistoricalUniverseReconstruction.replay_date.in_(list(trading_dates)),
                HistoricalUniverseReconstruction.inclusion_status == "included",
            )
            .order_by(
                HistoricalUniverseReconstruction.replay_date,
                HistoricalUniverseReconstruction.normalized_symbol,
            )
            .all()
        )
        out: dict[date, list[str]] = {day: [] for day in trading_dates}
        for replay_date, ticker in rows:
            out.setdefault(replay_date, []).append(str(ticker).upper())
        return {day: sorted(set(tickers)) for day, tickers in out.items() if tickers}

    def _has_existing_event(self, ticker: str, trading_date: date) -> bool:
        return (
            self._session.query(IntradayEventDetail.event_identity_hash)
            .filter(
                IntradayEventDetail.pattern_id == self.event_pattern_id,
                IntradayEventDetail.ticker == ticker.upper(),
                IntradayEventDetail.trading_date == trading_date,
            )
            .first()
            is not None
        )

    def _recover_after_transient_db_error(self) -> None:
        try:
            self._session.rollback()
        except Exception:
            pass
        bind = self._session.get_bind()
        try:
            self._session.close()
        finally:
            dispose = getattr(bind, "dispose", None)
            dialect_name = getattr(getattr(bind, "dialect", None), "name", None)
            if dialect_name == "postgresql" and callable(dispose):
                dispose()

    def _load_daily_ticker_input(
        self,
        *,
        ticker: str,
        trading_date: date,
        security_type: SecurityTypeClassification,
        daily_cache: dict[str, tuple[tuple[_DailyBar, ...], DataLineage]],
        counters: _RunCounters,
        job_run_id: str,
    ) -> _DailyTickerInput:
        if ticker not in daily_cache:
            from_date = self._start_date - timedelta(days=460)
            to_date = next_us_equity_session(self._end_date + timedelta(days=1))
            self._progress("daily_fetch_start", {
                "ticker": ticker,
                "trading_date": trading_date.isoformat(),
                "deadline_seconds": self._fetch_deadline_seconds,
                **self._fetch_watchdog.snapshot(),
            })

            def _fetch_daily(
                ticker: str = ticker,
                from_date: date = from_date,
                to_date: date = to_date,
            ) -> AdapterResponse[Any]:
                return self._fmp.get_historical_price(
                    ticker,
                    from_date=from_date,
                    to_date=to_date,
                    adjusted=False,
                )

            try:
                resp = call_with_daemon_deadline(
                    _fetch_daily,
                    timeout_seconds=self._fetch_deadline_seconds,
                    thread_name="fmp-daily-fetch",
                    state=self._fetch_watchdog,
                    context={
                        "ticker": ticker.upper(),
                        "trading_date": trading_date.isoformat(),
                        "stage": "fmp_daily_fetch",
                        "deadline_seconds": self._fetch_deadline_seconds,
                    },
                )
            except FuturesTimeoutError as exc:
                counters.fetch_errors += 1
                counters.watchdog_timeouts += 1
                raise _Quarantine({
                    "ticker": ticker,
                    "trading_date": trading_date.isoformat(),
                    "error": "daily_fetch_watchdog_timeout",
                    "deadline_seconds": self._fetch_deadline_seconds,
                    **self._fetch_watchdog.snapshot(),
                }) from exc
            if not resp.ok:
                counters.fetch_errors += 1
                raise _Quarantine({
                    "ticker": ticker,
                    "trading_date": trading_date.isoformat(),
                    "error": "daily_bar_fetch_error",
                    "provider_error": _provider_error_payload(resp),
                })
            bars = _clean_daily_bars(ticker, resp.data or [], counters)
            lineage = _record_bars_lineage(
                self._session,
                provider="FMP",
                endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                ticker=ticker,
                trading_date=trading_date,
                bars=[_daily_payload(bar) for bar in bars],
                job_run_id=job_run_id,
                run_timestamp=self._run_timestamp,
            )
            daily_cache[ticker] = (tuple(bars), lineage)
        daily_bars, daily_lineage = daily_cache[ticker]
        sessions_to_delist = self._sessions_to_delist(ticker, trading_date)
        return _DailyTickerInput(
            ticker=ticker,
            trading_date=trading_date,
            daily_bars=daily_bars,
            daily_lineage=daily_lineage,
            security_type=security_type,
            sessions_to_delist=sessions_to_delist,
        )

    def _load_ticker_day_input(
        self,
        *,
        daily_input: _DailyTickerInput,
        minute_cache: dict[tuple[str, date], tuple[tuple[_MinuteBar, ...], DataLineage]],
        counters: _RunCounters,
        job_run_id: str,
    ) -> _TickerDayInput:
        ticker = daily_input.ticker
        trading_date = daily_input.trading_date
        minute_key = (ticker, trading_date)
        if minute_key not in minute_cache:
            raw_minutes = self._load_cached_polygon_minutes(ticker, trading_date)
            if raw_minutes is None:
                counters.minute_cache_misses += 1
                self._throttle_polygon_fetch()
                resp = self._fetch_polygon_minutes_with_deadline(
                    ticker,
                    trading_date,
                    counters=counters,
                )
                if not resp.ok:
                    counters.fetch_errors += 1
                    raise _Quarantine({
                        "ticker": ticker,
                        "trading_date": trading_date.isoformat(),
                        "error": "minute_bar_fetch_error",
                        "provider_error": _provider_error_payload(resp),
                    })
                raw_minutes = list(resp.data or [])
                self._store_cached_polygon_minutes(ticker, trading_date, raw_minutes)
            else:
                counters.minute_cache_hits += 1
            minutes = _clean_minute_bars(trading_date, raw_minutes)
            if not minutes:
                raise _Quarantine({
                    "ticker": ticker,
                    "trading_date": trading_date.isoformat(),
                    "error": "missing_minute_bars",
                })
            lineage = _record_bars_lineage(
                self._session,
                provider="Polygon",
                endpoint=MINUTE_BAR_ENDPOINT.format(
                    ticker=ticker,
                    date=trading_date.isoformat(),
                ),
                ticker=ticker,
                trading_date=trading_date,
                bars=[_minute_payload(bar) for bar in minutes],
                job_run_id=job_run_id,
                run_timestamp=self._run_timestamp,
            )
            minute_cache[minute_key] = (tuple(minutes), lineage)
        minute_bars, minute_lineage = minute_cache[minute_key]

        return _TickerDayInput(
            ticker=ticker,
            trading_date=trading_date,
            daily_bars=daily_input.daily_bars,
            minute_bars=minute_bars,
            daily_lineage=daily_input.daily_lineage,
            minute_lineage=minute_lineage,
            security_type=daily_input.security_type,
            sessions_to_delist=daily_input.sessions_to_delist,
        )

    def _fetch_polygon_minutes_with_deadline(
        self,
        ticker: str,
        trading_date: date,
        *,
        counters: _RunCounters,
    ) -> AdapterResponse[Any]:
        self._progress("minute_fetch_start", {
            "ticker": ticker,
            "trading_date": trading_date.isoformat(),
            "deadline_seconds": self._fetch_deadline_seconds,
            "cache_status": "miss",
            **self._fetch_watchdog.snapshot(),
        })

        def _fetch(
            ticker: str = ticker,
            trading_date: date = trading_date,
        ) -> AdapterResponse[Any]:
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
                thread_name="polygon-minute-fetch",
                state=self._fetch_watchdog,
                context={
                    "ticker": ticker.upper(),
                    "trading_date": trading_date.isoformat(),
                    "stage": "polygon_minute_fetch",
                    "deadline_seconds": self._fetch_deadline_seconds,
                },
            )
        except FuturesTimeoutError as exc:
            counters.fetch_errors += 1
            counters.watchdog_timeouts += 1
            self._maybe_reset_polygon_session()
            raise _Quarantine({
                "ticker": ticker,
                "trading_date": trading_date.isoformat(),
                "error": "fetch_watchdog_timeout",
                "deadline_seconds": self._fetch_deadline_seconds,
                **self._fetch_watchdog.snapshot(),
            }) from exc

    def _maybe_reset_polygon_session(self) -> None:
        if (
            self._fetch_watchdog.total_timeouts == 0
            or self._fetch_watchdog.total_timeouts % POLYGON_SESSION_RESET_TIMEOUT_INTERVAL
        ):
            return
        reset = getattr(self._polygon, "reset_session", None)
        if callable(reset):
            reset()
            self._progress("polygon_session_reset", self._fetch_watchdog.snapshot())

    def _sessions_to_delist(self, ticker: str, trading_date: date) -> int | None:
        row = (
            self._session.query(FmpDelistedCompanyRecord.delisted_date)
            .filter(
                FmpDelistedCompanyRecord.normalized_symbol == ticker.upper(),
                FmpDelistedCompanyRecord.delisted_date.isnot(None),
                FmpDelistedCompanyRecord.delisted_date >= trading_date,
            )
            .order_by(FmpDelistedCompanyRecord.delisted_date.asc())
            .first()
        )
        if row is None or row[0] is None:
            return None
        target = _last_us_equity_session_on_or_before(row[0])
        if target < trading_date:
            return None
        sessions = 0
        cursor = trading_date
        while cursor < target:
            cursor = next_us_equity_session(cursor + timedelta(days=1))
            sessions += 1
        return sessions

    def _persist_event(self, event: _I12Event, job_run_id: str) -> dict[str, bool]:
        existing_detail = (
            self._session.query(IntradayEventDetail)
            .filter(IntradayEventDetail.event_identity_hash == event.event_identity_hash)
            .one_or_none()
        )
        if existing_detail is not None:
            if (
                existing_detail.input_hash != event.input_hash
                or existing_detail.output_hash != event.output_hash
            ):
                raise RuntimeError(
                    "I12 corpus event content changed for existing identity: "
                    f"{event.ticker} {event.trading_date.isoformat()}"
                )
            fro_inserted = False
            fro_reused = False
            if existing_detail.signal_id is not None:
                signal = self._session.get(SignalRegistry, existing_detail.signal_id)
                if signal is None:
                    raise RuntimeError(
                        "I12 corpus detail references missing signal_registry row: "
                        f"{event.ticker} {event.trading_date.isoformat()} "
                        f"signal_id={existing_detail.signal_id}"
                    )
                fro_inserted, fro_reused = self._record_i12_forward_return_observation(
                    event,
                    signal=signal,
                    job_run_id=job_run_id,
                )
            return {
                "inserted_detail": False,
                "inserted_signal": False,
                "reused_signal": existing_detail.signal_id is not None,
                "inserted_forward_return_observation": fro_inserted,
                "reused_forward_return_observation": fro_reused,
            }

        signal_id: str | None = None
        signal: SignalRegistry | None = None
        inserted_signal = False
        reused_signal = False
        if event.outcome == OUTCOME_CONFIRMED and not event.is_ml_excluded:
            assert event.signal_identity_hash is not None
            existing_same_date_signal = (
                self._session.query(SignalRegistry)
                .filter(
                    SignalRegistry.pattern_id == I12_PATTERN_ID,
                    SignalRegistry.ticker == event.ticker,
                    SignalRegistry.trading_date == event.trading_date.isoformat(),
                )
                .one_or_none()
            )
            if (
                existing_same_date_signal is not None
                and existing_same_date_signal.signal_identity_hash != event.signal_identity_hash
            ):
                raise RuntimeError(
                    "i12_signal_identity_conflict: "
                    f"ticker={event.ticker} "
                    f"trading_date={event.trading_date.isoformat()} "
                    f"existing_hash={existing_same_date_signal.signal_identity_hash} "
                    f"new_hash={event.signal_identity_hash}"
                )
            existing_signal = (
                self._session.query(SignalRegistry)
                .filter(
                    SignalRegistry.pattern_id == I12_PATTERN_ID,
                    SignalRegistry.ticker == event.ticker,
                    SignalRegistry.signal_identity_hash == event.signal_identity_hash,
                )
                .one_or_none()
            )
            if existing_signal is None:
                feature_snapshot = record_feature_snapshot(
                    self._session,
                    pattern_id=I12_PATTERN_ID,
                    ticker=event.ticker,
                    asof_timestamp=event.signal_timestamp or event.confirmation_timestamp,
                    features=event.feature_json,
                    data_lineage_ids=list(event.data_lineage_ids),
                    job_run_id=job_run_id,
                    feature_manifest_version=FEATURE_MANIFEST_VERSION,
                    fidelity_tier="historical_intraday_replay",
                    point_in_time_passed=False,
                    lookahead_guard_passed=True,
                    input_hashes={"i12_corpus_event_input": event.input_hash},
                )
                signal = record_signal(
                    self._session,
                    pattern_id=I12_PATTERN_ID,
                    ticker=event.ticker,
                    direction="long",
                    signal_timestamp=event.signal_timestamp or event.confirmation_timestamp,
                    raw_signal_strength=float(event.projected_vol_ratio_at_conf or 0.0),
                    # Outcome labels are not ex-ante detector estimates; keep this
                    # NOT NULL registry field neutral to avoid a perfect-label leak.
                    raw_expected_edge=0.0,
                    feature_snapshot_id=feature_snapshot.feature_snapshot_id,
                    job_run_id=job_run_id,
                    signal_horizon=I12_SIGNAL_HORIZON,
                    thesis_category="capitulation_volume_bounce",
                    route_class="i_track_intraday",
                    fidelity_tier="historical_intraday_replay",
                    data_confidence=1.0,
                    data_lineage_ids=list(event.data_lineage_ids),
                    trading_date=event.trading_date.isoformat(),
                    next_execution_session=event.trading_date.isoformat(),
                    detector_version=RECONSTRUCTION_METHOD,
                    point_in_time_passed=False,
                    lookahead_guard_passed=True,
                    signal_event_sequence=1,
                    signal_identity_hash=event.signal_identity_hash,
                    intended_entry_price=event.entry_price,
                    forward_return_status="computed",
                    forward_return_attempts=0,
                )
                signal.forward_return = event.ret_next_open
                signal_id = signal.signal_id
                inserted_signal = True
            else:
                signal = existing_signal
                signal_id = existing_signal.signal_id
                reused_signal = True
        fro_inserted = False
        fro_reused = False
        if signal is not None:
            fro_inserted, fro_reused = self._record_i12_forward_return_observation(
                event,
                signal=signal,
                job_run_id=job_run_id,
            )

        detail = IntradayEventDetail(
            signal_id=signal_id,
            job_run_id=job_run_id,
            pattern_id=I12_PATTERN_ID,
            ticker=event.ticker,
            trading_date=event.trading_date,
            outcome=event.outcome,
            event_identity_hash=event.event_identity_hash,
            input_hash=event.input_hash,
            output_hash=event.output_hash,
            data_lineage_ids_json=json.dumps(list(event.data_lineage_ids)),
            gate_values_json=json.dumps(event.gate_values, sort_keys=True, default=str),
            feature_json=json.dumps(event.feature_json, sort_keys=True, default=str),
            label_json=json.dumps(event.labels, sort_keys=True, default=str),
            artifact_flags_json=json.dumps(event.artifact_flags, sort_keys=True, default=str),
            confirmation_timestamp=event.confirmation_timestamp,
            entry_timestamp=event.entry_timestamp,
            exit_timestamp=event.exit_timestamp,
            conf_minute=event.conf_minute,
            entry_minute=event.entry_minute,
            entry_price=event.entry_price,
            exit_price=event.exit_price,
            session_open_price=event.session_open_price,
            session_close_price=event.session_close_price,
            next_open_price=event.next_open_price,
            projected_vol_at_conf=event.projected_vol_at_conf,
            projected_vol_ratio_at_conf=event.projected_vol_ratio_at_conf,
            full_day_volume_ratio=event.full_day_volume_ratio,
            chase_pct=event.chase_pct,
            gap_pct=event.gap_pct,
            distance_from_max252=event.distance_from_max252,
            ret_conf=event.ret_conf,
            ret_open_close=event.ret_open_close,
            ret_open_close_leaky_research_only=event.ret_open_close is not None,
            ret_next_open=event.ret_next_open,
            mae_pct=event.mae_pct,
            mfe_pct=event.mfe_pct,
            halted=event.halted,
            sub_dollar_at_open=event.sub_dollar_at_open,
            split_basis_mismatch=event.split_basis_mismatch,
            is_ml_excluded=event.is_ml_excluded,
            ml_exclusion_reason=event.ml_exclusion_reason,
            security_type=event.security_type,
            sessions_to_delist=event.sessions_to_delist,
            sessions_to_delist_not_pit=True,
        )
        self._session.add(detail)
        self._session.flush()
        return {
            "inserted_detail": True,
            "inserted_signal": inserted_signal,
            "reused_signal": reused_signal,
            "inserted_forward_return_observation": fro_inserted,
            "reused_forward_return_observation": fro_reused,
        }

    def _record_i12_forward_return_observation(
        self,
        event: _I12Event,
        *,
        signal: SignalRegistry,
        job_run_id: str,
    ) -> tuple[bool, bool]:
        if event.ret_next_open is None:
            raise RuntimeError(
                "I12 trainable signal is missing primary ret_next_open label: "
                f"{event.ticker} {event.trading_date.isoformat()}"
            )
        input_hash = stable_hash({
            "signal_id": signal.signal_id,
            "label": "ret_next_open",
            "event_input_hash": event.input_hash,
            "reconstruction_method": RECONSTRUCTION_METHOD,
        })
        outcome_hash = stable_hash({
            "signal_id": signal.signal_id,
            "label": "ret_next_open",
            "ret_next_open": event.ret_next_open,
            "labels": event.labels,
        })
        existing = (
            self._session.query(ForwardReturnObservation)
            .filter(
                ForwardReturnObservation.signal_id == signal.signal_id,
                ForwardReturnObservation.input_hash == input_hash,
            )
            .one_or_none()
        )
        if existing is not None:
            if existing.outcome_hash != outcome_hash:
                raise RuntimeError(
                    "I12 forward return changed for existing signal: "
                    f"{event.ticker} {event.trading_date.isoformat()}"
                )
            return False, True
        signal_timestamp = event.signal_timestamp or event.confirmation_timestamp
        if signal_timestamp is None:
            raise RuntimeError(
                "I12 confirmed signal is missing signal timestamp: "
                f"{event.ticker} {event.trading_date.isoformat()}"
            )
        daily_lineage_id = event.data_lineage_ids[0] if event.data_lineage_ids else None
        minute_lineage_id = (
            event.data_lineage_ids[1]
            if len(event.data_lineage_ids) > 1 else daily_lineage_id
        )
        exit_session = next_us_equity_session(event.trading_date + timedelta(days=1))
        observation = ForwardReturnObservation(
            signal_id=signal.signal_id,
            pattern_id=I12_PATTERN_ID,
            ticker=event.ticker,
            direction="long",
            signal_timestamp=signal_timestamp,
            signal_horizon=I12_SIGNAL_HORIZON,
            next_execution_session=event.trading_date.isoformat(),
            entry_session_date=event.trading_date.isoformat(),
            entry_price=event.entry_price,
            entry_price_source="polygon_adjusted_next_minute_open",
            entry_basis_proof="next_minute_open_after_intraday_confirmation",
            entry_data_lineage_id=minute_lineage_id,
            exit_session_date=exit_session.isoformat(),
            exit_price=event.next_open_price,
            exit_price_source="fmp_split_adjusted_next_session_open",
            exit_basis_proof="next_session_open_label",
            exit_data_lineage_id=daily_lineage_id,
            forward_return=event.ret_next_open,
            max_favorable_excursion=event.mfe_pct,
            max_adverse_excursion=event.mae_pct,
            status="computed",
            reason=None,
            attempts=0,
            job_run_id=job_run_id,
            input_hash=input_hash,
            outcome_hash=outcome_hash,
            data_lineage_ids=json.dumps(list(event.data_lineage_ids)),
            provider="FMP+Polygon",
            endpoint=RECONSTRUCTION_METHOD,
            provider_request_json=json.dumps({
                "label": "ret_next_open",
                "label_json": event.labels,
            }, sort_keys=True, default=str),
        )
        self._session.add(observation)
        self._session.flush()
        return True, False

    def _metrics(self, counters: _RunCounters, *, trading_dates: Sequence[date]) -> dict[str, Any]:
        return {
            "pattern_id": I12_PATTERN_ID,
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "start_date": self._start_date.isoformat(),
            "end_date": self._end_date.isoformat(),
            "trading_date_count": len(trading_dates),
            "batch_count": counters.batches,
            "ticker_days_scanned": counters.ticker_days_scanned,
            "candidates": counters.candidates,
            "candidates_screened_out": counters.candidates_screened_out,
            "candidate_screen_fail_reasons": dict(counters.candidate_screen_fail_reasons),
            "confirmed": counters.confirmed,
            "never_confirmed": counters.never_confirmed,
            "poison_blocked": counters.poison_blocked,
            "poison_premarket": counters.poison_premarket,
            "parabolic_blocked": counters.parabolic_blocked,
            "halted_unfillable": counters.halted_unfillable,
            "outcome_counts": dict(counters.outcomes),
            "excluded_by_type": counters.excluded_by_type,
            "artifact_excluded": counters.artifact_excluded,
            "sub_dollar_included": counters.sub_dollar_included,
            "non_session_bars_skipped": counters.non_session_bars_skipped,
            "non_session_bar_skip_sample": counters.non_session_bar_samples,
            "fetch_errors": counters.fetch_errors,
            "watchdog_timeouts": counters.watchdog_timeouts,
            **self._fetch_watchdog.snapshot(),
            "quarantined": counters.quarantined,
            "inserted_details": counters.inserted_details,
            "reused_details": counters.reused_details,
            "inserted_signals": counters.inserted_signals,
            "reused_signals": counters.reused_signals,
            "forward_return_observations_inserted": counters.forward_return_observations_inserted,
            "forward_return_observations_reused": counters.forward_return_observations_reused,
            "minute_cache_hits": counters.minute_cache_hits,
            "minute_cache_misses": counters.minute_cache_misses,
            "skipped_existing": counters.skipped_existing,
            "db_reconnect_retries": counters.db_reconnect_retries,
            "error_sample": counters.errors[:20],
        }

    def _progress(self, event: str, payload: Mapping[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("wall_clock_utc", _utc_progress_timestamp())
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            self._latest_metrics = dict(metrics)
        if self._progress_callback is not None:
            self._progress_callback(event, payload)

    def _load_cached_polygon_minutes(
        self,
        ticker: str,
        trading_date: date,
    ) -> list[PolygonBar] | None:
        path = self._minute_cache_path(ticker, trading_date)
        if path is None or not path.exists():
            return None
        with path.open("r") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            return None
        return [PolygonBar(**row) for row in rows]

    def _store_cached_polygon_minutes(
        self,
        ticker: str,
        trading_date: date,
        bars: Sequence[PolygonBar],
    ) -> None:
        path = self._minute_cache_path(ticker, trading_date)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w") as f:
            json.dump([asdict(bar) for bar in bars], f, sort_keys=True)
        tmp_path.replace(path)

    def _minute_cache_path(self, ticker: str, trading_date: date) -> Path | None:
        if self._minute_cache_dir is None:
            return None
        normalized = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in ticker.upper()
        )
        return (
            self._minute_cache_dir
            / f"{trading_date.year:04d}"
            / f"{trading_date.month:02d}"
            / f"{normalized}_{trading_date.isoformat()}.json"
        )

    def _throttle_polygon_fetch(self) -> None:
        if not self._polygon_rate_limit_per_minute:
            return
        interval = 60.0 / float(self._polygon_rate_limit_per_minute)
        if interval <= 0:
            return
        now = time_module.monotonic()
        if self._last_polygon_fetch_monotonic is not None:
            sleep_seconds = self._last_polygon_fetch_monotonic + interval - now
            if sleep_seconds > 0:
                time_module.sleep(sleep_seconds)
        self._last_polygon_fetch_monotonic = time_module.monotonic()


class _Quarantine(RuntimeError):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(str(payload))
        self.payload = payload


def _utc_progress_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _daily_context_from_bars(
    ticker: str,
    trading_date: date,
    daily_bars: Sequence[_DailyBar],
) -> _DailyContext:
    daily_by_date = {bar.date: bar for bar in daily_bars}
    day_bar = daily_by_date.get(trading_date)
    if day_bar is None:
        raise _Quarantine({
            "ticker": ticker,
            "trading_date": trading_date.isoformat(),
            "error": "missing_day0_daily_bar",
        })
    prior = [bar for bar in daily_bars if bar.date < trading_date]
    if len(prior) < MIN_PRIOR_DAILY_SESSIONS:
        raise _Quarantine({
            "ticker": ticker,
            "trading_date": trading_date.isoformat(),
            "error": "insufficient_prior_daily_history",
            "prior_count": len(prior),
            "min_required": MIN_PRIOR_DAILY_SESSIONS,
        })
    prior20 = prior[-20:]
    prior252 = prior[-252:]
    prior_close = prior[-1].split_adjusted_close
    max_prior = max(bar.split_adjusted_close for bar in prior252) if prior252 else prior_close
    avg20_volume = sum(bar.volume for bar in prior20) / 20.0
    mom20 = (
        prior_close / prior[-21].split_adjusted_close - 1.0
        if len(prior) >= 21 and prior[-21].split_adjusted_close > 0
        else None
    )
    adjusted_lows = [
        _split_adjusted_low(bar)
        for bar in prior252
        if _split_adjusted_low(bar) is not None
    ]
    low252 = min(adjusted_lows) if adjusted_lows else None
    off_low252 = _safe_return(prior_close, low252) if low252 else None
    if max_prior <= 0 or avg20_volume <= 0:
        raise _Quarantine({
            "ticker": ticker,
            "trading_date": trading_date.isoformat(),
            "error": "invalid_daily_context",
        })
    distance = prior_close / max_prior - 1.0
    gap = day_bar.open / prior_close - 1.0
    full_day_vr = _safe_ratio(day_bar.volume, avg20_volume)
    return _DailyContext(
        context=PremarketContext(
            ticker=ticker.upper(),
            context_date=trading_date,
            prior_close=prior_close,
            max_prior_252_closes=max_prior,
            avg20_volume=avg20_volume,
            mom20=mom20,
            off_low252=off_low252,
        ),
        day_bar=day_bar,
        prior_count=len(prior),
        distance_from_max252=distance,
        gap=gap,
        full_day_volume_ratio=full_day_vr,
    )


def _screen_daily_candidate(inp: _DailyTickerInput) -> _CandidateScreenResult:
    daily_context = _daily_context_from_bars(inp.ticker, inp.trading_date, inp.daily_bars)
    if daily_context.distance_from_max252 > -0.50:
        return _CandidateScreenResult(False, "drawdown_screen", daily_context)
    if daily_context.gap >= 0.05:
        return _CandidateScreenResult(False, "gap_screen_i1_exclusion", daily_context)
    if (
        daily_context.full_day_volume_ratio is None
        or daily_context.full_day_volume_ratio < CANDIDATE_FULL_DAY_VR_FLOOR
    ):
        return _CandidateScreenResult(False, "full_day_volume_screen", daily_context)
    if daily_context.gap < -0.05 - BOUNDARY_EPSILON:
        return _CandidateScreenResult(
            False,
            None,
            daily_context,
            control_outcome=OUTCOME_POISON_PREMARKET,
        )
    return _CandidateScreenResult(True, None, daily_context)


def _evaluate_i12_poison_premarket(
    inp: _DailyTickerInput,
    daily_context: _DailyContext,
) -> _I12Event:
    daily_by_date = {bar.date: bar for bar in inp.daily_bars}
    context = daily_context.context
    day_bar = daily_context.day_bar
    sub_dollar = day_bar.open < 1.0
    full_day_volume_ratio = daily_context.full_day_volume_ratio
    next_session = next_us_equity_session(inp.trading_date + timedelta(days=1))
    next_open_price = (
        daily_by_date[next_session].open
        if next_session in daily_by_date else None
    )
    ret_open_close = _safe_return(day_bar.close, day_bar.open)
    ret_next_open = _safe_return(next_open_price, day_bar.open)
    mae_pct = _safe_return(day_bar.low, day_bar.open)
    mfe_pct = _safe_return(day_bar.high, day_bar.open)
    gate_values = {
        "gap": daily_context.gap,
        "distance_from_max252": daily_context.distance_from_max252,
        "full_day_volume_ratio": full_day_volume_ratio,
        "full_day_volume_ratio_leaky_research_only": full_day_volume_ratio is not None,
        "candidate_screen": dict(CANDIDATE_SCREEN_STAMP),
    }
    labels = {
        "ret_conf": None,
        "ret_open_close": ret_open_close,
        "ret_open_close_leaky_research_only": ret_open_close is not None,
        "ret_next_open": ret_next_open,
        "mae_pct": mae_pct,
        "mfe_pct": mfe_pct,
        "full_day_volume_ratio": full_day_volume_ratio,
        "full_day_volume_ratio_leaky_research_only": full_day_volume_ratio is not None,
        "sessions_to_delist": inp.sessions_to_delist,
        "sessions_to_delist_not_pit": True,
    }
    artifact_flags = {
        "split_basis_mismatch": False,
        "split_basis_mismatch_excluded": False,
        "sub_dollar_at_open": sub_dollar,
        "full_day_volume_ratio_leaky_research_only": full_day_volume_ratio is not None,
        "sessions_to_delist_not_pit": True,
    }
    feature_json = _feature_payload(
        inp=inp,
        context=context,
        day_bar=day_bar,
        gate_values=gate_values,
        projected_vol_at_conf=None,
        projected_vr_at_conf=None,
        full_day_volume_ratio=full_day_volume_ratio,
        chase=None,
        gap=daily_context.gap,
        sub_dollar=sub_dollar,
    )
    input_payload = {
        "pattern_id": I12_PATTERN_ID,
        "ticker": inp.ticker,
        "trading_date": inp.trading_date.isoformat(),
        "daily_lineage": inp.daily_lineage.raw_payload_hash,
        "control_outcome": OUTCOME_POISON_PREMARKET,
        "reconstruction_method": RECONSTRUCTION_METHOD,
    }
    input_hash = stable_hash(input_payload)
    output_payload = {
        "outcome": OUTCOME_POISON_PREMARKET,
        "features": feature_json,
        "labels": labels,
        "artifact_flags": artifact_flags,
        "signal_identity_hash": None,
    }
    output_hash = stable_hash(output_payload)
    event_identity_hash = stable_hash({
        "pattern_id": I12_PATTERN_ID,
        "ticker": inp.ticker,
        "trading_date": inp.trading_date.isoformat(),
        "reconstruction_method": RECONSTRUCTION_METHOD,
    })
    return _I12Event(
        ticker=inp.ticker,
        trading_date=inp.trading_date,
        outcome=OUTCOME_POISON_PREMARKET,
        gate_values=gate_values,
        feature_json=feature_json,
        labels=labels,
        artifact_flags=artifact_flags,
        signal_timestamp=None,
        confirmation_timestamp=None,
        entry_timestamp=None,
        exit_timestamp=None,
        conf_minute=None,
        entry_minute=None,
        entry_price=None,
        exit_price=None,
        session_open_price=day_bar.open,
        session_close_price=day_bar.close,
        next_open_price=next_open_price,
        projected_vol_at_conf=None,
        projected_vol_ratio_at_conf=None,
        full_day_volume_ratio=full_day_volume_ratio,
        chase_pct=None,
        gap_pct=daily_context.gap,
        distance_from_max252=daily_context.distance_from_max252,
        ret_conf=None,
        ret_open_close=ret_open_close,
        ret_next_open=ret_next_open,
        mae_pct=mae_pct,
        mfe_pct=mfe_pct,
        halted=False,
        sub_dollar_at_open=sub_dollar,
        split_basis_mismatch=False,
        is_ml_excluded=inp.security_type.ml_excluded,
        ml_exclusion_reason=inp.security_type.reason,
        security_type=inp.security_type.security_type,
        sessions_to_delist=inp.sessions_to_delist,
        data_lineage_ids=(inp.daily_lineage.data_lineage_id,),
        input_hash=input_hash,
        output_hash=output_hash,
        event_identity_hash=event_identity_hash,
        signal_identity_hash=None,
    )


def _evaluate_i12_event(inp: _TickerDayInput) -> _I12Event:
    daily_by_date = {bar.date: bar for bar in inp.daily_bars}
    prior = [bar for bar in inp.daily_bars if bar.date < inp.trading_date]
    daily_context = _daily_context_from_bars(inp.ticker, inp.trading_date, inp.daily_bars)
    context = daily_context.context
    day_bar = daily_context.day_bar
    first_minute = inp.minute_bars[0]
    split_basis_mismatch = _basis_mismatch(day_bar.open, first_minute.open)
    sub_dollar = day_bar.open < 1.0
    full_day_volume_ratio = daily_context.full_day_volume_ratio
    session_close_price = day_bar.close
    next_session = next_us_equity_session(inp.trading_date + timedelta(days=1))
    next_open_price = (
        daily_by_date[next_session].open
        if next_session in daily_by_date else None
    )
    ret_open_close = _safe_return(session_close_price, day_bar.open)
    cumulative_volume = 0.0
    cumulative_high: float | None = None
    cumulative_low: float | None = None
    latest_gate_values: dict[str, Any] = {}
    max_projected_vol = None
    max_projected_vr = None
    event_outcome = OUTCOME_NEVER_CONFIRMED
    confirm_index: int | None = None
    confirm_shared = None
    confirm_timestamp: datetime | None = None

    for index, minute in enumerate(inp.minute_bars):
        if minute.minute_index < 0:
            continue
        if minute.minute_index > I12_CONFIRMATION_MAX_MINUTE:
            break
        cumulative_volume += minute.volume
        cumulative_high = minute.high if cumulative_high is None else max(cumulative_high, minute.high)
        cumulative_low = minute.low if cumulative_low is None else min(cumulative_low, minute.low)
        snapshot = PolygonSnapshotTicker(
            ticker=inp.ticker,
            day_open=day_bar.open,
            day_high=cumulative_high,
            day_low=cumulative_low,
            day_close=minute.close,
            day_volume=cumulative_volume,
            minute_timestamp=int(minute.timestamp.timestamp() * 1000),
            minute_open=minute.open,
            minute_high=minute.high,
            minute_low=minute.low,
            minute_close=minute.close,
            minute_volume=minute.volume,
        )
        shared = compute_shared_intraday_math(
            context,
            snapshot,
            trading_date=inp.trading_date,
        )
        if shared is None:
            continue
        max_projected_vol = max(max_projected_vol or 0.0, shared.projected_vol)
        max_projected_vr = max(max_projected_vr or 0.0, shared.vol_ratio)
        decision = i12_entry_gate(context, snapshot, shared)
        latest_gate_values = dict(decision.gate_values)
        if decision.reason == "poison_blocked":
            event_outcome = OUTCOME_POISON
            confirm_index = index
            confirm_shared = shared
            confirm_timestamp = minute.timestamp
            break
        if decision.reason == "parabolic_skipped":
            event_outcome = OUTCOME_PARABOLIC
            confirm_index = index
            confirm_shared = shared
            confirm_timestamp = minute.timestamp
            break
        if decision.enter:
            event_outcome = OUTCOME_CONFIRMED
            confirm_index = index
            confirm_shared = shared
            confirm_timestamp = minute.timestamp
            break

    entry_bar: _MinuteBar | None = None
    entry_timestamp: datetime | None = None
    entry_price: float | None = None
    entry_minute: int | None = None
    exit_bar = _exit_bar(inp.minute_bars)
    exit_price = exit_bar.close if exit_bar is not None else None
    exit_timestamp = exit_bar.timestamp if exit_bar is not None else None
    mae_pct = None
    mfe_pct = None
    ret_conf = None
    ret_next_open = None
    halted = False
    if event_outcome == OUTCOME_CONFIRMED:
        entry_bar = _next_minute_bar(inp.minute_bars, confirm_index)
        if entry_bar is None:
            event_outcome = OUTCOME_HALTED_UNFILLABLE
            halted = True
        else:
            entry_timestamp = entry_bar.timestamp
            entry_price = entry_bar.open
            entry_minute = entry_bar.minute_index
            if exit_price is not None:
                ret_conf = _safe_return(exit_price, entry_price)
            if next_open_price is not None:
                ret_next_open = _safe_return(next_open_price, entry_price)
            path = [
                bar for bar in inp.minute_bars
                if bar.minute_index >= entry_bar.minute_index
                and bar.timestamp <= (exit_timestamp or bar.timestamp)
            ]
            if path:
                mae_pct = _safe_return(min(bar.low for bar in path), entry_price)
                mfe_pct = _safe_return(max(bar.high for bar in path), entry_price)

    projected_vol_at_conf = (
        confirm_shared.projected_vol if confirm_shared is not None else max_projected_vol
    )
    projected_vr_at_conf = (
        confirm_shared.vol_ratio if confirm_shared is not None else max_projected_vr
    )
    chase = confirm_shared.chase if confirm_shared is not None else latest_gate_values.get("chase")
    gap = confirm_shared.gap if confirm_shared is not None else latest_gate_values.get("gap")
    distance = latest_gate_values.get("distance_from_max252")
    labels = {
        "ret_conf": ret_conf,
        "ret_open_close": ret_open_close,
        "ret_open_close_leaky_research_only": ret_open_close is not None,
        "ret_next_open": ret_next_open,
        "mae_pct": mae_pct,
        "mfe_pct": mfe_pct,
        "full_day_volume_ratio": full_day_volume_ratio,
        "full_day_volume_ratio_leaky_research_only": full_day_volume_ratio is not None,
        "sessions_to_delist": inp.sessions_to_delist,
        "sessions_to_delist_not_pit": True,
    }
    artifact_flags = {
        "split_basis_mismatch": split_basis_mismatch,
        "split_basis_mismatch_excluded": split_basis_mismatch,
        "primary_label_unavailable": (
            event_outcome == OUTCOME_CONFIRMED and ret_next_open is None
        ),
        "primary_label_unavailable_excluded": (
            event_outcome == OUTCOME_CONFIRMED and ret_next_open is None
        ),
        "sub_dollar_at_open": sub_dollar,
        "full_day_volume_ratio_leaky_research_only": full_day_volume_ratio is not None,
        "sessions_to_delist_not_pit": True,
    }
    event_is_ml_excluded, event_ml_exclusion_reason = _ml_exclusion_state(
        inp.security_type,
        split_basis_mismatch=split_basis_mismatch,
        primary_label_unavailable=bool(artifact_flags["primary_label_unavailable"]),
    )
    latest_gate_values = dict(latest_gate_values)
    latest_gate_values["full_day_volume_ratio"] = full_day_volume_ratio
    latest_gate_values["full_day_volume_ratio_leaky_research_only"] = (
        full_day_volume_ratio is not None
    )
    feature_json = _feature_payload(
        inp=inp,
        context=context,
        day_bar=day_bar,
        gate_values=latest_gate_values,
        projected_vol_at_conf=projected_vol_at_conf,
        projected_vr_at_conf=projected_vr_at_conf,
        full_day_volume_ratio=full_day_volume_ratio,
        chase=chase,
        gap=gap,
        sub_dollar=sub_dollar,
    )
    feature_json["is_ml_excluded"] = event_is_ml_excluded
    feature_json["ml_exclusion_reason"] = event_ml_exclusion_reason
    input_payload = {
        "pattern_id": I12_PATTERN_ID,
        "ticker": inp.ticker,
        "trading_date": inp.trading_date.isoformat(),
        "daily_lineage": inp.daily_lineage.raw_payload_hash,
        "minute_lineage": inp.minute_lineage.raw_payload_hash,
        "reconstruction_method": RECONSTRUCTION_METHOD,
    }
    input_hash = stable_hash(input_payload)
    signal_identity_hash = (
        stable_hash({
            "pattern_id": I12_PATTERN_ID,
            "ticker": inp.ticker,
            "trading_date": inp.trading_date.isoformat(),
            "outcome": OUTCOME_CONFIRMED,
            "reconstruction_method": RECONSTRUCTION_METHOD,
        })
        if event_outcome == OUTCOME_CONFIRMED and not event_is_ml_excluded else None
    )
    output_payload = {
        "outcome": event_outcome,
        "features": feature_json,
        "labels": labels,
        "artifact_flags": artifact_flags,
        "signal_identity_hash": signal_identity_hash,
    }
    output_hash = stable_hash(output_payload)
    event_identity_hash = stable_hash({
        "pattern_id": I12_PATTERN_ID,
        "ticker": inp.ticker,
        "trading_date": inp.trading_date.isoformat(),
        "reconstruction_method": RECONSTRUCTION_METHOD,
    })
    return _I12Event(
        ticker=inp.ticker,
        trading_date=inp.trading_date,
        outcome=event_outcome,
        gate_values=latest_gate_values,
        feature_json=feature_json,
        labels=labels,
        artifact_flags=artifact_flags,
        signal_timestamp=confirm_timestamp,
        confirmation_timestamp=confirm_timestamp,
        entry_timestamp=entry_timestamp,
        exit_timestamp=exit_timestamp,
        conf_minute=int(confirm_shared.data_elapsed_min) if confirm_shared else None,
        entry_minute=entry_minute,
        entry_price=entry_price,
        exit_price=exit_price,
        session_open_price=day_bar.open,
        session_close_price=session_close_price,
        next_open_price=next_open_price,
        projected_vol_at_conf=projected_vol_at_conf,
        projected_vol_ratio_at_conf=projected_vr_at_conf,
        full_day_volume_ratio=full_day_volume_ratio,
        chase_pct=chase,
        gap_pct=gap,
        distance_from_max252=distance,
        ret_conf=ret_conf,
        ret_open_close=ret_open_close,
        ret_next_open=ret_next_open,
        mae_pct=mae_pct,
        mfe_pct=mfe_pct,
        halted=halted,
        sub_dollar_at_open=sub_dollar,
        split_basis_mismatch=split_basis_mismatch,
        is_ml_excluded=event_is_ml_excluded,
        ml_exclusion_reason=event_ml_exclusion_reason,
        security_type=inp.security_type.security_type,
        sessions_to_delist=inp.sessions_to_delist,
        data_lineage_ids=(inp.daily_lineage.data_lineage_id, inp.minute_lineage.data_lineage_id),
        input_hash=input_hash,
        output_hash=output_hash,
        event_identity_hash=event_identity_hash,
        signal_identity_hash=signal_identity_hash,
    )


def _feature_payload(
    *,
    inp: _TickerDayInput | _DailyTickerInput,
    context: PremarketContext,
    day_bar: _DailyBar,
    gate_values: Mapping[str, Any],
    projected_vol_at_conf: float | None,
    projected_vr_at_conf: float | None,
    full_day_volume_ratio: float | None,
    chase: float | None,
    gap: float | None,
    sub_dollar: bool,
) -> dict[str, Any]:
    prior = [bar for bar in inp.daily_bars if bar.date < inp.trading_date]
    prior20 = prior[-20:]
    prior252 = prior[-252:]
    spy_prior_day_return = None
    minute_lineage = getattr(inp, "minute_lineage", None)
    source_lineage = {
        "daily_bar_lineage_id": inp.daily_lineage.data_lineage_id,
        "daily_bar_lineage_hash": inp.daily_lineage.raw_payload_hash,
    }
    if minute_lineage is not None:
        source_lineage.update({
            "minute_bar_lineage_id": minute_lineage.data_lineage_id,
            "minute_bar_lineage_hash": minute_lineage.raw_payload_hash,
        })
    payload = {
        "reconstructed": True,
        "reconstruction_method": RECONSTRUCTION_METHOD,
        "pattern_id": I12_PATTERN_ID,
        "ticker": inp.ticker,
        "trading_date": inp.trading_date.isoformat(),
        "price_basis": "fmp_full_split_adjusted_close",
        "candidate_screen": dict(CANDIDATE_SCREEN_STAMP),
        "lookback_end": prior[-1].date.isoformat() if prior else None,
        "drawdown_from_max252": context.prior_close / context.max_prior_252_closes - 1.0,
        "distance_from_max252": context.prior_close / context.max_prior_252_closes - 1.0,
        "projected_volume_ratio_at_confirmation": projected_vr_at_conf,
        "projected_volume_at_confirmation": projected_vol_at_conf,
        "gap": gap,
        "prev_day_return": _safe_return(prior[-1].split_adjusted_close, prior[-2].split_adjusted_close)
        if len(prior) >= 2 else None,
        "prev_day_green": prior[-1].close > prior[-1].open if prior else None,
        "mom20": context.mom20,
        "off_low252": context.off_low252,
        "sigma20": _sigma(prior20),
        "sub_dollar_at_open": sub_dollar,
        "halt_state": "unobserved_or_no_halt",
        "catalyst_tags": [],
        "catalyst_source_status": "not_implemented_pit_safe_empty",
        "spy_prior_day_return": spy_prior_day_return,
        "security_type": inp.security_type.security_type,
        "is_ml_excluded": inp.security_type.ml_excluded,
        "ml_exclusion_reason": inp.security_type.reason,
        "gate_values": _without_none(_feature_gate_values(gate_values)),
        "research_only_leaky": _without_none(
            {
                "avg20_volume": context.avg20_volume,
                "dollar_volume": day_bar.close * day_bar.volume,
                "price": day_bar.open,
            }
        ),
        "source_lineage": source_lineage,
        "pit_caveats": {
            "security_type": "classified_asof_profile_applied_retroactively",
            "research_only_leaky": "excluded_from_stage1_feature_allowlists",
        },
    }
    if minute_lineage is not None:
        payload["minute_price_basis"] = "polygon_adjusted_minute"
    if len(prior252) < 252:
        payload["insufficient_history"] = {"prior252": True}
    if chase is not None:
        payload["chase_pct"] = chase
    return payload


def _ml_exclusion_state(
    security_type: SecurityTypeClassification,
    *,
    split_basis_mismatch: bool,
    primary_label_unavailable: bool,
) -> tuple[bool, str]:
    if split_basis_mismatch:
        return True, ML_EXCLUSION_SPLIT_BASIS_MISMATCH
    if primary_label_unavailable:
        return True, ML_EXCLUSION_PRIMARY_LABEL_UNAVAILABLE
    if security_type.ml_excluded:
        return True, security_type.reason
    return False, security_type.reason


def _clean_daily_bars(
    ticker: str,
    bars: Sequence[FmpBar],
    counters: _RunCounters,
) -> list[_DailyBar]:
    clean: list[_DailyBar] = []
    seen: set[date] = set()
    for bar in bars:
        try:
            parsed_date = date.fromisoformat(str(bar.date)[:10])
        except ValueError:
            continue
        if not is_us_equity_session(parsed_date):
            counters.record_non_session(ticker, parsed_date)
            continue
        if parsed_date in seen:
            continue
        values = (bar.open, bar.high, bar.low, bar.close, bar.split_adjusted_close)
        if any(value is None or not math.isfinite(float(value)) or float(value) <= 0 for value in values):
            continue
        if bar.volume is None or float(bar.volume) < 0:
            continue
        clean.append(_DailyBar(
            date=parsed_date,
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
            split_adjusted_close=float(bar.split_adjusted_close),
        ))
        seen.add(parsed_date)
    return sorted(clean, key=lambda item: item.date)


def _clean_minute_bars(trading_date: date, bars: Sequence[PolygonBar]) -> list[_MinuteBar]:
    market_open = us_equity_session_open_timestamp(trading_date).astimezone(EASTERN)
    market_close = us_equity_session_close_timestamp(trading_date).astimezone(EASTERN)
    clean: list[_MinuteBar] = []
    for bar in bars:
        ts = datetime.fromtimestamp(float(bar.timestamp) / 1000.0, timezone.utc)
        ts_et = ts.astimezone(EASTERN)
        if ts_et.date() != trading_date or ts_et < market_open or ts_et > market_close:
            continue
        values = (bar.open, bar.high, bar.low, bar.close)
        if any(value is None or not math.isfinite(float(value)) or float(value) <= 0 for value in values):
            continue
        if bar.volume is None or float(bar.volume) < 0:
            continue
        minute_index = int((ts_et - market_open).total_seconds() // 60)
        clean.append(_MinuteBar(
            timestamp=ts,
            minute_index=minute_index,
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
        ))
    return sorted(clean, key=lambda item: item.timestamp)


def _record_bars_lineage(
    session: Session,
    *,
    provider: str,
    endpoint: str,
    ticker: str,
    trading_date: date,
    bars: list[dict[str, Any]],
    job_run_id: str,
    run_timestamp: datetime,
) -> DataLineage:
    payload = {
        "ticker": ticker,
        "trading_date": trading_date.isoformat(),
        "bars": bars,
    }
    raw_payload_hash = stable_hash(payload)
    data_quality_flags = {
        "bar_count": len(bars),
        "bars_digest": raw_payload_hash,
    }
    existing = (
        session.query(DataLineage)
        .filter(
            DataLineage.provider == provider,
            DataLineage.endpoint == endpoint,
            DataLineage.raw_payload_hash == raw_payload_hash,
        )
        .order_by(DataLineage.data_lineage_id.asc())
        .first()
    )
    if existing is not None:
        existing.raw_payload_json = None
        existing.data_quality_flags = json.dumps(data_quality_flags, sort_keys=True)
        return existing
    return record_data_lineage(
        session,
        provider=provider,
        endpoint=endpoint,
        request_timestamp=run_timestamp,
        asof_timestamp=run_timestamp,
        raw_payload_hash=raw_payload_hash,
        raw_payload=None,
        data_quality_flags=data_quality_flags,
        job_run_id=job_run_id,
    )


def _daily_payload(bar: _DailyBar) -> dict[str, Any]:
    return {
        "date": bar.date.isoformat(),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
        "split_adjusted_close": bar.split_adjusted_close,
    }


def _minute_payload(bar: _MinuteBar) -> dict[str, Any]:
    return {
        "timestamp": bar.timestamp.isoformat(),
        "minute_index": bar.minute_index,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
    }


def _trading_dates(start: date, end: date) -> list[date]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        if is_us_equity_session(cursor):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _chunks(values: Sequence[date], size: int) -> list[Sequence[date]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def _classification_for(
    classifications: Mapping[str, SecurityTypeClassification],
    ticker: str,
) -> SecurityTypeClassification:
    normalized = ticker.upper()
    if normalized not in classifications:
        raise ExclusionArtifactError(
            f"ticker {normalized!r} is not covered by the I12 exclusion artifact"
        )
    return classifications[normalized]


def _classification_for_hur_ticker(
    classifications: Mapping[str, SecurityTypeClassification],
    ticker: str,
    *,
    hur_included: bool,
) -> SecurityTypeClassification:
    if not hur_included:
        raise RuntimeError(
            "i12_non_hur_ticker_reached_security_type_lookup: "
            f"ticker={ticker.upper()}"
        )
    return _classification_for(classifications, ticker)


def _last_us_equity_session_on_or_before(day: date) -> date:
    cursor = day
    while not is_us_equity_session(cursor):
        cursor -= timedelta(days=1)
    return cursor


def _provider_error_payload(resp: AdapterResponse[Any]) -> dict[str, Any] | None:
    if resp.error is None:
        return None
    return {
        "provider": resp.error.provider,
        "endpoint": resp.error.endpoint,
        "status_code": resp.error.status_code,
        "error_type": resp.error.error_type,
        "message": resp.error.message,
        "retryable": resp.error.retryable,
    }


def _is_transient_db_error(exc: BaseException) -> bool:
    if isinstance(exc, DBAPIError) and getattr(exc, "connection_invalidated", False):
        return True
    if isinstance(exc, OperationalError):
        return _looks_like_transient_disconnect(exc)
    return _looks_like_transient_disconnect(exc)


def _looks_like_transient_disconnect(exc: BaseException) -> bool:
    parts = [str(exc), exc.__class__.__name__]
    original = getattr(exc, "orig", None)
    if original is not None:
        parts.extend([str(original), original.__class__.__name__])
    text = " ".join(parts).lower()
    return any(
        marker in text
        for marker in (
            "adminshutdown",
            "administrator command",
            "server closed the connection",
            "connection is closed",
            "terminating connection",
            "connection not open",
            "connection already closed",
            "ssl connection has been closed",
        )
    )


def _transient_error_summary(exc: BaseException) -> str:
    original = getattr(exc, "orig", None)
    if original is not None:
        return f"{original.__class__.__name__}: {original}"
    return f"{exc.__class__.__name__}: {exc}"


def _basis_mismatch(fmp_open: float, polygon_open: float) -> bool:
    if fmp_open <= 0:
        return True
    return abs(polygon_open / fmp_open - 1.0) > SPLIT_BASIS_OPEN_TOLERANCE_PCT


def _split_adjusted_low(bar: _DailyBar) -> float | None:
    if bar.close <= 0 or bar.low <= 0 or bar.split_adjusted_close <= 0:
        return None
    return bar.low * (bar.split_adjusted_close / bar.close)


def _exit_bar(minutes: Sequence[_MinuteBar]) -> _MinuteBar | None:
    eligible = [
        bar for bar in minutes
        if bar.timestamp.astimezone(EASTERN).time() <= SESSION_EXIT_TIME
    ]
    return eligible[-1] if eligible else None


def _next_minute_bar(minutes: Sequence[_MinuteBar], confirm_index: int | None) -> _MinuteBar | None:
    if confirm_index is None:
        return None
    if confirm_index + 1 >= len(minutes):
        return None
    return minutes[confirm_index + 1]


def _safe_ratio(value: float | None, basis: float | None) -> float | None:
    if value is None or basis is None or basis <= 0:
        return None
    return value / basis


def _safe_return(value: float | None, basis: float | None) -> float | None:
    ratio = _safe_ratio(value, basis)
    return ratio - 1.0 if ratio is not None else None


def _sigma(bars: Sequence[_DailyBar]) -> float | None:
    returns = [
        current.split_adjusted_close / previous.split_adjusted_close - 1.0
        for previous, current in zip(bars[:-1], bars[1:])
        if previous.split_adjusted_close > 0
    ]
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((item - mean) ** 2 for item in returns) / (len(returns) - 1)
    return math.sqrt(variance)


def _without_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _feature_gate_values(gate_values: Mapping[str, Any]) -> dict[str, Any]:
    blocked = {
        "candidate_screen",
        "full_day_volume_ratio",
        "full_day_volume_ratio_leaky_research_only",
    }
    return {key: value for key, value in gate_values.items() if key not in blocked}
