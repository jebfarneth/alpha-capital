"""Durable historical I11 intraday corpus builder.

This is an I-track research corpus path, not the canonical detector stack.
It reuses the I12 corpus persistence contract while replaying the frozen I11
live gate over HUR-included historical membership. Confirmed entries persist
to ``signal_registry`` with the next-session open return as the official
economics; all lifecycle rows, including positive controls, persist to
``intraday_event_details``.

The job is intentionally scratch-first. It does not rebuild the universe and
does not write to ``signal_registry`` for controls. Feature payloads are
restricted to D-1/session-context and confirmation-time fields; richer labels,
post-hoc delist proximity, and research-only control anatomy live in detail
JSON. Security-type classification intentionally reuses the M4-labeled
exclusion artifact extended over HUR-included symbols, matching the I12 corpus
contract. A non-HUR ticker reaching classification is a job invariant failure.
"""

from __future__ import annotations

import json
from copy import deepcopy
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from sqlalchemy.orm import Session

from alpha.data.contracts import stable_hash, utcnow
from alpha.db.models import (
    ForwardReturnObservation,
    IntradayEventDetail,
    SignalRegistry,
)
from alpha.evidence.writer import record_feature_snapshot, record_signal
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.watchdog import ProviderOutageCircuitBreaker
from alpha.jobs.i12_historical_corpus import (
    DEFAULT_FETCH_DEADLINE_SECONDS,
    DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS,
    MIN_PRIOR_DAILY_SESSIONS,
    OUTCOME_CONFIRMED,
    OUTCOME_HALTED_UNFILLABLE,
    PROGRESS_HEARTBEAT_EVERY_TICKER_DAYS,
    _basis_mismatch,
    _classification_for_hur_ticker,
    _DailyBar,
    _daily_context_from_bars,
    _DailyContext,
    _DailyTickerInput,
    _exit_bar,
    _feature_gate_values,
    _I12Event,
    I12HistoricalCorpusJob,
    _MinuteBar,
    _next_minute_bar,
    _Quarantine,
    _safe_return,
    _sigma,
    _TickerDayInput,
    _trading_dates,
    _chunks,
    _without_none,
)
from alpha.jobs.paper_execution import (
    I11_PATTERN_ID,
    PremarketContext,
    PolygonSnapshotTicker,
    compute_shared_intraday_math,
    i11_entry_gate,
)
from alpha.market_calendar import us_equity_session_open_timestamp
from alpha.ml.security_type_exclusions import (
    ExclusionArtifactError,
    SecurityTypeClassification,
    load_classifications,
)


JOB_NAME = "i11_historical_corpus"
RECONSTRUCTION_METHOD = "historical_i11_replay_polygon_minute_fmp_eod_v1"
FEATURE_MANIFEST_VERSION = "i11_historical_corpus_v1"
AT_OPEN_RECONSTRUCTION_METHOD = "historical_i11_at_open_replay_polygon_minute_fmp_eod_v1"
AT_OPEN_FEATURE_MANIFEST_VERSION = "i11_at_open_historical_corpus_v1"
I11_SIGNAL_HORIZON = "1d"
OUTCOME_NEVER_CONFIRMED = "never_confirmed"
OUTCOME_FAILED_TEST = "failed_test"
OUTCOME_AT_OPEN_CROSSED = "at_open_crossed_high"
OUTCOME_AT_OPEN_NO_CROSS = "at_open_no_cross"
AT_OPEN_DISTANCE_BAND = (-0.05, 0.05)
CANDIDATE_SCREEN_STAMP = {
    "fresh_cross_basis": "prior_close_lte_prior_252_session_high",
    "day_high_crosses_hb": True,
    "selection_uses_day_high": True,
    "caveat": "daily_high_cross_screen_is_candidate_conditioning_for_fetch_tractability",
}
AT_OPEN_CANDIDATE_SCREEN_STAMP = {
    "open_distance_from_prior_252_high_band": list(AT_OPEN_DISTANCE_BAND),
    "prior_close_distance_from_prior_252_high_band": list(AT_OPEN_DISTANCE_BAND),
    "current_crossers_ignore_upper_band": True,
    "screen_basis": "open_or_prior_close_near_prior_252_session_high",
    "crossers_and_non_crossers_retained": True,
    "caveat": "row_population_is_conditioned_on_near_high_at_open_not_a_full_no_fire_cohort",
}
AT_OPEN_FORBIDDEN_LOOKAHEAD_FEATURES = frozenset({
    "first_1min_volume",
    "first_5min_volume",
    "opening_relative_volume",
    "opening_range_high",
    "opening_range_low",
})


@dataclass
class _RunCounters:
    ticker_days_scanned: int = 0
    candidates: int = 0
    candidates_screened_out: int = 0
    confirmed: int = 0
    never_confirmed: int = 0
    failed_test: int = 0
    halted_unfillable: int = 0
    excluded_by_type: int = 0
    artifact_excluded: int = 0
    sub_dollar_included: int = 0
    primary_label_unavailable: int = 0
    non_session_bars_skipped: int = 0
    fetch_errors: int = 0
    watchdog_timeouts: int = 0
    quarantined: int = 0
    inserted_details: int = 0
    reused_details: int = 0
    inserted_signals: int = 0
    reused_signals: int = 0
    minute_cache_hits: int = 0
    minute_cache_misses: int = 0
    skipped_existing: int = 0
    db_reconnect_retries: int = 0
    batches: int = 0
    at_open_added_candidates: int = 0
    at_open_crosser_screen_candidates: int = 0
    forward_return_observations_inserted: int = 0
    forward_return_observations_reused: int = 0
    at_open_missing_minute_bars_by_bucket: Counter[str] = field(default_factory=Counter)
    errors: list[dict[str, Any]] = field(default_factory=list)
    non_session_bar_samples: list[dict[str, str]] = field(default_factory=list)
    candidate_screen_fail_reasons: Counter[str] = field(default_factory=Counter)
    outcomes: Counter[str] = field(default_factory=Counter)
    outcomes_by_year: Counter[str] = field(default_factory=Counter)

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
        self.failed_test += other.failed_test
        self.halted_unfillable += other.halted_unfillable
        self.excluded_by_type += other.excluded_by_type
        self.artifact_excluded += other.artifact_excluded
        self.sub_dollar_included += other.sub_dollar_included
        self.primary_label_unavailable += other.primary_label_unavailable
        self.non_session_bars_skipped += other.non_session_bars_skipped
        self.fetch_errors += other.fetch_errors
        self.watchdog_timeouts += other.watchdog_timeouts
        self.quarantined += other.quarantined
        self.inserted_details += other.inserted_details
        self.reused_details += other.reused_details
        self.inserted_signals += other.inserted_signals
        self.reused_signals += other.reused_signals
        self.minute_cache_hits += other.minute_cache_hits
        self.minute_cache_misses += other.minute_cache_misses
        self.skipped_existing += other.skipped_existing
        self.db_reconnect_retries += other.db_reconnect_retries
        self.batches += other.batches
        self.at_open_added_candidates += other.at_open_added_candidates
        self.at_open_crosser_screen_candidates += other.at_open_crosser_screen_candidates
        self.forward_return_observations_inserted += other.forward_return_observations_inserted
        self.forward_return_observations_reused += other.forward_return_observations_reused
        self.at_open_missing_minute_bars_by_bucket.update(other.at_open_missing_minute_bars_by_bucket)
        self.errors.extend(other.errors)
        self.non_session_bar_samples.extend(
            other.non_session_bar_samples[: max(0, 10 - len(self.non_session_bar_samples))]
        )
        self.candidate_screen_fail_reasons.update(other.candidate_screen_fail_reasons)
        self.outcomes.update(other.outcomes)
        self.outcomes_by_year.update(other.outcomes_by_year)


@dataclass(frozen=True)
class _I11DailyScreen:
    passed: bool
    reason: str | None
    daily_context: _DailyContext
    current_crosser_candidate: bool = False


class I11HistoricalCorpusJob(I12HistoricalCorpusJob):
    """Replay the frozen I11 gate over historical HUR membership."""

    @property
    def job_name(self) -> str:
        return JOB_NAME

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
        max_outstanding_fetch_timeouts: int = DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS,
        progress_callback: Any | None = None,
        catalyst_tags_by_ticker_date: Mapping[tuple[str, date], Sequence[str]] | None = None,
        at_open: bool = False,
    ) -> None:
        super().__init__(
            session=session,
            fmp_adapter=fmp_adapter,
            polygon_adapter=polygon_adapter,
            start_date=start_date,
            end_date=end_date,
            run_timestamp=run_timestamp,
            batch_days=batch_days,
            classification_records=classification_records,
            minute_cache_dir=minute_cache_dir,
            polygon_rate_limit_per_minute=polygon_rate_limit_per_minute,
            skip_existing=skip_existing,
            max_db_retries=max_db_retries,
            db_retry_backoff_seconds=db_retry_backoff_seconds,
            fetch_deadline_seconds=fetch_deadline_seconds,
            max_outstanding_fetch_timeouts=max_outstanding_fetch_timeouts,
            progress_callback=progress_callback,
        )
        self._catalyst_tags_by_ticker_date = {
            (ticker.upper(), day): tuple(sorted(set(tags)))
            for (ticker, day), tags in (catalyst_tags_by_ticker_date or {}).items()
        }
        self._at_open = at_open

    @property
    def event_pattern_id(self) -> str:
        return I11_PATTERN_ID

    def _has_existing_event(self, ticker: str, trading_date: date) -> bool:
        payload = {
            "pattern_id": I11_PATTERN_ID,
            "ticker": ticker.upper(),
            "trading_date": trading_date.isoformat(),
            "reconstruction_method": (
                AT_OPEN_RECONSTRUCTION_METHOD if self._at_open else RECONSTRUCTION_METHOD
            ),
        }
        if self._at_open:
            payload["entry"] = "at_open"
        return (
            self._session.query(IntradayEventDetail.event_identity_hash)
            .filter(IntradayEventDetail.event_identity_hash == stable_hash(payload))
            .first()
            is not None
        )

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

        daily_cache: dict[str, tuple[tuple[_DailyBar, ...], Any]] = {}
        minute_cache: dict[tuple[str, date], tuple[tuple[_MinuteBar, ...], Any]] = {}

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

    def _run_batch_once(
        self,
        ctx: JobContext,
        *,
        batch_index: int,
        batch_dates: Sequence[date],
        classifications: Mapping[str, SecurityTypeClassification],
        trading_dates: Sequence[date],
        daily_cache: dict[str, tuple[tuple[_DailyBar, ...], Any]],
        minute_cache: dict[tuple[str, date], tuple[tuple[_MinuteBar, ...], Any]],
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
                screen: _I11DailyScreen | None = None
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
                    screen = (
                        _screen_i11_at_open_candidate(daily_input)
                        if self._at_open else _screen_i11_daily_candidate(daily_input)
                    )
                    if not screen.passed:
                        counters.candidates_screened_out += 1
                        counters.candidate_screen_fail_reasons[screen.reason or "unknown"] += 1
                        continue
                    counters.candidates += 1
                    if self._at_open:
                        if screen.current_crosser_candidate:
                            counters.at_open_crosser_screen_candidates += 1
                        else:
                            counters.at_open_added_candidates += 1
                    inp = self._load_ticker_day_input(
                        daily_input=daily_input,
                        minute_cache=minute_cache,
                        counters=counters,
                        job_run_id=ctx.job_run_id,
                    )
                    event = (
                        self._evaluate_i11_at_open_event(inp, screen.daily_context)
                        if self._at_open else None
                    )
                except _Quarantine as exc:
                    if (
                        self._at_open
                        and screen is not None
                        and exc.payload.get("error") in {"missing_minute_bars", "missing_open_minute_bar"}
                    ):
                        counters.at_open_missing_minute_bars_by_bucket[
                            _at_open_screen_bucket(screen)
                        ] += 1
                    counters.quarantined += 1
                    counters.errors.append(exc.payload)
                    continue

                if event is None:
                    event = self._evaluate_i11_event(inp, screen.daily_context)
                counters.outcomes[event.outcome] += 1
                counters.outcomes_by_year[f"{event.trading_date.year}:{event.outcome}"] += 1
                if event.outcome == OUTCOME_CONFIRMED:
                    counters.confirmed += 1
                elif event.outcome == OUTCOME_NEVER_CONFIRMED:
                    counters.never_confirmed += 1
                elif event.outcome == OUTCOME_FAILED_TEST:
                    counters.failed_test += 1
                elif event.outcome == OUTCOME_HALTED_UNFILLABLE:
                    counters.halted_unfillable += 1
                if event.split_basis_mismatch:
                    counters.artifact_excluded += 1
                if event.sub_dollar_at_open:
                    counters.sub_dollar_included += 1
                if event.outcome == OUTCOME_CONFIRMED and event.ret_next_open is None:
                    counters.primary_label_unavailable += 1
                persisted = (
                    self._persist_i11_at_open_event(event, ctx.job_run_id)
                    if self._at_open else self._persist_i11_event(event, ctx.job_run_id)
                )
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

    def _evaluate_i11_event(
        self,
        inp: _TickerDayInput,
        daily_context: _DailyContext,
    ) -> _I12Event:
        daily_by_date = {bar.date: bar for bar in inp.daily_bars}
        context = daily_context.context
        day_bar = daily_context.day_bar
        first_minute = inp.minute_bars[0]
        split_basis_mismatch = _basis_mismatch(day_bar.open, first_minute.open)
        sub_dollar = day_bar.open < 1.0
        session_close_price = day_bar.close
        next_session = _next_session_after(inp.trading_date)
        next_bar = daily_by_date.get(next_session)
        next_open_price = next_bar.open if next_bar is not None else None
        next_close_price = next_bar.close if next_bar is not None else None
        ret_open_close = _safe_return(session_close_price, day_bar.open)
        exit_bar = _exit_bar(inp.minute_bars)
        exit_price = exit_bar.close if exit_bar is not None else None
        exit_timestamp = exit_bar.timestamp if exit_bar is not None else None

        cumulative_volume = 0.0
        cumulative_high: float | None = None
        cumulative_low: float | None = None
        latest_gate_values: dict[str, Any] = {}
        max_projected_vol = None
        max_projected_vr = None
        cross_index: int | None = None
        cross_timestamp: datetime | None = None
        cross_price: float | None = None
        cross_hold_index: int | None = None
        cross_hold_timestamp: datetime | None = None
        cross_hold_price: float | None = None
        control_reference_index: int | None = None
        control_reference_price: float | None = None
        control_reference_timestamp: datetime | None = None
        confirm_index: int | None = None
        confirm_shared = None
        confirm_timestamp: datetime | None = None

        for index, minute in enumerate(inp.minute_bars):
            if minute.minute_index < 0:
                continue
            cumulative_volume += minute.volume
            cumulative_high = minute.high if cumulative_high is None else max(cumulative_high, minute.high)
            cumulative_low = minute.low if cumulative_low is None else min(cumulative_low, minute.low)
            if cross_index is None and minute.high > context.max_prior_252_closes:
                cross_index = index
                cross_timestamp = minute.timestamp
                cross_price = context.max_prior_252_closes
            if cross_hold_index is None and minute.close > context.max_prior_252_closes:
                cross_hold_index = index
                cross_hold_timestamp = minute.timestamp
                cross_hold_price = minute.close
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
            decision = i11_entry_gate(context, snapshot, shared)
            latest_gate_values = dict(decision.gate_values)
            if decision.enter:
                confirm_index = index
                confirm_shared = shared
                confirm_timestamp = minute.timestamp
                break

        event_outcome = OUTCOME_CONFIRMED if confirm_index is not None else OUTCOME_NEVER_CONFIRMED
        if confirm_index is None and cross_hold_index is None:
            event_outcome = OUTCOME_FAILED_TEST

        entry_bar: _MinuteBar | None = None
        entry_timestamp: datetime | None = None
        entry_price: float | None = None
        entry_minute: int | None = None
        ret_conf = None
        ret_next_open = None
        ret_next_close = None
        overnight_gap_ret = _safe_return(next_open_price, session_close_price)
        mae_pct = None
        mfe_pct = None
        halted = False
        entry_after_session_exit_reference = False

        if event_outcome == OUTCOME_CONFIRMED:
            entry_bar = _next_minute_bar(inp.minute_bars, confirm_index)
            if entry_bar is None:
                event_outcome = OUTCOME_HALTED_UNFILLABLE
                halted = True
            else:
                entry_timestamp = entry_bar.timestamp
                entry_price = entry_bar.open
                entry_minute = entry_bar.minute_index
                if exit_timestamp is not None and entry_timestamp <= exit_timestamp:
                    ret_conf = _safe_return(exit_price, entry_price)
                else:
                    entry_after_session_exit_reference = exit_timestamp is not None
                ret_next_open = _safe_return(next_open_price, entry_price)
                ret_next_close = _safe_return(next_close_price, entry_price)
                mae_pct, mfe_pct = _path_extrema_returns(
                    inp.minute_bars,
                    reference_price=entry_price,
                    start_minute=entry_bar.minute_index,
                    exit_timestamp=exit_timestamp,
                )
        else:
            control_reference_index = cross_hold_index if cross_hold_index is not None else cross_index
            reference_bar = _next_minute_bar(inp.minute_bars, control_reference_index)
            if reference_bar is not None:
                control_reference_price = reference_bar.open
                control_reference_timestamp = reference_bar.timestamp
                if (
                    exit_timestamp is not None
                    and control_reference_timestamp <= exit_timestamp
                ):
                    ret_conf = _safe_return(exit_price, control_reference_price)
                else:
                    entry_after_session_exit_reference = exit_timestamp is not None
                ret_next_open = _safe_return(next_open_price, control_reference_price)
                ret_next_close = _safe_return(next_close_price, control_reference_price)
                mae_pct, mfe_pct = _path_extrema_returns(
                    inp.minute_bars,
                    reference_price=control_reference_price,
                    start_minute=reference_bar.minute_index,
                    exit_timestamp=exit_timestamp,
                )

        projected_vol_at_conf = (
            confirm_shared.projected_vol if confirm_shared is not None else max_projected_vol
        )
        projected_vr_at_conf = (
            confirm_shared.vol_ratio if confirm_shared is not None else max_projected_vr
        )
        chase = (
            (entry_price / context.max_prior_252_closes - 1.0)
            if entry_price is not None else None
        )
        if chase is None and control_reference_price is not None:
            chase = control_reference_price / context.max_prior_252_closes - 1.0
        gap = confirm_shared.gap if confirm_shared is not None else latest_gate_values.get("gap")
        distance = latest_gate_values.get("distance_from_max252")
        if distance is None:
            distance = daily_context.distance_from_max252
        catalyst_tags = self._catalyst_tags(inp.ticker, inp.trading_date)
        minute_volume_up_to_confirmation = (
            sum(bar.volume for bar in inp.minute_bars[: confirm_index + 1])
            if confirm_index is not None else None
        )
        minute_volume_up_to_cross = (
            sum(bar.volume for bar in inp.minute_bars[: cross_index + 1])
            if cross_index is not None else None
        )

        latest_gate_values = dict(latest_gate_values)
        latest_gate_values.update({
            "prior_close": context.prior_close,
            "max252": context.max_prior_252_closes,
            "avg20": context.avg20_volume,
            "mom20": context.mom20,
            "off_low252": context.off_low252,
            "day_high": day_bar.high,
            "cross_minute": _minute_number(cross_timestamp, inp.trading_date),
            "cross_price": cross_price,
            "cross_hold_minute": _minute_number(cross_hold_timestamp, inp.trading_date),
            "cross_hold_price": cross_hold_price,
            "minute_volume_up_to_confirmation": minute_volume_up_to_confirmation,
            "minute_volume_up_to_cross": minute_volume_up_to_cross,
            "premarket_volume": None,
            "premarket_volume_status": "not_available_polygon_regular_session_cache",
            "max_projected_volume": max_projected_vol,
            "max_projected_volume_ratio": max_projected_vr,
        })
        labels = {
            "ret_conf": ret_conf,
            "ret_conf_reference_only": ret_conf is not None,
            "entry_after_session_exit_reference": entry_after_session_exit_reference,
            "ret_open_close": ret_open_close,
            "ret_open_close_leaky_research_only": ret_open_close is not None,
            "ret_next_open": ret_next_open,
            "ret_next_open_primary": ret_next_open is not None,
            "ret_next_close": ret_next_close,
            "overnight_gap_ret": overnight_gap_ret,
            "mae_pct": mae_pct,
            "mfe_pct": mfe_pct,
            "chase_over_hb_pct": chase,
            "conf_minute": int(confirm_shared.data_elapsed_min) if confirm_shared else None,
            "cross_minute": _minute_number(cross_timestamp, inp.trading_date),
            "cross_price": cross_price,
            "cross_hold_minute": _minute_number(cross_hold_timestamp, inp.trading_date),
            "cross_hold_price": cross_hold_price,
            "minute_volume_up_to_confirmation": minute_volume_up_to_confirmation,
            "minute_volume_up_to_cross": minute_volume_up_to_cross,
            "premarket_volume": None,
            "premarket_volume_status": "not_available_polygon_regular_session_cache",
            "control_reference_price": control_reference_price,
            "control_reference_timestamp": control_reference_timestamp.isoformat()
            if control_reference_timestamp else None,
            "control_reference_basis": _control_reference_basis(
                event_outcome,
                cross_hold_index,
                cross_index,
            ),
            "sessions_to_delist": inp.sessions_to_delist,
            "sessions_to_delist_not_pit": True,
        }
        artifact_flags = {
            "split_basis_mismatch": split_basis_mismatch,
            "split_basis_mismatch_excluded": split_basis_mismatch,
            "sub_dollar_at_open": sub_dollar,
            "primary_label_unavailable": (
                event_outcome == OUTCOME_CONFIRMED and ret_next_open is None
            ),
            "primary_label_unavailable_reason": (
                "missing_next_open_price"
                if event_outcome == OUTCOME_CONFIRMED and ret_next_open is None else None
            ),
            "entry_after_session_exit_reference": entry_after_session_exit_reference,
            "sessions_to_delist_not_pit": True,
            "catalyst_tags_required": True,
            "catalyst_tags_source_status": "provided" if catalyst_tags else "empty",
        }
        feature_json = _i11_feature_payload(
            inp=inp,
            context=context,
            day_bar=day_bar,
            gate_values=latest_gate_values,
            projected_vol_at_conf=projected_vol_at_conf,
            projected_vr_at_conf=projected_vr_at_conf,
            gap=gap,
            chase=chase,
            sub_dollar=sub_dollar,
            catalyst_tags=catalyst_tags,
        )
        input_payload = {
            "pattern_id": I11_PATTERN_ID,
            "ticker": inp.ticker,
            "trading_date": inp.trading_date.isoformat(),
            "daily_lineage": inp.daily_lineage.raw_payload_hash,
            "minute_lineage": inp.minute_lineage.raw_payload_hash,
            "reconstruction_method": RECONSTRUCTION_METHOD,
        }
        input_hash = stable_hash(input_payload)
        signal_identity_hash = (
            stable_hash({
                "pattern_id": I11_PATTERN_ID,
                "ticker": inp.ticker,
                "trading_date": inp.trading_date.isoformat(),
                "outcome": OUTCOME_CONFIRMED,
                "reconstruction_method": RECONSTRUCTION_METHOD,
            })
            if event_outcome == OUTCOME_CONFIRMED and not split_basis_mismatch else None
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
            "pattern_id": I11_PATTERN_ID,
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
            full_day_volume_ratio=None,
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
            is_ml_excluded=inp.security_type.ml_excluded,
            ml_exclusion_reason=inp.security_type.reason,
            security_type=inp.security_type.security_type,
            sessions_to_delist=inp.sessions_to_delist,
            data_lineage_ids=(inp.daily_lineage.data_lineage_id, inp.minute_lineage.data_lineage_id),
            input_hash=input_hash,
            output_hash=output_hash,
            event_identity_hash=event_identity_hash,
            signal_identity_hash=signal_identity_hash,
        )

    def _evaluate_i11_at_open_event(
        self,
        inp: _TickerDayInput,
        daily_context: _DailyContext,
    ) -> _I12Event:
        daily_by_date = {bar.date: bar for bar in inp.daily_bars}
        context = daily_context.context
        day_bar = daily_context.day_bar
        entry_timestamp = us_equity_session_open_timestamp(inp.trading_date)
        first_minute = _at_open_minute_bar(inp)
        split_basis_mismatch = _basis_mismatch(day_bar.open, first_minute.open)
        sub_dollar = day_bar.open < 1.0
        next_session = _next_session_after(inp.trading_date)
        next_bar = daily_by_date.get(next_session)
        next_open_price = next_bar.open if next_bar is not None else None
        ret_open_close = None if split_basis_mismatch else _safe_return(day_bar.close, day_bar.open)
        ret_open_next_open = None if split_basis_mismatch else _safe_return(next_open_price, day_bar.open)
        crossed_high = day_bar.high > context.max_prior_252_closes
        ran_plus_3 = day_bar.high >= day_bar.open * 1.03
        ran_plus_5 = day_bar.high >= day_bar.open * 1.05
        distance_at_open = day_bar.open / context.max_prior_252_closes - 1.0
        gap = day_bar.open / context.prior_close - 1.0
        catalyst_tags = self._catalyst_tags(inp.ticker, inp.trading_date)
        gate_values = {
            "gap": gap,
            "distance_from_max252": distance_at_open,
            "prior_close": context.prior_close,
            "max252": context.max_prior_252_closes,
            "opened_above_prior_high": day_bar.open > context.max_prior_252_closes,
        }
        labels = {
            "ret_open_close": ret_open_close,
            "ret_open_next_open": ret_open_next_open,
            "crossed_high": crossed_high,
            "ran_plus_3": ran_plus_3,
            "ran_plus_5": ran_plus_5,
            "sessions_to_delist": inp.sessions_to_delist,
            "sessions_to_delist_not_pit": True,
        }
        ml_excluded = inp.security_type.ml_excluded or split_basis_mismatch
        ml_exclusion_reason = (
            "split_basis_mismatch" if split_basis_mismatch else inp.security_type.reason
        )
        artifact_flags = {
            "split_basis_mismatch": split_basis_mismatch,
            "split_basis_mismatch_excluded": split_basis_mismatch,
            "split_basis_mismatch_label_excluded": split_basis_mismatch,
            "split_basis_mismatch_ml_excluded": split_basis_mismatch,
            "sub_dollar_at_open": sub_dollar,
            "sessions_to_delist_not_pit": True,
            "catalyst_tags_required": True,
            "catalyst_tags_source_status": "provided" if catalyst_tags else "empty",
            "at_open_candidate": True,
            "opened_above_prior_high": day_bar.open > context.max_prior_252_closes,
            "lookahead_guard_passed": True,
        }
        feature_json = _i11_at_open_feature_payload(
            inp=inp,
            context=context,
            day_bar=day_bar,
            daily_context=daily_context,
            gap=gap,
            distance_from_max252=distance_at_open,
            sub_dollar=sub_dollar,
            catalyst_tags=catalyst_tags,
            is_ml_excluded=ml_excluded,
            ml_exclusion_reason=ml_exclusion_reason,
        )
        artifact_flags["lookahead_guard_passed"] = _i11_at_open_lookahead_guard_passed(feature_json)
        input_payload = {
            "pattern_id": I11_PATTERN_ID,
            "ticker": inp.ticker,
            "trading_date": inp.trading_date.isoformat(),
            "daily_lineage": inp.daily_lineage.raw_payload_hash,
            "minute_lineage": inp.minute_lineage.raw_payload_hash,
            "reconstruction_method": AT_OPEN_RECONSTRUCTION_METHOD,
        }
        input_hash = stable_hash(input_payload)
        signal_identity_hash = stable_hash({
            "pattern_id": I11_PATTERN_ID,
            "ticker": inp.ticker,
            "trading_date": inp.trading_date.isoformat(),
            "entry": "at_open",
            "reconstruction_method": AT_OPEN_RECONSTRUCTION_METHOD,
        })
        output_payload = {
            "outcome": OUTCOME_AT_OPEN_CROSSED if crossed_high else OUTCOME_AT_OPEN_NO_CROSS,
            "features": feature_json,
            "labels": labels,
            "artifact_flags": artifact_flags,
            "signal_identity_hash": signal_identity_hash,
        }
        output_hash = stable_hash(output_payload)
        event_identity_hash = stable_hash({
            "pattern_id": I11_PATTERN_ID,
            "ticker": inp.ticker,
            "trading_date": inp.trading_date.isoformat(),
            "entry": "at_open",
            "reconstruction_method": AT_OPEN_RECONSTRUCTION_METHOD,
        })
        mae_pct, mfe_pct = _path_extrema_returns(
            inp.minute_bars,
            reference_price=day_bar.open,
            start_minute=0,
            exit_timestamp=None,
        )
        return _I12Event(
            ticker=inp.ticker,
            trading_date=inp.trading_date,
            outcome=OUTCOME_AT_OPEN_CROSSED if crossed_high else OUTCOME_AT_OPEN_NO_CROSS,
            gate_values=gate_values,
            feature_json=feature_json,
            labels=labels,
            artifact_flags=artifact_flags,
            signal_timestamp=entry_timestamp,
            confirmation_timestamp=entry_timestamp,
            entry_timestamp=entry_timestamp,
            exit_timestamp=None,
            conf_minute=0,
            entry_minute=0,
            entry_price=day_bar.open,
            exit_price=day_bar.close,
            session_open_price=day_bar.open,
            session_close_price=day_bar.close,
            next_open_price=next_open_price,
            projected_vol_at_conf=None,
            projected_vol_ratio_at_conf=None,
            full_day_volume_ratio=None,
            chase_pct=None,
            gap_pct=gap,
            distance_from_max252=distance_at_open,
            ret_conf=ret_open_close,
            ret_open_close=ret_open_close,
            ret_next_open=ret_open_next_open,
            mae_pct=mae_pct,
            mfe_pct=mfe_pct,
            halted=False,
            sub_dollar_at_open=sub_dollar,
            split_basis_mismatch=split_basis_mismatch,
            is_ml_excluded=ml_excluded,
            ml_exclusion_reason=ml_exclusion_reason,
            security_type=inp.security_type.security_type,
            sessions_to_delist=inp.sessions_to_delist,
            data_lineage_ids=(inp.daily_lineage.data_lineage_id, inp.minute_lineage.data_lineage_id),
            input_hash=input_hash,
            output_hash=output_hash,
            event_identity_hash=event_identity_hash,
            signal_identity_hash=signal_identity_hash,
        )

    def _persist_i11_event(self, event: _I12Event, job_run_id: str) -> dict[str, bool]:
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
                    "I11 corpus event content changed for existing identity: "
                    f"{event.ticker} {event.trading_date.isoformat()}"
                )
            return {
                "inserted_detail": False,
                "inserted_signal": False,
                "reused_signal": existing_detail.signal_id is not None,
            }

        signal_id: str | None = None
        inserted_signal = False
        reused_signal = False
        if event.outcome == OUTCOME_CONFIRMED and not event.split_basis_mismatch:
            assert event.signal_identity_hash is not None
            existing_signal = (
                self._session.query(SignalRegistry)
                .filter(
                    SignalRegistry.pattern_id == I11_PATTERN_ID,
                    SignalRegistry.ticker == event.ticker,
                    SignalRegistry.signal_identity_hash == event.signal_identity_hash,
                    SignalRegistry.detector_version == RECONSTRUCTION_METHOD,
                )
                .one_or_none()
            )
            if existing_signal is None:
                feature_snapshot = record_feature_snapshot(
                    self._session,
                    pattern_id=I11_PATTERN_ID,
                    ticker=event.ticker,
                    asof_timestamp=event.signal_timestamp or event.confirmation_timestamp,
                    features=event.feature_json,
                    data_lineage_ids=list(event.data_lineage_ids),
                    job_run_id=job_run_id,
                    feature_manifest_version=FEATURE_MANIFEST_VERSION,
                    fidelity_tier="historical_intraday_replay",
                    point_in_time_passed=False,
                    lookahead_guard_passed=True,
                    input_hashes={"i11_corpus_event_input": event.input_hash},
                )
                signal = record_signal(
                    self._session,
                    pattern_id=I11_PATTERN_ID,
                    ticker=event.ticker,
                    direction="long",
                    signal_timestamp=event.signal_timestamp or event.confirmation_timestamp,
                    raw_signal_strength=float(event.projected_vol_ratio_at_conf or 0.0),
                    raw_expected_edge=0.0,
                    feature_snapshot_id=feature_snapshot.feature_snapshot_id,
                    job_run_id=job_run_id,
                    signal_horizon=I11_SIGNAL_HORIZON,
                    thesis_category="intraday_52week_high_breakout",
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
                    forward_return_status=(
                        "computed" if event.ret_next_open is not None else "outcome_unavailable"
                    ),
                    forward_return_attempts=0,
                )
                signal.forward_return = event.ret_next_open
                if event.ret_next_open is None:
                    signal.outcome_unavailable_reason = "missing_next_open_price"
                signal_id = signal.signal_id
                inserted_signal = True
            else:
                signal_id = existing_signal.signal_id
                reused_signal = True

        detail = IntradayEventDetail(
            signal_id=signal_id,
            job_run_id=job_run_id,
            pattern_id=I11_PATTERN_ID,
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
        }

    def _persist_i11_at_open_event(self, event: _I12Event, job_run_id: str) -> dict[str, bool]:
        lookahead_guard_passed = _i11_at_open_lookahead_guard_passed(event.feature_json)
        if not lookahead_guard_passed:
            raise RuntimeError(
                "i11_at_open_lookahead_guard_failed: "
                f"{event.ticker} {event.trading_date.isoformat()}"
            )
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
                    "I11 at-open corpus event content changed for existing identity: "
                    f"{event.ticker} {event.trading_date.isoformat()}"
                )
            fro_inserted = False
            fro_reused = False
            if existing_detail.signal_id is not None:
                existing_signal = self._session.get(SignalRegistry, existing_detail.signal_id)
                if existing_signal is None:
                    raise RuntimeError(
                        "I11 at-open corpus detail references missing signal: "
                        f"signal_id={existing_detail.signal_id} "
                        f"ticker={event.ticker} trading_date={event.trading_date.isoformat()}"
                    )
                fro_inserted, fro_reused = self._record_at_open_forward_return_observation(
                    event,
                    signal=existing_signal,
                    job_run_id=job_run_id,
                )
            return {
                "inserted_detail": False,
                "inserted_signal": False,
                "reused_signal": existing_detail.signal_id is not None,
                "inserted_forward_return_observation": fro_inserted,
                "reused_forward_return_observation": fro_reused,
            }

        assert event.signal_identity_hash is not None
        existing_signal = (
            self._session.query(SignalRegistry)
            .filter(
                SignalRegistry.pattern_id == I11_PATTERN_ID,
                SignalRegistry.ticker == event.ticker,
                SignalRegistry.signal_identity_hash == event.signal_identity_hash,
            )
            .one_or_none()
        )
        inserted_signal = False
        reused_signal = False
        if existing_signal is None:
            feature_snapshot = record_feature_snapshot(
                self._session,
                pattern_id=I11_PATTERN_ID,
                ticker=event.ticker,
                asof_timestamp=event.signal_timestamp or event.confirmation_timestamp,
                features=event.feature_json,
                data_lineage_ids=list(event.data_lineage_ids),
                job_run_id=job_run_id,
                feature_manifest_version=AT_OPEN_FEATURE_MANIFEST_VERSION,
                fidelity_tier="historical_intraday_at_open_replay",
                point_in_time_passed=False,
                lookahead_guard_passed=lookahead_guard_passed,
                input_hashes={"i11_at_open_corpus_event_input": event.input_hash},
            )
            forward_return_status = (
                "computed"
                if event.ret_open_close is not None and not event.is_ml_excluded
                else "outcome_unavailable"
            )
            signal = record_signal(
                self._session,
                pattern_id=I11_PATTERN_ID,
                ticker=event.ticker,
                direction="long",
                signal_timestamp=event.signal_timestamp or event.confirmation_timestamp,
                raw_signal_strength=float(event.distance_from_max252 or 0.0),
                raw_expected_edge=0.0,
                feature_snapshot_id=feature_snapshot.feature_snapshot_id,
                job_run_id=job_run_id,
                signal_horizon=I11_SIGNAL_HORIZON,
                thesis_category="intraday_52week_high_breakout_at_open",
                route_class="i_track_intraday",
                fidelity_tier="historical_intraday_at_open_replay",
                data_confidence=1.0,
                signal_status="excluded" if event.is_ml_excluded else "active",
                data_lineage_ids=list(event.data_lineage_ids),
                trading_date=event.trading_date.isoformat(),
                next_execution_session=event.trading_date.isoformat(),
                detector_version=AT_OPEN_RECONSTRUCTION_METHOD,
                point_in_time_passed=False,
                lookahead_guard_passed=lookahead_guard_passed,
                signal_event_sequence=1,
                signal_identity_hash=event.signal_identity_hash,
                intended_entry_price=event.entry_price,
                forward_return_status=forward_return_status,
                forward_return_attempts=0,
            )
            signal.forward_return = event.ret_open_close if forward_return_status == "computed" else None
            if event.is_ml_excluded:
                signal.outcome_unavailable_reason = event.ml_exclusion_reason
            existing_signal = signal
            inserted_signal = True
        else:
            reused_signal = True

        fro_inserted, fro_reused = self._record_at_open_forward_return_observation(
            event,
            signal=existing_signal,
            job_run_id=job_run_id,
        )
        detail = IntradayEventDetail(
            signal_id=existing_signal.signal_id,
            job_run_id=job_run_id,
            pattern_id=I11_PATTERN_ID,
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

    def _record_at_open_forward_return_observation(
        self,
        event: _I12Event,
        *,
        signal: SignalRegistry,
        job_run_id: str,
    ) -> tuple[bool, bool]:
        if event.is_ml_excluded or event.ret_open_close is None:
            return False, False
        input_hash = stable_hash({
            "signal_id": signal.signal_id,
            "label": "ret_open_close",
            "event_input_hash": event.input_hash,
            "reconstruction_method": AT_OPEN_RECONSTRUCTION_METHOD,
        })
        outcome_hash = stable_hash({
            "signal_id": signal.signal_id,
            "label": "ret_open_close",
            "ret_open_close": event.ret_open_close,
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
                    "I11 at-open forward return changed for existing signal: "
                    f"{event.ticker} {event.trading_date.isoformat()}"
                )
            return False, True
        observation = ForwardReturnObservation(
            signal_id=signal.signal_id,
            pattern_id=I11_PATTERN_ID,
            ticker=event.ticker,
            direction="long",
            signal_timestamp=event.signal_timestamp or event.confirmation_timestamp,
            signal_horizon=I11_SIGNAL_HORIZON,
            next_execution_session=event.trading_date.isoformat(),
            entry_session_date=event.trading_date.isoformat(),
            entry_price=event.entry_price,
            entry_price_source="fmp_split_adjusted_session_open",
            entry_basis_proof="at_open_candidate_entry",
            entry_data_lineage_id=event.data_lineage_ids[0] if event.data_lineage_ids else None,
            exit_session_date=event.trading_date.isoformat(),
            exit_price=event.session_close_price,
            exit_price_source="fmp_split_adjusted_session_close",
            exit_basis_proof="open_to_close_label",
            exit_data_lineage_id=event.data_lineage_ids[0] if event.data_lineage_ids else None,
            forward_return=event.ret_open_close,
            max_favorable_excursion=event.mfe_pct,
            max_adverse_excursion=event.mae_pct,
            hit_t1_intraday=event.labels.get("ran_plus_3"),
            hit_t2_intraday=event.labels.get("ran_plus_5"),
            status="computed" if event.ret_open_close is not None else "outcome_unavailable",
            reason=None if event.ret_open_close is not None else "missing_open_close_label",
            attempts=0,
            job_run_id=job_run_id,
            input_hash=input_hash,
            outcome_hash=outcome_hash,
            data_lineage_ids=json.dumps(list(event.data_lineage_ids)),
            provider="FMP+Polygon",
            endpoint=AT_OPEN_RECONSTRUCTION_METHOD,
            provider_request_json=json.dumps({
                "label": "ret_open_close",
                "label_json": event.labels,
            }, sort_keys=True, default=str),
        )
        self._session.add(observation)
        self._session.flush()
        return True, False

    def _metrics(self, counters: _RunCounters, *, trading_dates: Sequence[date]) -> dict[str, Any]:
        return {
            "pattern_id": I11_PATTERN_ID,
            "reconstruction_method": (
                AT_OPEN_RECONSTRUCTION_METHOD if self._at_open else RECONSTRUCTION_METHOD
            ),
            "corpus_mode": "at_open" if self._at_open else "confirmed_entry",
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
            "failed_test": counters.failed_test,
            "halted_unfillable": counters.halted_unfillable,
            "outcome_counts": dict(counters.outcomes),
            "outcome_counts_by_year": dict(counters.outcomes_by_year),
            "at_open_added_candidates": counters.at_open_added_candidates,
            "at_open_crosser_screen_candidates": counters.at_open_crosser_screen_candidates,
            "at_open_missing_minute_bars_by_bucket": dict(
                counters.at_open_missing_minute_bars_by_bucket
            ),
            "excluded_by_type": counters.excluded_by_type,
            "artifact_excluded": counters.artifact_excluded,
            "sub_dollar_included": counters.sub_dollar_included,
            "primary_label_unavailable": counters.primary_label_unavailable,
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

    def _catalyst_tags(self, ticker: str, trading_date: date) -> list[str]:
        return list(self._catalyst_tags_by_ticker_date.get((ticker.upper(), trading_date), ()))


def _screen_i11_daily_candidate(inp: _DailyTickerInput) -> _I11DailyScreen:
    daily_context = _daily_context_from_bars(inp.ticker, inp.trading_date, inp.daily_bars)
    context = daily_context.context
    day_bar = daily_context.day_bar
    if context.prior_close > context.max_prior_252_closes:
        return _I11DailyScreen(False, "fresh_cross_screen", daily_context)
    if day_bar.high <= context.max_prior_252_closes:
        return _I11DailyScreen(False, "day_high_cross_screen", daily_context)
    return _I11DailyScreen(True, None, daily_context, current_crosser_candidate=True)


def _screen_i11_at_open_candidate(inp: _DailyTickerInput) -> _I11DailyScreen:
    daily_context = _daily_context_from_bars(inp.ticker, inp.trading_date, inp.daily_bars)
    context = daily_context.context
    day_bar = daily_context.day_bar
    lower, upper = AT_OPEN_DISTANCE_BAND
    open_distance = day_bar.open / context.max_prior_252_closes - 1.0
    prior_close_distance = context.prior_close / context.max_prior_252_closes - 1.0
    opened_above_high = day_bar.open > context.max_prior_252_closes
    if opened_above_high:
        return _I11DailyScreen(
            True,
            None,
            daily_context,
            current_crosser_candidate=True,
        )
    if not (open_distance >= lower or lower <= prior_close_distance <= upper):
        return _I11DailyScreen(
            False,
            "at_open_near_high_screen",
            daily_context,
            current_crosser_candidate=False,
        )
    return _I11DailyScreen(
        True,
        None,
        daily_context,
        current_crosser_candidate=False,
    )


def _i11_feature_payload(
    *,
    inp: _TickerDayInput,
    context: PremarketContext,
    day_bar: _DailyBar,
    gate_values: Mapping[str, Any],
    projected_vol_at_conf: float | None,
    projected_vr_at_conf: float | None,
    gap: float | None,
    chase: float | None,
    sub_dollar: bool,
    catalyst_tags: Sequence[str],
) -> dict[str, Any]:
    prior = [bar for bar in inp.daily_bars if bar.date < inp.trading_date]
    prior20 = prior[-20:]
    prior252 = prior[-252:]
    payload = {
        "reconstructed": True,
        "reconstruction_method": RECONSTRUCTION_METHOD,
        "pattern_id": I11_PATTERN_ID,
        "ticker": inp.ticker,
        "trading_date": inp.trading_date.isoformat(),
        "price_basis": "fmp_full_split_adjusted_close",
        "candidate_screen": dict(CANDIDATE_SCREEN_STAMP),
        "lookback_end": prior[-1].date.isoformat() if prior else None,
        "prior_close": context.prior_close,
        "max_prior_252_closes": context.max_prior_252_closes,
        "fresh_cross_prior_close_lte_hb": context.prior_close <= context.max_prior_252_closes,
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
        "catalyst_tags": list(catalyst_tags),
        "catalyst_source_status": "provided" if catalyst_tags else "empty",
        "security_type": inp.security_type.security_type,
        "is_ml_excluded": inp.security_type.ml_excluded,
        "ml_exclusion_reason": inp.security_type.reason,
        "gate_values": _without_none(_i11_feature_gate_values(gate_values)),
        "research_only_leaky": _without_none(
            {
                "avg20_volume": context.avg20_volume,
                "price": day_bar.open,
            }
        ),
        "source_lineage": {
            "daily_bar_lineage_id": inp.daily_lineage.data_lineage_id,
            "daily_bar_lineage_hash": inp.daily_lineage.raw_payload_hash,
            "minute_bar_lineage_id": inp.minute_lineage.data_lineage_id,
            "minute_bar_lineage_hash": inp.minute_lineage.raw_payload_hash,
        },
        "pit_caveats": {
            "security_type": "classified_asof_profile_applied_retroactively",
            "daily_high_cross_screen": "candidate_conditioning_for_fetch_tractability",
            "research_only_leaky": "excluded_from_stage1_feature_allowlists",
        },
        "minute_price_basis": "polygon_adjusted_minute",
    }
    if len(prior252) < 252:
        payload["insufficient_history"] = {"prior252": True}
    if chase is not None:
        payload["chase_over_hb_pct"] = chase
    return payload


def _i11_at_open_feature_payload(
    *,
    inp: _TickerDayInput,
    context: PremarketContext,
    day_bar: _DailyBar,
    daily_context: _DailyContext,
    gap: float,
    distance_from_max252: float,
    sub_dollar: bool,
    catalyst_tags: Sequence[str],
    is_ml_excluded: bool,
    ml_exclusion_reason: str | None,
) -> dict[str, Any]:
    prior = [bar for bar in inp.daily_bars if bar.date < inp.trading_date]
    prior20 = prior[-20:]
    prior252 = prior[-252:]
    payload = {
        "reconstructed": True,
        "reconstruction_method": AT_OPEN_RECONSTRUCTION_METHOD,
        "pattern_id": I11_PATTERN_ID,
        "ticker": inp.ticker,
        "trading_date": inp.trading_date.isoformat(),
        "price_basis": "fmp_full_split_adjusted_close",
        "candidate_screen": dict(AT_OPEN_CANDIDATE_SCREEN_STAMP),
        "lookback_end": prior[-1].date.isoformat() if prior else None,
        "prior_close": context.prior_close,
        "max_prior_252_closes": context.max_prior_252_closes,
        "gap": gap,
        "distance_from_max252": distance_from_max252,
        "drawdown_from_max252": daily_context.distance_from_max252,
        "prev_day_return": _safe_return(prior[-1].split_adjusted_close, prior[-2].split_adjusted_close)
        if len(prior) >= 2 else None,
        "prev_day_green": prior[-1].close > prior[-1].open if prior else None,
        "mom20": context.mom20,
        "off_low252": context.off_low252,
        "sigma20": _sigma(prior20),
        "price_at_open": day_bar.open,
        "sub_dollar_at_open": sub_dollar,
        "catalyst_tags": list(catalyst_tags),
        "catalyst_source_status": "provided" if catalyst_tags else "empty",
        "security_type": inp.security_type.security_type,
        "is_ml_excluded": is_ml_excluded,
        "ml_exclusion_reason": ml_exclusion_reason,
        "research_only_leaky": _without_none(
            {
                "avg20_volume": context.avg20_volume,
            }
        ),
        "source_lineage": {
            "daily_bar_lineage_id": inp.daily_lineage.data_lineage_id,
            "daily_bar_lineage_hash": inp.daily_lineage.raw_payload_hash,
            "minute_bar_lineage_id": inp.minute_lineage.data_lineage_id,
            "minute_bar_lineage_hash": inp.minute_lineage.raw_payload_hash,
        },
        "pit_caveats": {
            "security_type": "classified_asof_profile_applied_retroactively",
            "candidate_screen": "at_open_near_high_conditioning_not_a_full_no_fire_cohort",
            "research_only_leaky": "excluded_from_stage1_feature_allowlists",
        },
        "minute_price_basis": "polygon_adjusted_minute",
    }
    if len(prior252) < 252:
        payload["insufficient_history"] = {"prior252": True}
    return payload


def _at_open_minute_bar(inp: _TickerDayInput) -> _MinuteBar:
    for minute in inp.minute_bars:
        if minute.minute_index == 0:
            return minute
    raise _Quarantine({
        "ticker": inp.ticker,
        "trading_date": inp.trading_date.isoformat(),
        "error": "missing_open_minute_bar",
    })


def _i11_at_open_lookahead_guard_passed(feature_json: Mapping[str, Any]) -> bool:
    return not any(key in feature_json for key in AT_OPEN_FORBIDDEN_LOOKAHEAD_FEATURES)


def _at_open_screen_bucket(screen: _I11DailyScreen) -> str:
    return "current_crosser" if screen.current_crosser_candidate else "wide_non_crosser"


def _path_extrema_returns(
    minute_bars: Sequence[_MinuteBar],
    *,
    reference_price: float,
    start_minute: int,
    exit_timestamp: datetime | None,
) -> tuple[float | None, float | None]:
    path = [
        bar for bar in minute_bars
        if bar.minute_index >= start_minute
        and bar.timestamp <= (exit_timestamp or bar.timestamp)
    ]
    if not path:
        return None, None
    return (
        _safe_return(min(bar.low for bar in path), reference_price),
        _safe_return(max(bar.high for bar in path), reference_price),
    )


def _i11_feature_gate_values(gate_values: Mapping[str, Any]) -> dict[str, Any]:
    blocked = {
        "day_high",
        "cross_minute",
        "cross_price",
        "cross_hold_minute",
        "cross_hold_price",
        "minute_volume_up_to_confirmation",
        "minute_volume_up_to_cross",
        "premarket_volume",
        "premarket_volume_status",
        "max_projected_volume",
        "max_projected_volume_ratio",
    }
    return {
        key: value
        for key, value in _feature_gate_values(gate_values).items()
        if key not in blocked
    }


def _next_session_after(day: date) -> date:
    from alpha.market_calendar import next_us_equity_session

    return next_us_equity_session(day + timedelta(days=1))


def _minute_number(timestamp: datetime | None, trading_date: date) -> int | None:
    if timestamp is None:
        return None
    from alpha.jobs.paper_execution import EASTERN
    from alpha.market_calendar import us_equity_session_open_timestamp

    market_open = us_equity_session_open_timestamp(trading_date).astimezone(EASTERN)
    return int((timestamp.astimezone(EASTERN) - market_open).total_seconds() // 60)


def _control_reference_basis(
    outcome: str,
    cross_hold_index: int | None,
    cross_index: int | None,
) -> str | None:
    if outcome == OUTCOME_CONFIRMED:
        return None
    if cross_index is None:
        return "daily_high_unconfirmed_by_minute_bars"
    if cross_hold_index is not None:
        return "next_minute_open_after_held_cross"
    return "next_minute_open_after_intraday_touch"
