"""Derived market-path feature backfill/collector.

This job promotes provider-backed OHLCV path features into a queryable table
for ML and trade-selection analysis. It does not rewrite immutable signal
feature snapshots; every row carries reconstruction metadata and lineage.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence
from uuid import uuid4

from sqlalchemy import inspect, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError, stable_hash
from alpha.data.fmp import FmpAdapter, FmpBar, HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.db.models import DataLineage, MarketPathFeature, SignalRegistry
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.watchdog import (
    DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS,
    ProviderOutageCircuitBreaker,
    WatchdogState,
    call_with_daemon_deadline,
)
from alpha.jobs.historical_m4_signal_selector import (
    SIGNAL_SOURCE_LIVE,
    apply_signal_source_filter,
    normalize_signal_source,
)
from alpha.market_calendar import (
    EASTERN_TZ,
    is_us_equity_session,
    next_us_equity_session,
    nth_us_equity_session,
    resolve_us_equity_session,
    us_equity_session_close_timestamp,
    us_equity_session_open_timestamp,
)


JOB_NAME = "market_path_feature_collector"
FEATURE_VERSION = "market_path_daily_v3"
RECONSTRUCTION_METHOD = "fmp_eod_replay_v1"
PRICE_BASIS_SPLIT_ADJUSTED_OR_RAW = "split_adjusted_close_when_available_else_raw_close"
PRIOR_52W_SESSION_COUNT = 252
TOUCH_TOLERANCE_PCT = 0.005
DEFAULT_LOOKBACK_CALENDAR_DAYS = 420
DEFAULT_FETCH_DEADLINE_SECONDS = 120.0
BENCHMARK_SYMBOLS = ("SPY", "QQQ", "IWM")
BENCHMARK_RETURN_WINDOWS = (1, 5, 20, 60)
SECTOR_RELATIVE_RETURN_WINDOWS = (5, 20, 60)
SECTOR_ETF_BY_SECTOR = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}
RANK_INPUT_TO_OUTPUT = {
    "dollar_volume": ("dollar_volume_rank", "dollar_volume_percentile"),
    "volume_expansion_20d": (
        "volume_expansion_20d_rank",
        "volume_expansion_20d_percentile",
    ),
    "volume_expansion_60d": (
        "volume_expansion_60d_rank",
        "volume_expansion_60d_percentile",
    ),
    "dollar_volume_expansion_20d": (
        "dollar_volume_expansion_20d_rank",
        "dollar_volume_expansion_20d_percentile",
    ),
    "dollar_volume_expansion_60d": (
        "dollar_volume_expansion_60d_rank",
        "dollar_volume_expansion_60d_percentile",
    ),
    "liquidity_proxy_score": ("liquidity_proxy_rank", "liquidity_proxy_percentile"),
}
SameDayPatternStrengthCache = dict[tuple[str, date], dict[str, float]]
ProgressCallback = Callable[[str, dict[str, Any]], None]
_URL_RE = re.compile(r"https?://[^\s'\"<>),]+")
_RELATIVE_URL_QUERY_RE = re.compile(r"(?P<path>/[A-Za-z0-9._~:/%-]+)\?[^\s'\"<>),]+")
_SECRET_FIELD_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|apikey|token|access[_-]?token|secret|authorization|bearer|key)"
    r"\s*[:=]\s*[^\s,;&)\]}'\"]+"
)
_SECRET_QUERY_RE = re.compile(
    r"(?i)([?&])(?:api[_-]?key|apikey|token|access[_-]?token|secret|authorization|key)="
    r"[^&#\s,;)\]}'\"]+"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[^\s,;&)\]}'\"]+")

ADVANCED_CONTEXT_FIELDS = (
    "universe_pct_above_sma_20d",
    "universe_pct_above_sma_50d",
    "universe_pct_making_20d_highs",
    "universe_pct_making_52w_highs",
    "volatility_regime_proxy",
    "volatility_regime_source",
    "market_regime_status",
    "opening_range_high_5m",
    "opening_range_low_5m",
    "opening_range_high_15m",
    "opening_range_low_15m",
    "opening_range_high_30m",
    "opening_range_low_30m",
    "opening_range_high_60m",
    "opening_range_low_60m",
    "first_5m_return",
    "first_15m_return",
    "first_30m_return",
    "first_60m_return",
    "intraday_vwap",
    "open_vs_intraday_vwap_pct",
    "close_vs_intraday_vwap_pct",
    "intraday_volume_5m",
    "intraday_volume_15m",
    "intraday_volume_30m",
    "intraday_volume_60m",
    "pct_expected_volume_5m",
    "pct_expected_volume_15m",
    "pct_expected_volume_30m",
    "pct_expected_volume_60m",
    "held_above_breakout_after_first_hour",
    "intraday_mfe_timestamp",
    "intraday_mae_timestamp",
    "t1_before_stop",
    "intraday_structure_status",
    "missing_intraday_bars",
    "bid_ask_spread",
    "bid_ask_spread_pct",
    "quote_age_seconds",
    "bid_size",
    "ask_size",
    "intended_entry_vs_mid_pct",
    "intended_entry_vs_ask_pct",
    "intended_entry_vs_bid_pct",
    "volume_participation_pct",
    "halt_risk_flag",
    "offering_risk_flag",
    "missing_quote",
    "stale_quote",
    "quote_status",
    "execution_quality_status",
    "float_shares",
    "shares_outstanding",
    "turnover_float",
    "dollar_turnover_float",
    "short_volume_ratio",
    "short_interest_pct_float",
    "short_interest_shares",
    "short_interest_days_to_cover",
    "proxy_days_to_cover",
    "borrow_fee_rate",
    "float_source_status",
    "short_source_status",
    "borrow_fee_status",
    "supply_squeeze_status",
    "news_count_1d",
    "news_count_5d",
    "news_count_20d",
    "news_catalyst_flags_json",
    "earnings_days_to_next",
    "earnings_days_since_last",
    "offering_flag",
    "atm_flag",
    "shelf_registration_flag",
    "insider_buy_overlap_m2",
    "cofire_m1",
    "cofire_m2",
    "cofire_m3",
    "cofire_m4",
    "cofire_i11",
    "fda_clinical_flag",
    "corporate_action_flag",
    "cross_pattern_overlap_count",
    "strongest_overlap_pattern_id",
    "catalyst_context_status",
    "missing_catalyst_source",
    "rsi_2",
    "rsi_5",
    "rsi_14",
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "bollinger_bandwidth_20d",
    "bollinger_percent_b_20d",
    "keltner_channel_position_20d",
    "macd_histogram",
    "obv",
    "accumulation_distribution",
    "chaikin_money_flow_20d",
    "stochastic_oscillator_14d",
    "technical_indicator_status",
)

SIGNAL_SESSION_PREDICTOR_FIELDS = (
    "sigma_20d",
    "realized_volatility_5d",
    "realized_volatility_10d",
    "realized_volatility_20d",
    "base_range_10d",
    "base_range_20d",
    "base_range_60d",
    "base_max_drawdown_10d",
    "base_max_drawdown_20d",
    "base_max_drawdown_60d",
    "distance_from_sma_20d",
    "distance_from_sma_50d",
    "distance_from_sma_200d",
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",
    "prior_52w_high_touches_20d",
    "prior_52w_high_touches_60d",
    "prior_52w_high_touches_126d",
    "age_of_52w_high_sessions",
    "failed_breakout_count_20d",
    "failed_breakout_count_60d",
    "failed_breakout_count_126d",
    "rsi_2",
    "rsi_5",
    "rsi_14",
    "adx_14",
    "atr_14_pct",
    "range_expansion_vs_20d",
    "volume_expansion_20d",
    "volume_expansion_60d",
    "dollar_volume",
    "dollar_volume_expansion_20d",
    "dollar_volume_expansion_60d",
    "liquidity_proxy_score",
    "relative_strength_vs_spy_5d",
    "relative_strength_vs_spy_20d",
    "relative_strength_vs_spy_60d",
    "relative_strength_vs_qqq_5d",
    "relative_strength_vs_qqq_20d",
    "relative_strength_vs_qqq_60d",
    "relative_strength_vs_iwm_5d",
    "relative_strength_vs_iwm_20d",
    "relative_strength_vs_iwm_60d",
    "relative_strength_vs_sector_5d",
    "relative_strength_vs_sector_20d",
    "relative_strength_vs_sector_60d",
    "prior_close_vs_52w_high_pct",
)
SIGNAL_SESSION_OUTCOME_FIELDS = (
    "previous_close",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "open_to_close_return",
    "gap_pct",
    "breakout_extension_pct",
    "open_vs_52w_high_pct",
    "close_vs_52w_high_pct",
    "high_vs_52w_high_pct",
    "closed_above_breakout",
    "gap_over_breakout",
    "high_from_open_return",
    "low_from_open_return",
)

ML_OUTPUT_HASH_FIELDS = (
    "pattern_id",
    "ticker",
    "signal_horizon",
    "signal_date",
    "entry_session_date",
    "feature_session_date",
    "path_sequence",
    "feature_role",
    "feature_version",
    "reconstruction_method",
    "previous_close",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "split_adjusted_close",
    "adj_close",
    "dollar_volume",
    "median_volume_20d",
    "median_volume_60d",
    "median_dollar_volume_20d",
    "median_dollar_volume_60d",
    "volume_expansion_20d",
    "volume_expansion_60d",
    "dollar_volume_expansion_20d",
    "dollar_volume_expansion_60d",
    "gap_pct",
    "open_to_close_return",
    "high_from_open_return",
    "low_from_open_return",
    "return_from_entry_open",
    "return_from_entry_high",
    "return_from_entry_low",
    "return_from_entry_close",
    "sigma_20d",
    "effective_hard_stop_pct",
    "liquidity_proxy_score",
    "liquidity_proxy_passed",
    "prior_52w_high",
    "breakout_extension_pct",
    "open_vs_52w_high_pct",
    "close_vs_52w_high_pct",
    "high_vs_52w_high_pct",
    "gap_over_breakout",
    "closed_above_breakout",
    "close_location_value",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "true_range_pct",
    "atr_14_pct",
    "range_expansion_vs_20d",
    "volume_zscore_20d",
    "volume_zscore_60d",
    "dollar_volume_zscore_20d",
    "dollar_volume_zscore_60d",
    "volume_acceleration_1d_vs_5d",
    "volume_acceleration_1d_vs_20d",
    "realized_volatility_5d",
    "realized_volatility_10d",
    "realized_volatility_20d",
    "base_range_10d",
    "base_range_20d",
    "base_range_60d",
    "base_max_drawdown_10d",
    "base_max_drawdown_20d",
    "base_max_drawdown_60d",
    "distance_from_sma_20d",
    "distance_from_sma_50d",
    "distance_from_sma_200d",
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",
    "prior_52w_high_touches_20d",
    "prior_52w_high_touches_60d",
    "prior_52w_high_touches_126d",
    "age_of_52w_high_sessions",
    "failed_breakout_count_20d",
    "failed_breakout_count_60d",
    "failed_breakout_count_126d",
    "vwap",
    "open_vs_vwap_pct",
    "high_vs_vwap_pct",
    "low_vs_vwap_pct",
    "close_vs_vwap_pct",
    "dollar_volume_rank",
    "dollar_volume_percentile",
    "volume_expansion_20d_rank",
    "volume_expansion_20d_percentile",
    "volume_expansion_60d_rank",
    "volume_expansion_60d_percentile",
    "dollar_volume_expansion_20d_rank",
    "dollar_volume_expansion_20d_percentile",
    "dollar_volume_expansion_60d_rank",
    "dollar_volume_expansion_60d_percentile",
    "liquidity_proxy_rank",
    "liquidity_proxy_percentile",
    "cohort_feature_row_count",
    "cohort_pattern_row_count",
    "spy_return_1d",
    "spy_return_5d",
    "spy_return_20d",
    "spy_return_60d",
    "qqq_return_1d",
    "qqq_return_5d",
    "qqq_return_20d",
    "qqq_return_60d",
    "iwm_return_1d",
    "iwm_return_5d",
    "iwm_return_20d",
    "iwm_return_60d",
    "relative_strength_vs_spy_5d",
    "relative_strength_vs_spy_20d",
    "relative_strength_vs_spy_60d",
    "relative_strength_vs_qqq_5d",
    "relative_strength_vs_qqq_20d",
    "relative_strength_vs_qqq_60d",
    "relative_strength_vs_iwm_5d",
    "relative_strength_vs_iwm_20d",
    "relative_strength_vs_iwm_60d",
    "sector_etf",
    "sector_etf_return_5d",
    "sector_etf_return_20d",
    "sector_etf_return_60d",
    "relative_strength_vs_sector_5d",
    "relative_strength_vs_sector_20d",
    "relative_strength_vs_sector_60d",
    "sector_source",
    "sector_relative_status",
    *ADVANCED_CONTEXT_FIELDS,
    "opening_range_json",
    "intraday_continuation_json",
    "quote_spread_json",
    "feature_json",
)


@dataclass(frozen=True)
class _CleanBar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    split_adjusted_close: float | None
    adj_close: float | None
    vwap: float | None

    @property
    def dollar_volume(self) -> float:
        basis = self.split_adjusted_close if self.split_adjusted_close is not None else self.close
        return basis * self.volume


@dataclass(frozen=True)
class _ReferenceSeries:
    symbol: str
    bars: tuple[_CleanBar, ...]
    data_lineage_id: str | None
    raw_payload_hash: str | None
    status: str
    error: dict[str, Any] | None = None


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
class _SectorResolution:
    sector: str | None
    sector_etf: str | None
    source: str | None
    status: str
    pit_safe: bool


@dataclass
class _NonSessionBarSkipTracker:
    count: int = 0
    samples: list[dict[str, str]] = field(default_factory=list)

    def record(self, *, ticker: str, bar_date: date) -> None:
        self.count += 1
        if len(self.samples) < 10:
            self.samples.append({
                "ticker": ticker.upper(),
                "date": bar_date.isoformat(),
            })


@dataclass
class MarketPathFeatureCollection:
    decision_date: date | None
    signal_start: date | None
    signal_end: date | None
    through_date: date | None
    feature_version: str
    pattern_ids: tuple[str, ...]
    signal_source: str = SIGNAL_SOURCE_LIVE
    signals_scanned: int = 0
    ticker_planned_count: int = 0
    ticker_fetch_started_count: int = 0
    ticker_fetch_finished_count: int = 0
    ticker_fetch_error_count: int = 0
    ticker_fetch_count: int = 0
    lineages_recorded: int = 0
    rows_inserted: int = 0
    rows_updated: int = 0
    rows_unchanged: int = 0
    rows_skipped: int = 0
    missing_entry_rows: int = 0
    watchdog_timeouts: int = 0
    benchmark_fetch_count: int = 0
    benchmark_fetch_error_count: int = 0
    sector_etf_fetch_count: int = 0
    sector_etf_fetch_error_count: int = 0
    same_day_pattern_strength_key_count: int = 0
    non_session_bars_skipped: int = 0
    non_session_bar_skip_sample: list[dict[str, str]] | None = None
    pending_lineages: list[DataLineage] | None = None
    pending_feature_rows: list[dict[str, Any]] | None = None
    fetch_errors: list[dict[str, Any]] | None = None
    errors: list[dict[str, Any]] | None = None
    stage_timings: dict[str, float] | None = None
    watchdog_state: dict[str, Any] | None = None
    no_op_reason: str | None = None
    skip_existing: bool = False
    polygon_minute_layer_enabled: bool = False

    @property
    def status(self) -> str:
        if self.errors:
            return "failed"
        if self.fetch_errors:
            return "partial_failed"
        return "finished"

    @property
    def rows_upserted(self) -> int:
        return len(self.pending_feature_rows or [])

    def metrics(self) -> dict[str, Any]:
        metrics = {
            "decision_date": self.decision_date.isoformat() if self.decision_date else None,
            "pattern_ids": list(self.pattern_ids),
            "signal_source": self.signal_source,
            "signal_start_date": self.signal_start.isoformat() if self.signal_start else None,
            "signal_end_date": self.signal_end.isoformat() if self.signal_end else None,
            "through_date": self.through_date.isoformat() if self.through_date else None,
            "feature_version": self.feature_version,
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "skip_existing": self.skip_existing,
            "polygon_minute_layer_enabled": self.polygon_minute_layer_enabled,
            "signals_scanned": self.signals_scanned,
            "ticker_planned_count": self.ticker_planned_count,
            "ticker_fetch_started_count": self.ticker_fetch_started_count,
            "ticker_fetch_finished_count": self.ticker_fetch_finished_count,
            "ticker_fetch_error_count": self.ticker_fetch_error_count,
            "ticker_fetch_count": self.ticker_fetch_count,
            "lineages_recorded": self.lineages_recorded,
            "rows_inserted": self.rows_inserted,
            "rows_updated": self.rows_updated,
            "rows_unchanged": self.rows_unchanged,
            "rows_upserted": self.rows_upserted,
            "rows_skipped": self.rows_skipped,
            "missing_entry_row_count": self.missing_entry_rows,
            "fetch_error_count": len(self.fetch_errors or []),
            "watchdog_timeouts": self.watchdog_timeouts,
            **(self.watchdog_state or {}),
            "benchmark_fetch_count": self.benchmark_fetch_count,
            "benchmark_fetch_error_count": self.benchmark_fetch_error_count,
            "sector_etf_fetch_count": self.sector_etf_fetch_count,
            "sector_etf_fetch_error_count": self.sector_etf_fetch_error_count,
            "same_day_pattern_strength_key_count": self.same_day_pattern_strength_key_count,
            "non_session_bars_skipped": self.non_session_bars_skipped,
            "non_session_bar_skip_sample": self.non_session_bar_skip_sample or [],
            "stage_timing_seconds": self.stage_timings or {},
        }
        if self.no_op_reason:
            metrics["no_op_reason"] = self.no_op_reason
        return metrics


class MarketPathFeatureJob(BaseJob):
    """Compute and persist daily market-path features for existing signals."""

    def __init__(
        self,
        *,
        session: Session,
        fmp_adapter: FmpAdapter,
        run_timestamp: datetime | None = None,
        pattern_ids: Sequence[str] = ("M4",),
        decision_date: date | None = None,
        signal_start_date: date | None = None,
        signal_end_date: date | None = None,
        through_date: date | None = None,
        lookback_calendar_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS,
        feature_version: str = FEATURE_VERSION,
        include_signal_session: bool = False,
        liquidity_min_dollar_volume_20d: float = 100_000.0,
        liquidity_min_price: float = 1.0,
        progress_callback: ProgressCallback | None = None,
        progress_every: int = 10,
        max_fetch_concurrency: int = 1,
        signal_source: str = SIGNAL_SOURCE_LIVE,
        polygon_minute_adapter: Any | None = None,
        fetch_deadline_seconds: float = DEFAULT_FETCH_DEADLINE_SECONDS,
        max_outstanding_fetch_timeouts: int = DEFAULT_MAX_OUTSTANDING_FETCH_TIMEOUTS,
        skip_existing: bool = False,
    ) -> None:
        if progress_every < 1:
            raise ValueError("progress_every must be >= 1")
        if max_fetch_concurrency < 1:
            raise ValueError("max_fetch_concurrency must be >= 1")
        if fetch_deadline_seconds <= 0:
            raise ValueError("fetch_deadline_seconds must be > 0")
        if max_outstanding_fetch_timeouts < 1:
            raise ValueError("max_outstanding_fetch_timeouts must be >= 1")
        self._session = session
        self._fmp = fmp_adapter
        self._run_timestamp = run_timestamp
        self._pattern_ids = tuple(pid.upper() for pid in pattern_ids)
        self._decision_date = decision_date
        self._signal_start_date = signal_start_date
        self._signal_end_date = signal_end_date
        self._through_date = through_date
        self._lookback_calendar_days = lookback_calendar_days
        self._feature_version = feature_version
        self._include_signal_session = include_signal_session
        self._liquidity_min_dollar_volume_20d = liquidity_min_dollar_volume_20d
        self._liquidity_min_price = liquidity_min_price
        self._progress_callback = progress_callback
        self._progress_every = progress_every
        self._max_fetch_concurrency = max_fetch_concurrency
        self._signal_source = normalize_signal_source(signal_source)
        self._polygon_minute_adapter = polygon_minute_adapter
        self._fetch_deadline_seconds = float(fetch_deadline_seconds)
        self._fetch_watchdog = WatchdogState(
            max_outstanding_timeouts=max_outstanding_fetch_timeouts,
            max_consecutive_timeouts=max_outstanding_fetch_timeouts,
        )
        self._skip_existing = skip_existing

    @property
    def job_name(self) -> str:
        return JOB_NAME

    @property
    def job_type(self) -> str:
        return "feature_enrichment"

    def run(self, ctx: JobContext) -> JobResult:
        try:
            collection = self.collect_feature_rows(ctx)
        except ProviderOutageCircuitBreaker as exc:
            return JobResult(
                status="failed",
                metrics={
                    "feature_version": self._feature_version,
                    "pattern_ids": list(self._pattern_ids),
                    "fetch_deadline_seconds": self._fetch_deadline_seconds,
                    **self._fetch_watchdog.snapshot(),
                },
                errors=[exc.payload],
            )
        if collection.errors:
            return JobResult(
                status="failed",
                metrics=collection.metrics(),
                errors=collection.errors,
            )

        stage_timings = collection.stage_timings or {}
        started = perf_counter()
        if collection.pending_lineages:
            self._session.add_all(collection.pending_lineages)
            self._session.flush()
        _record_timing(stage_timings, "batched_lineage_flush_seconds", started)

        started = perf_counter()
        _bulk_upsert_market_path_features(
            self._session,
            collection.pending_feature_rows or [],
        )
        if collection.pending_feature_rows:
            self._session.expire_all()
        _record_timing(stage_timings, "row_upsert_persist_seconds", started)

        rank_rows_updated = 0
        if collection.signal_start is not None and collection.through_date is not None:
            started = perf_counter()
            rank_rows_updated = self._populate_cross_sectional_ranks(
                start_date=collection.signal_start,
                through_date=collection.through_date,
                progress_callback=self._emit_progress,
                progress_every=self._progress_every,
            )
            _record_timing(stage_timings, "cross_sectional_rank_pass_seconds", started)

        metrics = collection.metrics()
        metrics["rank_rows_updated"] = rank_rows_updated
        metrics["liquidity_proxy_min_dollar_volume_20d"] = (
            self._liquidity_min_dollar_volume_20d
        )
        metrics["liquidity_proxy_min_price"] = self._liquidity_min_price
        metrics["stage_timing_seconds"] = stage_timings
        return JobResult(
            status=collection.status,
            metrics=metrics,
            errors=collection.fetch_errors or [],
        )

    def collect_feature_rows(self, ctx: JobContext) -> MarketPathFeatureCollection:
        job_started = perf_counter()
        stage_timings: dict[str, float] = {}
        run_ts = _ensure_aware(self._run_timestamp or ctx.started_at)
        session_resolution = resolve_us_equity_session(run_ts)
        through_date = self._through_date or date.fromisoformat(
            session_resolution.evidence_session_date
        )
        signal_start = self._signal_start_date or through_date
        signal_end = self._signal_end_date or through_date
        decision_date = self._decision_date or signal_end
        if signal_start > signal_end:
            return MarketPathFeatureCollection(
                decision_date=decision_date,
                signal_start=signal_start,
                signal_end=signal_end,
                through_date=through_date,
                feature_version=self._feature_version,
                pattern_ids=self._pattern_ids,
                signal_source=self._signal_source,
                skip_existing=self._skip_existing,
                polygon_minute_layer_enabled=self._polygon_minute_adapter is not None,
                errors=[{
                    "stage": "args",
                    "message": "signal_start_date must be on or before signal_end_date",
                }],
            )
        if self._lookback_calendar_days < 70:
            return MarketPathFeatureCollection(
                decision_date=decision_date,
                signal_start=signal_start,
                signal_end=signal_end,
                through_date=through_date,
                feature_version=self._feature_version,
                pattern_ids=self._pattern_ids,
                signal_source=self._signal_source,
                skip_existing=self._skip_existing,
                polygon_minute_layer_enabled=self._polygon_minute_adapter is not None,
                errors=[{
                    "stage": "args",
                    "message": "lookback_calendar_days must be at least 70",
                }],
            )

        self._emit_progress(
            "signal_load_start",
            {
                "pattern_ids": list(self._pattern_ids),
                "signal_source": self._signal_source,
                "signal_start_date": signal_start.isoformat(),
                "signal_end_date": signal_end.isoformat(),
            },
        )
        started = perf_counter()
        signals = self._signals(signal_start, signal_end)
        _record_timing(stage_timings, "signal_load_seconds", started)
        self._emit_progress(
            "signal_load_finish",
            {
                "signals_loaded": len(signals),
                "elapsed_seconds": _elapsed_since(started),
            },
        )
        if not signals:
            _record_timing(stage_timings, "job_internal_total_seconds", job_started)
            return MarketPathFeatureCollection(
                decision_date=decision_date,
                signal_start=signal_start,
                signal_end=signal_end,
                through_date=through_date,
                feature_version=self._feature_version,
                pattern_ids=self._pattern_ids,
                signal_source=self._signal_source,
                skip_existing=self._skip_existing,
                polygon_minute_layer_enabled=self._polygon_minute_adapter is not None,
                pending_lineages=[],
                pending_feature_rows=[],
                fetch_errors=[],
                stage_timings=stage_timings,
                no_op_reason="no_matching_signals",
            )

        rows_inserted = 0
        rows_updated = 0
        rows_unchanged = 0
        rows_skipped = 0
        missing_entry_rows = 0
        watchdog_timeouts = 0
        fetch_errors: list[dict[str, Any]] = []
        lineages_recorded = 0
        tickers_fetched = 0
        non_session_tracker = _NonSessionBarSkipTracker()
        pending_lineages: list[DataLineage] = []
        pending_feature_rows: list[dict[str, Any]] = []
        reference_from_date = _fetch_start(signals, self._lookback_calendar_days)
        self._emit_progress(
            "reference_fetch_start",
            {
                "source_role": "market_benchmark",
                "symbol_count": len(BENCHMARK_SYMBOLS),
                "from_date": reference_from_date.isoformat(),
                "through_date": through_date.isoformat(),
            },
        )
        started = perf_counter()
        benchmark_series = self._fetch_reference_series(
            BENCHMARK_SYMBOLS,
            from_date=reference_from_date,
            through_date=through_date,
            run_ts=run_ts,
            job_run_id=ctx.job_run_id,
            source_role="market_benchmark",
            pending_lineages=pending_lineages,
            non_session_tracker=non_session_tracker,
        )
        _record_timing(stage_timings, "benchmark_reference_fetch_seconds", started)
        self._emit_progress(
            "reference_fetch_finish",
            {
                "source_role": "market_benchmark",
                "symbol_count": len(BENCHMARK_SYMBOLS),
                "fetch_error_count": _reference_error_count(benchmark_series),
                "elapsed_seconds": _elapsed_since(started),
            },
        )
        started = perf_counter()
        sector_resolver = _SectorResolver(self._session)
        _record_timing(stage_timings, "sector_resolver_table_check_seconds", started)
        started = perf_counter()
        needed_sector_etfs = self._needed_sector_etfs(
            signals,
            through_date=through_date,
            sector_resolver=sector_resolver,
        )
        _record_timing(stage_timings, "sector_etf_resolution_seconds", started)
        self._emit_progress(
            "reference_fetch_start",
            {
                "source_role": "sector_etf",
                "symbol_count": len(needed_sector_etfs),
                "from_date": reference_from_date.isoformat(),
                "through_date": through_date.isoformat(),
            },
        )
        started = perf_counter()
        sector_etf_series = self._fetch_reference_series(
            needed_sector_etfs,
            from_date=reference_from_date,
            through_date=through_date,
            run_ts=run_ts,
            job_run_id=ctx.job_run_id,
            source_role="sector_etf",
            pending_lineages=pending_lineages,
            non_session_tracker=non_session_tracker,
        )
        _record_timing(stage_timings, "sector_reference_fetch_seconds", started)
        self._emit_progress(
            "reference_fetch_finish",
            {
                "source_role": "sector_etf",
                "symbol_count": len(needed_sector_etfs),
                "fetch_error_count": _reference_error_count(sector_etf_series),
                "elapsed_seconds": _elapsed_since(started),
            },
        )

        started = perf_counter()
        by_ticker = _group_by_ticker(signals)
        _record_timing(stage_timings, "signal_grouping_seconds", started)
        self._emit_progress(
            "tickers_planned",
            {
                "ticker_count": len(by_ticker),
                "max_fetch_concurrency_requested": self._max_fetch_concurrency,
                "max_fetch_concurrency_effective": 1,
            },
        )
        started = perf_counter()
        same_day_pattern_strengths = _prefetch_same_day_pattern_strengths(
            self._session,
            signals,
        )
        _record_timing(stage_timings, "cofire_prefetch_seconds", started)
        started = perf_counter()
        existing = self._existing_rows(signals)
        _record_timing(stage_timings, "existing_row_lookup_seconds", started)
        ticker_fetch_started = 0
        ticker_fetch_finished = 0
        ticker_fetch_error_count = 0
        for ticker, ticker_signals in by_ticker.items():
            from_date = _fetch_start(ticker_signals, self._lookback_calendar_days)
            ticker_fetch_started += 1
            self._emit_ticker_progress(
                "ticker_fetch_start",
                ticker=ticker,
                started=ticker_fetch_started,
                finished=ticker_fetch_finished,
                errors=ticker_fetch_error_count,
                total=len(by_ticker),
                from_date=from_date,
                through_date=through_date,
            )
            started = perf_counter()
            resp = self._fmp_historical_price_with_deadline(
                ticker,
                from_date=from_date,
                to_date=through_date,
                asof=run_ts,
                adjusted=False,
                stage="fmp_historical_price",
            )
            _record_timing(stage_timings, "ticker_fmp_fetch_seconds", started)
            tickers_fetched += 1
            ticker_fetch_finished += 1
            if not resp.ok or resp.data is None:
                ticker_fetch_error_count += 1
                fetch_error = {
                    "ticker": ticker,
                    "stage": "fmp_historical_price",
                    "message": sanitize_provider_error_message(
                        getattr(resp.error, "message", "missing response")
                    ),
                    "error_type": getattr(resp.error, "error_type", None),
                    "retryable": getattr(resp.error, "retryable", None),
                    "status_code": getattr(resp.error, "status_code", None),
                    "provider": getattr(resp.error, "provider", None),
                    **_retry_metadata_from_lineage(resp),
                }
                fetch_errors.append(fetch_error)
                self._emit_ticker_progress(
                    "ticker_fetch_error",
                    ticker=ticker,
                    started=ticker_fetch_started,
                    finished=ticker_fetch_finished,
                    errors=ticker_fetch_error_count,
                    total=len(by_ticker),
                    from_date=from_date,
                    through_date=through_date,
                    elapsed_seconds=_elapsed_since(started),
                    message=sanitize_provider_error_message(
                        getattr(resp.error, "message", "missing response")
                    ),
                    error_type=getattr(resp.error, "error_type", None),
                    retry_attempt_count=fetch_error.get("retry_attempt_count"),
                    retry_exhausted=fetch_error.get("retry_exhausted"),
                )
                continue
            self._emit_ticker_progress(
                "ticker_fetch_finish",
                ticker=ticker,
                started=ticker_fetch_started,
                finished=ticker_fetch_finished,
                errors=ticker_fetch_error_count,
                total=len(by_ticker),
                from_date=from_date,
                through_date=through_date,
                elapsed_seconds=_elapsed_since(started),
                bar_count=len(resp.data or []),
            )
            started = perf_counter()
            bars = _clean_bars(
                resp.data,
                ticker=ticker,
                non_session_tracker=non_session_tracker,
            )
            _record_timing(stage_timings, "ticker_bar_clean_seconds", started)
            started = perf_counter()
            lineage = _build_data_lineage(
                provider="FMP",
                endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                asof_timestamp=run_ts,
                raw_payload=_bar_lineage_payload(
                    symbol=ticker,
                    from_date=from_date,
                    through_date=through_date,
                    bars=bars,
                    feature_version=self._feature_version,
                    symbol_field="ticker",
                ),
                source_authority="fmp_eod",
                data_quality_flags=_lineage_quality_flags(
                    resp,
                    derived_feature_replay=True,
                    lineage_payload_schema="compact_bar_digest_v1",
                    adapter_raw_payload_hash=resp.lineage.raw_payload_hash,
                ),
                job_run_id=ctx.job_run_id,
            )
            pending_lineages.append(lineage)
            _record_timing(stage_timings, "ticker_lineage_record_seconds", started)
            lineages_recorded += 1
            started = perf_counter()
            minute_bars_by_date, minute_watchdog_timeouts = self._fetch_ticker_signal_minutes(
                ticker,
                signal_dates=sorted({
                    signal.signal_timestamp.date()
                    for signal in ticker_signals
                    if self._include_signal_session
                }),
                fetch_errors=fetch_errors,
            )
            watchdog_timeouts += minute_watchdog_timeouts
            for signal in ticker_signals:
                result = self._persist_signal_rows(
                    signal,
                    bars=bars,
                    minute_bars_by_date=minute_bars_by_date,
                    through_date=through_date,
                    data_lineage_id=lineage.data_lineage_id,
                    job_run_id=ctx.job_run_id,
                    existing=existing,
                    benchmark_series=benchmark_series,
                    sector_resolver=sector_resolver,
                    sector_etf_series=sector_etf_series,
                    same_day_pattern_strengths=same_day_pattern_strengths,
                    pending_feature_rows=pending_feature_rows,
                    stage_timings=stage_timings,
                )
                rows_inserted += result["inserted"]
                rows_updated += result["updated"]
                rows_unchanged += result["unchanged"]
                rows_skipped += result["skipped"]
                missing_entry_rows += result["missing_entry"]
            _record_timing(stage_timings, "per_ticker_feature_persist_seconds", started)
            self._emit_progress(
                "feature_rows_generated",
                {
                    "ticker": ticker,
                    "pending_feature_rows": len(pending_feature_rows),
                    "inserted": rows_inserted,
                    "updated": rows_updated,
                    "unchanged": rows_unchanged,
                    "skipped": rows_skipped,
                },
            )

        _record_timing(stage_timings, "job_internal_total_seconds", job_started)
        return MarketPathFeatureCollection(
            decision_date=decision_date,
            signal_start=signal_start,
            signal_end=signal_end,
            through_date=through_date,
            feature_version=self._feature_version,
            pattern_ids=self._pattern_ids,
            signal_source=self._signal_source,
            skip_existing=self._skip_existing,
            polygon_minute_layer_enabled=self._polygon_minute_adapter is not None,
            signals_scanned=len(signals),
            ticker_planned_count=len(by_ticker),
            ticker_fetch_started_count=ticker_fetch_started,
            ticker_fetch_finished_count=ticker_fetch_finished,
            ticker_fetch_error_count=ticker_fetch_error_count,
            ticker_fetch_count=tickers_fetched,
            lineages_recorded=lineages_recorded,
            rows_inserted=rows_inserted,
            rows_updated=rows_updated,
            rows_unchanged=rows_unchanged,
            rows_skipped=rows_skipped,
            missing_entry_rows=missing_entry_rows,
            watchdog_timeouts=self._fetch_watchdog.total_timeouts,
            benchmark_fetch_count=len(BENCHMARK_SYMBOLS),
            benchmark_fetch_error_count=_reference_error_count(benchmark_series),
            sector_etf_fetch_count=len(sector_etf_series),
            sector_etf_fetch_error_count=_reference_error_count(sector_etf_series),
            same_day_pattern_strength_key_count=len(same_day_pattern_strengths),
            non_session_bars_skipped=non_session_tracker.count,
            non_session_bar_skip_sample=list(non_session_tracker.samples),
            pending_lineages=pending_lineages,
            pending_feature_rows=pending_feature_rows,
            fetch_errors=fetch_errors,
            stage_timings=stage_timings,
            watchdog_state=self._fetch_watchdog.snapshot(),
        )

    def _emit_progress(self, event: str, payload: dict[str, Any]) -> None:
        if self._progress_callback is None:
            return
        payload = dict(payload)
        payload.setdefault("wall_clock_utc", _utc_progress_timestamp())
        try:
            self._progress_callback(event, payload)
        except Exception:
            return

    def _emit_ticker_progress(
        self,
        event: str,
        *,
        ticker: str,
        started: int,
        finished: int,
        errors: int,
        total: int,
        from_date: date,
        through_date: date,
        **extra: Any,
    ) -> None:
        should_emit = (
            event == "ticker_fetch_error"
            or started == 1
            or finished == total
            or finished % self._progress_every == 0
        )
        if not should_emit:
            return
        self._emit_progress(
            event,
            {
                "ticker": ticker,
                "ticker_fetch_started_count": started,
                "ticker_fetch_finished_count": finished,
                "ticker_fetch_error_count": errors,
                "ticker_count": total,
                "from_date": from_date.isoformat(),
                "through_date": through_date.isoformat(),
                **self._fetch_watchdog.snapshot(),
                **extra,
            },
        )

    def _signals(self, start: date, end: date) -> list[SignalRegistry]:
        start_dt = datetime.combine(start, time.min, timezone.utc)
        end_dt = datetime.combine(end + timedelta(days=1), time.min, timezone.utc)
        query = (
            self._session.query(SignalRegistry)
            .filter(
                SignalRegistry.pattern_id.in_(self._pattern_ids),
                SignalRegistry.signal_timestamp >= start_dt,
                SignalRegistry.signal_timestamp < end_dt,
            )
        )
        query = apply_signal_source_filter(
            query,
            self._session,
            signal_source=self._signal_source,
            signal_start_date=start,
            signal_end_date=end,
        )
        return query.order_by(SignalRegistry.ticker, SignalRegistry.signal_timestamp).all()

    def _existing_rows(
        self,
        signals: Sequence[SignalRegistry],
    ) -> dict[tuple[str, str], MarketPathFeature]:
        signal_ids = [signal.signal_id for signal in signals]
        rows = (
            self._session.query(MarketPathFeature)
            .filter(
                MarketPathFeature.signal_id.in_(signal_ids),
                MarketPathFeature.feature_version == self._feature_version,
            )
            .all()
        )
        return {
            (row.signal_id, row.feature_session_date): row
            for row in rows
        }

    def _fmp_historical_price_with_deadline(
        self,
        ticker: str,
        *,
        from_date: date,
        to_date: date,
        asof: datetime,
        adjusted: bool,
        stage: str,
    ) -> Any:
        self._emit_progress(
            "fmp_fetch_start",
            {
                "ticker": ticker.upper(),
                "from_date": from_date.isoformat(),
                "to_date": to_date.isoformat(),
                "stage": stage,
                "deadline_seconds": self._fetch_deadline_seconds,
                **self._fetch_watchdog.snapshot(),
            },
        )

        def _fetch(
            ticker: str = ticker,
            from_date: date = from_date,
            to_date: date = to_date,
            asof: datetime = asof,
            adjusted: bool = adjusted,
        ) -> Any:
            return self._fmp.get_historical_price(
                ticker,
                from_date=from_date,
                to_date=to_date,
                asof=asof,
                adjusted=adjusted,
            )

        try:
            return call_with_daemon_deadline(
                _fetch,
                timeout_seconds=self._fetch_deadline_seconds,
                thread_name="market-path-fmp-fetch",
                state=self._fetch_watchdog,
                context={
                    "ticker": ticker.upper(),
                    "stage": stage,
                    "deadline_seconds": self._fetch_deadline_seconds,
                },
            )
        except FuturesTimeoutError:
            return AdapterResponse(
                data=None,
                lineage=LineageMeta(
                    provider="FMP",
                    endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                    request_timestamp=datetime.now(timezone.utc),
                    asof_timestamp=asof,
                    raw_payload_hash="",
                    data_quality_flags=self._fetch_watchdog.snapshot(),
                ),
                error=ProviderError(
                    provider="FMP",
                    endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                    status_code=None,
                    error_type="watchdog_timeout",
                    message="FMP historical price fetch exceeded watchdog deadline",
                    retryable=True,
                ),
            )

    def _fetch_ticker_signal_minutes(
        self,
        ticker: str,
        *,
        signal_dates: Sequence[date],
        fetch_errors: list[dict[str, Any]],
    ) -> tuple[dict[date, tuple[_MinuteBar, ...]], int]:
        if self._polygon_minute_adapter is None or not signal_dates:
            return {}, 0
        out: dict[date, tuple[_MinuteBar, ...]] = {}
        watchdog_timeouts = 0
        for trading_date in signal_dates:
            self._emit_progress(
                "minute_fetch_start",
                {
                    "ticker": ticker,
                    "trading_date": trading_date.isoformat(),
                    "deadline_seconds": self._fetch_deadline_seconds,
                    "cache_status": "unknown",
                    **self._fetch_watchdog.snapshot(),
                },
            )

            def _fetch(
                ticker: str = ticker,
                trading_date: date = trading_date,
            ) -> Any:
                return self._polygon_minute_adapter.get_minute_aggs(
                    ticker,
                    from_date=trading_date.isoformat(),
                    to_date=trading_date.isoformat(),
                    adjusted=True,
                )

            try:
                resp = call_with_daemon_deadline(
                    _fetch,
                    timeout_seconds=self._fetch_deadline_seconds,
                    thread_name="market-path-polygon-minute-fetch",
                    state=self._fetch_watchdog,
                    context={
                        "ticker": ticker.upper(),
                        "trading_date": trading_date.isoformat(),
                        "stage": "polygon_minute_fetch",
                        "deadline_seconds": self._fetch_deadline_seconds,
                    },
                )
            except FuturesTimeoutError:
                watchdog_timeouts += 1
                self._maybe_reset_polygon_session()
                fetch_error = {
                    "ticker": ticker.upper(),
                    "trading_date": trading_date.isoformat(),
                    "stage": "polygon_minute_fetch",
                    "error": "fetch_watchdog_timeout",
                    "deadline_seconds": self._fetch_deadline_seconds,
                    **self._fetch_watchdog.snapshot(),
                }
                fetch_errors.append(fetch_error)
                self._emit_progress("fetch_watchdog_timeout", fetch_error)
                out[trading_date] = ()
                continue
            except ProviderOutageCircuitBreaker:
                raise
            except Exception:
                fetch_errors.append({
                    "ticker": ticker.upper(),
                    "trading_date": trading_date.isoformat(),
                    "stage": "polygon_minute_fetch",
                    "error": "minute_fetch_exception",
                })
                out[trading_date] = ()
                continue
            if not getattr(resp, "ok", False) or getattr(resp, "data", None) is None:
                fetch_errors.append({
                    "ticker": ticker.upper(),
                    "trading_date": trading_date.isoformat(),
                    "stage": "polygon_minute_fetch",
                    "error": "minute_fetch_error",
                })
                out[trading_date] = ()
                continue
            out[trading_date] = tuple(
                _clean_minute_bars(trading_date, getattr(resp, "data") or ())
            )
        return out, watchdog_timeouts

    def _maybe_reset_polygon_session(self) -> None:
        if (
            self._fetch_watchdog.total_timeouts == 0
            or self._fetch_watchdog.total_timeouts % 3
        ):
            return
        reset = getattr(self._polygon_minute_adapter, "reset_session", None)
        if callable(reset):
            reset()
            self._emit_progress("polygon_session_reset", self._fetch_watchdog.snapshot())

    def _fetch_reference_series(
        self,
        symbols: Sequence[str],
        *,
        from_date: date,
        through_date: date,
        run_ts: datetime,
        job_run_id: str,
        source_role: str,
        pending_lineages: list[DataLineage],
        non_session_tracker: _NonSessionBarSkipTracker | None = None,
    ) -> dict[str, _ReferenceSeries]:
        series: dict[str, _ReferenceSeries] = {}
        for symbol in sorted({item.upper() for item in symbols if item}):
            try:
                resp = self._fmp_historical_price_with_deadline(
                    symbol,
                    from_date=from_date,
                    to_date=through_date,
                    asof=run_ts,
                    adjusted=False,
                    stage=f"fmp_{source_role}",
                )
            except FuturesTimeoutError:
                series[symbol] = _ReferenceSeries(
                    symbol=symbol,
                    bars=(),
                    data_lineage_id=None,
                    raw_payload_hash=None,
                    status="fetch_error",
                    error={
                        "symbol": symbol,
                        "stage": f"fmp_{source_role}",
                        "message": "fetch watchdog timeout",
                        "error_type": "watchdog_timeout",
                        "provider": "FMP",
                        "retryable": True,
                        "deadline_seconds": self._fetch_deadline_seconds,
                        **self._fetch_watchdog.snapshot(),
                    },
                )
                continue
            if not resp.ok or resp.data is None:
                series[symbol] = _ReferenceSeries(
                    symbol=symbol,
                    bars=(),
                    data_lineage_id=None,
                    raw_payload_hash=getattr(resp.lineage, "raw_payload_hash", None),
                    status="fetch_error",
                    error={
                        "symbol": symbol,
                        "stage": f"fmp_{source_role}",
                        "message": sanitize_provider_error_message(
                            getattr(resp.error, "message", "missing response")
                        ),
                        "error_type": getattr(resp.error, "error_type", None),
                        "provider": getattr(resp.error, "provider", None),
                        "status_code": getattr(resp.error, "status_code", None),
                        "retryable": getattr(resp.error, "retryable", None),
                        **_retry_metadata_from_lineage(resp),
                    },
                )
                continue
            bars = tuple(_clean_bars(
                resp.data,
                ticker=symbol,
                non_session_tracker=non_session_tracker,
            ))
            lineage = _build_data_lineage(
                provider="FMP",
                endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                asof_timestamp=run_ts,
                raw_payload=_bar_lineage_payload(
                    symbol=symbol,
                    from_date=from_date,
                    through_date=through_date,
                    bars=bars,
                    feature_version=self._feature_version,
                    symbol_field="symbol",
                    source_role=source_role,
                ),
                source_authority="fmp_eod",
                data_quality_flags=_lineage_quality_flags(
                    resp,
                    derived_feature_replay=True,
                    reference_series_role=source_role,
                    lineage_payload_schema="compact_bar_digest_v1",
                    adapter_raw_payload_hash=resp.lineage.raw_payload_hash,
                ),
                job_run_id=job_run_id,
            )
            pending_lineages.append(lineage)
            series[symbol] = _ReferenceSeries(
                symbol=symbol,
                bars=bars,
                data_lineage_id=lineage.data_lineage_id,
                raw_payload_hash=resp.lineage.raw_payload_hash,
                status="available" if bars else "empty",
            )
        return series

    def _needed_sector_etfs(
        self,
        signals: Sequence[SignalRegistry],
        *,
        through_date: date,
        sector_resolver: "_SectorResolver",
    ) -> tuple[str, ...]:
        etfs: set[str] = set()
        for signal in signals:
            signal_date = signal.signal_timestamp.date()
            entry_date = _entry_date(signal)
            start_date = signal_date if self._include_signal_session else entry_date
            if start_date is None:
                continue
            end_date = through_date
            horizon_sessions = _signal_horizon_sessions(signal.signal_horizon)
            if entry_date is not None and horizon_sessions is not None:
                end_date = min(through_date, nth_us_equity_session(entry_date, horizon_sessions))
            for feature_date in _weekdays_between(start_date, end_date):
                resolution = sector_resolver.resolve(signal.ticker, feature_date)
                if resolution.sector_etf:
                    etfs.add(resolution.sector_etf)
        return tuple(sorted(etfs))

    def _persist_signal_rows(
        self,
        signal: SignalRegistry,
        *,
        bars: Sequence[_CleanBar],
        minute_bars_by_date: dict[date, tuple[_MinuteBar, ...]] | None = None,
        through_date: date,
        data_lineage_id: str,
        job_run_id: str,
        existing: dict[tuple[str, str], MarketPathFeature],
        benchmark_series: dict[str, _ReferenceSeries],
        sector_resolver: "_SectorResolver",
        sector_etf_series: dict[str, _ReferenceSeries],
        same_day_pattern_strengths: SameDayPatternStrengthCache,
        pending_feature_rows: list[dict[str, Any]],
        stage_timings: dict[str, float] | None = None,
    ) -> dict[str, int]:
        by_date = {bar.date: bar for bar in bars}
        entry_date = _entry_date(signal)
        signal_date = signal.signal_timestamp.date()
        start_date = signal_date if self._include_signal_session else entry_date
        if start_date is None:
            return {"inserted": 0, "updated": 0, "unchanged": 0, "skipped": 0, "missing_entry": 1}
        if (
            (entry_date is None or entry_date not in by_date)
            and not self._include_signal_session
        ):
            return {"inserted": 0, "updated": 0, "unchanged": 0, "skipped": 0, "missing_entry": 1}

        entry_price = by_date[entry_date].open if entry_date in by_date else None
        end_date = through_date
        horizon_sessions = _signal_horizon_sessions(signal.signal_horizon)
        if entry_date is not None and horizon_sessions is not None:
            end_date = min(through_date, nth_us_equity_session(entry_date, horizon_sessions))
        path_bars = [
            bar for bar in bars
            if start_date <= bar.date <= end_date
        ]
        path_sequence = 0
        inserted = updated = unchanged = skipped = 0
        for bar in path_bars:
            if (
                entry_date is not None
                and bar.date < entry_date
                and not self._include_signal_session
            ):
                continue
            if entry_date is not None and bar.date >= entry_date:
                path_sequence += 1
            if entry_date is not None and bar.date > signal_date and bar.date < entry_date:
                skipped += 1
                continue
            role = "signal_session" if bar.date == signal_date else "forward_path_day"
            started = perf_counter()
            payload = self._feature_payload(
                signal,
                bar=bar,
                bars=bars,
                minute_bars=(
                    (minute_bars_by_date or {}).get(bar.date, ())
                    if role == "signal_session"
                    else ()
                ),
                entry_date=entry_date,
                entry_price=entry_price,
                path_sequence=max(path_sequence, 0),
                feature_role=role,
                batch_through_date=through_date,
                benchmark_series=benchmark_series,
                sector_resolver=sector_resolver,
                sector_etf_series=sector_etf_series,
                same_day_pattern_strengths=same_day_pattern_strengths,
            )
            if stage_timings is not None:
                _record_timing(stage_timings, "feature_compute_seconds", started)
            started = perf_counter()
            key = (signal.signal_id, bar.date.isoformat())
            row = existing.get(key)
            if row is not None and self._skip_existing:
                unchanged += 1
                continue
            if row is None:
                inserted += 1
            else:
                updated += 1
            row_mapping = _market_path_feature_row_mapping(
                payload,
                existing_row=row,
                data_lineage_id=data_lineage_id,
                job_run_id=job_run_id,
            )
            if row is not None and not _feature_row_materially_changed(row, row_mapping):
                unchanged += 1
            else:
                pending_feature_rows.append(row_mapping)
            if stage_timings is not None:
                _record_timing(stage_timings, "row_upsert_assign_seconds", started)
        return {
            "inserted": inserted,
            "updated": updated,
            "unchanged": unchanged,
            "skipped": skipped,
            "missing_entry": 0,
        }

    def _feature_payload(
        self,
        signal: SignalRegistry,
        *,
        bar: _CleanBar,
        bars: Sequence[_CleanBar],
        minute_bars: Sequence[_MinuteBar] = (),
        entry_date: date | None,
        entry_price: float | None,
        path_sequence: int,
        feature_role: str,
        batch_through_date: date,
        benchmark_series: dict[str, _ReferenceSeries],
        sector_resolver: "_SectorResolver",
        sector_etf_series: dict[str, _ReferenceSeries],
        same_day_pattern_strengths: SameDayPatternStrengthCache,
    ) -> dict[str, Any]:
        signal_date = signal.signal_timestamp.date()
        is_signal_session = feature_role == "signal_session" and bar.date == signal_date
        previous = _previous_bar(bars, bar.date)
        prior = [candidate for candidate in bars if candidate.date < bar.date]
        row_input_bars = [candidate for candidate in bars if candidate.date <= bar.date]
        predictor_input_bars = list(prior) if is_signal_session else row_input_bars
        prior20 = prior[-20:]
        prior60 = prior[-60:]
        close_basis = _price_basis(bar)
        prev_close = _price_basis(previous) if previous is not None else None
        volume_basis_bar = previous if is_signal_session and previous is not None else bar
        liquidity_price_basis = (
            _price_basis(volume_basis_bar) if volume_basis_bar is not None else close_basis
        )
        median_volume_20d = _median([b.volume for b in prior20])
        median_volume_60d = _median([b.volume for b in prior60])
        median_dollar_volume_20d = _median([b.dollar_volume for b in prior20])
        median_dollar_volume_60d = _median([b.dollar_volume for b in prior60])
        sigma_20d = _sigma_close_to_close(prior20)
        effective_stop = (
            min(0.10, max(0.04, 1.5 * sigma_20d))
            if sigma_20d is not None else None
        )
        entry_basis = (
            entry_price
            if entry_date is not None and bar.date >= entry_date
            else None
        )
        liquidity_passed = (
            median_dollar_volume_20d is not None
            and median_dollar_volume_20d >= self._liquidity_min_dollar_volume_20d
            and liquidity_price_basis >= self._liquidity_min_price
        )
        rich_features, rich_status = _rich_eod_features(
            bar=bar,
            previous=previous,
            prior=prior,
            asof_prior_close=is_signal_session,
        )
        relative_features, relative_status, relative_input_payload = (
            _relative_features(
                ticker=signal.ticker.upper(),
                feature_date=bar.date,
                ticker_momentum={
                    "5d": rich_features["momentum_5d"],
                    "20d": rich_features["momentum_20d"],
                    "60d": rich_features["momentum_60d"],
                },
                benchmark_series=benchmark_series,
                sector_resolution=sector_resolver.resolve(signal.ticker, bar.date),
                sector_etf_series=sector_etf_series,
            )
        )
        advanced_features, advanced_status = _advanced_context_features(
            signal=signal,
            feature_date=bar.date,
            bar=bar,
            minute_bars=minute_bars,
            minute_layer_enabled=(
                is_signal_session and self._polygon_minute_adapter is not None
            ),
            prior=prior,
            benchmark_series=benchmark_series,
            entry_price=entry_price,
            same_day_pattern_strengths=same_day_pattern_strengths,
        )
        top_level_predictor_features = {
            "sigma_20d": sigma_20d,
            "dollar_volume": volume_basis_bar.dollar_volume,
            "volume_expansion_20d": _safe_ratio(
                volume_basis_bar.volume,
                median_volume_20d,
            ),
            "volume_expansion_60d": _safe_ratio(
                volume_basis_bar.volume,
                median_volume_60d,
            ),
            "dollar_volume_expansion_20d": _safe_ratio(
                volume_basis_bar.dollar_volume,
                median_dollar_volume_20d,
            ),
            "dollar_volume_expansion_60d": _safe_ratio(
                volume_basis_bar.dollar_volume,
                median_dollar_volume_60d,
            ),
            "liquidity_proxy_score": 1.0 if liquidity_passed else 0.0,
        }
        relative_input_hash = stable_hash(relative_input_payload)
        row_input_payload = {
            "signal_id": signal.signal_id,
            "feature_session_date": bar.date.isoformat(),
            "feature_version": self._feature_version,
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "entry_session_date": entry_date.isoformat() if entry_date is not None else None,
            "source_provider": "FMP",
            "source_endpoint": HISTORICAL_PRICE_FULL_ENDPOINT,
            "bars_through_feature_session": [
                _bar_payload(candidate) for candidate in row_input_bars
            ],
            "predictor_bars_through": (
                predictor_input_bars[-1].date.isoformat()
                if predictor_input_bars else None
            ),
            "relative_input_hash": relative_input_hash,
            "relative_inputs": relative_input_payload,
        }
        row_input_hash = stable_hash(row_input_payload)
        feature_json = {
            "schema_version": self._feature_version,
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "feature_role": feature_role,
            "time_barrier_sessions": _signal_horizon_sessions(signal.signal_horizon),
            "lineage_scope": "batch_fetch",
            "row_input_window_start": (
                row_input_bars[0].date.isoformat() if row_input_bars else None
            ),
            "row_input_window_end": bar.date.isoformat(),
            "predictor_input_window_end": (
                predictor_input_bars[-1].date.isoformat()
                if predictor_input_bars else None
            ),
            "row_input_hash": row_input_hash,
            "row_input_hash_schema": "bars_through_feature_session_v1",
            "batch_lineage_contains_future_rows_for_earlier_feature_dates": (
                batch_through_date > bar.date
            ),
            "batch_lineage_window_end": batch_through_date.isoformat(),
            "sigma_basis": PRICE_BASIS_SPLIT_ADJUSTED_OR_RAW,
            "stop_basis": PRICE_BASIS_SPLIT_ADJUSTED_OR_RAW,
            "prior_window_counts": {
                "volume_20d": len(prior20),
                "volume_60d": len(prior60),
            },
            "liquidity_proxy": {
                "score": 1.0 if liquidity_passed else 0.0,
                "passed": liquidity_passed,
                "min_dollar_volume_20d": self._liquidity_min_dollar_volume_20d,
                "min_price": self._liquidity_min_price,
                "not_canonical_candidate_liquidity_score": True,
            },
            "intraday_status": {
                "opening_range_available": False,
                "intraday_continuation_available": False,
                "quote_spread_available": False,
                "reason": "not captured in daily replay",
            },
            "rich_eod_features": rich_features,
            "rich_eod_status": rich_status,
            "cross_sectional_features": {},
            "market_relative_features": {
                key: value for key, value in relative_features.items()
                if key.startswith(("spy_", "qqq_", "iwm_", "relative_strength_vs_spy", "relative_strength_vs_qqq", "relative_strength_vs_iwm"))
            },
            "sector_relative_features": {
                key: value for key, value in relative_features.items()
                if key.startswith(("sector_", "relative_strength_vs_sector"))
            },
            "market_regime_features": advanced_status["market_regime"]["features"],
            "market_regime_status": advanced_status["market_regime"]["status"],
            "intraday_structure_features": advanced_status["intraday_structure"]["features"],
            "intraday_structure_status": advanced_status["intraday_structure"]["status"],
            "execution_quality_features": advanced_status["execution_quality"]["features"],
            "execution_quality_status": advanced_status["execution_quality"]["status"],
            "supply_squeeze_features": advanced_status["supply_squeeze"]["features"],
            "supply_squeeze_status": advanced_status["supply_squeeze"]["status"],
            "catalyst_context_features": advanced_status["catalyst_context"]["features"],
            "catalyst_context_status": advanced_status["catalyst_context"]["status"],
            "classic_technical_features": advanced_status["technical_indicators"]["features"],
            "classic_technical_status": advanced_status["technical_indicators"]["status"],
            "relative_feature_status": relative_status,
            "relative_input_hash": relative_input_hash,
        }
        if is_signal_session:
            feature_json["leakage_contract"] = _signal_session_leakage_contract(
                predictor_features={
                    **top_level_predictor_features,
                    **{
                        key: rich_features.get(key)
                        for key in SIGNAL_SESSION_PREDICTOR_FIELDS
                        if key in rich_features
                    },
                    **{
                        key: relative_features.get(key)
                        for key in SIGNAL_SESSION_PREDICTOR_FIELDS
                        if key in relative_features
                    },
                    **{
                        key: advanced_features.get(key)
                        for key in SIGNAL_SESSION_PREDICTOR_FIELDS
                        if key in advanced_features
                    },
                },
                outcome_features={
                    "previous_close": prev_close,
                    "open_price": bar.open,
                    "high_price": bar.high,
                    "low_price": bar.low,
                    "close_price": bar.close,
                    "volume": bar.volume,
                    "open_to_close_return": _safe_return(bar.close, bar.open),
                    "gap_pct": _safe_return(bar.open, prev_close),
                    "breakout_extension_pct": rich_features["breakout_extension_pct"],
                    "open_vs_52w_high_pct": rich_features["open_vs_52w_high_pct"],
                    "close_vs_52w_high_pct": rich_features["close_vs_52w_high_pct"],
                    "high_vs_52w_high_pct": rich_features["high_vs_52w_high_pct"],
                    "closed_above_breakout": rich_features["closed_above_breakout"],
                    "gap_over_breakout": rich_features["gap_over_breakout"],
                    "high_from_open_return": _safe_return(bar.high, bar.open),
                    "low_from_open_return": _safe_return(bar.low, bar.open),
                },
                predictor_input_end=(
                    predictor_input_bars[-1].date if predictor_input_bars else None
                ),
            )
        feature_json_text = json.dumps(feature_json, sort_keys=True)
        input_payload = {
            "signal_id": signal.signal_id,
            "feature_session_date": bar.date.isoformat(),
            "feature_version": self._feature_version,
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "entry_session_date": entry_date.isoformat() if entry_date is not None else None,
            "source_provider": "FMP",
            "source_endpoint": HISTORICAL_PRICE_FULL_ENDPOINT,
            "row_input_hash": row_input_hash,
        }
        payload = {
            "signal_id": signal.signal_id,
            "pattern_id": signal.pattern_id,
            "ticker": signal.ticker.upper(),
            "signal_horizon": signal.signal_horizon,
            "signal_date": signal_date.isoformat(),
            "entry_session_date": entry_date.isoformat() if entry_date is not None else None,
            "feature_session_date": bar.date.isoformat(),
            "path_sequence": path_sequence,
            "feature_role": feature_role,
            "feature_version": self._feature_version,
            "asof_timestamp": us_equity_session_close_timestamp(bar.date),
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "previous_close": prev_close,
            "open_price": bar.open,
            "high_price": bar.high,
            "low_price": bar.low,
            "close_price": bar.close,
            "volume": bar.volume,
            "split_adjusted_close": bar.split_adjusted_close,
            "adj_close": bar.adj_close,
            "dollar_volume": volume_basis_bar.dollar_volume,
            "median_volume_20d": median_volume_20d,
            "median_volume_60d": median_volume_60d,
            "median_dollar_volume_20d": median_dollar_volume_20d,
            "median_dollar_volume_60d": median_dollar_volume_60d,
            "volume_expansion_20d": _safe_ratio(volume_basis_bar.volume, median_volume_20d),
            "volume_expansion_60d": _safe_ratio(volume_basis_bar.volume, median_volume_60d),
            "dollar_volume_expansion_20d": _safe_ratio(volume_basis_bar.dollar_volume, median_dollar_volume_20d),
            "dollar_volume_expansion_60d": _safe_ratio(volume_basis_bar.dollar_volume, median_dollar_volume_60d),
            "gap_pct": _safe_return(bar.open, prev_close),
            "open_to_close_return": _safe_return(bar.close, bar.open),
            "high_from_open_return": _safe_return(bar.high, bar.open),
            "low_from_open_return": _safe_return(bar.low, bar.open),
            "return_from_entry_open": _safe_return(bar.open, entry_basis),
            "return_from_entry_high": _safe_return(bar.high, entry_basis),
            "return_from_entry_low": _safe_return(bar.low, entry_basis),
            "return_from_entry_close": _safe_return(bar.close, entry_basis),
            "sigma_20d": sigma_20d,
            "effective_hard_stop_pct": effective_stop,
            "liquidity_proxy_score": 1.0 if liquidity_passed else 0.0,
            "liquidity_proxy_passed": liquidity_passed,
            **{
                key: value for key, value in rich_features.items()
                if key != "prior_close_vs_52w_high_pct"
            },
            **relative_features,
            **advanced_features,
            **_empty_cross_sectional_features(),
            "opening_range_json": None,
            "intraday_continuation_json": None,
            "quote_spread_json": None,
            "feature_json": feature_json_text,
            "source_provider": "FMP",
            "source_endpoint": HISTORICAL_PRICE_FULL_ENDPOINT,
            "input_hash": stable_hash(input_payload),
        }
        output_payload = {key: payload[key] for key in ML_OUTPUT_HASH_FIELDS}
        payload["output_hash"] = stable_hash(output_payload)
        return payload

    def _populate_cross_sectional_ranks(
        self,
        *,
        start_date: date,
        through_date: date,
        progress_callback: ProgressCallback | None = None,
        progress_every: int | None = None,
    ) -> int:
        rank_started = perf_counter()
        rows = (
            self._session.query(MarketPathFeature)
            .filter(
                MarketPathFeature.pattern_id.in_(self._pattern_ids),
                MarketPathFeature.feature_version == self._feature_version,
                MarketPathFeature.feature_session_date >= start_date.isoformat(),
                MarketPathFeature.feature_session_date <= through_date.isoformat(),
            )
            .order_by(
                MarketPathFeature.feature_session_date,
                MarketPathFeature.pattern_id,
                MarketPathFeature.ticker,
                MarketPathFeature.signal_id,
            )
            .all()
        )
        grouped: dict[tuple[str, str, str], list[MarketPathFeature]] = {}
        for row in rows:
            grouped.setdefault(
                (row.feature_session_date, row.pattern_id, row.feature_version),
                [],
            ).append(row)

        updated = 0
        rank_group_total = len(grouped)
        progress_step = progress_every or self._progress_every
        for group_index, ((feature_date, pattern_id, feature_version), group_rows) in enumerate(grouped.items(), start=1):
            pattern_count = len(group_rows)
            tracked_fields = _cross_sectional_tracked_fields()
            previous_values = {
                row.market_path_feature_id: {
                    **{field: getattr(row, field) for field in tracked_fields},
                    "feature_json": row.feature_json,
                    "output_hash": row.output_hash,
                }
                for row in group_rows
            }
            desired_values = {
                row.market_path_feature_id: {
                    field: None
                    for field in tracked_fields
                }
                for row in group_rows
            }
            rankable_any = {
                row.signal_id
                for row in group_rows
                if any(_rank_input_value(row, source) is not None for source in RANK_INPUT_TO_OUTPUT)
            }
            feature_count = len(rankable_any)
            for row in group_rows:
                desired_values[row.market_path_feature_id]["cohort_pattern_row_count"] = pattern_count
                desired_values[row.market_path_feature_id]["cohort_feature_row_count"] = feature_count

            rank_status: dict[str, dict[str, Any]] = {}
            for source_field, (rank_field, percentile_field) in RANK_INPUT_TO_OUTPUT.items():
                ranked = [
                    row for row in group_rows
                    if _rank_input_value(row, source_field) is not None
                ]
                ranked.sort(
                    key=lambda row: (
                        -float(_rank_input_value(row, source_field) or 0.0),
                        row.ticker,
                        row.signal_id,
                    )
                )
                value_count = len(ranked)
                for rank_index, row in enumerate(ranked, start=1):
                    desired = desired_values[row.market_path_feature_id]
                    desired[rank_field] = rank_index
                    desired[percentile_field] = (
                        ((value_count - rank_index + 1) / value_count)
                        if value_count else None
                    )
                rank_status[source_field] = {
                    "rank_field": rank_field,
                    "percentile_field": percentile_field,
                    "rank_direction": "higher_is_better",
                    "population_count": value_count,
                    "population_too_small": value_count < 2,
                }

            for row in group_rows:
                desired = desired_values[row.market_path_feature_id]
                payload = _json_dict(row.feature_json)
                payload["cross_sectional_features"] = {
                    "rank_scope": {
                        "feature_session_date": feature_date,
                        "pattern_id": pattern_id,
                        "feature_version": feature_version,
                        "cohort_pattern_row_count": pattern_count,
                        "cohort_feature_row_count": feature_count,
                        "rank_direction": "higher_is_better",
                    },
                    "dollar_volume_rank": desired["dollar_volume_rank"],
                    "dollar_volume_percentile": desired["dollar_volume_percentile"],
                    "volume_expansion_20d_rank": desired["volume_expansion_20d_rank"],
                    "volume_expansion_20d_percentile": desired["volume_expansion_20d_percentile"],
                    "volume_expansion_60d_rank": desired["volume_expansion_60d_rank"],
                    "volume_expansion_60d_percentile": desired["volume_expansion_60d_percentile"],
                    "dollar_volume_expansion_20d_rank": desired["dollar_volume_expansion_20d_rank"],
                    "dollar_volume_expansion_20d_percentile": desired["dollar_volume_expansion_20d_percentile"],
                    "dollar_volume_expansion_60d_rank": desired["dollar_volume_expansion_60d_rank"],
                    "dollar_volume_expansion_60d_percentile": desired["dollar_volume_expansion_60d_percentile"],
                    "liquidity_proxy_rank": desired["liquidity_proxy_rank"],
                    "liquidity_proxy_percentile": desired["liquidity_proxy_percentile"],
                }
                relative_status = payload.setdefault("relative_feature_status", {})
                row_status = {
                    source: {
                        **status,
                        "value_missing": _rank_input_value(row, source) is None,
                    }
                    for source, status in rank_status.items()
                }
                relative_status["cross_sectional_rank"] = row_status
                payload["feature_json_hash_includes_rank_pass"] = True
                new_feature_json = json.dumps(payload, sort_keys=True)
                new_output_hash = _output_hash_for_row(
                    row,
                    **desired,
                    feature_json=new_feature_json,
                )
                row_previous_values = previous_values.get(row.market_path_feature_id, {})
                rank_values_changed = any(
                    not _values_equal(desired[field], row_previous_values.get(field))
                    for field in tracked_fields
                )
                json_changed = row_previous_values.get("feature_json") != new_feature_json
                hash_changed = row_previous_values.get("output_hash") != new_output_hash
                if rank_values_changed:
                    for field, value in desired.items():
                        if not _values_equal(getattr(row, field), value):
                            setattr(row, field, value)
                if json_changed:
                    row.feature_json = new_feature_json
                if hash_changed:
                    row.output_hash = new_output_hash
                if rank_values_changed or json_changed or hash_changed:
                    updated += 1
            if (
                group_index == 1
                or group_index == rank_group_total
                or group_index % progress_step == 0
            ):
                _safe_rank_progress(
                    progress_callback,
                    {
                        "rank_group_processed": group_index,
                        "rank_group_total": rank_group_total,
                        "feature_session_date": feature_date,
                        "pattern_id": pattern_id,
                        "feature_version": feature_version,
                        "elapsed_seconds": _elapsed_since(rank_started),
                    },
                )
        if rank_group_total == 0:
            _safe_rank_progress(
                progress_callback,
                {
                    "rank_group_processed": 0,
                    "rank_group_total": 0,
                    "feature_session_date": None,
                    "pattern_id": None,
                    "feature_version": self._feature_version,
                    "elapsed_seconds": _elapsed_since(rank_started),
                },
            )
        return updated

    def populate_ranks_only(
        self,
        *,
        start_date: date,
        through_date: date,
        progress_callback: ProgressCallback | None = None,
        progress_every: int | None = None,
    ) -> dict[str, Any]:
        """Populate cross-sectional ranks month-by-month without fetching bars."""

        if start_date > through_date:
            raise ValueError("start_date must be on or before through_date")
        started = perf_counter()
        total_updated = 0
        month_records: list[dict[str, Any]] = []
        for month_start, month_end in _month_ranges(start_date, through_date):
            month_started = perf_counter()
            _safe_progress(
                progress_callback,
                "rank_month_start",
                {
                    "month_start": month_start.isoformat(),
                    "month_end": month_end.isoformat(),
                    "feature_version": self._feature_version,
                },
            )
            rows_updated = self._populate_cross_sectional_ranks(
                start_date=month_start,
                through_date=month_end,
                progress_callback=progress_callback,
                progress_every=progress_every or self._progress_every,
            )
            if rows_updated:
                self._session.flush()
            self._session.commit()
            total_updated += rows_updated
            record = {
                "month_start": month_start.isoformat(),
                "month_end": month_end.isoformat(),
                "rank_rows_updated": rows_updated,
                "elapsed_seconds": _elapsed_since(month_started),
            }
            month_records.append(record)
            _safe_progress(progress_callback, "rank_month_finish", record)
        return {
            "rank_rows_updated": total_updated,
            "rank_month_count": len(month_records),
            "rank_months": month_records,
            "elapsed_seconds": _elapsed_since(started),
        }


def _assign_row(
    row: MarketPathFeature,
    payload: dict[str, Any],
    *,
    data_lineage_id: str,
    job_run_id: str,
    preserve_rank_state: bool = False,
) -> None:
    cross_sectional_fields = set(_cross_sectional_tracked_fields())
    for key, value in payload.items():
        if preserve_rank_state and (key in cross_sectional_fields or key == "output_hash"):
            continue
        if preserve_rank_state and key == "feature_json":
            value = _merge_cross_sectional_feature_json(row.feature_json, value)
        setattr(row, key, value)
    row.data_lineage_id = data_lineage_id
    row.job_run_id = job_run_id


def _market_path_feature_row_mapping(
    payload: dict[str, Any],
    *,
    existing_row: MarketPathFeature | None,
    data_lineage_id: str,
    job_run_id: str,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    row = dict(payload)
    row["data_lineage_id"] = data_lineage_id
    row["job_run_id"] = job_run_id
    row["updated_at"] = now
    if existing_row is None:
        row["market_path_feature_id"] = str(uuid4())
        row["created_at"] = now
        return row

    row["market_path_feature_id"] = existing_row.market_path_feature_id
    row["created_at"] = existing_row.created_at
    for field in _cross_sectional_tracked_fields():
        row[field] = getattr(existing_row, field)
    row["feature_json"] = _merge_cross_sectional_feature_json(
        existing_row.feature_json,
        row.get("feature_json"),
    )
    row["output_hash"] = existing_row.output_hash
    return row


def _feature_row_materially_changed(
    existing_row: MarketPathFeature,
    row: dict[str, Any],
) -> bool:
    ignored = {
        "market_path_feature_id",
        "created_at",
        "updated_at",
        "data_lineage_id",
        "job_run_id",
        "output_hash",
        *_cross_sectional_tracked_fields(),
    }
    for key, value in row.items():
        if key in ignored:
            continue
        if not _values_equal(getattr(existing_row, key), value):
            return True
    return False


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, datetime) and isinstance(right, datetime):
        return _datetime_compare_value(left) == _datetime_compare_value(right)
    if (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and not isinstance(left, bool)
        and not isinstance(right, bool)
    ):
        left_float = float(left)
        right_float = float(right)
        if math.isfinite(left_float) and math.isfinite(right_float):
            return math.isclose(
                left_float,
                right_float,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
    return left == right


def _datetime_compare_value(value: datetime) -> datetime:
    if value.tzinfo is not None and value.utcoffset() is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _bulk_upsert_market_path_features(
    session: Session,
    rows: Sequence[dict[str, Any]],
) -> None:
    if not rows:
        return
    table = MarketPathFeature.__table__
    dialect_name = session.get_bind().dialect.name
    insert_factory = sqlite_insert if dialect_name == "sqlite" else pg_insert
    batch_size = 1 if dialect_name == "sqlite" else 100
    for batch in _batched(rows, batch_size):
        stmt = insert_factory(table).values(list(batch))
        excluded = stmt.excluded
        update_columns = {
            column.name: getattr(excluded, column.name)
            for column in table.columns
            if column.name not in {"market_path_feature_id", "created_at"}
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["signal_id", "feature_session_date", "feature_version"],
            set_=update_columns,
        )
        session.execute(stmt)


def _batched(
    rows: Sequence[dict[str, Any]],
    batch_size: int,
) -> Iterable[Sequence[dict[str, Any]]]:
    for index in range(0, len(rows), batch_size):
        yield rows[index:index + batch_size]


def _build_data_lineage(
    *,
    provider: str,
    endpoint: str,
    asof_timestamp: datetime,
    raw_payload: Any | None = None,
    raw_payload_hash: str | None = None,
    request_timestamp: datetime | None = None,
    freshness_seconds: float | None = None,
    source_authority: str | None = None,
    data_quality_flags: dict | None = None,
    job_run_id: str | None = None,
    dataset_id: str | None = None,
) -> DataLineage:
    if raw_payload_hash is None:
        raw_payload_hash = stable_hash(raw_payload)
    raw_payload_json = (
        json.dumps(raw_payload, sort_keys=True, default=str)
        if raw_payload is not None else None
    )
    return DataLineage(
        data_lineage_id=str(uuid4()),
        provider=provider,
        endpoint=endpoint,
        request_timestamp=request_timestamp or datetime.now(timezone.utc),
        asof_timestamp=asof_timestamp,
        raw_payload_hash=raw_payload_hash,
        raw_payload_json=raw_payload_json,
        freshness_seconds=freshness_seconds,
        source_authority=source_authority,
        data_quality_flags=(
            json.dumps(data_quality_flags, sort_keys=True, default=str)
            if data_quality_flags is not None else None
        ),
        job_run_id=job_run_id,
        dataset_id=dataset_id,
    )


def _record_timing(timings: dict[str, float], key: str, started: float) -> None:
    timings[key] = round(timings.get(key, 0.0) + (perf_counter() - started), 6)


def _elapsed_since(started: float) -> float:
    return round(perf_counter() - started, 6)


def _utc_progress_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _safe_rank_progress(
    callback: ProgressCallback | None,
    payload: dict[str, Any],
) -> None:
    if callback is None:
        return
    try:
        callback("rank_group_progress", payload)
    except Exception:
        return


def _safe_progress(
    callback: ProgressCallback | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if callback is None:
        return
    try:
        callback(event, payload)
    except Exception:
        return


def _month_ranges(start_date: date, through_date: date) -> Iterable[tuple[date, date]]:
    cursor = start_date.replace(day=1)
    while cursor <= through_date:
        next_month = _first_day_next_month(cursor)
        month_start = max(start_date, cursor)
        month_end = min(through_date, next_month - timedelta(days=1))
        if month_start <= month_end:
            yield month_start, month_end
        cursor = next_month


def _first_day_next_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _retry_metadata_from_lineage(response: Any) -> dict[str, Any]:
    lineage = getattr(response, "lineage", None)
    flags = getattr(lineage, "data_quality_flags", None)
    if not isinstance(flags, dict):
        return {}
    metadata: dict[str, Any] = {}
    field_map = {
        "market_path_bulk_retry_attempt_count": "retry_attempt_count",
        "market_path_bulk_retry_max_retries": "retry_max_retries",
        "market_path_bulk_retry_exhausted": "retry_exhausted",
        "market_path_bulk_request_timeout_seconds": "request_timeout_seconds",
    }
    for source, target in field_map.items():
        if source in flags:
            metadata[target] = flags[source]
    attempts = flags.get("market_path_bulk_retry_attempts")
    if isinstance(attempts, list):
        metadata["retry_attempts"] = [
            {
                key: _sanitize_retry_value(attempt.get(key))
                for key in (
                    "attempt",
                    "ok",
                    "elapsed_seconds",
                    "provider",
                    "endpoint",
                    "status_code",
                    "error_type",
                    "message",
                    "retryable",
                )
                if isinstance(attempt, dict) and key in attempt
            }
            for attempt in attempts
            if isinstance(attempt, dict)
        ]
    return metadata


def sanitize_provider_error_message(value: Any) -> Any:
    """Return a diagnostics-safe provider error string without credentials."""

    if value is None:
        return value
    text_value = str(value)

    def strip_url_query(match: re.Match[str]) -> str:
        url = match.group(0)
        return url.split("?", 1)[0]

    sanitized = _URL_RE.sub(strip_url_query, text_value)
    sanitized = _RELATIVE_URL_QUERY_RE.sub(lambda match: match.group("path"), sanitized)
    sanitized = _SECRET_QUERY_RE.sub(r"\1credential=<redacted>", sanitized)
    sanitized = _BEARER_TOKEN_RE.sub("credential=<redacted>", sanitized)
    sanitized = _SECRET_FIELD_RE.sub("credential=<redacted>", sanitized)
    return sanitized


def _sanitize_retry_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_provider_error_message(value)
    return value


def _merge_cross_sectional_feature_json(
    existing_json: str | None,
    new_json: str | None,
) -> str | None:
    if not existing_json or not new_json:
        return new_json
    existing = _json_dict(existing_json)
    new_payload = _json_dict(new_json)
    if not new_payload:
        return new_json
    if "cross_sectional_features" in existing:
        new_payload["cross_sectional_features"] = existing["cross_sectional_features"]
    if "feature_json_hash_includes_rank_pass" in existing:
        new_payload["feature_json_hash_includes_rank_pass"] = existing[
            "feature_json_hash_includes_rank_pass"
        ]
    existing_relative_status = existing.get("relative_feature_status")
    if isinstance(existing_relative_status, dict) and "cross_sectional_rank" in existing_relative_status:
        new_relative_status = new_payload.setdefault("relative_feature_status", {})
        if isinstance(new_relative_status, dict):
            new_relative_status["cross_sectional_rank"] = existing_relative_status[
                "cross_sectional_rank"
            ]
            for status_key in ("benchmark", "sector"):
                existing_status = existing_relative_status.get(status_key)
                new_status = new_relative_status.get(status_key)
                if _without_lineage_ids(existing_status) == _without_lineage_ids(new_status):
                    new_relative_status[status_key] = existing_status
    return json.dumps(new_payload, sort_keys=True)


def _without_lineage_ids(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_lineage_ids(item)
            for key, item in value.items()
            if key != "lineage_id"
        }
    if isinstance(value, list):
        return [_without_lineage_ids(item) for item in value]
    return value


class _SectorResolver:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._has_history_table = inspect(session.get_bind()).has_table(
            "firm_sector_assignments_history"
        )
        self._cache: dict[tuple[str, date], _SectorResolution] = {}

    def resolve(self, ticker: str, feature_date: date) -> _SectorResolution:
        key = (ticker.upper(), feature_date)
        if key in self._cache:
            return self._cache[key]
        if not self._has_history_table:
            resolution = _SectorResolution(
                sector=None,
                sector_etf=None,
                source=None,
                status="m3_sector_history_table_missing",
                pit_safe=False,
            )
            self._cache[key] = resolution
            return resolution
        try:
            row = (
                self._session.execute(
                    text(
                        "SELECT sector, source FROM firm_sector_assignments_history "
                        "WHERE ticker = :ticker AND valid_from <= :feature_date "
                        "AND valid_to > :feature_date "
                        "ORDER BY valid_from DESC LIMIT 1"
                    ),
                    {
                        "ticker": ticker.upper(),
                        "feature_date": feature_date,
                    },
                )
                .mappings()
                .first()
            )
        except SQLAlchemyError:
            resolution = _SectorResolution(
                sector=None,
                sector_etf=None,
                source=None,
                status="m3_sector_history_lookup_error",
                pit_safe=False,
            )
            self._cache[key] = resolution
            return resolution
        if row is None:
            resolution = _SectorResolution(
                sector=None,
                sector_etf=None,
                source=None,
                status="sector_missing",
                pit_safe=False,
            )
        else:
            sector = str(row["sector"])
            source = str(row["source"] or "UNKNOWN")
            sector_etf = SECTOR_ETF_BY_SECTOR.get(sector)
            resolution = _SectorResolution(
                sector=sector,
                sector_etf=sector_etf,
                source=f"M3_PIT:{source}",
                status="available" if sector_etf else "sector_etf_unmapped",
                pit_safe=True,
            )
        self._cache[key] = resolution
        return resolution


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("run_timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _group_by_ticker(signals: Iterable[SignalRegistry]) -> dict[str, list[SignalRegistry]]:
    grouped: dict[str, list[SignalRegistry]] = {}
    for signal in signals:
        grouped.setdefault(signal.ticker.upper(), []).append(signal)
    return grouped


def _entry_date(signal: SignalRegistry) -> date | None:
    if not signal.next_execution_session:
        return None
    return date.fromisoformat(signal.next_execution_session)


def _signal_horizon_sessions(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text.endswith("d"):
        return None
    try:
        sessions = int(text[:-1])
    except ValueError:
        return None
    if sessions < 1:
        return None
    return sessions


def _fetch_start(signals: Sequence[SignalRegistry], lookback_calendar_days: int) -> date:
    starts: list[date] = []
    for signal in signals:
        signal_date = signal.signal_timestamp.date()
        entry_date = _entry_date(signal) or next_us_equity_session(signal_date + timedelta(days=1))
        starts.append(min(signal_date, entry_date))
    return min(starts) - timedelta(days=lookback_calendar_days)


def _clean_bars(
    bars: Sequence[FmpBar],
    *,
    ticker: str | None = None,
    non_session_tracker: _NonSessionBarSkipTracker | None = None,
) -> list[_CleanBar]:
    clean: list[_CleanBar] = []
    for bar in bars:
        try:
            parsed_date = date.fromisoformat(str(bar.date)[:10])
        except ValueError:
            continue
        if non_session_tracker is not None and not is_us_equity_session(parsed_date):
            non_session_tracker.record(ticker=ticker or "UNKNOWN", bar_date=parsed_date)
            continue
        values = (bar.open, bar.high, bar.low, bar.close)
        if any(value is None or not math.isfinite(float(value)) or float(value) <= 0 for value in values):
            continue
        clean.append(
            _CleanBar(
                date=parsed_date,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume or 0),
                split_adjusted_close=(
                    float(bar.split_adjusted_close)
                    if bar.split_adjusted_close is not None else None
                ),
                adj_close=float(bar.adj_close) if bar.adj_close is not None else None,
                vwap=(
                    float(bar.vwap)
                    if getattr(bar, "vwap", None) is not None
                    and math.isfinite(float(bar.vwap))
                    and float(bar.vwap) > 0
                    else None
                ),
            )
        )
    return sorted(clean, key=lambda item: item.date)


def _clean_minute_bars(
    trading_date: date,
    bars: Sequence[Any],
) -> list[_MinuteBar]:
    market_open = us_equity_session_open_timestamp(trading_date).astimezone(EASTERN_TZ)
    market_close = us_equity_session_close_timestamp(trading_date).astimezone(EASTERN_TZ)
    clean: list[_MinuteBar] = []
    for bar in bars:
        raw_ts = getattr(bar, "timestamp", None)
        if raw_ts is None:
            continue
        ts = (
            datetime.fromtimestamp(float(raw_ts) / 1000.0, timezone.utc)
            if isinstance(raw_ts, (int, float))
            else _ensure_aware(raw_ts)
        )
        ts_et = ts.astimezone(EASTERN_TZ)
        if ts_et.date() != trading_date or ts_et < market_open or ts_et > market_close:
            continue
        values = (
            getattr(bar, "open", None),
            getattr(bar, "high", None),
            getattr(bar, "low", None),
            getattr(bar, "close", None),
        )
        if any(
            value is None or not math.isfinite(float(value)) or float(value) <= 0
            for value in values
        ):
            continue
        volume = getattr(bar, "volume", 0) or 0
        if not math.isfinite(float(volume)) or float(volume) < 0:
            continue
        minute_index = int((ts_et - market_open).total_seconds() // 60)
        clean.append(
            _MinuteBar(
                timestamp=ts,
                minute_index=minute_index,
                open=float(getattr(bar, "open")),
                high=float(getattr(bar, "high")),
                low=float(getattr(bar, "low")),
                close=float(getattr(bar, "close")),
                volume=float(volume),
            )
        )
    return sorted(clean, key=lambda row: row.timestamp)


def _bar_payload(bar: _CleanBar) -> dict[str, Any]:
    return {
        "date": bar.date.isoformat(),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
        "split_adjusted_close": bar.split_adjusted_close,
        "adj_close": bar.adj_close,
        "vwap": bar.vwap,
    }


def _bar_lineage_payload(
    *,
    symbol: str,
    from_date: date,
    through_date: date,
    bars: Sequence[_CleanBar],
    feature_version: str,
    symbol_field: str,
    source_role: str | None = None,
    reconstruction_method: str | None = None,
) -> dict[str, Any]:
    bar_payloads = [_bar_payload(bar) for bar in bars]
    payload: dict[str, Any] = {
        symbol_field: symbol,
        "from": from_date.isoformat(),
        "to": through_date.isoformat(),
        "bar_count": len(bar_payloads),
        "first_bar_date": bar_payloads[0]["date"] if bar_payloads else None,
        "last_bar_date": bar_payloads[-1]["date"] if bar_payloads else None,
        "bars_payload_hash": stable_hash(bar_payloads),
        "lineage_payload_schema": "compact_bar_digest_v1",
        "feature_version": feature_version,
        "reconstruction_method": reconstruction_method or RECONSTRUCTION_METHOD,
    }
    if source_role is not None:
        payload["source_role"] = source_role
    return payload


def _signal_session_leakage_contract(
    *,
    predictor_features: dict[str, Any],
    outcome_features: dict[str, Any],
    predictor_input_end: date | None,
) -> dict[str, Any]:
    return {
        "asof_rule": "predictors_prior_close_or_day0_open_only",
        "predictor_input_window_end": (
            predictor_input_end.isoformat() if predictor_input_end is not None else None
        ),
        "predictor_features": predictor_features,
        "predictor_field_names": sorted(predictor_features),
        "outcome_features": outcome_features,
        "outcome_field_names": list(SIGNAL_SESSION_OUTCOME_FIELDS),
        "forbidden_predictor_inputs": [
            "day0_close",
            "day0_high",
            "day0_low",
            "forward_bars_after_open",
        ],
    }


def _lineage_quality_flags(resp: Any, **flags: Any) -> dict[str, Any]:
    return {
        **(getattr(resp.lineage, "data_quality_flags", None) or {}),
        **flags,
    }


def _relative_features(
    *,
    ticker: str,
    feature_date: date,
    ticker_momentum: dict[str, float | None],
    benchmark_series: dict[str, _ReferenceSeries],
    sector_resolution: _SectorResolution,
    sector_etf_series: dict[str, _ReferenceSeries],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    features: dict[str, Any] = {}
    status: dict[str, Any] = {
        "benchmark": {},
        "sector": {
            "sector": sector_resolution.sector,
            "sector_etf": sector_resolution.sector_etf,
            "sector_source": sector_resolution.source,
            "sector_status": sector_resolution.status,
            "pit_safe": sector_resolution.pit_safe,
            "missing_sector": sector_resolution.sector is None,
            "non_pit_sector_fallback": False,
        },
    }
    reference_inputs: dict[str, Any] = {
        "ticker": ticker,
        "feature_session_date": feature_date.isoformat(),
        "benchmarks": {},
        "sector": {
            "sector": sector_resolution.sector,
            "sector_etf": sector_resolution.sector_etf,
            "sector_source": sector_resolution.source,
            "sector_status": sector_resolution.status,
            "pit_safe": sector_resolution.pit_safe,
        },
    }

    for symbol in BENCHMARK_SYMBOLS:
        lower = symbol.lower()
        series = benchmark_series.get(symbol)
        benchmark_status = _reference_status(series, feature_date)
        status["benchmark"][symbol] = benchmark_status
        reference_inputs["benchmarks"][symbol] = _reference_input(series, feature_date)
        for sessions in BENCHMARK_RETURN_WINDOWS:
            key = f"{lower}_return_{sessions}d"
            return_value = _reference_return(series, feature_date, sessions)
            features[key] = return_value
            if f"{sessions}d" in ticker_momentum:
                features[f"relative_strength_vs_{lower}_{sessions}d"] = _relative_return(
                    ticker_momentum[f"{sessions}d"],
                    return_value,
                )
            if return_value is None and series is not None and series.status == "available":
                benchmark_status.setdefault("insufficient_history", []).append(f"{sessions}d")

    sector_etf = sector_resolution.sector_etf
    sector_series = sector_etf_series.get(sector_etf or "")
    sector_status = status["sector"]
    sector_status.update(_sector_reference_status(sector_resolution, sector_series, feature_date))
    reference_inputs["sector"]["sector_etf_input"] = _reference_input(
        sector_series,
        feature_date,
    )
    features["sector_etf"] = sector_etf
    features["sector_source"] = sector_resolution.source
    features["sector_relative_status"] = sector_status["sector_status"]
    for sessions in SECTOR_RELATIVE_RETURN_WINDOWS:
        return_value = _reference_return(sector_series, feature_date, sessions)
        features[f"sector_etf_return_{sessions}d"] = return_value
        features[f"relative_strength_vs_sector_{sessions}d"] = _relative_return(
            ticker_momentum[f"{sessions}d"],
            return_value,
        )
        if return_value is None and sector_series is not None and sector_series.status == "available":
            sector_status.setdefault("insufficient_sector_etf_history", []).append(f"{sessions}d")
    return features, status, reference_inputs


def _reference_return(
    series: _ReferenceSeries | None,
    feature_date: date,
    sessions: int,
) -> float | None:
    if series is None or series.status != "available":
        return None
    prior = [bar for bar in series.bars if bar.date < feature_date]
    if len(prior) < sessions + 1:
        return None
    return _safe_return(_price_basis(prior[-1]), _price_basis(prior[-(sessions + 1)]))


def _relative_return(ticker_return: float | None, reference_return: float | None) -> float | None:
    if ticker_return is None or reference_return is None:
        return None
    return ticker_return - reference_return


def _reference_input(series: _ReferenceSeries | None, feature_date: date) -> dict[str, Any]:
    if series is None:
        return {"status": "missing"}
    return {
        "symbol": series.symbol,
        "status": series.status,
        "bars_through_feature_session": [
            _bar_payload(bar) for bar in series.bars if bar.date <= feature_date
        ],
    }


def _reference_status(series: _ReferenceSeries | None, feature_date: date) -> dict[str, Any]:
    if series is None:
        return {
            "missing_benchmark_bars": True,
            "fetch_status": "missing",
            "lineage_id": None,
        }
    return {
        "missing_benchmark_bars": series.status != "available",
        "fetch_status": series.status,
        "lineage_id": series.data_lineage_id,
        "bars_available_through_feature_date": sum(
            1 for bar in series.bars if bar.date <= feature_date
        ),
        "fetch_error": series.error,
    }


def _sector_reference_status(
    resolution: _SectorResolution,
    series: _ReferenceSeries | None,
    feature_date: date,
) -> dict[str, Any]:
    status = {
        "missing_sector": resolution.sector is None,
        "missing_sector_etf_bars": resolution.sector_etf is not None
        and (series is None or series.status != "available"),
        "lineage_id": series.data_lineage_id if series is not None else None,
        "bars_available_through_feature_date": (
            sum(1 for bar in series.bars if bar.date <= feature_date)
            if series is not None else 0
        ),
        "fetch_error": series.error if series is not None else None,
    }
    if resolution.status != "available":
        status["sector_status"] = resolution.status
    elif status["missing_sector_etf_bars"]:
        status["sector_status"] = "missing_sector_etf_bars"
    else:
        status["sector_status"] = "available"
    return status


def _advanced_context_features(
    *,
    signal: SignalRegistry,
    feature_date: date,
    bar: _CleanBar,
    minute_bars: Sequence[_MinuteBar] = (),
    minute_layer_enabled: bool = False,
    prior: Sequence[_CleanBar],
    benchmark_series: dict[str, _ReferenceSeries],
    entry_price: float | None,
    same_day_pattern_strengths: SameDayPatternStrengthCache,
) -> tuple[dict[str, Any], dict[str, Any]]:
    market_features, market_status = _market_regime_features(
        feature_date=feature_date,
        benchmark_series=benchmark_series,
    )
    intraday_features, intraday_status = _intraday_features_from_minutes(
        minute_bars,
        minute_layer_enabled=minute_layer_enabled,
        breakout_level=(
            max(candidate.high for candidate in prior[-PRIOR_52W_SESSION_COUNT:])
            if len(prior) >= PRIOR_52W_SESSION_COUNT else None
        ),
    )
    execution_features, execution_status = _execution_quality_unavailable_features()
    supply_features, supply_status = _supply_squeeze_features(signal)
    catalyst_features, catalyst_status = _catalyst_context_features(
        signal=signal,
        feature_date=feature_date,
        same_day_pattern_strengths=same_day_pattern_strengths,
    )
    technical_features, technical_status = _classic_technical_features(prior)
    features = {
        **market_features,
        **intraday_features,
        **execution_features,
        **supply_features,
        **catalyst_features,
        **technical_features,
    }
    status = {
        "market_regime": {"features": market_features, "status": market_status},
        "intraday_structure": {
            "features": intraday_features,
            "status": intraday_status,
        },
        "execution_quality": {
            "features": execution_features,
            "status": execution_status,
        },
        "supply_squeeze": {"features": supply_features, "status": supply_status},
        "catalyst_context": {
            "features": catalyst_features,
            "status": catalyst_status,
        },
        "technical_indicators": {
            "features": technical_features,
            "status": technical_status,
        },
    }
    return features, status


def _market_regime_features(
    *,
    feature_date: date,
    benchmark_series: dict[str, _ReferenceSeries],
) -> tuple[dict[str, Any], dict[str, Any]]:
    vol_inputs = {
        symbol: _reference_realized_volatility(
            benchmark_series.get(symbol),
            feature_date,
            20,
        )
        for symbol in ("SPY", "IWM")
    }
    available = {
        symbol: value for symbol, value in vol_inputs.items()
        if value is not None and math.isfinite(float(value))
    }
    volatility_proxy = (
        float(statistics.mean(available.values())) if available else None
    )
    volatility_source = (
        "_".join(sorted(available)) + "_REALIZED_VOL_20D"
        if available else None
    )
    status_text = (
        "volatility_proxy_available_breadth_unavailable"
        if volatility_proxy is not None else "regime_sources_unavailable"
    )
    features = {
        "universe_pct_above_sma_20d": None,
        "universe_pct_above_sma_50d": None,
        "universe_pct_making_20d_highs": None,
        "universe_pct_making_52w_highs": None,
        "volatility_regime_proxy": volatility_proxy,
        "volatility_regime_source": volatility_source,
        "market_regime_status": status_text,
    }
    status = {
        "status": status_text,
        "breadth_source_available": False,
        "breadth_source_status": "operating_universe_breadth_not_wired",
        "volatility_source": volatility_source,
        "volatility_proxy_available": volatility_proxy is not None,
        "volatility_fallback": "spy_iwm_realized_volatility_20d",
        "missing_regime_source": volatility_proxy is None,
        "prior_only": True,
    }
    return features, status


def _intraday_unavailable_features(
    *,
    status_text: str = "intraday_adapter_unavailable",
) -> tuple[dict[str, Any], dict[str, Any]]:
    features = {
        "opening_range_high_5m": None,
        "opening_range_low_5m": None,
        "opening_range_high_15m": None,
        "opening_range_low_15m": None,
        "opening_range_high_30m": None,
        "opening_range_low_30m": None,
        "opening_range_high_60m": None,
        "opening_range_low_60m": None,
        "first_5m_return": None,
        "first_15m_return": None,
        "first_30m_return": None,
        "first_60m_return": None,
        "intraday_vwap": None,
        "open_vs_intraday_vwap_pct": None,
        "close_vs_intraday_vwap_pct": None,
        "intraday_volume_5m": None,
        "intraday_volume_15m": None,
        "intraday_volume_30m": None,
        "intraday_volume_60m": None,
        "pct_expected_volume_5m": None,
        "pct_expected_volume_15m": None,
        "pct_expected_volume_30m": None,
        "pct_expected_volume_60m": None,
        "held_above_breakout_after_first_hour": None,
        "intraday_mfe_timestamp": None,
        "intraday_mae_timestamp": None,
        "t1_before_stop": None,
        "intraday_structure_status": status_text,
        "missing_intraday_bars": True,
    }
    status = {
        "status": status_text,
        "missing_intraday_bars": True,
        "values_null_by_design": True,
    }
    return features, status


def _intraday_features_from_minutes(
    minute_bars: Sequence[_MinuteBar],
    *,
    minute_layer_enabled: bool,
    breakout_level: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not minute_bars:
        status_text = (
            "minute_bars_missing"
            if minute_layer_enabled
            else "intraday_adapter_unavailable"
        )
        return _intraday_unavailable_features(status_text=status_text)
    ordered = sorted(minute_bars, key=lambda row: row.minute_index)
    day_open = ordered[0].open
    day_close = ordered[-1].close
    total_volume = sum(row.volume for row in ordered)

    def window(minutes: int) -> list[_MinuteBar]:
        return [row for row in ordered if row.minute_index < minutes]

    features, _status = _intraday_unavailable_features(status_text="available")
    for minutes in (5, 15, 30, 60):
        rows = window(minutes)
        high = max((row.high for row in rows), default=None)
        low = min((row.low for row in rows), default=None)
        close = rows[-1].close if rows else None
        volume = sum(row.volume for row in rows)
        features[f"opening_range_high_{minutes}m"] = high
        features[f"opening_range_low_{minutes}m"] = low
        features[f"first_{minutes}m_return"] = _safe_return(close, day_open)
        features[f"intraday_volume_{minutes}m"] = volume if rows else None
        features[f"pct_expected_volume_{minutes}m"] = (
            volume / total_volume if rows and total_volume > 0 else None
        )
    vwap_denom = sum(row.volume for row in ordered)
    intraday_vwap = (
        sum(((row.high + row.low + row.close) / 3.0) * row.volume for row in ordered)
        / vwap_denom
        if vwap_denom > 0 else None
    )
    features["intraday_vwap"] = intraday_vwap
    features["open_vs_intraday_vwap_pct"] = _safe_return(day_open, intraday_vwap)
    features["close_vs_intraday_vwap_pct"] = _safe_return(day_close, intraday_vwap)
    after_first_hour = [row for row in ordered if row.minute_index >= 60]
    features["held_above_breakout_after_first_hour"] = (
        None
        if breakout_level is None or not after_first_hour
        else all(row.low >= breakout_level for row in after_first_hour)
    )
    mfe_bar = max(ordered, key=lambda row: row.high)
    mae_bar = min(ordered, key=lambda row: row.low)
    features["intraday_mfe_timestamp"] = mfe_bar.timestamp
    features["intraday_mae_timestamp"] = mae_bar.timestamp
    features["intraday_structure_status"] = "available"
    features["missing_intraday_bars"] = False
    status = {
        "status": "available",
        "missing_intraday_bars": False,
        "minute_bar_count": len(ordered),
        "first_minute": ordered[0].timestamp.isoformat(),
        "last_minute": ordered[-1].timestamp.isoformat(),
    }
    return features, status


def _execution_quality_unavailable_features() -> tuple[dict[str, Any], dict[str, Any]]:
    features = {
        "bid_ask_spread": None,
        "bid_ask_spread_pct": None,
        "quote_age_seconds": None,
        "bid_size": None,
        "ask_size": None,
        "intended_entry_vs_mid_pct": None,
        "intended_entry_vs_ask_pct": None,
        "intended_entry_vs_bid_pct": None,
        "volume_participation_pct": None,
        "halt_risk_flag": None,
        "offering_risk_flag": None,
        "missing_quote": True,
        "stale_quote": None,
        "quote_status": "quote_source_unavailable",
        "execution_quality_status": "quote_execution_sources_unavailable",
    }
    status = {
        "status": "quote_execution_sources_unavailable",
        "missing_quote": True,
        "stale_quote": None,
        "realized_slippage_available": False,
        "missed_fill_rate_available": False,
    }
    return features, status


def _supply_squeeze_features(signal: SignalRegistry | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _signal_feature_payload(signal) if signal is not None else {}
    context = payload.get("signal_context") if isinstance(payload.get("signal_context"), dict) else {}
    profile = context.get("fmp_profile") if isinstance(context.get("fmp_profile"), dict) else {}
    short_interest = (
        context.get("polygon_short_interest")
        if isinstance(context.get("polygon_short_interest"), dict)
        else {}
    )
    short_volume = (
        context.get("polygon_short_volume")
        if isinstance(context.get("polygon_short_volume"), dict)
        else {}
    )
    float_shares = _optional_float(profile.get("float_shares") or profile.get("float"))
    shares_outstanding = _optional_float(profile.get("shares_outstanding"))
    short_interest_latest = (
        short_interest.get("latest")
        if isinstance(short_interest.get("latest"), dict)
        else {}
    )
    short_volume_latest = (
        short_volume.get("latest")
        if isinstance(short_volume.get("latest"), dict)
        else {}
    )
    short_interest_shares = _optional_float(
        short_interest.get("short_interest")
        if short_interest.get("short_interest") is not None
        else short_interest_latest.get("short_interest")
    )
    short_interest_days_to_cover = _optional_float(
        short_interest.get("days_to_cover")
        if short_interest.get("days_to_cover") is not None
        else short_interest_latest.get("days_to_cover")
    )
    short_volume_ratio = _optional_float(
        short_volume.get("short_volume_ratio")
        if short_volume.get("short_volume_ratio") is not None
        else short_volume_latest.get("short_volume_ratio")
    )
    short_interest_pct_float = (
        short_interest_shares / float_shares
        if short_interest_shares is not None and float_shares not in (None, 0)
        else None
    )
    if any(
        value is not None
        for value in (
            float_shares,
            shares_outstanding,
            short_interest_shares,
            short_interest_days_to_cover,
            short_volume_ratio,
        )
    ):
        features = {
            "float_shares": float_shares,
            "shares_outstanding": shares_outstanding,
            "turnover_float": None,
            "dollar_turnover_float": None,
            "short_volume_ratio": short_volume_ratio,
            "short_interest_pct_float": short_interest_pct_float,
            "short_interest_shares": short_interest_shares,
            "short_interest_days_to_cover": short_interest_days_to_cover,
            "proxy_days_to_cover": short_interest_days_to_cover,
            "borrow_fee_rate": None,
            "float_source_status": "snapshot_context_available" if float_shares is not None else "pit_float_source_unavailable",
            "short_source_status": "snapshot_context_available",
            "borrow_fee_status": "not_available_historical",
            "supply_squeeze_status": "snapshot_context_available_borrow_unavailable",
        }
        return features, {
            "status": features["supply_squeeze_status"],
            "float_source_status": features["float_source_status"],
            "short_source_status": features["short_source_status"],
            "borrow_fee_status": features["borrow_fee_status"],
        }
    features = {
        "float_shares": None,
        "shares_outstanding": None,
        "turnover_float": None,
        "dollar_turnover_float": None,
        "short_volume_ratio": None,
        "short_interest_pct_float": None,
        "short_interest_shares": None,
        "short_interest_days_to_cover": None,
        "proxy_days_to_cover": None,
        "borrow_fee_rate": None,
        "float_source_status": "pit_float_source_unavailable",
        "short_source_status": "short_interest_source_unavailable",
        "borrow_fee_status": "not_available_historical",
        "supply_squeeze_status": "pit_safe_sources_unavailable",
    }
    status = {
        "status": "pit_safe_sources_unavailable",
        "float_source_status": features["float_source_status"],
        "short_source_status": features["short_source_status"],
        "borrow_fee_status": features["borrow_fee_status"],
    }
    return features, status


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _catalyst_context_features(
    *,
    signal: SignalRegistry,
    feature_date: date,
    same_day_pattern_strengths: SameDayPatternStrengthCache,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pattern_strength = _same_day_pattern_strengths_from_cache(
        same_day_pattern_strengths,
        signal,
        feature_date,
    )
    cofire_m2 = any(pattern in pattern_strength for pattern in ("M2", "M2U"))
    strongest_pattern = None
    if pattern_strength:
        strongest_pattern = sorted(
            pattern_strength.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
    snapshot_catalyst = _snapshot_catalyst_features(signal, feature_date)
    features = {
        "news_count_1d": None,
        "news_count_5d": None,
        "news_count_20d": None,
        "news_catalyst_flags_json": None,
        "earnings_days_to_next": None,
        "earnings_days_since_last": None,
        "offering_flag": None,
        "atm_flag": None,
        "shelf_registration_flag": None,
        "insider_buy_overlap_m2": cofire_m2,
        "cofire_m1": "M1" in pattern_strength,
        "cofire_m2": cofire_m2,
        "cofire_m3": any(pattern in pattern_strength for pattern in ("M3", "M3S")),
        "cofire_m4": "M4" in pattern_strength,
        "cofire_i11": "I11" in pattern_strength,
        "fda_clinical_flag": None,
        "corporate_action_flag": None,
        "cross_pattern_overlap_count": len(pattern_strength),
        "strongest_overlap_pattern_id": strongest_pattern,
        "catalyst_context_status": "signal_registry_context_available_external_sources_missing",
        "missing_catalyst_source": True,
    }
    if snapshot_catalyst is not None:
        features.update(snapshot_catalyst["features"])
    status = {
        "status": features["catalyst_context_status"],
        "signal_registry_context_available": True,
        "external_catalyst_sources_available": snapshot_catalyst is not None,
        "patterns_present": sorted(pattern_strength),
        "missing_catalyst_source": features["missing_catalyst_source"],
    }
    if snapshot_catalyst is not None:
        status.update(snapshot_catalyst["status"])
    return features, status


def _snapshot_catalyst_features(
    signal: SignalRegistry,
    feature_date: date,
) -> dict[str, Any] | None:
    payload = _signal_feature_payload(signal)
    source_events = payload.get("day0_catalyst_events")
    if source_events is None:
        source_events = (
            payload.get("signal_context", {}).get("day0_catalyst_events")
            if isinstance(payload.get("signal_context"), dict)
            else None
        )
    if not isinstance(source_events, list):
        return None
    cutoff = us_equity_session_open_timestamp(feature_date)
    eligible: list[dict[str, Any]] = []
    excluded_after_open = 0
    for event in source_events:
        if not isinstance(event, dict):
            continue
        event_ts = _parse_event_timestamp(
            event.get("timestamp")
            or event.get("published_at")
            or event.get("accepted_at")
            or event.get("updated")
        )
        if event_ts is None:
            continue
        if event_ts > cutoff:
            excluded_after_open += 1
            continue
        eligible.append({**event, "_timestamp": event_ts})

    def news_count(days: int) -> int:
        start = cutoff - timedelta(days=days)
        return sum(
            1
            for event in eligible
            if str(event.get("type", "news")).lower() == "news"
            and event["_timestamp"] >= start
        )

    flags = sorted({
        str(event.get("flag") or event.get("category") or event.get("form") or "").lower()
        for event in eligible
        if event.get("flag") or event.get("category") or event.get("form")
    })
    earnings_dates = [
        _parse_date_or_none(event.get("earnings_date")) or event["_timestamp"].date()
        for event in eligible
        if str(event.get("type", "")).lower() == "earnings"
    ]
    future_earnings = [item for item in earnings_dates if item >= feature_date]
    past_earnings = [item for item in earnings_dates if item <= feature_date]
    features = {
        "news_count_1d": news_count(1),
        "news_count_5d": news_count(5),
        "news_count_20d": news_count(20),
        "news_catalyst_flags_json": json.dumps(flags, sort_keys=True),
        "earnings_days_to_next": (
            min((item - feature_date).days for item in future_earnings)
            if future_earnings else None
        ),
        "earnings_days_since_last": (
            min((feature_date - item).days for item in past_earnings)
            if past_earnings else None
        ),
        "offering_flag": any("offering" in flag for flag in flags),
        "atm_flag": any(flag == "atm" or "atm" in flag for flag in flags),
        "shelf_registration_flag": any("shelf" in flag or flag == "s-3" for flag in flags),
        "fda_clinical_flag": any("fda" in flag or "clinical" in flag for flag in flags),
        "corporate_action_flag": any("corporate" in flag for flag in flags),
        "catalyst_context_status": "snapshot_catalyst_events_pit_filtered",
        "missing_catalyst_source": False,
    }
    status = {
        "snapshot_event_count": len(source_events),
        "eligible_event_count": len(eligible),
        "excluded_after_day0_open_count": excluded_after_open,
        "pit_cutoff": cutoff.isoformat(),
    }
    return {"features": features, "status": status}


def _signal_feature_payload(signal: SignalRegistry) -> dict[str, Any]:
    feature = getattr(signal, "feature_snapshot", None)
    if feature is None or not getattr(feature, "feature_json", None):
        return {}
    try:
        parsed = json.loads(feature.feature_json)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_event_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        try:
            return _ensure_aware(value)
        except ValueError:
            return value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_date_or_none(value: Any) -> date | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _same_day_pattern_strengths_from_cache(
    same_day_pattern_strengths: SameDayPatternStrengthCache,
    signal: SignalRegistry,
    feature_date: date,
) -> dict[str, float]:
    signal_day = signal.signal_timestamp.date()
    if feature_date < signal_day:
        return {}
    return dict(same_day_pattern_strengths.get((signal.ticker.upper(), signal_day), {}))


def _prefetch_same_day_pattern_strengths(
    session: Session,
    signals: Sequence[SignalRegistry],
) -> SameDayPatternStrengthCache:
    if not signals:
        return {}
    tickers = sorted({signal.ticker for signal in signals if signal.ticker})
    signal_days = [signal.signal_timestamp.date() for signal in signals]
    start_dt = datetime.combine(min(signal_days), time.min, timezone.utc)
    end_dt = datetime.combine(max(signal_days) + timedelta(days=1), time.min, timezone.utc)
    rows = (
        session.query(
            SignalRegistry.ticker,
            SignalRegistry.pattern_id,
            SignalRegistry.signal_timestamp,
            SignalRegistry.raw_signal_strength,
        )
        .filter(
            SignalRegistry.ticker.in_(tickers),
            SignalRegistry.signal_timestamp >= start_dt,
            SignalRegistry.signal_timestamp < end_dt,
        )
        .all()
    )
    cache: SameDayPatternStrengthCache = {}
    for ticker, pattern_id, signal_timestamp, strength in rows:
        signal_day = signal_timestamp.date()
        key = (str(ticker).upper(), signal_day)
        strengths = cache.setdefault(key, {})
        normalized = str(pattern_id).upper()
        parsed_strength = float(strength or 0.0)
        strengths[normalized] = max(strengths.get(normalized, parsed_strength), parsed_strength)
    return cache


def _classic_technical_features(
    prior: Sequence[_CleanBar],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plus_di, minus_di, adx = _directional_indicators(prior, 14)
    features = {
        "rsi_2": _rsi(prior, 2),
        "rsi_5": _rsi(prior, 5),
        "rsi_14": _rsi(prior, 14),
        "adx_14": adx,
        "plus_di_14": plus_di,
        "minus_di_14": minus_di,
        "bollinger_bandwidth_20d": _bollinger_bandwidth(prior, 20),
        "bollinger_percent_b_20d": _bollinger_percent_b(prior, 20),
        "keltner_channel_position_20d": _keltner_position(prior, 20),
        "macd_histogram": _macd_histogram(prior),
        "obv": _obv(prior),
        "accumulation_distribution": _accumulation_distribution(prior),
        "chaikin_money_flow_20d": _chaikin_money_flow(prior, 20),
        "stochastic_oscillator_14d": _stochastic_oscillator(prior, 14),
    }
    insufficient = {
        key: value is None
        for key, value in features.items()
    }
    status_text = (
        "insufficient_history"
        if all(insufficient.values()) else "available_with_prior_only_inputs"
    )
    features["technical_indicator_status"] = status_text
    status = {
        "status": status_text,
        "prior_only": True,
        "prior_bar_count": len(prior),
        "insufficient_history": insufficient,
    }
    return features, status


def _reference_realized_volatility(
    series: _ReferenceSeries | None,
    feature_date: date,
    sessions: int,
) -> float | None:
    if series is None or series.status != "available":
        return None
    prior = [bar for bar in series.bars if bar.date < feature_date]
    return _realized_volatility(prior, sessions)


def _reference_error_count(series: dict[str, _ReferenceSeries]) -> int:
    return sum(1 for value in series.values() if value.status == "fetch_error")


def _cross_sectional_tracked_fields() -> tuple[str, ...]:
    fields = [
        "cohort_feature_row_count",
        "cohort_pattern_row_count",
    ]
    for rank_field, percentile_field in RANK_INPUT_TO_OUTPUT.values():
        fields.extend([rank_field, percentile_field])
    return tuple(fields)


def _empty_cross_sectional_features() -> dict[str, Any]:
    fields: dict[str, Any] = {
        "cohort_feature_row_count": None,
        "cohort_pattern_row_count": None,
    }
    for rank_field, percentile_field in RANK_INPUT_TO_OUTPUT.values():
        fields[rank_field] = None
        fields[percentile_field] = None
    return fields


def _weekdays_between(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _rank_input_value(row: MarketPathFeature, source_field: str) -> float | None:
    value = getattr(row, source_field)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _json_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _refresh_output_hash(row: MarketPathFeature) -> None:
    row.output_hash = _output_hash_for_row(row)


def _output_hash_for_row(
    row: MarketPathFeature,
    **overrides: Any,
) -> str:
    output_payload = {
        key: overrides.get(key, getattr(row, key))
        for key in ML_OUTPUT_HASH_FIELDS
    }
    return stable_hash(output_payload)


def _price_basis(bar: _CleanBar) -> float:
    return bar.split_adjusted_close if bar.split_adjusted_close is not None else bar.close


def _previous_bar(bars: Sequence[_CleanBar], current_date: date) -> _CleanBar | None:
    previous = [bar for bar in bars if bar.date < current_date]
    return previous[-1] if previous else None


def _median(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(statistics.median(finite))


def _sigma_close_to_close(bars: Sequence[_CleanBar]) -> float | None:
    returns = [
        _price_basis(current) / _price_basis(previous) - 1.0
        for previous, current in zip(bars[:-1], bars[1:])
        if _price_basis(previous) > 0
    ]
    if len(returns) < 2:
        return None
    return float(statistics.stdev(returns))


def _rich_eod_features(
    *,
    bar: _CleanBar,
    previous: _CleanBar | None,
    prior: Sequence[_CleanBar],
    asof_prior_close: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictor_bar = previous if asof_prior_close and previous is not None else bar
    predictor_history = list(prior)
    prior20 = prior[-20:]
    prior60 = prior[-60:]
    prior126 = prior[-126:]
    prior252 = prior[-PRIOR_52W_SESSION_COUNT:]
    prior_52w_high = (
        max(candidate.high for candidate in prior252)
        if len(prior252) >= PRIOR_52W_SESSION_COUNT else None
    )
    prior_close = _price_basis(previous) if previous is not None else None
    current_range = bar.high - bar.low
    predictor_range = predictor_bar.high - predictor_bar.low
    prior_ranges_20 = [candidate.high - candidate.low for candidate in prior20]
    vwap = bar.vwap if bar.vwap is not None and bar.vwap > 0 else None
    predictor_vwap = (
        predictor_bar.vwap
        if predictor_bar.vwap is not None and predictor_bar.vwap > 0
        else None
    )
    rich = {
        "prior_52w_high": prior_52w_high,
        "breakout_extension_pct": _safe_return(bar.close, prior_52w_high),
        "open_vs_52w_high_pct": _safe_return(bar.open, prior_52w_high),
        "close_vs_52w_high_pct": _safe_return(bar.close, prior_52w_high),
        "high_vs_52w_high_pct": _safe_return(bar.high, prior_52w_high),
        "prior_close_vs_52w_high_pct": _safe_return(prior_close, prior_52w_high),
        "gap_over_breakout": (
            bar.open > prior_52w_high if prior_52w_high is not None else None
        ),
        "closed_above_breakout": (
            bar.close > prior_52w_high if prior_52w_high is not None else None
        ),
        "close_location_value": (
            (bar.close - bar.low) / current_range if current_range > 0 else None
        ),
        "upper_wick_ratio": (
            (bar.high - max(bar.open, bar.close)) / current_range
            if current_range > 0 else None
        ),
        "lower_wick_ratio": (
            (min(bar.open, bar.close) - bar.low) / current_range
            if current_range > 0 else None
        ),
        "true_range_pct": _true_range_pct(bar, previous),
        # Prior-only by design: ATR uses the 14 completed sessions before the
        # feature date, not the feature-date range.
        "atr_14_pct": _atr_14_pct(prior),
        "range_expansion_vs_20d": _safe_ratio(
            predictor_range if predictor_range > 0 else None,
            _median(prior_ranges_20),
        ),
        "volume_zscore_20d": _zscore(predictor_bar.volume, [candidate.volume for candidate in prior20]),
        "volume_zscore_60d": _zscore(predictor_bar.volume, [candidate.volume for candidate in prior60]),
        "dollar_volume_zscore_20d": _zscore(
            predictor_bar.dollar_volume, [candidate.dollar_volume for candidate in prior20]
        ),
        "dollar_volume_zscore_60d": _zscore(
            predictor_bar.dollar_volume, [candidate.dollar_volume for candidate in prior60]
        ),
        "volume_acceleration_1d_vs_5d": _safe_return(
            predictor_bar.volume, _mean([candidate.volume for candidate in prior[-5:]])
        ),
        "volume_acceleration_1d_vs_20d": _safe_return(
            predictor_bar.volume, _mean([candidate.volume for candidate in prior20])
        ),
        "realized_volatility_5d": _realized_volatility(prior, 5),
        "realized_volatility_10d": _realized_volatility(prior, 10),
        "realized_volatility_20d": _realized_volatility(prior, 20),
        "base_range_10d": _base_range(prior, 10),
        "base_range_20d": _base_range(prior, 20),
        "base_range_60d": _base_range(prior, 60),
        "base_max_drawdown_10d": _base_max_drawdown(prior, 10),
        "base_max_drawdown_20d": _base_max_drawdown(prior, 20),
        "base_max_drawdown_60d": _base_max_drawdown(prior, 60),
        "distance_from_sma_20d": _sma_distance(predictor_bar, predictor_history, 20),
        "distance_from_sma_50d": _sma_distance(predictor_bar, predictor_history, 50),
        "distance_from_sma_200d": _sma_distance(predictor_bar, predictor_history, 200),
        "momentum_5d": _momentum(prior, 5),
        "momentum_20d": _momentum(prior, 20),
        "momentum_60d": _momentum(prior, 60),
        "prior_52w_high_touches_20d": _prior_high_touch_count(prior20, prior_52w_high),
        "prior_52w_high_touches_60d": _prior_high_touch_count(prior60, prior_52w_high),
        "prior_52w_high_touches_126d": _prior_high_touch_count(prior126, prior_52w_high),
        "age_of_52w_high_sessions": _age_of_prior_high(prior, prior_52w_high),
        "failed_breakout_count_20d": _failed_breakout_count(prior, 20),
        "failed_breakout_count_60d": _failed_breakout_count(prior, 60),
        "failed_breakout_count_126d": _failed_breakout_count(prior, 126),
        "vwap": predictor_vwap,
        "open_vs_vwap_pct": _safe_return(predictor_bar.open, predictor_vwap),
        "high_vs_vwap_pct": _safe_return(predictor_bar.high, predictor_vwap),
        "low_vs_vwap_pct": _safe_return(predictor_bar.low, predictor_vwap),
        "close_vs_vwap_pct": _safe_return(predictor_bar.close, predictor_vwap),
    }
    status = {
        "prior_only_trailing_features": True,
        "asof_prior_close": asof_prior_close,
        "atr_basis": "prior_14_completed_sessions",
        "touch_tolerance_pct": TOUCH_TOLERANCE_PCT,
        "prior_bar_count": len(prior),
        "missing_vwap": vwap is None,
        "insufficient_history": {
            "prior_52w_high": len(prior252) < PRIOR_52W_SESSION_COUNT,
            "atr_14_pct": len(_true_ranges_for_completed_sessions(prior)) < 14,
            "volume_zscore_20d": len(prior20) < 20,
            "volume_zscore_60d": len(prior60) < 60,
            "dollar_volume_zscore_20d": len(prior20) < 20,
            "dollar_volume_zscore_60d": len(prior60) < 60,
            "realized_volatility_5d": len(prior) < 6,
            "realized_volatility_10d": len(prior) < 11,
            "realized_volatility_20d": len(prior) < 21,
            "base_range_10d": len(prior) < 10,
            "base_range_20d": len(prior) < 20,
            "base_range_60d": len(prior) < 60,
            "base_max_drawdown_10d": len(prior) < 10,
            "base_max_drawdown_20d": len(prior) < 20,
            "base_max_drawdown_60d": len(prior) < 60,
            "distance_from_sma_20d": len(prior) < 20,
            "distance_from_sma_50d": len(prior) < 50,
            "distance_from_sma_200d": len(prior) < 200,
            "momentum_5d": len(prior) < 6,
            "momentum_20d": len(prior) < 21,
            "momentum_60d": len(prior) < 61,
            "prior_52w_high_touches_20d": prior_52w_high is None or len(prior20) < 20,
            "prior_52w_high_touches_60d": prior_52w_high is None or len(prior60) < 60,
            "prior_52w_high_touches_126d": prior_52w_high is None or len(prior126) < 126,
            "failed_breakout_count_20d": len(prior) < 20,
            "failed_breakout_count_60d": len(prior) < 60,
            "failed_breakout_count_126d": len(prior) < 126,
        },
    }
    return rich, status


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(statistics.mean(finite))


def _zscore(value: float | None, values: Sequence[float]) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    finite = [float(candidate) for candidate in values if math.isfinite(float(candidate))]
    if len(finite) < 2:
        return None
    std = float(statistics.stdev(finite))
    if std <= 0:
        return None
    return (float(value) - float(statistics.mean(finite))) / std


def _true_range_abs(bar: _CleanBar, previous: _CleanBar | None) -> float | None:
    if previous is None:
        return None
    prior_close = _price_basis(previous)
    if prior_close <= 0:
        return None
    return max(
        bar.high - bar.low,
        abs(bar.high - prior_close),
        abs(bar.low - prior_close),
    )


def _true_range_pct(bar: _CleanBar, previous: _CleanBar | None) -> float | None:
    true_range = _true_range_abs(bar, previous)
    prior_close = _price_basis(previous) if previous is not None else None
    return _safe_ratio(true_range, prior_close)


def _true_ranges_for_completed_sessions(bars: Sequence[_CleanBar]) -> list[float]:
    ranges: list[float] = []
    for previous, current in zip(bars[:-1], bars[1:]):
        true_range = _true_range_abs(current, previous)
        if true_range is not None:
            ranges.append(true_range)
    return ranges


def _atr_14_pct(prior: Sequence[_CleanBar]) -> float | None:
    if not prior:
        return None
    ranges = _true_ranges_for_completed_sessions(prior)[-14:]
    if len(ranges) < 14:
        return None
    return _safe_ratio(float(statistics.mean(ranges)), _price_basis(prior[-1]))


def _realized_volatility(prior: Sequence[_CleanBar], sessions: int) -> float | None:
    window = prior[-(sessions + 1):]
    if len(window) < sessions + 1:
        return None
    return _sigma_close_to_close(window)


def _base_range(prior: Sequence[_CleanBar], sessions: int) -> float | None:
    window = prior[-sessions:]
    if len(window) < sessions:
        return None
    last_prior_close = _price_basis(window[-1])
    if last_prior_close <= 0:
        return None
    return (max(bar.high for bar in window) - min(bar.low for bar in window)) / last_prior_close


def _base_max_drawdown(prior: Sequence[_CleanBar], sessions: int) -> float | None:
    window = prior[-sessions:]
    if len(window) < sessions:
        return None
    peak: float | None = None
    max_drawdown = 0.0
    for bar in window:
        close = _price_basis(bar)
        if peak is None or close > peak:
            peak = close
        if peak and peak > 0:
            max_drawdown = max(max_drawdown, 1.0 - close / peak)
    return max_drawdown


def _sma_distance(bar: _CleanBar, prior: Sequence[_CleanBar], sessions: int) -> float | None:
    window = prior[-sessions:]
    if len(window) < sessions:
        return None
    return _safe_return(bar.close, _mean([_price_basis(candidate) for candidate in window]))


def _momentum(prior: Sequence[_CleanBar], sessions: int) -> float | None:
    if len(prior) < sessions + 1:
        return None
    return _safe_return(_price_basis(prior[-1]), _price_basis(prior[-(sessions + 1)]))


def _prior_high_touch_count(
    bars: Sequence[_CleanBar],
    prior_high: float | None,
) -> int | None:
    if prior_high is None or prior_high <= 0:
        return None
    threshold = prior_high * (1.0 - TOUCH_TOLERANCE_PCT)
    return sum(1 for bar in bars if bar.high >= threshold)


def _age_of_prior_high(
    prior: Sequence[_CleanBar],
    prior_high: float | None,
) -> int | None:
    if prior_high is None:
        return None
    for index in range(len(prior) - 1, -1, -1):
        if prior[index].high == prior_high:
            return len(prior) - 1 - index
    return None


def _failed_breakout_count(prior: Sequence[_CleanBar], sessions: int) -> int | None:
    window = prior[-sessions:]
    if len(window) < sessions:
        return None
    count = 0
    for candidate in window:
        candidate_prior = [bar for bar in prior if bar.date < candidate.date]
        if not candidate_prior:
            continue
        threshold = max(bar.high for bar in candidate_prior[-PRIOR_52W_SESSION_COUNT:])
        if candidate.high > threshold and candidate.close <= threshold:
            count += 1
    return count


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _safe_return(value: float | None, basis: float | None) -> float | None:
    if value is None or basis is None or basis <= 0:
        return None
    return value / basis - 1.0


def _rsi(prior: Sequence[_CleanBar], sessions: int) -> float | None:
    window = prior[-(sessions + 1):]
    if len(window) < sessions + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for previous, current in zip(window[:-1], window[1:]):
        change = _price_basis(current) - _price_basis(previous)
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))
    avg_gain = _mean(gains)
    avg_loss = _mean(losses)
    if avg_gain is None or avg_loss is None:
        return None
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _directional_indicators(
    prior: Sequence[_CleanBar],
    sessions: int,
) -> tuple[float | None, float | None, float | None]:
    intervals: list[tuple[float, float, float]] = []
    for previous, current in zip(prior[:-1], prior[1:]):
        up_move = current.high - previous.high
        down_move = previous.low - current.low
        plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0
        true_range = _true_range_abs(current, previous)
        if true_range is None:
            continue
        intervals.append((true_range, plus_dm, minus_dm))

    if len(intervals) < sessions:
        return None, None, None

    smoothed_tr = sum(item[0] for item in intervals[:sessions])
    smoothed_plus_dm = sum(item[1] for item in intervals[:sessions])
    smoothed_minus_dm = sum(item[2] for item in intervals[:sessions])
    if smoothed_tr <= 0:
        return None, None, None

    dx_values: list[float] = []
    latest_plus_di: float | None = None
    latest_minus_di: float | None = None
    for index, (true_range, plus_dm, minus_dm) in enumerate(
        intervals[sessions - 1:],
        start=sessions - 1,
    ):
        if index > sessions - 1:
            smoothed_tr = smoothed_tr - (smoothed_tr / sessions) + true_range
            smoothed_plus_dm = (
                smoothed_plus_dm - (smoothed_plus_dm / sessions) + plus_dm
            )
            smoothed_minus_dm = (
                smoothed_minus_dm - (smoothed_minus_dm / sessions) + minus_dm
            )
        if smoothed_tr <= 0:
            continue
        latest_plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
        latest_minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
        di_sum = latest_plus_di + latest_minus_di
        dx_values.append(
            100.0 * abs(latest_plus_di - latest_minus_di) / di_sum
            if di_sum > 0 else 0.0
        )

    if latest_plus_di is None or latest_minus_di is None:
        return None, None, None
    if len(dx_values) < sessions:
        return latest_plus_di, latest_minus_di, None

    adx = float(statistics.mean(dx_values[:sessions]))
    for dx in dx_values[sessions:]:
        adx = ((adx * (sessions - 1)) + dx) / sessions
    return latest_plus_di, latest_minus_di, adx


def _bollinger_bandwidth(prior: Sequence[_CleanBar], sessions: int) -> float | None:
    window = prior[-sessions:]
    if len(window) < sessions:
        return None
    closes = [_price_basis(bar) for bar in window]
    middle = _mean(closes)
    if middle is None or middle <= 0 or len(closes) < 2:
        return None
    std = float(statistics.stdev(closes))
    return (4.0 * std) / middle


def _bollinger_percent_b(prior: Sequence[_CleanBar], sessions: int) -> float | None:
    window = prior[-sessions:]
    if len(window) < sessions:
        return None
    closes = [_price_basis(bar) for bar in window]
    middle = _mean(closes)
    if middle is None or len(closes) < 2:
        return None
    std = float(statistics.stdev(closes))
    upper = middle + 2.0 * std
    lower = middle - 2.0 * std
    width = upper - lower
    if width <= 0:
        return None
    return (_price_basis(window[-1]) - lower) / width


def _keltner_position(prior: Sequence[_CleanBar], sessions: int) -> float | None:
    window = prior[-sessions:]
    if len(window) < sessions:
        return None
    middle = _mean([_price_basis(bar) for bar in window])
    ranges = _true_ranges_for_completed_sessions(window)
    if middle is None or len(ranges) < sessions - 1:
        return None
    atr = float(statistics.mean(ranges))
    upper = middle + 2.0 * atr
    lower = middle - 2.0 * atr
    width = upper - lower
    if width <= 0:
        return None
    return (_price_basis(window[-1]) - lower) / width


def _macd_histogram(prior: Sequence[_CleanBar]) -> float | None:
    closes = [_price_basis(bar) for bar in prior]
    if len(closes) < 35:
        return None
    ema12 = _ema_series(closes, 12)
    ema26 = _ema_series(closes, 26)
    macd_line = [fast - slow for fast, slow in zip(ema12, ema26)]
    signal_line = _ema_series(macd_line, 9)
    if not signal_line:
        return None
    return macd_line[-1] - signal_line[-1]


def _ema_series(values: Sequence[float], sessions: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (sessions + 1.0)
    ema = [float(values[0])]
    for value in values[1:]:
        ema.append(alpha * float(value) + (1.0 - alpha) * ema[-1])
    return ema


def _obv(prior: Sequence[_CleanBar]) -> float | None:
    if len(prior) < 2:
        return None
    value = 0.0
    for previous, current in zip(prior[:-1], prior[1:]):
        if _price_basis(current) > _price_basis(previous):
            value += current.volume
        elif _price_basis(current) < _price_basis(previous):
            value -= current.volume
    return value


def _accumulation_distribution(prior: Sequence[_CleanBar]) -> float | None:
    if not prior:
        return None
    total = 0.0
    used = 0
    for bar in prior:
        range_width = bar.high - bar.low
        if range_width <= 0:
            continue
        money_flow_multiplier = ((bar.close - bar.low) - (bar.high - bar.close)) / range_width
        total += money_flow_multiplier * bar.volume
        used += 1
    return total if used else None


def _chaikin_money_flow(prior: Sequence[_CleanBar], sessions: int) -> float | None:
    window = prior[-sessions:]
    if len(window) < sessions:
        return None
    money_flow_volume = 0.0
    volume = 0.0
    for bar in window:
        range_width = bar.high - bar.low
        if range_width <= 0:
            continue
        money_flow_multiplier = ((bar.close - bar.low) - (bar.high - bar.close)) / range_width
        money_flow_volume += money_flow_multiplier * bar.volume
        volume += bar.volume
    if volume <= 0:
        return None
    return money_flow_volume / volume


def _stochastic_oscillator(prior: Sequence[_CleanBar], sessions: int) -> float | None:
    window = prior[-sessions:]
    if len(window) < sessions:
        return None
    high = max(bar.high for bar in window)
    low = min(bar.low for bar in window)
    width = high - low
    if width <= 0:
        return None
    return (_price_basis(window[-1]) - low) / width
