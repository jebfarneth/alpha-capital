"""Derived market-path feature backfill/collector.

This job promotes provider-backed OHLCV path features into a queryable table
for ML and trade-selection analysis. It does not rewrite immutable signal
feature snapshots; every row carries reconstruction metadata and lineage.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable, Sequence

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from alpha.data.contracts import stable_hash
from alpha.data.fmp import FmpAdapter, FmpBar, HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.db.models import MarketPathFeature, SignalRegistry
from alpha.evidence.writer import record_data_lineage
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.market_calendar import (
    next_us_equity_session,
    nth_us_equity_session,
    resolve_us_equity_session,
    us_equity_session_close_timestamp,
)


JOB_NAME = "market_path_feature_collector"
FEATURE_VERSION = "market_path_daily_v3"
RECONSTRUCTION_METHOD = "fmp_eod_replay_v1"
PRICE_BASIS_SPLIT_ADJUSTED_OR_RAW = "split_adjusted_close_when_available_else_raw_close"
PRIOR_52W_SESSION_COUNT = 252
TOUCH_TOLERANCE_PCT = 0.005
DEFAULT_LOOKBACK_CALENDAR_DAYS = 420
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
class _SectorResolution:
    sector: str | None
    sector_etf: str | None
    source: str | None
    status: str
    pit_safe: bool


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
    ) -> None:
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

    @property
    def job_name(self) -> str:
        return JOB_NAME

    @property
    def job_type(self) -> str:
        return "feature_enrichment"

    def run(self, ctx: JobContext) -> JobResult:
        run_ts = _ensure_aware(self._run_timestamp or ctx.started_at)
        session_resolution = resolve_us_equity_session(run_ts)
        through_date = self._through_date or date.fromisoformat(
            session_resolution.evidence_session_date
        )
        signal_start = self._signal_start_date or through_date
        signal_end = self._signal_end_date or through_date
        decision_date = self._decision_date or signal_end
        if signal_start > signal_end:
            return JobResult(
                status="failed",
                errors=[{
                    "stage": "args",
                    "message": "signal_start_date must be on or before signal_end_date",
                }],
            )
        if self._lookback_calendar_days < 70:
            return JobResult(
                status="failed",
                errors=[{
                    "stage": "args",
                    "message": "lookback_calendar_days must be at least 70",
                }],
            )

        signals = self._signals(signal_start, signal_end)
        if not signals:
            return JobResult(
                status="finished",
                metrics={
                    "decision_date": decision_date.isoformat(),
                    "pattern_ids": list(self._pattern_ids),
                    "signal_start_date": signal_start.isoformat(),
                    "signal_end_date": signal_end.isoformat(),
                    "through_date": through_date.isoformat(),
                    "signals_scanned": 0,
                    "rows_inserted": 0,
                    "rows_updated": 0,
                    "no_op_reason": "no_matching_signals",
                },
            )

        rows_inserted = 0
        rows_updated = 0
        rows_skipped = 0
        missing_entry_rows = 0
        fetch_errors: list[dict[str, Any]] = []
        lineages_recorded = 0
        tickers_fetched = 0
        reference_from_date = _fetch_start(signals, self._lookback_calendar_days)
        benchmark_series = self._fetch_reference_series(
            BENCHMARK_SYMBOLS,
            from_date=reference_from_date,
            through_date=through_date,
            run_ts=run_ts,
            job_run_id=ctx.job_run_id,
            source_role="market_benchmark",
        )
        sector_resolver = _SectorResolver(self._session)
        needed_sector_etfs = self._needed_sector_etfs(
            signals,
            through_date=through_date,
            sector_resolver=sector_resolver,
        )
        sector_etf_series = self._fetch_reference_series(
            needed_sector_etfs,
            from_date=reference_from_date,
            through_date=through_date,
            run_ts=run_ts,
            job_run_id=ctx.job_run_id,
            source_role="sector_etf",
        )

        by_ticker = _group_by_ticker(signals)
        for ticker, ticker_signals in by_ticker.items():
            from_date = _fetch_start(ticker_signals, self._lookback_calendar_days)
            resp = self._fmp.get_historical_price(
                ticker,
                from_date=from_date,
                to_date=through_date,
                asof=run_ts,
                adjusted=False,
            )
            tickers_fetched += 1
            if not resp.ok or resp.data is None:
                fetch_errors.append({
                    "ticker": ticker,
                    "stage": "fmp_historical_price",
                    "message": getattr(resp.error, "message", "missing response"),
                    "error_type": getattr(resp.error, "error_type", None),
                })
                continue
            bars = _clean_bars(resp.data)
            lineage = record_data_lineage(
                self._session,
                provider="FMP",
                endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                asof_timestamp=run_ts,
                raw_payload={
                    "ticker": ticker,
                    "from": from_date.isoformat(),
                    "to": through_date.isoformat(),
                    "bars": [_bar_payload(bar) for bar in bars],
                    "feature_version": self._feature_version,
                    "reconstruction_method": RECONSTRUCTION_METHOD,
                },
                source_authority="fmp_eod",
                data_quality_flags={
                    "derived_feature_replay": True,
                    "adapter_raw_payload_hash": resp.lineage.raw_payload_hash,
                },
                job_run_id=ctx.job_run_id,
            )
            lineages_recorded += 1
            existing = self._existing_rows(ticker_signals)
            for signal in ticker_signals:
                result = self._persist_signal_rows(
                    signal,
                    bars=bars,
                    through_date=through_date,
                    data_lineage_id=lineage.data_lineage_id,
                    job_run_id=ctx.job_run_id,
                    existing=existing,
                    benchmark_series=benchmark_series,
                    sector_resolver=sector_resolver,
                    sector_etf_series=sector_etf_series,
                )
                rows_inserted += result["inserted"]
                rows_updated += result["updated"]
                rows_skipped += result["skipped"]
                missing_entry_rows += result["missing_entry"]

        rank_rows_updated = self._populate_cross_sectional_ranks(
            start_date=signal_start,
            through_date=through_date,
        )
        metrics = {
            "decision_date": decision_date.isoformat(),
            "pattern_ids": list(self._pattern_ids),
            "signal_start_date": signal_start.isoformat(),
            "signal_end_date": signal_end.isoformat(),
            "through_date": through_date.isoformat(),
            "feature_version": self._feature_version,
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "signals_scanned": len(signals),
            "ticker_fetch_count": tickers_fetched,
            "lineages_recorded": lineages_recorded,
            "rows_inserted": rows_inserted,
            "rows_updated": rows_updated,
            "rows_skipped": rows_skipped,
            "rank_rows_updated": rank_rows_updated,
            "missing_entry_row_count": missing_entry_rows,
            "fetch_error_count": len(fetch_errors),
            "benchmark_fetch_count": len(BENCHMARK_SYMBOLS),
            "benchmark_fetch_error_count": _reference_error_count(benchmark_series),
            "sector_etf_fetch_count": len(sector_etf_series),
            "sector_etf_fetch_error_count": _reference_error_count(sector_etf_series),
            "liquidity_proxy_min_dollar_volume_20d": self._liquidity_min_dollar_volume_20d,
            "liquidity_proxy_min_price": self._liquidity_min_price,
        }
        status = "partial_failed" if fetch_errors else "finished"
        return JobResult(status=status, metrics=metrics, errors=fetch_errors)

    def _signals(self, start: date, end: date) -> list[SignalRegistry]:
        start_dt = datetime.combine(start, time.min, timezone.utc)
        end_dt = datetime.combine(end + timedelta(days=1), time.min, timezone.utc)
        return (
            self._session.query(SignalRegistry)
            .filter(
                SignalRegistry.pattern_id.in_(self._pattern_ids),
                SignalRegistry.signal_timestamp >= start_dt,
                SignalRegistry.signal_timestamp < end_dt,
            )
            .order_by(SignalRegistry.ticker, SignalRegistry.signal_timestamp)
            .all()
        )

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

    def _fetch_reference_series(
        self,
        symbols: Sequence[str],
        *,
        from_date: date,
        through_date: date,
        run_ts: datetime,
        job_run_id: str,
        source_role: str,
    ) -> dict[str, _ReferenceSeries]:
        series: dict[str, _ReferenceSeries] = {}
        for symbol in sorted({item.upper() for item in symbols if item}):
            resp = self._fmp.get_historical_price(
                symbol,
                from_date=from_date,
                to_date=through_date,
                asof=run_ts,
                adjusted=False,
            )
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
                        "message": getattr(resp.error, "message", "missing response"),
                        "error_type": getattr(resp.error, "error_type", None),
                    },
                )
                continue
            bars = tuple(_clean_bars(resp.data))
            lineage = record_data_lineage(
                self._session,
                provider="FMP",
                endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                asof_timestamp=run_ts,
                raw_payload={
                    "symbol": symbol,
                    "from": from_date.isoformat(),
                    "to": through_date.isoformat(),
                    "bars": [_bar_payload(bar) for bar in bars],
                    "feature_version": self._feature_version,
                    "reconstruction_method": RECONSTRUCTION_METHOD,
                    "source_role": source_role,
                },
                source_authority="fmp_eod",
                data_quality_flags={
                    "derived_feature_replay": True,
                    "reference_series_role": source_role,
                    "adapter_raw_payload_hash": resp.lineage.raw_payload_hash,
                },
                job_run_id=job_run_id,
            )
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
        through_date: date,
        data_lineage_id: str,
        job_run_id: str,
        existing: dict[tuple[str, str], MarketPathFeature],
        benchmark_series: dict[str, _ReferenceSeries],
        sector_resolver: "_SectorResolver",
        sector_etf_series: dict[str, _ReferenceSeries],
    ) -> dict[str, int]:
        by_date = {bar.date: bar for bar in bars}
        entry_date = _entry_date(signal)
        signal_date = signal.signal_timestamp.date()
        start_date = signal_date if self._include_signal_session else entry_date
        if start_date is None:
            return {"inserted": 0, "updated": 0, "skipped": 0, "missing_entry": 1}
        if (
            (entry_date is None or entry_date not in by_date)
            and not self._include_signal_session
        ):
            return {"inserted": 0, "updated": 0, "skipped": 0, "missing_entry": 1}

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
        inserted = updated = skipped = 0
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
            payload = self._feature_payload(
                signal,
                bar=bar,
                bars=bars,
                entry_date=entry_date,
                entry_price=entry_price,
                path_sequence=max(path_sequence, 0),
                feature_role=role,
                batch_through_date=through_date,
                benchmark_series=benchmark_series,
                sector_resolver=sector_resolver,
                sector_etf_series=sector_etf_series,
            )
            key = (signal.signal_id, bar.date.isoformat())
            row = existing.get(key)
            if row is None:
                row = MarketPathFeature()
                self._session.add(row)
                inserted += 1
            else:
                updated += 1
            _assign_row(row, payload, data_lineage_id=data_lineage_id, job_run_id=job_run_id)
        return {"inserted": inserted, "updated": updated, "skipped": skipped, "missing_entry": 0}

    def _feature_payload(
        self,
        signal: SignalRegistry,
        *,
        bar: _CleanBar,
        bars: Sequence[_CleanBar],
        entry_date: date | None,
        entry_price: float | None,
        path_sequence: int,
        feature_role: str,
        batch_through_date: date,
        benchmark_series: dict[str, _ReferenceSeries],
        sector_resolver: "_SectorResolver",
        sector_etf_series: dict[str, _ReferenceSeries],
    ) -> dict[str, Any]:
        signal_date = signal.signal_timestamp.date()
        previous = _previous_bar(bars, bar.date)
        prior = [candidate for candidate in bars if candidate.date < bar.date]
        row_input_bars = [candidate for candidate in bars if candidate.date <= bar.date]
        prior20 = prior[-20:]
        prior60 = prior[-60:]
        close_basis = _price_basis(bar)
        prev_close = _price_basis(previous) if previous is not None else None
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
            and close_basis >= self._liquidity_min_price
        )
        rich_features, rich_status = _rich_eod_features(
            bar=bar,
            previous=previous,
            prior=prior,
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
            session=self._session,
            signal=signal,
            feature_date=bar.date,
            bar=bar,
            prior=prior,
            benchmark_series=benchmark_series,
            entry_price=entry_price,
        )
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
            "dollar_volume": bar.dollar_volume,
            "median_volume_20d": median_volume_20d,
            "median_volume_60d": median_volume_60d,
            "median_dollar_volume_20d": median_dollar_volume_20d,
            "median_dollar_volume_60d": median_dollar_volume_60d,
            "volume_expansion_20d": _safe_ratio(bar.volume, median_volume_20d),
            "volume_expansion_60d": _safe_ratio(bar.volume, median_volume_60d),
            "dollar_volume_expansion_20d": _safe_ratio(bar.dollar_volume, median_dollar_volume_20d),
            "dollar_volume_expansion_60d": _safe_ratio(bar.dollar_volume, median_dollar_volume_60d),
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
            **rich_features,
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
    ) -> int:
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
        for (feature_date, pattern_id, feature_version), group_rows in grouped.items():
            pattern_count = len(group_rows)
            rankable_any = {
                row.signal_id
                for row in group_rows
                if any(_rank_input_value(row, source) is not None for source in RANK_INPUT_TO_OUTPUT)
            }
            feature_count = len(rankable_any)
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
                    setattr(row, rank_field, rank_index)
                    setattr(
                        row,
                        percentile_field,
                        ((value_count - rank_index + 1) / value_count)
                        if value_count else None,
                    )
                ranked_ids = {row.market_path_feature_id for row in ranked}
                for row in group_rows:
                    if row.market_path_feature_id not in ranked_ids:
                        setattr(row, rank_field, None)
                        setattr(row, percentile_field, None)
                rank_status[source_field] = {
                    "rank_field": rank_field,
                    "percentile_field": percentile_field,
                    "rank_direction": "higher_is_better",
                    "population_count": value_count,
                    "population_too_small": value_count < 2,
                }

            for row in group_rows:
                row.cohort_pattern_row_count = pattern_count
                row.cohort_feature_row_count = feature_count
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
                    "dollar_volume_rank": row.dollar_volume_rank,
                    "dollar_volume_percentile": row.dollar_volume_percentile,
                    "volume_expansion_20d_rank": row.volume_expansion_20d_rank,
                    "volume_expansion_20d_percentile": row.volume_expansion_20d_percentile,
                    "volume_expansion_60d_rank": row.volume_expansion_60d_rank,
                    "volume_expansion_60d_percentile": row.volume_expansion_60d_percentile,
                    "dollar_volume_expansion_20d_rank": row.dollar_volume_expansion_20d_rank,
                    "dollar_volume_expansion_20d_percentile": row.dollar_volume_expansion_20d_percentile,
                    "dollar_volume_expansion_60d_rank": row.dollar_volume_expansion_60d_rank,
                    "dollar_volume_expansion_60d_percentile": row.dollar_volume_expansion_60d_percentile,
                    "liquidity_proxy_rank": row.liquidity_proxy_rank,
                    "liquidity_proxy_percentile": row.liquidity_proxy_percentile,
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
                row.feature_json = json.dumps(payload, sort_keys=True)
                _refresh_output_hash(row)
                updated += 1
        return updated


def _assign_row(
    row: MarketPathFeature,
    payload: dict[str, Any],
    *,
    data_lineage_id: str,
    job_run_id: str,
) -> None:
    for key, value in payload.items():
        setattr(row, key, value)
    row.data_lineage_id = data_lineage_id
    row.job_run_id = job_run_id


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


def _clean_bars(bars: Sequence[FmpBar]) -> list[_CleanBar]:
    clean: list[_CleanBar] = []
    for bar in bars:
        try:
            parsed_date = date.fromisoformat(str(bar.date)[:10])
        except ValueError:
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
    session: Session,
    signal: SignalRegistry,
    feature_date: date,
    bar: _CleanBar,
    prior: Sequence[_CleanBar],
    benchmark_series: dict[str, _ReferenceSeries],
    entry_price: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    market_features, market_status = _market_regime_features(
        feature_date=feature_date,
        benchmark_series=benchmark_series,
    )
    intraday_features, intraday_status = _intraday_unavailable_features()
    execution_features, execution_status = _execution_quality_unavailable_features()
    supply_features, supply_status = _supply_squeeze_unavailable_features()
    catalyst_features, catalyst_status = _catalyst_context_features(
        session=session,
        signal=signal,
        feature_date=feature_date,
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


def _intraday_unavailable_features() -> tuple[dict[str, Any], dict[str, Any]]:
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
        "intraday_structure_status": "intraday_adapter_unavailable",
        "missing_intraday_bars": True,
    }
    status = {
        "status": "intraday_adapter_unavailable",
        "missing_intraday_bars": True,
        "values_null_by_design": True,
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


def _supply_squeeze_unavailable_features() -> tuple[dict[str, Any], dict[str, Any]]:
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
        "borrow_fee_status": "borrow_fee_source_unavailable",
        "supply_squeeze_status": "pit_safe_sources_unavailable",
    }
    status = {
        "status": "pit_safe_sources_unavailable",
        "float_source_status": features["float_source_status"],
        "short_source_status": features["short_source_status"],
        "borrow_fee_status": features["borrow_fee_status"],
    }
    return features, status


def _catalyst_context_features(
    *,
    session: Session,
    signal: SignalRegistry,
    feature_date: date,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pattern_strength = _same_day_pattern_strengths(session, signal, feature_date)
    cofire_m2 = any(pattern in pattern_strength for pattern in ("M2", "M2U"))
    strongest_pattern = None
    if pattern_strength:
        strongest_pattern = sorted(
            pattern_strength.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
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
    status = {
        "status": features["catalyst_context_status"],
        "signal_registry_context_available": True,
        "external_catalyst_sources_available": False,
        "patterns_present": sorted(pattern_strength),
        "missing_catalyst_source": True,
    }
    return features, status


def _same_day_pattern_strengths(
    session: Session,
    signal: SignalRegistry,
    feature_date: date,
) -> dict[str, float]:
    signal_day = signal.signal_timestamp.date()
    if feature_date < signal_day:
        return {}
    start_dt = datetime.combine(signal_day, time.min, timezone.utc)
    end_dt = datetime.combine(signal_day + timedelta(days=1), time.min, timezone.utc)
    rows = (
        session.query(SignalRegistry.pattern_id, SignalRegistry.raw_signal_strength)
        .filter(
            SignalRegistry.ticker == signal.ticker,
            SignalRegistry.signal_timestamp >= start_dt,
            SignalRegistry.signal_timestamp < end_dt,
        )
        .all()
    )
    strengths: dict[str, float] = {}
    for pattern_id, strength in rows:
        normalized = str(pattern_id).upper()
        parsed_strength = float(strength or 0.0)
        strengths[normalized] = max(strengths.get(normalized, parsed_strength), parsed_strength)
    return strengths


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
    output_payload = {
        key: getattr(row, key)
        for key in ML_OUTPUT_HASH_FIELDS
    }
    row.output_hash = stable_hash(output_payload)


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
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    prior_ranges_20 = [candidate.high - candidate.low for candidate in prior20]
    vwap = bar.vwap if bar.vwap is not None and bar.vwap > 0 else None
    rich = {
        "prior_52w_high": prior_52w_high,
        "breakout_extension_pct": _safe_return(bar.close, prior_52w_high),
        "open_vs_52w_high_pct": _safe_return(bar.open, prior_52w_high),
        "close_vs_52w_high_pct": _safe_return(bar.close, prior_52w_high),
        "high_vs_52w_high_pct": _safe_return(bar.high, prior_52w_high),
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
            current_range if current_range > 0 else None,
            _median(prior_ranges_20),
        ),
        "volume_zscore_20d": _zscore(bar.volume, [candidate.volume for candidate in prior20]),
        "volume_zscore_60d": _zscore(bar.volume, [candidate.volume for candidate in prior60]),
        "dollar_volume_zscore_20d": _zscore(
            bar.dollar_volume, [candidate.dollar_volume for candidate in prior20]
        ),
        "dollar_volume_zscore_60d": _zscore(
            bar.dollar_volume, [candidate.dollar_volume for candidate in prior60]
        ),
        "volume_acceleration_1d_vs_5d": _safe_return(
            bar.volume, _mean([candidate.volume for candidate in prior[-5:]])
        ),
        "volume_acceleration_1d_vs_20d": _safe_return(
            bar.volume, _mean([candidate.volume for candidate in prior20])
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
        "distance_from_sma_20d": _sma_distance(bar, prior, 20),
        "distance_from_sma_50d": _sma_distance(bar, prior, 50),
        "distance_from_sma_200d": _sma_distance(bar, prior, 200),
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
        "vwap": vwap,
        "open_vs_vwap_pct": _safe_return(bar.open, vwap),
        "high_vs_vwap_pct": _safe_return(bar.high, vwap),
        "low_vs_vwap_pct": _safe_return(bar.low, vwap),
        "close_vs_vwap_pct": _safe_return(bar.close, vwap),
    }
    status = {
        "prior_only_trailing_features": True,
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
