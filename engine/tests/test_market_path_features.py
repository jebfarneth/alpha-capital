from __future__ import annotations

import json
import math
import statistics
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import event, inspect, text

from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError, stable_hash
from alpha.data.fmp import FmpBar, HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.db.models import DataLineage, FeatureSnapshot, MarketPathFeature, SignalRegistry
from alpha.evidence.writer import record_data_lineage, record_feature_snapshot, record_signal
from alpha.jobs.contracts import JobResult
from alpha.jobs.market_path_features import MarketPathFeatureJob
from alpha.jobs.market_path_features import _same_day_pattern_strengths_from_cache
from alpha.jobs.market_path_features import sanitize_provider_error_message
from alpha.jobs.historical_m4_signal_selector import (
    HISTORICAL_M4_REPLAY_RECONSTRUCTION_METHOD,
    SIGNAL_SOURCE_HISTORICAL_M4_REPLAY,
)
from alpha.jobs.run_market_path_backfill import (
    BackfillRunConfig,
    CachedHistoricalPriceFmpAdapter,
    plan_chunks,
    run_backfill_chunks,
)
from alpha.jobs.run_market_path_bulk_backfill import (
    MarketPathBulkBackfillJob,
    MarketPathRankOnlyBackfillJob,
    RetryingHistoricalPriceFmpAdapter,
    _TimeoutRequestsSession,
    _validate_write_target,
    plan_bulk_batches,
    validate_market_path_bulk_backfill,
)
from alpha.jobs.runner import run_job
from alpha.market_calendar import is_us_equity_session


RUN_TS = datetime(2026, 6, 5, 21, 0, tzinfo=timezone.utc)


class FakeFmpAdapter:
    def __init__(
        self,
        bars_by_ticker,
        fail_symbols: set[str] | None = None,
        fail_message: str | None = None,
    ):
        self.bars_by_ticker = bars_by_ticker
        self.fail_symbols = {symbol.upper() for symbol in (fail_symbols or set())}
        self.fail_message = fail_message
        self.calls = []

    def get_historical_price(self, ticker, from_date=None, to_date=None, asof=None, **kwargs):
        ticker = ticker.upper()
        self.calls.append({
            "ticker": ticker,
            "from_date": from_date,
            "to_date": to_date,
            "asof": asof,
            "kwargs": kwargs,
        })
        if ticker in self.fail_symbols:
            return AdapterResponse(
                data=None,
                lineage=LineageMeta(
                    provider="FMP",
                    endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                    request_timestamp=RUN_TS,
                    asof_timestamp=RUN_TS,
                    raw_payload_hash=stable_hash({"ticker": ticker, "error": "forced"}),
                ),
                error=ProviderError(
                    provider="FMP",
                    endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                    status_code=503,
                    error_type="http",
                    message=self.fail_message or f"forced failure for {ticker}",
                    retryable=True,
                ),
            )
        bars = self.bars_by_ticker.get(ticker, [])
        return AdapterResponse(
            data=bars,
            lineage=LineageMeta(
                provider="FMP",
                endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                request_timestamp=RUN_TS,
                asof_timestamp=RUN_TS,
                raw_payload_hash=stable_hash([
                    {
                        "date": bar.date,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                    }
                    for bar in bars
                ]),
            ),
        )


class NoFetchFmpAdapter:
    cache_hits = 0
    cache_misses = 0

    def __init__(self) -> None:
        self.calls = []

    def get_historical_price(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        raise AssertionError("rank-only tests must not fetch historical prices")


def _add_signal(
    db_session,
    *,
    pattern_id: str = "M4",
    ticker: str = "LCUT",
    signal_day: date = date(2026, 6, 2),
    entry_day: date = date(2026, 6, 3),
):
    lineage = record_data_lineage(
        db_session,
        provider="FMP",
        endpoint="/test/signal",
        asof_timestamp=datetime.combine(signal_day, datetime.min.time(), timezone.utc),
        raw_payload={"ticker": ticker, "signal_day": signal_day.isoformat()},
    )
    feature = record_feature_snapshot(
        db_session,
        pattern_id=pattern_id,
        ticker=ticker,
        asof_timestamp=datetime.combine(signal_day, datetime.min.time(), timezone.utc),
        features={"ticker": ticker, "detector_signal_identity_hash": f"{ticker}-{signal_day}"},
        data_lineage_ids=[lineage.data_lineage_id],
        point_in_time_passed=True,
        lookahead_guard_passed=True,
    )
    return record_signal(
        db_session,
        pattern_id=pattern_id,
        ticker=ticker,
        direction="long",
        signal_timestamp=datetime.combine(signal_day, datetime.min.time(), timezone.utc),
        raw_signal_strength=1.0,
        raw_expected_edge=0.01,
        feature_snapshot_id=feature.feature_snapshot_id,
        signal_horizon="15d",
        trading_date=signal_day.isoformat(),
        next_execution_session=entry_day.isoformat(),
        point_in_time_passed=True,
        lookahead_guard_passed=True,
        signal_identity_hash=f"{ticker}-{signal_day}",
        data_lineage_ids=[lineage.data_lineage_id],
    )


def _stamp_signal_historical_m4_replay(db_session, signal_id: str) -> None:
    signal = db_session.get(SignalRegistry, signal_id)
    feature = db_session.get(FeatureSnapshot, signal.feature_snapshot_id)
    payload = json.loads(feature.feature_json)
    payload["reconstruction_method"] = HISTORICAL_M4_REPLAY_RECONSTRUCTION_METHOD
    payload["historical_replay"] = {
        "reconstruction_method": HISTORICAL_M4_REPLAY_RECONSTRUCTION_METHOD,
    }
    feature.feature_json = json.dumps(payload, sort_keys=True)
    db_session.flush()


def _seed_market_path_feature_row(
    db_session,
    signal,
    *,
    ticker: str,
    feature_date: date,
    feature_version: str = "market_path_daily_v3",
    pattern_id: str = "M4",
    feature_role: str = "forward_path_day",
    dollar_volume: float | None = None,
    volume_expansion_20d: float | None = None,
    volume_expansion_60d: float | None = None,
    dollar_volume_expansion_20d: float | None = None,
    dollar_volume_expansion_60d: float | None = None,
    liquidity_proxy_score: float | None = None,
) -> MarketPathFeature:
    lineage = record_data_lineage(
        db_session,
        provider="fixture",
        endpoint="/fixture/market-path",
        asof_timestamp=RUN_TS,
        raw_payload={
            "ticker": ticker,
            "feature_date": feature_date.isoformat(),
            "feature_version": feature_version,
        },
    )
    feature_json = json.dumps({
        "ticker": ticker,
        "feature_session_date": feature_date.isoformat(),
        "feature_version": feature_version,
        "fixture": True,
    }, sort_keys=True)
    input_hash = stable_hash({
        "ticker": ticker,
        "feature_date": feature_date.isoformat(),
        "feature_version": feature_version,
        "input": True,
    })
    row = MarketPathFeature(
        signal_id=signal.signal_id,
        pattern_id=pattern_id,
        ticker=ticker,
        signal_horizon=signal.signal_horizon,
        signal_date=signal.signal_timestamp.date().isoformat(),
        entry_session_date=feature_date.isoformat(),
        feature_session_date=feature_date.isoformat(),
        path_sequence=1,
        feature_role=feature_role,
        feature_version=feature_version,
        asof_timestamp=RUN_TS,
        reconstruction_method="fixture",
        dollar_volume=dollar_volume,
        volume_expansion_20d=volume_expansion_20d,
        volume_expansion_60d=volume_expansion_60d,
        dollar_volume_expansion_20d=dollar_volume_expansion_20d,
        dollar_volume_expansion_60d=dollar_volume_expansion_60d,
        liquidity_proxy_score=liquidity_proxy_score,
        feature_json=feature_json,
        source_provider="fixture",
        source_endpoint="/fixture/market-path",
        data_lineage_id=lineage.data_lineage_id,
        input_hash=input_hash,
        output_hash=stable_hash({
            "ticker": ticker,
            "feature_date": feature_date.isoformat(),
            "feature_version": feature_version,
            "ranked": False,
        }),
    )
    db_session.add(row)
    db_session.flush()
    return row


def _rank_state(row: MarketPathFeature) -> dict[str, object]:
    return {
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
        "cohort_feature_row_count": row.cohort_feature_row_count,
        "cohort_pattern_row_count": row.cohort_pattern_row_count,
    }


def _bars() -> list[FmpBar]:
    start = date(2026, 3, 30)
    bars = []
    day = start
    idx = 0
    while day <= date(2026, 6, 5):
        if day.weekday() < 5:
            close = 10.0 + idx * 0.01
            volume = 100_000 + idx * 1_000
            bars.append(FmpBar(
                date=day.isoformat(),
                open=close - 0.05,
                high=close + 0.25,
                low=close - 0.25,
                close=close,
                volume=volume,
                split_adjusted_close=close,
                adj_close=close,
            ))
            idx += 1
        day += timedelta(days=1)
    # Override the forward sessions with easy-to-audit values.
    overrides = {
        date(2026, 6, 3): (11.0, 11.8, 10.7, 11.5, 600_000),
        date(2026, 6, 4): (11.6, 12.1, 11.2, 11.9, 700_000),
        date(2026, 6, 5): (11.9, 12.0, 11.0, 11.2, 500_000),
    }
    by_date = {date.fromisoformat(bar.date): bar for bar in bars}
    for day, values in overrides.items():
        op, high, low, close, volume = values
        by_date[day] = FmpBar(
            date=day.isoformat(),
            open=op,
            high=high,
            low=low,
            close=close,
            volume=volume,
            split_adjusted_close=close,
            adj_close=close,
        )
    return [by_date[day] for day in sorted(by_date)]


def _rich_bars() -> list[FmpBar]:
    start = date(2025, 4, 1)
    end = date(2026, 6, 5)
    bars: dict[date, FmpBar] = {}
    day = start
    idx = 0
    while day <= end:
        if day.weekday() < 5:
            close = 20.0 + idx * 0.01
            volume = 100_000 + idx * 2_000
            bars[day] = FmpBar(
                date=day.isoformat(),
                open=close - 0.05,
                high=close + 0.20,
                low=close - 0.30,
                close=close,
                volume=volume,
                split_adjusted_close=close,
                adj_close=close,
                vwap=close - 0.02,
            )
            idx += 1
        day += timedelta(days=1)

    bars[date(2026, 5, 29)] = FmpBar(
        date="2026-05-29",
        open=28.0,
        high=30.0,
        low=27.5,
        close=29.0,
        volume=900_000,
        split_adjusted_close=29.0,
        adj_close=29.0,
        vwap=28.75,
    )
    bars[date(2026, 6, 3)] = FmpBar(
        date="2026-06-03",
        open=31.0,
        high=33.0,
        low=30.0,
        close=32.0,
        volume=1_000_000,
        split_adjusted_close=32.0,
        adj_close=32.0,
        vwap=31.5,
    )
    bars[date(2026, 6, 4)] = FmpBar(
        date="2026-06-04",
        open=32.0,
        high=34.0,
        low=31.5,
        close=33.0,
        volume=1_100_000,
        split_adjusted_close=33.0,
        adj_close=33.0,
        vwap=32.75,
    )
    bars[date(2026, 6, 5)] = FmpBar(
        date="2026-06-05",
        open=33.0,
        high=35.0,
        low=32.0,
        close=34.0,
        volume=1_200_000,
        split_adjusted_close=34.0,
        adj_close=34.0,
        vwap=33.5,
    )
    return [bars[day] for day in sorted(bars)]


def _reference_bars(
    *,
    start_price: float = 100.0,
    volume: int = 2_000_000,
    future_spike: bool = False,
) -> list[FmpBar]:
    start = date(2025, 4, 1)
    end = date(2026, 6, 5)
    bars: list[FmpBar] = []
    day = start
    idx = 0
    while day <= end:
        if day.weekday() < 5:
            close = start_price + idx * 0.05
            if future_spike and day == date(2026, 6, 4):
                close = 999.0
            bars.append(FmpBar(
                date=day.isoformat(),
                open=close - 0.05,
                high=close + 0.20,
                low=close - 0.20,
                close=close,
                volume=volume + idx,
                split_adjusted_close=close,
                adj_close=close,
                vwap=close,
            ))
            idx += 1
        day += timedelta(days=1)
    return bars


def _single_entry_bar() -> list[FmpBar]:
    return [
        FmpBar(
            date="2026-06-03",
            open=10.0,
            high=10.5,
            low=9.5,
            close=10.0,
            volume=100_000,
            split_adjusted_close=10.0,
            adj_close=10.0,
            vwap=None,
        )
    ]


def _adx_handcheck_bars() -> list[FmpBar]:
    days = []
    current = date(2026, 6, 3)
    while len(days) < 36:
        if current.weekday() < 5:
            days.append(current)
        current -= timedelta(days=1)
    days = list(reversed(days))
    bars: list[FmpBar] = []
    close = 50.0
    for idx, day in enumerate(days):
        close += (
            (1.4 if idx % 5 in (0, 1, 2) else -1.1)
            + math.sin(idx * 0.7) * 0.4
        )
        bars.append(FmpBar(
            date=day.isoformat(),
            open=close - 0.2,
            high=close + 1.0 + (idx % 3) * 0.3,
            low=close - 0.8 - (idx % 4) * 0.2,
            close=close,
            volume=500_000 + idx * 10_000,
            split_adjusted_close=close,
            adj_close=close,
            vwap=close,
        ))
    return bars


def _adapter_with_references(
    ticker_bars: dict[str, list[FmpBar]],
    *,
    fail_symbols: set[str] | None = None,
):
    payload = {
        "SPY": _reference_bars(start_price=400.0),
        "QQQ": _reference_bars(start_price=350.0),
        "IWM": _reference_bars(start_price=200.0),
    }
    payload.update({ticker.upper(): bars for ticker, bars in ticker_bars.items()})
    return FakeFmpAdapter(payload, fail_symbols=fail_symbols)


def _replace_bar(bars: list[FmpBar], day: date, **updates) -> list[FmpBar]:
    out = []
    for bar in bars:
        if date.fromisoformat(bar.date) != day:
            out.append(bar)
            continue
        values = {
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "split_adjusted_close": bar.split_adjusted_close,
            "adj_close": bar.adj_close,
            "vwap": getattr(bar, "vwap", None),
        }
        values.update(updates)
        out.append(FmpBar(**values))
    return out


def _with_weekend_duplicate(
    bars: list[FmpBar],
    *,
    duplicate_date: date,
    source_date: date,
) -> list[FmpBar]:
    by_date = {date.fromisoformat(bar.date): bar for bar in bars}
    source = by_date[source_date]
    duplicate = FmpBar(
        date=duplicate_date.isoformat(),
        open=source.open,
        high=source.high,
        low=source.low,
        close=source.close,
        volume=source.volume,
        split_adjusted_close=source.split_adjusted_close,
        adj_close=source.adj_close,
        vwap=getattr(source, "vwap", None),
    )
    return sorted([*bars, duplicate], key=lambda bar: bar.date)


def _regular_session_only_bars(bars: list[FmpBar]) -> list[FmpBar]:
    return [
        bar for bar in bars
        if is_us_equity_session(date.fromisoformat(bar.date))
    ]


def _split_adjusted_sigma(bars: list[FmpBar], row_day: date) -> float:
    prior = [
        bar for bar in bars
        if date.fromisoformat(bar.date) < row_day
        and is_us_equity_session(date.fromisoformat(bar.date))
    ][-20:]
    returns = [
        (current.split_adjusted_close or current.close)
        / (previous.split_adjusted_close or previous.close)
        - 1.0
        for previous, current in zip(prior[:-1], prior[1:])
    ]
    return float(statistics.stdev(returns))


def _expected_rsi(bars: list[FmpBar], row_day: date, sessions: int) -> float:
    prior = [
        bar for bar in bars
        if date.fromisoformat(bar.date) < row_day
        and is_us_equity_session(date.fromisoformat(bar.date))
    ]
    window = prior[-(sessions + 1):]
    gains = []
    losses = []
    for previous, current in zip(window[:-1], window[1:]):
        previous_close = previous.split_adjusted_close or previous.close
        current_close = current.split_adjusted_close or current.close
        change = current_close - previous_close
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))
    avg_gain = statistics.mean(gains)
    avg_loss = statistics.mean(losses)
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def test_market_path_feature_job_writes_daily_features(db_session):
    signal = _add_signal(db_session)
    adapter = FakeFmpAdapter({"LCUT": _bars()})
    job = MarketPathFeatureJob(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )

    result = run_job(db_session, job)

    assert result.status == "finished"
    assert result.metrics["signals_scanned"] == 1
    assert result.metrics["rows_inserted"] == 3
    rows = (
        db_session.query(MarketPathFeature)
        .filter(MarketPathFeature.signal_id == signal.signal_id)
        .order_by(MarketPathFeature.feature_session_date)
        .all()
    )
    assert [row.feature_session_date for row in rows] == [
        "2026-06-03",
        "2026-06-04",
        "2026-06-05",
    ]
    entry = rows[0]
    assert entry.open_price == 11.0
    assert entry.close_price == 11.5
    assert entry.dollar_volume == 11.5 * 600_000
    assert entry.median_dollar_volume_20d is not None
    assert entry.volume_expansion_20d is not None
    assert entry.gap_pct is not None
    assert entry.return_from_entry_open == 0.0
    assert entry.return_from_entry_close == (11.5 / 11.0) - 1.0
    assert entry.liquidity_proxy_score == 1.0
    assert entry.data_lineage_id is not None
    feature_json = json.loads(entry.feature_json)
    assert feature_json["lineage_scope"] == "batch_fetch"
    assert feature_json["row_input_window_end"] == "2026-06-03"
    assert feature_json["row_input_hash"]
    assert feature_json["row_input_hash_schema"] == "bars_through_feature_session_v1"
    assert feature_json["batch_lineage_contains_future_rows_for_earlier_feature_dates"] is True
    assert feature_json["batch_lineage_window_end"] == "2026-06-05"
    assert feature_json["sigma_basis"] == "split_adjusted_close_when_available_else_raw_close"
    assert feature_json["stop_basis"] == "split_adjusted_close_when_available_else_raw_close"
    assert feature_json["rich_eod_status"]["missing_vwap"] is True
    assert feature_json["rich_eod_status"]["insufficient_history"]["prior_52w_high"] is True


def test_market_path_skips_provider_non_session_duplicate_bar(db_session):
    signal = _add_signal(db_session, ticker="AHL")
    bars = _with_weekend_duplicate(
        _regular_session_only_bars(_rich_bars()),
        duplicate_date=date(2026, 2, 22),
        source_date=date(2026, 2, 23),
    )
    adapter = FakeFmpAdapter({
        "AHL": bars,
        "SPY": _regular_session_only_bars(_reference_bars(start_price=400.0)),
        "QQQ": _regular_session_only_bars(_reference_bars(start_price=350.0)),
        "IWM": _regular_session_only_bars(_reference_bars(start_price=200.0)),
    })
    job = MarketPathFeatureJob(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )

    result = run_job(db_session, job)

    assert result.status == "finished"
    assert result.metrics["non_session_bars_skipped"] == 1
    assert result.metrics["non_session_bar_skip_sample"] == [
        {"ticker": "AHL", "date": "2026-02-22"}
    ]
    rows = (
        db_session.query(MarketPathFeature)
        .filter(MarketPathFeature.signal_id == signal.signal_id)
        .order_by(MarketPathFeature.feature_session_date)
        .all()
    )
    assert [row.feature_session_date for row in rows] == [
        "2026-06-03",
        "2026-06-04",
        "2026-06-05",
    ]


def test_market_path_historical_m4_signal_source_excludes_unreplayed_live_rows(db_session):
    historical_signal = _add_signal(db_session, ticker="HIST")
    stale_live_signal = _add_signal(db_session, ticker="RKTO")
    _stamp_signal_historical_m4_replay(db_session, historical_signal.signal_id)

    default_job = MarketPathFeatureJob(
        session=db_session,
        fmp_adapter=FakeFmpAdapter({"HIST": _bars(), "RKTO": _bars()}),
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )
    assert {
        signal.ticker
        for signal in default_job._signals(date(2026, 6, 2), date(2026, 6, 2))
    } == {"HIST", "RKTO"}

    historical_job = MarketPathFeatureJob(
        session=db_session,
        fmp_adapter=FakeFmpAdapter({"HIST": _bars(), "RKTO": _bars()}),
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
        signal_source=SIGNAL_SOURCE_HISTORICAL_M4_REPLAY,
    )

    result = run_job(db_session, historical_job)

    assert result.status == "finished"
    assert result.metrics["signal_source"] == SIGNAL_SOURCE_HISTORICAL_M4_REPLAY
    assert result.metrics["signals_scanned"] == 1
    assert (
        db_session.query(MarketPathFeature)
        .filter(MarketPathFeature.signal_id == historical_signal.signal_id)
        .count()
    ) == 3
    assert (
        db_session.query(MarketPathFeature)
        .filter(MarketPathFeature.signal_id == stale_live_signal.signal_id)
        .count()
    ) == 0


def test_market_path_rich_columns_exist_in_metadata(db_session):
    columns = {
        column["name"]
        for column in inspect(db_session.get_bind()).get_columns("market_path_features")
    }

    assert {
        "prior_52w_high",
        "breakout_extension_pct",
        "atr_14_pct",
        "volume_zscore_20d",
        "base_max_drawdown_60d",
        "distance_from_sma_200d",
        "failed_breakout_count_126d",
        "vwap",
        "close_vs_vwap_pct",
        "dollar_volume_rank",
        "volume_expansion_20d_percentile",
        "cohort_pattern_row_count",
        "spy_return_1d",
        "spy_return_20d",
        "qqq_return_1d",
        "iwm_return_1d",
        "relative_strength_vs_iwm_60d",
        "sector_etf",
        "relative_strength_vs_sector_20d",
        "sector_relative_status",
        "universe_pct_above_sma_20d",
        "volatility_regime_proxy",
        "opening_range_high_5m",
        "intraday_structure_status",
        "bid_ask_spread",
        "execution_quality_status",
        "float_shares",
        "supply_squeeze_status",
        "news_count_20d",
        "cofire_m2",
        "cross_pattern_overlap_count",
        "rsi_14",
        "adx_14",
        "macd_histogram",
        "technical_indicator_status",
    }.issubset(columns)


def test_market_path_rich_eod_breakout_and_candle_math(db_session):
    signal = _add_signal(db_session)
    adapter = FakeFmpAdapter({"LCUT": _rich_bars()})
    job = MarketPathFeatureJob(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )

    result = run_job(db_session, job)

    assert result.status == "finished"
    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert row.prior_52w_high == 30.0
    assert row.breakout_extension_pct == pytest.approx((32.0 / 30.0) - 1.0)
    assert row.open_vs_52w_high_pct == pytest.approx((31.0 / 30.0) - 1.0)
    assert row.close_vs_52w_high_pct == pytest.approx((32.0 / 30.0) - 1.0)
    assert row.high_vs_52w_high_pct == pytest.approx((33.0 / 30.0) - 1.0)
    assert row.gap_over_breakout is True
    assert row.closed_above_breakout is True
    assert row.close_location_value == pytest.approx((32.0 - 30.0) / 3.0)
    assert row.upper_wick_ratio == pytest.approx((33.0 - 32.0) / 3.0)
    assert row.lower_wick_ratio == pytest.approx((31.0 - 30.0) / 3.0)
    assert row.vwap == 31.5
    assert row.open_vs_vwap_pct == pytest.approx((31.0 / 31.5) - 1.0)
    assert row.close_vs_vwap_pct == pytest.approx((32.0 / 31.5) - 1.0)
    rich_json = json.loads(row.feature_json)
    assert rich_json["rich_eod_features"]["prior_52w_high"] == 30.0
    assert rich_json["rich_eod_status"]["missing_vwap"] is False
    assert rich_json["rich_eod_status"]["atr_basis"] == "prior_14_completed_sessions"


def test_market_path_prior_52w_high_ignores_future_bars(db_session):
    signal = _add_signal(db_session)
    baseline_bars = _rich_bars()
    future_spike = _replace_bar(baseline_bars, date(2026, 6, 4), high=999.0)
    adapter = FakeFmpAdapter({"LCUT": future_spike})
    job = MarketPathFeatureJob(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )

    run_job(db_session, job)

    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert row.prior_52w_high == 30.0
    assert row.high_vs_52w_high_pct == pytest.approx((33.0 / 30.0) - 1.0)


def test_market_path_trailing_rich_features_ignore_future_bars(db_session):
    signal = _add_signal(db_session)
    kwargs = dict(
        session=db_session,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )
    baseline = FakeFmpAdapter({"LCUT": _rich_bars()})
    run_job(db_session, MarketPathFeatureJob(fmp_adapter=baseline, **kwargs))
    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    baseline_values = {
        "atr_14_pct": row.atr_14_pct,
        "volume_zscore_20d": row.volume_zscore_20d,
        "distance_from_sma_200d": row.distance_from_sma_200d,
        "prior_52w_high_touches_60d": row.prior_52w_high_touches_60d,
        "failed_breakout_count_60d": row.failed_breakout_count_60d,
    }

    future_spike = _replace_bar(
        _replace_bar(_rich_bars(), date(2026, 6, 4), high=999.0, close=998.0, volume=9_000_000),
        date(2026, 6, 5),
        high=999.0,
        close=997.0,
        volume=8_000_000,
    )
    db_session.expire_all()
    run_job(db_session, MarketPathFeatureJob(fmp_adapter=FakeFmpAdapter({"LCUT": future_spike}), **kwargs))
    updated = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )

    assert updated.atr_14_pct == baseline_values["atr_14_pct"]
    assert updated.volume_zscore_20d == baseline_values["volume_zscore_20d"]
    assert updated.distance_from_sma_200d == baseline_values["distance_from_sma_200d"]
    assert updated.prior_52w_high_touches_60d == baseline_values["prior_52w_high_touches_60d"]
    assert updated.failed_breakout_count_60d == baseline_values["failed_breakout_count_60d"]


def test_market_path_zero_range_candle_guards(db_session):
    signal = _add_signal(db_session)
    bars = _replace_bar(_rich_bars(), date(2026, 6, 3), open=31.0, high=31.0, low=31.0, close=31.0)
    adapter = FakeFmpAdapter({"LCUT": bars})
    job = MarketPathFeatureJob(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )

    run_job(db_session, job)

    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert row.close_location_value is None
    assert row.upper_wick_ratio is None
    assert row.lower_wick_ratio is None


def test_market_path_zscore_zero_std_guard(db_session):
    signal = _add_signal(db_session)
    bars = []
    for bar in _rich_bars():
        day = date.fromisoformat(bar.date)
        if day < date(2026, 6, 3):
            bars.append(FmpBar(
                date=bar.date,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=100_000,
                split_adjusted_close=bar.split_adjusted_close,
                adj_close=bar.adj_close,
                vwap=bar.vwap,
            ))
        else:
            bars.append(bar)
    adapter = FakeFmpAdapter({"LCUT": bars})
    job = MarketPathFeatureJob(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )

    run_job(db_session, job)

    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert row.volume_zscore_20d is None
    assert row.volume_zscore_60d is None


def test_market_path_insufficient_history_and_missing_vwap_status(db_session):
    signal = _add_signal(db_session)
    adapter = FakeFmpAdapter({"LCUT": _bars()})
    job = MarketPathFeatureJob(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )

    run_job(db_session, job)

    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    status = json.loads(row.feature_json)["rich_eod_status"]
    assert row.prior_52w_high is None
    assert row.distance_from_sma_200d is None
    assert row.vwap is None
    assert status["insufficient_history"]["prior_52w_high"] is True
    assert status["insufficient_history"]["distance_from_sma_200d"] is True
    assert status["missing_vwap"] is True


def test_market_path_feature_job_is_idempotent(db_session):
    signal = _add_signal(db_session)
    adapter = FakeFmpAdapter({"LCUT": _bars()})
    kwargs = dict(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )

    first = run_job(db_session, MarketPathFeatureJob(**kwargs))
    second = run_job(db_session, MarketPathFeatureJob(**kwargs))

    assert first.metrics["rows_inserted"] == 3
    assert second.metrics["rows_inserted"] == 0
    assert second.metrics["rows_updated"] == 3
    assert (
        db_session.query(MarketPathFeature)
        .filter(MarketPathFeature.signal_id == signal.signal_id)
        .count()
    ) == 3


def test_market_path_backfill_planner_splits_by_pattern_and_date():
    chunks = plan_chunks(
        ["M4", "M1", "M4"],
        signal_start_date=date(2026, 6, 1),
        signal_end_date=date(2026, 6, 3),
        chunk_days=1,
    )

    assert [
        (
            chunk.pattern_id,
            chunk.signal_start_date.isoformat(),
            chunk.signal_end_date.isoformat(),
        )
        for chunk in chunks
    ] == [
        ("M4", "2026-06-01", "2026-06-01"),
        ("M4", "2026-06-02", "2026-06-02"),
        ("M4", "2026-06-03", "2026-06-03"),
        ("M1", "2026-06-01", "2026-06-01"),
        ("M1", "2026-06-02", "2026-06-02"),
        ("M1", "2026-06-03", "2026-06-03"),
    ]

    two_day_chunks = plan_chunks(
        ["M2"],
        signal_start_date=date(2026, 6, 1),
        signal_end_date=date(2026, 6, 5),
        chunk_days=2,
    )
    assert [
        (chunk.signal_start_date.isoformat(), chunk.signal_end_date.isoformat())
        for chunk in two_day_chunks
    ] == [
        ("2026-06-01", "2026-06-02"),
        ("2026-06-03", "2026-06-04"),
        ("2026-06-05", "2026-06-05"),
    ]


class _FakeBackfillSession:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _NoCloseSession:
    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        pass


def test_market_path_backfill_runner_calls_collector_one_date_windows(tmp_path):
    chunks = plan_chunks(
        ["M4", "M1"],
        signal_start_date=date(2026, 6, 1),
        signal_end_date=date(2026, 6, 1),
    )
    calls = []
    sessions = []

    class CapturingJob:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    def session_factory():
        session = _FakeBackfillSession()
        sessions.append(session)
        return session

    def fake_runner(session, job, params):
        return JobResult(
            status="finished",
            metrics={"rows_inserted": 1, "rows_updated": 2, "fetch_error_count": 0},
        )

    summary = run_backfill_chunks(
        chunks,
        session_factory=session_factory,
        fmp_adapter=object(),
        config=BackfillRunConfig(
            through_date=date(2026, 6, 5),
            run_timestamp=RUN_TS,
            include_signal_session=True,
        ),
        artifact_path=tmp_path / "backfill.json",
        job_factory=CapturingJob,
        job_runner=fake_runner,
        print_fn=lambda _: None,
    )

    assert summary["chunks_finished"] == 2
    assert [call["pattern_ids"] for call in calls] == [("M4",), ("M1",)]
    assert [call["signal_start_date"] for call in calls] == [
        date(2026, 6, 1),
        date(2026, 6, 1),
    ]
    assert [call["signal_end_date"] for call in calls] == [
        date(2026, 6, 1),
        date(2026, 6, 1),
    ]
    assert all(call["through_date"] == date(2026, 6, 5) for call in calls)
    assert all(call["include_signal_session"] is True for call in calls)
    assert all(session.closed for session in sessions)
    artifact = json.loads((tmp_path / "backfill.json").read_text())
    assert artifact["summary"]["rows_inserted_total"] == 2
    assert artifact["summary"]["rows_updated_total"] == 4


def test_market_path_backfill_runner_stops_on_failed_chunk(tmp_path):
    chunks = plan_chunks(
        ["M4"],
        signal_start_date=date(2026, 6, 1),
        signal_end_date=date(2026, 6, 3),
    )
    calls = []

    class CapturingJob:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    def fake_runner(session, job, params):
        if len(calls) == 1:
            return JobResult(
                status="finished",
                metrics={"rows_inserted": 1, "rows_updated": 0, "fetch_error_count": 0},
            )
        return JobResult(
            status="partial_failed",
            metrics={"rows_inserted": 0, "rows_updated": 0, "fetch_error_count": 1},
            errors=[{"ticker": "FAIL"}],
        )

    summary = run_backfill_chunks(
        chunks,
        session_factory=_FakeBackfillSession,
        fmp_adapter=object(),
        config=BackfillRunConfig(through_date=date(2026, 6, 5)),
        artifact_path=tmp_path / "backfill_failed.json",
        job_factory=CapturingJob,
        job_runner=fake_runner,
        print_fn=lambda _: None,
    )

    assert len(calls) == 2
    assert summary["chunks_finished"] == 1
    assert summary["chunks_failed"] == 1
    assert summary["failed_chunk_start"] == "2026-06-02"
    artifact = json.loads((tmp_path / "backfill_failed.json").read_text())
    assert artifact["chunks"][1]["status"] == "partial_failed"
    assert artifact["chunks"][1]["rc"] == 1


def test_market_path_backfill_runner_rerun_uses_existing_unique_key(db_session, tmp_path):
    signal = _add_signal(db_session)
    adapter = _adapter_with_references({"LCUT": _rich_bars()})
    chunks = plan_chunks(
        ["M4"],
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
    )
    config = BackfillRunConfig(
        through_date=date(2026, 6, 3),
        run_timestamp=RUN_TS,
    )
    session_factory = lambda: _NoCloseSession(db_session)

    first = run_backfill_chunks(
        chunks,
        session_factory=session_factory,
        fmp_adapter=adapter,
        config=config,
        artifact_path=tmp_path / "first.json",
        print_fn=lambda _: None,
    )
    second = run_backfill_chunks(
        chunks,
        session_factory=session_factory,
        fmp_adapter=adapter,
        config=config,
        artifact_path=tmp_path / "second.json",
        print_fn=lambda _: None,
    )

    assert first["rows_inserted_total"] == 1
    assert second["rows_inserted_total"] == 0
    assert second["rows_updated_total"] == 1
    duplicate_groups = db_session.execute(text(
        "SELECT COUNT(*) FROM ("
        "SELECT signal_id, feature_session_date, feature_version, COUNT(*) "
        "FROM market_path_features "
        "GROUP BY signal_id, feature_session_date, feature_version "
        "HAVING COUNT(*) > 1"
        ") d"
    )).scalar()
    assert duplicate_groups == 0
    assert (
        db_session.query(MarketPathFeature)
        .filter(MarketPathFeature.signal_id == signal.signal_id)
        .count()
    ) == 1


def test_market_path_backfill_fmp_cache_reuses_superset_requests():
    adapter = _adapter_with_references({"LCUT": _rich_bars()})
    cached = CachedHistoricalPriceFmpAdapter(adapter)

    first = cached.get_historical_price(
        "LCUT",
        from_date=date(2025, 4, 1),
        to_date=date(2026, 6, 5),
        asof=RUN_TS,
        adjusted=False,
    )
    second = cached.get_historical_price(
        "LCUT",
        from_date=date(2026, 6, 1),
        to_date=date(2026, 6, 3),
        asof=RUN_TS,
        adjusted=False,
    )
    different_asof = cached.get_historical_price(
        "LCUT",
        from_date=date(2026, 6, 1),
        to_date=date(2026, 6, 3),
        asof=RUN_TS + timedelta(seconds=1),
        adjusted=False,
    )

    assert first.ok
    assert second.ok
    assert different_asof.ok
    assert len(adapter.calls) == 2
    assert cached.cache_hits == 1
    assert cached.cache_misses == 2
    assert [bar.date for bar in second.data] == ["2026-06-01", "2026-06-02", "2026-06-03"]
    assert second.lineage.data_quality_flags["market_path_backfill_cache_hit"] is True
    assert second.lineage.raw_payload_hash != first.lineage.raw_payload_hash
    assert different_asof.lineage.data_quality_flags is None


def test_market_path_bulk_retry_wrapper_retries_retryable_historical_fetch():
    class TransientAdapter:
        def __init__(self):
            self.calls = 0

        def get_historical_price(self, ticker, from_date=None, to_date=None, asof=None, **kwargs):
            self.calls += 1
            lineage = LineageMeta(
                provider="FMP",
                endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                request_timestamp=RUN_TS,
                asof_timestamp=asof or RUN_TS,
                raw_payload_hash=stable_hash({"ticker": ticker, "call": self.calls}),
            )
            if self.calls == 1:
                return AdapterResponse(
                    data=None,
                    lineage=lineage,
                    error=ProviderError(
                        provider="FMP",
                        endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                        status_code=503,
                        error_type="http",
                        message="transient",
                        retryable=True,
                    ),
                )
            return AdapterResponse(data=_bars(), lineage=lineage)

    adapter = TransientAdapter()
    retrying = RetryingHistoricalPriceFmpAdapter(
        adapter,
        max_retries=2,
        retry_sleep_seconds=0.0,
    )

    response = retrying.get_historical_price(
        "LCUT",
        from_date=date(2026, 6, 1),
        to_date=date(2026, 6, 3),
        asof=RUN_TS,
        adjusted=False,
    )

    assert response.ok
    assert adapter.calls == 2
    flags = response.lineage.data_quality_flags
    assert flags["market_path_bulk_retry_attempt_count"] == 2
    assert flags["market_path_bulk_retry_exhausted"] is False
    assert flags["market_path_bulk_retry_attempts"][0]["error_type"] == "http"


@pytest.mark.parametrize(
    "message",
    [
        "GET https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=ABC&apikey=SECRET failed",
        "GET https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=ABC&api_key=SECRET&from=2026-01-01 failed",
        "GET https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=ABC&token=SECRET&to=2026-01-02 failed",
        "Max retries exceeded with url: /stable/historical-price-eod/full?symbol=ABC&apikey=SECRET (Caused by NewConnectionError('https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=ABC&api_key=SECRET'))",
        "Authorization: Bearer SECRET failed for https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=ABC&token=SECRET&apikey=ALSOSECRET",
    ],
)
def test_market_path_bulk_sanitizes_provider_error_messages(message):
    sanitized = sanitize_provider_error_message(message)
    dumped = json.dumps({"message": sanitized})

    assert "SECRET" not in dumped
    assert "ALSOSECRET" not in dumped
    assert "apikey" not in dumped.lower()
    assert "api_key" not in dumped.lower()
    assert "token" not in dumped.lower()
    assert "authorization" not in dumped.lower()
    assert "?" not in sanitized
    assert "/stable/historical-price-eod/full" in sanitized


def test_market_path_bulk_retry_lineage_flags_are_sanitized_on_exhaustion():
    leaky_message = (
        "HTTPSConnectionPool(host='financialmodelingprep.com', port=443): "
        "Max retries exceeded with url: "
        "/stable/historical-price-eod/full?symbol=LCUT&from=2026-06-01"
        "&apikey=SECRET&token=ALSOSECRET"
    )

    class LeakyFailureAdapter:
        def get_historical_price(self, ticker, from_date=None, to_date=None, asof=None, **kwargs):
            return AdapterResponse(
                data=None,
                lineage=LineageMeta(
                    provider="FMP",
                    endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                    request_timestamp=RUN_TS,
                    asof_timestamp=asof or RUN_TS,
                    raw_payload_hash=stable_hash({"ticker": ticker, "failed": True}),
                    data_quality_flags={
                        "upstream_message": leaky_message,
                    },
                ),
                error=ProviderError(
                    provider="FMP",
                    endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                    status_code=None,
                    error_type="http",
                    message=leaky_message,
                    retryable=True,
                ),
            )

    retrying = RetryingHistoricalPriceFmpAdapter(
        LeakyFailureAdapter(),
        max_retries=1,
        retry_sleep_seconds=0.0,
        request_timeout_seconds=4.0,
    )

    response = retrying.get_historical_price(
        "LCUT",
        from_date=date(2026, 6, 1),
        to_date=date(2026, 6, 3),
        asof=RUN_TS,
        adjusted=False,
    )
    dumped = json.dumps(response.lineage.data_quality_flags, sort_keys=True)

    assert not response.ok
    assert response.error.message != leaky_message
    assert response.lineage.data_quality_flags["market_path_bulk_retry_attempt_count"] == 2
    assert response.lineage.data_quality_flags["market_path_bulk_retry_max_retries"] == 1
    assert response.lineage.data_quality_flags["market_path_bulk_retry_exhausted"] is True
    assert response.lineage.data_quality_flags["market_path_bulk_request_timeout_seconds"] == 4.0
    assert "SECRET" not in dumped
    assert "ALSOSECRET" not in dumped
    assert "apikey" not in dumped.lower()
    assert "api_key" not in dumped.lower()
    assert "token" not in dumped.lower()
    assert "authorization" not in dumped.lower()
    assert "?" not in dumped


def test_market_path_bulk_timeout_session_overrides_adapter_default_timeout(monkeypatch):
    captured = {}

    def fake_request(self, method, url, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        response = type("Response", (), {})()
        response.status_code = 200
        response.headers = {}
        response.text = "[]"
        response.json = lambda: []
        return response

    monkeypatch.setattr("requests.sessions.Session.request", fake_request)
    session = _TimeoutRequestsSession(7.5)
    session.get("https://example.invalid/test", timeout=30)

    assert captured["timeout"] == 7.5


def test_market_path_backfill_persists_cache_hit_lineage_flag(db_session, tmp_path):
    _add_signal(
        db_session,
        ticker="LCUT",
        signal_day=date(2026, 6, 1),
        entry_day=date(2026, 6, 2),
    )
    _add_signal(
        db_session,
        ticker="LCUT",
        signal_day=date(2026, 6, 2),
        entry_day=date(2026, 6, 3),
    )
    adapter = CachedHistoricalPriceFmpAdapter(_adapter_with_references({"LCUT": _rich_bars()}))
    chunks = plan_chunks(
        ["M4"],
        signal_start_date=date(2026, 6, 1),
        signal_end_date=date(2026, 6, 2),
    )
    summary = run_backfill_chunks(
        chunks,
        session_factory=lambda: _NoCloseSession(db_session),
        fmp_adapter=adapter,
        config=BackfillRunConfig(
            through_date=date(2026, 6, 3),
            run_timestamp=RUN_TS,
        ),
        artifact_path=tmp_path / "cache_lineage.json",
        print_fn=lambda _: None,
    )

    assert summary["rows_inserted_total"] == 3
    cached_row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_date == "2026-06-02",
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    lineage = db_session.get(DataLineage, cached_row.data_lineage_id)
    flags = json.loads(lineage.data_quality_flags)
    assert flags["market_path_backfill_cache_hit"] is True
    assert flags["derived_feature_replay"] is True
    assert flags["lineage_payload_schema"] == "compact_bar_digest_v1"
    assert flags["adapter_raw_payload_hash"]


def test_market_path_bulk_backfill_refuses_unconfirmed_public_write():
    with pytest.raises(ValueError, match="confirm-live-write"):
        _validate_write_target(schema=None, confirm_live_write=False)

    _validate_write_target(schema="scratch_market_path", confirm_live_write=False)
    _validate_write_target(schema=None, confirm_live_write=True)


def test_market_path_bulk_backfill_plans_pattern_date_batches():
    batches = plan_bulk_batches(
        ["M4", "M1"],
        signal_start_date=date(2026, 6, 1),
        signal_end_date=date(2026, 6, 5),
        batch_days=2,
    )

    assert [
        (batch.pattern_id, batch.signal_start_date.isoformat(), batch.signal_end_date.isoformat())
        for batch in batches
    ] == [
        ("M4", "2026-06-01", "2026-06-02"),
        ("M4", "2026-06-03", "2026-06-04"),
        ("M4", "2026-06-05", "2026-06-05"),
        ("M1", "2026-06-01", "2026-06-02"),
        ("M1", "2026-06-03", "2026-06-04"),
        ("M1", "2026-06-05", "2026-06-05"),
    ]


def test_market_path_bulk_validation_queries_are_feature_session_range_scoped():
    class _Result:
        def __init__(self, *, scalar_value=0, row=None):
            self._scalar_value = scalar_value
            self._row = row

        def scalar(self):
            return self._scalar_value

        def mappings(self):
            return self

        def one(self):
            return self._row

    class _Session:
        def __init__(self):
            self.calls = []

        def execute(self, statement, params=None):
            sql = str(statement)
            self.calls.append((sql, dict(params or {})))
            if "COUNT(*) AS scoped_feature_rows" in sql:
                return _Result(row={
                    "scoped_feature_rows": 0,
                    "missing_lineage_hash_rows": 0,
                    "prior_52w_high_rows": 0,
                    "rank_populated_rows": 0,
                    "pre_entry_leakage_rows": 0,
                })
            return _Result(scalar_value=0)

    session = _Session()

    metrics = validate_market_path_bulk_backfill(
        session,
        pattern_ids=("M4",),
        signal_start_date=date(2024, 1, 1),
        signal_end_date=date(2025, 12, 31),
        through_date=date(2026, 1, 31),
        feature_version="market_path_daily_v3",
    )

    assert metrics["duplicate_groups"] == 0
    assert len(session.calls) == 3
    for sql, params in session.calls:
        assert "FROM market_path_features" in sql
        assert "feature_session_date >= :feature_start" in sql
        assert "feature_session_date <= :feature_through" in sql
        assert params["feature_start"] == "2024-01-01"
        assert params["feature_through"] == "2026-01-31"
    duplicate_sql = session.calls[0][0]
    assert "GROUP BY signal_id, feature_session_date, feature_version" in duplicate_sql


def test_market_path_bulk_backfill_stage_merge_idempotent_and_deferred_rank(
    db_session,
    tmp_path,
):
    _add_signal(
        db_session,
        ticker="AAAA",
        signal_day=date(2026, 6, 1),
        entry_day=date(2026, 6, 2),
    )
    _add_signal(
        db_session,
        ticker="BBBB",
        signal_day=date(2026, 6, 1),
        entry_day=date(2026, 6, 2),
    )
    _add_signal(
        db_session,
        ticker="CCCC",
        signal_day=date(2026, 6, 2),
        entry_day=date(2026, 6, 3),
    )
    _add_signal(
        db_session,
        ticker="DDDD",
        signal_day=date(2026, 6, 2),
        entry_day=date(2026, 6, 3),
    )
    adapter = CachedHistoricalPriceFmpAdapter(_adapter_with_references({
        "AAAA": _replace_bar(_rich_bars(), date(2026, 6, 3), close=40.0, volume=1_000_000),
        "BBBB": _replace_bar(_rich_bars(), date(2026, 6, 3), close=20.0, volume=500_000),
        "CCCC": _replace_bar(_rich_bars(), date(2026, 6, 3), close=30.0, volume=800_000),
        "DDDD": _replace_bar(_rich_bars(), date(2026, 6, 3), close=10.0, volume=300_000),
    }))

    def bulk_job(fmp_adapter, artifact_name):
        return MarketPathBulkBackfillJob(
            session=db_session,
            fmp_adapter=fmp_adapter,
            pattern_ids=("M4",),
            signal_start_date=date(2026, 6, 1),
            signal_end_date=date(2026, 6, 2),
            through_date=date(2026, 6, 3),
            run_timestamp=RUN_TS,
            batch_days=1,
            include_signal_session=True,
            progress_artifact=tmp_path / artifact_name,
            schema="scratch_test",
        )

    first = run_job(db_session, bulk_job(adapter, "bulk_first.json"))
    assert first.status == "finished"
    assert first.metrics["batch_count"] == 2
    assert first.metrics["rank_pass_count"] == 1
    assert first.metrics["ticker_fetch_started_count"] == 4
    assert first.metrics["ticker_fetch_finished_count"] == 4
    assert first.metrics["ticker_fetch_error_count"] == 0
    assert first.metrics["scoped_feature_row_count"] == 10
    assert first.metrics["duplicate_groups"] == 0
    assert first.metrics["pre_entry_leakage_count"] == 0
    assert first.metrics["missing_lineage_hash_count"] == 0
    assert first.metrics["rank_populated_count"] == 10
    assert first.metrics["rows_merged"] == 10
    assert adapter.cache_hits > 0
    first_artifact = json.loads((tmp_path / "bulk_first.json").read_text())
    first_batch_events = {
        event["event"]
        for event in first_artifact["batches"][0]["progress_events"]
    }
    assert {
        "batch_start",
        "signal_load_start",
        "signal_load_finish",
        "reference_fetch_start",
        "reference_fetch_finish",
        "tickers_planned",
        "ticker_fetch_start",
        "ticker_fetch_finish",
        "feature_rows_generated",
        "collect_finish",
        "staging_write_start",
        "stage_load_start",
        "stage_load_finish",
        "merge_upsert_start",
        "merge_upsert_finish",
        "staging_write_finish",
        "batch_finish",
        "batch_artifact_written",
    } <= first_batch_events
    run_events = {event["event"] for event in first_artifact["progress_events"]}
    assert {
        "rank_pass_start",
        "rank_group_progress",
        "rank_pass_finish",
        "validation_start",
        "validation_finish",
    } <= run_events
    ordered_run_events = [event["event"] for event in first_artifact["progress_events"]]
    assert ordered_run_events.index("rank_group_progress") < ordered_run_events.index("rank_pass_finish")
    rank_events = [
        event for event in first_artifact["progress_events"]
        if event["event"] == "rank_group_progress"
    ]
    assert rank_events[-1]["rank_group_processed"] == rank_events[-1]["rank_group_total"]
    assert rank_events[-1]["feature_session_date"]
    assert rank_events[-1]["pattern_id"] == "M4"
    assert rank_events[-1]["feature_version"] == "market_path_daily_v3"
    assert rank_events[-1]["elapsed_seconds"] >= 0
    first_hashes = {
        (row.signal_id, row.feature_session_date): (
            row.output_hash,
            row.feature_json,
        )
        for row in db_session.query(MarketPathFeature).all()
    }

    second = run_job(db_session, bulk_job(adapter, "bulk_second.json"))
    assert second.status == "finished"
    assert second.metrics["rows_inserted"] == 0
    assert second.metrics["rows_updated"] == 0
    assert second.metrics["rows_unchanged"] == 10
    assert second.metrics["rows_merged"] == 0
    assert second.metrics["rank_rows_updated"] == 0
    assert second.metrics["duplicate_groups"] == 0
    assert second.metrics["pre_entry_leakage_count"] == 0
    assert {
        (row.signal_id, row.feature_session_date): (
            row.output_hash,
            row.feature_json,
        )
        for row in db_session.query(MarketPathFeature).all()
    } == first_hashes

    changed_adapter = CachedHistoricalPriceFmpAdapter(_adapter_with_references({
        "AAAA": _replace_bar(_rich_bars(), date(2026, 6, 3), close=40.0, volume=1_000_000),
        "BBBB": _replace_bar(_rich_bars(), date(2026, 6, 3), close=95.0, volume=3_000_000),
        "CCCC": _replace_bar(_rich_bars(), date(2026, 6, 3), close=30.0, volume=800_000),
        "DDDD": _replace_bar(_rich_bars(), date(2026, 6, 3), close=10.0, volume=300_000),
    }))
    material = run_job(db_session, bulk_job(changed_adapter, "bulk_material.json"))
    assert material.status == "finished"
    assert material.metrics["rows_updated"] > 0
    assert material.metrics["rows_merged"] > 0
    assert material.metrics["rank_rows_updated"] > 0
    assert any(
        first_hashes[key][0] != row.output_hash
        for row in db_session.query(MarketPathFeature).all()
        if (key := (row.signal_id, row.feature_session_date)) in first_hashes
    )


def test_market_path_rank_only_job_populates_existing_rows_without_fetching_and_is_idempotent(
    db_session,
    tmp_path,
):
    jan_a = _add_signal(db_session, ticker="RJAA", signal_day=date(2026, 1, 14), entry_day=date(2026, 1, 15))
    jan_b = _add_signal(db_session, ticker="RJBB", signal_day=date(2026, 1, 14), entry_day=date(2026, 1, 15))
    feb_a = _add_signal(db_session, ticker="RFAA", signal_day=date(2026, 2, 2), entry_day=date(2026, 2, 3))
    feb_b = _add_signal(db_session, ticker="RFBB", signal_day=date(2026, 2, 2), entry_day=date(2026, 2, 3))
    outside_date = _add_signal(db_session, ticker="ROUT", signal_day=date(2025, 12, 30), entry_day=date(2025, 12, 31))
    other_pattern = _add_signal(db_session, pattern_id="M1", ticker="RONE", signal_day=date(2026, 1, 14), entry_day=date(2026, 1, 15))
    other_version = _add_signal(db_session, ticker="RVVV", signal_day=date(2026, 1, 14), entry_day=date(2026, 1, 15))

    target_rows = [
        _seed_market_path_feature_row(
            db_session,
            jan_a,
            ticker="RJAA",
            feature_date=date(2026, 1, 15),
            dollar_volume=1_000_000.0,
            volume_expansion_20d=5.0,
            volume_expansion_60d=4.0,
            dollar_volume_expansion_20d=3.0,
            dollar_volume_expansion_60d=2.0,
            liquidity_proxy_score=1.0,
        ),
        _seed_market_path_feature_row(
            db_session,
            jan_b,
            ticker="RJBB",
            feature_date=date(2026, 1, 15),
            dollar_volume=500_000.0,
            volume_expansion_20d=2.0,
            volume_expansion_60d=1.0,
            dollar_volume_expansion_20d=1.5,
            dollar_volume_expansion_60d=1.2,
            liquidity_proxy_score=0.5,
        ),
        _seed_market_path_feature_row(
            db_session,
            feb_a,
            ticker="RFAA",
            feature_date=date(2026, 2, 3),
            dollar_volume=200_000.0,
            volume_expansion_20d=1.0,
            volume_expansion_60d=1.0,
            dollar_volume_expansion_20d=1.0,
            dollar_volume_expansion_60d=1.0,
            liquidity_proxy_score=0.4,
        ),
        _seed_market_path_feature_row(
            db_session,
            feb_b,
            ticker="RFBB",
            feature_date=date(2026, 2, 3),
            dollar_volume=900_000.0,
            volume_expansion_20d=4.0,
            volume_expansion_60d=2.5,
            dollar_volume_expansion_20d=2.5,
            dollar_volume_expansion_60d=2.0,
            liquidity_proxy_score=0.8,
        ),
    ]
    untouched_rows = [
        _seed_market_path_feature_row(
            db_session,
            outside_date,
            ticker="ROUT",
            feature_date=date(2025, 12, 31),
            dollar_volume=9_000_000.0,
        ),
        _seed_market_path_feature_row(
            db_session,
            other_pattern,
            ticker="RONE",
            pattern_id="M1",
            feature_date=date(2026, 1, 15),
            dollar_volume=9_000_000.0,
        ),
        _seed_market_path_feature_row(
            db_session,
            other_version,
            ticker="RVVV",
            feature_date=date(2026, 1, 15),
            feature_version="market_path_daily_v2",
            dollar_volume=9_000_000.0,
        ),
    ]
    untouched_before = {
        row.market_path_feature_id: (row.output_hash, row.feature_json, _rank_state(row))
        for row in untouched_rows
    }
    adapter = NoFetchFmpAdapter()
    job_kwargs = dict(
        session=db_session,
        fmp_adapter=adapter,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 1, 1),
        signal_end_date=date(2026, 2, 28),
        run_timestamp=RUN_TS,
        progress_artifact=tmp_path / "rank_only.json",
        schema="scratch_test",
        progress_every=1,
    )

    first = run_job(db_session, MarketPathRankOnlyBackfillJob(**job_kwargs))
    db_session.expire_all()

    assert first.status == "finished"
    assert first.metrics["source"] == "market_path_rank_only_backfill"
    assert first.metrics["rank_rows_updated"] == 4
    assert first.metrics["rank_month_count"] == 2
    assert first.metrics["rank_populated_count"] == 4
    assert first.metrics["rank_null_count"] == 0
    assert first.metrics["fmp_fetch_count"] == 0
    assert adapter.calls == []
    jan_a_row = db_session.get(MarketPathFeature, target_rows[0].market_path_feature_id)
    jan_b_row = db_session.get(MarketPathFeature, target_rows[1].market_path_feature_id)
    feb_a_row = db_session.get(MarketPathFeature, target_rows[2].market_path_feature_id)
    feb_b_row = db_session.get(MarketPathFeature, target_rows[3].market_path_feature_id)
    assert jan_a_row.dollar_volume_rank == 1
    assert jan_b_row.dollar_volume_rank == 2
    assert feb_b_row.dollar_volume_rank == 1
    assert feb_a_row.dollar_volume_rank == 2
    assert jan_a_row.cohort_pattern_row_count == 2
    assert jan_a_row.cohort_feature_row_count == 2
    assert "cross_sectional_features" in json.loads(jan_a_row.feature_json)
    artifact = json.loads((tmp_path / "rank_only.json").read_text())
    assert artifact["job"] == "market_path_rank_only_backfill"
    assert artifact["mode"] == "rank_only"
    assert {event["event"] for event in artifact["progress_events"]} >= {
        "rank_only_start",
        "rank_month_start",
        "rank_group_progress",
        "rank_month_finish",
        "rank_only_finish",
        "validation_finish",
    }
    assert {
        row.market_path_feature_id: (row.output_hash, row.feature_json, _rank_state(row))
        for row in untouched_rows
    } == untouched_before
    target_hashes = {
        row.market_path_feature_id: (row.output_hash, row.feature_json, _rank_state(row))
        for row in (jan_a_row, jan_b_row, feb_a_row, feb_b_row)
    }

    second = run_job(db_session, MarketPathRankOnlyBackfillJob(**job_kwargs))
    db_session.expire_all()

    assert second.status == "finished"
    assert second.metrics["rank_rows_updated"] == 0
    assert adapter.calls == []
    assert {
        row.market_path_feature_id: (row.output_hash, row.feature_json, _rank_state(row))
        for row in (
            db_session.get(MarketPathFeature, target_rows[0].market_path_feature_id),
            db_session.get(MarketPathFeature, target_rows[1].market_path_feature_id),
            db_session.get(MarketPathFeature, target_rows[2].market_path_feature_id),
            db_session.get(MarketPathFeature, target_rows[3].market_path_feature_id),
        )
    } == target_hashes


def test_market_path_rank_only_month_chunks_match_single_range_rank_call(db_session):
    signals = {
        ticker: _add_signal(
            db_session,
            ticker=ticker,
            signal_day=signal_day,
            entry_day=feature_day,
        )
        for ticker, signal_day, feature_day in (
            ("SJA", date(2026, 1, 14), date(2026, 1, 15)),
            ("SJB", date(2026, 1, 14), date(2026, 1, 15)),
            ("SFA", date(2026, 2, 2), date(2026, 2, 3)),
            ("SFB", date(2026, 2, 2), date(2026, 2, 3)),
        )
    }
    fixture = (
        ("SJA", date(2026, 1, 15), 1_000_000.0),
        ("SJB", date(2026, 1, 15), 500_000.0),
        ("SFA", date(2026, 2, 3), 200_000.0),
        ("SFB", date(2026, 2, 3), 900_000.0),
    )
    for ticker, feature_day, dollar_volume in fixture:
        _seed_market_path_feature_row(
            db_session,
            signals[ticker],
            ticker=ticker,
            feature_date=feature_day,
            feature_version="market_path_daily_single",
            dollar_volume=dollar_volume,
            volume_expansion_20d=dollar_volume / 100_000.0,
            volume_expansion_60d=dollar_volume / 200_000.0,
            dollar_volume_expansion_20d=dollar_volume / 300_000.0,
            dollar_volume_expansion_60d=dollar_volume / 400_000.0,
            liquidity_proxy_score=dollar_volume / 1_000_000.0,
        )
        _seed_market_path_feature_row(
            db_session,
            signals[ticker],
            ticker=ticker,
            feature_date=feature_day,
            feature_version="market_path_daily_chunked",
            dollar_volume=dollar_volume,
            volume_expansion_20d=dollar_volume / 100_000.0,
            volume_expansion_60d=dollar_volume / 200_000.0,
            dollar_volume_expansion_20d=dollar_volume / 300_000.0,
            dollar_volume_expansion_60d=dollar_volume / 400_000.0,
            liquidity_proxy_score=dollar_volume / 1_000_000.0,
        )

    single_adapter = NoFetchFmpAdapter()
    single_job = MarketPathFeatureJob(
        session=db_session,
        fmp_adapter=single_adapter,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 1, 1),
        signal_end_date=date(2026, 2, 28),
        through_date=date(2026, 2, 28),
        feature_version="market_path_daily_single",
    )
    single_updated = single_job._populate_cross_sectional_ranks(
        start_date=date(2026, 1, 1),
        through_date=date(2026, 2, 28),
    )
    db_session.commit()
    chunk_adapter = NoFetchFmpAdapter()
    chunk_result = MarketPathFeatureJob(
        session=db_session,
        fmp_adapter=chunk_adapter,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 1, 1),
        signal_end_date=date(2026, 2, 28),
        through_date=date(2026, 2, 28),
        feature_version="market_path_daily_chunked",
    ).populate_ranks_only(
        start_date=date(2026, 1, 1),
        through_date=date(2026, 2, 28),
    )
    db_session.expire_all()

    assert single_updated == 4
    assert chunk_result["rank_rows_updated"] == 4
    assert chunk_result["rank_month_count"] == 2
    assert single_adapter.calls == []
    assert chunk_adapter.calls == []
    for ticker, feature_day, _ in fixture:
        single_row = (
            db_session.query(MarketPathFeature)
            .filter(
                MarketPathFeature.ticker == ticker,
                MarketPathFeature.feature_session_date == feature_day.isoformat(),
                MarketPathFeature.feature_version == "market_path_daily_single",
            )
            .one()
        )
        chunk_row = (
            db_session.query(MarketPathFeature)
            .filter(
                MarketPathFeature.ticker == ticker,
                MarketPathFeature.feature_session_date == feature_day.isoformat(),
                MarketPathFeature.feature_version == "market_path_daily_chunked",
            )
            .one()
        )
        assert _rank_state(chunk_row) == _rank_state(single_row)


def test_market_path_bulk_failed_fetch_artifact_preserves_retry_metadata(
    db_session,
    tmp_path,
):
    _add_signal(
        db_session,
        ticker="LCUT",
        signal_day=date(2026, 6, 2),
        entry_day=date(2026, 6, 3),
    )
    leaky_message = (
        "HTTPSConnectionPool(host='financialmodelingprep.com', port=443): "
        "Max retries exceeded with url: "
        "/stable/historical-price-eod/full?symbol=LCUT&apikey=SECRET"
        "&api_key=ALSOSECRET&token=THIRDSECRET"
    )
    adapter = RetryingHistoricalPriceFmpAdapter(
        FakeFmpAdapter(
            {
                "LCUT": _rich_bars(),
                "SPY": _reference_bars(),
                "QQQ": _reference_bars(),
                "IWM": _reference_bars(),
            },
            fail_symbols={"LCUT"},
            fail_message=leaky_message,
        ),
        max_retries=1,
        retry_sleep_seconds=0.0,
        request_timeout_seconds=6.5,
    )
    artifact_path = tmp_path / "bulk_failed_fetch.json"

    result = run_job(
        db_session,
        MarketPathBulkBackfillJob(
            session=db_session,
            fmp_adapter=adapter,
            pattern_ids=("M4",),
            signal_start_date=date(2026, 6, 2),
            signal_end_date=date(2026, 6, 2),
            through_date=date(2026, 6, 3),
            run_timestamp=RUN_TS,
            batch_days=1,
            progress_artifact=artifact_path,
            schema="scratch_test",
        ),
    )

    assert result.status == "partial_failed"
    assert result.errors
    error = result.errors[0]
    assert error["ticker"] == "LCUT"
    assert error["provider"] == "FMP"
    assert error["status_code"] == 503
    assert error["error_type"] == "http"
    assert error["message"] != leaky_message
    assert error["retryable"] is True
    assert error["retry_attempt_count"] == 2
    assert error["retry_max_retries"] == 1
    assert error["retry_exhausted"] is True
    assert error["request_timeout_seconds"] == 6.5
    assert [attempt["attempt"] for attempt in error["retry_attempts"]] == [1, 2]

    artifact = json.loads(artifact_path.read_text())
    batch_error = artifact["batches"][0]["fetch_errors"][0]
    summary_error = artifact["summary"]["fetch_error_sample"][0]
    for persisted in (batch_error, summary_error):
        dumped = json.dumps(persisted, sort_keys=True)
        assert persisted["ticker"] == "LCUT"
        assert persisted["message"] != leaky_message
        assert persisted["retry_attempt_count"] == 2
        assert persisted["retry_max_retries"] == 1
        assert persisted["retry_exhausted"] is True
        assert persisted["request_timeout_seconds"] == 6.5
        assert persisted["retry_attempts"][0]["error_type"] == "http"
        assert "SECRET" not in dumped
        assert "ALSOSECRET" not in dumped
        assert "THIRDSECRET" not in dumped
        assert "apikey" not in dumped.lower()
        assert "api_key" not in dumped.lower()
        assert "token" not in dumped.lower()
        assert "?" not in dumped


def test_market_path_v3_rank_pass_isolates_date_pattern_and_version(db_session):
    m4_a = _add_signal(db_session, ticker="AAAA")
    m4_b = _add_signal(db_session, ticker="BBBB")
    m1 = _add_signal(db_session, pattern_id="M1", ticker="CCCC")
    later = _add_signal(
        db_session,
        ticker="DDDD",
        signal_day=date(2026, 6, 3),
        entry_day=date(2026, 6, 4),
    )
    bars_a = _replace_bar(_rich_bars(), date(2026, 6, 3), close=20.0, volume=1_000_000)
    bars_b = _replace_bar(_rich_bars(), date(2026, 6, 3), close=10.0, volume=500_000)
    bars_c = _replace_bar(_rich_bars(), date(2026, 6, 3), close=30.0, volume=1_500_000)
    bars_d = _replace_bar(_rich_bars(), date(2026, 6, 4), close=40.0, volume=2_000_000)
    adapter = _adapter_with_references({
        "AAAA": bars_a,
        "BBBB": bars_b,
        "CCCC": bars_c,
        "DDDD": bars_d,
    })
    kwargs = dict(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=RUN_TS,
        pattern_ids=("M4", "M1"),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 3),
        through_date=date(2026, 6, 4),
    )
    run_job(db_session, MarketPathFeatureJob(feature_version="market_path_daily_v2", **kwargs))
    v2_row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == m4_a.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
            MarketPathFeature.feature_version == "market_path_daily_v2",
        )
        .one()
    )
    v2_hash = v2_row.output_hash

    result = run_job(db_session, MarketPathFeatureJob(**kwargs))

    assert result.status == "finished"
    a_row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == m4_a.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
            MarketPathFeature.feature_version == "market_path_daily_v3",
        )
        .one()
    )
    b_row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == m4_b.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
            MarketPathFeature.feature_version == "market_path_daily_v3",
        )
        .one()
    )
    c_row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == m1.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
            MarketPathFeature.feature_version == "market_path_daily_v3",
        )
        .one()
    )
    d_row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == later.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-04",
            MarketPathFeature.feature_version == "market_path_daily_v3",
        )
        .one()
    )
    assert a_row.dollar_volume_rank == 1
    assert b_row.dollar_volume_rank == 2
    assert a_row.dollar_volume_percentile == 1.0
    assert b_row.dollar_volume_percentile == 0.5
    assert a_row.cohort_pattern_row_count == 2
    assert c_row.cohort_pattern_row_count == 1
    assert c_row.dollar_volume_rank == 1
    assert d_row.cohort_pattern_row_count == 3
    assert (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == m4_a.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
            MarketPathFeature.feature_version == "market_path_daily_v2",
        )
        .one()
        .output_hash
        == v2_hash
    )


def test_market_path_v3_rank_rerun_does_not_rewrite_unchanged_rank_state(db_session):
    _add_signal(db_session, ticker="AAAA")
    _add_signal(db_session, ticker="BBBB")
    adapter = _adapter_with_references({
        "AAAA": _replace_bar(_rich_bars(), date(2026, 6, 3), close=20.0, volume=1_000_000),
        "BBBB": _replace_bar(_rich_bars(), date(2026, 6, 3), close=10.0, volume=500_000),
    })
    kwargs = dict(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 3),
    )

    first = run_job(db_session, MarketPathFeatureJob(**kwargs))
    assert first.metrics["rows_upserted"] == 2
    assert first.metrics["rank_rows_updated"] == 2
    first_hashes = {
        (row.signal_id, row.feature_session_date): row.output_hash
        for row in db_session.query(MarketPathFeature).all()
    }

    second = run_job(db_session, MarketPathFeatureJob(**kwargs))
    db_session.expire_all()
    assert second.metrics["rows_upserted"] == 0
    assert second.metrics["rank_rows_updated"] == 0
    assert {
        (row.signal_id, row.feature_session_date): row.output_hash
        for row in db_session.query(MarketPathFeature).all()
    } == first_hashes
    third = run_job(db_session, MarketPathFeatureJob(**kwargs))
    db_session.expire_all()
    assert third.metrics["rows_upserted"] == 0
    assert third.metrics["rank_rows_updated"] == 0
    assert {
        (row.signal_id, row.feature_session_date): row.output_hash
        for row in db_session.query(MarketPathFeature).all()
    } == first_hashes


def test_market_path_chunked_overlap_rank_rerun_counts_only_material_updates(
    db_session,
    tmp_path,
):
    _add_signal(
        db_session,
        ticker="AAAA",
        signal_day=date(2026, 6, 1),
        entry_day=date(2026, 6, 2),
    )
    _add_signal(
        db_session,
        ticker="BBBB",
        signal_day=date(2026, 6, 1),
        entry_day=date(2026, 6, 2),
    )
    _add_signal(
        db_session,
        ticker="CCCC",
        signal_day=date(2026, 6, 2),
        entry_day=date(2026, 6, 3),
    )
    _add_signal(
        db_session,
        ticker="DDDD",
        signal_day=date(2026, 6, 2),
        entry_day=date(2026, 6, 3),
    )
    adapter = _adapter_with_references({
        "AAAA": _replace_bar(_rich_bars(), date(2026, 6, 3), close=40.0, volume=1_000_000),
        "BBBB": _replace_bar(_rich_bars(), date(2026, 6, 3), close=20.0, volume=500_000),
        "CCCC": _replace_bar(_rich_bars(), date(2026, 6, 3), close=30.0, volume=800_000),
        "DDDD": _replace_bar(_rich_bars(), date(2026, 6, 3), close=10.0, volume=300_000),
    })
    chunks = plan_chunks(
        ["M4"],
        signal_start_date=date(2026, 6, 1),
        signal_end_date=date(2026, 6, 2),
    )
    config = BackfillRunConfig(
        through_date=date(2026, 6, 3),
        run_timestamp=RUN_TS,
        include_signal_session=True,
    )
    session_factory = lambda: _NoCloseSession(db_session)

    def chunk_metrics(path):
        artifact = json.loads(path.read_text())
        return [chunk["metrics"] for chunk in artifact["chunks"]]

    first_path = tmp_path / "overlap_first.json"
    first = run_backfill_chunks(
        chunks,
        session_factory=session_factory,
        fmp_adapter=adapter,
        config=config,
        artifact_path=first_path,
        print_fn=lambda _: None,
    )
    assert first["chunks_failed"] == 0
    assert sum(int(metrics["rank_rows_updated"]) for metrics in chunk_metrics(first_path)) > 0

    second_path = tmp_path / "overlap_second.json"
    second = run_backfill_chunks(
        chunks,
        session_factory=session_factory,
        fmp_adapter=adapter,
        config=config,
        artifact_path=second_path,
        print_fn=lambda _: None,
    )
    assert second["chunks_failed"] == 0
    assert sum(int(metrics["rows_upserted"]) for metrics in chunk_metrics(second_path)) == 0
    stable_after_second = {
        (row.signal_id, row.feature_session_date): (
            row.output_hash,
            row.feature_json,
        )
        for row in db_session.query(MarketPathFeature).all()
    }

    third_path = tmp_path / "overlap_third.json"
    third = run_backfill_chunks(
        chunks,
        session_factory=session_factory,
        fmp_adapter=adapter,
        config=config,
        artifact_path=third_path,
        print_fn=lambda _: None,
    )
    assert third["chunks_failed"] == 0
    assert sum(int(metrics["rows_upserted"]) for metrics in chunk_metrics(third_path)) == 0
    assert sum(int(metrics["rank_rows_updated"]) for metrics in chunk_metrics(third_path)) == 0
    assert {
        (row.signal_id, row.feature_session_date): (
            row.output_hash,
            row.feature_json,
        )
        for row in db_session.query(MarketPathFeature).all()
    } == stable_after_second

    changed_adapter = _adapter_with_references({
        "AAAA": _replace_bar(_rich_bars(), date(2026, 6, 3), close=40.0, volume=1_000_000),
        "BBBB": _replace_bar(_rich_bars(), date(2026, 6, 3), close=90.0, volume=3_000_000),
        "CCCC": _replace_bar(_rich_bars(), date(2026, 6, 3), close=30.0, volume=800_000),
        "DDDD": _replace_bar(_rich_bars(), date(2026, 6, 3), close=10.0, volume=300_000),
    })
    material_path = tmp_path / "overlap_material.json"
    material = run_backfill_chunks(
        chunks,
        session_factory=session_factory,
        fmp_adapter=changed_adapter,
        config=config,
        artifact_path=material_path,
        print_fn=lambda _: None,
    )
    assert material["chunks_failed"] == 0
    assert sum(int(metrics["rank_rows_updated"]) for metrics in chunk_metrics(material_path)) > 0
    assert any(
        stable_after_second[key][0] != row.output_hash
        for row in db_session.query(MarketPathFeature).all()
        if (key := (row.signal_id, row.feature_session_date)) in stable_after_second
    )


def test_market_path_v3_rank_null_inputs_and_ties_are_deterministic(db_session):
    left = _add_signal(db_session, ticker="AAAA")
    right = _add_signal(db_session, ticker="BBBB")
    short = _add_signal(db_session, ticker="SHORT")
    adapter = _adapter_with_references({
        "AAAA": _replace_bar(_rich_bars(), date(2026, 6, 3), close=20.0, volume=1_000_000),
        "BBBB": _replace_bar(_rich_bars(), date(2026, 6, 3), close=20.0, volume=1_000_000),
        "SHORT": _single_entry_bar(),
    })
    result = run_job(
        db_session,
        MarketPathFeatureJob(
            session=db_session,
            fmp_adapter=adapter,
            run_timestamp=RUN_TS,
            pattern_ids=("M4",),
            signal_start_date=date(2026, 6, 2),
            signal_end_date=date(2026, 6, 2),
            through_date=date(2026, 6, 3),
        ),
    )

    assert result.status == "finished"
    left_row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == left.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    right_row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == right.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    short_row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == short.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert left_row.dollar_volume_rank == 1
    assert right_row.dollar_volume_rank == 2
    assert short_row.volume_expansion_60d_rank is None
    status = json.loads(short_row.feature_json)["relative_feature_status"]["cross_sectional_rank"]
    assert status["volume_expansion_60d"]["value_missing"] is True


def test_market_path_v3_fetches_benchmarks_once_and_uses_prior_only_returns(db_session):
    signal = _add_signal(db_session)
    baseline = _adapter_with_references({"LCUT": _rich_bars()})
    kwargs = dict(
        session=db_session,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )
    run_job(db_session, MarketPathFeatureJob(fmp_adapter=baseline, **kwargs))
    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    baseline_spy_return = row.spy_return_5d
    baseline_spy_return_1d = row.spy_return_1d
    baseline_hash = row.input_hash
    baseline_output_hash = row.output_hash
    feature_json = json.loads(row.feature_json)
    assert row.spy_return_1d is not None
    assert row.qqq_return_1d is not None
    assert row.iwm_return_1d is not None
    assert feature_json["market_relative_features"]["spy_return_1d"] == row.spy_return_1d
    assert feature_json["market_relative_features"]["qqq_return_1d"] == row.qqq_return_1d
    assert feature_json["market_relative_features"]["iwm_return_1d"] == row.iwm_return_1d
    assert [call["ticker"] for call in baseline.calls].count("SPY") == 1
    assert [call["ticker"] for call in baseline.calls].count("QQQ") == 1
    assert [call["ticker"] for call in baseline.calls].count("IWM") == 1

    changed = _adapter_with_references({
        "LCUT": _rich_bars(),
        "SPY": _reference_bars(start_price=400.0, future_spike=True),
    })
    run_job(db_session, MarketPathFeatureJob(fmp_adapter=changed, **kwargs))
    db_session.expire_all()
    updated = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert updated.spy_return_5d == baseline_spy_return
    assert updated.spy_return_1d == baseline_spy_return_1d
    assert updated.input_hash == baseline_hash

    prior_changed = _adapter_with_references({
        "LCUT": _rich_bars(),
        "SPY": _replace_bar(_reference_bars(start_price=400.0), date(2026, 6, 2), close=888.0, split_adjusted_close=888.0),
    })
    run_job(db_session, MarketPathFeatureJob(fmp_adapter=prior_changed, **kwargs))
    db_session.expire_all()
    prior_updated = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert prior_updated.input_hash != baseline_hash
    assert prior_updated.spy_return_5d != baseline_spy_return
    assert prior_updated.spy_return_1d != baseline_spy_return_1d
    assert prior_updated.output_hash != baseline_output_hash


def test_market_path_v3_missing_benchmark_stamps_status_without_failing(db_session):
    signal = _add_signal(db_session)
    adapter = _adapter_with_references({"LCUT": _rich_bars()}, fail_symbols={"SPY"})
    result = run_job(
        db_session,
        MarketPathFeatureJob(
            session=db_session,
            fmp_adapter=adapter,
            run_timestamp=RUN_TS,
            pattern_ids=("M4",),
            signal_start_date=date(2026, 6, 2),
            signal_end_date=date(2026, 6, 2),
            through_date=date(2026, 6, 5),
        ),
    )

    assert result.status == "finished"
    assert result.metrics["benchmark_fetch_error_count"] == 1
    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    status = json.loads(row.feature_json)["relative_feature_status"]["benchmark"]["SPY"]
    market_features = json.loads(row.feature_json)["market_relative_features"]
    assert row.spy_return_1d is None
    assert row.spy_return_5d is None
    assert market_features["spy_return_1d"] is None
    assert row.relative_strength_vs_spy_5d is None
    assert status["missing_benchmark_bars"] is True
    assert status["fetch_status"] == "fetch_error"


def test_market_path_ml_context_fields_compute_prior_only_and_stamp_missing_sources(db_session):
    signal = _add_signal(db_session)
    baseline = _adapter_with_references({"LCUT": _rich_bars()})
    kwargs = dict(
        session=db_session,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )
    run_job(db_session, MarketPathFeatureJob(fmp_adapter=baseline, **kwargs))
    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    original_hash = row.input_hash
    original_output_hash = row.output_hash
    original_rsi = row.rsi_14
    original_macd = row.macd_histogram
    original_vol_proxy = row.volatility_regime_proxy
    assert row.rsi_14 == pytest.approx(_expected_rsi(_rich_bars(), date(2026, 6, 3), 14))
    assert row.adx_14 is not None
    assert row.bollinger_bandwidth_20d is not None
    assert row.macd_histogram is not None
    assert row.volatility_regime_proxy is not None
    assert row.volatility_regime_source == "IWM_SPY_REALIZED_VOL_20D"
    assert row.market_regime_status == "volatility_proxy_available_breadth_unavailable"
    assert row.opening_range_high_5m is None
    assert row.intraday_structure_status == "intraday_adapter_unavailable"
    assert row.missing_intraday_bars is True
    assert row.bid_ask_spread is None
    assert row.quote_status == "quote_source_unavailable"
    assert row.missing_quote is True
    assert row.float_shares is None
    assert row.supply_squeeze_status == "pit_safe_sources_unavailable"
    assert row.news_count_20d is None
    assert row.missing_catalyst_source is True
    payload = json.loads(row.feature_json)
    assert payload["market_regime_status"]["prior_only"] is True
    assert payload["intraday_structure_status"]["missing_intraday_bars"] is True
    assert payload["execution_quality_status"]["missing_quote"] is True
    assert payload["supply_squeeze_status"]["float_source_status"] == "pit_float_source_unavailable"
    assert payload["classic_technical_status"]["prior_only"] is True
    assert payload["classic_technical_features"]["rsi_14"] == pytest.approx(row.rsi_14)

    future_changed = _adapter_with_references({
        "LCUT": _replace_bar(_rich_bars(), date(2026, 6, 4), close=200.0, split_adjusted_close=200.0),
        "SPY": _reference_bars(start_price=400.0, future_spike=True),
        "IWM": _reference_bars(start_price=200.0, future_spike=True),
    })
    run_job(db_session, MarketPathFeatureJob(fmp_adapter=future_changed, **kwargs))
    db_session.expire_all()
    future_row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert future_row.input_hash == original_hash
    assert future_row.rsi_14 == original_rsi
    assert future_row.macd_histogram == original_macd
    assert future_row.volatility_regime_proxy == original_vol_proxy

    prior_changed = _adapter_with_references({
        "LCUT": _replace_bar(_rich_bars(), date(2026, 6, 2), close=1.0, split_adjusted_close=1.0),
        "SPY": _replace_bar(_reference_bars(start_price=400.0), date(2026, 6, 2), close=999.0, split_adjusted_close=999.0),
        "IWM": _replace_bar(_reference_bars(start_price=200.0), date(2026, 6, 2), close=777.0, split_adjusted_close=777.0),
    })
    run_job(db_session, MarketPathFeatureJob(fmp_adapter=prior_changed, **kwargs))
    db_session.expire_all()
    prior_row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert prior_row.input_hash != original_hash
    assert prior_row.rsi_14 != original_rsi
    assert prior_row.volatility_regime_proxy != original_vol_proxy
    assert prior_row.output_hash != original_output_hash


def test_market_path_adx_14_uses_wilder_smoothed_adx_not_single_window_dx(db_session):
    signal = _add_signal(db_session)
    adapter = _adapter_with_references({"LCUT": _adx_handcheck_bars()})
    result = run_job(
        db_session,
        MarketPathFeatureJob(
            session=db_session,
            fmp_adapter=adapter,
            run_timestamp=RUN_TS,
            pattern_ids=("M4",),
            signal_start_date=date(2026, 6, 2),
            signal_end_date=date(2026, 6, 2),
            through_date=date(2026, 6, 3),
        ),
    )

    assert result.status == "finished"
    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert row.plus_di_14 == pytest.approx(31.77182603060645)
    assert row.minus_di_14 == pytest.approx(19.8645184075919)
    assert row.adx_14 == pytest.approx(36.95928479597885)
    assert row.adx_14 != pytest.approx(23.86407754815589)
    payload = json.loads(row.feature_json)
    assert payload["classic_technical_features"]["adx_14"] == pytest.approx(row.adx_14)


def test_market_path_ml_context_cross_pattern_cofire_uses_signal_registry(db_session):
    signal = _add_signal(db_session, ticker="COF", pattern_id="M4")
    _add_signal(db_session, ticker="COF", pattern_id="M2")
    adapter = _adapter_with_references({"COF": _rich_bars()})
    result = run_job(
        db_session,
        MarketPathFeatureJob(
            session=db_session,
            fmp_adapter=adapter,
            run_timestamp=RUN_TS,
            pattern_ids=("M4",),
            signal_start_date=date(2026, 6, 2),
            signal_end_date=date(2026, 6, 2),
            through_date=date(2026, 6, 3),
        ),
    )

    assert result.status == "finished"
    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert row.cofire_m4 is True
    assert row.cofire_m2 is True
    assert row.insider_buy_overlap_m2 is True
    assert row.cross_pattern_overlap_count == 2
    assert row.strongest_overlap_pattern_id == "M2"
    payload = json.loads(row.feature_json)
    assert payload["catalyst_context_status"]["signal_registry_context_available"] is True
    assert payload["catalyst_context_status"]["external_catalyst_sources_available"] is False


def test_market_path_cofire_prefetch_avoids_per_feature_row_signal_queries(db_session):
    _add_signal(db_session, ticker="NPLUS", pattern_id="M4")
    _add_signal(db_session, ticker="NPLUS", pattern_id="M1")
    adapter = _adapter_with_references({"NPLUS": _rich_bars()})
    signal_registry_selects = 0

    def count_signal_registry_selects(conn, cursor, statement, parameters, context, executemany):
        nonlocal signal_registry_selects
        normalized = " ".join(statement.lower().split())
        if normalized.startswith("select") and "from signal_registry" in normalized:
            signal_registry_selects += 1

    event.listen(
        db_session.get_bind(),
        "before_cursor_execute",
        count_signal_registry_selects,
    )
    try:
        result = run_job(
            db_session,
            MarketPathFeatureJob(
                session=db_session,
                fmp_adapter=adapter,
                run_timestamp=RUN_TS,
                pattern_ids=("M4",),
                signal_start_date=date(2026, 6, 2),
                signal_end_date=date(2026, 6, 2),
                through_date=date(2026, 6, 5),
            ),
        )
    finally:
        event.remove(
            db_session.get_bind(),
            "before_cursor_execute",
            count_signal_registry_selects,
        )

    assert result.status == "finished"
    assert result.metrics["rows_inserted"] == 3
    assert signal_registry_selects == 2


def test_market_path_cofire_cache_preserves_same_day_semantics_and_tie_break(db_session):
    signal = _add_signal(db_session, ticker="COFC", pattern_id="M4")
    _add_signal(db_session, ticker="COFC", pattern_id="M1")
    _add_signal(
        db_session,
        ticker="COFC",
        pattern_id="M2",
        signal_day=date(2026, 6, 3),
        entry_day=date(2026, 6, 4),
    )
    adapter = _adapter_with_references({"COFC": _rich_bars()})

    result = run_job(
        db_session,
        MarketPathFeatureJob(
            session=db_session,
            fmp_adapter=adapter,
            run_timestamp=RUN_TS,
            pattern_ids=("M4",),
            signal_start_date=date(2026, 6, 2),
            signal_end_date=date(2026, 6, 2),
            through_date=date(2026, 6, 4),
        ),
    )

    assert result.status == "finished"
    rows = (
        db_session.query(MarketPathFeature)
        .filter(MarketPathFeature.signal_id == signal.signal_id)
        .order_by(MarketPathFeature.feature_session_date)
        .all()
    )
    assert [row.feature_session_date for row in rows] == ["2026-06-03", "2026-06-04"]
    for row in rows:
        assert row.cofire_m1 is True
        assert row.cofire_m2 is False
        assert row.cofire_m4 is True
        assert row.cross_pattern_overlap_count == 2
        assert row.strongest_overlap_pattern_id == "M1"

    assert _same_day_pattern_strengths_from_cache(
        {("COFC", date(2026, 6, 2)): {"M4": 1.0}},
        signal,
        date(2026, 6, 1),
    ) == {}


def test_market_path_v3_sector_etf_map_and_sector_relative_no_lookahead(db_session):
    signal = _add_signal(db_session)
    db_session.execute(text(
        "INSERT INTO firm_sector_assignments_history "
        "(ticker, valid_from, sector, source, sic_to_sector_map_version, valid_to) "
        "VALUES ('LCUT', '2020-01-01', 'Technology', 'POLYGON_SIC', 'test_v1', '2099-01-01')"
    ))
    db_session.commit()
    baseline = _adapter_with_references({
        "LCUT": _rich_bars(),
        "XLK": _reference_bars(start_price=150.0),
    })
    kwargs = dict(
        session=db_session,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )
    run_job(db_session, MarketPathFeatureJob(fmp_adapter=baseline, **kwargs))
    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    baseline_sector_return = row.sector_etf_return_5d
    baseline_input_hash = row.input_hash
    assert row.sector_etf == "XLK"
    assert row.sector_source == "M3_PIT:POLYGON_SIC"
    assert row.sector_relative_status == "available"
    assert [call["ticker"] for call in baseline.calls].count("XLK") == 1

    changed = _adapter_with_references({
        "LCUT": _rich_bars(),
        "XLK": _reference_bars(start_price=150.0, future_spike=True),
    })
    run_job(db_session, MarketPathFeatureJob(fmp_adapter=changed, **kwargs))
    db_session.expire_all()
    updated = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert updated.sector_etf_return_5d == baseline_sector_return
    assert updated.input_hash == baseline_input_hash

    prior_changed = _adapter_with_references({
        "LCUT": _rich_bars(),
        "XLK": _replace_bar(_reference_bars(start_price=150.0), date(2026, 6, 2), close=333.0, split_adjusted_close=333.0),
    })
    run_job(db_session, MarketPathFeatureJob(fmp_adapter=prior_changed, **kwargs))
    db_session.expire_all()
    prior_updated = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert prior_updated.input_hash != baseline_input_hash
    assert prior_updated.sector_etf_return_5d != baseline_sector_return


def test_market_path_v3_missing_sector_and_missing_m3_tables_are_status_only(db_session):
    missing_sector_signal = _add_signal(db_session, ticker="MISS")
    adapter = _adapter_with_references({"MISS": _rich_bars()})
    result = run_job(
        db_session,
        MarketPathFeatureJob(
            session=db_session,
            fmp_adapter=adapter,
            run_timestamp=RUN_TS,
            pattern_ids=("M4",),
            signal_start_date=date(2026, 6, 2),
            signal_end_date=date(2026, 6, 2),
            through_date=date(2026, 6, 3),
        ),
    )
    assert result.status == "finished"
    missing_sector_row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == missing_sector_signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert missing_sector_row.sector_etf is None
    assert missing_sector_row.sector_relative_status == "sector_missing"

    db_session.execute(text("DROP TABLE firm_sector_assignments_history"))
    db_session.commit()
    missing_table_signal = _add_signal(db_session, ticker="NOTAB")
    adapter = _adapter_with_references({"NOTAB": _rich_bars()})
    result = run_job(
        db_session,
        MarketPathFeatureJob(
            session=db_session,
            fmp_adapter=adapter,
            run_timestamp=RUN_TS,
            pattern_ids=("M4",),
            signal_start_date=date(2026, 6, 2),
            signal_end_date=date(2026, 6, 2),
            through_date=date(2026, 6, 3),
        ),
    )
    assert result.status == "finished"
    missing_table_row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == missing_table_signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert missing_table_row.sector_relative_status == "m3_sector_history_table_missing"
    status = json.loads(missing_table_row.feature_json)["relative_feature_status"]["sector"]
    assert status["missing_sector"] is True
    assert status["non_pit_sector_fallback"] is False


def test_market_path_output_hash_covers_persisted_feature_fields(db_session):
    signal = _add_signal(db_session)
    adapter = FakeFmpAdapter({"LCUT": _bars()})
    kwargs = dict(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )
    run_job(db_session, MarketPathFeatureJob(**kwargs))
    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    original_hash = row.output_hash

    adapter.bars_by_ticker["LCUT"] = _replace_bar(_bars(), date(2026, 6, 3), low=10.1)
    run_job(db_session, MarketPathFeatureJob(**kwargs))
    db_session.expire_all()
    updated = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )

    assert updated.low_price == 10.1
    assert updated.low_from_open_return == (10.1 / 11.0) - 1.0
    assert updated.output_hash != original_hash


def test_market_path_output_hash_covers_rich_eod_feature_fields(db_session):
    signal = _add_signal(db_session)
    adapter = FakeFmpAdapter({"LCUT": _rich_bars()})
    kwargs = dict(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )
    run_job(db_session, MarketPathFeatureJob(**kwargs))
    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    original_hash = row.output_hash
    assert row.prior_52w_high == 30.0

    adapter.bars_by_ticker["LCUT"] = _replace_bar(_rich_bars(), date(2026, 5, 29), high=29.5)
    run_job(db_session, MarketPathFeatureJob(**kwargs))
    db_session.expire_all()
    updated = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )

    assert updated.prior_52w_high == 29.5
    assert updated.breakout_extension_pct == pytest.approx((32.0 / 29.5) - 1.0)
    assert updated.output_hash != original_hash


def test_market_path_input_hash_covers_prior_row_inputs(db_session):
    signal = _add_signal(db_session)
    adapter = FakeFmpAdapter({"LCUT": _bars()})
    kwargs = dict(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )
    run_job(db_session, MarketPathFeatureJob(**kwargs))
    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    original_hash = row.input_hash
    original_row_input_hash = json.loads(row.feature_json)["row_input_hash"]

    adapter.bars_by_ticker["LCUT"] = _replace_bar(_bars(), date(2026, 6, 1), high=99.0)
    run_job(db_session, MarketPathFeatureJob(**kwargs))
    db_session.expire_all()
    updated = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    updated_json = json.loads(updated.feature_json)

    assert updated.input_hash != original_hash
    assert updated_json["row_input_hash"] != original_row_input_hash


def test_market_path_sigma_uses_split_adjusted_close_basis(db_session):
    signal = _add_signal(db_session)
    bars = _replace_bar(_bars(), date(2026, 6, 1), close=99.0)
    adapter = FakeFmpAdapter({"LCUT": bars})
    job = MarketPathFeatureJob(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 2),
        signal_end_date=date(2026, 6, 2),
        through_date=date(2026, 6, 5),
    )

    result = run_job(db_session, job)

    assert result.status == "finished"
    row = (
        db_session.query(MarketPathFeature)
        .filter(
            MarketPathFeature.signal_id == signal.signal_id,
            MarketPathFeature.feature_session_date == "2026-06-03",
        )
        .one()
    )
    assert row.sigma_20d == _split_adjusted_sigma(bars, date(2026, 6, 3))
    assert json.loads(row.feature_json)["sigma_basis"] == (
        "split_adjusted_close_when_available_else_raw_close"
    )


def test_market_path_feature_job_writes_signal_session_before_future_entry(db_session):
    signal = _add_signal(
        db_session,
        signal_day=date(2026, 6, 5),
        entry_day=date(2026, 6, 8),
    )
    adapter = FakeFmpAdapter({"LCUT": _bars()})
    job = MarketPathFeatureJob(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=date(2026, 6, 5),
        signal_end_date=date(2026, 6, 5),
        through_date=date(2026, 6, 5),
        include_signal_session=True,
    )

    result = run_job(db_session, job)

    assert result.status == "finished"
    assert result.metrics["rows_inserted"] == 1
    row = (
        db_session.query(MarketPathFeature)
        .filter(MarketPathFeature.signal_id == signal.signal_id)
        .one()
    )
    assert row.feature_session_date == "2026-06-05"
    assert row.feature_role == "signal_session"
    assert row.entry_session_date == "2026-06-08"
    assert row.return_from_entry_close is None
