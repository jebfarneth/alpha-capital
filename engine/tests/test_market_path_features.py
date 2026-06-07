from __future__ import annotations

import json
import math
import statistics
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect, text

from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError, stable_hash
from alpha.data.fmp import FmpBar, HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.db.models import MarketPathFeature
from alpha.evidence.writer import record_data_lineage, record_feature_snapshot, record_signal
from alpha.jobs.market_path_features import MarketPathFeatureJob
from alpha.jobs.runner import run_job


RUN_TS = datetime(2026, 6, 5, 21, 0, tzinfo=timezone.utc)


class FakeFmpAdapter:
    def __init__(self, bars_by_ticker, fail_symbols: set[str] | None = None):
        self.bars_by_ticker = bars_by_ticker
        self.fail_symbols = {symbol.upper() for symbol in (fail_symbols or set())}
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
                    message=f"forced failure for {ticker}",
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


def _split_adjusted_sigma(bars: list[FmpBar], row_day: date) -> float:
    prior = [
        bar for bar in bars
        if date.fromisoformat(bar.date) < row_day
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
    assert row.plus_di_14 == pytest.approx(30.48297680904405)
    assert row.minus_di_14 == pytest.approx(19.36577366838155)
    assert row.adx_14 == pytest.approx(36.99635786455282)
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
