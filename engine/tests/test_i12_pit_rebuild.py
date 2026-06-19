from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import time as time_module
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text

import alpha.jobs.i12_pit_rebuild as i12_pit_rebuild
import alpha.jobs.run_i12_pit_rebuild as run_i12_pit_rebuild
from alpha.data.alpaca import AlpacaQuote
from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError, stable_hash
from alpha.data.fmp import FmpBar
from alpha.data.polygon import PolygonBar
from alpha.db.models import (
    HistoricalUniverseReconstruction,
    I12PitCandidate,
    I12PitCostReplay,
    I12PitQuoteReplay,
    MLModelRegistry,
)
from alpha.jobs.i12_pit_rebuild import (
    I12PitRebuildJob,
    _candidate_attempt_hash,
    _hur_source_row_from_model,
    build_i12_pit_candidate,
    evaluate_quote_cost_replay,
    i12_pit_rebuild_report,
    quote_windows_for_candidate,
    replay_quote_window,
)
from alpha.jobs.contracts import JobContext
from alpha.jobs.run_i12_pit_rebuild import main as run_i12_pit_rebuild_main
from alpha.jobs.i12_live_fill_test import (
    FROZEN_I12_STAGE0_FEATURE_SCHEMA_HASH,
    FROZEN_I12_STAGE0_MANIFEST_SHA256,
    FROZEN_I12_STAGE0_MANIFEST_VERSION,
    select_i12_model,
)
from alpha.jobs.i12_historical_corpus import _clean_daily_bars, _clean_minute_bars
from alpha.jobs.runner import run_job
from alpha.jobs.paper_execution import EASTERN
from alpha.market_calendar import (
    next_us_equity_session,
    previous_us_equity_session,
    us_equity_session_open_timestamp,
)


DAY = date(2026, 6, 16)
DECISION_TS = datetime(2026, 6, 16, 13, 40, tzinfo=timezone.utc)


def _decision_ts_for_label(label: str) -> datetime:
    hour, minute = [int(part) for part in label.split(":", 1)]
    return datetime.combine(DAY, time(hour, minute), tzinfo=EASTERN).astimezone(timezone.utc)


class FakeFmp:
    def __init__(self, bars):
        self.bars = bars
        self.calls = []

    def get_historical_price(self, ticker, **kwargs):
        self.calls.append((ticker, dict(kwargs)))
        return AdapterResponse(data=list(self.bars), lineage=_lineage("FMP", ticker))


class FakePolygon:
    def __init__(self, bars):
        self.bars = bars
        self.calls = []

    def get_minute_aggs(self, ticker, from_date, to_date, **kwargs):
        self.calls.append((ticker, from_date, to_date, dict(kwargs)))
        return AdapterResponse(data=list(self.bars), lineage=_lineage("Polygon", ticker))


class FakeAlpaca:
    def __init__(self, quotes_by_role):
        self.quotes_by_role = quotes_by_role
        self.calls = []

    def get_historical_quotes(self, symbol, *, start, end, feed="sip"):
        self.calls.append((symbol, start, end, feed))
        role = _role_from_window(start, end)
        return AdapterResponse(
            data=list(self.quotes_by_role.get(role, [])),
            lineage=_lineage("Alpaca", symbol),
        )


class SleepyFmp(FakeFmp):
    def __init__(self, bars, *, delay_seconds=0.05):
        super().__init__(bars)
        self.delay_seconds = delay_seconds
        self.reset_calls = 0

    def get_historical_price(self, ticker, **kwargs):
        time_module.sleep(self.delay_seconds)
        return super().get_historical_price(ticker, **kwargs)

    def reset_session(self):
        self.reset_calls += 1


class SleepyPolygon(FakePolygon):
    def __init__(self, bars, *, delay_seconds=0.05):
        super().__init__(bars)
        self.delay_seconds = delay_seconds
        self.reset_calls = 0

    def get_minute_aggs(self, ticker, from_date, to_date, **kwargs):
        time_module.sleep(self.delay_seconds)
        return super().get_minute_aggs(ticker, from_date, to_date, **kwargs)

    def reset_session(self):
        self.reset_calls += 1


class SleepyAlpaca(FakeAlpaca):
    def __init__(self, quotes_by_role, *, delay_seconds=0.05):
        super().__init__(quotes_by_role)
        self.delay_seconds = delay_seconds
        self.reset_calls = 0

    def get_historical_quotes(self, symbol, *, start, end, feed="sip"):
        time_module.sleep(self.delay_seconds)
        return super().get_historical_quotes(
            symbol,
            start=start,
            end=end,
            feed=feed,
        )

    def reset_session(self):
        self.reset_calls += 1


class AssertingNoTransactionFmp(FakeFmp):
    def __init__(self, bars, session):
        super().__init__(bars)
        self.session = session

    def get_historical_price(self, ticker, **kwargs):
        assert self.session.in_transaction() is False
        return super().get_historical_price(ticker, **kwargs)


class AssertingNoTransactionPolygon(FakePolygon):
    def __init__(self, bars, session):
        super().__init__(bars)
        self.session = session

    def get_minute_aggs(self, ticker, from_date, to_date, **kwargs):
        assert self.session.in_transaction() is False
        return super().get_minute_aggs(ticker, from_date, to_date, **kwargs)


class AssertingNoTransactionAlpaca(FakeAlpaca):
    def __init__(self, quotes_by_role, session):
        super().__init__(quotes_by_role)
        self.session = session

    def get_historical_quotes(self, symbol, *, start, end, feed="sip"):
        assert self.session.in_transaction() is False
        return super().get_historical_quotes(symbol, start=start, end=end, feed=feed)


class FakeAlpacaResponses:
    def __init__(self, responses_by_role):
        self.responses_by_role = responses_by_role
        self.calls = []

    def get_historical_quotes(self, symbol, *, start, end, feed="sip"):
        self.calls.append((symbol, start, end, feed))
        role = _role_from_window(start, end)
        response = self.responses_by_role.get(role, [])
        if isinstance(response, AdapterResponse):
            return response
        return AdapterResponse(
            data=list(response),
            lineage=_lineage("Alpaca", symbol),
        )


class FailingAlpaca:
    def get_historical_quotes(self, symbol, *, start, end, feed="sip"):
        del symbol, start, end, feed
        raise AssertionError("provider should not be called for active ok quotes")


class FakeFmpByTicker:
    def __init__(self, bars_by_ticker, error_tickers=None):
        self.bars_by_ticker = bars_by_ticker
        self.error_tickers = {ticker.upper() for ticker in (error_tickers or set())}
        self.calls = []

    def get_historical_price(self, ticker, **kwargs):
        self.calls.append((ticker, dict(kwargs)))
        normalized = ticker.upper()
        if normalized in self.error_tickers:
            return _provider_error_response("FMP", normalized, "daily_fetch_failed")
        return AdapterResponse(
            data=list(self.bars_by_ticker.get(normalized, _fmp_bars())),
            lineage=_lineage("FMP", normalized),
        )


class FakePolygonByTicker:
    def __init__(self, bars_by_ticker, error_tickers=None):
        self.bars_by_ticker = bars_by_ticker
        self.error_tickers = {ticker.upper() for ticker in (error_tickers or set())}
        self.calls = []

    def get_minute_aggs(self, ticker, from_date, to_date, **kwargs):
        self.calls.append((ticker, from_date, to_date, dict(kwargs)))
        normalized = ticker.upper()
        if normalized in self.error_tickers:
            return _provider_error_response("Polygon", normalized, "minute_fetch_failed")
        return AdapterResponse(
            data=list(self.bars_by_ticker.get(normalized, _polygon_bars())),
            lineage=_lineage("Polygon", normalized),
        )


def test_pit_candidate_uses_only_decision_time_evidence():
    daily = _daily_bars(day_volume=1_000_000_000)
    minutes = _minute_bars()
    result = build_i12_pit_candidate(
        ticker="PIT",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=daily,
        minute_bars=minutes,
        daily_source_hash="daily-a",
        minute_source_hash="minute-a",
    )

    assert result.candidate_status == "passed"
    assert result.coverage_status == "ok"
    assert result.feature_asof_ts == DECISION_TS
    assert result.feature_json["feature_asof_ts"] == DECISION_TS.isoformat()
    assert result.feature_json["completed_through_ts"] == DECISION_TS.isoformat()
    assert result.leakage_guard["feature_asof_ts"] == DECISION_TS.isoformat()
    assert result.leakage_guard["completed_through_ts"] == DECISION_TS.isoformat()
    assert result.leakage_guard["entry_quote_target_ts"] == DECISION_TS.isoformat()
    assert (
        result.leakage_guard["decision_time_semantics"]
        == "decision_after_prior_completed_minute_start_stamped_bars"
    )
    assert result.source_bars["minute_bar_count_before_decision"] == 10
    assert result.source_bars["expected_minute_bar_count_before_decision"] == 10
    assert result.source_bars["completed_minute_count"] == 10.0
    assert result.source_bars["minute_bar_last_ts"] == (
        DECISION_TS - timedelta(minutes=1)
    ).isoformat()
    assert result.source_bars["source_minute_bars_max_start_ts"] == (
        DECISION_TS - timedelta(minutes=1)
    ).isoformat()
    assert result.source_bars["completed_through_ts"] == DECISION_TS.isoformat()
    assert result.feature_json["source_minute_bars_max_start_ts"] == (
        DECISION_TS - timedelta(minutes=1)
    ).isoformat()
    assert result.leakage_guard["source_minute_bars_max_start_ts"] == (
        DECISION_TS - timedelta(minutes=1)
    ).isoformat()
    assert result.feature_json["decision_elapsed_minutes"] == 10.0
    assert result.feature_json["completed_minute_count"] == 10.0
    assert result.feature_json["projected_volume_at_decision"] == pytest.approx(9750.0)
    assert result.leakage_guard["uses_full_day_volume"] is False
    assert result.leakage_guard["uses_same_day_close"] is False
    assert result.leakage_guard["uses_full_day_high_low"] is False
    assert result.leakage_guard["uses_forward_bars"] is False
    feature_text = json.dumps(result.feature_json, sort_keys=True)
    assert "full_day" not in feature_text
    assert "same_day_close" not in feature_text
    assert "next_open" not in feature_text

    changed_full_day_volume = build_i12_pit_candidate(
        ticker="PIT",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(day_volume=1),
        minute_bars=minutes,
        daily_source_hash="daily-b",
        minute_source_hash="minute-a",
    )
    assert changed_full_day_volume.feature_json == result.feature_json
    assert changed_full_day_volume.gate_values == result.gate_values


def test_0935_decision_uses_five_completed_minutes_and_can_pass():
    decision_ts = datetime(2026, 6, 16, 13, 35, tzinfo=timezone.utc)
    result = build_i12_pit_candidate(
        ticker="PIT",
        trading_date=DAY,
        decision_ts=decision_ts,
        decision_time_label="09:35",
        daily_bars=_daily_bars(),
        minute_bars=_minute_bars(),
        daily_source_hash="daily",
        minute_source_hash="minute",
    )

    assert result.candidate_status == "passed"
    assert result.feature_asof_ts == decision_ts
    assert result.feature_json["feature_asof_ts"] == decision_ts.isoformat()
    assert result.feature_json["completed_through_ts"] == decision_ts.isoformat()
    assert result.source_bars["source_minute_bars_max_start_ts"] == (
        decision_ts - timedelta(minutes=1)
    ).isoformat()
    assert result.source_bars["minute_bar_count_before_decision"] == 5
    assert result.source_bars["completed_minute_count"] == 5.0
    assert result.feature_json["decision_elapsed_minutes"] == 5.0
    assert result.feature_json["projected_volume_at_decision"] == pytest.approx(9750.0)


def test_decision_time_excludes_named_start_stamped_minute_bar():
    baseline = build_i12_pit_candidate(
        ticker="PIT",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(),
        minute_bars=_minute_bars(),
        daily_source_hash="daily",
        minute_source_hash="minute",
    )
    changed_940 = _polygon_bars()
    changed_940[10] = PolygonBar(
        timestamp=changed_940[10].timestamp,
        open=99.0,
        high=100.0,
        low=98.0,
        close=99.5,
        volume=1_000_000,
    )
    modified = build_i12_pit_candidate(
        ticker="PIT",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(),
        minute_bars=_clean_minute_bars(DAY, changed_940),
        daily_source_hash="daily",
        minute_source_hash="minute",
    )

    assert modified.feature_json == baseline.feature_json
    assert modified.gate_values == baseline.gate_values


def test_missing_minute_bars_are_explicit_not_zero():
    result = build_i12_pit_candidate(
        ticker="MISS",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(),
        minute_bars=[],
        daily_source_hash="daily",
        minute_source_hash=None,
    )

    assert result.candidate_status == "failed"
    assert result.coverage_status == "missing_minute_bars"
    assert result.feature_json["missing_minute_bars"] is True
    assert "early_cumulative_volume" not in result.feature_json


def test_opening_minute_path_is_required():
    market_open = us_equity_session_open_timestamp(DAY)
    late_only = _clean_minute_bars(
        DAY,
        [
            PolygonBar(
                timestamp=int((market_open + timedelta(minutes=9)).timestamp() * 1000),
                open=4.1,
                high=4.2,
                low=4.0,
                close=4.15,
                volume=500,
            )
        ],
    )

    result = build_i12_pit_candidate(
        ticker="LATE",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(),
        minute_bars=late_only,
        daily_source_hash="daily",
        minute_source_hash="minute",
    )

    assert result.candidate_status == "failed"
    assert result.coverage_status == "missing_open_bar"
    assert result.fail_reason == "missing_open_bar"


def test_duplicate_minute_bars_fail_closed_before_projection():
    minutes = _minute_bars()
    duplicate_939 = minutes[9]
    inflated_duplicate = type(duplicate_939)(
        timestamp=duplicate_939.timestamp,
        minute_index=duplicate_939.minute_index,
        open=duplicate_939.open,
        high=duplicate_939.high,
        low=duplicate_939.low,
        close=duplicate_939.close,
        volume=1_000_000,
    )

    result = build_i12_pit_candidate(
        ticker="DUP",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(),
        minute_bars=minutes + [inflated_duplicate],
        daily_source_hash="daily",
        minute_source_hash="minute",
    )

    assert result.candidate_status == "failed"
    assert result.coverage_status == "duplicate_minute_bars"
    assert result.fail_reason == "duplicate_minute_bars_before_decision"
    assert "early_cumulative_volume" not in result.feature_json


def test_strict_mode_still_fails_partial_minute_paths():
    result = build_i12_pit_candidate(
        ticker="STRICT",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(),
        minute_bars=_sparse_minute_bars(),
        daily_source_hash="daily",
        minute_source_hash="minute",
        path_mode="strict_contiguous",
    )

    assert result.candidate_status == "failed"
    assert result.coverage_status == "partial_minute_path"
    assert result.feature_json["path_mode"] == "strict_contiguous"
    assert result.feature_json["missing_minute_count_before_decision"] == 8
    assert result.feature_asof_ts == DECISION_TS


def test_sparse_zero_fill_mode_passes_partial_no_trade_path():
    result = build_i12_pit_candidate(
        ticker="SPARSE",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(),
        minute_bars=_sparse_minute_bars(),
        daily_source_hash="daily",
        minute_source_hash="minute",
        path_mode="sparse_zero_fill",
    )

    assert result.candidate_status == "passed"
    assert result.coverage_status == "ok"
    assert result.feature_json["path_mode"] == "sparse_zero_fill"
    assert result.feature_json["early_cumulative_volume"] == 200
    assert result.feature_json["projected_volume_ratio_at_decision"] == pytest.approx(7.8)
    assert result.feature_json["zero_fill_projected_volume_ratio"] == pytest.approx(7.8)
    assert result.feature_json["zero_fill_imputed_minute_count"] == 8
    assert result.feature_json["path_coverage_ratio"] == pytest.approx(0.2)
    assert result.source_bars["path_diagnostics"]["missing_minute_offsets"] == [
        1, 2, 3, 4, 6, 7, 8, 9
    ]


def test_sparse_zero_fill_uses_only_past_observed_prices():
    sparse = build_i12_pit_candidate(
        ticker="SPARSE",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(),
        minute_bars=_sparse_minute_bars(include_decision_bar=True, decision_close=999.0),
        daily_source_hash="daily",
        minute_source_hash="minute",
        path_mode="sparse_zero_fill",
    )
    baseline = build_i12_pit_candidate(
        ticker="SPARSE",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(),
        minute_bars=_sparse_minute_bars(include_decision_bar=True, decision_close=4.0),
        daily_source_hash="daily",
        minute_source_hash="minute",
        path_mode="sparse_zero_fill",
    )

    assert sparse.feature_json == baseline.feature_json
    assert sparse.gate_values == baseline.gate_values
    assert sparse.source_bars["path_diagnostics"] == baseline.source_bars["path_diagnostics"]


def test_sparse_zero_fill_still_fails_duplicate_and_no_predecision_bars():
    duplicate = _sparse_minute_bars()
    duplicate = duplicate + [duplicate[0]]
    duplicate_result = build_i12_pit_candidate(
        ticker="DUPSPARSE",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(),
        minute_bars=duplicate,
        daily_source_hash="daily",
        minute_source_hash="minute",
        path_mode="sparse_zero_fill",
    )
    no_bar_result = build_i12_pit_candidate(
        ticker="NOSPARSE",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(),
        minute_bars=[],
        daily_source_hash="daily",
        minute_source_hash="minute",
        path_mode="sparse_zero_fill",
    )

    assert duplicate_result.coverage_status == "duplicate_minute_bars"
    assert no_bar_result.coverage_status == "missing_minute_bars"


def test_candidate_content_hash_differs_between_strict_and_sparse_modes():
    strict = build_i12_pit_candidate(
        ticker="HASH",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(),
        minute_bars=_minute_bars(),
        daily_source_hash="daily",
        minute_source_hash="minute",
        path_mode="strict_contiguous",
    )
    sparse = build_i12_pit_candidate(
        ticker="HASH",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(),
        minute_bars=_minute_bars(),
        daily_source_hash="daily",
        minute_source_hash="minute",
        path_mode="sparse_zero_fill",
    )

    assert strict.content_hash != sparse.content_hash
    assert strict.candidate_attempt_hash != sparse.candidate_attempt_hash


def test_quote_replay_selects_candidate_event_window_and_fails_closed():
    candidate = _candidate_row()
    windows = quote_windows_for_candidate(candidate)
    assert [window.quote_role for window in windows] == [
        "entry",
        "same_day_exit",
        "next_open_exit",
    ]
    assert all(
        (window.window_end_ts - window.window_start_ts).total_seconds() <= 130
        for window in windows
    )

    stale_quote = _quote(
        ts=windows[0].target_ts - timedelta(seconds=90),
        bid=10,
        ask=10.05,
        ask_size=100,
    )
    result = replay_quote_window(
        ticker="PIT",
        window=windows[0],
        response=AdapterResponse(data=[stale_quote], lineage=_lineage("Alpaca", "PIT")),
        feed="sip",
        max_quote_age_seconds=60,
    )
    assert result.coverage_status == "stale"

    missing = replay_quote_window(
        ticker="PIT",
        window=windows[0],
        response=AdapterResponse(data=[], lineage=_lineage("Alpaca", "PIT")),
        feed="sip",
        max_quote_age_seconds=60,
    )
    assert missing.coverage_status == "missing"


def test_quote_replay_marks_truncated_historical_window_as_error():
    candidate = _candidate_row()
    window = quote_windows_for_candidate(candidate)[0]
    response = AdapterResponse(
        data=None,
        lineage=_lineage("Alpaca", "PIT"),
        error=ProviderError(
            provider="Alpaca",
            endpoint="/v2/stocks/PIT/quotes",
            status_code=None,
            error_type="historical_quote_window_truncated",
            message="truncated",
            retryable=True,
        ),
    )

    result = replay_quote_window(
        ticker="PIT",
        window=window,
        response=response,
        feed="sip",
        max_quote_age_seconds=60,
    )

    assert result.coverage_status == "error"
    assert result.error_json["error_type"] == "historical_quote_window_truncated"
    assert result.quote is None


def test_quote_cost_replay_skips_spread_size_and_keeps_denominator(db_session):
    candidate = _persist_candidate(db_session, "COST")
    entry = _persist_quote(
        db_session,
        candidate,
        role="entry",
        bid=10.0,
        ask=10.05,
        ask_size=10,
        status="ok",
    )
    same_day = _persist_quote(
        db_session,
        candidate,
        role="same_day_exit",
        bid=10.5,
        ask=10.55,
        ask_size=10,
        status="ok",
    )
    next_open = _persist_quote(
        db_session,
        candidate,
        role="next_open_exit",
        bid=9.0,
        ask=9.5,
        ask_size=10,
        status="ok",
    )

    good = evaluate_quote_cost_replay(
        entry_quote=entry,
        exit_quote=same_day,
        exit_role="same_day_exit",
        intended_order_usd=50,
        max_spread_bps=200,
        slippage_bps=10,
    )
    assert good.tradeability_status == "tradeable"
    assert good.modeled_return > 0

    wide = evaluate_quote_cost_replay(
        entry_quote=entry,
        exit_quote=next_open,
        exit_role="next_open_exit",
        intended_order_usd=50,
        max_spread_bps=10,
        slippage_bps=0,
    )
    assert wide.tradeability_status == "skipped_cash"
    assert wide.skipped_reason == "spread"
    assert wide.modeled_return == 0.0

    _persist_cost(db_session, candidate, "same_day_exit", good)
    _persist_cost(db_session, candidate, "next_open_exit", wide)
    report = i12_pit_rebuild_report(db_session)
    assert report["exit_metrics"]["same_day_exit"]["row_count"] == 1
    assert report["exit_metrics"]["next_open_exit"]["row_count"] == 1
    assert report["exit_metrics"]["next_open_exit"]["skipped_cash_count"] == 1
    assert report["skip_reason_counts"]["spread"] == 1


def test_exit_side_liquidity_uses_bid_size_and_does_not_realize_skipped_exit(db_session):
    candidate = _persist_candidate(db_session, "BIDDEPTH")
    entry = _persist_quote(
        db_session,
        candidate,
        role="entry",
        bid=10.0,
        ask=10.05,
        ask_size=100,
    )
    exit_quote = _persist_quote(
        db_session,
        candidate,
        role="same_day_exit",
        bid=10.5,
        ask=10.55,
        bid_size=0,
        ask_size=100,
        status="ok",
    )

    result = evaluate_quote_cost_replay(
        entry_quote=entry,
        exit_quote=exit_quote,
        exit_role="same_day_exit",
        intended_order_usd=50,
        max_spread_bps=200,
        slippage_bps=0,
    )

    assert exit_quote.bid_notional == 0.0
    assert exit_quote.ask_notional > 50
    assert result.tradeability_status == "skipped_cash"
    assert result.skipped_reason == "size"
    assert result.exit_bid is None
    assert result.modeled_return == 0.0


@pytest.mark.parametrize(
    ("quote_kwargs", "expected_reason"),
    [
        ({"bid": 10.0, "ask": 10.8, "bid_size": 100}, "spread"),
        ({"bid": 10.5, "ask": 10.0, "bid_size": 100}, "exit_quote_invalid"),
        ({"bid": 10.5, "ask": 10.55, "bid_size": 100, "conditions": ["H"]}, "halt_or_condition_uncertain"),
    ],
)
def test_exit_quote_fail_closed_conditions(db_session, quote_kwargs, expected_reason):
    candidate = _persist_candidate(db_session, f"EXIT{expected_reason}")
    entry = _persist_quote(db_session, candidate, role="entry", ask_size=100)
    exit_quote = _persist_quote(
        db_session,
        candidate,
        role="next_open_exit",
        status="ok",
        **quote_kwargs,
    )

    result = evaluate_quote_cost_replay(
        entry_quote=entry,
        exit_quote=exit_quote,
        exit_role="next_open_exit",
        intended_order_usd=50,
        max_spread_bps=200,
        slippage_bps=0,
    )

    assert result.tradeability_status == "skipped_cash"
    assert result.skipped_reason == expected_reason
    assert result.exit_bid is None


def test_duplicate_quote_roles_do_not_mask_missing_roles(db_session):
    candidate = _persist_candidate(db_session, "DUPQ")
    _persist_quote(db_session, candidate, role="entry", bid=10.0, ask=10.05)
    _persist_quote(db_session, candidate, role="entry", bid=10.01, ask=10.06)
    _persist_quote(db_session, candidate, role="same_day_exit", bid=10.5, ask=10.55)
    _persist_quote(db_session, candidate, role="same_day_exit", bid=10.51, ask=10.56)

    report = i12_pit_rebuild_report(db_session)

    assert report["quote_replay_complete"] is False
    assert report["missing_quote_role_count"] == 1
    assert report["duplicate_quote_role_count"] == 2
    assert report["candidate_complete_quote_count"] == 0
    assert report["candidate_incomplete_quote_count"] == 1
    assert report["quote_coverage_by_role"]["next_open_exit"]["missing"] == 1


def test_job_replays_only_passing_candidate_event_windows(db_session):
    db_session.add(
        HistoricalUniverseReconstruction(
            replay_date=DAY,
            ticker="PIT",
            normalized_symbol="PIT",
            inclusion_status="included",
            source="fixture",
            source_provenance_json="{}",
            reconstruction_method="fixture",
            pit_filter_status_json="{}",
            input_hash="hur-input",
            output_hash="hur-output",
        )
    )
    entry_target = DECISION_TS
    same_day_target = datetime(2026, 6, 16, 19, 55, tzinfo=timezone.utc)
    next_open_target = datetime(2026, 6, 17, 13, 31, tzinfo=timezone.utc)
    alpaca = FakeAlpaca({
        "entry": [_quote(ts=entry_target, bid=10, ask=10.05, ask_size=100)],
        "same_day_exit": [_quote(ts=same_day_target, bid=10.5, ask=10.55, ask_size=100)],
        "next_open_exit": [_quote(ts=next_open_target, bid=10.2, ask=10.25, ask_size=100)],
    })
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_polygon_bars()),
        alpaca_adapter=alpaca,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        intended_order_usd=250,
        quote_replay=True,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    assert db_session.query(I12PitCandidate).count() == 1
    assert db_session.query(I12PitQuoteReplay).count() == 3
    assert db_session.query(I12PitCostReplay).count() == 2
    assert len(alpaca.calls) == 3
    assert {call[3] for call in alpaca.calls} == {"sip"}
    assert result.metrics["quote_replay_complete"] is True
    assert result.metrics["training_status"] == "eligible_for_retrain_evaluation"


def test_quote_fetch_watchdog_timeout_is_persisted_as_quote_error(
    db_session,
    monkeypatch,
):
    _add_hur(db_session, "QWATCH", output_hash="hur-quote-watchdog")

    def timeout_quotes(func, *, thread_name, **kwargs):
        if "alpaca-quote" in thread_name:
            raise i12_pit_rebuild.FuturesTimeoutError()
        return func()

    monkeypatch.setattr(i12_pit_rebuild, "call_with_daemon_deadline", timeout_quotes)
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_polygon_bars()),
        alpaca_adapter=FakeAlpaca({}),
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        intended_order_usd=250,
        quote_replay=True,
        fetch_deadline_seconds=1,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    assert db_session.in_transaction() is False
    quotes = db_session.query(I12PitQuoteReplay).all()
    assert len(quotes) == 3
    assert {row.coverage_status for row in quotes} == {"error"}
    assert {
        json.loads(row.error_json)["error_type"] for row in quotes
    } == {"watchdog_timeout"}
    assert db_session.query(I12PitCostReplay).count() == 2
    assert result.metrics["quote_replay_complete"] is True
    assert result.metrics["cost_replay_complete"] is True
    assert result.metrics["quote_ok_count"] == 0
    assert result.metrics["quote_non_ok_count"] == 3
    assert result.metrics["quote_ok_rate"] == 0.0
    assert result.metrics["quote_coverage_rate"] == 1.0
    assert result.metrics["training_status"] == "eligible_for_retrain_evaluation"
    assert result.metrics["exit_metrics"]["same_day_exit"]["tradeable_count"] == 0


def test_quote_fetch_breaker_opening_timeout_is_persisted_and_resets_alpaca(
    db_session,
):
    _add_hur(db_session, "QBREAK", output_hash="hur-quote-breaker")
    alpaca = SleepyAlpaca({}, delay_seconds=0.05)
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_polygon_bars()),
        alpaca_adapter=alpaca,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        intended_order_usd=250,
        quote_replay=True,
        fetch_deadline_seconds=0.001,
        max_consecutive_fetch_timeouts=1,
    )

    result = run_job(db_session, job, params={"test": True})

    assert not result.ok
    assert db_session.in_transaction() is False
    assert alpaca.reset_calls == 1
    quotes = db_session.query(I12PitQuoteReplay).all()
    assert len(quotes) == 1
    quote = quotes[0]
    assert quote.quote_role == "entry"
    assert quote.coverage_status == "error"
    error = json.loads(quote.error_json)
    assert error["error_type"] == "watchdog_timeout"
    flags = error["data_quality_flags"]
    breaker = flags["provider_outage_circuit_breaker"]
    assert breaker["breaker_opened_by_current_call"] is True
    assert breaker["circuit_reason"] == "watchdog_timeout:max_consecutive_timeouts"
    assert db_session.query(I12PitCandidate).one().candidate_status == "passed"


def test_i12_pit_job_does_not_call_providers_inside_open_db_transaction(db_session):
    _add_hur(db_session, "TXPROV1", output_hash="hur-tx-provider-1")
    _add_hur(db_session, "TXPROV2", output_hash="hur-tx-provider-2")
    quotes = _complete_quotes()
    alpaca = AssertingNoTransactionAlpaca(quotes, db_session)
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=AssertingNoTransactionFmp(_fmp_bars(), db_session),
        polygon_adapter=AssertingNoTransactionPolygon(_polygon_bars(), db_session),
        alpaca_adapter=alpaca,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        intended_order_usd=250,
        quote_replay=True,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    assert db_session.in_transaction() is False
    assert db_session.query(I12PitCandidate).count() == 2
    assert db_session.query(I12PitQuoteReplay).count() == 6
    assert db_session.query(I12PitCostReplay).count() == 4
    assert len(alpaca.calls) == 6


def test_clean_zero_pit_candidates_reports_explicit_zero_candidate_status(db_session):
    _add_hur(db_session, "LOWVOL", output_hash="hur-lowvol")
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_polygon_bars(volume=1)),
        alpaca_adapter=FailingAlpaca(),
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        intended_order_usd=250,
        quote_replay=True,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    assert result.metrics["expected_candidate_attempts"] == 1
    assert result.metrics["actual_candidate_row_count"] == 1
    assert result.metrics["source_replay_complete"] is True
    assert result.metrics["pit_candidate_count"] == 0
    assert result.metrics["candidate_status_counts"] == {"failed": 1}
    assert result.metrics["quote_replay_status"] == "not_applicable"
    assert result.metrics["cost_replay_status"] == "not_applicable"
    assert result.metrics["training_status"] == "blocked_zero_pit_candidates"
    assert result.metrics["ml_ranking_status"] == "blocked_zero_pit_candidates"
    assert result.metrics["conclusions_final"] is False


def test_daily_prefilter_skip_avoids_minute_fetch_and_counts_source_attempt(db_session):
    _add_hur(db_session, "NODRAW", output_hash="hur-no-drawdown")
    polygon = FakePolygonByTicker({"NODRAW": _polygon_bars()})
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmpByTicker({"NODRAW": _fmp_bars_no_drawdown()}),
        polygon_adapter=polygon,
        alpaca_adapter=None,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        quote_replay=False,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    assert polygon.calls == []
    row = db_session.query(I12PitCandidate).one()
    assert row.candidate_status == "failed"
    assert row.coverage_status == "daily_prefilter_skip"
    assert row.fail_reason == "drawdown"
    assert result.metrics["expected_candidate_attempts"] == 1
    assert result.metrics["actual_candidate_row_count"] == 1
    assert result.metrics["missing_source_attempt_count"] == 0
    assert result.metrics["extra_source_attempt_count"] == 0
    assert result.metrics["missing_source_attempt_identity_count"] == 0
    assert result.metrics["extra_source_attempt_identity_count"] == 0
    assert result.metrics["source_replay_complete"] is True
    assert result.metrics["candidate_coverage_status_counts"] == {
        "daily_prefilter_skip": 1,
    }


def test_daily_cache_reuses_fmp_and_preserves_passed_candidate_identity(db_session):
    day2 = next_us_equity_session(DAY + timedelta(days=1))
    _add_hur(db_session, "CACHE", day=DAY, output_hash="hur-cache-day1")
    _add_hur(db_session, "CACHE", day=day2, output_hash="hur-cache-day2")
    fmp = FakeFmpByTicker({"CACHE": _fmp_bars()})
    polygon = FakePolygonByTicker({"CACHE": _polygon_bars()})
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=fmp,
        polygon_adapter=polygon,
        alpaca_adapter=None,
        start_date=DAY,
        end_date=day2,
        decision_times=["09:40"],
        quote_replay=False,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    assert len(fmp.calls) == 1
    assert len(polygon.calls) == 1
    assert result.metrics["expected_candidate_attempts"] == 2
    assert result.metrics["actual_candidate_row_count"] == 2
    assert result.metrics["missing_source_attempt_count"] == 0
    assert result.metrics["missing_source_attempt_identity_count"] == 0
    passed = (
        db_session.query(I12PitCandidate)
        .filter(
            I12PitCandidate.ticker == "CACHE",
            I12PitCandidate.decision_date == DAY,
        )
        .one()
    )
    stored_source_bars = json.loads(passed.source_bars_json)
    daily_bars = i12_pit_rebuild._daily_bars_for_trading_date(
        "CACHE",
        DAY,
        _fmp_bars(),
    )
    expected = build_i12_pit_candidate(
        ticker="CACHE",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=daily_bars,
        minute_bars=_minute_bars(),
        daily_source_hash=stored_source_bars["daily_source_hash"],
        minute_source_hash=stored_source_bars["minute_source_hash"],
        source_hur_identity_hash=stored_source_bars["source_hur_identity_hash"],
        source_hur=stored_source_bars["source_hur"],
        path_mode="strict_contiguous",
    )
    assert passed.candidate_status == "passed"
    assert passed.content_hash == expected.content_hash
    assert json.loads(passed.feature_json) == expected.feature_json
    skipped = (
        db_session.query(I12PitCandidate)
        .filter(
            I12PitCandidate.ticker == "CACHE",
            I12PitCandidate.decision_date == day2,
        )
        .one()
    )
    assert skipped.coverage_status == "daily_prefilter_skip"


def test_daily_fetch_error_attempts_are_persisted_and_block_finality(db_session):
    _add_hur(db_session, "BADFMP", output_hash="hur-bad-fmp")
    _add_hur(db_session, "GOODFMP", output_hash="hur-good-fmp")
    entry_target = DECISION_TS
    same_day_target = datetime(2026, 6, 16, 19, 55, tzinfo=timezone.utc)
    next_open_target = datetime(2026, 6, 17, 13, 31, tzinfo=timezone.utc)
    alpaca = FakeAlpaca({
        "entry": [_quote(ts=entry_target, bid=10, ask=10.05, ask_size=100)],
        "same_day_exit": [_quote(ts=same_day_target, bid=10.5, ask=10.55, ask_size=100)],
        "next_open_exit": [_quote(ts=next_open_target, bid=10.2, ask=10.25, ask_size=100)],
    })
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmpByTicker({"GOODFMP": _fmp_bars()}, error_tickers={"BADFMP"}),
        polygon_adapter=FakePolygonByTicker({"GOODFMP": _polygon_bars()}),
        alpaca_adapter=alpaca,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        intended_order_usd=250,
        quote_replay=True,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    assert db_session.query(I12PitCandidate).count() == 2
    failed = (
        db_session.query(I12PitCandidate)
        .filter(I12PitCandidate.ticker == "BADFMP")
        .one()
    )
    assert failed.coverage_status == "daily_fetch_error"
    assert json.loads(failed.error_json)["source_errors"]["daily_error"]["error_type"] == (
        "daily_fetch_failed"
    )
    assert result.metrics["expected_candidate_attempts"] == 2
    assert result.metrics["actual_candidate_row_count"] == 2
    assert result.metrics["missing_source_attempt_count"] == 0
    assert result.metrics["daily_fetch_error_count"] == 1
    assert result.metrics["quote_replay_complete"] is True
    assert result.metrics["cost_replay_complete"] is True
    assert result.metrics["data_integrity_passed"] is False
    assert result.metrics["conclusions_final"] is False
    assert result.metrics["training_status"] == "blocked_source_provider_errors"


def test_daily_fetch_watchdog_timeout_is_persisted_as_provider_error(
    db_session,
    monkeypatch,
):
    _add_hur(db_session, "FMPWATCH", output_hash="hur-fmp-watchdog")

    def timeout_daily(func, *, thread_name, **kwargs):
        del func, kwargs
        if "fmp-daily" in thread_name:
            raise i12_pit_rebuild.FuturesTimeoutError()
        raise AssertionError(f"unexpected provider fetch: {thread_name}")

    monkeypatch.setattr(i12_pit_rebuild, "call_with_daemon_deadline", timeout_daily)
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_polygon_bars()),
        alpaca_adapter=None,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        quote_replay=False,
        fetch_deadline_seconds=1,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    assert db_session.in_transaction() is False
    row = db_session.query(I12PitCandidate).one()
    assert row.coverage_status == "daily_fetch_error"
    error = json.loads(row.error_json)["source_errors"]["daily_error"]
    assert error["error_type"] == "watchdog_timeout"
    assert "1 seconds" in error["message"]
    assert result.metrics["daily_fetch_error_count"] == 1
    assert result.metrics["training_status"] == "blocked_source_provider_errors"


def test_daily_fetch_breaker_opening_timeout_is_persisted_as_provider_error(
    db_session,
):
    _add_hur(db_session, "FMPBREAK", output_hash="hur-fmp-breaker")
    fmp = SleepyFmp(_fmp_bars(), delay_seconds=0.05)
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=fmp,
        polygon_adapter=FakePolygon(_polygon_bars()),
        alpaca_adapter=None,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        quote_replay=False,
        fetch_deadline_seconds=0.001,
        max_consecutive_fetch_timeouts=1,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    assert db_session.in_transaction() is False
    assert fmp.reset_calls == 1
    row = db_session.query(I12PitCandidate).one()
    assert row.coverage_status == "daily_fetch_error"
    error = json.loads(row.error_json)["source_errors"]["daily_error"]
    assert error["error_type"] == "watchdog_timeout"
    flags = error["data_quality_flags"]
    breaker = flags["provider_outage_circuit_breaker"]
    assert breaker["breaker_opened_by_current_call"] is True
    assert breaker["circuit_reason"] == "watchdog_timeout:max_consecutive_timeouts"
    assert result.metrics["daily_fetch_error_count"] == 1
    assert result.metrics["training_status"] == "blocked_source_provider_errors"


def test_daily_fetch_outstanding_timeout_breaker_persists_provider_error(
    db_session,
):
    _add_hur(db_session, "FMPOUT", output_hash="hur-fmp-outstanding-breaker")
    fmp = SleepyFmp(_fmp_bars(), delay_seconds=0.05)
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=fmp,
        polygon_adapter=FakePolygon(_polygon_bars()),
        alpaca_adapter=None,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        quote_replay=False,
        fetch_deadline_seconds=0.001,
        max_outstanding_fetch_timeouts=1,
        max_consecutive_fetch_timeouts=10,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    assert db_session.in_transaction() is False
    assert fmp.reset_calls == 1
    row = db_session.query(I12PitCandidate).one()
    assert row.coverage_status == "daily_fetch_error"
    error = json.loads(row.error_json)["source_errors"]["daily_error"]
    assert error["error_type"] == "watchdog_timeout"
    flags = error["data_quality_flags"]
    breaker = flags["provider_outage_circuit_breaker"]
    assert breaker["breaker_opened_by_current_call"] is True
    assert breaker["circuit_reason"] == "watchdog_timeout:max_outstanding_timeouts"
    assert result.metrics["daily_fetch_error_count"] == 1
    assert result.metrics["training_status"] == "blocked_source_provider_errors"


def test_daily_fetch_error_recovery_supersedes_failed_attempt(db_session):
    _add_hur(db_session, "BADFMP", output_hash="hur-bad-fmp-recover")
    _add_hur(db_session, "GOODFMP", output_hash="hur-good-fmp-recover")
    quotes = _complete_quotes()

    first = run_job(
        db_session,
        I12PitRebuildJob(
            session=db_session,
            fmp_adapter=FakeFmpByTicker({"GOODFMP": _fmp_bars()}, error_tickers={"BADFMP"}),
            polygon_adapter=FakePolygonByTicker({"GOODFMP": _polygon_bars()}),
            alpaca_adapter=FakeAlpaca(quotes),
            start_date=DAY,
            end_date=DAY,
            decision_times=["09:40"],
            intended_order_usd=250,
            quote_replay=True,
        ),
        params={"test": True},
    )
    assert first.metrics["training_status"] == "blocked_source_provider_errors"

    second = run_job(
        db_session,
        I12PitRebuildJob(
            session=db_session,
            fmp_adapter=FakeFmpByTicker({
                "BADFMP": _fmp_bars(),
                "GOODFMP": _fmp_bars(),
            }),
            polygon_adapter=FakePolygonByTicker({
                "BADFMP": _polygon_bars(),
                "GOODFMP": _polygon_bars(),
            }),
            alpaca_adapter=FakeAlpaca(quotes),
            start_date=DAY,
            end_date=DAY,
            decision_times=["09:40"],
            intended_order_usd=250,
            quote_replay=True,
        ),
        params={"test": True},
    )

    assert second.ok
    assert second.metrics["expected_candidate_attempts"] == 2
    assert second.metrics["actual_candidate_row_count"] == 2
    assert second.metrics["daily_fetch_error_count"] == 0
    assert second.metrics["training_status"] == "eligible_for_retrain_evaluation"
    assert second.metrics["conclusions_final"] is True
    assert db_session.query(I12PitCandidate).count() == 3
    assert (
        db_session.query(I12PitCandidate)
        .filter(I12PitCandidate.is_active.is_(True))
        .count()
        == 2
    )
    superseded = (
        db_session.query(I12PitCandidate)
        .filter(
            I12PitCandidate.ticker == "BADFMP",
            I12PitCandidate.coverage_status == "daily_fetch_error",
        )
        .one()
    )
    assert superseded.is_active is False
    assert superseded.superseded_by_candidate_id is not None
    current_run_ids = {
        row.job_run_id
        for row in db_session.query(I12PitCandidate)
        .filter(I12PitCandidate.is_active.is_(True))
        .all()
    }
    assert len(current_run_ids) == 1
    scoped_report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
        job_run_id=next(iter(current_run_ids)),
    )
    assert scoped_report["expected_candidate_attempts"] == 2
    assert scoped_report["actual_candidate_row_count"] == 2
    assert scoped_report["training_status"] == "eligible_for_retrain_evaluation"


def test_minute_fetch_error_is_distinct_from_legitimate_missing_bars(db_session):
    _add_hur(db_session, "BADMIN", output_hash="hur-bad-minute")
    _add_hur(db_session, "GOODMIN", output_hash="hur-good-minute")
    entry_target = DECISION_TS
    same_day_target = datetime(2026, 6, 16, 19, 55, tzinfo=timezone.utc)
    next_open_target = datetime(2026, 6, 17, 13, 31, tzinfo=timezone.utc)
    alpaca = FakeAlpaca({
        "entry": [_quote(ts=entry_target, bid=10, ask=10.05, ask_size=100)],
        "same_day_exit": [_quote(ts=same_day_target, bid=10.5, ask=10.55, ask_size=100)],
        "next_open_exit": [_quote(ts=next_open_target, bid=10.2, ask=10.25, ask_size=100)],
    })
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmpByTicker({"BADMIN": _fmp_bars(), "GOODMIN": _fmp_bars()}),
        polygon_adapter=FakePolygonByTicker(
            {"GOODMIN": _polygon_bars()},
            error_tickers={"BADMIN"},
        ),
        alpaca_adapter=alpaca,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        intended_order_usd=250,
        quote_replay=True,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    failed = (
        db_session.query(I12PitCandidate)
        .filter(I12PitCandidate.ticker == "BADMIN")
        .one()
    )
    assert failed.coverage_status == "minute_fetch_error"
    assert failed.fail_reason == "minute_fetch_error"
    assert json.loads(failed.source_bars_json)["minute_error"]["error_type"] == (
        "minute_fetch_failed"
    )
    assert result.metrics["expected_candidate_attempts"] == 2
    assert result.metrics["actual_candidate_row_count"] == 2
    assert result.metrics["minute_fetch_error_count"] == 1
    assert result.metrics["data_integrity_passed"] is False
    assert result.metrics["conclusions_final"] is False
    assert result.metrics["training_status"] == "blocked_source_provider_errors"


def test_minute_fetch_watchdog_timeout_is_persisted_as_provider_error(
    db_session,
    monkeypatch,
):
    _add_hur(db_session, "MINWATCH", output_hash="hur-minute-watchdog")

    def timeout_minute(func, *, thread_name, **kwargs):
        if "polygon-minute" in thread_name:
            raise i12_pit_rebuild.FuturesTimeoutError()
        return func()

    monkeypatch.setattr(i12_pit_rebuild, "call_with_daemon_deadline", timeout_minute)
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_polygon_bars()),
        alpaca_adapter=None,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        quote_replay=False,
        fetch_deadline_seconds=1,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    assert db_session.in_transaction() is False
    row = db_session.query(I12PitCandidate).one()
    assert row.coverage_status == "minute_fetch_error"
    error = json.loads(row.source_bars_json)["minute_error"]
    assert error["error_type"] == "watchdog_timeout"
    assert result.metrics["minute_fetch_error_count"] == 1
    assert result.metrics["training_status"] == "blocked_source_provider_errors"


def test_minute_fetch_breaker_opening_timeout_is_persisted_as_provider_error(
    db_session,
):
    _add_hur(db_session, "MINBREAK", output_hash="hur-minute-breaker")
    polygon = SleepyPolygon(_polygon_bars(), delay_seconds=0.05)
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=polygon,
        alpaca_adapter=None,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        quote_replay=False,
        fetch_deadline_seconds=0.001,
        max_consecutive_fetch_timeouts=1,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    assert db_session.in_transaction() is False
    assert polygon.reset_calls == 1
    row = db_session.query(I12PitCandidate).one()
    assert row.coverage_status == "minute_fetch_error"
    error = json.loads(row.source_bars_json)["minute_error"]
    assert error["error_type"] == "watchdog_timeout"
    flags = error["data_quality_flags"]
    breaker = flags["provider_outage_circuit_breaker"]
    assert breaker["breaker_opened_by_current_call"] is True
    assert breaker["circuit_reason"] == "watchdog_timeout:max_consecutive_timeouts"
    assert result.metrics["minute_fetch_error_count"] == 1
    assert result.metrics["training_status"] == "blocked_source_provider_errors"


def test_minute_fetch_error_recovery_supersedes_failed_attempt(db_session):
    _add_hur(db_session, "BADMIN", output_hash="hur-bad-minute-recover")
    _add_hur(db_session, "GOODMIN", output_hash="hur-good-minute-recover")
    quotes = _complete_quotes()

    first = run_job(
        db_session,
        I12PitRebuildJob(
            session=db_session,
            fmp_adapter=FakeFmpByTicker({"BADMIN": _fmp_bars(), "GOODMIN": _fmp_bars()}),
            polygon_adapter=FakePolygonByTicker(
                {"GOODMIN": _polygon_bars()},
                error_tickers={"BADMIN"},
            ),
            alpaca_adapter=FakeAlpaca(quotes),
            start_date=DAY,
            end_date=DAY,
            decision_times=["09:40"],
            intended_order_usd=250,
            quote_replay=True,
        ),
        params={"test": True},
    )
    assert first.metrics["training_status"] == "blocked_source_provider_errors"

    second = run_job(
        db_session,
        I12PitRebuildJob(
            session=db_session,
            fmp_adapter=FakeFmpByTicker({"BADMIN": _fmp_bars(), "GOODMIN": _fmp_bars()}),
            polygon_adapter=FakePolygonByTicker({
                "BADMIN": _polygon_bars(),
                "GOODMIN": _polygon_bars(),
            }),
            alpaca_adapter=FakeAlpaca(quotes),
            start_date=DAY,
            end_date=DAY,
            decision_times=["09:40"],
            intended_order_usd=250,
            quote_replay=True,
        ),
        params={"test": True},
    )

    assert second.ok
    assert second.metrics["expected_candidate_attempts"] == 2
    assert second.metrics["actual_candidate_row_count"] == 2
    assert second.metrics["minute_fetch_error_count"] == 0
    assert second.metrics["training_status"] == "eligible_for_retrain_evaluation"
    assert (
        db_session.query(I12PitCandidate)
        .filter(I12PitCandidate.is_active.is_(False))
        .count()
        == 1
    )


def test_expected_attempts_are_hur_rows_times_decision_times(db_session):
    _add_hur(db_session, "PIT1", output_hash="hur-pit1")
    _add_hur(db_session, "PIT2", output_hash="hur-pit2")
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmpByTicker({"PIT1": _fmp_bars(), "PIT2": _fmp_bars()}),
        polygon_adapter=FakePolygonByTicker({"PIT1": _polygon_bars(), "PIT2": _polygon_bars()}),
        alpaca_adapter=None,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:35", "09:40"],
        quote_replay=False,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    assert result.metrics["hur_rows_loaded"] == 2
    assert result.metrics["decision_time_count"] == 2
    assert result.metrics["expected_candidate_attempts"] == 4
    assert result.metrics["actual_candidate_row_count"] == 4
    assert result.metrics["missing_source_attempt_count"] == 0


def test_default_job_rerun_reuses_existing_pit_rows(db_session):
    db_session.add(
        HistoricalUniverseReconstruction(
            replay_date=DAY,
            ticker="PIT",
            normalized_symbol="PIT",
            inclusion_status="included",
            source="fixture",
            source_provenance_json="{}",
            reconstruction_method="fixture",
            pit_filter_status_json="{}",
            input_hash="hur-input-rerun",
            output_hash="hur-output-rerun",
        )
    )
    entry_target = DECISION_TS
    same_day_target = datetime(2026, 6, 16, 19, 55, tzinfo=timezone.utc)
    next_open_target = datetime(2026, 6, 17, 13, 31, tzinfo=timezone.utc)
    quotes = {
        "entry": [_quote(ts=entry_target, bid=10, ask=10.05, ask_size=100)],
        "same_day_exit": [_quote(ts=same_day_target, bid=10.5, ask=10.55, ask_size=100)],
        "next_open_exit": [_quote(ts=next_open_target, bid=10.2, ask=10.25, ask_size=100)],
    }

    for _ in range(2):
        job = I12PitRebuildJob(
            session=db_session,
            fmp_adapter=FakeFmp(_fmp_bars()),
            polygon_adapter=FakePolygon(_polygon_bars()),
            alpaca_adapter=FakeAlpaca(quotes),
            start_date=DAY,
            end_date=DAY,
            decision_times=["09:40"],
            intended_order_usd=250,
            quote_replay=True,
        )
        result = run_job(db_session, job, params={"test": True})
        assert result.ok

    assert db_session.query(I12PitCandidate).count() == 1
    assert db_session.query(I12PitQuoteReplay).count() == 3
    assert db_session.query(I12PitCostReplay).count() == 2


def test_quote_error_rerun_recovers_without_duplicate_active_roles(db_session):
    _add_hur(db_session, "QREC", output_hash="hur-qrec")
    quotes = _complete_quotes()

    first = run_job(
        db_session,
        I12PitRebuildJob(
            session=db_session,
            fmp_adapter=FakeFmp(_fmp_bars()),
            polygon_adapter=FakePolygon(_polygon_bars()),
            alpaca_adapter=FakeAlpacaResponses({
                "entry": _provider_error_response("Alpaca", "QREC", "entry_quote_failed"),
                "same_day_exit": quotes["same_day_exit"],
                "next_open_exit": quotes["next_open_exit"],
            }),
            start_date=DAY,
            end_date=DAY,
            decision_times=["09:40"],
            intended_order_usd=250,
            quote_replay=True,
        ),
        params={"test": True},
    )
    assert first.ok
    assert first.metrics["quote_replay_complete"] is True
    assert first.metrics["quote_non_ok_count"] == 1
    assert first.metrics["quote_ok_rate"] == pytest.approx(2 / 3)
    assert first.metrics["training_status"] == "eligible_for_retrain_evaluation"

    second = run_job(
        db_session,
        I12PitRebuildJob(
            session=db_session,
            fmp_adapter=FakeFmp(_fmp_bars()),
            polygon_adapter=FakePolygon(_polygon_bars()),
            alpaca_adapter=FakeAlpaca(quotes),
            start_date=DAY,
            end_date=DAY,
            decision_times=["09:40"],
            intended_order_usd=250,
            quote_replay=True,
        ),
        params={"test": True},
    )

    assert second.ok
    assert second.metrics["quote_replay_complete"] is True
    assert second.metrics["cost_replay_complete"] is True
    assert second.metrics["duplicate_quote_role_count"] == 0
    assert second.metrics["duplicate_cost_role_count"] == 0
    assert second.metrics["quote_replay_row_count"] == 3
    assert second.metrics["historical_quote_replay_row_count"] == 4
    assert second.metrics["cost_replay_row_count"] == 2
    assert second.metrics["historical_cost_replay_row_count"] == 4
    assert second.metrics["training_status"] == "eligible_for_retrain_evaluation"

    old_entry_error = (
        db_session.query(I12PitQuoteReplay)
        .filter(
            I12PitQuoteReplay.quote_role == "entry",
            I12PitQuoteReplay.coverage_status != "ok",
        )
        .one()
    )
    assert old_entry_error.is_active is False
    active_entry = (
        db_session.query(I12PitQuoteReplay)
        .filter(
            I12PitQuoteReplay.quote_role == "entry",
            I12PitQuoteReplay.is_active.is_(True),
        )
        .one()
    )
    assert active_entry.coverage_status == "ok"
    inactive_skipped_costs = (
        db_session.query(I12PitCostReplay)
        .filter(
            I12PitCostReplay.is_active.is_(False),
            I12PitCostReplay.tradeability_status == "skipped_cash",
        )
        .count()
    )
    assert inactive_skipped_costs == 2


def test_candidate_supersession_inactivates_old_child_evidence(db_session):
    _add_hur(db_session, "SUPER", output_hash="hur-super")
    quotes = _complete_quotes()

    first = run_job(
        db_session,
        I12PitRebuildJob(
            session=db_session,
            fmp_adapter=FakeFmp(_fmp_bars()),
            polygon_adapter=FakePolygon(_polygon_bars(volume=25)),
            alpaca_adapter=FakeAlpaca(quotes),
            start_date=DAY,
            end_date=DAY,
            decision_times=["09:40"],
            intended_order_usd=250,
            quote_replay=True,
        ),
        params={"test": True},
    )
    assert first.metrics["training_status"] == "eligible_for_retrain_evaluation"
    first_candidate_id = (
        db_session.query(I12PitCandidate.i12_pit_candidate_id)
        .filter(I12PitCandidate.is_active.is_(True))
        .one()[0]
    )

    second = run_job(
        db_session,
        I12PitRebuildJob(
            session=db_session,
            fmp_adapter=FakeFmp(_fmp_bars()),
            polygon_adapter=FakePolygon(_polygon_bars(volume=30)),
            alpaca_adapter=FakeAlpaca(quotes),
            start_date=DAY,
            end_date=DAY,
            decision_times=["09:40"],
            intended_order_usd=250,
            quote_replay=True,
        ),
        params={"test": True},
    )

    assert second.ok
    assert second.metrics["training_status"] == "eligible_for_retrain_evaluation"
    assert second.metrics["quote_replay_row_count"] == 3
    assert second.metrics["historical_quote_replay_row_count"] == 6
    assert second.metrics["cost_replay_row_count"] == 2
    assert second.metrics["historical_cost_replay_row_count"] == 4
    assert second.metrics["active_quote_rows_with_inactive_candidate_count"] == 0
    assert second.metrics["active_cost_rows_with_inactive_candidate_count"] == 0
    assert (
        db_session.query(I12PitCandidate)
        .filter(I12PitCandidate.is_active.is_(True))
        .count()
        == 1
    )
    assert (
        db_session.query(I12PitCandidate)
        .filter(I12PitCandidate.is_active.is_(False))
        .count()
        == 1
    )
    assert (
        db_session.query(I12PitQuoteReplay)
        .filter(
            I12PitQuoteReplay.i12_pit_candidate_id == first_candidate_id,
            I12PitQuoteReplay.is_active.is_(True),
        )
        .count()
        == 0
    )
    assert (
        db_session.query(I12PitCostReplay)
        .filter(
            I12PitCostReplay.i12_pit_candidate_id == first_candidate_id,
            I12PitCostReplay.is_active.is_(True),
        )
        .count()
        == 0
    )


def test_failed_candidate_supersession_inactivates_old_child_evidence(db_session):
    _add_hur(db_session, "FAILSUPER", output_hash="hur-fail-super")
    quotes = _complete_quotes()

    first = run_job(
        db_session,
        I12PitRebuildJob(
            session=db_session,
            fmp_adapter=FakeFmp(_fmp_bars()),
            polygon_adapter=FakePolygon(_polygon_bars()),
            alpaca_adapter=FakeAlpaca(quotes),
            start_date=DAY,
            end_date=DAY,
            decision_times=["09:40"],
            intended_order_usd=250,
            quote_replay=True,
        ),
        params={"test": True},
    )
    assert first.metrics["training_status"] == "eligible_for_retrain_evaluation"

    second = run_job(
        db_session,
        I12PitRebuildJob(
            session=db_session,
            fmp_adapter=FakeFmpByTicker({}, error_tickers={"FAILSUPER"}),
            polygon_adapter=FakePolygon(_polygon_bars()),
            alpaca_adapter=FakeAlpaca(quotes),
            start_date=DAY,
            end_date=DAY,
            decision_times=["09:40"],
            intended_order_usd=250,
            quote_replay=True,
        ),
        params={"test": True},
    )

    assert second.ok
    assert second.metrics["training_status"] == "blocked_source_provider_errors"
    assert second.metrics["quote_replay_row_count"] == 0
    assert second.metrics["cost_replay_row_count"] == 0
    assert second.metrics["historical_quote_replay_row_count"] == 3
    assert second.metrics["historical_cost_replay_row_count"] == 2
    assert second.metrics["active_quote_rows_with_inactive_candidate_count"] == 0
    assert second.metrics["active_cost_rows_with_inactive_candidate_count"] == 0


def test_active_child_rows_with_inactive_candidate_fail_report(db_session):
    candidate = _persist_candidate(db_session, "ORPHAN")
    entry = _persist_quote(db_session, candidate, role="entry", status="ok")
    same_day = _persist_quote(db_session, candidate, role="same_day_exit", status="ok")
    next_open = _persist_quote(db_session, candidate, role="next_open_exit", status="ok")
    _persist_cost(
        db_session,
        candidate,
        "same_day_exit",
        evaluate_quote_cost_replay(
            entry_quote=entry,
            exit_quote=same_day,
            exit_role="same_day_exit",
            intended_order_usd=50,
            max_spread_bps=200,
        ),
    )
    _persist_cost(
        db_session,
        candidate,
        "next_open_exit",
        evaluate_quote_cost_replay(
            entry_quote=entry,
            exit_quote=next_open,
            exit_role="next_open_exit",
            intended_order_usd=50,
            max_spread_bps=200,
        ),
    )
    candidate.is_active = False
    candidate.superseded_at = datetime.now(timezone.utc)
    db_session.flush()

    report = i12_pit_rebuild_report(
        db_session,
        hur_rows_loaded=0,
        decision_time_count=1,
    )

    assert report["active_quote_rows_with_inactive_candidate_count"] == 3
    assert report["active_cost_rows_with_inactive_candidate_count"] == 2
    assert report["training_status"] == "blocked_child_evidence_parent_inactive"
    assert report["conclusions_final"] is False


def test_clean_quote_and_cost_rows_are_reused_without_provider_calls(db_session):
    _add_hur(db_session, "QREUSE", output_hash="hur-qreuse")
    quotes = _complete_quotes()

    first = run_job(
        db_session,
        I12PitRebuildJob(
            session=db_session,
            fmp_adapter=FakeFmp(_fmp_bars()),
            polygon_adapter=FakePolygon(_polygon_bars()),
            alpaca_adapter=FakeAlpaca(quotes),
            start_date=DAY,
            end_date=DAY,
            decision_times=["09:40"],
            intended_order_usd=250,
            quote_replay=True,
        ),
        params={"test": True},
    )
    assert first.metrics["training_status"] == "eligible_for_retrain_evaluation"

    second = run_job(
        db_session,
        I12PitRebuildJob(
            session=db_session,
            fmp_adapter=FakeFmp(_fmp_bars()),
            polygon_adapter=FakePolygon(_polygon_bars()),
            alpaca_adapter=FailingAlpaca(),
            start_date=DAY,
            end_date=DAY,
            decision_times=["09:40"],
            intended_order_usd=250,
            quote_replay=True,
        ),
        params={"test": True},
    )

    assert second.ok
    assert second.metrics["training_status"] == "eligible_for_retrain_evaluation"
    assert db_session.query(I12PitQuoteReplay).count() == 3
    assert db_session.query(I12PitCostReplay).count() == 2
    assert second.metrics["historical_quote_replay_row_count"] == 3
    assert second.metrics["historical_cost_replay_row_count"] == 2


def test_strict_and_sparse_modes_report_separately_and_sparse_replays_quotes(db_session):
    _add_hur(db_session, "MODE", output_hash="hur-mode")
    quotes = _complete_quotes()

    strict = run_job(
        db_session,
            I12PitRebuildJob(
                session=db_session,
                fmp_adapter=FakeFmp(_fmp_bars()),
                polygon_adapter=FakePolygon(_sparse_polygon_bars()),
                alpaca_adapter=FakeAlpaca(quotes),
                start_date=DAY,
                end_date=DAY,
            decision_times=["09:40"],
            intended_order_usd=250,
            quote_replay=True,
            minute_path_mode="strict_contiguous",
        ),
        params={"test": True},
    )
    assert strict.metrics["training_status"] == "blocked_zero_pit_candidates"

    sparse = run_job(
        db_session,
            I12PitRebuildJob(
                session=db_session,
                fmp_adapter=FakeFmp(_fmp_bars()),
                polygon_adapter=FakePolygon(_sparse_polygon_bars()),
                alpaca_adapter=FakeAlpaca(quotes),
                start_date=DAY,
                end_date=DAY,
            decision_times=["09:40"],
            intended_order_usd=250,
            quote_replay=True,
            minute_path_mode="sparse_zero_fill",
        ),
        params={"test": True},
    )

    assert sparse.ok
    assert sparse.metrics["report_path_mode"] == "sparse_zero_fill"
    assert sparse.metrics["available_path_modes"] == [
        "sparse_zero_fill",
        "strict_contiguous",
    ]
    assert sparse.metrics["mixed_path_modes_present"] is True
    assert sparse.metrics["candidate_counts_by_path_mode"] == {
        "sparse_zero_fill": 1,
    }
    assert "strict_contiguous" not in sparse.metrics["coverage_status_by_path_mode"]
    assert sparse.metrics["passed_candidates_by_path_mode"] == {"sparse_zero_fill": 1}
    assert sparse.metrics["strict_partial_rows_that_would_pass_sparse_zero_fill"] == 0
    assert "strict_contiguous" not in sparse.metrics["path_mode_metrics"]
    assert sparse.metrics["path_mode_metrics"]["sparse_zero_fill"]["training_status"] == (
        "eligible_for_retrain_evaluation"
    )
    assert sparse.metrics["path_mode_metrics"]["sparse_zero_fill"]["quote_replay_status"] == (
        "complete"
    )
    assert sparse.metrics["path_mode_metrics"]["sparse_zero_fill"]["cost_replay_status"] == (
        "complete"
    )
    assert sparse.metrics["sparse_imputation_distributions"]["missing_minute_count"]["mean"] == 8
    assert sparse.metrics["quote_replay_row_count"] == 3
    assert sparse.metrics["cost_replay_row_count"] == 2
    assert sparse.metrics["conclusions_final"] is True

    strict_report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
        path_mode="strict_contiguous",
    )
    assert strict_report["report_path_mode"] == "strict_contiguous"
    assert strict_report["candidate_counts_by_path_mode"] == {"strict_contiguous": 1}
    assert strict_report["training_status"] == "blocked_zero_pit_candidates"
    assert strict_report["conclusions_final"] is False

    default_report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
    )
    assert default_report["report_path_mode"] == "strict_contiguous"
    assert default_report["training_status"] == "blocked_zero_pit_candidates"

    compare_report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
        compare_path_modes=True,
    )
    assert compare_report["compare_path_modes"] is True
    assert compare_report["report_path_mode"] is None
    assert compare_report["expected_candidate_attempts"] == 2
    assert compare_report["candidate_counts_by_path_mode"] == {
        "sparse_zero_fill": 1,
        "strict_contiguous": 1,
    }
    assert compare_report["strict_partial_rows_that_would_pass_sparse_zero_fill"] == 1
    assert compare_report["path_mode_metrics"]["strict_contiguous"]["training_status"] == (
        "blocked_zero_pit_candidates"
    )
    assert compare_report["path_mode_metrics"]["sparse_zero_fill"]["training_status"] == (
        "eligible_for_retrain_evaluation"
    )
    assert compare_report["comparison_conclusions_final"] is False
    assert compare_report["conclusions_final"] is False
    assert compare_report["training_status"] == "blocked_path_mode_comparison_incomplete"


def test_sparse_mode_rerun_is_idempotent_per_path_mode(db_session):
    _add_hur(db_session, "SPARSERERUN", output_hash="hur-sparse-rerun")
    quotes = _complete_quotes()
    for adapter in (FakeAlpaca(quotes), FailingAlpaca()):
        result = run_job(
            db_session,
                I12PitRebuildJob(
                    session=db_session,
                    fmp_adapter=FakeFmp(_fmp_bars()),
                    polygon_adapter=FakePolygon(_sparse_polygon_bars()),
                    alpaca_adapter=adapter,
                    start_date=DAY,
                    end_date=DAY,
                decision_times=["09:40"],
                intended_order_usd=250,
                quote_replay=True,
                minute_path_mode="sparse_zero_fill",
            ),
            params={"test": True},
        )
        assert result.ok

    assert (
        db_session.query(I12PitCandidate)
        .filter(I12PitCandidate.path_mode == "sparse_zero_fill")
        .count()
        == 1
    )
    assert db_session.query(I12PitQuoteReplay).count() == 3
    assert db_session.query(I12PitCostReplay).count() == 2


def test_compare_path_modes_can_be_final_only_when_each_mode_is_complete(db_session):
    _add_hur(db_session, "BOTH", output_hash="hur-both")
    hur_row = (
        db_session.query(HistoricalUniverseReconstruction)
        .filter(HistoricalUniverseReconstruction.normalized_symbol == "BOTH")
        .one()
    )
    source_hur = _hur_source_row_from_model(hur_row, source_schema="public")
    _persist_complete_candidate_replay(
        db_session,
        "BOTH",
        path_mode="strict_contiguous",
        source_hur_identity_hash=source_hur.source_hur_identity_hash,
    )
    _persist_complete_candidate_replay(
        db_session,
        "BOTH",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash=source_hur.source_hur_identity_hash,
    )

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
        compare_path_modes=True,
    )

    assert report["expected_candidate_attempts"] == 2
    assert report["path_mode_metrics"]["strict_contiguous"]["training_status"] == (
        "eligible_for_retrain_evaluation"
    )
    assert report["path_mode_metrics"]["sparse_zero_fill"]["training_status"] == (
        "eligible_for_retrain_evaluation"
    )
    assert report["comparison_conclusions_final"] is True
    assert report["conclusions_final"] is True


def test_count_only_report_cannot_be_final(db_session):
    _persist_complete_candidate_replay(db_session, "COUNTONLY", path_mode="sparse_zero_fill")

    report = i12_pit_rebuild_report(
        db_session,
        hur_rows_loaded=1,
        decision_time_count=1,
        path_mode="sparse_zero_fill",
    )

    assert report["source_denominator_known"] is True
    assert report["source_identity_denominator_known"] is False
    assert report["source_replay_complete"] is False
    assert report["training_status"] == "blocked_source_identity_denominator_unknown"
    assert report["conclusions_final"] is False


def test_count_only_compare_report_cannot_be_final(db_session):
    _persist_complete_candidate_replay(db_session, "COUNTBOTH", path_mode="strict_contiguous")
    _persist_complete_candidate_replay(db_session, "COUNTBOTH", path_mode="sparse_zero_fill")

    report = i12_pit_rebuild_report(
        db_session,
        hur_rows_loaded=1,
        decision_time_count=1,
        compare_path_modes=True,
    )

    assert report["source_denominator_known"] is True
    assert report["source_identity_denominator_known"] is False
    assert report["source_replay_complete"] is False
    assert report["comparison_conclusions_final"] is False
    assert report["conclusions_final"] is False
    assert report["training_status"] == "blocked_source_identity_denominator_unknown"


def test_i12_pit_report_summarizer_extracts_exit_metrics(tmp_path):
    summarizer = _load_report_summarizer()
    report = {
        "report_path_mode": "strict_contiguous",
        "report_decision_time_labels": ["09:40"],
        "conclusions_final": True,
        "training_status": "eligible_for_retrain_evaluation",
        "pit_candidate_count": 12,
        "actual_candidate_row_count": 20,
        "quote_replay_complete": True,
        "cost_replay_complete": True,
        "quote_coverage_rate": 1.0,
        "expected_candidate_attempts": 20,
        "missing_source_attempt_count": 0,
        "extra_source_attempt_count": 0,
        "source_identity_denominator_known": True,
        "missing_source_attempt_identity_count": 0,
        "extra_source_attempt_identity_count": 0,
        "exit_metrics": {
            "same_day_exit": {
                "candidates": 12,
                "tradeable_count": 9,
                "tradeable_rate": 0.75,
                "skipped_cash_count": 3,
                "skipped_cash_by_reason": {"spread": 2, "size": 1},
                "mean_modeled_return_skips_as_cash": 0.012,
                "win_rate_skips_as_cash": 0.58,
                "spread_bps": {"p50": 90, "p75": 120, "p90": 180},
                "executable_notional": {"p50": 450, "p75": 700, "p90": 1000},
            },
            "next_open_exit": {
                "candidates": 12,
                "tradeable_count": 8,
                "tradeable_rate": 0.6667,
                "skipped_cash_count": 4,
                "skipped_cash_by_reason": {"quote_missing": 4},
                "mean_modeled_return_skips_as_cash": 0.004,
                "win_rate_skips_as_cash": 0.5,
                "spread_bps": {"p50": 100, "p75": 140, "p90": 220},
                "executable_notional": {"p50": 400, "p75": 650, "p90": 900},
            },
        },
        "path_mode_metrics": {
            "strict_contiguous": {
                "training_status": "eligible_for_retrain_evaluation",
                "conclusions_final": True,
                "candidate_count": 20,
                "passed_candidate_count": 12,
                "missing_source_attempt_count": 0,
                "extra_source_attempt_count": 0,
                "missing_source_attempt_identity_count": 0,
                "extra_source_attempt_identity_count": 0,
            }
        },
    }
    path = tmp_path / "strict_report.json"
    path.write_text(json.dumps(report))

    summary = summarizer.summarize_report_paths([path], labels=["strict"])[0]
    text = summarizer.render_text_table([summary])

    assert summary["label"] == "strict"
    assert summary["report_path_mode"] == "strict_contiguous"
    assert summary["source_replay"]["expected_candidate_attempts"] == 20
    assert summary["exits"]["same_day_exit"]["tradeable_count"] == 9
    assert summary["exits"]["same_day_exit"]["spread_bps_p90"] == 180
    assert summary["path_mode_metrics"]["strict_contiguous"]["candidate_count"] == 20
    assert "strict" in text
    assert "same_day_exit" in text
    assert "Path Modes" in text


def test_i12_pit_report_summarizer_handles_missing_file(tmp_path):
    summarizer = _load_report_summarizer()
    missing = tmp_path / "missing_report.json"

    summary = summarizer.summarize_report_paths([missing], labels=["missing"])[0]
    text = summarizer.render_text_table([summary])

    assert summary["label"] == "missing"
    assert summary["error"] == "missing_report_file"
    assert "missing_report_file" in text


def test_i12_pit_shard_launchers_preflight_once_and_harden_workers():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    for script_name in (
        "run_i12_pit_0940_strict_shards.sh",
        "run_i12_pit_0940_sparse_shards.sh",
    ):
        text_value = (scripts_dir / script_name).read_text()
        assert "--preflight-only" in text_value
        assert "run_schema_preflight" in text_value
        assert "REPLACE_STALE" in text_value
        assert "REPLACE_RUNNING" in text_value
        assert "ONLY_SHARD" in text_value
        assert "ONLY_WINDOW" in text_value
        assert "MAX_NO_PROGRESS_MINUTES=\"${MAX_NO_PROGRESS_MINUTES:-20}\"" in text_value
        assert "--max-no-progress-minutes" in text_value
        assert '"${MAX_NO_PROGRESS_MINUTES}"' in text_value
        assert "validate_replacement_scope" in text_value
        assert "validate_selector_matches" in text_value
        assert "selector matched zero shards" in text_value
        assert "print_valid_selectors" in text_value
        assert "matched_shards=" in text_value
        assert "launched_windows=" in text_value
        assert "skipped_running_windows=" in text_value
        assert "replaced_stale_windows=" in text_value
        assert "replaced_running_windows=" in text_value
        assert "shard_selected" in text_value
        assert "REPLACE_RUNNING=1 requires exactly one" in text_value
        assert "window_running_expected" in text_value
        assert "command_has_arg_value" in text_value
        assert "regex_escape" in text_value
        assert 'normalized_command="${command//\\\\ / }"' in text_value
        assert 'command_has_arg_value "${pane_start}" "--max-no-progress-minutes" "${MAX_NO_PROGRESS_MINUTES}"' in text_value
        assert 'pane_start}" == *"--max-no-progress-minutes"*' not in text_value
        assert "window already running expected shard" in text_value
        assert "replacing running expected shard window" in text_value
        assert "REPLACE_RUNNING=1" in text_value
        assert "REPLACE_STALE=1" in text_value
        assert "ERROR: .env is required" in text_value
        assert "set -Eeuo pipefail" in text_value
        assert "pane_current_command" in text_value
        assert "pane_start_command" in text_value
        worker_loop = text_value.rsplit('for shard in "${SHARDS[@]}"; do', 1)[1]
        assert "--create-tables" not in worker_loop


def test_i12_pit_shard_launcher_no_progress_match_is_exact():
    def matches_pane_start(command: str, expected_value: str) -> bool:
        normalized = command.replace("\\ ", " ")
        pattern = re.compile(
            rf"(^|\s)--max-no-progress-minutes(\s+|=){re.escape(expected_value)}($|\s)"
        )
        return bool(pattern.search(normalized))

    assert matches_pane_start(
        "python -m alpha.jobs.run_i12_pit_rebuild --max-no-progress-minutes 20 ",
        "20",
    )
    assert matches_pane_start(
        "python -m alpha.jobs.run_i12_pit_rebuild --max-no-progress-minutes=20 ",
        "20",
    )
    assert matches_pane_start(
        r"python\ -m\ alpha.jobs.run_i12_pit_rebuild\ --max-no-progress-minutes\ 20\ ",
        "20",
    )
    assert matches_pane_start(
        r"python\ -m\ alpha.jobs.run_i12_pit_rebuild\ --max-no-progress-minutes=20\ ",
        "20",
    )
    assert not matches_pane_start(
        r"python\ -m\ alpha.jobs.run_i12_pit_rebuild\ --max-no-progress-minutes\ 30\ ",
        "20",
    )
    assert not matches_pane_start(
        r"python\ -m\ alpha.jobs.run_i12_pit_rebuild\ --max-no-progress-minutes\ 120\ ",
        "20",
    )
    assert not matches_pane_start(
        r"python\ -m\ alpha.jobs.run_i12_pit_rebuild\ --max-no-progress-minutes\ 20.0\ ",
        "20",
    )
    assert not matches_pane_start(
        r"python\ -m\ alpha.jobs.run_i12_pit_rebuild\ --max-no-progress-minutes\ 20.5\ ",
        "20",
    )
    assert matches_pane_start(
        r"python\ -m\ alpha.jobs.run_i12_pit_rebuild\ --max-no-progress-minutes\ 20.0\ ",
        "20.0",
    )


def test_pit_required_column_preflight_covers_quote_and_cost_tables(monkeypatch):
    required = run_i12_pit_rebuild.I12_PIT_REBUILD_REQUIRED_COLUMNS
    assert "quote_size_basis" in required["i12_pit_quote_replays"]
    assert "cost_replay_attempt_hash" in required["i12_pit_cost_replays"]

    class FakeSession:
        def get_bind(self):
            return object()

    class FakeInspector:
        def __init__(self, missing_by_table):
            self.missing_by_table = missing_by_table

        def get_columns(self, table_name, schema=None):
            del schema
            return [
                {"name": column}
                for column in required[table_name]
                if column not in self.missing_by_table.get(table_name, set())
            ]

    def assert_missing(table_name, missing_column):
        monkeypatch.setattr(
            run_i12_pit_rebuild,
            "inspect",
            lambda bind: FakeInspector({table_name: {missing_column}}),
        )
        with pytest.raises(ValueError) as exc:
            run_i12_pit_rebuild._assert_required_pit_columns(
                FakeSession(),
                "scratch_old",
            )
        assert table_name in str(exc.value)
        assert missing_column in str(exc.value)
        assert "fresh scratch schema or migrate it" in str(exc.value)

    assert_missing("i12_pit_candidates", "candidate_attempt_hash")
    assert_missing("i12_pit_quote_replays", "quote_size_basis")
    assert_missing("i12_pit_cost_replays", "cost_replay_attempt_hash")


def test_pit_required_index_preflight_catches_missing_active_attempt_indexes(monkeypatch):
    required_columns = run_i12_pit_rebuild.I12_PIT_REBUILD_REQUIRED_COLUMNS
    required_indexes = run_i12_pit_rebuild.I12_PIT_REBUILD_REQUIRED_INDEXES

    class FakeSession:
        def get_bind(self):
            return object()

    class FakeInspector:
        def __init__(self, missing_by_table):
            self.missing_by_table = missing_by_table

        def get_columns(self, table_name, schema=None):
            del schema
            return [{"name": column} for column in required_columns[table_name]]

        def get_indexes(self, table_name, schema=None):
            del schema
            return [
                {"name": index}
                for index in required_indexes[table_name]
                if index not in self.missing_by_table.get(table_name, set())
            ]

    def assert_missing(table_name, missing_index):
        monkeypatch.setattr(
            run_i12_pit_rebuild,
            "inspect",
            lambda bind: FakeInspector({table_name: {missing_index}}),
        )
        with pytest.raises(ValueError) as exc:
            run_i12_pit_rebuild._assert_required_pit_columns(
                FakeSession(),
                "scratch_old",
            )
        assert table_name in str(exc.value)
        assert missing_index in str(exc.value)
        assert "without index" in str(exc.value)
        assert "fresh scratch schema or migrate it" in str(exc.value)

    assert_missing(
        "i12_pit_candidates",
        "ux_i12_pit_candidates_active_attempt",
    )
    assert_missing(
        "i12_pit_quote_replays",
        "ux_i12_pit_quote_replays_active_attempt",
    )
    assert_missing(
        "i12_pit_cost_replays",
        "ux_i12_pit_cost_replays_active_attempt",
    )


def test_report_bad_hur_source_returns_structured_blocked_report(db_session):
    _persist_complete_candidate_replay(db_session, "BADHUR", path_mode="sparse_zero_fill")
    db_session.execute(text("ATTACH DATABASE ':memory:' AS missing_hur"))

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="missing_hur",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
        path_mode="sparse_zero_fill",
    )

    assert report["source_identity_denominator_known"] is False
    assert report["source_identity_denominator_error"]
    assert report["source_replay_complete"] is False
    assert report["training_status"] == "blocked_source_identity_denominator_error"
    assert report["conclusions_final"] is False


def test_compare_path_modes_marks_expected_empty_mode_as_incomplete(db_session):
    _add_hur(db_session, "SPARSEONLY", output_hash="hur-sparse-only")
    _persist_complete_candidate_replay(
        db_session,
        "SPARSEONLY",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash=_hur_identity_hash(db_session, "SPARSEONLY"),
    )

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
        compare_path_modes=True,
    )

    assert report["path_mode_metrics"]["strict_contiguous"]["training_status"] == (
        "blocked_source_replay_incomplete"
    )
    assert report["path_mode_metrics"]["strict_contiguous"]["missing_source_attempt_count"] == 1
    assert (
        report["path_mode_metrics"]["strict_contiguous"][
            "missing_source_attempt_identity_count"
        ]
        == 1
    )
    assert report["path_mode_metrics"]["strict_contiguous"]["quote_replay_status"] == (
        "not_applicable"
    )
    assert report["path_mode_metrics"]["sparse_zero_fill"]["training_status"] == (
        "eligible_for_retrain_evaluation"
    )
    assert report["comparison_conclusions_final"] is False
    assert report["conclusions_final"] is False


def test_extra_active_candidate_blocks_finality(db_session):
    _add_hur(db_session, "EXTRA1", output_hash="hur-extra-1")
    _persist_complete_candidate_replay(
        db_session,
        "EXTRA1",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash=_hur_identity_hash(db_session, "EXTRA1"),
    )
    _persist_complete_candidate_replay(
        db_session,
        "EXTRA2",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash="hur-extra-b",
    )

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
        path_mode="sparse_zero_fill",
    )

    assert report["expected_candidate_attempts"] == 1
    assert report["candidate_row_count"] == 2
    assert report["missing_source_attempt_count"] == 0
    assert report["extra_source_attempt_count"] == 1
    assert report["source_replay_complete"] is False
    assert report["data_integrity_passed"] is False
    assert report["conclusions_final"] is False
    assert report["training_status"] == "blocked_source_replay_extra_active_attempts"


def test_extra_active_candidate_blocks_path_mode_finality(db_session):
    _add_hur(db_session, "PATHEXTRA", output_hash="hur-path-extra")
    _persist_complete_candidate_replay(
        db_session,
        "PATHEXTRA",
        path_mode="strict_contiguous",
        source_hur_identity_hash=_hur_identity_hash(db_session, "PATHEXTRA"),
    )
    _persist_complete_candidate_replay(
        db_session,
        "PATHEXTRA",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash=_hur_identity_hash(db_session, "PATHEXTRA"),
    )
    _persist_complete_candidate_replay(
        db_session,
        "SPARSEEXTRA2",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash="hur-sparse-b",
    )

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
        compare_path_modes=True,
    )

    sparse_metrics = report["path_mode_metrics"]["sparse_zero_fill"]
    assert sparse_metrics["extra_source_attempt_count"] == 1
    assert sparse_metrics["conclusions_final"] is False
    assert sparse_metrics["training_status"] == "blocked_source_replay_extra_active_attempts"
    assert report["comparison_conclusions_final"] is False
    assert report["conclusions_final"] is False
    assert report["training_status"] == "blocked_source_replay_extra_active_attempts"


def test_compare_path_modes_offsetting_missing_and_extra_not_source_complete(db_session):
    _add_hur(db_session, "OFFSET1", output_hash="hur-offset-1")
    _persist_complete_candidate_replay(
        db_session,
        "OFFSET1",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash=_hur_identity_hash(db_session, "OFFSET1"),
    )
    _persist_complete_candidate_replay(
        db_session,
        "OFFSET2",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash="hur-offset-b",
    )

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
        compare_path_modes=True,
    )

    assert report["expected_candidate_attempts"] == 2
    assert report["candidate_row_count"] == 2
    assert report["missing_source_attempt_count"] == 0
    assert report["extra_source_attempt_count"] == 0
    assert report["path_mode_metrics"]["strict_contiguous"]["missing_source_attempt_count"] == 1
    assert report["path_mode_metrics"]["sparse_zero_fill"]["extra_source_attempt_count"] == 1
    assert report["source_replay_complete"] is False
    assert report["comparison_conclusions_final"] is False
    assert report["conclusions_final"] is False
    assert report["training_status"] == "blocked_source_identity_mismatch"


def test_same_count_wrong_hur_identity_blocks_source_replay(db_session):
    _add_hur(db_session, "STALEID", output_hash="hur-current")
    _persist_complete_candidate_replay(
        db_session,
        "STALEID",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash="stale-hur-identity",
    )

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
        path_mode="sparse_zero_fill",
    )

    assert report["expected_candidate_attempts"] == 1
    assert report["candidate_row_count"] == 1
    assert report["missing_source_attempt_count"] == 0
    assert report["extra_source_attempt_count"] == 0
    assert report["source_identity_denominator_known"] is True
    assert report["expected_source_attempt_identity_count"] == 1
    assert report["actual_source_attempt_identity_count"] == 1
    assert report["missing_source_attempt_identity_count"] == 1
    assert report["extra_source_attempt_identity_count"] == 1
    assert report["source_replay_complete"] is False
    assert report["conclusions_final"] is False
    assert report["training_status"] == "blocked_source_identity_mismatch"


def test_exact_hur_identity_source_replay_passes(db_session):
    _add_hur(db_session, "EXACTID", output_hash="hur-exact")
    hur_row = (
        db_session.query(HistoricalUniverseReconstruction)
        .filter(HistoricalUniverseReconstruction.normalized_symbol == "EXACTID")
        .one()
    )
    source_hur = _hur_source_row_from_model(hur_row, source_schema="public")
    _persist_complete_candidate_replay(
        db_session,
        "EXACTID",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash=source_hur.source_hur_identity_hash,
    )

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
        path_mode="sparse_zero_fill",
    )

    assert report["source_identity_denominator_known"] is True
    assert report["missing_source_attempt_identity_count"] == 0
    assert report["extra_source_attempt_identity_count"] == 0
    assert report["source_replay_complete"] is True
    assert report["training_status"] == "eligible_for_retrain_evaluation"
    assert report["conclusions_final"] is True


def test_path_mode_identity_mismatch_blocks_only_stale_mode(db_session):
    _add_hur(db_session, "PMID", output_hash="hur-pm")
    hur_row = (
        db_session.query(HistoricalUniverseReconstruction)
        .filter(HistoricalUniverseReconstruction.normalized_symbol == "PMID")
        .one()
    )
    source_hur = _hur_source_row_from_model(hur_row, source_schema="public")
    _persist_complete_candidate_replay(
        db_session,
        "PMID",
        path_mode="strict_contiguous",
        source_hur_identity_hash=source_hur.source_hur_identity_hash,
    )
    _persist_complete_candidate_replay(
        db_session,
        "PMID",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash="stale-sparse-hur",
    )

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
        compare_path_modes=True,
    )

    assert report["path_mode_metrics"]["strict_contiguous"]["conclusions_final"] is True
    sparse = report["path_mode_metrics"]["sparse_zero_fill"]
    assert sparse["missing_source_attempt_identity_count"] == 1
    assert sparse["extra_source_attempt_identity_count"] == 1
    assert sparse["conclusions_final"] is False
    assert sparse["training_status"] == "blocked_source_identity_mismatch"
    assert report["comparison_conclusions_final"] is False
    assert report["conclusions_final"] is False

def test_report_scopes_to_requested_decision_time_labels(db_session):
    _add_hur(db_session, "TIME", output_hash="hur-time")
    _persist_complete_candidate_replay(
        db_session,
        "TIME",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash=_hur_identity_hash(db_session, "TIME"),
        decision_time_label="09:35",
    )
    _persist_complete_candidate_replay(
        db_session,
        "TIME",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash=_hur_identity_hash(db_session, "TIME"),
        decision_time_label="09:40",
    )

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
        path_mode="sparse_zero_fill",
    )

    assert report["report_decision_time_labels"] == ["09:40"]
    assert report["available_decision_time_labels"] == ["09:35", "09:40"]
    assert report["available_decision_time_labels_for_report_scope"] == ["09:40"]
    assert report["mixed_decision_times_present"] is True
    assert report["decision_time_count"] == 1
    assert report["expected_candidate_attempts"] == 1
    assert report["candidate_row_count"] == 1
    assert report["quote_replay_row_count"] == 3
    assert report["cost_replay_row_count"] == 2
    assert report["conclusions_final"] is True


def test_compare_path_modes_respects_decision_time_labels(db_session):
    _add_hur(db_session, "TIMECMP", output_hash="hur-time-cmp")
    for label in ("09:35", "09:40"):
        _persist_complete_candidate_replay(
            db_session,
            "TIMECMP",
            path_mode="strict_contiguous",
            source_hur_identity_hash=_hur_identity_hash(db_session, "TIMECMP"),
            decision_time_label=label,
        )
        _persist_complete_candidate_replay(
            db_session,
            "TIMECMP",
            path_mode="sparse_zero_fill",
            source_hur_identity_hash=_hur_identity_hash(db_session, "TIMECMP"),
            decision_time_label=label,
        )

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
        compare_path_modes=True,
    )

    assert report["report_decision_time_labels"] == ["09:40"]
    assert report["available_decision_time_labels"] == ["09:35", "09:40"]
    assert report["available_decision_time_labels_for_report_scope"] == ["09:40"]
    assert report["expected_candidate_attempts"] == 2
    assert report["candidate_counts_by_path_mode"] == {
        "sparse_zero_fill": 1,
        "strict_contiguous": 1,
    }
    assert report["quote_replay_row_count"] == 6
    assert report["cost_replay_row_count"] == 4
    assert report["comparison_conclusions_final"] is True
    assert report["conclusions_final"] is True


def test_progress_metrics_are_scoped_to_path_mode(db_session):
    _persist_complete_candidate_replay(db_session, "STRICTPROGRESS", path_mode="strict_contiguous")
    _persist_complete_candidate_replay(db_session, "SPARSEPROGRESS", path_mode="sparse_zero_fill")
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_sparse_polygon_bars()),
        alpaca_adapter=FailingAlpaca(),
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        minute_path_mode="sparse_zero_fill",
        quote_replay=False,
    )

    job._update_partial_metrics(
        counters=Counter(),
        hur_rows_loaded=1,
        event="probe",
        trading_date=DAY,
    )

    assert job.partial_metrics["minute_path_mode"] == "sparse_zero_fill"
    assert job.partial_metrics["active_candidate_row_count_for_path_mode"] == 1
    assert job.partial_metrics["quote_replay_row_count_for_path_mode"] == 3
    assert job.partial_metrics["cost_replay_row_count_for_path_mode"] == 2
    assert job.partial_metrics["schema_total_active_candidate_row_count"] == 2
    assert job.partial_metrics["schema_total_active_quote_replay_row_count"] == 6


def test_progress_metrics_are_scoped_to_decision_time_labels(db_session):
    _persist_complete_candidate_replay(
        db_session,
        "PROGRESS0935",
        path_mode="sparse_zero_fill",
        decision_time_label="09:35",
    )
    _persist_complete_candidate_replay(
        db_session,
        "PROGRESS0940",
        path_mode="sparse_zero_fill",
        decision_time_label="09:40",
    )
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_sparse_polygon_bars()),
        alpaca_adapter=FailingAlpaca(),
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        minute_path_mode="sparse_zero_fill",
        quote_replay=False,
    )

    job._update_partial_metrics(
        counters=Counter(),
        hur_rows_loaded=1,
        event="probe",
        trading_date=DAY,
    )

    assert job.partial_metrics["decision_time_count"] == 1
    assert job.partial_metrics["active_candidate_row_count_for_path_mode"] == 1
    assert job.partial_metrics["quote_replay_row_count_for_path_mode"] == 3
    assert job.partial_metrics["cost_replay_row_count_for_path_mode"] == 2
    assert job.partial_metrics["schema_total_active_candidate_row_count"] == 2
    assert job.partial_metrics["schema_total_active_quote_replay_row_count"] == 6


def test_progress_metrics_expose_count_exactness(db_session):
    _persist_complete_candidate_replay(
        db_session,
        "PROGRESSEXTRA1",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash="hur-progress-a",
    )
    _persist_complete_candidate_replay(
        db_session,
        "PROGRESSEXTRA2",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash="hur-progress-b",
    )
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_sparse_polygon_bars()),
        alpaca_adapter=FailingAlpaca(),
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        minute_path_mode="sparse_zero_fill",
        quote_replay=False,
    )

    job._update_partial_metrics(
        counters=Counter(),
        hur_rows_loaded=1,
        event="probe",
        trading_date=DAY,
    )

    assert job.partial_metrics["active_candidate_row_count_for_path_mode"] == 2
    assert job.partial_metrics["missing_source_attempt_count_for_path_mode"] == 0
    assert job.partial_metrics["extra_source_attempt_count_for_path_mode"] == 1
    assert job.partial_metrics["source_attempt_count_exact_for_path_mode"] is False
    assert job.partial_metrics["source_identity_denominator_known_for_path_mode"] is True
    assert job.partial_metrics["missing_source_attempt_identity_count_for_path_mode"] == 0
    assert job.partial_metrics["extra_source_attempt_identity_count_for_path_mode"] == 2
    assert job.partial_metrics["source_attempt_identity_exact_for_path_mode"] is False


def test_progress_metrics_expose_identity_exactness(db_session):
    _add_hur(db_session, "PROGRESSID", output_hash="hur-progress-id")
    _persist_complete_candidate_replay(
        db_session,
        "PROGRESSID",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash=_hur_identity_hash(db_session, "PROGRESSID"),
    )
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_sparse_polygon_bars()),
        alpaca_adapter=FailingAlpaca(),
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        minute_path_mode="sparse_zero_fill",
        quote_replay=False,
    )

    job._update_partial_metrics(
        counters=Counter(),
        hur_rows_loaded=1,
        event="probe",
        trading_date=DAY,
    )

    assert job.partial_metrics["source_attempt_count_exact_for_path_mode"] is True
    assert job.partial_metrics["source_identity_denominator_known_for_path_mode"] is True
    assert job.partial_metrics["source_identity_denominator_error_for_path_mode"] is None
    assert job.partial_metrics["missing_source_attempt_identity_count_for_path_mode"] == 0
    assert job.partial_metrics["extra_source_attempt_identity_count_for_path_mode"] == 0
    assert job.partial_metrics["source_attempt_identity_exact_for_path_mode"] is True


def test_update_partial_metrics_closes_owned_read_transaction(db_session):
    _add_hur(db_session, "TXNPROGRESS", output_hash="hur-txn-progress")
    _persist_complete_candidate_replay(
        db_session,
        "TXNPROGRESS",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash=_hur_identity_hash(db_session, "TXNPROGRESS"),
    )
    db_session.commit()
    assert db_session.in_transaction() is False
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_sparse_polygon_bars()),
        alpaca_adapter=FailingAlpaca(),
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        minute_path_mode="sparse_zero_fill",
        quote_replay=False,
    )

    job._update_partial_metrics(
        counters=Counter(),
        hur_rows_loaded=1,
        event="date_finish",
        trading_date=DAY,
    )

    assert job.partial_metrics["source_attempt_identity_exact_for_path_mode"] is True
    assert db_session.in_transaction() is False


def test_i12_pit_run_rolls_back_hur_load_exception(db_session, monkeypatch):
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_polygon_bars()),
        alpaca_adapter=None,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        minute_path_mode="strict_contiguous",
        quote_replay=False,
    )

    def fail_load_hur_rows(trading_date):
        assert trading_date == DAY
        db_session.execute(text("SELECT 1"))
        assert db_session.in_transaction() is True
        raise RuntimeError("hur read failed")

    monkeypatch.setattr(job, "_load_hur_rows", fail_load_hur_rows)

    with pytest.raises(RuntimeError, match="hur read failed"):
        job.run(JobContext(
            job_id="job-hur-fail",
            job_run_id="run-hur-fail",
            started_at=datetime.now(timezone.utc),
        ))

    assert db_session.in_transaction() is False


def test_update_partial_metrics_error_rolls_back_and_writes_minimal_artifact(
    db_session,
    monkeypatch,
    tmp_path,
):
    _add_hur(db_session, "ERRPROGRESS", output_hash="hur-err-progress")
    _persist_complete_candidate_replay(
        db_session,
        "ERRPROGRESS",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash=_hur_identity_hash(db_session, "ERRPROGRESS"),
    )
    db_session.commit()
    progress_path = tmp_path / "progress_error.json"
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_sparse_polygon_bars()),
        alpaca_adapter=FailingAlpaca(),
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        minute_path_mode="sparse_zero_fill",
        quote_replay=False,
        progress_artifact=progress_path,
    )

    def fail_identity(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("progress identity failed")

    monkeypatch.setattr(i12_pit_rebuild, "_source_attempt_identity_audit", fail_identity)

    job._update_partial_metrics(
        counters=Counter({"candidate_passed": 1}),
        hur_rows_loaded=1,
        event="date_finish",
        trading_date=DAY,
        job_run_id="job-progress-error",
    )

    artifact = json.loads(progress_path.read_text())
    assert artifact["event"] == "progress_error"
    assert artifact["job_run_id"] == "job-progress-error"
    assert artifact["last_trading_date"] == DAY.isoformat()
    assert artifact["progress_error"]["error_type"] == "RuntimeError"
    assert "progress identity failed" in artifact["progress_error"]["message"]
    assert db_session.in_transaction() is False


def _progress_only_job(db_session, progress_path):
    return I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp([]),
        polygon_adapter=FakePolygon([]),
        alpaca_adapter=None,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        minute_path_mode="strict_contiguous",
        quote_replay=False,
        progress_artifact=progress_path,
    )


def test_progress_write_is_valid_json_and_creates_parent_directories(db_session, tmp_path):
    progress_path = tmp_path / "nested" / "progress" / "i12_progress.json"
    job = _progress_only_job(db_session, progress_path)

    job._progress("date_start", {"ticker": "ATOM", "sequence": 1})

    artifact = json.loads(progress_path.read_text())
    assert artifact["event"] == "date_start"
    assert artifact["ticker"] == "ATOM"
    assert artifact["sequence"] == 1
    assert artifact["wall_clock_utc"]


def test_progress_write_repeated_updates_leave_last_complete_record(
    db_session,
    tmp_path,
):
    progress_path = tmp_path / "i12_progress.json"
    job = _progress_only_job(db_session, progress_path)

    job._progress("ticker_progress", {"sequence": 1})
    job._progress("ticker_progress", {"sequence": 2})
    job._progress("date_finish", {"sequence": 3})

    artifact = json.loads(progress_path.read_text())
    assert artifact["event"] == "date_finish"
    assert artifact["sequence"] == 3


def test_progress_write_replace_failure_preserves_previous_complete_artifact(
    db_session,
    monkeypatch,
    tmp_path,
):
    progress_path = tmp_path / "i12_progress.json"
    job = _progress_only_job(db_session, progress_path)
    job._progress("date_start", {"sequence": 1})

    def fail_replace(src, dst):
        del src, dst
        raise OSError("replace failed")

    monkeypatch.setattr(i12_pit_rebuild.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        job._progress("ticker_progress", {"sequence": 2})

    artifact = json.loads(progress_path.read_text())
    assert artifact["event"] == "date_start"
    assert artifact["sequence"] == 1
    assert not list(progress_path.parent.glob(f".{progress_path.name}.*.tmp"))


def test_no_progress_timeout_payload_uses_last_progress_event(db_session):
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp([]),
        polygon_adapter=FakePolygon([]),
        alpaca_adapter=None,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        minute_path_mode="strict_contiguous",
        quote_replay=False,
        max_no_progress_seconds=30,
    )

    job._progress(
        "provider_fetch_start",
        {"job_run_id": "job-no-progress", "ticker": "WEDGE"},
    )
    with job._progress_state_lock:
        job._last_progress_monotonic -= 31

    payload = job._no_progress_timeout_payload("job-no-progress")

    assert payload is not None
    assert payload["job_run_id"] == "job-no-progress"
    assert payload["max_no_progress_seconds"] == 30
    assert payload["no_progress_age_seconds"] >= 30
    assert payload["last_progress_event"] == "provider_fetch_start"
    assert payload["last_progress_payload"]["ticker"] == "WEDGE"
    assert payload["fetch_watchdog"]["circuit_open"] is False


def test_i12_pit_run_emits_date_and_ticker_heartbeats(db_session):
    _add_hur(db_session, "HEART1", output_hash="hur-heart-1")
    _add_hur(db_session, "HEART2", output_hash="hur-heart-2")
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_polygon_bars()),
        alpaca_adapter=None,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        minute_path_mode="strict_contiguous",
        quote_replay=False,
        progress_interval_tickers=1,
    )
    events = []

    def capture_progress(event, payload):
        events.append((event, dict(payload)))

    job._progress = capture_progress

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    event_names = [event for event, _payload in events]
    assert event_names[0] == "start"
    assert "date_start" in event_names
    assert event_names.count("ticker_progress") == 2
    assert "date_finish" in event_names
    assert event_names[-1] == "finish"
    date_start = next(payload for event, payload in events if event == "date_start")
    assert date_start["hur_rows_for_date"] == 2
    assert date_start["hur_rows_processed_for_date"] == 0
    ticker_payloads = [payload for event, payload in events if event == "ticker_progress"]
    assert [payload["hur_rows_processed_for_date"] for payload in ticker_payloads] == [1, 2]
    assert ticker_payloads[-1]["cumulative_hur_rows_loaded"] == 2


def test_i12_pit_daily_fetch_errors_emit_ticker_progress(db_session):
    _add_hur(db_session, "DAILYERR1", output_hash="hur-daily-err-1")
    _add_hur(db_session, "DAILYERR2", output_hash="hur-daily-err-2")
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmpByTicker({}, error_tickers={"DAILYERR1", "DAILYERR2"}),
        polygon_adapter=FakePolygon(_polygon_bars()),
        alpaca_adapter=None,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        minute_path_mode="strict_contiguous",
        quote_replay=False,
        progress_interval_tickers=1,
    )
    events = []

    def capture_progress(event, payload):
        events.append((event, dict(payload)))

    job._progress = capture_progress

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    ticker_payloads = [payload for event, payload in events if event == "ticker_progress"]
    assert [payload["current_ticker"] for payload in ticker_payloads] == [
        "DAILYERR1",
        "DAILYERR2",
    ]
    assert [payload["hur_rows_processed_for_date"] for payload in ticker_payloads] == [1, 2]
    assert all(payload["hur_rows_for_date"] == 2 for payload in ticker_payloads)
    assert all(payload["job_run_id"] for payload in ticker_payloads)
    assert ticker_payloads[-1]["counters"]["coverage_daily_fetch_error"] == 2
    assert ticker_payloads[-1]["counters"]["candidate_failed"] == 2


def test_i12_pit_minute_fetch_errors_emit_ticker_progress(db_session):
    _add_hur(db_session, "MINERR1", output_hash="hur-minute-err-1")
    _add_hur(db_session, "MINERR2", output_hash="hur-minute-err-2")
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygonByTicker({}, error_tickers={"MINERR1", "MINERR2"}),
        alpaca_adapter=None,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        minute_path_mode="strict_contiguous",
        quote_replay=False,
        progress_interval_tickers=1,
    )
    events = []

    def capture_progress(event, payload):
        events.append((event, dict(payload)))

    job._progress = capture_progress

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    ticker_payloads = [payload for event, payload in events if event == "ticker_progress"]
    assert [payload["current_ticker"] for payload in ticker_payloads] == [
        "MINERR1",
        "MINERR2",
    ]
    assert [payload["hur_rows_processed_for_date"] for payload in ticker_payloads] == [1, 2]
    assert all(payload["hur_rows_for_date"] == 2 for payload in ticker_payloads)
    assert all(payload["job_run_id"] for payload in ticker_payloads)
    assert ticker_payloads[-1]["counters"]["coverage_minute_fetch_error"] == 2
    assert ticker_payloads[-1]["counters"]["candidate_failed"] == 2


def test_progress_identity_scope_uses_processed_dates_not_full_window(db_session):
    next_day = next_us_equity_session(DAY + timedelta(days=1))
    _add_hur(db_session, "PROGRESSWINDOW", day=DAY, output_hash="hur-progress-window-1")
    _add_hur(
        db_session,
        "PROGRESSWINDOW",
        day=next_day,
        output_hash="hur-progress-window-2",
    )
    _persist_complete_candidate_replay(
        db_session,
        "PROGRESSWINDOW",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash=_hur_identity_hash(
            db_session,
            "PROGRESSWINDOW",
            day=DAY,
        ),
    )
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_sparse_polygon_bars()),
        alpaca_adapter=FailingAlpaca(),
        start_date=DAY,
        end_date=next_day,
        decision_times=["09:40"],
        minute_path_mode="sparse_zero_fill",
        quote_replay=False,
    )

    job._update_partial_metrics(
        counters=Counter(),
        hur_rows_loaded=1,
        event="date_finish",
        trading_date=DAY,
    )

    assert job.partial_metrics["progress_source_scope"] == "processed_through_trading_date"
    assert job.partial_metrics["progress_source_end_date"] == DAY.isoformat()
    assert job.partial_metrics["source_attempt_count_exact_for_path_mode"] is True
    assert job.partial_metrics["source_identity_denominator_known_for_path_mode"] is True
    assert job.partial_metrics["missing_source_attempt_identity_count_for_path_mode"] == 0
    assert job.partial_metrics["extra_source_attempt_identity_count_for_path_mode"] == 0
    assert job.partial_metrics["source_attempt_identity_exact_for_path_mode"] is True

    final_report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=next_day,
        decision_time_labels=["09:40"],
        path_mode="sparse_zero_fill",
    )

    assert final_report["expected_candidate_attempts"] == 2
    assert final_report["actual_candidate_row_count"] == 1
    assert final_report["missing_source_attempt_count"] == 1
    assert final_report["source_identity_denominator_known"] is True
    assert final_report["missing_source_attempt_identity_count"] == 1
    assert final_report["source_replay_complete"] is False
    assert final_report["conclusions_final"] is False


def test_i12_pit_report_chunks_child_candidate_id_queries(db_session, monkeypatch):
    tickers = ["CHUNK1", "CHUNK2", "CHUNK3"]
    for ticker in tickers:
        _add_hur(db_session, ticker, output_hash=f"hur-{ticker}")
        _persist_complete_candidate_replay(
            db_session,
            ticker,
            source_hur_identity_hash=_hur_identity_hash(db_session, ticker),
        )
    db_session.commit()
    monkeypatch.setattr(i12_pit_rebuild, "CHILD_CANDIDATE_ID_QUERY_CHUNK_SIZE", 2)

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
        path_mode="strict_contiguous",
    )

    assert report["expected_candidate_attempts"] == 3
    assert report["pit_candidate_count"] == 3
    assert report["quote_replay_row_count"] == 9
    assert report["cost_replay_row_count"] == 6
    assert report["quote_replay_complete"] is True
    assert report["cost_replay_complete"] is True
    assert report["conclusions_final"] is True


def test_candidate_attempt_hash_includes_decision_time_label():
    first = _candidate_row(
        "SAMEHASH",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash="hur-same",
        decision_time_label="09:35",
    )
    second = _candidate_row(
        "SAMEHASH",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash="hur-same",
        decision_time_label="09:40",
    )

    assert first.candidate_attempt_hash != second.candidate_attempt_hash


def test_sparse_would_pass_diagnostic_uses_source_hur_identity(db_session):
    _persist_candidate(
        db_session,
        "MATCH",
        path_mode="strict_contiguous",
        source_hur_identity_hash="hur-a",
        candidate_status="failed",
        coverage_status="partial_minute_path",
    )
    _persist_complete_candidate_replay(
        db_session,
        "MATCH",
        path_mode="sparse_zero_fill",
        source_hur_identity_hash="hur-b",
    )

    report = i12_pit_rebuild_report(
        db_session,
        hur_rows_loaded=2,
        decision_time_count=1,
        compare_path_modes=True,
    )

    assert report["strict_partial_rows_that_would_pass_sparse_zero_fill"] == 0
    assert report["training_status"] == "blocked_source_identity_denominator_unknown"


def test_duplicate_active_quote_and_cost_rows_fail_closed(db_session):
    candidate = _persist_candidate(db_session, "DUPACTIVE")
    entry = _persist_quote(db_session, candidate, role="entry", status="ok")
    _persist_quote(db_session, candidate, role="entry", status="ok", bid=10.01, ask=10.06)
    same_day = _persist_quote(db_session, candidate, role="same_day_exit", status="ok")
    next_open = _persist_quote(db_session, candidate, role="next_open_exit", status="ok")
    same_day_result = evaluate_quote_cost_replay(
        entry_quote=entry,
        exit_quote=same_day,
        exit_role="same_day_exit",
        intended_order_usd=50,
        max_spread_bps=200,
        slippage_bps=0,
    )
    _persist_cost(db_session, candidate, "same_day_exit", same_day_result)
    _persist_cost(
        db_session,
        candidate,
        "same_day_exit",
        evaluate_quote_cost_replay(
            entry_quote=entry,
            exit_quote=same_day,
            exit_role="same_day_exit",
            intended_order_usd=50,
            max_spread_bps=200,
            slippage_bps=1,
        ),
    )
    _persist_cost(
        db_session,
        candidate,
        "next_open_exit",
        evaluate_quote_cost_replay(
            entry_quote=entry,
            exit_quote=next_open,
            exit_role="next_open_exit",
            intended_order_usd=50,
            max_spread_bps=200,
            slippage_bps=0,
        ),
    )

    report = i12_pit_rebuild_report(
        db_session,
        hur_rows_loaded=1,
        decision_time_count=1,
    )

    assert report["duplicate_quote_role_count"] == 1
    assert report["duplicate_cost_role_count"] == 1
    assert report["quote_replay_complete"] is False
    assert report["cost_replay_complete"] is False
    assert report["training_status"] == "blocked_source_identity_denominator_unknown"
    assert report["conclusions_final"] is False


def test_active_candidate_attempt_uniqueness_is_enforced(db_session):
    first = _candidate_row("DUPKEY")
    second = _candidate_row("DUPKEY2")
    second.i12_pit_candidate_id = "cand-DUPKEY2-alt"
    second.candidate_attempt_hash = first.candidate_attempt_hash
    second.content_hash = "content-DUPKEY2-alt"
    second.input_hash = "input-DUPKEY2-alt"
    second.candidate_identity_hash = "identity-DUPKEY2-alt"
    second.label_hash = "label-DUPKEY2-alt"
    db_session.add(first)
    db_session.flush()
    db_session.add(second)

    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_interrupted_rerun_preserves_committed_dates_without_duplicates(db_session):
    next_day = next_us_equity_session(DAY + timedelta(days=1))
    _add_hur(db_session, "RESUME1", day=DAY, output_hash="hur-resume-1")
    _add_hur(db_session, "RESUME2", day=next_day, output_hash="hur-resume-2")

    class FailsOnSecondFetch(FakeFmp):
        def __init__(self):
            super().__init__(_fmp_bars())
            self.call_count = 0

        def get_historical_price(self, ticker, **kwargs):
            self.call_count += 1
            if self.call_count > 1:
                raise RuntimeError("interrupted")
            return super().get_historical_price(ticker, **kwargs)

    failing_job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FailsOnSecondFetch(),
        polygon_adapter=FakePolygon(_polygon_bars()),
        alpaca_adapter=None,
        start_date=DAY,
        end_date=next_day,
        decision_times=["09:40"],
        quote_replay=False,
    )

    failed = run_job(db_session, failing_job, params={"test": True})

    assert not failed.ok
    assert db_session.query(I12PitCandidate).count() == 1
    interrupted_report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=next_day,
        decision_time_labels=["09:40"],
    )
    assert interrupted_report["expected_candidate_attempts"] == 2
    assert interrupted_report["actual_candidate_row_count"] == 1
    assert interrupted_report["missing_source_attempt_count"] == 1
    assert interrupted_report["training_status"] == "blocked_source_replay_incomplete"
    assert interrupted_report["conclusions_final"] is False

    rerun_job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_polygon_bars()),
        alpaca_adapter=None,
        start_date=DAY,
        end_date=next_day,
        decision_times=["09:40"],
        quote_replay=False,
    )
    rerun = run_job(db_session, rerun_job, params={"test": True})

    assert rerun.ok
    assert db_session.query(I12PitCandidate).count() == 2
    assert rerun.metrics["expected_candidate_attempts"] == 2
    assert rerun.metrics["actual_candidate_row_count"] == 2


def test_report_marks_incomplete_quote_replay_non_final(db_session):
    _add_hur(db_session, "INCOMP", output_hash="hur-incomp")
    candidate = _persist_candidate(
        db_session,
        "INCOMP",
        source_hur_identity_hash=_hur_identity_hash(db_session, "INCOMP"),
    )
    _persist_quote(db_session, candidate, role="entry", status="ok")
    _persist_quote(db_session, candidate, role="same_day_exit", status="missing")

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
    )

    assert report["quote_replay_complete"] is False
    assert report["conclusions_final"] is False
    assert report["training_status"] == "blocked_quote_replay_incomplete"
    assert report["ml_ranking_status"] == "blocked_until_quote_replay_complete"


def test_report_treats_non_ok_quote_rows_as_complete_durable_evidence(db_session):
    _add_hur(db_session, "MISSQ", output_hash="hur-missing-quote-final")
    candidate = _persist_candidate(
        db_session,
        "MISSQ",
        source_hur_identity_hash=_hur_identity_hash(db_session, "MISSQ"),
    )
    entry = _persist_quote(db_session, candidate, role="entry", status="ok")
    same_day = _persist_quote(
        db_session,
        candidate,
        role="same_day_exit",
        status="missing",
    )
    next_open = _persist_quote(db_session, candidate, role="next_open_exit", status="ok")
    for exit_role, exit_quote in {
        "same_day_exit": same_day,
        "next_open_exit": next_open,
    }.items():
        _persist_cost(
            db_session,
            candidate,
            exit_role,
            evaluate_quote_cost_replay(
                entry_quote=entry,
                exit_quote=exit_quote,
                exit_role=exit_role,
                intended_order_usd=50,
                max_spread_bps=200,
                slippage_bps=0,
            ),
        )

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
    )

    assert report["quote_replay_complete"] is True
    assert report["cost_replay_complete"] is True
    assert report["data_integrity_passed"] is True
    assert report["conclusions_final"] is True
    assert report["training_status"] == "eligible_for_retrain_evaluation"
    assert report["ml_ranking_status"] == "not_run_quote_layer_ready"
    assert report["quote_ok_count"] == 2
    assert report["quote_non_ok_count"] == 1
    assert report["quote_ok_rate"] == pytest.approx(2 / 3)
    assert report["quote_coverage_rate"] == 1.0
    assert report["quote_coverage_by_role"]["same_day_exit"][
        "coverage_status_counts"
    ] == {"missing": 1}
    assert report["exit_metrics"]["same_day_exit"]["skipped_cash_by_reason"] == {
        "exit_quote_missing": 1,
    }


def test_report_blocks_unknown_quote_coverage_status_as_integrity_error(db_session):
    _add_hur(db_session, "WEIRDQ", output_hash="hur-weird-quote")
    candidate = _persist_candidate(
        db_session,
        "WEIRDQ",
        source_hur_identity_hash=_hur_identity_hash(db_session, "WEIRDQ"),
    )
    entry = _persist_quote(db_session, candidate, role="entry", status="ok")
    same_day = _persist_quote(db_session, candidate, role="same_day_exit", status="ok")
    next_open = _persist_quote(
        db_session,
        candidate,
        role="next_open_exit",
        status="weird",
    )
    for exit_role, exit_quote in {
        "same_day_exit": same_day,
        "next_open_exit": next_open,
    }.items():
        _persist_cost(
            db_session,
            candidate,
            exit_role,
            evaluate_quote_cost_replay(
                entry_quote=entry,
                exit_quote=exit_quote,
                exit_role=exit_role,
                intended_order_usd=50,
                max_spread_bps=200,
                slippage_bps=0,
            ),
        )

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
    )

    assert report["quote_replay_complete"] is False
    assert report["unknown_quote_coverage_status_count"] == 1
    assert report["quote_coverage_by_role"]["next_open_exit"]["unknown_status"] == 1
    assert report["cost_replay_complete"] is True
    assert report["data_integrity_passed"] is False
    assert report["conclusions_final"] is False
    assert report["training_status"] == "blocked_quote_replay_integrity"
    assert report["ml_ranking_status"] == "blocked_quote_replay_integrity"


def test_report_blocks_when_quotes_complete_but_cost_rows_missing(db_session):
    _add_hur(db_session, "NOCOST", output_hash="hur-nocost")
    candidate = _persist_candidate(
        db_session,
        "NOCOST",
        source_hur_identity_hash=_hur_identity_hash(db_session, "NOCOST"),
    )
    _persist_quote(db_session, candidate, role="entry", status="ok")
    _persist_quote(db_session, candidate, role="same_day_exit", status="ok")
    _persist_quote(db_session, candidate, role="next_open_exit", status="ok")

    report = i12_pit_rebuild_report(
        db_session,
        source_hur_schema="public",
        start_date=DAY,
        end_date=DAY,
        decision_time_labels=["09:40"],
    )

    assert report["quote_replay_complete"] is True
    assert report["cost_replay_complete"] is False
    assert report["missing_cost_role_count"] == 2
    assert report["candidate_complete_cost_count"] == 0
    assert report["candidate_incomplete_cost_count"] == 1
    assert report["data_integrity_passed"] is False
    assert report["conclusions_final"] is False
    assert report["training_status"] == "blocked_cost_replay_incomplete"


def test_date_free_report_with_complete_rows_is_non_final_without_denominator(db_session):
    candidate = _persist_candidate(db_session, "NODENOM")
    entry = _persist_quote(db_session, candidate, role="entry", status="ok")
    same_day = _persist_quote(db_session, candidate, role="same_day_exit", status="ok")
    next_open = _persist_quote(db_session, candidate, role="next_open_exit", status="ok")
    _persist_cost(
        db_session,
        candidate,
        "same_day_exit",
        evaluate_quote_cost_replay(
            entry_quote=entry,
            exit_quote=same_day,
            exit_role="same_day_exit",
            intended_order_usd=50,
            max_spread_bps=200,
        ),
    )
    _persist_cost(
        db_session,
        candidate,
        "next_open_exit",
        evaluate_quote_cost_replay(
            entry_quote=entry,
            exit_quote=next_open,
            exit_role="next_open_exit",
            intended_order_usd=50,
            max_spread_bps=200,
        ),
    )

    report = i12_pit_rebuild_report(db_session)

    assert report["quote_replay_complete"] is True
    assert report["cost_replay_complete"] is True
    assert report["source_denominator_known"] is False
    assert report["data_integrity_passed"] is False
    assert report["conclusions_final"] is False
    assert report["training_status"] == "blocked_source_denominator_unknown"


def test_current_deferred_pit_model_remains_non_promotable(db_session):
    model = MLModelRegistry(
        model_id="stage1_i12_403a5ae359cd_accecdda",
        pattern_id="I12",
        model_family="hist_gradient_boosting_regressor",
        training_window_start=DAY,
        training_window_end=DAY,
        manifest_version=FROZEN_I12_STAGE0_MANIFEST_VERSION,
        manifest_sha256=FROZEN_I12_STAGE0_MANIFEST_SHA256,
        feature_schema_hash=FROZEN_I12_STAGE0_FEATURE_SCHEMA_HASH,
        feature_code_git_sha="test",
        status="shadow",
        training_params_json=json.dumps({
            "horizon_sessions": 1,
            "signal_horizon": "1d",
        }),
        cv_metrics_json=json.dumps({
            "training_selection": {
                "pit_deferred": True,
                "pit_failed_row_count": 10012,
            }
        }),
        feature_schema_json=json.dumps(_i12_feature_schema(), sort_keys=True),
        artifact_uri="/tmp/not-loaded-here.pkl",
    )
    db_session.add(model)
    db_session.flush()

    contract = select_i12_model(
        db_session,
        model_id=model.model_id,
        allow_latest_model=False,
        feed="sip",
    )

    assert contract.promotable_run is False
    assert "deferred_pit_model" in contract.non_promotable_reasons


def test_report_only_does_not_require_date_range(monkeypatch, capsys):
    class FakeSession:
        def close(self):
            pass

    session = FakeSession()
    captured = {}
    monkeypatch.setattr(run_i12_pit_rebuild, "load_runtime_env", lambda: None)
    monkeypatch.setattr(run_i12_pit_rebuild, "prepare_writable_schema_target", lambda **kwargs: None)
    monkeypatch.setattr(run_i12_pit_rebuild, "open_writable_session", lambda *, schema: session)
    monkeypatch.setattr(run_i12_pit_rebuild, "_assert_required_pit_columns", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_i12_pit_rebuild,
        "i12_pit_rebuild_report",
        lambda *args, **kwargs: captured.update(kwargs) or {
            "report": "ok",
            "start_date": kwargs.get("start_date"),
        },
    )

    code = run_i12_pit_rebuild.main([
        "--schema",
        "scratch_report",
        "--report-only",
    ])

    assert code == 0
    assert captured["path_mode"] == "strict_contiguous"
    assert captured["compare_path_modes"] is False
    assert captured["decision_time_labels"] == list(run_i12_pit_rebuild.DEFAULT_DECISION_TIMES)
    assert '"report": "ok"' in capsys.readouterr().out


def test_runner_accepts_optional_no_progress_timeout():
    args = run_i12_pit_rebuild._parse_args([
        "--schema",
        "scratch_i12",
        "--start-date",
        DAY.isoformat(),
        "--end-date",
        DAY.isoformat(),
        "--max-no-progress-minutes",
        "12.5",
    ])

    assert args.max_no_progress_minutes == 12.5


def test_runner_defaults_no_progress_timeout_for_rebuilds():
    args = run_i12_pit_rebuild._parse_args([
        "--schema",
        "scratch_i12",
        "--start-date",
        DAY.isoformat(),
        "--end-date",
        DAY.isoformat(),
    ])

    assert args.max_no_progress_minutes == 20.0


def test_runner_rejects_negative_no_progress_timeout():
    with pytest.raises(SystemExit):
        run_i12_pit_rebuild._parse_args([
            "--schema",
            "scratch_i12",
            "--start-date",
            DAY.isoformat(),
            "--end-date",
            DAY.isoformat(),
            "--max-no-progress-minutes",
            "-1",
        ])


def test_runner_no_progress_exit_callback_exits_70(monkeypatch, capsys):
    def fake_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(run_i12_pit_rebuild.os, "_exit", fake_exit)

    with pytest.raises(SystemExit) as exc:
        run_i12_pit_rebuild._exit_on_no_progress({
            "job_run_id": "job-wedged",
            "last_progress_event": "provider_fetch_start",
        })

    assert exc.value.code == 70
    err = capsys.readouterr().err
    assert "no-progress watchdog fired" in err
    assert "job-wedged" in err


def test_runner_no_progress_exit_subprocess_writes_artifact_and_exits_70(tmp_path):
    progress_path = tmp_path / "no_progress.json"
    code = f"""
from alpha.jobs.i12_pit_rebuild import I12PitRebuildJob
from alpha.jobs.run_i12_pit_rebuild import _exit_on_no_progress

class FakeSession:
    pass

job = I12PitRebuildJob(
    session=FakeSession(),
    fmp_adapter=object(),
    polygon_adapter=object(),
    alpaca_adapter=None,
    start_date=None,
    end_date=None,
    decision_times=["09:40"],
    minute_path_mode="strict_contiguous",
    quote_replay=False,
    progress_artifact={str(progress_path)!r},
    max_no_progress_seconds=0.1,
    no_progress_exit_callback=_exit_on_no_progress,
)
job._progress("provider_fetch_start", {{"job_run_id": "subprocess-no-progress"}})
with job._progress_state_lock:
    job._last_progress_monotonic -= 1
payload = job._no_progress_timeout_payload("subprocess-no-progress")
job._progress("no_progress_timeout", payload)
_exit_on_no_progress(payload)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 70
    assert "no-progress watchdog fired" in result.stderr
    artifact = json.loads(progress_path.read_text())
    assert artifact["event"] == "no_progress_timeout"
    assert artifact["job_run_id"] == "subprocess-no-progress"
    assert artifact["last_progress_event"] == "provider_fetch_start"


def test_preflight_only_does_not_require_date_range_or_providers(monkeypatch, capsys):
    class FakeSession:
        def close(self):
            pass

    session = FakeSession()
    monkeypatch.setattr(run_i12_pit_rebuild, "load_runtime_env", lambda: None)
    monkeypatch.setattr(run_i12_pit_rebuild, "prepare_writable_schema_target", lambda **kwargs: None)
    monkeypatch.setattr(run_i12_pit_rebuild, "open_writable_session", lambda *, schema: session)
    monkeypatch.setattr(run_i12_pit_rebuild, "_assert_required_pit_columns", lambda *args, **kwargs: None)

    class ExplodingFmpAdapter:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("provider constructed")

    monkeypatch.setattr(run_i12_pit_rebuild, "FmpAdapter", ExplodingFmpAdapter)

    code = run_i12_pit_rebuild.main([
        "--schema",
        "scratch_preflight",
        "--create-tables",
        "--preflight-only",
        "--minute-path-mode",
        "sparse_zero_fill",
    ])

    assert code == 0
    out = capsys.readouterr().out
    assert '"preflight": "ok"' in out
    assert '"schema": "scratch_preflight"' in out
    assert '"minute_path_mode": "sparse_zero_fill"' in out


def test_report_only_passes_requested_decision_time_labels(monkeypatch):
    class FakeSession:
        def close(self):
            pass

    captured = {}
    monkeypatch.setattr(run_i12_pit_rebuild, "load_runtime_env", lambda: None)
    monkeypatch.setattr(run_i12_pit_rebuild, "prepare_writable_schema_target", lambda **kwargs: None)
    monkeypatch.setattr(run_i12_pit_rebuild, "open_writable_session", lambda *, schema: FakeSession())
    monkeypatch.setattr(run_i12_pit_rebuild, "_assert_required_pit_columns", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_i12_pit_rebuild,
        "i12_pit_rebuild_report",
        lambda *args, **kwargs: captured.update(kwargs) or {"report": "ok"},
    )

    code = run_i12_pit_rebuild.main([
        "--schema",
        "scratch_report",
        "--report-only",
        "--decision-time",
        "09:40",
    ])

    assert code == 0
    assert captured["decision_time_labels"] == ["09:40"]


def test_report_only_compare_path_modes_passes_compare_flag(monkeypatch):
    class FakeSession:
        def close(self):
            pass

    captured = {}
    monkeypatch.setattr(run_i12_pit_rebuild, "load_runtime_env", lambda: None)
    monkeypatch.setattr(run_i12_pit_rebuild, "prepare_writable_schema_target", lambda **kwargs: None)
    monkeypatch.setattr(run_i12_pit_rebuild, "open_writable_session", lambda *, schema: FakeSession())
    monkeypatch.setattr(run_i12_pit_rebuild, "_assert_required_pit_columns", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_i12_pit_rebuild,
        "i12_pit_rebuild_report",
        lambda *args, **kwargs: captured.update(kwargs) or {"report": "ok"},
    )

    code = run_i12_pit_rebuild.main([
        "--schema",
        "scratch_report",
        "--report-only",
        "--compare-path-modes",
    ])

    assert code == 0
    assert captured["path_mode"] is None
    assert captured["compare_path_modes"] is True
    assert captured["decision_time_labels"] == list(run_i12_pit_rebuild.DEFAULT_DECISION_TIMES)


def test_runner_fails_old_schema_missing_path_mode(monkeypatch, capsys):
    class FakeSession:
        def close(self):
            pass

    monkeypatch.setattr(run_i12_pit_rebuild, "load_runtime_env", lambda: None)
    monkeypatch.setattr(run_i12_pit_rebuild, "prepare_writable_schema_target", lambda **kwargs: None)
    monkeypatch.setattr(run_i12_pit_rebuild, "open_writable_session", lambda *, schema: FakeSession())

    def fail_columns(*args, **kwargs):
        raise ValueError(
            "schema scratch_old has old i12_pit_candidates table without path_mode; "
            "create a fresh scratch schema or migrate it"
        )

    monkeypatch.setattr(run_i12_pit_rebuild, "_assert_required_pit_columns", fail_columns)

    code = run_i12_pit_rebuild.main([
        "--schema",
        "scratch_old",
        "--report-only",
    ])

    assert code == 1
    assert "without path_mode" in capsys.readouterr().out


def test_runner_refuses_no_schema_writes(monkeypatch, capsys):
    monkeypatch.setenv("ALPHA_DB_SCHEMA", "")

    code = run_i12_pit_rebuild_main([
        "--start-date",
        DAY.isoformat(),
        "--end-date",
        DAY.isoformat(),
    ])

    assert code == 1
    assert "requires --schema" in capsys.readouterr().out


def test_job_reads_hur_from_source_schema_and_writes_only_pit_tables(db_session):
    db_session.execute(text("ATTACH DATABASE ':memory:' AS source_hur"))
    db_session.execute(text(
        "CREATE TABLE source_hur.historical_universe_reconstructions ("
        "normalized_symbol TEXT NOT NULL, "
        "replay_date DATE NOT NULL, "
        "inclusion_status TEXT NOT NULL)"
    ))
    db_session.execute(
        text(
            "INSERT INTO source_hur.historical_universe_reconstructions "
            "(normalized_symbol, replay_date, inclusion_status) "
            "VALUES (:ticker, :day, 'included')"
        ),
        {"ticker": "PIT", "day": DAY.isoformat()},
    )
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_polygon_bars()),
        alpaca_adapter=None,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        source_hur_schema="source_hur",
        quote_replay=False,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    assert db_session.query(HistoricalUniverseReconstruction).count() == 0
    assert db_session.query(I12PitCandidate).count() == 1
    assert result.metrics["source_hur_schema"] == "source_hur"
    assert result.metrics["hur_rows_loaded"] == 1
    assert result.metrics["zero_hur_source_blocked"] is False


def test_candidate_identity_ignores_changed_future_label(db_session):
    job = I12PitRebuildJob(
        session=db_session,
        fmp_adapter=FakeFmp(_fmp_bars()),
        polygon_adapter=FakePolygon(_polygon_bars()),
        alpaca_adapter=None,
        start_date=DAY,
        end_date=DAY,
        decision_times=["09:40"],
        quote_replay=False,
    )
    first = build_i12_pit_candidate(
        ticker="LABEL",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(),
        minute_bars=_minute_bars(),
        daily_source_hash="same-source",
        minute_source_hash="same-minute",
    )
    changed_label = build_i12_pit_candidate(
        ticker="LABEL",
        trading_date=DAY,
        decision_ts=DECISION_TS,
        decision_time_label="09:40",
        daily_bars=_daily_bars(next_open=8.8),
        minute_bars=_minute_bars(),
        daily_source_hash="same-source",
        minute_source_hash="same-minute",
    )

    assert changed_label.content_hash == first.content_hash
    assert changed_label.label_hash != first.label_hash
    job._persist_candidate(first, job_run_id=None)
    job._persist_candidate(changed_label, job_run_id=None)

    assert db_session.query(I12PitCandidate).count() == 1
    stored = db_session.query(I12PitCandidate).one()
    assert stored.content_hash == first.content_hash
    assert stored.candidate_identity_hash == first.candidate_identity_hash
    assert stored.label_hash == changed_label.label_hash


class _NoopCounters:
    def record_non_session(self, ticker, parsed_date):
        del ticker, parsed_date


def _daily_bars(day_volume=100_000, next_open=4.4):
    return _clean_daily_bars(
        "PIT",
        _fmp_bars(day_volume=day_volume, next_open=next_open),
        _NoopCounters(),
    )


def _fmp_bars(day_volume=100_000, next_open=4.4):
    sessions = []
    cursor = DAY
    for _ in range(26):
        cursor = previous_us_equity_session(cursor)
        sessions.append(cursor)
    sessions = sorted(set(sessions))
    bars = []
    for idx, session_date in enumerate(sessions):
        close = 10.0 if idx == 0 else 4.0
        bars.append(
            FmpBar(
                date=session_date.isoformat(),
                open=close * 0.99,
                high=close * 1.01,
                low=close * 0.98,
                close=close,
                volume=1000,
                split_adjusted_close=close,
            )
        )
    bars.append(
        FmpBar(
            date=DAY.isoformat(),
            open=4.02,
            high=99.0,
            low=0.01,
            close=88.0,
            volume=day_volume,
            split_adjusted_close=88.0,
        )
    )
    bars.append(
        FmpBar(
            date=next_us_equity_session(DAY + timedelta(days=1)).isoformat(),
            open=next_open,
            high=4.5,
            low=4.2,
            close=4.3,
            volume=1000,
            split_adjusted_close=4.3,
        )
    )
    return bars


def _fmp_bars_no_drawdown():
    bars = _fmp_bars()
    out = []
    for bar in bars:
        if bar.date < DAY.isoformat():
            out.append(
                FmpBar(
                    date=bar.date,
                    open=4.0,
                    high=4.1,
                    low=3.9,
                    close=4.0,
                    volume=bar.volume,
                    split_adjusted_close=4.0,
                )
            )
        else:
            out.append(bar)
    return out


def _add_hur(db_session, ticker, *, day=DAY, output_hash):
    db_session.add(
        HistoricalUniverseReconstruction(
            replay_date=day,
            ticker=ticker,
            normalized_symbol=ticker,
            inclusion_status="included",
            source="fixture",
            source_provenance_json="{}",
            reconstruction_method="fixture",
            pit_filter_status_json="{}",
            input_hash=f"hur-input-{output_hash}",
            output_hash=output_hash,
        )
    )


def _load_report_summarizer():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "summarize_i12_pit_report.py"
    )
    spec = importlib.util.spec_from_file_location(
        "summarize_i12_pit_report",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _hur_identity_hash(db_session, ticker, *, day=DAY, source_schema="public"):
    row = (
        db_session.query(HistoricalUniverseReconstruction)
        .filter(
            HistoricalUniverseReconstruction.replay_date == day,
            HistoricalUniverseReconstruction.normalized_symbol == ticker,
            HistoricalUniverseReconstruction.inclusion_status == "included",
        )
        .one()
    )
    return _hur_source_row_from_model(
        row,
        source_schema=source_schema,
    ).source_hur_identity_hash


def _minute_bars():
    return _clean_minute_bars(DAY, _polygon_bars())


def _polygon_bars(volume=25):
    market_open = us_equity_session_open_timestamp(DAY)
    bars = []
    for idx in range(11):
        ts = market_open + timedelta(minutes=idx)
        bars.append(
            PolygonBar(
                timestamp=int(ts.timestamp() * 1000),
                open=4.02 + idx * 0.01,
                high=4.05 + idx * 0.01,
                low=4.00,
                close=4.03 + idx * 0.01,
                volume=volume,
            )
        )
    exit_ts = datetime(2026, 6, 16, 19, 55, tzinfo=timezone.utc)
    bars.append(
        PolygonBar(
            timestamp=int(exit_ts.timestamp() * 1000),
            open=4.5,
            high=4.55,
            low=4.4,
            close=4.52,
            volume=100,
        )
    )
    return bars


def _sparse_polygon_bars(*, include_decision_bar=False, decision_close=999.0):
    market_open = us_equity_session_open_timestamp(DAY)
    raw = [
        PolygonBar(
            timestamp=int(market_open.timestamp() * 1000),
            open=4.02,
            high=4.05,
            low=4.00,
            close=4.03,
            volume=100,
        ),
        PolygonBar(
            timestamp=int((market_open + timedelta(minutes=5)).timestamp() * 1000),
            open=4.07,
            high=4.10,
            low=4.06,
            close=4.08,
            volume=100,
        ),
    ]
    if include_decision_bar:
        raw.append(
            PolygonBar(
                timestamp=int(DECISION_TS.timestamp() * 1000),
                open=decision_close,
                high=decision_close,
                low=decision_close,
                close=decision_close,
                volume=1_000_000,
            )
        )
    return raw


def _sparse_minute_bars(*, include_decision_bar=False, decision_close=999.0):
    return _clean_minute_bars(
        DAY,
        _sparse_polygon_bars(
            include_decision_bar=include_decision_bar,
            decision_close=decision_close,
        ),
    )


def _quote(ts, bid=10.0, ask=10.05, ask_size=100, bid_size=100):
    return AlpacaQuote(
        symbol="PIT",
        bid_price=bid,
        ask_price=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        timestamp=ts.isoformat().replace("+00:00", "Z"),
        raw={
            "bp": bid,
            "ap": ask,
            "bs": bid_size,
            "as": ask_size,
            "t": ts.isoformat().replace("+00:00", "Z"),
        },
    )


def _complete_quotes():
    return {
        "entry": [_quote(ts=DECISION_TS, bid=10, ask=10.05, ask_size=100)],
        "same_day_exit": [
            _quote(ts=datetime(2026, 6, 16, 19, 55, tzinfo=timezone.utc), bid=10.5, ask=10.55, ask_size=100)
        ],
        "next_open_exit": [
            _quote(ts=datetime(2026, 6, 17, 13, 31, tzinfo=timezone.utc), bid=10.2, ask=10.25, ask_size=100)
        ],
    }


def _provider_error_response(provider, ticker, error_type):
    return AdapterResponse(
        data=None,
        lineage=_lineage(provider, ticker),
        error=ProviderError(
            provider=provider,
            endpoint=f"/fixture/{ticker}",
            status_code=503,
            error_type=error_type,
            message=error_type,
            retryable=True,
        ),
    )


def _candidate_row(
    ticker="PIT",
    *,
    path_mode="strict_contiguous",
    source_hur_identity_hash=None,
    decision_time_label="09:40",
    candidate_status="passed",
    coverage_status="ok",
):
    source_hur_identity_hash = source_hur_identity_hash or f"hur-{ticker}"
    identity_hash = (
        f"content-{ticker}-{path_mode}-{source_hur_identity_hash}-{decision_time_label}"
    )
    suffix = stable_hash({
        "ticker": ticker,
        "path_mode": path_mode,
        "source_hur_identity_hash": source_hur_identity_hash,
        "decision_time_label": decision_time_label,
    })[:12]
    return I12PitCandidate(
        i12_pit_candidate_id=f"cand-{ticker}-{suffix}",
        ticker=ticker,
        decision_date=DAY,
        decision_ts=_decision_ts_for_label(decision_time_label),
        decision_time_label=decision_time_label,
        path_mode=path_mode,
        feature_asof_ts=DECISION_TS,
        candidate_status=candidate_status,
        coverage_status=coverage_status,
        feature_json="{}",
        gate_values_json="{}",
        leakage_guard_json="{}",
        source_bars_json=json.dumps({
            "source_hur_identity_hash": source_hur_identity_hash,
        }),
        candidate_attempt_hash=_candidate_attempt_hash(
            ticker=ticker,
            trading_date=DAY,
            decision_ts=_decision_ts_for_label(decision_time_label),
            decision_time_label=decision_time_label,
            source_hur_identity_hash=source_hur_identity_hash,
            path_mode=path_mode,
        ),
        is_active=True,
        input_hash=f"input-{ticker}",
        candidate_identity_hash=identity_hash,
        label_hash=f"label-{ticker}",
        content_hash=identity_hash,
    )


def _persist_candidate(db_session, ticker, **kwargs):
    candidate = _candidate_row(ticker, **kwargs)
    db_session.add(candidate)
    db_session.flush()
    return candidate


def _persist_complete_candidate_replay(
    db_session,
    ticker,
    *,
    path_mode="strict_contiguous",
    source_hur_identity_hash=None,
    decision_time_label="09:40",
):
    candidate = _persist_candidate(
        db_session,
        ticker,
        path_mode=path_mode,
        source_hur_identity_hash=source_hur_identity_hash,
        decision_time_label=decision_time_label,
    )
    entry = _persist_quote(db_session, candidate, role="entry")
    same_day = _persist_quote(db_session, candidate, role="same_day_exit", bid=10.5, ask=10.55)
    next_open = _persist_quote(db_session, candidate, role="next_open_exit", bid=10.2, ask=10.25)
    for exit_role, exit_quote in {
        "same_day_exit": same_day,
        "next_open_exit": next_open,
    }.items():
        result = evaluate_quote_cost_replay(
            entry_quote=entry,
            exit_quote=exit_quote,
            exit_role=exit_role,
            intended_order_usd=50,
            max_spread_bps=200,
            slippage_bps=0,
        )
        _persist_cost(db_session, candidate, exit_role, result)
    return candidate


def _persist_quote(
    db_session,
    candidate,
    *,
    role,
    bid=10.0,
    ask=10.05,
    bid_size=100,
    ask_size=100,
    status="ok",
    conditions=None,
    attempt_hash=None,
    is_active=True,
):
    spread = (ask - bid) / ((ask + bid) / 2.0) * 10000 if ask >= bid else None
    bid_notional = bid * bid_size
    ask_notional = ask * ask_size
    executable_side = "buy" if role == "entry" else "sell"
    executable_notional = ask_notional if executable_side == "buy" else bid_notional
    content_hash = stable_hash({
        "candidate": candidate.content_hash,
        "role": role,
        "status": status,
        "bid": bid,
        "ask": ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "conditions": conditions or [],
    })
    row = I12PitQuoteReplay(
        i12_pit_candidate_id=candidate.i12_pit_candidate_id,
        ticker=candidate.ticker,
        decision_date=candidate.decision_date,
        decision_ts=candidate.decision_ts,
        quote_role=role,
        target_ts=candidate.decision_ts,
        window_start_ts=candidate.decision_ts - timedelta(minutes=2),
        window_end_ts=candidate.decision_ts,
        quote_ts=candidate.decision_ts,
        quote_age_seconds=0.0,
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
        feed="sip",
        source="fixture",
        quote_size_basis="shares_post_2025_11_03",
        coverage_status=status,
        raw_json=json.dumps({"c": conditions or []}),
        quote_replay_attempt_hash=attempt_hash or stable_hash({
            "candidate": candidate.content_hash,
            "role": role,
            "content": content_hash,
        }),
        is_active=is_active,
        content_hash=content_hash,
    )
    db_session.add(row)
    db_session.flush()
    return row


def _persist_cost(
    db_session,
    candidate,
    exit_role,
    result,
    *,
    attempt_hash=None,
    is_active=True,
):
    content_hash = stable_hash({
        "candidate": candidate.content_hash,
        "exit_role": exit_role,
        "tradeability_status": result.tradeability_status,
        "skipped_reason": result.skipped_reason,
        "modeled_return": result.modeled_return,
    })
    row = I12PitCostReplay(
        i12_pit_candidate_id=candidate.i12_pit_candidate_id,
        ticker=candidate.ticker,
        decision_date=candidate.decision_date,
        decision_ts=candidate.decision_ts,
        exit_role=exit_role,
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
        cost_replay_attempt_hash=attempt_hash or stable_hash({
            "candidate": candidate.content_hash,
            "exit_role": exit_role,
            "content": content_hash,
        }),
        is_active=is_active,
        content_hash=content_hash,
    )
    db_session.add(row)
    db_session.flush()
    return row


def _lineage(provider, ticker):
    now = datetime(2026, 6, 16, tzinfo=timezone.utc)
    return LineageMeta(
        provider=provider,
        endpoint="fixture",
        request_timestamp=now,
        asof_timestamp=now,
        raw_payload_hash=stable_hash({"provider": provider, "ticker": ticker}),
    )


def _role_from_window(start, end):
    midpoint = start + (end - start) / 2
    midpoint_et = midpoint.astimezone(EASTERN)
    if midpoint_et.date() == DAY and midpoint_et.hour == 9:
        return "entry"
    if midpoint_et.date() == DAY and midpoint_et.hour == 15:
        return "same_day_exit"
    return "next_open_exit"


def _i12_feature_schema():
    fields = [
        {
            "dtype": "float",
            "name": name,
            "path": name,
            "role": "feature",
            "source": "feature_snapshot_json",
        }
        for name in (
            "mom20",
            "off_low252",
            "sigma20",
            "distance_from_max252",
            "drawdown_from_max252",
            "gap",
            "prev_day_return",
            "prev_day_green",
            "projected_volume_ratio_at_confirmation",
        )
    ]
    return {
        "schema_version": "stage1_i12_live_features_9f_v2",
        "pattern_id": "I12",
        "pattern_clock": "intraday",
        "fields": fields,
    }
