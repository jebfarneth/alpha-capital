"""M1 PEAD producer and assembly tests."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pytest

from alpha.assembly.m1_daily import (
    MARKET_FACTOR_SYMBOL,
    FosterComputation,
    PriceDelayComputation,
    assemble_m1_daily,
    compute_foster_sue,
    compute_price_delay_metric,
    effective_announcement_session,
    rank_friction_metrics,
)
from alpha.data.contracts import AdapterResponse, LineageMeta, stable_hash
from alpha.data.fmp import (
    EARNINGS_CALENDAR_ENDPOINT,
    EARNINGS_HISTORY_ENDPOINT,
    HISTORICAL_PRICE_FULL_ENDPOINT,
    FmpBar,
    FmpEarningsCalendarEvent,
    FmpEpsRecord,
)
from alpha.db.models import (
    CanonicalUniverseScan,
    M1EarningsEvent,
    M1FrictionSnapshot,
    SignalRegistry,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.jobs.m1_daily import M1DailyAssemblyJob, _fetch_earnings_calendar_window
from alpha.jobs.runner import run_job
from alpha.patterns.m1 import M1Detector


def _ts() -> datetime:
    return datetime(2026, 5, 20, 21, 0, tzinfo=timezone.utc)


def _fiscal_end(index: int) -> str:
    year = index // 4
    quarter = index % 4 + 1
    month = quarter * 3
    day = 31 if quarter in {1, 4} else 30
    return date(year, month, day).isoformat()


def _eps_base(index: int, current_index: int) -> float:
    rel = index - current_index
    return (
        1.20
        + 0.018 * rel
        + 0.035 * math.sin(index * 0.91)
        + 0.017 * math.cos(index * 0.37)
    )


def _sample_std(values: List[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _foster_fixture(
    *,
    ticker: str = "FIRE",
    current_year: int = 2026,
    current_quarter: int = 1,
    sue_target: float = 2.5,
    restated_history_current: bool = False,
) -> Tuple[FmpEarningsCalendarEvent, List[FmpEpsRecord], float]:
    current_index = current_year * 4 + (current_quarter - 1)
    values = {
        idx: _eps_base(idx, current_index)
        for idx in range(current_index - 28, current_index + 1)
    }
    seasonal_diffs = [
        values[current_index - lag] - values[current_index - lag - 4]
        for lag in range(4, 16)
    ]
    expected = values[current_index - 4] + sum(seasonal_diffs) / len(seasonal_diffs)
    sigma = _sample_std(seasonal_diffs)
    actual = expected + sue_target * sigma
    records = []
    for idx in sorted(values, reverse=True):
        eps = actual if idx == current_index else values[idx]
        if restated_history_current and idx == current_index:
            eps = actual + 0.25
        records.append(FmpEpsRecord(
            symbol=ticker,
            date="2026-05-20" if idx == current_index else None,
            eps=eps,
            fiscal_date_ending=_fiscal_end(idx),
            fiscal_year=idx // 4,
            fiscal_quarter=idx % 4 + 1,
            filing_date="2026-05-20" if idx == current_index else _fiscal_end(idx),
            accepted_date="2026-05-20T18:00:00+00:00" if idx == current_index else "2026-01-01T18:00:00+00:00",
        ))
    event = FmpEarningsCalendarEvent(
        symbol=ticker,
        date="2026-05-20",
        actual_eps=actual,
        estimated_eps=expected - sigma,
        announcement_time="bmo",
        fiscal_date_ending=_fiscal_end(current_index),
        fiscal_year=current_year,
        fiscal_quarter=current_quarter,
    )
    return event, records, expected


def _snapshot(ticker: str = "FIRE") -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        universe_snapshot_id=f"snap-{ticker}",
        asof_timestamp=datetime(2026, 5, 20, 20, 0, tzinfo=timezone.utc),
        price=8.75,
        market_cap=75_000_000,
        primary_exchange="NASDAQ",
        security_type="common_stock",
        operating_universe_inclusion=True,
        liquidity_score=0.7,
        hazard_score=None,
        source_lineage_hash=f"snapshot-hash-{ticker}",
    )


def _price_bars_from_returns(
    symbol: str,
    returns: List[float],
    *,
    start: date = date(2025, 1, 1),
    initial_close: float = 20.0,
) -> List[FmpBar]:
    closes = [initial_close]
    for ret in returns:
        closes.append(closes[-1] * (1.0 + ret))
    bars = []
    for idx, close in enumerate(closes):
        day = start + timedelta(days=idx * 7)
        bars.append(FmpBar(
            date=day.isoformat(),
            open=close * 0.99,
            high=close * 1.01,
            low=close * 0.98,
            close=close,
            volume=100_000 + idx,
            split_adjusted_close=close,
        ))
    return bars


def _market_and_stock_bars(
    ticker: str = "FIRE",
    *,
    weeks: int = 72,
    salt: float = 0.0,
) -> Tuple[List[FmpBar], List[FmpBar]]:
    market_returns: List[float] = []
    stock_returns: List[float] = []
    for idx in range(1, weeks):
        market_ret = (
            0.002
            + 0.007 * math.sin(idx * 0.37)
            + 0.004 * math.cos(idx * 1.09)
            + (((idx * 13) % 17) - 8) * 0.00035
        )
        lag_market = market_returns[-1] if market_returns else 0.0
        lag_stock = stock_returns[-1] if stock_returns else 0.0
        idiosyncratic = (
            0.004 * math.sin(idx * (2.11 + salt))
            + 0.003 * math.cos(idx * (3.07 + salt))
            + (((idx * 23) % 19) - 9) * 0.00025
        )
        stock_ret = (
            0.22 * market_ret
            + 0.32 * lag_market
            + 0.18 * lag_stock
            + idiosyncratic
            + salt * 0.0003
        )
        market_returns.append(market_ret)
        stock_returns.append(stock_ret)
    return (
        _price_bars_from_returns(MARKET_FACTOR_SYMBOL, market_returns, initial_close=400.0),
        _price_bars_from_returns(ticker, stock_returns, initial_close=10.0 + salt),
    )


def _lineage(endpoint: str, data, asof: datetime) -> LineageMeta:
    return LineageMeta(
        provider="FMP",
        endpoint=endpoint,
        request_timestamp=asof,
        asof_timestamp=asof,
        raw_payload_hash=stable_hash({"endpoint": endpoint, "data": data}),
        source_authority="test",
    )


class TestFosterSueProducer:
    def test_computes_foster_sue_series_from_contiguous_history(self):
        event, history, _ = _foster_fixture(sue_target=2.5)

        result = compute_foster_sue(
            event=event,
            eps_history=history,
            effective_session=date(2026, 5, 20),
        )

        assert result.status == "computed"
        assert result.foster_history_quarters_used == 16
        assert result.sue_foster == pytest.approx(2.5)
        assert result.sue_sign_current == 1
        assert result.rho1 is not None
        assert len(result.sue_series) >= 8
        assert result.split_adjustment_continuity_check == "passed"

    def test_near_zero_sigma_delta_refuses_fabricated_extreme_sue(self):
        current_index = 2026 * 4
        records = []
        for idx in range(current_index - 28, current_index + 1):
            records.append(FmpEpsRecord(
                symbol="FLAT",
                eps=2.0 if idx == current_index else 1.0 + idx * 1e-12,
                fiscal_date_ending=_fiscal_end(idx),
                fiscal_year=idx // 4,
                fiscal_quarter=idx % 4 + 1,
                accepted_date="2026-01-01T18:00:00+00:00",
            ))
        event = FmpEarningsCalendarEvent(
            symbol="FLAT",
            date="2026-05-20",
            actual_eps=2.0,
            estimated_eps=1.0,
            announcement_time="bmo",
            fiscal_date_ending=_fiscal_end(current_index),
            fiscal_year=2026,
            fiscal_quarter=1,
        )

        result = compute_foster_sue(
            event=event,
            eps_history=records,
            effective_session=date(2026, 5, 20),
        )

        assert result.status == "insufficient_history"
        assert "near_zero_sigma_delta_eps" in result.diagnostics
        assert result.sue_foster is None

    def test_missing_required_fiscal_quarter_rejects_without_zero_fill(self):
        event, history, _ = _foster_fixture()
        gap_index = event.fiscal_year * 4 + (event.fiscal_quarter - 1) - 7
        history = [
            record for record in history
            if not (
                record.fiscal_year * 4 + (record.fiscal_quarter - 1) == gap_index
            )
        ]

        result = compute_foster_sue(
            event=event,
            eps_history=history,
            effective_session=date(2026, 5, 20),
        )

        assert result.status == "insufficient_history"
        assert any("missing_required_foster_lags:7" in item for item in result.diagnostics)
        assert result.sue_foster is None

    def test_history_eps_basis_anchors_current_surprise_when_calendar_differs(self):
        event, history, _ = _foster_fixture(
            sue_target=2.0,
            restated_history_current=True,
        )

        result = compute_foster_sue(
            event=event,
            eps_history=history,
            effective_session=date(2026, 5, 20),
        )

        assert result.status == "computed"
        assert result.sue_foster != pytest.approx(2.0)
        assert result.actual_eps == result.eps_history_current_eps
        assert result.restatement_exposure is True
        assert "calendar_eps_differs_from_eps_history_current" in result.diagnostics
        assert result.actual_eps != event.actual_eps

    def test_unresolved_current_quarter_refuses_instead_of_overwriting_prior_filing(self):
        event, history, _ = _foster_fixture(sue_target=2.0)
        event.fiscal_year = None
        event.fiscal_quarter = None
        event.fiscal_date_ending = None
        current_index = 2026 * 4
        for record in history:
            if record.fiscal_year * 4 + (record.fiscal_quarter - 1) == current_index:
                record.accepted_date = "2026-05-22T18:00:00+00:00"

        result = compute_foster_sue(
            event=event,
            eps_history=history,
            effective_session=date(2026, 5, 20),
            asof_timestamp=_ts(),
        )

        assert result.status == "insufficient_history"
        assert "current_eps_basis_unverified" in result.diagnostics
        assert any(item.startswith("eps_accepted_after_asof") for item in result.diagnostics)
        assert result.fiscal_quarter == 1

    def test_current_eps_restatement_after_asof_is_not_consumed(self):
        event, history, _ = _foster_fixture(sue_target=2.0)
        current_index = event.fiscal_year * 4 + (event.fiscal_quarter - 1)
        original_current = next(
            record for record in history
            if record.fiscal_year * 4 + (record.fiscal_quarter - 1) == current_index
        )
        history.insert(0, FmpEpsRecord(
            symbol=event.symbol,
            eps=original_current.eps + 0.75,
            fiscal_date_ending=original_current.fiscal_date_ending,
            fiscal_year=original_current.fiscal_year,
            fiscal_quarter=original_current.fiscal_quarter,
            accepted_date="2026-05-25T18:00:00+00:00",
        ))

        result = compute_foster_sue(
            event=event,
            eps_history=history,
            effective_session=date(2026, 5, 20),
            asof_timestamp=_ts(),
        )

        assert result.status == "computed"
        assert result.sue_foster == pytest.approx(2.0)
        assert result.actual_eps == pytest.approx(original_current.eps)

    def test_rho1_uses_pit_series_not_future_quarters(self):
        event, history, _ = _foster_fixture(sue_target=2.0)
        baseline = compute_foster_sue(
            event=event,
            eps_history=list(history),
            effective_session=date(2026, 5, 20),
        )
        current_index = event.fiscal_year * 4 + (event.fiscal_quarter - 1)
        with_future = list(history)
        for offset in range(1, 6):
            idx = current_index + offset
            with_future.append(FmpEpsRecord(
                symbol=event.symbol,
                eps=10.0 + offset,
                fiscal_date_ending=_fiscal_end(idx),
                fiscal_year=idx // 4,
                fiscal_quarter=idx % 4 + 1,
                accepted_date="2026-05-19T18:00:00+00:00",
            ))

        replay = compute_foster_sue(
            event=event,
            eps_history=with_future,
            effective_session=date(2026, 5, 20),
            asof_timestamp=_ts(),
        )

        assert replay.status == "computed"
        assert replay.rho1 == pytest.approx(baseline.rho1)

    def test_unknown_announcement_time_defaults_to_next_session(self):
        event, _, _ = _foster_fixture()
        event.announcement_time = None

        assert effective_announcement_session(event) == date(2026, 5, 21)

    def test_bmo_announcement_is_same_session(self):
        event, _, _ = _foster_fixture()
        event.announcement_time = "bmo"

        assert effective_announcement_session(event) == date(2026, 5, 20)


class TestPriceDelayProducer:
    def test_computes_return_only_d1_and_sigma_from_spy_factor(self):
        market_bars, stock_bars = _market_and_stock_bars()

        result = compute_price_delay_metric(
            ticker="FIRE",
            stock_bars=stock_bars,
            market_bars=market_bars,
        )

        assert result.status == "computed"
        assert 0.0 <= result.d1 <= 1.0
        assert result.sigma_epsilon is not None
        assert result.sigma_epsilon > 0
        assert result.weekly_return_count >= 60
        assert result.market_factor_symbol == MARKET_FACTOR_SYMBOL

    def test_ranks_friction_over_operating_universe_population(self):
        metrics = {
            "A": PriceDelayComputation(ticker="A", status="computed", d1=0.1, sigma_epsilon=0.01),
            "B": PriceDelayComputation(ticker="B", status="computed", d1=0.5, sigma_epsilon=0.03),
            "C": PriceDelayComputation(ticker="C", status="computed", d1=0.9, sigma_epsilon=0.05),
            "D": PriceDelayComputation(ticker="D", status="computed", d1=0.8, sigma_epsilon=0.02),
        }

        ranked = rank_friction_metrics(metrics)

        assert ranked["A"].d1_decile == 3
        assert ranked["B"].d1_decile == 5
        assert ranked["D"].d1_decile == 8
        assert ranked["C"].d1_decile == 10
        assert ranked["A"].sigma_epsilon_percentile == 0.0
        assert ranked["B"].sigma_epsilon_percentile == 0.0
        assert ranked["D"].sigma_epsilon_percentile == 0.0
        assert ranked["C"].sigma_epsilon_percentile == 1.0


class TestM1DailyAssembly:
    def test_singleton_announcing_cohort_does_not_auto_fire(self):
        event, history, _ = _foster_fixture(sue_target=3.0)
        foster = compute_foster_sue(
            event=event,
            eps_history=history,
            effective_session=date(2026, 5, 20),
        )
        friction = PriceDelayComputation(
            ticker="FIRE",
            status="computed",
            d1=0.74,
            d1_decile=8,
            sigma_epsilon=0.025,
            sigma_epsilon_percentile=0.65,
        )

        result = assemble_m1_daily(
            snapshots=[_snapshot("FIRE")],
            foster_by_ticker={"FIRE": foster},
            friction_by_ticker={"FIRE": friction},
            cutoff_timestamp=_ts(),
            decision_date="2026-05-20",
            evidence_session_date="2026-05-20",
            next_execution_session="2026-05-21",
            source_lineage_hash="earnings-hash",
        )

        assert result.assembled_count == 0
        assert result.insufficient_count == 1
        assert result.diagnostics[0].diagnostic_type == "announcing_cohort_too_small"

    def test_assembler_populates_frozen_detector_contract_and_fires(self):
        event, history, _ = _foster_fixture(sue_target=2.7)
        foster = compute_foster_sue(
            event=event,
            eps_history=history,
            effective_session=date(2026, 5, 20),
        )
        friction = PriceDelayComputation(
            ticker="FIRE",
            status="computed",
            d1=0.74,
            d1_decile=8,
            sigma_epsilon=0.025,
            sigma_epsilon_percentile=0.65,
        )

        result = assemble_m1_daily(
            snapshots=[_snapshot("FIRE")],
            foster_by_ticker={
                "FIRE": foster,
                "LOW1": FosterComputation(ticker="LOW1", status="computed", sue_foster=0.2),
                "LOW2": FosterComputation(ticker="LOW2", status="computed", sue_foster=0.4),
                "LOW3": FosterComputation(ticker="LOW3", status="computed", sue_foster=0.6),
                "LOW4": FosterComputation(ticker="LOW4", status="computed", sue_foster=0.8),
            },
            friction_by_ticker={"FIRE": friction},
            next_earnings_by_ticker={"FIRE": 10},
            cutoff_timestamp=_ts(),
            decision_date="2026-05-20",
            evidence_session_date="2026-05-20",
            next_execution_session="2026-05-21",
            source_lineage_hash="earnings-hash",
        )

        assert result.assembled_count == 1
        inp = result.inputs[0]
        for key in (
            "sue_foster",
            "delta_t_trading_days",
            "sue_signed_percentile",
            "rho1",
            "sue_sign_current",
            "sue_sign_prior",
            "d1_decile",
            "sigma_epsilon_percentile",
            "sue_streak_length",
        ):
            assert key in inp.market_data
        assert inp.market_data["market_factor_symbol"] == MARKET_FACTOR_SYMBOL
        assert inp.market_data["next_earnings_trading_days_from_signal"] == 10

        detection = M1Detector().detect(inp)

        assert detection.has_signal
        assert detection.signals[0].signal_horizon == "9d"

    @pytest.mark.parametrize(
        ("updates", "remove_keys", "reason"),
        [
            ({"delta_t_trading_days": 1.25}, (), "invalid_delta_t"),
            ({"delta_t_trading_days": 16}, (), "announcement_too_old"),
            ({"sue_foster": -0.25}, (), "sue_not_positive"),
            ({}, ("sue_signed_percentile",), "missing_sue_percentile"),
            ({"sue_signed_percentile": "bad"}, (), "invalid_sue_percentile"),
            ({"sue_signed_percentile": 0.70}, (), "sue_below_threshold"),
            ({"next_earnings_trading_days_from_signal": 1}, (), "no_remaining_hold_window"),
        ],
    )
    def test_detector_rejects_assembled_input_failure_paths(
        self,
        updates,
        remove_keys,
        reason,
    ):
        market_data = {
            "sue_foster": 2.75,
            "delta_t_trading_days": 0,
            "sue_signed_percentile": 0.92,
            "rho1": 0.3,
            "sue_sign_current": 1,
            "sue_sign_prior": 1,
            "d1_decile": 8,
            "sigma_epsilon_percentile": 0.65,
            "sue_streak_length": 2,
            "next_earnings_trading_days_from_signal": 65,
            "operating_universe_inclusion": True,
            "market_data_status": "current",
            "halt_status": "clear",
            "corporate_action_filter_passed": True,
            "earnings_event_id": "event-1",
            "announcement_date": "2026-05-20",
        }
        market_data.update(updates)
        for key in remove_keys:
            market_data.pop(key, None)

        detection = M1Detector().detect(SimpleNamespace(
            ticker="FIRE",
            asof_timestamp=_ts(),
            market_data=market_data,
            fundamental_data={},
            event_data={},
            lineage_hashes=["lineage-hash"],
            universe_snapshot_id="snap-FIRE",
        ))

        assert detection.features.features["rejection_reason"] == reason
        assert not detection.has_signal


class FakeM1Adapter:
    def __init__(self):
        self.tickers = ["FIRE", "LOW1", "LOW2", "LOW3", "LOW4"]
        sue_targets = {
            "FIRE": 2.7,
            "LOW1": -1.5,
            "LOW2": -1.0,
            "LOW3": -0.5,
            "LOW4": 0.0,
        }
        self.trailing_events = []
        self.histories = {}
        for ticker in self.tickers:
            event, history, _ = _foster_fixture(
                ticker=ticker,
                sue_target=sue_targets[ticker],
            )
            self.trailing_events.append(event)
            self.histories[ticker] = history
        next_event = FmpEarningsCalendarEvent(
            symbol="FIRE",
            date="2026-06-04",
            actual_eps=None,
            estimated_eps=1.0,
            announcement_time="bmo",
            fiscal_date_ending="2026-06-30",
            fiscal_year=2026,
            fiscal_quarter=2,
        )
        market_bars, stock_bars = _market_and_stock_bars("FIRE")
        self.history = history
        self.next_event = next_event
        self.price_bars = {MARKET_FACTOR_SYMBOL: market_bars}
        for ticker in self.tickers:
            _market, bars = _market_and_stock_bars(ticker)
            self.price_bars[ticker] = bars

    def get_earnings_calendar(self, *, from_date, to_date, asof=None, symbol=None):
        if from_date == date(2026, 5, 20):
            data = list(self.trailing_events)
        elif from_date == date(2026, 6, 4):
            data = [self.next_event]
        else:
            data = []
        return AdapterResponse(
            data=data,
            lineage=_lineage(EARNINGS_CALENDAR_ENDPOINT, data, asof or _ts()),
        )

    def get_earnings_history(self, ticker, *, limit=40, asof=None):
        return AdapterResponse(
            data=self.histories[ticker],
            lineage=_lineage(EARNINGS_HISTORY_ENDPOINT, self.histories[ticker], asof or _ts()),
        )

    def get_historical_price(self, ticker, from_date=None, to_date=None, asof=None, **kwargs):
        data = self.price_bars[ticker]
        return AdapterResponse(
            data=data,
            lineage=_lineage(HISTORICAL_PRICE_FULL_ENDPOINT, data, asof or _ts()),
        )


class TestEarningsCalendarCoverage:
    def test_calendar_window_refuses_provider_cap_instead_of_silent_truncation(self):
        class CappedAdapter:
            def get_earnings_calendar(self, *, from_date, to_date, asof=None, symbol=None):
                data = [
                    FmpEarningsCalendarEvent(
                        symbol=f"T{idx}",
                        date=from_date.isoformat(),
                        actual_eps=1.0,
                    )
                    for idx in range(4000)
                ]
                return AdapterResponse(
                    data=data,
                    lineage=_lineage(EARNINGS_CALENDAR_ENDPOINT, data, asof or _ts()),
                )

        fetch = _fetch_earnings_calendar_window(
            CappedAdapter(),
            from_date=date(2026, 5, 20),
            to_date=date(2026, 5, 20),
            asof=_ts(),
        )

        assert not fetch.ok
        assert fetch.coverage["max_page_row_count"] == 4000
        assert fetch.coverage["capped_page_dates"] == ["2026-05-20"]
        assert any("provider row cap" in error["message"] for error in fetch.errors)


def _setup_canonical_universe(db_session, tickers: Optional[List[str]] = None) -> None:
    tickers = tickers or ["FIRE"]
    scan = UniverseScan(
        scan_id="m1-test-scan",
        trading_date="2026-05-20",
        asof_timestamp=datetime(2026, 5, 20, 20, 0, tzinfo=timezone.utc),
        raw_count=len(tickers),
        deduped_count=len(tickers),
        included_count=len(tickers),
        excluded_count=0,
        run_status="finished",
        source_lineage_hash="scan-hash",
    )
    db_session.add(scan)
    db_session.flush()
    db_session.add(CanonicalUniverseScan(
        trading_date="2026-05-20",
        scan_id=scan.scan_id,
        selection_reason="test",
    ))
    for ticker in tickers:
        db_session.add(UniverseSnapshot(
            universe_snapshot_id=f"snap-{ticker}",
            scan_id=scan.scan_id,
            ticker=ticker,
            asof_timestamp=datetime(2026, 5, 20, 20, 0, tzinfo=timezone.utc),
            market_cap=75_000_000,
            price=8.75,
            primary_exchange="NASDAQ",
            security_type="common_stock",
            operating_universe_inclusion=True,
            source_lineage_hash=f"snapshot-hash-{ticker}",
        ))
    db_session.flush()


class TestM1DailyJob:
    def test_job_persists_friction_earnings_and_m1_signal(self, db_session):
        adapter = FakeM1Adapter()
        _setup_canonical_universe(db_session, adapter.tickers)
        job = M1DailyAssemblyJob(
            db_session,
            adapter=adapter,
            run_timestamp=_ts(),
        )

        result = run_job(
            db_session,
            job,
            params={"run_timestamp": _ts().isoformat()},
        )

        assert result.status == "finished"
        assert result.metrics["market_factor_symbol"] == MARKET_FACTOR_SYMBOL
        assert result.metrics["foster_computed_count"] == 5
        assert result.metrics["foster_insufficient_history_count"] == 0
        assert result.metrics["friction_computed_count"] == 5
        assert result.metrics["assembly"]["assembled_count"] == 5
        assert result.metrics["orchestration"]["total_signals_persisted"] >= 1

        earnings = db_session.query(M1EarningsEvent).filter_by(ticker="FIRE").one()
        friction = db_session.query(M1FrictionSnapshot).filter_by(ticker="FIRE").one()
        signal = db_session.query(SignalRegistry).filter_by(pattern_id="M1", ticker="FIRE").one()

        assert earnings.status == "computed"
        assert earnings.sue_foster is not None
        assert friction.market_factor_symbol == MARKET_FACTOR_SYMBOL
        assert friction.status == "computed"
        assert signal.ticker == "FIRE"
        assert signal.signal_horizon != "15d"
        assert signal.next_execution_session == "2026-05-21"
