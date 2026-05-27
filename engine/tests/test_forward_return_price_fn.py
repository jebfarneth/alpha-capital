"""Production M4 price_fn tests."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import pytest

from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError, stable_hash
from alpha.data.fmp import FmpBar, HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.db.engine import schema_connect_args
from alpha.db.models import (
    DataLineage,
    ForwardReturnObservation,
    ForwardReturnObservationEvent,
    SignalRegistry,
)
from alpha.evidence.writer import record_feature_snapshot, record_signal
from alpha.jobs.forward_return import (
    LEGACY_NEXT_EXECUTION_SESSION_FALLBACK_REASON,
    M4_EXIT_GEOMETRY,
    M4_PRICE_SOURCE,
    M4_SPLIT_ADJUSTED_OPEN_BASIS_PROOF,
    ForwardReturnJob,
    m4_entry_exit_plan,
)
from alpha.jobs.run_forward_return import _live_timestamp_error
from alpha.jobs.runner import run_job


ENTRY_DATE = date(2026, 5, 26)
EXIT_DATE = date(2026, 6, 15)
MATURE_RUN_TS = datetime(2026, 6, 16, 21, 0, tzinfo=timezone.utc)
IMMATURE_RUN_TS = datetime(2026, 6, 10, 21, 0, tzinfo=timezone.utc)
SIGNAL_TS = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)
REQUEST_TS = datetime(2026, 6, 16, 21, 1, tzinfo=timezone.utc)
PAST_ENTRY_DATE = date(2026, 5, 5)
PAST_EXIT_DATE = date(2026, 5, 26)
PAST_MATURE_RUN_TS = datetime(2026, 5, 27, 14, 0, tzinfo=timezone.utc)


class FakeHistoricalAdapter:
    def __init__(
        self,
        bars_by_ticker: Optional[Dict[str, List[FmpBar]]] = None,
        errors_by_ticker: Optional[Dict[str, ProviderError]] = None,
        flags_by_ticker: Optional[Dict[str, dict]] = None,
        survivorship_by_ticker: Optional[Dict[str, object]] = None,
        survivorship_errors_by_ticker: Optional[Dict[str, ProviderError]] = None,
    ):
        self.bars_by_ticker = bars_by_ticker or {}
        self.errors_by_ticker = errors_by_ticker or {}
        self.flags_by_ticker = flags_by_ticker or {}
        self.survivorship_by_ticker = survivorship_by_ticker or {}
        self.survivorship_errors_by_ticker = survivorship_errors_by_ticker or {}
        self.calls = []
        self.survivorship_calls = []

    def get_historical_price(
        self,
        ticker,
        from_date=None,
        to_date=None,
        asof=None,
        **kwargs,
    ):
        self.calls.append({
            "ticker": ticker,
            "from_date": from_date,
            "to_date": to_date,
            "asof": asof,
            "kwargs": kwargs,
        })
        bars = self.bars_by_ticker.get(ticker, [])
        payload_hash = stable_hash([
            {
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "split_adjusted_close": bar.split_adjusted_close,
                "adj_close": bar.adj_close,
            }
            for bar in bars
        ])
        lineage = LineageMeta(
            provider="FMP",
            endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
            request_timestamp=REQUEST_TS,
            asof_timestamp=asof,
            raw_payload_hash=payload_hash,
            source_authority="test",
            data_quality_flags=self.flags_by_ticker.get(ticker),
        )
        if ticker in self.errors_by_ticker:
            return AdapterResponse(
                data=None,
                lineage=lineage,
                error=self.errors_by_ticker[ticker],
            )
        return AdapterResponse(data=bars, lineage=lineage)

    def get_survivorship_events(
        self,
        ticker,
        from_date=None,
        to_date=None,
        asof=None,
    ):
        self.survivorship_calls.append({
            "ticker": ticker,
            "from_date": from_date,
            "to_date": to_date,
            "asof": asof,
        })
        raw_events = self.survivorship_by_ticker.get(ticker, [])
        if isinstance(raw_events, dict):
            events = [raw_events]
        else:
            events = list(raw_events or [])
        lineage = LineageMeta(
            provider="TEST_SURVIVORSHIP",
            endpoint="/test/survivorship-events",
            request_timestamp=REQUEST_TS,
            asof_timestamp=asof,
            raw_payload_hash=stable_hash(events),
            source_authority="test",
        )
        if ticker in self.survivorship_errors_by_ticker:
            return AdapterResponse(
                data=None,
                lineage=lineage,
                error=self.survivorship_errors_by_ticker[ticker],
            )
        return AdapterResponse(data=events, lineage=lineage)


def _bar(
    day: date,
    open_price,
    *,
    high=None,
    low=None,
    close=None,
    split_adjusted_close=None,
    adj_close=None,
) -> FmpBar:
    if split_adjusted_close == "missing":
        split_adjusted_value = None
    else:
        split_adjusted_value = (
            open_price if split_adjusted_close is None else split_adjusted_close
        )
    return FmpBar(
        date=day.isoformat(),
        open=open_price,
        high=open_price if high is None else high,
        low=open_price if low is None else low,
        close=open_price if close is None else close,
        volume=1000,
        split_adjusted_close=split_adjusted_value,
        adj_close=adj_close,
    )


def _bars(
    entry_open=10.0,
    exit_open=12.0,
    *,
    entry_date=ENTRY_DATE,
    exit_date=EXIT_DATE,
) -> List[FmpBar]:
    return [
        _bar(entry_date, entry_open, adj_close=1.0),
        _bar(exit_date, exit_open, adj_close=999.0),
    ]


def _make_signal(
    db_session,
    ticker="ACME",
    *,
    next_execution_session="2026-05-26",
    trading_date="2026-05-26",
    signal_timestamp=SIGNAL_TS,
) -> str:
    features = {"decision_date": trading_date, "signal_generated": True}
    if next_execution_session is not None:
        features["next_execution_session"] = next_execution_session
    feat = record_feature_snapshot(
        db_session,
        pattern_id="M4",
        ticker=ticker,
        asof_timestamp=signal_timestamp,
        features=features,
        data_lineage_ids=[],
    )
    sig = record_signal(
        db_session,
        pattern_id="M4",
        ticker=ticker,
        direction="long",
        signal_timestamp=signal_timestamp,
        raw_signal_strength=0.9,
        raw_expected_edge=0.01,
        feature_snapshot_id=feat.feature_snapshot_id,
        signal_horizon="15d",
        signal_identity_hash=f"m4-{ticker}",
        trading_date=trading_date,
        next_execution_session=next_execution_session,
    )
    db_session.flush()
    return sig.signal_id


def _run_job(db_session, adapter, *, run_ts=MATURE_RUN_TS, max_attempts=3):
    return run_job(
        db_session,
        ForwardReturnJob(
            session=db_session,
            adapter=adapter,
            run_timestamp=run_ts,
            max_attempts=max_attempts,
        ),
        params={"run_timestamp": run_ts.isoformat()},
    )


def _obs(db_session):
    return db_session.query(ForwardReturnObservation).one()


def _path_obs(db_session, *, high: float, low: float = 10.0, close: float = 10.0):
    _make_signal(db_session)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(ENTRY_DATE, 10.0, high=10.0, low=10.0, close=10.0),
            _bar(date(2026, 6, 1), 10.0, high=high, low=low, close=close),
            _bar(EXIT_DATE, 10.0, high=10.0, low=10.0, close=10.0),
        ]
    })
    _run_job(db_session, adapter)
    return _obs(db_session)


def test_m4_entry_exit_calculation_counts_entry_as_day_one():
    plan = m4_entry_exit_plan(
        decision_date=date(2026, 5, 26),
        next_execution_session=ENTRY_DATE,
        current_evidence_session_date=date(2026, 6, 16),
    )

    assert plan.entry_session_date == ENTRY_DATE
    assert plan.exit_session_date == EXIT_DATE
    assert plan.mature is True
    assert plan.entry_resolution_reason is None


def test_live_future_timestamp_guard_rejects_future_time():
    now = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)

    assert _live_timestamp_error(
        "2026-05-27T12:04:00+00:00",
        now=now,
        tolerance=timedelta(minutes=5),
    ) is None
    assert _live_timestamp_error(
        "2026-05-27T12:06:00+00:00",
        now=now,
        tolerance=timedelta(minutes=5),
    ) == (
        "live run_timestamp is in the future; use explicit audited "
        "historical/backfill mode instead of --live time travel"
    )


def test_fresh_signal_before_entry_open_stays_pending_without_fetch(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": _bars()})
    premarket_ts = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)

    result = _run_job(db_session, adapter, run_ts=premarket_ts)

    sig = db_session.get(SignalRegistry, sid)
    assert result.metrics["pending"] == 1
    assert adapter.calls == []
    assert adapter.survivorship_calls == []
    assert sig.forward_return_status == "pending"
    obs = _obs(db_session)
    assert obs.status == "pending"
    assert obs.reason == "entry_session_not_open"
    assert db_session.query(ForwardReturnObservationEvent).count() == 1


def test_signal_after_entry_before_exit_stays_pending_without_survivorship(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": _bars()})
    after_entry_ts = datetime(2026, 5, 27, 21, 0, tzinfo=timezone.utc)

    result = _run_job(db_session, adapter, run_ts=after_entry_ts)

    sig = db_session.get(SignalRegistry, sid)
    assert result.metrics["pending"] == 1
    assert adapter.calls == []
    assert adapter.survivorship_calls == []
    assert sig.forward_return_status == "pending"
    obs = _obs(db_session)
    assert obs.status == "pending"
    assert obs.reason == "exit_session_not_complete"


def test_immature_signal_stays_pending_and_does_not_fetch(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": _bars()})

    result = _run_job(db_session, adapter, run_ts=IMMATURE_RUN_TS)

    sig = db_session.get(SignalRegistry, sid)
    assert result.metrics["pending"] == 1
    assert adapter.calls == []
    assert sig.forward_return_status == "pending"
    assert sig.forward_return_attempts == 0
    obs = _obs(db_session)
    assert obs.status == "pending"
    assert obs.reason == "exit_session_not_complete"
    assert obs.attempts == 0


def test_computed_mature_signal_uses_full_open_prices_and_updates_summary(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=12.0)})

    result = _run_job(db_session, adapter)

    assert result.ok
    assert result.metrics["computed"] == 1
    call = adapter.calls[0]
    assert call["from_date"] == ENTRY_DATE
    assert call["to_date"] == EXIT_DATE
    assert call["kwargs"]["adjusted"] is False
    assert call["kwargs"]["require_split_adjusted_close"] is True

    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "computed"
    assert sig.forward_return == 0.2
    assert sig.forward_return_attempts == 1
    assert sig.intended_entry_price == 10.0

    obs = _obs(db_session)
    assert obs.entry_session_date == "2026-05-26"
    assert obs.exit_session_date == "2026-06-15"
    assert obs.entry_price == 10.0
    assert obs.exit_price == 12.0
    assert obs.forward_return == 0.2
    assert obs.entry_price_source == M4_PRICE_SOURCE
    assert obs.exit_price_source == M4_PRICE_SOURCE
    assert obs.next_execution_session == "2026-05-26"
    assert obs.entry_basis_proof == M4_SPLIT_ADJUSTED_OPEN_BASIS_PROOF
    assert obs.exit_basis_proof == M4_SPLIT_ADJUSTED_OPEN_BASIS_PROOF
    assert obs.entry_data_lineage_id
    assert obs.exit_data_lineage_id
    assert db_session.query(ForwardReturnObservationEvent).count() == 1

    lineage = db_session.get(DataLineage, obs.entry_data_lineage_id)
    payload = json.loads(lineage.raw_payload_json)
    assert lineage.endpoint == HISTORICAL_PRICE_FULL_ENDPOINT
    assert payload["ticker"] == "ACME"
    assert payload["request"]["from"] == "2026-05-26"
    assert payload["request"]["to"] == "2026-06-15"
    assert payload["request"]["basis"] == "split_adjusted_ohlcv_full_endpoint"
    event = db_session.query(ForwardReturnObservationEvent).one()
    assert event.provider == "FMP"
    assert json.loads(event.provider_request_json)["price_field"] == "open"


def test_mature_past_signal_computes_endpoint_and_path_telemetry(db_session):
    sid = _make_signal(
        db_session,
        trading_date="2026-05-04",
        next_execution_session=PAST_ENTRY_DATE.isoformat(),
        signal_timestamp=datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc),
    )
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(PAST_ENTRY_DATE, 10.0, high=10.0, low=10.0, close=10.0),
            _bar(date(2026, 5, 12), 10.0, high=11.2, low=9.6, close=11.0),
            _bar(PAST_EXIT_DATE, 11.0, high=11.0, low=10.5, close=11.0),
        ]
    })

    result = _run_job(db_session, adapter, run_ts=PAST_MATURE_RUN_TS)

    sig = db_session.get(SignalRegistry, sid)
    assert result.metrics["computed"] == 1
    assert sig.forward_return_status == "computed"
    assert sig.forward_return == 0.1
    obs = _obs(db_session)
    assert obs.entry_session_date == "2026-05-05"
    assert obs.exit_session_date == "2026-05-26"
    assert obs.entry_data_lineage_id
    assert obs.exit_data_lineage_id
    assert obs.forward_return == 0.1
    assert round(obs.max_favorable_excursion, 6) == 0.12
    assert round(obs.max_adverse_excursion, 6) == -0.04
    assert obs.hit_t1_intraday is True
    assert obs.hit_t2_intraday is True
    assert obs.hit_t3_intraday is False
    assert obs.hit_stop_intraday is True
    assert obs.same_day_barrier_ambiguity is True
    event = db_session.query(ForwardReturnObservationEvent).one()
    assert event.forward_return == 0.1
    assert event.data_lineage_ids


def test_persisted_next_execution_session_drives_entry_after_close(db_session):
    sid = _make_signal(db_session, next_execution_session="2026-05-27")
    entry_date = date(2026, 5, 27)
    exit_date = date(2026, 6, 16)
    adapter = FakeHistoricalAdapter({
        "ACME": _bars(
            entry_open=9.0,
            exit_open=12.0,
            entry_date=entry_date,
            exit_date=exit_date,
        )
    })

    _run_job(db_session, adapter)

    call = adapter.calls[0]
    assert call["from_date"] == entry_date
    assert call["to_date"] == exit_date
    sig = db_session.get(SignalRegistry, sid)
    assert sig.intended_entry_price == 9.0
    obs = _obs(db_session)
    assert obs.next_execution_session == "2026-05-27"
    assert obs.entry_session_date == "2026-05-27"
    assert obs.exit_session_date == "2026-06-16"


def test_missing_next_execution_session_uses_legacy_fallback_with_reason(db_session):
    _make_signal(db_session, next_execution_session=None)
    adapter = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=12.0)})

    _run_job(db_session, adapter)

    obs = _obs(db_session)
    assert obs.status == "computed"
    assert obs.next_execution_session is None
    assert obs.entry_session_date == "2026-05-26"
    assert obs.reason == LEGACY_NEXT_EXECUTION_SESSION_FALLBACK_REASON
    event = db_session.query(ForwardReturnObservationEvent).one()
    assert event.reason == LEGACY_NEXT_EXECUTION_SESSION_FALLBACK_REASON


def test_missing_entry_price_is_retryable(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": [_bar(EXIT_DATE, 12.0)]})

    _run_job(db_session, adapter)

    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "missing_entry_price_retry"
    assert sig.outcome_unavailable_reason == "missing_entry_price"
    assert sig.forward_return_attempts == 1
    assert _obs(db_session).status == "missing_entry_price_retry"


def test_missing_exit_price_runs_survivorship_resolver_and_requires_review(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})

    _run_job(db_session, adapter)

    sig = db_session.get(SignalRegistry, sid)
    assert adapter.survivorship_calls[0]["ticker"] == "ACME"
    assert sig.forward_return_status == "survivorship_unresolved_review"
    assert sig.outcome_unavailable_reason == "survivorship_unresolved_no_source_event"
    obs = _obs(db_session)
    assert obs.status == "survivorship_unresolved_review"
    assert json.loads(obs.data_lineage_ids)
    event = db_session.query(ForwardReturnObservationEvent).one()
    assert event.status == "survivorship_unresolved_review"
    assert "survivorship_request" in json.loads(event.provider_request_json)


def test_source_backed_performance_delisting_computes_terminal_loss(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter(
        {"ACME": [_bar(ENTRY_DATE, 10.0)]},
        survivorship_by_ticker={
            "ACME": {
                "type": "delisting",
                "classification": "performance",
                "source_backed": True,
            }
        },
    )

    _run_job(db_session, adapter)

    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "computed"
    assert sig.forward_return == -1.0
    assert sig.outcome_unavailable_reason is None
    obs = _obs(db_session)
    assert obs.exit_price == 0.0
    assert obs.reason == "performance_delisting_shumway_terminal_loss"
    assert json.loads(obs.data_lineage_ids)


def test_active_halt_remains_halted_pending(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter(
        {"ACME": [_bar(ENTRY_DATE, 10.0)]},
        survivorship_by_ticker={
            "ACME": {
                "type": "halt",
                "status": "active",
                "may_resume": True,
                "source_backed": True,
            }
        },
    )

    _run_job(db_session, adapter)

    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "halted_pending"
    assert sig.outcome_unavailable_reason == "active_halt_or_suspension"
    assert _obs(db_session).status == "halted_pending"


def test_unresolved_corporate_action_requires_review(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter(
        {"ACME": [_bar(ENTRY_DATE, 10.0)]},
        survivorship_by_ticker={
            "ACME": {
                "type": "merger",
                "status": "unresolved",
                "source_backed": True,
            }
        },
    )

    _run_job(db_session, adapter)

    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "corporate_action_review"
    assert sig.outcome_unavailable_reason == "corporate_action_review"
    assert _obs(db_session).status == "corporate_action_review"


def test_acquisition_realized_payoff_computes_return(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter(
        {"ACME": [_bar(ENTRY_DATE, 10.0)]},
        survivorship_by_ticker={
            "ACME": {
                "type": "acquisition",
                "realized_payoff": 14.0,
                "source_backed": True,
            }
        },
    )

    _run_job(db_session, adapter)

    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "computed"
    assert sig.forward_return == 0.4
    obs = _obs(db_session)
    assert obs.exit_price == 14.0
    assert obs.exit_price_source == "source_backed_realized_payoff"
    assert obs.reason == "corporate_action_realized_payoff"


def test_standard_price_bar_quality_flags_are_not_survivorship_evidence(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter(
        {"ACME": [_bar(ENTRY_DATE, 10.0)]},
        flags_by_ticker={
            "ACME": {
                "terminal_event": {
                    "type": "delisting",
                    "classification": "performance",
                    "source_backed": True,
                }
            }
        },
    )

    _run_job(db_session, adapter)

    sig = db_session.get(SignalRegistry, sid)
    assert adapter.survivorship_calls
    assert sig.forward_return_status == "survivorship_unresolved_review"
    assert sig.forward_return is None
    assert _obs(db_session).reason == "survivorship_unresolved_no_source_event"


def test_invalid_entry_price_is_retryable(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": _bars(entry_open=0.0, exit_open=12.0)})

    _run_job(db_session, adapter)

    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "invalid_entry_price_retry"
    assert sig.outcome_unavailable_reason == "invalid_entry_price"
    assert sig.forward_return is None


def test_invalid_exit_price_is_retryable(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=-1.0)})

    _run_job(db_session, adapter)

    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "invalid_exit_price_retry"
    assert sig.outcome_unavailable_reason == "invalid_exit_price"
    assert sig.forward_return is None


def test_split_adjusted_open_basis_uses_full_open_not_dividend_adjclose(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(ENTRY_DATE, 5.0, close=5.0, split_adjusted_close=5.0, adj_close=1.0),
            _bar(EXIT_DATE, 6.0, close=6.0, split_adjusted_close=6.0, adj_close=99.0),
        ]
    })

    _run_job(db_session, adapter)

    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "computed"
    assert sig.intended_entry_price == 5.0
    assert sig.forward_return == 0.2
    obs = _obs(db_session)
    assert obs.entry_basis_proof == M4_SPLIT_ADJUSTED_OPEN_BASIS_PROOF
    assert obs.exit_basis_proof == M4_SPLIT_ADJUSTED_OPEN_BASIS_PROOF


def test_missing_split_adjusted_basis_fails_closed_without_adjclose_fallback(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(
                ENTRY_DATE,
                10.0,
                split_adjusted_close="missing",
                adj_close=1.0,
            ),
            _bar(EXIT_DATE, 12.0, adj_close=999.0),
        ]
    })

    _run_job(db_session, adapter)

    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "pricing_unavailable_retry"
    assert sig.outcome_unavailable_reason == "split_adjusted_open_basis_unproven"
    assert sig.intended_entry_price is None
    assert sig.forward_return is None


def test_zero_exit_price_computes_minus_one_hundred_percent(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=0.0)})

    _run_job(db_session, adapter)

    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "computed"
    assert sig.forward_return == -1.0
    assert _obs(db_session).exit_price == 0.0


def test_path_telemetry_and_same_day_barrier_ambiguity_persist(db_session):
    _make_signal(db_session)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(ENTRY_DATE, 10.0, high=10.0, low=10.0, close=10.0),
            _bar(date(2026, 6, 1), 10.0, high=21.0, low=9.7, close=15.0),
            _bar(EXIT_DATE, 12.0, high=12.0, low=11.0, close=12.0),
        ]
    })

    _run_job(db_session, adapter)

    obs = _obs(db_session)
    assert obs.max_favorable_excursion == 1.1
    assert round(obs.max_adverse_excursion, 6) == -0.03
    assert obs.mfe_session_date == "2026-06-01"
    assert obs.mae_session_date == "2026-06-01"
    assert obs.max_close_return == 0.5
    assert obs.min_close_return == 0.0
    assert obs.hit_t1_intraday is True
    assert obs.hit_t2_intraday is True
    assert obs.hit_t3_intraday is True
    assert obs.hit_stop_intraday is False
    assert obs.same_day_barrier_ambiguity is False
    event = db_session.query(ForwardReturnObservationEvent).one()
    assert event.same_day_barrier_ambiguity is False


@pytest.mark.parametrize(
    (
        "high",
        "low",
        "expected_t1",
        "expected_t2",
        "expected_t3",
        "expected_stop",
        "expected_ambiguity",
    ),
    [
        (10.49, 10.0, False, False, False, False, False),
        (10.50, 10.0, True, False, False, False, False),
        (11.20, 10.0, True, True, False, False, False),
        (19.99, 10.0, True, True, False, False, False),
        (20.00, 10.0, True, True, True, False, False),
        (10.00, 9.61, False, False, False, False, False),
        (10.00, 9.60, False, False, False, True, False),
        (10.50, 9.60, True, False, False, True, True),
    ],
)
def test_m4_path_telemetry_exit_geometry_thresholds(
    db_session,
    high,
    low,
    expected_t1,
    expected_t2,
    expected_t3,
    expected_stop,
    expected_ambiguity,
):
    assert M4_EXIT_GEOMETRY.hard_stop_return == -0.04
    assert M4_EXIT_GEOMETRY.hard_stop_pct == 0.04

    obs = _path_obs(db_session, high=high, low=low)

    assert obs.hit_t1_intraday is expected_t1
    assert obs.hit_t2_intraday is expected_t2
    assert obs.hit_t3_intraday is expected_t3
    assert obs.hit_stop_intraday is expected_stop
    assert obs.same_day_barrier_ambiguity is expected_ambiguity


def test_retry_then_compute_updates_same_observation(db_session):
    sid = _make_signal(db_session)
    first = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})

    _run_job(db_session, first)
    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "survivorship_unresolved_review"
    assert sig.forward_return_attempts == 1
    first_obs_id = _obs(db_session).forward_return_observation_id

    second = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=11.0)})
    _run_job(db_session, second)

    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "computed"
    assert sig.forward_return_attempts == 2
    assert sig.forward_return == 0.1
    assert db_session.query(ForwardReturnObservation).count() == 1
    assert db_session.query(ForwardReturnObservationEvent).count() == 2
    assert _obs(db_session).forward_return_observation_id == first_obs_id


def test_unresolved_survivorship_does_not_terminalize_to_outcome_unavailable(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})

    _run_job(db_session, adapter, max_attempts=2)
    _run_job(db_session, adapter, max_attempts=2)

    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "survivorship_unresolved_review"
    assert sig.forward_return_attempts == 2
    assert sig.outcome_unavailable_reason == "survivorship_unresolved_no_source_event"
    assert _obs(db_session).status == "survivorship_unresolved_review"
    assert db_session.query(ForwardReturnObservationEvent).count() == 2


def test_deterministic_outcome_hash_excludes_database_ids(db_session):
    _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=12.0)})

    _run_job(db_session, adapter)

    first = _obs(db_session)
    first_hash = first.outcome_hash
    first_input_hash = first.input_hash

    db_session.query(ForwardReturnObservationEvent).delete()
    db_session.delete(first)
    sig = db_session.query(SignalRegistry).one()
    sig.forward_return_status = "pending"
    sig.forward_return = None
    sig.forward_return_attempts = 0
    sig.intended_entry_price = None
    db_session.flush()

    _run_job(db_session, adapter)

    second = _obs(db_session)
    assert second.input_hash == first_input_hash
    assert second.outcome_hash == first_hash


def test_idempotent_rerun_does_not_duplicate_observation_or_signal_summary(db_session):
    _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=12.0)})

    first = _run_job(db_session, adapter)
    second = _run_job(db_session, adapter)

    assert first.metrics["computed"] == 1
    assert second.metrics["total_eligible"] == 0
    assert db_session.query(ForwardReturnObservation).count() == 1
    assert db_session.query(ForwardReturnObservationEvent).count() == 1
    assert db_session.query(SignalRegistry).one().forward_return_status == "computed"


def test_postgres_schema_connect_args_sets_scratch_search_path():
    kwargs = schema_connect_args(
        "postgresql+psycopg://user:pass@example.com/db",
        "scratch_codex_m4_pricefn_audit_test",
    )

    assert kwargs == {
        "connect_args": {
            "options": (
                "-csearch_path=scratch_codex_m4_pricefn_audit_test,public"
            )
        }
    }

    with pytest.raises(ValueError):
        schema_connect_args(
            "postgresql+psycopg://user:pass@example.com/db",
            "bad-schema",
        )
