"""Production M4 daily wiring tests."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import List

from alpha.data.contracts import AdapterResponse, LineageMeta, stable_hash
from alpha.data.fmp import FmpBar, HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.db.models import (
    CanonicalUniverseScan,
    DataLineage,
    FeatureSnapshot,
    SignalRegistry,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.jobs.m4_daily import M4DailyAssemblyJob
from alpha.jobs.runner import run_job
from alpha.market_calendar import us_equity_session_close_timestamp


def _scan_ts() -> datetime:
    return datetime(2026, 5, 26, 8, 30, tzinfo=timezone.utc)


def _request_ts() -> datetime:
    return datetime(2026, 5, 26, 8, 31, tzinfo=timezone.utc)


def _setup_canonical_universe(db_session, *, trading_date: str = "2026-05-26"):
    scan = UniverseScan(
        scan_id="m4-prod-scan",
        trading_date=trading_date,
        asof_timestamp=_scan_ts(),
        raw_count=1,
        deduped_count=1,
        included_count=1,
        excluded_count=0,
        run_status="finished",
        source_lineage_hash="scan-lineage-hash",
    )
    db_session.add(scan)
    db_session.flush()
    db_session.add(CanonicalUniverseScan(
        trading_date=trading_date,
        scan_id=scan.scan_id,
        selection_reason="test",
    ))
    db_session.add(UniverseSnapshot(
        universe_snapshot_id="snap-LCUT",
        scan_id=scan.scan_id,
        ticker="LCUT",
        asof_timestamp=_scan_ts(),
        market_cap=75_000_000,
        price=8.83,
        primary_exchange="NASDAQ",
        security_type="common_stock",
        operating_universe_inclusion=True,
        source_lineage_hash="snapshot-lineage-hash",
    ))
    db_session.flush()


def _prior_weekdays(end_day: date, n: int) -> List[date]:
    days: List[date] = []
    cursor = end_day - timedelta(days=1)
    while len(days) < n:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return list(reversed(days))


def _m4_breakout_bars(evidence_day: date, *, prior_sessions: int = 60) -> List[FmpBar]:
    bars = [
        FmpBar(
            date=day.isoformat(),
            open=9.0,
            high=10.0,
            low=8.5,
            close=10.0,
            volume=100_000,
            split_adjusted_close=10.0,
        )
        for day in _prior_weekdays(evidence_day, prior_sessions)
    ]
    bars.append(FmpBar(
        date=evidence_day.isoformat(),
        open=10.2,
        high=11.1,
        low=10.0,
        close=11.0,
        volume=200_000,
        split_adjusted_close=11.0,
    ))
    return bars


class FakeHistoricalAdapter:
    def __init__(self, bars: List[FmpBar]):
        self.bars = bars
        self.calls = []

    def get_historical_price(self, ticker, from_date=None, to_date=None, asof=None, **kwargs):
        self.calls.append({
            "ticker": ticker,
            "from_date": from_date,
            "to_date": to_date,
            "asof": asof,
            "kwargs": kwargs,
        })
        return AdapterResponse(
            data=self.bars,
            lineage=LineageMeta(
                provider="FMP",
                endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                request_timestamp=_request_ts(),
                asof_timestamp=asof,
                raw_payload_hash=stable_hash([
                    {"date": bar.date, "close": bar.close}
                    for bar in self.bars
                ]),
                source_authority="test",
            ),
        )


def test_m4_production_job_caps_fetch_at_evidence_session_and_persists(db_session):
    _setup_canonical_universe(db_session)
    evidence_day = date(2026, 5, 22)
    adapter = FakeHistoricalAdapter(_m4_breakout_bars(evidence_day))
    job = M4DailyAssemblyJob(
        db_session,
        adapter=adapter,
        run_timestamp=datetime(2026, 5, 26, 8, 0, tzinfo=timezone.utc),
    )

    result = run_job(db_session, job, params={})

    assert result.ok
    assert result.metrics["decision_date"] == "2026-05-26"
    assert result.metrics["evidence_session_date"] == "2026-05-22"
    assert result.metrics["fetch_to_date"] == "2026-05-22"
    assert result.metrics["assembly"]["assembled_count"] == 1
    feature = db_session.query(FeatureSnapshot).filter_by(ticker="LCUT").one()
    assert result.metrics["orchestration"]["total_signals_persisted"] == 1, (
        feature.feature_json
    )

    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    assert call["ticker"] == "LCUT"
    assert call["to_date"] == evidence_day
    assert call["from_date"] < evidence_day
    assert call["asof"] == us_equity_session_close_timestamp(evidence_day)
    assert call["kwargs"]["adjusted"] is False
    assert call["kwargs"]["require_split_adjusted_close"] is True

    lineage_ids = json.loads(feature.data_lineage_ids)
    assert len(lineage_ids) == 1
    lineage = db_session.get(DataLineage, lineage_ids[0])
    assert lineage.endpoint == HISTORICAL_PRICE_FULL_ENDPOINT
    assert _as_utc(lineage.asof_timestamp) == us_equity_session_close_timestamp(evidence_day)
    assert _as_utc(lineage.request_timestamp) == _request_ts()
    raw_payload = json.loads(lineage.raw_payload_json)
    assert raw_payload["ticker"] == "LCUT"
    assert raw_payload["request"] == {
        "symbol": "LCUT",
        "from": call["from_date"].isoformat(),
        "to": "2026-05-22",
        "endpoint": HISTORICAL_PRICE_FULL_ENDPOINT,
    }
    assert max(bar["date"] for bar in raw_payload["bars"]) == "2026-05-22"

    signal = db_session.query(SignalRegistry).filter_by(ticker="LCUT").one()
    assert signal.pattern_id == "M4"
    assert signal.feature_snapshot_id == feature.feature_snapshot_id
    assert signal.trading_date == "2026-05-26"
    assert signal.next_execution_session == "2026-05-26"
    assert json.loads(feature.feature_json)["next_execution_session"] == "2026-05-26"


def test_m4_production_job_refuses_trading_date_that_bypasses_resolver(db_session):
    _setup_canonical_universe(db_session)
    adapter = FakeHistoricalAdapter(_m4_breakout_bars(date(2026, 5, 22)))
    job = M4DailyAssemblyJob(
        db_session,
        adapter=adapter,
        run_timestamp=datetime(2026, 5, 26, 8, 0, tzinfo=timezone.utc),
    )

    result = run_job(db_session, job, params={"trading_date": "2026-05-22"})

    assert not result.ok
    assert result.errors[0]["stage"] == "params"
    assert adapter.calls == []
    assert db_session.query(FeatureSnapshot).count() == 0
    assert db_session.query(SignalRegistry).count() == 0


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
