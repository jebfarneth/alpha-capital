"""Production M4 daily wiring tests."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import List

import pytest

from alpha.assembly.signal_context import (
    SOURCE_CONTEXT_VERSION,
    enrich_m4_signal_context,
)
from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError, stable_hash
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
from alpha.patterns.contracts import PatternInput


def _scan_ts() -> datetime:
    return datetime(2026, 5, 26, 8, 30, tzinfo=timezone.utc)


def _request_ts() -> datetime:
    return datetime(2026, 5, 26, 8, 31, tzinfo=timezone.utc)


def _setup_canonical_universe(
    db_session,
    *,
    trading_date: str = "2026-05-26",
    tickers=("LCUT",),
    scan_asof_timestamp: datetime | None = None,
):
    scan_asof = scan_asof_timestamp or _scan_ts()
    scan = UniverseScan(
        scan_id="m4-prod-scan",
        trading_date=trading_date,
        asof_timestamp=scan_asof,
        raw_count=len(tickers),
        deduped_count=len(tickers),
        included_count=len(tickers),
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
    for ticker in tickers:
        db_session.add(UniverseSnapshot(
            universe_snapshot_id=f"snap-{ticker}",
            scan_id=scan.scan_id,
            ticker=ticker,
            asof_timestamp=scan_asof,
            market_cap=75_000_000,
            price=8.83,
            primary_exchange="NASDAQ",
            security_type="common_stock",
            operating_universe_inclusion=True,
            source_lineage_hash=f"snapshot-lineage-hash-{ticker}",
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
    return _m4_bars(evidence_day, evidence_close=11.0, prior_sessions=prior_sessions)


def _m4_bars(
    evidence_day: date,
    *,
    evidence_close: float,
    prior_sessions: int = 60,
    prior_close: float = 10.0,
) -> List[FmpBar]:
    bars = [
        FmpBar(
            date=day.isoformat(),
            open=9.0,
            high=10.0,
            low=8.5,
            close=prior_close,
            volume=100_000,
            split_adjusted_close=prior_close,
        )
        for day in _prior_weekdays(evidence_day, prior_sessions)
    ]
    bars.append(FmpBar(
        date=evidence_day.isoformat(),
        open=evidence_close,
        high=evidence_close,
        low=evidence_close,
        close=evidence_close,
        volume=200_000,
        split_adjusted_close=evidence_close,
    ))
    return bars


class FakeHistoricalAdapter:
    def __init__(self, bars: List[FmpBar] | dict[str, List[FmpBar]]):
        self.bars = bars
        self.calls = []

    def get_historical_price(self, ticker, from_date=None, to_date=None, asof=None, **kwargs):
        bars = self.bars.get(ticker, []) if isinstance(self.bars, dict) else self.bars
        self.calls.append({
            "ticker": ticker,
            "from_date": from_date,
            "to_date": to_date,
            "asof": asof,
            "kwargs": kwargs,
        })
        return AdapterResponse(
            data=bars,
            lineage=LineageMeta(
                provider="FMP",
                endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                request_timestamp=_request_ts(),
                asof_timestamp=asof,
                raw_payload_hash=stable_hash([
                    {"date": bar.date, "close": bar.close}
                    for bar in bars
                ]),
                source_authority="test",
            ),
        )


def _adapter_response(
    *,
    provider: str,
    endpoint: str,
    data,
    asof: datetime,
    flags=None,
    error: ProviderError | None = None,
):
    return AdapterResponse(
        data=data,
        lineage=LineageMeta(
            provider=provider,
            endpoint=endpoint,
            request_timestamp=_request_ts(),
            asof_timestamp=asof,
            raw_payload_hash=stable_hash({"endpoint": endpoint, "data": data, "error": error}),
            source_authority=provider,
            data_quality_flags=flags,
        ),
        error=error,
    )


class FakePolygonContextAdapter:
    def __init__(
        self,
        *,
        short_interest_rows=None,
        short_volume_rows=None,
        split_rows=None,
        dividend_rows=None,
    ):
        self.calls = []
        self.short_interest_rows = short_interest_rows
        self.short_volume_rows = short_volume_rows
        self.split_rows = split_rows
        self.dividend_rows = dividend_rows

    def get_short_interest(self, **kwargs):
        self.calls.append(("get_short_interest", kwargs))
        rows = self.short_interest_rows
        if rows is None:
            rows = [
                SimpleNamespace(
                    ticker="LCUT",
                    settlement_date="2026-05-15",
                    short_interest=1200,
                    days_to_cover=Decimal("2.5"),
                    avg_daily_volume=480,
                    raw={"published_utc": "2026-05-20T12:00:00Z"},
                )
            ]
        return _adapter_response(
            provider="Polygon",
            endpoint="/stocks/v1/short-interest",
            asof=kwargs["asof"],
            data=rows,
        )

    def get_short_volume(self, **kwargs):
        self.calls.append(("get_short_volume", kwargs))
        rows = self.short_volume_rows if self.short_volume_rows is not None else []
        return _adapter_response(
            provider="Polygon",
            endpoint="/stocks/v1/short-volume",
            asof=kwargs["asof"],
            data=rows,
            flags={"raw_rows": 0, "parsed_rows": 0, "skipped_rows": 0},
        )

    def get_splits(self, **kwargs):
        self.calls.append(("get_splits", kwargs))
        rows = self.split_rows if self.split_rows is not None else []
        return _adapter_response(
            provider="Polygon",
            endpoint="/stocks/v1/splits",
            asof=kwargs["asof"],
            data=rows,
        )

    def get_dividends(self, **kwargs):
        self.calls.append(("get_dividends", kwargs))
        rows = self.dividend_rows
        if rows is None:
            rows = [
                SimpleNamespace(
                    ticker="LCUT",
                    ex_dividend_date="2026-05-20",
                    cash_amount=Decimal("0.05"),
                    dividend_type="CD",
                    distribution_type="regular",
                    frequency=4,
                    declaration_date="2026-05-01",
                )
            ]
        return _adapter_response(
            provider="Polygon",
            endpoint="/stocks/v1/dividends",
            asof=kwargs["asof"],
            data=rows,
        )

    def get_news(self, **kwargs):
        self.calls.append(("get_news", kwargs))
        return _adapter_response(
            provider="Polygon",
            endpoint="/v2/reference/news",
            asof=kwargs["asof"],
            data=None,
            error=ProviderError(
                provider="Polygon",
                endpoint="/v2/reference/news",
                status_code=500,
                error_type="http",
                message="provider unavailable",
                retryable=True,
            ),
        )


class FakeBenzingaContextAdapter:
    def __init__(self):
        self.calls = []

    def get_news(self, **kwargs):
        self.calls.append(("get_news", kwargs))
        return _adapter_response(
            provider="Benzinga",
            endpoint="/api/v2/news",
            asof=kwargs["asof"],
            data=[
                SimpleNamespace(
                    id="n1",
                    title="LCUT update",
                    url="https://example.test/news",
                    published=datetime(2026, 5, 21, 12, tzinfo=timezone.utc),
                    created=None,
                    updated=None,
                    tickers=["LCUT"],
                    channels=["general"],
                    source="Benzinga",
                    body="raw provider article body",
                )
            ],
        )

    def get_wiims(self, **kwargs):
        self.calls.append(("get_wiims", kwargs))
        return _adapter_response(
            provider="Benzinga",
            endpoint="/api/v2/news",
            asof=kwargs["asof"],
            data=[],
        )

    def get_earnings(self, **kwargs):
        self.calls.append(("get_earnings", kwargs))
        return _adapter_response(
            provider="Benzinga",
            endpoint="/api/v2.1/calendar/earnings",
            asof=kwargs["asof"],
            data=[
                SimpleNamespace(
                    ticker="LCUT",
                    date="2026-06-15",
                    updated=datetime(2026, 5, 20, tzinfo=timezone.utc),
                    eps_surprise=Decimal("0.01"),
                    eps_surprise_percent=Decimal("2.5"),
                    revenue_surprise=None,
                    revenue_surprise_percent=None,
                )
            ],
            flags={
                "knowledge_timestamp_warning_rows": 1,
                "knowledge_timestamp_warning_types": {"calendar_updated_future": 1},
            },
        )

    def get_guidance(self, **kwargs):
        self.calls.append(("get_guidance", kwargs))
        return _adapter_response(
            provider="Benzinga",
            endpoint="/api/v2.1/calendar/guidance",
            asof=kwargs["asof"],
            data=[
                SimpleNamespace(
                    ticker="LCUT",
                    date="2026-06-20",
                    updated=None,
                    eps_guidance_est=Decimal("0.10"),
                )
            ],
        )

    def get_ratings(self, **kwargs):
        self.calls.append(("get_ratings", kwargs))
        return _adapter_response(
            provider="Benzinga",
            endpoint="/api/v2.1/calendar/ratings",
            asof=kwargs["asof"],
            data=None,
            error=ProviderError(
                provider="Benzinga",
                endpoint="/api/v2.1/calendar/ratings",
                status_code=200,
                error_type="validation",
                message="validation failed",
                retryable=False,
            ),
        )

    def get_offerings(self, **kwargs):
        self.calls.append(("get_offerings", kwargs))
        return _adapter_response(
            provider="Benzinga",
            endpoint="/api/v2.1/calendar/offerings",
            asof=kwargs["asof"],
            data=[],
        )

    def get_dividends(self, **kwargs):
        self.calls.append(("get_dividends", kwargs))
        return _adapter_response(
            provider="Benzinga",
            endpoint="/api/v2.1/calendar/dividends",
            asof=kwargs["asof"],
            data=[],
        )

    def get_insider_filings(self, **kwargs):
        self.calls.append(("get_insider_filings", kwargs))
        return _adapter_response(
            provider="Benzinga",
            endpoint="/api/v1/sec/insider_transactions/filings",
            asof=kwargs["asof"],
            data=[
                SimpleNamespace(
                    id="f1",
                    accession_number="0001",
                    company_symbol="LCUT",
                    filing_date=datetime(2026, 5, 20, 21, tzinfo=timezone.utc),
                    updated=None,
                )
            ],
        )

    def get_insider_transactions(self, **kwargs):
        self.calls.append(("get_insider_transactions", kwargs))
        return _adapter_response(
            provider="Benzinga",
            endpoint="/api/v1/sec/insider_transactions/transactions",
            asof=kwargs["asof"],
            data=[
                SimpleNamespace(
                    transaction_id="p1",
                    company_symbol="LCUT",
                    filing_date=datetime(2026, 5, 20, 21, tzinfo=timezone.utc),
                    updated=None,
                    transaction_code="P",
                    acquired_or_disposed="A",
                    shares=Decimal("100"),
                    price_per_share=Decimal("2.00"),
                ),
                SimpleNamespace(
                    transaction_id="s1",
                    company_symbol="LCUT",
                    filing_date=datetime(2026, 5, 20, 21, tzinfo=timezone.utc),
                    updated=None,
                    transaction_code="S",
                    acquired_or_disposed="D",
                    shares=Decimal("25"),
                    price_per_share=Decimal("3.00"),
                ),
                SimpleNamespace(
                    transaction_id="f1",
                    company_symbol="LCUT",
                    filing_date=datetime(2026, 5, 20, 21, tzinfo=timezone.utc),
                    updated=None,
                    transaction_code="f",
                    acquired_or_disposed="D",
                    shares=Decimal("10"),
                    price_per_share=Decimal("1.00"),
                ),
            ],
        )

    def get_mergers_acquisitions(self, **kwargs):
        self.calls.append(("get_mergers_acquisitions", kwargs))
        return _adapter_response(
            provider="Benzinga",
            endpoint="/api/v2.1/calendar/ma",
            asof=kwargs["asof"],
            data=[],
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
    assert json.loads(feature.feature_json)["signal_context"]["schema_version"] == SOURCE_CONTEXT_VERSION


def test_m4_production_job_uses_early_close_asof_for_evidence_day(db_session, monkeypatch):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            frozen = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)
            if tz is None:
                return frozen.replace(tzinfo=None)
            return frozen.astimezone(tz)

    monkeypatch.setattr("alpha.patterns.guards.datetime", FrozenDatetime)
    evidence_day = date(2026, 11, 27)
    scan_asof = datetime(2026, 11, 27, 13, 0, tzinfo=timezone.utc)
    early_close_asof = us_equity_session_close_timestamp(evidence_day)
    assert scan_asof < early_close_asof
    _setup_canonical_universe(
        db_session,
        trading_date="2026-11-27",
        scan_asof_timestamp=scan_asof,
    )
    adapter = FakeHistoricalAdapter(_m4_breakout_bars(evidence_day))
    job = M4DailyAssemblyJob(
        db_session,
        adapter=adapter,
        run_timestamp=datetime(2026, 11, 27, 20, 0, tzinfo=timezone.utc),
    )

    result = run_job(db_session, job, params={})

    assert result.ok
    assert result.metrics["decision_date"] == "2026-11-27"
    assert result.metrics["evidence_session_date"] == "2026-11-27"
    assert adapter.calls[0]["asof"] == early_close_asof
    assert result.metrics["fetch_asof_timestamp"] == early_close_asof.isoformat()
    assert result.metrics["orchestration"]["total_signals_persisted"] == 1
    diag = result.metrics["orchestration"]["detector_diagnostics"][0]
    assert diag["lookahead_failure_count"] == 0

    feature = db_session.query(FeatureSnapshot).filter_by(ticker="LCUT").one()
    signal = db_session.query(SignalRegistry).filter_by(ticker="LCUT").one()
    assert _as_utc(feature.asof_timestamp) == early_close_asof
    assert _as_utc(signal.signal_timestamp) == early_close_asof
    assert _as_utc(feature.asof_timestamp) != scan_asof
    assert json.loads(feature.feature_json)["signal_context"]["asof_timestamp"] == (
        early_close_asof.isoformat()
    )


def test_m4_daily_signal_context_persists_attempts_lineage_and_pit_context(db_session):
    _setup_canonical_universe(db_session)
    evidence_day = date(2026, 5, 22)
    adapter = FakeHistoricalAdapter(_m4_breakout_bars(evidence_day))
    polygon = FakePolygonContextAdapter()
    benzinga = FakeBenzingaContextAdapter()
    job = M4DailyAssemblyJob(
        db_session,
        adapter=adapter,
        polygon_adapter=polygon,
        benzinga_adapter=benzinga,
        run_timestamp=datetime(2026, 5, 26, 8, 0, tzinfo=timezone.utc),
    )

    result = run_job(db_session, job, params={})

    assert result.ok
    assert result.metrics["signal_context"]["context_attached_count"] == 1
    assert result.metrics["signal_context"]["provider_error_count"] == 1
    feature = db_session.query(FeatureSnapshot).filter_by(ticker="LCUT").one()
    feature_json = json.loads(feature.feature_json)
    assert "raw provider article body" not in feature.feature_json
    assert "raw_payload" not in feature.feature_json
    context = feature_json["signal_context"]
    assert context["polygon_short_interest"]["short_interest"] == 1200
    assert context["polygon_short_volume"]["source_attempts"][0]["status"] == "no_data"
    assert context["polygon_news"]["source_attempts"][0]["status"] == "provider_error"
    assert context["benzinga_calendar"]["source_attempts"][2]["status"] == "validation_error"
    assert context["benzinga_calendar"]["guidance"]["event_dates"][0]["date"] == "2026-06-20"
    guidance_attempt = [
        item for item in context["benzinga_calendar"]["source_attempts"]
        if item["source"] == "Benzinga guidance"
    ][0]
    assert guidance_attempt["status"] == "pit_excluded"
    earnings_attempt = [
        item for item in context["benzinga_calendar"]["source_attempts"]
        if item["source"] == "Benzinga earnings"
    ][0]
    assert earnings_attempt["warnings"]["knowledge_timestamp_warning_rows"] == 1
    insider = context["benzinga_insider"]
    assert insider["routine_disposition_count"] == 1
    assert insider["discretionary_buy_count"] == 1
    assert insider["discretionary_sell_count"] == 1
    assert insider["net_discretionary_shares"] == "75"
    assert context["benzinga_ma"]["review_context_only"] is True

    lineage_ids = json.loads(feature.data_lineage_ids)
    assert len(lineage_ids) > 1
    context_lineage_ids = [
        attempt.get("lineage_id")
        for category in context.values()
        if isinstance(category, dict)
        for attempt in category.get("source_attempts", [])
    ]
    assert any(lineage_id in lineage_ids for lineage_id in context_lineage_ids if lineage_id)


def test_m4_daily_signal_context_prefilter_enriches_breakout_superset_only(db_session):
    _setup_canonical_universe(db_session, tickers=("FIRE", "NEAR", "FAR"))
    evidence_day = date(2026, 5, 22)
    adapter = FakeHistoricalAdapter({
        "FIRE": _m4_bars(evidence_day, evidence_close=10.05),
        "NEAR": _m4_bars(evidence_day, evidence_close=9.85),
        "FAR": _m4_bars(evidence_day, evidence_close=9.79),
    })
    polygon = FakePolygonContextAdapter()
    benzinga = FakeBenzingaContextAdapter()
    job = M4DailyAssemblyJob(
        db_session,
        adapter=adapter,
        polygon_adapter=polygon,
        benzinga_adapter=benzinga,
        run_timestamp=datetime(2026, 5, 26, 8, 0, tzinfo=timezone.utc),
        signal_context_breakout_buffer=0.02,
    )

    result = run_job(db_session, job, params={})

    assert result.ok
    metrics = result.metrics["signal_context"]
    assert metrics["context_prefilter_input_count"] == 3
    assert metrics["context_prefilter_candidate_count"] == 2
    assert metrics["context_prefilter_skipped_count"] == 1
    assert metrics["context_attached_count"] == 2
    assert metrics["context_enriched_count"] == 2
    assert result.metrics["orchestration"]["total_signals_persisted"] == 1

    polygon_tickers = {kwargs["ticker"] for _, kwargs in polygon.calls}
    benzinga_tickers = {
        kwargs.get("tickers") or kwargs.get("ticker")
        for _, kwargs in benzinga.calls
    }
    assert polygon_tickers == {"FIRE", "NEAR"}
    assert benzinga_tickers == {"FIRE", "NEAR"}

    features = {
        feature.ticker: json.loads(feature.feature_json)
        for feature in db_session.query(FeatureSnapshot).all()
    }
    assert "signal_context" in features["FIRE"]
    assert "signal_context" in features["NEAR"]
    assert "signal_context" not in features["FAR"]


def test_m4_daily_signal_context_prefilter_buffer_is_configurable(db_session):
    _setup_canonical_universe(db_session, tickers=("FIRE", "NEAR"))
    evidence_day = date(2026, 5, 22)
    adapter = FakeHistoricalAdapter({
        "FIRE": _m4_bars(evidence_day, evidence_close=10.05),
        "NEAR": _m4_bars(evidence_day, evidence_close=9.85),
    })
    polygon = FakePolygonContextAdapter()
    job = M4DailyAssemblyJob(
        db_session,
        adapter=adapter,
        polygon_adapter=polygon,
        benzinga_adapter=FakeBenzingaContextAdapter(),
        run_timestamp=datetime(2026, 5, 26, 8, 0, tzinfo=timezone.utc),
        signal_context_breakout_buffer=0.01,
    )

    result = run_job(db_session, job, params={})

    assert result.ok
    metrics = result.metrics["signal_context"]
    assert metrics["context_prefilter_breakout_buffer"] == 0.01
    assert metrics["context_prefilter_threshold_multiplier"] == 0.99
    assert metrics["context_prefilter_candidate_count"] == 1
    assert {kwargs["ticker"] for _, kwargs in polygon.calls} == {"FIRE"}


@pytest.mark.parametrize("invalid_buffer", [-0.01, 1.0, float("nan"), "bad"])
def test_m4_daily_invalid_signal_context_buffer_fails_before_fetch(db_session, invalid_buffer):
    _setup_canonical_universe(db_session)
    adapter = FakeHistoricalAdapter(_m4_breakout_bars(date(2026, 5, 22)))
    job = M4DailyAssemblyJob(
        db_session,
        adapter=adapter,
        polygon_adapter=FakePolygonContextAdapter(),
        benzinga_adapter=FakeBenzingaContextAdapter(),
        run_timestamp=datetime(2026, 5, 26, 8, 0, tzinfo=timezone.utc),
        signal_context_breakout_buffer=invalid_buffer,
    )

    result = run_job(db_session, job, params={})

    assert not result.ok
    assert result.errors[0]["stage"] == "params"
    assert "signal_context_breakout_buffer" in result.errors[0]["message"]
    assert adapter.calls == []
    assert db_session.query(FeatureSnapshot).count() == 0
    assert db_session.query(SignalRegistry).count() == 0


def test_m4_daily_reuses_persisted_signal_context_without_refetch(db_session):
    _setup_canonical_universe(db_session)
    evidence_day = date(2026, 5, 22)
    adapter = FakeHistoricalAdapter(_m4_breakout_bars(evidence_day))
    first_polygon = FakePolygonContextAdapter()
    first_benzinga = FakeBenzingaContextAdapter()
    first_job = M4DailyAssemblyJob(
        db_session,
        adapter=adapter,
        polygon_adapter=first_polygon,
        benzinga_adapter=first_benzinga,
        run_timestamp=datetime(2026, 5, 26, 8, 0, tzinfo=timezone.utc),
    )

    first_result = run_job(db_session, first_job, params={})
    feature = db_session.query(FeatureSnapshot).filter_by(ticker="LCUT").one()
    frozen_context = json.loads(feature.feature_json)["signal_context"]
    frozen_feature_json = feature.feature_json

    second_polygon = FakePolygonContextAdapter(short_interest_rows=[
        SimpleNamespace(
            ticker="LCUT",
            settlement_date="2026-05-01",
            short_interest=999999,
            raw={"published_utc": "2026-05-02T12:00:00Z"},
        )
    ])
    second_benzinga = FakeBenzingaContextAdapter()
    second_job = M4DailyAssemblyJob(
        db_session,
        adapter=adapter,
        polygon_adapter=second_polygon,
        benzinga_adapter=second_benzinga,
        run_timestamp=datetime(2026, 5, 26, 8, 0, tzinfo=timezone.utc),
    )

    second_result = run_job(db_session, second_job, params={})

    assert first_result.ok
    assert second_result.ok
    assert second_polygon.calls == []
    assert second_benzinga.calls == []
    metrics = second_result.metrics["signal_context"]
    assert metrics["context_reused_from_persistence_count"] == 1
    assert metrics["context_reused_in_memory_count"] == 1
    assert metrics["context_enriched_count"] == 0
    assert db_session.query(FeatureSnapshot).filter_by(ticker="LCUT").count() == 1
    feature = db_session.query(FeatureSnapshot).filter_by(ticker="LCUT").one()
    assert feature.feature_json == frozen_feature_json
    assert json.loads(feature.feature_json)["signal_context"] == frozen_context


def test_m4_daily_reenriches_when_persisted_signal_context_asof_mismatches(db_session):
    _setup_canonical_universe(db_session)
    evidence_day = date(2026, 5, 22)
    adapter = FakeHistoricalAdapter(_m4_breakout_bars(evidence_day))
    first_job = M4DailyAssemblyJob(
        db_session,
        adapter=adapter,
        polygon_adapter=FakePolygonContextAdapter(),
        benzinga_adapter=FakeBenzingaContextAdapter(),
        run_timestamp=datetime(2026, 5, 26, 8, 0, tzinfo=timezone.utc),
    )
    first_result = run_job(db_session, first_job, params={})
    assert first_result.ok

    feature = db_session.query(FeatureSnapshot).filter_by(ticker="LCUT").one()
    feature_payload = json.loads(feature.feature_json)
    feature_payload["signal_context"]["asof_timestamp"] = "2026-05-21T20:00:00+00:00"
    feature.feature_json = json.dumps(feature_payload, sort_keys=True)
    db_session.flush()

    second_polygon = FakePolygonContextAdapter()
    second_job = M4DailyAssemblyJob(
        db_session,
        adapter=adapter,
        polygon_adapter=second_polygon,
        benzinga_adapter=FakeBenzingaContextAdapter(),
        run_timestamp=datetime(2026, 5, 26, 8, 0, tzinfo=timezone.utc),
    )
    second_result = run_job(db_session, second_job, params={})

    assert second_result.ok
    assert second_polygon.calls
    metrics = second_result.metrics["signal_context"]
    assert metrics["context_reused_from_persistence_count"] == 0
    assert metrics["context_persistence_mismatch_count"] == 1
    assert metrics["context_persistence_mismatch_reasons"] == {"asof_mismatch": 1}
    assert metrics["context_enriched_count"] == 1


def test_polygon_short_interest_availability_lag_excludes_unpublished_recent_row(db_session):
    cutoff = us_equity_session_close_timestamp(date(2026, 5, 22))
    inp = PatternInput(
        ticker="LCUT",
        asof_timestamp=cutoff,
        market_data={},
        lineage_hashes=["base-hash"],
    )
    polygon = FakePolygonContextAdapter(short_interest_rows=[
        SimpleNamespace(
            ticker="LCUT",
            settlement_date="2026-05-01",
            short_interest=100,
            days_to_cover=Decimal("1.0"),
            avg_daily_volume=1000,
        ),
        SimpleNamespace(
            ticker="LCUT",
            settlement_date="2026-05-22",
            short_interest=999,
            days_to_cover=Decimal("9.0"),
            avg_daily_volume=1000,
        ),
    ])

    metrics = enrich_m4_signal_context(
        [inp],
        session=db_session,
        polygon_adapter=polygon,
        cutoff_timestamp=cutoff,
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
    )

    context = inp.market_data["signal_context"]["polygon_short_interest"]
    assert metrics["context_attached_count"] == 1
    assert context["status"] == "matched"
    assert context["settlement_date"] == "2026-05-01"
    assert context["short_interest"] == 100
    assert context["event_dates"] == [
        {"ticker": "LCUT", "settlement_date": "2026-05-01"},
        {"ticker": "LCUT", "settlement_date": "2026-05-22"},
    ]
    attempt = context["source_attempts"][0]
    assert attempt["eligible_row_count"] == 1
    assert attempt["warnings"]["short_interest_availability_lag_applied"] is True
    assert attempt["warnings"]["availability_lag_excluded_rows"] == 1


def test_polygon_short_volume_availability_lag_excludes_unpublished_recent_row(db_session):
    cutoff = us_equity_session_close_timestamp(date(2026, 5, 22))
    inp = PatternInput(
        ticker="LCUT",
        asof_timestamp=cutoff,
        market_data={},
        lineage_hashes=["base-hash"],
    )
    polygon = FakePolygonContextAdapter(
        short_interest_rows=[],
        short_volume_rows=[
            SimpleNamespace(
                ticker="LCUT",
                date="2026-05-21",
                short_volume=Decimal("100"),
                total_volume=Decimal("200"),
                short_volume_ratio=Decimal("50"),
            ),
            SimpleNamespace(
                ticker="LCUT",
                date="2026-05-22",
                short_volume=Decimal("999"),
                total_volume=Decimal("1000"),
                short_volume_ratio=Decimal("99.9"),
            ),
        ],
    )

    enrich_m4_signal_context(
        [inp],
        session=db_session,
        polygon_adapter=polygon,
        cutoff_timestamp=cutoff,
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
    )

    context = inp.market_data["signal_context"]["polygon_short_volume"]
    assert context["status"] == "matched"
    assert context["date"] == "2026-05-21"
    assert context["short_volume"] == "100"
    assert context["event_dates"] == [
        {"ticker": "LCUT", "date": "2026-05-21"},
        {"ticker": "LCUT", "date": "2026-05-22"},
    ]
    attempt = context["source_attempts"][0]
    assert attempt["eligible_row_count"] == 1
    assert attempt["warnings"]["short_volume_availability_lag_applied"] is True
    assert attempt["warnings"]["availability_lag_excluded_rows"] == 1


def test_polygon_short_row_event_date_alone_is_context_not_eligibility(db_session):
    cutoff = us_equity_session_close_timestamp(date(2026, 5, 22))
    inp = PatternInput(
        ticker="LCUT",
        asof_timestamp=cutoff,
        market_data={},
        lineage_hashes=["base-hash"],
    )
    polygon = FakePolygonContextAdapter(short_interest_rows=[
        SimpleNamespace(
            ticker="LCUT",
            settlement_date="2026-05-22",
            short_interest=999,
            days_to_cover=Decimal("9.0"),
            avg_daily_volume=1000,
        ),
    ])

    enrich_m4_signal_context(
        [inp],
        session=db_session,
        polygon_adapter=polygon,
        cutoff_timestamp=cutoff,
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
    )

    context = inp.market_data["signal_context"]["polygon_short_interest"]
    assert context["status"] == "pit_excluded"
    assert "short_interest" not in context
    assert context["event_dates"] == [
        {"ticker": "LCUT", "settlement_date": "2026-05-22"}
    ]
    attempt = context["source_attempts"][0]
    assert attempt["eligible_row_count"] == 0
    assert attempt["warnings"]["short_interest_availability_lag_applied"] is True
    assert attempt["warnings"]["availability_lag_excluded_rows"] == 1


def test_polygon_split_availability_excludes_future_announcement_and_selects_available(db_session):
    cutoff = us_equity_session_close_timestamp(date(2026, 5, 22))
    inp = PatternInput(
        ticker="LCUT",
        asof_timestamp=cutoff,
        market_data={},
        lineage_hashes=["base-hash"],
    )
    polygon = FakePolygonContextAdapter(
        short_interest_rows=[],
        split_rows=[
            SimpleNamespace(
                ticker="LCUT",
                execution_date="2026-05-01",
                split_from=Decimal("1"),
                split_to=Decimal("2"),
                raw={"announcement_date": "2026-04-25"},
            ),
            SimpleNamespace(
                ticker="LCUT",
                execution_date="2026-05-22",
                split_from=Decimal("1"),
                split_to=Decimal("4"),
                raw={"announcement_date": "2026-05-26"},
            ),
        ],
        dividend_rows=[],
    )

    enrich_m4_signal_context(
        [inp],
        session=db_session,
        polygon_adapter=polygon,
        cutoff_timestamp=cutoff,
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
    )

    context = inp.market_data["signal_context"]["polygon_corporate_actions"]
    assert context["status"] == "matched"
    assert context["split_count_window"] == 1
    assert context["latest_split"]["execution_date"] == "2026-05-01"
    assert context["split_event_dates"] == [
        {"ticker": "LCUT", "execution_date": "2026-05-01"},
        {"ticker": "LCUT", "execution_date": "2026-05-22"},
    ]
    attempt = [
        item for item in context["source_attempts"]
        if item["source"] == "Polygon splits"
    ][0]
    assert attempt["status"] == "matched"
    assert attempt["eligible_row_count"] == 1
    assert attempt["pit_excluded_row_count"] == 1
    assert attempt["warnings"]["availability_timestamp_field"] == "announcement_date"
    assert attempt["warnings"]["availability_timestamp_future_rows"] == 1


def test_polygon_dividend_availability_excludes_future_declaration_and_selects_available(db_session):
    cutoff = us_equity_session_close_timestamp(date(2026, 5, 22))
    inp = PatternInput(
        ticker="LCUT",
        asof_timestamp=cutoff,
        market_data={},
        lineage_hashes=["base-hash"],
    )
    polygon = FakePolygonContextAdapter(
        short_interest_rows=[],
        split_rows=[],
        dividend_rows=[
            SimpleNamespace(
                ticker="LCUT",
                ex_dividend_date="2026-05-20",
                cash_amount=Decimal("0.05"),
                dividend_type="CD",
                declaration_date="2026-05-01",
            ),
            SimpleNamespace(
                ticker="LCUT",
                ex_dividend_date="2026-05-22",
                cash_amount=Decimal("0.10"),
                dividend_type="CD",
                declaration_date="2026-05-26",
            ),
        ],
    )

    enrich_m4_signal_context(
        [inp],
        session=db_session,
        polygon_adapter=polygon,
        cutoff_timestamp=cutoff,
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
    )

    context = inp.market_data["signal_context"]["polygon_corporate_actions"]
    assert context["status"] == "matched"
    assert context["dividend_count_window"] == 1
    assert context["last_dividend"]["ex_dividend_date"] == "2026-05-20"
    assert context["dividend_proximity_days"] == -2
    assert context["dividend_event_dates"] == [
        {
            "ticker": "LCUT",
            "ex_dividend_date": "2026-05-20",
            "declaration_date": "2026-05-01",
        },
        {
            "ticker": "LCUT",
            "ex_dividend_date": "2026-05-22",
            "declaration_date": "2026-05-26",
        },
    ]
    attempt = [
        item for item in context["source_attempts"]
        if item["source"] == "Polygon dividends"
    ][0]
    assert attempt["status"] == "matched"
    assert attempt["eligible_row_count"] == 1
    assert attempt["pit_excluded_row_count"] == 1
    assert attempt["warnings"]["availability_timestamp_field"] == "declaration_date"
    assert attempt["warnings"]["availability_timestamp_future_rows"] == 1


def test_polygon_corporate_action_event_date_alone_is_context_not_eligibility(db_session):
    cutoff = us_equity_session_close_timestamp(date(2026, 5, 22))
    inp = PatternInput(
        ticker="LCUT",
        asof_timestamp=cutoff,
        market_data={},
        lineage_hashes=["base-hash"],
    )
    polygon = FakePolygonContextAdapter(
        short_interest_rows=[],
        split_rows=[
            SimpleNamespace(
                ticker="LCUT",
                execution_date="2026-05-22",
                split_from=Decimal("1"),
                split_to=Decimal("4"),
            ),
        ],
        dividend_rows=[
            SimpleNamespace(
                ticker="LCUT",
                ex_dividend_date="2026-05-22",
                cash_amount=Decimal("0.10"),
            ),
        ],
    )

    enrich_m4_signal_context(
        [inp],
        session=db_session,
        polygon_adapter=polygon,
        cutoff_timestamp=cutoff,
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
    )

    context = inp.market_data["signal_context"]["polygon_corporate_actions"]
    assert context["status"] == "pit_excluded"
    assert context["split_count_window"] == 0
    assert context["dividend_count_window"] == 0
    assert "latest_split" not in context
    assert "last_dividend" not in context
    assert context["split_event_dates"] == [
        {"ticker": "LCUT", "execution_date": "2026-05-22"}
    ]
    assert context["dividend_event_dates"] == [
        {"ticker": "LCUT", "ex_dividend_date": "2026-05-22"}
    ]
    split_attempt = [
        item for item in context["source_attempts"]
        if item["source"] == "Polygon splits"
    ][0]
    dividend_attempt = [
        item for item in context["source_attempts"]
        if item["source"] == "Polygon dividends"
    ][0]
    assert split_attempt["status"] == "pit_excluded"
    assert split_attempt["eligible_row_count"] == 0
    assert split_attempt["pit_excluded_row_count"] == 1
    assert split_attempt["warnings"]["split_availability_lag_applied"] is True
    assert split_attempt["warnings"]["availability_lag_applied"] is True
    assert split_attempt["warnings"]["availability_lag_excluded_rows"] == 1
    assert dividend_attempt["status"] == "pit_excluded"
    assert dividend_attempt["eligible_row_count"] == 0
    assert dividend_attempt["pit_excluded_row_count"] == 1
    assert dividend_attempt["warnings"]["dividend_availability_lag_applied"] is True
    assert dividend_attempt["warnings"]["availability_lag_applied"] is True
    assert dividend_attempt["warnings"]["availability_lag_excluded_rows"] == 1


def test_signal_context_reuses_matching_frozen_context_without_refetch(db_session):
    cutoff = us_equity_session_close_timestamp(date(2026, 5, 22))
    existing = {
        "schema_version": SOURCE_CONTEXT_VERSION,
        "asof_timestamp": cutoff.isoformat(),
        "polygon_short_interest": {
            "status": "matched",
            "source_attempts": [{"source": "Polygon short interest", "status": "matched"}],
        },
    }
    inp = PatternInput(
        ticker="LCUT",
        asof_timestamp=cutoff,
        market_data={"signal_context": existing},
        lineage_hashes=["base-hash"],
    )
    polygon = FakePolygonContextAdapter()

    metrics = enrich_m4_signal_context(
        [inp],
        session=db_session,
        polygon_adapter=polygon,
        cutoff_timestamp=cutoff,
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
    )

    assert inp.market_data["signal_context"] is existing
    assert polygon.calls == []
    assert metrics["context_reused_count"] == 1
    assert metrics["context_reused_in_memory_count"] == 1
    assert metrics["context_attached_count"] == 0


def test_signal_context_mismatched_asof_reenriches(db_session):
    cutoff = us_equity_session_close_timestamp(date(2026, 5, 22))
    inp = PatternInput(
        ticker="LCUT",
        asof_timestamp=cutoff,
        market_data={
            "signal_context": {
                "schema_version": SOURCE_CONTEXT_VERSION,
                "asof_timestamp": "2026-05-21T20:00:00+00:00",
                "sentinel": "old",
            }
        },
        lineage_hashes=["base-hash"],
    )
    polygon = FakePolygonContextAdapter()

    metrics = enrich_m4_signal_context(
        [inp],
        session=db_session,
        polygon_adapter=polygon,
        cutoff_timestamp=cutoff,
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
    )

    assert polygon.calls
    assert metrics["context_reused_count"] == 0
    assert metrics["context_attached_count"] == 1
    assert "sentinel" not in inp.market_data["signal_context"]
    assert inp.market_data["signal_context"]["asof_timestamp"] == cutoff.isoformat()


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
