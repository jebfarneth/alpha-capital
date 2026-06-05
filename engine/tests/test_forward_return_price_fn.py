"""Production M4 price_fn tests."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import pytest
from sqlalchemy import text

from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError, stable_hash
from alpha.data.edgar import SecCompanyTicker
from alpha.data.fmp import (
    DELISTED_COMPANIES_ENDPOINT,
    FmpBar,
    FmpDelistedCompany,
    HISTORICAL_PRICE_FULL_ENDPOINT,
)
from alpha.data.nasdaq import (
    ADDS_DELETES,
    HALT_RSS,
    NASDAQ_LISTED,
    OTHER_LISTED,
    NasdaqListingStatus,
    NasdaqListingStatusResult,
    NasdaqTraderListingAdapter,
)
from alpha.db.engine import schema_connect_args
from alpha.db.models import (
    DataLineage,
    ForwardReturnObservation,
    ForwardReturnObservationEvent,
    ForwardReturnPathRow,
    NasdaqListingSnapshot,
    NasdaqListingSnapshotRow,
    SignalRegistry,
)
from alpha.evidence.writer import record_feature_snapshot, record_signal
from alpha.jobs import run_forward_return
from alpha.jobs.contracts import JobResult
from alpha.jobs.forward_return import (
    LEGACY_NEXT_EXECUTION_SESSION_FALLBACK_REASON,
    M4_EXIT_GEOMETRY,
    M4_PRICE_SOURCE,
    M4_SPLIT_ADJUSTED_OPEN_BASIS_PROOF,
    NASDAQ_LISTING_SUPPRESSION_REASON,
    ForwardReturnJob,
    _apply_listing_authority_to_pending_edgar_reviews,
    _canonical_cik10,
    _security_identity_from_payload,
    current_forward_path_rows,
    m4_entry_exit_plan,
)
from alpha.jobs.run_forward_return import _live_timestamp_error
from alpha.jobs.runner import run_job
from alpha.market_calendar import next_us_equity_session


ENTRY_DATE = date(2026, 5, 26)
EXIT_DATE = date(2026, 6, 15)
MATURE_RUN_TS = datetime(2026, 6, 16, 21, 0, tzinfo=timezone.utc)
EXIT_SESSION_CLOSE_TS = datetime(2026, 6, 15, 20, 0, tzinfo=timezone.utc)
IMMATURE_RUN_TS = datetime(2026, 6, 10, 21, 0, tzinfo=timezone.utc)
SIGNAL_TS = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)
REQUEST_TS = datetime(2026, 6, 16, 21, 1, tzinfo=timezone.utc)
PAST_ENTRY_DATE = date(2026, 5, 5)
PAST_EXIT_DATE = date(2026, 5, 26)
PAST_MATURE_RUN_TS = datetime(2026, 5, 27, 21, 0, tzinfo=timezone.utc)


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
                "volume": bar.volume,
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


class FakeCikSurvivorshipAdapter:
    def __init__(self, events_by_ticker: Optional[Dict[str, object]] = None):
        self.events_by_ticker = events_by_ticker or {}
        self.calls = []

    def get_survivorship_events(
        self,
        ticker,
        from_date=None,
        to_date=None,
        asof=None,
        cik=None,
    ):
        self.calls.append({
            "ticker": ticker,
            "from_date": from_date,
            "to_date": to_date,
            "asof": asof,
            "cik": cik,
        })
        raw_events = self.events_by_ticker.get(ticker, [])
        events = [raw_events] if isinstance(raw_events, dict) else list(raw_events or [])
        return AdapterResponse(
            data=events,
            lineage=LineageMeta(
                provider="TEST_CIK_SURVIVORSHIP",
                endpoint="/test/cik-survivorship-events",
                request_timestamp=REQUEST_TS,
                asof_timestamp=asof,
                raw_payload_hash=stable_hash(events),
                source_authority="test",
            ),
        )


class FakeLegacySurvivorshipAdapter:
    def __init__(self):
        self.calls = []

    def get_survivorship_events(
        self,
        ticker,
        from_date=None,
        to_date=None,
        asof=None,
    ):
        self.calls.append({
            "ticker": ticker,
            "from_date": from_date,
            "to_date": to_date,
            "asof": asof,
        })
        return AdapterResponse(
            data=[],
            lineage=LineageMeta(
                provider="TEST_LEGACY_SURVIVORSHIP",
                endpoint="/test/legacy-survivorship-events",
                request_timestamp=REQUEST_TS,
                asof_timestamp=asof,
                raw_payload_hash=stable_hash([]),
                source_authority="test",
            ),
        )


class FakeCikHistoricalAdapter(FakeHistoricalAdapter):
    def get_survivorship_events(
        self,
        ticker,
        from_date=None,
        to_date=None,
        asof=None,
        cik=None,
    ):
        self.survivorship_calls.append({
            "ticker": ticker,
            "from_date": from_date,
            "to_date": to_date,
            "asof": asof,
            "cik": cik,
        })
        return AdapterResponse(
            data=[],
            lineage=LineageMeta(
                provider="TEST_HISTORICAL_CIK_SURVIVORSHIP",
                endpoint="/test/historical-cik-survivorship-events",
                request_timestamp=REQUEST_TS,
                asof_timestamp=asof,
                raw_payload_hash=stable_hash([]),
                source_authority="test",
            ),
        )


class FakeEdgarSurvivorshipAdapter:
    requires_cik_for_survivorship_events = True

    def __init__(
        self,
        events_by_ticker: Optional[Dict[str, object]] = None,
        error: Optional[ProviderError] = None,
        company_ticker_rows: Optional[List[SecCompanyTicker]] = None,
        company_tickers_error: Optional[ProviderError] = None,
    ):
        self.events_by_ticker = events_by_ticker or {}
        self.error = error
        self.company_ticker_rows = company_ticker_rows or []
        self.company_tickers_error = company_tickers_error
        self.calls = []
        self.company_ticker_calls = []

    def get_survivorship_events(
        self,
        ticker,
        from_date=None,
        to_date=None,
        asof=None,
        cik=None,
    ):
        self.calls.append({
            "ticker": ticker,
            "from_date": from_date,
            "to_date": to_date,
            "asof": asof,
            "cik": cik,
        })
        raw_events = self.events_by_ticker.get(ticker, [])
        events = [raw_events] if isinstance(raw_events, dict) else list(raw_events or [])
        lineage = LineageMeta(
            provider="SEC_EDGAR",
            endpoint="sec_edgar_survivorship_events",
            request_timestamp=REQUEST_TS,
            asof_timestamp=asof,
            raw_payload_hash=stable_hash(events),
            source_authority="SEC_EDGAR",
        )
        if self.error is not None:
            return AdapterResponse(data=None, lineage=lineage, error=self.error)
        return AdapterResponse(data=events, lineage=lineage)

    def get_company_tickers(self, *, asof=None):
        self.company_ticker_calls.append({"asof": asof})
        rows = list(self.company_ticker_rows)
        lineage = LineageMeta(
            provider="SEC_EDGAR",
            endpoint="/files/company_tickers_exchange.json",
            request_timestamp=REQUEST_TS,
            asof_timestamp=asof,
            raw_payload_hash=stable_hash([
                {
                    "cik": row.cik_str,
                    "ticker": row.ticker,
                    "company_name": row.company_name,
                    "exchange": row.exchange,
                }
                for row in rows
            ]),
            source_authority="SEC_EDGAR",
        )
        if self.company_tickers_error is not None:
            return AdapterResponse(
                data=None,
                lineage=lineage,
                error=self.company_tickers_error,
            )
        return AdapterResponse(data=rows, lineage=lineage)


class FakeOptionalCikEdgarSurvivorshipAdapter(FakeEdgarSurvivorshipAdapter):
    requires_cik_for_survivorship_events = False


class FakeListingAuthorityAdapter:
    def __init__(
        self,
        result: Optional[NasdaqListingStatusResult] = None,
        error: Optional[ProviderError] = None,
    ):
        self.result = result
        self.error = error
        self.calls = []

    def get_listing_status(
        self,
        symbol,
        *,
        asof,
        archive_session=None,
        use_live=True,
    ):
        self.calls.append({
            "symbol": symbol,
            "asof": asof,
            "archive_session": archive_session,
            "use_live": use_live,
        })
        payload = {
            "symbol": symbol,
            "status": (
                self.result.status.value
                if isinstance(getattr(self.result, "status", None), NasdaqListingStatus)
                else getattr(self.result, "status", None)
            ),
            "error": self.error.error_type if self.error else None,
        }
        return AdapterResponse(
            data=None if self.error else self.result,
            lineage=LineageMeta(
                provider="NASDAQ_TRADER",
                endpoint="nasdaq_listing_status_archive",
                request_timestamp=REQUEST_TS,
                asof_timestamp=asof,
                raw_payload_hash=stable_hash(payload),
                source_authority="NASDAQ_TRADER_LISTING",
            ),
            error=self.error,
        )


class FakeBenzingaAdapter:
    def __init__(
        self,
        events_by_ticker: Optional[Dict[str, object]] = None,
        errors_by_ticker: Optional[Dict[str, ProviderError]] = None,
    ):
        self.events_by_ticker = events_by_ticker or {}
        self.errors_by_ticker = errors_by_ticker or {}
        self.calls = []

    def get_mergers_acquisitions(
        self,
        tickers=None,
        date_from=None,
        date_to=None,
        asof=None,
        **kwargs,
    ):
        self.calls.append({
            "tickers": tickers,
            "date_from": date_from,
            "date_to": date_to,
            "asof": asof,
            "kwargs": kwargs,
        })
        raw_events = self.events_by_ticker.get(tickers, [])
        if isinstance(raw_events, dict):
            events = [raw_events]
        else:
            events = list(raw_events or [])
        lineage = LineageMeta(
            provider="Benzinga",
            endpoint="/api/v2.1/calendar/ma",
            request_timestamp=REQUEST_TS,
            asof_timestamp=asof,
            raw_payload_hash=stable_hash(events),
            source_authority="Benzinga",
        )
        if tickers in self.errors_by_ticker:
            return AdapterResponse(
                data=None,
                lineage=lineage,
                error=self.errors_by_ticker[tickers],
            )
        return AdapterResponse(data=events, lineage=lineage)


class FakeDelistedAdapter(FakeHistoricalAdapter):
    def __init__(
        self,
        bars_by_ticker: Optional[Dict[str, List[FmpBar]]] = None,
        delisted_rows_by_page: Optional[Dict[int, List[FmpDelistedCompany]]] = None,
    ):
        super().__init__(bars_by_ticker)
        self.delisted_rows_by_page = delisted_rows_by_page or {}
        self.delisted_calls = []

    def get_delisted_companies(self, page=0, limit=100, asof=None):
        self.delisted_calls.append({"page": page, "limit": limit, "asof": asof})
        rows = self.delisted_rows_by_page.get(page, [])
        lineage = LineageMeta(
            provider="FMP",
            endpoint=DELISTED_COMPANIES_ENDPOINT,
            request_timestamp=REQUEST_TS,
            asof_timestamp=asof,
            raw_payload_hash=stable_hash([
                {
                    "symbol": row.symbol,
                    "company_name": row.company_name,
                    "delisted_date": row.delisted_date,
                }
                for row in rows
            ]),
            source_authority="FMP",
        )
        return AdapterResponse(data=rows, lineage=lineage)


def _bar(
    day: date,
    open_price,
    *,
    high=None,
    low=None,
    close=None,
    split_adjusted_close=None,
    adj_close=None,
    volume=1000,
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
        volume=volume,
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


def _listing_status_result(
    status: NasdaqListingStatus = NasdaqListingStatus.LISTED_ACTIVE,
    *,
    pit_knowable: bool = True,
    reason: str = "symbol_present_in_archived_directory",
    matched_symbol: str = "ACME",
    source_ts: datetime = EXIT_SESSION_CLOSE_TS,
) -> NasdaqListingStatusResult:
    return NasdaqListingStatusResult(
        symbol="ACME",
        normalized_symbol="ACME",
        status=status,
        asof_timestamp=EXIT_SESSION_CLOSE_TS,
        source_knowledge_timestamp=source_ts,
        pit_knowable_at_asof=pit_knowable,
        source="nasdaq_self_archive",
        reason=reason,
        matched_symbol=matched_symbol,
        raw={"test": True},
    )


def _sec_company_ticker(
    ticker: str = "ACME",
    cik: str = "0001418091",
    *,
    company_name: str = "ACME Corp.",
    exchange: str = "Nasdaq",
) -> SecCompanyTicker:
    cik10 = cik.zfill(10)
    return SecCompanyTicker(
        cik=int(cik10),
        cik_str=cik10,
        ticker=ticker,
        company_name=company_name,
        exchange=exchange,
        raw={"ticker": ticker, "cik": cik10},
    )


def _insert_nasdaq_archive_snapshot(
    db_session,
    source_type,
    source_ts,
    rows,
):
    snapshot = NasdaqListingSnapshot(
        source_type=source_type,
        source_url=f"https://example.test/{source_type}",
        source_knowledge_timestamp=source_ts,
        raw_payload_hash=f"{source_type}-{source_ts.isoformat()}-{len(rows)}",
        raw_payload="test payload",
        row_count=len(rows),
        parse_status="parsed",
        data_quality_flags_json=json.dumps({"source": source_type}),
    )
    db_session.add(snapshot)
    db_session.flush()
    for row in rows:
        row.snapshot_id = snapshot.snapshot_id
        db_session.add(row)
    db_session.flush()
    return snapshot


def _seed_nasdaq_archive_for_acme(db_session):
    source_ts = datetime(2026, 6, 15, 22, 15, tzinfo=timezone.utc)
    _insert_nasdaq_archive_snapshot(
        db_session,
        NASDAQ_LISTED,
        source_ts,
        [
            NasdaqListingSnapshotRow(
                source_type=NASDAQ_LISTED,
                symbol="ACME",
                normalized_symbol="ACME",
                security_name="ACME Corp. - Common Stock",
                market="Q",
                raw_json=json.dumps({
                    "ETF": "N",
                    "Security Name": "ACME Corp. - Common Stock",
                    "Symbol": "ACME",
                    "Test Issue": "N",
                }),
            )
        ],
    )
    for source_type in (OTHER_LISTED, ADDS_DELETES, HALT_RSS):
        _insert_nasdaq_archive_snapshot(db_session, source_type, source_ts, [])


def _session_window(start: date, end: date) -> List[date]:
    sessions = []
    cursor = next_us_equity_session(start)
    while cursor <= end:
        sessions.append(cursor)
        cursor = next_us_equity_session(cursor + timedelta(days=1))
    return sessions


def _make_signal(
    db_session,
    ticker="ACME",
    *,
    pattern_id="M4",
    signal_horizon="15d",
    next_execution_session="2026-05-26",
    trading_date="2026-05-26",
    signal_timestamp=SIGNAL_TS,
    security_identity: Optional[Dict[str, str]] = None,
) -> str:
    features = {"decision_date": trading_date, "signal_generated": True}
    if next_execution_session is not None:
        features["next_execution_session"] = next_execution_session
    if security_identity:
        features["security_identity"] = dict(security_identity)
    feat = record_feature_snapshot(
        db_session,
        pattern_id=pattern_id,
        ticker=ticker,
        asof_timestamp=signal_timestamp,
        features=features,
        data_lineage_ids=[],
    )
    sig = record_signal(
        db_session,
        pattern_id=pattern_id,
        ticker=ticker,
        direction="long",
        signal_timestamp=signal_timestamp,
        raw_signal_strength=0.9,
        raw_expected_edge=0.01,
        feature_snapshot_id=feat.feature_snapshot_id,
        signal_horizon=signal_horizon,
        signal_identity_hash=f"{pattern_id.lower()}-{ticker}",
        trading_date=trading_date,
        next_execution_session=next_execution_session,
    )
    db_session.flush()
    return sig.signal_id


def _run_job(
    db_session,
    adapter,
    *,
    pattern_id="M4",
    run_ts=MATURE_RUN_TS,
    max_attempts=3,
    finality_lag_sessions=1,
    reconcile_computed=False,
    revision_window_sessions=10,
    price_drift_abs_tol=0.01,
    price_drift_rel_tol=0.0005,
    survivorship_adapters=None,
    listing_authority_adapter=None,
):
    return run_job(
        db_session,
        ForwardReturnJob(
            session=db_session,
            adapter=adapter,
            pattern_id=pattern_id,
            survivorship_adapters=survivorship_adapters,
            listing_authority_adapter=listing_authority_adapter,
            run_timestamp=run_ts,
            max_attempts=max_attempts,
            finality_lag_sessions=finality_lag_sessions,
            reconcile_computed=reconcile_computed,
            revision_window_sessions=revision_window_sessions,
            price_drift_abs_tol=price_drift_abs_tol,
            price_drift_rel_tol=price_drift_rel_tol,
        ),
        params={
            "run_timestamp": run_ts.isoformat(),
            "finality_lag_sessions": finality_lag_sessions,
            "reconcile_computed": reconcile_computed,
            "revision_window_sessions": revision_window_sessions,
            "price_drift_abs_tol": price_drift_abs_tol,
            "price_drift_rel_tol": price_drift_rel_tol,
        },
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


def test_forward_return_prices_m1_variable_horizon_signal(db_session):
    _make_signal(
        db_session,
        ticker="FIRE",
        pattern_id="M1",
        signal_horizon="9d",
        next_execution_session="2026-05-26",
        trading_date="2026-05-20",
        signal_timestamp=datetime(2026, 5, 20, 20, 0, tzinfo=timezone.utc),
    )
    adapter = FakeHistoricalAdapter({
        "FIRE": [
            _bar(date(2026, 5, 26), 10.0, high=10.0, low=10.0, close=10.0),
            _bar(date(2026, 6, 1), 10.5, high=10.8, low=10.2, close=10.6),
            _bar(date(2026, 6, 5), 11.0, high=11.0, low=11.0, close=11.0),
        ]
    })

    result = _run_job(
        db_session,
        adapter,
        pattern_id="M1",
        run_ts=datetime(2026, 6, 8, 21, 0, tzinfo=timezone.utc),
    )

    assert result.status == "finished"
    assert result.metrics["total_eligible"] == 1
    assert result.metrics["computed"] == 1
    obs = _obs(db_session)
    assert obs.pattern_id == "M1"
    assert obs.signal_horizon == "9d"
    assert obs.next_execution_session == "2026-05-26"
    assert obs.entry_session_date == "2026-05-26"
    assert obs.exit_session_date == "2026-06-05"
    assert obs.forward_return == pytest.approx(0.10)
    assert obs.hit_t1_intraday is False
    assert obs.hit_stop_intraday is False


def test_forward_return_records_m2_no_barrier_horizon_metadata(db_session):
    _make_signal(
        db_session,
        ticker="FIRE",
        pattern_id="M2",
        signal_horizon="20d",
        next_execution_session=ENTRY_DATE.isoformat(),
        trading_date="2026-05-20",
        signal_timestamp=datetime(2026, 5, 20, 20, 0, tzinfo=timezone.utc),
    )
    plan = m4_entry_exit_plan(
        decision_date=date(2026, 5, 20),
        next_execution_session=ENTRY_DATE,
        current_evidence_session_date=date(2026, 6, 30),
        time_barrier_sessions=20,
    )
    adapter = FakeHistoricalAdapter({
        "FIRE": [
            _bar(ENTRY_DATE, 10.0, high=10.0, low=10.0, close=10.0),
            _bar(plan.exit_session_date, 11.0, high=11.0, low=11.0, close=11.0),
        ]
    })

    result = _run_job(
        db_session,
        adapter,
        pattern_id="M2",
        run_ts=datetime(2026, 6, 30, 21, 0, tzinfo=timezone.utc),
    )

    assert result.status == "finished"
    assert result.metrics["computed"] == 1
    obs = _obs(db_session)
    provider_request = json.loads(obs.provider_request_json)
    reconstruction = provider_request["price_request"]["forward_return_reconstruction"]
    assert obs.pattern_id == "M2"
    assert obs.signal_horizon == "20d"
    assert obs.exit_session_date == plan.exit_session_date.isoformat()
    assert reconstruction["pattern_id"] == "M2"
    assert reconstruction["signal_horizon"] == "20d"
    assert reconstruction["horizon_sessions"] == 20
    assert reconstruction["exit_geometry_source"] == "pattern_time_barrier_only"
    assert reconstruction["exit_geometry_source"] != M4_EXIT_GEOMETRY.source_contract


def test_forward_return_records_m3s_shadow_horizon_separately(db_session):
    _make_signal(
        db_session,
        ticker="SHDW",
        pattern_id="M3S",
        signal_horizon="15d",
        next_execution_session=ENTRY_DATE.isoformat(),
        trading_date="2026-05-20",
        signal_timestamp=datetime(2026, 5, 20, 20, 0, tzinfo=timezone.utc),
    )
    plan = m4_entry_exit_plan(
        decision_date=date(2026, 5, 20),
        next_execution_session=ENTRY_DATE,
        current_evidence_session_date=date(2026, 6, 16),
        time_barrier_sessions=15,
    )
    adapter = FakeHistoricalAdapter({
        "SHDW": [
            _bar(ENTRY_DATE, 10.0, high=10.0, low=10.0, close=10.0),
            _bar(plan.exit_session_date, 11.0, high=11.0, low=11.0, close=11.0),
        ]
    })

    result = _run_job(
        db_session,
        adapter,
        pattern_id="M3S",
        run_ts=datetime(2026, 6, 16, 21, 0, tzinfo=timezone.utc),
    )

    assert result.status == "finished"
    assert result.metrics["computed"] == 1
    obs = _obs(db_session)
    provider_request = json.loads(obs.provider_request_json)
    reconstruction = provider_request["price_request"]["forward_return_reconstruction"]
    assert obs.pattern_id == "M3S"
    assert obs.signal_horizon == "15d"
    assert obs.exit_session_date == plan.exit_session_date.isoformat()
    assert reconstruction["pattern_id"] == "M3S"
    assert reconstruction["signal_horizon"] == "15d"
    assert reconstruction["horizon_sessions"] == 15
    assert reconstruction["exit_geometry_source"] == "pattern_time_barrier_only"


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


def test_run_forward_return_wires_sec_edgar_when_user_agent_is_set(monkeypatch, capsys):
    monkeypatch.setenv("FMP_API_KEY", "fmp-key")
    monkeypatch.setenv("SEC_USER_AGENT", "Alpha Capital ops@example.com")
    monkeypatch.delenv("BENZINGA_API_KEY", raising=False)
    monkeypatch.delenv("BENZINGA_TOKEN", raising=False)
    monkeypatch.delenv("NASDAQ_LISTING_AUTHORITY_ENABLED", raising=False)
    monkeypatch.setattr(run_forward_return, "load_runtime_env", lambda: None)
    monkeypatch.setattr(run_forward_return, "_live_timestamp_error", lambda _value: None)

    class FakeSession:
        def close(self):
            pass

    class FakeFmpAdapter:
        def __init__(self, _config):
            pass

    class FakeSecEdgarAdapter:
        def __init__(self, _config):
            pass

    captured = {}

    def fake_run_job(session, job, params):
        captured["session"] = session
        captured["job"] = job
        captured["params"] = params
        return JobResult(status="finished", metrics={})

    monkeypatch.setattr(run_forward_return, "get_session", lambda: FakeSession())
    monkeypatch.setattr(run_forward_return, "FmpAdapter", FakeFmpAdapter)
    monkeypatch.setattr(run_forward_return, "SecEdgarAdapter", FakeSecEdgarAdapter)
    monkeypatch.setattr(run_forward_return, "run_job", fake_run_job)

    exit_code = run_forward_return.main([
        "--live",
        "--run-timestamp",
        MATURE_RUN_TS.isoformat(),
    ])

    assert exit_code == 0
    assert any(
        isinstance(adapter, FakeSecEdgarAdapter)
        for adapter in captured["job"]._survivorship_adapters
    )
    assert captured["params"]["survivorship_sources"] == [
        "fmp_delisted_companies",
        "sec_edgar_survivorship_events",
    ]
    assert "sec_edgar_survivorship_events" in capsys.readouterr().out


def test_run_forward_return_skips_sec_edgar_without_user_agent(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "fmp-key")
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("BENZINGA_API_KEY", raising=False)
    monkeypatch.delenv("BENZINGA_TOKEN", raising=False)
    monkeypatch.delenv("NASDAQ_LISTING_AUTHORITY_ENABLED", raising=False)
    monkeypatch.setattr(run_forward_return, "load_runtime_env", lambda: None)
    monkeypatch.setattr(run_forward_return, "_live_timestamp_error", lambda _value: None)

    class FakeSession:
        def close(self):
            pass

    class FakeFmpAdapter:
        def __init__(self, _config):
            pass

    class ExplodingSecEdgarAdapter:
        def __init__(self, _config):
            raise AssertionError("SEC EDGAR should be key-gated off")

    captured = {}

    def fake_run_job(session, job, params):
        captured["job"] = job
        captured["params"] = params
        return JobResult(status="finished", metrics={})

    monkeypatch.setattr(run_forward_return, "get_session", lambda: FakeSession())
    monkeypatch.setattr(run_forward_return, "FmpAdapter", FakeFmpAdapter)
    monkeypatch.setattr(run_forward_return, "SecEdgarAdapter", ExplodingSecEdgarAdapter)
    monkeypatch.setattr(run_forward_return, "run_job", fake_run_job)

    exit_code = run_forward_return.main([
        "--live",
        "--run-timestamp",
        MATURE_RUN_TS.isoformat(),
    ])

    assert exit_code == 0
    assert captured["job"]._survivorship_adapters == []
    assert captured["params"]["survivorship_sources"] == ["fmp_delisted_companies"]


def test_run_forward_return_wires_nasdaq_listing_authority_when_enabled(
    monkeypatch,
):
    monkeypatch.setenv("FMP_API_KEY", "fmp-key")
    monkeypatch.setenv("NASDAQ_LISTING_AUTHORITY_ENABLED", "1")
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("BENZINGA_API_KEY", raising=False)
    monkeypatch.delenv("BENZINGA_TOKEN", raising=False)
    monkeypatch.setattr(run_forward_return, "load_runtime_env", lambda: None)
    monkeypatch.setattr(run_forward_return, "_live_timestamp_error", lambda _value: None)

    class FakeSession:
        def close(self):
            pass

    class FakeFmpAdapter:
        def __init__(self, _config):
            pass

    class FakeNasdaqListingAuthority:
        pass

    captured = {}

    def fake_run_job(session, job, params):
        captured["job"] = job
        captured["params"] = params
        return JobResult(status="finished", metrics={})

    monkeypatch.setattr(run_forward_return, "get_session", lambda: FakeSession())
    monkeypatch.setattr(run_forward_return, "FmpAdapter", FakeFmpAdapter)
    monkeypatch.setattr(
        run_forward_return,
        "NasdaqTraderListingAdapter",
        FakeNasdaqListingAuthority,
    )
    monkeypatch.setattr(run_forward_return, "run_job", fake_run_job)

    exit_code = run_forward_return.main([
        "--live",
        "--run-timestamp",
        MATURE_RUN_TS.isoformat(),
    ])

    assert exit_code == 0
    assert isinstance(
        captured["job"]._listing_authority_adapter,
        FakeNasdaqListingAuthority,
    )
    assert captured["job"]._survivorship_adapters == []
    assert captured["params"]["survivorship_sources"] == [
        "fmp_delisted_companies",
        "nasdaq_listing_status",
    ]


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


def test_exit_complete_before_finality_lag_stores_provisional_prices(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=12.0)})
    exit_after_close_ts = datetime(2026, 6, 15, 21, 0, tzinfo=timezone.utc)

    result = _run_job(db_session, adapter, run_ts=exit_after_close_ts)

    sig = db_session.get(SignalRegistry, sid)
    assert result.metrics["price_finality_pending"] == 1
    assert len(adapter.calls) == 1
    assert adapter.survivorship_calls == []
    assert sig.forward_return_status == "price_finality_pending"
    assert sig.forward_return is None
    obs = _obs(db_session)
    assert obs.status == "price_finality_pending"
    assert obs.reason == "finality_lag_not_elapsed"
    assert obs.entry_price == 10.0
    assert obs.exit_price == 12.0
    assert obs.forward_return is None
    assert obs.entry_data_lineage_id
    assert obs.exit_data_lineage_id
    payload = json.loads(obs.provider_request_json)
    assert payload["price_finality"]["status"] == "price_finality_pending"
    assert payload["price_finality"]["finality_lag_sessions"] == 1


def test_finality_lag_elapsed_reconciles_matching_provisional_lineage(db_session):
    sid = _make_signal(db_session)
    provisional = FakeHistoricalAdapter({
        "ACME": _bars(entry_open=10.0, exit_open=12.0)
    })
    final = FakeHistoricalAdapter({
        "ACME": _bars(entry_open=10.0, exit_open=12.0)
    })

    _run_job(
        db_session,
        provisional,
        run_ts=datetime(2026, 6, 15, 21, 0, tzinfo=timezone.utc),
    )
    first_obs_id = _obs(db_session).forward_return_observation_id
    result = _run_job(db_session, final, run_ts=MATURE_RUN_TS)

    sig = db_session.get(SignalRegistry, sid)
    assert result.metrics["computed"] == 1
    assert sig.forward_return_status == "computed"
    assert sig.forward_return == 0.2
    assert db_session.query(ForwardReturnObservation).count() == 1
    assert db_session.query(ForwardReturnObservationEvent).count() == 2
    obs = _obs(db_session)
    assert obs.forward_return_observation_id == first_obs_id
    assert obs.status == "computed"
    assert obs.forward_return == 0.2
    payload = json.loads(obs.provider_request_json)
    finality = payload["price_finality"]
    assert finality["status"] == "reconciled"
    assert finality["provisional_observation_found"] is True
    assert finality["material_drift"] is False
    assert len(json.loads(obs.data_lineage_ids)) == 2


def test_finality_lag_elapsed_with_material_drift_requires_review(db_session):
    sid = _make_signal(db_session)
    provisional = FakeHistoricalAdapter({
        "ACME": _bars(entry_open=10.0, exit_open=12.0)
    })
    revised = FakeHistoricalAdapter({
        "ACME": _bars(entry_open=10.0, exit_open=12.2)
    })

    _run_job(
        db_session,
        provisional,
        run_ts=datetime(2026, 6, 15, 21, 0, tzinfo=timezone.utc),
    )
    _run_job(db_session, revised, run_ts=MATURE_RUN_TS)

    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "price_drift_review"
    assert sig.forward_return is None
    assert sig.outcome_unavailable_reason == "provider_price_drift_exceeds_tolerance"
    obs = _obs(db_session)
    assert obs.status == "price_drift_review"
    assert obs.forward_return is None
    assert obs.entry_price == 10.0
    assert obs.exit_price == 12.0
    payload = json.loads(obs.provider_request_json)
    finality = payload["price_finality"]
    assert finality["status"] == "price_drift_review"
    assert finality["material_drift"] is True
    assert finality["original"]["exit_open"] == 12.0
    assert finality["current"]["exit_open"] == 12.2
    assert finality["drift"]["material_drift_count"] >= 1
    assert len(json.loads(obs.data_lineage_ids)) == 2
    assert db_session.query(ForwardReturnObservationEvent).count() == 2


def test_finality_lag_elapsed_with_tolerated_drift_computes(db_session):
    sid = _make_signal(db_session)
    provisional = FakeHistoricalAdapter({
        "ACME": _bars(entry_open=10.0, exit_open=12.0)
    })
    revised = FakeHistoricalAdapter({
        "ACME": _bars(entry_open=10.0, exit_open=12.005)
    })

    _run_job(
        db_session,
        provisional,
        run_ts=datetime(2026, 6, 15, 21, 0, tzinfo=timezone.utc),
    )
    _run_job(db_session, revised, run_ts=MATURE_RUN_TS)

    sig = db_session.get(SignalRegistry, sid)
    assert sig.forward_return_status == "computed"
    assert round(sig.forward_return, 6) == 0.2005
    obs = _obs(db_session)
    assert obs.status == "computed"
    assert round(obs.forward_return, 6) == 0.2005
    payload = json.loads(obs.provider_request_json)
    assert payload["price_finality"]["status"] == "reconciled"
    assert payload["price_finality"]["material_drift"] is False


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
    request_payload = json.loads(event.provider_request_json)
    assert request_payload["price_request"]["price_field"] == "open"
    assert (
        request_payload["price_finality"]["status"]
        == "no_provisional_lineage_first_final_run"
    )


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


def test_max_attempts_one_terminalizes_retryable_on_first_attempt(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": [_bar(EXIT_DATE, 12.0)]})

    _run_job(db_session, adapter, max_attempts=1)

    sig = db_session.get(SignalRegistry, sid)
    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    assert sig.forward_return_status == "outcome_unavailable"
    assert sig.outcome_unavailable_reason == "missing_entry_price"
    assert sig.forward_return_attempts == 1
    assert obs.status == "outcome_unavailable"
    assert obs.reason == "missing_entry_price"
    assert event.status == "outcome_unavailable"
    assert event.reason == "missing_entry_price"


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


def test_survivorship_events_source_adapter_receives_signal_cik(db_session):
    _make_signal(
        db_session,
        security_identity={"cik": "320193"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    survivorship = FakeCikSurvivorshipAdapter()

    _run_job(db_session, adapter, survivorship_adapters=[survivorship])

    assert survivorship.calls[0]["cik"] == "0000320193"
    event = db_session.query(ForwardReturnObservationEvent).one()
    provider_request = json.loads(event.provider_request_json)
    attempts = provider_request["survivorship_request"]["source_attempts"]
    lineage = db_session.get(DataLineage, attempts[0]["lineage_id"])
    payload = json.loads(lineage.raw_payload_json)
    assert payload["request"]["cik"] == "0000320193"
    assert payload["request"]["cik_sent"] is True


def test_sec_edgar_survivorship_channel_receives_signal_cik_and_persists_authority(db_session):
    _make_signal(
        db_session,
        security_identity={"cik": "1418091"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "0001418091-22-000001",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0001418091",
            }
        },
        company_ticker_rows=[_sec_company_ticker("ACME", "0001418091")],
    )

    _run_job(db_session, adapter, survivorship_adapters=[edgar])

    assert edgar.calls[0]["cik"] == "0001418091"
    obs = _obs(db_session)
    assert obs.status == "corporate_action_review"
    assert obs.reason == "sec_edgar_form25_survivorship_review"
    assert obs.forward_return is None
    event = db_session.query(ForwardReturnObservationEvent).one()
    provider_request = json.loads(event.provider_request_json)
    survivorship_request = provider_request["survivorship_request"]
    attempts = survivorship_request["source_attempts"]
    assert attempts[0]["source"] == "sec_edgar_survivorship_events"
    assert attempts[0]["status"] == "matched"
    lineage = db_session.get(DataLineage, attempts[0]["lineage_id"])
    payload = json.loads(lineage.raw_payload_json)
    assert lineage.source_authority == "SEC_EDGAR"
    assert payload["request"]["cik"] == "0001418091"
    assert payload["request"]["cik_sent"] is True
    assert payload["request"]["asof"] == EXIT_SESSION_CLOSE_TS.isoformat()


def test_sec_edgar_form25_review_suppressed_by_pit_listed_active_authority(
    db_session,
):
    sid = _make_signal(
        db_session,
        security_identity={"cik": "1418091"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "0001418091-22-000001",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0001418091",
            }
        },
        company_ticker_rows=[_sec_company_ticker("ACME", "0001418091")],
    )
    listing = FakeListingAuthorityAdapter(_listing_status_result())

    _run_job(
        db_session,
        adapter,
        survivorship_adapters=[edgar],
        listing_authority_adapter=listing,
    )

    sig = db_session.get(SignalRegistry, sid)
    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]

    assert listing.calls == [
        {
            "symbol": "ACME",
            "asof": EXIT_SESSION_CLOSE_TS,
            "archive_session": db_session,
            "use_live": False,
        }
    ]
    assert sig.forward_return_status == "survivorship_unresolved_review"
    assert sig.outcome_unavailable_reason == NASDAQ_LISTING_SUPPRESSION_REASON
    assert obs.status == "survivorship_unresolved_review"
    assert obs.reason == NASDAQ_LISTING_SUPPRESSION_REASON
    assert obs.provider == "NASDAQ_TRADER"
    assert obs.endpoint == "nasdaq_listing_status_archive"
    assert obs.exit_price is None
    assert obs.forward_return is None
    assert survivorship_request["listing_authority_suppression"]["status"] == (
        "LISTED_ACTIVE"
    )
    assert survivorship_request["listing_authority_suppression"]["signal_cik"] == (
        "0001418091"
    )
    assert survivorship_request["listing_authority_suppression"]["matched_cik"] == (
        "0001418091"
    )
    assert survivorship_request["listing_authority_suppression"]["entity_match"] is True
    assert [attempt["source"] for attempt in attempts] == [
        "sec_edgar_survivorship_events",
        "survivorship_events",
        "nasdaq_listing_status",
    ]
    assert [attempt["status"] for attempt in attempts] == [
        "matched",
        "no_match",
        "matched",
    ]
    assert attempts[-1]["suppressed_edgar_review"] is True
    assert attempts[-1]["entity_match"] is True
    assert attempts[-1]["signal_cik"] == "0001418091"
    assert attempts[-1]["matched_cik"] == "0001418091"
    authorities = [
        db_session.get(DataLineage, attempt["lineage_id"]).source_authority
        for attempt in attempts
    ]
    assert authorities == ["SEC_EDGAR", "test", "NASDAQ_TRADER_LISTING"]


def test_sec_edgar_suppression_uses_real_nasdaq_archive_replay(db_session):
    _seed_nasdaq_archive_for_acme(db_session)
    _make_signal(
        db_session,
        security_identity={"cik": "1418091"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "0001418091-22-000001",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0001418091",
            }
        },
        company_ticker_rows=[_sec_company_ticker("ACME", "0001418091")],
    )

    _run_job(
        db_session,
        adapter,
        survivorship_adapters=[edgar],
        listing_authority_adapter=NasdaqTraderListingAdapter(),
    )

    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]

    assert obs.status == "survivorship_unresolved_review"
    assert obs.reason == NASDAQ_LISTING_SUPPRESSION_REASON
    assert survivorship_request["listing_authority_suppression"]["reason"] == (
        "symbol_present_in_archived_directory"
    )
    assert attempts[-1]["source"] == "nasdaq_listing_status"
    assert attempts[-1]["listing_status"] == "LISTED_ACTIVE"
    assert attempts[-1]["pit_knowable_at_asof"] is True
    assert attempts[-1]["entity_match"] is True
    assert db_session.get(DataLineage, attempts[-1]["lineage_id"]).source_authority == (
        "NASDAQ_TRADER_LISTING"
    )


def test_sec_edgar_suppression_applies_after_fmp_clean_no_match(db_session):
    _make_signal(
        db_session,
        ticker="TCON",
        security_identity={"cik": "1418091"},
    )
    adapter = FakeDelistedAdapter(
        {"TCON": [_bar(ENTRY_DATE, 10.0)]},
        delisted_rows_by_page={},
    )
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "TCON": {
                "id": "0001418091-22-000001",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0001418091",
            }
        },
        company_ticker_rows=[_sec_company_ticker("TCON", "0001418091")],
    )
    listing = FakeListingAuthorityAdapter(
        _listing_status_result(matched_symbol="TCON")
    )

    _run_job(
        db_session,
        adapter,
        survivorship_adapters=[edgar],
        listing_authority_adapter=listing,
    )

    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]

    assert obs.status == "survivorship_unresolved_review"
    assert obs.reason == NASDAQ_LISTING_SUPPRESSION_REASON
    assert attempts[-1]["source"] == "nasdaq_listing_status"
    assert attempts[-1]["suppressed_edgar_review"] is True
    assert attempts[-1]["entity_match"] is True
    assert "authority_conflict" not in survivorship_request


def test_listing_authority_pending_reviews_empty_list_noop(db_session):
    class ExplodingListingAdapter:
        def get_listing_status(self, *args, **kwargs):
            raise AssertionError("empty pending reviews should not query Nasdaq")

    source_attempts = []
    source_lineage_ids = []

    result = _apply_listing_authority_to_pending_edgar_reviews(
        [],
        listing_authority_adapter=ExplodingListingAdapter(),
        session=db_session,
        ticker="ACME",
        signal_identity={"cik": "1111"},
        entry_session_date=ENTRY_DATE,
        exit_session_date=EXIT_DATE,
        asof=EXIT_SESSION_CLOSE_TS,
        job_run_id=None,
        source_attempts=source_attempts,
        source_lineage_ids=source_lineage_ids,
    )

    assert result is None
    assert source_attempts == []
    assert source_lineage_ids == []


def test_recycled_ticker_cik_mismatch_does_not_suppress_edgar_review(db_session):
    _make_signal(
        db_session,
        security_identity={"cik": "1111"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "0000001111-22-000001",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0000001111",
            }
        },
        company_ticker_rows=[
            _sec_company_ticker("ACME", "0000002222"),
            _sec_company_ticker("OLD", "0000001111"),
        ],
    )
    listing = FakeListingAuthorityAdapter(_listing_status_result())

    _run_job(
        db_session,
        adapter,
        survivorship_adapters=[edgar],
        listing_authority_adapter=listing,
    )

    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]

    assert obs.status == "corporate_action_review"
    assert obs.reason == "sec_edgar_form25_survivorship_review"
    assert obs.provider == "SEC_EDGAR"
    assert obs.exit_price is None
    assert obs.forward_return is None
    assert "listing_authority_suppression" not in survivorship_request
    assert attempts[-1]["source"] == "nasdaq_listing_status"
    assert attempts[-1]["listing_status"] == "LISTED_ACTIVE"
    assert attempts[-1]["signal_cik"] == "0000001111"
    assert attempts[-1]["matched_cik"] == "0000002222"
    assert attempts[-1]["entity_match"] is False
    assert attempts[-1]["entity_match_status"] == "mismatch"
    assert attempts[-1]["entity_match_refuse_reason"] == (
        "active_listing_cik_mismatch"
    )
    assert attempts[-1]["suppression_candidate"] is False
    assert attempts[-1]["suppressed_edgar_review"] is False


def test_mixed_cik_ambiguity_refuses_listing_suppression(db_session):
    _make_signal(
        db_session,
        security_identity={"cik": "1111"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "0000001111-22-000001",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0000001111",
            }
        },
        company_ticker_rows=[
            _sec_company_ticker("ACME", "0000001111"),
            _sec_company_ticker("ACME", "0000002222"),
        ],
    )
    listing = FakeListingAuthorityAdapter(_listing_status_result())

    _run_job(
        db_session,
        adapter,
        survivorship_adapters=[edgar],
        listing_authority_adapter=listing,
    )

    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]
    refusal = survivorship_request["listing_authority_refusal"]

    assert obs.status == "corporate_action_review"
    assert obs.reason == "sec_edgar_form25_survivorship_review"
    assert obs.exit_price is None
    assert obs.forward_return is None
    assert "listing_authority_suppression" not in survivorship_request
    assert refusal["reason"] == "ambiguous_ticker_multiple_active_ciks"
    assert refusal["signal_cik"] == "0000001111"
    assert refusal["matched_cik"] == "0000001111"
    assert refusal["matched_ciks"] == ["0000001111", "0000002222"]
    assert refusal["entity_match"] is False
    assert refusal["entity_match_status"] == "ambiguous"
    assert attempts[-1]["source"] == "nasdaq_listing_status"
    assert attempts[-1]["entity_match_refuse_reason"] == (
        "ambiguous_ticker_multiple_active_ciks"
    )
    assert attempts[-1]["suppression_candidate"] is False
    assert attempts[-1]["suppressed_edgar_review"] is False


def test_recycled_ticker_cik_mismatch_does_not_suppress_after_fmp_clean_no_match(
    db_session,
):
    _make_signal(
        db_session,
        ticker="TCON",
        security_identity={"cik": "1111"},
    )
    adapter = FakeDelistedAdapter(
        {"TCON": [_bar(ENTRY_DATE, 10.0)]},
        delisted_rows_by_page={},
    )
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "TCON": {
                "id": "0000001111-22-000001",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0000001111",
            }
        },
        company_ticker_rows=[
            _sec_company_ticker("TCON", "0000002222"),
            _sec_company_ticker("OLD", "0000001111"),
        ],
    )
    listing = FakeListingAuthorityAdapter(
        _listing_status_result(matched_symbol="TCON")
    )

    _run_job(
        db_session,
        adapter,
        survivorship_adapters=[edgar],
        listing_authority_adapter=listing,
    )

    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]

    assert obs.status == "corporate_action_review"
    assert obs.reason == "sec_edgar_form25_survivorship_review"
    assert "listing_authority_suppression" not in survivorship_request
    assert attempts[-1]["source"] == "nasdaq_listing_status"
    assert attempts[-1]["entity_match"] is False
    assert attempts[-1]["entity_match_refuse_reason"] == (
        "active_listing_cik_mismatch"
    )
    assert attempts[-1]["suppressed_edgar_review"] is False


def test_suppressed_numeric_cusip_cik_does_not_suppress_edgar_review(db_session):
    _make_signal(
        db_session,
        security_identity={"cik": "037833100", "cusip": "037833100"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeOptionalCikEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "bad-cusip-cik",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0037833100",
            }
        },
        company_ticker_rows=[_sec_company_ticker("ACME", "0037833100")],
    )
    listing = FakeListingAuthorityAdapter(_listing_status_result())

    _run_job(
        db_session,
        adapter,
        survivorship_adapters=[edgar],
        listing_authority_adapter=listing,
    )

    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]

    assert obs.status == "corporate_action_review"
    assert obs.reason == "sec_edgar_form25_survivorship_review"
    assert "listing_authority_suppression" not in survivorship_request
    assert attempts[-1]["source"] == "nasdaq_listing_status"
    assert attempts[-1]["signal_cik"] is None
    assert attempts[-1]["entity_match"] == "unresolved"
    assert attempts[-1]["entity_match_refuse_reason"] == "cik_equals_numeric_cusip"
    assert attempts[-1]["suppressed_edgar_review"] is False


def test_lone_numeric_cusip_in_cik_field_does_not_suppress_on_entity_mismatch(
    db_session,
):
    _make_signal(
        db_session,
        security_identity={"cik": "037833100"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "wrong-cik-form25",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0037833100",
            }
        },
        company_ticker_rows=[_sec_company_ticker("ACME", "0001418091")],
    )
    listing = FakeListingAuthorityAdapter(_listing_status_result())

    _run_job(
        db_session,
        adapter,
        survivorship_adapters=[edgar],
        listing_authority_adapter=listing,
    )

    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]

    assert obs.status == "corporate_action_review"
    assert obs.reason == "sec_edgar_form25_survivorship_review"
    assert "listing_authority_suppression" not in survivorship_request
    assert attempts[-1]["signal_cik"] == "0037833100"
    assert attempts[-1]["matched_cik"] == "0001418091"
    assert attempts[-1]["entity_match"] is False
    assert attempts[-1]["entity_match_refuse_reason"] == (
        "active_listing_cik_mismatch"
    )
    assert attempts[-1]["suppressed_edgar_review"] is False


def test_multi_edgar_reviews_use_their_own_identity_adapters(db_session):
    _make_signal(
        db_session,
        security_identity={"cik": "1111"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar_matching = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "matching-edgar-review",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0000001111",
            }
        },
        company_ticker_rows=[_sec_company_ticker("ACME", "0000001111")],
    )
    edgar_mismatch = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "mismatched-edgar-review",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0000001111",
            }
        },
        company_ticker_rows=[_sec_company_ticker("ACME", "0000002222")],
    )
    listing = FakeListingAuthorityAdapter(_listing_status_result())

    _run_job(
        db_session,
        adapter,
        survivorship_adapters=[edgar_matching, edgar_mismatch],
        listing_authority_adapter=listing,
    )

    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]
    listing_attempts = [
        attempt for attempt in attempts
        if attempt["source"] == "nasdaq_listing_status"
    ]

    assert obs.status == "corporate_action_review"
    assert obs.reason == "sec_edgar_form25_survivorship_review"
    assert "listing_authority_suppression" not in survivorship_request
    assert len(listing.calls) == 2
    assert len(edgar_matching.company_ticker_calls) == 1
    assert len(edgar_mismatch.company_ticker_calls) == 1
    assert len(listing_attempts) == 2
    assert listing_attempts[0]["entity_match"] is True
    assert listing_attempts[0]["suppressed_edgar_review"] is True
    assert listing_attempts[1]["entity_match"] is False
    assert listing_attempts[1]["matched_cik"] == "0000002222"
    assert listing_attempts[1]["entity_match_refuse_reason"] == (
        "active_listing_cik_mismatch"
    )
    first_identity_lineage = db_session.get(
        DataLineage,
        listing_attempts[0]["entity_lineage_id"],
    )
    second_identity_lineage = db_session.get(
        DataLineage,
        listing_attempts[1]["entity_lineage_id"],
    )
    first_payload = json.loads(first_identity_lineage.raw_payload_json)
    second_payload = json.loads(second_identity_lineage.raw_payload_json)
    assert first_payload["ticker_rows"][0]["cik"] == "0000001111"
    assert second_payload["ticker_rows"][0]["cik"] == "0000002222"


def test_listing_entity_mapping_error_does_not_suppress_edgar_review(db_session):
    _make_signal(
        db_session,
        security_identity={"cik": "1418091"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "0001418091-22-000001",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0001418091",
            }
        },
        company_tickers_error=ProviderError(
            provider="SEC_EDGAR",
            endpoint="/files/company_tickers_exchange.json",
            status_code=503,
            error_type="http",
            message="SEC temporarily unavailable",
            retryable=True,
        ),
    )
    listing = FakeListingAuthorityAdapter(_listing_status_result())

    _run_job(
        db_session,
        adapter,
        survivorship_adapters=[edgar],
        listing_authority_adapter=listing,
    )

    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]

    assert obs.status == "corporate_action_review"
    assert obs.reason == "sec_edgar_form25_survivorship_review"
    assert "listing_authority_suppression" not in survivorship_request
    assert attempts[-1]["source"] == "nasdaq_listing_status"
    assert attempts[-1]["entity_match"] == "unresolved"
    assert attempts[-1]["entity_match_refuse_reason"] == "entity_mapping_error:http"
    assert attempts[-1]["entity_lineage_id"]
    assert attempts[-1]["suppressed_edgar_review"] is False


@pytest.mark.parametrize(
    ("listing_result", "expected_reason"),
    [
        (
            _listing_status_result(
                NasdaqListingStatus.LISTED_ACTIVE,
                pit_knowable=False,
                reason="directory_match_not_knowable_at_asof",
            ),
            "directory_match_not_knowable_at_asof",
        ),
        (
            _listing_status_result(
                NasdaqListingStatus.INCONCLUSIVE,
                reason="archive_source_coverage_incomplete",
            ),
            "archive_source_coverage_incomplete",
        ),
        (
            _listing_status_result(
                NasdaqListingStatus.UNAVAILABLE,
                reason="provider_error",
            ),
            "provider_error",
        ),
        (
            _listing_status_result(
                NasdaqListingStatus.DELISTED,
                reason="archived_adds_deletes_delete",
            ),
            "archived_adds_deletes_delete",
        ),
    ],
)
def test_sec_edgar_review_not_suppressed_without_pit_active_listing(
    db_session,
    listing_result,
    expected_reason,
):
    _make_signal(
        db_session,
        security_identity={"cik": "1418091"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "0001418091-22-000001",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0001418091",
            }
        }
    )
    listing = FakeListingAuthorityAdapter(listing_result)

    _run_job(
        db_session,
        adapter,
        survivorship_adapters=[edgar],
        listing_authority_adapter=listing,
    )

    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]

    assert obs.status == "corporate_action_review"
    assert obs.reason == "sec_edgar_form25_survivorship_review"
    assert obs.provider == "SEC_EDGAR"
    assert "listing_authority_suppression" not in survivorship_request
    assert attempts[-1]["source"] == "nasdaq_listing_status"
    assert attempts[-1]["listing_reason"] == expected_reason
    assert attempts[-1]["suppressed_edgar_review"] is False


def test_sec_edgar_review_not_suppressed_when_listing_authority_errors(
    db_session,
):
    _make_signal(
        db_session,
        security_identity={"cik": "1418091"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "0001418091-22-000001",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0001418091",
            }
        }
    )
    listing = FakeListingAuthorityAdapter(
        error=ProviderError(
            provider="NASDAQ_TRADER",
            endpoint="nasdaq_listing_status_archive",
            status_code=None,
            error_type="coverage_incomplete",
            message="missing archive coverage",
            retryable=True,
        )
    )

    _run_job(
        db_session,
        adapter,
        survivorship_adapters=[edgar],
        listing_authority_adapter=listing,
    )

    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]

    assert obs.status == "corporate_action_review"
    assert obs.reason == "sec_edgar_form25_survivorship_review"
    assert attempts[-1]["source"] == "nasdaq_listing_status"
    assert attempts[-1]["status"] == "error"
    assert attempts[-1]["error"]["error_type"] == "coverage_incomplete"


def test_listing_authority_absent_preserves_sec_edgar_review_shape(db_session):
    _make_signal(
        db_session,
        security_identity={"cik": "1418091"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "0001418091-22-000001",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0001418091",
            }
        }
    )

    _run_job(db_session, adapter, survivorship_adapters=[edgar])

    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]

    assert obs.status == "corporate_action_review"
    assert obs.reason == "sec_edgar_form25_survivorship_review"
    assert obs.provider == "SEC_EDGAR"
    assert [attempt["source"] for attempt in survivorship_request["source_attempts"]] == [
        "sec_edgar_survivorship_events",
        "survivorship_events",
    ]
    assert "listing_authority_suppression" not in survivorship_request


def test_sec_edgar_without_signal_cik_records_identity_unavailable_not_ticker_lookup(db_session):
    _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "ticker-only-must-not-match",
                "type": "delisting_notice",
                "source_backed": True,
            }
        }
    )

    _run_job(db_session, adapter, survivorship_adapters=[edgar])

    assert edgar.calls == []
    obs = _obs(db_session)
    assert obs.status == "survivorship_unresolved_review"
    event = db_session.query(ForwardReturnObservationEvent).one()
    provider_request = json.loads(event.provider_request_json)
    attempts = provider_request["survivorship_request"]["source_attempts"]
    assert attempts[0]["source"] == "sec_edgar_survivorship_events"
    assert attempts[0]["status"] == "error"
    assert attempts[0]["identity_status"] == "identity_unavailable"
    assert attempts[0]["error"]["error_type"] == "unresolved_entity"
    lineage = db_session.get(DataLineage, attempts[0]["lineage_id"])
    payload = json.loads(lineage.raw_payload_json)
    assert lineage.source_authority == "SEC_EDGAR"
    assert payload["request"]["cik_sent"] is False
    assert payload["request"]["identity_status"] == "identity_unavailable"
    assert payload["request"]["asof"] == EXIT_SESSION_CLOSE_TS.isoformat()
    assert "cik" not in payload["request"]


def test_sec_edgar_form25_note_like_event_routes_to_review_not_terminal(db_session):
    sid = _make_signal(
        db_session,
        security_identity={"cik": "789019"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "0000789019-24-000001",
                "type": "delisting_notice",
                "classification": "sec_form_25",
                "source_backed": True,
                "form": "25",
                "security_title": "2.525% Notes due 2050",
                "cik": "0000789019",
            }
        }
    )

    _run_job(db_session, adapter, survivorship_adapters=[edgar])

    sig = db_session.get(SignalRegistry, sid)
    obs = _obs(db_session)
    assert sig.forward_return_status == "corporate_action_review"
    assert sig.forward_return is None
    assert obs.status == "corporate_action_review"
    assert obs.reason == "sec_edgar_form25_survivorship_review"


def test_sec_edgar_review_collects_benzinga_and_fmp_corroboration(db_session):
    sid = _make_signal(
        db_session,
        ticker="TCON",
        security_identity={"cik": "1418091", "cusip": "004397105"},
    )
    adapter = FakeDelistedAdapter(
        {"TCON": [_bar(ENTRY_DATE, 10.0)]},
        delisted_rows_by_page={
            0: [
                FmpDelistedCompany(
                    symbol="TCON",
                    company_name="Ticker Conflict Corp",
                    delisted_date=EXIT_DATE.isoformat(),
                )
            ]
        },
    )
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "TCON": {
                "id": "0001418091-22-000001",
                "type": "delisting_notice",
                "classification": "sec_form_25-nse",
                "source_backed": True,
                "form": "25-NSE",
                "cik": "0001418091",
            }
        }
    )
    benzinga = FakeBenzingaAdapter(
        {
            "TCON": {
                "id": "deal-conflict",
                "target_ticker": "TCON",
                "target_cusip": "004397105",
                "acquirer_ticker": "BUY",
                "date_completed": EXIT_DATE.isoformat(),
            }
        }
    )
    listing = FakeListingAuthorityAdapter(_listing_status_result())

    _run_job(
        db_session,
        adapter,
        survivorship_adapters=[edgar, benzinga],
        listing_authority_adapter=listing,
    )

    sig = db_session.get(SignalRegistry, sid)
    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]

    assert len(edgar.calls) == 1
    assert len(benzinga.calls) == 1
    assert len(adapter.delisted_calls) >= 1
    assert sig.forward_return_status == "corporate_action_review"
    assert obs.provider == "SEC_EDGAR"
    assert obs.endpoint == "sec_edgar_survivorship_events"
    assert obs.reason == "sec_edgar_form25_survivorship_review"
    assert obs.forward_return is None
    assert obs.exit_price is None
    assert (
        survivorship_request["authority_corroboration"][0]["provider"]
        == "Benzinga"
    )
    assert (
        survivorship_request["authority_corroboration"][0]["reason"]
        == "benzinga_merger_acquisition_review"
    )
    assert survivorship_request["authority_conflict"]["provider"] == "FMP"
    assert (
        survivorship_request["authority_conflict"]["reason"]
        == "delisting_unclassified_survivorship_review"
    )
    assert "listing_authority_suppression" not in survivorship_request
    assert [attempt["source"] for attempt in attempts] == [
        "sec_edgar_survivorship_events",
        "benzinga_calendar_ma",
        "survivorship_events",
        "fmp_delisted_companies",
        "nasdaq_listing_status",
    ]
    assert [attempt["status"] for attempt in attempts] == [
        "matched",
        "matched",
        "no_match",
        "matched",
        "matched",
    ]
    assert attempts[-1]["suppressed_edgar_review"] is False
    authorities = [
        db_session.get(DataLineage, attempt["lineage_id"]).source_authority
        for attempt in attempts
    ]
    assert authorities == [
        "SEC_EDGAR",
        "Benzinga",
        "test",
        "FMP",
        "NASDAQ_TRADER_LISTING",
    ]


def test_sec_edgar_incomplete_window_routes_to_review_with_lineage(db_session):
    _make_signal(
        db_session,
        security_identity={"cik": "1418091"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeEdgarSurvivorshipAdapter(
        error=ProviderError(
            provider="SEC_EDGAR",
            endpoint="sec_edgar_survivorship_events",
            status_code=None,
            error_type="incomplete_window",
            message="truncated submissions window",
            retryable=False,
        )
    )

    _run_job(db_session, adapter, survivorship_adapters=[edgar])

    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    provider_request = json.loads(event.provider_request_json)
    attempts = provider_request["survivorship_request"]["source_attempts"]
    assert obs.status == "survivorship_unresolved_review"
    assert obs.reason == "survivorship_source_error:incomplete_window"
    assert attempts[0]["source"] == "sec_edgar_survivorship_events"
    assert attempts[0]["status"] == "error"
    assert attempts[0]["error"]["error_type"] == "incomplete_window"
    lineage = db_session.get(DataLineage, attempts[0]["lineage_id"])
    assert lineage.source_authority == "SEC_EDGAR"


def test_survivorship_events_primary_adapter_receives_signal_cik(db_session):
    _make_signal(
        db_session,
        security_identity={"cik": "0000320193"},
    )
    adapter = FakeCikHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})

    _run_job(db_session, adapter)

    assert adapter.survivorship_calls[0]["cik"] == "0000320193"


def test_survivorship_events_legacy_adapter_without_cik_signature_still_works(
    db_session,
):
    _make_signal(
        db_session,
        security_identity={"cik": "320193"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    survivorship = FakeLegacySurvivorshipAdapter()

    _run_job(db_session, adapter, survivorship_adapters=[survivorship])

    assert survivorship.calls[0]["ticker"] == "ACME"
    assert "cik" not in survivorship.calls[0]
    assert _obs(db_session).status == "survivorship_unresolved_review"
    event = db_session.query(ForwardReturnObservationEvent).one()
    provider_request = json.loads(event.provider_request_json)
    attempts = provider_request["survivorship_request"]["source_attempts"]
    lineage = db_session.get(DataLineage, attempts[0]["lineage_id"])
    payload = json.loads(lineage.raw_payload_json)
    assert "cik" not in payload["request"]
    assert payload["request"]["cik_sent"] is False


def test_survivorship_events_without_signal_cik_remains_ticker_only(db_session):
    _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    survivorship = FakeCikSurvivorshipAdapter()

    _run_job(db_session, adapter, survivorship_adapters=[survivorship])

    assert survivorship.calls[0]["ticker"] == "ACME"
    assert survivorship.calls[0]["cik"] is None


def test_security_identity_from_payload_extracts_normalized_cik():
    assert _security_identity_from_payload({
        "security_identity": {"cik": "320193"},
    })["cik"] == "0000320193"
    assert _security_identity_from_payload({
        "signal_context": {
            "identity": {
                "security_identity": {"central_index_key": "0000320193"}
            }
        },
    })["cik"] == "0000320193"


def test_security_identity_suppresses_cik_manufactured_from_numeric_cusip(
    db_session,
):
    """Suppress only a CIK numerically equal to a co-present all-numeric CUSIP.

    A genuine issuer CIK should not equal the record's own CUSIP. In the
    unlikely event this suppresses a real CIK, the failure mode is safer than a
    wrong-issuer EDGAR query: EDGAR is skipped and other survivorship channels
    remain available. The diagnostic flag is inert because resolver consumers
    read explicit identity keys instead of iterating unknown keys.
    """

    identity = _security_identity_from_payload({
        "security_identity": {
            "cik": "037833100",
            "cusip": "037833100",
        }
    })
    assert identity == {
        "cusip": "037833100",
        "cik_suppressed_reason": "cik_equals_numeric_cusip",
    }

    _make_signal(
        db_session,
        security_identity={
            "cik": "037833100",
            "cusip": "037833100",
        },
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeEdgarSurvivorshipAdapter(
        events_by_ticker={
            "ACME": {
                "id": "wrong-issuer-must-not-query",
                "type": "delisting_notice",
                "source_backed": True,
            }
        }
    )

    _run_job(db_session, adapter, survivorship_adapters=[edgar])

    assert edgar.calls == []
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]
    assert attempts[0]["source"] == "sec_edgar_survivorship_events"
    assert attempts[0]["status"] == "error"
    assert attempts[0]["identity_status"] == "identity_unavailable"
    lineage = db_session.get(DataLineage, attempts[0]["lineage_id"])
    payload = json.loads(lineage.raw_payload_json)
    assert payload["request"]["cik_sent"] is False
    assert "cik" not in payload["request"]


def test_security_identity_preserves_real_cik_with_different_cusip(db_session):
    identity = _security_identity_from_payload({
        "security_identity": {
            "cik": "0000320193",
            "cusip": "037833100",
        }
    })
    assert identity == {"cik": "0000320193", "cusip": "037833100"}

    _make_signal(
        db_session,
        security_identity={
            "cik": "0000320193",
            "cusip": "037833100",
        },
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    edgar = FakeEdgarSurvivorshipAdapter()

    _run_job(db_session, adapter, survivorship_adapters=[edgar])

    assert edgar.calls[0]["cik"] == "0000320193"


def test_security_identity_preserves_cik_when_cusip_absent():
    identity = _security_identity_from_payload({
        "security_identity": {"cik": "0000320193"}
    })
    assert identity == {"cik": "0000320193"}


def test_security_identity_rejects_alphanumeric_cusip_in_cik_field():
    identity = _security_identity_from_payload({
        "security_identity": {
            "cik": "38259P508",
            "cusip": "38259P508",
        }
    })
    assert identity == {"cusip": "38259P508"}


@pytest.mark.parametrize("value", [
    "320193",
    "0000320193",
    "CIK0000320193",
    " cik 0000320193 ",
    320193,
])
def test_canonical_cik10_accepts_clean_cik_forms(value):
    assert _canonical_cik10(value) == "0000320193"


@pytest.mark.parametrize("value", [
    "0000320193-extra-99",
    "BBG000B9XRY4",
    "3.20193e5",
    "320000.0",
    "1234567890123",
    "",
    "0000000000",
    None,
])
def test_canonical_cik10_rejects_malformed_cik_values(value):
    assert _canonical_cik10(value) is None


def test_malformed_signal_cik_fails_safe_to_ticker_only_survivorship(db_session):
    _make_signal(
        db_session,
        security_identity={"cik": "BBG000B9XRY4"},
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    survivorship = FakeCikSurvivorshipAdapter()

    _run_job(db_session, adapter, survivorship_adapters=[survivorship])

    assert survivorship.calls[0]["ticker"] == "ACME"
    assert survivorship.calls[0]["cik"] is None
    assert _obs(db_session).status == "survivorship_unresolved_review"
    event = db_session.query(ForwardReturnObservationEvent).one()
    provider_request = json.loads(event.provider_request_json)
    attempts = provider_request["survivorship_request"]["source_attempts"]
    lineage = db_session.get(DataLineage, attempts[0]["lineage_id"])
    payload = json.loads(lineage.raw_payload_json)
    assert "cik" not in payload["request"]
    assert payload["request"]["cik_sent"] is False


def test_survivorship_events_source_no_match_primary_fallback_order_preserved(
    db_session,
):
    _make_signal(
        db_session,
        security_identity={"cik": "320193"},
    )
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
    survivorship = FakeCikSurvivorshipAdapter()

    _run_job(db_session, adapter, survivorship_adapters=[survivorship])

    event = db_session.query(ForwardReturnObservationEvent).one()
    attempts = json.loads(event.provider_request_json)["survivorship_request"][
        "source_attempts"
    ]
    assert survivorship.calls[0]["cik"] == "0000320193"
    assert adapter.survivorship_calls[0]["ticker"] == "ACME"
    assert attempts[0]["source"] == "survivorship_events"
    assert attempts[0]["status"] == "no_match"
    assert attempts[1]["source"] == "survivorship_events"
    assert attempts[1]["status"] == "matched"


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


def test_benzinga_merger_acquisition_routes_missing_exit_to_review(db_session):
    sid = _make_signal(
        db_session,
        security_identity={
            "cusip": "004397105",
            "isin": "US0043971052",
        },
    )
    adapter = FakeHistoricalAdapter({"ACME": [_bar(ENTRY_DATE, 10.0)]})
    benzinga = FakeBenzingaAdapter(
        {
            "ACME": {
                "id": "deal-42",
                "target_ticker": "ACME",
                "target_name": "Acme Corp",
                "target_cusip": "004397105",
                "target_isin": "US0043971052",
                "acquirer_ticker": "BUY",
                "acquirer_name": "Buyer Inc",
                "acquirer_cusip": "124857202",
                "acquirer_isin": "US1248572026",
                "deal_type": "Merger",
                "deal_status": "Completed",
                "deal_payment_type": "Cash",
                "date_completed": EXIT_DATE.isoformat(),
                "deal_terms_extra": "$14.00 per share in cash",
            }
        }
    )

    _run_job(db_session, adapter, survivorship_adapters=[benzinga])

    sig = db_session.get(SignalRegistry, sid)
    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    provider_request = json.loads(event.provider_request_json)
    survivorship_request = provider_request["survivorship_request"]

    assert benzinga.calls[0]["tickers"] == "ACME"
    assert benzinga.calls[0]["date_from"] == ENTRY_DATE.isoformat()
    assert benzinga.calls[0]["date_to"] == EXIT_DATE.isoformat()
    assert adapter.survivorship_calls[0]["ticker"] == "ACME"
    assert sig.forward_return_status == "corporate_action_review"
    assert sig.outcome_unavailable_reason == "benzinga_merger_acquisition_review"
    assert obs.status == "corporate_action_review"
    assert obs.provider == "Benzinga"
    assert obs.endpoint == "/api/v2.1/calendar/ma"
    assert json.loads(obs.data_lineage_ids)
    assert survivorship_request["source"] == "benzinga_calendar_ma"
    assert survivorship_request["matched_id"] == "deal-42"
    assert survivorship_request["matched_target_ticker"] == "ACME"
    assert survivorship_request["matched_target_cusip"] == "004397105"
    assert survivorship_request["matched_target_isin"] == "US0043971052"
    assert survivorship_request["matched_acquirer_ticker"] == "BUY"
    assert survivorship_request["matched_acquirer_cusip"] == "124857202"
    assert survivorship_request["matched_acquirer_isin"] == "US1248572026"
    assert survivorship_request["economic_classification"] == "review_required"
    assert survivorship_request["source_attempts"][0]["source"] == "benzinga_calendar_ma"
    assert survivorship_request["source_attempts"][0]["status"] == "matched"
    assert survivorship_request["source_attempts"][0]["match_basis"] == "target_cusip"
    assert survivorship_request["source_attempts"][1]["source"] == "survivorship_events"
    assert survivorship_request["source_attempts"][1]["status"] == "no_match"


def test_benzinga_recycled_ticker_identity_mismatch_does_not_match(db_session):
    sid = _make_signal(
        db_session,
        ticker="XYZ",
        security_identity={"cusip": "A11111111"},
    )
    adapter = FakeHistoricalAdapter({"XYZ": [_bar(ENTRY_DATE, 10.0)]})
    benzinga = FakeBenzingaAdapter(
        {
            "XYZ": {
                "id": "wrong-company-deal",
                "target_ticker": "XYZ",
                "target_cusip": "B22222222",
                "acquirer_ticker": "BUY",
                "date_completed": EXIT_DATE.isoformat(),
            }
        }
    )

    _run_job(db_session, adapter, survivorship_adapters=[benzinga])

    sig = db_session.get(SignalRegistry, sid)
    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    attempts = json.loads(event.provider_request_json)["survivorship_request"][
        "source_attempts"
    ]

    assert sig.forward_return_status == "survivorship_unresolved_review"
    assert obs.provider != "Benzinga"
    assert attempts[0]["source"] == "benzinga_calendar_ma"
    assert attempts[0]["status"] == "identity_mismatch"


def test_benzinga_no_match_attempt_is_preserved_when_fallback_handles_row(db_session):
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
    benzinga = FakeBenzingaAdapter({"ACME": []})

    _run_job(db_session, adapter, survivorship_adapters=[benzinga])

    sig = db_session.get(SignalRegistry, sid)
    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    provider_request = json.loads(event.provider_request_json)
    survivorship_request = provider_request["survivorship_request"]
    attempts = survivorship_request["source_attempts"]

    assert sig.forward_return_status == "computed"
    assert obs.forward_return == 0.4
    assert attempts[0]["source"] == "benzinga_calendar_ma"
    assert attempts[0]["status"] == "no_match"
    assert attempts[1]["source"] == "survivorship_events"
    assert attempts[1]["status"] == "matched"
    assert len(json.loads(obs.data_lineage_ids)) == 3


def test_fmp_delisted_fallback_uses_uniform_source_attempts(db_session):
    sid = _make_signal(db_session)
    adapter = FakeDelistedAdapter(
        {"ACME": [_bar(ENTRY_DATE, 10.0)]},
        delisted_rows_by_page={
            0: [
                FmpDelistedCompany(
                    symbol="ACME",
                    company_name="Acme Corp",
                    delisted_date=EXIT_DATE.isoformat(),
                )
            ]
        },
    )
    benzinga = FakeBenzingaAdapter({"ACME": []})

    _run_job(db_session, adapter, survivorship_adapters=[benzinga])

    sig = db_session.get(SignalRegistry, sid)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]

    assert sig.forward_return_status == "survivorship_unresolved_review"
    assert survivorship_request["source"] == DELISTED_COMPANIES_ENDPOINT
    assert attempts[0]["source"] == "benzinga_calendar_ma"
    assert attempts[0]["status"] == "no_match"
    assert attempts[1]["source"] == "survivorship_events"
    assert attempts[1]["status"] == "no_match"
    assert attempts[2]["source"] == "fmp_delisted_companies"
    assert attempts[2]["status"] == "matched"
    assert attempts[2]["matched_id"] == "ACME"


def test_benzinga_fmp_authority_conflict_is_surfaced(db_session):
    sid = _make_signal(
        db_session,
        ticker="TCON",
        security_identity={"cusip": "004397105"},
    )
    adapter = FakeDelistedAdapter(
        {"TCON": [_bar(ENTRY_DATE, 10.0)]},
        delisted_rows_by_page={
            0: [
                FmpDelistedCompany(
                    symbol="TCON",
                    company_name="Ticker Conflict Corp",
                    delisted_date=EXIT_DATE.isoformat(),
                )
            ]
        },
    )
    benzinga = FakeBenzingaAdapter(
        {
            "TCON": {
                "id": "deal-conflict",
                "target_ticker": "TCON",
                "target_cusip": "004397105",
                "acquirer_ticker": "BUY",
                "date_completed": EXIT_DATE.isoformat(),
            }
        }
    )

    _run_job(db_session, adapter, survivorship_adapters=[benzinga])

    sig = db_session.get(SignalRegistry, sid)
    obs = _obs(db_session)
    event = db_session.query(ForwardReturnObservationEvent).one()
    survivorship_request = json.loads(event.provider_request_json)[
        "survivorship_request"
    ]
    attempts = survivorship_request["source_attempts"]

    assert sig.forward_return_status == "corporate_action_review"
    assert obs.provider == "Benzinga"
    assert survivorship_request["authority_conflict"]["provider"] == "FMP"
    assert (
        survivorship_request["authority_conflict"]["reason"]
        == "delisting_unclassified_survivorship_review"
    )
    assert [attempt["source"] for attempt in attempts] == [
        "benzinga_calendar_ma",
        "survivorship_events",
        "fmp_delisted_companies",
    ]
    assert [attempt["status"] for attempt in attempts] == [
        "matched",
        "no_match",
        "matched",
    ]


def test_benzinga_source_error_does_not_block_survivorship_fallback(db_session):
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
    benzinga = FakeBenzingaAdapter(
        errors_by_ticker={
            "ACME": ProviderError(
                provider="Benzinga",
                endpoint="/api/v2.1/calendar/ma",
                status_code=429,
                error_type="rate_limit",
                message="rate limited",
                retryable=True,
            )
        }
    )

    _run_job(db_session, adapter, survivorship_adapters=[benzinga])

    sig = db_session.get(SignalRegistry, sid)
    event = db_session.query(ForwardReturnObservationEvent).one()
    provider_request = json.loads(event.provider_request_json)
    attempts = provider_request["survivorship_request"]["source_attempts"]

    assert sig.forward_return_status == "computed"
    assert sig.forward_return == 0.4
    assert attempts[0]["source"] == "benzinga_calendar_ma"
    assert attempts[0]["status"] == "error"
    assert attempts[0]["error"]["error_type"] == "rate_limit"
    assert attempts[1]["source"] == "survivorship_events"
    assert attempts[1]["status"] == "matched"


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


def test_negative_zero_entry_price_is_invalid(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": _bars(entry_open=-0.0, exit_open=12.0)})

    _run_job(db_session, adapter)

    sig = db_session.get(SignalRegistry, sid)
    obs = _obs(db_session)
    assert sig.forward_return_status == "invalid_entry_price_retry"
    assert sig.outcome_unavailable_reason == "invalid_entry_price"
    assert sig.forward_return is None
    assert obs.entry_price == -0.0
    assert obs.forward_return is None


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


def test_negative_zero_exit_price_is_terminal_mark(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=-0.0)})

    _run_job(db_session, adapter)

    sig = db_session.get(SignalRegistry, sid)
    obs = _obs(db_session)
    assert sig.forward_return_status == "computed"
    assert sig.forward_return == -1.0
    assert obs.exit_price == -0.0
    assert obs.forward_return == -1.0


def test_zero_volume_bars_with_valid_ohlc_remain_priceable(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(ENTRY_DATE, 10.0, volume=0),
            _bar(EXIT_DATE, 12.0, volume=0),
        ]
    })

    _run_job(db_session, adapter)

    sig = db_session.get(SignalRegistry, sid)
    obs = _obs(db_session)
    lineage = db_session.get(DataLineage, obs.entry_data_lineage_id)
    lineage_payload = json.loads(lineage.raw_payload_json)
    assert sig.forward_return_status == "computed"
    assert sig.forward_return == 0.2
    assert {bar["volume"] for bar in lineage_payload["bars"]} == {0}


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


def test_forward_return_daily_path_rows_persist(db_session):
    sid = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(
                ENTRY_DATE,
                10.0,
                high=10.5,
                low=9.8,
                close=10.2,
                volume=111,
                split_adjusted_close=10.2,
                adj_close=1.0,
            ),
            _bar(
                date(2026, 6, 1),
                11.0,
                high=12.0,
                low=10.0,
                close=11.5,
                volume=222.5,
                split_adjusted_close=11.5,
            ),
            _bar(
                EXIT_DATE,
                12.0,
                high=12.5,
                low=11.5,
                close=12.25,
                volume=333,
                split_adjusted_close=12.25,
            ),
        ]
    })

    _run_job(db_session, adapter)

    obs = _obs(db_session)
    rows = (
        db_session.query(ForwardReturnPathRow)
        .order_by(ForwardReturnPathRow.path_sequence)
        .all()
    )
    assert [row.path_sequence for row in rows] == [1, 2, 3]
    assert [row.session_date for row in rows] == [
        ENTRY_DATE.isoformat(),
        "2026-06-01",
        EXIT_DATE.isoformat(),
    ]
    assert {row.forward_return_observation_id for row in rows} == {
        obs.forward_return_observation_id
    }
    assert {row.signal_id for row in rows} == {sid}
    assert {row.data_lineage_id for row in rows} == {obs.entry_data_lineage_id}
    assert rows[0].is_entry_session is True
    assert rows[0].is_exit_session is False
    assert rows[1].volume == 222.5
    assert rows[1].return_from_entry_open == 0.1
    assert rows[1].return_from_entry_high == 0.2
    assert rows[1].return_from_entry_low == 0.0
    assert rows[1].return_from_entry_close == 0.15
    assert rows[2].is_entry_session is False
    assert rows[2].is_exit_session is True
    assert rows[2].return_from_entry_close == 0.225
    assert rows[2].input_hash == obs.input_hash
    assert rows[2].outcome_hash == obs.outcome_hash
    assert {row.expected_session_count for row in rows} == {15}
    assert {row.path_status for row in rows} == {"partial"}
    assert {row.is_synthetic_exit for row in rows} == {False}


def test_current_forward_path_rows_returns_ordered_computed_path(db_session):
    _make_signal(db_session)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(ENTRY_DATE, 10.0, close=10.0),
            _bar(date(2026, 6, 1), 11.0, close=11.5),
            _bar(EXIT_DATE, 12.0, close=12.25),
        ]
    })

    _run_job(db_session, adapter)

    obs = _obs(db_session)
    rows = current_forward_path_rows(db_session, obs)
    assert obs.status == "computed"
    assert [row.path_sequence for row in rows] == [1, 2, 3]
    assert [row.session_date for row in rows] == [
        ENTRY_DATE.isoformat(),
        "2026-06-01",
        EXIT_DATE.isoformat(),
    ]
    assert {row.outcome_hash for row in rows} == {obs.outcome_hash}


def test_current_forward_path_rows_rereads_stale_observation_object(db_session):
    _make_signal(db_session)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(ENTRY_DATE, 10.0, close=10.0),
            _bar(date(2026, 6, 1), 11.0, close=11.5),
            _bar(EXIT_DATE, 12.0, close=12.25),
        ]
    })
    _run_job(db_session, adapter)
    stale_obs = _obs(db_session)
    old_hash = stale_obs.outcome_hash

    db_session.execute(
        text(
            "UPDATE forward_return_observations "
            "SET status = :status, outcome_hash = :outcome_hash "
            "WHERE forward_return_observation_id = :observation_id"
        ),
        {
            "status": "pricing_unavailable_retry",
            "outcome_hash": "manual-new-outcome-hash",
            "observation_id": stale_obs.forward_return_observation_id,
        },
    )
    db_session.flush()

    assert stale_obs.status == "computed"
    assert stale_obs.outcome_hash == old_hash
    assert current_forward_path_rows(db_session, stale_obs) == []


def test_current_forward_path_rows_excludes_provisional_until_final(db_session):
    _make_signal(db_session)
    bars = [
        _bar(ENTRY_DATE, 10.0, close=10.0),
        _bar(date(2026, 6, 1), 11.0, close=11.5),
        _bar(EXIT_DATE, 12.0, close=12.25),
    ]
    adapter = FakeHistoricalAdapter({"ACME": bars})

    _run_job(
        db_session,
        adapter,
        run_ts=datetime(2026, 6, 15, 21, 0, tzinfo=timezone.utc),
        max_attempts=10,
    )

    obs = _obs(db_session)
    assert obs.status == "price_finality_pending"
    assert db_session.query(ForwardReturnPathRow).count() == 3
    assert current_forward_path_rows(db_session, obs) == []

    _run_job(db_session, FakeHistoricalAdapter({"ACME": bars}), max_attempts=10)

    obs = _obs(db_session)
    rows = current_forward_path_rows(db_session, obs)
    assert obs.status == "computed"
    assert [row.path_sequence for row in rows] == [1, 2, 3]
    assert {row.outcome_hash for row in rows} == {obs.outcome_hash}


def test_current_forward_path_rows_excludes_review_status(db_session):
    _make_signal(db_session)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(ENTRY_DATE, 10.0, close=10.0),
            _bar(date(2026, 6, 1), 11.0, close=11.5),
            _bar(EXIT_DATE, 12.0, close=12.25),
        ]
    })
    _run_job(db_session, adapter)

    obs = _obs(db_session)
    obs.status = "price_drift_review"
    db_session.flush()

    assert current_forward_path_rows(db_session, obs) == []


def test_current_forward_path_rows_excludes_empty_observation_hash(db_session):
    _make_signal(db_session)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(ENTRY_DATE, 10.0, close=10.0),
            _bar(date(2026, 6, 1), 11.0, close=11.5),
            _bar(EXIT_DATE, 12.0, close=12.25),
        ]
    })
    _run_job(db_session, adapter)

    obs = _obs(db_session)
    obs.outcome_hash = ""
    db_session.flush()

    assert current_forward_path_rows(db_session, obs) == []


def test_current_forward_path_rows_excludes_missing_live_observation(db_session):
    obs = ForwardReturnObservation(
        forward_return_observation_id="missing-observation",
        signal_id="missing-signal",
        pattern_id="M4",
        ticker="ACME",
        direction="long",
        signal_timestamp=SIGNAL_TS,
        signal_horizon="15d",
        next_execution_session=ENTRY_DATE.isoformat(),
        status="computed",
        input_hash="input",
        outcome_hash="outcome",
    )

    assert current_forward_path_rows(db_session, obs) == []


def test_current_forward_path_rows_isolates_observations(db_session):
    _make_signal(db_session, ticker="ACME")
    _make_signal(db_session, ticker="BETA")
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(ENTRY_DATE, 10.0, close=10.0),
            _bar(date(2026, 6, 1), 11.0, close=11.5),
            _bar(EXIT_DATE, 12.0, close=12.25),
        ],
        "BETA": [
            _bar(ENTRY_DATE, 20.0, close=20.0),
            _bar(date(2026, 6, 1), 21.0, close=21.5),
            _bar(EXIT_DATE, 22.0, close=22.25),
        ],
    })
    _run_job(db_session, adapter)

    observations = {
        obs.ticker: obs
        for obs in db_session.query(ForwardReturnObservation).all()
    }
    acme_rows = current_forward_path_rows(db_session, observations["ACME"])
    beta_rows = current_forward_path_rows(db_session, observations["BETA"])

    assert {row.ticker for row in acme_rows} == {"ACME"}
    assert {row.forward_return_observation_id for row in acme_rows} == {
        observations["ACME"].forward_return_observation_id
    }
    assert {row.ticker for row in beta_rows} == {"BETA"}
    assert {row.forward_return_observation_id for row in beta_rows} == {
        observations["BETA"].forward_return_observation_id
    }


def test_current_forward_path_rows_serves_partial_paths_for_consumers(db_session):
    _make_signal(db_session)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(ENTRY_DATE, 10.0, close=10.0),
            _bar(date(2026, 6, 1), 11.0, close=11.5),
            _bar(EXIT_DATE, 12.0, close=12.25),
        ]
    })
    _run_job(db_session, adapter)

    obs = _obs(db_session)
    rows = current_forward_path_rows(db_session, obs)

    assert len(rows) == 3
    assert {row.expected_session_count for row in rows} == {15}
    assert {row.path_status for row in rows} == {"partial"}


def test_current_forward_path_rows_serves_distinguishable_synthetic_exit(db_session):
    _make_signal(db_session)
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

    obs = _obs(db_session)
    rows = current_forward_path_rows(db_session, obs)
    synthetic_rows = [row for row in rows if row.is_synthetic_exit is True]

    assert len(rows) == 2
    assert len(synthetic_rows) == 1
    assert synthetic_rows[0].is_exit_session is True
    assert synthetic_rows[0].close_price == obs.exit_price == 14.0
    assert synthetic_rows[0].open_price is None
    assert synthetic_rows[0].high_price is None
    assert synthetic_rows[0].low_price is None


def test_current_forward_path_rows_does_not_autoflush_pending_mutation(db_session):
    _make_signal(db_session)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(ENTRY_DATE, 10.0, close=10.0),
            _bar(date(2026, 6, 1), 11.0, close=11.5),
            _bar(EXIT_DATE, 12.0, close=12.25),
        ]
    })
    _run_job(db_session, adapter)

    obs = _obs(db_session)
    assert obs.status == "computed"

    # Introduce an unrelated pending dirty mutation on a persistent object
    # without flushing it. A non-read-only reader would autoflush this when it
    # issues its own session.query(...), persisting the mutation mid-read.
    obs.reason = "AUTOFLUSH-PROBE-UNFLUSHED"
    assert obs in db_session.dirty

    rows = current_forward_path_rows(db_session, obs)
    assert [row.path_sequence for row in rows] == [1, 2, 3]

    # The reader must leave the pending mutation un-flushed: the ORM object is
    # still dirty, and a raw in-transaction SELECT does not see the new value.
    assert obs in db_session.dirty
    persisted_reason = db_session.execute(
        text(
            "SELECT reason FROM forward_return_observations "
            "WHERE forward_return_observation_id = :observation_id"
        ),
        {"observation_id": obs.forward_return_observation_id},
    ).scalar_one()
    assert persisted_reason != "AUTOFLUSH-PROBE-UNFLUSHED"


def test_current_forward_path_rows_trusts_writer_for_corrupt_rows(db_session):
    signal_id = _make_signal(db_session)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(ENTRY_DATE, 10.0, close=10.0),
            _bar(date(2026, 6, 1), 11.0, close=11.5),
            _bar(EXIT_DATE, 12.0, close=12.25),
        ]
    })
    _run_job(db_session, adapter)

    obs = _obs(db_session)
    assert obs.status == "computed"

    # Seed a row that matches the reader's only filter keys
    # (forward_return_observation_id + outcome_hash) but is otherwise corrupt:
    # wrong ticker, wrong input_hash, and an out-of-window session_date. The
    # signal_id is valid because the FK requires it. This documents that the
    # reader trusts the writer for ticker/input_hash/date-window bounds.
    db_session.add(
        ForwardReturnPathRow(
            forward_return_observation_id=obs.forward_return_observation_id,
            signal_id=signal_id,
            pattern_id="M4",
            ticker="WRONG-TICKER",
            path_sequence=99,
            session_date="1999-01-01",
            outcome_hash=obs.outcome_hash,
            input_hash="WRONG-INPUT-HASH",
        )
    )
    db_session.flush()

    rows = current_forward_path_rows(db_session, obs)
    corrupt = [row for row in rows if row.path_sequence == 99]
    assert len(corrupt) == 1
    assert corrupt[0].ticker == "WRONG-TICKER"
    assert corrupt[0].input_hash == "WRONG-INPUT-HASH"
    assert corrupt[0].session_date == "1999-01-01"


def test_forward_return_real_path_rows_share_entry_lineage_basis(db_session):
    _make_signal(db_session)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(ENTRY_DATE, 10.0, close=10.0),
            _bar(date(2026, 6, 1), 11.0, close=11.5),
            _bar(EXIT_DATE, 12.0, close=12.25),
        ]
    })

    _run_job(db_session, adapter)

    obs = _obs(db_session)
    rows = (
        db_session.query(ForwardReturnPathRow)
        .order_by(ForwardReturnPathRow.path_sequence)
        .all()
    )
    real_rows = [row for row in rows if row.is_synthetic_exit is not True]
    assert real_rows
    assert {row.data_lineage_id for row in real_rows} == {
        obs.entry_data_lineage_id
    }


def test_forward_return_partial_path_rows_expose_expected_session_count(db_session):
    _make_signal(db_session)
    sessions = _session_window(ENTRY_DATE, EXIT_DATE)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(sessions[0], 10.0),
            _bar(sessions[1], 11.0),
            _bar(sessions[-1], 12.0),
        ]
    })

    _run_job(db_session, adapter)

    rows = (
        db_session.query(ForwardReturnPathRow)
        .order_by(ForwardReturnPathRow.path_sequence)
        .all()
    )
    assert len(sessions) > 3
    assert [row.path_sequence for row in rows] == [1, 2, 3]
    assert [row.session_date for row in rows] == [
        sessions[0].isoformat(),
        sessions[1].isoformat(),
        sessions[-1].isoformat(),
    ]
    assert {row.expected_session_count for row in rows} == {len(sessions)}
    assert {row.path_status for row in rows} == {"partial"}
    assert {row.is_synthetic_exit for row in rows} == {False}


def test_forward_return_complete_path_rows_expose_complete_status(db_session):
    _make_signal(db_session)
    sessions = _session_window(ENTRY_DATE, EXIT_DATE)
    adapter = FakeHistoricalAdapter({
        "ACME": [
            _bar(day, 10.0 + idx, close=10.0 + idx)
            for idx, day in enumerate(sessions)
        ]
    })

    _run_job(db_session, adapter)

    rows = (
        db_session.query(ForwardReturnPathRow)
        .order_by(ForwardReturnPathRow.path_sequence)
        .all()
    )
    assert len(rows) == len(sessions)
    assert [row.path_sequence for row in rows] == list(range(1, len(sessions) + 1))
    assert {row.expected_session_count for row in rows} == {len(sessions)}
    assert {row.path_status for row in rows} == {"complete"}
    assert {row.is_synthetic_exit for row in rows} == {False}


def test_forward_return_survivorship_exit_adds_synthetic_exit_path_row(db_session):
    _make_signal(db_session)
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

    obs = _obs(db_session)
    rows = (
        db_session.query(ForwardReturnPathRow)
        .order_by(ForwardReturnPathRow.path_sequence)
        .all()
    )
    synthetic = [row for row in rows if row.is_synthetic_exit is True]
    assert len(synthetic) == 1
    assert synthetic[0].is_exit_session is True
    assert synthetic[0].session_date == EXIT_DATE.isoformat()
    assert synthetic[0].path_sequence == max(row.path_sequence for row in rows)
    assert synthetic[0].open_price is None
    assert synthetic[0].high_price is None
    assert synthetic[0].low_price is None
    assert synthetic[0].close_price == obs.exit_price == 14.0
    assert synthetic[0].return_from_entry_close == 0.4
    assert synthetic[0].provider == "TEST_SURVIVORSHIP"
    assert synthetic[0].endpoint == "/test/survivorship-events"
    assert {row.expected_session_count for row in rows} == {
        len(_session_window(ENTRY_DATE, EXIT_DATE))
    }
    assert {row.path_status for row in rows} == {"partial"}


def test_forward_return_daily_path_rows_absent_before_entry_price(db_session):
    _make_signal(db_session)
    adapter = FakeHistoricalAdapter({
        "ACME": [_bar(EXIT_DATE, 12.0)],
    })

    _run_job(db_session, adapter)

    assert db_session.query(ForwardReturnPathRow).count() == 0
    assert _obs(db_session).status == "missing_entry_price_retry"


def test_forward_return_daily_path_rows_replaced_but_not_wiped_by_pathless_reprice(db_session):
    _make_signal(db_session)
    pre_finality_ts = datetime(2026, 6, 15, 21, 0, tzinfo=timezone.utc)

    initial = FakeHistoricalAdapter({
        "ACME": [
            _bar(ENTRY_DATE, 10.0, close=10.0),
            _bar(date(2026, 6, 1), 11.0, close=11.5),
            _bar(EXIT_DATE, 12.0, close=12.25),
        ]
    })
    _run_job(db_session, initial, run_ts=pre_finality_ts, max_attempts=10)
    first_rows = (
        db_session.query(ForwardReturnPathRow)
        .order_by(ForwardReturnPathRow.path_sequence)
        .all()
    )
    assert [row.path_sequence for row in first_rows] == [1, 2, 3]
    assert [row.session_date for row in first_rows] == [
        ENTRY_DATE.isoformat(),
        "2026-06-01",
        EXIT_DATE.isoformat(),
    ]
    first_ids = {row.forward_return_path_row_id for row in first_rows}

    replacement = FakeHistoricalAdapter({
        "ACME": [
            _bar(ENTRY_DATE, 10.0, close=10.0),
            _bar(date(2026, 6, 1), 13.0, close=13.5),
            _bar(EXIT_DATE, 14.0, close=14.25),
        ]
    })
    _run_job(db_session, replacement, run_ts=pre_finality_ts, max_attempts=10)
    replacement_rows = (
        db_session.query(ForwardReturnPathRow)
        .order_by(ForwardReturnPathRow.path_sequence)
        .all()
    )
    assert [row.path_sequence for row in replacement_rows] == [1, 2, 3]
    assert [row.session_date for row in replacement_rows] == [
        ENTRY_DATE.isoformat(),
        "2026-06-01",
        EXIT_DATE.isoformat(),
    ]
    assert len({row.session_date for row in replacement_rows}) == 3
    assert {row.forward_return_path_row_id for row in replacement_rows}.isdisjoint(first_ids)
    assert replacement_rows[1].open_price == 13.0
    assert replacement_rows[1].return_from_entry_close == 0.35
    preserved_snapshot = [
        (
            row.path_sequence,
            row.session_date,
            row.open_price,
            row.close_price,
            row.return_from_entry_close,
        )
        for row in replacement_rows
    ]

    pathless = FakeHistoricalAdapter(
        errors_by_ticker={
            "ACME": ProviderError(
                provider="FMP",
                endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                status_code=503,
                error_type="http",
                message="temporary outage",
                retryable=True,
            )
        }
    )
    _run_job(db_session, pathless, run_ts=pre_finality_ts, max_attempts=10)

    after_pathless_rows = (
        db_session.query(ForwardReturnPathRow)
        .order_by(ForwardReturnPathRow.path_sequence)
        .all()
    )
    assert [
        (
            row.path_sequence,
            row.session_date,
            row.open_price,
            row.close_price,
            row.return_from_entry_close,
        )
        for row in after_pathless_rows
    ] == preserved_snapshot
    obs = _obs(db_session)
    assert obs.status == "pricing_unavailable_retry"
    assert {row.outcome_hash for row in after_pathless_rows} != {obs.outcome_hash}
    assert current_forward_path_rows(db_session, obs) == []


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


def test_reconcile_computed_mode_appends_pass_event_and_keeps_return(db_session):
    sid = _make_signal(db_session)
    initial = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=12.0)})
    revision = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=12.0)})

    _run_job(db_session, initial)
    first_obs = _obs(db_session)
    first_obs_id = first_obs.forward_return_observation_id
    first_outcome_hash = first_obs.outcome_hash
    result = _run_job(
        db_session,
        revision,
        run_ts=datetime(2026, 6, 17, 21, 0, tzinfo=timezone.utc),
        reconcile_computed=True,
    )

    sig = db_session.get(SignalRegistry, sid)
    obs = _obs(db_session)
    assert result.metrics["mode"] == "computed_reconciliation_m4"
    assert result.metrics["total_computed"] == 1
    assert result.metrics["reconciliation_passed"] == 1
    assert result.metrics["events_appended"] == 1
    assert len(revision.calls) == 1
    assert sig.forward_return_status == "computed"
    assert sig.forward_return == 0.2
    assert obs.forward_return_observation_id == first_obs_id
    assert obs.status == "computed"
    assert obs.forward_return == 0.2
    assert obs.outcome_hash == first_outcome_hash
    assert db_session.query(ForwardReturnObservation).count() == 1
    assert db_session.query(ForwardReturnObservationEvent).count() == 2
    event = (
        db_session.query(ForwardReturnObservationEvent)
        .filter(ForwardReturnObservationEvent.reason == "reconciliation_passed")
        .one()
    )
    payload = json.loads(event.provider_request_json)
    revision_payload = payload["post_compute_reconciliation"]
    assert revision_payload["status"] == "reconciliation_passed"
    assert revision_payload["material_drift"] is False
    assert revision_payload["original"]["exit_open"] == 12.0
    assert revision_payload["current"]["exit_open"] == 12.0
    assert len(json.loads(event.data_lineage_ids)) == 2


def test_reconcile_computed_mode_includes_exact_revision_window_boundary(db_session):
    _make_signal(db_session)
    initial = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=12.0)})
    revision = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=12.0)})

    _run_job(db_session, initial)
    result = _run_job(
        db_session,
        revision,
        run_ts=MATURE_RUN_TS,
        reconcile_computed=True,
        revision_window_sessions=0,
    )

    assert result.metrics["total_computed"] == 1
    assert result.metrics["reconciliation_passed"] == 1
    assert result.metrics["skipped_outside_window"] == 0
    assert len(revision.calls) == 1
    event = (
        db_session.query(ForwardReturnObservationEvent)
        .filter(ForwardReturnObservationEvent.reason == "reconciliation_passed")
        .one()
    )
    revision_payload = json.loads(event.provider_request_json)[
        "post_compute_reconciliation"
    ]
    assert revision_payload["current_evidence_session_date"] == "2026-06-16"
    assert revision_payload["revision_window_end"] == "2026-06-16"


def test_reconcile_computed_mode_ignores_rows_outside_revision_window(db_session):
    _make_signal(db_session)
    initial = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=12.0)})
    revision = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=12.2)})

    _run_job(db_session, initial)
    result = _run_job(
        db_session,
        revision,
        run_ts=datetime(2026, 7, 10, 21, 0, tzinfo=timezone.utc),
        reconcile_computed=True,
    )

    assert result.metrics["total_computed"] == 1
    assert result.metrics["skipped_outside_window"] == 1
    assert result.metrics["events_appended"] == 0
    assert revision.calls == []
    assert _obs(db_session).status == "computed"
    assert db_session.query(ForwardReturnObservationEvent).count() == 1


def test_reconcile_computed_mode_can_load_original_payload_from_exit_lineage(
    db_session,
):
    _make_signal(db_session)
    initial = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=12.0)})
    revision = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=12.0)})

    _run_job(db_session, initial)
    obs = _obs(db_session)
    assert obs.exit_data_lineage_id
    obs.entry_data_lineage_id = None
    obs.data_lineage_ids = json.dumps([])
    db_session.flush()

    result = _run_job(
        db_session,
        revision,
        run_ts=datetime(2026, 6, 17, 21, 0, tzinfo=timezone.utc),
        reconcile_computed=True,
    )

    assert result.metrics["reconciliation_passed"] == 1
    assert result.metrics["events_appended"] == 1
    event = (
        db_session.query(ForwardReturnObservationEvent)
        .filter(ForwardReturnObservationEvent.reason == "reconciliation_passed")
        .one()
    )
    lineage_ids = json.loads(event.data_lineage_ids)
    assert obs.exit_data_lineage_id in lineage_ids
    assert len(lineage_ids) == 2


def test_reconcile_computed_mode_moves_material_drift_to_review(db_session):
    sid = _make_signal(db_session)
    initial = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=12.0)})
    revised = FakeHistoricalAdapter({"ACME": _bars(entry_open=10.0, exit_open=12.2)})

    _run_job(db_session, initial)
    first_obs_id = _obs(db_session).forward_return_observation_id
    result = _run_job(
        db_session,
        revised,
        run_ts=datetime(2026, 6, 17, 21, 0, tzinfo=timezone.utc),
        reconcile_computed=True,
    )

    sig = db_session.get(SignalRegistry, sid)
    obs = _obs(db_session)
    assert result.metrics["price_drift_review"] == 1
    assert result.metrics["events_appended"] == 1
    assert len(revised.calls) == 1
    assert sig.forward_return_status == "price_drift_review"
    assert sig.forward_return is None
    assert sig.outcome_unavailable_reason == "provider_price_drift_exceeds_tolerance"
    assert obs.forward_return_observation_id == first_obs_id
    assert obs.status == "price_drift_review"
    assert obs.reason == "provider_price_drift_exceeds_tolerance"
    assert obs.forward_return is None
    assert obs.entry_price == 10.0
    assert obs.exit_price == 12.0
    assert db_session.query(ForwardReturnObservation).count() == 1
    assert db_session.query(ForwardReturnObservationEvent).count() == 2
    payload = json.loads(obs.provider_request_json)
    revision_payload = payload["post_compute_reconciliation"]
    assert revision_payload["status"] == "price_drift_review"
    assert revision_payload["material_drift"] is True
    assert revision_payload["original"]["exit_open"] == 12.0
    assert revision_payload["current"]["exit_open"] == 12.2
    assert revision_payload["drift"]["material_drift_count"] >= 1
    lineage_ids = json.loads(obs.data_lineage_ids)
    assert len(lineage_ids) == 2
    assert db_session.query(DataLineage).filter(
        DataLineage.data_lineage_id.in_(lineage_ids)
    ).count() == 2


def test_postgres_schema_connect_args_sets_scratch_search_path():
    kwargs = schema_connect_args(
        "postgresql+psycopg://user:pass@example.com/db",
        "scratch_codex_m4_pricefn_audit_test",
    )

    assert kwargs == {
        "connect_args": {
            "options": (
                "-csearch_path=scratch_codex_m4_pricefn_audit_test"
            )
        }
    }

    with pytest.raises(ValueError):
        schema_connect_args(
            "postgresql+psycopg://user:pass@example.com/db",
            "bad-schema",
        )
