"""Polygon security identity enrichment tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, List

from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError, stable_hash
from alpha.data.polygon import (
    PolygonTickerDetail,
    PolygonTickerEvent,
    PolygonTickerReference,
    PolygonTickerReferencePage,
)
from alpha.db.models import (
    DataLineage,
    SecurityIdentitySnapshot,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.jobs.identity_enrichment import (
    IDENTITY_STATUS_NO_DATA,
    IDENTITY_STATUS_PRESENT,
    IDENTITY_STATUS_PROVIDER_ERROR,
    IDENTITY_STATUS_UNAVAILABLE,
    PolygonIdentityEnrichmentJob,
)
from alpha.jobs.runner import run_job
from alpha.jobs.run_universe import _identity_strict_error
from alpha.patterns.contracts import PatternInput
from alpha.patterns.m4 import M4Detector


def _ts():
    return datetime(2026, 5, 27, 14, 0, 0, tzinfo=timezone.utc)


def _lineage(endpoint="/v3/reference/tickers"):
    return LineageMeta(
        provider="Polygon",
        endpoint=endpoint,
        request_timestamp=_ts(),
        asof_timestamp=_ts(),
        raw_payload_hash=stable_hash(endpoint),
        source_authority="Polygon",
    )


def _ref(ticker: str, *, cik: str | None = None) -> PolygonTickerReference:
    return PolygonTickerReference(
        ticker=ticker,
        name=f"{ticker} Corp",
        market="stocks",
        locale="us",
        primary_exchange="XNAS",
        type="CS",
        active=True,
        cik=cik,
        composite_figi=f"BBG{ticker:0<9}"[:12],
        share_class_figi=f"BBG1{ticker:0<8}"[:12],
        list_date="2020-01-01",
        raw={"ticker": ticker, "cik": cik, "composite_figi": f"BBG{ticker:0<9}"[:12]},
    )


def _page(rows: List[PolygonTickerReference], page_number: int = 0):
    return PolygonTickerReferencePage(
        results=rows,
        lineage=_lineage(),
        request_params={"limit": 1000, "page": page_number},
        page_number=page_number,
        raw_payload={"results": [row.raw for row in rows]},
    )


class FakePolygonAdapter:
    """Test adapter exposing bulk, detail, and event call counts."""

    def __init__(
        self,
        *,
        pages=None,
        details=None,
        events=None,
        bulk_error: ProviderError | None = None,
    ):
        self.pages = pages if pages is not None else []
        self.details = details or {}
        self.events = events or {}
        self.bulk_error = bulk_error
        self.get_tickers_calls = 0
        self.detail_calls = []
        self.event_calls = []

    def get_tickers(self, **kwargs):
        self.get_tickers_calls += 1
        if self.bulk_error is not None:
            return AdapterResponse(
                data=None,
                lineage=_lineage(),
                error=self.bulk_error,
            )
        return AdapterResponse(data=self.pages, lineage=_lineage())

    def get_ticker_details(self, ticker, **kwargs):
        self.detail_calls.append(ticker)
        detail = self.details.get(ticker)
        if isinstance(detail, ProviderError):
            return AdapterResponse(
                data=None,
                lineage=_lineage(f"/v3/reference/tickers/{ticker}"),
                error=detail,
            )
        return AdapterResponse(
            data=detail,
            lineage=_lineage(f"/v3/reference/tickers/{ticker}"),
        )

    def get_ticker_events(self, identifier, **kwargs):
        self.event_calls.append(identifier)
        return AdapterResponse(
            data=self.events.get(identifier, []),
            lineage=_lineage(f"/vX/reference/tickers/{identifier}/events"),
        )


def _setup_universe(db_session, tickers):
    scan = UniverseScan(
        scan_id="id-scan",
        trading_date="2026-05-27",
        asof_timestamp=_ts(),
        raw_count=len(tickers),
        deduped_count=len(tickers),
        included_count=len(tickers),
        excluded_count=0,
        run_status="finished",
        source_lineage_hash="hash",
    )
    db_session.add(scan)
    db_session.flush()
    for t in tickers:
        db_session.add(UniverseSnapshot(
            universe_snapshot_id=f"snap-{t}",
            scan_id="id-scan",
            ticker=t,
            asof_timestamp=_ts(),
            market_cap=75_000_000,
            price=5.0,
            primary_exchange="NASDAQ",
            security_type="common_stock",
            operating_universe_inclusion=True,
            source_lineage_hash="hash",
        ))
    db_session.flush()


class TestIdentityEnrichment:
    def test_bulk_first_for_full_universe_without_per_ticker_detail_calls(self, db_session):
        tickers = [f"T{i:03d}" for i in range(671)]
        _setup_universe(db_session, tickers)
        adapter = FakePolygonAdapter(pages=[_page([_ref(t, cik=str(i)) for i, t in enumerate(tickers)])])

        result = run_job(
            db_session,
            PolygonIdentityEnrichmentJob(
                session=db_session,
                adapter=adapter,
                scan_id="id-scan",
                tickers=tickers,
                max_exception_lookups=25,
            ),
        )

        assert result.ok
        assert adapter.get_tickers_calls == 1
        assert adapter.detail_calls == []
        assert adapter.event_calls == []
        assert result.metrics["identity_attempted_count"] == 671
        assert result.metrics["identity_present_count"] == 671
        assert result.metrics["bulk_pages_fetched"] == 1
        assert result.metrics["polygon_api_call_count"] == 1

    def test_exception_lookup_cap_is_enforced(self, db_session):
        tickers = ["A", "B", "C", "D"]
        _setup_universe(db_session, tickers)
        adapter = FakePolygonAdapter(
            pages=[_page([])],
            details={
                "A": PolygonTickerDetail(ticker="A", name="A Corp", cik="1"),
                "B": None,
            },
        )

        result = run_job(
            db_session,
            PolygonIdentityEnrichmentJob(
                session=db_session,
                adapter=adapter,
                scan_id="id-scan",
                tickers=tickers,
                max_exception_lookups=2,
            ),
        )

        assert result.ok
        assert adapter.detail_calls == ["A", "B"]
        assert result.metrics["identity_exception_lookup_count"] == 2
        assert result.metrics["identity_present_count"] == 1
        unavailable = db_session.query(SecurityIdentitySnapshot).filter(
            SecurityIdentitySnapshot.identity_status == IDENTITY_STATUS_UNAVAILABLE
        ).count()
        assert unavailable == 2

    def test_ticker_events_are_not_called_for_all_symbols_by_default(self, db_session):
        _setup_universe(db_session, ["META"])
        adapter = FakePolygonAdapter(pages=[_page([_ref("META", cik="1326801")])])

        result = run_job(
            db_session,
            PolygonIdentityEnrichmentJob(
                session=db_session,
                adapter=adapter,
                scan_id="id-scan",
                tickers=["META"],
            ),
        )

        assert result.ok
        assert adapter.event_calls == []
        assert result.metrics["ticker_event_attempted_count"] == 0

    def test_targeted_ticker_event_probe_persists_events(self, db_session):
        _setup_universe(db_session, ["META"])
        adapter = FakePolygonAdapter(
            pages=[_page([_ref("META", cik="1326801")])],
            events={
                "META": [
                    PolygonTickerEvent(
                        identifier_queried="META",
                        event_type="ticker_change",
                        date="2022-06-09",
                        old_ticker="FB",
                    )
                ]
            },
        )

        result = run_job(
            db_session,
            PolygonIdentityEnrichmentJob(
                session=db_session,
                adapter=adapter,
                scan_id="id-scan",
                tickers=["META"],
                ticker_event_probes=["META"],
            ),
        )

        assert result.ok
        assert adapter.event_calls == ["META"]
        assert result.metrics["ticker_event_attempted_count"] == 1
        assert result.metrics["ticker_event_present_count"] == 1
        snap = db_session.query(SecurityIdentitySnapshot).one()
        assert json.loads(snap.ticker_events_json)[0]["old_ticker"] == "FB"
        assert len(json.loads(snap.data_lineage_ids)) == 2

    def test_provider_error_does_not_change_universe_inclusion(self, db_session):
        _setup_universe(db_session, ["ERR"])
        adapter = FakePolygonAdapter(
            bulk_error=ProviderError(
                provider="Polygon",
                endpoint="/v3/reference/tickers",
                status_code=429,
                error_type="rate_limit",
                message="limited",
                retryable=True,
            )
        )

        result = run_job(
            db_session,
            PolygonIdentityEnrichmentJob(
                session=db_session,
                adapter=adapter,
                scan_id="id-scan",
                tickers=["ERR"],
            ),
        )

        assert result.ok
        snap = db_session.query(SecurityIdentitySnapshot).one()
        universe = db_session.query(UniverseSnapshot).one()
        assert snap.identity_status == IDENTITY_STATUS_PROVIDER_ERROR
        assert universe.operating_universe_inclusion is True

    def test_missing_identity_remains_explicit(self, db_session):
        _setup_universe(db_session, ["MISS"])
        adapter = FakePolygonAdapter(pages=[_page([])])

        result = run_job(
            db_session,
            PolygonIdentityEnrichmentJob(
                session=db_session,
                adapter=adapter,
                scan_id="id-scan",
                tickers=["MISS"],
                max_exception_lookups=0,
            ),
        )

        assert result.ok
        snap = db_session.query(SecurityIdentitySnapshot).one()
        assert snap.identity_status == IDENTITY_STATUS_UNAVAILABLE
        assert snap.identity_reason == "polygon_bulk_missing_exception_cap_exhausted"

    def test_identity_lineage_joins(self, db_session):
        _setup_universe(db_session, ["ACME"])
        adapter = FakePolygonAdapter(pages=[_page([_ref("ACME", cik="1234567")])])
        run_job(
            db_session,
            PolygonIdentityEnrichmentJob(
                session=db_session,
                adapter=adapter,
                scan_id="id-scan",
                tickers=["ACME"],
            ),
        )

        snap = db_session.query(SecurityIdentitySnapshot).one()
        ids = json.loads(snap.data_lineage_ids)
        assert len(ids) == 1
        assert db_session.get(DataLineage, ids[0]) is not None


class TestIdentityStrictMode:
    def test_disabled_strict_mode_allows_missing_metrics(self):
        args = SimpleNamespace(
            require_identity_enrichment=False,
            min_identity_coverage=1.0,
        )

        assert _identity_strict_error(
            args,
            {},
            ["A", "B"],
            identity_result_ok=False,
        ) is None

    def test_requires_successful_identity_job(self):
        args = SimpleNamespace(
            require_identity_enrichment=True,
            min_identity_coverage=0.0,
        )

        assert _identity_strict_error(
            args,
            {"identity_attempted_count": 2, "identity_present_count": 2},
            ["A", "B"],
            identity_result_ok=False,
        ) == "identity enrichment job returned a failed status"

    def test_requires_attempting_every_included_ticker(self):
        args = SimpleNamespace(
            require_identity_enrichment=True,
            min_identity_coverage=0.0,
        )

        assert _identity_strict_error(
            args,
            {"identity_attempted_count": 1, "identity_present_count": 1},
            ["A", "B"],
            identity_result_ok=True,
        ) == "attempted 1 of 2 included tickers"

    def test_optional_coverage_floor(self):
        args = SimpleNamespace(
            require_identity_enrichment=True,
            min_identity_coverage=0.75,
        )

        assert _identity_strict_error(
            args,
            {"identity_attempted_count": 4, "identity_present_count": 2},
            ["A", "B", "C", "D"],
            identity_result_ok=True,
        ) == "coverage 0.5000 below required 0.7500"


def test_m4_features_carry_security_identity_when_present():
    detector = M4Detector()
    identity = {
        "identity_status": "present",
        "cik": "0001326801",
        "composite_figi": "BBG000MM2P62",
        "share_class_figi": "BBG001SQCQC5",
        "source_provider": "Polygon",
        "identity_hash": "hash",
    }
    result = detector.detect(PatternInput(
        ticker="META",
        asof_timestamp=_ts(),
        market_data={
            "price": 100.0,
            "high_52w": 100.0,
            "operating_universe_inclusion": True,
            "n_sessions_in_window": 252,
            "security_identity": identity,
        },
        fundamental_data={"market_cap": 100_000_000},
        lineage_hashes=["lineage"],
    ))

    assert result.features is not None
    assert result.features.features["security_identity"] == identity
