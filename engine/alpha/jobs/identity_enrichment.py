"""Shared Polygon security-identity enrichment for universe scans.

The job enriches FMP-built universe rows with point-in-time identity evidence.
It deliberately does not alter operating-universe membership. The primary path
is Polygon's paginated bulk ticker reference feed; per-ticker details and ticker
events are capped exception paths for misses, probes, and later survivorship
work.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy.orm import Session

from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError, stable_hash
from alpha.data.polygon import (
    TICKER_EVENTS_ENDPOINT_PREFIX,
    TICKERS_ENDPOINT,
    PolygonAdapter,
    PolygonTickerDetail,
    PolygonTickerEvent,
    PolygonTickerReference,
    PolygonTickerReferencePage,
)
from alpha.db.models import SecurityIdentitySnapshot, UniverseSnapshot
from alpha.evidence.writer import record_data_lineage
from alpha.jobs.contracts import BaseJob, JobContext, JobResult

IDENTITY_STATUS_PRESENT = "present"
IDENTITY_STATUS_NO_DATA = "no_data"
IDENTITY_STATUS_PROVIDER_ERROR = "provider_error"
IDENTITY_STATUS_IDENTITY_CONFLICT = "identity_conflict"
IDENTITY_STATUS_UNAVAILABLE = "unavailable"


class PolygonIdentityEnrichmentJob(BaseJob):
    """Bulk-first Polygon identity enrichment for a universe scan."""

    job_name = "polygon_identity_enrichment"
    job_type = "identity_enrichment"

    def __init__(
        self,
        session: Session,
        *,
        adapter: PolygonAdapter,
        scan_id: str,
        tickers: Optional[List[str]] = None,
        include_excluded: bool = False,
        max_exception_lookups: int = 25,
        ticker_event_probes: Optional[List[str]] = None,
        bulk_limit: int = 1000,
        bulk_market: str = "stocks",
        bulk_locale: str = "us",
        fetch_events_for_inactive: bool = False,
        asof: Optional[datetime] = None,
    ):
        if max_exception_lookups < 0:
            raise ValueError("max_exception_lookups must be >= 0")
        if bulk_limit < 1:
            raise ValueError("bulk_limit must be >= 1")
        self._session = session
        self._adapter = adapter
        self._scan_id = scan_id
        self._tickers = _dedupe_tickers(tickers or [])
        self._include_excluded = include_excluded
        self._max_exception_lookups = max_exception_lookups
        self._ticker_event_probes = set(_dedupe_tickers(ticker_event_probes or []))
        self._bulk_limit = min(bulk_limit, 1000)
        self._bulk_market = bulk_market
        self._bulk_locale = bulk_locale
        self._fetch_events_for_inactive = fetch_events_for_inactive
        self._asof = asof

    def run(self, ctx: JobContext) -> JobResult:
        """Fetch bulk identity once, join locally, then persist snapshots."""

        tickers = self._tickers or self._load_scan_tickers()
        bulk_resp = self._adapter.get_tickers(
            market=self._bulk_market,
            locale=self._bulk_locale,
            limit=self._bulk_limit,
            asof=self._asof or ctx.started_at,
        )
        bulk_pages_fetched = 0
        polygon_api_call_count = 0
        bulk_lineage_by_ticker: Dict[str, str] = {}
        bulk_records: Dict[str, PolygonTickerReference] = {}
        bulk_conflicts: set[str] = set()
        fetch_errors: List[Dict[str, Any]] = []

        if not bulk_resp.ok:
            polygon_api_call_count = 1
            lineage_id = _record_response_lineage(
                self._session,
                resp=bulk_resp,
                raw_payload={
                    "request": {
                        "endpoint": TICKERS_ENDPOINT,
                        "market": self._bulk_market,
                        "locale": self._bulk_locale,
                        "limit": self._bulk_limit,
                    },
                    "error": _provider_error_payload(bulk_resp.error),
                },
                job_run_id=ctx.job_run_id,
            )
            for ticker in tickers:
                self._persist_snapshot(
                    ticker=ticker,
                    status=IDENTITY_STATUS_PROVIDER_ERROR,
                    reason=f"bulk_{getattr(bulk_resp.error, 'error_type', 'error')}",
                    source_endpoint=TICKERS_ENDPOINT,
                    lineage_ids=[lineage_id],
                    raw_payload_hash=bulk_resp.lineage.raw_payload_hash or None,
                    asof_timestamp=bulk_resp.lineage.asof_timestamp,
                    job_run_id=ctx.job_run_id,
                )
            self._session.flush()
            return JobResult(
                status="finished",
                metrics=self._metrics(
                    attempted=len(tickers),
                    present=0,
                    no_data=0,
                    errors=len(tickers),
                    exception_count=0,
                    event_attempted=0,
                    event_present=0,
                    cik_count=0,
                    composite_figi_count=0,
                    share_class_figi_count=0,
                    bulk_pages_fetched=0,
                    polygon_api_call_count=polygon_api_call_count,
                    fetch_errors=[{
                        "stage": "bulk_tickers",
                        "error_type": getattr(bulk_resp.error, "error_type", None),
                    }],
                ),
            )

        pages = list(bulk_resp.data or [])
        for page in pages:
            bulk_pages_fetched += 1
            polygon_api_call_count += 1
            page_lineage_id = _record_bulk_page_lineage(
                self._session,
                page=page,
                job_run_id=ctx.job_run_id,
            )
            for record in page.results:
                key = _ticker_key(record.ticker)
                if not key:
                    continue
                if key in bulk_records and _identity_hash_for_record(
                    bulk_records[key],
                    status=IDENTITY_STATUS_PRESENT,
                    reason=None,
                    events=[],
                    requested_ticker=bulk_records[key].ticker,
                ) != _identity_hash_for_record(
                    record,
                    status=IDENTITY_STATUS_PRESENT,
                    reason=None,
                    events=[],
                    requested_ticker=record.ticker,
                ):
                    bulk_conflicts.add(key)
                    continue
                bulk_records[key] = record
                bulk_lineage_by_ticker[key] = page_lineage_id

        attempted = len(tickers)
        present = 0
        no_data = 0
        errors = 0
        exception_count = 0
        event_attempted = 0
        event_present = 0
        cik_count = 0
        composite_figi_count = 0
        share_class_figi_count = 0

        for ticker in tickers:
            key = _ticker_key(ticker)
            lineage_ids: List[str] = []
            record: Optional[PolygonTickerReference | PolygonTickerDetail] = None
            source_endpoint = TICKERS_ENDPOINT
            raw_payload_hash = None
            asof_timestamp = bulk_resp.lineage.asof_timestamp
            status = IDENTITY_STATUS_NO_DATA
            reason = "polygon_bulk_reference_no_match"

            if key in bulk_conflicts:
                status = IDENTITY_STATUS_IDENTITY_CONFLICT
                reason = "polygon_bulk_duplicate_ticker_identity_conflict"
                lineage_id = bulk_lineage_by_ticker.get(key)
                if lineage_id:
                    lineage_ids.append(lineage_id)
            elif key in bulk_records:
                record = bulk_records[key]
                status = IDENTITY_STATUS_PRESENT
                reason = None
                raw_payload_hash = record.raw and stable_hash(record.raw)
                lineage_id = bulk_lineage_by_ticker.get(key)
                if lineage_id:
                    lineage_ids.append(lineage_id)
            elif exception_count < self._max_exception_lookups:
                exception_count += 1
                polygon_api_call_count += 1
                detail_resp = self._adapter.get_ticker_details(
                    ticker,
                    asof=self._asof or ctx.started_at,
                )
                detail_lineage_id = _record_response_lineage(
                    self._session,
                    resp=detail_resp,
                    raw_payload={
                        "request": {
                            "endpoint": f"{TICKERS_ENDPOINT}/{ticker}",
                            "ticker": ticker,
                            "exception_lookup": True,
                        },
                        "detail": _jsonable_identity(detail_resp.data),
                        "error": _provider_error_payload(detail_resp.error),
                    },
                    job_run_id=ctx.job_run_id,
                )
                lineage_ids.append(detail_lineage_id)
                source_endpoint = f"{TICKERS_ENDPOINT}/{ticker}"
                raw_payload_hash = detail_resp.lineage.raw_payload_hash or None
                asof_timestamp = detail_resp.lineage.asof_timestamp
                if not detail_resp.ok:
                    status = IDENTITY_STATUS_PROVIDER_ERROR
                    reason = f"detail_{getattr(detail_resp.error, 'error_type', 'error')}"
                    fetch_errors.append({
                        "ticker": ticker,
                        "stage": "detail_exception_lookup",
                        "error_type": getattr(detail_resp.error, "error_type", None),
                    })
                elif detail_resp.data is None:
                    status = IDENTITY_STATUS_NO_DATA
                    reason = "polygon_bulk_and_detail_no_data"
                else:
                    record = detail_resp.data
                    status = IDENTITY_STATUS_PRESENT
                    reason = "polygon_detail_exception_lookup"
            else:
                status = IDENTITY_STATUS_UNAVAILABLE
                reason = "polygon_bulk_missing_exception_cap_exhausted"

            events, events_lineage_id = [], None
            if self._should_fetch_events(ticker, record=record, status=status):
                event_attempted += 1
                polygon_api_call_count += 1
                event_resp = self._adapter.get_ticker_events(
                    ticker,
                    asof=self._asof or ctx.started_at,
                )
                events_lineage_id = _record_response_lineage(
                    self._session,
                    resp=event_resp,
                    raw_payload={
                        "request": {
                            "endpoint": f"{TICKER_EVENTS_ENDPOINT_PREFIX}/{ticker}/events",
                            "identifier": ticker,
                            "types": "ticker_change",
                        },
                        "events": _jsonable_events(event_resp.data),
                        "error": _provider_error_payload(event_resp.error),
                    },
                    job_run_id=ctx.job_run_id,
                )
                lineage_ids.append(events_lineage_id)
                if event_resp.ok and event_resp.data:
                    events = list(event_resp.data)
                    event_present += len(events)
                elif not event_resp.ok:
                    fetch_errors.append({
                        "ticker": ticker,
                        "stage": "ticker_events",
                        "error_type": getattr(event_resp.error, "error_type", None),
                    })

            self._persist_snapshot(
                ticker=ticker,
                status=status,
                reason=reason,
                source_endpoint=source_endpoint,
                record=record,
                events=events,
                events_lineage_id=events_lineage_id,
                lineage_ids=lineage_ids,
                raw_payload_hash=raw_payload_hash,
                asof_timestamp=asof_timestamp,
                job_run_id=ctx.job_run_id,
            )

            if status == IDENTITY_STATUS_PRESENT:
                present += 1
                if _record_cik(record):
                    cik_count += 1
                if _record_composite_figi(record):
                    composite_figi_count += 1
                if _record_share_class_figi(record):
                    share_class_figi_count += 1
            elif status == IDENTITY_STATUS_NO_DATA:
                no_data += 1
            else:
                errors += 1

        self._session.flush()
        return JobResult(
            status="finished",
            metrics=self._metrics(
                attempted=attempted,
                present=present,
                no_data=no_data,
                errors=errors,
                exception_count=exception_count,
                event_attempted=event_attempted,
                event_present=event_present,
                cik_count=cik_count,
                composite_figi_count=composite_figi_count,
                share_class_figi_count=share_class_figi_count,
                bulk_pages_fetched=bulk_pages_fetched,
                polygon_api_call_count=polygon_api_call_count,
                fetch_errors=fetch_errors,
            ),
        )

    def _load_scan_tickers(self) -> List[str]:
        query = self._session.query(UniverseSnapshot).filter(
            UniverseSnapshot.scan_id == self._scan_id
        )
        if not self._include_excluded:
            query = query.filter(UniverseSnapshot.operating_universe_inclusion.is_(True))
        return _dedupe_tickers(row.ticker for row in query.order_by(UniverseSnapshot.ticker))

    def _should_fetch_events(
        self,
        ticker: str,
        *,
        record: Optional[PolygonTickerReference | PolygonTickerDetail],
        status: str,
    ) -> bool:
        key = _ticker_key(ticker)
        if key in self._ticker_event_probes:
            return True
        if self._fetch_events_for_inactive and record is not None:
            return _record_active(record) is False or bool(_record_delisted_utc(record))
        return False

    def _persist_snapshot(
        self,
        *,
        ticker: str,
        status: str,
        reason: Optional[str],
        source_endpoint: str,
        job_run_id: Optional[str],
        record: Optional[PolygonTickerReference | PolygonTickerDetail] = None,
        events: Optional[List[PolygonTickerEvent]] = None,
        events_lineage_id: Optional[str] = None,
        lineage_ids: Optional[List[str]] = None,
        raw_payload_hash: Optional[str] = None,
        asof_timestamp: Optional[datetime] = None,
    ) -> SecurityIdentitySnapshot:
        events = events or []
        lineage_ids = _dedupe_strings(lineage_ids or [])
        identity_hash = _identity_hash_for_record(
            record,
            status=status,
            reason=reason,
            events=events,
            requested_ticker=ticker,
        )
        row = (
            self._session.query(SecurityIdentitySnapshot)
            .filter(
                SecurityIdentitySnapshot.scan_id == self._scan_id,
                SecurityIdentitySnapshot.ticker == ticker,
            )
            .first()
        )
        if row is None:
            row = SecurityIdentitySnapshot(
                scan_id=self._scan_id,
                ticker=ticker,
                identity_status=status,
            )
            self._session.add(row)

        row.job_run_id = job_run_id
        row.cik = _record_cik(record)
        row.composite_figi = _record_composite_figi(record)
        row.share_class_figi = _record_share_class_figi(record)
        row.active = _record_active(record)
        row.delisted_utc = _record_delisted_utc(record)
        row.list_date = _record_list_date(record)
        row.polygon_type = _record_type(record)
        row.polygon_market = _record_market(record)
        row.polygon_locale = _record_locale(record)
        row.polygon_primary_exchange = _record_primary_exchange(record)
        row.polygon_name = _record_name(record)
        row.sic_code = getattr(record, "sic_code", None)
        row.sic_description = getattr(record, "sic_description", None)
        row.ticker_events_json = (
            json.dumps(_jsonable_events(events), sort_keys=True)
            if events else None
        )
        row.identity_status = status
        row.identity_reason = reason
        row.identity_hash = identity_hash
        row.source_provider = "Polygon"
        row.source_endpoint = source_endpoint
        row.data_lineage_id = lineage_ids[0] if lineage_ids else None
        row.events_data_lineage_id = events_lineage_id
        row.data_lineage_ids = json.dumps(lineage_ids)
        row.raw_payload_hash = raw_payload_hash
        row.asof_timestamp = asof_timestamp
        return row

    def _metrics(
        self,
        *,
        attempted: int,
        present: int,
        no_data: int,
        errors: int,
        exception_count: int,
        event_attempted: int,
        event_present: int,
        cik_count: int,
        composite_figi_count: int,
        share_class_figi_count: int,
        bulk_pages_fetched: int,
        polygon_api_call_count: int,
        fetch_errors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        coverage_ratio = present / attempted if attempted else 0.0
        return {
            "scan_id": self._scan_id,
            "identity_attempted_count": attempted,
            "identity_present_count": present,
            "identity_no_data_count": no_data,
            "identity_error_count": errors,
            "identity_exception_lookup_count": exception_count,
            "ticker_event_attempted_count": event_attempted,
            "ticker_event_present_count": event_present,
            "cik_present_count": cik_count,
            "composite_figi_present_count": composite_figi_count,
            "share_class_figi_present_count": share_class_figi_count,
            "identity_coverage_ratio": round(coverage_ratio, 4),
            "bulk_pages_fetched": bulk_pages_fetched,
            "polygon_api_call_count": polygon_api_call_count,
            "fetch_error_count": len(fetch_errors),
            "fetch_errors": fetch_errors[:50],
            # Backward-compatible metric aliases for existing run_universe output.
            "polygon_identity_attempted_count": attempted,
            "polygon_identity_present_count": present,
            "polygon_identity_missing_count": no_data,
            "polygon_identity_error_count": errors,
            "polygon_ticker_event_present_count": event_present,
            "polygon_identity_coverage_ratio": round(coverage_ratio, 4),
            "polygon_cik_present_count": cik_count,
            "polygon_composite_figi_present_count": composite_figi_count,
            "polygon_share_class_figi_present_count": share_class_figi_count,
        }


def _record_bulk_page_lineage(
    session: Session,
    *,
    page: PolygonTickerReferencePage,
    job_run_id: Optional[str],
) -> str:
    lineage = record_data_lineage(
        session,
        provider=page.lineage.provider,
        endpoint=page.lineage.endpoint,
        asof_timestamp=page.lineage.asof_timestamp,
        raw_payload={
            "request": {
                "endpoint": TICKERS_ENDPOINT,
                "page_number": page.page_number,
                "request_params": page.request_params,
                "next_url_path": page.next_url,
            },
            "results": [record.raw for record in page.results],
        },
        raw_payload_hash=page.lineage.raw_payload_hash,
        request_timestamp=page.lineage.request_timestamp,
        freshness_seconds=page.lineage.freshness_seconds,
        source_authority=page.lineage.source_authority,
        data_quality_flags={
            "source": "polygon_bulk_tickers",
            "page_number": page.page_number,
        },
        job_run_id=job_run_id,
    )
    return lineage.data_lineage_id


def _record_response_lineage(
    session: Session,
    *,
    resp: AdapterResponse[Any],
    raw_payload: Dict[str, Any],
    job_run_id: Optional[str],
) -> str:
    lineage = record_data_lineage(
        session,
        provider=resp.lineage.provider,
        endpoint=resp.lineage.endpoint,
        asof_timestamp=resp.lineage.asof_timestamp,
        raw_payload=raw_payload,
        raw_payload_hash=resp.lineage.raw_payload_hash or stable_hash(raw_payload),
        request_timestamp=resp.lineage.request_timestamp,
        freshness_seconds=resp.lineage.freshness_seconds,
        source_authority=resp.lineage.source_authority,
        data_quality_flags=resp.lineage.data_quality_flags,
        job_run_id=job_run_id,
    )
    return lineage.data_lineage_id


def _identity_hash_for_record(
    record: Optional[PolygonTickerReference | PolygonTickerDetail],
    *,
    status: str,
    reason: Optional[str],
    events: List[PolygonTickerEvent],
    requested_ticker: Optional[str],
) -> str:
    return stable_hash({
        "layer": "security_identity",
        "provider": "Polygon",
        "requested_ticker": _ticker_key(requested_ticker),
        "ticker": _record_ticker(record),
        "cik": _record_cik(record),
        "composite_figi": _record_composite_figi(record),
        "share_class_figi": _record_share_class_figi(record),
        "active": _record_active(record),
        "delisted_utc": _record_delisted_utc(record),
        "list_date": _record_list_date(record),
        "polygon_type": _record_type(record),
        "polygon_market": _record_market(record),
        "polygon_locale": _record_locale(record),
        "polygon_primary_exchange": _record_primary_exchange(record),
        "polygon_name": _record_name(record),
        "status": status,
        "reason": reason,
        "events": _jsonable_events(events),
    })


def _jsonable_identity(
    record: Optional[PolygonTickerReference | PolygonTickerDetail],
) -> Optional[Dict[str, Any]]:
    if record is None:
        return None
    raw = getattr(record, "raw", None)
    if isinstance(raw, dict):
        return dict(raw)
    return dict(record.__dict__)


def _jsonable_events(events: Optional[List[PolygonTickerEvent]]) -> List[Dict[str, Any]]:
    out = []
    for event in events or []:
        out.append({
            "identifier_queried": event.identifier_queried,
            "event_type": event.event_type,
            "date": event.date,
            "event_date": event.event_date,
            "effective_date": event.effective_date,
            "ticker": event.ticker,
            "old_ticker": event.old_ticker,
            "new_ticker": event.new_ticker,
            "cik": event.cik,
            "composite_figi": event.composite_figi,
            "share_class_figi": event.share_class_figi,
            "name": event.name,
            "raw_event": event.raw_event,
        })
    return out


def _provider_error_payload(error: Optional[ProviderError]) -> Optional[Dict[str, Any]]:
    if error is None:
        return None
    return {
        "provider": error.provider,
        "endpoint": error.endpoint,
        "status_code": error.status_code,
        "error_type": error.error_type,
        "message": error.message,
        "retryable": error.retryable,
    }


def _dedupe_tickers(values: Iterable[str]) -> List[str]:
    return _dedupe_strings(_ticker_key(value) for value in values if _ticker_key(value))


def _dedupe_strings(values: Iterable[Optional[str]]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _ticker_key(value: Any) -> str:
    return str(value or "").strip().upper()


def _record_ticker(record: Optional[Any]) -> Optional[str]:
    return getattr(record, "ticker", None) if record is not None else None


def _record_cik(record: Optional[Any]) -> Optional[str]:
    return getattr(record, "cik", None) if record is not None else None


def _record_composite_figi(record: Optional[Any]) -> Optional[str]:
    return getattr(record, "composite_figi", None) if record is not None else None


def _record_share_class_figi(record: Optional[Any]) -> Optional[str]:
    return getattr(record, "share_class_figi", None) if record is not None else None


def _record_active(record: Optional[Any]) -> Optional[bool]:
    return getattr(record, "active", None) if record is not None else None


def _record_delisted_utc(record: Optional[Any]) -> Optional[str]:
    return getattr(record, "delisted_utc", None) if record is not None else None


def _record_list_date(record: Optional[Any]) -> Optional[str]:
    return getattr(record, "list_date", None) if record is not None else None


def _record_type(record: Optional[Any]) -> Optional[str]:
    return getattr(record, "type", None) if record is not None else None


def _record_market(record: Optional[Any]) -> Optional[str]:
    return getattr(record, "market", None) if record is not None else None


def _record_locale(record: Optional[Any]) -> Optional[str]:
    return getattr(record, "locale", None) if record is not None else None


def _record_primary_exchange(record: Optional[Any]) -> Optional[str]:
    return getattr(record, "primary_exchange", None) if record is not None else None


def _record_name(record: Optional[Any]) -> Optional[str]:
    return getattr(record, "name", None) if record is not None else None
