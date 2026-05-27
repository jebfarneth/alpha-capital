"""
Polygon.io adapter.

Supplemental source for:
  - Short interest / borrow / lending data for I3 (EERR 2024 loan fee signal)
  - Ticker details
  - Market data where needed

Does not write to DB. Returns AdapterResponse with LineageMeta.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from alpha.data.config import PolygonConfig
from alpha.data.contracts import (
    AdapterResponse,
    LineageMeta,
    ProviderError,
    RateLimitInfo,
    aware_utc_or_none,
    stable_hash,
    utcnow,
)

PROVIDER = "Polygon"
TICKERS_ENDPOINT = "/v3/reference/tickers"
TICKER_EVENTS_ENDPOINT_PREFIX = "/vX/reference/tickers"


# --- Response types ---

@dataclass
class PolygonShortInterest:
    """Normalized Polygon short-interest observation."""

    ticker: str
    settlement_date: str
    short_interest: Optional[int] = None
    avg_daily_volume: Optional[int] = None
    days_to_cover: Optional[float] = None


@dataclass
class PolygonTickerReference:
    """Reference identity row from Polygon's bulk all-tickers endpoint."""

    ticker: str
    name: str
    market: Optional[str] = None
    locale: Optional[str] = None
    primary_exchange: Optional[str] = None
    type: Optional[str] = None
    active: Optional[bool] = None
    cik: Optional[str] = None
    composite_figi: Optional[str] = None
    share_class_figi: Optional[str] = None
    list_date: Optional[str] = None
    delisted_utc: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class PolygonTickerReferencePage:
    """One fetched page of Polygon bulk ticker reference rows."""

    results: List[PolygonTickerReference]
    lineage: LineageMeta
    request_params: Dict[str, Any]
    page_number: int
    next_url: Optional[str] = None
    raw_payload: Optional[Dict[str, Any]] = None


@dataclass
class PolygonTickerDetail:
    """Reference metadata for one Polygon ticker detail response."""

    ticker: str
    name: str
    market: Optional[str] = None
    locale: Optional[str] = None
    primary_exchange: Optional[str] = None
    type: Optional[str] = None
    active: Optional[bool] = None
    cik: Optional[str] = None
    composite_figi: Optional[str] = None
    share_class_figi: Optional[str] = None
    list_date: Optional[str] = None
    delisted_utc: Optional[str] = None
    market_cap: Optional[float] = None
    sic_code: Optional[str] = None
    sic_description: Optional[str] = None
    share_class_shares_outstanding: Optional[int] = None
    weighted_shares_outstanding: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class PolygonTickerEvent:
    """Normalized Polygon ticker event (e.g. ticker_change)."""

    identifier_queried: str
    event_type: str
    date: Optional[str] = None
    event_date: Optional[str] = None
    effective_date: Optional[str] = None
    ticker: Optional[str] = None
    old_ticker: Optional[str] = None
    new_ticker: Optional[str] = None
    cik: Optional[str] = None
    composite_figi: Optional[str] = None
    share_class_figi: Optional[str] = None
    name: Optional[str] = None
    raw_event: Optional[Dict[str, Any]] = None


@dataclass
class PolygonBar:
    """Normalized Polygon aggregate daily bar."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    transactions: Optional[int] = None


# --- Adapter ---

class PolygonAdapter:
    """Polygon REST adapter returning typed reference/market data with lineage."""

    def __init__(
        self, config: PolygonConfig, session: Optional[requests.Session] = None
    ):
        self._config = config
        self._session = session or requests.Session()
        self._session.params = {"apiKey": config.api_key}  # type: ignore[assignment]

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        asof: Optional[datetime] = None,
        *,
        url_override: Optional[str] = None,
        lineage_endpoint: Optional[str] = None,
    ) -> AdapterResponse[Any]:
        url = url_override or f"{self._config.base_url}{endpoint}"
        lineage_endpoint = lineage_endpoint or endpoint
        request_ts = utcnow()
        if asof is None:
            asof_ts = request_ts
        else:
            asof_ts = aware_utc_or_none(asof)
            if asof_ts is None:
                return AdapterResponse(
                    data=None,
                    lineage=LineageMeta(
                        provider=PROVIDER,
                        endpoint=lineage_endpoint,
                        request_timestamp=request_ts,
                        asof_timestamp=request_ts,
                        raw_payload_hash="",
                    ),
                    error=ProviderError(
                        provider=PROVIDER,
                        endpoint=lineage_endpoint,
                        status_code=None,
                        error_type="validation",
                        message="Polygon adapter asof timestamp must be timezone-aware datetime",
                        retryable=False,
                    ),
                )

        try:
            resp = self._session.get(url, params=params or {}, timeout=30)
        except requests.exceptions.Timeout:
            return AdapterResponse(
                data=None,
                lineage=LineageMeta(
                    provider=PROVIDER,
                    endpoint=lineage_endpoint,
                    request_timestamp=request_ts,
                    asof_timestamp=asof_ts,
                    raw_payload_hash="",
                ),
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=lineage_endpoint,
                    status_code=None,
                    error_type="timeout",
                    message="Request timed out",
                    retryable=True,
                ),
            )
        except requests.exceptions.RequestException as exc:
            return AdapterResponse(
                data=None,
                lineage=LineageMeta(
                    provider=PROVIDER,
                    endpoint=lineage_endpoint,
                    request_timestamp=request_ts,
                    asof_timestamp=asof_ts,
                    raw_payload_hash="",
                ),
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=lineage_endpoint,
                    status_code=None,
                    error_type="http",
                    message=str(exc),
                    retryable=True,
                ),
            )

        payload_hash = stable_hash(resp.text)
        freshness = (utcnow() - request_ts).total_seconds()

        lineage = LineageMeta(
            provider=PROVIDER,
            endpoint=lineage_endpoint,
            request_timestamp=request_ts,
            asof_timestamp=asof_ts,
            raw_payload_hash=payload_hash,
            freshness_seconds=freshness,
            source_authority="Polygon",
        )

        if resp.status_code == 429:
            return AdapterResponse(
                data=None,
                lineage=lineage,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=lineage_endpoint,
                    status_code=429,
                    error_type="rate_limit",
                    message="Polygon rate limit exceeded",
                    retryable=True,
                ),
            )

        if resp.status_code in (401, 403):
            return AdapterResponse(
                data=None,
                lineage=lineage,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=lineage_endpoint,
                    status_code=resp.status_code,
                    error_type="auth",
                    message=f"Polygon auth error: {resp.status_code}",
                    retryable=False,
                ),
            )

        if resp.status_code != 200:
            return AdapterResponse(
                data=None,
                lineage=lineage,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=lineage_endpoint,
                    status_code=resp.status_code,
                    error_type="http",
                    message=f"Polygon HTTP {resp.status_code}",
                    retryable=resp.status_code >= 500,
                ),
            )

        try:
            data = resp.json()
        except ValueError as exc:
            return AdapterResponse(
                data=None,
                lineage=lineage,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=lineage_endpoint,
                    status_code=200,
                    error_type="parse",
                    message=f"JSON parse error: {exc}",
                    retryable=False,
                ),
            )

        return AdapterResponse(data=data, lineage=lineage)

    # --- Short interest / borrow ---

    def get_short_interest(
        self,
        ticker: str,
        date_str: Optional[str] = None,
    ) -> AdapterResponse[List[PolygonShortInterest]]:
        """Fetch reported short interest for a ticker, optionally by settlement date."""

        endpoint = "/stocks/v1/short-interest"
        params: Dict[str, Any] = {"ticker": ticker}
        if date_str:
            params["settlement_date"] = date_str
        resp = self._request(endpoint, params=params)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        results_list = resp.data.get("results", []) if isinstance(resp.data, dict) else []
        results = [
            PolygonShortInterest(
                ticker=r.get("ticker", ticker),
                settlement_date=r.get("settlement_date", ""),
                short_interest=r.get("short_interest"),
                avg_daily_volume=r.get("avg_daily_volume"),
                days_to_cover=r.get("days_to_cover"),
            )
            for r in results_list
        ]
        return AdapterResponse(data=results, lineage=resp.lineage)

    # --- Bulk ticker reference / identity ---

    def get_tickers(
        self,
        *,
        market: Optional[str] = "stocks",
        locale: Optional[str] = "us",
        active: Optional[bool] = None,
        ticker: Optional[str] = None,
        ticker_gte: Optional[str] = None,
        ticker_lte: Optional[str] = None,
        type: Optional[str] = None,
        sort: Optional[str] = "ticker",
        order: Optional[str] = "asc",
        limit: int = 1000,
        max_pages: Optional[int] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[PolygonTickerReferencePage]]:
        """Fetch Polygon's paginated all-tickers reference identity feed."""

        endpoint = TICKERS_ENDPOINT
        params: Dict[str, Any] = {"limit": min(int(limit), 1000)}
        if market is not None:
            params["market"] = market
        if locale is not None:
            params["locale"] = locale
        if active is not None:
            params["active"] = str(active).lower()
        if ticker is not None:
            params["ticker"] = ticker
        if ticker_gte is not None:
            params["ticker.gte"] = ticker_gte
        if ticker_lte is not None:
            params["ticker.lte"] = ticker_lte
        if type is not None:
            params["type"] = type
        if sort is not None:
            params["sort"] = sort
        if order is not None:
            params["order"] = order

        pages: List[PolygonTickerReferencePage] = []
        next_url: Optional[str] = None
        first_lineage: Optional[LineageMeta] = None
        page_number = 0

        while True:
            request_params = (
                dict(params)
                if page_number == 0
                else {"cursor_url_path": _safe_url_path(next_url)}
            )
            resp = self._request(
                endpoint,
                params=params if page_number == 0 else None,
                asof=asof,
                url_override=next_url,
                lineage_endpoint=endpoint,
            )
            first_lineage = first_lineage or resp.lineage
            if not resp.ok:
                return AdapterResponse(
                    data=None,
                    lineage=resp.lineage,
                    error=resp.error,
                )

            payload = resp.data if isinstance(resp.data, dict) else {}
            raw_results = payload.get("results") or []
            if not isinstance(raw_results, list):
                raw_results = []
            next_url = _str_or_none(payload.get("next_url"))
            pages.append(PolygonTickerReferencePage(
                results=[
                    _parse_ticker_reference_row(row)
                    for row in raw_results
                    if isinstance(row, dict)
                ],
                lineage=resp.lineage,
                request_params=request_params,
                page_number=page_number,
                next_url=_safe_url_path(next_url),
                raw_payload=payload,
            ))

            page_number += 1
            if not next_url:
                break
            if max_pages is not None and page_number >= max_pages:
                break

        return AdapterResponse(
            data=pages,
            lineage=first_lineage or LineageMeta(
                provider=PROVIDER,
                endpoint=endpoint,
                request_timestamp=utcnow(),
                asof_timestamp=utcnow(),
                raw_payload_hash="",
                source_authority="Polygon",
            ),
        )

    # --- Ticker details ---

    def get_ticker_details(
        self,
        ticker: str,
        *,
        date_str: Optional[str] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[Optional[PolygonTickerDetail]]:
        """Fetch reference details for one ticker including CIK/FIGI identity."""

        endpoint = f"{TICKERS_ENDPOINT}/{ticker}"
        params: Dict[str, Any] = {}
        if date_str:
            params["date"] = date_str
        resp = self._request(endpoint, params=params or None, asof=asof)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        r = resp.data.get("results", {}) if isinstance(resp.data, dict) else {}
        if not r:
            return AdapterResponse(data=None, lineage=resp.lineage)

        detail = PolygonTickerDetail(
            ticker=_str_or_none(r.get("ticker")) or ticker,
            name=_str_or_none(r.get("name")) or "",
            market=_str_or_none(r.get("market")),
            locale=_str_or_none(r.get("locale")),
            primary_exchange=_str_or_none(r.get("primary_exchange")),
            type=_str_or_none(r.get("type")),
            active=r.get("active"),
            cik=_normalized_cik(r.get("cik")),
            composite_figi=_str_or_none(r.get("composite_figi")),
            share_class_figi=_str_or_none(r.get("share_class_figi")),
            list_date=_str_or_none(r.get("list_date")),
            delisted_utc=_str_or_none(r.get("delisted_utc")),
            market_cap=r.get("market_cap"),
            sic_code=_str_or_none(r.get("sic_code")),
            sic_description=r.get("sic_description"),
            share_class_shares_outstanding=r.get("share_class_shares_outstanding"),
            weighted_shares_outstanding=r.get("weighted_shares_outstanding"),
            raw=dict(r),
        )
        return AdapterResponse(data=detail, lineage=resp.lineage)

    # --- Ticker events ---

    def get_ticker_events(
        self,
        identifier: str,
        *,
        types: str = "ticker_change",
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[PolygonTickerEvent]]:
        """Fetch ticker events (e.g. ticker_change) from experimental vX endpoint.

        ``identifier`` can be a ticker, CUSIP, or Composite FIGI.
        """
        endpoint = f"{TICKER_EVENTS_ENDPOINT_PREFIX}/{identifier}/events"
        params: Dict[str, Any] = {"types": types}
        resp = self._request(endpoint, params=params, asof=asof)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        raw_events = []
        if isinstance(resp.data, dict):
            raw_events = resp.data.get("results", {}).get("events", [])
            if not isinstance(raw_events, list):
                raw_events = []

        events = []
        for ev in raw_events:
            if not isinstance(ev, dict):
                continue
            event_type = _str_or_none(ev.get("type") or ev.get("event_type")) or "unknown"
            ev_date = _str_or_none(ev.get("date"))
            effective_date = _str_or_none(ev.get("effective_date") or ev_date)
            ticker_change = ev.get("ticker_change", {})
            if not isinstance(ticker_change, dict):
                ticker_change = {}
            events.append(PolygonTickerEvent(
                identifier_queried=identifier,
                event_type=event_type,
                date=ev_date,
                event_date=ev_date,
                effective_date=effective_date,
                ticker=_str_or_none(ticker_change.get("ticker") or ev.get("ticker")),
                old_ticker=_str_or_none(
                    ticker_change.get("ticker")
                    or ev.get("old_ticker")
                    or ev.get("previous_ticker")
                ),
                new_ticker=_str_or_none(
                    ticker_change.get("new_ticker")
                    or ev.get("new_ticker")
                    or ev.get("successor_ticker")
                ),
                cik=_normalized_cik(ev.get("cik")),
                composite_figi=_str_or_none(ev.get("composite_figi")),
                share_class_figi=_str_or_none(ev.get("share_class_figi")),
                name=ev.get("name"),
                raw_event=ev,
            ))

        return AdapterResponse(data=events, lineage=resp.lineage)

    # --- Market data ---

    def get_daily_bars(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        limit: int = 5000,
    ) -> AdapterResponse[List[PolygonBar]]:
        """Fetch daily aggregate bars for a ticker and date range."""

        endpoint = f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
        params: Dict[str, Any] = {"limit": limit, "sort": "asc"}
        resp = self._request(endpoint, params=params)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        results_list = resp.data.get("results", []) if isinstance(resp.data, dict) else []
        bars = [
            PolygonBar(
                timestamp=b.get("t", 0),
                open=b.get("o", 0.0),
                high=b.get("h", 0.0),
                low=b.get("l", 0.0),
                close=b.get("c", 0.0),
                volume=b.get("v", 0),
                vwap=b.get("vw"),
                transactions=b.get("n"),
            )
            for b in results_list
        ]
        return AdapterResponse(data=bars, lineage=resp.lineage)


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _normalized_cik(value: Any) -> Optional[str]:
    text = _str_or_none(value)
    if text is None:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(10)


def _safe_url_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        return parsed.path
    return value


def _parse_ticker_reference_row(row: Dict[str, Any]) -> PolygonTickerReference:
    return PolygonTickerReference(
        ticker=_str_or_none(row.get("ticker")) or "",
        name=_str_or_none(row.get("name")) or "",
        market=_str_or_none(row.get("market")),
        locale=_str_or_none(row.get("locale")),
        primary_exchange=_str_or_none(row.get("primary_exchange")),
        type=_str_or_none(row.get("type")),
        active=row.get("active"),
        cik=_normalized_cik(row.get("cik")),
        composite_figi=_str_or_none(row.get("composite_figi")),
        share_class_figi=_str_or_none(row.get("share_class_figi")),
        list_date=_str_or_none(row.get("list_date")),
        delisted_utc=_str_or_none(row.get("delisted_utc")),
        raw=dict(row),
    )
