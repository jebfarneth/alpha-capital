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


# --- Response types ---

@dataclass
class PolygonShortInterest:
    ticker: str
    settlement_date: str
    short_interest: Optional[int] = None
    avg_daily_volume: Optional[int] = None
    days_to_cover: Optional[float] = None


@dataclass
class PolygonTickerDetail:
    ticker: str
    name: str
    market_cap: Optional[float] = None
    share_class_shares_outstanding: Optional[int] = None
    weighted_shares_outstanding: Optional[int] = None
    primary_exchange: Optional[str] = None
    type: Optional[str] = None
    sic_code: Optional[str] = None
    sic_description: Optional[str] = None
    locale: Optional[str] = None


@dataclass
class PolygonBar:
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
    ) -> AdapterResponse[Any]:
        url = f"{self._config.base_url}{endpoint}"
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
                        endpoint=endpoint,
                        request_timestamp=request_ts,
                        asof_timestamp=request_ts,
                        raw_payload_hash="",
                    ),
                    error=ProviderError(
                        provider=PROVIDER,
                        endpoint=endpoint,
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
                    endpoint=endpoint,
                    request_timestamp=request_ts,
                    asof_timestamp=asof_ts,
                    raw_payload_hash="",
                ),
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
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
                    endpoint=endpoint,
                    request_timestamp=request_ts,
                    asof_timestamp=asof_ts,
                    raw_payload_hash="",
                ),
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
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
            endpoint=endpoint,
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
                    endpoint=endpoint,
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
                    endpoint=endpoint,
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
                    endpoint=endpoint,
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
                    endpoint=endpoint,
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

    # --- Ticker details ---

    def get_ticker_details(
        self, ticker: str
    ) -> AdapterResponse[Optional[PolygonTickerDetail]]:
        endpoint = f"/v3/reference/tickers/{ticker}"
        resp = self._request(endpoint)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        r = resp.data.get("results", {}) if isinstance(resp.data, dict) else {}
        if not r:
            return AdapterResponse(data=None, lineage=resp.lineage)

        detail = PolygonTickerDetail(
            ticker=r.get("ticker", ticker),
            name=r.get("name", ""),
            market_cap=r.get("market_cap"),
            share_class_shares_outstanding=r.get("share_class_shares_outstanding"),
            weighted_shares_outstanding=r.get("weighted_shares_outstanding"),
            primary_exchange=r.get("primary_exchange"),
            type=r.get("type"),
            sic_code=r.get("sic_code"),
            sic_description=r.get("sic_description"),
            locale=r.get("locale"),
        )
        return AdapterResponse(data=detail, lineage=resp.lineage)

    # --- Market data ---

    def get_daily_bars(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        limit: int = 5000,
    ) -> AdapterResponse[List[PolygonBar]]:
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
