"""
Polygon.io adapter.

Supplemental source for:
  - Short-interest / short-volume proxy data for future I3 LITE work
  - Ticker details
  - Market data where needed

Does not write to DB. Returns AdapterResponse with LineageMeta.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlparse

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
SPLITS_ENDPOINT = "/stocks/v1/splits"
DIVIDENDS_ENDPOINT = "/stocks/v1/dividends"
SHORT_INTEREST_ENDPOINT = "/stocks/v1/short-interest"
SHORT_VOLUME_ENDPOINT = "/stocks/v1/short-volume"
POLYGON_API_HOSTS = {"api.polygon.io", "api.massive.com"}


# --- Response types ---

@dataclass
class PolygonShortInterest:
    """Normalized Polygon short-interest observation."""

    ticker: str
    settlement_date: str
    short_interest: Optional[int] = None
    avg_daily_volume: Optional[int] = None
    days_to_cover: Optional[Decimal] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class PolygonShortVolume:
    """Normalized Polygon short-volume observation with raw payload preserved."""

    ticker: str
    date: str
    short_volume: Optional[Decimal] = None
    total_volume: Optional[Decimal] = None
    short_volume_ratio: Optional[Decimal] = None
    exempt_volume: Optional[Decimal] = None
    non_exempt_volume: Optional[Decimal] = None
    adf_short_volume: Optional[Decimal] = None
    adf_short_volume_exempt: Optional[Decimal] = None
    nasdaq_carteret_short_volume: Optional[Decimal] = None
    nasdaq_carteret_short_volume_exempt: Optional[Decimal] = None
    nasdaq_chicago_short_volume: Optional[Decimal] = None
    nasdaq_chicago_short_volume_exempt: Optional[Decimal] = None
    nyse_short_volume: Optional[Decimal] = None
    nyse_short_volume_exempt: Optional[Decimal] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class PolygonSplit:
    """Normalized Polygon split row with raw payload preserved."""

    id: Optional[str]
    ticker: str
    execution_date: str
    split_from: Decimal
    split_to: Decimal
    adjustment_type: Optional[str] = None
    historical_adjustment_factor: Optional[Decimal] = None
    status: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class PolygonDividend:
    """Normalized Polygon dividend row with raw payload preserved."""

    id: Optional[str]
    ticker: str
    ex_dividend_date: str
    cash_amount: Decimal
    currency: Optional[str] = None
    declaration_date: Optional[str] = None
    dividend_type: Optional[str] = None
    distribution_type: Optional[str] = None
    frequency: Optional[int] = None
    historical_adjustment_factor: Optional[Decimal] = None
    pay_date: Optional[str] = None
    record_date: Optional[str] = None
    split_adjusted_cash_amount: Optional[Decimal] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class _NextPageRequest:
    """Sanitized pagination request rebuilt from a provider next_url."""

    url: str
    params: Dict[str, Any]
    path: str


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
                    message=f"Polygon request failed: {exc.__class__.__name__}",
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

    # --- Corporate actions ---

    def get_splits(
        self,
        ticker: Optional[str] = None,
        execution_date: Optional[str] = None,
        *,
        execution_date_from: Optional[str] = None,
        execution_date_to: Optional[str] = None,
        limit: int = 1000,
        sort: Optional[str] = "execution_date",
        order: Optional[str] = "asc",
        max_pages: int = 10,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[PolygonSplit]]:
        """Fetch Polygon split rows as corporate-action cross-check evidence."""

        validation_error = _validate_corporate_action_query(
            endpoint=SPLITS_ENDPOINT,
            ticker=ticker,
            date_field="execution_date",
            date_value=execution_date,
            date_from=execution_date_from,
            date_to=execution_date_to,
            max_pages=max_pages,
            asof=asof,
        )
        if validation_error is not None:
            return validation_error  # type: ignore[return-value]

        params = _corporate_action_params(
            ticker=ticker,
            date_field="execution_date",
            date_value=execution_date,
            date_from=execution_date_from,
            date_to=execution_date_to,
            limit=limit,
            sort=sort,
            order=order,
        )
        resp = self._fetch_corporate_action_rows(
            SPLITS_ENDPOINT,
            params=params,
            max_pages=max_pages,
            asof=asof,
        )
        if not resp.ok:
            return resp  # type: ignore[return-value]

        splits = []
        raw_rows = 0
        for row in resp.data or []:
            raw_rows += 1
            if not isinstance(row, dict):
                continue
            split = _parse_split_row(row)
            if split is not None:
                splits.append(split)
        lineage = _corporate_action_row_lineage(resp.lineage, raw_rows, len(splits))
        return AdapterResponse(data=splits, lineage=lineage)

    def get_dividends(
        self,
        ticker: Optional[str] = None,
        ex_dividend_date: Optional[str] = None,
        *,
        ex_dividend_date_from: Optional[str] = None,
        ex_dividend_date_to: Optional[str] = None,
        limit: int = 1000,
        sort: Optional[str] = "ex_dividend_date",
        order: Optional[str] = "asc",
        max_pages: int = 10,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[PolygonDividend]]:
        """Fetch Polygon dividend rows as corporate-action cross-check evidence."""

        validation_error = _validate_corporate_action_query(
            endpoint=DIVIDENDS_ENDPOINT,
            ticker=ticker,
            date_field="ex_dividend_date",
            date_value=ex_dividend_date,
            date_from=ex_dividend_date_from,
            date_to=ex_dividend_date_to,
            max_pages=max_pages,
            asof=asof,
        )
        if validation_error is not None:
            return validation_error  # type: ignore[return-value]

        params = _corporate_action_params(
            ticker=ticker,
            date_field="ex_dividend_date",
            date_value=ex_dividend_date,
            date_from=ex_dividend_date_from,
            date_to=ex_dividend_date_to,
            limit=limit,
            sort=sort,
            order=order,
        )
        resp = self._fetch_corporate_action_rows(
            DIVIDENDS_ENDPOINT,
            params=params,
            max_pages=max_pages,
            asof=asof,
        )
        if not resp.ok:
            return resp  # type: ignore[return-value]

        dividends = []
        raw_rows = 0
        for row in resp.data or []:
            raw_rows += 1
            if not isinstance(row, dict):
                continue
            dividend = _parse_dividend_row(row)
            if dividend is not None:
                dividends.append(dividend)
        lineage = _corporate_action_row_lineage(resp.lineage, raw_rows, len(dividends))
        return AdapterResponse(data=dividends, lineage=lineage)

    def _fetch_corporate_action_rows(
        self,
        endpoint: str,
        *,
        params: Dict[str, Any],
        max_pages: int,
        asof: Optional[datetime],
        feed_label: str = "corporate-action",
    ) -> AdapterResponse[List[Any]]:
        """Fetch Polygon results rows across pages without leaking cursors."""

        rows: List[Any] = []
        next_url: Optional[str] = None
        next_request: Optional[_NextPageRequest] = None
        first_lineage: Optional[LineageMeta] = None
        page_hashes: List[str] = []
        next_url_paths: List[str] = []
        page_number = 0
        page_cap = int(max_pages)

        while True:
            resp = self._request(
                endpoint,
                params=params if page_number == 0 else next_request.params,
                asof=asof,
                url_override=None if page_number == 0 else next_request.url,
                lineage_endpoint=endpoint,
            )
            first_lineage = first_lineage or resp.lineage
            if not resp.ok:
                lineage = _corporate_action_error_lineage(
                    resp.lineage,
                    page_count=page_number + 1,
                    next_url_paths=next_url_paths,
                    truncated=page_number > 0 or bool(next_url_paths),
                )
                return AdapterResponse(data=None, lineage=lineage, error=resp.error)

            page_hashes.append(resp.lineage.raw_payload_hash)
            payload = resp.data
            if (
                not isinstance(payload, dict)
                or "results" not in payload
                or not isinstance(payload.get("results"), list)
            ):
                lineage = _corporate_action_lineage(
                    first_lineage=first_lineage,
                    page_hashes=page_hashes,
                    page_count=page_number + 1,
                    next_url_paths=next_url_paths,
                    truncated=page_number > 0 or bool(next_url_paths),
                )
                return AdapterResponse(
                    data=None,
                    lineage=lineage,
                    error=ProviderError(
                        provider=PROVIDER,
                        endpoint=endpoint,
                        status_code=200,
                        error_type="parse",
                        message=f"Polygon {feed_label} response missing list results",
                        retryable=False,
                    ),
                )
            rows.extend(payload["results"])

            next_url = _str_or_none(payload.get("next_url"))
            next_request = None
            if next_url:
                safe_path = _safe_url_path(next_url)
                if safe_path:
                    next_url_paths.append(safe_path)
                next_request = _corporate_action_next_request(
                    next_url,
                    endpoint=endpoint,
                    base_url=self._config.base_url,
                )
                if next_request is None:
                    lineage = _corporate_action_lineage(
                        first_lineage=first_lineage,
                        page_hashes=page_hashes,
                        page_count=page_number + 1,
                        next_url_paths=next_url_paths,
                        truncated=True,
                    )
                    return AdapterResponse(
                        data=None,
                        lineage=lineage,
                        error=ProviderError(
                            provider=PROVIDER,
                            endpoint=endpoint,
                            status_code=200,
                            error_type="pagination",
                            message=f"Polygon {feed_label} pagination next_url rejected",
                            retryable=False,
                        ),
                    )

            page_number += 1
            if not next_url:
                break
            if page_number >= page_cap:
                lineage = _corporate_action_lineage(
                    first_lineage=first_lineage,
                    page_hashes=page_hashes,
                    page_count=page_number,
                    next_url_paths=next_url_paths,
                    truncated=True,
                )
                return AdapterResponse(
                    data=None,
                    lineage=lineage,
                    error=ProviderError(
                        provider=PROVIDER,
                        endpoint=endpoint,
                        status_code=200,
                        error_type="pagination",
                        message=f"Polygon {feed_label} pagination exceeded max_pages",
                        retryable=False,
                    ),
                )

        lineage = _corporate_action_lineage(
            first_lineage=first_lineage,
            page_hashes=page_hashes,
            page_count=page_number,
            next_url_paths=next_url_paths,
            truncated=False,
        )
        return AdapterResponse(data=rows, lineage=lineage)

    # --- Short-interest / short-volume proxy ---

    def get_short_interest(
        self,
        ticker: Optional[str] = None,
        settlement_date: Optional[str] = None,
        *,
        settlement_date_from: Optional[str] = None,
        settlement_date_to: Optional[str] = None,
        limit: int = 1000,
        sort: Optional[str] = "settlement_date",
        order: Optional[str] = "desc",
        max_pages: int = 10,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[PolygonShortInterest]]:
        """Fetch Polygon short-interest rows as I3 LITE/proxy evidence."""

        validation_error = _validate_polygon_feed_query(
            endpoint=SHORT_INTEREST_ENDPOINT,
            ticker=ticker,
            date_value=settlement_date,
            date_from=settlement_date_from,
            date_to=settlement_date_to,
            max_pages=max_pages,
            asof=asof,
        )
        if validation_error is not None:
            return validation_error  # type: ignore[return-value]

        params = _dated_feed_params(
            ticker=ticker,
            date_field="settlement_date",
            date_value=settlement_date,
            date_from=settlement_date_from,
            date_to=settlement_date_to,
            limit=limit,
            sort=sort,
            order=order,
            max_limit=50000,
        )
        resp = self._fetch_corporate_action_rows(
            SHORT_INTEREST_ENDPOINT,
            params=params,
            max_pages=max_pages,
            asof=asof,
            feed_label="short-interest",
        )
        if not resp.ok:
            return resp  # type: ignore[return-value]

        results = []
        raw_rows = 0
        for row in resp.data or []:
            raw_rows += 1
            if not isinstance(row, dict):
                continue
            parsed = _parse_short_interest_row(row)
            if parsed is not None:
                results.append(parsed)
        lineage = _corporate_action_row_lineage(resp.lineage, raw_rows, len(results))
        return AdapterResponse(data=results, lineage=lineage)

    def get_short_volume(
        self,
        ticker: Optional[str] = None,
        date: Optional[str] = None,
        *,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 1000,
        sort: Optional[str] = "date",
        order: Optional[str] = "desc",
        max_pages: int = 10,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[PolygonShortVolume]]:
        """Fetch Polygon short-volume rows as I3 LITE/proxy evidence."""

        validation_error = _validate_polygon_feed_query(
            endpoint=SHORT_VOLUME_ENDPOINT,
            ticker=ticker,
            date_value=date,
            date_from=date_from,
            date_to=date_to,
            max_pages=max_pages,
            asof=asof,
        )
        if validation_error is not None:
            return validation_error  # type: ignore[return-value]

        params = _dated_feed_params(
            ticker=ticker,
            date_field="date",
            date_value=date,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            sort=sort,
            order=order,
            max_limit=50000,
        )
        resp = self._fetch_corporate_action_rows(
            SHORT_VOLUME_ENDPOINT,
            params=params,
            max_pages=max_pages,
            asof=asof,
            feed_label="short-volume",
        )
        if not resp.ok:
            return resp  # type: ignore[return-value]

        results = []
        raw_rows = 0
        for row in resp.data or []:
            raw_rows += 1
            if not isinstance(row, dict):
                continue
            parsed = _parse_short_volume_row(row)
            if parsed is not None:
                results.append(parsed)
        lineage = _corporate_action_row_lineage(resp.lineage, raw_rows, len(results))
        lineage = _short_volume_semantic_lineage(lineage, results)
        return AdapterResponse(data=results, lineage=lineage)

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


def _decimal_or_none(value: Any) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    text = _str_or_none(value)
    if text is None or text.upper() in {"N/A", "NA", "NONE", "NULL"}:
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    text = _str_or_none(value)
    if text is None or text.upper() in {"N/A", "NA", "NONE", "NULL"}:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _ticker_param(value: Optional[str]) -> Optional[str]:
    text = _str_or_none(value)
    return text.upper() if text else None


def _limited_int(value: int, *, default: int = 1000, maximum: int = 5000) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(limit, maximum))


def _provider_error_response(
    *,
    endpoint: str,
    error_type: str,
    message: str,
    retryable: bool,
    status_code: Optional[int] = None,
    asof: Optional[datetime] = None,
    flags: Optional[Dict[str, Any]] = None,
) -> AdapterResponse[Any]:
    request_ts = utcnow()
    asof_ts = aware_utc_or_none(asof) if asof is not None else request_ts
    if asof_ts is None:
        asof_ts = request_ts
    return AdapterResponse(
        data=None,
        lineage=LineageMeta(
            provider=PROVIDER,
            endpoint=endpoint,
            request_timestamp=request_ts,
            asof_timestamp=asof_ts,
            raw_payload_hash="",
            source_authority="Polygon",
            data_quality_flags=flags,
        ),
        error=ProviderError(
            provider=PROVIDER,
            endpoint=endpoint,
            status_code=status_code,
            error_type=error_type,
            message=message,
            retryable=retryable,
        ),
    )


def _validate_corporate_action_query(
    *,
    endpoint: str,
    ticker: Optional[str],
    date_field: str,
    date_value: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    max_pages: Any,
    asof: Optional[datetime],
) -> Optional[AdapterResponse[Any]]:
    try:
        page_cap = int(max_pages)
    except (TypeError, ValueError):
        return _provider_error_response(
            endpoint=endpoint,
            error_type="validation",
            message="Polygon corporate-action max_pages must be a positive integer",
            retryable=False,
            asof=asof,
        )
    if page_cap < 1:
        return _provider_error_response(
            endpoint=endpoint,
            error_type="validation",
            message="Polygon corporate-action max_pages must be a positive integer",
            retryable=False,
            asof=asof,
        )
    date_error = _validate_iso_date_params(
        endpoint=endpoint,
        date_fields={
            date_field: date_value,
            f"{date_field}_from": date_from,
            f"{date_field}_to": date_to,
        },
        asof=asof,
    )
    if date_error is not None:
        return date_error
    if _ticker_param(ticker) is None and not (date_value or (date_from and date_to)):
        return _provider_error_response(
            endpoint=endpoint,
            error_type="validation",
            message="Polygon corporate-action broad query requires an exact date or bounded date window",
            retryable=False,
            asof=asof,
        )
    return None


def _validate_polygon_feed_query(
    *,
    endpoint: str,
    ticker: Optional[str],
    date_value: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    max_pages: Any,
    asof: Optional[datetime],
) -> Optional[AdapterResponse[Any]]:
    try:
        page_cap = int(max_pages)
    except (TypeError, ValueError):
        return _provider_error_response(
            endpoint=endpoint,
            error_type="validation",
            message="Polygon feed max_pages must be a positive integer",
            retryable=False,
            asof=asof,
        )
    if page_cap < 1:
        return _provider_error_response(
            endpoint=endpoint,
            error_type="validation",
            message="Polygon feed max_pages must be a positive integer",
            retryable=False,
            asof=asof,
        )
    date_error = _validate_iso_date_params(
        endpoint=endpoint,
        date_fields={
            "date": date_value,
            "date_from": date_from,
            "date_to": date_to,
        },
        asof=asof,
    )
    if date_error is not None:
        return date_error
    if _ticker_param(ticker) is None and not (date_value or (date_from and date_to)):
        return _provider_error_response(
            endpoint=endpoint,
            error_type="validation",
            message="Polygon feed broad query requires an exact date or bounded date window",
            retryable=False,
            asof=asof,
        )
    return None


def _corporate_action_params(
    *,
    ticker: Optional[str],
    date_field: str,
    date_value: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    limit: int,
    sort: Optional[str],
    order: Optional[str],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"limit": _limited_int(limit)}
    ticker_param = _ticker_param(ticker)
    if ticker_param:
        params["ticker"] = ticker_param
    if date_value:
        params[date_field] = date_value
    else:
        if date_from:
            params[f"{date_field}.gte"] = date_from
        if date_to:
            params[f"{date_field}.lte"] = date_to
    sort_param = _corporate_action_sort(sort, order)
    if sort_param is not None:
        params["sort"] = sort_param
    return params


def _dated_feed_params(
    *,
    ticker: Optional[str],
    date_field: str,
    date_value: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    limit: int,
    sort: Optional[str],
    order: Optional[str],
    max_limit: int,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"limit": _limited_int(limit, maximum=max_limit)}
    ticker_param = _ticker_param(ticker)
    if ticker_param:
        params["ticker"] = ticker_param
    if date_value:
        params[date_field] = date_value
    else:
        if date_from:
            params[f"{date_field}.gte"] = date_from
        if date_to:
            params[f"{date_field}.lte"] = date_to
    sort_param = _corporate_action_sort(sort, order)
    if sort_param is not None:
        params["sort"] = sort_param
    return params


def _corporate_action_sort(
    sort: Optional[str],
    order: Optional[str],
) -> Optional[str]:
    if sort is None:
        return None
    sort_text = sort.strip()
    if not sort_text:
        return None
    if order is None:
        return sort_text
    order_text = order.strip().lower()
    if order_text not in {"asc", "desc"}:
        return sort_text
    parts = [part.strip() for part in sort_text.split(",") if part.strip()]
    if not parts:
        return None
    suffixed = []
    for part in parts:
        if part.endswith(".asc") or part.endswith(".desc"):
            suffixed.append(part)
        else:
            suffixed.append(f"{part}.{order_text}")
    return ",".join(suffixed)


def _results_list(payload: Any) -> List[Any]:
    if not isinstance(payload, dict):
        return []
    results = payload.get("results", [])
    return results if isinstance(results, list) else []


def _corporate_action_next_request(
    next_url: str,
    *,
    endpoint: str,
    base_url: str,
) -> Optional[_NextPageRequest]:
    parsed = urlparse(next_url)
    base_parsed = urlparse(base_url)
    base_host = _next_url_host_or_none(base_parsed)
    allowed_hosts = {
        host.lower()
        for host in (base_host, *POLYGON_API_HOSTS)
        if host
    }
    if parsed.scheme or parsed.netloc:
        host = _next_url_host_or_none(parsed)
        if parsed.scheme != "https" or host not in allowed_hosts:
            return None
    path = parsed.path or endpoint
    if not path.startswith("/"):
        path = f"/{path}"
    if path != endpoint:
        return None
    cursor = None
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        if key == "cursor" and value:
            cursor = value
            break
    if cursor is None:
        return None
    return _NextPageRequest(
        url=f"{base_url.rstrip('/')}{path}",
        params={"cursor": cursor},
        path=path,
    )


def _next_url_host_or_none(parsed_url) -> Optional[str]:
    if parsed_url.username or parsed_url.password:
        return None
    host = parsed_url.hostname
    if not host:
        return None
    try:
        port = parsed_url.port
    except ValueError:
        return None
    if port not in (None, 443):
        return None
    return host.lower()


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
    if parsed.query:
        return parsed.path or None
    return value


def _corporate_action_lineage(
    *,
    first_lineage: Optional[LineageMeta],
    page_hashes: List[str],
    page_count: int,
    next_url_paths: List[str],
    truncated: bool,
) -> LineageMeta:
    if first_lineage is None:
        now = utcnow()
        return LineageMeta(
            provider=PROVIDER,
            endpoint="",
            request_timestamp=now,
            asof_timestamp=now,
            raw_payload_hash="",
            source_authority="Polygon",
        )
    flags = dict(first_lineage.data_quality_flags or {})
    flags.update({
        "page_count": page_count,
        "paginated": page_count > 1,
        "truncated": truncated,
    })
    if next_url_paths:
        flags["next_url_paths"] = list(next_url_paths)
    payload_hash = (
        first_lineage.raw_payload_hash
        if page_count == 1 and not truncated
        else stable_hash({
            "page_hashes": page_hashes,
            "page_count": page_count,
            "truncated": truncated,
        })
    )
    return replace(
        first_lineage,
        raw_payload_hash=payload_hash,
        data_quality_flags=flags,
    )


def _corporate_action_error_lineage(
    lineage: LineageMeta,
    *,
    page_count: int,
    next_url_paths: List[str],
    truncated: bool,
) -> LineageMeta:
    flags = dict(lineage.data_quality_flags or {})
    flags.update({
        "page_count": page_count,
        "paginated": page_count > 1,
        "truncated": truncated,
    })
    if next_url_paths:
        flags["next_url_paths"] = list(next_url_paths)
    return replace(lineage, data_quality_flags=flags)


def _corporate_action_row_lineage(
    lineage: LineageMeta,
    raw_rows: int,
    parsed_rows: int,
) -> LineageMeta:
    flags = dict(lineage.data_quality_flags or {})
    skipped_rows = max(0, raw_rows - parsed_rows)
    flags.update({
        "raw_rows": raw_rows,
        "parsed_rows": parsed_rows,
        "skipped_rows": skipped_rows,
    })
    if raw_rows > 0 and parsed_rows == 0:
        flags["all_rows_skipped"] = True
    else:
        flags.pop("all_rows_skipped", None)
    return replace(lineage, data_quality_flags=flags)


def _short_volume_semantic_lineage(
    lineage: LineageMeta,
    rows: List[PolygonShortVolume],
) -> LineageMeta:
    warning_types: Dict[str, int] = {}
    warning_rows = 0
    for row in rows:
        row_warnings = _short_volume_semantic_warnings(row)
        if row_warnings:
            warning_rows += 1
            for warning in row_warnings:
                warning_types[warning] = warning_types.get(warning, 0) + 1

    flags = dict(lineage.data_quality_flags or {})
    flags["semantic_warning_rows"] = warning_rows
    flags["semantic_warning_types"] = warning_types
    return replace(lineage, data_quality_flags=flags)


def _short_volume_semantic_warnings(row: PolygonShortVolume) -> List[str]:
    warnings: List[str] = []
    if (
        row.short_volume is not None
        and row.total_volume is not None
        and row.total_volume > 0
        and row.short_volume > row.total_volume
    ):
        warnings.append("short_volume_gt_total")
    if row.short_volume_ratio is not None and row.short_volume_ratio > 100:
        warnings.append("short_volume_ratio_gt_100")
    if (
        row.short_volume_ratio is not None
        and row.short_volume_ratio == 0
        and row.short_volume is not None
        and row.short_volume > 0
    ):
        warnings.append("zero_ratio_with_positive_short_volume")
    if (
        row.total_volume is not None
        and row.total_volume == 0
        and row.short_volume is not None
        and row.short_volume > 0
    ):
        warnings.append("zero_total_with_positive_short_volume")
    if (
        row.exempt_volume is not None
        and row.non_exempt_volume is not None
        and row.short_volume is not None
        and row.exempt_volume + row.non_exempt_volume != row.short_volume
    ):
        warnings.append("exempt_non_exempt_sum_mismatch")
    return warnings


def _validate_iso_date_params(
    *,
    endpoint: str,
    date_fields: Dict[str, Optional[str]],
    asof: Optional[datetime],
) -> Optional[AdapterResponse[Any]]:
    for label, value in date_fields.items():
        if value is None:
            continue
        text = _str_or_none(value)
        if text is None or len(text) != 10:
            return _invalid_date_response(endpoint=endpoint, label=label, asof=asof)
        try:
            date.fromisoformat(text)
        except ValueError:
            return _invalid_date_response(endpoint=endpoint, label=label, asof=asof)
    return None


def _invalid_date_response(
    *,
    endpoint: str,
    label: str,
    asof: Optional[datetime],
) -> AdapterResponse[Any]:
    return _provider_error_response(
        endpoint=endpoint,
        error_type="validation",
        message=f"Polygon feed {label} must be YYYY-MM-DD",
        retryable=False,
        asof=asof,
    )


def _positive_decimal_or_none(value: Any) -> Optional[Decimal]:
    parsed = _decimal_or_none(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _nonnegative_decimal_or_none(value: Any) -> Optional[Decimal]:
    parsed = _decimal_or_none(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _optional_nonnegative_decimal(value: Any) -> tuple[Optional[Decimal], bool]:
    if value is None:
        return None, True
    parsed = _decimal_or_none(value)
    if parsed is None or parsed < 0:
        return None, False
    return parsed, True


def _optional_nonnegative_int(value: Any) -> tuple[Optional[int], bool]:
    if value is None:
        return None, True
    parsed = _decimal_or_none(value)
    if parsed is None or parsed < 0 or parsed != parsed.to_integral_value():
        return None, False
    return int(parsed), True


def _parse_short_interest_row(row: Dict[str, Any]) -> Optional[PolygonShortInterest]:
    ticker = _ticker_param(row.get("ticker"))  # type: ignore[arg-type]
    settlement_date = _str_or_none(row.get("settlement_date"))
    if not ticker or not settlement_date:
        return None
    short_interest, short_interest_ok = _optional_nonnegative_int(
        row.get("short_interest")
    )
    if short_interest is None or not short_interest_ok:
        return None
    avg_daily_volume, avg_volume_ok = _optional_nonnegative_int(
        row.get("avg_daily_volume")
    )
    days_to_cover, days_ok = _optional_nonnegative_decimal(row.get("days_to_cover"))
    if not (avg_volume_ok and days_ok):
        return None
    return PolygonShortInterest(
        ticker=ticker,
        settlement_date=settlement_date,
        short_interest=short_interest,
        avg_daily_volume=avg_daily_volume,
        days_to_cover=days_to_cover,
        raw=dict(row),
    )


def _parse_short_volume_row(row: Dict[str, Any]) -> Optional[PolygonShortVolume]:
    ticker = _ticker_param(row.get("ticker"))  # type: ignore[arg-type]
    date_value = _str_or_none(row.get("date"))
    if not ticker or not date_value:
        return None

    decimal_fields = [
        "short_volume",
        "total_volume",
        "short_volume_ratio",
        "exempt_volume",
        "non_exempt_volume",
        "adf_short_volume",
        "adf_short_volume_exempt",
        "nasdaq_carteret_short_volume",
        "nasdaq_carteret_short_volume_exempt",
        "nasdaq_chicago_short_volume",
        "nasdaq_chicago_short_volume_exempt",
        "nyse_short_volume",
        "nyse_short_volume_exempt",
    ]
    values: Dict[str, Optional[Decimal]] = {}
    for field in decimal_fields:
        parsed, ok = _optional_nonnegative_decimal(row.get(field))
        if not ok:
            return None
        values[field] = parsed
    if values["short_volume"] is None and values["short_volume_ratio"] is None:
        return None

    return PolygonShortVolume(
        ticker=ticker,
        date=date_value,
        short_volume=values["short_volume"],
        total_volume=values["total_volume"],
        short_volume_ratio=values["short_volume_ratio"],
        exempt_volume=values["exempt_volume"],
        non_exempt_volume=values["non_exempt_volume"],
        adf_short_volume=values["adf_short_volume"],
        adf_short_volume_exempt=values["adf_short_volume_exempt"],
        nasdaq_carteret_short_volume=values["nasdaq_carteret_short_volume"],
        nasdaq_carteret_short_volume_exempt=values[
            "nasdaq_carteret_short_volume_exempt"
        ],
        nasdaq_chicago_short_volume=values["nasdaq_chicago_short_volume"],
        nasdaq_chicago_short_volume_exempt=values[
            "nasdaq_chicago_short_volume_exempt"
        ],
        nyse_short_volume=values["nyse_short_volume"],
        nyse_short_volume_exempt=values["nyse_short_volume_exempt"],
        raw=dict(row),
    )


def _parse_split_row(row: Dict[str, Any]) -> Optional[PolygonSplit]:
    ticker = _str_or_none(row.get("ticker"))
    execution_date = _str_or_none(row.get("execution_date"))
    split_from = _positive_decimal_or_none(row.get("split_from"))
    split_to = _positive_decimal_or_none(row.get("split_to"))
    if not ticker or not execution_date or split_from is None or split_to is None:
        return None
    return PolygonSplit(
        id=_str_or_none(row.get("id")),
        ticker=ticker,
        execution_date=execution_date,
        split_from=split_from,
        split_to=split_to,
        adjustment_type=_str_or_none(row.get("adjustment_type")),
        historical_adjustment_factor=_positive_decimal_or_none(
            row.get("historical_adjustment_factor")
        ),
        status=_str_or_none(row.get("status")),
        raw=dict(row),
    )


def _parse_dividend_row(row: Dict[str, Any]) -> Optional[PolygonDividend]:
    ticker = _str_or_none(row.get("ticker"))
    ex_dividend_date = _str_or_none(row.get("ex_dividend_date"))
    cash_amount = _nonnegative_decimal_or_none(row.get("cash_amount"))
    if not ticker or not ex_dividend_date or cash_amount is None:
        return None
    return PolygonDividend(
        id=_str_or_none(row.get("id")),
        ticker=ticker,
        ex_dividend_date=ex_dividend_date,
        cash_amount=cash_amount,
        currency=_str_or_none(row.get("currency")),
        declaration_date=_str_or_none(row.get("declaration_date")),
        dividend_type=_str_or_none(row.get("dividend_type")),
        distribution_type=_str_or_none(row.get("distribution_type")),
        frequency=_int_or_none(row.get("frequency")),
        historical_adjustment_factor=_positive_decimal_or_none(
            row.get("historical_adjustment_factor")
        ),
        pay_date=_str_or_none(row.get("pay_date")),
        record_date=_str_or_none(row.get("record_date")),
        split_adjusted_cash_amount=_nonnegative_decimal_or_none(
            row.get("split_adjusted_cash_amount")
        ),
        raw=dict(row),
    )


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
