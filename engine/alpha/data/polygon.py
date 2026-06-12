"""
Polygon.io adapter.

Supplemental source for:
  - Short-interest / short-volume proxy data for future I3 LITE work
  - Ticker details
  - Market data where needed

Does not write to DB. Returns AdapterResponse with LineageMeta.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
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
NEWS_ENDPOINT = "/v2/reference/news"
FULL_MARKET_SNAPSHOT_ENDPOINT = "/v2/snapshot/locale/us/markets/stocks/tickers"
GROUPED_DAILY_AGGS_ENDPOINT_PREFIX = "/v2/aggs/grouped/locale/us/market/stocks"
POLYGON_API_HOSTS = {"api.polygon.io", "api.massive.com"}
TICKER_EVENT_ALLOWED_TYPES = {"ticker_change"}
TICKER_EVENT_IDENTIFIER_MAX_LENGTH = 64
TICKER_EVENT_IDENTIFIER_ALLOWED_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
PATH_TICKER_MAX_LENGTH = 64
PATH_TICKER_ALLOWED_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
DAILY_BAR_SPLIT_ADJUSTED_PRICE_BASIS = "polygon_daily_close_split_adjusted"
DAILY_BAR_UNADJUSTED_PRICE_BASIS = "polygon_daily_close_unadjusted"


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


@dataclass
class PolygonNewsArticle:
    """Normalized Polygon news article with publisher and insights preserved."""

    id: str
    title: str
    article_url: str
    publisher_name: Optional[str] = None
    publisher_homepage_url: Optional[str] = None
    publisher_logo_url: Optional[str] = None
    publisher_favicon_url: Optional[str] = None
    author: Optional[str] = None
    amp_url: Optional[str] = None
    image_url: Optional[str] = None
    description: Optional[str] = None
    published_utc: Optional[str] = None
    tickers: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    insights: List[Dict[str, Any]] = field(default_factory=list)
    publisher: Optional[Dict[str, Any]] = None
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
    old_cik: Optional[str] = None
    new_cik: Optional[str] = None
    composite_figi: Optional[str] = None
    old_composite_figi: Optional[str] = None
    new_composite_figi: Optional[str] = None
    share_class_figi: Optional[str] = None
    old_share_class_figi: Optional[str] = None
    new_share_class_figi: Optional[str] = None
    name: Optional[str] = None
    identity_continuity_status: str = "not_applicable"
    raw_event: Optional[Dict[str, Any]] = None


@dataclass
class PolygonBar:
    """Normalized Polygon aggregate daily bar."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    transactions: Optional[int] = None


@dataclass
class PolygonGroupedDailyBar(PolygonBar):
    """Grouped market daily bar with the row ticker preserved."""

    ticker: str = ""


@dataclass
class PolygonSnapshotTicker:
    """Normalized delayed full-market snapshot row for one ticker."""

    ticker: str
    day_open: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    day_close: Optional[float] = None
    day_volume: Optional[float] = None
    prev_day_close: Optional[float] = None
    prev_day_volume: Optional[float] = None
    minute_timestamp: Optional[int] = None
    minute_open: Optional[float] = None
    minute_high: Optional[float] = None
    minute_low: Optional[float] = None
    minute_close: Optional[float] = None
    minute_volume: Optional[float] = None
    last_trade_price: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None

    @property
    def decision_price(self) -> Optional[float]:
        return self.last_trade_price or self.minute_close or self.day_close


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

    # --- News / sentiment cross-check ---

    def get_news(
        self,
        ticker: Optional[str] = None,
        *,
        tickers: Optional[List[str]] = None,
        published_utc: Optional[str] = None,
        published_utc_from: Optional[str] = None,
        published_utc_to: Optional[str] = None,
        limit: int = 100,
        sort: Optional[str] = "published_utc",
        order: Optional[str] = "desc",
        max_pages: int = 10,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[PolygonNewsArticle]]:
        """Fetch Polygon news articles as news/sentiment cross-check evidence."""

        validation_error = _validate_polygon_news_query(
            ticker=ticker,
            tickers=tickers,
            published_utc=published_utc,
            published_utc_from=published_utc_from,
            published_utc_to=published_utc_to,
            max_pages=max_pages,
            asof=asof,
        )
        if validation_error is not None:
            return validation_error  # type: ignore[return-value]

        params = _news_params(
            ticker=ticker,
            tickers=tickers,
            published_utc=published_utc,
            published_utc_from=published_utc_from,
            published_utc_to=published_utc_to,
            limit=limit,
            sort=sort,
            order=order,
        )
        resp = self._fetch_corporate_action_rows(
            NEWS_ENDPOINT,
            params=params,
            max_pages=max_pages,
            asof=asof,
            feed_label="news",
        )
        if not resp.ok:
            return resp  # type: ignore[return-value]

        articles = []
        raw_rows = 0
        for row in resp.data or []:
            raw_rows += 1
            if not isinstance(row, dict):
                continue
            article = _parse_news_article_row(row)
            if article is not None:
                articles.append(article)
        lineage = _corporate_action_row_lineage(resp.lineage, raw_rows, len(articles))
        return AdapterResponse(data=articles, lineage=lineage)

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
        if max_pages is not None:
            try:
                page_cap = int(max_pages)
            except (TypeError, ValueError):
                return _provider_error_response(
                    endpoint=endpoint,
                    error_type="validation",
                    message="Polygon tickers max_pages must be a positive integer",
                    retryable=False,
                    asof=asof,
                )
            if page_cap < 1:
                return _provider_error_response(
                    endpoint=endpoint,
                    error_type="validation",
                    message="Polygon tickers max_pages must be a positive integer",
                    retryable=False,
                    asof=asof,
                )
        else:
            page_cap = None

        params: Dict[str, Any] = {"limit": _limited_int(limit, maximum=1000)}
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
        next_request: Optional[_NextPageRequest] = None
        first_lineage: Optional[LineageMeta] = None
        page_hashes: List[str] = []
        next_url_paths: List[str] = []
        page_number = 0
        raw_rows = 0
        parsed_rows = 0
        duplicate_same_identity_rows = 0
        seen_identity_hashes: Dict[str, str] = {}

        while True:
            request_params = dict(params) if page_number == 0 else dict(next_request.params)
            resp = self._request(
                endpoint,
                params=params if page_number == 0 else next_request.params,
                asof=asof,
                url_override=None if page_number == 0 else next_request.url,
                lineage_endpoint=endpoint,
            )
            first_lineage = first_lineage or resp.lineage
            if not resp.ok:
                lineage = resp.lineage
                if page_number > 0 or next_url_paths:
                    lineage = _corporate_action_error_lineage(
                        resp.lineage,
                        page_count=page_number + 1,
                        next_url_paths=next_url_paths,
                        truncated=True,
                    )
                return AdapterResponse(
                    data=None,
                    lineage=lineage,
                    error=resp.error,
                )

            page_hashes.append(resp.lineage.raw_payload_hash)
            payload = resp.data
            if (
                not isinstance(payload, dict)
                or "results" not in payload
                or not isinstance(payload.get("results"), list)
            ):
                lineage = _ticker_reference_lineage(
                    first_lineage=first_lineage,
                    page_hashes=page_hashes,
                    page_count=page_number + 1,
                    next_url_paths=next_url_paths,
                    truncated=page_number > 0 or bool(next_url_paths),
                    raw_rows=raw_rows,
                    parsed_rows=parsed_rows,
                    duplicate_same_identity_rows=duplicate_same_identity_rows,
                )
                return _parse_error_response(
                    lineage=lineage,
                    endpoint=endpoint,
                    message="Polygon tickers response missing list results",
                )

            raw_results = payload["results"]
            next_url = _str_or_none(payload.get("next_url"))
            parsed_results: List[PolygonTickerReference] = []
            for row in raw_results:
                raw_rows += 1
                if not isinstance(row, dict):
                    lineage = _ticker_reference_lineage(
                        first_lineage=first_lineage,
                        page_hashes=page_hashes,
                        page_count=page_number + 1,
                        next_url_paths=next_url_paths,
                        truncated=page_number > 0 or bool(next_url_paths),
                        raw_rows=raw_rows,
                        parsed_rows=parsed_rows,
                        duplicate_same_identity_rows=duplicate_same_identity_rows,
                    )
                    return _parse_error_response(
                        lineage=lineage,
                        endpoint=endpoint,
                        message="Polygon tickers response contains non-object result row",
                    )
                parsed = _parse_ticker_reference_row(row)
                if not parsed.ticker:
                    lineage = _ticker_reference_lineage(
                        first_lineage=first_lineage,
                        page_hashes=page_hashes,
                        page_count=page_number + 1,
                        next_url_paths=next_url_paths,
                        truncated=page_number > 0 or bool(next_url_paths),
                        raw_rows=raw_rows,
                        parsed_rows=parsed_rows,
                        duplicate_same_identity_rows=duplicate_same_identity_rows,
                    )
                    return _parse_error_response(
                        lineage=lineage,
                        endpoint=endpoint,
                        message="Polygon tickers response contains identity-less ticker row",
                    )
                identity_hash = _ticker_reference_identity_hash(parsed)
                previous_hash = seen_identity_hashes.get(parsed.ticker)
                if previous_hash is None:
                    seen_identity_hashes[parsed.ticker] = identity_hash
                elif previous_hash == identity_hash:
                    duplicate_same_identity_rows += 1
                else:
                    lineage = _ticker_reference_lineage(
                        first_lineage=first_lineage,
                        page_hashes=page_hashes,
                        page_count=page_number + 1,
                        next_url_paths=next_url_paths,
                        truncated=True,
                        raw_rows=raw_rows,
                        parsed_rows=parsed_rows,
                        duplicate_same_identity_rows=duplicate_same_identity_rows,
                        duplicate_conflict_rows=1,
                    )
                    return AdapterResponse(
                        data=None,
                        lineage=lineage,
                        error=ProviderError(
                            provider=PROVIDER,
                            endpoint=endpoint,
                            status_code=200,
                            error_type="identity_conflict",
                            message="Polygon tickers response contains conflicting duplicate ticker identity",
                            retryable=False,
                        ),
                    )
                parsed_results.append(parsed)
                parsed_rows += 1

            pages.append(PolygonTickerReferencePage(
                results=parsed_results,
                lineage=resp.lineage,
                request_params=request_params,
                page_number=page_number,
                next_url=_safe_url_path(next_url),
                raw_payload=_sanitized_page_payload(payload),
            ))

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
                    lineage = _ticker_reference_lineage(
                        first_lineage=first_lineage,
                        page_hashes=page_hashes,
                        page_count=page_number + 1,
                        next_url_paths=next_url_paths,
                        truncated=True,
                        raw_rows=raw_rows,
                        parsed_rows=parsed_rows,
                        duplicate_same_identity_rows=duplicate_same_identity_rows,
                    )
                    return AdapterResponse(
                        data=None,
                        lineage=lineage,
                        error=ProviderError(
                            provider=PROVIDER,
                            endpoint=endpoint,
                            status_code=200,
                            error_type="pagination",
                            message="Polygon tickers pagination next_url rejected",
                            retryable=False,
                        ),
                    )

            page_number += 1
            if not next_url:
                break
            if page_cap is not None and page_number >= page_cap:
                lineage = _ticker_reference_lineage(
                    first_lineage=first_lineage,
                    page_hashes=page_hashes,
                    page_count=page_number,
                    next_url_paths=next_url_paths,
                    truncated=True,
                    raw_rows=raw_rows,
                    parsed_rows=parsed_rows,
                    duplicate_same_identity_rows=duplicate_same_identity_rows,
                )
                return AdapterResponse(
                    data=None,
                    lineage=lineage,
                    error=ProviderError(
                        provider=PROVIDER,
                        endpoint=endpoint,
                        status_code=200,
                        error_type="pagination",
                        message="Polygon tickers pagination exceeded max_pages",
                        retryable=False,
                    ),
                )

        lineage = _ticker_reference_lineage(
            first_lineage=first_lineage,
            page_hashes=page_hashes,
            page_count=page_number,
            next_url_paths=next_url_paths,
            truncated=False,
            raw_rows=raw_rows,
            parsed_rows=parsed_rows,
            duplicate_same_identity_rows=duplicate_same_identity_rows,
        )
        return AdapterResponse(
            data=pages,
            lineage=lineage,
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

        normalized_ticker, ticker_error = _normalize_polygon_path_ticker(
            ticker,
            endpoint=TICKERS_ENDPOINT,
            asof=asof,
        )
        if ticker_error is not None:
            return ticker_error  # type: ignore[return-value]

        endpoint = f"{TICKERS_ENDPOINT}/{normalized_ticker}"
        date_error = _validate_iso_date_params(
            endpoint=endpoint,
            date_fields={"date": date_str},
            asof=asof,
        )
        if date_error is not None:
            return date_error  # type: ignore[return-value]

        params: Dict[str, Any] = {}
        date_text = _str_or_none(date_str)
        if date_text:
            params["date"] = date_text
        resp = self._request(endpoint, params=params or None, asof=asof)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        if not isinstance(resp.data, dict) or "results" not in resp.data:
            return _parse_error_response(
                lineage=resp.lineage,
                endpoint=endpoint,
                message="Polygon ticker details response missing results object",
            )
        r = resp.data.get("results")
        if r is None or r == {}:
            return AdapterResponse(data=None, lineage=resp.lineage)
        if not isinstance(r, dict):
            return _parse_error_response(
                lineage=resp.lineage,
                endpoint=endpoint,
                message="Polygon ticker details response missing results object",
            )
        if not _ticker_detail_has_known_field(r):
            return _parse_error_response(
                lineage=resp.lineage,
                endpoint=endpoint,
                message="Polygon ticker details response missing usable identity fields",
            )
        provider_ticker = _str_or_none(r.get("ticker"))
        if provider_ticker and provider_ticker.upper() != normalized_ticker:
            return _parse_error_response(
                lineage=resp.lineage,
                endpoint=endpoint,
                message="Polygon ticker details response ticker mismatch",
            )

        detail = PolygonTickerDetail(
            ticker=provider_ticker.upper() if provider_ticker else normalized_ticker,
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
            raw=deepcopy(r),
        )
        return AdapterResponse(data=detail, lineage=resp.lineage)

    # --- Ticker events ---

    def get_ticker_events(
        self,
        identifier: Any,
        *,
        types: Any = "ticker_change",
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[PolygonTickerEvent]]:
        """Fetch ticker events as targeted identity cross-check evidence only.

        ``identifier`` can be a ticker, CUSIP, or Composite FIGI.
        """

        normalized_identifier, identifier_error = _normalize_ticker_event_identifier(
            identifier,
            asof=asof,
        )
        if identifier_error is not None:
            return identifier_error  # type: ignore[return-value]

        normalized_types, types_error = _normalize_ticker_event_types(types, asof=asof)
        if types_error is not None:
            return types_error  # type: ignore[return-value]

        endpoint = f"{TICKER_EVENTS_ENDPOINT_PREFIX}/{normalized_identifier}/events"
        params: Dict[str, Any] = {"types": normalized_types}
        resp = self._request(endpoint, params=params, asof=asof)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        payload = resp.data
        if not isinstance(payload, dict):
            return _ticker_event_parse_error(
                lineage=resp.lineage,
                endpoint=endpoint,
                identifier=normalized_identifier,
            )
        results = payload.get("results")
        if not isinstance(results, dict) or not isinstance(results.get("events"), list):
            return _ticker_event_parse_error(
                lineage=resp.lineage,
                endpoint=endpoint,
                identifier=normalized_identifier,
            )
        if _str_or_none(payload.get("next_url")):
            lineage = _ticker_event_base_lineage(resp.lineage, normalized_identifier)
            return AdapterResponse(
                data=None,
                lineage=lineage,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=200,
                    error_type="pagination",
                    message="Polygon ticker-events pagination is not supported",
                    retryable=False,
                ),
            )

        events = []
        raw_events = results["events"]
        raw_rows = 0
        for ev in raw_events:
            raw_rows += 1
            if not isinstance(ev, dict):
                continue
            parsed = _parse_ticker_event_row(normalized_identifier, ev)
            if parsed is not None:
                events.append(parsed)

        lineage = _ticker_event_row_lineage(
            resp.lineage,
            identifier=normalized_identifier,
            raw_rows=raw_rows,
            events=events,
        )
        return AdapterResponse(data=events, lineage=lineage)

    # --- Market data ---

    def get_daily_bars(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        limit: int = 5000,
        *,
        adjusted: bool = True,
    ) -> AdapterResponse[List[PolygonBar]]:
        """Fetch daily aggregate bars for a ticker and date range."""

        endpoint_prefix = "/v2/aggs/ticker"
        normalized_ticker, ticker_error = _normalize_polygon_path_ticker(
            ticker,
            endpoint=endpoint_prefix,
        )
        if ticker_error is not None:
            return ticker_error  # type: ignore[return-value]
        date_error = _validate_iso_date_params(
            endpoint=endpoint_prefix,
            date_fields={"from_date": from_date, "to_date": to_date},
            asof=None,
        )
        if date_error is not None:
            return date_error  # type: ignore[return-value]
        from_text = _str_or_none(from_date) or ""
        to_text = _str_or_none(to_date) or ""
        from_parsed = date.fromisoformat(from_text)
        to_parsed = date.fromisoformat(to_text)
        if from_parsed > to_parsed:
            return _provider_error_response(
                endpoint=endpoint_prefix,
                error_type="validation",
                message="Polygon daily bars from_date must be on or before to_date",
                retryable=False,
            )

        endpoint = f"/v2/aggs/ticker/{normalized_ticker}/range/1/day/{from_text}/{to_text}"
        params: Dict[str, Any] = {
            "limit": limit,
            "sort": "asc",
            "adjusted": str(bool(adjusted)).lower(),
        }
        resp = self._request(endpoint, params=params)
        if not resp.ok:
            return AdapterResponse(
                data=resp.data,
                lineage=_polygon_daily_bar_row_lineage(
                    resp.lineage,
                    0,
                    0,
                    adjusted=adjusted,
                ),
                rate_limit=resp.rate_limit,
                error=resp.error,
            )  # type: ignore[arg-type]

        if not isinstance(resp.data, dict):
            return _parse_error_response(
                lineage=_polygon_daily_bar_row_lineage(
                    resp.lineage,
                    0,
                    0,
                    adjusted=adjusted,
                ),
                endpoint=endpoint,
                message="Polygon daily bars response missing results list",
            )
        if "results" not in resp.data:
            if resp.data.get("resultsCount") == 0 or resp.data.get("queryCount") == 0:
                lineage = _polygon_daily_bar_row_lineage(
                    resp.lineage,
                    0,
                    0,
                    adjusted=adjusted,
                )
                return AdapterResponse(data=[], lineage=lineage)
            return _parse_error_response(
                lineage=_polygon_daily_bar_row_lineage(
                    resp.lineage,
                    0,
                    0,
                    adjusted=adjusted,
                ),
                endpoint=endpoint,
                message="Polygon daily bars response missing results list",
            )
        results_list = resp.data.get("results")
        if not isinstance(results_list, list):
            return _parse_error_response(
                lineage=_polygon_daily_bar_row_lineage(
                    resp.lineage,
                    0,
                    0,
                    adjusted=adjusted,
                ),
                endpoint=endpoint,
                message="Polygon daily bars response missing results list",
            )

        bars = []
        raw_rows = 0
        for row in results_list:
            raw_rows += 1
            if not isinstance(row, dict):
                lineage = _polygon_daily_bar_row_lineage(
                    resp.lineage,
                    raw_rows,
                    len(bars),
                    adjusted=adjusted,
                )
                return _parse_error_response(
                    lineage=lineage,
                    endpoint=endpoint,
                    message="Polygon daily bars response contains malformed result row",
                )
            parsed = _parse_daily_bar_row(row)
            if parsed is None:
                lineage = _polygon_daily_bar_row_lineage(
                    resp.lineage,
                    raw_rows,
                    len(bars),
                    adjusted=adjusted,
                )
                return _parse_error_response(
                    lineage=lineage,
                    endpoint=endpoint,
                    message="Polygon daily bars response contains malformed result row",
                )
            bars.append(parsed)
        lineage = _polygon_daily_bar_row_lineage(
            resp.lineage,
            raw_rows,
            len(bars),
            adjusted=adjusted,
        )
        return AdapterResponse(data=bars, lineage=lineage)

    def get_today_minute_aggs(
        self,
        ticker: str,
        trading_date: str,
        limit: int = 50000,
        *,
        adjusted: bool = True,
    ) -> AdapterResponse[List[PolygonBar]]:
        """Fetch today's minute aggregate bars for one ticker."""

        return self.get_minute_aggs(
            ticker=ticker,
            from_date=trading_date,
            to_date=trading_date,
            limit=limit,
            adjusted=adjusted,
        )

    def get_minute_aggs(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        limit: int = 50000,
        *,
        adjusted: bool = True,
    ) -> AdapterResponse[List[PolygonBar]]:
        """Fetch minute aggregate bars for one ticker and date range."""

        endpoint_prefix = "/v2/aggs/ticker"
        normalized_ticker, ticker_error = _normalize_polygon_path_ticker(
            ticker,
            endpoint=endpoint_prefix,
        )
        if ticker_error is not None:
            return ticker_error  # type: ignore[return-value]
        date_error = _validate_iso_date_params(
            endpoint=endpoint_prefix,
            date_fields={"from_date": from_date, "to_date": to_date},
            asof=None,
        )
        if date_error is not None:
            return date_error  # type: ignore[return-value]

        from_text = _str_or_none(from_date) or ""
        to_text = _str_or_none(to_date) or ""
        if date.fromisoformat(from_text) > date.fromisoformat(to_text):
            return _provider_error_response(
                endpoint=endpoint_prefix,
                error_type="validation",
                message="Polygon minute aggs from_date must be on or before to_date",
                retryable=False,
            )
        endpoint = (
            f"/v2/aggs/ticker/{normalized_ticker}/range/1/minute/"
            f"{from_text}/{to_text}"
        )
        params: Dict[str, Any] = {
            "limit": limit,
            "sort": "asc",
            "adjusted": str(bool(adjusted)).lower(),
        }
        return self._parse_aggregate_bar_response(
            endpoint=endpoint,
            params=params,
            adjusted=adjusted,
            feed_label="minute aggs",
        )

    def get_grouped_daily_aggs(
        self,
        trading_date: str,
        *,
        adjusted: bool = True,
    ) -> AdapterResponse[List[PolygonGroupedDailyBar]]:
        """Fetch grouped daily aggregate bars for all U.S. stocks on one date."""

        endpoint_prefix = GROUPED_DAILY_AGGS_ENDPOINT_PREFIX
        date_error = _validate_iso_date_params(
            endpoint=endpoint_prefix,
            date_fields={"trading_date": trading_date},
            asof=None,
        )
        if date_error is not None:
            return date_error  # type: ignore[return-value]
        date_text = _str_or_none(trading_date) or ""
        endpoint = f"{GROUPED_DAILY_AGGS_ENDPOINT_PREFIX}/{date_text}"
        params: Dict[str, Any] = {"adjusted": str(bool(adjusted)).lower()}
        resp = self._request(endpoint, params=params)
        if not resp.ok:
            return AdapterResponse(
                data=resp.data,
                lineage=_polygon_daily_bar_row_lineage(
                    resp.lineage,
                    0,
                    0,
                    adjusted=adjusted,
                ),
                rate_limit=resp.rate_limit,
                error=resp.error,
            )  # type: ignore[arg-type]
        if not isinstance(resp.data, dict):
            return _parse_error_response(
                lineage=_polygon_daily_bar_row_lineage(
                    resp.lineage,
                    0,
                    0,
                    adjusted=adjusted,
                ),
                endpoint=endpoint,
                message="Polygon grouped daily aggs response missing results list",
            )
        results_list = resp.data.get("results")
        if results_list is None and (
            resp.data.get("resultsCount") == 0 or resp.data.get("queryCount") == 0
        ):
            return AdapterResponse(
                data=[],
                lineage=_polygon_daily_bar_row_lineage(
                    resp.lineage,
                    0,
                    0,
                    adjusted=adjusted,
                ),
            )
        if not isinstance(results_list, list):
            return _parse_error_response(
                lineage=_polygon_daily_bar_row_lineage(
                    resp.lineage,
                    0,
                    0,
                    adjusted=adjusted,
                ),
                endpoint=endpoint,
                message="Polygon grouped daily aggs response missing results list",
            )
        bars: List[PolygonGroupedDailyBar] = []
        raw_rows = 0
        for row in results_list:
            raw_rows += 1
            if not isinstance(row, dict):
                continue
            parsed = _parse_grouped_daily_bar_row(row)
            if parsed is not None:
                bars.append(parsed)
        lineage = _polygon_daily_bar_row_lineage(
            resp.lineage,
            raw_rows,
            len(bars),
            adjusted=adjusted,
        )
        return AdapterResponse(data=bars, lineage=lineage)

    def get_full_market_snapshot(self) -> AdapterResponse[List[PolygonSnapshotTicker]]:
        """Fetch the delayed full-market snapshot for U.S. stocks."""

        resp = self._request(FULL_MARKET_SNAPSHOT_ENDPOINT)
        if not resp.ok:
            return resp  # type: ignore[return-value]
        payload = resp.data
        if not isinstance(payload, dict) or not isinstance(payload.get("tickers"), list):
            return _parse_error_response(
                lineage=_corporate_action_row_lineage(resp.lineage, 0, 0),
                endpoint=FULL_MARKET_SNAPSHOT_ENDPOINT,
                message="Polygon full-market snapshot response missing tickers list",
            )
        rows: List[PolygonSnapshotTicker] = []
        raw_rows = 0
        for row in payload["tickers"]:
            raw_rows += 1
            if not isinstance(row, dict):
                continue
            parsed = _parse_snapshot_ticker(row)
            if parsed is not None:
                rows.append(parsed)
        lineage = _corporate_action_row_lineage(resp.lineage, raw_rows, len(rows))
        return AdapterResponse(data=rows, lineage=lineage)

    def _parse_aggregate_bar_response(
        self,
        *,
        endpoint: str,
        params: Dict[str, Any],
        adjusted: bool,
        feed_label: str,
    ) -> AdapterResponse[List[PolygonBar]]:
        resp = self._request(endpoint, params=params)
        if not resp.ok:
            return AdapterResponse(
                data=resp.data,
                lineage=_polygon_daily_bar_row_lineage(
                    resp.lineage,
                    0,
                    0,
                    adjusted=adjusted,
                ),
                rate_limit=resp.rate_limit,
                error=resp.error,
            )  # type: ignore[arg-type]
        if not isinstance(resp.data, dict):
            return _parse_error_response(
                lineage=_polygon_daily_bar_row_lineage(
                    resp.lineage,
                    0,
                    0,
                    adjusted=adjusted,
                ),
                endpoint=endpoint,
                message=f"Polygon {feed_label} response missing results list",
            )
        results_list = resp.data.get("results")
        if results_list is None and (
            resp.data.get("resultsCount") == 0 or resp.data.get("queryCount") == 0
        ):
            return AdapterResponse(
                data=[],
                lineage=_polygon_daily_bar_row_lineage(
                    resp.lineage,
                    0,
                    0,
                    adjusted=adjusted,
                ),
            )
        if not isinstance(results_list, list):
            return _parse_error_response(
                lineage=_polygon_daily_bar_row_lineage(
                    resp.lineage,
                    0,
                    0,
                    adjusted=adjusted,
                ),
                endpoint=endpoint,
                message=f"Polygon {feed_label} response missing results list",
            )

        bars: List[PolygonBar] = []
        raw_rows = 0
        for row in results_list:
            raw_rows += 1
            if not isinstance(row, dict):
                continue
            parsed = _parse_daily_bar_row(row)
            if parsed is not None:
                bars.append(parsed)
        lineage = _polygon_daily_bar_row_lineage(
            resp.lineage,
            raw_rows,
            len(bars),
            adjusted=adjusted,
        )
        return AdapterResponse(data=bars, lineage=lineage)


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _iso_date_str_or_none(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if len(text) != 10:
        return None
    try:
        date.fromisoformat(text)
    except ValueError:
        return None
    return text


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


def _parse_error_response(
    *,
    lineage: LineageMeta,
    endpoint: str,
    message: str,
) -> AdapterResponse[Any]:
    return AdapterResponse(
        data=None,
        lineage=lineage,
        error=ProviderError(
            provider=PROVIDER,
            endpoint=endpoint,
            status_code=200,
            error_type="parse",
            message=message,
            retryable=False,
        ),
    )


def _normalize_polygon_path_ticker(
    value: Any,
    *,
    endpoint: str,
    asof: Optional[datetime] = None,
) -> tuple[Optional[str], Optional[AdapterResponse[Any]]]:
    if not isinstance(value, str):
        return None, _provider_error_response(
            endpoint=endpoint,
            error_type="validation",
            message="Polygon ticker must be a nonblank path-safe string",
            retryable=False,
            asof=asof,
        )
    text = value.strip()
    if not text:
        return None, _provider_error_response(
            endpoint=endpoint,
            error_type="validation",
            message="Polygon ticker must be a nonblank path-safe string",
            retryable=False,
            asof=asof,
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        return None, _provider_error_response(
            endpoint=endpoint,
            error_type="validation",
            message="Polygon ticker contains unsafe characters",
            retryable=False,
            asof=asof,
        )
    normalized = text.upper()
    if (
        len(normalized) > PATH_TICKER_MAX_LENGTH
        or any(char not in PATH_TICKER_ALLOWED_CHARS for char in normalized)
        or not any(char.isalnum() for char in normalized)
        or ".." in normalized
        or normalized[0] in ".-"
        or normalized[-1] in ".-"
    ):
        return None, _provider_error_response(
            endpoint=endpoint,
            error_type="validation",
            message="Polygon ticker contains unsafe characters",
            retryable=False,
            asof=asof,
        )
    return normalized, None


def _normalize_ticker_event_identifier(
    identifier: Any,
    *,
    asof: Optional[datetime],
) -> tuple[Optional[str], Optional[AdapterResponse[Any]]]:
    if not isinstance(identifier, str):
        return None, _ticker_event_validation_error(
            message="Polygon ticker-events identifier must be a nonblank string",
            asof=asof,
        )
    text = identifier.strip()
    if not text:
        return None, _ticker_event_validation_error(
            message="Polygon ticker-events identifier must be a nonblank string",
            asof=asof,
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        return None, _ticker_event_validation_error(
            message="Polygon ticker-events identifier contains unsafe characters",
            asof=asof,
        )
    normalized = text.upper()
    if (
        len(normalized) > TICKER_EVENT_IDENTIFIER_MAX_LENGTH
        or any(char not in TICKER_EVENT_IDENTIFIER_ALLOWED_CHARS for char in normalized)
        or not any(char.isalnum() for char in normalized)
        or ".." in normalized
        or normalized[0] in "._-"
        or normalized[-1] in "._-"
    ):
        return None, _ticker_event_validation_error(
            message="Polygon ticker-events identifier contains unsafe characters",
            asof=asof,
        )
    return normalized, None


def _normalize_ticker_event_types(
    types: Any,
    *,
    asof: Optional[datetime],
) -> tuple[Optional[str], Optional[AdapterResponse[Any]]]:
    if isinstance(types, str):
        items = [types]
    elif isinstance(types, (list, tuple, set)):
        items = list(types)
    else:
        return None, _ticker_event_validation_error(
            message="Polygon ticker-events types must be strings",
            asof=asof,
        )

    values: List[str] = []
    for item in items:
        if not isinstance(item, str):
            return None, _ticker_event_validation_error(
                message="Polygon ticker-events types must be strings",
                asof=asof,
            )
        text = item.strip()
        if not text:
            return None, _ticker_event_validation_error(
                message="Polygon ticker-events types must be nonblank",
                asof=asof,
            )
        if any(char in text for char in [",", "/", "?", "#", "&"]):
            return None, _ticker_event_validation_error(
                message="Polygon ticker-events type contains unsafe characters",
                asof=asof,
            )
        value = text.lower()
        if value not in TICKER_EVENT_ALLOWED_TYPES:
            return None, _ticker_event_validation_error(
                message="Polygon ticker-events type is not supported",
                asof=asof,
            )
        if value not in values:
            values.append(value)

    if not values:
        return None, _ticker_event_validation_error(
            message="Polygon ticker-events types must be nonblank",
            asof=asof,
        )
    return ",".join(values), None


def _ticker_event_validation_error(
    *,
    message: str,
    asof: Optional[datetime],
) -> AdapterResponse[Any]:
    return _provider_error_response(
        endpoint=f"{TICKER_EVENTS_ENDPOINT_PREFIX}/events",
        error_type="validation",
        message=message,
        retryable=False,
        asof=asof,
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
    range_error = _validate_iso_date_range_order(
        endpoint=endpoint,
        start_label=f"{date_field}_from",
        start_value=date_from,
        end_label=f"{date_field}_to",
        end_value=date_to,
        asof=asof,
    )
    if range_error is not None:
        return range_error
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
    range_error = _validate_iso_date_range_order(
        endpoint=endpoint,
        start_label="date_from",
        start_value=date_from,
        end_label="date_to",
        end_value=date_to,
        asof=asof,
    )
    if range_error is not None:
        return range_error
    if _ticker_param(ticker) is None and not (date_value or (date_from and date_to)):
        return _provider_error_response(
            endpoint=endpoint,
            error_type="validation",
            message="Polygon feed broad query requires an exact date or bounded date window",
            retryable=False,
            asof=asof,
        )
    return None


def _validate_polygon_news_query(
    *,
    ticker: Optional[str],
    tickers: Optional[List[str]],
    published_utc: Optional[str],
    published_utc_from: Optional[str],
    published_utc_to: Optional[str],
    max_pages: Any,
    asof: Optional[datetime],
) -> Optional[AdapterResponse[Any]]:
    try:
        page_cap = int(max_pages)
    except (TypeError, ValueError):
        return _provider_error_response(
            endpoint=NEWS_ENDPOINT,
            error_type="validation",
            message="Polygon news max_pages must be a positive integer",
            retryable=False,
            asof=asof,
        )
    if page_cap < 1:
        return _provider_error_response(
            endpoint=NEWS_ENDPOINT,
            error_type="validation",
            message="Polygon news max_pages must be a positive integer",
            retryable=False,
            asof=asof,
        )
    date_error = _validate_iso_date_or_datetime_params(
        endpoint=NEWS_ENDPOINT,
        date_fields={
            "published_utc": published_utc,
            "published_utc_from": published_utc_from,
            "published_utc_to": published_utc_to,
        },
        asof=asof,
    )
    if date_error is not None:
        return date_error
    range_error = _validate_news_datetime_range_order(
        endpoint=NEWS_ENDPOINT,
        start_label="published_utc_from",
        start_value=published_utc_from,
        end_label="published_utc_to",
        end_value=published_utc_to,
        asof=asof,
    )
    if range_error is not None:
        return range_error

    ticker_error = _validate_news_ticker_inputs(
        ticker=ticker,
        tickers=tickers,
        asof=asof,
    )
    if ticker_error is not None:
        return ticker_error

    ticker_values = _news_ticker_values(ticker, tickers)
    if len(ticker_values) > 1:
        return _provider_error_response(
            endpoint=NEWS_ENDPOINT,
            error_type="validation",
            message="Polygon news supports one ticker per request",
            retryable=False,
            asof=asof,
        )
    has_exact_date = _str_or_none(published_utc) is not None
    has_bounded_date_range = (
        _str_or_none(published_utc_from) is not None
        and _str_or_none(published_utc_to) is not None
    )
    if not ticker_values and not (has_exact_date or has_bounded_date_range):
        return _provider_error_response(
            endpoint=NEWS_ENDPOINT,
            error_type="validation",
            message="Polygon news broad query requires an exact published_utc or bounded published_utc window",
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


def _news_params(
    *,
    ticker: Optional[str],
    tickers: Optional[List[str]],
    published_utc: Optional[str],
    published_utc_from: Optional[str],
    published_utc_to: Optional[str],
    limit: int,
    sort: Optional[str],
    order: Optional[str],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"limit": _limited_int(limit, maximum=1000)}
    ticker_param = _news_ticker_param(ticker, tickers)
    if ticker_param:
        params["ticker"] = ticker_param
    if published_utc:
        params["published_utc"] = published_utc
    else:
        if published_utc_from:
            params["published_utc.gte"] = published_utc_from
        if published_utc_to:
            params["published_utc.lte"] = published_utc_to
    sort_text = _str_or_none(sort)
    if sort_text:
        params["sort"] = sort_text
    order_text = _str_or_none(order)
    if order_text:
        params["order"] = order_text.lower()
    return params


def _news_ticker_param(
    ticker: Optional[str],
    tickers: Optional[List[str]],
) -> Optional[str]:
    values = _news_ticker_values(ticker, tickers)
    return values[0] if values else None


def _validate_news_ticker_inputs(
    *,
    ticker: Optional[str],
    tickers: Optional[List[str]],
    asof: Optional[datetime],
) -> Optional[AdapterResponse[Any]]:
    values = []
    for item in _news_ticker_items(ticker, tickers):
        if not isinstance(item, str):
            return _invalid_news_ticker_response(asof=asof)
        text = item.strip()
        if not text:
            continue
        if "," in text:
            return _invalid_news_ticker_response(asof=asof)
        value = text.upper()
        if value not in values:
            values.append(value)
    if len(values) > 1:
        return _invalid_news_ticker_response(asof=asof)
    return None


def _invalid_news_ticker_response(*, asof: Optional[datetime]) -> AdapterResponse[Any]:
    return _provider_error_response(
        endpoint=NEWS_ENDPOINT,
        error_type="validation",
        message="Polygon news supports one string ticker per request",
        retryable=False,
        asof=asof,
    )


def _news_ticker_items(
    ticker: Optional[str],
    tickers: Optional[List[str]],
) -> List[Any]:
    items: List[Any] = []
    if ticker is not None:
        items.append(ticker)
    if tickers is not None:
        if isinstance(tickers, str):
            items.append(tickers)
        elif isinstance(tickers, (list, tuple, set)):
            items.extend(tickers)
        else:
            items.append(tickers)
    return items


def _news_ticker_values(
    ticker: Optional[str],
    tickers: Optional[List[str]],
) -> List[str]:
    normalized: List[str] = []
    for item in _news_ticker_items(ticker, tickers):
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or "," in text:
            continue
        value = text.upper()
        if value not in normalized:
            normalized.append(value)
    return normalized


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
    if text.upper().startswith("CIK"):
        text = text[3:].strip()
    if (
        not text
        or not text.isascii()
        or not text.isdigit()
        or len(text) > 10
        or set(text) == {"0"}
    ):
        return None
    return text.zfill(10)


def _safe_url_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        return parsed.path
    if parsed.query:
        return parsed.path or None
    return value


def _sanitized_page_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = deepcopy(payload)
    next_url = _str_or_none(sanitized.get("next_url"))
    if next_url:
        safe_path = _safe_url_path(next_url)
        if safe_path:
            sanitized["next_url"] = safe_path
        else:
            sanitized.pop("next_url", None)
    return sanitized


def _ticker_detail_has_known_field(row: Dict[str, Any]) -> bool:
    string_fields = {
        "ticker",
        "name",
        "composite_figi",
        "share_class_figi",
        "list_date",
        "delisted_utc",
        "primary_exchange",
        "type",
    }
    if any(_str_or_none(row.get(key)) is not None for key in string_fields):
        return True
    if _normalized_cik(row.get("cik")) is not None:
        return True
    return row.get("active") is not None


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


def _polygon_daily_bar_row_lineage(
    lineage: LineageMeta,
    raw_rows: int,
    parsed_rows: int,
    *,
    adjusted: bool,
) -> LineageMeta:
    lineage = _corporate_action_row_lineage(lineage, raw_rows, parsed_rows)
    flags = dict(lineage.data_quality_flags or {})
    adjusted_bool = bool(adjusted)
    flags.update(
        {
            "adjusted": adjusted_bool,
            "requested_adjusted": adjusted_bool,
            "price_basis": (
                DAILY_BAR_SPLIT_ADJUSTED_PRICE_BASIS
                if adjusted_bool
                else DAILY_BAR_UNADJUSTED_PRICE_BASIS
            ),
            "adjustment_basis": "split_adjusted" if adjusted_bool else "unadjusted",
        }
    )
    return replace(lineage, data_quality_flags=flags)


def _ticker_event_base_lineage(
    lineage: LineageMeta,
    identifier: str,
) -> LineageMeta:
    flags = dict(lineage.data_quality_flags or {})
    flags["identifier_queried"] = identifier
    return replace(lineage, data_quality_flags=flags)


def _ticker_event_row_lineage(
    lineage: LineageMeta,
    *,
    identifier: str,
    raw_rows: int,
    events: List[PolygonTickerEvent],
) -> LineageMeta:
    lineage = _corporate_action_row_lineage(lineage, raw_rows, len(events))
    flags = dict(lineage.data_quality_flags or {})
    flags["identifier_queried"] = identifier
    flags["event_types_present"] = sorted(
        {event.event_type for event in events if event.event_type}
    )
    flags["identity_continuity_proved_rows"] = sum(
        1 for event in events if event.identity_continuity_status == "proved"
    )
    flags["identity_continuity_unproven_rows"] = sum(
        1 for event in events if event.identity_continuity_status == "unproven"
    )
    flags["identity_continuity_mismatch_rows"] = sum(
        1 for event in events if event.identity_continuity_status == "mismatch"
    )
    flags["identity_continuity_not_applicable_rows"] = sum(
        1 for event in events if event.identity_continuity_status == "not_applicable"
    )
    return replace(lineage, data_quality_flags=flags)


def _ticker_reference_lineage(
    *,
    first_lineage: Optional[LineageMeta],
    page_hashes: List[str],
    page_count: int,
    next_url_paths: List[str],
    truncated: bool,
    raw_rows: int,
    parsed_rows: int,
    duplicate_same_identity_rows: int,
    duplicate_conflict_rows: int = 0,
) -> LineageMeta:
    lineage = _corporate_action_lineage(
        first_lineage=first_lineage,
        page_hashes=page_hashes,
        page_count=page_count,
        next_url_paths=next_url_paths,
        truncated=truncated,
    )
    flags = dict(lineage.data_quality_flags or {})
    flags.update({
        "raw_rows": raw_rows,
        "parsed_rows": parsed_rows,
        "skipped_rows": 0,
        "duplicate_same_identity_rows": duplicate_same_identity_rows,
        "duplicate_conflict_rows": duplicate_conflict_rows,
    })
    if next_url_paths:
        flags["paginated"] = True
    return replace(lineage, data_quality_flags=flags)


def _ticker_event_parse_error(
    *,
    lineage: LineageMeta,
    endpoint: str,
    identifier: str,
) -> AdapterResponse[Any]:
    return AdapterResponse(
        data=None,
        lineage=_ticker_event_base_lineage(lineage, identifier),
        error=ProviderError(
            provider=PROVIDER,
            endpoint=endpoint,
            status_code=200,
            error_type="parse",
            message="Polygon ticker-events response missing results.events list",
            retryable=False,
        ),
    )


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


def _validate_iso_date_range_order(
    *,
    endpoint: str,
    start_label: str,
    start_value: Optional[str],
    end_label: str,
    end_value: Optional[str],
    asof: Optional[datetime],
) -> Optional[AdapterResponse[Any]]:
    start_text = _iso_date_str_or_none(start_value)
    end_text = _iso_date_str_or_none(end_value)
    if start_text is None or end_text is None:
        return None
    if date.fromisoformat(start_text) > date.fromisoformat(end_text):
        return _provider_error_response(
            endpoint=endpoint,
            error_type="validation",
            message=f"Polygon feed {start_label} must be on or before {end_label}",
            retryable=False,
            asof=asof,
        )
    return None


def _validate_news_datetime_range_order(
    *,
    endpoint: str,
    start_label: str,
    start_value: Optional[str],
    end_label: str,
    end_value: Optional[str],
    asof: Optional[datetime],
) -> Optional[AdapterResponse[Any]]:
    start_dt = _parse_iso_date_or_datetime_for_order(start_value)
    end_dt = _parse_iso_date_or_datetime_for_order(end_value)
    if start_dt is None or end_dt is None:
        return None
    if start_dt > end_dt:
        return _provider_error_response(
            endpoint=endpoint,
            error_type="validation",
            message=f"Polygon news {start_label} must be on or before {end_label}",
            retryable=False,
            asof=asof,
        )
    return None


def _parse_iso_date_or_datetime_for_order(value: Optional[str]) -> Optional[datetime]:
    text = _str_or_none(value)
    if text is None:
        return None
    if len(text) == 10:
        try:
            return datetime.combine(date.fromisoformat(text), datetime.min.time(), tzinfo=timezone.utc)
        except ValueError:
            return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _validate_iso_date_or_datetime_params(
    *,
    endpoint: str,
    date_fields: Dict[str, Optional[str]],
    asof: Optional[datetime],
) -> Optional[AdapterResponse[Any]]:
    for label, value in date_fields.items():
        if value is None:
            continue
        text = _str_or_none(value)
        if text is None:
            return _invalid_news_datetime_response(endpoint=endpoint, label=label, asof=asof)
        if len(text) == 10:
            try:
                date.fromisoformat(text)
            except ValueError:
                return _invalid_news_datetime_response(endpoint=endpoint, label=label, asof=asof)
            continue
        if "T" not in text:
            return _invalid_news_datetime_response(endpoint=endpoint, label=label, asof=asof)
        date_part = text.split("T", 1)[0]
        if len(date_part) != 10:
            return _invalid_news_datetime_response(endpoint=endpoint, label=label, asof=asof)
        try:
            date.fromisoformat(date_part)
            parsed_dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return _invalid_news_datetime_response(endpoint=endpoint, label=label, asof=asof)
        if parsed_dt.tzinfo is None or parsed_dt.utcoffset() is None:
            return _invalid_news_datetime_response(endpoint=endpoint, label=label, asof=asof)
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


def _invalid_news_datetime_response(
    *,
    endpoint: str,
    label: str,
    asof: Optional[datetime],
) -> AdapterResponse[Any]:
    return _provider_error_response(
        endpoint=endpoint,
        error_type="validation",
        message=f"Polygon news {label} must be YYYY-MM-DD or ISO datetime",
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


def _required_int_or_none(value: Any) -> Optional[int]:
    parsed = _decimal_or_none(value)
    if parsed is None or parsed != parsed.to_integral_value():
        return None
    return int(parsed)


def _required_nonnegative_int_or_none(value: Any) -> Optional[int]:
    parsed = _required_int_or_none(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _required_float_or_none(value: Any) -> Optional[float]:
    parsed = _decimal_or_none(value)
    if parsed is None:
        return None
    return float(parsed)


def _required_nonnegative_float_or_none(value: Any) -> Optional[float]:
    parsed = _required_float_or_none(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _parse_daily_bar_row(row: Dict[str, Any]) -> Optional[PolygonBar]:
    timestamp = _required_nonnegative_int_or_none(row.get("t"))
    open_price = _required_nonnegative_float_or_none(row.get("o"))
    high = _required_nonnegative_float_or_none(row.get("h"))
    low = _required_nonnegative_float_or_none(row.get("l"))
    close = _required_nonnegative_float_or_none(row.get("c"))
    volume = _required_nonnegative_float_or_none(row.get("v"))
    if None in (timestamp, open_price, high, low, close, volume):
        return None
    vwap = (
        _required_nonnegative_float_or_none(row.get("vw"))
        if row.get("vw") is not None
        else None
    )
    transactions = (
        _required_nonnegative_int_or_none(row.get("n"))
        if row.get("n") is not None
        else None
    )
    if (row.get("vw") is not None and vwap is None) or (
        row.get("n") is not None and transactions is None
    ):
        return None
    return PolygonBar(
        timestamp=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=vwap,
        transactions=transactions,
    )


def _parse_grouped_daily_bar_row(row: Dict[str, Any]) -> Optional[PolygonGroupedDailyBar]:
    ticker = _str_or_none(row.get("T"))
    parsed = _parse_daily_bar_row(row)
    if not ticker or parsed is None:
        return None
    return PolygonGroupedDailyBar(
        timestamp=parsed.timestamp,
        open=parsed.open,
        high=parsed.high,
        low=parsed.low,
        close=parsed.close,
        volume=parsed.volume,
        vwap=parsed.vwap,
        transactions=parsed.transactions,
        ticker=ticker.upper(),
    )


def _parse_snapshot_ticker(row: Dict[str, Any]) -> Optional[PolygonSnapshotTicker]:
    ticker = _str_or_none(row.get("ticker"))
    if not ticker:
        return None
    day = row.get("day") if isinstance(row.get("day"), dict) else {}
    prev_day = row.get("prevDay") if isinstance(row.get("prevDay"), dict) else {}
    minute = row.get("min") if isinstance(row.get("min"), dict) else {}
    last_trade = row.get("lastTrade") if isinstance(row.get("lastTrade"), dict) else {}
    return PolygonSnapshotTicker(
        ticker=ticker.upper(),
        day_open=_required_float_or_none(day.get("o")),
        day_high=_required_float_or_none(day.get("h")),
        day_low=_required_float_or_none(day.get("l")),
        day_close=_required_float_or_none(day.get("c")),
        day_volume=_required_nonnegative_float_or_none(day.get("v")),
        prev_day_close=_required_float_or_none(prev_day.get("c")),
        prev_day_volume=_required_nonnegative_float_or_none(prev_day.get("v")),
        minute_timestamp=_required_nonnegative_int_or_none(minute.get("t")),
        minute_open=_required_float_or_none(minute.get("o")),
        minute_high=_required_float_or_none(minute.get("h")),
        minute_low=_required_float_or_none(minute.get("l")),
        minute_close=_required_float_or_none(minute.get("c")),
        minute_volume=_required_nonnegative_float_or_none(minute.get("v")),
        last_trade_price=_required_float_or_none(last_trade.get("p")),
        raw=row,
    )


def _parse_short_interest_row(row: Dict[str, Any]) -> Optional[PolygonShortInterest]:
    ticker = _ticker_param(row.get("ticker"))  # type: ignore[arg-type]
    settlement_date = _iso_date_str_or_none(row.get("settlement_date"))
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
    date_value = _iso_date_str_or_none(row.get("date"))
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
    execution_date = _iso_date_str_or_none(row.get("execution_date"))
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
    ex_dividend_date = _iso_date_str_or_none(row.get("ex_dividend_date"))
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


def _parse_news_article_row(row: Dict[str, Any]) -> Optional[PolygonNewsArticle]:
    article_id = _strict_str_or_none(row.get("id"))
    title = _strict_str_or_none(row.get("title"))
    article_url = _url_str_or_none(row.get("article_url"))
    if not article_id or not title or not article_url:
        return None

    publisher_payload = row.get("publisher")
    publisher = deepcopy(publisher_payload) if isinstance(publisher_payload, dict) else None
    publisher_lookup = publisher or {}

    return PolygonNewsArticle(
        id=article_id,
        title=title,
        article_url=article_url,
        publisher_name=_str_or_none(publisher_lookup.get("name")),
        publisher_homepage_url=_str_or_none(publisher_lookup.get("homepage_url")),
        publisher_logo_url=_str_or_none(publisher_lookup.get("logo_url")),
        publisher_favicon_url=_str_or_none(publisher_lookup.get("favicon_url")),
        author=_str_or_none(row.get("author")),
        amp_url=_str_or_none(row.get("amp_url")),
        image_url=_str_or_none(row.get("image_url")),
        description=_str_or_none(row.get("description")),
        published_utc=_str_or_none(row.get("published_utc")),
        tickers=_string_list(row.get("tickers")),
        keywords=_string_list(row.get("keywords")),
        insights=_dict_list(row.get("insights")),
        publisher=publisher,
        raw=deepcopy(row),
    )


def _strict_str_or_none(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return _str_or_none(value)


def _url_str_or_none(value: Any) -> Optional[str]:
    text = _strict_str_or_none(value)
    if text is None:
        return None
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        return None
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc.strip():
        return None
    return text


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            text = _str_or_none(item)
            if text:
                result.append(text)
        return result
    text = _str_or_none(value)
    return [text] if text else []


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [deepcopy(item) for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [deepcopy(value)]
    return []


def _parse_ticker_event_row(
    identifier_queried: str,
    row: Dict[str, Any],
) -> Optional[PolygonTickerEvent]:
    event_type = _str_or_none(row.get("type") or row.get("event_type"))
    if event_type is None:
        return None
    event_type = event_type.lower()
    if event_type not in TICKER_EVENT_ALLOWED_TYPES:
        return None
    ev_date, date_ok = _ticker_event_iso_date_or_none(row.get("date"))
    event_date_value, event_date_ok = _ticker_event_iso_date_or_none(row.get("event_date"))
    effective_date_value, effective_date_ok = _ticker_event_iso_date_or_none(
        row.get("effective_date")
    )
    effective_utc_value, effective_utc_ok = _ticker_event_effective_utc_date_or_none(
        row.get("effective_utc")
    )
    execution_date_value, execution_date_ok = _ticker_event_iso_date_or_none(
        row.get("execution_date")
    )
    if not (
        date_ok
        and event_date_ok
        and effective_date_ok
        and effective_utc_ok
        and execution_date_ok
    ):
        return None
    event_date = event_date_value or ev_date
    effective_date = (
        effective_date_value
        or effective_utc_value
        or execution_date_value
        or event_date
    )
    ticker_change = row.get("ticker_change")
    if not isinstance(ticker_change, dict):
        ticker_change = {}

    old_ticker = _ticker_param(
        _first_present(
            ticker_change,
            row,
            ["old_ticker", "previous_ticker", "from_ticker", "ticker"],
        )
    )
    new_ticker = _ticker_param(
        _first_present(
            ticker_change,
            row,
            ["new_ticker", "successor_ticker", "to_ticker"],
        )
    )
    ticker = _ticker_param(
        row.get("ticker")
        or ticker_change.get("current_ticker")
        or ticker_change.get("new_ticker")
        or ticker_change.get("ticker")
    )
    if event_type == "ticker_change":
        if not (ev_date or event_date or effective_date):
            return None
        if not (ticker or old_ticker or new_ticker):
            return None

    old_cik = _normalized_cik(
        _first_present(
            ticker_change,
            row,
            ["old_cik", "previous_cik", "from_cik"],
        )
    )
    new_cik = _normalized_cik(
        _first_present(
            ticker_change,
            row,
            ["new_cik", "successor_cik", "current_cik", "to_cik"],
        )
    )
    cik = _normalized_cik(row.get("cik") or ticker_change.get("cik") or new_cik or old_cik)

    old_composite_figi = _upper_str_or_none(
        _first_present(
            ticker_change,
            row,
            [
                "old_composite_figi",
                "previous_composite_figi",
                "from_composite_figi",
            ],
        )
    )
    new_composite_figi = _upper_str_or_none(
        _first_present(
            ticker_change,
            row,
            [
                "new_composite_figi",
                "successor_composite_figi",
                "current_composite_figi",
                "to_composite_figi",
            ],
        )
    )
    composite_figi = _upper_str_or_none(
        row.get("composite_figi")
        or ticker_change.get("composite_figi")
        or new_composite_figi
        or old_composite_figi
    )

    old_share_class_figi = _upper_str_or_none(
        _first_present(
            ticker_change,
            row,
            [
                "old_share_class_figi",
                "previous_share_class_figi",
                "from_share_class_figi",
            ],
        )
    )
    new_share_class_figi = _upper_str_or_none(
        _first_present(
            ticker_change,
            row,
            [
                "new_share_class_figi",
                "successor_share_class_figi",
                "current_share_class_figi",
                "to_share_class_figi",
            ],
        )
    )
    share_class_figi = _upper_str_or_none(
        row.get("share_class_figi")
        or ticker_change.get("share_class_figi")
        or new_share_class_figi
        or old_share_class_figi
    )
    continuity_status = _ticker_event_continuity_status(
        event_type=event_type,
        old_cik=old_cik,
        new_cik=new_cik,
        old_composite_figi=old_composite_figi,
        new_composite_figi=new_composite_figi,
        old_share_class_figi=old_share_class_figi,
        new_share_class_figi=new_share_class_figi,
    )

    return PolygonTickerEvent(
        identifier_queried=identifier_queried,
        event_type=event_type,
        date=ev_date,
        event_date=event_date,
        effective_date=effective_date,
        ticker=ticker,
        old_ticker=old_ticker,
        new_ticker=new_ticker,
        cik=cik,
        old_cik=old_cik,
        new_cik=new_cik,
        composite_figi=composite_figi,
        old_composite_figi=old_composite_figi,
        new_composite_figi=new_composite_figi,
        share_class_figi=share_class_figi,
        old_share_class_figi=old_share_class_figi,
        new_share_class_figi=new_share_class_figi,
        name=_str_or_none(row.get("name") or row.get("company_name")),
        identity_continuity_status=continuity_status,
        raw_event=deepcopy(row),
    )


def _ticker_event_iso_date_or_none(value: Any) -> tuple[Optional[str], bool]:
    if value is None:
        return None, True
    if not isinstance(value, str):
        return None, False
    text = value.strip()
    if not text or len(text) != 10:
        return None, False
    try:
        date.fromisoformat(text)
    except ValueError:
        return None, False
    return text, True


def _ticker_event_effective_utc_date_or_none(value: Any) -> tuple[Optional[str], bool]:
    parsed_date, ok = _ticker_event_iso_date_or_none(value)
    if ok or value is None:
        return parsed_date, ok
    if not isinstance(value, str):
        return None, False
    text = value.strip()
    if not text or "T" not in text:
        return None, False
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None, False
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None, False
    return parsed.astimezone(timezone.utc).date().isoformat(), True


def _first_present(
    primary: Dict[str, Any],
    fallback: Dict[str, Any],
    keys: List[str],
) -> Any:
    for key in keys:
        value = primary.get(key)
        if _str_or_none(value) is not None:
            return value
        value = fallback.get(key)
        if _str_or_none(value) is not None:
            return value
    return None


def _upper_str_or_none(value: Any) -> Optional[str]:
    text = _str_or_none(value)
    return text.upper() if text else None


def _ticker_event_continuity_status(
    *,
    event_type: str,
    old_cik: Optional[str],
    new_cik: Optional[str],
    old_composite_figi: Optional[str],
    new_composite_figi: Optional[str],
    old_share_class_figi: Optional[str],
    new_share_class_figi: Optional[str],
) -> str:
    if event_type != "ticker_change":
        return "not_applicable"
    pairs = [
        (old_cik, new_cik),
        (old_composite_figi, new_composite_figi),
        (old_share_class_figi, new_share_class_figi),
    ]
    present_pairs = [(old, new) for old, new in pairs if old and new]
    if any(old != new for old, new in present_pairs):
        return "mismatch"
    if present_pairs:
        return "proved"
    return "unproven"


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


def _ticker_reference_identity_hash(row: PolygonTickerReference) -> str:
    return stable_hash({
        "ticker": row.ticker,
        "cik": row.cik,
        "composite_figi": row.composite_figi,
        "share_class_figi": row.share_class_figi,
        "active": row.active,
        "primary_exchange": row.primary_exchange,
        "type": row.type,
        "list_date": row.list_date,
        "delisted_utc": row.delisted_utc,
    })
