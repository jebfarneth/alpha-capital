"""
Benzinga adapter.

Supplemental event source for:
  - M&A and acquisition evidence for survivorship/corporate-action review
  - News/WIIMs catalyst context for event diagnostics

Does not write to DB. Returns AdapterResponse with LineageMeta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Sequence, Union

import requests

from alpha.data.config import BenzingaConfig
from alpha.data.contracts import (
    AdapterResponse,
    LineageMeta,
    ProviderError,
    aware_utc_or_none,
    stable_hash,
    utcnow,
)

PROVIDER = "Benzinga"
M_AND_A_ENDPOINT = "/api/v2.1/calendar/ma"
EARNINGS_ENDPOINT = "/api/v2.1/calendar/earnings"
GUIDANCE_ENDPOINT = "/api/v2.1/calendar/guidance"
RATINGS_ENDPOINT = "/api/v2.1/calendar/ratings"
OFFERINGS_ENDPOINT = "/api/v2.1/calendar/offerings"
NEWS_ENDPOINT = "/api/v2/news"
WIIM_CHANNEL = "wiim"


@dataclass
class BenzingaMergerAcquisition:
    """Normalized Benzinga M&A calendar row with raw payload preserved."""

    id: Optional[str]
    target_ticker: Optional[str] = None
    target_name: Optional[str] = None
    target_exchange: Optional[str] = None
    target_cusip: Optional[str] = None
    target_isin: Optional[str] = None
    acquirer_ticker: Optional[str] = None
    acquirer_name: Optional[str] = None
    acquirer_exchange: Optional[str] = None
    acquirer_cusip: Optional[str] = None
    acquirer_isin: Optional[str] = None
    deal_type: Optional[str] = None
    deal_status: Optional[str] = None
    deal_payment_type: Optional[str] = None
    deal_size: Optional[str] = None
    currency: Optional[str] = None
    date: Optional[str] = None
    date_completed: Optional[str] = None
    date_expected: Optional[str] = None
    deal_terms_extra: Optional[str] = None
    notes: Optional[str] = None
    importance: Optional[int] = None
    updated: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class BenzingaNewsArticle:
    """Normalized Benzinga news row with raw payload preserved."""

    id: Optional[str]
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    published: Optional[datetime] = None
    title: Optional[str] = None
    body: Optional[str] = None
    teaser: Optional[str] = None
    url: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    stocks: List[Dict[str, Any]] = field(default_factory=list)
    tickers: List[str] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None


@dataclass
class BenzingaEarnings:
    """Normalized Benzinga earnings calendar row with raw payload preserved."""

    id: Optional[str]
    ticker: Optional[str] = None
    name: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    cusip: Optional[str] = None
    isin: Optional[str] = None
    period: Optional[str] = None
    period_year: Optional[int] = None
    date: Optional[str] = None
    time: Optional[str] = None
    eps: Optional[Decimal] = None
    eps_est: Optional[Decimal] = None
    eps_prior: Optional[Decimal] = None
    eps_surprise: Optional[Decimal] = None
    eps_surprise_percent: Optional[Decimal] = None
    eps_type: Optional[str] = None
    revenue: Optional[Decimal] = None
    revenue_est: Optional[Decimal] = None
    revenue_prior: Optional[Decimal] = None
    revenue_surprise: Optional[Decimal] = None
    revenue_surprise_percent: Optional[Decimal] = None
    revenue_type: Optional[str] = None
    date_confirmed: Optional[bool] = None
    importance: Optional[int] = None
    notes: Optional[str] = None
    updated: Optional[datetime] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class BenzingaGuidance:
    """Normalized Benzinga guidance calendar row with raw payload preserved."""

    id: Optional[str]
    ticker: Optional[str] = None
    name: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    cusip: Optional[str] = None
    period: Optional[str] = None
    period_year: Optional[int] = None
    date: Optional[str] = None
    time: Optional[str] = None
    eps_guidance_est: Optional[Decimal] = None
    eps_guidance_min: Optional[Decimal] = None
    eps_guidance_max: Optional[Decimal] = None
    eps_guidance_prior_min: Optional[Decimal] = None
    eps_guidance_prior_max: Optional[Decimal] = None
    eps_type: Optional[str] = None
    revenue_guidance_est: Optional[Decimal] = None
    revenue_guidance_min: Optional[Decimal] = None
    revenue_guidance_max: Optional[Decimal] = None
    revenue_guidance_prior_min: Optional[Decimal] = None
    revenue_guidance_prior_max: Optional[Decimal] = None
    revenue_type: Optional[str] = None
    is_primary: Optional[bool] = None
    prelim: Optional[bool] = None
    importance: Optional[int] = None
    notes: Optional[str] = None
    updated: Optional[datetime] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class BenzingaRating:
    """Normalized Benzinga analyst rating calendar row with raw payload preserved."""

    id: Optional[str]
    ticker: Optional[str] = None
    name: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    cusip: Optional[str] = None
    isin: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    analyst: Optional[str] = None
    analyst_id: Optional[str] = None
    analyst_name: Optional[str] = None
    firm: Optional[str] = None
    firm_id: Optional[str] = None
    action_company: Optional[str] = None
    action_pt: Optional[str] = None
    rating_current: Optional[str] = None
    rating_prior: Optional[str] = None
    pt_current: Optional[Decimal] = None
    pt_prior: Optional[Decimal] = None
    adjusted_pt_current: Optional[Decimal] = None
    adjusted_pt_prior: Optional[Decimal] = None
    pt_pct_change: Optional[Decimal] = None
    ratings_accuracy: Optional[Decimal] = None
    importance: Optional[int] = None
    notes: Optional[str] = None
    url: Optional[str] = None
    url_calendar: Optional[str] = None
    url_news: Optional[str] = None
    updated: Optional[datetime] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class BenzingaOffering:
    """Normalized Benzinga offering calendar row with raw payload preserved."""

    id: Optional[str]
    ticker: Optional[str] = None
    name: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    cusip: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    offering_type: Optional[str] = None
    price: Optional[Decimal] = None
    number_shares: Optional[Decimal] = None
    dollar_shares: Optional[Decimal] = None
    proceeds: Optional[Decimal] = None
    shelf: Optional[bool] = None
    importance: Optional[int] = None
    notes: Optional[str] = None
    url: Optional[str] = None
    updated: Optional[datetime] = None
    raw: Optional[Dict[str, Any]] = None


class BenzingaAdapter:
    """Benzinga REST adapter returning typed event data with lineage."""

    def __init__(
        self, config: BenzingaConfig, session: Optional[requests.Session] = None
    ):
        self._config = config
        self._session = session or requests.Session()

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
                        message="Benzinga adapter asof timestamp must be timezone-aware datetime",
                        retryable=False,
                    ),
                )

        request_params = dict(params or {})
        request_params["token"] = self._config.api_key
        headers = {"Accept": "application/json"}

        try:
            resp = self._session.get(
                url, params=request_params, headers=headers, timeout=30
            )
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
                    message=f"Benzinga request failed: {exc.__class__.__name__}",
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
            source_authority="Benzinga",
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
                    message="Benzinga rate limit exceeded",
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
                    message=f"Benzinga auth error: {resp.status_code}",
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
                    message=f"Benzinga HTTP {resp.status_code}",
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

    def get_news(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
        *,
        symbols: Optional[Union[str, Sequence[str]]] = None,
        channels: Optional[Union[str, Sequence[str]]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        published_since: Optional[Union[str, int]] = None,
        updated_since: Optional[Union[str, int]] = None,
        page: Optional[int] = None,
        pagesize: Optional[int] = None,
        limit: Optional[int] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[BenzingaNewsArticle]]:
        """Fetch Benzinga news article rows as catalyst/context evidence."""

        params: Dict[str, Any] = {}
        ticker_values = tickers if tickers else symbols
        if ticker_values:
            params["tickers"] = _csv_param(ticker_values)
        if channels:
            params["channels"] = _csv_param(channels)
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to
        if published_since is not None:
            params["publishedSince"] = published_since
        if updated_since is not None:
            params["updatedSince"] = updated_since
        if page is not None:
            params["page"] = page
        page_size = pagesize if pagesize is not None else limit
        if page_size is not None:
            params["pageSize"] = page_size

        resp = self._request(NEWS_ENDPOINT, params=params or None, asof=asof)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        rows = _news_rows_from_payload(resp.data)
        articles = [
            _parse_news_article_row(row)
            for row in rows
            if isinstance(row, dict)
        ]
        return AdapterResponse(data=articles, lineage=resp.lineage)

    def get_wiims(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
        *,
        symbols: Optional[Union[str, Sequence[str]]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        published_since: Optional[Union[str, int]] = None,
        updated_since: Optional[Union[str, int]] = None,
        page: Optional[int] = None,
        pagesize: Optional[int] = None,
        limit: Optional[int] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[BenzingaNewsArticle]]:
        """Fetch Benzinga WIIM-channel news rows."""

        return self.get_news(
            tickers=tickers,
            symbols=symbols,
            channels=WIIM_CHANNEL,
            date_from=date_from,
            date_to=date_to,
            published_since=published_since,
            updated_since=updated_since,
            page=page,
            pagesize=pagesize,
            limit=limit,
            asof=asof,
        )

    def get_earnings(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
        *,
        symbols: Optional[Union[str, Sequence[str]]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: Optional[int] = None,
        pagesize: int = 100,
        updated: Optional[Union[int, str]] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[BenzingaEarnings]]:
        """Fetch Benzinga earnings calendar rows as catalyst/context evidence."""

        try:
            params = _calendar_params(
                tickers=tickers,
                symbols=symbols,
                date_from=date_from,
                date_to=date_to,
                page=page,
                pagesize=pagesize,
                updated=updated,
            )
        except ValueError as exc:
            return _validation_error_response(EARNINGS_ENDPOINT, str(exc), asof=asof)
        resp = self._request(EARNINGS_ENDPOINT, params=params, asof=asof)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        events = [
            _parse_earnings_row(row)
            for row in _calendar_rows_from_payload(resp.data, "earnings")
            if _calendar_row_has_usable_ticker(row)
        ]
        return AdapterResponse(data=events, lineage=resp.lineage)

    def get_guidance(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
        *,
        symbols: Optional[Union[str, Sequence[str]]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: Optional[int] = None,
        pagesize: int = 100,
        updated: Optional[Union[int, str]] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[BenzingaGuidance]]:
        """Fetch Benzinga guidance calendar rows as catalyst/context evidence."""

        try:
            params = _calendar_params(
                tickers=tickers,
                symbols=symbols,
                date_from=date_from,
                date_to=date_to,
                page=page,
                pagesize=pagesize,
                updated=updated,
            )
        except ValueError as exc:
            return _validation_error_response(GUIDANCE_ENDPOINT, str(exc), asof=asof)
        resp = self._request(GUIDANCE_ENDPOINT, params=params, asof=asof)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        events = [
            _parse_guidance_row(row)
            for row in _calendar_rows_from_payload(resp.data, "guidance")
            if _calendar_row_has_usable_ticker(row)
        ]
        return AdapterResponse(data=events, lineage=resp.lineage)

    def get_ratings(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
        *,
        symbols: Optional[Union[str, Sequence[str]]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: Optional[int] = None,
        pagesize: int = 100,
        updated: Optional[Union[int, str]] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[BenzingaRating]]:
        """Fetch Benzinga analyst rating calendar rows as catalyst context."""

        try:
            params = _calendar_params(
                tickers=tickers,
                symbols=symbols,
                date_from=date_from,
                date_to=date_to,
                page=page,
                pagesize=pagesize,
                updated=updated,
            )
        except ValueError as exc:
            return _validation_error_response(RATINGS_ENDPOINT, str(exc), asof=asof)
        resp = self._request(RATINGS_ENDPOINT, params=params, asof=asof)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        events = [
            _parse_rating_row(row)
            for row in _calendar_rows_from_payload(resp.data, "ratings")
            if _calendar_row_has_usable_ticker(row)
        ]
        return AdapterResponse(data=events, lineage=resp.lineage)

    def get_offerings(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
        *,
        symbols: Optional[Union[str, Sequence[str]]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: Optional[int] = None,
        pagesize: int = 100,
        updated: Optional[Union[int, str]] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[BenzingaOffering]]:
        """Fetch Benzinga secondary offering rows as catalyst/review evidence."""

        try:
            params = _calendar_params(
                tickers=tickers,
                symbols=symbols,
                date_from=date_from,
                date_to=date_to,
                page=page,
                pagesize=pagesize,
                updated=updated,
            )
        except ValueError as exc:
            return _validation_error_response(OFFERINGS_ENDPOINT, str(exc), asof=asof)
        resp = self._request(OFFERINGS_ENDPOINT, params=params, asof=asof)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        events = [
            _parse_offering_row(row)
            for row in _calendar_rows_from_payload(resp.data, "offerings")
            if _calendar_row_has_usable_ticker(row)
        ]
        return AdapterResponse(data=events, lineage=resp.lineage)

    def get_mergers_acquisitions(
        self,
        tickers: Optional[Union[str, Sequence[str]]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        pagesize: int = 100,
        page: Optional[int] = None,
        importance: Optional[Union[int, str]] = None,
        updated: Optional[Union[int, str]] = None,
        date_sort: Optional[str] = None,
        cusip: Optional[Union[str, Sequence[str]]] = None,
        isin: Optional[Union[str, Sequence[str]]] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[BenzingaMergerAcquisition]]:
        """Fetch Benzinga calendar M&A rows."""

        params: Dict[str, Any] = {"pagesize": pagesize}
        if page is not None:
            params["page"] = page
        if tickers:
            params["parameters[tickers]"] = _csv_param(tickers)
        if date_from:
            params["parameters[date_from]"] = date_from
        if date_to:
            params["parameters[date_to]"] = date_to
        if importance is not None:
            params["parameters[importance]"] = importance
        if updated is not None:
            params["parameters[updated]"] = updated
        if date_sort is not None:
            params["parameters[date_sort]"] = date_sort
        if cusip:
            normalized_cusip = _identifier_csv_param(cusip, kind="cusip")
            if normalized_cusip is None:
                return _validation_error_response(
                    M_AND_A_ENDPOINT,
                    "Benzinga M&A CUSIP parameters must be 9-character alphanumeric identifiers",
                    asof=asof,
                )
            params["parameters[cusip]"] = normalized_cusip
        if isin:
            normalized_isin = _identifier_csv_param(isin, kind="isin")
            if normalized_isin is None:
                return _validation_error_response(
                    M_AND_A_ENDPOINT,
                    "Benzinga M&A ISIN parameters must be 12-character alphanumeric identifiers",
                    asof=asof,
                )
            params["parameters[isin]"] = normalized_isin

        resp = self._request(M_AND_A_ENDPOINT, params=params, asof=asof)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        if isinstance(resp.data, dict):
            rows = resp.data.get("ma") or []
        elif isinstance(resp.data, list):
            rows = resp.data
        else:
            rows = []

        events = [
            _parse_merger_acquisition_row(row)
            for row in rows
            if isinstance(row, dict)
        ]
        return AdapterResponse(data=events, lineage=resp.lineage)


def _calendar_params(
    *,
    tickers: Optional[Union[str, Sequence[str]]],
    symbols: Optional[Union[str, Sequence[str]]],
    date_from: Optional[str],
    date_to: Optional[str],
    page: Optional[int],
    pagesize: int,
    updated: Optional[Union[int, str]],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"pagesize": pagesize}
    if page is not None:
        params["page"] = page

    ticker_values = tickers if tickers else symbols
    if ticker_values:
        normalized_tickers = _calendar_ticker_csv_param(ticker_values)
        if normalized_tickers:
            params["parameters[tickers]"] = normalized_tickers
    if date_from:
        params["parameters[date_from]"] = date_from
    if date_to:
        params["parameters[date_to]"] = date_to
    if updated is not None:
        params["parameters[updated]"] = updated
    return params


def _calendar_rows_from_payload(payload: Any, key: str) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    rows = payload.get(key) or []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _calendar_row_has_usable_ticker(row: Dict[str, Any]) -> bool:
    ticker = _string_or_none(row.get("ticker"))
    return ticker is not None and ticker.strip() != ""


def _calendar_ticker_csv_param(value: Union[str, Sequence[str]]) -> Optional[str]:
    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError as exc:
            raise ValueError("Benzinga calendar ticker parameters must be strings") from exc
    normalized = []
    for item in items:
        if not isinstance(item, str):
            raise ValueError("Benzinga calendar ticker parameters must be strings")
        ticker = item.strip().upper()
        if ticker:
            normalized.append(ticker)
    if not normalized:
        return None
    return ",".join(normalized)


def _news_rows_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("news", "articles", "data", "results"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _parse_news_article_row(row: Dict[str, Any]) -> BenzingaNewsArticle:
    stock_rows = _stock_rows(row.get("stocks"))
    return BenzingaNewsArticle(
        id=_string_or_none(row.get("id")),
        created=_timestamp_or_none(
            row.get("created")
            or row.get("created_at")
            or row.get("createdAt")
        ),
        updated=_timestamp_or_none(
            row.get("updated")
            or row.get("updated_at")
            or row.get("updatedAt")
        ),
        published=_timestamp_or_none(
            row.get("published")
            or row.get("published_at")
            or row.get("publishedAt")
            or row.get("published_date")
            or row.get("date")
        ),
        title=_string_or_none(row.get("title")),
        body=_first_string(row, "body", "content"),
        teaser=_first_string(row, "teaser", "summary", "description"),
        url=_first_string(row, "url", "link"),
        author=_string_or_none(row.get("author")),
        source=_first_string(row, "source", "source_name", "provider"),
        stocks=stock_rows,
        tickers=_dedupe_strings(
            _ticker_strings(row.get("tickers"))
            + _ticker_strings(row.get("symbols"))
            + _ticker_strings(stock_rows)
        ),
        channels=_named_strings(row.get("channels")),
        tags=_named_strings(row.get("tags")),
        categories=_dedupe_strings(
            _named_strings(row.get("categories"))
            + _named_strings(row.get("category"))
        ),
        raw=dict(row),
    )


def _parse_earnings_row(row: Dict[str, Any]) -> BenzingaEarnings:
    return BenzingaEarnings(
        id=_string_or_none(row.get("id")),
        ticker=_string_or_none(row.get("ticker")),
        name=_string_or_none(row.get("name")),
        exchange=_string_or_none(row.get("exchange")),
        currency=_string_or_none(row.get("currency")),
        cusip=_first_identifier(row, "cusip", kind="cusip"),
        isin=_first_identifier(row, "isin", kind="isin"),
        period=_string_or_none(row.get("period")),
        period_year=_int_or_none(row.get("period_year")),
        date=_string_or_none(row.get("date")),
        time=_string_or_none(row.get("time")),
        eps=_decimal_or_none(row.get("eps")),
        eps_est=_decimal_or_none(row.get("eps_est")),
        eps_prior=_decimal_or_none(row.get("eps_prior")),
        eps_surprise=_decimal_or_none(row.get("eps_surprise")),
        eps_surprise_percent=_decimal_or_none(row.get("eps_surprise_percent")),
        eps_type=_string_or_none(row.get("eps_type")),
        revenue=_decimal_or_none(row.get("revenue")),
        revenue_est=_decimal_or_none(row.get("revenue_est")),
        revenue_prior=_decimal_or_none(row.get("revenue_prior")),
        revenue_surprise=_decimal_or_none(row.get("revenue_surprise")),
        revenue_surprise_percent=_decimal_or_none(
            row.get("revenue_surprise_percent")
        ),
        revenue_type=_string_or_none(row.get("revenue_type")),
        date_confirmed=_bool_or_none(row.get("date_confirmed")),
        importance=_int_or_none(row.get("importance")),
        notes=_string_or_none(row.get("notes")),
        updated=_timestamp_or_none(row.get("updated")),
        raw=dict(row),
    )


def _parse_guidance_row(row: Dict[str, Any]) -> BenzingaGuidance:
    return BenzingaGuidance(
        id=_string_or_none(row.get("id")),
        ticker=_string_or_none(row.get("ticker")),
        name=_string_or_none(row.get("name")),
        exchange=_string_or_none(row.get("exchange")),
        currency=_string_or_none(row.get("currency")),
        cusip=_first_identifier(row, "cusip", kind="cusip"),
        period=_string_or_none(row.get("period")),
        period_year=_int_or_none(row.get("period_year")),
        date=_string_or_none(row.get("date")),
        time=_string_or_none(row.get("time")),
        eps_guidance_est=_decimal_or_none(row.get("eps_guidance_est")),
        eps_guidance_min=_decimal_or_none(row.get("eps_guidance_min")),
        eps_guidance_max=_decimal_or_none(row.get("eps_guidance_max")),
        eps_guidance_prior_min=_decimal_or_none(
            row.get("eps_guidance_prior_min")
        ),
        eps_guidance_prior_max=_decimal_or_none(
            row.get("eps_guidance_prior_max")
        ),
        eps_type=_string_or_none(row.get("eps_type")),
        revenue_guidance_est=_decimal_or_none(row.get("revenue_guidance_est")),
        revenue_guidance_min=_decimal_or_none(row.get("revenue_guidance_min")),
        revenue_guidance_max=_decimal_or_none(row.get("revenue_guidance_max")),
        revenue_guidance_prior_min=_decimal_or_none(
            row.get("revenue_guidance_prior_min")
        ),
        revenue_guidance_prior_max=_decimal_or_none(
            row.get("revenue_guidance_prior_max")
        ),
        revenue_type=_string_or_none(row.get("revenue_type")),
        is_primary=_bool_or_none(row.get("is_primary")),
        prelim=_bool_or_none(row.get("prelim")),
        importance=_int_or_none(row.get("importance")),
        notes=_string_or_none(row.get("notes")),
        updated=_timestamp_or_none(row.get("updated")),
        raw=dict(row),
    )


def _parse_rating_row(row: Dict[str, Any]) -> BenzingaRating:
    return BenzingaRating(
        id=_string_or_none(row.get("id")),
        ticker=_string_or_none(row.get("ticker")),
        name=_string_or_none(row.get("name")),
        exchange=_string_or_none(row.get("exchange")),
        currency=_string_or_none(row.get("currency")),
        cusip=_first_identifier(row, "cusip", kind="cusip"),
        isin=_first_identifier(row, "isin", kind="isin"),
        date=_string_or_none(row.get("date")),
        time=_string_or_none(row.get("time")),
        analyst=_string_or_none(row.get("analyst")),
        analyst_id=_string_or_none(row.get("analyst_id")),
        analyst_name=_string_or_none(row.get("analyst_name")),
        firm=_first_string(row, "firm", "firm_name", "brokerage"),
        firm_id=_string_or_none(row.get("firm_id")),
        action_company=_string_or_none(row.get("action_company")),
        action_pt=_string_or_none(row.get("action_pt")),
        rating_current=_string_or_none(row.get("rating_current")),
        rating_prior=_string_or_none(row.get("rating_prior")),
        pt_current=_decimal_or_none(row.get("pt_current")),
        pt_prior=_decimal_or_none(row.get("pt_prior")),
        adjusted_pt_current=_decimal_or_none(row.get("adjusted_pt_current")),
        adjusted_pt_prior=_decimal_or_none(row.get("adjusted_pt_prior")),
        pt_pct_change=_decimal_or_none(row.get("pt_pct_change")),
        ratings_accuracy=_decimal_or_none(row.get("ratings_accuracy")),
        importance=_int_or_none(row.get("importance")),
        notes=_string_or_none(row.get("notes")),
        url=_string_or_none(row.get("url")),
        url_calendar=_string_or_none(row.get("url_calendar")),
        url_news=_string_or_none(row.get("url_news")),
        updated=_timestamp_or_none(row.get("updated")),
        raw=dict(row),
    )


def _parse_offering_row(row: Dict[str, Any]) -> BenzingaOffering:
    return BenzingaOffering(
        id=_string_or_none(row.get("id")),
        ticker=_string_or_none(row.get("ticker")),
        name=_string_or_none(row.get("name")),
        exchange=_string_or_none(row.get("exchange")),
        currency=_string_or_none(row.get("currency")),
        cusip=_first_identifier(row, "cusip", kind="cusip"),
        date=_string_or_none(row.get("date")),
        time=_string_or_none(row.get("time")),
        offering_type=_string_or_none(row.get("offering_type")),
        price=_decimal_or_none(row.get("price")),
        number_shares=_decimal_or_none(row.get("number_shares")),
        dollar_shares=_decimal_or_none(row.get("dollar_shares")),
        proceeds=_decimal_or_none(row.get("proceeds")),
        shelf=_bool_or_none(row.get("shelf")),
        importance=_int_or_none(row.get("importance")),
        notes=_string_or_none(row.get("notes")),
        url=_string_or_none(row.get("url")),
        updated=_timestamp_or_none(row.get("updated")),
        raw=dict(row),
    )


def _parse_merger_acquisition_row(row: Dict[str, Any]) -> BenzingaMergerAcquisition:
    return BenzingaMergerAcquisition(
        id=_string_or_none(row.get("id")),
        target_ticker=_string_or_none(row.get("target_ticker")),
        target_name=_string_or_none(row.get("target_name")),
        target_exchange=_string_or_none(row.get("target_exchange")),
        target_cusip=_first_identifier(
            row,
            "target_cusip",
            "target_cusip_number",
            "target_security_cusip",
            "target_cusips",
            kind="cusip",
        ),
        target_isin=_first_identifier(
            row,
            "target_isin",
            "target_isin_number",
            "target_security_isin",
            "target_isins",
            kind="isin",
        ),
        acquirer_ticker=_string_or_none(row.get("acquirer_ticker")),
        acquirer_name=_string_or_none(row.get("acquirer_name")),
        acquirer_exchange=_string_or_none(row.get("acquirer_exchange")),
        acquirer_cusip=_first_identifier(
            row,
            "acquirer_cusip",
            "acquirer_cusip_number",
            "acquirer_security_cusip",
            "acquirer_cusips",
            kind="cusip",
        ),
        acquirer_isin=_first_identifier(
            row,
            "acquirer_isin",
            "acquirer_isin_number",
            "acquirer_security_isin",
            "acquirer_isins",
            kind="isin",
        ),
        deal_type=_string_or_none(row.get("deal_type")),
        deal_status=_string_or_none(row.get("deal_status")),
        deal_payment_type=_string_or_none(row.get("deal_payment_type")),
        deal_size=_string_or_none(row.get("deal_size")),
        currency=_string_or_none(row.get("currency")),
        date=_string_or_none(row.get("date")),
        date_completed=_string_or_none(row.get("date_completed")),
        date_expected=_string_or_none(row.get("date_expected")),
        deal_terms_extra=_string_or_none(row.get("deal_terms_extra")),
        notes=_string_or_none(row.get("notes")),
        importance=_int_or_none(row.get("importance")),
        updated=_int_or_none(row.get("updated")),
        raw=dict(row),
    )


def _csv_param(value: Union[str, Sequence[str]]) -> str:
    return value if isinstance(value, str) else ",".join(value)


def _stock_rows(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        rows: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                rows.append(dict(item))
            else:
                text = _string_or_none(item)
                if text is not None:
                    rows.append({"ticker": text})
        return rows
    if isinstance(value, dict):
        return [dict(value)]
    text = _string_or_none(value)
    if text is None:
        return []
    return [{"ticker": part.strip()} for part in text.split(",") if part.strip()]


def _ticker_strings(value: Any) -> List[str]:
    values: List[str] = []
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    for item in items:
        if isinstance(item, dict):
            text = _first_string(item, "ticker", "symbol", "name")
        else:
            text = _string_or_none(item)
        if text is None:
            continue
        values.extend(part.strip().upper() for part in text.split(",") if part.strip())
    return values


def _named_strings(value: Any) -> List[str]:
    values: List[str] = []
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    for item in items:
        if isinstance(item, dict):
            text = _first_string(item, "name", "channel", "category", "tag", "slug", "id")
        else:
            text = _string_or_none(item)
        if text is None:
            continue
        values.extend(part.strip() for part in text.split(",") if part.strip())
    return _dedupe_strings(values)


def _dedupe_strings(values: Sequence[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _identifier_csv_param(
    value: Union[str, Sequence[str]],
    *,
    kind: str,
) -> Optional[str]:
    values = [value] if isinstance(value, str) else list(value)
    normalized = [_normalize_identifier(item, kind=kind) for item in values]
    if any(item is None for item in normalized):
        return None
    return ",".join(item for item in normalized if item)


def _first_identifier(row: Dict[str, Any], *keys: str, kind: str) -> Optional[str]:
    for key in keys:
        value = _normalize_identifier(row.get(key), kind=kind)
        if value is not None:
            return value
    return None


def _first_string(row: Dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = _string_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _normalize_identifier(value: Any, *, kind: str) -> Optional[str]:
    text = _string_or_none(value)
    if text is None:
        return None
    normalized = text.strip().upper().replace("-", "").replace(" ", "")
    if kind == "cusip":
        return normalized if len(normalized) == 9 and normalized.isalnum() else None
    if kind == "isin":
        return normalized if len(normalized) == 12 and normalized.isalnum() else None
    return None


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text != "" else None


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _decimal_or_none(value: Any) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _bool_or_none(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if text in ("y", "yes", "true", "1"):
        return True
    if text in ("n", "no", "false", "0"):
        return False
    return None


def _timestamp_or_none(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 10_000_000_000:
            numeric = numeric / 1000
        return datetime.fromtimestamp(numeric, tz=timezone.utc)

    text = str(value).strip()
    if not text:
        return None
    try:
        return _timestamp_or_none(float(text))
    except ValueError:
        pass

    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError:
        parsed = None
    if parsed is not None:
        return _timestamp_or_none(parsed)

    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    return _timestamp_or_none(parsed)


def _validation_error_response(
    endpoint: str,
    message: str,
    *,
    asof: Optional[datetime],
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
            source_authority="Benzinga",
        ),
        error=ProviderError(
            provider=PROVIDER,
            endpoint=endpoint,
            status_code=None,
            error_type="validation",
            message=message,
            retryable=False,
        ),
    )
