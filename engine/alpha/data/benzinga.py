"""
Benzinga adapter.

Supplemental event source for:
  - M&A and acquisition evidence for survivorship/corporate-action review
  - News/WIIMs catalyst context for event diagnostics

Does not write to DB. Returns AdapterResponse with LineageMeta.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import urlparse

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
BENZINGA_REQUEST_TIMEOUT = (10, 30)
M_AND_A_ENDPOINT = "/api/v2.1/calendar/ma"
EARNINGS_ENDPOINT = "/api/v2.1/calendar/earnings"
GUIDANCE_ENDPOINT = "/api/v2.1/calendar/guidance"
RATINGS_ENDPOINT = "/api/v2.1/calendar/ratings"
OFFERINGS_ENDPOINT = "/api/v2.1/calendar/offerings"
DIVIDENDS_ENDPOINT = "/api/v2.1/calendar/dividends"
INSIDER_FILINGS_ENDPOINT = "/api/v1/sec/insider_transactions/filings"
INSIDER_TRANSACTIONS_ENDPOINT = "/api/v1/sec/insider_transactions/transactions"
NEWS_ENDPOINT = "/api/v2/news"
WIIM_CHANNEL = "wiim"
MAX_PAGESIZE = 1000
KNOWLEDGE_TIMESTAMP_FUTURE_TOLERANCE = timedelta(minutes=5)


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
    event_date: Optional[str] = None
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


@dataclass
class BenzingaDividend:
    """Normalized Benzinga dividend calendar row with raw payload preserved."""

    id: Optional[str]
    ticker: Optional[str] = None
    name: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    cusip: Optional[str] = None
    isin: Optional[str] = None
    date: Optional[str] = None
    ex_dividend_date: Optional[str] = None
    payable_date: Optional[str] = None
    record_date: Optional[str] = None
    dividend: Optional[Decimal] = None
    dividend_prior: Optional[Decimal] = None
    dividend_type: Optional[str] = None
    dividend_yield: Optional[Decimal] = None
    frequency: Optional[int] = None
    confirmed: Optional[bool] = None
    end_regular_dividend: Optional[bool] = None
    period: Optional[str] = None
    year: Optional[int] = None
    importance: Optional[int] = None
    notes: Optional[str] = None
    updated: Optional[datetime] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class BenzingaInsiderFiling:
    """Normalized Benzinga insider filing row with raw payload preserved."""

    id: Optional[str]
    accession_number: Optional[str] = None
    company_cik: Optional[str] = None
    company_name: Optional[str] = None
    company_symbol: Optional[str] = None
    filing_date: Optional[datetime] = None
    form_type: Optional[str] = None
    html_url: Optional[str] = None
    is_10b5: Optional[bool] = None
    insider_cik: Optional[str] = None
    insider_name: Optional[str] = None
    insider_title: Optional[str] = None
    is_director: Optional[bool] = None
    is_officer: Optional[bool] = None
    is_ten_percent_owner: Optional[bool] = None
    raw_signature: Optional[str] = None
    remaining_shares: Optional[Decimal] = None
    traded_percentage: Optional[str] = None
    footnotes: List[Dict[str, Any]] = field(default_factory=list)
    transactions: List[Dict[str, Any]] = field(default_factory=list)
    owner: Dict[str, Any] = field(default_factory=dict)
    updated: Optional[datetime] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class BenzingaInsiderTransaction:
    """Normalized Benzinga insider transaction row with raw payload preserved."""

    transaction_id: Optional[str]
    accession_number: Optional[str] = None
    company_cik: Optional[str] = None
    company_name: Optional[str] = None
    company_symbol: Optional[str] = None
    filing_date: Optional[datetime] = None
    form_type: Optional[str] = None
    filing_id: Optional[str] = None
    html_url: Optional[str] = None
    insider_cik: Optional[str] = None
    insider_name: Optional[str] = None
    insider_title: Optional[str] = None
    is_director: Optional[bool] = None
    is_officer: Optional[bool] = None
    is_ten_percent_owner: Optional[bool] = None
    raw_signature: Optional[str] = None
    acquired_or_disposed: Optional[str] = None
    conversion_exercise_price_derivative: Optional[Decimal] = None
    date_deemed_execution: Optional[datetime] = None
    date_exercisable: Optional[datetime] = None
    date_expiration: Optional[datetime] = None
    date_transaction: Optional[datetime] = None
    is_derivative: Optional[bool] = None
    ownership: Optional[str] = None
    post_transaction_quantity: Optional[Decimal] = None
    price_per_share: Optional[Decimal] = None
    remaining_underlying_shares: Optional[Decimal] = None
    security_title: Optional[str] = None
    shares: Optional[Decimal] = None
    transaction_code: Optional[str] = None
    underlying_security_title: Optional[str] = None
    underlying_shares: Optional[Decimal] = None
    voluntarily_reported: Optional[bool] = None
    owner: Dict[str, Any] = field(default_factory=dict)
    filing: Dict[str, Any] = field(default_factory=dict)
    updated: Optional[datetime] = None
    raw: Optional[Dict[str, Any]] = None


class BenzingaAdapter:
    """Benzinga REST adapter returning typed event data with lineage."""

    def __init__(
        self, config: BenzingaConfig, session: Optional[requests.Session] = None
    ):
        self._config = config
        self._session = session or requests.Session()

    def reset_session(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()
        self._session = requests.Session()

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
                url,
                params=request_params,
                headers=headers,
                timeout=BENZINGA_REQUEST_TIMEOUT,
            )
        except requests.exceptions.Timeout:
            self.reset_session()
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
            self.reset_session()
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
        except ValueError:
            return AdapterResponse(
                data=None,
                lineage=lineage,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=200,
                    error_type="parse",
                    message="Benzinga JSON parse error",
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

        try:
            page_value = _positive_int_param(page, "page", required=False)
            page_size = pagesize if pagesize is not None else limit
            pagesize_value = _positive_int_param(
                page_size, "pagesize", required=False
            )
            date_from_value, date_to_value = _validate_date_range(
                date_from,
                date_to,
                from_name="date_from",
                to_name="date_to",
            )
            ticker_param = _validated_ticker_filter(tickers, symbols)
            _require_ticker_or_bounded_dates(
                ticker_param,
                date_from_value,
                date_to_value,
            )
            published_since_value = _updated_param(
                published_since,
                "published_since",
            )
            updated_since_value = _updated_param(
                updated_since,
                "updated_since",
            )
        except ValueError as exc:
            return _validation_error_response(NEWS_ENDPOINT, str(exc), asof=asof)

        params: Dict[str, Any] = {}
        if ticker_param:
            params["tickers"] = ticker_param
        if channels:
            params["channels"] = _csv_param(channels)
        if date_from_value:
            params["dateFrom"] = date_from_value
        if date_to_value:
            params["dateTo"] = date_to_value
        if published_since_value is not None:
            params["publishedSince"] = published_since_value
        if updated_since_value is not None:
            params["updatedSince"] = updated_since_value
        if page_value is not None:
            params["page"] = page_value
        if pagesize_value is not None:
            params["pageSize"] = pagesize_value

        resp = self._request(NEWS_ENDPOINT, params=params or None, asof=asof)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        rows, error = _rows_from_payload(
            resp,
            endpoint=NEWS_ENDPOINT,
            keys=("news", "articles", "data", "results"),
            page=page_value,
            pagesize=pagesize_value,
        )
        if error is not None:
            return error  # type: ignore[return-value]
        articles, extra_flags = _parse_valid_rows(
            rows,
            is_valid=_news_row_has_identity,
            parser=_parse_news_article_row,
            cutoff=resp.lineage.asof_timestamp,
        )
        return _response_with_data(
            resp,
            articles,
            raw_rows=len(rows or []),
            page=page_value,
            pagesize=pagesize_value,
            extra_flags=extra_flags,
        )

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

        rows, error = _rows_from_payload(
            resp,
            endpoint=EARNINGS_ENDPOINT,
            keys=("earnings",),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
        )
        if error is not None:
            return error  # type: ignore[return-value]
        events, extra_flags = _parse_valid_rows(
            rows,
            is_valid=_earnings_row_is_valid,
            parser=_parse_earnings_row,
            cutoff=resp.lineage.asof_timestamp,
        )
        return _response_with_data(
            resp,
            events,
            raw_rows=len(rows or []),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
            extra_flags=extra_flags,
        )

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

        rows, error = _rows_from_payload(
            resp,
            endpoint=GUIDANCE_ENDPOINT,
            keys=("guidance",),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
        )
        if error is not None:
            return error  # type: ignore[return-value]
        events, extra_flags = _parse_valid_rows(
            rows,
            is_valid=_guidance_row_is_valid,
            parser=_parse_guidance_row,
            cutoff=resp.lineage.asof_timestamp,
        )
        return _response_with_data(
            resp,
            events,
            raw_rows=len(rows or []),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
            extra_flags=extra_flags,
        )

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

        rows, error = _rows_from_payload(
            resp,
            endpoint=RATINGS_ENDPOINT,
            keys=("ratings",),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
        )
        if error is not None:
            return error  # type: ignore[return-value]
        events, extra_flags = _parse_valid_rows(
            rows,
            is_valid=_rating_row_is_valid,
            parser=_parse_rating_row,
            cutoff=resp.lineage.asof_timestamp,
        )
        return _response_with_data(
            resp,
            events,
            raw_rows=len(rows or []),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
            extra_flags=extra_flags,
        )

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

        rows, error = _rows_from_payload(
            resp,
            endpoint=OFFERINGS_ENDPOINT,
            keys=("offerings",),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
        )
        if error is not None:
            return error  # type: ignore[return-value]
        events, extra_flags = _parse_valid_rows(
            rows,
            is_valid=_offering_row_is_valid,
            parser=_parse_offering_row,
            cutoff=resp.lineage.asof_timestamp,
        )
        return _response_with_data(
            resp,
            events,
            raw_rows=len(rows or []),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
            extra_flags=extra_flags,
        )

    def get_dividends(
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
    ) -> AdapterResponse[List[BenzingaDividend]]:
        """Fetch Benzinga dividend calendar rows as corporate-action evidence."""

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
            return _validation_error_response(DIVIDENDS_ENDPOINT, str(exc), asof=asof)
        resp = self._request(DIVIDENDS_ENDPOINT, params=params, asof=asof)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        rows, error = _rows_from_payload(
            resp,
            endpoint=DIVIDENDS_ENDPOINT,
            keys=("dividends",),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
        )
        if error is not None:
            return error  # type: ignore[return-value]
        events, extra_flags = _parse_valid_rows(
            rows,
            is_valid=_dividend_row_is_valid,
            parser=_parse_dividend_row,
            cutoff=resp.lineage.asof_timestamp,
        )
        return _response_with_data(
            resp,
            events,
            raw_rows=len(rows or []),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
            extra_flags=extra_flags,
        )

    def get_insider_filings(
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
    ) -> AdapterResponse[List[BenzingaInsiderFiling]]:
        """Fetch Benzinga insider Forms 3/4/5 filing rows as review evidence."""

        try:
            params = _insider_params(
                tickers=tickers,
                symbols=symbols,
                date_from=date_from,
                date_to=date_to,
                page=page,
                pagesize=pagesize,
                updated=updated,
            )
        except ValueError as exc:
            return _validation_error_response(INSIDER_FILINGS_ENDPOINT, str(exc), asof=asof)
        resp = self._request(INSIDER_FILINGS_ENDPOINT, params=params, asof=asof)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        rows, error = _rows_from_payload(
            resp,
            endpoint=INSIDER_FILINGS_ENDPOINT,
            keys=("data",),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
        )
        if error is not None:
            return error  # type: ignore[return-value]
        filings, extra_flags = _parse_valid_rows(
            rows,
            is_valid=_insider_filing_row_has_identity,
            parser=_parse_insider_filing_row,
            cutoff=resp.lineage.asof_timestamp,
        )
        return _response_with_data(
            resp,
            filings,
            raw_rows=len(rows or []),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
            extra_flags=extra_flags,
        )

    def get_insider_transactions(
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
    ) -> AdapterResponse[List[BenzingaInsiderTransaction]]:
        """Fetch Benzinga flattened insider transaction rows as review evidence."""

        try:
            params = _insider_params(
                tickers=tickers,
                symbols=symbols,
                date_from=date_from,
                date_to=date_to,
                page=page,
                pagesize=pagesize,
                updated=updated,
            )
        except ValueError as exc:
            return _validation_error_response(
                INSIDER_TRANSACTIONS_ENDPOINT,
                str(exc),
                asof=asof,
            )
        resp = self._request(INSIDER_TRANSACTIONS_ENDPOINT, params=params, asof=asof)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        rows, error = _rows_from_payload(
            resp,
            endpoint=INSIDER_TRANSACTIONS_ENDPOINT,
            keys=("data",),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
        )
        if error is not None:
            return error  # type: ignore[return-value]
        transactions, extra_flags = _parse_valid_rows(
            rows,
            is_valid=_insider_transaction_row_has_identity,
            parser=_parse_insider_transaction_row,
            cutoff=resp.lineage.asof_timestamp,
        )
        return _response_with_data(
            resp,
            transactions,
            raw_rows=len(rows or []),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
            extra_flags=extra_flags,
        )

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

        try:
            page_value = _positive_int_param(page, "page", required=False)
            pagesize_value = _positive_int_param(pagesize, "pagesize", required=True)
            date_from_value, date_to_value = _validate_date_range(date_from, date_to)
            ticker_param = _validated_ticker_filter(tickers, None)
            _require_ticker_or_bounded_dates(
                ticker_param,
                date_from_value,
                date_to_value,
            )
            updated_value = _updated_param(updated)
        except ValueError as exc:
            return _validation_error_response(M_AND_A_ENDPOINT, str(exc), asof=asof)

        params: Dict[str, Any] = {"pagesize": pagesize_value}
        if page_value is not None:
            params["page"] = page_value
        if ticker_param:
            params["parameters[tickers]"] = ticker_param
        if date_from_value:
            params["parameters[date_from]"] = date_from_value
        if date_to_value:
            params["parameters[date_to]"] = date_to_value
        if importance is not None:
            params["parameters[importance]"] = importance
        if updated_value is not None:
            params["parameters[updated]"] = updated_value
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

        rows, error = _rows_from_payload(
            resp,
            endpoint=M_AND_A_ENDPOINT,
            keys=("ma",),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
        )
        if error is not None:
            return error  # type: ignore[return-value]

        events, extra_flags = _parse_valid_rows(
            rows,
            is_valid=_merger_acquisition_row_is_valid,
            parser=_parse_merger_acquisition_row,
            cutoff=resp.lineage.asof_timestamp,
        )
        return _response_with_data(
            resp,
            events,
            raw_rows=len(rows or []),
            page=params.get("page"),
            pagesize=params.get("pagesize"),
            extra_flags=extra_flags,
        )


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
    page_value = _positive_int_param(page, "page", required=False)
    pagesize_value = _positive_int_param(pagesize, "pagesize", required=True)
    date_from_value, date_to_value = _validate_date_range(date_from, date_to)
    ticker_param = _validated_ticker_filter(tickers, symbols)
    _require_ticker_or_bounded_dates(ticker_param, date_from_value, date_to_value)

    params: Dict[str, Any] = {"pagesize": pagesize_value}
    if page_value is not None:
        params["page"] = page_value

    if ticker_param:
        params["parameters[tickers]"] = ticker_param
    if date_from_value:
        params["parameters[date_from]"] = date_from_value
    if date_to_value:
        params["parameters[date_to]"] = date_to_value
    if updated is not None:
        params["parameters[updated]"] = _updated_param(updated)
    return params


def _validate_ticker_values(value: Union[str, Sequence[str]]) -> Optional[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (set, frozenset)):
        raise ValueError("Benzinga ticker parameters must be ordered string sequences")
    else:
        try:
            items = list(value)
        except TypeError as exc:
            raise ValueError("Benzinga ticker parameters must be strings") from exc
    normalized: List[str] = []
    for item in items:
        if not isinstance(item, str):
            raise ValueError("Benzinga ticker parameters must be strings")
        if any(ord(ch) < 32 for ch in item):
            raise ValueError("Benzinga ticker parameters must not contain control characters")
        ticker = item.strip().upper()
        if not ticker:
            continue
        if "," in ticker:
            raise ValueError("Benzinga ticker parameters must not contain commas")
        normalized.append(ticker)
    if not normalized:
        return None
    return ",".join(dict.fromkeys(normalized))


def _validate_iso_date(value: Optional[str], name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Benzinga {name} must be YYYY-MM-DD")
    text = value.strip()
    if not text:
        raise ValueError(f"Benzinga {name} must be YYYY-MM-DD")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Benzinga {name} must be YYYY-MM-DD") from exc
    if parsed.strftime("%Y-%m-%d") != text:
        raise ValueError(f"Benzinga {name} must be YYYY-MM-DD")
    return text


def _validate_date_range(
    date_from: Optional[str],
    date_to: Optional[str],
    *,
    from_name: str = "date_from",
    to_name: str = "date_to",
) -> tuple[Optional[str], Optional[str]]:
    parsed_from = _validate_iso_date(date_from, from_name)
    parsed_to = _validate_iso_date(date_to, to_name)
    if parsed_from and parsed_to and parsed_from > parsed_to:
        raise ValueError(f"Benzinga {from_name} must be <= {to_name}")
    return parsed_from, parsed_to


def _positive_int_param(
    value: Any,
    name: str,
    *,
    required: bool,
    maximum: int = MAX_PAGESIZE,
) -> Optional[int]:
    if value is None:
        if required:
            raise ValueError(f"Benzinga {name} must be a positive integer")
        return None
    if isinstance(value, bool):
        raise ValueError(f"Benzinga {name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Benzinga {name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"Benzinga {name} must be a positive integer")
    if parsed > maximum:
        raise ValueError(f"Benzinga {name} must be <= {maximum}")
    return parsed


def _nonnegative_int_param(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Benzinga {name} must be a nonnegative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Benzinga {name} must be a nonnegative integer") from exc
    if parsed < 0:
        raise ValueError(f"Benzinga {name} must be a nonnegative integer")
    return parsed


def _updated_param(value: Any, name: str = "updated") -> Optional[Union[int, str]]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Benzinga {name} must be a nonnegative integer or timestamp")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        if text and _timestamp_or_none(text) is not None:
            return text
        raise ValueError(
            f"Benzinga {name} must be a nonnegative integer or timestamp"
        )
    if parsed < 0:
        raise ValueError(f"Benzinga {name} must be a nonnegative integer or timestamp")
    return parsed


def _validated_ticker_filter(
    tickers: Optional[Union[str, Sequence[str]]],
    symbols: Optional[Union[str, Sequence[str]]],
) -> Optional[str]:
    ticker_values = tickers if tickers is not None else symbols
    if ticker_values is None:
        return None
    return _validate_ticker_values(ticker_values)


def _require_ticker_or_bounded_dates(
    ticker_param: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
) -> None:
    if ticker_param:
        return
    if date_from and date_to:
        return
    raise ValueError(
        "Benzinga broad queries require a ticker filter or complete date_from/date_to bounds"
    )


def _page_flags(page: Optional[int], pagesize: Optional[int]) -> Dict[str, Any]:
    flags: Dict[str, Any] = {
        "page": page,
        "pagesize": pagesize,
        "pagination_unpaged": True,
        "truncation_risk": False,
    }
    return flags


def _row_flags(
    raw_rows: int,
    parsed_rows: int,
    *,
    page: Optional[int],
    pagesize: Optional[int],
) -> Dict[str, Any]:
    skipped_rows = max(raw_rows - parsed_rows, 0)
    flags = _page_flags(page, pagesize)
    flags.update(
        {
            "raw_rows": raw_rows,
            "parsed_rows": parsed_rows,
            "skipped_rows": skipped_rows,
            "truncation_risk": bool(pagesize and raw_rows >= pagesize),
        }
    )
    if raw_rows > 0 and parsed_rows == 0:
        flags["all_rows_skipped"] = True
    return flags


def _with_lineage_flags(lineage: LineageMeta, flags: Dict[str, Any]) -> LineageMeta:
    existing = dict(lineage.data_quality_flags or {})
    existing.update(flags)
    return replace(lineage, data_quality_flags=existing)


def _response_with_data(
    resp: AdapterResponse[Any],
    data: Any,
    *,
    raw_rows: int,
    page: Optional[int],
    pagesize: Optional[int],
    extra_flags: Optional[Dict[str, Any]] = None,
) -> AdapterResponse[Any]:
    flags = _row_flags(
        raw_rows,
        len(data) if isinstance(data, list) else 0,
        page=page,
        pagesize=pagesize,
    )
    if isinstance(resp.data, list):
        flags["bare_list_payload"] = True
    if extra_flags:
        flags.update(extra_flags)
    return AdapterResponse(
        data=data,
        lineage=_with_lineage_flags(resp.lineage, flags),
        rate_limit=resp.rate_limit,
    )


def _parse_valid_rows(
    rows: Optional[List[Any]],
    *,
    is_valid: Any,
    parser: Any,
    cutoff: datetime,
) -> tuple[List[Any], Dict[str, Any]]:
    parsed_rows: List[Any] = []
    warning_rows = 0
    warning_types: Dict[str, int] = {}

    for row in rows or []:
        if not isinstance(row, dict) or not is_valid(row):
            continue
        row_warning_types: Dict[str, int] = {}
        parsed_rows.append(
            parser(row, cutoff=cutoff, warning_types=row_warning_types)
        )
        if row_warning_types:
            warning_rows += 1
            for warning_type, count in row_warning_types.items():
                warning_types[warning_type] = warning_types.get(warning_type, 0) + count

    return parsed_rows, _knowledge_warning_flags(warning_rows, warning_types)


def _knowledge_warning_flags(
    warning_rows: int,
    warning_types: Dict[str, int],
) -> Dict[str, Any]:
    if warning_rows == 0:
        return {}
    return {
        "knowledge_timestamp_warning_rows": warning_rows,
        "knowledge_timestamp_warning_types": dict(sorted(warning_types.items())),
    }


def _increment_warning(warning_types: Optional[Dict[str, int]], warning_type: str) -> None:
    if warning_types is None:
        return
    warning_types[warning_type] = warning_types.get(warning_type, 0) + 1


def _parse_error_response(
    endpoint: str,
    lineage: LineageMeta,
    *,
    message: str = "Benzinga payload shape parse error",
    flags: Optional[Dict[str, Any]] = None,
) -> AdapterResponse[Any]:
    parse_flags = {"payload_shape_error": True}
    if flags:
        parse_flags.update(flags)
    return AdapterResponse(
        data=None,
        lineage=_with_lineage_flags(lineage, parse_flags),
        error=ProviderError(
            provider=PROVIDER,
            endpoint=endpoint,
            status_code=200,
            error_type="parse",
            message=message,
            retryable=False,
        ),
    )


def _rows_from_payload(
    resp: AdapterResponse[Any],
    *,
    endpoint: str,
    keys: Sequence[str],
    page: Optional[int],
    pagesize: Optional[int],
    allow_list_payload: bool = True,
) -> tuple[Optional[List[Any]], Optional[AdapterResponse[Any]]]:
    payload = resp.data
    if isinstance(payload, list):
        if allow_list_payload:
            return payload, None
        return None, _parse_error_response(
            endpoint,
            resp.lineage,
            flags=_page_flags(page, pagesize),
        )
    if not isinstance(payload, dict):
        return None, _parse_error_response(
            endpoint,
            resp.lineage,
            flags=_page_flags(page, pagesize),
        )
    if not payload:
        return [], None
    for key in keys:
        if key in payload:
            rows = payload.get(key)
            if isinstance(rows, list):
                return rows, None
            return None, _parse_error_response(
                endpoint,
                resp.lineage,
                flags=_page_flags(page, pagesize),
            )
    return None, _parse_error_response(
        endpoint,
        resp.lineage,
        flags=_page_flags(page, pagesize),
    )


def _insider_params(
    *,
    tickers: Optional[Union[str, Sequence[str]]],
    symbols: Optional[Union[str, Sequence[str]]],
    date_from: Optional[str],
    date_to: Optional[str],
    page: Optional[int],
    pagesize: int,
    updated: Optional[Union[int, str]],
) -> Dict[str, Any]:
    page_value = _positive_int_param(page, "page", required=False)
    pagesize_value = _positive_int_param(pagesize, "pagesize", required=True)
    date_from_value, date_to_value = _validate_date_range(date_from, date_to)
    ticker_param = _validated_ticker_filter(tickers, symbols)
    _require_ticker_or_bounded_dates(ticker_param, date_from_value, date_to_value)

    params: Dict[str, Any] = {"pagesize": pagesize_value}
    if page_value is not None:
        params["page"] = page_value

    if ticker_param:
        params["search_keys_type"] = "symbol"
        params["search_keys"] = ticker_param
    if date_from_value:
        params["date_from"] = date_from_value
    if date_to_value:
        params["date_to"] = date_to_value
    if updated is not None:
        params["updated_since"] = _updated_param(updated)
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
    return _has_nonblank_field(row, "ticker")


def _insider_filing_row_has_identity(row: Dict[str, Any]) -> bool:
    return (
        _has_nonblank_field(row, "id", "accession_number")
        and _has_nonblank_field(row, "company_symbol")
        and _has_valid_timestamp(row.get("filing_date"))
        and not _has_negative_decimal(row, "remaining_shares")
    )


def _insider_transaction_row_has_identity(row: Dict[str, Any]) -> bool:
    filing = _dict_or_empty(row.get("filing"))
    return _has_nonblank_field(row, "transaction_id") and (
        _has_nonblank_field(row, "company_symbol")
        or _has_nonblank_field(filing, "company_symbol")
    ) and (
        _has_valid_timestamp(row.get("filing_date"))
        or _has_valid_timestamp(filing.get("filing_date"))
    ) and _insider_transaction_numbers_are_valid(row)


def _news_row_has_identity(row: Dict[str, Any]) -> bool:
    if _has_nonblank_field(row, "id"):
        return True
    return _valid_http_url(_first_string(row, "url", "link"))


def _earnings_row_is_valid(row: Dict[str, Any]) -> bool:
    return (
        _calendar_row_has_usable_ticker(row)
        and _required_row_date(row, "date")
        and not _has_negative_decimal(
            row,
            "revenue",
            "revenue_est",
            "revenue_prior",
            "revenue_surprise",
        )
    )


def _guidance_row_is_valid(row: Dict[str, Any]) -> bool:
    return (
        _calendar_row_has_usable_ticker(row)
        and _required_row_date(row, "date")
        and not _has_negative_decimal(
            row,
            "revenue_guidance_est",
            "revenue_guidance_min",
            "revenue_guidance_max",
            "revenue_guidance_prior_min",
            "revenue_guidance_prior_max",
        )
    )


def _rating_row_is_valid(row: Dict[str, Any]) -> bool:
    return (
        _calendar_row_has_usable_ticker(row)
        and _required_row_date(row, "date")
        and not _has_negative_decimal(
            row,
            "pt_current",
            "pt_prior",
            "adjusted_pt_current",
            "adjusted_pt_prior",
        )
    )


def _offering_row_is_valid(row: Dict[str, Any]) -> bool:
    return (
        _calendar_row_has_usable_ticker(row)
        and _required_row_date(row, "date")
        and not _has_negative_decimal(
            row,
            "price",
            "number_shares",
            "dollar_shares",
            "proceeds",
        )
    )


def _dividend_row_is_valid(row: Dict[str, Any]) -> bool:
    return (
        _calendar_row_has_usable_ticker(row)
        and _required_row_date(row, "date", "ex_dividend_date")
        and not _has_negative_decimal(
            row,
            "dividend",
            "dividend_prior",
            "dividend_yield",
        )
    )


def _insider_transaction_numbers_are_valid(row: Dict[str, Any]) -> bool:
    return not _has_negative_decimal(
        row,
        "conversion_exercise_price_derivative",
        "post_transaction_quantity",
        "price_per_share",
        "remaining_underlying_shares",
        "shares",
        "underlying_shares",
    )


def _merger_acquisition_row_is_valid(row: Dict[str, Any]) -> bool:
    return (
        _has_nonblank_field(
            row,
            "id",
            "target_ticker",
            "target_cusip",
            "target_isin",
            "target_name",
        )
        and not _has_negative_deal_size(row.get("deal_size"))
    )


def _has_nonblank_field(row: Dict[str, Any], *keys: str) -> bool:
    for key in keys:
        text = _string_or_none(row.get(key))
        if text is not None and text.strip() != "":
            return True
    return False


def _valid_http_url(value: Optional[str]) -> bool:
    if value is None:
        return False
    text = value.strip()
    if not text or any(ord(ch) < 32 for ch in text):
        return False
    parsed = urlparse(text)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _row_iso_date_value(row: Dict[str, Any], key: str) -> Optional[str]:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None
    return text if parsed.strftime("%Y-%m-%d") == text else None


def _required_row_date(row: Dict[str, Any], *keys: str) -> bool:
    has_present = False
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        has_present = True
        if _row_iso_date_value(row, key) is None:
            return False
    return has_present


def _has_valid_timestamp(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return _timestamp_or_none(value) is not None


def _has_negative_decimal(row: Dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = row.get(key)
        if value is None or value == "":
            continue
        parsed = _decimal_or_none(value)
        if parsed is not None and parsed < 0:
            return True
    return False


def _has_negative_deal_size(value: Any) -> bool:
    if value is None or value == "":
        return False
    parsed = _decimal_or_none(value)
    return bool(parsed is not None and parsed < 0)


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


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _dict_rows(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, dict)]


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


def _parse_news_article_row(
    row: Dict[str, Any],
    *,
    cutoff: Optional[datetime] = None,
    warning_types: Optional[Dict[str, int]] = None,
) -> BenzingaNewsArticle:
    stock_rows = _stock_rows(row.get("stocks"))
    return BenzingaNewsArticle(
        id=_string_or_none(row.get("id")),
        created=_knowledge_timestamp_or_none(
            row.get("created")
            or row.get("created_at")
            or row.get("createdAt"),
            cutoff=cutoff,
            warning_types=warning_types,
            warning_type="news_created_future",
        ),
        updated=_knowledge_timestamp_or_none(
            row.get("updated")
            or row.get("updated_at")
            or row.get("updatedAt"),
            cutoff=cutoff,
            warning_types=warning_types,
            warning_type="news_updated_future",
        ),
        published=_knowledge_timestamp_or_none(
            row.get("published")
            or row.get("published_at")
            or row.get("publishedAt")
            or row.get("published_date"),
            cutoff=cutoff,
            warning_types=warning_types,
            warning_type="news_published_future",
        ),
        event_date=_string_or_none(row.get("date")),
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
        raw=deepcopy(row),
    )


def _parse_earnings_row(
    row: Dict[str, Any],
    *,
    cutoff: Optional[datetime] = None,
    warning_types: Optional[Dict[str, int]] = None,
) -> BenzingaEarnings:
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
        updated=_knowledge_timestamp_or_none(
            _first_present(row, "updated", "updated_at", "last_updated"),
            cutoff=cutoff,
            warning_types=warning_types,
            warning_type="calendar_updated_future",
        ),
        raw=deepcopy(row),
    )


def _parse_guidance_row(
    row: Dict[str, Any],
    *,
    cutoff: Optional[datetime] = None,
    warning_types: Optional[Dict[str, int]] = None,
) -> BenzingaGuidance:
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
        updated=_knowledge_timestamp_or_none(
            _first_present(row, "updated", "updated_at", "last_updated"),
            cutoff=cutoff,
            warning_types=warning_types,
            warning_type="calendar_updated_future",
        ),
        raw=deepcopy(row),
    )


def _parse_rating_row(
    row: Dict[str, Any],
    *,
    cutoff: Optional[datetime] = None,
    warning_types: Optional[Dict[str, int]] = None,
) -> BenzingaRating:
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
        updated=_knowledge_timestamp_or_none(
            _first_present(row, "updated", "updated_at", "last_updated"),
            cutoff=cutoff,
            warning_types=warning_types,
            warning_type="calendar_updated_future",
        ),
        raw=deepcopy(row),
    )


def _parse_offering_row(
    row: Dict[str, Any],
    *,
    cutoff: Optional[datetime] = None,
    warning_types: Optional[Dict[str, int]] = None,
) -> BenzingaOffering:
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
        updated=_knowledge_timestamp_or_none(
            _first_present(row, "updated", "updated_at", "last_updated"),
            cutoff=cutoff,
            warning_types=warning_types,
            warning_type="calendar_updated_future",
        ),
        raw=deepcopy(row),
    )


def _parse_dividend_row(
    row: Dict[str, Any],
    *,
    cutoff: Optional[datetime] = None,
    warning_types: Optional[Dict[str, int]] = None,
) -> BenzingaDividend:
    return BenzingaDividend(
        id=_string_or_none(row.get("id")),
        ticker=_string_or_none(row.get("ticker")),
        name=_string_or_none(row.get("name")),
        exchange=_string_or_none(row.get("exchange")),
        currency=_string_or_none(row.get("currency")),
        cusip=_first_identifier(row, "cusip", kind="cusip"),
        isin=_first_identifier(row, "isin", kind="isin"),
        date=_string_or_none(row.get("date")),
        ex_dividend_date=_string_or_none(row.get("ex_dividend_date")),
        payable_date=_string_or_none(row.get("payable_date")),
        record_date=_string_or_none(row.get("record_date")),
        dividend=_decimal_or_none(row.get("dividend")),
        dividend_prior=_decimal_or_none(row.get("dividend_prior")),
        dividend_type=_string_or_none(row.get("dividend_type")),
        dividend_yield=_decimal_or_none(row.get("dividend_yield")),
        frequency=_int_or_none(row.get("frequency")),
        confirmed=_bool_or_none(row.get("confirmed")),
        end_regular_dividend=_bool_or_none(row.get("end_regular_dividend")),
        period=_string_or_none(row.get("period")),
        year=_int_or_none(row.get("year")),
        importance=_int_or_none(row.get("importance")),
        notes=_string_or_none(row.get("notes")),
        updated=_knowledge_timestamp_or_none(
            _first_present(row, "updated", "updated_at", "last_updated"),
            cutoff=cutoff,
            warning_types=warning_types,
            warning_type="calendar_updated_future",
        ),
        raw=deepcopy(row),
    )


def _parse_insider_filing_row(
    row: Dict[str, Any],
    *,
    cutoff: Optional[datetime] = None,
    warning_types: Optional[Dict[str, int]] = None,
) -> BenzingaInsiderFiling:
    owner = _dict_or_empty(row.get("owner"))
    _warn_future_knowledge_fields(
        row,
        ("accepted", "accepted_at", "accepted_date", "acceptance_date"),
        cutoff=cutoff,
        warning_types=warning_types,
        warning_type="insider_accepted_future",
    )
    return BenzingaInsiderFiling(
        id=_string_or_none(row.get("id")),
        accession_number=_string_or_none(row.get("accession_number")),
        company_cik=_string_or_none(row.get("company_cik")),
        company_name=_string_or_none(row.get("company_name")),
        company_symbol=_string_or_none(row.get("company_symbol")),
        filing_date=_knowledge_timestamp_or_none(
            row.get("filing_date"),
            cutoff=cutoff,
            warning_types=warning_types,
            warning_type="insider_filing_date_future",
        ),
        form_type=_string_or_none(row.get("form_type")),
        html_url=_string_or_none(row.get("html_url")),
        is_10b5=_bool_or_none(row.get("is_10b5")),
        insider_cik=_string_or_none(owner.get("insider_cik")),
        insider_name=_string_or_none(owner.get("insider_name")),
        insider_title=_string_or_none(owner.get("insider_title")),
        is_director=_bool_or_none(owner.get("is_director")),
        is_officer=_bool_or_none(owner.get("is_officer")),
        is_ten_percent_owner=_bool_or_none(owner.get("is_ten_percent_owner")),
        raw_signature=_string_or_none(owner.get("raw_signature")),
        remaining_shares=_decimal_or_none(row.get("remaining_shares")),
        traded_percentage=_string_or_none(row.get("traded_percentage")),
        footnotes=_dict_rows(row.get("footnotes")),
        transactions=_dict_rows(row.get("transactions")),
        owner=owner,
        updated=_knowledge_timestamp_or_none(
            _first_present(row, "updated", "updated_at", "last_updated"),
            cutoff=cutoff,
            warning_types=warning_types,
            warning_type="insider_updated_future",
        ),
        raw=deepcopy(row),
    )


def _parse_insider_transaction_row(
    row: Dict[str, Any],
    *,
    cutoff: Optional[datetime] = None,
    warning_types: Optional[Dict[str, int]] = None,
) -> BenzingaInsiderTransaction:
    filing = _dict_or_empty(row.get("filing"))
    owner = _dict_or_empty(row.get("owner"))
    if not owner:
        owner = _dict_or_empty(filing.get("owner"))
    _warn_future_knowledge_fields(
        row,
        ("accepted", "accepted_at", "accepted_date", "acceptance_date"),
        cutoff=cutoff,
        warning_types=warning_types,
        warning_type="insider_accepted_future",
    )
    _warn_future_knowledge_fields(
        filing,
        ("accepted", "accepted_at", "accepted_date", "acceptance_date"),
        cutoff=cutoff,
        warning_types=warning_types,
        warning_type="insider_accepted_future",
    )

    return BenzingaInsiderTransaction(
        transaction_id=_string_or_none(row.get("transaction_id")),
        accession_number=_string_or_none(
            row.get("accession_number") or filing.get("accession_number")
        ),
        company_cik=_string_or_none(row.get("company_cik") or filing.get("company_cik")),
        company_name=_string_or_none(
            row.get("company_name") or filing.get("company_name")
        ),
        company_symbol=_string_or_none(
            row.get("company_symbol") or filing.get("company_symbol")
        ),
        filing_date=_knowledge_timestamp_or_none(
            row.get("filing_date") or filing.get("filing_date"),
            cutoff=cutoff,
            warning_types=warning_types,
            warning_type="insider_filing_date_future",
        ),
        form_type=_string_or_none(row.get("form_type") or filing.get("form_type")),
        filing_id=_string_or_none(row.get("filing_id") or filing.get("id")),
        html_url=_string_or_none(row.get("html_url") or filing.get("html_url")),
        insider_cik=_string_or_none(owner.get("insider_cik")),
        insider_name=_string_or_none(owner.get("insider_name")),
        insider_title=_string_or_none(owner.get("insider_title")),
        is_director=_bool_or_none(owner.get("is_director")),
        is_officer=_bool_or_none(owner.get("is_officer")),
        is_ten_percent_owner=_bool_or_none(owner.get("is_ten_percent_owner")),
        raw_signature=_string_or_none(owner.get("raw_signature")),
        acquired_or_disposed=_string_or_none(row.get("acquired_or_disposed")),
        conversion_exercise_price_derivative=_decimal_or_none(
            row.get("conversion_exercise_price_derivative")
        ),
        date_deemed_execution=_timestamp_or_none(row.get("date_deemed_execution")),
        date_exercisable=_timestamp_or_none(row.get("date_exercisable")),
        date_expiration=_timestamp_or_none(row.get("date_expiration")),
        date_transaction=_timestamp_or_none(row.get("date_transaction")),
        is_derivative=_bool_or_none(row.get("is_derivative")),
        ownership=_string_or_none(row.get("ownership")),
        post_transaction_quantity=_decimal_or_none(
            row.get("post_transaction_quantity")
        ),
        price_per_share=_decimal_or_none(row.get("price_per_share")),
        remaining_underlying_shares=_decimal_or_none(
            row.get("remaining_underlying_shares")
        ),
        security_title=_string_or_none(row.get("security_title")),
        shares=_decimal_or_none(row.get("shares")),
        transaction_code=_string_or_none(row.get("transaction_code")),
        underlying_security_title=_string_or_none(
            row.get("underlying_security_title")
        ),
        underlying_shares=_decimal_or_none(row.get("underlying_shares")),
        voluntarily_reported=_bool_or_none(row.get("voluntarily_reported")),
        owner=owner,
        filing=filing,
        updated=_knowledge_timestamp_or_none(
            _first_present(row, "updated", "updated_at", "last_updated"),
            cutoff=cutoff,
            warning_types=warning_types,
            warning_type="insider_updated_future",
        ),
        raw=deepcopy(row),
    )


def _parse_merger_acquisition_row(
    row: Dict[str, Any],
    *,
    cutoff: Optional[datetime] = None,
    warning_types: Optional[Dict[str, int]] = None,
) -> BenzingaMergerAcquisition:
    _warn_future_knowledge_fields(
        row,
        (
            "announced",
            "announced_at",
            "date_announced",
            "published",
            "published_at",
            "created",
            "created_at",
        ),
        cutoff=cutoff,
        warning_types=warning_types,
        warning_type="ma_publication_future",
    )
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
        updated=_knowledge_epoch_int_or_none(
            row.get("updated"),
            cutoff=cutoff,
            warning_types=warning_types,
            warning_type="ma_updated_future",
        ),
        raw=deepcopy(row),
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


def _first_present(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
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


def _knowledge_timestamp_or_none(
    value: Any,
    *,
    cutoff: Optional[datetime],
    warning_types: Optional[Dict[str, int]],
    warning_type: str,
) -> Optional[datetime]:
    parsed = _timestamp_or_none(value)
    if parsed is None:
        return None
    if cutoff is not None and parsed > cutoff + KNOWLEDGE_TIMESTAMP_FUTURE_TOLERANCE:
        _increment_warning(warning_types, warning_type)
        return None
    return parsed


def _knowledge_epoch_int_or_none(
    value: Any,
    *,
    cutoff: Optional[datetime],
    warning_types: Optional[Dict[str, int]],
    warning_type: str,
) -> Optional[int]:
    parsed = _int_or_none(value)
    if parsed is None:
        return None
    parsed_timestamp = _timestamp_or_none(parsed)
    if (
        parsed_timestamp is not None
        and cutoff is not None
        and parsed_timestamp > cutoff + KNOWLEDGE_TIMESTAMP_FUTURE_TOLERANCE
    ):
        _increment_warning(warning_types, warning_type)
        return None
    return parsed


def _warn_future_knowledge_fields(
    row: Dict[str, Any],
    keys: Sequence[str],
    *,
    cutoff: Optional[datetime],
    warning_types: Optional[Dict[str, int]],
    warning_type: str,
) -> None:
    for key in keys:
        value = row.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        _knowledge_timestamp_or_none(
            value,
            cutoff=cutoff,
            warning_types=warning_types,
            warning_type=warning_type,
        )


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
