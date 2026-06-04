"""
SEC EDGAR public-data adapter.

Primary source for:
  - Company CIK/ticker/exchange mappings
  - EDGAR submission history by filer
  - Form 25 / 25-NSE delisting-notice events

Does not write to DB. Returns AdapterResponse with LineageMeta.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import requests

from alpha.data.config import SecEdgarConfig
from alpha.data.contracts import (
    AdapterResponse,
    LineageMeta,
    ProviderError,
    aware_utc_or_none,
    stable_hash,
    utcnow,
)

PROVIDER = "SEC_EDGAR"
SOURCE_AUTHORITY = "SEC_EDGAR"
COMPANY_TICKERS_EXCHANGE_ENDPOINT = "/files/company_tickers_exchange.json"
SUBMISSIONS_ENDPOINT_TEMPLATE = "/submissions/CIK{cik}.json"
SURVIVORSHIP_EVENTS_ENDPOINT = "sec_edgar_survivorship_events"
FORM4_TRANSACTIONS_ENDPOINT = "sec_edgar_form4_transactions"
FORM_25_FORMS = ("25", "25-NSE")
FORM_4_FORMS = ("4", "4/A")
SEC_MAX_REQUESTS_PER_SECOND = 10
MAX_SUBMISSIONS_OVERFLOW_PAGES = 20
EDGAR_ACCEPTANCE_TIMEZONE = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SecCompanyTicker:
    """One row from SEC's company_tickers_exchange mapping."""

    cik: int
    cik_str: str
    ticker: str
    company_name: str
    exchange: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class SecEdgarFiling:
    """Normalized filing metadata from the EDGAR submissions API."""

    cik: str
    accession_number: str
    form: str
    filing_date: Optional[date]
    report_date: Optional[date]
    acceptance_datetime: Optional[datetime]
    primary_document: Optional[str] = None
    primary_doc_description: Optional[str] = None
    act: Optional[str] = None
    file_number: Optional[str] = None
    film_number: Optional[str] = None
    items: Optional[str] = None
    size: Optional[int] = None
    is_xbrl: Optional[bool] = None
    is_inline_xbrl: Optional[bool] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class SecForm4Transaction:
    """One parsed owner/transaction row from a SEC Form 4 filing."""

    transaction_id: str
    accession_number: str
    filing_form: str
    filing_date: Optional[date]
    filing_accepted_at: Optional[datetime]
    issuer_cik: Optional[str]
    issuer_name: Optional[str]
    ticker: Optional[str]
    insider_cik: Optional[str]
    insider_name: Optional[str]
    insider_state: Optional[str]
    insider_roles: Dict[str, Any]
    transaction_date: Optional[date]
    transaction_code: Optional[str]
    acquired_disposed_code: Optional[str]
    security_title: Optional[str]
    shares: Optional[float]
    price_per_share: Optional[float]
    ownership_type: Optional[str]
    is_10b5_1: Optional[bool]
    raw: Optional[Dict[str, Any]] = None


class SecEdgarAdapter:
    """SEC EDGAR REST adapter returning typed public filing metadata."""

    requires_cik_for_survivorship_events = True

    def __init__(
        self,
        config: SecEdgarConfig,
        session: Optional[requests.Session] = None,
    ):
        self._config = config
        self._session = session or requests.Session()
        self._headers = {
            "User-Agent": config.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        }
        self._company_tickers_cache: Optional[List[SecCompanyTicker]] = None
        self._company_tickers_lineage: Optional[LineageMeta] = None
        self._last_request_monotonic: Optional[float] = None

    def _request(
        self,
        endpoint: str,
        *,
        base_url: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[Any]:
        request_base_url = base_url or self._config.data_base_url
        url = f"{request_base_url}{endpoint}"
        quality_flags = _request_data_quality_flags(url)
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
                        source_authority=SOURCE_AUTHORITY,
                        data_quality_flags=quality_flags,
                    ),
                    error=ProviderError(
                        provider=PROVIDER,
                        endpoint=endpoint,
                        status_code=None,
                        error_type="validation",
                        message="SEC EDGAR adapter asof timestamp must be timezone-aware datetime",
                        retryable=False,
                    ),
                )

        try:
            self._rate_gate()
            resp = self._session.get(
                url,
                params=params or {},
                headers=self._headers,
                timeout=30,
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
                    source_authority=SOURCE_AUTHORITY,
                    data_quality_flags=quality_flags,
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
                    source_authority=SOURCE_AUTHORITY,
                    data_quality_flags=quality_flags,
                ),
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=None,
                    error_type="http",
                    message=f"SEC EDGAR request failed: {exc.__class__.__name__}",
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
            source_authority=SOURCE_AUTHORITY,
            data_quality_flags=quality_flags,
        )

        if resp.status_code == 429:
            retry_after = _clean_string(resp.headers.get("Retry-After"))
            message = "SEC EDGAR rate limit exceeded"
            if retry_after:
                message = f"{message}; Retry-After: {retry_after}"
            return AdapterResponse(
                data=None,
                lineage=replace(
                    lineage,
                    data_quality_flags={
                        **(lineage.data_quality_flags or {}),
                        **({"retry_after": retry_after} if retry_after else {}),
                    },
                ),
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=429,
                    error_type="rate_limit",
                    message=message,
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
                    message=f"SEC EDGAR auth/fair-access error: {resp.status_code}",
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
                    message=f"SEC EDGAR HTTP {resp.status_code}",
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

    def _request_text(
        self,
        endpoint: str,
        *,
        base_url: Optional[str] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[str]:
        request_base_url = base_url or self._config.sec_base_url
        url = f"{request_base_url}{endpoint}"
        quality_flags = _request_data_quality_flags(url)
        request_ts = utcnow()
        asof_ts = aware_utc_or_none(asof) if asof is not None else request_ts
        if asof_ts is None:
            return AdapterResponse(
                data=None,
                lineage=LineageMeta(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    request_timestamp=request_ts,
                    asof_timestamp=request_ts,
                    raw_payload_hash="",
                    source_authority=SOURCE_AUTHORITY,
                    data_quality_flags=quality_flags,
                ),
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=None,
                    error_type="validation",
                    message="SEC EDGAR adapter asof timestamp must be timezone-aware datetime",
                    retryable=False,
                ),
            )
        try:
            self._rate_gate()
            resp = self._session.get(
                url,
                params={},
                headers=self._headers,
                timeout=30,
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
                    source_authority=SOURCE_AUTHORITY,
                    data_quality_flags=quality_flags,
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
                    source_authority=SOURCE_AUTHORITY,
                    data_quality_flags=quality_flags,
                ),
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=None,
                    error_type="http",
                    message=f"SEC EDGAR request failed: {exc.__class__.__name__}",
                    retryable=True,
                ),
            )

        lineage = LineageMeta(
            provider=PROVIDER,
            endpoint=endpoint,
            request_timestamp=request_ts,
            asof_timestamp=asof_ts,
            raw_payload_hash=stable_hash(resp.text),
            freshness_seconds=(utcnow() - request_ts).total_seconds(),
            source_authority=SOURCE_AUTHORITY,
            data_quality_flags=quality_flags,
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
                    message="SEC EDGAR rate limit exceeded",
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
                    message=f"SEC EDGAR auth/fair-access error: {resp.status_code}",
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
                    message=f"SEC EDGAR HTTP {resp.status_code}",
                    retryable=resp.status_code >= 500,
                ),
            )
        return AdapterResponse(data=resp.text, lineage=lineage)

    def _rate_gate(self) -> None:
        interval = 1.0 / SEC_MAX_REQUESTS_PER_SECOND
        now = time.monotonic()
        if self._last_request_monotonic is not None:
            elapsed = now - self._last_request_monotonic
            if elapsed < interval:
                time.sleep(interval - elapsed)
                now = time.monotonic()
        self._last_request_monotonic = now

    def get_company_tickers(
        self,
        *,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[SecCompanyTicker]]:
        """Fetch SEC's current company CIK/ticker/exchange mapping.

        Warm-cache hits reuse current snapshot bytes, which is traceable via
        the preserved request_timestamp/cache_hit=True but not PIT-historical.
        """

        if asof is not None and _asof_utc(asof) is None:
            resp = self._request(
                COMPANY_TICKERS_EXCHANGE_ENDPOINT,
                base_url=self._config.sec_base_url,
                asof=asof,
            )
            return AdapterResponse(data=None, lineage=resp.lineage, error=resp.error)
        if self._company_tickers_cache is not None:
            cache_lineage = self._company_tickers_lineage
            if cache_lineage is not None:
                cache_asof = _asof_utc(asof)
                cache_lineage = replace(
                    cache_lineage,
                    asof_timestamp=cache_asof or utcnow(),
                    data_quality_flags={
                        **(cache_lineage.data_quality_flags or {}),
                        "cache_hit": True,
                    },
                )
            return AdapterResponse(
                data=list(self._company_tickers_cache),
                lineage=cache_lineage,
            )

        resp = self._request(
            COMPANY_TICKERS_EXCHANGE_ENDPOINT,
            base_url=self._config.sec_base_url,
            asof=asof,
        )
        if not resp.ok:
            return AdapterResponse(data=None, lineage=resp.lineage, error=resp.error)
        rows = _parse_company_tickers(resp.data)
        self._company_tickers_cache = list(rows)
        self._company_tickers_lineage = resp.lineage
        return AdapterResponse(data=rows, lineage=resp.lineage)

    def get_company_ticker(
        self,
        ticker: str,
        *,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[Optional[SecCompanyTicker]]:
        """Resolve a ticker to SEC's current CIK mapping."""

        resp = self.get_company_tickers(asof=asof)
        if not resp.ok:
            return AdapterResponse(data=None, lineage=resp.lineage, error=resp.error)
        normalized = str(ticker or "").strip().upper()
        for row in resp.data or []:
            if row.ticker == normalized:
                return AdapterResponse(data=row, lineage=resp.lineage)
        return AdapterResponse(data=None, lineage=resp.lineage)

    def get_company_submissions(
        self,
        cik: Any,
        *,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[Dict[str, Any]]:
        """Fetch the EDGAR submissions JSON for one filer CIK."""

        cik10 = _cik10(cik)
        if cik10 is None:
            request_ts = utcnow()
            asof_ts = aware_utc_or_none(asof) if asof is not None else request_ts
            if asof_ts is None:
                asof_ts = request_ts
            endpoint = SUBMISSIONS_ENDPOINT_TEMPLATE.format(cik="INVALID")
            return AdapterResponse(
                data=None,
                lineage=LineageMeta(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    request_timestamp=request_ts,
                    asof_timestamp=asof_ts,
                    raw_payload_hash="",
                    source_authority=SOURCE_AUTHORITY,
                ),
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=None,
                    error_type="validation",
                    message="SEC EDGAR CIK must contain at least one digit",
                    retryable=False,
                ),
            )
        endpoint = SUBMISSIONS_ENDPOINT_TEMPLATE.format(cik=cik10)
        return self._request(endpoint, asof=asof)

    def get_filings(
        self,
        cik: Any,
        *,
        forms: Optional[Sequence[str]] = None,
        from_date: Optional[Any] = None,
        to_date: Optional[Any] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[SecEdgarFiling]]:
        """Fetch and PIT-filter recent EDGAR filings for one filer."""

        asof_validation = _asof_utc(asof)
        if asof_validation is None and asof is not None:
            request_ts = utcnow()
            endpoint = SUBMISSIONS_ENDPOINT_TEMPLATE.format(cik=_cik10(cik) or "INVALID")
            return AdapterResponse(
                data=None,
                lineage=LineageMeta(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    request_timestamp=request_ts,
                    asof_timestamp=request_ts,
                    raw_payload_hash="",
                    source_authority=SOURCE_AUTHORITY,
                ),
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=None,
                    error_type="validation",
                    message="SEC EDGAR adapter asof timestamp must be timezone-aware datetime",
                    retryable=False,
                ),
            )
        asof_ts = asof_validation or utcnow()
        resp = self.get_company_submissions(cik, asof=asof_ts)
        if not resp.ok:
            return AdapterResponse(data=None, lineage=resp.lineage, error=resp.error)

        cik10 = _cik10(cik) or ""
        parsed = _parse_recent_filings(cik10, resp.data or {})
        overflow_flags = _overflow_quality_flags(resp.data or {})
        if forms:
            overflow_resp = self._get_overflow_filings(
                cik10,
                resp.data or {},
                asof=asof_ts,
            )
            overflow_flags = overflow_resp["flags"]
            if overflow_resp["error"] is not None:
                return AdapterResponse(
                    data=None,
                    lineage=replace(
                        resp.lineage,
                        data_quality_flags={
                            **(resp.lineage.data_quality_flags or {}),
                            **overflow_flags,
                        },
                    ),
                    error=overflow_resp["error"],
                )
            parsed.extend(overflow_resp["filings"])
        parsed = _dedupe_and_sort_filings(parsed)
        filtered, flags = _filter_filings(
            parsed,
            forms=forms,
            from_date=_coerce_date(from_date),
            to_date=_coerce_date(to_date),
            asof=asof_ts,
        )
        lineage = replace(
            resp.lineage,
            data_quality_flags={
                **(resp.lineage.data_quality_flags or {}),
                **overflow_flags,
                **flags,
            },
        )
        return AdapterResponse(data=filtered, lineage=lineage)

    def get_form4_filings(
        self,
        cik: Any,
        *,
        from_date: Optional[Any] = None,
        to_date: Optional[Any] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[SecEdgarFiling]]:
        """Fetch PIT-filtered Form 4 / 4-A filing metadata for one issuer CIK."""

        return self.get_filings(
            cik,
            forms=FORM_4_FORMS,
            from_date=from_date,
            to_date=to_date,
            asof=asof,
        )

    def get_filing_document(
        self,
        filing: SecEdgarFiling,
        *,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[str]:
        """Fetch the primary document bytes for a filing accession."""

        if not filing.primary_document:
            request_ts = utcnow()
            asof_ts = aware_utc_or_none(asof) if asof is not None else request_ts
            if asof_ts is None:
                asof_ts = request_ts
            return AdapterResponse(
                data=None,
                lineage=LineageMeta(
                    provider=PROVIDER,
                    endpoint="sec_edgar_filing_document",
                    request_timestamp=request_ts,
                    asof_timestamp=asof_ts,
                    raw_payload_hash="",
                    source_authority=SOURCE_AUTHORITY,
                ),
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint="sec_edgar_filing_document",
                    status_code=None,
                    error_type="validation",
                    message="filing primary_document is required",
                    retryable=False,
                ),
            )
        accession_dir = filing.accession_number.replace("-", "")
        try:
            cik_int = str(int(filing.cik))
        except (TypeError, ValueError):
            cik_int = str(filing.cik).lstrip("0") or "0"
        document = str(filing.primary_document).lstrip("/")
        endpoint = f"/Archives/edgar/data/{cik_int}/{accession_dir}/{document}"
        return self._request_text(endpoint, base_url=self._config.sec_base_url, asof=asof)

    def get_form4_transactions(
        self,
        cik: Any,
        *,
        from_date: Optional[Any] = None,
        to_date: Optional[Any] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[List[SecForm4Transaction]]:
        """Fetch and parse SEC Form 4 owner/transaction rows for one issuer."""

        filings_resp = self.get_form4_filings(
            cik,
            from_date=from_date,
            to_date=to_date,
            asof=asof,
        )
        if not filings_resp.ok:
            return AdapterResponse(
                data=None,
                lineage=filings_resp.lineage,
                error=filings_resp.error,
            )
        transactions: List[SecForm4Transaction] = []
        document_hashes: List[str] = []
        document_count = 0
        for filing in filings_resp.data or []:
            doc_resp = self.get_filing_document(filing, asof=asof)
            if not doc_resp.ok:
                return AdapterResponse(
                    data=None,
                    lineage=replace(
                        filings_resp.lineage,
                        endpoint=FORM4_TRANSACTIONS_ENDPOINT,
                        data_quality_flags={
                            **(filings_resp.lineage.data_quality_flags or {}),
                            "document_fetch_error_accession": filing.accession_number,
                        },
                    ),
                    error=doc_resp.error,
                )
            document_count += 1
            document_hashes.append(doc_resp.lineage.raw_payload_hash)
            parsed, parse_error = _parse_form4_transactions_xml(
                doc_resp.data or "",
                filing=filing,
            )
            if parse_error is not None:
                return AdapterResponse(
                    data=None,
                    lineage=replace(
                        filings_resp.lineage,
                        endpoint=FORM4_TRANSACTIONS_ENDPOINT,
                        data_quality_flags={
                            **(filings_resp.lineage.data_quality_flags or {}),
                            "parse_error_accession": filing.accession_number,
                        },
                    ),
                    error=ProviderError(
                        provider=PROVIDER,
                        endpoint=FORM4_TRANSACTIONS_ENDPOINT,
                        status_code=200,
                        error_type="parse",
                        message=parse_error,
                        retryable=False,
                    ),
                )
            transactions.extend(parsed)
        lineage = replace(
            filings_resp.lineage,
            endpoint=FORM4_TRANSACTIONS_ENDPOINT,
            raw_payload_hash=stable_hash({
                "filing_hash": filings_resp.lineage.raw_payload_hash,
                "document_hashes": document_hashes,
                "transaction_ids": [row.transaction_id for row in transactions],
            }),
            data_quality_flags={
                **(filings_resp.lineage.data_quality_flags or {}),
                "document_count": document_count,
                "transaction_count": len(transactions),
            },
        )
        return AdapterResponse(data=transactions, lineage=lineage)

    def _get_overflow_filings(
        self,
        cik: str,
        payload: Dict[str, Any],
        *,
        asof: datetime,
    ) -> Dict[str, Any]:
        names = _submission_overflow_file_names(payload)
        selected = names[:MAX_SUBMISSIONS_OVERFLOW_PAGES]
        filings: List[SecEdgarFiling] = []
        flags = {
            "overflow_pages_available": len(names),
            "overflow_pages_fetched": 0,
            "truncated": len(names) > MAX_SUBMISSIONS_OVERFLOW_PAGES,
        }
        for name in selected:
            endpoint = f"/submissions/{name}"
            resp = self._request(endpoint, asof=asof)
            if not resp.ok:
                flags["overflow_fetch_error_endpoint"] = endpoint
                return {"filings": filings, "flags": flags, "error": resp.error}
            filings.extend(_parse_recent_filings(cik, resp.data or {}))
            flags["overflow_pages_fetched"] += 1
        return {"filings": filings, "flags": flags, "error": None}

    def get_survivorship_events(
        self,
        ticker: str,
        *,
        from_date: Optional[Any] = None,
        to_date: Optional[Any] = None,
        asof: Optional[datetime] = None,
        cik: Optional[Any] = None,
    ) -> AdapterResponse[List[Dict[str, Any]]]:
        """Return source-backed Form 25 / 25-NSE delisting events for a ticker."""

        asof_validation = _asof_utc(asof)
        if asof_validation is None and asof is not None:
            request_ts = utcnow()
            return AdapterResponse(
                data=None,
                lineage=LineageMeta(
                    provider=PROVIDER,
                    endpoint=SURVIVORSHIP_EVENTS_ENDPOINT,
                    request_timestamp=request_ts,
                    asof_timestamp=request_ts,
                    raw_payload_hash="",
                    source_authority=SOURCE_AUTHORITY,
                ),
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=SURVIVORSHIP_EVENTS_ENDPOINT,
                    status_code=None,
                    error_type="validation",
                    message="SEC EDGAR adapter asof timestamp must be timezone-aware datetime",
                    retryable=False,
                ),
            )
        asof_ts = asof_validation or utcnow()
        resolved_cik = _cik10(cik) if cik is not None else None
        ticker_row: Optional[SecCompanyTicker] = None

        if resolved_cik is None:
            ticker_resp = self.get_company_ticker(ticker, asof=asof_ts)
            if not ticker_resp.ok:
                return AdapterResponse(
                    data=None,
                    lineage=ticker_resp.lineage,
                    error=ticker_resp.error,
                )
            ticker_row = ticker_resp.data
            if ticker_row is None:
                lineage = replace(
                    ticker_resp.lineage,
                    endpoint=SURVIVORSHIP_EVENTS_ENDPOINT,
                    raw_payload_hash=stable_hash(
                        {
                            "ticker": ticker,
                            "from": _date_iso(from_date),
                            "to": _date_iso(to_date),
                            "events": [],
                        }
                    ),
                    data_quality_flags={
                        **(ticker_resp.lineage.data_quality_flags or {}),
                        "ticker_resolved": False,
                    },
                )
                return AdapterResponse(
                    data=None,
                    lineage=lineage,
                    error=ProviderError(
                        provider=PROVIDER,
                        endpoint=SURVIVORSHIP_EVENTS_ENDPOINT,
                        status_code=None,
                        error_type="unresolved_entity",
                        message=(
                            "SEC EDGAR survivorship lookup requires a resolved "
                            "CIK; ticker-only lookup failed"
                        ),
                        retryable=False,
                    ),
                )
            resolved_cik = ticker_row.cik_str

        filings_resp = self.get_filings(
            resolved_cik,
            forms=FORM_25_FORMS,
            from_date=from_date,
            to_date=to_date,
            asof=asof_ts,
        )
        if not filings_resp.ok:
            return AdapterResponse(
                data=None,
                lineage=filings_resp.lineage,
                error=filings_resp.error,
            )
        filing_flags = filings_resp.lineage.data_quality_flags or {}
        overflow_pages_available = filing_flags.get("overflow_pages_available")
        if filing_flags.get("truncated") is True or (
            isinstance(overflow_pages_available, int)
            and overflow_pages_available > MAX_SUBMISSIONS_OVERFLOW_PAGES
        ):
            lineage = replace(
                filings_resp.lineage,
                endpoint=SURVIVORSHIP_EVENTS_ENDPOINT,
                raw_payload_hash=stable_hash(
                    {
                        "ticker": ticker,
                        "cik": resolved_cik,
                        "from": _date_iso(from_date),
                        "to": _date_iso(to_date),
                        "truncated": True,
                    }
                ),
                data_quality_flags={
                    **filing_flags,
                    "ticker_resolved": ticker_row is not None or cik is not None,
                    "forms": list(FORM_25_FORMS),
                },
            )
            return AdapterResponse(
                data=None,
                lineage=lineage,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=SURVIVORSHIP_EVENTS_ENDPOINT,
                    status_code=None,
                    error_type="incomplete_window",
                    message=(
                        "SEC EDGAR survivorship window truncated; "
                        "Form 25 completeness not guaranteed"
                    ),
                    retryable=False,
                ),
            )

        events = [
            _filing_to_survivorship_event(
                filing,
                ticker=ticker,
                ticker_row=ticker_row,
            )
            for filing in filings_resp.data or []
        ]
        lineage = replace(
            filings_resp.lineage,
            endpoint=SURVIVORSHIP_EVENTS_ENDPOINT,
            raw_payload_hash=stable_hash(
                {
                    "ticker": ticker,
                    "cik": resolved_cik,
                    "from": _date_iso(from_date),
                    "to": _date_iso(to_date),
                    "events": events,
                }
            ),
            data_quality_flags={
                **(filings_resp.lineage.data_quality_flags or {}),
                "ticker_resolved": ticker_row is not None or cik is not None,
                "survivorship_event_count": len(events),
                "forms": list(FORM_25_FORMS),
            },
        )
        return AdapterResponse(data=events, lineage=lineage)


def _parse_form4_transactions_xml(
    text: str,
    *,
    filing: SecEdgarFiling,
) -> Tuple[List[SecForm4Transaction], Optional[str]]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        return [], f"Form 4 XML parse error: {exc}"

    issuer = _first_child(root, "issuer")
    issuer_cik = _cik10(_first_text(issuer, "issuerCik") if issuer is not None else None)
    issuer_name = _first_text(issuer, "issuerName") if issuer is not None else None
    ticker = _first_text(issuer, "issuerTradingSymbol") if issuer is not None else None
    owners = _reporting_owners(root)
    if not owners:
        owners = [{
            "insider_cik": None,
            "insider_name": None,
            "insider_state": None,
            "roles": {},
        }]
    remarks = (_first_text(root, "remarks") or "").lower()
    document_10b5_1 = "10b5-1" in remarks or "10b5" in remarks

    transaction_nodes = _descendants(root, "nonDerivativeTransaction")
    transaction_nodes.extend(_descendants(root, "derivativeTransaction"))
    rows: List[SecForm4Transaction] = []
    for tx_index, tx_node in enumerate(transaction_nodes):
        transaction_date = _coerce_date(_nested_text(tx_node, "transactionDate", "value"))
        transaction_code = _nested_text(tx_node, "transactionCoding", "transactionCode")
        acquired_disposed = _nested_text(
            tx_node,
            "transactionAmounts",
            "transactionAcquiredDisposedCode",
            "value",
        )
        shares = _optional_float(_nested_text(
            tx_node,
            "transactionAmounts",
            "transactionShares",
            "value",
        ))
        price = _optional_float(_nested_text(
            tx_node,
            "transactionAmounts",
            "transactionPricePerShare",
            "value",
        ))
        security_title = _nested_text(tx_node, "securityTitle", "value")
        ownership_type = _nested_text(
            tx_node,
            "ownershipNature",
            "directOrIndirectOwnership",
            "value",
        )
        tx_raw = {
            "security_title": security_title,
            "transaction_date": transaction_date.isoformat() if transaction_date else None,
            "transaction_code": transaction_code,
            "acquired_disposed_code": acquired_disposed,
            "shares": shares,
            "price_per_share": price,
            "ownership_type": ownership_type,
        }
        for owner in owners:
            owner_cik = _cik10(owner.get("insider_cik"))
            owner_name = owner.get("insider_name")
            transaction_id = stable_hash({
                "accession_number": filing.accession_number,
                "transaction_index": tx_index,
                "owner_cik": owner_cik,
                "owner_name": owner_name,
                "transaction_date": transaction_date,
                "transaction_code": transaction_code,
                "acquired_disposed_code": acquired_disposed,
                "shares": shares,
                "price_per_share": price,
                "security_title": security_title,
            })
            rows.append(SecForm4Transaction(
                transaction_id=transaction_id,
                accession_number=filing.accession_number,
                filing_form=filing.form,
                filing_date=filing.filing_date,
                filing_accepted_at=filing.acceptance_datetime,
                issuer_cik=issuer_cik or _cik10(filing.cik),
                issuer_name=issuer_name,
                ticker=ticker,
                insider_cik=owner_cik,
                insider_name=owner_name,
                insider_state=owner.get("insider_state"),
                insider_roles=dict(owner.get("roles") or {}),
                transaction_date=transaction_date,
                transaction_code=_clean_string(transaction_code),
                acquired_disposed_code=_clean_string(acquired_disposed),
                security_title=_clean_string(security_title),
                shares=shares,
                price_per_share=price,
                ownership_type=_clean_string(ownership_type),
                is_10b5_1=document_10b5_1 or _transaction_mentions_10b5(tx_node),
                raw={
                    "owner": owner,
                    "transaction": tx_raw,
                    "accession_number": filing.accession_number,
                },
            ))
    return rows, None


def _reporting_owners(root: ET.Element) -> List[Dict[str, Any]]:
    owners: List[Dict[str, Any]] = []
    for owner_node in _descendants(root, "reportingOwner"):
        owner_id = _first_child(owner_node, "reportingOwnerId")
        address = _first_child(owner_node, "reportingOwnerAddress")
        relationship = _first_child(owner_node, "reportingOwnerRelationship")
        officer_title = _first_text(relationship, "officerTitle") if relationship is not None else None
        roles = {
            "is_director": _xml_bool(_first_text(relationship, "isDirector") if relationship is not None else None),
            "is_officer": _xml_bool(_first_text(relationship, "isOfficer") if relationship is not None else None),
            "is_ten_percent_owner": _xml_bool(_first_text(relationship, "isTenPercentOwner") if relationship is not None else None),
            "is_other": _xml_bool(_first_text(relationship, "isOther") if relationship is not None else None),
            "officer_title": officer_title,
        }
        owners.append({
            "insider_cik": _first_text(owner_id, "rptOwnerCik") if owner_id is not None else None,
            "insider_name": _first_text(owner_id, "rptOwnerName") if owner_id is not None else None,
            "insider_state": _first_text(address, "rptOwnerState") if address is not None else None,
            "roles": roles,
        })
    return owners


def _transaction_mentions_10b5(node: ET.Element) -> Optional[bool]:
    text = " ".join(
        item.text or ""
        for item in node.iter()
        if item.text
    ).lower()
    if not text:
        return None
    return "10b5-1" in text or "10b5" in text


def _nested_text(node: Optional[ET.Element], *path: str) -> Optional[str]:
    current = node
    for name in path:
        current = _first_child(current, name)
        if current is None:
            return None
    return _clean_string(current.text)


def _first_text(node: Optional[ET.Element], name: str) -> Optional[str]:
    child = _first_child(node, name)
    return _clean_string(child.text) if child is not None else None


def _first_child(node: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    if node is None:
        return None
    wanted = name.lower()
    for child in node:
        if _local_name(child.tag).lower() == wanted:
            return child
    for child in node.iter():
        if child is not node and _local_name(child.tag).lower() == wanted:
            return child
    return None


def _descendants(node: ET.Element, name: str) -> List[ET.Element]:
    wanted = name.lower()
    return [
        item
        for item in node.iter()
        if _local_name(item.tag).lower() == wanted
    ]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _xml_bool(value: Any) -> Optional[bool]:
    text = _clean_string(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    return None


def _asof_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return utcnow()
    return aware_utc_or_none(value)


def _cik10(value: Any) -> Optional[str]:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if not digits:
        return None
    return digits[-10:].zfill(10)


def _parse_company_tickers(payload: Any) -> List[SecCompanyTicker]:
    rows: List[SecCompanyTicker] = []
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        fields = [str(item).lower() for item in payload.get("fields", [])]
        for item in payload["data"]:
            if not isinstance(item, list):
                continue
            raw = {
                fields[index]: item[index]
                for index in range(min(len(fields), len(item)))
            }
            row = _company_ticker_from_mapping(raw)
            if row is not None:
                rows.append(row)
        return rows

    if isinstance(payload, dict):
        for item in payload.values():
            if isinstance(item, dict):
                row = _company_ticker_from_mapping(item)
                if row is not None:
                    rows.append(row)
    return rows


def _company_ticker_from_mapping(raw: Dict[str, Any]) -> Optional[SecCompanyTicker]:
    cik_value = raw.get("cik") or raw.get("cik_str")
    ticker = str(raw.get("ticker") or "").strip().upper()
    company_name = str(raw.get("name") or raw.get("title") or "").strip()
    cik10 = _cik10(cik_value)
    if cik10 is None or not ticker:
        return None
    return SecCompanyTicker(
        cik=int(cik10),
        cik_str=cik10,
        ticker=ticker,
        company_name=company_name,
        exchange=_clean_string(raw.get("exchange")),
        raw=dict(raw),
    )


def _parse_recent_filings(cik: str, payload: Dict[str, Any]) -> List[SecEdgarFiling]:
    recent = payload.get("filings", {}).get("recent", {})
    if not isinstance(recent, dict):
        return []
    length = max(
        (len(value) for value in recent.values() if isinstance(value, list)),
        default=0,
    )
    rows: List[SecEdgarFiling] = []
    for index in range(length):
        raw = {
            key: value[index]
            for key, value in recent.items()
            if isinstance(value, list) and index < len(value)
        }
        filing = _filing_from_recent_row(cik, raw)
        if filing is not None:
            rows.append(filing)
    return rows


def _submission_overflow_file_names(payload: Dict[str, Any]) -> List[str]:
    files = payload.get("filings", {}).get("files", [])
    if not isinstance(files, list):
        return []
    names: List[str] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        name = _clean_string(item.get("name"))
        if name:
            names.append(name.lstrip("/"))
    return names


def _overflow_quality_flags(payload: Dict[str, Any]) -> Dict[str, Any]:
    names = _submission_overflow_file_names(payload)
    return {
        "overflow_pages_available": len(names),
        "overflow_pages_fetched": 0,
        "truncated": len(names) > MAX_SUBMISSIONS_OVERFLOW_PAGES,
    }


def _filing_from_recent_row(cik: str, raw: Dict[str, Any]) -> Optional[SecEdgarFiling]:
    accession = _clean_string(raw.get("accessionNumber"))
    form = _clean_string(raw.get("form"))
    if not accession or not form:
        return None
    return SecEdgarFiling(
        cik=cik,
        accession_number=accession,
        form=form.upper(),
        filing_date=_coerce_date(raw.get("filingDate")),
        report_date=_coerce_date(raw.get("reportDate")),
        acceptance_datetime=_parse_acceptance_datetime(raw.get("acceptanceDateTime")),
        primary_document=_clean_string(raw.get("primaryDocument")),
        primary_doc_description=_clean_string(raw.get("primaryDocDescription")),
        act=_clean_string(raw.get("act")),
        file_number=_clean_string(raw.get("fileNumber")),
        film_number=_clean_string(raw.get("filmNumber")),
        items=_clean_string(raw.get("items")),
        size=_optional_int(raw.get("size")),
        is_xbrl=_optional_bool(raw.get("isXBRL")),
        is_inline_xbrl=_optional_bool(raw.get("isInlineXBRL")),
        raw=dict(raw),
    )


def _filter_filings(
    filings: Iterable[SecEdgarFiling],
    *,
    forms: Optional[Sequence[str]],
    from_date: Optional[date],
    to_date: Optional[date],
    asof: datetime,
) -> Tuple[List[SecEdgarFiling], Dict[str, Any]]:
    allowed_forms = _normalized_forms(forms)
    included: List[SecEdgarFiling] = []
    pit_excluded = 0
    date_filtered = 0
    form_filtered = 0
    for filing in filings:
        if allowed_forms and not _form_allowed(filing.form, allowed_forms):
            form_filtered += 1
            continue
        if filing.acceptance_datetime is None or filing.acceptance_datetime > asof:
            pit_excluded += 1
            continue
        if from_date is not None or to_date is not None:
            if filing.filing_date is None:
                date_filtered += 1
                continue
            if from_date is not None and filing.filing_date < from_date:
                date_filtered += 1
                continue
            if to_date is not None and filing.filing_date > to_date:
                date_filtered += 1
                continue
        included.append(filing)
    return included, {
        "included_count": len(included),
        "form_filtered_count": form_filtered,
        "date_filtered_count": date_filtered,
        "pit_excluded_count": pit_excluded,
    }


def _dedupe_and_sort_filings(
    filings: Iterable[SecEdgarFiling],
) -> List[SecEdgarFiling]:
    by_accession: Dict[str, SecEdgarFiling] = {}
    for filing in sorted(
        sorted(filings, key=lambda item: item.accession_number),
        key=lambda item: item.acceptance_datetime
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    ):
        by_accession.setdefault(filing.accession_number, filing)
    return list(by_accession.values())


def _normalized_forms(forms: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not forms:
        return ()
    return tuple(
        str(form or "").strip().upper()
        for form in forms
        if str(form or "").strip()
    )


def _form_allowed(form: str, allowed_forms: Sequence[str]) -> bool:
    normalized = str(form or "").strip().upper()
    return any(
        normalized == allowed
        or normalized.startswith(f"{allowed}/")
        for allowed in allowed_forms
    )


def _filing_to_survivorship_event(
    filing: SecEdgarFiling,
    *,
    ticker: str,
    ticker_row: Optional[SecCompanyTicker],
) -> Dict[str, Any]:
    """Convert Form 25 notice metadata without fabricating effective date.

    Form 25 effectiveness is generally post-filing and must be modeled by the
    consumer; submissions metadata does not carry a reliable effective date.
    """

    knowledge_ts = filing.acceptance_datetime
    event_date = filing.filing_date or filing.report_date
    return {
        "id": filing.accession_number,
        "event_id": filing.accession_number,
        "type": "delisting_notice",
        "event_type": "delisting_notice",
        "classification": f"sec_form_{filing.form.lower()}",
        "source_backed": True,
        "source": "sec_edgar_submissions",
        "ticker": str(ticker or "").strip().upper(),
        "cik": filing.cik,
        "company_name": ticker_row.company_name if ticker_row else None,
        "exchange": ticker_row.exchange if ticker_row else None,
        "form": filing.form,
        "accession_number": filing.accession_number,
        "filing_date": filing.filing_date.isoformat() if filing.filing_date else None,
        "report_date": filing.report_date.isoformat() if filing.report_date else None,
        "event_date": event_date.isoformat() if event_date else None,
        "effective_date": None,
        "knowledge_timestamp": (
            knowledge_ts.isoformat() if knowledge_ts is not None else None
        ),
        "acceptance_datetime": (
            knowledge_ts.isoformat() if knowledge_ts is not None else None
        ),
        "primary_document": filing.primary_document,
        "primary_doc_description": filing.primary_doc_description,
        "raw": filing.raw,
    }


def _parse_acceptance_datetime(value: Any) -> Optional[datetime]:
    """EDGAR acceptanceDateTime is Eastern wall-clock; offsets are stripped
    before localizing to America/New_York.
    """

    text = _clean_string(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=None).replace(
        tzinfo=EDGAR_ACCEPTANCE_TIMEZONE
    ).astimezone(timezone.utc)


def _coerce_date(value: Any) -> Optional[date]:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    text = _clean_string(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _date_iso(value: Any) -> Optional[str]:
    parsed = _coerce_date(value)
    return parsed.isoformat() if parsed is not None else None


def _request_data_quality_flags(url: str) -> Dict[str, Any]:
    host = (urlparse(url).hostname or "").lower()
    if host == "sec.gov" or host.endswith(".sec.gov"):
        return {}
    return {"non_sec_host": host}


def _clean_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None
