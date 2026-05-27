"""
Benzinga adapter.

Supplemental event source for:
  - M&A and acquisition evidence for survivorship/corporate-action review

Does not write to DB. Returns AdapterResponse with LineageMeta.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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


def _validation_error_response(
    endpoint: str,
    message: str,
    *,
    asof: Optional[datetime],
) -> AdapterResponse[List[BenzingaMergerAcquisition]]:
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
