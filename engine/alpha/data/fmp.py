"""
FMP (Financial Modeling Prep) adapter.

Primary source for:
  - Universe / security profiles (stock screener, company profile)
  - Quote / price history (historical price, real-time quote)
  - SEC filings / company facts

Does not write to DB. Returns AdapterResponse with LineageMeta.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import requests

from alpha.data.config import FmpConfig
from alpha.data.contracts import (
    AdapterResponse,
    LineageMeta,
    ProviderError,
    RateLimitInfo,
    stable_hash,
    utcnow,
)

PROVIDER = "FMP"


def _bool_or_raw(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        if value == 0:
            return False
        if value == 1:
            return True
    return value


# --- Response types ---

@dataclass
class FmpQuote:
    symbol: str
    price: Optional[float]
    volume: Optional[int]
    market_cap: Optional[float] = None
    name: Optional[str] = None
    exchange: Optional[str] = None
    avg_volume: Optional[int] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    open: Optional[float] = None
    previous_close: Optional[float] = None
    timestamp: Optional[int] = None


@dataclass
class FmpBar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None


@dataclass
class FmpCompanyProfile:
    symbol: str
    company_name: str
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: Optional[str] = None
    country: Optional[str] = None
    is_etf: Optional[bool] = None
    is_actively_trading: Optional[bool] = None
    ipo_date: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class FmpScreenerResult:
    symbol: str
    company_name: str
    market_cap: Optional[float]
    price: Optional[float] = None
    volume: Optional[int] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: Optional[str] = None
    country: Optional[str] = None
    is_etf: Optional[bool] = None
    is_actively_trading: Optional[bool] = None


@dataclass
class FmpSecFiling:
    symbol: str
    filing_date: str
    accepted_date: Optional[str] = None
    cik: Optional[str] = None
    filing_type: Optional[str] = None
    link: Optional[str] = None
    final_link: Optional[str] = None


# --- Adapter ---

class FmpAdapter:
    def __init__(self, config: FmpConfig, session: Optional[requests.Session] = None):
        self._config = config
        self._session = session or requests.Session()
        self._session.params = {"apikey": config.api_key}  # type: ignore[assignment]

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[Any]:
        url = f"{self._config.base_url}{endpoint}"
        request_ts = utcnow()
        asof_ts = asof or request_ts

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
            source_authority="FMP_Ultimate",
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
                    message="FMP rate limit exceeded",
                    retryable=True,
                ),
            )

        if resp.status_code == 401 or resp.status_code == 403:
            return AdapterResponse(
                data=None,
                lineage=lineage,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=resp.status_code,
                    error_type="auth",
                    message=f"FMP auth error: {resp.status_code}",
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
                    message=f"FMP HTTP {resp.status_code}",
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

    def _no_data_response(
        self,
        *,
        endpoint: str,
        lineage: LineageMeta,
        message: str,
    ) -> AdapterResponse[None]:
        return AdapterResponse(
            data=None,
            lineage=lineage,
            error=ProviderError(
                provider=PROVIDER,
                endpoint=endpoint,
                status_code=200,
                error_type="no_data",
                message=message,
                retryable=False,
            ),
        )

    # --- Universe / security profile ---

    def get_stock_screener(
        self,
        market_cap_min: int = 30_000_000,
        market_cap_max: int = 200_000_000,
        country: Optional[str] = "US",
        is_etf: Optional[bool] = False,
        limit: int = 5000,
    ) -> AdapterResponse[List[FmpScreenerResult]]:
        endpoint = "/stable/company-screener"
        params = {
            "marketCapMoreThan": market_cap_min,
            "marketCapLowerThan": market_cap_max,
            "limit": limit,
        }
        if country is not None:
            params["country"] = country
        if is_etf is not None:
            params["isEtf"] = str(is_etf).lower()
        resp = self._request(endpoint, params=params)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        rows = resp.data or []
        results = [
            FmpScreenerResult(
                symbol=r.get("symbol", ""),
                company_name=r.get("companyName", ""),
                market_cap=r.get("marketCap"),
                price=r.get("price"),
                volume=r.get("volume"),
                sector=r.get("sector"),
                industry=r.get("industry"),
                exchange=r.get("exchangeShortName"),
                country=r.get("country"),
                is_etf=_bool_or_raw(r.get("isEtf")),
                is_actively_trading=_bool_or_raw(r.get("isActivelyTrading")),
            )
            for r in rows
        ]
        return AdapterResponse(data=results, lineage=resp.lineage)

    def get_company_profile(
        self, ticker: str
    ) -> AdapterResponse[Optional[FmpCompanyProfile]]:
        endpoint = "/stable/profile"
        resp = self._request(endpoint, params={"symbol": ticker})
        if not resp.ok:
            return resp  # type: ignore[return-value]

        rows = resp.data
        if not rows:
            return self._no_data_response(
                endpoint=endpoint,
                lineage=resp.lineage,
                message=f"No company profile found for {ticker}",
            )
        r = rows[0]
        profile = FmpCompanyProfile(
            symbol=r.get("symbol", ticker),
            company_name=r.get("companyName", ""),
            market_cap=r.get("marketCap") or r.get("mktCap"),
            sector=r.get("sector"),
            industry=r.get("industry"),
            exchange=r.get("exchangeShortName") or r.get("exchange"),
            country=r.get("country"),
            is_etf=_bool_or_raw(r.get("isEtf")),
            is_actively_trading=_bool_or_raw(r.get("isActivelyTrading")),
            ipo_date=r.get("ipoDate"),
            raw=dict(r),
        )
        return AdapterResponse(data=profile, lineage=resp.lineage)

    # --- Quote / price history ---

    def get_quote(self, ticker: str) -> AdapterResponse[Optional[FmpQuote]]:
        endpoint = "/stable/quote"
        resp = self._request(endpoint, params={"symbol": ticker})
        if not resp.ok:
            return resp  # type: ignore[return-value]

        rows = resp.data
        if not rows:
            return self._no_data_response(
                endpoint=endpoint,
                lineage=resp.lineage,
                message=f"No quote found for {ticker}",
            )
        r = rows[0]
        quote = FmpQuote(
            symbol=r.get("symbol", ticker),
            price=r.get("price"),
            volume=r.get("volume"),
            market_cap=r.get("marketCap"),
            name=r.get("name"),
            exchange=r.get("exchange"),
            avg_volume=r.get("avgVolume"),
            day_high=r.get("dayHigh"),
            day_low=r.get("dayLow"),
            open=r.get("open"),
            previous_close=r.get("previousClose"),
            timestamp=r.get("timestamp"),
        )
        return AdapterResponse(data=quote, lineage=resp.lineage)

    def get_historical_price(
        self,
        ticker: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> AdapterResponse[List[FmpBar]]:
        endpoint = "/stable/historical-price-eod/full"
        params: Dict[str, Any] = {"symbol": ticker}
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()
        resp = self._request(endpoint, params=params)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        if isinstance(resp.data, dict):
            historical = resp.data.get("historical") or []
        elif resp.data is None:
            historical = []
        elif isinstance(resp.data, list):
            historical = resp.data
        else:
            historical = []
        bars = [
            FmpBar(
                date=b.get("date", ""),
                open=b.get("open", 0.0),
                high=b.get("high", 0.0),
                low=b.get("low", 0.0),
                close=b.get("close", 0.0),
                volume=b.get("volume", 0),
                adj_close=b.get("adjClose"),
            )
            for b in historical
        ]
        return AdapterResponse(data=bars, lineage=resp.lineage)

    # --- SEC filings ---

    def get_sec_filings(
        self, ticker: str, filing_type: Optional[str] = None, limit: int = 100
    ) -> AdapterResponse[List[FmpSecFiling]]:
        endpoint = "/stable/sec-filings-search/symbol"
        params: Dict[str, Any] = {"symbol": ticker, "limit": limit}
        if filing_type:
            params["formType"] = filing_type
        resp = self._request(endpoint, params=params)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        filings = [
            FmpSecFiling(
                symbol=f.get("symbol", ticker),
                filing_date=f.get("filingDate") or f.get("fillingDate", ""),
                accepted_date=f.get("acceptedDate"),
                cik=f.get("cik"),
                filing_type=f.get("formType") or f.get("type"),
                link=f.get("link"),
                final_link=f.get("finalLink"),
            )
            for f in (resp.data or [])
        ]
        return AdapterResponse(data=filings, lineage=resp.lineage)
