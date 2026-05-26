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
    aware_utc_or_none,
    stable_hash,
    utcnow,
)
from alpha.data.universe_config import MCAP_MAX, MCAP_MIN

PROVIDER = "FMP"
HISTORICAL_PRICE_FULL_ENDPOINT = "/stable/historical-price-eod/full"
HISTORICAL_PRICE_DIVIDEND_ADJUSTED_ENDPOINT = (
    "/stable/historical-price-eod/dividend-adjusted"
)


def _bool_or_raw(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        if value == 0:
            return False
        if value == 1:
            return True
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "1", "yes", "y"}:
            return True
        if cleaned in {"false", "0", "no", "n"}:
            return False
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
    split_adjusted_close: Optional[float] = None
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
                        message="FMP adapter asof timestamp must be timezone-aware datetime",
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
        market_cap_min: int = MCAP_MIN,
        market_cap_max: int = MCAP_MAX,
        country: Optional[str] = None,
        is_etf: Optional[bool] = None,
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
        asof: Optional[datetime] = None,
        *,
        adjusted: bool = False,
        require_split_adjusted_close: bool = True,
        require_adjusted_close: bool = False,
    ) -> AdapterResponse[List[FmpBar]]:
        endpoint = (
            HISTORICAL_PRICE_DIVIDEND_ADJUSTED_ENDPOINT
            if adjusted else HISTORICAL_PRICE_FULL_ENDPOINT
        )
        params: Dict[str, Any] = {"symbol": ticker}
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()
        resp = self._request(endpoint, params=params, asof=asof)
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
            _parse_fmp_bar(b, split_adjusted_close_from_close=not adjusted)
            for b in historical
        ]
        if require_split_adjusted_close and any(
            bar.split_adjusted_close is None for bar in bars
        ):
            missing_count = sum(
                1 for bar in bars if bar.split_adjusted_close is None
            )
            return AdapterResponse(
                data=None,
                lineage=resp.lineage,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=200,
                    error_type="data_contract",
                    message=(
                        f"{ticker} historical daily response missing split-adjusted "
                        f"close on {missing_count}/{len(bars)} rows; refusing "
                        "dividend-adjusted/raw fallback"
                    ),
                    retryable=False,
                ),
            )
        if require_adjusted_close and any(bar.adj_close is None for bar in bars):
            missing_count = sum(1 for bar in bars if bar.adj_close is None)
            return AdapterResponse(
                data=None,
                lineage=resp.lineage,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=200,
                    error_type="data_contract",
                    message=(
                        f"{ticker} historical daily response missing adjClose on "
                        f"{missing_count}/{len(bars)} rows; refusing raw-close fallback"
                    ),
                    retryable=False,
                ),
            )
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


def _parse_fmp_bar(
    row: Dict[str, Any],
    *,
    split_adjusted_close_from_close: bool,
) -> FmpBar:
    """Parse either full OHLC rows or dividend-adjusted EOD rows.

    The stable adjusted endpoint returns adjOpen/adjHigh/adjLow/adjClose,
    while the full endpoint returns split-adjusted open/high/low/close
    without adjClose. M4 uses split_adjusted_close, not dividend-adjusted
    adjClose.
    """
    adj_close = row.get("adjClose")
    split_adjusted_close = (
        row.get("close") if split_adjusted_close_from_close else None
    )
    return FmpBar(
        date=row.get("date", ""),
        open=_first_present(row, "open", "adjOpen", default=0.0),
        high=_first_present(row, "high", "adjHigh", default=0.0),
        low=_first_present(row, "low", "adjLow", default=0.0),
        close=_first_present(row, "close", "adjClose", default=0.0),
        volume=row.get("volume", 0),
        split_adjusted_close=split_adjusted_close,
        adj_close=adj_close,
    )


def _first_present(row: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return default
