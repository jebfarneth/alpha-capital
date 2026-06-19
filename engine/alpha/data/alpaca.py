"""
Alpaca adapter.

Primary source for:
  - Account status
  - Tradable assets
  - Paper/live order submission and status
  - Order cancellation

Does not write to DB. Returns AdapterResponse with LineageMeta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import requests

from alpha.data.config import AlpacaConfig
from alpha.data.contracts import (
    AdapterResponse,
    LineageMeta,
    ProviderError,
    RateLimitInfo,
    aware_utc_or_none,
    stable_hash,
    utcnow,
)

PROVIDER = "Alpaca"
ALPACA_REQUEST_TIMEOUT = (10, 30)


# --- Response types ---

@dataclass
class AlpacaAccount:
    """Normalized account state returned by Alpaca's account endpoint."""

    account_id: str
    status: str
    cash: float
    buying_power: float
    portfolio_value: float
    currency: str = "USD"
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    account_blocked: bool = False


@dataclass
class AlpacaAsset:
    """Tradability and reference metadata for one Alpaca asset."""

    id: str
    symbol: str
    name: str
    exchange: str
    asset_class: str
    tradable: bool
    fractionable: bool
    status: str
    shortable: Optional[bool] = None
    easy_to_borrow: Optional[bool] = None


@dataclass
class AlpacaOrder:
    """Normalized Alpaca order payload used by execution-facing code."""

    id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    qty: Optional[str] = None
    notional: Optional[str] = None
    status: str = ""
    filled_qty: Optional[str] = None
    filled_avg_price: Optional[str] = None
    time_in_force: str = "day"
    limit_price: Optional[str] = None
    stop_price: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    submitted_at: Optional[str] = None
    filled_at: Optional[str] = None
    canceled_at: Optional[str] = None
    failed_at: Optional[str] = None


@dataclass
class AlpacaPosition:
    """Normalized Alpaca position payload used for startup reconciliation."""

    asset_id: str
    symbol: str
    exchange: Optional[str] = None
    asset_class: Optional[str] = None
    qty: Optional[str] = None
    avg_entry_price: Optional[str] = None
    side: Optional[str] = None
    market_value: Optional[str] = None
    cost_basis: Optional[str] = None
    unrealized_pl: Optional[str] = None
    current_price: Optional[str] = None
    lastday_price: Optional[str] = None
    change_today: Optional[str] = None


@dataclass
class AlpacaClock:
    """Normalized Alpaca market clock payload."""

    timestamp: Optional[str]
    is_open: bool
    next_open: Optional[str] = None
    next_close: Optional[str] = None


@dataclass(frozen=True)
class AlpacaQuote:
    """Normalized latest stock quote from Alpaca market data."""

    symbol: str
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    timestamp: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    tape: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlpacaStockSnapshot:
    """Normalized stock snapshot used by read-only live detectors."""

    symbol: str
    daily_open: Optional[float] = None
    daily_high: Optional[float] = None
    daily_low: Optional[float] = None
    daily_close: Optional[float] = None
    daily_volume: Optional[float] = None
    minute_open: Optional[float] = None
    minute_high: Optional[float] = None
    minute_low: Optional[float] = None
    minute_close: Optional[float] = None
    minute_volume: Optional[float] = None
    minute_timestamp: Optional[str] = None
    latest_trade_price: Optional[float] = None
    latest_trade_timestamp: Optional[str] = None
    latest_quote: Optional[AlpacaQuote] = None
    raw: Dict[str, Any] = field(default_factory=dict)


# --- Adapter ---

class AlpacaAdapter:
    """Thin Alpaca REST adapter that returns typed payloads with lineage."""

    def __init__(
        self, config: AlpacaConfig, session: Optional[requests.Session] = None
    ):
        self._config = config
        self._session = session or requests.Session()
        self._configure_session(self._session)

    def _configure_session(self, session: requests.Session) -> None:
        session.headers.update(
            {
                "APCA-API-KEY-ID": self._config.api_key,
                "APCA-API-SECRET-KEY": self._config.secret_key,
            }
        )

    def reset_session(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()
        self._session = requests.Session()
        self._configure_session(self._session)

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        asof: Optional[datetime] = None,
        base_url: Optional[str] = None,
    ) -> AdapterResponse[Any]:
        url = f"{base_url or self._config.base_url}{endpoint}"
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
                        message="Alpaca adapter asof timestamp must be timezone-aware datetime",
                        retryable=False,
                    ),
                )

        try:
            resp = self._session.request(
                method,
                url,
                params=params,
                json=json_body,
                timeout=ALPACA_REQUEST_TIMEOUT,
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
            source_authority="Alpaca",
        )

        rate_limit = RateLimitInfo(
            calls_remaining=_int_header(resp, "x-ratelimit-remaining"),
            calls_limit=_int_header(resp, "x-ratelimit-limit"),
        )

        if resp.status_code == 429:
            return AdapterResponse(
                data=None,
                lineage=lineage,
                rate_limit=rate_limit,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=429,
                    error_type="rate_limit",
                    message="Alpaca rate limit exceeded",
                    retryable=True,
                ),
            )

        if resp.status_code in (401, 403):
            return AdapterResponse(
                data=None,
                lineage=lineage,
                rate_limit=rate_limit,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=resp.status_code,
                    error_type="auth",
                    message=f"Alpaca auth error: {resp.status_code}",
                    retryable=False,
                ),
            )

        if resp.status_code >= 400:
            msg = resp.text[:200]
            return AdapterResponse(
                data=None,
                lineage=lineage,
                rate_limit=rate_limit,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=resp.status_code,
                    error_type="http",
                    message=f"Alpaca HTTP {resp.status_code}: {msg}",
                    retryable=resp.status_code >= 500,
                ),
            )

        if resp.status_code == 204 or not resp.text:
            return AdapterResponse(data=None, lineage=lineage, rate_limit=rate_limit)

        try:
            data = resp.json()
        except ValueError as exc:
            return AdapterResponse(
                data=None,
                lineage=lineage,
                rate_limit=rate_limit,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=resp.status_code,
                    error_type="parse",
                    message=f"JSON parse error: {exc}",
                    retryable=False,
                ),
            )

        return AdapterResponse(data=data, lineage=lineage, rate_limit=rate_limit)

    # --- Account ---

    def get_account(self) -> AdapterResponse[Optional[AlpacaAccount]]:
        """Fetch the configured account's cash, buying power, and status."""

        resp = self._request("GET", "/v2/account")
        if not resp.ok:
            return resp  # type: ignore[return-value]
        r = resp.data
        acct = AlpacaAccount(
            account_id=r.get("id", ""),
            status=r.get("status", ""),
            cash=float(r.get("cash", 0)),
            buying_power=float(r.get("buying_power", 0)),
            portfolio_value=float(r.get("portfolio_value", 0)),
            currency=r.get("currency", "USD"),
            pattern_day_trader=r.get("pattern_day_trader", False),
            trading_blocked=r.get("trading_blocked", False),
            account_blocked=r.get("account_blocked", False),
        )
        return AdapterResponse(data=acct, lineage=resp.lineage, rate_limit=resp.rate_limit)

    # --- Assets ---

    def get_tradable_assets(
        self, status: str = "active", asset_class: str = "us_equity"
    ) -> AdapterResponse[List[AlpacaAsset]]:
        """Return Alpaca assets that match the supplied status and asset class."""

        resp = self._request(
            "GET",
            "/v2/assets",
            params={"status": status, "asset_class": asset_class},
        )
        if not resp.ok:
            return resp  # type: ignore[return-value]

        assets = [
            AlpacaAsset(
                id=a.get("id", ""),
                symbol=a.get("symbol", ""),
                name=a.get("name", ""),
                exchange=a.get("exchange", ""),
                asset_class=a.get("class", ""),
                tradable=a.get("tradable", False),
                fractionable=a.get("fractionable", False),
                status=a.get("status", ""),
                shortable=a.get("shortable"),
                easy_to_borrow=a.get("easy_to_borrow"),
            )
            for a in resp.data
        ]
        return AdapterResponse(
            data=assets, lineage=resp.lineage, rate_limit=resp.rate_limit
        )

    # --- Orders ---

    def submit_order(
        self,
        symbol: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        side: str = "buy",
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> AdapterResponse[Optional[AlpacaOrder]]:
        """Submit an order after local shape validation."""

        body: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if qty is not None:
            body["qty"] = str(qty)
        if notional is not None:
            body["notional"] = str(notional)
        if limit_price is not None:
            body["limit_price"] = str(limit_price)
        if stop_price is not None:
            body["stop_price"] = str(stop_price)
        if client_order_id:
            body["client_order_id"] = client_order_id

        validation_error = _validate_order_request(
            qty=qty,
            notional=notional,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
        )
        if validation_error:
            return _validation_error_response("/v2/orders", body, validation_error)

        resp = self._request("POST", "/v2/orders", json_body=body)
        if not resp.ok:
            return resp  # type: ignore[return-value]

        return AdapterResponse(
            data=_parse_order(resp.data),
            lineage=resp.lineage,
            rate_limit=resp.rate_limit,
        )

    def get_order(self, order_id: str) -> AdapterResponse[Optional[AlpacaOrder]]:
        """Fetch the latest broker state for an order id."""

        resp = self._request("GET", f"/v2/orders/{order_id}")
        if not resp.ok:
            return resp  # type: ignore[return-value]
        return AdapterResponse(
            data=_parse_order(resp.data),
            lineage=resp.lineage,
            rate_limit=resp.rate_limit,
        )

    def list_orders(self, status: str = "open") -> AdapterResponse[List[AlpacaOrder]]:
        """List broker orders by status for startup reconciliation."""

        resp = self._request("GET", "/v2/orders", params={"status": status})
        if not resp.ok:
            return resp  # type: ignore[return-value]
        orders = [
            _parse_order(row)
            for row in (resp.data or [])
            if isinstance(row, dict)
        ]
        return AdapterResponse(
            data=orders,
            lineage=resp.lineage,
            rate_limit=resp.rate_limit,
        )

    def cancel_order(self, order_id: str) -> AdapterResponse[bool]:
        """Cancel an open Alpaca order and report whether the request succeeded."""

        resp = self._request("DELETE", f"/v2/orders/{order_id}")
        if resp.error:
            return AdapterResponse(
                data=False, lineage=resp.lineage, rate_limit=resp.rate_limit, error=resp.error
            )
        return AdapterResponse(
            data=True, lineage=resp.lineage, rate_limit=resp.rate_limit
        )

    # --- Positions / clock ---

    def get_positions(self) -> AdapterResponse[List[AlpacaPosition]]:
        """Fetch all open Alpaca positions."""

        resp = self._request("GET", "/v2/positions")
        if not resp.ok:
            return resp  # type: ignore[return-value]
        positions = [
            _parse_position(row)
            for row in (resp.data or [])
            if isinstance(row, dict)
        ]
        return AdapterResponse(
            data=positions,
            lineage=resp.lineage,
            rate_limit=resp.rate_limit,
        )

    def get_position(self, symbol: str) -> AdapterResponse[Optional[AlpacaPosition]]:
        """Fetch one open Alpaca position by symbol."""

        normalized = symbol.upper().strip()
        resp = self._request("GET", f"/v2/positions/{normalized}")
        if not resp.ok:
            return resp  # type: ignore[return-value]
        return AdapterResponse(
            data=_parse_position(resp.data),
            lineage=resp.lineage,
            rate_limit=resp.rate_limit,
        )

    def close_position(self, symbol: str) -> AdapterResponse[Optional[AlpacaOrder]]:
        """Submit an Alpaca market close request for one open position."""

        normalized = symbol.upper().strip()
        resp = self._request("DELETE", f"/v2/positions/{normalized}")
        if not resp.ok:
            return resp  # type: ignore[return-value]
        data = resp.data if isinstance(resp.data, dict) else None
        return AdapterResponse(
            data=_parse_order(data or {}),
            lineage=resp.lineage,
            rate_limit=resp.rate_limit,
        )

    def get_clock(self) -> AdapterResponse[Optional[AlpacaClock]]:
        """Fetch Alpaca's market clock."""

        resp = self._request("GET", "/v2/clock")
        if not resp.ok:
            return resp  # type: ignore[return-value]
        row = resp.data or {}
        clock = AlpacaClock(
            timestamp=row.get("timestamp"),
            is_open=bool(row.get("is_open", False)),
            next_open=row.get("next_open"),
            next_close=row.get("next_close"),
        )
        return AdapterResponse(data=clock, lineage=resp.lineage, rate_limit=resp.rate_limit)

    # --- Market data (read-only) ---

    def get_latest_quote(
        self,
        symbol: str,
        *,
        feed: str = "iex",
    ) -> AdapterResponse[Optional[AlpacaQuote]]:
        """Fetch one latest stock quote from Alpaca market data."""

        normalized = symbol.upper().strip()
        resp = self._request(
            "GET",
            f"/v2/stocks/{normalized}/quotes/latest",
            params={"feed": feed},
            base_url=self._config.market_data_base_url,
        )
        if not resp.ok:
            return resp  # type: ignore[return-value]
        quote = _parse_alpaca_quote(normalized, (resp.data or {}).get("quote"))
        return AdapterResponse(
            data=quote,
            lineage=resp.lineage,
            rate_limit=resp.rate_limit,
        )

    def get_latest_quotes(
        self,
        symbols: Sequence[str],
        *,
        feed: str = "iex",
    ) -> AdapterResponse[Dict[str, AlpacaQuote]]:
        """Fetch latest stock quotes for a bounded symbol batch."""

        normalized = [str(symbol).upper().strip() for symbol in symbols if symbol]
        if not normalized:
            return _validation_error_response(
                "/v2/stocks/quotes/latest",
                {},
                "symbols are required",
            )  # type: ignore[return-value]
        resp = self._request(
            "GET",
            "/v2/stocks/quotes/latest",
            params={"symbols": ",".join(normalized), "feed": feed},
            base_url=self._config.market_data_base_url,
        )
        if not resp.ok:
            return resp  # type: ignore[return-value]
        raw_quotes = (resp.data or {}).get("quotes") or {}
        quotes = {
            symbol.upper(): quote
            for symbol, payload in raw_quotes.items()
            if (quote := _parse_alpaca_quote(symbol, payload)) is not None
        }
        return AdapterResponse(
            data=quotes,
            lineage=resp.lineage,
            rate_limit=resp.rate_limit,
        )

    def get_historical_quotes(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        feed: str = "sip",
        limit: int = 10000,
        max_pages: int = 3,
    ) -> AdapterResponse[List[AlpacaQuote]]:
        """Fetch a compact historical stock quote window for one symbol."""

        normalized = str(symbol).upper().strip()
        start_ts = aware_utc_or_none(start)
        end_ts = aware_utc_or_none(end)
        if not normalized:
            return _validation_error_response(
                "/v2/stocks/{symbol}/quotes",
                {},
                "symbol is required",
            )  # type: ignore[return-value]
        if start_ts is None or end_ts is None:
            return _validation_error_response(
                f"/v2/stocks/{normalized}/quotes",
                {"start": str(start), "end": str(end)},
                "historical quote start/end must be timezone-aware datetimes",
            )  # type: ignore[return-value]
        if end_ts <= start_ts:
            return _validation_error_response(
                f"/v2/stocks/{normalized}/quotes",
                {"start": start_ts.isoformat(), "end": end_ts.isoformat()},
                "historical quote end must be after start",
            )  # type: ignore[return-value]

        endpoint = f"/v2/stocks/{normalized}/quotes"
        quotes: List[AlpacaQuote] = []
        next_page_token: str | None = None
        lineage: LineageMeta | None = None
        rate_limit = None
        for _ in range(max(1, int(max_pages))):
            params: Dict[str, Any] = {
                "start": start_ts.isoformat().replace("+00:00", "Z"),
                "end": end_ts.isoformat().replace("+00:00", "Z"),
                "feed": feed,
                "limit": int(limit),
            }
            if next_page_token:
                params["page_token"] = next_page_token
            resp = self._request(
                "GET",
                endpoint,
                params=params,
                base_url=self._config.market_data_base_url,
            )
            lineage = resp.lineage
            rate_limit = resp.rate_limit
            if not resp.ok:
                return resp  # type: ignore[return-value]
            payload = resp.data or {}
            rows = payload.get("quotes") or []
            if isinstance(rows, dict):
                rows = rows.get(normalized) or rows.get(normalized.upper()) or []
            if isinstance(rows, list):
                for row in rows:
                    quote = _parse_alpaca_quote(normalized, row)
                    if quote is not None:
                        quotes.append(quote)
            next_page_token = payload.get("next_page_token")
            if not next_page_token:
                break
        if next_page_token:
            return AdapterResponse(
                data=None,
                lineage=lineage or _local_lineage(endpoint, {
                    "symbol": normalized,
                    "start": start_ts.isoformat(),
                    "end": end_ts.isoformat(),
                    "feed": feed,
                    "limit": int(limit),
                    "max_pages": int(max_pages),
                }),
                rate_limit=rate_limit,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=endpoint,
                    status_code=None,
                    error_type="historical_quote_window_truncated",
                    message=(
                        "historical quote window still had next_page_token "
                        "after max_pages was exhausted"
                    ),
                    retryable=True,
                ),
            )
        return AdapterResponse(
            data=quotes,
            lineage=lineage or _local_lineage(endpoint, {
                "symbol": normalized,
                "start": start_ts.isoformat(),
                "end": end_ts.isoformat(),
                "feed": feed,
            }),
            rate_limit=rate_limit,
        )

    def get_stock_snapshots(
        self,
        symbols: Sequence[str],
        *,
        feed: str = "iex",
    ) -> AdapterResponse[Dict[str, AlpacaStockSnapshot]]:
        """Fetch read-only stock snapshots for detector inputs."""

        normalized = [str(symbol).upper().strip() for symbol in symbols if symbol]
        if not normalized:
            return _validation_error_response(
                "/v2/stocks/snapshots",
                {},
                "symbols are required",
            )  # type: ignore[return-value]
        resp = self._request(
            "GET",
            "/v2/stocks/snapshots",
            params={"symbols": ",".join(normalized), "feed": feed},
            base_url=self._config.market_data_base_url,
        )
        if not resp.ok:
            return resp  # type: ignore[return-value]
        snapshots = {
            symbol.upper(): snapshot
            for symbol, payload in (resp.data or {}).items()
            if (snapshot := _parse_stock_snapshot(symbol, payload)) is not None
        }
        return AdapterResponse(
            data=snapshots,
            lineage=resp.lineage,
            rate_limit=resp.rate_limit,
        )


# --- helpers ---

def _parse_alpaca_quote(symbol: str, payload: Any) -> Optional[AlpacaQuote]:
    if not isinstance(payload, dict):
        return None
    conditions = payload.get("c") or payload.get("conditions") or []
    if isinstance(conditions, str):
        conditions = [conditions]
    if not isinstance(conditions, list):
        conditions = []
    return AlpacaQuote(
        symbol=str(symbol).upper(),
        bid_price=_optional_float(_first_present(payload, "bp", "bid_price")),
        ask_price=_optional_float(_first_present(payload, "ap", "ask_price")),
        bid_size=_optional_float(_first_present(payload, "bs", "bid_size")),
        ask_size=_optional_float(_first_present(payload, "as", "ask_size")),
        timestamp=payload.get("t") or payload.get("timestamp"),
        conditions=[str(value) for value in conditions],
        tape=payload.get("z") or payload.get("tape"),
        raw=payload,
    )


def _parse_stock_snapshot(symbol: str, payload: Any) -> Optional[AlpacaStockSnapshot]:
    if not isinstance(payload, dict):
        return None
    daily = payload.get("dailyBar") or payload.get("daily_bar") or {}
    minute = payload.get("minuteBar") or payload.get("minute_bar") or {}
    trade = payload.get("latestTrade") or payload.get("latest_trade") or {}
    quote = _parse_alpaca_quote(
        symbol,
        payload.get("latestQuote") or payload.get("latest_quote"),
    )
    return AlpacaStockSnapshot(
        symbol=str(symbol).upper(),
        daily_open=_optional_float(daily.get("o")),
        daily_high=_optional_float(daily.get("h")),
        daily_low=_optional_float(daily.get("l")),
        daily_close=_optional_float(daily.get("c")),
        daily_volume=_optional_float(daily.get("v")),
        minute_open=_optional_float(minute.get("o")),
        minute_high=_optional_float(minute.get("h")),
        minute_low=_optional_float(minute.get("l")),
        minute_close=_optional_float(minute.get("c")),
        minute_volume=_optional_float(minute.get("v")),
        minute_timestamp=minute.get("t"),
        latest_trade_price=_optional_float(trade.get("p") or trade.get("price")),
        latest_trade_timestamp=trade.get("t") or trade.get("timestamp"),
        latest_quote=quote,
        raw=payload,
    )


def _parse_order(r: dict) -> AlpacaOrder:
    return AlpacaOrder(
        id=r.get("id", ""),
        client_order_id=r.get("client_order_id", ""),
        symbol=r.get("symbol", ""),
        side=r.get("side", ""),
        order_type=r.get("type", ""),
        qty=r.get("qty"),
        notional=r.get("notional"),
        status=r.get("status", ""),
        filled_qty=r.get("filled_qty"),
        filled_avg_price=r.get("filled_avg_price"),
        time_in_force=r.get("time_in_force", "day"),
        limit_price=r.get("limit_price"),
        stop_price=r.get("stop_price"),
        created_at=r.get("created_at"),
        updated_at=r.get("updated_at"),
        submitted_at=r.get("submitted_at"),
        filled_at=r.get("filled_at"),
        canceled_at=r.get("canceled_at"),
        failed_at=r.get("failed_at"),
    )


def _parse_position(r: dict) -> AlpacaPosition:
    return AlpacaPosition(
        asset_id=r.get("asset_id", ""),
        symbol=r.get("symbol", ""),
        exchange=r.get("exchange"),
        asset_class=r.get("asset_class") or r.get("class"),
        qty=r.get("qty"),
        avg_entry_price=r.get("avg_entry_price"),
        side=r.get("side"),
        market_value=r.get("market_value"),
        cost_basis=r.get("cost_basis"),
        unrealized_pl=r.get("unrealized_pl"),
        current_price=r.get("current_price"),
        lastday_price=r.get("lastday_price"),
        change_today=r.get("change_today"),
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(payload: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _validate_order_request(
    *,
    qty: Optional[float],
    notional: Optional[float],
    side: str,
    order_type: str,
    limit_price: Optional[float],
    stop_price: Optional[float],
) -> Optional[str]:
    if qty is None and notional is None:
        return "Exactly one of qty or notional is required."
    if qty is not None and notional is not None:
        return "qty and notional cannot both be supplied."
    if qty is not None and qty <= 0:
        return "qty must be positive."
    if notional is not None and notional <= 0:
        return "notional must be positive."
    if side not in {"buy", "sell"}:
        return "side must be 'buy' or 'sell'."
    if order_type not in {"market", "limit", "stop", "stop_limit"}:
        return "order_type must be market, limit, stop, or stop_limit."
    if order_type in {"limit", "stop_limit"} and limit_price is None:
        return "limit_price is required for limit and stop_limit orders."
    if order_type in {"stop", "stop_limit"} and stop_price is None:
        return "stop_price is required for stop and stop_limit orders."
    return None


def _validation_error_response(
    endpoint: str, payload: dict, message: str
) -> AdapterResponse[None]:
    request_ts = utcnow()
    return AdapterResponse(
        data=None,
        lineage=LineageMeta(
            provider=PROVIDER,
            endpoint=endpoint,
            request_timestamp=request_ts,
            asof_timestamp=request_ts,
            raw_payload_hash=stable_hash(payload),
            source_authority="local_validation",
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


def _local_lineage(endpoint: str, payload: dict) -> LineageMeta:
    request_ts = utcnow()
    return LineageMeta(
        provider=PROVIDER,
        endpoint=endpoint,
        request_timestamp=request_ts,
        asof_timestamp=request_ts,
        raw_payload_hash=stable_hash(payload),
        source_authority="local",
    )


def _int_header(resp: requests.Response, name: str) -> Optional[int]:
    val = resp.headers.get(name)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None
