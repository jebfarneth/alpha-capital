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
from typing import Any, Dict, List, Optional

import requests

from alpha.data.config import AlpacaConfig
from alpha.data.contracts import (
    AdapterResponse,
    LineageMeta,
    ProviderError,
    RateLimitInfo,
    stable_hash,
    utcnow,
)

PROVIDER = "Alpaca"


# --- Response types ---

@dataclass
class AlpacaAccount:
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


# --- Adapter ---

class AlpacaAdapter:
    def __init__(
        self, config: AlpacaConfig, session: Optional[requests.Session] = None
    ):
        self._config = config
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "APCA-API-KEY-ID": config.api_key,
                "APCA-API-SECRET-KEY": config.secret_key,
            }
        )

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[Any]:
        url = f"{self._config.base_url}{endpoint}"
        request_ts = utcnow()
        asof_ts = asof or request_ts

        try:
            resp = self._session.request(
                method, url, params=params, json=json_body, timeout=15
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
        resp = self._request("GET", f"/v2/orders/{order_id}")
        if not resp.ok:
            return resp  # type: ignore[return-value]
        return AdapterResponse(
            data=_parse_order(resp.data),
            lineage=resp.lineage,
            rate_limit=resp.rate_limit,
        )

    def cancel_order(self, order_id: str) -> AdapterResponse[bool]:
        resp = self._request("DELETE", f"/v2/orders/{order_id}")
        if resp.error:
            return AdapterResponse(
                data=False, lineage=resp.lineage, rate_limit=resp.rate_limit, error=resp.error
            )
        return AdapterResponse(
            data=True, lineage=resp.lineage, rate_limit=resp.rate_limit
        )


# --- helpers ---

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


def _int_header(resp: requests.Response, name: str) -> Optional[int]:
    val = resp.headers.get(name)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None
