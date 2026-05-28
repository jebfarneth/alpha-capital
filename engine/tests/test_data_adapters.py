"""
Data adapter tests.

All tests use mocked HTTP responses — no real network calls.
Tests verify:
  - Raw payload hashes are stable (same input -> same hash)
  - Missing env vars fail clearly
  - Adapter responses carry provider/asof/lineage metadata
  - Error paths (auth, rate limit, timeout, parse) produce correct ProviderError
  - Typed response parsing
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
import requests

from alpha.data.benzinga import BenzingaAdapter
from alpha.data.config import (
    AlpacaConfig,
    BenzingaConfig,
    ConfigError,
    FmpConfig,
    PolygonConfig,
)
from alpha.data.contracts import stable_hash
from alpha.data.fmp import FmpAdapter
from alpha.data.alpaca import AlpacaAdapter
from alpha.data.polygon import PolygonAdapter
from alpha.jobs.security_type import (
    ADR,
    BUSINESS_DEVELOPMENT_COMPANY,
    COMMON_STOCK,
    ETF,
    MUTUAL_FUND,
    classify_security_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code: int = 200, json_data=None, text: str = "", headers=None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    if json_data is not None:
        resp.json.return_value = json_data
        resp.text = json.dumps(json_data, sort_keys=True, default=str)
    else:
        resp.text = text
        resp.json.side_effect = ValueError("No JSON")
    return resp


def _assert_aware_utc(value: datetime) -> None:
    assert value.tzinfo is not None
    assert value.utcoffset() == timezone.utc.utcoffset(value)


def _fmp_config():
    return FmpConfig(api_key="test-fmp-key")


def _alpaca_config():
    return AlpacaConfig(
        api_key="test-alpaca-key",
        secret_key="test-alpaca-secret",
        base_url="https://paper-api.alpaca.markets",
    )


def _polygon_config():
    return PolygonConfig(api_key="test-polygon-key")


def _benzinga_config():
    return BenzingaConfig(api_key="test-benzinga-key")


def _fixture_json(name: str):
    path = Path(__file__).parent / "fixtures" / name
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Contract: stable_hash
# ---------------------------------------------------------------------------

class TestStableHash:
    def test_same_input_same_hash(self):
        payload = {"price": 5.25, "volume": 1000, "ticker": "ACME"}
        h1 = stable_hash(payload)
        h2 = stable_hash(payload)
        assert h1 == h2
        assert len(h1) == 64  # sha256 hex

    def test_key_order_irrelevant(self):
        h1 = stable_hash({"a": 1, "b": 2})
        h2 = stable_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_payload_different_hash(self):
        h1 = stable_hash({"a": 1})
        h2 = stable_hash({"a": 2})
        assert h1 != h2


# ---------------------------------------------------------------------------
# Config: missing env vars fail clearly
# ---------------------------------------------------------------------------

class TestConfig:
    def test_fmp_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        with pytest.raises(ConfigError, match="FMP_API_KEY"):
            FmpConfig.from_env()

    def test_alpaca_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        with pytest.raises(ConfigError, match="ALPACA_API_KEY"):
            AlpacaConfig.from_env()

    def test_alpaca_missing_secret_raises(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(ConfigError, match="ALPACA_SECRET_KEY"):
            AlpacaConfig.from_env()

    def test_polygon_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        with pytest.raises(ConfigError, match="POLYGON_API_KEY"):
            PolygonConfig.from_env()

    def test_benzinga_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("BENZINGA_API_KEY", raising=False)
        monkeypatch.delenv("BENZINGA_TOKEN", raising=False)
        with pytest.raises(ConfigError, match="BENZINGA_API_KEY"):
            BenzingaConfig.from_env()

    def test_fmp_from_env_success(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "my-key")
        cfg = FmpConfig.from_env()
        assert cfg.api_key == "my-key"

    def test_alpaca_default_paper_url(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
        cfg = AlpacaConfig.from_env()
        assert "paper" in cfg.base_url

    def test_benzinga_from_env_success(self, monkeypatch):
        monkeypatch.setenv("BENZINGA_API_KEY", "bz-key")
        monkeypatch.setenv("BENZINGA_BASE_URL", "https://benzinga.test")
        cfg = BenzingaConfig.from_env()
        assert cfg.api_key == "bz-key"
        assert cfg.base_url == "https://benzinga.test"

    def test_benzinga_token_alias_from_env_success(self, monkeypatch):
        monkeypatch.delenv("BENZINGA_API_KEY", raising=False)
        monkeypatch.setenv("BENZINGA_TOKEN", "bz-token")
        cfg = BenzingaConfig.from_env()
        assert cfg.api_key == "bz-token"


# ---------------------------------------------------------------------------
# FMP adapter
# ---------------------------------------------------------------------------

class TestFmpAdapter:
    def _adapter(self, mock_session):
        return FmpAdapter(_fmp_config(), session=mock_session)

    def test_get_quote_ok(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {
                "symbol": "ACME",
                "price": 5.25,
                "volume": 100000,
                "marketCap": 75000000,
                "name": "Acme Corp",
                "exchange": "NASDAQ",
                "avgVolume": 80000,
                "dayHigh": 5.50,
                "dayLow": 5.00,
                "open": 5.10,
                "previousClose": 5.00,
                "timestamp": 1716206400,
            }
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_quote("ACME")

        assert resp.ok
        assert resp.data.symbol == "ACME"
        assert resp.data.price == 5.25
        assert resp.data.volume == 100000
        assert resp.lineage.provider == "FMP"
        assert resp.lineage.endpoint == "/stable/quote"
        assert resp.lineage.raw_payload_hash != ""
        assert resp.lineage.source_authority == "FMP_Ultimate"
        _assert_aware_utc(resp.lineage.request_timestamp)
        _assert_aware_utc(resp.lineage.asof_timestamp)
        session.get.assert_called_with(
            "https://financialmodelingprep.com/stable/quote",
            params={"symbol": "ACME"},
            timeout=30,
        )

    def test_get_quote_empty_result_is_no_data_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(200, [])
        adapter = self._adapter(session)
        resp = adapter.get_quote("ZZZZ_NOT_A_TICKER")

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "no_data"
        assert resp.error.retryable is False

    def test_get_quote_missing_price_stays_none(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200, [{"symbol": "ACME", "volume": 100000}]
        )
        adapter = self._adapter(session)
        resp = adapter.get_quote("ACME")

        assert resp.ok
        assert resp.data.price is None
        assert resp.data.volume == 100000

    def test_get_historical_price_ok(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {
                "date": "2026-05-19",
                "open": 5.0,
                "high": 5.5,
                "low": 4.9,
                "close": 5.25,
                "volume": 100000,
            },
            {
                "date": "2026-05-18",
                "open": 4.9,
                "high": 5.1,
                "low": 4.8,
                "close": 5.0,
                "volume": 90000,
            },
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_historical_price("ACME")

        assert resp.ok
        assert len(resp.data) == 2
        assert resp.data[0].close == 5.25
        assert resp.data[0].split_adjusted_close == 5.25
        assert resp.data[0].adj_close is None
        assert resp.lineage.provider == "FMP"
        assert resp.lineage.endpoint == "/stable/historical-price-eod/full"
        session.get.assert_called_with(
            "https://financialmodelingprep.com/stable/historical-price-eod/full",
            params={"symbol": "ACME"},
            timeout=30,
        )

    def test_get_historical_price_passes_date_window_and_asof(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {
                "date": "2026-05-19",
                "open": 5.0,
                "high": 5.5,
                "low": 4.9,
                "close": 5.25,
                "volume": 100000,
            },
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        asof = datetime(2026, 5, 19, 20, 0, tzinfo=timezone.utc)

        resp = adapter.get_historical_price(
            "ACME",
            from_date=date(2025, 3, 15),
            to_date=date(2026, 5, 19),
            asof=asof,
        )

        assert resp.ok
        assert resp.lineage.asof_timestamp == asof
        session.get.assert_called_with(
            "https://financialmodelingprep.com/stable/historical-price-eod/full",
            params={
                "symbol": "ACME",
                "from": "2025-03-15",
                "to": "2026-05-19",
            },
            timeout=30,
        )

    def test_get_historical_price_missing_split_adjusted_close_is_contract_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {
                "date": "2026-05-19",
                "open": 5.0,
                "high": 5.5,
                "low": 4.9,
                "adjClose": 5.25,
                "volume": 100000,
            },
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_historical_price("ACME")

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "data_contract"
        assert "missing split-adjusted close" in resp.error.message

    def test_get_historical_price_dividend_adjusted_is_optional_analytics_feed(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {
                "date": "2026-05-19",
                "adjOpen": 5.0,
                "adjHigh": 5.5,
                "adjLow": 4.9,
                "adjClose": 5.25,
                "volume": 100000,
            },
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_historical_price(
            "ACME",
            adjusted=True,
            require_split_adjusted_close=False,
            require_adjusted_close=True,
        )

        assert resp.ok
        assert resp.data[0].close == 5.25
        assert resp.data[0].split_adjusted_close is None
        assert resp.data[0].adj_close == 5.25
        assert resp.lineage.endpoint == "/stable/historical-price-eod/dividend-adjusted"

    def test_get_historical_price_null_response_returns_empty_list(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        resp_mock = MagicMock(spec=requests.Response)
        resp_mock.status_code = 200
        resp_mock.headers = {}
        resp_mock.text = "null"
        resp_mock.json.return_value = None
        session.get.return_value = resp_mock
        adapter = self._adapter(session)
        resp = adapter.get_historical_price("DELISTED")

        assert resp.ok
        assert resp.data == []

    def test_get_stock_screener_ok(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {"symbol": "ACME", "companyName": "Acme Corp", "marketCap": 75000000, "price": 5.25, "sector": "Technology"},
            {"symbol": "BETA", "companyName": "Beta Inc", "marketCap": 50000000, "price": 3.00, "sector": "Healthcare"},
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_stock_screener()

        assert resp.ok
        assert len(resp.data) == 2
        assert resp.data[0].symbol == "ACME"
        assert resp.data[0].market_cap == 75000000
        assert resp.lineage.endpoint == "/stable/company-screener"
        _assert_aware_utc(resp.lineage.request_timestamp)
        _assert_aware_utc(resp.lineage.asof_timestamp)
        params = session.get.call_args.kwargs["params"]
        assert "country" not in params
        assert "isEtf" not in params

    def test_request_converts_aware_asof_to_utc(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(200, [{"symbol": "ACME"}])
        adapter = self._adapter(session)

        resp = adapter._request(
            "/stable/company-screener",
            asof=datetime(2026, 5, 20, 0, 0, tzinfo=ZoneInfo("America/New_York")),
        )

        assert resp.ok
        assert resp.lineage.asof_timestamp == datetime(
            2026, 5, 20, 4, 0, tzinfo=timezone.utc
        )
        _assert_aware_utc(resp.lineage.request_timestamp)

    def test_request_rejects_naive_asof(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        adapter = self._adapter(session)

        resp = adapter._request(
            "/stable/company-screener",
            asof=datetime(2026, 5, 20, 14, 30),
        )

        assert not resp.ok
        assert resp.error.error_type == "validation"
        assert resp.error.retryable is False
        assert resp.error.message == "FMP adapter asof timestamp must be timezone-aware datetime"
        session.get.assert_not_called()

    def test_request_rejects_malformed_asof(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        adapter = self._adapter(session)

        resp = adapter._request(
            "/stable/company-screener",
            asof="",  # type: ignore[arg-type]
        )

        assert not resp.ok
        assert resp.error.error_type == "validation"
        assert resp.error.retryable is False
        assert resp.error.message == "FMP adapter asof timestamp must be timezone-aware datetime"
        session.get.assert_not_called()

    def test_get_stock_screener_missing_market_cap_stays_none(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {"symbol": "ACME", "companyName": "Acme Corp", "price": 5.25}
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_stock_screener()

        assert resp.ok
        assert resp.data[0].market_cap is None

    def test_get_stock_screener_normalizes_integer_booleans(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {
                "symbol": "ACME",
                "companyName": "Acme Corp",
                "marketCap": 75000000,
                "price": 5.25,
                "isEtf": 0,
                "isActivelyTrading": 1,
            }
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_stock_screener()

        assert resp.ok
        assert resp.data[0].is_etf is False
        assert resp.data[0].is_actively_trading is True

    def test_get_stock_screener_normalizes_string_booleans(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {
                "symbol": "ACME",
                "companyName": "Acme Corp",
                "marketCap": 75000000,
                "price": 5.25,
                "isEtf": "false",
                "isActivelyTrading": "true",
            }
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_stock_screener()

        assert resp.ok
        assert resp.data[0].is_etf is False
        assert resp.data[0].is_actively_trading is True

    def test_get_company_profile_ok(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {
                "symbol": "ACME",
                "companyName": "Acme Corp",
                "mktCap": 75000000,
                "sector": "Technology",
                "industry": "Software",
                "exchange": "NASDAQ",
                "isEtf": 0,
                "isActivelyTrading": 1,
            }
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_company_profile("ACME")

        assert resp.ok
        assert resp.data.symbol == "ACME"
        assert resp.data.market_cap == 75000000
        assert resp.data.exchange == "NASDAQ"
        assert resp.data.is_etf is False
        assert resp.data.is_actively_trading is True
        assert resp.data.raw["exchange"] == "NASDAQ"
        assert resp.lineage.endpoint == "/stable/profile"

    def test_recorded_fmp_profile_fixture_classifies_fund_from_adapter_raw(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200, _fixture_json("fmp_profile_bmez.json")
        )
        adapter = self._adapter(session)
        resp = adapter.get_company_profile("BMEZ")

        assert resp.ok
        assert resp.data.exchange == "NYSE"
        assert resp.data.raw["isFund"] is True
        security_type, reason = classify_security_type(resp.data, raw_json=resp.data.raw)
        assert security_type == MUTUAL_FUND
        assert reason == "raw_flag:isFund"

    def test_recorded_fmp_profile_fixture_classifies_adr_from_adapter_raw(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200, _fixture_json("fmp_profile_baba.json")
        )
        adapter = self._adapter(session)
        resp = adapter.get_company_profile("BABA")

        assert resp.ok
        assert resp.data.exchange == "NYSE"
        assert resp.data.raw["isAdr"] is True
        security_type, reason = classify_security_type(resp.data, raw_json=resp.data.raw)
        assert security_type == ADR
        assert reason == "raw_flag:isAdr"

    def test_recorded_fmp_profile_fixture_classifies_common_stock(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200, _fixture_json("fmp_profile_aapl.json")
        )
        adapter = self._adapter(session)
        resp = adapter.get_company_profile("AAPL")

        assert resp.ok
        assert resp.data.raw["isFund"] is False
        security_type, reason = classify_security_type(resp.data, raw_json=resp.data.raw)
        assert security_type == COMMON_STOCK
        assert reason == "profile_fields_present"

    def test_recorded_fmp_profile_fixture_keeps_cpbi_shell_label_common_stock(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200, _fixture_json("fmp_profile_cpbi.json")
        )
        adapter = self._adapter(session)
        resp = adapter.get_company_profile("CPBI")

        assert resp.ok
        assert resp.data.raw["industry"] == "Shell Companies"
        assert "banking products and services" in resp.data.raw["description"]
        security_type, reason = classify_security_type(resp.data, raw_json=resp.data.raw)
        assert security_type == COMMON_STOCK
        assert reason == "profile_fields_present"

    def test_recorded_fmp_profile_fixture_classifies_etf(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200, _fixture_json("fmp_profile_spy.json")
        )
        adapter = self._adapter(session)
        resp = adapter.get_company_profile("SPY")

        assert resp.ok
        assert resp.data.raw["isEtf"] is True
        security_type, reason = classify_security_type(resp.data, raw_json=resp.data.raw)
        assert security_type == ETF
        assert reason == "is_etf=True"

    def test_recorded_fmp_profile_fixture_classifies_bdc(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200, _fixture_json("fmp_profile_arcc.json")
        )
        adapter = self._adapter(session)
        resp = adapter.get_company_profile("ARCC")

        assert resp.ok
        assert resp.data.raw["isFund"] is False
        assert "business development company" in resp.data.raw["description"].lower()
        security_type, reason = classify_security_type(resp.data, raw_json=resp.data.raw)
        assert security_type == BUSINESS_DEVELOPMENT_COMPANY
        assert reason == "raw_description:BUSINESS_DEVELOPMENT_COMPANY"

    def test_get_company_profile_empty_result_is_no_data_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(200, [])
        adapter = self._adapter(session)
        resp = adapter.get_company_profile("ZZZZ_NOT_A_TICKER")

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "no_data"

    def test_get_sec_filings_ok(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {"symbol": "ACME", "fillingDate": "2026-05-01", "type": "10-K", "cik": "0001234567"}
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_sec_filings("ACME")

        assert resp.ok
        assert len(resp.data) == 1
        assert resp.data[0].filing_type == "10-K"
        assert resp.lineage.endpoint == "/stable/sec-filings-search/symbol"

    def test_auth_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(401, text="Unauthorized")
        adapter = self._adapter(session)
        resp = adapter.get_quote("ACME")

        assert not resp.ok
        assert resp.error.error_type == "auth"
        assert resp.error.retryable is False

    def test_rate_limit_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(429, text="Rate limit")
        adapter = self._adapter(session)
        resp = adapter.get_quote("ACME")

        assert not resp.ok
        assert resp.error.error_type == "rate_limit"
        assert resp.error.retryable is True

    def test_timeout_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = requests.exceptions.Timeout("timed out")
        adapter = self._adapter(session)
        resp = adapter.get_quote("ACME")

        assert not resp.ok
        assert resp.error.error_type == "timeout"
        assert resp.error.retryable is True

    def test_parse_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        resp_mock = _mock_response(200, text="not json")
        resp_mock.json.side_effect = ValueError("parse fail")
        session.get.return_value = resp_mock
        adapter = self._adapter(session)
        resp = adapter.get_quote("ACME")

        assert not resp.ok
        assert resp.error.error_type == "parse"

    def test_lineage_hash_stability(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [{"symbol": "ACME", "price": 5.25, "volume": 100000}]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp1 = adapter.get_quote("ACME")
        resp2 = adapter.get_quote("ACME")
        assert resp1.lineage.raw_payload_hash == resp2.lineage.raw_payload_hash


# ---------------------------------------------------------------------------
# Alpaca adapter
# ---------------------------------------------------------------------------

class TestAlpacaAdapter:
    def _adapter(self, mock_session):
        return AlpacaAdapter(_alpaca_config(), session=mock_session)

    def test_get_account_ok(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        json_data = {
            "id": "acct-123",
            "status": "ACTIVE",
            "cash": "1000.00",
            "buying_power": "1000.00",
            "portfolio_value": "1000.00",
            "currency": "USD",
            "pattern_day_trader": False,
            "trading_blocked": False,
            "account_blocked": False,
        }
        session.request.return_value = _mock_response(200, json_data, headers={"x-ratelimit-remaining": "180", "x-ratelimit-limit": "200"})
        adapter = self._adapter(session)
        resp = adapter.get_account()

        assert resp.ok
        assert resp.data.account_id == "acct-123"
        assert resp.data.cash == 1000.0
        assert resp.data.status == "ACTIVE"
        assert resp.lineage.provider == "Alpaca"
        assert resp.lineage.endpoint == "/v2/account"
        assert resp.rate_limit.calls_remaining == 180
        assert resp.rate_limit.calls_limit == 200

    def test_get_tradable_assets_ok(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        json_data = [
            {
                "id": "asset-1",
                "symbol": "ACME",
                "name": "Acme Corp",
                "exchange": "NASDAQ",
                "class": "us_equity",
                "tradable": True,
                "fractionable": True,
                "status": "active",
                "shortable": True,
                "easy_to_borrow": True,
            }
        ]
        session.request.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_tradable_assets()

        assert resp.ok
        assert len(resp.data) == 1
        assert resp.data[0].symbol == "ACME"
        assert resp.data[0].fractionable is True

    def test_submit_order_ok(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        json_data = {
            "id": "order-42",
            "client_order_id": "co-42",
            "symbol": "ACME",
            "side": "buy",
            "type": "market",
            "qty": "10",
            "status": "accepted",
            "time_in_force": "day",
            "created_at": "2026-05-20T14:30:00Z",
        }
        session.request.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.submit_order(symbol="ACME", qty=10)

        assert resp.ok
        assert resp.data.id == "order-42"
        assert resp.data.status == "accepted"
        assert resp.lineage.provider == "Alpaca"

    def test_get_order_ok(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        json_data = {
            "id": "order-42",
            "client_order_id": "co-42",
            "symbol": "ACME",
            "side": "buy",
            "type": "market",
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "5.25",
        }
        session.request.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_order("order-42")

        assert resp.ok
        assert resp.data.status == "filled"
        assert resp.data.filled_avg_price == "5.25"

    def test_cancel_order_ok(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.request.return_value = _mock_response(204, text="")
        adapter = self._adapter(session)
        resp = adapter.cancel_order("order-42")

        assert resp.ok
        assert resp.data is True

    def test_submit_order_rejects_missing_quantity(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        adapter = self._adapter(session)
        resp = adapter.submit_order(symbol="ACME")

        assert not resp.ok
        assert resp.error.error_type == "validation"
        assert "qty or notional" in resp.error.message
        session.request.assert_not_called()

    def test_submit_order_rejects_qty_and_notional_together(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        adapter = self._adapter(session)
        resp = adapter.submit_order(symbol="ACME", qty=10, notional=50)

        assert not resp.ok
        assert resp.error.error_type == "validation"
        assert "cannot both" in resp.error.message
        session.request.assert_not_called()

    def test_auth_error(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.request.return_value = _mock_response(403, text="Forbidden")
        adapter = self._adapter(session)
        resp = adapter.get_account()

        assert not resp.ok
        assert resp.error.error_type == "auth"

    def test_timeout_error(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.request.side_effect = requests.exceptions.Timeout("timed out")
        adapter = self._adapter(session)
        resp = adapter.get_account()

        assert not resp.ok
        assert resp.error.error_type == "timeout"

    def test_request_converts_aware_asof_to_utc(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.request.return_value = _mock_response(200, {"ok": True})
        adapter = self._adapter(session)

        resp = adapter._request(
            "GET",
            "/v2/account",
            asof=datetime(2026, 5, 20, 0, 0, tzinfo=ZoneInfo("America/New_York")),
        )

        assert resp.ok
        assert resp.lineage.asof_timestamp == datetime(
            2026, 5, 20, 4, 0, tzinfo=timezone.utc
        )
        _assert_aware_utc(resp.lineage.request_timestamp)

    def test_request_rejects_naive_asof(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        adapter = self._adapter(session)

        resp = adapter._request(
            "GET",
            "/v2/account",
            asof=datetime(2026, 5, 20, 14, 30),
        )

        assert not resp.ok
        assert resp.error.error_type == "validation"
        assert resp.error.retryable is False
        assert resp.error.message == "Alpaca adapter asof timestamp must be timezone-aware datetime"
        session.request.assert_not_called()

    def test_request_rejects_malformed_asof(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        adapter = self._adapter(session)

        resp = adapter._request(
            "GET",
            "/v2/account",
            asof="",  # type: ignore[arg-type]
        )

        assert not resp.ok
        assert resp.error.error_type == "validation"
        assert resp.error.retryable is False
        assert resp.error.message == "Alpaca adapter asof timestamp must be timezone-aware datetime"
        session.request.assert_not_called()


# ---------------------------------------------------------------------------
# Polygon adapter
# ---------------------------------------------------------------------------

class TestPolygonAdapter:
    def _adapter(self, mock_session):
        return PolygonAdapter(_polygon_config(), session=mock_session)

    def test_get_short_interest_ok(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {
                    "ticker": "ACME",
                    "settlement_date": "2026-05-15",
                    "short_interest": 50000,
                    "avg_daily_volume": 200000,
                    "days_to_cover": 0.25,
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_short_interest("ACME")

        assert resp.ok
        assert len(resp.data) == 1
        assert resp.data[0].short_interest == 50000
        assert resp.data[0].avg_daily_volume == 200000
        assert resp.data[0].days_to_cover == 0.25
        assert resp.lineage.provider == "Polygon"
        assert resp.lineage.endpoint == "/stocks/v1/short-interest"
        assert resp.lineage.source_authority == "Polygon"
        session.get.assert_called_with(
            "https://api.polygon.io/stocks/v1/short-interest",
            params={"ticker": "ACME"},
            timeout=30,
        )

    def test_get_ticker_details_ok(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": {
                "ticker": "ACME",
                "name": "Acme Corp",
                "market_cap": 75000000,
                "primary_exchange": "XNAS",
                "type": "CS",
                "sic_code": "3674",
            }
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_ticker_details("ACME")

        assert resp.ok
        assert resp.data.ticker == "ACME"
        assert resp.data.market_cap == 75000000

    def test_get_daily_bars_ok(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {"t": 1716163200000, "o": 5.0, "h": 5.5, "l": 4.9, "c": 5.25, "v": 100000, "vw": 5.15, "n": 450},
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_daily_bars("ACME", "2026-05-01", "2026-05-19")

        assert resp.ok
        assert len(resp.data) == 1
        assert resp.data[0].close == 5.25
        assert resp.data[0].vwap == 5.15

    def test_rate_limit_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(429, text="Too many requests")
        adapter = self._adapter(session)
        resp = adapter.get_short_interest("ACME")

        assert not resp.ok
        assert resp.error.error_type == "rate_limit"
        assert resp.error.retryable is True

    def test_timeout_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = requests.exceptions.Timeout("timed out")
        adapter = self._adapter(session)
        resp = adapter.get_ticker_details("ACME")

        assert not resp.ok
        assert resp.error.error_type == "timeout"

    def test_lineage_hash_stability(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {
                    "ticker": "ACME",
                    "settlement_date": "2026-05-15",
                    "short_interest": 50000,
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp1 = adapter.get_short_interest("ACME")
        resp2 = adapter.get_short_interest("ACME")
        assert resp1.lineage.raw_payload_hash == resp2.lineage.raw_payload_hash

    def test_request_converts_aware_asof_to_utc(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(200, {"results": []})
        adapter = self._adapter(session)

        resp = adapter._request(
            "/stocks/v1/short-interest",
            asof=datetime(2026, 5, 20, 0, 0, tzinfo=ZoneInfo("America/New_York")),
        )

        assert resp.ok
        assert resp.lineage.asof_timestamp == datetime(
            2026, 5, 20, 4, 0, tzinfo=timezone.utc
        )
        _assert_aware_utc(resp.lineage.request_timestamp)

    def test_request_rejects_naive_asof(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        adapter = self._adapter(session)

        resp = adapter._request(
            "/stocks/v1/short-interest",
            asof=datetime(2026, 5, 20, 14, 30),
        )

        assert not resp.ok
        assert resp.error.error_type == "validation"
        assert resp.error.retryable is False
        assert resp.error.message == "Polygon adapter asof timestamp must be timezone-aware datetime"
        session.get.assert_not_called()

    def test_request_rejects_malformed_asof(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        adapter = self._adapter(session)

        resp = adapter._request(
            "/stocks/v1/short-interest",
            asof="",  # type: ignore[arg-type]
        )

        assert not resp.ok
        assert resp.error.error_type == "validation"
        assert resp.error.retryable is False
        assert resp.error.message == "Polygon adapter asof timestamp must be timezone-aware datetime"
        session.get.assert_not_called()

    def test_get_tickers_bulk_paginates_and_parses_identity(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        page_1 = {
            "results": [
                {
                    "ticker": "AAPL",
                    "name": "Apple Inc.",
                    "market": "stocks",
                    "locale": "us",
                    "primary_exchange": "XNAS",
                    "type": "CS",
                    "active": True,
                    "cik": "320193",
                    "composite_figi": "BBG000B9XRY4",
                    "share_class_figi": "BBG001S5N8V8",
                    "list_date": "1980-12-12",
                }
            ],
            "next_url": "https://api.polygon.io/v3/reference/tickers?cursor=abc",
        }
        page_2 = {
            "results": [
                {
                    "ticker": "META",
                    "name": "Meta Platforms Inc.",
                    "market": "stocks",
                    "locale": "us",
                    "primary_exchange": "XNAS",
                    "type": "CS",
                    "active": True,
                    "cik": "1326801",
                    "composite_figi": "BBG000MM2P62",
                    "share_class_figi": "BBG001SQCQC5",
                }
            ]
        }
        session.get.side_effect = [
            _mock_response(200, page_1),
            _mock_response(200, page_2),
        ]
        adapter = self._adapter(session)

        resp = adapter.get_tickers(limit=1000)

        assert resp.ok
        assert len(resp.data) == 2
        assert resp.data[0].results[0].ticker == "AAPL"
        assert resp.data[0].results[0].cik == "0000320193"
        assert resp.data[1].results[0].ticker == "META"
        assert resp.data[1].results[0].cik == "0001326801"
        assert resp.data[0].next_url == "/v3/reference/tickers"
        assert session.get.call_count == 2
        assert session.get.call_args_list[0].kwargs["params"]["limit"] == 1000

    def test_get_tickers_no_data_is_successful_empty_page(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(200, {"results": []})
        adapter = self._adapter(session)

        resp = adapter.get_tickers()

        assert resp.ok
        assert len(resp.data) == 1
        assert resp.data[0].results == []

    def test_get_tickers_rate_limit_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(429, text="Too many requests")
        adapter = self._adapter(session)

        resp = adapter.get_tickers()

        assert not resp.ok
        assert resp.error.error_type == "rate_limit"
        assert resp.error.retryable is True

    def test_get_tickers_parse_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(200, text="not json")
        adapter = self._adapter(session)

        resp = adapter.get_tickers()

        assert not resp.ok
        assert resp.error.error_type == "parse"


    def test_get_ticker_details_with_cik_figi(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "market": "stocks",
                "locale": "us",
                "primary_exchange": "XNAS",
                "type": "CS",
                "active": True,
                "cik": "0000320193",
                "composite_figi": "BBG000B9XRY4",
                "share_class_figi": "BBG001S5N8V8",
                "list_date": "1980-12-12",
                "market_cap": 3200000000000,
                "sic_code": "3571",
                "sic_description": "Electronic Computers",
            }
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_ticker_details("AAPL")

        assert resp.ok
        assert resp.data.ticker == "AAPL"
        assert resp.data.cik == "0000320193"
        assert resp.data.composite_figi == "BBG000B9XRY4"
        assert resp.data.share_class_figi == "BBG001S5N8V8"
        assert resp.data.active is True
        assert resp.data.list_date == "1980-12-12"
        assert resp.data.market == "stocks"
        assert resp.data.sic_code == "3571"

    def test_get_ticker_details_no_data(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(200, {"results": {}})
        adapter = self._adapter(session)
        resp = adapter.get_ticker_details("FAKE")

        assert resp.ok
        assert resp.data is None

    def test_get_ticker_details_delisted(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": {
                "ticker": "FORA",
                "name": "Forian Inc",
                "active": False,
                "delisted_utc": "2025-11-15T00:00:00Z",
                "cik": "0001831097",
                "composite_figi": "BBG012JMR4P3",
            }
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_ticker_details("FORA")

        assert resp.ok
        assert resp.data.active is False
        assert resp.data.delisted_utc == "2025-11-15T00:00:00Z"
        assert resp.data.cik == "0001831097"

    def test_get_ticker_events_ticker_change(self):
        from alpha.data.polygon import PolygonTickerEvent
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": {
                "name": "Meta Platforms",
                "events": [
                    {
                        "type": "ticker_change",
                        "date": "2022-06-09",
                        "ticker_change": {"ticker": "FB"},
                    }
                ],
            }
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_ticker_events("META")

        assert resp.ok
        assert len(resp.data) == 1
        ev = resp.data[0]
        assert ev.event_type == "ticker_change"
        assert ev.date == "2022-06-09"
        assert ev.old_ticker == "FB"
        assert ev.identifier_queried == "META"

    def test_get_ticker_events_no_events(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {"results": {"name": "Acme", "events": []}}
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_ticker_events("ACME")

        assert resp.ok
        assert resp.data == []

    def test_get_ticker_events_403_not_accessible(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(403, text="Forbidden")
        adapter = self._adapter(session)
        resp = adapter.get_ticker_events("META")

        assert not resp.ok
        assert resp.error.error_type == "auth"
        assert resp.error.status_code == 403


# ---------------------------------------------------------------------------
# Benzinga adapter
# ---------------------------------------------------------------------------

class TestBenzingaAdapter:
    def _adapter(self, mock_session):
        return BenzingaAdapter(_benzinga_config(), session=mock_session)

    def test_get_news_ok(self):
        session = MagicMock(spec=requests.Session)
        json_data = {
            "news": [
                {
                    "id": 123,
                    "created": "2026-05-27T12:00:00Z",
                    "updated": 1779883200,
                    "published": "Wed, 27 May 2026 08:15:00 -0400",
                    "title": "Acme wins contract",
                    "body": "Full article body",
                    "teaser": "Short article teaser",
                    "url": "https://benzinga.example/news/123",
                    "author": "Newsdesk",
                    "source": "Benzinga",
                    "stocks": [{"name": "ACME", "exchange": "NASDAQ"}],
                    "channels": [{"name": "News"}, {"name": "WIIM"}],
                    "tags": [{"name": "contract"}],
                    "categories": ["press releases"],
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_news(
            symbols=["ACME", "BETA"],
            channels=["news", "wiim"],
            date_from="2026-05-01",
            date_to="2026-05-27",
            published_since="2026-05-01T00:00:00Z",
            updated_since=1779796800,
            page=1,
            limit=25,
        )

        assert resp.ok
        assert len(resp.data) == 1
        article = resp.data[0]
        assert article.id == "123"
        assert article.created == datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
        assert article.updated == datetime.fromtimestamp(1779883200, tz=timezone.utc)
        assert article.published == datetime(2026, 5, 27, 12, 15, tzinfo=timezone.utc)
        assert article.title == "Acme wins contract"
        assert article.body == "Full article body"
        assert article.teaser == "Short article teaser"
        assert article.url == "https://benzinga.example/news/123"
        assert article.author == "Newsdesk"
        assert article.source == "Benzinga"
        assert article.stocks == [{"name": "ACME", "exchange": "NASDAQ"}]
        assert article.tickers == ["ACME"]
        assert article.channels == ["News", "WIIM"]
        assert article.tags == ["contract"]
        assert article.categories == ["press releases"]
        assert article.raw["id"] == 123
        assert resp.lineage.provider == "Benzinga"
        assert resp.lineage.endpoint == "/api/v2/news"
        assert resp.lineage.source_authority == "Benzinga"
        session.get.assert_called_with(
            "https://api.benzinga.com/api/v2/news",
            params={
                "tickers": "ACME,BETA",
                "channels": "news,wiim",
                "dateFrom": "2026-05-01",
                "dateTo": "2026-05-27",
                "publishedSince": "2026-05-01T00:00:00Z",
                "updatedSince": 1779796800,
                "page": 1,
                "pageSize": 25,
                "token": "test-benzinga-key",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )

    def test_get_wiims_uses_wiim_channel(self):
        session = MagicMock(spec=requests.Session)
        json_data = {
            "news": [
                {
                    "id": "wiim-1",
                    "title": "Why ACME shares are trading higher",
                    "stocks": "ACME",
                    "channels": [{"name": "WIIM"}],
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_wiims("ACME", pagesize=5)

        assert resp.ok
        assert len(resp.data) == 1
        assert resp.data[0].id == "wiim-1"
        assert resp.data[0].tickers == ["ACME"]
        assert resp.data[0].channels == ["WIIM"]
        params = session.get.call_args.kwargs["params"]
        assert params["channels"] == "wiim"
        assert params["tickers"] == "ACME"
        assert params["pageSize"] == 5
        assert params["token"] == "test-benzinga-key"

    def test_get_news_empty_list_response_is_empty_success(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, [])
        adapter = self._adapter(session)

        resp = adapter.get_news("ACME")

        assert resp.ok
        assert resp.data == []

    @pytest.mark.parametrize("status_code", [401, 403])
    def test_get_news_auth_error(self, status_code):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(status_code, text="Unauthorized")
        adapter = self._adapter(session)

        resp = adapter.get_news("ACME")

        assert not resp.ok
        assert resp.error.error_type == "auth"
        assert resp.error.status_code == status_code
        assert resp.error.retryable is False

    def test_get_news_rate_limit_error(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(429, text="Rate limit")
        adapter = self._adapter(session)

        resp = adapter.get_news("ACME")

        assert not resp.ok
        assert resp.error.error_type == "rate_limit"
        assert resp.error.retryable is True

    def test_get_news_provider_error(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(500, text="Internal Server Error")
        adapter = self._adapter(session)

        resp = adapter.get_news("ACME")

        assert not resp.ok
        assert resp.error.error_type == "http"
        assert resp.error.status_code == 500
        assert resp.error.retryable is True

    def test_get_news_timeout_error(self):
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = requests.exceptions.Timeout("timed out")
        adapter = self._adapter(session)

        resp = adapter.get_news("ACME")

        assert not resp.ok
        assert resp.error.error_type == "timeout"
        assert resp.error.retryable is True

    def test_get_news_parse_error(self):
        session = MagicMock(spec=requests.Session)
        resp_mock = _mock_response(200, text="not json")
        resp_mock.json.side_effect = ValueError("parse fail")
        session.get.return_value = resp_mock
        adapter = self._adapter(session)

        resp = adapter.get_news("ACME")

        assert not resp.ok
        assert resp.error.error_type == "parse"

    def test_get_news_timestamp_parsing_variants(self):
        session = MagicMock(spec=requests.Session)
        json_data = {
            "news": [
                {
                    "id": "time-1",
                    "created": "2026-05-27 12:00:00",
                    "updated": "1779883200000",
                    "published": "Wed, 27 May 2026 08:15:00 -0400",
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_news()

        assert resp.ok
        article = resp.data[0]
        assert article.created == datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
        assert article.updated == datetime.fromtimestamp(1779883200, tz=timezone.utc)
        assert article.published == datetime(2026, 5, 27, 12, 15, tzinfo=timezone.utc)

    def test_get_news_lineage_does_not_expose_secret(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, {"news": []})
        adapter = self._adapter(session)

        resp = adapter.get_news("ACME")

        assert resp.ok
        assert "test-benzinga-key" not in repr(resp.lineage)
        assert resp.lineage.data_quality_flags is None

    def test_get_mergers_acquisitions_ok(self):
        session = MagicMock(spec=requests.Session)
        json_data = {
            "ma": [
                {
                    "id": "deal-1",
                    "target_ticker": "ACME",
                    "target_name": "Acme Corp",
                    "target_exchange": "NASDAQ",
                    "target_cusip": "004397105",
                    "target_isin": "US0043971052",
                    "acquirer_ticker": "BUY",
                    "acquirer_name": "Buyer Inc",
                    "acquirer_exchange": "NYSE",
                    "acquirer_cusip": "124857202",
                    "acquirer_isin": "US1248572026",
                    "deal_type": "Merger",
                    "deal_status": "Completed",
                    "deal_payment_type": "Cash",
                    "deal_size": "125000000",
                    "currency": "USD",
                    "date": "2026-05-01",
                    "date_completed": "2026-05-20",
                    "date_expected": "2026-05-31",
                    "deal_terms_extra": "$7.50 per share in cash",
                    "notes": "Definitive agreement announced.",
                    "importance": 5,
                    "updated": 1779307200,
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_mergers_acquisitions("ACME", pagesize=25)

        assert resp.ok
        assert len(resp.data) == 1
        deal = resp.data[0]
        assert deal.id == "deal-1"
        assert deal.target_ticker == "ACME"
        assert deal.target_cusip == "004397105"
        assert deal.target_isin == "US0043971052"
        assert deal.acquirer_ticker == "BUY"
        assert deal.acquirer_cusip == "124857202"
        assert deal.acquirer_isin == "US1248572026"
        assert deal.deal_payment_type == "Cash"
        assert deal.deal_terms_extra == "$7.50 per share in cash"
        assert deal.raw["target_exchange"] == "NASDAQ"
        assert resp.lineage.provider == "Benzinga"
        assert resp.lineage.endpoint == "/api/v2.1/calendar/ma"
        assert resp.lineage.source_authority == "Benzinga"
        session.get.assert_called_with(
            "https://api.benzinga.com/api/v2.1/calendar/ma",
            params={
                "pagesize": 25,
                "parameters[tickers]": "ACME",
                "token": "test-benzinga-key",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )

    def test_get_mergers_acquisitions_date_window_and_ticker_list(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, {"ma": []})
        adapter = self._adapter(session)
        resp = adapter.get_mergers_acquisitions(
            ["ACME", "BETA"],
            date_from="2026-01-01",
            date_to="2026-05-31",
            page=2,
            importance=5,
            updated=1779307200,
            date_sort="desc",
            cusip=["004397105", "08160H101"],
            isin="US0043971052",
        )

        assert resp.ok
        assert resp.data == []
        params = session.get.call_args.kwargs["params"]
        assert params["parameters[tickers]"] == "ACME,BETA"
        assert params["parameters[date_from]"] == "2026-01-01"
        assert params["parameters[date_to]"] == "2026-05-31"
        assert params["page"] == 2
        assert params["parameters[importance]"] == 5
        assert params["parameters[updated]"] == 1779307200
        assert params["parameters[date_sort]"] == "desc"
        assert params["parameters[cusip]"] == "004397105,08160H101"
        assert params["parameters[isin]"] == "US0043971052"
        assert params["token"] == "test-benzinga-key"

    def test_get_mergers_acquisitions_rejects_malformed_identifier_params(self):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        bad_cusip = adapter.get_mergers_acquisitions(cusip="00439710")
        bad_isin = adapter.get_mergers_acquisitions(isin="US004397105")

        assert not bad_cusip.ok
        assert bad_cusip.error.error_type == "validation"
        assert not bad_isin.ok
        assert bad_isin.error.error_type == "validation"
        session.get.assert_not_called()

    def test_get_mergers_acquisitions_canonicalizes_identifier_fields(self):
        session = MagicMock(spec=requests.Session)
        json_data = {
            "ma": [
                {
                    "id": "deal-1",
                    "target_ticker": "ACME",
                    "target_cusip": " 004397-105 ",
                    "target_isin": " us0043971052 ",
                    "acquirer_cusip": "00439710",
                    "acquirer_isin": "US004397105",
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_mergers_acquisitions("ACME")

        assert resp.ok
        deal = resp.data[0]
        assert deal.target_cusip == "004397105"
        assert deal.target_isin == "US0043971052"
        assert deal.acquirer_cusip is None
        assert deal.acquirer_isin is None

    def test_get_mergers_acquisitions_accepts_list_payload(self):
        session = MagicMock(spec=requests.Session)
        json_data = [
            {
                "id": 42,
                "target_ticker": "ACME",
                "deal_status": "Announced",
                "importance": "3",
                "updated": "1779307200",
            }
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_mergers_acquisitions()

        assert resp.ok
        assert len(resp.data) == 1
        assert resp.data[0].id == "42"
        assert resp.data[0].importance == 3
        assert resp.data[0].updated == 1779307200

    def test_get_mergers_acquisitions_empty_unexpected_payload_is_empty(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, {"unexpected": []})
        adapter = self._adapter(session)
        resp = adapter.get_mergers_acquisitions("ACME")

        assert resp.ok
        assert resp.data == []

    def test_auth_error(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(403, text="Forbidden")
        adapter = self._adapter(session)
        resp = adapter.get_mergers_acquisitions("ACME")

        assert not resp.ok
        assert resp.error.error_type == "auth"
        assert resp.error.retryable is False

    def test_rate_limit_error(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(429, text="Rate limit")
        adapter = self._adapter(session)
        resp = adapter.get_mergers_acquisitions("ACME")

        assert not resp.ok
        assert resp.error.error_type == "rate_limit"
        assert resp.error.retryable is True

    def test_timeout_error(self):
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = requests.exceptions.Timeout("timed out")
        adapter = self._adapter(session)
        resp = adapter.get_mergers_acquisitions("ACME")

        assert not resp.ok
        assert resp.error.error_type == "timeout"
        assert resp.error.retryable is True

    def test_parse_error(self):
        session = MagicMock(spec=requests.Session)
        resp_mock = _mock_response(200, text="not json")
        resp_mock.json.side_effect = ValueError("parse fail")
        session.get.return_value = resp_mock
        adapter = self._adapter(session)
        resp = adapter.get_mergers_acquisitions("ACME")

        assert not resp.ok
        assert resp.error.error_type == "parse"

    def test_lineage_hash_stability(self):
        session = MagicMock(spec=requests.Session)
        json_data = {"ma": [{"id": "deal-1", "target_ticker": "ACME"}]}
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp1 = adapter.get_mergers_acquisitions("ACME")
        resp2 = adapter.get_mergers_acquisitions("ACME")
        assert resp1.lineage.raw_payload_hash == resp2.lineage.raw_payload_hash

    def test_request_converts_aware_asof_to_utc(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, {"ma": []})
        adapter = self._adapter(session)

        resp = adapter._request(
            "/api/v2.1/calendar/ma",
            asof=datetime(2026, 5, 20, 0, 0, tzinfo=ZoneInfo("America/New_York")),
        )

        assert resp.ok
        assert resp.lineage.asof_timestamp == datetime(
            2026, 5, 20, 4, 0, tzinfo=timezone.utc
        )
        _assert_aware_utc(resp.lineage.request_timestamp)

    def test_request_rejects_naive_asof(self):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        resp = adapter._request(
            "/api/v2.1/calendar/ma",
            asof=datetime(2026, 5, 20, 14, 30),
        )

        assert not resp.ok
        assert resp.error.error_type == "validation"
        assert resp.error.retryable is False
        assert resp.error.message == "Benzinga adapter asof timestamp must be timezone-aware datetime"
        session.get.assert_not_called()


# ---------------------------------------------------------------------------
# Cross-adapter: lineage metadata completeness
# ---------------------------------------------------------------------------

class TestLineageMetadata:
    def test_every_success_response_has_full_lineage(self):
        """Every adapter success response must have provider, endpoint, timestamps, and hash."""
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.headers = {}
        json_data = [{"symbol": "ACME", "price": 5.0, "volume": 1000}]
        session.get.return_value = _mock_response(200, json_data)
        session.request.return_value = _mock_response(
            200,
            {"id": "acct-1", "status": "ACTIVE", "cash": "1000", "buying_power": "1000", "portfolio_value": "1000"},
        )

        fmp = FmpAdapter(_fmp_config(), session=session)
        alpaca = AlpacaAdapter(_alpaca_config(), session=session)
        polygon = PolygonAdapter(_polygon_config(), session=session)
        benzinga = BenzingaAdapter(_benzinga_config(), session=session)

        for resp in [
            fmp.get_quote("ACME"),
            alpaca.get_account(),
            polygon.get_short_interest("ACME"),
            benzinga.get_mergers_acquisitions("ACME"),
        ]:
            assert resp.lineage.provider != ""
            assert resp.lineage.endpoint != ""
            assert resp.lineage.request_timestamp is not None
            assert resp.lineage.asof_timestamp is not None
            _assert_aware_utc(resp.lineage.request_timestamp)
            _assert_aware_utc(resp.lineage.asof_timestamp)
            assert resp.lineage.raw_payload_hash != ""

    def test_every_error_response_has_lineage(self):
        """Even on error, lineage metadata is present."""
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.headers = {}
        session.get.return_value = _mock_response(500, text="Internal Server Error")
        session.request.return_value = _mock_response(500, text="Internal Server Error")

        fmp = FmpAdapter(_fmp_config(), session=session)
        alpaca = AlpacaAdapter(_alpaca_config(), session=session)
        polygon = PolygonAdapter(_polygon_config(), session=session)
        benzinga = BenzingaAdapter(_benzinga_config(), session=session)

        for resp in [
            fmp.get_quote("ACME"),
            alpaca.get_account(),
            polygon.get_short_interest("ACME"),
            benzinga.get_mergers_acquisitions("ACME"),
        ]:
            assert not resp.ok
            assert resp.lineage.provider != ""
            assert resp.lineage.endpoint != ""
            _assert_aware_utc(resp.lineage.request_timestamp)
            _assert_aware_utc(resp.lineage.asof_timestamp)
            assert resp.error is not None
