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
from decimal import Decimal
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
    SecEdgarConfig,
)
from alpha.data.contracts import stable_hash
from alpha.data.fmp import FMP_REQUEST_TIMEOUT, FmpAdapter
from alpha.data.alpaca import AlpacaAdapter
from alpha.data.edgar import SecEdgarAdapter
from alpha.data.polygon import POLYGON_REQUEST_TIMEOUT, PolygonAdapter, _normalized_cik
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


def _edgar_config():
    return SecEdgarConfig(user_agent="Alpha Capital test@example.com")


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

    def test_edgar_missing_user_agent_raises(self, monkeypatch):
        monkeypatch.delenv("SEC_USER_AGENT", raising=False)
        with pytest.raises(ConfigError, match="SEC_USER_AGENT"):
            SecEdgarConfig.from_env()

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

    def test_edgar_from_env_success(self, monkeypatch):
        monkeypatch.setenv("SEC_USER_AGENT", "Alpha Capital ops@example.com")
        cfg = SecEdgarConfig.from_env()
        assert cfg.user_agent == "Alpha Capital ops@example.com"
        assert cfg.data_base_url == "https://data.sec.gov"
        assert cfg.sec_base_url == "https://www.sec.gov"


# ---------------------------------------------------------------------------
# SEC EDGAR adapter
# ---------------------------------------------------------------------------

class TestSecEdgarAdapter:
    def _adapter(self, mock_session):
        return SecEdgarAdapter(_edgar_config(), session=mock_session)

    def _edgar_submissions_payload(
        self,
        *,
        accessions=None,
        forms=None,
        acceptances=None,
        files=None,
    ):
        accessions = accessions or []
        row_count = len(accessions)
        payload = {
            "filings": {
                "recent": {
                    "accessionNumber": accessions,
                    "form": forms or ["8-K"] * row_count,
                    "filingDate": ["2026-06-01"] * row_count,
                    "reportDate": ["2026-06-01"] * row_count,
                    "acceptanceDateTime": (
                        acceptances
                        or ["2026-06-01T12:00:00Z"] * row_count
                    ),
                }
            }
        }
        if files is not None:
            payload["filings"]["files"] = [
                {"name": name, "filingCount": 1}
                for name in files
            ]
        return payload

    def _edgar_overflow_file_names(self, count):
        return [
            f"CIK0000000001-submissions-{idx:03d}.json"
            for idx in range(1, count + 1)
        ]

    def test_get_company_tickers_parses_exchange_mapping_and_headers(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [
                    [320193, "Apple Inc.", "AAPL", "Nasdaq"],
                    [789019, "Microsoft Corp", "MSFT", "Nasdaq"],
                ],
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_company_tickers(
            asof=datetime(2026, 5, 29, 20, 0, tzinfo=timezone.utc)
        )

        assert resp.ok
        assert [row.ticker for row in resp.data] == ["AAPL", "MSFT"]
        assert resp.data[0].cik_str == "0000320193"
        assert resp.data[0].exchange == "Nasdaq"
        assert resp.lineage.provider == "SEC_EDGAR"
        assert resp.lineage.endpoint == "/files/company_tickers_exchange.json"
        assert resp.lineage.source_authority == "SEC_EDGAR"
        assert not (resp.lineage.data_quality_flags or {}).get("cache_hit")
        session.get.assert_called_once_with(
            "https://www.sec.gov/files/company_tickers_exchange.json",
            params={},
            headers={
                "User-Agent": "Alpha Capital test@example.com",
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json",
            },
            timeout=30,
        )

    def test_get_company_ticker_returns_none_for_unknown_ticker(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"]],
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_company_ticker("MISSING")

        assert resp.ok
        assert resp.data is None

    def test_get_company_submissions_pads_cik(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {"cik": "320193", "filings": {"recent": {}}},
        )
        adapter = self._adapter(session)

        resp = adapter.get_company_submissions("320193")

        assert resp.ok
        session.get.assert_called_once()
        assert (
            session.get.call_args.args[0]
            == "https://data.sec.gov/submissions/CIK0000320193.json"
        )

    def test_edgar_non_sec_base_url_is_visible_in_lineage_quality_flags(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {"cik": "320193", "filings": {"recent": {}}},
        )
        cfg = SecEdgarConfig(
            user_agent="Alpha Capital test@example.com",
            data_base_url="https://example.test",
        )
        adapter = SecEdgarAdapter(cfg, session=session)

        resp = adapter.get_company_submissions("320193")

        assert resp.ok
        assert resp.lineage.data_quality_flags["non_sec_host"] == "example.test"

    def test_get_company_submissions_rejects_invalid_cik(self):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        resp = adapter.get_company_submissions("not-a-cik")

        assert not resp.ok
        assert resp.error.error_type == "validation"
        assert "CIK" in resp.error.message
        session.get.assert_not_called()

    def test_get_filings_parses_recent_columnar_payload_and_filters_forms(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "filings": {
                    "recent": {
                        "accessionNumber": ["0001-25-000001", "0001-25-000002"],
                        "form": ["25-NSE", "8-K"],
                        "filingDate": ["2026-06-03", "2026-06-03"],
                        "reportDate": ["2026-06-02", "2026-06-02"],
                        "acceptanceDateTime": [
                            "2026-06-03T12:30:00.000Z",
                            "2026-06-03T12:31:00.000Z",
                        ],
                        "primaryDocument": ["xslF25X02/form25.xml", "a8k.htm"],
                        "primaryDocDescription": ["NOTIFICATION", "8-K"],
                        "act": ["34", "34"],
                        "fileNumber": ["001-00001", "001-00001"],
                        "filmNumber": ["26999999", "26999998"],
                        "items": ["", "2.02"],
                        "size": [1234, 4321],
                        "isXBRL": [0, 0],
                        "isInlineXBRL": [0, 1],
                    }
                }
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_filings(
            1,
            forms=["25", "25-NSE"],
            from_date=date(2026, 6, 1),
            to_date=date(2026, 6, 4),
            asof=datetime(2026, 6, 3, 17, 0, tzinfo=timezone.utc),
        )

        assert resp.ok
        assert len(resp.data) == 1
        filing = resp.data[0]
        assert filing.accession_number == "0001-25-000001"
        assert filing.form == "25-NSE"
        assert filing.filing_date == date(2026, 6, 3)
        assert filing.report_date == date(2026, 6, 2)
        assert filing.acceptance_datetime == datetime(
            2026, 6, 3, 16, 30, tzinfo=timezone.utc
        )
        assert filing.primary_document == "xslF25X02/form25.xml"
        assert filing.size == 1234
        assert filing.is_xbrl is False
        assert filing.is_inline_xbrl is False
        assert resp.lineage.data_quality_flags["included_count"] == 1
        assert resp.lineage.data_quality_flags["form_filtered_count"] == 1

    def test_get_filings_excludes_future_and_missing_acceptance_for_pit(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "filings": {
                    "recent": {
                        "accessionNumber": [
                            "known",
                            "future-knowledge",
                            "missing-knowledge",
                        ],
                        "form": ["25", "25", "25"],
                        "filingDate": ["2026-06-02", "2026-06-02", "2026-06-02"],
                        "reportDate": ["2026-06-02", "2026-06-02", "2026-06-02"],
                        "acceptanceDateTime": [
                            "2026-06-02T12:00:00Z",
                            "2026-06-03T12:00:00Z",
                            None,
                        ],
                        "primaryDocument": ["known.htm", "future.htm", "missing.htm"],
                    }
                }
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_filings(
            "0000000001",
            forms=["25"],
            from_date="2026-06-01",
            to_date="2026-06-04",
            asof=datetime(2026, 6, 2, 23, 59, tzinfo=timezone.utc),
        )

        assert resp.ok
        assert [filing.accession_number for filing in resp.data] == ["known"]
        assert resp.lineage.data_quality_flags["pit_excluded_count"] == 2

    def test_edgar_acceptance_datetime_interpreted_as_eastern_wall_clock(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "filings": {
                    "recent": {
                        "accessionNumber": ["summer", "winter"],
                        "form": ["25", "25"],
                        "filingDate": ["2022-10-28", "2022-12-01"],
                        "reportDate": ["2022-10-28", "2022-12-01"],
                        "acceptanceDateTime": [
                            "2022-10-28T20:22:36.000Z",
                            "2022-12-01T20:22:36.000Z",
                        ],
                    }
                }
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_filings(
            "0001418091",
            forms=["25"],
            asof=datetime(2022, 12, 2, 2, 0, tzinfo=timezone.utc),
        )

        assert resp.ok
        assert [filing.acceptance_datetime for filing in resp.data] == [
            datetime(2022, 12, 2, 1, 22, 36, tzinfo=timezone.utc),
            datetime(2022, 10, 29, 0, 22, 36, tzinfo=timezone.utc),
        ]

    def test_get_filings_excludes_after_close_eastern_acceptance_at_pit_cutoff(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "filings": {
                    "recent": {
                        "accessionNumber": ["after-close"],
                        "form": ["25"],
                        "filingDate": ["2026-06-04"],
                        "reportDate": ["2026-06-04"],
                        "acceptanceDateTime": ["2026-06-04T17:00:00Z"],
                    }
                }
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_filings(
            "0000000001",
            forms=["25"],
            asof=datetime(2026, 6, 4, 20, 0, tzinfo=timezone.utc),
        )

        assert resp.ok
        assert resp.data == []
        assert resp.lineage.data_quality_flags["pit_excluded_count"] == 1

    def test_get_filings_rejects_naive_asof(self):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        resp = adapter.get_filings(
            "0000000001",
            asof=datetime(2026, 6, 2, 12, 0),
        )

        assert not resp.ok
        assert resp.error.error_type == "validation"
        assert "timezone-aware" in resp.error.message
        session.get.assert_not_called()

    def test_edgar_invalid_asof_lineage_keeps_source_authority(self):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        resp = adapter.get_company_tickers(asof=datetime(2026, 6, 2, 12, 0))

        assert not resp.ok
        assert resp.error.error_type == "validation"
        assert resp.lineage.source_authority == "SEC_EDGAR"
        session.get.assert_not_called()

    def test_get_filings_handles_transport_errors_without_secrets(self):
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = requests.exceptions.ConnectionError("boom secret")
        adapter = self._adapter(session)

        resp = adapter.get_company_submissions("1")

        assert not resp.ok
        assert resp.error.error_type == "http"
        assert resp.error.message == "SEC EDGAR request failed: ConnectionError"
        assert "secret" not in resp.error.message

    def test_get_filings_maps_rate_limit_and_auth_errors(self):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        session.get.return_value = _mock_response(
            429,
            text="Too many",
            headers={"Retry-After": "17"},
        )
        rate_limited = adapter.get_company_submissions("1")
        assert not rate_limited.ok
        assert rate_limited.error.error_type == "rate_limit"
        assert rate_limited.error.retryable is True
        assert "Retry-After: 17" in rate_limited.error.message
        assert rate_limited.lineage.data_quality_flags["retry_after"] == "17"

        session.get.return_value = _mock_response(403, text="Forbidden")
        forbidden = adapter.get_company_submissions("1")
        assert not forbidden.ok
        assert forbidden.error.error_type == "auth"
        assert forbidden.error.retryable is False

    def test_get_company_ticker_reuses_company_ticker_cache(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [
                    [320193, "Apple Inc.", "AAPL", "Nasdaq"],
                    [789019, "Microsoft Corp", "MSFT", "Nasdaq"],
                    [1045810, "Nvidia Corp", "NVDA", "Nasdaq"],
                ],
            },
        )
        adapter = self._adapter(session)

        assert adapter.get_company_ticker("AAPL").data.cik_str == "0000320193"
        assert adapter.get_company_ticker("MSFT").data.cik_str == "0000789019"
        assert adapter.get_company_ticker("NVDA").data.cik_str == "0001045810"

        session.get.assert_called_once()

    def test_get_company_tickers_cache_hit_uses_current_asof_lineage(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"]],
            },
        )
        adapter = self._adapter(session)
        first_asof = datetime(2026, 6, 1, 20, 0, tzinfo=timezone.utc)
        second_asof = datetime(2026, 6, 2, 20, 0, tzinfo=timezone.utc)

        first = adapter.get_company_tickers(asof=first_asof)
        first.data.append("caller mutation")
        second = adapter.get_company_tickers(asof=second_asof)
        ticker = adapter.get_company_ticker("AAPL", asof=second_asof)
        invalid = adapter.get_company_tickers(asof=datetime(2026, 6, 3, 12, 0))

        assert first.ok
        assert second.ok
        assert ticker.ok
        assert second.lineage.asof_timestamp == second_asof
        assert second.lineage.request_timestamp == first.lineage.request_timestamp
        assert second.lineage.raw_payload_hash == first.lineage.raw_payload_hash
        assert second.lineage.data_quality_flags["cache_hit"] is True
        assert ticker.lineage.asof_timestamp == second_asof
        assert ticker.lineage.data_quality_flags["cache_hit"] is True
        assert len(second.data) == 1
        assert not invalid.ok
        assert invalid.error.error_type == "validation"
        session.get.assert_called_once()

    def test_get_filings_fetches_overflow_pages_for_specific_forms(self):
        recent_rows = 1000
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            _mock_response(
                200,
                {
                    "filings": {
                        "recent": {
                            "accessionNumber": [
                                f"recent-{idx}" for idx in range(recent_rows)
                            ],
                            "form": ["8-K"] * recent_rows,
                            "filingDate": ["2026-06-01"] * recent_rows,
                            "reportDate": ["2026-06-01"] * recent_rows,
                            "acceptanceDateTime": [
                                "2026-06-01T12:00:00Z"
                            ] * recent_rows,
                        },
                        "files": [
                            {
                                "name": "CIK0000000001-submissions-001.json",
                                "filingCount": 1,
                            }
                        ],
                    }
                },
            ),
            _mock_response(
                200,
                {
                    "filings": {
                        "recent": {
                            "accessionNumber": ["overflow-25"],
                            "form": ["25-NSE"],
                            "filingDate": ["2026-05-15"],
                            "reportDate": ["2026-05-15"],
                            "acceptanceDateTime": ["2026-05-15T12:00:00Z"],
                        }
                    }
                },
            ),
        ]
        adapter = self._adapter(session)

        resp = adapter.get_filings(
            "1",
            forms=["25", "25-NSE"],
            asof=datetime(2026, 6, 2, tzinfo=timezone.utc),
        )

        assert resp.ok
        assert [filing.accession_number for filing in resp.data] == ["overflow-25"]
        assert session.get.call_count == 2
        assert (
            session.get.call_args_list[1].args[0]
            == "https://data.sec.gov/submissions/CIK0000000001-submissions-001.json"
        )
        assert resp.lineage.data_quality_flags["overflow_pages_available"] == 1
        assert resp.lineage.data_quality_flags["overflow_pages_fetched"] == 1
        assert resp.lineage.data_quality_flags["truncated"] is False

    def test_get_filings_deduplicates_accessions_and_orders_deterministically(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "filings": {
                    "recent": {
                        "accessionNumber": ["dup", "dup", "amend", "base"],
                        "form": ["25-NSE", "25-NSE", "25-NSE/A", "25-NSE"],
                        "filingDate": [
                            "2026-06-01",
                            "2026-06-01",
                            "2026-06-01",
                            "2026-06-01",
                        ],
                        "reportDate": [
                            "2026-06-01",
                            "2026-06-01",
                            "2026-06-01",
                            "2026-06-01",
                        ],
                        "acceptanceDateTime": [
                            "2026-06-01T12:00:00Z",
                            "2026-06-01T12:01:00Z",
                            "2026-06-01T12:02:00Z",
                            "2026-06-01T12:03:00Z",
                        ],
                    }
                }
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_filings(
            "1",
            forms=["25-NSE"],
            asof=datetime(2026, 6, 2, tzinfo=timezone.utc),
        )

        assert resp.ok
        assert [
            (filing.accession_number, filing.form)
            for filing in resp.data
        ] == [
            ("base", "25-NSE"),
            ("amend", "25-NSE/A"),
            ("dup", "25-NSE"),
        ]

    def test_get_survivorship_events_resolves_ticker_and_returns_form25_event(self):
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            _mock_response(
                200,
                {
                    "fields": ["cik", "name", "ticker", "exchange"],
                    "data": [[1234567, "Acme Corp", "ACME", "NYSE"]],
                },
            ),
            _mock_response(
                200,
                {
                    "filings": {
                        "recent": {
                            "accessionNumber": ["0001234567-26-000025"],
                            "form": ["25-NSE"],
                            "filingDate": ["2026-06-04"],
                            "reportDate": ["2026-06-03"],
                            "acceptanceDateTime": ["2026-06-04T21:15:00Z"],
                            "primaryDocument": ["form25.xml"],
                            "primaryDocDescription": ["Form 25 NSE"],
                        }
                    }
                },
            ),
        ]
        adapter = self._adapter(session)

        resp = adapter.get_survivorship_events(
            "acme",
            from_date=date(2026, 6, 1),
            to_date=date(2026, 6, 5),
            asof=datetime(2026, 6, 5, 2, 0, tzinfo=timezone.utc),
        )

        assert resp.ok
        assert len(resp.data) == 1
        event = resp.data[0]
        assert event["id"] == "0001234567-26-000025"
        assert event["event_type"] == "delisting_notice"
        assert event["classification"] == "sec_form_25-nse"
        assert event["source_backed"] is True
        assert event["ticker"] == "ACME"
        assert event["cik"] == "0001234567"
        assert event["company_name"] == "Acme Corp"
        assert event["event_date"] == "2026-06-04"
        assert event["effective_date"] is None
        assert event["knowledge_timestamp"] == "2026-06-05T01:15:00+00:00"
        assert resp.lineage.endpoint == "sec_edgar_survivorship_events"
        assert resp.lineage.data_quality_flags["survivorship_event_count"] == 1
        assert resp.lineage.data_quality_flags["pit_excluded_count"] == 0

    def test_get_survivorship_events_unknown_ticker_returns_unresolved_error(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [[1234567, "Acme Corp", "ACME", "NYSE"]],
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_survivorship_events(
            "NOPE",
            from_date=date(2026, 6, 1),
            to_date=date(2026, 6, 5),
            asof=datetime(2026, 6, 4, 22, 0, tzinfo=timezone.utc),
        )

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "unresolved_entity"
        assert resp.error.retryable is False
        assert resp.lineage.endpoint == "sec_edgar_survivorship_events"
        assert resp.lineage.data_quality_flags["ticker_resolved"] is False

    def test_get_survivorship_events_accepts_direct_cik_without_ticker_lookup(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "filings": {
                    "recent": {
                        "accessionNumber": ["direct"],
                        "form": ["25"],
                        "filingDate": ["2026-06-04"],
                        "reportDate": ["2026-06-03"],
                        "acceptanceDateTime": ["2026-06-04T21:15:00Z"],
                    }
                }
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_survivorship_events(
            "ACME",
            cik="1234567",
            from_date=date(2026, 6, 1),
            to_date=date(2026, 6, 5),
            asof=datetime(2026, 6, 5, 2, 0, tzinfo=timezone.utc),
        )

        assert resp.ok
        assert [event["id"] for event in resp.data] == ["direct"]
        session.get.assert_called_once()
        assert (
            session.get.call_args.args[0]
            == "https://data.sec.gov/submissions/CIK0001234567.json"
        )

    def test_get_survivorship_events_truncated_overflow_without_form25_errors(self):
        files = self._edgar_overflow_file_names(21)
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            _mock_response(
                200,
                {
                    "fields": ["cik", "name", "ticker", "exchange"],
                    "data": [[1, "Acme Corp", "ACME", "NYSE"]],
                },
            ),
            _mock_response(
                200,
                self._edgar_submissions_payload(files=files),
            ),
            *[
                _mock_response(
                    200,
                    self._edgar_submissions_payload(
                        accessions=[f"page-{idx}-8k"],
                        forms=["8-K"],
                    ),
                )
                for idx in range(20)
            ],
        ]
        adapter = self._adapter(session)

        resp = adapter.get_survivorship_events(
            "ACME",
            asof=datetime(2026, 6, 10, tzinfo=timezone.utc),
        )

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "incomplete_window"
        assert resp.error.retryable is False
        flags = resp.lineage.data_quality_flags
        assert flags["truncated"] is True
        assert flags["overflow_pages_available"] == 21
        assert flags["overflow_pages_fetched"] == 20
        assert resp.lineage.source_authority == "SEC_EDGAR"

    def test_get_survivorship_events_exact_cap_empty_overflow_is_valid_empty(self):
        files = self._edgar_overflow_file_names(20)
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            _mock_response(
                200,
                self._edgar_submissions_payload(files=files),
            ),
            *[
                _mock_response(
                    200,
                    self._edgar_submissions_payload(
                        accessions=[f"page-{idx}-8k"],
                        forms=["8-K"],
                    ),
                )
                for idx in range(20)
            ],
        ]
        adapter = self._adapter(session)

        resp = adapter.get_survivorship_events(
            "ACME",
            cik="1",
            asof=datetime(2026, 6, 10, tzinfo=timezone.utc),
        )

        assert resp.ok
        assert resp.data == []
        flags = resp.lineage.data_quality_flags
        assert flags["truncated"] is False
        assert flags["overflow_pages_available"] == 20
        assert flags["overflow_pages_fetched"] == 20
        assert flags["survivorship_event_count"] == 0

    def test_get_survivorship_events_truncation_errors_even_when_form25_found(self):
        files = self._edgar_overflow_file_names(21)
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            _mock_response(
                200,
                self._edgar_submissions_payload(files=files),
            ),
            _mock_response(
                200,
                self._edgar_submissions_payload(
                    accessions=["overflow-25"],
                    forms=["25-NSE"],
                ),
            ),
            *[
                _mock_response(
                    200,
                    self._edgar_submissions_payload(
                        accessions=[f"page-{idx}-8k"],
                        forms=["8-K"],
                    ),
                )
                for idx in range(1, 20)
            ],
        ]
        adapter = self._adapter(session)

        resp = adapter.get_survivorship_events(
            "ACME",
            cik="1",
            asof=datetime(2026, 6, 10, tzinfo=timezone.utc),
        )

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "incomplete_window"
        flags = resp.lineage.data_quality_flags
        assert flags["truncated"] is True
        assert flags["included_count"] == 1
        assert flags["overflow_pages_fetched"] == 20

    def test_get_survivorship_events_direct_cik_truncated_overflow_errors(self):
        files = self._edgar_overflow_file_names(21)
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            _mock_response(
                200,
                self._edgar_submissions_payload(files=files),
            ),
            *[
                _mock_response(
                    200,
                    self._edgar_submissions_payload(
                        accessions=[f"page-{idx}-8k"],
                        forms=["8-K"],
                    ),
                )
                for idx in range(20)
            ],
        ]
        adapter = self._adapter(session)

        resp = adapter.get_survivorship_events(
            "ACME",
            cik="1",
            asof=datetime(2026, 6, 10, tzinfo=timezone.utc),
        )

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "incomplete_window"
        assert resp.lineage.data_quality_flags["ticker_resolved"] is True
        assert resp.lineage.data_quality_flags["truncated"] is True

    def test_get_form4_transactions_fetches_primary_document_and_parses_owner_rows(self):
        xml = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer>
    <issuerCik>0000001234</issuerCik>
    <issuerName>Acme Microcap Inc</issuerName>
    <issuerTradingSymbol>ACME</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0000007777</rptOwnerCik>
      <rptOwnerName>Jane Doe</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerAddress><rptOwnerState>CA</rptOwnerState></reportingOwnerAddress>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>0</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <isOther>0</isOther>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-06-03</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare><value>2.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            _mock_response(
                200,
                {
                    "filings": {
                        "recent": {
                            "accessionNumber": ["0000001234-26-000001"],
                            "form": ["4"],
                            "filingDate": ["2026-06-03"],
                            "reportDate": ["2026-06-03"],
                            "acceptanceDateTime": ["2026-06-03T18:00:00Z"],
                            "primaryDocument": ["xslF345X06/ownership.xml"],
                        }
                    }
                },
            ),
            _mock_response(200, text=xml),
        ]
        adapter = self._adapter(session)

        resp = adapter.get_form4_transactions(
            "0000001234",
            from_date=date(2026, 6, 1),
            to_date=date(2026, 6, 4),
            asof=datetime(2026, 6, 4, 2, 30, tzinfo=timezone.utc),
        )

        assert resp.ok
        assert len(resp.data) == 1
        row = resp.data[0]
        assert row.accession_number == "0000001234-26-000001"
        assert row.ticker == "ACME"
        assert row.insider_cik == "0000007777"
        assert row.transaction_code == "P"
        assert row.acquired_disposed_code == "A"
        assert row.shares == 10000
        assert row.price_per_share == 2.5
        assert row.insider_roles["is_director"] is True
        assert resp.lineage.endpoint == "sec_edgar_form4_transactions"
        assert resp.lineage.data_quality_flags["transaction_count"] == 1
        assert session.get.call_args_list[1].args[0] == (
            "https://www.sec.gov/Archives/edgar/data/1234/000000123426000001/ownership.xml"
        )

    def test_get_form4_transactions_rejects_non_ownership_xml(self):
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            _mock_response(
                200,
                {
                    "filings": {
                        "recent": {
                            "accessionNumber": ["0000001234-26-000001"],
                            "form": ["4"],
                            "filingDate": ["2026-06-03"],
                            "reportDate": ["2026-06-03"],
                            "acceptanceDateTime": ["2026-06-03T18:00:00Z"],
                            "primaryDocument": ["xslF345X06/ownership.xml"],
                        }
                    }
                },
            ),
            _mock_response(200, text="<html><body>not ownership XML</body></html>"),
        ]
        adapter = self._adapter(session)

        resp = adapter.get_form4_transactions(
            "0000001234",
            from_date=date(2026, 6, 1),
            to_date=date(2026, 6, 4),
            asof=datetime(2026, 6, 4, 2, 30, tzinfo=timezone.utc),
        )

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "parse"
        assert "expected ownershipDocument" in resp.error.message
        assert resp.lineage.data_quality_flags["parse_error_accession"] == (
            "0000001234-26-000001"
        )

    @pytest.mark.parametrize(
        ("primary_document", "expected_document"),
        [
            ("xslF345X05/ownership.xml", "ownership.xml"),
            ("xslF345X06/ownership.xml", "ownership.xml"),
            ("xslF345X06/primary_doc.xml", "primary_doc.xml"),
            ("ownership.xml", "ownership.xml"),
        ],
    )
    def test_get_filing_document_strips_form4_xsl_viewer_prefix(
        self,
        primary_document,
        expected_document,
    ):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            text="<?xml version='1.0'?><ownershipDocument/>",
        )
        adapter = self._adapter(session)
        # Avoid a second mocked submissions shape; only the document path is
        # under test here.
        from alpha.data.edgar import SecEdgarFiling

        doc_resp = adapter.get_filing_document(
            SecEdgarFiling(
                cik="0000001234",
                accession_number="0000001234-26-000001",
                form="4",
                filing_date=date(2026, 6, 3),
                report_date=date(2026, 6, 3),
                acceptance_datetime=datetime(2026, 6, 3, 22, tzinfo=timezone.utc),
                primary_document=primary_document,
            ),
            asof=datetime(2026, 6, 4, tzinfo=timezone.utc),
        )

        assert doc_resp.ok
        assert session.get.call_args.args[0] == (
            "https://www.sec.gov/Archives/edgar/data/1234/"
            f"000000123426000001/{expected_document}"
        )


# ---------------------------------------------------------------------------
# FMP adapter
# ---------------------------------------------------------------------------

class TestFmpAdapter:
    def _adapter(self, mock_session):
        return FmpAdapter(_fmp_config(), session=mock_session)

    def test_retry_adapter_and_timeout_contract(self):
        session = requests.Session()

        adapter = self._adapter(session)

        retry_adapter = adapter._session.get_adapter("https://")
        assert retry_adapter.max_retries.total == 3
        assert retry_adapter.max_retries.connect == 3
        assert retry_adapter.max_retries.read == 3
        assert 429 in retry_adapter.max_retries.status_forcelist
        assert FMP_REQUEST_TIMEOUT == (10, 30)

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
            timeout=FMP_REQUEST_TIMEOUT,
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

    def test_get_insider_trades_parses_accession_and_owner_fields(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200,
            [
                {
                    "symbol": "ACME",
                    "filingDate": "2026-06-03",
                    "transactionDate": "2026-06-02",
                    "reportingName": "Jane Doe",
                    "reportingCik": "7777",
                    "companyCik": "1234",
                    "transactionType": "P-Purchase",
                    "acquistionOrDisposition": "A",
                    "securitiesTransacted": "10000",
                    "price": "2.50",
                    "securityName": "Common Stock",
                    "finalLink": "https://www.sec.gov/Archives/edgar/data/1234/000000123426000001/form4.xml",
                }
            ],
        )
        adapter = self._adapter(session)
        asof = datetime(2026, 6, 4, 2, 30, tzinfo=timezone.utc)

        resp = adapter.get_insider_trades(symbol="ACME", page=0, limit=50, asof=asof)

        assert resp.ok
        assert len(resp.data) == 1
        row = resp.data[0]
        assert row.symbol == "ACME"
        assert row.reporting_cik == "0000007777"
        assert row.company_cik == "0000001234"
        assert row.accession_number == "0000001234-26-000001"
        assert row.securities_transacted == 10000
        assert row.price == 2.5
        assert resp.lineage.endpoint == "/stable/insider-trading/search"
        session.get.assert_called_with(
            "https://financialmodelingprep.com/stable/insider-trading/search",
            params={"page": 0, "limit": 50, "symbol": "ACME"},
            timeout=FMP_REQUEST_TIMEOUT,
        )

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
            timeout=FMP_REQUEST_TIMEOUT,
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
            timeout=FMP_REQUEST_TIMEOUT,
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

    def test_get_earnings_calendar_ok(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {
                "symbol": "FIRE",
                "date": "2026-05-20",
                "epsActual": 1.23,
                "epsEstimated": 0.98,
                "time": "bmo",
                "fiscalDateEnding": "2026-03-31",
            }
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        asof = datetime(2026, 5, 20, 21, 0, tzinfo=timezone.utc)

        resp = adapter.get_earnings_calendar(
            from_date=date(2026, 5, 1),
            to_date=date(2026, 5, 20),
            asof=asof,
        )

        assert resp.ok
        assert len(resp.data) == 1
        event = resp.data[0]
        assert event.symbol == "FIRE"
        assert event.actual_eps == 1.23
        assert event.estimated_eps == 0.98
        assert event.announcement_time == "bmo"
        assert event.fiscal_year == 2026
        assert event.fiscal_quarter == 1
        assert resp.lineage.endpoint == "/stable/earnings-calendar"
        assert resp.lineage.asof_timestamp == asof
        session.get.assert_called_with(
            "https://financialmodelingprep.com/stable/earnings-calendar",
            params={"from": "2026-05-01", "to": "2026-05-20"},
            timeout=FMP_REQUEST_TIMEOUT,
        )

    def test_get_earnings_calendar_filters_symbol_client_side(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {"symbol": "FIRE", "date": "2026-05-20", "epsActual": 1.23},
            {"symbol": "LEAK", "date": "2026-05-20", "epsActual": 9.99},
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_earnings_calendar(
            from_date=date(2026, 5, 20),
            to_date=date(2026, 5, 20),
            symbol="FIRE",
        )

        assert resp.ok
        assert [event.symbol for event in resp.data] == ["FIRE"]
        session.get.assert_called_with(
            "https://financialmodelingprep.com/stable/earnings-calendar",
            params={"from": "2026-05-20", "to": "2026-05-20", "symbol": "FIRE"},
            timeout=FMP_REQUEST_TIMEOUT,
        )

    def test_get_earnings_calendar_marks_malformed_eps(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {
                "symbol": "FIRE",
                "date": "2026-05-20",
                "epsActual": "nan",
                "epsEstimated": "bad",
            }
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_earnings_calendar(
            from_date=date(2026, 5, 20),
            to_date=date(2026, 5, 20),
        )

        assert resp.ok
        event = resp.data[0]
        assert event.actual_eps is None
        assert event.estimated_eps is None
        assert "invalid_actual_eps" in event.diagnostics
        assert "invalid_estimated_eps" in event.diagnostics
        assert "announcement_time_missing_conservative_next_session" in event.diagnostics

    def test_get_earnings_history_ok(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {
                "symbol": "FIRE",
                "date": "2026-03-31",
                "eps": 1.23,
                "fiscalYear": "2026",
                "period": "Q1",
            },
            {
                "symbol": "FIRE",
                "date": "2025-12-31",
                "eps": 0.88,
                "fiscalYear": "2025",
                "period": "Q4",
            },
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_earnings_history("FIRE", limit=20)

        assert resp.ok
        assert len(resp.data) == 2
        assert resp.data[0].eps == 1.23
        assert resp.data[0].fiscal_date_ending == "2026-03-31"
        assert resp.data[0].fiscal_year == 2026
        assert resp.data[0].fiscal_quarter == 1
        assert "missing_accepted_date" in resp.data[0].diagnostics
        assert resp.data[1].eps == 0.88
        assert resp.lineage.endpoint == "/stable/income-statement"
        session.get.assert_called_with(
            "https://financialmodelingprep.com/stable/income-statement",
            params={"symbol": "FIRE", "period": "quarter", "limit": 20},
            timeout=FMP_REQUEST_TIMEOUT,
        )

    def test_get_earnings_history_marks_malformed_eps_and_accepted_date(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = [
            {
                "symbol": "FIRE",
                "date": "2026-03-31",
                "eps": "nan",
                "fiscalYear": "2026",
                "period": "Q1",
            }
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_earnings_history("FIRE", limit=20)

        assert resp.ok
        record = resp.data[0]
        assert record.eps is None
        assert "invalid_eps" in record.diagnostics
        assert "missing_accepted_date" in record.diagnostics

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

    def test_reset_session_replaces_session_and_restores_auth_headers(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        adapter = self._adapter(session)

        adapter.reset_session()

        session.close.assert_called_once()
        assert adapter._session is not session
        assert adapter._session.headers["APCA-API-KEY-ID"] == "test-alpaca-key"
        assert adapter._session.headers["APCA-API-SECRET-KEY"] == (
            "test-alpaca-secret"
        )

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

    def test_get_latest_quote_uses_market_data_base_url(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.request.return_value = _mock_response(
            200,
            {
                "quote": {
                    "bp": 10.0,
                    "ap": 10.05,
                    "bs": 100,
                    "as": 200,
                    "t": "2026-06-16T13:40:00Z",
                    "c": ["R"],
                }
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_latest_quote("acme", feed="iex")

        assert resp.ok
        assert resp.data.symbol == "ACME"
        assert resp.data.bid_price == 10.0
        assert resp.data.ask_size == 200.0
        session.request.assert_called_once()
        assert session.request.call_args.args[1].startswith("https://data.alpaca.markets")
        assert session.request.call_args.kwargs["params"] == {"feed": "iex"}

    def test_get_latest_quote_preserves_zero_sizes(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.request.return_value = _mock_response(
            200,
            {
                "quote": {
                    "bp": 10.0,
                    "ap": 10.05,
                    "bs": 0,
                    "as": 0,
                    "bid_size": 100,
                    "ask_size": 200,
                    "t": "2026-06-16T13:40:00Z",
                }
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_latest_quote("acme", feed="iex")

        assert resp.ok
        assert resp.data.bid_size == 0.0
        assert resp.data.ask_size == 0.0

    def test_get_historical_quotes_uses_sip_window_and_paginates(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.request.side_effect = [
            _mock_response(
                200,
                {
                    "quotes": [
                        {
                            "bp": 10.0,
                            "ap": 10.05,
                            "bs": 10,
                            "as": 0,
                            "t": "2026-06-16T13:39:58Z",
                            "c": ["R"],
                        }
                    ],
                    "next_page_token": "next-page",
                },
            ),
            _mock_response(
                200,
                {
                    "quotes": [
                        {
                            "bp": 10.01,
                            "ap": 10.06,
                            "bs": 20,
                            "as": 25,
                            "t": "2026-06-16T13:40:00Z",
                        }
                    ]
                },
            ),
        ]
        adapter = self._adapter(session)
        start = datetime(2026, 6, 16, 13, 38, tzinfo=timezone.utc)
        end = datetime(2026, 6, 16, 13, 40, 5, tzinfo=timezone.utc)

        resp = adapter.get_historical_quotes(
            "acme",
            start=start,
            end=end,
            feed="sip",
            limit=2,
            max_pages=2,
        )

        assert resp.ok
        assert [quote.symbol for quote in resp.data] == ["ACME", "ACME"]
        assert resp.data[0].ask_size == 0.0
        assert resp.data[1].ask_size == 25.0
        assert session.request.call_count == 2
        first_call = session.request.call_args_list[0]
        assert first_call.args[1] == "https://data.alpaca.markets/v2/stocks/ACME/quotes"
        assert first_call.kwargs["params"] == {
            "start": "2026-06-16T13:38:00Z",
            "end": "2026-06-16T13:40:05Z",
            "feed": "sip",
            "limit": 2,
        }
        second_params = session.request.call_args_list[1].kwargs["params"]
        assert second_params["page_token"] == "next-page"
        assert second_params["feed"] == "sip"

    def test_get_historical_quotes_fails_closed_when_page_window_truncated(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.request.return_value = _mock_response(
            200,
            {
                "quotes": [
                    {
                        "bp": 10.0,
                        "ap": 10.05,
                        "bs": 10,
                        "as": 25,
                        "t": "2026-06-16T13:39:58Z",
                    }
                ],
                "next_page_token": "still-more",
            },
        )
        adapter = self._adapter(session)
        start = datetime(2026, 6, 16, 13, 38, tzinfo=timezone.utc)
        end = datetime(2026, 6, 16, 13, 40, 5, tzinfo=timezone.utc)

        resp = adapter.get_historical_quotes(
            "acme",
            start=start,
            end=end,
            feed="sip",
            limit=1,
            max_pages=1,
        )

        assert not resp.ok
        assert resp.data is None
        assert resp.error is not None
        assert resp.error.error_type == "historical_quote_window_truncated"
        assert session.request.call_count == 1

    def test_get_stock_snapshots_parses_intraday_snapshot(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.request.return_value = _mock_response(
            200,
            {
                "ACME": {
                    "dailyBar": {"o": 10.1, "h": 10.2, "l": 9.9, "c": 10.0, "v": 13000},
                    "minuteBar": {
                        "o": 10.08,
                        "h": 10.12,
                        "l": 10.01,
                        "c": 10.08,
                        "v": 2000,
                        "t": "2026-06-16T13:40:00Z",
                    },
                    "latestTrade": {"p": 10.08, "t": "2026-06-16T13:40:00Z"},
                    "latestQuote": {
                        "bp": 10.0,
                        "ap": 10.05,
                        "bs": 100,
                        "as": 200,
                        "t": "2026-06-16T13:40:00Z",
                    },
                }
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_stock_snapshots(["ACME"], feed="iex")

        assert resp.ok
        snapshot = resp.data["ACME"]
        assert snapshot.daily_open == 10.1
        assert snapshot.daily_volume == 13000.0
        assert snapshot.minute_timestamp == "2026-06-16T13:40:00Z"
        assert snapshot.latest_quote.ask_price == 10.05
        session.request.assert_called_once()
        assert session.request.call_args.args[1].startswith("https://data.alpaca.markets")
        assert session.request.call_args.kwargs["params"] == {
            "symbols": "ACME",
            "feed": "iex",
        }

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

    def test_polygon_adapter_mounts_retrying_http_adapter(self):
        session = requests.Session()

        self._adapter(session)

        https_adapter = session.get_adapter("https://api.polygon.io")
        retry = https_adapter.max_retries
        assert retry.total == 3
        assert retry.connect == 3
        assert retry.read == 3
        assert retry.backoff_factor == 0.5
        assert set(retry.status_forcelist) == {429, 500, 502, 503, 504}
        assert retry.respect_retry_after_header is True
        assert set(retry.allowed_methods) == {"GET"}

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("320193", "0000320193"),
            ("0000320193", "0000320193"),
            ("CIK0000320193", "0000320193"),
            ("cik320193", "0000320193"),
        ],
    )
    def test_polygon_normalized_cik_accepts_cik_shaped_values(self, raw, expected):
        assert _normalized_cik(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "BBG000B9XB24",
            "037833100A",
            "abc123",
            "",
            "   ",
            "0",
            "0000000000",
            "12345678901",
        ],
    )
    def test_polygon_normalized_cik_rejects_non_cik_shaped_values(self, raw):
        assert _normalized_cik(raw) is None

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
        resp = adapter.get_short_interest(
            " acme ",
            settlement_date_from="2026-05-01",
            settlement_date_to="2026-05-31",
            limit=60000,
            sort="settlement_date",
            order="desc",
        )

        assert resp.ok
        assert len(resp.data) == 1
        assert resp.data[0].ticker == "ACME"
        assert resp.data[0].settlement_date == "2026-05-15"
        assert resp.data[0].short_interest == 50000
        assert resp.data[0].avg_daily_volume == 200000
        assert resp.data[0].days_to_cover == Decimal("0.25")
        assert resp.data[0].raw == json_data["results"][0]
        assert resp.lineage.provider == "Polygon"
        assert resp.lineage.endpoint == "/stocks/v1/short-interest"
        assert resp.lineage.source_authority == "Polygon"
        assert resp.lineage.data_quality_flags["raw_rows"] == 1
        assert resp.lineage.data_quality_flags["parsed_rows"] == 1
        assert resp.lineage.data_quality_flags["skipped_rows"] == 0
        session.get.assert_called_with(
            "https://api.polygon.io/stocks/v1/short-interest",
            params={
                "ticker": "ACME",
                "settlement_date.gte": "2026-05-01",
                "settlement_date.lte": "2026-05-31",
                "limit": 50000,
                "sort": "settlement_date.desc",
            },
            timeout=POLYGON_REQUEST_TIMEOUT,
        )

    def test_get_short_volume_ok(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {
                    "ticker": "AAPL",
                    "date": "2026-05-27",
                    "short_volume": "11705767.246896",
                    "total_volume": "20717490.359599",
                    "short_volume_ratio": "56.5",
                    "exempt_volume": "33578.0",
                    "non_exempt_volume": "11672189.246896",
                    "adf_short_volume": "0",
                    "nyse_short_volume": "345339",
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_short_volume(
            " aapl ",
            date_from="2026-05-01",
            date_to="2026-05-28",
            limit=25,
            sort="date",
            order="asc",
        )

        assert resp.ok
        assert len(resp.data) == 1
        row = resp.data[0]
        assert row.ticker == "AAPL"
        assert row.date == "2026-05-27"
        assert row.short_volume == Decimal("11705767.246896")
        assert row.total_volume == Decimal("20717490.359599")
        assert row.short_volume_ratio == Decimal("56.5")
        assert row.exempt_volume == Decimal("33578.0")
        assert row.nyse_short_volume == Decimal("345339")
        assert row.raw == json_data["results"][0]
        assert resp.lineage.endpoint == "/stocks/v1/short-volume"
        assert resp.lineage.source_authority == "Polygon"
        assert resp.lineage.data_quality_flags["raw_rows"] == 1
        assert resp.lineage.data_quality_flags["parsed_rows"] == 1
        assert resp.lineage.data_quality_flags["skipped_rows"] == 0
        session.get.assert_called_with(
            "https://api.polygon.io/stocks/v1/short-volume",
            params={
                "ticker": "AAPL",
                "date.gte": "2026-05-01",
                "date.lte": "2026-05-28",
                "limit": 25,
                "sort": "date.asc",
            },
            timeout=POLYGON_REQUEST_TIMEOUT,
        )

    def test_polygon_short_feeds_empty_payload_success(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = [
            _mock_response(200, {"results": []}),
            _mock_response(200, {"results": []}),
        ]
        adapter = self._adapter(session)

        short_interest = adapter.get_short_interest("AAPL")
        short_volume = adapter.get_short_volume("AAPL")

        for resp in (short_interest, short_volume):
            assert resp.ok
            assert resp.data == []
            assert resp.lineage.data_quality_flags["raw_rows"] == 0
            assert resp.lineage.data_quality_flags["parsed_rows"] == 0
            assert resp.lineage.data_quality_flags["skipped_rows"] == 0
            assert "all_rows_skipped" not in resp.lineage.data_quality_flags

    def test_polygon_short_feed_invalid_shape_is_parse_error(self):
        cases = [
            {"unexpected": True},
            {"results": {"not": "a list"}},
            [{"ticker": "AAPL"}],
        ]
        for payload in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            session.get.return_value = _mock_response(200, payload)
            adapter = self._adapter(session)

            resp = adapter.get_short_interest("AAPL")

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "parse"
            assert resp.error.retryable is False
            assert resp.lineage.endpoint == "/stocks/v1/short-interest"
            assert resp.lineage.data_quality_flags["page_count"] == 1
            assert resp.lineage.data_quality_flags["truncated"] is False

    def test_polygon_short_interest_skips_invalid_rows_with_telemetry(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {
                    "ticker": "VALID",
                    "settlement_date": "2026-05-15",
                    "short_interest": "50000",
                    "avg_daily_volume": "200000",
                    "days_to_cover": "0.25",
                },
                {},
                {"settlement_date": "2026-05-15", "short_interest": 1},
                {"ticker": "MISSDATE", "short_interest": 1},
                {"ticker": "IDENTITY", "settlement_date": "2026-05-15"},
                {"ticker": "ONLYAVG", "settlement_date": "2026-05-15", "avg_daily_volume": 200000},
                {"ticker": "ONLYDTC", "settlement_date": "2026-05-15", "days_to_cover": "0.25"},
                {"ticker": "NEGSI", "settlement_date": "2026-05-15", "short_interest": -1},
                {"ticker": "NEGAVG", "settlement_date": "2026-05-15", "avg_daily_volume": -1},
                {"ticker": "NEGDTC", "settlement_date": "2026-05-15", "days_to_cover": "-0.1"},
                {"ticker": "FRACTION", "settlement_date": "2026-05-15", "short_interest": "1.5"},
                None,
                "bad",
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_short_interest("VALID")

        assert resp.ok
        assert [row.ticker for row in resp.data] == ["VALID"]
        assert resp.data[0].short_interest == 50000
        assert resp.data[0].days_to_cover == Decimal("0.25")
        assert resp.lineage.data_quality_flags["raw_rows"] == 13
        assert resp.lineage.data_quality_flags["parsed_rows"] == 1
        assert resp.lineage.data_quality_flags["skipped_rows"] == 12

    def test_polygon_short_interest_requires_short_interest_metric(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {},
                {"ticker": "IDENTITY", "settlement_date": "2026-05-15"},
                {"ticker": "ONLYAVG", "settlement_date": "2026-05-15", "avg_daily_volume": 200000},
                {"ticker": "ONLYDTC", "settlement_date": "2026-05-15", "days_to_cover": "0.25"},
                None,
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_short_interest("AAPL")

        assert resp.ok
        assert resp.data == []
        assert resp.lineage.data_quality_flags["raw_rows"] == 5
        assert resp.lineage.data_quality_flags["parsed_rows"] == 0
        assert resp.lineage.data_quality_flags["skipped_rows"] == 5
        assert resp.lineage.data_quality_flags["all_rows_skipped"] is True

    def test_polygon_short_volume_skips_invalid_rows_and_flags_all_invalid(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {},
                {"date": "2026-05-27", "short_volume": "1"},
                {"ticker": "MISSDATE", "short_volume": "1"},
                {"ticker": "IDENTITY", "date": "2026-05-27"},
                {"ticker": "ONLYTOTAL", "date": "2026-05-27", "total_volume": "100"},
                {"ticker": "ONLYVENUE", "date": "2026-05-27", "nyse_short_volume": "10"},
                {"ticker": "NEGSHORT", "date": "2026-05-27", "short_volume": "-1"},
                {"ticker": "NEGTOTAL", "date": "2026-05-27", "short_volume": "1", "total_volume": "-1"},
                {"ticker": "NEGRATIO", "date": "2026-05-27", "short_volume_ratio": "-0.1"},
                {"ticker": "BADNUM", "date": "2026-05-27", "short_volume": "N/A"},
                None,
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_short_volume("AAPL")

        assert resp.ok
        assert resp.data == []
        assert resp.lineage.data_quality_flags["raw_rows"] == 11
        assert resp.lineage.data_quality_flags["parsed_rows"] == 0
        assert resp.lineage.data_quality_flags["skipped_rows"] == 11
        assert resp.lineage.data_quality_flags["all_rows_skipped"] is True

    def test_polygon_short_volume_requires_short_metric(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {"ticker": "IDENTITY", "date": "2026-05-27"},
                {"ticker": "ONLYTOTAL", "date": "2026-05-27", "total_volume": "100"},
                {"ticker": "ONLYVENUE", "date": "2026-05-27", "nyse_short_volume": "10"},
                {"ticker": "WITHSHORT", "date": "2026-05-27", "short_volume": "1"},
                {"ticker": "WITHRATIO", "date": "2026-05-26", "short_volume_ratio": "12.5"},
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_short_volume("AAPL")

        assert resp.ok
        assert [row.ticker for row in resp.data] == ["WITHSHORT", "WITHRATIO"]
        assert resp.data[0].short_volume == Decimal("1")
        assert resp.data[0].short_volume_ratio is None
        assert resp.data[1].short_volume is None
        assert resp.data[1].short_volume_ratio == Decimal("12.5")
        assert resp.lineage.data_quality_flags["raw_rows"] == 5
        assert resp.lineage.data_quality_flags["parsed_rows"] == 2
        assert resp.lineage.data_quality_flags["skipped_rows"] == 3

    def test_polygon_short_volume_preserves_decimal_precision(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {
                    "ticker": "AAPL",
                    "date": "2026-05-27",
                    "short_volume": "0.333333333333333333",
                    "total_volume": "1.000000000000000001",
                    "short_volume_ratio": "33.333333333333333333",
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_short_volume("AAPL")

        assert resp.ok
        assert resp.data[0].short_volume == Decimal("0.333333333333333333")
        assert resp.data[0].total_volume == Decimal("1.000000000000000001")
        assert resp.data[0].short_volume_ratio == Decimal("33.333333333333333333")
        assert resp.lineage.data_quality_flags["semantic_warning_rows"] == 0
        assert resp.lineage.data_quality_flags["semantic_warning_types"] == {}

    def test_polygon_short_feed_rejects_invalid_dates(self):
        cases = [
            lambda adapter: adapter.get_short_interest("AAPL", settlement_date="bad"),
            lambda adapter: adapter.get_short_interest(
                "AAPL",
                settlement_date_from="2026-99-99",
                settlement_date_to="2026-05-31",
            ),
            lambda adapter: adapter.get_short_interest(
                "AAPL",
                settlement_date_from="2026-05-01",
                settlement_date_to="2026/05/31",
            ),
            lambda adapter: adapter.get_short_interest(
                "AAPL",
                settlement_date_from="2026-05-31",
                settlement_date_to="2026-05-01",
            ),
            lambda adapter: adapter.get_short_volume("AAPL", date=" "),
            lambda adapter: adapter.get_short_volume(
                "AAPL",
                date_from="2026-99-99",
                date_to="2026-05-31",
            ),
            lambda adapter: adapter.get_short_volume(
                "AAPL",
                date_from="2026-05-01",
                date_to="",
            ),
            lambda adapter: adapter.get_short_volume(
                "AAPL",
                date_from="2026-05-31",
                date_to="2026-05-01",
            ),
        ]

        for call in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            adapter = self._adapter(session)

            resp = call(adapter)

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "validation"
            assert resp.error.retryable is False
            session.get.assert_not_called()

    def test_polygon_short_feed_valid_dates_request_normally(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(200, {"results": []})
        adapter = self._adapter(session)

        resp = adapter.get_short_volume("AAPL", date="2026-05-27")

        assert resp.ok
        session.get.assert_called_once_with(
            "https://api.polygon.io/stocks/v1/short-volume",
            params={
                "ticker": "AAPL",
                "date": "2026-05-27",
                "limit": 1000,
                "sort": "date.desc",
            },
            timeout=POLYGON_REQUEST_TIMEOUT,
        )

    def test_polygon_short_volume_flags_semantic_warnings(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {
                    "ticker": "AAPL",
                    "date": "2026-05-27",
                    "short_volume": "200",
                    "total_volume": "100",
                },
                {
                    "ticker": "AAPL",
                    "date": "2026-05-26",
                    "short_volume_ratio": "101",
                },
                {
                    "ticker": "AAPL",
                    "date": "2026-05-25",
                    "short_volume": "10",
                    "short_volume_ratio": "0",
                },
                {
                    "ticker": "AAPL",
                    "date": "2026-05-24",
                    "short_volume": "10",
                    "total_volume": "0",
                },
                {
                    "ticker": "AAPL",
                    "date": "2026-05-23",
                    "short_volume": "100",
                    "exempt_volume": "10",
                    "non_exempt_volume": "20",
                },
                {
                    "ticker": "AAPL",
                    "date": "2026-05-22",
                    "short_volume": "10",
                    "total_volume": "20",
                    "short_volume_ratio": "50",
                    "exempt_volume": "4",
                    "non_exempt_volume": "6",
                },
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_short_volume("AAPL")

        assert resp.ok
        assert len(resp.data) == 6
        flags = resp.lineage.data_quality_flags
        assert flags["semantic_warning_rows"] == 5
        assert flags["semantic_warning_types"] == {
            "short_volume_gt_total": 1,
            "short_volume_ratio_gt_100": 1,
            "zero_ratio_with_positive_short_volume": 1,
            "zero_total_with_positive_short_volume": 1,
            "exempt_non_exempt_sum_mismatch": 1,
        }

    def test_polygon_short_feed_broad_query_requires_date_bound(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        adapter = self._adapter(session)

        no_interest_ticker = adapter.get_short_interest(None)
        blank_volume_ticker = adapter.get_short_volume(" ")

        assert not no_interest_ticker.ok
        assert no_interest_ticker.error.error_type == "validation"
        assert not blank_volume_ticker.ok
        assert blank_volume_ticker.error.error_type == "validation"
        session.get.assert_not_called()

    def test_polygon_short_feed_bounded_broad_query_is_allowed(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = [
            _mock_response(200, {"results": []}),
            _mock_response(200, {"results": []}),
            _mock_response(200, {"results": []}),
        ]
        adapter = self._adapter(session)

        exact_interest = adapter.get_short_interest(None, settlement_date="2026-05-15")
        ranged_interest = adapter.get_short_interest(
            None,
            settlement_date_from="2026-05-01",
            settlement_date_to="2026-05-31",
        )
        ranged_volume = adapter.get_short_volume(
            " ",
            date_from="2026-05-01",
            date_to="2026-05-31",
        )

        assert exact_interest.ok
        assert ranged_interest.ok
        assert ranged_volume.ok
        exact_params = session.get.call_args_list[0].kwargs["params"]
        interest_params = session.get.call_args_list[1].kwargs["params"]
        volume_params = session.get.call_args_list[2].kwargs["params"]
        assert "ticker" not in exact_params
        assert exact_params["settlement_date"] == "2026-05-15"
        assert "ticker" not in interest_params
        assert interest_params["settlement_date.gte"] == "2026-05-01"
        assert interest_params["settlement_date.lte"] == "2026-05-31"
        assert "ticker" not in volume_params
        assert volume_params["date.gte"] == "2026-05-01"
        assert volume_params["date.lte"] == "2026-05-31"

    def test_polygon_short_feed_provider_errors(self):
        expected = {
            403: "auth",
            429: "rate_limit",
            500: "http",
        }
        for status_code, error_type in expected.items():
            session = MagicMock(spec=requests.Session)
            session.params = {}
            session.get.return_value = _mock_response(status_code, text="provider error")
            adapter = self._adapter(session)

            resp = adapter.get_short_volume("AAPL")

            assert not resp.ok
            assert resp.error.error_type == error_type
            assert resp.error.status_code == status_code

    def test_polygon_short_feed_timeout_and_request_exception_are_safe(self):
        timeout_session = MagicMock(spec=requests.Session)
        timeout_session.params = {}
        timeout_session.get.side_effect = requests.exceptions.Timeout("timed out")
        timeout_adapter = self._adapter(timeout_session)

        timeout_resp = timeout_adapter.get_short_interest("AAPL")

        assert not timeout_resp.ok
        assert timeout_resp.error.error_type == "timeout"
        assert timeout_resp.error.retryable is True

        error_session = MagicMock(spec=requests.Session)
        error_session.params = {}
        error_session.get.side_effect = requests.exceptions.ConnectionError(
            "failed https://api.polygon.io/stocks/v1/short-volume?apiKey=test-polygon-key&ticker=AAPL"
        )
        error_adapter = self._adapter(error_session)

        error_resp = error_adapter.get_short_volume("AAPL")

        assert not error_resp.ok
        assert error_resp.error.error_type == "http"
        assert error_resp.error.message == "Polygon request failed: ConnectionError"
        assert "test-polygon-key" not in error_resp.error.message
        assert "apiKey" not in error_resp.error.message
        assert "https://api.polygon.io" not in error_resp.error.message

    def test_polygon_short_feed_malformed_json_is_parse_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(200, text="not json")
        adapter = self._adapter(session)

        resp = adapter.get_short_volume("AAPL")

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "parse"

    def test_polygon_short_feed_paginates_and_sanitizes_next_url(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        page_1 = {
            "results": [
                {
                    "ticker": "AAPL",
                    "date": "2026-05-27",
                    "short_volume": "10",
                    "total_volume": "20",
                    "short_volume_ratio": "50",
                }
            ],
            "next_url": "https://api.massive.com/stocks/v1/short-volume?cursor=abc&apiKey=SECRET",
        }
        page_2 = {
            "results": [
                {
                    "ticker": "AAPL",
                    "date": "2026-05-26",
                    "short_volume": "11",
                    "total_volume": "22",
                    "short_volume_ratio": "50",
                }
            ]
        }
        session.get.side_effect = [
            _mock_response(200, page_1),
            _mock_response(200, page_2),
        ]
        adapter = self._adapter(session)

        resp = adapter.get_short_volume("AAPL", limit=1)

        assert resp.ok
        assert [row.date for row in resp.data] == ["2026-05-27", "2026-05-26"]
        assert session.get.call_count == 2
        assert session.get.call_args_list[1].args[0] == "https://api.polygon.io/stocks/v1/short-volume"
        assert session.get.call_args_list[1].kwargs["params"] == {"cursor": "abc"}
        assert "SECRET" not in repr(session.get.call_args_list[1])
        assert resp.lineage.data_quality_flags["page_count"] == 2
        assert resp.lineage.data_quality_flags["paginated"] is True
        assert resp.lineage.data_quality_flags["truncated"] is False
        assert resp.lineage.data_quality_flags["next_url_paths"] == ["/stocks/v1/short-volume"]
        assert resp.lineage.data_quality_flags["raw_rows"] == 2
        assert resp.lineage.data_quality_flags["parsed_rows"] == 2
        assert "SECRET" not in repr(resp.lineage)
        assert "apiKey" not in repr(resp.lineage)

    def test_polygon_short_feed_accepts_default_port_next_urls(self):
        cases = [
            (
                "https://api.polygon.io:443/stocks/v1/short-volume?cursor=abc&apiKey=SECRET&token=SECRET",
                "get_short_volume",
                "/stocks/v1/short-volume",
            ),
            (
                "https://api.massive.com:443/stocks/v1/short-interest?cursor=def&apiKey=SECRET",
                "get_short_interest",
                "/stocks/v1/short-interest",
            ),
        ]
        for next_url, method_name, endpoint in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            if method_name == "get_short_volume":
                page_1 = {
                    "results": [
                        {
                            "ticker": "AAPL",
                            "date": "2026-05-27",
                            "short_volume": "10",
                        }
                    ],
                    "next_url": next_url,
                }
                page_2 = {
                    "results": [
                        {
                            "ticker": "AAPL",
                            "date": "2026-05-26",
                            "short_volume": "11",
                        }
                    ]
                }
            else:
                page_1 = {
                    "results": [
                        {
                            "ticker": "AAPL",
                            "settlement_date": "2026-05-15",
                            "short_interest": "10",
                        }
                    ],
                    "next_url": next_url,
                }
                page_2 = {
                    "results": [
                        {
                            "ticker": "AAPL",
                            "settlement_date": "2026-04-30",
                            "short_interest": "11",
                        }
                    ]
                }
            session.get.side_effect = [
                _mock_response(200, page_1),
                _mock_response(200, page_2),
            ]
            adapter = self._adapter(session)

            resp = getattr(adapter, method_name)("AAPL", limit=1)

            assert resp.ok
            assert len(resp.data) == 2
            assert session.get.call_count == 2
            assert session.get.call_args_list[1].args[0] == f"https://api.polygon.io{endpoint}"
            expected_cursor = "abc" if method_name == "get_short_volume" else "def"
            assert session.get.call_args_list[1].kwargs["params"] == {
                "cursor": expected_cursor
            }
            assert "SECRET" not in repr(session.get.call_args_list[1])
            assert "apiKey" not in repr(session.get.call_args_list[1])
            assert "token" not in repr(session.get.call_args_list[1])
            assert resp.lineage.data_quality_flags["next_url_paths"] == [endpoint]
            assert "SECRET" not in repr(resp.lineage)
            assert "apiKey" not in repr(resp.lineage)
            assert "token" not in repr(resp.lineage)

    def test_polygon_short_feed_evil_next_url_is_rejected(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        page_1 = {
            "results": [
                {
                    "ticker": "AAPL",
                    "settlement_date": "2026-05-15",
                    "short_interest": "50000",
                }
            ],
            "next_url": "https://evil.example/stocks/v1/short-interest?cursor=abc&apiKey=SECRET",
        }
        session.get.return_value = _mock_response(200, page_1)
        adapter = self._adapter(session)

        resp = adapter.get_short_interest("AAPL", limit=1)

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "pagination"
        assert session.get.call_count == 1
        assert "evil.example" not in repr(resp.lineage)
        assert "SECRET" not in repr(resp.lineage)
        assert "apiKey" not in repr(resp.lineage)

    def test_polygon_short_feed_invalid_max_pages_is_validation_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        adapter = self._adapter(session)

        bad = adapter.get_short_interest("AAPL", max_pages="bad")  # type: ignore[arg-type]
        zero = adapter.get_short_volume("AAPL", max_pages=0)
        negative = adapter.get_short_interest("AAPL", max_pages=-1)

        for resp in (bad, zero, negative):
            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "validation"
            assert resp.error.retryable is False
        session.get.assert_not_called()

    def test_get_news_ok_preserves_sentiment_and_params(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {
                    "id": "news-1",
                    "publisher": {
                        "name": "Example Wire",
                        "homepage_url": "https://example.com",
                        "logo_url": "https://example.com/logo.png",
                        "favicon_url": "https://example.com/favicon.ico",
                    },
                    "title": "Apple shares move on new product",
                    "author": "Market Desk",
                    "article_url": "https://example.com/aapl-news",
                    "amp_url": "https://amp.example.com/aapl-news",
                    "image_url": "https://example.com/aapl.jpg",
                    "description": "AAPL article summary",
                    "published_utc": "2026-05-27T12:00:00Z",
                    "tickers": ["AAPL", "MSFT"],
                    "keywords": ["products", "sentiment"],
                    "insights": [
                        {
                            "ticker": "AAPL",
                            "sentiment": "positive",
                            "sentiment_reasoning": "Demand commentary was constructive.",
                        }
                    ],
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_news(
            ticker=" aapl ",
            published_utc_from="2026-05-01",
            published_utc_to="2026-05-28T00:00:00Z",
            limit=2000,
            sort="published_utc",
            order="DESC",
        )

        assert resp.ok
        assert len(resp.data) == 1
        article = resp.data[0]
        assert article.id == "news-1"
        assert article.publisher_name == "Example Wire"
        assert article.publisher_homepage_url == "https://example.com"
        assert article.publisher_logo_url == "https://example.com/logo.png"
        assert article.publisher_favicon_url == "https://example.com/favicon.ico"
        assert article.title == "Apple shares move on new product"
        assert article.author == "Market Desk"
        assert article.article_url == "https://example.com/aapl-news"
        assert article.amp_url == "https://amp.example.com/aapl-news"
        assert article.image_url == "https://example.com/aapl.jpg"
        assert article.description == "AAPL article summary"
        assert article.published_utc == "2026-05-27T12:00:00Z"
        assert article.tickers == ["AAPL", "MSFT"]
        assert article.keywords == ["products", "sentiment"]
        assert article.insights[0]["sentiment"] == "positive"
        assert article.insights[0]["sentiment_reasoning"] == (
            "Demand commentary was constructive."
        )
        assert article.publisher == json_data["results"][0]["publisher"]
        assert article.raw == json_data["results"][0]
        assert resp.lineage.provider == "Polygon"
        assert resp.lineage.endpoint == "/v2/reference/news"
        assert resp.lineage.source_authority == "Polygon"
        assert resp.lineage.data_quality_flags["raw_rows"] == 1
        assert resp.lineage.data_quality_flags["parsed_rows"] == 1
        assert resp.lineage.data_quality_flags["skipped_rows"] == 0
        session.get.assert_called_with(
            "https://api.polygon.io/v2/reference/news",
            params={
                "ticker": "AAPL",
                "published_utc.gte": "2026-05-01",
                "published_utc.lte": "2026-05-28T00:00:00Z",
                "limit": 1000,
                "sort": "published_utc",
                "order": "desc",
            },
            timeout=POLYGON_REQUEST_TIMEOUT,
        )

    def test_polygon_news_skips_invalid_rows_with_telemetry(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {
                    "id": "valid",
                    "title": "Valid article",
                    "article_url": "https://example.com/valid",
                },
                {
                    "id": "valid-http",
                    "title": "Valid article over http",
                    "article_url": "http://example.com/valid",
                },
                {},
                {"id": "missing-title", "article_url": "https://example.com/no-title"},
                {"id": "missing-url", "title": "No URL"},
                {"title": "No ID", "article_url": "https://example.com/no-id"},
                {"id": 123, "title": "Numeric ID", "article_url": "https://example.com/id"},
                {"id": "numeric-title", "title": 456, "article_url": "https://example.com/title"},
                {"id": "numeric-url", "title": "Numeric URL", "article_url": 789},
                {"id": "blank-title", "title": " ", "article_url": "https://example.com/blank"},
                {"id": "blank-url", "title": "Blank URL", "article_url": " "},
                {"id": "no-scheme", "title": "No scheme", "article_url": "example.com/no-scheme"},
                {"id": "ftp", "title": "FTP URL", "article_url": "ftp://example.com/no"},
                None,
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_news("AAPL")

        assert resp.ok
        assert [row.id for row in resp.data] == ["valid", "valid-http"]
        assert resp.lineage.data_quality_flags["raw_rows"] == 14
        assert resp.lineage.data_quality_flags["parsed_rows"] == 2
        assert resp.lineage.data_quality_flags["skipped_rows"] == 12

    def test_polygon_news_rejects_malformed_article_urls_with_telemetry(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {
                    "id": "valid-http",
                    "title": "Valid HTTP article",
                    "article_url": "http://example.com/valid",
                },
                {
                    "id": "valid-https",
                    "title": "Valid HTTPS article",
                    "article_url": "https://sub.example.com/path?x=1",
                },
                {"id": "empty-https", "title": "Empty HTTPS", "article_url": "https://"},
                {"id": "path-only", "title": "Path only", "article_url": "https:///path"},
                {"id": "empty-http", "title": "Empty HTTP", "article_url": "http://"},
                {"id": "ftp", "title": "FTP", "article_url": "ftp://example.com/a"},
                {"id": "no-scheme", "title": "No scheme", "article_url": "example.com/a"},
                {
                    "id": "control",
                    "title": "Control char",
                    "article_url": "https://example.com/a\nsecret",
                },
                {"id": "numeric-url", "title": "Numeric URL", "article_url": 123},
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_news("AAPL")

        assert resp.ok
        assert [row.id for row in resp.data] == ["valid-http", "valid-https"]
        assert resp.lineage.data_quality_flags["raw_rows"] == 9
        assert resp.lineage.data_quality_flags["parsed_rows"] == 2
        assert resp.lineage.data_quality_flags["skipped_rows"] == 7

    def test_polygon_news_empty_payload_success(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(200, {"results": []})
        adapter = self._adapter(session)

        resp = adapter.get_news("AAPL")

        assert resp.ok
        assert resp.data == []
        assert resp.lineage.endpoint == "/v2/reference/news"
        assert resp.lineage.data_quality_flags["raw_rows"] == 0
        assert resp.lineage.data_quality_flags["parsed_rows"] == 0
        assert resp.lineage.data_quality_flags["skipped_rows"] == 0
        assert "all_rows_skipped" not in resp.lineage.data_quality_flags

    def test_polygon_news_invalid_shape_is_parse_error(self):
        cases = [
            {"unexpected": True},
            {"results": {"not": "a list"}},
            [{"id": "news-1"}],
        ]
        for payload in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            session.get.return_value = _mock_response(200, payload)
            adapter = self._adapter(session)

            resp = adapter.get_news("AAPL")

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "parse"
            assert resp.error.retryable is False
            assert resp.lineage.endpoint == "/v2/reference/news"
            assert resp.lineage.data_quality_flags["page_count"] == 1

    def test_polygon_news_malformed_json_is_parse_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(200, text="not json")
        adapter = self._adapter(session)

        resp = adapter.get_news("AAPL")

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "parse"
        assert resp.lineage.endpoint == "/v2/reference/news"

    def test_polygon_news_provider_errors(self):
        expected = {
            403: "auth",
            429: "rate_limit",
            500: "http",
        }
        for status_code, error_type in expected.items():
            session = MagicMock(spec=requests.Session)
            session.params = {}
            session.get.return_value = _mock_response(status_code, text="provider error")
            adapter = self._adapter(session)

            resp = adapter.get_news("AAPL")

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == error_type
            assert resp.error.status_code == status_code

    def test_polygon_news_timeout_and_request_exception_are_safe(self):
        timeout_session = MagicMock(spec=requests.Session)
        timeout_session.params = {}
        timeout_session.get.side_effect = requests.exceptions.Timeout("timed out")
        timeout_adapter = self._adapter(timeout_session)

        timeout_resp = timeout_adapter.get_news("AAPL")

        assert not timeout_resp.ok
        assert timeout_resp.error.error_type == "timeout"
        assert timeout_resp.error.retryable is True

        error_session = MagicMock(spec=requests.Session)
        error_session.params = {}
        error_session.get.side_effect = requests.exceptions.ConnectionError(
            "failed https://api.polygon.io/v2/reference/news?apiKey=test-polygon-key&ticker=AAPL"
        )
        error_adapter = self._adapter(error_session)

        error_resp = error_adapter.get_news("AAPL")

        assert not error_resp.ok
        assert error_resp.error.error_type == "http"
        assert error_resp.error.message == "Polygon request failed: ConnectionError"
        assert "test-polygon-key" not in error_resp.error.message
        assert "apiKey" not in error_resp.error.message
        assert "https://api.polygon.io" not in error_resp.error.message

    def test_polygon_news_rejects_invalid_published_dates(self):
        cases = [
            lambda adapter: adapter.get_news("AAPL", published_utc="bad"),
            lambda adapter: adapter.get_news("AAPL", published_utc="2026-99-99"),
            lambda adapter: adapter.get_news("AAPL", published_utc="2026/05/27"),
            lambda adapter: adapter.get_news("AAPL", published_utc=" "),
            lambda adapter: adapter.get_news("AAPL", published_utc="2026-05-27T12:00:00"),
            lambda adapter: adapter.get_news("AAPL", published_utc="2026-05-27 12:00:00"),
            lambda adapter: adapter.get_news(
                "AAPL",
                published_utc_from="2026-05-01",
                published_utc_to="2026-99-99",
            ),
            lambda adapter: adapter.get_news(
                "AAPL",
                published_utc_from="2026-05-31",
                published_utc_to="2026-05-01",
            ),
        ]

        for call in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            adapter = self._adapter(session)

            resp = call(adapter)

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "validation"
            assert resp.error.retryable is False
            session.get.assert_not_called()

    def test_polygon_news_accepts_date_and_timezone_aware_datetime_filters(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = [
            _mock_response(200, {"results": []}),
            _mock_response(200, {"results": []}),
            _mock_response(200, {"results": []}),
        ]
        adapter = self._adapter(session)

        date_resp = adapter.get_news("AAPL", published_utc="2026-05-27")
        z_resp = adapter.get_news("AAPL", published_utc="2026-05-27T12:00:00Z")
        offset_resp = adapter.get_news("AAPL", published_utc="2026-05-27T12:00:00+00:00")

        assert date_resp.ok
        assert z_resp.ok
        assert offset_resp.ok
        assert session.get.call_args_list[0].kwargs["params"]["published_utc"] == "2026-05-27"
        assert (
            session.get.call_args_list[1].kwargs["params"]["published_utc"]
            == "2026-05-27T12:00:00Z"
        )
        assert (
            session.get.call_args_list[2].kwargs["params"]["published_utc"]
            == "2026-05-27T12:00:00+00:00"
        )

    def test_polygon_same_day_date_ranges_request_normally(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = [_mock_response(200, {"results": []}) for _ in range(5)]
        adapter = self._adapter(session)

        responses = [
            adapter.get_splits(
                None,
                execution_date_from="2026-05-27",
                execution_date_to="2026-05-27",
            ),
            adapter.get_dividends(
                None,
                ex_dividend_date_from="2026-05-27",
                ex_dividend_date_to="2026-05-27",
            ),
            adapter.get_short_interest(
                None,
                settlement_date_from="2026-05-27",
                settlement_date_to="2026-05-27",
            ),
            adapter.get_short_volume(
                None,
                date_from="2026-05-27",
                date_to="2026-05-27",
            ),
            adapter.get_news(
                published_utc_from="2026-05-27",
                published_utc_to="2026-05-27",
            ),
        ]

        assert all(resp.ok for resp in responses)
        assert session.get.call_count == 5

    def test_polygon_news_normalizes_tickers_and_omits_blank_filter(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = [
            _mock_response(200, {"results": []}),
            _mock_response(200, {"results": []}),
            _mock_response(200, {"results": []}),
        ]
        adapter = self._adapter(session)

        ticker_resp = adapter.get_news(tickers=[" msft ", "", "MSFT"])
        blank_resp = adapter.get_news(
            ticker=" ",
            tickers=["", " "],
            published_utc_from="2026-05-01",
            published_utc_to="2026-05-28",
        )
        exact_resp = adapter.get_news(
            "qqq",
            published_utc="2026-05-27T12:00:00Z",
        )

        assert ticker_resp.ok
        assert blank_resp.ok
        assert exact_resp.ok
        assert session.get.call_args_list[0].kwargs["params"]["ticker"] == "MSFT"
        assert "ticker" not in session.get.call_args_list[1].kwargs["params"]
        assert session.get.call_args_list[1].kwargs["params"]["published_utc.gte"] == "2026-05-01"
        assert session.get.call_args_list[1].kwargs["params"]["published_utc.lte"] == "2026-05-28"
        assert session.get.call_args_list[2].kwargs["params"]["ticker"] == "QQQ"
        assert (
            session.get.call_args_list[2].kwargs["params"]["published_utc"]
            == "2026-05-27T12:00:00Z"
        )

    def test_polygon_news_rejects_multi_ticker_input_without_request(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        adapter = self._adapter(session)

        list_resp = adapter.get_news(tickers=["AAPL", "MSFT"])
        tuple_resp = adapter.get_news(tickers=("AAPL", "MSFT"))  # type: ignore[arg-type]
        set_resp = adapter.get_news(tickers={"AAPL", "MSFT"})  # type: ignore[arg-type]
        mixed_resp = adapter.get_news(ticker="QQQ", tickers=["AAPL"])

        for resp in (list_resp, tuple_resp, set_resp, mixed_resp):
            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "validation"
            assert resp.error.retryable is False
        session.get.assert_not_called()

    def test_polygon_news_rejects_comma_and_non_string_tickers_without_request(self):
        cases = [
            lambda adapter: adapter.get_news(ticker="AAPL,MSFT"),
            lambda adapter: adapter.get_news(ticker=" AAPL , MSFT "),
            lambda adapter: adapter.get_news(tickers="AAPL,MSFT"),  # type: ignore[arg-type]
            lambda adapter: adapter.get_news(tickers=["AAPL,MSFT"]),
            lambda adapter: adapter.get_news(tickers=[" AAPL , MSFT "]),
            lambda adapter: adapter.get_news(tickers=[123]),  # type: ignore[list-item]
            lambda adapter: adapter.get_news(tickers=[object()]),  # type: ignore[list-item]
        ]
        for call in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            adapter = self._adapter(session)

            resp = call(adapter)

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "validation"
            assert resp.error.retryable is False
            session.get.assert_not_called()

    def test_polygon_news_broad_query_requires_date_bounds(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        adapter = self._adapter(session)

        empty_resp = adapter.get_news()
        blank_resp = adapter.get_news(ticker=" ", tickers=["", " "])
        one_sided_resp = adapter.get_news(published_utc_from="2026-05-01")

        for resp in (empty_resp, blank_resp, one_sided_resp):
            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "validation"
            assert resp.error.retryable is False
        session.get.assert_not_called()

    def test_polygon_news_bounded_broad_query_is_allowed(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = [
            _mock_response(200, {"results": []}),
            _mock_response(200, {"results": []}),
        ]
        adapter = self._adapter(session)

        exact_resp = adapter.get_news(published_utc="2026-05-27")
        ranged_resp = adapter.get_news(
            published_utc_from="2026-05-01",
            published_utc_to="2026-05-28",
        )

        assert exact_resp.ok
        assert ranged_resp.ok
        exact_params = session.get.call_args_list[0].kwargs["params"]
        ranged_params = session.get.call_args_list[1].kwargs["params"]
        assert "ticker" not in exact_params
        assert exact_params["published_utc"] == "2026-05-27"
        assert "ticker" not in ranged_params
        assert ranged_params["published_utc.gte"] == "2026-05-01"
        assert ranged_params["published_utc.lte"] == "2026-05-28"

    def test_polygon_news_parsed_mutation_does_not_mutate_raw_payload(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {
                    "id": "news-1",
                    "publisher": {"name": "Publisher", "nested": {"rank": 1}},
                    "title": "Article",
                    "article_url": "https://example.com/article",
                    "tickers": ["AAPL"],
                    "keywords": ["sentiment"],
                    "insights": [
                        {
                            "ticker": "AAPL",
                            "sentiment": "positive",
                            "nested": {"score": 1},
                        }
                    ],
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_news("AAPL")

        assert resp.ok
        article = resp.data[0]
        article.publisher["name"] = "Changed"
        article.publisher["nested"]["rank"] = 2
        article.insights[0]["sentiment"] = "negative"
        article.insights[0]["nested"]["score"] = 2
        article.tickers.append("MSFT")
        article.keywords.append("changed")
        assert article.raw["publisher"]["name"] == "Publisher"
        assert article.raw["publisher"]["nested"]["rank"] == 1
        assert article.raw["insights"][0]["sentiment"] == "positive"
        assert article.raw["insights"][0]["nested"]["score"] == 1
        assert article.raw["tickers"] == ["AAPL"]
        assert article.raw["keywords"] == ["sentiment"]

    def test_polygon_news_paginates_and_sanitizes_next_url(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        page_1 = {
            "results": [
                {
                    "id": "news-1",
                    "title": "First",
                    "article_url": "https://example.com/first",
                }
            ],
            "next_url": "https://api.massive.com/v2/reference/news?cursor=abc&apiKey=SECRET&token=SECRET",
        }
        page_2 = {
            "results": [
                {
                    "id": "news-2",
                    "title": "Second",
                    "article_url": "https://example.com/second",
                }
            ]
        }
        session.get.side_effect = [
            _mock_response(200, page_1),
            _mock_response(200, page_2),
        ]
        adapter = self._adapter(session)

        resp = adapter.get_news("AAPL", limit=1)

        assert resp.ok
        assert [row.id for row in resp.data] == ["news-1", "news-2"]
        assert session.get.call_count == 2
        assert session.get.call_args_list[0].args[0] == "https://api.polygon.io/v2/reference/news"
        assert session.get.call_args_list[1].args[0] == "https://api.polygon.io/v2/reference/news"
        assert session.get.call_args_list[1].kwargs["params"] == {"cursor": "abc"}
        assert "SECRET" not in repr(session.get.call_args_list[1])
        assert "apiKey" not in repr(session.get.call_args_list[1])
        assert "token" not in repr(session.get.call_args_list[1])
        assert resp.lineage.data_quality_flags["page_count"] == 2
        assert resp.lineage.data_quality_flags["paginated"] is True
        assert resp.lineage.data_quality_flags["truncated"] is False
        assert resp.lineage.data_quality_flags["next_url_paths"] == ["/v2/reference/news"]
        assert resp.lineage.data_quality_flags["raw_rows"] == 2
        assert resp.lineage.data_quality_flags["parsed_rows"] == 2
        assert resp.lineage.data_quality_flags["skipped_rows"] == 0
        assert "SECRET" not in repr(resp.lineage)
        assert "apiKey" not in repr(resp.lineage)
        assert "token" not in repr(resp.lineage)

    def test_polygon_news_second_page_failures_are_loud(self):
        page_1 = {
            "results": [
                {
                    "id": "news-1",
                    "title": "First",
                    "article_url": "https://example.com/first",
                }
            ],
            "next_url": "/v2/reference/news?cursor=abc&apiKey=SECRET",
        }
        cases = [
            (_mock_response(429, text="Too many requests"), "rate_limit"),
            (_mock_response(200, text="not json"), "parse"),
            (_mock_response(200, {"unexpected": True}), "parse"),
        ]
        for response, error_type in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            session.get.side_effect = [
                _mock_response(200, page_1),
                response,
            ]
            adapter = self._adapter(session)

            resp = adapter.get_news("AAPL", limit=1)

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == error_type
            assert session.get.call_count == 2
            assert resp.lineage.endpoint == "/v2/reference/news"
            assert resp.lineage.data_quality_flags["page_count"] == 2
            assert resp.lineage.data_quality_flags["paginated"] is True
            assert resp.lineage.data_quality_flags["truncated"] is True
            assert resp.lineage.data_quality_flags["next_url_paths"] == ["/v2/reference/news"]
            assert "SECRET" not in repr(resp.lineage)
            assert "apiKey" not in repr(resp.lineage)

    def test_polygon_news_evil_next_url_is_rejected_before_request(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        page_1 = {
            "results": [
                {
                    "id": "news-1",
                    "title": "First",
                    "article_url": "https://example.com/first",
                }
            ],
            "next_url": "https://evil.example/v2/reference/news?cursor=abc&apiKey=SECRET",
        }
        session.get.return_value = _mock_response(200, page_1)
        adapter = self._adapter(session)

        resp = adapter.get_news("AAPL", limit=1)

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "pagination"
        assert resp.error.retryable is False
        assert session.get.call_count == 1
        assert "evil.example" not in repr(resp.lineage)
        assert "SECRET" not in repr(resp.lineage)
        assert "apiKey" not in repr(resp.lineage)

    def test_polygon_news_lineage_hash_stability(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {
                    "id": "news-1",
                    "title": "First",
                    "article_url": "https://example.com/first",
                }
            ]
        }
        adapter = self._adapter(session)

        session.get.return_value = _mock_response(200, json_data)
        resp1 = adapter.get_news("AAPL")
        resp2 = adapter.get_news("AAPL")

        assert resp1.lineage.raw_payload_hash == resp2.lineage.raw_payload_hash

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
        assert resp.data[0].volume == 100000
        assert resp.data[0].vwap == 5.15
        assert resp.lineage.data_quality_flags["raw_rows"] == 1
        assert resp.lineage.data_quality_flags["parsed_rows"] == 1
        assert resp.lineage.data_quality_flags["skipped_rows"] == 0
        session.get.assert_called_once_with(
            "https://api.polygon.io/v2/aggs/ticker/ACME/range/1/day/2026-05-01/2026-05-19",
            params={"limit": 5000, "sort": "asc", "adjusted": "true"},
            timeout=POLYGON_REQUEST_TIMEOUT,
        )

    def test_get_daily_bars_accepts_float_volume(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {
                    "t": 1716163200000,
                    "o": 5.0,
                    "h": 5.5,
                    "l": 4.9,
                    "c": 5.25,
                    "v": 35324922.433075,
                    "vw": 5.15,
                    "n": 450,
                },
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_daily_bars("ACME", "2026-05-01", "2026-05-19")

        assert resp.ok
        assert resp.data[0].close == 5.25
        assert resp.data[0].volume == pytest.approx(35324922.433075)
        assert resp.lineage.data_quality_flags["parsed_rows"] == 1

    def test_get_daily_bars_rejects_invalid_ticker_and_dates_without_request(self):
        invalid_calls = [
            lambda adapter: adapter.get_daily_bars("", "2026-05-01", "2026-05-02"),
            lambda adapter: adapter.get_daily_bars("AAPL/../../x", "2026-05-01", "2026-05-02"),
            lambda adapter: adapter.get_daily_bars("AAPL?x=1", "2026-05-01", "2026-05-02"),
            lambda adapter: adapter.get_daily_bars("AAPL", "bad", "2026-05-02"),
            lambda adapter: adapter.get_daily_bars("AAPL", "2026-05-01", "2026-13-45"),
            lambda adapter: adapter.get_daily_bars("AAPL", "2026-05-03", "2026-05-02"),
        ]
        for call in invalid_calls:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            adapter = self._adapter(session)

            resp = call(adapter)

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "validation"
            session.get.assert_not_called()

    def test_get_daily_bars_malformed_results_are_parse_errors(self):
        cases = [
            {"unexpected": True},
            {"results": {"not": "list"}},
            [],
            {"results": [None]},
            {"results": [{"t": 1716163200000, "o": 5.0, "h": 5.5, "l": 4.9, "v": 100000}]},
            {"results": [{"t": 1716163200000, "o": 5.0, "h": 5.5, "l": 4.9, "c": 5.25, "v": "N/A"}]},
            {"results": [{"t": 1716163200000, "o": -1.0, "h": 5.5, "l": 4.9, "c": 5.25, "v": 100000}]},
            {"results": [{"t": 1716163200000, "o": 5.0, "h": -1.0, "l": 4.9, "c": 5.25, "v": 100000}]},
            {"results": [{"t": 1716163200000, "o": 5.0, "h": 5.5, "l": -1.0, "c": 5.25, "v": 100000}]},
            {"results": [{"t": 1716163200000, "o": 5.0, "h": 5.5, "l": 4.9, "c": -1.0, "v": 100000}]},
            {"results": [{"t": 1716163200000, "o": 5.0, "h": 5.5, "l": 4.9, "c": 5.25, "v": -1}]},
            {"results": [{"t": 1716163200000, "o": 5.0, "h": 5.5, "l": 4.9, "c": 5.25}]},
            {"results": [{"t": 1716163200000, "o": 5.0, "h": 5.5, "l": 4.9, "c": 5.25, "v": 100000, "vw": -1}]},
            {"results": [{"t": 1716163200000, "o": 5.0, "h": 5.5, "l": 4.9, "c": 5.25, "v": 100000, "n": -1}]},
            {"results": [{"t": -1, "o": 5.0, "h": 5.5, "l": 4.9, "c": 5.25, "v": 100000}]},
        ]
        for payload in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            session.get.return_value = _mock_response(200, payload)
            adapter = self._adapter(session)

            resp = adapter.get_daily_bars("AAPL", "2026-05-01", "2026-05-02")

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "parse"

    def test_get_daily_bars_zero_values_are_allowed(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200,
            {
                "results": [
                    {
                        "t": 0,
                        "o": 0,
                        "h": 0,
                        "l": 0,
                        "c": 0,
                        "v": 0,
                        "vw": 0,
                        "n": 0,
                    }
                ]
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_daily_bars("AAPL", "2026-05-01", "2026-05-02")

        assert resp.ok
        assert resp.data[0].open == 0
        assert resp.data[0].volume == 0
        assert resp.data[0].vwap == 0
        assert resp.data[0].transactions == 0

    def test_get_daily_bars_request_exception_does_not_leak_secret_or_url(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = requests.exceptions.ConnectionError(
            "failed https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2026-05-01/2026-05-02?apiKey=SECRET&token=LEAK"
        )
        adapter = self._adapter(session)

        resp = adapter.get_daily_bars("AAPL", "2026-05-01", "2026-05-02")

        assert not resp.ok
        assert resp.error.error_type == "http"
        assert resp.error.message == "Polygon request failed: ConnectionError"
        assert "SECRET" not in repr(resp)
        assert "apiKey" not in repr(resp)
        assert "token" not in repr(resp)

    def test_get_splits_ok(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {
                    "id": "split-1",
                    "ticker": "AAPL",
                    "execution_date": "2020-08-31",
                    "split_from": 1,
                    "split_to": 4,
                    "adjustment_type": "forward_split",
                    "historical_adjustment_factor": "0.25",
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_splits(
            " aapl ",
            execution_date_from="2020-08-01",
            execution_date_to="2020-09-30",
            limit=6000,
            sort="execution_date",
            order="asc",
        )

        assert resp.ok
        assert len(resp.data) == 1
        split = resp.data[0]
        assert split.id == "split-1"
        assert split.ticker == "AAPL"
        assert split.execution_date == "2020-08-31"
        assert split.split_from == Decimal("1")
        assert split.split_to == Decimal("4")
        assert split.adjustment_type == "forward_split"
        assert split.historical_adjustment_factor == Decimal("0.25")
        assert split.raw == json_data["results"][0]
        assert resp.lineage.endpoint == "/stocks/v1/splits"
        session.get.assert_called_with(
            "https://api.polygon.io/stocks/v1/splits",
            params={
                "ticker": "AAPL",
                "execution_date.gte": "2020-08-01",
                "execution_date.lte": "2020-09-30",
                "limit": 5000,
                "sort": "execution_date.asc",
            },
            timeout=POLYGON_REQUEST_TIMEOUT,
        )

    def test_get_dividends_ok(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {
                    "id": "div-1",
                    "ticker": "MSFT",
                    "cash_amount": "0.333333333333333333",
                    "currency": "USD",
                    "declaration_date": "2026-03-10",
                    "distribution_type": "recurring",
                    "ex_dividend_date": "2026-05-21",
                    "frequency": "4",
                    "historical_adjustment_factor": "0.99908",
                    "pay_date": "2026-06-11",
                    "record_date": "2026-05-21",
                    "split_adjusted_cash_amount": "0.333333333333333333",
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_dividends(
            " msft ",
            ex_dividend_date_from="2026-01-01",
            ex_dividend_date_to="2026-05-28",
            limit=10,
            sort="ex_dividend_date",
            order="desc",
        )

        assert resp.ok
        assert len(resp.data) == 1
        dividend = resp.data[0]
        assert dividend.id == "div-1"
        assert dividend.ticker == "MSFT"
        assert dividend.ex_dividend_date == "2026-05-21"
        assert dividend.cash_amount == Decimal("0.333333333333333333")
        assert dividend.currency == "USD"
        assert dividend.distribution_type == "recurring"
        assert dividend.frequency == 4
        assert dividend.historical_adjustment_factor == Decimal("0.99908")
        assert dividend.split_adjusted_cash_amount == Decimal("0.333333333333333333")
        assert dividend.raw == json_data["results"][0]
        assert resp.lineage.endpoint == "/stocks/v1/dividends"
        session.get.assert_called_with(
            "https://api.polygon.io/stocks/v1/dividends",
            params={
                "ticker": "MSFT",
                "ex_dividend_date.gte": "2026-01-01",
                "ex_dividend_date.lte": "2026-05-28",
                "limit": 10,
                "sort": "ex_dividend_date.desc",
            },
            timeout=POLYGON_REQUEST_TIMEOUT,
        )

    def test_polygon_corporate_action_rejects_invalid_dates(self):
        cases = [
            lambda adapter: adapter.get_splits("AAPL", execution_date="bad"),
            lambda adapter: adapter.get_splits(
                "AAPL",
                execution_date_from="2026-99-99",
                execution_date_to="2026-05-31",
            ),
            lambda adapter: adapter.get_splits(
                "AAPL",
                execution_date_from="2026-05-01",
                execution_date_to="2026/05/31",
            ),
            lambda adapter: adapter.get_splits(
                "AAPL",
                execution_date_from="2026-05-31",
                execution_date_to="2026-05-01",
            ),
            lambda adapter: adapter.get_dividends("MSFT", ex_dividend_date=" "),
            lambda adapter: adapter.get_dividends(
                "MSFT",
                ex_dividend_date_from="2026-99-99",
                ex_dividend_date_to="2026-05-31",
            ),
            lambda adapter: adapter.get_dividends(
                "MSFT",
                ex_dividend_date_from="2026-05-01",
                ex_dividend_date_to="",
            ),
            lambda adapter: adapter.get_dividends(
                "MSFT",
                ex_dividend_date_from="2026-05-31",
                ex_dividend_date_to="2026-05-01",
            ),
        ]

        for call in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            adapter = self._adapter(session)

            resp = call(adapter)

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "validation"
            assert resp.error.retryable is False
            session.get.assert_not_called()

    def test_polygon_corporate_action_valid_exact_dates_request_normally(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = [
            _mock_response(200, {"results": []}),
            _mock_response(200, {"results": []}),
        ]
        adapter = self._adapter(session)

        splits = adapter.get_splits("AAPL", execution_date="2026-05-27")
        dividends = adapter.get_dividends("MSFT", ex_dividend_date="2026-05-27")

        assert splits.ok
        assert dividends.ok
        session.get.assert_any_call(
            "https://api.polygon.io/stocks/v1/splits",
            params={
                "ticker": "AAPL",
                "execution_date": "2026-05-27",
                "limit": 1000,
                "sort": "execution_date.asc",
            },
            timeout=POLYGON_REQUEST_TIMEOUT,
        )
        session.get.assert_any_call(
            "https://api.polygon.io/stocks/v1/dividends",
            params={
                "ticker": "MSFT",
                "ex_dividend_date": "2026-05-27",
                "limit": 1000,
                "sort": "ex_dividend_date.asc",
            },
            timeout=POLYGON_REQUEST_TIMEOUT,
        )

    def test_polygon_splits_skip_invalid_rows(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {},
                {"execution_date": "2026-01-01", "split_from": 1, "split_to": 4},
                {"ticker": "MISSDATE", "split_from": 1, "split_to": 4},
                {"ticker": "MISSFROM", "execution_date": "2026-01-02", "split_to": 4},
                {"ticker": "MISSTO", "execution_date": "2026-01-03", "split_from": 1},
                {"ticker": "ZEROFROM", "execution_date": "2026-01-04", "split_from": 0, "split_to": 4},
                {"ticker": "ZEROTO", "execution_date": "2026-01-05", "split_from": 1, "split_to": 0},
                {"ticker": "NEGFROM", "execution_date": "2026-01-06", "split_from": -1, "split_to": 4},
                {"ticker": "VALID", "execution_date": "2026-01-07", "split_from": "1", "split_to": "1000"},
                None,
                "bad",
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_splits("VALID")

        assert resp.ok
        assert len(resp.data) == 1
        assert resp.data[0].ticker == "VALID"
        assert resp.data[0].split_from == Decimal("1")
        assert resp.data[0].split_to == Decimal("1000")
        assert resp.lineage.data_quality_flags["raw_rows"] == 11
        assert resp.lineage.data_quality_flags["parsed_rows"] == 1
        assert resp.lineage.data_quality_flags["skipped_rows"] == 10

    def test_polygon_dividends_skip_invalid_rows(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {},
                {"ex_dividend_date": "2026-05-21", "cash_amount": "0.91"},
                {"ticker": "MISSDATE", "cash_amount": "0.91"},
                {"ticker": "MISSCASH", "ex_dividend_date": "2026-05-21"},
                {"ticker": "NEG", "ex_dividend_date": "2026-05-21", "cash_amount": "-0.10"},
                {"ticker": "NA", "ex_dividend_date": "2026-05-21", "cash_amount": "N/A"},
                {
                    "ticker": "VALID",
                    "ex_dividend_date": "2026-05-21",
                    "cash_amount": "0.333333333333333333",
                    "frequency": "N/A",
                },
                None,
                7,
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_dividends("VALID")

        assert resp.ok
        assert len(resp.data) == 1
        assert resp.data[0].ticker == "VALID"
        assert resp.data[0].cash_amount == Decimal("0.333333333333333333")
        assert resp.data[0].frequency is None
        assert resp.lineage.data_quality_flags["raw_rows"] == 9
        assert resp.lineage.data_quality_flags["parsed_rows"] == 1
        assert resp.lineage.data_quality_flags["skipped_rows"] == 8

    def test_polygon_dated_feeds_skip_malformed_provider_row_dates(self):
        cases = [
            (
                lambda adapter: adapter.get_splits("AAPL"),
                [
                    {"ticker": "VALID", "execution_date": "2026-05-27", "split_from": 1, "split_to": 2},
                    {"ticker": "BAD", "execution_date": "bad", "split_from": 1, "split_to": 2},
                    {"ticker": "IMPOSSIBLE", "execution_date": "2026-13-45", "split_from": 1, "split_to": 2},
                    {"ticker": "BLANK", "execution_date": "", "split_from": 1, "split_to": 2},
                    {"ticker": "NUMERIC", "execution_date": 20260527, "split_from": 1, "split_to": 2},
                ],
            ),
            (
                lambda adapter: adapter.get_dividends("AAPL"),
                [
                    {"ticker": "VALID", "ex_dividend_date": "2026-05-27", "cash_amount": "0.25"},
                    {"ticker": "BAD", "ex_dividend_date": "bad", "cash_amount": "0.25"},
                    {"ticker": "IMPOSSIBLE", "ex_dividend_date": "2026-13-45", "cash_amount": "0.25"},
                    {"ticker": "BLANK", "ex_dividend_date": "", "cash_amount": "0.25"},
                    {"ticker": "NUMERIC", "ex_dividend_date": 20260527, "cash_amount": "0.25"},
                ],
            ),
            (
                lambda adapter: adapter.get_short_interest("AAPL"),
                [
                    {"ticker": "VALID", "settlement_date": "2026-05-27", "short_interest": "10"},
                    {"ticker": "BAD", "settlement_date": "bad", "short_interest": "10"},
                    {"ticker": "IMPOSSIBLE", "settlement_date": "2026-13-45", "short_interest": "10"},
                    {"ticker": "BLANK", "settlement_date": "", "short_interest": "10"},
                    {"ticker": "NUMERIC", "settlement_date": 20260527, "short_interest": "10"},
                ],
            ),
            (
                lambda adapter: adapter.get_short_volume("AAPL"),
                [
                    {"ticker": "VALID", "date": "2026-05-27", "short_volume": "10"},
                    {"ticker": "BAD", "date": "bad", "short_volume": "10"},
                    {"ticker": "IMPOSSIBLE", "date": "2026-13-45", "short_volume": "10"},
                    {"ticker": "BLANK", "date": "", "short_volume": "10"},
                    {"ticker": "NUMERIC", "date": 20260527, "short_volume": "10"},
                ],
            ),
        ]

        for call, rows in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            session.get.return_value = _mock_response(200, {"results": rows})
            adapter = self._adapter(session)

            resp = call(adapter)

            assert resp.ok
            assert [row.ticker for row in resp.data] == ["VALID"]
            assert resp.lineage.data_quality_flags["raw_rows"] == 5
            assert resp.lineage.data_quality_flags["parsed_rows"] == 1
            assert resp.lineage.data_quality_flags["skipped_rows"] == 4

    def test_polygon_corporate_action_skipped_row_telemetry_counts_rows(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {"ticker": "V1", "execution_date": "2026-01-01", "split_from": 1, "split_to": 2},
                {"ticker": "V2", "execution_date": "2026-01-02", "split_from": 2, "split_to": 1},
                {"ticker": "V3", "execution_date": "2026-01-03", "split_from": "1", "split_to": "1000"},
                {},
                {"ticker": "BAD1", "execution_date": "2026-01-04", "split_from": 0, "split_to": 1},
                {"ticker": "BAD2", "execution_date": "2026-01-05", "split_from": 1, "split_to": 0},
                {"ticker": "BAD3", "split_from": 1, "split_to": 2},
                {"execution_date": "2026-01-06", "split_from": 1, "split_to": 2},
                "bad",
                None,
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_splits("ACME")

        assert resp.ok
        assert [row.ticker for row in resp.data] == ["V1", "V2", "V3"]
        assert resp.lineage.data_quality_flags["raw_rows"] == 10
        assert resp.lineage.data_quality_flags["parsed_rows"] == 3
        assert resp.lineage.data_quality_flags["skipped_rows"] == 7
        assert "all_rows_skipped" not in resp.lineage.data_quality_flags

    def test_polygon_corporate_action_all_invalid_rows_are_flagged(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": [
                {},
                {"ticker": "BAD1", "execution_date": "2026-01-04", "split_from": 0, "split_to": 1},
                {"ticker": "BAD2", "execution_date": "2026-01-05", "split_from": 1, "split_to": 0},
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_splits("ACME")

        assert resp.ok
        assert resp.data == []
        assert resp.lineage.data_quality_flags["raw_rows"] == 3
        assert resp.lineage.data_quality_flags["parsed_rows"] == 0
        assert resp.lineage.data_quality_flags["skipped_rows"] == 3
        assert resp.lineage.data_quality_flags["all_rows_skipped"] is True

    def test_polygon_corporate_action_optional_economic_fields_are_validated(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        split_json = {
            "results": [
                {
                    "ticker": "NEG",
                    "execution_date": "2026-01-01",
                    "split_from": 1,
                    "split_to": 4,
                    "historical_adjustment_factor": "-0.25",
                },
                {
                    "ticker": "ZERO",
                    "execution_date": "2026-01-02",
                    "split_from": 1,
                    "split_to": 4,
                    "historical_adjustment_factor": "0",
                },
                {
                    "ticker": "VALID",
                    "execution_date": "2026-01-03",
                    "split_from": 1,
                    "split_to": 4,
                    "historical_adjustment_factor": "0.25",
                },
            ]
        }
        dividend_json = {
            "results": [
                {
                    "ticker": "NEGFACTOR",
                    "ex_dividend_date": "2026-05-01",
                    "cash_amount": "0.27",
                    "historical_adjustment_factor": "-1",
                    "split_adjusted_cash_amount": "0.27",
                },
                {
                    "ticker": "ZEROFACTOR",
                    "ex_dividend_date": "2026-05-02",
                    "cash_amount": "0.27",
                    "historical_adjustment_factor": "0",
                    "split_adjusted_cash_amount": "0.27",
                },
                {
                    "ticker": "NEGCASH",
                    "ex_dividend_date": "2026-05-03",
                    "cash_amount": "0.27",
                    "historical_adjustment_factor": "0.999",
                    "split_adjusted_cash_amount": "-0.10",
                },
                {
                    "ticker": "VALID",
                    "ex_dividend_date": "2026-05-04",
                    "cash_amount": "0.27",
                    "historical_adjustment_factor": "0.999",
                    "split_adjusted_cash_amount": "0.27",
                },
            ]
        }
        session.get.side_effect = [
            _mock_response(200, split_json),
            _mock_response(200, dividend_json),
        ]
        adapter = self._adapter(session)

        splits = adapter.get_splits("ACME")
        dividends = adapter.get_dividends("ACME")

        assert splits.ok
        assert [row.historical_adjustment_factor for row in splits.data] == [
            None,
            None,
            Decimal("0.25"),
        ]
        assert dividends.ok
        assert dividends.data[0].historical_adjustment_factor is None
        assert dividends.data[0].split_adjusted_cash_amount == Decimal("0.27")
        assert dividends.data[1].historical_adjustment_factor is None
        assert dividends.data[1].split_adjusted_cash_amount == Decimal("0.27")
        assert dividends.data[2].historical_adjustment_factor == Decimal("0.999")
        assert dividends.data[2].split_adjusted_cash_amount is None
        assert dividends.data[3].historical_adjustment_factor == Decimal("0.999")
        assert dividends.data[3].split_adjusted_cash_amount == Decimal("0.27")

    def test_polygon_corporate_actions_empty_payload_success(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = [
            _mock_response(200, {"results": []}),
            _mock_response(200, {"results": []}),
        ]
        adapter = self._adapter(session)

        splits = adapter.get_splits("AAPL")
        dividends = adapter.get_dividends("AAPL")

        assert splits.ok
        assert splits.data == []
        assert splits.lineage.data_quality_flags["raw_rows"] == 0
        assert splits.lineage.data_quality_flags["parsed_rows"] == 0
        assert splits.lineage.data_quality_flags["skipped_rows"] == 0
        assert "all_rows_skipped" not in splits.lineage.data_quality_flags
        assert dividends.ok
        assert dividends.data == []
        assert dividends.lineage.data_quality_flags["raw_rows"] == 0
        assert dividends.lineage.data_quality_flags["parsed_rows"] == 0
        assert dividends.lineage.data_quality_flags["skipped_rows"] == 0
        assert "all_rows_skipped" not in dividends.lineage.data_quality_flags

    def test_polygon_corporate_action_first_page_invalid_shape_is_parse_error(self):
        cases = [
            {"unexpected": True},
            {"results": {"not": "a list"}},
            [{"ticker": "AAPL"}],
        ]
        for payload in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            session.get.return_value = _mock_response(200, payload)
            adapter = self._adapter(session)

            resp = adapter.get_splits("AAPL")

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "parse"
            assert resp.error.retryable is False
            assert resp.lineage.endpoint == "/stocks/v1/splits"
            assert resp.lineage.data_quality_flags["page_count"] == 1
            assert resp.lineage.data_quality_flags["paginated"] is False
            assert resp.lineage.data_quality_flags["truncated"] is False

    def test_polygon_corporate_action_blank_ticker_is_omitted(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(200, {"results": []})
        adapter = self._adapter(session)

        resp = adapter.get_splits(
            " ",
            execution_date_from="2020-08-01",
            execution_date_to="2020-09-30",
        )

        assert resp.ok
        params = session.get.call_args.kwargs["params"]
        assert "ticker" not in params
        assert params["execution_date.gte"] == "2020-08-01"
        assert params["execution_date.lte"] == "2020-09-30"

    def test_polygon_corporate_action_blank_ticker_without_dates_is_validation_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        adapter = self._adapter(session)

        splits = adapter.get_splits(" ")
        dividends = adapter.get_dividends(" ")

        assert not splits.ok
        assert splits.error.error_type == "validation"
        assert splits.error.retryable is False
        assert splits.lineage.endpoint == "/stocks/v1/splits"
        assert not dividends.ok
        assert dividends.error.error_type == "validation"
        assert dividends.error.retryable is False
        assert dividends.lineage.endpoint == "/stocks/v1/dividends"
        session.get.assert_not_called()

    def test_polygon_corporate_action_bounded_broad_query_is_allowed(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = [
            _mock_response(200, {"results": []}),
            _mock_response(200, {"results": []}),
            _mock_response(200, {"results": []}),
        ]
        adapter = self._adapter(session)

        splits_none = adapter.get_splits(
            None,
            execution_date_from="2020-08-01",
            execution_date_to="2020-09-30",
        )
        dividends_none = adapter.get_dividends(
            None,
            ex_dividend_date_from="2026-01-01",
            ex_dividend_date_to="2026-05-28",
        )
        dividends_blank = adapter.get_dividends(
            " ",
            ex_dividend_date_from="2026-01-01",
            ex_dividend_date_to="2026-05-28",
        )

        assert splits_none.ok
        assert dividends_none.ok
        assert dividends_blank.ok
        split_params = session.get.call_args_list[0].kwargs["params"]
        dividend_params = session.get.call_args_list[1].kwargs["params"]
        blank_dividend_params = session.get.call_args_list[2].kwargs["params"]
        assert "ticker" not in split_params
        assert split_params["execution_date.gte"] == "2020-08-01"
        assert split_params["execution_date.lte"] == "2020-09-30"
        assert "ticker" not in dividend_params
        assert dividend_params["ex_dividend_date.gte"] == "2026-01-01"
        assert dividend_params["ex_dividend_date.lte"] == "2026-05-28"
        assert "ticker" not in blank_dividend_params

    def test_polygon_corporate_action_parse_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(200, text="not json")
        adapter = self._adapter(session)

        resp = adapter.get_splits("AAPL")

        assert not resp.ok
        assert resp.error.error_type == "parse"
        assert resp.lineage.endpoint == "/stocks/v1/splits"

    def test_polygon_corporate_action_auth_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(403, text="Forbidden")
        adapter = self._adapter(session)

        resp = adapter.get_dividends("AAPL")

        assert not resp.ok
        assert resp.error.error_type == "auth"
        assert resp.error.status_code == 403

    def test_polygon_corporate_action_rate_limit_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(429, text="Too many requests")
        adapter = self._adapter(session)

        resp = adapter.get_splits("AAPL")

        assert not resp.ok
        assert resp.error.error_type == "rate_limit"
        assert resp.error.retryable is True

    def test_polygon_corporate_action_provider_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(500, text="Internal Server Error")
        adapter = self._adapter(session)

        resp = adapter.get_dividends("AAPL")

        assert not resp.ok
        assert resp.error.error_type == "http"
        assert resp.error.status_code == 500
        assert resp.error.retryable is True

    def test_polygon_corporate_action_timeout_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = requests.exceptions.Timeout("timed out")
        adapter = self._adapter(session)

        resp = adapter.get_splits("AAPL")

        assert not resp.ok
        assert resp.error.error_type == "timeout"
        assert resp.error.retryable is True

    def test_polygon_request_exception_does_not_leak_secret_or_url(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = requests.exceptions.ConnectionError(
            "failed https://api.polygon.io/stocks/v1/dividends?apiKey=test-polygon-key&ticker=AAPL"
        )
        adapter = self._adapter(session)

        resp = adapter.get_dividends("AAPL")

        assert not resp.ok
        assert resp.error.error_type == "http"
        assert resp.error.message == "Polygon request failed: ConnectionError"
        assert "test-polygon-key" not in resp.error.message
        assert "apiKey" not in resp.error.message
        assert "https://api.polygon.io" not in resp.error.message

    def test_polygon_corporate_action_paginates_and_sanitizes_next_url(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        page_1 = {
            "results": [
                {
                    "id": "div-1",
                    "ticker": "AAPL",
                    "ex_dividend_date": "2026-05-11",
                    "cash_amount": 0.27,
                }
            ],
            "next_url": "https://api.polygon.io/stocks/v1/dividends?cursor=abc&apiKey=SECRET",
        }
        page_2 = {
            "results": [
                {
                    "id": "div-2",
                    "ticker": "AAPL",
                    "ex_dividend_date": "2026-02-09",
                    "cash_amount": 0.26,
                }
            ]
        }
        session.get.side_effect = [
            _mock_response(200, page_1),
            _mock_response(200, page_2),
        ]
        adapter = self._adapter(session)

        resp = adapter.get_dividends("AAPL", limit=1)

        assert resp.ok
        assert [row.id for row in resp.data] == ["div-1", "div-2"]
        assert session.get.call_count == 2
        assert session.params == {"apiKey": "test-polygon-key"}
        assert session.get.call_args_list[0].args[0] == "https://api.polygon.io/stocks/v1/dividends"
        assert session.get.call_args_list[1].args[0] == "https://api.polygon.io/stocks/v1/dividends"
        assert session.get.call_args_list[1].kwargs["params"] == {"cursor": "abc"}
        assert "SECRET" not in session.get.call_args_list[1].args[0]
        assert "SECRET" not in repr(session.get.call_args_list[1].kwargs["params"])
        assert "apiKey" not in session.get.call_args_list[1].args[0]
        assert "apiKey" not in session.get.call_args_list[1].kwargs["params"]
        assert resp.lineage.data_quality_flags["page_count"] == 2
        assert resp.lineage.data_quality_flags["paginated"] is True
        assert resp.lineage.data_quality_flags["truncated"] is False
        assert resp.lineage.data_quality_flags["next_url_paths"] == ["/stocks/v1/dividends"]
        assert resp.lineage.data_quality_flags["raw_rows"] == 2
        assert resp.lineage.data_quality_flags["parsed_rows"] == 2
        assert resp.lineage.data_quality_flags["skipped_rows"] == 0
        assert "SECRET" not in repr(resp.lineage)
        assert "apiKey" not in repr(resp.lineage)

    def test_polygon_corporate_action_relative_next_url_is_normalized(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        page_1 = {
            "results": [
                {
                    "id": "split-1",
                    "ticker": "AAPL",
                    "execution_date": "2020-08-31",
                    "split_from": 1,
                    "split_to": 4,
                }
            ],
            "next_url": "/stocks/v1/splits?cursor=abc&apiKey=SECRET",
        }
        page_2 = {
            "results": [
                {
                    "id": "split-2",
                    "ticker": "MSFT",
                    "execution_date": "2021-01-01",
                    "split_from": 1,
                    "split_to": 2,
                }
            ]
        }
        session.get.side_effect = [
            _mock_response(200, page_1),
            _mock_response(200, page_2),
        ]
        adapter = self._adapter(session)

        resp = adapter.get_splits("AAPL", limit=1)

        assert resp.ok
        assert [row.id for row in resp.data] == ["split-1", "split-2"]
        assert session.get.call_count == 2
        assert session.get.call_args_list[1].args[0] == "https://api.polygon.io/stocks/v1/splits"
        assert session.get.call_args_list[1].kwargs["params"] == {"cursor": "abc"}
        assert resp.lineage.data_quality_flags["next_url_paths"] == ["/stocks/v1/splits"]
        assert "SECRET" not in repr(resp.lineage)
        assert "apiKey" not in repr(resp.lineage)

    def test_polygon_corporate_action_massive_next_url_host_is_allowed(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        page_1 = {
            "results": [
                {
                    "id": "split-1",
                    "ticker": "AAPL",
                    "execution_date": "2020-08-31",
                    "split_from": 1,
                    "split_to": 4,
                }
            ],
            "next_url": "https://api.massive.com/stocks/v1/splits?cursor=abc",
        }
        page_2 = {
            "results": [
                {
                    "id": "split-2",
                    "ticker": "AAPL",
                    "execution_date": "2020-09-01",
                    "split_from": 1,
                    "split_to": 2,
                }
            ]
        }
        session.get.side_effect = [
            _mock_response(200, page_1),
            _mock_response(200, page_2),
        ]
        adapter = self._adapter(session)

        resp = adapter.get_splits("AAPL", limit=1)

        assert resp.ok
        assert [row.id for row in resp.data] == ["split-1", "split-2"]
        assert session.get.call_args_list[1].args[0] == "https://api.polygon.io/stocks/v1/splits"
        assert session.get.call_args_list[1].kwargs["params"] == {"cursor": "abc"}
        assert "api.massive.com" not in repr(resp.lineage)

    def test_polygon_corporate_action_second_page_invalid_shape_is_parse_error(self):
        page_1 = {
            "results": [
                {
                    "id": "div-1",
                    "ticker": "AAPL",
                    "ex_dividend_date": "2026-05-11",
                    "cash_amount": 0.27,
                }
            ],
            "next_url": "/stocks/v1/dividends?cursor=abc",
        }
        cases = [
            {"unexpected": True},
            {"results": {"not": "a list"}},
            [{"ticker": "AAPL"}],
        ]
        for payload in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            session.get.side_effect = [
                _mock_response(200, page_1),
                _mock_response(200, payload),
            ]
            adapter = self._adapter(session)

            resp = adapter.get_dividends("AAPL", limit=1)

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "parse"
            assert resp.error.retryable is False
            assert session.get.call_count == 2
            assert resp.lineage.endpoint == "/stocks/v1/dividends"
            assert resp.lineage.data_quality_flags["page_count"] == 2
            assert resp.lineage.data_quality_flags["paginated"] is True
            assert resp.lineage.data_quality_flags["truncated"] is True
            assert resp.lineage.data_quality_flags["next_url_paths"] == ["/stocks/v1/dividends"]

    def test_polygon_corporate_action_empty_second_page_preserves_page_one_data(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        page_1 = {
            "results": [
                {
                    "id": "div-1",
                    "ticker": "AAPL",
                    "ex_dividend_date": "2026-05-11",
                    "cash_amount": 0.27,
                }
            ],
            "next_url": "/stocks/v1/dividends?cursor=abc",
        }
        page_2 = {"results": []}
        session.get.side_effect = [
            _mock_response(200, page_1),
            _mock_response(200, page_2),
        ]
        adapter = self._adapter(session)

        resp = adapter.get_dividends("AAPL", limit=1)

        assert resp.ok
        assert [row.id for row in resp.data] == ["div-1"]
        assert resp.lineage.data_quality_flags["page_count"] == 2
        assert resp.lineage.data_quality_flags["paginated"] is True
        assert resp.lineage.data_quality_flags["truncated"] is False
        assert resp.lineage.data_quality_flags["next_url_paths"] == ["/stocks/v1/dividends"]
        assert resp.lineage.data_quality_flags["raw_rows"] == 1
        assert resp.lineage.data_quality_flags["parsed_rows"] == 1
        assert resp.lineage.data_quality_flags["skipped_rows"] == 0

    def test_polygon_corporate_action_second_page_rate_limit_has_pagination_flags(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        page_1 = {
            "results": [
                {
                    "id": "div-1",
                    "ticker": "AAPL",
                    "ex_dividend_date": "2026-05-11",
                    "cash_amount": 0.27,
                }
            ],
            "next_url": "/stocks/v1/dividends?cursor=abc&apiKey=SECRET",
        }
        session.get.side_effect = [
            _mock_response(200, page_1),
            _mock_response(429, text="Too many requests"),
        ]
        adapter = self._adapter(session)

        resp = adapter.get_dividends("AAPL", limit=1)

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "rate_limit"
        assert resp.lineage.data_quality_flags["page_count"] == 2
        assert resp.lineage.data_quality_flags["paginated"] is True
        assert resp.lineage.data_quality_flags["truncated"] is True
        assert resp.lineage.data_quality_flags["next_url_paths"] == ["/stocks/v1/dividends"]
        assert "SECRET" not in repr(resp.lineage)
        assert "apiKey" not in repr(resp.lineage)

    def test_polygon_corporate_action_evil_next_url_is_rejected_before_request(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        page_1 = {
            "results": [
                {
                    "id": "div-1",
                    "ticker": "AAPL",
                    "ex_dividend_date": "2026-05-11",
                    "cash_amount": 0.27,
                }
            ],
            "next_url": "https://evil.example/stocks/v1/dividends?cursor=abc&apiKey=SECRET",
        }
        session.get.return_value = _mock_response(200, page_1)
        adapter = self._adapter(session)

        resp = adapter.get_dividends("AAPL", limit=1)

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "pagination"
        assert resp.error.retryable is False
        assert session.get.call_count == 1
        assert "evil.example" not in repr(resp.lineage)
        assert "SECRET" not in repr(resp.lineage)
        assert "apiKey" not in repr(resp.lineage)

    def test_polygon_corporate_action_pagination_cap_fails_loud(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        page_1 = {
            "results": [
                {
                    "id": "split-1",
                    "ticker": "AAPL",
                    "execution_date": "2020-08-31",
                    "split_from": 1,
                    "split_to": 4,
                }
            ],
            "next_url": "/stocks/v1/splits?cursor=abc&apiKey=SECRET",
        }
        session.get.return_value = _mock_response(200, page_1)
        adapter = self._adapter(session)

        resp = adapter.get_splits("AAPL", max_pages=1)

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "pagination"
        assert resp.lineage.data_quality_flags["truncated"] is True
        assert resp.lineage.data_quality_flags["next_url_paths"] == ["/stocks/v1/splits"]
        assert "SECRET" not in repr(resp.lineage)
        assert "apiKey" not in repr(resp.lineage)

    def test_polygon_corporate_action_invalid_max_pages_is_validation_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        adapter = self._adapter(session)

        bad = adapter.get_splits("AAPL", max_pages="bad")  # type: ignore[arg-type]
        zero = adapter.get_dividends("AAPL", max_pages=0)
        negative = adapter.get_splits("AAPL", max_pages=-1)

        for resp in (bad, zero, negative):
            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "validation"
            assert resp.error.retryable is False
        session.get.assert_not_called()

    def test_polygon_corporate_action_lineage_hash_stability(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        split_json = {
            "results": [
                {
                    "id": "split-1",
                    "ticker": "AAPL",
                    "execution_date": "2020-08-31",
                    "split_from": 1,
                    "split_to": 4,
                }
            ]
        }
        dividend_json = {
            "results": [
                {
                    "id": "div-1",
                    "ticker": "AAPL",
                    "cash_amount": 0.27,
                    "ex_dividend_date": "2026-05-11",
                    "frequency": 4,
                }
            ]
        }
        adapter = self._adapter(session)

        session.get.return_value = _mock_response(200, split_json)
        split_1 = adapter.get_splits("AAPL")
        split_2 = adapter.get_splits("AAPL")
        session.get.return_value = _mock_response(200, dividend_json)
        dividend_1 = adapter.get_dividends("AAPL")
        dividend_2 = adapter.get_dividends("AAPL")

        assert split_1.lineage.raw_payload_hash == split_2.lineage.raw_payload_hash
        assert dividend_1.lineage.raw_payload_hash == dividend_2.lineage.raw_payload_hash

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
        short_interest_json = {
            "results": [
                {
                    "ticker": "ACME",
                    "settlement_date": "2026-05-15",
                    "short_interest": 50000,
                }
            ]
        }
        short_volume_json = {
            "results": [
                {
                    "ticker": "ACME",
                    "date": "2026-05-15",
                    "short_volume": "100",
                    "total_volume": "250",
                    "short_volume_ratio": "40",
                }
            ]
        }
        adapter = self._adapter(session)

        session.get.return_value = _mock_response(200, short_interest_json)
        resp1 = adapter.get_short_interest("ACME")
        resp2 = adapter.get_short_interest("ACME")
        session.get.return_value = _mock_response(200, short_volume_json)
        resp3 = adapter.get_short_volume("ACME")
        resp4 = adapter.get_short_volume("ACME")
        assert resp1.lineage.raw_payload_hash == resp2.lineage.raw_payload_hash
        assert resp3.lineage.raw_payload_hash == resp4.lineage.raw_payload_hash

    def test_legacy_polygon_lineage_hash_stability_and_changes(self):
        tickers_page_1 = {
            "results": [{"ticker": "AAPL", "name": "Apple Inc.", "cik": "320193"}],
            "next_url": "/v3/reference/tickers?cursor=abc",
        }
        tickers_page_2 = {
            "results": [{"ticker": "MSFT", "name": "Microsoft Corp.", "cik": "789019"}]
        }
        tickers_page_2_changed = {
            "results": [{"ticker": "MSFT", "name": "Microsoft Corporation", "cik": "789019"}]
        }
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = [
            _mock_response(200, tickers_page_1),
            _mock_response(200, tickers_page_2),
            _mock_response(200, tickers_page_1),
            _mock_response(200, tickers_page_2),
            _mock_response(200, tickers_page_1),
            _mock_response(200, tickers_page_2_changed),
        ]
        adapter = self._adapter(session)

        tickers_1 = adapter.get_tickers(limit=1)
        tickers_2 = adapter.get_tickers(limit=1)
        tickers_3 = adapter.get_tickers(limit=1)

        assert tickers_1.ok
        assert tickers_2.ok
        assert tickers_3.ok
        assert tickers_1.lineage.raw_payload_hash == tickers_2.lineage.raw_payload_hash
        assert tickers_1.lineage.raw_payload_hash != tickers_3.lineage.raw_payload_hash

        details_payload = {"results": {"ticker": "AAPL", "name": "Apple Inc."}}
        details_changed = {"results": {"ticker": "AAPL", "name": "Apple Incorporated"}}
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = [
            _mock_response(200, details_payload),
            _mock_response(200, details_payload),
            _mock_response(200, details_changed),
        ]
        adapter = self._adapter(session)

        details_1 = adapter.get_ticker_details("AAPL")
        details_2 = adapter.get_ticker_details("AAPL")
        details_3 = adapter.get_ticker_details("AAPL")

        assert details_1.ok
        assert details_2.ok
        assert details_3.ok
        assert details_1.lineage.raw_payload_hash == details_2.lineage.raw_payload_hash
        assert details_1.lineage.raw_payload_hash != details_3.lineage.raw_payload_hash

        bars_payload = {
            "results": [
                {"t": 1716163200000, "o": 5.0, "h": 5.5, "l": 4.9, "c": 5.25, "v": 100000}
            ]
        }
        bars_changed = {
            "results": [
                {"t": 1716163200000, "o": 5.0, "h": 5.5, "l": 4.9, "c": 5.30, "v": 100000}
            ]
        }
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = [
            _mock_response(200, bars_payload),
            _mock_response(200, bars_payload),
            _mock_response(200, bars_changed),
        ]
        adapter = self._adapter(session)

        bars_1 = adapter.get_daily_bars("AAPL", "2026-05-01", "2026-05-02")
        bars_2 = adapter.get_daily_bars("AAPL", "2026-05-01", "2026-05-02")
        bars_3 = adapter.get_daily_bars("AAPL", "2026-05-01", "2026-05-02")

        assert bars_1.ok
        assert bars_2.ok
        assert bars_3.ok
        assert bars_1.lineage.raw_payload_hash == bars_2.lineage.raw_payload_hash
        assert bars_1.lineage.raw_payload_hash != bars_3.lineage.raw_payload_hash

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
            "next_url": "https://api.polygon.io/v3/reference/tickers?cursor=abc&apiKey=SECRET&token=LEAK",
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
        assert resp.data[0].raw_payload["next_url"] == "/v3/reference/tickers"
        assert "SECRET" not in repr(resp)
        assert "apiKey" not in repr(resp)
        assert "token" not in repr(resp)
        assert session.get.call_count == 2
        assert session.get.call_args_list[0].kwargs["params"]["limit"] == 1000
        assert session.get.call_args_list[1].args[0] == "https://api.polygon.io/v3/reference/tickers"
        assert session.get.call_args_list[1].kwargs["params"] == {"cursor": "abc"}
        assert "SECRET" not in repr(session.get.call_args_list[1])
        assert "apiKey" not in repr(session.get.call_args_list[1])
        flags = resp.lineage.data_quality_flags
        assert flags["page_count"] == 2
        assert flags["paginated"] is True
        assert flags["truncated"] is False
        assert flags["next_url_paths"] == ["/v3/reference/tickers"]
        assert flags["raw_rows"] == 2
        assert flags["parsed_rows"] == 2
        assert flags["duplicate_same_identity_rows"] == 0
        assert flags["duplicate_conflict_rows"] == 0

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

    def test_get_tickers_rejects_unsafe_next_urls(self):
        bad_next_urls = [
            "https://evil.example/v3/reference/tickers?cursor=abc&apiKey=SECRET",
            "http://api.polygon.io/v3/reference/tickers?cursor=abc",
            "https://api.polygon.io.evil.example/v3/reference/tickers?cursor=abc",
            "https://api.polygon.io/v3/reference/tickers/other?cursor=abc",
            "https://api.polygon.io/v3/reference/tickers?apiKey=SECRET",
        ]
        for next_url in bad_next_urls:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            session.get.return_value = _mock_response(
                200,
                {
                    "results": [{"ticker": "AAPL", "name": "Apple Inc."}],
                    "next_url": next_url,
                },
            )
            adapter = self._adapter(session)

            resp = adapter.get_tickers(limit=1)

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "pagination"
            assert resp.error.retryable is False
            assert session.get.call_count == 1
            assert "SECRET" not in repr(resp)
            assert "apiKey" not in repr(resp)
            assert "evil.example" not in repr(resp)

    def test_get_tickers_pagination_cap_fails_loud(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200,
            {
                "results": [{"ticker": "AAPL", "name": "Apple Inc."}],
                "next_url": "/v3/reference/tickers?cursor=abc&apiKey=SECRET",
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_tickers(limit=1, max_pages=1)

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "pagination"
        flags = resp.lineage.data_quality_flags
        assert flags["page_count"] == 1
        assert flags["paginated"] is True
        assert flags["truncated"] is True
        assert flags["next_url_paths"] == ["/v3/reference/tickers"]
        assert "SECRET" not in repr(resp)
        assert "apiKey" not in repr(resp)

    def test_get_tickers_unexpected_payload_shapes_are_parse_errors(self):
        cases = [
            {"unexpected": True},
            {"results": {"not": "list"}},
            [],
            {"results": [None]},
        ]
        for payload in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            session.get.return_value = _mock_response(200, payload)
            adapter = self._adapter(session)

            resp = adapter.get_tickers()

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "parse"

    def test_get_tickers_duplicate_identity_telemetry_and_conflict(self):
        same_identity = {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "active": True,
            "cik": "320193",
            "primary_exchange": "XNAS",
            "type": "CS",
            "list_date": "1980-12-12",
        }
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = [
            _mock_response(
                200,
                {
                    "results": [same_identity],
                    "next_url": "/v3/reference/tickers?cursor=abc",
                },
            ),
            _mock_response(200, {"results": [same_identity]}),
        ]
        adapter = self._adapter(session)

        same_resp = adapter.get_tickers(limit=1)

        assert same_resp.ok
        assert same_resp.lineage.data_quality_flags["duplicate_same_identity_rows"] == 1
        assert same_resp.lineage.data_quality_flags["duplicate_conflict_rows"] == 0

        conflict = dict(same_identity)
        conflict["cik"] = "1326801"
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = [
            _mock_response(
                200,
                {
                    "results": [same_identity],
                    "next_url": "/v3/reference/tickers?cursor=abc",
                },
            ),
            _mock_response(200, {"results": [conflict]}),
        ]
        adapter = self._adapter(session)

        conflict_resp = adapter.get_tickers(limit=1)

        assert not conflict_resp.ok
        assert conflict_resp.data is None
        assert conflict_resp.error.error_type == "identity_conflict"
        assert conflict_resp.lineage.data_quality_flags["duplicate_conflict_rows"] == 1

    def test_get_tickers_request_exception_does_not_leak_secret_or_url(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = requests.exceptions.ConnectionError(
            "failed https://api.polygon.io/v3/reference/tickers?apiKey=SECRET&token=LEAK"
        )
        adapter = self._adapter(session)

        resp = adapter.get_tickers()

        assert not resp.ok
        assert resp.error.error_type == "http"
        assert resp.error.message == "Polygon request failed: ConnectionError"
        assert "SECRET" not in repr(resp)
        assert "apiKey" not in repr(resp)
        assert "token" not in repr(resp)


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
        session.get.assert_called_once_with(
            "https://api.polygon.io/v3/reference/tickers/AAPL",
            params={},
            timeout=POLYGON_REQUEST_TIMEOUT,
        )

    def test_get_ticker_details_figi_in_cik_field_does_not_fabricate_cik(self):
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
                "cik": "BBG000B9XB24",
                "composite_figi": "BBG000B9XRY4",
                "share_class_figi": "BBG001S5N8V8",
            }
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_ticker_details("AAPL")

        assert resp.ok
        assert resp.data.cik is None
        assert resp.data.composite_figi == "BBG000B9XRY4"
        assert resp.data.share_class_figi == "BBG001S5N8V8"

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

    def test_get_ticker_details_rejects_invalid_ticker_and_date_without_request(self):
        invalid_calls = [
            lambda adapter: adapter.get_ticker_details(""),
            lambda adapter: adapter.get_ticker_details(" "),
            lambda adapter: adapter.get_ticker_details("AAPL/../../x"),
            lambda adapter: adapter.get_ticker_details("AAPL?x=1"),
            lambda adapter: adapter.get_ticker_details(".AAPL"),
            lambda adapter: adapter.get_ticker_details("AAPL."),
            lambda adapter: adapter.get_ticker_details("A..PL"),
            lambda adapter: adapter.get_ticker_details("AAPL", date_str="bad"),
            lambda adapter: adapter.get_ticker_details("AAPL", date_str="2026-13-45"),
        ]
        for call in invalid_calls:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            adapter = self._adapter(session)

            resp = call(adapter)

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "validation"
            session.get.assert_not_called()

    def test_get_ticker_details_bad_shapes_are_parse_errors(self):
        cases = [
            {"unexpected": True},
            {"results": []},
            {"results": "bad"},
            {"results": {"not": "list"}},
            {"results": {"name": " "}},
            [],
        ]
        for payload in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            session.get.return_value = _mock_response(200, payload)
            adapter = self._adapter(session)

            resp = adapter.get_ticker_details("AAPL")

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "parse"

    def test_get_ticker_details_rejects_provider_ticker_mismatch(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200,
            {
                "results": {
                    "ticker": "MSFT",
                    "name": "Microsoft",
                    "active": True,
                }
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_ticker_details("AAPL")

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "parse"

    def test_get_ticker_details_request_exception_does_not_leak_secret_or_url(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = requests.exceptions.ConnectionError(
            "failed https://api.polygon.io/v3/reference/tickers/AAPL?apiKey=SECRET&token=LEAK"
        )
        adapter = self._adapter(session)

        resp = adapter.get_ticker_details("AAPL")

        assert not resp.ok
        assert resp.error.error_type == "http"
        assert resp.error.message == "Polygon request failed: ConnectionError"
        assert "SECRET" not in repr(resp)
        assert "apiKey" not in repr(resp)
        assert "token" not in repr(resp)

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
        assert ev.identity_continuity_status == "unproven"
        assert ev.raw_event == json_data["results"]["events"][0]
        assert resp.lineage.data_quality_flags["identifier_queried"] == "META"
        assert resp.lineage.data_quality_flags["raw_rows"] == 1
        assert resp.lineage.data_quality_flags["parsed_rows"] == 1
        assert resp.lineage.data_quality_flags["skipped_rows"] == 0
        assert resp.lineage.data_quality_flags["identity_continuity_unproven_rows"] == 1
        session.get.assert_called_with(
            "https://api.polygon.io/vX/reference/tickers/META/events",
            params={"types": "ticker_change"},
            timeout=POLYGON_REQUEST_TIMEOUT,
        )

    def test_get_ticker_events_rejects_invalid_identifiers_without_request(self):
        cases = [
            "",
            " ",
            "META,FB",
            "META/FB",
            "META?x=1",
            "META#frag",
            "META&x=1",
            "META%2FFB",
            "META%3FFB",
            "META%26FB",
            "META\\FB",
            "META:FB",
            "META FB",
            ".",
            "..",
            "-",
            "_",
            ".META",
            "META.",
            "-META",
            "META-",
            "_META",
            "META_",
            "A..B",
            "X" * 300,
            123,
        ]
        for identifier in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            adapter = self._adapter(session)

            resp = adapter.get_ticker_events(identifier)  # type: ignore[arg-type]

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "validation"
            assert resp.error.retryable is False
            session.get.assert_not_called()

    def test_get_ticker_events_accepts_path_safe_identifiers(self):
        cases = {
            " meta ": "META",
            "BRK.B": "BRK.B",
            "BRK-B": "BRK-B",
            "BBG000B9XRY4": "BBG000B9XRY4",
            "123456789": "123456789",
        }
        for identifier, normalized in cases.items():
            session = MagicMock(spec=requests.Session)
            session.params = {}
            session.get.return_value = _mock_response(200, {"results": {"events": []}})
            adapter = self._adapter(session)

            resp = adapter.get_ticker_events(identifier)

            assert resp.ok
            session.get.assert_called_once_with(
                f"https://api.polygon.io/vX/reference/tickers/{normalized}/events",
                params={"types": "ticker_change"},
                timeout=POLYGON_REQUEST_TIMEOUT,
            )
            assert resp.lineage.data_quality_flags["identifier_queried"] == normalized

    def test_get_ticker_events_validates_event_types_without_injection(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(200, {"results": {"events": []}})
        adapter = self._adapter(session)

        default_resp = adapter.get_ticker_events(" meta ")
        list_resp = adapter.get_ticker_events("meta", types=[" ticker_change ", "ticker_change"])

        assert default_resp.ok
        assert list_resp.ok
        assert session.get.call_args_list[0].args[0] == (
            "https://api.polygon.io/vX/reference/tickers/META/events"
        )
        assert session.get.call_args_list[0].kwargs["params"] == {"types": "ticker_change"}
        assert session.get.call_args_list[1].kwargs["params"] == {"types": "ticker_change"}

        bad_cases = [
            ["ticker_change", 123],
            [" "],
            "ticker_change,delisting",
            "ticker_change&x=1",
            "name_change",
        ]
        for types in bad_cases:
            bad_session = MagicMock(spec=requests.Session)
            bad_session.params = {}
            bad_adapter = self._adapter(bad_session)

            resp = bad_adapter.get_ticker_events("META", types=types)  # type: ignore[arg-type]

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "validation"
            assert resp.error.retryable is False
            bad_session.get.assert_not_called()

    def test_get_ticker_events_no_events(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {"results": {"name": "Acme", "events": []}}
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)
        resp = adapter.get_ticker_events("ACME")

        assert resp.ok
        assert resp.data == []
        assert resp.lineage.data_quality_flags["identifier_queried"] == "ACME"
        assert resp.lineage.data_quality_flags["raw_rows"] == 0
        assert resp.lineage.data_quality_flags["parsed_rows"] == 0
        assert resp.lineage.data_quality_flags["skipped_rows"] == 0
        assert resp.lineage.data_quality_flags["event_types_present"] == []

    def test_get_ticker_events_invalid_shape_is_parse_error(self):
        cases = [
            {"unexpected": True},
            {"results": {"events": {"not": "list"}}},
            {"results": {}},
            [{"events": []}],
        ]
        for payload in cases:
            session = MagicMock(spec=requests.Session)
            session.params = {}
            session.get.return_value = _mock_response(200, payload)
            adapter = self._adapter(session)

            resp = adapter.get_ticker_events("META")

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "parse"
            assert resp.error.retryable is False
            assert resp.lineage.endpoint == "/vX/reference/tickers/META/events"
            assert resp.lineage.data_quality_flags["identifier_queried"] == "META"

    def test_get_ticker_events_malformed_json_is_parse_error(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(200, text="not json")
        adapter = self._adapter(session)

        resp = adapter.get_ticker_events("META")

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "parse"

    def test_get_ticker_events_skips_invalid_rows_and_tracks_continuity(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": {
                "events": [
                    None,
                    "bad",
                    {},
                    {
                        "type": "ticker_change",
                        "date": "2022-06-09",
                        "ticker_change": {"ticker": "FB", "new_ticker": "META"},
                    },
                    {
                        "type": "ticker_change",
                        "date": "2022-06-10",
                        "ticker_change": {
                            "ticker": "OLD",
                            "new_ticker": "NEW",
                            "old_cik": "0001234567",
                            "new_cik": "1234567",
                        },
                    },
                    {
                        "type": "ticker_change",
                        "date": "2022-06-11",
                        "ticker_change": {
                            "ticker": "OLD2",
                            "new_ticker": "NEW2",
                            "old_cik": "0001234567",
                            "new_cik": "0007654321",
                        },
                    },
                    {
                        "type": "name_change",
                        "date": "2022-06-12",
                        "name": "New Name",
                    },
                ]
            }
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_ticker_events("META")

        assert resp.ok
        assert [row.event_type for row in resp.data] == [
            "ticker_change",
            "ticker_change",
            "ticker_change",
        ]
        assert [row.identity_continuity_status for row in resp.data] == [
            "unproven",
            "proved",
            "mismatch",
        ]
        assert resp.data[0].old_ticker == "FB"
        assert resp.data[0].new_ticker == "META"
        assert resp.data[1].old_cik == "0001234567"
        assert resp.data[1].new_cik == "0001234567"
        assert resp.data[2].new_cik == "0007654321"
        flags = resp.lineage.data_quality_flags
        assert flags["raw_rows"] == 7
        assert flags["parsed_rows"] == 3
        assert flags["skipped_rows"] == 4
        assert flags["identity_continuity_unproven_rows"] == 1
        assert flags["identity_continuity_proved_rows"] == 1
        assert flags["identity_continuity_mismatch_rows"] == 1
        assert flags["identity_continuity_not_applicable_rows"] == 0
        assert flags["event_types_present"] == ["ticker_change"]

    def test_get_ticker_events_figi_in_cik_field_does_not_fabricate_cik(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": {
                "events": [
                    {
                        "type": "ticker_change",
                        "date": "2022-06-10",
                        "cik": "BBG000B9XB24",
                        "ticker_change": {
                            "ticker": "OLD",
                            "new_ticker": "NEW",
                        },
                    }
                ]
            }
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_ticker_events("NEW")

        assert resp.ok
        assert len(resp.data) == 1
        assert resp.data[0].cik is None
        assert resp.data[0].old_cik is None
        assert resp.data[0].new_cik is None

    def test_get_ticker_events_skips_undated_ticker_change_rows(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200,
            {
                "results": {
                    "events": [
                        {
                            "type": "ticker_change",
                            "ticker_change": {"ticker": "FB", "new_ticker": "META"},
                        }
                    ]
                }
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_ticker_events("META")

        assert resp.ok
        assert resp.data == []
        flags = resp.lineage.data_quality_flags
        assert flags["raw_rows"] == 1
        assert flags["parsed_rows"] == 0
        assert flags["skipped_rows"] == 1
        assert flags["all_rows_skipped"] is True

    def test_get_ticker_events_accepts_valid_iso_date_fields(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        rows = [
            {
                "type": "ticker_change",
                "date": "2022-06-09",
                "ticker_change": {"ticker": "FB", "new_ticker": "META"},
            },
            {
                "type": "ticker_change",
                "event_date": "2022-06-10",
                "ticker_change": {"ticker": "OLD1", "new_ticker": "NEW1"},
            },
            {
                "type": "ticker_change",
                "effective_date": "2022-06-11",
                "ticker_change": {"ticker": "OLD2", "new_ticker": "NEW2"},
            },
            {
                "type": "ticker_change",
                "effective_utc": "2022-06-12",
                "ticker_change": {"ticker": "OLD3", "new_ticker": "NEW3"},
            },
            {
                "type": "ticker_change",
                "effective_utc": "2022-06-14T00:00:00Z",
                "ticker_change": {"ticker": "OLD5", "new_ticker": "NEW5"},
            },
            {
                "type": "ticker_change",
                "effective_utc": "2022-06-14T23:30:00-04:00",
                "ticker_change": {"ticker": "OLD6", "new_ticker": "NEW6"},
            },
            {
                "type": "ticker_change",
                "execution_date": "2022-06-13",
                "ticker_change": {"ticker": "OLD4", "new_ticker": "NEW4"},
            },
        ]
        session.get.return_value = _mock_response(200, {"results": {"events": rows}})
        adapter = self._adapter(session)

        resp = adapter.get_ticker_events("META")

        assert resp.ok
        assert len(resp.data) == 7
        assert [event.date for event in resp.data] == [
            "2022-06-09",
            None,
            None,
            None,
            None,
            None,
            None,
        ]
        assert [event.event_date for event in resp.data] == [
            "2022-06-09",
            "2022-06-10",
            None,
            None,
            None,
            None,
            None,
        ]
        assert [event.effective_date for event in resp.data] == [
            "2022-06-09",
            "2022-06-10",
            "2022-06-11",
            "2022-06-12",
            "2022-06-14",
            "2022-06-15",
            "2022-06-13",
        ]
        assert resp.lineage.data_quality_flags["raw_rows"] == 7
        assert resp.lineage.data_quality_flags["parsed_rows"] == 7
        assert resp.lineage.data_quality_flags["skipped_rows"] == 0

    def test_get_ticker_events_skips_malformed_date_rows(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        rows = [
            {
                "type": "ticker_change",
                "date": "not-a-date",
                "ticker_change": {"ticker": "FB", "new_ticker": "META"},
            },
            {
                "type": "ticker_change",
                "date": "2022-99-99",
                "ticker_change": {"ticker": "FB", "new_ticker": "META"},
            },
            {
                "type": "ticker_change",
                "date": "",
                "ticker_change": {"ticker": "FB", "new_ticker": "META"},
            },
            {
                "type": "ticker_change",
                "date": "   ",
                "ticker_change": {"ticker": "FB", "new_ticker": "META"},
            },
            {
                "type": "ticker_change",
                "date": 20220609,
                "ticker_change": {"ticker": "FB", "new_ticker": "META"},
            },
            {
                "type": "ticker_change",
                "effective_utc": "2022-06-09T00:00:00",
                "ticker_change": {"ticker": "FB", "new_ticker": "META"},
            },
        ]
        session.get.return_value = _mock_response(200, {"results": {"events": rows}})
        adapter = self._adapter(session)

        resp = adapter.get_ticker_events("META")

        assert resp.ok
        assert resp.data == []
        flags = resp.lineage.data_quality_flags
        assert flags["raw_rows"] == 6
        assert flags["parsed_rows"] == 0
        assert flags["skipped_rows"] == 6
        assert flags["all_rows_skipped"] is True

    def test_get_ticker_events_mixed_valid_and_malformed_date_rows(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        rows = [
            {
                "type": "ticker_change",
                "date": "2022-06-09",
                "ticker_change": {"ticker": "FB", "new_ticker": "META"},
            },
            {
                "type": "ticker_change",
                "date": "not-a-date",
                "ticker_change": {"ticker": "BAD", "new_ticker": "WORSE"},
            },
        ]
        session.get.return_value = _mock_response(200, {"results": {"events": rows}})
        adapter = self._adapter(session)

        resp = adapter.get_ticker_events("META")

        assert resp.ok
        assert [event.old_ticker for event in resp.data] == ["FB"]
        flags = resp.lineage.data_quality_flags
        assert flags["raw_rows"] == 2
        assert flags["parsed_rows"] == 1
        assert flags["skipped_rows"] == 1
        assert "all_rows_skipped" not in flags

    def test_get_ticker_events_skips_unsupported_returned_event_types(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200,
            {
                "results": {
                    "events": [
                        {"type": "name_change", "date": "2022-06-09", "name": "New"},
                        {"type": "delisting", "date": "2022-06-10", "ticker": "OLD"},
                        {"type": "merger", "date": "2022-06-11", "ticker": "OLD"},
                        {"type": "unknown_type", "date": "2022-06-12", "ticker": "OLD"},
                    ]
                }
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_ticker_events("META")

        assert resp.ok
        assert resp.data == []
        flags = resp.lineage.data_quality_flags
        assert flags["raw_rows"] == 4
        assert flags["parsed_rows"] == 0
        assert flags["skipped_rows"] == 4
        assert flags["all_rows_skipped"] is True
        assert flags["event_types_present"] == []

    def test_get_ticker_events_mixed_rows_only_reports_supported_events(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200,
            {
                "results": {
                    "events": [
                        {
                            "type": "ticker_change",
                            "date": "2022-06-09",
                            "ticker_change": {"ticker": "FB", "new_ticker": "META"},
                        },
                        {
                            "type": "ticker_change",
                            "ticker_change": {"ticker": "UNDATED", "new_ticker": "NEW"},
                        },
                        {"type": "name_change", "date": "2022-06-10", "name": "New"},
                    ]
                }
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_ticker_events("META")

        assert resp.ok
        assert [row.event_type for row in resp.data] == ["ticker_change"]
        assert resp.data[0].identity_continuity_status == "unproven"
        flags = resp.lineage.data_quality_flags
        assert flags["raw_rows"] == 3
        assert flags["parsed_rows"] == 1
        assert flags["skipped_rows"] == 2
        assert flags["event_types_present"] == ["ticker_change"]
        assert flags["identity_continuity_unproven_rows"] == 1

    def test_get_ticker_events_all_invalid_rows_are_flagged(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200,
            {"results": {"events": [None, {}, {"type": "ticker_change"}]}},
        )
        adapter = self._adapter(session)

        resp = adapter.get_ticker_events("META")

        assert resp.ok
        assert resp.data == []
        assert resp.lineage.data_quality_flags["raw_rows"] == 3
        assert resp.lineage.data_quality_flags["parsed_rows"] == 0
        assert resp.lineage.data_quality_flags["skipped_rows"] == 3
        assert resp.lineage.data_quality_flags["all_rows_skipped"] is True

    def test_get_ticker_events_raw_event_is_deep_copied(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        row = {
            "type": "ticker_change",
            "date": "2022-06-09",
            "ticker_change": {"ticker": "FB", "new_ticker": "META", "nested": {"rank": 1}},
        }
        json_data = {"results": {"events": [row]}}
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_ticker_events("META")

        assert resp.ok
        event = resp.data[0]
        row["ticker_change"]["nested"]["rank"] = 2
        assert event.raw_event["ticker_change"]["nested"]["rank"] == 1
        event.raw_event["ticker_change"]["nested"]["rank"] = 3
        assert event.raw_event["ticker_change"]["nested"]["rank"] == 3

    def test_get_ticker_events_pagination_marker_fails_loud(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.return_value = _mock_response(
            200,
            {
                "results": {
                    "events": [
                        {
                            "type": "ticker_change",
                            "ticker_change": {"ticker": "FB"},
                        }
                    ]
                },
                "next_url": "https://api.polygon.io/vX/reference/tickers/META/events?cursor=abc&apiKey=SECRET",
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_ticker_events("META")

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "pagination"
        assert resp.error.retryable is False
        assert "SECRET" not in repr(resp.lineage)
        assert "apiKey" not in repr(resp.lineage)
        assert session.get.call_count == 1

    def test_get_ticker_events_request_exception_does_not_leak_secret_or_url(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        session.get.side_effect = requests.exceptions.ConnectionError(
            "failed https://api.polygon.io/vX/reference/tickers/META/events?apiKey=test-polygon-key&types=ticker_change"
        )
        adapter = self._adapter(session)

        resp = adapter.get_ticker_events("META")

        assert not resp.ok
        assert resp.error.error_type == "http"
        assert resp.error.message == "Polygon request failed: ConnectionError"
        assert "test-polygon-key" not in resp.error.message
        assert "apiKey" not in resp.error.message
        assert "https://api.polygon.io" not in resp.error.message

    def test_get_ticker_events_lineage_hash_stability(self):
        session = MagicMock(spec=requests.Session)
        session.params = {}
        json_data = {
            "results": {
                "events": [
                    {
                        "type": "ticker_change",
                        "date": "2022-06-09",
                        "ticker_change": {"ticker": "FB"},
                    }
                ]
            }
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp1 = adapter.get_ticker_events("META")
        resp2 = adapter.get_ticker_events("META")

        assert resp1.ok
        assert resp2.ok
        assert resp1.lineage.raw_payload_hash == resp2.lineage.raw_payload_hash

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
        assert resp.lineage.data_quality_flags["bare_list_payload"] is True

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

        resp = adapter.get_news("ACME")

        assert resp.ok
        article = resp.data[0]
        assert article.created == datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
        assert article.updated == datetime.fromtimestamp(1779883200, tz=timezone.utc)
        assert article.published == datetime(2026, 5, 27, 12, 15, tzinfo=timezone.utc)

    def test_get_news_future_knowledge_timestamps_are_nulled_and_flagged(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "news": [
                    {
                        "id": "future-news",
                        "created": "2099-01-01T00:00:00Z",
                        "updated": "2099-01-02T00:00:00Z",
                        "published": "2099-01-03T00:00:00Z",
                        "date": "2099-01-04",
                    }
                ]
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_news(
            "ACME",
            asof=datetime(2026, 5, 28, tzinfo=timezone.utc),
        )

        assert resp.ok
        article = resp.data[0]
        assert article.created is None
        assert article.updated is None
        assert article.published is None
        assert article.event_date == "2099-01-04"
        flags = resp.lineage.data_quality_flags
        assert flags["knowledge_timestamp_warning_rows"] == 1
        assert flags["knowledge_timestamp_warning_types"] == {
            "news_created_future": 1,
            "news_published_future": 1,
            "news_updated_future": 1,
        }

    def test_get_news_event_date_does_not_populate_publication_timestamp(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {"news": [{"id": "event-only", "date": "2099-01-04"}]},
        )
        adapter = self._adapter(session)

        resp = adapter.get_news(
            "ACME",
            asof=datetime(2026, 5, 28, tzinfo=timezone.utc),
        )

        assert resp.ok
        article = resp.data[0]
        assert article.published is None
        assert article.created is None
        assert article.updated is None
        assert article.event_date == "2099-01-04"
        assert "knowledge_timestamp_warning_rows" not in resp.lineage.data_quality_flags

    def test_get_news_lineage_does_not_expose_secret(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, {"news": []})
        adapter = self._adapter(session)

        resp = adapter.get_news("ACME")

        assert resp.ok
        assert "test-benzinga-key" not in repr(resp.lineage)
        assert "test-benzinga-key" not in repr(resp.lineage.data_quality_flags)
        assert resp.lineage.data_quality_flags["raw_rows"] == 0

    def test_get_earnings_ok_and_calendar_params(self):
        session = MagicMock(spec=requests.Session)
        json_data = {
            "earnings": [
                {
                    "id": "earn-1",
                    "ticker": "ACME",
                    "name": "Acme Corp",
                    "exchange": "NASDAQ",
                    "currency": "USD",
                    "cusip": "004397105",
                    "isin": "US0043971052",
                    "period": "Q1",
                    "period_year": "2026",
                    "date": "2026-05-27",
                    "time": "bmo",
                    "eps": "1.25",
                    "eps_est": "1.10",
                    "eps_prior": "0.92",
                    "eps_surprise": "0.15",
                    "eps_surprise_percent": "13.64",
                    "eps_type": "GAAP",
                    "revenue": "1000000.00",
                    "revenue_est": "950000",
                    "revenue_prior": "875000",
                    "revenue_surprise": "50000",
                    "revenue_surprise_percent": "5.26",
                    "revenue_type": "reported",
                    "date_confirmed": "1",
                    "importance": "5",
                    "notes": "Beat estimates",
                    "updated": 1779883200,
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_earnings(
            symbols=["ACME", "BETA"],
            date_from="2026-05-01",
            date_to="2026-05-31",
            page=2,
            pagesize=50,
            updated=1779796800,
        )

        assert resp.ok
        assert len(resp.data) == 1
        event = resp.data[0]
        assert event.id == "earn-1"
        assert event.ticker == "ACME"
        assert event.currency == "USD"
        assert event.period == "Q1"
        assert event.period_year == 2026
        assert event.date == "2026-05-27"
        assert event.time == "bmo"
        assert event.eps == Decimal("1.25")
        assert event.eps_est == Decimal("1.10")
        assert event.revenue == Decimal("1000000.00")
        assert event.revenue_est == Decimal("950000")
        assert event.date_confirmed is True
        assert event.importance == 5
        assert event.notes == "Beat estimates"
        assert event.updated == datetime.fromtimestamp(1779883200, tz=timezone.utc)
        assert event.raw["ticker"] == "ACME"
        assert resp.lineage.endpoint == "/api/v2.1/calendar/earnings"
        session.get.assert_called_with(
            "https://api.benzinga.com/api/v2.1/calendar/earnings",
            params={
                "pagesize": 50,
                "page": 2,
                "parameters[tickers]": "ACME,BETA",
                "parameters[date_from]": "2026-05-01",
                "parameters[date_to]": "2026-05-31",
                "parameters[updated]": 1779796800,
                "token": "test-benzinga-key",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )

    def test_get_guidance_ok(self):
        session = MagicMock(spec=requests.Session)
        json_data = [
            {
                "id": "guidance-1",
                "ticker": "ACME",
                "name": "Acme Corp",
                "exchange": "NASDAQ",
                "currency": "USD",
                "cusip": "004397105",
                "period": "FY",
                "period_year": 2026,
                "date": "2026-05-27",
                "time": "amc",
                "eps_guidance_est": "1.80",
                "eps_guidance_min": "1.70",
                "eps_guidance_max": "1.95",
                "eps_guidance_prior_min": "1.50",
                "eps_guidance_prior_max": "1.60",
                "eps_type": "adjusted",
                "revenue_guidance_est": "2000000",
                "revenue_guidance_min": "1900000",
                "revenue_guidance_max": "2100000",
                "revenue_guidance_prior_min": "1800000",
                "revenue_guidance_prior_max": "1850000",
                "revenue_type": "sales",
                "is_primary": 1,
                "prelim": "false",
                "importance": 4,
                "notes": "Raised full-year guidance",
                "updated": "1779883200000",
            }
        ]
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_guidance("ACME")

        assert resp.ok
        assert len(resp.data) == 1
        event = resp.data[0]
        assert event.id == "guidance-1"
        assert event.ticker == "ACME"
        assert event.currency == "USD"
        assert event.cusip == "004397105"
        assert event.eps_guidance_min == Decimal("1.70")
        assert event.eps_guidance_max == Decimal("1.95")
        assert event.revenue_guidance_est == Decimal("2000000")
        assert event.is_primary is True
        assert event.prelim is False
        assert event.notes == "Raised full-year guidance"
        assert event.updated == datetime.fromtimestamp(1779883200, tz=timezone.utc)
        assert event.raw["id"] == "guidance-1"
        assert resp.lineage.endpoint == "/api/v2.1/calendar/guidance"

    def test_get_ratings_ok(self):
        session = MagicMock(spec=requests.Session)
        json_data = {
            "ratings": [
                {
                    "id": "rating-1",
                    "ticker": "ACME",
                    "name": "Acme Corp",
                    "exchange": "NASDAQ",
                    "currency": "USD",
                    "cusip": "004397105",
                    "isin": "US0043971052",
                    "date": "2026-05-27",
                    "time": "13:30:00",
                    "analyst": "A. Analyst",
                    "analyst_id": "42",
                    "analyst_name": "A. Analyst",
                    "firm": "Example Securities",
                    "firm_id": "123",
                    "action_company": "Initiates Coverage On",
                    "action_pt": "Announces",
                    "rating_current": "Buy",
                    "rating_prior": "Neutral",
                    "pt_current": "12.50",
                    "pt_prior": "10",
                    "adjusted_pt_current": "12.00",
                    "adjusted_pt_prior": "9.75",
                    "pt_pct_change": "25.0",
                    "ratings_accuracy": "0.72",
                    "importance": "3",
                    "notes": "New coverage",
                    "url": "https://benzinga.example/ratings/1",
                    "url_calendar": "https://benzinga.example/calendar/1",
                    "url_news": "https://benzinga.example/news/1",
                    "updated": "Wed, 27 May 2026 08:15:00 -0400",
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_ratings("ACME")

        assert resp.ok
        event = resp.data[0]
        assert event.id == "rating-1"
        assert event.ticker == "ACME"
        assert event.analyst == "A. Analyst"
        assert event.firm == "Example Securities"
        assert event.action_company == "Initiates Coverage On"
        assert event.rating_current == "Buy"
        assert event.rating_prior == "Neutral"
        assert event.pt_current == Decimal("12.50")
        assert event.pt_prior == Decimal("10")
        assert event.adjusted_pt_current == Decimal("12.00")
        assert event.ratings_accuracy == Decimal("0.72")
        assert event.updated == datetime(2026, 5, 27, 12, 15, tzinfo=timezone.utc)
        assert event.raw["url_news"] == "https://benzinga.example/news/1"
        assert resp.lineage.endpoint == "/api/v2.1/calendar/ratings"

    def test_get_offerings_ok(self):
        session = MagicMock(spec=requests.Session)
        json_data = {
            "offerings": [
                {
                    "id": "offering-1",
                    "ticker": "ACME",
                    "name": "Acme Corp",
                    "exchange": "NASDAQ",
                    "currency": "USD",
                    "cusip": "004397105",
                    "date": "2026-05-27",
                    "time": "09:00:00",
                    "offering_type": "Secondary",
                    "price": "2.50",
                    "number_shares": "4000000",
                    "dollar_shares": "10000000",
                    "proceeds": "9500000",
                    "shelf": "true",
                    "importance": 5,
                    "notes": "Registered direct offering",
                    "url": "https://benzinga.example/offerings/1",
                    "updated": "2026-05-27T12:00:00Z",
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_offerings("ACME")

        assert resp.ok
        event = resp.data[0]
        assert event.id == "offering-1"
        assert event.ticker == "ACME"
        assert event.cusip == "004397105"
        assert event.offering_type == "Secondary"
        assert event.price == Decimal("2.50")
        assert event.number_shares == Decimal("4000000")
        assert event.proceeds == Decimal("9500000")
        assert event.shelf is True
        assert event.updated == datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
        assert event.raw["notes"] == "Registered direct offering"
        assert resp.lineage.endpoint == "/api/v2.1/calendar/offerings"

    def test_get_dividends_ok_and_calendar_params(self):
        session = MagicMock(spec=requests.Session)
        json_data = {
            "dividends": [
                {
                    "id": "dividend-1",
                    "ticker": "ACME",
                    "name": "Acme Corp",
                    "exchange": "NYSE",
                    "currency": "USD",
                    "cusip": "004397105",
                    "isin": "US0043971052",
                    "date": "2026-05-26",
                    "ex_dividend_date": "2026-06-15",
                    "payable_date": "2026-06-30",
                    "record_date": "2026-06-16",
                    "dividend": "0.8500",
                    "dividend_prior": "0.8100",
                    "dividend_type": "Cash",
                    "dividend_yield": "0.0277755085368842",
                    "frequency": "4",
                    "confirmed": "true",
                    "end_regular_dividend": False,
                    "period": "Q2",
                    "year": "2026",
                    "importance": 5,
                    "notes": "Quarterly dividend",
                    "updated": 1779883200,
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_dividends(
            symbols=[" acme ", "BETA"],
            date_from="2026-05-01",
            date_to="2026-05-31",
            page=2,
            pagesize=25,
            updated=1779796800,
        )

        assert resp.ok
        assert len(resp.data) == 1
        event = resp.data[0]
        assert event.id == "dividend-1"
        assert event.ticker == "ACME"
        assert event.dividend == Decimal("0.8500")
        assert event.dividend_prior == Decimal("0.8100")
        assert event.dividend_type == "Cash"
        assert event.dividend_yield == Decimal("0.0277755085368842")
        assert event.frequency == 4
        assert event.confirmed is True
        assert event.end_regular_dividend is False
        assert event.updated == datetime.fromtimestamp(1779883200, tz=timezone.utc)
        assert event.raw["ticker"] == "ACME"
        assert resp.lineage.endpoint == "/api/v2.1/calendar/dividends"
        session.get.assert_called_with(
            "https://api.benzinga.com/api/v2.1/calendar/dividends",
            params={
                "pagesize": 25,
                "page": 2,
                "parameters[tickers]": "ACME,BETA",
                "parameters[date_from]": "2026-05-01",
                "parameters[date_to]": "2026-05-31",
                "parameters[updated]": 1779796800,
                "token": "test-benzinga-key",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )

    def test_get_insider_filings_ok_and_params(self):
        session = MagicMock(spec=requests.Session)
        json_data = {
            "data": [
                {
                    "id": "filing-1",
                    "accession_number": "0001234567-26-000001",
                    "company_cik": "0001234567",
                    "company_name": "Acme Corp",
                    "company_symbol": "ACME",
                    "filing_date": "2026-05-28T01:49:13Z",
                    "form_type": "4",
                    "html_url": "https://www.sec.gov/Archives/example-index.htm",
                    "is_10b5": False,
                    "owner": {
                        "insider_cik": "0007654321",
                        "insider_name": "Jane Insider",
                        "insider_title": "CEO",
                        "is_director": True,
                        "is_officer": True,
                        "is_ten_percent_owner": False,
                        "raw_signature": "/s/ Jane Insider",
                    },
                    "remaining_shares": "34400",
                    "traded_percentage": "0.68%",
                    "footnotes": [{"id": "F1", "text": "Example footnote"}],
                    "transactions": [{"transaction_id": "tx-1"}],
                    "updated": "1779883200000",
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_insider_filings(
            tickers=["acme"],
            date_from="2026-05-01",
            date_to="2026-05-31",
            page=3,
            pagesize=20,
            updated=1779796800,
        )

        assert resp.ok
        filing = resp.data[0]
        assert filing.id == "filing-1"
        assert filing.accession_number == "0001234567-26-000001"
        assert filing.company_symbol == "ACME"
        assert filing.form_type == "4"
        assert filing.filing_date == datetime(2026, 5, 28, 1, 49, 13, tzinfo=timezone.utc)
        assert filing.insider_name == "Jane Insider"
        assert filing.is_director is True
        assert filing.remaining_shares == Decimal("34400")
        assert filing.traded_percentage == "0.68%"
        assert filing.footnotes == [{"id": "F1", "text": "Example footnote"}]
        assert filing.transactions == [{"transaction_id": "tx-1"}]
        assert filing.updated == datetime.fromtimestamp(1779883200, tz=timezone.utc)
        assert filing.raw["id"] == "filing-1"
        assert resp.lineage.endpoint == "/api/v1/sec/insider_transactions/filings"
        session.get.assert_called_with(
            "https://api.benzinga.com/api/v1/sec/insider_transactions/filings",
            params={
                "pagesize": 20,
                "page": 3,
                "search_keys_type": "symbol",
                "search_keys": "ACME",
                "date_from": "2026-05-01",
                "date_to": "2026-05-31",
                "updated_since": 1779796800,
                "token": "test-benzinga-key",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )

    def test_get_insider_transactions_ok_and_params(self):
        session = MagicMock(spec=requests.Session)
        json_data = {
            "data": [
                {
                    "transaction_id": "tx-1",
                    "acquired_or_disposed": "A",
                    "conversion_exercise_price_derivative": "0",
                    "date_transaction": "2026-05-22T00:00:00Z",
                    "is_derivative": False,
                    "owner": {
                        "insider_cik": "0007654321",
                        "insider_name": "Jane Insider",
                        "insider_title": "CEO",
                        "is_director": True,
                        "is_officer": True,
                        "is_ten_percent_owner": False,
                    },
                    "ownership": "D",
                    "post_transaction_quantity": "34400",
                    "price_per_share": "2.50",
                    "security_title": "Common Stock",
                    "shares": "10000",
                    "transaction_code": "P",
                    "underlying_security_title": "Option",
                    "underlying_shares": "10000",
                    "voluntarily_reported": "false",
                    "filing": {
                        "id": "filing-1",
                        "accession_number": "0001234567-26-000001",
                        "company_cik": "0001234567",
                        "company_name": "Acme Corp",
                        "company_symbol": "ACME",
                        "filing_date": "2026-05-28T01:49:13Z",
                        "form_type": "4",
                        "html_url": "https://www.sec.gov/Archives/example-index.htm",
                    },
                    "updated": "Wed, 27 May 2026 08:15:00 -0400",
                }
            ]
        }
        session.get.return_value = _mock_response(200, json_data)
        adapter = self._adapter(session)

        resp = adapter.get_insider_transactions(
            symbols=[" ACME "],
            date_from="2026-05-01",
            date_to="2026-05-31",
            page=4,
            pagesize=10,
            updated="2026-05-01T00:00:00Z",
        )

        assert resp.ok
        transaction = resp.data[0]
        assert transaction.transaction_id == "tx-1"
        assert transaction.company_symbol == "ACME"
        assert transaction.accession_number == "0001234567-26-000001"
        assert transaction.filing_id == "filing-1"
        assert transaction.insider_name == "Jane Insider"
        assert transaction.acquired_or_disposed == "A"
        assert transaction.date_transaction == datetime(2026, 5, 22, tzinfo=timezone.utc)
        assert transaction.post_transaction_quantity == Decimal("34400")
        assert transaction.price_per_share == Decimal("2.50")
        assert transaction.shares == Decimal("10000")
        assert transaction.transaction_code == "P"
        assert transaction.voluntarily_reported is False
        assert transaction.updated == datetime(2026, 5, 27, 12, 15, tzinfo=timezone.utc)
        assert transaction.raw["transaction_id"] == "tx-1"
        assert resp.lineage.endpoint == "/api/v1/sec/insider_transactions/transactions"
        session.get.assert_called_with(
            "https://api.benzinga.com/api/v1/sec/insider_transactions/transactions",
            params={
                "pagesize": 10,
                "page": 4,
                "search_keys_type": "symbol",
                "search_keys": "ACME",
                "date_from": "2026-05-01",
                "date_to": "2026-05-31",
                "updated_since": "2026-05-01T00:00:00Z",
                "token": "test-benzinga-key",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )

    def test_calendar_adapters_empty_payload_success(self):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        for method_name, payload in [
            ("get_earnings", {"earnings": []}),
            ("get_guidance", {"guidance": []}),
            ("get_ratings", {"ratings": []}),
            ("get_offerings", {"offerings": []}),
        ]:
            session.get.return_value = _mock_response(200, payload)
            resp = getattr(adapter, method_name)("ACME")

            assert resp.ok
            assert resp.data == []

    def test_calendar_adapters_malformed_payload_is_parse_error(self):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        for method_name in [
            "get_earnings",
            "get_guidance",
            "get_ratings",
            "get_offerings",
        ]:
            session.get.return_value = _mock_response(200, {"unexpected": []})
            resp = getattr(adapter, method_name)("ACME")

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "parse"
            assert resp.lineage.data_quality_flags["payload_shape_error"] is True

    def test_calendar_adapters_skip_rows_without_ticker(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, {"ratings": [{}]})
        adapter = self._adapter(session)

        resp = adapter.get_ratings("ACME")

        assert resp.ok
        assert resp.data == []

    def test_calendar_ticker_params_are_normalized_and_blank_only_filter_omitted(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, {"earnings": []})
        adapter = self._adapter(session)

        resp = adapter.get_earnings(tickers=[" ACME ", ""])

        assert resp.ok
        params = session.get.call_args.kwargs["params"]
        assert params["parameters[tickers]"] == "ACME"

        session.get.return_value = _mock_response(200, {"guidance": []})
        resp = adapter.get_guidance(
            tickers=[" ", ""],
            date_from="2026-05-01",
            date_to="2026-05-31",
        )

        assert resp.ok
        params = session.get.call_args.kwargs["params"]
        assert "parameters[tickers]" not in params

    def test_calendar_ticker_params_reject_non_string_members(self):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        for method_name, endpoint in [
            ("get_earnings", "/api/v2.1/calendar/earnings"),
            ("get_dividends", "/api/v2.1/calendar/dividends"),
            (
                "get_insider_filings",
                "/api/v1/sec/insider_transactions/filings",
            ),
            (
                "get_insider_transactions",
                "/api/v1/sec/insider_transactions/transactions",
            ),
        ]:
            resp = getattr(adapter, method_name)(tickers=["AAPL", 123])  # type: ignore[list-item]

            assert not resp.ok
            assert resp.error.error_type == "validation"
            assert resp.error.retryable is False
            assert resp.error.message == "Benzinga ticker parameters must be strings"
            assert resp.lineage.endpoint == endpoint
        session.get.assert_not_called()

    def test_request_exception_message_does_not_expose_secret_or_url(self):
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = requests.exceptions.ConnectionError(
            "GET https://api.benzinga.com/api/v2/news?tickers=ACME&token=test-benzinga-key failed"
        )
        adapter = self._adapter(session)

        resp = adapter.get_news("ACME")

        assert not resp.ok
        assert resp.error.error_type == "http"
        assert "ConnectionError" in resp.error.message
        assert "test-benzinga-key" not in resp.error.message
        assert "token" not in resp.error.message
        assert "https://api.benzinga.com" not in resp.error.message

    @pytest.mark.parametrize(
        ("method_name", "endpoint"),
        [
            ("get_earnings", "/api/v2.1/calendar/earnings"),
            ("get_guidance", "/api/v2.1/calendar/guidance"),
            ("get_ratings", "/api/v2.1/calendar/ratings"),
            ("get_offerings", "/api/v2.1/calendar/offerings"),
        ],
    )
    def test_calendar_adapters_parse_error(self, method_name, endpoint):
        session = MagicMock(spec=requests.Session)
        resp_mock = _mock_response(200, text="not json")
        resp_mock.json.side_effect = ValueError("parse fail")
        session.get.return_value = resp_mock
        adapter = self._adapter(session)

        resp = getattr(adapter, method_name)("ACME")

        assert not resp.ok
        assert resp.error.error_type == "parse"
        assert resp.error.retryable is False
        assert resp.lineage.endpoint == endpoint

    @pytest.mark.parametrize(
        ("method_name", "payload"),
        [
            ("get_earnings", {"earnings": [{"id": "earn-1", "ticker": "ACME"}]}),
            ("get_guidance", [{"id": "guidance-1", "ticker": "ACME"}]),
            ("get_ratings", {"ratings": [{"id": "rating-1", "ticker": "ACME"}]}),
            ("get_offerings", {"offerings": [{"id": "offering-1", "ticker": "ACME"}]}),
        ],
    )
    def test_calendar_lineage_hash_stability(self, method_name, payload):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, payload)
        adapter = self._adapter(session)

        resp1 = getattr(adapter, method_name)("ACME")
        resp2 = getattr(adapter, method_name)("ACME")

        assert resp1.ok
        assert resp2.ok
        assert resp1.lineage.raw_payload_hash == resp2.lineage.raw_payload_hash

    def test_dividends_and_insider_adapters_empty_payload_success(self):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        for method_name, payload in [
            ("get_dividends", {"dividends": []}),
            ("get_insider_filings", {"data": []}),
            ("get_insider_transactions", {"data": []}),
        ]:
            session.get.return_value = _mock_response(200, payload)
            resp = getattr(adapter, method_name)("ACME")

            assert resp.ok
            assert resp.data == []

    def test_dividends_and_insider_adapters_malformed_payload_is_parse_error(self):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        for method_name in [
            "get_dividends",
            "get_insider_filings",
            "get_insider_transactions",
        ]:
            session.get.return_value = _mock_response(200, {"unexpected": []})
            resp = getattr(adapter, method_name)("ACME")

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "parse"
            assert resp.lineage.data_quality_flags["payload_shape_error"] is True

    def test_dividends_and_insider_adapters_skip_identity_less_rows(self):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        for method_name, payload in [
            ("get_dividends", {"dividends": [{}]}),
            ("get_insider_filings", {"data": [{}]}),
            ("get_insider_transactions", {"data": [{}]}),
        ]:
            session.get.return_value = _mock_response(200, payload)
            resp = getattr(adapter, method_name)("ACME")

            assert resp.ok
            assert resp.data == []

    def test_dividends_and_insider_adapters_parse_error(self):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        for method_name in [
            "get_dividends",
            "get_insider_filings",
            "get_insider_transactions",
        ]:
            resp_mock = _mock_response(200, text="not json")
            resp_mock.json.side_effect = ValueError("parse fail")
            session.get.return_value = resp_mock
            resp = getattr(adapter, method_name)("ACME")

            assert not resp.ok
            assert resp.data is None
            assert resp.error.error_type == "parse"

    @pytest.mark.parametrize("status_code", [403, 429, 500])
    def test_dividends_and_insider_adapters_provider_errors(self, status_code):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        expected_error = {
            403: "auth",
            429: "rate_limit",
            500: "http",
        }[status_code]
        for method_name in [
            "get_dividends",
            "get_insider_filings",
            "get_insider_transactions",
        ]:
            session.get.return_value = _mock_response(status_code, text="provider error")
            resp = getattr(adapter, method_name)("ACME")

            assert not resp.ok
            assert resp.error.error_type == expected_error
            assert resp.error.status_code == status_code

    @pytest.mark.parametrize(
        ("method_name", "payload"),
        [
            ("get_dividends", {"dividends": [{"id": "div-1", "ticker": "ACME"}]}),
            (
                "get_insider_filings",
                {"data": [{"id": "filing-1", "accession_number": "0001"}]},
            ),
            (
                "get_insider_transactions",
                {"data": [{"transaction_id": "tx-1"}]},
            ),
        ],
    )
    def test_dividends_and_insider_lineage_hash_stability(self, method_name, payload):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, payload)
        adapter = self._adapter(session)

        resp1 = getattr(adapter, method_name)("ACME")
        resp2 = getattr(adapter, method_name)("ACME")

        assert resp1.ok
        assert resp2.ok
        assert resp1.lineage.raw_payload_hash == resp2.lineage.raw_payload_hash

    def test_dividends_and_insider_lineage_does_not_expose_secret(self):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        for method_name, payload in [
            ("get_dividends", {"dividends": []}),
            ("get_insider_filings", {"data": []}),
            ("get_insider_transactions", {"data": []}),
        ]:
            session.get.return_value = _mock_response(200, payload)
            resp = getattr(adapter, method_name)("ACME")

            assert resp.ok
            assert "test-benzinga-key" not in repr(resp.lineage)
            assert resp.lineage.source_authority == "Benzinga"

    def test_calendar_lineage_does_not_expose_secret(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, {"offerings": []})
        adapter = self._adapter(session)

        resp = adapter.get_offerings("ACME")

        assert resp.ok
        assert "test-benzinga-key" not in repr(resp.lineage)
        assert resp.lineage.source_authority == "Benzinga"

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
        resp = adapter.get_mergers_acquisitions(
            date_from="2026-05-01",
            date_to="2026-05-31",
        )

        assert resp.ok
        assert len(resp.data) == 1
        assert resp.data[0].id == "42"
        assert resp.data[0].importance == 3
        assert resp.data[0].updated == 1779307200

    def test_get_mergers_acquisitions_unexpected_payload_is_parse_error(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, {"unexpected": []})
        adapter = self._adapter(session)
        resp = adapter.get_mergers_acquisitions("ACME")

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "parse"
        assert resp.lineage.data_quality_flags["payload_shape_error"] is True

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

    def test_json_parse_error_message_is_sanitized(self):
        session = MagicMock(spec=requests.Session)
        resp_mock = _mock_response(200, text="not json token=SECRET")
        resp_mock.json.side_effect = ValueError(
            "GET https://api.benzinga.com/api/v2/news?token=SECRET failed"
        )
        session.get.return_value = resp_mock
        adapter = self._adapter(session)

        resp = adapter.get_news("ACME")

        assert not resp.ok
        assert resp.error.error_type == "parse"
        assert resp.error.message == "Benzinga JSON parse error"
        assert "SECRET" not in repr(resp)
        assert "token" not in resp.error.message

    @pytest.mark.parametrize(
        ("method_name", "payload"),
        [
            ("get_news", {"unexpected": True}),
            ("get_wiims", {"unexpected": True}),
            ("get_earnings", {"unexpected": True}),
            ("get_guidance", {"unexpected": True}),
            ("get_ratings", {"unexpected": True}),
            ("get_offerings", {"unexpected": True}),
            ("get_dividends", {"unexpected": True}),
            ("get_insider_filings", {"unexpected": True}),
            ("get_insider_transactions", {"unexpected": True}),
            ("get_mergers_acquisitions", {"unexpected": True}),
        ],
    )
    def test_benzinga_unexpected_payload_shapes_are_parse_errors(
        self, method_name, payload
    ):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, payload)
        adapter = self._adapter(session)

        resp = getattr(adapter, method_name)("ACME")

        assert not resp.ok
        assert resp.data is None
        assert resp.error.error_type == "parse"
        assert resp.lineage.data_quality_flags["payload_shape_error"] is True

    @pytest.mark.parametrize(
        ("method_name", "payload"),
        [
            ("get_news", {"news": {"not": "list"}}),
            ("get_earnings", {"earnings": {"not": "list"}}),
            ("get_guidance", {"guidance": {"not": "list"}}),
            ("get_ratings", {"ratings": {"not": "list"}}),
            ("get_offerings", {"offerings": {"not": "list"}}),
            ("get_dividends", {"dividends": {"not": "list"}}),
            ("get_insider_filings", {"data": {"not": "list"}}),
            ("get_insider_transactions", {"data": {"not": "list"}}),
            ("get_mergers_acquisitions", {"ma": {"not": "list"}}),
        ],
    )
    def test_benzinga_expected_key_non_list_is_parse_error(self, method_name, payload):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, payload)
        adapter = self._adapter(session)

        resp = getattr(adapter, method_name)("ACME")

        assert not resp.ok
        assert resp.error.error_type == "parse"
        assert resp.lineage.data_quality_flags["payload_shape_error"] is True

    @pytest.mark.parametrize(
        ("method_name", "payload"),
        [
            ("get_news", {"news": [{}]}),
            ("get_wiims", {"news": [{}]}),
            ("get_mergers_acquisitions", {"ma": [{}]}),
        ],
    )
    def test_benzinga_news_wiims_and_ma_skip_identity_less_rows(
        self, method_name, payload
    ):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, payload)
        adapter = self._adapter(session)

        resp = getattr(adapter, method_name)("ACME")

        assert resp.ok
        assert resp.data == []
        assert resp.lineage.data_quality_flags["raw_rows"] == 1
        assert resp.lineage.data_quality_flags["parsed_rows"] == 0
        assert resp.lineage.data_quality_flags["skipped_rows"] == 1
        assert resp.lineage.data_quality_flags["all_rows_skipped"] is True

    def test_benzinga_non_dict_rows_are_skipped_with_telemetry(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {"ratings": [None, 7, {"ticker": "ACME", "date": "2026-05-27"}]},
        )
        adapter = self._adapter(session)

        resp = adapter.get_ratings("ACME")

        assert resp.ok
        assert len(resp.data) == 1
        assert resp.lineage.data_quality_flags["raw_rows"] == 3
        assert resp.lineage.data_quality_flags["parsed_rows"] == 1
        assert resp.lineage.data_quality_flags["skipped_rows"] == 2

    def test_benzinga_bare_list_payload_is_flagged_and_still_row_guarded(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            [{}, {"ticker": "ACME", "date": "2026-05-27"}],
        )
        adapter = self._adapter(session)

        resp = adapter.get_guidance("ACME")

        assert resp.ok
        assert len(resp.data) == 1
        flags = resp.lineage.data_quality_flags
        assert flags["bare_list_payload"] is True
        assert flags["raw_rows"] == 2
        assert flags["parsed_rows"] == 1
        assert flags["skipped_rows"] == 1

    @pytest.mark.parametrize(
        "call",
        [
            lambda adapter: adapter.get_news(tickers=["AAPL", 123]),  # type: ignore[list-item]
            lambda adapter: adapter.get_news(tickers="AAPL,MSFT"),
            lambda adapter: adapter.get_news(tickers="AAPL\x00MSFT"),
            lambda adapter: adapter.get_news(tickers="AAPL\nMSFT"),
            lambda adapter: adapter.get_news(tickers="AAPL\tMSFT"),
            lambda adapter: adapter.get_news(tickers={"AAPL", "MSFT"}),  # type: ignore[arg-type]
            lambda adapter: adapter.get_earnings(tickers=["AAPL", 123]),  # type: ignore[list-item]
            lambda adapter: adapter.get_earnings(tickers="AAPL,MSFT"),
            lambda adapter: adapter.get_earnings(tickers="AAPL\x00MSFT"),
            lambda adapter: adapter.get_earnings(tickers={"AAPL", "MSFT"}),  # type: ignore[arg-type]
            lambda adapter: adapter.get_insider_filings(tickers=[object()]),  # type: ignore[list-item]
            lambda adapter: adapter.get_insider_filings(tickers="AAPL\nMSFT"),
            lambda adapter: adapter.get_insider_filings(tickers=frozenset(["AAPL"])),  # type: ignore[arg-type]
            lambda adapter: adapter.get_mergers_acquisitions(tickers=[123]),  # type: ignore[list-item]
            lambda adapter: adapter.get_mergers_acquisitions(tickers="AAPL\tMSFT"),
            lambda adapter: adapter.get_mergers_acquisitions(tickers={"AAPL"}),  # type: ignore[arg-type]
        ],
    )
    def test_benzinga_ticker_validation_rejects_bad_inputs_before_http(self, call):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        resp = call(adapter)

        assert not resp.ok
        assert resp.error.error_type == "validation"
        session.get.assert_not_called()

    @pytest.mark.parametrize(
        "call",
        [
            lambda adapter: adapter.get_news(),
            lambda adapter: adapter.get_news(tickers=["", " "]),
            lambda adapter: adapter.get_earnings(tickers=["", " "]),
            lambda adapter: adapter.get_insider_filings(tickers=" "),
            lambda adapter: adapter.get_mergers_acquisitions(tickers=None),
        ],
    )
    def test_benzinga_broad_queries_require_ticker_or_date_bounds(self, call):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        resp = call(adapter)

        assert not resp.ok
        assert resp.error.error_type == "validation"
        session.get.assert_not_called()

    def test_benzinga_blank_ticker_with_full_date_bounds_is_allowed(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, {"earnings": []})
        adapter = self._adapter(session)

        resp = adapter.get_earnings(
            tickers=["", " "],
            date_from="2026-05-01",
            date_to="2026-05-31",
        )

        assert resp.ok
        params = session.get.call_args.kwargs["params"]
        assert "parameters[tickers]" not in params
        assert params["parameters[date_from]"] == "2026-05-01"
        assert params["parameters[date_to]"] == "2026-05-31"

    @pytest.mark.parametrize(
        "call",
        [
            lambda adapter: adapter.get_news("ACME", date_from="bad", date_to="2026-05-31"),
            lambda adapter: adapter.get_earnings("ACME", date_from="2026-05-31", date_to="2026-05-01"),
            lambda adapter: adapter.get_dividends("ACME", page=0),
            lambda adapter: adapter.get_insider_transactions("ACME", pagesize=1001),
            lambda adapter: adapter.get_mergers_acquisitions("ACME", page="bad"),  # type: ignore[arg-type]
        ],
    )
    def test_benzinga_date_page_and_pagesize_validation_is_local(self, call):
        session = MagicMock(spec=requests.Session)
        adapter = self._adapter(session)

        resp = call(adapter)

        assert not resp.ok
        assert resp.error.error_type == "validation"
        session.get.assert_not_called()

    @pytest.mark.parametrize(
        ("method_name", "payload"),
        [
            (
                "get_earnings",
                {"earnings": [{"ticker": "ACME", "date": "bad"}, {"ticker": "ACME", "date": "2026-05-27"}]},
            ),
            (
                "get_guidance",
                {"guidance": [{"ticker": "ACME", "date": "2026-13-45"}, {"ticker": "ACME", "date": "2026-05-27"}]},
            ),
            (
                "get_ratings",
                {"ratings": [{"ticker": "ACME", "date": ""}, {"ticker": "ACME", "date": "2026-05-27"}]},
            ),
            (
                "get_offerings",
                {"offerings": [{"ticker": "ACME", "date": 20260527}, {"ticker": "ACME", "date": "2026-05-27"}]},
            ),
            (
                "get_dividends",
                {"dividends": [{"ticker": "ACME", "date": "bad"}, {"ticker": "ACME", "ex_dividend_date": "2026-05-27"}]},
            ),
        ],
    )
    def test_benzinga_calendar_provider_row_dates_are_validated(
        self, method_name, payload
    ):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, payload)
        adapter = self._adapter(session)

        resp = getattr(adapter, method_name)("ACME")

        assert resp.ok
        assert len(resp.data) == 1
        assert resp.lineage.data_quality_flags["raw_rows"] == 2
        assert resp.lineage.data_quality_flags["parsed_rows"] == 1
        assert resp.lineage.data_quality_flags["skipped_rows"] == 1

    @pytest.mark.parametrize(
        ("method_name", "payload"),
        [
            (
                "get_insider_filings",
                {
                    "data": [
                        {"id": "bad", "company_symbol": "ACME", "filing_date": "bad"},
                        {
                            "id": "good",
                            "company_symbol": "ACME",
                            "filing_date": "2026-05-27T12:00:00Z",
                        },
                    ]
                },
            ),
            (
                "get_insider_transactions",
                {
                    "data": [
                        {
                            "transaction_id": "bad",
                            "company_symbol": "ACME",
                            "filing_date": "bad",
                        },
                        {
                            "transaction_id": "good",
                            "company_symbol": "ACME",
                            "filing_date": "2026-05-27T12:00:00Z",
                        },
                    ]
                },
            ),
        ],
    )
    def test_benzinga_insider_provider_row_dates_are_validated(
        self, method_name, payload
    ):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, payload)
        adapter = self._adapter(session)

        resp = getattr(adapter, method_name)("ACME")

        assert resp.ok
        assert len(resp.data) == 1
        assert resp.lineage.data_quality_flags["raw_rows"] == 2
        assert resp.lineage.data_quality_flags["parsed_rows"] == 1
        assert resp.lineage.data_quality_flags["skipped_rows"] == 1

    def test_benzinga_calendar_future_updated_is_nulled_without_rejecting_event_date(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "earnings": [
                    {
                        "ticker": "ACME",
                        "date": "2099-01-01",
                        "updated": 9999999999,
                    }
                ]
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_earnings(
            "ACME",
            asof=datetime(2026, 5, 28, tzinfo=timezone.utc),
        )

        assert resp.ok
        assert resp.data[0].date == "2099-01-01"
        assert resp.data[0].updated is None
        flags = resp.lineage.data_quality_flags
        assert flags["knowledge_timestamp_warning_rows"] == 1
        assert flags["knowledge_timestamp_warning_types"] == {
            "calendar_updated_future": 1
        }

    def test_benzinga_insider_future_filing_knowledge_timestamp_is_nulled_and_flagged(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "data": [
                    {
                        "id": "filing-1",
                        "company_symbol": "ACME",
                        "filing_date": "2099-01-01T00:00:00Z",
                        "accepted": "2099-01-01T00:01:00Z",
                        "updated": "2099-01-02T00:00:00Z",
                    }
                ]
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_insider_filings(
            "ACME",
            asof=datetime(2026, 5, 28, tzinfo=timezone.utc),
        )

        assert resp.ok
        assert resp.data[0].filing_date is None
        assert resp.data[0].updated is None
        flags = resp.lineage.data_quality_flags
        assert flags["knowledge_timestamp_warning_rows"] == 1
        assert flags["knowledge_timestamp_warning_types"] == {
            "insider_accepted_future": 1,
            "insider_filing_date_future": 1,
            "insider_updated_future": 1,
        }

    def test_benzinga_ma_future_updated_is_nulled_and_flagged(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "ma": [
                    {
                        "id": "deal-1",
                        "target_ticker": "ACME",
                        "date_announced": "2099-01-01T00:00:00Z",
                        "updated": 9999999999,
                    }
                ]
            },
        )
        adapter = self._adapter(session)

        resp = adapter.get_mergers_acquisitions(
            "ACME",
            asof=datetime(2026, 5, 28, tzinfo=timezone.utc),
        )

        assert resp.ok
        assert resp.data[0].updated is None
        flags = resp.lineage.data_quality_flags
        assert flags["knowledge_timestamp_warning_rows"] == 1
        assert flags["knowledge_timestamp_warning_types"] == {
            "ma_publication_future": 1,
            "ma_updated_future": 1,
        }

    @pytest.mark.parametrize(
        ("method_name", "payload"),
        [
            ("get_earnings", {"earnings": [{"ticker": "ACME", "date": "2026-05-27", "revenue": "-1"}]}),
            ("get_guidance", {"guidance": [{"ticker": "ACME", "date": "2026-05-27", "revenue_guidance_min": "-1"}]}),
            ("get_ratings", {"ratings": [{"ticker": "ACME", "date": "2026-05-27", "pt_current": "-1"}]}),
            ("get_offerings", {"offerings": [{"ticker": "ACME", "date": "2026-05-27", "number_shares": "-1"}]}),
            ("get_dividends", {"dividends": [{"ticker": "ACME", "date": "2026-05-27", "dividend": "-0.10"}]}),
            ("get_insider_transactions", {"data": [{"transaction_id": "tx-1", "company_symbol": "ACME", "shares": "-1"}]}),
            ("get_mergers_acquisitions", {"ma": [{"id": "deal-1", "target_ticker": "ACME", "deal_size": "-1"}]}),
        ],
    )
    def test_benzinga_negative_impossible_economics_are_skipped(
        self, method_name, payload
    ):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, payload)
        adapter = self._adapter(session)

        resp = getattr(adapter, method_name)("ACME")

        assert resp.ok
        assert resp.data == []
        assert resp.lineage.data_quality_flags["raw_rows"] == 1
        assert resp.lineage.data_quality_flags["parsed_rows"] == 0
        assert resp.lineage.data_quality_flags["skipped_rows"] == 1
        assert resp.lineage.data_quality_flags["all_rows_skipped"] is True

    def test_benzinga_raw_nested_payload_is_isolated_for_news_and_insider(self):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            200,
            {
                "news": [
                    {
                        "id": "news-1",
                        "stocks": [{"name": "ACME"}],
                        "channels": [{"name": "News"}],
                    }
                ]
            },
        )
        adapter = self._adapter(session)

        news = adapter.get_news("ACME").data[0]
        news.stocks[0]["name"] = "MUTATED_PARSED"
        assert news.raw["stocks"][0]["name"] == "ACME"

        session.get.return_value = _mock_response(
            200,
            {
                "data": [
                    {
                        "id": "filing-1",
                        "accession_number": "0001",
                        "company_symbol": "ACME",
                        "filing_date": "2026-05-27T12:00:00Z",
                        "owner": {"insider_name": "Original"},
                        "transactions": [{"transaction_code": "P"}],
                    }
                ]
            },
        )
        filing = adapter.get_insider_filings("ACME").data[0]
        filing.owner["insider_name"] = "MUTATED_OWNER"
        filing.transactions[0]["transaction_code"] = "S"
        assert filing.raw["owner"]["insider_name"] == "Original"
        assert filing.raw["transactions"][0]["transaction_code"] == "P"

    @pytest.mark.parametrize(
        ("method_name", "payload1", "payload2"),
        [
            ("get_news", {"news": [{"id": "news-1"}]}, {"news": [{"id": "news-2"}]}),
            ("get_wiims", {"news": [{"id": "wiim-1"}]}, {"news": [{"id": "wiim-2"}]}),
            ("get_earnings", {"earnings": [{"ticker": "ACME", "date": "2026-05-27"}]}, {"earnings": [{"ticker": "ACME", "date": "2026-05-28"}]}),
            ("get_guidance", {"guidance": [{"ticker": "ACME", "date": "2026-05-27"}]}, {"guidance": [{"ticker": "ACME", "date": "2026-05-28"}]}),
            ("get_ratings", {"ratings": [{"ticker": "ACME", "date": "2026-05-27"}]}, {"ratings": [{"ticker": "ACME", "date": "2026-05-28"}]}),
            ("get_offerings", {"offerings": [{"ticker": "ACME", "date": "2026-05-27"}]}, {"offerings": [{"ticker": "ACME", "date": "2026-05-28"}]}),
            ("get_dividends", {"dividends": [{"ticker": "ACME", "date": "2026-05-27"}]}, {"dividends": [{"ticker": "ACME", "date": "2026-05-28"}]}),
            ("get_insider_filings", {"data": [{"id": "filing-1", "company_symbol": "ACME", "filing_date": "2026-05-27T12:00:00Z"}]}, {"data": [{"id": "filing-2", "company_symbol": "ACME", "filing_date": "2026-05-27T12:00:00Z"}]}),
            ("get_insider_transactions", {"data": [{"transaction_id": "tx-1", "company_symbol": "ACME", "filing_date": "2026-05-27T12:00:00Z"}]}, {"data": [{"transaction_id": "tx-2", "company_symbol": "ACME", "filing_date": "2026-05-27T12:00:00Z"}]}),
            ("get_mergers_acquisitions", {"ma": [{"id": "deal-1", "target_ticker": "ACME"}]}, {"ma": [{"id": "deal-2", "target_ticker": "ACME"}]}),
        ],
    )
    def test_benzinga_lineage_hash_stability_and_change_all_methods(
        self, method_name, payload1, payload2
    ):
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, payload1)
        adapter = self._adapter(session)

        resp1 = getattr(adapter, method_name)("ACME")
        resp2 = getattr(adapter, method_name)("ACME")
        session.get.return_value = _mock_response(200, payload2)
        resp3 = getattr(adapter, method_name)("ACME")

        assert resp1.ok
        assert resp2.ok
        assert resp3.ok
        assert resp1.lineage.raw_payload_hash == resp2.lineage.raw_payload_hash
        assert resp1.lineage.raw_payload_hash != resp3.lineage.raw_payload_hash

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
