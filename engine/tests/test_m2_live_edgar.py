"""Opt-in live SEC EDGAR regression tests for M2 Form 4 ingestion."""

from __future__ import annotations

from datetime import date, datetime, timezone
import os

import pytest

from alpha.data.config import ConfigError, SecEdgarConfig
from alpha.data.edgar import SecEdgarAdapter, SecEdgarFiling
from alpha.runtime_env import load_runtime_env


LIVE_FORM4_TICKERS = ("CRMD", "INVE", "AMPG")


def test_live_sec_form4_transactions_parse_recent_xsl_primary_documents():
    load_runtime_env()
    if os.environ.get("ALPHA_RUN_LIVE_EDGAR_TESTS") != "1":
        pytest.skip("set ALPHA_RUN_LIVE_EDGAR_TESTS=1 to run live SEC EDGAR proof")

    try:
        adapter = SecEdgarAdapter(SecEdgarConfig.from_env())
    except ConfigError as exc:
        pytest.skip(str(exc))

    asof = datetime.now(timezone.utc)
    for ticker in LIVE_FORM4_TICKERS:
        ticker_resp = adapter.get_company_ticker(ticker, asof=asof)
        assert ticker_resp.ok, f"{ticker} CIK lookup failed: {ticker_resp.error}"
        filing = _recent_form4_filing(adapter, ticker, ticker_resp.data.cik_str, asof)

        tx_resp = adapter.get_form4_transactions(
            ticker_resp.data.cik_str,
            from_date=filing.filing_date,
            to_date=filing.filing_date,
            asof=asof,
        )

        assert tx_resp.ok, (
            f"{ticker} Form 4 parse failed for {filing.accession_number} "
            f"{filing.primary_document}: {tx_resp.error}"
        )
        rows = list(tx_resp.data or [])
        codes = sorted({
            str(row.transaction_code)
            for row in rows
            if row.transaction_code is not None
        })
        assert rows, f"{ticker} Form 4 produced no parsed transaction rows"
        assert all(row.accession_number for row in rows)
        assert codes, f"{ticker} Form 4 transaction code distribution is empty"


def _recent_form4_filing(
    adapter: SecEdgarAdapter,
    ticker: str,
    cik: str,
    asof: datetime,
) -> SecEdgarFiling:
    filings_resp = adapter.get_form4_filings(
        cik,
        from_date=date(2025, 1, 1),
        to_date=asof.date(),
        asof=asof,
    )
    assert filings_resp.ok, f"{ticker} Form 4 filing lookup failed: {filings_resp.error}"
    for filing in filings_resp.data or []:
        if filing.filing_date is not None and filing.primary_document:
            return filing
    raise AssertionError(f"{ticker} has no recent Form 4 filing with a primary document")
