"""
Job orchestration tests.

  - Successful job run persists evidence_job_runs as finished.
  - Failed job run persists evidence_job_runs as failed with error_json.
  - Universe job records included and excluded symbols.
  - Universe snapshots link to job_run_id and data lineage.
  - Deterministic output hashes.
  - Hardened filter rules per Data-Sourcing-Audit.md.
  - Sliced FMP universe source per MeasurementSpine.md section 1.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import List
from unittest.mock import MagicMock

import pytest

from alpha.data.contracts import (
    AdapterResponse,
    LineageMeta,
    ProviderError,
    stable_hash,
)
from alpha.data.fmp import FmpAdapter, FmpScreenerResult
from alpha.data.universe import SliceDiagnostic, SlicedUniverseFetcher
from alpha.db.models import DataLineage, EvidenceJob, EvidenceJobRun, UniverseSnapshot
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.runner import run_job
from alpha.jobs.universe_builder import (
    ALLOWED_EXCHANGES,
    MCAP_MAX,
    MCAP_MIN,
    PRICE_MIN,
    UniverseBuilderJob,
    _classify,
    _is_non_common_symbol,
)


def _ts():
    return datetime(2026, 5, 20, 14, 30, 0, tzinfo=timezone.utc)


def _mock_lineage():
    return LineageMeta(
        provider="FMP",
        endpoint="/stable/company-screener",
        request_timestamp=_ts(),
        asof_timestamp=_ts(),
        raw_payload_hash="abcdef1234567890",
        source_authority="mock",
    )


def _stock(
    symbol="ACME",
    market_cap=75_000_000,
    price=5.0,
    exchange="NASDAQ",
    country="US",
    is_etf=False,
    is_actively_trading=True,
    **kw,
) -> FmpScreenerResult:
    return FmpScreenerResult(
        symbol=symbol,
        company_name=kw.get("company_name", f"{symbol} Corp"),
        market_cap=market_cap,
        price=price,
        exchange=exchange,
        country=country,
        is_etf=is_etf,
        is_actively_trading=is_actively_trading,
    )


def _mock_screener_data() -> List[FmpScreenerResult]:
    return [
        _stock("INCL1", market_cap=75_000_000, price=5.0, exchange="NASDAQ"),
        _stock("INCL2", market_cap=150_000_000, price=10.0, exchange="NYSE"),
        _stock("ETF1", market_cap=80_000_000, price=25.0, exchange="NASDAQ", is_etf=True),
        _stock("TINY", market_cap=5_000_000, price=0.50, exchange="NASDAQ"),
        _stock("HUGE", market_cap=500_000_000, price=50.0, exchange="NYSE"),
        _stock("DEAD", market_cap=60_000_000, price=4.0, exchange="NASDAQ", is_actively_trading=False),
        _stock("FRGN", market_cap=90_000_000, price=7.0, exchange="NASDAQ", country="CA"),
    ]


def _mock_screener_response():
    return AdapterResponse(
        data=_mock_screener_data(),
        lineage=_mock_lineage(),
    )


# --- Helpers: trivial jobs ---

class SuccessJob(BaseJob):
    job_name = "test_success"
    job_type = "test"

    def run(self, ctx: JobContext) -> JobResult:
        return JobResult(
            status="finished",
            metrics={"rows": 42},
            input_hashes={"input": "hash_in"},
            output_hashes={"output": "hash_out"},
        )


class FailingJob(BaseJob):
    job_name = "test_failing"
    job_type = "test"

    def run(self, ctx: JobContext) -> JobResult:
        raise RuntimeError("something broke")


class TransactionFailingJob(BaseJob):
    job_name = "test_transaction_failing"
    job_type = "test"

    def __init__(self, session):
        self._session = session

    def run(self, ctx: JobContext) -> JobResult:
        self._session.add(EvidenceJobRun(job_run_id="bad-run-without-job"))
        self._session.flush()
        return JobResult(status="finished")


class ExplicitFailJob(BaseJob):
    job_name = "test_explicit_fail"
    job_type = "test"

    def run(self, ctx: JobContext) -> JobResult:
        return JobResult(
            status="failed",
            errors=[{"stage": "processing", "message": "bad data"}],
            input_hashes={"input": "hash_in"},
        )


# -----------------------------------------------------------------------
# Test runner: success path
# -----------------------------------------------------------------------

class TestRunnerSuccess:
    def test_finished_run_persisted(self, db_session):
        result = run_job(db_session, SuccessJob(), params={"key": "val"})

        assert result.ok
        assert result.metrics["rows"] == 42

        runs = db_session.query(EvidenceJobRun).all()
        assert len(runs) == 1
        assert runs[0].run_status == "finished"
        assert json.loads(runs[0].input_hashes) == {"input": "hash_in"}
        assert json.loads(runs[0].output_hashes) == {"output": "hash_out"}
        assert json.loads(runs[0].metric_json) == {"rows": 42}

    def test_job_row_created(self, db_session):
        run_job(db_session, SuccessJob())

        jobs = db_session.query(EvidenceJob).all()
        assert len(jobs) == 1
        assert jobs[0].job_name == "test_success"
        assert jobs[0].job_type == "test"

    def test_reuses_existing_job_row(self, db_session):
        run_job(db_session, SuccessJob())
        run_job(db_session, SuccessJob())

        jobs = db_session.query(EvidenceJob).all()
        assert len(jobs) == 1
        runs = db_session.query(EvidenceJobRun).all()
        assert len(runs) == 2


# -----------------------------------------------------------------------
# Test runner: failure paths
# -----------------------------------------------------------------------

class TestRunnerFailure:
    def test_exception_persisted_as_failed(self, db_session):
        result = run_job(db_session, FailingJob())

        assert not result.ok
        assert result.status == "failed"
        assert any("something broke" in e.get("exception", "") for e in result.errors)

        runs = db_session.query(EvidenceJobRun).all()
        assert len(runs) == 1
        assert runs[0].run_status == "failed"
        err = json.loads(runs[0].error_json)
        assert "something broke" in err["errors"][0]["exception"]

    def test_transaction_failure_still_records_failed_run(self, db_session):
        result = run_job(db_session, TransactionFailingJob(db_session))

        assert not result.ok
        assert result.status == "failed"
        assert any("IntegrityError" in e.get("traceback", "") for e in result.errors)

        runs = (
            db_session.query(EvidenceJobRun)
            .filter(EvidenceJobRun.job_id.isnot(None))
            .all()
        )
        assert len(runs) == 1
        assert runs[0].run_status == "failed"
        err = json.loads(runs[0].error_json)
        assert "IntegrityError" in err["errors"][0]["traceback"]

    def test_explicit_fail_persisted(self, db_session):
        result = run_job(db_session, ExplicitFailJob())

        assert not result.ok
        runs = db_session.query(EvidenceJobRun).all()
        assert runs[0].run_status == "failed"
        err = json.loads(runs[0].error_json)
        assert err["errors"][0]["message"] == "bad data"


# -----------------------------------------------------------------------
# Test universe builder: inclusion/exclusion
# -----------------------------------------------------------------------

class TestUniverseBuilder:
    def test_included_and_excluded_symbols(self, db_session):
        resp = _mock_screener_response()
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job)

        assert result.ok
        assert result.metrics["raw_count"] == 7
        assert result.metrics["included"] == 2
        assert result.metrics["excluded"] == 5

        snaps = db_session.query(UniverseSnapshot).all()
        assert len(snaps) == 7

        included = [s for s in snaps if s.operating_universe_inclusion]
        excluded = [s for s in snaps if not s.operating_universe_inclusion]
        assert len(included) == 2
        assert len(excluded) == 5

        included_tickers = {s.ticker for s in included}
        assert included_tickers == {"INCL1", "INCL2"}

    def test_exclusion_reasons(self, db_session):
        resp = _mock_screener_response()
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        run_job(db_session, job)

        snaps = db_session.query(UniverseSnapshot).all()
        reasons = {s.ticker: s.exclusion_reason for s in snaps if s.exclusion_reason}

        assert reasons["ETF1"] == "etf"
        assert "mcap_below" in reasons["TINY"]
        assert "mcap_above" in reasons["HUGE"]
        assert reasons["DEAD"] == "not_actively_trading"
        assert "country" in reasons["FRGN"]

    def test_exclusion_counts_in_metrics(self, db_session):
        resp = _mock_screener_response()
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job)

        counts = result.metrics["exclusion_counts"]
        assert counts["etf"] == 1
        assert counts["not_actively_trading"] == 1
        assert "country:CA" in counts

    def test_snapshots_link_to_job_run(self, db_session):
        resp = _mock_screener_response()
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        run_job(db_session, job)

        runs = db_session.query(EvidenceJobRun).all()
        assert len(runs) == 1
        run_id = runs[0].job_run_id

        snaps = db_session.query(UniverseSnapshot).all()
        for snap in snaps:
            assert snap.job_run_id == run_id

    def test_data_lineage_recorded(self, db_session):
        resp = _mock_screener_response()
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        run_job(db_session, job)

        lineage_rows = db_session.query(DataLineage).all()
        assert len(lineage_rows) == 1
        assert lineage_rows[0].provider == "FMP"
        assert lineage_rows[0].endpoint == "/stable/company-screener"
        assert lineage_rows[0].job_run_id is not None
        assert lineage_rows[0].raw_payload_hash == resp.lineage.raw_payload_hash
        assert lineage_rows[0].source_authority == "mock"

        snaps = db_session.query(UniverseSnapshot).all()
        assert {s.source_lineage_hash for s in snaps} == {
            resp.lineage.raw_payload_hash
        }

    def test_deterministic_output_hash(self, db_session):
        resp = _mock_screener_response()
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job)
        hash1 = result.output_hashes["universe_snapshots"]

        from sqlalchemy import create_engine, event
        from sqlalchemy.orm import sessionmaker
        from alpha.db.models import Base

        engine2 = create_engine("sqlite:///:memory:")
        event.listen(engine2, "connect", lambda c, _: c.cursor().execute("PRAGMA foreign_keys=ON") or c.cursor().close())
        Base.metadata.create_all(engine2)
        session2 = sessionmaker(bind=engine2)()

        job2 = UniverseBuilderJob(session=session2, screener_response=resp)
        result2 = run_job(session2, job2)
        hash2 = result2.output_hashes["universe_snapshots"]

        assert hash1 == hash2
        session2.close()
        engine2.dispose()

    def test_input_hashes_present(self, db_session):
        resp = _mock_screener_response()
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job)

        assert "screener" in result.input_hashes
        assert result.input_hashes["screener"] == resp.lineage.raw_payload_hash

    def test_screener_error_produces_failed_run(self, db_session):
        error_resp = AdapterResponse(
            data=None,
            lineage=_mock_lineage(),
            error=ProviderError(
                provider="FMP",
                endpoint="/stable/company-screener",
                status_code=500,
                error_type="http",
                message="Internal Server Error",
                retryable=True,
            ),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=error_resp)
        result = run_job(db_session, job)

        assert not result.ok
        assert result.status == "failed"

        runs = db_session.query(EvidenceJobRun).all()
        assert runs[0].run_status == "failed"

    def test_market_cap_fields_captured(self, db_session):
        resp = _mock_screener_response()
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        run_job(db_session, job)

        snap = (
            db_session.query(UniverseSnapshot)
            .filter(UniverseSnapshot.ticker == "INCL1")
            .one()
        )
        assert snap.market_cap == 75_000_000
        assert snap.price == 5.0
        assert snap.source_provider == "FMP"
        assert snap.primary_exchange == "NASDAQ"

    def test_symbol_and_exchange_normalized_in_snapshot(self, db_session):
        resp = AdapterResponse(
            data=[
                _stock(
                    symbol=" acme ",
                    market_cap=75_000_000,
                    price=5.0,
                    exchange=" nasdaq ",
                )
            ],
            lineage=_mock_lineage(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        run_job(db_session, job)

        snap = db_session.query(UniverseSnapshot).one()
        assert snap.operating_universe_inclusion is True
        assert snap.ticker == "ACME"
        assert snap.primary_exchange == "NASDAQ"

    def test_slice_diagnostics_in_metrics(self, db_session):
        resp = _mock_screener_response()
        diags = [
            {"lower": 30_000_000, "upper": 40_000_000, "returned_count": 50, "hit_limit": False},
            SliceDiagnostic(
                lower=40_000_000,
                upper=50_000_000,
                returned_count=100,
                hit_limit=True,
                query_lower=39_999_000,
                query_upper=50_001_000,
                subdivided=True,
            ),
        ]
        job = UniverseBuilderJob(
            session=db_session, screener_response=resp, slice_diagnostics=diags,
        )
        result = run_job(db_session, job)

        assert result.metrics["slice_count"] == 2
        assert result.metrics["slice_limit_hits"] == 1
        assert result.metrics["slice_subdivision_count"] == 1
        assert result.metrics["slice_limit_exhausted"] is False
        assert result.metrics["slice_diagnostics"][0] == diags[0]
        assert result.metrics["slice_diagnostics"][1]["query_lower"] == 39_999_000


# -----------------------------------------------------------------------
# Test hardened filter rules
# -----------------------------------------------------------------------

class TestHardenedFilter:
    """Per Data-Sourcing-Audit.md Universe Filter section."""

    # --- Hard include rules ---

    def test_valid_stock_included(self):
        included, reason = _classify(_stock())
        assert included is True
        assert reason is None

    def test_etf_excluded(self):
        included, reason = _classify(_stock(is_etf=True))
        assert not included
        assert reason == "etf"

    def test_unknown_etf_status_fails_closed(self):
        included, reason = _classify(_stock(is_etf=None))
        assert not included
        assert reason == "etf_status_missing_or_invalid"

    def test_string_etf_status_fails_closed(self):
        included, reason = _classify(_stock(is_etf="false"))
        assert not included
        assert reason == "etf_status_missing_or_invalid"

    def test_inactive_excluded(self):
        included, reason = _classify(_stock(is_actively_trading=False))
        assert not included
        assert reason == "not_actively_trading"

    def test_none_active_treated_as_inactive(self):
        included, reason = _classify(_stock(is_actively_trading=None))
        assert not included
        assert reason == "not_actively_trading"

    def test_non_us_excluded(self):
        included, reason = _classify(_stock(country="CA"))
        assert not included
        assert reason == "country:CA"

    def test_otc_exchange_excluded(self):
        included, reason = _classify(_stock(exchange="OTC"))
        assert not included
        assert reason == "exchange:OTC"

    def test_pnk_exchange_excluded(self):
        included, reason = _classify(_stock(exchange="PNK"))
        assert not included
        assert reason == "exchange:PNK"

    def test_foreign_exchange_excluded(self):
        included, reason = _classify(_stock(exchange="XETRA"))
        assert not included
        assert reason == "exchange:XETRA"

    def test_none_exchange_excluded(self):
        included, reason = _classify(_stock(exchange=None))
        assert not included
        assert reason == "exchange_missing"

    def test_exchange_normalized_before_classification(self):
        assert _classify(_stock(exchange=" nasdaq "))[0]
        assert _classify(_stock(exchange="nyse"))[0]
        assert _classify(_stock(exchange=" AmEx "))[0]

    def test_missing_country_reason_is_explicit(self):
        included, reason = _classify(_stock(country=None))
        assert not included
        assert reason == "country_missing"

    def test_nasdaq_included(self):
        assert _classify(_stock(exchange="NASDAQ"))[0]

    def test_nyse_included(self):
        assert _classify(_stock(exchange="NYSE"))[0]

    def test_amex_included(self):
        assert _classify(_stock(exchange="AMEX"))[0]

    def test_mcap_below_excluded(self):
        included, reason = _classify(_stock(market_cap=29_000_000))
        assert not included
        assert reason == "mcap_below_30000000"

    def test_mcap_above_excluded(self):
        included, reason = _classify(_stock(market_cap=201_000_000))
        assert not included
        assert reason == "mcap_above_200000000"

    def test_mcap_missing_excluded(self):
        included, reason = _classify(_stock(market_cap=None))
        assert not included
        assert reason == "mcap_missing_or_invalid"

    def test_mcap_nan_excluded(self):
        included, reason = _classify(_stock(market_cap=float("nan")))
        assert not included
        assert reason == "mcap_missing_or_invalid"

    def test_mcap_inf_excluded(self):
        included, reason = _classify(_stock(market_cap=float("inf")))
        assert not included
        assert reason == "mcap_missing_or_invalid"

    def test_mcap_string_excluded_without_crash(self):
        included, reason = _classify(_stock(market_cap="75000000"))
        assert not included
        assert reason == "mcap_missing_or_invalid"

    def test_mcap_decimal_accepted(self):
        included, reason = _classify(_stock(market_cap=Decimal("75000000")))
        assert included is True
        assert reason is None

    def test_price_below_3_excluded(self):
        included, reason = _classify(_stock(price=2.99))
        assert not included
        assert reason == "price_below_3"

    def test_price_exactly_3_included(self):
        assert _classify(_stock(price=3.0))[0]

    def test_price_missing_excluded(self):
        included, reason = _classify(_stock(price=None))
        assert not included
        assert reason == "price_missing_or_invalid"

    def test_price_nan_excluded(self):
        included, reason = _classify(_stock(price=float("nan")))
        assert not included
        assert reason == "price_missing_or_invalid"

    def test_price_string_excluded_without_crash(self):
        included, reason = _classify(_stock(price="5.0"))
        assert not included
        assert reason == "price_missing_or_invalid"

    def test_price_decimal_accepted(self):
        included, reason = _classify(_stock(price=Decimal("5.0")))
        assert included is True
        assert reason is None

    # --- Conservative symbol cleanup ---

    def test_dot_separator_excluded(self):
        included, reason = _classify(_stock(symbol="ABCD.W"))
        assert not included
        assert reason == "non_common_symbol_separator"

    def test_dash_separator_excluded(self):
        included, reason = _classify(_stock(symbol="ABCD-P"))
        assert not included
        assert reason == "non_common_symbol_separator"

    def test_ws_suffix_excluded(self):
        included, reason = _classify(_stock(symbol="ABCWS"))
        assert not included
        assert reason == "non_common_symbol_suffix"

    def test_wt_suffix_excluded(self):
        included, reason = _classify(_stock(symbol="ABCWT"))
        assert not included
        assert reason == "non_common_symbol_suffix"

    def test_four_char_ws_wt_tickers_kept(self):
        """Avoid false positives like NEWS/NEWT; WS/WT suffix needs 5+ chars."""
        assert _classify(_stock(symbol="NEWS"))[0]
        assert _classify(_stock(symbol="NEWT"))[0]
        assert _classify(_stock(symbol="LAWS"))[0]

    def test_five_char_w_excluded(self):
        included, reason = _classify(_stock(symbol="ABCDW"))
        assert not included
        assert reason == "non_common_symbol_suffix"

    def test_five_char_u_excluded(self):
        included, reason = _classify(_stock(symbol="ABCDU"))
        assert not included
        assert reason == "non_common_symbol_suffix"

    # --- Non-exclusions: no broad final-letter rule ---

    def test_watt_kept(self):
        """4-letter symbol ending T — ordinary ticker, not a suffix rule."""
        assert _classify(_stock(symbol="WATT"))[0]

    def test_koss_kept(self):
        assert _classify(_stock(symbol="KOSS"))[0]

    def test_loan_kept(self):
        assert _classify(_stock(symbol="LOAN"))[0]

    def test_four_letter_w_kept(self):
        """4-letter symbol ending W is NOT excluded (only 5-char W is)."""
        assert _classify(_stock(symbol="ABCW"))[0]

    def test_four_letter_u_kept(self):
        assert _classify(_stock(symbol="ABCU"))[0]

    def test_four_letter_r_kept(self):
        assert _classify(_stock(symbol="ABCR"))[0]

    def test_four_letter_p_kept(self):
        assert _classify(_stock(symbol="ABCP"))[0]

    def test_four_letter_x_kept(self):
        assert _classify(_stock(symbol="ABCX"))[0]

    def test_five_letter_x_excluded_as_fund_proxy(self):
        """Until profile type exists, 5-char X is the mutual-fund cleanup rule."""
        included, reason = _classify(_stock(symbol="ABCDX"))
        assert not included
        assert reason == "non_common_symbol_suffix"

    def test_five_letter_r_kept(self):
        assert _classify(_stock(symbol="ABCDR"))[0]

    def test_five_letter_p_kept(self):
        assert _classify(_stock(symbol="ABCDP"))[0]

    def test_no_broad_final_letter_exclusion(self):
        """Behaviorally prove no broad final-letter exclusion exists."""
        for suffix in ("W", "U", "R", "P", "X"):
            sym = f"TST{suffix}"
            included, _ = _classify(_stock(symbol=sym))
            assert included, f"{sym} should be kept (4-letter, ends {suffix})"


# -----------------------------------------------------------------------
# Test _is_non_common_symbol directly
# -----------------------------------------------------------------------

class TestNonCommonSymbol:
    def test_empty_excluded(self):
        assert _is_non_common_symbol("") == (True, "non_common_symbol_separator")

    def test_whitespace_excluded(self):
        assert _is_non_common_symbol("  ") == (True, "non_common_symbol_separator")

    def test_normal_kept(self):
        assert _is_non_common_symbol("ACME") == (False, None)

    def test_dot(self):
        assert _is_non_common_symbol("BRK.B")[0]

    def test_dash(self):
        assert _is_non_common_symbol("FOO-A")[0]

    def test_case_insensitive_ws(self):
        assert _is_non_common_symbol("abcws")[0]

    def test_three_char_ws_kept(self):
        """Short common symbols ending WS are not warrant suffixes."""
        assert _is_non_common_symbol("AWS") == (False, None)


# -----------------------------------------------------------------------
# Test sliced universe fetcher
# -----------------------------------------------------------------------

def _make_mock_adapter(responses_by_range=None, default_response=None):
    """Create a mock FmpAdapter where get_stock_screener returns per-range data."""
    adapter = MagicMock(spec=FmpAdapter)

    def mock_screener(market_cap_min=0, market_cap_max=0, country=None, is_etf=None, limit=1000):
        key = (market_cap_min, market_cap_max)
        if responses_by_range and key in responses_by_range:
            return responses_by_range[key]
        if default_response is not None:
            return default_response
        return AdapterResponse(
            data=[],
            lineage=LineageMeta(
                provider="FMP",
                endpoint="/stable/company-screener",
                request_timestamp=_ts(),
                asof_timestamp=_ts(),
                raw_payload_hash=stable_hash({"range": key}),
                source_authority="FMP_Ultimate",
            ),
        )

    adapter.get_stock_screener.side_effect = mock_screener
    return adapter


def _ok_response(stocks, key=(0, 0)):
    return AdapterResponse(
        data=stocks,
        lineage=LineageMeta(
            provider="FMP",
            endpoint="/stable/company-screener",
            request_timestamp=_ts(),
            asof_timestamp=_ts(),
            raw_payload_hash=stable_hash({"range": key, "count": len(stocks)}),
            source_authority="FMP_Ultimate",
        ),
    )


class TestSlicedUniverseFetcher:
    def test_calls_once_per_slice(self):
        """17 slices from 30M to 200M in 10M increments."""
        adapter = _make_mock_adapter()
        fetcher = SlicedUniverseFetcher(adapter)
        result = fetcher.fetch()

        assert result.response.ok
        assert result.slice_count == 17
        assert adapter.get_stock_screener.call_count == 17

    def test_unions_and_dedups_symbols(self):
        """Same symbol in two slices appears once in results."""
        responses = {
            (29_999_000, 40_001_000): _ok_response(
                [_stock("ACME", market_cap=35_000_000)], (30_000_000, 40_000_000),
            ),
            (39_999_000, 50_001_000): _ok_response(
                [_stock("ACME", market_cap=35_000_000), _stock("BETA", market_cap=45_000_000)],
                (40_000_000, 50_000_000),
            ),
        }
        adapter = _make_mock_adapter(responses_by_range=responses)
        fetcher = SlicedUniverseFetcher(adapter)
        result = fetcher.fetch()

        assert result.response.ok
        symbols = {s.symbol for s in result.response.data}
        assert symbols == {"ACME", "BETA"}
        assert result.unique_raw_count == 2
        assert result.total_raw_count == 3
        assert result.duplicate_count == 1

    def test_unions_and_dedups_normalized_symbols(self):
        """Provider symbol whitespace/case variants collapse to one raw row."""
        responses = {
            (29_999_000, 40_001_000): _ok_response(
                [
                    _stock("ACME", market_cap=35_000_000),
                    _stock(" acme ", market_cap=35_100_000),
                ],
                (30_000_000, 40_000_000),
            ),
        }
        adapter = _make_mock_adapter(responses_by_range=responses)
        fetcher = SlicedUniverseFetcher(
            adapter, mcap_min=30_000_000, mcap_max=40_000_000,
            slice_width=10_000_000,
        )
        result = fetcher.fetch()

        assert result.response.ok
        assert result.unique_raw_count == 1
        assert result.duplicate_count == 1

    def test_limit_hit_triggers_subdivision(self):
        """A slice returning exactly limit results causes recursive subdivision."""
        limit = 5
        big_slice = [_stock(f"S{i}", market_cap=35_000_000 + i * 100) for i in range(limit)]
        sub1 = [_stock(f"S{i}", market_cap=35_000_000 + i * 100) for i in range(3)]
        sub2 = [_stock(f"S{i}", market_cap=35_000_000 + i * 100) for i in range(3, limit)]

        responses = {
            (29_999_000, 40_001_000): _ok_response(big_slice, (30_000_000, 40_000_000)),
            (29_999_000, 35_001_000): _ok_response(sub1, (30_000_000, 35_000_000)),
            (34_999_000, 40_001_000): _ok_response(sub2, (35_000_000, 40_000_000)),
        }
        adapter = _make_mock_adapter(responses_by_range=responses)
        fetcher = SlicedUniverseFetcher(
            adapter, mcap_min=30_000_000, mcap_max=40_000_000,
            slice_width=10_000_000, limit_per_slice=limit,
        )
        result = fetcher.fetch()

        assert result.response.ok
        assert result.unique_raw_count == limit
        assert result.total_raw_count == limit
        assert result.slice_limit_hits == 1
        assert result.slice_subdivision_count == 1
        assert result.slice_limit_exhausted is False
        assert any(d.hit_limit and d.subdivided for d in result.slice_diagnostics)
        assert any(d.lower == 30_000_000 and d.upper == 35_000_000 for d in result.slice_diagnostics)

    def test_min_width_limit_exhausted_fails_closed(self):
        """Subdivision at minimum width that still hits limit fails with slice_limit_exhausted."""
        limit = 3
        fat_slice = [_stock(f"S{i}") for i in range(limit)]

        # Every range returns exactly `limit` results, so subdivision
        # recurses until min_slice_width is reached and fails closed.
        default = _ok_response(fat_slice, (0, 0))
        adapter = _make_mock_adapter(default_response=default)
        fetcher = SlicedUniverseFetcher(
            adapter, mcap_min=30_000_000, mcap_max=40_000_000,
            slice_width=10_000_000, min_slice_width=1_000_000,
            limit_per_slice=limit,
        )
        result = fetcher.fetch()

        assert not result.response.ok
        assert result.response.error.error_type == "slice_limit_exhausted"
        assert result.slice_limit_hits >= 1
        assert result.slice_limit_exhausted is True

    def test_provider_error_propagates(self):
        """An FMP error on one slice fails the whole fetch."""
        error_resp = AdapterResponse(
            data=None,
            lineage=LineageMeta(
                provider="FMP", endpoint="/stable/company-screener",
                request_timestamp=_ts(), asof_timestamp=_ts(),
                raw_payload_hash="", source_authority="FMP_Ultimate",
            ),
            error=ProviderError(
                provider="FMP", endpoint="/stable/company-screener",
                status_code=500, error_type="http",
                message="Internal Server Error", retryable=True,
            ),
        )
        responses = {
            (29_999_000, 40_001_000): _ok_response([], (30_000_000, 40_000_000)),
            (39_999_000, 50_001_000): error_resp,
        }
        adapter = _make_mock_adapter(responses_by_range=responses)
        fetcher = SlicedUniverseFetcher(
            adapter, mcap_min=30_000_000, mcap_max=50_000_000,
            slice_width=10_000_000,
        )
        result = fetcher.fetch()

        assert not result.response.ok
        assert result.response.error.error_type == "http"

    def test_retryable_slice_error_retries_and_recovers(self):
        """A transient slice failure is retried before failing the whole snapshot."""
        error_resp = AdapterResponse(
            data=None,
            lineage=LineageMeta(
                provider="FMP", endpoint="/stable/company-screener",
                request_timestamp=_ts(), asof_timestamp=_ts(),
                raw_payload_hash="", source_authority="FMP_Ultimate",
            ),
            error=ProviderError(
                provider="FMP", endpoint="/stable/company-screener",
                status_code=429, error_type="rate_limit",
                message="rate limited", retryable=True,
            ),
        )
        success_resp = _ok_response(
            [_stock("ACME", market_cap=35_000_000)],
            (30_000_000, 40_000_000),
        )
        adapter = MagicMock(spec=FmpAdapter)
        adapter.get_stock_screener.side_effect = [error_resp, success_resp]
        sleeps = []
        fetcher = SlicedUniverseFetcher(
            adapter, mcap_min=30_000_000, mcap_max=40_000_000,
            slice_width=10_000_000, max_slice_retries=1,
            sleep_fn=sleeps.append,
        )
        result = fetcher.fetch()

        assert result.response.ok
        assert result.unique_raw_count == 1
        assert adapter.get_stock_screener.call_count == 2
        assert sleeps == [1.0]

    def test_retryable_slice_error_uses_exponential_backoff(self):
        error_resp = AdapterResponse(
            data=None,
            lineage=LineageMeta(
                provider="FMP", endpoint="/stable/company-screener",
                request_timestamp=_ts(), asof_timestamp=_ts(),
                raw_payload_hash="", source_authority="FMP_Ultimate",
            ),
            error=ProviderError(
                provider="FMP", endpoint="/stable/company-screener",
                status_code=429, error_type="rate_limit",
                message="rate limited", retryable=True,
            ),
        )
        success_resp = _ok_response(
            [_stock("ACME", market_cap=35_000_000)],
            (30_000_000, 40_000_000),
        )
        adapter = MagicMock(spec=FmpAdapter)
        adapter.get_stock_screener.side_effect = [
            error_resp, error_resp, success_resp,
        ]
        sleeps = []
        fetcher = SlicedUniverseFetcher(
            adapter, mcap_min=30_000_000, mcap_max=40_000_000,
            slice_width=10_000_000, max_slice_retries=2,
            retry_backoff_seconds=0.5, sleep_fn=sleeps.append,
        )
        result = fetcher.fetch()

        assert result.response.ok
        assert adapter.get_stock_screener.call_count == 3
        assert sleeps == [0.5, 1.0]

    def test_non_retryable_slice_error_does_not_retry(self):
        error_resp = AdapterResponse(
            data=None,
            lineage=LineageMeta(
                provider="FMP", endpoint="/stable/company-screener",
                request_timestamp=_ts(), asof_timestamp=_ts(),
                raw_payload_hash="", source_authority="FMP_Ultimate",
            ),
            error=ProviderError(
                provider="FMP", endpoint="/stable/company-screener",
                status_code=403, error_type="auth",
                message="auth failed", retryable=False,
            ),
        )
        adapter = MagicMock(spec=FmpAdapter)
        adapter.get_stock_screener.return_value = error_resp
        fetcher = SlicedUniverseFetcher(
            adapter, mcap_min=30_000_000, mcap_max=40_000_000,
            slice_width=10_000_000, max_slice_retries=2,
        )
        result = fetcher.fetch()

        assert not result.response.ok
        assert result.response.error.error_type == "auth"
        assert adapter.get_stock_screener.call_count == 1

    def test_rejects_negative_max_slice_retries(self):
        adapter = _make_mock_adapter()
        with pytest.raises(ValueError):
            SlicedUniverseFetcher(adapter, max_slice_retries=-1)

    def test_rejects_negative_retry_backoff_and_overlap(self):
        adapter = _make_mock_adapter()
        with pytest.raises(ValueError):
            SlicedUniverseFetcher(adapter, retry_backoff_seconds=-0.1)
        with pytest.raises(ValueError):
            SlicedUniverseFetcher(adapter, boundary_overlap=-1)

    def test_empty_slices_produce_zero_results(self):
        """All-empty slices produce a valid empty result."""
        adapter = _make_mock_adapter()
        fetcher = SlicedUniverseFetcher(adapter)
        result = fetcher.fetch()

        assert result.response.ok
        assert result.response.data == []
        assert result.unique_raw_count == 0

    def test_single_slice_no_subdivision(self):
        """A single narrow range that fits under limit returns directly."""
        stocks = [_stock("ONLY", market_cap=35_000_000)]
        responses = {
            (29_999_000, 40_001_000): _ok_response(stocks, (30_000_000, 40_000_000)),
        }
        adapter = _make_mock_adapter(responses_by_range=responses)
        fetcher = SlicedUniverseFetcher(
            adapter, mcap_min=30_000_000, mcap_max=40_000_000,
            slice_width=10_000_000,
        )
        result = fetcher.fetch()

        assert result.response.ok
        assert len(result.response.data) == 1
        assert result.slice_count == 1
        assert result.slice_diagnostics[0].hit_limit is False

    def test_fetcher_uses_overlap_and_no_classification_filters(self):
        """Server-side calls only carve by cap/limit; classification stays client-side."""
        adapter = _make_mock_adapter()
        fetcher = SlicedUniverseFetcher(
            adapter, mcap_min=30_000_000, mcap_max=40_000_000,
            slice_width=10_000_000,
        )
        result = fetcher.fetch()

        assert result.response.ok
        adapter.get_stock_screener.assert_called_once_with(
            market_cap_min=29_999_000,
            market_cap_max=40_001_000,
            country=None,
            is_etf=None,
            limit=1000,
        )
