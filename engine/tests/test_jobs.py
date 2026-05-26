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
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import List
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.exc import IntegrityError

from alpha.data.contracts import (
    AdapterResponse,
    LineageMeta,
    ProviderError,
    stable_hash,
)
from alpha.data.fmp import FmpAdapter, FmpScreenerResult
from alpha.data.universe import SliceDiagnostic, SlicedUniverseFetcher
from alpha.db.models import (
    CanonicalUniverseScan,
    DataLineage,
    EvidenceJob,
    EvidenceJobRun,
    SecurityProfile,
    SecurityProfileScanSnapshot,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.runner import run_job
from alpha.jobs.run_universe import _parse_args, _required_profile_symbols
from alpha.jobs.security_type import (
    COMMON_STOCK,
    MUTUAL_FUND,
    REFRESH_STATUS_ENRICHED,
    SPAC_OR_BLANK_CHECK,
)
from alpha.jobs.universe_builder import (
    ALLOWED_EXCHANGES,
    COUNTRY_REQUIRES_SECURITY_PROFILE_PREFIX,
    MCAP_MAX,
    MCAP_MIN,
    PRICE_MIN,
    UniverseBuilderJob,
    _classify,
    _dedupe_screener_rows,
    _is_non_common_symbol,
    _market_cap_bucket,
    _price_bucket,
    _price_floor_reason,
    _requires_security_profile,
    _upsert_canonical_universe_scan,
    get_canonical_universe_members,
    get_canonical_universe_scan,
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
        assert result.metrics["mcap_min"] == MCAP_MIN
        assert result.metrics["mcap_max"] == MCAP_MAX
        assert result.metrics["price_min"] == PRICE_MIN
        assert result.metrics["included_market_cap_bucket_counts"] == {
            "30m_100m": 1,
            "100m_200m": 1,
        }
        assert result.metrics["included_price_bucket_counts"] == {
            "5_plus": 2,
        }
        assert (
            sum(result.metrics["included_market_cap_bucket_counts"].values())
            == result.metrics["included"]
        )
        assert (
            sum(result.metrics["included_price_bucket_counts"].values())
            == result.metrics["included"]
        )
        assert (
            sum(result.metrics["included_country_counts"].values())
            == result.metrics["included"]
        )

        snaps = db_session.query(UniverseSnapshot).all()
        assert len(snaps) == 7

        included = [s for s in snaps if s.operating_universe_inclusion]
        excluded = [s for s in snaps if not s.operating_universe_inclusion]
        assert len(included) == 2
        assert len(excluded) == 5

        included_tickers = {s.ticker for s in included}
        assert included_tickers == {"INCL1", "INCL2"}
        included_country_counts = {}
        for snap in included:
            country = snap.country or "MISSING"
            included_country_counts[country] = included_country_counts.get(country, 0) + 1
        assert included_country_counts == result.metrics["included_country_counts"]

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
        assert reasons["FRGN"] == f"{COUNTRY_REQUIRES_SECURITY_PROFILE_PREFIX}:CA"

    def test_exclusion_counts_in_metrics(self, db_session):
        resp = _mock_screener_response()
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job)

        counts = result.metrics["exclusion_counts"]
        assert counts["etf"] == 1
        assert counts["not_actively_trading"] == 1
        assert counts[f"{COUNTRY_REQUIRES_SECURITY_PROFILE_PREFIX}:CA"] == 1

    def test_market_cap_bucket_helper(self):
        assert _market_cap_bucket(30_000_000) == "30m_100m"
        assert _market_cap_bucket(75_000_000) == "30m_100m"
        assert _market_cap_bucket(100_000_000) == "100m_200m"
        assert _market_cap_bucket(150_000_000) == "100m_200m"
        assert _market_cap_bucket(200_000_000) == "200m_250m"
        assert _market_cap_bucket(225_000_000) == "200m_250m"
        assert _market_cap_bucket(None) == "unknown"

    def test_price_bucket_helper(self):
        assert _price_bucket(2.0) == "2_3"
        assert _price_bucket(2.99) == "2_3"
        assert _price_bucket(3.0) == "3_5"
        assert _price_bucket(4.99) == "3_5"
        assert _price_bucket(5.0) == "5_plus"
        assert _price_bucket(None) == "unknown"

    def test_non_us_common_stock_profile_is_included(self, db_session):
        db_session.add(SecurityProfile(
            symbol="FRGN",
            security_type=COMMON_STOCK,
            last_refreshed_at=_ts(),
            refresh_status=REFRESH_STATUS_ENRICHED,
        ))
        db_session.flush()

        resp = AdapterResponse(
            data=[_stock("FRGN", market_cap=90_000_000, price=7.0, country="CA")],
            lineage=_mock_lineage(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "FRGN"
        ).one()
        assert snap.operating_universe_inclusion is True
        assert snap.exclusion_reason is None
        assert snap.security_type == COMMON_STOCK
        assert result.metrics["country_profile_rescue_count"] == 1
        assert result.metrics["included_country_counts"]["CA"] == 1
        assert result.metrics["included_market_cap_bucket_counts"]["30m_100m"] == 1

    def test_non_us_non_common_profile_is_excluded_by_security_type(self, db_session):
        db_session.add(SecurityProfile(
            symbol="FRGN",
            security_type=MUTUAL_FUND,
            last_refreshed_at=_ts(),
            refresh_status=REFRESH_STATUS_ENRICHED,
        ))
        db_session.flush()

        resp = AdapterResponse(
            data=[
                _stock("FRGN", market_cap=90_000_000, price=7.0, country="CA"),
                _stock("GOOD"),
            ],
            lineage=_mock_lineage(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        snap = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.ticker == "FRGN"
        ).one()
        assert snap.operating_universe_inclusion is False
        assert snap.exclusion_reason == "security_type:mutual_fund"

    def test_shell_company_exclusion_operational_metrics(self, db_session):
        db_session.add(SecurityProfile(
            symbol="SHEL",
            security_type=SPAC_OR_BLANK_CHECK,
            classification_reason="industry_description:SHELL_COMPANIES+BUSINESS_COMBINATION",
            last_refreshed_at=_ts(),
            refresh_status=REFRESH_STATUS_ENRICHED,
        ))
        db_session.flush()

        resp = AdapterResponse(
            data=[_stock("SHEL"), _stock("GOOD")],
            lineage=_mock_lineage(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        assert result.metrics["shell_company_exclusion_count"] == 1
        assert result.metrics["included_shell_company_count"] == 0
        assert result.metrics["shell_company_exclusion_symbols_sample"] == ["SHEL"]
        assert result.metrics["shell_company_exclusion_review_sample"] == [{
            "symbol": "SHEL",
            "company_name": "SHEL Corp",
            "classification_reason": "industry_description:SHELL_COMPANIES+BUSINESS_COMBINATION",
        }]
        assert result.metrics["shell_company_exclusion_review_records"] == [{
            "symbol": "SHEL",
            "company_name": "SHEL Corp",
            "classification_reason": "industry_description:SHELL_COMPANIES+BUSINESS_COMBINATION",
        }]
        assert result.metrics["spac_pattern_exclusion_count"] == 1
        assert result.metrics["spac_pattern_exclusion_symbols_sample"] == ["SHEL"]
        assert result.metrics["spac_pattern_exclusion_review_sample"] == [{
            "symbol": "SHEL",
            "company_name": "SHEL Corp",
            "classification_reason": "industry_description:SHELL_COMPANIES+BUSINESS_COMBINATION",
        }]
        assert result.metrics["spac_pattern_exclusion_review_records"] == [{
            "symbol": "SHEL",
            "company_name": "SHEL Corp",
            "classification_reason": "industry_description:SHELL_COMPANIES+BUSINESS_COMBINATION",
        }]
        assert (
            result.metrics["security_type_classification_reason_counts"][
                "industry_description:SHELL_COMPANIES+BUSINESS_COMBINATION"
            ]
            == 1
        )
        assert (
            sum(result.metrics["security_type_classification_reason_counts"].values())
            == sum(result.metrics["security_type_exclusion_counts"].values())
        )

    def test_included_shell_company_operational_metrics(self, db_session):
        db_session.add(SecurityProfile(
            symbol="CPBI",
            security_type=COMMON_STOCK,
            classification_reason="profile_fields_present",
            raw_profile_json=json.dumps({
                "companyName": "Central Plains Bancshares, Inc. Common Stock",
                "industry": "Shell Companies",
                "description": (
                    "Central Plains Bancshares, Inc. focuses on providing "
                    "various banking products and services."
                ),
            }),
            last_refreshed_at=_ts(),
            refresh_status=REFRESH_STATUS_ENRICHED,
        ))
        db_session.flush()

        resp = AdapterResponse(
            data=[
                _stock(
                    "CPBI",
                    company_name="Central Plains Bancshares, Inc. Common Stock",
                ),
                _stock("GOOD"),
            ],
            lineage=_mock_lineage(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        assert result.metrics["included"] == 2
        assert result.metrics["included_shell_company_count"] == 1
        assert result.metrics["included_shell_company_symbols_sample"] == ["CPBI"]
        expected_record = {
            "symbol": "CPBI",
            "company_name": "Central Plains Bancshares, Inc. Common Stock",
            "classification_reason": "profile_fields_present",
        }
        assert result.metrics["included_shell_company_review_sample"] == [expected_record]
        assert result.metrics["included_shell_company_review_records"] == [expected_record]
        assert result.metrics["shell_company_exclusion_count"] == 0
        assert result.metrics["spac_pattern_exclusion_count"] == 0

    def test_included_shell_company_metric_covers_country_rescue(self, db_session):
        db_session.add(SecurityProfile(
            symbol="FRGN",
            security_type=COMMON_STOCK,
            classification_reason="profile_fields_present",
            raw_profile_json=json.dumps({
                "companyName": "Foreign Operating Co",
                "industry": "Shell Companies",
                "description": "Foreign Operating Co provides software services.",
            }),
            last_refreshed_at=_ts(),
            refresh_status=REFRESH_STATUS_ENRICHED,
        ))
        db_session.flush()

        resp = AdapterResponse(
            data=[
                _stock(
                    "FRGN",
                    company_name="Foreign Operating Co",
                    country="CA",
                ),
                _stock("GOOD"),
            ],
            lineage=_mock_lineage(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        assert result.metrics["country_profile_rescue_count"] == 1
        assert result.metrics["included_shell_company_count"] == 1
        assert result.metrics["included_shell_company_review_records"] == [{
            "symbol": "FRGN",
            "company_name": "Foreign Operating Co",
            "classification_reason": "profile_fields_present",
        }]

    def test_spac_pattern_operational_metrics_cover_regex_reasons(self, db_session):
        db_session.add_all([
            SecurityProfile(
                symbol="ACQ",
                security_type=SPAC_OR_BLANK_CHECK,
                classification_reason="name_pattern:ACQUISITION_SEQUENCE",
                last_refreshed_at=_ts(),
                refresh_status=REFRESH_STATUS_ENRICHED,
            ),
            SecurityProfile(
                symbol="INVI",
                security_type=SPAC_OR_BLANK_CHECK,
                classification_reason="name_pattern:INVESTMENT_CORP_SEQUENCE",
                last_refreshed_at=_ts(),
                refresh_status=REFRESH_STATUS_ENRICHED,
            ),
        ])
        db_session.flush()

        resp = AdapterResponse(
            data=[
                _stock("ACQ", company_name="Alpha Acquisition III Co"),
                _stock("INVI", company_name="Origin Investment Corp I"),
                _stock("GOOD"),
            ],
            lineage=_mock_lineage(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        assert result.metrics["spac_pattern_exclusion_count"] == 2
        assert result.metrics["shell_company_exclusion_count"] == 0
        assert result.metrics["spac_pattern_exclusion_symbols_sample"] == [
            "ACQ",
            "INVI",
        ]
        assert result.metrics["spac_pattern_exclusion_review_sample"] == [
            {
                "symbol": "ACQ",
                "company_name": "Alpha Acquisition III Co",
                "classification_reason": "name_pattern:ACQUISITION_SEQUENCE",
            },
            {
                "symbol": "INVI",
                "company_name": "Origin Investment Corp I",
                "classification_reason": "name_pattern:INVESTMENT_CORP_SEQUENCE",
            },
        ]
        assert result.metrics["security_type_classification_reason_counts"] == {
            "name_pattern:ACQUISITION_SEQUENCE": 1,
            "name_pattern:INVESTMENT_CORP_SEQUENCE": 1,
        }

    def test_spac_pattern_review_records_are_exhaustive_beyond_preview_cap(self, db_session):
        profiles = []
        stocks = []
        for idx in range(26):
            symbol = f"S{idx:02d}"
            profiles.append(SecurityProfile(
                symbol=symbol,
                security_type=SPAC_OR_BLANK_CHECK,
                classification_reason="name_pattern:ACQUISITION_SEQUENCE",
                last_refreshed_at=_ts(),
                refresh_status=REFRESH_STATUS_ENRICHED,
            ))
            stocks.append(_stock(
                symbol,
                company_name=f"Sample Acquisition {idx} Co",
            ))
        db_session.add_all(profiles)
        db_session.flush()

        resp = AdapterResponse(
            data=[*stocks, _stock("GOOD")],
            lineage=_mock_lineage(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        assert result.metrics["spac_pattern_exclusion_count"] == 26
        assert len(result.metrics["spac_pattern_exclusion_review_sample"]) == 25
        assert len(result.metrics["spac_pattern_exclusion_review_records"]) == 26
        assert result.metrics["spac_pattern_exclusion_review_records"][-1] == {
            "symbol": "S25",
            "company_name": "Sample Acquisition 25 Co",
            "classification_reason": "name_pattern:ACQUISITION_SEQUENCE",
        }

    def test_zero_included_scan_fails_without_canonical(self, db_session):
        resp = AdapterResponse(
            data=[
                _stock("ETF1", market_cap=80_000_000, price=25.0, is_etf=True),
                _stock("DEAD", market_cap=60_000_000, price=4.0, is_actively_trading=False),
            ],
            lineage=_mock_lineage(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert not result.ok
        assert result.status == "failed"
        assert result.metrics["included"] == 0
        assert result.metrics["failure_stage"] == "empty_universe"

        scan = db_session.query(UniverseScan).one()
        assert scan.run_status == "failed"
        assert db_session.query(CanonicalUniverseScan).count() == 0

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
        raw_payload = json.loads(lineage_rows[0].raw_payload_json)
        assert len(raw_payload) == 7
        assert {row["symbol"] for row in raw_payload} == {
            "DEAD", "ETF1", "FRGN", "HUGE", "INCL1", "INCL2", "TINY"
        }

        snaps = db_session.query(UniverseSnapshot).all()
        assert {s.source_lineage_hash for s in snaps} == {
            resp.lineage.raw_payload_hash
        }

    def test_screener_raw_payload_replays_lineage_hash(self, db_session):
        data = [_stock("BETA"), _stock("ACME")]
        expected_payload = [
            {
                "symbol": "ACME",
                "company_name": "ACME Corp",
                "market_cap": 75_000_000,
                "price": 5.0,
                "volume": None,
                "sector": None,
                "industry": None,
                "exchange": "NASDAQ",
                "country": "US",
                "is_etf": False,
                "is_actively_trading": True,
            },
            {
                "symbol": "BETA",
                "company_name": "BETA Corp",
                "market_cap": 75_000_000,
                "price": 5.0,
                "volume": None,
                "sector": None,
                "industry": None,
                "exchange": "NASDAQ",
                "country": "US",
                "is_etf": False,
                "is_actively_trading": True,
            },
        ]
        resp = AdapterResponse(
            data=data,
            lineage=LineageMeta(
                provider="FMP",
                endpoint="/stable/company-screener",
                request_timestamp=_ts(),
                asof_timestamp=_ts(),
                raw_payload_hash=stable_hash(expected_payload),
                source_authority="mock",
            ),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        run_job(db_session, job, params={"trading_date": "2026-05-20"})

        lineage = db_session.query(DataLineage).one()
        raw_payload = json.loads(lineage.raw_payload_json)
        assert raw_payload == expected_payload
        assert stable_hash(raw_payload) == lineage.raw_payload_hash

    def test_universe_builder_defers_per_row_flushes(self, db_session, monkeypatch):
        import alpha.jobs.universe_builder as builder_module

        original_snapshot_writer = builder_module.record_universe_snapshot
        original_profile_snapshot_writer = (
            builder_module.record_security_profile_scan_snapshot
        )
        universe_flush_flags = []
        profile_flush_flags = []

        def capture_universe_snapshot(*args, **kwargs):
            universe_flush_flags.append(kwargs.get("flush", True))
            return original_snapshot_writer(*args, **kwargs)

        def capture_profile_snapshot(*args, **kwargs):
            profile_flush_flags.append(kwargs.get("flush", True))
            return original_profile_snapshot_writer(*args, **kwargs)

        monkeypatch.setattr(
            builder_module,
            "record_universe_snapshot",
            capture_universe_snapshot,
        )
        monkeypatch.setattr(
            builder_module,
            "record_security_profile_scan_snapshot",
            capture_profile_snapshot,
        )

        resp = _mock_screener_response()
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job)

        assert result.ok
        assert result.metrics["raw_count"] == 7
        assert result.metrics["security_profile_scan_snapshot_count"] == 7
        assert universe_flush_flags == [False] * 7
        assert profile_flush_flags == [False] * 7

    def test_profile_snapshot_failure_rolls_back_scan_rows(self, db_session, monkeypatch):
        import alpha.jobs.universe_builder as builder_module

        call_count = 0
        original_writer = builder_module.record_security_profile_scan_snapshot

        def failing_profile_snapshot(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("profile snapshot write failed")
            return original_writer(*args, **kwargs)

        monkeypatch.setattr(
            builder_module,
            "record_security_profile_scan_snapshot",
            failing_profile_snapshot,
        )

        resp = _mock_screener_response()
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job)

        assert result.status == "failed"
        assert db_session.query(UniverseScan).count() == 0
        assert db_session.query(UniverseSnapshot).count() == 0
        assert db_session.query(SecurityProfileScanSnapshot).count() == 0
        assert db_session.query(DataLineage).count() == 0
        assert db_session.query(CanonicalUniverseScan).count() == 0

    def test_security_profile_scan_snapshots_replay_cache_hash(self, db_session):
        db_session.add(SecurityProfile(
            symbol="INCL1",
            security_type=COMMON_STOCK,
            source_lineage_hash="profile-lineage",
            profile_payload_hash="profile-payload",
            classification_input_hash="input-hash",
            classification_output_hash="output-hash",
            classifier_version="classifier-v1",
            profile_asof_timestamp=_ts(),
            last_refreshed_at=_ts(),
            refresh_status=REFRESH_STATUS_ENRICHED,
            raw_profile_json=json.dumps({"industry": "Banks"}),
            classification_reason="profile_fields_present",
        ))
        db_session.flush()

        resp = AdapterResponse(
            data=[_stock("INCL1"), _stock("MISS")],
            lineage=_mock_lineage(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        scan = db_session.query(UniverseScan).one()
        rows = (
            db_session.query(SecurityProfileScanSnapshot)
            .filter(SecurityProfileScanSnapshot.scan_id == scan.scan_id)
            .order_by(SecurityProfileScanSnapshot.symbol)
            .all()
        )
        assert [row.symbol for row in rows] == ["INCL1", "MISS"]
        assert [row.cache_status for row in rows] == ["hit", "missing"]
        assert [row.profile_required for row in rows] == [True, True]
        assert rows[0].raw_profile_json == json.dumps({"industry": "Banks"})

        cache_rows = []
        for row in rows:
            if row.cache_status == "missing":
                cache_rows.append({
                    "symbol": row.symbol,
                    "cache_status": "missing",
                })
            else:
                cache_rows.append({
                    "symbol": row.symbol,
                    "cache_status": "hit",
                    "security_type": row.security_type,
                    "refresh_status": row.refresh_status,
                    "classifier_version": row.classifier_version,
                    "classification_input_hash": row.classification_input_hash,
                    "classification_output_hash": row.classification_output_hash,
                    "stale": row.stale,
                })
        recomputed = stable_hash({
            "profile_cache_max_age_days": 7,
            "security_profiles": cache_rows,
        })
        assert recomputed == scan.security_profile_cache_hash
        assert result.metrics["security_profile_scan_snapshot_count"] == 2

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
        scan = db_session.query(UniverseScan).one()
        assert scan.run_status == "failed"
        assert scan.raw_count == 0
        assert scan.deduped_count == 0
        assert scan.duplicate_symbol_count == 0
        assert scan.included_count == 0
        assert scan.excluded_count == 0
        assert db_session.query(CanonicalUniverseScan).count() == 0

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

    def test_duplicate_normalized_symbols_are_deduped(self, db_session):
        resp = AdapterResponse(
            data=[
                _stock(symbol="ACME", market_cap=75_000_000, price=5.0),
                _stock(symbol=" acme ", market_cap=80_000_000, price=6.0),
            ],
            lineage=_mock_lineage(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job)

        assert result.ok
        assert result.metrics["raw_count"] == 2
        assert result.metrics["deduped_count"] == 1
        assert result.metrics["duplicate_symbol_count"] == 1
        snaps = db_session.query(UniverseSnapshot).all()
        assert len(snaps) == 1
        assert snaps[0].ticker == "ACME"
        scan = db_session.query(UniverseScan).one()
        assert scan.raw_count == 2
        assert scan.deduped_count == 1
        assert scan.duplicate_symbol_count == 1
        assert scan.included_count + scan.excluded_count == scan.deduped_count

    def test_duplicate_symbol_prefers_included_row(self, db_session):
        resp = AdapterResponse(
            data=[
                _stock(symbol="ACME", exchange="OTC"),
                _stock(symbol=" acme ", exchange="NASDAQ"),
            ],
            lineage=_mock_lineage(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job)

        assert result.ok
        snap = db_session.query(UniverseSnapshot).one()
        assert snap.ticker == "ACME"
        assert snap.operating_universe_inclusion is True
        assert snap.primary_exchange == "NASDAQ"
        assert result.metrics["duplicate_symbol_count"] == 1

    def test_duplicate_symbol_tie_break_is_deterministic(self):
        low_cap = _stock(symbol="ACME", market_cap=75_000_000, price=5.0)
        high_cap = _stock(symbol=" acme ", market_cap=80_000_000, price=6.0)

        rows1, dupes1 = _dedupe_screener_rows([low_cap, high_cap])
        rows2, dupes2 = _dedupe_screener_rows([high_cap, low_cap])

        assert dupes1 == 1
        assert dupes2 == 1
        assert rows1[0][0].market_cap == 80_000_000
        assert rows2[0][0].market_cap == 80_000_000

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

    def test_non_us_requires_security_profile(self):
        included, reason = _classify(_stock(country="CA"))
        assert not included
        assert reason == f"{COUNTRY_REQUIRES_SECURITY_PROFILE_PREFIX}:CA"
        assert _requires_security_profile(included, reason)

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
        assert reason == f"mcap_below_{MCAP_MIN}"

    def test_mcap_above_excluded(self):
        included, reason = _classify(_stock(market_cap=MCAP_MAX + 1))
        assert not included
        assert reason == f"mcap_above_{MCAP_MAX}"

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

    def test_price_below_floor_excluded(self):
        included, reason = _classify(_stock(price=PRICE_MIN - 0.01))
        assert not included
        assert reason == _price_floor_reason()

    def test_price_exactly_floor_included(self):
        assert _classify(_stock(price=PRICE_MIN))[0]

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
# Test run_universe helpers
# -----------------------------------------------------------------------

class TestRunUniverseEntrypointHelpers:
    def test_live_cli_defaults_to_full_security_profile_coverage(self):
        args = _parse_args(["--live"])

        assert args.min_security_profile_coverage == 1.0
        assert args.allow_incomplete_security_cache is False
        assert args.profile_max_workers == 20
        assert args.profile_rate_limit_per_minute == 2000

    def test_required_profile_symbols_include_included_and_suffix_excluded(self):
        symbols = _required_profile_symbols([
            _stock("INCL", market_cap=75_000_000, price=5.0),
            _stock("ABCDX", market_cap=75_000_000, price=5.0),
            _stock("FRGN", market_cap=75_000_000, price=5.0, country="CA"),
            _stock("ETF1", market_cap=75_000_000, price=25.0, is_etf=True),
            _stock("PENY", market_cap=75_000_000, price=1.0),
        ])

        assert symbols == ["ABCDX", "FRGN", "INCL"]


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
        """22 slices from 30M to 250M in 10M increments."""
        adapter = _make_mock_adapter()
        fetcher = SlicedUniverseFetcher(adapter)
        result = fetcher.fetch()

        assert result.response.ok
        assert result.slice_count == 22
        assert adapter.get_stock_screener.call_count == 22
        assert result.response.lineage.request_timestamp.tzinfo is not None
        assert result.response.lineage.request_timestamp.utcoffset() == timezone.utc.utcoffset(
            result.response.lineage.request_timestamp
        )
        assert result.response.lineage.asof_timestamp.tzinfo is not None
        assert result.response.lineage.asof_timestamp.utcoffset() == timezone.utc.utcoffset(
            result.response.lineage.asof_timestamp
        )
        assert result.response.lineage.data_quality_flags["asof_source"] == (
            "request_timestamp_no_historical_screener_asof"
        )
        assert result.response.lineage.data_quality_flags["historical_backfill_supported"] is False

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


# -----------------------------------------------------------------------
# Test canonical universe scan selection
# -----------------------------------------------------------------------

class TestCanonicalUniverseScan:
    def _run_builder(self, db_session, trading_date="2026-05-20"):
        asof_date = date.fromisoformat(trading_date)
        asof = datetime(
            asof_date.year, asof_date.month, asof_date.day,
            14, 30, tzinfo=timezone.utc,
        )
        resp = AdapterResponse(
            data=_mock_screener_data(),
            lineage=LineageMeta(
                provider="FMP",
                endpoint="/stable/company-screener",
                request_timestamp=asof,
                asof_timestamp=asof,
                raw_payload_hash=f"mock-{trading_date}",
                source_authority="mock",
            ),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        return run_job(db_session, job, params={"trading_date": trading_date})

    def test_first_successful_run_becomes_canonical(self, db_session):
        result = self._run_builder(db_session)

        assert result.ok
        scan = get_canonical_universe_scan(db_session, "2026-05-20")
        assert scan is not None
        assert scan.trading_date == "2026-05-20"
        assert scan.run_status == "finished"
        assert scan.included_count == 2

        canonical = db_session.query(CanonicalUniverseScan).filter(
            CanonicalUniverseScan.trading_date == "2026-05-20"
        ).one()
        assert canonical.selection_reason == "first_successful_scan"

    def test_second_successful_run_replaces_canonical(self, db_session):
        r1 = self._run_builder(db_session)
        scan1 = get_canonical_universe_scan(db_session, "2026-05-20")
        scan1_id = scan1.scan_id

        r2 = self._run_builder(db_session)
        scan2 = get_canonical_universe_scan(db_session, "2026-05-20")

        assert scan2.scan_id != scan1_id
        assert scan2.run_status == "finished"

        canonical = db_session.query(CanonicalUniverseScan).filter(
            CanonicalUniverseScan.trading_date == "2026-05-20"
        ).one()
        assert canonical.scan_id == scan2.scan_id
        assert canonical.selection_reason == "latest_successful_scan"

        # Both scans are immutable records
        all_scans = db_session.query(UniverseScan).filter(
            UniverseScan.trading_date == "2026-05-20"
        ).all()
        assert len(all_scans) == 2

    def test_stale_successful_run_does_not_replace_canonical(self, db_session):
        fresh_resp = AdapterResponse(
            data=_mock_screener_data(),
            lineage=LineageMeta(
                provider="FMP",
                endpoint="/stable/company-screener",
                request_timestamp=datetime(2026, 5, 20, 16, 0, tzinfo=timezone.utc),
                asof_timestamp=datetime(2026, 5, 20, 16, 0, tzinfo=timezone.utc),
                raw_payload_hash="fresh",
                source_authority="mock",
            ),
        )
        stale_resp = AdapterResponse(
            data=_mock_screener_data(),
            lineage=LineageMeta(
                provider="FMP",
                endpoint="/stable/company-screener",
                request_timestamp=datetime(2026, 5, 20, 15, 0, tzinfo=timezone.utc),
                asof_timestamp=datetime(2026, 5, 20, 15, 0, tzinfo=timezone.utc),
                raw_payload_hash="stale",
                source_authority="mock",
            ),
        )

        run_job(
            db_session,
            UniverseBuilderJob(session=db_session, screener_response=fresh_resp),
            params={"trading_date": "2026-05-20"},
        )
        fresh_scan = get_canonical_universe_scan(db_session, "2026-05-20")

        stale_result = run_job(
            db_session,
            UniverseBuilderJob(session=db_session, screener_response=stale_resp),
            params={"trading_date": "2026-05-20"},
        )
        after_stale = get_canonical_universe_scan(db_session, "2026-05-20")

        assert stale_result.ok
        assert after_stale.scan_id == fresh_scan.scan_id
        assert db_session.query(UniverseScan).filter(
            UniverseScan.trading_date == "2026-05-20"
        ).count() == 2

    def test_screener_asof_must_match_trading_date_market_date(self, db_session):
        resp = AdapterResponse(
            data=_mock_screener_data(),
            lineage=LineageMeta(
                provider="FMP",
                endpoint="/stable/company-screener",
                request_timestamp=datetime(2026, 5, 25, 8, 3, tzinfo=timezone.utc),
                asof_timestamp=datetime(2026, 5, 25, 8, 3, tzinfo=timezone.utc),
                raw_payload_hash="late-live-screener",
                source_authority="FMP_Ultimate",
            ),
        )

        result = run_job(
            db_session,
            UniverseBuilderJob(session=db_session, screener_response=resp),
            params={"trading_date": "2026-05-24"},
        )

        assert not result.ok
        assert result.errors == [{
            "stage": "screener_asof",
            "message": (
                "company screener asof market date 2026-05-25 does not match "
                "trading_date 2026-05-24; live FMP screener cannot be used "
                "as a historical backfill input"
            ),
        }]
        assert db_session.query(DataLineage).count() == 0
        assert db_session.query(UniverseScan).count() == 0
        assert db_session.query(UniverseSnapshot).count() == 0
        assert db_session.query(CanonicalUniverseScan).count() == 0

    def test_screener_asof_uses_market_date_not_utc_date(self, db_session):
        resp = AdapterResponse(
            data=_mock_screener_data(),
            lineage=LineageMeta(
                provider="FMP",
                endpoint="/stable/company-screener",
                request_timestamp=datetime(2026, 5, 21, 0, 30, tzinfo=timezone.utc),
                asof_timestamp=datetime(2026, 5, 21, 0, 30, tzinfo=timezone.utc),
                raw_payload_hash="after-close-same-market-date",
                source_authority="FMP_Ultimate",
            ),
        )

        result = run_job(
            db_session,
            UniverseBuilderJob(session=db_session, screener_response=resp),
            params={"trading_date": "2026-05-20"},
        )

        assert result.ok
        assert get_canonical_universe_scan(db_session, "2026-05-20") is not None

    def test_null_screener_asof_is_typed_failure(self, db_session):
        resp = AdapterResponse(
            data=_mock_screener_data(),
            lineage=LineageMeta(
                provider="FMP",
                endpoint="/stable/company-screener",
                request_timestamp=_ts(),
                asof_timestamp=None,
                raw_payload_hash="missing-asof",
                source_authority="mock",
            ),
        )

        result = run_job(
            db_session,
            UniverseBuilderJob(session=db_session, screener_response=resp),
            params={"trading_date": "2026-05-20"},
        )

        assert not result.ok
        assert result.errors == [{
            "stage": "screener_asof",
            "message": "company screener asof_timestamp is missing",
        }]
        assert db_session.query(DataLineage).count() == 0
        assert db_session.query(UniverseScan).count() == 0
        assert db_session.query(UniverseSnapshot).count() == 0
        assert db_session.query(CanonicalUniverseScan).count() == 0

    def test_aware_et_screener_asof_persists_as_utc(self, db_session):
        resp = AdapterResponse(
            data=_mock_screener_data(),
            lineage=LineageMeta(
                provider="FMP",
                endpoint="/stable/company-screener",
                request_timestamp=datetime(
                    2026, 5, 20, 0, 0, tzinfo=ZoneInfo("America/New_York")
                ),
                asof_timestamp=datetime(
                    2026, 5, 20, 0, 0, tzinfo=ZoneInfo("America/New_York")
                ),
                raw_payload_hash="et-midnight",
                source_authority="mock",
            ),
        )

        result = run_job(
            db_session,
            UniverseBuilderJob(session=db_session, screener_response=resp),
            params={"trading_date": "2026-05-20"},
        )
        db_session.expire_all()

        assert result.ok
        scan = db_session.query(UniverseScan).one()
        assert scan.asof_timestamp == datetime(2026, 5, 20, 4, 0)
        assert {
            snap.asof_timestamp
            for snap in db_session.query(UniverseSnapshot).all()
        } == {datetime(2026, 5, 20, 4, 0)}
        lineage = db_session.query(DataLineage).one()
        assert lineage.asof_timestamp == datetime(2026, 5, 20, 4, 0)
        assert lineage.request_timestamp == datetime(2026, 5, 20, 4, 0)

    def test_failed_run_does_not_replace_canonical(self, db_session):
        r1 = self._run_builder(db_session)
        scan1 = get_canonical_universe_scan(db_session, "2026-05-20")
        scan1_id = scan1.scan_id

        # Failed run
        error_resp = AdapterResponse(
            data=None,
            lineage=_mock_lineage(),
            error=ProviderError(
                provider="FMP", endpoint="/stable/company-screener",
                status_code=500, error_type="http",
                message="Internal Server Error", retryable=True,
            ),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=error_resp)
        r2 = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert not r2.ok

        # Canonical pointer unchanged
        scan_after = get_canonical_universe_scan(db_session, "2026-05-20")
        assert scan_after.scan_id == scan1_id
        scans = db_session.query(UniverseScan).filter(
            UniverseScan.trading_date == "2026-05-20"
        ).all()
        assert len(scans) == 2
        assert {scan.run_status for scan in scans} == {"finished", "failed"}

    def test_failed_first_run_records_scan_without_canonical(self, db_session):
        error_resp = AdapterResponse(
            data=None,
            lineage=_mock_lineage(),
            error=ProviderError(
                provider="FMP", endpoint="/stable/company-screener",
                status_code=500, error_type="http",
                message="Internal Server Error", retryable=True,
            ),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=error_resp)
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert not result.ok
        scan = db_session.query(UniverseScan).one()
        assert scan.trading_date == "2026-05-20"
        assert scan.run_status == "failed"
        assert get_canonical_universe_scan(db_session, "2026-05-20") is None

    def test_canonical_members_returns_only_canonical_scan(self, db_session):
        self._run_builder(db_session)
        self._run_builder(db_session)

        members = get_canonical_universe_members(db_session, "2026-05-20")
        assert len(members) == 2
        tickers = {m.ticker for m in members}
        assert tickers == {"INCL1", "INCL2"}

        scan = get_canonical_universe_scan(db_session, "2026-05-20")
        assert all(m.scan_id == scan.scan_id for m in members)

    def test_canonical_members_included_only(self, db_session):
        self._run_builder(db_session)

        included = get_canonical_universe_members(db_session, "2026-05-20", included_only=True)
        all_members = get_canonical_universe_members(db_session, "2026-05-20", included_only=False)

        assert len(included) == 2
        assert len(all_members) == 7

    def test_canonical_members_ordered_by_ticker(self, db_session):
        resp = AdapterResponse(
            data=[
                _stock("ZZZZ", market_cap=75_000_000, price=5.0),
                _stock("AAAA", market_cap=80_000_000, price=6.0),
            ],
            lineage=_mock_lineage(),
        )
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        run_job(db_session, job, params={"trading_date": "2026-05-20"})

        members = get_canonical_universe_members(db_session, "2026-05-20")
        assert [m.ticker for m in members] == ["AAAA", "ZZZZ"]

    def test_canonical_query_nonexistent_date_returns_empty(self, db_session):
        assert get_canonical_universe_scan(db_session, "2099-01-01") is None
        assert get_canonical_universe_members(db_session, "2099-01-01") == []

    def test_different_trading_dates_independent(self, db_session):
        self._run_builder(db_session, "2026-05-20")
        self._run_builder(db_session, "2026-05-21")

        scan_20 = get_canonical_universe_scan(db_session, "2026-05-20")
        scan_21 = get_canonical_universe_scan(db_session, "2026-05-21")
        assert scan_20.scan_id != scan_21.scan_id

    def test_universe_scans_row_created(self, db_session):
        self._run_builder(db_session)

        scans = db_session.query(UniverseScan).all()
        assert len(scans) == 1
        assert scans[0].trading_date == "2026-05-20"
        assert scans[0].raw_count == 7
        assert scans[0].deduped_count == 7
        assert scans[0].duplicate_symbol_count == 0
        assert scans[0].included_count == 2
        assert scans[0].excluded_count == 5
        assert scans[0].run_status == "finished"
        assert scans[0].output_hash is not None

    def test_scan_id_links_snapshots(self, db_session):
        self._run_builder(db_session)

        scan = db_session.query(UniverseScan).one()
        snaps = db_session.query(UniverseSnapshot).filter(
            UniverseSnapshot.scan_id == scan.scan_id
        ).all()
        assert len(snaps) == 7

    def test_duplicate_ticker_in_same_scan_blocked(self, db_session):
        """DB unique constraint blocks same ticker twice in one scan."""
        from sqlalchemy.exc import IntegrityError
        from alpha.evidence.writer import record_universe_snapshot

        scan = UniverseScan(
            scan_id="test-scan",
            trading_date="2026-05-20",
            asof_timestamp=_ts(),
            raw_count=0,
            included_count=0,
            excluded_count=0,
            run_status="finished",
        )
        db_session.add(scan)
        db_session.flush()

        record_universe_snapshot(
            db_session, ticker="ACME", asof_timestamp=_ts(),
            operating_universe_inclusion=True, scan_id="test-scan",
        )
        db_session.flush()

        with pytest.raises(IntegrityError):
            record_universe_snapshot(
                db_session, ticker="ACME", asof_timestamp=_ts(),
                operating_universe_inclusion=True, scan_id="test-scan",
            )
            db_session.flush()

    def test_upsert_rejects_missing_scan_id(self, db_session):
        with pytest.raises(ValueError, match="scan_id does not exist"):
            _upsert_canonical_universe_scan(
                db_session,
                trading_date="2026-05-20",
                scan_id="missing-scan",
                job_run_id="job-1",
            )

    def test_upsert_integrity_fallback_updates_visible_competing_pointer(self):
        class FakeNested:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeQuery:
            def __init__(self, session):
                self.session = session

            def filter(self, *_args):
                return self

            def first(self):
                self.session.first_calls += 1
                if self.session.first_calls == 1:
                    return None
                return self.session.existing_pointer

        class FakeSession:
            def __init__(self):
                self.first_calls = 0
                self.flush_calls = 0
                self.candidate_scan = UniverseScan(
                    scan_id="candidate",
                    trading_date="2026-05-20",
                    asof_timestamp=datetime(2026, 5, 20, 16, 0, tzinfo=timezone.utc),
                    run_status="finished",
                )
                self.existing_scan = UniverseScan(
                    scan_id="existing",
                    trading_date="2026-05-20",
                    asof_timestamp=datetime(2026, 5, 20, 15, 0, tzinfo=timezone.utc),
                    run_status="finished",
                )
                self.existing_pointer = CanonicalUniverseScan(
                    trading_date="2026-05-20",
                    scan_id="existing",
                    selected_job_run_id="old-job",
                )

            def get(self, model, key):
                if model is UniverseScan:
                    return {
                        "candidate": self.candidate_scan,
                        "existing": self.existing_scan,
                    }.get(key)
                return None

            def query(self, _model):
                return FakeQuery(self)

            def begin_nested(self):
                return FakeNested()

            def add(self, _obj):
                pass

            def flush(self):
                self.flush_calls += 1
                if self.flush_calls == 1:
                    raise IntegrityError("insert", {}, Exception("unique"))

        session = FakeSession()

        _upsert_canonical_universe_scan(
            session,
            trading_date="2026-05-20",
            scan_id="candidate",
            job_run_id="new-job",
        )

        assert session.existing_pointer.scan_id == "candidate"
        assert session.existing_pointer.selected_job_run_id == "new-job"
        assert session.existing_pointer.selection_reason == "latest_successful_scan"

    def test_upsert_integrity_fallback_raises_when_pointer_not_visible(self):
        class FakeNested:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeQuery:
            def filter(self, *_args):
                return self

            def first(self):
                return None

        class FakeSession:
            def __init__(self):
                self.candidate_scan = UniverseScan(
                    scan_id="candidate",
                    trading_date="2026-05-20",
                    asof_timestamp=datetime(2026, 5, 20, 16, 0, tzinfo=timezone.utc),
                    run_status="finished",
                )

            def get(self, model, key):
                if model is UniverseScan and key == "candidate":
                    return self.candidate_scan
                return None

            def query(self, _model):
                return FakeQuery()

            def begin_nested(self):
                return FakeNested()

            def add(self, _obj):
                pass

            def flush(self):
                raise IntegrityError("insert", {}, Exception("unique"))

        with pytest.raises(RuntimeError, match="no competing pointer is visible"):
            _upsert_canonical_universe_scan(
                FakeSession(),
                trading_date="2026-05-20",
                scan_id="candidate",
                job_run_id="new-job",
            )

    def test_trading_date_derived_from_asof_when_not_provided(self, db_session):
        resp = _mock_screener_response()
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        run_job(db_session, job)

        scan = db_session.query(UniverseScan).one()
        assert scan.trading_date == "2026-05-20"

    @pytest.mark.parametrize(
        "trading_date_param",
        [
            date(2026, 5, 20),
            datetime(2026, 5, 20, 14, 30, 0, tzinfo=timezone.utc),
            " 2026-05-20 ",
        ],
    )
    def test_trading_date_param_normalized(self, db_session, trading_date_param):
        resp = _mock_screener_response()
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        run_job(db_session, job, params={"trading_date": trading_date_param})

        scan = db_session.query(UniverseScan).one()
        assert scan.trading_date == "2026-05-20"
        assert get_canonical_universe_scan(db_session, "2026-05-20") is not None

    def test_invalid_trading_date_param_fails(self, db_session):
        resp = _mock_screener_response()
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        result = run_job(db_session, job, params={"trading_date": "bad-date"})

        assert not result.ok
        assert db_session.query(UniverseScan).count() == 0
        assert db_session.query(CanonicalUniverseScan).count() == 0
