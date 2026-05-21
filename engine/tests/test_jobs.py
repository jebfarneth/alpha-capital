"""
Job orchestration tests.

  - Successful job run persists evidence_job_runs as finished.
  - Failed job run persists evidence_job_runs as failed with error_json.
  - Universe job records included and excluded symbols.
  - Universe snapshots link to job_run_id and data lineage.
  - Deterministic output hashes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List

from alpha.data.contracts import (
    AdapterResponse,
    LineageMeta,
    ProviderError,
    stable_hash,
)
from alpha.data.fmp import FmpScreenerResult
from alpha.db.models import DataLineage, EvidenceJob, EvidenceJobRun, UniverseSnapshot
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.runner import run_job
from alpha.jobs.universe_builder import UniverseBuilderJob


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


def _mock_screener_data() -> List[FmpScreenerResult]:
    return [
        FmpScreenerResult(symbol="INCL1", company_name="Included One", market_cap=75_000_000, price=5.0, country="US", is_etf=False, is_actively_trading=True),
        FmpScreenerResult(symbol="INCL2", company_name="Included Two", market_cap=150_000_000, price=10.0, country="US", is_etf=False, is_actively_trading=True),
        FmpScreenerResult(symbol="ETF1", company_name="ETF Fund", market_cap=80_000_000, price=25.0, country="US", is_etf=True, is_actively_trading=True),
        FmpScreenerResult(symbol="TINY", company_name="Too Small", market_cap=5_000_000, price=0.50, country="US", is_etf=False, is_actively_trading=True),
        FmpScreenerResult(symbol="HUGE", company_name="Too Large", market_cap=500_000_000, price=50.0, country="US", is_etf=False, is_actively_trading=True),
        FmpScreenerResult(symbol="DEAD", company_name="Not Active", market_cap=60_000_000, price=2.0, country="US", is_etf=False, is_actively_trading=False),
        FmpScreenerResult(symbol="FRGN", company_name="Foreign Co", market_cap=90_000_000, price=7.0, country="CA", is_etf=False, is_actively_trading=True),
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
        assert result.metrics["total_screened"] == 7
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

        # Run again in a fresh session — same input should produce same hash
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
