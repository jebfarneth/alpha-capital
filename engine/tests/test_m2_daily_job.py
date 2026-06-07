from __future__ import annotations

import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError, stable_hash
from alpha.data.edgar import FORM4_TRANSACTIONS_ENDPOINT, SecForm4Transaction
from alpha.db.models import (
    CanonicalUniverseScan,
    EvidenceJob,
    EvidenceJobRun,
    M2SecFetchCoverage,
    SecurityIdentitySnapshot,
    SignalRegistry,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.jobs import m2_daily, run_m2_daily
from alpha.jobs.contracts import JobResult
from alpha.jobs.m2_daily import M2DailyAssemblyJob, _persist_transaction
from alpha.jobs.runner import run_job
from alpha.assembly.m2_daily import transaction_evidence_from_sec


RUN_TS = datetime(2026, 6, 3, 22, 0, tzinfo=timezone.utc)
DECISION_DATE = "2026-06-03"


def _lineage(endpoint: str, data, asof: datetime, error=None) -> LineageMeta:
    return LineageMeta(
        provider="SEC",
        endpoint=endpoint,
        request_timestamp=RUN_TS,
        asof_timestamp=asof,
        raw_payload_hash=stable_hash({"endpoint": endpoint, "data": data, "error": error}),
        source_authority="sec_edgar",
    )


def _provider_error(endpoint: str) -> ProviderError:
    return ProviderError(
        provider="SEC",
        endpoint=endpoint,
        status_code=503,
        error_type="http",
        message="transient SEC unavailable",
        retryable=True,
    )


def _issuer_cik(index: int) -> str:
    return str(1000 + index).zfill(10)


def _setup_m2_universe(db_session, tickers: list[str]) -> dict[str, str]:
    scan = UniverseScan(
        scan_id="m2-test-scan",
        trading_date=DECISION_DATE,
        asof_timestamp=datetime(2026, 6, 3, 20, 0, tzinfo=timezone.utc),
        raw_count=len(tickers),
        deduped_count=len(tickers),
        included_count=len(tickers),
        excluded_count=0,
        run_status="finished",
        source_lineage_hash="m2-scan-hash",
    )
    db_session.add(scan)
    db_session.flush()
    db_session.add(CanonicalUniverseScan(
        trading_date=DECISION_DATE,
        scan_id=scan.scan_id,
        selection_reason="test",
    ))
    cik_by_ticker = {}
    for idx, ticker in enumerate(tickers, start=1):
        ticker = ticker.upper()
        cik = _issuer_cik(idx)
        cik_by_ticker[ticker] = cik
        db_session.add(UniverseSnapshot(
            universe_snapshot_id=f"snap-{ticker}",
            scan_id=scan.scan_id,
            ticker=ticker,
            asof_timestamp=scan.asof_timestamp,
            market_cap=75_000_000,
            price=5.25,
            primary_exchange="NASDAQ",
            security_type="common_stock",
            operating_universe_inclusion=True,
            source_lineage_hash=f"snapshot-hash-{ticker}",
        ))
        db_session.add(SecurityIdentitySnapshot(
            security_identity_snapshot_id=f"identity-{ticker}",
            scan_id=scan.scan_id,
            ticker=ticker,
            cik=cik,
            identity_status="present",
            source_provider="test",
        ))
    db_session.flush()
    return cik_by_ticker


def _sec_tx(
    ticker: str,
    insider_cik: str,
    *,
    accession: str,
    accepted_at: datetime,
    tx_date: date,
    code: str,
) -> SecForm4Transaction:
    acquired_disposed = "A" if code == "P" else "D"
    return SecForm4Transaction(
        transaction_id=stable_hash({
            "ticker": ticker,
            "insider_cik": insider_cik,
            "accession": accession,
            "tx_date": tx_date,
            "code": code,
        }),
        accession_number=accession,
        filing_form="4",
        filing_date=accepted_at.date(),
        filing_accepted_at=accepted_at,
        issuer_cik="0000009999",
        issuer_name=f"{ticker} Corp",
        ticker=ticker,
        insider_cik=insider_cik,
        insider_name=f"Owner {insider_cik}",
        insider_state="CA",
        insider_roles={},
        transaction_date=tx_date,
        transaction_code=code,
        acquired_disposed_code=acquired_disposed,
        security_title="Common Stock",
        shares=10_000,
        price_per_share=2.0,
        ownership_type="D",
        is_10b5_1=False,
        raw={},
    )


def _opportunistic_history_rows(ticker: str, insider_cik: str) -> list[SecForm4Transaction]:
    return [
        _sec_tx(
            ticker,
            insider_cik,
            accession=f"{ticker}-{insider_cik}-23",
            accepted_at=datetime(2023, 1, 16, 17, tzinfo=timezone.utc),
            tx_date=date(2023, 1, 15),
            code="S",
        ),
        _sec_tx(
            ticker,
            insider_cik,
            accession=f"{ticker}-{insider_cik}-24",
            accepted_at=datetime(2024, 2, 16, 17, tzinfo=timezone.utc),
            tx_date=date(2024, 2, 15),
            code="S",
        ),
        _sec_tx(
            ticker,
            insider_cik,
            accession=f"{ticker}-{insider_cik}-25",
            accepted_at=datetime(2025, 3, 17, 17, tzinfo=timezone.utc),
            tx_date=date(2025, 3, 15),
            code="S",
        ),
    ]


def _m2_fire_rows(ticker: str) -> list[SecForm4Transaction]:
    rows: list[SecForm4Transaction] = []
    for insider_cik in ("0000000001", "0000000002"):
        rows.extend(_opportunistic_history_rows(ticker, insider_cik))
        rows.append(_sec_tx(
            ticker,
            insider_cik,
            accession=f"{ticker}-{insider_cik}-26",
            accepted_at=datetime(2026, 6, 3, 22, tzinfo=timezone.utc),
            tx_date=date(2026, 6, 3),
            code="P",
        ))
    return rows


def _m2u_fire_rows(ticker: str) -> list[SecForm4Transaction]:
    return [
        _sec_tx(
            ticker,
            insider_cik,
            accession=f"{ticker}-{insider_cik}-26",
            accepted_at=datetime(2026, 6, 3, 22, tzinfo=timezone.utc),
            tx_date=date(2026, 6, 3),
            code="P",
        )
        for insider_cik in ("0000000003", "0000000004")
    ]


def _seed_transactions(db_session, ticker: str, rows: list[SecForm4Transaction]) -> None:
    snapshot = db_session.get(UniverseSnapshot, f"snap-{ticker}")
    assert snapshot is not None
    for row in rows:
        evidence = transaction_evidence_from_sec(
            row,
            detected_at=row.filing_accepted_at,
            market_cap_usd=snapshot.market_cap,
            ticker=ticker,
            lineage_ids=["seed-lineage"],
            lineage_hashes=["seed-hash"],
        )
        _persist_transaction(
            db_session,
            evidence,
            scan_id=snapshot.scan_id,
            snapshot=snapshot,
            job_run_id=None,
        )


class FakeSecAdapter:
    def __init__(
        self,
        *,
        cik_by_ticker: dict[str, str],
        rows_by_ticker: dict[str, list[SecForm4Transaction]] | None = None,
        error_tickers: set[str] | None = None,
    ):
        self.cik_by_ticker = {ticker.upper(): cik for ticker, cik in cik_by_ticker.items()}
        self.ticker_by_cik = {cik: ticker for ticker, cik in self.cik_by_ticker.items()}
        self.rows_by_ticker = {
            ticker.upper(): list(rows)
            for ticker, rows in (rows_by_ticker or {}).items()
        }
        self.error_tickers = {ticker.upper() for ticker in (error_tickers or set())}
        self.form4_requests: list[dict[str, object]] = []

    def get_company_ticker(self, ticker, *, asof=None):
        ticker = ticker.upper()
        data = SimpleNamespace(cik_str=self.cik_by_ticker[ticker])
        return AdapterResponse(
            data=data,
            lineage=_lineage("sec_company_ticker", {"ticker": ticker}, asof or RUN_TS),
        )

    def get_form4_transactions(self, cik, *, from_date, to_date, asof=None):
        ticker = self.ticker_by_cik[cik]
        self.form4_requests.append({
            "ticker": ticker,
            "from_date": from_date,
            "to_date": to_date,
        })
        if ticker in self.error_tickers:
            error = _provider_error(FORM4_TRANSACTIONS_ENDPOINT)
            return AdapterResponse(
                data=None,
                lineage=_lineage(FORM4_TRANSACTIONS_ENDPOINT, None, asof or RUN_TS, error),
                error=error,
            )
        rows = self.rows_by_ticker.get(ticker, [])
        return AdapterResponse(
            data=rows,
            lineage=_lineage(FORM4_TRANSACTIONS_ENDPOINT, rows, asof or RUN_TS),
        )


def _run_direct(db_session, adapter: FakeSecAdapter):
    job = M2DailyAssemblyJob(
        db_session,
        sec_adapter=adapter,
        run_timestamp=RUN_TS,
        skip_fmp_enrichment=True,
    )
    return run_job(
        db_session,
        job,
        params={
            "run_timestamp": RUN_TS.isoformat(),
            "skip_fmp_enrichment": True,
        },
    )


def _run_main(db_session, adapter: FakeSecAdapter, monkeypatch) -> int:
    monkeypatch.setenv("SEC_USER_AGENT", "tests@example.com")
    monkeypatch.setattr(run_m2_daily, "load_runtime_env", lambda: None)
    monkeypatch.setattr(run_m2_daily, "get_session", lambda: db_session)
    monkeypatch.setattr(run_m2_daily, "SecEdgarAdapter", lambda _config: adapter)
    return run_m2_daily.main([
        "--live",
        "--run-timestamp", RUN_TS.isoformat(),
        "--skip-fmp-enrichment",
    ])


def _latest_m2_run(db_session) -> EvidenceJobRun:
    return (
        db_session.query(EvidenceJobRun)
        .join(EvidenceJob, EvidenceJob.job_id == EvidenceJobRun.job_id)
        .filter(EvidenceJob.job_name == "m2_daily_feature_assembly")
        .order_by(EvidenceJobRun.started_at.desc())
        .one()
    )


def test_non_firing_fetch_error_finishes_and_cli_returns_zero(
    db_session,
    monkeypatch,
):
    cik_by_ticker = _setup_m2_universe(db_session, ["FIRE", "MISS"])
    _seed_transactions(db_session, "FIRE", _m2_fire_rows("FIRE"))
    adapter = FakeSecAdapter(
        cik_by_ticker=cik_by_ticker,
        error_tickers={"MISS"},
    )

    rc = _run_main(db_session, adapter, monkeypatch)
    run = _latest_m2_run(db_session)
    metrics = json.loads(run.metric_json)

    assert rc == 0
    assert run.run_status == "finished"
    assert metrics["fetch_error_count"] == 1
    assert metrics["fatal_fetch_error_count"] == 0
    assert metrics["tolerated_fetch_error_count"] == 1
    assert metrics["included_market_cap_bucket_counts"]["30m_100m"] == 2
    assert metrics["tickers_with_transaction_history_count"] == 1
    assert metrics["full_history_fetch_ticker_count"] == 2
    assert metrics["m2_warm_path_requires_seeded_history"] is True
    assert db_session.query(SignalRegistry).filter_by(pattern_id="M2", ticker="FIRE").count() == 1


def test_fetch_error_on_m2_firing_ticker_is_partial_failed_and_cli_returns_one(
    db_session,
    monkeypatch,
):
    cik_by_ticker = _setup_m2_universe(db_session, ["FIRE"])
    _seed_transactions(db_session, "FIRE", _m2_fire_rows("FIRE"))
    adapter = FakeSecAdapter(
        cik_by_ticker=cik_by_ticker,
        error_tickers={"FIRE"},
    )

    rc = _run_main(db_session, adapter, monkeypatch)
    run = _latest_m2_run(db_session)
    metrics = json.loads(run.metric_json)

    assert rc == 1
    assert run.run_status == "partial_failed"
    assert metrics["fatal_fetch_error_count"] == 1
    assert metrics["tolerated_fetch_error_count"] == 0
    assert metrics["fatal_fetch_errors"] == [{"ticker": "FIRE", "stage": "sec_form4"}]


def test_fetch_error_on_m2u_firing_ticker_is_fatal(db_session, monkeypatch):
    cik_by_ticker = _setup_m2_universe(db_session, ["SHDW"])
    _seed_transactions(db_session, "SHDW", _m2u_fire_rows("SHDW"))
    adapter = FakeSecAdapter(
        cik_by_ticker=cik_by_ticker,
        error_tickers={"SHDW"},
    )

    rc = _run_main(db_session, adapter, monkeypatch)
    run = _latest_m2_run(db_session)
    metrics = json.loads(run.metric_json)

    assert rc == 1
    assert run.run_status == "partial_failed"
    assert metrics["fatal_fetch_error_count"] == 1
    assert db_session.query(SignalRegistry).filter_by(pattern_id="M2U", ticker="SHDW").count() == 1


def test_zero_fetch_errors_finishes(db_session):
    cik_by_ticker = _setup_m2_universe(db_session, ["QUIET"])
    result = _run_direct(db_session, FakeSecAdapter(cik_by_ticker=cik_by_ticker))

    assert result.status == "finished"
    assert result.metrics["fetch_error_count"] == 0
    assert result.metrics["fatal_fetch_error_count"] == 0
    assert result.metrics["tolerated_fetch_error_count"] == 0
    assert result.metrics["included_market_cap_bucket_counts"]["30m_100m"] == 1
    assert result.metrics["tickers_with_transaction_history_count"] == 0
    assert result.metrics["tickers_with_sec_fetch_coverage_count"] == 0
    assert result.metrics["tickers_with_m2_warm_coverage_count"] == 0
    assert result.metrics["full_history_fetch_ticker_count"] == 1


def test_no_form4_fetch_coverage_warms_next_run(db_session):
    cik_by_ticker = _setup_m2_universe(db_session, ["QUIET"])
    first_adapter = FakeSecAdapter(cik_by_ticker=cik_by_ticker)
    first = _run_direct(db_session, first_adapter)

    assert first.status == "finished"
    assert first.metrics["full_history_fetch_ticker_count"] == 1
    assert db_session.query(M2SecFetchCoverage).filter_by(ticker="QUIET").count() == 1

    second_adapter = FakeSecAdapter(cik_by_ticker=cik_by_ticker)
    second = _run_direct(db_session, second_adapter)

    assert second.status == "finished"
    assert second.metrics["tickers_with_transaction_history_count"] == 0
    assert second.metrics["tickers_with_sec_fetch_coverage_count"] == 1
    assert second.metrics["tickers_with_m2_warm_coverage_count"] == 1
    assert second.metrics["full_history_fetch_ticker_count"] == 0
    assert len(second_adapter.form4_requests) == 1
    assert first_adapter.form4_requests
    assert second_adapter.form4_requests[0]["from_date"] != first_adapter.form4_requests[0]["from_date"]


def test_orchestration_partial_failed_still_escalates_without_fetch_errors(
    db_session,
    monkeypatch,
):
    class PartialOrchestration:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, _ctx):
            return JobResult(
                status="partial_failed",
                metrics={"total_signals_persisted": 0},
                errors=[{"stage": "detector", "message": "boom"}],
            )

    monkeypatch.setattr(m2_daily, "DetectorOrchestrationJob", PartialOrchestration)
    cik_by_ticker = _setup_m2_universe(db_session, ["QUIET"])
    result = _run_direct(db_session, FakeSecAdapter(cik_by_ticker=cik_by_ticker))

    assert result.status == "partial_failed"
    assert result.metrics["fetch_error_count"] == 0
    assert result.metrics["fatal_fetch_error_count"] == 0
    assert result.metrics["tolerated_fetch_error_count"] == 0
