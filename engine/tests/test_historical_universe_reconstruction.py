from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import func, inspect

from alpha.db.models import (
    CanonicalUniverseScan,
    EvidenceJobRun,
    FmpDelistedCompanyRecord,
    HistoricalUniverseReconstruction,
    SecurityProfile,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.evidence.writer import create_job, finish_run, start_run
from alpha.jobs.fmp_delisted_companies import JOB_NAME as FMP_DELISTED_JOB_NAME
from alpha.jobs.contracts import BaseJob
from alpha.jobs.historical_universe_reconstruction import (
    HistoricalUniverseReconstructionJob,
    _COPY_NULL_SENTINEL,
    _INSERT_COLUMNS,
    _copy_csv_payload,
    _historical_reconstruction_stage_update_changed_stmt,
)
from alpha.jobs.runner import run_job


def _ts() -> datetime:
    return datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc)


def _ensure_active_scan(db_session) -> str:
    scan_id = "active-current-scan"
    if db_session.get(UniverseScan, scan_id) is None:
        db_session.add(
            UniverseScan(
                scan_id=scan_id,
                trading_date="2026-06-06",
                asof_timestamp=_ts(),
                provider="FMP",
                raw_count=0,
                deduped_count=0,
                included_count=0,
                excluded_count=0,
            )
        )
        db_session.flush()
        db_session.add(
            CanonicalUniverseScan(
                trading_date="2026-06-06",
                scan_id=scan_id,
                selected_at=_ts(),
                selection_reason="test_current_active_source",
            )
        )
        db_session.flush()
    return scan_id


def _active(
    db_session,
    ticker: str,
    *,
    exchange: str = "NASDAQ",
    ipo_date: str | None = "2020-01-01",
    company_name: str | None = None,
    market_cap: float | None = 100_000_000,
) -> None:
    scan_id = _ensure_active_scan(db_session)
    db_session.add(
        UniverseSnapshot(
            universe_snapshot_id=f"snap-{ticker}",
            scan_id=scan_id,
            ticker=ticker,
            asof_timestamp=_ts(),
            source_provider="FMP",
            market_cap=market_cap,
            price=10.0,
            security_type="common_stock",
            primary_exchange=exchange,
            operating_universe_inclusion=True,
            source_lineage_hash=f"lineage-{ticker}",
        )
    )
    raw_profile = {
        "symbol": ticker,
        "companyName": company_name or f"{ticker} Inc.",
        "exchange": exchange,
    }
    if ipo_date is not None:
        raw_profile["ipoDate"] = ipo_date
    db_session.merge(
        SecurityProfile(
            symbol=ticker.upper(),
            security_type="common_stock",
            source_provider="FMP",
            profile_payload_hash=f"profile-{ticker}",
            raw_profile_json=json.dumps(raw_profile, sort_keys=True),
        )
    )


def _delisted(
    db_session,
    symbol: str,
    *,
    exchange: str | None = "NASDAQ",
    ipo_date: date | None = date(2020, 1, 1),
    delisted_date: date | None = date(2025, 1, 1),
    company_name: str | None = None,
    raw_payload_extra: dict | None = None,
) -> None:
    exchange_key = exchange.upper() if exchange else "UNKNOWN"
    raw_payload = {
        "symbol": symbol,
        "exchange": exchange,
        "ipoDate": ipo_date,
        "delistedDate": delisted_date,
    }
    if raw_payload_extra:
        raw_payload.update(raw_payload_extra)
    db_session.add(
        FmpDelistedCompanyRecord(
            fmp_delisted_company_id=f"fmp-{symbol}",
            symbol=symbol,
            normalized_symbol=symbol.upper(),
            company_name=company_name or f"{symbol} Corp.",
            exchange=exchange,
            exchange_key=exchange_key,
            ipo_date=ipo_date,
            delisted_date=delisted_date,
            delisted_date_key=delisted_date.isoformat() if delisted_date else "UNKNOWN",
            page_number=0,
            page_limit=100,
            page_row_index=0,
            exchange_relevance_status=(
                "us_listed_relevant"
                if exchange_key in {"NASDAQ", "NYSE", "AMEX"}
                else "non_us_or_unknown_exchange"
            ),
            raw_payload_hash=f"hash-{symbol}",
            raw_payload_json=json.dumps(
                raw_payload,
                default=str,
                sort_keys=True,
            ),
        )
    )


def _record_delisted_ingest_run(
    db_session,
    *,
    status: str = "finished",
    max_pages_reached: bool = False,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
) -> str:
    job = create_job(
        db_session,
        name=FMP_DELISTED_JOB_NAME,
        job_type="ingestion",
        owner="historical_replay",
    )
    run = start_run(db_session, job_id=job.job_id, params={"source": "test"})
    if started_at is not None:
        run.started_at = started_at
    if status == "running":
        run.metric_json = json.dumps(
            {"max_pages_reached": max_pages_reached, "rows_seen": 1}
        )
        db_session.flush()
    else:
        finish_run(
            db_session,
            run,
            status=status,
            metrics={"max_pages_reached": max_pages_reached, "rows_seen": 1},
        )
        if ended_at is not None:
            run.ended_at = ended_at
        elif started_at is not None:
            run.ended_at = started_at
        db_session.flush()
    return run.job_run_id


def _run(
    db_session,
    replay_day: date = date(2024, 6, 1),
    *,
    allow_partial_delisted_source: bool = False,
):
    job = HistoricalUniverseReconstructionJob(
        session=db_session,
        replay_date=replay_day,
        run_timestamp=_ts(),
        allow_partial_delisted_source=allow_partial_delisted_source,
    )
    return run_job(
        db_session,
        job,
        params={
            "source": "test_historical_universe_reconstruction",
            "allow_partial_delisted_source": allow_partial_delisted_source,
        },
    )


def _row(db_session, ticker: str) -> HistoricalUniverseReconstruction:
    return (
        db_session.query(HistoricalUniverseReconstruction)
        .filter(HistoricalUniverseReconstruction.normalized_symbol == ticker.upper())
        .one()
    )


def test_schema_contains_reconstruction_table_columns_and_uniqueness(db_session):
    inspector = inspect(db_session.bind)
    columns = {
        column["name"]
        for column in inspector.get_columns("historical_universe_reconstructions")
    }
    assert {
        "replay_date",
        "ticker",
        "normalized_symbol",
        "exchange",
        "company_name",
        "ipo_date",
        "delisted_date",
        "inclusion_status",
        "rejection_reason",
        "source",
        "source_provenance_json",
        "reconstructed",
        "pit_filter_status_json",
        "input_hash",
        "output_hash",
    } <= columns
    uniques = inspector.get_unique_constraints("historical_universe_reconstructions")
    assert any(
        constraint["name"] == "ux_historical_universe_recon_date_symbol"
        and constraint["column_names"] == ["replay_date", "normalized_symbol"]
        for constraint in uniques
    )


def test_postgres_stage_update_changed_predicate_is_grouped():
    sql = _historical_reconstruction_stage_update_changed_stmt("tmp_hur_stage").text

    assert "WHERE t.replay_date = s.replay_date " in sql
    assert "AND t.normalized_symbol = s.normalized_symbol " in sql
    assert "AND (t.replay_date IS DISTINCT FROM s.replay_date OR" in sql
    assert "s.output_hash)" in sql


def test_reconstruction_inclusion_and_rejection_rules(db_session):
    _record_delisted_ingest_run(db_session)
    _active(db_session, "ACTIVE", ipo_date="2020-01-01")
    _active(db_session, "NEWIPO", ipo_date="2025-01-01")
    _active(db_session, "MISSIPO", ipo_date=None)
    _delisted(
        db_session,
        "LATERD",
        ipo_date=date(2020, 1, 1),
        delisted_date=date(2025, 1, 1),
    )
    _delisted(
        db_session,
        "DEADOLD",
        ipo_date=date(2020, 1, 1),
        delisted_date=date(2023, 12, 31),
    )
    _delisted(
        db_session,
        "FOREIGN",
        exchange="XETRA",
        ipo_date=date(2020, 1, 1),
        delisted_date=date(2025, 1, 1),
    )
    _delisted(
        db_session,
        "NODEL",
        ipo_date=date(2020, 1, 1),
        delisted_date=None,
    )

    result = _run(db_session, date(2024, 6, 1))

    assert result.status == "finished"
    assert _row(db_session, "ACTIVE").inclusion_status == "included"
    assert _row(db_session, "NEWIPO").rejection_reason == "ipo_after_replay_date"
    assert _row(db_session, "MISSIPO").rejection_reason == "missing_ipo_date"
    assert _row(db_session, "LATERD").inclusion_status == "included"
    assert (
        _row(db_session, "DEADOLD").rejection_reason
        == "delisted_on_or_before_replay_date"
    )
    assert _row(db_session, "FOREIGN").rejection_reason == "exchange_not_operating_universe"
    missing_delisted = _row(db_session, "NODEL")
    assert missing_delisted.inclusion_status == "included"
    provenance = json.loads(missing_delisted.source_provenance_json)
    assert provenance["missing_delisted_date_source"] is True
    assert result.metrics["included_count"] == 3
    assert result.metrics["excluded_count"] == 4


def test_delisted_non_common_symbols_are_excluded_before_replay_eligibility(db_session):
    _record_delisted_ingest_run(db_session)
    _delisted(db_session, "BTMWW")
    _delisted(db_session, "WARRW")
    _delisted(db_session, "ABCDU")
    _delisted(
        db_session,
        "PREFR",
        raw_payload_extra={"securityType": "Preferred Stock"},
    )

    result = _run(db_session, date(2024, 6, 1))

    assert result.status == "finished"
    assert _row(db_session, "BTMWW").rejection_reason == "non_common_symbol_suffix"
    assert _row(db_session, "WARRW").rejection_reason == "non_common_symbol_suffix"
    assert _row(db_session, "ABCDU").rejection_reason == "non_common_symbol_suffix"
    assert (
        _row(db_session, "PREFR").rejection_reason
        == "security_type_non_common_delisted"
    )
    pref_provenance = json.loads(_row(db_session, "PREFR").source_provenance_json)
    assert pref_provenance["source_intervals"][0]["security_type"] == "preferred_stock"


def test_ticker_reuse_evaluates_delisted_and_current_intervals_independently(db_session):
    _record_delisted_ingest_run(db_session)
    _active(db_session, "REUSE", ipo_date="2025-01-01", company_name="New Issuer")
    _delisted(
        db_session,
        "REUSE",
        ipo_date=date(2020, 1, 1),
        delisted_date=date(2024, 12, 31),
        company_name="Old Issuer",
    )

    result = _run(db_session, date(2024, 6, 1))

    assert result.status == "finished"
    row = _row(db_session, "REUSE")
    assert row.inclusion_status == "included"
    assert row.company_name == "Old Issuer"
    assert row.ipo_date == date(2020, 1, 1)
    assert row.delisted_date == date(2024, 12, 31)
    provenance = json.loads(row.source_provenance_json)
    intervals = provenance["source_intervals"]
    assert len(intervals) == 2
    assert {
        (interval["source"], interval["inclusion_status"], interval["rejection_reason"])
        for interval in intervals
    } == {
        ("current_active_universe", "excluded", "ipo_after_replay_date"),
        ("fmp_delisted_companies", "included", None),
    }


@pytest.mark.parametrize("duplicate_status", ["failed", "partial_failed"])
def test_finished_delisted_run_ignores_newer_failed_duplicate(
    db_session,
    duplicate_status: str,
):
    accepted_run_id = _record_delisted_ingest_run(
        db_session,
        max_pages_reached=False,
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    _record_delisted_ingest_run(
        db_session,
        status=duplicate_status,
        max_pages_reached=True,
        started_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )
    _delisted(db_session, "LATERD")

    result = _run(db_session, date(2024, 6, 1))

    assert result.status == "finished"
    assert result.metrics["delisted_source_complete"] is True
    assert result.metrics["delisted_source_partial_reason"] is None
    assert result.metrics["delisted_source_latest_job_run_id"] == accepted_run_id
    assert result.metrics["delisted_source_latest_run_status"] == "finished"
    provenance = json.loads(_row(db_session, "LATERD").source_provenance_json)
    assert provenance["delisted_source_complete"] is True


def test_partial_delisted_source_returns_partial_failed_by_default(db_session):
    _record_delisted_ingest_run(
        db_session,
        status="finished",
        max_pages_reached=True,
    )
    _delisted(db_session, "LATERD")

    result = _run(db_session, date(2024, 6, 1))

    assert result.status == "partial_failed"
    assert result.metrics["delisted_source_complete"] is False
    assert result.metrics["delisted_source_partial_reason"] == "max_pages_reached"
    assert result.errors[0]["error_type"] == "delisted_source_partial"
    row = _row(db_session, "LATERD")
    provenance = json.loads(row.source_provenance_json)
    assert provenance["delisted_source_complete"] is False
    assert provenance["delisted_source_partial_reason"] == "max_pages_reached"


def test_allow_partial_delisted_source_keeps_success_but_stamps_provenance(db_session):
    _record_delisted_ingest_run(
        db_session,
        status="finished",
        max_pages_reached=True,
    )
    _delisted(db_session, "LATERD")

    result = _run(
        db_session,
        date(2024, 6, 1),
        allow_partial_delisted_source=True,
    )

    assert result.status == "finished"
    assert result.metrics["delisted_source_complete"] is False
    assert result.metrics["delisted_source_partial_reason"] == "max_pages_reached"
    provenance = json.loads(_row(db_session, "LATERD").source_provenance_json)
    assert provenance["allow_partial_delisted_source"] is True
    assert provenance["delisted_source_complete"] is False


def test_no_finished_delisted_ingest_run_returns_missing_run_partial(db_session):
    _record_delisted_ingest_run(
        db_session,
        status="failed",
        max_pages_reached=False,
    )
    _delisted(db_session, "LATERD")

    result = _run(db_session, date(2024, 6, 1))

    assert result.status == "partial_failed"
    assert result.metrics["delisted_source_complete"] is False
    assert (
        result.metrics["delisted_source_partial_reason"]
        == "fmp_delisted_ingestion_run_not_found"
    )
    assert result.metrics["delisted_source_latest_job_run_id"] is None
    assert result.errors[0]["partial_reason"] == (
        "fmp_delisted_ingestion_run_not_found"
    )
    provenance = json.loads(_row(db_session, "LATERD").source_provenance_json)
    assert provenance["delisted_source_complete"] is False
    assert (
        provenance["delisted_source_partial_reason"]
        == "fmp_delisted_ingestion_run_not_found"
    )


def test_rerun_updates_without_duplicate_rows(db_session):
    _active(db_session, "ACTIVE", ipo_date="2020-01-01")
    first = _run(db_session)
    second = _run(db_session)

    assert first.metrics["rows_inserted"] == 1
    assert second.metrics["rows_inserted"] == 0
    assert second.metrics["rows_updated"] == 1
    duplicate_groups = (
        db_session.query(
            HistoricalUniverseReconstruction.replay_date,
            HistoricalUniverseReconstruction.normalized_symbol,
            func.count().label("row_count"),
        )
        .group_by(
            HistoricalUniverseReconstruction.replay_date,
            HistoricalUniverseReconstruction.normalized_symbol,
        )
        .having(func.count() > 1)
        .count()
    )
    assert duplicate_groups == 0


def test_pre_replay_delisted_exclusion_persistence_can_be_suppressed(db_session):
    _record_delisted_ingest_run(db_session)
    _delisted(
        db_session,
        "DEADOLD",
        ipo_date=date(2020, 1, 1),
        delisted_date=date(2023, 12, 31),
    )
    job = HistoricalUniverseReconstructionJob(
        session=db_session,
        replay_date=date(2024, 6, 1),
        run_timestamp=_ts(),
        persist_pre_replay_delisted_exclusions=False,
    )

    result = run_job(db_session, job, params={"source": "test_suppression"})

    assert result.status == "finished"
    assert result.metrics["excluded_count"] == 1
    assert result.metrics["rows_inserted"] == 0
    assert result.metrics["rows_suppressed_pre_replay_delisted_exclusions"] == 1
    assert result.metrics["rows_suppressed_excluded_fmp_delisted"] == 1
    assert db_session.query(HistoricalUniverseReconstruction).count() == 0


def test_reconstruction_progress_events_cover_load_eval_and_persist(db_session):
    _record_delisted_ingest_run(db_session)
    _active(db_session, "ACTIVE", ipo_date="2020-01-01")
    _delisted(db_session, "LATERD")
    events = []
    job = HistoricalUniverseReconstructionJob(
        session=db_session,
        replay_date=date(2024, 6, 1),
        run_timestamp=_ts(),
        progress_callback=lambda event, payload: events.append((event, payload)),
        progress_every=1,
    )

    result = run_job(
        db_session,
        job,
        params={"source": "test_progress"},
    )

    assert result.status == "finished"
    event_names = [event for event, _payload in events]
    assert "candidate_load_start" in event_names
    assert "candidate_load_finish" in event_names
    assert "interval_evaluation_progress" in event_names
    assert "persistence_start" in event_names
    assert "persistence_progress" in event_names
    assert "persistence_finish" in event_names
    assert result.metrics["rows_processed"] == 2
    assert result.metrics["total_serialized_payload_bytes"] > 0
    assert result.metrics["source_provenance_json_bytes"] > 0
    assert result.metrics["max_row_serialized_payload_bytes"] > 0
    assert result.metrics["progress_events"][-1]["event"] == "persistence_finish"
    provenance = json.loads(_row(db_session, "ACTIVE").source_provenance_json)
    assert provenance["provenance_payload_policy"] == (
        "compact_row_interval_summary_full_run_facts_in_lineage"
    )
    assert "source_rows" not in provenance
    assert provenance["source_row_hashes"]


def test_compact_persisted_provenance_omits_repeated_interval_payloads(db_session):
    _record_delisted_ingest_run(db_session)
    _active(db_session, "REUSE", ipo_date="2025-01-01", company_name="New Issuer")
    _delisted(
        db_session,
        "REUSE",
        ipo_date=date(2020, 1, 1),
        delisted_date=date(2024, 12, 31),
        company_name="Old Issuer",
    )

    job = HistoricalUniverseReconstructionJob(
        session=db_session,
        replay_date=date(2024, 6, 1),
        run_timestamp=_ts(),
        compact_persisted_provenance=True,
    )
    result = run_job(
        db_session,
        job,
        params={"source": "test_compact_persisted_provenance"},
    )

    assert result.status == "finished"
    provenance = json.loads(_row(db_session, "REUSE").source_provenance_json)
    assert provenance["provenance_payload_policy"] == "compact_public_cohort_row_v4"
    assert provenance["source_interval_count"] == 2
    assert "source_intervals" not in provenance
    assert "selected_source_interval" not in provenance
    assert "source_row_hashes" not in provenance
    assert provenance["source_row_hash"]
    assert provenance["delisted_source_complete"] is True
    filter_status = json.loads(_row(db_session, "REUSE").pit_filter_status_json)
    assert filter_status["pit_filter_payload_policy"] == "compact_public_cohort_row_v3"


def test_copy_csv_payload_sanitizes_nul_and_preserves_null_sentinel():
    row = {column: None for column in _INSERT_COLUMNS}
    row.update(
        {
            "historical_universe_reconstruction_id": "hur-test",
            "replay_date": date(2024, 6, 1),
            "ticker": "NUL",
            "normalized_symbol": "NUL",
            "company_name": "Bad\x00Name",
            "inclusion_status": "included",
            "source_provenance_json": json.dumps({"note": "contains\x00nul"}),
            "reconstructed": True,
            "reconstruction_method": "historical_pit_universe_reconstruction_v1",
            "pit_filter_status_json": "{}",
            "data_lineage_id": "lineage",
            "job_run_id": "job-run",
            "input_hash": "input",
            "output_hash": "output",
            "created_at": _ts(),
            "updated_at": _ts(),
        }
    )

    payload = _copy_csv_payload([row])

    assert "\x00" not in payload
    assert "BadName" in payload
    assert _COPY_NULL_SENTINEL in payload


def test_keyboard_interrupt_marks_evidence_run_failed(db_session):
    class InterruptingJob(BaseJob):
        @property
        def job_name(self) -> str:
            return "interrupting_historical_universe_test"

        @property
        def job_type(self) -> str:
            return "test"

        def run(self, _ctx):
            raise KeyboardInterrupt("operator stopped test")

    result = run_job(db_session, InterruptingJob(), params={"source": "test"})

    assert result.status == "failed"
    run = (
        db_session.query(EvidenceJobRun)
        .order_by(EvidenceJobRun.started_at.desc())
        .first()
    )
    assert run.run_status == "failed"
    assert "KeyboardInterrupt" in run.error_json


def test_reconstruction_rows_are_distinct_from_live_universe_snapshots(db_session):
    _active(db_session, "ACTIVE", ipo_date="2020-01-01")
    before = db_session.query(UniverseSnapshot).count()

    _run(db_session)

    assert db_session.query(UniverseSnapshot).count() == before
    assert db_session.query(HistoricalUniverseReconstruction).count() == 1
    assert _row(db_session, "ACTIVE").current_universe_snapshot_id == "snap-ACTIVE"


def test_current_market_cap_and_liquidity_are_not_applied_as_pit_filters(db_session):
    _active(
        db_session,
        "HUGECAP",
        ipo_date="2020-01-01",
        market_cap=99_000_000_000,
    )

    _run(db_session)

    row = _row(db_session, "HUGECAP")
    assert row.inclusion_status == "included"
    filter_status = json.loads(row.pit_filter_status_json)
    assert filter_status["market_cap_filter"] == "not_applied_not_pit_safe"
    assert filter_status["price_filter"] == "not_applied_not_pit_safe"
    assert filter_status["liquidity_filter"] == "not_applied_not_pit_safe"
