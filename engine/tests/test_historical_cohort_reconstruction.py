from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from alpha.data.contracts import stable_hash
from alpha.db.models import (
    EvidenceJob,
    EvidenceJobRun,
    FmpDelistedCompanyRecord,
)
from alpha.jobs.fmp_delisted_companies import JOB_NAME as FMP_DELISTED_JOB_NAME
from alpha.jobs.contracts import JobResult
from alpha.jobs.run_historical_cohort_reconstruction import (
    CohortReconstructionResult,
    HISTORICAL_M4_RANGE_REPLAY_JOB_NAME,
    MARKET_PATH_ALEMBIC_REVISION,
    _has_completed_m4_replay_evidence,
    _preflight_public_delisted_source,
    _validate_pattern_ids,
    _validate_write_target,
    main as cohort_cli_main,
    run_historical_cohort_reconstruction,
)


class _FakeSession:
    def __init__(self) -> None:
        self.commit_count = 0

    def commit(self) -> None:
        self.commit_count += 1


class _FakeJob:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    def job_name(self):
        return "fake"

    @property
    def job_type(self):
        return "fake"

    def run(self, _ctx):  # pragma: no cover - tests inject the runner.
        raise AssertionError("fake job should be handled by injected job_runner")


def _range_replay_metrics(replay_dates: list[str]) -> dict:
    date_results = []
    for index, replay_date in enumerate(replay_dates):
        date_results.append(
            {
                "replay_date": replay_date,
                "historical_universe_included_count": 3,
                "m4_evaluable_count": 3 - index,
                "m4_non_evaluable_count": index,
                "rows_inserted": 1 if index == 0 else 0,
                "rows_reused": 0 if index == 0 else 1,
                "fired_m4_signal_count": 1 if index == 0 else 0,
                "rejected_or_no_fire_count": 2,
                "polygon_fallback_count": 1 if index == 0 else 0,
                "missing_price_evidence_count": index,
                "non_evaluable_ticker_count": index,
                "fetch_error_count": 0,
                "coverage_status": (
                    "complete_price_evidence"
                    if index == 0
                    else "partial_price_evidence"
                ),
                "completion_classification": (
                    "completed"
                    if index == 0
                    else "completed_with_non_evaluable_price_evidence"
                ),
            }
        )
    return {
        "replay_dates": replay_dates,
        "active_replay_dates": replay_dates,
        "skipped_replay_dates": [],
        "unique_ticker_count": 3,
        "date_ticker_equivalent_fetch_count": len(replay_dates) * 3,
        "fmp_fetch": {
            "requested_ticker_count": 3,
            "fmp_success_count": 2,
            "polygon_fallback_count": 1,
        },
        "validation": {
            "duplicate_hur_groups": 0,
            "duplicate_signal_identity_groups": 0,
            "feature_lineage_missing_count": 0,
            "replay_stamp_missing_count": 0,
            "lookahead_violation_count": 0,
        },
        "stage_timing_seconds": {
            "universe_load_reconstruction_seconds": 0.1,
            "fmp_fetch_seconds": 0.2,
            "validation_seconds": 0.01,
        },
        "completion_classification": "completed_with_non_evaluable_price_evidence",
        "coverage_status": "partial_price_evidence",
        "non_evaluable_price_evidence_samples": [
            {
                "ticker": "MISS",
                "source": "current_active_universe",
                "exchange": "NASDAQ",
                "security_type": "common_stock",
                "category_hint": None,
                "provider_attempt_statuses": [
                    {"provider": "FMP", "status": "no_usable_bars"},
                    {"provider": "Polygon", "status": "no_usable_bars"},
                ],
                "missing_evidence_dates": replay_dates[1:],
            }
        ]
        if len(replay_dates) > 1
        else [],
        "universe": {
            "persistence": {
                "rows_inserted": 4,
                "rows_updated": 1,
            },
            "date_metrics": [
                {
                    "replay_date": replay_date,
                    "included_count": 3,
                    "rows_inserted": 2,
                    "rows_updated": 0,
                }
                for replay_date in replay_dates
            ],
        },
        "total_rows_inserted": sum(row["rows_inserted"] for row in date_results),
        "total_rows_reused": sum(row["rows_reused"] for row in date_results),
        "total_historical_universe_included_count": sum(
            row["historical_universe_included_count"] for row in date_results
        ),
        "total_m4_evaluable_count": sum(
            row["m4_evaluable_count"] for row in date_results
        ),
        "total_m4_non_evaluable_count": sum(
            row["m4_non_evaluable_count"] for row in date_results
        ),
        "total_missing_price_evidence_count": sum(
            row["missing_price_evidence_count"] for row in date_results
        ),
        "total_polygon_fallback_count": sum(
            row["polygon_fallback_count"] for row in date_results
        ),
        "total_fired_m4_signal_count": sum(
            row["fired_m4_signal_count"] for row in date_results
        ),
        "total_rejected_or_no_fire_count": sum(
            row["rejected_or_no_fire_count"] for row in date_results
        ),
        "date_results": date_results,
    }


def test_cohort_cli_refuses_unconfirmed_public_write(monkeypatch):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)

    rc = cohort_cli_main(
        [
            "--live",
            "--start-date",
            "2026-01-02",
            "--end-date",
            "2026-01-05",
        ]
    )

    assert rc == 1


def test_cohort_cli_refuses_schema_public(monkeypatch):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)

    rc = cohort_cli_main(
        [
            "--live",
            "--schema",
            "public",
            "--start-date",
            "2026-01-02",
            "--end-date",
            "2026-01-05",
        ]
    )

    assert rc == 1


def test_cohort_public_cli_requires_polygon_fallback_config_before_db(
    monkeypatch,
    capsys,
):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@host/db")
    monkeypatch.setattr(
        "alpha.jobs.run_historical_cohort_reconstruction."
        "_verify_public_market_path_revision",
        lambda _url: {"current": [MARKET_PATH_ALEMBIC_REVISION]},
    )
    monkeypatch.setattr(
        "alpha.jobs.run_historical_cohort_reconstruction._optional_polygon_adapter",
        lambda: None,
    )

    def fail_get_session():
        raise AssertionError("missing Polygon config should fail before DB session")

    monkeypatch.setattr(
        "alpha.jobs.run_historical_cohort_reconstruction.get_session",
        fail_get_session,
    )

    rc = cohort_cli_main(
        [
            "--live",
            "--confirm-live-write",
            "--start-date",
            "2026-01-02",
            "--end-date",
            "2026-01-05",
        ]
    )

    assert rc == 1
    assert "requires Polygon fallback configuration" in capsys.readouterr().out


def test_cohort_public_cli_refuses_date_by_date_before_db(monkeypatch, capsys):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@host/db")
    monkeypatch.setattr(
        "alpha.jobs.run_historical_cohort_reconstruction."
        "_verify_public_market_path_revision",
        lambda _url: {"current": [MARKET_PATH_ALEMBIC_REVISION]},
    )
    monkeypatch.setattr(
        "alpha.jobs.run_historical_cohort_reconstruction._optional_polygon_adapter",
        lambda: object(),
    )

    def fail_get_session():
        raise AssertionError("date-by-date public mode should fail before DB session")

    monkeypatch.setattr(
        "alpha.jobs.run_historical_cohort_reconstruction.get_session",
        fail_get_session,
    )

    rc = cohort_cli_main(
        [
            "--live",
            "--confirm-live-write",
            "--start-date",
            "2026-01-02",
            "--end-date",
            "2026-01-05",
            "--execution-mode",
            "date-by-date",
        ]
    )

    assert rc == 1
    assert "--execution-mode range-cached" in capsys.readouterr().out


def test_cohort_cli_summary_prints_evaluable_coverage(monkeypatch, capsys, tmp_path):
    class _ClosableSession(_FakeSession):
        def close(self) -> None:
            pass

    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setattr(
        "alpha.jobs.run_historical_cohort_reconstruction.prepare_writable_schema_target",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "alpha.jobs.run_historical_cohort_reconstruction.get_session",
        lambda: _ClosableSession(),
    )
    monkeypatch.setattr(
        "alpha.jobs.run_historical_cohort_reconstruction.FmpConfig.from_env",
        lambda: object(),
    )
    monkeypatch.setattr(
        "alpha.jobs.run_historical_cohort_reconstruction.FmpAdapter",
        lambda _config: object(),
    )
    monkeypatch.setattr(
        "alpha.jobs.run_historical_cohort_reconstruction."
        "CachedHistoricalPriceFmpAdapter",
        lambda adapter: adapter,
    )

    def fake_cohort(**_kwargs):
        return CohortReconstructionResult(
            status="finished",
            metrics={
                "dates_finished": 1,
                "dates_failed": 0,
                "dates_skipped": 0,
                "execution_mode": "range_cached",
                "completion_classification": (
                    "completed_with_non_evaluable_price_evidence"
                ),
                "coverage_status": "partial_price_evidence",
                "historical_universe_included_count_total": 3,
                "m4_evaluable_count_total": 2,
                "m4_non_evaluable_count_total": 1,
                "m4_rows_inserted_total": 1,
                "m4_rows_reused_total": 0,
                "m4_fired_signal_count_total": 1,
                "m4_rejected_or_no_fire_count_total": 1,
                "polygon_fallback_configured": False,
                "polygon_fallback_count_total": 0,
                "missing_price_evidence_count_total": 1,
                "non_evaluable_ticker_count_total": 1,
                "range_unique_ticker_count": 3,
                "range_date_ticker_equivalent_fetch_count": 3,
            },
            errors=[],
            artifact_path=str(tmp_path / "artifact.json"),
        )

    monkeypatch.setattr(
        "alpha.jobs.run_historical_cohort_reconstruction."
        "run_historical_cohort_reconstruction",
        fake_cohort,
    )

    rc = cohort_cli_main(
        [
            "--live",
            "--schema",
            "scratch_cli",
            "--start-date",
            "2026-01-02",
            "--end-date",
            "2026-01-02",
            "--progress-artifact",
            str(tmp_path / "artifact.json"),
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "Completion class:       completed_with_non_evaluable_price_evidence" in output
    assert "Coverage status:        partial_price_evidence" in output
    assert "Historical universe:    3" in output
    assert "M4 evaluable:           2" in output
    assert "M4 non-evaluable:       1" in output


def test_cohort_public_write_requires_market_path_pre_m3_revision(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@host/db")
    calls = []

    def checker(url):
        calls.append(url)
        return {
            "current": [MARKET_PATH_ALEMBIC_REVISION],
            "heads": ["a4b5c6d7e8f9"],
            "default_off_m3_pending": True,
        }

    target = _validate_write_target(
        schema=None,
        confirm_live_write=True,
        database_url=None,
        revision_checker=checker,
    )

    assert target["mode"] == "public"
    assert calls == ["postgresql+psycopg://u:p@host/db"]


def test_cohort_public_write_refuses_partial_flags():
    def checker(_url):
        return {"current": [MARKET_PATH_ALEMBIC_REVISION]}

    with pytest.raises(ValueError, match="allow-partial-delisted-source"):
        _validate_write_target(
            schema=None,
            confirm_live_write=True,
            database_url="postgresql+psycopg://u:p@host/db",
            allow_partial_delisted_source=True,
            revision_checker=checker,
        )
    with pytest.raises(ValueError, match="allow-partial-universe"):
        _validate_write_target(
            schema=None,
            confirm_live_write=True,
            database_url="postgresql+psycopg://u:p@host/db",
            allow_partial_universe=True,
            revision_checker=checker,
        )


def test_cohort_public_write_refuses_create_tables():
    with pytest.raises(ValueError, match="create-tables"):
        _validate_write_target(
            schema=None,
            confirm_live_write=True,
            database_url="postgresql+psycopg://u:p@host/db",
            create_tables=True,
            revision_checker=lambda _url: {"current": [MARKET_PATH_ALEMBIC_REVISION]},
        )


def test_cohort_scratch_create_tables_still_allowed():
    target = _validate_write_target(
        schema="scratch_cohort",
        confirm_live_write=False,
        database_url=None,
        create_tables=True,
        allow_partial_delisted_source=True,
        allow_partial_universe=True,
    )

    assert target == {"mode": "scratch", "schema": "scratch_cohort", "alembic": None}


def test_cohort_public_write_rejects_wrong_revision():
    def checker(_url):
        raise ValueError("requires Alembic 3456789abcde")

    with pytest.raises(ValueError, match="3456789abcde"):
        _validate_write_target(
            schema=None,
            confirm_live_write=True,
            database_url="postgresql+psycopg://u:p@host/db",
            revision_checker=checker,
        )


def test_cohort_public_preflight_requires_delisted_source(db_session):
    with pytest.raises(ValueError, match="populated fmp_delisted_companies"):
        _preflight_public_delisted_source(db_session)


def test_cohort_public_preflight_rejects_truncated_delisted_ingestion(db_session):
    _seed_fmp_delisted_source(db_session, max_pages_reached=True)

    with pytest.raises(ValueError, match="max_pages_reached=false"):
        _preflight_public_delisted_source(db_session)


def test_cohort_public_preflight_accepts_complete_delisted_ingestion(db_session):
    _seed_fmp_delisted_source(db_session, max_pages_reached=False)

    metrics = _preflight_public_delisted_source(db_session)

    assert metrics["fmp_delisted_companies_row_count"] == 1
    assert metrics["fmp_delisted_latest_run_status"] == "finished"
    assert metrics["fmp_delisted_max_pages_reached"] is False


def test_cohort_public_preflight_ignores_newer_failed_duplicate_run(db_session):
    _seed_fmp_delisted_source(
        db_session,
        max_pages_reached=False,
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    _seed_fmp_delisted_source(
        db_session,
        max_pages_reached=True,
        run_status="failed",
        started_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        add_record=False,
    )

    metrics = _preflight_public_delisted_source(db_session)

    assert metrics["fmp_delisted_latest_run_status"] == "finished"
    assert metrics["fmp_delisted_max_pages_reached"] is False


def test_cohort_public_preflight_rejects_running_duplicate_run(db_session):
    _seed_fmp_delisted_source(
        db_session,
        max_pages_reached=False,
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    _seed_fmp_delisted_source(
        db_session,
        max_pages_reached=False,
        run_status="running",
        started_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        add_record=False,
    )

    with pytest.raises(ValueError, match="still running"):
        _preflight_public_delisted_source(db_session)


def test_cohort_public_preflight_accepts_bounded_delisted_coverage(db_session):
    _seed_fmp_delisted_source(
        db_session,
        max_pages_reached=False,
        date_cutoff_reached=True,
        stop_after_delisted_before="2026-01-01",
    )

    metrics = _preflight_public_delisted_source(
        db_session,
        replay_start_date=date(2026, 1, 2),
    )

    assert metrics["fmp_delisted_date_cutoff_reached"] is True
    assert metrics["fmp_delisted_stop_after_delisted_before"] == "2026-01-01"


def test_cohort_public_preflight_rejects_bounded_delisted_gap(db_session):
    _seed_fmp_delisted_source(
        db_session,
        max_pages_reached=False,
        date_cutoff_reached=True,
        stop_after_delisted_before="2026-02-01",
    )

    with pytest.raises(ValueError, match="only covers replay dates"):
        _preflight_public_delisted_source(
            db_session,
            replay_start_date=date(2026, 1, 2),
        )


def test_cohort_completion_checker_accepts_range_replay_evidence(db_session):
    job = EvidenceJob(
        job_name=HISTORICAL_M4_RANGE_REPLAY_JOB_NAME,
        job_type="historical_replay",
        owner_component="test",
    )
    db_session.add(job)
    db_session.flush()
    db_session.add(
        EvidenceJobRun(
            job_id=job.job_id,
            run_status="finished",
            started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            ended_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            metric_json=json.dumps(
                {
                    "date_results": [
                        {
                            "replay_date": "2026-01-02",
                            "rows_inserted": 1,
                        }
                    ]
                }
            ),
        )
    )
    db_session.commit()

    assert _has_completed_m4_replay_evidence(db_session, date(2026, 1, 2))
    assert not _has_completed_m4_replay_evidence(db_session, date(2026, 1, 5))


def test_cohort_runner_explicitly_defers_m1():
    with pytest.raises(ValueError, match="Only audited M4 replay is implemented"):
        _validate_pattern_ids(["M1"])


def test_cohort_checkpoint_and_idempotent_rerun(tmp_path):
    session = _FakeSession()
    signal_dates_seen: set[str] = set()
    calls: list[dict] = []

    def fake_runner(_session, _job, *, params=None, **_kwargs):
        params = params or {}
        calls.append(params)
        replay_date = params["replay_date"]
        if params["stage"] == "historical_universe_reconstruction":
            progress = _job.kwargs.get("progress_callback")
            if progress:
                progress(
                    "persistence_finish",
                    {
                        "event": "persistence_finish",
                        "replay_date": replay_date,
                        "rows_inserted": 1,
                    },
                )
            return JobResult(
                status="finished",
                metrics={
                    "replay_date": replay_date,
                    "rows_inserted": 1,
                    "rows_updated": 0,
                    "included_count": 2,
                },
            )
        first_replay = replay_date not in signal_dates_seen
        signal_dates_seen.add(replay_date)
        progress = _job.kwargs.get("progress_callback")
        if progress:
            progress(
                "ticker_fetch_progress",
                {
                    "event": "ticker_fetch_progress",
                    "replay_date": replay_date,
                    "started": 1,
                    "finished": 0,
                    "ticker_total": 2,
                },
            )
        return JobResult(
            status="finished",
            metrics={
                "replay_dates": [replay_date],
                "total_rows_inserted": 1 if first_replay else 0,
                "total_rows_reused": 0 if first_replay else 1,
                "total_fired_m4_signal_count": 1 if first_replay else 0,
                "total_rejected_or_no_fire_count": 1,
                "date_results": [
                    {
                        "replay_date": replay_date,
                        "rows_inserted": 1 if first_replay else 0,
                        "rows_reused": 0 if first_replay else 1,
                    }
                ],
            },
        )

    first_artifact = tmp_path / "first.json"
    first = run_historical_cohort_reconstruction(
        session=session,
        fmp_adapter=object(),
        pattern_ids=["M4"],
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 5),
        schema="scratch_cohort",
        progress_artifact=first_artifact,
        job_runner=fake_runner,
        universe_job_factory=_FakeJob,
        m4_replay_job_factory=_FakeJob,
        execution_mode="date-by-date",
        print_fn=lambda _message: None,
    )
    second_artifact = tmp_path / "second.json"
    second = run_historical_cohort_reconstruction(
        session=session,
        fmp_adapter=object(),
        pattern_ids=["M4"],
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 5),
        schema="scratch_cohort",
        progress_artifact=second_artifact,
        job_runner=fake_runner,
        universe_job_factory=_FakeJob,
        m4_replay_job_factory=_FakeJob,
        execution_mode="date-by-date",
        print_fn=lambda _message: None,
    )

    assert first.status == "finished"
    assert first.metrics["dates_finished"] == 2
    assert first.metrics["m4_rows_inserted_total"] == 2
    assert second.status == "finished"
    assert second.metrics["dates_finished"] == 2
    assert second.metrics["m4_rows_inserted_total"] == 0
    assert second.metrics["m4_rows_reused_total"] == 2
    assert session.commit_count == 4
    assert [call["stage"] for call in calls].count("historical_m4_replay") == 4

    artifact = json.loads(first_artifact.read_text())
    assert artifact["status"] == "finished"
    assert [row["status"] for row in artifact["date_results"]] == [
        "finished",
        "finished",
    ]
    assert artifact["date_results"][0]["m4_date_metrics"]["rows_inserted"] == 1
    assert artifact["date_results"][0]["historical_universe_progress_last"]["event"] == (
        "persistence_finish"
    )
    assert artifact["date_results"][0]["historical_m4_replay_progress_last"]["event"] == (
        "ticker_fetch_progress"
    )


def test_cohort_range_cached_mode_invokes_one_range_replay_and_maps_metrics(tmp_path):
    session = _FakeSession()
    artifact_path = tmp_path / "range_cached.json"
    calls: list[dict] = []

    def fake_runner(_session, _job, *, params=None, **_kwargs):
        params = params or {}
        calls.append(params)
        assert params["stage"] == "historical_m4_range_replay"
        progress = _job.kwargs.get("progress_callback")
        if progress:
            progress(
                "range_date_start",
                {
                    "event": "range_date_start",
                    "replay_date": "2026-01-02",
                    "ticker_total": 3,
                },
            )
        return JobResult(
            status="finished",
            metrics=_range_replay_metrics(params["replay_dates"]),
        )

    result = run_historical_cohort_reconstruction(
        session=session,
        fmp_adapter=object(),
        polygon_adapter=object(),
        pattern_ids=["M4"],
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 5),
        schema="scratch_cohort",
        progress_artifact=artifact_path,
        job_runner=fake_runner,
        m4_range_replay_job_factory=_FakeJob,
        print_fn=lambda _message: None,
    )

    assert result.status == "finished"
    assert len(calls) == 1
    assert calls[0]["replay_dates"] == ["2026-01-02", "2026-01-05"]
    assert calls[0]["execution_mode"] == "range_cached"
    assert result.metrics["execution_mode"] == "range_cached"
    assert result.metrics["completion_classification"] == (
        "completed_with_non_evaluable_price_evidence"
    )
    assert result.metrics["coverage_status"] == "partial_price_evidence"
    assert result.metrics["dates_finished"] == 2
    assert result.metrics["range_unique_ticker_count"] == 3
    assert result.metrics["range_date_ticker_equivalent_fetch_count"] == 6
    assert result.metrics["historical_universe_included_count_total"] == 6
    assert result.metrics["m4_evaluable_count_total"] == 5
    assert result.metrics["m4_non_evaluable_count_total"] == 1
    assert result.metrics["m4_rows_inserted_total"] == 1
    assert result.metrics["m4_rows_reused_total"] == 1
    assert result.metrics["m4_fired_signal_count_total"] == 1
    assert result.metrics["m4_rejected_or_no_fire_count_total"] == 4
    assert result.metrics["polygon_fallback_count_total"] == 1
    assert result.metrics["missing_price_evidence_count_total"] == 1
    assert result.metrics["non_evaluable_ticker_count_total"] == 1
    assert result.metrics["universe_rows_inserted_total"] == 4
    assert result.metrics["universe_rows_updated_total"] == 1
    assert session.commit_count == 1

    artifact = json.loads(artifact_path.read_text())
    assert artifact["execution_mode"] == "range_cached"
    assert artifact["completion_classification"] == (
        "completed_with_non_evaluable_price_evidence"
    )
    assert artifact["coverage_status"] == "partial_price_evidence"
    assert artifact["historical_m4_range_replay_status"] == "finished"
    assert artifact["summary"]["range_fetch_metrics"]["requested_ticker_count"] == 3
    assert artifact["summary"]["non_evaluable_price_evidence_samples"][0]["ticker"] == (
        "MISS"
    )
    assert [row["status"] for row in artifact["date_results"]] == [
        "finished",
        "finished",
    ]
    assert artifact["date_results"][0]["m4_replay_metrics"]["range_replay"] is True
    assert artifact["date_results"][0]["historical_m4_range_replay_progress_last"][
        "event"
    ] == "range_date_start"


def test_cohort_range_cached_hard_failure_remains_failed(tmp_path):
    session = _FakeSession()

    def fake_runner(_session, _job, *, params=None, **_kwargs):
        return JobResult(
            status="partial_failed",
            metrics={
                "replay_dates": (params or {})["replay_dates"],
                "date_results": [
                    {
                        "replay_date": "2026-01-02",
                        "historical_universe_included_count": 1,
                        "m4_evaluable_count": 1,
                        "m4_non_evaluable_count": 0,
                        "missing_price_evidence_count": 0,
                        "fetch_error_count": 0,
                        "rows_inserted": 0,
                        "rows_reused": 0,
                        "fired_m4_signal_count": 0,
                        "rejected_or_no_fire_count": 0,
                    }
                ],
            },
            errors=[{"stage": "detector", "error_type": "runtime_error"}],
        )

    result = run_historical_cohort_reconstruction(
        session=session,
        fmp_adapter=object(),
        polygon_adapter=object(),
        pattern_ids=["M4"],
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 2),
        schema="scratch_cohort",
        progress_artifact=tmp_path / "hard_failure.json",
        job_runner=fake_runner,
        m4_range_replay_job_factory=_FakeJob,
        print_fn=lambda _message: None,
    )

    assert result.status == "failed"
    assert result.metrics["completion_classification"] == "hard_failure"
    assert result.metrics["coverage_status"] == "hard_error"
    artifact = json.loads((tmp_path / "hard_failure.json").read_text())
    assert artifact["status"] == "failed"
    assert artifact["completion_classification"] == "hard_failure"


def test_cohort_range_cached_resume_skips_verified_dates_before_range_job(tmp_path):
    session = _FakeSession()
    previous_artifact = tmp_path / "previous_range.json"
    previous_artifact.write_text(
        json.dumps(
            {
                "date_results": [
                    {
                        "replay_date": "2026-01-02",
                        "status": "finished",
                        "m4_replay_status": "finished",
                    },
                    {
                        "replay_date": "2026-01-05",
                        "status": "failed",
                        "m4_replay_status": "failed",
                    },
                ]
            }
        )
    )
    calls: list[dict] = []

    def fake_runner(_session, _job, *, params=None, **_kwargs):
        params = params or {}
        calls.append(params)
        return JobResult(
            status="finished",
            metrics=_range_replay_metrics(params["replay_dates"]),
        )

    result = run_historical_cohort_reconstruction(
        session=session,
        fmp_adapter=object(),
        polygon_adapter=object(),
        pattern_ids=["M4"],
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 5),
        schema="scratch_cohort",
        progress_artifact=tmp_path / "resume_range.json",
        resume_from_artifact=previous_artifact,
        completion_checker=lambda _session, replay_day: replay_day == date(2026, 1, 2),
        job_runner=fake_runner,
        m4_range_replay_job_factory=_FakeJob,
        print_fn=lambda _message: None,
    )

    assert result.status == "finished"
    assert len(calls) == 1
    assert calls[0]["stage"] == "historical_m4_range_replay"
    assert calls[0]["replay_dates"] == ["2026-01-05"]
    assert result.metrics["dates_skipped"] == 1
    assert result.metrics["dates_finished"] == 1
    artifact = json.loads((tmp_path / "resume_range.json").read_text())
    assert [row["status"] for row in artifact["date_results"]] == [
        "skipped",
        "finished",
    ]


def test_cohort_artifact_and_summary_expose_polygon_fallback_metrics(tmp_path):
    session = _FakeSession()
    artifact_path = tmp_path / "polygon_metrics.json"

    def fake_runner(_session, _job, *, params=None, **_kwargs):
        params = params or {}
        replay_date = params["replay_date"]
        if params["stage"] == "historical_universe_reconstruction":
            return JobResult(
                status="finished",
                metrics={
                    "replay_date": replay_date,
                    "rows_inserted": 1,
                    "rows_updated": 0,
                },
            )
        return JobResult(
            status="finished",
            metrics={
                "replay_dates": [replay_date],
                "total_rows_inserted": 2,
                "total_rows_reused": 0,
                "total_fired_m4_signal_count": 1,
                "total_rejected_or_no_fire_count": 1,
                "total_polygon_fallback_count": 2,
                "total_missing_price_evidence_count": 1,
                "total_non_evaluable_ticker_count": 1,
                "date_results": [
                    {
                        "replay_date": replay_date,
                        "rows_inserted": 2,
                        "polygon_fallback_count": 2,
                        "missing_price_evidence_count": 1,
                        "non_evaluable_ticker_count": 1,
                    }
                ],
            },
        )

    result = run_historical_cohort_reconstruction(
        session=session,
        fmp_adapter=object(),
        polygon_adapter=object(),
        pattern_ids=["M4"],
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 2),
        schema="scratch_cohort",
        progress_artifact=artifact_path,
        job_runner=fake_runner,
        universe_job_factory=_FakeJob,
        m4_replay_job_factory=_FakeJob,
        execution_mode="date-by-date",
        print_fn=lambda _message: None,
    )

    assert result.status == "finished"
    assert result.metrics["polygon_fallback_configured"] is True
    assert result.metrics["polygon_fallback_count_total"] == 2
    assert result.metrics["missing_price_evidence_count_total"] == 1
    assert result.metrics["non_evaluable_ticker_count_total"] == 1
    artifact = json.loads(artifact_path.read_text())
    assert artifact["polygon_fallback_configured"] is True
    assert artifact["summary"]["polygon_fallback_configured"] is True
    assert artifact["summary"]["polygon_fallback_count_total"] == 2
    assert artifact["summary"]["missing_price_evidence_count_total"] == 1
    assert artifact["summary"]["non_evaluable_ticker_count_total"] == 1


def test_cohort_resume_from_artifact_skips_only_verified_finished_dates(tmp_path):
    session = _FakeSession()
    previous_artifact = tmp_path / "previous.json"
    previous_artifact.write_text(
        json.dumps(
            {
                "date_results": [
                    {
                        "replay_date": "2026-01-02",
                        "status": "finished",
                        "m4_replay_status": "finished",
                    },
                    {
                        "replay_date": "2026-01-05",
                        "status": "failed",
                        "m4_replay_status": "failed",
                    },
                ]
            }
        )
    )
    calls: list[dict] = []

    def fake_runner(_session, _job, *, params=None, **_kwargs):
        params = params or {}
        calls.append(params)
        replay_date = params["replay_date"]
        if params["stage"] == "historical_universe_reconstruction":
            return JobResult(
                status="finished",
                metrics={
                    "replay_date": replay_date,
                    "rows_inserted": 1,
                    "rows_updated": 0,
                },
            )
        return JobResult(
            status="finished",
            metrics={
                "replay_dates": [replay_date],
                "total_rows_inserted": 1,
                "total_rows_reused": 0,
                "total_fired_m4_signal_count": 1,
                "total_rejected_or_no_fire_count": 0,
                "date_results": [{"replay_date": replay_date, "rows_inserted": 1}],
            },
        )

    result = run_historical_cohort_reconstruction(
        session=session,
        fmp_adapter=object(),
        pattern_ids=["M4"],
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 5),
        schema="scratch_cohort",
        progress_artifact=tmp_path / "resume.json",
        resume_from_artifact=previous_artifact,
        completion_checker=lambda _session, replay_day: replay_day == date(2026, 1, 2),
        job_runner=fake_runner,
        universe_job_factory=_FakeJob,
        m4_replay_job_factory=_FakeJob,
        execution_mode="date-by-date",
        print_fn=lambda _message: None,
    )

    assert result.status == "finished"
    assert result.metrics["dates_skipped"] == 1
    assert result.metrics["dates_finished"] == 1
    assert {call["replay_date"] for call in calls} == {"2026-01-05"}
    artifact = json.loads((tmp_path / "resume.json").read_text())
    assert [row["status"] for row in artifact["date_results"]] == [
        "skipped",
        "finished",
    ]


def _seed_fmp_delisted_source(
    db_session,
    *,
    max_pages_reached: bool,
    date_cutoff_reached: bool = False,
    stop_after_delisted_before: str | None = None,
    run_status: str = "finished",
    started_at: datetime | None = None,
    add_record: bool = True,
) -> None:
    job = EvidenceJob(
        job_name=FMP_DELISTED_JOB_NAME,
        job_type="data_ingestion",
        owner_component="test",
    )
    db_session.add(job)
    db_session.flush()
    db_session.add(
        EvidenceJobRun(
            job_id=job.job_id,
            run_status=run_status,
            started_at=started_at or datetime(2026, 6, 1, tzinfo=timezone.utc),
            ended_at=(
                None
                if run_status == "running"
                else (started_at or datetime(2026, 6, 1, tzinfo=timezone.utc))
            ),
            metric_json=json.dumps(
                {
                    "max_pages_reached": max_pages_reached,
                    "date_cutoff_reached": date_cutoff_reached,
                    "stop_after_delisted_before": stop_after_delisted_before,
                }
            ),
        )
    )
    if not add_record:
        db_session.commit()
        return
    db_session.add(
        FmpDelistedCompanyRecord(
            symbol="DEAD",
            normalized_symbol="DEAD",
            company_name="Dead Co.",
            exchange="NASDAQ",
            exchange_key="NASDAQ",
            ipo_date=date(2020, 1, 1),
            delisted_date=date(2024, 1, 1),
            delisted_date_key="2024-01-01",
            page_number=0,
            page_limit=1000,
            row_status="active",
            exchange_relevance_status="us_operating_exchange",
            raw_payload_hash=stable_hash({"symbol": "DEAD"}),
            raw_payload_json=json.dumps({"symbol": "DEAD"}),
        )
    )
    db_session.commit()
