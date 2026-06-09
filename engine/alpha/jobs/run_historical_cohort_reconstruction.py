#!/usr/bin/env python3
"""Guarded historical daily cohort reconstruction runner.

This entrypoint orchestrates historical PIT universe reconstruction and audited
M4 replay with public/default write guards. Public/default M4 uses the
range-cached replay path; scratch/debug runs can explicitly retain the legacy
date-by-date path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from sqlalchemy import func

from alpha.data.config import ConfigError, FmpConfig, PolygonConfig
from alpha.data.fmp import FmpAdapter
from alpha.data.polygon import PolygonAdapter
from alpha.db.engine import (
    SchemaTargetError,
    get_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.db.models import (
    EvidenceJob,
    EvidenceJobRun,
    FmpDelistedCompanyRecord,
    SignalRegistry,
)
from alpha.jobs.contracts import BaseJob, JobResult
from alpha.jobs.fmp_delisted_companies import JOB_NAME as FMP_DELISTED_JOB_NAME
from alpha.jobs.historical_m4_replay import HistoricalM4ReplayJob
from alpha.jobs.historical_m4_replay import JOB_NAME as HISTORICAL_M4_REPLAY_JOB_NAME
from alpha.jobs.historical_universe_reconstruction import (
    HistoricalUniverseReconstructionJob,
)
from alpha.jobs.run_historical_m4_replay import HISTORICAL_M4_REPLAY_REQUIRED_TABLES
from alpha.jobs.run_market_path_backfill import CachedHistoricalPriceFmpAdapter
from alpha.jobs.run_historical_universe_reconstruction import (
    HISTORICAL_UNIVERSE_REQUIRED_TABLES,
)
from alpha.jobs.run_nightly_canonical import (
    MARKET_PATH_ALEMBIC_REVISION,
    verify_alembic_head,
)
from alpha.jobs.runner import run_job
from alpha.market_calendar import is_us_equity_session
from alpha.runtime_env import load_runtime_env


JOB_NAME = "historical_cohort_reconstruction"
SUPPORTED_PATTERN_IDS = frozenset({"M4"})
HISTORICAL_COHORT_REQUIRED_TABLES = tuple(
    dict.fromkeys(
        [
            *HISTORICAL_UNIVERSE_REQUIRED_TABLES,
            *HISTORICAL_M4_REPLAY_REQUIRED_TABLES,
        ]
    )
)
EXECUTION_MODE_RANGE_CACHED = "range_cached"
EXECUTION_MODE_DATE_BY_DATE = "date_by_date"
HISTORICAL_M4_RANGE_REPLAY_JOB_NAME = "historical_m4_range_replay"

PrintFn = Callable[[str], None]
JobRunner = Callable[..., JobResult]


@dataclass
class CohortReconstructionResult:
    status: str
    metrics: dict[str, Any]
    errors: list[dict[str, Any]]
    artifact_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "finished"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _trading_dates(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")
    days: list[date] = []
    cursor = start
    while cursor <= end:
        if is_us_equity_session(cursor):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _validate_pattern_ids(pattern_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(pattern.strip().upper() for pattern in pattern_ids))
    if not normalized:
        raise ValueError("At least one --pattern-id is required")
    unsupported = [pattern for pattern in normalized if pattern not in SUPPORTED_PATTERN_IDS]
    if unsupported:
        raise ValueError(
            "Unsupported historical cohort pattern(s): "
            f"{', '.join(unsupported)}. Only audited M4 replay is implemented; "
            "M1 is explicitly deferred."
        )
    return normalized


def _normalize_execution_mode(value: str | None) -> str:
    mode = (value or EXECUTION_MODE_RANGE_CACHED).strip().casefold().replace("-", "_")
    if mode not in {EXECUTION_MODE_RANGE_CACHED, EXECUTION_MODE_DATE_BY_DATE}:
        raise ValueError(
            "Unsupported historical cohort execution mode "
            f"{value!r}; expected range-cached or date-by-date."
        )
    return mode


def _validate_write_target(
    *,
    schema: str | None,
    confirm_live_write: bool,
    database_url: str | None,
    create_tables: bool = False,
    allow_partial_delisted_source: bool = False,
    allow_partial_universe: bool = False,
    revision_checker: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if schema and schema.strip().casefold() == "public":
        raise ValueError("historical cohort reconstruction refuses --schema public")
    if schema:
        return {
            "mode": "scratch",
            "schema": schema,
            "alembic": None,
        }
    if not confirm_live_write:
        raise ValueError(
            "Refusing public/default historical cohort reconstruction without "
            "--confirm-live-write"
        )
    if create_tables:
        raise ValueError("--create-tables is only valid with scratch --schema")
    if allow_partial_delisted_source:
        raise ValueError(
            "Public/default historical cohort reconstruction refuses "
            "--allow-partial-delisted-source"
        )
    if allow_partial_universe:
        raise ValueError(
            "Public/default historical cohort reconstruction refuses "
            "--allow-partial-universe"
        )
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL is required for public/default writes")
    checker = revision_checker or _verify_public_market_path_revision
    revision = checker(url)
    return {
        "mode": "public",
        "schema": None,
        "alembic": revision,
    }


def _verify_public_market_path_revision(url: str) -> dict[str, Any]:
    revision = verify_alembic_head(url, allow_default_off_m3_pending=True)
    current = revision.get("current") or []
    if current != [MARKET_PATH_ALEMBIC_REVISION]:
        raise ValueError(
            "Public/default historical cohort reconstruction requires Alembic "
            f"{MARKET_PATH_ALEMBIC_REVISION} with M3 unapplied/default-off; "
            f"current={current}"
        )
    return revision


def _preflight_public_delisted_source(
    session: Any,
    *,
    replay_start_date: date | None = None,
) -> dict[str, Any]:
    row_count = int(
        session.query(func.count(FmpDelistedCompanyRecord.fmp_delisted_company_id))
        .scalar()
        or 0
    )
    if row_count <= 0:
        raise ValueError(
            "Public/default historical cohort reconstruction requires a populated "
            "fmp_delisted_companies table before replay."
        )

    latest_running = (
        session.query(EvidenceJobRun)
        .join(EvidenceJob, EvidenceJob.job_id == EvidenceJobRun.job_id)
        .filter(EvidenceJob.job_name == FMP_DELISTED_JOB_NAME)
        .filter(EvidenceJobRun.run_status == "running")
        .order_by(EvidenceJobRun.started_at.desc(), EvidenceJobRun.ended_at.desc())
        .first()
    )
    if latest_running is not None:
        raise ValueError(
            "Latest FMP delisted-company ingestion is still running "
            f"(job_run_id={latest_running.job_run_id})."
        )

    latest_run = (
        session.query(EvidenceJobRun)
        .join(EvidenceJob, EvidenceJob.job_id == EvidenceJobRun.job_id)
        .filter(EvidenceJob.job_name == FMP_DELISTED_JOB_NAME)
        .filter(EvidenceJobRun.run_status == "finished")
        .order_by(EvidenceJobRun.started_at.desc(), EvidenceJobRun.ended_at.desc())
        .first()
    )
    if latest_run is None:
        raise ValueError(
            "Public/default historical cohort reconstruction requires a completed "
            "FMP delisted-company ingestion evidence run."
        )
    metrics = _json_dict(latest_run.metric_json)
    if not _json_false(metrics.get("max_pages_reached")):
        raise ValueError(
            "Latest FMP delisted-company ingestion must have "
            "max_pages_reached=false before public cohort reconstruction."
        )
    stop_before = _parse_optional_date(metrics.get("stop_after_delisted_before"))
    if metrics.get("date_cutoff_reached") is True:
        if replay_start_date is None or stop_before is None:
            raise ValueError(
                "Latest bounded FMP delisted-company ingestion is missing replay "
                "coverage metadata."
            )
        if stop_before > replay_start_date:
            raise ValueError(
                "Latest bounded FMP delisted-company ingestion only covers replay "
                f"dates on/after {stop_before}; requested start={replay_start_date}."
            )
    return {
        "fmp_delisted_companies_row_count": row_count,
        "fmp_delisted_latest_job_run_id": latest_run.job_run_id,
        "fmp_delisted_latest_run_status": latest_run.run_status,
        "fmp_delisted_max_pages_reached": metrics.get("max_pages_reached"),
        "fmp_delisted_date_cutoff_reached": metrics.get("date_cutoff_reached"),
        "fmp_delisted_stop_after_delisted_before": metrics.get(
            "stop_after_delisted_before"
        ),
    }


def run_historical_cohort_reconstruction(
    *,
    session: Any,
    fmp_adapter: Any,
    pattern_ids: Sequence[str],
    start_date: date,
    end_date: date,
    polygon_adapter: Any | None = None,
    run_timestamp: datetime | None = None,
    schema: str | None = None,
    progress_artifact: str | Path | None = None,
    allow_partial_delisted_source: bool = False,
    allow_partial_universe: bool = False,
    lookback_calendar_days: int = 430,
    skip_completed_dates: bool = False,
    resume_from_artifact: str | Path | None = None,
    completion_checker: Callable[[Any, date], bool] | None = None,
    job_runner: JobRunner = run_job,
    universe_job_factory: Callable[..., BaseJob] = HistoricalUniverseReconstructionJob,
    m4_replay_job_factory: Callable[..., BaseJob] = HistoricalM4ReplayJob,
    m4_range_replay_job_factory: Callable[..., BaseJob] | None = None,
    execution_mode: str = EXECUTION_MODE_RANGE_CACHED,
    print_fn: PrintFn = print,
) -> CohortReconstructionResult:
    patterns = _validate_pattern_ids(pattern_ids)
    normalized_execution_mode = _normalize_execution_mode(execution_mode)
    replay_dates = _trading_dates(start_date, end_date)
    resume_completed_dates = _completed_dates_from_artifact(resume_from_artifact)
    should_check_completion = skip_completed_dates or resume_from_artifact is not None
    is_completed = completion_checker or _has_completed_m4_replay_evidence
    started_at = datetime.now(timezone.utc)
    artifact_path = Path(progress_artifact) if progress_artifact else None
    polygon_fallback_configured = polygon_adapter is not None
    artifact: dict[str, Any] = {
        "job_name": JOB_NAME,
        "schema": schema or "default",
        "pattern_ids": list(patterns),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "trading_dates": [day.isoformat() for day in replay_dates],
        "started_at": started_at.isoformat(),
        "execution_mode": normalized_execution_mode,
        "polygon_fallback_configured": polygon_fallback_configured,
        "skip_completed_dates": skip_completed_dates,
        "resume_from_artifact": str(resume_from_artifact) if resume_from_artifact else None,
        "date_results": [],
        "summary": {},
    }
    _write_artifact(artifact_path, artifact)

    summary: dict[str, Any] = {
        "dates_total": len(replay_dates),
        "dates_finished": 0,
        "dates_failed": 0,
        "dates_partial": 0,
        "dates_skipped": 0,
        "execution_mode": normalized_execution_mode,
        "m4_rows_inserted_total": 0,
        "m4_rows_reused_total": 0,
        "m4_fired_signal_count_total": 0,
        "m4_rejected_or_no_fire_count_total": 0,
        "polygon_fallback_configured": polygon_fallback_configured,
        "polygon_fallback_count_total": 0,
        "missing_price_evidence_count_total": 0,
        "non_evaluable_ticker_count_total": 0,
        "universe_rows_inserted_total": 0,
        "universe_rows_updated_total": 0,
    }
    errors: list[dict[str, Any]] = []

    if normalized_execution_mode == EXECUTION_MODE_RANGE_CACHED:
        return _run_range_cached_m4_cohort(
            session=session,
            fmp_adapter=fmp_adapter,
            polygon_adapter=polygon_adapter,
            replay_dates=replay_dates,
            run_timestamp=run_timestamp,
            schema=schema,
            artifact=artifact,
            artifact_path=artifact_path,
            summary=summary,
            errors=errors,
            started_at=started_at,
            allow_partial_delisted_source=allow_partial_delisted_source,
            allow_partial_universe=allow_partial_universe,
            lookback_calendar_days=lookback_calendar_days,
            skip_completed_dates=skip_completed_dates,
            resume_from_artifact=resume_from_artifact,
            resume_completed_dates=resume_completed_dates,
            should_check_completion=should_check_completion,
            is_completed=is_completed,
            job_runner=job_runner,
            m4_range_replay_job_factory=m4_range_replay_job_factory,
            print_fn=print_fn,
        )

    for replay_day in replay_dates:
        date_started = datetime.now(timezone.utc)
        date_record: dict[str, Any] = {
            "replay_date": replay_day.isoformat(),
            "pattern_id": "M4",
            "status": "running",
            "started_at": date_started.isoformat(),
        }
        artifact["date_results"].append(date_record)
        _write_artifact(artifact_path, artifact)
        if should_check_completion:
            artifact_allows_skip = (
                resume_from_artifact is None
                or replay_day.isoformat() in resume_completed_dates
            )
            if artifact_allows_skip and is_completed(session, replay_day):
                date_record["status"] = "skipped"
                date_record["skip_reason"] = "completed_m4_replay_evidence_present"
                date_record["elapsed_seconds"] = 0.0
                summary["dates_skipped"] += 1
                _write_artifact(artifact_path, artifact)
                print_fn(
                    "date_skipped "
                    f"replay_date={replay_day.isoformat()} "
                    "reason=completed_m4_replay_evidence_present"
                )
                continue
        print_fn(f"date_start replay_date={replay_day.isoformat()} pattern_id=M4")

        def universe_progress(_event: str, payload: dict[str, Any]) -> None:
            date_record["historical_universe_progress_last"] = payload
            events = date_record.setdefault("historical_universe_progress_events", [])
            events.append(payload)
            if len(events) > 25:
                del events[:-25]
            _write_artifact(artifact_path, artifact)

        universe_job = universe_job_factory(
            session=session,
            replay_date=replay_day,
            run_timestamp=run_timestamp,
            allow_partial_delisted_source=allow_partial_delisted_source,
            progress_callback=universe_progress,
            persist_pre_replay_delisted_exclusions=False,
            compact_persisted_provenance=True,
        )
        universe_result = job_runner(
            session,
            universe_job,
            params={
                "source": JOB_NAME,
                "stage": "historical_universe_reconstruction",
                "replay_date": replay_day.isoformat(),
                "schema": schema,
                "allow_partial_delisted_source": allow_partial_delisted_source,
            },
        )
        universe_metrics = universe_result.metrics or {}
        date_record["universe_status"] = universe_result.status
        date_record["universe_metrics"] = universe_metrics
        summary["universe_rows_inserted_total"] += int(
            universe_metrics.get("rows_inserted") or 0
        )
        summary["universe_rows_updated_total"] += int(
            universe_metrics.get("rows_updated") or 0
        )
        _write_artifact(artifact_path, artifact)
        if not universe_result.ok:
            date_record["status"] = universe_result.status
            date_record["errors"] = universe_result.errors
            errors.extend(universe_result.errors)
            _mark_summary_failure(summary, universe_result.status)
            _finish_artifact(artifact_path, artifact, summary, "failed")
            print_fn(
                "date_failed "
                f"replay_date={replay_day.isoformat()} "
                f"stage=historical_universe_reconstruction "
                f"status={universe_result.status}"
            )
            return CohortReconstructionResult(
                status="failed",
                metrics=summary,
                errors=errors,
                artifact_path=str(artifact_path) if artifact_path else None,
            )

        def replay_progress(_event: str, payload: dict[str, Any]) -> None:
            date_record["historical_m4_replay_progress_last"] = payload
            events = date_record.setdefault("historical_m4_replay_progress_events", [])
            events.append(payload)
            if len(events) > 25:
                del events[:-25]
            _write_artifact(artifact_path, artifact)

        replay_job = m4_replay_job_factory(
            session=session,
            fmp_adapter=fmp_adapter,
            polygon_adapter=polygon_adapter,
            replay_dates=[replay_day],
            run_timestamp=run_timestamp,
            allow_partial_universe=allow_partial_universe,
            lookback_calendar_days=lookback_calendar_days,
            progress_callback=replay_progress,
        )
        replay_result = job_runner(
            session,
            replay_job,
            params={
                "source": JOB_NAME,
                "stage": "historical_m4_replay",
                "replay_date": replay_day.isoformat(),
                "schema": schema,
                "allow_partial_universe": allow_partial_universe,
                "lookback_calendar_days": lookback_calendar_days,
                "polygon_fallback_configured": polygon_adapter is not None,
            },
        )
        replay_metrics = replay_result.metrics or {}
        date_record["m4_replay_status"] = replay_result.status
        date_record["m4_replay_metrics"] = replay_metrics
        date_detail = _first_date_result(replay_metrics)
        if date_detail:
            date_record["m4_date_metrics"] = date_detail
        summary["m4_rows_inserted_total"] += int(
            replay_metrics.get("total_rows_inserted") or 0
        )
        summary["m4_rows_reused_total"] += int(replay_metrics.get("total_rows_reused") or 0)
        summary["m4_fired_signal_count_total"] += int(
            replay_metrics.get("total_fired_m4_signal_count") or 0
        )
        summary["m4_rejected_or_no_fire_count_total"] += int(
            replay_metrics.get("total_rejected_or_no_fire_count") or 0
        )
        summary["polygon_fallback_count_total"] += int(
            replay_metrics.get("total_polygon_fallback_count") or 0
        )
        summary["missing_price_evidence_count_total"] += int(
            replay_metrics.get("total_missing_price_evidence_count") or 0
        )
        summary["non_evaluable_ticker_count_total"] += int(
            replay_metrics.get("total_non_evaluable_ticker_count") or 0
        )
        if not replay_result.ok:
            date_record["status"] = replay_result.status
            date_record["errors"] = replay_result.errors
            errors.extend(replay_result.errors)
            _mark_summary_failure(summary, replay_result.status)
            _finish_artifact(artifact_path, artifact, summary, "failed")
            print_fn(
                "date_failed "
                f"replay_date={replay_day.isoformat()} "
                f"stage=historical_m4_replay status={replay_result.status}"
            )
            return CohortReconstructionResult(
                status="failed",
                metrics=summary,
                errors=errors,
                artifact_path=str(artifact_path) if artifact_path else None,
            )

        if hasattr(session, "commit"):
            session.commit()
        elapsed = (datetime.now(timezone.utc) - date_started).total_seconds()
        date_record["status"] = "finished"
        date_record["elapsed_seconds"] = elapsed
        summary["dates_finished"] += 1
        _write_artifact(artifact_path, artifact)
        print_fn(
            "date_finished "
            f"replay_date={replay_day.isoformat()} "
            f"m4_inserted={replay_metrics.get('total_rows_inserted', 0)} "
            f"m4_reused={replay_metrics.get('total_rows_reused', 0)} "
            f"elapsed_seconds={elapsed:.3f}"
        )

    _finish_artifact(artifact_path, artifact, summary, "finished")
    return CohortReconstructionResult(
        status="finished",
        metrics=summary,
        errors=errors,
        artifact_path=str(artifact_path) if artifact_path else None,
    )


def _run_range_cached_m4_cohort(
    *,
    session: Any,
    fmp_adapter: Any,
    polygon_adapter: Any | None,
    replay_dates: Sequence[date],
    run_timestamp: datetime | None,
    schema: str | None,
    artifact: dict[str, Any],
    artifact_path: Path | None,
    summary: dict[str, Any],
    errors: list[dict[str, Any]],
    started_at: datetime,
    allow_partial_delisted_source: bool,
    allow_partial_universe: bool,
    lookback_calendar_days: int,
    skip_completed_dates: bool,
    resume_from_artifact: str | Path | None,
    resume_completed_dates: set[str],
    should_check_completion: bool,
    is_completed: Callable[[Any, date], bool],
    job_runner: JobRunner,
    m4_range_replay_job_factory: Callable[..., BaseJob] | None,
    print_fn: PrintFn,
) -> CohortReconstructionResult:
    active_dates: list[date] = []
    date_records: dict[str, dict[str, Any]] = {}
    for replay_day in replay_dates:
        replay_iso = replay_day.isoformat()
        date_record: dict[str, Any] = {
            "replay_date": replay_iso,
            "pattern_id": "M4",
            "execution_mode": EXECUTION_MODE_RANGE_CACHED,
            "status": "pending",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        artifact["date_results"].append(date_record)
        date_records[replay_iso] = date_record
        if should_check_completion:
            artifact_allows_skip = (
                resume_from_artifact is None
                or replay_iso in resume_completed_dates
            )
            if artifact_allows_skip and is_completed(session, replay_day):
                date_record["status"] = "skipped"
                date_record["m4_replay_status"] = "skipped"
                date_record["skip_reason"] = "completed_m4_replay_evidence_present"
                date_record["elapsed_seconds"] = 0.0
                summary["dates_skipped"] += 1
                print_fn(
                    "date_skipped "
                    f"replay_date={replay_iso} "
                    "reason=completed_m4_replay_evidence_present"
                )
                continue
        active_dates.append(replay_day)
    _write_artifact(artifact_path, artifact)

    if not active_dates:
        _finish_artifact(artifact_path, artifact, summary, "finished")
        return CohortReconstructionResult(
            status="finished",
            metrics=summary,
            errors=errors,
            artifact_path=str(artifact_path) if artifact_path else None,
        )

    def range_progress(_event: str, payload: dict[str, Any]) -> None:
        artifact["historical_m4_range_replay_progress_last"] = payload
        events = artifact.setdefault("historical_m4_range_replay_progress_events", [])
        events.append(payload)
        if len(events) > 50:
            del events[:-50]
        replay_iso = payload.get("replay_date")
        if replay_iso in date_records:
            date_record = date_records[replay_iso]
            date_record["historical_m4_range_replay_progress_last"] = payload
            date_events = date_record.setdefault(
                "historical_m4_range_replay_progress_events",
                [],
            )
            date_events.append(payload)
            if len(date_events) > 25:
                del date_events[:-25]
        _write_artifact(artifact_path, artifact)

    if m4_range_replay_job_factory is None:
        from alpha.jobs.run_historical_m4_range_replay import (  # noqa: PLC0415
            HistoricalM4RangeReplayJob,
        )

        m4_range_replay_job_factory = HistoricalM4RangeReplayJob

    range_job = m4_range_replay_job_factory(
        session=session,
        fmp_adapter=fmp_adapter,
        polygon_adapter=polygon_adapter,
        replay_dates=active_dates,
        run_timestamp=run_timestamp,
        allow_partial_delisted_source=allow_partial_delisted_source,
        allow_partial_universe=allow_partial_universe,
        lookback_calendar_days=lookback_calendar_days,
        skip_completed_dates=False,
        progress_callback=range_progress,
    )
    print_fn(
        "range_replay_start "
        f"start_date={active_dates[0].isoformat()} "
        f"end_date={active_dates[-1].isoformat()} "
        f"date_count={len(active_dates)}"
    )
    range_result = job_runner(
        session,
        range_job,
        params={
            "source": JOB_NAME,
            "stage": "historical_m4_range_replay",
            "schema": schema,
            "start_date": active_dates[0].isoformat(),
            "end_date": active_dates[-1].isoformat(),
            "replay_dates": [day.isoformat() for day in active_dates],
            "allow_partial_delisted_source": allow_partial_delisted_source,
            "allow_partial_universe": allow_partial_universe,
            "lookback_calendar_days": lookback_calendar_days,
            "polygon_fallback_configured": polygon_adapter is not None,
            "execution_mode": EXECUTION_MODE_RANGE_CACHED,
        },
    )
    range_metrics = range_result.metrics or {}
    artifact["historical_m4_range_replay_status"] = range_result.status
    artifact["historical_m4_range_replay_metrics"] = range_metrics
    summary["range_fetch_metrics"] = range_metrics.get("fmp_fetch") or {}
    summary["range_validation"] = range_metrics.get("validation") or {}
    summary["range_stage_timing_seconds"] = (
        range_metrics.get("stage_timing_seconds") or {}
    )
    summary["range_unique_ticker_count"] = int(
        range_metrics.get("unique_ticker_count") or 0
    )
    summary["range_date_ticker_equivalent_fetch_count"] = int(
        range_metrics.get("date_ticker_equivalent_fetch_count") or 0
    )
    universe_metrics = range_metrics.get("universe") or {}
    summary["universe_rows_inserted_total"] += int(
        (universe_metrics.get("persistence") or {}).get("rows_inserted") or 0
    )
    summary["universe_rows_updated_total"] += int(
        (universe_metrics.get("persistence") or {}).get("rows_updated") or 0
    )
    summary["m4_rows_inserted_total"] += int(
        range_metrics.get("total_rows_inserted") or 0
    )
    summary["m4_rows_reused_total"] += int(range_metrics.get("total_rows_reused") or 0)
    summary["m4_fired_signal_count_total"] += int(
        range_metrics.get("total_fired_m4_signal_count") or 0
    )
    summary["m4_rejected_or_no_fire_count_total"] += int(
        range_metrics.get("total_rejected_or_no_fire_count") or 0
    )
    summary["polygon_fallback_count_total"] += _sum_date_metric(
        range_metrics,
        "polygon_fallback_count",
    )
    summary["missing_price_evidence_count_total"] += _sum_date_metric(
        range_metrics,
        "missing_price_evidence_count",
    )
    summary["non_evaluable_ticker_count_total"] += _sum_date_metric(
        range_metrics,
        "non_evaluable_ticker_count",
    )

    for date_detail in range_metrics.get("date_results") or []:
        replay_iso = date_detail.get("replay_date")
        if replay_iso not in date_records:
            continue
        date_record = date_records[replay_iso]
        date_status = _date_status_from_range_detail(date_detail)
        date_record["status"] = date_status
        date_record["m4_replay_status"] = date_status
        date_record["m4_date_metrics"] = date_detail
        date_record["m4_replay_metrics"] = {
            "stage": "historical_m4_range_replay",
            "range_replay": True,
            "range_fetch": range_metrics.get("fmp_fetch") or {},
            "validation": range_metrics.get("validation") or {},
            "stage_timing_seconds": range_metrics.get("stage_timing_seconds") or {},
        }
        date_record["elapsed_seconds"] = (
            (datetime.now(timezone.utc) - started_at).total_seconds()
        )
        if date_status == "finished":
            summary["dates_finished"] += 1
        else:
            _mark_summary_failure(summary, date_status)

    unfinished = [
        row
        for row in date_records.values()
        if row.get("status") == "pending"
    ]
    for row in unfinished:
        row["status"] = range_result.status
        row["m4_replay_status"] = range_result.status
        _mark_summary_failure(summary, range_result.status)

    if range_result.errors:
        errors.extend(range_result.errors)
    if not range_result.ok:
        _finish_artifact(artifact_path, artifact, summary, "failed")
        print_fn(
            "range_replay_failed "
            f"status={range_result.status} "
            f"date_count={len(active_dates)}"
        )
        return CohortReconstructionResult(
            status="failed",
            metrics=summary,
            errors=errors,
            artifact_path=str(artifact_path) if artifact_path else None,
        )

    if hasattr(session, "commit"):
        session.commit()
    _finish_artifact(artifact_path, artifact, summary, "finished")
    print_fn(
        "range_replay_finished "
        f"date_count={len(active_dates)} "
        f"m4_inserted={range_metrics.get('total_rows_inserted', 0)} "
        f"m4_reused={range_metrics.get('total_rows_reused', 0)}"
    )
    return CohortReconstructionResult(
        status="finished",
        metrics=summary,
        errors=errors,
        artifact_path=str(artifact_path) if artifact_path else None,
    )


def _first_date_result(metrics: dict[str, Any]) -> dict[str, Any] | None:
    rows = metrics.get("date_results") or []
    if rows:
        return rows[0]
    return None


def _sum_date_metric(metrics: dict[str, Any], key: str) -> int:
    return sum(
        int((row or {}).get(key) or 0)
        for row in metrics.get("date_results") or []
    )


def _date_status_from_range_detail(date_detail: dict[str, Any]) -> str:
    explicit_status = date_detail.get("status")
    if explicit_status in {"finished", "partial_failed", "failed", "skipped"}:
        return str(explicit_status)
    if int(date_detail.get("fetch_error_count") or 0) <= 0:
        return "finished"
    has_evaluable_outputs = any(
        int(date_detail.get(key) or 0) > 0
        for key in (
            "assembled_count",
            "fired_m4_signal_count",
            "rejected_or_no_fire_count",
            "rows_inserted",
            "rows_reused",
        )
    )
    return "partial_failed" if has_evaluable_outputs else "failed"


def _mark_summary_failure(summary: dict[str, Any], status: str) -> None:
    if status == "partial_failed":
        summary["dates_partial"] += 1
    else:
        summary["dates_failed"] += 1


def _completed_dates_from_artifact(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(Path(path).read_text())
    completed: set[str] = set()
    for row in payload.get("date_results") or []:
        if row.get("status") == "finished" and row.get("m4_replay_status") == "finished":
            replay_date = row.get("replay_date")
            if replay_date:
                completed.add(replay_date)
    return completed


def _has_completed_m4_replay_evidence(session: Any, replay_day: date) -> bool:
    if _duplicate_m4_signal_identity_groups(session, replay_day) > 0:
        return False
    runs = (
        session.query(EvidenceJobRun)
        .join(EvidenceJob, EvidenceJob.job_id == EvidenceJobRun.job_id)
        .filter(
            EvidenceJob.job_name.in_(
                (
                    HISTORICAL_M4_REPLAY_JOB_NAME,
                    HISTORICAL_M4_RANGE_REPLAY_JOB_NAME,
                )
            ),
            EvidenceJobRun.run_status == "finished",
        )
        .order_by(EvidenceJobRun.started_at.desc(), EvidenceJobRun.ended_at.desc())
        .all()
    )
    for run in runs:
        if _metrics_cover_replay_date(_json_dict(run.metric_json), replay_day):
            return True
    return False


def _duplicate_m4_signal_identity_groups(session: Any, replay_day: date) -> int:
    return int(
        session.query(
            SignalRegistry.pattern_id,
            SignalRegistry.ticker,
            SignalRegistry.signal_identity_hash,
            func.count().label("row_count"),
        )
        .filter(
            SignalRegistry.pattern_id == "M4",
            SignalRegistry.trading_date == replay_day.isoformat(),
        )
        .group_by(
            SignalRegistry.pattern_id,
            SignalRegistry.ticker,
            SignalRegistry.signal_identity_hash,
        )
        .having(func.count() > 1)
        .count()
    )


def _metrics_cover_replay_date(metrics: dict[str, Any], replay_day: date) -> bool:
    replay_iso = replay_day.isoformat()
    replay_dates = metrics.get("replay_dates") or []
    if replay_iso in replay_dates:
        return True
    for row in metrics.get("date_results") or []:
        if row.get("replay_date") == replay_iso:
            return True
    return False


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_false(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"false", "0", "no"}
    if isinstance(value, int):
        return value == 0
    return False


def _parse_optional_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _write_artifact(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _finish_artifact(
    path: Path | None,
    artifact: dict[str, Any],
    summary: dict[str, Any],
    status: str,
) -> None:
    artifact["status"] = status
    artifact["summary"] = summary
    artifact["ended_at"] = datetime.now(timezone.utc).isoformat()
    _write_artifact(path, artifact)


def _default_artifact_path() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"/tmp/historical_cohort_reconstruction_{ts}.json"


def _run_live(args: argparse.Namespace) -> int:
    load_runtime_env()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if args.schema:
        os.environ["ALPHA_DB_SCHEMA"] = args.schema
        reset_globals()
    target_schema = args.schema or os.environ.get("ALPHA_DB_SCHEMA")
    try:
        target = _validate_write_target(
            schema=target_schema,
            confirm_live_write=args.confirm_live_write,
            database_url=os.environ.get("DATABASE_URL"),
            create_tables=args.create_tables,
            allow_partial_delisted_source=args.allow_partial_delisted_source,
            allow_partial_universe=args.allow_partial_universe,
        )
        patterns = _validate_pattern_ids(args.pattern_id)
        execution_mode = _normalize_execution_mode(args.execution_mode)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    if target["mode"] == "public" and execution_mode != EXECUTION_MODE_RANGE_CACHED:
        print(
            "ERROR: public/default historical cohort reconstruction requires "
            "--execution-mode range-cached for M4."
        )
        return 1

    if target_schema:
        try:
            prepare_writable_schema_target(
                schema=target_schema,
                create_tables=args.create_tables,
                required_tables=HISTORICAL_COHORT_REQUIRED_TABLES,
            )
        except (SchemaTargetError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1

    polygon_adapter = _optional_polygon_adapter()
    if target["mode"] == "public" and polygon_adapter is None:
        print(
            "ERROR: public/default historical cohort reconstruction requires "
            "Polygon fallback configuration for historical M4 price evidence."
        )
        return 1

    session = get_session()
    artifact_path = args.progress_artifact or _default_artifact_path()
    try:
        public_preflight = None
        if target["mode"] == "public":
            try:
                public_preflight = _preflight_public_delisted_source(
                    session,
                    replay_start_date=_parse_date(args.start_date),
                )
            except ValueError as exc:
                print(f"ERROR: {exc}")
                return 1
        try:
            fmp_adapter = CachedHistoricalPriceFmpAdapter(
                FmpAdapter(FmpConfig.from_env())
            )
        except ConfigError as exc:
            print(f"ERROR: {exc}")
            return 1

        result = run_historical_cohort_reconstruction(
            session=session,
            fmp_adapter=fmp_adapter,
            polygon_adapter=polygon_adapter,
            pattern_ids=patterns,
            start_date=_parse_date(args.start_date),
            end_date=_parse_date(args.end_date),
            run_timestamp=_parse_timestamp(args.run_timestamp),
            schema=target_schema,
            progress_artifact=artifact_path,
            allow_partial_delisted_source=args.allow_partial_delisted_source,
            allow_partial_universe=args.allow_partial_universe,
            lookback_calendar_days=args.lookback_calendar_days,
            skip_completed_dates=args.skip_completed_dates,
            resume_from_artifact=args.resume_from_artifact,
            execution_mode=execution_mode,
        )
        metrics = result.metrics or {}
        print(f"Status:                 {result.status}")
        print(f"Target mode:            {target['mode']}")
        print(f"Schema:                 {target_schema or 'default'}")
        print(f"Execution mode:         {metrics.get('execution_mode')}")
        print(f"Patterns:               {', '.join(patterns)}")
        print(f"Start date:             {args.start_date}")
        print(f"End date:               {args.end_date}")
        print(f"Dates finished:         {metrics.get('dates_finished')}")
        print(f"Dates failed:           {metrics.get('dates_failed')}")
        print(f"Dates skipped:          {metrics.get('dates_skipped')}")
        print(f"M4 rows inserted:       {metrics.get('m4_rows_inserted_total')}")
        print(f"M4 rows reused:         {metrics.get('m4_rows_reused_total')}")
        print(f"M4 fired signals:       {metrics.get('m4_fired_signal_count_total')}")
        print(
            "M4 rejected/no-fire:    "
            f"{metrics.get('m4_rejected_or_no_fire_count_total')}"
        )
        print(f"Polygon fallback:       {metrics.get('polygon_fallback_configured')}")
        print(f"Polygon fallbacks:      {metrics.get('polygon_fallback_count_total')}")
        print(
            "Missing price evidence: "
            f"{metrics.get('missing_price_evidence_count_total')}"
        )
        print(
            "Non-evaluable tickers:  "
            f"{metrics.get('non_evaluable_ticker_count_total')}"
        )
        if metrics.get("range_unique_ticker_count") is not None:
            print(f"Range unique tickers:   {metrics.get('range_unique_ticker_count')}")
        if metrics.get("range_date_ticker_equivalent_fetch_count") is not None:
            print(
                "Range date-ticker eq.:  "
                f"{metrics.get('range_date_ticker_equivalent_fetch_count')}"
            )
        print(f"Artifact:               {artifact_path}")
        if target.get("alembic"):
            print(f"Alembic current:        {target['alembic'].get('current')}")
        if public_preflight:
            print(
                "FMP delisted rows:     "
                f"{public_preflight['fmp_delisted_companies_row_count']}"
            )
        if result.errors:
            print("Errors:")
            for error in result.errors[:20]:
                print(f"  {error}")
        return 0 if result.ok else 1
    finally:
        session.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct historical daily M4 fired cohorts with guarded PIT "
            "universe reconstruction and audited M4 replay."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--database-url")
    parser.add_argument("--schema")
    parser.add_argument("--create-tables", action="store_true")
    parser.add_argument("--confirm-live-write", action="store_true")
    parser.add_argument("--pattern-id", action="append", default=["M4"])
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--run-timestamp")
    parser.add_argument("--lookback-calendar-days", type=int, default=430)
    parser.add_argument(
        "--execution-mode",
        choices=("range-cached", "date-by-date"),
        default="range-cached",
        help=(
            "Use range-cached M4 replay by default; date-by-date is retained "
            "for scratch/debug runs only."
        ),
    )
    parser.add_argument("--allow-partial-delisted-source", action="store_true")
    parser.add_argument("--allow-partial-universe", action="store_true")
    parser.add_argument("--skip-completed-dates", action="store_true")
    parser.add_argument("--resume-from-artifact")
    parser.add_argument("--progress-artifact")
    return parser.parse_args(argv)


def _optional_polygon_adapter() -> PolygonAdapter | None:
    try:
        return PolygonAdapter(PolygonConfig.from_env())
    except ConfigError:
        return None


def main(argv: list[str] | None = None) -> int:
    previous_env = {
        "DATABASE_URL": os.environ.get("DATABASE_URL"),
        "ALPHA_DB_SCHEMA": os.environ.get("ALPHA_DB_SCHEMA"),
    }
    try:
        args = _parse_args(argv or sys.argv[1:])
        if args.live:
            return _run_live(args)
        return 1
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_globals()


if __name__ == "__main__":
    raise SystemExit(main())
