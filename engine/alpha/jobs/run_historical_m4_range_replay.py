#!/usr/bin/env python3
"""Range-level cached historical M4 replay runner.

This entrypoint is the scratch-first scaling path for historical M4 replay. It
loads the PIT universe source intervals once for a date range, fetches each
union ticker's FMP EOD range once, then slices cached bars per replay date
before calling the existing M4 assembler and detector orchestration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence
from uuid import uuid4

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from alpha.assembly.m4_daily import DailyBar, assemble_m4_daily
from alpha.data.config import ConfigError, FmpConfig, PolygonConfig
from alpha.data.contracts import stable_hash
from alpha.data.fmp import FmpAdapter, HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.data.polygon import PolygonAdapter
from alpha.db.engine import (
    SchemaTargetError,
    get_session,
    prepare_writable_schema_target,
    reset_globals,
)
from alpha.db.models import (
    CanonicalUniverseScan,
    DataLineage,
    EvidenceJob,
    EvidenceJobRun,
    FeatureSnapshot,
    FmpDelistedCompanyRecord,
    HistoricalUniverseReconstruction,
    SignalRegistry,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.evidence.writer import record_data_lineage
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.detector_orchestration import (
    DetectorDiagnostics,
    _input_asof_ceiling,
    _result_guard_passed,
    check_lookahead_guard,
    compute_signal_identity_hash,
)
from alpha.jobs.historical_m4_replay import (
    HISTORICAL_REPLAY_MIN_DATE,
    JOB_NAME as HISTORICAL_M4_REPLAY_JOB_NAME,
    LOOKBACK_CALENDAR_DAYS,
    RECONSTRUCTION_METHOD,
    SOURCE_UNIVERSE_METHOD,
    HistoricalM4ReplayJob,
    BAR_PROVIDER_POLICY,
    FMP_PRICE_BASIS,
    _bar_metadata_from_lineage,
    _daily_bar_lineage_payload,
    _partial_universe_reason,
    _replay_scan_id,
    _replay_snapshot_id,
    _signal_ids_for_replay,
    _signal_tickers,
)
from alpha.jobs.historical_universe_reconstruction import (
    DERIVED_ENDPOINT,
    RECONSTRUCTION_METHOD as UNIVERSE_RECONSTRUCTION_METHOD,
    HistoricalUniverseReconstructionJob,
    _evaluate_interval,
    bulk_persist_historical_universe_reconstructions,
)
from alpha.jobs.m4_daily import _assembly_metrics
from alpha.jobs.run_historical_cohort_reconstruction import (
    HISTORICAL_COHORT_REQUIRED_TABLES,
    _has_completed_m4_replay_evidence,
)
from alpha.jobs.run_market_path_backfill import CachedHistoricalPriceFmpAdapter
from alpha.jobs.runner import run_job
from alpha.market_calendar import (
    is_us_equity_session,
    next_us_equity_session,
    us_equity_session_close_timestamp,
)
from alpha.patterns.m4 import M4Detector
from alpha.runtime_env import load_runtime_env


JOB_NAME = "historical_m4_range_replay"
SUPPORTED_PATTERN_IDS = frozenset({"M4"})
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class HistoricalM4RangeReplayResult:
    status: str
    metrics: dict[str, Any]
    errors: list[dict[str, Any]]

    @property
    def ok(self) -> bool:
        return self.status == "finished"


@dataclass
class _BulkDetectionRecord:
    result: Any
    ticker: str
    feature_payload: dict[str, Any]
    feature_hash: str
    feature_json: str
    output_hash: str
    data_lineage_ids: list[str]
    universe_snapshot_id: str | None
    next_execution_session: str | None
    signal_identity_hash: str | None = None
    signal_identity_components: dict[str, Any] | None = None


class HistoricalM4RangeReplayJob(BaseJob):
    """Replay M4 for a whole historical date range with range-level bar caching."""

    def __init__(
        self,
        *,
        session: Session,
        fmp_adapter: Any,
        replay_dates: Sequence[date],
        polygon_adapter: Any | None = None,
        run_timestamp: datetime | None = None,
        allow_partial_delisted_source: bool = False,
        allow_partial_universe: bool = False,
        lookback_calendar_days: int = LOOKBACK_CALENDAR_DAYS,
        skip_completed_dates: bool = False,
        progress_callback: ProgressCallback | None = None,
        progress_every: int = 25,
    ) -> None:
        dates = sorted(set(replay_dates))
        if not dates:
            raise ValueError("HistoricalM4RangeReplayJob requires replay dates")
        bad_dates = [day for day in dates if day < HISTORICAL_REPLAY_MIN_DATE]
        if bad_dates:
            raise ValueError("historical M4 replay starts at 2024-01-01")
        self._session = session
        self._fmp = fmp_adapter
        self._polygon = polygon_adapter
        self._replay_dates = dates
        self._run_timestamp = _aware_utc(run_timestamp)
        self._allow_partial_delisted_source = allow_partial_delisted_source
        self._allow_partial_universe = allow_partial_universe
        self._lookback_calendar_days = lookback_calendar_days
        self._skip_completed_dates = skip_completed_dates
        self._progress_callback = progress_callback
        self._progress_every = max(int(progress_every), 1)
        self._started_perf = perf_counter()
        self._progress_events: list[dict[str, Any]] = []

    @property
    def job_name(self) -> str:
        return JOB_NAME

    @property
    def job_type(self) -> str:
        return "historical_replay"

    @property
    def owner_component(self) -> str:
        return "historical_replay"

    def run(self, ctx: JobContext) -> JobResult:
        active_dates = [
            day
            for day in self._replay_dates
            if not (
                self._skip_completed_dates
                and _has_completed_m4_replay_evidence(self._session, day)
            )
        ]
        skipped_dates = [
            day.isoformat()
            for day in self._replay_dates
            if day not in active_dates
        ]
        metrics: dict[str, Any] = {
            "replay_dates": [day.isoformat() for day in self._replay_dates],
            "active_replay_dates": [day.isoformat() for day in active_dates],
            "skipped_replay_dates": skipped_dates,
            "allow_partial_delisted_source": self._allow_partial_delisted_source,
            "allow_partial_universe": self._allow_partial_universe,
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "source_universe_method": SOURCE_UNIVERSE_METHOD,
            "date_results": [],
            "stage_timing_seconds": {},
            "errors": [],
        }
        if not active_dates:
            metrics.update(
                {
                    "status_reason": "all_dates_skipped_completed",
                    "dates_finished": 0,
                    "dates_skipped": len(skipped_dates),
                    "total_rows_inserted": 0,
                    "total_rows_reused": 0,
                }
            )
            return JobResult(status="finished", metrics=metrics, errors=[])

        errors: list[dict[str, Any]] = []
        universe_started = perf_counter()
        universe = self._reconstruct_universe_range(active_dates, ctx)
        metrics["stage_timing_seconds"]["universe_load_reconstruction_seconds"] = round(
            perf_counter() - universe_started,
            6,
        )
        metrics["universe"] = universe["metrics"]
        if universe["status"] != "finished":
            errors.extend(universe["errors"])
            metrics["errors"] = errors
            return JobResult(status=universe["status"], metrics=metrics, errors=errors)

        included_rows_by_date = self._load_included_universe_rows(active_dates)
        partial_by_date = {
            day: _partial_universe_reason(rows)
            for day, rows in included_rows_by_date.items()
        }
        blocking_partial = {
            day.isoformat(): reason
            for day, reason in partial_by_date.items()
            if reason and not self._allow_partial_universe
        }
        if blocking_partial:
            errors.append(
                {
                    "stage": "historical_universe",
                    "error_type": "partial_historical_universe",
                    "partial_universe_by_date": blocking_partial,
                    "message": (
                        "historical universe reconstruction source is partial; "
                        "allow partial only for bounded scratch probes"
                    ),
                }
            )
            metrics["errors"] = errors
            return JobResult(status="failed", metrics=metrics, errors=errors)

        replay_helper = HistoricalM4ReplayJob(
            session=self._session,
            fmp_adapter=self._fmp,
            polygon_adapter=self._polygon,
            replay_dates=active_dates,
            run_timestamp=self._run_timestamp,
            allow_partial_universe=self._allow_partial_universe,
            lookback_calendar_days=self._lookback_calendar_days,
        )
        snapshots_by_date = self._ensure_replay_scans(
            replay_helper,
            included_rows_by_date,
            ctx,
            partial_by_date,
        )
        unique_tickers = sorted(
            {
                str(_snap_attr(snapshot, "ticker")).upper()
                for snapshots in snapshots_by_date.values()
                for snapshot in snapshots
                if _snap_attr(snapshot, "ticker")
            }
        )
        metrics["unique_ticker_count"] = len(unique_tickers)
        metrics["date_ticker_equivalent_fetch_count"] = sum(
            len(snapshots) for snapshots in snapshots_by_date.values()
        )

        fetch_started = perf_counter()
        range_bars = self._fetch_range_bars(
            unique_tickers,
            active_dates,
            replay_helper=replay_helper,
            job_run_id=ctx.job_run_id,
        )
        metrics["stage_timing_seconds"]["fmp_fetch_seconds"] = round(
            perf_counter() - fetch_started,
            6,
        )
        metrics["fmp_fetch"] = range_bars["metrics"]

        total_inserted = 0
        total_reused = 0
        total_fired = 0
        total_rejected = 0
        total_historical_universe_included = 0
        total_m4_evaluable = 0
        total_m4_non_evaluable = 0
        total_missing_price_evidence = 0
        total_polygon_fallback = 0
        output_hashes: list[str] = []
        for replay_day in active_dates:
            date_result = self._run_one_replay_date(
                replay_day,
                ctx,
                replay_helper,
                included_rows_by_date[replay_day],
                snapshots_by_date[replay_day],
                range_bars["bars_by_ticker"],
                range_bars["fetch_by_ticker"],
                partial_by_date.get(replay_day),
            )
            metrics["date_results"].append(date_result.metrics)
            output_hashes.append(stable_hash(date_result.metrics))
            errors.extend(date_result.errors)
            total_inserted += int(date_result.metrics.get("rows_inserted") or 0)
            total_reused += int(date_result.metrics.get("rows_reused") or 0)
            total_fired += int(date_result.metrics.get("fired_m4_signal_count") or 0)
            total_rejected += int(date_result.metrics.get("rejected_or_no_fire_count") or 0)
            total_historical_universe_included += int(
                date_result.metrics.get("historical_universe_included_count")
                or date_result.metrics.get("universe_included_count")
                or 0
            )
            total_m4_evaluable += int(
                date_result.metrics.get("m4_evaluable_count") or 0
            )
            total_m4_non_evaluable += int(
                date_result.metrics.get("m4_non_evaluable_count")
                or date_result.metrics.get("non_evaluable_ticker_count")
                or 0
            )
            total_missing_price_evidence += int(
                date_result.metrics.get("missing_price_evidence_count") or 0
            )
            total_polygon_fallback += int(
                date_result.metrics.get("polygon_fallback_count") or 0
            )

        validation_started = perf_counter()
        validation = _validate_range_replay(self._session, active_dates)
        metrics["stage_timing_seconds"]["validation_seconds"] = round(
            perf_counter() - validation_started,
            6,
        )
        metrics.update(
            {
                "dates_finished": len(active_dates),
                "dates_skipped": len(skipped_dates),
                "total_rows_inserted": total_inserted,
                "total_rows_reused": total_reused,
                "total_fired_m4_signal_count": total_fired,
                "total_rejected_or_no_fire_count": total_rejected,
                "total_historical_universe_included_count": (
                    total_historical_universe_included
                ),
                "total_m4_evaluable_count": total_m4_evaluable,
                "total_m4_non_evaluable_count": total_m4_non_evaluable,
                "total_missing_price_evidence_count": total_missing_price_evidence,
                "total_polygon_fallback_count": total_polygon_fallback,
                "validation": validation,
                "progress_events": self._progress_events[-200:],
                "total_seconds": round(perf_counter() - self._started_perf, 6),
            }
        )
        non_evaluable_samples = _summarize_non_evaluable_samples(
            metrics["date_results"],
        )
        metrics["non_evaluable_price_evidence_samples"] = non_evaluable_samples
        metrics["non_evaluable_symbol_count"] = len(non_evaluable_samples)
        metrics["coverage_status"] = _coverage_status(
            missing_price_evidence_count=total_missing_price_evidence,
            m4_evaluable_count=total_m4_evaluable,
            hard_error_count=len(errors),
        )
        metrics["completion_classification"] = _completion_classification(
            missing_price_evidence_count=total_missing_price_evidence,
            m4_evaluable_count=total_m4_evaluable,
            hard_error_count=len(errors),
        )
        status = "finished"
        if errors:
            status = "partial_failed" if total_fired or total_rejected else "failed"
        elif total_missing_price_evidence and total_m4_evaluable <= 0:
            status = "failed"
            errors = _missing_price_evidence_errors_from_date_results(
                metrics["date_results"],
            )
        metrics["errors"] = errors[:50]
        return JobResult(
            status=status,
            metrics=metrics,
            errors=errors,
            input_hashes={
                "historical_m4_range_replay_input": stable_hash(
                    {
                        "replay_dates": [day.isoformat() for day in active_dates],
                        "lookback_calendar_days": self._lookback_calendar_days,
                        "unique_tickers": unique_tickers,
                    }
                )
            },
            output_hashes={
                "historical_m4_range_replay_output": stable_hash(output_hashes)
            },
        )

    def _reconstruct_universe_range(
        self,
        replay_dates: Sequence[date],
        ctx: JobContext,
    ) -> dict[str, Any]:
        self._emit_progress("range_universe_candidate_load_start", {})
        template = HistoricalUniverseReconstructionJob(
            session=self._session,
            replay_date=replay_dates[0],
            run_timestamp=self._run_timestamp,
            allow_partial_delisted_source=self._allow_partial_delisted_source,
            persist_pre_replay_delisted_exclusions=False,
            compact_persisted_provenance=True,
        )
        candidates: dict[str, Any] = {}
        active_rows = template._load_active_current_candidates(candidates)
        delisted_rows = template._load_fmp_delisted_candidates(candidates)
        source_interval_count = sum(
            len(candidate.intervals) for candidate in candidates.values()
        )
        delisted_source = template._delisted_source_status(delisted_rows)
        self._emit_progress(
            "range_universe_candidate_load_finish",
            {
                "active_current_rows_seen": active_rows,
                "fmp_delisted_rows_seen": delisted_rows,
                "candidate_count": len(candidates),
                "source_interval_count": source_interval_count,
                "delisted_source_complete": delisted_source["complete"],
                "delisted_source_partial_reason": delisted_source["partial_reason"],
            },
        )

        row_mappings: list[dict[str, Any]] = []
        date_metrics: list[dict[str, Any]] = []
        candidate_values = sorted(
            candidates.values(),
            key=lambda candidate: candidate.normalized_symbol,
        )
        for replay_day in replay_dates:
            date_started = perf_counter()
            date_job = HistoricalUniverseReconstructionJob(
                session=self._session,
                replay_date=replay_day,
                run_timestamp=self._run_timestamp,
                allow_partial_delisted_source=self._allow_partial_delisted_source,
                persist_pre_replay_delisted_exclusions=False,
                compact_persisted_provenance=True,
            )
            lineage = record_data_lineage(
                self._session,
                provider="DERIVED",
                endpoint=DERIVED_ENDPOINT,
                asof_timestamp=self._run_timestamp,
                request_timestamp=self._run_timestamp,
                raw_payload={
                    "replay_date": replay_day.isoformat(),
                    "reconstruction_method": UNIVERSE_RECONSTRUCTION_METHOD,
                    "active_current_rows": active_rows,
                    "fmp_delisted_rows": delisted_rows,
                    "candidate_count": len(candidates),
                    "source_interval_count": source_interval_count,
                    "delisted_source_complete": delisted_source["complete"],
                    "delisted_source_partial_reason": delisted_source["partial_reason"],
                    "range_replay": True,
                },
                source_authority="alpha_engine",
                data_quality_flags={
                    "scratch_reconstruction": True,
                    "market_cap_price_liquidity_filters": "not_applied_not_pit_safe",
                    "delisted_source_complete": delisted_source["complete"],
                    "delisted_source_partial_reason": delisted_source["partial_reason"],
                    "allow_partial_delisted_source": (
                        self._allow_partial_delisted_source
                    ),
                    "historical_m4_range_replay": True,
                },
                job_run_id=ctx.job_run_id,
            )
            included_count = 0
            excluded_count = 0
            suppressed_count = 0
            rejection_reason_counts: dict[str, int] = {}
            self._emit_progress(
                "range_universe_date_evaluation_start",
                {
                    "replay_date": replay_day.isoformat(),
                    "candidate_count": len(candidate_values),
                },
            )
            for index, candidate in enumerate(candidate_values, start=1):
                evaluated = [
                    _evaluate_interval(interval, replay_day)
                    for interval in candidate.intervals
                ]
                row_values = date_job._row_values(
                    candidate,
                    evaluated,
                    lineage.data_lineage_id,
                    ctx.job_run_id,
                    delisted_source,
                )
                if row_values["inclusion_status"] == "included":
                    included_count += 1
                else:
                    excluded_count += 1
                    reason = row_values["rejection_reason"] or "unknown"
                    rejection_reason_counts[reason] = (
                        rejection_reason_counts.get(reason, 0) + 1
                    )
                if row_values["inclusion_status"] == "included":
                    row_mappings.append(row_values)
                else:
                    suppressed_count += 1
                if index == len(candidate_values) or index % self._progress_every == 0:
                    self._emit_progress(
                        "range_universe_date_evaluation_progress",
                        {
                            "replay_date": replay_day.isoformat(),
                            "rows_processed": index,
                            "rows_total": len(candidate_values),
                            "included_count": included_count,
                            "excluded_count": excluded_count,
                        },
                    )
            date_metrics.append(
                {
                    "replay_date": replay_day.isoformat(),
                    "included_count": included_count,
                    "excluded_count": excluded_count,
                    "suppressed_excluded_count": suppressed_count,
                    "rejection_reason_counts": rejection_reason_counts,
                    "elapsed_seconds": round(perf_counter() - date_started, 6),
                }
            )
            self._emit_progress(
                "range_universe_date_evaluation_finish",
                date_metrics[-1],
            )

        self._emit_progress(
            "range_universe_persistence_start",
            {
                "replay_date_count": len(replay_dates),
                "rows_total": len(row_mappings),
            },
        )
        persist_started = perf_counter()
        persistence = bulk_persist_historical_universe_reconstructions(
            self._session,
            row_mappings,
            progress_callback=lambda event, payload: self._emit_progress(
                f"range_universe_{event}",
                payload,
            ),
        )
        persistence["elapsed_seconds"] = round(perf_counter() - persist_started, 6)
        self._emit_progress("range_universe_persistence_finish", persistence)

        errors: list[dict[str, Any]] = []
        status = "finished"
        if delisted_source["partial"] and not self._allow_partial_delisted_source:
            status = "partial_failed"
            errors.append(
                {
                    "stage": "delisted_source_completeness",
                    "error_type": "delisted_source_partial",
                    "partial_reason": delisted_source["partial_reason"],
                    "job_run_id": delisted_source["job_run_id"],
                }
            )
        return {
            "status": status,
            "errors": errors,
            "metrics": {
                "active_current_rows_seen": active_rows,
                "fmp_delisted_rows_seen": delisted_rows,
                "candidate_count": len(candidates),
                "source_interval_count": source_interval_count,
                "delisted_source_complete": delisted_source["complete"],
                "delisted_source_partial_reason": delisted_source["partial_reason"],
                "date_metrics": date_metrics,
                "persistence": persistence,
            },
        }

    def _load_included_universe_rows(
        self,
        replay_dates: Sequence[date],
    ) -> dict[date, list[HistoricalUniverseReconstruction]]:
        rows = (
            self._session.query(HistoricalUniverseReconstruction)
            .filter(
                HistoricalUniverseReconstruction.replay_date.in_(list(replay_dates)),
                HistoricalUniverseReconstruction.inclusion_status == "included",
            )
            .order_by(
                HistoricalUniverseReconstruction.replay_date,
                HistoricalUniverseReconstruction.normalized_symbol,
            )
            .all()
        )
        by_date = {day: [] for day in replay_dates}
        for row in rows:
            by_date.setdefault(row.replay_date, []).append(row)
        return by_date

    def _ensure_replay_scans(
        self,
        replay_helper: HistoricalM4ReplayJob,
        included_rows_by_date: dict[date, list[HistoricalUniverseReconstruction]],
        ctx: JobContext,
        partial_by_date: dict[date, str | None],
    ) -> dict[date, list[Any]]:
        snapshots_by_date: dict[date, list[Any]] = {}
        for replay_day, rows in included_rows_by_date.items():
            cutoff_timestamp = us_equity_session_close_timestamp(replay_day)
            self._emit_progress(
                "range_replay_scan_snapshot_start",
                {
                    "replay_date": replay_day.isoformat(),
                    "snapshot_count": len(rows),
                    "bulk_persistence": True,
                },
            )
            started = perf_counter()
            universe_lineage = self._record_compact_replay_universe_lineage(
                rows,
                replay_day=replay_day,
                cutoff_timestamp=cutoff_timestamp,
                job_run_id=ctx.job_run_id,
                partial_reason=partial_by_date.get(replay_day),
            )
            snapshots = self._bulk_ensure_replay_scan_and_snapshots(
                rows,
                replay_day=replay_day,
                cutoff_timestamp=cutoff_timestamp,
                job_run_id=ctx.job_run_id,
                universe_lineage=universe_lineage,
                partial_reason=partial_by_date.get(replay_day),
            )
            snapshots_by_date[replay_day] = snapshots
            self._emit_progress(
                "range_replay_scan_snapshot_finish",
                {
                    "replay_date": replay_day.isoformat(),
                    "snapshot_count": len(snapshots),
                    "bulk_persistence": True,
                    "elapsed_seconds": round(perf_counter() - started, 6),
                },
            )
        return snapshots_by_date

    def _record_compact_replay_universe_lineage(
        self,
        rows: list[HistoricalUniverseReconstruction],
        *,
        replay_day: date,
        cutoff_timestamp: datetime,
        job_run_id: str,
        partial_reason: str | None,
    ) -> DataLineage:
        row_digest_payload = [
            {
                "historical_universe_reconstruction_id": (
                    row.historical_universe_reconstruction_id
                ),
                "ticker": row.normalized_symbol,
                "input_hash": row.input_hash,
                "output_hash": row.output_hash,
            }
            for row in rows
        ]
        payload = {
            "payload_policy": "compact_historical_m4_range_universe_digest_v1",
            "replay_date": replay_day.isoformat(),
            "source_universe_method": SOURCE_UNIVERSE_METHOD,
            "included_row_count": len(rows),
            "included_rows_digest": stable_hash(row_digest_payload),
        }
        raw_payload_hash = stable_hash(payload)
        lineage = DataLineage(
            data_lineage_id=str(uuid4()),
            provider="DERIVED",
            endpoint="historical_m4_replay_universe",
            request_timestamp=self._run_timestamp,
            asof_timestamp=cutoff_timestamp,
            raw_payload_hash=raw_payload_hash,
            raw_payload_json=json.dumps(payload, sort_keys=True, default=str),
            source_authority="alpha_engine",
            data_quality_flags=json.dumps(
                {
                    "historical_m4_replay": True,
                    "historical_m4_range_replay": True,
                    "reconstructed": True,
                    "source_universe_method": SOURCE_UNIVERSE_METHOD,
                    "partial_universe_reason": partial_reason,
                    "lineage_payload_policy": payload["payload_policy"],
                },
                sort_keys=True,
                default=str,
            ),
            job_run_id=job_run_id,
        )
        self._session.add(lineage)
        return lineage

    def _bulk_ensure_replay_scan_and_snapshots(
        self,
        rows: list[HistoricalUniverseReconstruction],
        *,
        replay_day: date,
        cutoff_timestamp: datetime,
        job_run_id: str,
        universe_lineage: DataLineage,
        partial_reason: str | None,
    ) -> list[dict[str, Any]]:
        scan_id = _replay_scan_id(replay_day)
        scan = self._session.get(UniverseScan, scan_id)
        scan_values = {
            "trading_date": replay_day.isoformat(),
            "job_run_id": job_run_id,
            "asof_timestamp": cutoff_timestamp,
            "provider": "HISTORICAL_REPLAY",
            "raw_count": len(rows),
            "deduped_count": len(rows),
            "included_count": len(rows),
            "excluded_count": 0,
            "source_lineage_hash": universe_lineage.raw_payload_hash,
            "run_status": "finished" if partial_reason is None else "partial_replay",
            "metric_json": json.dumps(
                {
                    "reconstructed": True,
                    "reconstruction_method": RECONSTRUCTION_METHOD,
                    "source_universe_method": SOURCE_UNIVERSE_METHOD,
                    "partial_universe_reason": partial_reason,
                    "range_level_cached_replay": True,
                },
                sort_keys=True,
            ),
        }
        if scan is None:
            self._session.add(UniverseScan(scan_id=scan_id, **scan_values))
        else:
            for key, value in scan_values.items():
                setattr(scan, key, value)

        canonical = self._session.get(CanonicalUniverseScan, replay_day.isoformat())
        canonical_values = {
            "scan_id": scan_id,
            "selected_job_run_id": job_run_id,
            "selected_at": cutoff_timestamp,
            "selection_reason": "historical_m4_replay_scratch_scan",
        }
        if canonical is None:
            self._session.add(
                CanonicalUniverseScan(
                    trading_date=replay_day.isoformat(),
                    **canonical_values,
                )
            )
        else:
            for key, value in canonical_values.items():
                setattr(canonical, key, value)
        self._session.flush()

        snapshot_mappings: list[dict[str, Any]] = []
        for row in rows:
            snapshot_mappings.append(
                {
                    "universe_snapshot_id": _replay_snapshot_id(
                        replay_day,
                        row.normalized_symbol,
                    ),
                    "job_run_id": job_run_id,
                    "scan_id": scan_id,
                    "ticker": row.normalized_symbol,
                    "asof_timestamp": cutoff_timestamp,
                    "source_provider": "HISTORICAL_REPLAY",
                    "market_cap": None,
                    "price": None,
                    "country": "US",
                    "security_type": "common_stock",
                    "primary_exchange": row.exchange,
                    "operating_universe_inclusion": True,
                    "exclusion_reason": None,
                    "dataset_version": SOURCE_UNIVERSE_METHOD,
                    "schema_hash": RECONSTRUCTION_METHOD,
                    "source_lineage_hash": row.output_hash
                    or universe_lineage.raw_payload_hash,
                }
            )
        self._bulk_upsert_replay_universe_snapshots(snapshot_mappings)
        return snapshot_mappings

    def _bulk_upsert_replay_universe_snapshots(
        self,
        snapshot_mappings: list[dict[str, Any]],
    ) -> None:
        if not snapshot_mappings:
            return
        if self._session.bind is not None and self._session.bind.dialect.name == "postgresql":
            _postgres_stage_merge_replay_snapshots(self._session, snapshot_mappings)
            return

        existing_ids = {
            row.universe_snapshot_id
            for row in (
                self._session.query(UniverseSnapshot.universe_snapshot_id)
                .filter(
                    UniverseSnapshot.universe_snapshot_id.in_(
                        [mapping["universe_snapshot_id"] for mapping in snapshot_mappings]
                    )
                )
                .all()
            )
        }
        inserts = [
            mapping
            for mapping in snapshot_mappings
            if mapping["universe_snapshot_id"] not in existing_ids
        ]
        updates = [
            mapping
            for mapping in snapshot_mappings
            if mapping["universe_snapshot_id"] in existing_ids
        ]
        if inserts:
            self._session.bulk_insert_mappings(UniverseSnapshot, inserts)
        if updates:
            self._session.bulk_update_mappings(UniverseSnapshot, updates)
        self._session.flush()

    def _fetch_range_bars(
        self,
        tickers: Sequence[str],
        replay_dates: Sequence[date],
        *,
        replay_helper: HistoricalM4ReplayJob,
        job_run_id: str,
    ) -> dict[str, Any]:
        from_date = min(replay_dates) - timedelta(days=self._lookback_calendar_days)
        to_date = max(replay_dates)
        asof = us_equity_session_close_timestamp(to_date)
        bars_by_ticker: dict[str, list[Any]] = {}
        fetch_by_ticker: dict[str, Any] = {}
        fetch_errors: list[dict[str, Any]] = []
        total_bars = 0
        polygon_fallback_count = 0
        missing_price_evidence_count = 0
        self._emit_progress(
            "range_ticker_fetch_start",
            {
                "ticker_count": len(tickers),
                "from_date": from_date.isoformat(),
                "to_date": to_date.isoformat(),
            },
        )
        for index, ticker in enumerate(tickers, start=1):
            if index == 1 or index % self._progress_every == 0 or index == len(tickers):
                self._emit_progress(
                    "range_ticker_fetch_progress",
                    {
                        "ticker": ticker,
                        "started": index,
                        "finished": len(bars_by_ticker),
                        "errors": len(fetch_errors),
                        "ticker_total": len(tickers),
                    },
                )
            fmp_fetch = replay_helper._fetch_fmp_bars_for_ticker(
                ticker=ticker,
                from_date=from_date,
                to_date=to_date,
                asof=asof,
                job_run_id=job_run_id,
                range_replay=True,
                required_evidence_dates=replay_dates,
            )
            providers: dict[str, Any] = {"FMP": fmp_fetch}
            source_attempts = list(fmp_fetch.source_attempts)
            if fmp_fetch.missing_evidence_dates:
                polygon_fetch = replay_helper._fetch_polygon_bars_for_ticker(
                    ticker=ticker,
                    from_date=from_date,
                    to_date=to_date,
                    asof=asof,
                    job_run_id=job_run_id,
                    range_replay=True,
                    required_evidence_dates=replay_dates,
                    source_attempts=source_attempts,
                )
                providers["Polygon"] = polygon_fetch
                source_attempts = list(polygon_fetch.source_attempts)

            fetch_entry = {
                "providers": providers,
                "source_attempts": source_attempts,
            }
            fetch_by_ticker[ticker.upper()] = fetch_entry

            covered_dates = [
                replay_day
                for replay_day in replay_dates
                if _select_range_fetch_for_date(fetch_entry, replay_day) is not None
            ]
            if not covered_dates:
                missing_price_evidence_count += 1
            if any(
                _select_range_fetch_for_date(fetch_entry, replay_day) is not None
                and bool(
                    getattr(
                        _select_range_fetch_for_date(fetch_entry, replay_day),
                        "fallback_used",
                        False,
                    )
                )
                for replay_day in replay_dates
            ):
                polygon_fallback_count += 1
            if not covered_dates:
                fetch_error = _range_missing_price_evidence_error(
                    ticker=ticker,
                    source_attempts=source_attempts,
                )
                fetch_errors.append(fetch_error)
                continue
            selected_bars = {
                (getattr(fetch, "provider", None), bar.date): bar
                for fetch in providers.values()
                for bar in getattr(fetch, "bars", [])
            }
            bars_by_ticker[ticker.upper()] = sorted(
                selected_bars.values(),
                key=lambda bar: str(bar.date),
            )
            total_bars += sum(len(getattr(fetch, "bars", []) or []) for fetch in providers.values())
        self._emit_progress(
            "range_ticker_fetch_finish",
            {
                "ticker_count": len(tickers),
                "tickers_with_bars": len(bars_by_ticker),
                "tickers_missing_bars": len(tickers) - len(bars_by_ticker),
                "non_evaluable_ticker_count": missing_price_evidence_count,
                "fetched_bar_count": total_bars,
                "fetch_error_count": len(fetch_errors),
                "missing_price_evidence_count": missing_price_evidence_count,
                "polygon_fallback_count": polygon_fallback_count,
                "cache_hits": int(getattr(self._fmp, "cache_hits", 0) or 0),
                "cache_misses": int(getattr(self._fmp, "cache_misses", 0) or 0),
            },
        )
        return {
            "bars_by_ticker": bars_by_ticker,
            "fetch_by_ticker": fetch_by_ticker,
            "errors": fetch_errors,
            "metrics": {
                "from_date": from_date.isoformat(),
                "to_date": to_date.isoformat(),
                "requested_ticker_count": len(tickers),
                "tickers_with_bars": len(bars_by_ticker),
                "tickers_missing_bars": len(tickers) - len(bars_by_ticker),
                "non_evaluable_ticker_count": missing_price_evidence_count,
                "fetched_bar_count": total_bars,
                "fetch_error_count": len(fetch_errors),
                "missing_price_evidence_count": missing_price_evidence_count,
                "polygon_fallback_count": polygon_fallback_count,
                "cache_hits": int(getattr(self._fmp, "cache_hits", 0) or 0),
                "cache_misses": int(getattr(self._fmp, "cache_misses", 0) or 0),
                "errors": fetch_errors[:50],
            },
        }

    def _run_one_replay_date(
        self,
        replay_day: date,
        ctx: JobContext,
        replay_helper: HistoricalM4ReplayJob,
        included_rows: list[HistoricalUniverseReconstruction],
        snapshots: list[Any],
        range_bars_by_ticker: dict[str, list[Any]],
        range_fetch_by_ticker: dict[str, Any],
        partial_reason: str | None,
    ) -> JobResult:
        replay_date_str = replay_day.isoformat()
        cutoff_timestamp = us_equity_session_close_timestamp(replay_day)
        next_execution = next_us_equity_session(replay_day + timedelta(days=1))
        from_date = replay_day - timedelta(days=self._lookback_calendar_days)
        stage_timings: dict[str, float] = {}
        fetch_errors: list[dict[str, Any]] = []
        daily_bars: dict[str, list[Any]] = {}
        bar_lineage_by_ticker: dict[str, DataLineage] = {}
        rows_by_ticker = {
            str(row.normalized_symbol).upper(): row
            for row in included_rows
            if row.normalized_symbol
        }
        polygon_fallback_count = 0
        lineage_started = perf_counter()
        self._emit_progress(
            "range_date_lineage_stage_start",
            {
                "replay_date": replay_date_str,
                "ticker_count": len(snapshots),
            },
        )
        for snapshot in snapshots:
            ticker = _snap_attr(snapshot, "ticker").upper()
            fetch_entry = range_fetch_by_ticker.get(ticker)
            fetch = _select_range_fetch_for_date(fetch_entry, replay_day)
            if fetch is None:
                source_attempts = _range_fetch_source_attempts(fetch_entry)
                fetch_errors.append(
                    {
                        "stage": "cached_historical_price",
                        "ticker": ticker,
                        "error_type": "missing_price_evidence",
                        "message": "no cached evidence-session bar for replay date",
                        "retryable": False,
                        "source_attempts": source_attempts,
                    }
                )
                continue
            slice_bars = _slice_fetch_bars(
                fetch,
                from_date=from_date,
                replay_day=replay_day,
            )
            if not _slice_has_evidence_session_bar(slice_bars, replay_day):
                source_attempts = _range_fetch_source_attempts(fetch_entry)
                fetch_errors.append(
                    {
                        "stage": "cached_historical_price",
                        "ticker": ticker,
                        "error_type": "missing_price_evidence",
                        "message": "cached bars missing replay-date evidence-session bar",
                        "retryable": False,
                        "source_attempts": source_attempts,
                    }
                )
                continue
            provider = getattr(fetch, "provider", None) or "FMP"
            endpoint = getattr(fetch, "endpoint", None) or HISTORICAL_PRICE_FULL_ENDPOINT
            price_basis = getattr(fetch, "price_basis", None) or FMP_PRICE_BASIS
            fallback_used = bool(getattr(fetch, "fallback_used", False))
            source_attempts = list(getattr(fetch, "source_attempts", []) or [])
            fetch_metadata = _bar_metadata_from_lineage(getattr(fetch, "lineage", None))
            if fallback_used:
                polygon_fallback_count += 1
            payload = _daily_bar_lineage_payload(
                slice_bars,
                ticker=ticker,
                from_date=from_date,
                to_date=replay_day,
                endpoint=endpoint,
                price_basis=price_basis,
            )
            lineage = replay_helper._build_bar_lineage(
                provider=provider,
                endpoint=endpoint,
                asof_timestamp=cutoff_timestamp,
                request_timestamp=self._run_timestamp,
                raw_payload=payload,
                raw_payload_hash=stable_hash(payload),
                freshness_seconds=None,
                source_authority=provider,
                data_quality_flags={
                    "historical_m4_replay": True,
                    "historical_m4_range_replay": True,
                    "bar_provider": provider,
                    "bar_endpoint": endpoint,
                    "bar_provider_policy": BAR_PROVIDER_POLICY,
                    "price_basis": price_basis,
                    "adjusted": fetch_metadata.get("adjusted"),
                    "requested_adjusted": fetch_metadata.get("requested_adjusted"),
                    "adjustment_basis": fetch_metadata.get("adjustment_basis"),
                    "fallback_used": fallback_used,
                    "source_attempts": source_attempts,
                    "source_attempt_count": len(source_attempts),
                    "range_fetch_contains_future_bars_for_earlier_dates": True,
                    "row_input_window_end": replay_date_str,
                },
                job_run_id=ctx.job_run_id,
            )
            bar_lineage_by_ticker[ticker] = lineage
            daily_bars[ticker] = [
                DailyBar(
                    date=bar.date,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    split_adjusted_close=bar.split_adjusted_close,
                    adj_close=bar.adj_close,
                    source_timestamp=cutoff_timestamp,
                    source_provider=provider,
                    lineage_id=lineage.data_lineage_id,
                    lineage_hash=lineage.raw_payload_hash,
                )
                for bar in slice_bars
            ]
        self._session.flush()
        stage_timings["bar_lineage_build_seconds"] = round(
            perf_counter() - lineage_started,
            6,
        )
        self._emit_progress(
            "range_date_lineage_stage_finish",
            {
                "replay_date": replay_date_str,
                "lineage_count": len(bar_lineage_by_ticker),
                "elapsed_seconds": stage_timings["bar_lineage_build_seconds"],
            },
        )

        assembly_snapshots = [
            snapshot
            for snapshot in snapshots
            if _snap_attr(snapshot, "ticker").upper() in daily_bars
        ]
        self._emit_progress(
            "range_date_assembly_start",
            {
                "replay_date": replay_date_str,
                "snapshot_count": len(assembly_snapshots),
                "tickers_with_bars": len(daily_bars),
            },
        )
        started = perf_counter()
        assembly = assemble_m4_daily(
            snapshots=assembly_snapshots,
            daily_bars=daily_bars,
            cutoff_timestamp=cutoff_timestamp,
            universe_cutoff_timestamp=cutoff_timestamp,
            decision_date=replay_date_str,
            evidence_session_date=replay_date_str,
            next_execution_session=next_execution.isoformat(),
            source_provider="FMP",
        )
        stage_timings["assembly_seconds"] = round(perf_counter() - started, 6)
        self._emit_progress(
            "range_date_assembly_finish",
            {
                "replay_date": replay_date_str,
                "assembled_count": assembly.assembled_count,
                "rejected_count": assembly.rejected_count,
                "insufficient_count": assembly.insufficient_count,
                "elapsed_seconds": stage_timings["assembly_seconds"],
            },
        )

        self._emit_progress(
            "range_date_detector_start",
            {
                "replay_date": replay_date_str,
                "assembled_count": assembly.assembled_count,
                "bulk_persistence": True,
            },
        )
        started = perf_counter()
        bulk_result = self._run_bulk_m4_detector_persistence(
            inputs=assembly.inputs,
            replay_day=replay_day,
            cutoff_timestamp=cutoff_timestamp,
            scan_id=_replay_scan_id(replay_day),
            ctx=ctx,
            bar_lineage_by_ticker=bar_lineage_by_ticker,
            partial_reason=partial_reason,
        )
        stage_timings["detector_seconds"] = round(perf_counter() - started, 6)
        self._emit_progress(
            "range_date_detector_finish",
            {
                "replay_date": replay_date_str,
                "status": bulk_result.status,
                "elapsed_seconds": stage_timings["detector_seconds"],
                "bulk_persistence": True,
            },
        )

        detector_diag = _m4_detector_diagnostic(bulk_result.metrics)
        fired_count = int(bulk_result.metrics.get("signals_inserted") or 0)
        reused_signal_count = int(bulk_result.metrics.get("signals_reused") or 0)
        no_fire_count = int(bulk_result.metrics.get("no_fire_count") or 0)
        stamped_feature_count = int(bulk_result.metrics.get("feature_snapshot_count") or 0)
        stamped_fired_feature_count = int(
            bulk_result.metrics.get("fired_feature_snapshot_count") or 0
        )
        missing_price_evidence_count = sum(
            1
            for error in fetch_errors
            if error.get("error_type") == "missing_price_evidence"
        )
        hard_errors = list(bulk_result.errors or [])
        has_hard_failure = bulk_result.status != "finished" or bool(hard_errors)
        completion_classification = _date_completion_classification(
            missing_price_evidence_count=missing_price_evidence_count,
            m4_evaluable_count=len(daily_bars),
            has_hard_failure=has_hard_failure,
            detector_status=bulk_result.status,
        )
        coverage_status = _coverage_status(
            missing_price_evidence_count=missing_price_evidence_count,
            m4_evaluable_count=len(daily_bars),
            hard_error_count=1 if has_hard_failure else 0,
        )
        non_evaluable_samples = _non_evaluable_price_evidence_samples(
            self._session,
            rows_by_ticker=rows_by_ticker,
            fetch_errors=fetch_errors,
            replay_day=replay_day,
        )
        metrics = {
            "replay_date": replay_date_str,
            "evidence_session_date": replay_date_str,
            "next_execution_session": next_execution.isoformat(),
            "historical_universe_included_count": len(snapshots),
            "universe_included_count": len(snapshots),
            "partial_universe_reason": partial_reason,
            "m4_evaluable_count": len(daily_bars),
            "m4_non_evaluable_count": missing_price_evidence_count,
            "tickers_with_bars": len(daily_bars),
            "tickers_missing_bars": len(snapshots) - len(daily_bars),
            "non_evaluable_ticker_count": missing_price_evidence_count,
            "fetch_error_count": len(fetch_errors),
            "fetch_errors": fetch_errors[:50],
            "missing_price_evidence_count": missing_price_evidence_count,
            "coverage_status": coverage_status,
            "completion_classification": completion_classification,
            "non_evaluable_price_evidence_samples": non_evaluable_samples,
            "polygon_fallback_count": polygon_fallback_count,
            "assembled_count": assembly.assembled_count,
            "assembly": _assembly_metrics(assembly),
            "fired_m4_signal_count": fired_count,
            "reused_existing_signal_count": reused_signal_count,
            "duplicate_suppressed_count": detector_diag.get("duplicate_suppressed_count", 0),
            "rejected_or_no_fire_count": (
                assembly.rejected_count
                + assembly.insufficient_count
                + no_fire_count
                + detector_diag.get("identity_refused_count", 0)
                + detector_diag.get("lookahead_failure_count", 0)
                + detector_diag.get("error_count", 0)
            ),
            "rows_inserted": fired_count,
            "rows_reused": reused_signal_count,
            "stamped_feature_count": stamped_feature_count,
            "stamped_fired_feature_count": stamped_fired_feature_count,
            "stamped_no_fire_feature_count": max(
                0,
                stamped_feature_count - stamped_fired_feature_count,
            ),
            "sample_fired_tickers": _signal_tickers(
                self._session,
                bulk_result.metrics.get("inserted_signal_ids", [])[:20],
            )[:20],
            "scan_id": _replay_scan_id(replay_day),
            "orchestration": bulk_result.metrics,
            "stage_timing_seconds": stage_timings,
        }
        errors = hard_errors
        status = bulk_result.status if has_hard_failure else "finished"
        return JobResult(status=status, metrics=metrics, errors=errors)

    def _run_bulk_m4_detector_persistence(
        self,
        *,
        inputs: Sequence[Any],
        replay_day: date,
        cutoff_timestamp: datetime,
        scan_id: str,
        ctx: JobContext,
        bar_lineage_by_ticker: dict[str, DataLineage],
        partial_reason: str | None,
    ) -> JobResult:
        replay_date_str = replay_day.isoformat()
        detector = M4Detector()
        detector_version = getattr(detector, "version", None)
        diag = DetectorDiagnostics(
            detector_id=detector.pattern_id,
            detector_version=detector_version or "missing",
        )
        if not detector_version:
            diag.detector_status = "failed"
            diag.error_count = 1
            diag.errors.append(
                {
                    "detector_id": detector.pattern_id,
                    "error": "missing_explicit_detector_version",
                }
            )
            return JobResult(status="failed", metrics=_bulk_metrics(diag), errors=diag.errors)

        valid_snapshot_ids = {
            str(inp.universe_snapshot_id)
            for inp in inputs
            if inp.universe_snapshot_id is not None
        }
        lineage_ids_by_hash = self._lineage_ids_by_hash(inputs)
        records: list[_BulkDetectionRecord] = []
        persistence_started = perf_counter()
        for inp in inputs:
            diag.evaluated_count += 1
            if diag.evaluated_count == 1 or diag.evaluated_count % self._progress_every == 0:
                self._emit_progress(
                    "range_date_bulk_detector_progress",
                    {
                        "replay_date": replay_date_str,
                        "evaluated_count": diag.evaluated_count,
                        "input_count": len(inputs),
                        "feature_snapshot_count": len(records),
                        "fired_count": diag.fired_count,
                        "skipped_count": diag.skipped_count,
                        "error_count": diag.error_count,
                    },
                )
            if inp.universe_snapshot_id not in valid_snapshot_ids:
                diag.identity_refused_count += 1
                diag.skipped_count += 1
                diag.errors.append(
                    {
                        "detector_id": detector.pattern_id,
                        "ticker": inp.ticker,
                        "error": "universe_snapshot_not_in_canonical_scan",
                        "universe_snapshot_id": inp.universe_snapshot_id,
                        "scan_id": scan_id,
                    }
                )
                continue
            try:
                max_asof_timestamp, max_asof_label = _input_asof_ceiling(
                    inp,
                    cutoff_timestamp,
                    replay_date_str,
                )
                pit_passed, pit_reason = check_lookahead_guard(
                    inp,
                    replay_date_str,
                    max_asof_timestamp=max_asof_timestamp,
                    max_asof_label=max_asof_label,
                )
            except Exception as exc:
                diag.lookahead_failure_count += 1
                diag.skipped_count += 1
                diag.errors.append(
                    {
                        "detector_id": detector.pattern_id,
                        "ticker": inp.ticker,
                        "error": f"invalid_evidence_session_date:{exc}",
                    }
                )
                continue
            if not pit_passed:
                diag.lookahead_failure_count += 1
                diag.skipped_count += 1
                diag.errors.append(
                    {
                        "detector_id": detector.pattern_id,
                        "ticker": inp.ticker,
                        "error": f"lookahead_guard_failed:{pit_reason}",
                    }
                )
                continue

            try:
                result = detector.detect(inp)
            except Exception as exc:
                diag.error_count += 1
                diag.errors.append(
                    {
                        "detector_id": detector.pattern_id,
                        "ticker": inp.ticker,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                continue

            for lineage_hash in inp.lineage_hashes:
                lineage_hash_str = str(lineage_hash or "").strip()
                if lineage_hash_str:
                    diag.input_lineage_hashes.append(lineage_hash_str)

            if result.features is None:
                if result.has_signal:
                    diag.identity_refused_count += 1
                    diag.errors.append(
                        {
                            "detector_id": detector.pattern_id,
                            "ticker": inp.ticker,
                            "error": "missing_features",
                        }
                    )
                else:
                    diag.skipped_count += 1
                continue

            result_guard_passed, result_guard_reason = _result_guard_passed(result)
            if not result_guard_passed:
                diag.lookahead_failure_count += 1
                if result.has_signal:
                    diag.identity_refused_count += 1
                diag.skipped_count += 1
                diag.errors.append(
                    {
                        "detector_id": detector.pattern_id,
                        "ticker": inp.ticker,
                        "error": result_guard_reason,
                    }
                )
                continue

            feature_payload = dict(result.features.features)
            signal_identity_hash: str | None = None
            signal_identity_components: dict[str, Any] | None = None
            if result.has_signal:
                detector_identity_hash = feature_payload.get("signal_identity_hash")
                detector_identity_components = feature_payload.get(
                    "signal_identity_components"
                )
                if not detector_identity_hash:
                    diag.identity_refused_count += 1
                    diag.errors.append(
                        {
                            "detector_id": detector.pattern_id,
                            "ticker": inp.ticker,
                            "error": "missing_detector_signal_identity_hash",
                        }
                    )
                    continue
                if len(result.signals) != 1:
                    diag.identity_refused_count += 1
                    diag.errors.append(
                        {
                            "detector_id": detector.pattern_id,
                            "ticker": inp.ticker,
                            "error": "orchestration_requires_exactly_one_signal_per_result",
                        }
                    )
                    continue
                signal = result.signals[0]
                signal_identity_hash = compute_signal_identity_hash(
                    detector_id=detector.pattern_id,
                    detector_version=detector_version,
                    ticker=inp.ticker,
                    trading_date=replay_date_str,
                    direction=signal.direction,
                    detector_signal_identity_hash=detector_identity_hash,
                    detector_signal_identity_components=detector_identity_components,
                    route_class=signal.route_class or detector.route_class,
                    signal_horizon=signal.signal_horizon,
                    signal_event_sequence=1,
                )
                if not signal_identity_hash or not scan_id or not inp.universe_snapshot_id:
                    diag.identity_refused_count += 1
                    continue
                signal_identity_components = {
                    "detector_id": detector.pattern_id,
                    "detector_version": detector_version,
                    "ticker": inp.ticker,
                    "trading_date": replay_date_str,
                    "direction": signal.direction,
                    "detector_signal_identity_hash": detector_identity_hash,
                    "detector_signal_identity_components": detector_identity_components,
                    "route_class": signal.route_class or detector.route_class,
                    "signal_horizon": signal.signal_horizon,
                    "signal_event_sequence": 1,
                }
                feature_payload["detector_signal_identity_hash"] = detector_identity_hash
                feature_payload["detector_signal_identity_components"] = (
                    detector_identity_components
                )
                feature_payload["signal_identity_hash"] = signal_identity_hash
                feature_payload["signal_identity_components"] = signal_identity_components

            lineage_ids = _resolved_input_lineage_ids_from_cache(inp, lineage_ids_by_hash)
            feature_hash = stable_hash(feature_payload)
            stamped_payload = _stamped_replay_feature_payload(
                feature_payload,
                replay_day=replay_day,
                partial_reason=partial_reason,
                bar_lineage=bar_lineage_by_ticker.get(inp.ticker.upper()),
            )
            records.append(
                _BulkDetectionRecord(
                    result=result,
                    ticker=inp.ticker,
                    feature_payload=stamped_payload,
                    feature_hash=feature_hash,
                    feature_json=json.dumps(stamped_payload, sort_keys=True, default=str),
                    output_hash=stable_hash(stamped_payload),
                    data_lineage_ids=lineage_ids,
                    universe_snapshot_id=inp.universe_snapshot_id,
                    next_execution_session=(
                        str(stamped_payload.get("next_execution_session"))
                        if stamped_payload.get("next_execution_session") is not None
                        else None
                    ),
                    signal_identity_hash=signal_identity_hash,
                    signal_identity_components=signal_identity_components,
                )
            )
            if result.has_signal:
                diag.fired_count += 1
            else:
                diag.skipped_count += 1

        self._emit_progress(
            "range_date_bulk_detector_finish",
            {
                "replay_date": replay_date_str,
                "evaluated_count": diag.evaluated_count,
                "feature_record_count": len(records),
                "fired_count": diag.fired_count,
                "skipped_count": diag.skipped_count,
                "error_count": diag.error_count,
                "elapsed_seconds": round(perf_counter() - persistence_started, 6),
            },
        )
        persistence = self._bulk_persist_detection_records(
            records,
            detector=detector,
            detector_version=detector_version,
            replay_date=replay_date_str,
            scan_id=scan_id,
            job_run_id=ctx.job_run_id,
            code_commit_sha=ctx.app_commit_sha,
        )
        if persistence["signals_reused"]:
            diag.duplicate_suppressed_count += persistence["signals_reused"]
            diag.fired_count = max(0, diag.fired_count - persistence["signals_reused"])
        diag.feature_snapshot_count = persistence["feature_snapshot_count"]
        if diag.error_count + diag.identity_refused_count + diag.lookahead_failure_count:
            diag.detector_status = "partial_failed" if diag.fired_count > 0 else "failed"
        metrics = _bulk_metrics(diag)
        metrics.update(persistence)
        metrics["detector_seconds"] = round(perf_counter() - persistence_started, 6)
        status = "partial_failed" if diag.detector_status == "partial_failed" else diag.detector_status
        return JobResult(status=status, metrics=metrics, errors=diag.errors)

    def _lineage_ids_by_hash(self, inputs: Sequence[Any]) -> dict[str, list[str]]:
        lineage_hashes = sorted(
            {
                str(value).strip()
                for inp in inputs
                for value in getattr(inp, "lineage_hashes", [])
                if str(value or "").strip()
            }
        )
        if not lineage_hashes:
            return {}
        rows = (
            self._session.query(DataLineage.raw_payload_hash, DataLineage.data_lineage_id)
            .filter(DataLineage.raw_payload_hash.in_(lineage_hashes))
            .all()
        )
        by_hash: dict[str, list[str]] = {}
        for raw_payload_hash, data_lineage_id in rows:
            by_hash.setdefault(str(raw_payload_hash), []).append(str(data_lineage_id))
        return by_hash

    def _bulk_persist_detection_records(
        self,
        records: Sequence[_BulkDetectionRecord],
        *,
        detector: M4Detector,
        detector_version: str,
        replay_date: str,
        scan_id: str,
        job_run_id: str,
        code_commit_sha: str | None,
    ) -> dict[str, Any]:
        started = perf_counter()
        self._emit_progress(
            "range_date_feature_snapshot_stage_start",
            {
                "replay_date": replay_date,
                "record_count": len(records),
            },
        )
        existing_features = _existing_feature_snapshots(self._session, records)
        existing_signals = _existing_signals(self._session, records)

        feature_ids_by_key: dict[tuple[str, str, datetime, str], str] = {
            key: row.feature_snapshot_id for key, row in existing_features.items()
        }
        feature_insert_mappings: list[dict[str, Any]] = []
        feature_records_by_id: dict[str, _BulkDetectionRecord] = {}
        fired_feature_ids: set[str] = set()
        no_fire_feature_ids: set[str] = set()
        signals_to_insert: list[tuple[_BulkDetectionRecord, Any]] = []
        signal_insert_mappings: list[dict[str, Any]] = []
        inserted_signal_ids: list[str] = []
        reused_signal_ids: list[str] = []
        signal_next_execution_updates = 0

        for record in records:
            signal = record.result.signals[0] if record.signal_identity_hash else None
            signal_key = (
                record.result.pattern_id,
                record.ticker,
                record.signal_identity_hash,
            )
            existing_signal = (
                existing_signals.get(signal_key)
                if record.signal_identity_hash is not None
                else None
            )
            if existing_signal is not None:
                reused_signal_ids.append(existing_signal.signal_id)
                if record.next_execution_session and existing_signal.next_execution_session is None:
                    existing_signal.next_execution_session = record.next_execution_session
                    signal_next_execution_updates += 1
                if existing_signal.feature_snapshot_id:
                    fired_feature_ids.add(existing_signal.feature_snapshot_id)
                continue

            feature_key = _feature_key(record)
            feature_id = feature_ids_by_key.get(feature_key)
            if feature_id is None:
                feature_id = str(uuid4())
                feature_ids_by_key[feature_key] = feature_id
                feature_records_by_id[feature_id] = record
                feature_insert_mappings.append(
                    _feature_snapshot_mapping(
                        record,
                        feature_snapshot_id=feature_id,
                        job_run_id=job_run_id,
                        code_commit_sha=code_commit_sha,
                    )
                )
            if signal is None:
                no_fire_feature_ids.add(feature_id)
                continue
            fired_feature_ids.add(feature_id)
            signals_to_insert.append((record, signal))
            signal_id = str(uuid4())
            inserted_signal_ids.append(signal_id)
            signal_insert_mappings.append(
                _signal_mapping(
                    record,
                    signal,
                    detector=detector,
                    detector_version=detector_version,
                    signal_id=signal_id,
                    feature_snapshot_id=feature_id,
                    job_run_id=job_run_id,
                    replay_date=replay_date,
                    scan_id=scan_id,
                )
            )

        if feature_insert_mappings:
            self._session.execute(FeatureSnapshot.__table__.insert(), feature_insert_mappings)
        self._session.flush()
        feature_stage_seconds = round(perf_counter() - started, 6)
        self._emit_progress(
            "range_date_feature_snapshot_stage_finish",
            {
                "replay_date": replay_date,
                "features_inserted": len(feature_insert_mappings),
                "features_reused": len(records) - len(feature_insert_mappings),
                "elapsed_seconds": feature_stage_seconds,
            },
        )

        self._emit_progress(
            "range_date_signal_stage_start",
            {
                "replay_date": replay_date,
                "signals_to_insert": len(signal_insert_mappings),
                "signals_reused": len(reused_signal_ids),
            },
        )
        signal_stage_started = perf_counter()
        if signal_insert_mappings:
            self._session.execute(SignalRegistry.__table__.insert(), signal_insert_mappings)
        self._session.flush()
        signal_stage_seconds = round(perf_counter() - signal_stage_started, 6)
        self._emit_progress(
            "range_date_signal_stage_finish",
            {
                "replay_date": replay_date,
                "signals_inserted": len(signal_insert_mappings),
                "signals_reused": len(reused_signal_ids),
                "next_execution_updates": signal_next_execution_updates,
                "elapsed_seconds": signal_stage_seconds,
            },
        )
        self._emit_progress(
            "range_date_link_stage_start",
            {
                "replay_date": replay_date,
                "link_table": "signal_registry.feature_snapshot_id",
            },
        )
        self._emit_progress(
            "range_date_link_stage_finish",
            {
                "replay_date": replay_date,
                "link_rows_inserted": len(signal_insert_mappings),
                "elapsed_seconds": 0.0,
            },
        )
        return {
            "features_inserted": len(feature_insert_mappings),
            "features_reused": len(records) - len(feature_insert_mappings),
            "feature_snapshot_count": len(records),
            "fired_feature_snapshot_count": len(fired_feature_ids),
            "no_fire_feature_snapshot_count": len(no_fire_feature_ids),
            "signals_inserted": len(signal_insert_mappings),
            "signals_reused": len(reused_signal_ids),
            "inserted_signal_ids": inserted_signal_ids,
            "reused_signal_ids": reused_signal_ids,
            "signal_next_execution_updates": signal_next_execution_updates,
            "persistence_timing_seconds": {
                "feature_snapshot_stage_seconds": feature_stage_seconds,
                "signal_stage_seconds": signal_stage_seconds,
                "total_persistence_seconds": round(perf_counter() - started, 6),
            },
        }

    def _emit_progress(self, event: str, payload: dict[str, Any]) -> None:
        event_payload = {
            "event": event,
            "elapsed_seconds": round(perf_counter() - self._started_perf, 6),
            **payload,
        }
        self._progress_events.append(event_payload)
        if len(self._progress_events) > 500:
            del self._progress_events[:-500]
        if self._progress_callback is not None:
            try:
                self._progress_callback(event, event_payload)
            except Exception:
                pass


def run_historical_m4_range_replay(
    *,
    session: Session,
    fmp_adapter: Any,
    start_date: date,
    end_date: date,
    polygon_adapter: Any | None = None,
    run_timestamp: datetime | None = None,
    allow_partial_delisted_source: bool = False,
    allow_partial_universe: bool = False,
    lookback_calendar_days: int = LOOKBACK_CALENDAR_DAYS,
    skip_completed_dates: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 25,
) -> HistoricalM4RangeReplayResult:
    replay_dates = _trading_dates(start_date, end_date)
    job = HistoricalM4RangeReplayJob(
        session=session,
        fmp_adapter=fmp_adapter,
        polygon_adapter=polygon_adapter,
        replay_dates=replay_dates,
        run_timestamp=run_timestamp,
        allow_partial_delisted_source=allow_partial_delisted_source,
        allow_partial_universe=allow_partial_universe,
        lookback_calendar_days=lookback_calendar_days,
        skip_completed_dates=skip_completed_dates,
        progress_callback=progress_callback,
        progress_every=progress_every,
    )
    result = run_job(
        session,
        job,
        params={
            "source": JOB_NAME,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "allow_partial_delisted_source": allow_partial_delisted_source,
            "allow_partial_universe": allow_partial_universe,
            "lookback_calendar_days": lookback_calendar_days,
            "skip_completed_dates": skip_completed_dates,
            "polygon_fallback_configured": polygon_adapter is not None,
        },
    )
    return HistoricalM4RangeReplayResult(
        status=result.status,
        metrics=result.metrics or {},
        errors=result.errors or [],
    )


def _validate_pattern_ids(pattern_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(pattern.strip().upper() for pattern in pattern_ids))
    if not normalized:
        raise ValueError("At least one --pattern-id is required")
    unsupported = [pattern for pattern in normalized if pattern not in SUPPORTED_PATTERN_IDS]
    if unsupported:
        raise ValueError(
            "Unsupported historical range replay pattern(s): "
            f"{', '.join(unsupported)}. Only audited M4 replay is implemented."
        )
    return normalized


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


def _bar_date(bar: Any) -> date:
    value = getattr(bar, "date")
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _range_fetch_providers(fetch_entry: Any) -> dict[str, Any]:
    if isinstance(fetch_entry, dict):
        providers = fetch_entry.get("providers")
        if isinstance(providers, dict):
            return providers
    if fetch_entry is None:
        return {}
    provider = getattr(fetch_entry, "provider", None) or "FMP"
    return {str(provider): fetch_entry}


def _range_fetch_source_attempts(fetch_entry: Any) -> list[dict[str, Any]]:
    if isinstance(fetch_entry, dict):
        attempts = fetch_entry.get("source_attempts")
        if isinstance(attempts, list):
            return list(attempts)
    attempts = getattr(fetch_entry, "source_attempts", None)
    return list(attempts or [])


def _slice_fetch_bars(
    fetch: Any,
    *,
    from_date: date,
    replay_day: date,
) -> list[Any]:
    return [
        bar
        for bar in (getattr(fetch, "bars", None) or [])
        if from_date <= _bar_date(bar) <= replay_day
    ]


def _slice_has_evidence_session_bar(bars: Sequence[Any], replay_day: date) -> bool:
    return any(
        _bar_date(bar) == replay_day
        and getattr(bar, "split_adjusted_close", None) is not None
        for bar in bars
    )


def _fetch_has_evidence_session_bar(fetch: Any, replay_day: date) -> bool:
    return any(
        _bar_date(bar) == replay_day
        and getattr(bar, "split_adjusted_close", None) is not None
        for bar in (getattr(fetch, "bars", None) or [])
    )


def _select_range_fetch_for_date(fetch_entry: Any, replay_day: date) -> Any | None:
    providers = _range_fetch_providers(fetch_entry)
    fmp_fetch = providers.get("FMP")
    if fmp_fetch is not None and _fetch_has_evidence_session_bar(fmp_fetch, replay_day):
        return fmp_fetch
    polygon_fetch = providers.get("Polygon")
    if polygon_fetch is not None and _fetch_has_evidence_session_bar(
        polygon_fetch,
        replay_day,
    ):
        return polygon_fetch
    for provider, fetch in providers.items():
        if provider in {"FMP", "Polygon"}:
            continue
        if _fetch_has_evidence_session_bar(fetch, replay_day):
            return fetch
    return None


def _range_missing_price_evidence_error(
    *,
    ticker: str,
    source_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "stage": "historical_price_evidence",
        "ticker": ticker,
        "error_type": "missing_price_evidence",
        "message": "no usable historical daily bars with evidence-session coverage",
        "retryable": False,
        "source_attempts": source_attempts,
    }


def _snap_attr(snapshot: Any, name: str) -> Any:
    if isinstance(snapshot, dict):
        return snapshot.get(name)
    return getattr(snapshot, name, None)


_REPLAY_SNAPSHOT_COLUMNS = [
    "universe_snapshot_id",
    "job_run_id",
    "scan_id",
    "ticker",
    "asof_timestamp",
    "source_provider",
    "market_cap",
    "price",
    "country",
    "security_type",
    "primary_exchange",
    "operating_universe_inclusion",
    "exclusion_reason",
    "dataset_version",
    "schema_hash",
    "source_lineage_hash",
]


def _postgres_stage_merge_replay_snapshots(
    session: Session,
    rows: list[dict[str, Any]],
) -> None:
    stage_table = f"pg_temp.replay_snapshot_stage_{uuid4().hex}"
    session.execute(
        text(
            f"""
            CREATE TEMP TABLE {stage_table} (
                universe_snapshot_id text PRIMARY KEY,
                job_run_id text,
                scan_id text,
                ticker text,
                asof_timestamp timestamptz,
                source_provider text,
                market_cap double precision,
                price double precision,
                country text,
                security_type text,
                primary_exchange text,
                operating_universe_inclusion boolean,
                exclusion_reason text,
                dataset_version text,
                schema_hash text,
                source_lineage_hash text
            ) ON COMMIT DROP
            """
        )
    )
    columns = ", ".join(_REPLAY_SNAPSHOT_COLUMNS)
    copy_sql = f"COPY {stage_table} ({columns}) FROM STDIN"
    connection = session.connection()
    driver_connection = getattr(
        getattr(connection, "connection", None),
        "driver_connection",
        None,
    )
    if driver_connection is None:
        session.execute(
            text(
                f"""
                INSERT INTO {stage_table} ({columns})
                VALUES (
                    :universe_snapshot_id, :job_run_id, :scan_id, :ticker,
                    :asof_timestamp, :source_provider, :market_cap, :price,
                    :country, :security_type, :primary_exchange,
                    :operating_universe_inclusion, :exclusion_reason,
                    :dataset_version, :schema_hash, :source_lineage_hash
                )
                """
            ),
            rows,
        )
    else:
        with driver_connection.cursor() as cursor:
            with cursor.copy(copy_sql) as copy:
                for row in rows:
                    copy.write_row(
                        [_copy_stage_scalar(row.get(column)) for column in _REPLAY_SNAPSHOT_COLUMNS]
                    )
    update_columns = [
        column
        for column in _REPLAY_SNAPSHOT_COLUMNS
        if column != "universe_snapshot_id"
    ]
    update_assignments = ", ".join(
        f"{column} = EXCLUDED.{column}" for column in update_columns
    )
    distinct_lhs = ", ".join(f"universe_snapshots.{column}" for column in update_columns)
    distinct_rhs = ", ".join(f"EXCLUDED.{column}" for column in update_columns)
    session.execute(
        text(
            f"""
            INSERT INTO universe_snapshots ({columns})
            SELECT {columns}
            FROM {stage_table}
            ON CONFLICT (universe_snapshot_id) DO UPDATE
            SET {update_assignments}
            WHERE ({distinct_lhs}) IS DISTINCT FROM ({distinct_rhs})
            """
        )
    )
    session.flush()


def _copy_stage_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\x00", "")
    return value


def _feature_key(record: _BulkDetectionRecord) -> tuple[str, str, datetime, str]:
    return (
        record.result.pattern_id,
        record.ticker,
        _aware_utc(record.result.asof_timestamp),
        record.feature_hash,
    )


def _existing_feature_snapshots(
    session: Session,
    records: Sequence[_BulkDetectionRecord],
) -> dict[tuple[str, str, datetime, str], FeatureSnapshot]:
    if not records:
        return {}
    tickers = sorted({record.ticker for record in records})
    asofs = sorted({_aware_utc(record.result.asof_timestamp) for record in records})
    feature_hashes = sorted({record.feature_hash for record in records})
    rows = (
        session.query(FeatureSnapshot)
        .filter(
            FeatureSnapshot.pattern_id == "M4",
            FeatureSnapshot.ticker.in_(tickers),
            FeatureSnapshot.asof_timestamp.in_(asofs),
            FeatureSnapshot.feature_hash.in_(feature_hashes),
        )
        .all()
    )
    by_key: dict[tuple[str, str, datetime, str], FeatureSnapshot] = {}
    for row in rows:
        by_key[
            (
                row.pattern_id,
                row.ticker,
                _aware_utc(row.asof_timestamp),
                row.feature_hash,
            )
        ] = row
    return by_key


def _existing_signals(
    session: Session,
    records: Sequence[_BulkDetectionRecord],
) -> dict[tuple[str, str, str | None], SignalRegistry]:
    hashes = sorted(
        {
            record.signal_identity_hash
            for record in records
            if record.signal_identity_hash
        }
    )
    if not hashes:
        return {}
    tickers = sorted({record.ticker for record in records if record.signal_identity_hash})
    rows = (
        session.query(SignalRegistry)
        .filter(
            SignalRegistry.pattern_id == "M4",
            SignalRegistry.ticker.in_(tickers),
            SignalRegistry.signal_identity_hash.in_(hashes),
        )
        .all()
    )
    return {
        (row.pattern_id, row.ticker, row.signal_identity_hash): row
        for row in rows
    }


def _resolved_input_lineage_ids_from_cache(
    inp: Any,
    lineage_ids_by_hash: dict[str, list[str]],
) -> list[str]:
    lineage_ids: list[str] = []
    seen: set[str] = set()
    for lineage_id in getattr(inp, "lineage_ids", []):
        lineage_id_str = str(lineage_id or "").strip()
        if lineage_id_str and lineage_id_str not in seen:
            lineage_ids.append(lineage_id_str)
            seen.add(lineage_id_str)
    for lineage_hash in getattr(inp, "lineage_hashes", []):
        lineage_hash_str = str(lineage_hash or "").strip()
        if not lineage_hash_str:
            continue
        for lineage_id in lineage_ids_by_hash.get(lineage_hash_str, []):
            if lineage_id and lineage_id not in seen:
                lineage_ids.append(lineage_id)
                seen.add(lineage_id)
    return lineage_ids


def _stamped_replay_feature_payload(
    features: dict[str, Any],
    *,
    replay_day: date,
    partial_reason: str | None,
    bar_lineage: DataLineage | None,
) -> dict[str, Any]:
    replay_date = replay_day.isoformat()
    lineage_metadata = _bar_metadata_from_lineage(bar_lineage)
    metadata = {
        "reconstructed": True,
        "reconstruction_method": RECONSTRUCTION_METHOD,
        "replay_date": replay_date,
        "evidence_session_date": replay_date,
        "source_universe_method": SOURCE_UNIVERSE_METHOD,
        "bar_provider": "FMP",
        "bar_endpoint": HISTORICAL_PRICE_FULL_ENDPOINT,
        "bar_provider_policy": BAR_PROVIDER_POLICY,
        "price_basis": FMP_PRICE_BASIS,
        "h52w_basis": "split_adjusted_close_prior_252_sessions",
        "partial_universe_reason": partial_reason,
        "range_level_cached_replay": True,
        "fallback_used": False,
    }
    metadata.update(lineage_metadata)
    if bar_lineage is not None:
        metadata.update(
            {
                "bar_lineage_id": bar_lineage.data_lineage_id,
                "bar_lineage_hash": bar_lineage.raw_payload_hash,
            }
        )
    stamped = dict(features)
    stamped["historical_replay"] = metadata
    stamped.update(
        {
            "reconstructed": True,
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "replay_date": replay_date,
            "evidence_session_date": replay_date,
            "source_universe_method": SOURCE_UNIVERSE_METHOD,
            "bar_provider": metadata.get("bar_provider"),
            "bar_endpoint": metadata.get("bar_endpoint"),
            "bar_provider_policy": metadata.get("bar_provider_policy"),
            "fallback_used": metadata.get("fallback_used"),
            "bar_lineage_id": metadata.get("bar_lineage_id"),
            "bar_lineage_hash": metadata.get("bar_lineage_hash"),
            "price_basis": metadata.get("price_basis"),
            "bar_adjusted": metadata.get("adjusted"),
            "bar_requested_adjusted": metadata.get("requested_adjusted"),
            "bar_adjustment_basis": metadata.get("adjustment_basis"),
        }
    )
    return stamped


def _feature_snapshot_mapping(
    record: _BulkDetectionRecord,
    *,
    feature_snapshot_id: str,
    job_run_id: str,
    code_commit_sha: str | None,
) -> dict[str, Any]:
    return {
        "feature_snapshot_id": feature_snapshot_id,
        "job_run_id": job_run_id,
        "evidence_snapshot_id": None,
        "pattern_id": record.result.pattern_id,
        "ticker": record.ticker,
        "asof_timestamp": record.result.asof_timestamp,
        "feature_manifest_version": record.result.features.feature_manifest_version,
        "feature_json": record.feature_json,
        "feature_hash": record.feature_hash,
        "data_lineage_ids": json.dumps(record.data_lineage_ids),
        "code_commit_sha": code_commit_sha,
        "fidelity_tier": record.result.features.fidelity_tier,
        "point_in_time_passed": record.result.features.point_in_time_passed,
        "lookahead_guard_passed": record.result.features.lookahead_guard_passed,
        "input_hashes": (
            json.dumps(record.result.input_hashes, sort_keys=True, default=str)
            if record.result.input_hashes
            else None
        ),
        "output_hash": record.output_hash,
    }


def _signal_mapping(
    record: _BulkDetectionRecord,
    signal: Any,
    *,
    detector: M4Detector,
    detector_version: str,
    signal_id: str,
    feature_snapshot_id: str,
    job_run_id: str,
    replay_date: str,
    scan_id: str,
) -> dict[str, Any]:
    return {
        "signal_id": signal_id,
        "job_run_id": job_run_id,
        "pattern_id": record.result.pattern_id,
        "ticker": record.ticker,
        "direction": signal.direction,
        "signal_timestamp": record.result.asof_timestamp,
        "raw_signal_strength": signal.raw_signal_strength,
        "raw_expected_edge": signal.raw_expected_edge,
        "feature_snapshot_id": feature_snapshot_id,
        "signal_status": signal.signal_status,
        "signal_horizon": signal.signal_horizon,
        "thesis_category": detector.thesis_category,
        "route_class": signal.route_class or detector.route_class,
        "fidelity_tier": record.result.features.fidelity_tier,
        "data_confidence": signal.data_confidence,
        "universe_snapshot_id": record.universe_snapshot_id,
        "trading_date": replay_date,
        "next_execution_session": record.next_execution_session,
        "scan_id": scan_id,
        "detector_version": detector_version,
        "point_in_time_passed": record.result.features.point_in_time_passed,
        "lookahead_guard_passed": record.result.features.lookahead_guard_passed,
        "signal_event_sequence": 1,
        "data_lineage_ids": (
            json.dumps(record.data_lineage_ids) if record.data_lineage_ids else None
        ),
        "signal_identity_hash": record.signal_identity_hash,
        "intended_entry_price": None,
        "forward_return_status": "pending",
        "forward_return_attempts": 0,
    }


def _bulk_metrics(diag: DetectorDiagnostics) -> dict[str, Any]:
    no_fire_count = max(
        0,
        diag.skipped_count
        - diag.identity_refused_count
        - diag.lookahead_failure_count
        - diag.error_count,
    )
    return {
        "trading_date": None,
        "scan_id": None,
        "universe_size": diag.evaluated_count,
        "detector_count": 1,
        "total_signals_persisted": diag.fired_count,
        "detector_diagnostics": [diag.to_dict()],
        "any_detector_failed": diag.detector_status in {"failed", "partial_failed"},
        "assembly_diagnostics": [],
        "no_fire_count": no_fire_count,
    }


def _m4_detector_diagnostic(metrics: dict[str, Any] | None) -> dict[str, Any]:
    for diag in ((metrics or {}).get("detector_diagnostics") or []):
        if diag.get("detector_id") == "M4":
            return diag
    return {}


def _validate_range_replay(session: Session, replay_dates: Sequence[date]) -> dict[str, int]:
    replay_date_strings = [day.isoformat() for day in replay_dates]
    duplicate_hur_groups = int(
        session.query(
            HistoricalUniverseReconstruction.replay_date,
            HistoricalUniverseReconstruction.normalized_symbol,
            func.count().label("row_count"),
        )
        .filter(HistoricalUniverseReconstruction.replay_date.in_(list(replay_dates)))
        .group_by(
            HistoricalUniverseReconstruction.replay_date,
            HistoricalUniverseReconstruction.normalized_symbol,
        )
        .having(func.count() > 1)
        .count()
    )
    duplicate_signal_groups = int(
        session.query(
            SignalRegistry.pattern_id,
            SignalRegistry.trading_date,
            SignalRegistry.ticker,
            SignalRegistry.signal_identity_hash,
            func.count().label("row_count"),
        )
        .filter(
            SignalRegistry.pattern_id == "M4",
            SignalRegistry.trading_date.in_(replay_date_strings),
        )
        .group_by(
            SignalRegistry.pattern_id,
            SignalRegistry.trading_date,
            SignalRegistry.ticker,
            SignalRegistry.signal_identity_hash,
        )
        .having(func.count() > 1)
        .count()
    )
    duplicate_feature_groups = int(
        session.query(
            FeatureSnapshot.pattern_id,
            FeatureSnapshot.ticker,
            FeatureSnapshot.asof_timestamp,
            func.count().label("row_count"),
        )
        .filter(
            FeatureSnapshot.pattern_id == "M4",
            FeatureSnapshot.asof_timestamp.in_(
                [us_equity_session_close_timestamp(day) for day in replay_dates]
            ),
        )
        .group_by(
            FeatureSnapshot.pattern_id,
            FeatureSnapshot.ticker,
            FeatureSnapshot.asof_timestamp,
        )
        .having(func.count() > 1)
        .count()
    )
    missing_hur_hash = int(
        session.query(func.count(HistoricalUniverseReconstruction.historical_universe_reconstruction_id))
        .filter(
            HistoricalUniverseReconstruction.replay_date.in_(list(replay_dates)),
            (
                (HistoricalUniverseReconstruction.input_hash.is_(None))
                | (HistoricalUniverseReconstruction.output_hash.is_(None))
                | (HistoricalUniverseReconstruction.data_lineage_id.is_(None))
            ),
        )
        .scalar()
        or 0
    )
    feature_rows = (
        session.query(FeatureSnapshot)
        .filter(
            FeatureSnapshot.pattern_id == "M4",
            FeatureSnapshot.asof_timestamp.in_(
                [us_equity_session_close_timestamp(day) for day in replay_dates]
            ),
        )
        .all()
    )
    missing_feature_hash = 0
    missing_replay_stamp = 0
    lookahead_violations = 0
    for row in feature_rows:
        if not row.input_hashes or not row.output_hash or not row.data_lineage_ids:
            missing_feature_hash += 1
        features = _json_dict(row.feature_json)
        replay_date = features.get("replay_date")
        if features.get("reconstruction_method") != RECONSTRUCTION_METHOD:
            missing_replay_stamp += 1
        lookback_end = features.get("lookback_end")
        if replay_date and lookback_end and str(lookback_end) >= str(replay_date):
            lookahead_violations += 1
    return {
        "duplicate_historical_universe_groups": duplicate_hur_groups,
        "duplicate_m4_signal_identity_groups": duplicate_signal_groups,
        "duplicate_m4_feature_snapshot_groups": duplicate_feature_groups,
        "missing_historical_universe_lineage_hash_count": missing_hur_hash,
        "missing_feature_lineage_hash_count": missing_feature_hash,
        "missing_historical_replay_stamp_count": missing_replay_stamp,
        "feature_lookahead_violation_count": lookahead_violations,
    }


def _coverage_status(
    *,
    missing_price_evidence_count: int,
    m4_evaluable_count: int,
    hard_error_count: int,
) -> str:
    if hard_error_count:
        return "hard_error"
    if missing_price_evidence_count and m4_evaluable_count <= 0:
        return "no_evaluable_price_evidence"
    if missing_price_evidence_count:
        return "partial_price_evidence"
    return "complete_price_evidence"


def _completion_classification(
    *,
    missing_price_evidence_count: int,
    m4_evaluable_count: int,
    hard_error_count: int,
) -> str:
    if hard_error_count:
        return "hard_failure"
    if missing_price_evidence_count and m4_evaluable_count <= 0:
        return "failed_no_evaluable_price_evidence"
    if missing_price_evidence_count:
        return "completed_with_non_evaluable_price_evidence"
    return "completed"


def _date_completion_classification(
    *,
    missing_price_evidence_count: int,
    m4_evaluable_count: int,
    has_hard_failure: bool,
    detector_status: str,
) -> str:
    if has_hard_failure:
        return (
            "partial_hard_failure"
            if detector_status == "partial_failed"
            else "hard_failure"
        )
    if missing_price_evidence_count and m4_evaluable_count <= 0:
        return "completed_with_no_evaluable_price_evidence"
    if missing_price_evidence_count:
        return "completed_with_non_evaluable_price_evidence"
    return "completed"


def _missing_price_evidence_errors_from_date_results(
    date_results: Sequence[dict[str, Any]],
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for row in date_results:
        for error in row.get("fetch_errors") or []:
            if error.get("error_type") != "missing_price_evidence":
                continue
            errors.append(error)
            if len(errors) >= limit:
                return errors
    return errors


def _non_evaluable_price_evidence_samples(
    session: Session,
    *,
    rows_by_ticker: dict[str, HistoricalUniverseReconstruction],
    fetch_errors: Sequence[dict[str, Any]],
    replay_day: date,
    limit: int = 25,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    replay_iso = replay_day.isoformat()
    for error in fetch_errors:
        if error.get("error_type") != "missing_price_evidence":
            continue
        ticker = str(error.get("ticker") or "").upper()
        if not ticker:
            continue
        row = rows_by_ticker.get(ticker)
        source_attempts = error.get("source_attempts") or []
        samples.append(
            {
                "ticker": ticker,
                "source": getattr(row, "source", None) if row is not None else None,
                "exchange": getattr(row, "exchange", None) if row is not None else None,
                "security_type": _source_security_type(session, row),
                "category_hint": _symbol_category_hint(ticker),
                "provider_attempt_statuses": _compact_source_attempts(source_attempts),
                "missing_evidence_dates": _missing_evidence_dates(
                    source_attempts,
                    default_date=replay_iso,
                ),
            }
        )
        if len(samples) >= limit:
            break
    return samples


def _summarize_non_evaluable_samples(
    date_results: Sequence[dict[str, Any]],
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    by_ticker: dict[str, dict[str, Any]] = {}
    for row in date_results:
        for sample in row.get("non_evaluable_price_evidence_samples") or []:
            ticker = sample.get("ticker")
            if not ticker:
                continue
            existing = by_ticker.setdefault(ticker, {**sample})
            dates = set(existing.get("missing_evidence_dates") or [])
            dates.update(sample.get("missing_evidence_dates") or [])
            existing["missing_evidence_dates"] = sorted(dates)
    return [by_ticker[ticker] for ticker in sorted(by_ticker)][:limit]


def _compact_source_attempts(
    source_attempts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    allowed = {
        "provider",
        "status",
        "row_count",
        "usable_bar_count",
        "missing_evidence_dates",
        "evidence_session_bar_present",
        "requested_adjusted",
        "adjusted",
        "price_basis",
        "fallback_used",
        "error_type",
        "message",
        "retryable",
    }
    for attempt in source_attempts:
        if not isinstance(attempt, dict):
            continue
        compact.append({key: attempt.get(key) for key in allowed if key in attempt})
    return compact


def _missing_evidence_dates(
    source_attempts: Sequence[dict[str, Any]],
    *,
    default_date: str,
) -> list[str]:
    dates: set[str] = {default_date}
    for attempt in source_attempts:
        if not isinstance(attempt, dict):
            continue
        for value in attempt.get("missing_evidence_dates") or []:
            dates.add(str(value))
    return sorted(dates)


def _source_security_type(
    session: Session,
    row: HistoricalUniverseReconstruction | None,
) -> str | None:
    if row is None:
        return None
    if row.current_universe_snapshot_id:
        snapshot = session.get(UniverseSnapshot, row.current_universe_snapshot_id)
        if snapshot is not None:
            return snapshot.security_type
    if row.fmp_delisted_company_id:
        delisted = session.get(FmpDelistedCompanyRecord, row.fmp_delisted_company_id)
        if delisted is not None:
            payload = _json_dict(delisted.raw_payload_json)
            for key in ("securityType", "security_type", "type"):
                value = payload.get(key)
                if value:
                    return str(value)
    return None


def _symbol_category_hint(ticker: str) -> str | None:
    symbol = ticker.upper()
    if "." in symbol or "-" in symbol:
        return "contains_symbol_separator"
    if symbol.endswith(("WS", "WT", "WTS", "W")):
        return "possible_warrant_suffix"
    if symbol.endswith(("UN", "U")):
        return "possible_unit_suffix"
    if symbol.endswith(("RT", "R")):
        return "possible_right_suffix"
    return None


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


def _aware_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _write_artifact(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _default_artifact_path() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"/tmp/historical_m4_range_replay_{ts}.json"


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
        _validate_pattern_ids(args.pattern_id)
        if not target_schema:
            raise ValueError("historical M4 range replay requires scratch --schema")
        if target_schema.casefold() == "public":
            raise ValueError("historical M4 range replay refuses --schema public")
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        prepare_writable_schema_target(
            schema=target_schema,
            create_tables=args.create_tables,
            required_tables=HISTORICAL_COHORT_REQUIRED_TABLES,
        )
    except (SchemaTargetError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        fmp_adapter = CachedHistoricalPriceFmpAdapter(FmpAdapter(FmpConfig.from_env()))
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 1
    polygon_adapter = _optional_polygon_adapter()

    artifact_path = Path(args.progress_artifact or _default_artifact_path())
    artifact: dict[str, Any] = {
        "job_name": JOB_NAME,
        "schema": target_schema,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "pattern_ids": args.pattern_id,
        "events": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_artifact(artifact_path, artifact)

    def progress(event: str, payload: dict[str, Any]) -> None:
        artifact["events"].append(payload)
        if len(artifact["events"]) > 500:
            del artifact["events"][:-500]
        artifact["last_event"] = payload
        _write_artifact(artifact_path, artifact)
        print(json.dumps({"event": event, **payload}, sort_keys=True, default=str))

    session = get_session()
    try:
        result = run_historical_m4_range_replay(
            session=session,
            fmp_adapter=fmp_adapter,
            polygon_adapter=polygon_adapter,
            start_date=_parse_date(args.start_date),
            end_date=_parse_date(args.end_date),
            run_timestamp=_parse_timestamp(args.run_timestamp),
            allow_partial_delisted_source=args.allow_partial_delisted_source,
            allow_partial_universe=args.allow_partial_universe,
            lookback_calendar_days=args.lookback_calendar_days,
            skip_completed_dates=args.skip_completed_dates,
            progress_callback=progress,
            progress_every=args.progress_every,
        )
        artifact["status"] = result.status
        artifact["summary"] = result.metrics
        artifact["errors"] = result.errors
        artifact["ended_at"] = datetime.now(timezone.utc).isoformat()
        _write_artifact(artifact_path, artifact)
        print(json.dumps(result.metrics, indent=2, sort_keys=True, default=str))
        print(f"Artifact: {artifact_path}")
        return 0 if result.ok else 1
    finally:
        session.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scratch-only range-cached historical M4 replay."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--database-url")
    parser.add_argument("--schema")
    parser.add_argument("--create-tables", action="store_true")
    parser.add_argument("--pattern-id", action="append", default=["M4"])
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--run-timestamp")
    parser.add_argument("--lookback-calendar-days", type=int, default=LOOKBACK_CALENDAR_DAYS)
    parser.add_argument("--allow-partial-delisted-source", action="store_true")
    parser.add_argument("--allow-partial-universe", action="store_true")
    parser.add_argument("--skip-completed-dates", action="store_true")
    parser.add_argument("--progress-artifact")
    parser.add_argument("--progress-every", type=int, default=25)
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
