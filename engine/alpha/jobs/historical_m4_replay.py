"""Scratch-only historical M4 signal replay.

Replays base-daily M4 signals for reconstructed PIT historical universe rows.
This job intentionally reuses the production M4 assembler, detector, and
orchestration persistence path. It builds a synthetic scratch canonical scan for
the replay date so DetectorOrchestrationJob can keep its normal identity and
lookahead checks.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy.orm import Session

from alpha.assembly.m4_daily import DailyBar, assemble_m4_daily
from alpha.data.contracts import stable_hash
from alpha.data.fmp import FmpBar, HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.db.models import (
    CanonicalUniverseScan,
    DataLineage,
    FeatureSnapshot,
    HistoricalUniverseReconstruction,
    SignalRegistry,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.evidence.writer import record_data_lineage
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.detector_orchestration import DetectorOrchestrationJob
from alpha.jobs.m4_daily import _assembly_metrics
from alpha.market_calendar import (
    next_us_equity_session,
    us_equity_session_close_timestamp,
)
from alpha.patterns.m4 import M4Detector


JOB_NAME = "historical_m4_replay"
RECONSTRUCTION_METHOD = "historical_m4_replay_fmp_eod"
SOURCE_UNIVERSE_METHOD = "active_current_plus_fmp_delisted_v1"
REPLAY_SCAN_PROVIDER = "HISTORICAL_REPLAY"
LOOKBACK_CALENDAR_DAYS = 430
HISTORICAL_REPLAY_MIN_DATE = date(2024, 1, 1)


class HistoricalM4ReplayJob(BaseJob):
    """Replay M4 over historical PIT universe reconstruction rows."""

    def __init__(
        self,
        *,
        session: Session,
        fmp_adapter: Any,
        replay_dates: list[date],
        run_timestamp: datetime | None = None,
        allow_partial_universe: bool = False,
        lookback_calendar_days: int = LOOKBACK_CALENDAR_DAYS,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        progress_every: int = 25,
    ) -> None:
        if not replay_dates:
            raise ValueError("HistoricalM4ReplayJob requires at least one replay date")
        bad_dates = [day for day in replay_dates if day < HISTORICAL_REPLAY_MIN_DATE]
        if bad_dates:
            raise ValueError("historical M4 replay starts at 2024-01-01")
        self._session = session
        self._fmp = fmp_adapter
        self._replay_dates = sorted(set(replay_dates))
        self._run_timestamp = _aware_utc(run_timestamp)
        self._allow_partial_universe = allow_partial_universe
        self._lookback_calendar_days = lookback_calendar_days
        self._progress_callback = progress_callback
        self._progress_every = max(int(progress_every), 1)
        self._started_perf = perf_counter()

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
        metrics: dict[str, Any] = {
            "replay_dates": [day.isoformat() for day in self._replay_dates],
            "allow_partial_universe": self._allow_partial_universe,
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "source_universe_method": SOURCE_UNIVERSE_METHOD,
            "date_results": [],
            "total_universe_included_count": 0,
            "total_tickers_with_bars": 0,
            "total_tickers_missing_bars": 0,
            "total_assembled_count": 0,
            "total_fired_m4_signal_count": 0,
            "total_rejected_or_no_fire_count": 0,
            "total_fetch_error_count": 0,
            "total_rows_inserted": 0,
            "total_rows_reused": 0,
        }
        errors: list[dict[str, Any]] = []
        output_hashes: list[str] = []

        for replay_day in self._replay_dates:
            date_result = self._run_one_date(replay_day, ctx)
            metrics["date_results"].append(date_result.metrics)
            output_hashes.append(stable_hash(date_result.metrics))
            metrics["total_universe_included_count"] += date_result.metrics.get(
                "universe_included_count", 0
            )
            metrics["total_tickers_with_bars"] += date_result.metrics.get(
                "tickers_with_bars", 0
            )
            metrics["total_tickers_missing_bars"] += date_result.metrics.get(
                "tickers_missing_bars", 0
            )
            metrics["total_assembled_count"] += date_result.metrics.get(
                "assembled_count", 0
            )
            metrics["total_fired_m4_signal_count"] += date_result.metrics.get(
                "fired_m4_signal_count", 0
            )
            metrics["total_rejected_or_no_fire_count"] += date_result.metrics.get(
                "rejected_or_no_fire_count", 0
            )
            metrics["total_fetch_error_count"] += date_result.metrics.get(
                "fetch_error_count", 0
            )
            metrics["total_rows_inserted"] += date_result.metrics.get(
                "rows_inserted", 0
            )
            metrics["total_rows_reused"] += date_result.metrics.get("rows_reused", 0)
            errors.extend(date_result.errors)

        status = "finished"
        if errors:
            status = "partial_failed" if metrics["total_assembled_count"] else "failed"

        return JobResult(
            status=status,
            metrics=metrics,
            input_hashes={
                "historical_m4_replay_input": stable_hash(
                    {
                        "replay_dates": [day.isoformat() for day in self._replay_dates],
                        "allow_partial_universe": self._allow_partial_universe,
                        "lookback_calendar_days": self._lookback_calendar_days,
                    }
                )
            },
            output_hashes={
                "historical_m4_replay_output": stable_hash(output_hashes),
            },
            errors=errors,
        )

    def _run_one_date(self, replay_day: date, ctx: JobContext) -> JobResult:
        replay_date_str = replay_day.isoformat()
        cutoff_timestamp = us_equity_session_close_timestamp(replay_day)
        next_execution = next_us_equity_session(replay_day + timedelta(days=1))
        self._emit_progress("included_universe_load_start", {"replay_date": replay_date_str})
        included_rows = (
            self._session.query(HistoricalUniverseReconstruction)
            .filter(
                HistoricalUniverseReconstruction.replay_date == replay_day,
                HistoricalUniverseReconstruction.inclusion_status == "included",
            )
            .order_by(HistoricalUniverseReconstruction.normalized_symbol)
            .all()
        )
        self._emit_progress(
            "included_universe_load_finish",
            {
                "replay_date": replay_date_str,
                "universe_included_count": len(included_rows),
            },
        )
        partial_reason = _partial_universe_reason(included_rows)
        if partial_reason and not self._allow_partial_universe:
            return JobResult(
                status="failed",
                metrics={
                    "replay_date": replay_date_str,
                    "universe_included_count": len(included_rows),
                    "partial_universe_reason": partial_reason,
                    "rows_inserted": 0,
                    "rows_reused": 0,
                },
                errors=[
                    {
                        "stage": "historical_universe",
                        "replay_date": replay_date_str,
                        "error_type": "partial_historical_universe",
                        "message": (
                            "historical universe reconstruction source is partial; "
                            "rerun with allow_partial_universe only for bounded "
                            "scratch probes"
                        ),
                        "partial_reason": partial_reason,
                    }
                ],
            )

        stage_timings: dict[str, float] = {}
        stage_started = perf_counter()
        universe_lineage = self._record_replay_universe_lineage(
            included_rows,
            replay_day=replay_day,
            cutoff_timestamp=cutoff_timestamp,
            job_run_id=ctx.job_run_id,
            partial_reason=partial_reason,
        )
        stage_timings["universe_lineage_record_seconds"] = round(
            perf_counter() - stage_started,
            6,
        )
        self._emit_progress(
            "replay_scan_snapshot_start",
            {
                "replay_date": replay_date_str,
                "snapshot_count": len(included_rows),
            },
        )
        stage_started = perf_counter()
        snapshots = self._ensure_replay_scan_and_snapshots(
            included_rows,
            replay_day=replay_day,
            cutoff_timestamp=cutoff_timestamp,
            job_run_id=ctx.job_run_id,
            universe_lineage=universe_lineage,
            partial_reason=partial_reason,
        )
        stage_timings["replay_scan_snapshot_seconds"] = round(
            perf_counter() - stage_started,
            6,
        )
        self._emit_progress(
            "replay_scan_snapshot_finish",
            {
                "replay_date": replay_date_str,
                "snapshot_count": len(snapshots),
                "elapsed_seconds": stage_timings["replay_scan_snapshot_seconds"],
            },
        )

        daily_bars: dict[str, list[DailyBar]] = {}
        bar_lineage_by_ticker: dict[str, DataLineage] = {}
        fetch_errors: list[dict[str, Any]] = []
        fetched_bar_count = 0
        from_date = replay_day - timedelta(days=self._lookback_calendar_days)
        ticker_fetch_started = 0
        ticker_fetch_finished = 0
        ticker_fetch_errors = 0
        ticker_fetch_started_at = perf_counter()
        self._emit_progress(
            "ticker_fetch_start",
            {
                "replay_date": replay_date_str,
                "ticker_count": len(snapshots),
                "from_date": from_date.isoformat(),
                "to_date": replay_day.isoformat(),
            },
        )
        for snapshot in snapshots:
            ticker = snapshot.ticker
            ticker_fetch_started += 1
            if (
                ticker_fetch_started == 1
                or ticker_fetch_started % self._progress_every == 0
            ):
                self._emit_progress(
                    "ticker_fetch_progress",
                    {
                        "replay_date": replay_date_str,
                        "ticker": ticker,
                        "started": ticker_fetch_started,
                        "finished": ticker_fetch_finished,
                        "errors": ticker_fetch_errors,
                        "ticker_total": len(snapshots),
                    },
                )
            resp = self._fmp.get_historical_price(
                ticker,
                from_date=from_date,
                to_date=replay_day,
                asof=cutoff_timestamp,
                adjusted=False,
                require_split_adjusted_close=True,
            )
            lineage = self._build_bar_lineage(
                provider=resp.lineage.provider,
                endpoint=resp.lineage.endpoint,
                asof_timestamp=resp.lineage.asof_timestamp,
                request_timestamp=resp.lineage.request_timestamp,
                raw_payload=_lineage_payload(
                    resp.data,
                    ticker=ticker,
                    from_date=from_date,
                    to_date=replay_day,
                ),
                raw_payload_hash=resp.lineage.raw_payload_hash,
                freshness_seconds=resp.lineage.freshness_seconds,
                source_authority=resp.lineage.source_authority,
                data_quality_flags={
                    **(resp.lineage.data_quality_flags or {}),
                    "historical_m4_replay": True,
                    "bar_provider_policy": "fmp_primary_polygon_fallback_not_used",
                    "price_basis": "fmp_full_close_as_split_adjusted_close",
                    "fallback_used": False,
                },
                job_run_id=ctx.job_run_id,
            )
            bar_lineage_by_ticker[ticker] = lineage
            ticker_fetch_finished += 1
            if not resp.ok:
                err = resp.error
                ticker_fetch_errors += 1
                fetch_errors.append(
                    {
                        "stage": "fmp_historical_price",
                        "ticker": ticker,
                        "error_type": getattr(err, "error_type", None),
                        "status_code": getattr(err, "status_code", None),
                        "message": getattr(err, "message", None),
                        "retryable": getattr(err, "retryable", None),
                    }
                )
                continue
            bars = [
                _to_daily_bar(
                    bar,
                    source_timestamp=resp.lineage.asof_timestamp,
                    source_provider=resp.lineage.provider,
                    lineage_id=lineage.data_lineage_id,
                    lineage_hash=resp.lineage.raw_payload_hash,
                )
                for bar in (resp.data or [])
                if _bar_has_required_m4_fields(bar)
            ]
            if not bars:
                fetch_errors.append(
                    {
                        "stage": "fmp_historical_price",
                        "ticker": ticker,
                        "error_type": "missing_bars",
                        "message": "no complete FMP /full bars for replay window",
                        "retryable": False,
                    }
                )
                continue
            daily_bars[ticker] = bars
            fetched_bar_count += len(bars)
        self._session.flush()
        stage_timings["ticker_fetch_seconds"] = round(
            perf_counter() - ticker_fetch_started_at,
            6,
        )
        self._emit_progress(
            "ticker_fetch_finish",
            {
                "replay_date": replay_date_str,
                "started": ticker_fetch_started,
                "finished": ticker_fetch_finished,
                "errors": ticker_fetch_errors,
                "ticker_total": len(snapshots),
                "tickers_with_bars": len(daily_bars),
                "tickers_missing_bars": len(snapshots) - len(daily_bars),
                "fetched_bar_count": fetched_bar_count,
                "cache_hits": int(getattr(self._fmp, "cache_hits", 0) or 0),
                "cache_misses": int(getattr(self._fmp, "cache_misses", 0) or 0),
                "elapsed_seconds": stage_timings["ticker_fetch_seconds"],
            },
        )

        self._emit_progress(
            "assembly_start",
            {
                "replay_date": replay_date_str,
                "snapshot_count": len(snapshots),
                "tickers_with_bars": len(daily_bars),
            },
        )
        stage_started = perf_counter()
        assembly = assemble_m4_daily(
            snapshots=snapshots,
            daily_bars=daily_bars,
            cutoff_timestamp=cutoff_timestamp,
            universe_cutoff_timestamp=cutoff_timestamp,
            decision_date=replay_date_str,
            evidence_session_date=replay_date_str,
            next_execution_session=next_execution.isoformat(),
            source_provider="FMP",
        )
        stage_timings["assembly_seconds"] = round(perf_counter() - stage_started, 6)
        self._emit_progress(
            "assembly_finish",
            {
                "replay_date": replay_date_str,
                "assembled_count": assembly.assembled_count,
                "rejected_count": assembly.rejected_count,
                "insufficient_count": assembly.insufficient_count,
                "elapsed_seconds": stage_timings["assembly_seconds"],
            },
        )

        before_signal_ids = _signal_ids_for_replay(
            self._session,
            replay_date_str,
            scan_id=_replay_scan_id(replay_day),
        )

        self._emit_progress(
            "detector_start",
            {
                "replay_date": replay_date_str,
                "assembled_count": assembly.assembled_count,
            },
        )
        stage_started = perf_counter()
        orchestration = DetectorOrchestrationJob(
            self._session,
            detectors=[M4Detector()],
            trading_date=replay_date_str,
            assembled_inputs={"M4": assembly.inputs},
            nested_persistence=False,
            progress_every=self._progress_every,
            progress_callback=lambda event, payload: self._emit_progress(
                event,
                {"replay_date": replay_date_str, **payload},
            ),
        )
        orchestration_result = orchestration.run(ctx)
        stage_timings["detector_seconds"] = round(perf_counter() - stage_started, 6)
        self._emit_progress(
            "detector_finish",
            {
                "replay_date": replay_date_str,
                "status": orchestration_result.status,
                "elapsed_seconds": stage_timings["detector_seconds"],
            },
        )

        self._emit_progress(
            "persistence_metadata_start",
            {"replay_date": replay_date_str},
        )
        stage_started = perf_counter()
        after_signal_ids = _signal_ids_for_replay(
            self._session,
            replay_date_str,
            scan_id=_replay_scan_id(replay_day),
        )
        inserted_signal_ids = sorted(after_signal_ids - before_signal_ids)
        reused_signal_ids = sorted(after_signal_ids & before_signal_ids)
        replay_metadata = {
            "reconstructed": True,
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "replay_date": replay_date_str,
            "evidence_session_date": replay_date_str,
            "source_universe_method": SOURCE_UNIVERSE_METHOD,
            "source_universe_lineage_id": universe_lineage.data_lineage_id,
            "source_universe_lineage_hash": universe_lineage.raw_payload_hash,
            "bar_provider": "FMP",
            "bar_endpoint": HISTORICAL_PRICE_FULL_ENDPOINT,
            "bar_provider_policy": "fmp_primary_polygon_fallback_not_used",
            "price_basis": "fmp_full_close_as_split_adjusted_close",
            "h52w_basis": "split_adjusted_close_prior_252_sessions",
            "partial_universe_reason": partial_reason,
        }
        replay_feature_ids = _feature_ids_for_replay_inputs(
            self._session,
            replay_date_str,
            tickers=[snapshot.ticker for snapshot in snapshots],
            cutoff_timestamp=cutoff_timestamp,
            bar_lineage_by_ticker=bar_lineage_by_ticker,
        )
        replay_signal_feature_ids = set(
            _feature_ids_for_signals(self._session, sorted(after_signal_ids))
        )
        stamped_feature_count = self._stamp_replay_feature_metadata(
            feature_snapshot_ids=replay_feature_ids,
            replay_metadata=replay_metadata,
            bar_lineage_by_ticker=bar_lineage_by_ticker,
        )
        stamped_fired_feature_count = len(
            set(replay_feature_ids).intersection(replay_signal_feature_ids)
        )
        stage_timings["persistence_metadata_seconds"] = round(
            perf_counter() - stage_started,
            6,
        )
        self._emit_progress(
            "persistence_metadata_finish",
            {
                "replay_date": replay_date_str,
                "inserted_signal_count": len(inserted_signal_ids),
                "reused_signal_count": len(reused_signal_ids),
                "stamped_feature_count": stamped_feature_count,
                "elapsed_seconds": stage_timings["persistence_metadata_seconds"],
            },
        )

        detector_diag = _m4_detector_diagnostic(orchestration_result.metrics)
        fired_count = len(inserted_signal_ids)
        duplicate_reused_count = detector_diag.get("duplicate_suppressed_count", 0)
        evaluated_count = detector_diag.get("evaluated_count", assembly.assembled_count)
        no_fire_count = max(
            0,
            evaluated_count
            - fired_count
            - duplicate_reused_count
            - detector_diag.get("identity_refused_count", 0)
            - detector_diag.get("lookahead_failure_count", 0)
            - detector_diag.get("error_count", 0),
        )
        metrics = {
            "replay_date": replay_date_str,
            "evidence_session_date": replay_date_str,
            "next_execution_session": next_execution.isoformat(),
            "universe_included_count": len(snapshots),
            "partial_universe_reason": partial_reason,
            "tickers_with_bars": len(daily_bars),
            "tickers_missing_bars": len(snapshots) - len(daily_bars),
            "fetched_bar_count": fetched_bar_count,
            "fetch_error_count": len(fetch_errors),
            "fetch_errors": fetch_errors[:50],
            "polygon_fallback_count": 0,
            "assembled_count": assembly.assembled_count,
            "assembly": _assembly_metrics(assembly),
            "fired_m4_signal_count": fired_count,
            "reused_existing_signal_count": len(reused_signal_ids),
            "duplicate_suppressed_count": duplicate_reused_count,
            "rejected_or_no_fire_count": (
                assembly.rejected_count
                + assembly.insufficient_count
                + no_fire_count
                + detector_diag.get("identity_refused_count", 0)
                + detector_diag.get("lookahead_failure_count", 0)
                + detector_diag.get("error_count", 0)
            ),
            "rows_inserted": fired_count,
            "rows_reused": len(reused_signal_ids),
            "stamped_feature_count": stamped_feature_count,
            "stamped_fired_feature_count": stamped_fired_feature_count,
            "stamped_no_fire_feature_count": max(
                0, stamped_feature_count - stamped_fired_feature_count
            ),
            "sample_fired_tickers": _signal_tickers(self._session, inserted_signal_ids)[:20],
            "scan_id": _replay_scan_id(replay_day),
            "orchestration": orchestration_result.metrics,
            "stage_timing_seconds": stage_timings,
            "progress_events": self._recent_progress_events_for_date(replay_date_str),
            "fmp_historical_price_cache_hits": int(
                getattr(self._fmp, "cache_hits", 0) or 0
            ),
            "fmp_historical_price_cache_misses": int(
                getattr(self._fmp, "cache_misses", 0) or 0
            ),
        }
        errors = list(fetch_errors)
        errors.extend(orchestration_result.errors or [])
        status = orchestration_result.status
        if not assembly.inputs and fetch_errors:
            status = "failed"
        elif fetch_errors and status == "finished":
            status = "partial_failed"
        return JobResult(status=status, metrics=metrics, errors=errors)

    def _record_replay_universe_lineage(
        self,
        rows: list[HistoricalUniverseReconstruction],
        *,
        replay_day: date,
        cutoff_timestamp: datetime,
        job_run_id: str,
        partial_reason: str | None,
    ) -> DataLineage:
        payload = {
            "replay_date": replay_day.isoformat(),
            "source_universe_method": SOURCE_UNIVERSE_METHOD,
            "included_rows": [
                {
                    "historical_universe_reconstruction_id": (
                        row.historical_universe_reconstruction_id
                    ),
                    "ticker": row.normalized_symbol,
                    "input_hash": row.input_hash,
                    "output_hash": row.output_hash,
                }
                for row in rows
            ],
        }
        return record_data_lineage(
            self._session,
            provider="DERIVED",
            endpoint="historical_m4_replay_universe",
            asof_timestamp=cutoff_timestamp,
            request_timestamp=self._run_timestamp,
            raw_payload=payload,
            source_authority="alpha_engine",
            data_quality_flags={
                "historical_m4_replay": True,
                "reconstructed": True,
                "source_universe_method": SOURCE_UNIVERSE_METHOD,
                "partial_universe_reason": partial_reason,
            },
            job_run_id=job_run_id,
        )

    def _build_bar_lineage(
        self,
        *,
        provider: str,
        endpoint: str,
        asof_timestamp: datetime,
        request_timestamp: datetime | None,
        raw_payload: Any | None,
        raw_payload_hash: str | None,
        freshness_seconds: float | None,
        source_authority: str | None,
        data_quality_flags: dict[str, Any] | None,
        job_run_id: str | None,
    ) -> DataLineage:
        lineage = DataLineage(
            data_lineage_id=str(uuid4()),
            provider=provider,
            endpoint=endpoint,
            request_timestamp=request_timestamp or datetime.now(timezone.utc),
            asof_timestamp=asof_timestamp,
            raw_payload_hash=raw_payload_hash or stable_hash(raw_payload),
            raw_payload_json=(
                json.dumps(raw_payload, sort_keys=True, default=str)
                if raw_payload is not None
                else None
            ),
            freshness_seconds=freshness_seconds,
            source_authority=source_authority,
            data_quality_flags=(
                json.dumps(data_quality_flags, sort_keys=True, default=str)
                if data_quality_flags is not None
                else None
            ),
            job_run_id=job_run_id,
        )
        self._session.add(lineage)
        return lineage

    def _ensure_replay_scan_and_snapshots(
        self,
        rows: list[HistoricalUniverseReconstruction],
        *,
        replay_day: date,
        cutoff_timestamp: datetime,
        job_run_id: str,
        universe_lineage: DataLineage,
        partial_reason: str | None,
    ) -> list[UniverseSnapshot]:
        scan_id = _replay_scan_id(replay_day)
        scan = self._session.get(UniverseScan, scan_id)
        if scan is None:
            scan = UniverseScan(
                scan_id=scan_id,
                trading_date=replay_day.isoformat(),
                job_run_id=job_run_id,
                asof_timestamp=cutoff_timestamp,
                provider=REPLAY_SCAN_PROVIDER,
                raw_count=len(rows),
                deduped_count=len(rows),
                included_count=len(rows),
                excluded_count=0,
                source_lineage_hash=universe_lineage.raw_payload_hash,
                run_status="finished" if partial_reason is None else "partial_replay",
                metric_json=json.dumps(
                    {
                        "reconstructed": True,
                        "reconstruction_method": RECONSTRUCTION_METHOD,
                        "source_universe_method": SOURCE_UNIVERSE_METHOD,
                        "partial_universe_reason": partial_reason,
                    },
                    sort_keys=True,
                ),
            )
            self._session.add(scan)
        else:
            scan.job_run_id = job_run_id
            scan.asof_timestamp = cutoff_timestamp
            scan.raw_count = len(rows)
            scan.deduped_count = len(rows)
            scan.included_count = len(rows)
            scan.source_lineage_hash = universe_lineage.raw_payload_hash

        canonical = self._session.get(CanonicalUniverseScan, replay_day.isoformat())
        if canonical is None:
            self._session.add(
                CanonicalUniverseScan(
                    trading_date=replay_day.isoformat(),
                    scan_id=scan_id,
                    selected_job_run_id=job_run_id,
                    selected_at=cutoff_timestamp,
                    selection_reason="historical_m4_replay_scratch_scan",
                )
            )
        else:
            canonical.scan_id = scan_id
            canonical.selected_job_run_id = job_run_id
            canonical.selected_at = cutoff_timestamp
            canonical.selection_reason = "historical_m4_replay_scratch_scan"

        snapshots: list[UniverseSnapshot] = []
        snapshot_ids = [_replay_snapshot_id(replay_day, row.normalized_symbol) for row in rows]
        existing_by_id = {
            row.universe_snapshot_id: row
            for row in (
                self._session.query(UniverseSnapshot)
                .filter(UniverseSnapshot.universe_snapshot_id.in_(snapshot_ids))
                .all()
                if snapshot_ids
                else []
            )
        }
        for row in rows:
            snapshot_id = _replay_snapshot_id(replay_day, row.normalized_symbol)
            snapshot = existing_by_id.get(snapshot_id)
            values = {
                "job_run_id": job_run_id,
                "scan_id": scan_id,
                "ticker": row.normalized_symbol,
                "asof_timestamp": cutoff_timestamp,
                "source_provider": REPLAY_SCAN_PROVIDER,
                "market_cap": None,
                "price": None,
                "country": "US",
                "security_type": "common_stock",
                "primary_exchange": row.exchange,
                "operating_universe_inclusion": True,
                "exclusion_reason": None,
                "dataset_version": SOURCE_UNIVERSE_METHOD,
                "schema_hash": RECONSTRUCTION_METHOD,
                "source_lineage_hash": row.output_hash or universe_lineage.raw_payload_hash,
            }
            if snapshot is None:
                snapshot = UniverseSnapshot(
                    universe_snapshot_id=snapshot_id,
                    **values,
                )
                self._session.add(snapshot)
            else:
                for key, value in values.items():
                    setattr(snapshot, key, value)
            snapshots.append(snapshot)
        self._session.flush()
        return snapshots

    def _stamp_replay_feature_metadata(
        self,
        *,
        feature_snapshot_ids: list[str],
        replay_metadata: dict[str, Any],
        bar_lineage_by_ticker: dict[str, DataLineage],
    ) -> int:
        if not feature_snapshot_ids:
            return 0
        rows = (
            self._session.query(FeatureSnapshot)
            .filter(FeatureSnapshot.feature_snapshot_id.in_(feature_snapshot_ids))
            .all()
        )
        stamped = 0
        for row in rows:
            features = _json_dict(row.feature_json)
            lineage = bar_lineage_by_ticker.get(row.ticker)
            metadata = dict(replay_metadata)
            if lineage is not None:
                metadata.update(
                    {
                        "bar_lineage_id": lineage.data_lineage_id,
                        "bar_lineage_hash": lineage.raw_payload_hash,
                    }
                )
            features["historical_replay"] = metadata
            features.update(
                {
                    "reconstructed": True,
                    "reconstruction_method": RECONSTRUCTION_METHOD,
                    "replay_date": replay_metadata["replay_date"],
                    "evidence_session_date": replay_metadata["evidence_session_date"],
                    "source_universe_method": SOURCE_UNIVERSE_METHOD,
                    "bar_provider": "FMP",
                    "bar_lineage_id": metadata.get("bar_lineage_id"),
                    "bar_lineage_hash": metadata.get("bar_lineage_hash"),
                    "price_basis": metadata.get("price_basis"),
                }
            )
            row.feature_json = json.dumps(features, sort_keys=True, default=str)
            row.output_hash = stable_hash(features)
            stamped += 1
        self._session.flush()
        return stamped

    def _emit_progress(self, event: str, payload: dict[str, Any]) -> None:
        event_payload = {
            "event": event,
            "elapsed_seconds": round(perf_counter() - self._started_perf, 6),
            **payload,
        }
        if not hasattr(self, "_progress_events"):
            self._progress_events: list[dict[str, Any]] = []
        self._progress_events.append(event_payload)
        if len(self._progress_events) > 100:
            del self._progress_events[:-100]
        if self._progress_callback is not None:
            try:
                self._progress_callback(event, event_payload)
            except Exception:
                pass

    def _recent_progress_events_for_date(self, replay_date: str) -> list[dict[str, Any]]:
        return [
            event
            for event in getattr(self, "_progress_events", [])
            if event.get("replay_date") == replay_date
        ][-50:]


def _partial_universe_reason(rows: list[HistoricalUniverseReconstruction]) -> str | None:
    reasons: set[str] = set()
    for row in rows:
        provenance = _json_dict(row.source_provenance_json)
        if provenance.get("delisted_source_complete") is False:
            reasons.add(str(provenance.get("delisted_source_partial_reason") or "unknown"))
    if not reasons:
        return None
    return ",".join(sorted(reasons))


def _to_daily_bar(
    bar: FmpBar,
    *,
    source_timestamp: datetime,
    source_provider: str,
    lineage_id: str,
    lineage_hash: str,
) -> DailyBar:
    return DailyBar(
        date=bar.date,
        open=float(bar.open),
        high=float(bar.high),
        low=float(bar.low),
        close=float(bar.close),
        volume=float(bar.volume),
        split_adjusted_close=float(bar.split_adjusted_close),
        adj_close=float(bar.adj_close) if bar.adj_close is not None else None,
        source_timestamp=source_timestamp,
        source_provider=source_provider,
        lineage_id=lineage_id,
        lineage_hash=lineage_hash,
    )


def _bar_has_required_m4_fields(bar: FmpBar) -> bool:
    return all(
        value is not None
        for value in (
            bar.date,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
            bar.split_adjusted_close,
        )
    )


def _lineage_payload(
    bars: Any,
    *,
    ticker: str,
    from_date: date,
    to_date: date,
) -> dict[str, Any]:
    bar_payloads = [
        {
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "split_adjusted_close": bar.split_adjusted_close,
            "adj_close": bar.adj_close,
        }
        for bar in (bars or [])
    ]
    return {
        "ticker": ticker,
        "request": {
            "symbol": ticker,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "endpoint": HISTORICAL_PRICE_FULL_ENDPOINT,
            "adjusted": False,
            "require_split_adjusted_close": True,
        },
        "payload_policy": "compact_bar_digest",
        "bar_count": len(bar_payloads),
        "first_bar_date": bar_payloads[0]["date"] if bar_payloads else None,
        "last_bar_date": bar_payloads[-1]["date"] if bar_payloads else None,
        "bars_digest": stable_hash(bar_payloads),
    }


def _replay_scan_id(replay_day: date) -> str:
    return f"historical-m4-replay-{replay_day.isoformat()}"


def _replay_snapshot_id(replay_day: date, ticker: str) -> str:
    return f"historical-m4-replay-{replay_day.isoformat()}-{ticker.upper()}"


def _signal_ids_for_replay(session: Session, replay_date: str, *, scan_id: str) -> set[str]:
    rows = (
        session.query(SignalRegistry.signal_id)
        .filter(
            SignalRegistry.pattern_id == "M4",
            SignalRegistry.trading_date == replay_date,
            SignalRegistry.scan_id == scan_id,
        )
        .all()
    )
    return {row.signal_id for row in rows}


def _feature_ids_for_signals(session: Session, signal_ids: list[str]) -> list[str]:
    if not signal_ids:
        return []
    rows = (
        session.query(SignalRegistry.feature_snapshot_id)
        .filter(SignalRegistry.signal_id.in_(signal_ids))
        .order_by(SignalRegistry.feature_snapshot_id)
        .all()
    )
    return [row.feature_snapshot_id for row in rows]


def _feature_ids_for_replay_inputs(
    session: Session,
    replay_date: str,
    *,
    tickers: list[str],
    cutoff_timestamp: datetime,
    bar_lineage_by_ticker: dict[str, DataLineage],
) -> list[str]:
    normalized_tickers = sorted({ticker.upper() for ticker in tickers if ticker})
    if not normalized_tickers:
        return []

    lineage_ids_by_ticker = _bar_lineage_ids_by_ticker(session, bar_lineage_by_ticker)
    rows = (
        session.query(FeatureSnapshot)
        .filter(
            FeatureSnapshot.pattern_id == "M4",
            FeatureSnapshot.ticker.in_(normalized_tickers),
        )
        .order_by(FeatureSnapshot.ticker, FeatureSnapshot.feature_snapshot_id)
        .all()
    )
    feature_ids: list[str] = []
    expected_asof = _aware_utc(cutoff_timestamp)
    for row in rows:
        if _aware_utc(row.asof_timestamp) != expected_asof:
            continue
        features = _json_dict(row.feature_json)
        data_lineage_ids = set(_json_list(row.data_lineage_ids))
        ticker_lineage_ids = lineage_ids_by_ticker.get(row.ticker.upper(), set())
        if ticker_lineage_ids.intersection(data_lineage_ids) or _is_stamped_replay_feature(
            features,
            replay_date,
        ):
            feature_ids.append(row.feature_snapshot_id)
    return feature_ids


def _bar_lineage_ids_by_ticker(
    session: Session,
    bar_lineage_by_ticker: dict[str, DataLineage],
) -> dict[str, set[str]]:
    ids_by_ticker: dict[str, set[str]] = {}
    for ticker, lineage in bar_lineage_by_ticker.items():
        ticker_key = ticker.upper()
        ids_by_ticker.setdefault(ticker_key, set()).add(lineage.data_lineage_id)
        if not lineage.raw_payload_hash:
            continue
        rows = (
            session.query(DataLineage.data_lineage_id)
            .filter(DataLineage.raw_payload_hash == lineage.raw_payload_hash)
            .all()
        )
        ids_by_ticker[ticker_key].update(row.data_lineage_id for row in rows)
    return ids_by_ticker


def _is_stamped_replay_feature(features: dict[str, Any], replay_date: str) -> bool:
    historical_replay = _json_dict(features.get("historical_replay"))
    return (
        features.get("reconstruction_method") == RECONSTRUCTION_METHOD
        and features.get("replay_date") == replay_date
    ) or (
        historical_replay.get("reconstruction_method") == RECONSTRUCTION_METHOD
        and historical_replay.get("replay_date") == replay_date
    )


def _m4_detector_diagnostic(metrics: dict[str, Any] | None) -> dict[str, Any]:
    for diag in ((metrics or {}).get("detector_diagnostics") or []):
        if diag.get("detector_id") == "M4":
            return diag
    return {}


def _signal_tickers(session: Session, signal_ids: list[str]) -> list[str]:
    if not signal_ids:
        return []
    rows = (
        session.query(SignalRegistry.ticker)
        .filter(SignalRegistry.signal_id.in_(signal_ids))
        .order_by(SignalRegistry.ticker)
        .all()
    )
    return [row.ticker for row in rows]


def _json_dict(value: str | dict[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: str | list[Any] | None) -> list[str]:
    if not value:
        return []
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item or "").strip()]


def _aware_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
