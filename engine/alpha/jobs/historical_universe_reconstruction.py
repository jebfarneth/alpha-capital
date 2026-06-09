"""Historical PIT universe reconstruction for replay research.

This job answers whether a ticker was eligible for the operating universe on a
historical replay date using current active rows plus FMP delisted-company
directory rows. It writes only to the separate reconstruction table and never
mutates live canonical universe snapshots.
"""

from __future__ import annotations

import json
import csv
from io import StringIO
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import Column, MetaData, Table, or_, text, tuple_
from sqlalchemy.orm import Session

from alpha.data.contracts import stable_hash
from alpha.db.models import (
    CanonicalUniverseScan,
    EvidenceJob,
    EvidenceJobRun,
    FmpDelistedCompanyRecord,
    HistoricalUniverseReconstruction,
    SecurityProfile,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.evidence.writer import record_data_lineage
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.fmp_delisted_companies import JOB_NAME as FMP_DELISTED_JOB_NAME
from alpha.jobs.security_type import NON_COMMON_TYPES
from alpha.jobs.universe_builder import ALLOWED_EXCHANGES, _is_non_common_symbol


JOB_NAME = "historical_universe_reconstruction"
RECONSTRUCTION_METHOD = "active_current_plus_fmp_delisted_v1"
ALLOWED_OPERATING_EXCHANGES = ALLOWED_EXCHANGES
DERIVED_ENDPOINT = "historical_universe_reconstruction"
HISTORICAL_REPLAY_SCAN_PROVIDER = "HISTORICAL_REPLAY"
HISTORICAL_REPLAY_SCAN_SELECTION_REASON = "historical_m4_replay_scratch_scan"


@dataclass
class _SourceInterval:
    symbol: str
    normalized_symbol: str
    source_name: str
    exchange: str | None = None
    company_name: str | None = None
    ipo_date: date | None = None
    delisted_date: date | None = None
    current_universe_snapshot_id: str | None = None
    fmp_delisted_company_id: str | None = None
    data_lineage_id: str | None = None
    input_hash: str | None = None
    source_row: dict[str, Any] = field(default_factory=dict)
    missing_delisted_date_source: bool = False
    security_type: str | None = None


@dataclass
class _EvaluatedInterval:
    interval: _SourceInterval
    inclusion_status: str
    rejection_reason: str | None


@dataclass
class _Candidate:
    symbol: str
    normalized_symbol: str
    intervals: list[_SourceInterval] = field(default_factory=list)


class HistoricalUniverseReconstructionJob(BaseJob):
    """Reconstruct one replay-date universe membership surface."""

    def __init__(
        self,
        *,
        session: Session,
        replay_date: date,
        run_timestamp: datetime | None = None,
        allow_partial_delisted_source: bool = False,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        progress_every: int = 1000,
        persistence_batch_size: int = 1000,
        persist_pre_replay_delisted_exclusions: bool = True,
        compact_persisted_provenance: bool = False,
    ) -> None:
        if replay_date < date(2024, 1, 1):
            raise ValueError("historical replay reconstruction starts at 2024-01-01")
        self.session = session
        self.replay_date = replay_date
        self.run_timestamp = _aware_utc(run_timestamp)
        self.allow_partial_delisted_source = allow_partial_delisted_source
        self.progress_callback = progress_callback
        self.progress_every = max(int(progress_every), 1)
        self.persistence_batch_size = max(int(persistence_batch_size), 1)
        self.persist_pre_replay_delisted_exclusions = (
            persist_pre_replay_delisted_exclusions
        )
        self.compact_persisted_provenance = compact_persisted_provenance
        self._progress_events: list[dict[str, Any]] = []
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
        candidates: dict[str, _Candidate] = {}
        self._emit_progress("candidate_load_start", {})
        active_rows = self._load_active_current_candidates(candidates)
        delisted_rows = self._load_fmp_delisted_candidates(candidates)
        source_interval_count = sum(
            len(candidate.intervals) for candidate in candidates.values()
        )
        self._emit_progress(
            "candidate_load_finish",
            {
                "active_current_rows_seen": active_rows,
                "fmp_delisted_rows_seen": delisted_rows,
                "candidate_count": len(candidates),
                "source_interval_count": source_interval_count,
            },
        )
        delisted_source = self._delisted_source_status(delisted_rows)
        lineage = record_data_lineage(
            self.session,
            provider="DERIVED",
            endpoint=DERIVED_ENDPOINT,
            asof_timestamp=self.run_timestamp,
            request_timestamp=self.run_timestamp,
            raw_payload={
                "replay_date": self.replay_date.isoformat(),
                "reconstruction_method": RECONSTRUCTION_METHOD,
                "active_current_rows": active_rows,
                "fmp_delisted_rows": delisted_rows,
                "candidate_count": len(candidates),
                "source_interval_count": sum(
                    len(candidate.intervals) for candidate in candidates.values()
                ),
                "delisted_source_complete": delisted_source["complete"],
                "delisted_source_partial_reason": delisted_source["partial_reason"],
            },
            source_authority="alpha_engine",
            data_quality_flags={
                "scratch_reconstruction": True,
                "market_cap_price_liquidity_filters": "not_applied_not_pit_safe",
                "delisted_source_complete": delisted_source["complete"],
                "delisted_source_partial_reason": delisted_source["partial_reason"],
                "allow_partial_delisted_source": self.allow_partial_delisted_source,
            },
            job_run_id=ctx.job_run_id,
        )

        metrics: dict[str, Any] = {
            "replay_date": self.replay_date.isoformat(),
            "active_current_rows_seen": active_rows,
            "fmp_delisted_rows_seen": delisted_rows,
            "candidate_count": len(candidates),
            "source_interval_count": source_interval_count,
            "delisted_source_complete": delisted_source["complete"],
            "delisted_source_partial_reason": delisted_source["partial_reason"],
            "delisted_source_latest_job_run_id": delisted_source["job_run_id"],
            "delisted_source_latest_run_status": delisted_source["run_status"],
            "allow_partial_delisted_source": self.allow_partial_delisted_source,
            "rows_inserted": 0,
            "rows_updated": 0,
            "rows_unchanged": 0,
            "rows_processed": 0,
            "rows_suppressed_pre_replay_delisted_exclusions": 0,
            "rows_suppressed_excluded_fmp_delisted": 0,
            "included_count": 0,
            "excluded_count": 0,
            "rejection_reason_counts": {},
            "source_counts": {},
            "source_interval_rejection_reason_counts": {},
        }
        output_hashes: list[str] = []
        row_mappings: list[dict[str, Any]] = []
        sorted_candidates = sorted(candidates.values(), key=lambda row: row.normalized_symbol)
        total_candidates = len(sorted_candidates)
        self._emit_progress(
            "interval_evaluation_start",
            {"candidate_count": total_candidates},
        )
        for index, candidate in enumerate(sorted_candidates, start=1):
            evaluated_intervals = [
                _evaluate_interval(interval, self.replay_date)
                for interval in candidate.intervals
            ]
            for evaluated in evaluated_intervals:
                if evaluated.inclusion_status != "included":
                    reason = evaluated.rejection_reason or "unknown"
                    counts = metrics["source_interval_rejection_reason_counts"]
                    counts[reason] = counts.get(reason, 0) + 1
            row_values = self._row_values(
                candidate,
                evaluated_intervals,
                lineage.data_lineage_id,
                ctx.job_run_id,
                delisted_source,
            )

            if row_values["inclusion_status"] == "included":
                metrics["included_count"] += 1
            else:
                metrics["excluded_count"] += 1
                reason = row_values["rejection_reason"] or "unknown"
                metrics["rejection_reason_counts"][reason] = (
                    metrics["rejection_reason_counts"].get(reason, 0) + 1
                )
            source = row_values["source"]
            metrics["source_counts"][source] = metrics["source_counts"].get(source, 0) + 1
            if self._should_persist_row(row_values):
                row_mappings.append(row_values)
                output_hashes.append(row_values["output_hash"])
            else:
                metrics["rows_suppressed_excluded_fmp_delisted"] += 1
                if (
                    row_values.get("rejection_reason")
                    == "delisted_on_or_before_replay_date"
                ):
                    metrics["rows_suppressed_pre_replay_delisted_exclusions"] += 1
            if index == total_candidates or index % self.progress_every == 0:
                self._emit_progress(
                    "interval_evaluation_progress",
                    {
                        "rows_processed": index,
                        "rows_total": total_candidates,
                        "included_count": metrics["included_count"],
                        "excluded_count": metrics["excluded_count"],
                    },
                )

        self._emit_progress(
            "interval_evaluation_finish",
            {
                "rows_processed": len(row_mappings),
                "included_count": metrics["included_count"],
                "excluded_count": metrics["excluded_count"],
            },
        )
        self._emit_progress("persistence_start", {"rows_total": len(row_mappings)})
        persistence_metrics = self._bulk_persist_rows(row_mappings)
        metrics.update(persistence_metrics)
        self._emit_progress("persistence_finish", persistence_metrics)
        metrics["progress_events"] = list(self._progress_events)
        errors: list[dict[str, Any]] = []
        status = "finished"
        if delisted_source["partial"] and not self.allow_partial_delisted_source:
            status = "partial_failed"
            errors.append(
                {
                    "stage": "delisted_source_completeness",
                    "error_type": "delisted_source_partial",
                    "message": (
                        "FMP delisted-company source is incomplete; rerun with "
                        "allow_partial_delisted_source only for bounded scratch probes."
                    ),
                    "partial_reason": delisted_source["partial_reason"],
                    "job_run_id": delisted_source["job_run_id"],
                }
            )
        return JobResult(
            status=status,
            metrics=metrics,
            input_hashes={
                "historical_universe_reconstruction_input": stable_hash(
                    {
                        "replay_date": self.replay_date.isoformat(),
                        "candidate_hashes": sorted(
                            stable_hash(
                                [
                                    interval.input_hash
                                    for interval in candidate.intervals
                                    if interval.input_hash
                                ]
                            )
                            for candidate in candidates.values()
                        ),
                        "delisted_source_complete": delisted_source["complete"],
                        "delisted_source_partial_reason": delisted_source["partial_reason"],
                    }
                )
            },
            output_hashes={
                "historical_universe_reconstruction_rows": stable_hash(output_hashes)
            },
            errors=errors,
        )

    def _load_active_current_candidates(
        self,
        candidates: dict[str, _Candidate],
    ) -> int:
        snapshots = self._current_active_snapshots()
        snapshot_symbols = [
            symbol
            for symbol in (_clean_symbol(snapshot.ticker) for snapshot in snapshots)
            if symbol
        ]
        profile_by_symbol = {
            row.symbol.upper(): row
            for row in (
                self.session.query(
                    SecurityProfile.symbol,
                    SecurityProfile.profile_payload_hash,
                    SecurityProfile.raw_profile_json,
                )
                .filter(SecurityProfile.symbol.in_(snapshot_symbols))
                .all()
                if snapshot_symbols
                else []
            )
        }
        for snapshot in snapshots:
            symbol = _clean_symbol(snapshot.ticker)
            if not symbol:
                continue
            profile = profile_by_symbol.get(symbol)
            profile_payload = _json_dict(getattr(profile, "raw_profile_json", None))
            exchange = _clean_symbol(
                snapshot.primary_exchange
                or profile_payload.get("exchange")
                or profile_payload.get("exchangeShortName")
            )
            candidate = candidates.setdefault(
                symbol,
                _Candidate(symbol=symbol, normalized_symbol=symbol),
            )
            source_row = {
                "source": "current_active_universe",
                "universe_snapshot_id": snapshot.universe_snapshot_id,
                "scan_id": snapshot.scan_id,
                "profile_payload_hash": getattr(profile, "profile_payload_hash", None),
                "source_lineage_hash": snapshot.source_lineage_hash,
                "security_type": snapshot.security_type,
            }
            input_hash = stable_hash(
                {
                    "source": "current_active_universe",
                    "universe_snapshot_id": snapshot.universe_snapshot_id,
                    "source_lineage_hash": snapshot.source_lineage_hash,
                    "profile_hash": getattr(profile, "profile_payload_hash", None),
                    "ipo_date": profile_payload.get("ipoDate")
                    or profile_payload.get("ipo_date"),
                    "exchange": exchange,
                }
            )
            candidate.intervals.append(
                _SourceInterval(
                    symbol=symbol,
                    normalized_symbol=symbol,
                    source_name="current_active_universe",
                    exchange=exchange,
                    company_name=_clean_string(
                        profile_payload.get("companyName") or profile_payload.get("name")
                    ),
                    ipo_date=_parse_optional_date(
                        profile_payload.get("ipoDate") or profile_payload.get("ipo_date")
                    ),
                    delisted_date=None,
                    current_universe_snapshot_id=snapshot.universe_snapshot_id,
                    data_lineage_id=None,
                    input_hash=input_hash,
                    source_row=source_row,
                    security_type=snapshot.security_type,
                )
            )
        return len(snapshots)

    def _load_fmp_delisted_candidates(
        self,
        candidates: dict[str, _Candidate],
    ) -> int:
        rows = (
            self.session.query(
                FmpDelistedCompanyRecord.fmp_delisted_company_id,
                FmpDelistedCompanyRecord.symbol,
                FmpDelistedCompanyRecord.normalized_symbol,
                FmpDelistedCompanyRecord.company_name,
                FmpDelistedCompanyRecord.exchange_key,
                FmpDelistedCompanyRecord.ipo_date,
                FmpDelistedCompanyRecord.delisted_date,
                FmpDelistedCompanyRecord.data_lineage_id,
                FmpDelistedCompanyRecord.raw_payload_hash,
                FmpDelistedCompanyRecord.raw_payload_json,
                FmpDelistedCompanyRecord.exchange_relevance_status,
            )
            .yield_per(1000)
            .all()
        )
        for row in rows:
            symbol = _clean_symbol(row.normalized_symbol or row.symbol)
            if not symbol:
                continue
            candidate = candidates.setdefault(
                symbol,
                _Candidate(symbol=symbol, normalized_symbol=symbol),
            )
            if row.data_lineage_id:
                data_lineage_id = row.data_lineage_id
            else:
                data_lineage_id = None
            raw_payload = _json_dict(row.raw_payload_json)
            source_row = {
                "source": "fmp_delisted_companies",
                "fmp_delisted_company_id": row.fmp_delisted_company_id,
                "data_lineage_id": row.data_lineage_id,
                "raw_payload_hash": row.raw_payload_hash,
                "exchange_relevance_status": row.exchange_relevance_status,
                "missing_delisted_date": row.delisted_date is None,
                "security_type": _delisted_security_type(raw_payload),
            }
            input_hash = stable_hash(
                {
                    "source": "fmp_delisted_companies",
                    "fmp_delisted_company_id": row.fmp_delisted_company_id,
                    "raw_payload_hash": row.raw_payload_hash,
                    "exchange": row.exchange_key,
                    "ipo_date": row.ipo_date,
                    "delisted_date": row.delisted_date,
                }
            )
            candidate.intervals.append(
                _SourceInterval(
                    symbol=symbol,
                    normalized_symbol=symbol,
                    source_name="fmp_delisted_companies",
                    exchange=_clean_symbol(row.exchange_key),
                    company_name=row.company_name,
                    ipo_date=row.ipo_date,
                    delisted_date=row.delisted_date,
                    fmp_delisted_company_id=row.fmp_delisted_company_id,
                    data_lineage_id=data_lineage_id,
                    input_hash=input_hash,
                    source_row=source_row,
                    missing_delisted_date_source=row.delisted_date is None,
                    security_type=_delisted_security_type(raw_payload),
                )
            )
        return len(rows)

    def _current_active_snapshots(self) -> list[Any]:
        canonical = (
            self.session.query(CanonicalUniverseScan)
            .outerjoin(UniverseScan, UniverseScan.scan_id == CanonicalUniverseScan.scan_id)
            .filter(
                or_(
                    UniverseScan.scan_id.is_(None),
                    UniverseScan.provider.is_(None),
                    UniverseScan.provider != HISTORICAL_REPLAY_SCAN_PROVIDER,
                ),
                or_(
                    CanonicalUniverseScan.selection_reason.is_(None),
                    CanonicalUniverseScan.selection_reason
                    != HISTORICAL_REPLAY_SCAN_SELECTION_REASON,
                ),
            )
            .order_by(CanonicalUniverseScan.trading_date.desc())
            .first()
        )
        query = self.session.query(
            UniverseSnapshot.universe_snapshot_id,
            UniverseSnapshot.scan_id,
            UniverseSnapshot.ticker,
            UniverseSnapshot.primary_exchange,
            UniverseSnapshot.source_lineage_hash,
            UniverseSnapshot.security_type,
        ).filter(
            UniverseSnapshot.operating_universe_inclusion.is_(True),
        )
        if canonical is not None:
            query = query.filter(UniverseSnapshot.scan_id == canonical.scan_id)
        return query.order_by(UniverseSnapshot.ticker).all()

    def _row_values(
        self,
        candidate: _Candidate,
        evaluated_intervals: list[_EvaluatedInterval],
        data_lineage_id: str,
        job_run_id: str,
        delisted_source: dict[str, Any],
    ) -> dict[str, Any]:
        selected = _select_replay_interval(evaluated_intervals)
        interval = selected.interval
        inclusion_status = selected.inclusion_status
        rejection_reason = selected.rejection_reason
        source_names = {evaluated.interval.source_name for evaluated in evaluated_intervals}
        lineage_ids = {
            evaluated.interval.data_lineage_id
            for evaluated in evaluated_intervals
            if evaluated.interval.data_lineage_id
        }
        source = "+".join(sorted(source_names))
        interval_payloads = [
            _interval_provenance_payload(evaluated)
            for evaluated in sorted(
                evaluated_intervals,
                key=lambda item: (
                    item.interval.source_name,
                    item.interval.ipo_date or date.min,
                    item.interval.delisted_date or date.max,
                    item.interval.fmp_delisted_company_id or "",
                    item.interval.current_universe_snapshot_id or "",
                ),
            )
        ]
        source_row_hashes = sorted(
            stable_hash(evaluated.interval.source_row)
            for evaluated in evaluated_intervals
        )
        missing_delisted_date_source = any(
            evaluated.interval.missing_delisted_date_source
            for evaluated in evaluated_intervals
        )
        if self.compact_persisted_provenance:
            provenance = {
                "provenance_payload_policy": "compact_public_cohort_row_v4",
                "source_interval_count": len(interval_payloads),
                "source_row_hash": stable_hash(source_row_hashes),
                "selected_interval_input_hash": interval.input_hash,
                "selected_interval_lineage_id": interval.data_lineage_id,
                "missing_delisted_date_source": missing_delisted_date_source,
                "delisted_source_complete": delisted_source["complete"],
                "delisted_source_partial_reason": delisted_source["partial_reason"],
                "allow_partial_delisted_source": self.allow_partial_delisted_source,
            }
            pit_filter_status = {
                "pit_filter_payload_policy": "compact_public_cohort_row_v3",
                "applied": "exchange,ipo,delisted,security,suffix",
                "not_pit": "mcap,price,liquidity",
                "delisted_source_complete": delisted_source["complete"],
                "delisted_source_partial_reason": delisted_source["partial_reason"],
            }
        else:
            provenance = {
                "provenance_payload_policy": (
                    "compact_row_interval_summary_full_run_facts_in_lineage"
                ),
                "source_intervals": interval_payloads,
                "selected_source_interval": _interval_identity(interval),
                "source_interval_count": len(interval_payloads),
                "source_row_hashes": source_row_hashes,
                "missing_delisted_date_source": missing_delisted_date_source,
                "lineage_ids": sorted(lineage_ids),
                "delisted_source_complete": delisted_source["complete"],
                "delisted_source_partial_reason": delisted_source["partial_reason"],
                "allow_partial_delisted_source": self.allow_partial_delisted_source,
            }
            pit_filter_status = {
                "exchange_filter": "applied",
                "ipo_date_filter": "applied",
                "delisted_date_filter": "applied",
                "delisted_security_type_filter": "applied",
                "delisted_symbol_suffix_filter": "applied",
                "market_cap_filter": "not_applied_not_pit_safe",
                "price_filter": "not_applied_not_pit_safe",
                "liquidity_filter": "not_applied_not_pit_safe",
                "delisted_source_complete": delisted_source["complete"],
                "delisted_source_partial_reason": delisted_source["partial_reason"],
            }
        input_hash = stable_hash(
            {
                "replay_date": self.replay_date.isoformat(),
                "normalized_symbol": candidate.normalized_symbol,
                "candidate_inputs": sorted(
                    interval.input_hash
                    for interval in candidate.intervals
                    if interval.input_hash
                ),
                "delisted_source_complete": delisted_source["complete"],
                "delisted_source_partial_reason": delisted_source["partial_reason"],
            }
        )
        output_payload = {
            "replay_date": self.replay_date.isoformat(),
            "ticker": interval.symbol,
            "exchange": interval.exchange,
            "ipo_date": interval.ipo_date,
            "delisted_date": interval.delisted_date,
            "inclusion_status": inclusion_status,
            "rejection_reason": rejection_reason,
            "source": source,
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "selected_interval": _interval_identity(interval),
        }
        return {
            "replay_date": self.replay_date,
            "ticker": interval.symbol,
            "normalized_symbol": candidate.normalized_symbol,
            "exchange": interval.exchange,
            "company_name": interval.company_name,
            "ipo_date": interval.ipo_date,
            "delisted_date": interval.delisted_date,
            "inclusion_status": inclusion_status,
            "rejection_reason": rejection_reason,
            "source": source,
            "source_provenance_json": json.dumps(provenance, sort_keys=True, default=str),
            "reconstructed": True,
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "pit_filter_status_json": json.dumps(pit_filter_status, sort_keys=True),
            "current_universe_snapshot_id": interval.current_universe_snapshot_id,
            "fmp_delisted_company_id": interval.fmp_delisted_company_id,
            "data_lineage_id": data_lineage_id,
            "job_run_id": job_run_id,
            "input_hash": input_hash,
            "output_hash": stable_hash(output_payload),
        }

    def _delisted_source_status(self, delisted_rows: int) -> dict[str, Any]:
        run = (
            self.session.query(EvidenceJobRun)
            .join(EvidenceJob, EvidenceJob.job_id == EvidenceJobRun.job_id)
            .filter(EvidenceJob.job_name == FMP_DELISTED_JOB_NAME)
            .filter(EvidenceJobRun.run_status == "finished")
            .order_by(EvidenceJobRun.started_at.desc(), EvidenceJobRun.ended_at.desc())
            .first()
        )
        if run is None:
            partial = bool(delisted_rows)
            return {
                "complete": not partial,
                "partial": partial,
                "partial_reason": (
                    "fmp_delisted_ingestion_run_not_found" if delisted_rows else None
                ),
                "job_run_id": None,
                "run_status": None,
                "max_pages_reached": None,
            }
        metrics = _json_dict(run.metric_json)
        max_pages_reached = bool(metrics.get("max_pages_reached"))
        partial_reason = None
        partial = False
        if max_pages_reached:
            partial = True
            partial_reason = "max_pages_reached"
        return {
            "complete": not partial,
            "partial": partial,
            "partial_reason": partial_reason,
            "job_run_id": run.job_run_id,
            "run_status": run.run_status,
            "max_pages_reached": max_pages_reached,
        }

    def _should_persist_row(self, row_values: dict[str, Any]) -> bool:
        if self.persist_pre_replay_delisted_exclusions:
            return True
        return not (
            row_values.get("fmp_delisted_company_id") is not None
            and row_values.get("current_universe_snapshot_id") is None
            and row_values.get("inclusion_status") == "excluded"
        )

    def _bulk_persist_rows(self, row_mappings: list[dict[str, Any]]) -> dict[str, int]:
        return bulk_persist_historical_universe_reconstructions(
            self.session,
            row_mappings,
            progress_callback=self._emit_progress,
            persistence_batch_size=self.persistence_batch_size,
        )

    def _emit_progress(self, event: str, payload: dict[str, Any]) -> None:
        event_payload = {
            "event": event,
            "replay_date": self.replay_date.isoformat(),
            "elapsed_seconds": round(perf_counter() - self._started_perf, 6),
            **payload,
        }
        self._progress_events.append(event_payload)
        if self.progress_callback is not None:
            try:
                self.progress_callback(event, event_payload)
            except Exception:
                pass



def _row_mapping_matches_existing(
    existing: HistoricalUniverseReconstruction,
    mapping: dict[str, Any],
) -> bool:
    ignored = {
        "historical_universe_reconstruction_id",
        "created_at",
        "updated_at",
    }
    for key, value in mapping.items():
        if key in ignored:
            continue
        if getattr(existing, key) != value:
            return False
    return True


def bulk_persist_historical_universe_reconstructions(
    session: Session,
    row_mappings: list[dict[str, Any]],
    *,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    persistence_batch_size: int = 1000,
) -> dict[str, int]:
    """Persist historical universe rows across one or more replay dates.

    The conflict key is the table's natural replay identity:
    ``(replay_date, normalized_symbol)``. This keeps date-range replay from
    paying the remote write cost once per date.
    """

    payload_metrics = _serialized_payload_metrics(row_mappings)
    if not row_mappings:
        return {
            "rows_processed": 0,
            "rows_inserted": 0,
            "rows_updated": 0,
            "rows_unchanged": 0,
            **payload_metrics,
        }
    if session.bind is not None and session.bind.dialect.name == "postgresql":
        return _postgres_stage_merge_historical_universe_rows(
            session,
            row_mappings=row_mappings,
            payload_metrics=payload_metrics,
            progress_callback=progress_callback,
        )

    keys = sorted(
        {
            (row["replay_date"], row["normalized_symbol"])
            for row in row_mappings
        }
    )
    existing_rows = (
        session.query(HistoricalUniverseReconstruction)
        .filter(
            tuple_(
                HistoricalUniverseReconstruction.replay_date,
                HistoricalUniverseReconstruction.normalized_symbol,
            ).in_(keys)
        )
        .all()
        if keys
        else []
    )
    existing_by_key = {
        (row.replay_date, row.normalized_symbol): row
        for row in existing_rows
    }
    now = datetime.now(timezone.utc)
    insert_mappings: list[dict[str, Any]] = []
    update_mappings: list[dict[str, Any]] = []
    unchanged = 0
    for row_values in row_mappings:
        key = (row_values["replay_date"], row_values["normalized_symbol"])
        existing = existing_by_key.get(key)
        if existing is None:
            insert_mappings.append(
                {
                    **row_values,
                    "historical_universe_reconstruction_id": str(uuid4()),
                    "created_at": now,
                    "updated_at": now,
                }
            )
            continue
        update_mapping = {
            **row_values,
            "historical_universe_reconstruction_id": (
                existing.historical_universe_reconstruction_id
            ),
            "created_at": existing.created_at,
            "updated_at": now,
        }
        if _row_mapping_matches_existing(existing, update_mapping):
            unchanged += 1
        else:
            update_mappings.append(update_mapping)

    processed = 0
    total_write_rows = len(insert_mappings) + len(update_mappings)
    insert_stmt = _historical_reconstruction_insert_stmt()
    update_stmt = _historical_reconstruction_update_stmt()
    batch_size = max(int(persistence_batch_size), 1)
    for chunk in _chunks(insert_mappings, batch_size):
        session.execute(insert_stmt, chunk)
        session.flush()
        processed += len(chunk)
        _safe_progress(
            progress_callback,
            "persistence_progress",
            {
                "operation": "insert",
                "rows_processed": processed,
                "rows_total": total_write_rows,
                "rows_inserted": processed,
                "rows_updated": 0,
            },
        )
    updated_processed = 0
    for chunk in _chunks(update_mappings, batch_size):
        session.execute(update_stmt, chunk)
        session.flush()
        updated_processed += len(chunk)
        processed += len(chunk)
        _safe_progress(
            progress_callback,
            "persistence_progress",
            {
                "operation": "update",
                "rows_processed": processed,
                "rows_total": total_write_rows,
                "rows_inserted": len(insert_mappings),
                "rows_updated": updated_processed,
            },
        )
    return {
        "rows_processed": len(row_mappings),
        "rows_inserted": len(insert_mappings),
        "rows_updated": len(update_mappings),
        "rows_unchanged": unchanged,
        **payload_metrics,
    }


def _postgres_stage_merge_historical_universe_rows(
    session: Session,
    *,
    row_mappings: list[dict[str, Any]],
    payload_metrics: dict[str, int],
    progress_callback: Callable[[str, dict[str, Any]], None] | None,
) -> dict[str, int]:
    now = datetime.now(timezone.utc)
    write_mappings = [
        {
            **row,
            "historical_universe_reconstruction_id": str(uuid4()),
            "created_at": now,
            "updated_at": now,
        }
        for row in row_mappings
    ]
    inserted = 0
    updated = 0
    unchanged = 0
    if write_mappings:
        stage_table = f"tmp_hur_{uuid4().hex}"
        stage_started = perf_counter()
        _safe_progress(
            progress_callback,
            "copy_stage_start",
            {
                "rows_total": len(write_mappings),
                **payload_metrics,
            },
        )
        session.execute(text("SET LOCAL statement_timeout = 0"))
        session.execute(
            text(
                f"CREATE TEMP TABLE {stage_table} "
                "(LIKE historical_universe_reconstructions INCLUDING DEFAULTS) "
                "ON COMMIT DROP"
            )
        )
        stage_load_method = _copy_historical_reconstruction_stage_rows(
            session,
            stage_table,
            write_mappings,
            progress_callback=progress_callback,
        )
        stage_elapsed = perf_counter() - stage_started
        stage_payload = {
            "stage_load_method": stage_load_method,
            "rows_processed": len(write_mappings),
            "rows_total": len(write_mappings),
            "elapsed_seconds": round(stage_elapsed, 6),
        }
        _safe_progress(progress_callback, "copy_stage_progress", stage_payload)
        _safe_progress(
            progress_callback,
            "persistence_progress",
            {"operation": "postgres_stage_insert", **stage_payload},
        )
        _safe_progress(
            progress_callback,
            "copy_stage_finish",
            {**stage_payload, **payload_metrics},
        )
        merge_started = perf_counter()
        inserted = int(
            session.execute(
                text(
                    f"SELECT COUNT(*) FROM {stage_table} s "
                    "LEFT JOIN historical_universe_reconstructions t "
                    "ON t.replay_date = s.replay_date "
                    "AND t.normalized_symbol = s.normalized_symbol "
                    "WHERE t.historical_universe_reconstruction_id IS NULL"
                )
            ).scalar()
            or 0
        )
        changed_predicate = _historical_reconstruction_changed_predicate("t", "s")
        updated = int(
            session.execute(
                text(
                    f"SELECT COUNT(*) FROM {stage_table} s "
                    "JOIN historical_universe_reconstructions t "
                    "ON t.replay_date = s.replay_date "
                    "AND t.normalized_symbol = s.normalized_symbol "
                    f"WHERE {changed_predicate}"
                )
            ).scalar()
            or 0
        )
        unchanged = len(write_mappings) - inserted - updated
        merge_start_payload = {
            "rows_processed": len(write_mappings),
            "rows_total": len(write_mappings),
            "rows_inserted": inserted,
            "rows_updated": updated,
            "rows_unchanged": unchanged,
        }
        _safe_progress(progress_callback, "merge_upsert_start", merge_start_payload)
        _safe_progress(
            progress_callback,
            "persistence_progress",
            {"operation": "postgres_stage_merge_start", **merge_start_payload},
        )
        session.execute(_historical_reconstruction_stage_insert_new_stmt(stage_table))
        if updated:
            session.execute(
                _historical_reconstruction_stage_update_changed_stmt(stage_table)
            )
        session.flush()
        merge_elapsed = perf_counter() - merge_started
        merge_finish_payload = {
            **merge_start_payload,
            "elapsed_seconds": round(merge_elapsed, 6),
        }
        _safe_progress(progress_callback, "merge_upsert_finish", merge_finish_payload)
        _safe_progress(
            progress_callback,
            "persistence_progress",
            {"operation": "postgres_stage_merge_finish", **merge_finish_payload},
        )
    return {
        "rows_processed": len(row_mappings),
        "rows_inserted": inserted,
        "rows_updated": updated,
        "rows_unchanged": unchanged,
        **payload_metrics,
    }


def _safe_progress(
    callback: Callable[[str, dict[str, Any]], None] | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if callback is None:
        return
    try:
        callback(event, payload)
    except Exception:
        pass


def _copy_historical_reconstruction_stage_rows(
    session: Session,
    stage_table: str,
    rows: list[dict[str, Any]],
    *,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> str:
    """Load HUR rows into a PostgreSQL temp stage table with psycopg COPY."""

    columns = ", ".join(_INSERT_COLUMNS)
    copy_sql = f"COPY {stage_table} ({columns}) FROM STDIN"
    connection = session.connection()
    driver_connection = getattr(
        getattr(connection, "connection", None),
        "driver_connection",
        None,
    )
    if driver_connection is None:
        stage_model = _historical_reconstruction_stage_table(stage_table)
        session.execute(stage_model.insert(), rows)
        return "sqlalchemy_insertmanyvalues_temp_insert"
    try:
        with driver_connection.cursor() as cursor:
            with cursor.copy(copy_sql) as copy:
                for index, row in enumerate(rows, start=1):
                    copy.write_row(
                        [_copy_row_scalar(row.get(column)) for column in _INSERT_COLUMNS]
                    )
                    if index == len(rows) or index % _COPY_ROW_PROGRESS_EVERY == 0:
                        _safe_progress(
                            progress_callback,
                            "copy_stage_row_progress",
                            {
                                "rows_processed": index,
                                "rows_total": len(rows),
                            },
                        )
        return "psycopg_copy_write_row"
    except AttributeError:
        stage_model = _historical_reconstruction_stage_table(stage_table)
        session.execute(stage_model.insert(), rows)
        return "sqlalchemy_insertmanyvalues_temp_insert"


def _copy_csv_payload(rows: list[dict[str, Any]]) -> str:
    output = StringIO()
    writer = csv.writer(
        output,
        delimiter="\t",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    for row in rows:
        writer.writerow([_copy_scalar(row.get(column)) for column in _INSERT_COLUMNS])
    return output.getvalue()


def _copy_scalar(value: Any) -> str:
    if value is None:
        return _COPY_NULL_SENTINEL
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value).replace("\x00", "")


def _copy_row_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\x00", "")
    return value


def _chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index:index + size] for index in range(0, len(rows), size)]


def _historical_reconstruction_stage_table(stage_name: str) -> Table:
    metadata = MetaData()
    return Table(
        stage_name,
        metadata,
        *[
            Column(column.name, column.type)
            for column in HistoricalUniverseReconstruction.__table__.columns
        ],
    )


def _serialized_payload_metrics(rows: list[dict[str, Any]]) -> dict[str, int]:
    total_bytes = 0
    max_row_bytes = 0
    source_provenance_bytes = 0
    pit_filter_status_bytes = 0
    hash_bytes = 0
    identifier_bytes = 0
    for row in rows:
        row_bytes = 0
        for column in _INSERT_COLUMNS:
            value = row.get(column)
            encoded = _metric_bytes(value)
            row_bytes += encoded
            if column == "source_provenance_json":
                source_provenance_bytes += encoded
            elif column == "pit_filter_status_json":
                pit_filter_status_bytes += encoded
            elif column in {"input_hash", "output_hash"}:
                hash_bytes += encoded
            elif column in {
                "historical_universe_reconstruction_id",
                "ticker",
                "normalized_symbol",
                "current_universe_snapshot_id",
                "fmp_delisted_company_id",
                "data_lineage_id",
                "job_run_id",
            }:
                identifier_bytes += encoded
        total_bytes += row_bytes
        max_row_bytes = max(max_row_bytes, row_bytes)
    return {
        "total_serialized_payload_bytes": total_bytes,
        "max_row_serialized_payload_bytes": max_row_bytes,
        "source_provenance_json_bytes": source_provenance_bytes,
        "pit_filter_status_json_bytes": pit_filter_status_bytes,
        "input_output_hash_bytes": hash_bytes,
        "identifier_bytes": identifier_bytes,
    }


def _metric_bytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, (date, datetime)):
        return len(value.isoformat().encode())
    return len(str(value).encode())


_INSERT_COLUMNS = (
    "historical_universe_reconstruction_id",
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
    "reconstruction_method",
    "pit_filter_status_json",
    "current_universe_snapshot_id",
    "fmp_delisted_company_id",
    "data_lineage_id",
    "job_run_id",
    "input_hash",
    "output_hash",
    "created_at",
    "updated_at",
)

_COPY_NULL_SENTINEL = "__ALPHA_COPY_NULL__"
_COPY_WRITE_CHUNK_SIZE = 250
_COPY_ROW_PROGRESS_EVERY = 100

_UPDATE_COLUMNS = tuple(
    column
    for column in _INSERT_COLUMNS
    if column not in {"historical_universe_reconstruction_id", "created_at"}
)


def _historical_reconstruction_insert_stmt(
    table_name: str = "historical_universe_reconstructions",
):
    columns = ", ".join(_INSERT_COLUMNS)
    values = ", ".join(f":{column}" for column in _INSERT_COLUMNS)
    return text(
        f"INSERT INTO {table_name} ({columns}) "
        f"VALUES ({values})"
    )


def _historical_reconstruction_update_stmt():
    assignments = ", ".join(
        f"{column} = :{column}"
        for column in _UPDATE_COLUMNS
    )
    return text(
        "UPDATE historical_universe_reconstructions "
        f"SET {assignments} "
        "WHERE historical_universe_reconstruction_id = "
        ":historical_universe_reconstruction_id"
    )


def _historical_reconstruction_stage_merge_stmt(stage_table: str):
    columns = ", ".join(_INSERT_COLUMNS)
    update_assignments = ", ".join(
        f"{column} = EXCLUDED.{column}"
        for column in _UPDATE_COLUMNS
        if column != "created_at"
    )
    return text(
        f"INSERT INTO historical_universe_reconstructions ({columns}) "
        f"SELECT {columns} FROM {stage_table} "
        "ON CONFLICT (replay_date, normalized_symbol) DO UPDATE SET "
        f"{update_assignments}"
    )


def _historical_reconstruction_stage_insert_new_stmt(stage_table: str):
    columns = ", ".join(_INSERT_COLUMNS)
    return text(
        f"INSERT INTO historical_universe_reconstructions ({columns}) "
        f"SELECT {columns} FROM {stage_table} "
        "ON CONFLICT (replay_date, normalized_symbol) DO NOTHING"
    )


def _historical_reconstruction_stage_update_changed_stmt(stage_table: str):
    assignments = ", ".join(
        f"{column} = s.{column}"
        for column in _UPDATE_COLUMNS
    )
    changed_predicate = _historical_reconstruction_changed_predicate("t", "s")
    return text(
        "UPDATE historical_universe_reconstructions t "
        f"SET {assignments} "
        f"FROM {stage_table} s "
        "WHERE t.replay_date = s.replay_date "
        "AND t.normalized_symbol = s.normalized_symbol "
        f"AND ({changed_predicate})"
    )


def _historical_reconstruction_changed_predicate(
    target_alias: str,
    stage_alias: str,
) -> str:
    compared_columns = [
        column
        for column in _UPDATE_COLUMNS
        if column not in {
            "updated_at",
            "job_run_id",
            "data_lineage_id",
        }
    ]
    return " OR ".join(
        f"{target_alias}.{column} IS DISTINCT FROM {stage_alias}.{column}"
        for column in compared_columns
    )


def _evaluate_interval(interval: _SourceInterval, replay_date: date) -> _EvaluatedInterval:
    if interval.source_name == "fmp_delisted_companies":
        excluded, reason = _is_non_common_symbol(interval.symbol)
        if excluded:
            return _EvaluatedInterval(interval, "excluded", reason)
        if _is_non_common_delisted_security_type(interval.security_type):
            return _EvaluatedInterval(
                interval,
                "excluded",
                "security_type_non_common_delisted",
            )

    exchange = _clean_symbol(interval.exchange)
    if not exchange:
        return _EvaluatedInterval(interval, "excluded", "exchange_missing")
    if exchange not in ALLOWED_OPERATING_EXCHANGES:
        return _EvaluatedInterval(interval, "excluded", "exchange_not_operating_universe")
    if interval.ipo_date is None:
        return _EvaluatedInterval(interval, "excluded", "missing_ipo_date")
    if interval.ipo_date > replay_date:
        return _EvaluatedInterval(interval, "excluded", "ipo_after_replay_date")
    if interval.delisted_date is not None and interval.delisted_date <= replay_date:
        return _EvaluatedInterval(
            interval, "excluded", "delisted_on_or_before_replay_date"
        )
    return _EvaluatedInterval(interval, "included", None)


def _select_replay_interval(evaluated_intervals: list[_EvaluatedInterval]) -> _EvaluatedInterval:
    included = [
        evaluated
        for evaluated in evaluated_intervals
        if evaluated.inclusion_status == "included"
    ]
    if included:
        return sorted(included, key=_interval_selection_key, reverse=True)[0]
    return sorted(evaluated_intervals, key=_excluded_interval_selection_key)[0]


def _interval_selection_key(evaluated: _EvaluatedInterval) -> tuple[Any, ...]:
    interval = evaluated.interval
    return (
        interval.ipo_date or date.min,
        interval.delisted_date or date.max,
        1 if interval.source_name == "current_active_universe" else 0,
        interval.symbol,
    )


def _excluded_interval_selection_key(evaluated: _EvaluatedInterval) -> tuple[Any, ...]:
    reason_order = {
        "non_common_symbol_suffix": 0,
        "non_common_symbol_separator": 1,
        "security_type_non_common_delisted": 2,
        "exchange_missing": 3,
        "exchange_not_operating_universe": 4,
        "missing_ipo_date": 5,
        "ipo_after_replay_date": 6,
        "delisted_on_or_before_replay_date": 7,
    }
    interval = evaluated.interval
    return (
        reason_order.get(evaluated.rejection_reason or "unknown", 99),
        interval.ipo_date or date.max,
        interval.delisted_date or date.max,
        interval.source_name,
        interval.symbol,
    )


def _interval_identity(interval: _SourceInterval) -> dict[str, Any]:
    return {
        "source": interval.source_name,
        "symbol": interval.symbol,
        "normalized_symbol": interval.normalized_symbol,
        "ipo_date": interval.ipo_date,
        "delisted_date": interval.delisted_date,
        "current_universe_snapshot_id": interval.current_universe_snapshot_id,
        "fmp_delisted_company_id": interval.fmp_delisted_company_id,
    }


def _interval_provenance_payload(evaluated: _EvaluatedInterval) -> dict[str, Any]:
    interval = evaluated.interval
    return {
        **_interval_identity(interval),
        "exchange": interval.exchange,
        "company_name": interval.company_name,
        "security_type": interval.security_type,
        "inclusion_status": evaluated.inclusion_status,
        "rejection_reason": evaluated.rejection_reason,
        "data_lineage_id": interval.data_lineage_id,
        "input_hash": interval.input_hash,
        "missing_delisted_date_source": interval.missing_delisted_date_source,
    }


def _delisted_security_type(raw_payload: dict[str, Any]) -> str | None:
    for key in (
        "securityType",
        "security_type",
        "assetType",
        "asset_type",
        "type",
        "instrumentType",
        "instrument_type",
    ):
        value = _clean_string(raw_payload.get(key))
        if value:
            return value.strip().casefold().replace(" ", "_").replace("-", "_")
    return None


def _is_non_common_delisted_security_type(value: str | None) -> bool:
    normalized = _clean_string(value)
    if not normalized:
        return False
    security_type = normalized.casefold().replace(" ", "_").replace("-", "_")
    if security_type in NON_COMMON_TYPES:
        return True
    non_common_markers = (
        "preferred",
        "warrant",
        "unit",
        "right",
        "etf",
        "fund",
        "adr",
        "spac",
        "blank_check",
        "business_development",
    )
    return any(marker in security_type for marker in non_common_markers)


def _parse_optional_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        return date.fromisoformat(cleaned[:10])
    return None


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


def _clean_symbol(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().upper()
    return cleaned or None


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _aware_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
