"""Historical PIT universe reconstruction for replay research.

This job answers whether a ticker was eligible for the operating universe on a
historical replay date using current active rows plus FMP delisted-company
directory rows. It writes only to the separate reconstruction table and never
mutates live canonical universe snapshots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from alpha.data.contracts import stable_hash
from alpha.db.models import (
    CanonicalUniverseScan,
    EvidenceJob,
    EvidenceJobRun,
    FmpDelistedCompanyRecord,
    HistoricalUniverseReconstruction,
    SecurityProfile,
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
    ) -> None:
        if replay_date < date(2024, 1, 1):
            raise ValueError("historical replay reconstruction starts at 2024-01-01")
        self.session = session
        self.replay_date = replay_date
        self.run_timestamp = _aware_utc(run_timestamp)
        self.allow_partial_delisted_source = allow_partial_delisted_source

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
        active_rows = self._load_active_current_candidates(candidates)
        delisted_rows = self._load_fmp_delisted_candidates(candidates)
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

        source_interval_count = sum(
            len(candidate.intervals) for candidate in candidates.values()
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
            "included_count": 0,
            "excluded_count": 0,
            "rejection_reason_counts": {},
            "source_counts": {},
            "source_interval_rejection_reason_counts": {},
        }
        output_hashes: list[str] = []
        for candidate in sorted(candidates.values(), key=lambda row: row.normalized_symbol):
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
            output_hashes.append(row_values["output_hash"])
            existing = (
                self.session.query(HistoricalUniverseReconstruction)
                .filter(
                    HistoricalUniverseReconstruction.replay_date == self.replay_date,
                    HistoricalUniverseReconstruction.normalized_symbol
                    == candidate.normalized_symbol,
                )
                .one_or_none()
            )
            if existing is None:
                self.session.add(HistoricalUniverseReconstruction(**row_values))
                metrics["rows_inserted"] += 1
            else:
                for key, value in row_values.items():
                    setattr(existing, key, value)
                metrics["rows_updated"] += 1

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

        self.session.flush()
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
        profile_by_symbol = {
            profile.symbol.upper(): profile
            for profile in self.session.query(SecurityProfile).all()
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
        rows = self.session.query(FmpDelistedCompanyRecord).all()
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

    def _current_active_snapshots(self) -> list[UniverseSnapshot]:
        canonical = (
            self.session.query(CanonicalUniverseScan)
            .order_by(CanonicalUniverseScan.trading_date.desc())
            .first()
        )
        query = self.session.query(UniverseSnapshot).filter(
            UniverseSnapshot.operating_universe_inclusion.is_(True)
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
        provenance = {
            "source_rows": [payload["source_row"] for payload in interval_payloads],
            "source_intervals": interval_payloads,
            "selected_source_interval": _interval_identity(interval),
            "missing_delisted_date_source": any(
                evaluated.interval.missing_delisted_date_source
                for evaluated in evaluated_intervals
            ),
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
            .order_by(EvidenceJobRun.started_at.desc(), EvidenceJobRun.ended_at.desc())
            .first()
        )
        if run is None:
            return {
                "complete": None if delisted_rows else True,
                "partial": False,
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
        elif run.run_status != "finished":
            partial = True
            partial_reason = f"ingestion_run_status:{run.run_status}"
        return {
            "complete": not partial,
            "partial": partial,
            "partial_reason": partial_reason,
            "job_run_id": run.job_run_id,
            "run_status": run.run_status,
            "max_pages_reached": max_pages_reached,
        }


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
        "source_row": interval.source_row,
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
