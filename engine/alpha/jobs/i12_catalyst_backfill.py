"""In-place catalyst retag backfill for existing I12 corpus rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy.orm import Session

from alpha.data.contracts import stable_hash
from alpha.db.models import FeatureSnapshot, IntradayEventDetail, SignalRegistry
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.i12_catalysts import (
    CATALYST_STATUS_IMPLEMENTED,
    I12CatalystResolver,
    apply_i12_catalyst_result_to_feature_payload,
    empty_catalyst_result,
)
from alpha.market_calendar import us_equity_session_open_timestamp


JOB_NAME = "i12_catalyst_retag_backfill"
I12_PATTERN_ID = "I12"


@dataclass
class _BackfillCounters:
    rows_seen: int = 0
    rows_updated: int = 0
    rows_skipped_existing: int = 0
    rows_missing_feature_json: int = 0
    feature_snapshots_updated: int = 0
    catalyst_tagged_rows: int = 0
    source_error_rows: int = 0


class I12CatalystBackfillJob(BaseJob):
    """Retag existing I12 intraday rows without rebuilding the corpus."""

    def __init__(
        self,
        *,
        session: Session,
        catalyst_resolver: I12CatalystResolver,
        start_date: date | None = None,
        end_date: date | None = None,
        skip_existing: bool = False,
        batch_size: int = 500,
        progress_artifact: str | Path | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if start_date is not None and end_date is not None and end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        self._session = session
        self._resolver = catalyst_resolver
        self._start_date = start_date
        self._end_date = end_date
        self._skip_existing = skip_existing
        self._batch_size = int(batch_size)
        self._progress_artifact = Path(progress_artifact) if progress_artifact else None
        self._latest_metrics: dict[str, Any] = {}

    @property
    def job_name(self) -> str:
        return JOB_NAME

    @property
    def job_type(self) -> str:
        return "feature_enrichment"

    @property
    def partial_metrics(self) -> dict[str, Any]:
        return dict(self._latest_metrics)

    def run(self, ctx: JobContext) -> JobResult:
        counters = _BackfillCounters()
        artifact = {
            "job": JOB_NAME,
            "start_date": self._start_date.isoformat() if self._start_date else None,
            "end_date": self._end_date.isoformat() if self._end_date else None,
            "skip_existing": self._skip_existing,
            "events": [],
            "summary": {},
        }
        self._write_artifact(artifact)
        rows = self._query_rows()
        for index, detail in enumerate(rows, start=1):
            counters.rows_seen += 1
            self._retag_detail(detail, counters=counters)
            if index % self._batch_size == 0:
                self._session.flush()
                self._progress(artifact, "batch_finish", counters)
        self._session.flush()
        self._progress(artifact, "finish", counters)
        return JobResult(status="finished", metrics=self._metrics(counters), errors=[])

    def _query_rows(self) -> list[IntradayEventDetail]:
        query = (
            self._session.query(IntradayEventDetail)
            .filter(IntradayEventDetail.pattern_id == I12_PATTERN_ID)
            .order_by(IntradayEventDetail.trading_date, IntradayEventDetail.ticker)
        )
        if self._start_date is not None:
            query = query.filter(IntradayEventDetail.trading_date >= self._start_date)
        if self._end_date is not None:
            query = query.filter(IntradayEventDetail.trading_date <= self._end_date)
        return list(query.all())

    def _retag_detail(
        self,
        detail: IntradayEventDetail,
        *,
        counters: _BackfillCounters,
    ) -> None:
        payload = _json_object(detail.feature_json)
        if not payload:
            counters.rows_missing_feature_json += 1
            return
        if (
            self._skip_existing
            and payload.get("catalyst_source_status") == CATALYST_STATUS_IMPLEMENTED
            and not payload.get("catalyst_source_errors")
        ):
            counters.rows_skipped_existing += 1
            return
        cutoff = _detail_cutoff(detail)
        if cutoff is None:
            catalyst = empty_catalyst_result(cutoff_timestamp=None)
        else:
            catalyst = self._resolver.resolve(
                ticker=detail.ticker,
                trading_date=detail.trading_date,
                cutoff_timestamp=cutoff,
            )
        updated = apply_i12_catalyst_result_to_feature_payload(payload, catalyst)
        detail.feature_json = json.dumps(updated, sort_keys=True, default=str)
        detail.output_hash = stable_hash({
            "outcome": detail.outcome,
            "features": updated,
            "labels": _json_object(detail.label_json),
            "artifact_flags": _json_object(detail.artifact_flags_json),
            "signal_identity_hash": _signal_identity_hash(self._session, detail.signal_id),
        })
        counters.rows_updated += 1
        counters.catalyst_tagged_rows += int(bool(updated.get("catalyst_tags")))
        counters.source_error_rows += int(bool(updated.get("catalyst_source_errors")))
        if detail.signal_id:
            counters.feature_snapshots_updated += self._update_feature_snapshot(detail.signal_id, updated)

    def _update_feature_snapshot(self, signal_id: str, updated: Mapping[str, Any]) -> int:
        signal = self._session.get(SignalRegistry, signal_id)
        if signal is None:
            return 0
        snap = self._session.get(FeatureSnapshot, signal.feature_snapshot_id)
        if snap is None:
            return 0
        snap.feature_json = json.dumps(dict(updated), sort_keys=True, default=str)
        snap.feature_hash = stable_hash(updated)
        snap.output_hash = stable_hash(updated)
        return 1

    def _metrics(self, counters: _BackfillCounters) -> dict[str, Any]:
        metrics = {
            "pattern_id": I12_PATTERN_ID,
            "start_date": self._start_date.isoformat() if self._start_date else None,
            "end_date": self._end_date.isoformat() if self._end_date else None,
            "skip_existing": self._skip_existing,
            "batch_size": self._batch_size,
            "rows_seen": counters.rows_seen,
            "rows_updated": counters.rows_updated,
            "rows_skipped_existing": counters.rows_skipped_existing,
            "rows_missing_feature_json": counters.rows_missing_feature_json,
            "feature_snapshots_updated": counters.feature_snapshots_updated,
            "catalyst_tagged_rows": counters.catalyst_tagged_rows,
            "source_error_rows": counters.source_error_rows,
        }
        self._latest_metrics = metrics
        return metrics

    def _progress(self, artifact: dict[str, Any], event: str, counters: _BackfillCounters) -> None:
        artifact["events"].append({"event": event, "metrics": self._metrics(counters)})
        artifact["summary"] = self._metrics(counters)
        self._write_artifact(artifact)

    def _write_artifact(self, artifact: dict[str, Any]) -> None:
        if self._progress_artifact is None:
            return
        self._progress_artifact.parent.mkdir(parents=True, exist_ok=True)
        self._progress_artifact.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str))


def _detail_cutoff(detail: IntradayEventDetail) -> datetime | None:
    return (
        detail.entry_timestamp
        or detail.confirmation_timestamp
        or us_equity_session_open_timestamp(detail.trading_date)
    )


def _json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _signal_identity_hash(session: Session, signal_id: str | None) -> str | None:
    if not signal_id:
        return None
    signal = session.get(SignalRegistry, signal_id)
    return signal.signal_identity_hash if signal is not None else None
