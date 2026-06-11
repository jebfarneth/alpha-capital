"""Historical M4 replay signal corpus selection helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from sqlalchemy import Column, MetaData, String, Table, false, or_, text
from sqlalchemy.orm import Query, Session

from alpha.db.models import EvidenceJob, EvidenceJobRun, FeatureSnapshot, SignalRegistry


HISTORICAL_M4_REPLAY_RECONSTRUCTION_METHOD = "historical_m4_replay_fmp_eod"
HISTORICAL_M4_REPLAY_JOB_NAMES = (
    "historical_m4_range_replay",
    "historical_m4_replay",
)
HISTORICAL_M4_REPLAY_RUN_STATUSES = ("finished", "partial_failed")
M4_PATTERN_ID = "M4"
SIGNAL_SOURCE_LIVE = "live"
SIGNAL_SOURCE_HISTORICAL_M4_REPLAY = "historical-m4-replay"
SIGNAL_SOURCE_CHOICES = (
    SIGNAL_SOURCE_LIVE,
    SIGNAL_SOURCE_HISTORICAL_M4_REPLAY,
)
_REPLAY_MEMBERSHIP_TEMP_TABLE = "_tmp_historical_m4_replay_signal_ids"
_REPLAY_MEMBERSHIP_STAGE_BATCH_SIZE = 1000


def normalize_signal_source(value: str | None) -> str:
    source = str(value or SIGNAL_SOURCE_LIVE).strip().lower()
    if source in {"historical_m4_replay", "historical-m4-only"}:
        source = SIGNAL_SOURCE_HISTORICAL_M4_REPLAY
    if source not in SIGNAL_SOURCE_CHOICES:
        raise ValueError(
            "signal_source must be one of: "
            + ", ".join(sorted(SIGNAL_SOURCE_CHOICES))
        )
    return source


def apply_signal_source_filter(
    query: Query,
    session: Session,
    *,
    signal_source: str | None,
    signal_start_date: date | None = None,
    signal_end_date: date | None = None,
) -> Query:
    """Restrict a SignalRegistry query to the requested corpus source."""

    source = normalize_signal_source(signal_source)
    if source == SIGNAL_SOURCE_LIVE:
        return query

    signal_ids = historical_m4_replay_signal_ids(
        session,
        signal_start_date=signal_start_date,
        signal_end_date=signal_end_date,
    )
    if not signal_ids:
        return query.filter(false())
    membership = _stage_replay_membership_signal_ids(session, signal_ids)
    return query.join(
        membership,
        SignalRegistry.signal_id == membership.c.signal_id,
    )


def historical_m4_replay_signal_query(
    session: Session,
    *,
    signal_start_date: date | None = None,
    signal_end_date: date | None = None,
) -> Query:
    query = session.query(SignalRegistry).filter(
        SignalRegistry.pattern_id == M4_PATTERN_ID
    )
    query = _apply_signal_date_bounds(
        query,
        signal_start_date=signal_start_date,
        signal_end_date=signal_end_date,
    )
    return apply_signal_source_filter(
        query,
        session,
        signal_source=SIGNAL_SOURCE_HISTORICAL_M4_REPLAY,
        signal_start_date=signal_start_date,
        signal_end_date=signal_end_date,
    )


def historical_m4_replay_signal_ids(
    session: Session,
    *,
    signal_start_date: date | None = None,
    signal_end_date: date | None = None,
) -> set[str]:
    stamped_ids = _historical_replay_stamped_signal_ids(
        session,
        signal_start_date=signal_start_date,
        signal_end_date=signal_end_date,
    )
    reused_ids = _historical_replay_reused_signal_ids(
        session,
        signal_start_date=signal_start_date,
        signal_end_date=signal_end_date,
    )
    return stamped_ids | reused_ids


def _historical_replay_stamped_signal_ids(
    session: Session,
    *,
    signal_start_date: date | None,
    signal_end_date: date | None,
) -> set[str]:
    query = (
        session.query(SignalRegistry.signal_id)
        .join(
            FeatureSnapshot,
            SignalRegistry.feature_snapshot_id == FeatureSnapshot.feature_snapshot_id,
        )
        .filter(
            SignalRegistry.pattern_id == M4_PATTERN_ID,
            or_(
                FeatureSnapshot.feature_json.like(
                    _json_like_pattern(HISTORICAL_M4_REPLAY_RECONSTRUCTION_METHOD)
                ),
                FeatureSnapshot.feature_json.like(
                    _compact_json_like_pattern(
                        HISTORICAL_M4_REPLAY_RECONSTRUCTION_METHOD
                    )
                ),
            ),
        )
    )
    query = _apply_signal_date_bounds(
        query,
        signal_start_date=signal_start_date,
        signal_end_date=signal_end_date,
    )
    return {str(signal_id) for (signal_id,) in query.all()}


def _historical_replay_reused_signal_ids(
    session: Session,
    *,
    signal_start_date: date | None,
    signal_end_date: date | None,
) -> set[str]:
    rows = (
        session.query(EvidenceJobRun.metric_json)
        .join(EvidenceJob, EvidenceJobRun.job_id == EvidenceJob.job_id)
        .filter(
            EvidenceJob.job_name.in_(HISTORICAL_M4_REPLAY_JOB_NAMES),
            EvidenceJobRun.run_status.in_(HISTORICAL_M4_REPLAY_RUN_STATUSES),
            EvidenceJobRun.metric_json.isnot(None),
        )
        .all()
    )
    signal_ids: set[str] = set()
    for (metric_json,) in rows:
        payload = _json_dict(metric_json)
        if not payload:
            continue
        signal_ids.update(
            _reused_signal_ids_from_metrics(
                payload,
                signal_start_date=signal_start_date,
                signal_end_date=signal_end_date,
            )
        )
    return _filter_signal_ids_by_date(
        session,
        signal_ids,
        signal_start_date=signal_start_date,
        signal_end_date=signal_end_date,
    )


def _reused_signal_ids_from_metrics(
    payload: Any,
    *,
    signal_start_date: date | None,
    signal_end_date: date | None,
    date_allowed: bool = True,
) -> set[str]:
    if isinstance(payload, dict):
        local_allowed = date_allowed
        metric_date = _metric_date(payload)
        if metric_date is not None:
            local_allowed = _date_in_bounds(
                metric_date,
                signal_start_date=signal_start_date,
                signal_end_date=signal_end_date,
            )
        signal_ids: set[str] = set()
        if local_allowed:
            signal_ids.update(_string_values(payload.get("reused_signal_ids")))
        for value in payload.values():
            signal_ids.update(
                _reused_signal_ids_from_metrics(
                    value,
                    signal_start_date=signal_start_date,
                    signal_end_date=signal_end_date,
                    date_allowed=local_allowed,
                )
            )
        return signal_ids
    if isinstance(payload, list):
        signal_ids: set[str] = set()
        for value in payload:
            signal_ids.update(
                _reused_signal_ids_from_metrics(
                    value,
                    signal_start_date=signal_start_date,
                    signal_end_date=signal_end_date,
                    date_allowed=date_allowed,
                )
            )
        return signal_ids
    return set()


def _apply_signal_date_bounds(
    query: Query,
    *,
    signal_start_date: date | None,
    signal_end_date: date | None,
) -> Query:
    if signal_start_date is not None:
        query = query.filter(
            SignalRegistry.signal_timestamp
            >= datetime.combine(signal_start_date, time.min, timezone.utc)
        )
    if signal_end_date is not None:
        query = query.filter(
            SignalRegistry.signal_timestamp
            < datetime.combine(
                signal_end_date + timedelta(days=1),
                time.min,
                timezone.utc,
            )
        )
    return query


def _filter_signal_ids_by_date(
    session: Session,
    signal_ids: set[str],
    *,
    signal_start_date: date | None,
    signal_end_date: date | None,
) -> set[str]:
    if not signal_ids:
        return set()
    membership = _stage_replay_membership_signal_ids(session, signal_ids)
    query = session.query(SignalRegistry.signal_id).join(
        membership,
        SignalRegistry.signal_id == membership.c.signal_id,
    )
    query = _apply_signal_date_bounds(
        query,
        signal_start_date=signal_start_date,
        signal_end_date=signal_end_date,
    )
    return {str(signal_id) for (signal_id,) in query.all()}


def _stage_replay_membership_signal_ids(
    session: Session,
    signal_ids: set[str],
) -> Table:
    membership = _replay_membership_table()
    dialect_name = session.get_bind().dialect.name
    create_sql = (
        f"CREATE TEMPORARY TABLE IF NOT EXISTS {_REPLAY_MEMBERSHIP_TEMP_TABLE} "
        "(signal_id VARCHAR PRIMARY KEY)"
    )
    if dialect_name == "postgresql":
        create_sql += " ON COMMIT DROP"
    session.execute(text(create_sql))
    session.execute(text(f"DELETE FROM {_REPLAY_MEMBERSHIP_TEMP_TABLE}"))
    rows = [{"signal_id": signal_id} for signal_id in sorted(signal_ids)]
    for chunk in _chunks(rows, _REPLAY_MEMBERSHIP_STAGE_BATCH_SIZE):
        session.execute(membership.insert(), chunk)
    return membership


def _replay_membership_table() -> Table:
    metadata = MetaData()
    return Table(
        _REPLAY_MEMBERSHIP_TEMP_TABLE,
        metadata,
        Column("signal_id", String, primary_key=True),
    )


def _chunks(rows: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [rows[index:index + size] for index in range(0, len(rows), size)]


def _json_like_pattern(method: str) -> str:
    return f'%"reconstruction_method": "{method}"%'


def _compact_json_like_pattern(method: str) -> str:
    return f'%"reconstruction_method":"{method}"%'


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        loaded = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _string_values(value: Any) -> set[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        return set()
    return {str(item) for item in value if str(item or "").strip()}


def _metric_date(payload: dict[str, Any]) -> date | None:
    raw = payload.get("replay_date") or payload.get("trading_date")
    if raw is None:
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError:
        return None


def _date_in_bounds(
    value: date,
    *,
    signal_start_date: date | None,
    signal_end_date: date | None,
) -> bool:
    if signal_start_date is not None and value < signal_start_date:
        return False
    if signal_end_date is not None and value > signal_end_date:
        return False
    return True
