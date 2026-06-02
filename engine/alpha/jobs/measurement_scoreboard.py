"""Read-only measurement scoreboard over forward-return observations.

The scoreboard consumes the all-firings forward-return spine exactly as
persisted by ``forward_return.py``. It does not re-grade, filter for
tradeability, or re-derive PIT/survivorship truth.

One firing is one ``signal_id``. If producer input identity changes for a fixed
signal, stale ``(signal_id, input_hash)`` rows are excluded and the canonical
observation is the latest persisted row by ``updated_at``, ``created_at``, then
primary key. That recency choice is corpus-load-bearing, so timestamps are
compared as aware UTC datetimes rather than strings; missing timestamps sort
oldest. Pattern-filtered runs are scoped reporting, not a global drift sentinel;
only an unfiltered run fails loud on a globally unknown status.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha.db.models import ForwardReturnObservation
from alpha.jobs.forward_return import (
    REQUIRED_FORWARD_RETURN_STATUSES,
    RETRYABLE_FORWARD_RETURN_STATUSES,
    STATUS_COMPUTED,
    STATUS_CORPORATE_ACTION_REVIEW,
    STATUS_HALTED_PENDING,
    STATUS_INVALID_ENTRY_PRICE_RETRY,
    STATUS_INVALID_EXIT_PRICE_RETRY,
    STATUS_MISSING_ENTRY_PRICE_RETRY,
    STATUS_MISSING_EXIT_PRICE_RETRY,
    STATUS_OUTCOME_UNAVAILABLE,
    STATUS_PENDING,
    STATUS_PRICE_DRIFT_REVIEW,
    STATUS_PRICE_FINALITY_PENDING,
    STATUS_PRICING_UNAVAILABLE_RETRY,
    STATUS_PROVIDER_REVISION_REVIEW,
    STATUS_SURVIVORSHIP_UNRESOLVED_REVIEW,
)

DEFAULT_MFE_TAIL_THRESHOLD = 0.25

BUCKET_GRADED = "graded"
BUCKET_PENDING_LIKE = "pending_like"
BUCKET_RETRY_IN_FLIGHT = "retry_in_flight"
BUCKET_REVIEW_UNRESOLVED = "review_unresolved"
BUCKET_TERMINAL_UNAVAILABLE = "terminal_unavailable"

GRADED_STATUSES = frozenset({STATUS_COMPUTED})
PENDING_LIKE_STATUSES = frozenset({
    STATUS_PENDING,
    STATUS_HALTED_PENDING,
    STATUS_PRICE_FINALITY_PENDING,
})
RETRY_IN_FLIGHT_STATUSES = frozenset({
    STATUS_PRICING_UNAVAILABLE_RETRY,
    STATUS_MISSING_ENTRY_PRICE_RETRY,
    STATUS_MISSING_EXIT_PRICE_RETRY,
    STATUS_INVALID_ENTRY_PRICE_RETRY,
    STATUS_INVALID_EXIT_PRICE_RETRY,
})
REVIEW_UNRESOLVED_STATUSES = frozenset({
    STATUS_CORPORATE_ACTION_REVIEW,
    STATUS_SURVIVORSHIP_UNRESOLVED_REVIEW,
    STATUS_PROVIDER_REVISION_REVIEW,
    STATUS_PRICE_DRIFT_REVIEW,
})
TERMINAL_UNAVAILABLE_STATUSES = frozenset({STATUS_OUTCOME_UNAVAILABLE})

ROLLUP_STATUS_BUCKETS: Mapping[str, frozenset[str]] = {
    BUCKET_GRADED: GRADED_STATUSES,
    BUCKET_PENDING_LIKE: PENDING_LIKE_STATUSES,
    BUCKET_RETRY_IN_FLIGHT: RETRY_IN_FLIGHT_STATUSES,
    BUCKET_REVIEW_UNRESOLVED: REVIEW_UNRESOLVED_STATUSES,
    BUCKET_TERMINAL_UNAVAILABLE: TERMINAL_UNAVAILABLE_STATUSES,
}


class ScoreboardPartitionError(RuntimeError):
    """Raised when persisted observations no longer match the status universe."""

    def __init__(
        self,
        message: str,
        *,
        unknown_status_counts: Optional[Dict[str, int]] = None,
        unknown_status_details: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ):
        super().__init__(message)
        self.unknown_status_counts = unknown_status_counts or {}
        self.unknown_status_details = unknown_status_details or {}


@dataclass(frozen=True)
class AnomalySummary:
    computed_missing_forward_return: int
    non_computed_with_forward_return: int
    computed_missing_forward_return_ids: tuple[str, ...]
    non_computed_with_forward_return_ids: tuple[str, ...]

    @property
    def total(self) -> int:
        return self.computed_missing_forward_return + self.non_computed_with_forward_return


@dataclass(frozen=True)
class ComputedStats:
    n: int
    no_graded_firings: bool
    expectancy: Optional[float]
    median: Optional[float]
    best: Optional[float]
    worst: Optional[float]
    win_count: int
    flat_count: int
    loss_count: int
    avg_win: Optional[float]
    avg_loss: Optional[float]
    win_loss_ratio: Optional[float]
    hit_t1_count: int
    hit_t1_rate: Optional[float]
    hit_t2_count: int
    hit_t2_rate: Optional[float]
    hit_t3_count: int
    hit_t3_rate: Optional[float]
    hit_stop_count: int
    hit_stop_rate: Optional[float]
    same_day_barrier_ambiguity_count: int
    mfe_mean: Optional[float]
    mfe_median: Optional[float]
    mfe_max: Optional[float]
    mae_mean: Optional[float]
    mae_median: Optional[float]
    mae_worst: Optional[float]
    tail_event_count: int
    tail_event_fraction: Optional[float]


@dataclass(frozen=True)
class GradedRollupReconciliation:
    graded_rollup_count: int
    computed_sample_n: int
    computed_missing_forward_return: int
    reconciles: bool


@dataclass(frozen=True)
class ScoreboardResult:
    pattern_id: Optional[str]
    mfe_tail_threshold: float
    total_observations: int
    raw_observation_rows: int
    stale_duplicate_observation_rows: int
    per_status_counts: Dict[str, int]
    rollup_counts: Dict[str, int]
    unknown_status_counts: Dict[str, int]
    anomalies: AnomalySummary
    graded_rollup_reconciliation: GradedRollupReconciliation
    computed_stats: ComputedStats

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_status_partition() -> None:
    """Fail if the scoreboard buckets drift from the producer status universe."""

    required = set(REQUIRED_FORWARD_RETURN_STATUSES)
    bucket_sets = list(ROLLUP_STATUS_BUCKETS.values())
    union = set().union(*bucket_sets)
    if union != required:
        missing = sorted(required - union)
        extra = sorted(union - required)
        raise ScoreboardPartitionError(
            f"forward-return status partition is not exhaustive: missing={missing} extra={extra}"
        )

    seen: set[str] = set()
    duplicates: set[str] = set()
    for statuses in bucket_sets:
        duplicates.update(seen.intersection(statuses))
        seen.update(statuses)
    if duplicates:
        raise ScoreboardPartitionError(
            f"forward-return statuses mapped to multiple scoreboard buckets: {sorted(duplicates)}"
        )

    retryable = set(RETRYABLE_FORWARD_RETURN_STATUSES)
    if not RETRY_IN_FLIGHT_STATUSES.issubset(retryable):
        raise ScoreboardPartitionError(
            "retry_in_flight statuses must remain producer-declared retryable"
        )


def build_measurement_scoreboard(
    session: Session,
    *,
    pattern_id: Optional[str] = None,
    mfe_tail_threshold: float = DEFAULT_MFE_TAIL_THRESHOLD,
) -> ScoreboardResult:
    """Build a read-only scoreboard from ``forward_return_observations``.

    ``mfe_tail_threshold`` is a return fraction, so ``0.25`` means +25%.
    Tail events use an inclusive ``>=`` boundary.
    """

    if not math.isfinite(mfe_tail_threshold) or mfe_tail_threshold < 0:
        raise ValueError("mfe_tail_threshold must be a finite, non-negative fraction")

    validate_status_partition()
    raw_rows = _load_observation_rows(session, pattern_id=pattern_id)
    canonical_rows = _canonical_observation_rows(raw_rows)
    canonical_observation_ids = {
        row["forward_return_observation_id"] for row in canonical_rows
    }

    per_status_counts = {status: 0 for status in REQUIRED_FORWARD_RETURN_STATUSES}
    unknown_status_counts: Dict[str, int] = {}
    unknown_status_details: Dict[str, List[Dict[str, Any]]] = {}
    for row in raw_rows:
        status = row["status"]
        if status in per_status_counts:
            continue
        else:
            unknown_status_counts[status] = unknown_status_counts.get(status, 0) + 1
            unknown_status_details.setdefault(status, []).append({
                "signal_id": row["signal_id"],
                "observation_id": row["forward_return_observation_id"],
                "stale": row["forward_return_observation_id"] not in canonical_observation_ids,
            })

    if unknown_status_counts:
        raise ScoreboardPartitionError(
            f"unknown forward-return statuses present: {unknown_status_counts}",
            unknown_status_counts=unknown_status_counts,
            unknown_status_details=unknown_status_details,
        )

    rows = canonical_rows
    stale_duplicate_count = len(raw_rows) - len(rows)
    for row in rows:
        per_status_counts[row["status"]] += 1

    status_to_bucket = _status_to_bucket()
    rollup_counts = {bucket: 0 for bucket in ROLLUP_STATUS_BUCKETS}
    computed_rows: List[Mapping[str, Any]] = []
    computed_missing_forward_return_ids: List[str] = []
    non_computed_with_forward_return_ids: List[str] = []

    for row in rows:
        status = row["status"]
        rollup_counts[status_to_bucket[status]] += 1
        forward_return = row["forward_return"]
        if status == STATUS_COMPUTED and not _is_finite_number(forward_return):
            computed_missing_forward_return_ids.append(row["forward_return_observation_id"])
            continue
        if status != STATUS_COMPUTED and forward_return is not None:
            non_computed_with_forward_return_ids.append(row["forward_return_observation_id"])
            continue
        if status == STATUS_COMPUTED:
            computed_rows.append(row)

    anomalies = AnomalySummary(
        computed_missing_forward_return=len(computed_missing_forward_return_ids),
        non_computed_with_forward_return=len(non_computed_with_forward_return_ids),
        computed_missing_forward_return_ids=tuple(computed_missing_forward_return_ids[:20]),
        non_computed_with_forward_return_ids=tuple(non_computed_with_forward_return_ids[:20]),
    )
    computed_stats = _computed_stats(computed_rows, mfe_tail_threshold=mfe_tail_threshold)
    reconciliation = GradedRollupReconciliation(
        graded_rollup_count=rollup_counts[BUCKET_GRADED],
        computed_sample_n=computed_stats.n,
        computed_missing_forward_return=anomalies.computed_missing_forward_return,
        reconciles=(
            rollup_counts[BUCKET_GRADED]
            == computed_stats.n + anomalies.computed_missing_forward_return
        ),
    )

    return ScoreboardResult(
        pattern_id=pattern_id,
        mfe_tail_threshold=mfe_tail_threshold,
        total_observations=len(rows),
        raw_observation_rows=len(raw_rows),
        stale_duplicate_observation_rows=stale_duplicate_count,
        per_status_counts=per_status_counts,
        rollup_counts=rollup_counts,
        unknown_status_counts={},
        anomalies=anomalies,
        graded_rollup_reconciliation=reconciliation,
        computed_stats=computed_stats,
    )


def _load_observation_rows(
    session: Session,
    *,
    pattern_id: Optional[str],
) -> List[Mapping[str, Any]]:
    columns = (
        ForwardReturnObservation.forward_return_observation_id,
        ForwardReturnObservation.signal_id,
        ForwardReturnObservation.input_hash,
        ForwardReturnObservation.status,
        ForwardReturnObservation.forward_return,
        ForwardReturnObservation.max_close_return,
        ForwardReturnObservation.min_close_return,
        ForwardReturnObservation.max_favorable_excursion,
        ForwardReturnObservation.max_adverse_excursion,
        ForwardReturnObservation.mfe_session_date,
        ForwardReturnObservation.mae_session_date,
        ForwardReturnObservation.hit_t1_intraday,
        ForwardReturnObservation.hit_t2_intraday,
        ForwardReturnObservation.hit_t3_intraday,
        ForwardReturnObservation.hit_stop_intraday,
        ForwardReturnObservation.same_day_barrier_ambiguity,
        ForwardReturnObservation.pattern_id,
        ForwardReturnObservation.ticker,
        ForwardReturnObservation.direction,
        ForwardReturnObservation.signal_timestamp,
        ForwardReturnObservation.entry_session_date,
        ForwardReturnObservation.exit_session_date,
        ForwardReturnObservation.created_at,
        ForwardReturnObservation.updated_at,
    )
    statement = select(*columns)
    if pattern_id:
        statement = statement.where(ForwardReturnObservation.pattern_id == pattern_id)
    with session.no_autoflush:
        rows = session.execute(statement).all()
    return [dict(row._mapping) for row in rows]


def _canonical_observation_rows(
    rows: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    by_signal: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        signal_id = row["signal_id"]
        current = by_signal.get(signal_id)
        if current is None or _canonical_sort_key(row) > _canonical_sort_key(current):
            by_signal[signal_id] = row
    return sorted(
        by_signal.values(),
        key=lambda row: (
            _timestamp_sort_key(row.get("signal_timestamp")),
            str(row.get("ticker") or ""),
            str(row.get("forward_return_observation_id") or ""),
        ),
    )


def _canonical_sort_key(row: Mapping[str, Any]) -> Tuple[Tuple[int, datetime], Tuple[int, datetime], str]:
    return (
        _timestamp_sort_key(row.get("updated_at")),
        _timestamp_sort_key(row.get("created_at")),
        str(row.get("forward_return_observation_id") or ""),
    )


def _timestamp_sort_key(value: Any) -> Tuple[int, datetime]:
    if value is None:
        return (0, datetime.min.replace(tzinfo=timezone.utc))
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return (1, value.replace(tzinfo=timezone.utc))
        return (1, value.astimezone(timezone.utc))
    raise TypeError(f"expected datetime or None, got {type(value).__name__}")


def _status_to_bucket() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for bucket, statuses in ROLLUP_STATUS_BUCKETS.items():
        for status in statuses:
            mapping[status] = bucket
    return mapping


def _computed_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    mfe_tail_threshold: float,
) -> ComputedStats:
    """Compute graded-only stats.

    Barrier hit rates are count(True) / n(graded); by convention the denominator
    includes NULL barrier rows because the producer initializes mature barrier
    flags to False.
    """

    returns = [float(row["forward_return"]) for row in rows]
    if not returns:
        return ComputedStats(
            n=0,
            no_graded_firings=True,
            expectancy=None,
            median=None,
            best=None,
            worst=None,
            win_count=0,
            flat_count=0,
            loss_count=0,
            avg_win=None,
            avg_loss=None,
            win_loss_ratio=None,
            hit_t1_count=0,
            hit_t1_rate=None,
            hit_t2_count=0,
            hit_t2_rate=None,
            hit_t3_count=0,
            hit_t3_rate=None,
            hit_stop_count=0,
            hit_stop_rate=None,
            same_day_barrier_ambiguity_count=0,
            mfe_mean=None,
            mfe_median=None,
            mfe_max=None,
            mae_mean=None,
            mae_median=None,
            mae_worst=None,
            tail_event_count=0,
            tail_event_fraction=None,
        )

    wins = [value for value in returns if value > 0]
    flats = [value for value in returns if value == 0]
    losses = [value for value in returns if value < 0]
    avg_win = _mean_or_none(wins)
    avg_loss = _mean_or_none(losses)
    win_loss_ratio = (
        abs(avg_win / avg_loss)
        if avg_win is not None and avg_loss not in (None, 0)
        else None
    )
    n = len(rows)
    hit_t1_count = _truthy_count(row["hit_t1_intraday"] for row in rows)
    hit_t2_count = _truthy_count(row["hit_t2_intraday"] for row in rows)
    hit_t3_count = _truthy_count(row["hit_t3_intraday"] for row in rows)
    hit_stop_count = _truthy_count(row["hit_stop_intraday"] for row in rows)
    same_day_ambiguity_count = _truthy_count(
        row["same_day_barrier_ambiguity"] for row in rows
    )
    mfe_values = _float_values(row["max_favorable_excursion"] for row in rows)
    mae_values = _float_values(row["max_adverse_excursion"] for row in rows)
    tail_event_count = sum(
        1
        for value in (row["max_favorable_excursion"] for row in rows)
        if _is_finite_number(value) and float(value) >= mfe_tail_threshold
    )

    return ComputedStats(
        n=n,
        no_graded_firings=False,
        expectancy=mean(returns),
        median=median(returns),
        best=max(returns),
        worst=min(returns),
        win_count=len(wins),
        flat_count=len(flats),
        loss_count=len(losses),
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
        hit_t1_count=hit_t1_count,
        hit_t1_rate=hit_t1_count / n,
        hit_t2_count=hit_t2_count,
        hit_t2_rate=hit_t2_count / n,
        hit_t3_count=hit_t3_count,
        hit_t3_rate=hit_t3_count / n,
        hit_stop_count=hit_stop_count,
        hit_stop_rate=hit_stop_count / n,
        same_day_barrier_ambiguity_count=same_day_ambiguity_count,
        mfe_mean=_mean_or_none(mfe_values),
        mfe_median=_median_or_none(mfe_values),
        mfe_max=max(mfe_values) if mfe_values else None,
        mae_mean=_mean_or_none(mae_values),
        mae_median=_median_or_none(mae_values),
        mae_worst=min(mae_values) if mae_values else None,
        tail_event_count=tail_event_count,
        tail_event_fraction=tail_event_count / n,
    )


def _float_values(values: Iterable[Any]) -> List[float]:
    return [float(value) for value in values if _is_finite_number(value)]


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    return math.isfinite(float(value))


def _truthy_count(values: Iterable[Any]) -> int:
    return sum(1 for value in values if value is True)


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return mean(values) if values else None


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    return median(values) if values else None
