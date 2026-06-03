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
The same scoping applies to the orphan-integrity guard: a ``--pattern-id`` run
only sees observations claiming that pattern, so the unfiltered nightly run is
the required global orphan sentinel.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from alpha.db.models import ForwardReturnObservation, SignalRegistry
from alpha.market_calendar import is_us_equity_session, next_us_equity_session
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
CANONICAL_SIGNAL_HORIZON_RE = re.compile(r"^[1-9][0-9]*d$")

# forward_return is a raw price return with no direction term, so the
# win/loss, expectancy, and tail math is only correct for long positions.
# Until the scoreboard signs returns by direction, it refuses to grade
# anything that is not long.
SUPPORTED_DIRECTION = "long"

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


class ScoreboardDirectionError(RuntimeError):
    """Raised when an observation carries a direction the scoreboard cannot grade.

    forward_return is a raw price return with no direction term, so the
    win/loss, expectancy, and tail math is only correct for long positions. A
    short (or unknown-direction) row would silently count losing trades as wins
    and invert expectancy and the tail. The scoreboard fails loud until it signs
    returns by direction.
    """

    def __init__(
        self,
        message: str,
        *,
        unsupported_direction_counts: Optional[Dict[str, int]] = None,
        unsupported_direction_details: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ):
        super().__init__(message)
        self.unsupported_direction_counts = unsupported_direction_counts or {}
        self.unsupported_direction_details = unsupported_direction_details or {}


class ScoreboardSignalIntegrityError(RuntimeError):
    """Raised when an observation references a missing parent signal."""

    def __init__(
        self,
        message: str,
        *,
        orphans: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(message)
        self.orphans = orphans or []


class ScoreboardPoolingError(RuntimeError):
    """Raised when headline stats would pool multiple pattern distributions."""

    def __init__(
        self,
        message: str,
        *,
        pattern_counts: Optional[Dict[str, int]] = None,
        pattern_horizons: Optional[Dict[str, str]] = None,
        horizon_counts: Optional[Dict[str, int]] = None,
        pattern_horizon_counts: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        super().__init__(message)
        self.pattern_counts = pattern_counts or {}
        self.pattern_horizons = pattern_horizons or {}
        self.horizon_counts = horizon_counts or {}
        self.pattern_horizon_counts = pattern_horizon_counts or {}


class ScoreboardPatternIntegrityError(RuntimeError):
    """Raised when an observation's pattern disagrees with its parent signal."""

    def __init__(
        self,
        message: str,
        *,
        mismatches: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(message)
        self.mismatches = mismatches or []


class ScoreboardHorizonIntegrityError(RuntimeError):
    """Raised when a graded observation lacks a canonical signal horizon."""

    def __init__(
        self,
        message: str,
        *,
        horizon_errors: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(message)
        self.horizon_errors = horizon_errors or []


class ScoreboardWindowIntegrityError(RuntimeError):
    """Raised when a graded observation lacks a valid persisted forward window."""

    def __init__(
        self,
        message: str,
        *,
        window_errors: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(message)
        self.window_errors = window_errors or []


@dataclass(frozen=True)
class AnomalySummary:
    computed_missing_forward_return: int
    non_computed_with_forward_return: int
    graded_missing_excursion_count: int
    computed_missing_forward_return_by_pattern: Dict[str, int]
    computed_missing_forward_return_ids: tuple[str, ...]
    non_computed_with_forward_return_ids: tuple[str, ...]
    graded_missing_excursion_ids: tuple[str, ...]

    @property
    def total(self) -> int:
        return (
            self.computed_missing_forward_return
            + self.non_computed_with_forward_return
            + self.graded_missing_excursion_count
        )


@dataclass(frozen=True)
class ComputedStats:
    n: int
    total_firings: int
    no_graded_firings: bool
    distinct_tickers: int
    overlapping_window_firings: int
    max_concurrent_same_ticker: int
    effective_sample_size: Optional[float]
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
    mfe_finite_count: int
    mae_finite_count: int
    mfe_mean: Optional[float]
    mfe_median: Optional[float]
    mfe_max: Optional[float]
    mae_mean: Optional[float]
    mae_median: Optional[float]
    mae_worst: Optional[float]
    tail_event_count: int
    tail_event_denominator: int
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


@dataclass(frozen=True)
class HorizonScoreboardResult:
    pattern_id: Optional[str]
    mfe_tail_threshold: float
    computed_stats_by_horizon: Dict[str, ComputedStats]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ScoreboardFold:
    raw_rows: List[Mapping[str, Any]]
    rows: List[Mapping[str, Any]]
    stale_duplicate_count: int
    per_status_counts: Dict[str, int]
    rollup_counts: Dict[str, int]
    computed_rows: List[Mapping[str, Any]]
    anomalies: AnomalySummary


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
    signal_horizon: Optional[str] = None,
    mfe_tail_threshold: float = DEFAULT_MFE_TAIL_THRESHOLD,
) -> ScoreboardResult:
    """Build a read-only scoreboard from ``forward_return_observations``.

    ``mfe_tail_threshold`` is a return fraction, so ``0.25`` means +25%.
    Tail events use an inclusive ``>=`` boundary.
    """

    if not math.isfinite(mfe_tail_threshold) or mfe_tail_threshold < 0:
        raise ValueError("mfe_tail_threshold must be a finite, non-negative fraction")
    if pattern_id is not None and (
        not isinstance(pattern_id, str) or pattern_id.strip() == ""
    ):
        raise ValueError("pattern_id must be a non-empty string")
    canonical_signal_horizon = _validate_signal_horizon_filter(signal_horizon)

    fold = _fold_scoreboard_rows(
        session,
        pattern_id=pattern_id,
        signal_horizon=canonical_signal_horizon,
    )
    _assert_pattern_integrity(fold.computed_rows)
    _assert_horizon_integrity(fold.computed_rows)
    _assert_single_graded_pattern(fold.computed_rows)
    if signal_horizon is None:
        _assert_single_graded_horizon(fold.computed_rows)

    computed_stats = _computed_stats(
        fold.computed_rows,
        mfe_tail_threshold=mfe_tail_threshold,
    )
    reconciliation = GradedRollupReconciliation(
        graded_rollup_count=fold.rollup_counts[BUCKET_GRADED],
        computed_sample_n=computed_stats.n,
        computed_missing_forward_return=fold.anomalies.computed_missing_forward_return,
        reconciles=(
            fold.rollup_counts[BUCKET_GRADED]
            == computed_stats.n + fold.anomalies.computed_missing_forward_return
        ),
    )

    return ScoreboardResult(
        pattern_id=pattern_id,
        mfe_tail_threshold=mfe_tail_threshold,
        total_observations=len(fold.rows),
        raw_observation_rows=len(fold.raw_rows),
        stale_duplicate_observation_rows=fold.stale_duplicate_count,
        per_status_counts=fold.per_status_counts,
        rollup_counts=fold.rollup_counts,
        unknown_status_counts={},
        anomalies=fold.anomalies,
        graded_rollup_reconciliation=reconciliation,
        computed_stats=computed_stats,
    )


def build_measurement_scoreboard_by_horizon(
    session: Session,
    *,
    pattern_id: Optional[str] = None,
    mfe_tail_threshold: float = DEFAULT_MFE_TAIL_THRESHOLD,
) -> HorizonScoreboardResult:
    """Build computed stats partitioned by persisted signal horizon."""

    if pattern_id is not None and (
        not isinstance(pattern_id, str) or pattern_id.strip() == ""
    ):
        raise ValueError("pattern_id must be a non-empty string")
    if not math.isfinite(mfe_tail_threshold) or mfe_tail_threshold < 0:
        raise ValueError("mfe_tail_threshold must be a finite, non-negative fraction")

    fold = _fold_scoreboard_rows(
        session,
        pattern_id=pattern_id,
        signal_horizon=None,
    )
    _assert_pattern_integrity(fold.computed_rows)
    _assert_horizon_integrity(fold.computed_rows)
    _assert_single_graded_pattern(fold.computed_rows)

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in fold.computed_rows:
        horizon = _horizon_key(row)
        grouped.setdefault(horizon, []).append(row)

    return HorizonScoreboardResult(
        pattern_id=pattern_id,
        mfe_tail_threshold=mfe_tail_threshold,
        computed_stats_by_horizon={
            horizon: _computed_stats(rows, mfe_tail_threshold=mfe_tail_threshold)
            for horizon, rows in sorted(grouped.items())
        },
    )


def _fold_scoreboard_rows(
    session: Session,
    *,
    pattern_id: Optional[str],
    signal_horizon: Optional[str],
) -> _ScoreboardFold:
    validate_status_partition()
    raw_rows = _load_observation_rows(
        session,
        pattern_id=pattern_id,
        signal_horizon=None,
    )
    raw_rows = _filter_rows_for_signal_horizon(raw_rows, signal_horizon)
    # Guard order is load-bearing: orphan integrity gates the denominator before
    # status/direction classification; a row missing its parent must fail loud,
    # never be classified or folded.
    _assert_no_orphan_signals(raw_rows)
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

    _assert_supported_direction(raw_rows, canonical_observation_ids)

    rows = canonical_rows
    stale_duplicate_count = len(raw_rows) - len(rows)
    for row in rows:
        per_status_counts[row["status"]] += 1

    status_to_bucket = _status_to_bucket()
    rollup_counts = {bucket: 0 for bucket in ROLLUP_STATUS_BUCKETS}
    computed_rows: List[Mapping[str, Any]] = []
    computed_missing_forward_return_ids: List[str] = []
    computed_missing_forward_return_by_pattern: Dict[str, int] = {}
    non_computed_with_forward_return_ids: List[str] = []
    graded_missing_excursion_ids: List[str] = []

    for row in rows:
        status = row["status"]
        rollup_counts[status_to_bucket[status]] += 1
        forward_return = row["forward_return"]
        if status == STATUS_COMPUTED and not _is_finite_number(forward_return):
            computed_missing_forward_return_ids.append(row["forward_return_observation_id"])
            pattern = str(row["signal_pattern_id"])
            computed_missing_forward_return_by_pattern[pattern] = (
                computed_missing_forward_return_by_pattern.get(pattern, 0) + 1
            )
            continue
        if status != STATUS_COMPUTED and forward_return is not None:
            non_computed_with_forward_return_ids.append(row["forward_return_observation_id"])
            continue
        if status == STATUS_COMPUTED:
            if (
                not _is_finite_number(row["max_favorable_excursion"])
                or not _is_finite_number(row["max_adverse_excursion"])
            ):
                graded_missing_excursion_ids.append(row["forward_return_observation_id"])
            computed_rows.append(row)

    anomalies = AnomalySummary(
        computed_missing_forward_return=len(computed_missing_forward_return_ids),
        non_computed_with_forward_return=len(non_computed_with_forward_return_ids),
        graded_missing_excursion_count=len(graded_missing_excursion_ids),
        computed_missing_forward_return_by_pattern=computed_missing_forward_return_by_pattern,
        computed_missing_forward_return_ids=tuple(computed_missing_forward_return_ids[:20]),
        non_computed_with_forward_return_ids=tuple(non_computed_with_forward_return_ids[:20]),
        graded_missing_excursion_ids=tuple(graded_missing_excursion_ids[:20]),
    )
    return _ScoreboardFold(
        raw_rows=raw_rows,
        rows=rows,
        stale_duplicate_count=stale_duplicate_count,
        per_status_counts=per_status_counts,
        rollup_counts=rollup_counts,
        computed_rows=computed_rows,
        anomalies=anomalies,
    )


def _assert_no_orphan_signals(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail if any observation is missing its parent signal registry row."""

    orphans: List[Dict[str, Any]] = []
    for row in rows:
        if row["parent_signal_id"] is not None:
            continue
        orphans.append({
            "observation_id": row["forward_return_observation_id"],
            "signal_id": row["signal_id"],
            "status": row["status"],
            "direction": row["direction"],
            "pattern_id": row["pattern_id"],
        })

    if orphans:
        raise ScoreboardSignalIntegrityError(
            "forward-return observations reference missing signal_registry "
            "parent rows",
            orphans=orphans,
        )


def _assert_supported_direction(
    rows: Sequence[Mapping[str, Any]],
    canonical_observation_ids: set[str],
) -> None:
    """Fail loud if any observation is not a long position.

    Scans the raw (pre-dedup) rows, mirroring the unknown-status check, so a
    non-long direction trips the guard even on a stale duplicate row.
    """

    counts: Dict[str, int] = {}
    details: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        direction = row["direction"]
        if direction == SUPPORTED_DIRECTION:
            continue
        key = "null" if direction is None else str(direction)
        counts[key] = counts.get(key, 0) + 1
        details.setdefault(key, []).append({
            "signal_id": row["signal_id"],
            "observation_id": row["forward_return_observation_id"],
            "status": row["status"],
            "stale": row["forward_return_observation_id"] not in canonical_observation_ids,
        })

    if counts:
        raise ScoreboardDirectionError(
            "forward-return observations carry non-long direction the scoreboard "
            f"cannot grade (raw return is unsigned): {counts}",
            unsupported_direction_counts=counts,
            unsupported_direction_details=details,
        )


def _assert_pattern_integrity(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail if a graded observation is attributed to a different parent pattern."""

    mismatches: List[Dict[str, Any]] = []
    for row in rows:
        observation_pattern = row["pattern_id"]
        signal_pattern = row["signal_pattern_id"]
        if observation_pattern == signal_pattern:
            continue
        mismatches.append({
            "signal_id": row["signal_id"],
            "observation_id": row["forward_return_observation_id"],
            "observation_pattern_id": observation_pattern,
            "signal_pattern_id": signal_pattern,
        })

    if mismatches:
        raise ScoreboardPatternIntegrityError(
            "forward-return observation pattern_id does not match parent "
            "signal_registry pattern_id",
            mismatches=mismatches,
        )


def _assert_horizon_integrity(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail if a graded observation cannot be assigned one canonical horizon."""

    horizon_errors: List[Dict[str, Any]] = []
    for row in rows:
        raw_horizon = row.get("signal_horizon")
        if _canonical_signal_horizon(raw_horizon) is not None:
            continue
        horizon_errors.append({
            "signal_id": row["signal_id"],
            "observation_id": row["forward_return_observation_id"],
            "pattern_id": row["signal_pattern_id"],
            "signal_horizon": raw_horizon,
            "status": row["status"],
        })

    if horizon_errors:
        raise ScoreboardHorizonIntegrityError(
            "graded forward-return observations carry missing or "
            "non-canonical signal_horizon values",
            horizon_errors=horizon_errors,
        )


def _assert_single_graded_pattern(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail loud before reducing multiple pattern distributions into one stat block."""

    pattern_counts: Dict[str, int] = {}
    pattern_horizons: Dict[str, str] = {}
    for row in rows:
        pattern = str(row["signal_pattern_id"])
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        horizon = _canonical_signal_horizon(row.get("signal_horizon"))
        if horizon is not None:
            pattern_horizons.setdefault(pattern, horizon)

    if len(pattern_counts) > 1:
        raise ScoreboardPoolingError(
            "scoreboard headline stats cannot pool multiple graded patterns; "
            "re-run with --pattern-id <pattern_id> to isolate one pattern",
            pattern_counts=pattern_counts,
            pattern_horizons=pattern_horizons,
        )


def _assert_single_graded_horizon(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail loud before reducing multiple holding periods into one stat block."""

    horizon_counts: Dict[str, int] = {}
    pattern_counts: Dict[str, int] = {}
    pattern_horizon_counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        pattern = str(row["signal_pattern_id"])
        horizon = _horizon_key(row)
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        horizon_counts[horizon] = horizon_counts.get(horizon, 0) + 1
        per_pattern = pattern_horizon_counts.setdefault(pattern, {})
        per_pattern[horizon] = per_pattern.get(horizon, 0) + 1

    if len(horizon_counts) > 1:
        raise ScoreboardPoolingError(
            "scoreboard headline stats cannot pool multiple graded horizons; "
            "re-run with --signal-horizon <signal_horizon> or "
            "--group-by-horizon to isolate holding periods",
            pattern_counts=pattern_counts,
            pattern_horizons={
                pattern: ",".join(sorted(counts))
                for pattern, counts in pattern_horizon_counts.items()
            },
            horizon_counts=horizon_counts,
            pattern_horizon_counts=pattern_horizon_counts,
        )


def _horizon_key(row: Mapping[str, Any]) -> str:
    horizon = _canonical_signal_horizon(row.get("signal_horizon"))
    return "invalid" if horizon is None else horizon


def _validate_signal_horizon_filter(signal_horizon: Optional[str]) -> Optional[str]:
    if signal_horizon is None:
        return None
    if not isinstance(signal_horizon, str):
        raise ValueError("signal_horizon must be a canonical Nd string, such as 15d")
    canonical = _canonical_signal_horizon(signal_horizon)
    if canonical is None:
        raise ValueError("signal_horizon must be a canonical Nd string, such as 15d")
    return canonical


def _canonical_signal_horizon(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not CANONICAL_SIGNAL_HORIZON_RE.fullmatch(text):
        return None
    return text


def _filter_rows_for_signal_horizon(
    rows: Sequence[Mapping[str, Any]],
    signal_horizon: Optional[str],
) -> List[Mapping[str, Any]]:
    if signal_horizon is None:
        return list(rows)

    filtered: List[Mapping[str, Any]] = []
    for row in rows:
        canonical_horizon = _canonical_signal_horizon(row.get("signal_horizon"))
        if canonical_horizon == signal_horizon:
            filtered.append(row)
            continue
        # A bad finite-computed horizon must fail closed even for a filtered
        # run; otherwise --signal-horizon 15d could hide exactly the rows that
        # would have polluted the unfiltered truth summary.
        if (
            canonical_horizon is None
            and row["status"] == STATUS_COMPUTED
            and _is_finite_number(row["forward_return"])
        ):
            filtered.append(row)
    return filtered


def _load_observation_rows(
    session: Session,
    *,
    pattern_id: Optional[str],
    signal_horizon: Optional[str],
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
        SignalRegistry.signal_id.label("parent_signal_id"),
        SignalRegistry.pattern_id.label("signal_pattern_id"),
        ForwardReturnObservation.ticker,
        ForwardReturnObservation.direction,
        ForwardReturnObservation.signal_horizon,
        ForwardReturnObservation.signal_timestamp,
        ForwardReturnObservation.entry_session_date,
        ForwardReturnObservation.exit_session_date,
        ForwardReturnObservation.created_at,
        ForwardReturnObservation.updated_at,
    )
    statement = select(*columns).join(
        SignalRegistry,
        SignalRegistry.signal_id == ForwardReturnObservation.signal_id,
        isouter=True,
    )
    if pattern_id is not None:
        statement = statement.where(
            or_(
                ForwardReturnObservation.pattern_id == pattern_id,
                SignalRegistry.pattern_id == pattern_id,
            )
        )
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
            total_firings=0,
            no_graded_firings=True,
            distinct_tickers=0,
            overlapping_window_firings=0,
            max_concurrent_same_ticker=0,
            effective_sample_size=None,
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
            mfe_finite_count=0,
            mae_finite_count=0,
            mfe_mean=None,
            mfe_median=None,
            mfe_max=None,
            mae_mean=None,
            mae_median=None,
            mae_worst=None,
            tail_event_count=0,
            tail_event_denominator=0,
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
    window_stats = _window_overlap_stats(rows)
    hit_t1_count = _truthy_count(row["hit_t1_intraday"] for row in rows)
    hit_t2_count = _truthy_count(row["hit_t2_intraday"] for row in rows)
    hit_t3_count = _truthy_count(row["hit_t3_intraday"] for row in rows)
    hit_stop_count = _truthy_count(row["hit_stop_intraday"] for row in rows)
    same_day_ambiguity_count = _truthy_count(
        row["same_day_barrier_ambiguity"] for row in rows
    )
    mfe_values = _float_values(row["max_favorable_excursion"] for row in rows)
    mae_values = _float_values(row["max_adverse_excursion"] for row in rows)
    mfe_finite_count = len(mfe_values)
    mae_finite_count = len(mae_values)
    tail_event_count = sum(
        1
        for value in (row["max_favorable_excursion"] for row in rows)
        if _is_finite_number(value) and float(value) >= mfe_tail_threshold
    )
    tail_event_denominator = mfe_finite_count

    return ComputedStats(
        n=n,
        total_firings=n,
        no_graded_firings=False,
        distinct_tickers=window_stats["distinct_tickers"],
        overlapping_window_firings=window_stats["overlapping_window_firings"],
        max_concurrent_same_ticker=window_stats["max_concurrent_same_ticker"],
        effective_sample_size=window_stats["effective_sample_size"],
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
        mfe_finite_count=mfe_finite_count,
        mae_finite_count=mae_finite_count,
        mfe_mean=_mean_or_none(mfe_values),
        mfe_median=_median_or_none(mfe_values),
        mfe_max=max(mfe_values) if mfe_values else None,
        mae_mean=_mean_or_none(mae_values),
        mae_median=_median_or_none(mae_values),
        mae_worst=min(mae_values) if mae_values else None,
        tail_event_count=tail_event_count,
        tail_event_denominator=tail_event_denominator,
        tail_event_fraction=(
            tail_event_count / tail_event_denominator
            if tail_event_denominator else None
        ),
    )


def _window_overlap_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    windows = _forward_windows(rows)
    if not windows:
        return {
            "distinct_tickers": 0,
            "overlapping_window_firings": 0,
            "max_concurrent_same_ticker": 0,
            "effective_sample_size": None,
        }

    distinct_tickers = len({window["ticker"] for window in windows})
    same_ticker_concurrency: Dict[Tuple[str, date], int] = {}
    for window in windows:
        ticker = window["ticker"]
        for session_date in window["sessions"]:
            same_key = (ticker, session_date)
            same_ticker_concurrency[same_key] = same_ticker_concurrency.get(same_key, 0) + 1

    overlapping_window_firings = sum(
        1
        for window in windows
        if any(
            same_ticker_concurrency[(window["ticker"], session_date)] > 1
            for session_date in window["sessions"]
        )
    )
    effective_sample_size = sum(
        mean(
            1 / same_ticker_concurrency[(window["ticker"], session_date)]
            for session_date in window["sessions"]
        )
        for window in windows
    )

    return {
        "distinct_tickers": distinct_tickers,
        "overlapping_window_firings": overlapping_window_firings,
        "max_concurrent_same_ticker": max(same_ticker_concurrency.values()),
        "effective_sample_size": effective_sample_size,
    }


def _forward_windows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    window_errors: List[Dict[str, Any]] = []
    for row in rows:
        entry_raw = row.get("entry_session_date")
        exit_raw = row.get("exit_session_date")
        entry = _parse_persisted_session_date(entry_raw)
        exit = _parse_persisted_session_date(exit_raw)
        error: Optional[str] = None
        if entry is None or exit is None:
            error = "missing_entry_or_exit_session"
        elif entry > exit:
            error = "entry_after_exit_session"
        elif not is_us_equity_session(entry):
            error = "entry_session_date_is_not_trading_session"
        elif not is_us_equity_session(exit):
            error = "exit_session_date_is_not_trading_session"

        if error is not None:
            window_errors.append({
                "signal_id": row["signal_id"],
                "observation_id": row["forward_return_observation_id"],
                "ticker": row["ticker"],
                "entry_session_date": entry_raw,
                "exit_session_date": exit_raw,
                "error": error,
            })
            continue

        windows.append({
            "observation_id": row["forward_return_observation_id"],
            "ticker": str(row["ticker"]),
            "sessions": _trading_sessions_inclusive(entry, exit),
        })

    if window_errors:
        raise ScoreboardWindowIntegrityError(
            "graded observations lack valid persisted forward windows",
            window_errors=window_errors,
        )
    return windows


def _parse_persisted_session_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _trading_sessions_inclusive(entry: date, exit: date) -> Tuple[date, ...]:
    sessions: List[date] = []
    cursor = entry
    while cursor <= exit:
        sessions.append(cursor)
        cursor = next_us_equity_session(cursor + timedelta(days=1))
    return tuple(sessions)


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
