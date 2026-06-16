"""Purged, embargoed time-series CV helpers for Stage-1 ML."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence


class CrossValidationError(RuntimeError):
    """The requested purged/embargoed CV split is not feasible."""


@dataclass(frozen=True)
class CVExample:
    signal_id: str
    ticker: str
    security_identity: str
    signal_date: date


@dataclass(frozen=True)
class PurgedEmbargoedFold:
    train_indices: list[int]
    test_indices: list[int]
    test_start_date: date
    test_end_date: date
    embargo_sessions: int
    horizon_sessions: int


def _security_identity_key(row: CVExample) -> str:
    return row.security_identity or row.ticker


def unique_name_weights(examples: Sequence[CVExample]) -> list[float]:
    """Cluster-by-security weighting: each security contributes total weight 1."""

    counts: dict[str, int] = {}
    for row in examples:
        key = _security_identity_key(row)
        counts[key] = counts.get(key, 0) + 1
    return [1.0 / counts[_security_identity_key(row)] for row in examples]


def purged_embargoed_walk_forward_splits(
    examples: Sequence[CVExample],
    *,
    n_splits: int,
    horizon_sessions: int,
    embargo_sessions: int,
) -> list[PurgedEmbargoedFold]:
    """Return expanding walk-forward splits with a date purge before test.

    The train side is strictly earlier than the test block by at least
    ``max(horizon_sessions, embargo_sessions)`` unique signal dates.
    """

    if n_splits <= 0:
        raise CrossValidationError("n_splits must be positive")
    if horizon_sessions < 0 or embargo_sessions < 0:
        raise CrossValidationError("horizon_sessions and embargo_sessions must be >= 0")
    dates = sorted({row.signal_date for row in examples})
    if len(dates) < n_splits + 2:
        raise CrossValidationError(
            "not enough dated cohorts for purged/embargoed walk-forward CV"
        )
    date_to_position = {value: idx for idx, value in enumerate(dates)}
    embargo = max(horizon_sessions, embargo_sessions)
    fold_count = min(n_splits, max(1, len(dates) - embargo - 1))
    warmup_date_count = max(embargo + 1, len(dates) // (fold_count + 1))
    test_dates = dates[warmup_date_count:]
    if len(test_dates) < fold_count:
        raise CrossValidationError("embargo leaves no feasible test cohorts")
    chunk_size = max(1, len(test_dates) // fold_count)
    folds: list[PurgedEmbargoedFold] = []
    for fold_idx in range(fold_count):
        start = fold_idx * chunk_size
        end = len(test_dates) if fold_idx == fold_count - 1 else (fold_idx + 1) * chunk_size
        block = test_dates[start:end]
        if not block:
            continue
        test_positions = {date_to_position[value] for value in block}
        train_cutoff = min(test_positions) - embargo
        test_identities = {
            _security_identity_key(row)
            for row in examples
            if row.signal_date in set(block)
        }
        train_indices = [
            idx
            for idx, row in enumerate(examples)
            if date_to_position[row.signal_date] < train_cutoff
            and _security_identity_key(row) not in test_identities
        ]
        test_indices = [
            idx
            for idx, row in enumerate(examples)
            if row.signal_date in set(block)
        ]
        if not train_indices or not test_indices:
            continue
        folds.append(
            PurgedEmbargoedFold(
                train_indices=train_indices,
                test_indices=test_indices,
                test_start_date=block[0],
                test_end_date=block[-1],
                embargo_sessions=embargo_sessions,
                horizon_sessions=horizon_sessions,
            )
        )
    if not folds:
        raise CrossValidationError("no feasible purged/embargoed CV folds")
    return folds
