"""Measurement scoreboard tests."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import sessionmaker

from alpha.db.models import (
    Base,
    FeatureSnapshot,
    ForwardReturnObservation,
    ForwardReturnObservationEvent,
    ForwardReturnPathRow,
    SignalRegistry,
)
from alpha.jobs import run_measurement_scoreboard
from alpha.jobs.forward_return import (
    NASDAQ_LISTING_SUPPRESSION_REASON,
    REQUIRED_FORWARD_RETURN_STATUSES,
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
from alpha.jobs.measurement_scoreboard import (
    BUCKET_GRADED,
    BUCKET_PENDING_LIKE,
    BUCKET_RETRY_IN_FLIGHT,
    BUCKET_REVIEW_UNRESOLVED,
    BUCKET_TERMINAL_UNAVAILABLE,
    ROLLUP_STATUS_BUCKETS,
    ScoreboardDirectionError,
    ScoreboardPatternIntegrityError,
    ScoreboardPartitionError,
    ScoreboardPoolingError,
    ScoreboardSignalIntegrityError,
    _canonical_observation_rows,
    build_measurement_scoreboard,
    validate_status_partition,
)

SIGNAL_TS = datetime(2026, 6, 1, 20, 0, tzinfo=timezone.utc)


@pytest.fixture()
def db_session_without_fk():
    """SQLite session for deliberate orphan-corruption injection."""

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


def _add_signal(
    db_session,
    signal_id: str,
    *,
    pattern_id: str = "M4",
    ticker: str = "ACME",
    signal_timestamp: datetime = SIGNAL_TS,
    signal_horizon: str = "15d",
) -> SignalRegistry:
    feature = FeatureSnapshot(
        feature_snapshot_id=f"feature-{signal_id}",
        pattern_id=pattern_id,
        ticker=ticker,
        asof_timestamp=SIGNAL_TS,
        feature_json="{}",
        feature_hash=f"feature-hash-{signal_id}",
        data_lineage_ids="[]",
        point_in_time_passed=True,
        lookahead_guard_passed=True,
    )
    signal = SignalRegistry(
        signal_id=signal_id,
        pattern_id=pattern_id,
        ticker=ticker,
        direction="long",
        signal_timestamp=signal_timestamp,
        raw_signal_strength=1.0,
        raw_expected_edge=0.1,
        signal_horizon=signal_horizon,
        thesis_category=pattern_id,
        route_class="base",
        fidelity_tier="test",
        data_confidence=1.0,
        feature_snapshot=feature,
        signal_status="active",
        trading_date="2026-06-01",
        next_execution_session="2026-06-02",
        detector_version="test",
        point_in_time_passed=True,
        lookahead_guard_passed=True,
        data_lineage_ids="[]",
        signal_identity_hash=f"identity-{signal_id}",
        forward_return_status=STATUS_PENDING,
    )
    db_session.add(signal)
    db_session.flush()
    return signal


def _add_observation(
    db_session,
    obs_id: str,
    *,
    status: str,
    forward_return: Optional[float] = None,
    pattern_id: Optional[str] = None,
    ticker: str = "ACME",
    reason: Optional[str] = None,
    mfe: Optional[float] = None,
    mae: Optional[float] = None,
    hit_t1: Optional[bool] = None,
    hit_t2: Optional[bool] = None,
    hit_t3: Optional[bool] = None,
    hit_stop: Optional[bool] = None,
    same_day_ambiguity: Optional[bool] = None,
    signal: Optional[SignalRegistry] = None,
    input_hash: Optional[str] = None,
    created_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None,
    signal_timestamp: datetime = SIGNAL_TS,
    signal_horizon: str = "15d",
    direction: Optional[str] = "long",
) -> ForwardReturnObservation:
    observation_pattern_id = pattern_id
    if observation_pattern_id is None:
        observation_pattern_id = signal.pattern_id if signal is not None else "M4"
    if signal is None:
        signal = _add_signal(
            db_session,
            f"signal-{obs_id}",
            pattern_id=observation_pattern_id,
            ticker=ticker,
            signal_timestamp=signal_timestamp,
            signal_horizon=signal_horizon,
        )
    obs = ForwardReturnObservation(
        forward_return_observation_id=obs_id,
        signal_id=signal.signal_id,
        pattern_id=observation_pattern_id,
        ticker=ticker,
        direction=direction,
        signal_timestamp=signal_timestamp,
        signal_horizon=signal_horizon,
        next_execution_session="2026-06-02",
        entry_session_date="2026-06-02",
        exit_session_date="2026-06-23",
        forward_return=forward_return,
        max_favorable_excursion=mfe,
        max_adverse_excursion=mae,
        mfe_session_date="2026-06-10" if mfe is not None else None,
        mae_session_date="2026-06-11" if mae is not None else None,
        max_close_return=mfe,
        min_close_return=mae,
        hit_t1_intraday=hit_t1,
        hit_t2_intraday=hit_t2,
        hit_t3_intraday=hit_t3,
        hit_stop_intraday=hit_stop,
        same_day_barrier_ambiguity=same_day_ambiguity,
        status=status,
        reason=reason,
        attempts=0,
        input_hash=input_hash or f"input-{obs_id}",
        outcome_hash=f"outcome-{obs_id}",
    )
    if created_at is not None:
        obs.created_at = created_at
    if updated_at is not None:
        obs.updated_at = updated_at
    db_session.add(obs)
    db_session.flush()
    return obs


def _add_orphan_observation(
    db_session,
    obs_id: str,
    *,
    signal_id: str = "missing-parent",
    status: str = STATUS_COMPUTED,
    forward_return: Optional[float] = 0.10,
    pattern_id: str = "M4",
    ticker: str = "ORPH",
    direction: str = "long",
) -> ForwardReturnObservation:
    obs = ForwardReturnObservation(
        forward_return_observation_id=obs_id,
        signal_id=signal_id,
        pattern_id=pattern_id,
        ticker=ticker,
        direction=direction,
        signal_timestamp=SIGNAL_TS,
        signal_horizon="15d",
        next_execution_session="2026-06-02",
        entry_session_date="2026-06-02",
        exit_session_date="2026-06-23",
        forward_return=forward_return,
        status=status,
        attempts=0,
        input_hash=f"input-{obs_id}",
        outcome_hash=f"outcome-{obs_id}",
    )
    db_session.add(obs)
    db_session.flush()
    return obs


def test_status_partition_is_exhaustive_and_disjoint():
    validate_status_partition()
    bucket_members = [status for statuses in ROLLUP_STATUS_BUCKETS.values() for status in statuses]

    assert set(bucket_members) == set(REQUIRED_FORWARD_RETURN_STATUSES)
    assert len(bucket_members) == len(set(bucket_members))
    for status in REQUIRED_FORWARD_RETURN_STATUSES:
        assert sum(status in statuses for statuses in ROLLUP_STATUS_BUCKETS.values()) == 1


def test_unknown_status_fails_loud(db_session):
    _add_observation(db_session, "unknown", status="future_new_status")

    with pytest.raises(ScoreboardPartitionError) as exc_info:
        build_measurement_scoreboard(db_session)

    assert exc_info.value.unknown_status_counts == {"future_new_status": 1}
    assert exc_info.value.unknown_status_details == {
        "future_new_status": [{
            "signal_id": "signal-unknown",
            "observation_id": "unknown",
            "stale": False,
        }]
    }


def test_unknown_status_payload_marks_stale_duplicate_locus(db_session):
    signal = _add_signal(db_session, "signal-with-stale-unknown")
    _add_observation(
        db_session,
        "stale-unknown",
        signal=signal,
        status="future_new_status",
        input_hash="old-input",
        updated_at=SIGNAL_TS + timedelta(minutes=1),
    )
    _add_observation(
        db_session,
        "canonical-valid",
        signal=signal,
        status=STATUS_COMPUTED,
        forward_return=0.20,
        input_hash="new-input",
        updated_at=SIGNAL_TS + timedelta(minutes=2),
    )

    with pytest.raises(ScoreboardPartitionError) as exc_info:
        build_measurement_scoreboard(db_session)

    assert exc_info.value.unknown_status_counts == {"future_new_status": 1}
    assert exc_info.value.unknown_status_details["future_new_status"] == [{
        "signal_id": "signal-with-stale-unknown",
        "observation_id": "stale-unknown",
        "stale": True,
    }]


def test_orphan_signal_integrity_error_fails_loud(db_session_without_fk):
    _add_orphan_observation(
        db_session_without_fk,
        "orphan-computed",
        signal_id="missing-signal",
    )

    with pytest.raises(ScoreboardSignalIntegrityError) as exc_info:
        build_measurement_scoreboard(db_session_without_fk)

    assert exc_info.value.orphans == [{
        "observation_id": "orphan-computed",
        "signal_id": "missing-signal",
        "status": STATUS_COMPUTED,
        "direction": "long",
        "pattern_id": "M4",
    }]


@pytest.mark.parametrize(
    ("obs_id", "status", "direction"),
    [
        ("orphan-unknown-status", "future_new_status", "long"),
        ("orphan-short", STATUS_COMPUTED, "short"),
    ],
)
def test_orphan_signal_integrity_precedes_partition_and_direction(
    db_session_without_fk,
    obs_id,
    status,
    direction,
):
    _add_orphan_observation(
        db_session_without_fk,
        obs_id,
        signal_id=f"missing-{obs_id}",
        status=status,
        direction=direction,
    )

    with pytest.raises(ScoreboardSignalIntegrityError) as exc_info:
        build_measurement_scoreboard(db_session_without_fk)

    assert exc_info.value.orphans == [{
        "observation_id": obs_id,
        "signal_id": f"missing-{obs_id}",
        "status": status,
        "direction": direction,
        "pattern_id": "M4",
    }]


def test_orphan_signal_integrity_runs_on_pattern_filtered_rows(db_session_without_fk):
    _add_orphan_observation(
        db_session_without_fk,
        "orphan-m4",
        signal_id="missing-m4",
        pattern_id="M4",
    )
    _add_observation(
        db_session_without_fk,
        "m5-computed",
        status=STATUS_COMPUTED,
        forward_return=0.40,
        pattern_id="M5",
    )

    with pytest.raises(ScoreboardSignalIntegrityError) as exc_info:
        build_measurement_scoreboard(db_session_without_fk, pattern_id="M4")

    assert exc_info.value.orphans == [{
        "observation_id": "orphan-m4",
        "signal_id": "missing-m4",
        "status": STATUS_COMPUTED,
        "direction": "long",
        "pattern_id": "M4",
    }]

    result = build_measurement_scoreboard(db_session_without_fk, pattern_id="M5")
    assert result.raw_observation_rows == 1
    assert result.total_observations == 1
    assert result.computed_stats.expectancy == pytest.approx(0.40)


def test_short_direction_fails_loud(db_session):
    _add_observation(
        db_session,
        "short-graded",
        status=STATUS_COMPUTED,
        forward_return=-0.10,
        direction="short",
    )

    with pytest.raises(ScoreboardDirectionError) as exc_info:
        build_measurement_scoreboard(db_session)

    assert exc_info.value.unsupported_direction_counts == {"short": 1}
    assert exc_info.value.unsupported_direction_details == {
        "short": [{
            "signal_id": "signal-short-graded",
            "observation_id": "short-graded",
            "status": STATUS_COMPUTED,
            "stale": False,
        }]
    }


def test_short_direction_trips_guard_even_on_stale_duplicate(db_session):
    signal = _add_signal(db_session, "signal-with-stale-short")
    _add_observation(
        db_session,
        "stale-short",
        signal=signal,
        status=STATUS_COMPUTED,
        forward_return=-0.10,
        direction="short",
        input_hash="old-input",
        updated_at=SIGNAL_TS + timedelta(minutes=1),
    )
    _add_observation(
        db_session,
        "canonical-long",
        signal=signal,
        status=STATUS_COMPUTED,
        forward_return=0.20,
        direction="long",
        input_hash="new-input",
        updated_at=SIGNAL_TS + timedelta(minutes=2),
    )

    with pytest.raises(ScoreboardDirectionError) as exc_info:
        build_measurement_scoreboard(db_session)

    assert exc_info.value.unsupported_direction_details["short"] == [{
        "signal_id": "signal-with-stale-short",
        "observation_id": "stale-short",
        "status": STATUS_COMPUTED,
        "stale": True,
    }]


def test_canonical_recency_uses_aware_utc_not_timestamp_strings():
    older_clock_later_string = _scoreboard_row(
        "old",
        updated_at=datetime(2026, 6, 1, 23, 30, tzinfo=timezone(timedelta(hours=14))),
    )
    genuinely_later = _scoreboard_row(
        "new",
        updated_at=datetime(2026, 6, 1, 10, 0),
    )

    [canonical] = _canonical_observation_rows([
        older_clock_later_string,
        genuinely_later,
    ])

    assert canonical["forward_return_observation_id"] == "new"


def test_canonical_recency_tie_is_deterministic_for_aware_and_naive_utc():
    aware = _scoreboard_row(
        "same-a",
        updated_at=datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
    )
    naive = _scoreboard_row(
        "same-b",
        updated_at=datetime(2026, 6, 1, 10, 0),
    )

    [canonical] = _canonical_observation_rows([aware, naive])

    assert canonical["forward_return_observation_id"] == "same-b"


def test_canonical_recency_preserves_microsecond_ordering():
    bare = _scoreboard_row("bare", updated_at=datetime(2026, 6, 1, 10, 0, 0))
    micros = _scoreboard_row(
        "micros",
        updated_at=datetime(2026, 6, 1, 10, 0, 0, 123456),
    )

    [canonical] = _canonical_observation_rows([micros, bare])

    assert canonical["forward_return_observation_id"] == "micros"


def test_canonical_recency_null_timestamps_sort_oldest_and_created_at_breaks_ties():
    missing_updated = _scoreboard_row(
        "missing-updated",
        updated_at=None,
        created_at=datetime(2026, 6, 1, 10, 1),
    )
    present_updated = _scoreboard_row(
        "present-updated",
        updated_at=datetime(2026, 6, 1, 10, 0),
        created_at=datetime(2026, 6, 1, 9, 0),
    )
    [canonical] = _canonical_observation_rows([missing_updated, present_updated])

    assert canonical["forward_return_observation_id"] == "present-updated"

    created_old = _scoreboard_row(
        "created-old",
        updated_at=None,
        created_at=datetime(2026, 6, 1, 9, 0),
    )
    created_new = _scoreboard_row(
        "created-new",
        updated_at=None,
        created_at=datetime(2026, 6, 1, 9, 1),
    )
    [canonical] = _canonical_observation_rows([created_old, created_new])

    assert canonical["forward_return_observation_id"] == "created-new"


def test_one_signal_counts_only_latest_canonical_observation(db_session):
    signal = _add_signal(db_session, "signal-replanned")
    old_ts = SIGNAL_TS + timedelta(minutes=1)
    new_ts = SIGNAL_TS + timedelta(minutes=2)
    _add_observation(
        db_session,
        "old-plan",
        signal=signal,
        status=STATUS_COMPUTED,
        forward_return=-1.0,
        input_hash="old-calendar-plan",
        created_at=old_ts,
        updated_at=old_ts,
    )
    _add_observation(
        db_session,
        "new-plan",
        signal=signal,
        status=STATUS_COMPUTED,
        forward_return=0.50,
        input_hash="new-calendar-plan",
        created_at=new_ts,
        updated_at=new_ts,
    )

    result = build_measurement_scoreboard(db_session)

    assert result.raw_observation_rows == 2
    assert result.total_observations == 1
    assert result.stale_duplicate_observation_rows == 1
    assert result.per_status_counts[STATUS_COMPUTED] == 1
    assert result.rollup_counts[BUCKET_GRADED] == 1
    assert result.computed_stats.n == 1
    assert result.computed_stats.expectancy == pytest.approx(0.50)


def test_suppressed_edgar_review_lands_in_review_bucket_and_not_stats(db_session):
    _add_observation(
        db_session,
        "suppressed-review",
        status=STATUS_SURVIVORSHIP_UNRESOLVED_REVIEW,
        reason=NASDAQ_LISTING_SUPPRESSION_REASON,
    )

    result = build_measurement_scoreboard(db_session)

    assert result.rollup_counts[BUCKET_REVIEW_UNRESOLVED] == 1
    assert result.per_status_counts[STATUS_SURVIVORSHIP_UNRESOLVED_REVIEW] == 1
    assert result.computed_stats.n == 0
    assert result.computed_stats.no_graded_firings is True


def test_corporate_action_review_is_distinct_from_terminal_unavailable(db_session):
    _add_observation(db_session, "corp-review", status=STATUS_CORPORATE_ACTION_REVIEW)
    _add_observation(db_session, "terminal", status=STATUS_OUTCOME_UNAVAILABLE)

    result = build_measurement_scoreboard(db_session)

    assert result.rollup_counts[BUCKET_REVIEW_UNRESOLVED] == 1
    assert result.rollup_counts[BUCKET_TERMINAL_UNAVAILABLE] == 1
    assert result.computed_stats.n == 0


def test_anomalies_are_surfaced_and_excluded_from_stats(db_session):
    _add_observation(db_session, "computed-null", status=STATUS_COMPUTED, forward_return=None)
    _add_observation(db_session, "pending-with-return", status=STATUS_PENDING, forward_return=0.50)
    _add_observation(db_session, "valid-computed", status=STATUS_COMPUTED, forward_return=0.20)

    result = build_measurement_scoreboard(db_session)

    assert result.anomalies.computed_missing_forward_return == 1
    assert result.anomalies.non_computed_with_forward_return == 1
    assert result.anomalies.computed_missing_forward_return_by_pattern == {"M4": 1}
    assert result.anomalies.computed_missing_forward_return_ids == ("computed-null",)
    assert result.anomalies.non_computed_with_forward_return_ids == ("pending-with-return",)
    assert result.computed_stats.n == 1
    assert result.computed_stats.expectancy == pytest.approx(0.20)
    assert result.rollup_counts[BUCKET_GRADED] == 2
    assert result.graded_rollup_reconciliation.graded_rollup_count == 2
    assert result.graded_rollup_reconciliation.computed_sample_n == 1
    assert result.graded_rollup_reconciliation.computed_missing_forward_return == 1
    assert result.graded_rollup_reconciliation.reconciles is True


def test_nonfinite_computed_return_uses_missing_return_anomaly(db_session):
    _add_observation(
        db_session,
        "computed-inf",
        status=STATUS_COMPUTED,
        forward_return=float("inf"),
    )
    _add_observation(
        db_session,
        "valid-computed",
        status=STATUS_COMPUTED,
        forward_return=0.20,
    )

    result = build_measurement_scoreboard(db_session)

    assert result.anomalies.computed_missing_forward_return == 1
    assert result.anomalies.computed_missing_forward_return_by_pattern == {"M4": 1}
    assert result.anomalies.computed_missing_forward_return_ids == ("computed-inf",)
    assert result.computed_stats.n == 1
    assert result.computed_stats.expectancy == pytest.approx(0.20)
    assert result.rollup_counts[BUCKET_GRADED] == 2
    assert result.graded_rollup_reconciliation.graded_rollup_count == 2
    assert result.graded_rollup_reconciliation.computed_sample_n == 1
    assert result.graded_rollup_reconciliation.computed_missing_forward_return == 1
    assert result.graded_rollup_reconciliation.reconciles is True


def test_anomaly_id_truncation_is_stable_by_signal_timestamp(db_session):
    expected_first_twenty = []
    for index in range(25):
        obs_id = f"{25 - index:02d}-computed-null"
        if index < 20:
            expected_first_twenty.append(obs_id)
        _add_observation(
            db_session,
            obs_id,
            status=STATUS_COMPUTED,
            forward_return=None,
            signal_timestamp=SIGNAL_TS + timedelta(minutes=index),
        )

    result = build_measurement_scoreboard(db_session)

    assert result.anomalies.computed_missing_forward_return == 25
    assert result.anomalies.computed_missing_forward_return_ids == tuple(
        expected_first_twenty
    )


def test_partition_integrity_excludes_pending_retry_and_review_from_stats(db_session):
    _add_observation(db_session, "pending", status=STATUS_PENDING)
    _add_observation(db_session, "retry", status=STATUS_PRICING_UNAVAILABLE_RETRY)
    _add_observation(db_session, "review", status=STATUS_PRICE_DRIFT_REVIEW)
    _add_observation(db_session, "computed-a", status=STATUS_COMPUTED, forward_return=0.10)
    _add_observation(db_session, "computed-b", status=STATUS_COMPUTED, forward_return=0.20)

    result = build_measurement_scoreboard(db_session)

    assert result.rollup_counts[BUCKET_PENDING_LIKE] == 1
    assert result.rollup_counts[BUCKET_RETRY_IN_FLIGHT] == 1
    assert result.rollup_counts[BUCKET_REVIEW_UNRESOLVED] == 1
    assert result.rollup_counts[BUCKET_GRADED] == 2
    assert result.computed_stats.n == 2
    assert result.computed_stats.expectancy == pytest.approx(0.15)


def test_three_return_bins_and_barrier_stats(db_session):
    _add_observation(
        db_session,
        "win",
        status=STATUS_COMPUTED,
        forward_return=0.20,
        mfe=0.35,
        mae=-0.02,
        hit_t1=True,
        hit_t2=True,
        same_day_ambiguity=True,
    )
    _add_observation(
        db_session,
        "flat",
        status=STATUS_COMPUTED,
        forward_return=0.0,
        mfe=0.05,
        mae=-0.01,
        hit_t1=False,
    )
    _add_observation(
        db_session,
        "loss",
        status=STATUS_COMPUTED,
        forward_return=-0.10,
        mfe=0.02,
        mae=-0.20,
        hit_stop=True,
    )

    result = build_measurement_scoreboard(db_session)
    stats = result.computed_stats

    assert stats.n == 3
    assert stats.win_count == 1
    assert stats.flat_count == 1
    assert stats.loss_count == 1
    assert stats.avg_win == pytest.approx(0.20)
    assert stats.avg_loss == pytest.approx(-0.10)
    assert stats.win_loss_ratio == pytest.approx(2.0)
    assert stats.hit_t1_count == 1
    assert stats.hit_t1_rate == pytest.approx(1 / 3)
    assert stats.hit_t2_count == 1
    assert stats.hit_stop_count == 1
    assert stats.same_day_barrier_ambiguity_count == 1
    assert stats.mfe_mean == pytest.approx((0.35 + 0.05 + 0.02) / 3)
    assert stats.mfe_median == pytest.approx(0.05)
    assert stats.mfe_max == pytest.approx(0.35)
    assert stats.mae_mean == pytest.approx((-0.02 - 0.01 - 0.20) / 3)
    assert stats.mae_median == pytest.approx(-0.02)
    assert stats.mae_worst == pytest.approx(-0.20)


@pytest.mark.parametrize(
    ("returns", "expected_avg_win", "expected_avg_loss", "expected_ratio"),
    [
        ([0.10, 0.20], 0.15, None, None),
        ([-0.10, -0.20], None, -0.15, None),
    ],
)
def test_all_win_and_all_loss_do_not_explode(
    db_session,
    returns,
    expected_avg_win,
    expected_avg_loss,
    expected_ratio,
):
    for index, value in enumerate(returns):
        _add_observation(
            db_session,
            f"computed-{index}",
            status=STATUS_COMPUTED,
            forward_return=value,
        )

    stats = build_measurement_scoreboard(db_session).computed_stats

    assert stats.avg_win == pytest.approx(expected_avg_win) if expected_avg_win is not None else stats.avg_win is None
    assert stats.avg_loss == pytest.approx(expected_avg_loss) if expected_avg_loss is not None else stats.avg_loss is None
    assert stats.win_loss_ratio == expected_ratio


def test_tail_counter_threshold_is_inclusive(db_session):
    _add_observation(db_session, "below", status=STATUS_COMPUTED, forward_return=0.01, mfe=0.249)
    _add_observation(db_session, "boundary", status=STATUS_COMPUTED, forward_return=0.02, mfe=0.25)
    _add_observation(db_session, "above", status=STATUS_COMPUTED, forward_return=0.03, mfe=0.30)

    stats = build_measurement_scoreboard(db_session, mfe_tail_threshold=0.25).computed_stats

    assert stats.tail_event_count == 2
    assert stats.tail_event_fraction == pytest.approx(2 / 3)


def test_nonfinite_mfe_and_mae_do_not_poison_stats_or_tail_counter(db_session):
    _add_observation(
        db_session,
        "clean",
        status=STATUS_COMPUTED,
        forward_return=0.10,
        mfe=0.30,
        mae=-0.10,
    )
    _add_observation(
        db_session,
        "nonfinite-excursions",
        status=STATUS_COMPUTED,
        forward_return=0.20,
        mfe=float("inf"),
        mae=float("-inf"),
    )

    stats = build_measurement_scoreboard(
        db_session,
        mfe_tail_threshold=0.25,
    ).computed_stats

    assert stats.n == 2
    assert stats.mfe_max == pytest.approx(0.30)
    assert stats.mfe_mean == pytest.approx(0.30)
    assert stats.mae_worst == pytest.approx(-0.10)
    assert stats.mae_mean == pytest.approx(-0.10)
    assert stats.tail_event_count == 1
    assert stats.tail_event_fraction == pytest.approx(1 / 2)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.01])
def test_tail_threshold_must_be_finite_non_negative(db_session, threshold):
    with pytest.raises(
        ValueError,
        match="mfe_tail_threshold must be a finite, non-negative fraction",
    ):
        build_measurement_scoreboard(db_session, mfe_tail_threshold=threshold)


def test_tail_threshold_accepts_fraction(db_session):
    result = build_measurement_scoreboard(db_session, mfe_tail_threshold=0.25)

    assert result.mfe_tail_threshold == pytest.approx(0.25)

@pytest.mark.parametrize("threshold", ["nan", "inf"])
def test_json_error_path_rejects_nonfinite_tail_threshold(
    tmp_path,
    monkeypatch,
    capsys,
    threshold,
):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    db_url = _seed_file_database(
        tmp_path,
        f"threshold-{threshold}.db",
        [("computed", STATUS_COMPUTED, 0.10)],
    )

    rc = run_measurement_scoreboard.main([
        "--live",
        "--database-url",
        db_url,
        "--mfe-tail-threshold",
        threshold,
        "--json",
    ])
    captured = capsys.readouterr()

    assert rc == 1
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"
    assert payload["message"] == "mfe_tail_threshold must be a finite, non-negative fraction"


@pytest.mark.parametrize("threshold", ["nan", "inf"])
def test_human_error_path_rejects_nonfinite_tail_threshold(
    tmp_path,
    monkeypatch,
    capsys,
    threshold,
):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    db_url = _seed_file_database(
        tmp_path,
        f"threshold-{threshold}-human.db",
        [("computed", STATUS_COMPUTED, 0.10)],
    )

    rc = run_measurement_scoreboard.main([
        "--live",
        "--database-url",
        db_url,
        "--mfe-tail-threshold",
        threshold,
    ])
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out == (
        "ERROR: mfe_tail_threshold must be a finite, non-negative fraction\n"
    )


def test_zero_graded_is_explicit_and_has_no_nan(db_session):
    _add_observation(db_session, "pending", status=STATUS_PENDING)

    stats = build_measurement_scoreboard(db_session).computed_stats

    assert stats.n == 0
    assert stats.no_graded_firings is True
    assert stats.expectancy is None
    assert stats.tail_event_fraction is None
    numeric_values = [
        value
        for value in stats.__dict__.values()
        if isinstance(value, float)
    ]
    assert all(not math.isnan(value) for value in numeric_values)


def test_all_required_statuses_have_counts_even_when_zero(db_session):
    _add_observation(db_session, "one", status=STATUS_PENDING)

    result = build_measurement_scoreboard(db_session)

    assert list(result.per_status_counts) == list(REQUIRED_FORWARD_RETURN_STATUSES)
    assert result.per_status_counts[STATUS_PENDING] == 1
    for status in REQUIRED_FORWARD_RETURN_STATUSES:
        assert status in result.per_status_counts


def test_pattern_filter_limits_rows_without_cross_tabs(db_session):
    _add_observation(db_session, "m4-computed", status=STATUS_COMPUTED, forward_return=0.10, pattern_id="M4")
    _add_observation(db_session, "m5-computed", status=STATUS_COMPUTED, forward_return=0.40, pattern_id="M5")

    result = build_measurement_scoreboard(db_session, pattern_id="M4")

    assert result.pattern_id == "M4"
    assert result.total_observations == 1
    assert result.computed_stats.expectancy == pytest.approx(0.10)


@pytest.mark.parametrize("pattern_id", ["", "   "])
def test_pattern_filter_rejects_empty_or_whitespace_pattern_id(db_session, pattern_id):
    with pytest.raises(ValueError, match="pattern_id must be a non-empty string"):
        build_measurement_scoreboard(db_session, pattern_id=pattern_id)


def test_single_graded_pattern_all_patterns_passes(db_session):
    _add_observation(
        db_session,
        "m4-a",
        status=STATUS_COMPUTED,
        forward_return=0.10,
        pattern_id="M4",
        signal_horizon="15d",
    )
    _add_observation(
        db_session,
        "m4-b",
        status=STATUS_COMPUTED,
        forward_return=0.30,
        pattern_id="M4",
        signal_horizon="15d",
    )

    result = build_measurement_scoreboard(db_session)

    assert result.computed_stats.n == 2
    assert result.computed_stats.expectancy == pytest.approx(0.20)


def test_pattern_integrity_error_refuses_mislabeled_graded_observation(db_session):
    signal = _add_signal(
        db_session,
        "mislabeled-parent",
        pattern_id="M5",
        signal_horizon="7d",
    )
    _add_observation(
        db_session,
        "mislabeled-observation",
        signal=signal,
        status=STATUS_COMPUTED,
        forward_return=0.20,
        pattern_id="M4",
        signal_horizon="15d",
    )

    with pytest.raises(ScoreboardPatternIntegrityError) as exc_info:
        build_measurement_scoreboard(db_session)

    assert exc_info.value.mismatches == [{
        "signal_id": "mislabeled-parent",
        "observation_id": "mislabeled-observation",
        "observation_pattern_id": "M4",
        "signal_pattern_id": "M5",
    }]


def test_pattern_integrity_error_refuses_mislabeled_filtered_observation(db_session):
    signal = _add_signal(
        db_session,
        "mislabeled-filtered-parent",
        pattern_id="M5",
        signal_horizon="7d",
    )
    _add_observation(
        db_session,
        "mislabeled-filtered-observation",
        signal=signal,
        status=STATUS_COMPUTED,
        forward_return=0.20,
        pattern_id="M4",
        signal_horizon="15d",
    )

    with pytest.raises(ScoreboardPatternIntegrityError):
        build_measurement_scoreboard(db_session, pattern_id="M4")
    with pytest.raises(ScoreboardPatternIntegrityError):
        build_measurement_scoreboard(db_session, pattern_id="M5")


def test_multiple_graded_patterns_raise_pooling_error(db_session):
    _add_observation(
        db_session,
        "m4-computed",
        status=STATUS_COMPUTED,
        forward_return=0.10,
        pattern_id="M4",
        signal_horizon="15d",
    )
    _add_observation(
        db_session,
        "m5-computed",
        status=STATUS_COMPUTED,
        forward_return=0.40,
        pattern_id="M5",
        signal_horizon="7d",
    )

    with pytest.raises(ScoreboardPoolingError) as exc_info:
        build_measurement_scoreboard(db_session)

    assert "--pattern-id" in str(exc_info.value)
    assert exc_info.value.pattern_counts == {"M4": 1, "M5": 1}
    assert exc_info.value.pattern_horizons == {"M4": "15d", "M5": "7d"}


def test_pattern_filter_isolates_one_graded_pattern_after_pooling_guard(db_session):
    _add_observation(
        db_session,
        "m4-computed",
        status=STATUS_COMPUTED,
        forward_return=0.10,
        pattern_id="M4",
        signal_horizon="15d",
    )
    _add_observation(
        db_session,
        "m5-computed",
        status=STATUS_COMPUTED,
        forward_return=0.40,
        pattern_id="M5",
        signal_horizon="7d",
    )

    result = build_measurement_scoreboard(db_session, pattern_id="M4")

    assert result.pattern_id == "M4"
    assert result.total_observations == 1
    assert result.computed_stats.n == 1
    assert result.computed_stats.expectancy == pytest.approx(0.10)


def test_pooling_guard_ignores_non_graded_patterns(db_session):
    _add_observation(
        db_session,
        "m4-computed",
        status=STATUS_COMPUTED,
        forward_return=0.10,
        pattern_id="M4",
        signal_horizon="15d",
    )
    _add_observation(
        db_session,
        "m5-pending",
        status=STATUS_PENDING,
        pattern_id="M5",
        signal_horizon="7d",
    )

    result = build_measurement_scoreboard(db_session)

    assert result.total_observations == 2
    assert result.rollup_counts[BUCKET_GRADED] == 1
    assert result.rollup_counts[BUCKET_PENDING_LIKE] == 1
    assert result.computed_stats.n == 1
    assert result.computed_stats.expectancy == pytest.approx(0.10)


def test_pooling_guard_uses_finite_graded_sample_only(db_session):
    _add_observation(
        db_session,
        "m4-computed",
        status=STATUS_COMPUTED,
        forward_return=0.10,
        pattern_id="M4",
        signal_horizon="15d",
    )
    _add_observation(
        db_session,
        "m5-nonfinite",
        status=STATUS_COMPUTED,
        forward_return=float("inf"),
        pattern_id="M5",
        signal_horizon="7d",
    )

    result = build_measurement_scoreboard(db_session)

    assert result.rollup_counts[BUCKET_GRADED] == 2
    assert result.anomalies.computed_missing_forward_return == 1
    assert result.anomalies.computed_missing_forward_return_by_pattern == {"M5": 1}
    assert result.computed_stats.n == 1
    assert result.computed_stats.expectancy == pytest.approx(0.10)
    assert result.graded_rollup_reconciliation.reconciles is True


def test_nonfinite_mislabeled_observation_is_attributed_by_parent_pattern(db_session):
    signal = _add_signal(
        db_session,
        "mislabeled-nonfinite-parent",
        pattern_id="M5",
        signal_horizon="7d",
    )
    _add_observation(
        db_session,
        "m4-computed",
        status=STATUS_COMPUTED,
        forward_return=0.10,
        pattern_id="M4",
        signal_horizon="15d",
    )
    _add_observation(
        db_session,
        "mislabeled-nonfinite-observation",
        signal=signal,
        status=STATUS_COMPUTED,
        forward_return=float("inf"),
        pattern_id="M4",
        signal_horizon="15d",
    )

    result = build_measurement_scoreboard(db_session)

    assert result.computed_stats.n == 1
    assert result.computed_stats.expectancy == pytest.approx(0.10)
    assert result.anomalies.computed_missing_forward_return == 1
    assert result.anomalies.computed_missing_forward_return_by_pattern == {"M5": 1}


def test_json_error_path_is_parseable_and_human_error_remains_plain_text(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    db_url = _seed_file_database(
        tmp_path,
        "unknown.db",
        [("unknown-status", "future_new_status", None)],
    )

    rc = run_measurement_scoreboard.main(["--live", "--database-url", db_url, "--json"])
    captured = capsys.readouterr()

    assert rc == 1
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert payload["error_type"] == "ScoreboardPartitionError"
    assert payload["unknown_status_counts"] == {"future_new_status": 1}
    assert payload["unknown_status_details"] == {
        "future_new_status": [{
            "signal_id": "signal-unknown-status",
            "observation_id": "unknown-status",
            "stale": False,
        }]
    }

    rc = run_measurement_scoreboard.main(["--live", "--database-url", db_url])
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out.startswith("ERROR: unknown forward-return statuses present")


def test_json_error_path_surfaces_direction_guard(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    path = tmp_path / "short.db"
    url = f"sqlite:///{path}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        _add_observation(
            session,
            "short-obs",
            status=STATUS_COMPUTED,
            forward_return=-0.10,
            direction="short",
        )
        session.commit()
    finally:
        session.close()
        engine.dispose()

    rc = run_measurement_scoreboard.main(["--live", "--database-url", url, "--json"])
    captured = capsys.readouterr()

    assert rc == 1
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert payload["error_type"] == "ScoreboardDirectionError"
    assert payload["unsupported_direction_counts"] == {"short": 1}


def test_json_error_path_surfaces_pattern_pooling_guard(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    path = tmp_path / "pooling.db"
    url = f"sqlite:///{path}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        _add_observation(
            session,
            "m4-computed",
            status=STATUS_COMPUTED,
            forward_return=0.10,
            pattern_id="M4",
            signal_horizon="15d",
        )
        _add_observation(
            session,
            "m5-computed",
            status=STATUS_COMPUTED,
            forward_return=0.40,
            pattern_id="M5",
            signal_horizon="7d",
        )
        session.commit()
    finally:
        session.close()
        engine.dispose()

    rc = run_measurement_scoreboard.main(["--live", "--database-url", url, "--json"])
    captured = capsys.readouterr()

    assert rc == 1
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert payload["error_type"] == "ScoreboardPoolingError"
    assert payload["pattern_counts"] == {"M4": 1, "M5": 1}
    assert payload["pattern_horizons"] == {"M4": "15d", "M5": "7d"}


def test_human_error_path_surfaces_pattern_pooling_details(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    path = tmp_path / "pooling-human.db"
    url = f"sqlite:///{path}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        _add_observation(
            session,
            "m4-computed",
            status=STATUS_COMPUTED,
            forward_return=0.10,
            pattern_id="M4",
            signal_horizon="15d",
        )
        _add_observation(
            session,
            "m5-computed",
            status=STATUS_COMPUTED,
            forward_return=0.40,
            pattern_id="M5",
            signal_horizon="7d",
        )
        session.commit()
    finally:
        session.close()
        engine.dispose()

    rc = run_measurement_scoreboard.main(["--live", "--database-url", url])
    captured = capsys.readouterr()

    assert rc == 1
    assert "ERROR: scoreboard headline stats cannot pool" in captured.out
    assert 'Pattern counts: {"M4": 1, "M5": 1}' in captured.out
    assert 'Pattern horizons: {"M4": "15d", "M5": "7d"}' in captured.out


def test_json_and_human_error_paths_surface_pattern_integrity_guard(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    path = tmp_path / "integrity.db"
    url = f"sqlite:///{path}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        signal = _add_signal(
            session,
            "mislabeled-parent",
            pattern_id="M5",
            signal_horizon="7d",
        )
        _add_observation(
            session,
            "mislabeled-observation",
            signal=signal,
            status=STATUS_COMPUTED,
            forward_return=0.20,
            pattern_id="M4",
            signal_horizon="15d",
        )
        session.commit()
    finally:
        session.close()
        engine.dispose()

    expected = [{
        "signal_id": "mislabeled-parent",
        "observation_id": "mislabeled-observation",
        "observation_pattern_id": "M4",
        "signal_pattern_id": "M5",
    }]

    rc = run_measurement_scoreboard.main(["--live", "--database-url", url, "--json"])
    captured = capsys.readouterr()

    assert rc == 1
    payload = json.loads(captured.out)
    assert payload["error_type"] == "ScoreboardPatternIntegrityError"
    assert payload["mismatches"] == expected

    rc = run_measurement_scoreboard.main(["--live", "--database-url", url])
    captured = capsys.readouterr()

    assert rc == 1
    assert "Pattern mismatches:" in captured.out
    assert '"observation_pattern_id": "M4"' in captured.out
    assert '"signal_pattern_id": "M5"' in captured.out


def test_json_and_human_error_paths_surface_signal_integrity_guard(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    path = tmp_path / "signal-integrity.db"
    url = f"sqlite:///{path}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        _add_orphan_observation(
            session,
            "orphan-cli",
            signal_id="missing-cli",
            status=STATUS_COMPUTED,
            direction="long",
            pattern_id="M4",
        )
        session.commit()
    finally:
        session.close()
        engine.dispose()

    expected = [{
        "observation_id": "orphan-cli",
        "signal_id": "missing-cli",
        "status": STATUS_COMPUTED,
        "direction": "long",
        "pattern_id": "M4",
    }]

    rc = run_measurement_scoreboard.main(["--live", "--database-url", url, "--json"])
    captured = capsys.readouterr()

    assert rc == 1
    payload = json.loads(captured.out)
    assert payload["error_type"] == "ScoreboardSignalIntegrityError"
    assert payload["orphans"] == expected

    rc = run_measurement_scoreboard.main(["--live", "--database-url", url])
    captured = capsys.readouterr()

    assert rc == 1
    assert "Orphan observations:" in captured.out
    assert '"observation_id": "orphan-cli"' in captured.out
    assert '"signal_id": "missing-cli"' in captured.out


def test_cli_rejects_empty_pattern_id(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    db_url = _seed_file_database(
        tmp_path,
        "empty-pattern.db",
        [("computed", STATUS_COMPUTED, 0.10)],
    )

    rc = run_measurement_scoreboard.main([
        "--live",
        "--database-url",
        db_url,
        "--pattern-id",
        "",
        "--json",
    ])
    captured = capsys.readouterr()

    assert rc == 1
    payload = json.loads(captured.out)
    assert payload["error_type"] == "ValueError"
    assert payload["message"] == "pattern_id must be a non-empty string"


@pytest.mark.parametrize(
    ("threshold", "should_warn"),
    [(0.25, False), (1.0, False), (25.0, True)],
)
def test_tail_threshold_warning_is_soft(
    tmp_path,
    monkeypatch,
    capsys,
    threshold,
    should_warn,
):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    db_url = _seed_file_database(
        tmp_path,
        f"threshold-{threshold}.db",
        [("computed", STATUS_COMPUTED, 0.10)],
    )

    rc = run_measurement_scoreboard.main([
        "--live",
        "--database-url",
        db_url,
        "--mfe-tail-threshold",
        str(threshold),
        "--json",
    ])
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out)["mfe_tail_threshold"] == threshold
    if should_warn:
        assert "25 == +2500% - pass 0.25 for +25%" in captured.err
    else:
        assert captured.err == ""


def test_cli_database_override_does_not_leak_between_in_process_calls(
    tmp_path,
    monkeypatch,
    capsys,
):
    baseline_url = _seed_file_database(
        tmp_path,
        "baseline.db",
        [("baseline", STATUS_COMPUTED, 0.20)],
    )
    override_url = _seed_file_database(
        tmp_path,
        "override.db",
        [("override", STATUS_COMPUTED, 0.80)],
    )
    monkeypatch.setenv("DATABASE_URL", baseline_url)
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)

    rc = run_measurement_scoreboard.main([
        "--live",
        "--database-url",
        override_url,
        "--json",
    ])
    first = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert first["computed_stats"]["expectancy"] == pytest.approx(0.80)

    rc = run_measurement_scoreboard.main(["--live", "--json"])
    second = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert second["computed_stats"]["expectancy"] == pytest.approx(0.20)


def test_each_required_status_maps_to_one_rollup_bucket():
    status_to_bucket = {}
    for bucket, statuses in ROLLUP_STATUS_BUCKETS.items():
        for status in statuses:
            assert status not in status_to_bucket
            status_to_bucket[status] = bucket

    assert set(status_to_bucket) == set(REQUIRED_FORWARD_RETURN_STATUSES)
    assert status_to_bucket[STATUS_PENDING] == BUCKET_PENDING_LIKE
    assert status_to_bucket[STATUS_HALTED_PENDING] == BUCKET_PENDING_LIKE
    assert status_to_bucket[STATUS_PRICE_FINALITY_PENDING] == BUCKET_PENDING_LIKE
    assert status_to_bucket[STATUS_MISSING_ENTRY_PRICE_RETRY] == BUCKET_RETRY_IN_FLIGHT
    assert status_to_bucket[STATUS_MISSING_EXIT_PRICE_RETRY] == BUCKET_RETRY_IN_FLIGHT
    assert status_to_bucket[STATUS_INVALID_ENTRY_PRICE_RETRY] == BUCKET_RETRY_IN_FLIGHT
    assert status_to_bucket[STATUS_INVALID_EXIT_PRICE_RETRY] == BUCKET_RETRY_IN_FLIGHT
    assert status_to_bucket[STATUS_PROVIDER_REVISION_REVIEW] == BUCKET_REVIEW_UNRESOLVED
    assert status_to_bucket[STATUS_SURVIVORSHIP_UNRESOLVED_REVIEW] == BUCKET_REVIEW_UNRESOLVED
    assert status_to_bucket[STATUS_CORPORATE_ACTION_REVIEW] == BUCKET_REVIEW_UNRESOLVED
    assert status_to_bucket[STATUS_OUTCOME_UNAVAILABLE] == BUCKET_TERMINAL_UNAVAILABLE


def test_scoreboard_query_is_read_only(db_session):
    _add_observation(db_session, "computed", status=STATUS_COMPUTED, forward_return=0.10)
    before_counts = _forward_return_table_counts(db_session)
    mutating_statements = []

    def _record_mutation(conn, cursor, statement, parameters, context, executemany):
        first = statement.strip().split(None, 1)[0].upper() if statement.strip() else ""
        if first in {"INSERT", "UPDATE", "DELETE", "UPSERT", "ALTER", "DROP", "CREATE"}:
            mutating_statements.append(statement)

    event.listen(db_session.bind, "before_cursor_execute", _record_mutation)
    try:
        result = build_measurement_scoreboard(db_session)
    finally:
        event.remove(db_session.bind, "before_cursor_execute", _record_mutation)

    assert result.computed_stats.n == 1
    assert mutating_statements == []
    assert _forward_return_table_counts(db_session) == before_counts
    assert not db_session.new
    assert not db_session.dirty


def _forward_return_table_counts(db_session) -> dict[str, int]:
    return {
        "forward_return_observations": db_session.scalar(
            select(func.count()).select_from(ForwardReturnObservation)
        ),
        "forward_return_observation_events": db_session.scalar(
            select(func.count()).select_from(ForwardReturnObservationEvent)
        ),
        "forward_return_path_rows": db_session.scalar(
            select(func.count()).select_from(ForwardReturnPathRow)
        ),
    }


def _scoreboard_row(
    observation_id: str,
    *,
    updated_at: Optional[datetime],
    created_at: Optional[datetime] = None,
) -> dict:
    return {
        "forward_return_observation_id": observation_id,
        "signal_id": "same-signal",
        "input_hash": f"input-{observation_id}",
        "status": STATUS_COMPUTED,
        "forward_return": 0.10,
        "max_close_return": None,
        "min_close_return": None,
        "max_favorable_excursion": None,
        "max_adverse_excursion": None,
        "mfe_session_date": None,
        "mae_session_date": None,
        "hit_t1_intraday": None,
        "hit_t2_intraday": None,
        "hit_t3_intraday": None,
        "hit_stop_intraday": None,
        "same_day_barrier_ambiguity": None,
        "pattern_id": "M4",
        "signal_pattern_id": "M4",
        "ticker": "ACME",
        "direction": "long",
        "signal_horizon": "15d",
        "signal_timestamp": SIGNAL_TS,
        "entry_session_date": "2026-06-02",
        "exit_session_date": "2026-06-23",
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _seed_file_database(tmp_path, filename: str, rows) -> str:
    path = tmp_path / filename
    url = f"sqlite:///{path}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        for obs_id, status, forward_return in rows:
            _add_observation(
                session,
                obs_id,
                status=status,
                forward_return=forward_return,
            )
        session.commit()
    finally:
        session.close()
        engine.dispose()
    return url
