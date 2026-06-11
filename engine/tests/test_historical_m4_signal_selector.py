from __future__ import annotations

import json
from datetime import date, datetime, timezone

from sqlalchemy import event

from alpha.db.models import EvidenceJob, EvidenceJobRun, SignalRegistry
from alpha.evidence.writer import record_feature_snapshot, record_signal
from alpha.jobs import historical_m4_signal_selector as selector
from alpha.jobs.historical_m4_signal_selector import (
    HISTORICAL_M4_REPLAY_RECONSTRUCTION_METHOD,
    SIGNAL_SOURCE_HISTORICAL_M4_REPLAY,
    SIGNAL_SOURCE_LIVE,
    apply_signal_source_filter,
    historical_m4_replay_signal_query,
)


SIGNAL_DAY = date(2026, 6, 2)
SIGNAL_TS = datetime(2026, 6, 2, 20, 0, tzinfo=timezone.utc)


def _add_signal(db_session, ticker: str, features: dict) -> str:
    feature = record_feature_snapshot(
        db_session,
        pattern_id="M4",
        ticker=ticker,
        asof_timestamp=SIGNAL_TS,
        features=features,
        data_lineage_ids=[],
    )
    signal = record_signal(
        db_session,
        pattern_id="M4",
        ticker=ticker,
        direction="long",
        signal_timestamp=SIGNAL_TS,
        raw_signal_strength=1.0,
        raw_expected_edge=0.01,
        feature_snapshot_id=feature.feature_snapshot_id,
        signal_horizon="15d",
        trading_date=SIGNAL_DAY.isoformat(),
        next_execution_session="2026-06-03",
        signal_identity_hash=f"m4-{ticker.lower()}-{SIGNAL_DAY.isoformat()}",
    )
    return signal.signal_id


def _record_range_replay_reuse(db_session, signal_id: str) -> None:
    job = EvidenceJob(
        job_id="historical-m4-range-replay-job",
        job_name="historical_m4_range_replay",
        job_type="historical_replay",
        owner_component="historical_m4",
    )
    db_session.add(job)
    db_session.add(
        EvidenceJobRun(
            job_run_id="historical-m4-range-replay-run",
            job_id=job.job_id,
            run_status="finished",
            started_at=SIGNAL_TS,
            ended_at=SIGNAL_TS,
            metric_json=json.dumps(
                {
                    "date_results": [
                        {
                            "replay_date": SIGNAL_DAY.isoformat(),
                            "orchestration": {
                                "reused_signal_ids": [signal_id],
                            },
                        }
                    ]
                },
                sort_keys=True,
            ),
        )
    )
    db_session.flush()


def test_historical_m4_replay_selector_includes_stamped_and_reused_members_only(db_session):
    top_level_id = _add_signal(
        db_session,
        "HIST",
        {
            "decision_date": SIGNAL_DAY.isoformat(),
            "reconstruction_method": HISTORICAL_M4_REPLAY_RECONSTRUCTION_METHOD,
        },
    )
    nested_id = _add_signal(
        db_session,
        "NEST",
        {
            "decision_date": SIGNAL_DAY.isoformat(),
            "historical_replay": {
                "reconstruction_method": HISTORICAL_M4_REPLAY_RECONSTRUCTION_METHOD,
            },
        },
    )
    reused_live_id = _add_signal(
        db_session,
        "REUSE",
        {"decision_date": SIGNAL_DAY.isoformat(), "source": "live"},
    )
    stale_live_id = _add_signal(
        db_session,
        "RKTO",
        {"decision_date": SIGNAL_DAY.isoformat(), "source": "live"},
    )
    _record_range_replay_reuse(db_session, reused_live_id)

    selected = {
        signal.signal_id
        for signal in historical_m4_replay_signal_query(
            db_session,
            signal_start_date=SIGNAL_DAY,
            signal_end_date=SIGNAL_DAY,
        ).all()
    }

    assert selected == {top_level_id, nested_id, reused_live_id}
    assert stale_live_id not in selected

    default_live = {
        signal.signal_id
        for signal in apply_signal_source_filter(
            db_session.query(SignalRegistry),
            db_session,
            signal_source=SIGNAL_SOURCE_LIVE,
            signal_start_date=SIGNAL_DAY,
            signal_end_date=SIGNAL_DAY,
        ).all()
    }
    assert default_live == {top_level_id, nested_id, reused_live_id, stale_live_id}


def test_historical_m4_replay_selector_stages_large_membership(
    db_session,
    monkeypatch,
):
    selected_id = _add_signal(
        db_session,
        "HIST",
        {
            "decision_date": SIGNAL_DAY.isoformat(),
            "reconstruction_method": HISTORICAL_M4_REPLAY_RECONSTRUCTION_METHOD,
        },
    )
    _add_signal(
        db_session,
        "RKTO",
        {"decision_date": SIGNAL_DAY.isoformat(), "source": "live"},
    )
    large_membership = {f"synthetic-{index:05d}" for index in range(70_500)}
    large_membership.add(selected_id)
    monkeypatch.setattr(
        selector,
        "historical_m4_replay_signal_ids",
        lambda *args, **kwargs: large_membership,
    )
    statements: list[str] = []

    def _capture_sql(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    bind = db_session.get_bind()
    event.listen(bind, "before_cursor_execute", _capture_sql)
    try:
        selected = (
            selector.apply_signal_source_filter(
                db_session.query(SignalRegistry),
                db_session,
                signal_source=SIGNAL_SOURCE_HISTORICAL_M4_REPLAY,
            )
            .order_by(SignalRegistry.signal_timestamp, SignalRegistry.ticker)
            .all()
        )
    finally:
        event.remove(bind, "before_cursor_execute", _capture_sql)

    assert [signal.signal_id for signal in selected] == [selected_id]
    assert any(
        f"CREATE TEMPORARY TABLE IF NOT EXISTS {selector._REPLAY_MEMBERSHIP_TEMP_TABLE}"
        in statement
        for statement in statements
    )
    assert any(
        f"JOIN {selector._REPLAY_MEMBERSHIP_TEMP_TABLE}" in statement
        for statement in statements
    )
    assert max(statement.count("?") for statement in statements) <= 1
