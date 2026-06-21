from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text


LEFT_SCHEMA = "scratch_i12_tape_left"
RIGHT_SCHEMA = "scratch_i12_tape_right"
DAY = "2026-05-01"


def _load_tape():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_i12_pit_event_tape.py"
    )
    spec = importlib.util.spec_from_file_location("build_i12_pit_event_tape", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _q(schema: str, table: str) -> str:
    return f'"{schema}"."{table}"'


def _create_schema_tables(db_session, schema: str) -> None:
    db_session.execute(text(f"ATTACH DATABASE ':memory:' AS {schema}"))
    db_session.execute(
        text(
            f"""
            CREATE TABLE {_q(schema, 'i12_pit_candidates')} (
              i12_pit_candidate_id TEXT PRIMARY KEY,
              ticker TEXT NOT NULL,
              decision_date TEXT NOT NULL,
              decision_ts TEXT NOT NULL,
              decision_time_label TEXT NOT NULL,
              path_mode TEXT NOT NULL,
              candidate_status TEXT NOT NULL,
              coverage_status TEXT NOT NULL,
              fail_reason TEXT,
              feature_json TEXT NOT NULL,
              source_bars_json TEXT NOT NULL,
              leakage_guard_json TEXT,
              candidate_attempt_hash TEXT,
              content_hash TEXT,
              is_active INTEGER NOT NULL
            )
            """
        )
    )
    db_session.execute(
        text(
            f"""
            CREATE TABLE {_q(schema, 'i12_pit_quote_replays')} (
              i12_pit_quote_replay_id TEXT PRIMARY KEY,
              i12_pit_candidate_id TEXT NOT NULL,
              quote_role TEXT NOT NULL,
              coverage_status TEXT NOT NULL,
              quote_ts TEXT,
              bid REAL,
              ask REAL,
              spread_bps REAL,
              executable_notional REAL,
              quote_age_seconds REAL,
              raw_json TEXT,
              is_active INTEGER NOT NULL
            )
            """
        )
    )
    db_session.execute(
        text(
            f"""
            CREATE TABLE {_q(schema, 'i12_pit_cost_replays')} (
              i12_pit_cost_replay_id TEXT PRIMARY KEY,
              i12_pit_candidate_id TEXT NOT NULL,
              exit_role TEXT NOT NULL,
              entry_quote_replay_id TEXT,
              exit_quote_replay_id TEXT,
              tradeability_status TEXT NOT NULL,
              skipped_reason TEXT NOT NULL,
              intended_order_usd REAL NOT NULL,
              max_spread_bps REAL NOT NULL,
              slippage_bps REAL NOT NULL,
              quote_cost_return REAL,
              slippage_return REAL,
              modeled_return REAL NOT NULL,
              is_active INTEGER NOT NULL
            )
            """
        )
    )


def _feature_json(
    volume: float | None = 1000,
    projected: float = 999999,
    early_volume: float | None = None,
) -> str:
    payload = {
        "prior_close": 10.0,
        "distance_from_max252": -0.6,
        "drawdown_from_max252": -0.6,
        "off_low252": 0.2,
        "mom20": 0.1,
        "sigma20": 0.05,
        "prev_day_return": -0.03,
        "prev_day_green": False,
        "gap": 0.02,
        "early_return": 0.04,
        "early_high_return": 0.05,
        "early_low_return": -0.01,
        "observed_open_to_decision_return": 0.04,
        "observed_cumulative_volume_before_decision": volume,
        "observed_minute_count_before_decision": 5,
        "opening_bar_present": True,
        "path_coverage_ratio": 1.0,
        "completed_minute_count": 5,
        "zero_fill_imputed_minute_count": 0,
        "zero_fill_imputed_minute_ratio": 0.0,
        "projected_volume_at_decision": projected,
        "zero_fill_projected_volume_ratio": projected,
        "same_day_close": 99,
        "same_day_exit_return": 10,
    }
    if early_volume is not None:
        payload["early_cumulative_volume"] = early_volume
    return json.dumps(payload, sort_keys=True)


def _guard_json(decision_time: str, *, leaky: bool = False) -> str:
    max_start = "09:34" if decision_time == "09:35" else "09:39"
    return json.dumps(
        {
            "decision_ts": f"{DAY}T{decision_time}:00Z",
            "source_minute_bars_max_start_ts": f"{DAY}T{max_start}:00Z",
            "completed_through_ts": f"{DAY}T{decision_time}:00Z",
            "feature_asof_ts": f"{DAY}T{decision_time}:00Z",
            "uses_forward_bars": leaky,
            "uses_full_day_volume": False,
            "uses_full_day_high_low": False,
            "uses_same_day_close": False,
        },
        sort_keys=True,
    )


def _source_bars_json(decision_time: str) -> str:
    max_start = "09:34" if decision_time == "09:35" else "09:39"
    return json.dumps(
        {
            "source_minute_bars_max_start_ts": f"{DAY}T{max_start}:00Z",
            "completed_through_ts": f"{DAY}T{decision_time}:00Z",
        },
        sort_keys=True,
    )


def _insert_candidate(
    db_session,
    schema: str,
    candidate_id: str,
    ticker: str,
    *,
    decision_time: str,
    status: str,
    fail_reason: str | None = None,
    leaky: bool = False,
    feature_json: str | None = None,
    source_bars_json: str | None = None,
) -> None:
    db_session.execute(
        text(
            f"""
            INSERT INTO {_q(schema, 'i12_pit_candidates')} (
              i12_pit_candidate_id, ticker, decision_date, decision_ts,
              decision_time_label, path_mode, candidate_status, coverage_status,
              fail_reason, feature_json, source_bars_json, leakage_guard_json,
              candidate_attempt_hash, content_hash, is_active
            ) VALUES (
              :id, :ticker, :day, :decision_ts, :decision_time, 'strict_contiguous',
              :status, 'ok', :fail_reason, :feature_json, :source_bars_json,
              :guard_json, :attempt_hash, :content_hash, 1
            )
            """
        ),
        {
            "id": candidate_id,
            "ticker": ticker,
            "day": DAY,
            "decision_ts": f"{DAY}T{decision_time}:00Z",
            "decision_time": decision_time,
            "status": status,
            "fail_reason": fail_reason,
            "feature_json": feature_json or _feature_json(),
            "source_bars_json": source_bars_json or _source_bars_json(decision_time),
            "guard_json": _guard_json(decision_time, leaky=leaky),
            "attempt_hash": f"attempt-{candidate_id}",
            "content_hash": f"content-{candidate_id}",
        },
    )


def _insert_quote(
    db_session,
    schema: str,
    quote_id: str,
    candidate_id: str,
    role: str,
    *,
    status: str = "ok",
    spread: float = 100,
    notional: float = 250,
    bid: float = 10.0,
    ask: float = 10.1,
) -> None:
    db_session.execute(
        text(
            f"""
            INSERT INTO {_q(schema, 'i12_pit_quote_replays')} (
              i12_pit_quote_replay_id, i12_pit_candidate_id, quote_role,
              coverage_status, quote_ts, bid, ask, spread_bps,
              executable_notional, quote_age_seconds, raw_json, is_active
            ) VALUES (
              :id, :candidate_id, :role, :status, :quote_ts, :bid, :ask,
              :spread, :notional, 1.5, '{{}}', 1
            )
            """
        ),
        {
            "id": quote_id,
            "candidate_id": candidate_id,
            "role": role,
            "status": status,
            "quote_ts": f"{DAY}T13:40:00Z",
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "notional": notional,
        },
    )


def _insert_cost(
    db_session,
    schema: str,
    cost_id: str,
    candidate_id: str,
    role: str,
    *,
    modeled_return: float = 0.0,
    status: str = "tradeable",
    reason: str = "none",
    intended_order_usd: float = 250.0,
    max_spread_bps: float = 200.0,
    slippage_bps: float = 0.0,
    entry_quote_replay_id: str | None = None,
    exit_quote_replay_id: str | None = None,
) -> None:
    entry_quote_replay_id = entry_quote_replay_id or f"q-{candidate_id}-entry"
    role_suffix = "sd" if role == "same_day_exit" else "no"
    exit_quote_replay_id = exit_quote_replay_id or f"q-{candidate_id}-{role_suffix}"
    db_session.execute(
        text(
            f"""
            INSERT INTO {_q(schema, 'i12_pit_cost_replays')} (
              i12_pit_cost_replay_id, i12_pit_candidate_id, exit_role,
              entry_quote_replay_id, exit_quote_replay_id, tradeability_status,
              skipped_reason, intended_order_usd, max_spread_bps, slippage_bps,
              quote_cost_return, slippage_return, modeled_return, is_active
            ) VALUES (
              :id, :candidate_id, :role, :entry_quote_replay_id, :exit_quote_replay_id,
              :status, :reason, :intended_order_usd, :max_spread_bps, :slippage_bps,
              :modeled_return, :modeled_return, :modeled_return, 1
            )
            """
        ),
        {
            "id": cost_id,
            "candidate_id": candidate_id,
            "role": role,
            "entry_quote_replay_id": entry_quote_replay_id,
            "exit_quote_replay_id": exit_quote_replay_id,
            "status": status,
            "reason": reason,
            "intended_order_usd": intended_order_usd,
            "max_spread_bps": max_spread_bps,
            "slippage_bps": slippage_bps,
            "modeled_return": modeled_return,
        },
    )


def _insert_complete_evidence(db_session, schema: str, candidate_id: str, *, ret: float = 0.01) -> None:
    _insert_quote(db_session, schema, f"q-{candidate_id}-entry", candidate_id, "entry")
    _insert_quote(db_session, schema, f"q-{candidate_id}-sd", candidate_id, "same_day_exit")
    _insert_quote(db_session, schema, f"q-{candidate_id}-no", candidate_id, "next_open_exit")
    _insert_cost(db_session, schema, f"c-{candidate_id}-sd", candidate_id, "same_day_exit", modeled_return=ret)
    _insert_cost(db_session, schema, f"c-{candidate_id}-no", candidate_id, "next_open_exit", modeled_return=ret / 2)


def _seed_fixture(db_session, *, missing_right_evidence: bool = False, leaky_left: bool = False) -> None:
    _create_schema_tables(db_session, LEFT_SCHEMA)
    _create_schema_tables(db_session, RIGHT_SCHEMA)
    _insert_candidate(db_session, LEFT_SCHEMA, "left-a", "AAA", decision_time="09:35", status="passed", leaky=leaky_left)
    _insert_candidate(db_session, LEFT_SCHEMA, "left-b", "BBB", decision_time="09:35", status="passed")
    _insert_candidate(db_session, LEFT_SCHEMA, "left-c", "CCC", decision_time="09:35", status="failed", fail_reason="drawdown")
    _insert_candidate(db_session, LEFT_SCHEMA, "left-d", "DDD", decision_time="09:35", status="failed", fail_reason="volume")
    _insert_candidate(db_session, RIGHT_SCHEMA, "right-a", "AAA", decision_time="09:40", status="passed")
    _insert_candidate(db_session, RIGHT_SCHEMA, "right-b", "BBB", decision_time="09:40", status="failed", fail_reason="volume")
    _insert_candidate(db_session, RIGHT_SCHEMA, "right-c", "CCC", decision_time="09:40", status="passed")
    _insert_candidate(db_session, RIGHT_SCHEMA, "right-d", "DDD", decision_time="09:40", status="failed", fail_reason="volume")
    _insert_complete_evidence(db_session, LEFT_SCHEMA, "left-a", ret=0.10)
    _insert_complete_evidence(db_session, LEFT_SCHEMA, "left-b", ret=0.20)
    _insert_complete_evidence(db_session, RIGHT_SCHEMA, "right-a", ret=0.08)
    if missing_right_evidence:
        _insert_quote(db_session, RIGHT_SCHEMA, "q-right-c-sd", "right-c", "same_day_exit")
        _insert_cost(db_session, RIGHT_SCHEMA, "c-right-c-sd", "right-c", "same_day_exit", modeled_return=0.30)
    else:
        _insert_complete_evidence(db_session, RIGHT_SCHEMA, "right-c", ret=0.30)
    db_session.flush()


def _seed_single_volume_case(
    db_session,
    *,
    volume: float | None = 1000,
    projected: float = 999999,
    early_volume: float | None = None,
    entry_status: str = "ok",
    exit_status: str = "ok",
    entry_spread: float = 100,
    exit_spread: float = 100,
    source_bars_json: str | None = None,
    displayed_status: str = "skipped_cash",
    displayed_reason: str = "size",
) -> None:
    _create_schema_tables(db_session, LEFT_SCHEMA)
    _insert_candidate(
        db_session,
        LEFT_SCHEMA,
        "left-volume",
        "VOL",
        decision_time="09:35",
        status="passed",
        feature_json=_feature_json(
            volume=volume,
            projected=projected,
            early_volume=early_volume,
        ),
        source_bars_json=source_bars_json,
    )
    _insert_quote(
        db_session,
        LEFT_SCHEMA,
        "q-left-volume-entry",
        "left-volume",
        "entry",
        status=entry_status,
        spread=entry_spread,
    )
    _insert_quote(
        db_session,
        LEFT_SCHEMA,
        "q-left-volume-sd",
        "left-volume",
        "same_day_exit",
        status=exit_status,
        spread=exit_spread,
    )
    _insert_quote(
        db_session,
        LEFT_SCHEMA,
        "q-left-volume-no",
        "left-volume",
        "next_open_exit",
        status=exit_status,
        spread=exit_spread,
    )
    _insert_cost(
        db_session,
        LEFT_SCHEMA,
        "c-left-volume-sd",
        "left-volume",
        "same_day_exit",
        modeled_return=0.0,
        status=displayed_status,
        reason=displayed_reason,
    )
    _insert_cost(
        db_session,
        LEFT_SCHEMA,
        "c-left-volume-no",
        "left-volume",
        "next_open_exit",
        modeled_return=0.0,
        status=displayed_status,
        reason=displayed_reason,
    )
    db_session.flush()


def _build_single(db_session, **kwargs):
    tape = _load_tape()
    return tape.build_event_tape(
        snapshots=[tape.SnapshotSpec("09:35", LEFT_SCHEMA)],
        start_date=DAY,
        end_date=DAY,
        minute_path_mode="strict_contiguous",
        db_session=db_session,
        **kwargs,
    )


def _snapshot(label: str, schema: str, report: Path | None = None):
    tape = _load_tape()
    return tape.SnapshotSpec(label=label, schema=schema, report=report)


def _build(db_session, **kwargs):
    tape = _load_tape()
    return tape.build_event_tape(
        snapshots=[
            tape.SnapshotSpec("09:35", LEFT_SCHEMA),
            tape.SnapshotSpec("09:40", RIGHT_SCHEMA),
        ],
        start_date=DAY,
        end_date=DAY,
        minute_path_mode="strict_contiguous",
        db_session=db_session,
        **kwargs,
    )


def _final_report(schema: str, label: str) -> dict:
    return {
        "conclusions_final": True,
        "data_integrity_passed": True,
        "source_replay_complete": True,
        "quote_replay_complete": True,
        "cost_replay_complete": True,
        "schema": schema,
        "output_schema": schema,
        "start_date": "2026-04-01",
        "end_date": "2026-06-01",
        "report_path_mode": "strict_contiguous",
        "minute_path_mode": "strict_contiguous",
        "report_decision_time_labels": [label],
        "source_hur_schema": "public",
        "missing_source_attempt_count": 0,
        "extra_source_attempt_count": 0,
        "missing_source_attempt_identity_count": 0,
        "extra_source_attempt_identity_count": 0,
    }


def test_event_tape_rejects_non_scratch_schema(db_session):
    tape = _load_tape()
    with pytest.raises(ValueError, match="scratch schema"):
        tape.build_event_tape(
            snapshots=[tape.SnapshotSpec("09:35", "public")],
            start_date=DAY,
            end_date=DAY,
            minute_path_mode="strict_contiguous",
            db_session=db_session,
        )


def test_event_tape_parse_snapshot_keeps_hhmm_label():
    tape = _load_tape()
    snapshot = tape.parse_snapshot("09:35:scratch_i12_example:/tmp/report.json")

    assert snapshot.label == "09:35"
    assert snapshot.schema == "scratch_i12_example"
    assert str(snapshot.report) == "/tmp/report.json"


def test_event_tape_requires_bounded_date_range(db_session):
    tape = _load_tape()
    with pytest.raises(ValueError, match="start_date"):
        tape.build_event_tape(
            snapshots=[tape.SnapshotSpec("09:35", LEFT_SCHEMA)],
            start_date=None,
            end_date=DAY,
            minute_path_mode="strict_contiguous",
            db_session=db_session,
        )


def test_event_tape_rejects_duplicate_snapshot_labels_before_db_work():
    tape = _load_tape()
    with pytest.raises(ValueError, match="duplicate snapshot decision-time labels"):
        tape.build_event_tape(
            snapshots=[
                tape.SnapshotSpec("09:35", "scratch_i12_tape_a"),
                tape.SnapshotSpec("09:35", "scratch_i12_tape_b"),
            ],
            start_date=DAY,
            end_date=DAY,
            minute_path_mode="strict_contiguous",
        )


def test_event_tape_rejects_duplicate_snapshot_schemas_before_db_work():
    tape = _load_tape()
    with pytest.raises(ValueError, match="duplicate snapshot schemas"):
        tape.build_event_tape(
            snapshots=[
                tape.SnapshotSpec("09:35", "scratch_i12_tape_same"),
                tape.SnapshotSpec("09:40", "scratch_i12_tape_same"),
            ],
            start_date=DAY,
            end_date=DAY,
            minute_path_mode="strict_contiguous",
        )


def test_event_tape_require_final_fails_without_reports(db_session):
    tape = _load_tape()
    with pytest.raises(RuntimeError, match="requires every --snapshot"):
        tape.build_event_tape(
            snapshots=[tape.SnapshotSpec("09:35", LEFT_SCHEMA)],
            start_date=DAY,
            end_date=DAY,
            minute_path_mode="strict_contiguous",
            db_session=db_session,
            require_final=True,
        )


def test_event_tape_require_final_rejects_metadata_less_report(db_session, tmp_path):
    tape = _load_tape()
    report = tmp_path / "old.json"
    report.write_text(json.dumps({"conclusions_final": True, "data_integrity_passed": True}))
    with pytest.raises(RuntimeError, match="regenerate report with current code"):
        tape.build_event_tape(
            snapshots=[tape.SnapshotSpec("09:35", LEFT_SCHEMA, report)],
            start_date=DAY,
            end_date=DAY,
            minute_path_mode="strict_contiguous",
            db_session=db_session,
            require_final=True,
        )


def test_event_tape_require_final_accepts_covering_reports(db_session, tmp_path):
    _seed_fixture(db_session)
    tape = _load_tape()
    left_report = tmp_path / "left.json"
    right_report = tmp_path / "right.json"
    left_report.write_text(json.dumps(_final_report(LEFT_SCHEMA, "09:35")))
    right_report.write_text(json.dumps(_final_report(RIGHT_SCHEMA, "09:40")))

    result = tape.build_event_tape(
        snapshots=[
            tape.SnapshotSpec("09:35", LEFT_SCHEMA, left_report),
            tape.SnapshotSpec("09:40", RIGHT_SCHEMA, right_report),
        ],
        start_date=DAY,
        end_date=DAY,
        minute_path_mode="strict_contiguous",
        db_session=db_session,
        require_final=True,
    )

    assert result["summary"]["require_final"] is True
    assert result["summary"]["row_count"] == 4
    assert result["summary"]["training_tape_status"] == (
        "eligible_for_volume_participation_ranker_dataset"
    )


def test_event_tape_all_source_attempts_require_final_is_not_training_ready(db_session, tmp_path):
    _seed_fixture(db_session)
    tape = _load_tape()
    left_report = tmp_path / "left.json"
    right_report = tmp_path / "right.json"
    left_report.write_text(json.dumps(_final_report(LEFT_SCHEMA, "09:35")))
    right_report.write_text(json.dumps(_final_report(RIGHT_SCHEMA, "09:40")))

    result = tape.build_event_tape(
        snapshots=[
            tape.SnapshotSpec("09:35", LEFT_SCHEMA, left_report),
            tape.SnapshotSpec("09:40", RIGHT_SCHEMA, right_report),
        ],
        start_date=DAY,
        end_date=DAY,
        minute_path_mode="strict_contiguous",
        db_session=db_session,
        require_final=True,
        scope="all-source-attempts",
    )

    assert result["summary"]["scope"] == "all-source-attempts"
    assert result["summary"]["training_tape_status"] == "diagnostic_not_training_ready"


def test_event_tape_missing_volume_label_blocks_training_ready(db_session, tmp_path, monkeypatch):
    _seed_fixture(db_session)
    tape = _load_tape()
    left_report = tmp_path / "left.json"
    right_report = tmp_path / "right.json"
    left_report.write_text(json.dumps(_final_report(LEFT_SCHEMA, "09:35")))
    right_report.write_text(json.dumps(_final_report(RIGHT_SCHEMA, "09:40")))

    monkeypatch.setattr(
        tape,
        "_volume_participation_fields_for_raw_row",
        lambda *args, **kwargs: tape._empty_volume_participation_fields(),
    )
    result = tape.build_event_tape(
        snapshots=[
            tape.SnapshotSpec("09:35", LEFT_SCHEMA, left_report),
            tape.SnapshotSpec("09:40", RIGHT_SCHEMA, right_report),
        ],
        start_date=DAY,
        end_date=DAY,
        minute_path_mode="strict_contiguous",
        db_session=db_session,
        require_final=True,
    )

    assert result["summary"]["training_tape_status"] == "diagnostic_not_training_ready"


def test_event_tape_builds_rows_and_buckets(db_session):
    _seed_fixture(db_session)
    result = _build(db_session)

    assert result["summary"]["scope"] == "passed"
    assert result["summary"]["tape_kind"] == "fired_event_tape"
    assert result["summary"]["row_count"] == 4
    assert result["summary"]["unique_ticker_date_count"] == 3
    assert result["summary"]["bucket_counts"] == {
        "only_0935": 1,
        "only_0940": 1,
        "shared_0935_0940": 1,
    }
    assert result["summary"]["two_snapshot_summary"]["shared_passed_count"] == 1
    assert result["summary"]["two_snapshot_summary"]["09:35_only_passed_count"] == 1
    assert result["summary"]["two_snapshot_summary"]["09:40_only_passed_count"] == 1


def test_event_tape_all_source_attempts_mode_is_explicit(db_session):
    _seed_fixture(db_session)
    result = _build(db_session, scope="all-source-attempts")

    assert result["summary"]["scope"] == "all-source-attempts"
    assert result["summary"]["tape_kind"] == "source_attempt_tape"
    assert result["summary"]["row_count"] == 8
    assert result["summary"]["unique_ticker_date_count"] == 4
    assert result["summary"]["bucket_counts"] == {
        "failed_both": 1,
        "only_0935": 1,
        "only_0940": 1,
        "shared_0935_0940": 1,
    }
    assert "source_attempt_tape_not_ml_event_tape" in result["summary"]["warnings"]


def test_event_tape_membership_fields(db_session):
    _seed_fixture(db_session)
    rows = {(row["ticker"], row["decision_time_label"]): row for row in _build(db_session)["events"]}

    assert rows[("AAA", "09:35")]["first_seen_decision_time"] == "09:35"
    assert rows[("AAA", "09:35")]["first_fire_decision_time"] == "09:35"
    assert rows[("AAA", "09:35")]["first_source_seen_decision_time"] == "09:35"
    assert rows[("AAA", "09:35")]["first_passed_decision_time"] == "09:35"
    assert rows[("AAA", "09:35")]["survived_to_later_snapshot"] is True
    assert rows[("BBB", "09:35")]["dropped_by_later_snapshot"] is True
    assert rows[("CCC", "09:40")]["first_passed_decision_time"] == "09:40"


def test_event_tape_passed_scope_uses_real_source_presence_for_earlier_failed_row(db_session):
    _seed_fixture(db_session)
    row = next(row for row in _build(db_session)["events"] if row["ticker"] == "CCC")

    assert row["decision_time_label"] == "09:40"
    assert row["bucket"] == "only_0940"
    assert row["source_presence_by_decision_time"] == {"09:35": True, "09:40": True}
    assert row["source_candidate_status_by_decision_time"]["09:35"] == "failed"
    assert row["source_fail_reason_by_decision_time"]["09:35"] == "drawdown"
    assert row["first_source_seen_decision_time"] == "09:35"
    assert row["first_seen_decision_time"] == "09:40"
    assert row["first_fire_decision_time"] == "09:40"


def test_event_tape_all_source_first_source_vs_first_fire(db_session):
    _seed_fixture(db_session)
    rows = {
        (row["ticker"], row["decision_time_label"]): row
        for row in _build(db_session, scope="all-source-attempts")["events"]
    }

    assert rows[("CCC", "09:35")]["first_source_seen_decision_time"] == "09:35"
    assert rows[("CCC", "09:35")]["first_seen_decision_time"] == "09:40"
    assert rows[("CCC", "09:35")]["first_fire_decision_time"] == "09:40"
    assert rows[("DDD", "09:35")]["first_source_seen_decision_time"] == "09:35"
    assert rows[("DDD", "09:35")]["first_seen_decision_time"] is None
    assert rows[("DDD", "09:35")]["first_fire_decision_time"] is None


def test_event_tape_shared_bucket_metrics_are_split_by_decision_time(db_session):
    _seed_fixture(db_session)
    two = _build(db_session)["summary"]["two_snapshot_summary"]

    volume_shared = two["same_day_volume_return_by_bucket_decision_time"]["shared_0935_0940"]
    assert volume_shared["09:35"]["mean"] == pytest.approx(10.0 / 10.1 - 1.0)
    assert volume_shared["09:40"]["mean"] == pytest.approx(10.0 / 10.1 - 1.0)
    diagnostic_shared = two[
        "diagnostic_displayed_size_same_day_return_by_bucket_decision_time"
    ]["shared_0935_0940"]
    assert diagnostic_shared["09:35"]["mean"] == pytest.approx(0.10)
    assert diagnostic_shared["09:40"]["mean"] == pytest.approx(0.08)
    assert two["shared_timing_deltas_right_minus_left"]["same_day_modeled_return_displayed_size"][
        "mean"
    ] == pytest.approx(-0.02)
    assert "same_day_return_by_bucket" not in two


def test_event_tape_volume_label_ignores_displayed_size_skip_when_volume_sufficient(db_session):
    _seed_single_volume_case(db_session, volume=1000)

    result = _build_single(db_session)
    row = result["events"][0]

    assert row["same_day_tradeability_status"] == "skipped_cash"
    assert row["same_day_skipped_reason"] == "size"
    assert row["same_day_volume_tradeability_status"] == "tradeable_volume"
    assert row["same_day_volume_skipped_reason"] == "none"
    assert row["same_day_modeled_return_volume_participation"] == pytest.approx(
        10.0 / 10.1 - 1.0
    )
    assert row["same_day_volume_quote_cost_return"] == pytest.approx(10.0 / 10.1 - 1.0)
    assert row["same_day_volume_slippage_return"] == pytest.approx(10.0 / 10.1 - 1.0)
    assert row["entry_window_dollar_volume"] == pytest.approx(1000 * 10.05)
    assert row["entry_window_share_volume"] == pytest.approx(1000)
    assert row["intended_order_participation_rate"] == pytest.approx(250 / (1000 * 10.05))
    assert row["volume_denominator_basis"] == "observed_cumulative_volume_before_decision"
    assert row["volume_price_basis"] == "entry_quote_mid"
    assert row["volume_timestamp_proof_status"] == "ok"
    assert result["summary"]["displayed_size_skip_ignored_count"]["same_day_exit"] == 1
    assert result["summary"]["volume_tradeability_status_counts"]["same_day_exit"] == {
        "tradeable_volume": 1
    }


def test_event_tape_volume_label_matches_train_model_shared_helper(db_session):
    from alpha.jobs import train_model

    _seed_single_volume_case(db_session, volume=1000)

    row = _build_single(db_session)["events"][0]
    candidate = SimpleNamespace(
        i12_pit_candidate_id="left-volume",
        feature_json=_feature_json(volume=1000),
        source_bars_json=_source_bars_json("09:35"),
        decision_ts=datetime.fromisoformat(f"{DAY}T09:35:00+00:00"),
    )
    entry_quote = SimpleNamespace(
        coverage_status="ok",
        bid=10.0,
        ask=10.1,
        spread_bps=100.0,
        raw_json="{}",
    )
    exit_quote = SimpleNamespace(
        coverage_status="ok",
        bid=10.0,
        ask=10.1,
        spread_bps=100.0,
        raw_json="{}",
    )
    cost = SimpleNamespace(
        i12_pit_candidate_id="left-volume",
        exit_role="same_day_exit",
        intended_order_usd=250.0,
        max_spread_bps=200.0,
        slippage_bps=0.0,
    )

    expected = train_model.volume_tradeability_for_cost(
        candidate=candidate,
        entry_quote=entry_quote,
        exit_quote=exit_quote,
        cost=cost,
        threshold=train_model.DEFAULT_VOLUME_PARTICIPATION_THRESHOLD,
        evidence_getter=train_model.predecision_volume_evidence,
    )

    assert row["same_day_modeled_return_volume_participation"] == pytest.approx(
        expected["modeled_return"]
    )
    assert row["same_day_volume_tradeability_status"] == expected["volume_tradeability_status"]
    assert row["same_day_volume_skipped_reason"] == expected["volume_skipped_reason"]


def test_event_tape_volume_label_skips_when_volume_too_thin(db_session):
    _seed_single_volume_case(db_session, volume=10)

    row = _build_single(db_session)["events"][0]

    assert row["same_day_volume_tradeability_status"] == "skipped_cash"
    assert row["same_day_volume_skipped_reason"] == "volume_too_thin"
    assert row["same_day_modeled_return_volume_participation"] == 0.0


def test_event_tape_volume_label_missing_volume_ignores_projected_volume(db_session):
    _seed_single_volume_case(db_session, volume=0, projected=999999999)

    row = _build_single(db_session)["events"][0]

    assert row["same_day_volume_tradeability_status"] == "skipped_cash"
    assert row["same_day_volume_skipped_reason"] == "volume_missing"
    assert row["same_day_modeled_return_volume_participation"] == 0.0
    assert row["volume_denominator_basis"] == "missing"


def test_event_tape_volume_label_missing_observed_volume_ignores_early_volume(db_session):
    _seed_single_volume_case(
        db_session,
        volume=None,
        early_volume=999999999,
        projected=999999999,
    )

    row = _build_single(db_session)["events"][0]

    assert row["same_day_volume_tradeability_status"] == "skipped_cash"
    assert row["same_day_volume_skipped_reason"] == "volume_missing"
    assert row["next_open_volume_tradeability_status"] == "skipped_cash"
    assert row["next_open_volume_skipped_reason"] == "volume_missing"
    assert row["entry_window_dollar_volume"] is None
    assert row["volume_denominator_basis"] == "missing"


@pytest.mark.parametrize(
    ("role", "field", "bad_quote_id", "message"),
    [
        ("same_day_exit", "entry_quote_replay_id", "q-other-entry", "entry_quote_replay_id"),
        ("next_open_exit", "exit_quote_replay_id", "q-other-exit", "exit_quote_replay_id"),
    ],
)
def test_event_tape_cost_quote_id_mismatch_fails_closed(
    db_session,
    role,
    field,
    bad_quote_id,
    message,
):
    _seed_single_volume_case(db_session, volume=1000)
    db_session.execute(
        text(
            f"""
            UPDATE {_q(LEFT_SCHEMA, 'i12_pit_cost_replays')}
            SET {field} = :bad_quote_id
            WHERE exit_role = :role
            """
        ),
        {"bad_quote_id": bad_quote_id, "role": role},
    )
    db_session.flush()

    with pytest.raises(RuntimeError, match=message):
        _build_single(db_session)


@pytest.mark.parametrize(
    ("role", "field", "message"),
    [
        ("same_day_exit", "entry_quote_replay_id", "missing entry_quote_replay_id"),
        ("next_open_exit", "exit_quote_replay_id", "missing exit_quote_replay_id"),
    ],
)
def test_event_tape_cost_quote_id_null_fails_closed(
    db_session,
    role,
    field,
    message,
):
    _seed_single_volume_case(db_session, volume=1000)
    db_session.execute(
        text(
            f"""
            UPDATE {_q(LEFT_SCHEMA, 'i12_pit_cost_replays')}
            SET {field} = NULL
            WHERE exit_role = :role
            """
        ),
        {"role": role},
    )
    db_session.flush()

    with pytest.raises(RuntimeError, match=message):
        _build_single(db_session)


@pytest.mark.parametrize(
    ("entry_status", "exit_status", "entry_spread", "exit_spread", "expected_reason"),
    [
        ("stale", "ok", 100, 100, "entry_quote_stale"),
        ("missing", "ok", 100, 100, "entry_quote_missing"),
        ("error", "ok", 100, 100, "halt_or_bad_quote"),
        ("ok", "stale", 100, 100, "same_day_exit_quote_stale"),
        ("ok", "missing", 100, 100, "same_day_exit_quote_missing"),
        ("ok", "error", 100, 100, "halt_or_bad_quote"),
        ("ok", "ok", 250, 100, "spread"),
        ("ok", "ok", 100, 250, "spread"),
    ],
)
def test_event_tape_volume_label_quote_frictions_skip_cash(
    db_session,
    entry_status,
    exit_status,
    entry_spread,
    exit_spread,
    expected_reason,
):
    _seed_single_volume_case(
        db_session,
        volume=1000,
        entry_status=entry_status,
        exit_status=exit_status,
        entry_spread=entry_spread,
        exit_spread=exit_spread,
    )

    row = _build_single(db_session)["events"][0]

    assert row["same_day_volume_tradeability_status"] == "skipped_cash"
    assert row["same_day_volume_skipped_reason"] == expected_reason
    assert row["same_day_modeled_return_volume_participation"] == 0.0


def test_event_tape_volume_label_unsafe_timestamp_proof_fails_closed(db_session):
    unsafe_source_bars = json.dumps(
        {
            "source_minute_bars_max_start_ts": f"{DAY}T09:35:00Z",
            "completed_through_ts": f"{DAY}T09:35:00Z",
        },
        sort_keys=True,
    )
    _seed_single_volume_case(db_session, volume=1000, source_bars_json=unsafe_source_bars)

    row = _build_single(db_session)["events"][0]

    assert row["same_day_volume_tradeability_status"] == "skipped_cash"
    assert row["same_day_volume_skipped_reason"] == "volume_missing"
    assert row["same_day_modeled_return_volume_participation"] == 0.0
    assert row["volume_timestamp_proof_status"] == (
        "source_minute_bars_max_start_ts_at_or_after_decision_ts"
    )


def test_event_tape_predictors_exclude_projected_and_outcome_fields(db_session):
    _seed_fixture(db_session)
    row = next(row for row in _build(db_session)["events"] if row["ticker"] == "AAA" and row["decision_time_label"] == "09:35")

    assert row["predictor_status"] == "ok"
    assert set(row["predictors"]) == set(_load_tape().PREDICTOR_ALLOWLIST)
    assert "observed_cumulative_volume_before_decision" in row["predictors"]
    assert "projected_volume_at_decision" not in row["predictors"]
    assert "zero_fill_projected_volume_ratio" not in row["predictors"]
    assert "same_day_close" not in row["predictors"]
    assert "same_day_exit_return" not in row["predictors"]
    result_missing_counts = _build(db_session)["summary"]["predictor_missing_counts"]
    assert result_missing_counts
    assert set(result_missing_counts) == set(_load_tape().PREDICTOR_ALLOWLIST)


def test_event_tape_leakage_guard_violation_blocks_predictors(db_session):
    _seed_fixture(db_session, leaky_left=True)
    row = next(row for row in _build(db_session)["events"] if row["ticker"] == "AAA" and row["decision_time_label"] == "09:35")

    assert row["predictor_status"] == "blocked_leakage_guard"
    assert row["predictor_block_reason"] == "uses_forward_bars"
    assert set(row["predictors"]) == set(_load_tape().PREDICTOR_ALLOWLIST)
    assert all(value is None for value in row["predictors"].values())


def test_event_tape_duplicate_active_candidate_rows_fail_closed(db_session):
    _seed_fixture(db_session)
    _insert_candidate(db_session, LEFT_SCHEMA, "left-a-dup", "AAA", decision_time="09:35", status="passed")
    db_session.flush()

    with pytest.raises(RuntimeError, match="duplicate_active_candidate_count"):
        _build(db_session)


def test_event_tape_duplicate_active_quote_rows_fail_closed(db_session):
    _seed_fixture(db_session)
    _insert_quote(db_session, LEFT_SCHEMA, "q-left-a-entry-dup", "left-a", "entry")
    db_session.flush()

    with pytest.raises(RuntimeError, match="duplicate_quote_role_count"):
        _build(db_session)


def test_event_tape_duplicate_active_cost_rows_fail_closed(db_session):
    _seed_fixture(db_session)
    _insert_cost(db_session, LEFT_SCHEMA, "c-left-a-sd-dup", "left-a", "same_day_exit")
    db_session.flush()

    with pytest.raises(RuntimeError, match="duplicate_cost_role_count"):
        _build(db_session)


def test_event_tape_missing_quote_cost_are_warnings(db_session):
    _seed_fixture(db_session, missing_right_evidence=True)
    result = _build(db_session)

    assert "09:40_missing_quote_evidence" in result["summary"]["warnings"]
    assert "09:40_missing_cost_evidence" in result["summary"]["warnings"]


def test_event_tape_json_jsonl_csv_outputs_serialize(db_session, tmp_path):
    _seed_fixture(db_session)
    tape = _load_tape()
    result = _build(db_session)

    json.loads(tape.render_json(result, max_rows=10))
    jsonl = tape.render_jsonl(result)
    assert '"type": "summary"' in jsonl
    csv_path = tmp_path / "tape.csv"
    tape.write_csv(result, csv_path)
    assert csv_path.read_text().startswith("bucket,")


def test_event_tape_large_json_guard_prefers_streaming_formats(db_session):
    _seed_fixture(db_session)
    tape = _load_tape()

    with pytest.raises(RuntimeError, match="use --format jsonl or --format csv"):
        tape.render_json(_build(db_session), max_rows=1)


def test_event_tape_preflight_all_source_guard_fires_before_row_fetch(db_session, monkeypatch):
    _seed_fixture(db_session)
    tape = _load_tape()

    def fail_row_fetch(*args, **kwargs):
        raise AssertionError("event rows should not be materialized after preflight guard")

    monkeypatch.setattr(tape, "_event_rows_for_snapshot", fail_row_fetch)
    with pytest.raises(RuntimeError, match="preflight all-source-attempts row count"):
        tape.build_event_tape(
            snapshots=[
                tape.SnapshotSpec("09:35", LEFT_SCHEMA),
                tape.SnapshotSpec("09:40", RIGHT_SCHEMA),
            ],
            start_date=DAY,
            end_date=DAY,
            minute_path_mode="strict_contiguous",
            db_session=db_session,
            scope="all-source-attempts",
            output_format="json",
            max_event_rows=1,
        )


def test_event_tape_direct_help_works():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_i12_pit_event_tape.py"
    result = subprocess.run([str(script), "--help"], cwd=script.parents[1], text=True, capture_output=True)

    assert result.returncode == 0
    assert "--snapshot" in result.stdout
    assert "--scope" in result.stdout
    assert "--allow-large-source-attempts" in result.stdout


def test_event_tape_direct_help_works_without_sqlalchemy_site_packages():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_i12_pit_event_tape.py"
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        cwd=script.parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--snapshot" in result.stdout
