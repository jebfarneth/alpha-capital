from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from sqlalchemy import text


LEFT_SCHEMA = "scratch_i12_cmp_left"
RIGHT_SCHEMA = "scratch_i12_cmp_right"
DAY = "2026-05-01"


def _load_compare():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compare_i12_pit_snapshots.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_i12_pit_snapshots",
        script_path,
    )
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
              tradeability_status TEXT NOT NULL,
              skipped_reason TEXT NOT NULL,
              modeled_return REAL NOT NULL,
              is_active INTEGER NOT NULL
            )
            """
        )
    )


def _feature_json(volume: float, minutes: float, coverage: float, early_return: float = 0.01) -> str:
    return json.dumps(
        {
            "observed_cumulative_volume_before_decision": volume,
            "observed_minute_count_before_decision": minutes,
            "path_coverage_ratio": coverage,
            "zero_fill_imputed_minute_count": 0,
            "zero_fill_imputed_minute_ratio": 0.0,
            "early_return": early_return,
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
    volume: float = 1000,
    minutes: float = 10,
    coverage: float = 1.0,
) -> None:
    db_session.execute(
        text(
            f"""
            INSERT INTO {_q(schema, 'i12_pit_candidates')} (
              i12_pit_candidate_id, ticker, decision_date, decision_ts,
              decision_time_label, path_mode, candidate_status, coverage_status,
              fail_reason, feature_json, leakage_guard_json, candidate_attempt_hash,
              content_hash, is_active
            ) VALUES (
              :id, :ticker, :day, :decision_ts, :decision_time, 'strict_contiguous',
              :status, 'ok', :fail_reason, :feature_json, '{{}}', :attempt_hash,
              :content_hash, 1
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
            "feature_json": _feature_json(volume, minutes, coverage),
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
    spread: float = 100.0,
    notional: float = 250.0,
    age: float = 1.0,
) -> None:
    db_session.execute(
        text(
            f"""
            INSERT INTO {_q(schema, 'i12_pit_quote_replays')} (
              i12_pit_quote_replay_id, i12_pit_candidate_id, quote_role,
              coverage_status, quote_ts, bid, ask, spread_bps,
              executable_notional, quote_age_seconds, is_active
            ) VALUES (
              :id, :candidate_id, :role, :status, :quote_ts, 10.0, 10.1,
              :spread, :notional, :age, 1
            )
            """
        ),
        {
            "id": quote_id,
            "candidate_id": candidate_id,
            "role": role,
            "status": status,
            "quote_ts": f"{DAY}T13:40:00Z" if status != "missing" else None,
            "spread": spread,
            "notional": notional,
            "age": age,
        },
    )


def _insert_cost(
    db_session,
    schema: str,
    cost_id: str,
    candidate_id: str,
    role: str,
    *,
    status: str = "tradeable",
    reason: str = "none",
    modeled_return: float = 0.0,
) -> None:
    db_session.execute(
        text(
            f"""
            INSERT INTO {_q(schema, 'i12_pit_cost_replays')} (
              i12_pit_cost_replay_id, i12_pit_candidate_id, exit_role,
              tradeability_status, skipped_reason, modeled_return, is_active
            ) VALUES (
              :id, :candidate_id, :role, :status, :reason, :modeled_return, 1
            )
            """
        ),
        {
            "id": cost_id,
            "candidate_id": candidate_id,
            "role": role,
            "status": status,
            "reason": reason,
            "modeled_return": modeled_return,
        },
    )


def _insert_complete_evidence(
    db_session,
    schema: str,
    candidate_id: str,
    *,
    entry_spread: float,
    entry_notional: float,
    same_day_return: float,
    next_open_return: float = 0.01,
) -> None:
    _insert_quote(db_session, schema, f"q-{candidate_id}-entry", candidate_id, "entry", spread=entry_spread, notional=entry_notional)
    _insert_quote(db_session, schema, f"q-{candidate_id}-sd", candidate_id, "same_day_exit")
    _insert_quote(db_session, schema, f"q-{candidate_id}-no", candidate_id, "next_open_exit")
    _insert_cost(db_session, schema, f"c-{candidate_id}-sd", candidate_id, "same_day_exit", modeled_return=same_day_return)
    _insert_cost(db_session, schema, f"c-{candidate_id}-no", candidate_id, "next_open_exit", modeled_return=next_open_return)


def _seed_compare_fixture(db_session) -> None:
    _create_schema_tables(db_session, LEFT_SCHEMA)
    _create_schema_tables(db_session, RIGHT_SCHEMA)
    # A passes both, B left-only, C right-only, D fails both.
    _insert_candidate(db_session, LEFT_SCHEMA, "left-a", "AAA", decision_time="09:35", status="passed", volume=1000, minutes=5)
    _insert_candidate(db_session, LEFT_SCHEMA, "left-b", "BBB", decision_time="09:35", status="passed", volume=800, minutes=5)
    _insert_candidate(db_session, LEFT_SCHEMA, "left-c", "CCC", decision_time="09:35", status="failed", fail_reason="drawdown")
    _insert_candidate(db_session, LEFT_SCHEMA, "left-d", "DDD", decision_time="09:35", status="failed", fail_reason="volume")
    _insert_candidate(db_session, RIGHT_SCHEMA, "right-a", "AAA", decision_time="09:40", status="passed", volume=2000, minutes=10)
    _insert_candidate(db_session, RIGHT_SCHEMA, "right-b", "BBB", decision_time="09:40", status="failed", fail_reason="volume")
    _insert_candidate(db_session, RIGHT_SCHEMA, "right-c", "CCC", decision_time="09:40", status="passed", volume=1500, minutes=10)
    _insert_candidate(db_session, RIGHT_SCHEMA, "right-d", "DDD", decision_time="09:40", status="failed", fail_reason="volume")
    _insert_complete_evidence(db_session, LEFT_SCHEMA, "left-a", entry_spread=100, entry_notional=250, same_day_return=0.10)
    _insert_complete_evidence(db_session, LEFT_SCHEMA, "left-b", entry_spread=110, entry_notional=220, same_day_return=0.20)
    _insert_complete_evidence(db_session, RIGHT_SCHEMA, "right-a", entry_spread=150, entry_notional=300, same_day_return=0.08)
    # Intentionally incomplete right-only evidence for warnings.
    _insert_quote(db_session, RIGHT_SCHEMA, "q-right-c-sd", "right-c", "same_day_exit")
    _insert_quote(db_session, RIGHT_SCHEMA, "q-right-c-no", "right-c", "next_open_exit")
    _insert_cost(db_session, RIGHT_SCHEMA, "c-right-c-sd", "right-c", "same_day_exit", modeled_return=0.30)
    db_session.flush()


def _analysis(db_session):
    compare = _load_compare()
    _seed_compare_fixture(db_session)
    return compare.compare_snapshots(
        left_schema=LEFT_SCHEMA,
        left_label="09:35",
        right_schema=RIGHT_SCHEMA,
        right_label="09:40",
        start_date=DAY,
        end_date=DAY,
        decision_time_left="09:35",
        decision_time_right="09:40",
        minute_path_mode="strict_contiguous",
        db_session=db_session,
    )


def test_snapshot_compare_rejects_non_scratch_schema(db_session):
    compare = _load_compare()
    with pytest.raises(ValueError, match="scratch schema"):
        compare.compare_snapshots(
            left_schema="public",
            left_label="bad",
            right_schema=RIGHT_SCHEMA,
            right_label="right",
            start_date=DAY,
            end_date=DAY,
            decision_time_left="09:35",
            decision_time_right="09:40",
            minute_path_mode="strict_contiguous",
            db_session=db_session,
        )


def test_snapshot_compare_requires_bounded_dates(db_session):
    compare = _load_compare()
    with pytest.raises(ValueError, match="start_date"):
        compare.compare_snapshots(
            left_schema=LEFT_SCHEMA,
            left_label="left",
            right_schema=RIGHT_SCHEMA,
            right_label="right",
            start_date=None,
            end_date=DAY,
            decision_time_left="09:35",
            decision_time_right="09:40",
            minute_path_mode="strict_contiguous",
            db_session=db_session,
        )


def test_snapshot_compare_overlap_counts(db_session):
    analysis = _analysis(db_session)

    assert analysis["overlap"]["passed_left_count"] == 2
    assert analysis["overlap"]["passed_right_count"] == 2
    assert analysis["overlap"]["passed_both_count"] == 1
    assert analysis["overlap"]["left_only_count"] == 1
    assert analysis["overlap"]["right_only_count"] == 1
    assert analysis["overlap"]["jaccard_overlap"] == pytest.approx(1 / 3)


def test_snapshot_compare_transition_matrix_counts(db_session):
    analysis = _analysis(db_session)

    assert analysis["transitions"]["matrix"] == {
        "left_failed -> right_failed": 1,
        "left_failed -> right_passed": 1,
        "left_passed -> right_failed": 1,
        "left_passed -> right_passed": 1,
    }


def test_snapshot_compare_quote_spread_and_liquidity_deltas(db_session):
    analysis = _analysis(db_session)
    deltas = analysis["liquidity_deltas"]

    assert deltas["paired_passed_count"] == 1
    assert deltas["entry_spread_bps_delta"]["mean"] == pytest.approx(50)
    assert deltas["entry_executable_notional_delta"]["mean"] == pytest.approx(50)
    assert deltas["observed_cumulative_volume_delta"]["mean"] == pytest.approx(1000)
    assert deltas["observed_minute_count_delta"]["mean"] == pytest.approx(5)


def test_snapshot_compare_missing_quote_and_cost_warning(db_session):
    analysis = _analysis(db_session)

    assert analysis["integrity"]["right"]["missing_quote_role_count"] == 1
    assert analysis["integrity"]["right"]["missing_cost_role_count"] == 1
    assert "right_missing_quote_evidence" in analysis["warnings"]
    assert "right_missing_cost_evidence" in analysis["warnings"]


def test_snapshot_compare_sql_uses_postgres_boolean_predicates():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compare_i12_pit_snapshots.py"
    )
    source = script_path.read_text()

    assert "is_active = 1" not in source
    assert "is_active IS TRUE" in source


def test_snapshot_compare_require_final_requires_reports(db_session):
    compare = _load_compare()

    with pytest.raises(RuntimeError, match="--require-final requires"):
        compare.compare_snapshots(
            left_schema=LEFT_SCHEMA,
            left_label="09:35",
            right_schema=RIGHT_SCHEMA,
            right_label="09:40",
            start_date=DAY,
            end_date=DAY,
            decision_time_left="09:35",
            decision_time_right="09:40",
            minute_path_mode="strict_contiguous",
            db_session=db_session,
            require_final=True,
        )


def test_snapshot_compare_require_final_validates_report_scope(db_session, tmp_path):
    compare = _load_compare()
    final_report = {
        "conclusions_final": True,
        "data_integrity_passed": True,
        "source_replay_complete": True,
        "quote_replay_complete": True,
        "cost_replay_complete": True,
        "schema": LEFT_SCHEMA,
        "start_date": DAY,
        "end_date": DAY,
        "report_path_mode": "strict_contiguous",
        "report_decision_time_labels": ["09:35"],
        "source_hur_schema": "public",
        "missing_source_attempt_count": 0,
        "extra_source_attempt_count": 0,
        "missing_source_attempt_identity_count": 0,
        "extra_source_attempt_identity_count": 0,
    }
    left_report = tmp_path / "left.json"
    right_report = tmp_path / "right.json"
    left_report.write_text(json.dumps(final_report))
    bad_right = dict(final_report)
    bad_right["schema"] = RIGHT_SCHEMA
    bad_right["report_decision_time_labels"] = ["09:35"]
    right_report.write_text(json.dumps(bad_right))

    with pytest.raises(RuntimeError, match="decision_time"):
        compare.compare_snapshots(
            left_schema=LEFT_SCHEMA,
            left_label="09:35",
            right_schema=RIGHT_SCHEMA,
            right_label="09:40",
            start_date=DAY,
            end_date=DAY,
            decision_time_left="09:35",
            decision_time_right="09:40",
            minute_path_mode="strict_contiguous",
            left_report=left_report,
            right_report=right_report,
            db_session=db_session,
            require_final=True,
        )


def test_snapshot_compare_require_final_rejects_metadata_less_final_report(db_session, tmp_path):
    compare = _load_compare()
    old_report = {
        "conclusions_final": True,
        "data_integrity_passed": True,
        "source_replay_complete": True,
        "quote_replay_complete": True,
        "cost_replay_complete": True,
        "missing_source_attempt_count": 0,
        "extra_source_attempt_count": 0,
        "missing_source_attempt_identity_count": 0,
        "extra_source_attempt_identity_count": 0,
    }
    left_report = tmp_path / "left-old.json"
    right_report = tmp_path / "right-old.json"
    left_report.write_text(json.dumps(old_report))
    right_report.write_text(json.dumps(old_report))

    with pytest.raises(RuntimeError, match="regenerate report with current code"):
        compare.compare_snapshots(
            left_schema=LEFT_SCHEMA,
            left_label="09:35",
            right_schema=RIGHT_SCHEMA,
            right_label="09:40",
            start_date=DAY,
            end_date=DAY,
            decision_time_left="09:35",
            decision_time_right="09:40",
            minute_path_mode="strict_contiguous",
            left_report=left_report,
            right_report=right_report,
            db_session=db_session,
            require_final=True,
        )


def test_snapshot_compare_fails_closed_on_duplicate_active_candidates(db_session):
    compare = _load_compare()
    _seed_compare_fixture(db_session)
    _insert_candidate(
        db_session,
        LEFT_SCHEMA,
        "left-a-duplicate",
        "AAA",
        decision_time="09:35",
        status="passed",
    )
    db_session.flush()

    with pytest.raises(RuntimeError, match="duplicate active ticker/date"):
        compare.compare_snapshots(
            left_schema=LEFT_SCHEMA,
            left_label="09:35",
            right_schema=RIGHT_SCHEMA,
            right_label="09:40",
            start_date=DAY,
            end_date=DAY,
            decision_time_left="09:35",
            decision_time_right="09:40",
            minute_path_mode="strict_contiguous",
            db_session=db_session,
        )


def test_snapshot_compare_fails_closed_on_duplicate_active_quote_roles(db_session):
    compare = _load_compare()
    _seed_compare_fixture(db_session)
    _insert_quote(
        db_session,
        LEFT_SCHEMA,
        "q-left-a-entry-duplicate",
        "left-a",
        "entry",
    )
    db_session.flush()

    with pytest.raises(RuntimeError, match="duplicate active child evidence"):
        compare.compare_snapshots(
            left_schema=LEFT_SCHEMA,
            left_label="09:35",
            right_schema=RIGHT_SCHEMA,
            right_label="09:40",
            start_date=DAY,
            end_date=DAY,
            decision_time_left="09:35",
            decision_time_right="09:40",
            minute_path_mode="strict_contiguous",
            db_session=db_session,
        )


def test_snapshot_compare_fails_closed_on_duplicate_active_cost_roles(db_session):
    compare = _load_compare()
    _seed_compare_fixture(db_session)
    _insert_cost(
        db_session,
        LEFT_SCHEMA,
        "c-left-a-sd-duplicate",
        "left-a",
        "same_day_exit",
        modeled_return=0.05,
    )
    db_session.flush()

    with pytest.raises(RuntimeError, match="duplicate active child evidence"):
        compare.compare_snapshots(
            left_schema=LEFT_SCHEMA,
            left_label="09:35",
            right_schema=RIGHT_SCHEMA,
            right_label="09:40",
            start_date=DAY,
            end_date=DAY,
            decision_time_left="09:35",
            decision_time_right="09:40",
            minute_path_mode="strict_contiguous",
            db_session=db_session,
        )


def test_snapshot_compare_json_output_shape(db_session):
    analysis = _analysis(db_session)

    assert set(analysis) >= {
        "integrity",
        "overlap",
        "transitions",
        "economics",
        "liquidity_deltas",
        "edge_timing",
        "samples",
        "warnings",
    }
    json.dumps(analysis, sort_keys=True, default=str)


def test_snapshot_compare_text_output_includes_sections(db_session):
    compare = _load_compare()
    text_output = compare.render_text(_analysis(db_session))

    assert "Corpus / Integrity" in text_output
    assert "Overlap" in text_output
    assert "Transition Matrix" in text_output
    assert "Liquidity Deltas" in text_output
