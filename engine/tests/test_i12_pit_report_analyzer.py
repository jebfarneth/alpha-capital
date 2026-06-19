from __future__ import annotations

import importlib.util
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from alpha.data.contracts import stable_hash
from alpha.db.models import I12PitCandidate, I12PitCostReplay, I12PitQuoteReplay


DAY = date(2026, 5, 1)
DECISION_TS = datetime(2026, 5, 1, 13, 40, tzinfo=timezone.utc)


def _load_analyzer():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analyze_i12_pit_reports.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_i12_pit_reports",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_i12_pit_analyzer_accepts_final_report(tmp_path):
    analyzer = _load_analyzer()
    payload = _report_payload(conclusions_final=True)
    payload["candidate_coverage_status_counts"] = {"daily_prefilter_skip": 2, "ok": 8}
    payload["daily_source_hash_basis_counts"] = {"legacy_per_date_raw_payload": 10}
    path = _write_report(tmp_path, payload)

    analysis = analyzer.analyze_report_paths([path], labels=["one-month"], require_final=True)

    assert analysis["require_final_failures"] == []
    report = analysis["reports"][0]
    assert report["integrity"]["conclusions_final"] is True
    assert report["integrity"]["source_replay_complete"] is True
    assert report["integrity"]["candidate_coverage_status_counts"] == {
        "daily_prefilter_skip": 2,
        "ok": 8,
    }
    assert report["integrity"]["daily_source_hash_basis_counts"] == {
        "legacy_per_date_raw_payload": 10,
    }
    assert report["exit_comparison"]["same_day_exit"]["tradeable_count"] == 8
    assert "report_non_final" not in report["warnings"]


def test_i12_pit_analyzer_require_final_returns_nonzero(tmp_path):
    analyzer = _load_analyzer()
    path = _write_report(
        tmp_path,
        _report_payload(
            conclusions_final=False,
            data_integrity_passed=False,
            training_status="blocked_quote_replay_incomplete",
        ),
    )

    rc = analyzer.main(["--report", str(path), "--require-final"])

    assert rc == 2


def test_i12_pit_analyzer_emits_quality_warnings(tmp_path):
    analyzer = _load_analyzer()
    payload = _report_payload()
    payload["quote_non_ok_count"] = 2
    payload["daily_source_hash_basis_counts"] = {"clean_slice_v1": 4, "unknown": 1}
    payload["daily_source_hash_reuse_status_counts"] = {
        "existing_active_attempt_reuse": 5,
    }
    payload["exit_metrics"]["same_day_exit"]["tradeable_rate"] = 0.50
    payload["exit_metrics"]["same_day_exit"]["mean_modeled_return_skips_as_cash"] = 0.01
    payload["exit_metrics"]["same_day_exit"]["win_rate_skips_as_cash"] = 0.40
    payload["exit_metrics"]["next_open_exit"]["mean_modeled_return_skips_as_cash"] = -0.01
    path = _write_report(tmp_path, payload)

    warnings = analyzer.analyze_report_paths([path])["reports"][0]["warnings"]

    assert "quote_non_ok_rows_present" in warnings
    assert "same_day_positive_mean_but_win_rate_below_50pct" in warnings
    assert "next_open_materially_worse_than_same_day" in warnings
    assert "same_day_exit_tradeable_rate_below_threshold" in warnings
    assert "daily_source_hash_clean_slice_v1_present" in warnings
    assert "daily_source_hash_unknown_basis_present" in warnings


def test_i12_pit_analyzer_does_not_warn_clean_slice_for_reused_legacy_rows(tmp_path):
    analyzer = _load_analyzer()
    payload = _report_payload()
    payload["daily_source_hash_basis_counts"] = {"legacy_per_date_raw_payload": 4}
    payload["daily_source_hash_reuse_status_counts"] = {
        "existing_active_attempt_reuse": 4,
    }
    path = _write_report(tmp_path, payload)

    warnings = analyzer.analyze_report_paths([path])["reports"][0]["warnings"]

    assert "daily_source_hash_clean_slice_v1_present" not in warnings
    assert "daily_source_hash_unknown_basis_present" not in warnings


@pytest.mark.parametrize("schema", ["public", "canonical", "default", "analysis"])
def test_i12_pit_analyzer_refuses_non_scratch_schema(tmp_path, schema):
    analyzer = _load_analyzer()
    path = _write_report(tmp_path, _report_payload())

    with pytest.raises(ValueError, match="named scratch schema"):
        analyzer.analyze_report_paths([path], schema=schema, db_session=object())


def test_i12_pit_analyzer_db_backed_monthly_daily_sections(
    db_session,
    monkeypatch,
    tmp_path,
):
    analyzer = _load_analyzer()
    path = _write_report(tmp_path, _report_payload())
    _persist_candidate_with_replay(db_session, "A", DAY, 0.10, -0.02)
    _persist_candidate_with_replay(db_session, "B", date(2026, 5, 2), 0.01, -0.01)
    _persist_candidate_with_replay(db_session, "C", date(2026, 6, 1), 0.01, 0.02)
    db_session.commit()
    monkeypatch.setattr(
        analyzer,
        "_load_candidate_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("row loader should not be used")
        ),
    )

    analysis = analyzer.analyze_report_paths(
        [path],
        labels=["db"],
        schema="scratch_test",
        db_session=db_session,
    )
    json.loads(json.dumps(analysis, default=str))
    report = analysis["reports"][0]

    assert [row["month"] for row in report["monthly_stability"]["rows"]] == [
        "2026-05",
        "2026-06",
    ]
    assert report["monthly_stability"]["rows"][0]["passed_candidates"] == 2
    assert len(report["daily_distribution"]["rows"]) == 3
    summary = report["daily_distribution"]["summary"]["same_day_exit"]
    assert summary["day_count"] == 3
    assert summary["positive_day_count"] == 3
    assert summary["top_3_positive_return_share"] == pytest.approx(1.0)
    assert "top_3_days_concentrated_positive_return" in report["warnings"]


def test_i12_pit_analyzer_db_backed_refuses_unbounded_by_default(db_session, tmp_path):
    analyzer = _load_analyzer()
    payload = _report_payload()
    payload.pop("start_date")
    payload.pop("end_date")
    path = _write_report(tmp_path, payload)

    with pytest.raises(ValueError, match="requires bounded dates"):
        analyzer.analyze_report_paths(
            [path],
            labels=["unbounded"],
            schema="scratch_test",
            db_session=db_session,
        )


def test_i12_pit_analyzer_closes_owned_db_session(monkeypatch, tmp_path):
    analyzer = _load_analyzer()
    path = _write_report(tmp_path, _report_payload())

    class SpySession:
        def __init__(self):
            self.rollback_called = False
            self.close_called = False

        def in_transaction(self):
            return True

        def rollback(self):
            self.rollback_called = True

        def close(self):
            self.close_called = True

    spy_session = SpySession()
    monkeypatch.setattr(analyzer, "create_engine", lambda url: object())
    monkeypatch.setattr(analyzer, "Session", lambda engine: spy_session)
    monkeypatch.setattr(
        analyzer,
        "_analyze_loaded_report",
        lambda loaded, *, db_context, tradeable_rate_threshold: {
            "label": loaded.label,
            "integrity": {
                "conclusions_final": True,
                "data_integrity_passed": True,
            },
            "warnings": [],
        },
    )

    analysis = analyzer.analyze_report_paths(
        [path],
        labels=["owned"],
        schema="scratch_owned",
        database_url="postgresql://example.invalid/db",
        start_date="2026-05-01",
        end_date="2026-05-01",
    )

    assert analysis["reports"][0]["label"] == "owned"
    assert spy_session.rollback_called is True
    assert spy_session.close_called is True


def test_i12_pit_analyzer_daily_distribution_includes_no_trade_source_days(
    db_session,
    tmp_path,
):
    analyzer = _load_analyzer()
    path = _write_report(tmp_path, _report_payload())
    _persist_candidate_with_replay(db_session, "TRADE", DAY, 0.02, -0.01)
    _persist_failed_candidate(db_session, "NOTRADE", date(2026, 5, 2))
    db_session.commit()

    report = analyzer.analyze_report_paths(
        [path],
        labels=["db"],
        schema="scratch_test",
        db_session=db_session,
        start_date="2026-05-01",
        end_date="2026-05-02",
    )["reports"][0]

    daily = report["daily_distribution"]
    assert daily["source_day_count"] == 2
    assert daily["candidate_day_count"] == 2
    assert daily["trading_candidate_day_count"] == 1
    assert daily["no_trade_day_count"] == 1
    no_trade = next(
        row for row in daily["rows"] if row["decision_date"] == "2026-05-02"
    )
    assert no_trade["passed_candidates"] == 0
    assert no_trade["same_day_exit"]["mean_modeled_return_skips_as_cash"] == 0.0
    assert daily["summary"]["same_day_exit"]["day_count"] == 2
    assert daily["candidate_days_only_summary"]["same_day_exit"]["day_count"] == 1


def _write_report(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload))
    return path


def _report_payload(
    *,
    conclusions_final: bool = True,
    data_integrity_passed: bool = True,
    training_status: str = "eligible_for_retrain_evaluation",
) -> dict:
    return {
        "schema": "scratch_test",
        "start_date": "2026-05-01",
        "end_date": "2026-06-05",
        "report_path_mode": "strict_contiguous",
        "report_decision_time_labels": ["09:40"],
        "conclusions_final": conclusions_final,
        "data_integrity_passed": data_integrity_passed,
        "training_status": training_status,
        "source_replay_complete": True,
        "quote_replay_complete": True,
        "cost_replay_complete": True,
        "expected_candidate_attempts": 10,
        "missing_source_attempt_count": 0,
        "extra_source_attempt_count": 0,
        "missing_quote_role_count": 0,
        "duplicate_quote_role_count": 0,
        "missing_cost_role_count": 0,
        "duplicate_cost_role_count": 0,
        "unknown_quote_coverage_status_count": 0,
        "quote_ok_count": 30,
        "quote_non_ok_count": 0,
        "quote_ok_rate": 1.0,
        "pit_candidate_count": 10,
        "exit_metrics": {
            "same_day_exit": _exit_metrics(mean_return=0.005, win_rate=0.55),
            "next_open_exit": _exit_metrics(mean_return=-0.005, win_rate=0.35),
        },
    }


def _exit_metrics(*, mean_return: float, win_rate: float) -> dict:
    return {
        "candidates": 10,
        "tradeable_count": 8,
        "tradeable_rate": 0.80,
        "skipped_cash_count": 2,
        "skipped_cash_by_reason": {"size": 2},
        "mean_modeled_return_skips_as_cash": mean_return,
        "win_rate_skips_as_cash": win_rate,
        "mean_quote_cost_return_tradeable": mean_return,
        "spread_bps": {"p50": 50, "p75": 75, "p90": 100},
        "executable_notional": {"p50": 500, "p75": 750, "p90": 1000},
        "top_of_book_sufficient_rate": 0.80,
    }


def _persist_candidate_with_replay(
    db_session,
    ticker: str,
    day: date,
    same_day_return: float,
    next_open_return: float,
) -> None:
    candidate = I12PitCandidate(
        i12_pit_candidate_id=f"cand-{ticker}",
        ticker=ticker,
        decision_date=day,
        decision_ts=datetime(day.year, day.month, day.day, 13, 40, tzinfo=timezone.utc),
        decision_time_label="09:40",
        path_mode="strict_contiguous",
        feature_asof_ts=DECISION_TS,
        candidate_status="passed",
        coverage_status="ok",
        feature_json="{}",
        gate_values_json="{}",
        leakage_guard_json="{}",
        source_bars_json="{}",
        candidate_attempt_hash=f"attempt-{ticker}",
        is_active=True,
        input_hash=f"input-{ticker}",
        candidate_identity_hash=f"identity-{ticker}",
        label_hash=f"label-{ticker}",
        content_hash=f"content-{ticker}",
    )
    db_session.add(candidate)
    db_session.flush()
    for role in ("entry", "same_day_exit", "next_open_exit"):
        db_session.add(
            I12PitQuoteReplay(
                i12_pit_quote_replay_id=f"quote-{ticker}-{role}",
                i12_pit_candidate_id=candidate.i12_pit_candidate_id,
                ticker=ticker,
                decision_date=day,
                decision_ts=candidate.decision_ts,
                quote_role=role,
                target_ts=candidate.decision_ts,
                window_start_ts=candidate.decision_ts,
                window_end_ts=candidate.decision_ts,
                quote_ts=candidate.decision_ts,
                quote_age_seconds=0.0,
                bid=10.0,
                ask=10.05,
                bid_size=100,
                ask_size=100,
                spread_bps=49.875,
                top_of_book_notional=1000,
                bid_notional=1000,
                ask_notional=1005,
                executable_notional=1005 if role == "entry" else 1000,
                executable_side="buy" if role == "entry" else "sell",
                feed="sip",
                source="fixture",
                quote_size_basis="shares_post_2025_11_03",
                coverage_status="ok",
                quote_replay_attempt_hash=f"quote-attempt-{ticker}-{role}",
                is_active=True,
                content_hash=stable_hash({"ticker": ticker, "role": role}),
            )
        )
    for role, modeled_return in {
        "same_day_exit": same_day_return,
        "next_open_exit": next_open_return,
    }.items():
        db_session.add(
            I12PitCostReplay(
                i12_pit_cost_replay_id=f"cost-{ticker}-{role}",
                i12_pit_candidate_id=candidate.i12_pit_candidate_id,
                ticker=ticker,
                decision_date=day,
                decision_ts=candidate.decision_ts,
                exit_role=role,
                tradeability_status="tradeable",
                skipped_reason="none",
                intended_order_usd=250,
                max_spread_bps=200,
                slippage_bps=0,
                entry_ask=10.05,
                exit_bid=10.05 * (1 + modeled_return),
                gross_return=modeled_return,
                quote_cost_return=modeled_return,
                slippage_return=modeled_return,
                modeled_return=modeled_return,
                cost_replay_attempt_hash=f"cost-attempt-{ticker}-{role}",
                is_active=True,
                content_hash=stable_hash({"ticker": ticker, "role": role, "return": modeled_return}),
            )
        )


def _persist_failed_candidate(db_session, ticker: str, day: date) -> None:
    candidate = I12PitCandidate(
        i12_pit_candidate_id=f"cand-{ticker}",
        ticker=ticker,
        decision_date=day,
        decision_ts=datetime(day.year, day.month, day.day, 13, 40, tzinfo=timezone.utc),
        decision_time_label="09:40",
        path_mode="strict_contiguous",
        feature_asof_ts=DECISION_TS,
        candidate_status="failed",
        coverage_status="partial_minute_path",
        fail_reason="partial_minute_path",
        feature_json="{}",
        gate_values_json="{}",
        leakage_guard_json="{}",
        source_bars_json="{}",
        candidate_attempt_hash=f"attempt-{ticker}",
        is_active=True,
        input_hash=f"input-{ticker}",
        candidate_identity_hash=f"identity-{ticker}",
        label_hash=f"label-{ticker}",
        content_hash=f"content-{ticker}",
    )
    db_session.add(candidate)
