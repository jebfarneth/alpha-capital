import json
import os
import pickle
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from alpha.data.alpaca import AlpacaClock, AlpacaQuote, AlpacaStockSnapshot
from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError
from alpha.db.models import (
    Base,
    FeatureSnapshot,
    I12FillLog,
    MLModelRegistry,
    SignalMLScore,
    SignalRegistry,
)
from alpha.jobs.i12_live_fill_test import (
    ALPACA_QUOTE_SIZE_BASIS,
    EXPECTED_I12_LIVE_FEATURES,
    I12LiveFillConfig,
    I12LiveFillTestJob,
    assert_i12_live_feature_payload_leakage_clean,
    build_i12_live_feature_payload,
    capture_i12_exit_quotes,
    evaluate_quote_liquidity,
    i12_gate0_report,
    select_i12_model,
    _stage0_run_config_hash,
    validate_i12_stage0_model_contract,
)
from alpha.jobs.paper_execution import PremarketContext
from alpha.jobs.runner import run_job
from alpha.jobs.run_i12_live_fill_test import (
    I12_FILL_TEST_REQUIRED_TABLES,
    _copy_model_registry_values,
    ensure_model_registry_row_in_scratch,
    require_stage0_scratch_schema,
    validate_i12_stage0_artifact_preflight,
)
import alpha.jobs.run_i12_live_fill_test as run_i12_live_fill_test
from alpha.db.engine import SchemaTargetError
from alpha.ml.model_features import feature_schema_hash


TRADING_DATE = date(2026, 6, 16)
DECISION_TS = datetime(2026, 6, 16, 13, 40, tzinfo=timezone.utc)


class DummyPredictModel:
    def predict(self, values):
        return [0.25 for _ in values]


class RangeCheckingPredictModel:
    def __init__(self, feature_name, minimum, maximum):
        self.feature_name = feature_name
        self.minimum = minimum
        self.maximum = maximum

    def predict(self, values):
        idx = EXPECTED_I12_LIVE_FEATURES.index(self.feature_name)
        observed = float(values[0][idx])
        if not self.minimum <= observed <= self.maximum:
            raise RuntimeError(f"{self.feature_name} out of range: {observed}")
        return [0.25]


class FakeAlpaca:
    def __init__(self, *, snapshots=None, quotes=None, clock_open=True, snapshot_error=False):
        self.snapshots = snapshots or {}
        self.quotes = quotes or {}
        self.clock_open = clock_open
        self.snapshot_error = snapshot_error

    def get_clock(self):
        return AdapterResponse(
            data=AlpacaClock(
                timestamp=DECISION_TS.isoformat(),
                is_open=self.clock_open,
            ),
            lineage=_lineage("/v2/clock"),
        )

    def get_stock_snapshots(self, symbols, *, feed="iex"):
        del feed
        if self.snapshot_error:
            return AdapterResponse(
                data=None,
                lineage=_lineage("/v2/stocks/snapshots"),
                error=ProviderError(
                    provider="Alpaca",
                    endpoint="/v2/stocks/snapshots",
                    status_code=500,
                    error_type="http",
                    message="snapshot batch failed",
                    retryable=True,
                ),
            )
        return AdapterResponse(
            data={
                symbol.upper(): self.snapshots[symbol.upper()]
                for symbol in symbols
                if symbol.upper() in self.snapshots
            },
            lineage=_lineage("/v2/stocks/snapshots"),
        )

    def get_latest_quotes(self, symbols, *, feed="iex"):
        del feed
        return AdapterResponse(
            data={
                symbol.upper(): self.quotes[symbol.upper()]
                for symbol in symbols
                if symbol.upper() in self.quotes
            },
            lineage=_lineage("/v2/stocks/quotes/latest"),
        )


def test_i12_live_feature_payload_rejects_full_day_and_forward_fields():
    context = _context("LIVE")
    payload = build_i12_live_feature_payload(
        context,
        {
            "distance_from_max252": -0.60,
            "gap": 0.01,
            "projected_volume_ratio_at_confirmation": 6.0,
            "projected_volume_at_confirmation": 600_000,
        },
    )
    assert_i12_live_feature_payload_leakage_clean(payload)

    leaky = dict(payload)
    leaky["full_day_volume"] = 1_000_000
    with pytest.raises(RuntimeError, match="leaky live I12 feature path"):
        assert_i12_live_feature_payload_leakage_clean(leaky)

    leaky_forward = dict(payload)
    leaky_forward["forward_return"] = 0.12
    with pytest.raises(RuntimeError, match="leaky live I12 feature path"):
        assert_i12_live_feature_payload_leakage_clean(leaky_forward)

    leaky_contract = dict(payload)
    leaky_contract["leakage_contract"] = {
        "decision_basis": "live_intraday_projected_volume",
        "uses_full_day_volume": False,
        "uses_forward_bars": True,
    }
    with pytest.raises(RuntimeError, match="uses_forward_bars=false"):
        assert_i12_live_feature_payload_leakage_clean(leaky_contract)


def test_i12_stage0_logs_top_k_and_counts_liquidity_skips(db_session, monkeypatch):
    model = _add_model(db_session)
    snapshots = {
        "AAA": _snapshot("AAA", quote=_quote("AAA", bid=10.00, ask=10.05, ask_size=100)),
        "BBB": _snapshot("BBB", quote=_quote("BBB", bid=10.00, ask=10.35, ask_size=100)),
        "CCC": _snapshot("CCC", quote=_quote("CCC", bid=10.00, ask=10.04, ask_size=100)),
    }
    _patch_scores(monkeypatch, {"AAA": 0.90, "BBB": 0.80, "CCC": 0.10})
    job = I12LiveFillTestJob(
        session=db_session,
        alpaca_adapter=FakeAlpaca(snapshots=snapshots),
        contexts={ticker: _context(ticker) for ticker in snapshots},
        config=I12LiveFillConfig(
            model_id=model.model_id,
            top_k=2,
            require_market_open=False,
        ),
        snapshots=snapshots,
        asof=DECISION_TS,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.status == "finished"
    assert result.metrics["fire_count"] == 3
    assert result.metrics["selected_top_k"] == 2
    assert result.metrics["skipped_as_cash_count"] == 1
    rows = {row.ticker: row for row in db_session.query(I12FillLog).all()}
    assert set(rows) == {"AAA", "BBB", "CCC"}
    assert rows["AAA"].skipped_reason == "none"
    assert rows["BBB"].skipped_reason == "spread"
    assert rows["BBB"].exit_capture_status == "skipped_cash"
    assert rows["BBB"].modeled_return == pytest.approx(0.0)
    assert rows["CCC"].selection_status == "not_selected"
    assert rows["AAA"].intended_order_usd == pytest.approx(250.0)
    assert rows["AAA"].feed == "sip"
    assert rows["AAA"].stage0_run_config_hash
    assert _stored_test_utc(rows["AAA"].minute_ts) == DECISION_TS
    assert rows["AAA"].minute_age_seconds == pytest.approx(0.0)
    assert _stored_test_utc(rows["AAA"].latest_trade_ts) == DECISION_TS
    assert rows["AAA"].latest_trade_age_seconds == pytest.approx(0.0)
    assert _stored_test_utc(rows["AAA"].quote_ts) == DECISION_TS
    assert rows["AAA"].quote_age_seconds == pytest.approx(0.0)
    assert rows["AAA"].entry_quote_age_seconds == pytest.approx(0.0)
    assert result.metrics["coverage_gate_passed"] is True
    assert json.loads(rows["AAA"].feature_json)["leakage_contract"] == {
        "decision_basis": "live_intraday_projected_volume",
        "uses_forward_bars": False,
        "uses_full_day_volume": False,
    }


def test_i12_stage0_fallback_scores_do_not_create_intended_trade_logs(
    db_session,
    monkeypatch,
):
    model = _add_model(db_session)
    snapshots = {"AAA": _snapshot("AAA", quote=_quote("AAA"))}
    _patch_scores(monkeypatch, {"AAA": 0.90}, source="fallback_raw_strength")
    job = I12LiveFillTestJob(
        session=db_session,
        alpaca_adapter=FakeAlpaca(snapshots=snapshots),
        contexts={"AAA": _context("AAA")},
        config=I12LiveFillConfig(model_id=model.model_id, require_market_open=False),
        snapshots=snapshots,
        asof=DECISION_TS,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.status == "finished"
    assert result.metrics["fire_count"] == 1
    assert result.metrics["score_model_ok"] == 0
    assert result.metrics["score_fallback"] == 1
    assert result.metrics["logged_intended_trades"] == 0
    row = db_session.query(I12FillLog).one()
    assert row.score_stage0_status == "fallback"
    assert row.selection_status == "not_selected"


def test_i12_stage0_scoring_exception_is_logged_not_dropped(db_session, monkeypatch):
    model = _add_model(db_session)
    snapshots = {"AAA": _snapshot("AAA", quote=_quote("AAA"))}
    import alpha.jobs.i12_live_fill_test as module

    def boom(*args, **kwargs):
        raise RuntimeError("score service down")

    monkeypatch.setattr(module, "score_signal_shadow", boom)
    job = I12LiveFillTestJob(
        session=db_session,
        alpaca_adapter=FakeAlpaca(snapshots=snapshots),
        contexts={"AAA": _context("AAA")},
        config=I12LiveFillConfig(model_id=model.model_id, require_market_open=False),
        snapshots=snapshots,
        asof=DECISION_TS,
    )

    result = run_job(db_session, job, params={"test": True})
    row = db_session.query(I12FillLog).one()
    report = i12_gate0_report(
        db_session,
        decision_date=TRADING_DATE,
        asof=DECISION_TS,
    )

    assert result.status == "finished"
    assert row.score_stage0_status == "failed"
    assert "score service down" in row.coverage_error
    assert report["score_failed"] == 1


def test_i12_stage0_fill_log_is_idempotent(db_session, monkeypatch):
    model = _add_model(db_session)
    snapshots = {"AAA": _snapshot("AAA", quote=_quote("AAA"))}
    _patch_scores(monkeypatch, {"AAA": 0.90})
    config = I12LiveFillConfig(
        model_id=model.model_id,
        context_artifact_hash="context-a",
        require_market_open=False,
    )

    for _ in range(2):
        job = I12LiveFillTestJob(
            session=db_session,
            alpaca_adapter=FakeAlpaca(snapshots=snapshots),
            contexts={"AAA": _context("AAA")},
            config=config,
            snapshots=snapshots,
            asof=DECISION_TS,
        )
        result = run_job(db_session, job, params={"test": True})
        assert result.status == "finished"

    assert db_session.query(I12FillLog).count() == 1


def test_i12_stage0_same_day_context_artifact_drift_is_recorded_and_non_promotable(
    db_session,
    monkeypatch,
):
    model = _add_model(db_session)
    snapshots = {"AAA": _snapshot("AAA", quote=_quote("AAA"))}
    _patch_scores(monkeypatch, {"AAA": 0.90})

    for context_hash in ("context-a", "context-b"):
        job = I12LiveFillTestJob(
            session=db_session,
            alpaca_adapter=FakeAlpaca(snapshots=snapshots),
            contexts={"AAA": _context("AAA")},
            config=I12LiveFillConfig(
                model_id=model.model_id,
                context_artifact_hash=context_hash,
                require_market_open=False,
            ),
            snapshots=snapshots,
            asof=DECISION_TS,
        )
        result = run_job(db_session, job, params={"test": True})
        assert result.status == "finished"

    rows = db_session.query(I12FillLog).order_by(I12FillLog.context_artifact_hash).all()
    report = i12_gate0_report(
        db_session,
        decision_date=TRADING_DATE,
        asof=DECISION_TS,
        min_gate0_intended_count=0,
        min_gate0_distinct_trading_days=0,
    )

    assert len(rows) == 2
    assert {row.stage0_run_config_hash for row in rows} == {
        rows[0].stage0_run_config_hash
    }
    assert [row.context_artifact_hash for row in rows] == ["context-a", "context-b"]
    assert report["stage0_run_config_hashes"] == [rows[0].stage0_run_config_hash]
    assert report["context_artifact_hashes_by_day"] == {
        TRADING_DATE.isoformat(): ["context-a", "context-b"]
    }
    assert report["mixed_context_artifact_days"] == [TRADING_DATE.isoformat()]
    assert (
        "mixed_context_artifact_hash_for_day"
        in report["non_promotable_reasons"]
    )
    assert report["passed"] is False


def test_i12_stage0_policy_hash_ignores_context_artifact_hash(db_session):
    model = _add_model(db_session)
    contract = select_i12_model(
        db_session,
        model_id=model.model_id,
        allow_latest_model=False,
    )
    config_a = I12LiveFillConfig(
        model_id=model.model_id,
        context_artifact_hash="context-a",
        require_market_open=False,
    )
    config_b = I12LiveFillConfig(
        model_id=model.model_id,
        context_artifact_hash="context-b",
        require_market_open=False,
    )

    assert _stage0_run_config_hash(config_a, contract) == _stage0_run_config_hash(
        config_b,
        contract,
    )


def test_i12_stage0_policy_hash_includes_quote_size_basis(db_session, monkeypatch):
    import alpha.jobs.i12_live_fill_test as live_fill_test

    model = _add_model(db_session)
    contract = select_i12_model(
        db_session,
        model_id=model.model_id,
        allow_latest_model=False,
    )
    config = I12LiveFillConfig(
        model_id=model.model_id,
        require_market_open=False,
    )
    original_hash = _stage0_run_config_hash(config, contract)

    monkeypatch.setattr(
        live_fill_test,
        "ALPACA_QUOTE_SIZE_BASIS",
        "shares_post_future_change",
    )

    assert _stage0_run_config_hash(config, contract) != original_hash


def test_i12_stage0_repeated_poll_preserves_first_intended_row(db_session, monkeypatch):
    model = _add_model(db_session)
    first_snapshot = _snapshot(
        "AAA",
        quote=_quote("AAA", bid=10.00, ask=10.05, timestamp=DECISION_TS),
        timestamp=DECISION_TS,
    )
    _patch_scores(monkeypatch, {"AAA": 0.90})
    config = I12LiveFillConfig(model_id=model.model_id, require_market_open=False)
    first_job = I12LiveFillTestJob(
        session=db_session,
        alpaca_adapter=FakeAlpaca(snapshots={"AAA": first_snapshot}),
        contexts={"AAA": _context("AAA")},
        config=config,
        snapshots={"AAA": first_snapshot},
        asof=DECISION_TS,
    )
    run_job(db_session, first_job, params={"test": True})
    row = db_session.query(I12FillLog).one()
    first_state = {
        "signal_id": row.signal_id,
        "score_id": row.score_id,
        "feature_json": row.feature_json,
        "gate_values_json": row.gate_values_json,
        "ml_score": row.ml_score,
        "ask": row.ask,
        "quote_json": row.quote_json,
        "quote_ts": row.quote_ts,
        "snapshot_ts": row.snapshot_ts,
        "status": row.selection_status,
    }

    second_ts = DECISION_TS + timedelta(minutes=1)
    second_snapshot = _snapshot(
        "AAA",
        quote=_quote("AAA", bid=20.00, ask=20.50, timestamp=second_ts),
        timestamp=second_ts,
    )
    second_job = I12LiveFillTestJob(
        session=db_session,
        alpaca_adapter=FakeAlpaca(snapshots={"AAA": second_snapshot}),
        contexts={"AAA": _context("AAA")},
        config=config,
        snapshots={"AAA": second_snapshot},
        asof=second_ts,
    )
    run_job(db_session, second_job, params={"test": True})
    row = db_session.query(I12FillLog).one()

    assert row.signal_id == first_state["signal_id"]
    assert row.score_id == first_state["score_id"]
    assert row.feature_json == first_state["feature_json"]
    assert row.gate_values_json == first_state["gate_values_json"]
    assert row.ml_score == first_state["ml_score"]
    assert row.ask == first_state["ask"]
    assert row.quote_json == first_state["quote_json"]
    assert row.quote_ts == first_state["quote_ts"]
    assert row.snapshot_ts == first_state["snapshot_ts"]
    assert row.selection_status == first_state["status"]
    assert db_session.query(I12FillLog).count() == 1


def test_i12_stage0_exit_quote_capture_and_gate0_report(db_session, monkeypatch):
    model = _add_model(db_session)
    snapshots = {"AAA": _snapshot("AAA", quote=_quote("AAA", bid=10.00, ask=10.05))}
    _patch_scores(monkeypatch, {"AAA": 0.90})
    job = I12LiveFillTestJob(
        session=db_session,
        alpaca_adapter=FakeAlpaca(snapshots=snapshots),
        contexts={"AAA": _context("AAA")},
        config=I12LiveFillConfig(model_id=model.model_id, require_market_open=False),
        snapshots=snapshots,
        asof=DECISION_TS,
    )
    run_job(db_session, job, params={"test": True})

    exit_asof = datetime(2026, 6, 17, 13, 31, tzinfo=timezone.utc)
    exit_quote = _quote("AAA", bid=10.50, ask=10.55, timestamp=exit_asof)
    result = capture_i12_exit_quotes(
        db_session,
        FakeAlpaca(quotes={"AAA": exit_quote}),
        asof=exit_asof,
    )
    report = i12_gate0_report(
        db_session,
        decision_date=TRADING_DATE,
        asof=DECISION_TS,
    )

    assert result["exit_quote_updates"] == 1
    row = db_session.query(I12FillLog).one()
    assert row.exit_bid == pytest.approx(10.50)
    assert row.exit_capture_status == "ok"
    assert row.exit_quote_age_seconds == pytest.approx(0.0)
    assert row.modeled_return == pytest.approx((10.50 / 10.05) - 1.0)
    assert report["context_count"] == 1
    assert report["tradeable"] == 1
    assert report["passed"] is False
    assert "min_gate0_intended_count" in report["coverage_gate_failures"]
    assert "min_gate0_distinct_trading_days" in report["coverage_gate_failures"]


def test_i12_stage0_gate0_one_perfect_trade_does_not_pass_promotion(db_session):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    _add_gate0_trade_row(db_session, ticker="AAA", day=TRADING_DATE, asof=asof)

    report = i12_gate0_report(db_session, asof=asof)

    assert report["intended_count"] == 1
    assert report["distinct_trading_days"] == 1
    assert report["tradeable"] == 1
    assert report["coverage_gate_passed"] is False
    assert "min_gate0_intended_count" in report["coverage_gate_failures"]
    assert "min_gate0_distinct_trading_days" in report["coverage_gate_failures"]
    assert report["passed"] is False


def test_i12_stage0_gate0_enough_rows_too_few_days_does_not_pass(db_session):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    days = [TRADING_DATE, TRADING_DATE + timedelta(days=1)]
    for idx in range(20):
        _add_gate0_trade_row(
            db_session,
            ticker=f"T{idx:03d}",
            day=days[idx % len(days)],
            asof=asof,
        )

    report = i12_gate0_report(db_session, asof=asof)

    assert report["intended_count"] == 20
    assert report["distinct_trading_days"] == 2
    assert "min_gate0_intended_count" not in report["coverage_gate_failures"]
    assert "min_gate0_distinct_trading_days" in report["coverage_gate_failures"]
    assert report["passed"] is False


def test_i12_stage0_gate0_context_days_do_not_satisfy_intended_day_gate(db_session):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    intended_day = TRADING_DATE
    for idx in range(20):
        _add_gate0_trade_row(
            db_session,
            ticker=f"T{idx:03d}",
            day=intended_day,
            asof=asof,
        )
    _add_gate0_context_row(
        db_session,
        ticker="CTX1",
        day=TRADING_DATE + timedelta(days=1),
    )
    _add_gate0_context_row(
        db_session,
        ticker="CTX2",
        day=TRADING_DATE + timedelta(days=2),
    )

    report = i12_gate0_report(db_session, asof=asof)

    assert report["context_distinct_trading_days"] == 3
    assert report["intended_distinct_trading_days"] == 1
    assert report["distinct_trading_days"] == 1
    assert report["intended_count"] == 20
    assert "min_gate0_intended_count" not in report["coverage_gate_failures"]
    assert "min_gate0_distinct_trading_days" in report["coverage_gate_failures"]
    assert report["passed"] is False


def test_i12_stage0_gate0_enough_rows_days_and_coverage_can_pass(db_session):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    days = [
        TRADING_DATE,
        TRADING_DATE + timedelta(days=1),
        TRADING_DATE + timedelta(days=2),
    ]
    for idx in range(20):
        _add_gate0_trade_row(
            db_session,
            ticker=f"T{idx:03d}",
            day=days[idx % len(days)],
            asof=asof,
        )

    report = i12_gate0_report(db_session, asof=asof)

    assert report["intended_count"] == 20
    assert report["context_distinct_trading_days"] == 3
    assert report["intended_distinct_trading_days"] == 3
    assert report["distinct_trading_days"] == 3
    assert report["spread_ok"] == 20
    assert report["size_ok"] == 20
    assert report["evidence_conflict_count"] == 0
    assert report["entry_integrity_conflict_count"] == 0
    assert report["coverage_gate_failures"] == []
    assert report["coverage_gate_passed"] is True
    assert report["passed"] is True


def test_i12_stage0_gate0_allows_daily_context_hashes_for_same_policy(db_session):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    days = [
        TRADING_DATE,
        TRADING_DATE + timedelta(days=1),
        TRADING_DATE + timedelta(days=2),
    ]
    for idx in range(20):
        day = days[idx % len(days)]
        _add_gate0_trade_row(
            db_session,
            ticker=f"T{idx:03d}",
            day=day,
            asof=asof,
            config_hash="same-policy",
            context_hash=f"context-{day.isoformat()}",
        )

    report = i12_gate0_report(db_session, asof=asof)

    assert report["intended_count"] == 20
    assert report["intended_distinct_trading_days"] == 3
    assert report["stage0_run_config_hash"] == "same-policy"
    assert report["stage0_run_config_hashes"] == ["same-policy"]
    assert report["context_artifact_hashes"] == [
        f"context-{day.isoformat()}" for day in days
    ]
    assert report["context_artifact_hashes_by_day"] == {
        day.isoformat(): [f"context-{day.isoformat()}"] for day in days
    }
    assert report["mixed_context_artifact_days"] == []
    assert report["missing_context_artifact_hash_count"] == 0
    assert "mixed_or_missing_stage0_run_config" not in report["non_promotable_reasons"]
    assert "mixed_context_artifact_hash_for_day" not in report["non_promotable_reasons"]
    assert report["coverage_gate_failures"] == []
    assert report["passed"] is True


def test_i12_stage0_gate0_fails_when_tradeable_rate_is_too_low(db_session):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    days = [
        TRADING_DATE,
        TRADING_DATE + timedelta(days=1),
        TRADING_DATE + timedelta(days=2),
    ]
    for idx in range(20):
        row = _add_gate0_trade_row(
            db_session,
            ticker=f"T{idx:03d}",
            day=days[idx % len(days)],
            asof=asof,
        )
        if idx >= 10:
            _mark_gate0_row_skipped_cash(row, reason="size")

    report = i12_gate0_report(db_session, asof=asof)

    assert report["intended_count"] == 20
    assert report["tradeable"] == 10
    assert report["tradeable_rate"] == pytest.approx(0.5)
    assert report["coverage_thresholds"]["min_gate0_tradeable_rate"] == pytest.approx(0.70)
    assert "min_gate0_tradeable_rate" in report["coverage_gate_failures"]
    assert report["quote_ok_rate"] == pytest.approx(1.0)
    assert report["exit_quote_ok_rate"] == pytest.approx(1.0)
    assert report["passed"] is False


@pytest.mark.parametrize(
    (
        "mutate_row",
        "report_kwargs",
        "expected_spread_ok",
        "expected_size_ok",
        "expected_entry_integrity_conflicts",
    ),
    [
        (
            lambda row: setattr(row, "bid", None),
            {},
            1,
            1,
            1,
        ),
        (
            lambda row: setattr(row, "ask", None),
            {},
            1,
            1,
            1,
        ),
        (
            lambda row: setattr(row, "bid", 0.0),
            {},
            1,
            1,
            1,
        ),
        (
            lambda row: (setattr(row, "bid", 10.0), setattr(row, "ask", 9.99)),
            {},
            1,
            1,
            1,
        ),
        (
            lambda row: setattr(row, "spread_bps", -1.0),
            {},
            1,
            1,
            1,
        ),
        (
            lambda row: setattr(row, "spread_bps", 250.0),
            {},
            1,
            1,
            1,
        ),
        (
            lambda row: setattr(row, "top_of_book_size", 249.0),
            {},
            1,
            1,
            1,
        ),
        (
            lambda row: setattr(row, "size_sufficient", False),
            {},
            1,
            0,
            1,
        ),
        (
            lambda row: setattr(row, "quote_status", "stale"),
            {},
            1,
            1,
            1,
        ),
        (
            lambda row: setattr(row, "quote_condition_halt_inferred", True),
            {},
            1,
            1,
            1,
        ),
        (
            lambda row: row,
            {"max_spread_bps": 10.0},
            0,
            1,
            0,
        ),
        (
            lambda row: row,
            {"intended_order_usd": 2_000.0},
            1,
            0,
            0,
        ),
    ],
)
def test_i12_stage0_gate0_recomputes_tradeability_from_quote_evidence(
    db_session,
    mutate_row,
    report_kwargs,
    expected_spread_ok,
    expected_size_ok,
    expected_entry_integrity_conflicts,
):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    row = _add_gate0_trade_row(
        db_session,
        ticker="AAA",
        day=TRADING_DATE,
        asof=asof,
    )
    mutate_row(row)

    report = i12_gate0_report(
        db_session,
        asof=asof,
        min_context_count=0,
        min_intended_count=0,
        min_snapshot_ok_rate=0.0,
        max_snapshot_error_or_missing_rate=1.0,
        min_score_model_ok_rate=0.0,
        min_quote_ok_rate=0.0,
        min_exit_quote_ok_rate=0.0,
        min_gate0_intended_count=0,
        min_gate0_distinct_trading_days=0,
        min_gate0_tradeable_rate=1.0,
        **report_kwargs,
    )

    assert row.skipped_reason == "none"
    assert report["intended_count"] == 1
    assert report["spread_ok"] == expected_spread_ok
    assert report["size_ok"] == expected_size_ok
    assert report["spread_ok_rate"] == pytest.approx(float(expected_spread_ok))
    assert report["size_ok_rate"] == pytest.approx(float(expected_size_ok))
    assert report["tradeable"] == 0
    assert report["tradeable_rate"] == pytest.approx(0.0)
    assert report["evidence_conflict_count"] == 1
    assert report["entry_integrity_conflict_count"] == expected_entry_integrity_conflicts
    if expected_entry_integrity_conflicts:
        assert "entry_integrity_conflict" in report["non_promotable_reasons"]
    else:
        assert "entry_integrity_conflict" not in report["non_promotable_reasons"]
    assert report["exit_quote_tradeable_denominator"] == 0
    assert "min_gate0_tradeable_rate" in report["coverage_gate_failures"]
    assert report["passed"] is False


def test_i12_stage0_raw_spread_mismatch_uses_observed_spread(db_session):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    row = _add_gate0_trade_row(
        db_session,
        ticker="AAA",
        day=TRADING_DATE,
        asof=asof,
    )
    row.bid = 10.0
    row.ask = 11.0
    row.spread_bps = 20.0
    row.quote_json = _quote_json("AAA", bid=10.0, ask=11.0)

    report = i12_gate0_report(
        db_session,
        asof=asof,
        max_spread_bps=200.0,
        min_context_count=0,
        min_intended_count=0,
        min_snapshot_ok_rate=0.0,
        max_snapshot_error_or_missing_rate=1.0,
        min_score_model_ok_rate=0.0,
        min_quote_ok_rate=0.0,
        min_exit_quote_ok_rate=0.0,
        min_gate0_intended_count=0,
        min_gate0_distinct_trading_days=0,
        min_gate0_tradeable_rate=0.0,
    )

    assert report["entry_integrity_conflict_count"] == 1
    assert report["spread_ok"] == 0
    assert report["tradeable"] == 0
    assert report["tradeable_rate"] == pytest.approx(0.0)
    assert report["exit_quote_tradeable_denominator"] == 0
    assert "entry_integrity_conflict" in report["non_promotable_reasons"]
    assert report["passed"] is False


def test_i12_stage0_wide_quote_matching_spread_is_threshold_miss_only(
    db_session,
):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    row = _add_gate0_trade_row(
        db_session,
        ticker="AAA",
        day=TRADING_DATE,
        asof=asof,
    )
    row.bid = 10.0
    row.ask = 11.0
    row.spread_bps = _spread_bps(10.0, 11.0)
    row.quote_json = _quote_json("AAA", bid=10.0, ask=11.0)
    row.top_of_book_size = 1_100.0

    report = i12_gate0_report(
        db_session,
        asof=asof,
        max_spread_bps=200.0,
        min_context_count=0,
        min_intended_count=0,
        min_snapshot_ok_rate=0.0,
        max_snapshot_error_or_missing_rate=1.0,
        min_score_model_ok_rate=0.0,
        min_quote_ok_rate=0.0,
        min_exit_quote_ok_rate=0.0,
        min_gate0_intended_count=0,
        min_gate0_distinct_trading_days=0,
        min_gate0_tradeable_rate=1.0,
    )

    assert report["entry_integrity_conflict_count"] == 0
    assert report["spread_ok"] == 0
    assert report["tradeable"] == 0
    assert "entry_integrity_conflict" not in report["non_promotable_reasons"]
    assert "min_gate0_tradeable_rate" in report["coverage_gate_failures"]
    assert report["passed"] is False


def test_i12_stage0_quote_liquidity_treats_ask_size_as_current_shares():
    quote = _quote("AAA", ask=10.05, ask_size=1)

    liquidity = evaluate_quote_liquidity(
        quote,
        intended_order_usd=250.0,
        max_spread_bps=200.0,
        asof=DECISION_TS,
        max_quote_age_seconds=60.0,
    )

    assert ALPACA_QUOTE_SIZE_BASIS == "shares_post_2025_11_03"
    assert liquidity["top_of_book_size"] == pytest.approx(10.05)
    assert liquidity["size_sufficient"] is False
    assert liquidity["skipped_reason"] == "size"


def test_i12_stage0_quote_liquidity_passes_current_share_size():
    quote = _quote("AAA", ask=10.05, ask_size=25)

    liquidity = evaluate_quote_liquidity(
        quote,
        intended_order_usd=250.0,
        max_spread_bps=200.0,
        asof=DECISION_TS,
        max_quote_age_seconds=60.0,
    )

    assert liquidity["top_of_book_size"] == pytest.approx(251.25)
    assert liquidity["size_sufficient"] is True
    assert liquidity["skipped_reason"] == "none"


def test_i12_stage0_quote_liquidity_zero_ask_shares_fails_size():
    quote = _quote("AAA", ask=10.05, ask_size=0)

    liquidity = evaluate_quote_liquidity(
        quote,
        intended_order_usd=250.0,
        max_spread_bps=200.0,
        asof=DECISION_TS,
        max_quote_age_seconds=60.0,
    )

    assert liquidity["top_of_book_size"] == pytest.approx(0.0)
    assert liquidity["size_sufficient"] is False
    assert liquidity["skipped_reason"] == "size"


def test_i12_stage0_raw_size_integrity_uses_logged_order_size(db_session):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    row = _add_gate0_trade_row(
        db_session,
        ticker="AAA",
        day=TRADING_DATE,
        asof=asof,
    )
    row.intended_order_usd = 250.0
    row.top_of_book_size = 200.0
    row.size_sufficient = True

    report = i12_gate0_report(
        db_session,
        asof=asof,
        intended_order_usd=100.0,
        min_context_count=0,
        min_intended_count=0,
        min_snapshot_ok_rate=0.0,
        max_snapshot_error_or_missing_rate=1.0,
        min_score_model_ok_rate=0.0,
        min_quote_ok_rate=0.0,
        min_exit_quote_ok_rate=0.0,
        min_gate0_intended_count=0,
        min_gate0_distinct_trading_days=0,
        min_gate0_tradeable_rate=0.0,
    )

    assert report["size_ok"] == 1
    assert report["tradeable"] == 0
    assert report["tradeable_rate"] == pytest.approx(0.0)
    assert report["exit_quote_tradeable_denominator"] == 0
    assert report["entry_integrity_conflict_count"] == 1
    assert "entry_integrity_conflict" in report["non_promotable_reasons"]
    assert report["passed"] is False


def test_i12_stage0_report_size_threshold_miss_is_not_raw_integrity_conflict(
    db_session,
):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    row = _add_gate0_trade_row(
        db_session,
        ticker="AAA",
        day=TRADING_DATE,
        asof=asof,
    )
    row.intended_order_usd = 250.0
    row.top_of_book_size = 1_005.0
    row.size_sufficient = True

    report = i12_gate0_report(
        db_session,
        asof=asof,
        intended_order_usd=2_000.0,
        min_context_count=0,
        min_intended_count=0,
        min_snapshot_ok_rate=0.0,
        max_snapshot_error_or_missing_rate=1.0,
        min_score_model_ok_rate=0.0,
        min_quote_ok_rate=0.0,
        min_exit_quote_ok_rate=0.0,
        min_gate0_intended_count=0,
        min_gate0_distinct_trading_days=0,
        min_gate0_tradeable_rate=1.0,
    )

    assert report["size_ok"] == 0
    assert report["tradeable"] == 0
    assert report["tradeable_rate"] == pytest.approx(0.0)
    assert report["entry_integrity_conflict_count"] == 0
    assert "entry_integrity_conflict" not in report["non_promotable_reasons"]
    assert "min_gate0_tradeable_rate" in report["coverage_gate_failures"]
    assert report["passed"] is False


def test_i12_stage0_quote_json_thin_book_inflated_size_is_integrity_conflict(
    db_session,
):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    row = _add_gate0_trade_row(
        db_session,
        ticker="AAA",
        day=TRADING_DATE,
        asof=asof,
    )
    row.quote_json = _quote_json("AAA", ask_size=1)
    row.top_of_book_size = 1_000.0
    row.size_sufficient = True

    report = i12_gate0_report(
        db_session,
        asof=asof,
        min_context_count=0,
        min_intended_count=0,
        min_snapshot_ok_rate=0.0,
        max_snapshot_error_or_missing_rate=1.0,
        min_score_model_ok_rate=0.0,
        min_quote_ok_rate=0.0,
        min_exit_quote_ok_rate=0.0,
        min_gate0_intended_count=0,
        min_gate0_distinct_trading_days=0,
        min_gate0_tradeable_rate=0.0,
    )

    assert report["size_ok"] == 0
    assert report["tradeable"] == 0
    assert report["exit_quote_tradeable_denominator"] == 0
    assert report["entry_integrity_conflict_count"] == 1
    assert "entry_integrity_conflict" in report["non_promotable_reasons"]
    assert report["passed"] is False


def test_i12_stage0_old_round_lot_size_interpretation_is_integrity_conflict(
    db_session,
):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    row = _add_gate0_trade_row(
        db_session,
        ticker="AAA",
        day=TRADING_DATE,
        asof=asof,
    )
    row.quote_json = _quote_json("AAA", ask_size=1)
    row.top_of_book_size = 1_005.0
    row.size_sufficient = True

    report = i12_gate0_report(
        db_session,
        asof=asof,
        min_context_count=0,
        min_intended_count=0,
        min_snapshot_ok_rate=0.0,
        max_snapshot_error_or_missing_rate=1.0,
        min_score_model_ok_rate=0.0,
        min_quote_ok_rate=0.0,
        min_exit_quote_ok_rate=0.0,
        min_gate0_intended_count=0,
        min_gate0_distinct_trading_days=0,
        min_gate0_tradeable_rate=0.0,
    )

    assert report["size_ok"] == 0
    assert report["tradeable"] == 0
    assert report["entry_integrity_conflict_count"] == 1
    assert "entry_integrity_conflict" in report["non_promotable_reasons"]


def test_i12_stage0_quote_json_clean_size_passes(db_session):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    _add_gate0_trade_row(
        db_session,
        ticker="AAA",
        day=TRADING_DATE,
        asof=asof,
    )

    report = i12_gate0_report(
        db_session,
        asof=asof,
        min_context_count=0,
        min_intended_count=0,
        min_snapshot_ok_rate=0.0,
        max_snapshot_error_or_missing_rate=1.0,
        min_score_model_ok_rate=0.0,
        min_quote_ok_rate=0.0,
        min_exit_quote_ok_rate=0.0,
        min_gate0_intended_count=0,
        min_gate0_distinct_trading_days=0,
        min_gate0_tradeable_rate=1.0,
    )

    assert report["size_ok"] == 1
    assert report["tradeable"] == 1
    assert report["exit_quote_tradeable_denominator"] == 1
    assert report["entry_integrity_conflict_count"] == 0
    assert report["quote_size_basis"] == ALPACA_QUOTE_SIZE_BASIS
    assert report["quote_size_bases"] == [ALPACA_QUOTE_SIZE_BASIS]
    assert report["missing_quote_size_basis_count"] == 0
    assert report["unsupported_quote_size_basis_count"] == 0
    assert report["passed"] is True


def test_i12_stage0_missing_quote_size_basis_is_non_promotable(db_session):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    row = _add_gate0_trade_row(
        db_session,
        ticker="AAA",
        day=TRADING_DATE,
        asof=asof,
    )
    payload = json.loads(row.quote_json)
    payload.pop("quote_size_basis")
    row.quote_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    report = i12_gate0_report(
        db_session,
        asof=asof,
        min_context_count=0,
        min_intended_count=0,
        min_snapshot_ok_rate=0.0,
        max_snapshot_error_or_missing_rate=1.0,
        min_score_model_ok_rate=0.0,
        min_quote_ok_rate=0.0,
        min_exit_quote_ok_rate=0.0,
        min_gate0_intended_count=0,
        min_gate0_distinct_trading_days=0,
        min_gate0_tradeable_rate=0.0,
    )

    assert report["quote_size_basis"] is None
    assert report["quote_size_bases"] == []
    assert report["missing_quote_size_basis_count"] == 1
    assert report["unsupported_quote_size_basis_count"] == 0
    assert "missing_quote_size_basis" in report["non_promotable_reasons"]
    assert report["passed"] is False


def test_i12_stage0_missing_quote_size_basis_skipped_cash_is_non_promotable(
    db_session,
):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    row = _add_gate0_trade_row(
        db_session,
        ticker="AAA",
        day=TRADING_DATE,
        asof=asof,
    )
    _mark_gate0_row_skipped_cash(row, reason="size")
    payload = json.loads(row.quote_json)
    payload.pop("quote_size_basis")
    row.quote_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    report = i12_gate0_report(
        db_session,
        asof=asof,
        min_context_count=0,
        min_intended_count=0,
        min_snapshot_ok_rate=0.0,
        max_snapshot_error_or_missing_rate=1.0,
        min_score_model_ok_rate=0.0,
        min_quote_ok_rate=0.0,
        min_exit_quote_ok_rate=0.0,
        min_gate0_intended_count=0,
        min_gate0_distinct_trading_days=0,
        min_gate0_tradeable_rate=0.0,
    )

    assert report["missing_quote_size_basis_count"] == 1
    assert report["entry_integrity_conflict_count"] == 0
    assert "missing_quote_size_basis" in report["non_promotable_reasons"]
    assert report["passed"] is False


def test_i12_stage0_unknown_quote_size_basis_is_non_promotable(db_session):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    row = _add_gate0_trade_row(
        db_session,
        ticker="AAA",
        day=TRADING_DATE,
        asof=asof,
    )
    payload = json.loads(row.quote_json)
    payload["quote_size_basis"] = "round_lots_pre_2025_11_03"
    row.quote_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    report = i12_gate0_report(
        db_session,
        asof=asof,
        min_context_count=0,
        min_intended_count=0,
        min_snapshot_ok_rate=0.0,
        max_snapshot_error_or_missing_rate=1.0,
        min_score_model_ok_rate=0.0,
        min_quote_ok_rate=0.0,
        min_exit_quote_ok_rate=0.0,
        min_gate0_intended_count=0,
        min_gate0_distinct_trading_days=0,
        min_gate0_tradeable_rate=0.0,
    )

    assert report["quote_size_basis"] == "round_lots_pre_2025_11_03"
    assert report["quote_size_bases"] == ["round_lots_pre_2025_11_03"]
    assert report["missing_quote_size_basis_count"] == 0
    assert report["unsupported_quote_size_basis_count"] == 1
    assert "unsupported_quote_size_basis" in report["non_promotable_reasons"]
    assert report["passed"] is False


def test_i12_stage0_mixed_quote_size_basis_is_non_promotable(db_session):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    _add_gate0_trade_row(
        db_session,
        ticker="AAA",
        day=TRADING_DATE,
        asof=asof,
    )
    row = _add_gate0_trade_row(
        db_session,
        ticker="BBB",
        day=TRADING_DATE,
        asof=asof,
    )
    payload = json.loads(row.quote_json)
    payload["quote_size_basis"] = "shares_post_future_change"
    row.quote_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    report = i12_gate0_report(
        db_session,
        asof=asof,
        min_context_count=0,
        min_intended_count=0,
        min_snapshot_ok_rate=0.0,
        max_snapshot_error_or_missing_rate=1.0,
        min_score_model_ok_rate=0.0,
        min_quote_ok_rate=0.0,
        min_exit_quote_ok_rate=0.0,
        min_gate0_intended_count=0,
        min_gate0_distinct_trading_days=0,
        min_gate0_tradeable_rate=0.0,
    )

    assert report["quote_size_basis"] is None
    assert report["quote_size_bases"] == [
        ALPACA_QUOTE_SIZE_BASIS,
        "shares_post_future_change",
    ]
    assert report["unsupported_quote_size_basis_count"] == 1
    assert "mixed_quote_size_basis" in report["non_promotable_reasons"]
    assert "unsupported_quote_size_basis" in report["non_promotable_reasons"]
    assert report["passed"] is False


def test_i12_stage0_quote_json_thin_book_is_size_miss_not_integrity_conflict(
    db_session,
):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    row = _add_gate0_trade_row(
        db_session,
        ticker="AAA",
        day=TRADING_DATE,
        asof=asof,
    )
    _mark_gate0_row_skipped_cash(row, reason="size")
    row.quote_json = _quote_json("AAA", ask_size=1)
    row.top_of_book_size = 10.05
    row.size_sufficient = False

    report = i12_gate0_report(
        db_session,
        asof=asof,
        min_context_count=0,
        min_intended_count=0,
        min_snapshot_ok_rate=0.0,
        max_snapshot_error_or_missing_rate=1.0,
        min_score_model_ok_rate=0.0,
        min_quote_ok_rate=0.0,
        min_exit_quote_ok_rate=0.0,
        min_gate0_intended_count=0,
        min_gate0_distinct_trading_days=0,
        min_gate0_tradeable_rate=0.0,
    )

    assert report["size_ok"] == 0
    assert report["tradeable"] == 0
    assert report["exit_quote_tradeable_denominator"] == 0
    assert report["entry_integrity_conflict_count"] == 0
    assert "entry_integrity_conflict" not in report["non_promotable_reasons"]


def test_i12_stage0_none_skip_with_raw_insufficient_size_is_integrity_conflict(
    db_session,
):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    row = _add_gate0_trade_row(
        db_session,
        ticker="AAA",
        day=TRADING_DATE,
        asof=asof,
    )
    row.quote_json = _quote_json("AAA", ask_size=1)
    row.top_of_book_size = 10.05
    row.size_sufficient = False
    row.skipped_reason = "none"

    report = i12_gate0_report(
        db_session,
        asof=asof,
        min_context_count=0,
        min_intended_count=0,
        min_snapshot_ok_rate=0.0,
        max_snapshot_error_or_missing_rate=1.0,
        min_score_model_ok_rate=0.0,
        min_quote_ok_rate=0.0,
        min_exit_quote_ok_rate=0.0,
        min_gate0_intended_count=0,
        min_gate0_distinct_trading_days=0,
        min_gate0_tradeable_rate=0.0,
    )

    assert report["size_ok"] == 0
    assert report["tradeable"] == 0
    assert report["entry_integrity_conflict_count"] == 1
    assert "entry_integrity_conflict" in report["non_promotable_reasons"]


def test_i12_stage0_entry_integrity_conflict_blocks_promotion_above_tradeable_rate(
    db_session,
):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    days = [
        TRADING_DATE,
        TRADING_DATE + timedelta(days=1),
        TRADING_DATE + timedelta(days=2),
    ]
    for idx in range(20):
        row = _add_gate0_trade_row(
            db_session,
            ticker=f"T{idx:03d}",
            day=days[idx % len(days)],
            asof=asof,
        )
        if idx == 15:
            row.quote_status = "stale"
        elif idx > 15:
            _mark_gate0_row_skipped_cash(row, reason="size")

    report = i12_gate0_report(db_session, asof=asof)

    assert report["intended_count"] == 20
    assert report["tradeable"] == 15
    assert report["tradeable_rate"] == pytest.approx(0.75)
    assert report["quote_ok_rate"] == pytest.approx(0.95)
    assert report["coverage_gate_failures"] == []
    assert report["entry_integrity_conflict_count"] == 1
    assert "entry_integrity_conflict" in report["non_promotable_reasons"]
    assert report["passed"] is False


def test_i12_stage0_gate0_passes_just_above_tradeable_rate_threshold(db_session):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    days = [
        TRADING_DATE,
        TRADING_DATE + timedelta(days=1),
        TRADING_DATE + timedelta(days=2),
    ]
    for idx in range(21):
        row = _add_gate0_trade_row(
            db_session,
            ticker=f"T{idx:03d}",
            day=days[idx % len(days)],
            asof=asof,
        )
        if idx >= 15:
            _mark_gate0_row_skipped_cash(row, reason="size")

    report = i12_gate0_report(db_session, asof=asof)

    assert report["intended_count"] == 21
    assert report["tradeable"] == 15
    assert report["tradeable_rate"] == pytest.approx(15 / 21)
    assert report["spread_ok"] == 21
    assert report["size_ok"] == 15
    assert report["evidence_conflict_count"] == 0
    assert report["entry_integrity_conflict_count"] == 0
    assert "min_gate0_tradeable_rate" not in report["coverage_gate_failures"]
    assert report["coverage_gate_failures"] == []
    assert report["passed"] is True


def test_i12_stage0_gate0_rejects_mixed_policy_hashes_across_days(db_session):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    days = [
        TRADING_DATE,
        TRADING_DATE + timedelta(days=1),
        TRADING_DATE + timedelta(days=2),
    ]
    for idx in range(20):
        day = days[idx % len(days)]
        _add_gate0_trade_row(
            db_session,
            ticker=f"T{idx:03d}",
            day=day,
            asof=asof,
            config_hash=f"policy-{idx % 2}",
            context_hash=f"context-{day.isoformat()}",
        )

    report = i12_gate0_report(db_session, asof=asof)

    assert report["intended_count"] == 20
    assert report["intended_distinct_trading_days"] == 3
    assert report["stage0_run_config_hash"] is None
    assert report["stage0_run_config_hashes"] == ["policy-0", "policy-1"]
    assert report["context_artifact_hashes"] == [
        f"context-{day.isoformat()}" for day in days
    ]
    assert "mixed_or_missing_stage0_run_config" in report["non_promotable_reasons"]
    assert report["coverage_gate_failures"] == []
    assert report["passed"] is False


def test_i12_stage0_exit_capture_ignores_skipped_cash_rows(db_session, monkeypatch):
    model = _add_model(db_session)
    snapshots = {
        "AAA": _snapshot("AAA", quote=_quote("AAA", bid=10.00, ask=10.05)),
        "BBB": _snapshot("BBB", quote=_quote("BBB", bid=10.00, ask=10.35)),
    }
    _patch_scores(monkeypatch, {"AAA": 0.90, "BBB": 0.80})
    job = I12LiveFillTestJob(
        session=db_session,
        alpaca_adapter=FakeAlpaca(snapshots=snapshots),
        contexts={ticker: _context(ticker) for ticker in snapshots},
        config=I12LiveFillConfig(
            model_id=model.model_id,
            top_k=2,
            require_market_open=False,
        ),
        snapshots=snapshots,
        asof=DECISION_TS,
    )
    run_job(db_session, job, params={"test": True})

    exit_asof = datetime(2026, 6, 17, 13, 31, tzinfo=timezone.utc)
    result = capture_i12_exit_quotes(
        db_session,
        FakeAlpaca(
            quotes={
                "AAA": _quote("AAA", bid=10.50, ask=10.55, timestamp=exit_asof),
                "BBB": _quote("BBB", bid=20.00, ask=20.10, timestamp=exit_asof),
            }
        ),
        asof=exit_asof,
    )
    rows = {row.ticker: row for row in db_session.query(I12FillLog).all()}
    report = i12_gate0_report(
        db_session,
        decision_date=TRADING_DATE,
        asof=exit_asof,
    )

    assert result["exit_quote_updates"] == 1
    assert rows["AAA"].exit_capture_status == "ok"
    assert rows["BBB"].exit_capture_status == "skipped_cash"
    assert rows["BBB"].exit_bid is None
    assert rows["BBB"].modeled_return == pytest.approx(0.0)
    assert report["exit_quote_tradeable_denominator"] == 1
    assert report["exit_quote_skipped_cash_count"] == 1


def test_i12_stage0_gate0_fails_past_due_uncaptured_exit(db_session, monkeypatch):
    model = _add_model(db_session)
    snapshots = {"AAA": _snapshot("AAA", quote=_quote("AAA"))}
    _patch_scores(monkeypatch, {"AAA": 0.90})
    job = I12LiveFillTestJob(
        session=db_session,
        alpaca_adapter=FakeAlpaca(snapshots=snapshots),
        contexts={"AAA": _context("AAA")},
        config=I12LiveFillConfig(model_id=model.model_id, require_market_open=False),
        snapshots=snapshots,
        asof=DECISION_TS,
    )
    run_job(db_session, job, params={"test": True})

    exit_asof = datetime(2026, 6, 17, 13, 31, tzinfo=timezone.utc)
    report = i12_gate0_report(
        db_session,
        decision_date=TRADING_DATE,
        asof=exit_asof,
    )

    assert report["exit_quote_pending_due"] == 1
    assert report["exit_quote_coverage_count"] == 1
    assert report["exit_quote_ok_rate"] == pytest.approx(0.0)
    assert "min_exit_quote_ok_rate" in report["coverage_gate_failures"]
    assert report["passed"] is False


def test_i12_stage0_runner_requires_non_public_scratch_schema():
    with pytest.raises(SchemaTargetError, match="requires --schema"):
        require_stage0_scratch_schema(None)
    with pytest.raises(SchemaTargetError, match="refuses public"):
        require_stage0_scratch_schema("public")
    assert require_stage0_scratch_schema("scratch_i12_stage0") == "scratch_i12_stage0"


def test_i12_stage0_logs_snapshot_missing_and_batch_errors(db_session, monkeypatch):
    model = _add_model(db_session)
    _patch_scores(monkeypatch, {})
    contexts = {"AAA": _context("AAA"), "BBB": _context("BBB")}
    job = I12LiveFillTestJob(
        session=db_session,
        alpaca_adapter=FakeAlpaca(snapshot_error=True),
        contexts=contexts,
        config=I12LiveFillConfig(model_id=model.model_id, require_market_open=False),
        asof=DECISION_TS,
    )

    result = run_job(db_session, job, params={"test": True})
    report = i12_gate0_report(
        db_session,
        decision_date=TRADING_DATE,
        asof=DECISION_TS,
    )

    assert result.status == "finished"
    assert report["context_count"] == 2
    assert report["snapshot_batch_errors"] == 2
    assert report["fire_count"] == 0
    assert {row.snapshot_status for row in db_session.query(I12FillLog).all()} == {
        "batch_error"
    }


def test_i12_stage0_stale_snapshot_and_quote_fail_closed(db_session, monkeypatch):
    model = _add_model(db_session)
    stale_ts = DECISION_TS - timedelta(minutes=10)
    fresh_snapshot_stale_quote = _snapshot(
        "AAA",
        quote=_quote("AAA", timestamp=stale_ts),
    )
    stale_snapshot = _snapshot(
        "BBB",
        quote=_quote("BBB"),
        minute_timestamp=stale_ts,
        latest_trade_timestamp=DECISION_TS,
    )
    snapshots = {"AAA": fresh_snapshot_stale_quote, "BBB": stale_snapshot}
    _patch_scores(monkeypatch, {"AAA": 0.90, "BBB": 0.80})
    job = I12LiveFillTestJob(
        session=db_session,
        alpaca_adapter=FakeAlpaca(snapshots=snapshots),
        contexts={"AAA": _context("AAA"), "BBB": _context("BBB")},
        config=I12LiveFillConfig(
            model_id=model.model_id,
            require_market_open=False,
            max_quote_age_seconds=60.0,
            max_snapshot_age_seconds=60.0,
        ),
        snapshots=snapshots,
        asof=DECISION_TS,
    )

    result = run_job(db_session, job, params={"test": True})
    rows = {row.ticker: row for row in db_session.query(I12FillLog).all()}
    report = i12_gate0_report(
        db_session,
        decision_date=TRADING_DATE,
        asof=DECISION_TS,
    )

    assert result.status == "finished"
    assert rows["AAA"].snapshot_status == "ok"
    assert rows["AAA"].quote_status == "stale"
    assert rows["AAA"].skipped_reason == "quote_stale"
    assert rows["BBB"].snapshot_status == "stale_minute_data"
    assert rows["BBB"].latest_trade_age_seconds == pytest.approx(0.0)
    assert rows["BBB"].minute_age_seconds == pytest.approx(600.0)
    assert rows["BBB"].selection_status == "not_selected"
    assert report["snapshot_stale_minute_data"] == 1
    assert report["quote_stale"] == 1
    assert report["skipped_as_cash_count"] == 1


def test_i12_stage0_gate0_requires_coverage_thresholds(db_session, monkeypatch):
    model = _add_model(db_session)
    contexts = {"AAA": _context("AAA"), "BBB": _context("BBB")}
    snapshots = {"AAA": _snapshot("AAA", quote=_quote("AAA"))}
    _patch_scores(monkeypatch, {"AAA": 0.90})
    job = I12LiveFillTestJob(
        session=db_session,
        alpaca_adapter=FakeAlpaca(snapshots=snapshots),
        contexts=contexts,
        config=I12LiveFillConfig(model_id=model.model_id, require_market_open=False),
        snapshots=snapshots,
        asof=DECISION_TS,
    )

    run_job(db_session, job, params={"test": True})
    report = i12_gate0_report(db_session, decision_date=TRADING_DATE)

    assert report["snapshot_ok_rate"] == pytest.approx(0.5)
    assert report["snapshot_error_or_missing_rate"] == pytest.approx(0.5)
    assert report["coverage_gate_passed"] is False
    assert "min_snapshot_ok_rate" in report["coverage_gate_failures"]
    assert "max_snapshot_error_or_missing_rate" in report["coverage_gate_failures"]
    assert report["passed"] is False


def test_i12_stage0_mixed_feed_configs_are_non_promotable(db_session, monkeypatch):
    model = _add_model(db_session)
    snapshots = {"AAA": _snapshot("AAA", quote=_quote("AAA"))}
    _patch_scores(monkeypatch, {"AAA": 0.90})
    for feed in ("sip", "iex"):
        job = I12LiveFillTestJob(
            session=db_session,
            alpaca_adapter=FakeAlpaca(snapshots=snapshots),
            contexts={"AAA": _context("AAA")},
            config=I12LiveFillConfig(
                model_id=model.model_id,
                feed=feed,
                require_market_open=False,
            ),
            snapshots=snapshots,
            asof=DECISION_TS,
        )
        run_job(db_session, job, params={"test": True})

    report = i12_gate0_report(db_session, decision_date=TRADING_DATE)
    assert db_session.query(I12FillLog).count() == 2
    assert len(report["stage0_run_config_hashes"]) == 2
    assert "mixed_or_missing_stage0_run_config" in report["non_promotable_reasons"]
    assert "diagnostic_feed" in report["non_promotable_reasons"]
    assert report["passed"] is False


def test_i12_stage0_top_k_config_change_does_not_false_pass(db_session, monkeypatch):
    model = _add_model(db_session)
    snapshots = {
        "AAA": _snapshot("AAA", quote=_quote("AAA")),
        "BBB": _snapshot("BBB", quote=_quote("BBB")),
    }
    _patch_scores(monkeypatch, {"AAA": 0.90, "BBB": 0.80})
    for top_k in (2, 1):
        job = I12LiveFillTestJob(
            session=db_session,
            alpaca_adapter=FakeAlpaca(snapshots=snapshots),
            contexts={ticker: _context(ticker) for ticker in snapshots},
            config=I12LiveFillConfig(
                model_id=model.model_id,
                top_k=top_k,
                require_market_open=False,
            ),
            snapshots=snapshots,
            asof=DECISION_TS,
        )
        run_job(db_session, job, params={"test": True})

    report = i12_gate0_report(db_session, decision_date=TRADING_DATE)
    assert db_session.query(I12FillLog).count() == 4
    assert len(report["stage0_run_config_hashes"]) == 2
    assert "mixed_or_missing_stage0_run_config" in report["non_promotable_reasons"]
    assert report["passed"] is False


def test_i12_stage0_half_day_is_diagnostic_not_promotable(db_session, monkeypatch):
    model = _add_model(db_session)
    half_day = date(2026, 11, 27)
    asof = datetime(2026, 11, 27, 16, 30, tzinfo=timezone.utc)
    snapshots = {"AAA": _snapshot("AAA", quote=_quote("AAA", timestamp=asof), timestamp=asof)}
    _patch_scores(monkeypatch, {"AAA": 0.90})
    job = I12LiveFillTestJob(
        session=db_session,
        alpaca_adapter=FakeAlpaca(snapshots=snapshots),
        contexts={"AAA": _context("AAA", context_date=half_day)},
        config=I12LiveFillConfig(model_id=model.model_id, require_market_open=False),
        snapshots=snapshots,
        asof=asof,
    )

    run_job(db_session, job, params={"test": True})
    report = i12_gate0_report(db_session, decision_date=half_day)

    assert report["half_day"] is True
    assert report["session_minutes"] == 210
    assert report["projection_basis"] == "half_day_diagnostic_390m_projected_volume"
    assert "half_day_diagnostic" in report["non_promotable_reasons"]
    assert report["passed"] is False


def test_i12_stage0_iex_feed_is_diagnostic_not_promotable(db_session, monkeypatch):
    model = _add_model(db_session)
    snapshots = {"AAA": _snapshot("AAA", quote=_quote("AAA"))}
    _patch_scores(monkeypatch, {"AAA": 0.90})
    job = I12LiveFillTestJob(
        session=db_session,
        alpaca_adapter=FakeAlpaca(snapshots=snapshots),
        contexts={"AAA": _context("AAA")},
        config=I12LiveFillConfig(
            model_id=model.model_id,
            feed="iex",
            require_market_open=False,
        ),
        snapshots=snapshots,
        asof=DECISION_TS,
    )

    run_job(db_session, job, params={"test": True})
    report = i12_gate0_report(db_session, decision_date=TRADING_DATE)

    assert report["tradeable"] == 1
    assert report["promotable"] is False
    assert report["passed"] is False
    assert "diagnostic_feed" in report["non_promotable_reasons"]


def test_i12_stage0_model_contract_rejects_schema_or_horizon_mismatch(db_session):
    model = _add_model(db_session)
    assert validate_i12_stage0_model_contract(model) == EXPECTED_I12_LIVE_FEATURES

    bad_schema = _add_model(db_session, model_id="bad-schema")
    schema = {"fields": [{"name": "gap", "source": "feature_snapshot.gap"}]}
    bad_schema.feature_schema_json = json.dumps(schema, sort_keys=True)
    bad_schema.feature_schema_hash = feature_schema_hash(schema)
    with pytest.raises(RuntimeError, match="feature list"):
        validate_i12_stage0_model_contract(bad_schema)

    bad_horizon = _add_model(db_session, model_id="bad-horizon")
    bad_horizon.training_params_json = json.dumps({"horizon_sessions": 2})
    with pytest.raises(RuntimeError, match="one-session"):
        validate_i12_stage0_model_contract(bad_horizon)


def test_i12_stage0_model_contract_requires_promotable_status(db_session):
    bad_status = _add_model(
        db_session,
        model_id="bad-status",
        status="research",
    )

    with pytest.raises(RuntimeError, match="promotable allowlist"):
        validate_i12_stage0_model_contract(bad_status)


def test_i12_stage0_artifact_preflight_validates_identity_and_scores(
    db_session,
    tmp_path,
):
    good_path = tmp_path / "good.pkl"
    model = _add_model(db_session, artifact_uri=str(good_path))
    _write_stage0_artifact(good_path, model)

    validate_i12_stage0_artifact_preflight(model, session=db_session)

    mismatch_path = tmp_path / "mismatch.pkl"
    mismatch = _add_model(
        db_session,
        model_id="mismatch-model",
        artifact_uri=str(mismatch_path),
    )
    _write_stage0_artifact(
        mismatch_path,
        mismatch,
        model_id="different-model",
    )
    with pytest.raises(RuntimeError, match="artifact identity mismatch"):
        validate_i12_stage0_artifact_preflight(mismatch, session=db_session)

    unscorable_path = tmp_path / "unscorable.pkl"
    unscorable = _add_model(
        db_session,
        model_id="unscorable-model",
        artifact_uri=str(unscorable_path),
    )
    _write_stage0_artifact(unscorable_path, unscorable, model=object())
    with pytest.raises(RuntimeError, match="does not contain a scoring model"):
        validate_i12_stage0_artifact_preflight(unscorable, session=db_session)

    bad_schema_path = tmp_path / "bad-schema.pkl"
    bad_schema = _add_model(
        db_session,
        model_id="bad-artifact-schema",
        artifact_uri=str(bad_schema_path),
    )
    schema = _i12_feature_schema()
    schema["fields"][0]["source"] = "feature_snapshot.mom20"
    bad_schema.feature_schema_json = json.dumps(schema, sort_keys=True)
    bad_schema.feature_schema_hash = feature_schema_hash(schema)
    _write_stage0_artifact(bad_schema_path, bad_schema)
    with pytest.raises(RuntimeError, match="feature source"):
        validate_i12_stage0_artifact_preflight(bad_schema, session=db_session)

    bad_ranges_path = tmp_path / "bad-ranges.pkl"
    bad_ranges = _add_model(
        db_session,
        model_id="bad-ranges-model",
        artifact_uri=str(bad_ranges_path),
    )
    _write_stage0_artifact(bad_ranges_path, bad_ranges, training_feature_ranges=[])
    with pytest.raises(RuntimeError, match="training_feature_ranges"):
        validate_i12_stage0_artifact_preflight(bad_ranges, session=db_session)


def test_i12_stage0_artifact_preflight_uses_in_range_smoke_vector(
    db_session,
    tmp_path,
):
    path = tmp_path / "range-smoke.pkl"
    model = _add_model(db_session, model_id="range-smoke", artifact_uri=str(path))
    ranges = [{"min": -1.0, "max": 1.0} for _ in EXPECTED_I12_LIVE_FEATURES]
    projected_idx = EXPECTED_I12_LIVE_FEATURES.index(
        "projected_volume_ratio_at_confirmation"
    )
    ranges[projected_idx] = {"min": 5.0, "max": 20.0}
    _write_stage0_artifact(
        path,
        model,
        training_feature_ranges=ranges,
        model=RangeCheckingPredictModel(
            "projected_volume_ratio_at_confirmation",
            5.0,
            20.0,
        ),
    )

    validate_i12_stage0_artifact_preflight(model, session=db_session)
    assert db_session.query(FeatureSnapshot).count() == 0
    assert db_session.query(SignalRegistry).count() == 0
    assert db_session.query(SignalMLScore).count() == 0


def test_i12_stage0_model_copy_smoke_scores_only_in_scratch_session(
    db_session,
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "scratch-smoke.pkl"
    source = _add_model(db_session, model_id="scratch-smoke", artifact_uri=str(path))
    _write_stage0_artifact(path, source)
    scratch_engine, scratch_session = _new_memory_session()

    class CanonicalSession:
        def __getattr__(self, name):
            return getattr(db_session, name)

        def close(self):
            pass

    monkeypatch.setattr(
        run_i12_live_fill_test,
        "_open_canonical_session",
        lambda url: CanonicalSession(),
    )

    try:
        copied_model_id = ensure_model_registry_row_in_scratch(
            database_url="sqlite:///:memory:",
            scratch_session=scratch_session,
            model_id=source.model_id,
            allow_latest_model=False,
            feed="sip",
        )

        assert copied_model_id == source.model_id
        assert scratch_session.get(MLModelRegistry, source.model_id) is not None
        assert db_session.query(FeatureSnapshot).count() == 0
        assert db_session.query(SignalRegistry).count() == 0
        assert db_session.query(SignalMLScore).count() == 0
        assert scratch_session.query(FeatureSnapshot).count() == 0
        assert scratch_session.query(SignalRegistry).count() == 0
        assert scratch_session.query(SignalMLScore).count() == 0
    finally:
        scratch_session.close()
        scratch_engine.dispose()


def test_i12_stage0_exit_quote_missing_and_stale_are_reported(db_session, monkeypatch):
    model = _add_model(db_session)
    snapshots = {"AAA": _snapshot("AAA", quote=_quote("AAA"))}
    _patch_scores(monkeypatch, {"AAA": 0.90})
    job = I12LiveFillTestJob(
        session=db_session,
        alpaca_adapter=FakeAlpaca(snapshots=snapshots),
        contexts={"AAA": _context("AAA")},
        config=I12LiveFillConfig(model_id=model.model_id, require_market_open=False),
        snapshots=snapshots,
        asof=DECISION_TS,
    )
    run_job(db_session, job, params={"test": True})
    exit_asof = datetime(2026, 6, 17, 13, 31, tzinfo=timezone.utc)

    missing = capture_i12_exit_quotes(
        db_session,
        FakeAlpaca(quotes={}),
        asof=exit_asof,
    )
    row = db_session.query(I12FillLog).one()
    assert missing["exit_quote_missing"] == 1
    assert row.exit_capture_status == "missing"

    row.exit_capture_status = "not_due"
    stale = capture_i12_exit_quotes(
        db_session,
        FakeAlpaca(quotes={"AAA": _quote("AAA", timestamp=DECISION_TS)}),
        asof=exit_asof,
    )
    report = i12_gate0_report(
        db_session,
        decision_date=TRADING_DATE,
        asof=exit_asof,
    )
    assert stale["exit_quote_stale"] == 1
    assert report["exit_quote_stale"] == 1


def test_i12_stage0_required_tables_include_scratch_model_registry():
    assert "ml_model_registry" in I12_FILL_TEST_REQUIRED_TABLES


def test_i12_stage0_create_tables_preflights_empty_scratch(monkeypatch, capsys):
    captured = {}

    class DummyQuery:
        def filter(self, *args, **kwargs):
            return self

        def all(self):
            return []

    class DummySession:
        def query(self, *args, **kwargs):
            return DummyQuery()

        def close(self):
            pass

    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    monkeypatch.setattr(run_i12_live_fill_test, "load_runtime_env", lambda: None)
    monkeypatch.setattr(
        run_i12_live_fill_test,
        "prepare_writable_schema_target",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        run_i12_live_fill_test,
        "open_writable_session",
        lambda **kwargs: DummySession(),
    )

    try:
        rc = run_i12_live_fill_test.main([
            "--schema",
            "scratch_stage0",
            "--create-tables",
            "--gate0-report",
        ])
    finally:
        os.environ.pop("ALPHA_DB_SCHEMA", None)

    assert rc == 0
    assert captured["schema"] == "scratch_stage0"
    assert captured["create_tables"] is True
    assert "ml_model_registry" in captured["required_tables"]
    assert json.loads(capsys.readouterr().out)["rows"] == 0


def test_i12_stage0_exit_quotes_cli_passes_asof(monkeypatch, capsys):
    captured = {}

    class DummySession:
        def close(self):
            pass

    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    monkeypatch.setattr(run_i12_live_fill_test, "load_runtime_env", lambda: None)
    monkeypatch.setattr(run_i12_live_fill_test, "prepare_writable_schema_target", lambda **kwargs: None)
    monkeypatch.setattr(run_i12_live_fill_test, "open_writable_session", lambda **kwargs: DummySession())

    def fake_capture(session, alpaca, **kwargs):
        del session, alpaca
        captured.update(kwargs)
        return {"exit_quote_updates": 0}

    monkeypatch.setattr(run_i12_live_fill_test, "capture_i12_exit_quotes", fake_capture)

    try:
        rc = run_i12_live_fill_test.main([
            "--schema",
            "scratch_stage0",
            "--exit-quotes",
            "--asof",
            "2026-06-17T13:31:00Z",
        ])
    finally:
        os.environ.pop("ALPHA_DB_SCHEMA", None)

    assert rc == 0
    assert captured["asof"] == datetime(2026, 6, 17, 13, 31, tzinfo=timezone.utc)
    assert json.loads(capsys.readouterr().out)["exit_quote_updates"] == 0


def test_i12_stage0_gate0_report_cli_defaults_asof_fail_closed(
    db_session,
    monkeypatch,
    capsys,
):
    fixed_now = datetime(2026, 6, 17, 13, 31, tzinfo=timezone.utc)
    db_session.add(
        I12FillLog(
            model_id=None,
            ticker="AAA",
            decision_date=TRADING_DATE,
            decision_ts=DECISION_TS,
            exit_capture_due_ts=fixed_now - timedelta(minutes=1),
            feed="sip",
            model_selection_mode="explicit",
            promotable_run=True,
            stage0_run_config_hash="same-config",
            half_day=False,
            session_minutes=390,
            projection_basis="regular_session_390m_projected_volume",
            attempt_stage="intended",
            snapshot_status="ok",
            fire_status="fired",
            score_stage0_status="model_ok",
            selection_status="intended",
            quote_status="ok",
            exit_capture_status="not_due",
            intended_order_usd=250.0,
            size_sufficient=True,
            skipped_reason="none",
            bid=10.0,
            ask=10.05,
            spread_bps=_spread_bps(10.0, 10.05),
            top_of_book_size=1_005.0,
            quote_json=_quote_json("AAA"),
            feature_json="{}",
            gate_values_json="{}",
            content_hash="gate0-default-asof",
        )
    )
    db_session.commit()

    class DummySession:
        def __getattr__(self, name):
            return getattr(db_session, name)

        def close(self):
            pass

    import alpha.jobs.i12_live_fill_test as live_module

    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    monkeypatch.setattr(live_module, "utcnow", lambda: fixed_now)
    monkeypatch.setattr(run_i12_live_fill_test, "load_runtime_env", lambda: None)
    monkeypatch.setattr(run_i12_live_fill_test, "prepare_writable_schema_target", lambda **kwargs: None)
    monkeypatch.setattr(run_i12_live_fill_test, "open_writable_session", lambda **kwargs: DummySession())

    try:
        rc = run_i12_live_fill_test.main([
            "--schema",
            "scratch_stage0",
            "--gate0-report",
            "--trading-date",
            TRADING_DATE.isoformat(),
        ])
    finally:
        os.environ.pop("ALPHA_DB_SCHEMA", None)

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["exit_quote_pending_due"] == 1
    assert payload["exit_quote_ok_rate"] == pytest.approx(0.0)
    assert "min_exit_quote_ok_rate" in payload["coverage_gate_failures"]
    assert payload["passed"] is False
    assert payload["asof"] == fixed_now.isoformat()


def test_i12_stage0_gate0_report_cli_fail_flag_returns_2(
    db_session,
    monkeypatch,
    capsys,
):
    fixed_now = datetime(2026, 6, 17, 13, 31, tzinfo=timezone.utc)
    row = _add_gate0_trade_row(
        db_session,
        ticker="AAA",
        day=TRADING_DATE,
        asof=fixed_now,
    )
    row.exit_capture_status = "not_due"
    row.exit_bid = None
    row.exit_ask = None
    row.modeled_return = None
    db_session.commit()

    class DummySession:
        def __getattr__(self, name):
            return getattr(db_session, name)

        def close(self):
            pass

    import alpha.jobs.i12_live_fill_test as live_module

    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    monkeypatch.setattr(live_module, "utcnow", lambda: fixed_now)
    monkeypatch.setattr(run_i12_live_fill_test, "load_runtime_env", lambda: None)
    monkeypatch.setattr(run_i12_live_fill_test, "prepare_writable_schema_target", lambda **kwargs: None)
    monkeypatch.setattr(run_i12_live_fill_test, "open_writable_session", lambda **kwargs: DummySession())

    try:
        rc = run_i12_live_fill_test.main([
            "--schema",
            "scratch_stage0",
            "--gate0-report",
            "--trading-date",
            TRADING_DATE.isoformat(),
            "--fail-on-gate0-fail",
        ])
    finally:
        os.environ.pop("ALPHA_DB_SCHEMA", None)

    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["passed"] is False
    assert payload["exit_quote_pending_due"] == 1


def test_i12_stage0_gate0_report_cli_rejects_invalid_tradeable_rate(capsys):
    with pytest.raises(SystemExit) as exc:
        run_i12_live_fill_test.main([
            "--gate0-report",
            "--min-gate0-tradeable-rate",
            "1.1",
        ])

    assert exc.value.code == 2
    assert "--min-gate0-tradeable-rate must be between 0 and 1" in capsys.readouterr().err


def test_i12_stage0_gate0_report_cli_fail_flag_tradeable_rate_only(
    db_session,
    monkeypatch,
    capsys,
):
    asof = datetime(2026, 6, 19, 13, 31, tzinfo=timezone.utc)
    days = [
        TRADING_DATE,
        TRADING_DATE + timedelta(days=1),
        TRADING_DATE + timedelta(days=2),
    ]
    for idx in range(20):
        row = _add_gate0_trade_row(
            db_session,
            ticker=f"T{idx:03d}",
            day=days[idx % len(days)],
            asof=asof,
        )
        if idx >= 10:
            _mark_gate0_row_skipped_cash(row, reason="size")
    db_session.commit()

    class DummySession:
        def __getattr__(self, name):
            return getattr(db_session, name)

        def close(self):
            pass

    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    monkeypatch.setattr(run_i12_live_fill_test, "load_runtime_env", lambda: None)
    monkeypatch.setattr(run_i12_live_fill_test, "prepare_writable_schema_target", lambda **kwargs: None)
    monkeypatch.setattr(run_i12_live_fill_test, "open_writable_session", lambda **kwargs: DummySession())

    try:
        rc = run_i12_live_fill_test.main([
            "--schema",
            "scratch_stage0",
            "--gate0-report",
            "--asof",
            asof.isoformat(),
            "--fail-on-gate0-fail",
        ])
    finally:
        os.environ.pop("ALPHA_DB_SCHEMA", None)

    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["tradeable_rate"] == pytest.approx(0.5)
    assert payload["coverage_gate_failures"] == ["min_gate0_tradeable_rate"]
    assert payload["passed"] is False


def test_i12_stage0_monitor_exits_when_context_date_has_elapsed(monkeypatch, capsys):
    class DummySession:
        def close(self):
            pass

    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    monkeypatch.setattr(run_i12_live_fill_test, "load_runtime_env", lambda: None)
    monkeypatch.setattr(run_i12_live_fill_test, "prepare_writable_schema_target", lambda **kwargs: None)
    monkeypatch.setattr(run_i12_live_fill_test, "open_writable_session", lambda **kwargs: DummySession())
    monkeypatch.setattr(
        run_i12_live_fill_test,
        "load_premarket_context_artifact",
        lambda *args, **kwargs: {"AAA": _context("AAA")},
    )
    monkeypatch.setattr(
        run_i12_live_fill_test,
        "_context_artifact_hash",
        lambda path: "context-hash",
    )
    monkeypatch.setattr(
        run_i12_live_fill_test,
        "ensure_model_registry_row_in_scratch",
        lambda **kwargs: "i12-live-test-model",
    )

    def fail_run_job(*args, **kwargs):
        raise AssertionError("stale context should exit before running the job")

    monkeypatch.setattr(run_i12_live_fill_test, "run_job", fail_run_job)

    try:
        rc = run_i12_live_fill_test.main([
            "--schema",
            "scratch_stage0",
            "--context-artifact",
            "context.json",
            "--trading-date",
            TRADING_DATE.isoformat(),
            "--model-id",
            "i12-live-test-model",
            "--asof",
            "2026-06-17T13:31:00Z",
        ])
    finally:
        os.environ.pop("ALPHA_DB_SCHEMA", None)

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["reason"] == "trading_date_elapsed"
    assert payload["configured_trading_date"] == TRADING_DATE.isoformat()


def test_i12_stage0_scratch_model_copy_drops_canonical_job_run_fk(db_session):
    source = _add_model(db_session)
    target = MLModelRegistry(
        model_id=source.model_id,
        pattern_id="I12",
        model_family="old",
        manifest_version="old",
        manifest_sha256="old",
        feature_schema_hash="old",
        cv_metrics_json="{}",
        feature_schema_json=json.dumps(_i12_feature_schema(), sort_keys=True),
        artifact_uri="old",
        status="shadow",
        job_run_id="canonical-run",
    )
    _copy_model_registry_values(source, target)

    assert target.job_run_id is None
    assert target.model_id == source.model_id
    assert target.manifest_sha256 == source.manifest_sha256


def _patch_scores(monkeypatch, scores, *, source="model_shadow"):
    import alpha.jobs.i12_live_fill_test as module

    def fake_score(session, *, signal_id, model_id=None, score_status="shadow"):
        signal = session.get(module.SignalRegistry, signal_id)
        assert signal is not None
        existing = (
            session.query(SignalMLScore)
            .filter(
                SignalMLScore.signal_id == signal_id,
                SignalMLScore.model_id == model_id,
                SignalMLScore.score_status == score_status,
            )
            .one_or_none()
        )
        if existing is not None:
            return existing
        score = float(scores[signal.ticker])
        row = SignalMLScore(
            score_id=f"score-{signal.ticker}-{score_status}",
            signal_id=signal_id,
            model_id=model_id,
            pattern_id=signal.pattern_id,
            ticker=signal.ticker,
            score=score,
            fallback_score=None if source == "model_shadow" else score,
            score_source=source,
            fallback_reason=None if source == "model_shadow" else "test_fallback",
            score_status=score_status,
            score_metadata_json=json.dumps({"acts_on_book": source == "model_shadow"}),
            scored_at=DECISION_TS,
        )
        session.add(row)
        session.flush()
        return row

    monkeypatch.setattr(module, "score_signal_shadow", fake_score)


def _context(ticker, *, context_date=TRADING_DATE):
    return PremarketContext(
        ticker=ticker,
        context_date=context_date,
        prior_close=10.0,
        max_prior_252_closes=25.0,
        avg20_volume=100_000.0,
        mom20=0.15,
        off_low252=0.20,
        sigma20=0.08,
        prev_day_return=-0.04,
        prev_day_green=False,
        spy_prior_day_return=0.003,
    )


def _snapshot(
    ticker,
    *,
    quote=None,
    timestamp=DECISION_TS,
    minute_timestamp=None,
    latest_trade_timestamp=None,
):
    minute_timestamp = minute_timestamp or timestamp
    latest_trade_timestamp = latest_trade_timestamp or timestamp
    return AlpacaStockSnapshot(
        symbol=ticker,
        daily_open=10.10,
        daily_high=10.20,
        daily_low=9.95,
        daily_volume=13_000.0,
        minute_open=10.08,
        minute_high=10.12,
        minute_low=10.01,
        minute_close=10.08,
        minute_volume=2_000.0,
        minute_timestamp=minute_timestamp.isoformat().replace("+00:00", "Z"),
        latest_trade_price=10.08,
        latest_trade_timestamp=latest_trade_timestamp.isoformat().replace("+00:00", "Z"),
        latest_quote=quote or _quote(ticker),
        raw={},
    )


def _quote(
    ticker,
    *,
    bid=10.00,
    ask=10.05,
    ask_size=100,
    conditions=None,
    timestamp=DECISION_TS,
):
    return AlpacaQuote(
        symbol=ticker,
        bid_price=bid,
        ask_price=ask,
        bid_size=100,
        ask_size=ask_size,
        timestamp=timestamp.isoformat().replace("+00:00", "Z"),
        conditions=conditions or [],
    )


def _add_model(
    db_session,
    *,
    model_id="i12-live-test-model",
    artifact_uri="/tmp/not-used-by-monkeypatch.pkl",
    status="shadow",
):
    schema = _i12_feature_schema()
    row = MLModelRegistry(
        model_id=model_id,
        pattern_id="I12",
        model_family="hist_gradient_boosting",
        training_window_start=date(2025, 1, 1),
        training_window_end=date(2026, 6, 15),
        manifest_version="i12_test",
        manifest_sha256="manifest-sha",
        feature_schema_hash=feature_schema_hash(schema),
        training_params_json=json.dumps({
            "horizon_sessions": 1,
            "signal_horizon": "1d",
        }),
        cv_metrics_json=json.dumps({
            "top_decile_lift": 2.0,
            "rank_ic": 0.1,
            "horizon_sessions": 1,
            "signal_horizon": "1d",
        }),
        feature_schema_json=json.dumps(schema, sort_keys=True),
        artifact_uri=artifact_uri,
        status=status,
    )
    db_session.add(row)
    db_session.flush()
    return row


def _add_gate0_trade_row(
    db_session,
    *,
    ticker,
    day,
    asof,
    config_hash="gate0-config",
    context_hash="context-hash",
):
    row = I12FillLog(
        model_id=None,
        ticker=ticker,
        decision_date=day,
        decision_ts=DECISION_TS + timedelta(days=(day - TRADING_DATE).days),
        exit_capture_due_ts=asof - timedelta(minutes=1),
        feed="sip",
        model_selection_mode="explicit",
        promotable_run=True,
        stage0_run_config_hash=config_hash,
        context_artifact_hash=context_hash,
        half_day=False,
        session_minutes=390,
        projection_basis="regular_session_390m_projected_volume",
        attempt_stage="intended",
        snapshot_status="ok",
        fire_status="fired",
        score_stage0_status="model_ok",
        selection_status="intended",
        quote_status="ok",
        exit_capture_status="ok",
        intended_order_usd=250.0,
        size_sufficient=True,
        skipped_reason="none",
        bid=10.0,
        ask=10.05,
        spread_bps=_spread_bps(10.0, 10.05),
        top_of_book_size=1_005.0,
        quote_json=_quote_json(ticker),
        exit_bid=10.20,
        exit_ask=10.25,
        exit_quote_ts=asof,
        exit_quote_age_seconds=0.0,
        modeled_return=(10.20 / 10.05) - 1.0,
        feature_json="{}",
        gate_values_json="{}",
        content_hash=f"gate0-{ticker}-{day.isoformat()}",
    )
    db_session.add(row)
    db_session.flush()
    return row


def _spread_bps(bid, ask):
    return ((ask - bid) / ((ask + bid) / 2.0)) * 10_000.0


def _quote_json(ticker, *, bid=10.0, ask=10.05, ask_size=100):
    return json.dumps(
        {
            "symbol": ticker,
            "bid_price": bid,
            "ask_price": ask,
            "bid_size": 100,
            "ask_size": ask_size,
            "quote_size_basis": ALPACA_QUOTE_SIZE_BASIS,
            "timestamp": DECISION_TS.isoformat().replace("+00:00", "Z"),
            "conditions": [],
            "tape": None,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _mark_gate0_row_skipped_cash(row, *, reason):
    row.skipped_reason = reason
    row.size_sufficient = False
    row.top_of_book_size = 0.0
    row.exit_capture_status = "skipped_cash"
    row.exit_bid = None
    row.exit_ask = None
    row.exit_quote_ts = None
    row.exit_quote_age_seconds = None
    row.modeled_return = 0.0
    return row


def _add_gate0_context_row(
    db_session,
    *,
    ticker,
    day,
    config_hash="gate0-config",
    context_hash="context-hash",
):
    row = I12FillLog(
        model_id=None,
        ticker=ticker,
        decision_date=day,
        decision_ts=DECISION_TS + timedelta(days=(day - TRADING_DATE).days),
        feed="sip",
        model_selection_mode="explicit",
        promotable_run=True,
        stage0_run_config_hash=config_hash,
        context_artifact_hash=context_hash,
        half_day=False,
        session_minutes=390,
        projection_basis="regular_session_390m_projected_volume",
        attempt_stage="context",
        snapshot_status="ok",
        fire_status="not_fired",
        score_stage0_status="not_evaluated",
        selection_status="not_selected",
        quote_status="not_requested",
        exit_capture_status="not_due",
        intended_order_usd=250.0,
        skipped_reason="not_selected",
        feature_json="{}",
        gate_values_json="{}",
        content_hash=f"gate0-context-{ticker}-{day.isoformat()}",
    )
    db_session.add(row)
    db_session.flush()
    return row


def _i12_feature_schema():
    return {
        "pattern_id": "I12",
        "pattern_clock": "intraday",
        "fields": [
            {
                "name": name,
                "source": "feature_snapshot_json",
                "path": name,
                "role": "feature",
                "dtype": "float",
            }
            for name in EXPECTED_I12_LIVE_FEATURES
        ]
    }


def _write_stage0_artifact(
    path,
    model_row,
    *,
    model_id=None,
    model=None,
    training_feature_ranges=None,
):
    schema = json.loads(model_row.feature_schema_json)
    payload = {
        "model_id": model_id or model_row.model_id,
        "pattern_id": model_row.pattern_id,
        "feature_schema": schema,
        "feature_schema_hash": model_row.feature_schema_hash,
        "manifest_sha256": model_row.manifest_sha256,
        "manifest_version": model_row.manifest_version,
        "model_family": model_row.model_family,
        "feature_names": EXPECTED_I12_LIVE_FEATURES,
        "training_feature_ranges": (
            training_feature_ranges
            if training_feature_ranges is not None
            else [{"min": -1.0, "max": 1.0} for _ in EXPECTED_I12_LIVE_FEATURES]
        ),
        "model": model or DummyPredictModel(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _lineage(endpoint):
    return LineageMeta(
        provider="Alpaca",
        endpoint=endpoint,
        request_timestamp=DECISION_TS,
        asof_timestamp=DECISION_TS,
        raw_payload_hash="test",
    )


def _stored_test_utc(value):
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _new_memory_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session()
