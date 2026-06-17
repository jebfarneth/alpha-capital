import json
from datetime import date, datetime, timezone

import pytest

from alpha.data.alpaca import AlpacaClock, AlpacaQuote, AlpacaStockSnapshot
from alpha.data.contracts import AdapterResponse, LineageMeta
from alpha.db.models import I12FillLog, MLModelRegistry, SignalMLScore
from alpha.jobs.i12_live_fill_test import (
    I12LiveFillConfig,
    I12LiveFillTestJob,
    assert_i12_live_feature_payload_leakage_clean,
    build_i12_live_feature_payload,
    capture_i12_exit_quotes,
    i12_gate0_report,
)
from alpha.jobs.paper_execution import PremarketContext
from alpha.jobs.runner import run_job
from alpha.jobs.run_i12_live_fill_test import require_stage0_scratch_schema
from alpha.db.engine import SchemaTargetError


TRADING_DATE = date(2026, 6, 16)
DECISION_TS = datetime(2026, 6, 16, 13, 40, tzinfo=timezone.utc)


class FakeAlpaca:
    def __init__(self, *, snapshots=None, quotes=None, clock_open=True):
        self.snapshots = snapshots or {}
        self.quotes = quotes or {}
        self.clock_open = clock_open

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
    assert result.metrics["fires"] == 3
    assert result.metrics["selected_top_k"] == 2
    assert result.metrics["liquidity_skips"] == 1
    rows = {row.ticker: row for row in db_session.query(I12FillLog).all()}
    assert set(rows) == {"AAA", "BBB"}
    assert rows["AAA"].skipped_reason == "none"
    assert rows["BBB"].skipped_reason == "spread"
    assert rows["AAA"].intended_order_usd == pytest.approx(250.0)
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
    assert result.metrics["fires"] == 1
    assert result.metrics["model_scored_fires"] == 0
    assert result.metrics["logged_intended_trades"] == 0
    assert db_session.query(I12FillLog).count() == 0


def test_i12_stage0_fill_log_is_idempotent(db_session, monkeypatch):
    model = _add_model(db_session)
    snapshots = {"AAA": _snapshot("AAA", quote=_quote("AAA"))}
    _patch_scores(monkeypatch, {"AAA": 0.90})
    config = I12LiveFillConfig(model_id=model.model_id, require_market_open=False)

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

    exit_quote = _quote("AAA", bid=10.50, ask=10.55)
    result = capture_i12_exit_quotes(
        db_session,
        FakeAlpaca(quotes={"AAA": exit_quote}),
        asof=datetime(2026, 6, 17, 13, 31, tzinfo=timezone.utc),
    )
    report = i12_gate0_report(db_session, decision_date=TRADING_DATE)

    assert result == {"exit_quote_updates": 1}
    row = db_session.query(I12FillLog).one()
    assert row.exit_bid == pytest.approx(10.50)
    assert row.modeled_return == pytest.approx((10.50 / 10.05) - 1.0)
    assert report["rows"] == 1
    assert report["tradeable"] == 1
    assert report["passed"] is True


def test_i12_stage0_runner_requires_non_public_scratch_schema():
    with pytest.raises(SchemaTargetError, match="requires --schema"):
        require_stage0_scratch_schema(None)
    with pytest.raises(SchemaTargetError, match="refuses public"):
        require_stage0_scratch_schema("public")
    assert require_stage0_scratch_schema("scratch_i12_stage0") == "scratch_i12_stage0"


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


def _context(ticker):
    return PremarketContext(
        ticker=ticker,
        context_date=TRADING_DATE,
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


def _snapshot(ticker, *, quote=None):
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
        minute_timestamp=DECISION_TS.isoformat().replace("+00:00", "Z"),
        latest_trade_price=10.08,
        latest_trade_timestamp=DECISION_TS.isoformat().replace("+00:00", "Z"),
        latest_quote=quote or _quote(ticker),
        raw={},
    )


def _quote(ticker, *, bid=10.00, ask=10.05, ask_size=100, conditions=None):
    return AlpacaQuote(
        symbol=ticker,
        bid_price=bid,
        ask_price=ask,
        bid_size=100,
        ask_size=ask_size,
        timestamp=DECISION_TS.isoformat().replace("+00:00", "Z"),
        conditions=conditions or [],
    )


def _add_model(db_session):
    row = MLModelRegistry(
        model_id="i12-live-test-model",
        pattern_id="I12",
        model_family="hist_gradient_boosting",
        training_window_start=date(2025, 1, 1),
        training_window_end=date(2026, 6, 15),
        manifest_version="i12_test",
        manifest_sha256="manifest-sha",
        feature_schema_hash="schema-sha",
        cv_metrics_json=json.dumps({"top_decile_lift": 2.0, "rank_ic": 0.1}),
        feature_schema_json=json.dumps({"fields": []}),
        artifact_uri="/tmp/not-used-by-monkeypatch.pkl",
        status="shadow",
    )
    db_session.add(row)
    db_session.flush()
    return row


def _lineage(endpoint):
    return LineageMeta(
        provider="Alpaca",
        endpoint=endpoint,
        request_timestamp=DECISION_TS,
        asof_timestamp=DECISION_TS,
        raw_payload_hash="test",
    )
