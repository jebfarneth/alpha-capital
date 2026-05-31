"""Forward-context panel collector tests."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest

from alpha.assembly.signal_context import SOURCE_CONTEXT_VERSION
from alpha.data.config import ConfigError
from alpha.data.contracts import stable_hash
from alpha.db.models import ForwardContextPathRow
from alpha.evidence.writer import record_data_lineage, record_feature_snapshot, record_signal
from alpha.jobs import run_forward_context
from alpha.jobs.forward_context import (
    FORWARD_CONTEXT_VERSION,
    ForwardContextCollectorJob,
    forward_context_rows_through,
)
from alpha.jobs.run_m4_canonical import _forward_context_panel_capture_ok
from alpha.jobs.runner import run_job
from alpha.market_calendar import us_equity_session_close_timestamp

from tests.test_m4_daily_job import (
    FakeBenzingaContextAdapter,
    FakePolygonContextAdapter,
    _adapter_response,
)


def _ts(day: int = 1) -> datetime:
    return datetime(2026, 6, day, 20, 0, tzinfo=timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _add_m4_signal(
    db_session,
    *,
    ticker: str = "LCUT",
    identity_hash: str = "signal-1",
    trading_date: str = "2026-06-01",
    next_execution_session: str = "2026-06-02",
):
    lineage = record_data_lineage(
        db_session,
        provider="FMP",
        endpoint="/test/m4",
        asof_timestamp=_ts(),
        raw_payload={"ticker": ticker, "identity_hash": identity_hash},
    )
    feature = record_feature_snapshot(
        db_session,
        pattern_id="M4",
        ticker=ticker,
        asof_timestamp=_ts(),
        features={
            "price": 11.0,
            "high_52w": 10.0,
            "detector_signal_identity_hash": identity_hash,
        },
        data_lineage_ids=[lineage.data_lineage_id],
        point_in_time_passed=True,
        lookahead_guard_passed=True,
    )
    return record_signal(
        db_session,
        pattern_id="M4",
        ticker=ticker,
        direction="long",
        signal_timestamp=_ts(),
        raw_signal_strength=1.1,
        raw_expected_edge=0.05,
        feature_snapshot_id=feature.feature_snapshot_id,
        signal_horizon="15d",
        trading_date=trading_date,
        next_execution_session=next_execution_session,
        point_in_time_passed=True,
        lookahead_guard_passed=True,
        signal_identity_hash=identity_hash,
    )


class QuietPolygonContextAdapter:
    def __init__(self):
        self.calls = []

    def _empty(self, method_name: str, endpoint: str, kwargs):
        self.calls.append((method_name, kwargs))
        return _adapter_response(
            provider="Polygon",
            endpoint=endpoint,
            asof=kwargs["asof"],
            data=[],
        )

    def get_short_interest(self, **kwargs):
        return self._empty("get_short_interest", "/stocks/v1/short-interest", kwargs)

    def get_short_volume(self, **kwargs):
        return self._empty("get_short_volume", "/stocks/v1/short-volume", kwargs)

    def get_splits(self, **kwargs):
        return self._empty("get_splits", "/stocks/v1/splits", kwargs)

    def get_dividends(self, **kwargs):
        return self._empty("get_dividends", "/stocks/v1/dividends", kwargs)

    def get_news(self, **kwargs):
        return self._empty("get_news", "/v2/reference/news", kwargs)


class QuietBenzingaContextAdapter:
    def __init__(self):
        self.calls = []

    def _empty(self, method_name: str, endpoint: str, kwargs):
        self.calls.append((method_name, kwargs))
        return _adapter_response(
            provider="Benzinga",
            endpoint=endpoint,
            asof=kwargs["asof"],
            data=[],
        )

    def get_news(self, **kwargs):
        return self._empty("get_news", "/api/v2/news", kwargs)

    def get_wiims(self, **kwargs):
        return self._empty("get_wiims", "/api/v2/news", kwargs)

    def get_earnings(self, **kwargs):
        return self._empty("get_earnings", "/api/v2.1/calendar/earnings", kwargs)

    def get_guidance(self, **kwargs):
        return self._empty("get_guidance", "/api/v2.1/calendar/guidance", kwargs)

    def get_ratings(self, **kwargs):
        return self._empty("get_ratings", "/api/v2.1/calendar/ratings", kwargs)

    def get_offerings(self, **kwargs):
        return self._empty("get_offerings", "/api/v2.1/calendar/offerings", kwargs)

    def get_dividends(self, **kwargs):
        return self._empty("get_dividends", "/api/v2.1/calendar/dividends", kwargs)

    def get_insider_filings(self, **kwargs):
        return self._empty(
            "get_insider_filings",
            "/api/v1/sec/insider_transactions/filings",
            kwargs,
        )

    def get_insider_transactions(self, **kwargs):
        return self._empty(
            "get_insider_transactions",
            "/api/v1/sec/insider_transactions/transactions",
            kwargs,
        )

    def get_mergers_acquisitions(self, **kwargs):
        return self._empty("get_mergers_acquisitions", "/api/v2.1/calendar/ma", kwargs)


class DeadPolygonContextAdapter:
    def _raise(self, **_kwargs):
        raise RuntimeError("polygon key revoked")

    get_short_interest = _raise
    get_short_volume = _raise
    get_splits = _raise
    get_dividends = _raise
    get_news = _raise


class DeadBenzingaContextAdapter:
    def _raise(self, **_kwargs):
        raise RuntimeError("benzinga key revoked")

    get_news = _raise
    get_wiims = _raise
    get_earnings = _raise
    get_guidance = _raise
    get_ratings = _raise
    get_offerings = _raise
    get_dividends = _raise
    get_insider_filings = _raise
    get_insider_transactions = _raise
    get_mergers_acquisitions = _raise


class SelectiveDeadPolygonContextAdapter(QuietPolygonContextAdapter):
    def __init__(self, dead_ticker: str):
        super().__init__()
        self.dead_ticker = dead_ticker

    def _empty(self, method_name: str, endpoint: str, kwargs):
        if kwargs.get("ticker") == self.dead_ticker:
            raise RuntimeError("polygon key revoked for ticker")
        return super()._empty(method_name, endpoint, kwargs)


def test_forward_context_collector_writes_rows_and_dedupes_provider_pulls(db_session):
    sig1 = _add_m4_signal(db_session, identity_hash="signal-a")
    sig2 = _add_m4_signal(db_session, identity_hash="signal-b")
    polygon = FakePolygonContextAdapter()
    benzinga = FakeBenzingaContextAdapter()
    job = ForwardContextCollectorJob(
        db_session,
        polygon_adapter=polygon,
        benzinga_adapter=benzinga,
        run_timestamp=datetime(2026, 6, 2, 21, 15, tzinfo=timezone.utc),
    )

    result = run_job(db_session, job, params={})

    assert result.ok
    assert result.metrics["forward_session_date"] == "2026-06-02"
    assert result.metrics["eligible_signal_count"] == 2
    assert result.metrics["pending_signal_count"] == 2
    assert result.metrics["rows_inserted"] == 2
    assert result.metrics["ticker_fetch_count"] == 1
    assert Counter(name for name, _kwargs in polygon.calls) == {
        "get_short_interest": 1,
        "get_short_volume": 1,
        "get_splits": 1,
        "get_dividends": 1,
        "get_news": 1,
    }
    assert all(count == 1 for count in Counter(name for name, _ in benzinga.calls).values())

    rows = (
        db_session.query(ForwardContextPathRow)
        .order_by(ForwardContextPathRow.signal_id)
        .all()
    )
    assert len(rows) == 2
    assert {row.signal_id for row in rows} == {sig1.signal_id, sig2.signal_id}
    for row in rows:
        context = json.loads(row.context_json)
        attempts = json.loads(row.source_attempts_json)
        assert context["schema_version"] == FORWARD_CONTEXT_VERSION
        assert context["source_context_version"] == SOURCE_CONTEXT_VERSION
        assert context["context_role"] == "forward_context_panel"
        assert context["signal_id"] == row.signal_id
        assert context["forward_session_date"] == "2026-06-02"
        assert context["entry_session_date"] == "2026-06-02"
        assert context["path_sequence"] == 1
        assert row.path_sequence == 1
        assert _as_utc(row.asof_timestamp) == us_equity_session_close_timestamp(
            datetime(2026, 6, 2, tzinfo=timezone.utc).date()
        )
        assert row.context_hash == stable_hash(context)
        assert attempts
        assert json.loads(row.data_lineage_ids)
        assert row.is_terminal_snapshot is False

    rerun_polygon = FakePolygonContextAdapter()
    rerun_benzinga = FakeBenzingaContextAdapter()
    rerun = run_job(
        db_session,
        ForwardContextCollectorJob(
            db_session,
            polygon_adapter=rerun_polygon,
            benzinga_adapter=rerun_benzinga,
            run_timestamp=datetime(2026, 6, 2, 21, 15, tzinfo=timezone.utc),
        ),
        params={},
    )

    assert rerun.ok
    assert rerun.metrics["rows_existing"] == 2
    assert rerun.metrics["rows_inserted"] == 0
    assert rerun.metrics["ticker_fetch_count"] == 0
    assert rerun_polygon.calls == []
    assert rerun_benzinga.calls == []
    assert db_session.query(ForwardContextPathRow).count() == 2


def test_forward_context_collector_fails_missing_required_adapter_with_pending_rows(db_session):
    _add_m4_signal(db_session)
    job = ForwardContextCollectorJob(
        db_session,
        polygon_adapter=None,
        benzinga_adapter=FakeBenzingaContextAdapter(),
        run_timestamp=datetime(2026, 6, 2, 21, 15, tzinfo=timezone.utc),
    )

    result = run_job(db_session, job, params={})

    assert not result.ok
    assert result.metrics["eligible_signal_count"] == 1
    assert result.metrics["pending_signal_count"] == 1
    assert result.metrics["rows_inserted"] == 0
    assert result.metrics["missing_required_adapters"] == ["polygon"]
    assert result.errors[0]["stage"] == "provider_adapters"
    assert db_session.query(ForwardContextPathRow).count() == 0


def test_forward_context_collector_writes_quiet_rows_when_adapters_are_present(db_session):
    _add_m4_signal(db_session)
    polygon = QuietPolygonContextAdapter()
    benzinga = QuietBenzingaContextAdapter()
    job = ForwardContextCollectorJob(
        db_session,
        polygon_adapter=polygon,
        benzinga_adapter=benzinga,
        run_timestamp=datetime(2026, 6, 2, 21, 15, tzinfo=timezone.utc),
    )

    result = run_job(db_session, job, params={})

    assert result.ok
    assert result.metrics["eligible_signal_count"] == 1
    assert result.metrics["rows_inserted"] == 1
    assert result.metrics["source_attempt_status_counts"]["no_data"] >= 1
    assert db_session.query(ForwardContextPathRow).count() == 1
    assert polygon.calls
    assert benzinga.calls


def test_forward_context_collector_fails_dead_polygon_and_allows_healthy_recapture(db_session):
    sig1 = _add_m4_signal(db_session, identity_hash="dead-polygon-a")
    sig2 = _add_m4_signal(db_session, identity_hash="dead-polygon-b")

    failed = run_job(
        db_session,
        ForwardContextCollectorJob(
            db_session,
            polygon_adapter=DeadPolygonContextAdapter(),
            benzinga_adapter=QuietBenzingaContextAdapter(),
            run_timestamp=datetime(2026, 6, 2, 21, 15, tzinfo=timezone.utc),
        ),
        params={},
    )

    assert not failed.ok
    assert failed.metrics["degraded_signal_count"] == 2
    assert failed.metrics["dead_providers"] == {"polygon": 2}
    assert failed.metrics["rows_inserted"] == 0
    assert failed.errors[0]["stage"] == "provider_quality"
    assert db_session.query(ForwardContextPathRow).count() == 0

    healthy = run_job(
        db_session,
        ForwardContextCollectorJob(
            db_session,
            polygon_adapter=FakePolygonContextAdapter(),
            benzinga_adapter=FakeBenzingaContextAdapter(),
            run_timestamp=datetime(2026, 6, 2, 21, 15, tzinfo=timezone.utc),
        ),
        params={},
    )

    assert healthy.ok
    assert healthy.metrics["rows_inserted"] == 2
    rows = db_session.query(ForwardContextPathRow).all()
    assert {row.signal_id for row in rows} == {sig1.signal_id, sig2.signal_id}


def test_forward_context_collector_quarantines_one_dead_ticker_and_writes_healthy_rows(db_session):
    healthy_a = _add_m4_signal(db_session, ticker="HLTA", identity_hash="healthy-a")
    dead = _add_m4_signal(db_session, ticker="DEAD", identity_hash="dead-one")
    healthy_b = _add_m4_signal(db_session, ticker="HLTB", identity_hash="healthy-b")

    result = run_job(
        db_session,
        ForwardContextCollectorJob(
            db_session,
            polygon_adapter=SelectiveDeadPolygonContextAdapter("DEAD"),
            benzinga_adapter=QuietBenzingaContextAdapter(),
            run_timestamp=datetime(2026, 6, 2, 21, 15, tzinfo=timezone.utc),
        ),
        params={},
    )

    assert not result.ok
    assert result.metrics["rows_inserted"] == 2
    assert result.metrics["degraded_signal_count"] == 1
    assert result.metrics["dead_providers"] == {"polygon": 1}
    assert _forward_context_panel_capture_ok(result.metrics) is False
    rows = db_session.query(ForwardContextPathRow).all()
    assert {row.signal_id for row in rows} == {
        healthy_a.signal_id,
        healthy_b.signal_id,
    }
    assert dead.signal_id not in {row.signal_id for row in rows}


def test_forward_context_collector_recaptures_quarantined_slot_on_next_run(db_session):
    healthy_a = _add_m4_signal(db_session, ticker="HLTA", identity_hash="healthy-a")
    dead = _add_m4_signal(db_session, ticker="DEAD", identity_hash="dead-one")
    healthy_b = _add_m4_signal(db_session, ticker="HLTB", identity_hash="healthy-b")

    first = run_job(
        db_session,
        ForwardContextCollectorJob(
            db_session,
            polygon_adapter=SelectiveDeadPolygonContextAdapter("DEAD"),
            benzinga_adapter=QuietBenzingaContextAdapter(),
            run_timestamp=datetime(2026, 6, 2, 21, 15, tzinfo=timezone.utc),
        ),
        params={},
    )

    assert not first.ok
    assert first.metrics["rows_inserted"] == 2

    second = run_job(
        db_session,
        ForwardContextCollectorJob(
            db_session,
            polygon_adapter=QuietPolygonContextAdapter(),
            benzinga_adapter=QuietBenzingaContextAdapter(),
            run_timestamp=datetime(2026, 6, 2, 21, 15, tzinfo=timezone.utc),
        ),
        params={},
    )

    assert second.ok
    assert second.metrics["rows_existing"] == 2
    assert second.metrics["rows_inserted"] == 1
    rows = db_session.query(ForwardContextPathRow).all()
    assert {row.signal_id for row in rows} == {
        healthy_a.signal_id,
        dead.signal_id,
        healthy_b.signal_id,
    }


def test_forward_context_collector_fails_when_both_required_providers_are_dead(db_session):
    _add_m4_signal(db_session)

    result = run_job(
        db_session,
        ForwardContextCollectorJob(
            db_session,
            polygon_adapter=DeadPolygonContextAdapter(),
            benzinga_adapter=DeadBenzingaContextAdapter(),
            run_timestamp=datetime(2026, 6, 2, 21, 15, tzinfo=timezone.utc),
        ),
        params={},
    )

    assert not result.ok
    assert result.metrics["degraded_signal_count"] == 1
    assert result.metrics["dead_providers"] == {"benzinga": 1, "polygon": 1}
    assert result.metrics["rows_inserted"] == 0
    assert db_session.query(ForwardContextPathRow).count() == 0


def test_forward_context_collector_fails_dead_polygon_even_with_quiet_benzinga(db_session):
    _add_m4_signal(db_session)

    result = run_job(
        db_session,
        ForwardContextCollectorJob(
            db_session,
            polygon_adapter=DeadPolygonContextAdapter(),
            benzinga_adapter=QuietBenzingaContextAdapter(),
            run_timestamp=datetime(2026, 6, 2, 21, 15, tzinfo=timezone.utc),
        ),
        params={},
    )

    assert not result.ok
    assert result.metrics["dead_providers"] == {"polygon": 1}
    assert result.metrics["required_provider_status_counts"]["benzinga"]["no_data"] > 0
    assert result.metrics["rows_inserted"] == 0
    assert db_session.query(ForwardContextPathRow).count() == 0


def test_forward_context_collector_writes_isolated_transient_category_error(db_session):
    _add_m4_signal(db_session)

    result = run_job(
        db_session,
        ForwardContextCollectorJob(
            db_session,
            polygon_adapter=FakePolygonContextAdapter(),
            benzinga_adapter=QuietBenzingaContextAdapter(),
            run_timestamp=datetime(2026, 6, 2, 21, 15, tzinfo=timezone.utc),
        ),
        params={},
    )

    assert result.ok
    assert result.metrics["degraded_signal_count"] == 0
    assert result.metrics["required_provider_status_counts"]["polygon"]["provider_error"] == 1
    assert result.metrics["required_provider_status_counts"]["polygon"]["matched"] >= 1
    assert result.metrics["rows_inserted"] == 1
    assert db_session.query(ForwardContextPathRow).count() == 1


def test_forward_context_collector_skips_sessions_before_entry(db_session):
    _add_m4_signal(db_session)
    polygon = FakePolygonContextAdapter()
    job = ForwardContextCollectorJob(
        db_session,
        polygon_adapter=polygon,
        run_timestamp=datetime(2026, 6, 1, 21, 15, tzinfo=timezone.utc),
    )

    result = run_job(db_session, job, params={})

    assert result.ok
    assert result.metrics["eligible_signal_count"] == 0
    assert result.metrics["rows_inserted"] == 0
    assert result.metrics["skipped_signals"] == {"before_entry_session": 1}
    assert polygon.calls == []


def test_forward_context_collector_refuses_future_forward_session(db_session):
    _add_m4_signal(db_session)
    job = ForwardContextCollectorJob(
        db_session,
        run_timestamp=datetime(2026, 6, 2, 21, 15, tzinfo=timezone.utc),
    )

    result = run_job(
        db_session,
        job,
        params={"forward_session_date": "2026-06-03"},
    )

    assert not result.ok
    assert "future" in result.errors[0]["message"]
    assert db_session.query(ForwardContextPathRow).count() == 0


def test_forward_context_collector_refuses_stale_forward_session(db_session):
    _add_m4_signal(db_session)
    job = ForwardContextCollectorJob(
        db_session,
        polygon_adapter=FakePolygonContextAdapter(),
        benzinga_adapter=FakeBenzingaContextAdapter(),
        run_timestamp=datetime(2026, 6, 5, 21, 15, tzinfo=timezone.utc),
    )

    result = run_job(
        db_session,
        job,
        params={"forward_session_date": "2026-06-02"},
    )

    assert not result.ok
    assert "does not match" in result.errors[0]["message"]
    assert db_session.query(ForwardContextPathRow).count() == 0


def test_forward_context_default_weekend_run_is_clean_noop(db_session):
    _add_m4_signal(db_session)
    job = ForwardContextCollectorJob(
        db_session,
        run_timestamp=datetime(2026, 6, 6, 16, 0, tzinfo=timezone.utc),
    )

    result = run_job(db_session, job, params={})

    assert result.ok
    assert result.metrics["no_op_reason"] == "no_current_completed_session"
    assert result.metrics["rows_inserted"] == 0
    assert db_session.query(ForwardContextPathRow).count() == 0


def test_forward_context_reader_never_returns_rows_after_decision(db_session):
    sig = _add_m4_signal(db_session)
    db_session.add_all([
        ForwardContextPathRow(
            signal_id=sig.signal_id,
            pattern_id="M4",
            ticker="LCUT",
            signal_horizon="15d",
            forward_session_date="2026-06-02",
            path_sequence=1,
            asof_timestamp=us_equity_session_close_timestamp(
                datetime(2026, 6, 2, tzinfo=timezone.utc).date()
            ),
            context_json=json.dumps({"session": "2026-06-02"}),
            source_attempts_json="[]",
            data_lineage_ids="[]",
            context_hash=stable_hash({"session": "2026-06-02"}),
            is_terminal_snapshot=False,
        ),
        ForwardContextPathRow(
            signal_id=sig.signal_id,
            pattern_id="M4",
            ticker="LCUT",
            signal_horizon="15d",
            forward_session_date="2026-06-03",
            path_sequence=2,
            asof_timestamp=us_equity_session_close_timestamp(
                datetime(2026, 6, 3, tzinfo=timezone.utc).date()
            ),
            context_json=json.dumps({"session": "2026-06-03"}),
            source_attempts_json="[]",
            data_lineage_ids="[]",
            context_hash=stable_hash({"session": "2026-06-03"}),
            is_terminal_snapshot=False,
        ),
    ])
    db_session.commit()

    rows = forward_context_rows_through(
        db_session,
        signal_id=sig.signal_id,
        decision_session_date="2026-06-02",
    )

    assert [row.forward_session_date for row in rows] == ["2026-06-02"]
    with pytest.raises(ValueError, match="regular U.S. equity session"):
        forward_context_rows_through(
            db_session,
            signal_id=sig.signal_id,
            decision_session_date="2026-06-06",
        )


def test_forward_context_reader_excludes_rows_after_decision_close(db_session):
    sig = _add_m4_signal(db_session)
    close = us_equity_session_close_timestamp(
        datetime(2026, 6, 2, tzinfo=timezone.utc).date()
    )
    db_session.add(ForwardContextPathRow(
        signal_id=sig.signal_id,
        pattern_id="M4",
        ticker="LCUT",
        signal_horizon="15d",
        forward_session_date="2026-06-02",
        path_sequence=1,
        asof_timestamp=close + timedelta(minutes=1),
        context_json=json.dumps({"session": "2026-06-02", "corrupt": True}),
        source_attempts_json="[]",
        data_lineage_ids="[]",
        context_hash=stable_hash({"session": "2026-06-02", "corrupt": True}),
        is_terminal_snapshot=False,
    ))
    db_session.commit()

    rows = forward_context_rows_through(
        db_session,
        signal_id=sig.signal_id,
        decision_session_date="2026-06-02",
    )

    assert rows == []


def test_run_forward_context_live_requires_provider_config_before_db(monkeypatch, capsys):
    monkeypatch.setattr(run_forward_context, "load_runtime_env", lambda: None)
    monkeypatch.setattr(
        run_forward_context,
        "_required_polygon_adapter",
        lambda: (_ for _ in ()).throw(ConfigError("missing polygon config")),
    )
    monkeypatch.setattr(
        run_forward_context,
        "get_session",
        lambda: (_ for _ in ()).throw(AssertionError("database should not open")),
    )

    rc = run_forward_context.main(["--live"])

    assert rc == 1
    assert "missing polygon config" in capsys.readouterr().out
