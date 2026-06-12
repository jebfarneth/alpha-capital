from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from alpha.data.alpaca import AlpacaClock, AlpacaOrder, AlpacaPosition
from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError, stable_hash
from alpha.data.polygon import PolygonSnapshotTicker
from alpha.db.models import PaperExecutionEvent
from alpha.jobs.paper_execution import (
    EASTERN,
    ExitPolicy,
    FatalBrokerAuthError,
    GateDecision,
    PaperExecutionConfig,
    PaperExecutionEventStore,
    PaperTradingLoop,
    PatternPlugin,
    PatternRegistry,
    PremarketContext,
    compute_shared_intraday_math,
    deterministic_entry_client_order_id,
    i12_entry_gate,
    validate_paper_base_url,
)
from alpha.jobs.run_paper_execution import _run_polling_loop


def _lineage(endpoint: str = "/fake") -> LineageMeta:
    ts = datetime(2026, 6, 11, 14, 0, tzinfo=timezone.utc)
    return LineageMeta(
        provider="fake",
        endpoint=endpoint,
        request_timestamp=ts,
        asof_timestamp=ts,
        raw_payload_hash=stable_hash(endpoint),
    )


def _context() -> PremarketContext:
    return PremarketContext(
        ticker="TEST",
        context_date=date(2026, 6, 11),
        prior_close=40.0,
        max_prior_252_closes=100.0,
        avg20_volume=1000.0,
        mom20=0.1,
        off_low252=0.5,
    )


def _snapshot(
    *,
    ticker: str = "TEST",
    day_open: float = 40.0,
    day_high: float = 41.0,
    day_volume: float = 65.0,
    last_trade_price: float | None = None,
    minute_et: datetime | None = None,
) -> PolygonSnapshotTicker:
    minute_et = minute_et or datetime(2026, 6, 11, 9, 35, tzinfo=EASTERN)
    return PolygonSnapshotTicker(
        ticker=ticker,
        day_open=day_open,
        day_high=day_high,
        day_low=day_open,
        day_close=last_trade_price or day_open,
        day_volume=day_volume,
        prev_day_close=40.0,
        minute_timestamp=int(minute_et.timestamp() * 1000),
        minute_close=last_trade_price or day_open,
        minute_volume=day_volume,
        last_trade_price=last_trade_price,
    )


class FakeAlpaca:
    def __init__(
        self,
        *,
        positions: list[AlpacaPosition] | None = None,
        orders: list[AlpacaOrder] | None = None,
        closed_orders: list[AlpacaOrder] | None = None,
        clock_open: bool = True,
        auth_error: bool = False,
    ) -> None:
        self.positions = positions or []
        self.orders = orders or []
        self.closed_orders = closed_orders or []
        self.clock_open = clock_open
        self.auth_error = auth_error
        self.submitted: list[dict] = []
        self.closed: list[str] = []

    def get_positions(self):
        if self.auth_error:
            return _auth_error()
        return AdapterResponse(data=self.positions, lineage=_lineage("/v2/positions"))

    def list_orders(self, status="open", **kwargs):
        if self.auth_error:
            return _auth_error()
        orders = self.closed_orders if status == "closed" else self.orders
        return AdapterResponse(data=orders, lineage=_lineage("/v2/orders"))

    def get_clock(self):
        if self.auth_error:
            return _auth_error()
        return AdapterResponse(
            data=AlpacaClock(
                timestamp="2026-06-11T14:00:00Z",
                is_open=self.clock_open,
            ),
            lineage=_lineage("/v2/clock"),
        )

    def submit_order(self, **kwargs):
        self.submitted.append(kwargs)
        order = AlpacaOrder(
            id=f"broker-{len(self.submitted)}",
            client_order_id=kwargs.get("client_order_id", ""),
            symbol=kwargs.get("symbol", ""),
            side=kwargs.get("side", "buy"),
            order_type=kwargs.get("order_type", "market"),
            notional=str(kwargs.get("notional")),
            status="filled",
            filled_qty="2",
            filled_avg_price="12.50",
        )
        return AdapterResponse(data=order, lineage=_lineage("/v2/orders"))

    def close_position(self, symbol):
        self.closed.append(symbol)
        order = AlpacaOrder(
            id=f"close-{symbol}",
            client_order_id=f"exit-{symbol}",
            symbol=symbol,
            side="sell",
            order_type="market",
            status="filled",
            filled_qty="2",
            filled_avg_price="13.00",
        )
        return AdapterResponse(data=order, lineage=_lineage("/v2/positions"))


def _auth_error():
    return AdapterResponse(
        data=None,
        lineage=_lineage("/auth"),
        error=ProviderError(
            provider="Alpaca",
            endpoint="/auth",
            status_code=403,
            error_type="auth",
            message="auth failed",
            retryable=False,
        ),
    )


def _always_enter(pattern_id: str) -> PatternPlugin:
    return PatternPlugin(
        pattern_id=pattern_id,
        entry_gate=lambda context, snapshot, shared: GateDecision(
            True,
            "candidate_confirmed",
            {"pattern": pattern_id, **shared.to_gate_values()},
        ),
        exit_policy=ExitPolicy.SAME_DAY_CLOSE_1555,
    )


class FakePolygon:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get_full_market_snapshot(self):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return AdapterResponse(data=[], lineage=_lineage("/snapshot"))


class LoopStub:
    def __init__(self):
        self.polls = 0
        self.exit_calls = 0

    def run_snapshot_poll(self, **kwargs):
        self.polls += 1
        return {
            "snapshots": len(kwargs.get("snapshots") or []),
            "candidate_confirmed": 0,
            "orders_submitted": 0,
            "skipped": 0,
        }

    def submit_due_exits(self, **kwargs):
        self.exit_calls += 1
        return 0


def _snapshot_error_response():
    return AdapterResponse(
        data=None,
        lineage=_lineage("/snapshot"),
        error=ProviderError(
            provider="Polygon",
            endpoint="/snapshot",
            status_code=502,
            error_type="http",
            message="bad gateway",
            retryable=True,
        ),
    )


def _snapshot_ok_response(rows=None):
    return AdapterResponse(data=rows or [], lineage=_lineage("/snapshot"))


def test_i12_gate_boundaries():
    ctx = _context()

    shared = compute_shared_intraday_math(ctx, _snapshot(day_open=38.0))
    decision = i12_entry_gate(ctx, _snapshot(day_open=38.0), shared)
    assert decision.enter is True

    poison_shared = compute_shared_intraday_math(ctx, _snapshot(day_open=37.99))
    poison = i12_entry_gate(ctx, _snapshot(day_open=37.99), poison_shared)
    assert poison.enter is False
    assert poison.reason == "poison_blocked"

    high_gap_shared = compute_shared_intraday_math(ctx, _snapshot(day_open=42.0))
    high_gap = i12_entry_gate(ctx, _snapshot(day_open=42.0), high_gap_shared)
    assert high_gap.enter is False
    assert high_gap.reason == "gate_skipped"

    chase_ok_snapshot = _snapshot(day_open=40.0, last_trade_price=44.0)
    chase_ok = i12_entry_gate(ctx, chase_ok_snapshot, compute_shared_intraday_math(ctx, chase_ok_snapshot))
    assert chase_ok.enter is True
    assert chase_ok.gate_values["a_book_chase"] is True

    parabolic_snapshot = _snapshot(day_open=40.0, last_trade_price=44.01)
    parabolic = i12_entry_gate(ctx, parabolic_snapshot, compute_shared_intraday_math(ctx, parabolic_snapshot))
    assert parabolic.enter is False
    assert parabolic.reason == "parabolic_skipped"


def test_projection_uses_delayed_data_clock_not_wall_clock():
    ctx = _context()
    snapshot = _snapshot(minute_et=datetime(2026, 6, 11, 9, 35, tzinfo=EASTERN))
    shared = compute_shared_intraday_math(ctx, snapshot)

    assert shared.data_elapsed_min == pytest.approx(5.0)
    assert shared.projected_vol == pytest.approx(65.0 * 390.0 / 5.0)
    assert shared.vol_ratio == pytest.approx(5.07)


def test_pattern_registry_accepts_dummy_pattern_with_single_registration():
    registry = PatternRegistry()
    plugin = _always_enter("IX")

    registry.register(plugin)

    assert registry.selected(["IX"])[0].pattern_id == "IX"


def test_loop_dedups_same_ticker_across_patterns(db_session):
    fake = FakeAlpaca()
    loop = PaperTradingLoop(
        session=db_session,
        alpaca_adapter=fake,
        config=PaperExecutionConfig(dry_run=True),
        plugins=[_always_enter("IA"), _always_enter("IB")],
    )
    loop.reconcile_startup(wall_timestamp=datetime(2026, 6, 11, 13, 0, tzinfo=timezone.utc))

    counters = loop.run_snapshot_poll(
        snapshots=[_snapshot(ticker="SAME")],
        contexts={"SAME": _context()},
        wall_timestamp=datetime(2026, 6, 11, 14, 0, tzinfo=timezone.utc),
    )

    assert counters["candidate_confirmed"] == 2
    assert fake.submitted == []
    assert db_session.query(PaperExecutionEvent).filter_by(event_type="entry_submitted").count() == 1
    assert db_session.query(PaperExecutionEvent).filter_by(event_type="dedup_skipped").count() == 1


def test_caps_limit_new_entries_per_day(db_session):
    fake = FakeAlpaca()
    loop = PaperTradingLoop(
        session=db_session,
        alpaca_adapter=fake,
        config=PaperExecutionConfig(dry_run=True, max_new_entries_per_day=1, max_concurrent_positions=4),
        plugins=[_always_enter("IA")],
    )
    loop.reconcile_startup(wall_timestamp=datetime(2026, 6, 11, 13, 0, tzinfo=timezone.utc))

    loop.run_snapshot_poll(
        snapshots=[_snapshot(ticker="ONE"), _snapshot(ticker="TWO")],
        contexts={"ONE": _context(), "TWO": _context()},
        wall_timestamp=datetime(2026, 6, 11, 14, 0, tzinfo=timezone.utc),
    )

    assert db_session.query(PaperExecutionEvent).filter_by(event_type="entry_submitted").count() == 1
    assert db_session.query(PaperExecutionEvent).filter_by(event_type="cap_skipped").count() == 1


def test_startup_reconciliation_logs_positions_and_orders(db_session):
    fake = FakeAlpaca(
        positions=[
            AlpacaPosition(
                asset_id="asset-1",
                symbol="HELD",
                qty="3",
                avg_entry_price="10.00",
                side="long",
            )
        ],
        orders=[
            AlpacaOrder(
                id="order-1",
                client_order_id="I12_OPEN_2026-06-11",
                symbol="OPEN",
                side="buy",
                order_type="market",
                status="new",
            )
        ],
        closed_orders=[
            AlpacaOrder(
                id="entry-1",
                client_order_id="I12_HELD_2026-06-11",
                symbol="HELD",
                side="buy",
                order_type="market",
                status="filled",
                filled_qty="3",
                filled_avg_price="10.00",
            )
        ],
    )
    loop = PaperTradingLoop(
        session=db_session,
        alpaca_adapter=fake,
        config=PaperExecutionConfig(dry_run=True),
        plugins=[_always_enter("I12")],
    )

    loop.reconcile_startup(wall_timestamp=datetime(2026, 6, 11, 13, 0, tzinfo=timezone.utc))

    assert "HELD" in loop.active_positions
    assert loop.active_positions["HELD"].pattern_id == "I12"
    assert "I12_OPEN_2026-06-11" in loop.submitted_client_order_ids
    assert db_session.query(PaperExecutionEvent).filter_by(event_type="reconciled_at_startup").count() == 2


def test_reconciled_i11_position_exits_after_restart_next_open(db_session):
    fake = FakeAlpaca(
        positions=[
            AlpacaPosition(
                asset_id="asset-1",
                symbol="HELD",
                qty="2",
                avg_entry_price="10.00",
                side="long",
            )
        ],
        closed_orders=[
            AlpacaOrder(
                id="entry-i11",
                client_order_id="I11_HELD_2026-06-10",
                symbol="HELD",
                side="buy",
                order_type="market",
                status="filled",
                filled_qty="2",
                filled_avg_price="10.00",
            )
        ],
    )
    loop = PaperTradingLoop(
        session=db_session,
        alpaca_adapter=fake,
        config=PaperExecutionConfig(dry_run=False, paper_trade=True),
        plugins=[
            PatternPlugin(
                pattern_id="I11",
                entry_gate=lambda context, snapshot, shared: GateDecision(False, "gate_skipped", {}),
                exit_policy=ExitPolicy.NEXT_OPEN_0931,
            )
        ],
    )

    loop.reconcile_startup(wall_timestamp=datetime(2026, 6, 11, 13, 20, tzinfo=timezone.utc))
    submitted = loop.submit_due_exits(wall_timestamp=datetime(2026, 6, 11, 13, 32, tzinfo=timezone.utc))

    assert submitted == 1
    assert fake.submitted[0]["side"] == "sell"
    assert fake.submitted[0]["client_order_id"] == "exit_I11_HELD_2026-06-10"
    assert "HELD" not in loop.active_positions


def test_unmatched_reconciled_position_fails_safe_to_same_day_exit(db_session):
    fake = FakeAlpaca(
        positions=[
            AlpacaPosition(
                asset_id="asset-1",
                symbol="LOST",
                qty="2",
                avg_entry_price="10.00",
                side="long",
            )
        ]
    )
    loop = PaperTradingLoop(
        session=db_session,
        alpaca_adapter=fake,
        config=PaperExecutionConfig(dry_run=False, paper_trade=True),
        plugins=[_always_enter("I12")],
    )

    loop.reconcile_startup(wall_timestamp=datetime(2026, 6, 11, 13, 20, tzinfo=timezone.utc))
    submitted = loop.submit_due_exits(wall_timestamp=datetime(2026, 6, 11, 20, 0, tzinfo=timezone.utc))

    assert submitted == 1
    assert fake.submitted[0]["client_order_id"] == "exit_reconciled_LOST_2026-06-11"
    assert db_session.query(PaperExecutionEvent).filter_by(event_type="reconciled_unmatched").count() == 1


def test_same_day_exit_missed_close_sells_next_day(db_session):
    fake = FakeAlpaca(
        positions=[
            AlpacaPosition(
                asset_id="asset-1",
                symbol="LATE",
                qty="2",
                avg_entry_price="10.00",
                side="long",
            )
        ],
        closed_orders=[
            AlpacaOrder(
                id="entry-i12",
                client_order_id="I12_LATE_2026-06-10",
                symbol="LATE",
                side="buy",
                order_type="market",
                status="filled",
            )
        ],
    )
    loop = PaperTradingLoop(
        session=db_session,
        alpaca_adapter=fake,
        config=PaperExecutionConfig(dry_run=False, paper_trade=True),
        plugins=[_always_enter("I12")],
    )

    loop.reconcile_startup(wall_timestamp=datetime(2026, 6, 11, 13, 0, tzinfo=timezone.utc))
    submitted = loop.submit_due_exits(wall_timestamp=datetime(2026, 6, 11, 14, 0, tzinfo=timezone.utc))

    assert submitted == 1
    assert fake.submitted[0]["client_order_id"] == "exit_I12_LATE_2026-06-10"


def test_broker_auth_error_stops_loop(db_session):
    loop = PaperTradingLoop(
        session=db_session,
        alpaca_adapter=FakeAlpaca(auth_error=True),
        config=PaperExecutionConfig(dry_run=True),
        plugins=[_always_enter("IA")],
    )

    with pytest.raises(FatalBrokerAuthError):
        loop.reconcile_startup()


def test_event_store_is_content_idempotent(db_session):
    store = PaperExecutionEventStore(db_session)
    kwargs = {
        "ticker": "IDEM",
        "pattern_id": "I12",
        "event_type": "candidate_confirmed",
        "event_date": date(2026, 6, 11),
        "gate_values": {"gap": 0.01},
        "wall_timestamp": datetime(2026, 6, 11, 14, 0, tzinfo=timezone.utc),
        "decision_price": 10.0,
    }

    first = store.record(**kwargs)
    second = store.record(**kwargs)

    assert first.paper_execution_event_id == second.paper_execution_event_id
    assert db_session.query(PaperExecutionEvent).count() == 1


def test_client_order_id_and_paper_url_guard():
    assert deterministic_entry_client_order_id("i12", "abc", date(2026, 6, 11)) == "I12_ABC_2026-06-11"
    validate_paper_base_url("https://paper-api.alpaca.markets")
    with pytest.raises(ValueError):
        validate_paper_base_url("https://api.alpaca.markets")


def test_stale_snapshot_data_clock_does_not_enter(db_session):
    fake = FakeAlpaca()
    loop = PaperTradingLoop(
        session=db_session,
        alpaca_adapter=fake,
        config=PaperExecutionConfig(dry_run=True),
        plugins=[_always_enter("I12")],
    )
    loop.reconcile_startup(wall_timestamp=datetime(2026, 6, 11, 13, 0, tzinfo=timezone.utc))

    counters = loop.run_snapshot_poll(
        snapshots=[_snapshot(ticker="OLD", minute_et=datetime(2026, 6, 10, 15, 55, tzinfo=EASTERN))],
        contexts={"OLD": _context()},
        wall_timestamp=datetime(2026, 6, 11, 14, 0, tzinfo=timezone.utc),
    )

    assert counters["orders_submitted"] == 0
    assert db_session.query(PaperExecutionEvent).filter_by(event_type="entry_submitted").count() == 0


def test_i12_poison_is_recorded_only_for_live_volume_candidates_once_per_day(db_session):
    fake = FakeAlpaca()
    loop = PaperTradingLoop(
        session=db_session,
        alpaca_adapter=fake,
        config=PaperExecutionConfig(dry_run=True),
        plugins=[
            PatternPlugin(
                pattern_id="I12",
                entry_gate=i12_entry_gate,
                exit_policy=ExitPolicy.SAME_DAY_CLOSE_1555,
            )
        ],
    )
    loop.reconcile_startup(wall_timestamp=datetime(2026, 6, 11, 13, 0, tzinfo=timezone.utc))

    loop.run_snapshot_poll(
        snapshots=[_snapshot(ticker="LOWV", day_open=37.99, day_volume=1)],
        contexts={"LOWV": _context()},
        wall_timestamp=datetime(2026, 6, 11, 14, 0, tzinfo=timezone.utc),
    )
    assert db_session.query(PaperExecutionEvent).filter_by(event_type="poison_blocked").count() == 0

    loop.run_snapshot_poll(
        snapshots=[_snapshot(ticker="HIGHV", day_open=37.99, day_volume=80)],
        contexts={"HIGHV": _context()},
        wall_timestamp=datetime(2026, 6, 11, 14, 1, tzinfo=timezone.utc),
    )
    loop.run_snapshot_poll(
        snapshots=[
            _snapshot(
                ticker="HIGHV",
                day_open=37.99,
                day_volume=80,
                minute_et=datetime(2026, 6, 11, 9, 36, tzinfo=EASTERN),
            )
        ],
        contexts={"HIGHV": _context()},
        wall_timestamp=datetime(2026, 6, 11, 14, 2, tzinfo=timezone.utc),
    )

    assert db_session.query(PaperExecutionEvent).filter_by(event_type="poison_blocked").count() == 1


def test_restart_seeds_entry_cap_and_submitted_order_ids(db_session):
    store = PaperExecutionEventStore(db_session)
    event_date = date(2026, 6, 11)
    for idx in range(4):
        ticker = f"CAP{idx}"
        store.record(
            ticker=ticker,
            pattern_id="I12",
            event_type="entry_submitted",
            event_date=event_date,
            wall_timestamp=datetime(2026, 6, 11, 14, idx, tzinfo=timezone.utc),
            client_order_id=f"I12_{ticker}_2026-06-11",
        )
    fake = FakeAlpaca()
    loop = PaperTradingLoop(
        session=db_session,
        alpaca_adapter=fake,
        config=PaperExecutionConfig(dry_run=True, max_new_entries_per_day=4, max_concurrent_positions=10),
        plugins=[_always_enter("I12")],
    )

    loop.reconcile_startup(wall_timestamp=datetime(2026, 6, 11, 14, 30, tzinfo=timezone.utc))
    loop.run_snapshot_poll(
        snapshots=[_snapshot(ticker="NEXT")],
        contexts={"NEXT": _context()},
        wall_timestamp=datetime(2026, 6, 11, 14, 31, tzinfo=timezone.utc),
    )

    assert db_session.query(PaperExecutionEvent).filter_by(event_type="entry_submitted").count() == 4
    assert db_session.query(PaperExecutionEvent).filter_by(event_type="cap_skipped").count() == 1


def test_exited_i11_overnights_free_slots_for_new_entries(db_session):
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    fake = FakeAlpaca(
        positions=[
            AlpacaPosition(
                asset_id=f"asset-{ticker}",
                symbol=ticker,
                qty="2",
                avg_entry_price="10.00",
                side="long",
            )
            for ticker in tickers
        ],
        closed_orders=[
            AlpacaOrder(
                id=f"entry-{ticker}",
                client_order_id=f"I11_{ticker}_2026-06-10",
                symbol=ticker,
                side="buy",
                order_type="market",
                status="filled",
                filled_qty="2",
                filled_avg_price="10.00",
            )
            for ticker in tickers
        ],
    )
    loop = PaperTradingLoop(
        session=db_session,
        alpaca_adapter=fake,
        config=PaperExecutionConfig(
            dry_run=True,
            max_concurrent_positions=4,
            max_new_entries_per_day=4,
        ),
        plugins=[
            PatternPlugin(
                pattern_id="I11",
                entry_gate=lambda context, snapshot, shared: GateDecision(False, "gate_skipped", {}),
                exit_policy=ExitPolicy.NEXT_OPEN_0931,
            ),
            _always_enter("I12"),
        ],
    )

    loop.reconcile_startup(wall_timestamp=datetime(2026, 6, 11, 13, 20, tzinfo=timezone.utc))
    exits = loop.submit_due_exits(wall_timestamp=datetime(2026, 6, 11, 13, 32, tzinfo=timezone.utc))
    counters = loop.run_snapshot_poll(
        snapshots=[_snapshot(ticker="FRESH")],
        contexts={"FRESH": _context()},
        wall_timestamp=datetime(2026, 6, 11, 14, 0, tzinfo=timezone.utc),
    )

    assert exits == 4
    assert loop.active_positions.keys() == {"FRESH"}
    assert counters["orders_submitted"] == 1
    assert db_session.query(PaperExecutionEvent).filter_by(event_type="entry_submitted").count() == 1
    assert db_session.query(PaperExecutionEvent).filter_by(event_type="cap_skipped").count() == 0


def test_reconciled_open_exit_order_strips_exit_prefix_for_pattern(db_session):
    fake = FakeAlpaca(
        orders=[
            AlpacaOrder(
                id="exit-order-1",
                client_order_id="exit_I11_HELD_2026-06-10",
                symbol="HELD",
                side="sell",
                order_type="market",
                status="new",
            )
        ],
    )
    loop = PaperTradingLoop(
        session=db_session,
        alpaca_adapter=fake,
        config=PaperExecutionConfig(dry_run=True),
        plugins=[_always_enter("I11")],
    )

    loop.reconcile_startup(wall_timestamp=datetime(2026, 6, 11, 13, 0, tzinfo=timezone.utc))

    event = db_session.query(PaperExecutionEvent).filter_by(client_order_id="exit_I11_HELD_2026-06-10").one()
    assert event.pattern_id == "I11"


def test_snapshot_failures_retry_until_success():
    loop = LoopStub()
    polygon = FakePolygon([
        _snapshot_error_response(),
        _snapshot_error_response(),
        _snapshot_ok_response([_snapshot(ticker="OK")]),
    ])

    rc = _run_polling_loop(
        loop=loop,
        polygon=polygon,
        alpaca_adapter=FakeAlpaca(clock_open=True),
        contexts={"OK": _context()},
        poll_interval_seconds=1,
        once=True,
        max_consecutive_snapshot_failures=3,
        sleep_fn=lambda _: None,
    )

    assert rc == 0
    assert polygon.calls == 3
    assert loop.polls == 1


def test_snapshot_failures_exit_after_threshold():
    loop = LoopStub()
    polygon = FakePolygon([
        _snapshot_error_response(),
        _snapshot_error_response(),
        _snapshot_error_response(),
    ])

    rc = _run_polling_loop(
        loop=loop,
        polygon=polygon,
        alpaca_adapter=FakeAlpaca(clock_open=True),
        contexts={},
        poll_interval_seconds=1,
        once=True,
        max_consecutive_snapshot_failures=3,
        sleep_fn=lambda _: None,
    )

    assert rc == 1
    assert polygon.calls == 3
    assert loop.polls == 0
    assert loop.exit_calls == 4
