"""Pattern-agnostic paper-trading execution loop.

This is a live-leg research path, not the canonical detector stack. The loop
uses Polygon's delayed consolidated data as the data clock for entries and
Alpaca paper trading for optional order submission. It never writes
``signal_registry``; all telemetry is persisted to ``paper_execution_events``.
Exit scheduling is wall-clock driven because broker order timing is real time,
while entry gate elapsed-time math is data-clock driven because Polygon data is
delayed by roughly 15 minutes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from alpha.data.alpaca import AlpacaAdapter, AlpacaOrder, AlpacaPosition
from alpha.data.contracts import AdapterResponse, ProviderError, stable_hash, utcnow
from alpha.data.polygon import (
    PolygonAdapter,
    PolygonBar,
    PolygonGroupedDailyBar,
    PolygonSnapshotTicker,
)
from alpha.db.models import PaperExecutionEvent


EASTERN = ZoneInfo("America/New_York")
CONTEXT_ARTIFACT_VERSION = "paper_execution_premarket_context_v1"
DEFAULT_NOTIONAL = 250.0
DEFAULT_MAX_CONCURRENT_POSITIONS = 4
DEFAULT_MAX_NEW_ENTRIES_PER_DAY = 4
I12_PATTERN_ID = "I12"
I11_PATTERN_ID = "I11"
BOUNDARY_EPSILON = 1e-12


class FatalBrokerAuthError(RuntimeError):
    """Raised when Alpaca returns an auth error and the loop must stop."""


class ExitPolicy(str, Enum):
    SAME_DAY_CLOSE_1555 = "same_day_close_1555"
    NEXT_OPEN_0931 = "next_open_0931"
    HORIZON_N_DAYS = "horizon_n_days"


@dataclass(frozen=True)
class GateDecision:
    """Pattern plug-in entry decision."""

    enter: bool
    reason: str
    gate_values: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatternPlugin:
    """Minimal pattern plug-in contract."""

    pattern_id: str
    entry_gate: Callable[["PremarketContext", PolygonSnapshotTicker, "SharedIntradayMath"], GateDecision]
    exit_policy: ExitPolicy
    horizon_days: Optional[int] = None


@dataclass(frozen=True)
class PremarketContext:
    """Daily context required before the intraday loop starts."""

    ticker: str
    context_date: date
    prior_close: float
    max_prior_252_closes: float
    avg20_volume: float
    mom20: Optional[float] = None
    off_low252: Optional[float] = None
    sigma20: Optional[float] = None
    prev_day_return: Optional[float] = None
    prev_day_green: Optional[bool] = None
    spy_prior_day_return: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "context_date": self.context_date.isoformat(),
            "prior_close": self.prior_close,
            "max_prior_252_closes": self.max_prior_252_closes,
            "avg20_volume": self.avg20_volume,
            "mom20": self.mom20,
            "off_low252": self.off_low252,
            "sigma20": self.sigma20,
            "prev_day_return": self.prev_day_return,
            "prev_day_green": self.prev_day_green,
            "spy_prior_day_return": self.spy_prior_day_return,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "PremarketContext":
        return cls(
            ticker=str(payload["ticker"]).upper(),
            context_date=date.fromisoformat(str(payload["context_date"])),
            prior_close=float(payload["prior_close"]),
            max_prior_252_closes=float(payload["max_prior_252_closes"]),
            avg20_volume=float(payload["avg20_volume"]),
            mom20=_optional_float(payload.get("mom20")),
            off_low252=_optional_float(payload.get("off_low252")),
            sigma20=_optional_float(payload.get("sigma20")),
            prev_day_return=_optional_float(payload.get("prev_day_return")),
            prev_day_green=(
                bool(payload["prev_day_green"])
                if payload.get("prev_day_green") is not None
                else None
            ),
            spy_prior_day_return=_optional_float(payload.get("spy_prior_day_return")),
        )


@dataclass(frozen=True)
class SharedIntradayMath:
    """Per-ticker math computed once by the loop and shared with plug-ins."""

    gap: float
    data_elapsed_min: float
    projected_vol: float
    vol_ratio: float
    chase: Optional[float]
    data_clock: datetime

    def to_gate_values(self) -> Dict[str, Any]:
        return {
            "gap": self.gap,
            "data_elapsed_min": self.data_elapsed_min,
            "projected_vol": self.projected_vol,
            "vol_ratio": self.vol_ratio,
            "chase": self.chase,
            "data_clock": self.data_clock.isoformat(),
        }


@dataclass
class ManagedPosition:
    """In-memory execution state reconstructed from broker and local orders."""

    ticker: str
    pattern_id: str
    entry_date: date
    exit_policy: ExitPolicy
    entry_client_order_id: str
    broker_order_id: Optional[str] = None
    qty: Optional[float] = None
    horizon_days: Optional[int] = None


@dataclass(frozen=True)
class ParsedEntryClientOrderId:
    """Decoded deterministic entry order id."""

    pattern_id: str
    ticker: str
    entry_date: date


@dataclass(frozen=True)
class PaperExecutionConfig:
    """Runtime policy owned by the pattern-agnostic execution core."""

    notional: float = DEFAULT_NOTIONAL
    max_concurrent_positions: int = DEFAULT_MAX_CONCURRENT_POSITIONS
    max_new_entries_per_day: int = DEFAULT_MAX_NEW_ENTRIES_PER_DAY
    dry_run: bool = True
    paper_trade: bool = False
    confirm_live_trade: bool = False

    def __post_init__(self) -> None:
        if self.notional <= 0:
            raise ValueError("notional must be positive")
        if self.max_concurrent_positions < 1:
            raise ValueError("max_concurrent_positions must be >= 1")
        if self.max_new_entries_per_day < 1:
            raise ValueError("max_new_entries_per_day must be >= 1")
        if self.dry_run and self.paper_trade:
            raise ValueError("dry_run and paper_trade cannot both be true")


class PatternRegistry:
    """Small registry proving future patterns only need a gate and registration."""

    def __init__(self) -> None:
        self._plugins: Dict[str, PatternPlugin] = {}

    def register(self, plugin: PatternPlugin) -> None:
        pattern_id = plugin.pattern_id.upper().strip()
        if not pattern_id:
            raise ValueError("pattern_id is required")
        if pattern_id in self._plugins:
            raise ValueError(f"pattern {pattern_id} is already registered")
        self._plugins[pattern_id] = PatternPlugin(
            pattern_id=pattern_id,
            entry_gate=plugin.entry_gate,
            exit_policy=plugin.exit_policy,
            horizon_days=plugin.horizon_days,
        )

    def selected(self, pattern_ids: Sequence[str] | None = None) -> List[PatternPlugin]:
        if not pattern_ids:
            return list(self._plugins.values())
        selected: List[PatternPlugin] = []
        for raw in pattern_ids:
            pattern_id = raw.upper().strip()
            if pattern_id not in self._plugins:
                raise ValueError(f"unknown pattern {pattern_id}")
            selected.append(self._plugins[pattern_id])
        return selected


def default_pattern_registry() -> PatternRegistry:
    registry = PatternRegistry()
    registry.register(
        PatternPlugin(
            pattern_id=I12_PATTERN_ID,
            entry_gate=i12_entry_gate,
            exit_policy=ExitPolicy.SAME_DAY_CLOSE_1555,
        )
    )
    registry.register(
        PatternPlugin(
            pattern_id=I11_PATTERN_ID,
            entry_gate=i11_entry_gate,
            exit_policy=ExitPolicy.NEXT_OPEN_0931,
        )
    )
    return registry


def i12_entry_gate(
    context: PremarketContext,
    snapshot: PolygonSnapshotTicker,
    shared: SharedIntradayMath,
) -> GateDecision:
    """Frozen I12 delayed-data gate."""

    gate_values = {
        **shared.to_gate_values(),
        "prior_close": context.prior_close,
        "max252": context.max_prior_252_closes,
        "avg20": context.avg20_volume,
        "mom20": context.mom20,
        "off_low252": context.off_low252,
    }
    distance_from_max = (context.prior_close / context.max_prior_252_closes) - 1.0
    gate_values["distance_from_max252"] = distance_from_max
    gate_values["a_book_chase"] = (
        shared.chase is not None
        and 0.02 - BOUNDARY_EPSILON <= shared.chase <= 0.10 + BOUNDARY_EPSILON
        and shared.gap >= 0
    )
    otherwise_live_candidate = (
        distance_from_max <= -0.50
        and shared.vol_ratio >= 5
        and shared.data_elapsed_min >= 5
    )
    if otherwise_live_candidate and shared.gap < -0.05 - BOUNDARY_EPSILON:
        return GateDecision(False, "poison_blocked", gate_values)
    if (
        otherwise_live_candidate
        and shared.chase is not None
        and shared.chase > 0.10 + BOUNDARY_EPSILON
    ):
        return GateDecision(False, "parabolic_skipped", gate_values)
    enter = (
        distance_from_max <= -0.50
        and -0.05 - BOUNDARY_EPSILON <= shared.gap < 0.05
        and shared.vol_ratio >= 5
        and shared.data_elapsed_min >= 5
        and shared.data_elapsed_min <= 60
    )
    return GateDecision(enter, "candidate_confirmed" if enter else "gate_skipped", gate_values)


def i11_entry_gate(
    context: PremarketContext,
    snapshot: PolygonSnapshotTicker,
    shared: SharedIntradayMath,
) -> GateDecision:
    """Frozen I11 delayed-data gate."""

    gate_values = {
        **shared.to_gate_values(),
        "prior_close": context.prior_close,
        "max252": context.max_prior_252_closes,
        "avg20": context.avg20_volume,
        "mom20": context.mom20,
        "off_low252": context.off_low252,
        "day_high": snapshot.day_high,
    }
    enter = (
        snapshot.day_high is not None
        and snapshot.day_high > context.max_prior_252_closes
        and context.prior_close <= context.max_prior_252_closes
        and shared.vol_ratio >= 5
        and shared.data_elapsed_min >= 5
    )
    return GateDecision(enter, "candidate_confirmed" if enter else "gate_skipped", gate_values)


class PaperExecutionEventStore:
    """Content-idempotent writer for paper execution telemetry."""

    def __init__(self, session: Session):
        self._session = session

    def record(
        self,
        *,
        ticker: str,
        pattern_id: str,
        event_type: str,
        event_date: date,
        gate_values: Optional[Mapping[str, Any]] = None,
        event_payload: Optional[Mapping[str, Any]] = None,
        data_timestamp: Optional[datetime] = None,
        wall_timestamp: Optional[datetime] = None,
        decision_price: Optional[float] = None,
        broker_order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        fill_price: Optional[float] = None,
        fill_qty: Optional[float] = None,
        lineage_hash: Optional[str] = None,
    ) -> PaperExecutionEvent:
        wall_ts = _aware_utc(wall_timestamp or utcnow())
        data_ts = _aware_utc(data_timestamp) if data_timestamp else None
        gate_json = _json_dumps(gate_values or {})
        payload_json = _json_dumps(event_payload or {})
        content_hash = stable_hash(
            {
                "ticker": ticker.upper(),
                "pattern_id": pattern_id.upper(),
                "event_type": event_type,
                "event_date": event_date.isoformat(),
                "gate_values": json.loads(gate_json),
                "event_payload": json.loads(payload_json),
                "data_timestamp": data_ts.isoformat() if data_ts else None,
                "decision_price": decision_price,
                "broker_order_id": broker_order_id,
                "client_order_id": client_order_id,
                "fill_price": fill_price,
                "fill_qty": fill_qty,
                "lineage_hash": lineage_hash,
            }
        )
        existing = (
            self._session.query(PaperExecutionEvent)
            .filter(PaperExecutionEvent.content_hash == content_hash)
            .one_or_none()
        )
        if existing is not None:
            return existing
        row = PaperExecutionEvent(
            ticker=ticker.upper(),
            pattern_id=pattern_id.upper(),
            event_type=event_type,
            event_date=event_date,
            gate_values_json=gate_json,
            event_payload_json=payload_json,
            data_timestamp=data_ts,
            wall_timestamp=wall_ts,
            decision_price=decision_price,
            broker_order_id=broker_order_id,
            client_order_id=client_order_id,
            fill_price=fill_price,
            fill_qty=fill_qty,
            lineage_hash=lineage_hash,
            content_hash=content_hash,
        )
        self._session.add(row)
        self._session.flush()
        return row


class PaperTradingLoop:
    """Pattern-agnostic intraday loop core."""

    def __init__(
        self,
        *,
        session: Session,
        alpaca_adapter: AlpacaAdapter,
        config: PaperExecutionConfig,
        plugins: Sequence[PatternPlugin],
    ) -> None:
        self._session = session
        self._alpaca = alpaca_adapter
        self._config = config
        self._plugins = list(plugins)
        self._plugins_by_id = {plugin.pattern_id.upper(): plugin for plugin in self._plugins}
        self._events = PaperExecutionEventStore(session)
        self.active_positions: Dict[str, ManagedPosition] = {}
        self.submitted_client_order_ids: set[str] = set()
        self.entries_by_day: Dict[date, int] = {}
        self._recorded_daily_events: set[tuple[str, str, str, date]] = set()

    def reconcile_startup(self, *, wall_timestamp: Optional[datetime] = None) -> None:
        wall_ts = _aware_utc(wall_timestamp or utcnow())
        event_date = wall_ts.astimezone(EASTERN).date()
        self._seed_intraday_state_from_events(event_date)
        positions_resp = self._alpaca.get_positions()
        _raise_on_auth(positions_resp.error)
        orders_resp = self._alpaca.list_orders(status="open")
        _raise_on_auth(orders_resp.error)
        closed_orders_resp = self._alpaca.list_orders(status="closed")
        _raise_on_auth(closed_orders_resp.error)
        closed_orders = closed_orders_resp.data or []
        for position in positions_resp.data or []:
            self._reconcile_position(position, wall_ts, closed_orders)
        for order in orders_resp.data or []:
            if order.client_order_id:
                self.submitted_client_order_ids.add(order.client_order_id)
            self._events.record(
                ticker=order.symbol or "UNKNOWN",
                pattern_id=_pattern_from_client_order_id(order.client_order_id) or "UNKNOWN",
                event_type="reconciled_at_startup",
                event_date=wall_ts.astimezone(EASTERN).date(),
                wall_timestamp=wall_ts,
                client_order_id=order.client_order_id,
                broker_order_id=order.id,
                event_payload={"broker_status": order.status, "source": "open_order"},
            )
        self._session.commit()

    def run_snapshot_poll(
        self,
        *,
        snapshots: Iterable[PolygonSnapshotTicker],
        contexts: Mapping[str, PremarketContext],
        wall_timestamp: Optional[datetime] = None,
        lineage_hash: Optional[str] = None,
    ) -> Dict[str, int]:
        wall_ts = _aware_utc(wall_timestamp or utcnow())
        counters = {
            "snapshots": 0,
            "candidate_confirmed": 0,
            "orders_submitted": 0,
            "skipped": 0,
            "exits_submitted": 0,
        }
        counters["exits_submitted"] = self.submit_due_exits(wall_timestamp=wall_ts)
        for snapshot in snapshots:
            counters["snapshots"] += 1
            ticker = snapshot.ticker.upper()
            context = contexts.get(ticker)
            if context is None:
                counters["skipped"] += 1
                continue
            hygiene_reason = _universe_hygiene_skip(snapshot)
            if hygiene_reason:
                counters["skipped"] += 1
                self._record_skip(
                    snapshot=snapshot,
                    context=context,
                    pattern_id="CORE",
                    event_type="hygiene_skipped",
                    reason=hygiene_reason,
                    wall_timestamp=wall_ts,
                    lineage_hash=lineage_hash,
                )
                continue
            shared = compute_shared_intraday_math(
                context,
                snapshot,
                trading_date=context.context_date,
            )
            if shared is None:
                counters["skipped"] += 1
                continue
            for plugin in self._plugins:
                decision = plugin.entry_gate(context, snapshot, shared)
                event_type = _decision_event_type(decision.reason)
                if not decision.enter:
                    if event_type in {"poison_blocked", "parabolic_skipped"}:
                        self._record_skip(
                            snapshot=snapshot,
                            context=context,
                            pattern_id=plugin.pattern_id,
                            event_type=event_type,
                            reason=decision.reason,
                            wall_timestamp=wall_ts,
                            gate_values=decision.gate_values,
                            lineage_hash=lineage_hash,
                        )
                    counters["skipped"] += 1
                    continue
                counters["candidate_confirmed"] += 1
                self._record_once_daily(
                    ticker=ticker,
                    pattern_id=plugin.pattern_id,
                    event_type="candidate_confirmed",
                    event_date=wall_ts.astimezone(EASTERN).date(),
                    gate_values=decision.gate_values,
                    data_timestamp=shared.data_clock,
                    wall_timestamp=wall_ts,
                    decision_price=snapshot.decision_price,
                    lineage_hash=lineage_hash,
                )
                if self._can_enter(ticker, plugin.pattern_id, wall_ts, snapshot, decision):
                    submitted = self._submit_entry(
                        plugin=plugin,
                        snapshot=snapshot,
                        decision=decision,
                        shared=shared,
                        wall_timestamp=wall_ts,
                        lineage_hash=lineage_hash,
                    )
                    counters["orders_submitted"] += int(submitted)
        self._session.commit()
        return counters

    def submit_due_exits(self, *, wall_timestamp: Optional[datetime] = None) -> int:
        wall_ts = _aware_utc(wall_timestamp or utcnow())
        submitted = 0
        for ticker, position in list(self.active_positions.items()):
            if not _exit_due(position, wall_ts):
                continue
            client_order_id = f"exit_{position.entry_client_order_id}"
            if client_order_id in self.submitted_client_order_ids:
                continue
            if self._config.dry_run:
                self._events.record(
                    ticker=ticker,
                    pattern_id=position.pattern_id,
                    event_type="exit_submitted",
                    event_date=wall_ts.astimezone(EASTERN).date(),
                    wall_timestamp=wall_ts,
                    client_order_id=client_order_id,
                    event_payload={"dry_run": True, "exit_policy": position.exit_policy.value},
                )
                self.submitted_client_order_ids.add(client_order_id)
                submitted += 1
                self.active_positions.pop(ticker, None)
                continue
            if position.qty is not None and position.qty > 0:
                order_resp = self._alpaca.submit_order(
                    symbol=ticker,
                    qty=position.qty,
                    side="sell",
                    order_type="market",
                    time_in_force="day",
                    client_order_id=client_order_id,
                )
            else:
                order_resp = self._alpaca.close_position(ticker)
            _raise_on_auth(order_resp.error)
            if not order_resp.ok or order_resp.data is None:
                self._events.record(
                    ticker=ticker,
                    pattern_id=position.pattern_id,
                    event_type="order_error",
                    event_date=wall_ts.astimezone(EASTERN).date(),
                    wall_timestamp=wall_ts,
                    client_order_id=client_order_id,
                    event_payload=_provider_error_payload(order_resp.error),
                )
                continue
            order = order_resp.data
            self.submitted_client_order_ids.add(client_order_id)
            submitted += 1
            self.active_positions.pop(ticker, None)
            self._events.record(
                ticker=ticker,
                pattern_id=position.pattern_id,
                event_type="exit_submitted",
                event_date=wall_ts.astimezone(EASTERN).date(),
                wall_timestamp=wall_ts,
                client_order_id=client_order_id,
                broker_order_id=order.id,
                fill_price=_optional_float(order.filled_avg_price),
                fill_qty=_optional_float(order.filled_qty),
                event_payload={"broker_status": order.status, "exit_policy": position.exit_policy.value},
            )
            if order.status == "filled":
                self._events.record(
                    ticker=ticker,
                    pattern_id=position.pattern_id,
                    event_type="exit_filled",
                    event_date=wall_ts.astimezone(EASTERN).date(),
                    wall_timestamp=wall_ts,
                    client_order_id=client_order_id,
                    broker_order_id=order.id,
                    fill_price=_optional_float(order.filled_avg_price),
                    fill_qty=_optional_float(order.filled_qty),
                )
        if submitted:
            self._session.commit()
        return submitted

    def _reconcile_position(
        self,
        position: AlpacaPosition,
        wall_ts: datetime,
        closed_orders: Sequence[AlpacaOrder],
    ) -> None:
        ticker = position.symbol.upper()
        event_date = wall_ts.astimezone(EASTERN).date()
        entry_order, parsed_order = _matching_closed_entry_order(position, closed_orders)
        event_type = "reconciled_unmatched"
        pattern_id = "UNKNOWN"
        entry_date = event_date
        exit_policy = ExitPolicy.SAME_DAY_CLOSE_1555
        horizon_days = None
        broker_order_id = None
        entry_client_order_id = f"reconciled_{ticker}_{event_date.isoformat()}"
        payload: Dict[str, Any] = {
            "source": "position",
            "side": position.side,
            "matched_entry_order": False,
        }
        if entry_order is not None and parsed_order is not None:
            plugin = self._plugins_by_id.get(parsed_order.pattern_id)
            if plugin is not None:
                event_type = "reconciled_at_startup"
                pattern_id = plugin.pattern_id
                entry_date = parsed_order.entry_date
                exit_policy = plugin.exit_policy
                horizon_days = plugin.horizon_days
                broker_order_id = entry_order.id
                entry_client_order_id = entry_order.client_order_id
                payload.update(
                    {
                        "matched_entry_order": True,
                        "broker_status": entry_order.status,
                        "entry_order_side": entry_order.side,
                    }
                )
            else:
                payload["unmatched_reason"] = "entry_order_pattern_not_registered"
        self.submitted_client_order_ids.add(entry_client_order_id)
        self.active_positions[ticker] = ManagedPosition(
            ticker=ticker,
            pattern_id=pattern_id,
            entry_date=entry_date,
            exit_policy=exit_policy,
            entry_client_order_id=entry_client_order_id,
            broker_order_id=broker_order_id,
            qty=_optional_float(position.qty),
            horizon_days=horizon_days,
        )
        self._events.record(
            ticker=ticker,
            pattern_id=pattern_id,
            event_type=event_type,
            event_date=event_date,
            wall_timestamp=wall_ts,
            broker_order_id=broker_order_id,
            client_order_id=entry_client_order_id,
            fill_price=_optional_float(position.avg_entry_price),
            fill_qty=_optional_float(position.qty),
            event_payload=payload,
        )

    def _seed_intraday_state_from_events(self, event_date: date) -> None:
        rows = (
            self._session.query(PaperExecutionEvent)
            .filter(
                PaperExecutionEvent.event_date == event_date,
                PaperExecutionEvent.event_type.in_(
                    [
                        "entry_submitted",
                        "exit_submitted",
                        "candidate_confirmed",
                        "poison_blocked",
                        "parabolic_skipped",
                    ]
                ),
            )
            .all()
        )
        entry_count = 0
        for row in rows:
            if row.client_order_id:
                self.submitted_client_order_ids.add(row.client_order_id)
            if row.event_type == "entry_submitted":
                entry_count += 1
            if row.event_type in {"candidate_confirmed", "poison_blocked", "parabolic_skipped"}:
                self._recorded_daily_events.add(
                    (
                        row.ticker.upper(),
                        row.pattern_id.upper(),
                        row.event_type,
                        row.event_date,
                    )
                )
        if entry_count:
            self.entries_by_day[event_date] = max(
                self.entries_by_day.get(event_date, 0),
                entry_count,
            )

    def _can_enter(
        self,
        ticker: str,
        pattern_id: str,
        wall_ts: datetime,
        snapshot: PolygonSnapshotTicker,
        decision: GateDecision,
    ) -> bool:
        event_date = wall_ts.astimezone(EASTERN).date()
        if ticker in self.active_positions:
            self._record_core_skip(
                snapshot,
                pattern_id,
                "dedup_skipped",
                "ticker_already_has_position",
                wall_ts,
                decision.gate_values,
            )
            return False
        if len(self.active_positions) >= self._config.max_concurrent_positions:
            self._record_core_skip(
                snapshot,
                pattern_id,
                "cap_skipped",
                "max_concurrent_positions",
                wall_ts,
                decision.gate_values,
            )
            return False
        if self.entries_by_day.get(event_date, 0) >= self._config.max_new_entries_per_day:
            self._record_core_skip(
                snapshot,
                pattern_id,
                "cap_skipped",
                "max_new_entries_per_day",
                wall_ts,
                decision.gate_values,
            )
            return False
        client_order_id = deterministic_entry_client_order_id(pattern_id, ticker, event_date)
        if client_order_id in self.submitted_client_order_ids:
            self._record_core_skip(
                snapshot,
                pattern_id,
                "dedup_skipped",
                "client_order_id_already_submitted",
                wall_ts,
                decision.gate_values,
            )
            return False
        return True

    def _submit_entry(
        self,
        *,
        plugin: PatternPlugin,
        snapshot: PolygonSnapshotTicker,
        decision: GateDecision,
        shared: SharedIntradayMath,
        wall_timestamp: datetime,
        lineage_hash: Optional[str],
    ) -> bool:
        ticker = snapshot.ticker.upper()
        event_date = wall_timestamp.astimezone(EASTERN).date()
        client_order_id = deterministic_entry_client_order_id(
            plugin.pattern_id,
            ticker,
            event_date,
        )
        payload = {
            "notional": self._config.notional,
            "dry_run": self._config.dry_run,
            "paper_trade": self._config.paper_trade,
            "exit_policy": plugin.exit_policy.value,
        }
        if self._config.dry_run:
            self.submitted_client_order_ids.add(client_order_id)
            self.entries_by_day[event_date] = self.entries_by_day.get(event_date, 0) + 1
            self._events.record(
                ticker=ticker,
                pattern_id=plugin.pattern_id,
                event_type="entry_submitted",
                event_date=event_date,
                gate_values=decision.gate_values,
                data_timestamp=shared.data_clock,
                wall_timestamp=wall_timestamp,
                decision_price=snapshot.decision_price,
                client_order_id=client_order_id,
                lineage_hash=lineage_hash,
                event_payload=payload,
            )
            self.active_positions[ticker] = ManagedPosition(
                ticker=ticker,
                pattern_id=plugin.pattern_id,
                entry_date=event_date,
                exit_policy=plugin.exit_policy,
                entry_client_order_id=client_order_id,
                qty=None,
                horizon_days=plugin.horizon_days,
            )
            return True

        order_resp = self._alpaca.submit_order(
            symbol=ticker,
            notional=self._config.notional,
            side="buy",
            order_type="market",
            time_in_force="day",
            client_order_id=client_order_id,
        )
        _raise_on_auth(order_resp.error)
        if not order_resp.ok or order_resp.data is None:
            self._events.record(
                ticker=ticker,
                pattern_id=plugin.pattern_id,
                event_type="order_error",
                event_date=event_date,
                gate_values=decision.gate_values,
                data_timestamp=shared.data_clock,
                wall_timestamp=wall_timestamp,
                decision_price=snapshot.decision_price,
                client_order_id=client_order_id,
                lineage_hash=lineage_hash,
                event_payload=_provider_error_payload(order_resp.error),
            )
            return False
        order = order_resp.data
        self.submitted_client_order_ids.add(client_order_id)
        self.entries_by_day[event_date] = self.entries_by_day.get(event_date, 0) + 1
        self.active_positions[ticker] = ManagedPosition(
            ticker=ticker,
            pattern_id=plugin.pattern_id,
            entry_date=event_date,
            exit_policy=plugin.exit_policy,
            entry_client_order_id=client_order_id,
            broker_order_id=order.id,
            qty=_optional_float(order.filled_qty),
            horizon_days=plugin.horizon_days,
        )
        self._events.record(
            ticker=ticker,
            pattern_id=plugin.pattern_id,
            event_type="entry_submitted",
            event_date=event_date,
            gate_values=decision.gate_values,
            data_timestamp=shared.data_clock,
            wall_timestamp=wall_timestamp,
            decision_price=snapshot.decision_price,
            broker_order_id=order.id,
            client_order_id=client_order_id,
            fill_price=_optional_float(order.filled_avg_price),
            fill_qty=_optional_float(order.filled_qty),
            lineage_hash=lineage_hash,
            event_payload={**payload, "broker_status": order.status},
        )
        if order.status == "filled":
            self._events.record(
                ticker=ticker,
                pattern_id=plugin.pattern_id,
                event_type="entry_filled",
                event_date=event_date,
                gate_values=decision.gate_values,
                data_timestamp=shared.data_clock,
                wall_timestamp=wall_timestamp,
                decision_price=snapshot.decision_price,
                broker_order_id=order.id,
                client_order_id=client_order_id,
                fill_price=_optional_float(order.filled_avg_price),
                fill_qty=_optional_float(order.filled_qty),
                lineage_hash=lineage_hash,
            )
        return True

    def _record_skip(
        self,
        *,
        snapshot: PolygonSnapshotTicker,
        context: PremarketContext,
        pattern_id: str,
        event_type: str,
        reason: str,
        wall_timestamp: datetime,
        gate_values: Optional[Mapping[str, Any]] = None,
        lineage_hash: Optional[str] = None,
    ) -> None:
        values = dict(gate_values or {})
        values.setdefault("prior_close", context.prior_close)
        values.setdefault("max252", context.max_prior_252_closes)
        values.setdefault("avg20", context.avg20_volume)
        values.setdefault("mom20", context.mom20)
        values.setdefault("off_low252", context.off_low252)
        self._record_once_daily(
            ticker=snapshot.ticker,
            pattern_id=pattern_id,
            event_type=event_type,
            event_date=wall_timestamp.astimezone(EASTERN).date(),
            gate_values=values,
            data_timestamp=_snapshot_data_clock(snapshot),
            wall_timestamp=wall_timestamp,
            decision_price=snapshot.decision_price,
            lineage_hash=lineage_hash,
            event_payload={"reason": reason},
        )

    def _record_core_skip(
        self,
        snapshot: PolygonSnapshotTicker,
        pattern_id: str,
        event_type: str,
        reason: str,
        wall_timestamp: datetime,
        gate_values: Mapping[str, Any],
    ) -> None:
        self._events.record(
            ticker=snapshot.ticker,
            pattern_id=pattern_id,
            event_type=event_type,
            event_date=wall_timestamp.astimezone(EASTERN).date(),
            gate_values=gate_values,
            data_timestamp=_snapshot_data_clock(snapshot),
            wall_timestamp=wall_timestamp,
            decision_price=snapshot.decision_price,
            event_payload={"reason": reason},
        )

    def _record_once_daily(
        self,
        *,
        ticker: str,
        pattern_id: str,
        event_type: str,
        event_date: date,
        gate_values: Optional[Mapping[str, Any]] = None,
        event_payload: Optional[Mapping[str, Any]] = None,
        data_timestamp: Optional[datetime] = None,
        wall_timestamp: Optional[datetime] = None,
        decision_price: Optional[float] = None,
        broker_order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        fill_price: Optional[float] = None,
        fill_qty: Optional[float] = None,
        lineage_hash: Optional[str] = None,
    ) -> Optional[PaperExecutionEvent]:
        if event_type in {"candidate_confirmed", "poison_blocked", "parabolic_skipped"}:
            key = (ticker.upper(), pattern_id.upper(), event_type, event_date)
            if key in self._recorded_daily_events:
                return None
            self._recorded_daily_events.add(key)
        return self._events.record(
            ticker=ticker,
            pattern_id=pattern_id,
            event_type=event_type,
            event_date=event_date,
            gate_values=gate_values,
            event_payload=event_payload,
            data_timestamp=data_timestamp,
            wall_timestamp=wall_timestamp,
            decision_price=decision_price,
            broker_order_id=broker_order_id,
            client_order_id=client_order_id,
            fill_price=fill_price,
            fill_qty=fill_qty,
            lineage_hash=lineage_hash,
        )


def compute_shared_intraday_math(
    context: PremarketContext,
    snapshot: PolygonSnapshotTicker,
    *,
    trading_date: Optional[date] = None,
) -> Optional[SharedIntradayMath]:
    if (
        context.prior_close <= 0
        or context.avg20_volume <= 0
        or snapshot.day_open is None
        or snapshot.day_volume is None
    ):
        return None
    data_clock = _snapshot_data_clock(snapshot)
    if data_clock is None:
        return None
    data_et = data_clock.astimezone(EASTERN)
    if trading_date is not None and data_et.date() != trading_date:
        return None
    market_open_date = trading_date or data_et.date()
    market_open = datetime.combine(market_open_date, time(9, 30), EASTERN)
    data_elapsed_min = max(
        0.0,
        (data_et - market_open).total_seconds() / 60.0,
    )
    projected_vol = snapshot.day_volume * 390.0 / max(data_elapsed_min, 1.0)
    gap = (snapshot.day_open / context.prior_close) - 1.0
    vol_ratio = projected_vol / context.avg20_volume
    decision_price = snapshot.decision_price
    chase = (
        (decision_price / snapshot.day_open) - 1.0
        if decision_price is not None and snapshot.day_open > 0
        else None
    )
    return SharedIntradayMath(
        gap=gap,
        data_elapsed_min=data_elapsed_min,
        projected_vol=projected_vol,
        vol_ratio=vol_ratio,
        chase=chase,
        data_clock=data_clock,
    )


class PremarketContextBuilder:
    """Build and persist a current-date premarket context artifact."""

    def __init__(
        self,
        polygon_adapter: PolygonAdapter,
        *,
        lookback_sessions: int = 252,
    ) -> None:
        self._polygon = polygon_adapter
        self._lookback_sessions = lookback_sessions

    def build(
        self,
        *,
        context_date: date,
        output_path: str | Path,
    ) -> Dict[str, PremarketContext]:
        rows_by_ticker: Dict[str, List[PolygonGroupedDailyBar]] = {}
        for session_date in _prior_weekdays(context_date, self._lookback_sessions + 21):
            resp = self._polygon.get_grouped_daily_aggs(session_date.isoformat())
            if not resp.ok:
                raise RuntimeError(
                    f"Polygon grouped daily context fetch failed for {session_date}: "
                    f"{_provider_error_payload(resp.error)}"
                )
            for row in resp.data or []:
                rows_by_ticker.setdefault(row.ticker.upper(), []).append(row)
        contexts: Dict[str, PremarketContext] = {}
        for ticker, rows in rows_by_ticker.items():
            rows = sorted(rows, key=lambda r: r.timestamp)
            context = _context_from_daily_rows(ticker, context_date, rows)
            if context is not None:
                contexts[ticker] = context
        spy_return = contexts.get("SPY").prev_day_return if contexts.get("SPY") else None
        if spy_return is not None:
            contexts = {
                ticker: PremarketContext(
                    ticker=context.ticker,
                    context_date=context.context_date,
                    prior_close=context.prior_close,
                    max_prior_252_closes=context.max_prior_252_closes,
                    avg20_volume=context.avg20_volume,
                    mom20=context.mom20,
                    off_low252=context.off_low252,
                    sigma20=context.sigma20,
                    prev_day_return=context.prev_day_return,
                    prev_day_green=context.prev_day_green,
                    spy_prior_day_return=spy_return,
                )
                for ticker, context in contexts.items()
            }
        save_premarket_context_artifact(output_path, context_date, contexts)
        return contexts


def save_premarket_context_artifact(
    path: str | Path,
    context_date: date,
    contexts: Mapping[str, PremarketContext],
) -> None:
    payload = {
        "artifact_version": CONTEXT_ARTIFACT_VERSION,
        "context_date": context_date.isoformat(),
        "ticker_count": len(contexts),
        "contexts": {
            ticker.upper(): context.to_json()
            for ticker, context in sorted(contexts.items())
        },
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_json_dumps(payload))


def load_premarket_context_artifact(
    path: str | Path,
    *,
    expected_date: date,
) -> Dict[str, PremarketContext]:
    payload = json.loads(Path(path).read_text())
    if payload.get("artifact_version") != CONTEXT_ARTIFACT_VERSION:
        raise ValueError("premarket context artifact version mismatch")
    context_date = date.fromisoformat(str(payload.get("context_date")))
    if context_date != expected_date:
        raise ValueError(
            "premarket context artifact date mismatch: "
            f"expected {expected_date.isoformat()}, found {context_date.isoformat()}"
        )
    contexts = payload.get("contexts") or {}
    return {
        ticker.upper(): PremarketContext.from_json(row)
        for ticker, row in contexts.items()
    }


def deterministic_entry_client_order_id(pattern_id: str, ticker: str, event_date: date) -> str:
    return f"{pattern_id.upper()}_{ticker.upper()}_{event_date.isoformat()}"


def parse_entry_client_order_id(client_order_id: Optional[str]) -> Optional[ParsedEntryClientOrderId]:
    if not client_order_id:
        return None
    parts = client_order_id.rsplit("_", 2)
    if len(parts) != 3:
        return None
    pattern_id, ticker, raw_date = parts
    if not pattern_id or not ticker:
        return None
    try:
        entry_date = date.fromisoformat(raw_date)
    except ValueError:
        return None
    return ParsedEntryClientOrderId(
        pattern_id=pattern_id.upper(),
        ticker=ticker.upper(),
        entry_date=entry_date,
    )


def validate_paper_base_url(base_url: str, *, confirm_live_trade: bool = False) -> None:
    if "paper-api" not in base_url and not confirm_live_trade:
        raise ValueError(
            "Refusing to start paper execution unless Alpaca base_url contains "
            "'paper-api'. Pass --confirm-live-trade only for an explicit future live path."
        )


def _context_from_daily_rows(
    ticker: str,
    context_date: date,
    rows: Sequence[PolygonGroupedDailyBar],
) -> Optional[PremarketContext]:
    if len(rows) < 20:
        return None
    closes = [row.close for row in rows if row.close is not None and row.close > 0]
    volumes = [row.volume for row in rows if row.volume is not None and row.volume >= 0]
    lows = [row.low for row in rows if row.low is not None and row.low > 0]
    if len(closes) < 20 or len(volumes) < 20:
        return None
    prior_close = closes[-1]
    max_prior = max(closes[-252:]) if closes[-252:] else prior_close
    avg20_volume = sum(volumes[-20:]) / 20.0
    mom20 = (prior_close / closes[-21] - 1.0) if len(closes) >= 21 and closes[-21] > 0 else None
    low252 = min(lows[-252:]) if lows[-252:] else None
    off_low252 = (prior_close / low252 - 1.0) if low252 and low252 > 0 else None
    returns = [
        closes[idx] / closes[idx - 1] - 1.0
        for idx in range(1, len(closes))
        if closes[idx - 1] > 0
    ]
    trailing_returns = returns[-20:]
    sigma20 = _stddev(trailing_returns) if len(trailing_returns) >= 2 else None
    prev_day_return = returns[-1] if returns else None
    prev_day_green = prev_day_return is not None and prev_day_return > 0
    if max_prior <= 0 or avg20_volume <= 0:
        return None
    return PremarketContext(
        ticker=ticker.upper(),
        context_date=context_date,
        prior_close=prior_close,
        max_prior_252_closes=max_prior,
        avg20_volume=avg20_volume,
        mom20=mom20,
        off_low252=off_low252,
        sigma20=sigma20,
        prev_day_return=prev_day_return,
        prev_day_green=prev_day_green,
    )


def _prior_weekdays(context_date: date, count: int) -> List[date]:
    days: List[date] = []
    cursor = context_date - timedelta(days=1)
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return list(reversed(days))


def _snapshot_data_clock(snapshot: PolygonSnapshotTicker) -> Optional[datetime]:
    if snapshot.minute_timestamp is None:
        return None
    ts = snapshot.minute_timestamp
    if ts > 10_000_000_000:
        ts = ts / 1000.0
    return datetime.fromtimestamp(ts, timezone.utc)


def _universe_hygiene_skip(snapshot: PolygonSnapshotTicker) -> Optional[str]:
    ticker = snapshot.ticker.upper()
    price = snapshot.decision_price
    if price is None or price < 1:
        return "price_below_1"
    if len(ticker) >= 5 and ticker[4] in {"W", "U", "R"}:
        return "fifth_char_suffix_excluded"
    return None


def _decision_event_type(reason: str) -> str:
    if reason in {"poison_blocked", "parabolic_skipped"}:
        return reason
    return "candidate_confirmed" if reason == "candidate_confirmed" else "gate_skipped"


def _exit_due(position: ManagedPosition, wall_timestamp: datetime) -> bool:
    wall_et = wall_timestamp.astimezone(EASTERN)
    entry_date = position.entry_date
    if position.exit_policy == ExitPolicy.SAME_DAY_CLOSE_1555:
        return (
            wall_et.date() == entry_date and wall_et.time() >= time(15, 55)
        ) or wall_et.date() > entry_date
    if position.exit_policy == ExitPolicy.NEXT_OPEN_0931:
        return wall_et.date() > entry_date and wall_et.time() >= time(9, 31)
    if position.exit_policy == ExitPolicy.HORIZON_N_DAYS and position.horizon_days is not None:
        return wall_et.date() >= entry_date + timedelta(days=position.horizon_days)
    return False


def _pattern_from_client_order_id(client_order_id: Optional[str]) -> Optional[str]:
    if not client_order_id:
        return None
    normalized = (
        client_order_id[5:]
        if client_order_id.startswith("exit_")
        else client_order_id
    )
    parsed = parse_entry_client_order_id(normalized)
    if parsed is not None:
        return parsed.pattern_id
    return normalized.split("_", 1)[0].upper()


def _matching_closed_entry_order(
    position: AlpacaPosition,
    closed_orders: Sequence[AlpacaOrder],
) -> tuple[Optional[AlpacaOrder], Optional[ParsedEntryClientOrderId]]:
    ticker = position.symbol.upper()
    candidates: List[tuple[date, AlpacaOrder, ParsedEntryClientOrderId]] = []
    for order in closed_orders:
        parsed = parse_entry_client_order_id(order.client_order_id)
        if parsed is None:
            continue
        if parsed.ticker != ticker or order.symbol.upper() != ticker:
            continue
        if order.side.lower() != "buy":
            continue
        candidates.append((parsed.entry_date, order, parsed))
    if not candidates:
        return None, None
    _, order, parsed = max(candidates, key=lambda item: item[0])
    return order, parsed


def _raise_on_auth(error: Optional[ProviderError]) -> None:
    if error is not None and error.error_type == "auth":
        raise FatalBrokerAuthError(error.message)


def _provider_error_payload(error: Optional[ProviderError]) -> Dict[str, Any]:
    if error is None:
        return {}
    return {
        "provider": error.provider,
        "endpoint": error.endpoint,
        "status_code": error.status_code,
        "error_type": error.error_type,
        "message": error.message,
        "retryable": error.retryable,
    }


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _stddev(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2:
        return None
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return math.sqrt(variance)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
