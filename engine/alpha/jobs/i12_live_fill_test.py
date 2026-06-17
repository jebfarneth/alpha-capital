"""Read-only Stage-0 live fill-test for the I12 ML-ranked book."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from alpha.data.alpaca import AlpacaAdapter, AlpacaQuote, AlpacaStockSnapshot
from alpha.data.contracts import stable_hash, utcnow
from alpha.data.polygon import PolygonSnapshotTicker
from alpha.db.models import (
    FeatureSnapshot,
    I12FillLog,
    MLModelRegistry,
    SignalMLScore,
    SignalRegistry,
)
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.paper_execution import (
    PremarketContext,
    compute_shared_intraday_math,
    i12_entry_gate,
)
from alpha.market_calendar import (
    is_us_equity_session,
    next_us_equity_session,
    us_equity_session_open_timestamp,
)
from alpha.ml.inference import score_signal_shadow


I12_PATTERN_ID = "I12"
JOB_NAME = "i12_live_fill_test_stage0"
FEATURE_MANIFEST_VERSION = "i12_live_stage0_v1"
DEFAULT_TOP_K = 10
DEFAULT_INTENDED_ORDER_USD = 250.0
DEFAULT_MAX_SPREAD_BPS = 200.0
EASTERN = ZoneInfo("America/New_York")
HALT_CONDITIONS = frozenset({"H", "T1", "T2", "T5", "HALT", "HALTED"})
LIVE_I12_ALLOWED_FEATURES = frozenset(
    {
        "mom20",
        "off_low252",
        "sigma20",
        "distance_from_max252",
        "drawdown_from_max252",
        "gap",
        "prev_day_return",
        "prev_day_green",
        "spy_prior_day_return",
        "projected_volume_ratio_at_confirmation",
    }
)
LEAKY_ENTRY_TOKENS = (
    "full_day",
    "close_price",
    "day_close",
    "forward",
    "future",
    "label",
    "outcome",
    "ret_",
    "return_from_entry",
    "mfe",
    "mae",
    "exit",
)


@dataclass(frozen=True)
class I12LiveFillConfig:
    """Stage-0 read-only policy."""

    model_id: str | None
    allow_latest_model: bool = False
    top_k: int = DEFAULT_TOP_K
    intended_order_usd: float = DEFAULT_INTENDED_ORDER_USD
    max_spread_bps: float = DEFAULT_MAX_SPREAD_BPS
    feed: str = "iex"
    require_market_open: bool = True

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.intended_order_usd <= 0:
            raise ValueError("intended_order_usd must be positive")
        if self.max_spread_bps <= 0:
            raise ValueError("max_spread_bps must be positive")
        if not self.model_id and not self.allow_latest_model:
            raise ValueError(
                "Stage-0 live scoring requires --model-id, or explicit "
                "--allow-latest-model for operator-selected latest non-rejected I12 model"
            )


@dataclass(frozen=True)
class LiveFire:
    ticker: str
    signal: SignalRegistry
    score: SignalMLScore
    feature_payload: dict[str, Any]
    gate_values: dict[str, Any]
    snapshot: AlpacaStockSnapshot


class I12LiveFillTestJob(BaseJob):
    """Detect, score, select, liquidity-filter, and log I12 intended trades."""

    def __init__(
        self,
        *,
        session: Session,
        alpaca_adapter: AlpacaAdapter,
        contexts: Mapping[str, PremarketContext],
        config: I12LiveFillConfig,
        asof: datetime | None = None,
        snapshots: Mapping[str, AlpacaStockSnapshot] | None = None,
    ) -> None:
        self._session = session
        self._alpaca = alpaca_adapter
        self._contexts = {ticker.upper(): context for ticker, context in contexts.items()}
        self._config = config
        self._asof = _aware_utc(asof or utcnow())
        self._snapshots = {ticker.upper(): row for ticker, row in (snapshots or {}).items()}

    @property
    def job_name(self) -> str:
        return JOB_NAME

    @property
    def job_type(self) -> str:
        return "live_fill_test"

    def run(self, ctx: JobContext) -> JobResult:
        del ctx
        decision_date = self._asof.astimezone(EASTERN).date()
        if not is_us_equity_session(decision_date):
            return JobResult(
                status="finished",
                metrics={
                    "status": "non_trading_day_noop",
                    "decision_date": decision_date.isoformat(),
                },
            )
        if self._config.require_market_open:
            clock = self._alpaca.get_clock()
            if not clock.ok or clock.data is None:
                return JobResult(
                    status="failed",
                    errors=[{"reason": "alpaca_clock_unavailable", "error": str(clock.error)}],
                )
            if not clock.data.is_open:
                return JobResult(
                    status="finished",
                    metrics={
                        "status": "market_closed_noop",
                        "decision_date": decision_date.isoformat(),
                    },
                )

        model = select_i12_model(
            self._session,
            model_id=self._config.model_id,
            allow_latest_model=self._config.allow_latest_model,
        )
        snapshots = self._snapshots or self._fetch_snapshots()
        fires = self._detect_and_score(model, snapshots)
        selected = sorted(
            [
                fire for fire in fires
                if fire.score.score_source == "model_shadow"
                and fire.score.score is not None
                and math.isfinite(float(fire.score.score))
            ],
            key=lambda fire: float(fire.score.score),
            reverse=True,
        )[: self._config.top_k]
        logged = [self._log_intended_trade(model, fire) for fire in selected]
        self._session.commit()
        return JobResult(
            status="finished",
            metrics={
                "decision_date": decision_date.isoformat(),
                "contexts": len(self._contexts),
                "snapshots": len(snapshots),
                "fires": len(fires),
                "model_scored_fires": len([f for f in fires if f.score.score_source == "model_shadow"]),
                "selected_top_k": len(selected),
                "logged_intended_trades": len(logged),
                "liquidity_skips": len([row for row in logged if row.skipped_reason != "none"]),
                "model_id": model.model_id,
                "read_only": True,
            },
        )

    def capture_exit_quotes(self, *, asof: datetime | None = None) -> dict[str, Any]:
        return capture_i12_exit_quotes(
            self._session,
            self._alpaca,
            feed=self._config.feed,
            asof=asof,
        )

    def gate0_report(self, *, decision_date: date | None = None) -> dict[str, Any]:
        return i12_gate0_report(
            self._session,
            decision_date=decision_date,
            max_spread_bps=self._config.max_spread_bps,
            intended_order_usd=self._config.intended_order_usd,
        )

    def _fetch_snapshots(self) -> dict[str, AlpacaStockSnapshot]:
        snapshots: dict[str, AlpacaStockSnapshot] = {}
        tickers = sorted(self._contexts)
        for batch in _chunks(tickers, 200):
            resp = self._alpaca.get_stock_snapshots(batch, feed=self._config.feed)
            if not resp.ok:
                continue
            snapshots.update(resp.data or {})
        return snapshots

    def _detect_and_score(
        self,
        model: MLModelRegistry,
        snapshots: Mapping[str, AlpacaStockSnapshot],
    ) -> list[LiveFire]:
        fires: list[LiveFire] = []
        for ticker, context in self._contexts.items():
            snapshot = snapshots.get(ticker)
            if snapshot is None:
                continue
            polygon_like = _polygon_snapshot_from_alpaca(snapshot)
            shared = compute_shared_intraday_math(
                context,
                polygon_like,
                trading_date=context.context_date,
            )
            if shared is None:
                continue
            decision = i12_entry_gate(context, polygon_like, shared)
            if not decision.enter:
                continue
            feature_payload = build_i12_live_feature_payload(context, decision.gate_values)
            assert_i12_live_feature_payload_leakage_clean(feature_payload)
            signal = self._upsert_signal(
                ticker=ticker,
                context=context,
                feature_payload=feature_payload,
                gate_values=decision.gate_values,
                decision_ts=shared.data_clock,
            )
            score = score_signal_shadow(
                self._session,
                signal_id=signal.signal_id,
                model_id=model.model_id,
                score_status="stage0_read_only",
            )
            fires.append(
                LiveFire(
                    ticker=ticker,
                    signal=signal,
                    score=score,
                    feature_payload=feature_payload,
                    gate_values=dict(decision.gate_values),
                    snapshot=snapshot,
                )
            )
        return fires

    def _upsert_signal(
        self,
        *,
        ticker: str,
        context: PremarketContext,
        feature_payload: dict[str, Any],
        gate_values: Mapping[str, Any],
        decision_ts: datetime,
    ) -> SignalRegistry:
        identity_hash = stable_hash(
            {
                "pattern_id": I12_PATTERN_ID,
                "ticker": ticker.upper(),
                "decision_ts": decision_ts.isoformat(),
                "gate": "i12_live_stage0_read_only",
            }
        )
        existing = (
            self._session.query(SignalRegistry)
            .filter(
                SignalRegistry.pattern_id == I12_PATTERN_ID,
                SignalRegistry.ticker == ticker.upper(),
                SignalRegistry.signal_identity_hash == identity_hash,
            )
            .one_or_none()
        )
        if existing is not None:
            return existing
        feature_snapshot_id = f"i12-stage0-fs-{identity_hash[:32]}"
        feature_json = _json_dumps(feature_payload)
        feature = FeatureSnapshot(
            feature_snapshot_id=feature_snapshot_id,
            pattern_id=I12_PATTERN_ID,
            ticker=ticker.upper(),
            asof_timestamp=decision_ts,
            feature_manifest_version=FEATURE_MANIFEST_VERSION,
            feature_json=feature_json,
            feature_hash=stable_hash(feature_json),
            data_lineage_ids="[]",
            fidelity_tier="live_stage0_read_only",
            point_in_time_passed=True,
            lookahead_guard_passed=True,
            input_hashes=_json_dumps({"gate_values": dict(gate_values)}),
            output_hash=identity_hash,
        )
        self._session.add(feature)
        signal = SignalRegistry(
            signal_id=f"i12-stage0-sig-{identity_hash[:32]}",
            pattern_id=I12_PATTERN_ID,
            ticker=ticker.upper(),
            direction="long",
            signal_timestamp=decision_ts,
            raw_signal_strength=float(
                gate_values.get("projected_volume_ratio_at_confirmation")
                or gate_values.get("vol_ratio")
                or 0.0
            ),
            raw_expected_edge=0.0,
            signal_horizon="1d",
            thesis_category="capitulation_volume_bounce",
            route_class="stage0_read_only",
            fidelity_tier="live_stage0_read_only",
            data_confidence=1.0,
            feature_snapshot_id=feature_snapshot_id,
            signal_status="active",
            trading_date=context.context_date.isoformat(),
            next_execution_session=context.context_date.isoformat(),
            detector_version=FEATURE_MANIFEST_VERSION,
            point_in_time_passed=True,
            lookahead_guard_passed=True,
            data_lineage_ids="[]",
            signal_identity_hash=identity_hash,
            intended_entry_price=_optional_float(gate_values.get("decision_price")),
        )
        self._session.add(signal)
        self._session.flush()
        return signal

    def _log_intended_trade(
        self,
        model: MLModelRegistry,
        fire: LiveFire,
    ) -> I12FillLog:
        quote = fire.snapshot.latest_quote
        liquidity = evaluate_quote_liquidity(
            quote,
            intended_order_usd=self._config.intended_order_usd,
            max_spread_bps=self._config.max_spread_bps,
        )
        decision_ts = _stored_utc(fire.signal.signal_timestamp)
        content_hash = stable_hash(
            {
                "signal_id": fire.signal.signal_id,
                "score_id": fire.score.score_id,
                "intended_order_usd": self._config.intended_order_usd,
                "stage": "i12_live_fill_test_stage0",
            }
        )
        existing = (
            self._session.query(I12FillLog)
            .filter(I12FillLog.content_hash == content_hash)
            .one_or_none()
        )
        if existing is not None:
            return existing
        row = I12FillLog(
            signal_id=fire.signal.signal_id,
            score_id=fire.score.score_id,
            model_id=model.model_id,
            ticker=fire.ticker,
            decision_date=decision_ts.astimezone(EASTERN).date(),
            decision_ts=decision_ts,
            exit_capture_due_ts=_next_session_open_after(decision_ts),
            ml_score=fire.score.score,
            score_source=fire.score.score_source,
            score_status=fire.score.score_status,
            fallback_reason=fire.score.fallback_reason,
            projected_vol_ratio=_optional_float(
                fire.feature_payload.get("projected_volume_ratio_at_confirmation")
            ),
            gap=_optional_float(fire.feature_payload.get("gap")),
            off_52w_high=_optional_float(fire.feature_payload.get("distance_from_max252")),
            bid=liquidity["bid"],
            ask=liquidity["ask"],
            spread_bps=liquidity["spread_bps"],
            top_of_book_size=liquidity["top_of_book_size"],
            intended_order_usd=self._config.intended_order_usd,
            size_sufficient=liquidity["size_sufficient"],
            halted=liquidity["halted"],
            skipped_reason=liquidity["skipped_reason"],
            feature_json=_json_dumps(fire.feature_payload),
            gate_values_json=_json_dumps(fire.gate_values),
            quote_json=_json_dumps(_quote_payload(quote)) if quote else None,
            content_hash=content_hash,
        )
        self._session.add(row)
        self._session.flush()
        return row


def capture_i12_exit_quotes(
    session: Session,
    alpaca_adapter: AlpacaAdapter,
    *,
    feed: str = "iex",
    asof: datetime | None = None,
) -> dict[str, Any]:
    asof_ts = _aware_utc(asof or utcnow())
    tickers = [
        row.ticker
        for row in session.query(I12FillLog)
        .filter(
            I12FillLog.exit_bid.is_(None),
            I12FillLog.skipped_reason == "none",
        )
        .all()
        if _exit_quote_due(row, asof_ts)
    ]
    if not tickers:
        return {"exit_quote_updates": 0}
    quotes_resp = alpaca_adapter.get_latest_quotes(sorted(set(tickers)), feed=feed)
    if not quotes_resp.ok:
        return {
            "exit_quote_updates": 0,
            "error": str(quotes_resp.error),
        }
    quotes = quotes_resp.data or {}
    updated = 0
    for row in (
        session.query(I12FillLog)
        .filter(
            I12FillLog.exit_bid.is_(None),
            I12FillLog.skipped_reason == "none",
            I12FillLog.ticker.in_(quotes.keys()),
        )
        .all()
    ):
        if not _exit_quote_due(row, asof_ts):
            continue
        quote = quotes.get(row.ticker.upper())
        if quote is None:
            continue
        row.exit_bid = quote.bid_price
        row.exit_ask = quote.ask_price
        row.exit_quote_ts = _parse_provider_ts(quote.timestamp) or asof_ts
        row.exit_quote_json = _json_dumps(_quote_payload(quote))
        if row.ask and row.ask > 0 and row.exit_bid is not None:
            row.modeled_return = (row.exit_bid / row.ask) - 1.0
        updated += 1
    session.commit()
    return {"exit_quote_updates": updated}


def i12_gate0_report(
    session: Session,
    *,
    decision_date: date | None = None,
    max_spread_bps: float = DEFAULT_MAX_SPREAD_BPS,
    intended_order_usd: float = DEFAULT_INTENDED_ORDER_USD,
) -> dict[str, Any]:
    query = session.query(I12FillLog)
    if decision_date is not None:
        query = query.filter(I12FillLog.decision_date == decision_date)
    rows = query.all()
    total = len(rows)
    spread_ok = len(
        [
            row
            for row in rows
            if row.spread_bps is not None and row.spread_bps <= max_spread_bps
        ]
    )
    size_ok = len([row for row in rows if row.size_sufficient is True])
    tradeable = len([row for row in rows if row.skipped_reason == "none"])
    return {
        "rows": total,
        "spread_ok": spread_ok,
        "size_ok": size_ok,
        "tradeable": tradeable,
        "spread_ok_rate": spread_ok / total if total else None,
        "size_ok_rate": size_ok / total if total else None,
        "tradeable_rate": tradeable / total if total else None,
        "passed": bool(total and tradeable / total >= 0.5),
        "max_spread_bps": max_spread_bps,
        "intended_order_usd": intended_order_usd,
    }


def select_i12_model(
    session: Session,
    *,
    model_id: str | None,
    allow_latest_model: bool,
) -> MLModelRegistry:
    if model_id:
        model = session.get(MLModelRegistry, model_id)
        if model is None:
            raise RuntimeError(f"I12 model {model_id!r} not found in ml_model_registry")
        if model.pattern_id != I12_PATTERN_ID:
            raise RuntimeError(f"model {model_id!r} is pattern {model.pattern_id}, not I12")
        if model.status == "rejected":
            raise RuntimeError(f"model {model_id!r} is rejected")
        return model
    if not allow_latest_model:
        raise RuntimeError("model_id required unless allow_latest_model is true")
    model = (
        session.query(MLModelRegistry)
        .filter(
            MLModelRegistry.pattern_id == I12_PATTERN_ID,
            MLModelRegistry.status != "rejected",
        )
        .order_by(MLModelRegistry.created_at.desc())
        .first()
    )
    if model is None:
        raise RuntimeError("no non-rejected I12 model found in ml_model_registry")
    return model


def build_i12_live_feature_payload(
    context: PremarketContext,
    gate_values: Mapping[str, Any],
) -> dict[str, Any]:
    distance = _optional_float(gate_values.get("distance_from_max252"))
    projected_ratio = _optional_float(
        gate_values.get("projected_volume_ratio_at_confirmation")
        or gate_values.get("vol_ratio")
    )
    payload = {
        "mom20": context.mom20,
        "off_low252": context.off_low252,
        "sigma20": context.sigma20,
        "distance_from_max252": distance,
        "drawdown_from_max252": distance,
        "gap": _optional_float(gate_values.get("gap")),
        "prev_day_return": context.prev_day_return,
        "prev_day_green": context.prev_day_green,
        "spy_prior_day_return": context.spy_prior_day_return,
        "projected_volume_ratio_at_confirmation": projected_ratio,
        "projected_volume_at_confirmation": _optional_float(
            gate_values.get("projected_volume_at_confirmation")
            or gate_values.get("projected_vol")
        ),
        "ticker": context.ticker,
        "trading_date": context.context_date.isoformat(),
        "leakage_contract": {
            "decision_basis": "live_intraday_projected_volume",
            "uses_full_day_volume": False,
            "uses_forward_bars": False,
        },
    }
    return payload


def assert_i12_live_feature_payload_leakage_clean(payload: Mapping[str, Any]) -> None:
    feature_keys = set(payload).intersection(LIVE_I12_ALLOWED_FEATURES)
    missing = LIVE_I12_ALLOWED_FEATURES - feature_keys
    if missing:
        raise RuntimeError(f"live I12 feature payload missing ranker fields: {sorted(missing)}")
    allowed_contract_paths = {
        "leakage_contract.uses_forward_bars",
        "leakage_contract.uses_full_day_volume",
    }
    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                lower = child_path.lower()
                if child_path not in allowed_contract_paths:
                    if any(token in lower for token in LEAKY_ENTRY_TOKENS):
                        raise RuntimeError(f"leaky live I12 feature path: {child_path}")
                visit(child, child_path)
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                visit(child, f"{path}[{idx}]")
    visit(payload)
    contract = payload.get("leakage_contract")
    if not isinstance(contract, dict) or contract.get("uses_full_day_volume") is not False:
        raise RuntimeError("live I12 payload must record uses_full_day_volume=false")


def evaluate_quote_liquidity(
    quote: AlpacaQuote | None,
    *,
    intended_order_usd: float,
    max_spread_bps: float,
) -> dict[str, Any]:
    if quote is None:
        return {
            "bid": None,
            "ask": None,
            "spread_bps": None,
            "top_of_book_size": None,
            "size_sufficient": False,
            "halted": None,
            "skipped_reason": "quote_unavailable",
        }
    bid = quote.bid_price
    ask = quote.ask_price
    ask_size = quote.ask_size
    halted = _quote_halted(quote)
    spread_bps = None
    if bid is not None and ask is not None and bid > 0 and ask >= bid:
        spread_bps = (ask - bid) / ((ask + bid) / 2.0) * 10_000.0
    top_of_book_size = ask * ask_size if ask is not None and ask_size is not None else None
    size_sufficient = (
        top_of_book_size is not None
        and top_of_book_size >= intended_order_usd
    )
    skipped_reason = "none"
    if halted:
        skipped_reason = "halt"
    elif spread_bps is None:
        skipped_reason = "spread"
    elif spread_bps > max_spread_bps:
        skipped_reason = "spread"
    elif not size_sufficient:
        skipped_reason = "size"
    return {
        "bid": bid,
        "ask": ask,
        "spread_bps": spread_bps,
        "top_of_book_size": top_of_book_size,
        "size_sufficient": size_sufficient,
        "halted": halted,
        "skipped_reason": skipped_reason,
    }


def _polygon_snapshot_from_alpaca(snapshot: AlpacaStockSnapshot) -> PolygonSnapshotTicker:
    ts = (
        _parse_provider_ts(snapshot.latest_trade_timestamp)
        or _parse_provider_ts(snapshot.minute_timestamp)
        or (
            _parse_provider_ts(snapshot.latest_quote.timestamp)
            if snapshot.latest_quote is not None else None
        )
    )
    timestamp_ms = int(ts.timestamp() * 1000) if ts is not None else None
    return PolygonSnapshotTicker(
        ticker=snapshot.symbol,
        day_open=snapshot.daily_open,
        day_high=snapshot.daily_high,
        day_low=snapshot.daily_low,
        day_close=None,
        day_volume=snapshot.daily_volume,
        minute_timestamp=timestamp_ms,
        minute_open=snapshot.minute_open,
        minute_high=snapshot.minute_high,
        minute_low=snapshot.minute_low,
        minute_close=snapshot.minute_close,
        minute_volume=snapshot.minute_volume,
        last_trade_price=snapshot.latest_trade_price,
        raw=snapshot.raw,
    )


def _quote_halted(quote: AlpacaQuote) -> bool:
    return bool({condition.upper() for condition in quote.conditions}.intersection(HALT_CONDITIONS))


def _quote_payload(quote: AlpacaQuote | None) -> dict[str, Any]:
    if quote is None:
        return {}
    return {
        "symbol": quote.symbol,
        "bid_price": quote.bid_price,
        "ask_price": quote.ask_price,
        "bid_size": quote.bid_size,
        "ask_size": quote.ask_size,
        "timestamp": quote.timestamp,
        "conditions": quote.conditions,
        "tape": quote.tape,
    }


def _exit_quote_due(row: I12FillLog, asof_ts: datetime) -> bool:
    due_ts = row.exit_capture_due_ts
    if due_ts is None:
        return row.decision_date < asof_ts.astimezone(EASTERN).date()
    return _stored_utc(due_ts) <= asof_ts


def _next_session_open_after(decision_ts: datetime) -> datetime:
    decision_day = _stored_utc(decision_ts).astimezone(EASTERN).date()
    exit_session = next_us_equity_session(decision_day + timedelta(days=1))
    return us_equity_session_open_timestamp(exit_session)


def _parse_provider_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _aware_utc(value)
    raw = str(value)
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _stored_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _chunks(values: Sequence[str], size: int) -> list[list[str]]:
    return [list(values[idx: idx + size]) for idx in range(0, len(values), size)]
