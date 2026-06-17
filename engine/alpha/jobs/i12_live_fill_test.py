"""Read-only Stage-0 live fill-test for the I12 ML-ranked book."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from sqlalchemy import or_
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
    us_equity_session_close_time,
)
from alpha.ml.inference import score_signal_shadow
from alpha.ml.model_features import feature_schema_hash


I12_PATTERN_ID = "I12"
JOB_NAME = "i12_live_fill_test_stage0"
FEATURE_MANIFEST_VERSION = "i12_live_stage0_v1"
DEFAULT_TOP_K = 10
DEFAULT_INTENDED_ORDER_USD = 250.0
DEFAULT_MAX_SPREAD_BPS = 200.0
ALPACA_QUOTE_SIZE_BASIS = "shares_post_2025_11_03"
SUPPORTED_ALPACA_QUOTE_SIZE_BASES = frozenset({ALPACA_QUOTE_SIZE_BASIS})
DEFAULT_MAX_QUOTE_AGE_SECONDS = 60.0
DEFAULT_MAX_SNAPSHOT_AGE_SECONDS = 120.0
DEFAULT_MIN_CONTEXT_COUNT = 1
DEFAULT_MIN_INTENDED_COUNT = 1
DEFAULT_MIN_SNAPSHOT_OK_RATE = 0.95
DEFAULT_MAX_SNAPSHOT_ERROR_OR_MISSING_RATE = 0.05
DEFAULT_MIN_SCORE_MODEL_OK_RATE = 0.95
DEFAULT_MIN_QUOTE_OK_RATE = 0.95
DEFAULT_MIN_EXIT_QUOTE_OK_RATE = 0.95
DEFAULT_MIN_GATE0_INTENDED_COUNT = 20
DEFAULT_MIN_GATE0_DISTINCT_TRADING_DAYS = 3
DEFAULT_MIN_GATE0_TRADEABLE_RATE = 0.70
EASTERN = ZoneInfo("America/New_York")
HALT_CONDITIONS = frozenset({"H", "T1", "T2", "T5", "HALT", "HALTED"})
PROMOTION_FEEDS = frozenset({"sip"})
PROMOTABLE_MODEL_STATUSES = frozenset({"shadow"})
EXIT_CAPTURE_TERMINAL_STATUSES = frozenset(
    {"ok", "missing", "stale", "condition_halt_inferred", "error", "skipped_cash"}
)
EXIT_CAPTURE_TRADEABLE_TERMINAL_STATUSES = frozenset(
    {"ok", "missing", "stale", "condition_halt_inferred", "error"}
)
EXPECTED_I12_LIVE_FEATURES = (
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
)
LIVE_I12_ALLOWED_FEATURES = frozenset(EXPECTED_I12_LIVE_FEATURES)
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
    feed: str = "sip"
    max_quote_age_seconds: float = DEFAULT_MAX_QUOTE_AGE_SECONDS
    max_snapshot_age_seconds: float = DEFAULT_MAX_SNAPSHOT_AGE_SECONDS
    min_context_count: int = DEFAULT_MIN_CONTEXT_COUNT
    min_intended_count: int = DEFAULT_MIN_INTENDED_COUNT
    min_snapshot_ok_rate: float = DEFAULT_MIN_SNAPSHOT_OK_RATE
    max_snapshot_error_or_missing_rate: float = DEFAULT_MAX_SNAPSHOT_ERROR_OR_MISSING_RATE
    min_score_model_ok_rate: float = DEFAULT_MIN_SCORE_MODEL_OK_RATE
    min_quote_ok_rate: float = DEFAULT_MIN_QUOTE_OK_RATE
    min_exit_quote_ok_rate: float = DEFAULT_MIN_EXIT_QUOTE_OK_RATE
    context_artifact_hash: str | None = None
    require_market_open: bool = True

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.intended_order_usd <= 0:
            raise ValueError("intended_order_usd must be positive")
        if self.max_spread_bps <= 0:
            raise ValueError("max_spread_bps must be positive")
        if self.max_quote_age_seconds <= 0:
            raise ValueError("max_quote_age_seconds must be positive")
        if self.max_snapshot_age_seconds <= 0:
            raise ValueError("max_snapshot_age_seconds must be positive")
        if self.min_context_count < 0:
            raise ValueError("min_context_count must be >= 0")
        if self.min_intended_count < 0:
            raise ValueError("min_intended_count must be >= 0")
        for name in (
            "min_snapshot_ok_rate",
            "max_snapshot_error_or_missing_rate",
            "min_score_model_ok_rate",
            "min_quote_ok_rate",
            "min_exit_quote_ok_rate",
        ):
            value = getattr(self, name)
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not self.model_id and not self.allow_latest_model:
            raise ValueError(
                "Stage-0 live scoring requires --model-id, or explicit "
                "--allow-latest-model for operator-selected latest non-rejected I12 model"
            )


@dataclass(frozen=True)
class Stage0ModelContract:
    model: MLModelRegistry
    feature_names: tuple[str, ...]
    model_selection_mode: str
    promotable_run: bool
    non_promotable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class LiveFire:
    ticker: str
    signal: SignalRegistry
    score: SignalMLScore
    feature_payload: dict[str, Any]
    gate_values: dict[str, Any]
    snapshot: AlpacaStockSnapshot
    attempt: I12FillLog


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

        model_contract = select_i12_model(
            self._session,
            model_id=self._config.model_id,
            allow_latest_model=self._config.allow_latest_model,
            feed=self._config.feed,
        )
        run_config_hash = _stage0_run_config_hash(self._config, model_contract)
        session_minutes = _session_minutes(decision_date)
        half_day = session_minutes != 390
        projection_basis = (
            "regular_session_390m_projected_volume"
            if not half_day
            else "half_day_diagnostic_390m_projected_volume"
        )
        attempts = {
            ticker: self._attempt_row(
                ticker=ticker,
                context=context,
                model_contract=model_contract,
                decision_date=decision_date,
                run_config_hash=run_config_hash,
                half_day=half_day,
                session_minutes=session_minutes,
                projection_basis=projection_basis,
            )
            for ticker, context in self._contexts.items()
        }
        snapshots, snapshot_batch_errors = (
            (self._snapshots, {})
            if self._snapshots
            else self._fetch_snapshots()
        )
        fires = self._detect_and_score(
            model_contract,
            snapshots,
            attempts=attempts,
            snapshot_batch_errors=snapshot_batch_errors,
        )
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
        selected_ids = {fire.attempt.i12_fill_log_id for fire in selected}
        for fire in fires:
            if fire.attempt.i12_fill_log_id not in selected_ids and fire.attempt.selection_status != "intended":
                fire.attempt.selection_status = "not_selected"
                fire.attempt.skipped_reason = "not_selected"
        logged = [self._log_intended_trade(model_contract, fire) for fire in selected]
        self._session.commit()
        report = i12_gate0_report(
            self._session,
            decision_date=decision_date,
            asof=self._asof,
            max_spread_bps=self._config.max_spread_bps,
            intended_order_usd=self._config.intended_order_usd,
            min_context_count=self._config.min_context_count,
            min_intended_count=self._config.min_intended_count,
            min_snapshot_ok_rate=self._config.min_snapshot_ok_rate,
            max_snapshot_error_or_missing_rate=(
                self._config.max_snapshot_error_or_missing_rate
            ),
            min_score_model_ok_rate=self._config.min_score_model_ok_rate,
            min_quote_ok_rate=self._config.min_quote_ok_rate,
            min_exit_quote_ok_rate=self._config.min_exit_quote_ok_rate,
            min_gate0_intended_count=0,
            min_gate0_distinct_trading_days=0,
            min_gate0_tradeable_rate=0.0,
        )
        return JobResult(
            status="finished",
            metrics={
                "decision_date": decision_date.isoformat(),
                "context_count": report["context_count"],
                "snapshot_ok": report["snapshot_ok"],
                "snapshot_missing": report["snapshot_missing"],
                "snapshot_batch_errors": report["snapshot_batch_errors"],
                "fire_count": report["fire_count"],
                "score_model_ok": report["score_model_ok"],
                "score_fallback": report["score_fallback"],
                "score_failed": report["score_failed"],
                "selected_top_k": len(selected),
                "logged_intended_trades": len(logged),
                "skipped_as_cash_count": report["skipped_as_cash_count"],
                "model_id": model_contract.model.model_id,
                "model_selection_mode": model_contract.model_selection_mode,
                "promotable_run": model_contract.promotable_run,
                "feed": self._config.feed,
                "stage0_run_config_hash": run_config_hash,
                "half_day": half_day,
                "session_minutes": session_minutes,
                "coverage_gate_passed": report["coverage_gate_passed"],
                "read_only": True,
            },
        )

    def capture_exit_quotes(self, *, asof: datetime | None = None) -> dict[str, Any]:
        return capture_i12_exit_quotes(
            self._session,
            self._alpaca,
            feed=self._config.feed,
            asof=asof,
            max_quote_age_seconds=self._config.max_quote_age_seconds,
        )

    def gate0_report(
        self,
        *,
        decision_date: date | None = None,
        asof: datetime | None = None,
    ) -> dict[str, Any]:
        return i12_gate0_report(
            self._session,
            decision_date=decision_date,
            asof=asof,
            max_spread_bps=self._config.max_spread_bps,
            intended_order_usd=self._config.intended_order_usd,
            min_context_count=self._config.min_context_count,
            min_intended_count=self._config.min_intended_count,
            min_snapshot_ok_rate=self._config.min_snapshot_ok_rate,
            max_snapshot_error_or_missing_rate=(
                self._config.max_snapshot_error_or_missing_rate
            ),
            min_score_model_ok_rate=self._config.min_score_model_ok_rate,
            min_quote_ok_rate=self._config.min_quote_ok_rate,
            min_exit_quote_ok_rate=self._config.min_exit_quote_ok_rate,
            min_gate0_intended_count=0,
            min_gate0_distinct_trading_days=0,
            min_gate0_tradeable_rate=0.0,
        )

    def _fetch_snapshots(self) -> tuple[dict[str, AlpacaStockSnapshot], dict[str, str]]:
        snapshots: dict[str, AlpacaStockSnapshot] = {}
        errors: dict[str, str] = {}
        tickers = sorted(self._contexts)
        for batch in _chunks(tickers, 200):
            resp = self._alpaca.get_stock_snapshots(batch, feed=self._config.feed)
            if not resp.ok:
                message = str(resp.error)
                for ticker in batch:
                    errors[ticker.upper()] = message
                continue
            snapshots.update(resp.data or {})
        return snapshots, errors

    def _detect_and_score(
        self,
        model_contract: Stage0ModelContract,
        snapshots: Mapping[str, AlpacaStockSnapshot],
        *,
        attempts: Mapping[str, I12FillLog],
        snapshot_batch_errors: Mapping[str, str],
    ) -> list[LiveFire]:
        fires: list[LiveFire] = []
        for ticker, context in self._contexts.items():
            attempt = attempts[ticker]
            if attempt.selection_status == "intended":
                continue
            if ticker in snapshot_batch_errors:
                attempt.snapshot_status = "batch_error"
                attempt.coverage_error = snapshot_batch_errors[ticker]
                attempt.fire_status = "not_evaluated"
                continue
            snapshot = snapshots.get(ticker)
            if snapshot is None:
                attempt.snapshot_status = "missing"
                attempt.fire_status = "not_evaluated"
                continue
            minute_ts = _minute_timestamp(snapshot)
            latest_trade_ts = _latest_trade_timestamp(snapshot)
            minute_age = _age_seconds(minute_ts, self._asof)
            latest_trade_age = _age_seconds(latest_trade_ts, self._asof)
            attempt.minute_ts = minute_ts
            attempt.minute_age_seconds = minute_age
            attempt.latest_trade_ts = latest_trade_ts
            attempt.latest_trade_age_seconds = latest_trade_age
            attempt.snapshot_ts = minute_ts
            attempt.snapshot_age_seconds = minute_age
            if minute_age is None:
                attempt.snapshot_status = "missing_minute_data"
                attempt.coverage_error = "minute_timestamp_missing"
                attempt.fire_status = "not_evaluated"
                continue
            if minute_age > self._config.max_snapshot_age_seconds:
                attempt.snapshot_status = "stale_minute_data"
                attempt.coverage_error = (
                    f"minute_age_seconds={minute_age:.3f} exceeds "
                    f"{self._config.max_snapshot_age_seconds:.3f}"
                )
                attempt.fire_status = "not_evaluated"
                continue
            attempt.snapshot_status = "ok"
            polygon_like = _polygon_snapshot_from_alpaca(snapshot)
            shared = compute_shared_intraday_math(
                context,
                polygon_like,
                trading_date=context.context_date,
            )
            if shared is None:
                attempt.fire_status = "invalid_shared_math"
                continue
            decision = i12_entry_gate(context, polygon_like, shared)
            if not decision.enter:
                attempt.fire_status = "not_fired"
                continue
            attempt.fire_status = "fired"
            feature_payload = build_i12_live_feature_payload(context, decision.gate_values)
            assert_i12_live_feature_payload_leakage_clean(feature_payload)
            assert_live_payload_matches_model_schema(
                feature_payload,
                model_contract.feature_names,
            )
            signal = self._upsert_signal(
                ticker=ticker,
                context=context,
                feature_payload=feature_payload,
                gate_values=decision.gate_values,
                decision_ts=shared.data_clock,
            )
            attempt.signal_id = signal.signal_id
            attempt.feature_json = _json_dumps(feature_payload)
            attempt.gate_values_json = _json_dumps(dict(decision.gate_values))
            attempt.projected_vol_ratio = _optional_float(
                feature_payload.get("projected_volume_ratio_at_confirmation")
            )
            attempt.gap = _optional_float(feature_payload.get("gap"))
            attempt.off_52w_high = _optional_float(
                feature_payload.get("distance_from_max252")
            )
            try:
                score = score_signal_shadow(
                    self._session,
                    signal_id=signal.signal_id,
                    model_id=model_contract.model.model_id,
                    score_status="stage0_read_only",
                )
            except Exception as exc:
                attempt.score_stage0_status = "failed"
                attempt.coverage_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                continue
            attempt.score_id = score.score_id
            attempt.ml_score = score.score
            attempt.score_source = score.score_source
            attempt.score_status = score.score_status
            attempt.fallback_reason = score.fallback_reason
            if (
                score.score_source == "model_shadow"
                and score.score is not None
                and math.isfinite(float(score.score))
            ):
                attempt.score_stage0_status = "model_ok"
            elif score.score_source.startswith("fallback"):
                attempt.score_stage0_status = "fallback"
            else:
                attempt.score_stage0_status = "failed"
            fires.append(
                LiveFire(
                    ticker=ticker,
                    signal=signal,
                    score=score,
                    feature_payload=feature_payload,
                    gate_values=dict(decision.gate_values),
                    snapshot=snapshot,
                    attempt=attempt,
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

    def _attempt_row(
        self,
        *,
        ticker: str,
        context: PremarketContext,
        model_contract: Stage0ModelContract,
        decision_date: date,
        run_config_hash: str,
        half_day: bool,
        session_minutes: int,
        projection_basis: str,
    ) -> I12FillLog:
        content_hash = _attempt_content_hash(
            ticker=ticker,
            decision_date=decision_date,
            model_id=model_contract.model.model_id,
            intended_order_usd=self._config.intended_order_usd,
            stage0_run_config_hash=run_config_hash,
            context_artifact_hash=self._config.context_artifact_hash,
        )
        existing = (
            self._session.query(I12FillLog)
            .filter(I12FillLog.content_hash == content_hash)
            .one_or_none()
        )
        if existing is not None:
            return existing
        row = I12FillLog(
            model_id=model_contract.model.model_id,
            ticker=ticker.upper(),
            decision_date=decision_date,
            decision_ts=self._asof,
            feed=self._config.feed,
            model_selection_mode=model_contract.model_selection_mode,
            promotable_run=model_contract.promotable_run,
            stage0_run_config_hash=run_config_hash,
            context_artifact_hash=self._config.context_artifact_hash,
            half_day=half_day,
            session_minutes=session_minutes,
            projection_basis=projection_basis,
            attempt_stage="context",
            snapshot_status="pending",
            fire_status="not_evaluated",
            score_stage0_status="not_evaluated",
            selection_status="not_selected",
            quote_status="not_requested",
            exit_capture_status="not_due",
            skipped_reason="not_selected",
            intended_order_usd=self._config.intended_order_usd,
            feature_json="{}",
            gate_values_json="{}",
            content_hash=content_hash,
            projected_vol_ratio=None,
            gap=None,
            off_52w_high=None,
            coverage_error=None,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def _log_intended_trade(
        self,
        model_contract: Stage0ModelContract,
        fire: LiveFire,
    ) -> I12FillLog:
        row = fire.attempt
        if row.selection_status == "intended":
            return row
        quote = fire.snapshot.latest_quote
        liquidity = evaluate_quote_liquidity(
            quote,
            intended_order_usd=self._config.intended_order_usd,
            max_spread_bps=self._config.max_spread_bps,
            asof=self._asof,
            max_quote_age_seconds=self._config.max_quote_age_seconds,
        )
        decision_ts = _stored_utc(fire.signal.signal_timestamp)
        row.signal_id = fire.signal.signal_id
        row.score_id = fire.score.score_id
        row.model_id = model_contract.model.model_id
        row.decision_ts = decision_ts
        row.exit_capture_due_ts = _next_session_open_after(decision_ts)
        row.feed = self._config.feed
        row.model_selection_mode = model_contract.model_selection_mode
        row.promotable_run = model_contract.promotable_run
        row.attempt_stage = "intended"
        row.selection_status = "intended"
        row.ml_score = fire.score.score
        row.score_source = fire.score.score_source
        row.score_status = fire.score.score_status
        row.fallback_reason = fire.score.fallback_reason
        row.projected_vol_ratio = _optional_float(
            fire.feature_payload.get("projected_volume_ratio_at_confirmation")
        )
        row.gap = _optional_float(fire.feature_payload.get("gap"))
        row.off_52w_high = _optional_float(fire.feature_payload.get("distance_from_max252"))
        row.bid = liquidity["bid"]
        row.ask = liquidity["ask"]
        row.spread_bps = liquidity["spread_bps"]
        row.top_of_book_size = liquidity["top_of_book_size"]
        row.intended_order_usd = self._config.intended_order_usd
        row.size_sufficient = liquidity["size_sufficient"]
        row.halted = None
        row.quote_condition_halt_inferred = liquidity["quote_condition_halt_inferred"]
        row.quote_status = liquidity["quote_status"]
        row.quote_ts = liquidity["entry_quote_ts"]
        row.quote_age_seconds = liquidity["entry_quote_age_seconds"]
        row.entry_quote_ts = liquidity["entry_quote_ts"]
        row.entry_quote_age_seconds = liquidity["entry_quote_age_seconds"]
        row.skipped_reason = liquidity["skipped_reason"]
        if row.skipped_reason != "none":
            row.exit_capture_status = "skipped_cash"
            row.modeled_return = 0.0
        row.feature_json = _json_dumps(fire.feature_payload)
        row.gate_values_json = _json_dumps(fire.gate_values)
        row.quote_json = _json_dumps(_quote_payload(quote)) if quote else None
        self._session.flush()
        return row


def capture_i12_exit_quotes(
    session: Session,
    alpaca_adapter: AlpacaAdapter,
    *,
    feed: str = "sip",
    asof: datetime | None = None,
    max_quote_age_seconds: float = DEFAULT_MAX_QUOTE_AGE_SECONDS,
) -> dict[str, Any]:
    asof_ts = _aware_utc(asof or utcnow())
    due_rows = [
        row
        for row in session.query(I12FillLog)
        .filter(
            I12FillLog.selection_status == "intended",
            I12FillLog.quote_status == "ok",
            I12FillLog.skipped_reason == "none",
            or_(
                I12FillLog.exit_capture_status.is_(None),
                I12FillLog.exit_capture_status.notin_(
                    list(EXIT_CAPTURE_TERMINAL_STATUSES)
                ),
            ),
        )
        .all()
        if _exit_quote_due(row, asof_ts)
    ]
    if not due_rows:
        return {"exit_quote_updates": 0}
    tickers = [row.ticker for row in due_rows]
    quotes_resp = alpaca_adapter.get_latest_quotes(sorted(set(tickers)), feed=feed)
    if not quotes_resp.ok:
        for row in due_rows:
            row.exit_capture_status = "error"
            row.coverage_error = str(quotes_resp.error)
        session.commit()
        return {
            "exit_quote_updates": 0,
            "exit_quote_errors": len(due_rows),
            "error": str(quotes_resp.error),
        }
    quotes = quotes_resp.data or {}
    updated = 0
    missing = 0
    stale = 0
    halted = 0
    for row in due_rows:
        quote = quotes.get(row.ticker.upper())
        if quote is None:
            row.exit_capture_status = "missing"
            missing += 1
            continue
        quote_ts = _parse_provider_ts(quote.timestamp)
        quote_age = _age_seconds(quote_ts, asof_ts)
        row.exit_quote_ts = quote_ts
        row.exit_quote_age_seconds = quote_age
        row.exit_quote_json = _json_dumps(_quote_payload(quote))
        if quote_age is None or quote_age > max_quote_age_seconds:
            row.exit_capture_status = "stale"
            stale += 1
            continue
        if _quote_halted(quote):
            row.exit_capture_status = "condition_halt_inferred"
            halted += 1
            continue
        row.exit_bid = quote.bid_price
        row.exit_ask = quote.ask_price
        row.exit_capture_status = "ok"
        if row.ask and row.ask > 0 and row.exit_bid is not None:
            row.modeled_return = (row.exit_bid / row.ask) - 1.0
        updated += 1
    session.commit()
    return {
        "exit_quote_updates": updated,
        "exit_quote_missing": missing,
        "exit_quote_stale": stale,
        "exit_quote_halted_inferred": halted,
    }


def i12_gate0_report(
    session: Session,
    *,
    decision_date: date | None = None,
    asof: datetime | None = None,
    max_spread_bps: float = DEFAULT_MAX_SPREAD_BPS,
    intended_order_usd: float = DEFAULT_INTENDED_ORDER_USD,
    min_context_count: int = DEFAULT_MIN_CONTEXT_COUNT,
    min_intended_count: int = DEFAULT_MIN_INTENDED_COUNT,
    min_snapshot_ok_rate: float = DEFAULT_MIN_SNAPSHOT_OK_RATE,
    max_snapshot_error_or_missing_rate: float = DEFAULT_MAX_SNAPSHOT_ERROR_OR_MISSING_RATE,
    min_score_model_ok_rate: float = DEFAULT_MIN_SCORE_MODEL_OK_RATE,
    min_quote_ok_rate: float = DEFAULT_MIN_QUOTE_OK_RATE,
    min_exit_quote_ok_rate: float = DEFAULT_MIN_EXIT_QUOTE_OK_RATE,
    min_gate0_intended_count: int = DEFAULT_MIN_GATE0_INTENDED_COUNT,
    min_gate0_distinct_trading_days: int = DEFAULT_MIN_GATE0_DISTINCT_TRADING_DAYS,
    min_gate0_tradeable_rate: float = DEFAULT_MIN_GATE0_TRADEABLE_RATE,
) -> dict[str, Any]:
    query = session.query(I12FillLog)
    if decision_date is not None:
        query = query.filter(I12FillLog.decision_date == decision_date)
    rows = query.all()
    asof_ts = _aware_utc(asof or utcnow())
    total = len(rows)
    intended_rows = [row for row in rows if row.selection_status == "intended"]
    context_distinct_trading_days = len({
        row.decision_date for row in rows if row.decision_date
    })
    intended_distinct_trading_days = len({
        row.decision_date for row in intended_rows if row.decision_date
    })
    tradeable_rows = [
        row
        for row in intended_rows
        if _gate0_row_is_tradeable(
            row,
            max_spread_bps=max_spread_bps,
            intended_order_usd=intended_order_usd,
        )
    ]
    skipped_cash_rows = [
        row
        for row in intended_rows
        if row.skipped_reason != "none"
        or row.exit_capture_status == "skipped_cash"
    ]
    spread_ok = len([
        row
        for row in intended_rows
        if _gate0_row_has_report_ok_spread(row, max_spread_bps=max_spread_bps)
    ])
    size_ok = len([
        row
        for row in intended_rows
        if _gate0_row_has_report_ok_size(
            row,
            intended_order_usd=intended_order_usd,
        )
    ])
    tradeable = len(tradeable_rows)
    evidence_conflict_count = len([
        row
        for row in intended_rows
        if row.skipped_reason == "none"
        and not _gate0_row_is_tradeable(
            row,
            max_spread_bps=max_spread_bps,
            intended_order_usd=intended_order_usd,
        )
    ])
    entry_integrity_conflict_count = len([
        row
        for row in intended_rows
        if _gate0_row_has_entry_integrity_conflict(row)
    ])
    quote_size_bases = sorted({
        basis
        for row in intended_rows
        if (basis := _quote_size_basis_from_row(row)) is not None
    })
    missing_quote_size_basis_count = len([
        row for row in intended_rows if _quote_size_basis_from_row(row) is None
    ])
    unsupported_quote_size_basis_count = len([
        row
        for row in intended_rows
        if (
            (basis := _quote_size_basis_from_row(row)) is not None
            and basis not in SUPPORTED_ALPACA_QUOTE_SIZE_BASES
        )
    ])
    feeds = sorted({row.feed for row in rows if row.feed})
    run_config_hashes = sorted({row.stage0_run_config_hash for row in rows if row.stage0_run_config_hash})
    missing_run_config_count = len([row for row in rows if not row.stage0_run_config_hash])
    context_artifact_hashes = sorted({
        row.context_artifact_hash for row in rows if row.context_artifact_hash
    })
    missing_context_artifact_hash_count = len([
        row for row in rows if not row.context_artifact_hash
    ])
    context_hashes_by_day: dict[str, set[str]] = {}
    for row in rows:
        if row.decision_date and row.context_artifact_hash:
            context_hashes_by_day.setdefault(row.decision_date.isoformat(), set()).add(
                row.context_artifact_hash
            )
    context_artifact_hashes_by_day = {
        day: sorted(hashes)
        for day, hashes in sorted(context_hashes_by_day.items())
    }
    mixed_context_artifact_days = [
        day
        for day, hashes in context_artifact_hashes_by_day.items()
        if len(hashes) > 1
    ]
    non_promotable_reasons = []
    if any(feed not in PROMOTION_FEEDS for feed in feeds):
        non_promotable_reasons.append("diagnostic_feed")
    if any(row.model_selection_mode == "latest_diagnostic" for row in rows):
        non_promotable_reasons.append("latest_model_diagnostic")
    if any(row.promotable_run is False for row in rows):
        non_promotable_reasons.append("non_promotable_row")
    if (missing_run_config_count or len(run_config_hashes) != 1) and rows:
        non_promotable_reasons.append("mixed_or_missing_stage0_run_config")
    if missing_context_artifact_hash_count and rows:
        non_promotable_reasons.append("missing_context_artifact_hash")
    if mixed_context_artifact_days:
        non_promotable_reasons.append("mixed_context_artifact_hash_for_day")
    if missing_quote_size_basis_count:
        non_promotable_reasons.append("missing_quote_size_basis")
    if len(quote_size_bases) > 1:
        non_promotable_reasons.append("mixed_quote_size_basis")
    if unsupported_quote_size_basis_count:
        non_promotable_reasons.append("unsupported_quote_size_basis")
    if entry_integrity_conflict_count:
        non_promotable_reasons.append("entry_integrity_conflict")
    if any(row.half_day is True for row in rows):
        non_promotable_reasons.append("half_day_diagnostic")
    promotable = not non_promotable_reasons and bool(rows)
    snapshot_ok = _count(rows, "snapshot_status", "ok")
    snapshot_error_or_missing = len([
        row for row in rows
        if row.snapshot_status
        in {
            "missing",
            "batch_error",
            "stale",
            "missing_minute_data",
            "stale_minute_data",
        }
    ])
    fire_count = _count(rows, "fire_status", "fired")
    score_model_ok = _count(rows, "score_stage0_status", "model_ok")
    quote_ok = _count(intended_rows, "quote_status", "ok")
    exit_coverage_rows = [
        row for row in tradeable_rows if _exit_quote_due(row, asof_ts)
    ]
    exit_quote_pending_due = len([
        row
        for row in exit_coverage_rows
        if row.exit_capture_status not in EXIT_CAPTURE_TRADEABLE_TERMINAL_STATUSES
    ])
    exit_quote_ok = _count(exit_coverage_rows, "exit_capture_status", "ok")
    snapshot_ok_rate = snapshot_ok / total if total else None
    snapshot_error_or_missing_rate = snapshot_error_or_missing / total if total else None
    score_model_ok_rate = score_model_ok / fire_count if fire_count else None
    quote_ok_rate = quote_ok / len(intended_rows) if intended_rows else None
    tradeable_rate = tradeable / len(intended_rows) if intended_rows else None
    exit_quote_ok_rate = (
        exit_quote_ok / len(exit_coverage_rows) if exit_coverage_rows else None
    )
    coverage_failures = []
    if total < min_context_count:
        coverage_failures.append("min_context_count")
    if len(intended_rows) < min_intended_count:
        coverage_failures.append("min_intended_count")
    if len(intended_rows) < min_gate0_intended_count:
        coverage_failures.append("min_gate0_intended_count")
    if intended_distinct_trading_days < min_gate0_distinct_trading_days:
        coverage_failures.append("min_gate0_distinct_trading_days")
    if tradeable_rate is None or tradeable_rate < min_gate0_tradeable_rate:
        coverage_failures.append("min_gate0_tradeable_rate")
    if snapshot_ok_rate is None or snapshot_ok_rate < min_snapshot_ok_rate:
        coverage_failures.append("min_snapshot_ok_rate")
    if (
        snapshot_error_or_missing_rate is None
        or snapshot_error_or_missing_rate > max_snapshot_error_or_missing_rate
    ):
        coverage_failures.append("max_snapshot_error_or_missing_rate")
    if fire_count and (
        score_model_ok_rate is None
        or score_model_ok_rate < min_score_model_ok_rate
    ):
        coverage_failures.append("min_score_model_ok_rate")
    if quote_ok_rate is None or quote_ok_rate < min_quote_ok_rate:
        coverage_failures.append("min_quote_ok_rate")
    if exit_coverage_rows and (
        exit_quote_ok_rate is None
        or exit_quote_ok_rate < min_exit_quote_ok_rate
    ):
        coverage_failures.append("min_exit_quote_ok_rate")
    coverage_gate_passed = not coverage_failures
    return {
        "rows": total,
        "context_count": total,
        "snapshot_ok": snapshot_ok,
        "snapshot_missing": _count(rows, "snapshot_status", "missing"),
        "snapshot_batch_errors": _count(rows, "snapshot_status", "batch_error"),
        "snapshot_stale": _count(rows, "snapshot_status", "stale"),
        "snapshot_missing_minute_data": _count(rows, "snapshot_status", "missing_minute_data"),
        "snapshot_stale_minute_data": _count(rows, "snapshot_status", "stale_minute_data"),
        "snapshot_ok_rate": snapshot_ok_rate,
        "snapshot_error_or_missing_rate": snapshot_error_or_missing_rate,
        "fire_count": fire_count,
        "score_model_ok": score_model_ok,
        "score_fallback": _count(rows, "score_stage0_status", "fallback"),
        "score_failed": _count(rows, "score_stage0_status", "failed"),
        "score_model_ok_rate": score_model_ok_rate,
        "selected_count": len(intended_rows),
        "intended_count": len(intended_rows),
        "context_distinct_trading_days": context_distinct_trading_days,
        "intended_distinct_trading_days": intended_distinct_trading_days,
        "distinct_trading_days": intended_distinct_trading_days,
        "quote_ok": quote_ok,
        "quote_missing": _count(intended_rows, "quote_status", "missing"),
        "quote_stale": _count(intended_rows, "quote_status", "stale"),
        "quote_halted": _count(intended_rows, "quote_status", "condition_halt_inferred"),
        "quote_ok_rate": quote_ok_rate,
        "quote_halt_source": "quote_conditions_inferred",
        "skipped_as_cash_count": len(skipped_cash_rows),
        "exit_quote_ok": exit_quote_ok,
        "exit_quote_missing": _count(exit_coverage_rows, "exit_capture_status", "missing"),
        "exit_quote_stale": _count(exit_coverage_rows, "exit_capture_status", "stale"),
        "exit_quote_halted": _count(
            exit_coverage_rows,
            "exit_capture_status",
            "condition_halt_inferred",
        ),
        "exit_quote_errors": _count(exit_coverage_rows, "exit_capture_status", "error"),
        "exit_quote_pending_due": exit_quote_pending_due,
        "exit_quote_ok_rate": exit_quote_ok_rate,
        "exit_quote_coverage_count": len(exit_coverage_rows),
        "exit_quote_tradeable_denominator": len(exit_coverage_rows),
        "exit_quote_skipped_cash_count": len(skipped_cash_rows),
        "spread_ok": spread_ok,
        "size_ok": size_ok,
        "tradeable": tradeable,
        "evidence_conflict_count": evidence_conflict_count,
        "entry_integrity_conflict_count": entry_integrity_conflict_count,
        "spread_ok_rate": spread_ok / len(intended_rows) if intended_rows else None,
        "size_ok_rate": size_ok / len(intended_rows) if intended_rows else None,
        "tradeable_rate": tradeable_rate,
        "passed": bool(
            promotable
            and coverage_gate_passed
            and intended_rows
            and tradeable_rate is not None
            and tradeable_rate >= min_gate0_tradeable_rate
        ),
        "coverage_gate_passed": coverage_gate_passed,
        "coverage_gate_failures": sorted(set(coverage_failures)),
        "coverage_thresholds": {
            "min_context_count": min_context_count,
            "min_intended_count": min_intended_count,
            "min_snapshot_ok_rate": min_snapshot_ok_rate,
            "max_snapshot_error_or_missing_rate": max_snapshot_error_or_missing_rate,
            "min_score_model_ok_rate": min_score_model_ok_rate,
            "min_quote_ok_rate": min_quote_ok_rate,
            "min_exit_quote_ok_rate": min_exit_quote_ok_rate,
            "min_gate0_intended_count": min_gate0_intended_count,
            "min_gate0_distinct_trading_days": min_gate0_distinct_trading_days,
            "min_gate0_tradeable_rate": min_gate0_tradeable_rate,
        },
        "asof": asof_ts.isoformat(),
        "promotable": promotable,
        "non_promotable_reasons": sorted(set(non_promotable_reasons)),
        "stage0_run_config_hash": run_config_hashes[0] if len(run_config_hashes) == 1 else None,
        "stage0_run_config_hashes": run_config_hashes,
        "missing_stage0_run_config_count": missing_run_config_count,
        "quote_size_basis": quote_size_bases[0] if len(quote_size_bases) == 1 else None,
        "quote_size_bases": quote_size_bases,
        "missing_quote_size_basis_count": missing_quote_size_basis_count,
        "unsupported_quote_size_basis_count": unsupported_quote_size_basis_count,
        "context_artifact_hashes": context_artifact_hashes,
        "context_artifact_hashes_by_day": context_artifact_hashes_by_day,
        "mixed_context_artifact_days": mixed_context_artifact_days,
        "missing_context_artifact_hash_count": missing_context_artifact_hash_count,
        "half_day": any(row.half_day is True for row in rows) if rows else None,
        "session_minutes": _single_value([row.session_minutes for row in rows]),
        "projection_basis": _single_value([row.projection_basis for row in rows]),
        "feed": feeds[0] if len(feeds) == 1 else None,
        "feeds": feeds,
        "max_spread_bps": max_spread_bps,
        "intended_order_usd": intended_order_usd,
    }


def select_i12_model(
    session: Session,
    *,
    model_id: str | None,
    allow_latest_model: bool,
    feed: str = "sip",
) -> Stage0ModelContract:
    if model_id:
        model = session.get(MLModelRegistry, model_id)
        if model is None:
            raise RuntimeError(f"I12 model {model_id!r} not found in ml_model_registry")
        selection_mode = "explicit"
    if not allow_latest_model:
        if not model_id:
            raise RuntimeError("model_id required unless allow_latest_model is true")
    if not model_id:
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
        selection_mode = "latest_diagnostic"
    feature_names = validate_i12_stage0_model_contract(model)
    non_promotable_reasons: list[str] = []
    if selection_mode == "latest_diagnostic":
        non_promotable_reasons.append("latest_model_diagnostic")
    if feed not in PROMOTION_FEEDS:
        non_promotable_reasons.append("diagnostic_feed")
    return Stage0ModelContract(
        model=model,
        feature_names=feature_names,
        model_selection_mode=selection_mode,
        promotable_run=not non_promotable_reasons,
        non_promotable_reasons=tuple(non_promotable_reasons),
    )


def validate_i12_stage0_model_contract(model: MLModelRegistry) -> tuple[str, ...]:
    if model.pattern_id != I12_PATTERN_ID:
        raise RuntimeError(f"model {model.model_id!r} is pattern {model.pattern_id}, not I12")
    if model.status == "rejected":
        raise RuntimeError(f"model {model.model_id!r} is rejected")
    if model.status not in PROMOTABLE_MODEL_STATUSES:
        raise RuntimeError(
            f"model {model.model_id!r} status {model.status!r} is not in "
            f"Stage-0 promotable allowlist {sorted(PROMOTABLE_MODEL_STATUSES)!r}"
        )
    if not model.manifest_sha256:
        raise RuntimeError(f"model {model.model_id!r} missing manifest_sha256")
    if not model.feature_schema_hash:
        raise RuntimeError(f"model {model.model_id!r} missing feature_schema_hash")
    feature_schema = _loads_json_object(model.feature_schema_json, "feature_schema_json")
    actual_schema_hash = feature_schema_hash(feature_schema)
    if actual_schema_hash != model.feature_schema_hash:
        raise RuntimeError(
            f"model {model.model_id!r} feature_schema_hash mismatch: "
            f"registry={model.feature_schema_hash} actual={actual_schema_hash}"
        )
    feature_names = _feature_names_from_schema(feature_schema)
    if feature_names != EXPECTED_I12_LIVE_FEATURES:
        raise RuntimeError(
            f"model {model.model_id!r} feature list does not match live I12 contract: "
            f"{feature_names!r}"
        )
    training_params = _loads_json_object(model.training_params_json or "{}", "training_params_json")
    cv_metrics = _loads_json_object(model.cv_metrics_json or "{}", "cv_metrics_json")
    if _first_recursive_value((training_params, cv_metrics), "horizon_sessions") != 1:
        raise RuntimeError(f"model {model.model_id!r} is not a one-session horizon model")
    signal_horizon = _first_recursive_value((training_params, cv_metrics), "signal_horizon")
    if signal_horizon not in (None, "1d"):
        raise RuntimeError(
            f"model {model.model_id!r} signal_horizon is {signal_horizon!r}, not '1d'"
        )
    return feature_names


def assert_live_payload_matches_model_schema(
    payload: Mapping[str, Any],
    feature_names: Sequence[str],
) -> None:
    payload_feature_names = tuple(name for name in feature_names if name in payload)
    if tuple(feature_names) != EXPECTED_I12_LIVE_FEATURES:
        raise RuntimeError("frozen model schema no longer matches expected I12 live features")
    if payload_feature_names != tuple(feature_names):
        raise RuntimeError(
            "live I12 feature payload does not match frozen model schema exactly: "
            f"{payload_feature_names!r} != {tuple(feature_names)!r}"
        )


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
    if contract.get("uses_forward_bars") is not False:
        raise RuntimeError("live I12 payload must record uses_forward_bars=false")


def evaluate_quote_liquidity(
    quote: AlpacaQuote | None,
    *,
    intended_order_usd: float,
    max_spread_bps: float,
    asof: datetime,
    max_quote_age_seconds: float,
) -> dict[str, Any]:
    if quote is None:
        return {
            "bid": None,
            "ask": None,
            "spread_bps": None,
            "top_of_book_size": None,
            "size_sufficient": False,
            "quote_condition_halt_inferred": None,
            "quote_status": "missing",
            "entry_quote_ts": None,
            "entry_quote_age_seconds": None,
            "skipped_reason": "quote_missing",
        }
    quote_ts = _parse_provider_ts(quote.timestamp)
    quote_age = _age_seconds(quote_ts, asof)
    bid = quote.bid_price
    ask = quote.ask_price
    ask_size = quote.ask_size
    halt_inferred = _quote_halted(quote)
    spread_bps = None
    if bid is not None and ask is not None and bid > 0 and ask >= bid:
        spread_bps = (ask - bid) / ((ask + bid) / 2.0) * 10_000.0
    top_of_book_size = ask * ask_size if ask is not None and ask_size is not None else None
    size_sufficient = (
        top_of_book_size is not None
        and top_of_book_size >= intended_order_usd
    )
    skipped_reason = "none"
    quote_status = "ok"
    if quote_age is None or quote_age > max_quote_age_seconds:
        quote_status = "stale"
        skipped_reason = "quote_stale"
    elif halt_inferred:
        quote_status = "condition_halt_inferred"
        skipped_reason = "quote_condition_halt_inferred"
    elif spread_bps is None:
        quote_status = "missing"
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
        "quote_condition_halt_inferred": halt_inferred,
        "quote_status": quote_status,
        "entry_quote_ts": quote_ts,
        "entry_quote_age_seconds": quote_age,
        "skipped_reason": skipped_reason,
    }


def _polygon_snapshot_from_alpaca(snapshot: AlpacaStockSnapshot) -> PolygonSnapshotTicker:
    ts = _minute_timestamp(snapshot)
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
        "quote_size_basis": ALPACA_QUOTE_SIZE_BASIS,
        "timestamp": quote.timestamp,
        "conditions": quote.conditions,
        "tape": quote.tape,
    }


def _attempt_content_hash(
    *,
    ticker: str,
    decision_date: date,
    model_id: str,
    intended_order_usd: float,
    stage0_run_config_hash: str,
    context_artifact_hash: str | None,
) -> str:
    return stable_hash(
        {
            "stage": "i12_live_fill_test_stage0",
            "ticker": ticker.upper(),
            "decision_date": decision_date.isoformat(),
            "model_id": model_id,
            "intended_order_usd": intended_order_usd,
            "stage0_run_config_hash": stage0_run_config_hash,
            "context_artifact_hash": (
                context_artifact_hash or "missing_context_artifact"
            ),
        }
    )


def _minute_timestamp(snapshot: AlpacaStockSnapshot) -> datetime | None:
    return _parse_provider_ts(snapshot.minute_timestamp)


def _latest_trade_timestamp(snapshot: AlpacaStockSnapshot) -> datetime | None:
    return _parse_provider_ts(snapshot.latest_trade_timestamp)


def _age_seconds(provider_ts: datetime | None, asof: datetime) -> float | None:
    if provider_ts is None:
        return None
    return max(0.0, (_stored_utc(asof) - _stored_utc(provider_ts)).total_seconds())


def _count(rows: Sequence[I12FillLog], field_name: str, expected: str) -> int:
    return len([row for row in rows if getattr(row, field_name) == expected])


def _gate0_row_is_tradeable(
    row: I12FillLog,
    *,
    max_spread_bps: float,
    intended_order_usd: float,
) -> bool:
    return (
        row.selection_status == "intended"
        and row.quote_status == "ok"
        and row.skipped_reason == "none"
        and row.quote_condition_halt_inferred is not True
        and not _gate0_row_has_entry_integrity_conflict(row)
        and _gate0_row_has_report_ok_spread(
            row,
            max_spread_bps=max_spread_bps,
        )
        and _gate0_row_has_report_ok_size(
            row,
            intended_order_usd=intended_order_usd,
        )
    )


def _gate0_row_has_report_ok_spread(
    row: I12FillLog,
    *,
    max_spread_bps: float,
) -> bool:
    observed_spread_bps = _gate0_row_observed_spread_bps(row)
    return observed_spread_bps is not None and observed_spread_bps <= max_spread_bps


def _gate0_row_has_report_ok_size(
    row: I12FillLog,
    *,
    intended_order_usd: float,
) -> bool:
    observed_quote = _gate0_row_observed_quote(row)
    top_of_book_size = (
        observed_quote["top_of_book_size"] if observed_quote is not None else None
    )
    return (
        top_of_book_size is not None
        and top_of_book_size >= intended_order_usd
        and row.size_sufficient is True
    )


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    observed = float(value)
    return observed if math.isfinite(observed) else None


def _materially_close(stored: float, observed: float) -> bool:
    return abs(stored - observed) <= max(1e-6, abs(observed) * 1e-9)


def _gate0_row_observed_quote(row: I12FillLog) -> dict[str, float] | None:
    try:
        payload = json.loads(row.quote_json or "")
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    quote_size_basis = payload.get("quote_size_basis")
    if quote_size_basis not in SUPPORTED_ALPACA_QUOTE_SIZE_BASES:
        return None
    bid = _finite_float(payload.get("bid_price"))
    ask = _finite_float(payload.get("ask_price"))
    ask_size = _finite_float(payload.get("ask_size"))
    if (
        bid is None
        or ask is None
        or ask_size is None
        or bid <= 0.0
        or ask <= 0.0
        or ask < bid
        or ask_size < 0.0
    ):
        return None
    return {
        "bid": bid,
        "ask": ask,
        "ask_size": ask_size,
        "spread_bps": ((ask - bid) / ((ask + bid) / 2.0)) * 10_000.0,
        "top_of_book_size": ask * ask_size,
    }


def _gate0_row_observed_spread_bps(row: I12FillLog) -> float | None:
    observed_quote = _gate0_row_observed_quote(row)
    return observed_quote["spread_bps"] if observed_quote is not None else None


def _gate0_row_has_entry_integrity_conflict(row: I12FillLog) -> bool:
    if row.skipped_reason != "none":
        return False
    observed_quote = _gate0_row_observed_quote(row)
    bid = _finite_float(row.bid)
    ask = _finite_float(row.ask)
    spread_bps = _finite_float(row.spread_bps)
    top_of_book_size = _finite_float(row.top_of_book_size)
    intended_order_usd = _finite_float(row.intended_order_usd)
    expected_size_sufficient = (
        observed_quote is not None
        and intended_order_usd is not None
        and intended_order_usd > 0.0
        and observed_quote["top_of_book_size"] >= intended_order_usd
    )
    return (
        row.quote_status != "ok"
        or row.quote_condition_halt_inferred is True
        or observed_quote is None
        or bid is None
        or ask is None
        or spread_bps is None
        or spread_bps < 0.0
        or top_of_book_size is None
        or intended_order_usd is None
        or intended_order_usd <= 0.0
        or not _materially_close(bid, observed_quote["bid"])
        or not _materially_close(ask, observed_quote["ask"])
        or not _materially_close(spread_bps, observed_quote["spread_bps"])
        or top_of_book_size < 0.0
        or not _materially_close(
            top_of_book_size,
            observed_quote["top_of_book_size"],
        )
        or not expected_size_sufficient
        or row.size_sufficient is not expected_size_sufficient
    )


def _quote_size_basis_from_row(row: I12FillLog) -> str | None:
    try:
        payload = json.loads(row.quote_json or "")
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    basis = payload.get("quote_size_basis")
    return basis if isinstance(basis, str) and basis else None


def _single_value(values: Sequence[Any]) -> Any:
    non_null = {value for value in values if value is not None}
    return next(iter(non_null)) if len(non_null) == 1 else None


def _session_minutes(day: date) -> int:
    close_time = us_equity_session_close_time(day)
    open_minutes = 9 * 60 + 30
    close_minutes = close_time.hour * 60 + close_time.minute
    return close_minutes - open_minutes


def _stage0_run_config_hash(
    config: I12LiveFillConfig,
    model_contract: Stage0ModelContract,
) -> str:
    return stable_hash(
        {
            "feed": config.feed,
            "top_k": config.top_k,
            "intended_order_usd": config.intended_order_usd,
            "max_spread_bps": config.max_spread_bps,
            "max_quote_age_seconds": config.max_quote_age_seconds,
            "max_snapshot_age_seconds": config.max_snapshot_age_seconds,
            "quote_size_basis": ALPACA_QUOTE_SIZE_BASIS,
            "model_id": model_contract.model.model_id,
            "model_selection_mode": model_contract.model_selection_mode,
        }
    )


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


def _loads_json_object(raw: str, field_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"model {field_name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"model {field_name} must be a JSON object")
    return payload


def _feature_names_from_schema(feature_schema: Mapping[str, Any]) -> tuple[str, ...]:
    fields = feature_schema.get("fields")
    if not isinstance(fields, list):
        raise RuntimeError("model feature_schema_json missing fields list")
    names: list[str] = []
    for field in fields:
        if not isinstance(field, dict) or not isinstance(field.get("name"), str):
            raise RuntimeError("model feature_schema_json fields must contain string names")
        names.append(field["name"])
    return tuple(names)


def _first_recursive_value(payloads: Sequence[Any], key: str) -> Any:
    for payload in payloads:
        found = _recursive_value(payload, key)
        if found is not None:
            if key == "horizon_sessions":
                try:
                    return int(found)
                except (TypeError, ValueError):
                    return found
            return found
    return None


def _recursive_value(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _recursive_value(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _recursive_value(value, key)
            if found is not None:
                return found
    return None


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _chunks(values: Sequence[str], size: int) -> list[list[str]]:
    return [list(values[idx: idx + size]) for idx in range(0, len(values), size)]
