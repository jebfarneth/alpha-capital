"""All-firings forward-return population.

The production path prices durable M4 signal_registry rows from intended
entry open to the 15-session time barrier open, records a self-auditing
forward_return_observations row, and keeps signal_registry as the summary
surface. The injected price_fn mode is retained for existing measurement
spine tests and simple non-production fixtures.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session

from alpha.data.contracts import stable_hash
from alpha.data.fmp import (
    DELISTED_COMPANIES_ENDPOINT,
    FmpBar,
    HISTORICAL_PRICE_FULL_ENDPOINT,
)
from alpha.db.models import (
    ForwardReturnObservation,
    ForwardReturnObservationEvent,
    SignalRegistry,
)
from alpha.evidence.writer import record_data_lineage
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.market_calendar import (
    next_us_equity_session,
    nth_us_equity_session,
    resolve_us_equity_session,
    us_equity_session_close_timestamp,
    us_equity_session_open_timestamp,
)

STATUS_PENDING = "pending"
STATUS_COMPUTED = "computed"
STATUS_PRICING_UNAVAILABLE_RETRY = "pricing_unavailable_retry"
STATUS_MISSING_ENTRY_PRICE_RETRY = "missing_entry_price_retry"
STATUS_MISSING_EXIT_PRICE_RETRY = "missing_exit_price_retry"
STATUS_INVALID_ENTRY_PRICE_RETRY = "invalid_entry_price_retry"
STATUS_INVALID_EXIT_PRICE_RETRY = "invalid_exit_price_retry"
STATUS_HALTED_PENDING = "halted_pending"
STATUS_CORPORATE_ACTION_REVIEW = "corporate_action_review"
STATUS_SURVIVORSHIP_UNRESOLVED_REVIEW = "survivorship_unresolved_review"
STATUS_OUTCOME_UNAVAILABLE = "outcome_unavailable"

# Legacy fixture-only status retained so existing scaffold tests keep working.
STATUS_INVALID_PRICE_SHAPE_RETRY = "invalid_price_shape_retry"

RETRYABLE_FORWARD_RETURN_STATUSES = (
    STATUS_PENDING,
    STATUS_PRICING_UNAVAILABLE_RETRY,
    STATUS_INVALID_PRICE_SHAPE_RETRY,
    STATUS_INVALID_ENTRY_PRICE_RETRY,
    STATUS_INVALID_EXIT_PRICE_RETRY,
    STATUS_MISSING_ENTRY_PRICE_RETRY,
    STATUS_MISSING_EXIT_PRICE_RETRY,
    STATUS_HALTED_PENDING,
    STATUS_SURVIVORSHIP_UNRESOLVED_REVIEW,
)

PRODUCTION_RETRYABLE_STATUSES = (
    STATUS_PENDING,
    STATUS_PRICING_UNAVAILABLE_RETRY,
    STATUS_INVALID_ENTRY_PRICE_RETRY,
    STATUS_INVALID_EXIT_PRICE_RETRY,
    STATUS_MISSING_ENTRY_PRICE_RETRY,
    STATUS_MISSING_EXIT_PRICE_RETRY,
    STATUS_HALTED_PENDING,
    STATUS_SURVIVORSHIP_UNRESOLVED_REVIEW,
)

REQUIRED_FORWARD_RETURN_STATUSES = (
    STATUS_PENDING,
    STATUS_COMPUTED,
    STATUS_PRICING_UNAVAILABLE_RETRY,
    STATUS_MISSING_ENTRY_PRICE_RETRY,
    STATUS_MISSING_EXIT_PRICE_RETRY,
    STATUS_INVALID_ENTRY_PRICE_RETRY,
    STATUS_INVALID_EXIT_PRICE_RETRY,
    STATUS_HALTED_PENDING,
    STATUS_CORPORATE_ACTION_REVIEW,
    STATUS_SURVIVORSHIP_UNRESOLVED_REVIEW,
    STATUS_OUTCOME_UNAVAILABLE,
)

M4_PATTERN_ID = "M4"
M4_SIGNAL_HORIZON = "15d"
M4_PRICE_SOURCE = "fmp_full_split_adjusted_regular_session_open"
M4_SPLIT_ADJUSTED_OPEN_BASIS_PROOF = "fmp_full_ohlc_split_adjusted_contract_open"
MAX_FORWARD_RETURN_ATTEMPTS = 3
LEGACY_NEXT_EXECUTION_SESSION_FALLBACK_REASON = (
    "legacy_next_execution_session_fallback"
)


@dataclass(frozen=True)
class M4ExitGeometry:
    pattern_id: str
    t1_return: float
    t2_return: float
    t3_return: float
    hard_stop_return: float
    hard_stop_pct: float
    time_barrier_sessions: int
    source_contract: str


M4_EXIT_GEOMETRY = M4ExitGeometry(
    pattern_id=M4_PATTERN_ID,
    t1_return=0.05,
    t2_return=0.12,
    t3_return=1.00,
    hard_stop_return=-0.04,
    hard_stop_pct=0.04,
    time_barrier_sessions=15,
    source_contract="Engineering/Patterns/M4-52WeekHigh/SPEC.md",
)
RETURN_COMPARISON_EPSILON = 1e-12


@dataclass(frozen=True)
class M4ForwardReturnPlan:
    decision_date: date
    next_execution_session: Optional[date]
    entry_session_date: date
    exit_session_date: date
    current_evidence_session_date: date
    mature: bool
    entry_resolution_reason: Optional[str] = None
    pending_reason: Optional[str] = None


@dataclass(frozen=True)
class PathTelemetry:
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    mfe_session_date: Optional[str] = None
    mae_session_date: Optional[str] = None
    max_close_return: Optional[float] = None
    min_close_return: Optional[float] = None
    hit_t1_intraday: Optional[bool] = None
    hit_t2_intraday: Optional[bool] = None
    hit_t3_intraday: Optional[bool] = None
    hit_stop_intraday: Optional[bool] = None
    same_day_barrier_ambiguity: Optional[bool] = None


@dataclass(frozen=True)
class SurvivorshipDecision:
    status: str
    reason: str
    exit_price: Optional[float]
    exit_price_source: Optional[str]
    exit_basis_proof: Optional[str]


@dataclass(frozen=True)
class SurvivorshipResolution:
    decision: SurvivorshipDecision
    data_lineage_ids: List[str]
    provider: Optional[str]
    endpoint: Optional[str]
    provider_request: Dict[str, Any]
    primary_data_lineage_id: Optional[str] = None


@dataclass(frozen=True)
class ProductionPricingResult:
    status: str
    reason: Optional[str]
    entry_price: Optional[float]
    exit_price: Optional[float]
    forward_return: Optional[float]
    entry_data_lineage_id: Optional[str]
    exit_data_lineage_id: Optional[str]
    entry_price_source: Optional[str] = None
    exit_price_source: Optional[str] = None
    entry_basis_proof: Optional[str] = None
    exit_basis_proof: Optional[str] = None
    provider_error_type: Optional[str] = None
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    mfe_session_date: Optional[str] = None
    mae_session_date: Optional[str] = None
    max_close_return: Optional[float] = None
    min_close_return: Optional[float] = None
    hit_t1_intraday: Optional[bool] = None
    hit_t2_intraday: Optional[bool] = None
    hit_t3_intraday: Optional[bool] = None
    hit_stop_intraday: Optional[bool] = None
    same_day_barrier_ambiguity: Optional[bool] = None
    provider: Optional[str] = None
    endpoint: Optional[str] = None
    provider_request: Optional[Dict[str, Any]] = None
    data_lineage_ids: Optional[List[str]] = None


def _finite_price(value: object) -> Optional[float]:
    """Coerce provider prices and reject NaN/Inf before arithmetic."""
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price):
        return None
    return price


def _price_pair(value: object) -> Optional[Tuple[object, object]]:
    """Accept injected fixture prices only when shaped as an entry/exit tuple."""
    if not isinstance(value, tuple) or len(value) != 2:
        return None
    return value[0], value[1]


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _parse_date(value: str) -> date:
    return date.fromisoformat(str(value))


def m4_entry_exit_plan(
    *,
    decision_date: date,
    next_execution_session: Optional[date] = None,
    current_evidence_session_date: date,
    run_timestamp: Optional[datetime] = None,
) -> M4ForwardReturnPlan:
    """Resolve M4 all-firings entry/exit sessions.

    Production M4 signals persist next_execution_session at signal creation
    time. Re-deriving from decision_date is retained only for legacy rows and
    is recorded on the observation as a diagnostic reason.
    """
    entry_resolution_reason = None
    if next_execution_session is None:
        entry_session = next_us_equity_session(decision_date)
        entry_resolution_reason = LEGACY_NEXT_EXECUTION_SESSION_FALLBACK_REASON
    else:
        entry_session = next_execution_session
    exit_session = nth_us_equity_session(
        entry_session,
        M4_EXIT_GEOMETRY.time_barrier_sessions,
    )
    mature = current_evidence_session_date >= exit_session
    pending_reason = None
    if not mature:
        if (
            run_timestamp is not None
            and run_timestamp < us_equity_session_open_timestamp(entry_session)
        ):
            pending_reason = "entry_session_not_open"
        else:
            pending_reason = "exit_session_not_complete"
    return M4ForwardReturnPlan(
        decision_date=decision_date,
        next_execution_session=next_execution_session,
        entry_session_date=entry_session,
        exit_session_date=exit_session,
        current_evidence_session_date=current_evidence_session_date,
        mature=mature,
        entry_resolution_reason=entry_resolution_reason,
        pending_reason=pending_reason,
    )


def m4_forward_return_input_hash(sig: SignalRegistry, plan: M4ForwardReturnPlan) -> str:
    return stable_hash(_input_payload(sig, plan))


def m4_forward_return_outcome_hash(
    sig: SignalRegistry,
    plan: M4ForwardReturnPlan,
    pricing: ProductionPricingResult,
    *,
    entry_payload_hash: Optional[str] = None,
    exit_payload_hash: Optional[str] = None,
) -> str:
    payload = _input_payload(sig, plan)
    payload.update({
        "status": pricing.status,
        "reason": pricing.reason,
        "entry_price": pricing.entry_price,
        "entry_price_source": pricing.entry_price_source,
        "entry_basis_proof": pricing.entry_basis_proof,
        "exit_price": pricing.exit_price,
        "exit_price_source": pricing.exit_price_source,
        "exit_basis_proof": pricing.exit_basis_proof,
        "forward_return": pricing.forward_return,
        "path_telemetry": {
            "max_favorable_excursion": pricing.max_favorable_excursion,
            "max_adverse_excursion": pricing.max_adverse_excursion,
            "mfe_session_date": pricing.mfe_session_date,
            "mae_session_date": pricing.mae_session_date,
            "max_close_return": pricing.max_close_return,
            "min_close_return": pricing.min_close_return,
            "hit_t1_intraday": pricing.hit_t1_intraday,
            "hit_t2_intraday": pricing.hit_t2_intraday,
            "hit_t3_intraday": pricing.hit_t3_intraday,
            "hit_stop_intraday": pricing.hit_stop_intraday,
            "same_day_barrier_ambiguity": pricing.same_day_barrier_ambiguity,
        },
        "entry_payload_hash": entry_payload_hash,
        "exit_payload_hash": exit_payload_hash,
    })
    return stable_hash(payload)


def _input_payload(sig: SignalRegistry, plan: M4ForwardReturnPlan) -> Dict[str, Any]:
    return {
        "layer": "price_fn_forward_return",
        "pattern_id": sig.pattern_id,
        "ticker": sig.ticker,
        "direction": sig.direction,
        "signal_timestamp": _iso(_ensure_aware(sig.signal_timestamp)),
        "signal_horizon": sig.signal_horizon,
        "decision_date": plan.decision_date.isoformat(),
        "next_execution_session": (
            plan.next_execution_session.isoformat()
            if plan.next_execution_session is not None else None
        ),
        "entry_resolution_reason": plan.entry_resolution_reason,
        "entry_session_date": plan.entry_session_date.isoformat(),
        "exit_session_date": plan.exit_session_date.isoformat(),
        "horizon_sessions": M4_EXIT_GEOMETRY.time_barrier_sessions,
        "exit_geometry_source": M4_EXIT_GEOMETRY.source_contract,
        "entry_price_source": M4_PRICE_SOURCE,
        "exit_price_source": M4_PRICE_SOURCE,
        "endpoint": HISTORICAL_PRICE_FULL_ENDPOINT,
        "price_field": "open",
        "basis": "split_adjusted_ohlcv_full_endpoint",
    }


class ForwardReturnJob(BaseJob):
    """Populate forward_return for all M4 signal firings."""

    job_name = "forward_return_population"
    job_type = "measurement"

    def __init__(
        self,
        session: Session,
        price_fn: Optional[
            Callable[
                [str, object, Optional[str]],
                Optional[Tuple[Optional[float], Optional[float]]],
            ]
        ] = None,
        maturity_fn: Optional[
            Callable[[object, Optional[str]], bool]
        ] = None,
        *,
        adapter: Any = None,
        run_timestamp: Optional[datetime] = None,
        pattern_id: str = M4_PATTERN_ID,
        max_attempts: int = MAX_FORWARD_RETURN_ATTEMPTS,
        survivorship_resolver: Any = None,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if price_fn is None and adapter is None:
            raise ValueError("ForwardReturnJob requires price_fn or adapter")
        self._session = session
        self._price_fn = price_fn
        self._maturity_fn = maturity_fn
        self._adapter = adapter
        self._run_timestamp = run_timestamp
        self._pattern_id = pattern_id
        self._max_attempts = max_attempts
        self._survivorship_resolver = survivorship_resolver

    def run(self, ctx: JobContext) -> JobResult:
        if self._adapter is None:
            return self._run_injected_price_fn(ctx)
        return self._run_production_m4(ctx)

    # ------------------------------------------------------------------
    # Production M4 path
    # ------------------------------------------------------------------
    def _run_production_m4(self, ctx: JobContext) -> JobResult:
        run_timestamp, timestamp_error = _resolve_run_timestamp(
            self._run_timestamp,
            ctx.params.get("run_timestamp"),
            ctx.started_at,
        )
        if timestamp_error:
            return JobResult(
                status="failed",
                errors=[{"stage": "params", "message": timestamp_error}],
            )

        session_resolution = resolve_us_equity_session(run_timestamp)
        current_evidence_date = _parse_date(session_resolution.evidence_session_date)
        signals = self._production_signal_query().all()

        computed = 0
        pending = 0
        retryable_unavailable = 0
        terminal_unavailable = 0
        pricing_errors = 0
        observations_upserted = 0
        fetch_errors: List[Dict[str, Any]] = []

        for sig in signals:
            plan, plan_error = self._plan_for_signal(
                sig,
                current_evidence_date,
                run_timestamp=run_timestamp,
            )
            if plan_error:
                attempts = (sig.forward_return_attempts or 0) + 1
                status = _terminalize_if_needed(
                    STATUS_PRICING_UNAVAILABLE_RETRY,
                    attempts,
                    self._max_attempts,
                )
                pricing = ProductionPricingResult(
                    status=status,
                    reason=plan_error,
                    entry_price=None,
                    exit_price=None,
                    forward_return=None,
                    entry_data_lineage_id=None,
                    exit_data_lineage_id=None,
                )
                self._persist_production_outcome(
                    sig, plan=None, pricing=pricing,
                    attempts=attempts, job_run_id=ctx.job_run_id,
                )
                retryable_unavailable += int(status != STATUS_OUTCOME_UNAVAILABLE)
                terminal_unavailable += int(status == STATUS_OUTCOME_UNAVAILABLE)
                pricing_errors += 1
                observations_upserted += 1
                continue

            if not plan.mature:
                pricing = ProductionPricingResult(
                    status=STATUS_PENDING,
                    reason=plan.pending_reason or "not_mature",
                    entry_price=None,
                    exit_price=None,
                    forward_return=None,
                    entry_data_lineage_id=None,
                    exit_data_lineage_id=None,
                )
                self._persist_production_outcome(
                    sig, plan=plan, pricing=pricing,
                    attempts=sig.forward_return_attempts or 0,
                    job_run_id=ctx.job_run_id,
                )
                pending += 1
                observations_upserted += 1
                continue

            attempts = (sig.forward_return_attempts or 0) + 1
            pricing, payload_hash = self._price_m4_signal(sig, plan, ctx.job_run_id)
            if pricing.status not in (STATUS_COMPUTED, STATUS_PENDING):
                final_status = _terminalize_if_needed(
                    pricing.status, attempts, self._max_attempts,
                )
                if final_status != pricing.status:
                    pricing = replace(pricing, status=final_status)
            self._persist_production_outcome(
                sig,
                plan=plan,
                pricing=pricing,
                attempts=attempts,
                job_run_id=ctx.job_run_id,
                entry_payload_hash=payload_hash,
                exit_payload_hash=payload_hash,
            )
            observations_upserted += 1

            if pricing.status == STATUS_COMPUTED:
                computed += 1
            elif pricing.status == STATUS_OUTCOME_UNAVAILABLE:
                terminal_unavailable += 1
            else:
                retryable_unavailable += 1
            if pricing.provider_error_type:
                fetch_errors.append({
                    "ticker": sig.ticker,
                    "error_type": pricing.provider_error_type,
                    "reason": pricing.reason,
                })

        self._session.flush()

        return JobResult(
            status="finished",
            metrics={
                "mode": "production_m4",
                "pattern_id": self._pattern_id,
                "decision_evidence_session_date": (
                    session_resolution.evidence_session_date
                ),
                "total_eligible": len(signals),
                "computed": computed,
                "pending": pending,
                "retryable_unavailable": retryable_unavailable,
                "terminal_unavailable": terminal_unavailable,
                "pricing_errors": pricing_errors,
                "observations_upserted": observations_upserted,
                "fetch_error_count": len(fetch_errors),
                "fetch_errors": fetch_errors[:50],
                "required_statuses": list(REQUIRED_FORWARD_RETURN_STATUSES),
            },
        )

    def _production_signal_query(self):
        if self._pattern_id != M4_PATTERN_ID:
            return self._session.query(SignalRegistry).filter(False)
        return (
            self._session.query(SignalRegistry)
            .filter(
                SignalRegistry.pattern_id == M4_PATTERN_ID,
                SignalRegistry.signal_horizon == M4_SIGNAL_HORIZON,
                or_(
                    SignalRegistry.forward_return_status.in_(
                        PRODUCTION_RETRYABLE_STATUSES
                    ),
                    SignalRegistry.forward_return_status.is_(None),
                ),
            )
            .order_by(SignalRegistry.signal_timestamp, SignalRegistry.ticker)
        )

    def _plan_for_signal(
        self,
        sig: SignalRegistry,
        current_evidence_date: date,
        *,
        run_timestamp: datetime,
    ) -> Tuple[Optional[M4ForwardReturnPlan], Optional[str]]:
        if not sig.trading_date:
            return None, "missing_decision_date"
        try:
            decision_date = _parse_date(sig.trading_date)
        except ValueError:
            return None, "invalid_decision_date"
        next_execution_session = None
        if sig.next_execution_session:
            try:
                next_execution_session = _parse_date(sig.next_execution_session)
            except ValueError:
                return None, "invalid_next_execution_session"
        try:
            return (
                m4_entry_exit_plan(
                    decision_date=decision_date,
                    next_execution_session=next_execution_session,
                    current_evidence_session_date=current_evidence_date,
                    run_timestamp=run_timestamp,
                ),
                None,
            )
        except ValueError as exc:
            return None, f"session_resolution_error:{type(exc).__name__}"

    def _price_m4_signal(
        self,
        sig: SignalRegistry,
        plan: M4ForwardReturnPlan,
        job_run_id: Optional[str],
    ) -> Tuple[ProductionPricingResult, Optional[str]]:
        asof = us_equity_session_close_timestamp(plan.exit_session_date)
        provider_request = _provider_request_payload(
            ticker=sig.ticker,
            from_date=plan.entry_session_date,
            to_date=plan.exit_session_date,
        )
        resp = self._adapter.get_historical_price(
            sig.ticker,
            from_date=plan.entry_session_date,
            to_date=plan.exit_session_date,
            asof=asof,
            adjusted=False,
            require_split_adjusted_close=True,
        )
        lineage_payload = _price_lineage_payload(
            resp.data,
            ticker=sig.ticker,
            from_date=plan.entry_session_date,
            to_date=plan.exit_session_date,
        )
        payload_hash = stable_hash(lineage_payload)
        lineage = record_data_lineage(
            self._session,
            provider=resp.lineage.provider,
            endpoint=resp.lineage.endpoint,
            asof_timestamp=resp.lineage.asof_timestamp,
            raw_payload=lineage_payload,
            raw_payload_hash=resp.lineage.raw_payload_hash or payload_hash,
            request_timestamp=resp.lineage.request_timestamp,
            freshness_seconds=resp.lineage.freshness_seconds,
            source_authority=resp.lineage.source_authority,
            data_quality_flags=resp.lineage.data_quality_flags,
            job_run_id=job_run_id,
        )
        lineage_ids = [lineage.data_lineage_id]

        def build_result(
            *,
            status: str,
            reason: Optional[str],
            entry_price: Optional[float],
            exit_price: Optional[float],
            forward_return: Optional[float],
            entry_price_source: Optional[str] = None,
            exit_price_source: Optional[str] = None,
            entry_basis_proof: Optional[str] = None,
            exit_basis_proof: Optional[str] = None,
            entry_data_lineage_id: Optional[str] = None,
            exit_data_lineage_id: Optional[str] = None,
            provider_error_type: Optional[str] = None,
            telemetry: Optional[PathTelemetry] = None,
            provider: Optional[str] = None,
            endpoint: Optional[str] = None,
            provider_request_payload: Optional[Dict[str, Any]] = None,
            data_lineage_ids: Optional[List[str]] = None,
        ) -> ProductionPricingResult:
            telemetry = telemetry or PathTelemetry()
            return ProductionPricingResult(
                status=status,
                reason=reason,
                entry_price=entry_price,
                exit_price=exit_price,
                forward_return=forward_return,
                entry_data_lineage_id=entry_data_lineage_id or lineage.data_lineage_id,
                exit_data_lineage_id=exit_data_lineage_id or lineage.data_lineage_id,
                entry_price_source=entry_price_source,
                exit_price_source=exit_price_source,
                entry_basis_proof=entry_basis_proof,
                exit_basis_proof=exit_basis_proof,
                provider_error_type=provider_error_type,
                max_favorable_excursion=telemetry.max_favorable_excursion,
                max_adverse_excursion=telemetry.max_adverse_excursion,
                mfe_session_date=telemetry.mfe_session_date,
                mae_session_date=telemetry.mae_session_date,
                max_close_return=telemetry.max_close_return,
                min_close_return=telemetry.min_close_return,
                hit_t1_intraday=telemetry.hit_t1_intraday,
                hit_t2_intraday=telemetry.hit_t2_intraday,
                hit_t3_intraday=telemetry.hit_t3_intraday,
                hit_stop_intraday=telemetry.hit_stop_intraday,
                same_day_barrier_ambiguity=telemetry.same_day_barrier_ambiguity,
                provider=provider or resp.lineage.provider,
                endpoint=endpoint or resp.lineage.endpoint,
                provider_request=provider_request_payload or provider_request,
                data_lineage_ids=data_lineage_ids or lineage_ids,
            )

        if not resp.ok:
            err = resp.error
            reason = (
                f"provider_{getattr(err, 'error_type', 'error')}"
                if err else "pricing_unavailable"
            )
            return (
                build_result(
                    status=STATUS_PRICING_UNAVAILABLE_RETRY,
                    reason=reason,
                    entry_price=None,
                    exit_price=None,
                    forward_return=None,
                    provider_error_type=getattr(err, "error_type", None),
                ),
                payload_hash,
            )

        bars = list(resp.data or [])
        entry_bar = _find_bar(bars, plan.entry_session_date)
        exit_bar = _find_bar(bars, plan.exit_session_date)

        if entry_bar is None or entry_bar.open is None:
            return (
                build_result(
                    status=STATUS_MISSING_ENTRY_PRICE_RETRY,
                    reason="missing_entry_price",
                    entry_price=None,
                    exit_price=None,
                    forward_return=None,
                ),
                payload_hash,
            )

        entry_basis_proof = _split_adjusted_open_basis_proof(entry_bar)
        if entry_basis_proof is None:
            return (
                build_result(
                    status=STATUS_PRICING_UNAVAILABLE_RETRY,
                    reason="split_adjusted_open_basis_unproven",
                    entry_price=None,
                    exit_price=None,
                    forward_return=None,
                ),
                payload_hash,
            )

        entry_price = _finite_price(entry_bar.open)
        if entry_price is None or entry_price <= 0:
            return (
                build_result(
                    status=STATUS_INVALID_ENTRY_PRICE_RETRY,
                    reason="invalid_entry_price",
                    entry_price=entry_price,
                    exit_price=None,
                    forward_return=None,
                    entry_basis_proof=entry_basis_proof,
                ),
                payload_hash,
            )

        telemetry = _path_telemetry(
            bars,
            entry_price=entry_price,
            entry_session_date=plan.entry_session_date,
            exit_session_date=plan.exit_session_date,
            geometry=M4_EXIT_GEOMETRY,
        )

        if exit_bar is None or exit_bar.open is None:
            survivorship = self._resolve_missing_exit_survivorship(
                sig=sig,
                plan=plan,
                entry_price=entry_price,
                asof=asof,
                job_run_id=job_run_id,
            )
            decision = survivorship.decision
            forward_return = (
                (decision.exit_price - entry_price) / entry_price
                if decision.exit_price is not None
                else None
            )
            provider_request_payload = {
                "price_request": provider_request,
                "survivorship_request": survivorship.provider_request,
            }
            return (
                build_result(
                    status=decision.status,
                    reason=decision.reason,
                    entry_price=entry_price,
                    exit_price=decision.exit_price,
                    forward_return=forward_return,
                    entry_price_source=M4_PRICE_SOURCE,
                    exit_price_source=decision.exit_price_source,
                    entry_basis_proof=entry_basis_proof,
                    exit_basis_proof=decision.exit_basis_proof,
                    exit_data_lineage_id=(
                        survivorship.primary_data_lineage_id
                        or lineage.data_lineage_id
                    ),
                    telemetry=telemetry,
                    provider=survivorship.provider or resp.lineage.provider,
                    endpoint=survivorship.endpoint or resp.lineage.endpoint,
                    provider_request_payload=provider_request_payload,
                    data_lineage_ids=lineage_ids + survivorship.data_lineage_ids,
                ),
                payload_hash,
            )

        exit_basis_proof = _split_adjusted_open_basis_proof(exit_bar)
        if exit_basis_proof is None:
            return (
                build_result(
                    status=STATUS_PRICING_UNAVAILABLE_RETRY,
                    reason="split_adjusted_open_basis_unproven",
                    entry_price=entry_price,
                    exit_price=None,
                    forward_return=None,
                    entry_price_source=M4_PRICE_SOURCE,
                    entry_basis_proof=entry_basis_proof,
                    telemetry=telemetry,
                ),
                payload_hash,
            )

        exit_price = _finite_price(exit_bar.open)
        if exit_price is None or exit_price < 0:
            return (
                build_result(
                    status=STATUS_INVALID_EXIT_PRICE_RETRY,
                    reason="invalid_exit_price",
                    entry_price=entry_price,
                    exit_price=exit_price,
                    forward_return=None,
                    entry_price_source=M4_PRICE_SOURCE,
                    entry_basis_proof=entry_basis_proof,
                    exit_basis_proof=exit_basis_proof,
                    telemetry=telemetry,
                ),
                payload_hash,
            )

        forward_return = (exit_price - entry_price) / entry_price
        return (
            build_result(
                status=STATUS_COMPUTED,
                reason=plan.entry_resolution_reason,
                entry_price=entry_price,
                exit_price=exit_price,
                forward_return=forward_return,
                entry_price_source=M4_PRICE_SOURCE,
                exit_price_source=M4_PRICE_SOURCE,
                entry_basis_proof=entry_basis_proof,
                exit_basis_proof=exit_basis_proof,
                telemetry=telemetry,
            ),
            payload_hash,
        )

    def _resolve_missing_exit_survivorship(
        self,
        *,
        sig: SignalRegistry,
        plan: M4ForwardReturnPlan,
        entry_price: float,
        asof: datetime,
        job_run_id: Optional[str],
    ) -> SurvivorshipResolution:
        if self._survivorship_resolver is not None:
            return self._survivorship_resolver.resolve(
                session=self._session,
                adapter=self._adapter,
                ticker=sig.ticker,
                entry_session_date=plan.entry_session_date,
                exit_session_date=plan.exit_session_date,
                entry_price=entry_price,
                asof=asof,
                job_run_id=job_run_id,
            )
        return resolve_missing_exit_survivorship(
            session=self._session,
            adapter=self._adapter,
            ticker=sig.ticker,
            entry_session_date=plan.entry_session_date,
            exit_session_date=plan.exit_session_date,
            entry_price=entry_price,
            asof=asof,
            job_run_id=job_run_id,
        )

    def _persist_production_outcome(
        self,
        sig: SignalRegistry,
        *,
        plan: Optional[M4ForwardReturnPlan],
        pricing: ProductionPricingResult,
        attempts: int,
        job_run_id: Optional[str],
        entry_payload_hash: Optional[str] = None,
        exit_payload_hash: Optional[str] = None,
    ) -> None:
        if plan is None:
            try:
                decision_date = (
                    _parse_date(sig.trading_date) if sig.trading_date else date.min
                )
            except ValueError:
                decision_date = date.min
            plan = M4ForwardReturnPlan(
                decision_date=decision_date,
                next_execution_session=None,
                entry_session_date=decision_date,
                exit_session_date=decision_date,
                current_evidence_session_date=decision_date,
                mature=True,
                entry_resolution_reason="plan_unavailable",
            )
        input_hash = m4_forward_return_input_hash(sig, plan)
        outcome_hash = m4_forward_return_outcome_hash(
            sig,
            plan,
            pricing,
            entry_payload_hash=entry_payload_hash,
            exit_payload_hash=exit_payload_hash,
        )
        obs = (
            self._session.query(ForwardReturnObservation)
            .filter(
                ForwardReturnObservation.signal_id == sig.signal_id,
                ForwardReturnObservation.input_hash == input_hash,
            )
            .first()
        )
        if obs is None:
            obs = ForwardReturnObservation(
                signal_id=sig.signal_id,
                pattern_id=sig.pattern_id,
                ticker=sig.ticker,
                direction=sig.direction,
                signal_timestamp=sig.signal_timestamp,
                signal_horizon=sig.signal_horizon,
                input_hash=input_hash,
                outcome_hash=outcome_hash,
                status=pricing.status,
            )
            self._session.add(obs)

        obs.pattern_id = sig.pattern_id
        obs.ticker = sig.ticker
        obs.direction = sig.direction
        obs.signal_timestamp = sig.signal_timestamp
        obs.signal_horizon = sig.signal_horizon
        obs.next_execution_session = (
            plan.next_execution_session.isoformat()
            if plan.next_execution_session is not None else None
        )
        obs.entry_session_date = plan.entry_session_date.isoformat()
        obs.entry_price = pricing.entry_price
        obs.entry_price_source = pricing.entry_price_source
        obs.entry_basis_proof = pricing.entry_basis_proof
        obs.entry_data_lineage_id = pricing.entry_data_lineage_id
        obs.exit_session_date = plan.exit_session_date.isoformat()
        obs.exit_price = pricing.exit_price
        obs.exit_price_source = pricing.exit_price_source
        obs.exit_basis_proof = pricing.exit_basis_proof
        obs.exit_data_lineage_id = pricing.exit_data_lineage_id
        obs.forward_return = pricing.forward_return
        obs.max_favorable_excursion = pricing.max_favorable_excursion
        obs.max_adverse_excursion = pricing.max_adverse_excursion
        obs.mfe_session_date = pricing.mfe_session_date
        obs.mae_session_date = pricing.mae_session_date
        obs.max_close_return = pricing.max_close_return
        obs.min_close_return = pricing.min_close_return
        obs.hit_t1_intraday = pricing.hit_t1_intraday
        obs.hit_t2_intraday = pricing.hit_t2_intraday
        obs.hit_t3_intraday = pricing.hit_t3_intraday
        obs.hit_stop_intraday = pricing.hit_stop_intraday
        obs.same_day_barrier_ambiguity = pricing.same_day_barrier_ambiguity
        obs.status = pricing.status
        obs.reason = pricing.reason
        obs.attempts = attempts
        obs.job_run_id = job_run_id
        obs.outcome_hash = outcome_hash
        obs.data_lineage_ids = json.dumps(pricing.data_lineage_ids or [])
        obs.provider = pricing.provider
        obs.endpoint = pricing.endpoint
        obs.provider_request_json = (
            json.dumps(pricing.provider_request, sort_keys=True, default=str)
            if pricing.provider_request is not None else None
        )

        sig.forward_return = pricing.forward_return
        sig.forward_return_status = pricing.status
        sig.forward_return_attempts = attempts
        sig.outcome_unavailable_reason = pricing.reason
        sig.intended_entry_price = pricing.entry_price
        if pricing.status == STATUS_COMPUTED:
            sig.outcome_unavailable_reason = None

        self._session.flush()
        event = ForwardReturnObservationEvent(
            forward_return_observation_id=obs.forward_return_observation_id,
            signal_id=sig.signal_id,
            pattern_id=sig.pattern_id,
            ticker=sig.ticker,
            direction=sig.direction,
            signal_timestamp=sig.signal_timestamp,
            signal_horizon=sig.signal_horizon,
            next_execution_session=obs.next_execution_session,
            entry_session_date=obs.entry_session_date,
            entry_price=pricing.entry_price,
            entry_price_source=pricing.entry_price_source,
            entry_basis_proof=pricing.entry_basis_proof,
            entry_data_lineage_id=pricing.entry_data_lineage_id,
            exit_session_date=obs.exit_session_date,
            exit_price=pricing.exit_price,
            exit_price_source=pricing.exit_price_source,
            exit_basis_proof=pricing.exit_basis_proof,
            exit_data_lineage_id=pricing.exit_data_lineage_id,
            forward_return=pricing.forward_return,
            max_favorable_excursion=pricing.max_favorable_excursion,
            max_adverse_excursion=pricing.max_adverse_excursion,
            mfe_session_date=pricing.mfe_session_date,
            mae_session_date=pricing.mae_session_date,
            max_close_return=pricing.max_close_return,
            min_close_return=pricing.min_close_return,
            hit_t1_intraday=pricing.hit_t1_intraday,
            hit_t2_intraday=pricing.hit_t2_intraday,
            hit_t3_intraday=pricing.hit_t3_intraday,
            hit_stop_intraday=pricing.hit_stop_intraday,
            same_day_barrier_ambiguity=pricing.same_day_barrier_ambiguity,
            status=pricing.status,
            reason=pricing.reason,
            attempts=attempts,
            job_run_id=job_run_id,
            input_hash=input_hash,
            outcome_hash=outcome_hash,
            data_lineage_ids=json.dumps(pricing.data_lineage_ids or []),
            provider=pricing.provider,
            endpoint=pricing.endpoint,
            provider_request_json=(
                json.dumps(pricing.provider_request, sort_keys=True, default=str)
                if pricing.provider_request is not None else None
            ),
        )
        self._session.add(event)

    # ------------------------------------------------------------------
    # Legacy injected-price scaffold path
    # ------------------------------------------------------------------
    def _begin_attempt(self, sig: SignalRegistry) -> int:
        attempts = (sig.forward_return_attempts or 0) + 1
        sig.forward_return_attempts = attempts
        return attempts

    def _mark_unavailable(
        self,
        sig: SignalRegistry,
        *,
        retry_status: str,
        reason: str,
    ) -> bool:
        """Return True if retryable, False if terminal outcome_unavailable."""
        sig.outcome_unavailable_reason = reason
        if (sig.forward_return_attempts or 0) >= self._max_attempts:
            sig.forward_return_status = STATUS_OUTCOME_UNAVAILABLE
            return False
        sig.forward_return_status = retry_status
        return True

    def _run_injected_price_fn(self, ctx: JobContext) -> JobResult:
        pending = (
            self._session.query(SignalRegistry)
            .filter(
                or_(
                    SignalRegistry.forward_return_status.in_(
                        RETRYABLE_FORWARD_RETURN_STATUSES
                    ),
                    SignalRegistry.forward_return_status.is_(None),
                )
            )
            .all()
        )

        computed = 0
        unavailable = 0
        retryable_unavailable = 0
        immature = 0
        pricing_errors = 0

        for sig in pending:
            try:
                if self._maturity_fn and not self._maturity_fn(
                    sig.signal_timestamp, sig.signal_horizon
                ):
                    immature += 1
                    sig.forward_return_status = STATUS_PENDING
                    continue
            except Exception as exc:
                self._begin_attempt(sig)
                retryable = self._mark_unavailable(
                    sig,
                    retry_status=STATUS_PRICING_UNAVAILABLE_RETRY,
                    reason=f"maturity_fn_error:{type(exc).__name__}",
                )
                retryable_unavailable += int(retryable)
                unavailable += int(not retryable)
                pricing_errors += 1
                continue

            self._begin_attempt(sig)

            try:
                prices = self._price_fn(
                    sig.ticker, sig.signal_timestamp, sig.signal_horizon
                )
            except Exception as exc:
                retryable = self._mark_unavailable(
                    sig,
                    retry_status=STATUS_PRICING_UNAVAILABLE_RETRY,
                    reason=f"price_fn_error:{type(exc).__name__}",
                )
                retryable_unavailable += int(retryable)
                unavailable += int(not retryable)
                pricing_errors += 1
                continue

            if prices is None:
                retryable = self._mark_unavailable(
                    sig,
                    retry_status=STATUS_PRICING_UNAVAILABLE_RETRY,
                    reason="pricing_unavailable",
                )
                retryable_unavailable += int(retryable)
                unavailable += int(not retryable)
                continue

            price_pair = _price_pair(prices)
            if price_pair is None:
                retryable = self._mark_unavailable(
                    sig,
                    retry_status=STATUS_INVALID_PRICE_SHAPE_RETRY,
                    reason="invalid_price_shape",
                )
                retryable_unavailable += int(retryable)
                unavailable += int(not retryable)
                pricing_errors += 1
                continue

            raw_entry_price, raw_exit_price = price_pair
            entry_price = _finite_price(raw_entry_price)
            exit_price = _finite_price(raw_exit_price)

            if entry_price is None or entry_price <= 0:
                retryable = self._mark_unavailable(
                    sig,
                    retry_status=STATUS_INVALID_ENTRY_PRICE_RETRY,
                    reason="invalid_entry_price",
                )
                retryable_unavailable += int(retryable)
                unavailable += int(not retryable)
                continue

            if raw_exit_price is None:
                retryable = self._mark_unavailable(
                    sig,
                    retry_status=STATUS_MISSING_EXIT_PRICE_RETRY,
                    reason="missing_exit_price",
                )
                retryable_unavailable += int(retryable)
                unavailable += int(not retryable)
                continue

            if exit_price is None or exit_price < 0:
                retryable = self._mark_unavailable(
                    sig,
                    retry_status=STATUS_INVALID_EXIT_PRICE_RETRY,
                    reason="invalid_exit_price",
                )
                retryable_unavailable += int(retryable)
                unavailable += int(not retryable)
                continue

            sig.intended_entry_price = entry_price
            sig.forward_return = (exit_price - entry_price) / entry_price
            sig.forward_return_status = STATUS_COMPUTED
            sig.outcome_unavailable_reason = None
            computed += 1

        self._session.flush()

        return JobResult(
            status="finished",
            metrics={
                "total_pending": len(pending),
                "computed": computed,
                "unavailable": unavailable,
                "retryable_unavailable": retryable_unavailable,
                "immature": immature,
                "pricing_errors": pricing_errors,
            },
        )


def resolve_missing_exit_survivorship(
    *,
    session: Session,
    adapter: Any,
    ticker: str,
    entry_session_date: date,
    exit_session_date: date,
    entry_price: float,
    asof: datetime,
    job_run_id: Optional[str],
) -> SurvivorshipResolution:
    """Resolve a mature missing exit through explicit survivorship sources.

    Standard FMP /full price bars are intentionally excluded here. The first
    source is a test/future-provider ``get_survivorship_events`` hook. The
    production FMP source currently available is the paginated
    ``/stable/delisted-companies`` directory, which can prove a delisting date
    but not the economic reason; that lands in visible survivorship review.
    """
    if hasattr(adapter, "get_survivorship_events"):
        resolution = _resolve_from_survivorship_events(
            session=session,
            adapter=adapter,
            ticker=ticker,
            entry_session_date=entry_session_date,
            exit_session_date=exit_session_date,
            entry_price=entry_price,
            asof=asof,
            job_run_id=job_run_id,
        )
        if resolution is not None:
            return resolution

    if hasattr(adapter, "get_delisted_companies"):
        resolution = _resolve_from_fmp_delisted_companies(
            session=session,
            adapter=adapter,
            ticker=ticker,
            entry_session_date=entry_session_date,
            exit_session_date=exit_session_date,
            asof=asof,
            job_run_id=job_run_id,
        )
        if resolution is not None:
            return resolution

    return _survivorship_unresolved(
        reason="survivorship_resolver_no_supported_source",
        provider_request={
            "ticker": ticker,
            "from": entry_session_date.isoformat(),
            "to": exit_session_date.isoformat(),
            "event_basis": "missing_mature_exit_survivorship_review",
            "sources_attempted": [],
        },
    )


def _resolve_run_timestamp(
    explicit: Optional[datetime],
    param_value: Any,
    fallback: datetime,
) -> Tuple[datetime, Optional[str]]:
    value = explicit
    if value is None and param_value:
        try:
            value = datetime.fromisoformat(str(param_value).replace("Z", "+00:00"))
        except ValueError:
            return fallback, f"invalid run_timestamp: {param_value}"
    if value is None:
        value = fallback
    if value.tzinfo is None or value.utcoffset() is None:
        return value, "run_timestamp must be timezone-aware"
    return value.astimezone(timezone.utc), None


def _terminalize_if_needed(status: str, attempts: int, max_attempts: int) -> str:
    if status in (
        STATUS_COMPUTED,
        STATUS_PENDING,
        STATUS_HALTED_PENDING,
        STATUS_CORPORATE_ACTION_REVIEW,
        STATUS_SURVIVORSHIP_UNRESOLVED_REVIEW,
    ):
        return status
    if attempts >= max_attempts:
        return STATUS_OUTCOME_UNAVAILABLE
    return status


def _resolve_from_survivorship_events(
    *,
    session: Session,
    adapter: Any,
    ticker: str,
    entry_session_date: date,
    exit_session_date: date,
    entry_price: float,
    asof: datetime,
    job_run_id: Optional[str],
) -> Optional[SurvivorshipResolution]:
    provider_request = {
        "ticker": ticker,
        "from": entry_session_date.isoformat(),
        "to": exit_session_date.isoformat(),
        "event_basis": "missing_mature_exit_survivorship_review",
        "endpoint": "get_survivorship_events",
    }
    resp = adapter.get_survivorship_events(
        ticker,
        from_date=entry_session_date,
        to_date=exit_session_date,
        asof=asof,
    )
    payload = {
        "request": provider_request,
        "events": _jsonable_survivorship_payload(resp.data),
        "error": _provider_error_payload(resp.error),
    }
    lineage = _record_survivorship_lineage(
        session,
        resp=resp,
        raw_payload=payload,
        job_run_id=job_run_id,
    )
    lineage_ids = [lineage.data_lineage_id]
    if not resp.ok:
        return _survivorship_unresolved(
            reason=f"survivorship_source_error:{resp.error.error_type}",
            provider=resp.lineage.provider,
            endpoint=resp.lineage.endpoint,
            provider_request=provider_request,
            data_lineage_ids=lineage_ids,
            primary_data_lineage_id=lineage.data_lineage_id,
        )

    event = _first_survivorship_source_event(resp.data)
    if not event:
        return _survivorship_unresolved(
            reason="survivorship_unresolved_no_source_event",
            provider=resp.lineage.provider,
            endpoint=resp.lineage.endpoint,
            provider_request=provider_request,
            data_lineage_ids=lineage_ids,
            primary_data_lineage_id=lineage.data_lineage_id,
        )
    decision = _survivorship_decision(event, entry_price=entry_price)
    if decision is None:
        decision = SurvivorshipDecision(
            status=STATUS_SURVIVORSHIP_UNRESOLVED_REVIEW,
            reason="survivorship_event_unclassified_review",
            exit_price=None,
            exit_price_source=None,
            exit_basis_proof=None,
        )
    return SurvivorshipResolution(
        decision=decision,
        data_lineage_ids=lineage_ids,
        provider=resp.lineage.provider,
        endpoint=resp.lineage.endpoint,
        provider_request=provider_request,
        primary_data_lineage_id=lineage.data_lineage_id,
    )


def _resolve_from_fmp_delisted_companies(
    *,
    session: Session,
    adapter: Any,
    ticker: str,
    entry_session_date: date,
    exit_session_date: date,
    asof: datetime,
    job_run_id: Optional[str],
    max_pages: int = 10,
    page_limit: int = 100,
) -> Optional[SurvivorshipResolution]:
    lineage_ids: List[str] = []
    primary_lineage_id: Optional[str] = None
    provider = "FMP"
    endpoint = DELISTED_COMPANIES_ENDPOINT
    pages_examined = []
    normalized_ticker = ticker.upper()

    for page in range(max_pages):
        provider_request = {
            "ticker": ticker,
            "from": entry_session_date.isoformat(),
            "to": exit_session_date.isoformat(),
            "event_basis": "missing_mature_exit_survivorship_review",
            "endpoint": DELISTED_COMPANIES_ENDPOINT,
            "page": page,
            "limit": page_limit,
        }
        resp = adapter.get_delisted_companies(
            page=page,
            limit=page_limit,
            asof=asof,
        )
        provider = resp.lineage.provider
        endpoint = resp.lineage.endpoint
        payload = {
            "request": provider_request,
            "rows": _jsonable_survivorship_payload(resp.data),
            "error": _provider_error_payload(resp.error),
        }
        lineage = _record_survivorship_lineage(
            session,
            resp=resp,
            raw_payload=payload,
            job_run_id=job_run_id,
        )
        lineage_ids.append(lineage.data_lineage_id)
        primary_lineage_id = primary_lineage_id or lineage.data_lineage_id
        pages_examined.append({"page": page, "limit": page_limit})

        if not resp.ok:
            return _survivorship_unresolved(
                reason=f"survivorship_source_error:{resp.error.error_type}",
                provider=provider,
                endpoint=endpoint,
                provider_request={
                    "ticker": ticker,
                    "from": entry_session_date.isoformat(),
                    "to": exit_session_date.isoformat(),
                    "event_basis": "missing_mature_exit_survivorship_review",
                    "source": DELISTED_COMPANIES_ENDPOINT,
                    "pages_examined": pages_examined,
                },
                data_lineage_ids=lineage_ids,
                primary_data_lineage_id=primary_lineage_id,
            )

        rows = list(resp.data or [])
        for row in rows:
            symbol = str(getattr(row, "symbol", "") or "").upper()
            if symbol != normalized_ticker:
                continue
            delisted_date = _safe_parse_date(getattr(row, "delisted_date", None))
            reason = "delisting_unclassified_survivorship_review"
            if delisted_date is not None and delisted_date > exit_session_date:
                reason = "delisting_after_exit_window_survivorship_review"
            return SurvivorshipResolution(
                decision=SurvivorshipDecision(
                    status=STATUS_SURVIVORSHIP_UNRESOLVED_REVIEW,
                    reason=reason,
                    exit_price=None,
                    exit_price_source=None,
                    exit_basis_proof=None,
                ),
                data_lineage_ids=lineage_ids,
                provider=provider,
                endpoint=endpoint,
                provider_request={
                    "ticker": ticker,
                    "from": entry_session_date.isoformat(),
                    "to": exit_session_date.isoformat(),
                    "event_basis": "missing_mature_exit_survivorship_review",
                    "source": DELISTED_COMPANIES_ENDPOINT,
                    "pages_examined": pages_examined,
                    "matched_symbol": symbol,
                    "delisted_date": (
                        delisted_date.isoformat()
                        if delisted_date is not None else None
                    ),
                },
                primary_data_lineage_id=primary_lineage_id,
            )

        if not rows:
            break
        oldest = _oldest_delisted_date(rows)
        if oldest is not None and oldest < entry_session_date:
            break

    return _survivorship_unresolved(
        reason="survivorship_unresolved_no_source_event",
        provider=provider,
        endpoint=endpoint,
        provider_request={
            "ticker": ticker,
            "from": entry_session_date.isoformat(),
            "to": exit_session_date.isoformat(),
            "event_basis": "missing_mature_exit_survivorship_review",
            "source": DELISTED_COMPANIES_ENDPOINT,
            "pages_examined": pages_examined,
        },
        data_lineage_ids=lineage_ids,
        primary_data_lineage_id=primary_lineage_id,
    )


def _record_survivorship_lineage(
    session: Session,
    *,
    resp: Any,
    raw_payload: Dict[str, Any],
    job_run_id: Optional[str],
):
    return record_data_lineage(
        session,
        provider=resp.lineage.provider,
        endpoint=resp.lineage.endpoint,
        asof_timestamp=resp.lineage.asof_timestamp,
        raw_payload=raw_payload,
        raw_payload_hash=resp.lineage.raw_payload_hash or stable_hash(raw_payload),
        request_timestamp=resp.lineage.request_timestamp,
        freshness_seconds=resp.lineage.freshness_seconds,
        source_authority=resp.lineage.source_authority,
        data_quality_flags=resp.lineage.data_quality_flags,
        job_run_id=job_run_id,
    )


def _survivorship_unresolved(
    *,
    reason: str,
    provider: Optional[str] = None,
    endpoint: Optional[str] = None,
    provider_request: Optional[Dict[str, Any]] = None,
    data_lineage_ids: Optional[List[str]] = None,
    primary_data_lineage_id: Optional[str] = None,
) -> SurvivorshipResolution:
    return SurvivorshipResolution(
        decision=SurvivorshipDecision(
            status=STATUS_SURVIVORSHIP_UNRESOLVED_REVIEW,
            reason=reason,
            exit_price=None,
            exit_price_source=None,
            exit_basis_proof=None,
        ),
        data_lineage_ids=data_lineage_ids or [],
        provider=provider,
        endpoint=endpoint,
        provider_request=provider_request or {},
        primary_data_lineage_id=primary_data_lineage_id,
    )


def _find_bar(bars: List[FmpBar], session_date: date) -> Optional[FmpBar]:
    wanted = session_date.isoformat()
    for bar in bars:
        if bar.date == wanted:
            return bar
    return None


def _provider_request_payload(
    *,
    ticker: str,
    from_date: date,
    to_date: date,
) -> Dict[str, Any]:
    return {
        "symbol": ticker,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "endpoint": HISTORICAL_PRICE_FULL_ENDPOINT,
        "basis": "split_adjusted_ohlcv_full_endpoint",
        "price_field": "open",
    }


def _split_adjusted_open_basis_proof(bar: FmpBar) -> Optional[str]:
    """Return proof that FMP /full open is on the split-adjusted OHLC basis."""
    if bar.open is None:
        return None
    if bar.split_adjusted_close is None:
        return None
    return M4_SPLIT_ADJUSTED_OPEN_BASIS_PROOF


def _path_telemetry(
    bars: List[FmpBar],
    *,
    entry_price: float,
    entry_session_date: date,
    exit_session_date: date,
    geometry: M4ExitGeometry,
) -> PathTelemetry:
    window = []
    for bar in bars:
        try:
            bar_day = _parse_date(bar.date)
        except ValueError:
            continue
        if entry_session_date <= bar_day <= exit_session_date:
            window.append(bar)
    window.sort(key=lambda bar: bar.date)

    max_favorable = None
    max_adverse = None
    max_close = None
    min_close = None
    mfe_date = None
    mae_date = None
    hit_t1 = False
    hit_t2 = False
    hit_t3 = False
    hit_stop = False
    same_day_ambiguity = False

    for bar in window:
        high = _finite_price(bar.high)
        low = _finite_price(bar.low)
        close = _finite_price(bar.close)
        high_return = _bar_return(high, entry_price)
        low_return = _bar_return(low, entry_price)
        close_return = _bar_return(close, entry_price)

        if high_return is not None:
            if max_favorable is None or high_return > max_favorable:
                max_favorable = high_return
                mfe_date = bar.date
            hit_t1 = hit_t1 or _return_at_or_above(
                high_return, geometry.t1_return
            )
            hit_t2 = hit_t2 or _return_at_or_above(
                high_return, geometry.t2_return
            )
            hit_t3 = hit_t3 or _return_at_or_above(
                high_return, geometry.t3_return
            )
        if low_return is not None:
            if max_adverse is None or low_return < max_adverse:
                max_adverse = low_return
                mae_date = bar.date
            hit_stop = hit_stop or _return_at_or_below(
                low_return,
                geometry.hard_stop_return,
            )
        if high_return is not None and low_return is not None:
            same_day_ambiguity = same_day_ambiguity or (
                _return_at_or_above(high_return, geometry.t1_return)
                and _return_at_or_below(low_return, geometry.hard_stop_return)
            )
        if close_return is not None:
            if max_close is None or close_return > max_close:
                max_close = close_return
            if min_close is None or close_return < min_close:
                min_close = close_return

    return PathTelemetry(
        max_favorable_excursion=max_favorable,
        max_adverse_excursion=max_adverse,
        mfe_session_date=mfe_date,
        mae_session_date=mae_date,
        max_close_return=max_close,
        min_close_return=min_close,
        hit_t1_intraday=hit_t1,
        hit_t2_intraday=hit_t2,
        hit_t3_intraday=hit_t3,
        hit_stop_intraday=hit_stop,
        same_day_barrier_ambiguity=same_day_ambiguity,
    )


def _bar_return(value: Optional[float], entry_price: float) -> Optional[float]:
    if value is None:
        return None
    return (value - entry_price) / entry_price


def _return_at_or_above(value: float, threshold: float) -> bool:
    return value + RETURN_COMPARISON_EPSILON >= threshold


def _return_at_or_below(value: float, threshold: float) -> bool:
    return value - RETURN_COMPARISON_EPSILON <= threshold


def _survivorship_decision(
    raw_flags: Any,
    *,
    entry_price: float,
) -> Optional[SurvivorshipDecision]:
    flags = _quality_flags(raw_flags)
    if not flags:
        return None
    event = _first_event(flags)
    if not event:
        return None

    event_type = _clean_lower(
        event.get("type")
        or event.get("event_type")
        or event.get("status")
        or flags.get("status")
    )
    classification = _clean_lower(
        event.get("classification")
        or event.get("reason")
        or event.get("delisting_reason")
    )
    source_backed = event.get("source_backed", True) is not False
    if not source_backed:
        return SurvivorshipDecision(
            status=STATUS_SURVIVORSHIP_UNRESOLVED_REVIEW,
            reason="survivorship_event_not_source_backed",
            exit_price=None,
            exit_price_source=None,
            exit_basis_proof=None,
        )

    if "halt" in event_type or "suspension" in event_type:
        if event.get("may_resume", True) is not False:
            return SurvivorshipDecision(
                status=STATUS_HALTED_PENDING,
                reason="active_halt_or_suspension",
                exit_price=None,
                exit_price_source=None,
                exit_basis_proof=None,
            )
        return SurvivorshipDecision(
            status=STATUS_CORPORATE_ACTION_REVIEW,
            reason="resolved_halt_requires_review",
            exit_price=None,
            exit_price_source=None,
            exit_basis_proof=None,
        )

    realized_payoff = _finite_price(
        event.get("realized_payoff")
        if "realized_payoff" in event
        else event.get("cash_value")
    )
    if realized_payoff is not None and realized_payoff >= 0:
        return SurvivorshipDecision(
            status=STATUS_COMPUTED,
            reason="corporate_action_realized_payoff",
            exit_price=realized_payoff,
            exit_price_source="source_backed_realized_payoff",
            exit_basis_proof="source_backed_corporate_action_payoff",
        )

    performance_text = f"{event_type} {classification}"
    performance_delisting = any(token in performance_text for token in (
        "performance",
        "bankrupt",
        "liquidat",
        "deficien",
        "insolv",
    ))
    if performance_delisting and source_backed:
        terminal_value = _finite_price(
            event.get("terminal_value")
            if "terminal_value" in event
            else event.get("terminal_price")
        )
        if terminal_value is None or terminal_value < 0:
            terminal_value = 0.0
            reason = "performance_delisting_shumway_terminal_loss"
        else:
            reason = "performance_delisting_terminal_value"
        return SurvivorshipDecision(
            status=STATUS_COMPUTED,
            reason=reason,
            exit_price=terminal_value,
            exit_price_source="source_backed_terminal_value",
            exit_basis_proof="source_backed_survivorship_terminal_value",
        )

    if any(token in event_type for token in (
        "corporate",
        "merger",
        "acquisition",
        "ticker_change",
        "symbol_change",
        "delist",
    )):
        return SurvivorshipDecision(
            status=STATUS_CORPORATE_ACTION_REVIEW,
            reason="corporate_action_review",
            exit_price=None,
            exit_price_source=None,
            exit_basis_proof=None,
        )

    return None


def _quality_flags(raw_flags: Any) -> Dict[str, Any]:
    if isinstance(raw_flags, dict):
        return raw_flags
    if isinstance(raw_flags, str) and raw_flags.strip():
        try:
            parsed = json.loads(raw_flags)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _first_event(flags: Dict[str, Any]) -> Dict[str, Any]:
    for key in (
        "terminal_event",
        "corporate_action",
        "market_status",
        "survivorship_event",
    ):
        value = flags.get(key)
        if isinstance(value, dict):
            return value
    if any(key in flags for key in (
        "type",
        "event_type",
        "status",
        "classification",
    )):
        return flags
    return {}


def _clean_lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _first_survivorship_source_event(raw_events: Any) -> Dict[str, Any]:
    if isinstance(raw_events, dict):
        return raw_events
    if isinstance(raw_events, list):
        for event in raw_events:
            if isinstance(event, dict):
                return event
    return {}


def _jsonable_survivorship_payload(payload: Any) -> Any:
    if isinstance(payload, list):
        return [_jsonable_survivorship_payload(item) for item in payload]
    if isinstance(payload, dict):
        return dict(payload)
    raw = getattr(payload, "raw", None)
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    return payload


def _provider_error_payload(error: Any) -> Optional[Dict[str, Any]]:
    if error is None:
        return None
    return {
        "provider": getattr(error, "provider", None),
        "endpoint": getattr(error, "endpoint", None),
        "status_code": getattr(error, "status_code", None),
        "error_type": getattr(error, "error_type", None),
        "message": getattr(error, "message", None),
        "retryable": getattr(error, "retryable", None),
    }


def _safe_parse_date(value: Any) -> Optional[date]:
    if value in (None, ""):
        return None
    try:
        return _parse_date(str(value))
    except ValueError:
        return None


def _oldest_delisted_date(rows: List[Any]) -> Optional[date]:
    dates = [
        parsed for row in rows
        if (parsed := _safe_parse_date(getattr(row, "delisted_date", None)))
        is not None
    ]
    if not dates:
        return None
    return min(dates)


def _price_lineage_payload(
    bars: Any,
    *,
    ticker: str,
    from_date: date,
    to_date: date,
) -> Dict[str, Any]:
    sorted_bars = sorted(list(bars or []), key=lambda bar: bar.date)
    return {
        "ticker": ticker,
        "request": _provider_request_payload(
            ticker=ticker,
            from_date=from_date,
            to_date=to_date,
        ),
        "bars": [
            {
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "split_adjusted_close": bar.split_adjusted_close,
                "adj_close": bar.adj_close,
            }
            for bar in sorted_bars
        ],
    }
