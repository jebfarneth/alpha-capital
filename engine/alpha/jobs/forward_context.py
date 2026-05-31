"""Forward-context panel collection for live M4 signals.

The panel is observational evidence: signal x forward session x provider
context. It is deliberately parallel to entry feature generation and forward
return pricing, and must never feed signal_identity_hash construction.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from alpha.assembly.signal_context import (
    SOURCE_CONTEXT_VERSION,
    build_m4_signal_context,
)
from alpha.data.contracts import stable_hash
from alpha.db.models import ForwardContextPathRow, SignalRegistry
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.forward_return import (
    M4_PATTERN_ID,
    m4_entry_exit_plan,
)
from alpha.market_calendar import (
    is_us_equity_session,
    next_us_equity_session,
    resolve_us_equity_session,
    us_equity_session_close_timestamp,
)
from alpha.patterns.contracts import PatternInput
from alpha.security_identity import (
    load_security_identity_by_ticker,
    security_identity_lineage_ids,
    security_identity_payload,
)

FORWARD_CONTEXT_VERSION = "m4-forward-context-panel-v1"
JOB_NAME = "forward_context_panel_collector"
REQUIRED_CONTEXT_PROVIDERS = ("polygon", "benzinga")
USABLE_PROVIDER_STATUSES = {"matched", "no_data", "pit_excluded"}
HARD_PROVIDER_FAILURE_STATUSES = {
    "provider_error",
    "parse_error",
    "validation_error",
    "unavailable",
}


@dataclass(frozen=True)
class ForwardContextPlan:
    """Resolved per-signal forward-context capture plan for one session."""

    signal: SignalRegistry
    forward_session_date: date
    asof_timestamp: datetime
    entry_session_date: date
    exit_session_date: date
    path_sequence: int
    is_terminal_snapshot: bool


class ForwardContextCollectorJob(BaseJob):
    """Snapshot rich provider context for open M4 signal forward windows."""

    job_name = JOB_NAME
    job_type = "context_panel"

    def __init__(
        self,
        session: Session,
        *,
        polygon_adapter: Any = None,
        benzinga_adapter: Any = None,
        run_timestamp: Optional[datetime] = None,
        pattern_id: str = M4_PATTERN_ID,
    ):
        self._session = session
        self._polygon_adapter = polygon_adapter
        self._benzinga_adapter = benzinga_adapter
        self._run_timestamp = run_timestamp
        self._pattern_id = pattern_id
        self._identity_cache: Dict[str, Dict[str, Any]] = {}

    def run(self, ctx: JobContext) -> JobResult:
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

        (
            forward_session,
            asof_timestamp,
            clock_metrics,
            no_op_reason,
            session_error,
        ) = _resolve_forward_session(
            run_timestamp,
            ctx.params.get("forward_session_date"),
        )
        if session_error:
            return JobResult(
                status="failed",
                metrics={"run_timestamp": run_timestamp.isoformat()},
                errors=[{"stage": "params", "message": session_error}],
            )
        if no_op_reason:
            return JobResult(
                status="finished",
                metrics={
                    **clock_metrics,
                    "pattern_id": self._pattern_id,
                    "no_op_reason": no_op_reason,
                    "eligible_signal_count": 0,
                    "rows_inserted": 0,
                    "rows_existing": 0,
                },
            )

        assert forward_session is not None
        assert asof_timestamp is not None

        signals = _load_active_signals(self._session, self._pattern_id)
        plans, skipped = _build_capture_plans(
            signals,
            forward_session=forward_session,
            asof_timestamp=asof_timestamp,
            run_timestamp=run_timestamp,
        )
        existing_ids = _existing_signal_ids(
            self._session,
            signal_ids=[plan.signal.signal_id for plan in plans],
            forward_session_date=forward_session.isoformat(),
        )
        pending = [plan for plan in plans if plan.signal.signal_id not in existing_ids]
        missing_adapters = _missing_required_context_adapters(
            polygon_adapter=self._polygon_adapter,
            benzinga_adapter=self._benzinga_adapter,
        )
        if pending and missing_adapters:
            metrics = {
                **clock_metrics,
                "schema_version": FORWARD_CONTEXT_VERSION,
                "source_context_version": SOURCE_CONTEXT_VERSION,
                "pattern_id": self._pattern_id,
                "active_signal_count": len(signals),
                "eligible_signal_count": len(plans),
                "pending_signal_count": len(pending),
                "rows_existing": len(existing_ids),
                "rows_inserted": 0,
                "ticker_fetch_count": 0,
                "skipped_signal_count": sum(skipped.values()),
                "skipped_signals": dict(sorted(skipped.items())),
                "missing_required_adapters": missing_adapters,
            }
            return JobResult(
                status="failed",
                metrics=metrics,
                errors=[{
                    "stage": "provider_adapters",
                    "message": (
                        "required provider adapter(s) absent for pending "
                        "forward-context capture: "
                        + ", ".join(missing_adapters)
                    ),
                }],
            )
        by_ticker = _group_plans_by_ticker(pending)

        rows_inserted = 0
        source_attempt_status_counts: Dict[str, int] = {}
        required_provider_status_counts: Dict[str, Dict[str, int]] = {
            provider: {} for provider in REQUIRED_CONTEXT_PROVIDERS
        }
        lineage_id_count = 0
        row_payloads: List[Dict[str, Any]] = []
        degraded_signals: List[Dict[str, Any]] = []
        dead_provider_counts: Dict[str, int] = {}

        for ticker, ticker_plans in by_ticker.items():
            base_context, base_lineage_ids = self._build_ticker_context(
                ticker_plans[0],
                ticker=ticker,
                job_run_id=ctx.job_run_id,
            )
            for plan in ticker_plans:
                identity_payload, identity_lineage_ids = self._identity_for_signal(
                    plan.signal
                )
                context = _context_for_signal(
                    base_context,
                    plan,
                    identity_payload=identity_payload,
                )
                attempts = _source_attempts(context)
                for status, count in _attempt_status_counts(attempts).items():
                    source_attempt_status_counts[status] = (
                        source_attempt_status_counts.get(status, 0) + count
                    )
                provider_status_counts = _required_provider_status_counts(context)
                _merge_provider_status_counts(
                    required_provider_status_counts,
                    provider_status_counts,
                )
                dead_providers = _dead_required_providers(provider_status_counts)
                if dead_providers:
                    degraded_signals.append({
                        "signal_id": plan.signal.signal_id,
                        "ticker": plan.signal.ticker.upper(),
                        "dead_providers": dead_providers,
                    })
                    for provider in dead_providers:
                        dead_provider_counts[provider] = (
                            dead_provider_counts.get(provider, 0) + 1
                        )
                    continue
                data_lineage_ids = list(dict.fromkeys(
                    base_lineage_ids
                    + identity_lineage_ids
                    + _lineage_ids_from_attempts(attempts)
                ))
                lineage_id_count += len(data_lineage_ids)
                context_hash = stable_hash(context)
                row_payloads.append({
                    "signal_id": plan.signal.signal_id,
                    "pattern_id": plan.signal.pattern_id,
                    "ticker": plan.signal.ticker.upper(),
                    "signal_horizon": plan.signal.signal_horizon,
                    "forward_session_date": plan.forward_session_date.isoformat(),
                    "path_sequence": plan.path_sequence,
                    "asof_timestamp": plan.asof_timestamp,
                    "context_json": json.dumps(context, sort_keys=True, default=str),
                    "source_attempts_json": json.dumps(
                        attempts,
                        sort_keys=True,
                        default=str,
                    ),
                    "data_lineage_ids": json.dumps(data_lineage_ids),
                    "context_hash": context_hash,
                    "is_terminal_snapshot": plan.is_terminal_snapshot,
                    "job_run_id": ctx.job_run_id,
                })

        for payload in row_payloads:
            self._session.add(ForwardContextPathRow(**payload))
        rows_inserted = len(row_payloads)

        # The production VM runs under a single-instance flock; uniqueness stays the loud duplicate-run backstop.
        self._session.flush()
        metrics = {
            **clock_metrics,
            "schema_version": FORWARD_CONTEXT_VERSION,
            "source_context_version": SOURCE_CONTEXT_VERSION,
            "pattern_id": self._pattern_id,
            "active_signal_count": len(signals),
            "eligible_signal_count": len(plans),
            "pending_signal_count": len(pending),
            "rows_existing": len(existing_ids),
            "rows_inserted": rows_inserted,
            "ticker_fetch_count": len(by_ticker),
            "skipped_signal_count": sum(skipped.values()),
            "skipped_signals": dict(sorted(skipped.items())),
            "source_attempt_status_counts": dict(
                sorted(source_attempt_status_counts.items())
            ),
            "required_provider_status_counts": _sorted_provider_status_counts(
                required_provider_status_counts
            ),
            "degraded_signal_count": len(degraded_signals),
            "dead_providers": dict(sorted(dead_provider_counts.items())),
            "degraded_signals": degraded_signals[:20],
            "data_lineage_id_count": lineage_id_count,
        }
        status = "failed" if degraded_signals else "finished"
        errors = []
        if degraded_signals:
            errors.append({
                "stage": "provider_quality",
                "message": (
                    "required provider dead for some pending forward-context "
                    "captures; healthy panel rows were written and degraded "
                    "slots were left open"
                ),
                "dead_providers": dict(sorted(dead_provider_counts.items())),
                "degraded_signal_count": len(degraded_signals),
            })
        return JobResult(
            status=status,
            metrics=metrics,
            errors=errors,
            input_hashes={
                "forward_session_date": forward_session.isoformat(),
                "pattern_id": self._pattern_id,
            },
            output_hashes={
                "forward_context_rows": stable_hash({
                    "forward_session_date": forward_session.isoformat(),
                    "rows_inserted": rows_inserted,
                    "rows_existing": len(existing_ids),
                    "eligible_signal_count": len(plans),
                    "degraded_signal_count": len(degraded_signals),
                }),
            },
        )

    def _build_ticker_context(
        self,
        plan: ForwardContextPlan,
        *,
        ticker: str,
        job_run_id: str,
    ) -> Tuple[Dict[str, Any], List[str]]:
        identity_payload, _identity_lineage_ids = self._identity_for_signal(plan.signal)
        inp = PatternInput(
            ticker=ticker,
            asof_timestamp=plan.asof_timestamp,
            market_data={"security_identity": identity_payload},
        )
        context, lineage_refs = build_m4_signal_context(
            inp,
            session=self._session,
            polygon_adapter=self._polygon_adapter,
            benzinga_adapter=self._benzinga_adapter,
            cutoff_timestamp=plan.asof_timestamp,
            decision_date=plan.signal.trading_date or plan.forward_session_date.isoformat(),
            evidence_session_date=plan.forward_session_date.isoformat(),
            evidence_day=plan.forward_session_date,
            job_run_id=job_run_id,
        )
        return context, [lineage_id for lineage_id, _hash in lineage_refs if lineage_id]

    def _identity_for_signal(
        self,
        signal: SignalRegistry,
    ) -> Tuple[Dict[str, Any], List[str]]:
        scan_id = signal.scan_id
        identity = None
        if scan_id:
            if scan_id not in self._identity_cache:
                self._identity_cache[scan_id] = load_security_identity_by_ticker(
                    self._session,
                    scan_id,
                )
            identity = self._identity_cache[scan_id].get(signal.ticker.upper())
        return security_identity_payload(identity), security_identity_lineage_ids(identity)


def forward_context_rows_through(
    session: Session,
    *,
    signal_id: str,
    decision_session_date: str,
) -> List[ForwardContextPathRow]:
    """Return panel rows available through a consumer's decision session.

    This is the sanctioned reader for live consumers that need a PIT boundary:
    it never returns panel rows after ``decision_session_date``.
    """

    decision_day = _parse_session_date(decision_session_date)
    asof_ceiling = us_equity_session_close_timestamp(decision_day)
    with session.no_autoflush:
        return (
            session.query(ForwardContextPathRow)
            .filter(
                ForwardContextPathRow.signal_id == signal_id,
                ForwardContextPathRow.forward_session_date
                <= decision_session_date,
                ForwardContextPathRow.asof_timestamp <= asof_ceiling,
            )
            .order_by(ForwardContextPathRow.path_sequence)
            .all()
        )


def _resolve_run_timestamp(
    explicit: Optional[datetime],
    param_value: Any,
    fallback: datetime,
) -> Tuple[datetime, Optional[str]]:
    value = explicit
    if value is None and param_value:
        try:
            raw = str(param_value).replace("Z", "+00:00")
            value = datetime.fromisoformat(raw)
        except ValueError:
            return fallback, f"invalid run_timestamp: {param_value}"
    if value is None:
        value = fallback
    if value.tzinfo is None or value.utcoffset() is None:
        return value, "run_timestamp must be timezone-aware"
    return value.astimezone(timezone.utc), None


def _resolve_forward_session(
    run_timestamp: datetime,
    override: Any,
) -> Tuple[
    Optional[date],
    Optional[datetime],
    Dict[str, Any],
    Optional[str],
    Optional[str],
]:
    resolution = resolve_us_equity_session(run_timestamp)
    metrics = {
        "decision_date": resolution.decision_date,
        "evidence_session_date": resolution.evidence_session_date,
        "next_execution_session": resolution.next_execution_session,
        "run_timestamp": run_timestamp.isoformat(),
    }
    if override:
        try:
            forward_session = _parse_session_date(str(override))
        except ValueError as exc:
            return None, None, metrics, None, str(exc)
        current_evidence = date.fromisoformat(resolution.evidence_session_date)
        if forward_session > current_evidence:
            return (
                None,
                None,
                metrics,
                None,
                (
                    f"forward_session_date {forward_session.isoformat()} is in "
                    f"the future relative to current evidence session "
                    f"{current_evidence.isoformat()}"
                ),
            )
        if forward_session != current_evidence:
            return (
                None,
                None,
                metrics,
                None,
                (
                    f"forward_session_date {forward_session.isoformat()} does "
                    "not match the current completed evidence session "
                    f"{current_evidence.isoformat()}"
                ),
            )
    else:
        decision_day = date.fromisoformat(resolution.decision_date)
        if (
            not is_us_equity_session(decision_day)
            or resolution.decision_date != resolution.evidence_session_date
        ):
            return (
                None,
                None,
                metrics,
                "no_current_completed_session",
                None,
            )
        forward_session = decision_day

    asof = us_equity_session_close_timestamp(forward_session)
    metrics.update({
        "forward_session_date": forward_session.isoformat(),
        "asof_timestamp": asof.isoformat(),
    })
    return forward_session, asof, metrics, None, None


def _parse_session_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"forward_session_date {value!r} is not a valid ISO date") from exc
    if not is_us_equity_session(parsed):
        raise ValueError(
            f"forward_session_date {value} is not a regular U.S. equity session"
        )
    return parsed


def _missing_required_context_adapters(
    *,
    polygon_adapter: Any,
    benzinga_adapter: Any,
) -> List[str]:
    missing: List[str] = []
    if polygon_adapter is None:
        missing.append("polygon")
    if benzinga_adapter is None:
        missing.append("benzinga")
    return missing


def _load_active_signals(session: Session, pattern_id: str) -> List[SignalRegistry]:
    return (
        session.query(SignalRegistry)
        .filter(
            SignalRegistry.pattern_id == pattern_id,
            SignalRegistry.signal_status == "active",
        )
        .order_by(
            SignalRegistry.ticker.asc(),
            SignalRegistry.trading_date.asc().nullslast(),
            SignalRegistry.signal_id.asc(),
        )
        .all()
    )


def _build_capture_plans(
    signals: Sequence[SignalRegistry],
    *,
    forward_session: date,
    asof_timestamp: datetime,
    run_timestamp: datetime,
) -> Tuple[List[ForwardContextPlan], Dict[str, int]]:
    plans: List[ForwardContextPlan] = []
    skipped: Dict[str, int] = {}
    for signal in signals:
        try:
            signal_date = date.fromisoformat(str(signal.trading_date))
        except (TypeError, ValueError):
            _increment(skipped, "missing_or_invalid_signal_trading_date")
            continue
        try:
            next_execution = (
                date.fromisoformat(str(signal.next_execution_session))
                if signal.next_execution_session else None
            )
        except ValueError:
            _increment(skipped, "invalid_next_execution_session")
            continue
        plan = m4_entry_exit_plan(
            decision_date=signal_date,
            next_execution_session=next_execution,
            current_evidence_session_date=forward_session,
            run_timestamp=run_timestamp,
        )
        sequence = _path_sequence(plan.entry_session_date, forward_session)
        if sequence is None:
            _increment(skipped, "before_entry_session")
            continue
        if forward_session > plan.exit_session_date:
            _increment(skipped, "after_exit_session")
            continue
        plans.append(ForwardContextPlan(
            signal=signal,
            forward_session_date=forward_session,
            asof_timestamp=asof_timestamp,
            entry_session_date=plan.entry_session_date,
            exit_session_date=plan.exit_session_date,
            path_sequence=sequence,
            is_terminal_snapshot=forward_session == plan.exit_session_date,
        ))
    return plans, skipped


def _path_sequence(entry_session: date, forward_session: date) -> Optional[int]:
    if forward_session < entry_session:
        return None
    sequence = 1
    cursor = next_us_equity_session(entry_session)
    while cursor < forward_session:
        cursor = next_us_equity_session(cursor + timedelta(days=1))
        sequence += 1
    return sequence if cursor == forward_session else None


def _existing_signal_ids(
    session: Session,
    *,
    signal_ids: Sequence[str],
    forward_session_date: str,
) -> set[str]:
    if not signal_ids:
        return set()
    rows = (
        session.query(ForwardContextPathRow.signal_id)
        .filter(
            ForwardContextPathRow.signal_id.in_(signal_ids),
            ForwardContextPathRow.forward_session_date == forward_session_date,
        )
        .all()
    )
    return {row[0] for row in rows}


def _group_plans_by_ticker(
    plans: Sequence[ForwardContextPlan],
) -> Dict[str, List[ForwardContextPlan]]:
    grouped: Dict[str, List[ForwardContextPlan]] = {}
    for plan in plans:
        grouped.setdefault(plan.signal.ticker.upper(), []).append(plan)
    return grouped


def _context_for_signal(
    base_context: Dict[str, Any],
    plan: ForwardContextPlan,
    *,
    identity_payload: Dict[str, Any],
) -> Dict[str, Any]:
    context = deepcopy(base_context)
    context["schema_version"] = FORWARD_CONTEXT_VERSION
    context["source_context_version"] = SOURCE_CONTEXT_VERSION
    context["context_role"] = "forward_context_panel"
    context["ticker"] = plan.signal.ticker.upper()
    context["decision_date"] = (
        plan.signal.trading_date or plan.forward_session_date.isoformat()
    )
    context["evidence_session_date"] = plan.forward_session_date.isoformat()
    context["asof_timestamp"] = plan.asof_timestamp.isoformat()
    context["signal_id"] = plan.signal.signal_id
    context["signal_trading_date"] = plan.signal.trading_date
    context["signal_timestamp"] = _iso(plan.signal.signal_timestamp)
    context["forward_session_date"] = plan.forward_session_date.isoformat()
    context["path_sequence"] = plan.path_sequence
    context["entry_session_date"] = plan.entry_session_date.isoformat()
    context["exit_session_date"] = plan.exit_session_date.isoformat()
    context["is_terminal_snapshot"] = plan.is_terminal_snapshot
    context["identity"] = _identity_context(plan.signal.ticker.upper(), identity_payload)
    return context


def _identity_context(ticker: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    status = payload.get("identity_status") or payload.get("status") or "unavailable"
    context_status = "present" if status == "present" else (
        "missing" if status else "unavailable"
    )
    return {
        "status": context_status,
        "security_identity": payload,
        "source_attempts": [{
            "source": "Polygon identity",
            "status": context_status,
            "row_count": 1 if context_status == "present" else 0,
            "eligible_row_count": 1 if context_status == "present" else 0,
            "pit_excluded_row_count": 0,
            "lineage_id": None,
            "endpoint": "security_identity_snapshot",
            "query": {"ticker": ticker},
            "warnings": {},
        }],
    }


def _source_attempts(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    for value in context.values():
        if isinstance(value, dict):
            source_attempts = value.get("source_attempts")
            if isinstance(source_attempts, list):
                attempts.extend(
                    item for item in source_attempts if isinstance(item, dict)
                )
    return attempts


def _attempt_status_counts(attempts: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for attempt in attempts:
        status = attempt.get("status")
        if not status:
            continue
        counts[str(status)] = counts.get(str(status), 0) + 1
    return counts


def _lineage_ids_from_attempts(attempts: Iterable[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for attempt in attempts:
        value = attempt.get("lineage_id")
        if isinstance(value, str) and value and value not in ids:
            ids.append(value)
    return ids


def _required_provider_status_counts(
    context: Dict[str, Any],
) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {
        provider: {} for provider in REQUIRED_CONTEXT_PROVIDERS
    }
    for key, value in context.items():
        provider = _provider_for_context_key(str(key))
        if provider is None or not isinstance(value, dict):
            continue
        source_attempts = value.get("source_attempts")
        if not isinstance(source_attempts, list):
            continue
        for attempt in source_attempts:
            if not isinstance(attempt, dict):
                continue
            status = attempt.get("status")
            if not status:
                continue
            provider_counts = counts[provider]
            status_key = str(status)
            provider_counts[status_key] = provider_counts.get(status_key, 0) + 1
    return counts


def _provider_for_context_key(key: str) -> Optional[str]:
    for provider in REQUIRED_CONTEXT_PROVIDERS:
        if key.startswith(f"{provider}_"):
            return provider
    return None


def _merge_provider_status_counts(
    target: Dict[str, Dict[str, int]],
    source: Dict[str, Dict[str, int]],
) -> None:
    for provider, status_counts in source.items():
        provider_counts = target.setdefault(provider, {})
        for status, count in status_counts.items():
            provider_counts[status] = provider_counts.get(status, 0) + count


def _dead_required_providers(
    provider_status_counts: Dict[str, Dict[str, int]],
) -> List[str]:
    dead: List[str] = []
    for provider in REQUIRED_CONTEXT_PROVIDERS:
        counts = provider_status_counts.get(provider, {})
        usable = sum(counts.get(status, 0) for status in USABLE_PROVIDER_STATUSES)
        if usable > 0:
            continue
        hard = sum(counts.get(status, 0) for status in HARD_PROVIDER_FAILURE_STATUSES)
        total = sum(counts.values())
        if total == 0 or (hard > 0 and hard == total):
            dead.append(provider)
    return dead


def _sorted_provider_status_counts(
    provider_status_counts: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    return {
        provider: dict(sorted(provider_status_counts.get(provider, {}).items()))
        for provider in REQUIRED_CONTEXT_PROVIDERS
    }


def _increment(counts: Dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _iso(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value
