"""Shared PIT-clean I12 volume-participation tradeability helpers."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Callable, Mapping


DEFAULT_VOLUME_PARTICIPATION_THRESHOLD = 0.05
VOLUME_TRADEABILITY_OK_STATUS = "tradeable_volume"
VOLUME_TRADEABILITY_SKIP_STATUS = "skipped_cash"
VOLUME_WINDOW_BASIS_PRE_DECISION = "pre_decision_completed_minutes"
VOLUME_PRICE_BASIS_LAST_PRE_DECISION = "last_predecision_price_proxy"
HALT_CONDITIONS = frozenset({"H", "T1", "T2", "T5", "HALT", "HALTED"})


def volume_tradeability_for_cost(
    *,
    candidate: Any | None,
    entry_quote: Any | None,
    exit_quote: Any | None,
    cost: Any,
    threshold: float,
    evidence_getter: Callable[[Any | None, Any | None], dict[str, Any]],
) -> dict[str, Any]:
    evidence = evidence_getter(candidate, entry_quote)
    entry_reason = quote_quality_skip_reason(
        entry_quote,
        missing_reason="entry_quote_missing",
        stale_reason="entry_quote_stale",
        error_reason="halt_or_bad_quote",
        max_spread_bps=cost.max_spread_bps,
    )
    participation_rate = None
    share_participation_rate = None
    observed_share_volume = evidence["entry_window_share_volume_denominator"]
    if entry_quote is not None and entry_quote.ask and entry_quote.ask > 0:
        if evidence["entry_window_dollar_volume"]:
            participation_rate = (
                cost.intended_order_usd / evidence["entry_window_dollar_volume"]
            )
        if observed_share_volume:
            intended_shares = cost.intended_order_usd / entry_quote.ask
            share_participation_rate = intended_shares / observed_share_volume
    if entry_reason == "none":
        if evidence["volume_evidence_available"] is not True:
            skipped_reason = "volume_missing"
        elif participation_rate is None:
            skipped_reason = "volume_missing"
        elif participation_rate > threshold:
            skipped_reason = "volume_too_thin"
        else:
            skipped_reason = quote_quality_skip_reason(
                exit_quote,
                missing_reason=f"{cost.exit_role}_quote_missing",
                stale_reason=f"{cost.exit_role}_quote_stale",
                error_reason="halt_or_bad_quote",
                max_spread_bps=cost.max_spread_bps,
            )
    else:
        skipped_reason = entry_reason

    quote_cost_return = None
    slippage_return = None
    modeled_return = 0.0
    status = VOLUME_TRADEABILITY_SKIP_STATUS
    if skipped_reason == "none" and entry_quote is not None and exit_quote is not None:
        assert entry_quote.ask is not None and exit_quote.bid is not None
        quote_cost_return = exit_quote.bid / entry_quote.ask - 1.0
        slip = cost.slippage_bps / 10000.0
        slip_entry = entry_quote.ask * (1.0 + slip)
        slip_exit = exit_quote.bid * (1.0 - slip)
        slippage_return = slip_exit / slip_entry - 1.0
        modeled_return = slippage_return
        status = VOLUME_TRADEABILITY_OK_STATUS

    return {
        "volume_tradeability_status": status,
        "volume_skipped_reason": skipped_reason,
        "entry_window_share_volume": observed_share_volume,
        "intended_order_share_participation_rate": share_participation_rate,
        "entry_window_dollar_volume": evidence["entry_window_dollar_volume"],
        "intended_order_usd": cost.intended_order_usd,
        "intended_order_participation_rate": participation_rate,
        "quote_cost_return": quote_cost_return,
        "slippage_return": slippage_return,
        "modeled_return": modeled_return,
        "volume_evidence_available": evidence["volume_evidence_available"],
        "window_basis": evidence["window_basis"],
        "window_price_basis": evidence.get("window_price_basis"),
        "denominator_basis": evidence.get("denominator_basis"),
    }


def predecision_volume_evidence(
    candidate: Any | None,
    entry_quote: Any | None,
) -> dict[str, Any]:
    if candidate is None:
        return missing_entry_volume_evidence()
    features = _json_loads(getattr(candidate, "feature_json", None))
    source_bars = _json_loads(getattr(candidate, "source_bars_json", None))
    observed_volume = _finite_float(features.get("observed_cumulative_volume_before_decision"))
    if observed_volume is not None and observed_volume > 0:
        share_volume = observed_volume
        denominator_basis = "observed_cumulative_volume_before_decision"
    else:
        share_volume = None
        denominator_basis = "missing"
    if share_volume is not None and share_volume > 0:
        unsafe_reason = predecision_volume_timestamp_violation(
            candidate,
            features=features,
            source_bars=source_bars,
        )
        if unsafe_reason is not None:
            return missing_entry_volume_evidence(
                window_basis="unsafe_predecision_timestamp",
                denominator_basis=unsafe_reason,
            )
    prior_close = _finite_float(features.get("prior_close"))
    gap = _finite_float(features.get("gap"))
    early_return = _finite_float(features.get("early_return"))
    price = None
    price_basis = "missing"
    if (
        entry_quote is not None
        and entry_quote.bid is not None
        and entry_quote.ask is not None
        and entry_quote.bid > 0
        and entry_quote.ask > 0
        and entry_quote.ask >= entry_quote.bid
    ):
        price = (entry_quote.bid + entry_quote.ask) / 2.0
        price_basis = "entry_quote_mid"
    if (
        price is None
        and prior_close is not None
        and prior_close > 0
        and gap is not None
        and early_return is not None
    ):
        day_open = prior_close * (1.0 + gap)
        price = day_open * (1.0 + early_return)
        price_basis = VOLUME_PRICE_BASIS_LAST_PRE_DECISION
    dollar_volume = (
        share_volume * price
        if share_volume is not None
        and share_volume > 0
        and price is not None
        and price > 0
        else None
    )
    return {
        "volume_evidence_available": dollar_volume is not None,
        "entry_window_dollar_volume": dollar_volume,
        "entry_window_share_volume_denominator": (
            share_volume if share_volume is not None and share_volume > 0 else None
        ),
        "window_basis": VOLUME_WINDOW_BASIS_PRE_DECISION,
        "window_price_basis": price_basis,
        "denominator_basis": denominator_basis,
        "source_minute_bars_max_start_ts": source_bars.get("source_minute_bars_max_start_ts"),
        "completed_through_ts": (
            features.get("completed_through_ts") or source_bars.get("completed_through_ts")
        ),
    }


def missing_entry_volume_evidence(
    *,
    window_basis: str = "missing",
    denominator_basis: str = "missing",
) -> dict[str, Any]:
    return {
        "volume_evidence_available": False,
        "entry_window_dollar_volume": None,
        "entry_window_share_volume_denominator": None,
        "window_basis": window_basis,
        "window_price_basis": None,
        "denominator_basis": denominator_basis,
        "source_minute_bars_max_start_ts": None,
        "completed_through_ts": None,
    }


def predecision_volume_timestamp_violation(
    candidate: Any,
    *,
    features: Mapping[str, Any],
    source_bars: Mapping[str, Any],
) -> str | None:
    decision_ts = _coerce_persisted_utc(getattr(candidate, "decision_ts", None))
    if decision_ts is None:
        return "missing_predecision_timestamp_proof"
    source_max_ts, source_max_error = _select_predecision_timestamp_proof(
        features,
        source_bars,
        "source_minute_bars_max_start_ts",
    )
    if source_max_error is not None:
        return source_max_error
    if source_max_ts is not None and source_max_ts >= decision_ts:
        return "source_minute_bars_max_start_ts_at_or_after_decision_ts"
    completed_through_ts, completed_error = _select_predecision_timestamp_proof(
        features,
        source_bars,
        "completed_through_ts",
    )
    if completed_error is not None:
        return completed_error
    if completed_through_ts is not None and completed_through_ts > decision_ts:
        return "completed_through_ts_after_decision_ts"
    return None


def quote_quality_skip_reason(
    quote: Any | None,
    *,
    missing_reason: str,
    stale_reason: str,
    error_reason: str,
    max_spread_bps: float,
) -> str:
    if quote is None:
        return missing_reason
    if quote.coverage_status == "missing":
        return missing_reason
    if quote.coverage_status == "stale":
        return stale_reason
    if quote.coverage_status != "ok":
        return error_reason
    raw = _json_loads(getattr(quote, "raw_json", None))
    conditions = raw.get("c") or raw.get("conditions") or []
    if any(str(item).upper() in HALT_CONDITIONS for item in conditions):
        return "halt_or_bad_quote"
    if quote.bid is None or quote.ask is None or quote.bid <= 0 or quote.ask <= 0:
        return "halt_or_bad_quote"
    if quote.ask < quote.bid:
        return "halt_or_bad_quote"
    if quote.spread_bps is None or quote.spread_bps > max_spread_bps:
        return "spread"
    return "none"


def _select_predecision_timestamp_proof(
    features: Mapping[str, Any],
    source_bars: Mapping[str, Any],
    key: str,
) -> tuple[datetime | None, str | None]:
    saw_value = False
    for mapping in (features, source_bars):
        value = mapping.get(key)
        if _timestamp_value_missing(value):
            continue
        saw_value = True
        parsed = _parse_timestamp_candidate(value)
        if parsed is not None:
            return parsed, None
    if not saw_value:
        return None, "missing_predecision_timestamp_proof"
    return None, "malformed_predecision_timestamp_proof"


def _timestamp_value_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _parse_timestamp_candidate(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _coerce_persisted_utc(value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return _coerce_persisted_utc(parsed)


def _coerce_persisted_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _finite_float(*values: Any) -> float | None:
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _json_loads(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}
