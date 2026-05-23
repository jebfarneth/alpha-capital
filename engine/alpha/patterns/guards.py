"""
Point-in-time and quality guard utilities.

These produce warnings and quality flags. They do NOT block pattern
admission — validation affects confidence, not eligibility.

Structural invalidity (missing timestamp) raises ValueError because
the caller cannot proceed without it.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from alpha.data.contracts import stable_hash
from alpha.patterns.contracts import FidelityTier


QUALITY_CONFIDENCE_MULTIPLIERS = {
    "missing_lineage": 0.90,
    "gk_low_transaction_warning": 0.90,
    "baseline_volume_proxy": 0.95,
    "baseline_range_proxy": 0.95,
    "baseline_spread_proxy": 0.95,
    "missing_expansion_data": 0.95,
    "missing_volume_data": 0.95,
    "missing_prior_close": 0.95,
    "invalid_prior_close": 0.95,
    "missing_gk_avg_5d_for_gap_proxy": 0.95,
    "invalid_gk_avg_5d_for_gap_proxy": 0.95,
}
DEFAULT_QUOTE_FIELDS = ("candidate_eval_bid", "candidate_eval_ask", "candidate_eval_quote_timestamp", "quote_age_ms")
DEFAULT_QUOTE_DIAGNOSTIC_FIELDS = (*DEFAULT_QUOTE_FIELDS, "quote_freshness_max_ms")
DATA_QUALITY_FIELDS = ("market_data_status", "halt_status", "corporate_action_filter_passed")
BAD_MARKET_DATA_STATUSES = {"delayed", "partial_outage", "unavailable", "stale"}


def require_asof_timestamp(
    asof: Optional[datetime],
) -> datetime:
    """Raise if asof is None — structural requirement, not a validation gate."""
    if asof is None:
        raise ValueError("asof_timestamp is required for point-in-time detection")
    return asof


def reject_future_timestamp(
    asof: datetime,
    warnings: List[str],
    quality_flags: Dict[str, object],
    *,
    reference: Optional[datetime] = None,
) -> bool:
    """
    Flag if asof is in the future. Returns True if the timestamp is valid.

    Adds a warning and quality flag if the timestamp is future.
    Does NOT raise — the detector can still run but the output is flagged.
    """
    now = reference or datetime.now(timezone.utc)
    if asof > now:
        warnings.append(f"asof_timestamp {asof.isoformat()} is in the future")
        quality_flags["future_timestamp"] = True
        quality_flags["point_in_time_passed"] = False
        return False
    return True


def require_lineage_hash(
    lineage_hashes: List[str],
    warnings: List[str],
    quality_flags: Dict[str, object],
) -> bool:
    """
    Warn if no lineage hashes are provided. Returns True if present.

    Missing lineage means the feature snapshot cannot prove its data origin.
    This is a quality flag, not a blocker.
    """
    if not lineage_hashes:
        warnings.append("no data lineage hashes provided — feature origin unverifiable")
        quality_flags["missing_lineage"] = True
        return False
    return True


def classify_fidelity(
    *,
    has_primary_data: bool,
    has_secondary_data: bool = True,
    point_in_time_passed: Optional[bool] = None,
    lookahead_guard_passed: Optional[bool] = None,
) -> str:
    """
    Classify fidelity tier per vault contract.

    FULL: all required data present + PIT/lookahead guards pass.
    LITE: primary data present but secondary missing or guards unclear.
    UNAVAILABLE: primary data missing.
    """
    if not has_primary_data:
        return FidelityTier.UNAVAILABLE
    if not has_secondary_data:
        return FidelityTier.LITE
    if point_in_time_passed is False or lookahead_guard_passed is False:
        return FidelityTier.LITE
    return FidelityTier.FULL


def operating_universe_rejection(
    market_data: Dict[str, Any],
    warnings: List[str],
    quality_flags: Dict[str, Any],
    *,
    pattern_id: str,
) -> Optional[str]:
    """
    Fail closed on operating-universe membership.

    The detector layer must not emit signals when the shared universe job did
    not compute membership. A missing flag is a data readiness failure, not a
    validation haircut.
    """
    if market_data.get("operating_universe_inclusion") is True:
        return None
    if market_data.get("operating_universe_inclusion") is False:
        quality_flags["not_operating_universe_member"] = True
        warnings.append(f"{pattern_id}: ticker is not marked as operating-universe eligible")
        return "not_operating_universe"

    quality_flags["operating_universe_not_computed"] = True
    warnings.append(f"{pattern_id}: operating-universe membership missing; failing closed")
    return "missing_operating_universe"


def copy_fields(feat_dict: Dict[str, Any], market_data: Dict[str, Any], fields: Iterable[str]) -> None:
    """Copy present market-data fields into feature evidence."""
    for key in fields:
        val = market_data.get(key)
        if val is not None:
            feat_dict[key] = val


def set_signal_identity(
    feat_dict: Dict[str, Any],
    *,
    pattern_id: str,
    ticker: str,
    components: Dict[str, Any],
    source: str,
    identity_hash: Optional[str] = None,
) -> Optional[str]:
    """
    Persist a stable event/setup identity for downstream dedup.

    Components must describe the underlying event/setup, not the scan or
    evaluation timestamp. Empty components are ignored; if nothing remains,
    no identity is emitted.
    """
    clean_components: Dict[str, Any] = {}
    for key, value in components.items():
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        clean_components[key] = value
    if not clean_components:
        return None

    identity_components = {
        "pattern_id": pattern_id,
        "ticker": ticker,
        **clean_components,
    }
    identity_hash = identity_hash or stable_hash(identity_components)
    feat_dict["signal_identity_hash"] = identity_hash
    feat_dict["signal_identity_components"] = identity_components
    feat_dict["signal_identity_source"] = source
    return identity_hash


def market_data_quality_rejection(
    feat_dict: Dict[str, Any],
    market_data: Dict[str, Any],
    *,
    require_fields: bool = False,
    missing_rejection: str = "missing_market_data_quality",
    data_delay_rejection: str = "data_delay",
    halt_rejection: str = "halted",
    corporate_action_rejection: str = "spurious_corporate_action",
) -> Optional[str]:
    """
    Canonical market-data quality guard.

    Patterns may choose whether the status fields are required. Intraday
    live-data detectors should require them; EOD lanes may treat absent fields
    as unavailable diagnostics while still rejecting explicit bad values.
    """
    copy_fields(feat_dict, market_data, DATA_QUALITY_FIELDS)

    if require_fields and any(market_data.get(field) is None for field in DATA_QUALITY_FIELDS):
        return missing_rejection

    if market_data.get("market_data_status") in BAD_MARKET_DATA_STATUSES:
        return data_delay_rejection

    halt_status = market_data.get("halt_status")
    if halt_status is not None and halt_status != "clear":
        return halt_rejection

    if market_data.get("corporate_action_filter_passed") is False:
        return corporate_action_rejection

    return None


def quote_rejection(
    market_data: Dict[str, Any],
    *,
    quote_fields: Iterable[str] = DEFAULT_QUOTE_FIELDS,
    quote_unavailable_rejection: str = "quote_unavailable",
) -> Optional[str]:
    """Canonical candidate-evaluation quote guard for Class C lanes."""
    if any(market_data.get(field) is None for field in quote_fields):
        return quote_unavailable_rejection
    try:
        bid = float(market_data["candidate_eval_bid"])
        ask = float(market_data["candidate_eval_ask"])
        quote_age_ms = int(market_data["quote_age_ms"])
    except (TypeError, ValueError):
        return quote_unavailable_rejection

    if bid <= 0 or ask <= 0:
        return quote_unavailable_rejection
    if quote_age_ms < 0:
        return quote_unavailable_rejection
    max_age_ms = market_data.get("quote_freshness_max_ms")
    if max_age_ms is not None:
        try:
            if quote_age_ms > int(max_age_ms):
                return quote_unavailable_rejection
        except (TypeError, ValueError):
            return quote_unavailable_rejection
    return None


def compute_data_confidence(
    quality_flags: Dict[str, Any],
    *,
    field_confidence_sources: Optional[Iterable[Dict[str, Any]]] = None,
    extra_multipliers: Optional[Dict[str, float]] = None,
) -> float:
    """Canonical detector-layer data-confidence multiplier."""
    confidence = 1.0
    multipliers = dict(QUALITY_CONFIDENCE_MULTIPLIERS)
    if extra_multipliers:
        multipliers.update(extra_multipliers)

    for flag, multiplier in multipliers.items():
        if quality_flags.get(flag):
            confidence *= multiplier

    for source in field_confidence_sources or ():
        field_confidence = source.get("field_confidence")
        if isinstance(field_confidence, dict):
            for value in field_confidence.values():
                parsed = finite_float(value)
                if parsed is not None:
                    confidence *= min(max(parsed, 0.0), 1.0)

    return round(confidence, 4)


def finite_float(value: Any) -> Optional[float]:
    """Parse *value* to float, returning None for non-finite or unparseable inputs."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def integral_int(value: Any) -> Optional[int]:
    """Parse *value* to int, returning None for non-integral, non-finite, or unparseable inputs."""
    parsed = finite_float(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)
