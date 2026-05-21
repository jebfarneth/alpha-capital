"""
Point-in-time and quality guard utilities.

These produce warnings and quality flags. They do NOT block pattern
admission — validation affects confidence, not eligibility.

Structural invalidity (missing timestamp) raises ValueError because
the caller cannot proceed without it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

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
                confidence *= float(value)

    return round(confidence, 4)
