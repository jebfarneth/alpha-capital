"""
Point-in-time and quality guard utilities.

These produce warnings and quality flags. They do NOT block pattern
admission — validation affects confidence, not eligibility.

Structural invalidity (missing timestamp) raises ValueError because
the caller cannot proceed without it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from alpha.patterns.contracts import FidelityTier


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
