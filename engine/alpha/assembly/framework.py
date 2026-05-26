"""
Feature assembly framework.

Owns field-level lookahead enforcement. Assemblers declare source timestamps
and allowed cutoffs; the framework rejects future-contaminated fields before
detector orchestration.

Missing data is never coerced to zero. 0 means observed zero only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from alpha.data.contracts import stable_hash
from alpha.patterns.contracts import PatternInput


class FieldPresence:
    PRESENT = "present"
    MISSING = "missing"
    UNAVAILABLE = "unavailable"
    REJECTED_LOOKAHEAD = "rejected_lookahead"


@dataclass
class AssembledField:
    """A single assembled field with full provenance."""

    name: str
    value: Any
    presence: str  # FieldPresence
    source_timestamp: Optional[datetime] = None
    allowed_cutoff: Optional[datetime] = None
    source_provider: Optional[str] = None
    lineage_id: Optional[str] = None
    lineage_hash: Optional[str] = None
    rejection_reason: Optional[str] = None


@dataclass
class AssemblyDiagnostic:
    """Diagnostic record for a ticker/pattern assembly outcome."""

    ticker: str
    pattern_id: str
    diagnostic_type: str
    detail: Optional[str] = None


@dataclass
class PatternAssemblyResult:
    """Output from assembling inputs for one pattern across all tickers."""

    pattern_id: str
    inputs: List[PatternInput] = field(default_factory=list)
    diagnostics: List[AssemblyDiagnostic] = field(default_factory=list)
    rejected_fields: List[AssembledField] = field(default_factory=list)
    assembled_count: int = 0
    rejected_count: int = 0
    insufficient_count: int = 0


def _comparable_utc(dt: datetime) -> datetime:
    """Normalize to naive UTC for comparison."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def validate_assembled_fields(
    fields: List[AssembledField],
    cutoff: datetime,
) -> Tuple[Dict[str, Any], List[AssembledField]]:
    """Validate fields against the cutoff timestamp.

    Returns (validated_values, rejected_fields).
    Present fields with source_timestamp after cutoff are reclassified as
    rejected_lookahead. Missing/unavailable fields pass through to rejected.
    Only present, non-future fields go into validated_values.
    """
    validated: Dict[str, Any] = {}
    rejected: List[AssembledField] = []
    cutoff_cmp = _comparable_utc(cutoff)

    for f in fields:
        if f.presence == FieldPresence.PRESENT:
            if f.source_timestamp is not None:
                src_cmp = _comparable_utc(f.source_timestamp)
                if src_cmp > cutoff_cmp:
                    f = AssembledField(
                        name=f.name, value=f.value,
                        presence=FieldPresence.REJECTED_LOOKAHEAD,
                        source_timestamp=f.source_timestamp,
                        allowed_cutoff=f.allowed_cutoff,
                        source_provider=f.source_provider,
                        lineage_id=f.lineage_id,
                        lineage_hash=f.lineage_hash,
                        rejection_reason=(
                            f"source_timestamp {f.source_timestamp.isoformat()} "
                            f"after cutoff {cutoff.isoformat()}"
                        ),
                    )
                    rejected.append(f)
                    continue
            validated[f.name] = f.value
        else:
            rejected.append(f)

    return validated, rejected


def build_pattern_input(
    *,
    ticker: str,
    pattern_id: str,
    asof_timestamp: datetime,
    validated_fields: Dict[str, Any],
    lineage_ids: List[str],
    lineage_hashes: List[str],
    universe_snapshot_id: Optional[str] = None,
) -> PatternInput:
    """Build a PatternInput from validated assembly output.

    Only validated (present, non-future) fields are placed in market_data.
    """
    return PatternInput(
        ticker=ticker,
        asof_timestamp=asof_timestamp,
        market_data=dict(validated_fields),
        lineage_ids=list(lineage_ids),
        lineage_hashes=list(lineage_hashes),
        universe_snapshot_id=universe_snapshot_id,
    )


def compute_assembly_lineage_hash(
    pattern_id: str,
    ticker: str,
    validated_fields: Dict[str, Any],
    lineage_hashes: List[str],
) -> str:
    """Deterministic hash covering pattern, ticker, fields, and source lineage."""
    return stable_hash({
        "pattern_id": pattern_id,
        "ticker": ticker,
        "fields": validated_fields,
        "lineage_hashes": sorted(lineage_hashes),
    })
