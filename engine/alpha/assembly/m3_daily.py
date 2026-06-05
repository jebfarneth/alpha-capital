"""M3 Sector Rotation daily producer and assembler helpers.

M3's detector consumes a pre-baked sector-rank payload. This module owns the
PIT formation-cohort math that makes that payload trustworthy: sector identity
comes from interval history, sector returns are computed from the t-126
formation cohort, and detector proof flags are set only from that path.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from alpha.assembly.framework import (
    AssembledField,
    AssemblyDiagnostic,
    FieldPresence,
    PatternAssemblyResult,
    build_pattern_input,
    validate_assembled_fields,
)
from alpha.assembly.m3_sector_map import SIC_TO_SECTOR_MAP_VERSION
from alpha.market_calendar import previous_us_equity_session
from alpha.patterns.contracts import PatternId


PATTERN_ID = PatternId.M3
SHADOW_PATTERN_ID = "M3S"
PRODUCTION_TAXONOMY_SOURCE = "POLYGON_SIC"
OPEN_INTERVAL_END = date(9999, 12, 31)
SECTOR_RETURN_LOOKBACK_SESSIONS = 126
MIN_PRODUCTION_SECTOR_HISTORY_COVERAGE_YEARS = 3.0
SHUMWAY_PERFORMANCE_DELISTING_RETURN = -0.30
DELISTING_REASON_SOURCE_PROVIDER = "provider_reason"
DELISTING_REASON_SOURCE_UNKNOWN_REVIEW = "unknown_review"
DELISTING_TREATMENT_SHUMWAY_FAILURE = "shumway_failure"
DELISTING_TREATMENT_SHUMWAY_UNKNOWN_DEFAULT = "shumway_unknown_default"
DELISTING_TREATMENT_ACQUISITION_PAYOFF = "acquisition_realized_payoff"


@dataclass
class SectorAssignmentSnapshot:
    ticker: str
    sector: str
    source: str
    valid_from: date
    valid_to: date
    sic_code: Optional[str] = None
    sic_description: Optional[str] = None
    industry: Optional[str] = None
    sic_to_sector_map_version: str = SIC_TO_SECTOR_MAP_VERSION
    sector_history_coverage_years: Optional[float] = None
    last_verified: Optional[date] = None


@dataclass
class SectorReturnComponent:
    ticker: str
    sector: str
    market_cap: float
    return_6mo: float
    return_3mo: Optional[float] = None
    return_1mo: Optional[float] = None
    delisting_adjustment_applied: bool = False
    sector_history_coverage_years: Optional[float] = None
    delisted_date: Optional[date] = None
    delisting_reason: Optional[str] = None
    delisting_reason_source: Optional[str] = None
    delisting_adjustment_treatment: Optional[str] = None
    shumway_suppressed_reason: Optional[str] = None


@dataclass
class SectorReturnSnapshot:
    date: date
    sector: str
    return_6mo: float
    return_6mo_ew: Optional[float]
    return_1mo: Optional[float]
    return_3mo: Optional[float]
    sector_rank: int
    sector_rank_normalized: float
    n_sectors: int
    n_firms_in_sector: int
    total_market_cap_in_sector: float
    formation_date: date
    source: str = PRODUCTION_TAXONOMY_SOURCE
    sic_to_sector_map_version: str = SIC_TO_SECTOR_MAP_VERSION
    point_in_time_passed: bool = False
    formation_cohort_passed: bool = True
    sector_history_coverage_years: Optional[float] = None
    delisting_shumway_adjustment_count: int = 0
    delisting_unknown_review_count: int = 0
    delisting_adjustment_audit: Optional[Dict[str, Any]] = None


def nth_previous_session(day: date, sessions: int) -> date:
    """Return the session `sessions` regular sessions strictly before `day`."""

    if sessions < 1:
        raise ValueError("sessions must be >= 1")
    cursor = day
    for _ in range(sessions):
        cursor = previous_us_equity_session(cursor)
    return cursor


def _ensure_aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _snap_attr(snapshot: Any, name: str, default: Any = None) -> Any:
    if isinstance(snapshot, dict):
        return snapshot.get(name, default)
    return getattr(snapshot, name, default)


def _field(
    name: str,
    value: Any,
    source_timestamp: Optional[datetime],
    source_provider: str,
    lineage_hash: Optional[str] = None,
) -> AssembledField:
    return AssembledField(
        name=name,
        value=value,
        presence=FieldPresence.PRESENT if value is not None else FieldPresence.MISSING,
        source_timestamp=_ensure_aware(source_timestamp),
        source_provider=source_provider,
        lineage_hash=lineage_hash,
    )


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _bar_date(bar: Any) -> Optional[date]:
    value = getattr(bar, "date", None)
    if value is None and isinstance(bar, dict):
        value = bar.get("date")
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        return date.fromisoformat(value)
    return None


def _adjusted_close(bar: Any) -> Optional[float]:
    for name in ("adj_close", "adjusted_close", "split_adjusted_close", "close"):
        value = getattr(bar, name, None)
        if value is None and isinstance(bar, dict):
            value = bar.get(name)
        if _is_finite(value) and float(value) > 0:
            return float(value)
    return None


def adjusted_return(
    bars: Sequence[Any],
    *,
    start_date: date,
    end_date: date,
    delisted_date: Optional[date] = None,
    delisting_reason: Optional[str] = None,
    shumway_return: float = SHUMWAY_PERFORMANCE_DELISTING_RETURN,
) -> Tuple[Optional[float], bool]:
    """Compute adjusted return over [start_date, end_date].

    If a firm delisted inside the window and the end-date price is absent, use
    the last available adjusted close before delisting. Apply Shumway's
    performance-delisting adjustment unless the delisting reason is positively
    classified as an acquisition/merger payoff. Unknown reasons take the
    survivorship-conservative Shumway default and remain stamped for review.
    """

    prices: Dict[date, float] = {}
    for bar in bars:
        day = _bar_date(bar)
        close = _adjusted_close(bar)
        if day is not None and close is not None:
            prices[day] = close
    start_price = prices.get(start_date)
    if start_price is None or start_price <= 0:
        return None, False
    end_price = prices.get(end_date)
    if end_price is not None and end_price > 0:
        return (end_price / start_price) - 1.0, False

    if delisted_date is None or not (start_date <= delisted_date <= end_date):
        return None, False
    candidate_days = [day for day in prices if start_date <= day <= delisted_date]
    if not candidate_days:
        return None, False
    last_price = prices[max(candidate_days)]
    partial = (last_price / start_price) - 1.0
    reason_class = _classify_delisting_reason(delisting_reason)
    if reason_class == "acquisition":
        return partial, False
    adjusted = (1.0 + partial) * (1.0 + shumway_return) - 1.0
    return adjusted, True


def _is_acquisition_delisting(reason: Optional[str]) -> bool:
    return _classify_delisting_reason(reason) == "acquisition"


def _classify_delisting_reason(reason: Optional[str]) -> str:
    if not reason:
        return "unknown_review"
    text = reason.casefold()
    failure_markers = (
        "bankrupt",
        "chapter 7",
        "chapter 11",
        "deficien",
        "delinquent",
        "distress",
        "exchange-mandated",
        "failed",
        "failure",
        "going concern",
        "insolv",
        "listing deficien",
        "liquidat",
        "minimum equity",
        "noncompliance",
        "receivership",
        "reorganization",
        "suspended",
    )
    if any(marker in text for marker in failure_markers):
        return "failure"
    acquisition_patterns = (
        r"\bcash\s+merger\b",
        r"\bmerger\b",
        r"\bacquisition\b",
        r"\btakeover\b",
        r"\bbuyout\b",
        r"\bm\s*&\s*a\b",
        r"\b(?:to be|being|was|were|is|has been|will be)\s+acquired\b",
        r"\bacquired\s+(?:by|for|in|via|through)\b",
    )
    if any(re.search(pattern, text) for pattern in acquisition_patterns):
        return "acquisition"
    return "unknown_review"


def _delisting_adjustment_metadata(
    *,
    delisted_date: Optional[date],
    delisting_reason: Optional[str],
    shumway_applied: bool,
) -> Dict[str, Optional[str]]:
    if delisted_date is None:
        return {
            "delisting_reason_source": None,
            "delisting_adjustment_treatment": None,
            "shumway_suppressed_reason": None,
        }
    reason_class = _classify_delisting_reason(delisting_reason)
    reason_source = (
        DELISTING_REASON_SOURCE_PROVIDER
        if delisting_reason and delisting_reason.strip()
        else DELISTING_REASON_SOURCE_UNKNOWN_REVIEW
    )
    if shumway_applied:
        treatment = (
            DELISTING_TREATMENT_SHUMWAY_UNKNOWN_DEFAULT
            if reason_class == "unknown_review"
            else DELISTING_TREATMENT_SHUMWAY_FAILURE
        )
        suppressed_reason = None
    elif reason_class == "acquisition":
        treatment = DELISTING_TREATMENT_ACQUISITION_PAYOFF
        suppressed_reason = "acquisition_or_merger"
    else:
        treatment = None
        suppressed_reason = None
    return {
        "delisting_reason_source": reason_source,
        "delisting_adjustment_treatment": treatment,
        "shumway_suppressed_reason": suppressed_reason,
    }


def _component_return(
    bars: Sequence[Any],
    *,
    start_date: date,
    end_date: date,
    delisted_date: Optional[date],
    delisting_reason: Optional[str] = None,
) -> Optional[float]:
    value, _ = adjusted_return(
        bars,
        start_date=start_date,
        end_date=end_date,
        delisted_date=delisted_date,
        delisting_reason=delisting_reason,
    )
    return value


def build_sector_return_components(
    *,
    formation_snapshots: Sequence[Any],
    assignments_by_ticker: Dict[str, SectorAssignmentSnapshot],
    bars_by_ticker: Dict[str, Sequence[Any]],
    evidence_date: date,
    formation_date: date,
    one_month_date: Optional[date] = None,
    three_month_date: Optional[date] = None,
    delisted_dates_by_ticker: Optional[Dict[str, date]] = None,
    delisting_reasons_by_ticker: Optional[Dict[str, str]] = None,
) -> Tuple[List[SectorReturnComponent], List[AssemblyDiagnostic]]:
    """Build firm-level return components from the formation-date cohort."""

    components: List[SectorReturnComponent] = []
    diagnostics: List[AssemblyDiagnostic] = []
    delisted_dates_by_ticker = delisted_dates_by_ticker or {}
    delisting_reasons_by_ticker = delisting_reasons_by_ticker or {}
    for snap in formation_snapshots:
        ticker = str(_snap_attr(snap, "ticker") or "").upper()
        assignment = assignments_by_ticker.get(ticker)
        if assignment is None:
            diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=PATTERN_ID,
                diagnostic_type="formation_sector_unknown",
            ))
            continue
        market_cap = _snap_attr(snap, "market_cap")
        if not _is_finite(market_cap) or float(market_cap) <= 0:
            diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=PATTERN_ID,
                diagnostic_type="formation_market_cap_missing",
            ))
            continue
        bars = bars_by_ticker.get(ticker, ())
        delisted_date = delisted_dates_by_ticker.get(ticker)
        delisting_reason = delisting_reasons_by_ticker.get(ticker)
        ret_6mo, delisting_applied = adjusted_return(
            bars,
            start_date=formation_date,
            end_date=evidence_date,
            delisted_date=delisted_date,
            delisting_reason=delisting_reason,
        )
        if ret_6mo is None:
            diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=PATTERN_ID,
                diagnostic_type="return_6mo_unavailable",
            ))
            continue
        delisting_meta = _delisting_adjustment_metadata(
            delisted_date=delisted_date,
            delisting_reason=delisting_reason,
            shumway_applied=delisting_applied,
        )
        components.append(SectorReturnComponent(
            ticker=ticker,
            sector=assignment.sector,
            market_cap=float(market_cap),
            return_6mo=ret_6mo,
            return_3mo=(
                _component_return(
                    bars,
                    start_date=three_month_date,
                    end_date=evidence_date,
                    delisted_date=delisted_date,
                    delisting_reason=delisting_reason,
                )
                if three_month_date is not None
                else None
            ),
            return_1mo=(
                _component_return(
                    bars,
                    start_date=one_month_date,
                    end_date=evidence_date,
                    delisted_date=delisted_date,
                    delisting_reason=delisting_reason,
                )
                if one_month_date is not None
                else None
            ),
            delisting_adjustment_applied=delisting_applied,
            sector_history_coverage_years=assignment.sector_history_coverage_years,
            delisted_date=delisted_date,
            delisting_reason=delisting_reason,
            delisting_reason_source=delisting_meta["delisting_reason_source"],
            delisting_adjustment_treatment=delisting_meta["delisting_adjustment_treatment"],
            shumway_suppressed_reason=delisting_meta["shumway_suppressed_reason"],
        ))
    return components, diagnostics


def compute_sector_return_snapshots(
    *,
    components: Sequence[SectorReturnComponent],
    asof_date: date,
    formation_date: date,
) -> List[SectorReturnSnapshot]:
    """Aggregate firm components into ranked sector returns."""

    by_sector: Dict[str, List[SectorReturnComponent]] = {}
    for component in components:
        if component.market_cap > 0 and _is_finite(component.return_6mo):
            by_sector.setdefault(component.sector, []).append(component)
    raw_rows: List[Dict[str, Any]] = []
    for sector, rows in by_sector.items():
        total_cap = sum(row.market_cap for row in rows)
        if total_cap <= 0:
            continue
        vw = sum(row.market_cap * row.return_6mo for row in rows) / total_cap
        ew = sum(row.return_6mo for row in rows) / len(rows)
        one_month_values = [row.return_1mo for row in rows if row.return_1mo is not None]
        three_month_values = [row.return_3mo for row in rows if row.return_3mo is not None]
        component_coverages = [row.sector_history_coverage_years for row in rows]
        if any(coverage is None for coverage in component_coverages):
            sector_coverage_years = None
        else:
            sector_coverage_years = min(component_coverages)  # type: ignore[arg-type]
        point_in_time_passed = (
            sector_coverage_years is not None
            and sector_coverage_years >= MIN_PRODUCTION_SECTOR_HISTORY_COVERAGE_YEARS
        )
        raw_rows.append({
            "sector": sector,
            "return_6mo": vw,
            "return_6mo_ew": ew,
            "return_1mo": (
                sum(one_month_values) / len(one_month_values)
                if one_month_values else None
            ),
            "return_3mo": (
                sum(three_month_values) / len(three_month_values)
                if three_month_values else None
            ),
            "n_firms_in_sector": len(rows),
            "total_market_cap_in_sector": total_cap,
            "point_in_time_passed": point_in_time_passed,
            "sector_history_coverage_years": sector_coverage_years,
            "delisting_shumway_adjustment_count": sum(
                1 for row in rows if row.delisting_adjustment_applied
            ),
            "delisting_unknown_review_count": sum(
                1
                for row in rows
                if row.delisting_reason_source == DELISTING_REASON_SOURCE_UNKNOWN_REVIEW
            ),
            "delisting_adjustment_audit": _sector_delisting_adjustment_audit(rows),
        })
    ranked = sorted(raw_rows, key=lambda row: (row["return_6mo"], row["sector"]))
    n = len(ranked)
    result: List[SectorReturnSnapshot] = []
    for index, row in enumerate(ranked, start=1):
        result.append(SectorReturnSnapshot(
            date=asof_date,
            sector=row["sector"],
            return_6mo=row["return_6mo"],
            return_6mo_ew=row["return_6mo_ew"],
            return_1mo=row["return_1mo"],
            return_3mo=row["return_3mo"],
            sector_rank=index,
            sector_rank_normalized=(index - 0.5) / n if n else 0.0,
            n_sectors=n,
            n_firms_in_sector=row["n_firms_in_sector"],
            total_market_cap_in_sector=row["total_market_cap_in_sector"],
            formation_date=formation_date,
            point_in_time_passed=row["point_in_time_passed"],
            formation_cohort_passed=True,
            sector_history_coverage_years=row["sector_history_coverage_years"],
            delisting_shumway_adjustment_count=row["delisting_shumway_adjustment_count"],
            delisting_unknown_review_count=row["delisting_unknown_review_count"],
            delisting_adjustment_audit=row["delisting_adjustment_audit"],
        ))
    return result


def _sector_delisting_adjustment_audit(
    components: Sequence[SectorReturnComponent],
) -> Optional[Dict[str, Any]]:
    rows = []
    for component in components:
        if component.delisted_date is None:
            continue
        rows.append({
            "ticker": component.ticker,
            "delisted_date": component.delisted_date.isoformat(),
            "delisting_reason": component.delisting_reason,
            "delisting_reason_source": component.delisting_reason_source,
            "delisting_adjustment_treatment": component.delisting_adjustment_treatment,
            "shumway_applied": component.delisting_adjustment_applied,
            "shumway_suppressed_reason": component.shumway_suppressed_reason,
        })
    if not rows:
        return None
    return {"components": rows[:50], "component_count": len(rows)}


def _current_assignment_field_confidence(
    assignment: SectorAssignmentSnapshot,
    *,
    evidence_date: date,
    min_coverage_years: float,
) -> Dict[str, float]:
    """Confidence multipliers for current assignment freshness only.

    Formation-date interval lookups remain exact-as-of and are not discounted
    here. The coverage multiplier reflects the DATA.md shadow-only mandate for
    the current taxonomy history feeding production M3.
    """

    coverage = assignment.sector_history_coverage_years
    coverage_confidence = (
        min(max(coverage / min_coverage_years, 0.0), 1.0)
        if coverage is not None and min_coverage_years > 0
        else 0.0
    )
    if assignment.last_verified is None:
        freshness_confidence = 0.5
    else:
        age_days = max(0, (evidence_date - assignment.last_verified).days)
        if age_days <= 7:
            freshness_confidence = 1.0
        elif age_days <= 30:
            freshness_confidence = 0.9
        elif age_days <= 90:
            freshness_confidence = 0.75
        else:
            freshness_confidence = 0.5
    return {
        "current_sector_assignment_coverage": round(coverage_confidence, 4),
        "current_sector_assignment_freshness": round(freshness_confidence, 4),
    }


def assemble_m3_daily(
    *,
    snapshots: Sequence[Any],
    assignments_by_ticker: Dict[str, SectorAssignmentSnapshot],
    sector_returns_by_sector: Dict[str, SectorReturnSnapshot],
    cutoff_timestamp: datetime,
    universe_cutoff_timestamp: Optional[datetime],
    decision_date: str,
    evidence_session_date: str,
    next_execution_session: Optional[str] = None,
    source_provider: str = PRODUCTION_TAXONOMY_SOURCE,
    source_lineage_hash: Optional[str] = None,
    lineage_ids_by_ticker: Optional[Dict[str, List[str]]] = None,
    lineage_hashes_by_ticker: Optional[Dict[str, List[str]]] = None,
    pattern_id: str = PATTERN_ID,
    allow_undercoverage: bool = False,
    min_coverage_years: float = MIN_PRODUCTION_SECTOR_HISTORY_COVERAGE_YEARS,
) -> PatternAssemblyResult:
    """Assemble M3 PatternInput objects from PIT sector history and returns."""

    result = PatternAssemblyResult(pattern_id=pattern_id)
    lineage_ids_by_ticker = lineage_ids_by_ticker or {}
    lineage_hashes_by_ticker = lineage_hashes_by_ticker or {}
    resolved_universe_cutoff = universe_cutoff_timestamp or cutoff_timestamp

    for snap in snapshots:
        ticker = str(_snap_attr(snap, "ticker") or "").upper()
        snap_id = _snap_attr(snap, "universe_snapshot_id")
        snap_lineage = _snap_attr(snap, "source_lineage_hash")
        assignment = assignments_by_ticker.get(ticker)
        if assignment is None:
            result.insufficient_count += 1
            result.diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=pattern_id,
                diagnostic_type="sector_unknown",
            ))
            continue
        sector_return = sector_returns_by_sector.get(assignment.sector)
        if sector_return is None:
            result.insufficient_count += 1
            result.diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=pattern_id,
                diagnostic_type="sector_return_missing",
                detail=assignment.sector,
            ))
            continue
        if not allow_undercoverage and (
            sector_return.sector_history_coverage_years is None
            or sector_return.sector_history_coverage_years < min_coverage_years
            or sector_return.point_in_time_passed is not True
        ):
            result.insufficient_count += 1
            result.diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=pattern_id,
                diagnostic_type="sector_history_coverage_below_minimum",
                detail=str(sector_return.sector_history_coverage_years),
            ))
            continue
        field_confidence = _current_assignment_field_confidence(
            assignment,
            evidence_date=date.fromisoformat(evidence_session_date),
            min_coverage_years=min_coverage_years,
        )

        fields = [
            _field("sector", assignment.sector, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("industry", assignment.industry, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_taxonomy_source", assignment.source, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_assignment_source", assignment.source, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sic_code", assignment.sic_code, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sic_description", assignment.sic_description, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sic_to_sector_map_version", assignment.sic_to_sector_map_version, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_return_6mo", sector_return.return_6mo, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_return_6mo_ew", sector_return.return_6mo_ew, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("return_1mo", sector_return.return_1mo, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("return_3mo", sector_return.return_3mo, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_return_point_in_time_passed", sector_return.point_in_time_passed, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_return_formation_cohort_passed", sector_return.formation_cohort_passed, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_history_coverage_years", sector_return.sector_history_coverage_years or assignment.sector_history_coverage_years, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_delisting_shumway_adjustment_count", sector_return.delisting_shumway_adjustment_count, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_delisting_unknown_review_count", sector_return.delisting_unknown_review_count, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_delisting_adjustment_audit", sector_return.delisting_adjustment_audit, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_rank", sector_return.sector_rank, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("n_sectors_in_universe", sector_return.n_sectors, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_rank_normalized", sector_return.sector_rank_normalized, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_return_formation_date", sector_return.formation_date.isoformat(), cutoff_timestamp, source_provider, source_lineage_hash),
            _field("field_confidence", field_confidence, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("price", _snap_attr(snap, "price"), resolved_universe_cutoff, "universe", snap_lineage),
            _field("market_cap", _snap_attr(snap, "market_cap"), resolved_universe_cutoff, "universe", snap_lineage),
            _field("market_cap_usd", _snap_attr(snap, "market_cap"), resolved_universe_cutoff, "universe", snap_lineage),
            _field("primary_exchange", _snap_attr(snap, "primary_exchange"), resolved_universe_cutoff, "universe", snap_lineage),
            _field("security_type", _snap_attr(snap, "security_type"), resolved_universe_cutoff, "universe", snap_lineage),
            _field("operating_universe_inclusion", _snap_attr(snap, "operating_universe_inclusion"), resolved_universe_cutoff, "universe", snap_lineage),
            _field("liquidity_score", _snap_attr(snap, "liquidity_score"), resolved_universe_cutoff, "universe", snap_lineage),
            _field("hazard_score_at_signal", _snap_attr(snap, "hazard_score"), resolved_universe_cutoff, "universe", snap_lineage),
            _field("market_data_status", "current", cutoff_timestamp, source_provider, source_lineage_hash),
            _field("halt_status", "clear", cutoff_timestamp, source_provider, source_lineage_hash),
            _field("corporate_action_filter_passed", True, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("trading_date", decision_date, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("evidence_session_date", evidence_session_date, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("next_execution_session", next_execution_session, cutoff_timestamp, source_provider, source_lineage_hash),
        ]
        validated, rejected = validate_assembled_fields(fields, cutoff_timestamp)
        result.rejected_fields.extend(rejected)
        required = [
            "sector",
            "sector_return_6mo",
            "sector_return_point_in_time_passed",
            "sector_return_formation_cohort_passed",
            "sector_rank",
            "n_sectors_in_universe",
            "sector_rank_normalized",
        ]
        missing = [key for key in required if key not in validated]
        if missing:
            result.rejected_count += 1
            result.diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=pattern_id,
                diagnostic_type="missing_m3_fields",
                detail=",".join(missing),
            ))
            continue

        lineage_ids = list(lineage_ids_by_ticker.get(ticker, []))
        lineage_hashes = list(lineage_hashes_by_ticker.get(ticker, []))
        if snap_lineage and snap_lineage not in lineage_hashes:
            lineage_hashes.append(snap_lineage)
        if source_lineage_hash and source_lineage_hash not in lineage_hashes:
            lineage_hashes.append(source_lineage_hash)
        inp = build_pattern_input(
            ticker=ticker,
            pattern_id=pattern_id,
            asof_timestamp=cutoff_timestamp,
            validated_fields=validated,
            lineage_ids=lineage_ids,
            lineage_hashes=lineage_hashes,
            universe_snapshot_id=snap_id,
        )
        result.inputs.append(inp)
        result.assembled_count += 1

    return result


def assemble_m3_shadow_daily(**kwargs: Any) -> PatternAssemblyResult:
    """Assemble M3 shadow inputs under M3S for sub-3-year PIT coverage."""

    kwargs.setdefault("pattern_id", SHADOW_PATTERN_ID)
    kwargs.setdefault("allow_undercoverage", True)
    return assemble_m3_daily(**kwargs)
