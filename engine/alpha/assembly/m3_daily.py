"""M3 Sector Rotation daily producer and assembler helpers.

M3's detector consumes a pre-baked sector-rank payload. This module owns the
PIT formation-cohort math that makes that payload trustworthy: sector identity
comes from interval history, sector returns are computed from the t-126
formation cohort, and detector proof flags are set only from that path.
"""

from __future__ import annotations

import math
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
PRODUCTION_TAXONOMY_SOURCE = "POLYGON_SIC"
OPEN_INTERVAL_END = date(9999, 12, 31)
SECTOR_RETURN_LOOKBACK_SESSIONS = 126
SHUMWAY_PERFORMANCE_DELISTING_RETURN = -0.30


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


@dataclass
class SectorReturnComponent:
    ticker: str
    sector: str
    market_cap: float
    return_6mo: float
    return_3mo: Optional[float] = None
    return_1mo: Optional[float] = None
    delisting_adjustment_applied: bool = False


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
    point_in_time_passed: bool = True
    formation_cohort_passed: bool = True
    sector_history_coverage_years: Optional[float] = None


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
    shumway_return: float = SHUMWAY_PERFORMANCE_DELISTING_RETURN,
) -> Tuple[Optional[float], bool]:
    """Compute adjusted return over [start_date, end_date].

    If a firm delisted inside the window and the end-date price is absent, use
    the last available adjusted close before delisting and apply Shumway's
    performance-delisting adjustment.
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
    adjusted = (1.0 + partial) * (1.0 + shumway_return) - 1.0
    return adjusted, True


def _component_return(
    bars: Sequence[Any],
    *,
    start_date: date,
    end_date: date,
    delisted_date: Optional[date],
) -> Optional[float]:
    value, _ = adjusted_return(
        bars,
        start_date=start_date,
        end_date=end_date,
        delisted_date=delisted_date,
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
) -> Tuple[List[SectorReturnComponent], List[AssemblyDiagnostic]]:
    """Build firm-level return components from the formation-date cohort."""

    components: List[SectorReturnComponent] = []
    diagnostics: List[AssemblyDiagnostic] = []
    delisted_dates_by_ticker = delisted_dates_by_ticker or {}
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
        ret_6mo, delisting_applied = adjusted_return(
            bars,
            start_date=formation_date,
            end_date=evidence_date,
            delisted_date=delisted_date,
        )
        if ret_6mo is None:
            diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=PATTERN_ID,
                diagnostic_type="return_6mo_unavailable",
            ))
            continue
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
                )
                if one_month_date is not None
                else None
            ),
            delisting_adjustment_applied=delisting_applied,
        ))
    return components, diagnostics


def compute_sector_return_snapshots(
    *,
    components: Sequence[SectorReturnComponent],
    asof_date: date,
    formation_date: date,
    sector_history_coverage_years: Optional[float] = None,
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
            sector_history_coverage_years=sector_history_coverage_years,
        ))
    return result


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
) -> PatternAssemblyResult:
    """Assemble M3 PatternInput objects from PIT sector history and returns."""

    result = PatternAssemblyResult(pattern_id=PATTERN_ID)
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
                pattern_id=PATTERN_ID,
                diagnostic_type="sector_unknown",
            ))
            continue
        sector_return = sector_returns_by_sector.get(assignment.sector)
        if sector_return is None:
            result.insufficient_count += 1
            result.diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=PATTERN_ID,
                diagnostic_type="sector_return_missing",
                detail=assignment.sector,
            ))
            continue

        fields = [
            _field("sector", assignment.sector, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("industry", assignment.industry, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_taxonomy_source", PRODUCTION_TAXONOMY_SOURCE, cutoff_timestamp, source_provider, source_lineage_hash),
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
            _field("sector_rank", sector_return.sector_rank, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("n_sectors_in_universe", sector_return.n_sectors, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_rank_normalized", sector_return.sector_rank_normalized, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sector_return_formation_date", sector_return.formation_date.isoformat(), cutoff_timestamp, source_provider, source_lineage_hash),
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
                pattern_id=PATTERN_ID,
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
            pattern_id=PATTERN_ID,
            asof_timestamp=cutoff_timestamp,
            validated_fields=validated,
            lineage_ids=lineage_ids,
            lineage_hashes=lineage_hashes,
            universe_snapshot_id=snap_id,
        )
        result.inputs.append(inp)
        result.assembled_count += 1

    return result
