"""Production M3 sector-rotation daily feature assembly wiring."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from alpha.assembly.m3_daily import (
    PATTERN_ID,
    SECTOR_RETURN_LOOKBACK_SESSIONS,
    SectorReturnSnapshot,
    assemble_m3_daily,
    build_sector_return_components,
    compute_sector_return_snapshots,
    nth_previous_session,
)
from alpha.assembly.m3_sector_map import SIC_TO_SECTOR_MAP_VERSION
from alpha.data.contracts import stable_hash
from alpha.db.models import SectorReturnDaily
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.detector_orchestration import DetectorOrchestrationJob
from alpha.jobs.m3_sector_history import (
    SOURCE_FMP_FALLBACK,
    SOURCE_POLYGON_SIC,
    _jsonable,
    _load_included_canonical_snapshots,
    _provider_error,
    _record_response_lineage,
    _resolve_run_timestamp,
    load_sector_assignments_at,
    resolve_sector_assignment,
    write_sector_assignment_interval,
)
from alpha.market_calendar import (
    resolve_us_equity_session,
    us_equity_session_close_timestamp,
)
from alpha.patterns.m3 import M3Detector


class M3DailyAssemblyJob(BaseJob):
    """Run daily M3 from PIT sector history through persisted orchestration."""

    job_name = "m3_daily_feature_assembly"
    job_type = "feature_assembly"

    def __init__(
        self,
        session: Session,
        *,
        polygon_adapter: Any,
        fmp_adapter: Any,
        run_timestamp: Optional[datetime] = None,
        sector_lookback_sessions: int = SECTOR_RETURN_LOOKBACK_SESSIONS,
        refresh_sector_history: bool = True,
    ):
        self._session = session
        self._polygon_adapter = polygon_adapter
        self._fmp_adapter = fmp_adapter
        self._run_timestamp = run_timestamp
        self._sector_lookback_sessions = sector_lookback_sessions
        self._refresh_sector_history = refresh_sector_history

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

        session_resolution = resolve_us_equity_session(run_timestamp)
        decision_date = session_resolution.decision_date
        evidence_session_date = session_resolution.evidence_session_date
        evidence_day = date.fromisoformat(evidence_session_date)
        cutoff_timestamp = us_equity_session_close_timestamp(evidence_day)

        requested_trading_date = ctx.params.get("trading_date")
        if requested_trading_date and requested_trading_date != decision_date:
            return JobResult(
                status="failed",
                metrics={"decision_date": decision_date},
                errors=[{
                    "stage": "params",
                    "message": (
                        "trading_date must match resolver decision_date; "
                        f"got {requested_trading_date}, resolved {decision_date}"
                    ),
                }],
            )

        scan_id, scan_asof_timestamp, snapshots, canonical_error = (
            _load_included_canonical_snapshots(self._session, decision_date)
        )
        if canonical_error:
            return JobResult(
                status="failed",
                metrics=_base_metrics(session_resolution, scan_id=None),
                errors=[{"stage": "canonical_universe", "message": canonical_error}],
            )

        tickers = [str(snapshot.ticker).upper() for snapshot in snapshots]
        lineage_ids_by_ticker: Dict[str, List[str]] = {}
        lineage_hashes_by_ticker: Dict[str, List[str]] = {}
        refresh_errors: List[Dict[str, Any]] = []
        refresh_counts = {
            "resolved": 0,
            "polygon_sic": 0,
            "fmp_fallback": 0,
            "unknown": 0,
            "changes": 0,
        }
        if self._refresh_sector_history:
            for ticker in tickers:
                resolved = resolve_sector_assignment(
                    ticker=ticker,
                    asof_date=evidence_day,
                    polygon_adapter=self._polygon_adapter,
                    fmp_adapter=self._fmp_adapter,
                    asof_timestamp=run_timestamp,
                    session=self._session,
                    job_run_id=ctx.job_run_id,
                )
                lineage_ids_by_ticker[ticker] = list(resolved.lineage_ids or [])
                lineage_hashes_by_ticker[ticker] = list(resolved.lineage_hashes or [])
                if not resolved.resolved:
                    refresh_counts["unknown"] += 1
                    refresh_errors.append({
                        "ticker": ticker,
                        "stage": "sector_assignment",
                        "diagnostics": resolved.diagnostics or ["sector_unknown"],
                    })
                    continue
                refresh_counts["resolved"] += 1
                if resolved.source == SOURCE_POLYGON_SIC:
                    refresh_counts["polygon_sic"] += 1
                elif resolved.source == SOURCE_FMP_FALLBACK:
                    refresh_counts["fmp_fallback"] += 1
                if write_sector_assignment_interval(
                    self._session,
                    resolved,
                    job_run_id=ctx.job_run_id,
                ):
                    refresh_counts["changes"] += 1

        formation_date = nth_previous_session(evidence_day, self._sector_lookback_sessions)
        one_month_date = nth_previous_session(evidence_day, 21)
        three_month_date = nth_previous_session(evidence_day, 63)
        formation_scan_id, _, formation_snapshots, formation_error = (
            _load_included_canonical_snapshots(
                self._session,
                formation_date.isoformat(),
            )
        )
        if formation_error:
            return JobResult(
                status="partial_failed",
                metrics=_base_metrics(
                    session_resolution,
                    scan_id=scan_id,
                ) | {
                    "formation_date": formation_date.isoformat(),
                    "formation_scan_id": formation_scan_id,
                    "sector_return_computed": False,
                    "refresh": refresh_counts,
                },
                errors=[{"stage": "formation_universe", "message": formation_error}],
            )

        formation_tickers = [str(snapshot.ticker).upper() for snapshot in formation_snapshots]
        formation_assignments = load_sector_assignments_at(
            self._session,
            tickers=formation_tickers,
            asof_date=formation_date,
        )
        current_assignments = load_sector_assignments_at(
            self._session,
            tickers=tickers,
            asof_date=evidence_day,
        )

        bars_by_ticker, price_fetch_errors, price_lineages = _fetch_price_bars(
            self._session,
            self._fmp_adapter,
            tickers=formation_tickers,
            from_date=formation_date,
            to_date=evidence_day,
            asof=cutoff_timestamp,
            job_run_id=ctx.job_run_id,
        )
        for ticker, lineages in price_lineages.items():
            lineage_ids_by_ticker.setdefault(ticker, []).extend(lineages["ids"])
            lineage_hashes_by_ticker.setdefault(ticker, []).extend(lineages["hashes"])
        delisted_dates = _resolve_delisted_dates_for_missing_end_prices(
            self._session,
            self._polygon_adapter,
            tickers=formation_tickers,
            bars_by_ticker=bars_by_ticker,
            evidence_day=evidence_day,
            asof=run_timestamp,
            job_run_id=ctx.job_run_id,
        )
        components, component_diagnostics = build_sector_return_components(
            formation_snapshots=formation_snapshots,
            assignments_by_ticker=formation_assignments,
            bars_by_ticker=bars_by_ticker,
            evidence_date=evidence_day,
            formation_date=formation_date,
            one_month_date=one_month_date,
            three_month_date=three_month_date,
            delisted_dates_by_ticker=delisted_dates,
        )
        coverage_years = _min_coverage_years(current_assignments.values())
        sector_returns = compute_sector_return_snapshots(
            components=components,
            asof_date=evidence_day,
            formation_date=formation_date,
            sector_history_coverage_years=coverage_years,
        )
        _persist_sector_returns(self._session, sector_returns)
        sector_returns_by_sector = {row.sector: row for row in sector_returns}

        assembly = assemble_m3_daily(
            snapshots=snapshots,
            assignments_by_ticker=current_assignments,
            sector_returns_by_sector=sector_returns_by_sector,
            cutoff_timestamp=cutoff_timestamp,
            universe_cutoff_timestamp=scan_asof_timestamp,
            decision_date=decision_date,
            evidence_session_date=evidence_session_date,
            next_execution_session=session_resolution.next_execution_session,
            source_lineage_hash=stable_hash({
                "pattern_id": PATTERN_ID,
                "scan_id": scan_id,
                "formation_scan_id": formation_scan_id,
                "formation_date": formation_date.isoformat(),
                "sector_return_count": len(sector_returns),
                "sic_to_sector_map_version": SIC_TO_SECTOR_MAP_VERSION,
            }),
            lineage_ids_by_ticker=lineage_ids_by_ticker,
            lineage_hashes_by_ticker=lineage_hashes_by_ticker,
        )
        orchestration = DetectorOrchestrationJob(
            self._session,
            detectors=[M3Detector()],
            trading_date=decision_date,
            assembled_inputs={"M3": assembly.inputs},
        )
        orchestration_result = orchestration.run(ctx)

        metrics = _base_metrics(session_resolution, scan_id=scan_id)
        metrics.update({
            "formation_date": formation_date.isoformat(),
            "formation_scan_id": formation_scan_id,
            "formation_universe_size": len(formation_snapshots),
            "formation_sector_assignment_count": len(formation_assignments),
            "current_sector_assignment_count": len(current_assignments),
            "sector_return_count": len(sector_returns),
            "sector_return_component_count": len(components),
            "delisted_shumway_adjustment_count": sum(
                1 for component in components if component.delisting_adjustment_applied
            ),
            "price_fetch_error_count": len(price_fetch_errors),
            "sector_history_refresh": refresh_counts,
            "sic_to_sector_map_version": SIC_TO_SECTOR_MAP_VERSION,
            "assembly": _assembly_metrics(assembly),
            "orchestration": orchestration_result.metrics,
        })
        errors = list(refresh_errors)
        errors.extend(price_fetch_errors)
        errors.extend({
            "ticker": diag.ticker,
            "stage": "sector_return_component",
            "diagnostic_type": diag.diagnostic_type,
            "detail": diag.detail,
        } for diag in component_diagnostics)
        errors.extend(orchestration_result.errors or [])
        status = (
            "partial_failed"
            if orchestration_result.status == "partial_failed"
            else "finished"
        )
        return JobResult(
            status=status,
            metrics=metrics,
            input_hashes={
                "scan_id": scan_id,
                "formation_scan_id": formation_scan_id,
                "decision_date": decision_date,
                "formation_date": formation_date.isoformat(),
            },
            output_hashes={
                "sector_returns": stable_hash([
                    {
                        "sector": row.sector,
                        "return_6mo": row.return_6mo,
                        "rank": row.sector_rank,
                    }
                    for row in sector_returns
                ]),
                **(orchestration_result.output_hashes or {}),
            },
            errors=errors,
        )


def _base_metrics(session_resolution: Any, *, scan_id: Optional[str]) -> Dict[str, Any]:
    return {
        "decision_date": session_resolution.decision_date,
        "evidence_session_date": session_resolution.evidence_session_date,
        "next_execution_session": session_resolution.next_execution_session,
        "session_resolution": asdict(session_resolution),
        "scan_id": scan_id,
    }


def _assembly_metrics(result: Any) -> Dict[str, Any]:
    return {
        "pattern_id": result.pattern_id,
        "assembled_count": result.assembled_count,
        "insufficient_count": result.insufficient_count,
        "rejected_count": result.rejected_count,
        "diagnostic_count": len(result.diagnostics),
        "rejected_field_count": len(result.rejected_fields),
        "diagnostics": [
            {
                "ticker": diag.ticker,
                "pattern_id": diag.pattern_id,
                "diagnostic_type": diag.diagnostic_type,
                "detail": diag.detail,
            }
            for diag in result.diagnostics[:50]
        ],
    }


def _fetch_price_bars(
    session: Session,
    adapter: Any,
    *,
    tickers: Sequence[str],
    from_date: date,
    to_date: date,
    asof: datetime,
    job_run_id: str,
) -> Tuple[Dict[str, List[Any]], List[Dict[str, Any]], Dict[str, Dict[str, List[str]]]]:
    bars_by_ticker: Dict[str, List[Any]] = {}
    errors: List[Dict[str, Any]] = []
    lineages: Dict[str, Dict[str, List[str]]] = {}
    for ticker in tickers:
        resp = adapter.get_historical_price(
            ticker,
            from_date=from_date,
            to_date=to_date,
            asof=asof,
            adjusted=True,
            require_split_adjusted_close=False,
            require_adjusted_close=True,
        )
        lineage = _record_response_lineage(
            session,
            resp,
            job_run_id=job_run_id,
            raw_payload={
                "endpoint": "fmp_historical_price_adjusted",
                "ticker": ticker,
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "row_count": len(resp.data or []) if resp.ok else None,
            },
        )
        lineages[ticker] = {
            "ids": [lineage.data_lineage_id],
            "hashes": [resp.lineage.raw_payload_hash],
        }
        if not resp.ok:
            errors.append({"ticker": ticker, "stage": "fmp_historical_price", **_provider_error(resp.error)})
            continue
        bars_by_ticker[ticker] = list(resp.data or [])
    return bars_by_ticker, errors, lineages


def _resolve_delisted_dates_for_missing_end_prices(
    session: Session,
    polygon_adapter: Any,
    *,
    tickers: Sequence[str],
    bars_by_ticker: Dict[str, Sequence[Any]],
    evidence_day: date,
    asof: datetime,
    job_run_id: str,
) -> Dict[str, date]:
    out: Dict[str, date] = {}
    for ticker in tickers:
        bars = bars_by_ticker.get(ticker, ())
        if _has_bar_on_date(bars, evidence_day):
            continue
        resp = polygon_adapter.get_ticker_details(
            ticker,
            date_str=evidence_day.isoformat(),
            asof=asof,
        )
        _record_response_lineage(
            session,
            resp,
            job_run_id=job_run_id,
            raw_payload={
                "endpoint": "polygon_ticker_details_delisting_probe",
                "ticker": ticker,
                "date": evidence_day.isoformat(),
                "row": _jsonable(getattr(resp, "data", None)),
            },
        )
        if resp.ok and resp.data is not None:
            delisted = _parse_delisted_date(getattr(resp.data, "delisted_utc", None))
            if delisted is not None:
                out[ticker] = delisted
        elif not resp.ok:
            if _provider_error(resp.error).get("status_code") == 404:
                last_bar_date = _last_bar_date(bars)
                if last_bar_date is not None and last_bar_date <= evidence_day:
                    out[ticker] = last_bar_date
    return out


def _has_bar_on_date(bars: Sequence[Any], day: date) -> bool:
    for bar in bars:
        value = getattr(bar, "date", None)
        if isinstance(value, str) and value == day.isoformat():
            return True
        if value == day:
            return True
    return False


def _last_bar_date(bars: Sequence[Any]) -> Optional[date]:
    days: List[date] = []
    for bar in bars:
        value = getattr(bar, "date", None)
        if isinstance(value, date):
            days.append(value)
        elif isinstance(value, str) and value:
            try:
                days.append(date.fromisoformat(value[:10]))
            except ValueError:
                continue
    return max(days) if days else None


def _parse_delisted_date(value: Any) -> Optional[date]:
    if not value:
        return None
    text = str(value)
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _persist_sector_returns(
    session: Session,
    rows: Sequence[SectorReturnSnapshot],
) -> None:
    for row in rows:
        existing = session.get(SectorReturnDaily, (row.date, row.sector))
        payload = {
            "return_6mo": row.return_6mo,
            "return_6mo_ew": row.return_6mo_ew,
            "return_1mo": row.return_1mo,
            "return_3mo": row.return_3mo,
            "sector_rank": row.sector_rank,
            "sector_rank_normalized": row.sector_rank_normalized,
            "n_sectors": row.n_sectors,
            "n_firms_in_sector": row.n_firms_in_sector,
            "total_market_cap_in_sector": row.total_market_cap_in_sector,
            "source": row.source,
            "sic_to_sector_map_version": row.sic_to_sector_map_version,
            "formation_date": row.formation_date,
            "point_in_time_passed": row.point_in_time_passed,
            "formation_cohort_passed": row.formation_cohort_passed,
            "sector_history_coverage_years": row.sector_history_coverage_years,
        }
        if existing is None:
            session.add(SectorReturnDaily(date=row.date, sector=row.sector, **payload))
        else:
            for key, value in payload.items():
                setattr(existing, key, value)
    session.flush()


def _min_coverage_years(assignments: Sequence[Any]) -> Optional[float]:
    values = [
        assignment.sector_history_coverage_years
        for assignment in assignments
        if assignment.sector_history_coverage_years is not None
    ]
    return min(values) if values else None
