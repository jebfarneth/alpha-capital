"""M3 PIT sector assignment history pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from alpha.assembly.m3_daily import OPEN_INTERVAL_END, SectorAssignmentSnapshot
from alpha.assembly.m3_sector_map import (
    SIC_TO_SECTOR_MAP_VERSION,
    sector_for_sic,
    normalize_sic_code,
)
from alpha.db.models import (
    CanonicalUniverseScan,
    FirmSectorAssignment,
    FirmSectorAssignmentHistory,
    SectorChangeLog,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.evidence.writer import record_data_lineage
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.market_calendar import resolve_us_equity_session


SOURCE_POLYGON_SIC = "POLYGON_SIC"
SOURCE_FMP_FALLBACK = "FMP_FALLBACK"


@dataclass
class ResolvedSectorAssignment:
    ticker: str
    asof_date: date
    sector: Optional[str]
    source: Optional[str]
    sic_code: Optional[str] = None
    sic_description: Optional[str] = None
    industry: Optional[str] = None
    diagnostics: List[str] = None
    lineage_ids: List[str] = None
    lineage_hashes: List[str] = None

    @property
    def resolved(self) -> bool:
        return bool(self.sector and self.source)


class M3SectorHistoryJob(BaseJob):
    """Backfill or forward-capture M3 sector assignment intervals."""

    job_name = "m3_sector_history"
    job_type = "feature_assembly"

    def __init__(
        self,
        session: Session,
        *,
        polygon_adapter: Any,
        fmp_adapter: Optional[Any] = None,
        run_timestamp: Optional[datetime] = None,
        mode: str = "forward_capture",
        lookback_years: float = 3.0,
        sample_frequency_days: int = 1,
    ):
        if mode not in {"backfill", "forward_capture"}:
            raise ValueError("mode must be backfill or forward_capture")
        if sample_frequency_days < 1:
            raise ValueError("sample_frequency_days must be >= 1")
        self._session = session
        self._polygon_adapter = polygon_adapter
        self._fmp_adapter = fmp_adapter
        self._run_timestamp = run_timestamp
        self._mode = mode
        self._lookback_years = lookback_years
        self._sample_frequency_days = sample_frequency_days

    def run(self, ctx: JobContext) -> JobResult:
        run_timestamp, timestamp_error = _resolve_run_timestamp(
            self._run_timestamp,
            ctx.params.get("run_timestamp"),
            ctx.started_at,
        )
        if timestamp_error:
            return JobResult(status="failed", errors=[{"stage": "params", "message": timestamp_error}])

        session_resolution = resolve_us_equity_session(run_timestamp)
        decision_date = session_resolution.decision_date
        evidence_day = date.fromisoformat(session_resolution.evidence_session_date)
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

        scan_id, _, snapshots, canonical_error = _load_included_canonical_snapshots(
            self._session,
            decision_date,
        )
        if canonical_error:
            return JobResult(
                status="failed",
                metrics={"decision_date": decision_date, "mode": self._mode},
                errors=[{"stage": "canonical_universe", "message": canonical_error}],
            )

        asof_dates = (
            _backfill_sample_dates(
                end_date=evidence_day,
                lookback_years=self._lookback_years,
                frequency_days=self._sample_frequency_days,
            )
            if self._mode == "backfill"
            else [evidence_day]
        )
        tickers = sorted({str(snapshot.ticker).upper() for snapshot in snapshots})
        resolved_count = 0
        unknown_count = 0
        polygon_count = 0
        fallback_count = 0
        change_count = 0
        errors: List[Dict[str, Any]] = []
        for asof_date in asof_dates:
            for ticker in tickers:
                resolved = resolve_sector_assignment(
                    ticker=ticker,
                    asof_date=asof_date,
                    polygon_adapter=self._polygon_adapter,
                    fmp_adapter=self._fmp_adapter,
                    asof_timestamp=run_timestamp,
                    session=self._session,
                    job_run_id=ctx.job_run_id,
                )
                if not resolved.resolved:
                    unknown_count += 1
                    errors.append({
                        "ticker": ticker,
                        "asof_date": asof_date.isoformat(),
                        "stage": "sector_assignment",
                        "diagnostics": resolved.diagnostics or ["sector_unknown"],
                    })
                    continue
                resolved_count += 1
                if resolved.source == SOURCE_POLYGON_SIC:
                    polygon_count += 1
                elif resolved.source == SOURCE_FMP_FALLBACK:
                    fallback_count += 1
                changed = write_sector_assignment_interval(
                    self._session,
                    resolved,
                    job_run_id=ctx.job_run_id,
                )
                if changed:
                    change_count += 1
        metrics = {
            "mode": self._mode,
            "decision_date": decision_date,
            "evidence_session_date": session_resolution.evidence_session_date,
            "scan_id": scan_id,
            "ticker_count": len(tickers),
            "asof_date_count": len(asof_dates),
            "resolved_assignment_count": resolved_count,
            "polygon_sic_assignment_count": polygon_count,
            "fmp_fallback_assignment_count": fallback_count,
            "sector_unknown_count": unknown_count,
            "sector_change_count": change_count,
            "sic_to_sector_map_version": SIC_TO_SECTOR_MAP_VERSION,
            "history_start_date": asof_dates[0].isoformat() if asof_dates else None,
            "history_end_date": asof_dates[-1].isoformat() if asof_dates else None,
        }
        return JobResult(
            status="partial_failed" if errors else "finished",
            metrics=metrics,
            errors=errors,
        )


def resolve_sector_assignment(
    *,
    ticker: str,
    asof_date: date,
    polygon_adapter: Any,
    fmp_adapter: Optional[Any] = None,
    asof_timestamp: Optional[datetime] = None,
    session: Optional[Session] = None,
    job_run_id: Optional[str] = None,
) -> ResolvedSectorAssignment:
    """Resolve one ticker's production M3 sector assignment for an as-of date."""

    ticker = ticker.upper()
    diagnostics: List[str] = []
    lineage_ids: List[str] = []
    lineage_hashes: List[str] = []
    polygon_resp = polygon_adapter.get_ticker_details(
        ticker,
        date_str=asof_date.isoformat(),
        asof=asof_timestamp,
    )
    if session is not None:
        lineage = _record_response_lineage(
            session,
            polygon_resp,
            job_run_id=job_run_id,
            raw_payload={
                "endpoint": "polygon_ticker_details",
                "ticker": ticker,
                "date": asof_date.isoformat(),
                "row": _jsonable(getattr(polygon_resp, "data", None)),
            },
        )
        lineage_ids.append(lineage.data_lineage_id)
        lineage_hashes.append(polygon_resp.lineage.raw_payload_hash)
    if polygon_resp.ok and polygon_resp.data is not None:
        detail = polygon_resp.data
        sic_code = normalize_sic_code(getattr(detail, "sic_code", None))
        sector = sector_for_sic(sic_code)
        if sector:
            return ResolvedSectorAssignment(
                ticker=ticker,
                asof_date=asof_date,
                sector=sector,
                source=SOURCE_POLYGON_SIC,
                sic_code=sic_code,
                sic_description=getattr(detail, "sic_description", None),
                industry=getattr(detail, "sic_description", None),
                diagnostics=diagnostics,
                lineage_ids=lineage_ids,
                lineage_hashes=lineage_hashes,
            )
        diagnostics.append("polygon_sic_null_or_unmapped")
    elif not polygon_resp.ok:
        diagnostics.append(f"polygon_error:{_provider_error(polygon_resp.error).get('message')}")
    else:
        diagnostics.append("polygon_no_data")

    if fmp_adapter is not None:
        fmp_resp = fmp_adapter.get_company_profile(ticker)
        if session is not None:
            lineage = _record_response_lineage(
                session,
                fmp_resp,
                job_run_id=job_run_id,
                raw_payload={
                    "endpoint": "fmp_company_profile",
                    "ticker": ticker,
                    "row": _jsonable(getattr(fmp_resp, "data", None)),
                },
            )
            lineage_ids.append(lineage.data_lineage_id)
            lineage_hashes.append(fmp_resp.lineage.raw_payload_hash)
        if fmp_resp.ok and fmp_resp.data is not None:
            profile = fmp_resp.data
            sector = _clean_sector(getattr(profile, "sector", None))
            if sector:
                return ResolvedSectorAssignment(
                    ticker=ticker,
                    asof_date=asof_date,
                    sector=sector,
                    source=SOURCE_FMP_FALLBACK,
                    industry=getattr(profile, "industry", None),
                    diagnostics=diagnostics,
                    lineage_ids=lineage_ids,
                    lineage_hashes=lineage_hashes,
                )
            diagnostics.append("fmp_sector_null")
        elif not fmp_resp.ok:
            diagnostics.append(f"fmp_error:{_provider_error(fmp_resp.error).get('message')}")
        else:
            diagnostics.append("fmp_no_data")
    return ResolvedSectorAssignment(
        ticker=ticker,
        asof_date=asof_date,
        sector=None,
        source=None,
        diagnostics=diagnostics or ["sector_unknown"],
        lineage_ids=lineage_ids,
        lineage_hashes=lineage_hashes,
    )


def write_sector_assignment_interval(
    session: Session,
    assignment: ResolvedSectorAssignment,
    *,
    job_run_id: Optional[str] = None,
) -> bool:
    """Insert/update one assignment date into exact-as-of interval history.

    Returns True when this date opens a new interval or changes an existing one.
    """

    if not assignment.resolved:
        return False
    ticker = assignment.ticker.upper()
    asof_date = assignment.asof_date
    existing = (
        session.query(FirmSectorAssignmentHistory)
        .filter(
            FirmSectorAssignmentHistory.ticker == ticker,
            FirmSectorAssignmentHistory.valid_from <= asof_date,
            FirmSectorAssignmentHistory.valid_to > asof_date,
        )
        .first()
    )
    changed = False
    if existing is None:
        session.add(FirmSectorAssignmentHistory(
            ticker=ticker,
            sector=assignment.sector,
            sic_code=assignment.sic_code,
            sic_description=assignment.sic_description,
            industry=assignment.industry,
            source=assignment.source,
            sic_to_sector_map_version=SIC_TO_SECTOR_MAP_VERSION,
            valid_from=asof_date,
            valid_to=OPEN_INTERVAL_END,
        ))
        session.add(SectorChangeLog(
            ticker=ticker,
            old_sector=None,
            new_sector=assignment.sector,
            old_sic_code=None,
            new_sic_code=assignment.sic_code,
            old_source=None,
            new_source=assignment.source,
            sic_to_sector_map_version=SIC_TO_SECTOR_MAP_VERSION,
            change_date=asof_date,
            job_run_id=job_run_id,
            diagnostic_json=json.dumps({"reason": "initial_assignment"}),
        ))
        changed = True
    elif _same_assignment(existing, assignment):
        pass
    elif existing.valid_from == asof_date:
        session.add(SectorChangeLog(
            ticker=ticker,
            old_sector=existing.sector,
            new_sector=assignment.sector,
            old_sic_code=existing.sic_code,
            new_sic_code=assignment.sic_code,
            old_source=existing.source,
            new_source=assignment.source,
            sic_to_sector_map_version=SIC_TO_SECTOR_MAP_VERSION,
            change_date=asof_date,
            job_run_id=job_run_id,
            diagnostic_json=json.dumps({"reason": "same_day_reclassification"}),
        ))
        existing.sector = assignment.sector
        existing.sic_code = assignment.sic_code
        existing.sic_description = assignment.sic_description
        existing.industry = assignment.industry
        existing.source = assignment.source
        existing.sic_to_sector_map_version = SIC_TO_SECTOR_MAP_VERSION
        changed = True
    else:
        old_valid_to = existing.valid_to
        existing.valid_to = asof_date
        session.add(FirmSectorAssignmentHistory(
            ticker=ticker,
            sector=assignment.sector,
            sic_code=assignment.sic_code,
            sic_description=assignment.sic_description,
            industry=assignment.industry,
            source=assignment.source,
            sic_to_sector_map_version=SIC_TO_SECTOR_MAP_VERSION,
            valid_from=asof_date,
            valid_to=old_valid_to,
        ))
        session.add(SectorChangeLog(
            ticker=ticker,
            old_sector=existing.sector,
            new_sector=assignment.sector,
            old_sic_code=existing.sic_code,
            new_sic_code=assignment.sic_code,
            old_source=existing.source,
            new_source=assignment.source,
            sic_to_sector_map_version=SIC_TO_SECTOR_MAP_VERSION,
            change_date=asof_date,
            job_run_id=job_run_id,
            diagnostic_json=json.dumps({"reason": "detected_sector_change"}),
        ))
        changed = True

    current = session.get(FirmSectorAssignment, ticker)
    if current is None:
        current = FirmSectorAssignment(
            ticker=ticker,
            sector=assignment.sector,
            industry=assignment.industry,
            source=assignment.source,
            classification_date=asof_date,
            last_verified=asof_date,
        )
        session.add(current)
    elif current.classification_date <= asof_date:
        current.sector = assignment.sector
        current.industry = assignment.industry
        current.source = assignment.source
        current.classification_date = asof_date
        current.last_verified = asof_date
    session.flush()
    return changed


def load_sector_assignments_at(
    session: Session,
    *,
    tickers: Iterable[str],
    asof_date: date,
) -> Dict[str, SectorAssignmentSnapshot]:
    """Load sector assignment intervals active at an as-of date."""

    normalized = {ticker.upper() for ticker in tickers}
    if not normalized:
        return {}
    rows = (
        session.query(FirmSectorAssignmentHistory)
        .filter(
            FirmSectorAssignmentHistory.ticker.in_(normalized),
            FirmSectorAssignmentHistory.valid_from <= asof_date,
            FirmSectorAssignmentHistory.valid_to > asof_date,
        )
        .all()
    )
    coverage = _coverage_years_by_ticker(session, normalized, asof_date)
    return {
        row.ticker.upper(): SectorAssignmentSnapshot(
            ticker=row.ticker.upper(),
            sector=row.sector,
            source=row.source,
            valid_from=row.valid_from,
            valid_to=row.valid_to,
            sic_code=row.sic_code,
            sic_description=row.sic_description,
            industry=row.industry,
            sic_to_sector_map_version=row.sic_to_sector_map_version,
            sector_history_coverage_years=coverage.get(row.ticker.upper()),
        )
        for row in rows
    }


def _same_assignment(
    row: FirmSectorAssignmentHistory,
    assignment: ResolvedSectorAssignment,
) -> bool:
    return (
        row.sector == assignment.sector
        and (row.sic_code or None) == (assignment.sic_code or None)
        and row.source == assignment.source
        and row.sic_to_sector_map_version == SIC_TO_SECTOR_MAP_VERSION
    )


def _coverage_years_by_ticker(
    session: Session,
    tickers: Iterable[str],
    asof_date: date,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for ticker in tickers:
        first = (
            session.query(FirmSectorAssignmentHistory.valid_from)
            .filter(FirmSectorAssignmentHistory.ticker == ticker)
            .order_by(FirmSectorAssignmentHistory.valid_from.asc())
            .first()
        )
        if first:
            out[ticker] = max(0.0, (asof_date - first[0]).days / 365.25)
    return out


def _backfill_sample_dates(
    *,
    end_date: date,
    lookback_years: float,
    frequency_days: int,
) -> List[date]:
    start = end_date - timedelta(days=int(round(lookback_years * 365.25)))
    dates: List[date] = []
    cursor = start
    while cursor < end_date:
        dates.append(cursor)
        cursor += timedelta(days=frequency_days)
    if not dates or dates[-1] != end_date:
        dates.append(end_date)
    return dates


def _load_included_canonical_snapshots(
    session: Session,
    trading_date: str,
) -> Tuple[Optional[str], Optional[datetime], List[UniverseSnapshot], Optional[str]]:
    canonical = (
        session.query(CanonicalUniverseScan)
        .filter(CanonicalUniverseScan.trading_date == trading_date)
        .first()
    )
    if canonical is None:
        return None, None, [], f"no canonical universe scan for trading_date={trading_date}"
    scan = session.get(UniverseScan, canonical.scan_id)
    if scan is None:
        return None, None, [], f"canonical scan_id {canonical.scan_id} not found"
    snapshots = (
        session.query(UniverseSnapshot)
        .filter(
            UniverseSnapshot.scan_id == canonical.scan_id,
            UniverseSnapshot.operating_universe_inclusion.is_(True),
        )
        .all()
    )
    return canonical.scan_id, _ensure_aware(scan.asof_timestamp), snapshots, None


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


def _ensure_aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _clean_sector(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() in {"unknown", "none", "null"}:
        return None
    return text


def _record_response_lineage(
    session: Session,
    resp: Any,
    *,
    job_run_id: Optional[str],
    raw_payload: Dict[str, Any],
) -> Any:
    return record_data_lineage(
        session,
        provider=resp.lineage.provider,
        endpoint=resp.lineage.endpoint,
        asof_timestamp=resp.lineage.asof_timestamp,
        raw_payload=raw_payload,
        raw_payload_hash=resp.lineage.raw_payload_hash,
        source_authority=resp.lineage.source_authority,
        data_quality_flags=resp.lineage.data_quality_flags,
        job_run_id=job_run_id,
    )


def _provider_error(error: Any) -> Dict[str, Any]:
    if error is None:
        return {"message": "unknown provider error"}
    return {
        "error_type": getattr(error, "error_type", None),
        "status_code": getattr(error, "status_code", None),
        "message": getattr(error, "message", str(error)),
        "retryable": getattr(error, "retryable", None),
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(val)
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)
