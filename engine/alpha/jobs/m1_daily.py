"""Production M1 PEAD daily feature assembly wiring.

This job reads the canonical operating-universe scan, fetches M1-specific
earnings data, computes Foster SUE and return-based H-M friction metrics, and
persists M1 signal firings through detector orchestration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from alpha.assembly.m1_daily import (
    MARKET_FACTOR_SYMBOL,
    FosterComputation,
    PriceDelayComputation,
    assemble_m1_daily,
    compute_foster_sue,
    compute_price_delay_metric,
    effective_announcement_session,
    rank_friction_metrics,
    trading_session_distance,
)
from alpha.data.contracts import stable_hash
from alpha.data.fmp import (
    EARNINGS_CALENDAR_ENDPOINT,
    EARNINGS_HISTORY_ENDPOINT,
    HISTORICAL_PRICE_FULL_ENDPOINT,
    FmpEarningsCalendarEvent,
)
from alpha.db.models import (
    CanonicalUniverseScan,
    M1EarningsEvent,
    M1FrictionSnapshot,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.evidence.writer import record_data_lineage
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.detector_orchestration import DetectorOrchestrationJob
from alpha.jobs.universe_builder import market_cap_bucket_counts
from alpha.market_calendar import (
    next_us_equity_session,
    previous_us_equity_session,
    resolve_us_equity_session,
    us_equity_session_close_timestamp,
)
from alpha.patterns.m1 import M1Detector


@dataclass
class CalendarPageFetch:
    request_date: date
    response: Any


@dataclass
class CalendarWindowFetch:
    events: List[FmpEarningsCalendarEvent]
    pages: List[CalendarPageFetch]
    coverage: Dict[str, Any]
    errors: List[Dict[str, Any]]

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.coverage.get("covers_requested_window"))


class M1DailyAssemblyJob(BaseJob):
    """Run daily M1 from canonical universe through persisted orchestration."""

    job_name = "m1_daily_feature_assembly"
    job_type = "feature_assembly"

    def __init__(
        self,
        session: Session,
        *,
        adapter: Any,
        run_timestamp: Optional[datetime] = None,
        earnings_window_sessions: int = 15,
        next_earnings_calendar_days: int = 140,
        price_lookback_calendar_days: int = 430,
    ):
        self._session = session
        self._adapter = adapter
        self._run_timestamp = run_timestamp
        self._earnings_window_sessions = earnings_window_sessions
        self._next_earnings_calendar_days = next_earnings_calendar_days
        self._price_lookback_calendar_days = price_lookback_calendar_days

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
        earnings_from = _session_window_start(evidence_day, self._earnings_window_sessions)
        forward_to = evidence_day + timedelta(days=self._next_earnings_calendar_days)
        price_from = evidence_day - timedelta(days=self._price_lookback_calendar_days)

        requested_trading_date = ctx.params.get("trading_date")
        if requested_trading_date and requested_trading_date != decision_date:
            return JobResult(
                status="failed",
                metrics={
                    "decision_date": decision_date,
                    "requested_trading_date": requested_trading_date,
                },
                errors=[{
                    "stage": "params",
                    "message": (
                        "trading_date must match resolver decision_date; "
                        f"got {requested_trading_date}, resolved {decision_date}"
                    ),
                }],
            )

        scan_id, scan_asof_timestamp, snapshots, canonical_error = _load_included_canonical_snapshots(
            self._session,
            decision_date,
        )
        if canonical_error:
            return JobResult(
                status="failed",
                metrics={
                    "decision_date": decision_date,
                    "evidence_session_date": evidence_session_date,
                    "session_resolution": asdict(session_resolution),
                },
                errors=[{"stage": "canonical_universe", "message": canonical_error}],
            )
        included_market_cap_bucket_counts = market_cap_bucket_counts(snapshots)
        snapshot_by_ticker = {snapshot.ticker.upper(): snapshot for snapshot in snapshots}

        trailing_fetch = _fetch_earnings_calendar_window(
            self._adapter,
            from_date=earnings_from,
            to_date=evidence_day,
            asof=cutoff_timestamp,
        )
        trailing_lineages = _record_calendar_page_lineages(
            self._session,
            trailing_fetch,
            job_run_id=ctx.job_run_id,
            coverage_kind="trailing",
        )
        if not trailing_fetch.ok:
            return JobResult(
                status="failed",
                metrics=_base_metrics(
                    session_resolution,
                    scan_id=scan_id,
                    included_universe_size=len(snapshots),
                    included_market_cap_bucket_counts=included_market_cap_bucket_counts,
                    earnings_from=earnings_from,
                    earnings_to=evidence_day,
                    price_from=price_from,
                ) | {"earnings_calendar_coverage": trailing_fetch.coverage},
                errors=[
                    {"stage": "earnings_calendar", **err}
                    for err in trailing_fetch.errors
                ] or [{
                    "stage": "earnings_calendar",
                    "message": "earnings calendar coverage incomplete",
                    "coverage": trailing_fetch.coverage,
                }],
            )

        forward_from = next_us_equity_session(evidence_day + timedelta(days=1))
        forward_fetch = _fetch_earnings_calendar_window(
            self._adapter,
            from_date=forward_from,
            to_date=forward_to,
            asof=cutoff_timestamp,
        )
        forward_lineages = _record_calendar_page_lineages(
            self._session,
            forward_fetch,
            job_run_id=ctx.job_run_id,
            coverage_kind="forward",
        )
        if not forward_fetch.ok:
            return JobResult(
                status="failed",
                metrics=_base_metrics(
                    session_resolution,
                    scan_id=scan_id,
                    included_universe_size=len(snapshots),
                    included_market_cap_bucket_counts=included_market_cap_bucket_counts,
                    earnings_from=earnings_from,
                    earnings_to=evidence_day,
                    price_from=price_from,
                ) | {"forward_earnings_calendar_coverage": forward_fetch.coverage},
                errors=[
                    {"stage": "forward_earnings_calendar", **err}
                    for err in forward_fetch.errors
                ] or [{
                    "stage": "forward_earnings_calendar",
                    "message": "forward earnings calendar coverage incomplete",
                    "coverage": forward_fetch.coverage,
                }],
            )

        trailing_events = _select_trailing_announcements(
            trailing_fetch.events,
            snapshot_by_ticker=snapshot_by_ticker,
            evidence_day=evidence_day,
            window_start=earnings_from,
        )
        next_earnings_by_ticker = _next_earnings_distance(
            forward_fetch.events,
            evidence_day=evidence_day,
            snapshot_by_ticker=snapshot_by_ticker,
        )
        trailing_lineage_ids = [lineage.data_lineage_id for lineage in trailing_lineages]
        trailing_lineage_hashes = [
            page.response.lineage.raw_payload_hash for page in trailing_fetch.pages
        ]

        eps_fetch_errors: List[Dict[str, Any]] = []
        foster_by_ticker: Dict[str, FosterComputation] = {}
        lineage_ids_by_ticker: Dict[str, List[str]] = {}
        lineage_hashes_by_ticker: Dict[str, List[str]] = {}
        for ticker, event in trailing_events.items():
            lineage_ids_by_ticker.setdefault(ticker, []).extend(trailing_lineage_ids)
            lineage_hashes_by_ticker.setdefault(ticker, []).extend(trailing_lineage_hashes)
            eps_resp = self._adapter.get_earnings_history(
                ticker,
                limit=40,
                asof=cutoff_timestamp,
            )
            eps_lineage = _record_response_lineage(
                self._session,
                eps_resp,
                job_run_id=ctx.job_run_id,
                raw_payload={
                    "endpoint": EARNINGS_HISTORY_ENDPOINT,
                    "symbol": ticker,
                    "rows": _jsonable(eps_resp.data),
                },
            )
            lineage_ids_by_ticker.setdefault(ticker, []).append(eps_lineage.data_lineage_id)
            lineage_hashes_by_ticker.setdefault(ticker, []).append(eps_resp.lineage.raw_payload_hash)
            if not eps_resp.ok:
                eps_fetch_errors.append({"ticker": ticker, **_provider_error(eps_resp.error)})
                computation = FosterComputation(
                    ticker=ticker,
                    status="insufficient_history",
                    diagnostics=[getattr(eps_resp.error, "message", "eps_history_unavailable")],
                    event_id=None,
                    announcement_date=event.date,
                )
            else:
                computation = compute_foster_sue(
                    event=event,
                    eps_history=eps_resp.data or [],
                    effective_session=effective_announcement_session(event),
                    asof_timestamp=cutoff_timestamp,
                )
            foster_by_ticker[ticker] = computation
            _persist_m1_earnings_event(
                self._session,
                scan_id=scan_id,
                snapshot=snapshot_by_ticker.get(ticker),
                job_run_id=ctx.job_run_id,
                computation=computation,
                lineage_ids=lineage_ids_by_ticker.get(ticker, []),
            )

        market_resp = self._adapter.get_historical_price(
            MARKET_FACTOR_SYMBOL,
            from_date=price_from,
            to_date=evidence_day,
            asof=cutoff_timestamp,
            adjusted=False,
            require_split_adjusted_close=True,
        )
        market_lineage = _record_response_lineage(
            self._session,
            market_resp,
            job_run_id=ctx.job_run_id,
            raw_payload={
                "endpoint": HISTORICAL_PRICE_FULL_ENDPOINT,
                "symbol": MARKET_FACTOR_SYMBOL,
                "from": price_from.isoformat(),
                "to": evidence_session_date,
                "bars": _jsonable(market_resp.data),
            },
        )
        if not market_resp.ok:
            return JobResult(
                status="failed",
                metrics=_base_metrics(
                    session_resolution,
                    scan_id=scan_id,
                    included_universe_size=len(snapshots),
                    included_market_cap_bucket_counts=included_market_cap_bucket_counts,
                    earnings_from=earnings_from,
                    earnings_to=evidence_day,
                    price_from=price_from,
                ),
                errors=[{"stage": "market_factor_price", **_provider_error(market_resp.error)}],
            )

        friction_by_ticker: Dict[str, PriceDelayComputation] = {}
        price_fetch_errors: List[Dict[str, Any]] = []
        for snapshot in snapshots:
            ticker = snapshot.ticker.upper()
            price_resp = self._adapter.get_historical_price(
                ticker,
                from_date=price_from,
                to_date=evidence_day,
                asof=cutoff_timestamp,
                adjusted=False,
                require_split_adjusted_close=True,
            )
            price_lineage = _record_response_lineage(
                self._session,
                price_resp,
                job_run_id=ctx.job_run_id,
                raw_payload={
                    "endpoint": HISTORICAL_PRICE_FULL_ENDPOINT,
                    "symbol": ticker,
                    "from": price_from.isoformat(),
                    "to": evidence_session_date,
                    "bars": _jsonable(price_resp.data),
                },
            )
            lineage_ids_by_ticker.setdefault(ticker, []).append(price_lineage.data_lineage_id)
            lineage_hashes_by_ticker.setdefault(ticker, []).append(price_resp.lineage.raw_payload_hash)
            lineage_ids_by_ticker[ticker].append(market_lineage.data_lineage_id)
            lineage_hashes_by_ticker[ticker].append(market_resp.lineage.raw_payload_hash)
            if not price_resp.ok:
                price_fetch_errors.append({"ticker": ticker, **_provider_error(price_resp.error)})
                metric = PriceDelayComputation(
                    ticker=ticker,
                    status="insufficient_price_history",
                    diagnostics=[getattr(price_resp.error, "message", "price_history_unavailable")],
                )
            else:
                metric = compute_price_delay_metric(
                    ticker=ticker,
                    stock_bars=price_resp.data or [],
                    market_bars=market_resp.data or [],
                )
            friction_by_ticker[ticker] = metric

        rank_friction_metrics(friction_by_ticker)
        for ticker, metric in friction_by_ticker.items():
            _persist_m1_friction_snapshot(
                self._session,
                scan_id=scan_id,
                snapshot=snapshot_by_ticker.get(ticker),
                job_run_id=ctx.job_run_id,
                metric=metric,
                lineage_ids=lineage_ids_by_ticker.get(ticker, []),
            )

        assembly = assemble_m1_daily(
            snapshots=snapshots,
            foster_by_ticker=foster_by_ticker,
            friction_by_ticker=friction_by_ticker,
            next_earnings_by_ticker=next_earnings_by_ticker,
            cutoff_timestamp=cutoff_timestamp,
            universe_cutoff_timestamp=scan_asof_timestamp,
            decision_date=decision_date,
            evidence_session_date=evidence_session_date,
            next_execution_session=session_resolution.next_execution_session,
            source_provider="FMP",
            source_lineage_hash=stable_hash(trailing_lineage_hashes),
            lineage_ids_by_ticker=lineage_ids_by_ticker,
            lineage_hashes_by_ticker=lineage_hashes_by_ticker,
        )

        orchestration = DetectorOrchestrationJob(
            self._session,
            detectors=[M1Detector()],
            trading_date=decision_date,
            assembled_inputs={"M1": assembly.inputs},
        )
        orchestration_result = orchestration.run(ctx)

        metrics = _base_metrics(
            session_resolution,
            scan_id=scan_id,
            included_universe_size=len(snapshots),
            included_market_cap_bucket_counts=included_market_cap_bucket_counts,
            earnings_from=earnings_from,
            earnings_to=evidence_day,
            price_from=price_from,
        )
        computed_foster = sum(1 for item in foster_by_ticker.values() if item.computed)
        insufficient_foster = sum(1 for item in foster_by_ticker.values() if not item.computed)
        computed_friction = sum(1 for item in friction_by_ticker.values() if item.computed)
        metrics.update({
            "trailing_earnings_event_count": len(trailing_fetch.events),
            "announcing_universe_event_count": len(trailing_events),
            "forward_earnings_event_count": len(forward_fetch.events),
            "next_earnings_distance_count": len(next_earnings_by_ticker),
            "eps_history_fetch_error_count": len(eps_fetch_errors),
            "price_fetch_error_count": len(price_fetch_errors),
            "foster_computed_count": computed_foster,
            "foster_insufficient_history_count": insufficient_foster,
            "foster_eligible_fraction": (
                computed_foster / len(trailing_events) if trailing_events else None
            ),
            "friction_computed_count": computed_friction,
            "friction_population_count": len(friction_by_ticker),
            "market_factor_symbol": MARKET_FACTOR_SYMBOL,
            "market_factor_lineage_id": market_lineage.data_lineage_id,
            "earnings_calendar_lineage_ids": trailing_lineage_ids,
            "forward_earnings_calendar_lineage_ids": [
                lineage.data_lineage_id for lineage in forward_lineages
            ],
            "earnings_calendar_coverage": trailing_fetch.coverage,
            "forward_earnings_calendar_coverage": forward_fetch.coverage,
            "assembly": _assembly_metrics(assembly),
            "orchestration": orchestration_result.metrics,
        })

        errors = list(orchestration_result.errors or [])
        errors.extend({"stage": "eps_history", **err} for err in eps_fetch_errors)
        errors.extend({"stage": "price_history", **err} for err in price_fetch_errors)

        status = "partial_failed" if orchestration_result.status == "partial_failed" else "finished"
        return JobResult(
            status=status,
            metrics=metrics,
            input_hashes={
                "scan_id": scan_id,
                "decision_date": decision_date,
                "evidence_session_date": evidence_session_date,
                "market_factor_symbol": MARKET_FACTOR_SYMBOL,
            },
            output_hashes=orchestration_result.output_hashes,
            errors=errors,
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


def _select_trailing_announcements(
    events: List[FmpEarningsCalendarEvent],
    *,
    snapshot_by_ticker: Dict[str, UniverseSnapshot],
    evidence_day: date,
    window_start: date,
) -> Dict[str, FmpEarningsCalendarEvent]:
    selected: Dict[str, Tuple[date, FmpEarningsCalendarEvent]] = {}
    for event in events:
        ticker = event.symbol.upper()
        if ticker not in snapshot_by_ticker:
            continue
        effective = effective_announcement_session(event)
        if effective is None or effective < window_start or effective > evidence_day:
            continue
        existing = selected.get(ticker)
        if existing is None or effective > existing[0]:
            selected[ticker] = (effective, event)
    return {ticker: event for ticker, (_, event) in selected.items()}


def _next_earnings_distance(
    events: List[FmpEarningsCalendarEvent],
    *,
    evidence_day: date,
    snapshot_by_ticker: Dict[str, UniverseSnapshot],
) -> Dict[str, int]:
    distances: Dict[str, int] = {}
    for event in events:
        ticker = event.symbol.upper()
        if ticker not in snapshot_by_ticker:
            continue
        effective = effective_announcement_session(event)
        if effective is None or effective <= evidence_day:
            continue
        distance = trading_session_distance(evidence_day, effective)
        if distance is None:
            continue
        if ticker not in distances or distance < distances[ticker]:
            distances[ticker] = distance
    return distances


def _fetch_earnings_calendar_window(
    adapter: Any,
    *,
    from_date: date,
    to_date: date,
    asof: datetime,
) -> CalendarWindowFetch:
    pages: List[CalendarPageFetch] = []
    events: List[FmpEarningsCalendarEvent] = []
    errors: List[Dict[str, Any]] = []
    returned_dates: List[date] = []
    max_page_rows = 0
    capped_page_dates: List[str] = []
    day_count = 0

    cursor = from_date
    while cursor <= to_date:
        day_count += 1
        resp = adapter.get_earnings_calendar(
            from_date=cursor,
            to_date=cursor,
            asof=asof,
        )
        pages.append(CalendarPageFetch(request_date=cursor, response=resp))
        if not resp.ok:
            errors.append({
                "request_date": cursor.isoformat(),
                **_provider_error(resp.error),
            })
            cursor += timedelta(days=1)
            continue
        rows = list(resp.data or [])
        max_page_rows = max(max_page_rows, len(rows))
        if len(rows) >= 4000:
            capped_page_dates.append(cursor.isoformat())
        for event in rows:
            event_day = _event_calendar_date(event)
            if event_day is None:
                errors.append({
                    "request_date": cursor.isoformat(),
                    "message": "earnings calendar row missing event date",
                })
                continue
            if from_date <= event_day <= to_date:
                returned_dates.append(event_day)
                events.append(event)
        cursor += timedelta(days=1)

    coverage = {
        "fetch_strategy": "per_day",
        "requested_from": from_date.isoformat(),
        "requested_to": to_date.isoformat(),
        "requested_calendar_days": day_count,
        "successful_calendar_days": day_count - len({
            error.get("request_date") for error in errors if error.get("request_date")
        }),
        "error_count": len(errors),
        "returned_event_count": len(events),
        "returned_min_date": min(returned_dates).isoformat() if returned_dates else None,
        "returned_max_date": max(returned_dates).isoformat() if returned_dates else None,
        "max_page_row_count": max_page_rows,
        "capped_page_dates": capped_page_dates,
    }
    coverage["covers_requested_window"] = (
        not errors
        and not capped_page_dates
        and len(pages) == day_count
    )
    if capped_page_dates:
        errors.append({
            "message": "earnings calendar page hit provider row cap",
            "capped_page_dates": capped_page_dates,
        })
    return CalendarWindowFetch(
        events=events,
        pages=pages,
        coverage=coverage,
        errors=errors,
    )


def _record_calendar_page_lineages(
    session: Session,
    fetch: CalendarWindowFetch,
    *,
    job_run_id: str,
    coverage_kind: str,
) -> List[Any]:
    lineages: List[Any] = []
    for page in fetch.pages:
        resp = page.response
        lineages.append(_record_response_lineage(
            session,
            resp,
            job_run_id=job_run_id,
            raw_payload={
                "endpoint": EARNINGS_CALENDAR_ENDPOINT,
                "coverage_kind": coverage_kind,
                "from": page.request_date.isoformat(),
                "to": page.request_date.isoformat(),
                "coverage": fetch.coverage,
                "rows": _jsonable(resp.data),
            },
        ))
    return lineages


def _event_calendar_date(event: FmpEarningsCalendarEvent) -> Optional[date]:
    try:
        return date.fromisoformat(str(event.date)[:10])
    except (TypeError, ValueError):
        return None


def _session_window_start(evidence_day: date, sessions: int) -> date:
    cursor = evidence_day
    for _ in range(sessions):
        cursor = previous_us_equity_session(cursor)
    return cursor


def _persist_m1_earnings_event(
    session: Session,
    *,
    scan_id: str,
    snapshot: Optional[UniverseSnapshot],
    job_run_id: str,
    computation: FosterComputation,
    lineage_ids: List[str],
) -> None:
    event_id = computation.event_id or f"{computation.ticker}:unresolved"
    existing = (
        session.query(M1EarningsEvent)
        .filter(
            M1EarningsEvent.scan_id == scan_id,
            M1EarningsEvent.ticker == computation.ticker,
            M1EarningsEvent.earnings_event_id == event_id,
        )
        .first()
    )
    if existing is not None:
        session.delete(existing)
        session.flush()
    row = M1EarningsEvent(
        scan_id=scan_id,
        universe_snapshot_id=getattr(snapshot, "universe_snapshot_id", None),
        job_run_id=job_run_id,
        ticker=computation.ticker,
        earnings_event_id=event_id,
        announcement_date=computation.announcement_date,
        effective_announcement_session=computation.effective_announcement_session,
        announcement_time=computation.announcement_time,
        fiscal_period_end=computation.fiscal_period_end,
        fiscal_year=computation.fiscal_year,
        fiscal_quarter=computation.fiscal_quarter,
        actual_eps=computation.actual_eps,
        estimated_eps=computation.estimated_eps,
        expected_eps=computation.expected_eps,
        sigma_delta_eps=computation.sigma_delta_eps,
        sue_foster=computation.sue_foster,
        rho1=computation.rho1,
        sue_sign_current=computation.sue_sign_current,
        sue_sign_prior=computation.sue_sign_prior,
        sue_streak_length=computation.sue_streak_length,
        foster_history_quarters_used=computation.foster_history_quarters_used,
        split_adjustment_continuity_check=computation.split_adjustment_continuity_check,
        restatement_exposure=computation.restatement_exposure,
        status=computation.status,
        diagnostic_json=json.dumps(computation.diagnostics),
        sue_series_json=json.dumps(computation.sue_series, default=str),
        data_lineage_ids=json.dumps(lineage_ids),
    )
    session.add(row)
    session.flush()


def _persist_m1_friction_snapshot(
    session: Session,
    *,
    scan_id: str,
    snapshot: Optional[UniverseSnapshot],
    job_run_id: str,
    metric: PriceDelayComputation,
    lineage_ids: List[str],
) -> None:
    existing = (
        session.query(M1FrictionSnapshot)
        .filter(
            M1FrictionSnapshot.scan_id == scan_id,
            M1FrictionSnapshot.ticker == metric.ticker,
        )
        .first()
    )
    if existing is not None:
        session.delete(existing)
        session.flush()
    row = M1FrictionSnapshot(
        scan_id=scan_id,
        universe_snapshot_id=getattr(snapshot, "universe_snapshot_id", None),
        job_run_id=job_run_id,
        ticker=metric.ticker,
        market_factor_symbol=metric.market_factor_symbol,
        d1=metric.d1,
        d1_decile=metric.d1_decile,
        sigma_epsilon=metric.sigma_epsilon,
        sigma_epsilon_percentile=metric.sigma_epsilon_percentile,
        weekly_return_count=metric.weekly_return_count,
        status=metric.status,
        diagnostic_json=json.dumps(metric.diagnostics),
        data_lineage_ids=json.dumps(lineage_ids),
    )
    session.add(row)
    session.flush()


def _record_response_lineage(
    session: Session,
    resp: Any,
    *,
    job_run_id: str,
    raw_payload: Dict[str, Any],
) -> Any:
    return record_data_lineage(
        session,
        provider=resp.lineage.provider,
        endpoint=resp.lineage.endpoint,
        asof_timestamp=resp.lineage.asof_timestamp,
        raw_payload=raw_payload,
        raw_payload_hash=resp.lineage.raw_payload_hash,
        request_timestamp=resp.lineage.request_timestamp,
        freshness_seconds=resp.lineage.freshness_seconds,
        source_authority=resp.lineage.source_authority,
        data_quality_flags=resp.lineage.data_quality_flags,
        job_run_id=job_run_id,
    )


def _base_metrics(
    session_resolution: Any,
    *,
    scan_id: Optional[str],
    included_universe_size: int,
    included_market_cap_bucket_counts: Dict[str, int],
    earnings_from: date,
    earnings_to: date,
    price_from: date,
) -> Dict[str, Any]:
    return {
        "decision_date": session_resolution.decision_date,
        "evidence_session_date": session_resolution.evidence_session_date,
        "next_execution_session": session_resolution.next_execution_session,
        "is_premarket_decision_window": session_resolution.is_premarket_decision_window,
        "session_resolution": asdict(session_resolution),
        "canonical_scan_id": scan_id,
        "included_universe_size": included_universe_size,
        "included_market_cap_bucket_counts": included_market_cap_bucket_counts,
        "earnings_from_date": earnings_from.isoformat(),
        "earnings_to_date": earnings_to.isoformat(),
        "price_from_date": price_from.isoformat(),
        "fetch_endpoints": {
            "earnings_calendar": EARNINGS_CALENDAR_ENDPOINT,
            "eps_history": EARNINGS_HISTORY_ENDPOINT,
            "price": HISTORICAL_PRICE_FULL_ENDPOINT,
        },
    }


def _assembly_metrics(assembly: Any) -> Dict[str, Any]:
    return {
        "pattern_id": assembly.pattern_id,
        "assembled_count": assembly.assembled_count,
        "rejected_count": assembly.rejected_count,
        "insufficient_count": assembly.insufficient_count,
        "diagnostic_count": len(assembly.diagnostics),
        "diagnostics": [asdict(d) for d in assembly.diagnostics[:50]],
        "rejected_field_count": len(assembly.rejected_fields),
    }


def _provider_error(error: Any) -> Dict[str, Any]:
    if error is None:
        return {}
    return {
        "error_type": getattr(error, "error_type", None),
        "status_code": getattr(error, "status_code", None),
        "message": getattr(error, "message", None),
        "retryable": getattr(error, "retryable", None),
    }


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _ensure_aware(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
