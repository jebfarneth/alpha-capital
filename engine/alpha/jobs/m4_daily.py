"""Production M4 daily feature assembly wiring.

This job is the production path for the completed daily M4 feature assembly
slice. It resolves market-session semantics first, caps provider fetches at
the evidence session, assembles M4 inputs, and persists through detector
orchestration.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from alpha.assembly.m4_daily import DailyBar, assemble_m4_daily
from alpha.assembly.signal_context import (
    DEFAULT_M4_CONTEXT_BREAKOUT_BUFFER,
    enrich_m4_signal_context,
    reuse_persisted_m4_signal_context,
    select_m4_signal_context_inputs,
    validate_m4_context_breakout_buffer,
)
from alpha.data.fmp import FmpBar, HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.db.models import CanonicalUniverseScan, UniverseScan, UniverseSnapshot
from alpha.evidence.writer import record_data_lineage
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.detector_orchestration import DetectorOrchestrationJob
from alpha.jobs.universe_builder import market_cap_bucket_counts
from alpha.market_calendar import (
    resolve_us_equity_session,
    us_equity_session_close_timestamp,
)
from alpha.patterns.m4 import M4Detector
from alpha.security_identity import (
    load_security_identity_by_ticker,
    security_identity_lineage_ids,
    security_identity_payload,
)


class M4DailyAssemblyJob(BaseJob):
    """Run daily M4 from canonical universe through persisted orchestration."""

    job_name = "m4_daily_feature_assembly"
    job_type = "feature_assembly"

    def __init__(
        self,
        session: Session,
        *,
        adapter: Any,
        polygon_adapter: Any = None,
        benzinga_adapter: Any = None,
        enable_signal_context: bool = True,
        signal_context_breakout_buffer: float = DEFAULT_M4_CONTEXT_BREAKOUT_BUFFER,
        run_timestamp: Optional[datetime] = None,
        lookback_calendar_days: int = 430,
    ):
        self._session = session
        self._adapter = adapter
        self._polygon_adapter = polygon_adapter
        self._benzinga_adapter = benzinga_adapter
        self._enable_signal_context = enable_signal_context
        self._signal_context_breakout_buffer = signal_context_breakout_buffer
        self._run_timestamp = run_timestamp
        self._lookback_calendar_days = lookback_calendar_days

    def run(self, ctx: JobContext) -> JobResult:
        """Run the production M4 daily assembly and orchestration path."""

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
        signal_context_breakout_buffer = self._signal_context_breakout_buffer
        if self._enable_signal_context:
            try:
                signal_context_breakout_buffer = validate_m4_context_breakout_buffer(
                    signal_context_breakout_buffer
                )
            except ValueError as exc:
                return JobResult(
                    status="failed",
                    errors=[{"stage": "params", "message": str(exc)}],
                )

        session_resolution = resolve_us_equity_session(run_timestamp)
        decision_date = session_resolution.decision_date
        evidence_session_date = session_resolution.evidence_session_date
        evidence_day = date.fromisoformat(evidence_session_date)
        cutoff_timestamp = us_equity_session_close_timestamp(evidence_day)
        from_date = evidence_day - timedelta(days=self._lookback_calendar_days)

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

        daily_bars: Dict[str, List[DailyBar]] = {}
        fetch_errors: List[Dict[str, Any]] = []
        fetched_symbol_count = 0
        fetched_bar_count = 0

        for snapshot in snapshots:
            ticker = snapshot.ticker
            resp = self._adapter.get_historical_price(
                ticker,
                from_date=from_date,
                to_date=evidence_day,
                asof=cutoff_timestamp,
                adjusted=False,
                require_split_adjusted_close=True,
            )
            lineage = record_data_lineage(
                self._session,
                provider=resp.lineage.provider,
                endpoint=resp.lineage.endpoint,
                asof_timestamp=resp.lineage.asof_timestamp,
                raw_payload=_lineage_payload(
                    resp.data,
                    ticker=ticker,
                    from_date=from_date,
                    to_date=evidence_day,
                ),
                raw_payload_hash=resp.lineage.raw_payload_hash,
                request_timestamp=resp.lineage.request_timestamp,
                freshness_seconds=resp.lineage.freshness_seconds,
                source_authority=resp.lineage.source_authority,
                data_quality_flags=resp.lineage.data_quality_flags,
                job_run_id=ctx.job_run_id,
            )

            if not resp.ok:
                err = resp.error
                fetch_errors.append({
                    "ticker": ticker,
                    "error_type": getattr(err, "error_type", None),
                    "status_code": getattr(err, "status_code", None),
                    "message": getattr(err, "message", None),
                    "retryable": getattr(err, "retryable", None),
                })
                continue

            bars = list(resp.data or [])
            fetched_symbol_count += 1
            fetched_bar_count += len(bars)
            daily_bars[ticker] = [
                _to_daily_bar(
                    bar,
                    source_timestamp=resp.lineage.asof_timestamp,
                    source_provider=resp.lineage.provider,
                    lineage_id=lineage.data_lineage_id,
                    lineage_hash=resp.lineage.raw_payload_hash,
                )
                for bar in bars
            ]

        assembly = assemble_m4_daily(
            snapshots=snapshots,
            daily_bars=daily_bars,
            cutoff_timestamp=cutoff_timestamp,
            universe_cutoff_timestamp=scan_asof_timestamp,
            decision_date=decision_date,
            evidence_session_date=evidence_session_date,
            next_execution_session=session_resolution.next_execution_session,
            source_provider="FMP",
        )

        identity_by_ticker = load_security_identity_by_ticker(self._session, scan_id)
        identity_injected = 0
        for inp in assembly.inputs:
            identity = identity_by_ticker.get(inp.ticker.upper())
            inp.market_data["security_identity"] = security_identity_payload(identity)
            if identity is not None and identity.identity_status == "present":
                identity_injected += 1
            for lineage_id in security_identity_lineage_ids(identity):
                if lineage_id not in inp.lineage_ids:
                    inp.lineage_ids.append(lineage_id)

        signal_context_metrics: Dict[str, Any] = {}
        if self._enable_signal_context:
            context_inputs, prefilter_metrics = select_m4_signal_context_inputs(
                assembly.inputs,
                breakout_buffer=signal_context_breakout_buffer,
            )
            persisted_context_metrics = reuse_persisted_m4_signal_context(
                context_inputs,
                session=self._session,
                cutoff_timestamp=cutoff_timestamp,
                decision_date=decision_date,
            )
            signal_context_metrics = enrich_m4_signal_context(
                context_inputs,
                session=self._session,
                polygon_adapter=self._polygon_adapter,
                benzinga_adapter=self._benzinga_adapter,
                cutoff_timestamp=cutoff_timestamp,
                decision_date=decision_date,
                evidence_session_date=evidence_session_date,
                job_run_id=ctx.job_run_id,
            )
            signal_context_metrics.update(persisted_context_metrics)
            signal_context_metrics.update(prefilter_metrics)

        if not assembly.inputs:
            return JobResult(
                status="failed",
                metrics={
                    "decision_date": decision_date,
                    "evidence_session_date": evidence_session_date,
                    "session_resolution": asdict(session_resolution),
                    "canonical_scan_id": scan_id,
                    "included_universe_size": len(snapshots),
                    "included_market_cap_bucket_counts": included_market_cap_bucket_counts,
                    "fetched_symbol_count": fetched_symbol_count,
                    "fetched_bar_count": fetched_bar_count,
                    "fetch_error_count": len(fetch_errors),
                    "assembly": _assembly_metrics(assembly),
                },
                errors=fetch_errors or [{
                    "stage": "assembly",
                    "message": "M4 assembly produced zero inputs",
                }],
            )

        orchestration = DetectorOrchestrationJob(
            self._session,
            detectors=[M4Detector()],
            trading_date=decision_date,
            assembled_inputs={"M4": assembly.inputs},
        )
        orchestration_result = orchestration.run(ctx)

        metrics = {
            "decision_date": decision_date,
            "evidence_session_date": evidence_session_date,
            "next_execution_session": session_resolution.next_execution_session,
            "is_premarket_decision_window": (
                session_resolution.is_premarket_decision_window
            ),
            "session_resolution": asdict(session_resolution),
            "canonical_scan_id": scan_id,
            "included_universe_size": len(snapshots),
            "included_market_cap_bucket_counts": included_market_cap_bucket_counts,
            "fetch_endpoint": HISTORICAL_PRICE_FULL_ENDPOINT,
            "fetch_from_date": from_date.isoformat(),
            "fetch_to_date": evidence_session_date,
            "fetch_asof_timestamp": cutoff_timestamp.isoformat(),
            "fetched_symbol_count": fetched_symbol_count,
            "fetched_bar_count": fetched_bar_count,
            "fetch_error_count": len(fetch_errors),
            "fetch_errors": fetch_errors[:50],
            "assembly": _assembly_metrics(assembly),
            "security_identity_present_count": identity_injected,
            "security_identity_snapshot_count": len(identity_by_ticker),
            "signal_context": signal_context_metrics,
            "orchestration": orchestration_result.metrics,
        }

        errors = list(orchestration_result.errors or [])
        if fetch_errors:
            errors.extend({"stage": "fetch", **err} for err in fetch_errors)

        return JobResult(
            status=orchestration_result.status,
            metrics=metrics,
            input_hashes={
                "scan_id": scan_id,
                "decision_date": decision_date,
                "evidence_session_date": evidence_session_date,
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


def _to_daily_bar(
    bar: FmpBar,
    *,
    source_timestamp: datetime,
    source_provider: str,
    lineage_id: str,
    lineage_hash: str,
) -> DailyBar:
    return DailyBar(
        date=bar.date,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        split_adjusted_close=bar.split_adjusted_close,
        adj_close=bar.adj_close,
        source_timestamp=source_timestamp,
        source_provider=source_provider,
        lineage_id=lineage_id,
        lineage_hash=lineage_hash,
    )


def _lineage_payload(
    bars: Any,
    *,
    ticker: str,
    from_date: date,
    to_date: date,
) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "request": {
            "symbol": ticker,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "endpoint": HISTORICAL_PRICE_FULL_ENDPOINT,
        },
        "bars": [
            {
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "split_adjusted_close": bar.split_adjusted_close,
                "adj_close": bar.adj_close,
            }
            for bar in (bars or [])
        ],
    }


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _assembly_metrics(assembly) -> Dict[str, Any]:
    diagnostic_counts: Dict[str, int] = {}
    for diag in assembly.diagnostics:
        diagnostic_counts[diag.diagnostic_type] = (
            diagnostic_counts.get(diag.diagnostic_type, 0) + 1
        )
    return {
        "pattern_id": assembly.pattern_id,
        "assembled_count": assembly.assembled_count,
        "rejected_count": assembly.rejected_count,
        "insufficient_count": assembly.insufficient_count,
        "diagnostic_counts": diagnostic_counts,
        "diagnostics": [
            {
                "ticker": diag.ticker,
                "pattern_id": diag.pattern_id,
                "diagnostic_type": diag.diagnostic_type,
                "detail": diag.detail,
            }
            for diag in assembly.diagnostics[:100]
        ],
    }
