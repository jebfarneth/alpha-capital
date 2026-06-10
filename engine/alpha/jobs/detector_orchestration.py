"""
Detector orchestration job.

Runs implemented detectors over the canonical universe and persists every
pattern-intrinsic firing with reproducible identity. Deduplicates by
(pattern_id, ticker, signal_identity_hash). Refuses signals without
required identity/lineage fields. Isolates per-detector failures.

Per MeasurementSpine.md section 2.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from alpha.data.contracts import stable_hash
from alpha.db.models import (
    CanonicalUniverseScan,
    DataLineage,
    SignalRegistry,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.market_calendar import us_equity_session_close_timestamp
from alpha.patterns.contracts import (
    BasePatternDetector,
    PatternDetectionResult,
    PatternInput,
)
from alpha.patterns.evidence_bridge import persist_detection_result


MARKET_TIMEZONE = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Detector enumeration
# ---------------------------------------------------------------------------

def enumerate_callable_detectors() -> List[BasePatternDetector]:
    """Discover implemented detector instances from the patterns package.

    Only returns detectors that are instantiable and implement detect().
    Does NOT rely on a doc-only roster.
    """
    import alpha.patterns as patterns_pkg

    skipped_modules = {
        "__init__",
        "activation",
        "contracts",
        "evidence_bridge",
        "fixture_detector",
        "guards",
    }
    detector_classes = []
    for module_info in pkgutil.iter_modules(patterns_pkg.__path__):
        if module_info.name in skipped_modules or module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{patterns_pkg.__name__}.{module_info.name}")
        for _, attr in inspect.getmembers(module, inspect.isclass):
            if (
                attr.__module__ == module.__name__
                and issubclass(attr, BasePatternDetector)
                and attr is not BasePatternDetector
            ):
                detector_classes.append(attr)

    instances: List[BasePatternDetector] = []
    instantiation_errors: List[str] = []
    for cls in sorted(detector_classes, key=lambda c: str(getattr(c, "pattern_id", c.__name__))):
        try:
            instances.append(cls())
        except Exception as exc:
            instantiation_errors.append(f"{cls.__module__}.{cls.__name__}: {exc}")

    if instantiation_errors:
        raise RuntimeError(
            "detector enumeration found non-callable detector classes: "
            + "; ".join(instantiation_errors)
        )

    return instances


# ---------------------------------------------------------------------------
# Signal identity computation
# ---------------------------------------------------------------------------

def compute_signal_identity_hash(
    *,
    detector_id: str,
    detector_version: str,
    ticker: str,
    trading_date: str,
    direction: str,
    detector_signal_identity_hash: Optional[str] = None,
    detector_signal_identity_components: Optional[Dict[str, Any]] = None,
    route_class: Optional[str] = None,
    signal_family: Optional[str] = None,
    event_id: Optional[str] = None,
    signal_horizon: Optional[str] = None,
    signal_event_sequence: Optional[int] = None,
) -> str:
    """Deterministic content-based identity for a signal.

    Includes only stable economic/setup identity. Excludes job_run_id,
    UUIDs, wall-clock now(), insert timestamps, scheduler metadata.
    """
    components = {
        "detector_id": detector_id,
        "detector_version": detector_version,
        "ticker": ticker,
        "trading_date": trading_date,
        "direction": direction,
    }
    if detector_signal_identity_hash is not None:
        components["detector_signal_identity_hash"] = detector_signal_identity_hash
    if detector_signal_identity_components is not None:
        components["detector_signal_identity_components"] = detector_signal_identity_components
    if route_class is not None:
        components["route_class"] = route_class
    if signal_family is not None:
        components["signal_family"] = signal_family
    if event_id is not None:
        components["event_id"] = event_id
    if signal_horizon is not None:
        components["signal_horizon"] = signal_horizon
    if signal_event_sequence is not None:
        components["signal_event_sequence"] = signal_event_sequence
    return stable_hash(components)


# ---------------------------------------------------------------------------
# Lookahead guard
# ---------------------------------------------------------------------------

def check_lookahead_guard(
    inp: PatternInput,
    trading_date: str,
    *,
    max_asof_timestamp: Optional[datetime] = None,
    max_asof_label: str = "allowed asof",
) -> Tuple[bool, Optional[str]]:
    """Real lookahead guard — not hardcoded True.

    Verifies that input data timestamps are not after the trading date cutoff.
    Returns (passed, failure_reason).
    """
    if not inp.asof_timestamp:
        return False, "missing_asof_timestamp"

    requested_trading_date = date.fromisoformat(trading_date)
    input_market_date = _market_date(inp.asof_timestamp)
    if input_market_date > requested_trading_date:
        return (
            False,
            "asof_timestamp "
            f"{inp.asof_timestamp.isoformat()} has market date "
            f"{input_market_date.isoformat()} after trading_date {trading_date}",
        )

    if max_asof_timestamp is not None:
        if _comparable_datetime(inp.asof_timestamp) > _comparable_datetime(
            max_asof_timestamp
        ):
            return (
                False,
                "asof_timestamp "
                f"{inp.asof_timestamp.isoformat()} is after {max_asof_label} "
                f"{max_asof_timestamp.isoformat()}",
            )
    # Check lineage hashes are present and nonblank — proves data provenance exists.
    if not any(str(value or "").strip() for value in inp.lineage_hashes):
        return False, "missing_lineage_hashes"

    return True, None


def _comparable_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _utc_aware_datetime(value: datetime) -> datetime:
    """Normalize SQLite naive UTC round trips back to aware UTC datetimes."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _market_date(value: datetime) -> date:
    return _utc_aware_datetime(value).astimezone(MARKET_TIMEZONE).date()


def _input_asof_ceiling(
    inp: PatternInput,
    scan_asof_timestamp: Optional[datetime],
    trading_date: str,
) -> Tuple[Optional[datetime], str]:
    explicit_ceiling = inp.market_data.get("asof_ceiling_timestamp")
    if isinstance(explicit_ceiling, str) and explicit_ceiling.strip():
        try:
            parsed = datetime.fromisoformat(explicit_ceiling.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"invalid_asof_ceiling_timestamp:{explicit_ceiling}"
            ) from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(
                f"invalid_asof_ceiling_timestamp_naive:{explicit_ceiling}"
            )
        if _market_date(parsed) > date.fromisoformat(trading_date):
            raise ValueError(
                f"future_asof_ceiling_timestamp:{explicit_ceiling}>{trading_date}"
            )
        return parsed.astimezone(timezone.utc), "input asof ceiling"
    evidence_session_date = inp.market_data.get("evidence_session_date")
    if isinstance(evidence_session_date, str) and evidence_session_date:
        evidence_day = date.fromisoformat(evidence_session_date)
        if evidence_day > date.fromisoformat(trading_date):
            raise ValueError(
                f"future_evidence_session_date:{evidence_session_date}>{trading_date}"
            )
        return us_equity_session_close_timestamp(evidence_day), "evidence session close"
    return scan_asof_timestamp, "canonical scan asof"


def _result_guard_passed(
    result: PatternDetectionResult,
    *,
    allow_point_in_time_failure: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Require detector-emitted guard flags to agree with the orchestration guard."""
    if result.features is None:
        return False, "missing_features"
    if (
        result.features.point_in_time_passed is not True
        and not allow_point_in_time_failure
    ):
        return False, "detector_point_in_time_guard_failed"
    if result.features.lookahead_guard_passed is not True:
        return False, "detector_lookahead_guard_failed"
    return True, None


# ---------------------------------------------------------------------------
# Per-detector diagnostics
# ---------------------------------------------------------------------------

@dataclass
class DetectorDiagnostics:
    """Per-detector execution counters persisted in orchestration metrics."""

    detector_id: str
    detector_version: str
    callable_status: str = "callable"
    evaluated_count: int = 0
    fired_count: int = 0
    feature_snapshot_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    lookahead_failure_count: int = 0
    duplicate_suppressed_count: int = 0
    identity_refused_count: int = 0
    detector_status: str = "finished"
    errors: List[Dict[str, Any]] = field(default_factory=list)
    input_lineage_hashes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable diagnostic payload."""

        return {
            "detector_id": self.detector_id,
            "detector_version": self.detector_version,
            "callable_status": self.callable_status,
            "evaluated_count": self.evaluated_count,
            "fired_count": self.fired_count,
            "feature_snapshot_count": self.feature_snapshot_count,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
            "lookahead_failure_count": self.lookahead_failure_count,
            "duplicate_suppressed_count": self.duplicate_suppressed_count,
            "identity_refused_count": self.identity_refused_count,
            "detector_status": self.detector_status,
            "errors": self.errors,
            "input_lineage_hashes": sorted(set(self.input_lineage_hashes)),
        }


# ---------------------------------------------------------------------------
# Orchestration job
# ---------------------------------------------------------------------------

class DetectorOrchestrationJob(BaseJob):
    """Run detectors over canonical universe, persist signals, dedup by identity.

    Requires:
    - canonical_universe_scans row for trading_date
    - All detectors produce signal_identity_hash
    - Lookahead guard passes for each input

    Refuses:
    - null signal_identity_hash
    - null scan_id / universe_snapshot_id / feature_snapshot_id
    - failed lookahead guard

    Partial failure:
    - One detector crash doesn't block others
    - Status is partial_failed when any detector errors
    - Reruns are idempotent (dedup by identity hash)
    """

    job_name = "detector_orchestration"
    job_type = "detector_scan"

    def __init__(
        self,
        session: Session,
        *,
        detectors: Optional[List[BasePatternDetector]] = None,
        trading_date: Optional[str] = None,
        inputs: Optional[List[PatternInput]] = None,
        assembled_inputs: Optional[Dict[str, List[PatternInput]]] = None,
        nested_persistence: bool = True,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        progress_every: int = 100,
    ):
        self._session = session
        self._detectors = detectors
        self._trading_date = trading_date
        self._inputs = inputs
        self._assembled_inputs = assembled_inputs
        self._nested_persistence = nested_persistence
        self._progress_callback = progress_callback
        self._progress_every = max(int(progress_every), 1)

    def run(self, ctx: JobContext) -> JobResult:
        """Run callable detectors over canonical or explicitly assembled inputs."""

        trading_date = self._trading_date or ctx.params.get("trading_date")
        if not trading_date:
            return JobResult(
                status="failed",
                errors=[{"stage": "params", "message": "trading_date is required"}],
            )

        # 1. Load canonical universe
        scan_id, scan_asof_timestamp, snapshots, error = self._load_canonical_universe(trading_date)
        if error:
            return JobResult(
                status="failed",
                metrics={
                    "trading_date": trading_date,
                    "total_signals_persisted": 0,
                },
                errors=[{"stage": "canonical_universe", "message": error}],
            )

        # 2. Enumerate detectors
        try:
            detectors = self._detectors or enumerate_callable_detectors()
        except Exception as exc:
            return JobResult(
                status="failed",
                errors=[{"stage": "detector_enumeration", "message": str(exc)}],
            )
        if not detectors:
            return JobResult(
                status="failed",
                errors=[{"stage": "detector_enumeration", "message": "no callable detectors found"}],
            )

        # 3. Build inputs from universe snapshots if not injected
        valid_snapshot_ids = {
            snap.universe_snapshot_id
            for snap in snapshots
            if snap.universe_snapshot_id is not None
        }
        assembled = self._assembled_inputs  # dict[pattern_id, list[PatternInput]] or None
        assembly_mode = assembled is not None
        flat_inputs = None
        if not assembly_mode:
            flat_inputs = (
                self._inputs
                if self._inputs is not None
                else self._build_inputs_from_snapshots(snapshots, trading_date)
            )

        # 4. Run each detector with isolation
        all_diagnostics: List[DetectorDiagnostics] = []
        assembly_diagnostics: List[Dict[str, Any]] = []
        total_signals = 0
        any_detector_failed = False

        for detector in detectors:
            if assembly_mode:
                if detector.pattern_id not in assembled:
                    assembly_diagnostics.append({
                        "detector_id": detector.pattern_id,
                        "diagnostic": "assembled_inputs_missing",
                    })
                    all_diagnostics.append(self._assembly_skip_diagnostic(
                        detector, callable_status="assembly_missing_inputs"
                    ))
                    continue

                detector_inputs = assembled[detector.pattern_id]
                if not detector_inputs:
                    assembly_diagnostics.append({
                        "detector_id": detector.pattern_id,
                        "diagnostic": "assembled_inputs_empty",
                    })
                    all_diagnostics.append(self._assembly_skip_diagnostic(
                        detector, callable_status="assembly_empty_inputs"
                    ))
                    continue
            else:
                detector_inputs = flat_inputs or []

            diag = self._run_detector(
                detector=detector,
                inputs=detector_inputs,
                trading_date=trading_date,
                scan_id=scan_id,
                scan_asof_timestamp=scan_asof_timestamp,
                valid_universe_snapshot_ids=valid_snapshot_ids,
                job_run_id=ctx.job_run_id,
                code_commit_sha=ctx.app_commit_sha,
            )
            all_diagnostics.append(diag)
            total_signals += diag.fired_count
            if diag.detector_status in {"failed", "partial_failed"}:
                any_detector_failed = True

        self._session.flush()

        # 5. Determine overall status
        if any_detector_failed:
            status = "partial_failed"
        else:
            status = "finished"

        metrics = {
            "trading_date": trading_date,
            "scan_id": scan_id,
            "universe_size": len(snapshots),
            "detector_count": len(detectors),
            "total_signals_persisted": total_signals,
            "detector_diagnostics": [d.to_dict() for d in all_diagnostics],
            "any_detector_failed": any_detector_failed,
            "assembly_diagnostics": assembly_diagnostics,
        }

        errors = []
        for diag in all_diagnostics:
            if diag.errors:
                errors.extend(diag.errors)

        return JobResult(
            status=status,
            metrics=metrics,
            input_hashes={"scan_id": scan_id},
            errors=errors if errors else [],
        )

    def _assembly_skip_diagnostic(
        self,
        detector: BasePatternDetector,
        *,
        callable_status: str,
    ) -> DetectorDiagnostics:
        return DetectorDiagnostics(
            detector_id=detector.pattern_id,
            detector_version=getattr(detector, "version", None) or "missing",
            callable_status=callable_status,
            detector_status="skipped",
        )

    def _load_canonical_universe(
        self, trading_date: str
    ) -> Tuple[Optional[str], Optional[datetime], List[UniverseSnapshot], Optional[str]]:
        """Load canonical scan and included snapshots."""
        canonical = (
            self._session.query(CanonicalUniverseScan)
            .filter(CanonicalUniverseScan.trading_date == trading_date)
            .first()
        )
        if canonical is None:
            return None, None, [], f"no canonical universe scan for trading_date={trading_date}"

        scan = self._session.get(UniverseScan, canonical.scan_id)
        if scan is None:
            return None, None, [], f"canonical scan_id {canonical.scan_id} not found"
        scan_asof_timestamp = _utc_aware_datetime(scan.asof_timestamp)
        requested_trading_date = date.fromisoformat(trading_date)
        scan_market_date = _market_date(scan_asof_timestamp)
        if scan_market_date != requested_trading_date:
            return (
                None,
                None,
                [],
                "canonical scan asof market date "
                f"{scan_market_date.isoformat()} does not match trading_date "
                f"{trading_date}; refusing lookahead-contaminated universe scan",
            )

        snapshots = (
            self._session.query(UniverseSnapshot)
            .filter(
                UniverseSnapshot.scan_id == canonical.scan_id,
                UniverseSnapshot.operating_universe_inclusion.is_(True),
            )
            .all()
        )
        return canonical.scan_id, scan_asof_timestamp, snapshots, None

    def _build_inputs_from_snapshots(
        self,
        snapshots: List[UniverseSnapshot],
        trading_date: str,
    ) -> List[PatternInput]:
        """Build PatternInput per included ticker from universe snapshots."""
        inputs = []
        lineage_hashes = {
            snap.source_lineage_hash
            for snap in snapshots
            if snap.source_lineage_hash
        }
        lineage_by_hash: Dict[str, List[str]] = {}
        if lineage_hashes:
            rows = (
                self._session.query(DataLineage.raw_payload_hash, DataLineage.data_lineage_id)
                .filter(DataLineage.raw_payload_hash.in_(lineage_hashes))
                .all()
            )
            for raw_payload_hash, data_lineage_id in rows:
                lineage_by_hash.setdefault(raw_payload_hash, []).append(data_lineage_id)
        for snap in snapshots:
            lineage_ids = lineage_by_hash.get(snap.source_lineage_hash or "", [])
            inp = PatternInput(
                ticker=snap.ticker,
                asof_timestamp=_utc_aware_datetime(snap.asof_timestamp),
                market_data={
                    "price": snap.price,
                    "market_cap": snap.market_cap,
                    "primary_exchange": snap.primary_exchange,
                    "security_type": snap.security_type,
                    "trading_date": trading_date,
                },
                lineage_ids=lineage_ids,
                lineage_hashes=[snap.source_lineage_hash] if snap.source_lineage_hash else [],
                universe_snapshot_id=snap.universe_snapshot_id,
            )
            inputs.append(inp)
        return inputs

    def _run_detector(
        self,
        *,
        detector: BasePatternDetector,
        inputs: List[PatternInput],
        trading_date: str,
        scan_id: str,
        scan_asof_timestamp: Optional[datetime],
        valid_universe_snapshot_ids: set[str],
        job_run_id: str,
        code_commit_sha: Optional[str],
    ) -> DetectorDiagnostics:
        """Run a single detector across all inputs with full isolation."""
        detector_version = getattr(detector, "version", None)
        if not detector_version:
            return DetectorDiagnostics(
                detector_id=detector.pattern_id,
                detector_version="missing",
                detector_status="failed",
                error_count=1,
                errors=[{
                    "detector_id": detector.pattern_id,
                    "error": "missing_explicit_detector_version",
                }],
            )
        diag = DetectorDiagnostics(
            detector_id=detector.pattern_id,
            detector_version=detector_version,
        )

        previous_autoflush = self._session.autoflush
        if not self._nested_persistence:
            self._session.autoflush = False
        for inp in inputs:
            diag.evaluated_count += 1
            if diag.evaluated_count == 1 or diag.evaluated_count % self._progress_every == 0:
                self._emit_progress(
                    "detector_progress",
                    {
                        "detector_id": detector.pattern_id,
                        "trading_date": trading_date,
                        "evaluated_count": diag.evaluated_count,
                        "input_count": len(inputs),
                        "feature_snapshot_count": diag.feature_snapshot_count,
                        "fired_count": diag.fired_count,
                        "skipped_count": diag.skipped_count,
                        "error_count": diag.error_count,
                        "nested_persistence": self._nested_persistence,
                    },
                )

            if inp.universe_snapshot_id not in valid_universe_snapshot_ids:
                diag.identity_refused_count += 1
                diag.skipped_count += 1
                diag.errors.append({
                    "detector_id": detector.pattern_id,
                    "ticker": inp.ticker,
                    "error": "universe_snapshot_not_in_canonical_scan",
                    "universe_snapshot_id": inp.universe_snapshot_id,
                    "scan_id": scan_id,
                })
                continue

            # Lookahead guard
            try:
                max_asof_timestamp, max_asof_label = _input_asof_ceiling(
                    inp,
                    scan_asof_timestamp,
                    trading_date,
                )
            except ValueError as exc:
                diag.lookahead_failure_count += 1
                diag.skipped_count += 1
                diag.errors.append({
                    "detector_id": detector.pattern_id,
                    "ticker": inp.ticker,
                    "error": f"invalid_evidence_session_date:{exc}",
                })
                continue
            pit_passed, pit_reason = check_lookahead_guard(
                inp,
                trading_date,
                max_asof_timestamp=max_asof_timestamp,
                max_asof_label=max_asof_label,
            )
            if not pit_passed:
                diag.lookahead_failure_count += 1
                diag.skipped_count += 1
                diag.errors.append({
                    "detector_id": detector.pattern_id,
                    "ticker": inp.ticker,
                    "error": f"lookahead_guard_failed:{pit_reason}",
                })
                continue

            # Run detector
            try:
                result = detector.detect(inp)
            except Exception as exc:
                diag.error_count += 1
                diag.errors.append({
                    "detector_id": detector.pattern_id,
                    "ticker": inp.ticker,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                continue

            for lineage_hash in inp.lineage_hashes:
                lineage_hash_str = str(lineage_hash or "").strip()
                if lineage_hash_str:
                    diag.input_lineage_hashes.append(lineage_hash_str)

            if result.features is None:
                if result.has_signal:
                    diag.identity_refused_count += 1
                    diag.errors.append({
                        "detector_id": detector.pattern_id,
                        "ticker": inp.ticker,
                        "error": "missing_features",
                    })
                else:
                    diag.skipped_count += 1
                continue

            result_guard_passed, result_guard_reason = _result_guard_passed(
                result,
                allow_point_in_time_failure=bool(
                    getattr(detector, "allow_shadow_point_in_time_failure", False)
                ),
            )
            if not result_guard_passed:
                diag.lookahead_failure_count += 1
                if result.has_signal:
                    diag.identity_refused_count += 1
                diag.skipped_count += 1
                diag.errors.append({
                    "detector_id": detector.pattern_id,
                    "ticker": inp.ticker,
                    "error": result_guard_reason,
                })
                continue

            if not result.has_signal:
                try:
                    persisted = self._persist_detection_result_with_lineage(
                        inp=inp,
                        result=result,
                        detector=detector,
                        job_run_id=job_run_id,
                        universe_snapshot_id=inp.universe_snapshot_id,
                        code_commit_sha=code_commit_sha,
                        trading_date=trading_date,
                        scan_id=scan_id,
                        detector_version=detector_version,
                        point_in_time_passed=result.features.point_in_time_passed,
                        lookahead_guard_passed=result.features.lookahead_guard_passed,
                    )
                except Exception as exc:
                    diag.error_count += 1
                    diag.errors.append({
                        "detector_id": detector.pattern_id,
                        "ticker": inp.ticker,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })
                    continue

                if persisted.feature_snapshot_id:
                    diag.feature_snapshot_count += 1
                diag.skipped_count += 1
                continue

            detector_identity_hash = result.features.features.get("signal_identity_hash")
            detector_identity_components = result.features.features.get(
                "signal_identity_components"
            )
            if not detector_identity_hash:
                diag.identity_refused_count += 1
                diag.errors.append({
                    "detector_id": detector.pattern_id,
                    "ticker": inp.ticker,
                    "error": "missing_detector_signal_identity_hash",
                })
                continue

            if len(result.signals) != 1:
                diag.identity_refused_count += 1
                diag.errors.append({
                    "detector_id": detector.pattern_id,
                    "ticker": inp.ticker,
                    "error": "orchestration_requires_exactly_one_signal_per_result",
                })
                continue

            signal = result.signals[0]

            # Compute signal identity from detector-native setup identity plus
            # canonical scan/date/version anchors. Do not replace the detector's
            # economic setup identity with a ticker/date-only fallback.
            identity_hash = compute_signal_identity_hash(
                detector_id=detector.pattern_id,
                detector_version=detector_version,
                ticker=inp.ticker,
                trading_date=trading_date,
                direction=signal.direction,
                detector_signal_identity_hash=detector_identity_hash,
                detector_signal_identity_components=detector_identity_components,
                route_class=signal.route_class or detector.route_class,
                signal_horizon=signal.signal_horizon,
                signal_event_sequence=1,
            )

            # Strict refusal: all required fields must be present
            if not identity_hash:
                diag.identity_refused_count += 1
                continue
            if not scan_id or not inp.universe_snapshot_id:
                diag.identity_refused_count += 1
                continue

            # Dedup check
            try:
                existing = self._existing_signal_id(
                    detector.pattern_id,
                    inp.ticker,
                    identity_hash,
                )
            except SQLAlchemyError as exc:
                diag.error_count += 1
                diag.errors.append({
                    "detector_id": detector.pattern_id,
                    "ticker": inp.ticker,
                    "stage": "duplicate_check",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                continue
            if existing:
                diag.duplicate_suppressed_count += 1
                continue

            # Inject identity into features for persistence
            result.features.features["detector_signal_identity_hash"] = (
                detector_identity_hash
            )
            result.features.features["detector_signal_identity_components"] = (
                detector_identity_components
            )
            result.features.features["signal_identity_hash"] = identity_hash
            result.features.features["signal_identity_components"] = {
                "detector_id": detector.pattern_id,
                "detector_version": detector_version,
                "ticker": inp.ticker,
                "trading_date": trading_date,
                "direction": signal.direction,
                "detector_signal_identity_hash": detector_identity_hash,
                "detector_signal_identity_components": detector_identity_components,
                "route_class": signal.route_class or detector.route_class,
                "signal_horizon": signal.signal_horizon,
                "signal_event_sequence": 1,
            }

            # Persist through evidence bridge
            try:
                persisted = self._persist_detection_result_with_lineage(
                    inp=inp,
                    result=result,
                    detector=detector,
                    job_run_id=job_run_id,
                    universe_snapshot_id=inp.universe_snapshot_id,
                    code_commit_sha=code_commit_sha,
                    trading_date=trading_date,
                    scan_id=scan_id,
                    detector_version=detector_version,
                    point_in_time_passed=result.features.point_in_time_passed,
                    lookahead_guard_passed=result.features.lookahead_guard_passed,
                )
            except IntegrityError:
                existing_after_race = (
                    self._existing_signal_id(
                        detector.pattern_id,
                        inp.ticker,
                        identity_hash,
                    )
                )
                if existing_after_race:
                    diag.duplicate_suppressed_count += 1
                    continue
                diag.error_count += 1
                diag.errors.append({
                    "detector_id": detector.pattern_id,
                    "ticker": inp.ticker,
                    "error": "integrity_error_without_existing_signal",
                    "traceback": traceback.format_exc(),
                })
                continue
            except Exception as exc:
                diag.error_count += 1
                diag.errors.append({
                    "detector_id": detector.pattern_id,
                    "ticker": inp.ticker,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                continue

            if persisted.signal_ids:
                if persisted.feature_snapshot_id:
                    diag.feature_snapshot_count += 1
                diag.fired_count += len(persisted.signal_ids)
            else:
                diag.identity_refused_count += 1

        if not self._nested_persistence:
            self._session.autoflush = previous_autoflush
        self._emit_progress(
            "detector_progress",
            {
                "detector_id": detector.pattern_id,
                "trading_date": trading_date,
                "evaluated_count": diag.evaluated_count,
                "input_count": len(inputs),
                "feature_snapshot_count": diag.feature_snapshot_count,
                "fired_count": diag.fired_count,
                "skipped_count": diag.skipped_count,
                "error_count": diag.error_count,
                "nested_persistence": self._nested_persistence,
            },
        )

        failure_count = (
            diag.error_count
            + diag.identity_refused_count
            + diag.lookahead_failure_count
        )
        if failure_count > 0:
            diag.detector_status = "partial_failed" if diag.fired_count > 0 else "failed"

        return diag

    def _existing_signal_id(
        self,
        pattern_id: str,
        ticker: str,
        signal_identity_hash: str,
    ):
        context = (
            self._session.begin_nested()
            if self._nested_persistence
            else nullcontext()
        )
        with context:
            return (
                self._session.query(SignalRegistry.signal_id)
                .filter(
                    SignalRegistry.pattern_id == pattern_id,
                    SignalRegistry.ticker == ticker,
                    SignalRegistry.signal_identity_hash == signal_identity_hash,
                )
                .first()
            )

    def _persist_detection_result_with_lineage(self, *, inp: PatternInput, **kwargs):
        if self._nested_persistence:
            with self._session.begin_nested():
                return persist_detection_result(
                    self._session,
                    flush=True,
                    data_lineage_ids=self._resolved_input_lineage_ids(inp),
                    **kwargs,
                )
        return persist_detection_result(
            self._session,
            flush=False,
            data_lineage_ids=self._resolved_input_lineage_ids(inp),
            **kwargs,
        )

    def _emit_progress(self, event: str, payload: dict[str, Any]) -> None:
        if self._progress_callback is None:
            return
        try:
            self._progress_callback(event, payload)
        except Exception:
            pass

    def _resolved_input_lineage_ids(self, inp: PatternInput) -> List[str]:
        """Resolve input lineage hashes to data_lineage IDs when rows exist."""
        lineage_ids: List[str] = []
        seen: set[str] = set()

        for lineage_id in inp.lineage_ids:
            if lineage_id and lineage_id not in seen:
                lineage_ids.append(lineage_id)
                seen.add(lineage_id)

        lineage_hashes = [
            str(value).strip()
            for value in inp.lineage_hashes
            if str(value or "").strip()
        ]
        if lineage_hashes:
            rows = (
                self._session.query(DataLineage.data_lineage_id)
                .filter(DataLineage.raw_payload_hash.in_(lineage_hashes))
                .all()
            )
            for (data_lineage_id,) in rows:
                if data_lineage_id and data_lineage_id not in seen:
                    lineage_ids.append(data_lineage_id)
                    seen.add(data_lineage_id)

        return lineage_ids
