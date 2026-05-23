"""
Detector orchestration job.

Runs implemented detectors over supplied PatternInput batches and persists
every firing through the evidence bridge. Deduplicates tradable signals by
(pattern_id, ticker, signal_identity_hash) when the detector emits a stable
identity. Signals without identity are persisted but cannot be deduped.

Per MeasurementSpine.md section 2.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy.orm import Session

from alpha.db.models import SignalRegistry
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.patterns.contracts import BasePatternDetector, PatternInput
from alpha.patterns.evidence_bridge import persist_detection_result


class DetectorOrchestrationJob(BaseJob):
    """Run detectors over inputs, persist signals, dedup by identity."""

    job_name = "detector_orchestration"
    job_type = "detector_scan"

    def __init__(
        self,
        session: Session,
        detectors: List[BasePatternDetector],
        inputs: List[PatternInput],
    ):
        self._session = session
        self._detectors = detectors
        self._inputs = inputs

    def run(self, ctx: JobContext) -> JobResult:
        signals_persisted = 0
        duplicates_suppressed = 0
        no_signal_count = 0
        identity_missing_count = 0
        errors: list = []

        for inp in self._inputs:
            for detector in self._detectors:
                try:
                    result = detector.detect(inp)
                except Exception as exc:
                    errors.append({
                        "pattern_id": detector.pattern_id,
                        "ticker": inp.ticker,
                        "error": str(exc),
                    })
                    continue

                if not result.has_signal:
                    no_signal_count += 1
                    continue

                identity_hash: Optional[str] = None
                if result.features:
                    identity_hash = result.features.features.get("signal_identity_hash")

                if identity_hash:
                    existing = (
                        self._session.query(SignalRegistry.signal_id)
                        .filter(
                            SignalRegistry.pattern_id == result.pattern_id,
                            SignalRegistry.ticker == result.ticker,
                            SignalRegistry.signal_identity_hash == identity_hash,
                        )
                        .first()
                    )
                    if existing:
                        duplicates_suppressed += 1
                        continue
                else:
                    identity_missing_count += 1

                persisted = persist_detection_result(
                    self._session,
                    result,
                    detector,
                    job_run_id=ctx.job_run_id,
                    universe_snapshot_id=inp.universe_snapshot_id,
                    data_lineage_ids=inp.lineage_ids,
                    code_commit_sha=ctx.app_commit_sha,
                )
                signals_persisted += len(persisted.signal_ids)

        self._session.flush()

        return JobResult(
            status="finished",
            metrics={
                "signals_persisted": signals_persisted,
                "duplicates_suppressed": duplicates_suppressed,
                "no_signal_evaluations": no_signal_count,
                "identity_missing": identity_missing_count,
                "detector_errors": len(errors),
                "finished_with_errors": bool(errors),
            },
            errors=errors,
        )
