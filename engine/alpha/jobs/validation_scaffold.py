"""
Validation scaffold job.

Minimal validation-job scaffolding proving that validation consumes
all-firings forward_return, not filled-only returns. Full FM/NW/DSR/PBO
math is deferred; this scaffold enforces sample-sufficiency guards.

Per MeasurementSpine.md section 4.
"""

from __future__ import annotations

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from alpha.db.models import SignalRegistry
from alpha.evidence.writer import record_validation_run
from alpha.jobs.contracts import BaseJob, JobContext, JobResult

MINIMUM_SAMPLE_SIZE = 30
OBSERVED_OUTCOME_STATUSES = (
    "computed",
    "outcome_unavailable",
    "pricing_unavailable_retry",
    "invalid_price_shape_retry",
    "invalid_entry_price_retry",
    "invalid_exit_price_retry",
    "missing_exit_price_retry",
)

CONFIDENCE_TIERS = {
    "insufficient_sample": 0.50,
    "monitoring": 0.75,
    "reduced_confidence": 0.75,
    "validated": 1.0,
    "fail_confidence": 0.50,
}


class ValidationScaffoldJob(BaseJob):
    """Sample-sufficiency guard over all-firings forward returns."""

    job_name = "validation_scaffold"
    job_type = "validation"

    def __init__(self, session: Session, *, minimum_sample: int = MINIMUM_SAMPLE_SIZE):
        self._session = session
        self._minimum_sample = minimum_sample

    def run(self, ctx: JobContext) -> JobResult:
        pattern_stats = (
            self._session.query(
                SignalRegistry.pattern_id,
                func.count().label("total_firings"),
                func.sum(
                    case((SignalRegistry.forward_return_status == "computed", 1), else_=0)
                ).label("computed_sample_size"),
                func.sum(
                    case((SignalRegistry.forward_return_status != "computed", 1), else_=0)
                ).label("unavailable_sample_size"),
                func.avg(SignalRegistry.forward_return).label("mean_forward_return_computed"),
            )
            .filter(SignalRegistry.forward_return_status.in_(OBSERVED_OUTCOME_STATUSES))
            .group_by(SignalRegistry.pattern_id)
            .all()
        )

        results = {}
        for pattern_id, total_firings, computed_sample_size, unavailable_sample_size, mean_forward_return_computed in pattern_stats:
            computed_count = int(computed_sample_size or 0)
            unavailable_count = int(unavailable_sample_size or 0)
            total_count = int(total_firings or 0)

            if computed_count < self._minimum_sample:
                tier = "insufficient_sample"
            else:
                tier = "monitoring"

            results[pattern_id] = {
                "sample_size": total_count,
                "computed_sample_size": computed_count,
                "unavailable_sample_size": unavailable_count,
                "mean_forward_return_computed": (
                    float(mean_forward_return_computed)
                    if mean_forward_return_computed is not None
                    else None
                ),
                "confidence_tier": tier,
                "validation_weight_multiplier": CONFIDENCE_TIERS[tier],
            }

            record_validation_run(
                self._session,
                job_run_id=ctx.job_run_id,
                run_type="measurement_spine_scaffold",
                pattern_id=pattern_id,
                sample_size=total_count,
                metrics={
                    "mean_forward_return_computed": (
                        float(mean_forward_return_computed)
                        if mean_forward_return_computed is not None
                        else None
                    ),
                    "sample_size": total_count,
                    "computed_sample_size": computed_count,
                    "unavailable_sample_size": unavailable_count,
                },
                confidence_tier=tier,
                validation_weight_multiplier=CONFIDENCE_TIERS[tier],
            )

        self._session.flush()

        return JobResult(
            status="finished",
            metrics={
                "patterns_evaluated": len(results),
                "pattern_results": results,
            },
        )
