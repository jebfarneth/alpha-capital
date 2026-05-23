"""
All-firings forward-return population job.

Computes forward_return for every mature signal_registry row that can be
priced. Includes selected, skipped, vetoed, cash-throttled, unfilled, and
untraded candidates. Missing provider pricing stays retryable; terminal bad
price states write outcome_unavailable with reason instead of silently
dropping rows.

Per MeasurementSpine.md section 3.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from sqlalchemy.orm import Session

from alpha.db.models import SignalRegistry
from alpha.jobs.contracts import BaseJob, JobContext, JobResult


class ForwardReturnJob(BaseJob):
    """Populate forward_return for mature signal firings."""

    job_name = "forward_return_population"
    job_type = "measurement"

    def __init__(
        self,
        session: Session,
        price_fn: Callable[
            [str, object, Optional[str]],
            Optional[Tuple[Optional[float], Optional[float]]],
        ],
        maturity_fn: Optional[
            Callable[[object, Optional[str]], bool]
        ] = None,
    ):
        """
        Args:
            session: DB session.
            price_fn: (ticker, signal_timestamp, signal_horizon) ->
                      (entry_price, exit_price) or None if pricing unavailable.
            maturity_fn: (signal_timestamp, signal_horizon) -> bool.
                         If None, all pending signals are considered mature.
        """
        self._session = session
        self._price_fn = price_fn
        self._maturity_fn = maturity_fn

    def run(self, ctx: JobContext) -> JobResult:
        pending = (
            self._session.query(SignalRegistry)
            .filter(
                (SignalRegistry.forward_return_status == "pending")
                | (SignalRegistry.forward_return_status == "pricing_unavailable_retry")
                | (SignalRegistry.forward_return_status.is_(None))
            )
            .all()
        )

        computed = 0
        unavailable = 0
        retryable_unavailable = 0
        immature = 0

        for sig in pending:
            if self._maturity_fn and not self._maturity_fn(
                sig.signal_timestamp, sig.signal_horizon
            ):
                immature += 1
                continue

            prices = self._price_fn(
                sig.ticker, sig.signal_timestamp, sig.signal_horizon
            )
            if prices is None:
                sig.forward_return_status = "pricing_unavailable_retry"
                sig.outcome_unavailable_reason = "pricing_unavailable"
                retryable_unavailable += 1
                continue

            entry_price, exit_price = prices

            if entry_price is None or entry_price <= 0:
                sig.forward_return_status = "outcome_unavailable"
                sig.outcome_unavailable_reason = "invalid_entry_price"
                unavailable += 1
                continue

            if exit_price is None:
                sig.forward_return_status = "outcome_unavailable"
                sig.outcome_unavailable_reason = "missing_exit_price"
                unavailable += 1
                continue

            sig.intended_entry_price = entry_price
            sig.forward_return = (exit_price - entry_price) / entry_price
            sig.forward_return_status = "computed"
            computed += 1

        self._session.flush()

        return JobResult(
            status="finished",
            metrics={
                "total_pending": len(pending),
                "computed": computed,
                "unavailable": unavailable,
                "retryable_unavailable": retryable_unavailable,
                "immature": immature,
            },
        )
