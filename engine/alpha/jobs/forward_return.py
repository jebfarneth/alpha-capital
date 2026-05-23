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

import math
from typing import Callable, Optional, Tuple

from sqlalchemy.orm import Session

from alpha.db.models import SignalRegistry
from alpha.jobs.contracts import BaseJob, JobContext, JobResult

RETRYABLE_FORWARD_RETURN_STATUSES = (
    "pending",
    "pricing_unavailable_retry",
    "invalid_price_shape_retry",
    "invalid_entry_price_retry",
    "invalid_exit_price_retry",
    "missing_exit_price_retry",
)
MAX_FORWARD_RETURN_ATTEMPTS = 3


def _finite_price(value: object) -> Optional[float]:
    """Coerce provider prices and reject NaN/Inf before arithmetic."""
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price):
        return None
    return price


def _price_pair(value: object) -> Optional[Tuple[object, object]]:
    """Accept provider prices only when shaped as entry/exit pair."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    return value[0], value[1]


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
        max_attempts: int = MAX_FORWARD_RETURN_ATTEMPTS,
    ):
        """
        Args:
            session: DB session.
            price_fn: (ticker, signal_timestamp, signal_horizon) ->
                      (entry_price, exit_price) or None if pricing unavailable.
            maturity_fn: (signal_timestamp, signal_horizon) -> bool.
                         If None, all pending signals are considered mature.
            max_attempts: terminalize unavailable outcomes at this total attempt count.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._session = session
        self._price_fn = price_fn
        self._maturity_fn = maturity_fn
        self._max_attempts = max_attempts

    def _begin_attempt(self, sig: SignalRegistry) -> int:
        attempts = (sig.forward_return_attempts or 0) + 1
        sig.forward_return_attempts = attempts
        return attempts

    def _mark_unavailable(
        self,
        sig: SignalRegistry,
        *,
        retry_status: str,
        reason: str,
    ) -> bool:
        """Return True if retryable, False if terminal outcome_unavailable."""
        sig.outcome_unavailable_reason = reason
        if (sig.forward_return_attempts or 0) >= self._max_attempts:
            sig.forward_return_status = "outcome_unavailable"
            return False
        sig.forward_return_status = retry_status
        return True

    def run(self, ctx: JobContext) -> JobResult:
        pending = (
            self._session.query(SignalRegistry)
            .filter(
                SignalRegistry.forward_return_status.in_(RETRYABLE_FORWARD_RETURN_STATUSES)
                | SignalRegistry.forward_return_status.is_(None)
            )
            .all()
        )

        computed = 0
        unavailable = 0
        retryable_unavailable = 0
        immature = 0
        pricing_errors = 0

        for sig in pending:
            try:
                if self._maturity_fn and not self._maturity_fn(
                    sig.signal_timestamp, sig.signal_horizon
                ):
                    immature += 1
                    continue
            except Exception as exc:
                self._begin_attempt(sig)
                retryable = self._mark_unavailable(
                    sig,
                    retry_status="pricing_unavailable_retry",
                    reason=f"maturity_fn_error:{type(exc).__name__}",
                )
                retryable_unavailable += int(retryable)
                unavailable += int(not retryable)
                pricing_errors += 1
                continue

            self._begin_attempt(sig)

            try:
                prices = self._price_fn(
                    sig.ticker, sig.signal_timestamp, sig.signal_horizon
                )
            except Exception as exc:
                retryable = self._mark_unavailable(
                    sig,
                    retry_status="pricing_unavailable_retry",
                    reason=f"price_fn_error:{type(exc).__name__}",
                )
                retryable_unavailable += int(retryable)
                unavailable += int(not retryable)
                pricing_errors += 1
                continue

            if prices is None:
                retryable = self._mark_unavailable(
                    sig,
                    retry_status="pricing_unavailable_retry",
                    reason="pricing_unavailable",
                )
                retryable_unavailable += int(retryable)
                unavailable += int(not retryable)
                continue

            price_pair = _price_pair(prices)
            if price_pair is None:
                retryable = self._mark_unavailable(
                    sig,
                    retry_status="invalid_price_shape_retry",
                    reason="invalid_price_shape",
                )
                retryable_unavailable += int(retryable)
                unavailable += int(not retryable)
                pricing_errors += 1
                continue

            raw_entry_price, raw_exit_price = price_pair
            entry_price = _finite_price(raw_entry_price)
            exit_price = _finite_price(raw_exit_price)

            if entry_price is None or entry_price <= 0:
                retryable = self._mark_unavailable(
                    sig,
                    retry_status="invalid_entry_price_retry",
                    reason="invalid_entry_price",
                )
                retryable_unavailable += int(retryable)
                unavailable += int(not retryable)
                continue

            if raw_exit_price is None:
                retryable = self._mark_unavailable(
                    sig,
                    retry_status="missing_exit_price_retry",
                    reason="missing_exit_price",
                )
                retryable_unavailable += int(retryable)
                unavailable += int(not retryable)
                continue

            if exit_price is None or exit_price < 0:
                retryable = self._mark_unavailable(
                    sig,
                    retry_status="invalid_exit_price_retry",
                    reason="invalid_exit_price",
                )
                retryable_unavailable += int(retryable)
                unavailable += int(not retryable)
                continue

            sig.intended_entry_price = entry_price
            sig.forward_return = (exit_price - entry_price) / entry_price
            sig.forward_return_status = "computed"
            sig.outcome_unavailable_reason = None
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
                "pricing_errors": pricing_errors,
            },
        )
