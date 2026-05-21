"""
Universe builder job.

Builds the operating universe from screener data, applying vault rules:
  - Market cap $30M-$200M canonical target
  - US common stocks
  - Actively trading
  - No ETFs
  - Price/fractional/liquidity fields captured when present

Accepts screener results via injection (not live FMP).
Records universe_snapshots and data_lineage via the evidence writer.
Preserves excluded symbols with operating_universe_inclusion=False.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from alpha.data.contracts import AdapterResponse, LineageMeta, stable_hash
from alpha.data.fmp import FmpScreenerResult
from alpha.evidence.writer import record_data_lineage, record_universe_snapshot
from alpha.jobs.contracts import BaseJob, JobContext, JobResult

MCAP_MIN = 30_000_000
MCAP_MAX = 200_000_000


def _classify(stock: FmpScreenerResult) -> tuple:
    """Return (included: bool, exclusion_reason: str | None)."""
    if stock.is_etf:
        return False, "etf"
    if stock.is_actively_trading is False:
        return False, "not_actively_trading"
    if stock.country and stock.country != "US":
        return False, f"country:{stock.country}"
    if stock.market_cap < MCAP_MIN:
        return False, f"mcap_below_{MCAP_MIN}"
    if stock.market_cap > MCAP_MAX:
        return False, f"mcap_above_{MCAP_MAX}"
    if stock.price is not None and stock.price <= 0:
        return False, "zero_or_negative_price"
    return True, None


class UniverseBuilderJob(BaseJob):
    """Builds operating universe from injected screener data."""

    job_name = "universe_builder"
    job_type = "universe_scan"

    def __init__(
        self,
        session: Session,
        screener_response: AdapterResponse[List[FmpScreenerResult]],
    ):
        self._session = session
        self._screener_response = screener_response

    def run(self, ctx: JobContext) -> JobResult:
        resp = self._screener_response

        # Record data lineage from the adapter response
        lineage = record_data_lineage(
            self._session,
            provider=resp.lineage.provider,
            endpoint=resp.lineage.endpoint,
            asof_timestamp=resp.lineage.asof_timestamp,
            raw_payload_hash=resp.lineage.raw_payload_hash,
            request_timestamp=resp.lineage.request_timestamp,
            freshness_seconds=resp.lineage.freshness_seconds,
            source_authority=resp.lineage.source_authority,
            data_quality_flags=resp.lineage.data_quality_flags,
            job_run_id=ctx.job_run_id,
        )

        if not resp.ok or resp.data is None:
            return JobResult(
                status="failed",
                input_hashes={"screener": resp.lineage.raw_payload_hash},
                errors=[{
                    "stage": "screener_fetch",
                    "message": resp.error.message if resp.error else "no data",
                }],
            )

        included_count = 0
        excluded_count = 0
        ticker_decisions = []

        for stock in resp.data:
            included, reason = _classify(stock)
            record_universe_snapshot(
                self._session,
                job_run_id=ctx.job_run_id,
                ticker=stock.symbol,
                asof_timestamp=resp.lineage.asof_timestamp,
                source_provider=resp.lineage.provider,
                market_cap=stock.market_cap,
                price=stock.price,
                operating_universe_inclusion=included,
                exclusion_reason=reason,
                source_lineage_hash=lineage.raw_payload_hash,
            )
            ticker_decisions.append((stock.symbol, included, reason))
            if included:
                included_count += 1
            else:
                excluded_count += 1

        self._session.flush()

        output_hash = stable_hash({
            "tickers": sorted(ticker_decisions),
            "included": included_count,
            "excluded": excluded_count,
        })

        return JobResult(
            status="finished",
            metrics={
                "total_screened": len(resp.data),
                "included": included_count,
                "excluded": excluded_count,
            },
            input_hashes={"screener": resp.lineage.raw_payload_hash},
            output_hashes={"universe_snapshots": output_hash},
        )
