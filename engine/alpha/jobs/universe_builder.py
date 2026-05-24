"""
Universe builder job.

Builds the operating universe from screener data, applying vault rules:
  - Market cap $30M-$200M, finite
  - Price >= $3.00, finite
  - US country only
  - NASDAQ / NYSE / AMEX exchanges only
  - Actively trading
  - No ETFs
  - Conservative symbol cleanup (separators, warrant/unit suffixes)
  - No broad final-letter exclusion

Per Data-Sourcing-Audit.md Universe Filter section and
MeasurementSpine.md section 1.

Accepts screener results via injection (not live FMP).
Records universe_snapshots, universe_scans, and canonical_universe_scans.
Preserves excluded symbols with operating_universe_inclusion=False.
"""

from __future__ import annotations

import json
import math
import uuid
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from numbers import Real
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from alpha.data.contracts import AdapterResponse, stable_hash
from alpha.data.fmp import FmpScreenerResult
from alpha.db.models import CanonicalUniverseScan, UniverseScan, UniverseSnapshot
from alpha.evidence.writer import record_data_lineage, record_universe_snapshot
from alpha.jobs.contracts import BaseJob, JobContext, JobResult

MCAP_MIN = 30_000_000
MCAP_MAX = 200_000_000
PRICE_MIN = 3.0
ALLOWED_EXCHANGES = frozenset({"NASDAQ", "NYSE", "AMEX"})


def _clean_symbol(symbol: object) -> str:
    if symbol is None:
        return ""
    return str(symbol).strip().upper()


def _finite_real(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, Real) or isinstance(value, Decimal):
        number = float(value)
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def _is_non_common_symbol(symbol: object) -> Tuple[bool, Optional[str]]:
    """Conservative symbol cleanup per vault rules.

    Excludes:
      - Symbols containing "." or "-" (non-common separators)
      - Five-or-more-character symbols ending in "WS" or "WT" (warrant forms)
      - Five-character symbols ending in "W", "U", or "X" (warrant/unit/fund forms)

    Does NOT exclude ordinary tickers ending in single letters W/U/R/P/X.

    Future authority: profile/security-type enrichment should replace suffix
    heuristics for fund, ADR, preferred, unit, warrant, right, and
    closed-end-fund classification.
    """
    upper = _clean_symbol(symbol)
    if not upper:
        return True, "non_common_symbol_separator"
    if "." in upper or "-" in upper:
        return True, "non_common_symbol_separator"
    if len(upper) >= 5 and (upper.endswith("WS") or upper.endswith("WT")):
        return True, "non_common_symbol_suffix"
    if len(upper) == 5 and upper[-1] in ("W", "U", "X"):
        return True, "non_common_symbol_suffix"
    return False, None


def _classify(stock: FmpScreenerResult) -> Tuple[bool, Optional[str]]:
    """Return (included, exclusion_reason).

    Check order matches the vault exclusion-reason list so the first
    failing rule determines the persisted reason.
    """
    if stock.is_etf is True:
        return False, "etf"
    if stock.is_etf is not False:
        return False, "etf_status_missing_or_invalid"
    if stock.is_actively_trading is not True:
        return False, "not_actively_trading"
    country = _clean_symbol(stock.country)
    if not country:
        return False, "country_missing"
    if country != "US":
        return False, f"country:{country}"
    exchange = _clean_symbol(stock.exchange)
    if not exchange:
        return False, "exchange_missing"
    if exchange not in ALLOWED_EXCHANGES:
        return False, f"exchange:{exchange}"

    market_cap = _finite_real(stock.market_cap)
    if market_cap is None:
        return False, "mcap_missing_or_invalid"
    if market_cap < MCAP_MIN:
        return False, "mcap_below_30000000"
    if market_cap > MCAP_MAX:
        return False, "mcap_above_200000000"

    price = _finite_real(stock.price)
    if price is None:
        return False, "price_missing_or_invalid"
    if price < PRICE_MIN:
        return False, "price_below_3"

    excluded, reason = _is_non_common_symbol(stock.symbol)
    if excluded:
        return False, reason

    return True, None


def _derive_trading_date(params: Dict[str, Any], asof: datetime) -> str:
    """Extract trading_date from job params or derive from asof timestamp."""
    raw = params.get("trading_date")
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    if isinstance(raw, date):
        return raw.isoformat()
    if isinstance(raw, str):
        cleaned = raw.strip()
        if cleaned:
            return date.fromisoformat(cleaned).isoformat()
    return asof.date().isoformat()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_canonical_universe_scan(
    session: Session, trading_date: str,
) -> Optional[UniverseScan]:
    """Return the canonical universe scan for a trading date, or None."""
    canonical = (
        session.query(CanonicalUniverseScan)
        .filter(CanonicalUniverseScan.trading_date == trading_date)
        .first()
    )
    if canonical is None:
        return None
    return session.get(UniverseScan, canonical.scan_id)


def get_canonical_universe_members(
    session: Session,
    trading_date: str,
    *,
    included_only: bool = True,
) -> List[UniverseSnapshot]:
    """Return universe snapshot rows from the canonical scan for a trading date."""
    canonical = (
        session.query(CanonicalUniverseScan)
        .filter(CanonicalUniverseScan.trading_date == trading_date)
        .first()
    )
    if canonical is None:
        return []
    query = (
        session.query(UniverseSnapshot)
        .filter(UniverseSnapshot.scan_id == canonical.scan_id)
    )
    if included_only:
        query = query.filter(
            UniverseSnapshot.operating_universe_inclusion.is_(True)
        )
    return query.order_by(UniverseSnapshot.ticker).all()


def _upsert_canonical_universe_scan(
    session: Session,
    *,
    trading_date: str,
    scan_id: str,
    job_run_id: str,
) -> None:
    """Select scan_id as the canonical scan for trading_date."""
    existing = (
        session.query(CanonicalUniverseScan)
        .filter(CanonicalUniverseScan.trading_date == trading_date)
        .first()
    )
    if existing:
        existing.scan_id = scan_id
        existing.selected_job_run_id = job_run_id
        existing.selected_at = datetime.now(timezone.utc)
        existing.selection_reason = "latest_successful_scan"
        session.flush()
        return

    try:
        with session.begin_nested():
            session.add(CanonicalUniverseScan(
                trading_date=trading_date,
                scan_id=scan_id,
                selected_job_run_id=job_run_id,
                selection_reason="first_successful_scan",
            ))
            session.flush()
        return
    except IntegrityError:
        # Another runner inserted the date between our read and insert.
        pass

    existing = (
        session.query(CanonicalUniverseScan)
        .filter(CanonicalUniverseScan.trading_date == trading_date)
        .one()
    )
    existing.scan_id = scan_id
    existing.selected_job_run_id = job_run_id
    existing.selected_at = datetime.now(timezone.utc)
    existing.selection_reason = "latest_successful_scan"
    session.flush()


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

class UniverseBuilderJob(BaseJob):
    """Builds operating universe from injected screener data."""

    job_name = "universe_builder"
    job_type = "universe_scan"

    def __init__(
        self,
        session: Session,
        screener_response: AdapterResponse[List[FmpScreenerResult]],
        *,
        slice_diagnostics: Optional[List[Any]] = None,
    ):
        self._session = session
        self._screener_response = screener_response
        self._slice_diagnostics = (
            [_slice_diagnostic_to_dict(d) for d in slice_diagnostics]
            if slice_diagnostics is not None
            else None
        )

    def run(self, ctx: JobContext) -> JobResult:
        resp = self._screener_response
        trading_date = _derive_trading_date(ctx.params, resp.lineage.asof_timestamp)

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
            scan_id = str(uuid.uuid4())
            self._session.add(UniverseScan(
                scan_id=scan_id,
                trading_date=trading_date,
                job_run_id=ctx.job_run_id,
                asof_timestamp=resp.lineage.asof_timestamp,
                provider=resp.lineage.provider,
                raw_count=0,
                included_count=0,
                excluded_count=0,
                source_lineage_hash=lineage.raw_payload_hash,
                run_status="failed",
                metric_json=json.dumps({
                    "raw_count": 0,
                    "included": 0,
                    "excluded": 0,
                    "failure_stage": "screener_fetch",
                }),
            ))
            self._session.flush()
            return JobResult(
                status="failed",
                input_hashes={"screener": resp.lineage.raw_payload_hash},
                errors=[{
                    "stage": "screener_fetch",
                    "message": resp.error.message if resp.error else "no data",
                }],
            )

        scan_id = str(uuid.uuid4())

        # Create scan row before snapshots so FK is satisfied
        scan = UniverseScan(
            scan_id=scan_id,
            trading_date=trading_date,
            job_run_id=ctx.job_run_id,
            asof_timestamp=resp.lineage.asof_timestamp,
            provider=resp.lineage.provider,
            raw_count=0,
            included_count=0,
            excluded_count=0,
            source_lineage_hash=lineage.raw_payload_hash,
            run_status="building",
        )
        self._session.add(scan)
        self._session.flush()

        included_count = 0
        excluded_count = 0
        duplicate_symbol_count = 0
        exclusion_counts: Counter = Counter()
        snapshot_content: list = []
        seen_symbols: set[str] = set()

        for stock in resp.data:
            included, reason = _classify(stock)
            symbol = _clean_symbol(stock.symbol)
            if symbol in seen_symbols:
                duplicate_symbol_count += 1
                continue
            seen_symbols.add(symbol)
            market_cap = _finite_real(stock.market_cap)
            price = _finite_real(stock.price)
            exchange = _clean_symbol(stock.exchange)
            record_universe_snapshot(
                self._session,
                job_run_id=ctx.job_run_id,
                scan_id=scan_id,
                ticker=symbol,
                asof_timestamp=resp.lineage.asof_timestamp,
                source_provider=resp.lineage.provider,
                market_cap=market_cap,
                price=price,
                primary_exchange=exchange or None,
                operating_universe_inclusion=included,
                exclusion_reason=reason,
                source_lineage_hash=lineage.raw_payload_hash,
            )
            snapshot_content.append({
                "symbol": symbol,
                "market_cap": market_cap,
                "price": price,
                "country": stock.country,
                "exchange": exchange,
                "is_etf": stock.is_etf,
                "is_actively_trading": stock.is_actively_trading,
                "included": included,
                "exclusion_reason": reason,
            })
            if included:
                included_count += 1
            else:
                excluded_count += 1
                if reason:
                    exclusion_counts[reason] += 1

        self._session.flush()

        output_hash = stable_hash({
            "snapshots": sorted(snapshot_content, key=lambda row: row["symbol"]),
            "included": included_count,
            "excluded": excluded_count,
        })

        metrics: Dict = {
            "raw_count": len(resp.data),
            "deduped_count": len(seen_symbols),
            "duplicate_symbol_count": duplicate_symbol_count,
            "included": included_count,
            "excluded": excluded_count,
            "exclusion_counts": dict(exclusion_counts),
        }
        if self._slice_diagnostics is not None:
            metrics["slice_count"] = len(self._slice_diagnostics)
            metrics["slice_limit_hits"] = sum(
                1 for d in self._slice_diagnostics if d.get("hit_limit")
            )
            metrics["slice_subdivision_count"] = sum(
                1 for d in self._slice_diagnostics
                if d.get("hit_limit") and d.get("subdivided")
            )
            metrics["slice_limit_exhausted"] = any(
                d.get("hit_limit") and not d.get("subdivided")
                for d in self._slice_diagnostics
            )
            metrics["slice_diagnostics"] = self._slice_diagnostics

        # Finalize universe_scans row
        scan.raw_count = len(resp.data)
        scan.included_count = included_count
        scan.excluded_count = excluded_count
        scan.output_hash = output_hash
        scan.run_status = "finished"
        scan.metric_json = json.dumps(metrics, default=str)
        self._session.flush()

        # Update canonical pointer — successful builds only
        _upsert_canonical_universe_scan(
            self._session,
            trading_date=trading_date,
            scan_id=scan_id,
            job_run_id=ctx.job_run_id,
        )

        return JobResult(
            status="finished",
            metrics=metrics,
            input_hashes={"screener": resp.lineage.raw_payload_hash},
            output_hashes={"universe_snapshots": output_hash},
        )


def _slice_diagnostic_to_dict(diagnostic: Any) -> Dict:
    if isinstance(diagnostic, dict):
        return diagnostic
    if is_dataclass(diagnostic):
        return asdict(diagnostic)
    return {
        "lower": diagnostic.lower,
        "upper": diagnostic.upper,
        "returned_count": diagnostic.returned_count,
        "hit_limit": diagnostic.hit_limit,
        "query_lower": getattr(diagnostic, "query_lower", None),
        "query_upper": getattr(diagnostic, "query_upper", None),
        "subdivided": getattr(diagnostic, "subdivided", False),
    }
