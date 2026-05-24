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
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from numbers import Real
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from alpha.data.contracts import AdapterResponse, stable_hash
from alpha.data.fmp import FmpScreenerResult
from alpha.db.models import CanonicalUniverseScan, SecurityProfile, UniverseScan, UniverseSnapshot
from alpha.evidence.writer import record_data_lineage, record_universe_snapshot
from alpha.jobs.contracts import BaseJob, JobContext, JobResult
from alpha.jobs.security_type import (
    COMMON_STOCK,
    NON_COMMON_TYPES,
    REFRESH_STATUS_ENRICHED,
    UNKNOWN,
)

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


def _datetime_order_key(value: datetime) -> datetime:
    """Return a comparable timestamp key across SQLite aware/naive round trips."""
    if value.tzinfo is not None and value.utcoffset() is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _security_profile_stale(
    profile: SecurityProfile,
    asof: datetime,
    max_age_days: Optional[int],
) -> bool:
    if max_age_days is None:
        return False
    if profile.last_refreshed_at is None:
        return True
    cutoff = _datetime_order_key(asof) - timedelta(days=max_age_days)
    return _datetime_order_key(profile.last_refreshed_at) < cutoff


def _security_profile_cache_hash(
    profile_cache: Dict[str, SecurityProfile],
    symbols: List[str],
    *,
    asof: datetime,
    max_age_days: Optional[int],
) -> str:
    rows = []
    for symbol in sorted(set(symbols)):
        profile = profile_cache.get(symbol)
        if profile is None:
            rows.append({"symbol": symbol, "cache_status": "missing"})
            continue
        rows.append({
            "symbol": symbol,
            "cache_status": "hit",
            "security_type": profile.security_type,
            "refresh_status": profile.refresh_status,
            "classifier_version": profile.classifier_version,
            "classification_input_hash": profile.classification_input_hash,
            "classification_output_hash": profile.classification_output_hash,
            "stale": _security_profile_stale(profile, asof, max_age_days),
        })
    return stable_hash({
        "profile_cache_max_age_days": max_age_days,
        "security_profiles": rows,
    })


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
    candidate_scan = session.get(UniverseScan, scan_id)
    if candidate_scan is None:
        raise ValueError(f"canonical universe scan_id does not exist: {scan_id}")

    def _replace_if_current_or_newer(existing: CanonicalUniverseScan) -> None:
        existing_scan = session.get(UniverseScan, existing.scan_id)
        if (
            existing_scan is not None
            and _datetime_order_key(candidate_scan.asof_timestamp)
            < _datetime_order_key(existing_scan.asof_timestamp)
        ):
            return

        existing.scan_id = scan_id
        existing.selected_job_run_id = job_run_id
        existing.selected_at = datetime.now(timezone.utc)
        existing.selection_reason = "latest_successful_scan"
        session.flush()

    existing = (
        session.query(CanonicalUniverseScan)
        .filter(CanonicalUniverseScan.trading_date == trading_date)
        .first()
    )
    if existing:
        _replace_if_current_or_newer(existing)
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
        if session.get(UniverseScan, scan_id) is None:
            raise ValueError(
                f"canonical universe scan_id does not exist after insert race: {scan_id}"
            )

    existing = (
        session.query(CanonicalUniverseScan)
        .filter(CanonicalUniverseScan.trading_date == trading_date)
        .first()
    )
    if existing is None:
        raise RuntimeError(
            "canonical universe scan insert raced but no competing pointer is visible; "
            "retry the universe builder job"
        )
    _replace_if_current_or_newer(existing)


def _dedupe_screener_rows(
    rows: List[FmpScreenerResult],
) -> Tuple[List[Tuple[FmpScreenerResult, bool, Optional[str], str]], int]:
    """Deduplicate rows by symbol, preferring rows that pass inclusion."""
    selected: Dict[str, Tuple[FmpScreenerResult, bool, Optional[str], str]] = {}

    def _rank(candidate: Tuple[FmpScreenerResult, bool, Optional[str], str]) -> Tuple:
        stock, included, reason, symbol = candidate
        market_cap = _finite_real(stock.market_cap)
        price = _finite_real(stock.price)
        return (
            1 if included else 0,
            market_cap if market_cap is not None else float("-inf"),
            price if price is not None else float("-inf"),
            _clean_symbol(stock.exchange),
            _clean_symbol(stock.country),
            _clean_symbol(reason),
            symbol,
        )

    duplicate_count = 0
    for stock in rows:
        included, reason = _classify(stock)
        symbol = _clean_symbol(stock.symbol)
        candidate = (stock, included, reason, symbol)
        existing = selected.get(symbol)
        if existing is None:
            selected[symbol] = candidate
            continue

        duplicate_count += 1
        if _rank(candidate) > _rank(existing):
            selected[symbol] = candidate

    return list(selected.values()), duplicate_count


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
        profile_cache_max_age_days: Optional[int] = 7,
        require_security_profile_cache: bool = False,
        min_security_profile_coverage: float = 0.95,
    ):
        if profile_cache_max_age_days is not None and profile_cache_max_age_days < 0:
            raise ValueError("profile_cache_max_age_days must be non-negative or None")
        if not 0.0 <= min_security_profile_coverage <= 1.0:
            raise ValueError("min_security_profile_coverage must be between 0 and 1")
        self._session = session
        self._screener_response = screener_response
        self._profile_cache_max_age_days = profile_cache_max_age_days
        self._require_security_profile_cache = require_security_profile_cache
        self._min_security_profile_coverage = min_security_profile_coverage
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
                deduped_count=0,
                duplicate_symbol_count=0,
                included_count=0,
                excluded_count=0,
                source_lineage_hash=lineage.raw_payload_hash,
                run_status="failed",
                metric_json=json.dumps({
                    "raw_count": 0,
                    "deduped_count": 0,
                    "duplicate_symbol_count": 0,
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
            deduped_count=0,
            duplicate_symbol_count=0,
            included_count=0,
            excluded_count=0,
            source_lineage_hash=lineage.raw_payload_hash,
            run_status="building",
        )
        self._session.add(scan)
        self._session.flush()

        included_count = 0
        excluded_count = 0
        exclusion_counts: Counter = Counter()
        snapshot_content: list = []
        deduped_rows, duplicate_symbol_count = _dedupe_screener_rows(resp.data)

        # Load security profile cache
        profile_cache = self._load_security_profile_cache()
        security_profile_cache_hash = _security_profile_cache_hash(
            profile_cache,
            [symbol for _, _, _, symbol in deduped_rows],
            asof=resp.lineage.asof_timestamp,
            max_age_days=self._profile_cache_max_age_days,
        )
        cache_hit_count = 0
        cache_miss_count = 0
        cache_miss_included_count = 0
        security_type_exclusion_counts: Counter = Counter()
        security_type_unknown_count = 0
        security_profile_unresolved_count = 0
        security_profile_stale_count = 0
        security_profile_required_count = 0
        security_profile_enriched_count = 0
        security_profile_cache_miss_required_count = 0
        security_type_suffix_rescue_count = 0

        for stock, included, reason, symbol in deduped_rows:
            market_cap = _finite_real(stock.market_cap)
            price = _finite_real(stock.price)
            exchange = _clean_symbol(stock.exchange)
            profile_required = included or reason == "non_common_symbol_suffix"
            if profile_required:
                security_profile_required_count += 1

            # Security profile cache lookup
            security_type = None
            cached_profile = profile_cache.get(symbol)
            if cached_profile is not None:
                cache_hit_count += 1
                security_type = cached_profile.security_type
                refresh_status = cached_profile.refresh_status
                profile_usable = False
                stale = _security_profile_stale(
                    cached_profile,
                    resp.lineage.asof_timestamp,
                    self._profile_cache_max_age_days,
                )
                if stale:
                    if profile_required:
                        security_profile_stale_count += 1
                    if included:
                        included = False
                        reason = "security_profile_stale"
                elif refresh_status != REFRESH_STATUS_ENRICHED or security_type == UNKNOWN:
                    if profile_required:
                        security_profile_unresolved_count += 1
                    if security_type == UNKNOWN:
                        security_type_unknown_count += 1
                    if included:
                        included = False
                        reason = f"security_profile_unresolved:{refresh_status or 'unknown'}"
                else:
                    profile_usable = True
                    if profile_required:
                        security_profile_enriched_count += 1
                    if (
                        security_type == COMMON_STOCK
                        and not included
                        and reason == "non_common_symbol_suffix"
                    ):
                        included = True
                        reason = None
                        security_type_suffix_rescue_count += 1
                if profile_usable and security_type in NON_COMMON_TYPES and (
                    included or reason == "non_common_symbol_suffix"
                ):
                    included = False
                    reason = f"security_type:{security_type}"
                    security_type_exclusion_counts[security_type] += 1
            else:
                cache_miss_count += 1
                if profile_required:
                    security_profile_cache_miss_required_count += 1
                if included:
                    cache_miss_included_count += 1

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
                security_type=security_type,
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
                "security_type": security_type,
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
        if security_profile_required_count:
            security_profile_coverage_ratio = (
                security_profile_enriched_count / security_profile_required_count
            )
        else:
            security_profile_coverage_ratio = 1.0

        metrics: Dict = {
            "raw_count": len(resp.data),
            "deduped_count": len(deduped_rows),
            "duplicate_symbol_count": duplicate_symbol_count,
            "included": included_count,
            "excluded": excluded_count,
            "exclusion_counts": dict(exclusion_counts),
            "security_profile_cache_hash": security_profile_cache_hash,
            "security_profile_cache_hit_count": cache_hit_count,
            "security_profile_cache_miss_count": cache_miss_count,
            "security_profile_cache_miss_included_count": cache_miss_included_count,
            "security_profile_cache_miss_required_count": security_profile_cache_miss_required_count,
            "security_type_exclusion_counts": dict(security_type_exclusion_counts),
            "security_type_unknown_count": security_type_unknown_count,
            "security_profile_unresolved_count": security_profile_unresolved_count,
            "security_profile_stale_count": security_profile_stale_count,
            "security_profile_required_count": security_profile_required_count,
            "security_profile_enriched_count": security_profile_enriched_count,
            "security_profile_coverage_ratio": security_profile_coverage_ratio,
            "require_security_profile_cache": self._require_security_profile_cache,
            "min_security_profile_coverage": self._min_security_profile_coverage,
            "security_type_suffix_rescue_count": security_type_suffix_rescue_count,
            "security_profile_cache_max_age_days": self._profile_cache_max_age_days,
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
        scan.deduped_count = len(deduped_rows)
        scan.duplicate_symbol_count = duplicate_symbol_count
        scan.included_count = included_count
        scan.excluded_count = excluded_count
        scan.security_profile_cache_hash = security_profile_cache_hash
        scan.output_hash = output_hash
        scan.run_status = "finished"
        scan.metric_json = json.dumps(metrics, default=str)
        self._session.flush()

        if (
            self._require_security_profile_cache
            and security_profile_coverage_ratio < self._min_security_profile_coverage
        ):
            metrics["failure_stage"] = "security_profile_coverage"
            scan.run_status = "failed"
            scan.metric_json = json.dumps(metrics, default=str)
            self._session.flush()
            return JobResult(
                status="failed",
                metrics=metrics,
                input_hashes={
                    "screener": resp.lineage.raw_payload_hash,
                    "security_profile_cache": security_profile_cache_hash,
                },
                output_hashes={"universe_snapshots": output_hash},
                errors=[{
                    "stage": "security_profile_coverage",
                    "message": (
                        "security profile coverage "
                        f"{security_profile_coverage_ratio:.4f} below required "
                        f"{self._min_security_profile_coverage:.4f}"
                    ),
                }],
            )

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
            input_hashes={
                "screener": resp.lineage.raw_payload_hash,
                "security_profile_cache": security_profile_cache_hash,
            },
            output_hashes={"universe_snapshots": output_hash},
        )

    def _load_security_profile_cache(self) -> Dict[str, SecurityProfile]:
        """Load all SecurityProfile rows into a dict keyed by normalized symbol."""
        rows = self._session.query(SecurityProfile).all()
        return {row.symbol: row for row in rows}


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
