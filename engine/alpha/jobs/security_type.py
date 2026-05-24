"""
Security-type enrichment job and classifier.

Cached reference-data job that calls FmpAdapter.get_company_profile() for
each symbol and writes SecurityProfile rows. The universe builder reads
this cache to exclude confirmed non-common instruments.

Per MeasurementSpine.md section 1 (profile/security-type enrichment).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from alpha.data.contracts import stable_hash
from alpha.data.fmp import FmpAdapter, FmpCompanyProfile
from alpha.db.models import SecurityProfile
from alpha.jobs.contracts import BaseJob, JobContext, JobResult


# ---------------------------------------------------------------------------
# Canonical security_type values
# ---------------------------------------------------------------------------

COMMON_STOCK = "common_stock"
ETF = "etf"
MUTUAL_FUND = "mutual_fund"
CLOSED_END_FUND = "closed_end_fund"
ADR = "adr"
PREFERRED = "preferred"
WARRANT = "warrant"
UNIT = "unit"
RIGHT = "right"
SPAC_OR_BLANK_CHECK = "spac_or_blank_check"
UNKNOWN = "unknown"

NON_COMMON_TYPES = frozenset({
    ETF,
    MUTUAL_FUND,
    CLOSED_END_FUND,
    ADR,
    PREFERRED,
    WARRANT,
    UNIT,
    RIGHT,
    SPAC_OR_BLANK_CHECK,
})


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _clean_text(value: object) -> str:
    return " ".join(str(value or "").upper().split())


def _phrase_pattern(phrase: str) -> str:
    return r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"


def _has_phrase(text: str, phrase: str) -> bool:
    return re.search(_phrase_pattern(phrase), text) is not None


def _has_any_phrase(text: str, phrases: tuple[str, ...]) -> Optional[str]:
    for phrase in phrases:
        if _has_phrase(text, phrase):
            return phrase
    return None


def _raw_bool(raw_json: Dict[str, Any], *keys: str) -> Optional[bool]:
    for key in keys:
        if key not in raw_json:
            continue
        value = raw_json.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            if value == 0:
                return False
            if value == 1:
                return True
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in {"true", "1", "yes", "y"}:
                return True
            if cleaned in {"false", "0", "no", "n"}:
                return False
    return None


def _raw_type_text(raw_json: Dict[str, Any]) -> str:
    values = []
    for key in ("type", "securityType", "security_type", "assetType", "asset_class"):
        value = raw_json.get(key)
        if value:
            values.append(str(value))
    return _clean_text(" ".join(values))


def classify_security_type(
    profile: FmpCompanyProfile,
    *,
    raw_json: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Deterministic classifier. Returns (security_type, classification_reason)."""
    raw_json = raw_json or getattr(profile, "raw", None) or {}
    name = _clean_text(profile.company_name)
    industry = _clean_text(profile.industry)
    sector = _clean_text(profile.sector)
    country = _clean_text(profile.country)
    exchange = _clean_text(profile.exchange)
    raw_type = _raw_type_text(raw_json)

    if profile.is_etf is True or _raw_bool(raw_json, "isEtf", "isETF", "is_etf") is True:
        return ETF, "is_etf=True"
    if _raw_bool(raw_json, "isFund", "is_fund") is True:
        return MUTUAL_FUND, "raw_flag:isFund"
    if _raw_bool(raw_json, "isAdr", "isADR", "is_adr") is True:
        return ADR, "raw_flag:isAdr"

    if raw_type:
        if _has_phrase(raw_type, "EXCHANGE TRADED FUND") or _has_phrase(raw_type, "ETF"):
            return ETF, "raw_type:ETF"
        if _has_phrase(raw_type, "MUTUAL FUND"):
            return MUTUAL_FUND, "raw_type:MUTUAL_FUND"
        if _has_phrase(raw_type, "CLOSED END FUND") or _has_phrase(raw_type, "CLOSED-END FUND"):
            return CLOSED_END_FUND, "raw_type:CLOSED_END_FUND"
        if _has_phrase(raw_type, "ADR") or _has_phrase(raw_type, "AMERICAN DEPOSITARY"):
            return ADR, "raw_type:ADR"
        if _has_phrase(raw_type, "PREFERRED"):
            return PREFERRED, "raw_type:PREFERRED"
        if _has_phrase(raw_type, "WARRANT"):
            return WARRANT, "raw_type:WARRANT"
        if _has_phrase(raw_type, "UNIT"):
            return UNIT, "raw_type:UNIT"
        if _has_phrase(raw_type, "RIGHT"):
            return RIGHT, "raw_type:RIGHT"

    has_closed_indicator = (
        _has_phrase(name, "CLOSED")
        or _has_phrase(name, "CLOSED-END")
        or _has_phrase(name, "CLOSED END")
    )

    # Fund indicators from company name — require word boundary
    fund_phrases = ("MUTUAL FUND", "INCOME FUND", "BOND FUND", "MONEY MARKET")
    kw = _has_any_phrase(name, fund_phrases)
    if kw:
        if has_closed_indicator:
            return CLOSED_END_FUND, f"name_contains:{kw}+CLOSED"
        return MUTUAL_FUND, f"name_contains:{kw}"
    if _has_phrase(name, "FUND"):
        if has_closed_indicator:
            return CLOSED_END_FUND, "name_contains:FUND+CLOSED"
        return MUTUAL_FUND, "name_contains:FUND"

    # Closed-end fund indicators
    cef_keywords = ("CLOSED-END", "CLOSED END", "CEF")
    kw = _has_any_phrase(name, cef_keywords)
    if kw:
        return CLOSED_END_FUND, f"name_contains:{kw}"

    # Industry-level fund classification
    if industry in ("ASSET MANAGEMENT", "SHELL COMPANIES"):
        if _has_any_phrase(name, ("TRUST", "INCOME", "CAPITAL ALLOCATION")):
            return CLOSED_END_FUND, f"industry:{industry}+trust_name"

    # ADR indicators
    if _has_phrase(name, "ADR") or _has_phrase(name, "AMERICAN DEPOSITARY"):
        return ADR, "name_contains:ADR"
    if industry == "SHELL COMPANIES" and _has_phrase(name, "SPONSORED"):
        return ADR, "sponsored_adr_indicator"

    # Preferred indicators
    if _has_phrase(name, "PREFERRED") or _has_phrase(name, "PFD"):
        return PREFERRED, "name_contains:PREFERRED"
    if "%" in name and (_has_phrase(name, "SERIES") or _has_phrase(name, "FIXED")):
        return PREFERRED, "name_pattern:coupon_series"

    # Warrant indicators
    if _has_phrase(name, "WARRANT") or _has_phrase(name, "WARRANTS"):
        return WARRANT, "name_contains:WARRANT"

    # Unit indicators
    if (_has_phrase(name, "UNIT") or _has_phrase(name, "UNITS")) and (
        _has_phrase(name, "CONSISTING") or _has_phrase(name, "COMMON")
    ):
        return UNIT, "name_contains:UNIT_COMPOSITE"
    if re.search(r"\bUNITS?\b$", name):
        return UNIT, "name_ends:UNIT"

    # Right indicators
    if (_has_phrase(name, "RIGHT") or _has_phrase(name, "RIGHTS")) and (
        _has_phrase(name, "SUBSCRIPTION") or _has_phrase(name, "CONTINGENT")
    ):
        return RIGHT, "name_contains:RIGHT"

    # SPAC / blank-check indicators
    spac_keywords = ("SPAC", "BLANK CHECK", "ACQUISITION CORP", "ACQUISITION CO")
    kw = _has_any_phrase(name, spac_keywords)
    if kw:
        return SPAC_OR_BLANK_CHECK, f"name_contains:{kw}"
    if sector == "FINANCIAL SERVICES" and _has_phrase(name, "ACQUISITION"):
        return SPAC_OR_BLANK_CHECK, "sector:financial+acquisition_name"

    # ETF fallback from sector/industry
    if sector == "ETF" or industry == "EXCHANGE TRADED FUND":
        return ETF, f"sector_or_industry:ETF"

    # The operating universe is US common stock. A non-US issuer on a US
    # exchange is non-common for this universe even when the name omits ADR.
    if country and country != "US" and exchange in {"NASDAQ", "NYSE", "AMEX"}:
        return ADR, f"profile_country:{country}"

    # Sufficient data for common_stock classification
    if profile.company_name and profile.exchange:
        return COMMON_STOCK, "profile_fields_present"

    return UNKNOWN, "insufficient_profile_data"


# ---------------------------------------------------------------------------
# Enrichment job
# ---------------------------------------------------------------------------

class SecurityTypeEnrichmentJob(BaseJob):
    """Enrich security profiles from FMP company profiles."""

    job_name = "security_type_enrichment"
    job_type = "reference_data"

    def __init__(
        self,
        session: Session,
        adapter: FmpAdapter,
        symbols: Optional[List[str]] = None,
    ):
        self._session = session
        self._adapter = adapter
        self._symbols = symbols

    def run(self, ctx: JobContext) -> JobResult:
        symbols = self._symbols
        if symbols is None:
            symbols = self._derive_symbols()

        enriched_count = 0
        no_data_count = 0
        failed_count = 0
        security_type_counts: Counter = Counter()

        for symbol in symbols:
            try:
                resp = self._adapter.get_company_profile(symbol)
            except Exception as exc:
                failed_count += 1
                self._mark_profile_failed(symbol, exc)
                continue

            if not resp.ok or resp.data is None:
                no_data_count += 1
                self._upsert_profile(
                    symbol=symbol,
                    security_type=UNKNOWN,
                    classification_reason="no_profile_data",
                    source_lineage_hash=resp.lineage.raw_payload_hash,
                    profile_payload_hash="",
                    profile_asof_timestamp=resp.lineage.asof_timestamp,
                    raw_profile_json=None,
                    refresh_status="no_data",
                )
                continue

            profile = resp.data
            raw_json = getattr(profile, "raw", None)
            security_type, reason = classify_security_type(profile, raw_json=raw_json)
            fallback_profile_payload = {
                "symbol": profile.symbol,
                "company_name": profile.company_name,
                "market_cap": profile.market_cap,
                "sector": profile.sector,
                "industry": profile.industry,
                "exchange": profile.exchange,
                "country": profile.country,
                "is_etf": profile.is_etf,
                "is_actively_trading": profile.is_actively_trading,
                "ipo_date": profile.ipo_date,
            }
            profile_payload = raw_json or fallback_profile_payload
            profile_hash = stable_hash(profile_payload)

            self._upsert_profile(
                symbol=symbol,
                security_type=security_type,
                classification_reason=reason,
                source_lineage_hash=resp.lineage.raw_payload_hash,
                profile_payload_hash=profile_hash,
                profile_asof_timestamp=resp.lineage.asof_timestamp,
                raw_profile_json=json.dumps(profile_payload, sort_keys=True, default=str),
                refresh_status="enriched",
            )
            security_type_counts[security_type] += 1
            enriched_count += 1

        self._session.flush()

        return JobResult(
            status="finished",
            metrics={
                "total_symbols": len(symbols),
                "enriched_count": enriched_count,
                "no_data_count": no_data_count,
                "failed_count": failed_count,
                "security_type_counts": dict(security_type_counts),
            },
        )

    def _derive_symbols(self) -> List[str]:
        from alpha.db.models import UniverseSnapshot
        rows = (
            self._session.query(UniverseSnapshot.ticker)
            .filter(UniverseSnapshot.operating_universe_inclusion.is_(True))
            .distinct()
            .all()
        )
        return [r[0] for r in rows]

    def _mark_profile_failed(self, symbol: str, exc: Exception) -> None:
        normalized = symbol.strip().upper()
        if not normalized:
            return
        existing = (
            self._session.query(SecurityProfile)
            .filter(SecurityProfile.symbol == normalized)
            .first()
        )
        now = datetime.now(timezone.utc)
        reason = f"profile_fetch_exception:{exc.__class__.__name__}"
        if existing:
            existing.refresh_status = "failed"
            existing.last_refreshed_at = now
            if not existing.classification_reason:
                existing.classification_reason = reason
        else:
            self._session.add(SecurityProfile(
                symbol=normalized,
                security_type=UNKNOWN,
                source_provider="FMP",
                source_lineage_hash="",
                profile_payload_hash="",
                profile_asof_timestamp=now,
                last_refreshed_at=now,
                refresh_status="failed",
                raw_profile_json=None,
                classification_reason=reason,
            ))
        self._session.flush()

    def _upsert_profile(
        self,
        *,
        symbol: str,
        security_type: str,
        classification_reason: str,
        source_lineage_hash: str,
        profile_payload_hash: str,
        profile_asof_timestamp: datetime,
        raw_profile_json: Optional[str],
        refresh_status: str,
    ) -> None:
        normalized = symbol.strip().upper()
        existing = (
            self._session.query(SecurityProfile)
            .filter(SecurityProfile.symbol == normalized)
            .first()
        )
        now = datetime.now(timezone.utc)
        if existing:
            existing.security_type = security_type
            existing.classification_reason = classification_reason
            existing.source_lineage_hash = source_lineage_hash
            existing.profile_payload_hash = profile_payload_hash
            existing.profile_asof_timestamp = profile_asof_timestamp
            existing.raw_profile_json = raw_profile_json
            existing.refresh_status = refresh_status
            existing.last_refreshed_at = now
        else:
            self._session.add(SecurityProfile(
                symbol=normalized,
                security_type=security_type,
                source_provider="FMP",
                source_lineage_hash=source_lineage_hash,
                profile_payload_hash=profile_payload_hash,
                profile_asof_timestamp=profile_asof_timestamp,
                last_refreshed_at=now,
                refresh_status=refresh_status,
                raw_profile_json=raw_profile_json,
                classification_reason=classification_reason,
            ))
        self._session.flush()
