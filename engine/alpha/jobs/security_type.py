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
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import or_
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
BUSINESS_DEVELOPMENT_COMPANY = "business_development_company"
UNKNOWN = "unknown"

CLASSIFIER_VERSION = "security_type_v3"

REFRESH_STATUS_ENRICHED = "enriched"
REFRESH_STATUS_NO_DATA = "no_data"
REFRESH_STATUS_RETRYABLE_ERROR = "retryable_error"
REFRESH_STATUS_FAILED = "failed"

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
    BUSINESS_DEVELOPMENT_COMPANY,
})

BDC_INDUSTRIES = frozenset({"ASSET MANAGEMENT", "FINANCIAL - CREDIT SERVICES"})
SHELL_COMPANY_DESCRIPTION_PHRASES = (
    "BLANK CHECK COMPANY",
    "DOES NOT HAVE SIGNIFICANT OPERATIONS",
    "BUSINESS COMBINATION",
    "SHARE EXCHANGE",
    "ASSET ACQUISITION",
    "STOCK PURCHASE",
    "SHARE PURCHASE",
    "CAPITAL STOCK EXCHANGE",
)
SPAC_ACQUISITION_SEQUENCE_RE = re.compile(
    r"\bACQUISITION\s+(?:(?:I{1,3}|IV|V|VI{0,3}|IX|X|\d+)\s+)?"
    r"(?:CORP(?:ORATION)?|CO|LIMITED|LTD)\b"
)
SPAC_INVESTMENT_CORP_SEQUENCE_RE = re.compile(
    r"\bINVESTMENT\s+CORP(?:ORATION)?\s+(?:I{1,3}|IV|V|VI{0,3}|IX|X|\d+)\b"
)


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


def _raw_description_text(raw_json: Dict[str, Any]) -> str:
    return _clean_text(raw_json.get("description"))


def _classification_input(profile: FmpCompanyProfile, raw_json: Dict[str, Any]) -> Dict[str, Any]:
    raw_keys = (
        "isFund", "is_fund",
        "isAdr", "isADR", "is_adr",
        "isEtf", "isETF", "is_etf",
        "type", "securityType", "security_type", "assetType", "asset_class",
        "description",
    )
    return {
        "symbol": profile.symbol,
        "company_name": profile.company_name,
        "sector": profile.sector,
        "industry": profile.industry,
        "exchange": profile.exchange,
        "country": profile.country,
        "is_etf": profile.is_etf,
        "raw": {key: raw_json.get(key) for key in raw_keys if key in raw_json},
    }


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
    raw_type = _raw_type_text(raw_json)
    raw_description = _raw_description_text(raw_json)

    if profile.is_etf is True or _raw_bool(raw_json, "isEtf", "isETF", "is_etf") is True:
        return ETF, "is_etf=True"
    if _raw_bool(raw_json, "isFund", "is_fund") is True:
        return MUTUAL_FUND, "raw_flag:isFund"
    if _raw_bool(raw_json, "isAdr", "isADR", "is_adr") is True:
        return ADR, "raw_flag:isAdr"

    if raw_type:
        if _has_phrase(raw_type, "EXCHANGE TRADED FUND") or _has_phrase(raw_type, "ETF"):
            return ETF, "raw_type:ETF"
        if _has_phrase(raw_type, "MUTUAL FUND") or _has_phrase(raw_type, "MUTUAL FUNDS"):
            return MUTUAL_FUND, "raw_type:MUTUAL_FUND"
        if (
            _has_phrase(raw_type, "CLOSED END FUND")
            or _has_phrase(raw_type, "CLOSED END FUNDS")
            or _has_phrase(raw_type, "CLOSED-END FUND")
            or _has_phrase(raw_type, "CLOSED-END FUNDS")
        ):
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
        if _has_phrase(raw_type, "BUSINESS DEVELOPMENT COMPANY") or _has_phrase(raw_type, "BDC"):
            return BUSINESS_DEVELOPMENT_COMPANY, "raw_type:BUSINESS_DEVELOPMENT_COMPANY"

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
    if _has_phrase(name, "FUND") or _has_phrase(name, "FUNDS"):
        if has_closed_indicator:
            return CLOSED_END_FUND, "name_contains:FUND+CLOSED"
        return MUTUAL_FUND, "name_contains:FUND"

    # Closed-end fund indicators
    cef_keywords = ("CLOSED-END", "CLOSED END", "CEF")
    kw = _has_any_phrase(name, cef_keywords)
    if kw:
        return CLOSED_END_FUND, f"name_contains:{kw}"

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
    spac_keywords = (
        "SPAC",
        "BLANK CHECK",
        "ACQUISITION CORP",
        "ACQUISITION CORPORATION",
        "ACQUISITION CO",
    )
    kw = _has_any_phrase(name, spac_keywords)
    if kw:
        return SPAC_OR_BLANK_CHECK, f"name_contains:{kw}"
    if industry == "SHELL COMPANIES":
        kw = _has_any_phrase(raw_description, SHELL_COMPANY_DESCRIPTION_PHRASES)
        if kw:
            return SPAC_OR_BLANK_CHECK, (
                "industry_description:SHELL_COMPANIES+"
                f"{kw.replace(' ', '_')}"
            )
    if SPAC_ACQUISITION_SEQUENCE_RE.search(name):
        return SPAC_OR_BLANK_CHECK, "name_pattern:ACQUISITION_SEQUENCE"
    if SPAC_INVESTMENT_CORP_SEQUENCE_RE.search(name):
        return SPAC_OR_BLANK_CHECK, "name_pattern:INVESTMENT_CORP_SEQUENCE"

    # BDC indicators. FMP does not mark BDCs like ARCC/MAIN with isFund=True.
    if (
        _has_phrase(raw_description, "BUSINESS DEVELOPMENT COMPANY")
        or _has_phrase(raw_description, "BUSINESS DEVELOPMENT COMPANIES")
    ):
        return BUSINESS_DEVELOPMENT_COMPANY, "raw_description:BUSINESS_DEVELOPMENT_COMPANY"
    if _has_phrase(name, "BUSINESS DEVELOPMENT COMPANY") or _has_phrase(name, "BDC"):
        return BUSINESS_DEVELOPMENT_COMPANY, "name_contains:BUSINESS_DEVELOPMENT_COMPANY"
    if sector == "FINANCIAL SERVICES" and industry in BDC_INDUSTRIES:
        if _has_phrase(name, "CAPITAL CORPORATION") or _has_phrase(name, "CAPITAL CORP"):
            return BUSINESS_DEVELOPMENT_COMPANY, "name_industry:CAPITAL_CORPORATION_BDC"

    # ETF fallback from sector/industry
    if sector == "ETF" or industry == "EXCHANGE TRADED FUND":
        return ETF, f"sector_or_industry:ETF"

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
        *,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        self._session = session
        self._adapter = adapter
        self._symbols = symbols
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._sleep_fn = sleep_fn

    def run(self, ctx: JobContext) -> JobResult:
        symbols = self._symbols
        if symbols is None:
            symbols = self._derive_symbols()

        enriched_count = 0
        no_data_count = 0
        retryable_error_count = 0
        failed_count = 0
        retry_attempt_count = 0
        security_type_counts: Counter = Counter()

        for symbol in symbols:
            try:
                resp, attempts = self._get_company_profile_with_retry(symbol)
                retry_attempt_count += max(0, attempts - 1)
            except Exception as exc:
                failed_count += 1
                self._mark_profile_failed(symbol, exc)
                continue

            if not resp.ok or resp.data is None:
                error_type = resp.error.error_type if resp.error else "no_data"
                if resp.error and resp.error.retryable:
                    retryable_error_count += 1
                    refresh_status = REFRESH_STATUS_RETRYABLE_ERROR
                    reason = f"profile_fetch_retryable:{error_type}"
                elif resp.error is None or error_type == "no_data":
                    no_data_count += 1
                    refresh_status = REFRESH_STATUS_NO_DATA
                    reason = "no_profile_data"
                else:
                    failed_count += 1
                    refresh_status = REFRESH_STATUS_FAILED
                    reason = f"profile_fetch_failed:{error_type}"
                self._upsert_profile(
                    symbol=symbol,
                    security_type=UNKNOWN,
                    classification_reason=reason,
                    source_lineage_hash=resp.lineage.raw_payload_hash,
                    profile_payload_hash="",
                    classification_input_hash="",
                    classification_output_hash=stable_hash({
                        "classifier_version": CLASSIFIER_VERSION,
                        "security_type": UNKNOWN,
                        "classification_reason": reason,
                        "refresh_status": refresh_status,
                    }),
                    classifier_version=CLASSIFIER_VERSION,
                    profile_asof_timestamp=resp.lineage.asof_timestamp,
                    raw_profile_json=None,
                    refresh_status=refresh_status,
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
            classification_input = _classification_input(profile, raw_json or {})
            classification_input_hash = stable_hash(classification_input)
            classification_output_hash = stable_hash({
                "classifier_version": CLASSIFIER_VERSION,
                "security_type": security_type,
                "classification_reason": reason,
                "refresh_status": REFRESH_STATUS_ENRICHED,
            })

            self._upsert_profile(
                symbol=symbol,
                security_type=security_type,
                classification_reason=reason,
                source_lineage_hash=resp.lineage.raw_payload_hash,
                profile_payload_hash=profile_hash,
                classification_input_hash=classification_input_hash,
                classification_output_hash=classification_output_hash,
                classifier_version=CLASSIFIER_VERSION,
                profile_asof_timestamp=resp.lineage.asof_timestamp,
                raw_profile_json=json.dumps(profile_payload, sort_keys=True, default=str),
                refresh_status=REFRESH_STATUS_ENRICHED,
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
                "retryable_error_count": retryable_error_count,
                "failed_count": failed_count,
                "retry_attempt_count": retry_attempt_count,
                "security_type_counts": dict(security_type_counts),
                "classifier_version": CLASSIFIER_VERSION,
            },
        )

    def _get_company_profile_with_retry(self, symbol: str):
        attempts = 0
        while True:
            attempts += 1
            resp = self._adapter.get_company_profile(symbol)
            if resp.ok or not (resp.error and resp.error.retryable):
                return resp, attempts
            if attempts > self._max_retries:
                return resp, attempts
            delay = self._retry_backoff_seconds * (2 ** (attempts - 1))
            if delay > 0:
                self._sleep_fn(delay)

    def _derive_symbols(self) -> List[str]:
        from alpha.db.models import UniverseSnapshot
        rows = (
            self._session.query(UniverseSnapshot.ticker)
            .filter(or_(
                UniverseSnapshot.operating_universe_inclusion.is_(True),
                UniverseSnapshot.exclusion_reason == "non_common_symbol_suffix",
            ))
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
        output_hash = stable_hash({
            "classifier_version": CLASSIFIER_VERSION,
            "security_type": UNKNOWN,
            "classification_reason": reason,
            "refresh_status": REFRESH_STATUS_FAILED,
        })
        if existing:
            existing.security_type = UNKNOWN
            existing.refresh_status = REFRESH_STATUS_FAILED
            existing.last_refreshed_at = now
            existing.classification_reason = reason
            existing.classification_input_hash = ""
            existing.classification_output_hash = output_hash
            existing.classifier_version = CLASSIFIER_VERSION
        else:
            self._session.add(SecurityProfile(
                symbol=normalized,
                security_type=UNKNOWN,
                source_provider="FMP",
                source_lineage_hash="",
                profile_payload_hash="",
                classification_input_hash="",
                classification_output_hash=output_hash,
                classifier_version=CLASSIFIER_VERSION,
                profile_asof_timestamp=now,
                last_refreshed_at=now,
                refresh_status=REFRESH_STATUS_FAILED,
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
        classification_input_hash: str,
        classification_output_hash: str,
        classifier_version: str,
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
            existing.classification_input_hash = classification_input_hash
            existing.classification_output_hash = classification_output_hash
            existing.classifier_version = classifier_version
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
                classification_input_hash=classification_input_hash,
                classification_output_hash=classification_output_hash,
                classifier_version=classifier_version,
                profile_asof_timestamp=profile_asof_timestamp,
                last_refreshed_at=now,
                refresh_status=refresh_status,
                raw_profile_json=raw_profile_json,
                classification_reason=classification_reason,
            ))
        self._session.flush()
