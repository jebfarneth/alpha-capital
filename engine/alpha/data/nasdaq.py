"""Nasdaq Trader listing-status adapter.

Primary source for:
  - Current Nasdaq Trader listed-security directories
  - Current-day trading-system adds/deletes
  - Historical trade-halt RSS reason-D deletion corroboration

Does not wire into survivorship scoring. Returns AdapterResponse with
LineageMeta and can archive live public snapshots for future PIT queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import requests

from alpha.data.contracts import (
    AdapterResponse,
    LineageMeta,
    ProviderError,
    aware_utc_or_none,
    stable_hash,
    utcnow,
)
from alpha.db.models import NasdaqListingSnapshot, NasdaqListingSnapshotRow
from alpha.market_calendar import us_equity_session_close_timestamp

PROVIDER = "NASDAQ_TRADER"
SOURCE_AUTHORITY = "NASDAQ_TRADER_LISTING"
NASDAQ_LISTED_ENDPOINT = "/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_ENDPOINT = "/dynamic/SymDir/otherlisted.txt"
TRADING_SYSTEM_ADDS_DELETES_ENDPOINT = "/dynamic/SymDir/TradingSystemAddsDeletes.txt"
TRADE_HALTS_RSS_ENDPOINT = "/rss.aspx"
NASDAQ_TRADER_BASE_URL = "https://www.nasdaqtrader.com"
NASDAQ_LISTED = "nasdaqlisted"
OTHER_LISTED = "otherlisted"
ADDS_DELETES = "trading_system_adds_deletes"
HALT_RSS = "trade_halt_rss"
ARCHIVE_REQUIRED_SOURCE_TYPES = (
    NASDAQ_LISTED,
    OTHER_LISTED,
    ADDS_DELETES,
    HALT_RSS,
)
NASDAQ_LISTED_MIN_ROWS = 1000
OTHER_LISTED_MIN_ROWS = 1000
NASDAQ_TZ = ZoneInfo("America/New_York")
DIRECTORY_FOOTER_PREFIX = "File Creation Time:"
REASON_SECURITY_DELETION = "D"
UNIT_SUFFIX_RE = re.compile(r"^[A-Z0-9]+[.\-/]U$")
UNIT_SECURITY_NAME_RE = re.compile(r"\bUNITS?\b")
NON_COMMON_SECURITY_NAME_PATTERNS = (
    UNIT_SECURITY_NAME_RE,
    re.compile(r"\bWARRANTS?\b"),
    re.compile(r"\bRIGHTS?\b"),
    re.compile(r"\bPREFERRED\b"),
    re.compile(r"\bPREFERENCE\b"),
    re.compile(r"\bNOTES?\b"),
    re.compile(r"\bBONDS?\b"),
    re.compile(r"\bDEBENTURE\b"),
    re.compile(r"\bETF\b"),
    re.compile(r"\bETN\b"),
    re.compile(r"\bFUNDS?\b"),
)
LIMITED_PARTNER_EQUITY_RE = re.compile(
    r"(\bCOMMON\s+UNITS?\b|\bCLASS\s+A\s+SHARES?\b).*(\bL\.?P\.?\b|\bLIMITED\s+PARTNER\b|\bPARTNERSHIP\b)"
    r"|(\bL\.?P\.?\b|\bLIMITED\s+PARTNER\b|\bPARTNERSHIP\b).*(\bCOMMON\s+UNITS?\b|\bCLASS\s+A\s+SHARES?\b)"
)


class NasdaqListingStatus(str, Enum):
    """Safety-shaped listing status values."""

    LISTED_ACTIVE = "LISTED_ACTIVE"
    DELISTED = "DELISTED"
    SUSPENDED = "SUSPENDED"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class NasdaqDirectoryRecord:
    """One normalized row from a Nasdaq Trader symbol directory."""

    source: str
    symbol: str
    security_name: str
    market: Optional[str]
    test_issue: Optional[str]
    etf: Optional[str]
    raw: Dict[str, Any]


@dataclass(frozen=True)
class NasdaqAddsDeletesRecord:
    """One normalized row from TradingSystemAddsDeletes.txt."""

    symbol: str
    company_name: str
    effective_date: Optional[date]
    nasdaq_action: Optional[str]
    bx_action: Optional[str]
    psx_action: Optional[str]
    primary_listing_market: Optional[str]
    raw: Dict[str, Any]


@dataclass(frozen=True)
class NasdaqHaltEvent:
    """One normalized item from the Nasdaq trade-halt RSS feed."""

    symbol: str
    issue_name: Optional[str]
    market: Optional[str]
    reason_code: Optional[str]
    halt_timestamp: Optional[datetime]
    resumption_timestamp: Optional[datetime]
    published_timestamp: Optional[datetime]
    raw: Dict[str, Any]


@dataclass(frozen=True)
class NasdaqSourceFile:
    """Parsed public source payload with integrity metadata."""

    source: str
    url: str
    knowledge_timestamp: datetime
    raw_payload: str
    raw_payload_hash: str
    row_count: int
    records: Tuple[Any, ...]
    footer: Optional[str] = None


@dataclass(frozen=True)
class NasdaqListingStatusResult:
    """Listing-status answer for one symbol at one asof timestamp."""

    symbol: str
    normalized_symbol: str
    status: NasdaqListingStatus
    asof_timestamp: datetime
    source_knowledge_timestamp: Optional[datetime]
    pit_knowable_at_asof: bool
    source: Optional[str]
    reason: str
    matched_symbol: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class NasdaqArchiveCaptureResult:
    """Summary of a self-archive capture."""

    captured_sources: Tuple[str, ...]
    inserted_snapshots: int
    existing_snapshots: int
    inserted_rows: int
    raw_payload_hashes: Dict[str, str]
    failed_sources: Tuple[str, ...] = ()


class NasdaqTraderListingAdapter:
    """Nasdaq Trader public listing-status adapter."""

    def __init__(
        self,
        *,
        base_url: str = NASDAQ_TRADER_BASE_URL,
        session: Optional[requests.Session] = None,
        timeout_seconds: int = 30,
        user_agent: str = "AlphaCapital Nasdaq listing adapter",
    ):
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds
        self._headers = {"User-Agent": user_agent}

    def get_listing_status(
        self,
        symbol: str,
        *,
        asof: datetime,
        archive_session: Any = None,
        use_live: bool = True,
    ) -> AdapterResponse[NasdaqListingStatusResult]:
        """Return daily-grain listing status without treating absence as delisted.

        Contract for the future survivorship gate: directory rosters are
        end-of-day facts. A directory-derived LISTED_ACTIVE is PIT-knowable
        when its source timestamp is at or before asof, or when the source
        timestamp and asof share the same ET trading date and asof is at or
        after that session close. LISTED_ACTIVE may suppress an EDGAR review
        only when no same-trading-date knowable DELETE or reason-D halt exists,
        and archive replay may return LISTED_ACTIVE only when all required
        source families have same-ET-date parsed snapshots. Delisting records
        outrank directory presence and keep strict timestamp knowledge.
        """

        asof_ts = aware_utc_or_none(asof)
        if asof_ts is None:
            return _error_response(
                endpoint="nasdaq_listing_status",
                asof=utcnow(),
                error_type="validation",
                message="Nasdaq listing-status asof timestamp must be timezone-aware datetime",
                retryable=False,
            )

        if archive_session is not None and not use_live:
            return self._get_listing_status_from_archive(
                symbol,
                asof=asof_ts,
                archive_session=archive_session,
            )

        directories_resp = self.get_current_directories(asof=asof_ts)
        if not directories_resp.ok:
            return _unavailable_from_error(
                symbol,
                asof_ts,
                endpoint="nasdaq_listing_status",
                error=directories_resp.error,
                lineage=directories_resp.lineage,
            )
        directories = directories_resp.data or ()
        directory_status = _status_from_directories(symbol, asof_ts, directories)

        deletes_resp = self.get_current_adds_deletes(asof=asof_ts)
        if not deletes_resp.ok:
            return _unavailable_from_error(
                symbol,
                asof_ts,
                endpoint="nasdaq_listing_status",
                error=deletes_resp.error,
                lineage=deletes_resp.lineage,
            )
        deletion_status = _status_from_adds_deletes(
            symbol, asof_ts, deletes_resp.data
        )

        halt_resp = self.get_halt_events(asof=asof_ts, halt_date=_et_date(asof_ts))
        if not halt_resp.ok:
            return _unavailable_from_error(
                symbol,
                asof_ts,
                endpoint="nasdaq_listing_status",
                error=halt_resp.error,
                lineage=halt_resp.lineage,
            )
        halt_status = _status_from_halt_events(symbol, asof_ts, halt_resp.data)
        delisting_status = _select_delisting_status(deletion_status, halt_status)
        if delisting_status is not None:
            return AdapterResponse(
                data=delisting_status,
                lineage=_combined_lineage(
                    "nasdaq_listing_status",
                    asof_ts,
                    [directories_resp.lineage, deletes_resp.lineage, halt_resp.lineage],
                    {"sources": [NASDAQ_LISTED, OTHER_LISTED, ADDS_DELETES, HALT_RSS]},
                ),
            )

        return AdapterResponse(
            data=directory_status,
            lineage=_combined_lineage(
                "nasdaq_listing_status",
                asof_ts,
                [directories_resp.lineage, deletes_resp.lineage, halt_resp.lineage],
                {"sources": [NASDAQ_LISTED, OTHER_LISTED, ADDS_DELETES, HALT_RSS]},
            ),
        )

    def get_listing_statuses(
        self,
        symbols: Sequence[str],
        *,
        asof: datetime,
    ) -> AdapterResponse[List[NasdaqListingStatusResult]]:
        """Batch status lookup with INCONCLUSIVE-rate telemetry."""

        results: List[NasdaqListingStatusResult] = []
        lineages: List[LineageMeta] = []
        for symbol in symbols:
            resp = self.get_listing_status(symbol, asof=asof)
            if not resp.ok:
                return AdapterResponse(data=None, lineage=resp.lineage, error=resp.error)
            if resp.data is not None:
                results.append(resp.data)
            lineages.append(resp.lineage)
        inconclusive_count = sum(
            1 for result in results
            if result.status is NasdaqListingStatus.INCONCLUSIVE
        )
        flags = {
            "symbol_count": len(results),
            "inconclusive_count": inconclusive_count,
            "inconclusive_rate": (
                inconclusive_count / len(results) if results else 0.0
            ),
        }
        return AdapterResponse(
            data=results,
            lineage=_combined_lineage("nasdaq_listing_status_batch", aware_utc_or_none(asof) or utcnow(), lineages, flags),
        )

    def get_current_directories(
        self, *, asof: Optional[datetime] = None
    ) -> AdapterResponse[Tuple[NasdaqSourceFile, NasdaqSourceFile]]:
        """Fetch and parse both live Nasdaq Trader directory files."""

        asof_ts = aware_utc_or_none(asof) if asof is not None else utcnow()
        if asof_ts is None:
            return _error_response(
                endpoint="nasdaq_current_directories",
                asof=utcnow(),
                error_type="validation",
                message="Nasdaq directory asof timestamp must be timezone-aware datetime",
                retryable=False,
            )
        files: List[NasdaqSourceFile] = []
        lineages: List[LineageMeta] = []
        for source, endpoint, min_rows in (
            (NASDAQ_LISTED, NASDAQ_LISTED_ENDPOINT, NASDAQ_LISTED_MIN_ROWS),
            (OTHER_LISTED, OTHER_LISTED_ENDPOINT, OTHER_LISTED_MIN_ROWS),
        ):
            resp = self._fetch_text(endpoint, source=source, asof=asof_ts)
            if not resp.ok:
                return resp  # type: ignore[return-value]
            parsed = _parse_directory_file(
                source,
                self._url(endpoint),
                resp.data or "",
                min_rows=min_rows,
            )
            if isinstance(parsed, ProviderError):
                return AdapterResponse(data=None, lineage=resp.lineage, error=parsed)
            files.append(parsed)
            lineages.append(resp.lineage)
        return AdapterResponse(
            data=tuple(files),  # type: ignore[arg-type]
            lineage=_combined_lineage(
                "nasdaq_current_directories",
                asof_ts,
                lineages,
                {
                    "sources": [NASDAQ_LISTED, OTHER_LISTED],
                    "row_counts": {file.source: file.row_count for file in files},
                    "knowledge_timestamps": {
                        file.source: file.knowledge_timestamp.isoformat()
                        for file in files
                    },
                },
            ),
        )

    def get_current_adds_deletes(
        self, *, asof: Optional[datetime] = None
    ) -> AdapterResponse[NasdaqSourceFile]:
        """Fetch and parse current-day TradingSystemAddsDeletes.txt."""

        asof_ts = aware_utc_or_none(asof) if asof is not None else utcnow()
        if asof_ts is None:
            return _error_response(
                endpoint=ADDS_DELETES,
                asof=utcnow(),
                error_type="validation",
                message="Nasdaq adds/deletes asof timestamp must be timezone-aware datetime",
                retryable=False,
            )
        resp = self._fetch_text(
            TRADING_SYSTEM_ADDS_DELETES_ENDPOINT,
            source=ADDS_DELETES,
            asof=asof_ts,
        )
        if not resp.ok:
            return resp  # type: ignore[return-value]
        parsed = _parse_adds_deletes_file(
            self._url(TRADING_SYSTEM_ADDS_DELETES_ENDPOINT),
            resp.data or "",
        )
        if isinstance(parsed, ProviderError):
            return AdapterResponse(data=None, lineage=resp.lineage, error=parsed)
        return AdapterResponse(
            data=parsed,
            lineage=_combined_lineage(
                ADDS_DELETES,
                asof_ts,
                [resp.lineage],
                {
                    "row_count": parsed.row_count,
                    "knowledge_timestamp": parsed.knowledge_timestamp.isoformat(),
                },
            ),
        )

    def get_halt_events(
        self,
        *,
        asof: Optional[datetime] = None,
        halt_date: Optional[date] = None,
    ) -> AdapterResponse[NasdaqSourceFile]:
        """Fetch and parse Nasdaq trade-halt RSS for one halt date."""

        asof_ts = aware_utc_or_none(asof) if asof is not None else utcnow()
        if asof_ts is None:
            return _error_response(
                endpoint=HALT_RSS,
                asof=utcnow(),
                error_type="validation",
                message="Nasdaq halt RSS asof timestamp must be timezone-aware datetime",
                retryable=False,
            )
        params = {"feed": "tradehalts"}
        if halt_date is not None:
            params["haltdate"] = halt_date.strftime("%m%d%Y")
        endpoint = f"{TRADE_HALTS_RSS_ENDPOINT}?{urlencode(params)}"
        resp = self._fetch_text(endpoint, source=HALT_RSS, asof=asof_ts)
        if not resp.ok:
            return resp  # type: ignore[return-value]
        parsed = _parse_halt_rss_file(self._url(endpoint), resp.data or "")
        if isinstance(parsed, ProviderError):
            return AdapterResponse(data=None, lineage=resp.lineage, error=parsed)
        return AdapterResponse(data=parsed, lineage=resp.lineage)

    def archive_current_snapshot(
        self,
        archive_session: Any,
        *,
        asof: Optional[datetime] = None,
    ) -> AdapterResponse[NasdaqArchiveCaptureResult]:
        """Persist current public sources for future historical-asof queries."""

        asof_ts = aware_utc_or_none(asof) if asof is not None else utcnow()
        if asof_ts is None:
            return _error_response(
                endpoint="nasdaq_archive_current_snapshot",
                asof=utcnow(),
                error_type="validation",
                message="Nasdaq archive asof timestamp must be timezone-aware datetime",
                retryable=False,
            )
        directories_resp = self.get_current_directories(asof=asof_ts)
        if not directories_resp.ok:
            return AdapterResponse(
                data=None, lineage=directories_resp.lineage, error=directories_resp.error
            )
        adds_resp = self.get_current_adds_deletes(asof=asof_ts)
        if not adds_resp.ok:
            return AdapterResponse(
                data=None, lineage=adds_resp.lineage, error=adds_resp.error
            )
        halt_resp = self.get_halt_events(asof=asof_ts, halt_date=_et_date(asof_ts))
        failed_sources: List[str] = []
        files: List[NasdaqSourceFile] = list(directories_resp.data or ()) + [
            adds_resp.data
        ]
        if halt_resp.ok:
            files.append(halt_resp.data)
        else:
            failed_sources.append(HALT_RSS)
        inserted_snapshots = 0
        existing_snapshots = 0
        inserted_rows = 0
        hashes: Dict[str, str] = {}
        for file in files:
            if file is None:
                continue
            existing = _find_existing_snapshot(archive_session, file)
            if existing is not None:
                existing_snapshots += 1
                hashes[file.source] = file.raw_payload_hash
                continue
            snapshot = NasdaqListingSnapshot(
                source_type=file.source,
                source_url=file.url,
                source_knowledge_timestamp=file.knowledge_timestamp,
                raw_payload_hash=file.raw_payload_hash,
                raw_payload=file.raw_payload,
                row_count=file.row_count,
                data_quality_flags_json=json.dumps(
                    {"footer": file.footer, "source": file.source},
                    sort_keys=True,
                    default=str,
                ),
            )
            archive_session.add(snapshot)
            archive_session.flush()
            inserted_snapshots += 1
            hashes[file.source] = file.raw_payload_hash
            for row in _archive_rows(file):
                row.snapshot_id = snapshot.snapshot_id
                archive_session.add(row)
                inserted_rows += 1
        archive_session.flush()
        result = NasdaqArchiveCaptureResult(
            captured_sources=tuple(file.source for file in files if file is not None),
            inserted_snapshots=inserted_snapshots,
            existing_snapshots=existing_snapshots,
            inserted_rows=inserted_rows,
            raw_payload_hashes=hashes,
            failed_sources=tuple(failed_sources),
        )
        return AdapterResponse(
            data=result,
            lineage=_combined_lineage(
                "nasdaq_archive_current_snapshot",
                asof_ts,
                [directories_resp.lineage, adds_resp.lineage, halt_resp.lineage],
                {
                    "inserted_snapshots": inserted_snapshots,
                    "existing_snapshots": existing_snapshots,
                    "inserted_rows": inserted_rows,
                    "failed_sources": failed_sources,
                },
            ),
        )

    def _get_listing_status_from_archive(
        self,
        symbol: str,
        *,
        asof: datetime,
        archive_session: Any,
    ) -> AdapterResponse[NasdaqListingStatusResult]:
        variants = tuple(symbol_variants(symbol))
        directory_snapshots, latest_prior_snapshot = _latest_archive_snapshots_by_source(
            archive_session,
            (NASDAQ_LISTED, OTHER_LISTED),
            asof,
        )
        delisting_status = _status_from_archived_delisting_evidence(
            symbol,
            asof,
            archive_session,
        )
        if delisting_status is not None:
            return AdapterResponse(
                data=delisting_status,
                lineage=_lineage(
                    "nasdaq_listing_status_archive",
                    asof,
                    stable_hash(delisting_status),
                    {
                        "archive_hit": True,
                        "delisting_overlay": True,
                    },
                ),
            )
        if not directory_snapshots:
            if latest_prior_snapshot is not None:
                result = NasdaqListingStatusResult(
                    symbol=str(symbol or "").strip(),
                    normalized_symbol=_normalize_symbol(symbol),
                    status=NasdaqListingStatus.INCONCLUSIVE,
                    asof_timestamp=asof,
                    source_knowledge_timestamp=latest_prior_snapshot.source_knowledge_timestamp,
                    pit_knowable_at_asof=False,
                    source="nasdaq_self_archive",
                    reason="archived_snapshot_stale_for_asof",
                    raw={
                        "symbol_variants": list(variants),
                        "safety": "prior-session archive cannot answer this asof",
                    },
                )
                return AdapterResponse(
                    data=result,
                    lineage=_lineage(
                        "nasdaq_listing_status_archive",
                        asof,
                        stable_hash(result),
                        {"archive_hit": True, "archive_stale_for_asof": True},
                    ),
                )
            result = NasdaqListingStatusResult(
                symbol=str(symbol or "").strip(),
                normalized_symbol=_normalize_symbol(symbol),
                status=NasdaqListingStatus.INCONCLUSIVE,
                asof_timestamp=asof,
                source_knowledge_timestamp=None,
                pit_knowable_at_asof=False,
                source="nasdaq_self_archive",
                reason="no_archived_snapshot_for_asof",
                raw={
                    "symbol_variants": list(variants),
                    "safety": "pre-capture historical status cannot be fabricated",
                },
            )
            return AdapterResponse(
                data=result,
                lineage=_lineage(
                    "nasdaq_listing_status_archive",
                    asof,
                    stable_hash({"symbol": symbol, "archive": "miss"}),
                    {"archive_hit": False},
                ),
            )
        result = _status_from_archived_directories(
            symbol,
            variants,
            asof,
            archive_session,
            tuple(directory_snapshots.values()),
        )
        if result.status is NasdaqListingStatus.LISTED_ACTIVE:
            captured_sources, missing_sources = _archive_required_source_coverage(
                archive_session,
                asof,
            )
            if missing_sources:
                result = _archive_source_coverage_incomplete_status(
                    symbol,
                    asof,
                    captured_sources,
                    missing_sources,
                )
        return AdapterResponse(
            data=result,
            lineage=_lineage(
                "nasdaq_listing_status_archive",
                asof,
                stable_hash(result),
                {"archive_hit": True},
            ),
        )

    def _fetch_text(
        self,
        endpoint: str,
        *,
        source: str,
        asof: datetime,
    ) -> AdapterResponse[str]:
        request_ts = utcnow()
        url = self._url(endpoint)
        try:
            resp = self._session.get(
                url,
                headers=self._headers,
                timeout=self._timeout_seconds,
            )
        except requests.exceptions.Timeout:
            return _error_response(
                endpoint=source,
                asof=asof,
                error_type="timeout",
                message="Nasdaq Trader request timed out",
                retryable=True,
                request_ts=request_ts,
            )
        except requests.exceptions.RequestException as exc:
            return _error_response(
                endpoint=source,
                asof=asof,
                error_type="http",
                message=f"Nasdaq Trader request failed: {exc.__class__.__name__}",
                retryable=True,
                request_ts=request_ts,
            )
        lineage = _lineage(
            source,
            asof,
            stable_hash(resp.text),
            {"url_host": "www.nasdaqtrader.com"},
            request_ts=request_ts,
        )
        if resp.status_code >= 400:
            return AdapterResponse(
                data=None,
                lineage=lineage,
                error=ProviderError(
                    provider=PROVIDER,
                    endpoint=source,
                    status_code=resp.status_code,
                    error_type="http",
                    message=f"Nasdaq Trader returned HTTP {resp.status_code}",
                    retryable=500 <= resp.status_code < 600,
                ),
            )
        return AdapterResponse(data=resp.text, lineage=lineage)

    def _url(self, endpoint: str) -> str:
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        return f"{self._base_url}{endpoint}"


def symbol_variants(symbol: Any) -> Tuple[str, ...]:
    """Return safe symbol variants for FMP-vs-Nasdaq punctuation differences."""

    text = _normalize_symbol(symbol)
    variants: List[str] = []

    def add(value: str) -> None:
        value = _normalize_symbol(value)
        if value and value not in variants:
            variants.append(value)

    for variant in _punctuation_preserving_symbol_variants(text):
        add(variant)
    unit_compact = _unit_compact_symbol(text)
    if unit_compact is not None:
        add(unit_compact)
    return tuple(variants)


def _punctuation_preserving_symbol_variants(symbol: str) -> Tuple[str, ...]:
    text = _normalize_symbol(symbol)
    variants: List[str] = []

    def add(value: str) -> None:
        value = _normalize_symbol(value)
        if value and value not in variants:
            variants.append(value)

    add(text)
    add(text.replace(".", "-"))
    add(text.replace("-", "."))
    add(text.replace("/", "."))
    add(text.replace("/", "-"))
    return tuple(variants)


def _parse_directory_file(
    source: str,
    url: str,
    text: str,
    *,
    min_rows: int,
) -> NasdaqSourceFile | ProviderError:
    lines = _clean_lines(text)
    if len(lines) < 3:
        return _parse_error(source, "Nasdaq directory file is too short")
    footer = lines[-1]
    knowledge_ts = _parse_footer_timestamp(source, footer)
    if isinstance(knowledge_ts, ProviderError):
        return knowledge_ts
    expected = (
        [
            "Symbol",
            "Security Name",
            "Market Category",
            "Test Issue",
            "Financial Status",
            "Round Lot Size",
            "ETF",
            "NextShares",
        ]
        if source == NASDAQ_LISTED
        else [
            "ACT Symbol",
            "Security Name",
            "Exchange",
            "CQS Symbol",
            "ETF",
            "Round Lot Size",
            "Test Issue",
            "NASDAQ Symbol",
        ]
    )
    rows = _pipe_rows(source, lines, expected)
    if isinstance(rows, ProviderError):
        return rows
    if len(rows) < min_rows:
        return _parse_error(
            source,
            f"Nasdaq directory row count {len(rows)} below minimum {min_rows}",
        )
    records: List[NasdaqDirectoryRecord] = []
    for row in rows:
        symbol_key = "Symbol" if source == NASDAQ_LISTED else "ACT Symbol"
        market_key = "Market Category" if source == NASDAQ_LISTED else "Exchange"
        records.append(
            NasdaqDirectoryRecord(
                source=source,
                symbol=_normalize_symbol(row.get(symbol_key)),
                security_name=str(row.get("Security Name") or "").strip(),
                market=_clean_optional(row.get(market_key)),
                test_issue=_clean_optional(row.get("Test Issue")),
                etf=_clean_optional(row.get("ETF")),
                raw=row,
            )
        )
    return NasdaqSourceFile(
        source=source,
        url=url,
        knowledge_timestamp=knowledge_ts,
        raw_payload=text,
        raw_payload_hash=stable_hash(text),
        row_count=len(records),
        records=tuple(records),
        footer=footer,
    )


def _parse_adds_deletes_file(
    url: str,
    text: str,
) -> NasdaqSourceFile | ProviderError:
    source = ADDS_DELETES
    lines = _clean_lines(text)
    if len(lines) < 2:
        return _parse_error(source, "Nasdaq adds/deletes file is too short")
    footer = lines[-1]
    knowledge_ts = _parse_footer_timestamp(source, footer)
    if isinstance(knowledge_ts, ProviderError):
        return knowledge_ts
    expected = [
        "Symbol",
        "Company Name",
        "NASDAQ Action",
        "BX Action",
        "PSX Action",
        "Effective Date",
        "Primary Listing Market",
    ]
    rows = _pipe_rows(source, lines, expected)
    if isinstance(rows, ProviderError):
        return rows
    records: List[NasdaqAddsDeletesRecord] = []
    for row in rows:
        records.append(
            NasdaqAddsDeletesRecord(
                symbol=_normalize_symbol(row.get("Symbol")),
                company_name=str(row.get("Company Name") or "").strip(),
                effective_date=_parse_us_date(row.get("Effective Date")),
                nasdaq_action=_clean_optional(row.get("NASDAQ Action")),
                bx_action=_clean_optional(row.get("BX Action")),
                psx_action=_clean_optional(row.get("PSX Action")),
                primary_listing_market=_clean_optional(row.get("Primary Listing Market")),
                raw=row,
            )
        )
    return NasdaqSourceFile(
        source=source,
        url=url,
        knowledge_timestamp=knowledge_ts,
        raw_payload=text,
        raw_payload_hash=stable_hash(text),
        row_count=len(records),
        records=tuple(records),
        footer=footer,
    )


def _parse_halt_rss_file(url: str, text: str) -> NasdaqSourceFile | ProviderError:
    source = HALT_RSS
    try:
        root = ET.fromstring(_clean_xml_text(text))
    except ET.ParseError as exc:
        return _parse_error(source, f"Nasdaq halt RSS parse failed: {exc}")
    ns = {"ndaq": "http://www.nasdaqtrader.com/"}
    channel = root.find("channel")
    if channel is None:
        return _parse_error(source, "Nasdaq halt RSS missing channel")
    pub_ts = _parse_rss_pubdate(channel.findtext("pubDate"))
    records: List[NasdaqHaltEvent] = []
    for item in channel.findall("item"):
        raw = {
            child.tag.split("}", 1)[-1]: child.text
            for child in list(item)
            if child.text is not None
        }
        symbol = _normalize_symbol(
            item.findtext("ndaq:IssueSymbol", namespaces=ns)
            or item.findtext("title")
        )
        records.append(
            NasdaqHaltEvent(
                symbol=symbol,
                issue_name=_clean_optional(item.findtext("ndaq:IssueName", namespaces=ns)),
                market=_clean_optional(
                    item.findtext("ndaq:Mkt", namespaces=ns)
                    or item.findtext("ndaq:Market", namespaces=ns)
                ),
                reason_code=_clean_optional(
                    item.findtext("ndaq:ReasonCode", namespaces=ns)
                ),
                halt_timestamp=_parse_feed_datetime(
                    item.findtext("ndaq:HaltDate", namespaces=ns),
                    item.findtext("ndaq:HaltTime", namespaces=ns),
                ),
                resumption_timestamp=_parse_feed_datetime(
                    item.findtext("ndaq:ResumptionDate", namespaces=ns),
                    item.findtext("ndaq:ResumptionTradeTime", namespaces=ns),
                ),
                published_timestamp=_parse_rss_pubdate(item.findtext("pubDate")),
                raw=raw,
            )
        )
    knowledge_ts = pub_ts or utcnow()
    return NasdaqSourceFile(
        source=source,
        url=url,
        knowledge_timestamp=knowledge_ts,
        raw_payload=text,
        raw_payload_hash=stable_hash(text),
        row_count=len(records),
        records=tuple(records),
        footer=None,
    )


def _status_from_directories(
    symbol: str,
    asof: datetime,
    files: Sequence[NasdaqSourceFile],
) -> NasdaqListingStatusResult:
    variants = tuple(symbol_variants(symbol))
    ineligible_match: Optional[Tuple[NasdaqSourceFile, NasdaqDirectoryRecord]] = None
    for file in files:
        for record in file.records:
            if isinstance(record, NasdaqDirectoryRecord) and record.symbol in variants:
                if not _directory_record_is_eligible(record, symbol, record.symbol):
                    if ineligible_match is None:
                        ineligible_match = (file, record)
                    continue
                if not _directory_status_pit_knowable(
                    file.knowledge_timestamp,
                    asof,
                ):
                    return NasdaqListingStatusResult(
                        symbol=str(symbol or "").strip(),
                        normalized_symbol=_normalize_symbol(symbol),
                        status=NasdaqListingStatus.INCONCLUSIVE,
                        asof_timestamp=asof,
                        source_knowledge_timestamp=file.knowledge_timestamp,
                        pit_knowable_at_asof=False,
                        source=file.source,
                        reason="directory_match_not_knowable_at_asof",
                        matched_symbol=record.symbol,
                        raw={"record": record.raw, "footer": file.footer},
                    )
                return NasdaqListingStatusResult(
                    symbol=str(symbol or "").strip(),
                    normalized_symbol=_normalize_symbol(symbol),
                    status=NasdaqListingStatus.LISTED_ACTIVE,
                    asof_timestamp=asof,
                    source_knowledge_timestamp=file.knowledge_timestamp,
                    pit_knowable_at_asof=_directory_status_pit_knowable(
                        file.knowledge_timestamp,
                        asof,
                    ),
                    source=file.source,
                    reason="symbol_present_in_current_directory",
                    matched_symbol=record.symbol,
                    raw={"record": record.raw, "footer": file.footer},
                )
    if ineligible_match is not None:
        file, record = ineligible_match
        return NasdaqListingStatusResult(
            symbol=str(symbol or "").strip(),
            normalized_symbol=_normalize_symbol(symbol),
            status=NasdaqListingStatus.INCONCLUSIVE,
            asof_timestamp=asof,
            source_knowledge_timestamp=file.knowledge_timestamp,
            pit_knowable_at_asof=_directory_status_pit_knowable(
                file.knowledge_timestamp,
                asof,
            ),
            source=file.source,
            reason="directory_record_not_common_stock_listing",
            matched_symbol=record.symbol,
            raw={"record": record.raw, "footer": file.footer},
        )
    knowledge = max(file.knowledge_timestamp for file in files) if files else None
    return NasdaqListingStatusResult(
        symbol=str(symbol or "").strip(),
        normalized_symbol=_normalize_symbol(symbol),
        status=NasdaqListingStatus.INCONCLUSIVE,
        asof_timestamp=asof,
        source_knowledge_timestamp=knowledge,
        pit_knowable_at_asof=False,
        source="current_directory_snapshot",
        reason="symbol_absent_from_current_snapshot",
        raw={"symbol_variants": list(variants)},
    )


def _status_from_adds_deletes(
    symbol: str,
    asof: datetime,
    file: Optional[NasdaqSourceFile],
) -> Optional[NasdaqListingStatusResult]:
    if file is None:
        return None
    variants = set(symbol_variants(symbol))
    asof_day = _et_date(asof)
    future_knowledge_result: Optional[NasdaqListingStatusResult] = None
    for record in file.records:
        if not isinstance(record, NasdaqAddsDeletesRecord):
            continue
        if record.symbol not in variants:
            continue
        if record.effective_date != asof_day:
            continue
        actions = {
            _normalize_symbol(record.nasdaq_action),
            _normalize_symbol(record.bx_action),
            _normalize_symbol(record.psx_action),
        }
        if "DELETE" in actions:
            if file.knowledge_timestamp > asof:
                if future_knowledge_result is None:
                    future_knowledge_result = NasdaqListingStatusResult(
                        symbol=str(symbol or "").strip(),
                        normalized_symbol=_normalize_symbol(symbol),
                        status=NasdaqListingStatus.INCONCLUSIVE,
                        asof_timestamp=asof,
                        source_knowledge_timestamp=file.knowledge_timestamp,
                        pit_knowable_at_asof=False,
                        source=file.source,
                        reason="adds_deletes_delete_not_knowable_at_asof",
                        matched_symbol=record.symbol,
                        raw={"record": record.raw, "footer": file.footer},
                    )
                continue
            return NasdaqListingStatusResult(
                symbol=str(symbol or "").strip(),
                normalized_symbol=_normalize_symbol(symbol),
                status=NasdaqListingStatus.DELISTED,
                asof_timestamp=asof,
                source_knowledge_timestamp=file.knowledge_timestamp,
                pit_knowable_at_asof=file.knowledge_timestamp <= asof,
                source=file.source,
                reason="trading_system_adds_deletes_delete",
                matched_symbol=record.symbol,
                raw={"record": record.raw, "footer": file.footer},
            )
    return future_knowledge_result


def _status_from_halt_events(
    symbol: str,
    asof: datetime,
    file: Optional[NasdaqSourceFile],
) -> Optional[NasdaqListingStatusResult]:
    if file is None:
        return None
    variants = set(symbol_variants(symbol))
    future_knowledge_result: Optional[NasdaqListingStatusResult] = None
    for event in file.records:
        if not isinstance(event, NasdaqHaltEvent):
            continue
        if event.symbol not in variants:
            continue
        if (event.reason_code or "").upper() != REASON_SECURITY_DELETION:
            continue
        if event.halt_timestamp is None or _et_date(event.halt_timestamp) != _et_date(asof):
            continue
        if event.halt_timestamp is not None and event.halt_timestamp > asof:
            continue
        knowledge = event.published_timestamp or file.knowledge_timestamp
        if knowledge > asof:
            if future_knowledge_result is None:
                future_knowledge_result = NasdaqListingStatusResult(
                    symbol=str(symbol or "").strip(),
                    normalized_symbol=_normalize_symbol(symbol),
                    status=NasdaqListingStatus.INCONCLUSIVE,
                    asof_timestamp=asof,
                    source_knowledge_timestamp=knowledge,
                    pit_knowable_at_asof=False,
                    source=file.source,
                    reason="halt_reason_D_not_knowable_at_asof",
                    matched_symbol=event.symbol,
                    raw={"record": event.raw},
                )
            continue
        return NasdaqListingStatusResult(
            symbol=str(symbol or "").strip(),
            normalized_symbol=_normalize_symbol(symbol),
            status=NasdaqListingStatus.DELISTED,
            asof_timestamp=asof,
            source_knowledge_timestamp=knowledge,
            pit_knowable_at_asof=knowledge <= asof,
            source=file.source,
            reason="trade_halt_reason_D_security_deletion",
            matched_symbol=event.symbol,
            raw={"record": event.raw},
        )
    return future_knowledge_result


def _select_delisting_status(
    *statuses: Optional[NasdaqListingStatusResult],
) -> Optional[NasdaqListingStatusResult]:
    inconclusive: Optional[NasdaqListingStatusResult] = None
    for status in statuses:
        if status is None:
            continue
        if status.status is NasdaqListingStatus.DELISTED:
            return status
        if (
            status.status is NasdaqListingStatus.INCONCLUSIVE
            and inconclusive is None
        ):
            inconclusive = status
    return inconclusive


def _status_from_archived_directories(
    symbol: str,
    variants: Sequence[str],
    asof: datetime,
    archive_session: Any,
    snapshots: Sequence[NasdaqListingSnapshot],
) -> NasdaqListingStatusResult:
    snapshot_by_id = {snapshot.snapshot_id: snapshot for snapshot in snapshots}
    rows = (
        archive_session.query(NasdaqListingSnapshotRow, NasdaqListingSnapshot)
        .join(
            NasdaqListingSnapshot,
            NasdaqListingSnapshotRow.snapshot_id == NasdaqListingSnapshot.snapshot_id,
        )
        .filter(
            NasdaqListingSnapshot.snapshot_id.in_(tuple(snapshot_by_id)),
            NasdaqListingSnapshotRow.symbol.in_(tuple(variants)),
        )
        .all()
    )
    ineligible_match: Optional[Tuple[NasdaqListingSnapshotRow, NasdaqListingSnapshot]] = None
    for row, snapshot in rows:
        if not _archive_snapshot_covers_asof(snapshot, asof):
            continue
        if not _archive_directory_row_is_eligible(row, symbol):
            if ineligible_match is None:
                ineligible_match = (row, snapshot)
            continue
        source_ts = _archive_source_timestamp(snapshot.source_knowledge_timestamp)
        pit_knowable = _directory_status_pit_knowable(source_ts, asof)
        if not pit_knowable:
            return NasdaqListingStatusResult(
                symbol=str(symbol or "").strip(),
                normalized_symbol=_normalize_symbol(symbol),
                status=NasdaqListingStatus.INCONCLUSIVE,
                asof_timestamp=asof,
                source_knowledge_timestamp=snapshot.source_knowledge_timestamp,
                pit_knowable_at_asof=False,
                source="nasdaq_self_archive",
                reason="directory_match_not_knowable_at_asof",
                matched_symbol=row.symbol,
                raw=_json_or_none(row.raw_json),
            )
        return NasdaqListingStatusResult(
            symbol=str(symbol or "").strip(),
            normalized_symbol=_normalize_symbol(symbol),
            status=NasdaqListingStatus.LISTED_ACTIVE,
            asof_timestamp=asof,
            source_knowledge_timestamp=snapshot.source_knowledge_timestamp,
            pit_knowable_at_asof=pit_knowable,
            source="nasdaq_self_archive",
            reason="symbol_present_in_archived_directory",
            matched_symbol=row.symbol,
            raw=_json_or_none(row.raw_json),
        )
    if ineligible_match is not None:
        row, snapshot = ineligible_match
        source_ts = _archive_source_timestamp(snapshot.source_knowledge_timestamp)
        return NasdaqListingStatusResult(
            symbol=str(symbol or "").strip(),
            normalized_symbol=_normalize_symbol(symbol),
            status=NasdaqListingStatus.INCONCLUSIVE,
            asof_timestamp=asof,
            source_knowledge_timestamp=snapshot.source_knowledge_timestamp,
            pit_knowable_at_asof=_directory_status_pit_knowable(source_ts, asof),
            source="nasdaq_self_archive",
            reason="directory_record_not_common_stock_listing",
            matched_symbol=row.symbol,
            raw=_json_or_none(row.raw_json),
        )
    knowledge = _latest_snapshot_timestamp(snapshots)
    return NasdaqListingStatusResult(
        symbol=str(symbol or "").strip(),
        normalized_symbol=_normalize_symbol(symbol),
        status=NasdaqListingStatus.INCONCLUSIVE,
        asof_timestamp=asof,
        source_knowledge_timestamp=knowledge,
        pit_knowable_at_asof=False,
        source="nasdaq_self_archive",
        reason="symbol_absent_from_archived_directory",
        raw={
            "symbol_variants": list(variants),
            "safety": "archive absence is inconclusive, never delisted",
        },
    )


def _status_from_archived_delisting_evidence(
    symbol: str,
    asof: datetime,
    archive_session: Any,
) -> Optional[NasdaqListingStatusResult]:
    adds_snapshots = _archive_snapshots_for_et_date(
        archive_session,
        (ADDS_DELETES,),
        asof,
    )
    halt_snapshots = _archive_snapshots_for_et_date(
        archive_session,
        (HALT_RSS,),
        asof,
    )
    return _select_delisting_status(
        _status_from_archived_adds_deletes(
            symbol,
            asof,
            archive_session,
            adds_snapshots,
        ),
        _status_from_archived_halt_events(
            symbol,
            asof,
            archive_session,
            halt_snapshots,
        ),
    )


def _status_from_archived_adds_deletes(
    symbol: str,
    asof: datetime,
    archive_session: Any,
    snapshots: Sequence[NasdaqListingSnapshot],
) -> Optional[NasdaqListingStatusResult]:
    if not snapshots:
        return None
    variants = tuple(symbol_variants(symbol))
    asof_day = _et_date(asof)
    rows = (
        archive_session.query(NasdaqListingSnapshotRow, NasdaqListingSnapshot)
        .join(
            NasdaqListingSnapshot,
            NasdaqListingSnapshotRow.snapshot_id == NasdaqListingSnapshot.snapshot_id,
        )
        .filter(
            NasdaqListingSnapshot.snapshot_id.in_(
                tuple(snapshot.snapshot_id for snapshot in snapshots)
            ),
            NasdaqListingSnapshotRow.symbol.in_(variants),
        )
        .all()
    )
    future_knowledge_result: Optional[NasdaqListingStatusResult] = None
    for row, snapshot in rows:
        if _parse_iso_date(row.effective_date) != asof_day:
            continue
        actions = {
            _normalize_symbol(action)
            for action in str(row.action or "").split("|")
        }
        if "DELETE" not in actions:
            continue
        knowledge = _archive_source_timestamp(snapshot.source_knowledge_timestamp)
        if knowledge is None or knowledge > asof:
            if future_knowledge_result is None:
                future_knowledge_result = NasdaqListingStatusResult(
                    symbol=str(symbol or "").strip(),
                    normalized_symbol=_normalize_symbol(symbol),
                    status=NasdaqListingStatus.INCONCLUSIVE,
                    asof_timestamp=asof,
                    source_knowledge_timestamp=(
                        snapshot.source_knowledge_timestamp
                        if knowledge is None
                        else knowledge
                    ),
                    pit_knowable_at_asof=False,
                    source=row.source_type,
                    reason="adds_deletes_delete_not_knowable_at_asof",
                    matched_symbol=row.symbol,
                    raw=_json_or_none(row.raw_json),
                )
            continue
        return NasdaqListingStatusResult(
            symbol=str(symbol or "").strip(),
            normalized_symbol=_normalize_symbol(symbol),
            status=NasdaqListingStatus.DELISTED,
            asof_timestamp=asof,
            source_knowledge_timestamp=knowledge,
            pit_knowable_at_asof=True,
            source=row.source_type,
            reason="trading_system_adds_deletes_delete",
            matched_symbol=row.symbol,
            raw=_json_or_none(row.raw_json),
        )
    return future_knowledge_result


def _status_from_archived_halt_events(
    symbol: str,
    asof: datetime,
    archive_session: Any,
    snapshots: Sequence[NasdaqListingSnapshot],
) -> Optional[NasdaqListingStatusResult]:
    if not snapshots:
        return None
    variants = tuple(symbol_variants(symbol))
    asof_day = _et_date(asof)
    future_knowledge_result: Optional[NasdaqListingStatusResult] = None
    rows = (
        archive_session.query(NasdaqListingSnapshotRow, NasdaqListingSnapshot)
        .join(
            NasdaqListingSnapshot,
            NasdaqListingSnapshotRow.snapshot_id == NasdaqListingSnapshot.snapshot_id,
        )
        .filter(
            NasdaqListingSnapshot.snapshot_id.in_(
                tuple(snapshot.snapshot_id for snapshot in snapshots)
            ),
            NasdaqListingSnapshotRow.symbol.in_(variants),
        )
        .all()
    )
    for row, snapshot in rows:
        if _normalize_symbol(row.reason_code) != REASON_SECURITY_DELETION:
            continue
        if _parse_iso_date(row.effective_date) != asof_day:
            continue
        knowledge = _archived_halt_row_knowledge(row, snapshot)
        if knowledge is None or knowledge > asof:
            if future_knowledge_result is None:
                future_knowledge_result = NasdaqListingStatusResult(
                    symbol=str(symbol or "").strip(),
                    normalized_symbol=_normalize_symbol(symbol),
                    status=NasdaqListingStatus.INCONCLUSIVE,
                    asof_timestamp=asof,
                    source_knowledge_timestamp=(
                        snapshot.source_knowledge_timestamp
                        if knowledge is None
                        else knowledge
                    ),
                    pit_knowable_at_asof=False,
                    source=row.source_type,
                    reason="halt_reason_D_not_knowable_at_asof",
                    matched_symbol=row.symbol,
                    raw=_json_or_none(row.raw_json),
                )
            continue
        return NasdaqListingStatusResult(
            symbol=str(symbol or "").strip(),
            normalized_symbol=_normalize_symbol(symbol),
            status=NasdaqListingStatus.DELISTED,
            asof_timestamp=asof,
            source_knowledge_timestamp=knowledge,
            pit_knowable_at_asof=True,
            source=row.source_type,
            reason="trade_halt_reason_D_security_deletion",
            matched_symbol=row.symbol,
            raw=_json_or_none(row.raw_json),
        )
    return future_knowledge_result


def _archive_rows(file: NasdaqSourceFile) -> List[NasdaqListingSnapshotRow]:
    rows: List[NasdaqListingSnapshotRow] = []
    for record in file.records:
        if isinstance(record, NasdaqDirectoryRecord):
            rows.append(
                NasdaqListingSnapshotRow(
                    source_type=file.source,
                    symbol=record.symbol,
                    normalized_symbol=_normalize_symbol(record.symbol),
                    security_name=record.security_name,
                    market=record.market,
                    raw_json=json.dumps(record.raw, sort_keys=True, default=str),
                )
            )
        elif isinstance(record, NasdaqAddsDeletesRecord):
            rows.append(
                NasdaqListingSnapshotRow(
                    source_type=file.source,
                    symbol=record.symbol,
                    normalized_symbol=_normalize_symbol(record.symbol),
                    security_name=record.company_name,
                    market=record.primary_listing_market,
                    action="|".join(
                        action or ""
                        for action in (
                            record.nasdaq_action,
                            record.bx_action,
                            record.psx_action,
                        )
                    ),
                    effective_date=(
                        record.effective_date.isoformat()
                        if record.effective_date is not None
                        else None
                    ),
                    raw_json=json.dumps(record.raw, sort_keys=True, default=str),
                )
            )
        elif isinstance(record, NasdaqHaltEvent):
            rows.append(
                NasdaqListingSnapshotRow(
                    source_type=file.source,
                    symbol=record.symbol,
                    normalized_symbol=_normalize_symbol(record.symbol),
                    security_name=record.issue_name,
                    market=record.market,
                    reason_code=record.reason_code,
                    effective_date=(
                        _et_date(record.halt_timestamp).isoformat()
                        if record.halt_timestamp is not None
                        else None
                    ),
                    raw_json=json.dumps(record.raw, sort_keys=True, default=str),
                )
            )
    return rows


def _find_existing_snapshot(session: Any, file: NasdaqSourceFile) -> Any:
    return (
        session.query(NasdaqListingSnapshot)
        .filter(
            NasdaqListingSnapshot.source_type == file.source,
            NasdaqListingSnapshot.source_knowledge_timestamp
            == file.knowledge_timestamp,
            NasdaqListingSnapshot.raw_payload_hash == file.raw_payload_hash,
        )
        .first()
    )


def _pipe_rows(
    source: str,
    lines: Sequence[str],
    expected_header: Sequence[str],
) -> List[Dict[str, str]] | ProviderError:
    header = lines[0].split("|")
    if header != list(expected_header):
        return _parse_error(
            source,
            f"Unexpected Nasdaq {source} header: {header!r}",
        )
    rows: List[Dict[str, str]] = []
    for line in lines[1:-1]:
        if not line.strip():
            continue
        parts = line.split("|")
        if parts == header or parts[:2] == header[:2]:
            return _parse_error(
                source,
                "Duplicate Nasdaq header row in data section",
            )
        if len(parts) != len(header):
            return _parse_error(
                source,
                f"Malformed Nasdaq {source} row with {len(parts)} fields: {line[:120]!r}",
            )
        rows.append(dict(zip(header, parts)))
    return rows


def _parse_footer_timestamp(source: str, footer: str) -> datetime | ProviderError:
    if not footer.startswith(DIRECTORY_FOOTER_PREFIX):
        return _parse_error(source, "Missing Nasdaq File Creation Time trailer")
    stamp = footer[len(DIRECTORY_FOOTER_PREFIX):].split("|", 1)[0].strip()
    try:
        parsed = datetime.strptime(stamp, "%m%d%Y%H:%M")
    except ValueError:
        return _parse_error(source, f"Malformed Nasdaq File Creation Time: {stamp!r}")
    return parsed.replace(tzinfo=NASDAQ_TZ).astimezone(timezone.utc)


def _parse_feed_datetime(date_value: Any, time_value: Any) -> Optional[datetime]:
    date_text = str(date_value or "").strip()
    time_text = str(time_value or "").strip()
    if not date_text or not time_text:
        return None
    time_text = re.sub(r"\s+", "", time_text)
    time_text = time_text.split(".", 1)[0]
    try:
        parsed_date = datetime.strptime(date_text, "%m/%d/%Y").date()
        parsed_time = datetime.strptime(time_text, "%H:%M:%S").time()
    except ValueError:
        return None
    return datetime.combine(parsed_date, parsed_time, tzinfo=NASDAQ_TZ).astimezone(timezone.utc)


def _parse_rss_pubdate(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=NASDAQ_TZ)
    return parsed.astimezone(timezone.utc)


def _parse_us_date(value: Any) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%m/%d/%Y").date()
    except ValueError:
        return None


def _lineage(
    endpoint: str,
    asof: datetime,
    raw_payload_hash: str,
    flags: Optional[Dict[str, Any]] = None,
    *,
    request_ts: Optional[datetime] = None,
) -> LineageMeta:
    request_ts = request_ts or utcnow()
    return LineageMeta(
        provider=PROVIDER,
        endpoint=endpoint,
        request_timestamp=request_ts,
        asof_timestamp=asof,
        raw_payload_hash=raw_payload_hash,
        freshness_seconds=(utcnow() - request_ts).total_seconds(),
        source_authority=SOURCE_AUTHORITY,
        data_quality_flags=flags or {},
    )


def _combined_lineage(
    endpoint: str,
    asof: datetime,
    lineages: Sequence[LineageMeta],
    flags: Optional[Dict[str, Any]] = None,
) -> LineageMeta:
    component_hashes = [lineage.raw_payload_hash for lineage in lineages]
    merged_flags = {"component_payload_hashes": component_hashes}
    for lineage in lineages:
        merged_flags.update(lineage.data_quality_flags or {})
    merged_flags.update(flags or {})
    return _lineage(
        endpoint,
        asof,
        stable_hash(component_hashes),
        merged_flags,
        request_ts=min(
            (lineage.request_timestamp for lineage in lineages),
            default=utcnow(),
        ),
    )


def _error_response(
    *,
    endpoint: str,
    asof: datetime,
    error_type: str,
    message: str,
    retryable: bool,
    status_code: Optional[int] = None,
    request_ts: Optional[datetime] = None,
) -> AdapterResponse[Any]:
    return AdapterResponse(
        data=None,
        lineage=_lineage(endpoint, asof, "", {}, request_ts=request_ts),
        error=ProviderError(
            provider=PROVIDER,
            endpoint=endpoint,
            status_code=status_code,
            error_type=error_type,
            message=message,
            retryable=retryable,
        ),
    )


def _unavailable_from_error(
    symbol: str,
    asof: datetime,
    *,
    endpoint: str,
    error: Optional[ProviderError],
    lineage: LineageMeta,
) -> AdapterResponse[NasdaqListingStatusResult]:
    message = error.message if error is not None else "Nasdaq listing-status unavailable"
    result = NasdaqListingStatusResult(
        symbol=str(symbol or "").strip(),
        normalized_symbol=_normalize_symbol(symbol),
        status=NasdaqListingStatus.UNAVAILABLE,
        asof_timestamp=asof,
        source_knowledge_timestamp=None,
        pit_knowable_at_asof=False,
        source=endpoint,
        reason=message,
        raw={"error_type": error.error_type if error else None},
    )
    return AdapterResponse(data=result, lineage=lineage, error=error)


def _parse_error(source: str, message: str) -> ProviderError:
    return ProviderError(
        provider=PROVIDER,
        endpoint=source,
        status_code=None,
        error_type="parse",
        message=message,
        retryable=False,
    )


def _clean_lines(text: str) -> List[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip("\n").split("\n")


def _clean_xml_text(text: str) -> str:
    if text.startswith("ï»¿"):
        try:
            text = text.encode("latin1").decode("utf-8-sig")
        except UnicodeError:
            text = text[3:]
    return text.lstrip("\ufeff").lstrip()


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _unit_compact_symbol(symbol: str) -> Optional[str]:
    text = _normalize_symbol(symbol)
    if UNIT_SUFFIX_RE.match(text):
        return re.sub(r"[.\-/]", "", text)
    return None


def _directory_record_is_eligible(
    record: NasdaqDirectoryRecord,
    query_symbol: Any,
    matched_symbol: str,
) -> bool:
    if _normalize_symbol(record.test_issue) == "Y":
        return False
    if _normalize_symbol(record.etf) == "Y":
        return False
    security_name = f" {record.security_name.upper()} "
    punctuation_variants = _punctuation_preserving_symbol_variants(
        _normalize_symbol(query_symbol)
    )
    unit_compact = _unit_compact_symbol(_normalize_symbol(query_symbol))
    if (
        unit_compact is not None
        and matched_symbol == unit_compact
        and matched_symbol not in punctuation_variants
    ):
        return _security_name_is_unit(security_name)
    if _security_name_is_limited_partner_equity(security_name):
        return True
    if not _security_name_has_non_common_descriptor(security_name):
        return True
    if (
        unit_compact is not None
        and matched_symbol == unit_compact
        and _security_name_is_unit(security_name)
    ):
        return True
    return False


def _security_name_has_non_common_descriptor(security_name: str) -> bool:
    return any(
        pattern.search(security_name)
        for pattern in NON_COMMON_SECURITY_NAME_PATTERNS
    )


def _security_name_is_unit(security_name: str) -> bool:
    return UNIT_SECURITY_NAME_RE.search(security_name) is not None


def _security_name_is_limited_partner_equity(security_name: str) -> bool:
    return LIMITED_PARTNER_EQUITY_RE.search(security_name) is not None


def _archive_directory_row_is_eligible(
    row: NasdaqListingSnapshotRow,
    query_symbol: Any,
) -> bool:
    raw = _json_or_none(row.raw_json)
    if (
        raw is None
        or raw.get("Test Issue") is None
        or raw.get("ETF") is None
    ):
        return False
    record = NasdaqDirectoryRecord(
        source=row.source_type,
        symbol=row.symbol,
        security_name=row.security_name or "",
        market=row.market,
        test_issue=raw.get("Test Issue"),
        etf=raw.get("ETF"),
        raw=raw,
    )
    return _directory_record_is_eligible(record, query_symbol, row.symbol)


def _latest_archive_snapshots_by_source(
    session: Any,
    source_types: Sequence[str],
    asof: datetime,
) -> Tuple[Dict[str, NasdaqListingSnapshot], Optional[NasdaqListingSnapshot]]:
    asof_day = _et_date(asof)
    snapshots = (
        session.query(NasdaqListingSnapshot)
        .filter(NasdaqListingSnapshot.source_type.in_(tuple(source_types)))
        .order_by(NasdaqListingSnapshot.source_knowledge_timestamp.desc())
        .all()
    )
    selected: Dict[str, NasdaqListingSnapshot] = {}
    latest_prior: Optional[NasdaqListingSnapshot] = None
    for snapshot in snapshots:
        source_ts = _archive_source_timestamp(snapshot.source_knowledge_timestamp)
        if source_ts is None:
            continue
        source_day = _et_date(source_ts)
        if source_day == asof_day and snapshot.source_type not in selected:
            selected[snapshot.source_type] = snapshot
            continue
        if source_day < asof_day and latest_prior is None:
            latest_prior = snapshot
    return selected, latest_prior


def _archive_snapshots_for_et_date(
    session: Any,
    source_types: Sequence[str],
    asof: datetime,
) -> Tuple[NasdaqListingSnapshot, ...]:
    asof_day = _et_date(asof)
    snapshots = (
        session.query(NasdaqListingSnapshot)
        .filter(NasdaqListingSnapshot.source_type.in_(tuple(source_types)))
        .order_by(NasdaqListingSnapshot.source_knowledge_timestamp.desc())
        .all()
    )
    same_day: List[NasdaqListingSnapshot] = []
    for snapshot in snapshots:
        source_ts = _archive_source_timestamp(snapshot.source_knowledge_timestamp)
        if source_ts is not None and _et_date(source_ts) == asof_day:
            same_day.append(snapshot)
    return tuple(same_day)


def _archive_required_source_coverage(
    session: Any,
    asof: datetime,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    snapshots = _archive_snapshots_for_et_date(
        session,
        ARCHIVE_REQUIRED_SOURCE_TYPES,
        asof,
    )
    captured_set = {
        snapshot.source_type
        for snapshot in snapshots
        if (getattr(snapshot, "parse_status", None) or "parsed") == "parsed"
    }
    captured = tuple(
        source for source in ARCHIVE_REQUIRED_SOURCE_TYPES if source in captured_set
    )
    missing = tuple(
        source for source in ARCHIVE_REQUIRED_SOURCE_TYPES if source not in captured_set
    )
    return captured, missing


def _archive_source_coverage_incomplete_status(
    symbol: str,
    asof: datetime,
    captured_sources: Sequence[str],
    missing_sources: Sequence[str],
) -> NasdaqListingStatusResult:
    return NasdaqListingStatusResult(
        symbol=str(symbol or "").strip(),
        normalized_symbol=_normalize_symbol(symbol),
        status=NasdaqListingStatus.INCONCLUSIVE,
        asof_timestamp=asof,
        source_knowledge_timestamp=None,
        pit_knowable_at_asof=False,
        source="nasdaq_self_archive",
        reason="archive_source_coverage_incomplete",
        raw={
            "required_sources": list(ARCHIVE_REQUIRED_SOURCE_TYPES),
            "captured_sources": list(captured_sources),
            "missing_sources": list(missing_sources),
            "safety": "archive LISTED_ACTIVE requires complete same-ET-date source coverage",
        },
    )


def _latest_snapshot_timestamp(
    snapshots: Sequence[NasdaqListingSnapshot],
) -> Optional[datetime]:
    timestamps = [
        _archive_source_timestamp(snapshot.source_knowledge_timestamp)
        for snapshot in snapshots
    ]
    known = [timestamp for timestamp in timestamps if timestamp is not None]
    if not known:
        return None
    return max(known)


def _archive_snapshot_covers_asof(
    snapshot: NasdaqListingSnapshot,
    asof: datetime,
) -> bool:
    source_ts = _archive_source_timestamp(snapshot.source_knowledge_timestamp)
    if source_ts is None:
        return False
    return _et_date(source_ts) == _et_date(asof)


def _directory_status_pit_knowable(
    source_ts: Optional[datetime],
    asof: datetime,
) -> bool:
    if source_ts is None:
        return False
    if source_ts <= asof:
        return True
    if _et_date(source_ts) != _et_date(asof):
        return False
    return _asof_at_or_after_session_close(asof)


def _asof_at_or_after_session_close(asof: datetime) -> bool:
    try:
        session_close = us_equity_session_close_timestamp(_et_date(asof))
    except ValueError:
        return False
    return asof.astimezone(timezone.utc) >= session_close


def _archive_source_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_iso_date(value: Any) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _archived_halt_row_knowledge(
    row: NasdaqListingSnapshotRow,
    snapshot: NasdaqListingSnapshot,
) -> Optional[datetime]:
    raw = _json_or_none(row.raw_json) or {}
    published = _parse_rss_pubdate(raw.get("pubDate"))
    if published is not None:
        return published
    return _archive_source_timestamp(snapshot.source_knowledge_timestamp)


def _clean_optional(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _et_date(value: datetime) -> date:
    return value.astimezone(NASDAQ_TZ).date()


def _json_or_none(value: Optional[str]) -> Optional[Dict[str, Any]]:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
