"""M2 insider-cluster assembler and CMP classifier.

This module is upstream of ``patterns.m2``. It ingests normalized Form 4
transactions, classifies insiders with the CMP annual sticky rule, builds
accession-backed clusters, and emits detector-ready ``PatternInput`` rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from alpha.assembly.framework import (
    AssembledField,
    AssemblyDiagnostic,
    FieldPresence,
    PatternAssemblyResult,
    build_pattern_input,
    validate_assembled_fields,
)
from alpha.data.contracts import stable_hash
from alpha.market_calendar import (
    EASTERN_TZ,
    is_us_equity_session,
    next_us_equity_session,
    us_equity_session_open_timestamp,
)
from alpha.patterns.contracts import (
    PatternId,
    PatternInput,
    PatternTrack,
    RouteClass,
    ThesisCategory,
)
from alpha.patterns.m2 import CLUSTER_WINDOW_DAYS, M2Detector

PATTERN_ID = PatternId.M2
SHADOW_PATTERN_ID = "M2U"
CLASSIFICATION_ROUTINE = "routine"
CLASSIFICATION_OPPORTUNISTIC = "opportunistic"
CLASSIFICATION_UNCLASSIFIABLE = "unclassifiable"
OPEN_MARKET_PURCHASE_CODE = "P"
OPEN_MARKET_SALE_CODE = "S"
MARKET_CAP_SIZE_ANCHOR = 0.001
SIZE_WEIGHT_MIN = 0.25
SIZE_WEIGHT_MAX = 3.0
LEGACY_ABS_WEIGHT_MIN = 0.5
LEGACY_ABS_WEIGHT_MAX = 3.0
LOCALITY_MATCH_WEIGHT = 1.25
LOCALITY_DEFAULT_WEIGHT = 1.0
_NAME_NORMALIZER = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ResolvedInsiderIdentity:
    insider_id: str
    cik: Optional[str]
    method: str
    confidence: float
    cik_backed: bool


@dataclass(frozen=True)
class Form4Clock:
    first_tradable_session: date
    public_timestamp: Optional[datetime]
    clock_quality: str


@dataclass(frozen=True)
class TradeSizeWeights:
    production_weight: float
    shadow_abs: float
    shadow_own_history: Optional[float]
    used_market_cap_relative: bool


@dataclass
class M2TransactionEvidence:
    transaction_id: str
    ticker: str
    source_authority: str
    insider_id: str
    insider_cik: Optional[str]
    insider_name: Optional[str]
    identity_resolution_method: str
    identity_resolution_confidence: float
    filing_accession_number: Optional[str]
    filing_form: Optional[str]
    filing_date: Optional[str]
    filing_accepted_at: Optional[datetime]
    filing_detected_at: Optional[datetime]
    first_tradable_session: Optional[str]
    clock_quality: str
    transaction_date: Optional[str]
    transaction_code: Optional[str]
    acquired_disposed_code: Optional[str]
    transaction_shares: Optional[float]
    transaction_price_per_share: Optional[float]
    purchase_notional_usd: Optional[float]
    market_cap_usd: Optional[float]
    issuer_cik: Optional[str] = None
    issuer_name: Optional[str] = None
    issuer_state: Optional[str] = None
    insider_state: Optional[str] = None
    insider_roles: Dict[str, Any] = field(default_factory=dict)
    ownership_type: Optional[str] = None
    is_open_market_purchase: bool = False
    is_buy: bool = False
    is_sell: bool = False
    is_10b5_1: Optional[bool] = None
    sec_fmp_mismatch: bool = False
    data_lineage_ids: List[str] = field(default_factory=list)
    lineage_hashes: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def cik_backed(self) -> bool:
        return self.insider_cik is not None and self.identity_resolution_method != "normalized_name"


@dataclass(frozen=True)
class CMPClassification:
    insider_id: str
    calendar_year: int
    classification: str
    routine_month: Optional[int]
    prior_year_count: int
    data_cutoff_at: datetime
    basis: Dict[str, Any]


class M2UDetector(M2Detector):
    """Shadow unclassifiable-cluster detector using M2 math under pattern_id M2U."""

    pattern_id = SHADOW_PATTERN_ID
    version = "1.0-shadow"
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.RIGHT_TAIL_CONVEX
    route_class = RouteClass.A


def normalize_cik(value: Any) -> Optional[str]:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if not digits:
        return None
    return digits[-10:].zfill(10)


def normalize_insider_name(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    normalized = _NAME_NORMALIZER.sub(" ", text).strip()
    return normalized or None


def resolve_insider_identity(
    *,
    sec_owner_cik: Any = None,
    fmp_owner_cik: Any = None,
    accession_owner_cik: Any = None,
    owner_name: Any = None,
) -> ResolvedInsiderIdentity:
    """Resolve one insider identity for both classifier and cluster gate."""

    sec_cik = normalize_cik(sec_owner_cik)
    fmp_cik = normalize_cik(fmp_owner_cik)
    accession_cik = normalize_cik(accession_owner_cik)
    if sec_cik:
        return ResolvedInsiderIdentity(
            insider_id=f"cik:{sec_cik}",
            cik=sec_cik,
            method="sec_reporting_owner_cik",
            confidence=1.0,
            cik_backed=True,
        )
    if fmp_cik and accession_cik and fmp_cik == accession_cik:
        return ResolvedInsiderIdentity(
            insider_id=f"cik:{fmp_cik}",
            cik=fmp_cik,
            method="fmp_cik_accession_match",
            confidence=0.95,
            cik_backed=True,
        )
    normalized_name = normalize_insider_name(owner_name)
    if normalized_name:
        return ResolvedInsiderIdentity(
            insider_id=f"name:{stable_hash(normalized_name)}",
            cik=None,
            method="normalized_name",
            confidence=0.55,
            cik_backed=False,
        )
    return ResolvedInsiderIdentity(
        insider_id="unresolved:missing_owner",
        cik=None,
        method="unresolved",
        confidence=0.0,
        cik_backed=False,
    )


def first_tradable_session_after_publication(
    *,
    filing_accepted_at: Optional[datetime],
    filing_detected_at: Optional[datetime],
    filing_date: Optional[date],
) -> Form4Clock:
    """First regular-session open strictly after the public Form 4 timestamp."""

    accepted = _ensure_aware(filing_accepted_at)
    detected = _ensure_aware(filing_detected_at)
    timestamps = [value for value in (accepted, detected) if value is not None]
    if timestamps:
        public_ts = max(timestamps)
        session = _first_session_open_after(public_ts)
        if accepted is not None and detected is not None:
            quality = "accepted_detected"
        elif accepted is not None:
            quality = "accepted_only"
        else:
            quality = "detected_only"
        return Form4Clock(
            first_tradable_session=session,
            public_timestamp=public_ts,
            clock_quality=quality,
        )
    if filing_date is None:
        raise ValueError("filing_date is required when acceptance/detection timestamps are missing")
    session = next_us_equity_session(filing_date + timedelta(days=1))
    return Form4Clock(
        first_tradable_session=session,
        public_timestamp=None,
        clock_quality="filing_date_only",
    )


def trade_size_weights(
    *,
    purchase_notional_usd: Optional[float],
    market_cap_usd: Optional[float],
    prior_purchase_notionals: Optional[Sequence[float]] = None,
) -> Optional[TradeSizeWeights]:
    notional = _finite_positive(purchase_notional_usd)
    if notional is None:
        return None
    shadow_abs = _clip(
        math.log1p(notional / 10_000.0),
        LEGACY_ABS_WEIGHT_MIN,
        LEGACY_ABS_WEIGHT_MAX,
    )
    market_cap = _finite_positive(market_cap_usd)
    if market_cap is not None:
        production = _clip(
            math.log1p((notional / market_cap) / MARKET_CAP_SIZE_ANCHOR),
            SIZE_WEIGHT_MIN,
            SIZE_WEIGHT_MAX,
        )
        used_market_cap = True
    else:
        production = shadow_abs
        used_market_cap = False
    prior = [
        float(value)
        for value in (prior_purchase_notionals or [])
        if _finite_positive(value) is not None
    ]
    shadow_own = None
    if prior:
        median = sorted(prior)[len(prior) // 2]
        if median > 0:
            shadow_own = _clip(math.log1p(notional / median), SIZE_WEIGHT_MIN, SIZE_WEIGHT_MAX)
    return TradeSizeWeights(
        production_weight=production,
        shadow_abs=shadow_abs,
        shadow_own_history=shadow_own,
        used_market_cap_relative=used_market_cap,
    )


def classify_cmp_insider(
    transactions: Sequence[Any],
    *,
    insider_id: str,
    calendar_year: int,
) -> CMPClassification:
    """CMP annual sticky classifier using only accepted data before Jan 1 Y."""

    cutoff = datetime.combine(
        date(calendar_year, 1, 1),
        time.min,
        EASTERN_TZ,
    ).astimezone(timezone.utc)
    eligible: List[Any] = []
    same_year_rejected = 0
    for tx in transactions:
        if _tx_attr(tx, "insider_id") != insider_id:
            continue
        accepted = _transaction_knowledge_timestamp(tx)
        if accepted is None or accepted >= cutoff:
            same_year_rejected += 1
            continue
        if _parse_date(_tx_attr(tx, "transaction_date")) is None:
            continue
        eligible.append(tx)

    months_by_year: Dict[int, set[int]] = {}
    for tx in eligible:
        tx_date = _parse_date(_tx_attr(tx, "transaction_date"))
        if tx_date is None or tx_date.year >= calendar_year:
            continue
        months_by_year.setdefault(tx_date.year, set()).add(tx_date.month)
    prior_years = sorted(months_by_year)
    prior_year_count = len(prior_years)
    routine_month = None
    for month in range(1, 13):
        if all(month in months_by_year.get(year, set()) for year in (
            calendar_year - 1,
            calendar_year - 2,
            calendar_year - 3,
        )):
            routine_month = month
            break
    if prior_year_count < 3:
        classification = CLASSIFICATION_UNCLASSIFIABLE
    elif routine_month is not None:
        classification = CLASSIFICATION_ROUTINE
    else:
        classification = CLASSIFICATION_OPPORTUNISTIC
    return CMPClassification(
        insider_id=insider_id,
        calendar_year=calendar_year,
        classification=classification,
        routine_month=routine_month,
        prior_year_count=prior_year_count,
        data_cutoff_at=cutoff,
        basis={
            "prior_years": prior_years,
            "months_by_year": {str(year): sorted(months) for year, months in months_by_year.items()},
            "same_year_or_unknown_acceptance_rejected_count": same_year_rejected,
            "cmp_rule": "routine_if_same_calendar_month_traded_in_each_of_y_minus_1_y_minus_2_y_minus_3",
        },
    )


def assemble_m2_daily(
    *,
    snapshots: List[Any],
    transactions: Sequence[Any],
    cutoff_timestamp: datetime,
    universe_cutoff_timestamp: Optional[datetime],
    decision_date: str,
    evidence_session_date: str,
    next_execution_session: str,
    source_provider: str = "SEC_EDGAR",
    lineage_ids_by_ticker: Optional[Dict[str, List[str]]] = None,
    lineage_hashes_by_ticker: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, PatternAssemblyResult]:
    """Assemble production M2 and M2U shadow detector inputs."""

    lineage_ids_by_ticker = lineage_ids_by_ticker or {}
    lineage_hashes_by_ticker = lineage_hashes_by_ticker or {}
    resolved_universe_cutoff = universe_cutoff_timestamp or cutoff_timestamp
    next_execution_day = date.fromisoformat(next_execution_session)
    cluster_start = next_execution_day - timedelta(days=CLUSTER_WINDOW_DAYS)
    signal_year = next_execution_day.year
    snapshot_by_ticker = {
        str(_snap_attr(snap, "ticker") or "").upper(): snap
        for snap in snapshots
    }
    classifications = {
        (insider_id, signal_year): classify_cmp_insider(
            transactions,
            insider_id=insider_id,
            calendar_year=signal_year,
        )
        for insider_id in sorted({
            str(_tx_attr(tx, "insider_id") or "")
            for tx in transactions
            if _tx_attr(tx, "insider_id")
        })
    }
    results = {
        PATTERN_ID: PatternAssemblyResult(pattern_id=PATTERN_ID),
        SHADOW_PATTERN_ID: PatternAssemblyResult(pattern_id=SHADOW_PATTERN_ID),
    }
    recent_by_ticker = _recent_cluster_candidates(
        transactions,
        snapshot_by_ticker=snapshot_by_ticker,
        cluster_start=cluster_start,
        next_execution_day=next_execution_day,
    )

    for ticker, rows in sorted(recent_by_ticker.items()):
        snapshot = snapshot_by_ticker[ticker]
        opp_rows: List[Any] = []
        unclassifiable_rows: List[Any] = []
        routine_buyers: set[str] = set()
        unclassifiable_buyers: set[str] = set()
        for tx in rows:
            classification = classifications.get((_tx_attr(tx, "insider_id"), signal_year))
            if classification is None:
                continue
            if classification.classification == CLASSIFICATION_OPPORTUNISTIC:
                opp_rows.append(tx)
            elif classification.classification == CLASSIFICATION_UNCLASSIFIABLE:
                unclassifiable_rows.append(tx)
                unclassifiable_buyers.add(str(_tx_attr(tx, "insider_id")))
            elif classification.classification == CLASSIFICATION_ROUTINE:
                routine_buyers.add(str(_tx_attr(tx, "insider_id")))

        _append_cluster_input(
            results[PATTERN_ID],
            pattern_id=PATTERN_ID,
            ticker=ticker,
            snapshot=snapshot,
            rows=opp_rows,
            all_window_rows=rows,
            classifications=classifications,
            signal_year=signal_year,
            next_execution_day=next_execution_day,
            cutoff_timestamp=cutoff_timestamp,
            resolved_universe_cutoff=resolved_universe_cutoff,
            decision_date=decision_date,
            evidence_session_date=evidence_session_date,
            next_execution_session=next_execution_session,
            source_provider=source_provider,
            lineage_ids=lineage_ids_by_ticker.get(ticker, []),
            lineage_hashes=lineage_hashes_by_ticker.get(ticker, []),
            routine_buyer_count=len(routine_buyers),
            unclassifiable_buyer_count=len(unclassifiable_buyers),
        )
        _append_cluster_input(
            results[SHADOW_PATTERN_ID],
            pattern_id=SHADOW_PATTERN_ID,
            ticker=ticker,
            snapshot=snapshot,
            rows=unclassifiable_rows,
            all_window_rows=rows,
            classifications=classifications,
            signal_year=signal_year,
            next_execution_day=next_execution_day,
            cutoff_timestamp=cutoff_timestamp,
            resolved_universe_cutoff=resolved_universe_cutoff,
            decision_date=decision_date,
            evidence_session_date=evidence_session_date,
            next_execution_session=next_execution_session,
            source_provider=source_provider,
            lineage_ids=lineage_ids_by_ticker.get(ticker, []),
            lineage_hashes=lineage_hashes_by_ticker.get(ticker, []),
            routine_buyer_count=len(routine_buyers),
            unclassifiable_buyer_count=len(unclassifiable_buyers),
        )

    return results


def cluster_member_rows(inp: PatternInput, pattern_id: str) -> List[Dict[str, Any]]:
    """Return persisted m2_cluster_members payloads from an assembled input."""

    rows = inp.market_data.get("m2_cluster_members")
    if not isinstance(rows, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        result.append({
            "pattern_id": pattern_id,
            "m2_cluster_id": inp.market_data.get("m2_cluster_id"),
            "m2_cluster_signature_hash": inp.market_data.get("m2_cluster_signature_hash"),
            "ticker": inp.ticker,
            "transaction_id": item.get("transaction_id"),
            "filing_accession_number": item.get("filing_accession_number"),
            "insider_id": item.get("insider_id"),
            "insider_cik": item.get("insider_cik"),
            "classification": item.get("classification"),
            "first_tradable_session": item.get("first_tradable_session"),
        })
    return result


def _append_cluster_input(
    result: PatternAssemblyResult,
    *,
    pattern_id: str,
    ticker: str,
    snapshot: Any,
    rows: List[Any],
    all_window_rows: List[Any],
    classifications: Dict[Tuple[str, int], CMPClassification],
    signal_year: int,
    next_execution_day: date,
    cutoff_timestamp: datetime,
    resolved_universe_cutoff: datetime,
    decision_date: str,
    evidence_session_date: str,
    next_execution_session: str,
    source_provider: str,
    lineage_ids: List[str],
    lineage_hashes: List[str],
    routine_buyer_count: int,
    unclassifiable_buyer_count: int,
) -> None:
    distinct_buyers = _distinct_cik_backed_buyers(rows)
    if len(distinct_buyers) < 2:
        result.insufficient_count += 1
        result.diagnostics.append(AssemblyDiagnostic(
            ticker=ticker,
            pattern_id=pattern_id,
            diagnostic_type="insufficient_cik_backed_buyers",
            detail=f"distinct_cik_backed_buyers={len(distinct_buyers)}",
        ))
        return
    if _name_only_could_collapse(rows):
        result.rejected_count += 1
        result.diagnostics.append(AssemblyDiagnostic(
            ticker=ticker,
            pattern_id=pattern_id,
            diagnostic_type="name_only_cluster_refused",
        ))
        return
    accessions = sorted({
        str(_tx_attr(tx, "filing_accession_number"))
        for tx in rows
        if str(_tx_attr(tx, "filing_accession_number") or "").strip()
    })
    if len(accessions) < len(rows):
        result.rejected_count += 1
        result.diagnostics.append(AssemblyDiagnostic(
            ticker=ticker,
            pattern_id=pattern_id,
            diagnostic_type="missing_accession_proof",
        ))
        return

    latest_session = max(
        date.fromisoformat(str(_tx_attr(tx, "first_tradable_session")))
        for tx in rows
    )
    days_since = (next_execution_day - latest_session).days
    if days_since < 0:
        result.rejected_count += 1
        result.diagnostics.append(AssemblyDiagnostic(
            ticker=ticker,
            pattern_id=pattern_id,
            diagnostic_type="future_first_tradable_session",
        ))
        return
    signature = stable_hash({
        "ticker": ticker,
        "sec_accession_numbers": accessions,
    })
    member_rows = _cluster_member_payloads(rows, classifications, signal_year)
    intensity_rows = _representative_rows_by_insider(rows)
    size_locality_weights = [
        _size_locality_weight(tx, rows)
        for tx in intensity_rows
    ]
    size_locality_weights = [value for value in size_locality_weights if value is not None]
    if not size_locality_weights:
        result.rejected_count += 1
        result.diagnostics.append(AssemblyDiagnostic(
            ticker=ticker,
            pattern_id=pattern_id,
            diagnostic_type="missing_trade_size_weight",
        ))
        return
    mean_intensity = sum(size_locality_weights) / len(size_locality_weights)
    sell_cluster_present = any(
        _tx_attr(tx, "is_sell") is True
        or str(_tx_attr(tx, "transaction_code") or "").strip().upper() == OPEN_MARKET_SALE_CODE
        for tx in all_window_rows
    )
    source_authority = _cluster_source_authority(rows)
    classifications_json = [
        classifications[(_tx_attr(tx, "insider_id"), signal_year)].basis
        | {
            "insider_id": _tx_attr(tx, "insider_id"),
            "classification": classifications[(_tx_attr(tx, "insider_id"), signal_year)].classification,
        }
        for tx in rows
        if (_tx_attr(tx, "insider_id"), signal_year) in classifications
    ]

    fields = [
        _field("n_distinct_opp_buyers_30d", len(distinct_buyers), cutoff_timestamp, source_provider),
        _field("days_since_last_opp_buy_filing_detected", days_since, cutoff_timestamp, source_provider),
        _field("mean_trade_size_weight", mean_intensity, cutoff_timestamp, source_provider),
        _field("mean_locality_weight", 1.0, cutoff_timestamp, source_provider),
        _field("mean_trade_intensity_weight", mean_intensity, cutoff_timestamp, source_provider),
        _field("cluster_window_days", CLUSTER_WINDOW_DAYS, cutoff_timestamp, source_provider),
        _field("source_authority", source_authority, cutoff_timestamp, source_provider),
        _field("sec_accession_numbers", accessions, cutoff_timestamp, source_provider),
        _field("m2_cluster_id", signature, cutoff_timestamp, source_provider),
        _field("m2_cluster_signature_hash", signature, cutoff_timestamp, source_provider),
        _field("m2_cluster_members", member_rows, cutoff_timestamp, source_provider),
        _field("identity_resolution_method", "cik_backed_cluster", cutoff_timestamp, source_provider),
        _field("identity_resolution_confidence", min(float(_tx_attr(tx, "identity_resolution_confidence") or 0.0) for tx in rows), cutoff_timestamp, source_provider),
        _field("clock_quality", _clock_quality(rows), cutoff_timestamp, source_provider),
        _field("clock_anchor", "first_regular_open_after_max_accepted_detected", cutoff_timestamp, source_provider),
        _field("source_features_classifications", classifications_json, cutoff_timestamp, source_provider),
        _field("n_routine_only_buyers_30d", routine_buyer_count, cutoff_timestamp, source_provider),
        _field("n_unclassifiable_only_buyers_30d", unclassifiable_buyer_count, cutoff_timestamp, source_provider),
        _field("routine_trades_30d", routine_buyer_count, cutoff_timestamp, source_provider),
        _field("unclassifiable_buyers_30d", unclassifiable_buyer_count, cutoff_timestamp, source_provider),
        _field("opportunistic_sell_cluster_30d", sell_cluster_present, cutoff_timestamp, source_provider),
        _field("opp_sell_cluster_present", sell_cluster_present, cutoff_timestamp, source_provider),
        _field("last_opp_buy_filing_detected_at", latest_session.isoformat(), cutoff_timestamp, source_provider),
        _field("market_cap_usd", _snap_attr(snapshot, "market_cap"), resolved_universe_cutoff, source_provider),
        _field("price", _snap_attr(snapshot, "price"), resolved_universe_cutoff, source_provider),
        _field("price_at_signal", _snap_attr(snapshot, "price"), resolved_universe_cutoff, source_provider),
        _field("primary_exchange", _snap_attr(snapshot, "primary_exchange"), resolved_universe_cutoff, source_provider),
        _field("security_type", _snap_attr(snapshot, "security_type"), resolved_universe_cutoff, source_provider),
        _field("operating_universe_inclusion", _snap_attr(snapshot, "operating_universe_inclusion"), resolved_universe_cutoff, source_provider),
        _field("liquidity_score", _snap_attr(snapshot, "liquidity_score"), resolved_universe_cutoff, source_provider),
        _field("hazard_score_at_signal", _snap_attr(snapshot, "hazard_score"), resolved_universe_cutoff, source_provider),
        _field("market_data_status", "current", cutoff_timestamp, source_provider),
        _field("halt_status", "clear", cutoff_timestamp, source_provider),
        _field("corporate_action_filter_passed", True, cutoff_timestamp, source_provider),
        _field("trading_date", decision_date, cutoff_timestamp, source_provider),
        _field("evidence_session_date", evidence_session_date, cutoff_timestamp, source_provider),
        _field("next_execution_session", next_execution_session, cutoff_timestamp, source_provider),
        _field("asof_ceiling_timestamp", cutoff_timestamp.isoformat(), cutoff_timestamp, source_provider),
        _field("shadow_pattern_id", pattern_id if pattern_id != PATTERN_ID else None, cutoff_timestamp, source_provider),
    ]
    validated, rejected = validate_assembled_fields(fields, cutoff_timestamp)
    result.rejected_fields.extend(rejected)
    required = (
        "n_distinct_opp_buyers_30d",
        "days_since_last_opp_buy_filing_detected",
        "mean_trade_size_weight",
        "mean_locality_weight",
        "cluster_window_days",
        "source_authority",
        "sec_accession_numbers",
        "m2_cluster_signature_hash",
        "operating_universe_inclusion",
        "next_execution_session",
    )
    missing = [key for key in required if key not in validated]
    if missing:
        result.rejected_count += 1
        result.diagnostics.append(AssemblyDiagnostic(
            ticker=ticker,
            pattern_id=pattern_id,
            diagnostic_type="missing_m2_fields",
            detail=",".join(missing),
        ))
        return
    snap_lineage = _snap_attr(snapshot, "source_lineage_hash")
    input_lineage_hashes = list(lineage_hashes)
    if snap_lineage and snap_lineage not in input_lineage_hashes:
        input_lineage_hashes.append(snap_lineage)
    for tx in rows:
        for lineage_hash in _tx_attr(tx, "lineage_hashes") or []:
            if lineage_hash and lineage_hash not in input_lineage_hashes:
                input_lineage_hashes.append(lineage_hash)
    input_lineage_ids = list(lineage_ids)
    for tx in rows:
        for lineage_id in _tx_attr(tx, "data_lineage_ids") or []:
            if lineage_id and lineage_id not in input_lineage_ids:
                input_lineage_ids.append(lineage_id)
    inp = build_pattern_input(
        ticker=ticker,
        pattern_id=pattern_id,
        asof_timestamp=cutoff_timestamp,
        validated_fields=validated,
        lineage_ids=input_lineage_ids,
        lineage_hashes=input_lineage_hashes,
        universe_snapshot_id=_snap_attr(snapshot, "universe_snapshot_id"),
    )
    result.inputs.append(inp)
    result.assembled_count += 1


def _recent_cluster_candidates(
    transactions: Sequence[Any],
    *,
    snapshot_by_ticker: Dict[str, Any],
    cluster_start: date,
    next_execution_day: date,
) -> Dict[str, List[Any]]:
    by_ticker: Dict[str, List[Any]] = {}
    for tx in transactions:
        ticker = str(_tx_attr(tx, "ticker") or "").upper()
        if ticker not in snapshot_by_ticker:
            continue
        if _tx_attr(tx, "is_open_market_purchase") is not True:
            continue
        session_raw = _tx_attr(tx, "first_tradable_session")
        if not session_raw:
            continue
        try:
            first_session = date.fromisoformat(str(session_raw))
        except ValueError:
            continue
        if first_session < cluster_start or first_session > next_execution_day:
            continue
        by_ticker.setdefault(ticker, []).append(tx)
    return by_ticker


def _distinct_cik_backed_buyers(rows: Sequence[Any]) -> set[str]:
    return {
        str(_tx_attr(tx, "insider_id"))
        for tx in rows
        if _tx_attr(tx, "insider_cik")
    }


def _name_only_could_collapse(rows: Sequence[Any]) -> bool:
    if len(rows) < 2:
        return False
    name_only = [tx for tx in rows if not _tx_attr(tx, "insider_cik")]
    return bool(name_only)


def _representative_rows_by_insider(rows: Sequence[Any]) -> List[Any]:
    latest: Dict[str, Any] = {}
    for tx in rows:
        insider = str(_tx_attr(tx, "insider_id"))
        current = latest.get(insider)
        if current is None or str(_tx_attr(tx, "first_tradable_session") or "") > str(_tx_attr(current, "first_tradable_session") or ""):
            latest[insider] = tx
    return list(latest.values())


def _size_locality_weight(tx: Any, all_rows: Sequence[Any]) -> Optional[float]:
    notional = _tx_attr(tx, "purchase_notional_usd")
    prior = [
        _tx_attr(row, "purchase_notional_usd")
        for row in all_rows
        if _tx_attr(row, "insider_id") == _tx_attr(tx, "insider_id")
        and str(_tx_attr(row, "transaction_date") or "") < str(_tx_attr(tx, "transaction_date") or "")
    ]
    weights = trade_size_weights(
        purchase_notional_usd=notional,
        market_cap_usd=_tx_attr(tx, "market_cap_usd"),
        prior_purchase_notionals=prior,
    )
    if weights is None:
        return None
    locality = LOCALITY_DEFAULT_WEIGHT
    insider_state = str(_tx_attr(tx, "insider_state") or "").strip().upper()
    issuer_state = str(_tx_attr(tx, "issuer_state") or "").strip().upper()
    if insider_state and issuer_state and insider_state == issuer_state:
        locality = LOCALITY_MATCH_WEIGHT
    return weights.production_weight * locality


def _cluster_member_payloads(
    rows: Sequence[Any],
    classifications: Dict[Tuple[str, int], CMPClassification],
    signal_year: int,
) -> List[Dict[str, Any]]:
    payloads = []
    for tx in rows:
        insider_id = str(_tx_attr(tx, "insider_id"))
        classification = classifications.get((insider_id, signal_year))
        payloads.append({
            "transaction_id": _tx_attr(tx, "transaction_id"),
            "filing_accession_number": _tx_attr(tx, "filing_accession_number"),
            "insider_id": insider_id,
            "insider_cik": _tx_attr(tx, "insider_cik"),
            "classification": (
                classification.classification
                if classification is not None else CLASSIFICATION_UNCLASSIFIABLE
            ),
            "first_tradable_session": _tx_attr(tx, "first_tradable_session"),
        })
    return payloads


def _cluster_source_authority(rows: Sequence[Any]) -> str:
    authorities = {
        str(_tx_attr(tx, "source_authority") or "").strip()
        for tx in rows
    }
    if authorities == {"sec_edgar"}:
        return "sec_edgar"
    return "fmp_backfill"


def _clock_quality(rows: Sequence[Any]) -> str:
    qualities = {str(_tx_attr(tx, "clock_quality") or "") for tx in rows}
    if "filing_date_only" in qualities:
        return "filing_date_only"
    if "detected_only" in qualities:
        return "detected_only"
    if "accepted_only" in qualities:
        return "accepted_only"
    return "accepted_detected"


def _field(
    name: str,
    value: Any,
    source_timestamp: datetime,
    source_provider: str,
    lineage_hash: Optional[str] = None,
) -> AssembledField:
    presence = FieldPresence.PRESENT if value is not None else FieldPresence.MISSING
    return AssembledField(
        name=name,
        value=value,
        presence=presence,
        source_timestamp=source_timestamp,
        source_provider=source_provider,
        lineage_hash=lineage_hash,
    )


def _first_session_open_after(public_ts: datetime) -> date:
    local_day = public_ts.astimezone(EASTERN_TZ).date()
    session = next_us_equity_session(local_day)
    open_ts = us_equity_session_open_timestamp(session)
    if open_ts <= public_ts.astimezone(timezone.utc):
        session = next_us_equity_session(session + timedelta(days=1))
    return session


def _transaction_knowledge_timestamp(tx: Any) -> Optional[datetime]:
    accepted = _ensure_aware(_tx_attr(tx, "filing_accepted_at"))
    if accepted is not None:
        return accepted
    filing_date = _parse_date(_tx_attr(tx, "filing_date"))
    if filing_date is None:
        return None
    return datetime.combine(filing_date, time.min, EASTERN_TZ).astimezone(timezone.utc)


def _tx_attr(tx: Any, name: str) -> Any:
    if isinstance(tx, dict):
        return tx.get(name)
    return getattr(tx, name, None)


def _snap_attr(snap: Any, name: str) -> Any:
    if isinstance(snap, dict):
        return snap.get(name)
    return getattr(snap, name, None)


def _parse_date(value: Any) -> Optional[date]:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _ensure_aware(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _finite_positive(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def transaction_evidence_from_sec(
    tx: Any,
    *,
    detected_at: Optional[datetime],
    market_cap_usd: Optional[float],
    ticker: Optional[str] = None,
    issuer_state: Optional[str] = None,
    lineage_ids: Optional[List[str]] = None,
    lineage_hashes: Optional[List[str]] = None,
) -> M2TransactionEvidence:
    """Convert a parsed SEC Form 4 adapter row to assembler evidence."""

    identity = resolve_insider_identity(
        sec_owner_cik=getattr(tx, "insider_cik", None),
        owner_name=getattr(tx, "insider_name", None),
    )
    filing_date = getattr(tx, "filing_date", None)
    clock = first_tradable_session_after_publication(
        filing_accepted_at=getattr(tx, "filing_accepted_at", None),
        filing_detected_at=detected_at or getattr(tx, "filing_accepted_at", None),
        filing_date=filing_date,
    )
    transaction_code = str(getattr(tx, "transaction_code", "") or "").strip().upper()
    acquired_disposed = str(getattr(tx, "acquired_disposed_code", "") or "").strip().upper()
    shares = getattr(tx, "shares", None)
    price = getattr(tx, "price_per_share", None)
    notional = None
    if _finite_positive(shares) is not None and _finite_positive(price) is not None:
        notional = abs(float(shares) * float(price))
    is_buy = acquired_disposed == "A" or transaction_code == OPEN_MARKET_PURCHASE_CODE
    is_sell = acquired_disposed == "D" or transaction_code == OPEN_MARKET_SALE_CODE
    is_open_purchase = transaction_code == OPEN_MARKET_PURCHASE_CODE and is_buy
    return M2TransactionEvidence(
        transaction_id=getattr(tx, "transaction_id"),
        ticker=str(ticker or getattr(tx, "ticker", "") or "").upper(),
        source_authority="sec_edgar",
        insider_id=identity.insider_id,
        insider_cik=identity.cik,
        insider_name=getattr(tx, "insider_name", None),
        identity_resolution_method=identity.method,
        identity_resolution_confidence=identity.confidence,
        filing_accession_number=getattr(tx, "accession_number", None),
        filing_form=getattr(tx, "filing_form", None),
        filing_date=filing_date.isoformat() if isinstance(filing_date, date) else None,
        filing_accepted_at=getattr(tx, "filing_accepted_at", None),
        filing_detected_at=detected_at or getattr(tx, "filing_accepted_at", None),
        first_tradable_session=clock.first_tradable_session.isoformat(),
        clock_quality=clock.clock_quality,
        transaction_date=(
            getattr(tx, "transaction_date").isoformat()
            if isinstance(getattr(tx, "transaction_date", None), date) else None
        ),
        transaction_code=transaction_code,
        acquired_disposed_code=acquired_disposed,
        transaction_shares=float(shares) if _finite_positive(shares) is not None else None,
        transaction_price_per_share=float(price) if _finite_positive(price) is not None else None,
        purchase_notional_usd=notional if is_open_purchase else None,
        market_cap_usd=market_cap_usd,
        issuer_cik=getattr(tx, "issuer_cik", None),
        issuer_name=getattr(tx, "issuer_name", None),
        issuer_state=issuer_state,
        insider_state=getattr(tx, "insider_state", None),
        insider_roles=dict(getattr(tx, "insider_roles", None) or {}),
        ownership_type=getattr(tx, "ownership_type", None),
        is_open_market_purchase=is_open_purchase,
        is_buy=is_buy,
        is_sell=is_sell,
        is_10b5_1=getattr(tx, "is_10b5_1", None),
        data_lineage_ids=list(lineage_ids or []),
        lineage_hashes=list(lineage_hashes or []),
        raw=dict(getattr(tx, "raw", None) or {}),
    )


def transaction_evidence_from_fmp(
    trade: Any,
    *,
    accession_owner_cik: Optional[str],
    detected_at: datetime,
    market_cap_usd: Optional[float],
    issuer_state: Optional[str] = None,
    lineage_ids: Optional[List[str]] = None,
    lineage_hashes: Optional[List[str]] = None,
) -> Optional[M2TransactionEvidence]:
    """Convert an accession-joined FMP enrichment row into M2 evidence."""

    accession = getattr(trade, "accession_number", None)
    if not accession:
        return None
    identity = resolve_insider_identity(
        fmp_owner_cik=getattr(trade, "reporting_cik", None),
        accession_owner_cik=accession_owner_cik,
        owner_name=getattr(trade, "reporting_name", None),
    )
    if not identity.cik_backed:
        return None
    filing_date = _parse_date(getattr(trade, "filing_date", None))
    clock = first_tradable_session_after_publication(
        filing_accepted_at=None,
        filing_detected_at=None,
        filing_date=filing_date,
    )
    shares = getattr(trade, "securities_transacted", None)
    price = getattr(trade, "price", None)
    notional = None
    if _finite_positive(shares) is not None and _finite_positive(price) is not None:
        notional = abs(float(shares) * float(price))
    code = str(getattr(trade, "transaction_code", None) or "").strip().upper()
    disposition = str(getattr(trade, "acquisition_or_disposition", None) or "").strip().upper()
    tx_type = str(getattr(trade, "transaction_type", None) or "").strip().lower()
    is_purchase_text = "purchase" in tx_type or "acquisition" in tx_type
    is_sale_text = "sale" in tx_type or "disposition" in tx_type
    is_buy = disposition == "A" or code == OPEN_MARKET_PURCHASE_CODE or is_purchase_text
    is_sell = disposition == "D" or code == OPEN_MARKET_SALE_CODE or is_sale_text
    is_open_purchase = (code in {"", OPEN_MARKET_PURCHASE_CODE} and is_buy)
    transaction_id = stable_hash({
        "source_authority": "fmp_backfill",
        "accession_number": accession,
        "reporting_cik": identity.cik,
        "transaction_date": getattr(trade, "transaction_date", None),
        "transaction_code": code,
        "shares": shares,
        "price": price,
    })
    return M2TransactionEvidence(
        transaction_id=transaction_id,
        ticker=str(getattr(trade, "symbol", "") or "").upper(),
        source_authority="fmp_backfill",
        insider_id=identity.insider_id,
        insider_cik=identity.cik,
        insider_name=getattr(trade, "reporting_name", None),
        identity_resolution_method=identity.method,
        identity_resolution_confidence=identity.confidence,
        filing_accession_number=accession,
        filing_form="4",
        filing_date=filing_date.isoformat() if filing_date else None,
        filing_accepted_at=None,
        filing_detected_at=detected_at,
        first_tradable_session=clock.first_tradable_session.isoformat(),
        clock_quality=clock.clock_quality,
        transaction_date=str(getattr(trade, "transaction_date", "") or "")[:10] or None,
        transaction_code=code or None,
        acquired_disposed_code=disposition or None,
        transaction_shares=float(shares) if _finite_positive(shares) is not None else None,
        transaction_price_per_share=float(price) if _finite_positive(price) is not None else None,
        purchase_notional_usd=notional if is_open_purchase else None,
        market_cap_usd=market_cap_usd,
        issuer_cik=getattr(trade, "company_cik", None),
        issuer_state=issuer_state,
        ownership_type=getattr(trade, "ownership_type", None),
        is_open_market_purchase=is_open_purchase,
        is_buy=is_buy,
        is_sell=is_sell,
        data_lineage_ids=list(lineage_ids or []),
        lineage_hashes=list(lineage_hashes or []),
        raw=dict(getattr(trade, "raw", None) or {}),
    )
