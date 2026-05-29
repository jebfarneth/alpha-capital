"""Signal-time source-context enrichment for M4 daily inputs.

This module freezes adapter-sourced context beside an assembled M4
``PatternInput``. The context is diagnostic/source evidence only: it must not
change M4 firing logic, price inputs, identity guards, or downstream ranking.

SIGNAL_CONTEXT limitations:
  - Polygon short-interest, short-volume, splits, and dividends expose event
    dates that are not availability proof. When a row lacks an explicit
    publication/announcement/dissemination timestamp, this module applies
    conservative lag constants before treating rows as PIT-eligible. The short
    interest lag is intentionally conservative because FINRA publishes short
    interest on a bi-monthly schedule after the settlement date; short volume
    uses a smaller daily-publication lag. Split/dividend lags are conservative
    source-adapter assumptions for replay safety when no row-level
    announcement timestamp is present.
  - Downstream ML feature assembly must never derive a PIT feature directly
    from context event date lists. PIT features must come from
    availability-gated counts/latest fields and source_attempt eligibility.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from alpha.data.contracts import (
    AdapterResponse,
    LineageMeta,
    ProviderError,
    stable_hash,
    utcnow,
)
from alpha.db.models import DataLineage, FeatureSnapshot, SignalRegistry
from alpha.evidence.writer import record_data_lineage
from alpha.market_calendar import next_us_equity_session
from alpha.patterns.contracts import PatternInput

SOURCE_CONTEXT_VERSION = "m4-signal-context-v1"
DEFAULT_M4_CONTEXT_BREAKOUT_BUFFER = 0.02
WINDOW_7D = 7
WINDOW_90D = 90
CORPORATE_ACTION_LOOKBACK_DAYS = 30
CORPORATE_ACTION_LOOKAHEAD_DAYS = 30
DEFAULT_PAGE_SIZE = 100
# FINRA short-interest rows are event-dated by settlement date and published on
# a later bi-monthly dissemination schedule. Without an explicit row-level
# publication timestamp, require this many U.S. equity sessions after the
# settlement date before replay eligibility.
SHORT_INTEREST_DISSEMINATION_LAG_TRADING_DAYS = 10
# Daily short-volume data is materially fresher than bi-monthly short-interest
# data, but trade date alone is still not availability proof under replay.
SHORT_VOLUME_DISSEMINATION_LAG_TRADING_DAYS = 1
# Polygon split rows expose execution_date as the event date. If the raw row has
# no announcement/publication timestamp, require several sessions before replay
# eligibility rather than treating execution_date as availability proof.
SPLIT_ANNOUNCEMENT_LAG_TRADING_DAYS = 5
# Dividend rows often expose declaration_date. If not, ex_dividend_date remains
# event context only and a conservative lag is used for replay eligibility.
DIVIDEND_ANNOUNCEMENT_LAG_TRADING_DAYS = 1
SHORT_INTEREST_AVAILABILITY_FIELDS = (
    "disseminated_at",
    "dissemination_date",
    "dissemination_timestamp",
    "published_at",
    "published_utc",
    "publication_date",
    "publication_timestamp",
    "reported_at",
    "report_timestamp",
)
SHORT_VOLUME_AVAILABILITY_FIELDS = SHORT_INTEREST_AVAILABILITY_FIELDS
CORPORATE_ACTION_AVAILABILITY_FIELDS = (
    "announcement_date",
    "announced_at",
    "announced_date",
    "declaration_date",
    "declared_at",
    "declared_date",
    "disseminated_at",
    "dissemination_date",
    "published_at",
    "published_utc",
    "publication_date",
    "reported_at",
)


def select_m4_signal_context_inputs(
    inputs: Sequence[PatternInput],
    *,
    breakout_buffer: float = DEFAULT_M4_CONTEXT_BREAKOUT_BUFFER,
) -> Tuple[List[PatternInput], Dict[str, Any]]:
    """Select a superset of base-lane M4 firings for expensive context calls.

    The base lane (``ENTRY_LANE_BASE``) fires only when ``price >= high_52w``
    (see ``M4Detector.detect``), so any non-negative ``breakout_buffer`` makes
    this a no-missed-firing superset *for that lane*. It is NOT a firing superset
    for the fresh-breakout lane (``ENTRY_LANE_FRESH``): the watchlist sub-path
    fires at ``base_nearness >= 0.97`` and the activation sub-path keys off
    intraday ``last_price``, neither of which this close-price filter accounts
    for. Do not reuse this for fresh-lane inputs without revisiting both the
    threshold and the price basis.
    """

    buffer = validate_m4_context_breakout_buffer(breakout_buffer)
    threshold_multiplier = 1.0 - buffer
    selected: List[PatternInput] = []
    invalid_count = 0

    for inp in inputs:
        price = _finite_float(inp.market_data.get("price"))
        high_52w = _finite_float(inp.market_data.get("high_52w"))
        if price is None or high_52w is None or high_52w <= 0:
            invalid_count += 1
            continue
        if price >= high_52w * threshold_multiplier:
            selected.append(inp)

    metrics = {
        "context_prefilter_enabled": True,
        "context_prefilter_breakout_buffer": buffer,
        "context_prefilter_threshold_multiplier": threshold_multiplier,
        "context_prefilter_input_count": len(inputs),
        "context_prefilter_candidate_count": len(selected),
        "context_prefilter_skipped_count": len(inputs) - len(selected),
        "context_prefilter_invalid_count": invalid_count,
    }
    return selected, metrics


def enrich_m4_signal_context(
    inputs: Sequence[PatternInput],
    *,
    session: Session,
    polygon_adapter: Any = None,
    benzinga_adapter: Any = None,
    cutoff_timestamp: datetime,
    decision_date: str,
    evidence_session_date: str,
    job_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach signal_context to each input and append context lineage.

    Adapter failures are represented inside the context source_attempts. They
    do not block persistence unless callers raise before this function is
    reached.
    """

    cutoff = _ensure_aware(cutoff_timestamp)
    evidence_day = date.fromisoformat(evidence_session_date)
    metrics = {
        "schema_version": SOURCE_CONTEXT_VERSION,
        "input_count": len(inputs),
        "context_attached_count": 0,
        "source_attempt_count": 0,
        "provider_error_count": 0,
        "parse_error_count": 0,
        "validation_error_count": 0,
        "unavailable_count": 0,
        "pit_excluded_count": 0,
        "lineage_count": 0,
        "context_reused_count": 0,
        "context_reused_in_memory_count": 0,
        "context_enriched_count": 0,
    }

    for inp in inputs:
        existing_context = inp.market_data.get("signal_context")
        if _reusable_signal_context(existing_context, cutoff):
            metrics["context_reused_count"] += 1
            metrics["context_reused_in_memory_count"] += 1
            attempts = _all_attempts(existing_context)
            metrics["source_attempt_count"] += len(attempts)
            metrics["provider_error_count"] += sum(
                1 for item in attempts if item.get("status") == "provider_error"
            )
            metrics["parse_error_count"] += sum(
                1 for item in attempts if item.get("status") == "parse_error"
            )
            metrics["validation_error_count"] += sum(
                1 for item in attempts if item.get("status") == "validation_error"
            )
            metrics["unavailable_count"] += sum(
                1 for item in attempts if item.get("status") == "unavailable"
            )
            metrics["pit_excluded_count"] += sum(
                1 for item in attempts if item.get("status") == "pit_excluded"
            )
            continue

        context, lineage_refs = build_m4_signal_context(
            inp,
            session=session,
            polygon_adapter=polygon_adapter,
            benzinga_adapter=benzinga_adapter,
            cutoff_timestamp=cutoff,
            decision_date=decision_date,
            evidence_session_date=evidence_session_date,
            evidence_day=evidence_day,
            job_run_id=job_run_id,
        )
        inp.market_data["signal_context"] = context
        for lineage_id, lineage_hash in lineage_refs:
            if lineage_id and lineage_id not in inp.lineage_ids:
                inp.lineage_ids.append(lineage_id)
            if lineage_hash and lineage_hash not in inp.lineage_hashes:
                inp.lineage_hashes.append(lineage_hash)

        metrics["context_attached_count"] += 1
        metrics["context_enriched_count"] += 1
        attempts = _all_attempts(context)
        metrics["source_attempt_count"] += len(attempts)
        metrics["provider_error_count"] += sum(
            1 for item in attempts if item.get("status") == "provider_error"
        )
        metrics["parse_error_count"] += sum(
            1 for item in attempts if item.get("status") == "parse_error"
        )
        metrics["validation_error_count"] += sum(
            1 for item in attempts if item.get("status") == "validation_error"
        )
        metrics["unavailable_count"] += sum(
            1 for item in attempts if item.get("status") == "unavailable"
        )
        metrics["pit_excluded_count"] += sum(
            1 for item in attempts if item.get("status") == "pit_excluded"
        )
        metrics["lineage_count"] += len(lineage_refs)

    return metrics


def reuse_persisted_m4_signal_context(
    inputs: Sequence[PatternInput],
    *,
    session: Session,
    cutoff_timestamp: datetime,
    decision_date: str,
) -> Dict[str, Any]:
    """Seed inputs with already-frozen persisted signal_context when available."""

    cutoff = _ensure_aware(cutoff_timestamp)
    metrics: Dict[str, Any] = {
        "context_reused_from_persistence_count": 0,
        "context_persistence_miss_count": 0,
        "context_persistence_mismatch_count": 0,
        "context_persistence_mismatch_reasons": {},
    }

    for inp in inputs:
        if _reusable_signal_context(inp.market_data.get("signal_context"), cutoff):
            continue

        identity_hash = _m4_setup_identity_hash(inp)
        if not identity_hash:
            metrics["context_persistence_miss_count"] += 1
            _increment_reason(metrics, "missing_signal_identity")
            continue

        signals = (
            session.query(SignalRegistry)
            .filter(
                SignalRegistry.pattern_id == "M4",
                SignalRegistry.ticker == inp.ticker.upper(),
                SignalRegistry.trading_date == decision_date,
            )
            .all()
        )
        if not signals:
            metrics["context_persistence_miss_count"] += 1
            continue

        feature: Optional[FeatureSnapshot] = None
        context: Optional[Dict[str, Any]] = None
        for signal in signals:
            candidate = session.get(FeatureSnapshot, signal.feature_snapshot_id)
            if not _feature_matches_m4_setup(candidate, identity_hash):
                continue
            feature = candidate
            context = _feature_signal_context(feature)
            break

        if feature is None:
            metrics["context_persistence_miss_count"] += 1
            _increment_reason(metrics, "signal_identity_mismatch")
            continue

        if not _reusable_signal_context(context, cutoff):
            metrics["context_persistence_mismatch_count"] += 1
            _increment_reason(metrics, _context_mismatch_reason(context, cutoff))
            continue

        inp.market_data["signal_context"] = context
        for lineage_id in _context_lineage_ids(context):
            if lineage_id and lineage_id not in inp.lineage_ids:
                inp.lineage_ids.append(lineage_id)
            lineage = session.get(DataLineage, lineage_id)
            lineage_hash = getattr(lineage, "raw_payload_hash", None)
            if lineage_hash and lineage_hash not in inp.lineage_hashes:
                inp.lineage_hashes.append(lineage_hash)
        metrics["context_reused_from_persistence_count"] += 1

    return metrics


def build_m4_signal_context(
    inp: PatternInput,
    *,
    session: Session,
    polygon_adapter: Any,
    benzinga_adapter: Any,
    cutoff_timestamp: datetime,
    decision_date: str,
    evidence_session_date: str,
    evidence_day: date,
    job_run_id: Optional[str],
) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    """Build one input's source context and data-lineage references."""

    ticker = inp.ticker.upper()
    lineage_refs: List[Tuple[str, str]] = []

    def register(resp: AdapterResponse[Any], source: str, request: Dict[str, Any]) -> Optional[str]:
        lineage_id = _record_context_lineage(
            session,
            resp,
            source=source,
            ticker=ticker,
            request=request,
            job_run_id=job_run_id,
        )
        if lineage_id:
            lineage_refs.append((lineage_id, resp.lineage.raw_payload_hash))
        return lineage_id

    context: Dict[str, Any] = {
        "schema_version": SOURCE_CONTEXT_VERSION,
        "ticker": ticker,
        "decision_date": decision_date,
        "evidence_session_date": evidence_session_date,
        "asof_timestamp": cutoff_timestamp.isoformat(),
        "identity": _identity_context(inp),
    }
    context.update(
        _polygon_context(
            ticker=ticker,
            adapter=polygon_adapter,
            cutoff=cutoff_timestamp,
            evidence_day=evidence_day,
            register=register,
        )
    )
    context.update(
        _benzinga_context(
            ticker=ticker,
            adapter=benzinga_adapter,
            cutoff=cutoff_timestamp,
            evidence_day=evidence_day,
            register=register,
        )
    )
    return context, lineage_refs


def _identity_context(inp: PatternInput) -> Dict[str, Any]:
    identity = inp.market_data.get("security_identity")
    if isinstance(identity, dict):
        status = identity.get("identity_status") or identity.get("status") or "present"
        if status == "present":
            context_status = "present"
        elif status:
            context_status = "missing"
        else:
            context_status = "missing"
        payload = _json_safe(identity)
    else:
        context_status = "unavailable"
        payload = None
    return {
        "status": context_status,
        "security_identity": payload,
        "source_attempts": [{
            "source": "Polygon identity",
            "status": context_status,
            "row_count": 1 if context_status == "present" else 0,
            "endpoint": "security_identity_snapshot",
            "query": {"ticker": inp.ticker.upper()},
            "warnings": {},
        }],
    }


def _polygon_context(
    *,
    ticker: str,
    adapter: Any,
    cutoff: datetime,
    evidence_day: date,
    register: Callable[[AdapterResponse[Any], str, Dict[str, Any]], Optional[str]],
) -> Dict[str, Any]:
    start_90 = evidence_day - timedelta(days=WINDOW_90D)
    start_7 = evidence_day - timedelta(days=WINDOW_7D)
    ca_start = evidence_day - timedelta(days=CORPORATE_ACTION_LOOKBACK_DAYS)
    ca_end = evidence_day + timedelta(days=CORPORATE_ACTION_LOOKAHEAD_DAYS)

    short_interest = _single_polygon_latest_context(
        adapter=adapter,
        method_name="get_short_interest",
        source="Polygon short interest",
        category="polygon_short_interest",
        ticker=ticker,
        cutoff=cutoff,
        request={
            "ticker": ticker,
            "settlement_date_from": start_90.isoformat(),
            "settlement_date_to": evidence_day.isoformat(),
            "limit": DEFAULT_PAGE_SIZE,
        },
        register=register,
        date_attr="settlement_date",
        fields=("short_interest", "days_to_cover", "avg_daily_volume"),
        availability_fields=SHORT_INTEREST_AVAILABILITY_FIELDS,
        lag_trading_days=SHORT_INTEREST_DISSEMINATION_LAG_TRADING_DAYS,
        lag_warning_flag="short_interest_availability_lag_applied",
    )

    short_volume = _single_polygon_latest_context(
        adapter=adapter,
        method_name="get_short_volume",
        source="Polygon short volume",
        category="polygon_short_volume",
        ticker=ticker,
        cutoff=cutoff,
        request={
            "ticker": ticker,
            "date_from": start_90.isoformat(),
            "date_to": evidence_day.isoformat(),
            "limit": DEFAULT_PAGE_SIZE,
        },
        register=register,
        date_attr="date",
        fields=("short_volume", "total_volume", "short_volume_ratio"),
        availability_fields=SHORT_VOLUME_AVAILABILITY_FIELDS,
        lag_trading_days=SHORT_VOLUME_DISSEMINATION_LAG_TRADING_DAYS,
        lag_warning_flag="short_volume_availability_lag_applied",
    )

    corporate = _polygon_corporate_context(
        adapter=adapter,
        ticker=ticker,
        cutoff=cutoff,
        evidence_day=evidence_day,
        ca_start=ca_start,
        ca_end=ca_end,
        register=register,
    )

    news = _polygon_news_context(
        adapter=adapter,
        ticker=ticker,
        cutoff=cutoff,
        start_7=start_7,
        start_90=start_90,
        evidence_day=evidence_day,
        register=register,
    )

    return {
        "polygon_short_interest": short_interest,
        "polygon_short_volume": short_volume,
        "polygon_corporate_actions": corporate,
        "polygon_news": news,
    }


def _single_polygon_latest_context(
    *,
    adapter: Any,
    method_name: str,
    source: str,
    category: str,
    ticker: str,
    cutoff: datetime,
    request: Dict[str, Any],
    register: Callable[[AdapterResponse[Any], str, Dict[str, Any]], Optional[str]],
    date_attr: str,
    fields: Sequence[str],
    availability_fields: Sequence[str],
    lag_trading_days: int,
    lag_warning_flag: str,
) -> Dict[str, Any]:
    resp = _call_adapter(adapter, method_name, asof=cutoff, **request)
    if resp is None:
        return _empty_category(source, "unavailable", request)
    lineage_id = register(resp, source, request)
    rows = list(resp.data or []) if resp.ok else []
    if not resp.ok:
        return _empty_category(
            source,
            _error_status(resp),
            request,
            attempt=_source_attempt(source, resp, request, lineage_id=lineage_id),
        )
    eligible, availability_flags = _polygon_availability_eligible_rows(
        rows,
        resp,
        cutoff,
        event_date_attr=date_attr,
        availability_fields=availability_fields,
        lag_trading_days=lag_trading_days,
        lag_warning_flag=lag_warning_flag,
    )
    latest = _latest_by_date(eligible, date_attr)
    attempt_status = _matched_no_data_or_pit(rows, eligible)
    attempt = _source_attempt(
        source,
        resp,
        request,
        lineage_id=lineage_id,
        status=attempt_status,
        eligible_rows=len(eligible),
        extra_warnings=availability_flags,
    )
    context: Dict[str, Any] = {
        "status": attempt_status,
        "event_dates": [_fields(row, ("ticker", date_attr)) for row in rows],
        "source_attempts": [attempt],
    }
    if latest is not None:
        context[date_attr] = _attr(latest, date_attr)
        for field in fields:
            context[field] = _json_safe(_attr(latest, field))
    return context


def _polygon_corporate_context(
    *,
    adapter: Any,
    ticker: str,
    cutoff: datetime,
    evidence_day: date,
    ca_start: date,
    ca_end: date,
    register: Callable[[AdapterResponse[Any], str, Dict[str, Any]], Optional[str]],
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    split_rows: List[Any] = []
    dividend_rows: List[Any] = []
    eligible_splits: List[Any] = []
    eligible_dividends: List[Any] = []

    split_request = {
        "ticker": ticker,
        "execution_date_from": ca_start.isoformat(),
        "execution_date_to": ca_end.isoformat(),
        "limit": DEFAULT_PAGE_SIZE,
    }
    split_resp = _call_adapter(adapter, "get_splits", asof=cutoff, **split_request)
    if split_resp is None:
        attempts.append(_unavailable_attempt("Polygon splits", split_request))
    else:
        lineage_id = register(split_resp, "Polygon splits", split_request)
        split_rows = list(split_resp.data or []) if split_resp.ok else []
        split_flags: Dict[str, Any] = {}
        if split_resp.ok:
            eligible_splits, split_flags = _polygon_availability_eligible_rows(
                split_rows,
                split_resp,
                cutoff,
                event_date_attr="execution_date",
                availability_fields=CORPORATE_ACTION_AVAILABILITY_FIELDS,
                lag_trading_days=SPLIT_ANNOUNCEMENT_LAG_TRADING_DAYS,
                lag_warning_flag="split_availability_lag_applied",
            )
        attempts.append(_source_attempt(
            "Polygon splits",
            split_resp,
            split_request,
            lineage_id=lineage_id,
            status=_matched_no_data_or_pit(split_rows, eligible_splits)
            if split_resp.ok else None,
            eligible_rows=len(eligible_splits),
            extra_warnings=split_flags,
        ))

    dividend_request = {
        "ticker": ticker,
        "ex_dividend_date_from": ca_start.isoformat(),
        "ex_dividend_date_to": ca_end.isoformat(),
        "limit": DEFAULT_PAGE_SIZE,
    }
    dividend_resp = _call_adapter(adapter, "get_dividends", asof=cutoff, **dividend_request)
    if dividend_resp is None:
        attempts.append(_unavailable_attempt("Polygon dividends", dividend_request))
    else:
        lineage_id = register(dividend_resp, "Polygon dividends", dividend_request)
        dividend_rows = list(dividend_resp.data or []) if dividend_resp.ok else []
        dividend_flags: Dict[str, Any] = {}
        if dividend_resp.ok:
            eligible_dividends, dividend_flags = _polygon_availability_eligible_rows(
                dividend_rows,
                dividend_resp,
                cutoff,
                event_date_attr="ex_dividend_date",
                availability_fields=CORPORATE_ACTION_AVAILABILITY_FIELDS,
                lag_trading_days=DIVIDEND_ANNOUNCEMENT_LAG_TRADING_DAYS,
                lag_warning_flag="dividend_availability_lag_applied",
            )
        attempts.append(_source_attempt(
            "Polygon dividends",
            dividend_resp,
            dividend_request,
            lineage_id=lineage_id,
            status=_matched_no_data_or_pit(dividend_rows, eligible_dividends)
            if dividend_resp.ok else None,
            eligible_rows=len(eligible_dividends),
            extra_warnings=dividend_flags,
        ))

    latest_dividend = _latest_by_date(eligible_dividends, "ex_dividend_date")
    context = {
        "status": "matched" if eligible_splits or eligible_dividends else _combined_status(attempts),
        "split_count_window": len(eligible_splits),
        "dividend_count_window": len(eligible_dividends),
        "split_event_dates": [
            _fields(row, ("ticker", "execution_date")) for row in split_rows
        ],
        "dividend_event_dates": [
            _fields(row, ("ticker", "ex_dividend_date", "declaration_date"))
            for row in dividend_rows
        ],
        "source_attempts": attempts,
    }
    if eligible_splits:
        latest_split = _latest_by_date(eligible_splits, "execution_date")
        if latest_split is not None:
            context["latest_split"] = _fields(latest_split, (
                "ticker",
                "execution_date",
                "split_from",
                "split_to",
                "adjustment_type",
                "historical_adjustment_factor",
            ))
    if latest_dividend is not None:
        ex_date = _attr(latest_dividend, "ex_dividend_date")
        context["last_dividend"] = _fields(latest_dividend, (
            "ticker",
            "ex_dividend_date",
            "cash_amount",
            "dividend_type",
            "distribution_type",
            "frequency",
        ))
        if isinstance(ex_date, str):
            try:
                context["dividend_proximity_days"] = (
                    date.fromisoformat(ex_date) - evidence_day
                ).days
            except ValueError:
                pass
    return context


def _polygon_news_context(
    *,
    adapter: Any,
    ticker: str,
    cutoff: datetime,
    start_7: date,
    start_90: date,
    evidence_day: date,
    register: Callable[[AdapterResponse[Any], str, Dict[str, Any]], Optional[str]],
) -> Dict[str, Any]:
    request = {
        "ticker": ticker,
        "published_utc_from": start_90.isoformat(),
        "published_utc_to": evidence_day.isoformat(),
        "limit": DEFAULT_PAGE_SIZE,
    }
    resp = _call_adapter(adapter, "get_news", asof=cutoff, **request)
    if resp is None:
        return _empty_category("Polygon news", "unavailable", request)
    lineage_id = register(resp, "Polygon news", request)
    if not resp.ok:
        return _empty_category(
            "Polygon news",
            _error_status(resp),
            request,
            attempt=_source_attempt("Polygon news", resp, request, lineage_id=lineage_id),
        )
    rows = list(resp.data or [])
    eligible = [
        row for row in rows
        if _knowledge_ts(row, ("published_utc",), cutoff=cutoff) is not None
    ]
    latest = _latest_by_datetime(eligible, ("published_utc",), cutoff=cutoff)
    context: Dict[str, Any] = {
        "status": _matched_no_data_or_pit(rows, eligible),
        "article_count_7d": _count_since(eligible, ("published_utc",), start_7, cutoff),
        "article_count_90d": _count_since(eligible, ("published_utc",), start_90, cutoff),
        "source_attempts": [_source_attempt(
            "Polygon news",
            resp,
            request,
            lineage_id=lineage_id,
            status=_matched_no_data_or_pit(rows, eligible),
            eligible_rows=len(eligible),
        )],
    }
    if latest is not None:
        context["latest_title"] = _attr(latest, "title")
        context["latest_published_utc"] = _attr(latest, "published_utc")
        context["latest_tickers"] = _json_safe(_attr(latest, "tickers")) or []
        context["latest_publisher"] = _attr(latest, "publisher_name")
        context["latest_sentiment"] = _sentiment_summary(_attr(latest, "insights"))
    return context


def _benzinga_context(
    *,
    ticker: str,
    adapter: Any,
    cutoff: datetime,
    evidence_day: date,
    register: Callable[[AdapterResponse[Any], str, Dict[str, Any]], Optional[str]],
) -> Dict[str, Any]:
    start_90 = evidence_day - timedelta(days=WINDOW_90D)
    start_7 = evidence_day - timedelta(days=WINDOW_7D)
    event_window_end = evidence_day + timedelta(days=30)

    news = _benzinga_news_context(
        adapter=adapter,
        ticker=ticker,
        cutoff=cutoff,
        start_7=start_7,
        start_90=start_90,
        evidence_day=evidence_day,
        register=register,
    )
    calendar = _benzinga_calendar_context(
        adapter=adapter,
        ticker=ticker,
        cutoff=cutoff,
        start_90=start_90,
        event_window_end=event_window_end,
        register=register,
    )
    insider = _benzinga_insider_context(
        adapter=adapter,
        ticker=ticker,
        cutoff=cutoff,
        start_90=start_90,
        evidence_day=evidence_day,
        register=register,
    )
    ma = _benzinga_ma_context(
        adapter=adapter,
        ticker=ticker,
        cutoff=cutoff,
        start_90=start_90,
        event_window_end=event_window_end,
        register=register,
    )
    return {
        "benzinga_news": news,
        "benzinga_calendar": calendar,
        "benzinga_insider": insider,
        "benzinga_ma": ma,
    }


def _benzinga_news_context(
    *,
    adapter: Any,
    ticker: str,
    cutoff: datetime,
    start_7: date,
    start_90: date,
    evidence_day: date,
    register: Callable[[AdapterResponse[Any], str, Dict[str, Any]], Optional[str]],
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    all_news: List[Any] = []
    wiims: List[Any] = []

    request = {
        "tickers": ticker,
        "date_from": start_90.isoformat(),
        "date_to": evidence_day.isoformat(),
        "pagesize": DEFAULT_PAGE_SIZE,
    }
    news_resp = _call_adapter(adapter, "get_news", asof=cutoff, **request)
    if news_resp is None:
        attempts.append(_unavailable_attempt("Benzinga news", request))
    else:
        lineage_id = register(news_resp, "Benzinga news", request)
        rows = list(news_resp.data or []) if news_resp.ok else []
        all_news = _benzinga_eligible_rows(rows, cutoff, ("published", "created", "updated"))
        attempts.append(_source_attempt(
            "Benzinga news",
            news_resp,
            request,
            lineage_id=lineage_id,
            status=_matched_no_data_or_pit(rows, all_news) if news_resp.ok else None,
            eligible_rows=len(all_news),
        ))

    wiim_request = dict(request)
    wiim_resp = _call_adapter(adapter, "get_wiims", asof=cutoff, **wiim_request)
    if wiim_resp is None:
        attempts.append(_unavailable_attempt("Benzinga WIIMs", wiim_request))
    else:
        lineage_id = register(wiim_resp, "Benzinga WIIMs", wiim_request)
        rows = list(wiim_resp.data or []) if wiim_resp.ok else []
        wiims = _benzinga_eligible_rows(rows, cutoff, ("published", "created", "updated"))
        attempts.append(_source_attempt(
            "Benzinga WIIMs",
            wiim_resp,
            wiim_request,
            lineage_id=lineage_id,
            status=_matched_no_data_or_pit(rows, wiims) if wiim_resp.ok else None,
            eligible_rows=len(wiims),
        ))

    latest = _latest_by_datetime(all_news + wiims, ("published", "created", "updated"), cutoff=cutoff)
    context: Dict[str, Any] = {
        "status": "matched" if all_news or wiims else _combined_status(attempts),
        "article_count_7d": _count_since(all_news, ("published", "created", "updated"), start_7, cutoff),
        "article_count_90d": _count_since(all_news, ("published", "created", "updated"), start_90, cutoff),
        "wiim_count_7d": _count_since(wiims, ("published", "created", "updated"), start_7, cutoff),
        "source_attempts": attempts,
    }
    if latest is not None:
        context["latest_title"] = _attr(latest, "title")
        context["latest_url"] = _attr(latest, "url")
        context["latest_tickers"] = _json_safe(_attr(latest, "tickers")) or []
        context["latest_channels"] = _json_safe(_attr(latest, "channels")) or []
        context["latest_source"] = _attr(latest, "source")
    return context


def _benzinga_calendar_context(
    *,
    adapter: Any,
    ticker: str,
    cutoff: datetime,
    start_90: date,
    event_window_end: date,
    register: Callable[[AdapterResponse[Any], str, Dict[str, Any]], Optional[str]],
) -> Dict[str, Any]:
    specs = [
        ("earnings", "get_earnings", "Benzinga earnings"),
        ("guidance", "get_guidance", "Benzinga guidance"),
        ("ratings", "get_ratings", "Benzinga ratings"),
        ("offerings", "get_offerings", "Benzinga offerings"),
        ("dividends", "get_dividends", "Benzinga dividends"),
    ]
    attempts: List[Dict[str, Any]] = []
    context: Dict[str, Any] = {
        "status": "no_data",
        "source_attempts": attempts,
        "earnings": {},
        "guidance": {},
        "ratings": {},
        "offerings": {},
        "dividends": {},
    }
    any_match = False
    any_pit_excluded = False
    for key, method_name, source in specs:
        request = {
            "tickers": ticker,
            "date_from": start_90.isoformat(),
            "date_to": event_window_end.isoformat(),
            "pagesize": DEFAULT_PAGE_SIZE,
        }
        resp = _call_adapter(adapter, method_name, asof=cutoff, **request)
        if resp is None:
            attempts.append(_unavailable_attempt(source, request))
            continue
        lineage_id = register(resp, source, request)
        rows = list(resp.data or []) if resp.ok else []
        eligible = _benzinga_eligible_rows(rows, cutoff, ("updated",))
        if rows and not eligible and resp.ok:
            any_pit_excluded = True
        if eligible:
            any_match = True
        attempts.append(_source_attempt(
            source,
            resp,
            request,
            lineage_id=lineage_id,
            status=_matched_no_data_or_pit(rows, eligible) if resp.ok else None,
            eligible_rows=len(eligible),
        ))
        context[key] = _calendar_summary(key, rows, eligible)
    context["status"] = "matched" if any_match else (
        "pit_excluded" if any_pit_excluded else _combined_status(attempts)
    )
    return context


def _benzinga_insider_context(
    *,
    adapter: Any,
    ticker: str,
    cutoff: datetime,
    start_90: date,
    evidence_day: date,
    register: Callable[[AdapterResponse[Any], str, Dict[str, Any]], Optional[str]],
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    filings: List[Any] = []
    transactions: List[Any] = []
    request = {
        "tickers": ticker,
        "date_from": start_90.isoformat(),
        "date_to": evidence_day.isoformat(),
        "pagesize": DEFAULT_PAGE_SIZE,
    }

    filings_resp = _call_adapter(adapter, "get_insider_filings", asof=cutoff, **request)
    if filings_resp is None:
        attempts.append(_unavailable_attempt("Benzinga insider filings", request))
    else:
        lineage_id = register(filings_resp, "Benzinga insider filings", request)
        rows = list(filings_resp.data or []) if filings_resp.ok else []
        filings = _benzinga_eligible_rows(rows, cutoff, ("filing_date", "updated"))
        attempts.append(_source_attempt(
            "Benzinga insider filings",
            filings_resp,
            request,
            lineage_id=lineage_id,
            status=_matched_no_data_or_pit(rows, filings) if filings_resp.ok else None,
            eligible_rows=len(filings),
        ))

    tx_resp = _call_adapter(adapter, "get_insider_transactions", asof=cutoff, **request)
    if tx_resp is None:
        attempts.append(_unavailable_attempt("Benzinga insider transactions", request))
    else:
        lineage_id = register(tx_resp, "Benzinga insider transactions", request)
        rows = list(tx_resp.data or []) if tx_resp.ok else []
        transactions = _benzinga_eligible_rows(rows, cutoff, ("filing_date", "updated"))
        attempts.append(_source_attempt(
            "Benzinga insider transactions",
            tx_resp,
            request,
            lineage_id=lineage_id,
            status=_matched_no_data_or_pit(rows, transactions) if tx_resp.ok else None,
            eligible_rows=len(transactions),
        ))

    tx_summary = _insider_transaction_summary(transactions)
    return {
        "status": "matched" if filings or transactions else _combined_status(attempts),
        "filing_count": len(filings),
        "transaction_count": len(transactions),
        **tx_summary,
        "source_attempts": attempts,
    }


def _benzinga_ma_context(
    *,
    adapter: Any,
    ticker: str,
    cutoff: datetime,
    start_90: date,
    event_window_end: date,
    register: Callable[[AdapterResponse[Any], str, Dict[str, Any]], Optional[str]],
) -> Dict[str, Any]:
    request = {
        "tickers": ticker,
        "date_from": start_90.isoformat(),
        "date_to": event_window_end.isoformat(),
        "pagesize": DEFAULT_PAGE_SIZE,
    }
    resp = _call_adapter(adapter, "get_mergers_acquisitions", asof=cutoff, **request)
    if resp is None:
        return _empty_category("Benzinga M&A", "unavailable", request)
    lineage_id = register(resp, "Benzinga M&A", request)
    if not resp.ok:
        return _empty_category(
            "Benzinga M&A",
            _error_status(resp),
            request,
            attempt=_source_attempt("Benzinga M&A", resp, request, lineage_id=lineage_id),
        )
    rows = list(resp.data or [])
    eligible = _benzinga_eligible_rows(rows, cutoff, ("updated",))
    latest = _latest_by_attr(eligible, "updated")
    context: Dict[str, Any] = {
        "status": _matched_no_data_or_pit(rows, eligible),
        "presence": bool(eligible),
        "review_context_only": True,
        "source_attempts": [_source_attempt(
            "Benzinga M&A",
            resp,
            request,
            lineage_id=lineage_id,
            status=_matched_no_data_or_pit(rows, eligible),
            eligible_rows=len(eligible),
        )],
        "event_dates": [
            _fields(row, ("date", "date_expected", "date_completed"))
            for row in rows
        ],
    }
    if latest is not None:
        context["latest"] = _fields(latest, (
            "target_ticker",
            "target_name",
            "deal_type",
            "deal_status",
            "deal_payment_type",
            "date",
            "date_expected",
            "date_completed",
            "updated",
        ))
    return context


def _call_adapter(adapter: Any, method_name: str, **kwargs: Any) -> Optional[AdapterResponse[Any]]:
    method = getattr(adapter, method_name, None) if adapter is not None else None
    if method is None:
        return None
    try:
        return method(**kwargs)
    except Exception:  # pragma: no cover - defensive isolation path
        request_ts = utcnow()
        asof = kwargs.get("asof")
        asof_ts = _ensure_aware(asof) if isinstance(asof, datetime) else request_ts
        endpoint = f"signal_context/{method_name}"
        return AdapterResponse(
            data=None,
            lineage=LineageMeta(
                provider="signal_context",
                endpoint=endpoint,
                request_timestamp=request_ts,
                asof_timestamp=asof_ts,
                raw_payload_hash=stable_hash({"method": method_name, "error": "exception"}),
                source_authority="signal_context",
            ),
            error=ProviderError(
                provider="signal_context",
                endpoint=endpoint,
                status_code=None,
                error_type="http",
                message="signal_context adapter call failed",
                retryable=True,
            ),
        )


def _record_context_lineage(
    session: Session,
    resp: AdapterResponse[Any],
    *,
    source: str,
    ticker: str,
    request: Dict[str, Any],
    job_run_id: Optional[str],
) -> Optional[str]:
    if resp.lineage is None:
        return None
    raw_payload = {
        "source": source,
        "ticker": ticker,
        "request": _json_safe(request),
        "data": _json_safe(resp.data),
        "error": _json_safe(resp.error),
    }
    lineage = record_data_lineage(
        session,
        provider=resp.lineage.provider,
        endpoint=resp.lineage.endpoint,
        asof_timestamp=resp.lineage.asof_timestamp,
        raw_payload=raw_payload,
        raw_payload_hash=resp.lineage.raw_payload_hash,
        request_timestamp=resp.lineage.request_timestamp,
        freshness_seconds=resp.lineage.freshness_seconds,
        source_authority=resp.lineage.source_authority,
        data_quality_flags=resp.lineage.data_quality_flags,
        job_run_id=job_run_id,
    )
    return lineage.data_lineage_id


def _source_attempt(
    source: str,
    resp: AdapterResponse[Any],
    request: Dict[str, Any],
    *,
    lineage_id: Optional[str],
    status: Optional[str] = None,
    eligible_rows: Optional[int] = None,
    extra_warnings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    flags = dict(resp.lineage.data_quality_flags or {})
    if extra_warnings:
        flags.update(extra_warnings)
    row_count = len(resp.data or []) if isinstance(resp.data, list) else (
        len(resp.data.results) if _has_attr(resp.data, "results") else 0
    )
    if status is None:
        status = "matched" if resp.ok and row_count > 0 else (
            "no_data" if resp.ok else _error_status(resp)
        )
    attempt = {
        "source": source,
        "status": status,
        "row_count": row_count,
        "eligible_row_count": eligible_rows,
        "pit_excluded_row_count": (
            max(row_count - eligible_rows, 0) if eligible_rows is not None else None
        ),
        "lineage_id": lineage_id,
        "endpoint": resp.lineage.endpoint,
        "query": _json_safe(request),
        "warnings": flags,
    }
    if resp.error is not None:
        attempt["error_type"] = resp.error.error_type
        attempt["retryable"] = resp.error.retryable
        attempt["status_code"] = resp.error.status_code
    return attempt


def _unavailable_attempt(source: str, request: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": source,
        "status": "unavailable",
        "row_count": 0,
        "eligible_row_count": 0,
        "pit_excluded_row_count": 0,
        "lineage_id": None,
        "endpoint": None,
        "query": _json_safe(request),
        "warnings": {},
    }


def _empty_category(
    source: str,
    status: str,
    request: Dict[str, Any],
    *,
    attempt: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "source_attempts": [attempt or _unavailable_attempt(source, request)],
    }


def _error_status(resp: AdapterResponse[Any]) -> str:
    error_type = getattr(resp.error, "error_type", None)
    if error_type == "validation":
        return "validation_error"
    if error_type == "parse":
        return "parse_error"
    return "provider_error"


def _matched_no_data_or_pit(rows: Sequence[Any], eligible_rows: Sequence[Any]) -> str:
    if eligible_rows:
        return "matched"
    if rows:
        return "pit_excluded"
    return "no_data"


def _combined_status(attempts: Sequence[Dict[str, Any]]) -> str:
    statuses = [attempt.get("status") for attempt in attempts]
    if "matched" in statuses:
        return "matched"
    if "pit_excluded" in statuses:
        return "pit_excluded"
    if "provider_error" in statuses:
        return "provider_error"
    if "parse_error" in statuses:
        return "parse_error"
    if "validation_error" in statuses:
        return "validation_error"
    if statuses and all(status == "unavailable" for status in statuses):
        return "unavailable"
    return "no_data"


def _polygon_eligible_rows(
    rows: Sequence[Any],
    resp: AdapterResponse[Any],
    cutoff: datetime,
) -> Tuple[List[Any], Dict[str, Any]]:
    return _polygon_availability_eligible_rows(
        rows,
        resp,
        cutoff,
        event_date_attr=None,
        availability_fields=(),
        lag_trading_days=0,
        lag_warning_flag="availability_lag_applied",
    )


def _polygon_availability_eligible_rows(
    rows: Sequence[Any],
    resp: AdapterResponse[Any],
    cutoff: datetime,
    *,
    event_date_attr: Optional[str],
    availability_fields: Sequence[str],
    lag_trading_days: int,
    lag_warning_flag: str,
) -> Tuple[List[Any], Dict[str, Any]]:
    asof = _ensure_aware(resp.lineage.asof_timestamp)
    flags: Dict[str, Any] = {}
    if asof > cutoff:
        flags["response_asof_after_cutoff"] = True
        return [], flags
    if event_date_attr is None:
        return list(rows), flags

    eligible: List[Any] = []
    lag_applied = 0
    explicit_timestamp_rows = 0
    explicit_timestamp_excluded = 0
    explicit_timestamp_fields: List[str] = []
    lag_excluded = 0
    unprovable = 0
    for row in rows:
        explicit_result = _polygon_row_availability_timestamp(
            row,
            availability_fields,
        )
        if explicit_result is not None:
            explicit_availability, explicit_field = explicit_result
            explicit_timestamp_rows += 1
            if explicit_field not in explicit_timestamp_fields:
                explicit_timestamp_fields.append(explicit_field)
            if explicit_availability <= cutoff:
                eligible.append(row)
            else:
                explicit_timestamp_excluded += 1
            continue

        event_date = _date_attr_or_none(row, event_date_attr)
        if event_date is None:
            unprovable += 1
            continue
        lag_applied += 1
        available_on = _add_us_equity_sessions_after(event_date, lag_trading_days)
        if available_on <= cutoff.date():
            eligible.append(row)
        else:
            lag_excluded += 1

    if lag_applied:
        flags[lag_warning_flag] = True
        flags["availability_lag_applied"] = True
        flags["availability_lag_applied_rows"] = lag_applied
        flags["availability_lag_trading_days"] = lag_trading_days
    if explicit_timestamp_rows:
        flags["availability_timestamp_rows"] = explicit_timestamp_rows
        flags["availability_timestamp_field"] = (
            explicit_timestamp_fields[0]
            if len(explicit_timestamp_fields) == 1
            else list(explicit_timestamp_fields)
        )
    if explicit_timestamp_excluded:
        flags["availability_timestamp_future_rows"] = explicit_timestamp_excluded
    if lag_excluded:
        flags["availability_lag_excluded_rows"] = lag_excluded
    if unprovable:
        flags["availability_unprovable_rows"] = unprovable
    return eligible, flags


def _polygon_row_availability_timestamp(
    row: Any,
    fields: Sequence[str],
) -> Optional[Tuple[datetime, str]]:
    for field in fields:
        value = _row_or_raw_value(row, field)
        parsed = _datetime_or_none(value)
        if parsed is not None:
            return parsed, field
        if isinstance(value, int):
            try:
                return datetime.fromtimestamp(value, tz=timezone.utc), field
            except (OverflowError, OSError, ValueError):
                continue
    return None


def _row_or_raw_value(row: Any, field: str) -> Any:
    value = _attr(row, field)
    if value is not None:
        return value
    raw = _attr(row, "raw")
    if isinstance(raw, dict):
        return raw.get(field)
    return None


def _date_attr_or_none(row: Any, attr: str) -> Optional[date]:
    value = _attr(row, attr)
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _add_us_equity_sessions_after(start: date, sessions: int) -> date:
    cursor = start
    for _ in range(max(sessions, 0)):
        cursor = next_us_equity_session(cursor + timedelta(days=1))
    return cursor


def _benzinga_eligible_rows(
    rows: Sequence[Any],
    cutoff: datetime,
    attrs: Sequence[str],
) -> List[Any]:
    eligible: List[Any] = []
    for row in rows:
        knowledge_time = _knowledge_ts(row, attrs, cutoff=cutoff)
        if knowledge_time is not None and knowledge_time <= cutoff:
            eligible.append(row)
    return eligible


def _knowledge_ts(
    row: Any,
    attrs: Sequence[str],
    *,
    cutoff: datetime,
) -> Optional[datetime]:
    for attr in attrs:
        value = _attr(row, attr)
        parsed = _datetime_or_none(value)
        if parsed is not None:
            return parsed
        if isinstance(value, int):
            try:
                parsed = datetime.fromtimestamp(value, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                parsed = None
            if parsed is not None:
                return parsed
        if isinstance(value, str) and value.isdigit():
            try:
                parsed = datetime.fromtimestamp(int(value), tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                parsed = None
            if parsed is not None:
                return parsed
    return None


def _datetime_or_none(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return _ensure_aware(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed_date = date.fromisoformat(text)
        except ValueError:
            return None
        return datetime.combine(parsed_date, datetime.min.time(), tzinfo=timezone.utc)
    return _ensure_aware(parsed)


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _reusable_signal_context(value: Any, cutoff: datetime) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    if value.get("schema_version") != SOURCE_CONTEXT_VERSION:
        return False
    asof = _datetime_or_none(value.get("asof_timestamp"))
    return asof == cutoff


def validate_m4_context_breakout_buffer(value: float) -> float:
    try:
        buffer = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("signal_context_breakout_buffer must be a number") from exc
    if not math.isfinite(buffer) or buffer < 0 or buffer >= 1:
        raise ValueError("signal_context_breakout_buffer must be >= 0 and < 1")
    return buffer


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _m4_setup_identity_hash(inp: PatternInput) -> Optional[str]:
    high_52w = inp.market_data.get("high_52w")
    if high_52w is None:
        return None
    try:
        rounded_high = round(float(high_52w), 6)
    except (TypeError, ValueError):
        return None
    components = {
        "pattern_id": "M4",
        "ticker": inp.ticker.upper(),
        "high_52w": rounded_high,
        "high_52w_date": inp.market_data.get("high_52w_date"),
    }
    return stable_hash({
        key: value for key, value in components.items()
        if value is not None and value != ""
    })


def _feature_signal_context(feature: Optional[FeatureSnapshot]) -> Optional[Dict[str, Any]]:
    payload = _feature_payload(feature)
    context = payload.get("signal_context") if isinstance(payload, dict) else None
    return context if isinstance(context, dict) else None


def _feature_matches_m4_setup(
    feature: Optional[FeatureSnapshot],
    detector_identity_hash: str,
) -> bool:
    payload = _feature_payload(feature)
    if not isinstance(payload, dict):
        return False
    if payload.get("detector_signal_identity_hash") == detector_identity_hash:
        return True
    components = payload.get("signal_identity_components")
    if isinstance(components, dict):
        return components.get("detector_signal_identity_hash") == detector_identity_hash
    return False


def _feature_payload(feature: Optional[FeatureSnapshot]) -> Optional[Dict[str, Any]]:
    if feature is None:
        return None
    try:
        payload = json.loads(feature.feature_json or "{}")
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _context_mismatch_reason(context: Optional[Dict[str, Any]], cutoff: datetime) -> str:
    if not isinstance(context, dict) or not context:
        return "missing_signal_context"
    if context.get("schema_version") != SOURCE_CONTEXT_VERSION:
        return "schema_mismatch"
    if _datetime_or_none(context.get("asof_timestamp")) != cutoff:
        return "asof_mismatch"
    return "not_reusable"


def _increment_reason(metrics: Dict[str, Any], reason: str) -> None:
    reasons = metrics.setdefault("context_persistence_mismatch_reasons", {})
    reasons[reason] = reasons.get(reason, 0) + 1


def _context_lineage_ids(value: Any) -> List[str]:
    ids: List[str] = []
    if isinstance(value, dict):
        lineage_id = value.get("lineage_id")
        if isinstance(lineage_id, str) and lineage_id:
            ids.append(lineage_id)
        for nested in value.values():
            ids.extend(_context_lineage_ids(nested))
    elif isinstance(value, list):
        for item in value:
            ids.extend(_context_lineage_ids(item))
    return list(dict.fromkeys(ids))


def _latest_by_date(rows: Sequence[Any], attr: str) -> Optional[Any]:
    def key(row: Any) -> date:
        value = _attr(row, attr)
        try:
            return date.fromisoformat(value) if isinstance(value, str) else date.min
        except ValueError:
            return date.min

    return max(rows, key=key) if rows else None


def _latest_by_datetime(
    rows: Sequence[Any],
    attrs: Sequence[str],
    *,
    cutoff: datetime,
) -> Optional[Any]:
    def key(row: Any) -> datetime:
        return _knowledge_ts(row, attrs, cutoff=cutoff) or datetime.min.replace(tzinfo=timezone.utc)

    return max(rows, key=key) if rows else None


def _latest_by_attr(rows: Sequence[Any], attr: str) -> Optional[Any]:
    def key(row: Any) -> Any:
        value = _attr(row, attr)
        return value if value is not None else -1

    return max(rows, key=key) if rows else None


def _count_since(
    rows: Sequence[Any],
    attrs: Sequence[str],
    start: date,
    cutoff: datetime,
) -> int:
    count = 0
    for row in rows:
        knowledge_time = _knowledge_ts(row, attrs, cutoff=cutoff)
        if knowledge_time and knowledge_time.date() >= start:
            count += 1
    return count


def _calendar_summary(key: str, rows: Sequence[Any], eligible: Sequence[Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "row_count": len(eligible),
        "pit_excluded_count": max(len(rows) - len(eligible), 0),
        "event_dates": [
            _fields(row, ("date", "ex_dividend_date", "payable_date", "record_date"))
            for row in rows
        ],
    }
    if key == "earnings":
        latest = _latest_by_datetime(eligible, ("updated",), cutoff=datetime.max.replace(tzinfo=timezone.utc))
        if latest is not None:
            summary["latest"] = _fields(latest, (
                "date",
                "time",
                "eps_surprise",
                "eps_surprise_percent",
                "revenue_surprise",
                "revenue_surprise_percent",
                "updated",
            ))
    elif key == "guidance":
        summary["guidance_count"] = len(eligible)
        if eligible:
            summary["latest"] = _fields(eligible[-1], (
                "date",
                "eps_guidance_est",
                "eps_guidance_min",
                "eps_guidance_max",
                "revenue_guidance_est",
                "updated",
            ))
    elif key == "ratings":
        upgrades = 0
        downgrades = 0
        for row in eligible:
            action = (_attr(row, "action_company") or "").lower()
            if "upgrade" in action:
                upgrades += 1
            if "downgrade" in action:
                downgrades += 1
        summary["upgrade_count"] = upgrades
        summary["downgrade_count"] = downgrades
        if eligible:
            summary["latest"] = _fields(eligible[-1], (
                "date",
                "firm",
                "action_company",
                "action_pt",
                "rating_current",
                "rating_prior",
                "pt_current",
                "pt_prior",
                "pt_pct_change",
                "updated",
            ))
    elif key == "offerings":
        summary["offering_count"] = len(eligible)
        summary["dilution_flag"] = bool(eligible)
        if eligible:
            summary["latest"] = _fields(eligible[-1], (
                "date",
                "offering_type",
                "price",
                "number_shares",
                "proceeds",
                "updated",
            ))
    elif key == "dividends":
        summary["dividend_count"] = len(eligible)
        if eligible:
            summary["latest"] = _fields(eligible[-1], (
                "date",
                "ex_dividend_date",
                "dividend",
                "dividend_prior",
                "dividend_yield",
                "frequency",
                "updated",
            ))
    return summary


def _insider_transaction_summary(transactions: Sequence[Any]) -> Dict[str, Any]:
    codes = sorted({
        str(_attr(row, "transaction_code")).upper()
        for row in transactions
        if _attr(row, "transaction_code")
    })
    routine_dispositions = 0
    discretionary_buys = 0
    discretionary_sells = 0
    net_discretionary_shares = Decimal("0")
    net_discretionary_value = Decimal("0")

    for row in transactions:
        code = str(_attr(row, "transaction_code") or "").upper()
        acquired_or_disposed = str(_attr(row, "acquired_or_disposed") or "").upper()
        shares = _decimal_value(_attr(row, "shares"))
        price = _decimal_value(_attr(row, "price_per_share"))
        if code in {"F", "W"}:
            routine_dispositions += 1
            continue
        if code == "P" or (code == "A" and acquired_or_disposed == "A"):
            discretionary_buys += 1
            net_discretionary_shares += shares
            net_discretionary_value += shares * price
        elif code == "S" or (code == "D" and acquired_or_disposed == "D"):
            discretionary_sells += 1
            net_discretionary_shares -= shares
            net_discretionary_value -= shares * price

    return {
        "transaction_codes_present": codes,
        "routine_disposition_count": routine_dispositions,
        "discretionary_buy_count": discretionary_buys,
        "discretionary_sell_count": discretionary_sells,
        "net_discretionary_shares": str(net_discretionary_shares),
        "net_discretionary_value": str(net_discretionary_value),
    }


def _sentiment_summary(insights: Any) -> List[Dict[str, Any]]:
    if not isinstance(insights, list):
        return []
    summary: List[Dict[str, Any]] = []
    for item in insights:
        if not isinstance(item, dict):
            continue
        summary.append({
            "ticker": item.get("ticker"),
            "sentiment": item.get("sentiment"),
            "sentiment_reasoning": item.get("sentiment_reasoning"),
        })
    return summary


def _fields(row: Any, names: Iterable[str]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for name in names:
        value = _attr(row, name)
        if value is not None:
            data[name] = _json_safe(value)
    return data


def _attr(row: Any, name: str) -> Any:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def _has_attr(row: Any, name: str) -> bool:
    return hasattr(row, name) or (isinstance(row, dict) and name in row)


def _decimal_value(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _json_safe(vars(value))
    return value


def _all_attempts(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    for value in context.values():
        if isinstance(value, dict):
            source_attempts = value.get("source_attempts")
            if isinstance(source_attempts, list):
                attempts.extend(
                    item for item in source_attempts if isinstance(item, dict)
                )
    return attempts
