"""
M4 52-Week High Breakout daily assembler.

Builds daily/base M4 PatternInput objects from canonical universe snapshots
and daily bar history. Computes high_52w, high_52w_date, n_sessions_in_window.

Insufficient history produces explicit diagnostics, not zero-filled data.
Missing data is never coerced to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from alpha.assembly.framework import (
    AssembledField,
    AssemblyDiagnostic,
    FieldPresence,
    PatternAssemblyResult,
    build_pattern_input,
    validate_assembled_fields,
)
from alpha.patterns.contracts import PatternId

PATTERN_ID = PatternId.M4
SESSIONS_52W = 252
MIN_SESSIONS = 1


@dataclass
class DailyBar:
    """A single daily OHLCV bar."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    split_adjusted_close: Optional[float] = None
    adj_close: Optional[float] = None
    source_timestamp: Optional[datetime] = None
    source_provider: Optional[str] = None
    lineage_id: Optional[str] = None
    lineage_hash: Optional[str] = None


def _compute_52w_high(
    bars: List[DailyBar],
) -> tuple:
    """Compute 52-week high, high date, and session count from bars.

    Returns (high_52w, high_52w_date, n_sessions).
    Bars should be sorted ascending by date and already filtered to
    the relevant window. The M4 production convention uses split-adjusted
    traded close, not raw intraday high or dividend-adjusted close.
    """
    if not bars:
        return None, None, 0
    high_val = bars[0].split_adjusted_close
    high_date = bars[0].date
    for bar in bars[1:]:
        if bar.split_adjusted_close > high_val:
            high_val = bar.split_adjusted_close
            high_date = bar.date
    return high_val, high_date, len(bars)


def assemble_m4_daily(
    *,
    snapshots: List[Any],
    daily_bars: Dict[str, List[DailyBar]],
    cutoff_timestamp: datetime,
    universe_cutoff_timestamp: Optional[datetime] = None,
    trading_date: Optional[str] = None,
    decision_date: Optional[str] = None,
    evidence_session_date: Optional[str] = None,
    source_provider: str = "FMP",
    source_lineage_hash: Optional[str] = None,
) -> PatternAssemblyResult:
    """Assemble daily M4 PatternInput objects from universe snapshots and bar history.

    Parameters
    ----------
    snapshots : list
        Included universe snapshots (ORM objects or dicts with ticker, price,
        market_cap, primary_exchange, security_type, universe_snapshot_id,
        asof_timestamp, source_lineage_hash, operating_universe_inclusion).
    daily_bars : dict
        Mapping from ticker to list of DailyBar objects, sorted ascending by date.
        Bars should cover the evidence session and the prior 252 sessions.
    decision_date : str
        ISO date for the engine decision/run date.
    evidence_session_date : str
        ISO date of the completed market session whose close is being evaluated.
        high_52w is computed from sessions strictly before this date.
    trading_date : str, optional
        Backward-compatible alias used as decision_date/evidence_session_date when
        the explicit session dates are not supplied.
    cutoff_timestamp : datetime
        Daily-bar lookahead cutoff — bar field source timestamps must be at or before this.
    universe_cutoff_timestamp : datetime, optional
        Universe/snapshot field cutoff. Defaults to cutoff_timestamp for backward
        compatibility. Production daily M4 uses evidence-session cutoff for bars
        and canonical-scan as-of for universe membership fields.
    source_provider : str
        Provider name for lineage.
    source_lineage_hash : str, optional
        Shared lineage hash for the bar data source.
    """
    resolved_decision_date = decision_date or trading_date
    resolved_evidence_date = evidence_session_date or trading_date or decision_date
    if resolved_decision_date is None or resolved_evidence_date is None:
        raise ValueError(
            "assemble_m4_daily requires decision_date/evidence_session_date "
            "or backward-compatible trading_date"
        )

    result = PatternAssemblyResult(pattern_id=PATTERN_ID)
    evidence_day = date.fromisoformat(resolved_evidence_date)
    resolved_universe_cutoff = universe_cutoff_timestamp or cutoff_timestamp

    for snap in snapshots:
        ticker = _snap_attr(snap, "ticker")
        snap_id = _snap_attr(snap, "universe_snapshot_id")
        asof = _snap_attr(snap, "asof_timestamp")
        snap_lineage = _snap_attr(snap, "source_lineage_hash")
        snapshot_price = _snap_attr(snap, "price")

        bars = sorted(daily_bars.get(ticker, []), key=lambda b: b.date)
        prior_bars: List[DailyBar] = []
        future_dated_rejections: List[AssembledField] = []
        evidence_bar: Optional[DailyBar] = None
        for bar in bars:
            bar_day = date.fromisoformat(bar.date)
            if bar_day > evidence_day:
                future_dated_rejections.append(AssembledField(
                    name="daily_bar",
                    value=bar.date,
                    presence=FieldPresence.REJECTED_LOOKAHEAD,
                    source_timestamp=bar.source_timestamp,
                    allowed_cutoff=cutoff_timestamp,
                    source_provider=bar.source_provider or source_provider,
                    lineage_hash=bar.lineage_hash or source_lineage_hash,
                    rejection_reason=(
                        f"bar.date {bar.date} after evidence_session_date "
                        f"{resolved_evidence_date}"
                    ),
                ))
                continue
            if bar_day == evidence_day:
                evidence_bar = bar
                continue
            if bar_day < evidence_day:
                prior_bars.append(bar)

        window_bars = prior_bars[-SESSIONS_52W:]
        missing_adjusted_bars = [
            bar for bar in window_bars
            if bar.split_adjusted_close is None
        ]
        evidence_missing_adjusted = (
            evidence_bar is not None
            and evidence_bar.split_adjusted_close is None
        )

        fields: List[AssembledField] = []
        lineage_hashes: List[str] = []
        lineage_ids: List[str] = []

        if snap_lineage:
            _append_unique(lineage_hashes, snap_lineage)

        if evidence_bar is not None:
            _append_unique(lineage_ids, evidence_bar.lineage_id)
            _append_unique(
                lineage_hashes,
                evidence_bar.lineage_hash or source_lineage_hash,
            )

        # M4 price is the completed evidence-session split-adjusted close.
        if (
            evidence_bar is not None
            and evidence_bar.split_adjusted_close is not None
        ):
            evidence_ts = _ensure_aware(evidence_bar.source_timestamp)
            fields.append(AssembledField(
                name="price", value=evidence_bar.split_adjusted_close,
                presence=FieldPresence.PRESENT,
                source_timestamp=evidence_ts,
                allowed_cutoff=cutoff_timestamp,
                source_provider=evidence_bar.source_provider or source_provider,
                lineage_hash=evidence_bar.lineage_hash or source_lineage_hash,
            ))
            fields.append(AssembledField(
                name="price_source", value="evidence_session_split_adjusted_close",
                presence=FieldPresence.PRESENT,
                source_timestamp=evidence_ts,
                allowed_cutoff=cutoff_timestamp,
                source_provider=evidence_bar.source_provider or source_provider,
                lineage_hash=evidence_bar.lineage_hash or source_lineage_hash,
            ))
            fields.append(AssembledField(
                name="evidence_close", value=evidence_bar.close,
                presence=FieldPresence.PRESENT,
                source_timestamp=evidence_ts,
                allowed_cutoff=cutoff_timestamp,
                source_provider=evidence_bar.source_provider or source_provider,
                lineage_hash=evidence_bar.lineage_hash or source_lineage_hash,
            ))
            fields.append(AssembledField(
                name="evidence_split_adjusted_close",
                value=evidence_bar.split_adjusted_close,
                presence=FieldPresence.PRESENT,
                source_timestamp=evidence_ts,
                allowed_cutoff=cutoff_timestamp,
                source_provider=evidence_bar.source_provider or source_provider,
                lineage_hash=evidence_bar.lineage_hash or source_lineage_hash,
            ))
        else:
            price_reason = (
                "split_adjusted_close_unavailable"
                if evidence_missing_adjusted else "evidence_session_bar_unavailable"
            )
            fields.append(AssembledField(
                name="price", value=None,
                presence=FieldPresence.UNAVAILABLE,
                allowed_cutoff=cutoff_timestamp,
                source_provider=source_provider,
                rejection_reason=price_reason,
            ))
        if snapshot_price is not None:
            fields.append(AssembledField(
                name="snapshot_price", value=snapshot_price,
                presence=FieldPresence.PRESENT,
                source_timestamp=_ensure_aware(asof),
                allowed_cutoff=resolved_universe_cutoff,
                source_provider=source_provider,
                lineage_hash=snap_lineage,
            ))

        for bar in window_bars:
            _append_unique(lineage_ids, bar.lineage_id)
            _append_unique(lineage_hashes, bar.lineage_hash or source_lineage_hash)

        # 52-week high from prior-session split-adjusted closes only.
        if window_bars and not missing_adjusted_bars:
            bar_lineage = window_bars[-1].lineage_hash or source_lineage_hash
            high_52w, high_52w_date, n_sessions = _compute_52w_high(window_bars)
            bar_source_ts = (
                _max_source_timestamp(window_bars)
                or _ensure_aware(asof)
            )

            fields.append(AssembledField(
                name="high_52w", value=high_52w,
                presence=FieldPresence.PRESENT,
                source_timestamp=bar_source_ts,
                allowed_cutoff=cutoff_timestamp,
                source_provider=window_bars[-1].source_provider or source_provider,
                lineage_hash=bar_lineage,
            ))
            if high_52w_date is not None:
                fields.append(AssembledField(
                    name="high_52w_date", value=high_52w_date,
                    presence=FieldPresence.PRESENT,
                    source_timestamp=bar_source_ts,
                    allowed_cutoff=cutoff_timestamp,
                    source_provider=window_bars[-1].source_provider or source_provider,
                    lineage_hash=bar_lineage,
                ))
            fields.append(AssembledField(
                name="n_sessions_in_window", value=n_sessions,
                presence=FieldPresence.PRESENT,
                source_timestamp=bar_source_ts,
                allowed_cutoff=cutoff_timestamp,
                source_provider=window_bars[-1].source_provider or source_provider,
                lineage_hash=bar_lineage,
            ))
            fields.append(AssembledField(
                name="high_52w_basis",
                value="split_adjusted_close_prior_252_sessions",
                presence=FieldPresence.PRESENT,
                source_timestamp=bar_source_ts,
                allowed_cutoff=cutoff_timestamp,
                source_provider=window_bars[-1].source_provider or source_provider,
                lineage_hash=bar_lineage,
            ))
            fields.append(AssembledField(
                name="lookback_start", value=window_bars[0].date,
                presence=FieldPresence.PRESENT,
                source_timestamp=bar_source_ts,
                allowed_cutoff=cutoff_timestamp,
                source_provider=window_bars[-1].source_provider or source_provider,
                lineage_hash=bar_lineage,
            ))
            fields.append(AssembledField(
                name="lookback_end", value=window_bars[-1].date,
                presence=FieldPresence.PRESENT,
                source_timestamp=bar_source_ts,
                allowed_cutoff=cutoff_timestamp,
                source_provider=window_bars[-1].source_provider or source_provider,
                lineage_hash=bar_lineage,
            ))
        else:
            rejection_reason = (
                "daily_bars_rejected_lookahead"
                if bars and future_dated_rejections else
                "split_adjusted_close_unavailable"
                if missing_adjusted_bars else "no_daily_bars_available"
            )
            fields.append(AssembledField(
                name="high_52w", value=None,
                presence=FieldPresence.UNAVAILABLE,
                allowed_cutoff=cutoff_timestamp,
                source_provider=source_provider,
                rejection_reason=rejection_reason,
            ))
            fields.append(AssembledField(
                name="n_sessions_in_window", value=0,
                presence=FieldPresence.UNAVAILABLE,
                allowed_cutoff=cutoff_timestamp,
                source_provider=source_provider,
                rejection_reason=rejection_reason,
            ))

        # Universe membership / classification fields
        for field_name in (
            "operating_universe_inclusion",
            "security_type",
            "primary_exchange",
        ):
            val = _snap_attr(snap, field_name)
            if val is not None:
                fields.append(AssembledField(
                    name=field_name, value=val,
                    presence=FieldPresence.PRESENT,
                    source_timestamp=_ensure_aware(asof),
                    allowed_cutoff=resolved_universe_cutoff,
                    source_provider=source_provider,
                    lineage_hash=snap_lineage,
                ))

        fields.append(AssembledField(
            name="decision_date", value=resolved_decision_date,
            presence=FieldPresence.PRESENT,
        ))
        fields.append(AssembledField(
            name="evidence_session_date", value=resolved_evidence_date,
            presence=FieldPresence.PRESENT,
        ))

        # Backward-compatible alias for existing detector inputs. New callers
        # should consume decision_date/evidence_session_date explicitly.
        fields.append(AssembledField(
            name="trading_date", value=resolved_decision_date,
            presence=FieldPresence.PRESENT,
        ))

        # Framework-level lookahead validation
        validated, rejected = validate_assembled_fields(fields, cutoff_timestamp)
        all_rejected = future_dated_rejections + rejected
        lookahead_rejections = [
            rf for rf in all_rejected
            if rf.presence == FieldPresence.REJECTED_LOOKAHEAD
        ]

        # Check for required fields after validation
        has_price = "price" in validated
        has_high = "high_52w" in validated
        n_sessions = validated.get("n_sessions_in_window", 0)

        if not has_price:
            diagnostic_type = (
                "field_rejected_lookahead"
                if lookahead_rejections
                else "split_adjusted_close_unavailable"
                if evidence_missing_adjusted
                else "evidence_session_bar_unavailable"
            )
            detail = (
                "evidence-session split-adjusted close missing"
                if evidence_missing_adjusted
                else f"no bar found for evidence_session_date {resolved_evidence_date}"
            )
            if diagnostic_type == "field_rejected_lookahead":
                detail = (
                    f"{lookahead_rejections[0].name}: "
                    f"{lookahead_rejections[0].rejection_reason}"
                )
            result.diagnostics.append(AssemblyDiagnostic(
                ticker=ticker, pattern_id=PATTERN_ID,
                diagnostic_type=diagnostic_type,
                detail=detail,
            ))
            result.rejected_count += 1
            result.rejected_fields.extend(all_rejected)
            continue

        if not has_high:
            if lookahead_rejections:
                for rf in lookahead_rejections:
                    result.diagnostics.append(AssemblyDiagnostic(
                        ticker=ticker, pattern_id=PATTERN_ID,
                        diagnostic_type="field_rejected_lookahead",
                        detail=f"{rf.name}: {rf.rejection_reason}",
                    ))
                result.rejected_count += 1
            else:
                diagnostic_type = (
                    "split_adjusted_close_unavailable"
                    if missing_adjusted_bars else "insufficient_history"
                )
                detail = (
                    f"{len(missing_adjusted_bars)} lookback bars lack "
                    "split_adjusted_close; refusing dividend-adjusted/raw "
                    "52-week high"
                    if missing_adjusted_bars
                    else "52-week high unavailable — no daily bar history"
                )
                result.diagnostics.append(AssemblyDiagnostic(
                    ticker=ticker, pattern_id=PATTERN_ID,
                    diagnostic_type=diagnostic_type,
                    detail=detail,
                ))
                if missing_adjusted_bars:
                    result.rejected_count += 1
                else:
                    result.insufficient_count += 1
            result.rejected_fields.extend(all_rejected)
            continue

        if n_sessions < MIN_SESSIONS:
            result.diagnostics.append(AssemblyDiagnostic(
                ticker=ticker, pattern_id=PATTERN_ID,
                diagnostic_type="insufficient_history",
                detail=f"only {n_sessions} sessions available, need >= {MIN_SESSIONS}",
            ))
            result.insufficient_count += 1
            result.rejected_fields.extend(all_rejected)
            continue

        if n_sessions < SESSIONS_52W:
            result.diagnostics.append(AssemblyDiagnostic(
                ticker=ticker, pattern_id=PATTERN_ID,
                diagnostic_type="short_history",
                detail=(
                    f"only {n_sessions} prior sessions available; "
                    f"using IPO/short-history M4 path"
                ),
            ))

        # Any lookahead rejections are tracked but don't block the input
        # if the required fields survived validation
        if all_rejected:
            for rf in lookahead_rejections:
                result.diagnostics.append(AssemblyDiagnostic(
                    ticker=ticker, pattern_id=PATTERN_ID,
                    diagnostic_type="field_rejected_lookahead",
                    detail=f"{rf.name}: {rf.rejection_reason}",
                ))
            result.rejected_fields.extend(all_rejected)

        inp = build_pattern_input(
            ticker=ticker,
            pattern_id=PATTERN_ID,
            asof_timestamp=_ensure_aware(asof) or cutoff_timestamp,
            validated_fields=validated,
            lineage_ids=lineage_ids,
            lineage_hashes=lineage_hashes,
            universe_snapshot_id=snap_id,
        )
        result.inputs.append(inp)
        result.assembled_count += 1

    return result


def _snap_attr(snap: Any, name: str) -> Any:
    """Get attribute from ORM object or dict."""
    if isinstance(snap, dict):
        return snap.get(name)
    return getattr(snap, name, None)


def _ensure_aware(dt: Any) -> Optional[datetime]:
    """Ensure a datetime is timezone-aware UTC."""
    if dt is None:
        return None
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _append_unique(values: List[str], value: Optional[str]) -> None:
    if value and value not in values:
        values.append(value)


def _max_source_timestamp(bars: List[DailyBar]) -> Optional[datetime]:
    timestamps = [
        _ensure_aware(bar.source_timestamp)
        for bar in bars
        if _ensure_aware(bar.source_timestamp) is not None
    ]
    if not timestamps:
        return None
    return max(timestamps)
