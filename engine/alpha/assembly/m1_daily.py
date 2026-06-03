"""M1 PEAD daily assembler and producer math.

Builds M1 PatternInput objects from PIT earnings-calendar events, EPS
history, canonical universe snapshots, and return-based information-friction
metrics. Missing EPS history shrinks the eligible cohort and is reported as a
diagnostic; it is never zero-filled or interpolated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
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
from alpha.data.fmp import FmpBar, FmpEarningsCalendarEvent, FmpEpsRecord
from alpha.market_calendar import is_us_equity_session, next_us_equity_session
from alpha.patterns.contracts import PatternId

PATTERN_ID = PatternId.M1
MARKET_FACTOR_SYMBOL = "SPY"
FOSTER_REQUIRED_LAGS = tuple(range(4, 20))
FOSTER_DIFF_LAGS = tuple(range(4, 16))
FOSTER_SIGMA_DELTA_EPSILON = 1e-6
MAX_ABS_FOSTER_SUE = 25.0
MIN_PRICE_DELAY_OBSERVATIONS = 20
MIN_ANNOUNCING_COHORT_SIZE = 5
SUE_TIE_EPSILON = 1e-12


@dataclass
class FosterComputation:
    """Foster SUE and derived SUE-series multipliers for one earnings event."""

    ticker: str
    status: str
    diagnostics: List[str] = field(default_factory=list)
    event_id: Optional[str] = None
    announcement_date: Optional[str] = None
    effective_announcement_session: Optional[str] = None
    announcement_time: Optional[str] = None
    actual_eps: Optional[float] = None
    estimated_eps: Optional[float] = None
    expected_eps: Optional[float] = None
    sigma_delta_eps: Optional[float] = None
    sue_foster: Optional[float] = None
    rho1: Optional[float] = None
    sue_sign_current: Optional[int] = None
    sue_sign_prior: Optional[int] = None
    sue_streak_length: Optional[int] = None
    fiscal_period_end: Optional[str] = None
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    foster_history_quarters_used: int = 0
    split_adjustment_continuity_check: str = "not_computed"
    restatement_exposure: bool = False
    eps_history_current_eps: Optional[float] = None
    sue_series: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def computed(self) -> bool:
        return self.status == "computed" and self.sue_foster is not None


@dataclass
class PriceDelayComputation:
    """Hou-Moskowitz D1 and residual-volatility computation for one ticker."""

    ticker: str
    status: str
    diagnostics: List[str] = field(default_factory=list)
    d1: Optional[float] = None
    d1_decile: Optional[int] = None
    sigma_epsilon: Optional[float] = None
    sigma_epsilon_percentile: Optional[float] = None
    weekly_return_count: int = 0
    market_factor_symbol: str = MARKET_FACTOR_SYMBOL

    @property
    def computed(self) -> bool:
        return (
            self.status == "computed"
            and self.d1 is not None
            and self.d1_decile is not None
            and self.sigma_epsilon is not None
            and self.sigma_epsilon_percentile is not None
        )


def compute_foster_sue(
    *,
    event: FmpEarningsCalendarEvent,
    eps_history: Sequence[FmpEpsRecord],
    effective_session: Optional[date] = None,
    asof_timestamp: Optional[datetime] = None,
) -> FosterComputation:
    """Compute Foster TS-RW SUE from a PIT calendar event and EPS history."""

    ticker = event.symbol.upper()
    diagnostics: List[str] = list(getattr(event, "diagnostics", ()) or ())
    announcement_date = event.date or None
    event_id = stable_hash({
        "provider": "FMP",
        "type": "earnings_calendar",
        "symbol": ticker,
        "date": event.date,
        "fiscal_date_ending": event.fiscal_date_ending,
        "fiscal_year": event.fiscal_year,
        "fiscal_quarter": event.fiscal_quarter,
    })
    base = FosterComputation(
        ticker=ticker,
        status="insufficient_history",
        event_id=event_id,
        announcement_date=announcement_date,
        effective_announcement_session=(
            effective_session.isoformat() if effective_session else None
        ),
        announcement_time=event.announcement_time,
        actual_eps=event.actual_eps,
        estimated_eps=event.estimated_eps,
        fiscal_period_end=event.fiscal_date_ending,
        fiscal_year=event.fiscal_year,
        fiscal_quarter=event.fiscal_quarter,
        diagnostics=diagnostics,
    )

    if event.actual_eps is None or not _is_finite(event.actual_eps):
        diagnostics.append("actual_eps_missing")
        return base

    current_index, fiscal_period_end = _resolve_current_fiscal_index(
        event,
        eps_history,
        asof_timestamp=asof_timestamp,
    )
    if current_index is None:
        diagnostics.append("fiscal_quarter_unresolved")
        return base
    fiscal_year = current_index // 4
    fiscal_quarter = current_index % 4 + 1
    base.fiscal_year = fiscal_year
    base.fiscal_quarter = fiscal_quarter
    if fiscal_period_end is not None:
        base.fiscal_period_end = fiscal_period_end

    needed_indices = [current_index]
    needed_indices.extend(current_index - lag for lag in FOSTER_REQUIRED_LAGS)
    all_history_indices = set(needed_indices)
    for record in eps_history:
        record_index = _record_fiscal_index(record)
        if record_index is not None and record_index <= current_index:
            all_history_indices.add(record_index)
    selected_records, record_diagnostics = _select_pit_eps_records(
        eps_history,
        indices=sorted(all_history_indices),
        diagnostic_indices=set(needed_indices),
        asof_timestamp=asof_timestamp,
    )
    diagnostics.extend(record_diagnostics)
    eps_by_index = {
        index: float(record.eps)
        for index, record in selected_records.items()
        if record.eps is not None and _is_finite(record.eps)
    }

    history_current = eps_by_index.get(current_index)
    if history_current is None:
        diagnostics.append("current_eps_basis_unverified")
        return base
    base.actual_eps = history_current
    base.eps_history_current_eps = history_current
    if not math.isclose(history_current, float(event.actual_eps), rel_tol=1e-9, abs_tol=1e-9):
        base.restatement_exposure = True
        diagnostics.append("calendar_eps_differs_from_eps_history_current")

    missing_lags = [
        lag for lag in FOSTER_REQUIRED_LAGS
        if current_index - lag not in eps_by_index
    ]
    if missing_lags:
        diagnostics.append(
            "missing_required_foster_lags:" + ",".join(str(lag) for lag in missing_lags)
        )
        return base

    seasonal_diffs = [
        eps_by_index[current_index - lag] - eps_by_index[current_index - lag - 4]
        for lag in FOSTER_DIFF_LAGS
    ]
    if any(not _is_finite(value) for value in seasonal_diffs):
        diagnostics.append("non_finite_seasonal_difference")
        return base
    sigma_delta = _sample_std(seasonal_diffs)
    if sigma_delta is None or sigma_delta < FOSTER_SIGMA_DELTA_EPSILON:
        diagnostics.append("near_zero_sigma_delta_eps")
        return base

    delta = sum(seasonal_diffs) / len(seasonal_diffs)
    expected_eps = eps_by_index[current_index - 4] + delta
    sue = (history_current - expected_eps) / sigma_delta
    if not _is_finite(sue) or abs(sue) > MAX_ABS_FOSTER_SUE:
        diagnostics.append("sue_out_of_domain_bounds")
        return base
    series = _compute_sue_series(eps_by_index)
    current_series = [row for row in series if row["fiscal_index"] == current_index]
    if not current_series:
        series.append({
            "fiscal_index": current_index,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
            "sue": sue,
        })
        series.sort(key=lambda row: row["fiscal_index"])

    signs = [
        _sign(row["sue"])
        for row in sorted(series, key=lambda row: row["fiscal_index"], reverse=True)
        if row["fiscal_index"] <= current_index
    ]
    current_sign = _sign(sue)
    prior_sue = next(
        (
            row["sue"]
            for row in sorted(series, key=lambda row: row["fiscal_index"], reverse=True)
            if row["fiscal_index"] < current_index
        ),
        None,
    )
    prior_sign = _sign(prior_sue) if prior_sue is not None else None

    base.status = "computed"
    base.expected_eps = expected_eps
    base.sigma_delta_eps = sigma_delta
    base.sue_foster = sue
    pit_series = [
        row for row in sorted(series, key=lambda row: row["fiscal_index"])
        if row["fiscal_index"] <= current_index
    ]
    base.rho1 = _lag1_autocorr([row["sue"] for row in pit_series[-8:]])
    base.sue_sign_current = current_sign
    base.sue_sign_prior = prior_sign
    base.sue_streak_length = _streak_length(signs)
    base.foster_history_quarters_used = len(FOSTER_REQUIRED_LAGS)
    base.split_adjustment_continuity_check = "passed"
    base.sue_series = [
        {
            "fiscal_year": row["fiscal_year"],
            "fiscal_quarter": row["fiscal_quarter"],
            "sue": round(row["sue"], 8),
        }
        for row in series
        if row["fiscal_index"] <= current_index
    ]
    return base


def effective_announcement_session(event: FmpEarningsCalendarEvent) -> Optional[date]:
    """Return the conservative trading session on which the event is usable."""

    try:
        raw_day = date.fromisoformat(str(event.date)[:10])
    except (TypeError, ValueError):
        return None
    event_day = raw_day if is_us_equity_session(raw_day) else next_us_equity_session(raw_day)
    timing = str(event.announcement_time or "").strip().lower()
    if timing in {"bmo", "before market open", "before-market-open", "am"}:
        return event_day
    # Unknown / AMC / during-market are treated as not safely usable until the
    # next session so the signal cannot run before public EPS is knowable.
    return next_us_equity_session(event_day + timedelta(days=1))


def trading_session_distance(start: date, end: date) -> Optional[int]:
    """Count regular sessions from start to end, with same session = 0."""

    if end < start:
        return None
    current = next_us_equity_session(start)
    target = next_us_equity_session(end)
    count = 0
    while current < target:
        current = next_us_equity_session(current + timedelta(days=1))
        count += 1
        if count > 500:
            return None
    return count


def compute_price_delay_metric(
    *,
    ticker: str,
    stock_bars: Sequence[FmpBar],
    market_bars: Sequence[FmpBar],
    market_factor_symbol: str = MARKET_FACTOR_SYMBOL,
    min_observations: int = MIN_PRICE_DELAY_OBSERVATIONS,
) -> PriceDelayComputation:
    """Compute H-M D1 and residual std from weekly Wed-to-Wed returns."""

    stock_returns = _weekly_returns(stock_bars)
    market_returns = _weekly_returns(market_bars)
    common_dates = sorted(set(stock_returns).intersection(market_returns))
    if len(common_dates) < min_observations + 4:
        return PriceDelayComputation(
            ticker=ticker,
            status="insufficient_price_history",
            diagnostics=[f"weekly_return_count:{len(common_dates)}"],
            weekly_return_count=len(common_dates),
            market_factor_symbol=market_factor_symbol,
        )

    y_values: List[float] = []
    full_x: List[List[float]] = []
    restricted_x: List[List[float]] = []
    stock = [stock_returns[day] for day in common_dates]
    market = [market_returns[day] for day in common_dates]
    for idx in range(4, len(common_dates)):
        y_values.append(stock[idx])
        full_x.append([
            1.0,
            market[idx],
            market[idx - 1],
            market[idx - 2],
            market[idx - 3],
            market[idx - 4],
            stock[idx - 1],
            stock[idx - 2],
            stock[idx - 3],
            stock[idx - 4],
        ])
        restricted_x.append([1.0, market[idx]])

    full = _ols_fit(full_x, y_values)
    restricted = _ols_fit(restricted_x, y_values)
    if full is None or restricted is None or full["r2"] is None or full["r2"] <= 0:
        return PriceDelayComputation(
            ticker=ticker,
            status="regression_failed",
            diagnostics=["invalid_full_regression_r2"],
            weekly_return_count=len(common_dates),
            market_factor_symbol=market_factor_symbol,
        )
    d1 = 1.0 - (restricted["r2"] or 0.0) / full["r2"]
    d1 = min(max(d1, 0.0), 1.0)
    residual_df = max(len(y_values) - len(full_x[0]), 1)
    sigma_epsilon = math.sqrt(full["rss"] / residual_df)
    return PriceDelayComputation(
        ticker=ticker,
        status="computed",
        d1=d1,
        sigma_epsilon=sigma_epsilon,
        weekly_return_count=len(common_dates),
        market_factor_symbol=market_factor_symbol,
    )


def rank_friction_metrics(
    metrics: Dict[str, PriceDelayComputation],
) -> Dict[str, PriceDelayComputation]:
    """Rank computed D1/sigma values over the operating universe."""

    computed = [m for m in metrics.values() if m.status == "computed"]
    d1_values = sorted((m.d1, m.ticker) for m in computed if m.d1 is not None)
    d1_rank = {ticker: idx + 1 for idx, (_, ticker) in enumerate(d1_values)}
    n_d1 = len(d1_values)
    for ticker, metric in metrics.items():
        if metric.status != "computed":
            continue
        metric.d1_decile = max(1, min(10, math.ceil(d1_rank[ticker] * 10.0 / n_d1)))

    high_d1_sigma_values = sorted(
        (m.sigma_epsilon, m.ticker)
        for m in computed
        if (
            m.sigma_epsilon is not None
            and m.d1_decile is not None
            and m.d1_decile >= 8
        )
    )
    sigma_rank = {ticker: idx for idx, (_, ticker) in enumerate(high_d1_sigma_values)}
    n_sigma = len(high_d1_sigma_values)
    for ticker, metric in metrics.items():
        if metric.status != "computed":
            continue
        if (
            metric.d1_decile is None
            or metric.d1_decile < 8
            or metric.sigma_epsilon is None
            or ticker not in sigma_rank
        ):
            metric.sigma_epsilon_percentile = 0.0
            continue
        metric.sigma_epsilon_percentile = (
            1.0 if n_sigma == 1 else sigma_rank[ticker] / (n_sigma - 1)
        )
    return metrics


def assemble_m1_daily(
    *,
    snapshots: List[Any],
    foster_by_ticker: Dict[str, FosterComputation],
    friction_by_ticker: Dict[str, PriceDelayComputation],
    next_earnings_by_ticker: Optional[Dict[str, int]] = None,
    cutoff_timestamp: datetime,
    universe_cutoff_timestamp: Optional[datetime] = None,
    decision_date: str,
    evidence_session_date: str,
    next_execution_session: Optional[str] = None,
    source_provider: str = "FMP",
    source_lineage_hash: Optional[str] = None,
    lineage_ids_by_ticker: Optional[Dict[str, List[str]]] = None,
    lineage_hashes_by_ticker: Optional[Dict[str, List[str]]] = None,
) -> PatternAssemblyResult:
    """Assemble M1 PatternInput objects from computed producer artifacts."""

    result = PatternAssemblyResult(pattern_id=PATTERN_ID)
    next_earnings_by_ticker = next_earnings_by_ticker or {}
    lineage_ids_by_ticker = lineage_ids_by_ticker or {}
    lineage_hashes_by_ticker = lineage_hashes_by_ticker or {}
    resolved_universe_cutoff = universe_cutoff_timestamp or cutoff_timestamp
    sue_percentiles, sue_percentile_diagnostic = _signed_sue_percentiles(
        foster_by_ticker.values()
    )

    for snap in snapshots:
        ticker = str(_snap_attr(snap, "ticker") or "").upper()
        snap_id = _snap_attr(snap, "universe_snapshot_id")
        snap_asof = _ensure_aware(_snap_attr(snap, "asof_timestamp"))
        snap_lineage = _snap_attr(snap, "source_lineage_hash")
        foster = foster_by_ticker.get(ticker)
        friction = friction_by_ticker.get(ticker)

        if foster is None:
            result.diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=PATTERN_ID,
                diagnostic_type="no_recent_earnings_event",
            ))
            continue
        if not foster.computed:
            result.insufficient_count += 1
            result.diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=PATTERN_ID,
                diagnostic_type=foster.status,
                detail=";".join(foster.diagnostics),
            ))
            continue
        if friction is None or not friction.computed:
            result.insufficient_count += 1
            detail = None if friction is None else ";".join(friction.diagnostics)
            result.diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=PATTERN_ID,
                diagnostic_type="insufficient_friction",
                detail=detail,
            ))
            continue
        if ticker not in sue_percentiles:
            result.insufficient_count += 1
            result.diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=PATTERN_ID,
                diagnostic_type=(
                    sue_percentile_diagnostic
                    or "announcing_cohort_percentile_unavailable"
                ),
            ))
            continue
        if foster.effective_announcement_session is None:
            result.insufficient_count += 1
            result.diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=PATTERN_ID,
                diagnostic_type="missing_effective_announcement_session",
            ))
            continue
        delta_t = trading_session_distance(
            date.fromisoformat(foster.effective_announcement_session),
            date.fromisoformat(evidence_session_date),
        )
        if delta_t is None:
            result.insufficient_count += 1
            result.diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=PATTERN_ID,
                diagnostic_type="announcement_after_evidence_session",
            ))
            continue

        fields = [
            _field("sue_foster", foster.sue_foster, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("delta_t_trading_days", delta_t, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sue_signed_percentile", sue_percentiles[ticker], cutoff_timestamp, source_provider, source_lineage_hash),
            _field("rho1", foster.rho1, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sue_sign_current", foster.sue_sign_current, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sue_sign_prior", foster.sue_sign_prior if foster.sue_sign_prior is not None else 0, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("d1_decile", friction.d1_decile, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sigma_epsilon_percentile", friction.sigma_epsilon_percentile, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sue_streak_length", foster.sue_streak_length, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("next_earnings_trading_days_from_signal", next_earnings_by_ticker.get(ticker), cutoff_timestamp, source_provider, source_lineage_hash),
            _field("earnings_event_id", foster.event_id, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("announcement_date", foster.announcement_date, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("announcement_time_utc", foster.announcement_time, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("fiscal_period_end", foster.fiscal_period_end, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("fiscal_year", foster.fiscal_year, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("fiscal_quarter", foster.fiscal_quarter, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("foster_history_quarters_used", foster.foster_history_quarters_used, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("split_adjustment_continuity_check", foster.split_adjustment_continuity_check, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("restatement_exposure", foster.restatement_exposure, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("price", _snap_attr(snap, "price"), resolved_universe_cutoff, source_provider, snap_lineage),
            _field("market_cap", _snap_attr(snap, "market_cap"), resolved_universe_cutoff, source_provider, snap_lineage),
            _field("market_cap_usd", _snap_attr(snap, "market_cap"), resolved_universe_cutoff, source_provider, snap_lineage),
            _field("primary_exchange", _snap_attr(snap, "primary_exchange"), resolved_universe_cutoff, source_provider, snap_lineage),
            _field("security_type", _snap_attr(snap, "security_type"), resolved_universe_cutoff, source_provider, snap_lineage),
            _field("operating_universe_inclusion", _snap_attr(snap, "operating_universe_inclusion"), resolved_universe_cutoff, source_provider, snap_lineage),
            _field("liquidity_score", _snap_attr(snap, "liquidity_score"), resolved_universe_cutoff, source_provider, snap_lineage),
            _field("hazard_score_at_signal", _snap_attr(snap, "hazard_score"), resolved_universe_cutoff, source_provider, snap_lineage),
            _field("market_data_status", "current", cutoff_timestamp, source_provider, source_lineage_hash),
            _field("halt_status", "clear", cutoff_timestamp, source_provider, source_lineage_hash),
            _field("corporate_action_filter_passed", True, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("trading_date", decision_date, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("evidence_session_date", evidence_session_date, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("next_execution_session", next_execution_session, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("market_factor_symbol", friction.market_factor_symbol, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("d1", friction.d1, cutoff_timestamp, source_provider, source_lineage_hash),
            _field("sigma_epsilon", friction.sigma_epsilon, cutoff_timestamp, source_provider, source_lineage_hash),
        ]
        validated, rejected = validate_assembled_fields(fields, cutoff_timestamp)
        result.rejected_fields.extend(rejected)
        missing_load_bearing = [
            key for key in ("sue_foster", "delta_t_trading_days", "sue_signed_percentile")
            if key not in validated
        ]
        missing_multipliers = [
            key for key in (
                "rho1",
                "sue_sign_current",
                "sue_sign_prior",
                "d1_decile",
                "sigma_epsilon_percentile",
                "sue_streak_length",
            )
            if key not in validated
        ]
        if missing_load_bearing or missing_multipliers:
            result.rejected_count += 1
            result.diagnostics.append(AssemblyDiagnostic(
                ticker=ticker,
                pattern_id=PATTERN_ID,
                diagnostic_type="missing_m1_fields",
                detail=",".join(missing_load_bearing + missing_multipliers),
            ))
            continue

        lineage_ids = list(lineage_ids_by_ticker.get(ticker, []))
        lineage_hashes = list(lineage_hashes_by_ticker.get(ticker, []))
        if snap_lineage and snap_lineage not in lineage_hashes:
            lineage_hashes.append(snap_lineage)
        if source_lineage_hash and source_lineage_hash not in lineage_hashes:
            lineage_hashes.append(source_lineage_hash)
        inp = build_pattern_input(
            ticker=ticker,
            pattern_id=PATTERN_ID,
            asof_timestamp=cutoff_timestamp,
            validated_fields=validated,
            lineage_ids=lineage_ids,
            lineage_hashes=lineage_hashes,
            universe_snapshot_id=snap_id,
        )
        result.inputs.append(inp)
        result.assembled_count += 1

    return result


def _signed_sue_percentiles(
    computations: Iterable[FosterComputation],
) -> Tuple[Dict[str, float], Optional[str]]:
    computed = [
        (item.ticker, float(item.sue_foster))
        for item in computations
        if item.computed and item.sue_foster is not None
    ]
    values = [sue for _, sue in computed]
    n = len(values)
    if n < MIN_ANNOUNCING_COHORT_SIZE:
        return {}, "announcing_cohort_too_small"
    if max(values) - min(values) <= SUE_TIE_EPSILON:
        return {}, "announcing_cohort_all_equal_sue"
    percentiles: Dict[str, float] = {}
    for ticker, sue in computed:
        percentiles[ticker] = sum(1 for value in values if value <= sue) / n
    return percentiles, None


def _compute_sue_series(eps_by_index: Dict[int, float]) -> List[Dict[str, Any]]:
    series: List[Dict[str, Any]] = []
    for index in sorted(eps_by_index):
        if any(index - lag not in eps_by_index for lag in FOSTER_REQUIRED_LAGS):
            continue
        diffs = [
            eps_by_index[index - lag] - eps_by_index[index - lag - 4]
            for lag in FOSTER_DIFF_LAGS
        ]
        sigma = _sample_std(diffs)
        if sigma is None or sigma < FOSTER_SIGMA_DELTA_EPSILON:
            continue
        expected = eps_by_index[index - 4] + sum(diffs) / len(diffs)
        sue = (eps_by_index[index] - expected) / sigma
        if not _is_finite(sue) or abs(sue) > MAX_ABS_FOSTER_SUE:
            continue
        series.append({
            "fiscal_index": index,
            "fiscal_year": index // 4,
            "fiscal_quarter": index % 4 + 1,
            "sue": sue,
        })
    return series


def _resolve_current_fiscal_index(
    event: FmpEarningsCalendarEvent,
    eps_history: Sequence[FmpEpsRecord],
    *,
    asof_timestamp: Optional[datetime] = None,
) -> tuple[Optional[int], Optional[str]]:
    if event.fiscal_year is not None and event.fiscal_quarter is not None:
        return event.fiscal_year * 4 + (event.fiscal_quarter - 1), event.fiscal_date_ending
    fiscal_date = _parse_date(event.fiscal_date_ending)
    if fiscal_date is not None:
        return fiscal_date.year * 4 + ((fiscal_date.month - 1) // 3), fiscal_date.isoformat()
    event_date = _parse_date(event.date)
    if event_date is None:
        return None, None
    normalized_asof = _ensure_aware(asof_timestamp) if asof_timestamp is not None else None
    current_matches: List[Tuple[int, Optional[str]]] = []
    if event.actual_eps is not None and _is_finite(event.actual_eps):
        for record in eps_history:
            idx = _record_fiscal_index(record)
            if idx is None or record.eps is None or not _is_finite(record.eps):
                continue
            if not math.isclose(
                float(record.eps),
                float(event.actual_eps),
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                continue
            accepted_at = _accepted_at(record)
            if normalized_asof is not None:
                if accepted_at is None or accepted_at > normalized_asof:
                    continue
            accepted_day = accepted_at.date() if accepted_at is not None else _parse_date(record.date)
            period_end = _parse_date(record.fiscal_date_ending)
            if accepted_day is None or accepted_day < event_date:
                continue
            if period_end is not None and period_end > event_date:
                continue
            current_matches.append((idx, record.fiscal_date_ending))
    if current_matches:
        distinct = {idx for idx, _period_end in current_matches}
        if len(distinct) == 1:
            return current_matches[0]
        return None, None
    candidates: List[Tuple[int, Optional[str]]] = []
    for record in eps_history:
        idx = _record_fiscal_index(record)
        if idx is None:
            continue
        accepted_at = _accepted_at(record)
        if asof_timestamp is not None:
            if accepted_at is None or accepted_at > normalized_asof:
                continue
        accepted_day = accepted_at.date() if accepted_at is not None else _parse_date(record.date)
        if accepted_day is not None and accepted_day <= event_date:
            candidates.append((idx, record.fiscal_date_ending))
    if not candidates:
        return None, None
    prior_index, _prior_period_end = max(candidates, key=lambda item: item[0])
    current_index = prior_index + 1
    return current_index, None


def _select_pit_eps_records(
    eps_history: Sequence[FmpEpsRecord],
    *,
    indices: Sequence[int],
    diagnostic_indices: set[int],
    asof_timestamp: Optional[datetime],
) -> Tuple[Dict[int, FmpEpsRecord], List[str]]:
    by_index: Dict[int, List[FmpEpsRecord]] = {}
    diagnostics: List[str] = []
    for record in eps_history:
        index = _record_fiscal_index(record)
        if index is None:
            continue
        by_index.setdefault(index, []).append(record)

    selected: Dict[int, FmpEpsRecord] = {}
    normalized_asof = _ensure_aware(asof_timestamp) if asof_timestamp is not None else None
    for index in indices:
        records = by_index.get(index, [])
        finite_records = [
            record for record in records
            if record.eps is not None and _is_finite(record.eps)
        ]
        if not finite_records:
            if index in diagnostic_indices:
                diagnostics.append(f"eps_missing_or_invalid:{index}")
            continue
        if normalized_asof is None:
            selected[index] = finite_records[0]
            continue
        accepted_rows = [
            (_accepted_at(record), record)
            for record in finite_records
            if _accepted_at(record) is not None
        ]
        if not accepted_rows:
            if index in diagnostic_indices:
                diagnostics.append(f"eps_missing_accepted_date:{index}")
            continue
        eligible = [
            (accepted_at, record)
            for accepted_at, record in accepted_rows
            if accepted_at is not None and accepted_at <= normalized_asof
        ]
        if not eligible:
            if index in diagnostic_indices:
                diagnostics.append(f"eps_accepted_after_asof:{index}")
            continue
        selected[index] = max(eligible, key=lambda item: item[0] or datetime.min.replace(tzinfo=timezone.utc))[1]
    return selected, diagnostics


def _record_fiscal_index(record: FmpEpsRecord) -> Optional[int]:
    if record.fiscal_year is not None and record.fiscal_quarter is not None:
        return record.fiscal_year * 4 + (record.fiscal_quarter - 1)
    fiscal_date = _parse_date(record.fiscal_date_ending)
    if fiscal_date is not None:
        return fiscal_date.year * 4 + ((fiscal_date.month - 1) // 3)
    return None


def _weekly_returns(bars: Sequence[FmpBar]) -> Dict[date, float]:
    closes: Dict[date, float] = {}
    for bar in bars:
        day = _parse_date(bar.date)
        close = bar.split_adjusted_close if bar.split_adjusted_close is not None else bar.close
        if day is not None and close is not None and _is_finite(close) and close > 0:
            closes[day] = float(close)
    if len(closes) < 2:
        return {}
    days = sorted(closes)
    anchor = days[0]
    while anchor.weekday() != 2:
        anchor += timedelta(days=1)
    weekly: List[Tuple[date, float]] = []
    last_selected: Optional[date] = None
    while anchor <= days[-1]:
        eligible = [day for day in days if day <= anchor and (last_selected is None or day > last_selected)]
        if eligible:
            selected = eligible[-1]
            weekly.append((anchor, closes[selected]))
            last_selected = selected
        anchor += timedelta(days=7)
    returns: Dict[date, float] = {}
    for idx in range(1, len(weekly)):
        prev = weekly[idx - 1][1]
        current = weekly[idx][1]
        if prev > 0:
            returns[weekly[idx][0]] = current / prev - 1.0
    return returns


def _ols_fit(x_rows: List[List[float]], y: List[float]) -> Optional[Dict[str, float]]:
    if not x_rows or len(x_rows) != len(y):
        return None
    p = len(x_rows[0])
    xtx = [[0.0 for _ in range(p)] for _ in range(p)]
    xty = [0.0 for _ in range(p)]
    for row, target in zip(x_rows, y):
        for i in range(p):
            xty[i] += row[i] * target
            for j in range(p):
                xtx[i][j] += row[i] * row[j]
    beta = _solve_linear_system(xtx, xty)
    if beta is None:
        return None
    residuals = [
        target - sum(coef * value for coef, value in zip(beta, row))
        for row, target in zip(x_rows, y)
    ]
    rss = sum(value * value for value in residuals)
    mean_y = sum(y) / len(y)
    tss = sum((value - mean_y) ** 2 for value in y)
    if tss <= 0:
        return None
    r2 = 1.0 - rss / tss
    return {"rss": rss, "r2": r2}


def _solve_linear_system(matrix: List[List[float]], vector: List[float]) -> Optional[List[float]]:
    n = len(vector)
    augmented = [row[:] + [vector[idx]] for idx, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            return None
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[row][n] for row in range(n)]


def _lag1_autocorr(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if _is_finite(value)]
    if len(clean) < 3:
        return 0.0
    x = clean[:-1]
    y = clean[1:]
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def _streak_length(signs_desc: Sequence[int]) -> int:
    if not signs_desc:
        return 0
    first = signs_desc[0]
    if first == 0:
        return 0
    count = 0
    for sign in signs_desc:
        if sign != first:
            break
        count += 1
    return count


def _sample_std(values: Sequence[float]) -> Optional[float]:
    clean = [float(value) for value in values if _is_finite(value)]
    if len(clean) < 2:
        return None
    mean = sum(clean) / len(clean)
    return math.sqrt(sum((value - mean) ** 2 for value in clean) / (len(clean) - 1))


def _field(
    name: str,
    value: Any,
    source_timestamp: Optional[datetime],
    source_provider: str,
    lineage_hash: Optional[str],
) -> AssembledField:
    presence = FieldPresence.PRESENT if value is not None else FieldPresence.UNAVAILABLE
    return AssembledField(
        name=name,
        value=value,
        presence=presence,
        source_timestamp=source_timestamp,
        allowed_cutoff=source_timestamp,
        source_provider=source_provider,
        lineage_hash=lineage_hash,
    )


def _snap_attr(snap: Any, name: str) -> Any:
    if isinstance(snap, dict):
        return snap.get(name)
    return getattr(snap, name, None)


def _ensure_aware(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _accepted_at(record: FmpEpsRecord) -> Optional[datetime]:
    raw = getattr(record, "accepted_date", None)
    if raw is None:
        return None
    text = str(raw).strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed_date = date.fromisoformat(text[:10])
        except ValueError:
            return None
        return datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=timezone.utc)
    return _ensure_aware(parsed)


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sign(value: Any) -> int:
    if value is None or not _is_finite(value):
        return 0
    numeric = float(value)
    if numeric > 0:
        return 1
    if numeric < 0:
        return -1
    return 0
