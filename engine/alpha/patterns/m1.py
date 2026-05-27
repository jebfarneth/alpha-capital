"""
M1 — Post-Earnings Announcement Drift (PEAD) Detector.

Vault source: Engineering/Patterns/M1-PEAD/

Thesis: event_drift (time-driven). Stocks with top-quintile positive
earnings surprises (Foster SUE) exhibit drift over the following
15 trading days as information incorporation completes.

Exposure formula (EXPOSURE.md):
  X_M1 = SUE * exp(-delta_t/5) * W_rho1 * Q_sign * W_hm * W_sigma * W_streak
  decay_integrated_avg = X_t0 * 0.35
  raw_expected_edge = remaining_decay_integrated_avg * lambda_M1_15td
      (capped at configurable raw_edge_cap, default 200 bps)

Signal admission:
  1. Operating-universe membership (fail-closed)
  2. Market data quality (shared guard, not require_fields for EOD M-track)
  3. sue_foster present, finite, positive
  4. sue_signed_percentile >= 0.80 (top positive quintile)
  5. delta_t_trading_days is an integer in [0, 15]
  6. remaining hold window is positive

Exit: Time-driven only. No T1/T2/T3/stop.
  max_hold = min(15, next_earnings_td - 1)
  remaining_horizon_days = max_hold - delta_t_trading_days
Routing: Class A (midpoint limit, day-valid).

Evidence: each fired signal persists validated_or_shadow_lambda_M1_15td,
lambda_M1_default_15td, lambda_M1_source, raw_edge_cap, and raw_edge_uncapped
so shadow validation can reconstruct the expected-edge assumption.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from alpha.data.contracts import stable_hash
from alpha.patterns.contracts import (
    BasePatternDetector,
    PatternDetectionResult,
    PatternFeatures,
    PatternInput,
    PatternSignal,
    PatternId,
    PatternTrack,
    RouteClass,
    SignalDirection,
    ThesisCategory,
)
from alpha.patterns.guards import (
    classify_fidelity,
    compute_data_confidence,
    finite_float,
    integral_int,
    market_data_quality_rejection,
    operating_universe_rejection,
    reject_future_timestamp,
    require_asof_timestamp,
    require_lineage_hash,
    set_signal_identity,
)

# Vault constants (SPEC.md / EXPOSURE.md)
LAMBDA_M1_MONTHLY = 0.0075  # 0.75%/month (amplified Martineau-Vamossy-Subrahmanyam midpoint)
MICROCAP_AMPLIFICATION = 1.75  # Vamossy 2025; Subrahmanyam 2025 microcap corroboration
LAMBDA_M1_15TD = LAMBDA_M1_MONTHLY * 15.0 / 21.0  # ~0.005357
MAX_SIGNAL_HORIZON = "15d"
MAX_DELTA_T = 15
DECAY_TAU = 5.0  # exp(-delta_t / 5); half-life ~3.5 trading days
DECAY_INTEGRATED_AVG = 0.35  # EXPOSURE.md: <exp(-dt/5)>_0..14 ≈ 0.35
RAW_EDGE_CAP = 0.02  # 200 bps optimizer-input prior cap
MIN_SUE_PERCENTILE = 0.80  # top positive quintile gate
RHO1_LOWER = -0.5  # winsorization bounds for lag-1 SUE autocorrelation
RHO1_UPPER = 0.95
STREAK_THRESHOLD = 4  # BSV regime-transition penalty at >= 4 consecutive same-sign


# ---------------------------------------------------------------------------
# Pure computation helpers
# ---------------------------------------------------------------------------

def compute_decay_factor(delta_t: int) -> float:
    """Per EXPOSURE.md: exp(-delta_t / 5), hard-zeroed for delta_t > 15."""
    if delta_t < 0 or delta_t > MAX_DELTA_T:
        return 0.0
    return math.exp(-delta_t / DECAY_TAU)


def compute_remaining_decay_integrated_avg(delta_t: int, max_hold_days: int = MAX_DELTA_T) -> float:
    """
    Decay-integrated exposure remaining from *delta_t* through *max_hold_days*.

    With default arguments (delta_t=0, max_hold_days=15) the result equals
    the vault constant ``DECAY_INTEGRATED_AVG`` exactly.  When the hold is
    shortened by next-earnings constraints, only the actually-executable
    decay mass is included.
    """
    if delta_t < 0 or max_hold_days <= 0 or delta_t >= max_hold_days:
        return 0.0
    capped = min(max_hold_days, MAX_DELTA_T)
    full_sum = sum(math.exp(-d / DECAY_TAU) for d in range(MAX_DELTA_T))
    remaining_sum = sum(math.exp(-d / DECAY_TAU) for d in range(delta_t, capped))
    return DECAY_INTEGRATED_AVG * remaining_sum / full_sum


def compute_w_rho1(rho1: float) -> float:
    """Per EXPOSURE.md: W_rho1 = 1 + rho1 * 0.4, rho1 winsorized to [-0.5, 0.95]."""
    rho1_w = max(RHO1_LOWER, min(rho1, RHO1_UPPER))
    return 1.0 + rho1_w * 0.4


def compute_q_sign(sign_current: int, sign_prior: Optional[int]) -> float:
    """Per EXPOSURE.md: sign-consistency quality flag {0.8, 1.0, 1.2}."""
    if sign_prior is None or sign_prior == 0 or sign_current == 0:
        return 1.0
    if sign_current == sign_prior:
        return 1.2
    return 0.8


def compute_w_hm(d1_decile: int) -> float:
    """Per EXPOSURE.md: W_hm = 1 + 0.2 * (2 * D1_decile/10 - 1). Range [0.80, 1.20]."""
    return 1.0 + 0.2 * (2.0 * d1_decile / 10.0 - 1.0)


def compute_w_sigma(d1_decile: int, sigma_epsilon_percentile: float) -> float:
    """Per EXPOSURE.md: conditional on D1 decile in {8,9,10}; else 1.0."""
    if d1_decile >= 8:
        return 1.0 + 0.2 * sigma_epsilon_percentile
    return 1.0


def compute_w_streak(streak_length: int) -> float:
    """Per EXPOSURE.md: BSV penalty 0.5 when streak >= 4; else 1.0."""
    if streak_length >= STREAK_THRESHOLD:
        return 0.5
    return 1.0


def compute_signal_strength(sue_signed_percentile: float) -> float:
    """Per DATA.md: map top-quintile [0.80, 1.00] to [0.0, 1.0]."""
    return round(min(max((sue_signed_percentile - MIN_SUE_PERCENTILE) / (1.0 - MIN_SUE_PERCENTILE), 0.0), 1.0), 6)


def _data_confidence(inp: PatternInput, quality_flags: Dict[str, Any]) -> float:
    return compute_data_confidence(
        quality_flags,
        field_confidence_sources=(inp.market_data, inp.fundamental_data, inp.event_data),
    )


# ---------------------------------------------------------------------------
# Diagnostic field helpers
# ---------------------------------------------------------------------------

def _copy_diagnostic_fields(feat_dict: Dict[str, Any], *sources: Dict[str, Any]) -> None:
    for source in sources:
        for key in (
            "earnings_event_id", "earnings_calendar_event_id", "m1_event_id",
            "announcement_date", "announcement_time_utc", "fiscal_period_end",
            "fiscal_year", "fiscal_quarter", "next_earnings_date_estimate",
            "next_earnings_trading_days_from_signal", "foster_history_quarters_used",
            "split_adjustment_continuity_check", "pre_announcement_car",
            "hazard_score_at_signal", "filing_veto_status",
            "market_data_status", "halt_status", "corporate_action_filter_passed",
            "market_cap_usd", "sector", "d1_decile", "sigma_epsilon_percentile",
            "cen_hq_county", "cen_sci_score", "i1_also_firing", "i5_also_firing",
            "overlapping_pattern_ids",
        ):
            val = source.get(key)
            if val is not None:
                feat_dict[key] = val
    feat_dict.setdefault("filing_veto_status", "not_computed")


def _set_m1_signal_identity(feat_dict: Dict[str, Any], inp: PatternInput) -> None:
    event_id = (
        feat_dict.get("earnings_event_id")
        or feat_dict.get("earnings_calendar_event_id")
        or feat_dict.get("m1_event_id")
    )
    if event_id is not None:
        components = {"earnings_event_id": event_id}
    else:
        components = {
            "announcement_date": feat_dict.get("announcement_date"),
            "announcement_time_utc": feat_dict.get("announcement_time_utc"),
            "fiscal_period_end": feat_dict.get("fiscal_period_end"),
            "fiscal_year": feat_dict.get("fiscal_year"),
            "fiscal_quarter": feat_dict.get("fiscal_quarter"),
        }
    set_signal_identity(
        feat_dict,
        pattern_id=PatternId.M1,
        ticker=inp.ticker,
        components=components,
        source="earnings_event_content",
    )


def _reject_signal(feat_dict: Dict[str, Any], reason: str) -> None:
    feat_dict["rejection_reason"] = reason
    feat_dict["signal_generated"] = False
    feat_dict["x_m1"] = 0.0


# ---------------------------------------------------------------------------
# Multiplier enrichment
# ---------------------------------------------------------------------------

def _compute_multipliers(
    feat_dict: Dict[str, Any],
    market_data: Dict[str, Any],
    warnings: List[str],
    quality_flags: Dict[str, Any],
) -> Dict[str, float]:
    rho1_raw = market_data.get("rho1")
    if rho1_raw is None:
        rho1 = 0.0
        quality_flags["missing_rho1"] = True
        warnings.append("rho1 unavailable — defaulting to 0.0")
    else:
        rho1_parsed = finite_float(rho1_raw)
        if rho1_parsed is None:
            rho1 = 0.0
            quality_flags["invalid_rho1"] = True
            warnings.append("rho1 invalid — defaulting to 0.0")
        else:
            rho1 = rho1_parsed
    w_rho1 = compute_w_rho1(rho1)

    sign_current_raw = market_data.get("sue_sign_current")
    sign_prior_raw = market_data.get("sue_sign_prior")
    sign_current = integral_int(sign_current_raw) if sign_current_raw is not None else 0
    sign_prior = integral_int(sign_prior_raw) if sign_prior_raw is not None else None
    if sign_current not in (-1, 0, 1):
        quality_flags["invalid_sue_sign"] = True
        warnings.append("sue_sign_current invalid — defaulting to 0")
        sign_current = 0
    if sign_prior is not None and sign_prior not in (-1, 0, 1):
        quality_flags["invalid_sue_sign"] = True
        warnings.append("sue_sign_prior invalid — defaulting to None")
        sign_prior = None
    q_sign = compute_q_sign(sign_current, sign_prior)

    d1_decile_raw = market_data.get("d1_decile")
    if d1_decile_raw is None:
        d1_decile = 5
        quality_flags["missing_d1_decile"] = True
        warnings.append("d1_decile unavailable — defaulting to 5 (neutral)")
    else:
        d1_decile_parsed = integral_int(d1_decile_raw)
        if d1_decile_parsed is None or not 1 <= d1_decile_parsed <= 10:
            d1_decile = 5
            quality_flags["invalid_d1_decile"] = True
            warnings.append("d1_decile invalid — defaulting to 5 (neutral)")
        else:
            d1_decile = d1_decile_parsed
    w_hm = compute_w_hm(d1_decile)

    sigma_pct_raw = market_data.get("sigma_epsilon_percentile")
    if sigma_pct_raw is None:
        sigma_pct = 0.0
    else:
        sigma_pct_parsed = finite_float(sigma_pct_raw)
        if sigma_pct_parsed is None or not 0.0 <= sigma_pct_parsed <= 1.0:
            sigma_pct = 0.0
            quality_flags["invalid_sigma_epsilon_percentile"] = True
            warnings.append("sigma_epsilon_percentile invalid — defaulting to 0.0")
        else:
            sigma_pct = sigma_pct_parsed
    w_sigma = compute_w_sigma(d1_decile, sigma_pct)

    streak_raw = market_data.get("sue_streak_length")
    if streak_raw is None:
        streak = 0
    else:
        streak_parsed = integral_int(streak_raw)
        if streak_parsed is None or streak_parsed < 0:
            streak = 0
            quality_flags["invalid_sue_streak_length"] = True
            warnings.append("sue_streak_length invalid — defaulting to 0")
        else:
            streak = streak_parsed
    w_streak = compute_w_streak(streak)

    multipliers = {
        "w_rho1": round(w_rho1, 4),
        "q_sign": q_sign,
        "w_hm": round(w_hm, 4),
        "w_sigma": round(w_sigma, 4),
        "w_streak": w_streak,
    }
    compound = w_rho1 * q_sign * w_hm * w_sigma * w_streak
    feat_dict["multipliers"] = multipliers
    feat_dict["compound_multiplier"] = round(compound, 6)
    feat_dict["rho1"] = rho1
    feat_dict["sue_sign_current"] = sign_current
    feat_dict["sue_sign_prior"] = sign_prior
    feat_dict["d1_decile"] = d1_decile
    feat_dict["sigma_epsilon_percentile"] = sigma_pct
    feat_dict["sue_streak_length"] = streak if streak_raw is not None else None
    return multipliers


# ---------------------------------------------------------------------------
# Signal enrichment
# ---------------------------------------------------------------------------

def _enrich_m1_signal(
    feat_dict: Dict[str, Any],
    inp: PatternInput,
    warnings: List[str],
    quality_flags: Dict[str, Any],
    lambda_m1_15td: float,
    raw_edge_cap: float,
) -> Optional[PatternSignal]:
    md = inp.market_data

    sue_foster = finite_float(md["sue_foster"])
    delta_t = integral_int(md["delta_t_trading_days"])

    if sue_foster is None:
        warnings.append("invalid sue_foster")
        return None

    if delta_t is None:
        feat_dict["sue_foster"] = round(sue_foster, 6)
        feat_dict["delta_t_trading_days"] = None
        _copy_diagnostic_fields(feat_dict, inp.market_data, inp.fundamental_data, inp.event_data)
        _set_m1_signal_identity(feat_dict, inp)
        _reject_signal(feat_dict, "invalid_delta_t")
        return None

    feat_dict["sue_foster"] = round(sue_foster, 6)
    feat_dict["delta_t_trading_days"] = delta_t
    _copy_diagnostic_fields(feat_dict, inp.market_data, inp.fundamental_data, inp.event_data)
    _set_m1_signal_identity(feat_dict, inp)

    # Market data quality (EOD M-track; not require_fields)
    pre_signal_rejection = market_data_quality_rejection(feat_dict, md)
    if pre_signal_rejection is not None:
        quality_flags["market_data_quality_rejected"] = True
        _reject_signal(feat_dict, pre_signal_rejection)
        return None

    # Announcement recency
    if delta_t < 0:
        _reject_signal(feat_dict, "invalid_delta_t")
        return None
    if delta_t > MAX_DELTA_T:
        _reject_signal(feat_dict, "announcement_too_old")
        return None

    # SUE positivity (V1 long-only)
    if sue_foster <= 0:
        _reject_signal(feat_dict, "sue_not_positive")
        return None

    # Top-quintile gate
    sue_pct_raw = md.get("sue_signed_percentile")
    if sue_pct_raw is None:
        _reject_signal(feat_dict, "missing_sue_percentile")
        return None
    sue_pct = finite_float(sue_pct_raw)
    if sue_pct is None or not 0.0 <= sue_pct <= 1.0:
        _reject_signal(feat_dict, "invalid_sue_percentile")
        return None
    feat_dict["sue_signed_percentile"] = round(sue_pct, 6)
    if sue_pct < MIN_SUE_PERCENTILE:
        _reject_signal(feat_dict, "sue_below_threshold")
        return None

    # Compute multipliers
    _compute_multipliers(feat_dict, md, warnings, quality_flags)

    # Compute exposure
    decay = compute_decay_factor(delta_t)
    feat_dict["decay_factor"] = round(decay, 6)
    compound = feat_dict["compound_multiplier"]
    x_m1_t0 = sue_foster * compound  # at delta_t=0, decay=1.0
    x_m1_t = sue_foster * decay * compound
    feat_dict["exposure_x_m1_t0"] = round(x_m1_t0, 6)
    feat_dict["x_m1"] = round(x_m1_t, 6)

    # Max hold (evidence only; enforcement is downstream)
    next_earnings_td_raw = md.get("next_earnings_trading_days_from_signal")
    if next_earnings_td_raw is not None:
        next_td = integral_int(next_earnings_td_raw)
        if next_td is None:
            _reject_signal(feat_dict, "invalid_next_earnings_window")
            return None
        max_hold = min(MAX_DELTA_T, next_td - 1)
    else:
        max_hold = MAX_DELTA_T
    feat_dict["max_hold_days"] = max_hold

    remaining_horizon_days = max_hold - delta_t
    feat_dict["remaining_horizon_days"] = remaining_horizon_days
    if remaining_horizon_days <= 0:
        _reject_signal(feat_dict, "no_remaining_hold_window")
        return None

    # Raw expected edge (remaining decay-integrated, capped by runtime config)
    remaining_decay_avg = compute_remaining_decay_integrated_avg(delta_t, max_hold)
    feat_dict["remaining_decay_integrated_avg"] = round(remaining_decay_avg, 6)
    x_bar = x_m1_t0 * remaining_decay_avg
    raw_edge = x_bar * lambda_m1_15td
    raw_edge_capped = min(raw_edge, raw_edge_cap)
    feat_dict["raw_edge_uncapped"] = round(raw_edge, 8)
    feat_dict["raw_edge_cap"] = raw_edge_cap
    feat_dict["raw_edge_cap_source"] = "shadow_prior" if raw_edge_cap == RAW_EDGE_CAP else "configured_or_injected"
    feat_dict["raw_edge_cap_applied"] = raw_edge > raw_edge_cap

    signal_strength = compute_signal_strength(sue_pct)

    feat_dict["signal_generated"] = True
    feat_dict["lambda_M1_monthly"] = LAMBDA_M1_MONTHLY
    feat_dict["microcap_amplification"] = MICROCAP_AMPLIFICATION
    feat_dict["validated_or_shadow_lambda_M1_15td"] = lambda_m1_15td
    feat_dict["lambda_M1_15td"] = lambda_m1_15td
    feat_dict["lambda_M1_default_15td"] = LAMBDA_M1_15TD
    feat_dict["amplified_lambda_M1_15td"] = round(lambda_m1_15td, 8)
    feat_dict["lambda_M1_source"] = (
        "shadow_prior" if lambda_m1_15td == LAMBDA_M1_15TD else "validated_or_injected"
    )
    feat_dict["expected_return_priors"] = {"gross_bps": round(raw_edge_capped * 10_000, 2)}

    return PatternSignal(
        direction=SignalDirection.LONG,
        raw_signal_strength=signal_strength,
        raw_expected_edge=round(raw_edge_capped, 6),
        signal_horizon=f"{remaining_horizon_days}d",
        route_class=RouteClass.A,
        data_confidence=_data_confidence(inp, quality_flags),
    )


# ---------------------------------------------------------------------------
# Hashes
# ---------------------------------------------------------------------------

def _compute_hashes(
    inp: PatternInput, asof: Any, feat_dict: Dict[str, Any],
    signals: List[PatternSignal], warnings: List[str], quality_flags: Dict[str, Any],
) -> tuple:
    input_hash = stable_hash({
        "ticker": inp.ticker, "asof_timestamp": asof,
        "market_data": inp.market_data, "fundamental_data": inp.fundamental_data,
        "event_data": inp.event_data,
        "lineage_hashes": inp.lineage_hashes, "universe_snapshot_id": inp.universe_snapshot_id,
    })
    output_hash = stable_hash({
        "features": feat_dict,
        "signals": [
            {"direction": s.direction, "raw_signal_strength": s.raw_signal_strength,
             "raw_expected_edge": s.raw_expected_edge, "signal_horizon": s.signal_horizon,
             "signal_status": s.signal_status, "route_class": s.route_class,
             "data_confidence": s.data_confidence}
            for s in signals
        ],
        "warnings": warnings, "quality_flags": quality_flags,
    })
    return input_hash, output_hash


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class M1Detector(BasePatternDetector):
    """M1 Post-Earnings Announcement Drift detector."""

    pattern_id = PatternId.M1
    version = "1.0"
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.EVENT_DRIFT
    route_class = RouteClass.A

    def __init__(
        self,
        lambda_m1_15td: float = LAMBDA_M1_15TD,
        raw_edge_cap: float = RAW_EDGE_CAP,
    ):
        parsed_lambda = finite_float(lambda_m1_15td)
        parsed_cap = finite_float(raw_edge_cap)
        if parsed_lambda is None or parsed_lambda <= 0:
            raise ValueError("lambda_m1_15td must be finite and positive")
        if parsed_cap is None or parsed_cap <= 0:
            raise ValueError("raw_edge_cap must be finite and positive")
        self._lambda_m1_15td = parsed_lambda
        self._raw_edge_cap = parsed_cap

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        """Evaluate an M1 post-earnings-announcement drift setup."""

        asof = require_asof_timestamp(inp.asof_timestamp)
        warnings: List[str] = []
        quality_flags: Dict[str, Any] = {}

        reject_future_timestamp(asof, warnings, quality_flags)
        require_lineage_hash(inp.lineage_hashes, warnings, quality_flags)

        md = inp.market_data
        sue_foster_raw = md.get("sue_foster")
        delta_t_raw = md.get("delta_t_trading_days")

        if sue_foster_raw is None or delta_t_raw is None:
            warnings.append("missing required fields (sue_foster or delta_t_trading_days)")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        sue_f = finite_float(sue_foster_raw)
        delta_t_f = finite_float(delta_t_raw)

        if sue_f is None or delta_t_f is None:
            warnings.append("non-finite sue_foster or delta_t_trading_days")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        feat_dict: Dict[str, Any] = {}

        universe_rejection = operating_universe_rejection(
            md, warnings, quality_flags, pattern_id=self.pattern_id,
        )

        pit_passed = quality_flags.get("point_in_time_passed") is not False
        fidelity = classify_fidelity(
            has_primary_data=True, has_secondary_data=True,
            point_in_time_passed=pit_passed, lookahead_guard_passed=True,
        )

        signals: List[PatternSignal] = []

        if universe_rejection is not None:
            feat_dict["sue_foster"] = round(sue_f, 6)
            feat_dict["delta_t_trading_days"] = int(delta_t_f) if delta_t_f.is_integer() else delta_t_f
            _copy_diagnostic_fields(feat_dict, inp.market_data, inp.fundamental_data, inp.event_data)
            _reject_signal(feat_dict, universe_rejection)
        else:
            sig = _enrich_m1_signal(
                feat_dict,
                inp,
                warnings,
                quality_flags,
                self._lambda_m1_15td,
                self._raw_edge_cap,
            )
            if sig is not None:
                signals.append(sig)

        features = PatternFeatures(
            features=feat_dict, feature_manifest_version="m1-v1",
            fidelity_tier=fidelity, point_in_time_passed=pit_passed, lookahead_guard_passed=True,
        )

        input_hash, output_hash = _compute_hashes(inp, asof, feat_dict, signals, warnings, quality_flags)

        return PatternDetectionResult(
            pattern_id=self.pattern_id, ticker=inp.ticker, asof_timestamp=asof,
            features=features, signals=signals, warnings=warnings, quality_flags=quality_flags,
            input_hashes={"market_data": input_hash}, output_hashes={"features": output_hash},
        )

    def _no_features_result(self, ticker, asof, warnings, quality_flags):
        return PatternDetectionResult(
            pattern_id=self.pattern_id, ticker=ticker, asof_timestamp=asof,
            features=None, warnings=warnings, quality_flags=quality_flags,
        )
