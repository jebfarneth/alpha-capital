"""
M5 — Failed Breakdown Reversal Detector.

Vault source: Engineering/Patterns/M5-FailedBreakdown/

Thesis: mean_reversion. Stocks that dropped sharply over the trailing
week exhibit positive expected excess returns over the following 7 trading
days when the selloff stabilizes and reclaims a valid reversal anchor.
Structural support breaks receive a setup bonus, but support breaks are
not a hard blocker because the Lehmann/Jegadeesh reversal effect is
return-dislocation based.

Exposure formula (EXPOSURE.md):
  decline_magnitude = clip(-R_5d / sigma_20d, 0, 4)
  support_break_attempt_weight =
      1.25 if P_low_5d is available and P_low_5d < support_level
      1.00 for decline_only and decline_only_missing_support paths
  X_M5_setup = min(decline_magnitude * support_break_attempt_weight, 3.0)
  X_M5_activation = min(decline_magnitude * reclaim_strength * stabilization
                        * vol_conf * watchlist_decay * spread_quality, 3.0)

Expected-return bridge (SPEC.md / EXPOSURE.md):
  lambda_M5_weekly = 1.05% (Lehmann midpoint)
  amplification = 1.45 (Jegadeesh small-cap Q1)
  amplified_lambda_M5_7td = 1.05% * 1.45 = ~1.52%
  raw_expected_edge = X_M5_activation * amplified_lambda_M5_7td

Signal admission:
  1. Operating-universe membership (fail-closed)
  2. decline_magnitude >= 1.5 (minimum 1.5-sigma 5-day decline)
  3. support_break_attempt_weight > 0 (support-break bonus or decline-only path)
  4. X_M5_setup > 0

Signal paths:
  1. Watchlist: nightly setup qualified, no activation data. Setup path is
     support_break, decline_only, or decline_only_missing_support.
  2. Activation: support/reversal-anchor reclaim + stabilization >= 1.0 + volume >= 1.0
     + activation identity + quote capture + spread discipline + watchlist
     freshness proof. If support_level is unavailable, activation requires a
     positive caller-provided reversal_anchor_price; otherwise it fails closed
     with support_anchor_unavailable.

Routing: Class B after activation (marketable limit at ask + 0.5%, 300s cancel).
Mutex (M5 ⊥ M1/M4/M6) enforced at TCB, NOT at detector layer.

Evidence: each fired executable signal persists lambda_M5_weekly,
microcap_amplification, and amplified_lambda_M5_7td so shadow validation
can audit the thesis assumption.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from alpha.data.contracts import stable_hash
from alpha.patterns.activation import expiring_watchlist_freshness, required_fields_present
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
    DEFAULT_QUOTE_FIELDS,
    copy_fields,
    classify_fidelity,
    compute_data_confidence,
    finite_float,
    market_data_quality_rejection,
    operating_universe_rejection,
    quote_rejection,
    reject_future_timestamp,
    require_asof_timestamp,
    require_lineage_hash,
    set_signal_identity,
)

# Vault constants (EXPOSURE.md / SPEC.md)
LAMBDA_M5_WEEKLY = 0.0105  # 1.05% per week (Lehmann midpoint of 0.86-1.24%)
AMPLIFICATION = 1.45  # Jegadeesh small-cap Q1: a1_Q1/a1_full = 0.1342/0.0923
LAMBDA_M5_7TD = LAMBDA_M5_WEEKLY * AMPLIFICATION  # ~0.015225
X_M5_CAP = 3.0
SIGNAL_HORIZON = "7d"
MIN_DECLINE_MAGNITUDE = 1.5
DECLINE_MAGNITUDE_CAP = 4.0
DECLINE_ONLY_SETUP_WEIGHT = 1.0
SUPPORT_BREAK_SETUP_WEIGHT = 1.25
SPREAD_CAP = 0.005  # Normal Class B-style spread cap.
WIDE_SPREAD_CAP = 0.010  # Strong dislocations can pass up to 1.0% with haircut.
WIDE_SPREAD_MIN_PRE_SPREAD_X = 2.0
TIGHT_SPREAD_QUALITY = 1.0
WIDE_SPREAD_QUALITY = 0.75
WATCHLIST_MAX_AGE_SESSIONS = 3
WATCHLIST_DECAY_BY_AGE = {
    1: 1.0,
    2: 0.85,
    3: 0.70,
}
QUOTE_FIELDS = DEFAULT_QUOTE_FIELDS
QUOTE_DIAGNOSTIC_FIELDS = (*QUOTE_FIELDS, "quote_freshness_max_ms")
ACTIVATION_IDENTITY_FIELDS = ("activation_id", "activation_timestamp")
WATCHLIST_FRESHNESS_FIELDS = (
    "watchlist_signal_id",
    "watchlist_scan_date",
    "watchlist_expiration_session",
    "activation_session",
    "watchlist_age_sessions",
)

MISSING_SUPPORT_SETUP_PATH = "decline_only_missing_support"
ACTIVATION_DIAGNOSTIC_FIELDS = (
    *ACTIVATION_IDENTITY_FIELDS,
    *WATCHLIST_FRESHNESS_FIELDS,
    *QUOTE_DIAGNOSTIC_FIELDS,
)

ACTIVATION_STATE_WATCHLIST = "watchlist"
ACTIVATION_STATE_ACTIVATED = "activated"
ACTIVATION_STATE_NO_SETUP = "no_setup"
ACTIVATION_STATE_FAILED = "activation_failed"


# ---------------------------------------------------------------------------
# Pure computation helpers
# ---------------------------------------------------------------------------

def compute_decline_magnitude(return_5d: float, sigma_20d: float) -> float:
    """Per EXPOSURE.md: clip(-R_5d / sigma_20d, 0.0, 4.0)."""
    if sigma_20d <= 0 or return_5d >= 0:
        return 0.0
    return max(0.0, min(-return_5d / sigma_20d, DECLINE_MAGNITUDE_CAP))


def compute_support_break_attempt_weight(
    low_5d: Optional[float], support_level: Optional[float],
) -> float:
    """Per EXPOSURE.md: 1.25 support-break bonus; 1.0 decline-only path."""
    if support_level is None:
        return DECLINE_ONLY_SETUP_WEIGHT
    if support_level <= 0:
        return 0.0
    if low_5d is None:
        return DECLINE_ONLY_SETUP_WEIGHT
    if low_5d < support_level:
        return SUPPORT_BREAK_SETUP_WEIGHT
    return DECLINE_ONLY_SETUP_WEIGHT


def compute_support_reclaim_extension(
    last_price: float, support_level: float, sigma_20d: float,
) -> float:
    """Per EXPOSURE.md: max((last_price / support_level) - 1.0, 0.0) / max(sigma_20d, 0.01)."""
    if support_level <= 0 or last_price <= support_level:
        return 0.0
    return max((last_price / support_level) - 1.0, 0.0) / max(sigma_20d, 0.01)


def compute_support_reclaim_strength(
    reclaim_extension: float, last_price: float, support_level: float,
    intraday_vwap: Optional[float],
) -> float:
    """Per EXPOSURE.md tiered strength."""
    if last_price <= support_level:
        return 0.0
    above_vwap = intraday_vwap is not None and last_price > intraday_vwap
    if reclaim_extension >= 0.75 and above_vwap:
        return 1.5
    if reclaim_extension >= 0.25 and above_vwap:
        return 1.25
    if last_price > support_level:
        return 1.0
    return 0.0


def compute_stabilization_confirmation(
    last_price: float, open_price: Optional[float],
    intraday_vwap: Optional[float], support_level: float,
) -> float:
    """Per EXPOSURE.md tiered stabilization."""
    above_open = open_price is not None and last_price > open_price
    above_vwap = intraday_vwap is not None and last_price > intraday_vwap
    if above_open and above_vwap:
        return 1.5
    if above_vwap:
        return 1.25
    if last_price > support_level:
        return 1.0
    return 0.5


def compute_volume_confirmation(volume_ratio: float) -> float:
    """Per EXPOSURE.md tiered volume confirmation."""
    if volume_ratio >= 2.0:
        return 1.5
    if volume_ratio >= 1.5:
        return 1.25
    if volume_ratio >= 0.75:
        return 1.0
    return 0.5


def compute_watchlist_decay(age_sessions: int) -> float:
    """Multi-session M5 watchlists decay but remain valid through session 3."""
    return WATCHLIST_DECAY_BY_AGE.get(age_sessions, 0.0)


def compute_spread_quality(spread_pct: Optional[float], pre_spread_x: float) -> float:
    """Cost-aware spread tier: normal pass, strong-candidate wide pass, or reject."""
    if spread_pct is None:
        return 0.0
    if spread_pct <= SPREAD_CAP:
        return TIGHT_SPREAD_QUALITY
    if spread_pct <= WIDE_SPREAD_CAP and pre_spread_x >= WIDE_SPREAD_MIN_PRE_SPREAD_X:
        return WIDE_SPREAD_QUALITY
    return 0.0


def _data_confidence(inp: PatternInput, quality_flags: Dict[str, Any]) -> float:
    return compute_data_confidence(
        quality_flags,
        field_confidence_sources=(inp.market_data, inp.fundamental_data, inp.event_data),
    )


# ---------------------------------------------------------------------------
# Activation enrichment helpers
# ---------------------------------------------------------------------------

def _activation_identity_passed(market_data: Dict[str, Any]) -> bool:
    return required_fields_present(market_data, ACTIVATION_IDENTITY_FIELDS)


def _watchlist_freshness(market_data: Dict[str, Any]) -> tuple[bool, bool, bool, bool, float, Optional[int]]:
    freshness = expiring_watchlist_freshness(
        market_data,
        identity_fields=WATCHLIST_FRESHNESS_FIELDS,
        expiration_session_field="watchlist_expiration_session",
        activation_session_field="activation_session",
        age_field="watchlist_age_sessions",
        decay_by_age=WATCHLIST_DECAY_BY_AGE,
    )
    return (
        freshness.source_freshness_passed,
        freshness.watchlist_identity_passed,
        freshness.watchlist_session_match,
        freshness.signal_freshness_passed,
        freshness.decay_weight,
        freshness.age_sessions,
    )


def _activation_failure_reason(feat: Dict[str, Any]) -> str:
    if feat.get("support_anchor_available") is not True:
        return "support_anchor_unavailable"
    if not feat.get("support_reclaim_passed", False):
        return "support_reclaim_failed"
    if not feat.get("stabilization_passed", False):
        return "stabilization_failed"
    if not feat.get("volume_confirmation_passed", False):
        return "volume_confirmation_failed"
    if feat.get("activation_identity_passed") is not True:
        return "activation_identity_missing"
    if feat.get("quote_capture_passed") is not True:
        return "quote_unavailable"
    if not feat.get("spread_discipline_passed", False):
        return "spread_unavailable" if feat.get("spread_pct_vs_eval_quote") is None else "spread_too_wide"
    if not feat.get("signal_freshness_passed", False):
        return "signal_expired"
    return "unknown"


def _set_m5_signal_identity(feat_dict: Dict[str, Any], inp: PatternInput) -> None:
    setup_id = inp.market_data.get("m5_setup_id") or inp.market_data.get("watchlist_signal_id")
    if setup_id is not None:
        components = {"m5_setup_id": setup_id}
        source = "upstream_m5_setup_id"
    else:
        components = {
            "setup_path": feat_dict.get("setup_path"),
            "support_level": feat_dict.get("support_level"),
            "low_5d": feat_dict.get("low_5d"),
        }
        source = "failed_breakdown_setup_content"
    set_signal_identity(
        feat_dict,
        pattern_id=PatternId.M5,
        ticker=inp.ticker,
        components=components,
        source=source,
    )


def _set_reclaim_features(
    feat_dict: Dict[str, Any], inp: PatternInput,
    support_level: Optional[float], sigma_20d: float,
) -> tuple[float, float]:
    last_price = finite_float(inp.market_data.get("price"))
    intraday_vwap_raw = inp.market_data.get("intraday_vwap")
    intraday_vwap = finite_float(intraday_vwap_raw) if intraday_vwap_raw is not None else None
    open_price_raw = inp.market_data.get("open_price")
    open_price = finite_float(open_price_raw) if open_price_raw is not None else None
    reversal_anchor_raw = inp.market_data.get("reversal_anchor_price")
    if support_level is not None:
        anchor_price = support_level
    elif reversal_anchor_raw is not None:
        anchor_price = finite_float(reversal_anchor_raw)
    else:
        anchor_price = None
    support_anchor_available = anchor_price is not None and anchor_price > 0
    price_available = last_price is not None and last_price > 0

    if support_anchor_available and price_available:
        reclaim_ext = compute_support_reclaim_extension(last_price, anchor_price, sigma_20d)
        reclaim_strength = compute_support_reclaim_strength(reclaim_ext, last_price, anchor_price, intraday_vwap)
        stabilization = compute_stabilization_confirmation(last_price, open_price, intraday_vwap, anchor_price)
    else:
        reclaim_ext = 0.0
        reclaim_strength = 0.0
        stabilization = 0.5

    feat_dict["last_price"] = last_price
    feat_dict["intraday_vwap"] = intraday_vwap
    feat_dict["open_price"] = open_price
    feat_dict["reversal_anchor_price"] = anchor_price
    feat_dict["support_anchor_available"] = support_anchor_available
    feat_dict["support_reclaim_extension"] = round(reclaim_ext, 6)
    feat_dict["support_reclaim_strength"] = reclaim_strength
    feat_dict["support_reclaim_passed"] = reclaim_strength > 0
    feat_dict["intraday_stabilization_confirmation"] = stabilization
    feat_dict["stabilization_passed"] = stabilization >= 1.0
    return reclaim_strength, stabilization


def _set_volume_features(
    feat_dict: Dict[str, Any], inp: PatternInput,
    quality_flags: Dict[str, Any], warnings: List[str],
) -> float:
    cumulative_volume = inp.market_data.get("cumulative_session_volume")
    expected_volume = inp.market_data.get("expected_same_clock_volume_20d")

    vol_ratio = None
    vol_conf = 0.0
    if cumulative_volume is not None and expected_volume is not None:
        cv, ev = finite_float(cumulative_volume), finite_float(expected_volume)
        if cv is not None and ev is not None and ev > 0:
            vol_ratio = cv / ev
            vol_conf = compute_volume_confirmation(vol_ratio)
        else:
            quality_flags["missing_volume_data"] = True
            warnings.append("invalid expected_same_clock_volume_20d for volume confirmation")
    else:
        quality_flags["missing_volume_data"] = True
        warnings.append("missing cumulative_session_volume/expected_same_clock_volume_20d")

    feat_dict["cumulative_session_volume"] = finite_float(cumulative_volume) if cumulative_volume is not None else None
    feat_dict["expected_same_clock_volume_20d"] = finite_float(expected_volume) if expected_volume is not None else None
    feat_dict["intraday_volume_ratio"] = round(vol_ratio, 6) if vol_ratio is not None else None
    feat_dict["intraday_volume_confirmation"] = vol_conf
    feat_dict["volume_confirmation_passed"] = vol_conf >= 1.0
    return vol_conf


def _set_execution_gate_features(
    feat_dict: Dict[str, Any],
    inp: PatternInput,
    pre_spread_x: float,
    freshness: tuple[bool, bool, bool, bool, float, Optional[int]],
) -> tuple[bool, bool, bool, float]:
    identity_passed = _activation_identity_passed(inp.market_data)
    quote_rej = quote_rejection(inp.market_data, quote_fields=QUOTE_FIELDS)
    spread_pct = inp.market_data.get("spread_pct_vs_eval_quote")
    spread_pct_float = finite_float(spread_pct) if spread_pct is not None else None
    spread_quality = compute_spread_quality(spread_pct_float, pre_spread_x)
    if "spread_discipline_passed" in inp.market_data and inp.market_data.get("spread_discipline_passed") is not True:
        spread_quality = 0.0
    spread_passed = spread_quality > 0
    (
        source_freshness_passed,
        watchlist_identity_passed,
        watchlist_session_match,
        freshness_passed,
        decay_weight,
        age_sessions,
    ) = freshness

    feat_dict["activation_identity_passed"] = identity_passed
    feat_dict["quote_capture_passed"] = quote_rej is None
    feat_dict["spread_pct_vs_eval_quote"] = spread_pct_float
    feat_dict["spread_quality"] = spread_quality
    feat_dict["wide_spread_exception_passed"] = (
        spread_pct_float is not None
        and SPREAD_CAP < spread_pct_float <= WIDE_SPREAD_CAP
        and spread_quality > 0
    )
    feat_dict["spread_discipline_passed"] = spread_passed
    feat_dict["signal_freshness_source_passed"] = source_freshness_passed
    feat_dict["watchlist_identity_passed"] = watchlist_identity_passed
    feat_dict["watchlist_session_match"] = watchlist_session_match
    feat_dict["watchlist_age_sessions"] = age_sessions
    feat_dict["watchlist_decay_weight"] = decay_weight
    feat_dict["signal_freshness_passed"] = freshness_passed
    return identity_passed, spread_passed, freshness_passed, spread_quality


def _enrich_m5_activation(
    feat_dict: Dict[str, Any],
    inp: PatternInput,
    decline_magnitude: float,
    support_level: Optional[float],
    sigma_20d: float,
    warnings: List[str],
    quality_flags: Dict[str, Any],
    lambda_7td: float = LAMBDA_M5_7TD,
) -> Optional[PatternSignal]:
    """Compute all activation features. Returns PatternSignal if passes."""
    reclaim_strength, stabilization = _set_reclaim_features(
        feat_dict, inp, support_level, sigma_20d,
    )
    vol_conf = _set_volume_features(feat_dict, inp, quality_flags, warnings)
    freshness = _watchlist_freshness(inp.market_data)
    (
        _source_freshness_passed,
        _watchlist_identity_passed,
        _watchlist_session_match,
        _signal_freshness_passed,
        decay_weight,
        age_sessions,
    ) = freshness
    feat_dict["watchlist_age_sessions"] = age_sessions
    feat_dict["watchlist_decay_weight"] = decay_weight
    pre_spread_x = min(
        decline_magnitude * reclaim_strength * stabilization * vol_conf * decay_weight,
        X_M5_CAP,
    )
    feat_dict["pre_spread_x_m5_at_activation"] = round(pre_spread_x, 6)
    identity_passed, spread_passed, freshness_passed, spread_quality = _set_execution_gate_features(
        feat_dict, inp, pre_spread_x, freshness,
    )

    x_m5_activation = min(
        pre_spread_x * spread_quality,
        X_M5_CAP,
    )
    feat_dict["x_m5_at_activation"] = round(x_m5_activation, 6)

    activation_passed = (
        feat_dict["support_reclaim_passed"]
        and feat_dict["stabilization_passed"]
        and feat_dict["volume_confirmation_passed"]
        and identity_passed
        and feat_dict["quote_capture_passed"]
        and spread_passed
        and freshness_passed
    )
    feat_dict["activation_passed"] = activation_passed

    if not activation_passed:
        feat_dict["activation_state"] = ACTIVATION_STATE_FAILED
        feat_dict["activation_failure_reason"] = _activation_failure_reason(feat_dict)
        feat_dict["rejection_reason"] = feat_dict["activation_failure_reason"]
        feat_dict["signal_generated"] = False
        return None

    raw_expected_edge = round(x_m5_activation * lambda_7td, 6)
    signal_strength = round(min(x_m5_activation / X_M5_CAP, 1.0), 6)

    feat_dict["activation_state"] = ACTIVATION_STATE_ACTIVATED
    feat_dict["signal_generated"] = True
    feat_dict["lambda_M5_weekly"] = LAMBDA_M5_WEEKLY
    feat_dict["microcap_amplification"] = AMPLIFICATION
    feat_dict["validated_or_shadow_lambda_M5_7td"] = lambda_7td
    feat_dict["lambda_M5_7td"] = round(lambda_7td, 8)
    feat_dict["lambda_M5_default_7td"] = round(LAMBDA_M5_7TD, 8)
    feat_dict["lambda_M5_source"] = (
        "shadow_prior" if lambda_7td == LAMBDA_M5_7TD else "validated_or_injected"
    )
    feat_dict["amplified_lambda_M5_7td"] = round(lambda_7td, 8)
    feat_dict["expected_return_priors"] = {"gross_bps": round(raw_expected_edge * 10_000, 2)}

    return PatternSignal(
        direction=SignalDirection.LONG,
        raw_signal_strength=signal_strength,
        raw_expected_edge=raw_expected_edge,
        signal_horizon=SIGNAL_HORIZON,
        route_class=RouteClass.B,
        data_confidence=_data_confidence(inp, quality_flags),
    )


def _build_watchlist_signal(
    feat_dict: Dict[str, Any], x_m5_setup: float, quality_flags: Dict[str, Any],
    inp: Optional[PatternInput] = None,
) -> PatternSignal:
    feat_dict["activation_state"] = ACTIVATION_STATE_WATCHLIST
    feat_dict["signal_generated"] = True
    feat_dict["expected_return_priors"] = {"gross_bps": 0.0}
    return PatternSignal(
        direction=SignalDirection.LONG,
        raw_signal_strength=round(min(x_m5_setup / X_M5_CAP, 1.0), 6),
        raw_expected_edge=0.0,
        signal_horizon=SIGNAL_HORIZON,
        signal_status="watchlist",
        route_class=RouteClass.B,
        data_confidence=_data_confidence(inp, quality_flags) if inp is not None else compute_data_confidence(quality_flags),
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

class M5Detector(BasePatternDetector):
    """M5 Failed Breakdown Reversal detector."""

    pattern_id = PatternId.M5
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.MEAN_REVERSION
    route_class = RouteClass.B

    def __init__(self, lambda_m5_7td: float = LAMBDA_M5_7TD):
        parsed = finite_float(lambda_m5_7td)
        if parsed is None or parsed <= 0:
            raise ValueError("lambda_m5_7td must be finite and positive")
        self._lambda_m5_7td = parsed

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        asof = require_asof_timestamp(inp.asof_timestamp)
        warnings: List[str] = []
        quality_flags: Dict[str, Any] = {}

        reject_future_timestamp(asof, warnings, quality_flags)
        require_lineage_hash(inp.lineage_hashes, warnings, quality_flags)

        return_5d = inp.market_data.get("return_5d")
        sigma_20d = inp.market_data.get("sigma_20d")
        support_level_raw = inp.market_data.get("support_level")
        low_5d_raw = inp.market_data.get("low_5d")

        if return_5d is None or sigma_20d is None:
            warnings.append("missing required fields (return_5d or sigma_20d)")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        return_5d_f = finite_float(return_5d)
        sigma_20d_f = finite_float(sigma_20d)
        if return_5d_f is None or sigma_20d_f is None or sigma_20d_f <= 0:
            warnings.append("invalid return_5d or sigma_20d")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)
        return_5d = return_5d_f
        sigma_20d = sigma_20d_f

        support_level = finite_float(support_level_raw)
        low_5d = finite_float(low_5d_raw)

        decline_mag = compute_decline_magnitude(return_5d, sigma_20d)
        support_break_weight = compute_support_break_attempt_weight(low_5d, support_level)
        x_m5_setup = min(decline_mag * support_break_weight, X_M5_CAP)
        setup_path = (
            "support_break"
            if support_break_weight == SUPPORT_BREAK_SETUP_WEIGHT
            else "decline_only"
            if support_break_weight == DECLINE_ONLY_SETUP_WEIGHT and support_level is not None
            else MISSING_SUPPORT_SETUP_PATH
            if support_break_weight == DECLINE_ONLY_SETUP_WEIGHT
            else "no_setup"
        )

        feat_dict: Dict[str, Any] = {
            "return_5d": round(return_5d, 6),
            "sigma_20d": sigma_20d,
            "decline_magnitude": round(decline_mag, 6),
            "support_level": support_level,
            "low_5d": low_5d,
            "support_break_attempted": support_break_weight == SUPPORT_BREAK_SETUP_WEIGHT,
            "setup_path": setup_path,
            "support_break_attempt_weight": support_break_weight,
            "x_m5_setup": round(x_m5_setup, 6),
        }
        _set_m5_signal_identity(feat_dict, inp)
        # Copy diagnostic fields from caller
        for source in (inp.market_data, inp.fundamental_data, inp.event_data):
            for key in ("hazard_score_at_signal", "filing_veto_status", "sector",
                        "dollar_volume_ratio", "recent_halt",
                        "m1_also_firing", "m4_also_firing", "m6_also_firing",
                        "overlapping_pattern_ids"):
                val = source.get(key)
                if val is not None:
                    feat_dict[key] = val
        feat_dict.setdefault("filing_veto_status", "not_computed")

        universe_rejection = operating_universe_rejection(
            inp.market_data, warnings, quality_flags, pattern_id=self.pattern_id,
        )
        pre_signal_rejection = market_data_quality_rejection(feat_dict, inp.market_data)
        if pre_signal_rejection is not None:
            quality_flags["market_data_quality_rejected"] = True

        pit_passed = quality_flags.get("point_in_time_passed") is not False
        fidelity = classify_fidelity(
            has_primary_data=True, has_secondary_data=True,
            point_in_time_passed=pit_passed, lookahead_guard_passed=True,
        )

        setup_qualified = decline_mag >= MIN_DECLINE_MAGNITUDE and support_break_weight > 0 and x_m5_setup > 0

        signals: List[PatternSignal] = []
        activation_requested = inp.market_data.get("price") is not None
        if activation_requested:
            copy_fields(feat_dict, inp.market_data, (*ACTIVATION_DIAGNOSTIC_FIELDS, "reversal_anchor_price"))

        if universe_rejection is not None:
            feat_dict["activation_state"] = universe_rejection
            feat_dict["activation_failure_reason"] = universe_rejection
            feat_dict["rejection_reason"] = universe_rejection
            feat_dict["signal_generated"] = False
        elif pre_signal_rejection is not None:
            feat_dict["activation_state"] = pre_signal_rejection
            feat_dict["activation_failure_reason"] = pre_signal_rejection
            feat_dict["rejection_reason"] = pre_signal_rejection
            feat_dict["signal_generated"] = False
        elif not setup_qualified:
            feat_dict["activation_state"] = ACTIVATION_STATE_NO_SETUP
            feat_dict["rejection_reason"] = "no_setup"
            feat_dict["signal_generated"] = False
        elif activation_requested:
            activation_quality_rejection = market_data_quality_rejection(
                feat_dict, inp.market_data, require_fields=True,
            )
            if activation_quality_rejection is not None:
                quality_flags["market_data_quality_rejected"] = True
                feat_dict["activation_state"] = ACTIVATION_STATE_FAILED
                feat_dict["activation_passed"] = False
                feat_dict["activation_failure_reason"] = activation_quality_rejection
                feat_dict["rejection_reason"] = activation_quality_rejection
                feat_dict["signal_generated"] = False
            else:
                sig = _enrich_m5_activation(
                    feat_dict, inp, decline_mag, support_level, sigma_20d,
                    warnings, quality_flags,
                    lambda_7td=self._lambda_m5_7td,
                )
                if sig is not None:
                    signals.append(sig)
        else:
            signals.append(_build_watchlist_signal(feat_dict, x_m5_setup, quality_flags, inp=inp))

        features = PatternFeatures(
            features=feat_dict, feature_manifest_version="m5-v1",
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
