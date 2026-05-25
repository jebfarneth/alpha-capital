"""
M6 — Volatility-Compression Breakout Detector.

Vault source: Engineering/Patterns/M6-VolCompression/

Thesis: right_tail_convex. Stocks emerging from a GK-measured
low-volatility compression regime into a breakout exhibit positive
expected excess returns over the following 12 trading days.

Exposure formula (EXPOSURE.md):
  compression_ratio = GK_vol_5d / GK_vol_60d
  compression_depth = clip((1.0 - compression_ratio) / 0.4, 0.0, 2.5)
  X_M6_setup = compression_depth
  X_M6_activation = min(compression_depth * breakout_ext * expansion_conf * vol_conf, 3.0)

Expected-return bridge (SPEC.md / EXPOSURE.md):
  lambda_monthly = 1.1% (J&T 6/6, inherited from M4)
  amplification = 1.45 (Cakici 2023 small-cap factor)
  lambda_M6_12td = 1.1% * 1.45 * (12/21) = ~0.912%
  raw_expected_edge = X_M6_activation * lambda_M6_12td
  Activated signals persist lambda_M6_monthly, microcap_amplification,
  and amplified_lambda_M6_12td for validation reconstruction.

Signal paths:
  1. Watchlist: compressed setup, no activation data
  2. Standard activation: breakout + range expansion + volume ignition + activation identity
     + quote capture + spread + watchlist freshness
  3. Early-gap activation: first 30 min, open gaps above compression_high,
     volume + activation identity + quote capture + spread + watchlist freshness required,
     range expansion not yet required

Routing: Class C after activation (marketable limit, 120-second cancel).
Evidence defaults filing_veto_status to "not_computed" when the caller
does not supply a filing veto result.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from alpha.data.contracts import stable_hash
from alpha.patterns.activation import required_fields_present, same_session_freshness
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
    compute_data_confidence as compute_quality_confidence,
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
LAMBDA_M6_MONTHLY = 0.011
AMPLIFICATION = 1.45
HOLD_DAYS = 12
LAMBDA_M6_12TD = LAMBDA_M6_MONTHLY * AMPLIFICATION * (HOLD_DAYS / 21.0)
X_M6_CAP = 3.0
SIGNAL_HORIZON = "12d"
MIN_COMPRESSION_DEPTH = 0.5
SPREAD_CAP = 0.01
EARLY_GAP_WINDOW_MINUTES = 30
QUOTE_FIELDS = DEFAULT_QUOTE_FIELDS
QUOTE_DIAGNOSTIC_FIELDS = (*QUOTE_FIELDS, "quote_freshness_max_ms")
ACTIVATION_IDENTITY_FIELDS = ("activation_id", "activation_timestamp")
WATCHLIST_FRESHNESS_FIELDS = (
    "watchlist_signal_id",
    "watchlist_scan_date",
    "watchlist_valid_session",
    "activation_session",
)
ACTIVATION_DIAGNOSTIC_FIELDS = (
    *ACTIVATION_IDENTITY_FIELDS,
    *WATCHLIST_FRESHNESS_FIELDS,
    *QUOTE_DIAGNOSTIC_FIELDS,
)

ACTIVATION_STATE_WATCHLIST = "watchlist"
ACTIVATION_STATE_ACTIVATED = "activated"
ACTIVATION_STATE_NOT_COMPRESSED = "not_compressed"
ACTIVATION_STATE_NO_BREAKOUT = "no_breakout"
ACTIVATION_STATE_FAILED = "activation_failed"


# ---------------------------------------------------------------------------
# Pure computation helpers
# ---------------------------------------------------------------------------

def compute_compression_depth(compression_ratio: float) -> float:
    return max(0.0, min((1.0 - compression_ratio) / 0.4, 2.5))


def compute_breakout_extension(price: float, compression_high: float, sigma_20d: float) -> float:
    if price <= compression_high or compression_high <= 0 or sigma_20d <= 0:
        return 0.0
    return max(0.0, min((price - compression_high) / compression_high / sigma_20d, 3.0))


def compute_expansion_confirmation(expansion_ratio: float) -> float:
    if expansion_ratio >= 2.0:
        return 1.5
    if expansion_ratio >= 1.5:
        return 1.25
    if expansion_ratio >= 1.0:
        return 1.0
    return 0.5


def compute_volume_confirmation(volume_ratio: float) -> float:
    if volume_ratio >= 2.0:
        return 1.5
    if volume_ratio >= 1.5:
        return 1.25
    if volume_ratio >= 1.0:
        return 1.0
    return 0.5


def compute_expansion_ratio(
    session_high: float, session_low: float, gk_avg_5d: float,
) -> Optional[float]:
    if session_high <= 0 or session_low <= 0 or session_high <= session_low:
        return None
    if gk_avg_5d is None or gk_avg_5d <= 0:
        return None
    return math.log(session_high / session_low) / math.sqrt(gk_avg_5d)


def compute_m6_data_confidence(
    gk_warning: bool,
    quality_flags: Dict[str, Any],
    *,
    field_confidence_sources: Optional[tuple] = None,
) -> float:
    flags = dict(quality_flags)
    if gk_warning:
        flags["gk_low_transaction_warning"] = True
    return compute_quality_confidence(flags, field_confidence_sources=field_confidence_sources)


def compute_effective_volume_confirmation(
    volume_ratio: Optional[float], latest_5m_ratio: Optional[float],
) -> float:
    confirmations = []
    if volume_ratio is not None:
        confirmations.append(compute_volume_confirmation(volume_ratio))
    if latest_5m_ratio is not None:
        confirmations.append(compute_volume_confirmation(latest_5m_ratio))
    return max(confirmations) if confirmations else 0.0


def _activation_failure_reason(feat: Dict[str, Any]) -> str:
    if not feat.get("breakout_ignition_passed", False):
        return "breakout_ignition_failed"
    if not feat.get("range_expansion_passed", False) and not feat.get("early_gap_candidate", False):
        return "range_expansion_failed"
    if not feat.get("volume_ignition_passed", False):
        return "volume_ignition_failed"
    if feat.get("activation_identity_passed") is not True:
        return "activation_identity_missing"
    if feat.get("quote_capture_passed") is not True:
        return "quote_unavailable"
    if not feat.get("spread_discipline_passed", False):
        return "spread_unavailable" if feat.get("spread_pct_vs_eval_quote") is None else "spread_too_wide"
    if not feat.get("signal_freshness_passed", False):
        return "signal_expired"
    if not feat.get("range_expansion_passed", False):
        return "range_expansion_failed"
    return "unknown"


def _set_m6_signal_identity(feat_dict: Dict[str, Any], inp: PatternInput) -> None:
    setup_id = inp.market_data.get("m6_setup_id") or inp.market_data.get("watchlist_signal_id")
    if setup_id is not None:
        components = {"m6_setup_id": setup_id}
        source = "upstream_m6_setup_id"
    else:
        components = {
            "compression_high": feat_dict.get("compression_high"),
            "compression_ratio": feat_dict.get("compression_ratio"),
        }
        source = "compression_setup_content"
    set_signal_identity(
        feat_dict,
        pattern_id=PatternId.M6,
        ticker=inp.ticker,
        components=components,
        source=source,
    )


def _activation_identity_passed(market_data: Dict[str, Any]) -> bool:
    """Executable M6 activations must be joinable to m6_intraday_activation."""
    return required_fields_present(market_data, ACTIVATION_IDENTITY_FIELDS)


def _watchlist_freshness(
    market_data: Dict[str, Any],
) -> tuple[bool, bool, bool, bool]:
    freshness = same_session_freshness(
        market_data,
        identity_fields=WATCHLIST_FRESHNESS_FIELDS,
        valid_session_field="watchlist_valid_session",
        activation_session_field="activation_session",
    )
    return (
        freshness.source_freshness_passed,
        freshness.watchlist_identity_passed,
        freshness.watchlist_session_match,
        freshness.signal_freshness_passed,
    )


# ---------------------------------------------------------------------------
# Activation enrichment (extracted from detect for readability)
# ---------------------------------------------------------------------------

def _set_breakout_features(
    feat_dict: Dict[str, Any], price: float, compression_high: float, sigma_20d: float,
) -> float:
    brk_ext = compute_breakout_extension(price, compression_high, sigma_20d)
    feat_dict["intraday_breakout_extension"] = round(brk_ext, 6)
    feat_dict["breakout_ignition_passed"] = brk_ext > 0
    return brk_ext


def _set_expansion_features(
    feat_dict: Dict[str, Any],
    inp: PatternInput,
    quality_flags: Dict[str, Any],
    warnings: List[str],
) -> float:
    session_high = inp.market_data.get("session_high")
    session_low = inp.market_data.get("session_low")
    gk_avg_5d = inp.market_data.get("gk_avg_5d")

    exp_ratio = None
    exp_conf = 1.0
    if session_high is not None and session_low is not None and gk_avg_5d is not None:
        session_high_f = finite_float(session_high)
        session_low_f = finite_float(session_low)
        gk_avg_5d_f = finite_float(gk_avg_5d)
        if session_high_f is not None and session_low_f is not None and gk_avg_5d_f is not None:
            exp_ratio = compute_expansion_ratio(session_high_f, session_low_f, gk_avg_5d_f)
        if exp_ratio is not None:
            exp_conf = compute_expansion_confirmation(exp_ratio)
        else:
            quality_flags["missing_expansion_data"] = True
            warnings.append("invalid session_high/session_low/gk_avg_5d for expansion confirmation")
    else:
        quality_flags["missing_expansion_data"] = True
        warnings.append("missing session_high/session_low/gk_avg_5d for expansion confirmation")

    feat_dict["intraday_expansion_ratio"] = round(exp_ratio, 6) if exp_ratio is not None else None
    feat_dict["intraday_expansion_confirmation"] = exp_conf
    feat_dict["range_expansion_passed"] = exp_ratio is not None and exp_ratio >= 1.0
    return exp_conf


def _set_volume_features(
    feat_dict: Dict[str, Any],
    inp: PatternInput,
    quality_flags: Dict[str, Any],
    warnings: List[str],
) -> float:
    cumulative_volume = inp.market_data.get("cumulative_volume")
    expected_tod_volume = inp.market_data.get("expected_tod_volume")
    latest_5m_volume_ratio = inp.market_data.get("latest_5m_volume_ratio")

    cumulative_vol_conf = None
    vol_ratio = None
    if cumulative_volume is not None and expected_tod_volume is not None:
        cv, ev = finite_float(cumulative_volume), finite_float(expected_tod_volume)
        if cv is not None and ev is not None and ev > 0:
            vol_ratio = cv / ev
            cumulative_vol_conf = compute_volume_confirmation(vol_ratio)
        else:
            quality_flags["missing_volume_data"] = True
            warnings.append("invalid cumulative_volume/expected_tod_volume for volume confirmation")
    else:
        quality_flags["missing_volume_data"] = True
        warnings.append("missing cumulative_volume/expected_tod_volume for volume confirmation")

    latest_5m_ratio = finite_float(latest_5m_volume_ratio) if latest_5m_volume_ratio is not None else None
    latest_5m_vol_conf = compute_volume_confirmation(latest_5m_ratio) if latest_5m_ratio is not None else None
    vol_conf = compute_effective_volume_confirmation(vol_ratio, latest_5m_ratio)
    feat_dict["intraday_volume_ratio"] = round(vol_ratio, 6) if vol_ratio is not None else None
    feat_dict["cumulative_volume_confirmation"] = cumulative_vol_conf
    feat_dict["latest_5m_volume_confirmation"] = latest_5m_vol_conf
    feat_dict["intraday_volume_confirmation"] = vol_conf
    feat_dict["latest_5m_volume_ratio"] = round(latest_5m_ratio, 6) if latest_5m_ratio is not None else None
    feat_dict["volume_ignition_passed"] = (
        (vol_ratio is not None and vol_ratio >= 1.5)
        or (latest_5m_ratio is not None and latest_5m_ratio >= 2.0)
    )
    return vol_conf


def _set_execution_gate_features(
    feat_dict: Dict[str, Any], inp: PatternInput,
) -> tuple[bool, bool, bool]:
    identity_passed = _activation_identity_passed(inp.market_data)
    quote_rej = quote_rejection(inp.market_data, quote_fields=QUOTE_FIELDS)
    spread_pct = inp.market_data.get("spread_pct_vs_eval_quote")
    spread_pct_float = finite_float(spread_pct) if spread_pct is not None else None
    spread_passed = (
        inp.market_data.get("spread_discipline_passed") is True
        if "spread_discipline_passed" in inp.market_data
        else spread_pct_float is not None and spread_pct_float <= SPREAD_CAP
    )
    (
        source_freshness_passed,
        watchlist_identity_passed,
        watchlist_session_match,
        signal_freshness_passed,
    ) = _watchlist_freshness(inp.market_data)

    feat_dict["activation_identity_passed"] = identity_passed
    feat_dict["quote_capture_passed"] = quote_rej is None
    feat_dict["spread_pct_vs_eval_quote"] = spread_pct_float
    feat_dict["spread_discipline_passed"] = spread_passed
    feat_dict["signal_freshness_source_passed"] = source_freshness_passed
    feat_dict["watchlist_identity_passed"] = watchlist_identity_passed
    feat_dict["watchlist_session_match"] = watchlist_session_match
    feat_dict["signal_freshness_passed"] = signal_freshness_passed
    return identity_passed, spread_passed, signal_freshness_passed


def _set_early_gap_features(
    feat_dict: Dict[str, Any],
    inp: PatternInput,
    compression_high: float,
    sigma_20d: float,
    brk_ext: float,
    quality_flags: Dict[str, Any],
    warnings: List[str],
) -> tuple[bool, bool, Optional[str], Optional[float]]:
    open_price = inp.market_data.get("open_price")
    prior_close = inp.market_data.get("prior_close")
    minutes_since_open = inp.market_data.get("minutes_since_open")
    gk_avg_5d = inp.market_data.get("gk_avg_5d")

    open_price_f = finite_float(open_price) if open_price is not None else None
    prior_close_f = finite_float(prior_close) if prior_close is not None else None
    minutes_since_open_f = finite_float(minutes_since_open) if minutes_since_open is not None else None
    gk_avg_5d_f = finite_float(gk_avg_5d) if gk_avg_5d is not None else None

    early_session_flag = minutes_since_open_f is not None and minutes_since_open_f <= EARLY_GAP_WINDOW_MINUTES
    gap_breakout_flag = open_price_f is not None and open_price_f > compression_high
    gap_brk_ext = compute_breakout_extension(open_price_f, compression_high, sigma_20d) if open_price_f is not None else 0.0

    gap_exp_proxy = None
    gap_exp_proxy_available = False
    gap_exp_proxy_reason = None
    if early_session_flag and gap_breakout_flag:
        if prior_close is None:
            gap_exp_proxy_reason = "missing_prior_close"
            quality_flags["missing_prior_close"] = True
            warnings.append("missing prior_close for gap expansion proxy; falling back conservatively")
        elif prior_close_f is None or prior_close_f <= 0:
            gap_exp_proxy_reason = "invalid_prior_close"
            quality_flags["invalid_prior_close"] = True
            warnings.append("invalid prior_close for gap expansion proxy; falling back conservatively")
        elif gk_avg_5d is None:
            gap_exp_proxy_reason = "missing_gk_avg_5d"
            quality_flags["missing_gk_avg_5d_for_gap_proxy"] = True
            warnings.append("missing gk_avg_5d for gap expansion proxy; falling back conservatively")
        elif gk_avg_5d_f is None or gk_avg_5d_f <= 0:
            gap_exp_proxy_reason = "invalid_gk_avg_5d"
            quality_flags["invalid_gk_avg_5d_for_gap_proxy"] = True
            warnings.append("invalid gk_avg_5d for gap expansion proxy; falling back conservatively")
        elif open_price_f <= prior_close_f:
            gap_exp_proxy_reason = "non_positive_open_gap"
        else:
            gap_exp_proxy = math.log(open_price_f / prior_close_f) / math.sqrt(gk_avg_5d_f)
            gap_exp_proxy_available = True
            gap_exp_proxy_reason = "valid"

    feat_dict["prior_close"] = prior_close_f
    feat_dict["gap_expansion_proxy"] = round(gap_exp_proxy, 6) if gap_exp_proxy is not None else None
    feat_dict["gap_expansion_proxy_available"] = gap_exp_proxy_available
    feat_dict["gap_expansion_proxy_reason"] = gap_exp_proxy_reason
    feat_dict["early_session_flag"] = early_session_flag
    feat_dict["gap_breakout_flag"] = gap_breakout_flag
    feat_dict["gap_breakout_extension"] = round(gap_brk_ext, 6)
    early_gap_candidate = early_session_flag and gap_breakout_flag and gap_brk_ext > 0 and brk_ext > 0
    feat_dict["early_gap_candidate"] = early_gap_candidate
    return early_gap_candidate, gap_exp_proxy_available, gap_exp_proxy_reason, gap_exp_proxy


def _set_activation_path_features(
    feat_dict: Dict[str, Any],
    *,
    identity_passed: bool,
    spread_passed: bool,
    signal_freshness_passed: bool,
    early_gap_candidate: bool,
) -> tuple[bool, bool, bool]:
    standard_passed = (
        feat_dict["breakout_ignition_passed"]
        and feat_dict["range_expansion_passed"]
        and feat_dict["volume_ignition_passed"]
        and identity_passed
        and feat_dict["quote_capture_passed"]
        and spread_passed
        and signal_freshness_passed
    )
    early_gap_passed = (
        early_gap_candidate
        and feat_dict["volume_ignition_passed"]
        and identity_passed
        and feat_dict["quote_capture_passed"]
        and spread_passed
        and signal_freshness_passed
    )
    activation_passed = standard_passed or early_gap_passed

    feat_dict["standard_activation_passed"] = standard_passed
    feat_dict["early_gap_activation_passed"] = early_gap_passed
    feat_dict["activation_passed"] = activation_passed
    if standard_passed:
        feat_dict["activation_path"] = "standard"
    elif early_gap_passed:
        feat_dict["activation_path"] = "early_gap_activation"
    else:
        feat_dict["activation_path"] = None
    return activation_passed, standard_passed, early_gap_passed


def _effective_expansion_confirmation(
    feat_dict: Dict[str, Any],
    *,
    standard_passed: bool,
    early_gap_passed: bool,
    exp_conf: float,
    gap_exp_proxy_available: bool,
    gap_exp_proxy_reason: Optional[str],
    gap_exp_proxy: Optional[float],
) -> float:
    if standard_passed:
        return exp_conf
    if not early_gap_passed:
        return exp_conf
    if feat_dict["range_expansion_passed"]:
        return exp_conf
    if gap_exp_proxy_available and gap_exp_proxy is not None:
        effective_exp_conf = compute_expansion_confirmation(gap_exp_proxy)
    elif gap_exp_proxy_reason == "non_positive_open_gap":
        effective_exp_conf = 0.5
    else:
        effective_exp_conf = 1.0
    feat_dict["early_gap_expansion_confirmation"] = effective_exp_conf
    return effective_exp_conf


def _set_activation_rejection(feat_dict: Dict[str, Any], brk_ext: float) -> None:
    feat_dict["activation_state"] = ACTIVATION_STATE_FAILED if brk_ext > 0 else ACTIVATION_STATE_NO_BREAKOUT
    feat_dict["activation_failure_reason"] = _activation_failure_reason(feat_dict)
    feat_dict["rejection_reason"] = feat_dict["activation_failure_reason"]
    feat_dict["signal_generated"] = False


def _build_activation_signal(
    feat_dict: Dict[str, Any],
    x_m6_activation: float,
    gk_warning: bool,
    quality_flags: Dict[str, Any],
    lambda_12td: float = LAMBDA_M6_12TD,
    field_confidence_sources: Optional[tuple] = None,
) -> PatternSignal:
    raw_expected_edge = round(x_m6_activation * lambda_12td, 6)
    signal_strength = round(min(x_m6_activation / X_M6_CAP, 1.0), 6)

    feat_dict["activation_state"] = ACTIVATION_STATE_ACTIVATED
    feat_dict["signal_generated"] = True
    feat_dict["lambda_M6_monthly"] = LAMBDA_M6_MONTHLY
    feat_dict["microcap_amplification"] = AMPLIFICATION
    feat_dict["validated_or_shadow_lambda_M6_12td"] = lambda_12td
    feat_dict["lambda_M6_12td"] = round(lambda_12td, 8)
    feat_dict["lambda_M6_default_12td"] = round(LAMBDA_M6_12TD, 8)
    feat_dict["lambda_M6_source"] = (
        "shadow_prior" if lambda_12td == LAMBDA_M6_12TD else "validated_or_injected"
    )
    feat_dict["amplified_lambda_M6_12td"] = round(lambda_12td, 8)
    feat_dict["expected_return_priors"] = {"gross_bps": round(raw_expected_edge * 10_000, 2)}

    return PatternSignal(
        direction=SignalDirection.LONG,
        raw_signal_strength=signal_strength,
        raw_expected_edge=raw_expected_edge,
        signal_horizon=SIGNAL_HORIZON,
        route_class=RouteClass.C,
        data_confidence=compute_m6_data_confidence(gk_warning, quality_flags, field_confidence_sources=field_confidence_sources),
    )


def _enrich_m6_activation(
    feat_dict: Dict[str, Any],
    inp: PatternInput,
    depth: float,
    compression_high: float,
    sigma_20d: float,
    gk_warning: bool,
    warnings: List[str],
    quality_flags: Dict[str, Any],
    lambda_12td: float = LAMBDA_M6_12TD,
    field_confidence_sources: Optional[tuple] = None,
) -> Optional[PatternSignal]:
    """
    Compute all activation features and gates for a compressed breakout.
    Returns a PatternSignal if activation passes, else None.
    All activation fields are set on feat_dict here.
    """
    price = finite_float(inp.market_data.get("price"))
    if price is None or price <= 0:
        feat_dict["activation_state"] = ACTIVATION_STATE_FAILED
        feat_dict["activation_passed"] = False
        feat_dict["activation_failure_reason"] = "breakout_ignition_failed"
        feat_dict["rejection_reason"] = "breakout_ignition_failed"
        feat_dict["signal_generated"] = False
        quality_flags["invalid_activation_price"] = True
        warnings.append("invalid price for M6 activation")
        return None
    feat_dict["P_activation"] = price
    copy_fields(feat_dict, inp.market_data, ACTIVATION_DIAGNOSTIC_FIELDS)

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
        warnings.append(f"M6 activation failed: {activation_quality_rejection}")
        return None

    brk_ext = _set_breakout_features(feat_dict, price, compression_high, sigma_20d)
    exp_conf = _set_expansion_features(feat_dict, inp, quality_flags, warnings)
    vol_conf = _set_volume_features(feat_dict, inp, quality_flags, warnings)
    identity_passed, spread_passed, signal_freshness_passed = _set_execution_gate_features(feat_dict, inp)
    (
        early_gap_candidate,
        gap_exp_proxy_available,
        gap_exp_proxy_reason,
        gap_exp_proxy,
    ) = _set_early_gap_features(
        feat_dict, inp, compression_high, sigma_20d, brk_ext, quality_flags, warnings,
    )
    activation_passed, standard_passed, early_gap_passed = _set_activation_path_features(
        feat_dict,
        identity_passed=identity_passed,
        spread_passed=spread_passed,
        signal_freshness_passed=signal_freshness_passed,
        early_gap_candidate=early_gap_candidate,
    )
    effective_exp_conf = _effective_expansion_confirmation(
        feat_dict,
        standard_passed=standard_passed,
        early_gap_passed=early_gap_passed,
        exp_conf=exp_conf,
        gap_exp_proxy_available=gap_exp_proxy_available,
        gap_exp_proxy_reason=gap_exp_proxy_reason,
        gap_exp_proxy=gap_exp_proxy,
    )
    x_m6_activation = min(depth * brk_ext * effective_exp_conf * vol_conf, X_M6_CAP)
    feat_dict["X_M6_activation"] = round(x_m6_activation, 6)

    if not activation_passed:
        _set_activation_rejection(feat_dict, brk_ext)
        return None

    return _build_activation_signal(
        feat_dict, x_m6_activation, gk_warning, quality_flags,
        lambda_12td=lambda_12td,
        field_confidence_sources=field_confidence_sources,
    )


def _build_watchlist_signal(
    feat_dict: Dict[str, Any], depth: float, gk_warning: bool, quality_flags: Dict[str, Any],
    field_confidence_sources: Optional[tuple] = None,
) -> PatternSignal:
    """Build a watchlist signal for a compressed setup without activation data."""
    feat_dict["activation_state"] = ACTIVATION_STATE_WATCHLIST
    feat_dict["signal_generated"] = True
    feat_dict["expected_return_priors"] = {"gross_bps": 0.0}
    return PatternSignal(
        direction=SignalDirection.LONG,
        raw_signal_strength=round(min(depth / X_M6_CAP, 1.0), 6),
        raw_expected_edge=0.0,
        signal_horizon=SIGNAL_HORIZON,
        signal_status="watchlist",
        route_class=RouteClass.C,
        data_confidence=compute_m6_data_confidence(gk_warning, quality_flags, field_confidence_sources=field_confidence_sources),
    )


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

class M6Detector(BasePatternDetector):
    """M6 Volatility-Compression Breakout detector."""

    pattern_id = PatternId.M6
    version = "1.0"
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.RIGHT_TAIL_CONVEX
    route_class = RouteClass.C

    def __init__(self, lambda_m6_12td: float = LAMBDA_M6_12TD):
        parsed = finite_float(lambda_m6_12td)
        if parsed is None or parsed <= 0:
            raise ValueError("lambda_m6_12td must be finite and positive")
        self._lambda_m6_12td = parsed

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        asof = require_asof_timestamp(inp.asof_timestamp)
        warnings: List[str] = []
        quality_flags: Dict[str, Any] = {}

        reject_future_timestamp(asof, warnings, quality_flags)
        require_lineage_hash(inp.lineage_hashes, warnings, quality_flags)

        compression_ratio = inp.market_data.get("compression_ratio")
        compression_high = inp.market_data.get("compression_high")
        sigma_20d = inp.market_data.get("sigma_20d")

        if compression_ratio is None or compression_high is None or sigma_20d is None:
            warnings.append("missing required compression features")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        compression_ratio_f = finite_float(compression_ratio)
        compression_high_f = finite_float(compression_high)
        sigma_20d_f = finite_float(sigma_20d)

        if (
            compression_ratio_f is None
            or compression_high_f is None
            or sigma_20d_f is None
            or compression_high_f <= 0
            or sigma_20d_f <= 0
        ):
            warnings.append("invalid compression features")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        compression_ratio = compression_ratio_f
        compression_high = compression_high_f
        sigma_20d = sigma_20d_f

        depth = compute_compression_depth(compression_ratio)
        gk_vol_5d = inp.market_data.get("gk_vol_5d")
        gk_vol_60d = inp.market_data.get("gk_vol_60d")

        feat_dict: Dict[str, Any] = {
            "compression_ratio": round(compression_ratio, 6),
            "compression_depth": round(depth, 6),
            "gk_vol_5d": finite_float(gk_vol_5d) if gk_vol_5d is not None else None,
            "gk_vol_60d": finite_float(gk_vol_60d) if gk_vol_60d is not None else None,
            "compression_high": compression_high,
            "sigma_20d": sigma_20d,
            "X_M6_setup": round(depth, 6),
        }
        _set_m6_signal_identity(feat_dict, inp)
        for source in (inp.market_data, inp.fundamental_data, inp.event_data):
            for _k in ("filing_veto_status", "m4_also_firing", "m5_also_firing", "overlapping_pattern_ids"):
                if _k in source:
                    feat_dict[_k] = source[_k]
        feat_dict.setdefault("filing_veto_status", "not_computed")

        gk_warning = inp.market_data.get("gk_low_transaction_warning", False)
        feat_dict["gk_low_transaction_warning"] = gk_warning

        market_cap = inp.fundamental_data.get("market_cap")
        market_cap_f = finite_float(market_cap) if market_cap is not None else None
        if market_cap_f is not None:
            feat_dict["market_cap_mm"] = round(market_cap_f / 1e6, 1)
        if "sector" in inp.fundamental_data:
            feat_dict["sector"] = inp.fundamental_data["sector"]

        universe_rejection = operating_universe_rejection(
            inp.market_data, warnings, quality_flags, pattern_id=self.pattern_id,
        )
        pre_signal_rejection = market_data_quality_rejection(feat_dict, inp.market_data)
        if pre_signal_rejection is not None:
            quality_flags["market_data_quality_rejected"] = True

        gk_vol_60d_f = finite_float(gk_vol_60d) if gk_vol_60d is not None else None
        has_ohlcv = gk_vol_60d_f is not None and gk_vol_60d_f > 0
        pit_passed = quality_flags.get("point_in_time_passed") is not False
        fidelity = classify_fidelity(
            has_primary_data=has_ohlcv, has_secondary_data=not gk_warning,
            point_in_time_passed=pit_passed, lookahead_guard_passed=True,
        )

        compressed = depth >= MIN_COMPRESSION_DEPTH
        feat_dict["compression_gate_passed"] = compressed

        signals: List[PatternSignal] = []

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
        elif compressed and inp.market_data.get("price") is not None:
            sig = _enrich_m6_activation(
                feat_dict, inp, depth, compression_high, sigma_20d,
                gk_warning, warnings, quality_flags,
                lambda_12td=self._lambda_m6_12td,
                field_confidence_sources=(inp.market_data, inp.fundamental_data, inp.event_data),
            )
            if sig is not None:
                signals.append(sig)
        elif compressed:
            signals.append(_build_watchlist_signal(
                feat_dict, depth, gk_warning, quality_flags,
                field_confidence_sources=(inp.market_data, inp.fundamental_data, inp.event_data),
            ))
        else:
            feat_dict["activation_state"] = ACTIVATION_STATE_NOT_COMPRESSED
            feat_dict["rejection_reason"] = "not_compressed"
            feat_dict["signal_generated"] = False

        features = PatternFeatures(
            features=feat_dict, feature_manifest_version="m6-v1",
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
