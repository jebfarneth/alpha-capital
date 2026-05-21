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

Signal paths:
  1. Watchlist: compressed setup, no activation data
  2. Standard activation: breakout + range expansion + volume ignition + spread + freshness
  3. Early-gap activation: first 30 min, open gaps above compression_high,
     volume + spread + freshness required, range expansion not yet required

Routing: Class C after activation (marketable limit, 120-second cancel).
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
    compute_data_confidence as compute_quality_confidence,
    operating_universe_rejection,
    reject_future_timestamp,
    require_asof_timestamp,
    require_lineage_hash,
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
QUOTE_FIELDS = ("candidate_eval_bid", "candidate_eval_ask", "candidate_eval_quote_timestamp", "quote_age_ms")
QUOTE_DIAGNOSTIC_FIELDS = (*QUOTE_FIELDS, "quote_freshness_max_ms")
DATA_QUALITY_FIELDS = ("market_data_status", "halt_status", "corporate_action_filter_passed")

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


def compute_data_confidence(gk_warning: bool, quality_flags: Dict[str, Any]) -> float:
    flags = dict(quality_flags)
    if gk_warning:
        flags["gk_low_transaction_warning"] = True
    return compute_quality_confidence(flags)


def compute_effective_volume_confirmation(
    volume_ratio: Optional[float], latest_5m_ratio: Optional[float],
) -> float:
    confirmations = []
    if volume_ratio is not None:
        confirmations.append(compute_volume_confirmation(volume_ratio))
    if latest_5m_ratio is not None:
        confirmations.append(compute_volume_confirmation(latest_5m_ratio))
    return max(confirmations) if confirmations else 1.0


def _copy_quote_fields(feat_dict: Dict[str, Any], market_data: Dict[str, Any]) -> None:
    for key in QUOTE_DIAGNOSTIC_FIELDS:
        if key in market_data:
            feat_dict[key] = market_data[key]


def _quote_rejection(market_data: Dict[str, Any]) -> str | None:
    if any(market_data.get(field) is None for field in QUOTE_FIELDS):
        return "quote_unavailable"
    if float(market_data["candidate_eval_bid"]) <= 0 or float(market_data["candidate_eval_ask"]) <= 0:
        return "quote_unavailable"
    quote_age_ms = int(market_data["quote_age_ms"])
    if quote_age_ms < 0:
        return "quote_unavailable"
    max_age_ms = market_data.get("quote_freshness_max_ms")
    if max_age_ms is not None and quote_age_ms > int(max_age_ms):
        return "quote_unavailable"
    return None


def _pre_signal_rejection(feat_dict: Dict[str, Any], market_data: Dict[str, Any]) -> str | None:
    for key in DATA_QUALITY_FIELDS:
        if key in market_data:
            feat_dict[key] = market_data[key]
    market_data_status = market_data.get("market_data_status")
    if market_data_status in {"delayed", "partial_outage", "unavailable", "stale"}:
        return "data_delay"
    halt_status = market_data.get("halt_status")
    if halt_status is not None and halt_status != "clear":
        return "halted"
    if market_data.get("corporate_action_filter_passed") is False:
        return "spurious_corporate_action"
    return None


def _activation_failure_reason(feat: Dict[str, Any]) -> str:
    if not feat.get("breakout_ignition_passed", False):
        return "breakout_ignition_failed"
    if not feat.get("range_expansion_passed", False) and not feat.get("early_gap_candidate", False):
        return "range_expansion_failed"
    if not feat.get("volume_ignition_passed", False):
        return "volume_ignition_failed"
    if not feat.get("quote_capture_passed", True):
        return "quote_unavailable"
    if not feat.get("spread_discipline_passed", False):
        return "spread_unavailable" if feat.get("spread_pct_vs_eval_quote") is None else "spread_too_wide"
    if not feat.get("signal_freshness_passed", False):
        return "signal_expired"
    if not feat.get("range_expansion_passed", False):
        return "range_expansion_failed"
    return "unknown"


# ---------------------------------------------------------------------------
# Activation enrichment (extracted from detect for readability)
# ---------------------------------------------------------------------------

def _enrich_m6_activation(
    feat_dict: Dict[str, Any],
    inp: PatternInput,
    depth: float,
    compression_high: float,
    sigma_20d: float,
    gk_warning: bool,
    warnings: List[str],
    quality_flags: Dict[str, Any],
) -> Optional[PatternSignal]:
    """
    Compute all activation features and gates for a compressed breakout.
    Returns a PatternSignal if activation passes, else None.
    All activation fields are set on feat_dict here.
    """
    price = float(inp.market_data["price"])
    feat_dict["P_activation"] = price
    _copy_quote_fields(feat_dict, inp.market_data)

    # Breakout extension
    brk_ext = compute_breakout_extension(price, compression_high, sigma_20d)
    feat_dict["intraday_breakout_extension"] = round(brk_ext, 6)
    feat_dict["breakout_ignition_passed"] = brk_ext > 0

    # Expansion confirmation
    session_high = inp.market_data.get("session_high")
    session_low = inp.market_data.get("session_low")
    gk_avg_5d = inp.market_data.get("gk_avg_5d")

    exp_ratio = None
    exp_conf = 1.0
    if session_high is not None and session_low is not None and gk_avg_5d is not None:
        exp_ratio = compute_expansion_ratio(float(session_high), float(session_low), float(gk_avg_5d))
        if exp_ratio is not None:
            exp_conf = compute_expansion_confirmation(exp_ratio)
    else:
        exp_conf = 1.0
        quality_flags["missing_expansion_data"] = True
        warnings.append("missing session_high/session_low/gk_avg_5d for expansion confirmation")

    feat_dict["intraday_expansion_ratio"] = round(exp_ratio, 6) if exp_ratio is not None else None
    feat_dict["intraday_expansion_confirmation"] = exp_conf
    feat_dict["range_expansion_passed"] = exp_ratio is not None and exp_ratio >= 1.0

    # Volume confirmation
    cumulative_volume = inp.market_data.get("cumulative_volume")
    expected_tod_volume = inp.market_data.get("expected_tod_volume")
    latest_5m_volume_ratio = inp.market_data.get("latest_5m_volume_ratio")

    cumulative_vol_conf = None
    vol_ratio = None
    if cumulative_volume is not None and expected_tod_volume is not None:
        cv, ev = float(cumulative_volume), float(expected_tod_volume)
        if ev > 0:
            vol_ratio = cv / ev
            cumulative_vol_conf = compute_volume_confirmation(vol_ratio)
    else:
        quality_flags["missing_volume_data"] = True
        warnings.append("missing cumulative_volume/expected_tod_volume for volume confirmation")

    latest_5m_ratio = float(latest_5m_volume_ratio) if latest_5m_volume_ratio is not None else None
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

    # Spread discipline
    quote_rejection = _quote_rejection(inp.market_data)
    feat_dict["quote_capture_passed"] = quote_rejection is None
    spread_pct = inp.market_data.get("spread_pct_vs_eval_quote")
    spread_passed = (
        bool(inp.market_data.get("spread_discipline_passed"))
        if "spread_discipline_passed" in inp.market_data
        else spread_pct is not None and float(spread_pct) <= SPREAD_CAP
    )
    feat_dict["spread_pct_vs_eval_quote"] = float(spread_pct) if spread_pct is not None else None
    feat_dict["spread_discipline_passed"] = spread_passed

    # Signal freshness
    signal_freshness_passed = bool(inp.market_data.get("signal_freshness_passed", True))
    feat_dict["signal_freshness_passed"] = signal_freshness_passed

    # Early-gap diagnostics
    open_price = inp.market_data.get("open_price")
    prior_close = inp.market_data.get("prior_close")
    minutes_since_open = inp.market_data.get("minutes_since_open")

    early_session_flag = minutes_since_open is not None and float(minutes_since_open) <= EARLY_GAP_WINDOW_MINUTES
    gap_breakout_flag = open_price is not None and float(open_price) > compression_high
    gap_brk_ext = compute_breakout_extension(float(open_price), compression_high, sigma_20d) if open_price is not None else 0.0

    # Gap expansion proxy: ln(open / prior_close) / sqrt(gk_avg_5d).
    # Long-only: only positive gaps get expansion credit.
    gap_exp_proxy = None
    gap_exp_proxy_available = False
    gap_exp_proxy_reason = None
    if early_session_flag and gap_breakout_flag:
        if prior_close is None:
            gap_exp_proxy_reason = "missing_prior_close"
            quality_flags["missing_prior_close"] = True
            warnings.append("missing prior_close for gap expansion proxy; falling back conservatively")
        elif float(prior_close) <= 0:
            gap_exp_proxy_reason = "invalid_prior_close"
            quality_flags["invalid_prior_close"] = True
            warnings.append("invalid prior_close for gap expansion proxy; falling back conservatively")
        elif gk_avg_5d is None:
            gap_exp_proxy_reason = "missing_gk_avg_5d"
            quality_flags["missing_gk_avg_5d_for_gap_proxy"] = True
            warnings.append("missing gk_avg_5d for gap expansion proxy; falling back conservatively")
        elif float(gk_avg_5d) <= 0:
            gap_exp_proxy_reason = "invalid_gk_avg_5d"
            quality_flags["invalid_gk_avg_5d_for_gap_proxy"] = True
            warnings.append("invalid gk_avg_5d for gap expansion proxy; falling back conservatively")
        elif float(open_price) <= float(prior_close):
            gap_exp_proxy_reason = "non_positive_open_gap"
        else:
            gap_exp_proxy = math.log(float(open_price) / float(prior_close)) / math.sqrt(float(gk_avg_5d))
            gap_exp_proxy_available = True
            gap_exp_proxy_reason = "valid"

    feat_dict["prior_close"] = float(prior_close) if prior_close is not None else None
    feat_dict["gap_expansion_proxy"] = round(gap_exp_proxy, 6) if gap_exp_proxy is not None else None
    feat_dict["gap_expansion_proxy_available"] = gap_exp_proxy_available
    feat_dict["gap_expansion_proxy_reason"] = gap_exp_proxy_reason
    feat_dict["early_session_flag"] = early_session_flag
    feat_dict["gap_breakout_flag"] = gap_breakout_flag
    feat_dict["gap_breakout_extension"] = round(gap_brk_ext, 6)
    early_gap_candidate = early_session_flag and gap_breakout_flag and gap_brk_ext > 0 and brk_ext > 0
    feat_dict["early_gap_candidate"] = early_gap_candidate

    # Standard activation
    standard_passed = (
        feat_dict["breakout_ignition_passed"]
        and feat_dict["range_expansion_passed"]
        and feat_dict["volume_ignition_passed"]
        and feat_dict["quote_capture_passed"]
        and spread_passed
        and signal_freshness_passed
    )
    feat_dict["standard_activation_passed"] = standard_passed

    # Early-gap activation
    early_gap_passed = (
        early_gap_candidate
        and feat_dict["volume_ignition_passed"]
        and feat_dict["quote_capture_passed"]
        and spread_passed
        and signal_freshness_passed
    )
    feat_dict["early_gap_activation_passed"] = early_gap_passed

    activation_passed = standard_passed or early_gap_passed
    feat_dict["activation_passed"] = activation_passed

    if standard_passed:
        feat_dict["activation_path"] = "standard"
    elif early_gap_passed:
        feat_dict["activation_path"] = "early_gap_activation"
    else:
        feat_dict["activation_path"] = None

    # Determine expansion confirmation for exposure calculation
    if standard_passed:
        effective_exp_conf = exp_conf
    elif early_gap_passed:
        if feat_dict["range_expansion_passed"]:
            effective_exp_conf = exp_conf
        elif gap_exp_proxy_available:
            # Use gap-derived expansion proxy through the standard tier function
            effective_exp_conf = compute_expansion_confirmation(gap_exp_proxy)
            feat_dict["early_gap_expansion_confirmation"] = effective_exp_conf
        elif gap_exp_proxy_reason == "non_positive_open_gap":
            effective_exp_conf = 0.5
            feat_dict["early_gap_expansion_confirmation"] = 0.5
        else:
            # No gap proxy data — conservative neutral fallback
            effective_exp_conf = 1.0
            feat_dict["early_gap_expansion_confirmation"] = 1.0
    else:
        effective_exp_conf = exp_conf

    # Activation exposure (uses effective expansion confirmation)
    x_m6_activation = min(depth * brk_ext * effective_exp_conf * vol_conf, X_M6_CAP)
    feat_dict["X_M6_activation"] = round(x_m6_activation, 6)

    if not activation_passed:
        feat_dict["activation_state"] = ACTIVATION_STATE_FAILED if brk_ext > 0 else ACTIVATION_STATE_NO_BREAKOUT
        feat_dict["activation_failure_reason"] = _activation_failure_reason(feat_dict)
        return None

    # Build signal
    raw_expected_edge = round(x_m6_activation * LAMBDA_M6_12TD, 6)
    signal_strength = round(min(x_m6_activation / X_M6_CAP, 1.0), 6)

    feat_dict["activation_state"] = ACTIVATION_STATE_ACTIVATED
    feat_dict["expected_return_priors"] = {"gross_bps": round(raw_expected_edge * 10_000, 2)}

    return PatternSignal(
        direction=SignalDirection.LONG,
        raw_signal_strength=signal_strength,
        raw_expected_edge=raw_expected_edge,
        signal_horizon=SIGNAL_HORIZON,
        data_confidence=compute_data_confidence(gk_warning, quality_flags),
    )


def _build_watchlist_signal(
    feat_dict: Dict[str, Any], depth: float, gk_warning: bool, quality_flags: Dict[str, Any],
) -> PatternSignal:
    """Build a watchlist signal for a compressed setup without activation data."""
    feat_dict["activation_state"] = ACTIVATION_STATE_WATCHLIST
    feat_dict["expected_return_priors"] = {"gross_bps": 0.0}
    return PatternSignal(
        direction=SignalDirection.LONG,
        raw_signal_strength=round(min(depth / X_M6_CAP, 1.0), 6),
        raw_expected_edge=0.0,
        signal_horizon=SIGNAL_HORIZON,
        signal_status="watchlist",
        data_confidence=compute_data_confidence(gk_warning, quality_flags),
    )


def _compute_hashes(
    inp: PatternInput, asof: Any, feat_dict: Dict[str, Any],
    signals: List[PatternSignal], warnings: List[str], quality_flags: Dict[str, Any],
) -> tuple:
    input_hash = stable_hash({
        "ticker": inp.ticker, "asof_timestamp": asof,
        "market_data": inp.market_data, "fundamental_data": inp.fundamental_data,
        "lineage_hashes": inp.lineage_hashes, "universe_snapshot_id": inp.universe_snapshot_id,
    })
    output_hash = stable_hash({
        "features": feat_dict,
        "signals": [
            {"direction": s.direction, "raw_signal_strength": s.raw_signal_strength,
             "raw_expected_edge": s.raw_expected_edge, "signal_horizon": s.signal_horizon,
             "signal_status": s.signal_status, "data_confidence": s.data_confidence}
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
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.RIGHT_TAIL_CONVEX
    route_class = RouteClass.C

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

        compression_ratio = float(compression_ratio)
        compression_high = float(compression_high)
        sigma_20d = float(sigma_20d)

        if compression_high <= 0 or sigma_20d <= 0:
            warnings.append(f"invalid compression_high={compression_high} or sigma_20d={sigma_20d}")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        depth = compute_compression_depth(compression_ratio)
        gk_vol_5d = inp.market_data.get("gk_vol_5d")
        gk_vol_60d = inp.market_data.get("gk_vol_60d")

        feat_dict: Dict[str, Any] = {
            "compression_ratio": round(compression_ratio, 6),
            "compression_depth": round(depth, 6),
            "gk_vol_5d": float(gk_vol_5d) if gk_vol_5d is not None else None,
            "gk_vol_60d": float(gk_vol_60d) if gk_vol_60d is not None else None,
            "compression_high": compression_high,
            "sigma_20d": sigma_20d,
            "X_M6_setup": round(depth, 6),
        }

        gk_warning = inp.market_data.get("gk_low_transaction_warning", False)
        feat_dict["gk_low_transaction_warning"] = gk_warning

        market_cap = inp.fundamental_data.get("market_cap")
        if market_cap is not None:
            feat_dict["market_cap_mm"] = round(float(market_cap) / 1e6, 1)
        if "sector" in inp.fundamental_data:
            feat_dict["sector"] = inp.fundamental_data["sector"]

        universe_rejection = operating_universe_rejection(
            inp.market_data, warnings, quality_flags, pattern_id=self.pattern_id,
        )
        pre_signal_rejection = _pre_signal_rejection(feat_dict, inp.market_data)
        if pre_signal_rejection is not None:
            quality_flags["market_data_quality_rejected"] = True

        has_ohlcv = gk_vol_60d is not None and float(gk_vol_60d) > 0
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
        elif pre_signal_rejection is not None:
            feat_dict["activation_state"] = pre_signal_rejection
            feat_dict["activation_failure_reason"] = pre_signal_rejection
        elif compressed and inp.market_data.get("price") is not None:
            sig = _enrich_m6_activation(
                feat_dict, inp, depth, compression_high, sigma_20d,
                gk_warning, warnings, quality_flags,
            )
            if sig is not None:
                signals.append(sig)
        elif compressed:
            signals.append(_build_watchlist_signal(feat_dict, depth, gk_warning, quality_flags))
        else:
            feat_dict["activation_state"] = ACTIVATION_STATE_NOT_COMPRESSED

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
