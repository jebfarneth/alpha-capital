"""
I8 — Opening Range Breakout Detector.

Vault source: Engineering/Patterns/I8-OpeningRangeBreakout/

Thesis: right_tail_convex. Stocks that break above their completed
9:30-10:00 opening range with volume, range, and spread confirmation
exhibit continuation over 1-3 trading days. HKS (2010): opening
half-hour = ~6x mid-day predictability.

Exposure formula (EXPOSURE.md):
  X_I8 = breakout_strength * volume_quality * range_quality * spread_quality
  breakout_strength = clip((price - range_high) / (sigma_20d * range_high), 0, 4)
  Minimum breakout gate: breakout_strength >= 0.5
  Spread gate: spread_quality > 0 (spread <= 2.5x normal)

Signal fires intraday 10:00-10:30 ET after opening bar completes.
Routing: Class C (marketable limit, 120-second cancel).
"""

from __future__ import annotations

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
    operating_universe_rejection,
    reject_future_timestamp,
    require_asof_timestamp,
    require_lineage_hash,
)

# Vault constants (EXPOSURE.md / SPEC.md)
LAMBDA_I8_3TD_DEFAULT = 0.0025  # ~25 bp shadow prior; replaced by validated lambda
X_I8_CAP = 5.0
X_I8_STRENGTH_DIVISOR = 5.0  # DATA.md: signal_strength = min(X_I8 / 5.0, 1.0)
SIGNAL_HORIZON = "3d"
MIN_BREAKOUT_STRENGTH = 0.5
BREAKOUT_STRENGTH_CAP = 4.0

QUOTE_FIELDS = ("candidate_eval_bid", "candidate_eval_ask", "candidate_eval_quote_timestamp", "quote_age_ms")
TIMESTAMP_FIELDS = ("opening_bar_close_timestamp", "breakout_eval_timestamp", "data_cutoff_timestamp")


# ---------------------------------------------------------------------------
# Pure computation helpers
# ---------------------------------------------------------------------------

def compute_breakout_strength(
    breakout_price: float, opening_range_high: float, sigma_20d: float,
) -> float:
    """Per EXPOSURE.md: clip((price - high) / (sigma * high), 0, 4)."""
    if breakout_price <= opening_range_high or opening_range_high <= 0 or sigma_20d <= 0:
        return 0.0
    raw = (breakout_price - opening_range_high) / (sigma_20d * opening_range_high)
    return max(0.0, min(raw, BREAKOUT_STRENGTH_CAP))


def compute_volume_quality(volume_ratio: float) -> float:
    """Per EXPOSURE.md tiered weighting."""
    if volume_ratio >= 2.0:
        return 1.5
    if volume_ratio >= 1.5:
        return 1.25
    if volume_ratio >= 1.0:
        return 1.0
    return 0.5


def compute_range_quality(range_ratio: float) -> float:
    """Per EXPOSURE.md tiered weighting."""
    if range_ratio >= 1.5:
        return 1.5
    if range_ratio >= 1.0:
        return 1.25
    if range_ratio >= 0.7:
        return 1.0
    return 0.75


def compute_spread_quality(spread_at_eval_bps: float, normal_spread_20d_bps: float) -> float:
    """Per EXPOSURE.md tiered weighting. Returns 0.0 if spread too wide (hard gate)."""
    if normal_spread_20d_bps <= 0:
        return 0.75  # no baseline available — use conservative wide-spread proxy
    ratio = spread_at_eval_bps / normal_spread_20d_bps
    if ratio <= 1.0:
        return 1.25
    if ratio <= 1.5:
        return 1.0
    if ratio <= 2.5:
        return 0.75
    return 0.0  # hard gate: do not signal


def _data_confidence(quality_flags: Dict[str, Any]) -> float:
    return compute_data_confidence(quality_flags)


def _copy_diagnostic_fields(feat_dict: Dict[str, Any], market_data: Dict[str, Any]) -> None:
    for key in (
        "run_id", "candidate_eval_id", "opening_bar_close_timestamp", "breakout_eval_timestamp",
        "data_cutoff_timestamp", "candidate_eval_bid", "candidate_eval_ask",
        "candidate_eval_quote_timestamp", "quote_age_ms", "quote_freshness_max_ms",
        "hks_lag13_return", "i1_also_firing", "late_evaluation",
        "halted_during_opening", "insufficient_bar_data",
        "hazard_score_at_signal", "filing_veto_status",
    ):
        val = market_data.get(key)
        if val is not None:
            feat_dict[key] = val
    feat_dict.setdefault("filing_veto_status", "clear")
    feat_dict.setdefault("late_evaluation", False)


def _pre_signal_rejection(feat_dict: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[str]:
    if market_data.get("halted_during_opening"):
        return "halted_during_opening"
    if market_data.get("insufficient_bar_data"):
        return "insufficient_bar_data"
    if any(market_data.get(f) is None for f in TIMESTAMP_FIELDS):
        return "insufficient_bar_data"
    if market_data.get("late_evaluation"):
        return "late_evaluation_stale"
    if any(market_data.get(f) is None for f in QUOTE_FIELDS):
        return "quote_unavailable"
    if float(market_data["candidate_eval_bid"]) <= 0 or float(market_data["candidate_eval_ask"]) <= 0:
        return "quote_unavailable"
    quote_age = int(market_data["quote_age_ms"])
    if quote_age < 0:
        return "quote_unavailable"
    max_age = market_data.get("quote_freshness_max_ms")
    if max_age is not None and quote_age > int(max_age):
        return "quote_unavailable"
    return None


# ---------------------------------------------------------------------------
# Activation enrichment
# ---------------------------------------------------------------------------

def _enrich_i8_signal(
    feat_dict: Dict[str, Any],
    inp: PatternInput,
    warnings: List[str],
    quality_flags: Dict[str, Any],
    lambda_i8_3td: float,
) -> Optional[PatternSignal]:
    """Compute opening-range breakout features and return signal if gates pass."""
    md = inp.market_data
    opening_range_high = float(md["opening_range_high"])
    opening_range_low = float(md["opening_range_low"])
    sigma_20d = float(md["sigma_20d"])
    opening_range_size = opening_range_high - opening_range_low

    feat_dict["opening_range_high"] = opening_range_high
    feat_dict["opening_range_low"] = opening_range_low
    feat_dict["opening_range_size"] = round(opening_range_size, 6)
    feat_dict["sigma_20d"] = sigma_20d
    _copy_diagnostic_fields(feat_dict, md)

    # Pre-signal rejection
    pre_rej = _pre_signal_rejection(feat_dict, md)
    if pre_rej is not None:
        feat_dict["rejection_reason"] = pre_rej
        feat_dict["signal_generated"] = False
        feat_dict["x_i8"] = 0.0
        return None

    # Breakout price
    breakout_price = md.get("breakout_price")
    if breakout_price is None:
        feat_dict["rejection_reason"] = "no_upside_breakout"
        feat_dict["signal_generated"] = False
        feat_dict["x_i8"] = 0.0
        return None

    breakout_price = float(breakout_price)
    feat_dict["breakout_price"] = breakout_price

    # Breakout strength
    brk_str = compute_breakout_strength(breakout_price, opening_range_high, sigma_20d)
    feat_dict["breakout_strength"] = round(brk_str, 6)

    if breakout_price <= opening_range_high:
        feat_dict["rejection_reason"] = "no_upside_breakout"
        feat_dict["signal_generated"] = False
        feat_dict["x_i8"] = 0.0
        return None

    if brk_str < MIN_BREAKOUT_STRENGTH:
        feat_dict["rejection_reason"] = "breakout_below_threshold"
        feat_dict["signal_generated"] = False
        feat_dict["x_i8"] = 0.0
        return None

    # Volume quality
    volume_30min = md.get("volume_30min")
    avg_volume_30min_20d = md.get("avg_volume_30min_20d")
    if volume_30min is None or float(volume_30min) <= 0:
        feat_dict["volume_30min"] = float(volume_30min) if volume_30min is not None else None
        feat_dict["volume_ratio"] = 0.0
        feat_dict["volume_quality"] = 0.0
        feat_dict["rejection_reason"] = "volume_below_minimum"
        feat_dict["signal_generated"] = False
        feat_dict["x_i8"] = 0.0
        return None

    volume_30min = float(volume_30min)
    baseline_volume_proxy = False
    if avg_volume_30min_20d is None or float(avg_volume_30min_20d) <= 0:
        avg_volume_30min_20d = volume_30min * 0.5
        baseline_volume_proxy = True
        quality_flags["baseline_volume_proxy"] = True
        warnings.append("avg_volume_30min_20d unavailable — using conservative proxy")
    else:
        avg_volume_30min_20d = float(avg_volume_30min_20d)

    vol_ratio = volume_30min / avg_volume_30min_20d if avg_volume_30min_20d > 0 else 0.0
    vol_qual = compute_volume_quality(vol_ratio)
    feat_dict["volume_30min"] = volume_30min
    feat_dict["avg_volume_30min_20d"] = round(avg_volume_30min_20d, 2)
    feat_dict["volume_ratio"] = round(vol_ratio, 6)
    feat_dict["volume_quality"] = vol_qual
    feat_dict["baseline_volume_proxy"] = baseline_volume_proxy

    # Range quality
    avg_range_30min_20d = md.get("avg_range_30min_20d")
    if avg_range_30min_20d is None or float(avg_range_30min_20d) <= 0:
        range_ratio = 1.0
        quality_flags["baseline_range_proxy"] = True
        warnings.append("avg_range_30min_20d unavailable — assuming normal range")
    else:
        range_ratio = opening_range_size / float(avg_range_30min_20d) if float(avg_range_30min_20d) > 0 else 1.0

    rng_qual = compute_range_quality(range_ratio)
    feat_dict["avg_range_30min_20d"] = float(avg_range_30min_20d) if avg_range_30min_20d is not None else None
    feat_dict["range_ratio"] = round(range_ratio, 6)
    feat_dict["range_quality"] = rng_qual

    # Spread quality
    spread_at_eval = md.get("spread_at_eval_bps")
    normal_spread = md.get("normal_spread_20d_bps")
    if spread_at_eval is None:
        sprd_qual = 0.75
        quality_flags["baseline_spread_proxy"] = True
        warnings.append("spread_at_eval_bps unavailable — using conservative wide-spread proxy")
    else:
        if normal_spread is None or float(normal_spread) <= 0:
            quality_flags["baseline_spread_proxy"] = True
            warnings.append("normal_spread_20d_bps unavailable — using conservative wide-spread proxy")
        sprd_qual = compute_spread_quality(
            float(spread_at_eval),
            float(normal_spread) if normal_spread is not None else 0.0,
        )

    feat_dict["spread_at_eval_bps"] = float(spread_at_eval) if spread_at_eval is not None else None
    feat_dict["normal_spread_20d_bps"] = float(normal_spread) if normal_spread is not None else None
    feat_dict["spread_quality"] = sprd_qual

    if sprd_qual == 0.0:
        feat_dict["rejection_reason"] = "spread_too_wide"
        feat_dict["signal_generated"] = False
        feat_dict["x_i8"] = 0.0
        return None

    # Exposure
    x_i8 = min(brk_str * vol_qual * rng_qual * sprd_qual, X_I8_CAP)
    feat_dict["x_i8"] = round(x_i8, 6)
    feat_dict["signal_generated"] = True

    # Expected-return priors
    raw_expected_edge = round(x_i8 * lambda_i8_3td, 6)
    signal_strength = round(min(x_i8 / X_I8_STRENGTH_DIVISOR, 1.0), 6)
    feat_dict["expected_return_priors"] = {"gross_bps": round(raw_expected_edge * 10_000, 2)}

    return PatternSignal(
        direction=SignalDirection.LONG,
        raw_signal_strength=signal_strength,
        raw_expected_edge=raw_expected_edge,
        signal_horizon=SIGNAL_HORIZON,
        data_confidence=_data_confidence(quality_flags),
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

class I8Detector(BasePatternDetector):
    """I8 Opening Range Breakout detector."""

    pattern_id = PatternId.I8
    track = PatternTrack.INTRADAY
    thesis_category = ThesisCategory.RIGHT_TAIL_CONVEX
    route_class = RouteClass.C

    def __init__(self, lambda_i8_3td: float = LAMBDA_I8_3TD_DEFAULT):
        self._lambda_i8_3td = lambda_i8_3td

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        asof = require_asof_timestamp(inp.asof_timestamp)
        warnings: List[str] = []
        quality_flags: Dict[str, Any] = {}

        reject_future_timestamp(asof, warnings, quality_flags)
        require_lineage_hash(inp.lineage_hashes, warnings, quality_flags)

        md = inp.market_data
        opening_range_high = md.get("opening_range_high")
        opening_range_low = md.get("opening_range_low")
        sigma_20d = md.get("sigma_20d")

        if opening_range_high is None or opening_range_low is None or sigma_20d is None:
            warnings.append("missing required fields (opening_range_high, opening_range_low, or sigma_20d)")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        if float(opening_range_high) <= 0 or float(opening_range_low) <= 0 or float(sigma_20d) <= 0:
            warnings.append("invalid opening range or sigma_20d values")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        if float(opening_range_high) <= float(opening_range_low):
            warnings.append("opening_range_high <= opening_range_low — invalid or zero-range bar")
            feat_dict = {
                "opening_range_high": float(opening_range_high),
                "opening_range_low": float(opening_range_low),
                "sigma_20d": float(sigma_20d),
                "signal_generated": False,
                "rejection_reason": "insufficient_bar_data",
            }
            pit_passed = quality_flags.get("point_in_time_passed") is not False
            fidelity = classify_fidelity(
                has_primary_data=True, has_secondary_data=True,
                point_in_time_passed=pit_passed, lookahead_guard_passed=True,
            )
            features = PatternFeatures(
                features=feat_dict, feature_manifest_version="i8-v1",
                fidelity_tier=fidelity, point_in_time_passed=pit_passed, lookahead_guard_passed=True,
            )
            input_hash, output_hash = _compute_hashes(inp, asof, feat_dict, [], warnings, quality_flags)
            return PatternDetectionResult(
                pattern_id=self.pattern_id, ticker=inp.ticker, asof_timestamp=asof,
                features=features, signals=[], warnings=warnings, quality_flags=quality_flags,
                input_hashes={"market_data": input_hash}, output_hashes={"features": output_hash},
            )

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
            feat_dict["signal_generated"] = False
            feat_dict["rejection_reason"] = universe_rejection
        else:
            sig = _enrich_i8_signal(feat_dict, inp, warnings, quality_flags, self._lambda_i8_3td)
            if sig is not None:
                signals.append(sig)

        features = PatternFeatures(
            features=feat_dict, feature_manifest_version="i8-v1",
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
