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

Signal eligibility:
  1. compression_depth >= 0.5 (compression_ratio <= 0.80)
  2. price > compression_high (breakout above compression range lid)
  3. intraday_breakout_extension > 0

This detector accepts pre-computed compression features from the nightly
scan pipeline. The GK estimator computation itself is a data-processing
concern, not detector logic.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from alpha.data.contracts import stable_hash
from alpha.patterns.contracts import (
    BasePatternDetector,
    FidelityTier,
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
    reject_future_timestamp,
    require_asof_timestamp,
    require_lineage_hash,
)

# Vault constants (EXPOSURE.md / SPEC.md)
LAMBDA_M6_MONTHLY = 0.011  # inherited from M4 (J&T 6/6)
AMPLIFICATION = 1.45  # Cakici 2023 small-cap factor
HOLD_DAYS = 12
LAMBDA_M6_12TD = LAMBDA_M6_MONTHLY * AMPLIFICATION * (HOLD_DAYS / 21.0)  # ~0.00912
X_M6_CAP = 3.0
SIGNAL_HORIZON = "12d"
MIN_COMPRESSION_DEPTH = 0.5  # compression_ratio <= 0.80


def compute_compression_depth(compression_ratio: float) -> float:
    """Per EXPOSURE.md: clip((1.0 - ratio) / 0.4, 0.0, 2.5)."""
    return max(0.0, min((1.0 - compression_ratio) / 0.4, 2.5))


def compute_breakout_extension(
    price: float,
    compression_high: float,
    sigma_20d: float,
) -> float:
    """
    Per EXPOSURE.md: (price - compression_high) / compression_high / sigma_20d,
    clipped to [0.0, 3.0]. Returns 0.0 if price <= compression_high.
    """
    if price <= compression_high or compression_high <= 0 or sigma_20d <= 0:
        return 0.0
    raw = (price - compression_high) / compression_high / sigma_20d
    return max(0.0, min(raw, 3.0))


def compute_expansion_confirmation(expansion_ratio: float) -> float:
    """Per EXPOSURE.md tiered weighting of GK expansion ratio."""
    if expansion_ratio >= 2.0:
        return 1.5
    if expansion_ratio >= 1.5:
        return 1.25
    if expansion_ratio >= 1.0:
        return 1.0
    return 0.5


def compute_volume_confirmation(volume_ratio: float) -> float:
    """Per EXPOSURE.md tiered weighting of intraday volume ratio."""
    if volume_ratio >= 2.0:
        return 1.5
    if volume_ratio >= 1.5:
        return 1.25
    if volume_ratio >= 1.0:
        return 1.0
    return 0.5


def compute_expansion_ratio(
    session_high: float,
    session_low: float,
    gk_avg_5d: float,
) -> Optional[float]:
    """
    Per EXPOSURE.md: intraday_range_proxy / sqrt(gk_avg_5d).
    Returns None if inputs are invalid.
    """
    if session_high <= 0 or session_low <= 0 or session_high <= session_low:
        return None
    if gk_avg_5d is None or gk_avg_5d <= 0:
        return None
    range_proxy = math.log(session_high / session_low)
    return range_proxy / math.sqrt(gk_avg_5d)


class M6Detector(BasePatternDetector):
    """M6 Volatility-Compression Breakout detector."""

    pattern_id = PatternId.M6
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.RIGHT_TAIL_CONVEX
    route_class = RouteClass.A

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        asof = require_asof_timestamp(inp.asof_timestamp)

        warnings: List[str] = []
        quality_flags: Dict[str, Any] = {}

        reject_future_timestamp(asof, warnings, quality_flags)
        require_lineage_hash(inp.lineage_hashes, warnings, quality_flags)

        # Required compression inputs (from nightly scan pipeline)
        compression_ratio = inp.market_data.get("compression_ratio")
        gk_vol_5d = inp.market_data.get("gk_vol_5d")
        gk_vol_60d = inp.market_data.get("gk_vol_60d")
        compression_high = inp.market_data.get("compression_high")
        sigma_20d = inp.market_data.get("sigma_20d")

        if compression_ratio is None or compression_high is None or sigma_20d is None:
            warnings.append("missing required compression features (compression_ratio, compression_high, or sigma_20d)")
            return PatternDetectionResult(
                pattern_id=self.pattern_id,
                ticker=inp.ticker,
                asof_timestamp=asof,
                features=None,
                warnings=warnings,
                quality_flags=quality_flags,
            )

        compression_ratio = float(compression_ratio)
        compression_high = float(compression_high)
        sigma_20d = float(sigma_20d)

        if compression_high <= 0 or sigma_20d <= 0:
            warnings.append(f"invalid compression_high={compression_high} or sigma_20d={sigma_20d}")
            return PatternDetectionResult(
                pattern_id=self.pattern_id,
                ticker=inp.ticker,
                asof_timestamp=asof,
                features=None,
                warnings=warnings,
                quality_flags=quality_flags,
            )

        # Compute compression depth
        depth = compute_compression_depth(compression_ratio)

        # Build features dict
        feat_dict: Dict[str, Any] = {
            "compression_ratio": round(compression_ratio, 6),
            "compression_depth": round(depth, 6),
            "gk_vol_5d": float(gk_vol_5d) if gk_vol_5d is not None else None,
            "gk_vol_60d": float(gk_vol_60d) if gk_vol_60d is not None else None,
            "compression_high": compression_high,
            "sigma_20d": sigma_20d,
            "X_M6_setup": round(depth, 6),
        }

        # GK quality flags
        gk_warning = inp.market_data.get("gk_low_transaction_warning", False)
        feat_dict["gk_low_transaction_warning"] = gk_warning

        # Optional enrichment (diagnostic, per DATA.md)
        market_cap = inp.fundamental_data.get("market_cap")
        if market_cap is not None:
            feat_dict["market_cap_mm"] = round(float(market_cap) / 1e6, 1)
        if "sector" in inp.fundamental_data:
            feat_dict["sector"] = inp.fundamental_data["sector"]

        # Fidelity
        has_ohlcv_history = gk_vol_60d is not None and float(gk_vol_60d) > 0
        pit_passed = quality_flags.get("point_in_time_passed") is not False
        fidelity = classify_fidelity(
            has_primary_data=has_ohlcv_history,
            has_secondary_data=not gk_warning,
            point_in_time_passed=pit_passed,
            lookahead_guard_passed=True,
        )

        # Compression gate
        compressed = depth >= MIN_COMPRESSION_DEPTH
        feat_dict["compression_gate_passed"] = compressed

        # Activation inputs (breakout data, may be absent for setup-only scans)
        price = inp.market_data.get("price")
        session_high = inp.market_data.get("session_high")
        session_low = inp.market_data.get("session_low")
        cumulative_volume = inp.market_data.get("cumulative_volume")
        expected_tod_volume = inp.market_data.get("expected_tod_volume")
        gk_avg_5d = inp.market_data.get("gk_avg_5d")

        signals: List[PatternSignal] = []

        if compressed and price is not None:
            price = float(price)
            feat_dict["P_activation"] = price

            # Breakout extension
            brk_ext = compute_breakout_extension(price, compression_high, sigma_20d)
            feat_dict["intraday_breakout_extension"] = round(brk_ext, 6)

            # Expansion confirmation
            exp_ratio = None
            exp_conf = 1.0  # default if session range data unavailable
            if session_high is not None and session_low is not None and gk_avg_5d is not None:
                exp_ratio = compute_expansion_ratio(
                    float(session_high), float(session_low), float(gk_avg_5d)
                )
                if exp_ratio is not None:
                    exp_conf = compute_expansion_confirmation(exp_ratio)
            else:
                # Missing expansion data degrades but does not block
                exp_conf = 1.0
                quality_flags["missing_expansion_data"] = True
                warnings.append("missing session_high/session_low/gk_avg_5d for expansion confirmation")

            feat_dict["intraday_expansion_ratio"] = round(exp_ratio, 6) if exp_ratio is not None else None
            feat_dict["intraday_expansion_confirmation"] = exp_conf

            # Volume confirmation
            vol_conf = 1.0  # default if volume data unavailable
            vol_ratio = None
            if cumulative_volume is not None and expected_tod_volume is not None:
                cv = float(cumulative_volume)
                ev = float(expected_tod_volume)
                if ev > 0:
                    vol_ratio = cv / ev
                    vol_conf = compute_volume_confirmation(vol_ratio)
            else:
                quality_flags["missing_volume_data"] = True
                warnings.append("missing cumulative_volume/expected_tod_volume for volume confirmation")

            feat_dict["intraday_volume_ratio"] = round(vol_ratio, 6) if vol_ratio is not None else None
            feat_dict["intraday_volume_confirmation"] = vol_conf

            # Activation exposure
            x_m6_activation = min(depth * brk_ext * exp_conf * vol_conf, X_M6_CAP)
            feat_dict["X_M6_activation"] = round(x_m6_activation, 6)

            # Signal fires if breakout above compression high
            if brk_ext > 0:
                raw_expected_edge = round(x_m6_activation * LAMBDA_M6_12TD, 6)
                signal_strength = round(min(x_m6_activation / X_M6_CAP, 1.0), 6)

                feat_dict["activation_state"] = "activated"

                # Data confidence
                data_conf = 1.0
                if gk_warning:
                    data_conf *= 0.9
                if quality_flags.get("missing_lineage"):
                    data_conf *= 0.9
                if quality_flags.get("missing_expansion_data"):
                    data_conf *= 0.95
                if quality_flags.get("missing_volume_data"):
                    data_conf *= 0.95

                gross_bps = round(raw_expected_edge * 10_000, 2)
                feat_dict["expected_return_priors"] = {"gross_bps": gross_bps}

                signals.append(
                    PatternSignal(
                        direction=SignalDirection.LONG,
                        raw_signal_strength=signal_strength,
                        raw_expected_edge=raw_expected_edge,
                        signal_horizon=SIGNAL_HORIZON,
                        data_confidence=round(data_conf, 4),
                    )
                )
            else:
                feat_dict["activation_state"] = "no_breakout"
        elif compressed:
            feat_dict["activation_state"] = "watchlist"
        else:
            feat_dict["activation_state"] = "not_compressed"

        features = PatternFeatures(
            features=feat_dict,
            feature_manifest_version="m6-v1",
            fidelity_tier=fidelity,
            point_in_time_passed=pit_passed,
            lookahead_guard_passed=True,
        )

        input_hash = stable_hash({
            "ticker": inp.ticker,
            "asof_timestamp": asof,
            "market_data": inp.market_data,
            "fundamental_data": inp.fundamental_data,
            "lineage_hashes": inp.lineage_hashes,
            "universe_snapshot_id": inp.universe_snapshot_id,
        })
        output_hash = stable_hash({
            "features": feat_dict,
            "signals": [
                {
                    "direction": sig.direction,
                    "raw_signal_strength": sig.raw_signal_strength,
                    "raw_expected_edge": sig.raw_expected_edge,
                    "signal_horizon": sig.signal_horizon,
                    "signal_status": sig.signal_status,
                    "data_confidence": sig.data_confidence,
                }
                for sig in signals
            ],
            "warnings": warnings,
            "quality_flags": quality_flags,
        })

        return PatternDetectionResult(
            pattern_id=self.pattern_id,
            ticker=inp.ticker,
            asof_timestamp=asof,
            features=features,
            signals=signals,
            warnings=warnings,
            quality_flags=quality_flags,
            input_hashes={"market_data": input_hash},
            output_hashes={"features": output_hash},
        )
