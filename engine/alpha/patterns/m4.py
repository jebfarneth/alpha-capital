"""
M4 — 52-Week High Breakout Detector.

Vault source: Engineering/Patterns/M4-52WeekHigh/

Thesis: right_tail_convex. Stocks closing at or above their prior
252-session high exhibit positive expected excess returns over the
following 15 trading days (J&T 1993 momentum premium).

Exposure formula (EXPOSURE.md):
  base_nearness = min(P / H52w, 1.0)
  breakout_extension = max(P / H52w - 1.0, 0)
  X_M4 = min(base_nearness + kappa * breakout_extension, 1.5)

Expected-return bridge (SPEC.md / EXPOSURE.md):
  lambda_M4_monthly = 1.1% (J&T 6/6 conservative)
  lambda_M4_15td = lambda_monthly * 15/21 = ~0.786%
  raw_expected_edge = X_M4 * lambda_M4_15td

Eligibility gate (EXPOSURE.md):
  1. P >= H52w (breakout event)
  2. Top-3-decile breakout_extension within same-day breakout cohort
  3. breakout_extension > 0 (exact-high crossings excluded)

This detector supports both vault-defined lanes:
  - base_daily close-confirmed breakouts
  - fresh_breakout_activation watchlist / activation rows
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

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
KAPPA = 1.0
X_M4_CAP = 1.5
LAMBDA_M4_MONTHLY = 0.011  # 1.1% per month (J&T 6/6 conservative)
LAMBDA_M4_15TD = LAMBDA_M4_MONTHLY * 15.0 / 21.0  # ~0.00786
SIGNAL_HORIZON = "15d"
BREAKOUT_COHORT_PERCENTILE = 0.70  # top-3-decile gate
SMALL_COHORT_THRESHOLD = 10
ENTRY_LANE_BASE = "base_daily"
ENTRY_LANE_FRESH = "fresh_breakout_activation"
ACTIVATION_STATE_WATCHLIST = "watchlist"
ACTIVATION_STATE_ACTIVATED = "activated"
FRESH_SPREAD_CAP = 0.01


def compute_m4_features(
    price: float,
    high_52w: float,
) -> Dict[str, Any]:
    """Compute M4 exposure features per EXPOSURE.md formula."""
    if high_52w <= 0:
        return {
            "base_nearness": 0.0,
            "breakout_extension": 0.0,
            "X_M4": 0.0,
            "ratio_P_H": 0.0,
            "kappa": KAPPA,
            "H_52w": high_52w,
            "P_close": price,
        }

    ratio = price / high_52w
    base_nearness = min(ratio, 1.0)
    breakout_extension = max(ratio - 1.0, 0.0)
    x_m4 = min(base_nearness + KAPPA * breakout_extension, X_M4_CAP)

    return {
        "base_nearness": round(base_nearness, 6),
        "breakout_extension": round(breakout_extension, 6),
        "X_M4": round(x_m4, 6),
        "ratio_P_H": round(ratio, 6),
        "kappa": KAPPA,
        "H_52w": high_52w,
        "P_close": price,
    }


def apply_cohort_gate(
    breakout_extension: float,
    cohort_extensions: List[float],
    *,
    ticker: str | None = None,
) -> Dict[str, Any]:
    """
    Apply the top-3-decile breakout_extension cohort gate.

    Returns gate metadata: passed, cohort_size, rank, decile,
    percentile_threshold, small_cohort_warning.
    """
    cohort_size = len(cohort_extensions)

    # Small-cohort handling: all pass per EXPOSURE.md
    if cohort_size < SMALL_COHORT_THRESHOLD:
        passed = breakout_extension > 0
        return {
            "cohort_gate_passed": passed,
            "breakout_cohort_size": cohort_size,
            "breakout_cohort_rank": None,
            "breakout_cohort_decile": None,
            "percentile_threshold": 0.0,
            "small_cohort_warning": True,
        }

    # Compute 70th percentile threshold (linear interpolation)
    sorted_ext = sorted(_cohort_extension_value(item) for item in cohort_extensions)
    idx = BREAKOUT_COHORT_PERCENTILE * (cohort_size - 1)
    lower = int(idx)
    frac = idx - lower
    if lower + 1 < cohort_size:
        threshold = sorted_ext[lower] + frac * (sorted_ext[lower + 1] - sorted_ext[lower])
    else:
        threshold = sorted_ext[lower]

    # Rank (1 = highest extension), with optional vault tie-break fields:
    # higher 20d median dollar volume, earlier signal_timestamp, ticker.
    ranked = sorted(
        (_cohort_sort_record(item) for item in cohort_extensions),
        key=lambda r: (
            -r["extension"],
            -r["median_dollar_volume_20d"],
            r["signal_timestamp"],
            r["ticker"],
        ),
    )
    rank = _rank_for_candidate(ranked, breakout_extension, ticker)
    decile = min(10, max(1, int((rank - 1) / cohort_size * 10) + 1))
    top_count = max(1, math.ceil((1.0 - BREAKOUT_COHORT_PERCENTILE) * cohort_size))
    passed = (
        breakout_extension > threshold
        or (breakout_extension == threshold and rank <= top_count)
    ) and breakout_extension > 0

    return {
        "cohort_gate_passed": passed,
        "breakout_cohort_size": cohort_size,
        "breakout_cohort_rank": rank,
        "breakout_cohort_decile": decile,
        "percentile_threshold": round(threshold, 6),
        "small_cohort_warning": False,
    }


def _cohort_extension_value(item: Any) -> float:
    if isinstance(item, dict):
        return float(item.get("breakout_extension", item.get("extension", 0.0)))
    return float(item)


def _cohort_sort_record(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return {
            "ticker": str(item.get("ticker", "")),
            "extension": _cohort_extension_value(item),
            "median_dollar_volume_20d": float(item.get("median_dollar_volume_20d", 0.0)),
            "signal_timestamp": str(item.get("signal_timestamp", "")),
        }
    return {
        "ticker": "",
        "extension": float(item),
        "median_dollar_volume_20d": 0.0,
        "signal_timestamp": "",
    }


def _rank_for_candidate(
    ranked: List[Dict[str, Any]], breakout_extension: float, ticker: str | None
) -> int:
    if ticker:
        for idx, record in enumerate(ranked, start=1):
            if record["ticker"] == ticker:
                return idx
    return sum(1 for record in ranked if record["extension"] > breakout_extension) + 1


def compute_m4_fresh_features(
    *,
    last_price: float,
    high_52w: float,
    intraday_range_confirmation: float,
    intraday_volume_confirmation: float,
) -> Dict[str, Any]:
    if high_52w <= 0:
        fresh_extension = 0.0
    else:
        fresh_extension = min(max((last_price / high_52w) - 1.0, 0.0), 0.50)
    x_fresh = min(
        1.0 + fresh_extension * intraday_range_confirmation * intraday_volume_confirmation,
        X_M4_CAP,
    )
    return {
        "fresh_breakout_extension": round(fresh_extension, 6),
        "intraday_range_confirmation": intraday_range_confirmation,
        "intraday_volume_confirmation": intraday_volume_confirmation,
        "x_m4_fresh": round(x_fresh, 6),
    }


class M4Detector(BasePatternDetector):
    """M4 52-Week High Breakout detector (base_daily lane)."""

    pattern_id = PatternId.M4
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.RIGHT_TAIL_CONVEX
    route_class = RouteClass.A

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        asof = require_asof_timestamp(inp.asof_timestamp)

        warnings: List[str] = []
        quality_flags: Dict[str, Any] = {}

        reject_future_timestamp(asof, warnings, quality_flags)
        require_lineage_hash(inp.lineage_hashes, warnings, quality_flags)

        # Required inputs
        price = inp.market_data.get("price")
        high_52w = inp.market_data.get("high_52w")
        cohort_extensions = inp.market_data.get("cohort_extensions")
        entry_lane = inp.market_data.get("entry_lane", ENTRY_LANE_BASE)

        if price is None or high_52w is None:
            warnings.append("missing required price or high_52w")
            return PatternDetectionResult(
                pattern_id=self.pattern_id,
                ticker=inp.ticker,
                asof_timestamp=asof,
                features=None,
                warnings=warnings,
                quality_flags=quality_flags,
            )

        price = float(price)
        high_52w = float(high_52w)

        if high_52w <= 0 or price <= 0:
            warnings.append(f"invalid price={price} or high_52w={high_52w}")
            return PatternDetectionResult(
                pattern_id=self.pattern_id,
                ticker=inp.ticker,
                asof_timestamp=asof,
                features=None,
                warnings=warnings,
                quality_flags=quality_flags,
            )

        # Compute features
        feat_dict = compute_m4_features(price, high_52w)

        # Optional enrichment fields (diagnostic, not load-bearing)
        market_cap = inp.fundamental_data.get("market_cap")
        if market_cap is not None:
            feat_dict["market_cap_mm"] = round(float(market_cap) / 1e6, 1)
            sub = "A" if float(market_cap) < 80_000_000 else "B"
            feat_dict["sub_universe"] = sub
        if "sector" in inp.fundamental_data:
            feat_dict["sector"] = inp.fundamental_data["sector"]
        if "industry" in inp.fundamental_data:
            feat_dict["industry"] = inp.fundamental_data["industry"]
        for source in (inp.market_data, inp.fundamental_data, inp.event_data):
            for key in (
                "D1_decile",
                "R_6_12m_skip",
                "analyst_count",
                "hamilton_regime_prob",
                "hazard_score_at_signal",
                "filing_veto_status",
            ):
                if key in source:
                    feat_dict[key] = source[key]
        feat_dict.setdefault("filing_veto_status", "clear")
        feat_dict["entry_lane"] = entry_lane

        # Determine fidelity
        n_sessions = inp.market_data.get("n_sessions_in_window")
        short_history = n_sessions is not None and int(n_sessions) < 252
        feat_dict["short_history_flag"] = short_history

        has_primary = price > 0 and high_52w > 0
        pit_passed = quality_flags.get("point_in_time_passed") is not False
        if inp.market_data.get("operating_universe_inclusion") is False:
            quality_flags["not_operating_universe_member"] = True
            warnings.append("ticker is not marked as operating-universe eligible")
        elif inp.universe_snapshot_id is None and "operating_universe_inclusion" not in inp.market_data:
            warnings.append("operating-universe membership not provided to M4 detector")
        fidelity = classify_fidelity(
            has_primary_data=has_primary,
            has_secondary_data=True,
            point_in_time_passed=pit_passed,
            lookahead_guard_passed=True,
        )

        features = PatternFeatures(
            features=feat_dict,
            feature_manifest_version="m4-v1",
            fidelity_tier=fidelity,
            point_in_time_passed=pit_passed,
            lookahead_guard_passed=True,
        )

        # Signal eligibility: breakout event
        breakout = price >= high_52w
        extension = feat_dict["breakout_extension"]

        signals: List[PatternSignal] = []

        if entry_lane == ENTRY_LANE_FRESH:
            self._apply_fresh_lane(inp, feat_dict, warnings, quality_flags, signals)
        elif quality_flags.get("not_operating_universe_member"):
            feat_dict["cohort_gate_passed"] = False
        elif breakout and extension > 0:
            # Cohort gate
            if cohort_extensions is None:
                # M4's top-3-decile breakout-extension cohort gate is part
                # of signal generation. Without the same-day cohort, preserve
                # features for audit but do not emit a signal.
                cohort_meta = {
                    "cohort_gate_passed": False,
                    "breakout_cohort_size": None,
                    "breakout_cohort_rank": None,
                    "breakout_cohort_decile": None,
                    "percentile_threshold": None,
                    "small_cohort_warning": False,
                    "cohort_missing": True,
                }
                warnings.append("missing breakout cohort data — no M4 signal emitted")
            else:
                cohort_meta = apply_cohort_gate(
                    extension, cohort_extensions, ticker=inp.ticker
                )

            feat_dict.update(cohort_meta)

            if cohort_meta["cohort_gate_passed"]:
                x_m4 = feat_dict["X_M4"]
                raw_expected_edge = round(x_m4 * LAMBDA_M4_15TD, 6)
                signal_strength = round(x_m4 / X_M4_CAP, 6)

                # Tier classification (audit metadata, per SPEC.md)
                cohort_exts = cohort_extensions or [extension]
                p75 = sorted(cohort_exts)[int(0.75 * (len(cohort_exts) - 1))] if len(cohort_exts) > 1 else extension
                tier = "high_conviction" if extension >= p75 else "default"
                feat_dict["tier_classification"] = tier

                data_conf = _data_confidence(inp)

                gross_bps = round(raw_expected_edge * 10_000, 2)
                feat_dict["expected_return_priors"] = {
                    "tier": tier,
                    "gross_bps": gross_bps,
                }

                signals.append(
                    PatternSignal(
                        direction=SignalDirection.LONG,
                        raw_signal_strength=signal_strength,
                        raw_expected_edge=raw_expected_edge,
                        signal_horizon=SIGNAL_HORIZON,
                        data_confidence=round(data_conf, 4),
                    )
                )

        input_hash = stable_hash({
            "ticker": inp.ticker,
            "asof_timestamp": asof,
            "market_data": inp.market_data,
            "fundamental_data": inp.fundamental_data,
            "event_data": inp.event_data,
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

    def _apply_fresh_lane(
        self,
        inp: PatternInput,
        feat_dict: Dict[str, Any],
        warnings: List[str],
        quality_flags: Dict[str, Any],
        signals: List[PatternSignal],
    ) -> None:
        activation_state = inp.market_data.get("activation_state", ACTIVATION_STATE_WATCHLIST)
        feat_dict["activation_state"] = activation_state

        base_nearness = feat_dict["base_nearness"]
        if activation_state == ACTIVATION_STATE_WATCHLIST:
            watchlist_passed = base_nearness >= 0.97 and not inp.market_data.get(
                "already_base_daily_fired", False
            )
            feat_dict["watchlist_passed"] = watchlist_passed
            if watchlist_passed:
                signals.append(
                    PatternSignal(
                        direction=SignalDirection.LONG,
                        raw_signal_strength=round(base_nearness, 6),
                        raw_expected_edge=0.0,
                        signal_horizon=SIGNAL_HORIZON,
                        signal_status="watchlist",
                        data_confidence=_data_confidence(inp),
                    )
                )
            return

        last_price = float(inp.market_data.get("last_price", inp.market_data.get("price", 0.0)))
        range_confirmation = float(inp.market_data.get("intraday_range_confirmation", 0.0))
        volume_confirmation = float(inp.market_data.get("intraday_volume_confirmation", 0.0))
        spread_pct = inp.market_data.get("spread_pct_vs_eval_quote")
        spread_passed = (
            bool(inp.market_data.get("spread_discipline_passed"))
            if "spread_discipline_passed" in inp.market_data
            else spread_pct is not None and float(spread_pct) <= FRESH_SPREAD_CAP
        )
        freshness_passed = bool(inp.market_data.get("signal_freshness_passed", True))

        fresh_features = compute_m4_fresh_features(
            last_price=last_price,
            high_52w=feat_dict["H_52w"],
            intraday_range_confirmation=range_confirmation,
            intraday_volume_confirmation=volume_confirmation,
        )
        feat_dict.update(fresh_features)
        feat_dict["last_price"] = last_price
        feat_dict["fresh_high_break_passed"] = last_price > feat_dict["H_52w"]
        feat_dict["range_confirmation_passed"] = range_confirmation >= 1.0
        feat_dict["volume_confirmation_passed"] = volume_confirmation >= 1.0
        feat_dict["spread_discipline_passed"] = spread_passed
        feat_dict["signal_freshness_passed"] = freshness_passed

        activation_passed = (
            feat_dict["fresh_high_break_passed"]
            and fresh_features["fresh_breakout_extension"] > 0
            and feat_dict["range_confirmation_passed"]
            and feat_dict["volume_confirmation_passed"]
            and spread_passed
            and freshness_passed
        )
        feat_dict["activation_passed"] = activation_passed
        if not activation_passed:
            feat_dict["activation_failure_reason"] = _fresh_activation_failure_reason(feat_dict)
            warnings.append(f"fresh activation failed: {feat_dict['activation_failure_reason']}")
            return

        x_fresh = fresh_features["x_m4_fresh"]
        raw_expected_edge = round(x_fresh * LAMBDA_M4_15TD, 6)
        feat_dict["expected_return_priors"] = {
            "tier": "default",
            "entry_lane": ENTRY_LANE_FRESH,
            "gross_bps": round(raw_expected_edge * 10_000, 2),
        }
        signals.append(
            PatternSignal(
                direction=SignalDirection.LONG,
                raw_signal_strength=round(min(x_fresh / X_M4_CAP, 1.0), 6),
                raw_expected_edge=raw_expected_edge,
                signal_horizon=SIGNAL_HORIZON,
                data_confidence=_data_confidence(inp),
            )
        )


def _data_confidence(inp: PatternInput) -> float:
    confidence = 1.0
    for source in (inp.market_data, inp.fundamental_data, inp.event_data):
        field_confidence = source.get("field_confidence")
        if isinstance(field_confidence, dict):
            for value in field_confidence.values():
                confidence *= float(value)
    return round(confidence, 4)


def _fresh_activation_failure_reason(feat: Dict[str, Any]) -> str:
    if not feat["fresh_high_break_passed"]:
        return "fresh_high_break_failed"
    if not feat["range_confirmation_passed"]:
        return "range_confirmation_failed"
    if not feat["volume_confirmation_passed"]:
        return "volume_confirmation_failed"
    if not feat["spread_discipline_passed"]:
        return "spread_too_wide"
    if not feat["signal_freshness_passed"]:
        return "signal_expired"
    return "unknown"
