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
  Fired signals persist lambda_M4_monthly and lambda_M4_15td so validation
  can reconstruct the expected-edge assumption.

Signal admission (EXPOSURE.md amended):
  1. P >= H52w (breakout event, including exact-high closes)
  2. Operating-universe membership
  Cohort rank and top-3-decile flag are metadata for KOTH/TCB/validation,
  not signal-generation gates.

This detector supports both vault-defined lanes:
  - base_daily close-confirmed breakouts
  - fresh_breakout_activation watchlist / activation rows

Route class:
  - base_daily uses Class A
  - fresh_breakout_activation uses Class C after activation
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from alpha.data.contracts import stable_hash
from alpha.patterns.activation import required_fields_present
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
)

# Vault constants (EXPOSURE.md / SPEC.md)
KAPPA = 1.0
X_M4_CAP = 1.5
LAMBDA_M4_MONTHLY = 0.011  # 1.1% per month (J&T 6/6 conservative)
LAMBDA_M4_15TD = LAMBDA_M4_MONTHLY * 15.0 / 21.0  # ~0.00786
SIGNAL_HORIZON = "15d"
BREAKOUT_COHORT_PERCENTILE = 0.70  # 70th pctl threshold for top3_decile_flag
SMALL_COHORT_THRESHOLD = 10
ENTRY_LANE_BASE = "base_daily"
ENTRY_LANE_FRESH = "fresh_breakout_activation"
ACTIVATION_STATE_WATCHLIST = "watchlist"
ACTIVATION_STATE_ACTIVATED = "activated"
FRESH_SPREAD_CAP = 0.01
FRESH_IDENTITY_FIELDS = ("activation_id", "activation_timestamp")
DIAGNOSTIC_SOURCE_KEYS = (
    "D1_decile",
    "R_6_12m_skip",
    "analyst_count",
    "hamilton_regime_prob",
    "hazard_score_at_signal",
    "filing_veto_status",
    "m3_also_firing",
    "m5_also_firing",
    "m6_also_firing",
    "overlapping_pattern_ids",
)
QUOTE_FIELDS = DEFAULT_QUOTE_FIELDS
FRESH_QUOTE_FIELDS = (*QUOTE_FIELDS, "quote_freshness_max_ms")
FRESH_ACTIVATION_FIELDS = ("activation_id", "activation_timestamp", *FRESH_QUOTE_FIELDS)


# ---------------------------------------------------------------------------
# Pure feature computation
# ---------------------------------------------------------------------------

def compute_m4_features(price: float, high_52w: float) -> Dict[str, Any]:
    """Compute M4 exposure features per EXPOSURE.md formula."""
    if high_52w <= 0:
        return {
            "base_nearness": 0.0, "breakout_extension": 0.0, "X_M4": 0.0,
            "ratio_P_H": 0.0, "kappa": KAPPA, "H_52w": high_52w, "P_close": price,
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


def build_m4_source_features(
    inp: PatternInput, *, price: float, high_52w: float, entry_lane: str,
) -> Dict[str, Any]:
    """Build the full source_features dict: exposure + diagnostics + enrichment."""
    features = compute_m4_features(price, high_52w)
    market_cap = inp.fundamental_data.get("market_cap")
    market_cap_f = finite_float(market_cap) if market_cap is not None else None
    if market_cap_f is not None:
        features["market_cap_mm"] = round(market_cap_f / 1e6, 1)
        features["sub_universe"] = "A" if market_cap_f < 80_000_000 else "B"
    for key in ("sector", "industry"):
        if key in inp.fundamental_data:
            features[key] = inp.fundamental_data[key]
    for source in (inp.market_data, inp.fundamental_data, inp.event_data):
        for key in DIAGNOSTIC_SOURCE_KEYS:
            if key in source:
                features[key] = source[key]
    features.setdefault("filing_veto_status", "not_computed")
    features["entry_lane"] = entry_lane
    n_sessions = inp.market_data.get("n_sessions_in_window")
    features["short_history_flag"] = n_sessions is not None and int(n_sessions) < 252
    return features


# ---------------------------------------------------------------------------
# Cohort metadata (ranking, not admission)
# ---------------------------------------------------------------------------

def compute_cohort_metadata(
    breakout_extension: float, cohort_extensions: List[Any], *, ticker: str | None = None,
) -> Dict[str, Any]:
    """
    Compute breakout-cohort ranking metadata per EXPOSURE.md.
    METADATA for KOTH/TCB/validation. Does NOT suppress signal generation.
    """
    cohort_size = len(cohort_extensions)
    if cohort_size < SMALL_COHORT_THRESHOLD:
        return {
            "breakout_cohort_size": cohort_size,
            "breakout_cohort_rank": None,
            "breakout_cohort_percentile": None,
            "breakout_cohort_decile": None,
            "cohort_threshold_70p": 0.0,
            "top3_decile_flag": True,
            "small_cohort_warning": True,
        }
    cohort_records = _cohort_records(cohort_extensions)
    sorted_ext = sorted(r["extension"] for r in cohort_records)
    idx = BREAKOUT_COHORT_PERCENTILE * (cohort_size - 1)
    lower = int(idx)
    frac = idx - lower
    threshold = (
        sorted_ext[lower] + frac * (sorted_ext[lower + 1] - sorted_ext[lower])
        if lower + 1 < cohort_size else sorted_ext[lower]
    )
    rank = _cohort_rank(cohort_records, breakout_extension, ticker)
    decile = min(10, max(1, int((rank - 1) / cohort_size * 10) + 1))
    percentile = round(1.0 - (rank - 1) / cohort_size, 4)
    top_count = max(1, math.ceil((1.0 - BREAKOUT_COHORT_PERCENTILE) * cohort_size))
    top3_flag = (
        breakout_extension > threshold
        or (breakout_extension == threshold and rank <= top_count)
    )
    return {
        "breakout_cohort_size": cohort_size,
        "breakout_cohort_rank": rank,
        "breakout_cohort_percentile": percentile,
        "breakout_cohort_decile": decile,
        "cohort_threshold_70p": round(threshold, 6),
        "top3_decile_flag": top3_flag,
        "small_cohort_warning": False,
    }


def _cohort_extension_value(item: Any) -> float:
    if isinstance(item, dict):
        return finite_float(item.get("breakout_extension", item.get("extension", 0.0))) or 0.0
    return finite_float(item) or 0.0


def _cohort_records(cohort_extensions: List[Any]) -> List[Dict[str, Any]]:
    records = []
    for item in cohort_extensions:
        if isinstance(item, dict):
            records.append({
                "ticker": str(item.get("ticker", "")),
                "extension": _cohort_extension_value(item),
                "median_dollar_volume_20d": finite_float(item.get("median_dollar_volume_20d", 0.0)) or 0.0,
                "signal_timestamp": str(item.get("signal_timestamp", "")),
            })
        else:
            records.append({
                "ticker": "", "extension": finite_float(item) or 0.0,
                "median_dollar_volume_20d": 0.0, "signal_timestamp": "",
            })
    return records


def _cohort_rank(
    cohort_records: List[Dict[str, Any]], breakout_extension: float, ticker: str | None,
) -> int:
    ranked = sorted(cohort_records, key=lambda r: (
        -r["extension"], -r["median_dollar_volume_20d"], r["signal_timestamp"], r["ticker"],
    ))
    if ticker:
        for idx, r in enumerate(ranked, start=1):
            if r["ticker"] == ticker:
                return idx
    return sum(1 for r in ranked if r["extension"] > breakout_extension) + 1


# ---------------------------------------------------------------------------
# Extension tier classification
# ---------------------------------------------------------------------------

def _classify_extension_tier(extension: float, cohort_extensions: List[Any] | None) -> str:
    """exact_high / high_conviction / default per SPEC.md."""
    if extension == 0:
        return "exact_high"
    if cohort_extensions:
        p75_values = sorted(_cohort_extension_value(item) for item in cohort_extensions)
        idx = 0.75 * (len(p75_values) - 1)
        lower = int(idx)
        frac = idx - lower
        p75 = (
            p75_values[lower] + frac * (p75_values[lower + 1] - p75_values[lower])
            if lower + 1 < len(p75_values) else p75_values[lower]
        )
        if extension >= p75:
            return "high_conviction"
    return "default"


# ---------------------------------------------------------------------------
# Base-daily breakout signal enrichment
# ---------------------------------------------------------------------------

def _enrich_base_daily_breakout(
    feat_dict: Dict[str, Any],
    inp: PatternInput,
    cohort_extensions: List[Any] | None,
    warnings: List[str],
    quality_flags: Dict[str, Any],
    lambda_15td: float = LAMBDA_M4_15TD,
) -> PatternSignal:
    """
    Enrich feat_dict with cohort metadata, tier, and priors for a base_daily
    breakout signal. Returns the PatternSignal to append.

    All base_daily signal fields are set here — the caller should not
    mutate feat_dict further for signal purposes.
    """
    extension = feat_dict["breakout_extension"]

    # Cohort metadata (ranking, not admission)
    if cohort_extensions is not None:
        feat_dict.update(compute_cohort_metadata(
            extension, cohort_extensions, ticker=inp.ticker,
        ))
    else:
        quality_flags["cohort_metadata_unavailable"] = True
        warnings.append("breakout cohort data unavailable — cohort metadata missing")

    # Extension tier
    tier = _classify_extension_tier(extension, cohort_extensions)
    feat_dict["extension_tier"] = tier

    # Expected-return priors
    x_m4 = feat_dict["X_M4"]
    raw_expected_edge = round(x_m4 * lambda_15td, 6)
    feat_dict["lambda_M4_monthly"] = LAMBDA_M4_MONTHLY
    feat_dict["validated_or_shadow_lambda_M4_15td"] = lambda_15td
    feat_dict["lambda_M4_15td"] = round(lambda_15td, 8)
    feat_dict["lambda_M4_default_15td"] = round(LAMBDA_M4_15TD, 8)
    feat_dict["lambda_M4_source"] = (
        "shadow_prior" if lambda_15td == LAMBDA_M4_15TD else "validated_or_injected"
    )
    feat_dict["expected_return_priors"] = {
        "tier": tier,
        "gross_bps": round(raw_expected_edge * 10_000, 2),
    }

    return PatternSignal(
        direction=SignalDirection.LONG,
        raw_signal_strength=round(x_m4 / X_M4_CAP, 6),
        raw_expected_edge=raw_expected_edge,
        signal_horizon=SIGNAL_HORIZON,
        route_class=RouteClass.A,
        data_confidence=_data_confidence(inp, quality_flags),
    )


# ---------------------------------------------------------------------------
# Fresh-breakout activation lane
# ---------------------------------------------------------------------------

def compute_m4_fresh_features(
    *, last_price: float, high_52w: float,
    intraday_range_confirmation: float, intraday_volume_confirmation: float,
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


def _fresh_activation_failure_reason(feat: Dict[str, Any]) -> str:
    if feat.get("activation_identity_passed") is not True:
        return "activation_identity_missing"
    if feat.get("quote_capture_passed") is not True:
        return "quote_unavailable"
    if not feat["fresh_high_break_passed"]:
        return "fresh_high_break_failed"
    if not feat["range_confirmation_passed"]:
        return "range_confirmation_failed"
    if not feat["volume_confirmation_passed"]:
        return "volume_confirmation_failed"
    if not feat["spread_discipline_passed"]:
        return "spread_unavailable" if feat.get("spread_pct_vs_eval_quote") is None else "spread_too_wide"
    if not feat["signal_freshness_passed"]:
        return "signal_expired"
    return "unknown"


def _fresh_activation_identity_passed(market_data: Dict[str, Any]) -> bool:
    """Executable fresh activations must be joinable to m4_intraday_activation."""
    return required_fields_present(market_data, FRESH_IDENTITY_FIELDS)


def _build_fresh_watchlist_signal(
    inp: PatternInput, base_nearness: float, quality_flags: Dict[str, Any],
) -> PatternSignal:
    return PatternSignal(
        direction=SignalDirection.LONG,
        raw_signal_strength=round(base_nearness, 6),
        raw_expected_edge=0.0,
        signal_horizon=SIGNAL_HORIZON,
        signal_status="watchlist",
        route_class=RouteClass.C,
        data_confidence=_data_confidence(inp, quality_flags),
    )


def _build_fresh_activation_signal(
    inp: PatternInput, feat_dict: Dict[str, Any], x_fresh: float,
    quality_flags: Dict[str, Any],
    lambda_15td: float = LAMBDA_M4_15TD,
) -> PatternSignal:
    raw_expected_edge = round(x_fresh * lambda_15td, 6)
    feat_dict["lambda_M4_monthly"] = LAMBDA_M4_MONTHLY
    feat_dict["validated_or_shadow_lambda_M4_15td"] = lambda_15td
    feat_dict["lambda_M4_15td"] = round(lambda_15td, 8)
    feat_dict["lambda_M4_default_15td"] = round(LAMBDA_M4_15TD, 8)
    feat_dict["lambda_M4_source"] = (
        "shadow_prior" if lambda_15td == LAMBDA_M4_15TD else "validated_or_injected"
    )
    feat_dict["expected_return_priors"] = {
        "tier": "default", "entry_lane": ENTRY_LANE_FRESH,
        "gross_bps": round(raw_expected_edge * 10_000, 2),
    }
    return PatternSignal(
        direction=SignalDirection.LONG,
        raw_signal_strength=round(min(x_fresh / X_M4_CAP, 1.0), 6),
        raw_expected_edge=raw_expected_edge,
        signal_horizon=SIGNAL_HORIZON,
        route_class=RouteClass.C,
        data_confidence=_data_confidence(inp, quality_flags),
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _data_confidence(inp: PatternInput, quality_flags: Dict[str, Any]) -> float:
    return compute_data_confidence(
        quality_flags,
        field_confidence_sources=(inp.market_data, inp.fundamental_data, inp.event_data),
    )


def _compute_hashes(
    inp: PatternInput, asof: Any, feat_dict: Dict[str, Any],
    signals: List[PatternSignal], warnings: List[str], quality_flags: Dict[str, Any],
) -> tuple:
    input_hash = stable_hash({
        "ticker": inp.ticker, "asof_timestamp": asof,
        "market_data": inp.market_data, "fundamental_data": inp.fundamental_data,
        "event_data": inp.event_data, "lineage_hashes": inp.lineage_hashes,
        "universe_snapshot_id": inp.universe_snapshot_id,
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

class M4Detector(BasePatternDetector):
    """M4 52-Week High Breakout detector."""

    pattern_id = PatternId.M4
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.RIGHT_TAIL_CONVEX
    route_class = RouteClass.A

    def __init__(self, lambda_m4_15td: float = LAMBDA_M4_15TD):
        parsed = finite_float(lambda_m4_15td)
        if parsed is None or parsed <= 0:
            raise ValueError("lambda_m4_15td must be finite and positive")
        self._lambda_m4_15td = parsed

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        asof = require_asof_timestamp(inp.asof_timestamp)
        warnings: List[str] = []
        quality_flags: Dict[str, Any] = {}

        reject_future_timestamp(asof, warnings, quality_flags)
        require_lineage_hash(inp.lineage_hashes, warnings, quality_flags)

        price = inp.market_data.get("price")
        high_52w = inp.market_data.get("high_52w")
        entry_lane = inp.market_data.get("entry_lane", ENTRY_LANE_BASE)

        if price is None or high_52w is None:
            warnings.append("missing required price or high_52w")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        price_f = finite_float(price)
        high_52w_f = finite_float(high_52w)
        if price_f is None or high_52w_f is None or high_52w_f <= 0 or price_f <= 0:
            warnings.append(f"invalid price or high_52w")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        price, high_52w = price_f, high_52w_f
        feat_dict = build_m4_source_features(inp, price=price, high_52w=high_52w, entry_lane=entry_lane)

        pit_passed = quality_flags.get("point_in_time_passed") is not False
        universe_rejection = operating_universe_rejection(
            inp.market_data, warnings, quality_flags, pattern_id=self.pattern_id,
        )
        pre_signal_rejection = market_data_quality_rejection(feat_dict, inp.market_data)
        if pre_signal_rejection is not None:
            quality_flags["market_data_quality_rejected"] = True
            feat_dict["rejection_reason"] = pre_signal_rejection

        fidelity = classify_fidelity(
            has_primary_data=True, has_secondary_data=True,
            point_in_time_passed=pit_passed, lookahead_guard_passed=True,
        )
        features = PatternFeatures(
            features=feat_dict, feature_manifest_version="m4-v1",
            fidelity_tier=fidelity, point_in_time_passed=pit_passed, lookahead_guard_passed=True,
        )

        signals: List[PatternSignal] = []
        breakout = price >= high_52w

        if universe_rejection is not None:
            feat_dict["rejection_reason"] = universe_rejection
            feat_dict["signal_generated"] = False
        elif pre_signal_rejection is not None:
            feat_dict["signal_generated"] = False
        elif entry_lane == ENTRY_LANE_FRESH:
            self._apply_fresh_lane(inp, feat_dict, warnings, quality_flags, signals)
        elif breakout:
            sig = _enrich_base_daily_breakout(
                feat_dict, inp, inp.market_data.get("cohort_extensions"), warnings, quality_flags,
                lambda_15td=self._lambda_m4_15td,
            )
            feat_dict["signal_generated"] = True
            signals.append(sig)
        else:
            feat_dict["rejection_reason"] = "below_high"
            feat_dict["signal_generated"] = False

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

    def _apply_fresh_lane(
        self, inp: PatternInput, feat_dict: Dict[str, Any],
        warnings: List[str], quality_flags: Dict[str, Any], signals: List[PatternSignal],
    ) -> None:
        activation_state = inp.market_data.get("activation_state", ACTIVATION_STATE_WATCHLIST)
        feat_dict["activation_state"] = activation_state
        copy_fields(feat_dict, inp.market_data, FRESH_ACTIVATION_FIELDS)

        base_nearness = feat_dict["base_nearness"]
        if activation_state == ACTIVATION_STATE_WATCHLIST:
            watchlist_passed = base_nearness >= 0.97 and not inp.market_data.get(
                "already_base_daily_fired", False
            )
            feat_dict["watchlist_passed"] = watchlist_passed
            if watchlist_passed:
                feat_dict["signal_generated"] = True
                signals.append(_build_fresh_watchlist_signal(inp, base_nearness, quality_flags))
            else:
                feat_dict["rejection_reason"] = "watchlist_not_qualified"
                feat_dict["signal_generated"] = False
            return

        activation_quality_rejection = market_data_quality_rejection(
            feat_dict, inp.market_data, require_fields=True,
        )
        if activation_quality_rejection is not None:
            quality_flags["market_data_quality_rejected"] = True
            feat_dict["activation_passed"] = False
            feat_dict["activation_failure_reason"] = activation_quality_rejection
            feat_dict["rejection_reason"] = activation_quality_rejection
            feat_dict["signal_generated"] = False
            warnings.append(f"fresh activation failed: {activation_quality_rejection}")
            return

        identity_passed = _fresh_activation_identity_passed(inp.market_data)
        quote_rej = quote_rejection(inp.market_data, quote_fields=QUOTE_FIELDS)
        feat_dict["activation_identity_passed"] = identity_passed
        feat_dict["quote_capture_passed"] = quote_rej is None

        last_price = finite_float(inp.market_data.get("last_price", inp.market_data.get("price")))
        range_conf = finite_float(inp.market_data.get("intraday_range_confirmation", 0.0))
        vol_conf = finite_float(inp.market_data.get("intraday_volume_confirmation", 0.0))
        if last_price is None:
            quality_flags["invalid_fresh_last_price"] = True
            warnings.append("invalid last_price for M4 fresh activation")
        if range_conf is None:
            quality_flags["invalid_range_confirmation"] = True
            warnings.append("invalid intraday_range_confirmation for M4 fresh activation")
            range_conf = 0.0
        if vol_conf is None:
            quality_flags["invalid_volume_confirmation"] = True
            warnings.append("invalid intraday_volume_confirmation for M4 fresh activation")
            vol_conf = 0.0
        spread_pct = inp.market_data.get("spread_pct_vs_eval_quote")
        spread_pct_float = finite_float(spread_pct) if spread_pct is not None else None
        spread_passed = (
            inp.market_data.get("spread_discipline_passed") is True
            if "spread_discipline_passed" in inp.market_data
            else spread_pct_float is not None and spread_pct_float <= FRESH_SPREAD_CAP
        )
        freshness_passed = inp.market_data.get("signal_freshness_passed") is True

        fresh = compute_m4_fresh_features(
            last_price=last_price or 0.0, high_52w=feat_dict["H_52w"],
            intraday_range_confirmation=range_conf, intraday_volume_confirmation=vol_conf,
        )
        feat_dict.update(fresh)
        feat_dict["last_price"] = last_price
        feat_dict["fresh_high_break_passed"] = last_price is not None and last_price > feat_dict["H_52w"]
        feat_dict["range_confirmation_passed"] = range_conf >= 1.0
        feat_dict["volume_confirmation_passed"] = vol_conf >= 1.0
        feat_dict["spread_pct_vs_eval_quote"] = spread_pct_float
        feat_dict["spread_discipline_passed"] = spread_passed
        feat_dict["signal_freshness_passed"] = freshness_passed

        activation_passed = (
            feat_dict["fresh_high_break_passed"]
            and fresh["fresh_breakout_extension"] > 0
            and feat_dict["range_confirmation_passed"]
            and feat_dict["volume_confirmation_passed"]
            and identity_passed
            and feat_dict["quote_capture_passed"]
            and spread_passed and freshness_passed
        )
        feat_dict["activation_passed"] = activation_passed
        if not activation_passed:
            feat_dict["activation_failure_reason"] = _fresh_activation_failure_reason(feat_dict)
            feat_dict["rejection_reason"] = feat_dict["activation_failure_reason"]
            feat_dict["signal_generated"] = False
            warnings.append(f"fresh activation failed: {feat_dict['activation_failure_reason']}")
            return

        x_fresh = fresh["x_m4_fresh"]
        feat_dict["signal_generated"] = True
        signals.append(_build_fresh_activation_signal(inp, feat_dict, x_fresh, quality_flags, lambda_15td=self._lambda_m4_15td))
