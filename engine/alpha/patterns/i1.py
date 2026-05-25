"""
I1 — Gap and Go Detector.

Vault source: Engineering/Patterns/I1-GapAndGo/

Thesis: right_tail_convex. Stocks that gap up >= 3% at the open AND
confirm in the first 30 minutes (positive return + above-average volume)
exhibit strong continuation over the following 3 trading days.

Exposure formula (EXPOSURE.md):
  gap_pct = (open - prev_close) / prev_close
  gap_magnitude = clip(gap_pct / sigma_20d, 0.0, 5.0)
  confirmation_gate = 1.0 if return_30min > 0 AND volume_30min > avg_volume_30min_20d else 0.0
  volume_weight: tiered 1.0/1.25/1.5/2.0 by volume_ratio_30min
  x_i1 = gap_magnitude * confirmation_gate * volume_weight

Expected-return bridge (SPEC.md / EXPOSURE.md):
  lambda_I1_monthly = 3.47% (LPS 2019 overnight alpha)
  amplification = 1.75 (microcap)
  lambda_I1_3td = 3.47% * 1.75 * (3/21) = ~0.867%
  raw_expected_edge = x_i1 * lambda_I1_3td

Signal admission:
  1. Operating-universe membership (fail-closed)
  2. Market data quality (market_data_status, halt_status, corporate_action_filter_passed required)
  3. Normal opening auction, valid timestamps, and fresh executable quote
  4. raw gap_pct >= 0.03 before source-feature rounding
  5. confirmation_gate = 1.0 (positive 30-min return AND above-avg volume)
  6. x_i1 > 0

Signal fires intraday at ~10:00 AM ET after 30-min confirmation window.
Evaluation cutoff: 10:15 AM ET. Signals after cutoff rejected as data_delay.
Routing: Class C (marketable limit, 120-second cancel).

Evidence: each fired signal persists lambda_I1_monthly, microcap_amplification,
and amplified_lambda_I1_3td so shadow validation can audit the thesis assumption.
Feature evidence uses canonical DATA.md key x_i1; X_I1 remains only formula notation.
"""

from __future__ import annotations

from datetime import datetime, time
import math
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

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
    DEFAULT_QUOTE_FIELDS,
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
LAMBDA_I1_MONTHLY = 0.0347  # 3.47% per month (LPS 2019 overnight alpha)
AMPLIFICATION = 1.75  # microcap amplification
HOLD_DAYS = 3
LAMBDA_I1_3TD = LAMBDA_I1_MONTHLY * AMPLIFICATION * (HOLD_DAYS / 21.0)  # ~0.00867
X_I1_STRENGTH_DIVISOR = 10.0  # I-track exposure range wider than M-track
SIGNAL_HORIZON = "3d"
MIN_GAP_PCT = 0.03  # minimum 3% gap
GAP_MAGNITUDE_CAP = 5.0
VOLUME_WEIGHT_CAP = 2.0
QUOTE_FIELDS = DEFAULT_QUOTE_FIELDS
TIMESTAMP_FIELDS = ("evaluation_timestamp", "data_cutoff_timestamp")
EASTERN_TZ = ZoneInfo("America/New_York")
EVALUATION_CUTOFF_ET = time(10, 15, 0)


# ---------------------------------------------------------------------------
# Pure computation helpers
# ---------------------------------------------------------------------------

def compute_gap_magnitude(gap_pct: float, sigma_20d: float) -> float:
    """Per EXPOSURE.md: clip(gap_pct / sigma_20d, 0.0, 5.0)."""
    if not math.isfinite(gap_pct) or not math.isfinite(sigma_20d):
        return 0.0
    if sigma_20d <= 0 or gap_pct <= 0:
        return 0.0
    return max(0.0, min(gap_pct / sigma_20d, GAP_MAGNITUDE_CAP))


def compute_confirmation_gate(return_30min: float, volume_30min: float, avg_volume_30min_20d: float) -> float:
    """
    Per EXPOSURE.md: 1.0 if return > 0 AND volume > avg, else 0.0.

    The strict boundary is intentional: a flat first 30 minutes is a
    stalled gap, not "go" confirmation, and exactly average volume is not
    above-average participation.
    """
    if return_30min > 0 and volume_30min > avg_volume_30min_20d:
        return 1.0
    return 0.0


def compute_volume_weight(volume_ratio_30min: float) -> float:
    """Per EXPOSURE.md tiered weighting."""
    if volume_ratio_30min >= 3.0:
        return VOLUME_WEIGHT_CAP
    if volume_ratio_30min >= 2.0:
        return 1.5
    if volume_ratio_30min >= 1.5:
        return 1.25
    if volume_ratio_30min >= 1.0:
        return 1.0
    return 0.0  # should not reach here if confirmation gate passed


def _data_confidence(inp: PatternInput, quality_flags: Dict[str, Any]) -> float:
    return compute_data_confidence(
        quality_flags,
        field_confidence_sources=(inp.market_data, inp.fundamental_data, inp.event_data),
    )


def _date_prefix(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "date"):
        return value.date().isoformat()
    text = str(value).strip()
    return text[:10] if len(text) >= 10 else None


def _copy_diagnostic_fields(feat_dict: Dict[str, Any], *sources: Dict[str, Any]) -> None:
    for source in sources:
        for key in (
            "evaluation_run_id", "data_cutoff_timestamp", "price_at_10am", "pre_market_price",
            "gap_source", "candidate_eval_bid", "candidate_eval_ask",
            "candidate_eval_quote_timestamp", "quote_age_ms", "quote_freshness_max_ms",
            "effective_spread_bps", "evaluation_timestamp", "opening_auction_quality",
            "halt_status", "corporate_action_filter_passed", "market_data_status",
            "hazard_score_at_signal", "filing_veto_status", "m4_also_firing", "m2_also_firing",
        ):
            val = source.get(key)
            if val is not None:
                feat_dict[key] = val

    feat_dict.setdefault("gap_source", "unknown")
    feat_dict.setdefault("opening_auction_quality", "normal")
    feat_dict.setdefault("filing_veto_status", "not_computed")


def _parse_market_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=EASTERN_TZ)
    return parsed.astimezone(EASTERN_TZ)


def _timestamp_rejection(market_data: Dict[str, Any]) -> Optional[str]:
    if any(market_data.get(field) is None for field in TIMESTAMP_FIELDS):
        return "insufficient_timestamp_data"

    evaluation_ts = _parse_market_timestamp(market_data["evaluation_timestamp"])
    data_cutoff_ts = _parse_market_timestamp(market_data["data_cutoff_timestamp"])
    if evaluation_ts is None or data_cutoff_ts is None:
        return "insufficient_timestamp_data"

    if data_cutoff_ts > evaluation_ts:
        return "insufficient_timestamp_data"

    if evaluation_ts.time() > EVALUATION_CUTOFF_ET:
        return "data_delay"

    return None


def _pre_signal_rejection_reason(feat_dict: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[str]:
    market_data_rejection = market_data_quality_rejection(
        feat_dict,
        market_data,
        require_fields=True,
        missing_rejection="missing_market_data_quality",
        halt_rejection="halt_during_confirmation",
        corporate_action_rejection="spurious_gap_corporate_action",
    )
    if market_data_rejection is not None:
        return market_data_rejection

    if feat_dict.get("opening_auction_quality") != "normal":
        return "opening_auction_quality_failed"

    timestamp_rej = _timestamp_rejection(market_data)
    if timestamp_rej is not None:
        return timestamp_rej

    quote_rej = quote_rejection(market_data, quote_fields=QUOTE_FIELDS)
    if quote_rej is not None:
        return quote_rej

    return None


def _reject_signal(
    feat_dict: Dict[str, Any],
    reason: str,
    *,
    confirmation_gate: float = 0.0,
    volume_weight: float = 0.0,
    x_i1: float = 0.0,
) -> None:
    feat_dict["rejection_reason"] = reason
    feat_dict["signal_generated"] = False
    feat_dict["confirmation_gate"] = confirmation_gate
    feat_dict["volume_weight"] = volume_weight
    feat_dict["x_i1"] = x_i1


def _copy_gap_features(feat_dict: Dict[str, Any], inp: PatternInput) -> tuple[float, float]:
    market_data = inp.market_data
    prev_close = finite_float(market_data["prev_close"])
    open_price = finite_float(market_data["open_price"])
    sigma_20d = finite_float(market_data["sigma_20d"])
    if prev_close is None or open_price is None or sigma_20d is None:
        raise ValueError("gap features require finite prev_close, open_price, and sigma_20d")

    gap_pct = (open_price - prev_close) / prev_close
    gap_mag = compute_gap_magnitude(gap_pct, sigma_20d)

    feat_dict["prev_close"] = prev_close
    feat_dict["open_price"] = open_price
    feat_dict["sigma_20d"] = sigma_20d
    feat_dict["gap_pct"] = round(gap_pct, 6)
    feat_dict["gap_magnitude"] = round(gap_mag, 6)
    _copy_diagnostic_fields(feat_dict, inp.market_data, inp.fundamental_data, inp.event_data)
    session_date = (
        market_data.get("gap_session_date")
        or market_data.get("trading_session")
        or _date_prefix(market_data.get("evaluation_timestamp"))
        or _date_prefix(market_data.get("data_cutoff_timestamp"))
    )
    set_signal_identity(
        feat_dict,
        pattern_id=PatternId.I1,
        ticker=inp.ticker,
        components={
            "session_date": session_date,
            "prev_close": prev_close,
            "open_price": open_price,
        },
        source="gap_session_event",
    )
    return gap_pct, gap_mag


def _copy_confirmation_inputs(
    feat_dict: Dict[str, Any],
    market_data: Dict[str, Any],
    warnings: List[str],
    quality_flags: Dict[str, Any],
) -> Optional[tuple[float, float, float, float]]:
    return_30min = market_data.get("return_30min")
    volume_30min = market_data.get("volume_30min")
    avg_volume_30min_20d = market_data.get("avg_volume_30min_20d")

    if return_30min is None or volume_30min is None:
        _reject_signal(feat_dict, "missing_confirmation_data")
        warnings.append("missing return_30min or volume_30min for confirmation")
        return None

    return_30min_f = finite_float(return_30min)
    volume_30min_f = finite_float(volume_30min)
    if return_30min_f is None or volume_30min_f is None:
        _reject_signal(feat_dict, "missing_confirmation_data")
        warnings.append("invalid return_30min or volume_30min for confirmation")
        quality_flags["missing_confirmation_data"] = True
        return None
    return_30min = return_30min_f
    volume_30min = volume_30min_f

    baseline_volume_proxy = False
    avg_volume_30min_20d_f = finite_float(avg_volume_30min_20d) if avg_volume_30min_20d is not None else None
    if avg_volume_30min_20d_f is None or avg_volume_30min_20d_f <= 0:
        avg_volume_30min_20d = volume_30min * 0.5  # conservative per DATA.md edge case
        baseline_volume_proxy = True
        quality_flags["baseline_volume_proxy"] = True
        warnings.append("avg_volume_30min_20d unavailable — using conservative proxy")
    else:
        avg_volume_30min_20d = avg_volume_30min_20d_f

    volume_ratio = volume_30min / avg_volume_30min_20d if avg_volume_30min_20d > 0 else 0.0

    feat_dict["return_30min"] = round(return_30min, 6)
    feat_dict["volume_30min"] = volume_30min
    feat_dict["avg_volume_30min_20d"] = round(avg_volume_30min_20d, 2)
    feat_dict["volume_ratio_30min"] = round(volume_ratio, 6)
    feat_dict["baseline_volume_proxy"] = baseline_volume_proxy
    return return_30min, volume_30min, avg_volume_30min_20d, volume_ratio


def _compute_i1_exposure(
    feat_dict: Dict[str, Any],
    gap_magnitude: float,
    return_30min: float,
    volume_30min: float,
    avg_volume_30min_20d: float,
    volume_ratio: float,
) -> Optional[float]:
    conf_gate = compute_confirmation_gate(return_30min, volume_30min, avg_volume_30min_20d)
    feat_dict["confirmation_gate"] = conf_gate

    if conf_gate == 0.0:
        _reject_signal(feat_dict, "confirmation_failed")
        return None

    vol_weight = compute_volume_weight(volume_ratio)
    x_i1 = gap_magnitude * conf_gate * vol_weight

    feat_dict["volume_weight"] = vol_weight
    feat_dict["x_i1"] = round(x_i1, 6)
    feat_dict["signal_generated"] = True
    return x_i1


def _build_i1_signal(
    feat_dict: Dict[str, Any],
    x_i1: float,
    quality_flags: Dict[str, Any],
    inp: Optional[PatternInput] = None,
    lambda_3td: float = LAMBDA_I1_3TD,
) -> PatternSignal:
    raw_expected_edge = round(x_i1 * lambda_3td, 6)
    signal_strength = round(min(x_i1 / X_I1_STRENGTH_DIVISOR, 1.0), 6)
    feat_dict["lambda_I1_monthly"] = LAMBDA_I1_MONTHLY
    feat_dict["microcap_amplification"] = AMPLIFICATION
    feat_dict["validated_or_shadow_lambda_I1_3td"] = lambda_3td
    feat_dict["lambda_I1_3td"] = round(lambda_3td, 8)
    feat_dict["lambda_I1_default_3td"] = round(LAMBDA_I1_3TD, 8)
    feat_dict["lambda_I1_source"] = (
        "shadow_prior" if lambda_3td == LAMBDA_I1_3TD else "validated_or_injected"
    )
    feat_dict["amplified_lambda_I1_3td"] = round(lambda_3td, 8)
    feat_dict["expected_return_priors"] = {"gross_bps": round(raw_expected_edge * 10_000, 2)}

    return PatternSignal(
        direction=SignalDirection.LONG,
        raw_signal_strength=signal_strength,
        raw_expected_edge=raw_expected_edge,
        signal_horizon=SIGNAL_HORIZON,
        route_class=RouteClass.C,
        data_confidence=_data_confidence(inp, quality_flags) if inp is not None else compute_data_confidence(quality_flags),
    )


# ---------------------------------------------------------------------------
# Activation enrichment
# ---------------------------------------------------------------------------

def _enrich_i1_signal(
    feat_dict: Dict[str, Any],
    inp: PatternInput,
    warnings: List[str],
    quality_flags: Dict[str, Any],
    lambda_3td: float = LAMBDA_I1_3TD,
) -> Optional[PatternSignal]:
    """
    Compute gap features, confirmation gate, exposure, and return
    PatternSignal if all admission gates pass. All feature fields set here.
    """
    gap_pct, gap_mag = _copy_gap_features(feat_dict, inp)

    # Minimum gap gate
    if gap_pct < MIN_GAP_PCT:
        _reject_signal(feat_dict, "gap_below_minimum")
        return None

    pre_signal_rejection = _pre_signal_rejection_reason(feat_dict, inp.market_data)
    if pre_signal_rejection is not None:
        _reject_signal(
            feat_dict,
            pre_signal_rejection,
            confirmation_gate=0.0,
            volume_weight=0.0,
            x_i1=0.0,
        )
        return None

    confirmation_inputs = _copy_confirmation_inputs(feat_dict, inp.market_data, warnings, quality_flags)
    if confirmation_inputs is None:
        return None

    x_i1 = _compute_i1_exposure(feat_dict, gap_mag, *confirmation_inputs)
    if x_i1 is None:
        return None

    return _build_i1_signal(feat_dict, x_i1, quality_flags, inp=inp, lambda_3td=lambda_3td)


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

class I1Detector(BasePatternDetector):
    """I1 Gap and Go detector."""

    pattern_id = PatternId.I1
    version = "1.0"
    track = PatternTrack.INTRADAY
    thesis_category = ThesisCategory.RIGHT_TAIL_CONVEX
    route_class = RouteClass.C

    def __init__(self, lambda_i1_3td: float = LAMBDA_I1_3TD):
        parsed = finite_float(lambda_i1_3td)
        if parsed is None or parsed <= 0:
            raise ValueError("lambda_i1_3td must be finite and positive")
        self._lambda_i1_3td = parsed

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        asof = require_asof_timestamp(inp.asof_timestamp)
        warnings: List[str] = []
        quality_flags: Dict[str, Any] = {}

        reject_future_timestamp(asof, warnings, quality_flags)
        require_lineage_hash(inp.lineage_hashes, warnings, quality_flags)

        # Required inputs
        prev_close = inp.market_data.get("prev_close")
        open_price = inp.market_data.get("open_price")
        sigma_20d = inp.market_data.get("sigma_20d")

        if prev_close is None or open_price is None or sigma_20d is None:
            warnings.append("missing required fields (prev_close, open_price, or sigma_20d)")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        prev_close_f = finite_float(prev_close)
        open_price_f = finite_float(open_price)
        sigma_20d_f = finite_float(sigma_20d)

        if (
            prev_close_f is None
            or open_price_f is None
            or sigma_20d_f is None
            or prev_close_f <= 0
            or open_price_f <= 0
            or sigma_20d_f <= 0
        ):
            warnings.append("invalid prev_close, open_price, or sigma_20d")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        feat_dict: Dict[str, Any] = {}

        # Universe check
        universe_rejection = operating_universe_rejection(
            inp.market_data, warnings, quality_flags, pattern_id=self.pattern_id,
        )

        # Fidelity
        pit_passed = quality_flags.get("point_in_time_passed") is not False
        fidelity = classify_fidelity(
            has_primary_data=True, has_secondary_data=True,
            point_in_time_passed=pit_passed, lookahead_guard_passed=True,
        )

        signals: List[PatternSignal] = []

        if universe_rejection is not None:
            _copy_gap_features(feat_dict, inp)
            _reject_signal(feat_dict, universe_rejection)
        else:
            sig = _enrich_i1_signal(feat_dict, inp, warnings, quality_flags, lambda_3td=self._lambda_i1_3td)
            if sig is not None:
                signals.append(sig)

        features = PatternFeatures(
            features=feat_dict, feature_manifest_version="i1-v1",
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
