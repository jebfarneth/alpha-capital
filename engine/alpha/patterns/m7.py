"""
M7 — Pure Technical Multi-Day Detector.

Vault source: Engineering/Patterns/M7-PureTechnical/

Thesis: continuation. ML-predicted top-decile stocks exhibit positive
expected excess returns over the following 10 trading days. Cakici et al.
(2023): 2.4x small-cap alpha amplification. MEDIUM-STRONG with Bootstrap
Reality Check (STW 1999) confidence tracking.

The detector does NOT train ML, run GBRT, or compute features.
It consumes an immutable prediction/feature row and verifies reproducible
lineage before emitting a signal.

Exposure formula (EXPOSURE.md):
  X_M7 = predicted_return_rank_pct * decay_haircut
  raw_expected_edge = X_M7 * lambda_M7_10td

Signal admission:
  1. Operating-universe membership (fail-closed)
  2. Market data quality (shared guard, EOD M-track)
  3. Reproducible lineage (model_version, training_run_id, prediction_run_id,
     feature_snapshot_id, feature_manifest_version, model_artifact_hash,
     data_cutoff_timestamp)
  4. point_in_time_passed is True
  5. predicted_return_rank_pct >= 0.90 (top decile)
  6. Valid decay_haircut in [0.50, 0.65]

RC confidence status is NOT a signal gate.
Routing: Class A (midpoint limit, day-valid).
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
LAMBDA_M7_MONTHLY = 0.0146  # 1.46%/month (Cakici et al. COMB six-factor alpha)
MICROCAP_AMPLIFICATION = 1.75
LAMBDA_M7_10TD = LAMBDA_M7_MONTHLY * MICROCAP_AMPLIFICATION * 10.0 / 21.0  # ~0.01216
SIGNAL_HORIZON = "10d"
MIN_RANK_PCT = 0.90  # top-decile gate
DECAY_HAIRCUT_MIN = 0.50
DECAY_HAIRCUT_MAX = 0.65
DECAY_DAYS_FULL_WINDOW = 63  # quarterly retraining cycle
X_M7_CAP = 1.0  # rank_pct in [0,1] * decay in [0.50,0.65] → max ~0.65
MISSING_FEATURE_RATIO_LIMIT = 0.30
VALID_RC_STATUSES = {"pending", "passed", "failed"}
PLACEHOLDER_VALUES = {"", "n/a", "na", "none", "null", "pending", "nan"}

LINEAGE_FIELDS = (
    "model_version", "training_run_id", "prediction_run_id",
    "feature_snapshot_id", "feature_manifest_version",
    "model_artifact_hash", "data_cutoff_timestamp",
)
LINEAGE_ID_FIELDS = tuple(field for field in LINEAGE_FIELDS if field != "data_cutoff_timestamp")

LOAD_BEARING_FIELDS = (
    "predicted_return_rank_pct", "predicted_return", "decay_haircut",
    "days_since_retrain", "point_in_time_passed",
    "feature_count", "missing_feature_count",
    "rc_confidence_status", "prediction_run_status", "is_canonical",
    *LINEAGE_FIELDS,
)

DIAGNOSTIC_FIELDS = (
    "validation_weight_multiplier", "top_3_features", "feature_importance_drift",
    "stale_fundamentals", "model_stale_alert",
    "hazard_score_at_signal", "filing_veto_status",
    "liquidity_score", "sector", "market_cap_usd", "price_at_signal",
    "market_data_status", "halt_status", "corporate_action_filter_passed",
    "overlapping_pattern_ids", "m4_also_firing", "m5_also_firing", "m6_also_firing",
)


# ---------------------------------------------------------------------------
# Pure computation helpers
# ---------------------------------------------------------------------------

def compute_decay_haircut(days_since_retrain: Any) -> Optional[float]:
    """Linear decay from 0.65 to 0.50 over 63 trading days, clipped."""
    days_since_retrain = integral_int(days_since_retrain)
    if days_since_retrain is None:
        return None
    if days_since_retrain < 0:
        return None
    raw = DECAY_HAIRCUT_MAX - (DECAY_HAIRCUT_MAX - DECAY_HAIRCUT_MIN) * (days_since_retrain / DECAY_DAYS_FULL_WINDOW)
    return max(DECAY_HAIRCUT_MIN, min(raw, DECAY_HAIRCUT_MAX))


def _data_confidence(inp: PatternInput, quality_flags: Dict[str, Any]) -> float:
    return compute_data_confidence(
        quality_flags,
        field_confidence_sources=(inp.market_data, inp.fundamental_data, inp.event_data),
    )


# ---------------------------------------------------------------------------
# Diagnostic field helpers
# ---------------------------------------------------------------------------

def _clean_required_value(value: Any, *, allow_datetime: bool = False) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in PLACEHOLDER_VALUES:
            return None
        return value
    if allow_datetime and hasattr(value, "isoformat"):
        return value
    return None


def _clean_lineage_field(field: str, value: Any) -> Optional[Any]:
    return _clean_required_value(value, allow_datetime=field == "data_cutoff_timestamp")


def _copy_diagnostic_fields(
    feat_dict: Dict[str, Any],
    market_data: Dict[str, Any],
    *sources: Dict[str, Any],
) -> None:
    # Prediction and lineage fields are load-bearing and must come from
    # market_data only; other sources are diagnostics and cannot overwrite them.
    for key in LOAD_BEARING_FIELDS:
        val = market_data.get(key)
        if val is not None:
            feat_dict[key] = val
    for key in DIAGNOSTIC_FIELDS:
        val = market_data.get(key)
        if val is not None:
            feat_dict[key] = val
    for source in sources:
        for key in DIAGNOSTIC_FIELDS:
            val = source.get(key)
            if val is not None:
                feat_dict[key] = val
    feat_dict.setdefault("filing_veto_status", "not_computed")


def _reject_signal(feat_dict: Dict[str, Any], reason: str) -> None:
    feat_dict["rejection_reason"] = reason
    feat_dict["signal_generated"] = False
    feat_dict.setdefault("exposure_x_m7", 0.0)


# ---------------------------------------------------------------------------
# Signal enrichment
# ---------------------------------------------------------------------------

def _enrich_m7_signal(
    feat_dict: Dict[str, Any],
    inp: PatternInput,
    warnings: List[str],
    quality_flags: Dict[str, Any],
    lambda_10td: float,
) -> Optional[PatternSignal]:
    md = inp.market_data

    _copy_diagnostic_fields(feat_dict, inp.market_data, inp.fundamental_data, inp.event_data)

    # Market data quality (EOD M-track; not require_fields)
    pre_signal_rejection = market_data_quality_rejection(feat_dict, md)
    if pre_signal_rejection is not None:
        quality_flags["market_data_quality_rejected"] = True
        _reject_signal(feat_dict, pre_signal_rejection)
        return None

    # --- Load-bearing fields from market_data only ---

    # PIT contract
    if md.get("point_in_time_passed") is not True:
        quality_flags["point_in_time_passed"] = False
        _reject_signal(feat_dict, "model_output_not_point_in_time")
        return None

    # Model lineage
    lineage_values = {field: _clean_lineage_field(field, md.get(field)) for field in LINEAGE_FIELDS}
    missing_lineage = [field for field, value in lineage_values.items() if value is None]
    if missing_lineage:
        quality_flags["missing_model_lineage"] = True
        warnings.append(f"missing lineage fields: {', '.join(missing_lineage)}")
        _reject_signal(feat_dict, "missing_model_lineage")
        return None
    for field, value in lineage_values.items():
        feat_dict[field] = value

    # Prediction run status
    run_status = md.get("prediction_run_status")
    if run_status is not None and str(run_status).strip() != "completed":
        _reject_signal(feat_dict, "prediction_run_not_completed")
        return None

    # Canonical flag
    is_canonical = md.get("is_canonical")
    if is_canonical is not None and is_canonical is not True:
        _reject_signal(feat_dict, "prediction_run_not_canonical")
        return None

    # Feature completeness
    feature_count_raw = md.get("feature_count")
    missing_feature_count_raw = md.get("missing_feature_count")
    feature_count = integral_int(feature_count_raw)
    missing_feature_count = integral_int(missing_feature_count_raw)
    if feature_count_raw is not None or missing_feature_count_raw is not None:
        if (
            feature_count is None
            or missing_feature_count is None
            or feature_count <= 0
            or missing_feature_count < 0
            or missing_feature_count > feature_count
        ):
            quality_flags["invalid_feature_completeness"] = True
            _reject_signal(feat_dict, "invalid_feature_completeness")
            return None
        missing_ratio = missing_feature_count / feature_count
        if missing_ratio > MISSING_FEATURE_RATIO_LIMIT:
            quality_flags["too_many_missing_features"] = True
            _reject_signal(feat_dict, "too_many_missing_features")
            return None

    # Predicted return rank
    rank_pct = finite_float(md.get("predicted_return_rank_pct"))
    if rank_pct is None or not (0.0 <= rank_pct <= 1.0):
        _reject_signal(feat_dict, "invalid_prediction_rank")
        return None
    feat_dict["predicted_return_rank_pct"] = round(rank_pct, 6)

    # Predicted return (diagnostic but required for reproducibility)
    predicted_return = finite_float(md.get("predicted_return"))
    if predicted_return is None:
        _reject_signal(feat_dict, "invalid_predicted_return")
        return None
    feat_dict["predicted_return"] = round(predicted_return, 8)

    # Decay haircut
    explicit_decay = finite_float(md.get("decay_haircut"))
    days_since_raw = md.get("days_since_retrain")
    days_since = integral_int(days_since_raw) if days_since_raw is not None else None
    if days_since_raw is not None and days_since is None:
        quality_flags["invalid_days_since_retrain"] = True
        _reject_signal(feat_dict, "invalid_days_since_retrain")
        return None
    computed_decay = compute_decay_haircut(days_since) if days_since is not None else None

    if explicit_decay is not None and computed_decay is not None:
        if abs(explicit_decay - computed_decay) > 0.02:
            quality_flags["inconsistent_decay"] = True
            _reject_signal(feat_dict, "inconsistent_decay_haircut")
            return None
        decay = explicit_decay
    elif explicit_decay is not None:
        decay = explicit_decay
    elif computed_decay is not None:
        decay = computed_decay
    else:
        _reject_signal(feat_dict, "invalid_decay_haircut")
        return None

    if not (DECAY_HAIRCUT_MIN <= decay <= DECAY_HAIRCUT_MAX):
        _reject_signal(feat_dict, "invalid_decay_haircut")
        return None

    feat_dict["decay_haircut"] = round(decay, 6)
    feat_dict["days_since_retrain"] = days_since

    # Top-decile gate
    if rank_pct < MIN_RANK_PCT:
        _reject_signal(feat_dict, "prediction_below_threshold")
        return None

    # Compute exposure
    x_m7 = rank_pct * decay
    feat_dict["exposure_x_m7"] = round(x_m7, 6)

    # RC confidence (NOT a gate)
    rc_status_raw = md.get("rc_confidence_status")
    if rc_status_raw is None:
        rc_status = "pending"
        quality_flags["missing_rc_status"] = True
    else:
        rc_status = str(rc_status_raw).strip().lower()
        if rc_status not in VALID_RC_STATUSES:
            rc_status = "pending"
            quality_flags["invalid_rc_status"] = True
            warnings.append("invalid rc_confidence_status defaulted to pending")
    feat_dict["rc_confidence_status"] = rc_status
    if rc_status == "failed":
        feat_dict["operator_review_flag"] = "M7_rc_failed"

    # Raw expected edge
    raw_expected_edge = round(x_m7 * lambda_10td, 6)
    signal_strength = round(rank_pct, 6)

    feat_dict["signal_generated"] = True
    feat_dict["lambda_M7_monthly"] = LAMBDA_M7_MONTHLY
    feat_dict["microcap_amplification"] = MICROCAP_AMPLIFICATION
    feat_dict["validated_or_shadow_lambda_M7_10td"] = lambda_10td
    feat_dict["lambda_M7_10td"] = round(lambda_10td, 8)
    feat_dict["lambda_M7_default_10td"] = round(LAMBDA_M7_10TD, 8)
    feat_dict["lambda_M7_source"] = (
        "shadow_prior" if lambda_10td == LAMBDA_M7_10TD else "validated_or_injected"
    )
    feat_dict["expected_return_priors"] = {"gross_bps": round(raw_expected_edge * 10_000, 2)}

    # Signal identity — keyed on immutable prediction row
    feature_snap_id = lineage_values["feature_snapshot_id"]
    prediction_run_id = lineage_values["prediction_run_id"]
    if feature_snap_id is not None:
        id_components = {"feature_snapshot_id": feature_snap_id}
    else:
        id_components = {"prediction_run_id": prediction_run_id}
    set_signal_identity(
        feat_dict,
        pattern_id=PatternId.M7,
        ticker=inp.ticker,
        components=id_components,
        source="m7_prediction_row",
    )

    return PatternSignal(
        direction=SignalDirection.LONG,
        raw_signal_strength=signal_strength,
        raw_expected_edge=raw_expected_edge,
        signal_horizon=SIGNAL_HORIZON,
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

class M7Detector(BasePatternDetector):
    """M7 Pure Technical Multi-Day detector."""

    pattern_id = PatternId.M7
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.CONTINUATION
    route_class = RouteClass.A

    def __init__(self, lambda_m7_10td: float = LAMBDA_M7_10TD):
        parsed = finite_float(lambda_m7_10td)
        if parsed is None or parsed <= 0:
            raise ValueError("lambda_m7_10td must be finite and positive")
        self._lambda_m7_10td = parsed

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        asof = require_asof_timestamp(inp.asof_timestamp)
        warnings: List[str] = []
        quality_flags: Dict[str, Any] = {}

        reject_future_timestamp(asof, warnings, quality_flags)
        require_lineage_hash(inp.lineage_hashes, warnings, quality_flags)

        md = inp.market_data

        # Required: prediction rank and lineage must be present
        rank_raw = md.get("predicted_return_rank_pct")
        has_lineage = all(_clean_lineage_field(field, md.get(field)) is not None for field in LINEAGE_FIELDS)

        if rank_raw is None or not has_lineage:
            warnings.append("missing required M7 prediction/lineage fields")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        # Validate rank is parseable
        rank_check = finite_float(rank_raw)
        if rank_check is None:
            warnings.append("invalid predicted_return_rank_pct")
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
            _copy_diagnostic_fields(feat_dict, inp.market_data, inp.fundamental_data, inp.event_data)
            _reject_signal(feat_dict, universe_rejection)
        else:
            sig = _enrich_m7_signal(feat_dict, inp, warnings, quality_flags, self._lambda_m7_10td)
            if sig is not None:
                signals.append(sig)

        # Re-evaluate PIT after enrichment may have set it
        pit_passed = quality_flags.get("point_in_time_passed") is not False
        fidelity = classify_fidelity(
            has_primary_data=True, has_secondary_data=True,
            point_in_time_passed=pit_passed, lookahead_guard_passed=True,
        )

        features = PatternFeatures(
            features=feat_dict, feature_manifest_version="m7-v1",
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
