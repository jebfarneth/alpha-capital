"""
M2 — Insider Cluster Detector.

Vault source: Engineering/Patterns/M2-InsiderCluster/

Thesis: right_tail_convex. Clusters of opportunistic insider open-market
purchases predict positive drift over the following 20 trading days.
CMP (2012) documents 82 bps/month five-factor alpha for opportunistic
trades vs zero for routine trades.

Exposure formula (EXPOSURE.md):
  X_M2 = min(
      exp(-days_since_last_opp_buy_filing_detected / 10)
      * log(1 + n_distinct_opp_buyers_30d) / log(3)
      * mean_trade_intensity_weight,
      3.0
  )

Signal admission:
  1. Operating-universe membership (fail-closed)
  2. Market data quality (shared guard, EOD M-track)
  3. At least 2 distinct classified opportunistic buyers in trailing 30d
  4. Most recent opp buy filing detection <= 20 calendar days
  5. X_M2 > 0
  6. Production clock: filing_detected_at / filing_date, NEVER transaction_date

Routing: Class A (midpoint limit, day-valid).
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
LAMBDA_M2_MONTHLY = 0.0082  # 0.82%/month (CMP 2012 five-factor alpha)
LAMBDA_M2_20TD = LAMBDA_M2_MONTHLY * 20.0 / 21.0  # ~0.00781
SIGNAL_HORIZON = "20d"
X_M2_CAP = 3.0
MIN_OPP_BUYERS = 2  # minimum distinct classified opportunistic buyers
MAX_DAYS_SINCE_LAST_FILING = 20  # filing must be within 20 calendar days
CLUSTER_WINDOW_DAYS = 30  # trailing 30 calendar days for cluster detection
DECAY_TAU = 10.0  # exp(-days / 10) filing-detection decay
LIVE_SOURCE_AUTHORITIES = {"sec_edgar", "fmp_backfill"}
MISSING_TRADE_INTENSITY_DEFAULT = 0.75
OPPORTUNISTIC_SELL_CLUSTER_HAIRCUT = 0.75
SHADOW_DECAY_TAUS = (5.0, 10.0, 15.0)
ROLE_SHADOW_PRIOR_WEIGHTS = {
    "unknown": 1.0,
    "ten_percent_holder": 1.0,
    "director": 1.0,
    "other_officer": 1.10,
    "c_suite": 1.20,
}
ROLE_SHADOW_SENIORITY = {
    "unknown": 0,
    "ten_percent_holder": 1,
    "director": 2,
    "other_officer": 3,
    "c_suite": 4,
}


# ---------------------------------------------------------------------------
# Pure computation helpers
# ---------------------------------------------------------------------------

def _accession_list(value: Any) -> List[str]:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized and normalized.lower() not in {"n/a", "na", "none", "null", "pending"}:
            return [normalized]
        return []
    if isinstance(value, (list, tuple, set)):
        accessions: List[str] = []
        for item in value:
            accessions.extend(_accession_list(item))
        return accessions
    return []


def _has_accession_proof(value: Any) -> bool:
    return bool(_accession_list(value))


def _flag_is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    return False


def _flag_or_positive_count(value: Any) -> bool:
    if _flag_is_true(value):
        return True
    count = finite_float(value)
    return count is not None and count > 0


def _flatten_role_terms(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        terms: List[str] = []
        for item in value.values():
            terms.extend(_flatten_role_terms(item))
        return terms
    if isinstance(value, (list, tuple, set)):
        terms = []
        for item in value:
            terms.extend(_flatten_role_terms(item))
        return terms
    return []


def _classify_role_term(term: str) -> str:
    normalized = term.lower().replace("-", " ")
    tokens = set(normalized.replace(",", " ").replace("/", " ").split())
    if (
        any(phrase in normalized for phrase in (
            "chief executive", "chief financial", "chief operating", "chief technology",
        ))
        or bool(tokens & {"ceo", "cfo", "coo", "cto"})
        or ("president" in tokens and "vice" not in tokens)
    ):
        return "c_suite"
    if any(token in normalized for token in (
        "chief", "officer", "vice president", "vp", "treasurer", "secretary",
    )):
        return "other_officer"
    if "director" in normalized:
        return "director"
    if any(token in normalized for token in ("10%", "ten percent", "beneficial owner", "owner")):
        return "ten_percent_holder"
    return "unknown"


def _role_shadow_metadata(insider_roles: Any, *, scoped_to_opportunistic_buyers: bool) -> Dict[str, Any]:
    tiers = [_classify_role_term(term) for term in _flatten_role_terms(insider_roles)]
    tier_counts = {tier_name: tiers.count(tier_name) for tier_name in ROLE_SHADOW_SENIORITY}
    tier = max(tiers or ["unknown"], key=lambda item: ROLE_SHADOW_SENIORITY[item])
    return {
        "insider_role_tier": tier,
        "insider_role_tier_counts": tier_counts,
        "insider_role_weight_shadow": ROLE_SHADOW_PRIOR_WEIGHTS[tier],
        "insider_role_resolution": "max_seniority",
        "insider_role_scope": "opportunistic_buyers" if scoped_to_opportunistic_buyers else "all_reported_insiders",
    }


def _filing_age_bucket(days_since: Optional[int]) -> str:
    if days_since is None:
        return "missing"
    if days_since < 0:
        return "invalid"
    if days_since <= 2:
        return "0_2d"
    if days_since <= 5:
        return "3_5d"
    if days_since <= 10:
        return "6_10d"
    if days_since <= 20:
        return "11_20d"
    return "gt_20d"


def compute_x_m2(
    days_since_last_filing: int,
    n_distinct_opp_buyers: int,
    mean_trade_intensity_weight: float,
    decay_tau: float = DECAY_TAU,
) -> float:
    """Per EXPOSURE.md: capped at 3.0."""
    if (
        days_since_last_filing < 0
        or n_distinct_opp_buyers < 1
        or mean_trade_intensity_weight <= 0
        or decay_tau <= 0
    ):
        return 0.0
    decay = math.exp(-days_since_last_filing / decay_tau)
    cluster_size = math.log(1 + n_distinct_opp_buyers) / math.log(3)
    raw = decay * cluster_size * mean_trade_intensity_weight
    return min(raw, X_M2_CAP)


def _data_confidence(inp: PatternInput, quality_flags: Dict[str, Any]) -> float:
    return compute_data_confidence(
        quality_flags,
        field_confidence_sources=(inp.market_data, inp.fundamental_data, inp.event_data),
        extra_multipliers={
            "fmp_backfill_authority": 0.90,
            "sec_fmp_mismatch": 0.95,
            "missing_cluster_window_proof": 0.95,
            "cluster_window_mismatch": 0.95,
        },
    )


# ---------------------------------------------------------------------------
# Diagnostic field helpers
# ---------------------------------------------------------------------------

def _copy_diagnostic_fields(feat_dict: Dict[str, Any], *sources: Dict[str, Any]) -> None:
    for source in sources:
        for key in (
            "hazard_score_at_signal", "filing_veto_status",
            "liquidity_score",
            "market_data_status", "halt_status", "corporate_action_filter_passed",
            "market_cap_usd", "market_cap_mm", "sub_universe",
            "adv_20d_dollars", "adv_60d_dollars", "effective_spread_20d_pct", "price_at_signal",
            "d1_decile", "sigma_epsilon_decile", "illiq_decile",
            "cen_hq_county", "cen_sci_score",
            "sector", "industry",
            "insider_roles", "opportunistic_insider_roles", "opportunistic_buy_insider_roles",
            "opportunistic_buyer_roles", "insider_roles_opportunistic",
            "filing_lag_days", "sec_vs_fmp_latency",
            "source_authority", "sec_accession_numbers", "sec_fmp_mismatch",
            "cluster_window_days", "identity_resolution_method", "identity_resolution_confidence",
            "m2_cluster_id", "m2_cluster_signature_hash",
            "exchange", "sector_stress_indicator", "hamilton_regime_prob",
            "last_opp_buy_transaction_date", "last_opp_buy_filing_detected_at", "max_filing_lag_days",
            "cluster_notional_usd", "routine_trades_30d", "unclassifiable_buyers_30d",
            "opportunistic_sell_cluster_30d", "opp_sell_cluster_present",
            "unclassifiable_cluster_intensity",
            "m1_combined_signal_window", "m1_also_firing", "m4_also_firing", "overlapping_pattern_ids",
            "next_earnings_days", "next_earnings_trading_days",
        ):
            val = source.get(key)
            if val is not None:
                feat_dict[key] = val
    feat_dict.setdefault("filing_veto_status", "not_computed")


def _reject_signal(feat_dict: Dict[str, Any], reason: str) -> None:
    feat_dict["rejection_reason"] = reason
    feat_dict["signal_generated"] = False
    feat_dict.setdefault("exposure_x_m2", 0.0)


# ---------------------------------------------------------------------------
# Signal enrichment
# ---------------------------------------------------------------------------

def _enrich_m2_signal(
    feat_dict: Dict[str, Any],
    inp: PatternInput,
    warnings: List[str],
    quality_flags: Dict[str, Any],
    lambda_20td: float,
) -> Optional[PatternSignal]:
    md = inp.market_data

    _copy_diagnostic_fields(feat_dict, inp.market_data, inp.fundamental_data, inp.event_data)

    # Market data quality (EOD M-track; not require_fields)
    pre_signal_rejection = market_data_quality_rejection(feat_dict, md)
    if pre_signal_rejection is not None:
        quality_flags["market_data_quality_rejected"] = True
        _reject_signal(feat_dict, pre_signal_rejection)
        return None

    # --- Load-bearing cluster fields from market_data only ---
    n_opp_buyers = integral_int(md.get("n_distinct_opp_buyers_30d"))
    days_since = integral_int(md.get("days_since_last_opp_buy_filing_detected"))
    mean_trade_size_weight = finite_float(md.get("mean_trade_size_weight"))
    mean_locality_weight = finite_float(md.get("mean_locality_weight"))
    legacy_mean_intensity = finite_float(md.get("mean_trade_intensity_weight"))
    if mean_trade_size_weight is not None and mean_locality_weight is not None:
        mean_intensity = mean_trade_size_weight * mean_locality_weight
    else:
        mean_intensity = legacy_mean_intensity
    n_routine_only = integral_int(md.get("n_routine_only_buyers_30d"))
    n_unclassifiable_only = integral_int(md.get("n_unclassifiable_only_buyers_30d"))
    routine_trades = integral_int(md.get("routine_trades_30d"))
    unclassifiable_buyers = integral_int(md.get("unclassifiable_buyers_30d"))
    role_source = md.get("opportunistic_insider_roles")
    if role_source is None:
        role_source = md.get("opportunistic_buy_insider_roles")
    if role_source is None:
        role_source = md.get("opportunistic_buyer_roles")
    if role_source is None:
        role_source = md.get("insider_roles_opportunistic")
    scoped_role_source = role_source is not None
    if role_source is None:
        role_source = md.get("insider_roles")
    role_meta = _role_shadow_metadata(role_source, scoped_to_opportunistic_buyers=scoped_role_source)
    cluster_window_days = integral_int(md.get("cluster_window_days"))
    source_authority = md.get("source_authority")
    sec_accession_numbers = md.get("sec_accession_numbers")
    sec_fmp_mismatch = md.get("sec_fmp_mismatch")
    accession_proof = _accession_list(sec_accession_numbers)
    m2_cluster_id = md.get("m2_cluster_id")
    m2_cluster_signature_hash = md.get("m2_cluster_signature_hash")
    if m2_cluster_signature_hash is None and accession_proof:
        m2_cluster_signature_hash = stable_hash({
            "ticker": inp.ticker,
            "sec_accession_numbers": sorted(set(accession_proof)),
        })
    if m2_cluster_id is None:
        m2_cluster_id = m2_cluster_signature_hash
    m1_combined_signal_window = md.get("m1_combined_signal_window")
    if m1_combined_signal_window is None and _flag_is_true(md.get("m1_also_firing")):
        m1_combined_signal_window = True
    opportunistic_sell_cluster = md.get("opportunistic_sell_cluster_30d")
    if opportunistic_sell_cluster is None:
        opportunistic_sell_cluster = md.get("opp_sell_cluster_present")

    # Persist raw cluster evidence
    feat_dict["n_distinct_opp_buyers_30d"] = n_opp_buyers
    feat_dict["days_since_last_opp_buy_filing_detected"] = days_since
    feat_dict["cluster_window_days"] = cluster_window_days
    feat_dict["source_authority"] = source_authority
    feat_dict["sec_accession_numbers"] = sec_accession_numbers
    feat_dict["m2_cluster_id"] = m2_cluster_id
    feat_dict["m2_cluster_signature_hash"] = m2_cluster_signature_hash
    if m2_cluster_signature_hash is not None:
        set_signal_identity(
            feat_dict,
            pattern_id=PatternId.M2,
            ticker=inp.ticker,
            components={"sec_accession_numbers": sorted(set(accession_proof))},
            source="sec_accession_cluster",
            identity_hash=m2_cluster_signature_hash,
        )
    feat_dict["sec_fmp_mismatch"] = sec_fmp_mismatch
    feat_dict["mean_trade_size_weight"] = round(mean_trade_size_weight, 6) if mean_trade_size_weight is not None else None
    feat_dict["mean_locality_weight"] = round(mean_locality_weight, 6) if mean_locality_weight is not None else None
    feat_dict["mean_trade_intensity_weight"] = round(mean_intensity, 6) if mean_intensity is not None else None
    feat_dict["n_routine_only_buyers_30d"] = n_routine_only
    feat_dict["n_unclassifiable_only_buyers_30d"] = n_unclassifiable_only
    feat_dict.update(role_meta)
    feat_dict["decay_tau_used"] = DECAY_TAU
    feat_dict["decay_shape_family"] = "exponential"
    feat_dict["filing_age_bucket"] = _filing_age_bucket(days_since)
    feat_dict["routine_trades_30d"] = routine_trades if routine_trades is not None else n_routine_only
    feat_dict["unclassifiable_buyers_30d"] = (
        unclassifiable_buyers if unclassifiable_buyers is not None else n_unclassifiable_only
    )
    if opportunistic_sell_cluster is not None:
        feat_dict["opportunistic_sell_cluster_30d"] = opportunistic_sell_cluster
        feat_dict["opp_sell_cluster_present"] = opportunistic_sell_cluster
    if m1_combined_signal_window is not None:
        feat_dict["m1_combined_signal_window"] = m1_combined_signal_window

    # Validate cluster fields
    if n_opp_buyers is None or days_since is None:
        _reject_signal(feat_dict, "missing_cluster_data")
        return None

    if cluster_window_days is None:
        quality_flags["missing_cluster_window_proof"] = True
        warnings.append("cluster_window_days missing — trusting n_distinct_opp_buyers_30d payload")
    elif cluster_window_days != CLUSTER_WINDOW_DAYS:
        quality_flags["cluster_window_mismatch"] = True
        _reject_signal(feat_dict, "invalid_cluster_window")
        return None

    if source_authority not in LIVE_SOURCE_AUTHORITIES:
        _reject_signal(feat_dict, "invalid_source_authority")
        return None
    if source_authority == "sec_edgar" and not _has_accession_proof(sec_accession_numbers):
        _reject_signal(feat_dict, "missing_sec_accession")
        return None
    if source_authority == "fmp_backfill":
        quality_flags["fmp_backfill_authority"] = True
        if not _has_accession_proof(sec_accession_numbers):
            _reject_signal(feat_dict, "missing_sec_accession")
            return None
    if sec_fmp_mismatch is True:
        quality_flags["sec_fmp_mismatch"] = True

    if mean_intensity is None or mean_intensity <= 0:
        if mean_intensity is not None and mean_intensity <= 0:
            quality_flags["invalid_trade_intensity"] = True
        mean_intensity = MISSING_TRADE_INTENSITY_DEFAULT
        quality_flags["missing_trade_intensity"] = True
        warnings.append(
            f"trade intensity unavailable or invalid — defaulting to {MISSING_TRADE_INTENSITY_DEFAULT}"
        )
        feat_dict["mean_trade_intensity_weight"] = MISSING_TRADE_INTENSITY_DEFAULT

    # Gate: minimum 2 classified opportunistic buyers
    if n_opp_buyers < MIN_OPP_BUYERS:
        if (n_routine_only or 0) > 0 or (n_unclassifiable_only or 0) > 0:
            _reject_signal(feat_dict, "diluted_by_routine_or_unclassifiable")
        else:
            _reject_signal(feat_dict, "no_opportunistic_cluster_present")
        return None

    # Gate: filing freshness (production clock is filing_detected_at, NOT transaction_date)
    if days_since < 0:
        _reject_signal(feat_dict, "invalid_filing_age")
        return None
    if days_since > MAX_DAYS_SINCE_LAST_FILING:
        _reject_signal(feat_dict, "filing_too_stale")
        return None

    # Warn if transaction_date would give a materially different decay
    txn_days_raw = md.get("days_since_last_opp_buy_transaction")
    txn_days = integral_int(txn_days_raw) if txn_days_raw is not None else None
    if txn_days is not None and days_since is not None and abs(txn_days - days_since) > 3:
        quality_flags["large_filing_lag"] = True
        warnings.append(f"filing lag {txn_days - days_since}d between transaction and filing detection")
    feat_dict["days_since_last_opp_buy_transaction"] = txn_days

    # Compute exposure
    sell_cluster_haircut = (
        OPPORTUNISTIC_SELL_CLUSTER_HAIRCUT if _flag_or_positive_count(opportunistic_sell_cluster) else 1.0
    )
    for tau in SHADOW_DECAY_TAUS:
        tau_label = str(int(tau))
        feat_dict[f"decay_multiplier_tau_{tau_label}"] = round(math.exp(-days_since / tau), 6)
        feat_dict[f"x_m2_shadow_tau_{tau_label}"] = round(
            compute_x_m2(days_since, n_opp_buyers, mean_intensity, decay_tau=tau) * sell_cluster_haircut,
            6,
        )

    x_m2 = compute_x_m2(days_since, n_opp_buyers, mean_intensity)
    feat_dict["pre_sell_cluster_exposure_x_m2"] = round(x_m2, 6)
    if sell_cluster_haircut != 1.0:
        quality_flags["opportunistic_sell_cluster_present"] = True
        warnings.append("opportunistic sell cluster present — applying M2 edge haircut")
        feat_dict["opportunistic_sell_cluster_haircut"] = sell_cluster_haircut
        x_m2 *= sell_cluster_haircut
    feat_dict["exposure_x_m2"] = round(x_m2, 6)

    if x_m2 <= 0:
        _reject_signal(feat_dict, "zero_exposure")
        return None

    # Raw expected edge
    raw_expected_edge = round(x_m2 * lambda_20td, 6)
    signal_strength = round(min(x_m2 / X_M2_CAP, 1.0), 6)

    feat_dict["signal_generated"] = True
    feat_dict["lambda_M2_monthly"] = LAMBDA_M2_MONTHLY
    feat_dict["validated_or_shadow_lambda_M2_20td"] = lambda_20td
    feat_dict["lambda_M2_20td"] = round(lambda_20td, 8)
    feat_dict["lambda_M2_default_20td"] = round(LAMBDA_M2_20TD, 8)
    feat_dict["lambda_M2_source"] = (
        "shadow_prior" if lambda_20td == LAMBDA_M2_20TD else "validated_or_injected"
    )
    feat_dict["expected_return_priors"] = {"gross_bps": round(raw_expected_edge * 10_000, 2)}

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

class M2Detector(BasePatternDetector):
    """M2 Insider Cluster detector."""

    pattern_id = PatternId.M2
    version = "1.0"
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.RIGHT_TAIL_CONVEX
    route_class = RouteClass.A

    def __init__(self, lambda_m2_20td: float = LAMBDA_M2_20TD):
        parsed = finite_float(lambda_m2_20td)
        if parsed is None or parsed <= 0:
            raise ValueError("lambda_m2_20td must be finite and positive")
        self._lambda_m2_20td = parsed

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        """Evaluate an M2 insider-cluster setup."""

        asof = require_asof_timestamp(inp.asof_timestamp)
        warnings: List[str] = []
        quality_flags: Dict[str, Any] = {}

        reject_future_timestamp(asof, warnings, quality_flags)
        require_lineage_hash(inp.lineage_hashes, warnings, quality_flags)

        md = inp.market_data

        # Required: cluster evidence must be present
        has_cluster = (
            md.get("n_distinct_opp_buyers_30d") is not None
            and md.get("days_since_last_opp_buy_filing_detected") is not None
        )

        if not has_cluster:
            warnings.append("missing required M2 cluster fields")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        # Validate required numerics are parseable
        n_buyers_check = integral_int(md.get("n_distinct_opp_buyers_30d"))
        days_check = integral_int(md.get("days_since_last_opp_buy_filing_detected"))
        if n_buyers_check is None or days_check is None:
            warnings.append("invalid M2 cluster field values")
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
            feat_dict["n_distinct_opp_buyers_30d"] = n_buyers_check
            feat_dict["days_since_last_opp_buy_filing_detected"] = days_check
            _reject_signal(feat_dict, universe_rejection)
        else:
            sig = _enrich_m2_signal(feat_dict, inp, warnings, quality_flags, self._lambda_m2_20td)
            if sig is not None:
                signals.append(sig)

        features = PatternFeatures(
            features=feat_dict, feature_manifest_version="m2-v1",
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
