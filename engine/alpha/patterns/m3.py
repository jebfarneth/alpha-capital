"""
M3 — Sector Rotation Beneficiary Detector.

Vault source: Engineering/Patterns/M3-SectorRotation/

Thesis: continuation. Stocks in sectors with strong 6-month relative
strength exhibit positive expected excess returns over the following
15 trading days (Moskowitz-Grinblatt 1999 sector momentum premium).

Exposure formula (EXPOSURE.md):
  X_M3 = sector_rank_normalized - 0.5
  raw_expected_edge = X_M3 * lambda_M3_15td

Signal admission:
  1. Operating-universe membership (fail-closed)
  2. Market data quality (shared guard, not require_fields for EOD M-track)
  3. Valid sector identity and sector_return_6mo
  4. sector_rank_normalized >= 0.70 (top-3-decile sectors)
  5. V1 long-only; no bottom-decile short signals

No stock-level multipliers in production edge. D1, sigma_epsilon,
ILLIQ, CEN are diagnostics only.

Routing: Class A (midpoint limit, day-valid).

Evidence: each fired signal persists lambda provenance so shadow
validation can reconstruct the expected-edge assumption.
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
LAMBDA_M3_MONTHLY_BASELINE = 0.0043  # 0.43%/month (M-G 1999 baseline)
LAMBDA_M3_MONTHLY = 0.0075  # 0.75%/month (amplified midpoint)
LAMBDA_M3_15TD = LAMBDA_M3_MONTHLY * 15.0 / 21.0  # ~0.005357
SIGNAL_HORIZON = "15d"
MIN_SECTOR_RANK = 0.70  # top-3-decile eligibility gate


# ---------------------------------------------------------------------------
# Pure computation helpers
# ---------------------------------------------------------------------------

def _valid_sector_identity(value: Any) -> bool:
    """Sector identity is load-bearing for M3; blank strings are missing."""
    return isinstance(value, str) and bool(value.strip())


def compute_sector_rank_normalized(sector_rank: int, n_sectors: int) -> Optional[float]:
    """Normalize sector rank to [0, 1] per EXPOSURE.md: (rank - 0.5) / n_sectors."""
    if n_sectors <= 0:
        return None
    return (sector_rank - 0.5) / n_sectors


def compute_x_m3(sector_rank_normalized: float) -> float:
    """Per EXPOSURE.md: X_M3 = sector_rank_normalized - 0.5."""
    return sector_rank_normalized - 0.5


def _data_confidence(inp: PatternInput, quality_flags: Dict[str, Any]) -> float:
    return compute_data_confidence(
        quality_flags,
        field_confidence_sources=(inp.market_data, inp.fundamental_data, inp.event_data),
    )


# ---------------------------------------------------------------------------
# Diagnostic field helpers
# ---------------------------------------------------------------------------

def _copy_diagnostic_fields(
    feat_dict: Dict[str, Any],
    market_data: Dict[str, Any],
    fundamental_data: Optional[Dict[str, Any]] = None,
    event_data: Optional[Dict[str, Any]] = None,
) -> None:
    market_only_keys = (
        "sector", "industry", "sector_taxonomy_source",
        "sector_return_6mo", "sector_return_6mo_ew",
        "sector_return_point_in_time_passed", "sector_return_formation_cohort_passed",
        "sector_history_coverage_years",
        "return_1mo", "return_3mo",
        "sector_rank", "n_sectors_in_universe", "sector_rank_normalized",
        "market_data_status", "halt_status", "corporate_action_filter_passed",
    )
    for key in market_only_keys:
        val = market_data.get(key)
        if val is not None:
            if key == "sector" and isinstance(val, str):
                val = val.strip()
                if not val:
                    continue
            feat_dict.setdefault(key, val)

    for source in (market_data, fundamental_data or {}, event_data or {}):
        for key in (
            "hazard_score_at_signal", "filing_veto_status",
            "liquidity_score",
            "market_cap_usd", "market_cap_mm", "sub_universe",
            "adv_20d_dollars", "adv_60d_dollars", "effective_spread_20d_pct", "price_at_signal",
            "d1_decile", "sigma_epsilon_decile", "illiq_decile",
            "cen_hq_county", "cen_sci_score",
            "sector_positive_earnings_surprises", "sector_upward_analyst_revisions",
            "sector_positive_material_news", "sector_total_confirmation_count",
            "sector_rank_snapshot_id", "sector_return_snapshot_id",
            "sector_return_formation_id", "sector_rank_formation_id",
            "m4_also_firing", "m6_also_firing", "overlapping_pattern_ids",
        ):
            val = source.get(key)
            if val is not None:
                feat_dict[key] = val
    feat_dict.setdefault("filing_veto_status", "not_computed")
    feat_dict.setdefault("sector_taxonomy_source", "FMP")


def _set_m3_signal_identity(feat_dict: Dict[str, Any], inp: PatternInput) -> None:
    snapshot_id = (
        feat_dict.get("sector_rank_snapshot_id")
        or feat_dict.get("sector_return_snapshot_id")
        or feat_dict.get("sector_return_formation_id")
        or feat_dict.get("sector_rank_formation_id")
    )
    if snapshot_id is not None:
        components = {"sector_rank_snapshot_id": snapshot_id}
        source = "upstream_sector_rank_snapshot"
    else:
        components = {
            "sector": feat_dict.get("sector"),
            "sector_rank_normalized": feat_dict.get("sector_rank_normalized"),
            "sector_return_6mo": feat_dict.get("sector_return_6mo"),
            "asof_date": inp.asof_timestamp.date().isoformat()
            if inp.asof_timestamp is not None
            else None,
        }
        source = "sector_rank_features"
    set_signal_identity(
        feat_dict,
        pattern_id=PatternId.M3,
        ticker=inp.ticker,
        components=components,
        source=source,
    )


def _reject_signal(feat_dict: Dict[str, Any], reason: str) -> None:
    feat_dict["rejection_reason"] = reason
    feat_dict["signal_generated"] = False
    feat_dict.setdefault("exposure_x_m3_t0", 0.0)


# ---------------------------------------------------------------------------
# Signal enrichment
# ---------------------------------------------------------------------------

def _enrich_m3_signal(
    feat_dict: Dict[str, Any],
    inp: PatternInput,
    warnings: List[str],
    quality_flags: Dict[str, Any],
    lambda_15td: float,
) -> Optional[PatternSignal]:
    md = inp.market_data

    _copy_diagnostic_fields(feat_dict, inp.market_data, inp.fundamental_data, inp.event_data)
    _set_m3_signal_identity(feat_dict, inp)

    # Market data quality (EOD M-track; not require_fields)
    pre_signal_rejection = market_data_quality_rejection(feat_dict, md)
    if pre_signal_rejection is not None:
        quality_flags["market_data_quality_rejected"] = True
        _reject_signal(feat_dict, pre_signal_rejection)
        return None

    # Resolve sector_rank_normalized
    rank_norm_raw = md.get("sector_rank_normalized")
    sector_rank_raw = md.get("sector_rank")
    n_sectors_raw = md.get("n_sectors_in_universe")
    sector_rank = integral_int(sector_rank_raw)
    n_sectors = integral_int(n_sectors_raw)
    computed_rank_norm = None
    if sector_rank is not None and n_sectors is not None and n_sectors > 0:
        computed_rank_norm = compute_sector_rank_normalized(sector_rank, n_sectors)

    if rank_norm_raw is not None:
        rank_norm = finite_float(rank_norm_raw)
    else:
        if computed_rank_norm is None:
            rank_norm = None
        else:
            rank_norm = computed_rank_norm

    if rank_norm is None or not (0.0 <= rank_norm <= 1.0):
        _reject_signal(feat_dict, "invalid_sector_rank")
        return None

    if (
        rank_norm_raw is not None
        and computed_rank_norm is not None
        and abs(rank_norm - computed_rank_norm) > 1e-3
    ):
        quality_flags["inconsistent_sector_rank"] = True
        warnings.append("sector_rank_normalized conflicts with sector_rank/n_sectors_in_universe")
        _reject_signal(feat_dict, "inconsistent_sector_rank")
        return None

    feat_dict["sector_rank_normalized"] = round(rank_norm, 6)

    # Compute exposure
    x_m3 = compute_x_m3(rank_norm)
    feat_dict["exposure_x_m3_t0"] = round(x_m3, 6)

    # Top-3-decile gate
    if rank_norm < MIN_SECTOR_RANK:
        _reject_signal(feat_dict, "sector_rank_below_threshold")
        return None

    # Tier classification (audit metadata only — does NOT modify edge)
    d1_decile_raw = feat_dict.get("d1_decile")
    d1_decile = integral_int(d1_decile_raw) if d1_decile_raw is not None else None
    if d1_decile is not None and not (1 <= d1_decile <= 10):
        quality_flags["invalid_d1_decile"] = True
        warnings.append("d1_decile out of range — defaulting tier to default")
        d1_decile = None
    if d1_decile == 10 and rank_norm >= 0.80:
        tier = "high_conviction"
    else:
        tier = "default"
    feat_dict["expected_return_tier"] = tier
    feat_dict["expected_return_priors_audit_metadata"] = {
        "tier_default_bps_range": [30, 60],
        "tier_high_conviction_bps_range": [60, 100],
        "note": "Tier priors are audit metadata only; they do NOT enter raw_expected_edge math.",
    }

    # Raw expected edge
    raw_expected_edge = round(x_m3 * lambda_15td, 6)

    signal_strength = round(rank_norm, 6)

    feat_dict["signal_generated"] = True
    feat_dict["lambda_M3_monthly"] = LAMBDA_M3_MONTHLY
    feat_dict["lambda_M3_monthly_baseline"] = LAMBDA_M3_MONTHLY_BASELINE
    feat_dict["validated_or_shadow_lambda_M3_15td"] = lambda_15td
    feat_dict["lambda_M3_15td"] = round(lambda_15td, 8)
    feat_dict["lambda_M3_default_15td"] = round(LAMBDA_M3_15TD, 8)
    feat_dict["lambda_M3_source"] = (
        "shadow_prior" if lambda_15td == LAMBDA_M3_15TD else "validated_or_injected"
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

class M3Detector(BasePatternDetector):
    """M3 Sector Rotation Beneficiary detector."""

    pattern_id = PatternId.M3
    version = "1.0"
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.CONTINUATION
    route_class = RouteClass.A

    def __init__(self, lambda_m3_15td: float = LAMBDA_M3_15TD):
        parsed = finite_float(lambda_m3_15td)
        if parsed is None or parsed <= 0:
            raise ValueError("lambda_m3_15td must be finite and positive")
        self._lambda_m3_15td = parsed

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        asof = require_asof_timestamp(inp.asof_timestamp)
        warnings: List[str] = []
        quality_flags: Dict[str, Any] = {}

        reject_future_timestamp(asof, warnings, quality_flags)
        require_lineage_hash(inp.lineage_hashes, warnings, quality_flags)

        md = inp.market_data

        # Required: sector identity and sector return
        sector = md.get("sector")
        sector_return_raw = md.get("sector_return_6mo")
        has_rank_evidence = (
            md.get("sector_rank_normalized") is not None
            or (md.get("sector_rank") is not None and md.get("n_sectors_in_universe") is not None)
        )

        if not _valid_sector_identity(sector) or sector_return_raw is None or not has_rank_evidence:
            warnings.append("missing required M3 fields (sector, sector_return_6mo, or rank evidence)")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        sector_return = finite_float(sector_return_raw)
        if sector_return is None:
            warnings.append("invalid sector_return_6mo")
            return self._no_features_result(inp.ticker, asof, warnings, quality_flags)

        feat_dict: Dict[str, Any] = {
            "sector": sector.strip(),
            "sector_return_6mo": round(sector_return, 6),
        }

        universe_rejection = operating_universe_rejection(
            md, warnings, quality_flags, pattern_id=self.pattern_id,
        )

        signals: List[PatternSignal] = []

        if universe_rejection is not None:
            _copy_diagnostic_fields(feat_dict, inp.market_data, inp.fundamental_data, inp.event_data)
            _reject_signal(feat_dict, universe_rejection)
        elif (
            md.get("sector_return_point_in_time_passed") is not True
            or md.get("sector_return_formation_cohort_passed") is False
        ):
            quality_flags["point_in_time_passed"] = False
            if md.get("sector_return_formation_cohort_passed") is False:
                quality_flags["formation_cohort_passed"] = False
            warnings.append("M3 sector return missing point-in-time formation-cohort proof")
            _copy_diagnostic_fields(feat_dict, inp.market_data, inp.fundamental_data, inp.event_data)
            _reject_signal(feat_dict, "sector_return_not_point_in_time")
        else:
            sig = _enrich_m3_signal(feat_dict, inp, warnings, quality_flags, self._lambda_m3_15td)
            if sig is not None:
                signals.append(sig)

        pit_passed = quality_flags.get("point_in_time_passed") is not False
        fidelity = classify_fidelity(
            has_primary_data=True, has_secondary_data=True,
            point_in_time_passed=pit_passed, lookahead_guard_passed=True,
        )

        features = PatternFeatures(
            features=feat_dict, feature_manifest_version="m3-v1",
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
