"""
M4 52-Week High Breakout detector tests.

Vault contract verification (amended):
  - All breakouts P >= H52w emit signals (including exact-high closes)
  - Cohort rank / top3_decile_flag are metadata, not admission gates
  - Missing cohort data does NOT block signal; adds quality flag
  - below-high and non-operating-universe are true no-signal cases
  - Fresh-breakout activation lane intact with strong assertions
  - Evidence bridge writes with correct FK chain and vault fields
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from alpha.data.contracts import stable_hash
from alpha.db.models import FeatureSnapshot, SignalRegistry
from alpha.evidence.writer import create_job, record_data_lineage, record_universe_snapshot, start_run
from alpha.patterns.contracts import (
    FidelityTier,
    PatternId,
    PatternInput,
    PatternTrack,
    RouteClass,
    SignalDirection,
    ThesisCategory,
)
from alpha.patterns.evidence_bridge import persist_detection_result
from alpha.patterns.m4 import (
    KAPPA,
    LAMBDA_M4_15TD,
    LAMBDA_M4_MONTHLY,
    M4Detector,
    MIN_BASE_DAILY_SIGNAL_SESSIONS,
    SHORT_HISTORY_BELOW_SIGNAL_FLOOR_REASON,
    X_M4_CAP,
    _classify_extension_tier,
    compute_m4_features,
    compute_cohort_metadata,
)


def _ts():
    return datetime(2026, 5, 20, 21, 0, 0, tzinfo=timezone.utc)


def _setup_run(db_session):
    job = create_job(db_session, name="m4_detector", job_type="detector", owner="pattern_engine")
    run = start_run(db_session, job_id=job.job_id)
    db_session.flush()
    return run


def _cohort_extensions(n=30, top_val=0.12):
    return [round(i * top_val / (n - 1), 4) for i in range(n)]


def _m4_base_data(**overrides):
    data = {"price": 11.00, "high_52w": 10.00, "operating_universe_inclusion": True}
    data.update(overrides)
    return data


def _m4_fresh_quote_fields():
    return {
        "candidate_eval_bid": 10.48,
        "candidate_eval_ask": 10.50,
        "candidate_eval_quote_timestamp": "2026-05-20T15:00:00Z",
        "quote_age_ms": 650,
        "quote_freshness_max_ms": 1000,
    }


def _m4_fresh_activation_fields():
    return {
        "activation_id": "m4-act-ACME-20260520-150000",
        "activation_timestamp": "2026-05-20T15:00:00Z",
        "signal_freshness_passed": True,
        "market_data_status": "current",
        "halt_status": "clear",
        "corporate_action_filter_passed": True,
        **_m4_fresh_quote_fields(),
    }


# -----------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------

class TestM4Metadata:
    def test_pattern_id(self):
        assert M4Detector().pattern_id == PatternId.M4

    def test_track(self):
        assert M4Detector().track == PatternTrack.MULTI_DAY

    def test_thesis_category(self):
        assert M4Detector().thesis_category == ThesisCategory.RIGHT_TAIL_CONVEX

    def test_route_class(self):
        assert M4Detector().route_class == RouteClass.A

    def test_vault_constants(self):
        assert KAPPA == 1.0
        assert X_M4_CAP == 1.5
        assert LAMBDA_M4_MONTHLY == 0.011
        assert abs(LAMBDA_M4_15TD - 0.011 * 15 / 21) < 1e-10


# -----------------------------------------------------------------------
# Exposure formula
# -----------------------------------------------------------------------

class TestExposureFormula:
    def test_below_high(self):
        feat = compute_m4_features(price=9.20, high_52w=10.00)
        assert feat["base_nearness"] == 0.92
        assert feat["breakout_extension"] == 0.0
        assert feat["X_M4"] == 0.92

    def test_at_high(self):
        feat = compute_m4_features(price=10.00, high_52w=10.00)
        assert feat["base_nearness"] == 1.0
        assert feat["breakout_extension"] == 0.0
        assert feat["X_M4"] == 1.0

    def test_above_high(self):
        feat = compute_m4_features(price=10.50, high_52w=10.00)
        assert feat["base_nearness"] == 1.0
        assert feat["breakout_extension"] == 0.05
        assert feat["X_M4"] == 1.05

    def test_cap_at_1_5(self):
        assert compute_m4_features(price=16.00, high_52w=10.00)["X_M4"] == X_M4_CAP

    def test_numerical_example_from_spec(self):
        feat = compute_m4_features(price=12.00, high_52w=10.00)
        assert feat["breakout_extension"] == 0.2
        assert feat["X_M4"] == 1.2


# -----------------------------------------------------------------------
# Cohort metadata (no longer a gate)
# -----------------------------------------------------------------------

class TestCohortMetadata:
    def test_top_decile_flagged(self):
        meta = compute_cohort_metadata(0.12, _cohort_extensions(30, 0.12))
        assert meta["top3_decile_flag"] is True
        assert meta["breakout_cohort_size"] == 30
        assert meta["breakout_cohort_rank"] is not None
        assert meta["breakout_cohort_percentile"] is not None
        assert "cohort_threshold_70p" in meta

    def test_bottom_decile_not_flagged(self):
        meta = compute_cohort_metadata(0.01, _cohort_extensions(30, 0.12))
        assert meta["top3_decile_flag"] is False

    def test_zero_extension_not_flagged(self):
        meta = compute_cohort_metadata(0.0, _cohort_extensions(30, 0.12))
        assert meta["top3_decile_flag"] is False

    def test_small_cohort_all_flagged(self):
        meta = compute_cohort_metadata(0.05, [0.02, 0.05, 0.08])
        assert meta["top3_decile_flag"] is True
        assert meta["small_cohort_warning"] is True

    def test_cohort_threshold_70p_value(self):
        """Deterministic cohort: 10 extensions 0.00..0.09 -> 70th pctl at index 6.3."""
        exts = [round(i * 0.01, 2) for i in range(10)]
        meta = compute_cohort_metadata(0.05, exts)
        # index = 0.70 * 9 = 6.3 -> interp(0.06, 0.07, 0.3) = 0.063
        assert meta["cohort_threshold_70p"] == 0.063

    def test_extension_tier_uses_interpolated_p75(self):
        exts = [round(i * 0.01, 2) for i in range(10)]
        assert _classify_extension_tier(0.065, exts) == "default"
        assert _classify_extension_tier(0.068, exts) == "high_conviction"

    def test_dict_cohort_tie_break_ordering(self):
        """Vault tie-break: higher dollar volume, earlier timestamp, ticker."""
        cohort = [
            {"ticker": "ZZZ", "breakout_extension": 0.10, "median_dollar_volume_20d": 1_000, "signal_timestamp": "2026-05-20T20:05:00Z"},
            {"ticker": "ACME", "breakout_extension": 0.10, "median_dollar_volume_20d": 2_000, "signal_timestamp": "2026-05-20T20:04:00Z"},
            {"ticker": "LOW1", "breakout_extension": 0.01},
            {"ticker": "LOW2", "breakout_extension": 0.02},
            {"ticker": "LOW3", "breakout_extension": 0.03},
            {"ticker": "LOW4", "breakout_extension": 0.04},
            {"ticker": "LOW5", "breakout_extension": 0.05},
            {"ticker": "LOW6", "breakout_extension": 0.06},
            {"ticker": "LOW7", "breakout_extension": 0.07},
            {"ticker": "LOW8", "breakout_extension": 0.08},
        ]
        meta = compute_cohort_metadata(0.10, cohort, ticker="ACME")
        assert meta["breakout_cohort_rank"] == 1  # ACME wins: higher volume

    def test_small_cohort_metadata_on_signal_features(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=10.50, high_52w=10.00, cohort_extensions=[0.05, 0.03]),
            lineage_hashes=["h"],
        )
        result = det.detect(inp)
        assert result.has_signal
        f = result.features.features
        assert f["small_cohort_warning"] is True
        assert f["top3_decile_flag"] is True

    def test_bottom_decile_emits_signal_with_metadata(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=10.01, high_52w=10.00, cohort_extensions=_cohort_extensions(30, 0.15)),
            lineage_hashes=["h"],
        )
        result = det.detect(inp)
        assert result.has_signal
        f = result.features.features
        assert f["top3_decile_flag"] is False
        assert f["breakout_cohort_rank"] is not None
        assert f["extension_tier"] == "default"


# -----------------------------------------------------------------------
# Firing cases: all breakouts emit signals
# -----------------------------------------------------------------------

class TestM4Firing:
    def test_extended_breakout_fires(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=11.50, high_52w=10.00, cohort_extensions=_cohort_extensions(30, 0.15)),
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.has_signal
        assert result.signals[0].direction == SignalDirection.LONG
        assert result.signals[0].signal_horizon == "15d"
        assert result.signals[0].route_class == RouteClass.A
        assert result.features.features["extension_tier"] in {"default", "high_conviction"}

    def test_exact_high_fires(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=10.00, high_52w=10.00, cohort_extensions=_cohort_extensions(30, 0.10)),
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.has_signal
        sig = result.signals[0]
        assert sig.raw_expected_edge == round(1.0 * LAMBDA_M4_15TD, 6)
        assert sig.raw_signal_strength == round(1.0 / X_M4_CAP, 6)
        f = result.features.features
        assert f["extension_tier"] == "exact_high"
        assert "tier_classification" not in f
        assert f["breakout_extension"] == 0.0
        assert f["X_M4"] == 1.0

    def test_missing_cohort_still_fires(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=11.00, high_52w=10.00),
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.has_signal
        assert result.quality_flags.get("cohort_metadata_unavailable") is True
        assert any("cohort" in w for w in result.warnings)

    def test_raw_expected_edge_deterministic(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=11.50, high_52w=10.00, cohort_extensions=_cohort_extensions(30, 0.15)),
            lineage_hashes=["hash1"],
        )
        r1 = det.detect(inp)
        r2 = det.detect(inp)
        expected = round(1.15 * LAMBDA_M4_15TD, 6)
        assert r1.signals[0].raw_expected_edge == expected
        assert r1.signals[0].raw_expected_edge == r2.signals[0].raw_expected_edge

    def test_signal_strength_is_x_over_cap(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=11.50, high_52w=10.00, cohort_extensions=_cohort_extensions(30, 0.15)),
            lineage_hashes=["hash1"],
        )
        assert det.detect(inp).signals[0].raw_signal_strength == round(1.15 / 1.5, 6)

    def test_data_confidence_default_1_0(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=11.50, high_52w=10.00, cohort_extensions=_cohort_extensions(30, 0.15)),
            lineage_hashes=["hash1"],
        )
        assert det.detect(inp).signals[0].data_confidence == 1.0

    def test_expected_return_priors_logged(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=11.50, high_52w=10.00, cohort_extensions=_cohort_extensions(30, 0.15)),
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        priors = result.features.features["expected_return_priors"]
        assert priors["gross_bps"] == round(result.signals[0].raw_expected_edge * 10_000, 2)
        assert result.features.features["lambda_M4_monthly"] == LAMBDA_M4_MONTHLY
        assert result.features.features["lambda_M4_15td"] == round(LAMBDA_M4_15TD, 8)

    def test_diagnostic_source_features_logged(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={
                "price": 11.50, "high_52w": 10.00, "cohort_extensions": _cohort_extensions(30, 0.15),
                "operating_universe_inclusion": True,
                "D1_decile": 8, "R_6_12m_skip": 0.234,
                "hamilton_regime_prob": 0.72, "hazard_score_at_signal": 22,
            },
            fundamental_data={"market_cap": 95_400_000, "sector": "Technology", "industry": "Software - Application", "analyst_count": 2},
            event_data={"filing_veto_status": "clear"},
            lineage_hashes=["hash1"],
        )
        f = det.detect(inp).features.features
        assert f["D1_decile"] == 8
        assert f["R_6_12m_skip"] == 0.234
        assert f["analyst_count"] == 2
        assert f["sector"] == "Technology"

    def test_field_confidence_product(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=11.00, high_52w=10.00, cohort_extensions=[0.10],
                                      field_confidence={"adj_close": 0.95, "high_52w": 0.90}),
            lineage_hashes=["hash1"],
        )
        assert det.detect(inp).signals[0].data_confidence == 0.855

    def test_short_history_at_signal_floor_can_fire_with_flag(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(
                price=11.00,
                high_52w=10.00,
                n_sessions_in_window=MIN_BASE_DAILY_SIGNAL_SESSIONS,
                cohort_extensions=_cohort_extensions(30, 0.15),
            ),
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.has_signal
        f = result.features.features
        assert f["short_history_flag"] is True
        assert f["short_history_below_signal_floor"] is False
        assert f["min_signal_sessions"] == MIN_BASE_DAILY_SIGNAL_SESSIONS


# -----------------------------------------------------------------------
# Fresh-breakout lane
# -----------------------------------------------------------------------

class TestM4FreshBreakout:
    def test_fresh_watchlist_signal(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(entry_lane="fresh_breakout_activation", activation_state="watchlist",
                                      price=9.80, high_52w=10.00),
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.has_signal
        sig = result.signals[0]
        assert sig.signal_status == "watchlist"
        assert sig.route_class == RouteClass.C
        assert sig.raw_expected_edge == 0.0
        assert result.features.features["watchlist_passed"] is True
        assert result.features.features["signal_generated"] is True

    def test_fresh_activation_signal(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={
                "entry_lane": "fresh_breakout_activation", "activation_state": "activated",
                "price": 9.90, "last_price": 10.50, "high_52w": 10.00,
                "intraday_range_confirmation": 1.25, "intraday_volume_confirmation": 1.50,
                "spread_pct_vs_eval_quote": 0.005,
                "operating_universe_inclusion": True,
                **_m4_fresh_activation_fields(),
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.has_signal
        sig = result.signals[0]
        assert sig.signal_status == "active"
        assert sig.route_class == RouteClass.C
        assert sig.raw_expected_edge > 0
        f = result.features.features
        assert f["activation_passed"] is True
        assert f["signal_generated"] is True
        assert f["activation_identity_passed"] is True
        assert f["activation_id"] == "m4-act-ACME-20260520-150000"
        assert f["activation_timestamp"] == "2026-05-20T15:00:00Z"
        assert f["signal_freshness_passed"] is True
        assert f["fresh_breakout_extension"] == 0.05
        assert f["lambda_M4_monthly"] == LAMBDA_M4_MONTHLY
        assert f["lambda_M4_15td"] == round(LAMBDA_M4_15TD, 8)
        assert f["expected_return_priors"]["entry_lane"] == "fresh_breakout_activation"

    def test_fresh_activation_spread_failure(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={
                "entry_lane": "fresh_breakout_activation", "activation_state": "activated",
                "price": 9.90, "last_price": 10.50, "high_52w": 10.00,
                "intraday_range_confirmation": 1.25, "intraday_volume_confirmation": 1.50,
                "spread_pct_vs_eval_quote": 0.02,
                "operating_universe_inclusion": True,
                **_m4_fresh_activation_fields(),
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        f = result.features.features
        assert f["activation_passed"] is False
        assert f["activation_failure_reason"] == "spread_too_wide"
        assert f["rejection_reason"] == "spread_too_wide"
        assert f["signal_generated"] is False

    def test_fresh_activation_missing_spread_is_unavailable_not_too_wide(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={
                "entry_lane": "fresh_breakout_activation", "activation_state": "activated",
                "price": 9.90, "last_price": 10.50, "high_52w": 10.00,
                "intraday_range_confirmation": 1.25, "intraday_volume_confirmation": 1.50,
                "operating_universe_inclusion": True,
                **_m4_fresh_activation_fields(),
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        f = result.features.features
        assert f["spread_pct_vs_eval_quote"] is None
        assert f["spread_discipline_passed"] is False
        assert f["activation_failure_reason"] == "spread_unavailable"
        assert f["rejection_reason"] == "spread_unavailable"
        assert f["signal_generated"] is False

    def test_fresh_activation_truthy_string_spread_flag_fails_closed(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={
                "entry_lane": "fresh_breakout_activation", "activation_state": "activated",
                "price": 9.90, "last_price": 10.50, "high_52w": 10.00,
                "intraday_range_confirmation": 1.25, "intraday_volume_confirmation": 1.50,
                "spread_pct_vs_eval_quote": 0.005,
                "spread_discipline_passed": "true",
                "operating_universe_inclusion": True,
                **_m4_fresh_activation_fields(),
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        f = result.features.features
        assert f["spread_discipline_passed"] is False
        assert f["activation_failure_reason"] == "spread_too_wide"
        assert f["rejection_reason"] == "spread_too_wide"
        assert f["signal_generated"] is False

    def test_fresh_activation_requires_live_market_data_quality_fields(self):
        det = M4Detector()
        fields = _m4_fresh_activation_fields()
        fields.pop("market_data_status")
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={
                "entry_lane": "fresh_breakout_activation", "activation_state": "activated",
                "price": 9.90, "last_price": 10.50, "high_52w": 10.00,
                "intraday_range_confirmation": 1.25, "intraday_volume_confirmation": 1.50,
                "spread_pct_vs_eval_quote": 0.005,
                "operating_universe_inclusion": True,
                **fields,
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        f = result.features.features
        assert f["activation_passed"] is False
        assert f["activation_failure_reason"] == "missing_market_data_quality"
        assert f["rejection_reason"] == "missing_market_data_quality"
        assert f["signal_generated"] is False

    def test_fresh_activation_missing_freshness_fails_closed(self):
        det = M4Detector()
        fields = _m4_fresh_activation_fields()
        fields.pop("signal_freshness_passed")
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={
                "entry_lane": "fresh_breakout_activation", "activation_state": "activated",
                "price": 9.90, "last_price": 10.50, "high_52w": 10.00,
                "intraday_range_confirmation": 1.25, "intraday_volume_confirmation": 1.50,
                "spread_pct_vs_eval_quote": 0.005,
                "operating_universe_inclusion": True,
                **fields,
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        f = result.features.features
        assert f["activation_passed"] is False
        assert f["signal_freshness_passed"] is False
        assert f["activation_failure_reason"] == "signal_expired"
        assert f["rejection_reason"] == "signal_expired"
        assert f["signal_generated"] is False

    def test_fresh_activation_false_freshness_fails_closed(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={
                "entry_lane": "fresh_breakout_activation", "activation_state": "activated",
                "price": 9.90, "last_price": 10.50, "high_52w": 10.00,
                "intraday_range_confirmation": 1.25, "intraday_volume_confirmation": 1.50,
                "spread_pct_vs_eval_quote": 0.005,
                "operating_universe_inclusion": True,
                **_m4_fresh_activation_fields(),
                "signal_freshness_passed": False,
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        f = result.features.features
        assert f["activation_passed"] is False
        assert f["signal_freshness_passed"] is False
        assert f["activation_failure_reason"] == "signal_expired"
        assert f["rejection_reason"] == "signal_expired"
        assert f["signal_generated"] is False

    def test_fresh_activation_truthy_string_freshness_fails_closed(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={
                "entry_lane": "fresh_breakout_activation", "activation_state": "activated",
                "price": 9.90, "last_price": 10.50, "high_52w": 10.00,
                "intraday_range_confirmation": 1.25, "intraday_volume_confirmation": 1.50,
                "spread_pct_vs_eval_quote": 0.005,
                "operating_universe_inclusion": True,
                **_m4_fresh_activation_fields(),
                "signal_freshness_passed": "true",
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        f = result.features.features
        assert f["activation_passed"] is False
        assert f["signal_freshness_passed"] is False
        assert f["activation_failure_reason"] == "signal_expired"
        assert f["signal_generated"] is False

    def test_fresh_activation_missing_activation_identity_fails_closed(self):
        det = M4Detector()
        fields = _m4_fresh_activation_fields()
        fields.pop("activation_id")
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={
                "entry_lane": "fresh_breakout_activation", "activation_state": "activated",
                "price": 9.90, "last_price": 10.50, "high_52w": 10.00,
                "intraday_range_confirmation": 1.25, "intraday_volume_confirmation": 1.50,
                "spread_pct_vs_eval_quote": 0.005,
                "operating_universe_inclusion": True,
                **fields,
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        f = result.features.features
        assert f["activation_passed"] is False
        assert f["activation_identity_passed"] is False
        assert f["activation_failure_reason"] == "activation_identity_missing"
        assert f["rejection_reason"] == "activation_identity_missing"
        assert f["signal_generated"] is False

    def test_fresh_activation_missing_activation_timestamp_fails_closed(self):
        det = M4Detector()
        fields = _m4_fresh_activation_fields()
        fields.pop("activation_timestamp")
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={
                "entry_lane": "fresh_breakout_activation", "activation_state": "activated",
                "price": 9.90, "last_price": 10.50, "high_52w": 10.00,
                "intraday_range_confirmation": 1.25, "intraday_volume_confirmation": 1.50,
                "spread_pct_vs_eval_quote": 0.005,
                "operating_universe_inclusion": True,
                **fields,
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        f = result.features.features
        assert f["activation_passed"] is False
        assert f["activation_identity_passed"] is False
        assert f["activation_failure_reason"] == "activation_identity_missing"
        assert f["rejection_reason"] == "activation_identity_missing"
        assert f["signal_generated"] is False

    def test_fresh_watchlist_respects_operating_universe_exclusion(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={"entry_lane": "fresh_breakout_activation", "activation_state": "watchlist",
                         "price": 9.80, "high_52w": 10.00, "operating_universe_inclusion": False},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        assert result.quality_flags["not_operating_universe_member"] is True
        assert "watchlist_passed" not in result.features.features

    def test_fresh_activation_respects_operating_universe_exclusion(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={
                "entry_lane": "fresh_breakout_activation", "activation_state": "activated",
                "price": 9.90, "last_price": 10.50, "high_52w": 10.00,
                "intraday_range_confirmation": 1.25, "intraday_volume_confirmation": 1.50,
                "spread_pct_vs_eval_quote": 0.005,
                "operating_universe_inclusion": False,
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        assert result.quality_flags["not_operating_universe_member"] is True
        assert "activation_passed" not in result.features.features


# -----------------------------------------------------------------------
# True no-signal cases
# -----------------------------------------------------------------------

class TestM4NoSignal:
    def test_below_high(self):
        det = M4Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(),
                           market_data=_m4_base_data(price=9.50, high_52w=10.00), lineage_hashes=["h"])
        result = det.detect(inp)
        assert not result.has_signal
        assert result.features is not None
        assert result.features.features["X_M4"] == 0.95
        assert result.features.features["signal_generated"] is False
        assert result.features.features["rejection_reason"] == "below_high"

    def test_missing_price(self):
        det = M4Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(),
                           market_data={"high_52w": 10.00}, lineage_hashes=["h"])
        result = det.detect(inp)
        assert not result.has_signal
        assert result.features is None

    def test_non_finite_price_returns_no_features(self):
        det = M4Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(),
                           market_data=_m4_base_data(price=float("nan")), lineage_hashes=["h"])
        result = det.detect(inp)
        assert not result.has_signal
        assert result.features is None

    def test_not_operating_universe(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={"price": 11.00, "high_52w": 10.00, "cohort_extensions": [0.10],
                         "operating_universe_inclusion": False},
            lineage_hashes=["h"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        assert result.quality_flags["not_operating_universe_member"] is True

    def test_missing_operating_universe_fails_closed(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={"price": 11.00, "high_52w": 10.00},
            lineage_hashes=["h"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        assert result.quality_flags["operating_universe_not_computed"] is True
        assert result.features.features["rejection_reason"] == "missing_operating_universe"

    def test_base_daily_short_history_below_floor_does_not_fire(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_m4_base_data(
                price=11.00,
                high_52w=10.00,
                n_sessions_in_window=3,
                cohort_extensions=_cohort_extensions(30, 0.15),
            ),
            lineage_hashes=["h"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        assert result.quality_flags["short_history_below_signal_floor"] is True
        f = result.features.features
        assert f["signal_generated"] is False
        assert f["short_history_flag"] is True
        assert f["short_history_below_signal_floor"] is True
        assert f["min_signal_sessions"] == MIN_BASE_DAILY_SIGNAL_SESSIONS
        assert f["rejection_reason"] == SHORT_HISTORY_BELOW_SIGNAL_FLOOR_REASON


# -----------------------------------------------------------------------
# Fidelity
# -----------------------------------------------------------------------

class TestM4Fidelity:
    def test_always_full(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=11.00, high_52w=10.00, n_sessions_in_window=100, cohort_extensions=[0.10]),
            lineage_hashes=["h"],
        )
        result = det.detect(inp)
        assert result.features.fidelity_tier == FidelityTier.FULL
        assert result.features.features["short_history_flag"] is True


# -----------------------------------------------------------------------
# Guards
# -----------------------------------------------------------------------

class TestM4Guards:
    def test_missing_lineage_warning(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=11.00, high_52w=10.00, cohort_extensions=[0.10]),
            lineage_hashes=[],
        )
        result = det.detect(inp)
        assert result.quality_flags.get("missing_lineage") is True
        assert result.has_signal

    def test_future_timestamp_warning(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=datetime(2099, 1, 1, tzinfo=timezone.utc),
            market_data=_m4_base_data(price=11.00, high_52w=10.00, cohort_extensions=[0.10]),
            lineage_hashes=["h"],
        )
        assert det.detect(inp).quality_flags.get("future_timestamp") is True


# -----------------------------------------------------------------------
# Hashes
# -----------------------------------------------------------------------

class TestM4Hashes:
    def test_stable(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=11.00, high_52w=10.00, cohort_extensions=[0.10]),
            lineage_hashes=["hash1"],
        )
        r1, r2 = det.detect(inp), det.detect(inp)
        assert r1.input_hashes == r2.input_hashes
        assert r1.output_hashes == r2.output_hashes

    def test_change_with_price(self):
        det = M4Detector()
        r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_m4_base_data(price=11.00, high_52w=10.00), lineage_hashes=["h"]))
        r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_m4_base_data(price=12.00, high_52w=10.00), lineage_hashes=["h"]))
        assert r1.output_hashes != r2.output_hashes

    def test_output_hash_matches_final_state(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=11.00, high_52w=10.00, cohort_extensions=[0.10]),
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        expected = stable_hash({
            "features": result.features.features,
            "signals": [{"direction": s.direction, "raw_signal_strength": s.raw_signal_strength,
                         "raw_expected_edge": s.raw_expected_edge, "signal_horizon": s.signal_horizon,
                         "signal_status": s.signal_status, "route_class": s.route_class,
                         "data_confidence": s.data_confidence}
                        for s in result.signals],
            "warnings": result.warnings, "quality_flags": result.quality_flags,
        })
        assert result.output_hashes["features"] == expected


# -----------------------------------------------------------------------
# Evidence bridge
# -----------------------------------------------------------------------

class TestM4EvidenceBridge:
    def _run_detection(self, db_session, *, price=11.50, high_52w=10.00):
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session, provider="FMP", endpoint="/stable/historical-price-eod/full",
            asof_timestamp=_ts(), raw_payload={"close": price, "high_52w": high_52w},
            job_run_id=run.job_run_id,
        )
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=price, high_52w=high_52w, cohort_extensions=_cohort_extensions(30, 0.15)),
            fundamental_data={"market_cap": 75_000_000},
            lineage_hashes=[lineage.raw_payload_hash], job_run_id=run.job_run_id,
        )
        result = det.detect(inp)
        persisted = persist_detection_result(
            db_session, result, det,
            job_run_id=run.job_run_id, data_lineage_ids=[lineage.data_lineage_id],
        )
        db_session.flush()
        return run, result, persisted

    def test_signal_persists_with_vault_fields(self, db_session):
        run, result, persisted = self._run_detection(db_session)
        assert persisted.feature_snapshot_id is not None
        assert len(persisted.signal_ids) == 1
        sig = db_session.get(SignalRegistry, persisted.signal_ids[0])
        assert sig.feature_snapshot_id == persisted.feature_snapshot_id
        assert sig.pattern_id == "M4"
        assert sig.signal_horizon == "15d"
        assert sig.route_class == "A"
        assert sig.thesis_category == "right_tail_convex"
        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        assert feat.feature_manifest_version == "m4-v1"

    def test_fresh_activation_persists_class_c_route_and_activation_identity(self, db_session):
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session, provider="Alpaca", endpoint="/v2/stocks/bars/latest",
            asof_timestamp=_ts(), raw_payload={"last_price": 10.5}, job_run_id=run.job_run_id,
        )
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={
                "entry_lane": "fresh_breakout_activation", "activation_state": "activated",
                "price": 9.90, "last_price": 10.50, "high_52w": 10.00,
                "intraday_range_confirmation": 1.25, "intraday_volume_confirmation": 1.50,
                "spread_pct_vs_eval_quote": 0.005,
                "operating_universe_inclusion": True,
                **_m4_fresh_activation_fields(),
            },
            lineage_hashes=[lineage.raw_payload_hash], job_run_id=run.job_run_id,
        )
        result = det.detect(inp)
        persisted = persist_detection_result(
            db_session, result, det,
            job_run_id=run.job_run_id, data_lineage_ids=[lineage.data_lineage_id],
        )
        db_session.flush()

        sig = db_session.get(SignalRegistry, persisted.signal_ids[0])
        assert sig.route_class == "C"
        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        features = json.loads(feat.feature_json)
        assert features["activation_id"] == "m4-act-ACME-20260520-150000"
        assert features["activation_timestamp"] == "2026-05-20T15:00:00Z"

    def test_job_run_id_preserved(self, db_session):
        run, _, persisted = self._run_detection(db_session)
        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        sig = db_session.get(SignalRegistry, persisted.signal_ids[0])
        assert feat.job_run_id == run.job_run_id
        assert sig.job_run_id == run.job_run_id

    def test_universe_snapshot_id_preserved(self, db_session):
        run = _setup_run(db_session)
        usn = record_universe_snapshot(
            db_session, ticker="ACME", asof_timestamp=_ts(),
            operating_universe_inclusion=True, job_run_id=run.job_run_id,
        )
        lineage = record_data_lineage(
            db_session, provider="FMP", endpoint="/stable/historical-price-eod/full",
            asof_timestamp=_ts(), raw_payload={"close": 11.0}, job_run_id=run.job_run_id,
        )
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=11.00, high_52w=10.00, cohort_extensions=[0.10]),
            lineage_hashes=[lineage.raw_payload_hash],
        )
        persisted = persist_detection_result(
            db_session, det.detect(inp), det,
            job_run_id=run.job_run_id, universe_snapshot_id=usn.universe_snapshot_id,
            data_lineage_ids=[lineage.data_lineage_id],
        )
        db_session.flush()
        assert db_session.get(SignalRegistry, persisted.signal_ids[0]).universe_snapshot_id == usn.universe_snapshot_id

    def test_no_signal_writes_feature_only(self, db_session):
        _, _, persisted = self._run_detection(db_session, price=9.50, high_52w=10.00)
        assert persisted.feature_snapshot_id is not None
        assert len(persisted.signal_ids) == 0

    def test_feature_hash_deterministic(self, db_session):
        _, _, p1 = self._run_detection(db_session)
        _, _, p2 = self._run_detection(db_session)
        f1 = db_session.get(FeatureSnapshot, p1.feature_snapshot_id)
        f2 = db_session.get(FeatureSnapshot, p2.feature_snapshot_id)
        assert f1.feature_hash == f2.feature_hash

    def test_exact_high_persists_through_bridge(self, db_session):
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session, provider="FMP", endpoint="/stable/historical-price-eod/full",
            asof_timestamp=_ts(), raw_payload={"close": 10.0}, job_run_id=run.job_run_id,
        )
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=_m4_base_data(price=10.00, high_52w=10.00),
            lineage_hashes=[lineage.raw_payload_hash],
        )
        result = det.detect(inp)
        assert result.has_signal
        persisted = persist_detection_result(
            db_session, result, det,
            job_run_id=run.job_run_id, data_lineage_ids=[lineage.data_lineage_id],
        )
        db_session.flush()
        assert len(persisted.signal_ids) == 1
        assert db_session.get(SignalRegistry, persisted.signal_ids[0]).pattern_id == "M4"
