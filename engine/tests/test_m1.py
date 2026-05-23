"""
M1 Post-Earnings Announcement Drift detector tests.

Vault contract verification:
  - Fires on top-quintile positive SUE with valid announcement recency
  - No signal below quintile threshold, negative SUE, stale announcement
  - Feature evidence preserved on all rejection paths
  - raw_expected_edge = remaining X_bar * lambda_M1_15td, capped by configured cap
  - signal_strength maps top-quintile [0.80, 1.00] to [0.0, 1.0]
  - Evidence bridge: pattern_id M1, route_class A, thesis event_drift, remaining horizon
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pytest

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
from alpha.patterns.m1 import (
    DECAY_INTEGRATED_AVG,
    DECAY_TAU,
    LAMBDA_M1_15TD,
    LAMBDA_M1_MONTHLY,
    MAX_DELTA_T,
    MICROCAP_AMPLIFICATION,
    MIN_SUE_PERCENTILE,
    RAW_EDGE_CAP,
    RHO1_LOWER,
    RHO1_UPPER,
    STREAK_THRESHOLD,
    M1Detector,
    compute_decay_factor,
    compute_q_sign,
    compute_remaining_decay_integrated_avg,
    compute_signal_strength,
    compute_w_hm,
    compute_w_rho1,
    compute_w_sigma,
    compute_w_streak,
)


def _ts():
    return datetime(2026, 5, 20, 21, 0, 0, tzinfo=timezone.utc)


def _setup_run(db_session):
    job = create_job(db_session, name="m1_detector", job_type="detector", owner="pattern_engine")
    run = start_run(db_session, job_id=job.job_id)
    db_session.flush()
    return run


def _firing_data():
    """Top-quintile SUE, fresh announcement, full multiplier inputs."""
    return {
        "sue_foster": 2.75,
        "delta_t_trading_days": 0,
        "sue_signed_percentile": 0.92,
        "rho1": 0.30,
        "sue_sign_current": 1,
        "sue_sign_prior": 1,
        "d1_decile": 8,
        "sigma_epsilon_percentile": 0.65,
        "sue_streak_length": 2,
        "announcement_date": "2026-05-20",
        "next_earnings_date_estimate": "2026-08-20",
        "next_earnings_trading_days_from_signal": 65,
        "market_data_status": "current",
        "halt_status": "clear",
        "corporate_action_filter_passed": True,
        "operating_universe_inclusion": True,
    }


def _below_threshold_data():
    """SUE positive but below top-quintile threshold."""
    data = _firing_data()
    data["sue_signed_percentile"] = 0.70
    return data


# -----------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------

class TestM1Metadata:
    def test_pattern_id(self):
        assert M1Detector().pattern_id == PatternId.M1

    def test_track(self):
        assert M1Detector().track == PatternTrack.MULTI_DAY

    def test_thesis_category(self):
        assert M1Detector().thesis_category == ThesisCategory.EVENT_DRIFT

    def test_route_class(self):
        assert M1Detector().route_class == RouteClass.A

    def test_vault_constants(self):
        assert LAMBDA_M1_MONTHLY == 0.0075
        assert MICROCAP_AMPLIFICATION == 1.75
        assert MAX_DELTA_T == 15
        assert MIN_SUE_PERCENTILE == 0.80
        assert DECAY_TAU == 5.0
        assert DECAY_INTEGRATED_AVG == 0.35
        assert RAW_EDGE_CAP == 0.02
        assert abs(LAMBDA_M1_15TD - 0.0075 * 15 / 21) < 1e-10


# -----------------------------------------------------------------------
# Formula helpers
# -----------------------------------------------------------------------

class TestFormulas:
    def test_decay_factor_day_0(self):
        assert compute_decay_factor(0) == 1.0

    def test_decay_factor_day_5(self):
        assert abs(compute_decay_factor(5) - math.exp(-1)) < 1e-10

    def test_decay_factor_day_15(self):
        assert compute_decay_factor(15) == math.exp(-15 / 5)

    def test_decay_factor_day_16_hard_zero(self):
        assert compute_decay_factor(16) == 0.0

    def test_decay_factor_negative_hard_zero(self):
        assert compute_decay_factor(-1) == 0.0

    def test_remaining_decay_average_day_0_matches_vault_constant(self):
        assert compute_remaining_decay_integrated_avg(0) == DECAY_INTEGRATED_AVG

    def test_remaining_decay_average_shrinks_with_age(self):
        assert 0.0 < compute_remaining_decay_integrated_avg(5) < compute_remaining_decay_integrated_avg(0)
        assert compute_remaining_decay_integrated_avg(15) == 0.0

    def test_w_rho1_neutral(self):
        assert compute_w_rho1(0.0) == 1.0

    def test_w_rho1_positive(self):
        assert abs(compute_w_rho1(0.30) - 1.12) < 1e-10

    def test_w_rho1_max_winsorized(self):
        # rho1 = 1.0 winsorized to 0.95 -> 1 + 0.95 * 0.4 = 1.38
        assert abs(compute_w_rho1(1.0) - 1.38) < 1e-10

    def test_w_rho1_min_winsorized(self):
        # rho1 = -1.0 winsorized to -0.5 -> 1 + (-0.5) * 0.4 = 0.80
        assert abs(compute_w_rho1(-1.0) - 0.80) < 1e-10

    def test_q_sign_same(self):
        assert compute_q_sign(1, 1) == 1.2

    def test_q_sign_opposite(self):
        assert compute_q_sign(1, -1) == 0.8

    def test_q_sign_prior_zero(self):
        assert compute_q_sign(1, 0) == 1.0

    def test_q_sign_prior_none(self):
        assert compute_q_sign(1, None) == 1.0

    def test_q_sign_current_zero(self):
        assert compute_q_sign(0, 1) == 1.0

    def test_w_hm_top_decile(self):
        # d1_decile=10: 1 + 0.2 * (2*10/10 - 1) = 1 + 0.2*1 = 1.20
        assert abs(compute_w_hm(10) - 1.20) < 1e-10

    def test_w_hm_bottom_decile(self):
        # d1_decile=1: 1 + 0.2 * (2*1/10 - 1) = 1 + 0.2*(-0.8) = 0.84
        assert abs(compute_w_hm(1) - 0.84) < 1e-10

    def test_w_hm_mid_decile(self):
        # d1_decile=5: 1 + 0.2 * (2*5/10 - 1) = 1 + 0.2*0 = 1.0
        assert compute_w_hm(5) == 1.0

    def test_w_sigma_high_d1(self):
        # d1=8, sigma_pct=0.65: 1 + 0.2 * 0.65 = 1.13
        assert abs(compute_w_sigma(8, 0.65) - 1.13) < 1e-10

    def test_w_sigma_low_d1_ignored(self):
        # d1=7: returns 1.0 regardless of sigma
        assert compute_w_sigma(7, 0.99) == 1.0

    def test_w_sigma_d1_boundary_8(self):
        assert compute_w_sigma(8, 0.0) == 1.0
        assert compute_w_sigma(8, 1.0) == 1.2

    def test_w_streak_short(self):
        assert compute_w_streak(2) == 1.0

    def test_w_streak_at_threshold(self):
        assert compute_w_streak(4) == 0.5

    def test_w_streak_long(self):
        assert compute_w_streak(6) == 0.5

    def test_signal_strength_at_threshold(self):
        assert compute_signal_strength(0.80) == 0.0

    def test_signal_strength_midpoint(self):
        assert compute_signal_strength(0.90) == 0.5

    def test_signal_strength_top(self):
        assert compute_signal_strength(1.00) == 1.0

    def test_signal_strength_below_threshold(self):
        assert compute_signal_strength(0.70) == 0.0

    def test_compound_multiplier_range(self):
        """EXPOSURE.md: compound range approximately [0.32, 2.38]."""
        low = compute_w_rho1(RHO1_LOWER) * 0.8 * compute_w_hm(1) * 1.0 * 0.5
        high = compute_w_rho1(RHO1_UPPER) * 1.2 * compute_w_hm(10) * compute_w_sigma(10, 1.0) * 1.0
        assert abs(low - 0.8 * 0.8 * 0.84 * 1.0 * 0.5) < 0.01
        assert abs(high - 1.38 * 1.2 * 1.20 * 1.2 * 1.0) < 0.01


# -----------------------------------------------------------------------
# Firing cases
# -----------------------------------------------------------------------

class TestM1Firing:
    def test_fires_with_complete_data(self):
        det = M1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.has_signal
        sig = result.signals[0]
        assert sig.direction == SignalDirection.LONG
        assert sig.signal_horizon == "15d"
        assert sig.route_class == RouteClass.A
        assert sig.raw_expected_edge > 0
        f = result.features.features
        assert f["signal_generated"] is True
        assert f["sue_foster"] == 2.75
        assert f["delta_t_trading_days"] == 0
        assert f["decay_factor"] == 1.0
        assert f["remaining_horizon_days"] == 15
        assert f["remaining_decay_integrated_avg"] == DECAY_INTEGRATED_AVG
        assert f["compound_multiplier"] > 0
        assert f["exposure_x_m1_t0"] > 0
        assert f["max_hold_days"] == 15

    def test_edge_deterministic(self):
        det = M1Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"])
        r1, r2 = det.detect(inp), det.detect(inp)
        assert r1.signals[0].raw_expected_edge == r2.signals[0].raw_expected_edge
        # Verify the bridge formula
        x_t0 = r1.features.features["exposure_x_m1_t0"]
        x_bar = x_t0 * DECAY_INTEGRATED_AVG
        expected = round(min(x_bar * LAMBDA_M1_15TD, RAW_EDGE_CAP), 6)
        assert r1.signals[0].raw_expected_edge == expected

    def test_signal_strength_from_percentile(self):
        det = M1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        expected = compute_signal_strength(0.92)
        assert result.signals[0].raw_signal_strength == expected

    def test_priors_logged(self):
        det = M1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        f = result.features.features
        assert f["lambda_M1_monthly"] == LAMBDA_M1_MONTHLY
        assert f["microcap_amplification"] == MICROCAP_AMPLIFICATION
        assert f["validated_or_shadow_lambda_M1_15td"] == LAMBDA_M1_15TD
        assert f["lambda_M1_15td"] == LAMBDA_M1_15TD
        assert f["lambda_M1_default_15td"] == LAMBDA_M1_15TD
        assert f["amplified_lambda_M1_15td"] == round(LAMBDA_M1_15TD, 8)
        assert f["lambda_M1_source"] == "shadow_prior"
        assert f["raw_edge_cap"] == RAW_EDGE_CAP
        assert f["raw_edge_cap_source"] == "shadow_prior"
        priors = f["expected_return_priors"]
        assert priors["gross_bps"] == round(result.signals[0].raw_expected_edge * 10_000, 2)

    def test_custom_lambda_injection(self):
        det = M1Detector(lambda_m1_15td=0.01)
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        default = M1Detector().detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        f = result.features.features
        assert result.signals[0].raw_expected_edge > default.signals[0].raw_expected_edge
        assert f["validated_or_shadow_lambda_M1_15td"] == 0.01
        assert f["lambda_M1_15td"] == 0.01
        assert f["lambda_M1_default_15td"] == LAMBDA_M1_15TD
        assert f["lambda_M1_source"] == "validated_or_injected"
        assert f["amplified_lambda_M1_15td"] == 0.01

    def test_multipliers_logged(self):
        det = M1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        m = result.features.features["multipliers"]
        assert "w_rho1" in m
        assert "q_sign" in m
        assert "w_hm" in m
        assert "w_sigma" in m
        assert "w_streak" in m

    def test_data_confidence_default(self):
        det = M1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.signals[0].data_confidence == 1.0

    def test_day_5_decay_reduces_exposure_edge_and_horizon(self):
        det = M1Detector()
        data_d0 = _firing_data()
        data_d5 = _firing_data()
        data_d5["delta_t_trading_days"] = 5
        r0 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data_d0, lineage_hashes=["h"]))
        r5 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data_d5, lineage_hashes=["h"]))
        assert r0.has_signal and r5.has_signal
        assert r5.features.features["x_m1"] < r0.features.features["x_m1"]
        assert r5.features.features["decay_factor"] < 1.0
        assert r5.features.features["remaining_decay_integrated_avg"] < r0.features.features["remaining_decay_integrated_avg"]
        assert r5.features.features["remaining_horizon_days"] == 10
        assert r5.signals[0].signal_horizon == "10d"
        assert r5.signals[0].raw_expected_edge < r0.signals[0].raw_expected_edge

    def test_200_bps_cap_binds(self):
        det = M1Detector()
        data = _firing_data()
        data["sue_foster"] = 10.0  # extreme SUE
        data["sue_signed_percentile"] = 0.99
        data["rho1"] = 0.95
        data["d1_decile"] = 10
        data["sigma_epsilon_percentile"] = 1.0
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.signals[0].raw_expected_edge == round(RAW_EDGE_CAP, 6)
        assert result.features.features["raw_edge_uncapped"] > RAW_EDGE_CAP
        assert result.features.features["raw_edge_cap_applied"] is True

    def test_configurable_raw_edge_cap(self):
        det = M1Detector(raw_edge_cap=0.04)
        data = _firing_data()
        data["sue_foster"] = 10.0
        data["sue_signed_percentile"] = 0.99
        data["rho1"] = 0.95
        data["d1_decile"] = 10
        data["sigma_epsilon_percentile"] = 1.0
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        f = result.features.features
        assert f["raw_edge_cap"] == 0.04
        assert f["raw_edge_cap_source"] == "configured_or_injected"
        assert result.signals[0].raw_expected_edge == round(min(f["raw_edge_uncapped"], 0.04), 6)
        assert result.signals[0].raw_expected_edge > RAW_EDGE_CAP

    def test_invalid_injected_lambda_rejected(self):
        with pytest.raises(ValueError, match="lambda_m1_15td"):
            M1Detector(lambda_m1_15td=float("nan"))

    def test_invalid_injected_raw_edge_cap_rejected(self):
        with pytest.raises(ValueError, match="raw_edge_cap"):
            M1Detector(raw_edge_cap=0.0)

    def test_max_hold_capped_by_next_earnings(self):
        det = M1Detector()
        data = _firing_data()
        data["next_earnings_trading_days_from_signal"] = 10
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features.features["max_hold_days"] == 9  # min(15, 10-1)
        assert result.features.features["remaining_horizon_days"] == 9
        assert result.signals[0].signal_horizon == "9d"

    def test_max_hold_default_when_next_earnings_missing(self):
        det = M1Detector()
        data = _firing_data()
        del data["next_earnings_trading_days_from_signal"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features.features["max_hold_days"] == 15

    def test_streak_penalty_halves_exposure(self):
        det = M1Detector()
        data_short = _firing_data()
        data_short["sue_streak_length"] = 2
        data_long = _firing_data()
        data_long["sue_streak_length"] = 5
        r_short = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data_short, lineage_hashes=["h"]))
        r_long = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data_long, lineage_hashes=["h"]))
        assert r_short.has_signal and r_long.has_signal
        assert r_long.features.features["multipliers"]["w_streak"] == 0.5
        assert r_short.features.features["multipliers"]["w_streak"] == 1.0


# -----------------------------------------------------------------------
# No-signal / rejection cases
# -----------------------------------------------------------------------

class TestM1NoSignal:
    def test_sue_below_threshold(self):
        det = M1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_below_threshold_data(), lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "sue_below_threshold"
        assert result.features.features["signal_generated"] is False
        assert result.features.features["x_m1"] == 0.0

    def test_negative_sue_rejected(self):
        det = M1Detector()
        data = _firing_data()
        data["sue_foster"] = -1.5
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "sue_not_positive"

    def test_zero_sue_rejected(self):
        det = M1Detector()
        data = _firing_data()
        data["sue_foster"] = 0.0
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "sue_not_positive"

    def test_announcement_too_old(self):
        det = M1Detector()
        data = _firing_data()
        data["delta_t_trading_days"] = 16
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "announcement_too_old"

    def test_negative_delta_t(self):
        det = M1Detector()
        data = _firing_data()
        data["delta_t_trading_days"] = -1
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_delta_t"

    def test_fractional_delta_t_rejected_without_truncation(self):
        det = M1Detector()
        data = _firing_data()
        data["delta_t_trading_days"] = 15.9
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_delta_t"
        assert result.features.features["delta_t_trading_days"] is None

    def test_day_15_has_no_remaining_hold_window(self):
        det = M1Detector()
        data = _firing_data()
        data["delta_t_trading_days"] = 15
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "no_remaining_hold_window"
        assert result.features.features["remaining_horizon_days"] == 0

    def test_missing_sue_percentile(self):
        det = M1Detector()
        data = _firing_data()
        del data["sue_signed_percentile"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "missing_sue_percentile"

    def test_non_finite_sue_percentile_rejected(self):
        det = M1Detector()
        data = _firing_data()
        data["sue_signed_percentile"] = float("nan")
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_sue_percentile"

    def test_out_of_range_sue_percentile_rejected(self):
        det = M1Detector()
        data = _firing_data()
        data["sue_signed_percentile"] = float("inf")
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_sue_percentile"

    def test_next_earnings_tomorrow_has_no_remaining_window(self):
        det = M1Detector()
        data = _firing_data()
        data["next_earnings_trading_days_from_signal"] = 1
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "no_remaining_hold_window"
        assert result.features.features["max_hold_days"] == 0

    def test_invalid_next_earnings_window_rejected_without_crash(self):
        det = M1Detector()
        data = _firing_data()
        data["next_earnings_trading_days_from_signal"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_next_earnings_window"

    def test_missing_required_fields(self):
        det = M1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data={"delta_t_trading_days": 0}, lineage_hashes=["h"]))
        assert result.features is None

    def test_non_finite_sue_returns_no_features(self):
        det = M1Detector()
        data = _firing_data()
        data["sue_foster"] = float("nan")
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features is None

    def test_non_finite_delta_t_returns_no_features(self):
        det = M1Detector()
        data = _firing_data()
        data["delta_t_trading_days"] = float("inf")
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features is None

    def test_malformed_required_fields_return_no_features(self):
        det = M1Detector()
        data = _firing_data()
        data["sue_foster"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features is None

    def test_not_operating_universe(self):
        det = M1Detector()
        data = _firing_data()
        data["operating_universe_inclusion"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.quality_flags["not_operating_universe_member"] is True
        assert result.features.features["rejection_reason"] == "not_operating_universe"
        assert result.features.features["sue_foster"] == 2.75
        assert result.features.features["delta_t_trading_days"] == 0
        assert result.features.features["filing_veto_status"] == "not_computed"

    def test_missing_operating_universe_fails_closed(self):
        det = M1Detector()
        data = _firing_data()
        del data["operating_universe_inclusion"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.quality_flags["operating_universe_not_computed"] is True
        assert result.features.features["rejection_reason"] == "missing_operating_universe"

    def test_delayed_market_data_rejected(self):
        det = M1Detector()
        data = _firing_data()
        data["market_data_status"] = "delayed"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "data_delay"

    def test_halted_rejected(self):
        det = M1Detector()
        data = _firing_data()
        data["halt_status"] = "pending"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "halted"

    def test_corporate_action_rejected(self):
        det = M1Detector()
        data = _firing_data()
        data["corporate_action_filter_passed"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "spurious_corporate_action"

    def test_percentile_exact_boundary_fires(self):
        det = M1Detector()
        data = _firing_data()
        data["sue_signed_percentile"] = 0.80
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.signals[0].raw_signal_strength == 0.0

    def test_percentile_just_below_boundary_rejected(self):
        det = M1Detector()
        data = _firing_data()
        data["sue_signed_percentile"] = 0.7999
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "sue_below_threshold"


# -----------------------------------------------------------------------
# Quality flags and fidelity
# -----------------------------------------------------------------------

class TestM1Quality:
    def test_always_full_fidelity(self):
        det = M1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.features.fidelity_tier == FidelityTier.FULL

    def test_missing_lineage_warns(self):
        det = M1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=[]))
        assert result.quality_flags.get("missing_lineage") is True
        assert result.has_signal

    def test_future_timestamp_warns(self):
        det = M1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=datetime(2099, 1, 1, tzinfo=timezone.utc), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.quality_flags.get("future_timestamp") is True

    def test_missing_multiplier_inputs_defaults_with_flags(self):
        det = M1Detector()
        data = _firing_data()
        del data["rho1"]
        del data["d1_decile"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.quality_flags.get("missing_rho1") is True
        assert result.quality_flags.get("missing_d1_decile") is True
        f = result.features.features
        assert f["multipliers"]["w_rho1"] == 1.0  # rho1=0.0 default
        assert f["multipliers"]["w_hm"] == 1.0  # d1_decile=5 default

    def test_invalid_multiplier_inputs_default_neutral_without_crash(self):
        det = M1Detector()
        data = _firing_data()
        data["rho1"] = "N/A"
        data["sue_sign_current"] = "bull"
        data["d1_decile"] = 99
        data["sigma_epsilon_percentile"] = 99
        data["sue_streak_length"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.quality_flags["invalid_rho1"] is True
        assert result.quality_flags["invalid_sue_sign"] is True
        assert result.quality_flags["invalid_d1_decile"] is True
        assert result.quality_flags["invalid_sigma_epsilon_percentile"] is True
        assert result.quality_flags["invalid_sue_streak_length"] is True
        f = result.features.features
        assert f["compound_multiplier"] == 1.0
        assert f["multipliers"] == {
            "w_rho1": 1.0,
            "q_sign": 1.0,
            "w_hm": 1.0,
            "w_sigma": 1.0,
            "w_streak": 1.0,
        }

    def test_field_confidence_sources_affect_data_confidence(self):
        det = M1Detector()
        data = _firing_data()
        data["field_confidence"] = {"eps_history": 0.5}
        result = det.detect(PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=data,
            fundamental_data={"field_confidence": {"fundamentals": 0.8}},
            event_data={"field_confidence": {"filing_veto": 0.9}},
            lineage_hashes=["h"],
        ))
        assert result.has_signal
        assert result.signals[0].data_confidence == 0.36

    def test_filing_veto_status_default(self):
        det = M1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.features.features["filing_veto_status"] == "not_computed"

    def test_filing_veto_status_forwarded_from_event_data(self):
        det = M1Detector()
        result = det.detect(PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_firing_data(),
            event_data={"filing_veto_status": "clear"},
            lineage_hashes=["h"],
        ))
        assert result.features.features["filing_veto_status"] == "clear"


# -----------------------------------------------------------------------
# Diagnostic fields
# -----------------------------------------------------------------------

class TestM1Diagnostics:
    def test_diagnostic_fields_forwarded_from_all_sources(self):
        det = M1Detector()
        result = det.detect(PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_firing_data(),
            fundamental_data={"hazard_score_at_signal": 22, "market_cap_usd": 95_000_000},
            event_data={"filing_veto_status": "clear"},
            lineage_hashes=["h"],
        ))
        assert result.has_signal
        f = result.features.features
        assert f["hazard_score_at_signal"] == 22
        assert f["market_cap_usd"] == 95_000_000
        assert f["filing_veto_status"] == "clear"

    def test_announcement_metadata_logged(self):
        det = M1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        f = result.features.features
        assert f["announcement_date"] == "2026-05-20"
        assert f["next_earnings_date_estimate"] == "2026-08-20"

    def test_overlap_diagnostics_forwarded(self):
        det = M1Detector()
        data = _firing_data()
        data["i1_also_firing"] = True
        data["i5_also_firing"] = False
        result = det.detect(PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=data,
            event_data={"overlapping_pattern_ids": ["I1", "I5"]},
            lineage_hashes=["h"],
        ))
        assert result.has_signal
        f = result.features.features
        assert f["i1_also_firing"] is True
        assert f["i5_also_firing"] is False
        assert f["overlapping_pattern_ids"] == ["I1", "I5"]


# -----------------------------------------------------------------------
# Hashes
# -----------------------------------------------------------------------

class TestM1Hashes:
    def test_stable(self):
        det = M1Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"])
        r1, r2 = det.detect(inp), det.detect(inp)
        assert r1.input_hashes == r2.input_hashes
        assert r1.output_hashes == r2.output_hashes

    def test_change_with_sue(self):
        det = M1Detector()
        d1 = _firing_data()
        d2 = _firing_data()
        d2["sue_foster"] = 3.50
        r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d1, lineage_hashes=["h"]))
        r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d2, lineage_hashes=["h"]))
        assert r1.output_hashes != r2.output_hashes

    def test_input_hash_changes_with_event_data(self):
        det = M1Detector()
        r1 = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(),
            event_data={"filing_veto_status": "not_computed"}, lineage_hashes=["h"],
        ))
        r2 = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(),
            event_data={"filing_veto_status": "clear"}, lineage_hashes=["h"],
        ))
        assert r1.input_hashes != r2.input_hashes

    def test_output_hash_matches(self):
        det = M1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
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

class TestM1EvidenceBridge:
    def _run_detection(self, db_session, *, market_data=None):
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session, provider="FMP", endpoint="/stable/earnings-surprises",
            asof_timestamp=_ts(), raw_payload={"sue": "fixture"}, job_run_id=run.job_run_id,
        )
        det = M1Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=market_data or _firing_data(),
            lineage_hashes=[lineage.raw_payload_hash], job_run_id=run.job_run_id,
        )
        result = det.detect(inp)
        persisted = persist_detection_result(
            db_session, result, det,
            job_run_id=run.job_run_id, data_lineage_ids=[lineage.data_lineage_id],
        )
        db_session.flush()
        return run, result, persisted

    def test_vault_fields(self, db_session):
        _, _, persisted = self._run_detection(db_session)
        sig = db_session.get(SignalRegistry, persisted.signal_ids[0])
        assert sig.pattern_id == "M1"
        assert sig.route_class == "A"
        assert sig.thesis_category == "event_drift"
        assert sig.signal_horizon == "15d"
        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        assert feat.feature_manifest_version == "m1-v1"

    def test_job_run_id(self, db_session):
        run, _, persisted = self._run_detection(db_session)
        assert db_session.get(FeatureSnapshot, persisted.feature_snapshot_id).job_run_id == run.job_run_id
        assert db_session.get(SignalRegistry, persisted.signal_ids[0]).job_run_id == run.job_run_id

    def test_universe_snapshot_id(self, db_session):
        run = _setup_run(db_session)
        usn = record_universe_snapshot(db_session, ticker="ACME", asof_timestamp=_ts(), operating_universe_inclusion=True, job_run_id=run.job_run_id)
        lineage = record_data_lineage(db_session, provider="FMP", endpoint="/stable/earnings-surprises", asof_timestamp=_ts(), raw_payload={"x": 1}, job_run_id=run.job_run_id)
        det = M1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=[lineage.raw_payload_hash]))
        persisted = persist_detection_result(db_session, result, det, job_run_id=run.job_run_id, universe_snapshot_id=usn.universe_snapshot_id, data_lineage_ids=[lineage.data_lineage_id])
        db_session.flush()
        assert db_session.get(SignalRegistry, persisted.signal_ids[0]).universe_snapshot_id == usn.universe_snapshot_id

    def test_no_signal_feature_only(self, db_session):
        _, _, persisted = self._run_detection(db_session, market_data=_below_threshold_data())
        assert persisted.feature_snapshot_id is not None
        assert len(persisted.signal_ids) == 0

    def test_feature_hash_deterministic(self, db_session):
        _, _, p1 = self._run_detection(db_session)
        _, _, p2 = self._run_detection(db_session)
        assert db_session.get(FeatureSnapshot, p1.feature_snapshot_id).feature_hash == db_session.get(FeatureSnapshot, p2.feature_snapshot_id).feature_hash
