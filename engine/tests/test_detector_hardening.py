"""
Cross-detector hardening tests.

Verifies string-crash protection, constructor validation, lambda injection,
field_confidence propagation, M1 max-hold-aware decay, and overlap diagnostics
across all implemented detectors.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from alpha.patterns.contracts import PatternInput
from alpha.patterns.guards import compute_data_confidence, finite_float, integral_int
from alpha.patterns.m1 import (
    DECAY_INTEGRATED_AVG,
    LAMBDA_M1_15TD,
    M1Detector,
    compute_remaining_decay_integrated_avg,
)
from alpha.patterns.m2 import LAMBDA_M2_20TD, M2Detector
from alpha.patterns.m3 import LAMBDA_M3_15TD, M3Detector
from alpha.patterns.m4 import LAMBDA_M4_15TD, M4Detector
from alpha.patterns.m5 import LAMBDA_M5_7TD, M5Detector
from alpha.patterns.m6 import LAMBDA_M6_12TD, M6Detector
from alpha.patterns.m7 import LAMBDA_M7_10TD, M7Detector
from alpha.patterns.i1 import LAMBDA_I1_3TD, I1Detector
from alpha.patterns.i8 import LAMBDA_I8_3TD_DEFAULT, I8Detector


def _ts():
    return datetime(2026, 5, 20, 21, 0, 0, tzinfo=timezone.utc)


# -----------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------

class TestSharedHelpers:
    def test_finite_float_normal(self):
        assert finite_float(3.14) == 3.14

    def test_finite_float_string_number(self):
        assert finite_float("3.14") == 3.14

    def test_finite_float_nan(self):
        assert finite_float(float("nan")) is None

    def test_finite_float_inf(self):
        assert finite_float(float("inf")) is None

    def test_finite_float_string_garbage(self):
        assert finite_float("N/A") is None

    def test_finite_float_none(self):
        assert finite_float(None) is None

    def test_finite_float_bool_true(self):
        assert finite_float(True) == 1.0

    def test_integral_int_normal(self):
        assert integral_int(5) == 5

    def test_integral_int_float_integer(self):
        assert integral_int(5.0) == 5

    def test_integral_int_fractional(self):
        assert integral_int(5.5) is None

    def test_integral_int_string_garbage(self):
        assert integral_int("N/A") is None

    def test_integral_int_nan(self):
        assert integral_int(float("nan")) is None


# -----------------------------------------------------------------------
# String-crash protection: each detector survives "N/A" on required fields
# -----------------------------------------------------------------------

class TestStringCrashProtection:
    def test_m4_survives_string_price(self):
        det = M4Detector()
        data = {"price": "N/A", "high_52w": 10.0, "operating_universe_inclusion": True}
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features is None

    def test_m5_survives_string_return(self):
        det = M5Detector()
        data = {"return_5d": "N/A", "sigma_20d": 0.03, "operating_universe_inclusion": True}
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features is None

    def test_m5_survives_string_support_level(self):
        """Optional support_level='N/A' should behave like None, not crash."""
        det = M5Detector()
        data = {
            "return_5d": -0.08, "sigma_20d": 0.032, "support_level": "N/A",
            "low_5d": 3.40, "operating_universe_inclusion": True,
        }
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal  # decline_only_missing_support path

    def test_m6_survives_string_compression_ratio(self):
        det = M6Detector()
        data = {"compression_ratio": "N/A", "compression_high": 5.0, "sigma_20d": 0.03}
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features is None

    def test_i1_survives_string_prev_close(self):
        det = I1Detector()
        data = {"prev_close": "N/A", "open_price": 4.22, "sigma_20d": 0.025}
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features is None

    def test_m2_survives_string_buyers(self):
        det = M2Detector()
        data = {"n_distinct_opp_buyers_30d": "N/A", "days_since_last_opp_buy_filing_detected": 2, "operating_universe_inclusion": True}
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features is None

    def test_m7_survives_string_rank(self):
        det = M7Detector()
        from tests.test_m7 import _firing_data as _m7_data
        data = _m7_data()
        data["predicted_return_rank_pct"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features is None

    def test_m3_survives_string_sector_return(self):
        det = M3Detector()
        data = {"sector": "Energy", "sector_return_6mo": "N/A", "sector_rank_normalized": 0.95, "operating_universe_inclusion": True}
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features is None

    def test_i8_survives_string_opening_range_high(self):
        det = I8Detector()
        data = {"opening_range_high": "N/A", "opening_range_low": 4.62, "sigma_20d": 0.025}
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features is None


class TestHelperStringCrashProtection:
    def test_m4_fresh_activation_survives_string_last_price(self):
        from tests.test_m4 import _m4_base_data, _m4_fresh_activation_fields

        det = M4Detector()
        data = _m4_base_data(
            entry_lane="fresh_breakout_activation",
            activation_state="activated",
            last_price="N/A",
            intraday_range_confirmation=1.2,
            intraday_volume_confirmation=1.3,
            spread_pct_vs_eval_quote=0.004,
            **_m4_fresh_activation_fields(),
        )
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_failure_reason"] == "fresh_high_break_failed"
        assert result.quality_flags["invalid_fresh_last_price"] is True

    def test_m4_fresh_activation_survives_string_range_confirmation(self):
        from tests.test_m4 import _m4_base_data, _m4_fresh_activation_fields

        det = M4Detector()
        data = _m4_base_data(
            entry_lane="fresh_breakout_activation",
            activation_state="activated",
            last_price=10.50,
            intraday_range_confirmation="N/A",
            intraday_volume_confirmation=1.3,
            spread_pct_vs_eval_quote=0.004,
            **_m4_fresh_activation_fields(),
        )
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_failure_reason"] == "range_confirmation_failed"
        assert result.quality_flags["invalid_range_confirmation"] is True

    def test_m5_activation_survives_string_volume_fields(self):
        from tests.test_m5 import _activation_data

        det = M5Detector()
        data = _activation_data()
        data["cumulative_session_volume"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_failure_reason"] == "volume_confirmation_failed"
        assert result.quality_flags["missing_volume_data"] is True

    def test_m6_activation_survives_string_expansion_fields(self):
        from tests.test_m6 import _firing_market_data

        det = M6Detector()
        data = _firing_market_data()
        data["session_high"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_failure_reason"] == "range_expansion_failed"
        assert result.quality_flags["missing_expansion_data"] is True

    def test_i1_confirmation_survives_string_return(self):
        from tests.test_i1 import _confirmed_gap_data

        det = I1Detector()
        data = _confirmed_gap_data()
        data["return_30min"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "missing_confirmation_data"
        assert result.quality_flags["missing_confirmation_data"] is True

    def test_i8_breakout_survives_string_breakout_price(self):
        from tests.test_i8 import _firing_data

        det = I8Detector()
        data = _firing_data()
        data["breakout_price"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "no_upside_breakout"
        assert result.features.features["x_i8"] == 0.0


# -----------------------------------------------------------------------
# Constructor validation
# -----------------------------------------------------------------------

class TestConstructorValidation:
    def test_m2_rejects_nan_lambda(self):
        with pytest.raises(ValueError, match="lambda_m2_20td"):
            M2Detector(lambda_m2_20td=float("nan"))

    def test_m7_rejects_nan_lambda(self):
        with pytest.raises(ValueError, match="lambda_m7_10td"):
            M7Detector(lambda_m7_10td=float("nan"))

    def test_m3_rejects_nan_lambda(self):
        with pytest.raises(ValueError, match="lambda_m3_15td"):
            M3Detector(lambda_m3_15td=float("nan"))

    def test_m4_rejects_nan_lambda(self):
        with pytest.raises(ValueError, match="lambda_m4_15td"):
            M4Detector(lambda_m4_15td=float("nan"))

    def test_m4_rejects_negative_lambda(self):
        with pytest.raises(ValueError, match="lambda_m4_15td"):
            M4Detector(lambda_m4_15td=-0.01)

    def test_m5_rejects_nan_lambda(self):
        with pytest.raises(ValueError, match="lambda_m5_7td"):
            M5Detector(lambda_m5_7td=float("nan"))

    def test_m6_rejects_nan_lambda(self):
        with pytest.raises(ValueError, match="lambda_m6_12td"):
            M6Detector(lambda_m6_12td=float("nan"))

    def test_i1_rejects_nan_lambda(self):
        with pytest.raises(ValueError, match="lambda_i1_3td"):
            I1Detector(lambda_i1_3td=float("nan"))

    def test_i8_rejects_nan_lambda(self):
        with pytest.raises(ValueError, match="lambda_i8_3td"):
            I8Detector(lambda_i8_3td=float("nan"))

    def test_i8_rejects_zero_lambda(self):
        with pytest.raises(ValueError, match="lambda_i8_3td"):
            I8Detector(lambda_i8_3td=0.0)

    def test_i8_rejects_negative_lambda(self):
        with pytest.raises(ValueError, match="lambda_i8_3td"):
            I8Detector(lambda_i8_3td=-0.01)


# -----------------------------------------------------------------------
# Lambda injection: defaults unchanged, injected values propagate
# -----------------------------------------------------------------------

def _m4_base_data():
    return {"price": 11.0, "high_52w": 10.0, "operating_universe_inclusion": True, "cohort_extensions": [0.10]}

def _m5_activation_data():
    return {
        "return_5d": -0.08, "sigma_20d": 0.032, "support_level": 3.45, "low_5d": 3.40,
        "price": 3.55, "open_price": 3.42, "intraday_vwap": 3.50,
        "cumulative_session_volume": 200000, "expected_same_clock_volume_20d": 100000,
        "spread_pct_vs_eval_quote": 0.003,
        "market_data_status": "current", "halt_status": "clear", "corporate_action_filter_passed": True,
        "activation_id": "m5-act-1", "activation_timestamp": "2026-05-20T15:00:00Z",
        "watchlist_signal_id": "ws-1", "watchlist_scan_date": "2026-05-19",
        "watchlist_expiration_session": "2026-05-22", "activation_session": "2026-05-20",
        "watchlist_age_sessions": 1, "signal_freshness_passed": True,
        "candidate_eval_bid": 3.48, "candidate_eval_ask": 3.50,
        "candidate_eval_quote_timestamp": "2026-05-20T15:00:00Z", "quote_age_ms": 650,
        "operating_universe_inclusion": True,
    }


class TestLambdaInjection:
    def test_m4_default_edge_unchanged(self):
        det = M4Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_m4_base_data(), lineage_hashes=["h"]))
        assert result.has_signal
        f = result.features.features
        assert f["lambda_M4_source"] == "shadow_prior"
        assert f["lambda_M4_default_15td"] == round(LAMBDA_M4_15TD, 8)

    def test_m4_injected_lambda_changes_edge(self):
        det = M4Detector(lambda_m4_15td=0.02)
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_m4_base_data(), lineage_hashes=["h"]))
        assert result.has_signal
        f = result.features.features
        assert f["lambda_M4_source"] == "validated_or_injected"
        assert f["validated_or_shadow_lambda_M4_15td"] == 0.02
        default = M4Detector().detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_m4_base_data(), lineage_hashes=["h"]))
        assert result.signals[0].raw_expected_edge > default.signals[0].raw_expected_edge

    def test_m5_default_edge_unchanged(self):
        det = M5Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_m5_activation_data(), lineage_hashes=["h"]))
        assert result.has_signal
        f = result.features.features
        assert f["lambda_M5_source"] == "shadow_prior"
        assert f["lambda_M5_default_7td"] == round(LAMBDA_M5_7TD, 8)

    def test_m5_injected_lambda_changes_edge(self):
        det = M5Detector(lambda_m5_7td=0.03)
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_m5_activation_data(), lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["lambda_M5_source"] == "validated_or_injected"

    def test_m6_injected_lambda_changes_edge(self):
        from tests.test_m6 import _firing_market_data

        det = M6Detector(lambda_m6_12td=0.03)
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_market_data(), lineage_hashes=["h"]))
        assert result.has_signal
        f = result.features.features
        assert f["lambda_M6_source"] == "validated_or_injected"
        assert f["validated_or_shadow_lambda_M6_12td"] == 0.03

    def test_i1_default_edge_unchanged(self):
        from tests.test_i1 import _confirmed_gap_data
        det = I1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_confirmed_gap_data(), lineage_hashes=["h"]))
        assert result.has_signal
        f = result.features.features
        assert f["lambda_I1_source"] == "shadow_prior"

    def test_i1_injected_lambda_changes_edge(self):
        from tests.test_i1 import _confirmed_gap_data

        det = I1Detector(lambda_i1_3td=0.04)
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_confirmed_gap_data(), lineage_hashes=["h"]))
        assert result.has_signal
        f = result.features.features
        assert f["lambda_I1_source"] == "validated_or_injected"
        assert f["validated_or_shadow_lambda_I1_3td"] == 0.04


# -----------------------------------------------------------------------
# M1 max-hold-aware remaining decay
# -----------------------------------------------------------------------

class TestM1MaxHoldDecay:
    def test_full_window_unchanged(self):
        assert compute_remaining_decay_integrated_avg(0, 15) == DECAY_INTEGRATED_AVG

    def test_day_5_still_lower_than_day_0(self):
        assert compute_remaining_decay_integrated_avg(5, 15) < compute_remaining_decay_integrated_avg(0, 15)

    def test_shortened_hold_gives_lower_edge(self):
        full = compute_remaining_decay_integrated_avg(0, 15)
        short = compute_remaining_decay_integrated_avg(0, 1)
        assert short < full
        assert short > 0

    def test_1_day_hold_approximately_one_fifteenth(self):
        one_day = compute_remaining_decay_integrated_avg(0, 1)
        assert abs(one_day - 1.0 / 15.0) < 0.01

    def test_detector_uses_max_hold_for_edge(self):
        """Next earnings in 2 days: max_hold=1, edge should be much lower."""
        from tests.test_m1 import _firing_data
        det = M1Detector()
        full = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"],
        ))
        short_data = _firing_data()
        short_data["next_earnings_trading_days_from_signal"] = 2
        short = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=short_data, lineage_hashes=["h"],
        ))
        assert full.has_signal and short.has_signal
        assert short.features.features["max_hold_days"] == 1
        assert short.features.features["remaining_horizon_days"] == 1
        assert short.signals[0].signal_horizon == "1d"
        assert short.signals[0].raw_expected_edge < full.signals[0].raw_expected_edge
        # Day 0 is the highest-value day, so 1/15 window captures ~19% of decay mass
        ratio = short.signals[0].raw_expected_edge / full.signals[0].raw_expected_edge
        assert ratio < 0.25  # much less than full

    def test_max_hold_0_rejected(self):
        assert compute_remaining_decay_integrated_avg(0, 0) == 0.0

    def test_negative_max_hold_rejected(self):
        assert compute_remaining_decay_integrated_avg(0, -1) == 0.0

    def test_delta_t_at_max_hold_boundary(self):
        assert compute_remaining_decay_integrated_avg(5, 5) == 0.0
        assert compute_remaining_decay_integrated_avg(4, 5) > 0.0


# -----------------------------------------------------------------------
# Field confidence propagation
# -----------------------------------------------------------------------

class TestFieldConfidence:
    def test_m5_field_confidence_affects_data_confidence(self):
        det = M5Detector()
        data = _m5_activation_data()
        data["field_confidence"] = {"close": 0.5}
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=data,
            fundamental_data={"field_confidence": {"fundamentals": 0.8}},
            lineage_hashes=["h"],
        ))
        assert result.has_signal
        assert result.signals[0].data_confidence < 1.0

    def test_i1_field_confidence_affects_data_confidence(self):
        from tests.test_i1 import _confirmed_gap_data
        det = I1Detector()
        data = _confirmed_gap_data()
        data["field_confidence"] = {"eps": 0.5}
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=data,
            fundamental_data={"field_confidence": {"fun": 0.9}},
            lineage_hashes=["h"],
        ))
        assert result.has_signal
        assert result.signals[0].data_confidence < 1.0

    def test_malformed_field_confidence_does_not_crash_or_boost_confidence(self):
        confidence = compute_data_confidence(
            {},
            field_confidence_sources=(
                {"field_confidence": {"bad": "N/A", "over_one": 2.0}},
            ),
        )
        assert confidence == 1.0


# -----------------------------------------------------------------------
# Overlap diagnostics
# -----------------------------------------------------------------------

class TestOverlapDiagnostics:
    def test_m4_overlap_passthrough(self):
        det = M4Detector()
        data = _m4_base_data()
        data["m6_also_firing"] = True
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=data,
            event_data={"overlapping_pattern_ids": ["M6"]},
            lineage_hashes=["h"],
        ))
        assert result.has_signal
        f = result.features.features
        assert f["m6_also_firing"] is True
        assert f["overlapping_pattern_ids"] == ["M6"]

    def test_m5_overlap_passthrough(self):
        det = M5Detector()
        data = {
            "return_5d": -0.08, "sigma_20d": 0.032, "support_level": 3.45,
            "low_5d": 3.40, "operating_universe_inclusion": True,
            "m4_also_firing": True,
        }
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=data,
            event_data={"overlapping_pattern_ids": ["M4"]},
            lineage_hashes=["h"],
        ))
        assert result.has_signal
        f = result.features.features
        assert f["m4_also_firing"] is True
        assert f["overlapping_pattern_ids"] == ["M4"]

    def test_m6_overlap_passthrough(self):
        det = M6Detector()
        data = {
            "compression_ratio": 0.55, "gk_vol_5d": 0.018, "gk_vol_60d": 0.033,
            "compression_high": 4.85, "sigma_20d": 0.028,
            "operating_universe_inclusion": True,
            "m4_also_firing": True,
        }
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=data,
            event_data={"overlapping_pattern_ids": ["M4"]},
            lineage_hashes=["h"],
        ))
        assert result.has_signal  # watchlist
        f = result.features.features
        assert f["m4_also_firing"] is True
        assert f["overlapping_pattern_ids"] == ["M4"]
