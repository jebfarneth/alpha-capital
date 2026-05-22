"""
I8 Opening Range Breakout detector tests.

Vault contract verification:
  - Fires on breakout above completed opening range with volume/range/spread quality
  - No signal below minimum breakout strength, spread too wide, or non-universe
  - Rejected candidates preserved with rejection_reason for shadow validation
  - raw_expected_edge = X_I8 * lambda_I8_3td
  - signal_strength = X_I8 / 5.0
  - Evidence bridge: pattern_id I8, route_class C, thesis right_tail_convex, horizon 3d
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
from alpha.patterns.i8 import (
    LAMBDA_I8_3TD_DEFAULT,
    MIN_BREAKOUT_STRENGTH,
    X_I8_CAP,
    X_I8_STRENGTH_DIVISOR,
    SIGNAL_HORIZON,
    I8Detector,
    _pre_signal_rejection,
    compute_breakout_strength,
    compute_volume_quality,
    compute_range_quality,
    compute_spread_quality,
)


def _ts():
    return datetime(2026, 5, 15, 14, 18, 0, tzinfo=timezone.utc)  # ~10:18 AM ET


def _setup_run(db_session):
    job = create_job(db_session, name="i8_detector", job_type="detector", owner="pattern_engine")
    run = start_run(db_session, job_id=job.job_id)
    db_session.flush()
    return run


def _firing_data():
    """Strong breakout: 1.5 sigma above range, 2x volume, expanded range, tight spread."""
    return {
        "opening_range_high": 4.85,
        "opening_range_low": 4.62,
        "sigma_20d": 0.025,
        "breakout_price": 5.03,
        "volume_30min": 200000,
        "avg_volume_30min_20d": 100000,
        "avg_range_30min_20d": 0.18,
        "spread_at_eval_bps": 60,
        "normal_spread_20d_bps": 75,
        "candidate_eval_bid": 5.01,
        "candidate_eval_ask": 5.04,
        "candidate_eval_quote_timestamp": "2026-05-15T14:18:00Z",
        "quote_age_ms": 650,
        "quote_freshness_max_ms": 1000,
        "run_id": "I8-20260515-100000",
        "candidate_eval_id": "I8-ACME-20260515-101800",
        "opening_bar_close_timestamp": "2026-05-15T14:00:00Z",
        "breakout_eval_timestamp": "2026-05-15T14:18:00Z",
        "data_cutoff_timestamp": "2026-05-15T14:18:00Z",
        "market_data_status": "current",
        "halt_status": "clear",
        "corporate_action_filter_passed": True,
        "operating_universe_inclusion": True,
    }


def _marginal_data():
    """Marginal breakout: barely above 0.5 sigma, normal everything."""
    return {
        "opening_range_high": 4.85,
        "opening_range_low": 4.62,
        "sigma_20d": 0.025,
        "breakout_price": 4.92,
        "volume_30min": 100000,
        "avg_volume_30min_20d": 100000,
        "avg_range_30min_20d": 0.23,
        "spread_at_eval_bps": 80,
        "normal_spread_20d_bps": 75,
        "candidate_eval_bid": 4.90,
        "candidate_eval_ask": 4.93,
        "candidate_eval_quote_timestamp": "2026-05-15T14:18:00Z",
        "quote_age_ms": 600,
        "quote_freshness_max_ms": 1000,
        "run_id": "I8-20260515-100000",
        "candidate_eval_id": "I8-ACME-20260515-101800",
        "opening_bar_close_timestamp": "2026-05-15T14:00:00Z",
        "breakout_eval_timestamp": "2026-05-15T14:18:00Z",
        "data_cutoff_timestamp": "2026-05-15T14:18:00Z",
        "market_data_status": "current",
        "halt_status": "clear",
        "corporate_action_filter_passed": True,
        "operating_universe_inclusion": True,
    }


# -----------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------

class TestI8Metadata:
    def test_pattern_id(self):
        assert I8Detector().pattern_id == PatternId.I8

    def test_track(self):
        assert I8Detector().track == PatternTrack.INTRADAY

    def test_thesis_category(self):
        assert I8Detector().thesis_category == ThesisCategory.RIGHT_TAIL_CONVEX

    def test_route_class(self):
        assert I8Detector().route_class == RouteClass.C

    def test_vault_constants(self):
        assert X_I8_CAP == 5.0
        assert X_I8_STRENGTH_DIVISOR == 5.0
        assert MIN_BREAKOUT_STRENGTH == 0.5
        assert SIGNAL_HORIZON == "3d"


# -----------------------------------------------------------------------
# Formula helpers
# -----------------------------------------------------------------------

class TestFormulas:
    def test_breakout_strength_normal(self):
        # price 5.03, high 4.85, sigma 0.025 -> (5.03 - 4.85) / (0.025 * 4.85) = 1.485
        s = compute_breakout_strength(5.03, 4.85, 0.025)
        assert round(s, 3) == 1.485

    def test_breakout_strength_capped(self):
        assert compute_breakout_strength(10.0, 4.85, 0.025) == 4.0

    def test_breakout_strength_no_breakout(self):
        assert compute_breakout_strength(4.80, 4.85, 0.025) == 0.0

    def test_volume_quality_tiers(self):
        assert compute_volume_quality(2.5) == 1.5
        assert compute_volume_quality(1.7) == 1.25
        assert compute_volume_quality(1.1) == 1.0
        assert compute_volume_quality(0.8) == 0.5

    def test_range_quality_tiers(self):
        assert compute_range_quality(1.8) == 1.5
        assert compute_range_quality(1.2) == 1.25
        assert compute_range_quality(0.8) == 1.0
        assert compute_range_quality(0.5) == 0.75

    def test_compressed_range_confirmed_boost(self):
        assert compute_range_quality(
            0.5, breakout_strength=1.2, volume_quality=1.25, spread_quality=1.0,
        ) == 1.25

    def test_compressed_range_confirmed_neutral(self):
        assert compute_range_quality(
            0.5, breakout_strength=0.7, volume_quality=1.0, spread_quality=1.0,
        ) == 1.0

    def test_compressed_range_penalized_when_unconfirmed(self):
        assert compute_range_quality(
            0.5, breakout_strength=1.2, volume_quality=0.5, spread_quality=1.0,
        ) == 0.75
        assert compute_range_quality(
            0.5, breakout_strength=1.2, volume_quality=1.25, spread_quality=0.75,
        ) == 0.75

    def test_spread_quality_tiers(self):
        assert compute_spread_quality(60, 75) == 1.25  # 0.8x normal -> tight
        assert compute_spread_quality(90, 75) == 1.0   # 1.2x normal
        assert compute_spread_quality(150, 75) == 0.75  # 2.0x normal
        assert compute_spread_quality(200, 75) == 0.0   # 2.67x normal -> hard gate
        assert compute_spread_quality(300, 0) == 0.75   # missing baseline -> conservative proxy

    def test_canonical_strong_from_spec(self):
        """EXPOSURE.md: 1.5 sigma, 2x vol, 1.5x range, tight spread -> 4.22."""
        x = min(1.5 * 1.5 * 1.5 * 1.25, 5.0)
        assert round(x, 2) == 4.22

    def test_canonical_marginal_from_spec(self):
        """EXPOSURE.md: 0.6 sigma, normal everything -> 0.6."""
        x = min(0.6 * 1.0 * 1.0 * 1.0, 5.0)
        assert x == 0.6


# -----------------------------------------------------------------------
# Firing cases
# -----------------------------------------------------------------------

class TestI8Firing:
    def test_strong_breakout_fires(self):
        det = I8Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.has_signal
        assert result.signals[0].direction == SignalDirection.LONG
        assert result.signals[0].signal_horizon == "3d"
        assert result.signals[0].route_class == RouteClass.C
        assert result.features.features["signal_generated"] is True
        assert result.features.features["x_i8"] > 0
        assert result.features.features["market_data_status"] == "current"
        assert result.features.features["halt_status"] == "clear"
        assert result.features.features["corporate_action_filter_passed"] is True
        assert result.features.features["filing_veto_status"] == "not_computed"

    def test_edge_deterministic(self):
        det = I8Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"])
        r1, r2 = det.detect(inp), det.detect(inp)
        x_i8 = r1.features.features["x_i8"]
        expected = round(x_i8 * LAMBDA_I8_3TD_DEFAULT, 6)
        assert r1.signals[0].raw_expected_edge == expected
        assert r1.signals[0].raw_expected_edge == r2.signals[0].raw_expected_edge

    def test_signal_strength_normalized(self):
        det = I8Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        x_i8 = result.features.features["x_i8"]
        expected = round(min(x_i8 / 5.0, 1.0), 6)
        assert abs(result.signals[0].raw_signal_strength - expected) < 1e-5

    def test_priors_logged(self):
        det = I8Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        priors = result.features.features["expected_return_priors"]
        assert priors["gross_bps"] == round(result.signals[0].raw_expected_edge * 10_000, 2)
        assert "lambda_i8_3td" not in priors
        assert result.features.features["validated_or_shadow_lambda_I8_3td"] == LAMBDA_I8_3TD_DEFAULT
        assert result.features.features["lambda_I8_3td"] == LAMBDA_I8_3TD_DEFAULT
        assert result.features.features["lambda_I8_default_3td"] == LAMBDA_I8_3TD_DEFAULT
        assert result.features.features["lambda_I8_source"] == "shadow_prior"

    def test_marginal_breakout_fires(self):
        det = I8Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_marginal_data(), lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["breakout_strength"] >= MIN_BREAKOUT_STRENGTH

    def test_custom_lambda(self):
        det = I8Detector(lambda_i8_3td=0.005)
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        x_i8 = result.features.features["x_i8"]
        assert result.signals[0].raw_expected_edge == round(x_i8 * 0.005, 6)
        assert result.features.features["validated_or_shadow_lambda_I8_3td"] == 0.005
        assert result.features.features["lambda_I8_3td"] == 0.005
        assert result.features.features["lambda_I8_default_3td"] == LAMBDA_I8_3TD_DEFAULT
        assert result.features.features["lambda_I8_source"] == "validated_or_injected"
        assert "lambda_i8_3td" not in result.features.features["expected_return_priors"]

    def test_compressed_range_with_strong_confirmation_gets_boost(self):
        det = I8Detector()
        data = _firing_data()
        data["avg_range_30min_20d"] = 0.50  # opening range 0.23 -> compressed ratio 0.46
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        f = result.features.features
        assert result.has_signal
        assert f["compressed_range_flag"] is True
        assert f["base_range_quality"] == 0.75
        assert f["range_quality"] == 1.25
        assert f["compressed_range_treatment"] == "compressed_confirmed_boost"

    def test_compressed_range_with_adequate_confirmation_gets_neutral(self):
        det = I8Detector()
        data = _firing_data()
        data["avg_range_30min_20d"] = 0.50
        data["volume_30min"] = 100000
        data["avg_volume_30min_20d"] = 100000
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        f = result.features.features
        assert result.has_signal
        assert f["compressed_range_flag"] is True
        assert f["volume_quality"] == 1.0
        assert f["range_quality"] == 1.0
        assert f["compressed_range_treatment"] == "compressed_confirmed_neutral"

    def test_compressed_range_with_weak_volume_stays_penalized(self):
        det = I8Detector()
        data = _firing_data()
        data["avg_range_30min_20d"] = 0.50
        data["volume_30min"] = 80000
        data["avg_volume_30min_20d"] = 100000
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        f = result.features.features
        assert result.has_signal
        assert f["compressed_range_flag"] is True
        assert f["volume_quality"] == 0.5
        assert f["range_quality"] == 0.75
        assert f["compressed_range_treatment"] == "compressed_penalized"


# -----------------------------------------------------------------------
# No-signal / rejection cases
# -----------------------------------------------------------------------

class TestI8NoSignal:
    def test_no_upside_breakout(self):
        det = I8Detector()
        data = _firing_data()
        data["breakout_price"] = 4.80  # below range high
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "no_upside_breakout"

    def test_exact_range_high_is_not_breakout(self):
        det = I8Detector()
        data = _firing_data()
        data["breakout_price"] = data["opening_range_high"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["breakout_strength"] == 0.0
        assert result.features.features["rejection_reason"] == "no_upside_breakout"

    def test_breakout_below_threshold(self):
        det = I8Detector()
        data = _firing_data()
        data["breakout_price"] = 4.86  # barely above high -> strength < 0.5
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "breakout_below_threshold"

    def test_spread_too_wide(self):
        det = I8Detector()
        data = _firing_data()
        data["spread_at_eval_bps"] = 200  # 2.67x normal -> hard gate
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "spread_too_wide"
        assert "range_quality" in result.features.features
        assert "compressed_range_treatment" in result.features.features

    def test_missing_breakout_price(self):
        det = I8Detector()
        data = _firing_data()
        del data["breakout_price"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "no_upside_breakout"

    def test_missing_required_fields(self):
        det = I8Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data={"sigma_20d": 0.02}, lineage_hashes=["h"]))
        assert result.features is None

    def test_not_operating_universe(self):
        det = I8Detector()
        data = _firing_data()
        data["operating_universe_inclusion"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.quality_flags["not_operating_universe_member"] is True
        assert result.features.features["opening_range_high"] == 4.85
        assert result.features.features["opening_range_low"] == 4.62
        assert result.features.features["sigma_20d"] == 0.025
        assert result.features.features["candidate_eval_id"] == "I8-ACME-20260515-101800"
        assert result.features.features["market_data_status"] == "current"
        assert result.features.features["halt_status"] == "clear"
        assert result.features.features["corporate_action_filter_passed"] is True

    def test_missing_operating_universe_fails_closed(self):
        det = I8Detector()
        data = _firing_data()
        del data["operating_universe_inclusion"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.quality_flags["operating_universe_not_computed"] is True
        assert result.features.features["rejection_reason"] == "missing_operating_universe"

    def test_halted_during_opening(self):
        det = I8Detector()
        data = _firing_data()
        data["halted_during_opening"] = True
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "halted_during_opening"

    def test_late_evaluation(self):
        det = I8Detector()
        data = _firing_data()
        data["late_evaluation"] = True
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "late_evaluation_stale"

    def test_missing_quote(self):
        det = I8Detector()
        data = _firing_data()
        del data["candidate_eval_ask"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "quote_unavailable"

    def test_stale_quote(self):
        det = I8Detector()
        data = _firing_data()
        data["quote_age_ms"] = 1250
        data["quote_freshness_max_ms"] = 1000
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "quote_unavailable"

    def test_malformed_quote_rejected_without_crashing(self):
        det = I8Detector()
        data = _firing_data()
        data["candidate_eval_bid"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "quote_unavailable"
        assert result.features.features["signal_generated"] is False

    def test_missing_timestamp_rejected_as_insufficient_bar_data(self):
        det = I8Detector()
        data = _firing_data()
        del data["data_cutoff_timestamp"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "insufficient_bar_data"

    def test_missing_market_data_quality_fails_closed(self):
        det = I8Detector()
        data = _firing_data()
        del data["market_data_status"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "missing_market_data_quality"
        assert result.quality_flags["market_data_quality_rejected"] is True

    def test_delayed_market_data_rejected(self):
        det = I8Detector()
        data = _firing_data()
        data["market_data_status"] = "delayed"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "data_delay"
        assert result.features.features["market_data_status"] == "delayed"

    def test_non_clear_halt_status_rejected(self):
        det = I8Detector()
        data = _firing_data()
        data["halt_status"] = "pending"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "halted"
        assert result.features.features["halt_status"] == "pending"

    def test_corporate_action_filter_rejected(self):
        det = I8Detector()
        data = _firing_data()
        data["corporate_action_filter_passed"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "spurious_corporate_action"
        assert result.features.features["corporate_action_filter_passed"] is False

    def test_missing_volume_rejected(self):
        det = I8Detector()
        data = _firing_data()
        del data["volume_30min"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "volume_below_minimum"

    def test_below_average_volume_is_penalized_not_blocked(self):
        det = I8Detector()
        data = _firing_data()
        data["volume_30min"] = 80000
        data["avg_volume_30min_20d"] = 100000
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["volume_ratio"] == 0.8
        assert result.features.features["volume_quality"] == 0.5

    def test_invalid_bar(self):
        det = I8Detector()
        data = _firing_data()
        data["opening_range_high"] = 4.50
        data["opening_range_low"] = 4.85  # high < low
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "insufficient_bar_data"

    def test_zero_range_bar_rejected(self):
        det = I8Detector()
        data = _firing_data()
        data["opening_range_high"] = 4.85
        data["opening_range_low"] = 4.85
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["opening_range_high"] == 4.85
        assert result.features.features["opening_range_low"] == 4.85
        assert result.features.features["rejection_reason"] == "insufficient_bar_data"
        assert result.features.features["candidate_eval_id"] == "I8-ACME-20260515-101800"
        assert result.features.features["opening_bar_close_timestamp"] == "2026-05-15T14:00:00Z"
        assert result.features.features["data_cutoff_timestamp"] == "2026-05-15T14:18:00Z"
        assert result.features.features["market_data_status"] == "current"
        assert result.features.features["halt_status"] == "clear"
        assert result.features.features["corporate_action_filter_passed"] is True

    def test_late_evaluation_rejection_reads_market_data(self):
        data = _firing_data()
        data["late_evaluation"] = True
        assert _pre_signal_rejection({}, data) == "late_evaluation_stale"


# -----------------------------------------------------------------------
# Quality flags and fidelity
# -----------------------------------------------------------------------

class TestI8Quality:
    def test_always_full_fidelity(self):
        det = I8Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.features.fidelity_tier == FidelityTier.FULL

    def test_missing_lineage_warns(self):
        det = I8Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=[]))
        assert result.quality_flags.get("missing_lineage") is True
        assert result.has_signal

    def test_baseline_volume_proxy(self):
        det = I8Detector()
        data = _firing_data()
        del data["avg_volume_30min_20d"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["baseline_volume_proxy"] is True
        assert result.features.features["avg_volume_30min_20d"] == result.features.features["volume_30min"]
        assert result.features.features["volume_ratio"] == 1.0
        assert result.features.features["volume_quality"] == 1.0
        assert result.signals[0].data_confidence < 1.0

    def test_missing_spread_uses_conservative_proxy(self):
        det = I8Detector()
        data = _firing_data()
        del data["spread_at_eval_bps"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["spread_quality"] == 0.75
        assert result.quality_flags["baseline_spread_proxy"] is True
        assert result.signals[0].data_confidence < 1.0

    def test_invalid_normal_spread_uses_conservative_proxy(self):
        det = I8Detector()
        data = _firing_data()
        data["normal_spread_20d_bps"] = 0
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["spread_quality"] == 0.75
        assert result.features.features["normal_spread_20d_bps"] == 0.0
        assert result.quality_flags["baseline_spread_proxy"] is True
        assert result.signals[0].data_confidence < 1.0

    def test_future_timestamp_warns(self):
        det = I8Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=datetime(2099, 1, 1, tzinfo=timezone.utc), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.quality_flags.get("future_timestamp") is True


# -----------------------------------------------------------------------
# Hashes
# -----------------------------------------------------------------------

class TestI8Hashes:
    def test_stable(self):
        det = I8Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"])
        r1, r2 = det.detect(inp), det.detect(inp)
        assert r1.input_hashes == r2.input_hashes
        assert r1.output_hashes == r2.output_hashes

    def test_change_with_price(self):
        det = I8Detector()
        d1 = _firing_data()
        d2 = _firing_data()
        d2["breakout_price"] = 5.20
        r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d1, lineage_hashes=["h"]))
        r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d2, lineage_hashes=["h"]))
        assert r1.output_hashes != r2.output_hashes

    def test_output_hash_matches(self):
        det = I8Detector()
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

class TestI8EvidenceBridge:
    def _run_detection(self, db_session, *, market_data=None):
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session, provider="Alpaca", endpoint="/v2/bars/30min",
            asof_timestamp=_ts(), raw_payload={"bars": "fixture"}, job_run_id=run.job_run_id,
        )
        det = I8Detector()
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
        assert sig.pattern_id == "I8"
        assert sig.route_class == "C"
        assert sig.thesis_category == "right_tail_convex"
        assert sig.signal_horizon == "3d"
        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        assert feat.feature_manifest_version == "i8-v1"
        features = json.loads(feat.feature_json)
        assert features["candidate_eval_id"] == "I8-ACME-20260515-101800"
        assert features["data_cutoff_timestamp"] == "2026-05-15T14:18:00Z"

    def test_job_run_id(self, db_session):
        run, _, persisted = self._run_detection(db_session)
        assert db_session.get(FeatureSnapshot, persisted.feature_snapshot_id).job_run_id == run.job_run_id
        assert db_session.get(SignalRegistry, persisted.signal_ids[0]).job_run_id == run.job_run_id

    def test_no_signal_feature_only(self, db_session):
        data = _firing_data()
        data["breakout_price"] = 4.80  # below range
        _, _, persisted = self._run_detection(db_session, market_data=data)
        assert persisted.feature_snapshot_id is not None
        assert len(persisted.signal_ids) == 0

    def test_feature_hash_deterministic(self, db_session):
        _, _, p1 = self._run_detection(db_session)
        _, _, p2 = self._run_detection(db_session)
        assert db_session.get(FeatureSnapshot, p1.feature_snapshot_id).feature_hash == db_session.get(FeatureSnapshot, p2.feature_snapshot_id).feature_hash
