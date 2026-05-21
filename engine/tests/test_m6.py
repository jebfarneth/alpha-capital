"""
M6 Volatility-Compression Breakout detector tests.

Vault contract verification:
  - Standard activation: breakout + range expansion + volume + spread + freshness
  - Early-gap activation: first 30 min, open gap above compression_high, volume + spread + freshness
  - Watchlist signals for compressed setups without activation data
  - Operating-universe exclusion blocks all paths
  - Evidence bridge writes correct vault fields
"""

from __future__ import annotations

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
from alpha.patterns.m6 import (
    AMPLIFICATION,
    HOLD_DAYS,
    LAMBDA_M6_12TD,
    LAMBDA_M6_MONTHLY,
    MIN_COMPRESSION_DEPTH,
    X_M6_CAP,
    M6Detector,
    compute_breakout_extension,
    compute_compression_depth,
    compute_expansion_confirmation,
    compute_volume_confirmation,
)


def _ts():
    return datetime(2026, 5, 20, 21, 0, 0, tzinfo=timezone.utc)


def _setup_run(db_session):
    job = create_job(db_session, name="m6_detector", job_type="detector", owner="pattern_engine")
    run = start_run(db_session, job_id=job.job_id)
    db_session.flush()
    return run


def _firing_market_data():
    """Deep compression + full standard activation data."""
    return {
        "compression_ratio": 0.55,
        "gk_vol_5d": 0.018,
        "gk_vol_60d": 0.033,
        "compression_high": 4.85,
        "sigma_20d": 0.028,
        "price": 5.20,
        "session_high": 5.25,
        "session_low": 4.80,
        "cumulative_volume": 250000,
        "expected_tod_volume": 120000,
        "gk_avg_5d": 0.0003,
        "spread_pct_vs_eval_quote": 0.005,
        "candidate_eval_bid": 5.18,
        "candidate_eval_ask": 5.20,
        "candidate_eval_quote_timestamp": "2026-05-20T15:00:00Z",
        "quote_age_ms": 650,
        "quote_freshness_max_ms": 1000,
        "operating_universe_inclusion": True,
    }


def _early_gap_market_data():
    """Deep compression + open gap breakout, early session, strong gap, no range expansion data."""
    return {
        "compression_ratio": 0.55,
        "gk_vol_5d": 0.018,
        "gk_vol_60d": 0.033,
        "compression_high": 4.85,
        "sigma_20d": 0.028,
        "price": 5.15,
        "open_price": 5.10,
        "prior_close": 4.83,
        "gk_avg_5d": 0.0003,
        "minutes_since_open": 12,
        "latest_5m_volume_ratio": 3.0,
        "spread_pct_vs_eval_quote": 0.005,
        "candidate_eval_bid": 5.13,
        "candidate_eval_ask": 5.15,
        "candidate_eval_quote_timestamp": "2026-05-20T14:42:00Z",
        "quote_age_ms": 600,
        "quote_freshness_max_ms": 1000,
        "operating_universe_inclusion": True,
    }


def _no_breakout_market_data():
    return {
        "compression_ratio": 0.60,
        "gk_vol_5d": 0.020,
        "gk_vol_60d": 0.033,
        "compression_high": 5.00,
        "sigma_20d": 0.030,
        "price": 4.90,
        "operating_universe_inclusion": True,
    }


def _not_compressed_market_data():
    return {
        "compression_ratio": 0.90,
        "gk_vol_5d": 0.030,
        "gk_vol_60d": 0.033,
        "compression_high": 5.00,
        "sigma_20d": 0.030,
        "price": 5.20,
        "operating_universe_inclusion": True,
    }


# -----------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------

class TestM6Metadata:
    def test_pattern_id(self):
        assert M6Detector().pattern_id == PatternId.M6

    def test_track(self):
        assert M6Detector().track == PatternTrack.MULTI_DAY

    def test_thesis_category(self):
        assert M6Detector().thesis_category == ThesisCategory.RIGHT_TAIL_CONVEX

    def test_route_class(self):
        assert M6Detector().route_class == RouteClass.C

    def test_vault_constants(self):
        assert LAMBDA_M6_MONTHLY == 0.011
        assert AMPLIFICATION == 1.45
        assert HOLD_DAYS == 12
        assert X_M6_CAP == 3.0
        assert MIN_COMPRESSION_DEPTH == 0.5
        assert abs(LAMBDA_M6_12TD - 0.011 * 1.45 * 12 / 21) < 1e-10


# -----------------------------------------------------------------------
# Compression formula
# -----------------------------------------------------------------------

class TestCompressionFormula:
    def test_deep(self):
        assert round(compute_compression_depth(0.50), 4) == 1.25

    def test_mild(self):
        assert round(compute_compression_depth(0.80), 4) == 0.5

    def test_none(self):
        assert compute_compression_depth(1.0) == 0.0

    def test_expanding(self):
        assert compute_compression_depth(1.2) == 0.0

    def test_extreme_capped(self):
        assert compute_compression_depth(0.0) == 2.5


class TestBreakoutExtension:
    def test_above_high(self):
        assert round(compute_breakout_extension(5.25, 5.00, 0.03), 3) == 1.667

    def test_no_breakout(self):
        assert compute_breakout_extension(4.90, 5.00, 0.03) == 0.0

    def test_capped(self):
        assert compute_breakout_extension(10.00, 5.00, 0.01) == 3.0


class TestConfirmationTiers:
    def test_expansion_strong(self):
        assert compute_expansion_confirmation(2.5) == 1.5

    def test_expansion_moderate(self):
        assert compute_expansion_confirmation(1.5) == 1.25

    def test_expansion_mild(self):
        assert compute_expansion_confirmation(1.0) == 1.0

    def test_expansion_weak(self):
        assert compute_expansion_confirmation(0.5) == 0.5

    def test_volume_strong(self):
        assert compute_volume_confirmation(2.0) == 1.5

    def test_volume_weak(self):
        assert compute_volume_confirmation(0.8) == 0.5


# -----------------------------------------------------------------------
# Standard activation
# -----------------------------------------------------------------------

class TestM6StandardActivation:
    def test_fires(self):
        det = M6Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_market_data(), lineage_hashes=["h"]))
        assert result.has_signal
        assert result.signals[0].direction == SignalDirection.LONG
        assert result.signals[0].signal_horizon == "12d"
        assert result.features.features["activation_state"] == "activated"
        assert result.features.features["activation_path"] == "standard"
        assert result.features.features["standard_activation_passed"] is True

    def test_edge_deterministic(self):
        det = M6Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_market_data(), lineage_hashes=["h"])
        r1, r2 = det.detect(inp), det.detect(inp)
        x_m6 = r1.features.features["X_M6_activation"]
        expected = round(x_m6 * LAMBDA_M6_12TD, 6)
        assert r1.signals[0].raw_expected_edge == expected
        assert r1.signals[0].raw_expected_edge == r2.signals[0].raw_expected_edge

    def test_strength_capped(self):
        det = M6Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_market_data(), lineage_hashes=["h"]))
        x_m6 = result.features.features["X_M6_activation"]
        assert result.signals[0].raw_signal_strength == round(min(x_m6 / X_M6_CAP, 1.0), 6)

    def test_latest_5m_volume_confirmation_feeds_exposure(self):
        det = M6Detector()
        data = _firing_market_data()
        data["cumulative_volume"] = 90000
        data["expected_tod_volume"] = 120000
        data["latest_5m_volume_ratio"] = 3.0
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        f = result.features.features
        assert f["volume_ignition_passed"] is True
        assert f["cumulative_volume_confirmation"] == 0.5
        assert f["latest_5m_volume_confirmation"] == 1.5
        assert f["intraday_volume_confirmation"] == 1.5

    def test_data_confidence_default(self):
        det = M6Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_market_data(), lineage_hashes=["h"]))
        assert result.signals[0].data_confidence == 1.0

    def test_priors_logged(self):
        det = M6Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_market_data(), lineage_hashes=["h"]))
        priors = result.features.features["expected_return_priors"]
        assert priors["gross_bps"] == round(result.signals[0].raw_expected_edge * 10_000, 2)

    def test_fails_without_range_expansion_and_no_early_gap(self):
        det = M6Detector()
        data = _firing_market_data()
        del data["session_high"]
        del data["session_low"]
        del data["gk_avg_5d"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_state"] == "activation_failed"
        assert result.features.features["standard_activation_passed"] is False
        assert result.features.features["early_gap_activation_passed"] is False


# -----------------------------------------------------------------------
# Early-gap activation
# -----------------------------------------------------------------------

class TestM6EarlyGapActivation:
    def test_fires_with_gap_and_volume(self):
        det = M6Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_early_gap_market_data(), lineage_hashes=["h"]))
        assert result.has_signal
        f = result.features.features
        assert f["activation_state"] == "activated"
        assert f["activation_path"] == "early_gap_activation"
        assert f["early_gap_activation_passed"] is True
        assert f["early_session_flag"] is True
        assert f["gap_breakout_flag"] is True
        assert f["gap_breakout_extension"] > 0
        assert f["gap_expansion_proxy_available"] is True
        assert f["gap_expansion_proxy_reason"] == "valid"
        assert f["gap_expansion_proxy"] is not None
        assert f["prior_close"] == 4.83
        assert f["range_expansion_passed"] is False

    def test_strong_gap_gets_1_5_expansion(self):
        """Strong positive gap -> gap_expansion_proxy >= 2.0 -> tier 1.5."""
        det = M6Detector()
        data = _early_gap_market_data()
        # open=5.10, prior_close=4.83 -> ln(5.10/4.83)=0.0544, / sqrt(0.0003)=0.01732 -> ~3.14
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        f = result.features.features
        assert f["gap_expansion_proxy"] > 2.0
        assert f["early_gap_expansion_confirmation"] == 1.5

    def test_strong_gap_increases_x_m6_vs_neutral(self):
        """With gap proxy, X_M6_activation uses 1.5 not 1.0 -> higher edge."""
        det = M6Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_early_gap_market_data(), lineage_hashes=["h"]))
        f = result.features.features
        depth = f["compression_depth"]
        brk_ext = f["intraday_breakout_extension"]
        vol_conf = f["intraday_volume_confirmation"]
        exp_conf = f["early_gap_expansion_confirmation"]
        expected_x = min(depth * brk_ext * exp_conf * vol_conf, X_M6_CAP)
        assert f["X_M6_activation"] == round(expected_x, 6)
        # Should be higher than the neutral 1.0 case
        neutral_x = min(depth * brk_ext * 1.0 * vol_conf, X_M6_CAP)
        assert f["X_M6_activation"] >= neutral_x

    def test_moderate_gap_gets_1_25(self):
        det = M6Detector()
        data = _early_gap_market_data()
        # Target proxy in [1.5, 2.0): ln(4.93/4.84)=0.01844 / sqrt(0.00012)=0.01095 -> 1.684
        data["prior_close"] = 4.84
        data["open_price"] = 4.93
        data["gk_avg_5d"] = 0.00012
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        f = result.features.features
        assert f["gap_expansion_proxy_available"] is True
        assert 1.5 <= f["gap_expansion_proxy"] < 2.0
        assert f["early_gap_expansion_confirmation"] == 1.25

    def test_weak_gap_gets_0_5(self):
        det = M6Detector()
        data = _early_gap_market_data()
        # Tiny gap: open barely above prior_close
        data["prior_close"] = 5.09
        data["open_price"] = 5.10
        # ln(5.10/5.09)=0.001963 / sqrt(0.0003)=0.01732 -> 0.113 -> tier 0.5
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        f = result.features.features
        assert f["gap_expansion_proxy_available"] is True
        assert f["gap_expansion_proxy"] < 1.0
        assert f["early_gap_expansion_confirmation"] == 0.5

    def test_down_gap_no_expansion_credit(self):
        det = M6Detector()
        data = _early_gap_market_data()
        data["prior_close"] = 5.20  # open 5.10 < prior_close 5.20 -> down gap
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        f = result.features.features
        assert f["gap_expansion_proxy_available"] is False
        assert f["gap_expansion_proxy_reason"] == "non_positive_open_gap"
        assert f["gap_expansion_proxy"] is None
        assert f["early_gap_expansion_confirmation"] == 0.5

    def test_missing_prior_close_fallback(self):
        det = M6Detector()
        data = _early_gap_market_data()
        del data["prior_close"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal  # still fires — early gap candidate with fallback
        f = result.features.features
        assert f["gap_expansion_proxy_available"] is False
        assert f["gap_expansion_proxy_reason"] == "missing_prior_close"
        assert f["early_gap_expansion_confirmation"] == 1.0  # conservative fallback
        assert result.quality_flags.get("missing_prior_close") is True
        assert any("prior_close" in w for w in result.warnings)

    def test_missing_gk_avg_fallback_reason(self):
        det = M6Detector()
        data = _early_gap_market_data()
        del data["gk_avg_5d"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        f = result.features.features
        assert f["gap_expansion_proxy_available"] is False
        assert f["gap_expansion_proxy_reason"] == "missing_gk_avg_5d"
        assert f["early_gap_expansion_confirmation"] == 1.0
        assert result.quality_flags.get("missing_gk_avg_5d_for_gap_proxy") is True

    def test_fails_after_30_minutes(self):
        det = M6Detector()
        data = _early_gap_market_data()
        data["minutes_since_open"] = 35
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["early_session_flag"] is False
        assert result.features.features["early_gap_activation_passed"] is False
        assert result.features.features["activation_failure_reason"] == "range_expansion_failed"

    def test_fails_if_open_below_compression_high(self):
        det = M6Detector()
        data = _early_gap_market_data()
        data["open_price"] = 4.80  # below compression_high 4.85
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["gap_breakout_flag"] is False
        assert result.features.features["activation_failure_reason"] == "range_expansion_failed"

    def test_fails_without_volume_ignition(self):
        det = M6Detector()
        data = _early_gap_market_data()
        data["latest_5m_volume_ratio"] = 1.0  # below 2.0x threshold
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["volume_ignition_passed"] is False
        assert result.features.features["activation_failure_reason"] == "volume_ignition_failed"

    def test_missing_quote_rejected_before_class_c_activation(self):
        det = M6Detector()
        data = _firing_market_data()
        del data["candidate_eval_ask"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["quote_capture_passed"] is False
        assert result.features.features["activation_failure_reason"] == "quote_unavailable"

    def test_fails_with_wide_spread(self):
        det = M6Detector()
        data = _early_gap_market_data()
        data["spread_pct_vs_eval_quote"] = 0.02
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["spread_discipline_passed"] is False
        assert result.features.features["activation_failure_reason"] == "spread_too_wide"

    def test_fails_if_not_operating_universe(self):
        det = M6Detector()
        data = _early_gap_market_data()
        data["operating_universe_inclusion"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.quality_flags["not_operating_universe_member"] is True

    def test_standard_activation_preferred_when_both_pass(self):
        det = M6Detector()
        data = _firing_market_data()
        data["open_price"] = 5.10
        data["prior_close"] = 4.83
        data["minutes_since_open"] = 15
        data["latest_5m_volume_ratio"] = 3.0
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["activation_path"] == "standard"
        assert result.features.features["standard_activation_passed"] is True
        assert result.features.features["early_gap_activation_passed"] is True


# -----------------------------------------------------------------------
# No-signal cases
# -----------------------------------------------------------------------

class TestM6NoSignal:
    def test_no_breakout(self):
        det = M6Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_no_breakout_market_data(), lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_state"] == "no_breakout"

    def test_not_compressed(self):
        det = M6Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_not_compressed_market_data(), lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_state"] == "not_compressed"

    def test_watchlist(self):
        det = M6Detector()
        data = {
            "compression_ratio": 0.55, "gk_vol_5d": 0.018, "gk_vol_60d": 0.033,
            "compression_high": 4.85, "sigma_20d": 0.028,
            "operating_universe_inclusion": True,
        }
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.signals[0].signal_status == "watchlist"
        assert result.signals[0].raw_expected_edge == 0.0
        assert result.features.features["activation_state"] == "watchlist"

    def test_missing_compression(self):
        det = M6Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data={"price": 5.0}, lineage_hashes=["h"]))
        assert result.features is None

    def test_weak_volume(self):
        det = M6Detector()
        data = _firing_market_data()
        data["cumulative_volume"] = 100000
        data["expected_tod_volume"] = 120000
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_failure_reason"] == "volume_ignition_failed"

    def test_wide_spread(self):
        det = M6Detector()
        data = _firing_market_data()
        data["spread_pct_vs_eval_quote"] = 0.02
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_failure_reason"] == "spread_too_wide"

    def test_stale_signal(self):
        det = M6Detector()
        data = _firing_market_data()
        data["signal_freshness_passed"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_failure_reason"] == "signal_expired"

    def test_universe_exclusion_watchlist(self):
        det = M6Detector()
        data = {"compression_ratio": 0.55, "gk_vol_5d": 0.018, "gk_vol_60d": 0.033, "compression_high": 4.85, "sigma_20d": 0.028, "operating_universe_inclusion": False}
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.quality_flags["not_operating_universe_member"] is True

    def test_universe_exclusion_activation(self):
        det = M6Detector()
        data = _firing_market_data()
        data["operating_universe_inclusion"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal

    def test_missing_operating_universe_fails_closed(self):
        det = M6Detector()
        data = _early_gap_market_data()
        del data["operating_universe_inclusion"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.quality_flags["operating_universe_not_computed"] is True
        assert result.features.features["rejection_reason"] == "missing_operating_universe"


# -----------------------------------------------------------------------
# Fidelity
# -----------------------------------------------------------------------

class TestM6Fidelity:
    def test_gk_warning_lite(self):
        det = M6Detector()
        data = _firing_market_data()
        data["gk_low_transaction_warning"] = True
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features.fidelity_tier == FidelityTier.LITE

    def test_gk_warning_reduces_confidence(self):
        det = M6Detector()
        data = _firing_market_data()
        data["gk_low_transaction_warning"] = True
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.signals[0].data_confidence < 1.0

    def test_clean_is_full(self):
        det = M6Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_market_data(), lineage_hashes=["h"]))
        assert result.features.fidelity_tier == FidelityTier.FULL


# -----------------------------------------------------------------------
# Guards
# -----------------------------------------------------------------------

class TestM6Guards:
    def test_missing_lineage(self):
        det = M6Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_market_data(), lineage_hashes=[]))
        assert result.quality_flags.get("missing_lineage") is True
        assert result.has_signal

    def test_future_timestamp(self):
        det = M6Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=datetime(2099, 1, 1, tzinfo=timezone.utc), market_data=_firing_market_data(), lineage_hashes=["h"]))
        assert result.quality_flags.get("future_timestamp") is True


# -----------------------------------------------------------------------
# Hashes
# -----------------------------------------------------------------------

class TestM6Hashes:
    def test_stable(self):
        det = M6Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_market_data(), lineage_hashes=["h"])
        r1, r2 = det.detect(inp), det.detect(inp)
        assert r1.input_hashes == r2.input_hashes
        assert r1.output_hashes == r2.output_hashes

    def test_change_with_compression(self):
        det = M6Detector()
        d1 = _firing_market_data()
        d2 = _firing_market_data()
        d2["compression_ratio"] = 0.90
        r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d1, lineage_hashes=["h"]))
        r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d2, lineage_hashes=["h"]))
        assert r1.has_signal
        assert not r2.has_signal
        assert r1.output_hashes != r2.output_hashes

    def test_output_hash_matches(self):
        det = M6Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_market_data(), lineage_hashes=["h"]))
        expected = stable_hash({
            "features": result.features.features,
            "signals": [{"direction": s.direction, "raw_signal_strength": s.raw_signal_strength,
                         "raw_expected_edge": s.raw_expected_edge, "signal_horizon": s.signal_horizon,
                         "signal_status": s.signal_status, "data_confidence": s.data_confidence}
                        for s in result.signals],
            "warnings": result.warnings, "quality_flags": result.quality_flags,
        })
        assert result.output_hashes["features"] == expected


# -----------------------------------------------------------------------
# Evidence bridge
# -----------------------------------------------------------------------

class TestM6EvidenceBridge:
    def _run_detection(self, db_session, *, market_data=None):
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session, provider="FMP", endpoint="/stable/historical-price-eod/full",
            asof_timestamp=_ts(), raw_payload={"ohlcv": "fixture"}, job_run_id=run.job_run_id,
        )
        det = M6Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=market_data or _firing_market_data(),
            fundamental_data={"market_cap": 60_000_000, "sector": "Technology"},
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
        assert sig.pattern_id == "M6"
        assert sig.route_class == "C"
        assert sig.thesis_category == "right_tail_convex"
        assert sig.signal_horizon == "12d"
        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        assert feat.feature_manifest_version == "m6-v1"

    def test_job_run_id(self, db_session):
        run, _, persisted = self._run_detection(db_session)
        assert db_session.get(FeatureSnapshot, persisted.feature_snapshot_id).job_run_id == run.job_run_id
        assert db_session.get(SignalRegistry, persisted.signal_ids[0]).job_run_id == run.job_run_id

    def test_universe_snapshot_id(self, db_session):
        run = _setup_run(db_session)
        usn = record_universe_snapshot(db_session, ticker="ACME", asof_timestamp=_ts(), operating_universe_inclusion=True, job_run_id=run.job_run_id)
        lineage = record_data_lineage(db_session, provider="FMP", endpoint="/stable/historical-price-eod/full", asof_timestamp=_ts(), raw_payload={"x": 1}, job_run_id=run.job_run_id)
        det = M6Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_market_data(), lineage_hashes=[lineage.raw_payload_hash]))
        persisted = persist_detection_result(db_session, result, det, job_run_id=run.job_run_id, universe_snapshot_id=usn.universe_snapshot_id, data_lineage_ids=[lineage.data_lineage_id])
        db_session.flush()
        assert db_session.get(SignalRegistry, persisted.signal_ids[0]).universe_snapshot_id == usn.universe_snapshot_id

    def test_no_signal_feature_only(self, db_session):
        _, _, persisted = self._run_detection(db_session, market_data=_no_breakout_market_data())
        assert persisted.feature_snapshot_id is not None
        assert len(persisted.signal_ids) == 0

    def test_feature_hash_deterministic(self, db_session):
        _, _, p1 = self._run_detection(db_session)
        _, _, p2 = self._run_detection(db_session)
        assert db_session.get(FeatureSnapshot, p1.feature_snapshot_id).feature_hash == db_session.get(FeatureSnapshot, p2.feature_snapshot_id).feature_hash

    def test_early_gap_persists_through_bridge(self, db_session):
        _, _, persisted = self._run_detection(db_session, market_data=_early_gap_market_data())
        assert len(persisted.signal_ids) == 1
        sig = db_session.get(SignalRegistry, persisted.signal_ids[0])
        assert sig.pattern_id == "M6"
        assert sig.route_class == "C"
