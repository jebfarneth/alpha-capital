"""
I1 Gap and Go detector tests.

Vault contract verification:
  - Fires on confirmed gap (gap >= 3%, positive 30-min return, above-avg volume)
  - No signal on failed confirmation, below-minimum gap, or non-universe
  - Feature snapshot with gap candidates preserved even without signal
  - raw_expected_edge = X_I1 * lambda_I1_3td
  - signal_strength = X_I1 / 10.0
  - Evidence bridge: pattern_id I1, route_class C, thesis right_tail_convex, horizon 3d
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
from alpha.patterns.i1 import (
    AMPLIFICATION,
    HOLD_DAYS,
    LAMBDA_I1_3TD,
    LAMBDA_I1_MONTHLY,
    MIN_GAP_PCT,
    X_I1_STRENGTH_DIVISOR,
    I1Detector,
    compute_gap_magnitude,
    compute_confirmation_gate,
    compute_volume_weight,
)


def _ts():
    return datetime(2026, 5, 15, 14, 1, 0, tzinfo=timezone.utc)  # ~10:01 AM ET, past date


def _setup_run(db_session):
    job = create_job(db_session, name="i1_detector", job_type="detector", owner="pattern_engine")
    run = start_run(db_session, job_id=job.job_id)
    db_session.flush()
    return run


def _confirmed_gap_data():
    """Strong confirmed gap: 5.2% gap, positive 30-min return, 2x volume."""
    return {
        "prev_close": 4.01,
        "open_price": 4.22,
        "sigma_20d": 0.025,
        "return_30min": 0.031,
        "volume_30min": 185000,
        "avg_volume_30min_20d": 92000,
        "price_at_10am": 4.35,
        "evaluation_run_id": "I1-20260515-100100",
        "evaluation_timestamp": "2026-05-15T14:01:00Z",
        "data_cutoff_timestamp": "2026-05-15T14:00:00Z",
        "candidate_eval_bid": 4.34,
        "candidate_eval_ask": 4.36,
        "candidate_eval_quote_timestamp": "2026-05-15T14:01:00Z",
        "quote_age_ms": 850,
        "quote_freshness_max_ms": 1000,
        "opening_auction_quality": "normal",
        "market_data_status": "current",
        "halt_status": "clear",
        "corporate_action_filter_passed": True,
        "operating_universe_inclusion": True,
    }


def _failed_confirmation_data():
    """Gap up but 30-min return is negative (reversal)."""
    return {
        "prev_close": 4.01,
        "open_price": 4.22,
        "sigma_20d": 0.025,
        "return_30min": -0.01,
        "volume_30min": 185000,
        "avg_volume_30min_20d": 92000,
        "evaluation_timestamp": "2026-05-15T14:01:00Z",
        "data_cutoff_timestamp": "2026-05-15T14:00:00Z",
        "candidate_eval_bid": 4.34,
        "candidate_eval_ask": 4.36,
        "candidate_eval_quote_timestamp": "2026-05-15T14:01:00Z",
        "quote_age_ms": 850,
        "quote_freshness_max_ms": 1000,
        "market_data_status": "current",
        "halt_status": "clear",
        "corporate_action_filter_passed": True,
        "operating_universe_inclusion": True,
    }


def _small_gap_data():
    """Gap below 3% minimum."""
    return {
        "prev_close": 4.01,
        "open_price": 4.08,  # ~1.7% gap
        "sigma_20d": 0.025,
        "return_30min": 0.02,
        "volume_30min": 100000,
        "avg_volume_30min_20d": 80000,
        "evaluation_timestamp": "2026-05-15T14:01:00Z",
        "data_cutoff_timestamp": "2026-05-15T14:00:00Z",
        "candidate_eval_bid": 4.10,
        "candidate_eval_ask": 4.11,
        "candidate_eval_quote_timestamp": "2026-05-15T14:01:00Z",
        "quote_age_ms": 500,
        "quote_freshness_max_ms": 1000,
        "market_data_status": "current",
        "halt_status": "clear",
        "corporate_action_filter_passed": True,
        "operating_universe_inclusion": True,
    }


# -----------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------

class TestI1Metadata:
    def test_pattern_id(self):
        assert I1Detector().pattern_id == PatternId.I1

    def test_track(self):
        assert I1Detector().track == PatternTrack.INTRADAY

    def test_thesis_category(self):
        assert I1Detector().thesis_category == ThesisCategory.RIGHT_TAIL_CONVEX

    def test_route_class(self):
        assert I1Detector().route_class == RouteClass.C

    def test_vault_constants(self):
        assert LAMBDA_I1_MONTHLY == 0.0347
        assert AMPLIFICATION == 1.75
        assert HOLD_DAYS == 3
        assert MIN_GAP_PCT == 0.03
        assert X_I1_STRENGTH_DIVISOR == 10.0
        assert abs(LAMBDA_I1_3TD - 0.0347 * 1.75 * 3 / 21) < 1e-10


# -----------------------------------------------------------------------
# Formula helpers
# -----------------------------------------------------------------------

class TestFormulas:
    def test_gap_magnitude_normal(self):
        # 4% gap / 2% sigma = 2.0
        assert compute_gap_magnitude(0.04, 0.02) == 2.0

    def test_gap_magnitude_capped(self):
        # 15% gap / 2% sigma = 7.5 -> capped at 5.0
        assert compute_gap_magnitude(0.15, 0.02) == 5.0

    def test_gap_magnitude_negative(self):
        assert compute_gap_magnitude(-0.02, 0.02) == 0.0

    def test_confirmation_gate_passes(self):
        assert compute_confirmation_gate(0.01, 100000, 90000) == 1.0

    def test_confirmation_gate_fails_negative_return(self):
        assert compute_confirmation_gate(-0.01, 100000, 90000) == 0.0

    def test_confirmation_gate_fails_low_volume(self):
        assert compute_confirmation_gate(0.01, 80000, 90000) == 0.0

    def test_confirmation_gate_fails_flat_return_boundary(self):
        assert compute_confirmation_gate(0.0, 100000, 90000) == 0.0

    def test_confirmation_gate_fails_exact_average_volume_boundary(self):
        assert compute_confirmation_gate(0.01, 90000, 90000) == 0.0

    def test_volume_weight_tiers(self):
        assert compute_volume_weight(3.5) == 2.0
        assert compute_volume_weight(2.5) == 1.5
        assert compute_volume_weight(1.7) == 1.25
        assert compute_volume_weight(1.1) == 1.0
        assert compute_volume_weight(0.8) == 0.0

    def test_canonical_baseline_from_spec(self):
        """EXPOSURE.md: 4% gap (2.0 sigma), positive return, 2x volume -> X_I1 = 3.0."""
        gap_mag = compute_gap_magnitude(0.04, 0.02)
        conf = compute_confirmation_gate(0.01, 200000, 100000)
        vol_w = compute_volume_weight(2.0)
        assert gap_mag * conf * vol_w == 3.0


# -----------------------------------------------------------------------
# Firing cases
# -----------------------------------------------------------------------

class TestI1Firing:
    def test_confirmed_gap_fires(self):
        det = I1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_confirmed_gap_data(), lineage_hashes=["h"]))
        assert result.has_signal
        assert result.signals[0].direction == SignalDirection.LONG
        assert result.signals[0].signal_horizon == "3d"
        assert result.features.features["confirmation_gate"] == 1.0
        assert result.features.features["signal_generated"] is True
        assert result.features.features["x_i1"] == result.features.features["X_I1"]

    def test_edge_deterministic(self):
        det = I1Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_confirmed_gap_data(), lineage_hashes=["h"])
        r1, r2 = det.detect(inp), det.detect(inp)
        x_i1 = r1.features.features["X_I1"]
        expected = round(x_i1 * LAMBDA_I1_3TD, 6)
        assert r1.signals[0].raw_expected_edge == expected
        assert r1.signals[0].raw_expected_edge == r2.signals[0].raw_expected_edge

    def test_signal_strength_normalized_to_10(self):
        det = I1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_confirmed_gap_data(), lineage_hashes=["h"]))
        x_i1 = result.features.features["X_I1"]
        # Both rounded to 6dp independently — verify they match within tolerance
        expected = round(min(x_i1 / 10.0, 1.0), 6)
        assert abs(result.signals[0].raw_signal_strength - expected) < 1e-5

    def test_priors_logged(self):
        det = I1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_confirmed_gap_data(), lineage_hashes=["h"]))
        priors = result.features.features["expected_return_priors"]
        assert priors["gross_bps"] == round(result.signals[0].raw_expected_edge * 10_000, 2)
        assert set(priors) == {"gross_bps"}
        assert result.features.features["lambda_I1_monthly"] == LAMBDA_I1_MONTHLY
        assert result.features.features["microcap_amplification"] == AMPLIFICATION
        assert result.features.features["amplified_lambda_I1_3td"] == round(LAMBDA_I1_3TD, 8)

    def test_data_confidence_default(self):
        det = I1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_confirmed_gap_data(), lineage_hashes=["h"]))
        assert result.signals[0].data_confidence == 1.0

    def test_monster_gap_high_exposure(self):
        """EXPOSURE.md: 8% gap (4.0 sigma), 4x volume -> X_I1 = 8.0."""
        det = I1Detector()
        data = {
            "prev_close": 4.00, "open_price": 4.32, "sigma_20d": 0.02,
            "return_30min": 0.05, "volume_30min": 400000, "avg_volume_30min_20d": 100000,
            "candidate_eval_bid": 4.34, "candidate_eval_ask": 4.36,
            "candidate_eval_quote_timestamp": "2026-05-15T14:01:00Z", "quote_age_ms": 700,
            "evaluation_timestamp": "2026-05-15T14:01:00Z", "data_cutoff_timestamp": "2026-05-15T14:00:00Z",
            "market_data_status": "current", "halt_status": "clear", "corporate_action_filter_passed": True,
            "operating_universe_inclusion": True,
        }
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["X_I1"] == 8.0
        assert result.features.features["x_i1"] == 8.0


# -----------------------------------------------------------------------
# No-signal cases
# -----------------------------------------------------------------------

class TestI1NoSignal:
    def test_failed_confirmation_no_signal(self):
        det = I1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_failed_confirmation_data(), lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features is not None
        assert result.features.features["confirmation_gate"] == 0.0
        assert result.features.features["volume_weight"] == 0.0
        assert result.features.features["X_I1"] == 0.0
        assert result.features.features["x_i1"] == 0.0
        assert result.features.features["rejection_reason"] == "confirmation_failed"

    def test_small_gap_no_signal(self):
        det = I1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_small_gap_data(), lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["confirmation_gate"] == 0.0
        assert result.features.features["volume_weight"] == 0.0
        assert result.features.features["X_I1"] == 0.0
        assert result.features.features["x_i1"] == 0.0
        assert result.features.features["rejection_reason"] == "gap_below_minimum"

    def test_missing_price_no_features(self):
        det = I1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data={"sigma_20d": 0.02}, lineage_hashes=["h"]))
        assert result.features is None

    def test_not_operating_universe(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        data["operating_universe_inclusion"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.quality_flags["not_operating_universe_member"] is True
        assert result.features.features["confirmation_gate"] == 0.0
        assert result.features.features["volume_weight"] == 0.0
        assert result.features.features["X_I1"] == 0.0
        assert result.features.features["x_i1"] == 0.0
        assert result.features.features["rejection_reason"] == "not_operating_universe"

    def test_missing_confirmation_data(self):
        det = I1Detector()
        data = {
            "prev_close": 4.01, "open_price": 4.22, "sigma_20d": 0.025,
            "candidate_eval_bid": 4.34, "candidate_eval_ask": 4.36,
            "candidate_eval_quote_timestamp": "2026-05-15T14:01:00Z", "quote_age_ms": 850,
            "evaluation_timestamp": "2026-05-15T14:01:00Z", "data_cutoff_timestamp": "2026-05-15T14:00:00Z",
            "market_data_status": "current", "halt_status": "clear", "corporate_action_filter_passed": True,
            "operating_universe_inclusion": True,
        }
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["confirmation_gate"] == 0.0
        assert result.features.features["volume_weight"] == 0.0
        assert result.features.features["X_I1"] == 0.0
        assert result.features.features["x_i1"] == 0.0
        assert result.features.features["rejection_reason"] == "missing_confirmation_data"

    def test_low_volume_fails_confirmation(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        data["volume_30min"] = 50000  # below avg 92000
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["confirmation_gate"] == 0.0

    def test_flat_30min_return_fails_confirmation(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        data["return_30min"] = 0.0
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["confirmation_gate"] == 0.0
        assert result.features.features["rejection_reason"] == "confirmation_failed"

    def test_exact_average_volume_fails_confirmation(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        data["volume_30min"] = data["avg_volume_30min_20d"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["confirmation_gate"] == 0.0
        assert result.features.features["rejection_reason"] == "confirmation_failed"

    def test_delayed_market_data_rejected_pre_signal(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        data["market_data_status"] = "delayed"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "data_delay"
        assert result.features.features["confirmation_gate"] == 0.0

    def test_1014_evaluation_allowed(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        data["evaluation_timestamp"] = "2026-05-15T14:14:00Z"  # 10:14 AM ET
        data["candidate_eval_quote_timestamp"] = "2026-05-15T14:14:00Z"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert "rejection_reason" not in result.features.features

    def test_1015_evaluation_boundary_allowed(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        data["evaluation_timestamp"] = "2026-05-15T14:15:00Z"  # 10:15 AM ET exactly
        data["candidate_eval_quote_timestamp"] = "2026-05-15T14:15:00Z"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert "rejection_reason" not in result.features.features

    def test_1016_evaluation_rejected_as_data_delay(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        data["evaluation_timestamp"] = "2026-05-15T14:16:00Z"  # 10:16 AM ET
        data["candidate_eval_quote_timestamp"] = "2026-05-15T14:16:00Z"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "data_delay"
        assert result.features.features["X_I1"] == 0.0
        assert result.features.features["x_i1"] == 0.0

    def test_invalid_evaluation_timestamp_rejected(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        data["evaluation_timestamp"] = "not-a-timestamp"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "insufficient_timestamp_data"

    def test_data_cutoff_after_evaluation_rejected(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        data["evaluation_timestamp"] = "2026-05-15T14:01:00Z"
        data["data_cutoff_timestamp"] = "2026-05-15T14:02:00Z"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "insufficient_timestamp_data"

    def test_halt_during_confirmation_rejected_pre_signal(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        data["halt_status"] = "halted_confirmation_window"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "halt_during_confirmation"

    def test_corporate_action_gap_rejected_pre_signal(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        data["corporate_action_filter_passed"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "spurious_gap_corporate_action"

    def test_missing_market_data_quality_rejected_pre_signal(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        del data["market_data_status"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "missing_market_data_quality"

    def test_missing_halt_status_rejected_pre_signal(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        del data["halt_status"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "missing_market_data_quality"

    def test_missing_corporate_action_filter_rejected_pre_signal(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        del data["corporate_action_filter_passed"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "missing_market_data_quality"

    def test_missing_quote_rejected_pre_signal(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        del data["candidate_eval_ask"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "quote_unavailable"

    def test_missing_timestamp_rejected_pre_signal(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        del data["data_cutoff_timestamp"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "insufficient_timestamp_data"

    def test_stale_quote_rejected_pre_signal(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        data["quote_age_ms"] = 1250
        data["quote_freshness_max_ms"] = 1000
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "quote_unavailable"

    def test_bad_opening_auction_rejected_pre_signal(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        data["opening_auction_quality"] = "thin_print"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "opening_auction_quality_failed"


# -----------------------------------------------------------------------
# Fidelity and quality flags
# -----------------------------------------------------------------------

class TestI1Quality:
    def test_baseline_volume_proxy(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        del data["avg_volume_30min_20d"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["baseline_volume_proxy"] is True
        assert result.quality_flags.get("baseline_volume_proxy") is True
        assert result.signals[0].data_confidence < 1.0

    def test_missing_lineage_warns(self):
        det = I1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_confirmed_gap_data(), lineage_hashes=[]))
        assert result.quality_flags.get("missing_lineage") is True
        assert result.has_signal

    def test_missing_operating_universe_fails_closed(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        del data["operating_universe_inclusion"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.quality_flags["operating_universe_not_computed"] is True
        assert result.features.features["rejection_reason"] == "missing_operating_universe"

    def test_future_timestamp_warns(self):
        det = I1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=datetime(2099, 1, 1, tzinfo=timezone.utc), market_data=_confirmed_gap_data(), lineage_hashes=["h"]))
        assert result.quality_flags.get("future_timestamp") is True

    def test_always_full_fidelity(self):
        det = I1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_confirmed_gap_data(), lineage_hashes=["h"]))
        assert result.features.fidelity_tier == FidelityTier.FULL


# -----------------------------------------------------------------------
# Diagnostic fields
# -----------------------------------------------------------------------

class TestI1Diagnostics:
    def test_diagnostic_fields_forwarded(self):
        det = I1Detector()
        data = _confirmed_gap_data()
        data["gap_source"] = "earnings"
        data["candidate_eval_ask"] = 4.36
        data["market_data_status"] = "current"
        data["halt_status"] = "clear"
        data["hazard_score_at_signal"] = 14
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        f = result.features.features
        assert f["gap_source"] == "earnings"
        assert f["candidate_eval_ask"] == 4.36
        assert f["hazard_score_at_signal"] == 14
        assert f["halt_status"] == "clear"

    def test_defaults_applied(self):
        det = I1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_confirmed_gap_data(), lineage_hashes=["h"]))
        f = result.features.features
        assert f["halt_status"] == "clear"
        assert f["corporate_action_filter_passed"] is True
        assert f["market_data_status"] == "current"
        assert f["filing_veto_status"] == "clear"


# -----------------------------------------------------------------------
# Hashes
# -----------------------------------------------------------------------

class TestI1Hashes:
    def test_stable(self):
        det = I1Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_confirmed_gap_data(), lineage_hashes=["h"])
        r1, r2 = det.detect(inp), det.detect(inp)
        assert r1.input_hashes == r2.input_hashes
        assert r1.output_hashes == r2.output_hashes

    def test_change_with_gap(self):
        det = I1Detector()
        d1 = _confirmed_gap_data()
        d2 = _confirmed_gap_data()
        d2["open_price"] = 4.50  # different gap
        r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d1, lineage_hashes=["h"]))
        r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d2, lineage_hashes=["h"]))
        assert r1.output_hashes != r2.output_hashes

    def test_output_hash_matches(self):
        det = I1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_confirmed_gap_data(), lineage_hashes=["h"]))
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

class TestI1EvidenceBridge:
    def _run_detection(self, db_session, *, market_data=None):
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session, provider="Alpaca", endpoint="/v2/bars/30min",
            asof_timestamp=_ts(), raw_payload={"bars": "fixture"}, job_run_id=run.job_run_id,
        )
        det = I1Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=market_data or _confirmed_gap_data(),
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
        assert sig.pattern_id == "I1"
        assert sig.route_class == "C"
        assert sig.thesis_category == "right_tail_convex"
        assert sig.signal_horizon == "3d"
        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        assert feat.feature_manifest_version == "i1-v1"

    def test_job_run_id(self, db_session):
        run, _, persisted = self._run_detection(db_session)
        assert db_session.get(FeatureSnapshot, persisted.feature_snapshot_id).job_run_id == run.job_run_id
        assert db_session.get(SignalRegistry, persisted.signal_ids[0]).job_run_id == run.job_run_id

    def test_universe_snapshot_id(self, db_session):
        run = _setup_run(db_session)
        usn = record_universe_snapshot(db_session, ticker="ACME", asof_timestamp=_ts(), operating_universe_inclusion=True, job_run_id=run.job_run_id)
        lineage = record_data_lineage(db_session, provider="Alpaca", endpoint="/v2/bars/30min", asof_timestamp=_ts(), raw_payload={"x": 1}, job_run_id=run.job_run_id)
        det = I1Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_confirmed_gap_data(), lineage_hashes=[lineage.raw_payload_hash]))
        persisted = persist_detection_result(db_session, result, det, job_run_id=run.job_run_id, universe_snapshot_id=usn.universe_snapshot_id, data_lineage_ids=[lineage.data_lineage_id])
        db_session.flush()
        assert db_session.get(SignalRegistry, persisted.signal_ids[0]).universe_snapshot_id == usn.universe_snapshot_id

    def test_no_signal_feature_only(self, db_session):
        _, _, persisted = self._run_detection(db_session, market_data=_failed_confirmation_data())
        assert persisted.feature_snapshot_id is not None
        assert len(persisted.signal_ids) == 0

    def test_feature_hash_deterministic(self, db_session):
        _, _, p1 = self._run_detection(db_session)
        _, _, p2 = self._run_detection(db_session)
        assert db_session.get(FeatureSnapshot, p1.feature_snapshot_id).feature_hash == db_session.get(FeatureSnapshot, p2.feature_snapshot_id).feature_hash
