"""
M5 Failed Breakdown Reversal detector tests.

Vault contract verification:
  - Watchlist fires on decline >= 1.5 sigma + support break
  - Activation fires on support reclaim + stabilization + volume + identity + quote + spread + freshness
  - No signal on insufficient decline, no support break, or non-universe
  - Rejected activation candidates preserved with rejection_reason
  - raw_expected_edge = X_M5_activation * amplified_lambda_M5_7td
  - signal_strength = X_M5 / 3.0
  - Evidence bridge: pattern_id M5, route_class B, thesis mean_reversion, horizon 7d
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from alpha.data.contracts import stable_hash
from alpha.db.models import FeatureSnapshot, SignalRegistry
from alpha.evidence.writer import create_job, record_data_lineage, start_run
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
from alpha.patterns.m5 import (
    AMPLIFICATION,
    LAMBDA_M5_7TD,
    LAMBDA_M5_WEEKLY,
    MIN_DECLINE_MAGNITUDE,
    X_M5_CAP,
    M5Detector,
    compute_decline_magnitude,
    compute_support_break_attempt_weight,
    compute_support_reclaim_extension,
    compute_support_reclaim_strength,
    compute_stabilization_confirmation,
    compute_volume_confirmation,
)


def _ts():
    return datetime(2026, 5, 20, 21, 0, 0, tzinfo=timezone.utc)


def _setup_run(db_session):
    job = create_job(db_session, name="m5_detector", job_type="detector", owner="pattern_engine")
    run = start_run(db_session, job_id=job.job_id)
    db_session.flush()
    return run


def _activation_fields():
    return {
        "activation_id": "m5-act-ACME-20260520-150000",
        "activation_timestamp": "2026-05-20T15:00:00Z",
        "watchlist_signal_id": "m5-watchlist-ACME-20260519",
        "watchlist_scan_date": "2026-05-19",
        "watchlist_valid_session": "2026-05-20",
        "activation_session": "2026-05-20",
        "signal_freshness_passed": True,
    }


def _quote_fields():
    return {
        "candidate_eval_bid": 3.48,
        "candidate_eval_ask": 3.50,
        "candidate_eval_quote_timestamp": "2026-05-20T15:00:00Z",
        "quote_age_ms": 650,
        "quote_freshness_max_ms": 1000,
    }


def _watchlist_data():
    """Deep decline + support break, no live price for activation."""
    return {
        "return_5d": -0.08,
        "sigma_20d": 0.032,
        "support_level": 3.45,
        "low_5d": 3.40,
        "operating_universe_inclusion": True,
    }


def _activation_data():
    """Deep decline + support break + live reclaim with all gates passing."""
    return {
        "return_5d": -0.08,
        "sigma_20d": 0.032,
        "support_level": 3.45,
        "low_5d": 3.40,
        "price": 3.55,
        "open_price": 3.42,
        "intraday_vwap": 3.50,
        "cumulative_session_volume": 200000,
        "expected_same_clock_volume_20d": 100000,
        "spread_pct_vs_eval_quote": 0.003,
        "market_data_status": "current",
        "halt_status": "clear",
        "corporate_action_filter_passed": True,
        **_activation_fields(),
        **_quote_fields(),
        "operating_universe_inclusion": True,
    }


def _no_setup_data():
    """Decline too small for M5 setup."""
    return {
        "return_5d": -0.02,
        "sigma_20d": 0.032,
        "support_level": 3.45,
        "low_5d": 3.50,
        "operating_universe_inclusion": True,
    }


# -----------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------

class TestM5Metadata:
    def test_pattern_id(self):
        assert M5Detector().pattern_id == PatternId.M5

    def test_track(self):
        assert M5Detector().track == PatternTrack.MULTI_DAY

    def test_thesis_category(self):
        assert M5Detector().thesis_category == ThesisCategory.MEAN_REVERSION

    def test_route_class(self):
        assert M5Detector().route_class == RouteClass.B

    def test_vault_constants(self):
        assert LAMBDA_M5_WEEKLY == 0.0105
        assert AMPLIFICATION == 1.45
        assert X_M5_CAP == 3.0
        assert MIN_DECLINE_MAGNITUDE == 1.5
        assert abs(LAMBDA_M5_7TD - 0.0105 * 1.45) < 1e-10


# -----------------------------------------------------------------------
# Formula helpers
# -----------------------------------------------------------------------

class TestFormulas:
    def test_decline_magnitude_normal(self):
        # -8% decline / 3.2% vol = 2.5
        assert round(compute_decline_magnitude(-0.08, 0.032), 4) == 2.5

    def test_decline_magnitude_capped(self):
        assert compute_decline_magnitude(-0.20, 0.02) == 4.0

    def test_decline_magnitude_positive_return(self):
        assert compute_decline_magnitude(0.05, 0.02) == 0.0

    def test_decline_magnitude_zero_return(self):
        assert compute_decline_magnitude(0.0, 0.02) == 0.0

    def test_support_break_below(self):
        assert compute_support_break_attempt_weight(3.40, 3.45) == 1.25

    def test_support_break_above(self):
        assert compute_support_break_attempt_weight(3.50, 3.45) == 0.0

    def test_support_break_exact(self):
        assert compute_support_break_attempt_weight(3.45, 3.45) == 0.0

    def test_reclaim_extension(self):
        # (3.55 / 3.45 - 1.0) / 0.032 = 0.02899 / 0.032 = 0.906
        ext = compute_support_reclaim_extension(3.55, 3.45, 0.032)
        assert round(ext, 3) == 0.906

    def test_reclaim_extension_below_support(self):
        assert compute_support_reclaim_extension(3.40, 3.45, 0.032) == 0.0

    def test_reclaim_strength_strong(self):
        # extension >= 0.75 and above VWAP
        assert compute_support_reclaim_strength(0.90, 3.55, 3.45, 3.50) == 1.5

    def test_reclaim_strength_confirmed(self):
        # extension >= 0.25 and above VWAP
        assert compute_support_reclaim_strength(0.30, 3.55, 3.45, 3.50) == 1.25

    def test_reclaim_strength_basic(self):
        # above support but not above VWAP
        assert compute_support_reclaim_strength(0.30, 3.55, 3.45, 3.60) == 1.0

    def test_reclaim_strength_below_support(self):
        assert compute_support_reclaim_strength(0.0, 3.40, 3.45, 3.50) == 0.0

    def test_stabilization_strong(self):
        assert compute_stabilization_confirmation(3.55, 3.42, 3.50, 3.45) == 1.5

    def test_stabilization_vwap_only(self):
        assert compute_stabilization_confirmation(3.55, 3.60, 3.50, 3.45) == 1.25

    def test_stabilization_basic(self):
        assert compute_stabilization_confirmation(3.47, 3.60, 3.55, 3.45) == 1.0

    def test_stabilization_weak(self):
        assert compute_stabilization_confirmation(3.40, 3.60, 3.55, 3.45) == 0.5

    def test_volume_confirmation_tiers(self):
        assert compute_volume_confirmation(2.5) == 1.5
        assert compute_volume_confirmation(1.7) == 1.25
        assert compute_volume_confirmation(0.8) == 1.0
        assert compute_volume_confirmation(0.5) == 0.5

    def test_canonical_setup_from_spec(self):
        """EXPOSURE.md: deep dislocation with support break -> X_M5_setup = 3.0 (capped)."""
        x = min(2.5 * 1.25, X_M5_CAP)
        assert x == 3.0  # capped


# -----------------------------------------------------------------------
# Watchlist signal
# -----------------------------------------------------------------------

class TestM5Watchlist:
    def test_watchlist_fires(self):
        det = M5Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_watchlist_data(), lineage_hashes=["h"]))
        assert result.has_signal
        sig = result.signals[0]
        assert sig.signal_status == "watchlist"
        assert sig.route_class == RouteClass.B
        assert sig.raw_expected_edge == 0.0
        assert sig.direction == SignalDirection.LONG
        assert sig.signal_horizon == "7d"
        f = result.features.features
        assert f["activation_state"] == "watchlist"
        assert f["signal_generated"] is True
        assert f["decline_magnitude"] == 2.5
        assert f["support_break_attempted"] is True
        assert f["x_m5_setup"] > 0

    def test_watchlist_data_confidence_default(self):
        det = M5Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_watchlist_data(), lineage_hashes=["h"]))
        assert result.signals[0].data_confidence == 1.0


# -----------------------------------------------------------------------
# Activation signal
# -----------------------------------------------------------------------

class TestM5Activation:
    def test_activation_fires(self):
        det = M5Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_activation_data(), lineage_hashes=["h"]))
        assert result.has_signal
        sig = result.signals[0]
        assert sig.direction == SignalDirection.LONG
        assert sig.signal_horizon == "7d"
        assert sig.route_class == RouteClass.B
        assert sig.signal_status == "active"
        assert sig.raw_expected_edge > 0
        f = result.features.features
        assert f["activation_state"] == "activated"
        assert f["signal_generated"] is True
        assert f["activation_passed"] is True
        assert f["activation_identity_passed"] is True
        assert f["watchlist_identity_passed"] is True
        assert f["watchlist_session_match"] is True
        assert f["signal_freshness_source_passed"] is True
        assert f["support_reclaim_passed"] is True
        assert f["stabilization_passed"] is True
        assert f["volume_confirmation_passed"] is True
        assert f["x_m5_at_activation"] > 0
        assert "x_m5_activation" not in f

    def test_edge_deterministic(self):
        det = M5Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_activation_data(), lineage_hashes=["h"])
        r1, r2 = det.detect(inp), det.detect(inp)
        x_m5 = r1.features.features["x_m5_at_activation"]
        expected = round(x_m5 * LAMBDA_M5_7TD, 6)
        assert r1.signals[0].raw_expected_edge == expected
        assert r1.signals[0].raw_expected_edge == r2.signals[0].raw_expected_edge

    def test_strength_capped(self):
        det = M5Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_activation_data(), lineage_hashes=["h"]))
        x_m5 = result.features.features["x_m5_at_activation"]
        assert result.signals[0].raw_signal_strength == round(min(x_m5 / X_M5_CAP, 1.0), 6)

    def test_priors_logged(self):
        det = M5Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_activation_data(), lineage_hashes=["h"]))
        priors = result.features.features["expected_return_priors"]
        assert priors["gross_bps"] == round(result.signals[0].raw_expected_edge * 10_000, 2)
        assert result.features.features["lambda_M5_weekly"] == LAMBDA_M5_WEEKLY
        assert result.features.features["microcap_amplification"] == AMPLIFICATION
        assert result.features.features["amplified_lambda_M5_7td"] == round(LAMBDA_M5_7TD, 8)


# -----------------------------------------------------------------------
# No-signal / rejection cases
# -----------------------------------------------------------------------

class TestM5NoSignal:
    def test_no_setup(self):
        det = M5Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_no_setup_data(), lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_state"] == "no_setup"
        assert result.features.features["rejection_reason"] == "no_setup"
        assert result.features.features["signal_generated"] is False

    def test_not_operating_universe(self):
        det = M5Detector()
        data = _watchlist_data()
        data["operating_universe_inclusion"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.quality_flags["not_operating_universe_member"] is True
        assert result.features.features["signal_generated"] is False

    def test_missing_operating_universe_fails_closed(self):
        det = M5Detector()
        data = _watchlist_data()
        del data["operating_universe_inclusion"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.quality_flags["operating_universe_not_computed"] is True
        assert result.features.features["rejection_reason"] == "missing_operating_universe"

    def test_missing_required_fields(self):
        det = M5Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data={"sigma_20d": 0.03}, lineage_hashes=["h"]))
        assert result.features is None

    def test_support_reclaim_failed(self):
        det = M5Detector()
        data = _activation_data()
        data["price"] = 3.40  # below support
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_failure_reason"] == "support_reclaim_failed"
        assert result.features.features["signal_generated"] is False

    def test_stabilization_below_support_is_0_5(self):
        """Stabilization returns 0.5 when price <= support, but reclaim also fails first."""
        det = M5Detector()
        data = _activation_data()
        data["price"] = 3.44  # below support
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        f = result.features.features
        assert f["intraday_stabilization_confirmation"] == 0.5
        assert f["support_reclaim_passed"] is False
        assert f["activation_failure_reason"] == "support_reclaim_failed"
        assert f["signal_generated"] is False

    def test_volume_confirmation_failed(self):
        det = M5Detector()
        data = _activation_data()
        data["cumulative_session_volume"] = 50000
        data["expected_same_clock_volume_20d"] = 100000  # ratio 0.5 → vol_conf 0.5 < 1.0
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_failure_reason"] == "volume_confirmation_failed"
        assert result.features.features["signal_generated"] is False

    def test_missing_volume_data_fails_closed(self):
        det = M5Detector()
        data = _activation_data()
        del data["cumulative_session_volume"]
        del data["expected_same_clock_volume_20d"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        f = result.features.features
        assert not result.has_signal
        assert f["intraday_volume_confirmation"] == 0.0
        assert f["volume_confirmation_passed"] is False
        assert f["activation_failure_reason"] == "volume_confirmation_failed"
        assert result.quality_flags["missing_volume_data"] is True

    def test_invalid_volume_baseline_fails_closed(self):
        det = M5Detector()
        data = _activation_data()
        data["expected_same_clock_volume_20d"] = 0
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        f = result.features.features
        assert not result.has_signal
        assert f["intraday_volume_ratio"] is None
        assert f["intraday_volume_confirmation"] == 0.0
        assert f["activation_failure_reason"] == "volume_confirmation_failed"

    def test_missing_activation_identity_fails_closed(self):
        det = M5Detector()
        data = _activation_data()
        del data["activation_id"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_identity_passed"] is False
        assert result.features.features["activation_failure_reason"] == "activation_identity_missing"
        assert result.features.features["signal_generated"] is False

    def test_blank_activation_identity_fails_closed(self):
        det = M5Detector()
        data = _activation_data()
        data["activation_id"] = "  "
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_identity_passed"] is False

    def test_missing_quote_rejected(self):
        det = M5Detector()
        data = _activation_data()
        del data["candidate_eval_ask"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["quote_capture_passed"] is False
        assert result.features.features["activation_failure_reason"] == "quote_unavailable"
        assert result.features.features["signal_generated"] is False

    def test_wide_spread_rejected(self):
        det = M5Detector()
        data = _activation_data()
        data["spread_pct_vs_eval_quote"] = 0.01  # > 0.5% cap
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["spread_discipline_passed"] is False
        assert result.features.features["activation_failure_reason"] == "spread_too_wide"
        assert result.features.features["signal_generated"] is False

    def test_stale_signal_rejected(self):
        det = M5Detector()
        data = _activation_data()
        data["signal_freshness_passed"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["activation_failure_reason"] == "signal_expired"
        assert result.features.features["signal_generated"] is False

    def test_missing_freshness_fails_closed(self):
        det = M5Detector()
        data = _activation_data()
        del data["signal_freshness_passed"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["signal_freshness_passed"] is False
        assert result.features.features["activation_failure_reason"] == "signal_expired"

    def test_truthy_string_freshness_fails_closed(self):
        det = M5Detector()
        data = _activation_data()
        data["signal_freshness_passed"] = "true"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["signal_freshness_passed"] is False

    def test_missing_watchlist_signal_id_fails_closed(self):
        det = M5Detector()
        data = _activation_data()
        del data["watchlist_signal_id"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        f = result.features.features
        assert not result.has_signal
        assert f["watchlist_identity_passed"] is False
        assert f["signal_freshness_passed"] is False
        assert f["activation_failure_reason"] == "signal_expired"

    def test_watchlist_session_mismatch_fails_closed(self):
        det = M5Detector()
        data = _activation_data()
        data["watchlist_valid_session"] = "2026-05-19"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        f = result.features.features
        assert not result.has_signal
        assert f["watchlist_session_match"] is False
        assert f["signal_freshness_passed"] is False
        assert f["activation_failure_reason"] == "signal_expired"

    def test_missing_activation_market_quality_fields_fails_closed(self):
        det = M5Detector()
        data = _activation_data()
        del data["market_data_status"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        f = result.features.features
        assert not result.has_signal
        assert f["activation_passed"] is False
        assert f["activation_failure_reason"] == "missing_market_data_quality"
        assert f["rejection_reason"] == "missing_market_data_quality"
        assert f["activation_id"] == "m5-act-ACME-20260520-150000"
        assert f["watchlist_signal_id"] == "m5-watchlist-ACME-20260519"

    def test_no_support_break_no_setup(self):
        det = M5Detector()
        data = _watchlist_data()
        data["low_5d"] = 3.50  # above support — no break
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["support_break_attempted"] is False
        assert result.features.features["rejection_reason"] == "no_setup"


# -----------------------------------------------------------------------
# Quality flags and fidelity
# -----------------------------------------------------------------------

class TestM5Quality:
    def test_always_full_fidelity(self):
        det = M5Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_watchlist_data(), lineage_hashes=["h"]))
        assert result.features.fidelity_tier == FidelityTier.FULL

    def test_missing_lineage_warns(self):
        det = M5Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_watchlist_data(), lineage_hashes=[]))
        assert result.quality_flags.get("missing_lineage") is True
        assert result.has_signal

    def test_future_timestamp_warns(self):
        det = M5Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=datetime(2099, 1, 1, tzinfo=timezone.utc), market_data=_watchlist_data(), lineage_hashes=["h"]))
        assert result.quality_flags.get("future_timestamp") is True

    def test_filing_veto_status_default(self):
        det = M5Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_watchlist_data(), lineage_hashes=["h"]))
        assert result.features.features["filing_veto_status"] == "not_computed"

    def test_filing_veto_status_forwarded(self):
        det = M5Detector()
        data = _watchlist_data()
        data["filing_veto_status"] = "clear"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features.features["filing_veto_status"] == "clear"

    def test_filing_veto_status_forwarded_from_event_data(self):
        det = M5Detector()
        result = det.detect(PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_watchlist_data(),
            event_data={"filing_veto_status": "clear"},
            lineage_hashes=["h"],
        ))
        assert result.features.features["filing_veto_status"] == "clear"


# -----------------------------------------------------------------------
# Hashes
# -----------------------------------------------------------------------

class TestM5Hashes:
    def test_stable(self):
        det = M5Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_watchlist_data(), lineage_hashes=["h"])
        r1, r2 = det.detect(inp), det.detect(inp)
        assert r1.input_hashes == r2.input_hashes
        assert r1.output_hashes == r2.output_hashes

    def test_change_with_decline(self):
        det = M5Detector()
        d1 = _watchlist_data()
        d2 = _watchlist_data()
        d2["return_5d"] = -0.12
        r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d1, lineage_hashes=["h"]))
        r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d2, lineage_hashes=["h"]))
        assert r1.output_hashes != r2.output_hashes

    def test_output_hash_matches(self):
        det = M5Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_activation_data(), lineage_hashes=["h"]))
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

class TestM5EvidenceBridge:
    def _run_detection(self, db_session, *, market_data=None):
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session, provider="FMP", endpoint="/stable/historical-price-eod/full",
            asof_timestamp=_ts(), raw_payload={"ohlcv": "fixture"}, job_run_id=run.job_run_id,
        )
        det = M5Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data=market_data or _activation_data(),
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
        assert sig.pattern_id == "M5"
        assert sig.route_class == "B"
        assert sig.thesis_category == "mean_reversion"
        assert sig.signal_horizon == "7d"
        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        assert feat.feature_manifest_version == "m5-v1"

    def test_job_run_id(self, db_session):
        run, _, persisted = self._run_detection(db_session)
        assert db_session.get(FeatureSnapshot, persisted.feature_snapshot_id).job_run_id == run.job_run_id
        assert db_session.get(SignalRegistry, persisted.signal_ids[0]).job_run_id == run.job_run_id

    def test_no_signal_feature_only(self, db_session):
        _, _, persisted = self._run_detection(db_session, market_data=_no_setup_data())
        assert persisted.feature_snapshot_id is not None
        assert len(persisted.signal_ids) == 0

    def test_watchlist_persists(self, db_session):
        _, _, persisted = self._run_detection(db_session, market_data=_watchlist_data())
        assert len(persisted.signal_ids) == 1
        sig = db_session.get(SignalRegistry, persisted.signal_ids[0])
        assert sig.pattern_id == "M5"
        assert sig.route_class == "B"

    def test_feature_hash_deterministic(self, db_session):
        _, _, p1 = self._run_detection(db_session)
        _, _, p2 = self._run_detection(db_session)
        assert db_session.get(FeatureSnapshot, p1.feature_snapshot_id).feature_hash == db_session.get(FeatureSnapshot, p2.feature_snapshot_id).feature_hash
