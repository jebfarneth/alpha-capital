"""
M2 Insider Cluster detector tests.

Vault contract verification:
  - Fires on >= 2 classified opportunistic buyers within 30d, filing <= 20d stale
  - Filing detection date drives decay, NOT transaction date
  - No signal with < 2 opp buyers, stale filing, or non-universe
  - Routine/unclassifiable-only clusters do NOT satisfy V1 signal gate
  - raw_expected_edge = X_M2 * lambda_M2_20td
  - signal_strength = min(X_M2 / 3.0, 1.0)
  - Evidence bridge: pattern_id M2, route_class A, thesis right_tail_convex, horizon 20d
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

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
from alpha.patterns.m2 import (
    LAMBDA_M2_20TD,
    LAMBDA_M2_MONTHLY,
    MAX_DAYS_SINCE_LAST_FILING,
    MIN_OPP_BUYERS,
    X_M2_CAP,
    M2Detector,
    compute_x_m2,
)


def _ts():
    return datetime(2026, 5, 20, 21, 0, 0, tzinfo=timezone.utc)


def _setup_run(db_session):
    job = create_job(db_session, name="m2_detector", job_type="detector", owner="pattern_engine")
    run = start_run(db_session, job_id=job.job_id)
    db_session.flush()
    return run


def _firing_data():
    """Two opportunistic buyers, fresh filing, full diagnostics."""
    return {
        "n_distinct_opp_buyers_30d": 3,
        "days_since_last_opp_buy_filing_detected": 2,
        "mean_trade_size_weight": 1.25,
        "mean_locality_weight": 1.2,
        "days_since_last_opp_buy_transaction": 5,
        "cluster_window_days": 30,
        "source_authority": "sec_edgar",
        "sec_accession_numbers": ["0001234567-26-000001", "0001234567-26-000002"],
        "sec_fmp_mismatch": False,
        "n_routine_only_buyers_30d": 0,
        "n_unclassifiable_only_buyers_30d": 1,
        "hazard_score_at_signal": 15,
        "liquidity_score": 1.0,
        "market_cap_usd": 75_000_000,
        "price_at_signal": 6.20,
        "sector": "Technology",
        "market_data_status": "current",
        "halt_status": "clear",
        "corporate_action_filter_passed": True,
        "operating_universe_inclusion": True,
    }


# -----------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------

class TestM2Metadata:
    def test_pattern_id(self):
        assert M2Detector().pattern_id == PatternId.M2

    def test_track(self):
        assert M2Detector().track == PatternTrack.MULTI_DAY

    def test_thesis_category(self):
        assert M2Detector().thesis_category == ThesisCategory.RIGHT_TAIL_CONVEX

    def test_route_class(self):
        assert M2Detector().route_class == RouteClass.A

    def test_vault_constants(self):
        assert LAMBDA_M2_MONTHLY == 0.0082
        assert MIN_OPP_BUYERS == 2
        assert MAX_DAYS_SINCE_LAST_FILING == 20
        assert X_M2_CAP == 3.0
        assert abs(LAMBDA_M2_20TD - 0.0082 * 20 / 21) < 1e-10


# -----------------------------------------------------------------------
# Formula helpers
# -----------------------------------------------------------------------

class TestFormulas:
    def test_x_m2_canonical_baseline(self):
        """Two opp buyers, day 0, intensity 1.0 -> near 1.0."""
        x = compute_x_m2(0, 2, 1.0)
        expected = math.exp(0) * math.log(3) / math.log(3) * 1.0  # = 1.0
        assert abs(x - expected) < 0.001

    def test_x_m2_decays_with_age(self):
        x0 = compute_x_m2(0, 2, 1.0)
        x5 = compute_x_m2(5, 2, 1.0)
        x10 = compute_x_m2(10, 2, 1.0)
        assert x0 > x5 > x10 > 0

    def test_x_m2_increases_with_buyers(self):
        x2 = compute_x_m2(0, 2, 1.0)
        x5 = compute_x_m2(0, 5, 1.0)
        assert x5 > x2

    def test_x_m2_capped_at_3(self):
        x = compute_x_m2(0, 10, 5.0)
        assert x == X_M2_CAP

    def test_x_m2_zero_buyers(self):
        assert compute_x_m2(0, 0, 1.0) == 0.0

    def test_x_m2_negative_days(self):
        assert compute_x_m2(-1, 2, 1.0) == 0.0

    def test_x_m2_zero_intensity(self):
        assert compute_x_m2(0, 2, 0.0) == 0.0

    def test_x_m2_one_buyer(self):
        x = compute_x_m2(0, 1, 1.0)
        assert x > 0  # log(2)/log(3) ≈ 0.63


# -----------------------------------------------------------------------
# Firing cases
# -----------------------------------------------------------------------

class TestM2Firing:
    def test_fires_with_complete_data(self):
        det = M2Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.has_signal
        sig = result.signals[0]
        assert sig.direction == SignalDirection.LONG
        assert sig.signal_horizon == "20d"
        assert sig.route_class == RouteClass.A
        assert sig.raw_expected_edge > 0
        f = result.features.features
        assert f["signal_generated"] is True
        assert f["exposure_x_m2"] > 0
        assert f["n_distinct_opp_buyers_30d"] == 3

    def test_edge_deterministic(self):
        det = M2Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"])
        r1, r2 = det.detect(inp), det.detect(inp)
        assert r1.signals[0].raw_expected_edge == r2.signals[0].raw_expected_edge
        x = r1.features.features["exposure_x_m2"]
        expected = round(x * LAMBDA_M2_20TD, 6)
        assert r1.signals[0].raw_expected_edge == expected

    def test_signal_strength_capped(self):
        det = M2Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        x = result.features.features["exposure_x_m2"]
        assert result.signals[0].raw_signal_strength == round(min(x / X_M2_CAP, 1.0), 6)

    def test_priors_logged(self):
        det = M2Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        f = result.features.features
        assert f["lambda_M2_monthly"] == LAMBDA_M2_MONTHLY
        assert f["validated_or_shadow_lambda_M2_20td"] == LAMBDA_M2_20TD
        assert f["lambda_M2_source"] == "shadow_prior"
        assert f["expected_return_priors"]["gross_bps"] == round(result.signals[0].raw_expected_edge * 10_000, 2)

    def test_custom_lambda(self):
        det = M2Detector(lambda_m2_20td=0.015)
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.features.features["lambda_M2_source"] == "validated_or_injected"
        assert result.features.features["validated_or_shadow_lambda_M2_20td"] == 0.015

    def test_invalid_lambda_rejected(self):
        with pytest.raises(ValueError, match="lambda_m2_20td"):
            M2Detector(lambda_m2_20td=float("nan"))

    def test_zero_lambda_rejected(self):
        with pytest.raises(ValueError, match="lambda_m2_20td"):
            M2Detector(lambda_m2_20td=0.0)

    def test_exact_2_buyers_fires(self):
        det = M2Detector()
        data = _firing_data()
        data["n_distinct_opp_buyers_30d"] = 2
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal

    def test_exact_20_day_boundary_fires(self):
        det = M2Detector()
        data = _firing_data()
        data["days_since_last_opp_buy_filing_detected"] = 20
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal

    def test_filing_lag_warning(self):
        det = M2Detector()
        data = _firing_data()
        data["days_since_last_opp_buy_filing_detected"] = 2
        data["days_since_last_opp_buy_transaction"] = 10
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.quality_flags.get("large_filing_lag") is True
        assert "filing lag 8d" in result.warnings[0]

    def test_data_confidence_default(self):
        det = M2Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.signals[0].data_confidence == 1.0

    def test_missing_intensity_defaults_neutral(self):
        det = M2Detector()
        data = _firing_data()
        del data["mean_trade_size_weight"]
        del data["mean_locality_weight"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.quality_flags["missing_trade_intensity"] is True
        assert result.features.features["mean_trade_intensity_weight"] == 1.0
        assert result.signals[0].data_confidence < 1.0

    def test_vault_size_and_locality_weights_drive_intensity(self):
        det = M2Detector()
        data = _firing_data()
        data["mean_trade_size_weight"] = 1.18
        data["mean_locality_weight"] = 1.08
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["mean_trade_intensity_weight"] == round(1.18 * 1.08, 6)

    def test_legacy_mean_intensity_fallback(self):
        det = M2Detector()
        data = _firing_data()
        del data["mean_trade_size_weight"]
        del data["mean_locality_weight"]
        data["mean_trade_intensity_weight"] = 1.4
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["mean_trade_intensity_weight"] == 1.4


# -----------------------------------------------------------------------
# No-signal / rejection cases
# -----------------------------------------------------------------------

class TestM2NoSignal:
    def test_1_buyer_rejected(self):
        det = M2Detector()
        data = _firing_data()
        data["n_distinct_opp_buyers_30d"] = 1
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "insufficient_opportunistic_buyers"

    def test_0_buyers_rejected(self):
        det = M2Detector()
        data = _firing_data()
        data["n_distinct_opp_buyers_30d"] = 0
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "insufficient_opportunistic_buyers"

    def test_stale_filing_rejected(self):
        det = M2Detector()
        data = _firing_data()
        data["days_since_last_opp_buy_filing_detected"] = 21
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "filing_too_stale"

    def test_negative_days_rejected(self):
        det = M2Detector()
        data = _firing_data()
        data["days_since_last_opp_buy_filing_detected"] = -1
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_filing_age"

    def test_routine_only_cluster_rejected(self):
        """Routine-only buyers do not satisfy V1 opportunistic gate."""
        det = M2Detector()
        data = _firing_data()
        data["n_distinct_opp_buyers_30d"] = 0
        data["n_routine_only_buyers_30d"] = 3
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "insufficient_opportunistic_buyers"

    def test_unclassifiable_only_cluster_rejected(self):
        det = M2Detector()
        data = _firing_data()
        data["n_distinct_opp_buyers_30d"] = 0
        data["n_unclassifiable_only_buyers_30d"] = 4
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal

    def test_transaction_date_does_not_drive_decay(self):
        """Filing detection date drives decay. A fresh filing with old transaction is still fresh."""
        det = M2Detector()
        data = _firing_data()
        data["days_since_last_opp_buy_filing_detected"] = 1  # fresh filing
        data["days_since_last_opp_buy_transaction"] = 15  # old transaction
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["days_since_last_opp_buy_filing_detected"] == 1
        assert result.features.features["days_since_last_opp_buy_transaction"] == 15

    def test_missing_cluster_data_no_features(self):
        det = M2Detector()
        data = {"operating_universe_inclusion": True}
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features is None

    def test_string_buyers_no_crash(self):
        det = M2Detector()
        data = _firing_data()
        data["n_distinct_opp_buyers_30d"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features is None

    def test_string_days_no_crash(self):
        det = M2Detector()
        data = _firing_data()
        data["days_since_last_opp_buy_filing_detected"] = "pending"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features is None

    def test_string_intensity_no_crash(self):
        det = M2Detector()
        data = _firing_data()
        data["mean_trade_size_weight"] = "N/A"
        data["mean_locality_weight"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal  # defaults to 1.0

    def test_invalid_cluster_window_rejected(self):
        det = M2Detector()
        data = _firing_data()
        data["cluster_window_days"] = 45
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_cluster_window"
        assert result.quality_flags["cluster_window_mismatch"] is True

    def test_missing_cluster_window_haircuts_confidence(self):
        det = M2Detector()
        data = _firing_data()
        del data["cluster_window_days"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.quality_flags["missing_cluster_window_proof"] is True
        assert result.signals[0].data_confidence < 1.0

    def test_invalid_source_authority_rejected(self):
        det = M2Detector()
        data = _firing_data()
        data["source_authority"] = "manual"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_source_authority"

    def test_sec_source_requires_accession_numbers(self):
        det = M2Detector()
        data = _firing_data()
        data["sec_accession_numbers"] = []
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "missing_sec_accession"

    def test_fmp_backfill_requires_accession_and_haircuts_confidence(self):
        det = M2Detector()
        data = _firing_data()
        data["source_authority"] = "fmp_backfill"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.quality_flags["fmp_backfill_authority"] is True
        assert result.signals[0].data_confidence < 1.0

    def test_sec_fmp_mismatch_haircuts_confidence(self):
        det = M2Detector()
        data = _firing_data()
        data["sec_fmp_mismatch"] = True
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.quality_flags["sec_fmp_mismatch"] is True
        assert result.signals[0].data_confidence < 1.0

    def test_not_operating_universe(self):
        det = M2Detector()
        data = _firing_data()
        data["operating_universe_inclusion"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "not_operating_universe"
        assert result.features.features["n_distinct_opp_buyers_30d"] == 3

    def test_missing_operating_universe_fails_closed(self):
        det = M2Detector()
        data = _firing_data()
        del data["operating_universe_inclusion"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "missing_operating_universe"

    def test_delayed_market_data_rejected(self):
        det = M2Detector()
        data = _firing_data()
        data["market_data_status"] = "delayed"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "data_delay"

    def test_hazard_does_not_block_signal(self):
        det = M2Detector()
        data = _firing_data()
        data["hazard_score_at_signal"] = 90
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["hazard_score_at_signal"] == 90

    def test_event_diagnostics_do_not_overwrite_cluster(self):
        """event_data should not clobber load-bearing cluster fields from market_data."""
        det = M2Detector()
        data = _firing_data()
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=data,
            event_data={"n_distinct_opp_buyers_30d": 999, "source_authority": "manual"},
            lineage_hashes=["h"],
        ))
        assert result.has_signal
        # Load-bearing field comes from market_data, not event_data
        assert result.features.features["n_distinct_opp_buyers_30d"] == 3
        assert result.features.features["source_authority"] == "sec_edgar"


# -----------------------------------------------------------------------
# Quality and diagnostics
# -----------------------------------------------------------------------

class TestM2Quality:
    def test_always_full_fidelity(self):
        det = M2Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.features.fidelity_tier == FidelityTier.FULL

    def test_missing_lineage_warns(self):
        det = M2Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=[]))
        assert result.quality_flags.get("missing_lineage") is True
        assert result.has_signal

    def test_future_timestamp_warns(self):
        det = M2Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=datetime(2099, 1, 1, tzinfo=timezone.utc), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.quality_flags.get("future_timestamp") is True

    def test_diagnostic_fields_from_all_sources(self):
        det = M2Detector()
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(),
            fundamental_data={"market_cap_usd": 75_000_000, "sub_universe": "A"},
            event_data={"filing_veto_status": "clear", "m1_also_firing": True, "overlapping_pattern_ids": ["M1"]},
            lineage_hashes=["h"],
        ))
        f = result.features.features
        assert f["market_cap_usd"] == 75_000_000
        assert f["filing_veto_status"] == "clear"
        assert f["m1_also_firing"] is True
        assert f["overlapping_pattern_ids"] == ["M1"]

    def test_filing_veto_defaults_not_computed(self):
        det = M2Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.features.features["filing_veto_status"] == "not_computed"

    def test_field_confidence_affects_data_confidence(self):
        det = M2Detector()
        data = _firing_data()
        data["field_confidence"] = {"form4": 0.5}
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=data,
            fundamental_data={"field_confidence": {"fun": 0.9}},
            lineage_hashes=["h"],
        ))
        assert result.has_signal
        assert result.signals[0].data_confidence < 1.0


# -----------------------------------------------------------------------
# Hashes
# -----------------------------------------------------------------------

class TestM2Hashes:
    def test_stable(self):
        det = M2Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"])
        r1, r2 = det.detect(inp), det.detect(inp)
        assert r1.input_hashes == r2.input_hashes
        assert r1.output_hashes == r2.output_hashes

    def test_change_with_cluster(self):
        det = M2Detector()
        d1 = _firing_data()
        d2 = _firing_data()
        d2["n_distinct_opp_buyers_30d"] = 5
        r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d1, lineage_hashes=["h"]))
        r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d2, lineage_hashes=["h"]))
        assert r1.output_hashes != r2.output_hashes

    def test_input_hash_changes_with_event_data(self):
        det = M2Detector()
        r1 = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(),
            event_data={"filing_veto_status": "not_computed"}, lineage_hashes=["h"],
        ))
        r2 = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(),
            event_data={"filing_veto_status": "clear"}, lineage_hashes=["h"],
        ))
        assert r1.input_hashes != r2.input_hashes

    def test_output_hash_changes_with_lambda(self):
        r1 = M2Detector().detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        r2 = M2Detector(lambda_m2_20td=0.015).detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert r1.output_hashes != r2.output_hashes

    def test_output_hash_matches(self):
        det = M2Detector()
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

class TestM2EvidenceBridge:
    def _run_detection(self, db_session, *, market_data=None):
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session, provider="SEC", endpoint="/edgar/form4",
            asof_timestamp=_ts(), raw_payload={"form4": "fixture"}, job_run_id=run.job_run_id,
        )
        det = M2Detector()
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
        assert sig.pattern_id == "M2"
        assert sig.route_class == "A"
        assert sig.thesis_category == "right_tail_convex"
        assert sig.signal_horizon == "20d"
        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        assert feat.feature_manifest_version == "m2-v1"

    def test_job_run_id(self, db_session):
        run, _, persisted = self._run_detection(db_session)
        assert db_session.get(FeatureSnapshot, persisted.feature_snapshot_id).job_run_id == run.job_run_id
        assert db_session.get(SignalRegistry, persisted.signal_ids[0]).job_run_id == run.job_run_id

    def test_no_signal_feature_only(self, db_session):
        data = _firing_data()
        data["n_distinct_opp_buyers_30d"] = 1
        _, _, persisted = self._run_detection(db_session, market_data=data)
        assert persisted.feature_snapshot_id is not None
        assert len(persisted.signal_ids) == 0

    def test_feature_hash_deterministic(self, db_session):
        _, _, p1 = self._run_detection(db_session)
        _, _, p2 = self._run_detection(db_session)
        assert db_session.get(FeatureSnapshot, p1.feature_snapshot_id).feature_hash == db_session.get(FeatureSnapshot, p2.feature_snapshot_id).feature_hash
