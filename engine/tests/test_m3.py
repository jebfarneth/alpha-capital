"""
M3 Sector Rotation Beneficiary detector tests.

Vault contract verification:
  - Fires on top-3-decile sector rank with valid sector data
  - No signal below threshold, missing sector/rank, or non-universe
  - Tier classification is audit metadata only — does NOT modify edge
  - raw_expected_edge = X_M3 * lambda_M3_15td
  - signal_strength = sector_rank_normalized
  - Evidence bridge: pattern_id M3, route_class A, thesis continuation, horizon 15d
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
from alpha.patterns.m3 import (
    LAMBDA_M3_15TD,
    LAMBDA_M3_MONTHLY,
    LAMBDA_M3_MONTHLY_BASELINE,
    MIN_SECTOR_RANK,
    M3Detector,
    compute_sector_rank_normalized,
    compute_x_m3,
)


def _ts():
    return datetime(2026, 5, 20, 21, 0, 0, tzinfo=timezone.utc)


def _setup_run(db_session):
    job = create_job(db_session, name="m3_detector", job_type="detector", owner="pattern_engine")
    run = start_run(db_session, job_id=job.job_id)
    db_session.flush()
    return run


def _firing_data():
    """Top-3-decile sector, full diagnostics."""
    return {
        "sector": "Energy",
        "industry": "Oil & Gas E&P",
        "sector_return_6mo": 0.221,
        "sector_return_point_in_time_passed": True,
        "sector_return_formation_cohort_passed": True,
        "sector_history_coverage_years": 3,
        "sector_rank_normalized": 0.955,
        "sector_rank": 11,
        "n_sectors_in_universe": 11,
        "d1_decile": 8,
        "sigma_epsilon_decile": 6,
        "illiq_decile": 4,
        "hazard_score_at_signal": 15,
        "liquidity_score": 1.0,
        "market_cap_usd": 95_000_000,
        "price_at_signal": 8.45,
        "market_data_status": "current",
        "halt_status": "clear",
        "corporate_action_filter_passed": True,
        "operating_universe_inclusion": True,
    }


def _below_threshold_data():
    """Valid sector data but rank below top-3-decile."""
    data = _firing_data()
    data["sector_rank_normalized"] = 0.50
    data["sector_rank"] = 6
    return data


# -----------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------

class TestM3Metadata:
    def test_pattern_id(self):
        assert M3Detector().pattern_id == PatternId.M3

    def test_track(self):
        assert M3Detector().track == PatternTrack.MULTI_DAY

    def test_thesis_category(self):
        assert M3Detector().thesis_category == ThesisCategory.CONTINUATION

    def test_route_class(self):
        assert M3Detector().route_class == RouteClass.A

    def test_vault_constants(self):
        assert LAMBDA_M3_MONTHLY_BASELINE == 0.0043
        assert LAMBDA_M3_MONTHLY == 0.0075
        assert MIN_SECTOR_RANK == 0.70
        assert abs(LAMBDA_M3_15TD - 0.0075 * 15 / 21) < 1e-10


# -----------------------------------------------------------------------
# Formula helpers
# -----------------------------------------------------------------------

class TestFormulas:
    def test_rank_normalization(self):
        # rank=11, n=11: (11-0.5)/11 = 0.9545...
        assert abs(compute_sector_rank_normalized(11, 11) - 0.9545) < 0.001

    def test_rank_normalization_bottom(self):
        # rank=1, n=11: (1-0.5)/11 = 0.0454...
        assert abs(compute_sector_rank_normalized(1, 11) - 0.0454) < 0.001

    def test_rank_normalization_zero_sectors(self):
        assert compute_sector_rank_normalized(1, 0) is None

    def test_x_m3_top_decile(self):
        # rank_norm=0.955 -> X_M3 = 0.455
        assert abs(compute_x_m3(0.955) - 0.455) < 0.001

    def test_x_m3_median(self):
        assert compute_x_m3(0.5) == 0.0

    def test_x_m3_bottom_decile(self):
        assert abs(compute_x_m3(0.045) - (-0.455)) < 0.001


# -----------------------------------------------------------------------
# Firing cases
# -----------------------------------------------------------------------

class TestM3Firing:
    def test_fires_with_complete_data(self):
        det = M3Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.has_signal
        sig = result.signals[0]
        assert sig.direction == SignalDirection.LONG
        assert sig.signal_horizon == "15d"
        assert sig.route_class == RouteClass.A
        assert sig.raw_expected_edge > 0
        f = result.features.features
        assert f["signal_generated"] is True
        assert f["sector"] == "Energy"
        assert f["sector_rank_normalized"] == 0.955
        assert f["exposure_x_m3_t0"] > 0
        assert f["sector_return_point_in_time_passed"] is True

    def test_edge_deterministic(self):
        det = M3Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"])
        r1, r2 = det.detect(inp), det.detect(inp)
        assert r1.signals[0].raw_expected_edge == r2.signals[0].raw_expected_edge
        x_m3 = r1.features.features["exposure_x_m3_t0"]
        expected = round(x_m3 * LAMBDA_M3_15TD, 6)
        assert r1.signals[0].raw_expected_edge == expected

    def test_signal_strength_is_rank_normalized(self):
        det = M3Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.signals[0].raw_signal_strength == round(0.955, 6)

    def test_priors_logged(self):
        det = M3Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        f = result.features.features
        assert f["lambda_M3_monthly"] == LAMBDA_M3_MONTHLY
        assert f["lambda_M3_monthly_baseline"] == LAMBDA_M3_MONTHLY_BASELINE
        assert f["validated_or_shadow_lambda_M3_15td"] == LAMBDA_M3_15TD
        assert f["lambda_M3_default_15td"] == round(LAMBDA_M3_15TD, 8)
        assert f["lambda_M3_source"] == "shadow_prior"
        priors = f["expected_return_priors"]
        assert priors["gross_bps"] == round(result.signals[0].raw_expected_edge * 10_000, 2)

    def test_custom_lambda_injection(self):
        det = M3Detector(lambda_m3_15td=0.01)
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        default = M3Detector().detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.signals[0].raw_expected_edge > default.signals[0].raw_expected_edge
        f = result.features.features
        assert f["validated_or_shadow_lambda_M3_15td"] == 0.01
        assert f["lambda_M3_source"] == "validated_or_injected"

    def test_invalid_lambda_rejected(self):
        with pytest.raises(ValueError, match="lambda_m3_15td"):
            M3Detector(lambda_m3_15td=float("nan"))

    def test_zero_lambda_rejected(self):
        with pytest.raises(ValueError, match="lambda_m3_15td"):
            M3Detector(lambda_m3_15td=0.0)

    def test_data_confidence_default(self):
        det = M3Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.signals[0].data_confidence == 1.0

    def test_tier_does_not_modify_edge(self):
        """Tier classification is audit metadata — same X_M3 should produce same edge regardless of tier."""
        det = M3Detector()
        data_hc = _firing_data()
        data_hc["d1_decile"] = 10
        data_hc["sector_rank_normalized"] = 0.85
        data_hc["sector_rank"] = 9
        data_hc["n_sectors_in_universe"] = 10
        data_default = _firing_data()
        data_default["d1_decile"] = 5
        data_default["sector_rank_normalized"] = 0.85
        data_default["sector_rank"] = 9
        data_default["n_sectors_in_universe"] = 10
        r_hc = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data_hc, lineage_hashes=["h"]))
        r_def = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data_default, lineage_hashes=["h"]))
        assert r_hc.features.features["expected_return_tier"] == "high_conviction"
        assert r_def.features.features["expected_return_tier"] == "default"
        assert r_hc.signals[0].raw_expected_edge == r_def.signals[0].raw_expected_edge

    def test_tier_reads_d1_from_all_data_sources(self):
        det = M3Detector()
        data = _firing_data()
        data.pop("d1_decile")
        data["sector_rank_normalized"] = 0.85
        data["sector_rank"] = 9
        data["n_sectors_in_universe"] = 10
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=data,
            fundamental_data={"d1_decile": 10}, lineage_hashes=["h"],
        ))
        assert result.has_signal
        assert result.features.features["d1_decile"] == 10
        assert result.features.features["expected_return_tier"] == "high_conviction"

    def test_rank_from_sector_rank_and_n_sectors(self):
        """Falls back to computing rank_normalized from sector_rank + n_sectors_in_universe."""
        det = M3Detector()
        data = _firing_data()
        del data["sector_rank_normalized"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        # (11 - 0.5) / 11 = 0.954545...
        assert abs(result.features.features["sector_rank_normalized"] - 0.9545) < 0.001

    def test_exact_boundary_fires(self):
        det = M3Detector()
        data = _firing_data()
        data["sector_rank_normalized"] = 0.70
        del data["sector_rank"]
        del data["n_sectors_in_universe"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal

    def test_audit_metadata_present(self):
        det = M3Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        meta = result.features.features["expected_return_priors_audit_metadata"]
        assert "note" in meta
        assert "tier_default_bps_range" in meta


# -----------------------------------------------------------------------
# No-signal / rejection cases
# -----------------------------------------------------------------------

class TestM3NoSignal:
    def test_sector_rank_below_threshold(self):
        det = M3Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_below_threshold_data(), lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "sector_rank_below_threshold"
        assert result.features.features["signal_generated"] is False
        assert result.features.features["exposure_x_m3_t0"] == 0.0

    def test_just_below_boundary_rejected(self):
        det = M3Detector()
        data = _firing_data()
        data["sector_rank_normalized"] = 0.699999
        del data["sector_rank"]
        del data["n_sectors_in_universe"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "sector_rank_below_threshold"

    def test_bottom_decile_no_short_signal(self):
        det = M3Detector()
        data = _firing_data()
        data["sector_rank_normalized"] = 0.05
        del data["sector_rank"]
        del data["n_sectors_in_universe"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "sector_rank_below_threshold"
        assert result.features.features["exposure_x_m3_t0"] < 0

    def test_missing_sector(self):
        det = M3Detector()
        data = _firing_data()
        del data["sector"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features is None

    def test_blank_sector_rejected_as_missing(self):
        det = M3Detector()
        for blank in ("", "   "):
            data = _firing_data()
            data["sector"] = blank
            result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
            assert not result.has_signal
            assert result.features is None

    def test_sector_identity_is_trimmed(self):
        det = M3Detector()
        data = _firing_data()
        data["sector"] = "  Energy  "
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["sector"] == "Energy"

    def test_missing_sector_return(self):
        det = M3Detector()
        data = _firing_data()
        del data["sector_return_6mo"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features is None

    def test_missing_rank_evidence(self):
        det = M3Detector()
        data = _firing_data()
        del data["sector_rank_normalized"]
        del data["sector_rank"]
        del data["n_sectors_in_universe"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features is None

    def test_missing_point_in_time_sector_return_proof_rejected(self):
        det = M3Detector()
        data = _firing_data()
        del data["sector_return_point_in_time_passed"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "sector_return_not_point_in_time"
        assert result.features.point_in_time_passed is False
        assert result.features.fidelity_tier == FidelityTier.LITE

    def test_false_point_in_time_sector_return_proof_rejected(self):
        det = M3Detector()
        data = _firing_data()
        data["sector_return_point_in_time_passed"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "sector_return_not_point_in_time"
        assert result.quality_flags["point_in_time_passed"] is False

    def test_false_formation_cohort_proof_rejected(self):
        det = M3Detector()
        data = _firing_data()
        data["sector_return_point_in_time_passed"] = True
        data["sector_return_formation_cohort_passed"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "sector_return_not_point_in_time"
        assert result.quality_flags["point_in_time_passed"] is False
        assert result.quality_flags["formation_cohort_passed"] is False

    def test_string_sector_return_no_crash(self):
        det = M3Detector()
        data = _firing_data()
        data["sector_return_6mo"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features is None

    def test_string_rank_normalized_no_crash(self):
        det = M3Detector()
        data = _firing_data()
        data["sector_rank_normalized"] = "N/A"
        del data["sector_rank"]
        del data["n_sectors_in_universe"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_sector_rank"

    def test_string_n_sectors_no_crash(self):
        det = M3Detector()
        data = _firing_data()
        del data["sector_rank_normalized"]
        data["n_sectors_in_universe"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_sector_rank"

    def test_rank_out_of_range_rejected(self):
        det = M3Detector()
        data = _firing_data()
        data["sector_rank_normalized"] = 1.5
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_sector_rank"

    def test_inconsistent_rank_evidence_rejected(self):
        det = M3Detector()
        data = _firing_data()
        data["sector_rank_normalized"] = 0.955
        data["sector_rank"] = 1
        data["n_sectors_in_universe"] = 11
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "inconsistent_sector_rank"
        assert result.quality_flags["inconsistent_sector_rank"] is True

    def test_not_operating_universe(self):
        det = M3Detector()
        data = _firing_data()
        data["operating_universe_inclusion"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.quality_flags["not_operating_universe_member"] is True
        assert result.features.features["rejection_reason"] == "not_operating_universe"
        assert result.features.features["sector"] == "Energy"

    def test_missing_operating_universe_fails_closed(self):
        det = M3Detector()
        data = _firing_data()
        del data["operating_universe_inclusion"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "missing_operating_universe"

    def test_delayed_market_data_rejected(self):
        det = M3Detector()
        data = _firing_data()
        data["market_data_status"] = "delayed"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "data_delay"

    def test_missing_market_data_status_does_not_block(self):
        det = M3Detector()
        data = _firing_data()
        del data["market_data_status"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal  # not blocked

    def test_high_hazard_does_not_block_signal(self):
        det = M3Detector()
        data = _firing_data()
        data["hazard_score_at_signal"] = 90
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal  # hazard is candidate-stage only
        assert result.features.features["hazard_score_at_signal"] == 90

    def test_filing_veto_does_not_block_signal(self):
        det = M3Detector()
        data = _firing_data()
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=data,
            event_data={"filing_veto_status": "blocked"},
            lineage_hashes=["h"],
        ))
        assert result.has_signal
        assert result.features.features["filing_veto_status"] == "blocked"

    def test_liquidity_zero_does_not_block_signal(self):
        det = M3Detector()
        data = _firing_data()
        data["liquidity_score"] = 0
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["liquidity_score"] == 0


# -----------------------------------------------------------------------
# Quality and diagnostics
# -----------------------------------------------------------------------

class TestM3Quality:
    def test_always_full_fidelity(self):
        det = M3Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.features.fidelity_tier == FidelityTier.FULL

    def test_missing_lineage_warns(self):
        det = M3Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=[]))
        assert result.quality_flags.get("missing_lineage") is True
        assert result.has_signal

    def test_future_timestamp_warns(self):
        det = M3Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=datetime(2099, 1, 1, tzinfo=timezone.utc), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.quality_flags.get("future_timestamp") is True

    def test_diagnostic_fields_from_all_sources(self):
        det = M3Detector()
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(),
            fundamental_data={"market_cap_usd": 95_000_000, "sub_universe": "B"},
            event_data={"filing_veto_status": "clear", "overlapping_pattern_ids": ["M4"]},
            lineage_hashes=["h"],
        ))
        f = result.features.features
        assert f["market_cap_usd"] == 95_000_000
        assert f["sub_universe"] == "B"
        assert f["filing_veto_status"] == "clear"
        assert f["overlapping_pattern_ids"] == ["M4"]

    def test_load_bearing_sector_fields_are_market_data_authoritative(self):
        det = M3Detector()
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(),
            fundamental_data={
                "sector": "Utilities",
                "sector_return_6mo": -0.50,
                "sector_rank_normalized": 0.05,
                "sector_return_point_in_time_passed": False,
            },
            event_data={
                "sector": "Real Estate",
                "sector_rank": 1,
                "n_sectors_in_universe": 11,
                "sector_return_formation_cohort_passed": False,
            },
            lineage_hashes=["h"],
        ))
        assert result.has_signal
        f = result.features.features
        assert f["sector"] == "Energy"
        assert f["sector_return_6mo"] == 0.221
        assert f["sector_rank_normalized"] == 0.955
        assert f["sector_rank"] == 11
        assert f["sector_return_point_in_time_passed"] is True
        assert f["sector_return_formation_cohort_passed"] is True

    def test_overlap_diagnostics(self):
        det = M3Detector()
        data = _firing_data()
        data["m4_also_firing"] = True
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features.features["m4_also_firing"] is True

    def test_filing_veto_defaults_not_computed(self):
        det = M3Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.features.features["filing_veto_status"] == "not_computed"

    def test_sector_taxonomy_source_defaults(self):
        det = M3Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.features.features["sector_taxonomy_source"] == "FMP"

    def test_field_confidence_affects_data_confidence(self):
        det = M3Detector()
        data = _firing_data()
        data["field_confidence"] = {"sector_return": 0.5}
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=data,
            fundamental_data={"field_confidence": {"fundamentals": 0.8}},
            lineage_hashes=["h"],
        ))
        assert result.has_signal
        assert result.signals[0].data_confidence < 1.0


# -----------------------------------------------------------------------
# Hashes
# -----------------------------------------------------------------------

class TestM3Hashes:
    def test_stable(self):
        det = M3Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"])
        r1, r2 = det.detect(inp), det.detect(inp)
        assert r1.input_hashes == r2.input_hashes
        assert r1.output_hashes == r2.output_hashes

    def test_change_with_sector_return(self):
        det = M3Detector()
        d1 = _firing_data()
        d2 = _firing_data()
        d2["sector_return_6mo"] = 0.30
        r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d1, lineage_hashes=["h"]))
        r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d2, lineage_hashes=["h"]))
        assert r1.output_hashes != r2.output_hashes

    def test_input_hash_changes_with_event_data(self):
        det = M3Detector()
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
        r1 = M3Detector().detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        r2 = M3Detector(lambda_m3_15td=0.01).detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert r1.output_hashes != r2.output_hashes

    def test_output_hash_matches(self):
        det = M3Detector()
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

class TestM3EvidenceBridge:
    def _run_detection(self, db_session, *, market_data=None):
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session, provider="FMP", endpoint="/stable/sector-returns",
            asof_timestamp=_ts(), raw_payload={"sector": "fixture"}, job_run_id=run.job_run_id,
        )
        det = M3Detector()
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
        assert sig.pattern_id == "M3"
        assert sig.route_class == "A"
        assert sig.thesis_category == "continuation"
        assert sig.signal_horizon == "15d"
        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        assert feat.feature_manifest_version == "m3-v1"

    def test_job_run_id(self, db_session):
        run, _, persisted = self._run_detection(db_session)
        assert db_session.get(FeatureSnapshot, persisted.feature_snapshot_id).job_run_id == run.job_run_id
        assert db_session.get(SignalRegistry, persisted.signal_ids[0]).job_run_id == run.job_run_id

    def test_universe_snapshot_id(self, db_session):
        run = _setup_run(db_session)
        usn = record_universe_snapshot(db_session, ticker="ACME", asof_timestamp=_ts(), operating_universe_inclusion=True, job_run_id=run.job_run_id)
        lineage = record_data_lineage(db_session, provider="FMP", endpoint="/stable/sector-returns", asof_timestamp=_ts(), raw_payload={"x": 1}, job_run_id=run.job_run_id)
        det = M3Detector()
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
