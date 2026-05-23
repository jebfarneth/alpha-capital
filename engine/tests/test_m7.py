"""
M7 Pure Technical Multi-Day detector tests.

Vault contract verification:
  - Fires on top-decile predicted return with reproducible model lineage
  - RC confidence status is NOT a signal gate
  - Decay haircut embedded in X_M7, not applied again in lambda
  - signal_strength = predicted_return_rank_pct, not X_M7
  - Evidence bridge: pattern_id M7, route_class A, thesis continuation, horizon 10d
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
from alpha.patterns.m7 import (
    DECAY_HAIRCUT_MAX,
    DECAY_HAIRCUT_MIN,
    LAMBDA_M7_10TD,
    LAMBDA_M7_MONTHLY,
    MICROCAP_AMPLIFICATION,
    MIN_RANK_PCT,
    M7Detector,
    compute_decay_haircut,
)


def _ts():
    return datetime(2026, 5, 20, 21, 0, 0, tzinfo=timezone.utc)


def _setup_run(db_session):
    job = create_job(db_session, name="m7_detector", job_type="detector", owner="pattern_engine")
    run = start_run(db_session, job_id=job.job_id)
    db_session.flush()
    return run


def _firing_data():
    """Top-decile prediction, full lineage, fresh model."""
    return {
        "predicted_return_rank_pct": 0.95,
        "predicted_return": 0.042,
        "decay_haircut": 0.63,
        "days_since_retrain": 5,
        "model_version": "m7-gbrt-v1.0",
        "training_run_id": "train-2026Q1-001",
        "prediction_run_id": "pred-20260520-001",
        "feature_snapshot_id": "feat-ACME-20260520",
        "feature_manifest_version": "m7-features-v1",
        "model_artifact_hash": "sha256:abc123def456",
        "data_cutoff_timestamp": "2026-05-20T20:00:00Z",
        "point_in_time_passed": True,
        "prediction_run_status": "completed",
        "is_canonical": True,
        "feature_count": 31,
        "missing_feature_count": 2,
        "rc_confidence_status": "pending",
        "hazard_score_at_signal": 12,
        "liquidity_score": 1.0,
        "market_cap_usd": 85_000_000,
        "price_at_signal": 7.50,
        "sector": "Technology",
        "market_data_status": "current",
        "halt_status": "clear",
        "corporate_action_filter_passed": True,
        "operating_universe_inclusion": True,
    }


# -----------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------

class TestM7Metadata:
    def test_pattern_id(self):
        assert M7Detector().pattern_id == PatternId.M7

    def test_track(self):
        assert M7Detector().track == PatternTrack.MULTI_DAY

    def test_thesis_category(self):
        assert M7Detector().thesis_category == ThesisCategory.CONTINUATION

    def test_route_class(self):
        assert M7Detector().route_class == RouteClass.A

    def test_vault_constants(self):
        assert LAMBDA_M7_MONTHLY == 0.0146
        assert MICROCAP_AMPLIFICATION == 1.75
        assert MIN_RANK_PCT == 0.90
        assert abs(LAMBDA_M7_10TD - 0.0146 * 1.75 * 10 / 21) < 1e-10


# -----------------------------------------------------------------------
# Decay helper
# -----------------------------------------------------------------------

class TestDecayHaircut:
    def test_day_0(self):
        assert compute_decay_haircut(0) == DECAY_HAIRCUT_MAX

    def test_day_63(self):
        assert compute_decay_haircut(63) == DECAY_HAIRCUT_MIN

    def test_day_30_midpoint(self):
        d = compute_decay_haircut(30)
        assert DECAY_HAIRCUT_MIN < d < DECAY_HAIRCUT_MAX

    def test_beyond_window_clipped(self):
        assert compute_decay_haircut(100) == DECAY_HAIRCUT_MIN

    def test_negative_returns_none(self):
        assert compute_decay_haircut(-1) is None

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), "N/A", None, 5.5])
    def test_invalid_inputs_return_none(self, value):
        assert compute_decay_haircut(value) is None


# -----------------------------------------------------------------------
# Firing cases
# -----------------------------------------------------------------------

class TestM7Firing:
    def test_fires_with_complete_data(self):
        det = M7Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.has_signal
        sig = result.signals[0]
        assert sig.direction == SignalDirection.LONG
        assert sig.signal_horizon == "10d"
        assert sig.route_class == RouteClass.A
        assert sig.raw_expected_edge > 0
        f = result.features.features
        assert f["signal_generated"] is True
        assert f["exposure_x_m7"] > 0

    def test_x_m7_equals_rank_times_decay(self):
        det = M7Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        f = result.features.features
        assert f["exposure_x_m7"] == round(0.95 * 0.63, 6)

    def test_edge_equals_x_times_lambda(self):
        det = M7Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        x = result.features.features["exposure_x_m7"]
        expected = round(x * LAMBDA_M7_10TD, 6)
        assert result.signals[0].raw_expected_edge == expected

    def test_signal_strength_is_rank_not_x(self):
        det = M7Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.signals[0].raw_signal_strength == round(0.95, 6)

    def test_exact_boundary_090_fires(self):
        det = M7Detector()
        data = _firing_data()
        data["predicted_return_rank_pct"] = 0.90
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal

    def test_below_boundary_rejected(self):
        det = M7Detector()
        data = _firing_data()
        data["predicted_return_rank_pct"] = 0.899999
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "prediction_below_threshold"

    def test_rc_pending_fires(self):
        det = M7Detector()
        data = _firing_data()
        data["rc_confidence_status"] = "pending"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["rc_confidence_status"] == "pending"

    def test_rc_failed_fires_with_review_flag(self):
        det = M7Detector()
        data = _firing_data()
        data["rc_confidence_status"] = "failed"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["operator_review_flag"] == "M7_rc_failed"

    def test_rc_missing_defaults_pending(self):
        det = M7Detector()
        data = _firing_data()
        del data["rc_confidence_status"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["rc_confidence_status"] == "pending"
        assert result.quality_flags["missing_rc_status"] is True

    def test_invalid_rc_status_defaults_pending_with_flag(self):
        det = M7Detector()
        data = _firing_data()
        data["rc_confidence_status"] = "garbage"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["rc_confidence_status"] == "pending"
        assert result.quality_flags["invalid_rc_status"] is True

    def test_priors_logged(self):
        det = M7Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        f = result.features.features
        assert f["lambda_M7_monthly"] == LAMBDA_M7_MONTHLY
        assert f["validated_or_shadow_lambda_M7_10td"] == LAMBDA_M7_10TD
        assert f["lambda_M7_source"] == "shadow_prior"

    def test_custom_lambda(self):
        det = M7Detector(lambda_m7_10td=0.02)
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.features.features["lambda_M7_source"] == "validated_or_injected"
        default = M7Detector().detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.signals[0].raw_expected_edge > default.signals[0].raw_expected_edge

    def test_invalid_lambda_rejected(self):
        with pytest.raises(ValueError, match="lambda_m7_10td"):
            M7Detector(lambda_m7_10td=float("nan"))

    def test_zero_lambda_rejected(self):
        with pytest.raises(ValueError, match="lambda_m7_10td"):
            M7Detector(lambda_m7_10td=0.0)

    def test_decay_from_days_since_retrain(self):
        det = M7Detector()
        data = _firing_data()
        del data["decay_haircut"]
        data["days_since_retrain"] = 0
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["decay_haircut"] == DECAY_HAIRCUT_MAX

    def test_data_confidence_default(self):
        det = M7Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.signals[0].data_confidence == 1.0


# -----------------------------------------------------------------------
# No-signal / rejection cases
# -----------------------------------------------------------------------

class TestM7NoSignal:
    def test_pit_false_rejected(self):
        det = M7Detector()
        data = _firing_data()
        data["point_in_time_passed"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "model_output_not_point_in_time"

    def test_missing_lineage_rejected(self):
        det = M7Detector()
        data = _firing_data()
        del data["model_artifact_hash"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features is None

    def test_blank_lineage_rejected(self):
        det = M7Detector()
        data = _firing_data()
        data["model_artifact_hash"] = "   "
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features is None

    def test_placeholder_lineage_rejected(self):
        det = M7Detector()
        data = _firing_data()
        data["feature_snapshot_id"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features is None

    @pytest.mark.parametrize("bad_value", [False, True, 123, 4.5, {"id": "model"}, ["model"]])
    def test_non_string_lineage_ids_rejected(self, bad_value):
        det = M7Detector()
        data = _firing_data()
        data["model_version"] = bad_value
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features is None

    def test_datetime_data_cutoff_lineage_is_accepted(self):
        det = M7Detector()
        data = _firing_data()
        data["data_cutoff_timestamp"] = datetime(2026, 5, 20, 20, 0, 0, tzinfo=timezone.utc)
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.has_signal
        assert result.features.features["data_cutoff_timestamp"] == data["data_cutoff_timestamp"]

    def test_run_not_completed_rejected(self):
        det = M7Detector()
        data = _firing_data()
        data["prediction_run_status"] = "failed"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "prediction_run_not_completed"

    def test_not_canonical_rejected(self):
        det = M7Detector()
        data = _firing_data()
        data["is_canonical"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "prediction_run_not_canonical"

    def test_too_many_missing_features_rejected(self):
        det = M7Detector()
        data = _firing_data()
        data["feature_count"] = 31
        data["missing_feature_count"] = 10  # 10/31 = 0.32 > 0.30
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "too_many_missing_features"

    def test_invalid_feature_count_rejected(self):
        det = M7Detector()
        data = _firing_data()
        data["feature_count"] = "N/A"
        data["missing_feature_count"] = 99
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_feature_completeness"

    def test_negative_missing_feature_count_rejected(self):
        det = M7Detector()
        data = _firing_data()
        data["feature_count"] = 31
        data["missing_feature_count"] = -5
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_feature_completeness"

    def test_missing_count_above_feature_count_rejected(self):
        det = M7Detector()
        data = _firing_data()
        data["feature_count"] = 31
        data["missing_feature_count"] = 32
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_feature_completeness"

    def test_invalid_decay_rejected(self):
        det = M7Detector()
        data = _firing_data()
        data["decay_haircut"] = 0.80  # above max
        del data["days_since_retrain"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_decay_haircut"

    def test_nan_decay_rejected(self):
        det = M7Detector()
        data = _firing_data()
        data["decay_haircut"] = float("nan")
        del data["days_since_retrain"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal

    def test_inconsistent_decay_rejected(self):
        det = M7Detector()
        data = _firing_data()
        data["decay_haircut"] = 0.65
        data["days_since_retrain"] = 60  # computed ~0.507, explicit 0.65 → delta > 0.02
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "inconsistent_decay_haircut"

    def test_invalid_days_since_retrain_rejected_when_present(self):
        det = M7Detector()
        data = _firing_data()
        data["days_since_retrain"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_days_since_retrain"

    def test_invalid_predicted_return_rejected(self):
        det = M7Detector()
        data = _firing_data()
        data["predicted_return"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "invalid_predicted_return"

    def test_string_rank_no_crash(self):
        det = M7Detector()
        data = _firing_data()
        data["predicted_return_rank_pct"] = "N/A"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features is None

    def test_not_operating_universe(self):
        det = M7Detector()
        data = _firing_data()
        data["operating_universe_inclusion"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "not_operating_universe"

    def test_missing_operating_universe_fails_closed(self):
        det = M7Detector()
        data = _firing_data()
        del data["operating_universe_inclusion"]
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "missing_operating_universe"

    def test_delayed_market_data_rejected(self):
        det = M7Detector()
        data = _firing_data()
        data["market_data_status"] = "delayed"
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert not result.has_signal
        assert result.features.features["rejection_reason"] == "data_delay"

    def test_event_data_cannot_override_rank(self):
        det = M7Detector()
        data = _firing_data()
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=data,
            event_data={"predicted_return_rank_pct": 0.50, "point_in_time_passed": False},
            lineage_hashes=["h"],
        ))
        assert result.has_signal  # market_data rank 0.95 wins
        assert result.features.features["predicted_return_rank_pct"] == 0.95

    def test_event_data_cannot_override_lineage_evidence(self):
        det = M7Detector()
        data = _firing_data()
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=data,
            event_data={
                "feature_snapshot_id": "EVIL-FEAT",
                "model_version": "EVIL-MODEL",
                "prediction_run_id": "EVIL-PRED",
            },
            lineage_hashes=["h"],
        ))
        assert result.has_signal
        f = result.features.features
        assert f["feature_snapshot_id"] == data["feature_snapshot_id"]
        assert f["model_version"] == data["model_version"]
        assert f["prediction_run_id"] == data["prediction_run_id"]
        assert f["signal_identity_components"]["feature_snapshot_id"] == data["feature_snapshot_id"]


# -----------------------------------------------------------------------
# Quality and diagnostics
# -----------------------------------------------------------------------

class TestM7Quality:
    def test_always_full_fidelity_when_pit_passes(self):
        det = M7Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.features.fidelity_tier == FidelityTier.FULL

    def test_lite_fidelity_when_pit_fails(self):
        det = M7Detector()
        data = _firing_data()
        data["point_in_time_passed"] = False
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=data, lineage_hashes=["h"]))
        assert result.features.fidelity_tier == FidelityTier.LITE

    def test_field_confidence_affects_data_confidence(self):
        det = M7Detector()
        data = _firing_data()
        data["field_confidence"] = {"prediction": 0.5}
        result = det.detect(PatternInput(
            ticker="ACME", asof_timestamp=_ts(), market_data=data,
            fundamental_data={"field_confidence": {"fun": 0.9}},
            event_data={"field_confidence": {"ev": 0.8}},
            lineage_hashes=["h"],
        ))
        assert result.has_signal
        assert result.signals[0].data_confidence < 1.0


# -----------------------------------------------------------------------
# Signal identity
# -----------------------------------------------------------------------

class TestM7Identity:
    def test_stable_across_asof_refresh(self):
        det = M7Detector()
        r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts() + timedelta(hours=1), market_data=_firing_data(), lineage_hashes=["h"]))
        assert r1.features.features["signal_identity_hash"] == r2.features.features["signal_identity_hash"]

    def test_changes_with_feature_snapshot(self):
        det = M7Detector()
        d1 = _firing_data()
        d2 = _firing_data()
        d2["feature_snapshot_id"] = "feat-ACME-20260521"
        r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d1, lineage_hashes=["h"]))
        r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d2, lineage_hashes=["h"]))
        assert r1.features.features["signal_identity_hash"] != r2.features.features["signal_identity_hash"]

    def test_source_is_prediction_row(self):
        det = M7Detector()
        result = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert result.features.features["signal_identity_source"] == "m7_prediction_row"


# -----------------------------------------------------------------------
# Hashes
# -----------------------------------------------------------------------

class TestM7Hashes:
    def test_stable(self):
        det = M7Detector()
        inp = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"])
        r1, r2 = det.detect(inp), det.detect(inp)
        assert r1.input_hashes == r2.input_hashes
        assert r1.output_hashes == r2.output_hashes

    def test_output_changes_with_lambda(self):
        r1 = M7Detector().detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        r2 = M7Detector(lambda_m7_10td=0.02).detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=_firing_data(), lineage_hashes=["h"]))
        assert r1.output_hashes != r2.output_hashes

    def test_output_hash_matches(self):
        det = M7Detector()
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

class TestM7EvidenceBridge:
    def _run_detection(self, db_session, *, market_data=None):
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session, provider="ML", endpoint="/m7/predict",
            asof_timestamp=_ts(), raw_payload={"prediction": "fixture"}, job_run_id=run.job_run_id,
        )
        det = M7Detector()
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
        assert sig.pattern_id == "M7"
        assert sig.route_class == "A"
        assert sig.thesis_category == "continuation"
        assert sig.signal_horizon == "10d"
        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        assert feat.feature_manifest_version == "m7-v1"

    def test_job_run_id(self, db_session):
        run, _, persisted = self._run_detection(db_session)
        assert db_session.get(FeatureSnapshot, persisted.feature_snapshot_id).job_run_id == run.job_run_id

    def test_no_signal_feature_only(self, db_session):
        data = _firing_data()
        data["predicted_return_rank_pct"] = 0.50
        _, _, persisted = self._run_detection(db_session, market_data=data)
        assert persisted.feature_snapshot_id is not None
        assert len(persisted.signal_ids) == 0

    def test_feature_hash_deterministic(self, db_session):
        _, _, p1 = self._run_detection(db_session)
        _, _, p2 = self._run_detection(db_session)
        assert db_session.get(FeatureSnapshot, p1.feature_snapshot_id).feature_hash == db_session.get(FeatureSnapshot, p2.feature_snapshot_id).feature_hash
