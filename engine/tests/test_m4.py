"""
M4 52-Week High Breakout detector tests.

Vault contract verification:
  - Detector metadata matches SPEC.md
  - Exposure formula matches EXPOSURE.md
  - Signal fires on breakout with top-3-decile extension
  - No signal below 52-week high
  - Feature snapshot written even without signal
  - raw_expected_edge = X_M4 * lambda_M4_15td (deterministic)
  - Fidelity degrades on short history
  - Evidence bridge writes with correct FK chain
  - Point-in-time and lineage guards produce warnings
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

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
    X_M4_CAP,
    compute_m4_features,
    apply_cohort_gate,
)


def _ts():
    return datetime(2026, 5, 20, 21, 0, 0, tzinfo=timezone.utc)


def _setup_run(db_session):
    job = create_job(db_session, name="m4_detector", job_type="detector", owner="pattern_engine")
    run = start_run(db_session, job_id=job.job_id)
    db_session.flush()
    return run


def _cohort_extensions(n=30, top_val=0.12):
    """Generate a realistic breakout cohort with extensions [0..top_val]."""
    return [round(i * top_val / (n - 1), 4) for i in range(n)]


# -----------------------------------------------------------------------
# Detector metadata
# -----------------------------------------------------------------------

class TestM4Metadata:
    def test_pattern_id(self):
        det = M4Detector()
        assert det.pattern_id == PatternId.M4

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
# Exposure formula (EXPOSURE.md)
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
        feat = compute_m4_features(price=16.00, high_52w=10.00)
        assert feat["X_M4"] == X_M4_CAP

    def test_numerical_example_from_spec(self):
        """EXPOSURE.md table: 20% above -> X_M4 = 1.20."""
        feat = compute_m4_features(price=12.00, high_52w=10.00)
        assert feat["breakout_extension"] == 0.2
        assert feat["X_M4"] == 1.2


# -----------------------------------------------------------------------
# Cohort gate
# -----------------------------------------------------------------------

class TestCohortGate:
    def test_top_decile_passes(self):
        exts = _cohort_extensions(30, 0.12)
        result = apply_cohort_gate(0.12, exts)
        assert result["cohort_gate_passed"] is True
        assert result["breakout_cohort_size"] == 30

    def test_bottom_decile_fails(self):
        exts = _cohort_extensions(30, 0.12)
        result = apply_cohort_gate(0.01, exts)
        assert result["cohort_gate_passed"] is False

    def test_zero_extension_excluded(self):
        exts = _cohort_extensions(30, 0.12)
        result = apply_cohort_gate(0.0, exts)
        assert result["cohort_gate_passed"] is False

    def test_small_cohort_all_pass(self):
        result = apply_cohort_gate(0.05, [0.02, 0.05, 0.08])
        assert result["cohort_gate_passed"] is True
        assert result["small_cohort_warning"] is True


# -----------------------------------------------------------------------
# Detector: firing case
# -----------------------------------------------------------------------

class TestM4Firing:
    def test_fires_on_breakout(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "price": 11.50,
                "high_52w": 10.00,
                "cohort_extensions": _cohort_extensions(30, 0.15),
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)

        assert result.has_signal
        assert len(result.signals) == 1
        sig = result.signals[0]
        assert sig.direction == SignalDirection.LONG
        assert sig.signal_horizon == "15d"

    def test_raw_expected_edge_is_deterministic(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "price": 11.50,
                "high_52w": 10.00,
                "cohort_extensions": _cohort_extensions(30, 0.15),
            },
            lineage_hashes=["hash1"],
        )
        r1 = det.detect(inp)
        r2 = det.detect(inp)

        # X_M4 = 1.15, raw_expected_edge = 1.15 * lambda_15td
        expected_x = 1.15
        expected_edge = round(expected_x * LAMBDA_M4_15TD, 6)

        assert r1.signals[0].raw_expected_edge == expected_edge
        assert r1.signals[0].raw_expected_edge == r2.signals[0].raw_expected_edge

    def test_signal_strength_is_x_over_cap(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "price": 11.50,
                "high_52w": 10.00,
                "cohort_extensions": _cohort_extensions(30, 0.15),
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        # X_M4 = 1.15, signal_strength = 1.15 / 1.5
        assert result.signals[0].raw_signal_strength == round(1.15 / 1.5, 6)

    def test_data_confidence_default_1_0(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "price": 11.50,
                "high_52w": 10.00,
                "cohort_extensions": _cohort_extensions(30, 0.15),
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.signals[0].data_confidence == 1.0

    def test_expected_return_priors_logged_in_features(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "price": 11.50,
                "high_52w": 10.00,
                "cohort_extensions": _cohort_extensions(30, 0.15),
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        priors = result.features.features["expected_return_priors"]
        assert priors["tier"] in {"default", "high_conviction"}
        assert priors["gross_bps"] == round(result.signals[0].raw_expected_edge * 10_000, 2)


# -----------------------------------------------------------------------
# Detector: no-signal cases
# -----------------------------------------------------------------------

class TestM4NoSignal:
    def test_below_high_no_signal(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"price": 9.50, "high_52w": 10.00},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        assert result.features is not None
        assert result.features.features["X_M4"] == 0.95

    def test_exact_high_no_extension_no_signal(self):
        """Exact-high crossings excluded per EXPOSURE.md: extension must be > 0."""
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "price": 10.00,
                "high_52w": 10.00,
                "cohort_extensions": _cohort_extensions(30, 0.10),
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal

    def test_missing_price_no_features(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"high_52w": 10.00},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        assert result.features is None
        assert any("missing" in w for w in result.warnings)

    def test_cohort_gate_reject(self):
        """Bottom-decile extension fails the top-3-decile gate."""
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "price": 10.01,
                "high_52w": 10.00,
                "cohort_extensions": _cohort_extensions(30, 0.15),
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        assert result.features is not None

    def test_missing_cohort_data_no_signal(self):
        """M4 cohort gate is signal-generation logic, not a downstream gate."""
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"price": 11.00, "high_52w": 10.00},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        assert result.features is not None
        assert result.features.features["cohort_gate_passed"] is False
        assert result.features.features["cohort_missing"] is True
        assert any("cohort" in w for w in result.warnings)


# -----------------------------------------------------------------------
# Fidelity degradation
# -----------------------------------------------------------------------

class TestM4Fidelity:
    def test_short_history_degrades_to_lite(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "price": 11.00,
                "high_52w": 10.00,
                "n_sessions_in_window": 100,
                "cohort_extensions": [0.10],
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.features.fidelity_tier == FidelityTier.LITE
        assert result.features.features["short_history_flag"] is True

    def test_full_history_is_full(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "price": 11.00,
                "high_52w": 10.00,
                "n_sessions_in_window": 252,
                "cohort_extensions": [0.10],
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.features.fidelity_tier == FidelityTier.FULL

    def test_short_history_reduces_data_confidence(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "price": 11.00,
                "high_52w": 10.00,
                "n_sessions_in_window": 100,
                "cohort_extensions": [0.10],
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.signals[0].data_confidence < 1.0


# -----------------------------------------------------------------------
# Quality guards
# -----------------------------------------------------------------------

class TestM4Guards:
    def test_missing_lineage_warning(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"price": 11.00, "high_52w": 10.00, "cohort_extensions": [0.10]},
            lineage_hashes=[],
        )
        result = det.detect(inp)
        assert result.quality_flags.get("missing_lineage") is True
        # Still fires — guards don't block admission
        assert result.has_signal

    def test_future_timestamp_warning(self):
        det = M4Detector()
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=future,
            market_data={"price": 11.00, "high_52w": 10.00, "cohort_extensions": [0.10]},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.quality_flags.get("future_timestamp") is True


# -----------------------------------------------------------------------
# Hash determinism
# -----------------------------------------------------------------------

class TestM4Hashes:
    def test_input_output_hashes_stable(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"price": 11.00, "high_52w": 10.00, "cohort_extensions": [0.10]},
            lineage_hashes=["hash1"],
        )
        r1 = det.detect(inp)
        r2 = det.detect(inp)
        assert r1.input_hashes == r2.input_hashes
        assert r1.output_hashes == r2.output_hashes

    def test_hashes_change_when_cohort_changes(self):
        det = M4Detector()
        base = {
            "price": 10.20,
            "high_52w": 10.00,
        }
        pass_inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={**base, "cohort_extensions": [0.01, 0.02]},
            lineage_hashes=["hash1"],
        )
        fail_inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={**base, "cohort_extensions": _cohort_extensions(30, 0.50)},
            lineage_hashes=["hash1"],
        )
        pass_result = det.detect(pass_inp)
        fail_result = det.detect(fail_inp)

        assert pass_result.has_signal
        assert not fail_result.has_signal
        assert pass_result.input_hashes != fail_result.input_hashes
        assert pass_result.output_hashes != fail_result.output_hashes

    def test_output_hash_matches_final_features_and_signals(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"price": 11.00, "high_52w": 10.00, "cohort_extensions": [0.10]},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        expected = stable_hash({
            "features": result.features.features,
            "signals": [
                {
                    "direction": sig.direction,
                    "raw_signal_strength": sig.raw_signal_strength,
                    "raw_expected_edge": sig.raw_expected_edge,
                    "signal_horizon": sig.signal_horizon,
                    "signal_status": sig.signal_status,
                    "data_confidence": sig.data_confidence,
                }
                for sig in result.signals
            ],
            "warnings": result.warnings,
            "quality_flags": result.quality_flags,
        })
        assert result.output_hashes["features"] == expected


# -----------------------------------------------------------------------
# Evidence bridge integration
# -----------------------------------------------------------------------

class TestM4EvidenceBridge:
    def _run_detection(self, db_session, *, price=11.50, high_52w=10.00):
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session,
            provider="FMP",
            endpoint="/stable/historical-price-eod/full",
            asof_timestamp=_ts(),
            raw_payload={"close": price, "high_52w": high_52w},
            job_run_id=run.job_run_id,
        )
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "price": price,
                "high_52w": high_52w,
                "cohort_extensions": _cohort_extensions(30, 0.15),
            },
            fundamental_data={"market_cap": 75_000_000},
            lineage_hashes=[lineage.raw_payload_hash],
            job_run_id=run.job_run_id,
        )
        result = det.detect(inp)
        persisted = persist_detection_result(
            db_session,
            result,
            det,
            job_run_id=run.job_run_id,
            data_lineage_ids=[lineage.data_lineage_id],
        )
        db_session.flush()
        return run, result, persisted

    def test_signal_persists_with_feature_fk(self, db_session):
        run, result, persisted = self._run_detection(db_session)
        assert persisted.feature_snapshot_id is not None
        assert len(persisted.signal_ids) == 1

        sig = db_session.get(SignalRegistry, persisted.signal_ids[0])
        assert sig.feature_snapshot_id == persisted.feature_snapshot_id
        assert sig.pattern_id == "M4"
        assert sig.route_class == "A"
        assert sig.thesis_category == "right_tail_convex"
        assert sig.signal_horizon == "15d"

        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        assert feat.feature_manifest_version == "m4-v1"

    def test_job_run_id_preserved(self, db_session):
        run, _, persisted = self._run_detection(db_session)
        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        sig = db_session.get(SignalRegistry, persisted.signal_ids[0])
        assert feat.job_run_id == run.job_run_id
        assert sig.job_run_id == run.job_run_id

    def test_universe_snapshot_id_preserved(self, db_session):
        run = _setup_run(db_session)
        usn = record_universe_snapshot(
            db_session,
            ticker="ACME",
            asof_timestamp=_ts(),
            operating_universe_inclusion=True,
            job_run_id=run.job_run_id,
        )
        lineage = record_data_lineage(
            db_session,
            provider="FMP",
            endpoint="/stable/historical-price-eod/full",
            asof_timestamp=_ts(),
            raw_payload={"close": 11.0},
            job_run_id=run.job_run_id,
        )
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"price": 11.00, "high_52w": 10.00, "cohort_extensions": [0.10]},
            lineage_hashes=[lineage.raw_payload_hash],
        )
        result = det.detect(inp)
        persisted = persist_detection_result(
            db_session,
            result,
            det,
            job_run_id=run.job_run_id,
            universe_snapshot_id=usn.universe_snapshot_id,
            data_lineage_ids=[lineage.data_lineage_id],
        )
        db_session.flush()

        sig = db_session.get(SignalRegistry, persisted.signal_ids[0])
        assert sig.universe_snapshot_id == usn.universe_snapshot_id

    def test_no_signal_writes_feature_only(self, db_session):
        run, result, persisted = self._run_detection(db_session, price=9.50, high_52w=10.00)
        assert persisted.feature_snapshot_id is not None
        assert len(persisted.signal_ids) == 0
        assert db_session.query(SignalRegistry).count() == 0
        assert db_session.query(FeatureSnapshot).count() == 1

    def test_feature_hash_deterministic(self, db_session):
        _, _, p1 = self._run_detection(db_session)
        _, _, p2 = self._run_detection(db_session)
        f1 = db_session.get(FeatureSnapshot, p1.feature_snapshot_id)
        f2 = db_session.get(FeatureSnapshot, p2.feature_snapshot_id)
        assert f1.feature_hash == f2.feature_hash
