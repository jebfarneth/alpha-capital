"""
M4 52-Week High Breakout detector tests.

Vault contract verification (amended):
  - All breakouts P >= H52w emit signals (including exact-high closes)
  - Cohort rank / top3_decile_flag are metadata, not admission gates
  - Missing cohort data does NOT block signal; adds quality flag
  - below-high and non-operating-universe are true no-signal cases
  - Fresh-breakout activation lane intact
  - Evidence bridge writes with correct FK chain
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
    compute_cohort_metadata,
)


def _ts():
    return datetime(2026, 5, 20, 21, 0, 0, tzinfo=timezone.utc)


def _setup_run(db_session):
    job = create_job(db_session, name="m4_detector", job_type="detector", owner="pattern_engine")
    run = start_run(db_session, job_id=job.job_id)
    db_session.flush()
    return run


def _cohort_extensions(n=30, top_val=0.12):
    return [round(i * top_val / (n - 1), 4) for i in range(n)]


# -----------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------

class TestM4Metadata:
    def test_pattern_id(self):
        assert M4Detector().pattern_id == PatternId.M4

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
# Exposure formula
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
        feat = compute_m4_features(price=12.00, high_52w=10.00)
        assert feat["breakout_extension"] == 0.2
        assert feat["X_M4"] == 1.2


# -----------------------------------------------------------------------
# Cohort metadata (no longer a gate)
# -----------------------------------------------------------------------

class TestCohortMetadata:
    def test_top_decile_flagged(self):
        exts = _cohort_extensions(30, 0.12)
        meta = compute_cohort_metadata(0.12, exts)
        assert meta["top3_decile_flag"] is True
        assert meta["breakout_cohort_size"] == 30
        assert meta["breakout_cohort_rank"] is not None
        assert meta["breakout_cohort_percentile"] is not None

    def test_bottom_decile_not_flagged(self):
        exts = _cohort_extensions(30, 0.12)
        meta = compute_cohort_metadata(0.01, exts)
        assert meta["top3_decile_flag"] is False

    def test_zero_extension_not_flagged(self):
        exts = _cohort_extensions(30, 0.12)
        meta = compute_cohort_metadata(0.0, exts)
        assert meta["top3_decile_flag"] is False

    def test_small_cohort_all_flagged(self):
        meta = compute_cohort_metadata(0.05, [0.02, 0.05, 0.08])
        assert meta["top3_decile_flag"] is True
        assert meta["small_cohort_warning"] is True


# -----------------------------------------------------------------------
# Firing cases: all breakouts emit signals
# -----------------------------------------------------------------------

class TestM4Firing:
    def test_extended_breakout_fires(self):
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
        assert result.signals[0].direction == SignalDirection.LONG
        assert result.signals[0].signal_horizon == "15d"
        assert result.features.features["extension_tier"] in {"default", "high_conviction"}

    def test_exact_high_fires(self):
        """Exact-high close P == H52w now emits a signal per amended vault."""
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
        assert result.has_signal
        sig = result.signals[0]
        assert sig.raw_expected_edge == round(1.0 * LAMBDA_M4_15TD, 6)
        assert sig.raw_signal_strength == round(1.0 / X_M4_CAP, 6)
        assert result.features.features["extension_tier"] == "exact_high"
        assert result.features.features["tier_classification"] == "exact_high"
        assert result.features.features["breakout_extension"] == 0.0
        assert result.features.features["X_M4"] == 1.0

    def test_bottom_decile_breakout_still_fires(self):
        """Non-top-3-decile breakout still emits signal; top3_decile_flag is metadata."""
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
        assert result.has_signal
        assert result.features.features.get("top3_decile_flag") is False

    def test_missing_cohort_still_fires(self):
        """Missing cohort data does NOT block signal; adds quality flag."""
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"price": 11.00, "high_52w": 10.00},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.has_signal
        assert result.quality_flags.get("cohort_metadata_unavailable") is True
        assert any("cohort" in w for w in result.warnings)

    def test_raw_expected_edge_deterministic(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"price": 11.50, "high_52w": 10.00, "cohort_extensions": _cohort_extensions(30, 0.15)},
            lineage_hashes=["hash1"],
        )
        r1 = det.detect(inp)
        r2 = det.detect(inp)
        expected = round(1.15 * LAMBDA_M4_15TD, 6)
        assert r1.signals[0].raw_expected_edge == expected
        assert r1.signals[0].raw_expected_edge == r2.signals[0].raw_expected_edge

    def test_signal_strength_is_x_over_cap(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"price": 11.50, "high_52w": 10.00, "cohort_extensions": _cohort_extensions(30, 0.15)},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.signals[0].raw_signal_strength == round(1.15 / 1.5, 6)

    def test_data_confidence_default_1_0(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"price": 11.50, "high_52w": 10.00, "cohort_extensions": _cohort_extensions(30, 0.15)},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.signals[0].data_confidence == 1.0

    def test_expected_return_priors_logged(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"price": 11.50, "high_52w": 10.00, "cohort_extensions": _cohort_extensions(30, 0.15)},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        priors = result.features.features["expected_return_priors"]
        assert priors["gross_bps"] == round(result.signals[0].raw_expected_edge * 10_000, 2)

    def test_diagnostic_source_features_logged(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "price": 11.50, "high_52w": 10.00,
                "cohort_extensions": _cohort_extensions(30, 0.15),
                "D1_decile": 8, "R_6_12m_skip": 0.234,
                "hamilton_regime_prob": 0.72, "hazard_score_at_signal": 22,
            },
            fundamental_data={"market_cap": 95_400_000, "sector": "Technology", "industry": "Software - Application", "analyst_count": 2},
            event_data={"filing_veto_status": "clear"},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        f = result.features.features
        assert f["D1_decile"] == 8
        assert f["R_6_12m_skip"] == 0.234
        assert f["analyst_count"] == 2
        assert f["sector"] == "Technology"

    def test_field_confidence_product(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "price": 11.00, "high_52w": 10.00, "cohort_extensions": [0.10],
                "field_confidence": {"adj_close": 0.95, "high_52w": 0.90},
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.signals[0].data_confidence == 0.855


# -----------------------------------------------------------------------
# Fresh-breakout lane (unchanged)
# -----------------------------------------------------------------------

class TestM4FreshBreakout:
    def test_fresh_watchlist_signal_near_high(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "entry_lane": "fresh_breakout_activation",
                "activation_state": "watchlist",
                "price": 9.80, "high_52w": 10.00,
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.has_signal
        assert result.signals[0].signal_status == "watchlist"
        assert result.signals[0].raw_expected_edge == 0.0

    def test_fresh_activation_signal(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "entry_lane": "fresh_breakout_activation",
                "activation_state": "activated",
                "price": 9.90, "last_price": 10.50, "high_52w": 10.00,
                "intraday_range_confirmation": 1.25,
                "intraday_volume_confirmation": 1.50,
                "spread_pct_vs_eval_quote": 0.005,
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.has_signal
        assert result.signals[0].signal_status == "active"
        assert result.signals[0].raw_expected_edge > 0

    def test_fresh_activation_spread_failure(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "entry_lane": "fresh_breakout_activation",
                "activation_state": "activated",
                "price": 9.90, "last_price": 10.50, "high_52w": 10.00,
                "intraday_range_confirmation": 1.25,
                "intraday_volume_confirmation": 1.50,
                "spread_pct_vs_eval_quote": 0.02,
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        assert result.features.features["activation_failure_reason"] == "spread_too_wide"


# -----------------------------------------------------------------------
# True no-signal cases
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

    def test_not_operating_universe_no_signal(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "price": 11.00, "high_52w": 10.00,
                "cohort_extensions": [0.10],
                "operating_universe_inclusion": False,
            },
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        assert result.quality_flags["not_operating_universe_member"] is True


# -----------------------------------------------------------------------
# Fidelity
# -----------------------------------------------------------------------

class TestM4Fidelity:
    def test_always_full_fidelity(self):
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"price": 11.00, "high_52w": 10.00, "n_sessions_in_window": 100, "cohort_extensions": [0.10]},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.features.fidelity_tier == FidelityTier.FULL
        assert result.features.features["short_history_flag"] is True


# -----------------------------------------------------------------------
# Guards
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
# Hashes
# -----------------------------------------------------------------------

class TestM4Hashes:
    def test_stable(self):
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

    def test_change_with_price(self):
        det = M4Detector()
        r1 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data={"price": 11.00, "high_52w": 10.00}, lineage_hashes=["h"]))
        r2 = det.detect(PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data={"price": 12.00, "high_52w": 10.00}, lineage_hashes=["h"]))
        assert r1.output_hashes != r2.output_hashes

    def test_output_hash_matches_final_state(self):
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
# Evidence bridge
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
                "price": price, "high_52w": high_52w,
                "cohort_extensions": _cohort_extensions(30, 0.15),
            },
            fundamental_data={"market_cap": 75_000_000},
            lineage_hashes=[lineage.raw_payload_hash],
            job_run_id=run.job_run_id,
        )
        result = det.detect(inp)
        persisted = persist_detection_result(
            db_session, result, det,
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
            db_session, ticker="ACME", asof_timestamp=_ts(),
            operating_universe_inclusion=True, job_run_id=run.job_run_id,
        )
        lineage = record_data_lineage(
            db_session, provider="FMP", endpoint="/stable/historical-price-eod/full",
            asof_timestamp=_ts(), raw_payload={"close": 11.0}, job_run_id=run.job_run_id,
        )
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={"price": 11.00, "high_52w": 10.00, "cohort_extensions": [0.10]},
            lineage_hashes=[lineage.raw_payload_hash],
        )
        result = det.detect(inp)
        persisted = persist_detection_result(
            db_session, result, det,
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

    def test_feature_hash_deterministic(self, db_session):
        _, _, p1 = self._run_detection(db_session)
        _, _, p2 = self._run_detection(db_session)
        f1 = db_session.get(FeatureSnapshot, p1.feature_snapshot_id)
        f2 = db_session.get(FeatureSnapshot, p2.feature_snapshot_id)
        assert f1.feature_hash == f2.feature_hash

    def test_exact_high_persists_through_bridge(self, db_session):
        """Exact-high signal writes feature_snapshot + signal_registry via bridge."""
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session, provider="FMP", endpoint="/stable/historical-price-eod/full",
            asof_timestamp=_ts(), raw_payload={"close": 10.0}, job_run_id=run.job_run_id,
        )
        det = M4Detector()
        inp = PatternInput(
            ticker="ACME", asof_timestamp=_ts(),
            market_data={"price": 10.00, "high_52w": 10.00},
            lineage_hashes=[lineage.raw_payload_hash],
        )
        result = det.detect(inp)
        assert result.has_signal
        persisted = persist_detection_result(
            db_session, result, det,
            job_run_id=run.job_run_id,
            data_lineage_ids=[lineage.data_lineage_id],
        )
        db_session.flush()
        assert persisted.feature_snapshot_id is not None
        assert len(persisted.signal_ids) == 1
        sig = db_session.get(SignalRegistry, persisted.signal_ids[0])
        assert sig.pattern_id == "M4"
