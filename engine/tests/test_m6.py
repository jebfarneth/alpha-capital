"""
M6 Volatility-Compression Breakout detector tests.

Vault contract verification:
  - Detector metadata matches SPEC.md
  - Compression formula matches EXPOSURE.md
  - Signal fires on breakout from compressed regime
  - No signal when not compressed or no breakout
  - Feature snapshot written even without signal
  - raw_expected_edge = X_M6_activation * lambda_M6_12td (deterministic)
  - Fidelity degrades on GK quality warnings
  - Evidence bridge writes with correct FK chain
  - Point-in-time and lineage guards produce warnings
  - Hashes are deterministic and change with inputs
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
    """Deep compression + confirmed breakout → should fire."""
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
    }


def _no_breakout_market_data():
    """Compressed but price below compression_high → no signal."""
    return {
        "compression_ratio": 0.60,
        "gk_vol_5d": 0.020,
        "gk_vol_60d": 0.033,
        "compression_high": 5.00,
        "sigma_20d": 0.030,
        "price": 4.90,
    }


def _not_compressed_market_data():
    """compression_ratio > 0.80 → no compression gate pass."""
    return {
        "compression_ratio": 0.90,
        "gk_vol_5d": 0.030,
        "gk_vol_60d": 0.033,
        "compression_high": 5.00,
        "sigma_20d": 0.030,
        "price": 5.20,
    }


# -----------------------------------------------------------------------
# Detector metadata
# -----------------------------------------------------------------------

class TestM6Metadata:
    def test_pattern_id(self):
        assert M6Detector().pattern_id == PatternId.M6

    def test_track(self):
        assert M6Detector().track == PatternTrack.MULTI_DAY

    def test_thesis_category(self):
        assert M6Detector().thesis_category == ThesisCategory.RIGHT_TAIL_CONVEX

    def test_route_class(self):
        assert M6Detector().route_class == RouteClass.A

    def test_vault_constants(self):
        assert LAMBDA_M6_MONTHLY == 0.011
        assert AMPLIFICATION == 1.45
        assert HOLD_DAYS == 12
        assert X_M6_CAP == 3.0
        assert MIN_COMPRESSION_DEPTH == 0.5
        assert abs(LAMBDA_M6_12TD - 0.011 * 1.45 * 12 / 21) < 1e-10


# -----------------------------------------------------------------------
# Compression formula (EXPOSURE.md)
# -----------------------------------------------------------------------

class TestCompressionFormula:
    def test_deep_compression(self):
        # ratio 0.50 → depth = (1.0 - 0.50) / 0.4 = 1.25
        assert round(compute_compression_depth(0.50), 4) == 1.25

    def test_mild_compression(self):
        # ratio 0.80 → depth = (1.0 - 0.80) / 0.4 = 0.5
        assert round(compute_compression_depth(0.80), 4) == 0.5

    def test_no_compression(self):
        # ratio 1.0 → depth = 0.0
        assert compute_compression_depth(1.0) == 0.0

    def test_expanding_vol(self):
        # ratio > 1.0 → clipped at 0.0
        assert compute_compression_depth(1.2) == 0.0

    def test_extreme_compression_capped(self):
        # ratio 0.0 → depth = 2.5 (capped)
        assert compute_compression_depth(0.0) == 2.5


class TestBreakoutExtension:
    def test_breakout_above_high(self):
        ext = compute_breakout_extension(price=5.25, compression_high=5.00, sigma_20d=0.03)
        # (5.25 - 5.00) / 5.00 / 0.03 = 1.667
        assert round(ext, 3) == 1.667

    def test_no_breakout(self):
        assert compute_breakout_extension(price=4.90, compression_high=5.00, sigma_20d=0.03) == 0.0

    def test_capped_at_3(self):
        ext = compute_breakout_extension(price=10.00, compression_high=5.00, sigma_20d=0.01)
        assert ext == 3.0


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
# Detector: firing case
# -----------------------------------------------------------------------

class TestM6Firing:
    def test_fires_on_breakout_from_compression(self):
        det = M6Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_firing_market_data(),
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)

        assert result.has_signal
        assert len(result.signals) == 1
        assert result.signals[0].direction == SignalDirection.LONG
        assert result.signals[0].signal_horizon == "12d"
        assert result.features.features["activation_state"] == "activated"

    def test_raw_expected_edge_is_deterministic(self):
        det = M6Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_firing_market_data(),
            lineage_hashes=["hash1"],
        )
        r1 = det.detect(inp)
        r2 = det.detect(inp)

        assert r1.signals[0].raw_expected_edge == r2.signals[0].raw_expected_edge
        # raw_expected_edge = X_M6_activation * LAMBDA_M6_12TD
        x_m6 = r1.features.features["X_M6_activation"]
        expected = round(x_m6 * LAMBDA_M6_12TD, 6)
        assert r1.signals[0].raw_expected_edge == expected

    def test_signal_strength_capped_at_1(self):
        det = M6Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_firing_market_data(),
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        x_m6 = result.features.features["X_M6_activation"]
        expected_strength = round(min(x_m6 / X_M6_CAP, 1.0), 6)
        assert result.signals[0].raw_signal_strength == expected_strength

    def test_data_confidence_default_1_0(self):
        det = M6Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_firing_market_data(),
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.signals[0].data_confidence == 1.0

    def test_expected_return_priors_in_features(self):
        det = M6Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_firing_market_data(),
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        priors = result.features.features["expected_return_priors"]
        assert priors["gross_bps"] == round(result.signals[0].raw_expected_edge * 10_000, 2)


# -----------------------------------------------------------------------
# Detector: no-signal cases
# -----------------------------------------------------------------------

class TestM6NoSignal:
    def test_no_breakout_no_signal(self):
        det = M6Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_no_breakout_market_data(),
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        assert result.features is not None
        assert result.features.features["activation_state"] == "no_breakout"

    def test_not_compressed_no_signal(self):
        det = M6Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_not_compressed_market_data(),
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        assert result.features is not None
        assert result.features.features["compression_gate_passed"] is False
        assert result.features.features["activation_state"] == "not_compressed"

    def test_compressed_but_no_price_watchlist(self):
        """Compressed without activation data → watchlist state, no signal."""
        det = M6Detector()
        data = {
            "compression_ratio": 0.55,
            "gk_vol_5d": 0.018,
            "gk_vol_60d": 0.033,
            "compression_high": 4.85,
            "sigma_20d": 0.028,
        }
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=data,
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        assert result.features is not None
        assert result.features.features["activation_state"] == "watchlist"
        assert result.features.features["compression_gate_passed"] is True

    def test_missing_compression_data_no_features(self):
        det = M6Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"price": 5.0},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.features is None
        assert any("missing" in w for w in result.warnings)


# -----------------------------------------------------------------------
# Fidelity degradation
# -----------------------------------------------------------------------

class TestM6Fidelity:
    def test_gk_warning_degrades_to_lite(self):
        det = M6Detector()
        data = _firing_market_data()
        data["gk_low_transaction_warning"] = True
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=data,
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.features.fidelity_tier == FidelityTier.LITE

    def test_gk_warning_reduces_data_confidence(self):
        det = M6Detector()
        data = _firing_market_data()
        data["gk_low_transaction_warning"] = True
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=data,
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.has_signal
        assert result.signals[0].data_confidence < 1.0

    def test_no_warning_is_full_fidelity(self):
        det = M6Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_firing_market_data(),
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.features.fidelity_tier == FidelityTier.FULL


# -----------------------------------------------------------------------
# Quality guards
# -----------------------------------------------------------------------

class TestM6Guards:
    def test_missing_lineage_warning(self):
        det = M6Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_firing_market_data(),
            lineage_hashes=[],
        )
        result = det.detect(inp)
        assert result.quality_flags.get("missing_lineage") is True
        # Still fires — guards don't block admission
        assert result.has_signal

    def test_future_timestamp_warning(self):
        det = M6Detector()
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=future,
            market_data=_firing_market_data(),
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.quality_flags.get("future_timestamp") is True

    def test_missing_expansion_data_warns(self):
        det = M6Detector()
        data = _firing_market_data()
        del data["session_high"]
        del data["session_low"]
        del data["gk_avg_5d"]
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=data,
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.quality_flags.get("missing_expansion_data") is True
        # Still fires — degraded but not blocked
        assert result.has_signal


# -----------------------------------------------------------------------
# Hash determinism
# -----------------------------------------------------------------------

class TestM6Hashes:
    def test_input_output_hashes_stable(self):
        det = M6Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_firing_market_data(),
            lineage_hashes=["hash1"],
        )
        r1 = det.detect(inp)
        r2 = det.detect(inp)
        assert r1.input_hashes == r2.input_hashes
        assert r1.output_hashes == r2.output_hashes

    def test_hashes_change_with_different_compression(self):
        det = M6Detector()
        d1 = _firing_market_data()
        d2 = _firing_market_data()
        d2["compression_ratio"] = 0.90  # not compressed
        inp1 = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d1, lineage_hashes=["h"])
        inp2 = PatternInput(ticker="ACME", asof_timestamp=_ts(), market_data=d2, lineage_hashes=["h"])
        r1 = det.detect(inp1)
        r2 = det.detect(inp2)

        assert r1.has_signal
        assert not r2.has_signal
        assert r1.output_hashes != r2.output_hashes

    def test_output_hash_matches_final_features_and_signals(self):
        det = M6Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_firing_market_data(),
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

class TestM6EvidenceBridge:
    def _run_detection(self, db_session, *, market_data=None):
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session,
            provider="FMP",
            endpoint="/stable/historical-price-eod/full",
            asof_timestamp=_ts(),
            raw_payload={"ohlcv": "fixture"},
            job_run_id=run.job_run_id,
        )
        det = M6Detector()
        data = market_data or _firing_market_data()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=data,
            fundamental_data={"market_cap": 60_000_000, "sector": "Technology"},
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
        assert sig.pattern_id == "M6"
        assert sig.route_class == "A"
        assert sig.thesis_category == "right_tail_convex"
        assert sig.signal_horizon == "12d"

        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        assert feat.feature_manifest_version == "m6-v1"

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
            raw_payload={"ohlcv": "fixture"},
            job_run_id=run.job_run_id,
        )
        det = M6Detector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data=_firing_market_data(),
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
        _, result, persisted = self._run_detection(
            db_session, market_data=_no_breakout_market_data()
        )
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
