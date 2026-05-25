"""
Pattern framework tests.

  - No-signal result writes feature snapshot but no signal row.
  - Signal result writes both feature_snapshot and signal_registry.
  - Signal links to feature_snapshot.
  - job_run_id is preserved.
  - universe_snapshot_id is preserved when provided.
  - Point-in-time guard catches future timestamps.
  - Missing lineage produces warning/quality flag.
  - Deterministic hashes are stable across repeated runs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from alpha.data.contracts import stable_hash
from alpha.db.models import FeatureSnapshot, SignalRegistry
from alpha.evidence.writer import create_job, record_data_lineage, start_run
from alpha.patterns.contracts import (
    FidelityTier,
    PatternDetectionResult,
    PatternFeatures,
    PatternId,
    PatternInput,
    PatternSignal,
    SignalDirection,
)
from alpha.patterns.evidence_bridge import persist_detection_result
from alpha.patterns.fixture_detector import FixtureDetector
from alpha.patterns.guards import (
    classify_fidelity,
    market_data_quality_rejection,
    operating_universe_rejection,
    quote_rejection,
    reject_future_timestamp,
    require_asof_timestamp,
    require_lineage_hash,
)


def _ts():
    return datetime(2026, 5, 20, 14, 30, 0, tzinfo=timezone.utc)


def _setup_run(db_session):
    job = create_job(db_session, name="test_detector", job_type="detector", owner="test")
    run = start_run(db_session, job_id=job.job_id)
    db_session.flush()
    return run


# -----------------------------------------------------------------------
# Guards
# -----------------------------------------------------------------------

class TestGuards:
    def test_require_asof_raises_on_none(self):
        with pytest.raises(ValueError, match="asof_timestamp"):
            require_asof_timestamp(None)

    def test_require_asof_returns_timestamp(self):
        ts = _ts()
        assert require_asof_timestamp(ts) is ts

    def test_reject_future_timestamp_flags(self):
        future = _ts() + timedelta(days=1)
        past = _ts()
        warnings = []
        flags = {}
        result = reject_future_timestamp(future, warnings, flags, reference=past)
        assert result is False
        assert len(warnings) == 1
        assert "future" in warnings[0]
        assert flags["future_timestamp"] is True
        assert flags["point_in_time_passed"] is False

    def test_reject_future_timestamp_passes_valid(self):
        past = _ts() - timedelta(hours=1)
        now = _ts()
        warnings = []
        flags = {}
        result = reject_future_timestamp(past, warnings, flags, reference=now)
        assert result is True
        assert len(warnings) == 0

    def test_require_lineage_hash_warns_on_empty(self):
        warnings = []
        flags = {}
        result = require_lineage_hash([], warnings, flags)
        assert result is False
        assert flags["missing_lineage"] is True
        assert len(warnings) == 1

    def test_require_lineage_hash_passes(self):
        warnings = []
        flags = {}
        result = require_lineage_hash(["abc123"], warnings, flags)
        assert result is True
        assert len(warnings) == 0

    def test_classify_fidelity_full(self):
        assert classify_fidelity(has_primary_data=True) == FidelityTier.FULL

    def test_classify_fidelity_lite_missing_secondary(self):
        assert classify_fidelity(has_primary_data=True, has_secondary_data=False) == FidelityTier.LITE

    def test_classify_fidelity_lite_pit_failed(self):
        assert classify_fidelity(has_primary_data=True, point_in_time_passed=False) == FidelityTier.LITE

    def test_classify_fidelity_unavailable(self):
        assert classify_fidelity(has_primary_data=False) == FidelityTier.UNAVAILABLE

    def test_operating_universe_missing_fails_closed(self):
        warnings = []
        flags = {}
        reason = operating_universe_rejection({}, warnings, flags, pattern_id="TEST")
        assert reason == "missing_operating_universe"
        assert flags["operating_universe_not_computed"] is True
        assert warnings

    def test_operating_universe_excluded(self):
        warnings = []
        flags = {}
        reason = operating_universe_rejection(
            {"operating_universe_inclusion": False}, warnings, flags, pattern_id="TEST",
        )
        assert reason == "not_operating_universe"
        assert flags["not_operating_universe_member"] is True

    def test_operating_universe_included(self):
        warnings = []
        flags = {}
        reason = operating_universe_rejection(
            {"operating_universe_inclusion": True}, warnings, flags, pattern_id="TEST",
        )
        assert reason is None
        assert flags == {}

    def test_market_data_quality_requires_fields_when_requested(self):
        feat = {}
        reason = market_data_quality_rejection(feat, {}, require_fields=True)
        assert reason == "missing_market_data_quality"
        assert feat == {}

    def test_market_data_quality_rejects_bad_values(self):
        base = {
            "market_data_status": "current",
            "halt_status": "clear",
            "corporate_action_filter_passed": True,
        }
        assert market_data_quality_rejection({}, base, require_fields=True) is None

        delayed = dict(base, market_data_status="delayed")
        assert market_data_quality_rejection({}, delayed, require_fields=True) == "data_delay"

        halted = dict(base, halt_status="halted")
        assert market_data_quality_rejection({}, halted, require_fields=True) == "halted"

        corp_action = dict(base, corporate_action_filter_passed=False)
        assert market_data_quality_rejection({}, corp_action, require_fields=True) == "spurious_corporate_action"

    def test_market_data_quality_copies_diagnostics(self):
        feat = {}
        data = {
            "market_data_status": "current",
            "halt_status": "clear",
            "corporate_action_filter_passed": True,
        }
        assert market_data_quality_rejection(feat, data) is None
        assert feat == data

    def test_quote_rejection_good_quote(self):
        data = {
            "candidate_eval_bid": 4.34,
            "candidate_eval_ask": 4.36,
            "candidate_eval_quote_timestamp": "2026-05-15T14:01:00Z",
            "quote_age_ms": 850,
            "quote_freshness_max_ms": 1000,
        }
        assert quote_rejection(data) is None

    def test_quote_rejection_missing_invalid_or_stale(self):
        data = {
            "candidate_eval_bid": 4.34,
            "candidate_eval_ask": 4.36,
            "candidate_eval_quote_timestamp": "2026-05-15T14:01:00Z",
            "quote_age_ms": 850,
            "quote_freshness_max_ms": 1000,
        }

        missing = dict(data)
        del missing["candidate_eval_ask"]
        assert quote_rejection(missing) == "quote_unavailable"

        invalid = dict(data, candidate_eval_bid="not-a-number")
        assert quote_rejection(invalid) == "quote_unavailable"

        stale = dict(data, quote_age_ms=1250)
        assert quote_rejection(stale) == "quote_unavailable"


# -----------------------------------------------------------------------
# Fixture detector: detection logic
# -----------------------------------------------------------------------

class TestFixtureDetector:
    def test_fires_above_threshold(self):
        det = FixtureDetector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"fixture_score": 0.95, "price": 5.0},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert result.has_signal
        assert len(result.signals) == 1
        assert result.signals[0].direction == SignalDirection.LONG
        assert result.signals[0].raw_signal_strength == 0.95

    def test_no_signal_below_threshold(self):
        det = FixtureDetector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"fixture_score": 0.50, "price": 5.0},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert not result.has_signal
        assert result.features is not None

    def test_missing_lineage_warns(self):
        det = FixtureDetector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"fixture_score": 0.95, "price": 5.0},
            lineage_hashes=[],
        )
        result = det.detect(inp)
        assert any("lineage" in w for w in result.warnings)
        assert result.quality_flags.get("missing_lineage") is True
        # Still fires — missing lineage is a flag, not a blocker
        assert result.has_signal

    def test_future_timestamp_warns(self):
        det = FixtureDetector()
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=future,
            market_data={"fixture_score": 0.95, "price": 5.0},
            lineage_hashes=["hash1"],
        )
        result = det.detect(inp)
        assert any("future" in w for w in result.warnings)
        assert result.quality_flags.get("future_timestamp") is True

    def test_deterministic_hashes(self):
        det = FixtureDetector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"fixture_score": 0.95, "price": 5.0},
            lineage_hashes=["hash1"],
        )
        r1 = det.detect(inp)
        r2 = det.detect(inp)
        assert r1.input_hashes == r2.input_hashes
        assert r1.output_hashes == r2.output_hashes

    def test_missing_asof_raises(self):
        det = FixtureDetector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=None,
            market_data={"fixture_score": 0.95, "price": 5.0},
        )
        with pytest.raises(ValueError, match="asof_timestamp"):
            det.detect(inp)


# -----------------------------------------------------------------------
# Evidence bridge
# -----------------------------------------------------------------------

class TestEvidenceBridge:
    def _run_detector(self, db_session, *, score=0.95, price=5.0):
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session,
            provider="FMP",
            endpoint="/stable/quote",
            asof_timestamp=_ts(),
            raw_payload={"price": price},
            job_run_id=run.job_run_id,
        )
        det = FixtureDetector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"fixture_score": score, "price": price},
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
        return run, lineage, result, persisted

    def test_signal_writes_feature_and_signal(self, db_session):
        run, lineage, result, persisted = self._run_detector(db_session, score=0.95)

        assert persisted.feature_snapshot_id is not None
        assert len(persisted.signal_ids) == 1

        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        assert feat is not None
        assert feat.pattern_id == "FIXTURE"
        assert feat.ticker == "ACME"

        sig = db_session.get(SignalRegistry, persisted.signal_ids[0])
        assert sig is not None
        assert sig.feature_snapshot_id == persisted.feature_snapshot_id

    def test_no_signal_writes_feature_only(self, db_session):
        run, lineage, result, persisted = self._run_detector(db_session, score=0.50)

        assert persisted.feature_snapshot_id is not None
        assert len(persisted.signal_ids) == 0

        feats = db_session.query(FeatureSnapshot).all()
        assert len(feats) == 1

        sigs = db_session.query(SignalRegistry).all()
        assert len(sigs) == 0

    def test_signal_links_to_feature_snapshot(self, db_session):
        _, _, _, persisted = self._run_detector(db_session, score=0.95)

        sig = db_session.get(SignalRegistry, persisted.signal_ids[0])
        assert sig.feature_snapshot_id == persisted.feature_snapshot_id

        feat = sig.feature_snapshot
        assert feat is not None
        assert feat.feature_snapshot_id == persisted.feature_snapshot_id

    def test_job_run_id_preserved(self, db_session):
        run, _, _, persisted = self._run_detector(db_session, score=0.95)

        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        assert feat.job_run_id == run.job_run_id

        sig = db_session.get(SignalRegistry, persisted.signal_ids[0])
        assert sig.job_run_id == run.job_run_id

    def test_universe_snapshot_id_preserved(self, db_session):
        run = _setup_run(db_session)
        lineage = record_data_lineage(
            db_session,
            provider="FMP",
            endpoint="/stable/quote",
            asof_timestamp=_ts(),
            raw_payload={"price": 5.0},
            job_run_id=run.job_run_id,
        )

        from alpha.evidence.writer import record_universe_snapshot

        usn = record_universe_snapshot(
            db_session,
            ticker="ACME",
            asof_timestamp=_ts(),
            operating_universe_inclusion=True,
            job_run_id=run.job_run_id,
        )

        det = FixtureDetector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={"fixture_score": 0.95, "price": 5.0},
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

    def test_fidelity_and_route_class_preserved(self, db_session):
        _, _, _, persisted = self._run_detector(db_session, score=0.95)

        feat = db_session.get(FeatureSnapshot, persisted.feature_snapshot_id)
        assert feat.fidelity_tier == FidelityTier.FULL
        assert feat.feature_manifest_version == "fixture-v1"

        sig = db_session.get(SignalRegistry, persisted.signal_ids[0])
        assert sig.fidelity_tier == FidelityTier.FULL
        assert sig.route_class == "A"
        assert sig.thesis_category == "right_tail_convex"
        assert sig.signal_horizon == "10d"
        assert sig.data_confidence == 0.95
        assert sig.signal_event_sequence == 1

    def test_multiple_signals_get_deterministic_sequence(self, db_session):
        run = _setup_run(db_session)
        result = PatternDetectionResult(
            pattern_id="FIXTURE",
            ticker="ACME",
            asof_timestamp=_ts(),
            features=PatternFeatures(features={
                "x": 1.0,
                "signal_identity_hash": stable_hash({
                    "pattern_id": "FIXTURE",
                    "ticker": "ACME",
                    "setup": "multi",
                }),
            }),
            signals=[
                PatternSignal(
                    direction=SignalDirection.LONG,
                    raw_signal_strength=0.91,
                    raw_expected_edge=0.05,
                ),
                PatternSignal(
                    direction=SignalDirection.LONG,
                    raw_signal_strength=0.93,
                    raw_expected_edge=0.06,
                ),
            ],
        )
        persisted = persist_detection_result(
            db_session,
            result,
            FixtureDetector(),
            job_run_id=run.job_run_id,
        )
        db_session.flush()

        signals = (
            db_session.query(SignalRegistry)
            .filter(SignalRegistry.signal_id.in_(persisted.signal_ids))
            .order_by(SignalRegistry.signal_event_sequence)
            .all()
        )
        assert [s.signal_event_sequence for s in signals] == [1, 2]

    def test_no_features_writes_nothing(self, db_session):
        result = PatternDetectionResult(
            pattern_id="FIXTURE",
            ticker="ACME",
            asof_timestamp=_ts(),
            features=None,
        )
        det = FixtureDetector()
        persisted = persist_detection_result(db_session, result, det)

        assert persisted.feature_snapshot_id is None
        assert len(persisted.signal_ids) == 0
        assert db_session.query(FeatureSnapshot).count() == 0

    def test_deterministic_feature_hash(self, db_session):
        """Same input produces same feature_hash across runs."""
        _, _, _, p1 = self._run_detector(db_session, score=0.95)
        _, _, _, p2 = self._run_detector(db_session, score=0.95)

        f1 = db_session.get(FeatureSnapshot, p1.feature_snapshot_id)
        f2 = db_session.get(FeatureSnapshot, p2.feature_snapshot_id)
        assert f1.feature_hash == f2.feature_hash


# -----------------------------------------------------------------------
# PatternId roster completeness
# -----------------------------------------------------------------------

class TestPatternIdRoster:
    def test_all_17_patterns(self):
        assert len(PatternId.ALL) == 17
        assert set(PatternId.ALL) == {
            "M1", "M2", "M3", "M4", "M5", "M6", "M7",
            "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10",
        }
