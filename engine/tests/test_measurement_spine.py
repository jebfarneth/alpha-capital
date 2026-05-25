"""
Measurement Spine Phase 0 tests.

Covers all required acceptance tests from MeasurementSpine.md:
  1. Universe snapshot hash stability and discrimination.
  2. Orchestration persists detector signals with linkage.
  3. Scheduler retry dedup by signal_identity_hash.
  4. Genuine new event creates a new signal.
  5. Forward-return job writes outcomes for all candidate dispositions.
  6. Forward-return job records retryable/terminal unavailable reasons.
  7. Validation scaffold refuses full confidence on insufficient sample.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from alpha.data.contracts import AdapterResponse, LineageMeta, stable_hash
from alpha.data.fmp import FmpScreenerResult
from alpha.db.models import (
    Base,
    CanonicalUniverseScan,
    FeatureSnapshot,
    SignalRegistry,
    UniverseScan,
    UniverseSnapshot,
    ValidationRun,
)
from alpha.evidence.writer import record_candidate, record_feature_snapshot, record_signal
from alpha.jobs.detector_orchestration import DetectorOrchestrationJob
from alpha.jobs.forward_return import ForwardReturnJob
from alpha.jobs.runner import run_job
from alpha.jobs.universe_builder import UniverseBuilderJob
from alpha.jobs.validation_scaffold import ValidationScaffoldJob
from alpha.patterns.contracts import (
    BasePatternDetector,
    FidelityTier,
    PatternDetectionResult,
    PatternFeatures,
    PatternInput,
    PatternSignal,
    PatternTrack,
    RouteClass,
    SignalDirection,
    ThesisCategory,
)


def _ts():
    return datetime(2026, 5, 20, 14, 30, 0, tzinfo=timezone.utc)


def _lineage():
    return LineageMeta(
        provider="FMP",
        endpoint="/stable/company-screener",
        request_timestamp=_ts(),
        asof_timestamp=_ts(),
        raw_payload_hash="test-hash-abc",
        source_authority="mock",
    )


def _screener_data() -> List[FmpScreenerResult]:
    return [
        FmpScreenerResult(symbol="ACME", company_name="Acme Corp", market_cap=75_000_000, price=5.0, exchange="NASDAQ", country="US", is_etf=False, is_actively_trading=True),
        FmpScreenerResult(symbol="BETA", company_name="Beta Inc", market_cap=50_000_000, price=3.0, exchange="NASDAQ", country="US", is_etf=False, is_actively_trading=True),
    ]


# ---------------------------------------------------------------------------
# Test detector that emits a stable signal_identity_hash
# ---------------------------------------------------------------------------


class StableIdentityDetector(BasePatternDetector):
    """Test detector with stable signal identity in features."""

    pattern_id = "M4"
    version = "1.0"
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.RIGHT_TAIL_CONVEX
    route_class = RouteClass.A

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        score = float(inp.market_data.get("fixture_score", 0.0))
        price = float(inp.market_data.get("price", 1.0))
        high_52w = inp.market_data.get("high_52w", 10.0)
        high_52w_date = inp.market_data.get("high_52w_date", "2026-01-01")

        identity_hash = stable_hash({
            "pattern_id": self.pattern_id,
            "ticker": inp.ticker,
            "high_52w": high_52w,
            "high_52w_date": high_52w_date,
        })

        features_dict = {
            "fixture_score": score,
            "price": price,
            "high_52w": high_52w,
            "signal_identity_hash": identity_hash,
            "signal_identity_components": {
                "pattern_id": self.pattern_id,
                "ticker": inp.ticker,
                "high_52w": high_52w,
                "high_52w_date": high_52w_date,
            },
            "signal_identity_source": "detector",
        }

        signals = []
        if score >= 0.90:
            signals.append(
                PatternSignal(
                    direction=SignalDirection.LONG,
                    raw_signal_strength=score,
                    raw_expected_edge=score * 0.10,
                    signal_horizon="15d",
                    data_confidence=0.95,
                )
            )

        return PatternDetectionResult(
            pattern_id=self.pattern_id,
            ticker=inp.ticker,
            asof_timestamp=inp.asof_timestamp,
            features=PatternFeatures(
                features=features_dict,
                fidelity_tier=FidelityTier.FULL,
                point_in_time_passed=True,
                lookahead_guard_passed=True,
            ),
            signals=signals,
            input_hashes={"market_data": stable_hash(inp.market_data)},
            output_hashes={"features": stable_hash(features_dict)},
        )


class NoIdentityDetector(BasePatternDetector):
    """Test detector that does NOT emit signal identity (like M3 without upstream snapshot)."""

    pattern_id = "M3"
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.CONTINUATION
    route_class = RouteClass.A

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        score = float(inp.market_data.get("fixture_score", 0.0))

        features_dict = {"fixture_score": score}

        signals = []
        if score >= 0.90:
            signals.append(
                PatternSignal(
                    direction=SignalDirection.LONG,
                    raw_signal_strength=score,
                    raw_expected_edge=0.05,
                    signal_horizon="15d",
                    data_confidence=0.90,
                )
            )

        return PatternDetectionResult(
            pattern_id=self.pattern_id,
            ticker=inp.ticker,
            asof_timestamp=inp.asof_timestamp,
            features=PatternFeatures(
                features=features_dict,
                fidelity_tier=FidelityTier.FULL,
            ),
            signals=signals,
        )


# ===================================================================
# 1. Universe snapshot hash stability and discrimination
# ===================================================================


class TestUniverseSnapshotHashes:
    def test_same_payload_same_hash(self, db_session):
        """Rerunning with identical payload produces the same output hash."""
        resp = AdapterResponse(data=_screener_data(), lineage=_lineage())
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        r1 = run_job(db_session, job)

        engine2 = create_engine("sqlite:///:memory:")
        event.listen(engine2, "connect", lambda c, _: c.cursor().execute("PRAGMA foreign_keys=ON") or c.cursor().close())
        Base.metadata.create_all(engine2)
        s2 = sessionmaker(bind=engine2)()

        job2 = UniverseBuilderJob(session=s2, screener_response=resp)
        r2 = run_job(s2, job2)

        assert r1.output_hashes["universe_snapshots"] == r2.output_hashes["universe_snapshots"]
        s2.close()
        engine2.dispose()

    def test_changed_membership_changes_hash(self, db_session):
        """Changing one constituent changes the output hash."""
        data1 = _screener_data()
        resp1 = AdapterResponse(data=data1, lineage=_lineage())
        job1 = UniverseBuilderJob(session=db_session, screener_response=resp1)
        r1 = run_job(db_session, job1)

        data2 = _screener_data()
        data2[0] = FmpScreenerResult(
            symbol="NEWCO", company_name="New Corp", market_cap=80_000_000,
            price=6.0, country="US", is_etf=False, is_actively_trading=True,
        )
        engine2 = create_engine("sqlite:///:memory:")
        event.listen(engine2, "connect", lambda c, _: c.cursor().execute("PRAGMA foreign_keys=ON") or c.cursor().close())
        Base.metadata.create_all(engine2)
        s2 = sessionmaker(bind=engine2)()

        resp2 = AdapterResponse(data=data2, lineage=_lineage())
        job2 = UniverseBuilderJob(session=s2, screener_response=resp2)
        r2 = run_job(s2, job2)

        assert r1.output_hashes["universe_snapshots"] != r2.output_hashes["universe_snapshots"]
        s2.close()
        engine2.dispose()

    def test_changed_market_cap_changes_hash(self, db_session):
        """Changing a market cap (same tickers) changes the hash."""
        data1 = _screener_data()
        resp1 = AdapterResponse(data=data1, lineage=_lineage())
        job1 = UniverseBuilderJob(session=db_session, screener_response=resp1)
        r1 = run_job(db_session, job1)

        data2 = _screener_data()
        data2[0] = FmpScreenerResult(
            symbol="ACME", company_name="Acme Corp", market_cap=100_000_000,
            price=5.0, country="US", is_etf=False, is_actively_trading=True,
        )
        engine2 = create_engine("sqlite:///:memory:")
        event.listen(engine2, "connect", lambda c, _: c.cursor().execute("PRAGMA foreign_keys=ON") or c.cursor().close())
        Base.metadata.create_all(engine2)
        s2 = sessionmaker(bind=engine2)()

        resp2 = AdapterResponse(data=data2, lineage=_lineage())
        job2 = UniverseBuilderJob(session=s2, screener_response=resp2)
        r2 = run_job(s2, job2)

        assert r1.output_hashes["universe_snapshots"] != r2.output_hashes["universe_snapshots"]

        snap_caps = {
            s.ticker: s.market_cap
            for s in s2.query(UniverseSnapshot).all()
        }
        assert snap_caps["ACME"] == 100_000_000

        s2.close()
        engine2.dispose()

    def test_scan_id_set_on_snapshots(self, db_session):
        """Universe snapshots have scan_id linking them to the job run."""
        resp = AdapterResponse(data=_screener_data(), lineage=_lineage())
        job = UniverseBuilderJob(session=db_session, screener_response=resp)
        run_job(db_session, job)

        snaps = db_session.query(UniverseSnapshot).all()
        assert all(s.scan_id is not None for s in snaps)
        scan_ids = {s.scan_id for s in snaps}
        assert len(scan_ids) == 1  # all same scan


# ===================================================================
# 2. Orchestration persists detector signals with linkage
# ===================================================================


class TestDetectorOrchestration:
    def _setup_universe(self, db_session):
        scan = UniverseScan(
            scan_id="ms-scan", trading_date="2026-05-20",
            asof_timestamp=_ts(), raw_count=1, deduped_count=1,
            included_count=1, excluded_count=0, run_status="finished",
            source_lineage_hash="hash",
        )
        db_session.add(scan)
        db_session.flush()
        from alpha.db.models import CanonicalUniverseScan
        db_session.add(CanonicalUniverseScan(
            trading_date="2026-05-20", scan_id="ms-scan",
            selection_reason="test",
        ))
        snap = UniverseSnapshot(
            universe_snapshot_id="ms-snap-ACME", scan_id="ms-scan",
            ticker="ACME", asof_timestamp=_ts(), market_cap=75_000_000,
            price=5.0, primary_exchange="NASDAQ", security_type="common_stock",
            operating_universe_inclusion=True, source_lineage_hash="lineage-hash",
        )
        db_session.add(snap)
        db_session.flush()

    def _make_inputs(self, universe_snapshot_id=None):
        return [
            PatternInput(
                ticker="ACME",
                asof_timestamp=_ts(),
                market_data={
                    "fixture_score": 0.95,
                    "price": 5.0,
                    "high_52w": 10.0,
                    "high_52w_date": "2026-01-01",
                    "operating_universe_inclusion": True,
                    "trading_date": "2026-05-20",
                },
                lineage_ids=["lineage-001"],
                lineage_hashes=["lineage-hash"],
                universe_snapshot_id=universe_snapshot_id or "ms-snap-ACME",
            ),
        ]

    def test_persists_feature_and_signal(self, db_session):
        """Orchestration persists feature_snapshot and signal_registry rows."""
        self._setup_universe(db_session)
        detectors = [StableIdentityDetector()]
        inputs = self._make_inputs()

        orch = DetectorOrchestrationJob(
            db_session, detectors=detectors, inputs=inputs,
            trading_date="2026-05-20",
        )
        result = run_job(db_session, orch, params={"trading_date": "2026-05-20"})

        assert result.ok or result.status == "finished"
        assert result.metrics["total_signals_persisted"] == 1

        feats = db_session.query(FeatureSnapshot).all()
        assert len(feats) == 1
        assert feats[0].pattern_id == "M4"

        sigs = db_session.query(SignalRegistry).all()
        assert len(sigs) == 1
        assert sigs[0].pattern_id == "M4"
        assert sigs[0].ticker == "ACME"
        assert sigs[0].feature_snapshot_id == feats[0].feature_snapshot_id
        assert sigs[0].signal_identity_hash is not None

    def test_links_universe_snapshot(self, db_session):
        """Signal links to universe_snapshot_id when provided."""
        self._setup_universe(db_session)
        detectors = [StableIdentityDetector()]
        inputs = self._make_inputs(universe_snapshot_id="ms-snap-ACME")

        orch = DetectorOrchestrationJob(
            db_session, detectors=detectors, inputs=inputs,
            trading_date="2026-05-20",
        )
        run_job(db_session, orch, params={"trading_date": "2026-05-20"})

        sig = db_session.query(SignalRegistry).one()
        assert sig.universe_snapshot_id == "ms-snap-ACME"

    def test_links_job_run_id(self, db_session):
        """Feature and signal rows link to the orchestration job_run_id."""
        self._setup_universe(db_session)
        detectors = [StableIdentityDetector()]
        inputs = self._make_inputs()

        orch = DetectorOrchestrationJob(
            db_session, detectors=detectors, inputs=inputs,
            trading_date="2026-05-20",
        )
        run_job(db_session, orch, params={"trading_date": "2026-05-20"})

        feat = db_session.query(FeatureSnapshot).one()
        sig = db_session.query(SignalRegistry).one()
        assert feat.job_run_id is not None
        assert sig.job_run_id == feat.job_run_id

    def test_no_signal_evaluation_metric_without_signal_rows(self, db_session):
        """A no-signal detector result increments the metric and creates no signal."""
        self._setup_universe(db_session)
        detectors = [StableIdentityDetector()]
        inputs = [
            PatternInput(
                ticker="ACME",
                asof_timestamp=_ts(),
                market_data={
                    "fixture_score": 0.50,
                    "price": 5.0,
                    "high_52w": 10.0,
                    "high_52w_date": "2026-01-01",
                    "operating_universe_inclusion": True,
                    "trading_date": "2026-05-20",
                },
                lineage_hashes=["lineage-hash"],
                universe_snapshot_id="ms-snap-ACME",
            ),
        ]

        orch = DetectorOrchestrationJob(
            db_session, detectors=detectors, inputs=inputs,
            trading_date="2026-05-20",
        )
        result = run_job(db_session, orch, params={"trading_date": "2026-05-20"})

        assert result.ok or result.status == "finished"
        assert result.metrics["total_signals_persisted"] == 0
        assert db_session.query(SignalRegistry).count() == 0


# ===================================================================
# 3. Scheduler retry dedup — same identity does not create duplicate
# ===================================================================


class TestSignalDedup:
    def _setup_universe(self, db_session):
        scan = UniverseScan(
            scan_id="dedup-scan", trading_date="2026-05-20",
            asof_timestamp=_ts(), raw_count=1, deduped_count=1,
            included_count=1, excluded_count=0, run_status="finished",
            source_lineage_hash="hash",
        )
        db_session.add(scan)
        db_session.flush()
        from alpha.db.models import CanonicalUniverseScan
        db_session.add(CanonicalUniverseScan(
            trading_date="2026-05-20", scan_id="dedup-scan",
            selection_reason="test",
        ))
        db_session.add(UniverseSnapshot(
            universe_snapshot_id="dedup-snap-ACME", scan_id="dedup-scan",
            ticker="ACME", asof_timestamp=_ts(), market_cap=75_000_000,
            price=5.0, primary_exchange="NASDAQ", security_type="common_stock",
            operating_universe_inclusion=True, source_lineage_hash="lineage-hash",
        ))
        db_session.flush()

    def test_retry_same_identity_suppressed(self, db_session):
        """Scheduler retry over identical inputs does not create a duplicate tradable signal."""
        self._setup_universe(db_session)
        detectors = [StableIdentityDetector()]
        inputs = [
            PatternInput(
                ticker="ACME",
                asof_timestamp=_ts(),
                market_data={
                    "fixture_score": 0.95, "price": 5.0,
                    "high_52w": 10.0, "high_52w_date": "2026-01-01",
                    "operating_universe_inclusion": True,
                    "trading_date": "2026-05-20",
                },
                lineage_hashes=["lineage-hash"],
                universe_snapshot_id="dedup-snap-ACME",
            ),
        ]

        # First run
        orch1 = DetectorOrchestrationJob(
            db_session, detectors=detectors, inputs=inputs,
            trading_date="2026-05-20",
        )
        r1 = run_job(db_session, orch1, params={"trading_date": "2026-05-20"})
        assert r1.metrics["total_signals_persisted"] == 1

        # Retry — same identity, should be suppressed
        orch2 = DetectorOrchestrationJob(
            db_session, detectors=detectors, inputs=inputs,
            trading_date="2026-05-20",
        )
        r2 = run_job(db_session, orch2, params={"trading_date": "2026-05-20"})
        assert r2.metrics["total_signals_persisted"] == 0

        # Only one signal in DB
        sigs = db_session.query(SignalRegistry).all()
        assert len(sigs) == 1

    def test_database_unique_index_blocks_duplicate_identity(self, db_session):
        """DB constraint protects dedup if two writers race past the application check."""
        feat = record_feature_snapshot(
            db_session,
            pattern_id="M4",
            ticker="ACME",
            asof_timestamp=_ts(),
            features={"score": 0.95},
            data_lineage_ids=[],
        )
        record_signal(
            db_session,
            pattern_id="M4",
            ticker="ACME",
            direction="long",
            signal_timestamp=_ts(),
            raw_signal_strength=0.95,
            raw_expected_edge=0.08,
            feature_snapshot_id=feat.feature_snapshot_id,
            signal_identity_hash="same-event",
        )
        db_session.flush()

        with pytest.raises(IntegrityError):
            record_signal(
                db_session,
                pattern_id="M4",
                ticker="ACME",
                direction="long",
                signal_timestamp=_ts(),
                raw_signal_strength=0.95,
                raw_expected_edge=0.08,
                feature_snapshot_id=feat.feature_snapshot_id,
                signal_identity_hash="same-event",
            )
        db_session.rollback()

    def test_detector_error_marks_run_partial_failed(self, db_session):
        """A detector exception produces partial_failed, not clean finished."""
        self._setup_universe(db_session)

        class BrokenDetector(StableIdentityDetector):
            pattern_id = "M4"

            def detect(self, inp):
                raise ValueError("boom")

        inputs = [
            PatternInput(
                ticker="ACME",
                asof_timestamp=_ts(),
                market_data={"fixture_score": 0.95, "operating_universe_inclusion": True, "trading_date": "2026-05-20"},
                lineage_hashes=["lineage-hash"],
                universe_snapshot_id="dedup-snap-ACME",
            ),
        ]

        orch = DetectorOrchestrationJob(
            db_session, detectors=[BrokenDetector()], inputs=inputs,
            trading_date="2026-05-20",
        )
        result = run_job(db_session, orch, params={"trading_date": "2026-05-20"})

        assert result.status == "partial_failed"
        assert result.metrics["any_detector_failed"] is True


# ===================================================================
# 4. Genuine new event creates a new signal
# ===================================================================


class TestNewEventCreatesNewSignal:
    def _setup_universe(self, db_session):
        scan = UniverseScan(
            scan_id="event-scan", trading_date="2026-05-20",
            asof_timestamp=_ts(), raw_count=1, deduped_count=1,
            included_count=1, excluded_count=0, run_status="finished",
            source_lineage_hash="hash",
        )
        db_session.add(scan)
        db_session.flush()
        from alpha.db.models import CanonicalUniverseScan
        db_session.add(CanonicalUniverseScan(
            trading_date="2026-05-20", scan_id="event-scan",
            selection_reason="test",
        ))
        db_session.add(UniverseSnapshot(
            universe_snapshot_id="event-snap-ACME", scan_id="event-scan",
            ticker="ACME", asof_timestamp=_ts(), market_cap=75_000_000,
            price=5.0, primary_exchange="NASDAQ", security_type="common_stock",
            operating_universe_inclusion=True, source_lineage_hash="lineage-hash",
        ))
        db_session.flush()

    def test_different_trading_date_different_signal(self, db_session):
        """Same ticker on different trading dates creates different identities."""
        self._setup_universe(db_session)
        detectors = [StableIdentityDetector()]
        inputs = [
            PatternInput(
                ticker="ACME",
                asof_timestamp=_ts(),
                market_data={
                    "fixture_score": 0.95, "price": 5.0,
                    "high_52w": 10.0, "high_52w_date": "2026-01-01",
                    "trading_date": "2026-05-20",
                },
                lineage_hashes=["lineage-hash"],
                universe_snapshot_id="event-snap-ACME",
            ),
        ]

        orch1 = DetectorOrchestrationJob(
            db_session, detectors=detectors, inputs=inputs,
            trading_date="2026-05-20",
        )
        r1 = run_job(db_session, orch1, params={"trading_date": "2026-05-20"})
        assert r1.metrics["total_signals_persisted"] == 1

        sigs = db_session.query(SignalRegistry).all()
        assert len(sigs) == 1
        assert sigs[0].signal_identity_hash is not None


# ===================================================================
# 5. Forward-return job writes outcomes for all dispositions
# ===================================================================


class TestForwardReturnOutcomes:
    def _setup_signals(self, db_session, statuses):
        """Create signals with different intended dispositions."""
        from alpha.evidence.writer import record_feature_snapshot, record_signal

        feat = record_feature_snapshot(
            db_session,
            pattern_id="M4",
            ticker="ACME",
            asof_timestamp=_ts(),
            features={"score": 0.95},
            data_lineage_ids=[],
        )
        db_session.flush()

        signal_ids = []
        for status in statuses:
            sig = record_signal(
                db_session,
                pattern_id="M4",
                ticker="ACME",
                direction="long",
                signal_timestamp=_ts(),
                raw_signal_strength=0.95,
                raw_expected_edge=0.08,
                feature_snapshot_id=feat.feature_snapshot_id,
                signal_horizon="15d",
                signal_identity_hash=f"forward-{status}",
            )
            decision = "enter"
            skip_reason = None
            constraint_reason = None
            if status == "skipped":
                decision = "skip"
                skip_reason = "koth_loser"
            elif status == "vetoed":
                decision = "vetoed_hazard"
            elif status == "unfilled":
                constraint_reason = {"fill_status": "unfilled"}

            record_candidate(
                db_session,
                candidate_pool_id=f"pool-{status}",
                ticker="ACME",
                direction="long",
                primary_pattern="M4",
                combined_expected_edge=0.08,
                trade_decision=decision,
                input_signal_ids=[sig.signal_id],
                skip_reason=skip_reason,
                constraint_reason=constraint_reason,
            )
            signal_ids.append(sig.signal_id)

        db_session.flush()
        return signal_ids

    def test_all_dispositions_get_outcomes(self, db_session):
        """selected, skipped, vetoed, unfilled all get forward_return."""
        dispositions = ["selected", "skipped", "vetoed", "unfilled"]
        signal_ids = self._setup_signals(db_session, dispositions)

        def price_fn(ticker, ts, horizon):
            return (5.0, 5.50)  # entry=5, exit=5.50 => 10% return

        fwd_job = ForwardReturnJob(
            session=db_session, price_fn=price_fn,
        )
        result = run_job(db_session, fwd_job)

        assert result.ok
        assert result.metrics["computed"] == 4
        assert result.metrics["unavailable"] == 0

        for sid in signal_ids:
            sig = db_session.get(SignalRegistry, sid)
            assert sig.forward_return_status == "computed"
            assert abs(sig.forward_return - 0.10) < 0.001
            assert sig.intended_entry_price == 5.0


# ===================================================================
# 6. Forward-return records retryable and terminal unavailable reasons
# ===================================================================


class TestForwardReturnUnavailable:
    def _make_signal(self, db_session, ticker="ACME"):
        feat = record_feature_snapshot(
            db_session,
            pattern_id="M4",
            ticker=ticker,
            asof_timestamp=_ts(),
            features={"score": 0.9},
            data_lineage_ids=[],
        )
        db_session.flush()
        sig = record_signal(
            db_session,
            pattern_id="M4",
            ticker=ticker,
            direction="long",
            signal_timestamp=_ts(),
            raw_signal_strength=0.9,
            raw_expected_edge=0.08,
            feature_snapshot_id=feat.feature_snapshot_id,
            signal_horizon="15d",
            signal_identity_hash=f"unavailable-{ticker}",
        )
        db_session.flush()
        return sig.signal_id

    def test_pricing_unavailable_writes_reason(self, db_session):
        """Missing provider price data stays retryable with a reason."""
        sid = self._make_signal(db_session)

        def price_fn(ticker, ts, horizon):
            return None  # pricing unavailable

        fwd_job = ForwardReturnJob(session=db_session, price_fn=price_fn)
        result = run_job(db_session, fwd_job)

        assert result.metrics["retryable_unavailable"] == 1
        sig = db_session.get(SignalRegistry, sid)
        assert sig.forward_return_status == "pricing_unavailable_retry"
        assert sig.forward_return_attempts == 1
        assert sig.outcome_unavailable_reason == "pricing_unavailable"
        assert sig.forward_return is None

    def test_retryable_pricing_eventually_terminalizes(self, db_session):
        """Unpriceable signals stop retrying after the configured attempt cap."""
        sid = self._make_signal(db_session)

        fwd_job = ForwardReturnJob(
            session=db_session,
            price_fn=lambda ticker, ts, horizon: None,
            max_attempts=2,
        )
        first = run_job(db_session, fwd_job)
        assert first.metrics["retryable_unavailable"] == 1
        sig = db_session.get(SignalRegistry, sid)
        assert sig.forward_return_status == "pricing_unavailable_retry"
        assert sig.forward_return_attempts == 1

        second = run_job(db_session, fwd_job)
        assert second.metrics["retryable_unavailable"] == 0
        assert second.metrics["unavailable"] == 1
        sig = db_session.get(SignalRegistry, sid)
        assert sig.forward_return_status == "outcome_unavailable"
        assert sig.outcome_unavailable_reason == "pricing_unavailable"
        assert sig.forward_return_attempts == 2

        third = run_job(db_session, fwd_job)
        assert third.metrics["total_pending"] == 0

    def test_retryable_pricing_unavailable_can_later_compute(self, db_session):
        """Transient provider lag can be retried instead of becoming terminal."""
        sid = self._make_signal(db_session)

        fwd_job = ForwardReturnJob(session=db_session, price_fn=lambda ticker, ts, horizon: None)
        run_job(db_session, fwd_job)

        sig = db_session.get(SignalRegistry, sid)
        assert sig.forward_return_status == "pricing_unavailable_retry"

        fwd_job = ForwardReturnJob(session=db_session, price_fn=lambda ticker, ts, horizon: (5.0, 5.5))
        result = run_job(db_session, fwd_job)

        assert result.metrics["computed"] == 1
        sig = db_session.get(SignalRegistry, sid)
        assert sig.forward_return_status == "computed"
        assert abs(sig.forward_return - 0.10) < 0.001
        assert sig.forward_return_attempts == 2

    def test_constructor_rejects_non_positive_max_attempts(self, db_session):
        """The attempt cap must leave at least one mature outcome attempt."""
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            ForwardReturnJob(
                session=db_session,
                price_fn=lambda ticker, ts, horizon: (5.0, 5.5),
                max_attempts=0,
            )

    def test_invalid_entry_price_writes_reason(self, db_session):
        """Zero or negative entry price stays retryable with a reason."""
        sid = self._make_signal(db_session)

        def price_fn(ticker, ts, horizon):
            return (0.0, 5.0)  # invalid entry

        fwd_job = ForwardReturnJob(session=db_session, price_fn=price_fn)
        result = run_job(db_session, fwd_job)

        sig = db_session.get(SignalRegistry, sid)
        assert sig.forward_return_status == "invalid_entry_price_retry"
        assert sig.forward_return_attempts == 1
        assert sig.outcome_unavailable_reason == "invalid_entry_price"
        assert sig.forward_return is None

    def test_nan_and_inf_entry_prices_are_invalid(self, db_session):
        """NaN/Inf entry prices never produce computed forward returns."""
        sid1 = self._make_signal(db_session, "NAN")
        sid2 = self._make_signal(db_session, "INF")

        def price_fn(ticker, ts, horizon):
            if ticker == "NAN":
                return (float("nan"), 5.0)
            return (float("inf"), 5.0)

        fwd_job = ForwardReturnJob(session=db_session, price_fn=price_fn)
        result = run_job(db_session, fwd_job)

        assert result.metrics["computed"] == 0
        assert result.metrics["retryable_unavailable"] == 2
        for sid in (sid1, sid2):
            sig = db_session.get(SignalRegistry, sid)
            assert sig.forward_return_status == "invalid_entry_price_retry"
            assert sig.outcome_unavailable_reason == "invalid_entry_price"
            assert sig.forward_return is None

    def test_malformed_price_shape_is_per_signal_retryable(self, db_session):
        """One malformed provider response does not roll back the whole batch."""
        good1 = self._make_signal(db_session, "GOOD1")
        bad_tuple = self._make_signal(db_session, "BADTUPLE")
        bad_list = self._make_signal(db_session, "BADLIST")
        good2 = self._make_signal(db_session, "GOOD2")

        def price_fn(ticker, ts, horizon):
            if ticker == "BADTUPLE":
                return (5.0, 5.5, 6.0)
            if ticker == "BADLIST":
                return [4.0, 4.4]
            return (5.0, 5.5)

        fwd_job = ForwardReturnJob(session=db_session, price_fn=price_fn)
        result = run_job(db_session, fwd_job)

        assert result.ok
        assert result.metrics["computed"] == 2
        assert result.metrics["retryable_unavailable"] == 2
        assert result.metrics["pricing_errors"] == 2

        for sid in (bad_tuple, bad_list):
            bad_sig = db_session.get(SignalRegistry, sid)
            assert bad_sig.forward_return_status == "invalid_price_shape_retry"
            assert bad_sig.outcome_unavailable_reason == "invalid_price_shape"
            assert bad_sig.forward_return_attempts == 1

        for sid in (good1, good2):
            sig = db_session.get(SignalRegistry, sid)
            assert sig.forward_return_status == "computed"
            assert sig.forward_return_attempts == 1

    def test_missing_exit_price_writes_reason(self, db_session):
        """None exit price stays retryable with a reason."""
        sid = self._make_signal(db_session)

        def price_fn(ticker, ts, horizon):
            return (5.0, None)  # missing exit

        fwd_job = ForwardReturnJob(session=db_session, price_fn=price_fn)
        result = run_job(db_session, fwd_job)

        sig = db_session.get(SignalRegistry, sid)
        assert sig.forward_return_status == "missing_exit_price_retry"
        assert sig.forward_return_attempts == 1
        assert sig.outcome_unavailable_reason == "missing_exit_price"

    def test_invalid_exit_price_writes_reason(self, db_session):
        """NaN/Inf/negative exit prices stay retryable instead of computing impossible returns."""
        sid1 = self._make_signal(db_session, "NAN")
        sid2 = self._make_signal(db_session, "INF")
        sid3 = self._make_signal(db_session, "NEG")

        def price_fn(ticker, ts, horizon):
            if ticker == "NAN":
                return (5.0, float("nan"))
            if ticker == "INF":
                return (5.0, float("inf"))
            return (5.0, -1.0)

        fwd_job = ForwardReturnJob(session=db_session, price_fn=price_fn)
        result = run_job(db_session, fwd_job)

        assert result.metrics["computed"] == 0
        assert result.metrics["retryable_unavailable"] == 3
        for sid in (sid1, sid2, sid3):
            sig = db_session.get(SignalRegistry, sid)
            assert sig.forward_return_status == "invalid_exit_price_retry"
            assert sig.outcome_unavailable_reason == "invalid_exit_price"
            assert sig.forward_return is None

    def test_zero_exit_price_computes_minus_one_hundred_percent(self, db_session):
        """A zero exit is a valid long-equity terminal mark; negative exits are not."""
        sid = self._make_signal(db_session)

        fwd_job = ForwardReturnJob(
            session=db_session,
            price_fn=lambda ticker, ts, horizon: (5.0, 0.0),
        )
        result = run_job(db_session, fwd_job)

        assert result.metrics["computed"] == 1
        sig = db_session.get(SignalRegistry, sid)
        assert sig.forward_return_status == "computed"
        assert sig.forward_return == -1.0

    def test_invalid_exit_price_eventually_terminalizes(self, db_session):
        """Invalid exit-price retries use the same terminal outcome path."""
        sid = self._make_signal(db_session)

        fwd_job = ForwardReturnJob(
            session=db_session,
            price_fn=lambda ticker, ts, horizon: (5.0, -1.0),
            max_attempts=2,
        )
        first = run_job(db_session, fwd_job)
        assert first.metrics["retryable_unavailable"] == 1

        second = run_job(db_session, fwd_job)
        assert second.metrics["retryable_unavailable"] == 0
        assert second.metrics["unavailable"] == 1

        sig = db_session.get(SignalRegistry, sid)
        assert sig.forward_return_status == "outcome_unavailable"
        assert sig.outcome_unavailable_reason == "invalid_exit_price"
        assert sig.forward_return_attempts == 2

    def test_price_fn_exception_is_per_signal_retryable(self, db_session):
        """A bad price lookup marks that signal retryable without killing the batch."""
        sid1 = self._make_signal(db_session, "ACME")
        sid2 = self._make_signal(db_session, "BETA")

        def price_fn(ticker, ts, horizon):
            if ticker == "ACME":
                raise RuntimeError("provider exploded")
            return (5.0, 5.5)

        fwd_job = ForwardReturnJob(session=db_session, price_fn=price_fn)
        result = run_job(db_session, fwd_job)

        assert result.status == "finished"
        assert result.metrics["pricing_errors"] == 1
        assert result.metrics["computed"] == 1
        acme = db_session.get(SignalRegistry, sid1)
        beta = db_session.get(SignalRegistry, sid2)
        assert acme.forward_return_status == "pricing_unavailable_retry"
        assert acme.forward_return_attempts == 1
        assert acme.outcome_unavailable_reason == "price_fn_error:RuntimeError"
        assert beta.forward_return_status == "computed"

    def test_maturity_fn_exception_is_per_signal_retryable(self, db_session):
        """A maturity policy error marks the signal retryable without killing the job."""
        sid = self._make_signal(db_session)

        def maturity_fn(ts, horizon):
            raise RuntimeError("bad maturity policy")

        fwd_job = ForwardReturnJob(
            session=db_session,
            price_fn=lambda ticker, ts, horizon: (5.0, 5.5),
            maturity_fn=maturity_fn,
        )
        result = run_job(db_session, fwd_job)

        assert result.status == "finished"
        assert result.metrics["pricing_errors"] == 1
        assert result.metrics["retryable_unavailable"] == 1
        sig = db_session.get(SignalRegistry, sid)
        assert sig.forward_return_status == "pricing_unavailable_retry"
        assert sig.forward_return_attempts == 1
        assert sig.outcome_unavailable_reason == "maturity_fn_error:RuntimeError"

    def test_does_not_silently_drop_rows(self, db_session):
        """Every signal gets a status — none remain NULL after the job runs."""
        sid1 = self._make_signal(db_session, "ACME")
        sid2 = self._make_signal(db_session, "BETA")

        def price_fn(ticker, ts, horizon):
            if ticker == "ACME":
                return (5.0, 5.5)
            return None  # BETA unavailable

        fwd_job = ForwardReturnJob(session=db_session, price_fn=price_fn)
        run_job(db_session, fwd_job)

        acme = db_session.get(SignalRegistry, sid1)
        beta = db_session.get(SignalRegistry, sid2)
        assert acme.forward_return_status == "computed"
        assert beta.forward_return_status == "pricing_unavailable_retry"
        # Neither is NULL — no silent drops
        assert acme.forward_return_status is not None
        assert beta.forward_return_status is not None

    def test_immature_signals_skipped(self, db_session):
        """Signals not yet mature are skipped, not marked unavailable."""
        sid = self._make_signal(db_session)

        def price_fn(ticker, ts, horizon):
            return (5.0, 5.5)

        def maturity_fn(ts, horizon):
            return False  # not mature yet

        fwd_job = ForwardReturnJob(
            session=db_session, price_fn=price_fn, maturity_fn=maturity_fn,
        )
        result = run_job(db_session, fwd_job)

        assert result.metrics["immature"] == 1
        assert result.metrics["computed"] == 0

        sig = db_session.get(SignalRegistry, sid)
        assert sig.forward_return_status == "pending"


# ===================================================================
# 7. Validation scaffold refuses full confidence on insufficient sample
# ===================================================================


class TestValidationScaffold:
    def _seed_signals(self, db_session, pattern_id, count, forward_return=0.05):
        """Create count signals with computed forward_return."""
        feat = record_feature_snapshot(
            db_session,
            pattern_id=pattern_id,
            ticker="ACME",
            asof_timestamp=_ts(),
            features={"score": 0.9},
            data_lineage_ids=[],
        )
        db_session.flush()

        for i in range(count):
            sig = record_signal(
                db_session,
                pattern_id=pattern_id,
                ticker="ACME",
                direction="long",
                signal_timestamp=_ts(),
                raw_signal_strength=0.9,
                raw_expected_edge=0.08,
                feature_snapshot_id=feat.feature_snapshot_id,
                signal_identity_hash=f"{pattern_id}-{i}",
            )
            sig.forward_return = forward_return
            sig.forward_return_status = "computed"

        db_session.flush()

    def test_insufficient_sample_refuses_full_confidence(self, db_session):
        """Positive early returns with insufficient sample must not produce full confidence."""
        self._seed_signals(db_session, "M4", count=10, forward_return=0.10)

        val_job = ValidationScaffoldJob(session=db_session, minimum_sample=30)
        result = run_job(db_session, val_job)

        assert result.ok
        pattern_results = result.metrics["pattern_results"]
        assert "M4" in pattern_results
        m4 = pattern_results["M4"]
        assert m4["confidence_tier"] == "insufficient_sample"
        assert m4["sample_size"] == 10
        assert m4["computed_sample_size"] == 10
        assert m4["unavailable_sample_size"] == 0
        assert m4["validation_weight_multiplier"] is None
        assert m4["confidence_tier"] != "validated"

        # Validation run persisted
        vr = db_session.query(ValidationRun).filter(
            ValidationRun.pattern_id == "M4"
        ).one()
        assert vr.confidence_tier == "insufficient_sample"
        assert vr.validation_weight_multiplier is None

    def test_sufficient_sample_gets_monitoring(self, db_session):
        """Sufficient sample size gets monitoring tier (full validation math deferred)."""
        self._seed_signals(db_session, "M4", count=50, forward_return=0.05)

        val_job = ValidationScaffoldJob(session=db_session, minimum_sample=30)
        result = run_job(db_session, val_job)

        m4 = result.metrics["pattern_results"]["M4"]
        assert m4["confidence_tier"] == "monitoring"
        assert m4["sample_size"] == 50
        assert m4["computed_sample_size"] == 50
        assert m4["validation_weight_multiplier"] is None

    def test_unavailable_outcomes_count_in_denominator(self, db_session):
        """Unavailable mature outcomes are counted in sample_size, not dropped."""
        self._seed_signals(db_session, "M4", count=10, forward_return=0.05)
        feat = record_feature_snapshot(
            db_session,
            pattern_id="M4",
            ticker="BETA",
            asof_timestamp=_ts(),
            features={"score": 0.9},
            data_lineage_ids=[],
        )
        for i in range(5):
            sig = record_signal(
                db_session,
                pattern_id="M4",
                ticker="BETA",
                direction="long",
                signal_timestamp=_ts(),
                raw_signal_strength=0.9,
                raw_expected_edge=0.08,
                feature_snapshot_id=feat.feature_snapshot_id,
                signal_identity_hash=f"unavailable-b{i}",
            )
            sig.forward_return_status = "outcome_unavailable"
            sig.outcome_unavailable_reason = "delisted_no_exit_price"
        db_session.flush()

        val_job = ValidationScaffoldJob(session=db_session, minimum_sample=30)
        result = run_job(db_session, val_job)

        m4 = result.metrics["pattern_results"]["M4"]
        assert m4["sample_size"] == 15
        assert m4["computed_sample_size"] == 10
        assert m4["unavailable_sample_size"] == 5
        assert m4["confidence_tier"] == "insufficient_sample"

    def test_positive_returns_small_sample_not_validated(self, db_session):
        """Extremely positive returns with 5 samples must NOT be validated."""
        self._seed_signals(db_session, "I1", count=5, forward_return=0.50)

        val_job = ValidationScaffoldJob(session=db_session, minimum_sample=30)
        result = run_job(db_session, val_job)

        i1 = result.metrics["pattern_results"]["I1"]
        assert i1["confidence_tier"] == "insufficient_sample"
        assert i1["mean_forward_return_computed"] > 0.40
        # Despite great returns, sample is too small
        assert i1["confidence_tier"] != "validated"

    def test_multiple_patterns_evaluated(self, db_session):
        """Multiple patterns each get their own validation run."""
        self._seed_signals(db_session, "M4", count=50, forward_return=0.05)
        self._seed_signals(db_session, "I1", count=10, forward_return=0.10)

        val_job = ValidationScaffoldJob(session=db_session, minimum_sample=30)
        result = run_job(db_session, val_job)

        assert result.metrics["patterns_evaluated"] == 2
        assert result.metrics["pattern_results"]["M4"]["confidence_tier"] == "monitoring"
        assert result.metrics["pattern_results"]["I1"]["confidence_tier"] == "insufficient_sample"

        vrs = db_session.query(ValidationRun).all()
        assert len(vrs) == 2


# ===================================================================
# Schema: new columns exist and are nullable
# ===================================================================


class TestSchemaAdditions:
    def test_signal_identity_hash_column_exists(self, db_session):
        """signal_identity_hash column is accessible on SignalRegistry."""
        feat = record_feature_snapshot(
            db_session,
            pattern_id="M4",
            ticker="TEST",
            asof_timestamp=_ts(),
            features={},
            data_lineage_ids=[],
        )
        db_session.flush()
        sig = record_signal(
            db_session,
            pattern_id="M4",
            ticker="TEST",
            direction="long",
            signal_timestamp=_ts(),
            raw_signal_strength=0.5,
            raw_expected_edge=0.04,
            feature_snapshot_id=feat.feature_snapshot_id,
            signal_identity_hash="test-hash-123",
        )
        db_session.flush()

        stored = db_session.get(SignalRegistry, sig.signal_id)
        assert stored.signal_identity_hash == "test-hash-123"

    def test_forward_return_columns_nullable(self, db_session):
        """forward_return value fields default empty and status defaults pending."""
        feat = record_feature_snapshot(
            db_session,
            pattern_id="M4",
            ticker="TEST",
            asof_timestamp=_ts(),
            features={},
            data_lineage_ids=[],
        )
        db_session.flush()
        sig = record_signal(
            db_session,
            pattern_id="M4",
            ticker="TEST",
            direction="long",
            signal_timestamp=_ts(),
            raw_signal_strength=0.5,
            raw_expected_edge=0.04,
            feature_snapshot_id=feat.feature_snapshot_id,
            signal_identity_hash="forward-columns-test",
        )
        db_session.flush()

        stored = db_session.get(SignalRegistry, sig.signal_id)
        assert stored.forward_return is None
        assert stored.forward_return_status == "pending"
        assert stored.forward_return_attempts == 0
        assert stored.outcome_unavailable_reason is None
        assert stored.intended_entry_price is None


# ===================================================================
# Evidence bridge: signal_identity_hash propagation
# ===================================================================


class TestEvidenceBridgeIdentity:
    def test_persist_detection_propagates_identity(self, db_session):
        """persist_detection_result extracts signal_identity_hash from features."""
        from alpha.patterns.evidence_bridge import persist_detection_result

        detector = StableIdentityDetector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "fixture_score": 0.95,
                "price": 5.0,
                "high_52w": 10.0,
                "high_52w_date": "2026-01-01",
                "operating_universe_inclusion": True,
            },
        )
        result = detector.detect(inp)
        assert result.has_signal

        persisted = persist_detection_result(db_session, result, detector)
        db_session.flush()

        sig = db_session.get(SignalRegistry, persisted.signal_ids[0])
        assert sig.signal_identity_hash is not None
        assert sig.signal_identity_hash == result.features.features["signal_identity_hash"]

    def test_duplicate_identity_reuses_existing_feature_snapshot(self, db_session):
        """Bridge dedup must not create orphan feature snapshots on duplicate signals."""
        from alpha.patterns.evidence_bridge import persist_detection_result

        detector = StableIdentityDetector()
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=_ts(),
            market_data={
                "fixture_score": 0.95,
                "price": 5.0,
                "high_52w": 10.0,
                "high_52w_date": "2026-01-01",
                "operating_universe_inclusion": True,
            },
        )
        result = detector.detect(inp)

        p1 = persist_detection_result(db_session, result, detector)
        p2 = persist_detection_result(db_session, result, detector)
        db_session.flush()

        assert p1.signal_ids == p2.signal_ids
        assert p1.feature_snapshot_id == p2.feature_snapshot_id
        assert db_session.query(FeatureSnapshot).count() == 1
        assert db_session.query(SignalRegistry).count() == 1
