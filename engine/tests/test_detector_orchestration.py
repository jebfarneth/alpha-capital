"""
Detector orchestration tests per MeasurementSpine.md section 2.

  - No canonical scan → typed failed run, no signals.
  - Callable detector enumeration from code.
  - Deterministic signal_identity_hash across job runs / wall-clock times.
  - Double-run idempotency → no duplicate signal identities.
  - Null identity / null scan_id / null universe_snapshot_id refuses persistence.
  - Failed look-ahead guard refuses valid signal persistence.
  - Partial failure: one detector fails, another persists, run is partial_failed.
  - Detector diagnostics: evaluated/fired/error counts.
  - Universe anchoring: every signal links to correct canonical scan.
"""

from __future__ import annotations

import json
import inspect
from datetime import datetime, timezone

from sqlalchemy.exc import SQLAlchemyError

from alpha.data.contracts import stable_hash
from alpha.db.models import (
    CanonicalUniverseScan,
    FeatureSnapshot,
    SignalRegistry,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.evidence.writer import record_data_lineage
from alpha.jobs.detector_orchestration import (
    DetectorOrchestrationJob,
    check_lookahead_guard,
    compute_signal_identity_hash,
    enumerate_callable_detectors,
)
from alpha.jobs.runner import run_job
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


def _detector_identity(pattern_id: str, ticker: str, setup: str = "fixture") -> str:
    return stable_hash({
        "pattern_id": pattern_id,
        "ticker": ticker,
        "setup": setup,
    })


def _setup_canonical_universe(db_session, trading_date="2026-05-20", tickers=None):
    """Create a canonical universe with included snapshots."""
    if tickers is None:
        tickers = ["ACME", "BETA"]

    scan = UniverseScan(
        scan_id="test-scan",
        trading_date=trading_date,
        asof_timestamp=_ts(),
        raw_count=len(tickers),
        deduped_count=len(tickers),
        included_count=len(tickers),
        excluded_count=0,
        run_status="finished",
        source_lineage_hash="screener-hash",
    )
    db_session.add(scan)
    db_session.flush()

    db_session.add(CanonicalUniverseScan(
        trading_date=trading_date,
        scan_id="test-scan",
        selection_reason="test",
    ))
    db_session.flush()

    snapshots = []
    for ticker in tickers:
        snap = UniverseSnapshot(
            universe_snapshot_id=f"snap-{ticker}",
            scan_id="test-scan",
            ticker=ticker,
            asof_timestamp=_ts(),
            market_cap=75_000_000,
            price=5.0,
            primary_exchange="NASDAQ",
            security_type="common_stock",
            operating_universe_inclusion=True,
            source_lineage_hash="lineage-hash",
        )
        db_session.add(snap)
        snapshots.append(snap)

    # Also add an excluded snapshot to verify it's not picked up
    db_session.add(UniverseSnapshot(
        universe_snapshot_id="snap-excluded",
        scan_id="test-scan",
        ticker="EXCL",
        asof_timestamp=_ts(),
        operating_universe_inclusion=False,
        exclusion_reason="etf",
        source_lineage_hash="lineage-hash",
    ))
    db_session.flush()
    return scan, snapshots


class AlwaysFiresDetector(BasePatternDetector):
    """Test detector that fires on every input."""

    pattern_id = "TEST_FIRES"
    version = "1.0"
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.RIGHT_TAIL_CONVEX
    route_class = RouteClass.A

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        identity = _detector_identity(self.pattern_id, inp.ticker)
        features = PatternFeatures(
            features={
                "score": 0.95,
                "price": inp.market_data.get("price", 0),
                "signal_identity_hash": identity,
                "signal_identity_components": {
                    "pattern_id": self.pattern_id,
                    "ticker": inp.ticker,
                    "setup": "fixture",
                },
            },
            feature_manifest_version="test-v1",
            fidelity_tier=FidelityTier.FULL,
            point_in_time_passed=True,
            lookahead_guard_passed=True,
        )
        return PatternDetectionResult(
            pattern_id=self.pattern_id,
            ticker=inp.ticker,
            asof_timestamp=inp.asof_timestamp,
            features=features,
            signals=[PatternSignal(
                direction=SignalDirection.LONG,
                raw_signal_strength=0.95,
                raw_expected_edge=0.05,
                signal_horizon="10d",
            )],
            input_hashes={"market_data": stable_hash(inp.market_data)},
        )


class NeverFiresDetector(BasePatternDetector):
    """Test detector that never fires."""

    pattern_id = "TEST_SILENT"
    version = "1.0"
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.CONTINUATION
    route_class = RouteClass.A

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        features = PatternFeatures(
            features={"score": 0.10},
            feature_manifest_version="test-v1",
            fidelity_tier=FidelityTier.FULL,
            point_in_time_passed=True,
            lookahead_guard_passed=True,
        )
        return PatternDetectionResult(
            pattern_id=self.pattern_id,
            ticker=inp.ticker,
            asof_timestamp=inp.asof_timestamp,
            features=features,
            signals=[],
        )


class CrashingDetector(BasePatternDetector):
    """Test detector that always crashes."""

    pattern_id = "TEST_CRASH"
    version = "1.0"
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.CONTINUATION
    route_class = RouteClass.A

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        raise RuntimeError("detector crashed")


class AssertingTimestampDetector(AlwaysFiresDetector):
    """Detector that proves default DB-built inputs are UTC-aware."""

    pattern_id = "TEST_AWARE_TS"

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        assert inp.asof_timestamp.tzinfo is not None
        assert inp.asof_timestamp.utcoffset() is not None
        return super().detect(inp)


class PartiallyCrashingDetector(AlwaysFiresDetector):
    """Detector that persists one ticker and crashes on another."""

    pattern_id = "TEST_PARTIAL"

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        if inp.ticker == "BETA":
            raise RuntimeError("detector crashed after prior ticker")
        return super().detect(inp)


class IdentitylessDetector(AlwaysFiresDetector):
    """Detector that fires without the required detector-native identity."""

    pattern_id = "TEST_NO_ID"

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        result = super().detect(inp)
        result.features.features.pop("signal_identity_hash", None)
        result.features.features.pop("signal_identity_components", None)
        return result


class FeatureGuardFailingDetector(AlwaysFiresDetector):
    """Detector that catches its own PIT/lookahead failure."""

    pattern_id = "TEST_GUARD_FAIL"

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        result = super().detect(inp)
        result.features.point_in_time_passed = False
        result.features.lookahead_guard_passed = False
        return result


# -----------------------------------------------------------------------
# Test: no canonical scan → failed run
# -----------------------------------------------------------------------

class TestNoCanonicalScan:
    def test_missing_canonical_scan_fails(self, db_session):
        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert not result.ok
        assert result.status == "failed"
        assert any("canonical" in e.get("message", "").lower() for e in result.errors)
        assert db_session.query(SignalRegistry).count() == 0

    def test_lookahead_contaminated_canonical_scan_refused(self, db_session):
        _setup_canonical_universe(db_session, trading_date="2026-05-24", tickers=["ACME"])
        scan = db_session.get(UniverseScan, "test-scan")
        scan.asof_timestamp = datetime(2026, 5, 25, 8, 3, tzinfo=timezone.utc)
        db_session.flush()

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-24",
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-24"})

        assert result.status == "failed"
        assert result.metrics["total_signals_persisted"] == 0
        assert db_session.query(SignalRegistry).count() == 0
        assert result.errors == [{
            "stage": "canonical_universe",
            "message": (
                "canonical scan asof market date 2026-05-25 does not match "
                "trading_date 2026-05-24; refusing lookahead-contaminated "
                "universe scan"
            ),
        }]

    def test_missing_trading_date_fails(self, db_session):
        job = DetectorOrchestrationJob(db_session, detectors=[AlwaysFiresDetector()])
        result = run_job(db_session, job)

        assert not result.ok
        assert result.status == "failed"


# -----------------------------------------------------------------------
# Test: detector enumeration
# -----------------------------------------------------------------------

class TestDetectorEnumeration:
    def test_enumerate_finds_real_detectors(self):
        detectors = enumerate_callable_detectors()
        ids = {d.pattern_id for d in detectors}
        assert ids == {"M1", "M2", "M3", "M4", "M5", "M6", "M7", "I1", "I8"}
        assert all(d.__class__.__dict__.get("version") == "1.0" for d in detectors)
        # Fixture detector is NOT included in enumeration
        assert "FIXTURE" not in ids


# -----------------------------------------------------------------------
# Test: deterministic signal_identity_hash
# -----------------------------------------------------------------------

class TestSignalIdentityHash:
    def test_same_content_same_hash(self):
        h1 = compute_signal_identity_hash(
            detector_id="M4",
            detector_version="1.0",
            ticker="ACME",
            trading_date="2026-05-20",
            direction="long",
        )
        h2 = compute_signal_identity_hash(
            detector_id="M4",
            detector_version="1.0",
            ticker="ACME",
            trading_date="2026-05-20",
            direction="long",
        )
        assert h1 == h2
        assert len(h1) == 64  # sha256

    def test_different_ticker_different_hash(self):
        h1 = compute_signal_identity_hash(
            detector_id="M4", detector_version="1.0",
            ticker="ACME", trading_date="2026-05-20", direction="long",
        )
        h2 = compute_signal_identity_hash(
            detector_id="M4", detector_version="1.0",
            ticker="BETA", trading_date="2026-05-20", direction="long",
        )
        assert h1 != h2

    def test_hash_excludes_job_run_id(self):
        """job_run_id must NOT participate in identity hash."""
        h1 = compute_signal_identity_hash(
            detector_id="M4", detector_version="1.0",
            ticker="ACME", trading_date="2026-05-20", direction="long",
        )
        # Same content — hash should be identical regardless of which job produced it
        h2 = compute_signal_identity_hash(
            detector_id="M4", detector_version="1.0",
            ticker="ACME", trading_date="2026-05-20", direction="long",
        )
        assert h1 == h2

    def test_hash_across_different_wall_clock(self, db_session):
        """Same firing across two job runs produces same identity hash."""
        _setup_canonical_universe(db_session)

        job1 = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
        )
        r1 = run_job(db_session, job1, params={"trading_date": "2026-05-20"})

        signals = db_session.query(SignalRegistry).all()
        hashes_run1 = {s.signal_identity_hash for s in signals}

        # Compute what the hash would be for a second run
        h = compute_signal_identity_hash(
            detector_id="TEST_FIRES", detector_version="1.0",
            ticker="ACME", trading_date="2026-05-20", direction="long",
            detector_signal_identity_hash=_detector_identity("TEST_FIRES", "ACME"),
            detector_signal_identity_components={
                "pattern_id": "TEST_FIRES",
                "ticker": "ACME",
                "setup": "fixture",
            },
            route_class=RouteClass.A,
            signal_horizon="10d",
            signal_event_sequence=1,
        )
        assert h in hashes_run1

    def test_hash_preserves_detector_native_identity(self):
        base = {
            "detector_id": "M4",
            "detector_version": "1.0",
            "ticker": "ACME",
            "trading_date": "2026-05-20",
            "direction": "long",
        }
        h1 = compute_signal_identity_hash(
            **base,
            detector_signal_identity_hash="setup-high-52w-jan",
        )
        h2 = compute_signal_identity_hash(
            **base,
            detector_signal_identity_hash="setup-high-52w-feb",
        )
        assert h1 != h2

    def test_hash_excludes_scan_id(self):
        assert "scan_id" not in inspect.signature(compute_signal_identity_hash).parameters
        base = {
            "detector_id": "M4",
            "detector_version": "1.0",
            "ticker": "ACME",
            "trading_date": "2026-05-20",
            "direction": "long",
            "detector_signal_identity_hash": "same-setup",
        }
        assert compute_signal_identity_hash(**base) == compute_signal_identity_hash(**base)

    def test_different_direction_different_hash(self):
        base = {
            "detector_id": "M4",
            "detector_version": "1.0",
            "ticker": "ACME",
            "trading_date": "2026-05-20",
            "detector_signal_identity_hash": "same-setup",
        }
        long_hash = compute_signal_identity_hash(**base, direction="long")
        short_hash = compute_signal_identity_hash(**base, direction="short")
        assert long_hash != short_hash


# -----------------------------------------------------------------------
# Test: double-run idempotency
# -----------------------------------------------------------------------

class TestIdempotency:
    def test_default_db_inputs_normalize_sqlite_naive_datetimes(self, db_session):
        _setup_canonical_universe(db_session)
        db_session.expire_all()

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AssertingTimestampDetector()],
            trading_date="2026-05-20",
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.status == "finished"
        assert db_session.query(SignalRegistry).count() == 2

    def test_double_run_no_duplicates(self, db_session):
        _setup_canonical_universe(db_session)

        job1 = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
        )
        r1 = run_job(db_session, job1, params={"trading_date": "2026-05-20"})
        count_after_first = db_session.query(SignalRegistry).count()

        job2 = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
        )
        r2 = run_job(db_session, job2, params={"trading_date": "2026-05-20"})
        count_after_second = db_session.query(SignalRegistry).count()

        assert count_after_first == count_after_second
        assert r2.metrics["total_signals_persisted"] == 0
        diag = r2.metrics["detector_diagnostics"][0]
        assert diag["duplicate_suppressed_count"] == 2  # ACME + BETA

    def test_canonical_rebuild_same_setup_dedups(self, db_session):
        _setup_canonical_universe(db_session)

        first = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
        )
        run_job(db_session, first, params={"trading_date": "2026-05-20"})
        first_rows = db_session.query(SignalRegistry).all()
        first_hashes = {row.signal_identity_hash for row in first_rows}

        db_session.add(UniverseScan(
            scan_id="test-scan-2",
            trading_date="2026-05-20",
            asof_timestamp=_ts(),
            raw_count=2,
            deduped_count=2,
            included_count=2,
            excluded_count=0,
            run_status="finished",
            source_lineage_hash="screener-hash-2",
        ))
        for ticker in ("ACME", "BETA"):
            db_session.add(UniverseSnapshot(
                universe_snapshot_id=f"snap2-{ticker}",
                scan_id="test-scan-2",
                ticker=ticker,
                asof_timestamp=_ts(),
                market_cap=75_000_000,
                price=5.0,
                primary_exchange="NASDAQ",
                security_type="common_stock",
                operating_universe_inclusion=True,
                source_lineage_hash="lineage-hash",
            ))
        db_session.get(CanonicalUniverseScan, "2026-05-20").scan_id = "test-scan-2"
        db_session.flush()

        second = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
        )
        result = run_job(db_session, second, params={"trading_date": "2026-05-20"})

        assert result.metrics["total_signals_persisted"] == 0
        assert db_session.query(SignalRegistry).count() == len(first_rows)
        assert {row.signal_identity_hash for row in db_session.query(SignalRegistry)} == first_hashes


# -----------------------------------------------------------------------
# Test: strict refusal
# -----------------------------------------------------------------------

class TestStrictRefusal:
    def test_failed_lookahead_guard_refuses_signal(self, db_session):
        _setup_canonical_universe(db_session)

        # Build inputs with future asof_timestamp (after trading_date)
        future_inputs = [
            PatternInput(
                ticker="ACME",
                asof_timestamp=datetime(2026, 5, 21, 14, 0, tzinfo=timezone.utc),
                market_data={"price": 5.0},
                lineage_hashes=["hash1"],
                universe_snapshot_id="snap-ACME",
            ),
        ]

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
            inputs=future_inputs,
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert db_session.query(SignalRegistry).count() == 0
        diag = result.metrics["detector_diagnostics"][0]
        assert result.status == "partial_failed"
        assert diag["detector_status"] == "failed"
        assert diag["lookahead_failure_count"] == 1

    def test_missing_lineage_hashes_fails_guard(self, db_session):
        _setup_canonical_universe(db_session)

        inputs = [
            PatternInput(
                ticker="ACME",
                asof_timestamp=_ts(),
                market_data={"price": 5.0},
                lineage_hashes=[],  # empty
                universe_snapshot_id="snap-ACME",
            ),
        ]

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
            inputs=inputs,
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert db_session.query(SignalRegistry).count() == 0
        diag = result.metrics["detector_diagnostics"][0]
        assert result.status == "partial_failed"
        assert diag["detector_status"] == "failed"
        assert diag["lookahead_failure_count"] == 1
        assert "lookahead_guard_failed" in diag["errors"][0]["error"]
        assert "missing_lineage_hashes" in diag["errors"][0]["error"]

    def test_blank_lineage_hash_fails_guard(self, db_session):
        _setup_canonical_universe(db_session)

        inputs = [
            PatternInput(
                ticker="ACME",
                asof_timestamp=_ts(),
                market_data={"price": 5.0},
                lineage_hashes=[""],
                universe_snapshot_id="snap-ACME",
            ),
        ]

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
            inputs=inputs,
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert db_session.query(SignalRegistry).count() == 0
        diag = result.metrics["detector_diagnostics"][0]
        assert result.status == "partial_failed"
        assert diag["detector_status"] == "failed"
        assert diag["lookahead_failure_count"] == 1

    def test_future_market_date_fails_even_with_later_scan_asof(self, db_session):
        passed, reason = check_lookahead_guard(
            PatternInput(
                ticker="ACME",
                asof_timestamp=datetime(2026, 5, 25, 4, 35, tzinfo=timezone.utc),
                market_data={"price": 5.0},
                lineage_hashes=["hash1"],
                universe_snapshot_id="snap-ACME",
            ),
            trading_date="2026-05-24",
            max_asof_timestamp=datetime(2026, 5, 25, 8, 3, tzinfo=timezone.utc),
        )

        assert passed is False
        assert "after trading_date 2026-05-24" in reason

    def test_utc_next_day_same_market_date_passes_without_scan_cutoff(self):
        passed, reason = check_lookahead_guard(
            PatternInput(
                ticker="ACME",
                asof_timestamp=datetime(2026, 5, 21, 0, 30, tzinfo=timezone.utc),
                market_data={"price": 5.0},
                lineage_hashes=["hash1"],
                universe_snapshot_id="snap-ACME",
            ),
            trading_date="2026-05-20",
        )

        assert passed is True
        assert reason is None

    def test_evidence_close_ceiling_rejects_after_close_timestamp(self):
        passed, reason = check_lookahead_guard(
            PatternInput(
                ticker="ACME",
                asof_timestamp=datetime(2026, 11, 27, 18, 1, tzinfo=timezone.utc),
                market_data={"evidence_session_date": "2026-11-27"},
                lineage_hashes=["hash1"],
                universe_snapshot_id="snap-ACME",
            ),
            trading_date="2026-11-27",
            max_asof_timestamp=datetime(2026, 11, 27, 18, 0, tzinfo=timezone.utc),
            max_asof_label="evidence session close",
        )

        assert passed is False
        assert "after evidence session close" in reason

    def test_future_evidence_session_date_fails_closed(self, db_session):
        _setup_canonical_universe(db_session, tickers=["ACME"])
        inputs = [
            PatternInput(
                ticker="ACME",
                asof_timestamp=datetime(2026, 5, 20, 20, 0, tzinfo=timezone.utc),
                market_data={
                    "price": 5.0,
                    "evidence_session_date": "2026-05-21",
                },
                lineage_hashes=["hash1"],
                universe_snapshot_id="snap-ACME",
            ),
        ]

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
            assembled_inputs={"TEST_FIRES": inputs},
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert db_session.query(SignalRegistry).count() == 0
        diag = result.metrics["detector_diagnostics"][0]
        assert result.status == "partial_failed"
        assert diag["lookahead_failure_count"] == 1
        assert diag["fired_count"] == 0
        assert "future_evidence_session_date" in diag["errors"][0]["error"]

    def test_same_day_evidence_session_date_still_passes(self, db_session):
        _setup_canonical_universe(db_session, tickers=["ACME"])
        inputs = [
            PatternInput(
                ticker="ACME",
                asof_timestamp=datetime(2026, 5, 20, 20, 0, tzinfo=timezone.utc),
                market_data={
                    "price": 5.0,
                    "evidence_session_date": "2026-05-20",
                },
                lineage_hashes=["hash1"],
                universe_snapshot_id="snap-ACME",
            ),
        ]

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
            assembled_inputs={"TEST_FIRES": inputs},
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        diag = result.metrics["detector_diagnostics"][0]
        assert result.status == "finished"
        assert diag["lookahead_failure_count"] == 0
        assert diag["fired_count"] == 1
        assert db_session.query(SignalRegistry).count() == 1

    def test_missing_evidence_session_date_uses_scan_asof_ceiling(self, db_session):
        _setup_canonical_universe(db_session, tickers=["ACME"])
        inputs = [
            PatternInput(
                ticker="ACME",
                asof_timestamp=_ts(),
                market_data={"price": 5.0},
                lineage_hashes=["hash1"],
                universe_snapshot_id="snap-ACME",
            ),
        ]

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
            assembled_inputs={"TEST_FIRES": inputs},
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        diag = result.metrics["detector_diagnostics"][0]
        assert result.status == "finished"
        assert diag["lookahead_failure_count"] == 0
        assert diag["fired_count"] == 1
        assert db_session.query(SignalRegistry).count() == 1

    def test_detector_feature_guard_false_refuses_signal(self, db_session):
        _setup_canonical_universe(db_session)

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[FeatureGuardFailingDetector()],
            trading_date="2026-05-20",
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.status == "partial_failed"
        assert db_session.query(SignalRegistry).count() == 0
        diag = result.metrics["detector_diagnostics"][0]
        assert diag["lookahead_failure_count"] == 2
        assert diag["identity_refused_count"] == 2
        assert diag["fired_count"] == 0

    def test_detector_missing_identity_refuses_signal(self, db_session):
        _setup_canonical_universe(db_session)

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[IdentitylessDetector()],
            trading_date="2026-05-20",
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.status == "partial_failed"
        assert db_session.query(SignalRegistry).count() == 0
        diag = result.metrics["detector_diagnostics"][0]
        assert diag["identity_refused_count"] == 2


# -----------------------------------------------------------------------
# Test: partial failure
# -----------------------------------------------------------------------

class TestPartialFailure:
    def test_one_detector_crashes_other_persists(self, db_session):
        _setup_canonical_universe(db_session)

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector(), CrashingDetector()],
            trading_date="2026-05-20",
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.status == "partial_failed"
        assert result.metrics["total_signals_persisted"] == 2  # ACME + BETA from fires
        assert result.metrics["any_detector_failed"] is True

        # Verify diagnostics
        diags = {d["detector_id"]: d for d in result.metrics["detector_diagnostics"]}
        assert diags["TEST_FIRES"]["detector_status"] == "finished"
        assert diags["TEST_FIRES"]["fired_count"] == 2
        assert diags["TEST_CRASH"]["detector_status"] == "failed"
        assert diags["TEST_CRASH"]["error_count"] == 2

        # Signals from the working detector are persisted
        signals = db_session.query(SignalRegistry).all()
        assert len(signals) == 2
        assert all(s.pattern_id == "TEST_FIRES" for s in signals)

    def test_rerun_after_partial_failure_is_idempotent(self, db_session):
        _setup_canonical_universe(db_session)

        # First run: crash + fires
        job1 = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector(), CrashingDetector()],
            trading_date="2026-05-20",
        )
        r1 = run_job(db_session, job1, params={"trading_date": "2026-05-20"})
        assert r1.status == "partial_failed"
        count1 = db_session.query(SignalRegistry).count()

        # Second run: same detectors — fires should dedup, crash should re-error
        job2 = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector(), CrashingDetector()],
            trading_date="2026-05-20",
        )
        r2 = run_job(db_session, job2, params={"trading_date": "2026-05-20"})
        count2 = db_session.query(SignalRegistry).count()

        assert count1 == count2  # No duplicate signals

    def test_single_detector_partial_failure_marks_run_partial(self, db_session):
        _setup_canonical_universe(db_session)

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[PartiallyCrashingDetector()],
            trading_date="2026-05-20",
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.status == "partial_failed"
        assert result.metrics["any_detector_failed"] is True
        diag = result.metrics["detector_diagnostics"][0]
        assert diag["detector_status"] == "partial_failed"
        assert diag["fired_count"] == 1
        assert diag["error_count"] == 1

    def test_lineage_lookup_failure_isolated_to_one_input(
        self,
        db_session,
        monkeypatch,
    ):
        _setup_canonical_universe(db_session)
        savepoint_state_by_ticker = {}

        def flaky_lineage_lookup(self, inp):
            savepoint_state_by_ticker[inp.ticker] = (
                self._session.in_nested_transaction()
            )
            if inp.ticker == "ACME":
                raise SQLAlchemyError("lineage lookup failed")
            return []

        monkeypatch.setattr(
            DetectorOrchestrationJob,
            "_resolved_input_lineage_ids",
            flaky_lineage_lookup,
        )

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.status == "partial_failed"
        assert result.metrics["total_signals_persisted"] == 1
        assert savepoint_state_by_ticker == {"ACME": True, "BETA": True}

        signals = db_session.query(SignalRegistry).all()
        assert [signal.ticker for signal in signals] == ["BETA"]

        diag = result.metrics["detector_diagnostics"][0]
        assert diag["fired_count"] == 1
        assert diag["error_count"] == 1
        assert "lineage lookup failed" in diag["errors"][0]["error"]


# -----------------------------------------------------------------------
# Test: detector diagnostics
# -----------------------------------------------------------------------

class TestDetectorDiagnostics:
    def test_no_setups_vs_no_run(self, db_session):
        """NeverFiresDetector should show evaluated > 0, fired == 0."""
        _setup_canonical_universe(db_session)

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[NeverFiresDetector()],
            trading_date="2026-05-20",
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.ok
        diag = result.metrics["detector_diagnostics"][0]
        assert diag["evaluated_count"] == 2
        assert diag["fired_count"] == 0
        assert diag["error_count"] == 0
        assert diag["detector_status"] == "finished"

    def test_diagnostics_count_errors(self, db_session):
        _setup_canonical_universe(db_session)

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[CrashingDetector()],
            trading_date="2026-05-20",
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        diag = result.metrics["detector_diagnostics"][0]
        assert diag["evaluated_count"] == 2
        assert diag["error_count"] == 2
        assert diag["fired_count"] == 0
        assert diag["detector_status"] == "failed"

    def test_diagnostics_include_lineage_and_errors(self, db_session):
        _setup_canonical_universe(db_session)

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector(), CrashingDetector()],
            trading_date="2026-05-20",
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        diags = {d["detector_id"]: d for d in result.metrics["detector_diagnostics"]}
        assert diags["TEST_FIRES"]["input_lineage_hashes"] == ["lineage-hash"]
        assert diags["TEST_CRASH"]["errors"]


# -----------------------------------------------------------------------
# Test: universe anchoring
# -----------------------------------------------------------------------

class TestUniverseAnchoring:
    def test_signals_link_to_canonical_scan(self, db_session):
        _setup_canonical_universe(db_session)

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        signals = db_session.query(SignalRegistry).all()
        for sig in signals:
            assert sig.universe_snapshot_id is not None
            assert sig.trading_date == "2026-05-20"
            assert sig.scan_id == "test-scan"
            assert sig.signal_identity_hash is not None
            assert sig.point_in_time_passed is True
            assert sig.lookahead_guard_passed is True

    def test_injected_demoted_snapshot_is_refused(self, db_session):
        _setup_canonical_universe(db_session)
        db_session.add(UniverseScan(
            scan_id="demoted-scan",
            trading_date="2026-05-20",
            asof_timestamp=_ts(),
            raw_count=1,
            deduped_count=1,
            included_count=1,
            excluded_count=0,
            run_status="finished",
            source_lineage_hash="demoted-hash",
        ))
        db_session.add(UniverseSnapshot(
            universe_snapshot_id="demoted-snap-ACME",
            scan_id="demoted-scan",
            ticker="ACME",
            asof_timestamp=_ts(),
            market_cap=75_000_000,
            price=5.0,
            primary_exchange="NASDAQ",
            security_type="common_stock",
            operating_universe_inclusion=True,
            source_lineage_hash="lineage-hash",
        ))
        db_session.flush()
        inputs = [
            PatternInput(
                ticker="ACME",
                asof_timestamp=_ts(),
                market_data={"price": 5.0},
                lineage_hashes=["lineage-hash"],
                universe_snapshot_id="demoted-snap-ACME",
            ),
        ]

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
            inputs=inputs,
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        assert result.status == "partial_failed"
        assert db_session.query(SignalRegistry).count() == 0
        diag = result.metrics["detector_diagnostics"][0]
        assert diag["identity_refused_count"] == 1
        assert diag["errors"][0]["error"] == "universe_snapshot_not_in_canonical_scan"

    def test_default_inputs_link_data_lineage_ids(self, db_session):
        _setup_canonical_universe(db_session)
        lineage = record_data_lineage(
            db_session,
            provider="FMP",
            endpoint="/stable/company-screener",
            asof_timestamp=_ts(),
            raw_payload_hash="lineage-hash",
        )
        db_session.flush()

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
        )
        run_job(db_session, job, params={"trading_date": "2026-05-20"})

        for signal in db_session.query(SignalRegistry).all():
            assert lineage.data_lineage_id in json.loads(signal.data_lineage_ids)

    def test_excluded_tickers_not_evaluated(self, db_session):
        _setup_canonical_universe(db_session)

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
        )
        result = run_job(db_session, job, params={"trading_date": "2026-05-20"})

        # Only ACME and BETA should have signals, not EXCL
        tickers = {s.ticker for s in db_session.query(SignalRegistry).all()}
        assert tickers == {"ACME", "BETA"}
        assert "EXCL" not in tickers

    def test_universe_metadata_available_via_join(self, db_session):
        _setup_canonical_universe(db_session)

        job = DetectorOrchestrationJob(
            db_session,
            detectors=[AlwaysFiresDetector()],
            trading_date="2026-05-20",
        )
        run_job(db_session, job, params={"trading_date": "2026-05-20"})

        # Verify we can join signals → universe_snapshots for metadata
        signals = db_session.query(SignalRegistry).all()
        for sig in signals:
            snap = db_session.get(UniverseSnapshot, sig.universe_snapshot_id)
            assert snap is not None
            assert snap.market_cap == 75_000_000
            assert snap.price == 5.0
            assert snap.security_type == "common_stock"
            assert snap.primary_exchange == "NASDAQ"


# -----------------------------------------------------------------------
# Test: lookahead guard
# -----------------------------------------------------------------------

class TestLookaheadGuard:
    def test_valid_same_day_passes(self):
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=datetime(2026, 5, 20, 14, 0, tzinfo=timezone.utc),
            lineage_hashes=["hash1"],
        )
        passed, reason = check_lookahead_guard(inp, "2026-05-20")
        assert passed
        assert reason is None

    def test_future_timestamp_fails(self):
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=datetime(2026, 5, 21, 14, 0, tzinfo=timezone.utc),
            lineage_hashes=["hash1"],
        )
        passed, reason = check_lookahead_guard(inp, "2026-05-20")
        assert not passed
        assert "after" in reason

    def test_missing_asof_fails(self):
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=None,
            lineage_hashes=["hash1"],
        )
        passed, reason = check_lookahead_guard(inp, "2026-05-20")
        assert not passed
        assert "missing_asof" in reason

    def test_missing_lineage_fails(self):
        inp = PatternInput(
            ticker="ACME",
            asof_timestamp=datetime(2026, 5, 20, 14, 0, tzinfo=timezone.utc),
            lineage_hashes=[],
        )
        passed, reason = check_lookahead_guard(inp, "2026-05-20")
        assert not passed
        assert "lineage" in reason
