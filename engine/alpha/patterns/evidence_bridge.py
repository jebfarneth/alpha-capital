"""
Evidence bridge: persists detection results into evidence tables.

Takes a PatternDetectionResult and writes:
  - feature_snapshots (always, when features present)
  - signal_registry (only when signals present)

Links signals to feature_snapshot_id, job_run_id, universe_snapshot_id.
Does NOT record data_lineage — that's the caller's job via adapter LineageMeta.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from alpha.data.contracts import stable_hash
from alpha.db.models import FeatureSnapshot, SignalRegistry
from alpha.evidence.writer import record_feature_snapshot, record_signal
from alpha.patterns.contracts import (
    BasePatternDetector,
    PatternDetectionResult,
    PatternSignal,
)


@dataclass
class PersistedDetection:
    """IDs of evidence rows written."""

    feature_snapshot_id: Optional[str] = None
    signal_ids: List[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.signal_ids is None:
            self.signal_ids = []


def persist_detection_result(
    session: Session,
    result: PatternDetectionResult,
    detector: BasePatternDetector,
    *,
    job_run_id: Optional[str] = None,
    universe_snapshot_id: Optional[str] = None,
    data_lineage_ids: Optional[List[str]] = None,
    code_commit_sha: Optional[str] = None,
    trading_date: Optional[str] = None,
    scan_id: Optional[str] = None,
    detector_version: Optional[str] = None,
    point_in_time_passed: Optional[bool] = None,
    lookahead_guard_passed: Optional[bool] = None,
) -> PersistedDetection:
    """
    Write a detection result into the evidence tables.

    Returns IDs of written rows so the caller can track them.
    """
    persisted = PersistedDetection()

    if result.features is None:
        if result.signals:
            raise ValueError("signals require a feature snapshot and signal identity")
        return persisted

    signal_identity_hashes: List[str] = []
    if result.signals:
        signal_identity_hashes = _signal_identity_hashes(result, detector)
    if result.signals and not signal_identity_hashes:
        raise ValueError("signals require signal_identity_hash in features")

    existing_by_hash = {}
    if signal_identity_hashes:
        existing_signals = (
            session.query(SignalRegistry)
            .filter(
                SignalRegistry.pattern_id == result.pattern_id,
                SignalRegistry.ticker == result.ticker,
                SignalRegistry.signal_identity_hash.in_(signal_identity_hashes),
            )
            .all()
        )
        existing_by_hash = {
            row.signal_identity_hash: row
            for row in existing_signals
            if row.signal_identity_hash is not None
        }
        if len(existing_by_hash) == len(signal_identity_hashes):
            first_existing = existing_signals[0]
            persisted.feature_snapshot_id = first_existing.feature_snapshot_id
            persisted.signal_ids.extend(row.signal_id for row in existing_signals)
            return persisted

    feat = record_feature_snapshot(
        session,
        pattern_id=result.pattern_id,
        ticker=result.ticker,
        asof_timestamp=result.asof_timestamp,
        features=result.features.features,
        data_lineage_ids=data_lineage_ids or [],
        job_run_id=job_run_id,
        feature_manifest_version=result.features.feature_manifest_version,
        code_commit_sha=code_commit_sha,
        fidelity_tier=result.features.fidelity_tier,
        point_in_time_passed=result.features.point_in_time_passed,
        lookahead_guard_passed=result.features.lookahead_guard_passed,
        input_hashes=result.input_hashes or None,
    )
    persisted.feature_snapshot_id = feat.feature_snapshot_id

    for sequence, sig in enumerate(result.signals, start=1):
        signal_identity_hash = signal_identity_hashes[sequence - 1]
        existing_signal = existing_by_hash.get(signal_identity_hash)
        if existing_signal is not None:
            persisted.signal_ids.append(existing_signal.signal_id)
            continue
        sr = record_signal(
            session,
            pattern_id=result.pattern_id,
            ticker=result.ticker,
            direction=sig.direction,
            signal_timestamp=result.asof_timestamp,
            raw_signal_strength=sig.raw_signal_strength,
            raw_expected_edge=sig.raw_expected_edge,
            feature_snapshot_id=feat.feature_snapshot_id,
            job_run_id=job_run_id,
            signal_status=sig.signal_status,
            signal_horizon=sig.signal_horizon,
            thesis_category=detector.thesis_category,
            route_class=sig.route_class or detector.route_class,
            fidelity_tier=result.features.fidelity_tier,
            data_confidence=sig.data_confidence,
            data_lineage_ids=data_lineage_ids,
            universe_snapshot_id=universe_snapshot_id,
            trading_date=trading_date,
            scan_id=scan_id,
            detector_version=detector_version,
            point_in_time_passed=point_in_time_passed,
            lookahead_guard_passed=lookahead_guard_passed,
            signal_event_sequence=sequence,
            signal_identity_hash=signal_identity_hash,
        )
        persisted.signal_ids.append(sr.signal_id)

    return persisted


def _signal_identity_hashes(
    result: PatternDetectionResult,
    detector: BasePatternDetector,
) -> List[str]:
    raw_hashes = result.features.features.get("signal_identity_hashes")
    if raw_hashes is not None:
        hashes = [str(value).strip() for value in raw_hashes if str(value).strip()]
        if len(hashes) != len(result.signals):
            raise ValueError("signal_identity_hashes length must match signals")
        return hashes

    base_hash = str(result.features.features.get("signal_identity_hash") or "").strip()
    if not base_hash:
        return []
    if len(result.signals) == 1:
        return [base_hash]

    hashes = []
    for sequence, sig in enumerate(result.signals, start=1):
        hashes.append(stable_hash({
            "base_signal_identity_hash": base_hash,
            "signal_event_sequence": sequence,
            "route_class": sig.route_class or detector.route_class,
            "signal_horizon": sig.signal_horizon,
        }))
    return hashes
