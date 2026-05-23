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
) -> PersistedDetection:
    """
    Write a detection result into the evidence tables.

    Returns IDs of written rows so the caller can track them.
    """
    persisted = PersistedDetection()

    if result.features is None:
        return persisted

    signal_identity_hash = result.features.features.get("signal_identity_hash") if result.features else None

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
        if signal_identity_hash:
            existing_signal_id = (
                session.query(SignalRegistry.signal_id)
                .filter(
                    SignalRegistry.pattern_id == result.pattern_id,
                    SignalRegistry.ticker == result.ticker,
                    SignalRegistry.signal_identity_hash == signal_identity_hash,
                )
                .scalar()
            )
            if existing_signal_id:
                persisted.signal_ids.append(existing_signal_id)
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
            signal_event_sequence=sequence,
            signal_identity_hash=signal_identity_hash,
        )
        persisted.signal_ids.append(sr.signal_id)

    return persisted
