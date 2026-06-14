from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from alpha.db.models import (
    FeatureSnapshot,
    ForwardReturnObservation,
    MLModelRegistry,
    SignalMLScore,
    SignalRegistry,
)
from alpha.jobs.runner import run_job
from alpha.jobs.train_model import Stage1TrainModelJob
from alpha.ml.cv import CVExample, purged_embargoed_walk_forward_splits
from alpha.ml.inference import score_signal_shadow
from alpha.ml.manifest_loader import load_manifest, manifest_payload_hash
from alpha.ml.model_features import (
    FeatureSelectionError,
    audit_feature_schema_no_leakage,
    select_features,
)


def _feature_schema() -> dict:
    return {
        "schema_version": "test_stage1_features_v1",
        "pattern_id": "M4",
        "fields": [
            {
                "name": "mom20",
                "source": "feature_snapshot_json",
                "path": "signal_context.mom20",
            },
            {
                "name": "volume_ratio",
                "source": "feature_snapshot_json",
                "path": "signal_context.volume_ratio",
            },
            {
                "name": "gap",
                "source": "feature_snapshot_json",
                "path": "signal_context.gap",
                "status_path": "statuses.gap",
            },
            {
                "name": "raw_signal_strength",
                "source": "signal_registry",
                "column": "raw_signal_strength",
            },
        ],
    }


def _manifest_payload(signal_ids: list[str] | None = None) -> dict:
    payload = {
        "manifest_version": "test_stage1_manifest_v1",
        "manifest_sha256": "",
        "patterns": {
            "M4": {
                "signal_horizon": "2d",
                "min_graded_cohorts": 8,
                "embargo_sessions": 2,
                "selection": {
                    "source": "forward_return_observations",
                    "statuses": ["computed"],
                    "horizon_sessions": 2,
                    "signal_ids": signal_ids or [],
                },
                "label": {"field": "forward_return"},
                "feature_schema": _feature_schema(),
                "diagnostics": {"pooled_metrics_diagnostic_only": True},
            }
        },
    }
    payload["manifest_sha256"] = manifest_payload_hash(payload)
    return payload


def _write_manifest(tmp_path: Path, signal_ids: list[str]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest_payload(signal_ids), sort_keys=True))
    return path


def _seed_signal(
    db_session,
    *,
    idx: int,
    ticker: str | None = None,
    gap: float | None = -0.01,
) -> str:
    signal_id = f"sig-{idx}"
    ticker = ticker or f"T{idx % 3}"
    ts = datetime(2025, 1, 2, tzinfo=timezone.utc) + timedelta(days=idx)
    feature_snapshot_id = f"fs-{idx}"
    feature_json = {
        "signal_context": {
            "mom20": 0.01 * idx,
            "volume_ratio": 2.0 + idx,
            "gap": gap,
        },
        "statuses": {"gap": "not_available" if gap is None else "computed"},
    }
    db_session.add(
        FeatureSnapshot(
            feature_snapshot_id=feature_snapshot_id,
            pattern_id="M4",
            ticker=ticker,
            asof_timestamp=ts,
            feature_json=json.dumps(feature_json, sort_keys=True),
            feature_hash=f"feature-hash-{idx}",
            data_lineage_ids="[]",
        )
    )
    signal = SignalRegistry(
        signal_id=signal_id,
        pattern_id="M4",
        ticker=ticker,
        direction="long",
        signal_timestamp=ts,
        raw_signal_strength=1.0 + idx,
        raw_expected_edge=0.0,
        signal_horizon="2d",
        feature_snapshot_id=feature_snapshot_id,
        signal_status="active",
        trading_date=ts.date().isoformat(),
        signal_identity_hash=f"signal-hash-{idx}",
    )
    db_session.add(signal)
    db_session.add(
        ForwardReturnObservation(
            forward_return_observation_id=f"fro-{idx}",
            signal_id=signal_id,
            pattern_id="M4",
            ticker=ticker,
            direction="long",
            signal_timestamp=ts,
            signal_horizon="2d",
            forward_return=(idx - 4) / 100.0,
            status="computed",
            input_hash=f"input-{idx}",
            outcome_hash=f"outcome-{idx}",
        )
    )
    return signal_id


def _seed_training_corpus(db_session, *, count: int = 14) -> list[str]:
    signal_ids = []
    for idx in range(count):
        signal_ids.append(
            _seed_signal(
                db_session,
                idx=idx,
                ticker=f"T{idx % 4}",
                gap=-0.02 + (idx % 5) * 0.01,
            )
        )
    db_session.commit()
    return signal_ids


def test_manifest_loader_fails_closed_on_hash_drift(tmp_path):
    payload = _manifest_payload(["sig-1"])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    assert load_manifest(path).manifest_version == "test_stage1_manifest_v1"

    payload["patterns"]["M4"]["min_graded_cohorts"] = 9
    path.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(Exception, match="sha256 mismatch"):
        load_manifest(path)


def test_feature_reader_uses_stored_fields_and_preserves_typed_missing(db_session):
    signal_id = _seed_signal(db_session, idx=1, gap=None)
    db_session.commit()

    vector = select_features(db_session, signal_id, _feature_schema())

    assert vector.feature_names == [
        "mom20",
        "volume_ratio",
        "gap",
        "raw_signal_strength",
    ]
    assert vector.values[0] == pytest.approx(0.01)
    assert math.isnan(vector.values[2])
    assert vector.missing_statuses["gap"] == "not_available"


def test_leakage_audit_raises_on_forward_path_feature():
    schema = {
        "fields": [
            {
                "name": "forward_return",
                "source": "feature_snapshot_json",
                "path": "forward_return",
            }
        ]
    }
    with pytest.raises(FeatureSelectionError):
        audit_feature_schema_no_leakage(schema)


def test_purged_embargoed_cv_excludes_nearby_training_dates():
    start = datetime(2025, 1, 1).date()
    examples = [
        CVExample(signal_id=f"s{i}", ticker=f"T{i % 2}", signal_date=start + timedelta(days=i))
        for i in range(12)
    ]
    folds = purged_embargoed_walk_forward_splits(
        examples,
        n_splits=2,
        horizon_sessions=2,
        embargo_sessions=2,
    )

    date_positions = {row.signal_date: i for i, row in enumerate(examples)}
    for fold in folds:
        test_start = date_positions[fold.test_start_date]
        assert all(
            date_positions[examples[idx].signal_date] < test_start - 2
            for idx in fold.train_indices
        )


def test_train_model_end_to_end_writes_artifact_registry_and_shadow_score(
    db_session, tmp_path
):
    signal_ids = _seed_training_corpus(db_session)
    manifest = load_manifest(_write_manifest(tmp_path, signal_ids))
    artifact_dir = tmp_path / "artifacts"
    job = Stage1TrainModelJob(
        session=db_session,
        manifest=manifest,
        pattern_id="M4",
        artifact_dir=artifact_dir,
        n_splits=2,
        max_iter=3,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    model = db_session.query(MLModelRegistry).one()
    assert model.status == "shadow"
    assert Path(model.artifact_uri).exists()
    metrics = json.loads(model.cv_metrics_json)
    assert metrics["per_pattern"]["M4"]["cv_type"] == "purged_embargoed_walk_forward"
    assert metrics["per_pattern"]["M4"]["random_kfold_forbidden"] is True
    assert metrics["pooled"]["diagnostic_only"] is True

    train_vector = select_features(db_session, signal_ids[0], _feature_schema())
    score = score_signal_shadow(db_session, signal_id=signal_ids[0], model_id=model.model_id)
    db_session.flush()
    infer_vector = select_features(
        db_session,
        signal_ids[0],
        json.loads(model.feature_schema_json),
    )
    assert train_vector.feature_vector_hash == infer_vector.feature_vector_hash
    assert score.score_source == "model_shadow"
    assert score.score_metadata_json is not None
    assert json.loads(score.score_metadata_json)["acts_on_book"] is False


def test_missing_model_degrades_to_hand_discriminant_fallback(db_session):
    signal_id = _seed_signal(db_session, idx=2, gap=-0.01)
    db_session.commit()

    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id="missing-model",
    )
    db_session.flush()

    assert score.score_source == "fallback_hand_discriminant"
    assert score.fallback_reason == "model_missing"
    assert score.score == pytest.approx(3.0)
    assert db_session.query(SignalMLScore).count() == 1
