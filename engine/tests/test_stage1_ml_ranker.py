from __future__ import annotations

import json
import math
import pickle
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from alpha.db.models import (
    FeatureSnapshot,
    ForwardReturnObservation,
    MLModelRegistry,
    MarketPathFeature,
    SignalMLScore,
    SignalRegistry,
)
from alpha.jobs.contracts import JobContext
from alpha.jobs.runner import run_job
from alpha.jobs.train_model import MODEL_FAMILY, Stage1TrainModelJob
from alpha.ml.cv import CVExample, purged_embargoed_walk_forward_splits
from alpha.ml.inference import score_signal_shadow
from alpha.ml.manifest_loader import load_manifest, manifest_payload_hash
from alpha.ml.model_features import (
    FeatureSelectionError,
    audit_feature_schema_no_leakage,
    feature_schema_hash,
    select_features,
)


class ConstantModel:
    def predict(self, rows):
        return [0.123 for _ in rows]


class ExplodingModel:
    def predict(self, rows):
        raise RuntimeError("predict exploded")


class NonFiniteModel:
    def __init__(self, value):
        self.value = value

    def predict(self, rows):
        return [self.value for _ in rows]


def _feature_schema() -> dict:
    return {
        "schema_version": "test_stage1_features_v1",
        "pattern_id": "M4",
        "pattern_clock": "eod",
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
    signal_date: date | None = None,
) -> str:
    signal_id = f"sig-{idx}"
    ticker = ticker or f"T{idx % 3}"
    ts_date = signal_date or (datetime(2025, 1, 2).date() + timedelta(days=idx))
    ts = datetime.combine(ts_date, datetime.min.time(), timezone.utc)
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


def _add_model_registry(
    db_session,
    *,
    model_id: str,
    artifact_uri: str,
    schema_hash: str | None = None,
    manifest_sha256: str = "manifest-sha",
) -> MLModelRegistry:
    schema = _feature_schema()
    row = MLModelRegistry(
        model_id=model_id,
        pattern_id="M4",
        model_family=MODEL_FAMILY,
        training_window_start=date(2025, 1, 1),
        training_window_end=date(2025, 1, 31),
        manifest_version="test_manifest",
        manifest_sha256=manifest_sha256,
        feature_schema_hash=schema_hash or feature_schema_hash(schema),
        feature_code_git_sha="test",
        cv_metrics_json=json.dumps({"per_pattern": {"M4": {}}}),
        feature_schema_json=json.dumps(schema, sort_keys=True),
        artifact_uri=artifact_uri,
        status="shadow",
    )
    db_session.add(row)
    return row


def _write_artifact(
    path: Path,
    *,
    model_id: str,
    model=ConstantModel(),
    schema: dict | None = None,
    schema_hash: str | None = None,
    manifest_sha256: str = "manifest-sha",
    training_feature_ranges: list | None = None,
) -> None:
    schema = schema or _feature_schema()
    payload = {
        "model_id": model_id,
        "model_family": MODEL_FAMILY,
        "pattern_id": "M4",
        "manifest_version": "test_manifest",
        "manifest_sha256": manifest_sha256,
        "feature_schema": schema,
        "feature_schema_hash": schema_hash or feature_schema_hash(schema),
        "feature_names": [field["name"] for field in schema["fields"]],
        "training_feature_ranges": training_feature_ranges
        if training_feature_ranges is not None
        else [{"min": -999.0, "max": 999.0} for _ in schema["fields"]],
        "model": model,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


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


def test_leakage_audit_rejects_forward_market_path_feature_role():
    schema = {
        "fields": [
            {
                "name": "win",
                "source": "market_path_feature_column",
                "feature_role": "forward_path_day",
                "feature_version": "market_path_daily_v3",
                "column": "close_price",
            }
        ]
    }
    with pytest.raises(FeatureSelectionError):
        audit_feature_schema_no_leakage(schema)


def test_leakage_audit_is_pattern_clock_aware_for_signal_session_fields():
    field = {
        "name": "close_price",
        "source": "market_path_feature_column",
        "feature_role": "signal_session",
        "feature_version": "market_path_daily_v3",
        "column": "close_price",
    }
    intraday_schema = {
        "pattern_id": "I12",
        "pattern_clock": "intraday",
        "fields": [field],
    }
    eod_schema = {
        "pattern_id": "M4",
        "pattern_clock": "eod",
        "fields": [field],
    }

    with pytest.raises(FeatureSelectionError):
        audit_feature_schema_no_leakage(intraday_schema)
    audit_feature_schema_no_leakage(eod_schema)


@pytest.mark.parametrize(
    "field_name",
    ["win_rate", "target_hit", "profit_factor", "pnl_pct", "realized_pnl"],
)
def test_leakage_audit_rejects_snake_case_label_names(field_name):
    schema = {
        "pattern_id": "M4",
        "pattern_clock": "eod",
        "fields": [
            {
                "name": field_name,
                "source": "feature_snapshot_json",
                "path": f"signal_context.{field_name}",
            }
        ],
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


def test_training_rejects_many_rows_on_too_few_graded_cohorts(db_session, tmp_path):
    signal_ids = []
    dates = [date(2025, 1, 2), date(2025, 1, 3)]
    for idx in range(8):
        signal_ids.append(
            _seed_signal(
                db_session,
                idx=idx,
                ticker=f"T{idx}",
                gap=-0.01,
                signal_date=dates[idx % 2],
            )
        )
    db_session.commit()
    manifest = load_manifest(_write_manifest(tmp_path, signal_ids))
    job = Stage1TrainModelJob(
        session=db_session,
        manifest=manifest,
        pattern_id="M4",
        artifact_dir=tmp_path / "artifacts",
        n_splits=2,
        max_iter=3,
    )

    with pytest.raises(RuntimeError, match="2 graded cohorts across 8 rows"):
        job.run(
            JobContext(
                job_id="job",
                job_run_id="run",
                started_at=datetime.now(timezone.utc),
            )
        )


def test_missing_model_degrades_to_raw_strength_fallback_and_is_idempotent(db_session):
    signal_id = _seed_signal(db_session, idx=2, gap=-0.01)
    db_session.commit()

    first = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id="missing-model",
    )
    second = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id="missing-model",
    )
    db_session.flush()

    assert first.score_id == second.score_id
    assert first.score_source == "fallback_raw_strength"
    assert first.fallback_reason == "model_missing"
    assert first.score == pytest.approx(3.0)
    assert db_session.query(SignalMLScore).count() == 1
    assert "ux_signal_ml_scores_fallback_null_model" in {
        idx.name for idx in SignalMLScore.__table__.indexes
    }


def test_artifact_load_error_persists_fallback_without_raising(db_session, tmp_path):
    signal_id = _seed_signal(db_session, idx=3, gap=-0.01)
    missing_path = tmp_path / "missing.pkl"
    _add_model_registry(
        db_session,
        model_id="model-missing-artifact",
        artifact_uri=str(missing_path),
    )
    db_session.commit()

    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id="model-missing-artifact",
    )
    db_session.flush()

    assert score.score_source == "fallback_raw_strength"
    assert score.fallback_reason == "artifact_load_error"
    assert json.loads(score.score_metadata_json)["acts_on_book"] is False


def test_artifact_identity_mismatch_falls_back(db_session, tmp_path):
    signal_id = _seed_signal(db_session, idx=4, gap=-0.01)
    artifact_path = tmp_path / "swapped.pkl"
    _write_artifact(
        artifact_path,
        model_id="model-identity",
        schema_hash="artifact-schema-hash",
    )
    _add_model_registry(
        db_session,
        model_id="model-identity",
        artifact_uri=str(artifact_path),
        schema_hash="registry-schema-hash",
    )
    db_session.commit()

    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id="model-identity",
    )
    db_session.flush()

    assert score.score_source == "fallback_raw_strength"
    assert score.fallback_reason == "artifact_schema_hash_mismatch"
    metadata = json.loads(score.score_metadata_json)
    assert "feature_schema_hash" in metadata["identity_mismatches"]


def test_mutated_artifact_schema_with_stale_declared_hash_falls_back(
    db_session, tmp_path
):
    signal_id = _seed_signal(db_session, idx=6, gap=-0.01)
    base_schema = _feature_schema()
    base_hash = feature_schema_hash(base_schema)
    mutated_schema = dict(base_schema)
    mutated_schema["fields"] = list(base_schema["fields"]) + [
        {
            "name": "extra_safe_field",
            "source": "feature_snapshot_json",
            "path": "signal_context.extra_safe_field",
        }
    ]
    artifact_path = tmp_path / "stale-schema.pkl"
    _write_artifact(
        artifact_path,
        model_id="model-stale-schema",
        schema=mutated_schema,
        schema_hash=base_hash,
        training_feature_ranges=[
            {"min": -999.0, "max": 999.0} for _ in mutated_schema["fields"]
        ],
    )
    _add_model_registry(
        db_session,
        model_id="model-stale-schema",
        artifact_uri=str(artifact_path),
        schema_hash=base_hash,
    )
    db_session.commit()

    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id="model-stale-schema",
    )
    db_session.flush()

    assert score.score_source == "fallback_raw_strength"
    assert score.fallback_reason == "artifact_schema_hash_mismatch"
    metadata = json.loads(score.score_metadata_json)
    assert "declared_vs_actual_feature_schema_hash" in metadata["identity_mismatches"]


@pytest.mark.parametrize(
    "ranges",
    [
        [{"min": -999.0, "max": 999.0}],
        [{"min": -999.0, "max": 999.0}, "bad", {}, {}],
    ],
)
def test_malformed_or_short_otd_ranges_fall_back(db_session, tmp_path, ranges):
    signal_id = _seed_signal(db_session, idx=7, gap=-0.01)
    artifact_path = tmp_path / f"bad-ranges-{len(ranges)}.pkl"
    _write_artifact(
        artifact_path,
        model_id=f"model-bad-ranges-{len(ranges)}",
        training_feature_ranges=ranges,
    )
    _add_model_registry(
        db_session,
        model_id=f"model-bad-ranges-{len(ranges)}",
        artifact_uri=str(artifact_path),
    )
    db_session.commit()

    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id=f"model-bad-ranges-{len(ranges)}",
    )
    db_session.flush()

    assert score.score_source == "fallback_raw_strength"
    assert score.fallback_reason == "otd_check_error"


def test_predict_error_persists_fallback_without_raising(db_session, tmp_path):
    signal_id = _seed_signal(db_session, idx=5, gap=-0.01)
    artifact_path = tmp_path / "predict-error.pkl"
    _write_artifact(
        artifact_path,
        model_id="model-predict-error",
        model=ExplodingModel(),
    )
    _add_model_registry(
        db_session,
        model_id="model-predict-error",
        artifact_uri=str(artifact_path),
    )
    db_session.commit()

    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id="model-predict-error",
    )
    db_session.flush()

    assert score.score_source == "fallback_raw_strength"
    assert score.fallback_reason == "predict_error"
    assert db_session.query(SignalMLScore).count() == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_prediction_falls_back(db_session, tmp_path, value):
    signal_id = _seed_signal(db_session, idx=8, gap=-0.01)
    artifact_path = tmp_path / f"non-finite-{value}.pkl"
    _write_artifact(
        artifact_path,
        model_id=f"model-non-finite-{value}",
        model=NonFiniteModel(value),
    )
    _add_model_registry(
        db_session,
        model_id=f"model-non-finite-{value}",
        artifact_uri=str(artifact_path),
    )
    db_session.commit()

    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id=f"model-non-finite-{value}",
    )
    db_session.flush()

    assert score.score_source == "fallback_raw_strength"
    assert score.fallback_reason == "non_finite_score"
