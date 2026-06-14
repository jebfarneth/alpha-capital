from __future__ import annotations

import json
import math
import pickle
from importlib import import_module
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text

from alpha.db.models import (
    FeatureSnapshot,
    ForwardReturnObservation,
    MLModelRegistry,
    SignalMLScore,
    SignalRegistry,
)
from alpha.jobs.contracts import JobContext
from alpha.jobs.runner import run_job
from alpha.jobs.train_model import (
    MODEL_FAMILY,
    Stage1TrainModelJob,
    _finite,
    _fold_train_weights,
    _load_training_examples,
)
from alpha.ml.cv import CVExample, purged_embargoed_walk_forward_splits
from alpha.ml.inference import _raw_strength_fallback_score, score_signal_shadow
from alpha.ml.manifest_loader import PatternManifest, load_manifest, manifest_payload_hash
from alpha.ml.model_features import (
    FeatureSelectionError,
    _as_float,
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


def _manifest_payload(
    signal_ids: list[str] | None = None,
    *,
    feature_schema: dict | None = None,
    signal_horizon: str = "2d",
    horizon_sessions: int = 2,
) -> dict:
    payload = {
        "manifest_version": "test_stage1_manifest_v1",
        "manifest_sha256": "",
        "patterns": {
            "M4": {
                "signal_horizon": signal_horizon,
                "min_graded_cohorts": 8,
                "embargo_sessions": 2,
                "selection": {
                    "source": "forward_return_observations",
                    "statuses": ["computed"],
                    "horizon_sessions": horizon_sessions,
                    "signal_ids": signal_ids or [],
                },
                "label": {"field": "forward_return"},
                "feature_schema": feature_schema or _feature_schema(),
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
    signal_horizon: str = "2d",
    forward_return: float | None = None,
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
        signal_horizon=signal_horizon,
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
            signal_horizon=signal_horizon,
            forward_return=forward_return
            if forward_return is not None
            else (idx - 4) / 100.0,
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
        "pattern_clock": "eod",
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
        "pattern_clock": "eod",
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
    [
        "volume",
        "dollar_volume",
        "volume_expansion_20d",
        "volume_expansion_60d",
        "dollar_volume_expansion_20d",
        "dollar_volume_expansion_60d",
        "sigma_20d",
        "first_60m_return",
        "held_above_breakout_after_first_hour",
        "t1_before_stop",
        "advanced_features",
    ],
)
def test_intraday_signal_session_fields_are_deny_by_default(field_name):
    schema = {
        "pattern_id": "I12",
        "pattern_clock": "intraday",
        "fields": [
            {
                "name": field_name,
                "source": "market_path_feature_column",
                "feature_role": "signal_session",
                "feature_version": "market_path_daily_v3",
                "column": field_name,
            }
        ],
    }

    with pytest.raises(FeatureSelectionError):
        audit_feature_schema_no_leakage(schema)


@pytest.mark.parametrize("field_name", ["open_price", "previous_close", "gap_pct"])
def test_intraday_signal_session_allows_only_asof_signal_time_fields(field_name):
    schema = {
        "pattern_id": "I12",
        "pattern_clock": "intraday",
        "fields": [
            {
                "name": field_name,
                "source": "market_path_feature_column",
                "feature_role": "signal_session",
                "feature_version": "market_path_daily_v3",
                "column": field_name,
            }
        ],
    }
    audit_feature_schema_no_leakage(schema)


@pytest.mark.parametrize(
    "field_name",
    ["volume", "dollar_volume", "volume_expansion_20d", "t1_before_stop"],
)
def test_eod_signal_session_keeps_session_fields_allowed(field_name):
    schema = {
        "pattern_id": "M4",
        "pattern_clock": "eod",
        "fields": [
            {
                "name": field_name,
                "source": "market_path_feature_column",
                "feature_role": "signal_session",
                "feature_version": "market_path_daily_v3",
                "column": field_name,
            }
        ],
    }
    audit_feature_schema_no_leakage(schema)


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


@pytest.mark.parametrize(
    "field_name",
    [
        "dollar_volume",
        "avg20_volume",
        "close",
        "high",
        "session_volume",
        "first_60m_return",
    ],
)
def test_intraday_snapshot_fields_are_deny_by_default(field_name):
    schema = {
        "pattern_id": "I12",
        "pattern_clock": "intraday",
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


@pytest.mark.parametrize("field_name", ["gap", "mom20", "off_low252"])
def test_intraday_snapshot_allows_only_signal_time_fields(field_name):
    schema = {
        "pattern_id": "I12",
        "pattern_clock": "intraday",
        "fields": [
            {
                "name": field_name,
                "source": "feature_snapshot_json",
                "path": field_name,
            }
        ],
    }

    audit_feature_schema_no_leakage(schema)


@pytest.mark.parametrize(
    "path",
    [
        "research_only_leaky.gap",
        "research_only_leaky.mom20",
        "exit.off_low252",
        "research_only_leaky.dollar_volume",
    ],
)
def test_intraday_snapshot_rejects_nested_allowlisted_terminal_paths(path):
    schema = {
        "pattern_id": "I12",
        "pattern_clock": "intraday",
        "fields": [
            {
                "name": path.replace(".", "_"),
                "source": "feature_snapshot_json",
                "path": path,
            }
        ],
    }

    with pytest.raises(FeatureSelectionError):
        audit_feature_schema_no_leakage(schema)


def test_intraday_snapshot_audit_uses_the_read_path_not_decoy_column():
    schema = {
        "pattern_id": "I12",
        "pattern_clock": "intraday",
        "fields": [
            {
                "name": "decoy_gap",
                "source": "feature_snapshot_json",
                "column": "gap",
                "path": "research_only_leaky.dollar_volume",
            }
        ],
    }

    with pytest.raises(FeatureSelectionError):
        audit_feature_schema_no_leakage(schema)


def test_intraday_market_path_json_audit_uses_path_locator():
    schema = {
        "pattern_id": "I12",
        "pattern_clock": "intraday",
        "fields": [
            {
                "name": "nested_open",
                "source": "market_path_feature_json",
                "feature_role": "signal_session",
                "feature_version": "market_path_daily_v3",
                "path": "research_only_leaky.open_price",
            }
        ],
    }

    with pytest.raises(FeatureSelectionError):
        audit_feature_schema_no_leakage(schema)


def test_intraday_market_path_json_allows_flat_prior_window_paths():
    schema = {
        "pattern_id": "I12",
        "pattern_clock": "intraday",
        "fields": [
            {
                "name": "median_volume_20d",
                "source": "market_path_feature_json",
                "feature_role": "signal_session",
                "feature_version": "market_path_daily_v3",
                "path": "median_volume_20d",
            }
        ],
    }

    audit_feature_schema_no_leakage(schema)


def test_eod_market_path_json_keeps_nested_payloads_allowed():
    schema = {
        "pattern_id": "M4",
        "pattern_clock": "eod",
        "fields": [
            {
                "name": "nested_open",
                "source": "market_path_feature_json",
                "feature_role": "signal_session",
                "feature_version": "market_path_daily_v3",
                "path": "research_only_leaky.open_price",
            }
        ],
    }

    audit_feature_schema_no_leakage(schema)


@pytest.mark.parametrize(
    "feature_role",
    ["signal_day", "signal_day_t0", "t0_signal_context"],
)
@pytest.mark.parametrize(
    "field_name",
    ["return_from_entry_close", "close_price", "dollar_volume"],
)
def test_intraday_signal_day_roles_share_the_signal_time_allowlist(
    feature_role,
    field_name,
):
    schema = {
        "pattern_id": "I12",
        "pattern_clock": "intraday",
        "fields": [
            {
                "name": field_name,
                "source": "market_path_feature_column",
                "feature_role": feature_role,
                "feature_version": "market_path_daily_v3",
                "column": field_name,
            }
        ],
    }

    with pytest.raises(FeatureSelectionError):
        audit_feature_schema_no_leakage(schema)


def test_manifest_loader_runs_leakage_audit_for_intraday_schema(tmp_path):
    schema = {
        "schema_version": "leaky_intraday_v1",
        "pattern_id": "I12",
        "pattern_clock": "intraday",
        "fields": [
            {
                "name": "dollar_volume",
                "source": "feature_snapshot_json",
                "path": "signal_context.dollar_volume",
            }
        ],
    }
    payload = _manifest_payload(
        ["sig-1"],
        feature_schema=schema,
        signal_horizon="2d",
        horizon_sessions=2,
    )
    path = tmp_path / "leaky_manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(Exception, match="leakage audit"):
        load_manifest(path)


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


def test_fold_train_weights_are_recomputed_inside_each_fold():
    start = date(2025, 1, 1)
    rows = [
        CVExample(signal_id="a0", ticker="AAA", signal_date=start),
        CVExample(signal_id="a1", ticker="AAA", signal_date=start + timedelta(days=1)),
        CVExample(signal_id="b0", ticker="BBB", signal_date=start + timedelta(days=2)),
        CVExample(signal_id="b1", ticker="BBB", signal_date=start + timedelta(days=3)),
        CVExample(signal_id="a2", ticker="AAA", signal_date=start + timedelta(days=4)),
        CVExample(signal_id="a3", ticker="AAA", signal_date=start + timedelta(days=5)),
    ]

    weights = _fold_train_weights(rows, [0, 1, 2, 3])

    assert sum(weight for row, weight in zip(rows[:4], weights) if row.ticker == "AAA") == pytest.approx(1.0)
    assert sum(weight for row, weight in zip(rows[:4], weights) if row.ticker == "BBB") == pytest.approx(1.0)


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


def test_manifest_loader_rejects_noncanonical_or_mismatched_signal_horizon(tmp_path):
    mismatch = _manifest_payload(
        ["sig-1"],
        signal_horizon="15d",
        horizon_sessions=5,
    )
    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(json.dumps(mismatch, sort_keys=True))
    with pytest.raises(Exception, match="does not match horizon_sessions"):
        load_manifest(mismatch_path)

    noncanonical = _manifest_payload(
        ["sig-1"],
        signal_horizon="15",
        horizon_sessions=15,
    )
    noncanonical_path = tmp_path / "noncanonical.json"
    noncanonical_path.write_text(json.dumps(noncanonical, sort_keys=True))
    with pytest.raises(Exception, match="signal_horizon must be canonical"):
        load_manifest(noncanonical_path)


def test_training_loader_filters_to_manifest_signal_horizon(db_session, tmp_path):
    sig_2d = _seed_signal(db_session, idx=20, signal_horizon="2d")
    sig_5d = _seed_signal(db_session, idx=21, signal_horizon="5d")
    db_session.commit()
    manifest = load_manifest(_write_manifest(tmp_path, [sig_2d, sig_5d]))

    examples = _load_training_examples(db_session, pattern=manifest.pattern("M4"))

    assert [row.signal_id for row in examples] == [sig_2d]


def test_training_loader_rejects_mixed_loaded_horizons(db_session):
    sig_2d = _seed_signal(db_session, idx=22, signal_horizon="2d")
    sig_5d = _seed_signal(db_session, idx=23, signal_horizon="5d")
    db_session.commit()
    pattern = PatternManifest(
        pattern_id="M4",
        signal_horizon="",
        min_graded_cohorts=1,
        embargo_sessions=2,
        feature_schema=_feature_schema(),
        label={"field": "forward_return"},
        selection={
            "source": "forward_return_observations",
            "statuses": ["computed"],
            "horizon_sessions": 2,
            "signal_ids": [sig_2d, sig_5d],
        },
        diagnostics={},
    )

    with pytest.raises(RuntimeError, match="mixed signal_horizon"):
        _load_training_examples(db_session, pattern=pattern)


def test_training_loader_picks_latest_canonical_observation(db_session, tmp_path):
    signal_id = _seed_signal(db_session, idx=24, forward_return=-0.50)
    old = db_session.query(ForwardReturnObservation).filter(
        ForwardReturnObservation.signal_id == signal_id
    ).one()
    old.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    old.updated_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    signal = db_session.get(SignalRegistry, signal_id)
    db_session.add(
        ForwardReturnObservation(
            forward_return_observation_id="fro-24-fresh",
            signal_id=signal_id,
            pattern_id="M4",
            ticker=signal.ticker,
            direction="long",
            signal_timestamp=signal.signal_timestamp,
            signal_horizon="2d",
            forward_return=0.42,
            status="computed",
            input_hash="input-24-fresh",
            outcome_hash="outcome-24-fresh",
            created_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
            updated_at=datetime(2025, 1, 3, tzinfo=timezone.utc),
        )
    )
    db_session.commit()
    manifest = load_manifest(_write_manifest(tmp_path, [signal_id]))

    examples = _load_training_examples(db_session, pattern=manifest.pattern("M4"))

    assert len(examples) == 1
    assert examples[0].label == pytest.approx(0.42)


def test_training_loader_drops_corrupt_vectors_but_keeps_stored_null(
    db_session,
    tmp_path,
):
    corrupt_signal_id = _seed_signal(db_session, idx=32, gap=-0.01)
    missing_signal_id = _seed_signal(db_session, idx=33, gap=-0.01)
    corrupt_snapshot = db_session.get(FeatureSnapshot, "fs-32")
    corrupt_json = json.loads(corrupt_snapshot.feature_json)
    corrupt_json["signal_context"]["mom20"] = "not-a-number"
    corrupt_json["statuses"]["mom20"] = "computed"
    corrupt_snapshot.feature_json = json.dumps(corrupt_json, sort_keys=True)
    missing_snapshot = db_session.get(FeatureSnapshot, "fs-33")
    missing_json = json.loads(missing_snapshot.feature_json)
    missing_json["signal_context"].pop("mom20")
    missing_json["statuses"].pop("mom20", None)
    missing_snapshot.feature_json = json.dumps(missing_json, sort_keys=True)
    db_session.commit()
    manifest = load_manifest(
        _write_manifest(tmp_path, [corrupt_signal_id, missing_signal_id])
    )

    examples, metrics = _load_training_examples(
        db_session,
        pattern=manifest.pattern("M4"),
        return_metrics=True,
    )

    assert [row.signal_id for row in examples] == [missing_signal_id]
    assert examples[0].vector.missing_statuses["mom20"] == "stored_null"
    assert metrics == {
        "dropped_non_finite": 1,
        "dropped_non_finite_by_feature": {"mom20": 1},
    }


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


def test_null_model_fallback_index_migration_dedupes_existing_rows(
    db_session, monkeypatch
):
    signal_id = _seed_signal(db_session, idx=9, gap=-0.01)
    db_session.flush()
    db_session.execute(text("DROP INDEX ux_signal_ml_scores_fallback_null_model"))
    for score_id, score in (("dup-old", 1.0), ("dup-new", 2.0)):
        db_session.add(
            SignalMLScore(
                score_id=score_id,
                signal_id=signal_id,
                model_id=None,
                requested_model_id="missing-model",
                pattern_id="M4",
                ticker="T0",
                score=score,
                fallback_score=score,
                score_source="fallback_raw_strength",
                fallback_reason="model_missing",
                score_status="shadow",
                score_metadata_json=json.dumps({"acts_on_book": False}),
                scored_at=datetime.now(timezone.utc)
                + timedelta(seconds=score),
            )
        )
    db_session.flush()

    migration = import_module(
        "migrations.versions.fa0123456789_add_null_safe_ml_score_fallback_index"
    )
    ops = Operations(MigrationContext.configure(db_session.connection()))
    monkeypatch.setattr(migration, "op", ops)
    migration.upgrade()

    rows = db_session.query(SignalMLScore).filter(
        SignalMLScore.signal_id == signal_id,
        SignalMLScore.model_id.is_(None),
    ).all()
    assert len(rows) == 1
    assert rows[0].score == pytest.approx(2.0)


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


def test_direct_artifact_uri_without_registry_never_persists_model_shadow(
    db_session, tmp_path
):
    signal_id = _seed_signal(db_session, idx=10, gap=-0.01)
    artifact_path = tmp_path / "direct.pkl"
    _write_artifact(artifact_path, model_id="unregistered-model")
    db_session.commit()

    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        artifact_uri=str(artifact_path),
    )
    db_session.flush()

    assert score.model_id is None
    assert score.score_source == "fallback_raw_strength"
    assert score.fallback_reason == "model_registry_missing"
    assert db_session.query(SignalMLScore).filter(
        SignalMLScore.score_source == "model_shadow"
    ).count() == 0


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


@pytest.mark.parametrize(
    "bad_range",
    [
        {"min": float("nan"), "max": 999.0},
        {"min": -999.0, "max": float("inf")},
        {"min": 10.0, "max": 1.0},
    ],
)
def test_nonfinite_or_inverted_otd_bounds_fall_back(db_session, tmp_path, bad_range):
    signal_id = _seed_signal(db_session, idx=11, gap=-0.01)
    ranges = [{"min": -999.0, "max": 999.0} for _ in _feature_schema()["fields"]]
    ranges[0] = bad_range
    artifact_path = tmp_path / "bad-bound.pkl"
    _write_artifact(
        artifact_path,
        model_id="model-bad-bound",
        training_feature_ranges=ranges,
    )
    _add_model_registry(
        db_session,
        model_id="model-bad-bound",
        artifact_uri=str(artifact_path),
    )
    db_session.commit()

    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id="model-bad-bound",
    )
    db_session.flush()

    assert score.score_source == "fallback_raw_strength"
    assert score.fallback_reason == "otd_check_error"


@pytest.mark.parametrize("raw_value", [float("inf"), "Infinity", "1e400"])
def test_feature_reader_treats_infinite_values_as_typed_missing(raw_value):
    assert math.isnan(_as_float(raw_value))


def test_open_bound_infinite_feature_value_falls_back(db_session, tmp_path):
    signal_id = _seed_signal(db_session, idx=25, gap=-0.01)
    snapshot = db_session.get(FeatureSnapshot, "fs-25")
    feature_json = json.loads(snapshot.feature_json)
    feature_json["signal_context"]["mom20"] = "1e400"
    snapshot.feature_json = json.dumps(feature_json, sort_keys=True)
    artifact_path = tmp_path / "open-bound-inf.pkl"
    _write_artifact(
        artifact_path,
        model_id="model-open-bound-inf",
        training_feature_ranges=[
            {"min": None, "max": None} for _ in _feature_schema()["fields"]
        ],
    )
    _add_model_registry(
        db_session,
        model_id="model-open-bound-inf",
        artifact_uri=str(artifact_path),
    )
    db_session.commit()

    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id="model-open-bound-inf",
    )
    db_session.flush()

    assert score.score_source == "fallback_raw_strength"
    assert score.fallback_reason == "out_of_training_distribution"
    assert db_session.query(SignalMLScore).filter(
        SignalMLScore.score_source == "model_shadow"
    ).count() == 0


def test_non_finite_stored_value_overrides_truthy_status_path(db_session, tmp_path):
    signal_id = _seed_signal(db_session, idx=26, gap=-0.01)
    snapshot = db_session.get(FeatureSnapshot, "fs-26")
    feature_json = json.loads(snapshot.feature_json)
    feature_json["signal_context"]["mom20"] = "1e400"
    feature_json["statuses"]["mom20"] = "computed"
    snapshot.feature_json = json.dumps(feature_json, sort_keys=True)
    schema = {
        "schema_version": "non_finite_status_v1",
        "pattern_id": "M4",
        "pattern_clock": "eod",
        "fields": [
            {
                "name": "mom20",
                "source": "feature_snapshot_json",
                "path": "signal_context.mom20",
                "status_path": "statuses.mom20",
            }
        ],
    }
    artifact_path = tmp_path / "non-finite-status.pkl"
    _write_artifact(
        artifact_path,
        model_id="model-non-finite-status",
        schema=schema,
        training_feature_ranges=[{"min": None, "max": None}],
    )
    _add_model_registry(
        db_session,
        model_id="model-non-finite-status",
        artifact_uri=str(artifact_path),
        schema_hash=feature_schema_hash(schema),
    )
    db_session.commit()

    vector = select_features(db_session, signal_id, schema)
    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id="model-non-finite-status",
    )
    db_session.flush()

    assert vector.missing_statuses["mom20"] == "non_finite_stored_value"
    assert score.score_source == "fallback_raw_strength"
    assert score.fallback_reason == "out_of_training_distribution"
    metadata = json.loads(score.score_metadata_json)
    assert metadata["reason"] == "non_finite_stored_value"
    assert db_session.query(SignalMLScore).filter(
        SignalMLScore.score_source == "model_shadow"
    ).count() == 0


@pytest.mark.parametrize(
    ("idx", "raw_value", "transform"),
    [
        (27, "not-a-number", None),
        (28, {}, None),
        (29, [], None),
        (30, -2.0, "log1p"),
    ],
)
def test_present_values_that_become_nan_fall_back(
    db_session,
    tmp_path,
    idx,
    raw_value,
    transform,
):
    signal_id = _seed_signal(db_session, idx=idx, gap=-0.01)
    snapshot = db_session.get(FeatureSnapshot, f"fs-{idx}")
    feature_json = json.loads(snapshot.feature_json)
    feature_json["signal_context"]["mom20"] = raw_value
    feature_json["statuses"]["mom20"] = "computed"
    snapshot.feature_json = json.dumps(feature_json, sort_keys=True)
    field = {
        "name": "mom20",
        "source": "feature_snapshot_json",
        "path": "signal_context.mom20",
        "status_path": "statuses.mom20",
    }
    if transform is not None:
        field["transform"] = transform
    schema = {
        "schema_version": f"present_nan_status_v{idx}",
        "pattern_id": "M4",
        "pattern_clock": "eod",
        "fields": [field],
    }
    artifact_path = tmp_path / f"present-nan-{idx}.pkl"
    _write_artifact(
        artifact_path,
        model_id=f"model-present-nan-{idx}",
        schema=schema,
        training_feature_ranges=[{"min": None, "max": None}],
    )
    _add_model_registry(
        db_session,
        model_id=f"model-present-nan-{idx}",
        artifact_uri=str(artifact_path),
        schema_hash=feature_schema_hash(schema),
    )
    db_session.commit()

    vector = select_features(db_session, signal_id, schema)
    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id=f"model-present-nan-{idx}",
    )
    db_session.flush()

    assert vector.missing_statuses["mom20"] == "non_finite_stored_value"
    assert score.score_source == "fallback_raw_strength"
    assert score.fallback_reason == "out_of_training_distribution"
    assert db_session.query(SignalMLScore).filter(
        SignalMLScore.score_source == "model_shadow"
    ).count() == 0


def test_absent_value_stays_typed_missing_and_scores_shadow(db_session, tmp_path):
    signal_id = _seed_signal(db_session, idx=31, gap=-0.01)
    snapshot = db_session.get(FeatureSnapshot, "fs-31")
    feature_json = json.loads(snapshot.feature_json)
    feature_json["signal_context"].pop("mom20")
    feature_json["statuses"].pop("mom20", None)
    snapshot.feature_json = json.dumps(feature_json, sort_keys=True)
    schema = {
        "schema_version": "absent_value_status_v1",
        "pattern_id": "M4",
        "pattern_clock": "eod",
        "fields": [
            {
                "name": "mom20",
                "source": "feature_snapshot_json",
                "path": "signal_context.mom20",
                "status_path": "statuses.mom20",
            }
        ],
    }
    artifact_path = tmp_path / "absent-value.pkl"
    _write_artifact(
        artifact_path,
        model_id="model-absent-value",
        schema=schema,
        training_feature_ranges=[{"min": None, "max": None}],
    )
    _add_model_registry(
        db_session,
        model_id="model-absent-value",
        artifact_uri=str(artifact_path),
        schema_hash=feature_schema_hash(schema),
    )
    db_session.commit()

    vector = select_features(db_session, signal_id, schema)
    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id="model-absent-value",
    )
    db_session.flush()

    assert vector.missing_statuses["mom20"] == "stored_null"
    assert score.score_source == "model_shadow"
    assert score.fallback_reason is None


def test_malformed_parent_path_falls_back(db_session, tmp_path):
    signal_id = _seed_signal(db_session, idx=34, gap=-0.01)
    snapshot = db_session.get(FeatureSnapshot, "fs-34")
    feature_json = json.loads(snapshot.feature_json)
    feature_json["signal_context"] = []
    feature_json["statuses"]["mom20"] = "computed"
    snapshot.feature_json = json.dumps(feature_json, sort_keys=True)
    schema = {
        "schema_version": "malformed_parent_status_v1",
        "pattern_id": "M4",
        "pattern_clock": "eod",
        "fields": [
            {
                "name": "mom20",
                "source": "feature_snapshot_json",
                "path": "signal_context.mom20",
                "status_path": "statuses.mom20",
            }
        ],
    }
    artifact_path = tmp_path / "malformed-parent.pkl"
    _write_artifact(
        artifact_path,
        model_id="model-malformed-parent",
        schema=schema,
        training_feature_ranges=[{"min": None, "max": None}],
    )
    _add_model_registry(
        db_session,
        model_id="model-malformed-parent",
        artifact_uri=str(artifact_path),
        schema_hash=feature_schema_hash(schema),
    )
    db_session.commit()

    vector = select_features(db_session, signal_id, schema)
    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id="model-malformed-parent",
    )
    db_session.flush()

    assert vector.missing_statuses["mom20"] == "non_finite_stored_value"
    assert score.score_source == "fallback_raw_strength"
    assert score.fallback_reason == "out_of_training_distribution"
    assert db_session.query(SignalMLScore).filter(
        SignalMLScore.score_source == "model_shadow"
    ).count() == 0


def test_flat_absent_key_stays_typed_missing_and_scores_shadow(db_session, tmp_path):
    signal_id = _seed_signal(db_session, idx=35, gap=-0.01)
    schema = {
        "schema_version": "flat_absent_status_v1",
        "pattern_id": "M4",
        "pattern_clock": "eod",
        "fields": [
            {
                "name": "mom20",
                "source": "feature_snapshot_json",
                "path": "mom20",
            }
        ],
    }
    artifact_path = tmp_path / "flat-absent.pkl"
    _write_artifact(
        artifact_path,
        model_id="model-flat-absent",
        schema=schema,
        training_feature_ranges=[{"min": None, "max": None}],
    )
    _add_model_registry(
        db_session,
        model_id="model-flat-absent",
        artifact_uri=str(artifact_path),
        schema_hash=feature_schema_hash(schema),
    )
    db_session.commit()

    vector = select_features(db_session, signal_id, schema)
    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id="model-flat-absent",
    )
    db_session.flush()

    assert vector.missing_statuses["mom20"] == "stored_null"
    assert score.score_source == "model_shadow"
    assert score.fallback_reason is None


def test_training_feature_range_filter_excludes_infinities():
    assert _finite([math.nan, float("inf"), float("-inf"), 1.25]) == [1.25]


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


def test_non_finite_raw_strength_fallback_uses_safe_default():
    signal = SignalRegistry(
        signal_id="nan-raw",
        pattern_id="M4",
        ticker="NAN",
        direction="long",
        signal_timestamp=datetime.now(timezone.utc),
        raw_signal_strength=float("nan"),
        raw_expected_edge=float("inf"),
        feature_snapshot_id="unused",
        signal_identity_hash="nan-raw",
    )

    score = _raw_strength_fallback_score(signal)

    assert math.isfinite(score)
    assert score == pytest.approx(0.0)


def test_infinite_raw_strength_fallback_uses_safe_default(db_session):
    signal_id = _seed_signal(db_session, idx=12, gap=-0.01)
    signal = db_session.get(SignalRegistry, signal_id)
    signal.raw_signal_strength = float("inf")
    signal.raw_expected_edge = float("inf")
    db_session.commit()

    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id="missing-model",
    )
    db_session.flush()

    assert score.score_source == "fallback_raw_strength"
    assert score.fallback_reason == "model_missing"
    assert math.isfinite(score.score)
    assert score.score == pytest.approx(0.0)
