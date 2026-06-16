from __future__ import annotations

import json
import math
import os
import pickle
import random
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
    MarketPathFeature,
    SecurityIdentitySnapshot,
    SignalMLScore,
    SignalRegistry,
    UniverseScan,
)
from alpha.jobs.contracts import JobContext
from alpha.jobs.runner import run_job
from alpha.jobs.train_model import (
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_TRAINER_DB_TIMEOUT_MS,
    MODEL_FAMILY,
    PREDICTION_VARIANCE_EPSILON,
    Stage1TrainModelJob,
    TrainingExample,
    _apply_trainer_db_timeout_env,
    _cross_validate,
    _finite,
    _fold_train_weights,
    _load_training_examples,
    _mean_fold_metrics,
    _parse_args,
    _prediction_quality_metrics,
    _prediction_weight_bins,
    _resolved_model_params,
    _score_percentiles,
    _top_quantile_mean,
    _top_quantile_unreliable,
)
from alpha.market_calendar import next_us_equity_session, nth_us_equity_session
from alpha.ml.cv import (
    CVExample,
    PurgedEmbargoedFold,
    purged_embargoed_walk_forward_splits,
)
from alpha.ml.inference import _raw_strength_fallback_score, score_signal_shadow
from alpha.ml.manifest_loader import PatternManifest, load_manifest, manifest_payload_hash
from alpha.ml.model_features import (
    FeatureSelectionError,
    _as_float,
    SelectedFeatureVector,
    audit_feature_schema_no_leakage,
    feature_schema_hash,
    feature_vector_hash,
    select_features,
)
from alpha.security_identity import _canonical_ticker, resolve_security_identities_for_tickers


class ConstantModel:
    def predict(self, rows):
        return [0.123 for _ in rows]


class ConstantFitModel:
    def fit(self, rows, labels, sample_weight=None):
        return self

    def predict(self, rows):
        return [0.0 for _ in rows]


class EpsilonLadderFitModel:
    def fit(self, rows, labels, sample_weight=None):
        return self

    def predict(self, rows):
        if not rows:
            return []
        step = PREDICTION_VARIANCE_EPSILON / (2.0 * max(len(rows), 1))
        return [idx * step for idx, _row in enumerate(rows)]


class NonFiniteFitModel:
    def __init__(self, value):
        self.value = value

    def fit(self, rows, labels, sample_weight=None):
        return self

    def predict(self, rows):
        return [self.value for _ in rows]


class FoldOffsetFitModel:
    def __init__(self, offset: float):
        self.offset = offset

    def fit(self, rows, labels, sample_weight=None):
        return self

    def predict(self, rows):
        return [self.offset + float(row[0]) for row in rows]


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
    direction: str = "long",
    allow_deferred_pit: bool = False,
    oos_quality_gate: dict | None = None,
    model_params: dict | None = None,
) -> dict:
    selection = {
        "source": "forward_return_observations",
        "statuses": ["computed"],
        "horizon_sessions": horizon_sessions,
        "signal_ids": signal_ids or [],
    }
    if allow_deferred_pit:
        selection["allow_deferred_pit"] = True
    payload = {
        "manifest_version": "test_stage1_manifest_v1",
        "manifest_sha256": "",
        "patterns": {
            "M4": {
                "direction": direction,
                "signal_horizon": signal_horizon,
                "min_graded_cohorts": 8,
                "embargo_sessions": 2,
                "selection": selection,
                "label": {"field": "forward_return"},
                "feature_schema": feature_schema or _feature_schema(),
                "diagnostics": {"pooled_metrics_diagnostic_only": True},
                "model_params": model_params
                if model_params is not None
                else {
                    "min_samples_leaf": 2,
                    "l2_regularization": 0.01,
                    "early_stopping": False,
                },
            }
        },
    }
    if oos_quality_gate is not None:
        payload["patterns"]["M4"]["oos_quality_gate"] = oos_quality_gate
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
    direction: str = "long",
    signal_status: str = "active",
    point_in_time_passed: bool = True,
    lookahead_guard_passed: bool = True,
    forward_return_status: str = "computed",
    entry_session_date: date | None = None,
    exit_session_date: date | None = None,
    missing_realized_dates: bool = False,
) -> str:
    signal_id = f"sig-{idx}"
    ticker = ticker or f"T{idx % 3}"
    ts_date = signal_date or (datetime(2025, 1, 2).date() + timedelta(days=idx))
    ts = datetime.combine(ts_date, datetime.min.time(), timezone.utc)
    if missing_realized_dates:
        realized_entry_session = None
        realized_exit_session = None
    else:
        realized_entry_session = entry_session_date or next_us_equity_session(ts_date)
        try:
            horizon_session_count = int(str(signal_horizon).removesuffix("d"))
        except ValueError:
            horizon_session_count = 2
        realized_exit_session = exit_session_date or nth_us_equity_session(
            realized_entry_session + timedelta(days=1),
            horizon_session_count,
        )
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
        direction=direction,
        signal_timestamp=ts,
        raw_signal_strength=1.0 + idx,
        raw_expected_edge=0.0,
        signal_horizon=signal_horizon,
        feature_snapshot_id=feature_snapshot_id,
        signal_status=signal_status,
        point_in_time_passed=point_in_time_passed,
        lookahead_guard_passed=lookahead_guard_passed,
        forward_return_status=forward_return_status,
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
            direction=direction,
            signal_timestamp=ts,
            signal_horizon=signal_horizon,
            forward_return=forward_return
            if forward_return is not None
            else (idx - 4) / 100.0,
            status="computed",
            entry_session_date=(
                realized_entry_session.isoformat()
                if realized_entry_session is not None
                else None
            ),
            exit_session_date=(
                realized_exit_session.isoformat()
                if realized_exit_session is not None
                else None
            ),
            input_hash=f"input-{idx}",
            outcome_hash=f"outcome-{idx}",
        )
    )
    return signal_id


def _seed_rename_identity_pair(db_session) -> None:
    scan = UniverseScan(
        scan_id="scan-rename",
        trading_date="2025-02-03",
        asof_timestamp=datetime(2025, 2, 3, tzinfo=timezone.utc),
        provider="test",
        raw_count=2,
        deduped_count=2,
        duplicate_symbol_count=0,
        included_count=2,
        excluded_count=0,
        run_status="finished",
    )
    db_session.add(scan)
    events = json.dumps(
        [
            {
                "old_ticker": "CEP",
                "new_ticker": "XXI",
                "event_date": "2025-01-15",
            }
        ],
        sort_keys=True,
    )
    db_session.add(
        SecurityIdentitySnapshot(
            security_identity_snapshot_id="identity-cep",
            scan_id=scan.scan_id,
            ticker="CEP",
            cik="0001234567",
            active=False,
            ticker_events_json=events,
            identity_status="present",
            source_provider="test",
            asof_timestamp=datetime(2025, 1, 20, tzinfo=timezone.utc),
        )
    )
    db_session.add(
        SecurityIdentitySnapshot(
            security_identity_snapshot_id="identity-xxi",
            scan_id=scan.scan_id,
            ticker="XXI",
            cik="0001234567",
            active=True,
            ticker_events_json=events,
            identity_status="present",
            source_provider="test",
            asof_timestamp=datetime(2025, 2, 3, tzinfo=timezone.utc),
        )
    )


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
    returns_by_bucket = [-0.03, -0.01, 0.01, 0.03, 0.05]
    for idx in range(count):
        signal_ids.append(
            _seed_signal(
                db_session,
                idx=idx,
                ticker=f"T{idx}",
                gap=-0.02 + (idx % 5) * 0.01,
                forward_return=returns_by_bucket[idx % 5],
            )
        )
    if count >= 2:
        scan = UniverseScan(
            scan_id="scan-training-cluster",
            trading_date="2025-01-02",
            asof_timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc),
            provider="test",
            raw_count=2,
            deduped_count=2,
            duplicate_symbol_count=0,
            included_count=2,
            excluded_count=0,
            run_status="finished",
        )
        db_session.add(scan)
        for ticker in ("T0", "T1"):
            db_session.add(
                SecurityIdentitySnapshot(
                    security_identity_snapshot_id=f"identity-{ticker.lower()}",
                    scan_id=scan.scan_id,
                    ticker=ticker,
                    cik="0007777777",
                    active=True,
                    identity_status="present",
                    source_provider="test",
                    asof_timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc),
                )
            )
    db_session.commit()
    return signal_ids


def _training_example(idx: int, *, label: float, signal_date: date) -> TrainingExample:
    names = ["x"]
    values = [float(idx)]
    return TrainingExample(
        signal_id=f"cv-{idx}",
        ticker=f"T{idx}",
        security_identity=f"ticker:T{idx}",
        signal_date=signal_date,
        label=label,
        vector=SelectedFeatureVector(
            signal_id=f"cv-{idx}",
            pattern_id="M4",
            feature_names=names,
            values=values,
            feature_schema_hash="fixture-schema",
            feature_vector_hash=feature_vector_hash(names, values),
            missing_statuses={},
        ),
    )


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


def test_leakage_audit_allows_intraday_pit_catalyst_flags():
    schema = {
        "pattern_id": "I12",
        "pattern_clock": "intraday",
        "fields": [
            {
                "name": name,
                "source": "feature_snapshot_json",
                "path": name,
            }
            for name in (
                "catalyst_dilution_avoid",
                "catalyst_recent_shelf_filing",
                "catalyst_nt_late_filer",
                "catalyst_fda_amplifier",
                "catalyst_compliance_amplifier",
            )
        ],
    }

    audit_feature_schema_no_leakage(schema)


def test_catalyst_null_flags_are_missing_not_zero():
    assert math.isnan(_as_float(None))
    assert _as_float(False) == 0.0


def test_signal_registry_unknown_column_fails_closed(tmp_path):
    schema = {
        "schema_version": "bad_signal_registry_column_v1",
        "pattern_id": "M4",
        "pattern_clock": "eod",
        "fields": [
            {
                "name": "raw_strength_typo",
                "source": "signal_registry",
                "column": "raw_signal_strenght",
            }
        ],
    }

    with pytest.raises(FeatureSelectionError, match="unknown column"):
        audit_feature_schema_no_leakage(schema)

    manifest_path = tmp_path / "bad-signal-registry-column.json"
    manifest_path.write_text(
        json.dumps(_manifest_payload(["sig-1"], feature_schema=schema), sort_keys=True)
    )
    with pytest.raises(Exception, match="leakage audit"):
        load_manifest(manifest_path)


def test_signal_registry_valid_null_column_stays_typed_missing_and_scores(
    db_session,
    tmp_path,
):
    audit_feature_schema_no_leakage(
        {
            "schema_version": "valid_raw_strength_column_v1",
            "pattern_id": "M4",
            "pattern_clock": "eod",
            "fields": [
                {
                    "name": "raw_signal_strength",
                    "source": "signal_registry",
                    "column": "raw_signal_strength",
                }
            ],
        }
    )
    signal_id = _seed_signal(db_session, idx=38, gap=-0.01)
    signal = db_session.get(SignalRegistry, signal_id)
    signal.data_confidence = None
    schema = {
        "schema_version": "signal_registry_null_column_v1",
        "pattern_id": "M4",
        "pattern_clock": "eod",
        "fields": [
            {
                "name": "data_confidence",
                "source": "signal_registry",
                "column": "data_confidence",
            }
        ],
    }
    artifact_path = tmp_path / "signal-registry-null.pkl"
    _write_artifact(
        artifact_path,
        model_id="model-signal-registry-null",
        schema=schema,
        training_feature_ranges=[{"min": None, "max": None}],
    )
    _add_model_registry(
        db_session,
        model_id="model-signal-registry-null",
        artifact_uri=str(artifact_path),
        schema_hash=feature_schema_hash(schema),
    )
    db_session.commit()

    vector = select_features(db_session, signal_id, schema)
    score = score_signal_shadow(
        db_session,
        signal_id=signal_id,
        model_id="model-signal-registry-null",
    )
    db_session.flush()

    assert vector.missing_statuses["data_confidence"] == "stored_null"
    assert score.score_source == "model_shadow"
    assert score.fallback_reason is None


def test_market_path_feature_column_unknown_column_fails_closed():
    assert not hasattr(MarketPathFeature, "open_prize")
    schema = {
        "schema_version": "bad_market_path_column_v1",
        "pattern_id": "M4",
        "pattern_clock": "eod",
        "fields": [
            {
                "name": "open_prize",
                "source": "market_path_feature_column",
                "feature_role": "signal_session",
                "feature_version": "market_path_daily_v3",
                "column": "open_prize",
            }
        ],
    }

    with pytest.raises(FeatureSelectionError, match="unknown column"):
        audit_feature_schema_no_leakage(schema)


def test_market_path_feature_column_valid_column_passes():
    assert hasattr(MarketPathFeature, "open_price")
    schema = {
        "schema_version": "valid_market_path_column_v1",
        "pattern_id": "M4",
        "pattern_clock": "eod",
        "fields": [
            {
                "name": "open_price",
                "source": "market_path_feature_column",
                "feature_role": "signal_session",
                "feature_version": "market_path_daily_v3",
                "column": "open_price",
            }
        ],
    }

    audit_feature_schema_no_leakage(schema)


def test_signal_session_fields_require_allowlist_for_all_pattern_clocks():
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
    with pytest.raises(FeatureSelectionError):
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
def test_eod_signal_session_rejects_full_day_or_outcome_fields(field_name):
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
    with pytest.raises(FeatureSelectionError):
        audit_feature_schema_no_leakage(schema)


@pytest.mark.parametrize("field_name", ["open_price", "previous_close", "gap_pct"])
def test_eod_signal_session_allows_only_asof_entry_fields(field_name):
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
    "path",
    ["gate_values.full_day_volume_ratio", "signal_context.t1_before_stop"],
)
def test_eod_snapshot_rejects_named_leaky_locators(path):
    schema = {
        "pattern_id": "M4",
        "pattern_clock": "eod",
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


@pytest.mark.parametrize("field_name", ["chase_pct", "chase_over_hb_pct"])
def test_intraday_snapshot_rejects_execution_chase_fields(field_name):
    schema = {
        "pattern_id": "I11",
        "pattern_clock": "intraday",
        "fields": [
            {
                "name": field_name,
                "source": "feature_snapshot_json",
                "path": field_name,
            }
        ],
    }

    with pytest.raises(FeatureSelectionError):
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


def test_eod_market_path_json_rejects_nested_payloads():
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

    with pytest.raises(FeatureSelectionError):
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
        CVExample(
            signal_id=f"s{i}",
            ticker=f"T{i}",
            security_identity=f"ticker:T{i}",
            signal_date=start + timedelta(days=i),
        )
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
        CVExample(
            signal_id="a0",
            ticker="AAA",
            security_identity="sec:A",
            signal_date=start,
        ),
        CVExample(
            signal_id="a1",
            ticker="AAA",
            security_identity="sec:A",
            signal_date=start + timedelta(days=1),
        ),
        CVExample(
            signal_id="b0",
            ticker="BBB",
            security_identity="sec:B",
            signal_date=start + timedelta(days=2),
        ),
        CVExample(
            signal_id="b1",
            ticker="BBB",
            security_identity="sec:B",
            signal_date=start + timedelta(days=3),
        ),
        CVExample(
            signal_id="a2",
            ticker="AAA",
            security_identity="sec:A",
            signal_date=start + timedelta(days=4),
        ),
        CVExample(
            signal_id="a3",
            ticker="AAA",
            security_identity="sec:A",
            signal_date=start + timedelta(days=5),
        ),
    ]

    weights = _fold_train_weights(rows, [0, 1, 2, 3])

    assert sum(
        weight
        for row, weight in zip(rows[:4], weights)
        if row.security_identity == "sec:A"
    ) == pytest.approx(1.0)
    assert sum(
        weight
        for row, weight in zip(rows[:4], weights)
        if row.security_identity == "sec:B"
    ) == pytest.approx(1.0)


def test_purged_embargoed_cv_excludes_same_security_identity_from_train_and_test():
    start = date(2025, 1, 1)
    rows = [
        CVExample("cep-old", "CEP", "sec:rename", start),
        CVExample("o1", "O1", "sec:o1", start + timedelta(days=1)),
        CVExample("o2", "O2", "sec:o2", start + timedelta(days=2)),
        CVExample("o3", "O3", "sec:o3", start + timedelta(days=3)),
        CVExample("xxi-new", "XXI", "sec:rename", start + timedelta(days=4)),
        CVExample("o4", "O4", "sec:o4", start + timedelta(days=5)),
        CVExample("o5", "O5", "sec:o5", start + timedelta(days=6)),
        CVExample("o6", "O6", "sec:o6", start + timedelta(days=7)),
    ]

    folds = purged_embargoed_walk_forward_splits(
        rows,
        n_splits=2,
        horizon_sessions=0,
        embargo_sessions=0,
    )

    saw_rename_test_fold = False
    for fold in folds:
        train_identities = {rows[idx].security_identity for idx in fold.train_indices}
        test_identities = {rows[idx].security_identity for idx in fold.test_indices}
        assert train_identities.isdisjoint(test_identities)
        if "sec:rename" in test_identities:
            saw_rename_test_fold = True
            assert "cep-old" not in {rows[idx].signal_id for idx in fold.train_indices}
    assert saw_rename_test_fold is True


def test_prediction_quality_metrics_capture_perfect_top_decile_lift():
    labels = [float(idx) for idx in range(100)]
    preds = list(labels)

    metrics = _prediction_quality_metrics(labels, preds)

    assert metrics["rank_ic"] == pytest.approx(1.0)
    assert metrics["value_ic"] == pytest.approx(1.0)
    assert metrics["top_decile_lift"] > 1.0
    assert metrics["top_decile_spread"] > 0.0
    assert metrics["top_decile_win_rate"] == pytest.approx(1.0)
    assert metrics["population_win_rate"] == pytest.approx(0.99)
    assert metrics["top_decile_too_small"] is False
    assert len(metrics["per_decile_mean_label"]) == 10
    assert metrics["per_decile_mean_label"] == sorted(metrics["per_decile_mean_label"])


def test_prediction_quality_metrics_constant_predictions_do_not_inherit_time_order():
    labels = [float(idx) for idx in range(100)]
    preds = [0.0 for _ in labels]

    metrics = _prediction_quality_metrics(labels, preds)

    assert math.isnan(metrics["rank_ic"])
    assert math.isnan(_top_quantile_mean(labels, preds))
    assert metrics["top_decile_unreliable"] is True
    assert math.isnan(metrics["top_decile_lift"])
    assert math.isnan(metrics["top_decile_spread"])
    assert math.isnan(metrics["top_decile_mean_label"])
    assert math.isnan(metrics["bottom_9_deciles_mean_label"])
    assert math.isnan(metrics["weighted_bottom_9_deciles_mean_label"])


def test_score_percentiles_group_sub_epsilon_spread_as_one_tie():
    preds = [
        idx * PREDICTION_VARIANCE_EPSILON / 20.0
        for idx in range(10)
    ]

    percentiles = _score_percentiles(preds)

    assert len(set(percentiles)) == 1
    assert percentiles[0] == pytest.approx(0.5)


def test_prediction_quality_metrics_sub_epsilon_predictions_have_no_ic_signal():
    labels = [float(idx) for idx in range(100)]
    preds = [
        idx * PREDICTION_VARIANCE_EPSILON / 200.0
        for idx in range(100)
    ]

    metrics = _prediction_quality_metrics(labels, preds)

    assert math.isnan(metrics["value_ic"])
    assert math.isnan(metrics["rank_ic"])
    assert metrics["top_decile_unreliable"] is True
    assert metrics["per_decile_unreliable"] is True


@pytest.mark.parametrize(
    ("labels", "preds", "weights", "match"),
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0], None, "matching lengths"),
        ([1.0, math.nan, 3.0], [1.0, 2.0, 3.0], None, "y_true contains non-finite"),
        ([1.0, 2.0, 3.0], [1.0, math.nan, 3.0], None, "y_pred contains non-finite"),
        ([1.0, 2.0, 3.0], [1.0, float("inf"), 3.0], None, "y_pred contains non-finite"),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 1.0], "weights must match"),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, -1.0, 1.0], "negative"),
    ],
)
def test_prediction_quality_metrics_rejects_corrupt_metric_vectors(
    labels,
    preds,
    weights,
    match,
):
    with pytest.raises(ValueError, match=match):
        _prediction_quality_metrics(labels, preds, weights=weights)


def test_cross_validate_constant_fold_predictions_keep_pooled_rank_ic_unreliable(
    monkeypatch,
):
    train_model_module = import_module("alpha.jobs.train_model")
    monkeypatch.setattr(
        train_model_module,
        "_new_gbrt_model",
        lambda **_kwargs: ConstantFitModel(),
    )
    start = date(2025, 1, 1)
    examples = [
        _training_example(idx, label=float(idx), signal_date=start + timedelta(days=idx))
        for idx in range(80)
    ]

    metrics = _cross_validate(
        examples,
        horizon_sessions=1,
        embargo_sessions=1,
        n_splits=2,
        max_iter=1,
        random_state=7,
    )

    assert math.isnan(metrics["rank_ic"])
    assert (
        metrics["rank_ic_score_basis"]
        == "fold_normalized_prediction_percentile_midrank"
    )
    assert metrics["top_decile_unreliable"] is True
    assert math.isnan(metrics["top_decile_lift"])
    assert math.isnan(metrics["top_decile_mean_label"])
    assert math.isnan(metrics["top_quantile_label_mean"])
    assert all(math.isnan(fold["top_quantile_label_mean"]) for fold in metrics["folds"])


def test_cross_validate_rejects_non_finite_fold_predictions(monkeypatch):
    train_model_module = import_module("alpha.jobs.train_model")
    monkeypatch.setattr(
        train_model_module,
        "_new_gbrt_model",
        lambda **_kwargs: NonFiniteFitModel(float("nan")),
    )
    start = date(2025, 1, 1)
    examples = [
        _training_example(idx, label=float(idx), signal_date=start + timedelta(days=idx))
        for idx in range(80)
    ]

    with pytest.raises(ValueError, match="y_pred contains non-finite"):
        _cross_validate(
            examples,
            horizon_sessions=1,
            embargo_sessions=1,
            n_splits=2,
            max_iter=1,
            random_state=7,
        )


def test_cross_validate_sub_epsilon_fold_predictions_do_not_resurrect_pooled_edge(
    monkeypatch,
):
    train_model_module = import_module("alpha.jobs.train_model")
    monkeypatch.setattr(
        train_model_module,
        "_new_gbrt_model",
        lambda **_kwargs: EpsilonLadderFitModel(),
    )
    start = date(2025, 1, 1)
    examples = [
        _training_example(idx, label=float(idx), signal_date=start + timedelta(days=idx))
        for idx in range(80)
    ]

    metrics = _cross_validate(
        examples,
        horizon_sessions=1,
        embargo_sessions=1,
        n_splits=2,
        max_iter=1,
        random_state=7,
    )

    assert all(fold["top_decile_unreliable"] for fold in metrics["folds"])
    assert metrics["top_decile_unreliable"] is True
    assert math.isnan(metrics["top_decile_lift"])
    assert math.isnan(metrics["top_decile_mean_label"])
    assert math.isnan(metrics["value_ic"])
    assert math.isnan(metrics["rank_ic"])


def test_cross_validate_keeps_normalized_pooled_ic_when_raw_scores_have_fold_offsets(
    monkeypatch,
):
    train_model_module = import_module("alpha.jobs.train_model")

    def model_factory(**kwargs):
        offset = 10000.0 if kwargs["random_state"] == 7 else 0.0
        return FoldOffsetFitModel(offset)

    monkeypatch.setattr(train_model_module, "_new_gbrt_model", model_factory)
    start = date(2025, 1, 1)
    examples = [
        _training_example(idx, label=float(idx), signal_date=start + timedelta(days=idx))
        for idx in range(120)
    ]

    metrics = _cross_validate(
        examples,
        horizon_sessions=1,
        embargo_sessions=1,
        n_splits=2,
        max_iter=1,
        random_state=7,
    )

    assert metrics["pooled_score_basis"] == "fold_normalized_prediction_percentile"
    assert (
        metrics["rank_ic_score_basis"]
        == "fold_normalized_prediction_percentile_midrank"
    )
    assert metrics["raw_pooled_score_basis"] == "raw_prediction"
    assert metrics["raw_pooled_value_ic"] < -0.8
    assert metrics["raw_pooled_pearson"] == pytest.approx(
        metrics["raw_pooled_value_ic"]
    )
    assert metrics["raw_pooled_rank_ic"] < -0.45
    assert metrics["value_ic"] > 0.45
    assert metrics["pearson"] == pytest.approx(metrics["value_ic"])
    assert metrics["rank_ic"] > 0.45
    assert metrics["top_decile_mean_label"] > 80.0
    assert metrics["top_decile_lift"] > 1.0


def test_cross_validate_drops_one_row_train_fold_from_pooled_oos(monkeypatch):
    train_model_module = import_module("alpha.jobs.train_model")
    monkeypatch.setattr(
        train_model_module,
        "_new_gbrt_model",
        lambda **_kwargs: FoldOffsetFitModel(0.0),
    )
    start = date(2025, 1, 1)
    examples = [
        _training_example(
            idx,
            label=float(idx + 1),
            signal_date=start + timedelta(days=idx),
        )
        for idx in range(60)
    ]

    def forced_splits(*_args, **_kwargs):
        return [
            PurgedEmbargoedFold(
                train_indices=[0],
                test_indices=list(range(1, 20)),
                test_start_date=examples[1].signal_date,
                test_end_date=examples[19].signal_date,
                embargo_sessions=0,
                horizon_sessions=0,
            ),
            PurgedEmbargoedFold(
                train_indices=list(range(10)),
                test_indices=list(range(20, 60)),
                test_start_date=examples[20].signal_date,
                test_end_date=examples[59].signal_date,
                embargo_sessions=0,
                horizon_sessions=0,
            ),
        ]

    monkeypatch.setattr(
        train_model_module,
        "purged_embargoed_walk_forward_splits",
        forced_splits,
    )

    metrics = _cross_validate(
        examples,
        horizon_sessions=0,
        embargo_sessions=0,
        n_splits=2,
        max_iter=1,
        random_state=7,
        model_params={"early_stopping": True},
    )

    assert metrics["dropped_nonviable_fold_count"] == 1
    assert metrics["dropped_nonviable_folds"][0]["train_count"] == 1
    assert metrics["oos_count"] == 40
    assert len(metrics["folds"]) == 1
    assert metrics["folds"][0]["early_stopping_disabled_for_fold"] is True
    assert metrics["top_decile_lift"] == pytest.approx(
        metrics["folds"][0]["top_decile_lift"]
    )


def test_top_quantile_mean_returns_nan_when_tie_crosses_cutoff():
    labels = [float(idx) for idx in range(100)]
    preds = [0.0 for _ in range(75)] + [1.0 for _ in range(10)] + [2.0 for _ in range(15)]

    assert math.isnan(_top_quantile_mean(labels, preds))


def test_top_quantile_unreliable_is_independent_of_top_decile_boundary():
    labels = [float(idx) for idx in range(100)]
    preds = (
        [0.0 for _ in range(75)]
        + [1.0 for _ in range(10)]
        + [float(2 + idx) for idx in range(15)]
    )

    metrics = _prediction_quality_metrics(labels, preds)

    assert _top_quantile_unreliable(preds) is True
    assert math.isnan(_top_quantile_mean(labels, preds))
    assert metrics["top_decile_unreliable"] is False
    assert math.isfinite(metrics["top_decile_mean_label"])


def test_per_decile_curve_fails_closed_when_lower_boundary_splits_tie():
    labels = [float(idx) for idx in range(100)]
    preds = [0.0 for _ in range(20)] + [float(idx) for idx in range(20, 100)]

    metrics = _prediction_quality_metrics(labels, preds)

    assert metrics["top_decile_unreliable"] is False
    assert all(math.isnan(value) for value in metrics["per_decile_mean_label"])


def test_prediction_quality_metrics_signed_negative_baseline_has_no_lift_ratio():
    labels = [float(idx) for idx in range(-100, 0)]
    preds = list(labels)

    metrics = _prediction_quality_metrics(labels, preds)

    assert metrics["population_mean_label"] < 0.0
    assert math.isnan(metrics["top_decile_lift"])
    assert metrics["top_decile_lift_unreliable"] is True
    assert metrics["top_decile_spread"] > 0.0
    assert metrics["top_decile_win_rate"] == pytest.approx(0.0)


def test_prediction_quality_metrics_random_distinct_ranker_lift_is_near_baseline():
    rng = random.Random(17)
    labels = [0.5 + rng.random() for _ in range(1000)]
    preds = list(range(1000))
    rng.shuffle(preds)
    preds = [float(value) for value in preds]

    metrics = _prediction_quality_metrics(labels, preds)

    assert abs(metrics["rank_ic"]) < 0.1
    assert metrics["top_decile_lift"] == pytest.approx(1.0, abs=0.15)


def test_fold_normalized_percentiles_prevent_pooled_score_offset_bias():
    fold1_labels = [0.0 for _ in range(50)]
    fold2_labels = [10.0 for _ in range(50)]
    fold1_preds = [1000.0 + idx for idx in range(50)]
    fold2_preds = [float(idx) for idx in range(50)]
    labels = fold1_labels + fold2_labels
    raw_preds = fold1_preds + fold2_preds
    normalized_preds = _score_percentiles(fold1_preds) + _score_percentiles(fold2_preds)

    raw_metrics = _prediction_quality_metrics(labels, raw_preds)
    normalized_metrics = _prediction_quality_metrics(labels, normalized_preds)

    assert raw_metrics["top_decile_mean_label"] == pytest.approx(0.0)
    assert normalized_metrics["top_decile_mean_label"] == pytest.approx(5.0)
    assert normalized_metrics["top_decile_lift"] == pytest.approx(1.0)


def test_prediction_quality_metrics_emit_weighted_edge_variants():
    labels = [float(idx + 1) for idx in range(100)]
    preds = [float(idx) for idx in range(100)]
    weights = [1.0 for _ in range(100)]

    metrics = _prediction_quality_metrics(labels, preds, weights=weights)

    assert metrics["weighted_top_decile_weight_share"] == pytest.approx(0.1)
    assert metrics["weighted_population_mean_label"] == pytest.approx(
        metrics["population_mean_label"]
    )
    assert metrics["weighted_top_decile_lift"] == pytest.approx(metrics["top_decile_lift"])
    assert len(metrics["weighted_per_decile_mean_label"]) == 10


@pytest.mark.parametrize("row_count", [99, 101, 109, 111])
def test_weighted_deciles_balance_equal_weights_for_non_multiple_sizes(row_count):
    labels = [float(idx + 1) for idx in range(row_count)]
    preds = [float(idx) for idx in range(row_count)]
    weights = [1.0 for _ in range(row_count)]

    metrics = _prediction_quality_metrics(labels, preds, weights=weights)

    assert metrics["weighted_top_decile_unreliable"] is False
    assert metrics["weighted_per_decile_unreliable"] is False
    assert metrics["weighted_top_decile_weight_share"] == pytest.approx(0.1, abs=0.015)
    assert math.isfinite(metrics["weighted_top_decile_mean_label"])
    assert math.isfinite(metrics["weighted_top_decile_lift"])
    assert all(math.isfinite(value) for value in metrics["weighted_per_decile_mean_label"])


def test_cluster_like_weights_do_not_report_empty_reliable_top_decile():
    labels = [float(idx) for idx in range(100)]
    preds = [float(idx) for idx in range(100)]
    weights = [1.0 / 91.0 for _ in range(91)] + [1.0 for _ in range(9)]

    metrics = _prediction_quality_metrics(labels, preds, weights=weights)

    if metrics["weighted_top_decile_unreliable"]:
        assert math.isnan(metrics["weighted_top_decile_weight_share"])
        assert math.isnan(metrics["weighted_top_decile_mean_label"])
    else:
        assert math.isfinite(metrics["weighted_top_decile_weight_share"])
        assert metrics["weighted_top_decile_weight_share"] > 0.05
        assert math.isfinite(metrics["weighted_top_decile_mean_label"])


def test_weighted_deciles_do_not_fractionally_represent_single_heavy_row():
    labels = [float(idx) for idx in range(99)] + [999.0]
    preds = [float(idx) for idx in range(100)]
    weights = [1.0 for _ in range(99)] + [100.0]

    weighted_buckets = _prediction_weight_bins(labels, preds, weights, bins=10)
    heavy_bucket_count = sum(
        1 for bucket in weighted_buckets if any(label == 999.0 for label, _weight in bucket)
    )
    metrics = _prediction_quality_metrics(labels, preds, weights=weights)

    assert heavy_bucket_count == 1
    assert metrics["weighted_top_decile_unreliable"] is True
    assert math.isnan(metrics["weighted_top_decile_weight_share"])
    assert all(math.isnan(value) for value in metrics["weighted_per_decile_mean_label"])
    assert math.isnan(metrics["weighted_top_decile_lift"])
    assert math.isnan(metrics["weighted_top_decile_mean_label"])


def test_weighted_near_tie_cutoff_uses_same_tolerance_as_unweighted():
    labels = [float(idx + 1) for idx in range(100)]
    preds = [float(idx) for idx in range(100)]
    preds[90] = preds[89] + (PREDICTION_VARIANCE_EPSILON / 2.0)
    weights = [1.0 for _ in labels]

    metrics = _prediction_quality_metrics(labels, preds, weights=weights)

    assert metrics["top_decile_unreliable"] is True
    assert metrics["weighted_top_decile_unreliable"] is True
    assert math.isnan(metrics["weighted_top_decile_weight_share"])
    assert math.isnan(metrics["weighted_top_decile_mean_label"])
    assert math.isnan(metrics["weighted_top_decile_lift"])


def test_weighted_chained_near_tie_cutoff_matches_unweighted_reliability():
    labels = [float(idx + 1) for idx in range(100)]
    preds = [
        idx * PREDICTION_VARIANCE_EPSILON * 0.6
        for idx in range(100)
    ]
    weights = [1.0 for _ in labels]

    metrics = _prediction_quality_metrics(labels, preds, weights=weights)

    assert metrics["top_decile_unreliable"] is True
    assert metrics["weighted_top_decile_unreliable"] is True
    assert math.isnan(metrics["weighted_top_decile_weight_share"])
    assert math.isnan(metrics["weighted_top_decile_mean_label"])


def test_mean_fold_decile_curves_fail_closed_when_any_fold_unreliable():
    finite_deciles = [float(idx) for idx in range(10)]
    fold_metrics = [
        {
            "per_decile_mean_label": [math.nan for _ in range(10)],
            "per_decile_unreliable": True,
            "weighted_per_decile_mean_label": [math.nan for _ in range(10)],
            "weighted_per_decile_unreliable": True,
        },
        {
            "per_decile_mean_label": finite_deciles,
            "per_decile_unreliable": False,
            "weighted_per_decile_mean_label": finite_deciles,
            "weighted_per_decile_unreliable": False,
        },
    ]

    metrics = _mean_fold_metrics(fold_metrics)

    assert metrics["unreliable_per_decile_fold_count"] == 1
    assert metrics["unreliable_weighted_per_decile_fold_count"] == 1
    assert all(math.isnan(value) for value in metrics["per_decile_mean_label"])
    assert all(math.isnan(value) for value in metrics["weighted_per_decile_mean_label"])


def test_mean_fold_top_scalars_fail_closed_when_any_fold_unreliable():
    fold_metrics = [
        {
            "top_decile_unreliable": True,
            "top_quantile_label_mean": math.nan,
            "top_decile_lift": math.nan,
            "top_decile_spread": math.nan,
            "top_decile_mean_label": math.nan,
            "top_decile_median_label": math.nan,
            "top_decile_win_rate": math.nan,
            "bottom_9_deciles_mean_label": math.nan,
        },
        {
            "top_decile_unreliable": False,
            "top_quantile_label_mean": 4.0,
            "top_decile_lift": 1.5,
            "top_decile_spread": 2.0,
            "top_decile_mean_label": 5.0,
            "top_decile_median_label": 5.0,
            "top_decile_win_rate": 0.8,
            "bottom_9_deciles_mean_label": 3.0,
        },
    ]

    metrics = _mean_fold_metrics(fold_metrics)

    assert metrics["unreliable_top_decile_fold_count"] == 1
    for field in (
        "top_decile_lift",
        "top_decile_spread",
        "top_decile_mean_label",
        "top_decile_median_label",
        "top_decile_win_rate",
        "bottom_9_deciles_mean_label",
    ):
        assert math.isnan(metrics[field])


def test_mean_fold_keeps_strict_decile_separate_from_quintile_fallback():
    fold_metrics = [
        {
            "top_decile_too_small": True,
            "top_decile_unreliable": True,
            "top_decile_lift": math.nan,
            "top_decile_spread": math.nan,
            "top_decile_mean_label": math.nan,
            "top_decile_median_label": math.nan,
            "top_decile_win_rate": math.nan,
            "bottom_9_deciles_mean_label": math.nan,
            "top_effective_unreliable": False,
            "top_effective_lift": 2.0,
            "top_effective_spread": 3.0,
            "top_effective_mean_label": 6.0,
            "top_effective_win_rate": 1.0,
        },
        {
            "top_decile_too_small": False,
            "top_decile_unreliable": False,
            "top_decile_lift": 1.5,
            "top_decile_spread": 2.0,
            "top_decile_mean_label": 5.0,
            "top_decile_median_label": 5.0,
            "top_decile_win_rate": 0.8,
            "bottom_9_deciles_mean_label": 3.0,
            "top_effective_unreliable": False,
            "top_effective_lift": 1.5,
            "top_effective_spread": 2.0,
            "top_effective_mean_label": 5.0,
            "top_effective_win_rate": 0.8,
        },
    ]

    metrics = _mean_fold_metrics(fold_metrics)

    assert metrics["too_small_fold_count"] == 1
    assert metrics["unreliable_top_decile_fold_count"] == 1
    assert math.isnan(metrics["top_decile_lift"])
    assert math.isnan(metrics["top_decile_mean_label"])
    assert metrics["unreliable_top_effective_fold_count"] == 0
    assert metrics["top_effective_lift"] == pytest.approx(1.75)
    assert metrics["top_effective_mean_label"] == pytest.approx(5.5)


def test_mean_fold_top_quantile_fails_closed_independent_of_top_decile():
    fold_metrics = [
        {
            "top_quantile_label_mean": math.nan,
            "top_quantile_unreliable": True,
            "top_decile_unreliable": False,
            "top_decile_lift": 1.5,
        },
        {
            "top_quantile_label_mean": 4.0,
            "top_quantile_unreliable": False,
            "top_decile_unreliable": False,
            "top_decile_lift": 2.0,
        },
    ]

    metrics = _mean_fold_metrics(fold_metrics)

    assert metrics["unreliable_top_quantile_fold_count"] == 1
    assert math.isnan(metrics["top_quantile_label_mean"])
    assert metrics["unreliable_top_decile_fold_count"] == 0
    assert metrics["top_decile_lift"] == pytest.approx(1.75)


def test_mean_fold_lift_denominator_unreliability_is_visible():
    fold_metrics = [
        {
            "top_decile_unreliable": False,
            "top_decile_lift_unreliable": True,
            "top_decile_lift": math.nan,
            "top_decile_spread": 0.2,
        },
        {
            "top_decile_unreliable": False,
            "top_decile_lift_unreliable": False,
            "top_decile_lift": 2.0,
            "top_decile_spread": 0.4,
        },
    ]

    metrics = _mean_fold_metrics(fold_metrics)

    assert metrics["unreliable_lift_fold_count"] == 1
    assert math.isnan(metrics["top_decile_lift"])
    assert metrics["top_decile_spread"] == pytest.approx(0.3)


def test_mean_fold_weighted_top_scalars_fail_closed_when_any_fold_unreliable():
    fold_metrics = [
        {
            "weighted_top_decile_unreliable": True,
            "weighted_top_decile_lift": math.nan,
            "weighted_top_decile_spread": math.nan,
            "weighted_top_decile_mean_label": math.nan,
            "weighted_top_decile_win_rate": math.nan,
            "weighted_top_decile_weight_share": math.nan,
            "weighted_bottom_9_deciles_mean_label": math.nan,
        },
        {
            "weighted_top_decile_unreliable": False,
            "weighted_top_decile_lift": 1.5,
            "weighted_top_decile_spread": 2.0,
            "weighted_top_decile_mean_label": 5.0,
            "weighted_top_decile_win_rate": 0.8,
            "weighted_top_decile_weight_share": 0.1,
            "weighted_bottom_9_deciles_mean_label": 3.0,
        },
    ]

    metrics = _mean_fold_metrics(fold_metrics)

    assert metrics["unreliable_weighted_top_decile_fold_count"] == 1
    for field in (
        "weighted_top_decile_lift",
        "weighted_top_decile_spread",
        "weighted_top_decile_mean_label",
        "weighted_top_decile_win_rate",
        "weighted_top_decile_weight_share",
        "weighted_bottom_9_deciles_mean_label",
    ):
        assert math.isnan(metrics[field])


@pytest.mark.parametrize("row_count", [5, 9])
def test_small_row_count_full_decile_curve_fails_closed_but_top_fallback_works(row_count):
    labels = [float(idx + 1) for idx in range(row_count)]
    preds = [float(idx) for idx in range(row_count)]

    metrics = _prediction_quality_metrics(labels, preds)

    assert metrics["per_decile_unreliable"] is True
    assert all(math.isnan(value) for value in metrics["per_decile_mean_label"])
    assert metrics["top_decile_too_small"] is True
    assert metrics["top_decile_effective_bins"] == 5
    assert metrics["top_effective_bins"] == 5
    assert metrics["top_effective_bucket_name"] == "top_quintile"
    assert math.isnan(metrics["top_decile_mean_label"])
    assert math.isnan(metrics["top_decile_lift"])
    assert math.isfinite(metrics["top_effective_mean_label"])
    assert math.isfinite(metrics["top_effective_lift"])


def test_weighted_tiny_full_decile_curve_fails_closed_but_top_fallback_is_sensible():
    labels = [float(idx + 1) for idx in range(5)]
    preds = [float(idx) for idx in range(5)]
    weights = [1.0 for _ in labels]

    metrics = _prediction_quality_metrics(labels, preds, weights=weights)

    assert metrics["weighted_per_decile_unreliable"] is True
    assert all(math.isnan(value) for value in metrics["weighted_per_decile_mean_label"])
    assert metrics["weighted_top_decile_unreliable"] is True
    assert math.isnan(metrics["weighted_top_decile_weight_share"])
    assert math.isnan(metrics["weighted_top_decile_mean_label"])
    assert metrics["weighted_top_effective_weight_share"] == pytest.approx(0.2)
    assert math.isfinite(metrics["weighted_top_effective_mean_label"])


def test_prediction_quality_metrics_tiny_fold_falls_back_without_crashing():
    labels = [float(idx) for idx in range(12)]
    preds = list(labels)

    metrics = _prediction_quality_metrics(labels, preds)

    assert metrics["top_decile_too_small"] is True
    assert metrics["top_decile_fallback"] == "quintile"
    assert metrics["top_decile_effective_bins"] == 5
    assert metrics["top_decile_effective_quantile"] == pytest.approx(0.2)
    assert metrics["top_effective_bins"] == 5
    assert metrics["top_effective_quantile"] == pytest.approx(0.2)
    assert metrics["top_effective_bucket_name"] == "top_quintile"
    assert len(metrics["per_decile_mean_label"]) == 10
    assert math.isnan(metrics["top_decile_lift"])
    assert math.isnan(metrics["top_decile_mean_label"])
    assert math.isfinite(metrics["top_effective_lift"])
    assert math.isfinite(metrics["top_effective_mean_label"])


def test_train_model_end_to_end_writes_artifact_registry_and_shadow_score(
    db_session, tmp_path
):
    signal_ids = _seed_training_corpus(db_session, count=160)
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
    assert "value_ic" in metrics["per_pattern"]["M4"]
    assert "rank_ic" in metrics["per_pattern"]["M4"]
    assert "top_decile_lift" in metrics["per_pattern"]["M4"]
    assert "top_effective_lift" in metrics["per_pattern"]["M4"]
    assert "per_decile_mean_label" in metrics["per_pattern"]["M4"]
    assert "weighted_top_decile_lift" in metrics["per_pattern"]["M4"]
    assert "raw_pooled_value_ic" in metrics["per_pattern"]["M4"]
    assert metrics["per_pattern"]["M4"]["pooled_score_basis"] == "fold_normalized_prediction_percentile"
    assert metrics["per_pattern"]["M4"]["raw_pooled_score_basis"] == "raw_prediction"
    assert "mean_fold_metrics" in metrics["per_pattern"]["M4"]
    assert "top_decile_lift" in metrics["per_pattern"]["M4"]["folds"][0]
    assert "top_effective_lift" in metrics["per_pattern"]["M4"]["folds"][0]
    assert "weighted_top_decile_lift" in metrics["per_pattern"]["M4"]["folds"][0]
    assert metrics["per_pattern"]["M4"]["training_selection"]["kept_row_count"] == 160
    assert (
        metrics["per_pattern"]["M4"]["training_selection"]["dropped_non_finite"]
        == 0
    )
    loaded_examples = _load_training_examples(
        db_session,
        pattern=manifest.pattern("M4"),
    )
    assert sum(
        1
        for row in loaded_examples
        if row.security_identity == "cik:0007777777"
    ) == 2
    assert metrics["per_pattern"]["M4"]["oos_quality_gate"]["passed"] is True
    assert metrics["per_pattern"]["M4"]["model_params"]["min_samples_leaf"] == 2
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


def test_train_model_rejected_when_pooled_oos_gate_fails(
    db_session,
    tmp_path,
    monkeypatch,
):
    train_model_module = import_module("alpha.jobs.train_model")

    def bad_cv(*_args, **_kwargs):
        return {
            "cv_type": "purged_embargoed_walk_forward",
            "random_kfold_forbidden": True,
            "top_decile_lift": 0.75,
            "rank_ic": 0.25,
            "top_decile_unreliable": False,
            "top_decile_too_small": False,
            "top_decile_lift_unreliable": False,
        }

    monkeypatch.setattr(train_model_module, "_cross_validate", bad_cv)
    signal_ids = _seed_training_corpus(db_session, count=20)
    manifest = load_manifest(_write_manifest(tmp_path, signal_ids))
    job = Stage1TrainModelJob(
        session=db_session,
        manifest=manifest,
        pattern_id="M4",
        artifact_dir=tmp_path / "artifacts",
        n_splits=2,
        max_iter=3,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    model = db_session.query(MLModelRegistry).one()
    metrics = json.loads(model.cv_metrics_json)["per_pattern"]["M4"]
    assert model.status == "rejected"
    assert metrics["oos_quality_gate"]["passed"] is False
    assert metrics["oos_quality_gate"]["failures"][0]["metric"] == "top_decile_lift"


def test_cv_and_final_fit_use_identical_resolved_model_params(
    db_session,
    tmp_path,
    monkeypatch,
):
    train_model_module = import_module("alpha.jobs.train_model")
    captured_params: list[dict] = []

    def fake_new_gbrt_model(*, max_iter, random_state, model_params=None):
        captured_params.append(dict(model_params or {}))
        return ConstantFitModel()

    monkeypatch.setattr(
        train_model_module,
        "_new_gbrt_model",
        fake_new_gbrt_model,
    )
    signal_ids = _seed_training_corpus(db_session, count=40)
    payload = _manifest_payload(
        signal_ids,
        model_params={
            "min_samples_leaf": 7,
            "l2_regularization": 0.2,
            "early_stopping": False,
        },
    )
    path = tmp_path / "model-param-parity.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    manifest = load_manifest(path)
    job = Stage1TrainModelJob(
        session=db_session,
        manifest=manifest,
        pattern_id="M4",
        artifact_dir=tmp_path / "artifacts",
        n_splits=2,
        max_iter=3,
    )

    result = run_job(db_session, job, params={"test": True})

    assert result.ok
    assert len(captured_params) >= 3
    assert all(params == captured_params[-1] for params in captured_params)
    assert captured_params[-1] == {
        "min_samples_leaf": 7,
        "l2_regularization": 0.2,
        "early_stopping": False,
    }


def test_default_model_params_raise_min_samples_leaf():
    pattern = PatternManifest(
        pattern_id="M4",
        signal_horizon="2d",
        min_graded_cohorts=1,
        embargo_sessions=2,
        feature_schema=_feature_schema(),
        label={"field": "forward_return"},
        selection={
            "source": "forward_return_observations",
            "statuses": ["computed"],
            "horizon_sessions": 2,
            "signal_ids": [],
        },
        diagnostics={},
    )

    params = _resolved_model_params(pattern)

    assert DEFAULT_MIN_SAMPLES_LEAF != 1
    assert params["min_samples_leaf"] == DEFAULT_MIN_SAMPLES_LEAF
    assert params["l2_regularization"] > 0.0
    assert params["early_stopping"] is True


def test_model_params_allow_small_corpus_to_disable_early_stopping():
    pattern = PatternManifest(
        pattern_id="M4",
        signal_horizon="2d",
        min_graded_cohorts=1,
        embargo_sessions=2,
        feature_schema=_feature_schema(),
        label={"field": "forward_return"},
        selection={
            "source": "forward_return_observations",
            "statuses": ["computed"],
            "horizon_sessions": 2,
            "signal_ids": [],
        },
        diagnostics={},
        model_params={"early_stopping": False},
    )

    params = _resolved_model_params(pattern)

    assert params["early_stopping"] is False


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


def test_manifest_loader_accepts_stricter_oos_quality_gate_override(tmp_path):
    payload = _manifest_payload(
        ["sig-1"],
        oos_quality_gate={
            "min_top_decile_lift": 1.25,
            "min_rank_ic": 0.05,
            "required_metrics": [
                "top_decile_lift",
                "rank_ic",
                "weighted_top_decile_lift",
            ],
            "reject_status": "rejected",
        },
    )
    path = tmp_path / "valid-oos-gate.json"
    path.write_text(json.dumps(payload, sort_keys=True))

    manifest = load_manifest(path)

    gate = manifest.pattern("M4").oos_quality_gate
    assert gate["min_top_decile_lift"] == pytest.approx(1.25)
    assert gate["min_rank_ic"] == pytest.approx(0.05)
    assert "top_decile_lift" in gate["required_metrics"]
    assert "rank_ic" in gate["required_metrics"]


def test_train_model_cli_sets_long_db_timeout_env():
    original_statement_timeout = os.environ.pop(
        "ALPHA_DB_STATEMENT_TIMEOUT_MS",
        None,
    )
    original_idle_timeout = os.environ.pop(
        "ALPHA_DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS",
        None,
    )
    args = _parse_args([
        "--manifest-path",
        "manifest.json",
        "--pattern-id",
        "M4",
        "--artifact-dir",
        "artifacts",
        "--db-statement-timeout-ms",
        "7200000",
        "--db-idle-in-transaction-timeout-ms",
        "7300000",
    ])

    _apply_trainer_db_timeout_env(args)

    try:
        assert args.db_statement_timeout_ms == 7_200_000
        assert args.db_idle_in_transaction_timeout_ms == 7_300_000
        assert DEFAULT_TRAINER_DB_TIMEOUT_MS > 300_000
        assert os.environ["ALPHA_DB_STATEMENT_TIMEOUT_MS"] == "7200000"
        assert (
            os.environ["ALPHA_DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS"]
            == "7300000"
        )
    finally:
        if original_statement_timeout is None:
            os.environ.pop("ALPHA_DB_STATEMENT_TIMEOUT_MS", None)
        else:
            os.environ["ALPHA_DB_STATEMENT_TIMEOUT_MS"] = original_statement_timeout
        if original_idle_timeout is None:
            os.environ.pop("ALPHA_DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS", None)
        else:
            os.environ[
                "ALPHA_DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS"
            ] = original_idle_timeout


@pytest.mark.parametrize(
    "oos_quality_gate, message",
    [
        ({"min_top_decile_lift": 0.99}, "min_top_decile_lift"),
        ({"min_top_decile_lift": math.nan}, "min_top_decile_lift"),
        ({"min_top_decile_lift": "nan"}, "min_top_decile_lift"),
        ({"min_top_decile_lift": float("inf")}, "min_top_decile_lift"),
        ({"min_top_decile_lift": float("-inf")}, "min_top_decile_lift"),
        ({"min_rank_ic": -0.01}, "min_rank_ic"),
        ({"min_rank_ic": math.nan}, "min_rank_ic"),
        ({"min_rank_ic": "nan"}, "min_rank_ic"),
        ({"min_rank_ic": float("inf")}, "min_rank_ic"),
        ({"min_rank_ic": float("-inf")}, "min_rank_ic"),
        ({"required_metrics": ["rank_ic"]}, "required_metrics"),
        ({"required_metrics": ["top_decile_lift"]}, "required_metrics"),
        ({"reject_status": "shadow"}, "reject_status"),
        ({"reject_status": "active"}, "reject_status"),
    ],
)
def test_manifest_loader_rejects_fail_open_oos_quality_gate_overrides(
    tmp_path,
    oos_quality_gate,
    message,
):
    payload = _manifest_payload(["sig-1"], oos_quality_gate=oos_quality_gate)
    path = tmp_path / f"bad-oos-gate-{message}.json"
    path.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(Exception, match=message):
        load_manifest(path)


def test_training_loader_filters_to_manifest_signal_horizon(db_session, tmp_path):
    sig_2d = _seed_signal(db_session, idx=20, signal_horizon="2d")
    sig_5d = _seed_signal(db_session, idx=21, signal_horizon="5d")
    db_session.commit()
    manifest = load_manifest(_write_manifest(tmp_path, [sig_2d, sig_5d]))

    examples = _load_training_examples(db_session, pattern=manifest.pattern("M4"))

    assert [row.signal_id for row in examples] == [sig_2d]


def test_training_loader_fails_closed_on_pit_failed_rows(db_session, tmp_path):
    signal_id = _seed_signal(db_session, idx=120, point_in_time_passed=False)
    db_session.commit()
    manifest = load_manifest(_write_manifest(tmp_path, [signal_id]))

    with pytest.raises(RuntimeError, match="pit_failed_row_count=1"):
        _load_training_examples(db_session, pattern=manifest.pattern("M4"))


def test_training_loader_allows_deferred_pit_only_with_manifest_opt_in(
    db_session,
    tmp_path,
    monkeypatch,
):
    train_model_module = import_module("alpha.jobs.train_model")
    warnings: list[str] = []
    monkeypatch.setattr(
        train_model_module.LOGGER,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )
    signal_id = _seed_signal(db_session, idx=121, point_in_time_passed=False)
    payload = _manifest_payload([signal_id], allow_deferred_pit=True)
    path = tmp_path / "deferred-pit.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    db_session.commit()
    manifest = load_manifest(path)

    examples, metrics = _load_training_examples(
        db_session,
        pattern=manifest.pattern("M4"),
        return_metrics=True,
    )

    assert [row.signal_id for row in examples] == [signal_id]
    assert metrics["pit_deferred"] is True
    assert metrics["pit_failed_row_count"] == 1
    assert any("allow_deferred_pit=true" in message for message in warnings)


def test_training_loader_excludes_non_active_or_uncomputed_registry_rows(
    db_session,
    tmp_path,
):
    active_id = _seed_signal(db_session, idx=122)
    excluded_id = _seed_signal(db_session, idx=123, signal_status="excluded")
    pending_id = _seed_signal(
        db_session,
        idx=124,
        forward_return_status="pending",
    )
    lookahead_failed_id = _seed_signal(
        db_session,
        idx=125,
        lookahead_guard_passed=False,
    )
    db_session.commit()
    manifest = load_manifest(
        _write_manifest(
            tmp_path,
            [active_id, excluded_id, pending_id, lookahead_failed_id],
        )
    )

    examples, metrics = _load_training_examples(
        db_session,
        pattern=manifest.pattern("M4"),
        return_metrics=True,
    )

    assert [row.signal_id for row in examples] == [active_id]
    assert metrics["hard_filter_drop_counts"] == {
        "signal_status_active": 1,
        "lookahead_guard_passed": 1,
        "forward_return_status_computed": 1,
        "point_in_time_passed": 0,
    }


def test_training_loader_collapses_same_security_same_date_rename_pair(
    db_session,
    tmp_path,
):
    _seed_rename_identity_pair(db_session)
    signal_date = date(2025, 2, 3)
    cep_id = _seed_signal(
        db_session,
        idx=140,
        ticker="CEP",
        signal_date=signal_date,
        forward_return=0.01,
    )
    xxi_id = _seed_signal(
        db_session,
        idx=141,
        ticker="XXI",
        signal_date=signal_date,
        forward_return=0.08,
    )
    db_session.commit()
    manifest = load_manifest(_write_manifest(tmp_path, [cep_id, xxi_id]))

    examples, metrics = _load_training_examples(
        db_session,
        pattern=manifest.pattern("M4"),
        return_metrics=True,
    )

    assert [row.signal_id for row in examples] == [xxi_id]
    assert examples[0].ticker == "XXI"
    assert examples[0].security_identity == "cik:0001234567"
    assert metrics["security_identity_duplicate_rows_dropped"] == 1


def test_training_loader_does_not_collapse_dual_share_classes(
    db_session,
    tmp_path,
):
    scan = UniverseScan(
        scan_id="scan-dual-class",
        trading_date="2025-02-04",
        asof_timestamp=datetime(2025, 2, 4, tzinfo=timezone.utc),
        provider="test",
        raw_count=2,
        deduped_count=2,
        duplicate_symbol_count=0,
        included_count=2,
        excluded_count=0,
        run_status="finished",
    )
    db_session.add(scan)
    for ticker, share_figi in (("GOOG", "BBG009S3NB30"), ("GOOGL", "BBG009S39JX6")):
        db_session.add(
            SecurityIdentitySnapshot(
                security_identity_snapshot_id=f"identity-{ticker.lower()}",
                scan_id=scan.scan_id,
                ticker=ticker,
                cik="0001652044",
                composite_figi=f"COMP-{ticker}",
                share_class_figi=share_figi,
                active=True,
                identity_status="present",
                source_provider="test",
                asof_timestamp=datetime(2025, 2, 4, tzinfo=timezone.utc),
            )
        )
    signal_date = date(2025, 2, 4)
    goog_id = _seed_signal(db_session, idx=142, ticker="GOOG", signal_date=signal_date)
    googl_id = _seed_signal(db_session, idx=143, ticker="GOOGL", signal_date=signal_date)
    db_session.commit()
    manifest = load_manifest(_write_manifest(tmp_path, [goog_id, googl_id]))

    examples, metrics = _load_training_examples(
        db_session,
        pattern=manifest.pattern("M4"),
        return_metrics=True,
    )

    assert {row.signal_id for row in examples} == {goog_id, googl_id}
    assert {row.security_identity for row in examples} == {
        "share_class_figi:BBG009S3NB30",
        "share_class_figi:BBG009S39JX6",
    }
    assert metrics["security_identity_duplicate_rows_dropped"] == 0


def test_security_identity_aliases_do_not_hijack_direct_distinct_tickers(
    db_session,
    tmp_path,
):
    scan = UniverseScan(
        scan_id="scan-recycled-alias",
        trading_date="2025-02-05",
        asof_timestamp=datetime(2025, 2, 5, tzinfo=timezone.utc),
        provider="test",
        raw_count=2,
        deduped_count=2,
        duplicate_symbol_count=0,
        included_count=2,
        excluded_count=0,
        run_status="finished",
    )
    db_session.add(scan)
    db_session.add_all([
        SecurityIdentitySnapshot(
            security_identity_snapshot_id="identity-foo-live",
            scan_id=scan.scan_id,
            ticker="FOO",
            cik="0000000001",
            active=True,
            ticker_events_json=json.dumps([{"old_ticker": "BAR", "new_ticker": "FOO"}]),
            identity_status="present",
            source_provider="test",
            asof_timestamp=datetime(2025, 2, 5, tzinfo=timezone.utc),
        ),
        SecurityIdentitySnapshot(
            security_identity_snapshot_id="identity-bar-dead",
            scan_id=scan.scan_id,
            ticker="BAR",
            cik="0000000002",
            active=False,
            ticker_events_json=json.dumps([{"old_ticker": "FOO", "new_ticker": "BAR"}]),
            identity_status="present",
            source_provider="test",
            asof_timestamp=datetime(2025, 2, 1, tzinfo=timezone.utc),
        ),
    ])
    signal_date = date(2025, 2, 5)
    foo_id = _seed_signal(db_session, idx=144, ticker="FOO", signal_date=signal_date)
    bar_id = _seed_signal(db_session, idx=145, ticker="BAR", signal_date=signal_date)
    db_session.commit()
    manifest = load_manifest(_write_manifest(tmp_path, [foo_id, bar_id]))

    resolved = resolve_security_identities_for_tickers(db_session, ["FOO", "BAR"])
    examples, metrics = _load_training_examples(
        db_session,
        pattern=manifest.pattern("M4"),
        return_metrics=True,
    )

    assert resolved["FOO"].security_identity == "cik:0000000001"
    assert resolved["BAR"].security_identity == "cik:0000000002"
    assert {row.signal_id for row in examples} == {foo_id, bar_id}
    assert metrics["security_identity_duplicate_rows_dropped"] == 0


def test_sparse_identity_canonical_ticker_is_deterministic_when_rows_shuffle():
    first = SecurityIdentitySnapshot(
        ticker="SPARSEA",
        identity_hash="sparse-hash",
        active=True,
        asof_timestamp=datetime(2025, 2, 6, tzinfo=timezone.utc),
    )
    second = SecurityIdentitySnapshot(
        ticker="SPARSEB",
        identity_hash="sparse-hash",
        active=True,
        asof_timestamp=datetime(2025, 2, 6, tzinfo=timezone.utc),
    )

    assert _canonical_ticker([first, second]) == "SPARSEB"
    assert _canonical_ticker([second, first]) == "SPARSEB"


def test_training_loader_dedups_after_non_finite_feature_qualification(
    db_session,
    tmp_path,
):
    _seed_rename_identity_pair(db_session)
    signal_date = date(2025, 2, 3)
    cep_id = _seed_signal(
        db_session,
        idx=146,
        ticker="CEP",
        signal_date=signal_date,
        forward_return=0.02,
    )
    xxi_id = _seed_signal(
        db_session,
        idx=147,
        ticker="XXI",
        signal_date=signal_date,
        forward_return=0.03,
    )
    corrupt_snapshot = db_session.get(FeatureSnapshot, "fs-147")
    corrupt_json = json.loads(corrupt_snapshot.feature_json)
    corrupt_json["signal_context"]["mom20"] = "not-a-number"
    corrupt_json["statuses"]["mom20"] = "computed"
    corrupt_snapshot.feature_json = json.dumps(corrupt_json, sort_keys=True)
    db_session.commit()
    manifest = load_manifest(_write_manifest(tmp_path, [cep_id, xxi_id]))

    examples, metrics = _load_training_examples(
        db_session,
        pattern=manifest.pattern("M4"),
        return_metrics=True,
    )

    assert [row.signal_id for row in examples] == [cep_id]
    assert examples[0].security_identity == "cik:0001234567"
    assert metrics["dropped_non_finite"] == 1
    assert metrics["security_identity_duplicate_rows_dropped"] == 0


def test_training_loader_signs_short_labels_by_direction(db_session, tmp_path):
    signal_id = _seed_signal(
        db_session,
        idx=126,
        direction="short",
        forward_return=0.12,
    )
    payload = _manifest_payload([signal_id], direction="short")
    path = tmp_path / "short-manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    db_session.commit()
    manifest = load_manifest(path)

    examples = _load_training_examples(db_session, pattern=manifest.pattern("M4"))

    assert len(examples) == 1
    assert examples[0].raw_label == pytest.approx(0.12)
    assert examples[0].label == pytest.approx(-0.12)
    assert examples[0].direction == "short"


def test_training_loader_rejects_mixed_direction_cohort(db_session):
    long_id = _seed_signal(db_session, idx=127, direction="long")
    short_id = _seed_signal(db_session, idx=128, direction="short")
    db_session.commit()
    pattern = PatternManifest(
        pattern_id="M4",
        signal_horizon="2d",
        min_graded_cohorts=1,
        embargo_sessions=2,
        feature_schema=_feature_schema(),
        label={"field": "forward_return"},
        selection={
            "source": "forward_return_observations",
            "statuses": ["computed"],
            "horizon_sessions": 2,
            "signal_ids": [long_id, short_id],
        },
        diagnostics={},
        direction="long",
    )

    with pytest.raises(RuntimeError, match="mixed directions"):
        _load_training_examples(db_session, pattern=pattern)


def test_training_loader_rejects_manifest_direction_mismatch(db_session, tmp_path):
    signal_id = _seed_signal(db_session, idx=129, direction="short")
    db_session.commit()
    manifest = load_manifest(_write_manifest(tmp_path, [signal_id]))

    with pytest.raises(RuntimeError, match="manifest direction"):
        _load_training_examples(db_session, pattern=manifest.pattern("M4"))


def test_training_loader_accepts_legitimate_realized_label_trading_session_window(
    db_session,
    tmp_path,
):
    signal_date = date(2025, 2, 3)
    exit_date = nth_us_equity_session(signal_date + timedelta(days=1), 15)
    signal_id = _seed_signal(
        db_session,
        idx=130,
        signal_date=signal_date,
        entry_session_date=signal_date,
        exit_session_date=exit_date,
        signal_horizon="15d",
    )
    db_session.commit()
    payload = _manifest_payload(
        [signal_id],
        signal_horizon="15d",
        horizon_sessions=15,
    )
    path = tmp_path / "m4-15-session.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    manifest = load_manifest(path)

    examples, metrics = _load_training_examples(
        db_session,
        pattern=manifest.pattern("M4"),
        return_metrics=True,
    )

    assert [row.signal_id for row in examples] == [signal_id]
    assert examples[0].realized_label_window_sessions == 15
    assert metrics["max_realized_label_window_sessions"] == 15


def test_training_loader_accepts_weekend_holiday_calendar_span_within_sessions(
    db_session,
    tmp_path,
):
    entry = date(2025, 1, 17)
    exit_date = date(2025, 1, 21)
    signal_id = _seed_signal(
        db_session,
        idx=135,
        signal_date=entry,
        entry_session_date=entry,
        exit_session_date=exit_date,
        signal_horizon="1d",
    )
    db_session.commit()
    payload = _manifest_payload([signal_id], signal_horizon="1d", horizon_sessions=1)
    path = tmp_path / "holiday-overnight.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    manifest = load_manifest(path)

    examples, metrics = _load_training_examples(
        db_session,
        pattern=manifest.pattern("M4"),
        return_metrics=True,
    )

    assert [row.signal_id for row in examples] == [signal_id]
    assert examples[0].realized_label_window_sessions == 1
    assert metrics["max_realized_label_window_sessions"] == 1


def test_training_loader_fails_closed_on_missing_realized_label_dates(
    db_session,
    tmp_path,
):
    signal_id = _seed_signal(db_session, idx=136, missing_realized_dates=True)
    db_session.commit()
    manifest = load_manifest(_write_manifest(tmp_path, [signal_id]))

    with pytest.raises(RuntimeError, match="missing realized label session dates"):
        _load_training_examples(db_session, pattern=manifest.pattern("M4"))


def test_training_loader_rejects_non_session_realized_label_dates(
    db_session,
    tmp_path,
):
    signal_id = _seed_signal(
        db_session,
        idx=148,
        signal_date=date(2025, 1, 17),
        entry_session_date=date(2025, 1, 18),
        exit_session_date=date(2025, 1, 21),
    )
    db_session.commit()
    manifest = load_manifest(_write_manifest(tmp_path, [signal_id]))

    with pytest.raises(RuntimeError, match="non-session realized label dates"):
        _load_training_examples(db_session, pattern=manifest.pattern("M4"))


def test_training_loader_rejects_exit_session_before_entry_session(
    db_session,
    tmp_path,
):
    signal_id = _seed_signal(
        db_session,
        idx=149,
        signal_date=date(2025, 1, 6),
        entry_session_date=date(2025, 1, 6),
        exit_session_date=date(2025, 1, 3),
    )
    db_session.commit()
    manifest = load_manifest(_write_manifest(tmp_path, [signal_id]))

    with pytest.raises(RuntimeError, match="exit session before entry session"):
        _load_training_examples(db_session, pattern=manifest.pattern("M4"))


def test_training_loader_rejects_realized_label_window_beyond_purge_horizon(
    db_session,
    tmp_path,
):
    signal_date = date(2025, 2, 3)
    signal_id = _seed_signal(
        db_session,
        idx=137,
        signal_date=signal_date,
        entry_session_date=signal_date,
        exit_session_date=nth_us_equity_session(signal_date + timedelta(days=1), 3),
    )
    db_session.commit()
    manifest = load_manifest(_write_manifest(tmp_path, [signal_id]))

    with pytest.raises(RuntimeError, match="realized label window"):
        _load_training_examples(db_session, pattern=manifest.pattern("M4"))


def test_training_loader_rejects_non_finite_label(db_session, tmp_path):
    signal_id = _seed_signal(db_session, idx=25, forward_return=float("inf"))
    db_session.commit()
    manifest = load_manifest(_write_manifest(tmp_path, [signal_id]))

    with pytest.raises(RuntimeError, match="non-finite.*sig-25"):
        _load_training_examples(db_session, pattern=manifest.pattern("M4"))


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
            entry_session_date=old.entry_session_date,
            exit_session_date=old.exit_session_date,
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
    assert metrics["dropped_non_finite"] == 1
    assert metrics["dropped_non_finite_by_feature"] == {"mom20": 1}
    assert metrics["selected_row_count"] == 2
    assert metrics["kept_row_count"] == 1
    assert metrics["dropped_non_finite_fraction"] == pytest.approx(0.5)
    assert math.isfinite(metrics["kept_label_mean"])
    assert math.isfinite(metrics["dropped_non_finite_label_mean"])
    assert metrics["dropped_non_finite_selection_bias_flag"] is False


def test_training_loader_rejects_skewed_non_finite_drops_without_ack(
    db_session,
    tmp_path,
):
    corrupt_signal_id = _seed_signal(
        db_session,
        idx=131,
        forward_return=1.0,
    )
    kept_signal_id = _seed_signal(
        db_session,
        idx=132,
        forward_return=-1.0,
    )
    corrupt_snapshot = db_session.get(FeatureSnapshot, "fs-131")
    corrupt_json = json.loads(corrupt_snapshot.feature_json)
    corrupt_json["signal_context"]["mom20"] = "not-a-number"
    corrupt_json["statuses"]["mom20"] = "computed"
    corrupt_snapshot.feature_json = json.dumps(corrupt_json, sort_keys=True)
    db_session.commit()
    manifest = load_manifest(
        _write_manifest(tmp_path, [corrupt_signal_id, kept_signal_id])
    )

    with pytest.raises(RuntimeError, match="selection-biased non-finite"):
        _load_training_examples(
            db_session,
            pattern=manifest.pattern("M4"),
            return_metrics=True,
        )


def test_training_loader_records_skewed_non_finite_drops_with_ack(
    db_session,
    tmp_path,
):
    corrupt_signal_id = _seed_signal(
        db_session,
        idx=133,
        forward_return=1.0,
    )
    kept_signal_id = _seed_signal(
        db_session,
        idx=134,
        forward_return=-1.0,
    )
    corrupt_snapshot = db_session.get(FeatureSnapshot, "fs-133")
    corrupt_json = json.loads(corrupt_snapshot.feature_json)
    corrupt_json["signal_context"]["mom20"] = "not-a-number"
    corrupt_json["statuses"]["mom20"] = "computed"
    corrupt_snapshot.feature_json = json.dumps(corrupt_json, sort_keys=True)
    payload = _manifest_payload([corrupt_signal_id, kept_signal_id])
    payload["patterns"]["M4"]["selection"]["allow_skewed_non_finite_drops"] = True
    payload["manifest_sha256"] = manifest_payload_hash(payload)
    path = tmp_path / "skewed-drops-ack.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    db_session.commit()
    manifest = load_manifest(path)

    examples, metrics = _load_training_examples(
        db_session,
        pattern=manifest.pattern("M4"),
        return_metrics=True,
    )

    assert [row.signal_id for row in examples] == [kept_signal_id]
    assert metrics["dropped_non_finite_selection_bias_flag"] is True
    assert metrics["dropped_non_finite_label_mean_delta"] == pytest.approx(2.0)


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


def test_log1p_domain_violation_raises_feature_selection_error(db_session):
    signal_id = _seed_signal(db_session, idx=30, gap=-0.01)
    snapshot = db_session.get(FeatureSnapshot, "fs-30")
    feature_json = json.loads(snapshot.feature_json)
    feature_json["signal_context"]["mom20"] = -2.0
    feature_json["statuses"]["mom20"] = "computed"
    snapshot.feature_json = json.dumps(feature_json, sort_keys=True)
    schema = {
        "schema_version": "log1p_domain_guard_v1",
        "pattern_id": "M4",
        "pattern_clock": "eod",
        "fields": [
            {
                "name": "mom20",
                "source": "feature_snapshot_json",
                "path": "signal_context.mom20",
                "status_path": "statuses.mom20",
                "transform": "log1p",
            }
        ],
    }
    db_session.commit()

    with pytest.raises(FeatureSelectionError, match="log1p transform domain"):
        select_features(db_session, signal_id, schema)


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


def test_malformed_status_path_is_not_stored_in_missing_statuses(db_session):
    signal_id = _seed_signal(db_session, idx=36, gap=-0.01)
    snapshot = db_session.get(FeatureSnapshot, "fs-36")
    feature_json = json.loads(snapshot.feature_json)
    feature_json["signal_context"].pop("mom20")
    feature_json["statuses"] = []
    snapshot.feature_json = json.dumps(feature_json, sort_keys=True)
    schema = {
        "schema_version": "malformed_status_path_v1",
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
    db_session.commit()

    vector = select_features(db_session, signal_id, schema)

    assert vector.missing_statuses == {"mom20": "stored_null"}
    assert all(isinstance(value, str) for value in vector.missing_statuses.values())
    assert json.dumps(vector.missing_statuses)


def test_string_status_path_value_is_preserved(db_session):
    signal_id = _seed_signal(db_session, idx=37, gap=None)
    db_session.commit()

    vector = select_features(db_session, signal_id, _feature_schema())

    assert vector.missing_statuses["gap"] == "not_available"


def test_intraday_status_path_must_be_flat_and_allowlisted():
    nested_status_schema = {
        "schema_version": "intraday_status_path_v1",
        "pattern_id": "I11",
        "pattern_clock": "intraday",
        "fields": [
            {
                "name": "gap",
                "source": "feature_snapshot_json",
                "path": "gap",
                "status_path": "statuses.gap",
            }
        ],
    }
    flat_status_schema = {
        "schema_version": "intraday_status_path_v1",
        "pattern_id": "I11",
        "pattern_clock": "intraday",
        "fields": [
            {
                "name": "gap",
                "source": "feature_snapshot_json",
                "path": "gap",
                "status_path": "gap",
            }
        ],
    }

    with pytest.raises(FeatureSelectionError):
        audit_feature_schema_no_leakage(nested_status_schema)
    audit_feature_schema_no_leakage(flat_status_schema)


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
