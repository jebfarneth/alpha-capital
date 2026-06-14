"""Shadow inference for Stage-1 per-pattern ML rankers."""

from __future__ import annotations

import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from alpha.db.models import MLModelRegistry, SignalMLScore, SignalRegistry
from alpha.ml.model_features import (
    SelectedFeatureVector,
    feature_schema_hash,
    select_features,
)


class Stage1InferenceError(RuntimeError):
    """A model artifact or score persistence step failed."""


def _load_artifact(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise Stage1InferenceError(f"malformed model artifact at {path}")
    return payload


def _raw_strength_fallback_score(signal: SignalRegistry) -> float:
    """Temporary raw-strength fallback until the frozen live discriminant is wired."""

    # TODO: before canary/live, replace this with the production frozen
    # recovery-context discriminant. Shadow fallback is logged explicitly as
    # raw strength so it cannot be mistaken for the final live fallback.
    if signal.raw_signal_strength is not None:
        return float(signal.raw_signal_strength)
    if signal.raw_expected_edge is not None:
        return float(signal.raw_expected_edge)
    return 0.0


def _out_of_training_distribution(
    vector: SelectedFeatureVector,
    ranges: Any,
) -> list[str]:
    if not isinstance(ranges, list):
        raise Stage1InferenceError("training_feature_ranges must be a list")
    if len(ranges) != len(vector.feature_names):
        raise Stage1InferenceError(
            "training_feature_ranges length does not match feature vector length"
        )
    out: list[str] = []
    for name, value, bounds in zip(vector.feature_names, vector.values, ranges):
        if not isinstance(bounds, dict):
            raise Stage1InferenceError(
                f"training_feature_ranges entry for {name!r} must be an object"
            )
        if math.isnan(value):
            continue
        lower = bounds.get("min")
        upper = bounds.get("max")
        if lower is not None and value < lower:
            out.append(name)
        elif upper is not None and value > upper:
            out.append(name)
    return out


def _upsert_score(session: Session, row: SignalMLScore) -> SignalMLScore:
    if row.scored_at is None:
        row.scored_at = datetime.now(timezone.utc)
    query = session.query(SignalMLScore).filter(
        SignalMLScore.signal_id == row.signal_id,
        SignalMLScore.score_status == row.score_status,
    )
    if row.model_id is None:
        query = query.filter(SignalMLScore.model_id.is_(None))
        if row.requested_model_id is None:
            query = query.filter(SignalMLScore.requested_model_id.is_(None))
        else:
            query = query.filter(
                SignalMLScore.requested_model_id == row.requested_model_id
            )
    else:
        query = query.filter(SignalMLScore.model_id == row.model_id)
    matches = query.order_by(SignalMLScore.scored_at.desc()).all()
    existing = matches[0] if matches else None
    if existing is None:
        session.add(row)
        return row
    for duplicate in matches[1:]:
        session.delete(duplicate)
    existing.requested_model_id = row.requested_model_id
    existing.score = row.score
    existing.fallback_score = row.fallback_score
    existing.score_source = row.score_source
    existing.fallback_reason = row.fallback_reason
    existing.feature_schema_hash = row.feature_schema_hash
    existing.feature_vector_hash = row.feature_vector_hash
    existing.score_metadata_json = row.score_metadata_json
    existing.scored_at = row.scored_at
    return existing


def _fallback_row(
    session: Session,
    *,
    signal: SignalRegistry,
    requested_model_id: str | None,
    model_row: MLModelRegistry | None,
    reason: str,
    score_status: str,
    vector: SelectedFeatureVector | None = None,
    metadata: dict[str, Any] | None = None,
) -> SignalMLScore:
    fallback_score = _raw_strength_fallback_score(signal)
    payload: dict[str, Any] = {"acts_on_book": False}
    if metadata:
        payload.update(metadata)
    row = SignalMLScore(
        signal_id=signal.signal_id,
        model_id=model_row.model_id if model_row is not None else None,
        requested_model_id=requested_model_id,
        pattern_id=signal.pattern_id,
        ticker=signal.ticker,
        score=fallback_score,
        fallback_score=fallback_score,
        score_source="fallback_raw_strength",
        fallback_reason=reason,
        score_status=score_status,
        feature_schema_hash=vector.feature_schema_hash if vector is not None else None,
        feature_vector_hash=vector.feature_vector_hash if vector is not None else None,
        score_metadata_json=json.dumps(payload, sort_keys=True),
        scored_at=datetime.now(timezone.utc),
    )
    return _upsert_score(session, row)


def _artifact_identity_mismatch(
    artifact: dict[str, Any],
    model_row: MLModelRegistry,
) -> dict[str, Any] | None:
    actual_schema_hash = None
    schema_error = None
    try:
        actual_schema_hash = feature_schema_hash(artifact["feature_schema"])
    except Exception as exc:
        schema_error = f"{type(exc).__name__}: {str(exc)[:200]}"
    checks = {
        "model_id": (artifact.get("model_id"), model_row.model_id),
        "pattern_id": (artifact.get("pattern_id"), model_row.pattern_id),
        "feature_schema_hash": (
            artifact.get("feature_schema_hash"),
            model_row.feature_schema_hash,
        ),
        "actual_feature_schema_hash": (
            actual_schema_hash,
            model_row.feature_schema_hash,
        ),
        "declared_vs_actual_feature_schema_hash": (
            artifact.get("feature_schema_hash"),
            actual_schema_hash,
        ),
        "manifest_sha256": (artifact.get("manifest_sha256"), model_row.manifest_sha256),
        "model_family": (artifact.get("model_family"), model_row.model_family),
    }
    mismatches = {
        key: {"artifact": actual, "registry": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if schema_error is not None:
        mismatches["feature_schema_hash_error"] = {
            "artifact": schema_error,
            "registry": model_row.feature_schema_hash,
        }
    return mismatches or None


def _safe_error_metadata(exc: Exception) -> dict[str, str]:
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:300],
    }


def _model_predict(model: Any, values: list[float]) -> float:
    prediction = model.predict([values])
    return float(prediction[0])


def score_signal_shadow(
    session: Session,
    *,
    signal_id: str,
    model_id: str | None = None,
    artifact_uri: str | None = None,
    score_status: str = "shadow",
) -> SignalMLScore:
    """Persist a shadow ML score, falling back safely when unavailable."""

    signal = session.get(SignalRegistry, signal_id)
    if signal is None:
        raise Stage1InferenceError(f"signal {signal_id!r} not found")

    model_row = session.get(MLModelRegistry, model_id) if model_id else None
    requested_model_id = model_id
    if model_row is not None:
        artifact_uri = model_row.artifact_uri

    if model_row is None and not artifact_uri:
        return _fallback_row(
            session,
            signal=signal,
            requested_model_id=requested_model_id,
            model_row=None,
            reason="model_missing",
            score_status=score_status,
        )

    try:
        artifact = _load_artifact(str(artifact_uri))
    except Exception as exc:
        return _fallback_row(
            session,
            signal=signal,
            requested_model_id=requested_model_id,
            model_row=model_row,
            reason="artifact_load_error",
            score_status=score_status,
            metadata=_safe_error_metadata(exc),
        )

    if model_row is not None:
        mismatches = _artifact_identity_mismatch(artifact, model_row)
        if mismatches:
            reason = (
                "artifact_schema_hash_mismatch"
                if any("feature_schema_hash" in key for key in mismatches)
                else "artifact_identity_mismatch"
            )
            return _fallback_row(
                session,
                signal=signal,
                requested_model_id=requested_model_id,
                model_row=model_row,
                reason=reason,
                score_status=score_status,
                metadata={"identity_mismatches": mismatches},
            )

    try:
        vector = select_features(session, signal_id, artifact["feature_schema"])
    except Exception as exc:
        return _fallback_row(
            session,
            signal=signal,
            requested_model_id=requested_model_id,
            model_row=model_row,
            reason="feature_selection_error",
            score_status=score_status,
            metadata=_safe_error_metadata(exc),
        )

    try:
        otd_fields = _out_of_training_distribution(
            vector, artifact.get("training_feature_ranges")
        )
    except Exception as exc:
        return _fallback_row(
            session,
            signal=signal,
            requested_model_id=requested_model_id,
            model_row=model_row,
            reason="otd_check_error",
            score_status=score_status,
            vector=vector,
            metadata=_safe_error_metadata(exc),
        )
    if otd_fields:
        return _fallback_row(
            session,
            signal=signal,
            requested_model_id=requested_model_id,
            model_row=model_row,
            reason="out_of_training_distribution",
            score_status=score_status,
            vector=vector,
            metadata={"otd_fields": otd_fields},
        )

    model = artifact.get("model")
    if model is None:
        return _fallback_row(
            session,
            signal=signal,
            requested_model_id=requested_model_id,
            model_row=model_row,
            reason="artifact_load_error",
            score_status=score_status,
            vector=vector,
            metadata={"error_type": "MissingModelObject"},
        )
    try:
        score = _model_predict(model, vector.values)
    except Exception as exc:
        return _fallback_row(
            session,
            signal=signal,
            requested_model_id=requested_model_id,
            model_row=model_row,
            reason="predict_error",
            score_status=score_status,
            vector=vector,
            metadata=_safe_error_metadata(exc),
        )
    if not math.isfinite(score):
        return _fallback_row(
            session,
            signal=signal,
            requested_model_id=requested_model_id,
            model_row=model_row,
            reason="non_finite_score",
            score_status=score_status,
            vector=vector,
            metadata={"score": str(score)},
        )
    row = SignalMLScore(
        signal_id=signal.signal_id,
        model_id=model_row.model_id if model_row is not None else None,
        requested_model_id=requested_model_id,
        pattern_id=signal.pattern_id,
        ticker=signal.ticker,
        score=score,
        fallback_score=None,
        score_source="model_shadow",
        fallback_reason=None,
        score_status=score_status,
        feature_schema_hash=vector.feature_schema_hash,
        feature_vector_hash=vector.feature_vector_hash,
        score_metadata_json=json.dumps(
            {
                "acts_on_book": False,
                "manifest_version": artifact.get("manifest_version"),
                "model_family": artifact.get("model_family"),
            },
            sort_keys=True,
        ),
        scored_at=datetime.now(timezone.utc),
    )
    return _upsert_score(session, row)
