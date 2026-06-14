"""Shadow inference for Stage-1 per-pattern ML rankers."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from alpha.db.models import MLModelRegistry, SignalMLScore, SignalRegistry
from alpha.ml.model_features import SelectedFeatureVector, select_features


class Stage1InferenceError(RuntimeError):
    """A model artifact or score persistence step failed."""


def _load_artifact(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise Stage1InferenceError(f"malformed model artifact at {path}")
    return payload


def _hand_discriminant(signal: SignalRegistry) -> float:
    """Frozen safe fallback rank when no shadow model can score."""

    if signal.raw_signal_strength is not None:
        return float(signal.raw_signal_strength)
    if signal.raw_expected_edge is not None:
        return float(signal.raw_expected_edge)
    return 0.0


def _out_of_training_distribution(
    vector: SelectedFeatureVector,
    ranges: list[dict[str, float | None]],
) -> list[str]:
    out: list[str] = []
    for name, value, bounds in zip(vector.feature_names, vector.values, ranges):
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
    existing = None
    if row.model_id is not None:
        existing = (
            session.query(SignalMLScore)
            .filter(
                SignalMLScore.signal_id == row.signal_id,
                SignalMLScore.model_id == row.model_id,
                SignalMLScore.score_status == row.score_status,
            )
            .one_or_none()
        )
    if existing is None:
        session.add(row)
        return row
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

    fallback_score = _hand_discriminant(signal)
    if model_row is None and not artifact_uri:
        row = SignalMLScore(
            signal_id=signal.signal_id,
            model_id=None,
            requested_model_id=requested_model_id,
            pattern_id=signal.pattern_id,
            ticker=signal.ticker,
            score=fallback_score,
            fallback_score=fallback_score,
            score_source="fallback_hand_discriminant",
            fallback_reason="model_missing",
            score_status=score_status,
            score_metadata_json=json.dumps({"acts_on_book": False}, sort_keys=True),
        )
        return _upsert_score(session, row)

    artifact = _load_artifact(str(artifact_uri))
    vector = select_features(session, signal_id, artifact["feature_schema"])
    otd_fields = _out_of_training_distribution(
        vector, list(artifact.get("training_feature_ranges") or [])
    )
    if otd_fields:
        row = SignalMLScore(
            signal_id=signal.signal_id,
            model_id=model_row.model_id if model_row is not None else None,
            requested_model_id=requested_model_id,
            pattern_id=signal.pattern_id,
            ticker=signal.ticker,
            score=fallback_score,
            fallback_score=fallback_score,
            score_source="fallback_hand_discriminant",
            fallback_reason="out_of_training_distribution",
            score_status=score_status,
            feature_schema_hash=vector.feature_schema_hash,
            feature_vector_hash=vector.feature_vector_hash,
            score_metadata_json=json.dumps(
                {"acts_on_book": False, "otd_fields": otd_fields}, sort_keys=True
            ),
        )
        return _upsert_score(session, row)

    model = artifact.get("model")
    if model is None:
        raise Stage1InferenceError("model artifact missing model object")
    score = float(model.predict([vector.values])[0])
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
    )
    return _upsert_score(session, row)
