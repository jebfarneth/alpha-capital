"""Stored-feature reader for Stage-1 ML rankers.

This module intentionally does not fetch provider data or rebuild pattern
features. It reads already-materialized feature payloads and turns a pinned,
ordered feature schema into the vector consumed by both training and inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from alpha.db.models import FeatureSnapshot, MarketPathFeature, SignalRegistry


class FeatureSelectionError(RuntimeError):
    """A stored feature vector cannot be assembled from the manifest contract."""


FORWARD_FEATURE_TOKENS = (
    "forward",
    "future",
    "label",
    "outcome",
    "ret_",
    "return_forward",
    "next_open",
    "next_close",
    "mae",
    "mfe",
    "exit_",
    "path_return",
)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def feature_schema_hash(feature_schema: dict[str, Any]) -> str:
    """Stable hash for a manifest feature schema."""

    return hashlib.sha256(_canonical_json(feature_schema).encode()).hexdigest()


def _json_loads_or_empty(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    payload = json.loads(raw)
    return payload if isinstance(payload, dict) else {}


def _path_parts(path: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if path is None:
        return []
    if isinstance(path, str):
        return [part for part in path.split(".") if part]
    return [str(part) for part in path]


def _get_path(payload: Any, path: str | list[str] | tuple[str, ...] | None) -> Any:
    current = payload
    for part in _path_parts(path):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _as_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _transform(value: float, transform: str | None) -> float:
    if transform in (None, "", "identity"):
        return value
    if math.isnan(value):
        return value
    if transform == "log1p":
        return math.log1p(value) if value > -1.0 else math.nan
    if transform == "signed_log1p":
        return math.copysign(math.log1p(abs(value)), value)
    raise FeatureSelectionError(f"unsupported deterministic transform {transform!r}")


def _normalize_value_for_hash(value: float) -> float | str:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


def feature_vector_hash(names: list[str], values: list[float]) -> str:
    payload = {
        "names": names,
        "values": [_normalize_value_for_hash(value) for value in values],
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def audit_feature_schema_no_leakage(feature_schema: dict[str, Any]) -> None:
    """Fail closed if the feature schema references forward-path predictors."""

    fields = feature_schema.get("fields")
    if not isinstance(fields, list) or not fields:
        raise FeatureSelectionError("feature_schema must define at least one field")
    for field in fields:
        if not isinstance(field, dict):
            raise FeatureSelectionError("feature_schema fields must be objects")
        haystack = " ".join(
            str(field.get(key) or "")
            for key in ("name", "source", "path", "column", "status_path")
        ).lower()
        if any(token in haystack for token in FORWARD_FEATURE_TOKENS):
            raise FeatureSelectionError(
                "forward-path or label field entered the Stage-1 feature schema: "
                f"{field!r}"
            )


@dataclass(frozen=True)
class SelectedFeatureVector:
    signal_id: str
    pattern_id: str
    feature_names: list[str]
    values: list[float]
    feature_schema_hash: str
    feature_vector_hash: str
    missing_statuses: dict[str, Any]


class _StoredFeatureSource:
    def __init__(self, session: Session, signal: SignalRegistry) -> None:
        self.session = session
        self.signal = signal
        self._feature_snapshot_payload: dict[str, Any] | None = None
        self._market_path_rows: dict[tuple[str, str | None, int | None], MarketPathFeature | None] = {}

    @property
    def feature_snapshot_payload(self) -> dict[str, Any]:
        if self._feature_snapshot_payload is None:
            snapshot = self.session.get(FeatureSnapshot, self.signal.feature_snapshot_id)
            if snapshot is None:
                raise FeatureSelectionError(
                    f"feature snapshot {self.signal.feature_snapshot_id!r} missing "
                    f"for signal {self.signal.signal_id!r}"
                )
            self._feature_snapshot_payload = _json_loads_or_empty(snapshot.feature_json)
        return self._feature_snapshot_payload

    def market_path_row(
        self,
        *,
        feature_version: str,
        feature_role: str | None,
        path_sequence: int | None,
    ) -> MarketPathFeature | None:
        key = (feature_version, feature_role, path_sequence)
        if key not in self._market_path_rows:
            query = self.session.query(MarketPathFeature).filter(
                MarketPathFeature.signal_id == self.signal.signal_id,
                MarketPathFeature.feature_version == feature_version,
            )
            if feature_role is not None:
                query = query.filter(MarketPathFeature.feature_role == feature_role)
            if path_sequence is not None:
                query = query.filter(MarketPathFeature.path_sequence == path_sequence)
            self._market_path_rows[key] = (
                query.order_by(MarketPathFeature.path_sequence.asc()).first()
            )
        return self._market_path_rows[key]

    def value_for_field(self, field: dict[str, Any]) -> tuple[Any, Any]:
        source = str(field.get("source") or "feature_snapshot_json")
        if source == "feature_snapshot_json":
            payload = self.feature_snapshot_payload
            return _get_path(payload, field.get("path")), _get_path(
                payload, field.get("status_path")
            )
        if source == "signal_registry":
            attr = str(field.get("column") or field.get("path") or "")
            return getattr(self.signal, attr, None), None
        if source in {"market_path_feature_column", "market_path_feature_json"}:
            feature_version = str(field.get("feature_version") or "")
            if not feature_version:
                raise FeatureSelectionError(
                    f"market-path feature field {field.get('name')!r} missing "
                    "feature_version"
                )
            path_sequence = field.get("path_sequence")
            row = self.market_path_row(
                feature_version=feature_version,
                feature_role=field.get("feature_role"),
                path_sequence=int(path_sequence) if path_sequence is not None else None,
            )
            if row is None:
                return None, "missing_market_path_feature"
            if source == "market_path_feature_column":
                return getattr(row, str(field.get("column") or ""), None), None
            return _get_path(_json_loads_or_empty(row.feature_json), field.get("path")), None
        raise FeatureSelectionError(f"unsupported stored feature source {source!r}")


def select_features(
    session: Session,
    signal_id: str,
    feature_schema: dict[str, Any],
) -> SelectedFeatureVector:
    """Build the pinned ordered vector for one signal from stored fields only."""

    audit_feature_schema_no_leakage(feature_schema)
    signal = session.get(SignalRegistry, signal_id)
    if signal is None:
        raise FeatureSelectionError(f"signal {signal_id!r} not found")
    expected_pattern = feature_schema.get("pattern_id")
    if expected_pattern and str(expected_pattern) != signal.pattern_id:
        raise FeatureSelectionError(
            f"feature schema pattern {expected_pattern!r} does not match signal "
            f"pattern {signal.pattern_id!r}"
        )
    source = _StoredFeatureSource(session, signal)
    names: list[str] = []
    values: list[float] = []
    statuses: dict[str, Any] = {}
    for field in feature_schema["fields"]:
        name = str(field.get("name") or "")
        if not name:
            raise FeatureSelectionError("feature_schema field missing name")
        raw_value, status = source.value_for_field(field)
        value = _transform(_as_float(raw_value), field.get("transform"))
        names.append(name)
        values.append(value)
        if math.isnan(value):
            statuses[name] = status or "stored_null"
    schema_hash = feature_schema_hash(feature_schema)
    vector_hash = feature_vector_hash(names, values)
    return SelectedFeatureVector(
        signal_id=signal_id,
        pattern_id=signal.pattern_id,
        feature_names=names,
        values=values,
        feature_schema_hash=schema_hash,
        feature_vector_hash=vector_hash,
        missing_statuses=statuses,
    )
