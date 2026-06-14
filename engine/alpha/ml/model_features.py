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
    "entry",
    "path_return",
    "return_from_entry",
    "realized_pnl",
)
FORWARD_FEATURE_EXACT_TOKENS = frozenset(
    {"win", "pnl", "target", "y", "q", "profit", "gain", "realized", "exit"}
)
ALLOWED_FEATURE_SOURCES = frozenset(
    {
        "feature_snapshot_json",
        "signal_registry",
        "market_path_feature_column",
        "market_path_feature_json",
    }
)
ALLOWED_MARKET_PATH_FEATURE_ROLES = frozenset(
    {
        "signal_session",
        "signal_day",
        "signal_day_t0",
        "t0_signal_context",
    }
)
FORBIDDEN_ZONE_TOKENS = (
    "forward",
    "post_signal",
    "post-signal",
    "outcome",
    "label",
    "future",
    "exit",
)
INTRADAY_ALLOWED_SIGNAL_SESSION_FIELDS = frozenset(
    {
        "open_price",
        "previous_close",
        "gap_pct",
        # These market_path_features columns are prior-window baselines produced
        # from sessions before the signal day; current-day expansion fields are
        # intentionally excluded.
        "median_volume_20d",
        "median_volume_60d",
        "median_dollar_volume_20d",
        "median_dollar_volume_60d",
    }
)
INTRADAY_ALLOWED_SNAPSHOT_PATHS = frozenset(
    {
        "gap",
        "prev_day_return",
        "prev_day_green",
        "mom20",
        "off_low252",
        "sigma20",
        "distance_from_max252",
        "drawdown_from_max252",
        "projected_volume_ratio_at_confirmation",
        "projected_volume_at_confirmation",
        "chase_pct",
        "spy_prior_day_return",
    }
)
INTRADAY_ALLOWED_SIGNAL_REGISTRY_COLUMNS = frozenset()


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
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _is_non_finite_raw_value(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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


def _tokenize(value: str) -> set[str]:
    out: list[str] = []
    current: list[str] = []
    for char in value.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return set(out)


def _read_locator_parts(field: dict[str, Any]) -> list[str]:
    """Return the field locator parts for the value this source actually reads."""

    source = str(field.get("source") or "feature_snapshot_json")
    has_column = field.get("column") not in (None, "")
    has_path = field.get("path") not in (None, "")
    if source in {"feature_snapshot_json", "market_path_feature_json"}:
        if has_column:
            raise FeatureSelectionError(
                f"{source} fields must not carry an ignored column locator: "
                f"{field!r}"
            )
        parts = _path_parts(field.get("path"))
    elif source == "market_path_feature_column":
        if has_path:
            raise FeatureSelectionError(
                "market_path_feature_column fields must not carry an ignored "
                f"path locator: {field!r}"
            )
        parts = [str(field["column"])] if has_column else []
    elif source == "signal_registry":
        if has_column and has_path:
            raise FeatureSelectionError(
                "signal_registry fields must declare either column or path, "
                f"not both: {field!r}"
            )
        parts = _path_parts(field.get("column") or field.get("path"))
    else:
        return []
    if not parts:
        raise FeatureSelectionError(
            f"feature field is missing a readable locator for source {source!r}: "
            f"{field!r}"
        )
    return parts


def _audit_feature_field_no_leakage(
    feature_schema: dict[str, Any],
    field: dict[str, Any],
) -> None:
    pattern_clock = str(feature_schema.get("pattern_clock") or "").lower()
    source = str(field.get("source") or "feature_snapshot_json")
    if source not in ALLOWED_FEATURE_SOURCES:
        raise FeatureSelectionError(
            f"feature source {source!r} is not allowed for Stage-1 predictors"
        )
    feature_role = str(field.get("feature_role") or "").strip()
    reference_parts = _read_locator_parts(field)
    terminal_name = reference_parts[-1] if reference_parts else ""
    if source.startswith("market_path_feature"):
        if not feature_role:
            raise FeatureSelectionError(
                "market_path_feature fields must declare an allowed signal-day "
                "feature_role"
            )
        if feature_role not in ALLOWED_MARKET_PATH_FEATURE_ROLES:
            raise FeatureSelectionError(
                "market_path_feature field references a non-signal-day zone: "
                f"{field!r}"
            )
        if pattern_clock == "intraday":
            if source == "market_path_feature_json" and len(reference_parts) != 1:
                raise FeatureSelectionError(
                    "intraday Stage-1 market-path JSON fields must use flat "
                    "as-of-signal-time top-level paths: "
                    f"{field!r}"
                )
            if terminal_name not in INTRADAY_ALLOWED_SIGNAL_SESSION_FIELDS:
                raise FeatureSelectionError(
                    "intraday Stage-1 predictors can read only explicitly "
                    "allowlisted as-of-signal-time market-path fields: "
                    f"{field!r}"
                )
    else:
        if feature_role and any(
            token in feature_role.lower() for token in FORBIDDEN_ZONE_TOKENS
        ):
            raise FeatureSelectionError(
                "feature field references a forbidden forward/outcome zone: "
                f"{field!r}"
            )
        if pattern_clock == "intraday" and source == "feature_snapshot_json":
            if len(reference_parts) != 1:
                raise FeatureSelectionError(
                    "intraday Stage-1 feature snapshots must use flat "
                    "as-of-signal-time top-level paths: "
                    f"{field!r}"
                )
            if terminal_name not in INTRADAY_ALLOWED_SNAPSHOT_PATHS:
                raise FeatureSelectionError(
                    "intraday Stage-1 feature snapshots can read only explicitly "
                    "allowlisted as-of-signal-time fields: "
                    f"{field!r}"
                )
        if pattern_clock == "intraday" and source == "signal_registry":
            if len(reference_parts) != 1:
                raise FeatureSelectionError(
                    "intraday Stage-1 signal_registry fields must use flat "
                    f"top-level column names: {field!r}"
                )
            if terminal_name not in INTRADAY_ALLOWED_SIGNAL_REGISTRY_COLUMNS:
                raise FeatureSelectionError(
                    "intraday Stage-1 signal_registry fields are denied by "
                    f"default: {field!r}"
                )
    haystack = " ".join(
        str(field.get(key) or "")
        for key in (
            "name",
            "source",
            "path",
            "column",
            "status_path",
            "feature_role",
        )
    ).lower()
    tokens = _tokenize(haystack)
    if any(token in haystack for token in FORWARD_FEATURE_TOKENS) or (
        tokens & FORWARD_FEATURE_EXACT_TOKENS
    ):
        raise FeatureSelectionError(
            "forward-path or label field entered the Stage-1 feature schema: "
            f"{field!r}"
        )


def audit_feature_schema_no_leakage(feature_schema: dict[str, Any]) -> None:
    """Fail closed if the feature schema references forward-path predictors."""

    pattern_clock = str(feature_schema.get("pattern_clock") or "").lower()
    if pattern_clock not in {"eod", "intraday"}:
        raise FeatureSelectionError(
            "feature_schema pattern_clock must be either 'eod' or 'intraday'"
        )
    fields = feature_schema.get("fields")
    if not isinstance(fields, list) or not fields:
        raise FeatureSelectionError("feature_schema must define at least one field")
    for field in fields:
        if not isinstance(field, dict):
            raise FeatureSelectionError("feature_schema fields must be objects")
        _audit_feature_field_no_leakage(feature_schema, field)


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
    def __init__(
        self,
        session: Session,
        signal: SignalRegistry,
        feature_schema: dict[str, Any],
    ) -> None:
        self.session = session
        self.signal = signal
        self.feature_schema = feature_schema
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
        _audit_feature_field_no_leakage(self.feature_schema, field)
        source = str(field.get("source") or "feature_snapshot_json")
        locator_parts = _read_locator_parts(field)
        if source == "feature_snapshot_json":
            payload = self.feature_snapshot_payload
            status_path = field.get("status_path")
            return _get_path(payload, locator_parts), (
                _get_path(payload, status_path) if status_path else None
            )
        if source == "signal_registry":
            attr = ".".join(locator_parts)
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
                return getattr(row, locator_parts[0], None), None
            return _get_path(_json_loads_or_empty(row.feature_json), locator_parts), None
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
    source = _StoredFeatureSource(session, signal, feature_schema)
    names: list[str] = []
    values: list[float] = []
    statuses: dict[str, Any] = {}
    for field in feature_schema["fields"]:
        name = str(field.get("name") or "")
        if not name:
            raise FeatureSelectionError("feature_schema field missing name")
        raw_value, status = source.value_for_field(field)
        non_finite_raw_value = _is_non_finite_raw_value(raw_value)
        value = _transform(_as_float(raw_value), field.get("transform"))
        names.append(name)
        values.append(value)
        if math.isnan(value):
            if non_finite_raw_value:
                statuses[name] = "non_finite_stored_value"
            else:
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
