"""Fail-closed loader for frozen Stage-1 ML manifests.

Training and inference must be driven by a manifest, not by ad hoc table
selection. The manifest pins the pattern, label source, feature schema, and
minimum cohort contract consumed by the Stage-1 ranker.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alpha.ml.model_features import audit_feature_schema_no_leakage


class MLManifestError(RuntimeError):
    """The ML manifest is missing, malformed, or drifted from its hash."""


_SIGNAL_HORIZON_RE = re.compile(r"^([1-9][0-9]*)d$")
_REQUIRED_OOS_METRICS = {"top_decile_lift", "rank_ic"}
_NON_PROMOTING_REJECT_STATUSES = {"rejected"}


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def manifest_payload_hash(payload: dict[str, Any]) -> str:
    """Hash a manifest payload excluding its self-referential sha field."""

    normalized = dict(payload)
    normalized.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json(normalized).encode()).hexdigest()


def _signal_horizon_sessions(signal_horizon: str) -> int:
    match = _SIGNAL_HORIZON_RE.fullmatch(signal_horizon)
    if match is None:
        raise MLManifestError(
            f"signal_horizon must be canonical '<positive integer>d', got {signal_horizon!r}"
        )
    return int(match.group(1))


def _validated_oos_quality_gate(
    pattern_id: str,
    oos_quality_gate: dict[str, Any],
) -> dict[str, Any]:
    gate = dict(oos_quality_gate)
    try:
        min_top_decile_lift = (
            float(gate["min_top_decile_lift"])
            if "min_top_decile_lift" in gate
            else None
        )
        min_rank_ic = (
            float(gate["min_rank_ic"]) if "min_rank_ic" in gate else None
        )
    except (TypeError, ValueError) as exc:
        raise MLManifestError(
            f"pattern {pattern_id!r} oos_quality_gate thresholds must be numeric"
        ) from exc
    if min_top_decile_lift is not None and min_top_decile_lift < 1.0:
        raise MLManifestError(
            f"pattern {pattern_id!r} oos_quality_gate.min_top_decile_lift "
            "must be >= 1.0"
        )
    if min_rank_ic is not None and min_rank_ic < 0.0:
        raise MLManifestError(
            f"pattern {pattern_id!r} oos_quality_gate.min_rank_ic must be >= 0.0"
        )
    if "required_metrics" in gate:
        required_metrics = gate["required_metrics"]
        if not isinstance(required_metrics, list) or not all(
            isinstance(value, str) for value in required_metrics
        ):
            raise MLManifestError(
                f"pattern {pattern_id!r} oos_quality_gate.required_metrics "
                "must be a list of strings"
            )
        missing = _REQUIRED_OOS_METRICS - set(required_metrics)
        if missing:
            raise MLManifestError(
                f"pattern {pattern_id!r} oos_quality_gate.required_metrics "
                f"must include {sorted(_REQUIRED_OOS_METRICS)}; "
                f"missing {sorted(missing)}"
            )
    if "reject_status" in gate:
        reject_status = str(gate["reject_status"])
        if reject_status not in _NON_PROMOTING_REJECT_STATUSES:
            raise MLManifestError(
                f"pattern {pattern_id!r} oos_quality_gate.reject_status must "
                f"be one of {sorted(_NON_PROMOTING_REJECT_STATUSES)}"
            )
    return gate


@dataclass(frozen=True)
class PatternManifest:
    pattern_id: str
    signal_horizon: str
    min_graded_cohorts: int
    embargo_sessions: int
    feature_schema: dict[str, Any]
    label: dict[str, Any]
    selection: dict[str, Any]
    diagnostics: dict[str, Any]
    direction: str = "long"
    allow_deferred_pit: bool = False
    oos_quality_gate: dict[str, Any] = field(default_factory=dict)
    model_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrozenMLManifest:
    manifest_version: str
    manifest_sha256: str
    patterns: dict[str, PatternManifest]
    raw: dict[str, Any]

    def pattern(self, pattern_id: str) -> PatternManifest:
        try:
            return self.patterns[pattern_id]
        except KeyError:
            raise MLManifestError(
                f"pattern {pattern_id!r} is absent from manifest "
                f"{self.manifest_version!r}"
            )


def load_manifest(path: str | Path) -> FrozenMLManifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise MLManifestError(f"missing ML manifest: {manifest_path}")
    with open(manifest_path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise MLManifestError("ML manifest root must be an object")
    version = str(payload.get("manifest_version") or "").strip()
    if not version:
        raise MLManifestError("ML manifest missing non-empty manifest_version")
    expected_hash = str(payload.get("manifest_sha256") or "").strip()
    if not expected_hash:
        raise MLManifestError("ML manifest missing manifest_sha256")
    actual_hash = manifest_payload_hash(payload)
    if actual_hash != expected_hash:
        raise MLManifestError(
            "ML manifest sha256 mismatch: manifest drifted from its pinned hash "
            f"(expected {expected_hash}, computed {actual_hash})"
        )
    raw_patterns = payload.get("patterns")
    if not isinstance(raw_patterns, dict) or not raw_patterns:
        raise MLManifestError("ML manifest must define at least one pattern")
    patterns: dict[str, PatternManifest] = {}
    for pattern_id, raw in raw_patterns.items():
        if not isinstance(raw, dict):
            raise MLManifestError(f"pattern manifest {pattern_id!r} must be an object")
        feature_schema = raw.get("feature_schema")
        if not isinstance(feature_schema, dict):
            raise MLManifestError(f"pattern {pattern_id!r} missing feature_schema")
        fields = feature_schema.get("fields")
        if not isinstance(fields, list) or not fields:
            raise MLManifestError(
                f"pattern {pattern_id!r} feature_schema must define fields"
            )
        try:
            audit_feature_schema_no_leakage(feature_schema)
        except Exception as exc:
            raise MLManifestError(
                f"pattern {pattern_id!r} feature_schema failed leakage audit: {exc}"
            ) from exc
        label = raw.get("label")
        if not isinstance(label, dict):
            raise MLManifestError(f"pattern {pattern_id!r} missing label contract")
        selection = raw.get("selection")
        if not isinstance(selection, dict):
            raise MLManifestError(f"pattern {pattern_id!r} missing selection contract")
        signal_horizon = str(raw.get("signal_horizon") or "").strip()
        horizon_sessions = selection.get("horizon_sessions")
        if horizon_sessions is None:
            raise MLManifestError(
                f"pattern {pattern_id!r} selection missing horizon_sessions"
            )
        try:
            parsed_horizon_sessions = _signal_horizon_sessions(signal_horizon)
            selected_horizon_sessions = int(horizon_sessions)
        except MLManifestError:
            raise
        except (TypeError, ValueError) as exc:
            raise MLManifestError(
                f"pattern {pattern_id!r} horizon_sessions must be an integer"
            ) from exc
        if parsed_horizon_sessions != selected_horizon_sessions:
            raise MLManifestError(
                f"pattern {pattern_id!r} signal_horizon {signal_horizon!r} "
                f"does not match horizon_sessions={selected_horizon_sessions}"
            )
        min_cohorts = int(raw.get("min_graded_cohorts", 0))
        if min_cohorts <= 0:
            raise MLManifestError(
                f"pattern {pattern_id!r} min_graded_cohorts must be positive"
            )
        embargo = int(raw.get("embargo_sessions", 0))
        if embargo < 0:
            raise MLManifestError(
                f"pattern {pattern_id!r} embargo_sessions must be >= 0"
            )
        direction = str(
            raw.get("direction")
            or selection.get("direction")
            or label.get("direction")
            or "long"
        ).strip().lower()
        if direction not in {"long", "short"}:
            raise MLManifestError(
                f"pattern {pattern_id!r} direction must be 'long' or 'short'"
            )
        oos_quality_gate = raw.get("oos_quality_gate") or selection.get("oos_quality_gate") or {}
        if not isinstance(oos_quality_gate, dict):
            raise MLManifestError(
                f"pattern {pattern_id!r} oos_quality_gate must be an object"
            )
        oos_quality_gate = _validated_oos_quality_gate(
            str(pattern_id),
            oos_quality_gate,
        )
        model_params = raw.get("model_params") or {}
        if not isinstance(model_params, dict):
            raise MLManifestError(
                f"pattern {pattern_id!r} model_params must be an object"
            )
        patterns[str(pattern_id)] = PatternManifest(
            pattern_id=str(pattern_id),
            signal_horizon=signal_horizon,
            min_graded_cohorts=min_cohorts,
            embargo_sessions=embargo,
            feature_schema=feature_schema,
            label=label,
            selection=selection,
            diagnostics=dict(raw.get("diagnostics") or {}),
            direction=direction,
            allow_deferred_pit=bool(selection.get("allow_deferred_pit", False)),
            oos_quality_gate=dict(oos_quality_gate),
            model_params=dict(model_params),
        )
    return FrozenMLManifest(
        manifest_version=version,
        manifest_sha256=expected_hash,
        patterns=patterns,
        raw=payload,
    )
