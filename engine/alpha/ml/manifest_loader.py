"""Fail-closed loader for frozen Stage-1 ML manifests.

Training and inference must be driven by a manifest, not by ad hoc table
selection. The manifest pins the pattern, label source, feature schema, and
minimum cohort contract consumed by the Stage-1 ranker.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MLManifestError(RuntimeError):
    """The ML manifest is missing, malformed, or drifted from its hash."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def manifest_payload_hash(payload: dict[str, Any]) -> str:
    """Hash a manifest payload excluding its self-referential sha field."""

    normalized = dict(payload)
    normalized.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json(normalized).encode()).hexdigest()


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
        label = raw.get("label")
        if not isinstance(label, dict):
            raise MLManifestError(f"pattern {pattern_id!r} missing label contract")
        selection = raw.get("selection")
        if not isinstance(selection, dict):
            raise MLManifestError(f"pattern {pattern_id!r} missing selection contract")
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
        patterns[str(pattern_id)] = PatternManifest(
            pattern_id=str(pattern_id),
            signal_horizon=str(raw.get("signal_horizon") or ""),
            min_graded_cohorts=min_cohorts,
            embargo_sessions=embargo,
            feature_schema=feature_schema,
            label=label,
            selection=selection,
            diagnostics=dict(raw.get("diagnostics") or {}),
        )
    return FrozenMLManifest(
        manifest_version=version,
        manifest_sha256=expected_hash,
        patterns=patterns,
        raw=payload,
    )
