"""
Fixture detector for testing the pattern framework.

NOT a real pattern — produces deterministic features and signals from
any PatternInput. Used to prove the framework writes feature_snapshots
and signal_registry correctly.
"""

from __future__ import annotations

from alpha.data.contracts import stable_hash
from alpha.patterns.contracts import (
    BasePatternDetector,
    FidelityTier,
    PatternDetectionResult,
    PatternFeatures,
    PatternInput,
    PatternSignal,
    PatternTrack,
    RouteClass,
    SignalDirection,
    ThesisCategory,
)
from alpha.patterns.guards import (
    classify_fidelity,
    reject_future_timestamp,
    require_asof_timestamp,
    require_lineage_hash,
)

SIGNAL_THRESHOLD = 0.90


class FixtureDetector(BasePatternDetector):
    """Test-only detector that fires when fixture_score >= threshold."""

    pattern_id = "FIXTURE"
    version = "1.0"
    track = PatternTrack.MULTI_DAY
    thesis_category = ThesisCategory.RIGHT_TAIL_CONVEX
    route_class = RouteClass.A

    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        """Evaluate fixture data and emit a test signal above threshold."""

        asof = require_asof_timestamp(inp.asof_timestamp)

        warnings = []
        quality_flags = {}

        reject_future_timestamp(asof, warnings, quality_flags)
        require_lineage_hash(inp.lineage_hashes, warnings, quality_flags)

        score = float(inp.market_data.get("fixture_score", 0.0))
        price = float(inp.market_data.get("price", 0.0))

        has_primary = price > 0
        fidelity = classify_fidelity(
            has_primary_data=has_primary,
            point_in_time_passed=quality_flags.get("point_in_time_passed") is not False,
            lookahead_guard_passed=True,
        )

        features_dict = {"fixture_score": score, "price": price}
        if score >= SIGNAL_THRESHOLD and has_primary:
            identity_components = {
                "pattern_id": self.pattern_id,
                "ticker": inp.ticker,
                "fixture_score": round(score, 6),
            }
            features_dict["signal_identity_components"] = identity_components
            features_dict["signal_identity_hash"] = stable_hash(identity_components)
        features = PatternFeatures(
            features=features_dict,
            feature_manifest_version="fixture-v1",
            fidelity_tier=fidelity,
            point_in_time_passed=quality_flags.get("point_in_time_passed") is not False,
            lookahead_guard_passed=True,
        )

        input_hash = stable_hash({"ticker": inp.ticker, "market_data": inp.market_data})
        output_hash = stable_hash(features_dict)

        signals = []
        if score >= SIGNAL_THRESHOLD and has_primary:
            signals.append(
                PatternSignal(
                    direction=SignalDirection.LONG,
                    raw_signal_strength=score,
                    raw_expected_edge=score * 0.10,
                    signal_horizon="10d",
                    data_confidence=0.95,
                )
            )

        return PatternDetectionResult(
            pattern_id=self.pattern_id,
            ticker=inp.ticker,
            asof_timestamp=asof,
            features=features,
            signals=signals,
            warnings=warnings,
            quality_flags=quality_flags,
            input_hashes={"market_data": input_hash},
            output_hashes={"features": output_hash},
        )
