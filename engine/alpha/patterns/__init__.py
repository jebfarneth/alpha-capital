from alpha.patterns.contracts import (
    BasePatternDetector,
    FidelityTier,
    PatternDetectionResult,
    PatternFeatures,
    PatternId,
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
from alpha.patterns.evidence_bridge import persist_detection_result
