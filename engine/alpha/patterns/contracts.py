"""
Shared pattern detector contracts for all 17 patterns.

Defines the typed vocabulary for detection inputs, features, signals,
and results. Every detector implements BasePatternDetector.detect().
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums / literals
# ---------------------------------------------------------------------------

class PatternId:
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    M4 = "M4"
    M5 = "M5"
    M6 = "M6"
    M7 = "M7"
    I1 = "I1"
    I2 = "I2"
    I3 = "I3"
    I4 = "I4"
    I5 = "I5"
    I6 = "I6"
    I7 = "I7"
    I8 = "I8"
    I9 = "I9"
    I10 = "I10"

    ALL = [
        "M1", "M2", "M3", "M4", "M5", "M6", "M7",
        "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10",
    ]


class PatternTrack:
    MULTI_DAY = "multi_day"
    INTRADAY = "intraday"


class ThesisCategory:
    RIGHT_TAIL_CONVEX = "right_tail_convex"
    CONTINUATION = "continuation"
    EVENT_DRIFT = "event_drift"
    MEAN_REVERSION = "mean_reversion"
    BINARY_EVENT_NO_STOP = "binary_event/no_stop"


class RouteClass:
    A = "A"
    B = "B"
    C = "C"


class FidelityTier:
    FULL = "FULL"
    LITE = "LITE"
    UNAVAILABLE = "UNAVAILABLE"


class SignalDirection:
    LONG = "long"
    SHORT = "short"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PatternInput:
    """Raw input to a detector — adapter data + context."""

    ticker: str
    asof_timestamp: datetime
    market_data: Dict[str, Any] = field(default_factory=dict)
    fundamental_data: Dict[str, Any] = field(default_factory=dict)
    event_data: Dict[str, Any] = field(default_factory=dict)
    lineage_ids: List[str] = field(default_factory=list)
    lineage_hashes: List[str] = field(default_factory=list)
    universe_snapshot_id: Optional[str] = None
    job_run_id: Optional[str] = None
    code_commit_sha: Optional[str] = None


@dataclass
class PatternFeatures:
    """Computed features from a detector — becomes a feature_snapshot."""

    features: Dict[str, Any]
    feature_manifest_version: Optional[str] = None
    fidelity_tier: str = FidelityTier.FULL
    point_in_time_passed: Optional[bool] = None
    lookahead_guard_passed: Optional[bool] = None


@dataclass
class PatternSignal:
    """A single detection signal."""

    direction: str  # SignalDirection
    raw_signal_strength: float
    raw_expected_edge: float
    signal_horizon: Optional[str] = None
    signal_status: str = "active"
    data_confidence: Optional[float] = None


@dataclass
class PatternDetectionResult:
    """Full output from a detector."""

    pattern_id: str
    ticker: str
    asof_timestamp: datetime
    features: Optional[PatternFeatures] = None
    signals: List[PatternSignal] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    quality_flags: Dict[str, Any] = field(default_factory=dict)
    input_hashes: Dict[str, str] = field(default_factory=dict)
    output_hashes: Dict[str, str] = field(default_factory=dict)

    @property
    def has_signal(self) -> bool:
        return len(self.signals) > 0


# ---------------------------------------------------------------------------
# Base detector
# ---------------------------------------------------------------------------

class BasePatternDetector(abc.ABC):

    @property
    @abc.abstractmethod
    def pattern_id(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def track(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def thesis_category(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def route_class(self) -> str:
        ...

    @abc.abstractmethod
    def detect(self, inp: PatternInput) -> PatternDetectionResult:
        ...
