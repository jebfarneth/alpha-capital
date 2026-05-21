"""
Shared adapter contracts.

Every external data call returns an AdapterResponse carrying:
  - Typed payload (or None on error)
  - LineageMeta sufficient for record_data_lineage
  - Rate-limit metadata when the provider exposes it
  - ProviderError on failure
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, List, Optional, TypeVar

T = TypeVar("T")


def stable_hash(payload: Any) -> str:
    """Deterministic SHA-256 of a JSON-serializable payload."""
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class LineageMeta:
    """Metadata produced by every adapter call, sufficient for record_data_lineage."""

    provider: str
    endpoint: str
    request_timestamp: datetime
    asof_timestamp: datetime
    raw_payload_hash: str
    freshness_seconds: Optional[float] = None
    source_authority: Optional[str] = None
    data_quality_flags: Optional[dict] = None


@dataclass(frozen=True)
class ProviderError:
    provider: str
    endpoint: str
    status_code: Optional[int]
    error_type: str  # http, timeout, auth, rate_limit, parse, validation
    message: str
    retryable: bool


@dataclass(frozen=True)
class RateLimitInfo:
    calls_remaining: Optional[int] = None
    calls_limit: Optional[int] = None
    reset_at: Optional[datetime] = None


@dataclass
class AdapterResponse(Generic[T]):
    """Unified response from any data adapter."""

    data: Optional[T]
    lineage: LineageMeta
    rate_limit: Optional[RateLimitInfo] = None
    error: Optional[ProviderError] = None

    @property
    def ok(self) -> bool:
        return self.error is None
