"""External market, reference-data, and provider adapter modules."""

from alpha.data.benzinga import BenzingaAdapter, BenzingaMergerAcquisition
from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError, RateLimitInfo

__all__ = [
    "AdapterResponse",
    "BenzingaAdapter",
    "BenzingaMergerAcquisition",
    "LineageMeta",
    "ProviderError",
    "RateLimitInfo",
]
