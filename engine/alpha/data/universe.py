"""
Sliced FMP universe source.

Pulls the $30M-$200M market-cap band in equal-width slices to avoid
provider limit truncation (FMP returns the largest names first when a
single broad request hits the limit). Recursively subdivides any slice
that returns exactly ``limit`` results until the slice width reaches
``min_slice_width``, then fails closed with ``slice_limit_exhausted``.

Per Data-Sourcing-Audit.md Universe Filter section.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from alpha.data.contracts import (
    AdapterResponse,
    LineageMeta,
    ProviderError,
    stable_hash,
    utcnow,
)
from alpha.data.fmp import FmpAdapter, FmpScreenerResult

MCAP_MIN = 30_000_000
MCAP_MAX = 200_000_000
SLICE_WIDTH = 10_000_000
MIN_SLICE_WIDTH = 1_000_000
DEFAULT_SLICE_LIMIT = 1000


@dataclass
class SliceDiagnostic:
    lower: int
    upper: int
    returned_count: int
    hit_limit: bool
    query_lower: int
    query_upper: int
    subdivided: bool = False


@dataclass
class SlicedUniverseResult:
    """Combined result from all market-cap slices."""

    response: AdapterResponse[List[FmpScreenerResult]]
    slice_diagnostics: List[SliceDiagnostic] = field(default_factory=list)
    unique_raw_count: int = 0
    total_raw_count: int = 0
    duplicate_count: int = 0
    slice_count: int = 0
    slice_limit_hits: int = 0


class SlicedUniverseFetcher:
    """Fetch the full operating-universe band via market-cap slices."""

    def __init__(
        self,
        adapter: FmpAdapter,
        *,
        mcap_min: int = MCAP_MIN,
        mcap_max: int = MCAP_MAX,
        slice_width: int = SLICE_WIDTH,
        min_slice_width: int = MIN_SLICE_WIDTH,
        limit_per_slice: int = DEFAULT_SLICE_LIMIT,
        max_slice_retries: int = 2,
    ):
        self._adapter = adapter
        self._mcap_min = mcap_min
        self._mcap_max = mcap_max
        self._slice_width = slice_width
        self._min_slice_width = min_slice_width
        self._limit = limit_per_slice
        if max_slice_retries < 0:
            raise ValueError("max_slice_retries must be non-negative")
        self._max_slice_retries = max_slice_retries

    def fetch(self) -> SlicedUniverseResult:
        seen: Dict[str, FmpScreenerResult] = {}
        diagnostics: List[SliceDiagnostic] = []
        payload_hashes: List[str] = []

        error = self._fetch_range(
            self._mcap_min,
            self._mcap_max,
            self._slice_width,
            seen,
            diagnostics,
            payload_hashes,
        )
        total_raw = sum(
            d.returned_count for d in diagnostics if not d.subdivided
        )
        unique_payload = sorted(
            (_stock_payload(stock) for stock in seen.values()),
            key=lambda row: row["symbol"],
        )

        ts = utcnow()
        combined_lineage = LineageMeta(
            provider="FMP",
            endpoint="/stable/company-screener",
            request_timestamp=ts,
            asof_timestamp=ts,
            raw_payload_hash=stable_hash(unique_payload),
            source_authority="FMP_Ultimate",
            data_quality_flags={
                "component_payload_hashes": sorted(payload_hashes),
                "slice_diagnostics": [asdict(d) for d in diagnostics],
            },
        )

        if error is not None:
            return SlicedUniverseResult(
                response=AdapterResponse(
                    data=None, lineage=combined_lineage, error=error,
                ),
                slice_diagnostics=diagnostics,
                total_raw_count=total_raw,
                unique_raw_count=len(seen),
                duplicate_count=total_raw - len(seen),
                slice_count=len(diagnostics),
                slice_limit_hits=sum(1 for d in diagnostics if d.hit_limit),
            )

        return SlicedUniverseResult(
            response=AdapterResponse(data=list(seen.values()), lineage=combined_lineage),
            slice_diagnostics=diagnostics,
            unique_raw_count=len(seen),
            total_raw_count=total_raw,
            duplicate_count=total_raw - len(seen),
            slice_count=len(diagnostics),
            slice_limit_hits=sum(1 for d in diagnostics if d.hit_limit),
        )

    def _fetch_range(
        self,
        lower: int,
        upper: int,
        width: int,
        seen: Dict[str, FmpScreenerResult],
        diagnostics: List[SliceDiagnostic],
        payload_hashes: List[str],
    ) -> Optional[ProviderError]:
        cursor = lower
        while cursor < upper:
            slice_upper = min(cursor + width, upper)
            query_lower = max(0, cursor - 1)
            query_upper = slice_upper + 1
            resp = self._fetch_slice(query_lower, query_upper)
            if not resp.ok:
                return resp.error

            payload_hashes.append(resp.lineage.raw_payload_hash)
            count = len(resp.data)
            hit_limit = count >= self._limit

            if hit_limit:
                diagnostics.append(SliceDiagnostic(
                    lower=cursor,
                    upper=slice_upper,
                    returned_count=count,
                    hit_limit=True,
                    query_lower=query_lower,
                    query_upper=query_upper,
                    subdivided=width > self._min_slice_width,
                ))
                if width <= self._min_slice_width:
                    return ProviderError(
                        provider="FMP",
                        endpoint="/stable/company-screener",
                        status_code=None,
                        error_type="slice_limit_exhausted",
                        message=(
                            f"Slice [{cursor:,}, {slice_upper:,}) hit limit "
                            f"{self._limit} at minimum width {width:,}"
                        ),
                        retryable=False,
                    )
                sub_width = max(width // 2, self._min_slice_width)
                error = self._fetch_range(
                    cursor, slice_upper, sub_width,
                    seen, diagnostics, payload_hashes,
                )
                if error is not None:
                    return error
            else:
                for stock in resp.data:
                    key = _normalize_symbol(stock.symbol)
                    if key and key not in seen:
                        seen[key] = stock
                diagnostics.append(SliceDiagnostic(
                    lower=cursor,
                    upper=slice_upper,
                    returned_count=count,
                    hit_limit=False,
                    query_lower=query_lower,
                    query_upper=query_upper,
                ))

            cursor = slice_upper

        return None

    def _fetch_slice(
        self, query_lower: int, query_upper: int
    ) -> AdapterResponse[List[FmpScreenerResult]]:
        attempts = self._max_slice_retries + 1
        for attempt in range(attempts):
            resp = self._adapter.get_stock_screener(
                market_cap_min=query_lower,
                market_cap_max=query_upper,
                country=None,
                is_etf=None,
                limit=self._limit,
            )
            if resp.ok:
                return resp
            if not resp.error or not resp.error.retryable:
                return resp
            if attempt == attempts - 1:
                return resp
        return resp


def _normalize_symbol(symbol: object) -> str:
    if symbol is None:
        return ""
    return str(symbol).strip().upper()


def _stock_payload(stock: FmpScreenerResult) -> dict:
    return {
        "symbol": _normalize_symbol(stock.symbol),
        "company_name": stock.company_name,
        "market_cap": stock.market_cap,
        "price": stock.price,
        "volume": stock.volume,
        "sector": stock.sector,
        "industry": stock.industry,
        "exchange": stock.exchange,
        "country": stock.country,
        "is_etf": stock.is_etf,
        "is_actively_trading": stock.is_actively_trading,
    }
