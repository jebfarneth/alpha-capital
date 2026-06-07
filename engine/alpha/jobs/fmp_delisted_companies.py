"""FMP delisted-company directory ingestion.

This stores provider delisting directory rows as a replay prerequisite only.
It does not reconstruct or mutate the canonical operating universe.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any, Iterable

from sqlalchemy.orm import Session

from alpha.data.contracts import stable_hash
from alpha.data.fmp import (
    DELISTED_COMPANIES_ENDPOINT,
    FmpAdapter,
    FmpDelistedCompany,
)
from alpha.db.models import FmpDelistedCompanyRecord
from alpha.evidence.writer import record_data_lineage
from alpha.jobs.contracts import BaseJob, JobContext, JobResult


JOB_NAME = "fmp_delisted_companies_ingestion"
SOURCE_PROVIDER = "FMP"
UNKNOWN_KEY = "UNKNOWN"
US_RELEVANT_EXCHANGES = {
    "AMEX",
    "NASDAQ",
    "NASDAQ CAPITAL MARKET",
    "NASDAQCM",
    "NASDAQ GLOBAL MARKET",
    "NASDAQ GLOBAL SELECT",
    "NASDAQGM",
    "NASDAQGS",
    "NYSE",
    "NYSE AMERICAN",
    "NYSE MKT",
    "NYSEAMERICAN",
    "NYSEARCA",
    "OTC",
    "OTC MARKETS",
    "OTCQB",
    "OTCQX",
    "PINK",
}


class FmpDelistedCompaniesIngestionJob(BaseJob):
    """Page through FMP delisted companies and upsert durable raw rows."""

    def __init__(
        self,
        *,
        session: Session,
        fmp_adapter: FmpAdapter,
        run_timestamp: datetime | None = None,
        page_limit: int = 100,
        max_pages: int = 1000,
    ) -> None:
        if page_limit <= 0:
            raise ValueError("page_limit must be positive")
        if max_pages <= 0:
            raise ValueError("max_pages must be positive")
        self.session = session
        self.fmp_adapter = fmp_adapter
        self.run_timestamp = _aware_utc(run_timestamp)
        self.page_limit = page_limit
        self.max_pages = max_pages

    @property
    def job_name(self) -> str:
        return JOB_NAME

    @property
    def job_type(self) -> str:
        return "ingestion"

    @property
    def owner_component(self) -> str:
        return "historical_replay"

    def run(self, ctx: JobContext) -> JobResult:
        metrics: dict[str, Any] = {
            "pages_fetched": 0,
            "pages_with_data": 0,
            "rows_seen": 0,
            "rows_inserted": 0,
            "rows_updated": 0,
            "rows_skipped": 0,
            "malformed_rows": 0,
            "us_listed_rows": 0,
            "non_us_or_unknown_exchange_rows": 0,
            "fetch_error_count": 0,
            "page_limit": self.page_limit,
            "max_pages": self.max_pages,
            "max_pages_reached": False,
        }
        errors: list[dict[str, Any]] = []
        output_hash_parts: list[str] = []

        for page in range(self.max_pages):
            response = self.fmp_adapter.get_delisted_companies(
                page=page,
                limit=self.page_limit,
                asof=self.run_timestamp,
            )
            rows = list(response.data or []) if response.ok else []
            metrics["pages_fetched"] += 1

            lineage = record_data_lineage(
                self.session,
                provider=SOURCE_PROVIDER,
                endpoint=DELISTED_COMPANIES_ENDPOINT,
                asof_timestamp=response.lineage.asof_timestamp,
                request_timestamp=response.lineage.request_timestamp,
                raw_payload={
                    "page": page,
                    "limit": self.page_limit,
                    "rows": [_row_payload(row) for row in rows],
                    "error": _provider_error_payload(response.error),
                },
                raw_payload_hash=response.lineage.raw_payload_hash,
                freshness_seconds=response.lineage.freshness_seconds,
                source_authority=response.lineage.source_authority or SOURCE_PROVIDER,
                data_quality_flags={
                    "lineage_scope": "fmp_delisted_companies_page_fetch",
                    "page": page,
                    "limit": self.page_limit,
                    "row_count": len(rows),
                    "status": "ok" if response.ok else "error",
                },
                job_run_id=ctx.job_run_id,
            )

            if not response.ok:
                metrics["fetch_error_count"] += 1
                errors.append(
                    {
                        "stage": "fetch",
                        "page": page,
                        "provider": SOURCE_PROVIDER,
                        "endpoint": DELISTED_COMPANIES_ENDPOINT,
                        "error": _provider_error_payload(response.error),
                    }
                )
                break

            if not rows:
                break

            metrics["pages_with_data"] += 1
            page_hashes = self._persist_page(
                rows=rows,
                page=page,
                lineage_id=lineage.data_lineage_id,
                job_run_id=ctx.job_run_id,
                metrics=metrics,
            )
            output_hash_parts.extend(page_hashes)
        else:
            metrics["max_pages_reached"] = True
            errors.append(
                {
                    "stage": "pagination",
                    "error_type": "max_pages_reached",
                    "message": (
                        "FMP delisted-company ingestion reached max_pages before "
                        "observing an empty terminal page"
                    ),
                    "max_pages": self.max_pages,
                    "page_limit": self.page_limit,
                }
            )

        self.session.flush()
        status = "partial_failed" if errors else "finished"
        return JobResult(
            status=status,
            metrics=metrics,
            input_hashes={
                "fmp_delisted_companies_request": stable_hash(
                    {
                        "endpoint": DELISTED_COMPANIES_ENDPOINT,
                        "page_limit": self.page_limit,
                        "max_pages": self.max_pages,
                    }
                )
            },
            output_hashes={
                "fmp_delisted_companies_rows": stable_hash(sorted(output_hash_parts))
            },
            errors=errors,
        )

    def _persist_page(
        self,
        *,
        rows: Iterable[Any],
        page: int,
        lineage_id: str,
        job_run_id: str,
        metrics: dict[str, Any],
    ) -> list[str]:
        output_hashes: list[str] = []
        for row_index, row in enumerate(rows):
            metrics["rows_seen"] += 1
            try:
                normalized = _normalize_row(row)
            except ValueError:
                metrics["malformed_rows"] += 1
                metrics["rows_skipped"] += 1
                continue

            raw_payload_hash = stable_hash(normalized["raw_payload"])
            output_hashes.append(raw_payload_hash)
            existing = (
                self.session.query(FmpDelistedCompanyRecord)
                .filter(
                    FmpDelistedCompanyRecord.normalized_symbol
                    == normalized["normalized_symbol"],
                    FmpDelistedCompanyRecord.exchange_key
                    == normalized["exchange_key"],
                    FmpDelistedCompanyRecord.delisted_date_key
                    == normalized["delisted_date_key"],
                )
                .one_or_none()
            )
            request_metadata_json = json.dumps(
                {
                    "page": page,
                    "limit": self.page_limit,
                    "page_row_index": row_index,
                    "lineage_scope": "page_fetch",
                },
                sort_keys=True,
            )
            values = {
                "symbol": normalized["symbol"],
                "normalized_symbol": normalized["normalized_symbol"],
                "company_name": normalized["company_name"],
                "exchange": normalized["exchange"],
                "exchange_key": normalized["exchange_key"],
                "ipo_date": normalized["ipo_date"],
                "delisted_date": normalized["delisted_date"],
                "delisted_date_key": normalized["delisted_date_key"],
                "source": SOURCE_PROVIDER,
                "source_endpoint": DELISTED_COMPANIES_ENDPOINT,
                "page_number": page,
                "page_limit": self.page_limit,
                "page_row_index": row_index,
                "row_status": "active",
                "exchange_relevance_status": normalized["exchange_relevance_status"],
                "raw_payload_hash": raw_payload_hash,
                "raw_payload_json": json.dumps(
                    normalized["raw_payload"], sort_keys=True, default=str
                ),
                "request_metadata_json": request_metadata_json,
                "data_lineage_id": lineage_id,
                "ingestion_job_run_id": job_run_id,
            }
            if existing is None:
                self.session.add(FmpDelistedCompanyRecord(**values))
                metrics["rows_inserted"] += 1
            else:
                for key, value in values.items():
                    setattr(existing, key, value)
                metrics["rows_updated"] += 1

            if normalized["exchange_relevance_status"] == "us_listed_relevant":
                metrics["us_listed_rows"] += 1
            else:
                metrics["non_us_or_unknown_exchange_rows"] += 1
        return output_hashes


def _normalize_row(row: Any) -> dict[str, Any]:
    payload = _row_payload(row)
    symbol = _clean_string(getattr(row, "symbol", None) or payload.get("symbol"))
    if not symbol:
        raise ValueError("missing symbol")
    ipo_date = _parse_optional_date(
        getattr(row, "ipo_date", None) or payload.get("ipoDate") or payload.get("ipo_date")
    )
    delisted_date = _parse_optional_date(
        getattr(row, "delisted_date", None)
        or payload.get("delistedDate")
        or payload.get("delisted_date")
    )
    exchange = _clean_string(getattr(row, "exchange", None) or payload.get("exchange"))
    exchange_key = _key(exchange)
    return {
        "symbol": symbol,
        "normalized_symbol": symbol.upper(),
        "company_name": _clean_string(
            getattr(row, "company_name", None)
            or payload.get("companyName")
            or payload.get("name")
        ),
        "exchange": exchange,
        "exchange_key": exchange_key,
        "ipo_date": ipo_date,
        "delisted_date": delisted_date,
        "delisted_date_key": delisted_date.isoformat() if delisted_date else UNKNOWN_KEY,
        "exchange_relevance_status": _exchange_relevance_status(exchange_key),
        "raw_payload": payload,
    }


def _row_payload(row: Any) -> dict[str, Any]:
    if isinstance(row, FmpDelistedCompany):
        if isinstance(row.raw, dict):
            return dict(row.raw)
        return {
            "symbol": row.symbol,
            "companyName": row.company_name,
            "exchange": row.exchange,
            "ipoDate": row.ipo_date,
            "delistedDate": row.delisted_date,
        }
    if isinstance(row, dict):
        return dict(row)
    raw = getattr(row, "raw", None)
    if isinstance(raw, dict):
        return dict(raw)
    return {
        "symbol": getattr(row, "symbol", None),
        "companyName": getattr(row, "company_name", None),
        "exchange": getattr(row, "exchange", None),
        "ipoDate": getattr(row, "ipo_date", None),
        "delistedDate": getattr(row, "delisted_date", None),
    }


def _parse_optional_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        return date.fromisoformat(cleaned[:10])
    raise ValueError(f"invalid date value: {value!r}")


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _key(value: str | None) -> str:
    if not value:
        return UNKNOWN_KEY
    return " ".join(value.upper().replace("_", " ").split())


def _exchange_relevance_status(exchange_key: str) -> str:
    if exchange_key in US_RELEVANT_EXCHANGES:
        return "us_listed_relevant"
    return "non_us_or_unknown_exchange"


def _provider_error_payload(error: Any) -> dict[str, Any] | None:
    if error is None:
        return None
    return {
        "status_code": getattr(error, "status_code", None),
        "error_type": getattr(error, "error_type", None),
        "message": getattr(error, "message", str(error)),
        "retryable": getattr(error, "retryable", None),
    }


def _aware_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
