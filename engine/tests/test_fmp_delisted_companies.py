from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import func, inspect
from sqlalchemy.exc import IntegrityError

from alpha.data.contracts import AdapterResponse, LineageMeta, stable_hash
from alpha.data.fmp import DELISTED_COMPANIES_ENDPOINT, FmpDelistedCompany
from alpha.db.models import FmpDelistedCompanyRecord
from alpha.jobs.contracts import JobResult
from alpha.jobs.fmp_delisted_companies import FmpDelistedCompaniesIngestionJob
from alpha.jobs.run_delisted_companies import _exit_code_for_result
from alpha.jobs.runner import run_job


def _ts() -> datetime:
    return datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc)


def _lineage(page: int, rows: list[FmpDelistedCompany]) -> LineageMeta:
    return LineageMeta(
        provider="FMP",
        endpoint=DELISTED_COMPANIES_ENDPOINT,
        request_timestamp=_ts(),
        asof_timestamp=_ts(),
        raw_payload_hash=stable_hash(
            {"page": page, "rows": [row.raw or row.symbol for row in rows]}
        ),
        source_authority="FMP",
    )


def _row(
    symbol: str,
    *,
    company_name: str | None = "Example Inc.",
    exchange: str | None = "NASDAQ",
    ipo_date: str | None = "2020-01-02",
    delisted_date: str | None = "2024-05-06",
) -> FmpDelistedCompany:
    raw = {
        "symbol": symbol,
        "companyName": company_name,
        "exchange": exchange,
        "ipoDate": ipo_date,
        "delistedDate": delisted_date,
    }
    return FmpDelistedCompany(
        symbol=symbol,
        company_name=company_name,
        exchange=exchange,
        ipo_date=ipo_date,
        delisted_date=delisted_date,
        raw=raw,
    )


class FakeFmpAdapter:
    def __init__(self, pages: dict[int, list[FmpDelistedCompany]]) -> None:
        self.pages = pages
        self.calls: list[tuple[int, int]] = []

    def get_delisted_companies(self, *, page: int = 0, limit: int = 100, asof=None):
        self.calls.append((page, limit))
        rows = self.pages.get(page, [])
        return AdapterResponse(data=rows, lineage=_lineage(page, rows))


def _run_ingestion(db_session, adapter: FakeFmpAdapter, *, max_pages: int = 10):
    job = FmpDelistedCompaniesIngestionJob(
        session=db_session,
        fmp_adapter=adapter,
        run_timestamp=_ts(),
        page_limit=2,
        max_pages=max_pages,
    )
    return run_job(
        db_session,
        job,
        params={"source": "test_fmp_delisted_companies"},
    )


def test_schema_contains_delisted_company_columns_and_uniqueness(db_session):
    inspector = inspect(db_session.bind)
    columns = {col["name"] for col in inspector.get_columns("fmp_delisted_companies")}
    assert {
        "symbol",
        "normalized_symbol",
        "company_name",
        "exchange",
        "exchange_key",
        "ipo_date",
        "delisted_date",
        "delisted_date_key",
        "source",
        "raw_payload_hash",
        "raw_payload_json",
        "request_metadata_json",
        "data_lineage_id",
        "ingestion_job_run_id",
        "created_at",
        "updated_at",
    } <= columns
    unique_constraints = inspector.get_unique_constraints("fmp_delisted_companies")
    assert any(
        constraint["name"] == "ux_fmp_delisted_companies_symbol_exchange_delisted"
        and constraint["column_names"]
        == ["normalized_symbol", "exchange_key", "delisted_date_key"]
        for constraint in unique_constraints
    )
    indexes = {
        index["name"]: index["column_names"]
        for index in inspector.get_indexes("fmp_delisted_companies")
    }
    assert indexes["ix_fmp_delisted_companies_ipo_date"] == ["ipo_date"]
    assert indexes["ix_fmp_delisted_companies_exchange_relevance"] == [
        "exchange_relevance_status"
    ]
    assert indexes["ix_fmp_delisted_companies_replay_filter"] == [
        "exchange_relevance_status",
        "ipo_date",
        "delisted_date",
    ]


def test_pagination_stops_on_empty_page(db_session):
    adapter = FakeFmpAdapter(
        {
            0: [_row("AAA")],
            1: [_row("BBB", exchange="NYSE", delisted_date="2024-06-07")],
        }
    )

    result = _run_ingestion(db_session, adapter)

    assert result.status == "finished"
    assert result.ok
    assert adapter.calls == [(0, 2), (1, 2), (2, 2)]
    assert result.metrics["pages_fetched"] == 3
    assert result.metrics["pages_with_data"] == 2
    assert result.metrics["max_pages_reached"] is False
    assert result.metrics["rows_inserted"] == 2
    assert db_session.query(FmpDelistedCompanyRecord).count() == 2


def test_max_pages_reached_is_partial_failed_not_finished(db_session):
    adapter = FakeFmpAdapter({0: [_row("AAA")], 1: [_row("BBB")]})

    result = _run_ingestion(db_session, adapter, max_pages=1)

    assert result.status == "partial_failed"
    assert not result.ok
    assert result.metrics["max_pages_reached"] is True
    assert result.metrics["pages_fetched"] == 1
    assert result.errors == [
        {
            "stage": "pagination",
            "error_type": "max_pages_reached",
            "message": (
                "FMP delisted-company ingestion reached max_pages before observing "
                "an empty terminal page"
            ),
            "max_pages": 1,
            "page_limit": 2,
        }
    ]


def test_cli_exit_code_requires_allow_partial_for_max_pages_reached():
    result = JobResult(
        status="partial_failed",
        metrics={"max_pages_reached": True, "fetch_error_count": 0},
        errors=[{"stage": "pagination", "error_type": "max_pages_reached"}],
    )

    assert _exit_code_for_result(result, allow_partial=False) == 1
    assert _exit_code_for_result(result, allow_partial=True) == 0


def test_idempotent_rerun_updates_existing_rows_instead_of_duplicates(db_session):
    adapter = FakeFmpAdapter(
        {
            0: [
                _row("AAA"),
                _row("BBB", exchange="NYSE", delisted_date="2024-06-07"),
            ],
        }
    )

    first = _run_ingestion(db_session, adapter)
    second = _run_ingestion(db_session, adapter)

    assert first.metrics["rows_inserted"] == 2
    assert second.metrics["rows_inserted"] == 0
    assert second.metrics["rows_updated"] == 2
    assert db_session.query(FmpDelistedCompanyRecord).count() == 2
    duplicate_groups = (
        db_session.query(
            FmpDelistedCompanyRecord.normalized_symbol,
            FmpDelistedCompanyRecord.exchange_key,
            FmpDelistedCompanyRecord.delisted_date_key,
            func.count().label("row_count"),
        )
        .group_by(
            FmpDelistedCompanyRecord.normalized_symbol,
            FmpDelistedCompanyRecord.exchange_key,
            FmpDelistedCompanyRecord.delisted_date_key,
        )
        .having(func.count() > 1)
        .count()
    )
    assert duplicate_groups == 0


def test_missing_optional_fields_are_persisted_with_explicit_keys(db_session):
    adapter = FakeFmpAdapter(
        {
            0: [
                _row(
                    "MISSING",
                    company_name=None,
                    exchange=None,
                    ipo_date=None,
                    delisted_date=None,
                )
            ],
        }
    )

    result = _run_ingestion(db_session, adapter)

    assert result.status == "finished"
    stored = db_session.query(FmpDelistedCompanyRecord).one()
    assert stored.symbol == "MISSING"
    assert stored.company_name is None
    assert stored.exchange is None
    assert stored.ipo_date is None
    assert stored.delisted_date is None
    assert stored.exchange_key == "UNKNOWN"
    assert stored.delisted_date_key == "UNKNOWN"
    assert stored.exchange_relevance_status == "non_us_or_unknown_exchange"


def test_malformed_rows_are_counted_and_not_persisted_as_good_rows(db_session):
    adapter = FakeFmpAdapter(
        {
            0: [
                _row("", delisted_date="2024-05-06"),
                _row("BADDATE", delisted_date="not-a-date"),
                _row("GOOD", delisted_date="2024-05-06"),
            ],
        }
    )

    result = _run_ingestion(db_session, adapter)

    assert result.status == "finished"
    assert result.metrics["malformed_rows"] == 2
    assert result.metrics["rows_skipped"] == 2
    assert db_session.query(FmpDelistedCompanyRecord).count() == 1
    assert db_session.query(FmpDelistedCompanyRecord).one().symbol == "GOOD"


def test_lineage_and_hashes_are_persisted(db_session):
    adapter = FakeFmpAdapter({0: [_row("AAA"), _row("BBB", exchange="NYSE")]})

    result = _run_ingestion(db_session, adapter)

    assert result.output_hashes["fmp_delisted_companies_rows"]
    missing = (
        db_session.query(FmpDelistedCompanyRecord)
        .filter(
            (FmpDelistedCompanyRecord.data_lineage_id.is_(None))
            | (FmpDelistedCompanyRecord.raw_payload_hash.is_(None))
        )
        .count()
    )
    assert missing == 0


def test_unique_constraint_prevents_direct_duplicate_inserts(db_session):
    adapter = FakeFmpAdapter({0: [_row("AAA")]})
    _run_ingestion(db_session, adapter)
    existing = db_session.query(FmpDelistedCompanyRecord).one()

    db_session.add(
        FmpDelistedCompanyRecord(
            symbol=existing.symbol,
            normalized_symbol=existing.normalized_symbol,
            company_name=existing.company_name,
            exchange=existing.exchange,
            exchange_key=existing.exchange_key,
            ipo_date=existing.ipo_date,
            delisted_date=existing.delisted_date,
            delisted_date_key=existing.delisted_date_key,
            page_number=existing.page_number,
            page_limit=existing.page_limit,
            exchange_relevance_status=existing.exchange_relevance_status,
            raw_payload_hash=existing.raw_payload_hash,
        )
    )

    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()
