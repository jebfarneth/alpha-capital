"""add fmp delisted companies

Revision ID: 123456789abc
Revises: f0123456789a
Create Date: 2026-06-06 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "123456789abc"
down_revision: Union[str, None] = "f0123456789a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "fmp_delisted_companies",
        sa.Column("fmp_delisted_company_id", sa.String(), nullable=False),
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("normalized_symbol", sa.String(), nullable=False),
        sa.Column("company_name", sa.Text(), nullable=True),
        sa.Column("exchange", sa.String(), nullable=True),
        sa.Column("exchange_key", sa.String(), nullable=False),
        sa.Column("ipo_date", sa.Date(), nullable=True),
        sa.Column("delisted_date", sa.Date(), nullable=True),
        sa.Column("delisted_date_key", sa.String(), nullable=False),
        sa.Column("source", sa.String(), server_default="FMP", nullable=False),
        sa.Column(
            "source_endpoint",
            sa.String(),
            server_default="/stable/delisted-companies",
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("page_limit", sa.Integer(), nullable=False),
        sa.Column("page_row_index", sa.Integer(), nullable=True),
        sa.Column("row_status", sa.String(), server_default="active", nullable=False),
        sa.Column("exchange_relevance_status", sa.String(), nullable=False),
        sa.Column("raw_payload_hash", sa.String(), nullable=False),
        sa.Column("raw_payload_json", sa.Text(), nullable=True),
        sa.Column("request_metadata_json", sa.Text(), nullable=True),
        sa.Column("data_lineage_id", sa.String(), nullable=True),
        sa.Column("ingestion_job_run_id", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["data_lineage_id"], ["data_lineage.data_lineage_id"]),
        sa.ForeignKeyConstraint(
            ["ingestion_job_run_id"], ["evidence_job_runs.job_run_id"]
        ),
        sa.PrimaryKeyConstraint("fmp_delisted_company_id"),
        sa.UniqueConstraint(
            "normalized_symbol",
            "exchange_key",
            "delisted_date_key",
            name="ux_fmp_delisted_companies_symbol_exchange_delisted",
        ),
    )
    op.create_index(
        "ix_fmp_delisted_companies_symbol",
        "fmp_delisted_companies",
        ["normalized_symbol"],
    )
    op.create_index(
        "ix_fmp_delisted_companies_delisted_date",
        "fmp_delisted_companies",
        ["delisted_date"],
    )
    op.create_index(
        "ix_fmp_delisted_companies_exchange",
        "fmp_delisted_companies",
        ["exchange_key"],
    )
    op.create_index(
        "ix_fmp_delisted_companies_job_run",
        "fmp_delisted_companies",
        ["ingestion_job_run_id"],
    )
    op.create_index(
        "ix_fmp_delisted_companies_lineage",
        "fmp_delisted_companies",
        ["data_lineage_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_fmp_delisted_companies_lineage",
        table_name="fmp_delisted_companies",
    )
    op.drop_index(
        "ix_fmp_delisted_companies_job_run",
        table_name="fmp_delisted_companies",
    )
    op.drop_index(
        "ix_fmp_delisted_companies_exchange",
        table_name="fmp_delisted_companies",
    )
    op.drop_index(
        "ix_fmp_delisted_companies_delisted_date",
        table_name="fmp_delisted_companies",
    )
    op.drop_index(
        "ix_fmp_delisted_companies_symbol",
        table_name="fmp_delisted_companies",
    )
    op.drop_table("fmp_delisted_companies")
