"""add fmp delisted replay indexes

Revision ID: 23456789abcd
Revises: 123456789abc
Create Date: 2026-06-06 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op


revision: str = "23456789abcd"
down_revision: Union[str, None] = "123456789abc"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_fmp_delisted_companies_ipo_date",
        "fmp_delisted_companies",
        ["ipo_date"],
    )
    op.create_index(
        "ix_fmp_delisted_companies_exchange_relevance",
        "fmp_delisted_companies",
        ["exchange_relevance_status"],
    )
    op.create_index(
        "ix_fmp_delisted_companies_replay_filter",
        "fmp_delisted_companies",
        ["exchange_relevance_status", "ipo_date", "delisted_date"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_fmp_delisted_companies_replay_filter",
        table_name="fmp_delisted_companies",
    )
    op.drop_index(
        "ix_fmp_delisted_companies_exchange_relevance",
        table_name="fmp_delisted_companies",
    )
    op.drop_index(
        "ix_fmp_delisted_companies_ipo_date",
        table_name="fmp_delisted_companies",
    )
