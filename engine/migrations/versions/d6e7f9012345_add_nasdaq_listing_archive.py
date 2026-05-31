"""add nasdaq listing archive tables

Revision ID: d6e7f9012345
Revises: c5d6e7f90123
Create Date: 2026-05-31 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d6e7f9012345"
down_revision: Union[str, None] = "c5d6e7f90123"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "nasdaq_listing_snapshots",
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("source_knowledge_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_payload_hash", sa.String(), nullable=False),
        sa.Column("raw_payload", sa.Text(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("parse_status", sa.String(), nullable=False),
        sa.Column("data_quality_flags_json", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("snapshot_id"),
        sa.UniqueConstraint(
            "source_type",
            "source_knowledge_timestamp",
            "raw_payload_hash",
            name="ux_nasdaq_listing_snapshot_source_time_hash",
        ),
    )
    op.create_index(
        "ix_nasdaq_listing_snapshots_source_time",
        "nasdaq_listing_snapshots",
        ["source_type", "source_knowledge_timestamp"],
    )
    op.create_table(
        "nasdaq_listing_snapshot_rows",
        sa.Column("snapshot_row_id", sa.String(), nullable=False),
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("normalized_symbol", sa.String(), nullable=False),
        sa.Column("security_name", sa.Text(), nullable=True),
        sa.Column("market", sa.String(), nullable=True),
        sa.Column("action", sa.String(), nullable=True),
        sa.Column("effective_date", sa.String(), nullable=True),
        sa.Column("reason_code", sa.String(), nullable=True),
        sa.Column("raw_json", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["snapshot_id"], ["nasdaq_listing_snapshots.snapshot_id"]
        ),
        sa.PrimaryKeyConstraint("snapshot_row_id"),
    )
    op.create_index(
        "ix_nasdaq_listing_snapshot_rows_symbol",
        "nasdaq_listing_snapshot_rows",
        ["source_type", "symbol"],
    )
    op.create_index(
        "ix_nasdaq_listing_snapshot_rows_effective",
        "nasdaq_listing_snapshot_rows",
        ["source_type", "effective_date"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_nasdaq_listing_snapshot_rows_effective",
        table_name="nasdaq_listing_snapshot_rows",
    )
    op.drop_index(
        "ix_nasdaq_listing_snapshot_rows_symbol",
        table_name="nasdaq_listing_snapshot_rows",
    )
    op.drop_table("nasdaq_listing_snapshot_rows")
    op.drop_index(
        "ix_nasdaq_listing_snapshots_source_time",
        table_name="nasdaq_listing_snapshots",
    )
    op.drop_table("nasdaq_listing_snapshots")
