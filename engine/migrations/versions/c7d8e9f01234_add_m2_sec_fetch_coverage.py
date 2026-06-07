"""add m2 sec fetch coverage

Revision ID: c7d8e9f01234
Revises: b6c7d8e9f012
Create Date: 2026-06-06 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c7d8e9f01234"
down_revision: Union[str, None] = "b6c7d8e9f012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "m2_sec_fetch_coverage",
        sa.Column("m2_sec_fetch_coverage_id", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("issuer_cik", sa.String(), nullable=False),
        sa.Column("from_date", sa.String(), nullable=False),
        sa.Column("to_date", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("transaction_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("scan_id", sa.String(), nullable=True),
        sa.Column("universe_snapshot_id", sa.String(), nullable=True),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("data_lineage_id", sa.String(), nullable=True),
        sa.Column("raw_payload_hash", sa.String(), nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
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
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.ForeignKeyConstraint(["scan_id"], ["universe_scans.scan_id"]),
        sa.ForeignKeyConstraint(
            ["universe_snapshot_id"],
            ["universe_snapshots.universe_snapshot_id"],
        ),
        sa.PrimaryKeyConstraint("m2_sec_fetch_coverage_id"),
        sa.UniqueConstraint(
            "ticker",
            "issuer_cik",
            "from_date",
            name="ux_m2_sec_fetch_coverage_ticker_cik_from",
        ),
    )
    op.create_index(
        "ix_m2_sec_fetch_coverage_job_run",
        "m2_sec_fetch_coverage",
        ["job_run_id"],
    )
    op.create_index(
        "ix_m2_sec_fetch_coverage_ticker_from",
        "m2_sec_fetch_coverage",
        ["ticker", "from_date"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_m2_sec_fetch_coverage_ticker_from",
        table_name="m2_sec_fetch_coverage",
    )
    op.drop_index(
        "ix_m2_sec_fetch_coverage_job_run",
        table_name="m2_sec_fetch_coverage",
    )
    op.drop_table("m2_sec_fetch_coverage")
