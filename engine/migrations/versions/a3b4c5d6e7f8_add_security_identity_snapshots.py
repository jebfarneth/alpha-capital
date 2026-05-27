"""add security identity snapshots table

Revision ID: a3b4c5d6e7f8
Revises: 1d2c3b4a5f60
Create Date: 2026-05-27 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a3b4c5d6e7f8"
down_revision: Union[str, None] = "1d2c3b4a5f60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "security_identity_snapshots",
        sa.Column("security_identity_snapshot_id", sa.String(), nullable=False),
        sa.Column("scan_id", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("cik", sa.String(), nullable=True),
        sa.Column("composite_figi", sa.String(), nullable=True),
        sa.Column("share_class_figi", sa.String(), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=True),
        sa.Column("delisted_utc", sa.String(), nullable=True),
        sa.Column("list_date", sa.String(), nullable=True),
        sa.Column("polygon_type", sa.String(), nullable=True),
        sa.Column("polygon_market", sa.String(), nullable=True),
        sa.Column("polygon_locale", sa.String(), nullable=True),
        sa.Column("polygon_primary_exchange", sa.String(), nullable=True),
        sa.Column("polygon_name", sa.String(), nullable=True),
        sa.Column("sic_code", sa.String(), nullable=True),
        sa.Column("sic_description", sa.String(), nullable=True),
        sa.Column("ticker_events_json", sa.Text(), nullable=True),
        sa.Column("identity_status", sa.String(), nullable=False),
        sa.Column("identity_reason", sa.String(), nullable=True),
        sa.Column("identity_hash", sa.String(), nullable=True),
        sa.Column("source_provider", sa.String(), nullable=True),
        sa.Column("source_endpoint", sa.String(), nullable=True),
        sa.Column("data_lineage_id", sa.String(), nullable=True),
        sa.Column("events_data_lineage_id", sa.String(), nullable=True),
        sa.Column("data_lineage_ids", sa.Text(), nullable=True),
        sa.Column("raw_payload_hash", sa.String(), nullable=True),
        sa.Column("asof_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["data_lineage_id"], ["data_lineage.data_lineage_id"]),
        sa.ForeignKeyConstraint(["events_data_lineage_id"], ["data_lineage.data_lineage_id"]),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.ForeignKeyConstraint(["scan_id"], ["universe_scans.scan_id"]),
        sa.PrimaryKeyConstraint("security_identity_snapshot_id"),
    )
    op.create_index(
        "ux_security_identity_snapshots_scan_ticker",
        "security_identity_snapshots",
        ["scan_id", "ticker"],
        unique=True,
    )
    op.create_index(
        "ix_security_identity_snapshots_cik",
        "security_identity_snapshots",
        ["cik"],
        unique=False,
    )
    op.create_index(
        "ix_security_identity_snapshots_composite_figi",
        "security_identity_snapshots",
        ["composite_figi"],
        unique=False,
    )
    op.create_index(
        "ix_security_identity_snapshots_share_class_figi",
        "security_identity_snapshots",
        ["share_class_figi"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_security_identity_snapshots_share_class_figi",
        table_name="security_identity_snapshots",
    )
    op.drop_index(
        "ix_security_identity_snapshots_composite_figi",
        table_name="security_identity_snapshots",
    )
    op.drop_index(
        "ix_security_identity_snapshots_cik",
        table_name="security_identity_snapshots",
    )
    op.drop_index(
        "ux_security_identity_snapshots_scan_ticker",
        table_name="security_identity_snapshots",
    )
    op.drop_table("security_identity_snapshots")
