"""add universe lineage replay tables

Revision ID: 0f4c6e8a9b21
Revises: 7b9c2d4e6f01
Create Date: 2026-05-25 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0f4c6e8a9b21"
down_revision: Union[str, None] = "7b9c2d4e6f01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "data_lineage",
        sa.Column("raw_payload_json", sa.Text(), nullable=True),
    )
    op.add_column(
        "universe_snapshots",
        sa.Column("country", sa.String(), nullable=True),
    )
    op.create_table(
        "security_profile_scan_snapshots",
        sa.Column(
            "profile_scan_snapshot_id",
            sa.String(),
            nullable=False,
        ),
        sa.Column("scan_id", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column(
            "profile_required",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("cache_status", sa.String(), nullable=False),
        sa.Column("stale", sa.Boolean(), nullable=True),
        sa.Column("security_type", sa.String(), nullable=True),
        sa.Column("refresh_status", sa.String(), nullable=True),
        sa.Column("classification_reason", sa.String(), nullable=True),
        sa.Column("classifier_version", sa.String(), nullable=True),
        sa.Column("classification_input_hash", sa.String(), nullable=True),
        sa.Column("classification_output_hash", sa.String(), nullable=True),
        sa.Column("source_lineage_hash", sa.String(), nullable=True),
        sa.Column("profile_payload_hash", sa.String(), nullable=True),
        sa.Column("profile_asof_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_refreshed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("raw_profile_json", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.ForeignKeyConstraint(["scan_id"], ["universe_scans.scan_id"]),
        sa.PrimaryKeyConstraint("profile_scan_snapshot_id"),
    )
    op.create_index(
        "ix_security_profile_scan_snapshots_scan_required",
        "security_profile_scan_snapshots",
        ["scan_id", "profile_required"],
        unique=False,
    )
    op.create_index(
        "ux_security_profile_scan_snapshots_scan_symbol",
        "security_profile_scan_snapshots",
        ["scan_id", "symbol"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ux_security_profile_scan_snapshots_scan_symbol",
        table_name="security_profile_scan_snapshots",
    )
    op.drop_index(
        "ix_security_profile_scan_snapshots_scan_required",
        table_name="security_profile_scan_snapshots",
    )
    op.drop_table("security_profile_scan_snapshots")
    op.drop_column("universe_snapshots", "country")
    op.drop_column("data_lineage", "raw_payload_json")
