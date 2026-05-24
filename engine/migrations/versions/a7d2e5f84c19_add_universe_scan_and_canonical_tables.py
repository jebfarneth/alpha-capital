"""add universe scan and canonical tables

Revision ID: a7d2e5f84c19
Revises: f2a43dc9b1e8
Create Date: 2026-05-24 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a7d2e5f84c19"
down_revision: Union[str, None] = "f2a43dc9b1e8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "universe_scans",
        sa.Column("scan_id", sa.String(), nullable=False),
        sa.Column("trading_date", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("asof_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider", sa.String(), nullable=True),
        sa.Column("raw_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("deduped_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("duplicate_symbol_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("included_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("excluded_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_lineage_hash", sa.String(), nullable=True),
        sa.Column("output_hash", sa.String(), nullable=True),
        sa.Column("run_status", sa.String(), nullable=False, server_default="finished"),
        sa.Column("metric_json", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("scan_id"),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
    )
    op.create_index("ix_universe_scans_trading_date", "universe_scans", ["trading_date"])
    op.create_index("ix_universe_scans_job_run_id", "universe_scans", ["job_run_id"])

    op.create_table(
        "canonical_universe_scans",
        sa.Column("trading_date", sa.String(), nullable=False),
        sa.Column("scan_id", sa.String(), nullable=False),
        sa.Column("selected_job_run_id", sa.String(), nullable=True),
        sa.Column("selected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("selection_reason", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("trading_date"),
        sa.ForeignKeyConstraint(["scan_id"], ["universe_scans.scan_id"]),
        sa.ForeignKeyConstraint(["selected_job_run_id"], ["evidence_job_runs.job_run_id"]),
    )

    # Existing rows predate universe_scans, so clear orphan scan references
    # before adding the FK. Old rows remain usable by universe_snapshot_id.
    op.execute(
        "UPDATE universe_snapshots SET scan_id = NULL "
        "WHERE scan_id IS NOT NULL "
        "AND scan_id NOT IN (SELECT scan_id FROM universe_scans)"
    )

    with op.batch_alter_table("universe_snapshots") as batch_op:
        batch_op.create_foreign_key(
            "fk_universe_snapshots_scan_id_universe_scans",
            "universe_scans",
            ["scan_id"],
            ["scan_id"],
        )

    op.create_index(
        "ux_universe_snapshots_scan_ticker",
        "universe_snapshots",
        ["scan_id", "ticker"],
        unique=True,
    )
    op.create_index(
        "ix_universe_snapshots_scan_inclusion",
        "universe_snapshots",
        ["scan_id", "operating_universe_inclusion"],
    )
    op.create_index(
        "ix_universe_snapshots_ticker_asof",
        "universe_snapshots",
        ["ticker", "asof_timestamp"],
    )


def downgrade() -> None:
    op.drop_index("ix_universe_snapshots_ticker_asof", table_name="universe_snapshots")
    op.drop_index("ix_universe_snapshots_scan_inclusion", table_name="universe_snapshots")
    op.drop_index("ux_universe_snapshots_scan_ticker", table_name="universe_snapshots")
    with op.batch_alter_table("universe_snapshots") as batch_op:
        batch_op.drop_constraint(
            "fk_universe_snapshots_scan_id_universe_scans",
            type_="foreignkey",
        )
    op.drop_table("canonical_universe_scans")
    op.drop_table("universe_scans")
