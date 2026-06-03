"""add m1 pead producer tables

Revision ID: e1f2a3b4c5d6
Revises: d6e7f9012345
Create Date: 2026-06-03 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, None] = "d6e7f9012345"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "m1_earnings_events",
        sa.Column("m1_earnings_event_id", sa.String(), nullable=False),
        sa.Column("scan_id", sa.String(), nullable=True),
        sa.Column("universe_snapshot_id", sa.String(), nullable=True),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("earnings_event_id", sa.String(), nullable=False),
        sa.Column("announcement_date", sa.String(), nullable=True),
        sa.Column("effective_announcement_session", sa.String(), nullable=True),
        sa.Column("announcement_time", sa.String(), nullable=True),
        sa.Column("fiscal_period_end", sa.String(), nullable=True),
        sa.Column("fiscal_year", sa.Integer(), nullable=True),
        sa.Column("fiscal_quarter", sa.Integer(), nullable=True),
        sa.Column("actual_eps", sa.Float(), nullable=True),
        sa.Column("estimated_eps", sa.Float(), nullable=True),
        sa.Column("expected_eps", sa.Float(), nullable=True),
        sa.Column("sigma_delta_eps", sa.Float(), nullable=True),
        sa.Column("sue_foster", sa.Float(), nullable=True),
        sa.Column("rho1", sa.Float(), nullable=True),
        sa.Column("sue_sign_current", sa.Integer(), nullable=True),
        sa.Column("sue_sign_prior", sa.Integer(), nullable=True),
        sa.Column("sue_streak_length", sa.Integer(), nullable=True),
        sa.Column("foster_history_quarters_used", sa.Integer(), nullable=False),
        sa.Column("split_adjustment_continuity_check", sa.String(), nullable=True),
        sa.Column("restatement_exposure", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("diagnostic_json", sa.Text(), nullable=True),
        sa.Column("sue_series_json", sa.Text(), nullable=True),
        sa.Column("data_lineage_ids", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.ForeignKeyConstraint(["scan_id"], ["universe_scans.scan_id"]),
        sa.ForeignKeyConstraint(
            ["universe_snapshot_id"],
            ["universe_snapshots.universe_snapshot_id"],
        ),
        sa.PrimaryKeyConstraint("m1_earnings_event_id"),
        sa.UniqueConstraint(
            "scan_id",
            "ticker",
            "earnings_event_id",
            name="ux_m1_earnings_events_scan_ticker_event",
        ),
    )
    op.create_index(
        "ix_m1_earnings_events_scan_status",
        "m1_earnings_events",
        ["scan_id", "status"],
    )
    op.create_index(
        "ix_m1_earnings_events_ticker_announcement",
        "m1_earnings_events",
        ["ticker", "announcement_date"],
    )

    op.create_table(
        "m1_friction_snapshots",
        sa.Column("m1_friction_snapshot_id", sa.String(), nullable=False),
        sa.Column("scan_id", sa.String(), nullable=True),
        sa.Column("universe_snapshot_id", sa.String(), nullable=True),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("market_factor_symbol", sa.String(), nullable=False),
        sa.Column("d1", sa.Float(), nullable=True),
        sa.Column("d1_decile", sa.Integer(), nullable=True),
        sa.Column("sigma_epsilon", sa.Float(), nullable=True),
        sa.Column("sigma_epsilon_percentile", sa.Float(), nullable=True),
        sa.Column("weekly_return_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("diagnostic_json", sa.Text(), nullable=True),
        sa.Column("data_lineage_ids", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.ForeignKeyConstraint(["scan_id"], ["universe_scans.scan_id"]),
        sa.ForeignKeyConstraint(
            ["universe_snapshot_id"],
            ["universe_snapshots.universe_snapshot_id"],
        ),
        sa.PrimaryKeyConstraint("m1_friction_snapshot_id"),
        sa.UniqueConstraint(
            "scan_id",
            "ticker",
            name="ux_m1_friction_snapshots_scan_ticker",
        ),
    )
    op.create_index(
        "ix_m1_friction_snapshots_scan_status",
        "m1_friction_snapshots",
        ["scan_id", "status"],
    )
    op.create_index(
        "ix_m1_friction_snapshots_d1_decile",
        "m1_friction_snapshots",
        ["scan_id", "d1_decile"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_m1_friction_snapshots_d1_decile",
        table_name="m1_friction_snapshots",
    )
    op.drop_index(
        "ix_m1_friction_snapshots_scan_status",
        table_name="m1_friction_snapshots",
    )
    op.drop_table("m1_friction_snapshots")
    op.drop_index(
        "ix_m1_earnings_events_ticker_announcement",
        table_name="m1_earnings_events",
    )
    op.drop_index(
        "ix_m1_earnings_events_scan_status",
        table_name="m1_earnings_events",
    )
    op.drop_table("m1_earnings_events")
