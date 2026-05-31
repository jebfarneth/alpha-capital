"""add forward context path rows

Revision ID: c5d6e7f90123
Revises: b4c5d6e7f901
Create Date: 2026-05-30 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c5d6e7f90123"
down_revision: Union[str, None] = "b4c5d6e7f901"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "forward_context_path_rows",
        sa.Column("forward_context_path_row_id", sa.String(), nullable=False),
        sa.Column("signal_id", sa.String(), nullable=False),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("signal_horizon", sa.String(), nullable=True),
        sa.Column("forward_session_date", sa.String(), nullable=False),
        sa.Column("path_sequence", sa.Integer(), nullable=False),
        sa.Column("asof_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("context_json", sa.Text(), nullable=False),
        sa.Column("source_attempts_json", sa.Text(), nullable=False),
        sa.Column("data_lineage_ids", sa.Text(), nullable=False),
        sa.Column("context_hash", sa.String(), nullable=False),
        sa.Column("is_terminal_snapshot", sa.Boolean(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.ForeignKeyConstraint(["signal_id"], ["signal_registry.signal_id"]),
        sa.PrimaryKeyConstraint("forward_context_path_row_id"),
        sa.UniqueConstraint(
            "signal_id",
            "forward_session_date",
            name="ux_forward_context_path_rows_signal_session",
        ),
        sa.UniqueConstraint(
            "signal_id",
            "path_sequence",
            name="ux_forward_context_path_rows_signal_sequence",
        ),
    )
    op.create_index(
        "ix_forward_context_path_rows_signal_session",
        "forward_context_path_rows",
        ["signal_id", "forward_session_date"],
        unique=False,
    )
    op.create_index(
        "ix_forward_context_path_rows_pattern_ticker_session",
        "forward_context_path_rows",
        ["pattern_id", "ticker", "forward_session_date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_forward_context_path_rows_pattern_ticker_session",
        table_name="forward_context_path_rows",
    )
    op.drop_index(
        "ix_forward_context_path_rows_signal_session",
        table_name="forward_context_path_rows",
    )
    op.drop_table("forward_context_path_rows")
