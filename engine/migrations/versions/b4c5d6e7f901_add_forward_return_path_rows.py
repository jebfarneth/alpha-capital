"""add forward return path rows

Revision ID: b4c5d6e7f901
Revises: a3b4c5d6e7f8
Create Date: 2026-05-29 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b4c5d6e7f901"
down_revision: Union[str, None] = "a3b4c5d6e7f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "forward_return_path_rows",
        sa.Column("forward_return_path_row_id", sa.String(), nullable=False),
        sa.Column("forward_return_observation_id", sa.String(), nullable=False),
        sa.Column("signal_id", sa.String(), nullable=False),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("signal_horizon", sa.String(), nullable=True),
        sa.Column("path_sequence", sa.Integer(), nullable=False),
        sa.Column("session_date", sa.String(), nullable=False),
        sa.Column("entry_session_date", sa.String(), nullable=True),
        sa.Column("exit_session_date", sa.String(), nullable=True),
        sa.Column("entry_price", sa.Float(), nullable=True),
        sa.Column("open_price", sa.Float(), nullable=True),
        sa.Column("high_price", sa.Float(), nullable=True),
        sa.Column("low_price", sa.Float(), nullable=True),
        sa.Column("close_price", sa.Float(), nullable=True),
        sa.Column("volume", sa.Float(), nullable=True),
        sa.Column("split_adjusted_close", sa.Float(), nullable=True),
        sa.Column("adj_close", sa.Float(), nullable=True),
        sa.Column("return_from_entry_open", sa.Float(), nullable=True),
        sa.Column("return_from_entry_high", sa.Float(), nullable=True),
        sa.Column("return_from_entry_low", sa.Float(), nullable=True),
        sa.Column("return_from_entry_close", sa.Float(), nullable=True),
        sa.Column("is_entry_session", sa.Boolean(), nullable=True),
        sa.Column("is_exit_session", sa.Boolean(), nullable=True),
        sa.Column("expected_session_count", sa.Integer(), nullable=True),
        sa.Column("path_status", sa.String(), nullable=True),
        sa.Column("is_synthetic_exit", sa.Boolean(), nullable=True),
        sa.Column("data_lineage_id", sa.String(), nullable=True),
        sa.Column("provider", sa.String(), nullable=True),
        sa.Column("endpoint", sa.String(), nullable=True),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("input_hash", sa.String(), nullable=True),
        sa.Column("outcome_hash", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["data_lineage_id"],
            ["data_lineage.data_lineage_id"],
        ),
        sa.ForeignKeyConstraint(
            ["forward_return_observation_id"],
            ["forward_return_observations.forward_return_observation_id"],
        ),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.ForeignKeyConstraint(["signal_id"], ["signal_registry.signal_id"]),
        sa.PrimaryKeyConstraint("forward_return_path_row_id"),
        sa.UniqueConstraint(
            "forward_return_observation_id",
            "session_date",
            name="ux_forward_return_path_rows_observation_session",
        ),
        sa.UniqueConstraint(
            "forward_return_observation_id",
            "path_sequence",
            name="ux_forward_return_path_rows_observation_sequence",
        ),
    )
    op.create_index(
        "ix_forward_return_path_rows_signal_session",
        "forward_return_path_rows",
        ["signal_id", "session_date"],
        unique=False,
    )
    op.create_index(
        "ix_forward_return_path_rows_pattern_ticker_session",
        "forward_return_path_rows",
        ["pattern_id", "ticker", "session_date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_forward_return_path_rows_pattern_ticker_session",
        table_name="forward_return_path_rows",
    )
    op.drop_index(
        "ix_forward_return_path_rows_signal_session",
        table_name="forward_return_path_rows",
    )
    op.drop_table("forward_return_path_rows")
