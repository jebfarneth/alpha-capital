"""expand i12 fill log stage0 audit fields

Revision ID: fc0123456790
Revises: fb0123456789
Create Date: 2026-06-17 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "fc0123456790"
down_revision: Union[str, None] = "fb0123456789"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("i12_fill_log", sa.Column("feed", sa.String(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("model_selection_mode", sa.String(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("promotable_run", sa.Boolean(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("attempt_stage", sa.String(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("snapshot_status", sa.String(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("fire_status", sa.String(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("score_stage0_status", sa.String(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("selection_status", sa.String(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("quote_status", sa.String(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("exit_capture_status", sa.String(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("stage0_run_config_hash", sa.String(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("context_artifact_hash", sa.String(), nullable=True))
    op.add_column(
        "i12_fill_log",
        sa.Column("latest_trade_ts", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("i12_fill_log", sa.Column("latest_trade_age_seconds", sa.Float(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("minute_ts", sa.DateTime(timezone=True), nullable=True))
    op.add_column("i12_fill_log", sa.Column("minute_age_seconds", sa.Float(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("quote_ts", sa.DateTime(timezone=True), nullable=True))
    op.add_column("i12_fill_log", sa.Column("quote_age_seconds", sa.Float(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("half_day", sa.Boolean(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("session_minutes", sa.Integer(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("projection_basis", sa.String(), nullable=True))
    op.add_column(
        "i12_fill_log",
        sa.Column("snapshot_ts", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("i12_fill_log", sa.Column("snapshot_age_seconds", sa.Float(), nullable=True))
    op.add_column(
        "i12_fill_log",
        sa.Column("entry_quote_ts", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("i12_fill_log", sa.Column("entry_quote_age_seconds", sa.Float(), nullable=True))
    op.add_column("i12_fill_log", sa.Column("exit_quote_age_seconds", sa.Float(), nullable=True))
    op.add_column(
        "i12_fill_log",
        sa.Column("quote_condition_halt_inferred", sa.Boolean(), nullable=True),
    )
    op.add_column("i12_fill_log", sa.Column("coverage_error", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("i12_fill_log", "coverage_error")
    op.drop_column("i12_fill_log", "quote_condition_halt_inferred")
    op.drop_column("i12_fill_log", "exit_quote_age_seconds")
    op.drop_column("i12_fill_log", "entry_quote_age_seconds")
    op.drop_column("i12_fill_log", "entry_quote_ts")
    op.drop_column("i12_fill_log", "snapshot_age_seconds")
    op.drop_column("i12_fill_log", "snapshot_ts")
    op.drop_column("i12_fill_log", "exit_capture_status")
    op.drop_column("i12_fill_log", "quote_status")
    op.drop_column("i12_fill_log", "projection_basis")
    op.drop_column("i12_fill_log", "session_minutes")
    op.drop_column("i12_fill_log", "half_day")
    op.drop_column("i12_fill_log", "quote_age_seconds")
    op.drop_column("i12_fill_log", "quote_ts")
    op.drop_column("i12_fill_log", "minute_age_seconds")
    op.drop_column("i12_fill_log", "minute_ts")
    op.drop_column("i12_fill_log", "latest_trade_age_seconds")
    op.drop_column("i12_fill_log", "latest_trade_ts")
    op.drop_column("i12_fill_log", "context_artifact_hash")
    op.drop_column("i12_fill_log", "stage0_run_config_hash")
    op.drop_column("i12_fill_log", "selection_status")
    op.drop_column("i12_fill_log", "score_stage0_status")
    op.drop_column("i12_fill_log", "fire_status")
    op.drop_column("i12_fill_log", "snapshot_status")
    op.drop_column("i12_fill_log", "attempt_stage")
    op.drop_column("i12_fill_log", "promotable_run")
    op.drop_column("i12_fill_log", "model_selection_mode")
    op.drop_column("i12_fill_log", "feed")
