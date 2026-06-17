"""add i12 fill log

Revision ID: fb0123456789
Revises: fa0123456789
Create Date: 2026-06-16 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "fb0123456789"
down_revision: Union[str, None] = "fa0123456789"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "i12_fill_log",
        sa.Column("i12_fill_log_id", sa.String(), nullable=False),
        sa.Column("signal_id", sa.String(), nullable=True),
        sa.Column("score_id", sa.String(), nullable=True),
        sa.Column("model_id", sa.String(), nullable=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("decision_date", sa.Date(), nullable=False),
        sa.Column("decision_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_capture_due_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ml_score", sa.Float(), nullable=True),
        sa.Column("score_source", sa.String(), nullable=True),
        sa.Column("score_status", sa.String(), nullable=True),
        sa.Column("fallback_reason", sa.String(), nullable=True),
        sa.Column("projected_vol_ratio", sa.Float(), nullable=True),
        sa.Column("gap", sa.Float(), nullable=True),
        sa.Column("off_52w_high", sa.Float(), nullable=True),
        sa.Column("bid", sa.Float(), nullable=True),
        sa.Column("ask", sa.Float(), nullable=True),
        sa.Column("spread_bps", sa.Float(), nullable=True),
        sa.Column("top_of_book_size", sa.Float(), nullable=True),
        sa.Column("intended_order_usd", sa.Float(), nullable=False),
        sa.Column("size_sufficient", sa.Boolean(), nullable=True),
        sa.Column("halted", sa.Boolean(), nullable=True),
        sa.Column("skipped_reason", sa.String(), nullable=False),
        sa.Column("exit_bid", sa.Float(), nullable=True),
        sa.Column("exit_ask", sa.Float(), nullable=True),
        sa.Column("exit_quote_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("modeled_return", sa.Float(), nullable=True),
        sa.Column("feature_json", sa.Text(), nullable=False),
        sa.Column("gate_values_json", sa.Text(), nullable=False),
        sa.Column("quote_json", sa.Text(), nullable=True),
        sa.Column("exit_quote_json", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(), nullable=False),
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
        sa.ForeignKeyConstraint(["model_id"], ["ml_model_registry.model_id"]),
        sa.ForeignKeyConstraint(["score_id"], ["signal_ml_scores.score_id"]),
        sa.ForeignKeyConstraint(["signal_id"], ["signal_registry.signal_id"]),
        sa.PrimaryKeyConstraint("i12_fill_log_id"),
        sa.UniqueConstraint(
            "content_hash",
            name="ux_i12_fill_log_content_hash",
        ),
    )
    op.create_index(
        "ix_i12_fill_log_decision_date",
        "i12_fill_log",
        ["decision_date", "skipped_reason"],
    )
    op.create_index(
        "ix_i12_fill_log_signal",
        "i12_fill_log",
        ["signal_id"],
    )
    op.create_index(
        "ix_i12_fill_log_ticker_decision",
        "i12_fill_log",
        ["ticker", "decision_ts"],
    )


def downgrade() -> None:
    op.drop_index("ix_i12_fill_log_ticker_decision", table_name="i12_fill_log")
    op.drop_index("ix_i12_fill_log_signal", table_name="i12_fill_log")
    op.drop_index("ix_i12_fill_log_decision_date", table_name="i12_fill_log")
    op.drop_table("i12_fill_log")
