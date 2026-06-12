"""add paper execution events

Revision ID: c6d7e8f90123
Revises: a4b5c6d7e8f9
Create Date: 2026-06-11 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c6d7e8f90123"
down_revision: Union[str, None] = "a4b5c6d7e8f9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "paper_execution_events",
        sa.Column("paper_execution_event_id", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=False),
        sa.Column("gate_values_json", sa.Text(), nullable=True),
        sa.Column("event_payload_json", sa.Text(), nullable=True),
        sa.Column("data_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("wall_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decision_price", sa.Float(), nullable=True),
        sa.Column("broker_order_id", sa.String(), nullable=True),
        sa.Column("client_order_id", sa.String(), nullable=True),
        sa.Column("fill_price", sa.Float(), nullable=True),
        sa.Column("fill_qty", sa.Float(), nullable=True),
        sa.Column("lineage_hash", sa.String(), nullable=True),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("paper_execution_event_id"),
        sa.UniqueConstraint(
            "content_hash",
            name="ux_paper_execution_events_content_hash",
        ),
    )
    op.create_index(
        "ix_paper_execution_events_pattern_ticker_time",
        "paper_execution_events",
        ["pattern_id", "ticker", "wall_timestamp"],
    )
    op.create_index(
        "ix_paper_execution_events_event_type",
        "paper_execution_events",
        ["event_type"],
    )
    op.create_index(
        "ix_paper_execution_events_client_order",
        "paper_execution_events",
        ["client_order_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_paper_execution_events_client_order",
        table_name="paper_execution_events",
    )
    op.drop_index(
        "ix_paper_execution_events_event_type",
        table_name="paper_execution_events",
    )
    op.drop_index(
        "ix_paper_execution_events_pattern_ticker_time",
        table_name="paper_execution_events",
    )
    op.drop_table("paper_execution_events")
