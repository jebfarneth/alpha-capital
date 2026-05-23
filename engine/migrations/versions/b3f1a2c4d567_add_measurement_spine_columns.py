"""add measurement spine columns to signal_registry

Revision ID: b3f1a2c4d567
Revises: 9e273a77556a
Create Date: 2026-05-23 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b3f1a2c4d567"
down_revision: Union[str, None] = "9e273a77556a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("signal_registry", sa.Column("signal_identity_hash", sa.String(), nullable=True))
    op.add_column("signal_registry", sa.Column("forward_return", sa.Float(), nullable=True))
    op.add_column("signal_registry", sa.Column("forward_return_status", sa.String(), nullable=True))
    op.add_column("signal_registry", sa.Column("outcome_unavailable_reason", sa.String(), nullable=True))
    op.add_column("signal_registry", sa.Column("intended_entry_price", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("signal_registry", "intended_entry_price")
    op.drop_column("signal_registry", "outcome_unavailable_reason")
    op.drop_column("signal_registry", "forward_return_status")
    op.drop_column("signal_registry", "forward_return")
    op.drop_column("signal_registry", "signal_identity_hash")
