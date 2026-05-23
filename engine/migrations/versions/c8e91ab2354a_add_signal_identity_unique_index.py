"""add unique signal identity index

Revision ID: c8e91ab2354a
Revises: b3f1a2c4d567
Create Date: 2026-05-23 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op


revision: str = "c8e91ab2354a"
down_revision: Union[str, None] = "b3f1a2c4d567"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ux_signal_registry_pattern_ticker_identity",
        "signal_registry",
        ["pattern_id", "ticker", "signal_identity_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ux_signal_registry_pattern_ticker_identity",
        table_name="signal_registry",
    )
