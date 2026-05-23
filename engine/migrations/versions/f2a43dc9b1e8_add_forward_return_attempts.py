"""add forward return attempts

Revision ID: f2a43dc9b1e8
Revises: c8e91ab2354a
Create Date: 2026-05-23 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f2a43dc9b1e8"
down_revision: Union[str, None] = "c8e91ab2354a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "signal_registry",
        sa.Column(
            "forward_return_attempts",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("signal_registry", "forward_return_attempts")
