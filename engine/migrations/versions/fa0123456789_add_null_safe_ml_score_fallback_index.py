"""add null safe ml score fallback index

Revision ID: fa0123456789
Revises: f90123456789
Create Date: 2026-06-14 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op


revision: str = "fa0123456789"
down_revision: Union[str, None] = "f90123456789"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


INDEX_NAME = "ux_signal_ml_scores_fallback_null_model"


def upgrade() -> None:
    op.execute(
        f"""
        CREATE UNIQUE INDEX {INDEX_NAME}
        ON signal_ml_scores (
            signal_id,
            COALESCE(requested_model_id, ''),
            score_status
        )
        WHERE model_id IS NULL
        """
    )


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="signal_ml_scores")
