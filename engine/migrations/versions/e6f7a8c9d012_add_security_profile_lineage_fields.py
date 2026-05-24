"""add security profile lineage fields

Revision ID: e6f7a8c9d012
Revises: d4b7c1e92f3a
Create Date: 2026-05-24 13:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e6f7a8c9d012"
down_revision: Union[str, None] = "d4b7c1e92f3a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "security_profiles",
        sa.Column("classification_input_hash", sa.String(), nullable=True),
    )
    op.add_column(
        "security_profiles",
        sa.Column("classification_output_hash", sa.String(), nullable=True),
    )
    op.add_column(
        "security_profiles",
        sa.Column("classifier_version", sa.String(), nullable=True),
    )
    op.add_column(
        "universe_scans",
        sa.Column("security_profile_cache_hash", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("universe_scans", "security_profile_cache_hash")
    op.drop_column("security_profiles", "classifier_version")
    op.drop_column("security_profiles", "classification_output_hash")
    op.drop_column("security_profiles", "classification_input_hash")
