"""add security profiles table

Revision ID: d4b7c1e92f3a
Revises: a7d2e5f84c19
Create Date: 2026-05-24 12:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4b7c1e92f3a"
down_revision: Union[str, None] = "a7d2e5f84c19"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "security_profiles",
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("security_type", sa.String(), nullable=False, server_default="unknown"),
        sa.Column("source_provider", sa.String(), nullable=True),
        sa.Column("source_lineage_hash", sa.String(), nullable=True),
        sa.Column("profile_payload_hash", sa.String(), nullable=True),
        sa.Column("profile_asof_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_refreshed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("refresh_status", sa.String(), nullable=True),
        sa.Column("raw_profile_json", sa.Text(), nullable=True),
        sa.Column("classification_reason", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("symbol"),
    )
    op.create_index("ix_security_profiles_security_type", "security_profiles", ["security_type"])
    op.create_index("ix_security_profiles_last_refreshed_at", "security_profiles", ["last_refreshed_at"])


def downgrade() -> None:
    op.drop_index("ix_security_profiles_last_refreshed_at", table_name="security_profiles")
    op.drop_index("ix_security_profiles_security_type", table_name="security_profiles")
    op.drop_table("security_profiles")
