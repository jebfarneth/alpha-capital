"""add historical replay lookup indexes

Revision ID: b5c6d7e8f901
Revises: 3456789abcde
Create Date: 2026-06-09 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op


revision: str = "b5c6d7e8f901"
down_revision: Union[str, None] = "3456789abcde"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        with op.get_context().autocommit_block():
            op.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "ix_data_lineage_raw_payload_hash "
                "ON data_lineage (raw_payload_hash)"
            )
            op.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "ix_feature_snapshots_pattern_asof_ticker_hash "
                "ON feature_snapshots (pattern_id, asof_timestamp, ticker, feature_hash)"
            )
        return

    op.create_index(
        "ix_data_lineage_raw_payload_hash",
        "data_lineage",
        ["raw_payload_hash"],
    )
    op.create_index(
        "ix_feature_snapshots_pattern_asof_ticker_hash",
        "feature_snapshots",
        ["pattern_id", "asof_timestamp", "ticker", "feature_hash"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        with op.get_context().autocommit_block():
            op.execute(
                "DROP INDEX CONCURRENTLY IF EXISTS "
                "ix_feature_snapshots_pattern_asof_ticker_hash"
            )
            op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_data_lineage_raw_payload_hash")
        return

    op.drop_index(
        "ix_feature_snapshots_pattern_asof_ticker_hash",
        table_name="feature_snapshots",
    )
    op.drop_index("ix_data_lineage_raw_payload_hash", table_name="data_lineage")
