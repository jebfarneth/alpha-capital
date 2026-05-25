"""add detector orchestration signal columns

Revision ID: 7b9c2d4e6f01
Revises: e6f7a8c9d012
Create Date: 2026-05-25 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7b9c2d4e6f01"
down_revision: Union[str, None] = "e6f7a8c9d012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("signal_registry", sa.Column("trading_date", sa.String(), nullable=True))
    op.add_column("signal_registry", sa.Column("scan_id", sa.String(), nullable=True))
    op.add_column("signal_registry", sa.Column("detector_version", sa.String(), nullable=True))
    op.add_column("signal_registry", sa.Column("point_in_time_passed", sa.Boolean(), nullable=True))
    op.add_column("signal_registry", sa.Column("lookahead_guard_passed", sa.Boolean(), nullable=True))
    op.execute(
        "UPDATE signal_registry "
        "SET signal_identity_hash = 'legacy-' || signal_id "
        "WHERE signal_identity_hash IS NULL OR signal_identity_hash = ''"
    )

    with op.batch_alter_table("signal_registry") as batch_op:
        batch_op.alter_column(
            "signal_identity_hash",
            existing_type=sa.String(),
            nullable=False,
        )
        batch_op.create_foreign_key(
            "fk_signal_registry_scan_id_universe_scans",
            "universe_scans",
            ["scan_id"],
            ["scan_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("signal_registry") as batch_op:
        batch_op.drop_constraint(
            "fk_signal_registry_scan_id_universe_scans",
            type_="foreignkey",
        )
        batch_op.alter_column(
            "signal_identity_hash",
            existing_type=sa.String(),
            nullable=True,
        )
    op.drop_column("signal_registry", "lookahead_guard_passed")
    op.drop_column("signal_registry", "point_in_time_passed")
    op.drop_column("signal_registry", "detector_version")
    op.drop_column("signal_registry", "scan_id")
    op.drop_column("signal_registry", "trading_date")
