"""add historical universe reconstruction

Revision ID: 3456789abcde
Revises: 23456789abcd
Create Date: 2026-06-06 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "3456789abcde"
down_revision: Union[str, None] = "23456789abcd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "historical_universe_reconstructions",
        sa.Column("historical_universe_reconstruction_id", sa.String(), nullable=False),
        sa.Column("replay_date", sa.Date(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("normalized_symbol", sa.String(), nullable=False),
        sa.Column("exchange", sa.String(), nullable=True),
        sa.Column("company_name", sa.Text(), nullable=True),
        sa.Column("ipo_date", sa.Date(), nullable=True),
        sa.Column("delisted_date", sa.Date(), nullable=True),
        sa.Column("inclusion_status", sa.String(), nullable=False),
        sa.Column("rejection_reason", sa.String(), nullable=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("source_provenance_json", sa.Text(), nullable=False),
        sa.Column("reconstructed", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("reconstruction_method", sa.String(), nullable=False),
        sa.Column("pit_filter_status_json", sa.Text(), nullable=False),
        sa.Column("current_universe_snapshot_id", sa.String(), nullable=True),
        sa.Column("fmp_delisted_company_id", sa.String(), nullable=True),
        sa.Column("data_lineage_id", sa.String(), nullable=True),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("input_hash", sa.String(), nullable=False),
        sa.Column("output_hash", sa.String(), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["current_universe_snapshot_id"],
            ["universe_snapshots.universe_snapshot_id"],
        ),
        sa.ForeignKeyConstraint(
            ["fmp_delisted_company_id"],
            ["fmp_delisted_companies.fmp_delisted_company_id"],
        ),
        sa.ForeignKeyConstraint(["data_lineage_id"], ["data_lineage.data_lineage_id"]),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.PrimaryKeyConstraint("historical_universe_reconstruction_id"),
        sa.UniqueConstraint(
            "replay_date",
            "normalized_symbol",
            name="ux_historical_universe_recon_date_symbol",
        ),
    )
    op.create_index(
        "ix_historical_universe_recon_date_status",
        "historical_universe_reconstructions",
        ["replay_date", "inclusion_status"],
    )
    op.create_index(
        "ix_historical_universe_recon_symbol_date",
        "historical_universe_reconstructions",
        ["normalized_symbol", "replay_date"],
    )
    op.create_index(
        "ix_historical_universe_recon_reason",
        "historical_universe_reconstructions",
        ["rejection_reason"],
    )
    op.create_index(
        "ix_historical_universe_recon_job_run",
        "historical_universe_reconstructions",
        ["job_run_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_historical_universe_recon_job_run",
        table_name="historical_universe_reconstructions",
    )
    op.drop_index(
        "ix_historical_universe_recon_reason",
        table_name="historical_universe_reconstructions",
    )
    op.drop_index(
        "ix_historical_universe_recon_symbol_date",
        table_name="historical_universe_reconstructions",
    )
    op.drop_index(
        "ix_historical_universe_recon_date_status",
        table_name="historical_universe_reconstructions",
    )
    op.drop_table("historical_universe_reconstructions")
