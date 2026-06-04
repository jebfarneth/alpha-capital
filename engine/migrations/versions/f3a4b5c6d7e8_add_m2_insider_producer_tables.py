"""add m2 insider producer tables

Revision ID: f3a4b5c6d7e8
Revises: e1f2a3b4c5d6
Create Date: 2026-06-04 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f3a4b5c6d7e8"
down_revision: Union[str, None] = "e1f2a3b4c5d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "m2_insider_transactions",
        sa.Column("transaction_id", sa.String(), nullable=False),
        sa.Column("scan_id", sa.String(), nullable=True),
        sa.Column("universe_snapshot_id", sa.String(), nullable=True),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("source_authority", sa.String(), nullable=False),
        sa.Column("enrichment_sources", sa.Text(), nullable=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("issuer_cik", sa.String(), nullable=True),
        sa.Column("issuer_name", sa.Text(), nullable=True),
        sa.Column("insider_id", sa.String(), nullable=False),
        sa.Column("insider_cik", sa.String(), nullable=True),
        sa.Column("insider_name", sa.Text(), nullable=True),
        sa.Column("issuer_state", sa.String(), nullable=True),
        sa.Column("insider_state", sa.String(), nullable=True),
        sa.Column("identity_resolution_method", sa.String(), nullable=False),
        sa.Column("identity_resolution_confidence", sa.Float(), nullable=False),
        sa.Column("filing_accession_number", sa.String(), nullable=True),
        sa.Column("filing_form", sa.String(), nullable=True),
        sa.Column("filing_date", sa.String(), nullable=True),
        sa.Column("filing_accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("filing_detected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("first_tradable_session", sa.String(), nullable=True),
        sa.Column("clock_quality", sa.String(), nullable=False),
        sa.Column("transaction_date", sa.String(), nullable=True),
        sa.Column("transaction_code", sa.String(), nullable=True),
        sa.Column("transaction_code_description", sa.Text(), nullable=True),
        sa.Column("acquired_disposed_code", sa.String(), nullable=True),
        sa.Column("transaction_shares", sa.Float(), nullable=True),
        sa.Column("transaction_price_per_share", sa.Float(), nullable=True),
        sa.Column("transaction_notional_usd", sa.Float(), nullable=True),
        sa.Column("purchase_notional_usd", sa.Float(), nullable=True),
        sa.Column("market_cap_usd", sa.Float(), nullable=True),
        sa.Column("ownership_type", sa.String(), nullable=True),
        sa.Column("insider_roles_json", sa.Text(), nullable=True),
        sa.Column("is_open_market_purchase", sa.Boolean(), nullable=False),
        sa.Column("is_buy", sa.Boolean(), nullable=False),
        sa.Column("is_sell", sa.Boolean(), nullable=False),
        sa.Column("is_10b5_1", sa.Boolean(), nullable=True),
        sa.Column("sec_fmp_mismatch", sa.Boolean(), nullable=False),
        sa.Column("data_lineage_ids", sa.Text(), nullable=True),
        sa.Column("raw_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.ForeignKeyConstraint(["scan_id"], ["universe_scans.scan_id"]),
        sa.ForeignKeyConstraint(
            ["universe_snapshot_id"],
            ["universe_snapshots.universe_snapshot_id"],
        ),
        sa.PrimaryKeyConstraint("transaction_id"),
    )
    op.create_index(
        "ix_m2_transactions_accession",
        "m2_insider_transactions",
        ["filing_accession_number"],
    )
    op.create_index(
        "ix_m2_transactions_insider_year",
        "m2_insider_transactions",
        ["insider_id", "transaction_date"],
    )
    op.create_index(
        "ix_m2_transactions_source",
        "m2_insider_transactions",
        ["source_authority"],
    )
    op.create_index(
        "ix_m2_transactions_ticker_tradable",
        "m2_insider_transactions",
        ["ticker", "first_tradable_session"],
    )

    op.create_table(
        "m2_insider_classifications",
        sa.Column("m2_insider_classification_id", sa.String(), nullable=False),
        sa.Column("insider_id", sa.String(), nullable=False),
        sa.Column("insider_cik", sa.String(), nullable=True),
        sa.Column("insider_name", sa.Text(), nullable=True),
        sa.Column("calendar_year", sa.Integer(), nullable=False),
        sa.Column("classification", sa.String(), nullable=False),
        sa.Column("routine_month", sa.Integer(), nullable=True),
        sa.Column("prior_year_count", sa.Integer(), nullable=False),
        sa.Column("data_cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("basis_json", sa.Text(), nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("m2_insider_classification_id"),
        sa.UniqueConstraint(
            "insider_id",
            "calendar_year",
            name="ux_m2_classifications_insider_year",
        ),
    )
    op.create_index(
        "ix_m2_classifications_year_class",
        "m2_insider_classifications",
        ["calendar_year", "classification"],
    )

    op.create_table(
        "m2_cluster_members",
        sa.Column("m2_cluster_member_id", sa.String(), nullable=False),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("m2_cluster_id", sa.String(), nullable=False),
        sa.Column("m2_cluster_signature_hash", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("transaction_id", sa.String(), nullable=False),
        sa.Column("filing_accession_number", sa.String(), nullable=True),
        sa.Column("insider_id", sa.String(), nullable=False),
        sa.Column("insider_cik", sa.String(), nullable=True),
        sa.Column("classification", sa.String(), nullable=False),
        sa.Column("first_tradable_session", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["transaction_id"],
            ["m2_insider_transactions.transaction_id"],
        ),
        sa.PrimaryKeyConstraint("m2_cluster_member_id"),
        sa.UniqueConstraint(
            "pattern_id",
            "m2_cluster_id",
            "transaction_id",
            name="ux_m2_cluster_members_pattern_cluster_transaction",
        ),
    )
    op.create_index(
        "ix_m2_cluster_members_accession",
        "m2_cluster_members",
        ["filing_accession_number"],
    )
    op.create_index(
        "ix_m2_cluster_members_cluster",
        "m2_cluster_members",
        ["pattern_id", "m2_cluster_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_m2_cluster_members_cluster", table_name="m2_cluster_members")
    op.drop_index("ix_m2_cluster_members_accession", table_name="m2_cluster_members")
    op.drop_table("m2_cluster_members")
    op.drop_index(
        "ix_m2_classifications_year_class",
        table_name="m2_insider_classifications",
    )
    op.drop_table("m2_insider_classifications")
    op.drop_index(
        "ix_m2_transactions_ticker_tradable",
        table_name="m2_insider_transactions",
    )
    op.drop_index("ix_m2_transactions_source", table_name="m2_insider_transactions")
    op.drop_index(
        "ix_m2_transactions_insider_year",
        table_name="m2_insider_transactions",
    )
    op.drop_index(
        "ix_m2_transactions_accession",
        table_name="m2_insider_transactions",
    )
    op.drop_table("m2_insider_transactions")
