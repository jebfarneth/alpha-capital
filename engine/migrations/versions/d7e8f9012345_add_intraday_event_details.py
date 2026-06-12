"""add intraday event details

Revision ID: d7e8f9012345
Revises: c6d7e8f90123
Create Date: 2026-06-12 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d7e8f9012345"
down_revision: Union[str, None] = "c6d7e8f90123"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "intraday_event_details",
        sa.Column("intraday_event_detail_id", sa.String(), nullable=False),
        sa.Column("signal_id", sa.String(), nullable=True),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("trading_date", sa.Date(), nullable=False),
        sa.Column("outcome", sa.String(), nullable=False),
        sa.Column("event_identity_hash", sa.String(), nullable=False),
        sa.Column("input_hash", sa.String(), nullable=False),
        sa.Column("output_hash", sa.String(), nullable=False),
        sa.Column("data_lineage_ids_json", sa.Text(), nullable=True),
        sa.Column("gate_values_json", sa.Text(), nullable=True),
        sa.Column("feature_json", sa.Text(), nullable=True),
        sa.Column("label_json", sa.Text(), nullable=True),
        sa.Column("artifact_flags_json", sa.Text(), nullable=True),
        sa.Column("quarantine_reason", sa.String(), nullable=True),
        sa.Column("confirmation_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("entry_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exit_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("conf_minute", sa.Integer(), nullable=True),
        sa.Column("entry_minute", sa.Integer(), nullable=True),
        sa.Column("entry_price", sa.Float(), nullable=True),
        sa.Column("exit_price", sa.Float(), nullable=True),
        sa.Column("session_open_price", sa.Float(), nullable=True),
        sa.Column("session_close_price", sa.Float(), nullable=True),
        sa.Column("next_open_price", sa.Float(), nullable=True),
        sa.Column("projected_vol_at_conf", sa.Float(), nullable=True),
        sa.Column("projected_vol_ratio_at_conf", sa.Float(), nullable=True),
        sa.Column("full_day_volume_ratio", sa.Float(), nullable=True),
        sa.Column("chase_pct", sa.Float(), nullable=True),
        sa.Column("gap_pct", sa.Float(), nullable=True),
        sa.Column("distance_from_max252", sa.Float(), nullable=True),
        sa.Column("ret_conf", sa.Float(), nullable=True),
        sa.Column("ret_open_close", sa.Float(), nullable=True),
        sa.Column(
            "ret_open_close_leaky_research_only",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("ret_next_open", sa.Float(), nullable=True),
        sa.Column("mae_pct", sa.Float(), nullable=True),
        sa.Column("mfe_pct", sa.Float(), nullable=True),
        sa.Column("halted", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "sub_dollar_at_open",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "split_basis_mismatch",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "is_ml_excluded",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("ml_exclusion_reason", sa.String(), nullable=False),
        sa.Column("security_type", sa.String(), nullable=False),
        sa.Column("sessions_to_delist", sa.Integer(), nullable=True),
        sa.Column(
            "sessions_to_delist_not_pit",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
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
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.ForeignKeyConstraint(["signal_id"], ["signal_registry.signal_id"]),
        sa.PrimaryKeyConstraint("intraday_event_detail_id"),
        sa.UniqueConstraint(
            "event_identity_hash",
            name="ux_intraday_event_details_identity",
        ),
    )
    op.create_index(
        "ix_intraday_event_details_pattern_date",
        "intraday_event_details",
        ["pattern_id", "trading_date"],
    )
    op.create_index(
        "ix_intraday_event_details_pattern_ticker_date",
        "intraday_event_details",
        ["pattern_id", "ticker", "trading_date"],
    )
    op.create_index(
        "ix_intraday_event_details_signal_id",
        "intraday_event_details",
        ["signal_id"],
    )
    op.create_index(
        "ix_intraday_event_details_outcome",
        "intraday_event_details",
        ["outcome"],
    )
    op.create_index(
        "ux_signal_registry_i12_pattern_ticker_trading_date",
        "signal_registry",
        ["pattern_id", "ticker", "trading_date"],
        unique=True,
        sqlite_where=sa.text("pattern_id = 'I12'"),
        postgresql_where=sa.text("pattern_id = 'I12'"),
    )


def downgrade() -> None:
    op.drop_index(
        "ux_signal_registry_i12_pattern_ticker_trading_date",
        table_name="signal_registry",
    )
    op.drop_index(
        "ix_intraday_event_details_outcome",
        table_name="intraday_event_details",
    )
    op.drop_index(
        "ix_intraday_event_details_signal_id",
        table_name="intraday_event_details",
    )
    op.drop_index(
        "ix_intraday_event_details_pattern_ticker_date",
        table_name="intraday_event_details",
    )
    op.drop_index(
        "ix_intraday_event_details_pattern_date",
        table_name="intraday_event_details",
    )
    op.drop_table("intraday_event_details")
