"""add market path features

Revision ID: b6c7d8e9f012
Revises: f3a4b5c6d7e8
Create Date: 2026-06-05 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b6c7d8e9f012"
down_revision: Union[str, None] = "f3a4b5c6d7e8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "market_path_features",
        sa.Column("market_path_feature_id", sa.String(), nullable=False),
        sa.Column("signal_id", sa.String(), nullable=False),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("signal_horizon", sa.String(), nullable=True),
        sa.Column("signal_date", sa.String(), nullable=False),
        sa.Column("entry_session_date", sa.String(), nullable=True),
        sa.Column("feature_session_date", sa.String(), nullable=False),
        sa.Column("path_sequence", sa.Integer(), nullable=False),
        sa.Column("feature_role", sa.String(), nullable=False),
        sa.Column("feature_version", sa.String(), nullable=False),
        sa.Column("asof_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reconstruction_method", sa.String(), nullable=False),
        sa.Column("previous_close", sa.Float(), nullable=True),
        sa.Column("open_price", sa.Float(), nullable=True),
        sa.Column("high_price", sa.Float(), nullable=True),
        sa.Column("low_price", sa.Float(), nullable=True),
        sa.Column("close_price", sa.Float(), nullable=True),
        sa.Column("volume", sa.Float(), nullable=True),
        sa.Column("split_adjusted_close", sa.Float(), nullable=True),
        sa.Column("adj_close", sa.Float(), nullable=True),
        sa.Column("dollar_volume", sa.Float(), nullable=True),
        sa.Column("median_volume_20d", sa.Float(), nullable=True),
        sa.Column("median_volume_60d", sa.Float(), nullable=True),
        sa.Column("median_dollar_volume_20d", sa.Float(), nullable=True),
        sa.Column("median_dollar_volume_60d", sa.Float(), nullable=True),
        sa.Column("volume_expansion_20d", sa.Float(), nullable=True),
        sa.Column("volume_expansion_60d", sa.Float(), nullable=True),
        sa.Column("dollar_volume_expansion_20d", sa.Float(), nullable=True),
        sa.Column("dollar_volume_expansion_60d", sa.Float(), nullable=True),
        sa.Column("gap_pct", sa.Float(), nullable=True),
        sa.Column("open_to_close_return", sa.Float(), nullable=True),
        sa.Column("high_from_open_return", sa.Float(), nullable=True),
        sa.Column("low_from_open_return", sa.Float(), nullable=True),
        sa.Column("return_from_entry_open", sa.Float(), nullable=True),
        sa.Column("return_from_entry_high", sa.Float(), nullable=True),
        sa.Column("return_from_entry_low", sa.Float(), nullable=True),
        sa.Column("return_from_entry_close", sa.Float(), nullable=True),
        sa.Column("sigma_20d", sa.Float(), nullable=True),
        sa.Column("effective_hard_stop_pct", sa.Float(), nullable=True),
        sa.Column("liquidity_proxy_score", sa.Float(), nullable=True),
        sa.Column("liquidity_proxy_passed", sa.Boolean(), nullable=True),
        sa.Column("opening_range_json", sa.Text(), nullable=True),
        sa.Column("intraday_continuation_json", sa.Text(), nullable=True),
        sa.Column("quote_spread_json", sa.Text(), nullable=True),
        sa.Column("feature_json", sa.Text(), nullable=False),
        sa.Column("source_provider", sa.String(), nullable=False),
        sa.Column("source_endpoint", sa.String(), nullable=False),
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
        sa.ForeignKeyConstraint(["data_lineage_id"], ["data_lineage.data_lineage_id"]),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.ForeignKeyConstraint(["signal_id"], ["signal_registry.signal_id"]),
        sa.PrimaryKeyConstraint("market_path_feature_id"),
        sa.UniqueConstraint(
            "signal_id",
            "feature_session_date",
            "feature_version",
            name="ux_market_path_features_signal_session_version",
        ),
    )
    op.create_index(
        "ix_market_path_features_signal_session",
        "market_path_features",
        ["signal_id", "feature_session_date"],
    )
    op.create_index(
        "ix_market_path_features_pattern_ticker_session",
        "market_path_features",
        ["pattern_id", "ticker", "feature_session_date"],
    )
    op.create_index(
        "ix_market_path_features_role_version",
        "market_path_features",
        ["feature_role", "feature_version"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_market_path_features_role_version",
        table_name="market_path_features",
    )
    op.drop_index(
        "ix_market_path_features_pattern_ticker_session",
        table_name="market_path_features",
    )
    op.drop_index(
        "ix_market_path_features_signal_session",
        table_name="market_path_features",
    )
    op.drop_table("market_path_features")
