"""add market path pre-signal context

Revision ID: e8f901234567
Revises: d7e8f9012345
Create Date: 2026-06-13 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e8f901234567"
down_revision: Union[str, None] = "d7e8f9012345"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "market_path_pre_signal_contexts",
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("feature_session_date", sa.Date(), nullable=False),
        sa.Column("feature_role", sa.String(), nullable=False),
        sa.Column("feature_version", sa.String(), nullable=False),
        sa.Column("row_status", sa.String(), nullable=False),
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
        sa.Column("sub_dollar", sa.Boolean(), nullable=True),
        sa.Column("median_volume_20d", sa.Float(), nullable=True),
        sa.Column("median_dollar_volume_20d", sa.Float(), nullable=True),
        sa.Column("volume_expansion_20d", sa.Float(), nullable=True),
        sa.Column("return_1d", sa.Float(), nullable=True),
        sa.Column("return_5d", sa.Float(), nullable=True),
        sa.Column("return_20d", sa.Float(), nullable=True),
        sa.Column("sigma_20d", sa.Float(), nullable=True),
        sa.Column("range_contraction_ratio_60d", sa.Float(), nullable=True),
        sa.Column("volume_trend_slope_60d", sa.Float(), nullable=True),
        sa.Column("base_depth_60d", sa.Float(), nullable=True),
        sa.Column("base_length_60d", sa.Integer(), nullable=True),
        sa.Column("off_low252", sa.Float(), nullable=True),
        sa.Column("dist_hi252", sa.Float(), nullable=True),
        sa.Column("rank_status", sa.String(), nullable=False),
        sa.Column(
            "retroactive_adjustment_caveat",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "conditional_on_fire",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column("feature_json", sa.Text(), nullable=False),
        sa.Column("status_json", sa.Text(), nullable=True),
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
        sa.PrimaryKeyConstraint(
            "ticker",
            "feature_session_date",
            "feature_role",
            "feature_version",
            name="pk_market_path_pre_signal_contexts",
        ),
    )
    op.create_index(
        "ix_market_path_pre_signal_contexts_date_status",
        "market_path_pre_signal_contexts",
        ["feature_session_date", "row_status"],
    )
    op.create_index(
        "ix_market_path_pre_signal_contexts_job_run",
        "market_path_pre_signal_contexts",
        ["job_run_id"],
    )
    op.create_index(
        "ix_market_path_pre_signal_contexts_role_version",
        "market_path_pre_signal_contexts",
        ["feature_role", "feature_version"],
    )

    op.create_table(
        "market_path_pre_signal_links",
        sa.Column("signal_id", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("signal_date", sa.Date(), nullable=False),
        sa.Column("feature_session_date", sa.Date(), nullable=False),
        sa.Column("relative_session_index", sa.Integer(), nullable=False),
        sa.Column("feature_role", sa.String(), nullable=False),
        sa.Column("feature_version", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["ticker", "feature_session_date", "feature_role", "feature_version"],
            [
                "market_path_pre_signal_contexts.ticker",
                "market_path_pre_signal_contexts.feature_session_date",
                "market_path_pre_signal_contexts.feature_role",
                "market_path_pre_signal_contexts.feature_version",
            ],
        ),
        sa.PrimaryKeyConstraint(
            "signal_id",
            "feature_session_date",
            "feature_role",
            "feature_version",
            name="pk_market_path_pre_signal_links",
        ),
    )
    op.create_index(
        "ix_market_path_pre_signal_links_pattern_signal_date",
        "market_path_pre_signal_links",
        ["pattern_id", "signal_date"],
    )
    op.create_index(
        "ix_market_path_pre_signal_links_ticker_date",
        "market_path_pre_signal_links",
        ["ticker", "feature_session_date"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_market_path_pre_signal_links_ticker_date",
        table_name="market_path_pre_signal_links",
    )
    op.drop_index(
        "ix_market_path_pre_signal_links_pattern_signal_date",
        table_name="market_path_pre_signal_links",
    )
    op.drop_table("market_path_pre_signal_links")
    op.drop_index(
        "ix_market_path_pre_signal_contexts_role_version",
        table_name="market_path_pre_signal_contexts",
    )
    op.drop_index(
        "ix_market_path_pre_signal_contexts_job_run",
        table_name="market_path_pre_signal_contexts",
    )
    op.drop_index(
        "ix_market_path_pre_signal_contexts_date_status",
        table_name="market_path_pre_signal_contexts",
    )
    op.drop_table("market_path_pre_signal_contexts")
