"""add market path v3 relative features

Revision ID: e9f012345678
Revises: d8e9f0123456
Create Date: 2026-06-06 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e9f012345678"
down_revision: Union[str, None] = "d8e9f0123456"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


V3_COLUMNS = (
    ("dollar_volume_rank", sa.Integer),
    ("dollar_volume_percentile", sa.Float),
    ("volume_expansion_20d_rank", sa.Integer),
    ("volume_expansion_20d_percentile", sa.Float),
    ("volume_expansion_60d_rank", sa.Integer),
    ("volume_expansion_60d_percentile", sa.Float),
    ("dollar_volume_expansion_20d_rank", sa.Integer),
    ("dollar_volume_expansion_20d_percentile", sa.Float),
    ("dollar_volume_expansion_60d_rank", sa.Integer),
    ("dollar_volume_expansion_60d_percentile", sa.Float),
    ("liquidity_proxy_rank", sa.Integer),
    ("liquidity_proxy_percentile", sa.Float),
    ("cohort_feature_row_count", sa.Integer),
    ("cohort_pattern_row_count", sa.Integer),
    ("spy_return_1d", sa.Float),
    ("spy_return_5d", sa.Float),
    ("spy_return_20d", sa.Float),
    ("spy_return_60d", sa.Float),
    ("qqq_return_1d", sa.Float),
    ("qqq_return_5d", sa.Float),
    ("qqq_return_20d", sa.Float),
    ("qqq_return_60d", sa.Float),
    ("iwm_return_1d", sa.Float),
    ("iwm_return_5d", sa.Float),
    ("iwm_return_20d", sa.Float),
    ("iwm_return_60d", sa.Float),
    ("relative_strength_vs_spy_5d", sa.Float),
    ("relative_strength_vs_spy_20d", sa.Float),
    ("relative_strength_vs_spy_60d", sa.Float),
    ("relative_strength_vs_qqq_5d", sa.Float),
    ("relative_strength_vs_qqq_20d", sa.Float),
    ("relative_strength_vs_qqq_60d", sa.Float),
    ("relative_strength_vs_iwm_5d", sa.Float),
    ("relative_strength_vs_iwm_20d", sa.Float),
    ("relative_strength_vs_iwm_60d", sa.Float),
    ("sector_etf", sa.String),
    ("sector_etf_return_5d", sa.Float),
    ("sector_etf_return_20d", sa.Float),
    ("sector_etf_return_60d", sa.Float),
    ("relative_strength_vs_sector_5d", sa.Float),
    ("relative_strength_vs_sector_20d", sa.Float),
    ("relative_strength_vs_sector_60d", sa.Float),
    ("sector_source", sa.String),
    ("sector_relative_status", sa.String),
)


def upgrade() -> None:
    for name, column_type in V3_COLUMNS:
        op.add_column("market_path_features", sa.Column(name, column_type(), nullable=True))


def downgrade() -> None:
    for name, _column_type in reversed(V3_COLUMNS):
        op.drop_column("market_path_features", name)
