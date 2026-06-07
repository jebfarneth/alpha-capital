"""add rich market path features

Revision ID: d8e9f0123456
Revises: c7d8e9f01234
Create Date: 2026-06-06 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d8e9f0123456"
down_revision: Union[str, None] = "c7d8e9f01234"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


RICH_COLUMNS = (
    ("prior_52w_high", sa.Float),
    ("breakout_extension_pct", sa.Float),
    ("open_vs_52w_high_pct", sa.Float),
    ("close_vs_52w_high_pct", sa.Float),
    ("high_vs_52w_high_pct", sa.Float),
    ("gap_over_breakout", sa.Boolean),
    ("closed_above_breakout", sa.Boolean),
    ("close_location_value", sa.Float),
    ("upper_wick_ratio", sa.Float),
    ("lower_wick_ratio", sa.Float),
    ("true_range_pct", sa.Float),
    ("atr_14_pct", sa.Float),
    ("range_expansion_vs_20d", sa.Float),
    ("volume_zscore_20d", sa.Float),
    ("volume_zscore_60d", sa.Float),
    ("dollar_volume_zscore_20d", sa.Float),
    ("dollar_volume_zscore_60d", sa.Float),
    ("volume_acceleration_1d_vs_5d", sa.Float),
    ("volume_acceleration_1d_vs_20d", sa.Float),
    ("realized_volatility_5d", sa.Float),
    ("realized_volatility_10d", sa.Float),
    ("realized_volatility_20d", sa.Float),
    ("base_range_10d", sa.Float),
    ("base_range_20d", sa.Float),
    ("base_range_60d", sa.Float),
    ("base_max_drawdown_10d", sa.Float),
    ("base_max_drawdown_20d", sa.Float),
    ("base_max_drawdown_60d", sa.Float),
    ("distance_from_sma_20d", sa.Float),
    ("distance_from_sma_50d", sa.Float),
    ("distance_from_sma_200d", sa.Float),
    ("momentum_5d", sa.Float),
    ("momentum_20d", sa.Float),
    ("momentum_60d", sa.Float),
    ("prior_52w_high_touches_20d", sa.Integer),
    ("prior_52w_high_touches_60d", sa.Integer),
    ("prior_52w_high_touches_126d", sa.Integer),
    ("age_of_52w_high_sessions", sa.Integer),
    ("failed_breakout_count_20d", sa.Integer),
    ("failed_breakout_count_60d", sa.Integer),
    ("failed_breakout_count_126d", sa.Integer),
    ("vwap", sa.Float),
    ("open_vs_vwap_pct", sa.Float),
    ("high_vs_vwap_pct", sa.Float),
    ("low_vs_vwap_pct", sa.Float),
    ("close_vs_vwap_pct", sa.Float),
)


def upgrade() -> None:
    for name, column_type in RICH_COLUMNS:
        op.add_column("market_path_features", sa.Column(name, column_type(), nullable=True))


def downgrade() -> None:
    for name, _column_type in reversed(RICH_COLUMNS):
        op.drop_column("market_path_features", name)
