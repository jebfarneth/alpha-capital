"""add market path ml context fields

Revision ID: f0123456789a
Revises: e9f012345678
Create Date: 2026-06-06 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f0123456789a"
down_revision: Union[str, None] = "e9f012345678"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


ML_CONTEXT_COLUMNS = (
    ("universe_pct_above_sma_20d", sa.Float()),
    ("universe_pct_above_sma_50d", sa.Float()),
    ("universe_pct_making_20d_highs", sa.Float()),
    ("universe_pct_making_52w_highs", sa.Float()),
    ("volatility_regime_proxy", sa.Float()),
    ("volatility_regime_source", sa.String()),
    ("market_regime_status", sa.String()),
    ("opening_range_high_5m", sa.Float()),
    ("opening_range_low_5m", sa.Float()),
    ("opening_range_high_15m", sa.Float()),
    ("opening_range_low_15m", sa.Float()),
    ("opening_range_high_30m", sa.Float()),
    ("opening_range_low_30m", sa.Float()),
    ("opening_range_high_60m", sa.Float()),
    ("opening_range_low_60m", sa.Float()),
    ("first_5m_return", sa.Float()),
    ("first_15m_return", sa.Float()),
    ("first_30m_return", sa.Float()),
    ("first_60m_return", sa.Float()),
    ("intraday_vwap", sa.Float()),
    ("open_vs_intraday_vwap_pct", sa.Float()),
    ("close_vs_intraday_vwap_pct", sa.Float()),
    ("intraday_volume_5m", sa.Float()),
    ("intraday_volume_15m", sa.Float()),
    ("intraday_volume_30m", sa.Float()),
    ("intraday_volume_60m", sa.Float()),
    ("pct_expected_volume_5m", sa.Float()),
    ("pct_expected_volume_15m", sa.Float()),
    ("pct_expected_volume_30m", sa.Float()),
    ("pct_expected_volume_60m", sa.Float()),
    ("held_above_breakout_after_first_hour", sa.Boolean()),
    ("intraday_mfe_timestamp", sa.DateTime(timezone=True)),
    ("intraday_mae_timestamp", sa.DateTime(timezone=True)),
    ("t1_before_stop", sa.Boolean()),
    ("intraday_structure_status", sa.String()),
    ("missing_intraday_bars", sa.Boolean()),
    ("bid_ask_spread", sa.Float()),
    ("bid_ask_spread_pct", sa.Float()),
    ("quote_age_seconds", sa.Float()),
    ("bid_size", sa.Float()),
    ("ask_size", sa.Float()),
    ("intended_entry_vs_mid_pct", sa.Float()),
    ("intended_entry_vs_ask_pct", sa.Float()),
    ("intended_entry_vs_bid_pct", sa.Float()),
    ("volume_participation_pct", sa.Float()),
    ("halt_risk_flag", sa.Boolean()),
    ("offering_risk_flag", sa.Boolean()),
    ("missing_quote", sa.Boolean()),
    ("stale_quote", sa.Boolean()),
    ("quote_status", sa.String()),
    ("execution_quality_status", sa.String()),
    ("float_shares", sa.Float()),
    ("shares_outstanding", sa.Float()),
    ("turnover_float", sa.Float()),
    ("dollar_turnover_float", sa.Float()),
    ("short_volume_ratio", sa.Float()),
    ("short_interest_pct_float", sa.Float()),
    ("short_interest_shares", sa.Float()),
    ("short_interest_days_to_cover", sa.Float()),
    ("proxy_days_to_cover", sa.Float()),
    ("borrow_fee_rate", sa.Float()),
    ("float_source_status", sa.String()),
    ("short_source_status", sa.String()),
    ("borrow_fee_status", sa.String()),
    ("supply_squeeze_status", sa.String()),
    ("news_count_1d", sa.Integer()),
    ("news_count_5d", sa.Integer()),
    ("news_count_20d", sa.Integer()),
    ("news_catalyst_flags_json", sa.Text()),
    ("earnings_days_to_next", sa.Integer()),
    ("earnings_days_since_last", sa.Integer()),
    ("offering_flag", sa.Boolean()),
    ("atm_flag", sa.Boolean()),
    ("shelf_registration_flag", sa.Boolean()),
    ("insider_buy_overlap_m2", sa.Boolean()),
    ("cofire_m1", sa.Boolean()),
    ("cofire_m2", sa.Boolean()),
    ("cofire_m3", sa.Boolean()),
    ("cofire_m4", sa.Boolean()),
    ("cofire_i11", sa.Boolean()),
    ("fda_clinical_flag", sa.Boolean()),
    ("corporate_action_flag", sa.Boolean()),
    ("cross_pattern_overlap_count", sa.Integer()),
    ("strongest_overlap_pattern_id", sa.String()),
    ("catalyst_context_status", sa.String()),
    ("missing_catalyst_source", sa.Boolean()),
    ("rsi_2", sa.Float()),
    ("rsi_5", sa.Float()),
    ("rsi_14", sa.Float()),
    ("adx_14", sa.Float()),
    ("plus_di_14", sa.Float()),
    ("minus_di_14", sa.Float()),
    ("bollinger_bandwidth_20d", sa.Float()),
    ("bollinger_percent_b_20d", sa.Float()),
    ("keltner_channel_position_20d", sa.Float()),
    ("macd_histogram", sa.Float()),
    ("obv", sa.Float()),
    ("accumulation_distribution", sa.Float()),
    ("chaikin_money_flow_20d", sa.Float()),
    ("stochastic_oscillator_14d", sa.Float()),
    ("technical_indicator_status", sa.String()),
)


def upgrade() -> None:
    for name, column_type in ML_CONTEXT_COLUMNS:
        op.add_column("market_path_features", sa.Column(name, column_type, nullable=True))


def downgrade() -> None:
    for name, _column_type in reversed(ML_CONTEXT_COLUMNS):
        op.drop_column("market_path_features", name)
