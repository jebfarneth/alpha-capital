"""add m3 sector rotation producer tables

Revision ID: a4b5c6d7e8f9
Revises: f3a4b5c6d7e8
Create Date: 2026-06-05 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a4b5c6d7e8f9"
down_revision: Union[str, None] = "f3a4b5c6d7e8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "firm_sector_assignments_history",
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("valid_from", sa.Date(), nullable=False),
        sa.Column("sector", sa.String(), nullable=False),
        sa.Column("sic_code", sa.String(), nullable=True),
        sa.Column("sic_description", sa.Text(), nullable=True),
        sa.Column("industry", sa.String(), nullable=True),
        sa.Column("source", sa.String(), server_default="POLYGON_SIC", nullable=False),
        sa.Column("sic_to_sector_map_version", sa.String(), nullable=False),
        sa.Column("valid_to", sa.Date(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("ticker", "valid_from"),
    )
    op.create_index(
        "ix_firm_sector_history_ticker_interval",
        "firm_sector_assignments_history",
        ["ticker", "valid_from", "valid_to"],
    )
    op.create_index(
        "ix_firm_sector_history_sector_interval",
        "firm_sector_assignments_history",
        ["sector", "valid_from", "valid_to"],
    )
    op.create_index(
        "ix_firm_sector_history_source",
        "firm_sector_assignments_history",
        ["source"],
    )
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")
        op.execute(
            "ALTER TABLE firm_sector_assignments_history "
            "ADD CONSTRAINT ex_firm_sector_history_no_overlap "
            "EXCLUDE USING gist "
            "(ticker WITH =, daterange(valid_from, valid_to, '[)') WITH &&)"
        )

    op.create_table(
        "firm_sector_assignments",
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("sector", sa.String(), nullable=False),
        sa.Column("industry", sa.String(), nullable=True),
        sa.Column("source", sa.String(), server_default="POLYGON_SIC", nullable=False),
        sa.Column("classification_date", sa.Date(), nullable=False),
        sa.Column("last_verified", sa.Date(), nullable=False),
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
        sa.PrimaryKeyConstraint("ticker"),
    )
    op.create_index(
        "ix_firm_sector_assignments_sector",
        "firm_sector_assignments",
        ["sector"],
    )
    op.create_index(
        "ix_firm_sector_assignments_last_verified",
        "firm_sector_assignments",
        ["last_verified"],
    )

    op.create_table(
        "sector_returns_daily",
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("sector", sa.String(), nullable=False),
        sa.Column("return_6mo", sa.Float(), nullable=False),
        sa.Column("return_6mo_ew", sa.Float(), nullable=True),
        sa.Column("return_1mo", sa.Float(), nullable=True),
        sa.Column("return_3mo", sa.Float(), nullable=True),
        sa.Column("sector_rank", sa.Integer(), nullable=False),
        sa.Column("sector_rank_normalized", sa.Float(), nullable=False),
        sa.Column("n_sectors", sa.Integer(), nullable=False),
        sa.Column("n_firms_in_sector", sa.Integer(), nullable=False),
        sa.Column("total_market_cap_in_sector", sa.Float(), nullable=False),
        sa.Column("source", sa.String(), server_default="POLYGON_SIC", nullable=False),
        sa.Column("sic_to_sector_map_version", sa.String(), nullable=False),
        sa.Column("formation_date", sa.Date(), nullable=False),
        sa.Column(
            "point_in_time_passed",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "formation_cohort_passed",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("sector_history_coverage_years", sa.Float(), nullable=True),
        sa.Column(
            "delisting_shumway_adjustment_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "delisting_unknown_review_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("delisting_adjustment_audit_json", sa.Text(), nullable=True),
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
        sa.PrimaryKeyConstraint("date", "sector"),
    )
    op.create_index(
        "ix_sector_returns_daily_rank",
        "sector_returns_daily",
        ["date", "sector_rank"],
    )
    op.create_index(
        "ix_sector_returns_daily_sector",
        "sector_returns_daily",
        ["sector"],
    )

    op.create_table(
        "sector_change_log",
        sa.Column("sector_change_log_id", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("old_sector", sa.String(), nullable=True),
        sa.Column("new_sector", sa.String(), nullable=False),
        sa.Column("old_sic_code", sa.String(), nullable=True),
        sa.Column("new_sic_code", sa.String(), nullable=True),
        sa.Column("old_source", sa.String(), nullable=True),
        sa.Column("new_source", sa.String(), nullable=False),
        sa.Column("sic_to_sector_map_version", sa.String(), nullable=False),
        sa.Column("change_date", sa.Date(), nullable=False),
        sa.Column(
            "detected_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("diagnostic_json", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.PrimaryKeyConstraint("sector_change_log_id"),
    )
    op.create_index(
        "ix_sector_change_log_ticker_date",
        "sector_change_log",
        ["ticker", "change_date"],
    )
    op.create_index(
        "ix_sector_change_log_job_run_id",
        "sector_change_log",
        ["job_run_id"],
    )

    op.create_table(
        "m3_validation_metadata",
        sa.Column("signal_id", sa.String(), nullable=False),
        sa.Column("candidate_id", sa.String(), nullable=False),
        sa.Column("position_id", sa.String(), nullable=True),
        sa.Column("entry_fill_price", sa.Float(), nullable=True),
        sa.Column("entry_filled_qty", sa.Float(), nullable=False),
        sa.Column("entry_avg_fill_price", sa.Float(), nullable=True),
        sa.Column("entry_fill_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fill_quality", sa.Integer(), nullable=False),
        sa.Column("realized_slippage_bps", sa.Float(), nullable=True),
        sa.Column("position_size_usd", sa.Float(), nullable=True),
        sa.Column("tcb_max_position_pct", sa.Float(), nullable=True),
        sa.Column("optimizer_max_position_pct", sa.Float(), nullable=True),
        sa.Column("thesis_category", sa.String(), nullable=False),
        sa.Column("t1_hit", sa.Boolean(), nullable=False),
        sa.Column("t1_hit_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("t1_hit_price", sa.Float(), nullable=True),
        sa.Column("t2_hit", sa.Boolean(), nullable=False),
        sa.Column("t2_hit_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("t2_hit_price", sa.Float(), nullable=True),
        sa.Column("t3_hit", sa.Boolean(), nullable=False),
        sa.Column("t3_hit_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("t3_hit_price", sa.Float(), nullable=True),
        sa.Column("trailing_stop_active", sa.Boolean(), nullable=False),
        sa.Column("trailing_stop_triggered", sa.Boolean(), nullable=False),
        sa.Column("trailing_stop_trigger_price", sa.Float(), nullable=True),
        sa.Column("hard_stop_triggered", sa.Boolean(), nullable=False),
        sa.Column("time_barrier_triggered", sa.Boolean(), nullable=False),
        sa.Column("stop_state", sa.String(), nullable=False),
        sa.Column("lifecycle_state", sa.String(), nullable=False),
        sa.Column("emergency_flatten_at_close", sa.Boolean(), nullable=False),
        sa.Column("current_stop_price", sa.Float(), nullable=True),
        sa.Column("t1_stop_adjustment_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("t2_stop_adjustment_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("order_group_id", sa.String(), nullable=True),
        sa.Column("framework_exit_cleanup_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("framework_exit_cleanup_failure_reason", sa.Text(), nullable=True),
        sa.Column("parent_entry_order_id", sa.String(), nullable=True),
        sa.Column("last_order_replace_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("race_condition_resolution", sa.String(), nullable=True),
        sa.Column("stbm_saga_state", sa.String(), nullable=True),
        sa.Column("market_flatten_order_id", sa.String(), nullable=True),
        sa.Column("market_flatten_attempts", sa.Integer(), nullable=False),
        sa.Column("temp_protective_stop_order_id", sa.String(), nullable=True),
        sa.Column("temp_protective_stop_active", sa.Boolean(), nullable=False),
        sa.Column("temp_protective_stop_cancel_recreate_count", sa.Integer(), nullable=False),
        sa.Column("temp_protective_stop_filled_during_entry", sa.Boolean(), nullable=False),
        sa.Column("temp_protective_stop_fill_during_entry_race", sa.Boolean(), nullable=False),
        sa.Column("pending_temp_stop_update_active", sa.Boolean(), nullable=False),
        sa.Column("pending_temp_stop_update_target_price", sa.Float(), nullable=True),
        sa.Column("pending_temp_stop_update_target_qty", sa.Float(), nullable=True),
        sa.Column("entry_partial_fill", sa.Boolean(), nullable=False),
        sa.Column("entry_target_qty", sa.Float(), nullable=True),
        sa.Column("entry_fill_ratio", sa.Float(), nullable=True),
        sa.Column("entry_unfilled_cancel_reason", sa.String(), nullable=True),
        sa.Column("minimum_size_gate_failure", sa.Boolean(), nullable=False),
        sa.Column("minimum_size_gate_failure_reason", sa.String(), nullable=True),
        sa.Column("failed_tranche_label", sa.String(), nullable=True),
        sa.Column("session_close_state", sa.Text(), nullable=True),
        sa.Column("session_boundary_replays", sa.Integer(), nullable=True),
        sa.Column("last_session_open_processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exit_fill_price", sa.Float(), nullable=True),
        sa.Column("exit_fill_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("final_exit_reason", sa.String(), nullable=True),
        sa.Column("universe_ejection_reason", sa.String(), nullable=True),
        sa.Column("hold_duration_trading_days", sa.Integer(), nullable=True),
        sa.Column("realized_return", sa.Float(), nullable=True),
        sa.Column("mae", sa.Float(), nullable=True),
        sa.Column("mfe", sa.Float(), nullable=True),
        sa.Column("realized_mfe_pct", sa.Float(), nullable=False),
        sa.Column("realized_mfe_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("optimizer_rebalance_rejected_convexity_lock", sa.Boolean(), nullable=False),
        sa.Column("optimizer_rebalance_rejection_count", sa.Integer(), nullable=False),
        sa.Column("high_since_t2", sa.Float(), nullable=True),
        sa.Column("n_firms_at_ranking_time", sa.Integer(), nullable=True),
        sa.Column("sparse_sector_excluded", sa.Boolean(), nullable=False),
        sa.Column("sparse_sector_warning", sa.Boolean(), nullable=False),
        sa.Column("counterfactual_benchmark_return", sa.Float(), nullable=True),
        sa.Column("counterfactual_sector_etf_return", sa.Float(), nullable=True),
        sa.Column("illiq_premium_contribution", sa.Float(), nullable=True),
        sa.Column("regime_contribution", sa.Float(), nullable=True),
        sa.Column("sigma_epsilon_contribution", sa.Float(), nullable=True),
        sa.Column("residual_m3_alpha", sa.Float(), nullable=True),
        sa.Column("override_affected", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint("fill_quality IN (0, 1)", name="ck_m3_validation_metadata_fill_quality"),
        sa.CheckConstraint(
            "thesis_category IN ('right_tail_convex', 'continuation', 'mean_reversion', 'event_drift')",
            name="ck_m3_validation_metadata_thesis_category",
        ),
        sa.CheckConstraint(
            "stop_state IN ('pre_T1', 'post_T1_BE', 'post_T2_floor', 'post_T2_trailing', 'closed', 'unfilled_expired')",
            name="ck_m3_validation_metadata_stop_state",
        ),
        sa.CheckConstraint(
            "lifecycle_state IN ('PRE_ENTRY', 'EXIT_ORDER_SETUP_PENDING', 'ACTIVE_PRE_T1', 'ACTIVE_POST_T1_BE', 'ACTIVE_POST_T2_FLOOR', 'ACTIVE_POST_T2_TRAILING', 'NATIVE_STOP_FLATTEN_REQUESTED', 'FRAMEWORK_EXIT_REQUESTED', 'OCO_CANCEL_PENDING', 'MARKET_FLATTEN_SUBMITTED', 'BROKER_FLAT_CONFIRMED', 'CLOSE_RECONCILIATION_PENDING', 'CLOSED')",
            name="ck_m3_validation_metadata_lifecycle_state",
        ),
        sa.CheckConstraint(
            "final_exit_reason IS NULL OR final_exit_reason IN ('T3_ceiling_hit', 'trailing_stop', 'hard_stop', 'break_even_stop', 'floor_stop', 'time_barrier', 'setup_failure', 'universe_ejection', 'circuit_breaker_flatten', 'optimizer_rebalance')",
            name="ck_m3_validation_metadata_final_exit_reason",
        ),
        sa.CheckConstraint(
            "race_condition_resolution IS NULL OR race_condition_resolution IN ('T1_T2_simultaneous', 'full_blowoff', 'T1_and_stop_simultaneous', 'stop_during_replace', 'stop_during_trailing_update', 'open_gap_through_stop', 'open_gap_through_target', 'multi_target_gap_up', 'gap_through_stop_down', 'position_divergence', 'tranche_state_anomaly', 'oco_state_drift', 'stop_invalid_at_submit', 'websocket_disconnect_reconciled', 'network_failure_escalated', 'framework_exit_during_native_fill', 't1_fill_stop_adjustment', 'emergency_flatten_cancel_failure', 'fill_during_temp_stop_race')",
            name="ck_m3_validation_metadata_race_condition_resolution",
        ),
        sa.CheckConstraint(
            "stbm_saga_state IS NULL OR stbm_saga_state IN ('FRAMEWORK_EXIT_REQUESTED', 'NATIVE_STOP_FLATTEN_REQUESTED', 'OCO_CANCEL_PENDING', 'MARKET_FLATTEN_SUBMITTED', 'BROKER_FLAT_CONFIRMED', 'CLOSED')",
            name="ck_m3_validation_metadata_stbm_saga_state",
        ),
        sa.CheckConstraint(
            "entry_unfilled_cancel_reason IS NULL OR entry_unfilled_cancel_reason IN ('close_cutoff_cancel_confirmed', 'close_cutoff_cancel_unconfirmed', 'intraday_cancel_complete', 'session_close_done_for_day')",
            name="ck_m3_validation_metadata_entry_unfilled_cancel_reason",
        ),
        sa.CheckConstraint(
            "minimum_size_gate_failure_reason IS NULL OR minimum_size_gate_failure_reason IN ('tranche_notional_below_$1', 'terminal_tranche_below_$5', 'fractional_below_minimum')",
            name="ck_m3_validation_metadata_minimum_size_gate_failure_reason",
        ),
        sa.CheckConstraint(
            "universe_ejection_reason IS NULL OR universe_ejection_reason IN ('liquidity_score_zero', 'market_cap_out_of_band', 'delisting_announced', 'fractionability_lost', 'manual_operator_eject', 'corporate_action_ineligible')",
            name="ck_m3_validation_metadata_universe_ejection_reason",
        ),
        sa.ForeignKeyConstraint(["candidate_id"], ["trade_candidates.candidate_id"]),
        sa.ForeignKeyConstraint(["signal_id"], ["signal_registry.signal_id"]),
        sa.PrimaryKeyConstraint("signal_id"),
        sa.UniqueConstraint(
            "position_id",
            name="ux_m3_validation_metadata_position_id_fk",
        ),
    )
    op.create_index(
        "idx_m3_validation_metadata_position_id_unique",
        "m3_validation_metadata",
        ["position_id"],
        unique=True,
        sqlite_where=sa.text("position_id IS NOT NULL"),
        postgresql_where=sa.text("position_id IS NOT NULL"),
    )

    op.create_table(
        "m3_tranche_fills",
        sa.Column("fill_id", sa.String(), nullable=False),
        sa.Column("signal_id", sa.String(), nullable=False),
        sa.Column("position_id", sa.String(), nullable=False),
        sa.Column("oco_group_id", sa.String(), nullable=False),
        sa.Column("tranche_label", sa.String(), nullable=False),
        sa.Column("fill_quantity", sa.Float(), nullable=False),
        sa.Column("fill_price", sa.Float(), nullable=False),
        sa.Column("fill_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fill_type", sa.String(), nullable=False),
        sa.Column("current_stop_level", sa.Float(), nullable=True),
        sa.CheckConstraint(
            "tranche_label IN ('T1', 'T2', 'T3', 'hard_stop', 'break_even_stop', 'floor_stop', 'trailing_stop', 'time_barrier', 'universe_ejection', 'circuit_breaker_flatten', 'optimizer_rebalance')",
            name="ck_m3_tranche_fills_tranche_label",
        ),
        sa.CheckConstraint(
            "fill_type IN ('take_profit', 'protective_stop', 'framework_exit', 'time_barrier_exit')",
            name="ck_m3_tranche_fills_fill_type",
        ),
        sa.ForeignKeyConstraint(["signal_id"], ["signal_registry.signal_id"]),
        sa.ForeignKeyConstraint(["position_id"], ["m3_validation_metadata.position_id"]),
        sa.PrimaryKeyConstraint("fill_id"),
    )
    op.create_index("ix_m3_tranche_fills_signal_id", "m3_tranche_fills", ["signal_id"])
    op.create_index("ix_m3_tranche_fills_position_id", "m3_tranche_fills", ["position_id"])

    op.create_table(
        "m3_oco_leg_state",
        sa.Column("leg_state_id", sa.String(), nullable=False),
        sa.Column("signal_id", sa.String(), nullable=False),
        sa.Column("position_id", sa.String(), nullable=False),
        sa.Column("oco_group_id", sa.String(), nullable=False),
        sa.Column("tranche_label", sa.String(), nullable=False),
        sa.Column("leg_role", sa.String(), nullable=False),
        sa.Column("broker_order_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("filled_qty", sa.Float(), nullable=True),
        sa.Column("remaining_qty", sa.Float(), nullable=False),
        sa.Column("avg_fill_price", sa.Float(), nullable=True),
        sa.Column("intended_price", sa.Float(), nullable=False),
        sa.Column("actual_price", sa.Float(), nullable=True),
        sa.Column("last_broker_update", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint("tranche_label IN ('T1', 'T2', 'T3')", name="ck_m3_oco_leg_state_tranche_label"),
        sa.CheckConstraint("leg_role IN ('take_profit', 'stop')", name="ck_m3_oco_leg_state_leg_role"),
        sa.CheckConstraint(
            "status IN ('new', 'accepted', 'partially_filled', 'filled', 'pending_cancel', 'pending_replace', 'canceled', 'expired', 'replaced', 'rejected', 'suspended')",
            name="ck_m3_oco_leg_state_status",
        ),
        sa.ForeignKeyConstraint(["signal_id"], ["signal_registry.signal_id"]),
        sa.ForeignKeyConstraint(["position_id"], ["m3_validation_metadata.position_id"]),
        sa.PrimaryKeyConstraint("leg_state_id"),
    )
    op.create_index("ix_m3_oco_leg_state_signal_id", "m3_oco_leg_state", ["signal_id"])
    op.create_index("ix_m3_oco_leg_state_position_id", "m3_oco_leg_state", ["position_id"])
    op.create_index("ix_m3_oco_leg_state_broker_order", "m3_oco_leg_state", ["broker_order_id"])


def downgrade() -> None:
    op.drop_index("ix_m3_oco_leg_state_broker_order", table_name="m3_oco_leg_state")
    op.drop_index("ix_m3_oco_leg_state_position_id", table_name="m3_oco_leg_state")
    op.drop_index("ix_m3_oco_leg_state_signal_id", table_name="m3_oco_leg_state")
    op.drop_table("m3_oco_leg_state")
    op.drop_index("ix_m3_tranche_fills_position_id", table_name="m3_tranche_fills")
    op.drop_index("ix_m3_tranche_fills_signal_id", table_name="m3_tranche_fills")
    op.drop_table("m3_tranche_fills")
    op.drop_index(
        "idx_m3_validation_metadata_position_id_unique",
        table_name="m3_validation_metadata",
    )
    op.drop_table("m3_validation_metadata")
    op.drop_index("ix_sector_change_log_job_run_id", table_name="sector_change_log")
    op.drop_index("ix_sector_change_log_ticker_date", table_name="sector_change_log")
    op.drop_table("sector_change_log")
    op.drop_index("ix_sector_returns_daily_sector", table_name="sector_returns_daily")
    op.drop_index("ix_sector_returns_daily_rank", table_name="sector_returns_daily")
    op.drop_table("sector_returns_daily")
    op.drop_index(
        "ix_firm_sector_assignments_last_verified",
        table_name="firm_sector_assignments",
    )
    op.drop_index(
        "ix_firm_sector_assignments_sector",
        table_name="firm_sector_assignments",
    )
    op.drop_table("firm_sector_assignments")
    op.drop_index(
        "ix_firm_sector_history_source",
        table_name="firm_sector_assignments_history",
    )
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "ALTER TABLE firm_sector_assignments_history "
            "DROP CONSTRAINT IF EXISTS ex_firm_sector_history_no_overlap"
        )
    op.drop_index(
        "ix_firm_sector_history_sector_interval",
        table_name="firm_sector_assignments_history",
    )
    op.drop_index(
        "ix_firm_sector_history_ticker_interval",
        table_name="firm_sector_assignments_history",
    )
    op.drop_table("firm_sector_assignments_history")
