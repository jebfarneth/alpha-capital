"""initial evidence spine - 18 canonical tables

Revision ID: 9e273a77556a
Revises:
Create Date: 2026-05-20 20:43:58.220263
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "9e273a77556a"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "evidence_datasets",
        sa.Column("dataset_id", sa.String(), nullable=False),
        sa.Column("dataset_name", sa.String(), nullable=False),
        sa.Column("dataset_type", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=True),
        sa.Column("frequency", sa.String(), nullable=True),
        sa.Column("unit", sa.String(), nullable=True),
        sa.Column("source_url", sa.String(), nullable=True),
        sa.Column("dataset_version", sa.String(), nullable=True),
        sa.Column("release_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("schema_hash", sa.String(), nullable=True),
        sa.Column("citation", sa.Text(), nullable=True),
        sa.Column("transformation_notes", sa.Text(), nullable=True),
        sa.Column("known_limitations", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("dataset_id"),
    )

    op.create_table(
        "evidence_jobs",
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("job_name", sa.String(), nullable=False),
        sa.Column("job_type", sa.String(), nullable=False),
        sa.Column("owner_component", sa.String(), nullable=False),
        sa.Column("schedule", sa.String(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("job_id"),
    )

    op.create_table(
        "manual_overrides",
        sa.Column("override_id", sa.String(), nullable=False),
        sa.Column("override_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("override_type", sa.String(), nullable=False),
        sa.Column("affected_entity_type", sa.String(), nullable=False),
        sa.Column("affected_entity_id", sa.String(), nullable=False),
        sa.Column("before_state", sa.Text(), nullable=True),
        sa.Column("after_state", sa.Text(), nullable=True),
        sa.Column("operator_rationale", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("override_id"),
    )

    op.create_table(
        "pattern_weights",
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("baseline_weight", sa.Float(), nullable=False),
        sa.Column("current_weight", sa.Float(), nullable=False),
        sa.Column("last_adjustment_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cumulative_adjustment_factor", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("pattern_id"),
    )

    op.create_table(
        "evidence_job_runs",
        sa.Column("job_run_id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("run_status", sa.String(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("app_commit_sha", sa.String(), nullable=True),
        sa.Column("vault_commit_sha", sa.String(), nullable=True),
        sa.Column("params_json", sa.Text(), nullable=True),
        sa.Column("metric_json", sa.Text(), nullable=True),
        sa.Column("tag_json", sa.Text(), nullable=True),
        sa.Column("input_dataset_ids", sa.Text(), nullable=True),
        sa.Column("output_dataset_ids", sa.Text(), nullable=True),
        sa.Column("artifact_uris", sa.Text(), nullable=True),
        sa.Column("input_hashes", sa.Text(), nullable=True),
        sa.Column("output_hashes", sa.Text(), nullable=True),
        sa.Column("error_json", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["evidence_jobs.job_id"]),
        sa.PrimaryKeyConstraint("job_run_id"),
    )

    op.create_table(
        "evidence_snapshots",
        sa.Column("evidence_snapshot_id", sa.String(), nullable=False),
        sa.Column("snapshot_name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by_job_run_id", sa.String(), nullable=True),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dataset_ids", sa.Text(), nullable=True),
        sa.Column("row_counts", sa.Text(), nullable=True),
        sa.Column("content_hashes", sa.Text(), nullable=True),
        sa.Column("snapshot_manifest_hash", sa.String(), nullable=True),
        sa.Column("storage_uri", sa.String(), nullable=True),
        sa.Column("known_limitations", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["created_by_job_run_id"], ["evidence_job_runs.job_run_id"]
        ),
        sa.PrimaryKeyConstraint("evidence_snapshot_id"),
    )

    op.create_table(
        "universe_snapshots",
        sa.Column("universe_snapshot_id", sa.String(), nullable=False),
        sa.Column("evidence_snapshot_id", sa.String(), nullable=True),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("scan_id", sa.String(), nullable=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("asof_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_provider", sa.String(), nullable=True),
        sa.Column("market_cap", sa.Float(), nullable=True),
        sa.Column("price", sa.Float(), nullable=True),
        sa.Column("security_type", sa.String(), nullable=True),
        sa.Column("primary_exchange", sa.String(), nullable=True),
        sa.Column("fractionable", sa.Boolean(), nullable=True),
        sa.Column("liquidity_score", sa.Float(), nullable=True),
        sa.Column("median_dollar_volume_20d", sa.Float(), nullable=True),
        sa.Column("median_dollar_volume_60d", sa.Float(), nullable=True),
        sa.Column("high_low_range_proxy_20d", sa.Float(), nullable=True),
        sa.Column("sub_dollar_exception_flag", sa.Boolean(), nullable=True),
        sa.Column("hazard_score", sa.Float(), nullable=True),
        sa.Column("active_vetoes", sa.Text(), nullable=True),
        sa.Column("operating_universe_inclusion", sa.Boolean(), nullable=False),
        sa.Column("exclusion_reason", sa.String(), nullable=True),
        sa.Column("dataset_version", sa.String(), nullable=True),
        sa.Column("schema_hash", sa.String(), nullable=True),
        sa.Column("source_lineage_hash", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["evidence_snapshot_id"], ["evidence_snapshots.evidence_snapshot_id"]
        ),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.PrimaryKeyConstraint("universe_snapshot_id"),
    )

    op.create_table(
        "data_lineage",
        sa.Column("data_lineage_id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("endpoint", sa.String(), nullable=False),
        sa.Column("request_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("asof_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_payload_hash", sa.String(), nullable=False),
        sa.Column("normalized_payload_hash", sa.String(), nullable=True),
        sa.Column("freshness_seconds", sa.Float(), nullable=True),
        sa.Column("source_authority", sa.String(), nullable=True),
        sa.Column("data_quality_flags", sa.Text(), nullable=True),
        sa.Column("dataset_id", sa.String(), nullable=True),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("lineage_facet_json", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["dataset_id"], ["evidence_datasets.dataset_id"]),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.PrimaryKeyConstraint("data_lineage_id"),
    )

    op.create_table(
        "feature_snapshots",
        sa.Column("feature_snapshot_id", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("evidence_snapshot_id", sa.String(), nullable=True),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("asof_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_manifest_version", sa.String(), nullable=True),
        sa.Column("feature_json", sa.Text(), nullable=False),
        sa.Column("feature_hash", sa.String(), nullable=False),
        sa.Column("code_commit_sha", sa.String(), nullable=True),
        sa.Column("data_lineage_ids", sa.Text(), nullable=False),
        sa.Column("fidelity_tier", sa.String(), nullable=True),
        sa.Column("point_in_time_passed", sa.Boolean(), nullable=True),
        sa.Column("lookahead_guard_passed", sa.Boolean(), nullable=True),
        sa.Column("input_hashes", sa.Text(), nullable=True),
        sa.Column("output_hash", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["evidence_snapshot_id"], ["evidence_snapshots.evidence_snapshot_id"]
        ),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.PrimaryKeyConstraint("feature_snapshot_id"),
    )

    op.create_table(
        "signal_registry",
        sa.Column("signal_id", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("direction", sa.String(), nullable=False),
        sa.Column("signal_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_signal_strength", sa.Float(), nullable=False),
        sa.Column("raw_expected_edge", sa.Float(), nullable=False),
        sa.Column("signal_horizon", sa.String(), nullable=True),
        sa.Column("thesis_category", sa.String(), nullable=True),
        sa.Column("route_class", sa.String(), nullable=True),
        sa.Column("fidelity_tier", sa.String(), nullable=True),
        sa.Column("data_confidence", sa.Float(), nullable=True),
        sa.Column("feature_snapshot_id", sa.String(), nullable=False),
        sa.Column("signal_status", sa.String(), nullable=False),
        sa.Column("signal_event_sequence", sa.Integer(), nullable=True),
        sa.Column("universe_snapshot_id", sa.String(), nullable=True),
        sa.Column("data_lineage_ids", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["feature_snapshot_id"], ["feature_snapshots.feature_snapshot_id"]
        ),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.ForeignKeyConstraint(
            ["universe_snapshot_id"], ["universe_snapshots.universe_snapshot_id"]
        ),
        sa.PrimaryKeyConstraint("signal_id"),
    )

    op.create_table(
        "trade_candidates",
        sa.Column("candidate_id", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("candidate_pool_id", sa.String(), nullable=False),
        sa.Column("scan_id", sa.String(), nullable=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("direction", sa.String(), nullable=False),
        sa.Column("primary_pattern", sa.String(), nullable=False),
        sa.Column("active_patterns", sa.Text(), nullable=True),
        sa.Column("combined_expected_edge", sa.Float(), nullable=False),
        sa.Column("expected_round_trip_cost", sa.Float(), nullable=True),
        sa.Column("missed_fill_adjustment", sa.Float(), nullable=True),
        sa.Column("optimizer_input_expected_edge", sa.Float(), nullable=True),
        sa.Column("validation_weight_multiplier", sa.Float(), nullable=True),
        sa.Column("pattern_weight", sa.Float(), nullable=True),
        sa.Column("shrinkage_weight", sa.Float(), nullable=True),
        sa.Column("hazard_multiplier", sa.Float(), nullable=True),
        sa.Column("liquidity_multiplier", sa.Float(), nullable=True),
        sa.Column("fidelity_multiplier", sa.Float(), nullable=True),
        sa.Column("max_position_pct", sa.Float(), nullable=True),
        sa.Column("skip_reason", sa.String(), nullable=True),
        sa.Column("trade_decision", sa.String(), nullable=False),
        sa.Column("catalyst_cluster", sa.String(), nullable=True),
        sa.Column("same_symbol_state", sa.String(), nullable=True),
        sa.Column("candidate_rank_pre_optimizer", sa.Integer(), nullable=True),
        sa.Column("candidate_percentile_pre_optimizer", sa.Float(), nullable=True),
        sa.Column("cash_available_at_decision", sa.Float(), nullable=True),
        sa.Column("settled_cash_required", sa.Float(), nullable=True),
        sa.Column("constraint_reason_json", sa.Text(), nullable=True),
        sa.Column("input_signal_ids", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "trade_decision IN "
            "('enter', 'skip', 'vetoed_filing', 'vetoed_hazard', 'vetoed_liquidity')",
            name="ck_trade_candidates_trade_decision",
        ),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.PrimaryKeyConstraint("candidate_id"),
    )

    op.create_table(
        "optimizer_runs",
        sa.Column("optimizer_run_id", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("candidate_pool_id", sa.String(), nullable=False),
        sa.Column("run_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("nav", sa.Float(), nullable=False),
        sa.Column("settled_cash", sa.Float(), nullable=True),
        sa.Column("reserve_target", sa.Float(), nullable=True),
        sa.Column("active_holdings_count", sa.Integer(), nullable=False),
        sa.Column("target_holdings_count", sa.Integer(), nullable=True),
        sa.Column("selected_candidate_ids", sa.Text(), nullable=False),
        sa.Column("constraint_bindings_json", sa.Text(), nullable=True),
        sa.Column("solver_status", sa.String(), nullable=True),
        sa.Column("objective_value", sa.Float(), nullable=True),
        sa.Column("params_json", sa.Text(), nullable=True),
        sa.Column("input_hashes", sa.Text(), nullable=True),
        sa.Column("output_hash", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.PrimaryKeyConstraint("optimizer_run_id"),
    )

    op.create_table(
        "order_events",
        sa.Column("order_event_id", sa.String(), nullable=False),
        sa.Column("order_request_id", sa.String(), nullable=False),
        sa.Column("candidate_id", sa.String(), nullable=True),
        sa.Column("real_position_id", sa.String(), nullable=True),
        sa.Column("order_ticket_id", sa.String(), nullable=True),
        sa.Column("broker_order_id", sa.String(), nullable=True),
        sa.Column("route_class", sa.String(), nullable=True),
        sa.Column("request_type", sa.String(), nullable=True),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("event_sequence", sa.Integer(), nullable=False),
        sa.Column("broker_status", sa.String(), nullable=True),
        sa.Column("broker_response_status", sa.String(), nullable=True),
        sa.Column("intended_price", sa.Float(), nullable=True),
        sa.Column("submitted_price", sa.Float(), nullable=True),
        sa.Column("filled_avg_price", sa.Float(), nullable=True),
        sa.Column("filled_qty", sa.Float(), nullable=True),
        sa.Column("cumulative_filled_qty", sa.Float(), nullable=True),
        sa.Column("cumulative_avg_fill_price", sa.Float(), nullable=True),
        sa.Column("slippage_bps", sa.Float(), nullable=True),
        sa.Column("fill_quality", sa.String(), nullable=True),
        sa.Column("reject_reason", sa.String(), nullable=True),
        sa.Column("cancel_reason", sa.String(), nullable=True),
        sa.Column("event_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["candidate_id"], ["trade_candidates.candidate_id"]),
        sa.PrimaryKeyConstraint("order_event_id"),
    )

    op.create_table(
        "stbm_lifecycle_events",
        sa.Column("stbm_event_id", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("position_id", sa.String(), nullable=False),
        sa.Column("candidate_id", sa.String(), nullable=True),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("previous_lifecycle_state", sa.String(), nullable=True),
        sa.Column("new_lifecycle_state", sa.String(), nullable=False),
        sa.Column("current_stop_state", sa.String(), nullable=True),
        sa.Column("current_stop_price", sa.Float(), nullable=True),
        sa.Column("tranche_state_json", sa.Text(), nullable=True),
        sa.Column("oco_state_json", sa.Text(), nullable=True),
        sa.Column("race_condition_resolution", sa.String(), nullable=True),
        sa.Column("session_boundary_replay_id", sa.String(), nullable=True),
        sa.Column("event_sequence", sa.Integer(), nullable=True),
        sa.Column("event_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["candidate_id"], ["trade_candidates.candidate_id"]),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.PrimaryKeyConstraint("stbm_event_id"),
    )

    op.create_table(
        "shadow_positions",
        sa.Column("shadow_position_id", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("signal_id", sa.String(), nullable=False),
        sa.Column("candidate_id", sa.String(), nullable=True),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("entry_price_shadow", sa.Float(), nullable=False),
        sa.Column("exit_price_shadow", sa.Float(), nullable=True),
        sa.Column("exit_bucket", sa.String(), nullable=True),
        sa.Column("shadow_return", sa.Float(), nullable=True),
        sa.Column("mae", sa.Float(), nullable=True),
        sa.Column("mfe", sa.Float(), nullable=True),
        sa.Column("t1_hit", sa.Boolean(), nullable=True),
        sa.Column("t2_hit", sa.Boolean(), nullable=True),
        sa.Column("right_tail_trade_flag", sa.Boolean(), nullable=True),
        sa.Column("terminal_tranche_mfe_capture_pct", sa.Float(), nullable=True),
        sa.Column("fill_quality", sa.String(), nullable=True),
        sa.Column("input_hashes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["candidate_id"], ["trade_candidates.candidate_id"]),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.ForeignKeyConstraint(["signal_id"], ["signal_registry.signal_id"]),
        sa.PrimaryKeyConstraint("shadow_position_id"),
    )

    op.create_table(
        "validation_runs",
        sa.Column("validation_run_id", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=False),
        sa.Column("pattern_id", sa.String(), nullable=True),
        sa.Column("run_type", sa.String(), nullable=False),
        sa.Column("run_status", sa.String(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sample_size", sa.Integer(), nullable=True),
        sa.Column("params_json", sa.Text(), nullable=True),
        sa.Column("metric_json", sa.Text(), nullable=True),
        sa.Column("tag_json", sa.Text(), nullable=True),
        sa.Column("confidence_tier", sa.String(), nullable=True),
        sa.Column("validation_weight_multiplier", sa.Float(), nullable=True),
        sa.Column("operator_review_flag", sa.Boolean(), nullable=True),
        sa.Column("artifact_uris", sa.Text(), nullable=True),
        sa.Column("input_snapshot_ids", sa.Text(), nullable=True),
        sa.Column("input_dataset_ids", sa.Text(), nullable=True),
        sa.Column("input_hashes", sa.Text(), nullable=True),
        sa.Column("output_hashes", sa.Text(), nullable=True),
        sa.Column("error_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.PrimaryKeyConstraint("validation_run_id"),
    )

    op.create_table(
        "agent_export_manifests",
        sa.Column("export_id", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("evidence_snapshot_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("pattern_scope", sa.Text(), nullable=True),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("redaction_mode", sa.String(), nullable=True),
        sa.Column("included_tables", sa.Text(), nullable=True),
        sa.Column("manifest_hash", sa.String(), nullable=True),
        sa.Column("export_path", sa.String(), nullable=True),
        sa.Column("source_dataset_ids", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["evidence_snapshot_id"], ["evidence_snapshots.evidence_snapshot_id"]
        ),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.PrimaryKeyConstraint("export_id"),
    )

    op.create_table(
        "real_positions",
        sa.Column("real_position_id", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("shadow_position_id", sa.String(), nullable=True),
        sa.Column("candidate_id", sa.String(), nullable=False),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("entry_price_real", sa.Float(), nullable=False),
        sa.Column("exit_price_real", sa.Float(), nullable=True),
        sa.Column("real_return", sa.Float(), nullable=True),
        sa.Column("realized_pnl", sa.Float(), nullable=True),
        sa.Column("execution_capture", sa.Float(), nullable=True),
        sa.Column("cash_drag_flag", sa.Boolean(), nullable=True),
        sa.Column("operator_override_flag", sa.Boolean(), nullable=True),
        sa.Column("override_affected", sa.Boolean(), nullable=True),
        sa.Column("broker_error_flag", sa.Boolean(), nullable=True),
        sa.Column("linked_order_event_ids", sa.Text(), nullable=True),
        sa.Column("input_hashes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["candidate_id"], ["trade_candidates.candidate_id"]),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.ForeignKeyConstraint(["shadow_position_id"], ["shadow_positions.shadow_position_id"]),
        sa.PrimaryKeyConstraint("real_position_id"),
    )


def downgrade() -> None:
    op.drop_table("real_positions")
    op.drop_table("agent_export_manifests")
    op.drop_table("validation_runs")
    op.drop_table("shadow_positions")
    op.drop_table("stbm_lifecycle_events")
    op.drop_table("order_events")
    op.drop_table("optimizer_runs")
    op.drop_table("trade_candidates")
    op.drop_table("signal_registry")
    op.drop_table("feature_snapshots")
    op.drop_table("data_lineage")
    op.drop_table("universe_snapshots")
    op.drop_table("evidence_snapshots")
    op.drop_table("evidence_job_runs")
    op.drop_table("pattern_weights")
    op.drop_table("manual_overrides")
    op.drop_table("evidence_jobs")
    op.drop_table("evidence_datasets")
