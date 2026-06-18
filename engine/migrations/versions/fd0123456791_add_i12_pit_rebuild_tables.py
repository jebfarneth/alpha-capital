"""add i12 pit rebuild research tables

Revision ID: fd0123456791
Revises: fc0123456790
Create Date: 2026-06-17 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "fd0123456791"
down_revision: Union[str, None] = "fc0123456790"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "i12_pit_candidates",
        sa.Column("i12_pit_candidate_id", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("decision_date", sa.Date(), nullable=False),
        sa.Column("decision_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decision_time_label", sa.String(), nullable=False),
        sa.Column(
            "path_mode",
            sa.String(),
            nullable=False,
            server_default="strict_contiguous",
        ),
        sa.Column("feature_asof_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("candidate_status", sa.String(), nullable=False),
        sa.Column("coverage_status", sa.String(), nullable=False),
        sa.Column("fail_reason", sa.String(), nullable=True),
        sa.Column("feature_json", sa.Text(), nullable=False),
        sa.Column("gate_values_json", sa.Text(), nullable=False),
        sa.Column("leakage_guard_json", sa.Text(), nullable=False),
        sa.Column("source_bars_json", sa.Text(), nullable=False),
        sa.Column("label_json", sa.Text(), nullable=True),
        sa.Column("error_json", sa.Text(), nullable=True),
        sa.Column("candidate_attempt_hash", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("superseded_by_candidate_id", sa.String(), nullable=True),
        sa.Column("input_hash", sa.String(), nullable=False),
        sa.Column("candidate_identity_hash", sa.String(), nullable=False),
        sa.Column("label_hash", sa.String(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.PrimaryKeyConstraint("i12_pit_candidate_id"),
        sa.UniqueConstraint("content_hash", name="ux_i12_pit_candidates_content_hash"),
    )
    op.create_index(
        "ix_i12_pit_candidates_attempt_active",
        "i12_pit_candidates",
        ["candidate_attempt_hash", "is_active"],
    )
    op.create_index(
        "ux_i12_pit_candidates_active_attempt",
        "i12_pit_candidates",
        ["candidate_attempt_hash"],
        unique=True,
        sqlite_where=sa.text("is_active = 1"),
        postgresql_where=sa.text("is_active"),
    )
    op.create_index(
        "ix_i12_pit_candidates_ticker_decision",
        "i12_pit_candidates",
        ["ticker", "decision_ts"],
    )
    op.create_index(
        "ix_i12_pit_candidates_status",
        "i12_pit_candidates",
        ["decision_date", "candidate_status", "coverage_status"],
    )

    op.create_table(
        "i12_pit_quote_replays",
        sa.Column("i12_pit_quote_replay_id", sa.String(), nullable=False),
        sa.Column("i12_pit_candidate_id", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("decision_date", sa.Date(), nullable=False),
        sa.Column("decision_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quote_role", sa.String(), nullable=False),
        sa.Column("target_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_start_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quote_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("quote_age_seconds", sa.Float(), nullable=True),
        sa.Column("bid", sa.Float(), nullable=True),
        sa.Column("ask", sa.Float(), nullable=True),
        sa.Column("bid_size", sa.Float(), nullable=True),
        sa.Column("ask_size", sa.Float(), nullable=True),
        sa.Column("spread_bps", sa.Float(), nullable=True),
        sa.Column("top_of_book_notional", sa.Float(), nullable=True),
        sa.Column("bid_notional", sa.Float(), nullable=True),
        sa.Column("ask_notional", sa.Float(), nullable=True),
        sa.Column("executable_notional", sa.Float(), nullable=True),
        sa.Column("executable_side", sa.String(), nullable=True),
        sa.Column("feed", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("quote_size_basis", sa.String(), nullable=False),
        sa.Column("coverage_status", sa.String(), nullable=False),
        sa.Column("raw_json", sa.Text(), nullable=True),
        sa.Column("error_json", sa.Text(), nullable=True),
        sa.Column("quote_replay_attempt_hash", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("superseded_by_quote_replay_id", sa.String(), nullable=True),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["i12_pit_candidate_id"],
            ["i12_pit_candidates.i12_pit_candidate_id"],
        ),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.PrimaryKeyConstraint("i12_pit_quote_replay_id"),
        sa.UniqueConstraint("content_hash", name="ux_i12_pit_quote_replays_content_hash"),
    )
    op.create_index(
        "ix_i12_pit_quote_replays_attempt_active",
        "i12_pit_quote_replays",
        ["quote_replay_attempt_hash", "is_active"],
    )
    op.create_index(
        "ux_i12_pit_quote_replays_active_attempt",
        "i12_pit_quote_replays",
        ["quote_replay_attempt_hash"],
        unique=True,
        sqlite_where=sa.text("is_active = 1"),
        postgresql_where=sa.text("is_active"),
    )
    op.create_index(
        "ix_i12_pit_quote_replays_candidate_role",
        "i12_pit_quote_replays",
        ["i12_pit_candidate_id", "quote_role"],
    )
    op.create_index(
        "ix_i12_pit_quote_replays_status",
        "i12_pit_quote_replays",
        ["quote_role", "coverage_status"],
    )

    op.create_table(
        "i12_pit_cost_replays",
        sa.Column("i12_pit_cost_replay_id", sa.String(), nullable=False),
        sa.Column("i12_pit_candidate_id", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("decision_date", sa.Date(), nullable=False),
        sa.Column("decision_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_role", sa.String(), nullable=False),
        sa.Column("entry_quote_replay_id", sa.String(), nullable=True),
        sa.Column("exit_quote_replay_id", sa.String(), nullable=True),
        sa.Column("tradeability_status", sa.String(), nullable=False),
        sa.Column("skipped_reason", sa.String(), nullable=False),
        sa.Column("intended_order_usd", sa.Float(), nullable=False),
        sa.Column("max_spread_bps", sa.Float(), nullable=False),
        sa.Column("slippage_bps", sa.Float(), nullable=False),
        sa.Column("entry_ask", sa.Float(), nullable=True),
        sa.Column("exit_bid", sa.Float(), nullable=True),
        sa.Column("gross_return", sa.Float(), nullable=True),
        sa.Column("quote_cost_return", sa.Float(), nullable=True),
        sa.Column("slippage_return", sa.Float(), nullable=True),
        sa.Column("modeled_return", sa.Float(), nullable=False),
        sa.Column("cost_replay_attempt_hash", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("superseded_by_cost_replay_id", sa.String(), nullable=True),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["entry_quote_replay_id"],
            ["i12_pit_quote_replays.i12_pit_quote_replay_id"],
        ),
        sa.ForeignKeyConstraint(
            ["exit_quote_replay_id"],
            ["i12_pit_quote_replays.i12_pit_quote_replay_id"],
        ),
        sa.ForeignKeyConstraint(
            ["i12_pit_candidate_id"],
            ["i12_pit_candidates.i12_pit_candidate_id"],
        ),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.PrimaryKeyConstraint("i12_pit_cost_replay_id"),
        sa.UniqueConstraint("content_hash", name="ux_i12_pit_cost_replays_content_hash"),
    )
    op.create_index(
        "ix_i12_pit_cost_replays_attempt_active",
        "i12_pit_cost_replays",
        ["cost_replay_attempt_hash", "is_active"],
    )
    op.create_index(
        "ux_i12_pit_cost_replays_active_attempt",
        "i12_pit_cost_replays",
        ["cost_replay_attempt_hash"],
        unique=True,
        sqlite_where=sa.text("is_active = 1"),
        postgresql_where=sa.text("is_active"),
    )
    op.create_index(
        "ix_i12_pit_cost_replays_candidate_exit",
        "i12_pit_cost_replays",
        ["i12_pit_candidate_id", "exit_role"],
    )
    op.create_index(
        "ix_i12_pit_cost_replays_status",
        "i12_pit_cost_replays",
        ["exit_role", "tradeability_status", "skipped_reason"],
    )


def downgrade() -> None:
    op.drop_index("ix_i12_pit_cost_replays_status", table_name="i12_pit_cost_replays")
    op.drop_index(
        "ux_i12_pit_cost_replays_active_attempt",
        table_name="i12_pit_cost_replays",
    )
    op.drop_index(
        "ix_i12_pit_cost_replays_attempt_active",
        table_name="i12_pit_cost_replays",
    )
    op.drop_index(
        "ix_i12_pit_cost_replays_candidate_exit",
        table_name="i12_pit_cost_replays",
    )
    op.drop_table("i12_pit_cost_replays")
    op.drop_index(
        "ix_i12_pit_quote_replays_status",
        table_name="i12_pit_quote_replays",
    )
    op.drop_index(
        "ux_i12_pit_quote_replays_active_attempt",
        table_name="i12_pit_quote_replays",
    )
    op.drop_index(
        "ix_i12_pit_quote_replays_attempt_active",
        table_name="i12_pit_quote_replays",
    )
    op.drop_index(
        "ix_i12_pit_quote_replays_candidate_role",
        table_name="i12_pit_quote_replays",
    )
    op.drop_table("i12_pit_quote_replays")
    op.drop_index("ix_i12_pit_candidates_status", table_name="i12_pit_candidates")
    op.drop_index(
        "ux_i12_pit_candidates_active_attempt",
        table_name="i12_pit_candidates",
    )
    op.drop_index("ix_i12_pit_candidates_attempt_active", table_name="i12_pit_candidates")
    op.drop_index(
        "ix_i12_pit_candidates_ticker_decision",
        table_name="i12_pit_candidates",
    )
    op.drop_table("i12_pit_candidates")
