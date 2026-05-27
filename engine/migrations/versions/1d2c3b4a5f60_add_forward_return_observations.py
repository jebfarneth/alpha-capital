"""add forward return observations table

Revision ID: 1d2c3b4a5f60
Revises: 0f4c6e8a9b21
Create Date: 2026-05-26 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "1d2c3b4a5f60"
down_revision: Union[str, None] = "0f4c6e8a9b21"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "signal_registry",
        sa.Column("next_execution_session", sa.String(), nullable=True),
    )
    op.create_table(
        "forward_return_observations",
        sa.Column("forward_return_observation_id", sa.String(), nullable=False),
        sa.Column("signal_id", sa.String(), nullable=False),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("direction", sa.String(), nullable=False),
        sa.Column("signal_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("signal_horizon", sa.String(), nullable=True),
        sa.Column("next_execution_session", sa.String(), nullable=True),
        sa.Column("entry_session_date", sa.String(), nullable=True),
        sa.Column("entry_price", sa.Float(), nullable=True),
        sa.Column("entry_price_source", sa.String(), nullable=True),
        sa.Column("entry_basis_proof", sa.String(), nullable=True),
        sa.Column("entry_data_lineage_id", sa.String(), nullable=True),
        sa.Column("exit_session_date", sa.String(), nullable=True),
        sa.Column("exit_price", sa.Float(), nullable=True),
        sa.Column("exit_price_source", sa.String(), nullable=True),
        sa.Column("exit_basis_proof", sa.String(), nullable=True),
        sa.Column("exit_data_lineage_id", sa.String(), nullable=True),
        sa.Column("forward_return", sa.Float(), nullable=True),
        sa.Column("max_favorable_excursion", sa.Float(), nullable=True),
        sa.Column("max_adverse_excursion", sa.Float(), nullable=True),
        sa.Column("mfe_session_date", sa.String(), nullable=True),
        sa.Column("mae_session_date", sa.String(), nullable=True),
        sa.Column("max_close_return", sa.Float(), nullable=True),
        sa.Column("min_close_return", sa.Float(), nullable=True),
        sa.Column("hit_t1_intraday", sa.Boolean(), nullable=True),
        sa.Column("hit_t2_intraday", sa.Boolean(), nullable=True),
        sa.Column("hit_t3_intraday", sa.Boolean(), nullable=True),
        sa.Column("hit_stop_intraday", sa.Boolean(), nullable=True),
        sa.Column("same_day_barrier_ambiguity", sa.Boolean(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("reason", sa.String(), nullable=True),
        sa.Column(
            "attempts",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("input_hash", sa.String(), nullable=False),
        sa.Column("outcome_hash", sa.String(), nullable=False),
        sa.Column("data_lineage_ids", sa.Text(), nullable=True),
        sa.Column("provider", sa.String(), nullable=True),
        sa.Column("endpoint", sa.String(), nullable=True),
        sa.Column("provider_request_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["entry_data_lineage_id"], ["data_lineage.data_lineage_id"]),
        sa.ForeignKeyConstraint(["exit_data_lineage_id"], ["data_lineage.data_lineage_id"]),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.ForeignKeyConstraint(["signal_id"], ["signal_registry.signal_id"]),
        sa.PrimaryKeyConstraint("forward_return_observation_id"),
    )
    op.create_index(
        "ux_forward_return_observations_signal_input",
        "forward_return_observations",
        ["signal_id", "input_hash"],
        unique=True,
    )
    op.create_index(
        "ix_forward_return_observations_pattern_status",
        "forward_return_observations",
        ["pattern_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_forward_return_observations_ticker",
        "forward_return_observations",
        ["ticker"],
        unique=False,
    )
    op.create_table(
        "forward_return_observation_events",
        sa.Column("forward_return_observation_event_id", sa.String(), nullable=False),
        sa.Column("forward_return_observation_id", sa.String(), nullable=True),
        sa.Column("signal_id", sa.String(), nullable=False),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("direction", sa.String(), nullable=False),
        sa.Column("signal_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("signal_horizon", sa.String(), nullable=True),
        sa.Column("next_execution_session", sa.String(), nullable=True),
        sa.Column("entry_session_date", sa.String(), nullable=True),
        sa.Column("entry_price", sa.Float(), nullable=True),
        sa.Column("entry_price_source", sa.String(), nullable=True),
        sa.Column("entry_basis_proof", sa.String(), nullable=True),
        sa.Column("entry_data_lineage_id", sa.String(), nullable=True),
        sa.Column("exit_session_date", sa.String(), nullable=True),
        sa.Column("exit_price", sa.Float(), nullable=True),
        sa.Column("exit_price_source", sa.String(), nullable=True),
        sa.Column("exit_basis_proof", sa.String(), nullable=True),
        sa.Column("exit_data_lineage_id", sa.String(), nullable=True),
        sa.Column("forward_return", sa.Float(), nullable=True),
        sa.Column("max_favorable_excursion", sa.Float(), nullable=True),
        sa.Column("max_adverse_excursion", sa.Float(), nullable=True),
        sa.Column("mfe_session_date", sa.String(), nullable=True),
        sa.Column("mae_session_date", sa.String(), nullable=True),
        sa.Column("max_close_return", sa.Float(), nullable=True),
        sa.Column("min_close_return", sa.Float(), nullable=True),
        sa.Column("hit_t1_intraday", sa.Boolean(), nullable=True),
        sa.Column("hit_t2_intraday", sa.Boolean(), nullable=True),
        sa.Column("hit_t3_intraday", sa.Boolean(), nullable=True),
        sa.Column("hit_stop_intraday", sa.Boolean(), nullable=True),
        sa.Column("same_day_barrier_ambiguity", sa.Boolean(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("reason", sa.String(), nullable=True),
        sa.Column(
            "attempts",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("input_hash", sa.String(), nullable=False),
        sa.Column("outcome_hash", sa.String(), nullable=False),
        sa.Column("data_lineage_ids", sa.Text(), nullable=True),
        sa.Column("provider", sa.String(), nullable=True),
        sa.Column("endpoint", sa.String(), nullable=True),
        sa.Column("provider_request_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["forward_return_observation_id"],
            ["forward_return_observations.forward_return_observation_id"],
        ),
        sa.ForeignKeyConstraint(["job_run_id"], ["evidence_job_runs.job_run_id"]),
        sa.ForeignKeyConstraint(["signal_id"], ["signal_registry.signal_id"]),
        sa.PrimaryKeyConstraint("forward_return_observation_event_id"),
    )
    op.create_index(
        "ix_forward_return_observation_events_signal_attempt",
        "forward_return_observation_events",
        ["signal_id", "attempts"],
        unique=False,
    )
    op.create_index(
        "ix_forward_return_observation_events_observation",
        "forward_return_observation_events",
        ["forward_return_observation_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_forward_return_observation_events_observation",
        table_name="forward_return_observation_events",
    )
    op.drop_index(
        "ix_forward_return_observation_events_signal_attempt",
        table_name="forward_return_observation_events",
    )
    op.drop_table("forward_return_observation_events")
    op.drop_index(
        "ix_forward_return_observations_ticker",
        table_name="forward_return_observations",
    )
    op.drop_index(
        "ix_forward_return_observations_pattern_status",
        table_name="forward_return_observations",
    )
    op.drop_index(
        "ux_forward_return_observations_signal_input",
        table_name="forward_return_observations",
    )
    op.drop_table("forward_return_observations")
    op.drop_column("signal_registry", "next_execution_session")
