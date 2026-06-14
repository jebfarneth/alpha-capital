"""add stage1 ml model registry

Revision ID: f90123456789
Revises: e8f901234567
Create Date: 2026-06-14 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f90123456789"
down_revision: Union[str, None] = "e8f901234567"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ml_model_registry",
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("job_run_id", sa.String(), nullable=True),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("model_family", sa.String(), nullable=False),
        sa.Column("training_window_start", sa.Date(), nullable=True),
        sa.Column("training_window_end", sa.Date(), nullable=True),
        sa.Column("manifest_version", sa.String(), nullable=False),
        sa.Column("manifest_sha256", sa.String(), nullable=False),
        sa.Column("feature_schema_hash", sa.String(), nullable=False),
        sa.Column("feature_code_git_sha", sa.String(), nullable=True),
        sa.Column("training_params_json", sa.Text(), nullable=True),
        sa.Column("cv_metrics_json", sa.Text(), nullable=False),
        sa.Column("feature_schema_json", sa.Text(), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
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
        sa.PrimaryKeyConstraint("model_id"),
    )
    op.create_index(
        "ix_ml_model_registry_pattern_status",
        "ml_model_registry",
        ["pattern_id", "status"],
    )
    op.create_index(
        "ix_ml_model_registry_schema_hash",
        "ml_model_registry",
        ["feature_schema_hash"],
    )

    op.create_table(
        "signal_ml_scores",
        sa.Column("score_id", sa.String(), nullable=False),
        sa.Column("signal_id", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(), nullable=True),
        sa.Column("requested_model_id", sa.String(), nullable=True),
        sa.Column("pattern_id", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("fallback_score", sa.Float(), nullable=True),
        sa.Column("score_source", sa.String(), nullable=False),
        sa.Column("fallback_reason", sa.String(), nullable=True),
        sa.Column("score_status", sa.String(), nullable=False),
        sa.Column("feature_schema_hash", sa.String(), nullable=True),
        sa.Column("feature_vector_hash", sa.String(), nullable=True),
        sa.Column("score_metadata_json", sa.Text(), nullable=True),
        sa.Column(
            "scored_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
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
        sa.ForeignKeyConstraint(["model_id"], ["ml_model_registry.model_id"]),
        sa.ForeignKeyConstraint(["signal_id"], ["signal_registry.signal_id"]),
        sa.PrimaryKeyConstraint("score_id"),
        sa.UniqueConstraint(
            "signal_id",
            "model_id",
            "score_status",
            name="ux_signal_ml_scores_signal_model_status",
        ),
    )
    op.create_index(
        "ix_signal_ml_scores_pattern_scored_at",
        "signal_ml_scores",
        ["pattern_id", "scored_at"],
    )
    op.create_index(
        "ix_signal_ml_scores_source",
        "signal_ml_scores",
        ["score_source", "score_status"],
    )


def downgrade() -> None:
    op.drop_index("ix_signal_ml_scores_source", table_name="signal_ml_scores")
    op.drop_index(
        "ix_signal_ml_scores_pattern_scored_at", table_name="signal_ml_scores"
    )
    op.drop_table("signal_ml_scores")
    op.drop_index("ix_ml_model_registry_schema_hash", table_name="ml_model_registry")
    op.drop_index("ix_ml_model_registry_pattern_status", table_name="ml_model_registry")
    op.drop_table("ml_model_registry")
