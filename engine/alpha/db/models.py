"""
Canonical evidence tables per Engineering/Validation/EvidenceSchema.md.

18 tables matching the spec entity graph:
  universe_snapshots, evidence_jobs, evidence_job_runs, evidence_datasets,
  evidence_snapshots, data_lineage, feature_snapshots, signal_registry,
  trade_candidates, optimizer_runs, order_events, stbm_lifecycle_events,
  shadow_positions, real_positions, validation_runs, agent_export_manifests,
  pattern_weights, manual_overrides.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# evidence_jobs
# ---------------------------------------------------------------------------
class EvidenceJob(Base):
    __tablename__ = "evidence_jobs"

    job_id = Column(String, primary_key=True, default=_uuid)
    job_name = Column(String, nullable=False)
    job_type = Column(String, nullable=False)
    owner_component = Column(String, nullable=False)
    schedule = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    active = Column(Boolean, default=True, nullable=False)

    runs = relationship("EvidenceJobRun", back_populates="job")


# ---------------------------------------------------------------------------
# evidence_job_runs
# ---------------------------------------------------------------------------
class EvidenceJobRun(Base):
    __tablename__ = "evidence_job_runs"

    job_run_id = Column(String, primary_key=True, default=_uuid)
    job_id = Column(String, ForeignKey("evidence_jobs.job_id"), nullable=False)
    run_status = Column(String, nullable=False, default="scheduled")
    started_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    app_commit_sha = Column(String, nullable=True)
    vault_commit_sha = Column(String, nullable=True)
    params_json = Column(Text, nullable=True)
    metric_json = Column(Text, nullable=True)
    tag_json = Column(Text, nullable=True)
    input_dataset_ids = Column(Text, nullable=True)  # JSON array
    output_dataset_ids = Column(Text, nullable=True)  # JSON array
    artifact_uris = Column(Text, nullable=True)  # JSON array
    input_hashes = Column(Text, nullable=True)  # JSON
    output_hashes = Column(Text, nullable=True)  # JSON
    error_json = Column(Text, nullable=True)

    job = relationship("EvidenceJob", back_populates="runs")


# ---------------------------------------------------------------------------
# evidence_datasets
# ---------------------------------------------------------------------------
class EvidenceDataset(Base):
    __tablename__ = "evidence_datasets"

    dataset_id = Column(String, primary_key=True, default=_uuid)
    dataset_name = Column(String, nullable=False)
    dataset_type = Column(String, nullable=False)
    provider = Column(String, nullable=True)
    frequency = Column(String, nullable=True)
    unit = Column(String, nullable=True)
    source_url = Column(String, nullable=True)
    dataset_version = Column(String, nullable=True)
    release_date = Column(DateTime(timezone=True), nullable=True)
    schema_hash = Column(String, nullable=True)
    citation = Column(Text, nullable=True)
    transformation_notes = Column(Text, nullable=True)
    known_limitations = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


# ---------------------------------------------------------------------------
# evidence_snapshots
# ---------------------------------------------------------------------------
class EvidenceSnapshot(Base):
    __tablename__ = "evidence_snapshots"

    evidence_snapshot_id = Column(String, primary_key=True, default=_uuid)
    snapshot_name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    created_by_job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    window_start = Column(DateTime(timezone=True), nullable=True)
    window_end = Column(DateTime(timezone=True), nullable=True)
    dataset_ids = Column(Text, nullable=True)  # JSON array
    row_counts = Column(Text, nullable=True)  # JSON
    content_hashes = Column(Text, nullable=True)  # JSON
    snapshot_manifest_hash = Column(String, nullable=True)
    storage_uri = Column(String, nullable=True)
    known_limitations = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# security_profiles
# ---------------------------------------------------------------------------
class SecurityProfile(Base):
    __tablename__ = "security_profiles"
    __table_args__ = (
        Index("ix_security_profiles_security_type", "security_type"),
        Index("ix_security_profiles_last_refreshed_at", "last_refreshed_at"),
    )

    symbol = Column(String, primary_key=True)
    security_type = Column(String, nullable=False, default="unknown")
    source_provider = Column(String, nullable=True)
    source_lineage_hash = Column(String, nullable=True)
    profile_payload_hash = Column(String, nullable=True)
    classification_input_hash = Column(String, nullable=True)
    classification_output_hash = Column(String, nullable=True)
    classifier_version = Column(String, nullable=True)
    profile_asof_timestamp = Column(DateTime(timezone=True), nullable=True)
    last_refreshed_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    refresh_status = Column(String, nullable=True)
    raw_profile_json = Column(Text, nullable=True)
    classification_reason = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# universe_scans
# ---------------------------------------------------------------------------
class UniverseScan(Base):
    __tablename__ = "universe_scans"
    __table_args__ = (
        Index("ix_universe_scans_trading_date", "trading_date"),
        Index("ix_universe_scans_job_run_id", "job_run_id"),
    )

    scan_id = Column(String, primary_key=True, default=_uuid)
    trading_date = Column(String, nullable=False)
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    asof_timestamp = Column(DateTime(timezone=True), nullable=False)
    provider = Column(String, nullable=True)
    raw_count = Column(Integer, nullable=False, default=0)
    deduped_count = Column(Integer, nullable=False, default=0)
    duplicate_symbol_count = Column(Integer, nullable=False, default=0)
    included_count = Column(Integer, nullable=False, default=0)
    excluded_count = Column(Integer, nullable=False, default=0)
    source_lineage_hash = Column(String, nullable=True)
    security_profile_cache_hash = Column(String, nullable=True)
    output_hash = Column(String, nullable=True)
    run_status = Column(String, nullable=False, default="finished")
    metric_json = Column(Text, nullable=True)

    snapshots = relationship("UniverseSnapshot", back_populates="scan")
    security_profile_snapshots = relationship(
        "SecurityProfileScanSnapshot", back_populates="scan"
    )


# ---------------------------------------------------------------------------
# canonical_universe_scans
# ---------------------------------------------------------------------------
class CanonicalUniverseScan(Base):
    __tablename__ = "canonical_universe_scans"

    trading_date = Column(String, primary_key=True)
    scan_id = Column(
        String, ForeignKey("universe_scans.scan_id"), nullable=False
    )
    selected_job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    selected_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    selection_reason = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# universe_snapshots
# ---------------------------------------------------------------------------
class UniverseSnapshot(Base):
    __tablename__ = "universe_snapshots"
    __table_args__ = (
        Index(
            "ux_universe_snapshots_scan_ticker",
            "scan_id", "ticker",
            unique=True,
        ),
        Index(
            "ix_universe_snapshots_scan_inclusion",
            "scan_id", "operating_universe_inclusion",
        ),
        Index(
            "ix_universe_snapshots_ticker_asof",
            "ticker", "asof_timestamp",
        ),
    )

    universe_snapshot_id = Column(String, primary_key=True, default=_uuid)
    evidence_snapshot_id = Column(
        String, ForeignKey("evidence_snapshots.evidence_snapshot_id"), nullable=True
    )
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    scan_id = Column(
        String, ForeignKey("universe_scans.scan_id"), nullable=True
    )
    ticker = Column(String, nullable=False)
    asof_timestamp = Column(DateTime(timezone=True), nullable=False)
    source_provider = Column(String, nullable=True)
    market_cap = Column(Float, nullable=True)
    price = Column(Float, nullable=True)
    country = Column(String, nullable=True)
    security_type = Column(String, nullable=True)
    primary_exchange = Column(String, nullable=True)
    fractionable = Column(Boolean, nullable=True)
    liquidity_score = Column(Float, nullable=True)
    median_dollar_volume_20d = Column(Float, nullable=True)
    median_dollar_volume_60d = Column(Float, nullable=True)
    high_low_range_proxy_20d = Column(Float, nullable=True)
    sub_dollar_exception_flag = Column(Boolean, nullable=True)
    hazard_score = Column(Float, nullable=True)
    active_vetoes = Column(Text, nullable=True)  # JSON array
    operating_universe_inclusion = Column(Boolean, nullable=False)
    exclusion_reason = Column(String, nullable=True)
    dataset_version = Column(String, nullable=True)
    schema_hash = Column(String, nullable=True)
    source_lineage_hash = Column(String, nullable=True)

    scan = relationship("UniverseScan", back_populates="snapshots")


# ---------------------------------------------------------------------------
# security_profile_scan_snapshots
# ---------------------------------------------------------------------------
class SecurityProfileScanSnapshot(Base):
    __tablename__ = "security_profile_scan_snapshots"
    __table_args__ = (
        Index(
            "ux_security_profile_scan_snapshots_scan_symbol",
            "scan_id", "symbol",
            unique=True,
        ),
        Index(
            "ix_security_profile_scan_snapshots_scan_required",
            "scan_id", "profile_required",
        ),
    )

    profile_scan_snapshot_id = Column(String, primary_key=True, default=_uuid)
    scan_id = Column(
        String, ForeignKey("universe_scans.scan_id"), nullable=False
    )
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    symbol = Column(String, nullable=False)
    profile_required = Column(Boolean, nullable=False, default=False)
    cache_status = Column(String, nullable=False)
    stale = Column(Boolean, nullable=True)
    security_type = Column(String, nullable=True)
    refresh_status = Column(String, nullable=True)
    classification_reason = Column(String, nullable=True)
    classifier_version = Column(String, nullable=True)
    classification_input_hash = Column(String, nullable=True)
    classification_output_hash = Column(String, nullable=True)
    source_lineage_hash = Column(String, nullable=True)
    profile_payload_hash = Column(String, nullable=True)
    profile_asof_timestamp = Column(DateTime(timezone=True), nullable=True)
    last_refreshed_at = Column(DateTime(timezone=True), nullable=True)
    raw_profile_json = Column(Text, nullable=True)

    scan = relationship("UniverseScan", back_populates="security_profile_snapshots")


# ---------------------------------------------------------------------------
# data_lineage
# ---------------------------------------------------------------------------
class DataLineage(Base):
    __tablename__ = "data_lineage"

    data_lineage_id = Column(String, primary_key=True, default=_uuid)
    provider = Column(String, nullable=False)
    endpoint = Column(String, nullable=False)
    request_timestamp = Column(DateTime(timezone=True), nullable=False)
    provider_timestamp = Column(DateTime(timezone=True), nullable=True)
    asof_timestamp = Column(DateTime(timezone=True), nullable=False)
    raw_payload_hash = Column(String, nullable=False)
    raw_payload_json = Column(Text, nullable=True)
    normalized_payload_hash = Column(String, nullable=True)
    freshness_seconds = Column(Float, nullable=True)
    source_authority = Column(String, nullable=True)
    data_quality_flags = Column(Text, nullable=True)  # JSON
    dataset_id = Column(
        String, ForeignKey("evidence_datasets.dataset_id"), nullable=True
    )
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    lineage_facet_json = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# feature_snapshots
# ---------------------------------------------------------------------------
class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"

    feature_snapshot_id = Column(String, primary_key=True, default=_uuid)
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    evidence_snapshot_id = Column(
        String, ForeignKey("evidence_snapshots.evidence_snapshot_id"), nullable=True
    )
    pattern_id = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    asof_timestamp = Column(DateTime(timezone=True), nullable=False)
    feature_manifest_version = Column(String, nullable=True)
    feature_json = Column(Text, nullable=False)  # JSON payload
    feature_hash = Column(String, nullable=False)
    code_commit_sha = Column(String, nullable=True)
    data_lineage_ids = Column(Text, nullable=False)  # JSON array
    fidelity_tier = Column(String, nullable=True)
    point_in_time_passed = Column(Boolean, nullable=True)
    lookahead_guard_passed = Column(Boolean, nullable=True)
    input_hashes = Column(Text, nullable=True)  # JSON
    output_hash = Column(String, nullable=True)

    signals = relationship("SignalRegistry", back_populates="feature_snapshot")


# ---------------------------------------------------------------------------
# signal_registry
# ---------------------------------------------------------------------------
class SignalRegistry(Base):
    __tablename__ = "signal_registry"
    __table_args__ = (
        Index(
            "ux_signal_registry_pattern_ticker_identity",
            "pattern_id",
            "ticker",
            "signal_identity_hash",
            unique=True,
        ),
    )

    signal_id = Column(String, primary_key=True, default=_uuid)
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    pattern_id = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    signal_timestamp = Column(DateTime(timezone=True), nullable=False)
    raw_signal_strength = Column(Float, nullable=False)
    raw_expected_edge = Column(Float, nullable=False)
    signal_horizon = Column(String, nullable=True)
    thesis_category = Column(String, nullable=True)
    route_class = Column(String, nullable=True)
    fidelity_tier = Column(String, nullable=True)
    data_confidence = Column(Float, nullable=True)
    feature_snapshot_id = Column(
        String,
        ForeignKey("feature_snapshots.feature_snapshot_id"),
        nullable=False,
    )
    signal_status = Column(String, nullable=False, default="active")
    signal_event_sequence = Column(Integer, nullable=True)
    universe_snapshot_id = Column(
        String, ForeignKey("universe_snapshots.universe_snapshot_id"), nullable=True
    )
    trading_date = Column(String, nullable=True)
    scan_id = Column(
        String, ForeignKey("universe_scans.scan_id"), nullable=True
    )
    detector_version = Column(String, nullable=True)
    point_in_time_passed = Column(Boolean, nullable=True)
    lookahead_guard_passed = Column(Boolean, nullable=True)
    data_lineage_ids = Column(Text, nullable=True)  # JSON array
    signal_identity_hash = Column(String, nullable=False)
    forward_return = Column(Float, nullable=True)
    forward_return_status = Column(String, nullable=True)
    forward_return_attempts = Column(Integer, nullable=False, default=0)
    outcome_unavailable_reason = Column(String, nullable=True)
    intended_entry_price = Column(Float, nullable=True)

    feature_snapshot = relationship("FeatureSnapshot", back_populates="signals")


# ---------------------------------------------------------------------------
# trade_candidates
# ---------------------------------------------------------------------------
class TradeCandidate(Base):
    __tablename__ = "trade_candidates"
    __table_args__ = (
        CheckConstraint(
            "trade_decision IN "
            "('enter', 'skip', 'vetoed_filing', 'vetoed_hazard', 'vetoed_liquidity')",
            name="ck_trade_candidates_trade_decision",
        ),
    )

    candidate_id = Column(String, primary_key=True, default=_uuid)
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    candidate_pool_id = Column(String, nullable=False)
    scan_id = Column(String, nullable=True)
    ticker = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    primary_pattern = Column(String, nullable=False)
    active_patterns = Column(Text, nullable=True)  # JSON array
    combined_expected_edge = Column(Float, nullable=False)
    effective_hard_stop_pct = Column(Float, nullable=True)
    base_risk_budget_pct = Column(Float, nullable=True)
    risk_budget_pct = Column(Float, nullable=True)
    risk_multiplier_product = Column(Float, nullable=True)
    risk_sized_cap = Column(Float, nullable=True)
    unstopped_heat_pct = Column(Float, nullable=True)
    expected_round_trip_cost = Column(Float, nullable=True)
    cost_to_edge_ratio = Column(Float, nullable=True)
    missed_fill_adjustment = Column(Float, nullable=True)
    optimizer_input_expected_edge = Column(Float, nullable=True)
    validation_weight_multiplier = Column(Float, nullable=True)
    pattern_weight = Column(Float, nullable=True)
    shrinkage_weight = Column(Float, nullable=True)
    hazard_multiplier = Column(Float, nullable=True)
    liquidity_multiplier = Column(Float, nullable=True)
    fidelity_multiplier = Column(Float, nullable=True)
    max_position_pct = Column(Float, nullable=True)
    skip_reason = Column(String, nullable=True)
    trade_decision = Column(String, nullable=False)
    catalyst_cluster = Column(String, nullable=True)
    same_symbol_state = Column(String, nullable=True)
    candidate_rank_pre_optimizer = Column(Integer, nullable=True)
    candidate_percentile_pre_optimizer = Column(Float, nullable=True)
    cash_available_at_decision = Column(Float, nullable=True)
    settled_cash_required = Column(Float, nullable=True)
    constraint_reason_json = Column(Text, nullable=True)  # JSON
    input_signal_ids = Column(Text, nullable=False)  # JSON array


# ---------------------------------------------------------------------------
# optimizer_runs
# ---------------------------------------------------------------------------
class OptimizerRun(Base):
    __tablename__ = "optimizer_runs"

    optimizer_run_id = Column(String, primary_key=True, default=_uuid)
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    candidate_pool_id = Column(String, nullable=False)
    run_timestamp = Column(DateTime(timezone=True), nullable=False)
    nav = Column(Float, nullable=False)
    settled_cash = Column(Float, nullable=True)
    reserve_target = Column(Float, nullable=True)
    active_holdings_count = Column(Integer, nullable=False)
    target_holdings_count = Column(Integer, nullable=True)
    selected_candidate_ids = Column(Text, nullable=False)  # JSON array
    constraint_bindings_json = Column(Text, nullable=True)  # JSON
    solver_status = Column(String, nullable=True)
    objective_value = Column(Float, nullable=True)
    params_json = Column(Text, nullable=True)
    input_hashes = Column(Text, nullable=True)  # JSON
    output_hash = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# order_events  (append-only)
# ---------------------------------------------------------------------------
class OrderEvent(Base):
    __tablename__ = "order_events"

    order_event_id = Column(String, primary_key=True, default=_uuid)
    order_request_id = Column(String, nullable=False)
    candidate_id = Column(
        String, ForeignKey("trade_candidates.candidate_id"), nullable=True
    )
    real_position_id = Column(String, nullable=True)
    order_ticket_id = Column(String, nullable=True)
    broker_order_id = Column(String, nullable=True)
    route_class = Column(String, nullable=True)
    request_type = Column(String, nullable=True)
    event_type = Column(String, nullable=False)
    event_sequence = Column(Integer, nullable=False)
    broker_status = Column(String, nullable=True)
    broker_response_status = Column(String, nullable=True)
    intended_price = Column(Float, nullable=True)
    submitted_price = Column(Float, nullable=True)
    filled_avg_price = Column(Float, nullable=True)
    filled_qty = Column(Float, nullable=True)
    cumulative_filled_qty = Column(Float, nullable=True)
    cumulative_avg_fill_price = Column(Float, nullable=True)
    slippage_bps = Column(Float, nullable=True)
    fill_quality = Column(String, nullable=True)
    reject_reason = Column(String, nullable=True)
    cancel_reason = Column(String, nullable=True)
    event_timestamp = Column(DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# stbm_lifecycle_events  (append-only)
# ---------------------------------------------------------------------------
class StbmLifecycleEvent(Base):
    __tablename__ = "stbm_lifecycle_events"

    stbm_event_id = Column(String, primary_key=True, default=_uuid)
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    position_id = Column(String, nullable=False)
    candidate_id = Column(
        String, ForeignKey("trade_candidates.candidate_id"), nullable=True
    )
    pattern_id = Column(String, nullable=False)
    previous_lifecycle_state = Column(String, nullable=True)
    new_lifecycle_state = Column(String, nullable=False)
    current_stop_state = Column(String, nullable=True)
    current_stop_price = Column(Float, nullable=True)
    tranche_state_json = Column(Text, nullable=True)
    oco_state_json = Column(Text, nullable=True)
    race_condition_resolution = Column(String, nullable=True)
    session_boundary_replay_id = Column(String, nullable=True)
    event_sequence = Column(Integer, nullable=True)
    event_timestamp = Column(DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# shadow_positions
# ---------------------------------------------------------------------------
class ShadowPosition(Base):
    __tablename__ = "shadow_positions"

    shadow_position_id = Column(String, primary_key=True, default=_uuid)
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    signal_id = Column(
        String, ForeignKey("signal_registry.signal_id"), nullable=False
    )
    candidate_id = Column(
        String, ForeignKey("trade_candidates.candidate_id"), nullable=True
    )
    pattern_id = Column(String, nullable=False)
    forward_return = Column(Float, nullable=True)
    fill_status = Column(String, nullable=True)
    intended_entry_price = Column(Float, nullable=True)
    realized_entry_price = Column(Float, nullable=True)
    execution_capture_gap = Column(Float, nullable=True)
    entry_price_shadow = Column(Float, nullable=False)
    exit_price_shadow = Column(Float, nullable=True)
    exit_bucket = Column(String, nullable=True)
    shadow_return = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    mfe = Column(Float, nullable=True)
    t1_hit = Column(Boolean, nullable=True)
    t2_hit = Column(Boolean, nullable=True)
    right_tail_trade_flag = Column(Boolean, nullable=True)
    terminal_tranche_mfe_capture_pct = Column(Float, nullable=True)
    fill_quality = Column(String, nullable=True)
    input_hashes = Column(Text, nullable=True)  # JSON


# ---------------------------------------------------------------------------
# real_positions
# ---------------------------------------------------------------------------
class RealPosition(Base):
    __tablename__ = "real_positions"

    real_position_id = Column(String, primary_key=True, default=_uuid)
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    shadow_position_id = Column(
        String,
        ForeignKey("shadow_positions.shadow_position_id"),
        nullable=True,
    )
    candidate_id = Column(
        String, ForeignKey("trade_candidates.candidate_id"), nullable=False
    )
    pattern_id = Column(String, nullable=False)
    fill_status = Column(String, nullable=True)
    intended_entry_price = Column(Float, nullable=True)
    realized_entry_price = Column(Float, nullable=True)
    execution_capture_gap = Column(Float, nullable=True)
    entry_price_real = Column(Float, nullable=False)
    exit_price_real = Column(Float, nullable=True)
    real_return = Column(Float, nullable=True)
    realized_pnl = Column(Float, nullable=True)
    execution_capture = Column(Float, nullable=True)
    cash_drag_flag = Column(Boolean, nullable=True)
    operator_override_flag = Column(Boolean, nullable=True)
    override_affected = Column(Boolean, nullable=True)
    broker_error_flag = Column(Boolean, nullable=True)
    linked_order_event_ids = Column(Text, nullable=True)  # JSON array
    input_hashes = Column(Text, nullable=True)  # JSON


# ---------------------------------------------------------------------------
# validation_runs
# ---------------------------------------------------------------------------
class ValidationRun(Base):
    __tablename__ = "validation_runs"

    validation_run_id = Column(String, primary_key=True, default=_uuid)
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=False
    )
    pattern_id = Column(String, nullable=True)
    run_type = Column(String, nullable=False)
    run_status = Column(String, nullable=False, default="scheduled")
    window_start = Column(DateTime(timezone=True), nullable=True)
    window_end = Column(DateTime(timezone=True), nullable=True)
    sample_size = Column(Integer, nullable=True)
    params_json = Column(Text, nullable=True)
    metric_json = Column(Text, nullable=True)
    tag_json = Column(Text, nullable=True)
    confidence_tier = Column(String, nullable=True)
    validation_weight_multiplier = Column(Float, nullable=True)
    operator_review_flag = Column(Boolean, nullable=True)
    artifact_uris = Column(Text, nullable=True)  # JSON array
    input_snapshot_ids = Column(Text, nullable=True)  # JSON array
    input_dataset_ids = Column(Text, nullable=True)  # JSON array
    input_hashes = Column(Text, nullable=True)  # JSON
    output_hashes = Column(Text, nullable=True)  # JSON
    error_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


# ---------------------------------------------------------------------------
# agent_export_manifests
# ---------------------------------------------------------------------------
class AgentExportManifest(Base):
    __tablename__ = "agent_export_manifests"

    export_id = Column(String, primary_key=True, default=_uuid)
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    evidence_snapshot_id = Column(
        String,
        ForeignKey("evidence_snapshots.evidence_snapshot_id"),
        nullable=True,
    )
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    created_by = Column(String, nullable=True)
    pattern_scope = Column(Text, nullable=True)  # JSON array
    window_start = Column(DateTime(timezone=True), nullable=True)
    window_end = Column(DateTime(timezone=True), nullable=True)
    redaction_mode = Column(String, nullable=True)
    included_tables = Column(Text, nullable=True)  # JSON array
    manifest_hash = Column(String, nullable=True)
    export_path = Column(String, nullable=True)
    source_dataset_ids = Column(Text, nullable=True)  # JSON array


# ---------------------------------------------------------------------------
# pattern_weights
# ---------------------------------------------------------------------------
class PatternWeight(Base):
    __tablename__ = "pattern_weights"

    pattern_id = Column(String, primary_key=True)
    baseline_weight = Column(Float, nullable=False)
    current_weight = Column(Float, nullable=False)
    last_adjustment_date = Column(DateTime(timezone=True), nullable=True)
    cumulative_adjustment_factor = Column(Float, nullable=True)


# ---------------------------------------------------------------------------
# manual_overrides
# ---------------------------------------------------------------------------
class ManualOverride(Base):
    __tablename__ = "manual_overrides"

    override_id = Column(String, primary_key=True, default=_uuid)
    override_timestamp = Column(DateTime(timezone=True), nullable=False)
    override_type = Column(String, nullable=False)
    affected_entity_type = Column(String, nullable=False)
    affected_entity_id = Column(String, nullable=False)
    before_state = Column(Text, nullable=True)  # JSON
    after_state = Column(Text, nullable=True)  # JSON
    operator_rationale = Column(Text, nullable=True)
    created_by = Column(String, nullable=True)
