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
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all Alpha Capital ORM models."""

    pass


# ---------------------------------------------------------------------------
# evidence_jobs
# ---------------------------------------------------------------------------
class EvidenceJob(Base):
    """Registered evidence-producing job definition."""

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
    """One execution attempt for an evidence-producing job."""

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
    """External or derived dataset metadata used by evidence records."""

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
# nasdaq_listing_snapshots
# ---------------------------------------------------------------------------
class NasdaqListingSnapshot(Base):
    """Archived raw Nasdaq Trader listing source payload."""

    __tablename__ = "nasdaq_listing_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "source_type",
            "source_knowledge_timestamp",
            "raw_payload_hash",
            name="ux_nasdaq_listing_snapshot_source_time_hash",
        ),
        Index(
            "ix_nasdaq_listing_snapshots_source_time",
            "source_type",
            "source_knowledge_timestamp",
        ),
    )

    snapshot_id = Column(String, primary_key=True, default=_uuid)
    captured_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    source_type = Column(String, nullable=False)
    source_url = Column(Text, nullable=False)
    source_knowledge_timestamp = Column(DateTime(timezone=True), nullable=False)
    raw_payload_hash = Column(String, nullable=False)
    raw_payload = Column(Text, nullable=False)
    row_count = Column(Integer, nullable=False)
    parse_status = Column(String, nullable=False, default="parsed")
    data_quality_flags_json = Column(Text, nullable=True)

    rows = relationship(
        "NasdaqListingSnapshotRow",
        back_populates="snapshot",
        cascade="all, delete-orphan",
    )


# ---------------------------------------------------------------------------
# nasdaq_listing_snapshot_rows
# ---------------------------------------------------------------------------
class NasdaqListingSnapshotRow(Base):
    """Parsed row from an archived Nasdaq Trader listing source."""

    __tablename__ = "nasdaq_listing_snapshot_rows"
    __table_args__ = (
        Index(
            "ix_nasdaq_listing_snapshot_rows_symbol",
            "source_type",
            "symbol",
        ),
        Index(
            "ix_nasdaq_listing_snapshot_rows_effective",
            "source_type",
            "effective_date",
        ),
    )

    snapshot_row_id = Column(String, primary_key=True, default=_uuid)
    snapshot_id = Column(
        String, ForeignKey("nasdaq_listing_snapshots.snapshot_id"), nullable=False
    )
    source_type = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    normalized_symbol = Column(String, nullable=False)
    security_name = Column(Text, nullable=True)
    market = Column(String, nullable=True)
    action = Column(String, nullable=True)
    effective_date = Column(String, nullable=True)
    reason_code = Column(String, nullable=True)
    raw_json = Column(Text, nullable=True)

    snapshot = relationship("NasdaqListingSnapshot", back_populates="rows")


# ---------------------------------------------------------------------------
# evidence_snapshots
# ---------------------------------------------------------------------------
class EvidenceSnapshot(Base):
    """Immutable manifest for a captured evidence snapshot."""

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
    """Cached security-type classification and source profile for a symbol."""

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
    """Immutable operating-universe scan summary."""

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
    """Pointer selecting the authoritative universe scan for a trading date."""

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
    """Point-in-time universe membership row for one ticker in one scan."""

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
    """Replayable view of the security-profile cache used by a universe scan."""

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
# security_identity_snapshots
# ---------------------------------------------------------------------------
class SecurityIdentitySnapshot(Base):
    """Point-in-time identity evidence from Polygon ticker details/events."""

    __tablename__ = "security_identity_snapshots"
    __table_args__ = (
        Index(
            "ux_security_identity_snapshots_scan_ticker",
            "scan_id", "ticker",
            unique=True,
        ),
        Index(
            "ix_security_identity_snapshots_cik",
            "cik",
        ),
        Index(
            "ix_security_identity_snapshots_composite_figi",
            "composite_figi",
        ),
        Index(
            "ix_security_identity_snapshots_share_class_figi",
            "share_class_figi",
        ),
    )

    security_identity_snapshot_id = Column(String, primary_key=True, default=_uuid)
    scan_id = Column(
        String, ForeignKey("universe_scans.scan_id"), nullable=False
    )
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    ticker = Column(String, nullable=False)
    cik = Column(String, nullable=True)
    composite_figi = Column(String, nullable=True)
    share_class_figi = Column(String, nullable=True)
    active = Column(Boolean, nullable=True)
    delisted_utc = Column(String, nullable=True)
    list_date = Column(String, nullable=True)
    polygon_type = Column(String, nullable=True)
    polygon_market = Column(String, nullable=True)
    polygon_locale = Column(String, nullable=True)
    polygon_primary_exchange = Column(String, nullable=True)
    polygon_name = Column(String, nullable=True)
    sic_code = Column(String, nullable=True)
    sic_description = Column(String, nullable=True)
    ticker_events_json = Column(Text, nullable=True)
    identity_status = Column(String, nullable=False)
    identity_reason = Column(String, nullable=True)
    identity_hash = Column(String, nullable=True)
    source_provider = Column(String, nullable=True)
    source_endpoint = Column(String, nullable=True)
    data_lineage_id = Column(
        String, ForeignKey("data_lineage.data_lineage_id"), nullable=True
    )
    events_data_lineage_id = Column(
        String, ForeignKey("data_lineage.data_lineage_id"), nullable=True
    )
    data_lineage_ids = Column(Text, nullable=True)  # JSON array
    raw_payload_hash = Column(String, nullable=True)
    asof_timestamp = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    scan = relationship("UniverseScan")


# ---------------------------------------------------------------------------
# m1_earnings_events
# ---------------------------------------------------------------------------
class M1EarningsEvent(Base):
    """PIT M1 earnings event and Foster SUE computation evidence."""

    __tablename__ = "m1_earnings_events"
    __table_args__ = (
        UniqueConstraint(
            "scan_id",
            "ticker",
            "earnings_event_id",
            name="ux_m1_earnings_events_scan_ticker_event",
        ),
        Index("ix_m1_earnings_events_scan_status", "scan_id", "status"),
        Index("ix_m1_earnings_events_ticker_announcement", "ticker", "announcement_date"),
    )

    m1_earnings_event_id = Column(String, primary_key=True, default=_uuid)
    scan_id = Column(String, ForeignKey("universe_scans.scan_id"), nullable=True)
    universe_snapshot_id = Column(
        String, ForeignKey("universe_snapshots.universe_snapshot_id"), nullable=True
    )
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    ticker = Column(String, nullable=False)
    earnings_event_id = Column(String, nullable=False)
    announcement_date = Column(String, nullable=True)
    effective_announcement_session = Column(String, nullable=True)
    announcement_time = Column(String, nullable=True)
    fiscal_period_end = Column(String, nullable=True)
    fiscal_year = Column(Integer, nullable=True)
    fiscal_quarter = Column(Integer, nullable=True)
    actual_eps = Column(Float, nullable=True)
    estimated_eps = Column(Float, nullable=True)
    expected_eps = Column(Float, nullable=True)
    sigma_delta_eps = Column(Float, nullable=True)
    sue_foster = Column(Float, nullable=True)
    rho1 = Column(Float, nullable=True)
    sue_sign_current = Column(Integer, nullable=True)
    sue_sign_prior = Column(Integer, nullable=True)
    sue_streak_length = Column(Integer, nullable=True)
    foster_history_quarters_used = Column(Integer, nullable=False, default=0)
    split_adjustment_continuity_check = Column(String, nullable=True)
    restatement_exposure = Column(Boolean, nullable=False, default=False)
    status = Column(String, nullable=False)
    diagnostic_json = Column(Text, nullable=True)
    sue_series_json = Column(Text, nullable=True)
    data_lineage_ids = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


# ---------------------------------------------------------------------------
# m1_friction_snapshots
# ---------------------------------------------------------------------------
class M1FrictionSnapshot(Base):
    """M1-local D1 and sigma_epsilon values ranked over the operating universe."""

    __tablename__ = "m1_friction_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "scan_id",
            "ticker",
            name="ux_m1_friction_snapshots_scan_ticker",
        ),
        Index("ix_m1_friction_snapshots_scan_status", "scan_id", "status"),
        Index("ix_m1_friction_snapshots_d1_decile", "scan_id", "d1_decile"),
    )

    m1_friction_snapshot_id = Column(String, primary_key=True, default=_uuid)
    scan_id = Column(String, ForeignKey("universe_scans.scan_id"), nullable=True)
    universe_snapshot_id = Column(
        String, ForeignKey("universe_snapshots.universe_snapshot_id"), nullable=True
    )
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    ticker = Column(String, nullable=False)
    market_factor_symbol = Column(String, nullable=False, default="SPY")
    d1 = Column(Float, nullable=True)
    d1_decile = Column(Integer, nullable=True)
    sigma_epsilon = Column(Float, nullable=True)
    sigma_epsilon_percentile = Column(Float, nullable=True)
    weekly_return_count = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False)
    diagnostic_json = Column(Text, nullable=True)
    data_lineage_ids = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


# ---------------------------------------------------------------------------
# m2_insider_transactions
# ---------------------------------------------------------------------------
class M2InsiderTransaction(Base):
    """Full Form 4 transaction stream used by M2 classification and clustering."""

    __tablename__ = "m2_insider_transactions"
    __table_args__ = (
        Index("ix_m2_transactions_ticker_tradable", "ticker", "first_tradable_session"),
        Index("ix_m2_transactions_insider_year", "insider_id", "transaction_date"),
        Index("ix_m2_transactions_accession", "filing_accession_number"),
        Index("ix_m2_transactions_source", "source_authority"),
    )

    transaction_id = Column(String, primary_key=True)
    scan_id = Column(String, ForeignKey("universe_scans.scan_id"), nullable=True)
    universe_snapshot_id = Column(
        String, ForeignKey("universe_snapshots.universe_snapshot_id"), nullable=True
    )
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    source_authority = Column(String, nullable=False)
    enrichment_sources = Column(Text, nullable=True)
    ticker = Column(String, nullable=False)
    issuer_cik = Column(String, nullable=True)
    issuer_name = Column(Text, nullable=True)
    insider_id = Column(String, nullable=False)
    insider_cik = Column(String, nullable=True)
    insider_name = Column(Text, nullable=True)
    issuer_state = Column(String, nullable=True)
    insider_state = Column(String, nullable=True)
    identity_resolution_method = Column(String, nullable=False)
    identity_resolution_confidence = Column(Float, nullable=False)
    filing_accession_number = Column(String, nullable=True)
    filing_form = Column(String, nullable=True)
    filing_date = Column(String, nullable=True)
    filing_accepted_at = Column(DateTime(timezone=True), nullable=True)
    filing_detected_at = Column(DateTime(timezone=True), nullable=True)
    first_tradable_session = Column(String, nullable=True)
    clock_quality = Column(String, nullable=False)
    transaction_date = Column(String, nullable=True)
    transaction_code = Column(String, nullable=True)
    transaction_code_description = Column(Text, nullable=True)
    acquired_disposed_code = Column(String, nullable=True)
    transaction_shares = Column(Float, nullable=True)
    transaction_price_per_share = Column(Float, nullable=True)
    transaction_notional_usd = Column(Float, nullable=True)
    purchase_notional_usd = Column(Float, nullable=True)
    market_cap_usd = Column(Float, nullable=True)
    ownership_type = Column(String, nullable=True)
    insider_roles_json = Column(Text, nullable=True)
    is_open_market_purchase = Column(Boolean, nullable=False, default=False)
    is_buy = Column(Boolean, nullable=False, default=False)
    is_sell = Column(Boolean, nullable=False, default=False)
    is_10b5_1 = Column(Boolean, nullable=True)
    sec_fmp_mismatch = Column(Boolean, nullable=False, default=False)
    data_lineage_ids = Column(Text, nullable=True)
    raw_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


# ---------------------------------------------------------------------------
# m2_insider_classifications
# ---------------------------------------------------------------------------
class M2InsiderClassification(Base):
    """Annual CMP routine/opportunistic/unclassifiable classification."""

    __tablename__ = "m2_insider_classifications"
    __table_args__ = (
        UniqueConstraint(
            "insider_id",
            "calendar_year",
            name="ux_m2_classifications_insider_year",
        ),
        Index("ix_m2_classifications_year_class", "calendar_year", "classification"),
    )

    m2_insider_classification_id = Column(String, primary_key=True, default=_uuid)
    insider_id = Column(String, nullable=False)
    insider_cik = Column(String, nullable=True)
    insider_name = Column(Text, nullable=True)
    calendar_year = Column(Integer, nullable=False)
    classification = Column(String, nullable=False)
    routine_month = Column(Integer, nullable=True)
    prior_year_count = Column(Integer, nullable=False, default=0)
    data_cutoff_at = Column(DateTime(timezone=True), nullable=False)
    basis_json = Column(Text, nullable=True)
    computed_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


# ---------------------------------------------------------------------------
# m2_cluster_members
# ---------------------------------------------------------------------------
class M2ClusterMember(Base):
    """Join table from M2/M2U cluster ids back to accession-level transactions."""

    __tablename__ = "m2_cluster_members"
    __table_args__ = (
        UniqueConstraint(
            "pattern_id",
            "m2_cluster_id",
            "transaction_id",
            name="ux_m2_cluster_members_pattern_cluster_transaction",
        ),
        Index("ix_m2_cluster_members_cluster", "pattern_id", "m2_cluster_id"),
        Index("ix_m2_cluster_members_accession", "filing_accession_number"),
    )

    m2_cluster_member_id = Column(String, primary_key=True, default=_uuid)
    pattern_id = Column(String, nullable=False)
    m2_cluster_id = Column(String, nullable=False)
    m2_cluster_signature_hash = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    transaction_id = Column(
        String, ForeignKey("m2_insider_transactions.transaction_id"), nullable=False
    )
    filing_accession_number = Column(String, nullable=True)
    insider_id = Column(String, nullable=False)
    insider_cik = Column(String, nullable=True)
    classification = Column(String, nullable=False)
    first_tradable_session = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


# ---------------------------------------------------------------------------
# M3 sector taxonomy and return producer tables
# ---------------------------------------------------------------------------
class FirmSectorAssignmentHistory(Base):
    """Point-in-time SIC-derived sector assignment intervals for M3."""

    __tablename__ = "firm_sector_assignments_history"
    __table_args__ = (
        Index(
            "ix_firm_sector_history_ticker_interval",
            "ticker",
            "valid_from",
            "valid_to",
        ),
        Index("ix_firm_sector_history_sector_interval", "sector", "valid_from", "valid_to"),
        Index("ix_firm_sector_history_source", "source"),
    )

    ticker = Column(String, primary_key=True)
    valid_from = Column(Date, primary_key=True)
    sector = Column(String, nullable=False)
    sic_code = Column(String, nullable=True)
    sic_description = Column(Text, nullable=True)
    industry = Column(String, nullable=True)
    source = Column(String, nullable=False, default="POLYGON_SIC")
    sic_to_sector_map_version = Column(String, nullable=False)
    valid_to = Column(Date, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class FirmSectorAssignment(Base):
    """Current/open M3 sector assignment snapshot."""

    __tablename__ = "firm_sector_assignments"
    __table_args__ = (
        Index("ix_firm_sector_assignments_sector", "sector"),
        Index("ix_firm_sector_assignments_last_verified", "last_verified"),
    )

    ticker = Column(String, primary_key=True)
    sector = Column(String, nullable=False)
    industry = Column(String, nullable=True)
    source = Column(String, nullable=False, default="POLYGON_SIC")
    classification_date = Column(Date, nullable=False)
    last_verified = Column(Date, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class SectorReturnDaily(Base):
    """PIT formation-cohort sector return surface consumed by M3."""

    __tablename__ = "sector_returns_daily"
    __table_args__ = (
        Index("ix_sector_returns_daily_rank", "date", "sector_rank"),
        Index("ix_sector_returns_daily_sector", "sector"),
    )

    date = Column(Date, primary_key=True)
    sector = Column(String, primary_key=True)
    return_6mo = Column(Float, nullable=False)
    return_6mo_ew = Column(Float, nullable=True)
    return_1mo = Column(Float, nullable=True)
    return_3mo = Column(Float, nullable=True)
    sector_rank = Column(Integer, nullable=False)
    sector_rank_normalized = Column(Float, nullable=False)
    n_sectors = Column(Integer, nullable=False)
    n_firms_in_sector = Column(Integer, nullable=False)
    total_market_cap_in_sector = Column(Float, nullable=False)
    source = Column(String, nullable=False, default="POLYGON_SIC")
    sic_to_sector_map_version = Column(String, nullable=False)
    formation_date = Column(Date, nullable=False)
    point_in_time_passed = Column(Boolean, nullable=False, default=True)
    formation_cohort_passed = Column(Boolean, nullable=False, default=True)
    sector_history_coverage_years = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class SectorChangeLog(Base):
    """Audit log for detected SIC/sector interval changes."""

    __tablename__ = "sector_change_log"
    __table_args__ = (
        Index("ix_sector_change_log_ticker_date", "ticker", "change_date"),
        Index("ix_sector_change_log_job_run_id", "job_run_id"),
    )

    sector_change_log_id = Column(String, primary_key=True, default=_uuid)
    ticker = Column(String, nullable=False)
    old_sector = Column(String, nullable=True)
    new_sector = Column(String, nullable=False)
    old_sic_code = Column(String, nullable=True)
    new_sic_code = Column(String, nullable=True)
    old_source = Column(String, nullable=True)
    new_source = Column(String, nullable=False)
    sic_to_sector_map_version = Column(String, nullable=False)
    change_date = Column(Date, nullable=False)
    detected_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    diagnostic_json = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# M3 schema-only STBM/validation parity tables
# ---------------------------------------------------------------------------
class M3ValidationMetadata(Base):
    """Post-exit M3 validation metadata parity table; no execution code writes it yet."""

    __tablename__ = "m3_validation_metadata"
    __table_args__ = (
        CheckConstraint(
            "fill_quality IN (0, 1)",
            name="ck_m3_validation_metadata_fill_quality",
        ),
        CheckConstraint(
            "thesis_category IN "
            "('right_tail_convex', 'continuation', 'mean_reversion', 'event_drift')",
            name="ck_m3_validation_metadata_thesis_category",
        ),
        CheckConstraint(
            "stop_state IN "
            "('pre_T1', 'post_T1_BE', 'post_T2_floor', 'post_T2_trailing', "
            "'closed', 'unfilled_expired')",
            name="ck_m3_validation_metadata_stop_state",
        ),
        CheckConstraint(
            "lifecycle_state IN "
            "('PRE_ENTRY', 'EXIT_ORDER_SETUP_PENDING', 'ACTIVE_PRE_T1', "
            "'ACTIVE_POST_T1_BE', 'ACTIVE_POST_T2_FLOOR', 'ACTIVE_POST_T2_TRAILING', "
            "'NATIVE_STOP_FLATTEN_REQUESTED', 'FRAMEWORK_EXIT_REQUESTED', "
            "'OCO_CANCEL_PENDING', 'MARKET_FLATTEN_SUBMITTED', "
            "'BROKER_FLAT_CONFIRMED', 'CLOSE_RECONCILIATION_PENDING', 'CLOSED')",
            name="ck_m3_validation_metadata_lifecycle_state",
        ),
        CheckConstraint(
            "final_exit_reason IS NULL OR final_exit_reason IN "
            "('T3_ceiling_hit', 'trailing_stop', 'hard_stop', 'break_even_stop', "
            "'floor_stop', 'time_barrier', 'setup_failure', 'universe_ejection', "
            "'circuit_breaker_flatten', 'optimizer_rebalance')",
            name="ck_m3_validation_metadata_final_exit_reason",
        ),
        CheckConstraint(
            "race_condition_resolution IS NULL OR race_condition_resolution IN "
            "('T1_T2_simultaneous', 'full_blowoff', 'T1_and_stop_simultaneous', "
            "'stop_during_replace', 'stop_during_trailing_update', "
            "'open_gap_through_stop', 'open_gap_through_target', "
            "'multi_target_gap_up', 'gap_through_stop_down', 'position_divergence', "
            "'tranche_state_anomaly', 'oco_state_drift', 'stop_invalid_at_submit', "
            "'websocket_disconnect_reconciled', 'network_failure_escalated', "
            "'framework_exit_during_native_fill', 't1_fill_stop_adjustment', "
            "'emergency_flatten_cancel_failure', 'fill_during_temp_stop_race')",
            name="ck_m3_validation_metadata_race_condition_resolution",
        ),
        CheckConstraint(
            "stbm_saga_state IS NULL OR stbm_saga_state IN "
            "('FRAMEWORK_EXIT_REQUESTED', 'NATIVE_STOP_FLATTEN_REQUESTED', "
            "'OCO_CANCEL_PENDING', 'MARKET_FLATTEN_SUBMITTED', "
            "'BROKER_FLAT_CONFIRMED', 'CLOSED')",
            name="ck_m3_validation_metadata_stbm_saga_state",
        ),
        CheckConstraint(
            "entry_unfilled_cancel_reason IS NULL OR entry_unfilled_cancel_reason IN "
            "('close_cutoff_cancel_confirmed', 'close_cutoff_cancel_unconfirmed', "
            "'intraday_cancel_complete', 'session_close_done_for_day')",
            name="ck_m3_validation_metadata_entry_unfilled_cancel_reason",
        ),
        CheckConstraint(
            "minimum_size_gate_failure_reason IS NULL OR "
            "minimum_size_gate_failure_reason IN "
            "('tranche_notional_below_$1', 'terminal_tranche_below_$5', "
            "'fractional_below_minimum')",
            name="ck_m3_validation_metadata_minimum_size_gate_failure_reason",
        ),
        CheckConstraint(
            "universe_ejection_reason IS NULL OR universe_ejection_reason IN "
            "('liquidity_score_zero', 'market_cap_out_of_band', "
            "'delisting_announced', 'fractionability_lost', "
            "'manual_operator_eject', 'corporate_action_ineligible')",
            name="ck_m3_validation_metadata_universe_ejection_reason",
        ),
        UniqueConstraint(
            "position_id",
            name="ux_m3_validation_metadata_position_id_fk",
        ),
        Index(
            "idx_m3_validation_metadata_position_id_unique",
            "position_id",
            unique=True,
            sqlite_where=text("position_id IS NOT NULL"),
            postgresql_where=text("position_id IS NOT NULL"),
        ),
    )

    signal_id = Column(
        String, ForeignKey("signal_registry.signal_id"), primary_key=True
    )
    candidate_id = Column(
        String, ForeignKey("trade_candidates.candidate_id"), nullable=False
    )
    position_id = Column(String, nullable=True)
    entry_fill_price = Column(Float, nullable=True)
    entry_filled_qty = Column(Float, nullable=False, default=0)
    entry_avg_fill_price = Column(Float, nullable=True)
    entry_fill_timestamp = Column(DateTime(timezone=True), nullable=True)
    fill_quality = Column(Integer, nullable=False)
    realized_slippage_bps = Column(Float, nullable=True)
    position_size_usd = Column(Float, nullable=True)
    tcb_max_position_pct = Column(Float, nullable=True)
    optimizer_max_position_pct = Column(Float, nullable=True)
    thesis_category = Column(String, nullable=False, default="continuation")
    t1_hit = Column(Boolean, nullable=False, default=False)
    t1_hit_timestamp = Column(DateTime(timezone=True), nullable=True)
    t1_hit_price = Column(Float, nullable=True)
    t2_hit = Column(Boolean, nullable=False, default=False)
    t2_hit_timestamp = Column(DateTime(timezone=True), nullable=True)
    t2_hit_price = Column(Float, nullable=True)
    t3_hit = Column(Boolean, nullable=False, default=False)
    t3_hit_timestamp = Column(DateTime(timezone=True), nullable=True)
    t3_hit_price = Column(Float, nullable=True)
    trailing_stop_active = Column(Boolean, nullable=False, default=False)
    trailing_stop_triggered = Column(Boolean, nullable=False, default=False)
    trailing_stop_trigger_price = Column(Float, nullable=True)
    hard_stop_triggered = Column(Boolean, nullable=False, default=False)
    time_barrier_triggered = Column(Boolean, nullable=False, default=False)
    stop_state = Column(String, nullable=False, default="pre_T1")
    lifecycle_state = Column(String, nullable=False, default="PRE_ENTRY")
    emergency_flatten_at_close = Column(Boolean, nullable=False, default=False)
    current_stop_price = Column(Float, nullable=True)
    t1_stop_adjustment_timestamp = Column(DateTime(timezone=True), nullable=True)
    t2_stop_adjustment_timestamp = Column(DateTime(timezone=True), nullable=True)
    order_group_id = Column(String, nullable=True)
    framework_exit_cleanup_timestamp = Column(DateTime(timezone=True), nullable=True)
    framework_exit_cleanup_failure_reason = Column(Text, nullable=True)
    parent_entry_order_id = Column(String, nullable=True)
    last_order_replace_timestamp = Column(DateTime(timezone=True), nullable=True)
    race_condition_resolution = Column(String, nullable=True)
    stbm_saga_state = Column(String, nullable=True)
    market_flatten_order_id = Column(String, nullable=True)
    market_flatten_attempts = Column(Integer, nullable=False, default=0)
    temp_protective_stop_order_id = Column(String, nullable=True)
    temp_protective_stop_active = Column(Boolean, nullable=False, default=False)
    temp_protective_stop_cancel_recreate_count = Column(Integer, nullable=False, default=0)
    temp_protective_stop_filled_during_entry = Column(Boolean, nullable=False, default=False)
    temp_protective_stop_fill_during_entry_race = Column(Boolean, nullable=False, default=False)
    pending_temp_stop_update_active = Column(Boolean, nullable=False, default=False)
    pending_temp_stop_update_target_price = Column(Float, nullable=True)
    pending_temp_stop_update_target_qty = Column(Float, nullable=True)
    entry_partial_fill = Column(Boolean, nullable=False, default=False)
    entry_target_qty = Column(Float, nullable=True)
    entry_fill_ratio = Column(Float, nullable=True)
    entry_unfilled_cancel_reason = Column(String, nullable=True)
    minimum_size_gate_failure = Column(Boolean, nullable=False, default=False)
    minimum_size_gate_failure_reason = Column(String, nullable=True)
    failed_tranche_label = Column(String, nullable=True)
    session_close_state = Column(Text, nullable=True)
    session_boundary_replays = Column(Integer, nullable=True, default=0)
    last_session_open_processed_at = Column(DateTime(timezone=True), nullable=True)
    exit_fill_price = Column(Float, nullable=True)
    exit_fill_timestamp = Column(DateTime(timezone=True), nullable=True)
    final_exit_reason = Column(String, nullable=True)
    universe_ejection_reason = Column(String, nullable=True)
    hold_duration_trading_days = Column(Integer, nullable=True)
    realized_return = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    mfe = Column(Float, nullable=True)
    realized_mfe_pct = Column(Float, nullable=False, default=0)
    realized_mfe_timestamp = Column(DateTime(timezone=True), nullable=True)
    optimizer_rebalance_rejected_convexity_lock = Column(Boolean, nullable=False, default=False)
    optimizer_rebalance_rejection_count = Column(Integer, nullable=False, default=0)
    high_since_t2 = Column(Float, nullable=True)
    n_firms_at_ranking_time = Column(Integer, nullable=True)
    sparse_sector_excluded = Column(Boolean, nullable=False, default=False)
    sparse_sector_warning = Column(Boolean, nullable=False, default=False)
    counterfactual_benchmark_return = Column(Float, nullable=True)
    counterfactual_sector_etf_return = Column(Float, nullable=True)
    illiq_premium_contribution = Column(Float, nullable=True)
    regime_contribution = Column(Float, nullable=True)
    sigma_epsilon_contribution = Column(Float, nullable=True)
    residual_m3_alpha = Column(Float, nullable=True)
    override_affected = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class M3TrancheFill(Base):
    """M3 tranche-fill audit rows for future STBM parity."""

    __tablename__ = "m3_tranche_fills"
    __table_args__ = (
        CheckConstraint(
            "tranche_label IN "
            "('T1', 'T2', 'T3', 'hard_stop', 'break_even_stop', 'floor_stop', "
            "'trailing_stop', 'time_barrier', 'universe_ejection', "
            "'circuit_breaker_flatten', 'optimizer_rebalance')",
            name="ck_m3_tranche_fills_tranche_label",
        ),
        CheckConstraint(
            "fill_type IN "
            "('take_profit', 'protective_stop', 'framework_exit', 'time_barrier_exit')",
            name="ck_m3_tranche_fills_fill_type",
        ),
        Index("ix_m3_tranche_fills_signal_id", "signal_id"),
        Index("ix_m3_tranche_fills_position_id", "position_id"),
    )

    fill_id = Column(String, primary_key=True, default=_uuid)
    signal_id = Column(String, ForeignKey("signal_registry.signal_id"), nullable=False)
    position_id = Column(
        String, ForeignKey("m3_validation_metadata.position_id"), nullable=False
    )
    oco_group_id = Column(String, nullable=False)
    tranche_label = Column(String, nullable=False)
    fill_quantity = Column(Float, nullable=False)
    fill_price = Column(Float, nullable=False)
    fill_timestamp = Column(DateTime(timezone=True), nullable=False)
    fill_type = Column(String, nullable=False)
    current_stop_level = Column(Float, nullable=True)


class M3OcoLegState(Base):
    """Per-leg OCO state table for future M3 STBM reconciliation."""

    __tablename__ = "m3_oco_leg_state"
    __table_args__ = (
        CheckConstraint(
            "tranche_label IN ('T1', 'T2', 'T3')",
            name="ck_m3_oco_leg_state_tranche_label",
        ),
        CheckConstraint(
            "leg_role IN ('take_profit', 'stop')",
            name="ck_m3_oco_leg_state_leg_role",
        ),
        CheckConstraint(
            "status IN "
            "('new', 'accepted', 'partially_filled', 'filled', 'pending_cancel', "
            "'pending_replace', 'canceled', 'expired', 'replaced', 'rejected', "
            "'suspended')",
            name="ck_m3_oco_leg_state_status",
        ),
        Index("ix_m3_oco_leg_state_signal_id", "signal_id"),
        Index("ix_m3_oco_leg_state_position_id", "position_id"),
        Index("ix_m3_oco_leg_state_broker_order", "broker_order_id"),
    )

    leg_state_id = Column(String, primary_key=True, default=_uuid)
    signal_id = Column(String, ForeignKey("signal_registry.signal_id"), nullable=False)
    position_id = Column(
        String, ForeignKey("m3_validation_metadata.position_id"), nullable=False
    )
    oco_group_id = Column(String, nullable=False)
    tranche_label = Column(String, nullable=False)
    leg_role = Column(String, nullable=False)
    broker_order_id = Column(String, nullable=False)
    status = Column(String, nullable=False)
    filled_qty = Column(Float, nullable=True, default=0)
    remaining_qty = Column(Float, nullable=False)
    avg_fill_price = Column(Float, nullable=True)
    intended_price = Column(Float, nullable=False)
    actual_price = Column(Float, nullable=True)
    last_broker_update = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


# ---------------------------------------------------------------------------
# data_lineage
# ---------------------------------------------------------------------------
class DataLineage(Base):
    """Persisted provenance for an external data payload or derived artifact."""

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
    """Pattern feature evidence captured before signal evaluation."""

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
    """Durable registry of pattern-intrinsic signal firings."""

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
    next_execution_session = Column(String, nullable=True)
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
    forward_return_observations = relationship(
        "ForwardReturnObservation", back_populates="signal"
    )
    forward_return_path_rows = relationship(
        "ForwardReturnPathRow", back_populates="signal"
    )
    forward_context_path_rows = relationship(
        "ForwardContextPathRow", back_populates="signal"
    )


# ---------------------------------------------------------------------------
# forward_return_observations
# ---------------------------------------------------------------------------
class ForwardReturnObservation(Base):
    """Latest canonical all-firings forward-return observation for a signal/input."""

    __tablename__ = "forward_return_observations"
    __table_args__ = (
        Index(
            "ux_forward_return_observations_signal_input",
            "signal_id",
            "input_hash",
            unique=True,
        ),
        Index(
            "ix_forward_return_observations_pattern_status",
            "pattern_id",
            "status",
        ),
        Index(
            "ix_forward_return_observations_ticker",
            "ticker",
        ),
    )

    forward_return_observation_id = Column(String, primary_key=True, default=_uuid)
    signal_id = Column(
        String, ForeignKey("signal_registry.signal_id"), nullable=False
    )
    pattern_id = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    signal_timestamp = Column(DateTime(timezone=True), nullable=False)
    signal_horizon = Column(String, nullable=True)
    next_execution_session = Column(String, nullable=True)
    entry_session_date = Column(String, nullable=True)
    entry_price = Column(Float, nullable=True)
    entry_price_source = Column(String, nullable=True)
    entry_basis_proof = Column(String, nullable=True)
    entry_data_lineage_id = Column(
        String, ForeignKey("data_lineage.data_lineage_id"), nullable=True
    )
    exit_session_date = Column(String, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_price_source = Column(String, nullable=True)
    exit_basis_proof = Column(String, nullable=True)
    exit_data_lineage_id = Column(
        String, ForeignKey("data_lineage.data_lineage_id"), nullable=True
    )
    forward_return = Column(Float, nullable=True)
    max_favorable_excursion = Column(Float, nullable=True)
    max_adverse_excursion = Column(Float, nullable=True)
    mfe_session_date = Column(String, nullable=True)
    mae_session_date = Column(String, nullable=True)
    max_close_return = Column(Float, nullable=True)
    min_close_return = Column(Float, nullable=True)
    hit_t1_intraday = Column(Boolean, nullable=True)
    hit_t2_intraday = Column(Boolean, nullable=True)
    hit_t3_intraday = Column(Boolean, nullable=True)
    hit_stop_intraday = Column(Boolean, nullable=True)
    same_day_barrier_ambiguity = Column(Boolean, nullable=True)
    status = Column(String, nullable=False)
    reason = Column(String, nullable=True)
    attempts = Column(Integer, nullable=False, default=0)
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    input_hash = Column(String, nullable=False)
    outcome_hash = Column(String, nullable=False)
    data_lineage_ids = Column(Text, nullable=True)
    provider = Column(String, nullable=True)
    endpoint = Column(String, nullable=True)
    provider_request_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    signal = relationship("SignalRegistry", back_populates="forward_return_observations")
    events = relationship(
        "ForwardReturnObservationEvent",
        back_populates="observation",
    )
    path_rows = relationship(
        "ForwardReturnPathRow",
        back_populates="observation",
        cascade="all, delete-orphan",
        order_by="ForwardReturnPathRow.path_sequence",
    )


# ---------------------------------------------------------------------------
# forward_return_path_rows
# ---------------------------------------------------------------------------
class ForwardReturnPathRow(Base):
    """Queryable per-session path captured for a forward-return observation.

    Consumer contract: current paths require row.outcome_hash ==
    observation.outcome_hash and an accepted finalized observation status;
    mismatched rows are last-good rows preserved across pathless retries.
    Return-from-entry values use the split-adjusted open entry basis and OHLC
    from one FMP /full response, sharing that response's adjustment basis.
    """

    __tablename__ = "forward_return_path_rows"
    __table_args__ = (
        UniqueConstraint(
            "forward_return_observation_id",
            "session_date",
            name="ux_forward_return_path_rows_observation_session",
        ),
        UniqueConstraint(
            "forward_return_observation_id",
            "path_sequence",
            name="ux_forward_return_path_rows_observation_sequence",
        ),
        Index(
            "ix_forward_return_path_rows_signal_session",
            "signal_id",
            "session_date",
        ),
        Index(
            "ix_forward_return_path_rows_pattern_ticker_session",
            "pattern_id",
            "ticker",
            "session_date",
        ),
    )

    forward_return_path_row_id = Column(String, primary_key=True, default=_uuid)
    forward_return_observation_id = Column(
        String,
        ForeignKey("forward_return_observations.forward_return_observation_id"),
        nullable=False,
    )
    signal_id = Column(
        String, ForeignKey("signal_registry.signal_id"), nullable=False
    )
    pattern_id = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    signal_horizon = Column(String, nullable=True)
    path_sequence = Column(Integer, nullable=False)
    session_date = Column(String, nullable=False)
    entry_session_date = Column(String, nullable=True)
    exit_session_date = Column(String, nullable=True)
    entry_price = Column(Float, nullable=True)
    open_price = Column(Float, nullable=True)
    high_price = Column(Float, nullable=True)
    low_price = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    split_adjusted_close = Column(Float, nullable=True)
    adj_close = Column(Float, nullable=True)
    return_from_entry_open = Column(Float, nullable=True)
    return_from_entry_high = Column(Float, nullable=True)
    return_from_entry_low = Column(Float, nullable=True)
    return_from_entry_close = Column(Float, nullable=True)
    is_entry_session = Column(Boolean, nullable=True)
    is_exit_session = Column(Boolean, nullable=True)
    expected_session_count = Column(Integer, nullable=True)
    path_status = Column(String, nullable=True)
    is_synthetic_exit = Column(Boolean, nullable=True)
    data_lineage_id = Column(
        String, ForeignKey("data_lineage.data_lineage_id"), nullable=True
    )
    provider = Column(String, nullable=True)
    endpoint = Column(String, nullable=True)
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    input_hash = Column(String, nullable=True)
    outcome_hash = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    observation = relationship("ForwardReturnObservation", back_populates="path_rows")
    signal = relationship("SignalRegistry", back_populates="forward_return_path_rows")


# ---------------------------------------------------------------------------
# forward_context_path_rows
# ---------------------------------------------------------------------------
class ForwardContextPathRow(Base):
    """Per-forward-session context snapshot for a live signal.

    These rows are observational panel data only. They must never feed entry
    signal generation or signal_identity_hash construction; consumers may read
    rows only up to their own decision moment.
    """

    __tablename__ = "forward_context_path_rows"
    __table_args__ = (
        UniqueConstraint(
            "signal_id",
            "forward_session_date",
            name="ux_forward_context_path_rows_signal_session",
        ),
        UniqueConstraint(
            "signal_id",
            "path_sequence",
            name="ux_forward_context_path_rows_signal_sequence",
        ),
        Index(
            "ix_forward_context_path_rows_signal_session",
            "signal_id",
            "forward_session_date",
        ),
        Index(
            "ix_forward_context_path_rows_pattern_ticker_session",
            "pattern_id",
            "ticker",
            "forward_session_date",
        ),
    )

    forward_context_path_row_id = Column(String, primary_key=True, default=_uuid)
    signal_id = Column(
        String, ForeignKey("signal_registry.signal_id"), nullable=False
    )
    pattern_id = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    signal_horizon = Column(String, nullable=True)
    forward_session_date = Column(String, nullable=False)
    path_sequence = Column(Integer, nullable=False)
    asof_timestamp = Column(DateTime(timezone=True), nullable=False)
    context_json = Column(Text, nullable=False)
    source_attempts_json = Column(Text, nullable=False)
    data_lineage_ids = Column(Text, nullable=False)
    context_hash = Column(String, nullable=False)
    is_terminal_snapshot = Column(Boolean, nullable=False, default=False)
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    signal = relationship("SignalRegistry", back_populates="forward_context_path_rows")


# ---------------------------------------------------------------------------
# forward_return_observation_events (append-only)
# ---------------------------------------------------------------------------
class ForwardReturnObservationEvent(Base):
    """Append-only attempt history for forward-return observation updates."""

    __tablename__ = "forward_return_observation_events"
    __table_args__ = (
        Index(
            "ix_forward_return_observation_events_signal_attempt",
            "signal_id",
            "attempts",
        ),
        Index(
            "ix_forward_return_observation_events_observation",
            "forward_return_observation_id",
        ),
    )

    forward_return_observation_event_id = Column(String, primary_key=True, default=_uuid)
    forward_return_observation_id = Column(
        String,
        ForeignKey("forward_return_observations.forward_return_observation_id"),
        nullable=True,
    )
    signal_id = Column(
        String, ForeignKey("signal_registry.signal_id"), nullable=False
    )
    pattern_id = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    signal_timestamp = Column(DateTime(timezone=True), nullable=False)
    signal_horizon = Column(String, nullable=True)
    next_execution_session = Column(String, nullable=True)
    entry_session_date = Column(String, nullable=True)
    entry_price = Column(Float, nullable=True)
    entry_price_source = Column(String, nullable=True)
    entry_basis_proof = Column(String, nullable=True)
    entry_data_lineage_id = Column(String, nullable=True)
    exit_session_date = Column(String, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_price_source = Column(String, nullable=True)
    exit_basis_proof = Column(String, nullable=True)
    exit_data_lineage_id = Column(String, nullable=True)
    forward_return = Column(Float, nullable=True)
    max_favorable_excursion = Column(Float, nullable=True)
    max_adverse_excursion = Column(Float, nullable=True)
    mfe_session_date = Column(String, nullable=True)
    mae_session_date = Column(String, nullable=True)
    max_close_return = Column(Float, nullable=True)
    min_close_return = Column(Float, nullable=True)
    hit_t1_intraday = Column(Boolean, nullable=True)
    hit_t2_intraday = Column(Boolean, nullable=True)
    hit_t3_intraday = Column(Boolean, nullable=True)
    hit_stop_intraday = Column(Boolean, nullable=True)
    same_day_barrier_ambiguity = Column(Boolean, nullable=True)
    status = Column(String, nullable=False)
    reason = Column(String, nullable=True)
    attempts = Column(Integer, nullable=False, default=0)
    job_run_id = Column(
        String, ForeignKey("evidence_job_runs.job_run_id"), nullable=True
    )
    input_hash = Column(String, nullable=False)
    outcome_hash = Column(String, nullable=False)
    data_lineage_ids = Column(Text, nullable=True)
    provider = Column(String, nullable=True)
    endpoint = Column(String, nullable=True)
    provider_request_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    observation = relationship(
        "ForwardReturnObservation",
        back_populates="events",
    )


# ---------------------------------------------------------------------------
# trade_candidates
# ---------------------------------------------------------------------------
class TradeCandidate(Base):
    """Candidate-stage trade decision record derived from a registered signal."""

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
    """KOTH/optimizer decision run and its auditable inputs/outputs."""

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
    """Broker/order lifecycle event persisted for execution reconstruction."""

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
    """Synthetic triple-barrier manager lifecycle event."""

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
    """Shadow-execution position outcome used to evaluate capture quality."""

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
    """Real broker position outcome and realized-return surface."""

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
    """Statistical validation run over mature all-firings outcomes."""

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
    """Manifest for agent-readable audit/export bundles."""

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
    """Runtime confidence and allocation weight for one pattern."""

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
    """Operator override with reason and audit metadata."""

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
