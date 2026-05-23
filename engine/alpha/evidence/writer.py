"""
Minimal evidence writer service.

Provides typed helpers to persist evidence records with the invariants
required by Engineering/Validation/EvidenceCapture.md:

  1. Every signal links to a feature snapshot and data lineage.
  2. Every candidate is persisted whether selected or skipped.
  3. Every order event is append-only.
  4. Every validation/export run has input hashes and output hashes.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from alpha.db.models import (
    AgentExportManifest,
    DataLineage,
    EvidenceJob,
    EvidenceJobRun,
    EvidenceSnapshot,
    FeatureSnapshot,
    OptimizerRun,
    OrderEvent,
    RealPosition,
    ShadowPosition,
    SignalRegistry,
    StbmLifecycleEvent,
    TradeCandidate,
    UniverseSnapshot,
    ValidationRun,
)


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Job / run lifecycle
# ---------------------------------------------------------------------------

def create_job(session: Session, *, name: str, job_type: str, owner: str) -> EvidenceJob:
    job = EvidenceJob(
        job_id=_uid(),
        job_name=name,
        job_type=job_type,
        owner_component=owner,
    )
    session.add(job)
    session.flush()
    return job


def start_run(
    session: Session,
    *,
    job_id: str,
    params: dict | None = None,
    app_commit_sha: str | None = None,
) -> EvidenceJobRun:
    run = EvidenceJobRun(
        job_run_id=_uid(),
        job_id=job_id,
        run_status="running",
        started_at=_now(),
        app_commit_sha=app_commit_sha,
        params_json=json.dumps(params) if params else None,
    )
    session.add(run)
    session.flush()
    return run


def finish_run(
    session: Session,
    run: EvidenceJobRun,
    *,
    status: str = "finished",
    metrics: dict | None = None,
    input_hashes: dict | None = None,
    output_hashes: dict | None = None,
    error: dict | None = None,
) -> EvidenceJobRun:
    run.run_status = status
    run.ended_at = _now()
    if metrics:
        run.metric_json = json.dumps(metrics)
    if input_hashes:
        run.input_hashes = json.dumps(input_hashes)
    if output_hashes:
        run.output_hashes = json.dumps(output_hashes)
    if error:
        run.error_json = json.dumps(error)
    session.flush()
    return run


# ---------------------------------------------------------------------------
# Data lineage
# ---------------------------------------------------------------------------

def record_data_lineage(
    session: Session,
    *,
    provider: str,
    endpoint: str,
    asof_timestamp: datetime,
    raw_payload: Any | None = None,
    raw_payload_hash: str | None = None,
    request_timestamp: datetime | None = None,
    freshness_seconds: float | None = None,
    source_authority: str | None = None,
    data_quality_flags: dict | None = None,
    job_run_id: str | None = None,
    dataset_id: str | None = None,
) -> DataLineage:
    if raw_payload_hash is None:
        raw_payload_hash = _hash(raw_payload)

    lineage = DataLineage(
        data_lineage_id=_uid(),
        provider=provider,
        endpoint=endpoint,
        request_timestamp=request_timestamp or _now(),
        asof_timestamp=asof_timestamp,
        raw_payload_hash=raw_payload_hash,
        freshness_seconds=freshness_seconds,
        source_authority=source_authority,
        data_quality_flags=(
            json.dumps(data_quality_flags) if data_quality_flags is not None else None
        ),
        job_run_id=job_run_id,
        dataset_id=dataset_id,
    )
    session.add(lineage)
    session.flush()
    return lineage


# ---------------------------------------------------------------------------
# Universe snapshot
# ---------------------------------------------------------------------------

def record_universe_snapshot(
    session: Session,
    *,
    ticker: str,
    asof_timestamp: datetime,
    operating_universe_inclusion: bool,
    job_run_id: str | None = None,
    evidence_snapshot_id: str | None = None,
    scan_id: str | None = None,
    source_provider: str | None = None,
    market_cap: float | None = None,
    price: float | None = None,
    security_type: str | None = None,
    primary_exchange: str | None = None,
    fractionable: bool | None = None,
    liquidity_score: float | None = None,
    median_dollar_volume_20d: float | None = None,
    median_dollar_volume_60d: float | None = None,
    high_low_range_proxy_20d: float | None = None,
    sub_dollar_exception_flag: bool | None = None,
    hazard_score: float | None = None,
    active_vetoes: list[str] | None = None,
    exclusion_reason: str | None = None,
    dataset_version: str | None = None,
    schema_hash: str | None = None,
    source_lineage_hash: str | None = None,
) -> UniverseSnapshot:
    snap = UniverseSnapshot(
        universe_snapshot_id=_uid(),
        evidence_snapshot_id=evidence_snapshot_id,
        job_run_id=job_run_id,
        scan_id=scan_id,
        ticker=ticker,
        asof_timestamp=asof_timestamp,
        source_provider=source_provider,
        market_cap=market_cap,
        price=price,
        security_type=security_type,
        primary_exchange=primary_exchange,
        fractionable=fractionable,
        liquidity_score=liquidity_score,
        median_dollar_volume_20d=median_dollar_volume_20d,
        median_dollar_volume_60d=median_dollar_volume_60d,
        high_low_range_proxy_20d=high_low_range_proxy_20d,
        sub_dollar_exception_flag=sub_dollar_exception_flag,
        hazard_score=hazard_score,
        active_vetoes=json.dumps(active_vetoes) if active_vetoes is not None else None,
        operating_universe_inclusion=operating_universe_inclusion,
        exclusion_reason=exclusion_reason,
        dataset_version=dataset_version,
        schema_hash=schema_hash,
        source_lineage_hash=source_lineage_hash,
    )
    session.add(snap)
    session.flush()
    return snap


# ---------------------------------------------------------------------------
# Feature snapshot
# ---------------------------------------------------------------------------

def record_feature_snapshot(
    session: Session,
    *,
    pattern_id: str,
    ticker: str,
    asof_timestamp: datetime,
    features: dict,
    data_lineage_ids: list[str],
    job_run_id: str | None = None,
    evidence_snapshot_id: str | None = None,
    feature_manifest_version: str | None = None,
    code_commit_sha: str | None = None,
    fidelity_tier: str | None = None,
    point_in_time_passed: bool | None = None,
    lookahead_guard_passed: bool | None = None,
    input_hashes: dict | None = None,
) -> FeatureSnapshot:
    snap = FeatureSnapshot(
        feature_snapshot_id=_uid(),
        job_run_id=job_run_id,
        evidence_snapshot_id=evidence_snapshot_id,
        pattern_id=pattern_id,
        ticker=ticker,
        asof_timestamp=asof_timestamp,
        feature_manifest_version=feature_manifest_version,
        feature_json=json.dumps(features, default=str),
        feature_hash=_hash(features),
        data_lineage_ids=json.dumps(data_lineage_ids),
        code_commit_sha=code_commit_sha,
        fidelity_tier=fidelity_tier,
        point_in_time_passed=point_in_time_passed,
        lookahead_guard_passed=lookahead_guard_passed,
        input_hashes=json.dumps(input_hashes) if input_hashes else None,
        output_hash=_hash(features),
    )
    session.add(snap)
    session.flush()
    return snap


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

def record_signal(
    session: Session,
    *,
    pattern_id: str,
    ticker: str,
    direction: str,
    signal_timestamp: datetime,
    raw_signal_strength: float,
    raw_expected_edge: float,
    feature_snapshot_id: str,
    job_run_id: str | None = None,
    signal_status: str = "active",
    signal_horizon: str | None = None,
    thesis_category: str | None = None,
    route_class: str | None = None,
    fidelity_tier: str | None = None,
    data_confidence: float | None = None,
    data_lineage_ids: list[str] | None = None,
    universe_snapshot_id: str | None = None,
    signal_event_sequence: int | None = None,
) -> SignalRegistry:
    sig = SignalRegistry(
        signal_id=_uid(),
        job_run_id=job_run_id,
        pattern_id=pattern_id,
        ticker=ticker,
        direction=direction,
        signal_timestamp=signal_timestamp,
        raw_signal_strength=raw_signal_strength,
        raw_expected_edge=raw_expected_edge,
        feature_snapshot_id=feature_snapshot_id,
        signal_status=signal_status,
        signal_horizon=signal_horizon,
        thesis_category=thesis_category,
        route_class=route_class,
        fidelity_tier=fidelity_tier,
        data_confidence=data_confidence,
        universe_snapshot_id=universe_snapshot_id,
        signal_event_sequence=signal_event_sequence,
        data_lineage_ids=json.dumps(data_lineage_ids) if data_lineage_ids else None,
    )
    session.add(sig)
    session.flush()
    return sig


# ---------------------------------------------------------------------------
# Trade candidate — persists ALL candidates (enter, skip, veto)
# ---------------------------------------------------------------------------

def record_candidate(
    session: Session,
    *,
    candidate_pool_id: str,
    ticker: str,
    direction: str,
    primary_pattern: str,
    combined_expected_edge: float,
    trade_decision: str,
    input_signal_ids: list[str],
    job_run_id: str | None = None,
    scan_id: str | None = None,
    active_patterns: list[str] | None = None,
    effective_hard_stop_pct: float | None = None,
    base_risk_budget_pct: float | None = None,
    risk_budget_pct: float | None = None,
    risk_multiplier_product: float | None = None,
    risk_sized_cap: float | None = None,
    unstopped_heat_pct: float | None = None,
    expected_round_trip_cost: float | None = None,
    cost_to_edge_ratio: float | None = None,
    missed_fill_adjustment: float | None = None,
    optimizer_input_expected_edge: float | None = None,
    validation_weight_multiplier: float | None = None,
    pattern_weight: float | None = None,
    shrinkage_weight: float | None = None,
    hazard_multiplier: float | None = None,
    liquidity_multiplier: float | None = None,
    fidelity_multiplier: float | None = None,
    max_position_pct: float | None = None,
    catalyst_cluster: str | None = None,
    same_symbol_state: str | None = None,
    skip_reason: str | None = None,
    rank: int | None = None,
    percentile: float | None = None,
    cash_available: float | None = None,
    settled_cash_required: float | None = None,
    constraint_reason: dict | None = None,
) -> TradeCandidate:
    cand = TradeCandidate(
        candidate_id=_uid(),
        job_run_id=job_run_id,
        candidate_pool_id=candidate_pool_id,
        scan_id=scan_id,
        ticker=ticker,
        direction=direction,
        primary_pattern=primary_pattern,
        active_patterns=json.dumps(active_patterns) if active_patterns else None,
        combined_expected_edge=combined_expected_edge,
        effective_hard_stop_pct=effective_hard_stop_pct,
        base_risk_budget_pct=base_risk_budget_pct,
        risk_budget_pct=risk_budget_pct,
        risk_multiplier_product=risk_multiplier_product,
        risk_sized_cap=risk_sized_cap,
        unstopped_heat_pct=unstopped_heat_pct,
        expected_round_trip_cost=expected_round_trip_cost,
        cost_to_edge_ratio=cost_to_edge_ratio,
        missed_fill_adjustment=missed_fill_adjustment,
        optimizer_input_expected_edge=optimizer_input_expected_edge,
        validation_weight_multiplier=validation_weight_multiplier,
        pattern_weight=pattern_weight,
        shrinkage_weight=shrinkage_weight,
        hazard_multiplier=hazard_multiplier,
        liquidity_multiplier=liquidity_multiplier,
        fidelity_multiplier=fidelity_multiplier,
        max_position_pct=max_position_pct,
        trade_decision=trade_decision,
        input_signal_ids=json.dumps(input_signal_ids),
        skip_reason=skip_reason,
        catalyst_cluster=catalyst_cluster,
        same_symbol_state=same_symbol_state,
        candidate_rank_pre_optimizer=rank,
        candidate_percentile_pre_optimizer=percentile,
        cash_available_at_decision=cash_available,
        settled_cash_required=settled_cash_required,
        constraint_reason_json=json.dumps(constraint_reason) if constraint_reason else None,
    )
    session.add(cand)
    session.flush()
    return cand


# ---------------------------------------------------------------------------
# Order event — append-only, never update prior rows
# ---------------------------------------------------------------------------

def append_order_event(
    session: Session,
    *,
    order_request_id: str,
    event_type: str,
    event_sequence: int,
    event_timestamp: datetime,
    candidate_id: str | None = None,
    real_position_id: str | None = None,
    order_ticket_id: str | None = None,
    broker_order_id: str | None = None,
    route_class: str | None = None,
    request_type: str | None = None,
    broker_status: str | None = None,
    broker_response_status: str | None = None,
    intended_price: float | None = None,
    submitted_price: float | None = None,
    filled_avg_price: float | None = None,
    filled_qty: float | None = None,
    cumulative_filled_qty: float | None = None,
    cumulative_avg_fill_price: float | None = None,
    slippage_bps: float | None = None,
    fill_quality: str | None = None,
    reject_reason: str | None = None,
    cancel_reason: str | None = None,
) -> OrderEvent:
    evt = OrderEvent(
        order_event_id=_uid(),
        order_request_id=order_request_id,
        candidate_id=candidate_id,
        real_position_id=real_position_id,
        order_ticket_id=order_ticket_id,
        broker_order_id=broker_order_id,
        route_class=route_class,
        request_type=request_type,
        event_type=event_type,
        event_sequence=event_sequence,
        broker_status=broker_status,
        broker_response_status=broker_response_status,
        intended_price=intended_price,
        submitted_price=submitted_price,
        filled_avg_price=filled_avg_price,
        filled_qty=filled_qty,
        cumulative_filled_qty=cumulative_filled_qty,
        cumulative_avg_fill_price=cumulative_avg_fill_price,
        slippage_bps=slippage_bps,
        fill_quality=fill_quality,
        reject_reason=reject_reason,
        cancel_reason=cancel_reason,
        event_timestamp=event_timestamp,
    )
    session.add(evt)
    session.flush()
    return evt


# ---------------------------------------------------------------------------
# Validation run — with input/output hashes
# ---------------------------------------------------------------------------

def record_validation_run(
    session: Session,
    *,
    job_run_id: str,
    run_type: str,
    pattern_id: str | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    sample_size: int | None = None,
    params: dict | None = None,
    metrics: dict | None = None,
    tags: dict | None = None,
    confidence_tier: str | None = None,
    validation_weight_multiplier: float | None = None,
    run_status: str = "finished",
    input_snapshot_ids: list[str] | None = None,
    input_dataset_ids: list[str] | None = None,
    input_hashes: dict | None = None,
    output_hashes: dict | None = None,
    error: dict | None = None,
) -> ValidationRun:
    vr = ValidationRun(
        validation_run_id=_uid(),
        job_run_id=job_run_id,
        pattern_id=pattern_id,
        run_type=run_type,
        run_status=run_status,
        window_start=window_start,
        window_end=window_end,
        sample_size=sample_size,
        params_json=json.dumps(params) if params else None,
        metric_json=json.dumps(metrics) if metrics else None,
        tag_json=json.dumps(tags) if tags else None,
        confidence_tier=confidence_tier,
        validation_weight_multiplier=validation_weight_multiplier,
        input_snapshot_ids=json.dumps(input_snapshot_ids) if input_snapshot_ids else None,
        input_dataset_ids=json.dumps(input_dataset_ids) if input_dataset_ids else None,
        input_hashes=json.dumps(input_hashes) if input_hashes else None,
        output_hashes=json.dumps(output_hashes) if output_hashes else None,
        error_json=json.dumps(error) if error else None,
    )
    session.add(vr)
    session.flush()
    return vr


# ---------------------------------------------------------------------------
# Agent export manifest — with hashes
# ---------------------------------------------------------------------------

def record_export_manifest(
    session: Session,
    *,
    job_run_id: str | None = None,
    evidence_snapshot_id: str | None = None,
    created_by: str | None = None,
    pattern_scope: list[str] | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    included_tables: list[str] | None = None,
    manifest_hash: str | None = None,
    export_path: str | None = None,
    source_dataset_ids: list[str] | None = None,
) -> AgentExportManifest:
    em = AgentExportManifest(
        export_id=_uid(),
        job_run_id=job_run_id,
        evidence_snapshot_id=evidence_snapshot_id,
        created_by=created_by,
        pattern_scope=json.dumps(pattern_scope) if pattern_scope else None,
        window_start=window_start,
        window_end=window_end,
        included_tables=json.dumps(included_tables) if included_tables else None,
        manifest_hash=manifest_hash,
        export_path=export_path,
        source_dataset_ids=json.dumps(source_dataset_ids) if source_dataset_ids else None,
    )
    session.add(em)
    session.flush()
    return em
