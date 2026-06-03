"""
Tests proving the four evidence-spine invariants from the vault spec:

  1. Every signal links to a feature snapshot and data lineage.
  2. Every candidate is persisted whether selected or skipped.
  3. Every order event is append-only.
  4. Every validation/export run has input hashes and output hashes.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from alpha.db.models import (
    AgentExportManifest,
    Base,
    OrderEvent,
    SignalRegistry,
    TradeCandidate,
    ValidationRun,
)
from alpha.evidence.writer import (
    append_order_event,
    create_job,
    finish_run,
    record_candidate,
    record_data_lineage,
    record_export_manifest,
    record_feature_snapshot,
    record_signal,
    record_universe_snapshot,
    record_validation_run,
    start_run,
)


def _ts():
    return datetime(2026, 5, 20, 14, 30, 0, tzinfo=timezone.utc)


# -------------------------------------------------------------------
# Invariant 1: every signal links to a feature snapshot and data lineage
# -------------------------------------------------------------------

class TestSignalLineage:
    def test_signal_links_feature_snapshot(self, db_session):
        job = create_job(
            db_session,
            name="m4_detector_nightly",
            job_type="detector",
            owner="pattern_engine",
        )
        run = start_run(db_session, job_id=job.job_id)
        universe = record_universe_snapshot(
            db_session,
            job_run_id=run.job_run_id,
            ticker="ACME",
            asof_timestamp=_ts(),
            source_provider="FMP",
            market_cap=75_000_000,
            price=5.0,
            operating_universe_inclusion=True,
        )
        lineage = record_data_lineage(
            db_session,
            provider="FMP",
            endpoint="/v3/historical-price-full/ACME",
            asof_timestamp=_ts(),
            raw_payload={"close": [1.0, 2.0, 3.0]},
            job_run_id=run.job_run_id,
        )
        assert json.loads(lineage.raw_payload_json) == {"close": [1.0, 2.0, 3.0]}

        feat = record_feature_snapshot(
            db_session,
            pattern_id="M4",
            ticker="ACME",
            asof_timestamp=_ts(),
            features={"price_to_52wk_high": 0.97},
            data_lineage_ids=[lineage.data_lineage_id],
            job_run_id=run.job_run_id,
            code_commit_sha="abc123",
            fidelity_tier="FULL",
            point_in_time_passed=True,
            lookahead_guard_passed=True,
        )

        sig = record_signal(
            db_session,
            pattern_id="M4",
            ticker="ACME",
            direction="long",
            signal_timestamp=_ts(),
            raw_signal_strength=0.97,
            raw_expected_edge=0.08,
            feature_snapshot_id=feat.feature_snapshot_id,
            job_run_id=run.job_run_id,
            universe_snapshot_id=universe.universe_snapshot_id,
            route_class="A",
            fidelity_tier="FULL",
            data_lineage_ids=[lineage.data_lineage_id],
            signal_identity_hash="writer-linkage-signal",
        )

        db_session.commit()

        # Verify FK chain: signal -> feature_snapshot -> data_lineage
        stored = db_session.get(SignalRegistry, sig.signal_id)
        assert stored.feature_snapshot_id == feat.feature_snapshot_id
        assert stored.job_run_id == run.job_run_id
        assert stored.universe_snapshot_id == universe.universe_snapshot_id

        linked_feat = stored.feature_snapshot
        assert linked_feat is not None
        assert linked_feat.feature_hash != ""

        lineage_ids = json.loads(linked_feat.data_lineage_ids)
        assert lineage.data_lineage_id in lineage_ids

    def test_signal_requires_feature_snapshot_fk(self, db_session):
        """Signal cannot be created without a valid feature_snapshot_id."""
        from sqlalchemy.exc import IntegrityError
        import pytest

        with pytest.raises(IntegrityError):
            record_signal(
                db_session,
                pattern_id="M4",
                ticker="ACME",
                direction="long",
                signal_timestamp=_ts(),
                raw_signal_strength=0.5,
                raw_expected_edge=0.04,
                feature_snapshot_id="nonexistent-id",
                signal_identity_hash="missing-feature-signal",
            )
            db_session.commit()

    def test_signal_requires_identity_hash(self, db_session):
        """Signal identity is a writer-level invariant, not just orchestration policy."""
        import pytest

        with pytest.raises(ValueError):
            record_signal(
                db_session,
                pattern_id="M4",
                ticker="ACME",
                direction="long",
                signal_timestamp=_ts(),
                raw_signal_strength=0.5,
                raw_expected_edge=0.04,
                feature_snapshot_id="nonexistent-id",
            )


# -------------------------------------------------------------------
# Invariant 2: every candidate persisted whether selected or skipped
# -------------------------------------------------------------------

class TestCandidatePersistence:
    def _setup_signal(self, db_session):
        lineage = record_data_lineage(
            db_session,
            provider="FMP",
            endpoint="/v3/quote/ACME",
            asof_timestamp=_ts(),
            raw_payload={"price": 5.0},
        )
        feat = record_feature_snapshot(
            db_session,
            pattern_id="M4",
            ticker="ACME",
            asof_timestamp=_ts(),
            features={"ratio": 0.95},
            data_lineage_ids=[lineage.data_lineage_id],
        )
        sig = record_signal(
            db_session,
            pattern_id="M4",
            ticker="ACME",
            direction="long",
            signal_timestamp=_ts(),
            raw_signal_strength=0.95,
            raw_expected_edge=0.06,
            feature_snapshot_id=feat.feature_snapshot_id,
            signal_identity_hash="candidate-input-signal",
        )
        db_session.flush()
        return sig

    def test_enter_candidate_persisted(self, db_session):
        sig = self._setup_signal(db_session)
        cand = record_candidate(
            db_session,
            candidate_pool_id="pool-001",
            ticker="ACME",
            direction="long",
            primary_pattern="M4",
            combined_expected_edge=0.06,
            trade_decision="enter",
            input_signal_ids=[sig.signal_id],
            active_patterns=["M4"],
            effective_hard_stop_pct=0.06,
            base_risk_budget_pct=0.01,
            risk_budget_pct=0.0049,
            risk_multiplier_product=0.49,
            risk_sized_cap=0.0817,
            unstopped_heat_pct=None,
            expected_round_trip_cost=0.012,
            cost_to_edge_ratio=0.20,
            missed_fill_adjustment=0.003,
            optimizer_input_expected_edge=0.045,
            validation_weight_multiplier=1.0,
            pattern_weight=1.0,
            shrinkage_weight=0.85,
            hazard_multiplier=1.0,
            liquidity_multiplier=1.0,
            fidelity_multiplier=1.0,
            max_position_pct=0.40,
            catalyst_cluster="earnings_momentum",
            same_symbol_state="clear",
            rank=1,
            percentile=0.95,
            cash_available=900.0,
            settled_cash_required=350.0,
        )
        db_session.commit()

        stored = db_session.get(TradeCandidate, cand.candidate_id)
        assert stored.trade_decision == "enter"
        assert stored.skip_reason is None
        assert stored.optimizer_input_expected_edge == 0.045
        assert stored.validation_weight_multiplier == 1.0
        assert stored.effective_hard_stop_pct == 0.06
        assert stored.base_risk_budget_pct == 0.01
        assert stored.risk_budget_pct == 0.0049
        assert stored.risk_multiplier_product == 0.49
        assert stored.risk_sized_cap == 0.0817
        assert stored.unstopped_heat_pct is None
        assert stored.cost_to_edge_ratio == 0.20
        assert stored.max_position_pct == 0.40
        assert stored.catalyst_cluster == "earnings_momentum"

    def test_skipped_candidate_persisted(self, db_session):
        sig = self._setup_signal(db_session)
        cand = record_candidate(
            db_session,
            candidate_pool_id="pool-001",
            ticker="ACME",
            direction="long",
            primary_pattern="M4",
            combined_expected_edge=0.02,
            trade_decision="skip",
            input_signal_ids=[sig.signal_id],
            skip_reason="insufficient_edge",
            rank=5,
            percentile=0.30,
            cash_available=900.0,
        )
        db_session.commit()

        stored = db_session.get(TradeCandidate, cand.candidate_id)
        assert stored.trade_decision == "skip"
        assert stored.skip_reason == "insufficient_edge"

    def test_vetoed_candidate_persisted(self, db_session):
        sig = self._setup_signal(db_session)
        cand = record_candidate(
            db_session,
            candidate_pool_id="pool-001",
            ticker="ACME",
            direction="long",
            primary_pattern="M4",
            combined_expected_edge=0.10,
            trade_decision="vetoed_hazard",
            input_signal_ids=[sig.signal_id],
            skip_reason="hazard_score_exceeded",
        )
        db_session.commit()

        stored = db_session.get(TradeCandidate, cand.candidate_id)
        assert stored.trade_decision == "vetoed_hazard"
        assert stored.skip_reason == "hazard_score_exceeded"

    def test_all_decisions_in_same_pool(self, db_session):
        sig = self._setup_signal(db_session)
        pool = "pool-002"
        for decision, reason in [
            ("enter", None),
            ("skip", "low_edge"),
            ("vetoed_liquidity", "liquidity_score_zero"),
        ]:
            record_candidate(
                db_session,
                candidate_pool_id=pool,
                ticker="ACME",
                direction="long",
                primary_pattern="M4",
                combined_expected_edge=0.05,
                trade_decision=decision,
                input_signal_ids=[sig.signal_id],
                skip_reason=reason,
            )
        db_session.commit()

        all_cands = (
            db_session.query(TradeCandidate)
            .filter(TradeCandidate.candidate_pool_id == pool)
            .all()
        )
        decisions = {c.trade_decision for c in all_cands}
        assert decisions == {"enter", "skip", "vetoed_liquidity"}

    def test_generic_veto_is_rejected(self, db_session):
        from sqlalchemy.exc import IntegrityError
        import pytest

        sig = self._setup_signal(db_session)
        with pytest.raises(IntegrityError):
            record_candidate(
                db_session,
                candidate_pool_id="pool-003",
                ticker="ACME",
                direction="long",
                primary_pattern="M4",
                combined_expected_edge=0.05,
                trade_decision="veto",
                input_signal_ids=[sig.signal_id],
                skip_reason="ambiguous_veto",
            )
            db_session.commit()


# -------------------------------------------------------------------
# Invariant 3: order events are append-only
# -------------------------------------------------------------------

class TestOrderEventAppendOnly:
    def test_multiple_events_per_order(self, db_session):
        req_id = "req-001"
        events = []
        for seq, etype in enumerate(
            ["submit", "broker_ack", "partial_fill", "full_fill"], start=1
        ):
            evt = append_order_event(
                db_session,
                order_request_id=req_id,
                event_type=etype,
                event_sequence=seq,
                event_timestamp=_ts(),
                broker_order_id="BRK-42",
                order_ticket_id="ticket-42",
                route_class="A",
                request_type="market_buy",
                broker_response_status="accepted",
                fill_quality="filled" if etype == "full_fill" else None,
            )
            events.append(evt)
        db_session.commit()

        stored = (
            db_session.query(OrderEvent)
            .filter(OrderEvent.order_request_id == req_id)
            .order_by(OrderEvent.event_sequence)
            .all()
        )
        assert len(stored) == 4
        assert [e.event_type for e in stored] == [
            "submit",
            "broker_ack",
            "partial_fill",
            "full_fill",
        ]
        # Each event has its own unique PK
        ids = [e.order_event_id for e in stored]
        assert len(set(ids)) == 4
        assert stored[0].order_ticket_id == "ticket-42"
        assert stored[0].broker_response_status == "accepted"
        assert stored[-1].fill_quality == "filled"

    def test_cancel_and_reject_are_separate_events(self, db_session):
        req_id = "req-002"
        append_order_event(
            db_session,
            order_request_id=req_id,
            event_type="submit",
            event_sequence=1,
            event_timestamp=_ts(),
        )
        append_order_event(
            db_session,
            order_request_id=req_id,
            event_type="reject",
            event_sequence=2,
            event_timestamp=_ts(),
            reject_reason="insufficient_buying_power",
        )
        db_session.commit()

        stored = (
            db_session.query(OrderEvent)
            .filter(OrderEvent.order_request_id == req_id)
            .all()
        )
        assert len(stored) == 2
        reject_evt = [e for e in stored if e.event_type == "reject"][0]
        assert reject_evt.reject_reason == "insufficient_buying_power"

    def test_no_update_to_prior_events(self, db_session):
        """Prior events remain immutable — new state is a new row."""
        req_id = "req-003"
        evt1 = append_order_event(
            db_session,
            order_request_id=req_id,
            event_type="submit",
            event_sequence=1,
            event_timestamp=_ts(),
            intended_price=5.00,
        )
        db_session.commit()
        original_id = evt1.order_event_id

        # A replacement is a NEW event, not an update to the old one
        evt2 = append_order_event(
            db_session,
            order_request_id=req_id,
            event_type="replace",
            event_sequence=2,
            event_timestamp=_ts(),
            intended_price=5.10,
        )
        db_session.commit()

        original = db_session.get(OrderEvent, original_id)
        assert original.intended_price == 5.00  # unchanged
        assert evt2.intended_price == 5.10
        assert evt2.order_event_id != original_id


# -------------------------------------------------------------------
# Invariant 4: validation/export runs have input and output hashes
# -------------------------------------------------------------------

class TestValidationHashes:
    def _make_job_run(self, db_session):
        job = create_job(
            db_session,
            name="monthly_factor_premia",
            job_type="validation",
            owner="validation_engine",
        )
        run = start_run(db_session, job_id=job.job_id)
        db_session.flush()
        return run

    def test_validation_run_has_hashes(self, db_session):
        run = self._make_job_run(db_session)
        input_h = {"shadow_positions": "sha256:aaa", "signal_registry": "sha256:bbb"}
        output_h = {"factor_premia_csv": "sha256:ccc"}

        vr = record_validation_run(
            db_session,
            job_run_id=run.job_run_id,
            run_type="monthly_factor_premia",
            pattern_id="M4",
            sample_size=250,
            params={"window_months": 3},
            metrics={"t_stat": 2.34, "lambda": 0.012},
            confidence_tier="full",
            validation_weight_multiplier=1.0,
            input_hashes=input_h,
            output_hashes=output_h,
        )
        db_session.commit()

        stored = db_session.get(ValidationRun, vr.validation_run_id)
        assert json.loads(stored.input_hashes) == input_h
        assert json.loads(stored.output_hashes) == output_h
        assert stored.confidence_tier == "full"
        assert stored.validation_weight_multiplier == 1.0

    def test_validation_run_persists_on_failure(self, db_session):
        run = self._make_job_run(db_session)
        vr = record_validation_run(
            db_session,
            job_run_id=run.job_run_id,
            run_type="monthly_factor_premia",
            pattern_id="M4",
            run_status="failed",
            input_hashes={"shadow_positions": "sha256:aaa"},
            output_hashes=None,
            error={"msg": "insufficient sample"},
        )
        # Mark parent run as failed
        finish_run(
            db_session,
            run,
            status="failed",
            input_hashes={"shadow_positions": "sha256:aaa"},
            error={"msg": "insufficient sample"},
        )
        db_session.commit()

        stored_run = db_session.get(type(run), run.job_run_id)
        stored_vr = db_session.get(ValidationRun, vr.validation_run_id)
        assert stored_run.run_status == "failed"
        assert json.loads(stored_run.input_hashes) == {"shadow_positions": "sha256:aaa"}
        assert json.loads(stored_run.error_json)["msg"] == "insufficient sample"
        assert stored_vr.run_status == "failed"
        assert json.loads(stored_vr.error_json)["msg"] == "insufficient sample"

    def test_export_manifest_has_hash(self, db_session):
        run = self._make_job_run(db_session)
        em = record_export_manifest(
            db_session,
            job_run_id=run.job_run_id,
            created_by="operator",
            pattern_scope=["M4", "M6"],
            included_tables=["signal_registry", "trade_candidates", "shadow_positions"],
            manifest_hash="sha256:deadbeef",
            export_path="/exports/M4_M6_2026Q1.tar.gz",
            source_dataset_ids=["ds-001", "ds-002"],
        )
        db_session.commit()

        stored = db_session.get(AgentExportManifest, em.export_id)
        assert stored.manifest_hash == "sha256:deadbeef"
        assert json.loads(stored.pattern_scope) == ["M4", "M6"]
        assert json.loads(stored.source_dataset_ids) == ["ds-001", "ds-002"]

    def test_job_run_finish_records_both_hashes(self, db_session):
        run = self._make_job_run(db_session)
        input_h = {"signals": "sha256:111"}
        output_h = {"report": "sha256:222"}
        finish_run(
            db_session,
            run,
            status="finished",
            metrics={"rows_processed": 1500},
            input_hashes=input_h,
            output_hashes=output_h,
        )
        db_session.commit()

        stored = db_session.get(type(run), run.job_run_id)
        assert stored.run_status == "finished"
        assert json.loads(stored.input_hashes) == input_h
        assert json.loads(stored.output_hashes) == output_h


# -------------------------------------------------------------------
# Schema completeness: verify all evidence tables exist
# -------------------------------------------------------------------

class TestSchemaCompleteness:
    def test_all_tables_created_by_metadata(self, db_session):
        expected = {
            "security_profiles",
            "security_profile_scan_snapshots",
            "security_identity_snapshots",
            "nasdaq_listing_snapshots",
            "nasdaq_listing_snapshot_rows",
            "universe_scans",
            "canonical_universe_scans",
            "universe_snapshots",
            "evidence_jobs",
            "evidence_job_runs",
            "evidence_datasets",
            "evidence_snapshots",
            "data_lineage",
            "feature_snapshots",
            "signal_registry",
            "forward_return_observations",
            "forward_return_path_rows",
            "forward_context_path_rows",
            "forward_return_observation_events",
            "m1_earnings_events",
            "m1_friction_snapshots",
            "trade_candidates",
            "optimizer_runs",
            "order_events",
            "stbm_lifecycle_events",
            "shadow_positions",
            "real_positions",
            "validation_runs",
            "agent_export_manifests",
            "pattern_weights",
            "manual_overrides",
        }
        actual = set(Base.metadata.tables.keys())
        assert expected == actual

    def test_alembic_upgrade_creates_all_18_tables(self, tmp_path, monkeypatch):
        db_path = tmp_path / "migration-smoke.db"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
        monkeypatch.setenv("ALPHA_ALLOW_SQLITE_ALEMBIC", "1")
        monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)

        cfg = Config(str(Path("alembic.ini")))
        command.upgrade(cfg, "head")

        engine = create_engine(f"sqlite:///{db_path}")
        try:
            tables = set(inspect(engine).get_table_names())
            columns = {
                table: {col["name"] for col in inspect(engine).get_columns(table)}
                for table in (
                    "data_lineage",
                    "security_profile_scan_snapshots",
                    "signal_registry",
                    "forward_context_path_rows",
                    "trade_candidates",
                    "universe_snapshots",
                    "shadow_positions",
                    "real_positions",
                )
            }
            signal_columns = {
                col["name"]: col for col in inspect(engine).get_columns("signal_registry")
            }
        finally:
            engine.dispose()

        assert set(Base.metadata.tables.keys()) <= tables
        assert {
            "trading_date",
            "scan_id",
            "detector_version",
            "point_in_time_passed",
            "lookahead_guard_passed",
            "next_execution_session",
        } <= columns["signal_registry"]
        assert signal_columns["signal_identity_hash"]["nullable"] is False
        assert "raw_payload_json" in columns["data_lineage"]
        assert {
            "signal_id",
            "forward_session_date",
            "path_sequence",
            "asof_timestamp",
            "context_json",
            "source_attempts_json",
        } <= columns["forward_context_path_rows"]
        assert "country" in columns["universe_snapshots"]
        assert {
            "scan_id",
            "symbol",
            "profile_required",
            "cache_status",
            "raw_profile_json",
        } <= columns["security_profile_scan_snapshots"]
        assert {
            "effective_hard_stop_pct",
            "base_risk_budget_pct",
            "risk_budget_pct",
            "risk_multiplier_product",
            "risk_sized_cap",
            "unstopped_heat_pct",
            "cost_to_edge_ratio",
        } <= columns["trade_candidates"]
        assert {
            "forward_return",
            "fill_status",
            "intended_entry_price",
            "realized_entry_price",
            "execution_capture_gap",
        } <= columns["shadow_positions"]
        assert {
            "fill_status",
            "intended_entry_price",
            "realized_entry_price",
            "execution_capture_gap",
        } <= columns["real_positions"]
