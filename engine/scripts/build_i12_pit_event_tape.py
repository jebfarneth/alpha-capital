#!/usr/bin/env python3
"""Build a read-only PIT-clean I12 intraday event tape from snapshot schemas."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by direct invocation.
    if exc.name == "sqlalchemy":
        venv_python = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
        if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
            os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_i12_pit_snapshots import (  # noqa: E402
    _first_present,
    _load_report_json,
    _parse_date_required,
    _q,
    _validate_decision_time,
    _validate_final_report,
    _validate_scratch_schema,
)


EXIT_ROLES = ("same_day_exit", "next_open_exit")
QUOTE_ROLES = ("entry", "same_day_exit", "next_open_exit")
TAPE_SCOPES = ("passed", "all-source-attempts")
DEFAULT_MAX_JSON_ROWS = 100_000
PREDICTOR_ALLOWLIST = (
    "prior_close",
    "distance_from_max252",
    "drawdown_from_max252",
    "off_low252",
    "mom20",
    "sigma20",
    "prev_day_return",
    "prev_day_green",
    "gap",
    "early_return",
    "early_high_return",
    "early_low_return",
    "observed_open_to_decision_return",
    "observed_cumulative_volume_before_decision",
    "observed_minute_count_before_decision",
    "opening_bar_present",
    "path_coverage_ratio",
    "completed_minute_count",
    "zero_fill_imputed_minute_count",
    "zero_fill_imputed_minute_ratio",
)
GUARD_FIELDS = (
    "decision_ts",
    "source_minute_bars_max_start_ts",
    "completed_through_ts",
    "feature_asof_ts",
    "uses_forward_bars",
    "uses_full_day_volume",
    "uses_full_day_high_low",
    "uses_same_day_close",
)
LEAKY_GUARD_FLAGS = (
    "uses_forward_bars",
    "uses_full_day_volume",
    "uses_full_day_high_low",
    "uses_same_day_close",
)


class SnapshotSpec:
    __slots__ = ("label", "schema", "report")

    def __init__(self, label: str, schema: str, report: Path | None = None) -> None:
        self.label = label
        self.schema = schema
        self.report = report


def build_event_tape(
    *,
    snapshots: Sequence[SnapshotSpec],
    start_date: str | date,
    end_date: str | date,
    minute_path_mode: str,
    database_url: str | None = None,
    db_session: Session | None = None,
    require_final: bool = False,
    strict_predictor_guards: bool = False,
    scope: str = "passed",
    output_format: str = "json",
    max_json_rows: int = DEFAULT_MAX_JSON_ROWS,
    max_event_rows: int = DEFAULT_MAX_JSON_ROWS,
    allow_large_source_attempts: bool = False,
) -> dict[str, Any]:
    if not snapshots:
        raise ValueError("at least one --snapshot is required")
    start = _parse_date_required(start_date, "start_date")
    end = _parse_date_required(end_date, "end_date")
    if end < start:
        raise ValueError("end_date must be >= start_date")
    if minute_path_mode not in {"strict_contiguous", "sparse_zero_fill"}:
        raise ValueError("minute_path_mode must be strict_contiguous or sparse_zero_fill")
    if scope not in TAPE_SCOPES:
        raise ValueError(f"scope must be one of {', '.join(TAPE_SCOPES)}")
    if output_format not in {"text", "json", "jsonl", "csv"}:
        raise ValueError("output_format must be text, json, jsonl, or csv")

    normalized = [_normalize_snapshot(snapshot) for snapshot in snapshots]
    _validate_snapshot_set(normalized)
    _validate_final_reports_for_snapshots(
        normalized,
        require_final=require_final,
        start_date=start,
        end_date=end,
        minute_path_mode=minute_path_mode,
    )

    owns_session = False
    session = db_session
    if session is None:
        url = database_url or os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL is required unless db_session is provided")
        session = Session(bind=create_engine(url))
        owns_session = True
    assert session is not None
    try:
        rows: list[dict[str, Any]] = []
        warnings: list[str] = []
        integrity: dict[str, Any] = {}
        event_row_estimates: dict[str, int] = {}
        for snapshot in normalized:
            snapshot_integrity = _snapshot_integrity(
                session,
                snapshot=snapshot,
                start_date=start,
                end_date=end,
                minute_path_mode=minute_path_mode,
            )
            _fail_on_duplicate_snapshot_evidence(snapshot, snapshot_integrity)
            if snapshot_integrity["missing_quote_role_count"]:
                warnings.append(f"{snapshot.label}_missing_quote_evidence")
            if snapshot_integrity["missing_cost_role_count"]:
                warnings.append(f"{snapshot.label}_missing_cost_evidence")
            integrity[snapshot.label] = snapshot_integrity
            event_row_estimates[snapshot.label] = (
                int(snapshot_integrity["passed_count"])
                if scope == "passed"
                else int(snapshot_integrity["active_candidate_row_count"])
            )
        _preflight_event_row_guard(
            event_row_estimates,
            scope=scope,
            output_format=output_format,
            max_json_rows=max_json_rows,
            max_event_rows=max_event_rows,
            allow_large_source_attempts=allow_large_source_attempts,
        )
        for snapshot in normalized:
            rows.extend(
                _event_rows_for_snapshot(
                    session,
                    snapshot=snapshot,
                    start_date=start,
                    end_date=end,
                    minute_path_mode=minute_path_mode,
                    strict_predictor_guards=strict_predictor_guards,
                    scope=scope,
                )
            )

        _annotate_source_presence(
            session,
            rows,
            snapshots=normalized,
            start_date=start,
            end_date=end,
            minute_path_mode=minute_path_mode,
        )
        if not require_final:
            warnings.append("diagnostic_partial_tape_finality_not_required")
        if scope == "all-source-attempts":
            warnings.append("source_attempt_tape_not_ml_event_tape")
        warnings.append("volume_participation_labels_unavailable_displayed_size_only")
        _annotate_membership(rows, [snapshot.label for snapshot in normalized], scope=scope)
        summary = _summary(
            rows,
            snapshots=normalized,
            start_date=start,
            end_date=end,
            minute_path_mode=minute_path_mode,
            integrity=integrity,
            warnings=warnings,
            require_final=require_final,
            scope=scope,
            event_row_estimates=event_row_estimates,
        )
        return {
            "summary": summary,
            "events": rows,
        }
    finally:
        if owns_session:
            session.rollback()
            session.close()


def _normalize_snapshot(snapshot: SnapshotSpec) -> SnapshotSpec:
    label = _validate_decision_time(snapshot.label)
    schema = _validate_scratch_schema(snapshot.schema, side=f"{label} snapshot")
    return SnapshotSpec(label=label, schema=schema, report=snapshot.report)


def _validate_snapshot_set(snapshots: Sequence[SnapshotSpec]) -> None:
    labels = [snapshot.label for snapshot in snapshots]
    duplicate_labels = sorted(label for label, count in Counter(labels).items() if count > 1)
    if duplicate_labels:
        raise ValueError(f"duplicate snapshot decision-time labels are ambiguous: {', '.join(duplicate_labels)}")
    schemas = [snapshot.schema for snapshot in snapshots]
    duplicate_schemas = sorted(schema for schema, count in Counter(schemas).items() if count > 1)
    if duplicate_schemas:
        raise ValueError(f"duplicate snapshot schemas are ambiguous: {', '.join(duplicate_schemas)}")
    specs = [(snapshot.label, snapshot.schema) for snapshot in snapshots]
    duplicate_specs = sorted(
        f"{label}:{schema}" for (label, schema), count in Counter(specs).items() if count > 1
    )
    if duplicate_specs:
        raise ValueError(f"duplicate snapshot specs are ambiguous: {', '.join(duplicate_specs)}")


def _preflight_event_row_guard(
    event_row_estimates: Mapping[str, int],
    *,
    scope: str,
    output_format: str,
    max_json_rows: int,
    max_event_rows: int,
    allow_large_source_attempts: bool,
) -> None:
    total = sum(int(value or 0) for value in event_row_estimates.values())
    if output_format == "json" and max_json_rows >= 0 and total > max_json_rows:
        raise RuntimeError(
            f"preflight event row count {total} exceeds --max-json-rows={max_json_rows}; "
            "use --format jsonl or --format csv for larger tapes"
        )
    if (
        scope == "all-source-attempts"
        and max_event_rows >= 0
        and total > max_event_rows
        and not allow_large_source_attempts
    ):
        raise RuntimeError(
            f"preflight all-source-attempts row count {total} exceeds --max-event-rows={max_event_rows}; "
            "raise --max-event-rows for a deliberate bounded diagnostic or pass "
            "--allow-large-source-attempts after confirming memory is acceptable"
        )


def _validate_final_reports_for_snapshots(
    snapshots: Sequence[SnapshotSpec],
    *,
    require_final: bool,
    start_date: date,
    end_date: date,
    minute_path_mode: str,
) -> None:
    if not require_final:
        return
    source_hur_schema: str | None = None
    for snapshot in snapshots:
        if snapshot.report is None:
            raise RuntimeError("--require-final requires every --snapshot to include a report path")
        report = _load_report_json(snapshot.report, side=snapshot.label)
        _validate_final_report(
            report,
            side=snapshot.label,
            expected_schema=snapshot.schema,
            expected_start_date=start_date,
            expected_end_date=end_date,
            expected_decision_time=snapshot.label,
            expected_path_mode=minute_path_mode,
        )
        report_source = _first_present(report, "source_hur_schema")
        if source_hur_schema is None:
            source_hur_schema = str(report_source)
        elif source_hur_schema != str(report_source):
            raise RuntimeError(
                "--require-final source_hur_schema mismatch across snapshots: "
                f"{source_hur_schema!r} vs {report_source!r}"
            )


def _snapshot_integrity(
    session: Session,
    *,
    snapshot: SnapshotSpec,
    start_date: date,
    end_date: date,
    minute_path_mode: str,
) -> dict[str, Any]:
    candidates = _q(snapshot.schema, "i12_pit_candidates")
    quotes = _q(snapshot.schema, "i12_pit_quote_replays")
    costs = _q(snapshot.schema, "i12_pit_cost_replays")
    params = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "decision_time": snapshot.label,
        "path_mode": minute_path_mode,
    }
    summary = _one(
        session,
        f"""
        SELECT
          COUNT(*) AS active_candidate_row_count,
          SUM(CASE WHEN candidate_status = 'passed' THEN 1 ELSE 0 END) AS passed_count,
          COUNT(DISTINCT decision_date) AS date_count,
          MIN(decision_date) AS min_date,
          MAX(decision_date) AS max_date
        FROM {candidates}
        WHERE is_active IS TRUE
          AND decision_date BETWEEN :start_date AND :end_date
          AND decision_time_label = :decision_time
          AND path_mode = :path_mode
        """,
        params,
    )
    duplicate_candidates = _scalar(
        session,
        f"""
        SELECT COUNT(*) FROM (
          SELECT ticker, decision_date, path_mode, decision_time_label, COUNT(*) AS row_count
          FROM {candidates}
          WHERE is_active IS TRUE
            AND decision_date BETWEEN :start_date AND :end_date
            AND decision_time_label = :decision_time
            AND path_mode = :path_mode
          GROUP BY ticker, decision_date, path_mode, decision_time_label
          HAVING COUNT(*) > 1
        ) d
        """,
        params,
    )
    quote_counts = _child_evidence_counts(
        session,
        candidates=candidates,
        child_table=quotes,
        child_role_column="quote_role",
        child_id_column="i12_pit_quote_replay_id",
        roles=QUOTE_ROLES,
        params=params,
    )
    cost_counts = _child_evidence_counts(
        session,
        candidates=candidates,
        child_table=costs,
        child_role_column="exit_role",
        child_id_column="i12_pit_cost_replay_id",
        roles=EXIT_ROLES,
        params=params,
    )
    return {
        "schema": snapshot.schema,
        "label": snapshot.label,
        "active_candidate_row_count": int(summary.get("active_candidate_row_count") or 0),
        "passed_count": int(summary.get("passed_count") or 0),
        "date_count": int(summary.get("date_count") or 0),
        "min_date": _str_or_none(summary.get("min_date")),
        "max_date": _str_or_none(summary.get("max_date")),
        "duplicate_active_candidate_count": int(duplicate_candidates or 0),
        "missing_quote_role_count": quote_counts["missing"],
        "duplicate_quote_role_count": quote_counts["duplicate"],
        "missing_cost_role_count": cost_counts["missing"],
        "duplicate_cost_role_count": cost_counts["duplicate"],
    }


def _child_evidence_counts(
    session: Session,
    *,
    candidates: str,
    child_table: str,
    child_role_column: str,
    child_id_column: str,
    roles: Sequence[str],
    params: Mapping[str, Any],
) -> dict[str, int]:
    missing_total = 0
    duplicate_total = 0
    for role in roles:
        role_params = dict(params)
        role_params["role"] = role
        row = _one(
            session,
            f"""
            WITH scoped AS (
              SELECT i12_pit_candidate_id
              FROM {candidates}
              WHERE is_active IS TRUE
                AND candidate_status = 'passed'
                AND decision_date BETWEEN :start_date AND :end_date
                AND decision_time_label = :decision_time
                AND path_mode = :path_mode
            ), counts AS (
              SELECT scoped.i12_pit_candidate_id, COUNT(child.{child_id_column}) AS row_count
              FROM scoped
              LEFT JOIN {child_table} child
                ON child.i12_pit_candidate_id = scoped.i12_pit_candidate_id
               AND child.{child_role_column} = :role
               AND child.is_active IS TRUE
              GROUP BY scoped.i12_pit_candidate_id
            )
            SELECT
              SUM(CASE WHEN row_count = 0 THEN 1 ELSE 0 END) AS missing_count,
              SUM(CASE WHEN row_count > 1 THEN 1 ELSE 0 END) AS duplicate_count
            FROM counts
            """,
            role_params,
        )
        missing_total += int(row.get("missing_count") or 0)
        duplicate_total += int(row.get("duplicate_count") or 0)
    return {"missing": missing_total, "duplicate": duplicate_total}


def _fail_on_duplicate_snapshot_evidence(
    snapshot: SnapshotSpec,
    integrity: Mapping[str, Any],
) -> None:
    blockers: list[str] = []
    if int(integrity.get("duplicate_active_candidate_count") or 0):
        blockers.append(f"duplicate_active_candidate_count={integrity['duplicate_active_candidate_count']}")
    if int(integrity.get("duplicate_quote_role_count") or 0):
        blockers.append(f"duplicate_quote_role_count={integrity['duplicate_quote_role_count']}")
    if int(integrity.get("duplicate_cost_role_count") or 0):
        blockers.append(f"duplicate_cost_role_count={integrity['duplicate_cost_role_count']}")
    if blockers:
        raise RuntimeError(f"{snapshot.label}:{snapshot.schema} has duplicate active evidence: {', '.join(blockers)}")


def _event_rows_for_snapshot(
    session: Session,
    *,
    snapshot: SnapshotSpec,
    start_date: date,
    end_date: date,
    minute_path_mode: str,
    strict_predictor_guards: bool,
    scope: str,
) -> list[dict[str, Any]]:
    candidates = _q(snapshot.schema, "i12_pit_candidates")
    quotes = _q(snapshot.schema, "i12_pit_quote_replays")
    costs = _q(snapshot.schema, "i12_pit_cost_replays")
    params = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "decision_time": snapshot.label,
        "path_mode": minute_path_mode,
    }
    status_filter = "AND c.candidate_status = 'passed'" if scope == "passed" else ""
    raw_rows = _all(
        session,
        f"""
        SELECT
          c.i12_pit_candidate_id,
          c.ticker,
          c.decision_date,
          c.decision_ts,
          c.decision_time_label,
          c.path_mode,
          c.candidate_status,
          c.coverage_status,
          c.fail_reason,
          c.feature_json,
          c.leakage_guard_json,
          c.candidate_attempt_hash,
          c.content_hash,
          eq.coverage_status AS entry_quote_status,
          eq.quote_ts AS entry_quote_ts,
          eq.quote_age_seconds AS entry_quote_age_seconds,
          eq.bid AS entry_bid,
          eq.ask AS entry_ask,
          eq.spread_bps AS entry_spread_bps,
          eq.executable_notional AS entry_executable_notional,
          sdq.coverage_status AS same_day_exit_quote_status,
          noq.coverage_status AS next_open_exit_quote_status,
          sdc.modeled_return AS same_day_modeled_return_displayed_size,
          noc.modeled_return AS next_open_modeled_return_displayed_size,
          sdc.tradeability_status AS same_day_tradeability_status,
          noc.tradeability_status AS next_open_tradeability_status,
          sdc.skipped_reason AS same_day_skipped_reason,
          noc.skipped_reason AS next_open_skipped_reason
        FROM {candidates} c
        LEFT JOIN {quotes} eq
          ON eq.i12_pit_candidate_id = c.i12_pit_candidate_id
         AND c.candidate_status = 'passed'
         AND eq.quote_role = 'entry'
         AND eq.is_active IS TRUE
        LEFT JOIN {quotes} sdq
          ON sdq.i12_pit_candidate_id = c.i12_pit_candidate_id
         AND c.candidate_status = 'passed'
         AND sdq.quote_role = 'same_day_exit'
         AND sdq.is_active IS TRUE
        LEFT JOIN {quotes} noq
          ON noq.i12_pit_candidate_id = c.i12_pit_candidate_id
         AND c.candidate_status = 'passed'
         AND noq.quote_role = 'next_open_exit'
         AND noq.is_active IS TRUE
        LEFT JOIN {costs} sdc
          ON sdc.i12_pit_candidate_id = c.i12_pit_candidate_id
         AND c.candidate_status = 'passed'
         AND sdc.exit_role = 'same_day_exit'
         AND sdc.is_active IS TRUE
        LEFT JOIN {costs} noc
          ON noc.i12_pit_candidate_id = c.i12_pit_candidate_id
         AND c.candidate_status = 'passed'
         AND noc.exit_role = 'next_open_exit'
         AND noc.is_active IS TRUE
        WHERE c.is_active IS TRUE
          AND c.decision_date BETWEEN :start_date AND :end_date
          AND c.decision_time_label = :decision_time
          AND c.path_mode = :path_mode
          {status_filter}
        ORDER BY c.decision_date, c.ticker
        """,
        params,
    )
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        feature_json = _json_loads(raw.get("feature_json"))
        leakage_guard_json = _json_loads(raw.get("leakage_guard_json"))
        predictor_status, predictor_reason, predictors = _extract_predictors(
            raw,
            feature_json,
            leakage_guard_json,
        )
        if strict_predictor_guards and predictor_status != "ok":
            raise RuntimeError(
                f"predictor leakage guard failed for {snapshot.label}:{snapshot.schema} "
                f"{raw.get('ticker')} {raw.get('decision_date')}: {predictor_reason}"
            )
        entry_mid = _mid(raw.get("entry_bid"), raw.get("entry_ask"))
        rows.append(
            {
                "ticker": raw.get("ticker"),
                "decision_date": _str_or_none(raw.get("decision_date")),
                "decision_time_label": raw.get("decision_time_label"),
                "path_mode": raw.get("path_mode"),
                "snapshot_schema": snapshot.schema,
                "snapshot_label": snapshot.label,
                "tape_scope": scope,
                "i12_pit_candidate_id": raw.get("i12_pit_candidate_id"),
                "candidate_attempt_hash": raw.get("candidate_attempt_hash"),
                "content_hash": raw.get("content_hash"),
                "candidate_status": raw.get("candidate_status"),
                "coverage_status": raw.get("coverage_status"),
                "fail_reason": raw.get("fail_reason"),
                "is_passed": raw.get("candidate_status") == "passed",
                "predictor_status": predictor_status,
                "predictor_block_reason": predictor_reason,
                "predictors": predictors,
                "leakage_guard": {field: leakage_guard_json.get(field) for field in GUARD_FIELDS},
                "entry_quote_status": raw.get("entry_quote_status"),
                "entry_quote_ts": _str_or_none_full(raw.get("entry_quote_ts")),
                "entry_quote_age_seconds": _float_or_none(raw.get("entry_quote_age_seconds")),
                "entry_bid": _float_or_none(raw.get("entry_bid")),
                "entry_ask": _float_or_none(raw.get("entry_ask")),
                "entry_mid": entry_mid,
                "entry_spread_bps": _float_or_none(raw.get("entry_spread_bps")),
                "entry_executable_notional": _float_or_none(raw.get("entry_executable_notional")),
                "same_day_exit_quote_status": raw.get("same_day_exit_quote_status"),
                "next_open_exit_quote_status": raw.get("next_open_exit_quote_status"),
                "same_day_modeled_return_displayed_size": _float_or_none(
                    raw.get("same_day_modeled_return_displayed_size")
                ),
                "next_open_modeled_return_displayed_size": _float_or_none(
                    raw.get("next_open_modeled_return_displayed_size")
                ),
                "same_day_tradeability_status": raw.get("same_day_tradeability_status"),
                "next_open_tradeability_status": raw.get("next_open_tradeability_status"),
                "same_day_skipped_reason": raw.get("same_day_skipped_reason"),
                "next_open_skipped_reason": raw.get("next_open_skipped_reason"),
                "same_day_modeled_return_volume_participation": None,
                "next_open_modeled_return_volume_participation": None,
                "volume_tradeability_status": "unavailable",
                "volume_skip_reason": "volume_participation_not_materialized",
            }
        )
    return rows


def _annotate_source_presence(
    session: Session,
    rows: list[dict[str, Any]],
    *,
    snapshots: Sequence[SnapshotSpec],
    start_date: date,
    end_date: date,
    minute_path_mode: str,
) -> None:
    if not rows:
        return
    keys = sorted(
        {
            (str(row["ticker"]), str(row["decision_date"]), str(row["path_mode"]))
            for row in rows
        }
    )
    presence: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    dialect = getattr(getattr(session.get_bind(), "dialect", None), "name", "")
    date_join = "c.decision_date = CAST(k.decision_date AS date)" if dialect == "postgresql" else "c.decision_date = k.decision_date"
    for snapshot in snapshots:
        candidates = _q(snapshot.schema, "i12_pit_candidates")
        for chunk in _chunks(keys, 400):
            params: dict[str, Any] = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "decision_time": snapshot.label,
                "path_mode_filter": minute_path_mode,
            }
            values_sql: list[str] = []
            for index, (ticker, decision_date, path_mode) in enumerate(chunk):
                params[f"ticker_{index}"] = ticker
                params[f"date_{index}"] = decision_date
                params[f"path_{index}"] = path_mode
                values_sql.append(f"(:ticker_{index}, :date_{index}, :path_{index})")
            source_rows = _all(
                session,
                f"""
                WITH keys(ticker, decision_date, path_mode) AS (
                  VALUES {", ".join(values_sql)}
                )
                SELECT
                  c.ticker,
                  c.decision_date,
                  c.path_mode,
                  c.decision_time_label,
                  c.candidate_status,
                  c.coverage_status,
                  c.fail_reason
                FROM {candidates} c
                JOIN keys k
                  ON k.ticker = c.ticker
                 AND {date_join}
                 AND k.path_mode = c.path_mode
                WHERE c.is_active IS TRUE
                  AND c.decision_date BETWEEN :start_date AND :end_date
                  AND c.decision_time_label = :decision_time
                  AND c.path_mode = :path_mode_filter
                """,
                params,
            )
            for source in source_rows:
                key = (
                    str(source.get("ticker")),
                    _str_or_none(source.get("decision_date")) or "",
                    str(source.get("path_mode")),
                )
                presence[key][snapshot.label] = {
                    "candidate_status": source.get("candidate_status"),
                    "coverage_status": source.get("coverage_status"),
                    "fail_reason": source.get("fail_reason"),
                }
    for row in rows:
        key = (str(row["ticker"]), str(row["decision_date"]), str(row["path_mode"]))
        by_time = presence.get(key) or {}
        row["source_presence_by_decision_time"] = {
            label: label in by_time for label in (snapshot.label for snapshot in snapshots)
        }
        row["source_candidate_status_by_decision_time"] = {
            label: by_time[label].get("candidate_status") for label in by_time
        }
        row["source_coverage_status_by_decision_time"] = {
            label: by_time[label].get("coverage_status") for label in by_time
        }
        row["source_fail_reason_by_decision_time"] = {
            label: by_time[label].get("fail_reason") for label in by_time
        }


def _extract_predictors(
    raw: Mapping[str, Any],
    features: Mapping[str, Any],
    guard: Mapping[str, Any],
) -> tuple[str, str | None, dict[str, Any]]:
    predictors = {field: features.get(field) for field in PREDICTOR_ALLOWLIST}
    reason = _leakage_guard_block_reason(raw, guard)
    if reason is not None:
        return "blocked_leakage_guard", reason, {field: None for field in PREDICTOR_ALLOWLIST}
    return "ok", None, predictors


def _leakage_guard_block_reason(raw: Mapping[str, Any], guard: Mapping[str, Any]) -> str | None:
    if not guard:
        return "missing_leakage_guard_json"
    for field in LEAKY_GUARD_FLAGS:
        if _truthy(guard.get(field)):
            return field
    decision_ts = _parse_timestamp(guard.get("decision_ts") or raw.get("decision_ts"))
    source_max_ts = _parse_timestamp(guard.get("source_minute_bars_max_start_ts"))
    completed_through_ts = _parse_timestamp(guard.get("completed_through_ts"))
    feature_asof_ts = _parse_timestamp(guard.get("feature_asof_ts"))
    if decision_ts is None:
        return "missing_or_malformed_decision_ts"
    if source_max_ts is None:
        return "missing_or_malformed_source_minute_bars_max_start_ts"
    if completed_through_ts is None:
        return "missing_or_malformed_completed_through_ts"
    if feature_asof_ts is None:
        return "missing_or_malformed_feature_asof_ts"
    if source_max_ts >= decision_ts:
        return "source_minute_bars_max_start_ts_at_or_after_decision_ts"
    if completed_through_ts > decision_ts:
        return "completed_through_ts_after_decision_ts"
    if feature_asof_ts > decision_ts:
        return "feature_asof_ts_after_decision_ts"
    return None


def _annotate_membership(rows: list[dict[str, Any]], snapshot_labels: Sequence[str], *, scope: str) -> None:
    order = {label: index for index, label in enumerate(snapshot_labels)}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["ticker"]), str(row["decision_date"]), str(row["path_mode"]))].append(row)
    for group_rows in grouped.values():
        source_presence: dict[str, bool] = {}
        source_status: dict[str, Any] = {}
        for row in group_rows:
            for label, is_present in _mapping(row.get("source_presence_by_decision_time")).items():
                source_presence[str(label)] = bool(is_present)
            for label, status in _mapping(row.get("source_candidate_status_by_decision_time")).items():
                source_status[str(label)] = status
        if not source_presence:
            source_presence = {str(row["decision_time_label"]): True for row in group_rows}
            source_status = {
                str(row["decision_time_label"]): row.get("candidate_status")
                for row in group_rows
            }
        present = sorted(
            {label for label, is_present in source_presence.items() if is_present},
            key=lambda item: order.get(item, 999),
        )
        passed = sorted(
            {label for label, status in source_status.items() if status == "passed"},
            key=lambda item: order.get(item, 999),
        )
        first_source_seen = present[0] if present else None
        last_source_seen = present[-1] if present else None
        first_fire = passed[0] if passed else None
        last_fire = passed[-1] if passed else None
        bucket = _bucket_for_membership(snapshot_labels, present, passed)
        for row in group_rows:
            label = str(row["decision_time_label"])
            later_present = any(order.get(item, 999) > order.get(label, 999) for item in present)
            later_passed = any(order.get(item, 999) > order.get(label, 999) for item in passed)
            later_snapshot_exists = any(order.get(item, 999) > order.get(label, 999) for item in snapshot_labels)
            first_seen = first_fire
            last_seen = last_fire
            dropped_basis = later_snapshot_exists if scope == "passed" else later_present
            row.update(
                {
                    "first_seen_decision_time": first_seen,
                    "last_seen_decision_time": last_seen,
                    "first_source_seen_decision_time": first_source_seen,
                    "last_source_seen_decision_time": last_source_seen,
                    "first_fire_decision_time": first_fire,
                    "last_fire_decision_time": last_fire,
                    "present_decision_times": present,
                    "passed_decision_times": passed,
                    "bucket": bucket,
                    "bucket_tags": [bucket],
                    "present_count": len(present),
                    "passed_count": len(passed),
                    "first_passed_decision_time": first_fire,
                    "last_passed_decision_time": last_fire,
                    "survived_to_later_snapshot": bool(row.get("is_passed") and later_passed),
                    "dropped_by_later_snapshot": bool(row.get("is_passed") and dropped_basis and not later_passed),
                }
            )


def _bucket_for_membership(
    snapshot_labels: Sequence[str],
    present: Sequence[str],
    passed: Sequence[str],
) -> str:
    if len(snapshot_labels) == 2:
        left, right = snapshot_labels
        left_tag = left.replace(":", "")
        right_tag = right.replace(":", "")
        passed_set = set(passed)
        present_set = set(present)
        if passed_set == {left, right}:
            return f"shared_{left_tag}_{right_tag}"
        if passed_set == {left}:
            return f"only_{left_tag}"
        if passed_set == {right}:
            return f"only_{right_tag}"
        if present_set == {left, right}:
            return "failed_both"
        return "nonpassed_seen"
    if len(passed) > 1:
        return "passed_multiple_snapshots"
    if len(passed) == 1:
        return f"only_{passed[0].replace(':', '')}"
    return "nonpassed_seen"


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    snapshots: Sequence[SnapshotSpec],
    start_date: date,
    end_date: date,
    minute_path_mode: str,
    integrity: Mapping[str, Any],
    warnings: Sequence[str],
    require_final: bool,
    scope: str,
    event_row_estimates: Mapping[str, int],
) -> dict[str, Any]:
    ticker_dates = {(row.get("ticker"), row.get("decision_date")) for row in rows}
    passed_by_time = Counter(
        str(row["decision_time_label"]) for row in rows if row.get("is_passed")
    )
    bucket_by_key: dict[tuple[Any, Any, Any], str] = {}
    for row in rows:
        bucket_by_key[(row.get("ticker"), row.get("decision_date"), row.get("path_mode"))] = str(row.get("bucket"))
    bucket_counts = Counter(bucket_by_key.values())
    predictor_blocked = sum(1 for row in rows if row.get("predictor_status") != "ok")
    predictor_missing_counts = {
        field: sum(1 for row in rows if _mapping(row.get("predictors")).get(field) is None)
        for field in PREDICTOR_ALLOWLIST
    }
    summary = {
        "snapshots": [
            {"label": snapshot.label, "schema": snapshot.schema, "report": str(snapshot.report) if snapshot.report else None}
            for snapshot in snapshots
        ],
        "scope": scope,
        "tape_kind": "fired_event_tape" if scope == "passed" else "source_attempt_tape",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "minute_path_mode": minute_path_mode,
        "require_final": require_final,
        "row_count": len(rows),
        "preflight_event_row_estimates": dict(event_row_estimates),
        "unique_ticker_date_count": len(ticker_dates),
        "passed_count_by_decision_time": dict(passed_by_time),
        "bucket_counts": dict(bucket_counts),
        "missing_quote_role_count": sum(int(item.get("missing_quote_role_count") or 0) for item in integrity.values()),
        "missing_cost_role_count": sum(int(item.get("missing_cost_role_count") or 0) for item in integrity.values()),
        "predictor_blocked_count": predictor_blocked,
        "predictor_missing_counts": predictor_missing_counts,
        "integrity": integrity,
        "warnings": list(dict.fromkeys(warnings)),
        "volume_participation_labels": "unavailable_displayed_size_label_only_not_training_ready",
        "training_tape_status": "blocked_volume_participation_labels_not_materialized",
    }
    if len(snapshots) == 2:
        left, right = snapshots
        shared_key = f"shared_{left.label.replace(':', '')}_{right.label.replace(':', '')}"
        left_key = f"only_{left.label.replace(':', '')}"
        right_key = f"only_{right.label.replace(':', '')}"
        left_total = passed_by_time.get(left.label, 0)
        right_total = passed_by_time.get(right.label, 0)
        shared = bucket_counts.get(shared_key, 0)
        summary["two_snapshot_summary"] = {
            "shared_passed_count": shared,
            f"{left.label}_only_passed_count": bucket_counts.get(left_key, 0),
            f"{right.label}_only_passed_count": bucket_counts.get(right_key, 0),
            f"{left.label}_passed_total": left_total,
            f"{right.label}_passed_total": right_total,
            "overlap_rate_left_to_right": shared / left_total if left_total else None,
            "overlap_rate_right_to_left": shared / right_total if right_total else None,
            "same_day_return_by_bucket_decision_time": _metric_summary_by_bucket_decision_time(
                rows, "same_day_modeled_return_displayed_size"
            ),
            "next_open_return_by_bucket_decision_time": _metric_summary_by_bucket_decision_time(
                rows, "next_open_modeled_return_displayed_size"
            ),
            "entry_spread_bps_by_bucket_decision_time": _metric_summary_by_bucket_decision_time(
                rows, "entry_spread_bps"
            ),
            "entry_executable_notional_by_bucket_decision_time": _metric_summary_by_bucket_decision_time(
                rows, "entry_executable_notional"
            ),
            "shared_timing_deltas_right_minus_left": _shared_timing_deltas(
                rows,
                left_label=left.label,
                right_label=right.label,
                shared_bucket=shared_key,
            ),
        }
    return summary


def _metric_summary_by_bucket_decision_time(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not row.get("is_passed"):
            continue
        value = _float_or_none(row.get(field))
        if value is not None:
            values[str(row.get("bucket"))][str(row.get("decision_time_label"))].append(value)
    return {
        bucket: {
            decision_time: _numeric_summary(numbers)
            for decision_time, numbers in sorted(decision_values.items())
        }
        for bucket, decision_values in sorted(values.items())
    }


def _shared_timing_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    left_label: str,
    right_label: str,
    shared_bucket: str,
) -> dict[str, Any]:
    fields = (
        "same_day_modeled_return_displayed_size",
        "next_open_modeled_return_displayed_size",
        "entry_spread_bps",
        "entry_executable_notional",
        "entry_quote_age_seconds",
    )
    paired: dict[tuple[Any, Any, Any], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("bucket") != shared_bucket or not row.get("is_passed"):
            continue
        key = (row.get("ticker"), row.get("decision_date"), row.get("path_mode"))
        paired[key][str(row.get("decision_time_label"))] = row
    deltas: dict[str, list[float]] = {field: [] for field in fields}
    pair_count = 0
    for by_time in paired.values():
        left = by_time.get(left_label)
        right = by_time.get(right_label)
        if left is None or right is None:
            continue
        pair_count += 1
        for field in fields:
            left_value = _float_or_none(left.get(field))
            right_value = _float_or_none(right.get(field))
            if left_value is not None and right_value is not None:
                deltas[field].append(right_value - left_value)
    return {
        "pair_count": pair_count,
        **{field: _numeric_summary(values) for field, values in deltas.items()},
    }


def render_text(tape: Mapping[str, Any]) -> str:
    summary = _mapping(tape.get("summary"))
    lines = [
        "I12 PIT Event Tape",
        f"range={summary.get('start_date')}..{summary.get('end_date')} path_mode={summary.get('minute_path_mode')}",
        f"scope={summary.get('scope')} tape_kind={summary.get('tape_kind')}",
        f"rows={summary.get('row_count')} unique_ticker_dates={summary.get('unique_ticker_date_count')}",
        "",
        "Passed By Decision Time",
    ]
    lines.append(json.dumps(summary.get("passed_count_by_decision_time") or {}, sort_keys=True))
    lines.append("")
    lines.append("Buckets")
    lines.append(json.dumps(summary.get("bucket_counts") or {}, sort_keys=True))
    two = summary.get("two_snapshot_summary")
    if two:
        lines.append("")
        lines.append("Two Snapshot Summary")
        lines.append(json.dumps(two, sort_keys=True, default=str))
    warnings = list(summary.get("warnings") or [])
    if warnings:
        lines.append("")
        lines.append("Warnings")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def render_jsonl(tape: Mapping[str, Any]) -> str:
    lines = [json.dumps({"type": "summary", "summary": tape.get("summary")}, sort_keys=True, default=str)]
    for row in tape.get("events") or []:
        lines.append(json.dumps({"type": "event", "event": row}, sort_keys=True, default=str))
    return "\n".join(lines)


def render_json(tape: Mapping[str, Any], *, max_rows: int = DEFAULT_MAX_JSON_ROWS) -> str:
    row_count = len(tape.get("events") or [])
    if max_rows >= 0 and row_count > max_rows:
        raise RuntimeError(
            f"JSON output would include {row_count} event rows, above --max-json-rows={max_rows}; "
            "use --format jsonl or --format csv for full-corpus output"
        )
    return json.dumps(tape, indent=2, sort_keys=True, default=str)


def write_csv(tape: Mapping[str, Any], output: Path | None) -> str | None:
    rows = [_flatten(row) for row in tape.get("events") or []]
    fieldnames = sorted({key for row in rows for key in row})
    if output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return None


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in value.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            out.update(_flatten(item, flat_key))
        elif isinstance(item, (list, tuple)):
            out[flat_key] = json.dumps(item, sort_keys=True, default=str)
        else:
            out[flat_key] = item
    return out


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def parse_snapshot(value: str) -> SnapshotSpec:
    match = re.fullmatch(r"([0-2][0-9]:[0-5][0-9]):([^:]+)(?::(.+))?", value)
    if not match:
        raise argparse.ArgumentTypeError("snapshot must be LABEL:SCHEMA or LABEL:SCHEMA:REPORT")
    label, schema, report_text = match.groups()
    report = Path(report_text) if report_text else None
    return SnapshotSpec(label=label, schema=schema, report=report)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="append", type=parse_snapshot, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--minute-path-mode",
        required=True,
        choices=["strict_contiguous", "sparse_zero_fill"],
    )
    parser.add_argument("--format", choices=["text", "json", "jsonl", "csv"], default="json")
    parser.add_argument("--scope", choices=TAPE_SCOPES, default="passed")
    parser.add_argument("--max-json-rows", type=int, default=DEFAULT_MAX_JSON_ROWS)
    parser.add_argument("--max-event-rows", type=int, default=DEFAULT_MAX_JSON_ROWS)
    parser.add_argument("--allow-large-source-attempts", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--require-final", action="store_true")
    parser.add_argument("--strict-predictor-guards", action="store_true")
    parser.add_argument("--database-url")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        tape = build_event_tape(
            snapshots=args.snapshot,
            start_date=args.start_date,
            end_date=args.end_date,
            minute_path_mode=args.minute_path_mode,
            database_url=args.database_url,
            require_final=args.require_final,
            strict_predictor_guards=args.strict_predictor_guards,
            scope=args.scope,
            output_format=args.format,
            max_json_rows=args.max_json_rows,
            max_event_rows=args.max_event_rows,
            allow_large_source_attempts=args.allow_large_source_attempts,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    output_path = Path(args.output) if args.output else None
    try:
        if args.format == "csv":
            write_csv(tape, output_path)
        else:
            if args.format == "json":
                payload = render_json(tape, max_rows=args.max_json_rows)
            elif args.format == "jsonl":
                payload = render_jsonl(tape)
            else:
                payload = render_text(tape)
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(payload + "\n")
            else:
                print(payload)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


def _one(session: Session, sql: str, params: Mapping[str, Any]) -> dict[str, Any]:
    rows = _all(session, sql, params)
    return rows[0] if rows else {}


def _all(session: Session, sql: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = session.execute(text(sql), dict(params))
    return [dict(row._mapping) for row in result]


def _scalar(session: Session, sql: str, params: Mapping[str, Any]) -> Any:
    return session.execute(text(sql), dict(params)).scalar()


def _json_loads(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _utc_naive(value)
    try:
        return _utc_naive(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except ValueError:
        return None


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone().replace(tzinfo=None)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _mid(bid: Any, ask: Any) -> float | None:
    bid_float = _float_or_none(bid)
    ask_float = _float_or_none(ask)
    if bid_float is None or ask_float is None:
        return None
    return (bid_float + ask_float) / 2.0


def _numeric_summary(values: Iterable[Any]) -> dict[str, Any]:
    numbers = sorted(value for value in (_float_or_none(item) for item in values) if value is not None)
    if not numbers:
        return {"count": 0, "mean": None, "median": None}
    midpoint = len(numbers) // 2
    median = (
        numbers[midpoint]
        if len(numbers) % 2
        else (numbers[midpoint - 1] + numbers[midpoint]) / 2.0
    )
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers),
        "median": median,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _str_or_none(value: Any) -> str | None:
    return None if value is None else str(value)[:10]


def _str_or_none_full(value: Any) -> str | None:
    return None if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
