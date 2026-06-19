#!/usr/bin/env python3
"""Analyze PIT-clean I12 report artifacts and optional scratch replay rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session


EXIT_ROLES = ("same_day_exit", "next_open_exit")
SUMMARY_KEYS = ("p50", "p75", "p90")
DEFAULT_TRADEABLE_RATE_THRESHOLD = 0.75
MIN_PASSED_CANDIDATES_FOR_RANKING = 100
CONCENTRATION_WARNING_THRESHOLD = 0.50
MATERIAL_EXIT_GAP = 0.005
FORBIDDEN_SCHEMAS = {
    "",
    "canonical",
    "default",
    "information_schema",
    "main",
    "pg_catalog",
    "public",
}


class LoadedReport:
    def __init__(
        self,
        *,
        label: str,
        path: Path,
        report: dict[str, Any] | None,
        error: str | None = None,
    ) -> None:
        self.label = label
        self.path = path
        self.report = report
        self.error = error


def analyze_report_paths(
    paths: Sequence[str | Path],
    *,
    labels: Sequence[str] | None = None,
    schema: str | None = None,
    database_url: str | None = None,
    db_session: Session | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    allow_unbounded_schema: bool = False,
    require_final: bool = False,
    tradeable_rate_threshold: float = DEFAULT_TRADEABLE_RATE_THRESHOLD,
) -> dict[str, Any]:
    loaded = _load_reports(paths, labels=labels)
    _validate_labels(paths, labels)
    db_context = None
    if schema is not None:
        _validate_scratch_schema(schema)
        db_context = _build_db_context(
            schema,
            database_url,
            db_session,
            start_date=start_date,
            end_date=end_date,
            allow_unbounded_schema=allow_unbounded_schema,
        )
    try:
        reports = [
            _analyze_loaded_report(
                item,
                db_context=db_context,
                tradeable_rate_threshold=tradeable_rate_threshold,
            )
            for item in loaded
        ]
        require_final_failures = [
            report["label"]
            for report in reports
            if require_final
            and (
                report["integrity"].get("conclusions_final") is not True
                or report["integrity"].get("data_integrity_passed") is not True
            )
        ]
        return {
            "report_count": len(reports),
            "require_final": require_final,
            "require_final_failures": require_final_failures,
            "schema": schema,
            "reports": reports,
        }
    finally:
        if db_context is not None and db_context.get("owns_session"):
            _close_owned_db_session(db_context["session"])


def render_text_analysis(analysis: Mapping[str, Any]) -> str:
    lines: list[str] = []
    reports = [
        report for report in analysis.get("reports", []) if isinstance(report, Mapping)
    ]
    lines.append("Report Integrity")
    lines.append(_format_table(
        [
            "label",
            "schema",
            "range",
            "db_range",
            "mode",
            "decision",
            "final",
            "integrity",
            "status",
            "source",
            "quotes",
            "costs",
            "src_miss",
            "src_extra",
            "q_miss",
            "q_dup",
            "c_miss",
            "c_dup",
            "q_ok",
            "q_non_ok",
            "q_ok_rate",
        ],
        [
            [
                report.get("label"),
                _nested(report, "integrity", "schema"),
                _date_range(report.get("integrity")),
                _date_range(report.get("db_analysis_scope")),
                _nested(report, "integrity", "report_path_mode"),
                ",".join(_nested(report, "integrity", "report_decision_time_labels") or []),
                _fmt_bool(_nested(report, "integrity", "conclusions_final")),
                _fmt_bool(_nested(report, "integrity", "data_integrity_passed")),
                _nested(report, "integrity", "training_status"),
                _fmt_bool(_nested(report, "integrity", "source_replay_complete")),
                _fmt_bool(_nested(report, "integrity", "quote_replay_complete")),
                _fmt_bool(_nested(report, "integrity", "cost_replay_complete")),
                _fmt(_nested(report, "integrity", "missing_source_attempt_count")),
                _fmt(_nested(report, "integrity", "extra_source_attempt_count")),
                _fmt(_nested(report, "integrity", "missing_quote_role_count")),
                _fmt(_nested(report, "integrity", "duplicate_quote_role_count")),
                _fmt(_nested(report, "integrity", "missing_cost_role_count")),
                _fmt(_nested(report, "integrity", "duplicate_cost_role_count")),
                _fmt(_nested(report, "integrity", "quote_ok_count")),
                _fmt(_nested(report, "integrity", "quote_non_ok_count")),
                _fmt_pct(_nested(report, "integrity", "quote_ok_rate")),
            ]
            for report in reports
        ],
    ))
    lines.append("")
    lines.append("Exit Comparison")
    exit_rows = []
    for report in reports:
        for role in EXIT_ROLES:
            metrics = _mapping(_nested(report, "exit_comparison", role))
            exit_rows.append([
                report.get("label"),
                role,
                _fmt(metrics.get("candidates")),
                _fmt(metrics.get("tradeable_count")),
                _fmt_pct(metrics.get("tradeable_rate")),
                _fmt(metrics.get("skipped_cash_count")),
                json.dumps(metrics.get("skipped_cash_by_reason") or {}, sort_keys=True),
                _fmt_pct(metrics.get("mean_modeled_return_skips_as_cash")),
                _fmt_pct(metrics.get("win_rate_skips_as_cash")),
                _fmt_pct(metrics.get("mean_tradeable_return")),
                _fmt(metrics.get("spread_bps_p50")),
                _fmt(metrics.get("spread_bps_p75")),
                _fmt(metrics.get("spread_bps_p90")),
                _fmt(metrics.get("executable_notional_p50")),
                _fmt(metrics.get("executable_notional_p75")),
                _fmt(metrics.get("executable_notional_p90")),
                _fmt_pct(metrics.get("top_of_book_sufficient_rate")),
            ])
    lines.append(_format_table(
        [
            "label",
            "exit",
            "cand",
            "trade",
            "trade_rate",
            "skipped",
            "skip_reasons",
            "mean_cash",
            "win_cash",
            "mean_trade",
            "spr_p50",
            "spr_p75",
            "spr_p90",
            "not_p50",
            "not_p75",
            "not_p90",
            "book_ok",
        ],
        exit_rows,
    ))
    for report in reports:
        monthly_rows = _nested(report, "monthly_stability", "rows") or []
        if monthly_rows:
            lines.append("")
            lines.append(f"Monthly Stability: {report.get('label')}")
            lines.append(_format_table(
                [
                    "month",
                    "source",
                    "passed",
                    "pass_rate",
                    "sd_trade",
                    "sd_skip",
                    "sd_mean",
                    "sd_win",
                    "no_mean",
                    "no_win",
                    "q_ok",
                    "q_non_ok",
                    "skip_reasons",
                ],
                [
                    [
                        row.get("month"),
                        _fmt(row.get("source_attempts")),
                        _fmt(row.get("passed_candidates")),
                        _fmt_pct(row.get("pass_rate")),
                        _fmt(_nested(row, "same_day_exit", "tradeable_count")),
                        _fmt(_nested(row, "same_day_exit", "skipped_cash_count")),
                        _fmt_pct(_nested(row, "same_day_exit", "mean_modeled_return_skips_as_cash")),
                        _fmt_pct(_nested(row, "same_day_exit", "win_rate_skips_as_cash")),
                        _fmt_pct(_nested(row, "next_open_exit", "mean_modeled_return_skips_as_cash")),
                        _fmt_pct(_nested(row, "next_open_exit", "win_rate_skips_as_cash")),
                        _fmt(row.get("quote_ok_count")),
                        _fmt(row.get("quote_non_ok_count")),
                        json.dumps(row.get("skip_reason_summary") or {}, sort_keys=True),
                    ]
                    for row in monthly_rows
                ],
            ))
        daily_summary = _nested(report, "daily_distribution", "summary") or {}
        if daily_summary:
            lines.append("")
            lines.append(f"Daily Distribution: {report.get('label')}")
            lines.append(json.dumps(daily_summary, indent=2, sort_keys=True))
        warnings = report.get("warnings") or []
        if warnings:
            lines.append("")
            lines.append(f"Warnings: {report.get('label')}")
            lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def _analyze_loaded_report(
    loaded: LoadedReport,
    *,
    db_context: Mapping[str, Any] | None,
    tradeable_rate_threshold: float,
) -> dict[str, Any]:
    if loaded.error or loaded.report is None:
        return {
            "label": loaded.label,
            "path": str(loaded.path),
            "error": loaded.error or "missing_report",
            "integrity": {
                "conclusions_final": False,
                "data_integrity_passed": False,
                "training_status": "report_load_error",
            },
            "exit_comparison": {},
            "monthly_stability": {"rows": []},
            "daily_distribution": {"rows": [], "summary": {}},
            "warnings": [loaded.error or "missing_report"],
        }
    report = loaded.report
    integrity = _integrity_summary(report, loaded)
    exit_comparison = _exit_comparison(report)
    db_analysis = (
        _db_backed_analysis(report, db_context)
        if db_context is not None else _empty_db_analysis("db_backed_analysis_not_requested")
    )
    warnings = _quality_warnings(
        integrity,
        exit_comparison,
        db_analysis,
        tradeable_rate_threshold=tradeable_rate_threshold,
    )
    return {
        "label": loaded.label,
        "path": str(loaded.path),
        "error": None,
        "integrity": integrity,
        "exit_comparison": exit_comparison,
        "monthly_stability": db_analysis["monthly_stability"],
        "daily_distribution": db_analysis["daily_distribution"],
        "db_analysis_scope": db_analysis.get("scope"),
        "db_backed": db_context is not None,
        "warnings": warnings,
    }


def _integrity_summary(report: Mapping[str, Any], loaded: LoadedReport) -> dict[str, Any]:
    return {
        "label": loaded.label,
        "path": str(loaded.path),
        "schema": report.get("schema") or report.get("source_schema"),
        "start_date": report.get("start_date") or report.get("progress_source_start_date"),
        "end_date": report.get("end_date") or report.get("progress_source_end_date"),
        "report_path_mode": report.get("report_path_mode"),
        "report_decision_time_labels": list(report.get("report_decision_time_labels") or []),
        "conclusions_final": report.get("conclusions_final"),
        "data_integrity_passed": report.get("data_integrity_passed"),
        "training_status": report.get("training_status"),
        "source_replay_complete": report.get("source_replay_complete"),
        "quote_replay_complete": report.get("quote_replay_complete"),
        "cost_replay_complete": report.get("cost_replay_complete"),
        "missing_source_attempt_count": report.get("missing_source_attempt_count"),
        "extra_source_attempt_count": report.get("extra_source_attempt_count"),
        "missing_source_attempt_identity_count": report.get("missing_source_attempt_identity_count"),
        "extra_source_attempt_identity_count": report.get("extra_source_attempt_identity_count"),
        "missing_quote_role_count": report.get("missing_quote_role_count"),
        "duplicate_quote_role_count": report.get("duplicate_quote_role_count"),
        "missing_cost_role_count": report.get("missing_cost_role_count"),
        "duplicate_cost_role_count": report.get("duplicate_cost_role_count"),
        "unknown_quote_coverage_status_count": report.get("unknown_quote_coverage_status_count", 0),
        "quote_ok_count": report.get("quote_ok_count"),
        "quote_non_ok_count": report.get("quote_non_ok_count"),
        "quote_ok_rate": report.get("quote_ok_rate"),
        "candidate_coverage_status_counts": (
            report.get("candidate_coverage_status_counts") or {}
        ),
        "quote_coverage_status_counts": report.get("quote_coverage_status_counts") or {},
    }


def _exit_comparison(report: Mapping[str, Any]) -> dict[str, Any]:
    exit_metrics = _mapping(report.get("exit_metrics"))
    return {role: _exit_summary(exit_metrics.get(role)) for role in EXIT_ROLES}


def _exit_summary(metrics_value: Any) -> dict[str, Any]:
    metrics = _mapping(metrics_value)
    spread = _mapping(metrics.get("spread_bps"))
    notional = _mapping(metrics.get("executable_notional"))
    out = {
        "candidates": metrics.get("candidates"),
        "tradeable_count": metrics.get("tradeable_count"),
        "tradeable_rate": metrics.get("tradeable_rate"),
        "skipped_cash_count": metrics.get("skipped_cash_count"),
        "skipped_cash_by_reason": metrics.get("skipped_cash_by_reason") or {},
        "mean_modeled_return_skips_as_cash": metrics.get("mean_modeled_return_skips_as_cash"),
        "win_rate_skips_as_cash": metrics.get("win_rate_skips_as_cash"),
        "mean_tradeable_return": _first_not_none(
            metrics.get("mean_slippage_return_tradeable"),
            metrics.get("mean_quote_cost_return_tradeable"),
        ),
        "top_of_book_sufficient_rate": metrics.get("top_of_book_sufficient_rate"),
    }
    for key in SUMMARY_KEYS:
        out[f"spread_bps_{key}"] = spread.get(key)
        out[f"executable_notional_{key}"] = notional.get(key)
    return out


def _db_backed_analysis(
    report: Mapping[str, Any],
    db_context: Mapping[str, Any],
) -> dict[str, Any]:
    session: Session = db_context["session"]
    schema: str = db_context["schema"]
    date_scope = _resolve_db_date_scope(report, db_context)
    return {
        "scope": {
            "schema": schema,
            "start_date": date_scope["start_date"],
            "end_date": date_scope["end_date"],
            "unbounded": date_scope["unbounded"],
        },
        "monthly_stability": {
            "rows": _monthly_rows_sql(session, schema, report, date_scope),
        },
        "daily_distribution": _daily_distribution_sql(
            session,
            schema,
            report,
            date_scope,
        ),
    }


def _empty_db_analysis(reason: str) -> dict[str, Any]:
    return {
        "scope": {
            "reason": reason,
            "start_date": None,
            "end_date": None,
            "unbounded": False,
        },
        "monthly_stability": {"rows": [], "reason": reason},
        "daily_distribution": {"rows": [], "summary": {}, "reason": reason},
    }


def _resolve_db_date_scope(
    report: Mapping[str, Any],
    db_context: Mapping[str, Any],
) -> dict[str, Any]:
    explicit_start = db_context.get("start_date")
    explicit_end = db_context.get("end_date")
    report_start = report.get("start_date") or report.get("progress_source_start_date")
    report_end = report.get("end_date") or report.get("progress_source_end_date")
    start = explicit_start or report_start
    end = explicit_end or report_end
    if bool(start) != bool(end):
        raise ValueError(
            "DB-backed analysis requires both start and end dates; provide "
            "--start-date/--end-date or use report date fields"
        )
    if not start and not end:
        if db_context.get("allow_unbounded_schema"):
            return {"start_date": None, "end_date": None, "unbounded": True}
        raise ValueError(
            "DB-backed analysis requires bounded dates from --start-date/--end-date "
            "or report start/end fields; pass --allow-unbounded-schema for a "
            "diagnostic full-schema analysis"
        )
    start_text = _validate_date_text(str(start), "--start-date")
    end_text = _validate_date_text(str(end), "--end-date")
    if date.fromisoformat(start_text) > date.fromisoformat(end_text):
        raise ValueError("--start-date must be <= --end-date")
    return {"start_date": start_text, "end_date": end_text, "unbounded": False}


def _candidate_scope_sql(
    session: Session,
    schema: str,
    report: Mapping[str, Any],
    date_scope: Mapping[str, Any],
    *,
    alias: str = "c",
) -> tuple[str, dict[str, Any]]:
    conditions = [f"{alias}.is_active = :candidate_active"]
    params: dict[str, Any] = {"candidate_active": True}
    path_mode = report.get("report_path_mode")
    if path_mode:
        conditions.append(f"{alias}.path_mode = :path_mode")
        params["path_mode"] = path_mode
    decision_labels = list(report.get("report_decision_time_labels") or [])
    _append_in_condition(
        conditions,
        params,
        f"{alias}.decision_time_label",
        decision_labels,
        "dt",
    )
    if date_scope.get("start_date"):
        conditions.append(f"{alias}.decision_date >= :start_date")
        params["start_date"] = date_scope["start_date"]
    if date_scope.get("end_date"):
        conditions.append(f"{alias}.decision_date <= :end_date")
        params["end_date"] = date_scope["end_date"]
    table = _qualified_table(session, schema, "i12_pit_candidates")
    return (
        f"(SELECT {alias}.* FROM {table} {alias} "
        f"WHERE {' AND '.join(conditions)}) {alias}"
    ), params


def _month_expr(session: Session, column: str) -> str:
    if session.get_bind().dialect.name == "sqlite":
        return f"substr(CAST({column} AS TEXT), 1, 7)"
    return f"to_char({column}, 'YYYY-MM')"


def _day_expr(session: Session, column: str) -> str:
    if session.get_bind().dialect.name == "sqlite":
        return f"substr(CAST({column} AS TEXT), 1, 10)"
    return f"to_char({column}, 'YYYY-MM-DD')"


def _sql_int(value: Any) -> int:
    return int(value or 0)


def _sql_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _monthly_rows_sql(
    session: Session,
    schema: str,
    report: Mapping[str, Any],
    date_scope: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidate_scope, params = _candidate_scope_sql(session, schema, report, date_scope)
    month_expr = _month_expr(session, "c.decision_date")
    rows_by_month: dict[str, dict[str, Any]] = {}
    candidate_query = text(
        f"SELECT {month_expr} AS bucket, "
        "COUNT(*) AS source_attempts, "
        "SUM(CASE WHEN c.candidate_status = 'passed' THEN 1 ELSE 0 END) AS passed_candidates "
        f"FROM {candidate_scope} GROUP BY bucket ORDER BY bucket"
    )
    for row in session.execute(candidate_query, params).mappings():
        month = str(row["bucket"])
        bucket = rows_by_month.setdefault(month, _empty_month_row(month))
        bucket["source_attempts"] = _sql_int(row["source_attempts"])
        bucket["passed_candidates"] = _sql_int(row["passed_candidates"])

    quote_table = _qualified_table(session, schema, "i12_pit_quote_replays")
    quote_query = text(
        f"SELECT {month_expr} AS bucket, "
        "SUM(CASE WHEN q.coverage_status = 'ok' THEN 1 ELSE 0 END) AS quote_ok_count, "
        "SUM(CASE WHEN q.coverage_status != 'ok' OR q.coverage_status IS NULL THEN 1 ELSE 0 END) AS quote_non_ok_count "
        f"FROM {candidate_scope} "
        f"JOIN {quote_table} q ON q.i12_pit_candidate_id = c.i12_pit_candidate_id "
        "WHERE q.is_active = :child_active GROUP BY bucket ORDER BY bucket"
    )
    quote_params = {**params, "child_active": True}
    for row in session.execute(quote_query, quote_params).mappings():
        month = str(row["bucket"])
        bucket = rows_by_month.setdefault(month, _empty_month_row(month))
        bucket["quote_ok_count"] = _sql_int(row["quote_ok_count"])
        bucket["quote_non_ok_count"] = _sql_int(row["quote_non_ok_count"])

    cost_table = _qualified_table(session, schema, "i12_pit_cost_replays")
    cost_query = text(
        f"SELECT {month_expr} AS bucket, k.exit_role, "
        "COUNT(*) AS candidates, "
        "SUM(CASE WHEN k.tradeability_status = 'tradeable' THEN 1 ELSE 0 END) AS tradeable_count, "
        "SUM(CASE WHEN k.tradeability_status = 'tradeable' THEN 0 ELSE 1 END) AS skipped_cash_count, "
        "SUM(k.modeled_return) AS return_sum, "
        "SUM(CASE WHEN k.modeled_return > 0 THEN 1 ELSE 0 END) AS positive_count, "
        "COUNT(k.modeled_return) AS return_count "
        f"FROM {candidate_scope} "
        f"JOIN {cost_table} k ON k.i12_pit_candidate_id = c.i12_pit_candidate_id "
        "WHERE k.is_active = :child_active GROUP BY bucket, k.exit_role ORDER BY bucket, k.exit_role"
    )
    for row in session.execute(cost_query, quote_params).mappings():
        month = str(row["bucket"])
        role = str(row["exit_role"])
        if role not in EXIT_ROLES:
            continue
        bucket = rows_by_month.setdefault(month, _empty_month_row(month))
        _set_cost_bucket_from_aggregate(bucket[role], row)

    skip_query = text(
        f"SELECT {month_expr} AS bucket, COALESCE(k.skipped_reason, 'unknown') AS skipped_reason, "
        "COUNT(*) AS count "
        f"FROM {candidate_scope} "
        f"JOIN {cost_table} k ON k.i12_pit_candidate_id = c.i12_pit_candidate_id "
        "WHERE k.is_active = :child_active GROUP BY bucket, skipped_reason"
    )
    for row in session.execute(skip_query, quote_params).mappings():
        month = str(row["bucket"])
        bucket = rows_by_month.setdefault(month, _empty_month_row(month))
        bucket["skip_reason_summary"][str(row["skipped_reason"])] += _sql_int(row["count"])

    out = []
    for month, row in sorted(rows_by_month.items()):
        row["pass_rate"] = _safe_rate(row["passed_candidates"], row["source_attempts"])
        row["skip_reason_summary"] = dict(row["skip_reason_summary"])
        for role in EXIT_ROLES:
            _finalize_aggregate_cost_bucket(row[role])
        out.append(row)
    return out


def _daily_distribution_sql(
    session: Session,
    schema: str,
    report: Mapping[str, Any],
    date_scope: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_scope, params = _candidate_scope_sql(session, schema, report, date_scope)
    day_expr = _day_expr(session, "c.decision_date")
    rows_by_day: dict[str, dict[str, Any]] = {}
    candidate_query = text(
        f"SELECT {day_expr} AS bucket, "
        "COUNT(*) AS source_attempts, "
        "SUM(CASE WHEN c.candidate_status = 'passed' THEN 1 ELSE 0 END) AS passed_candidates "
        f"FROM {candidate_scope} GROUP BY bucket ORDER BY bucket"
    )
    for row in session.execute(candidate_query, params).mappings():
        day = str(row["bucket"])
        bucket = rows_by_day.setdefault(day, _empty_daily_row(day))
        bucket["source_attempts"] = _sql_int(row["source_attempts"])
        bucket["passed_candidates"] = _sql_int(row["passed_candidates"])

    cost_table = _qualified_table(session, schema, "i12_pit_cost_replays")
    cost_query = text(
        f"SELECT {day_expr} AS bucket, k.exit_role, "
        "COUNT(*) AS candidates, "
        "SUM(CASE WHEN k.tradeability_status = 'tradeable' THEN 1 ELSE 0 END) AS tradeable_count, "
        "SUM(CASE WHEN k.tradeability_status = 'tradeable' THEN 0 ELSE 1 END) AS skipped_cash_count, "
        "SUM(k.modeled_return) AS return_sum, "
        "SUM(CASE WHEN k.modeled_return > 0 THEN 1 ELSE 0 END) AS positive_count, "
        "COUNT(k.modeled_return) AS return_count "
        f"FROM {candidate_scope} "
        f"JOIN {cost_table} k ON k.i12_pit_candidate_id = c.i12_pit_candidate_id "
        "WHERE k.is_active = :child_active GROUP BY bucket, k.exit_role ORDER BY bucket, k.exit_role"
    )
    cost_params = {**params, "child_active": True}
    for row in session.execute(cost_query, cost_params).mappings():
        day = str(row["bucket"])
        role = str(row["exit_role"])
        if role not in EXIT_ROLES:
            continue
        bucket = rows_by_day.setdefault(day, _empty_daily_row(day))
        _set_cost_bucket_from_aggregate(bucket[role], row)

    rows = []
    for day, row in sorted(rows_by_day.items()):
        for role in EXIT_ROLES:
            _finalize_aggregate_cost_bucket(row[role])
            if row["passed_candidates"] == 0:
                row[role]["mean_modeled_return_skips_as_cash"] = 0.0
            mean_value = row[role]["mean_modeled_return_skips_as_cash"]
            row[role]["positive_day"] = mean_value is not None and mean_value > 0.0
            row[role]["negative_day"] = mean_value is not None and mean_value < 0.0
        rows.append(row)
    source_day_count = len(rows)
    trading_candidate_day_count = sum(
        1 for row in rows if _sql_int(row.get("passed_candidates")) > 0
    )
    no_trade_day_count = source_day_count - trading_candidate_day_count
    return {
        "rows": rows,
        "source_day_count": source_day_count,
        "candidate_day_count": source_day_count,
        "no_trade_day_count": no_trade_day_count,
        "trading_candidate_day_count": trading_candidate_day_count,
        "summary": {
            role: _daily_role_summary(rows, role) for role in EXIT_ROLES
        },
        "candidate_days_only_summary": {
            role: _daily_role_summary(
                [row for row in rows if _sql_int(row.get("passed_candidates")) > 0],
                role,
            )
            for role in EXIT_ROLES
        },
    }


def _load_candidate_rows(
    session: Session,
    schema: str,
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    table = _qualified_table(session, schema, "i12_pit_candidates")
    conditions = ["is_active = :active"]
    params: dict[str, Any] = {"active": True}
    path_mode = report.get("report_path_mode")
    if path_mode:
        conditions.append("path_mode = :path_mode")
        params["path_mode"] = path_mode
    decision_labels = list(report.get("report_decision_time_labels") or [])
    _append_in_condition(conditions, params, "decision_time_label", decision_labels, "dt")
    start_date = report.get("start_date") or report.get("progress_source_start_date")
    end_date = report.get("end_date") or report.get("progress_source_end_date")
    if start_date:
        conditions.append("decision_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        conditions.append("decision_date <= :end_date")
        params["end_date"] = end_date
    query = text(
        "SELECT i12_pit_candidate_id, ticker, decision_date, decision_time_label, "
        "path_mode, candidate_status, coverage_status "
        f"FROM {table} WHERE {' AND '.join(conditions)}"
    )
    return [dict(row) for row in session.execute(query, params).mappings().all()]


def _load_quote_rows(
    session: Session,
    schema: str,
    candidate_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not candidate_ids:
        return []
    table = _qualified_table(session, schema, "i12_pit_quote_replays")
    rows: list[dict[str, Any]] = []
    for chunk in _chunks(candidate_ids, 900):
        conditions = ["is_active = :active"]
        params: dict[str, Any] = {"active": True}
        _append_in_condition(conditions, params, "i12_pit_candidate_id", chunk, "candidate")
        query = text(
            "SELECT i12_pit_candidate_id, quote_role, coverage_status "
            f"FROM {table} WHERE {' AND '.join(conditions)}"
        )
        rows.extend(dict(row) for row in session.execute(query, params).mappings().all())
    return rows


def _load_cost_rows(
    session: Session,
    schema: str,
    candidate_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not candidate_ids:
        return []
    table = _qualified_table(session, schema, "i12_pit_cost_replays")
    rows: list[dict[str, Any]] = []
    for chunk in _chunks(candidate_ids, 900):
        conditions = ["is_active = :active"]
        params: dict[str, Any] = {"active": True}
        _append_in_condition(conditions, params, "i12_pit_candidate_id", chunk, "candidate")
        query = text(
            "SELECT i12_pit_candidate_id, decision_date, exit_role, "
            "tradeability_status, skipped_reason, modeled_return "
            f"FROM {table} WHERE {' AND '.join(conditions)}"
        )
        rows.extend(dict(row) for row in session.execute(query, params).mappings().all())
    return rows


def _monthly_rows(
    candidates: Sequence[Mapping[str, Any]],
    quotes: Sequence[Mapping[str, Any]],
    costs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidate_by_id = {
        str(row["i12_pit_candidate_id"]): row for row in candidates
    }
    rows_by_month: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        month = _month_key(candidate.get("decision_date"))
        bucket = rows_by_month.setdefault(month, _empty_month_row(month))
        bucket["source_attempts"] += 1
        if candidate.get("candidate_status") == "passed":
            bucket["passed_candidates"] += 1
    for quote in quotes:
        candidate = candidate_by_id.get(str(quote.get("i12_pit_candidate_id")))
        if candidate is None:
            continue
        month = _month_key(candidate.get("decision_date"))
        bucket = rows_by_month.setdefault(month, _empty_month_row(month))
        if quote.get("coverage_status") == "ok":
            bucket["quote_ok_count"] += 1
        else:
            bucket["quote_non_ok_count"] += 1
    for cost in costs:
        candidate = candidate_by_id.get(str(cost.get("i12_pit_candidate_id")))
        if candidate is None:
            continue
        month = _month_key(candidate.get("decision_date"))
        bucket = rows_by_month.setdefault(month, _empty_month_row(month))
        role = str(cost.get("exit_role"))
        if role not in EXIT_ROLES:
            continue
        _accumulate_cost(bucket[role], cost)
        skipped_reason = str(cost.get("skipped_reason") or "unknown")
        bucket["skip_reason_summary"][skipped_reason] += 1
    out = []
    for month, row in sorted(rows_by_month.items()):
        row["pass_rate"] = _safe_rate(row["passed_candidates"], row["source_attempts"])
        row["skip_reason_summary"] = dict(row["skip_reason_summary"])
        for role in EXIT_ROLES:
            _finalize_cost_bucket(row[role])
        out.append(row)
    return out


def _daily_distribution(
    candidates: Sequence[Mapping[str, Any]],
    costs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_by_id = {
        str(row["i12_pit_candidate_id"]): row for row in candidates
    }
    rows_by_day: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        day = _date_key(candidate.get("decision_date"))
        bucket = rows_by_day.setdefault(day, _empty_daily_row(day))
        if candidate.get("candidate_status") == "passed":
            bucket["passed_candidates"] += 1
    for cost in costs:
        candidate = candidate_by_id.get(str(cost.get("i12_pit_candidate_id")))
        if candidate is None:
            continue
        day = _date_key(candidate.get("decision_date"))
        role = str(cost.get("exit_role"))
        if role not in EXIT_ROLES:
            continue
        _accumulate_cost(rows_by_day.setdefault(day, _empty_daily_row(day))[role], cost)
    rows = []
    for day, row in sorted(rows_by_day.items()):
        for role in EXIT_ROLES:
            _finalize_cost_bucket(row[role])
            mean_value = row[role]["mean_modeled_return_skips_as_cash"]
            row[role]["positive_day"] = mean_value is not None and mean_value > 0.0
            row[role]["negative_day"] = mean_value is not None and mean_value < 0.0
        rows.append(row)
    return {
        "rows": rows,
        "summary": {
            role: _daily_role_summary(rows, role) for role in EXIT_ROLES
        },
    }


def _empty_month_row(month: str) -> dict[str, Any]:
    return {
        "month": month,
        "source_attempts": 0,
        "passed_candidates": 0,
        "pass_rate": None,
        "same_day_exit": _empty_cost_bucket(),
        "next_open_exit": _empty_cost_bucket(),
        "quote_ok_count": 0,
        "quote_non_ok_count": 0,
        "skip_reason_summary": Counter(),
    }


def _empty_daily_row(day: str) -> dict[str, Any]:
    return {
        "decision_date": day,
        "passed_candidates": 0,
        "same_day_exit": _empty_cost_bucket(),
        "next_open_exit": _empty_cost_bucket(),
    }


def _empty_cost_bucket() -> dict[str, Any]:
    return {
        "candidates": 0,
        "tradeable_count": 0,
        "skipped_cash_count": 0,
        "modeled_returns": [],
        "mean_modeled_return_skips_as_cash": None,
        "win_rate_skips_as_cash": None,
        "skipped_cash_by_reason": Counter(),
    }


def _accumulate_cost(bucket: dict[str, Any], cost: Mapping[str, Any]) -> None:
    bucket["candidates"] += 1
    if cost.get("tradeability_status") == "tradeable":
        bucket["tradeable_count"] += 1
    else:
        bucket["skipped_cash_count"] += 1
    bucket["skipped_cash_by_reason"][str(cost.get("skipped_reason") or "unknown")] += 1
    value = _float_or_none(cost.get("modeled_return"))
    if value is not None:
        bucket["modeled_returns"].append(value)


def _set_cost_bucket_from_aggregate(
    bucket: dict[str, Any],
    aggregate: Mapping[str, Any],
) -> None:
    bucket.pop("modeled_returns", None)
    candidates = _sql_int(aggregate.get("candidates"))
    tradeable = _sql_int(aggregate.get("tradeable_count"))
    skipped = _sql_int(aggregate.get("skipped_cash_count"))
    return_count = _sql_int(aggregate.get("return_count"))
    positive = _sql_int(aggregate.get("positive_count"))
    return_sum = _sql_float(aggregate.get("return_sum"))
    bucket["candidates"] = candidates
    bucket["tradeable_count"] = tradeable
    bucket["skipped_cash_count"] = skipped
    bucket["tradeable_rate"] = _safe_rate(tradeable, candidates)
    bucket["mean_modeled_return_skips_as_cash"] = (
        return_sum / return_count if return_sum is not None and return_count else None
    )
    bucket["win_rate_skips_as_cash"] = _safe_rate(positive, return_count)


def _finalize_aggregate_cost_bucket(bucket: dict[str, Any]) -> None:
    bucket.pop("modeled_returns", None)
    bucket["tradeable_rate"] = _safe_rate(
        bucket.get("tradeable_count", 0),
        bucket.get("candidates", 0),
    )
    if isinstance(bucket.get("skipped_cash_by_reason"), Counter):
        bucket["skipped_cash_by_reason"] = dict(bucket["skipped_cash_by_reason"])


def _finalize_cost_bucket(bucket: dict[str, Any]) -> None:
    values = bucket.pop("modeled_returns", [])
    bucket["tradeable_rate"] = _safe_rate(bucket["tradeable_count"], bucket["candidates"])
    bucket["mean_modeled_return_skips_as_cash"] = _mean(values)
    bucket["win_rate_skips_as_cash"] = _safe_rate(
        sum(1 for value in values if value > 0.0),
        len(values),
    )
    bucket["skipped_cash_by_reason"] = dict(bucket["skipped_cash_by_reason"])


def _daily_role_summary(rows: Sequence[Mapping[str, Any]], role: str) -> dict[str, Any]:
    values = [
        _nested(row, role, "mean_modeled_return_skips_as_cash")
        for row in rows
    ]
    values = [float(value) for value in values if value is not None]
    if not values:
        return {}
    sorted_values = sorted(values)
    best = max(values)
    worst = min(values)
    positive_total = sum(value for value in values if value > 0.0)
    return {
        "day_count": len(values),
        "positive_day_count": sum(1 for value in values if value > 0.0),
        "positive_day_rate": _safe_rate(sum(1 for value in values if value > 0.0), len(values)),
        "mean_daily_return": _mean(values),
        "median_daily_return": median(values),
        "p10_daily_return": _percentile(sorted_values, 0.10),
        "p25_daily_return": _percentile(sorted_values, 0.25),
        "p75_daily_return": _percentile(sorted_values, 0.75),
        "p90_daily_return": _percentile(sorted_values, 0.90),
        "best_day_return": best,
        "worst_day_return": worst,
        "top_1_positive_return_share": _top_share(values, positive_total, 1),
        "top_3_positive_return_share": _top_share(values, positive_total, 3),
        "top_5_positive_return_share": _top_share(values, positive_total, 5),
    }


def _quality_warnings(
    integrity: Mapping[str, Any],
    exits: Mapping[str, Mapping[str, Any]],
    db_analysis: Mapping[str, Any],
    *,
    tradeable_rate_threshold: float,
) -> list[str]:
    warnings: list[str] = []
    if integrity.get("conclusions_final") is not True:
        warnings.append("report_non_final")
    if integrity.get("data_integrity_passed") is not True:
        warnings.append("data_integrity_not_passed")
    for key in ("source_replay_complete", "quote_replay_complete", "cost_replay_complete"):
        if integrity.get(key) is not True:
            warnings.append(f"{key}_false")
    if (integrity.get("quote_non_ok_count") or 0) > 0:
        warnings.append("quote_non_ok_rows_present")
    same_day = _mapping(exits.get("same_day_exit"))
    next_open = _mapping(exits.get("next_open_exit"))
    same_day_mean = _float_or_none(same_day.get("mean_modeled_return_skips_as_cash"))
    same_day_win = _float_or_none(same_day.get("win_rate_skips_as_cash"))
    next_open_mean = _float_or_none(next_open.get("mean_modeled_return_skips_as_cash"))
    if same_day_mean is not None and same_day_mean > 0.0 and (same_day_win or 0.0) < 0.50:
        warnings.append("same_day_positive_mean_but_win_rate_below_50pct")
    if (
        same_day_mean is not None
        and next_open_mean is not None
        and next_open_mean < same_day_mean - MATERIAL_EXIT_GAP
    ):
        warnings.append("next_open_materially_worse_than_same_day")
    for role, metrics in exits.items():
        tradeable_rate = _float_or_none(metrics.get("tradeable_rate"))
        if tradeable_rate is not None and tradeable_rate < tradeable_rate_threshold:
            warnings.append(f"{role}_tradeable_rate_below_threshold")
    if (integrity.get("quote_ok_count") or 0) < 1 and integrity.get("quote_ok_count") is not None:
        warnings.append("no_ok_quotes")
    if (integrity.get("quote_non_ok_count") or 0) > 0:
        warnings.append("quote_quality_review_required")
    passed_count = _first_not_none(
        same_day.get("candidates"),
        integrity.get("pit_candidate_count"),
    )
    if passed_count is not None and passed_count < MIN_PASSED_CANDIDATES_FOR_RANKING:
        warnings.append("passed_candidate_count_too_small_for_ml_conclusions")
    daily_summary = _nested(db_analysis, "daily_distribution", "summary", "same_day_exit")
    top3_share = _float_or_none(_mapping(daily_summary).get("top_3_positive_return_share"))
    if top3_share is not None and top3_share > CONCENTRATION_WARNING_THRESHOLD:
        warnings.append("top_3_days_concentrated_positive_return")
    return sorted(set(warnings))


def _load_reports(
    paths: Sequence[str | Path],
    *,
    labels: Sequence[str] | None = None,
) -> list[LoadedReport]:
    labels = labels or ()
    expanded: list[Path] = []
    for path_value in paths:
        for item in str(path_value).split(","):
            item = item.strip()
            if item:
                expanded.append(Path(item))
    loaded: list[LoadedReport] = []
    for index, path in enumerate(expanded):
        label = labels[index] if index < len(labels) else path.stem
        try:
            payload = json.loads(path.read_text())
        except FileNotFoundError:
            loaded.append(LoadedReport(label=label, path=path, report=None, error="missing_report_file"))
            continue
        except json.JSONDecodeError as exc:
            loaded.append(LoadedReport(label=label, path=path, report=None, error=f"invalid_json: {exc}"))
            continue
        if not isinstance(payload, dict):
            loaded.append(LoadedReport(label=label, path=path, report=None, error="report_root_not_object"))
            continue
        loaded.append(LoadedReport(label=label, path=path, report=payload))
    return loaded


def _validate_labels(paths: Sequence[str | Path], labels: Sequence[str] | None) -> None:
    if not labels:
        return
    expanded_count = sum(1 for path in paths for item in str(path).split(",") if item.strip())
    if len(labels) not in {0, expanded_count}:
        raise ValueError("--label count must match expanded --report count")


def _validate_scratch_schema(schema: str) -> None:
    normalized = str(schema or "").strip()
    lowered = normalized.casefold()
    if lowered in FORBIDDEN_SCHEMAS or not lowered.startswith("scratch_"):
        raise ValueError(
            "DB-backed analysis requires a named scratch schema; refusing "
            f"schema {schema!r}"
        )


def _build_db_context(
    schema: str,
    database_url: str | None,
    db_session: Session | None,
    *,
    start_date: str | None,
    end_date: str | None,
    allow_unbounded_schema: bool,
) -> dict[str, Any]:
    if bool(start_date) != bool(end_date):
        raise ValueError("--start-date and --end-date must be provided together")
    if start_date is not None:
        start_date = _validate_date_text(start_date, "--start-date")
    if end_date is not None:
        end_date = _validate_date_text(end_date, "--end-date")
    if start_date and end_date and date.fromisoformat(start_date) > date.fromisoformat(end_date):
        raise ValueError("--start-date must be <= --end-date")
    if db_session is not None:
        return {
            "schema": schema,
            "session": db_session,
            "owns_session": False,
            "start_date": start_date,
            "end_date": end_date,
            "allow_unbounded_schema": allow_unbounded_schema,
        }
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("--schema requires --database-url or DATABASE_URL")
    engine = create_engine(url)
    session = Session(engine)
    return {
        "schema": schema,
        "session": session,
        "owns_session": True,
        "start_date": start_date,
        "end_date": end_date,
        "allow_unbounded_schema": allow_unbounded_schema,
    }


def _close_owned_db_session(session: Session) -> None:
    try:
        in_transaction = getattr(session, "in_transaction", lambda: False)
        if in_transaction():
            session.rollback()
    finally:
        session.close()


def _validate_date_text(value: str, flag_name: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{flag_name} must be YYYY-MM-DD") from exc
    return parsed.isoformat()


def _qualified_table(session: Session, schema: str, table: str) -> str:
    dialect = session.get_bind().dialect.name
    if dialect == "sqlite":
        return table
    return f"{_quote_ident(schema)}.{_quote_ident(table)}"


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _append_in_condition(
    conditions: list[str],
    params: dict[str, Any],
    column: str,
    values: Sequence[Any],
    prefix: str,
) -> None:
    if not values:
        return
    names = []
    for index, value in enumerate(values):
        name = f"{prefix}_{index}"
        params[name] = value
        names.append(f":{name}")
    conditions.append(f"{column} IN ({', '.join(names)})")


def _chunks(values: Sequence[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield list(values[index:index + size])


def _month_key(value: Any) -> str:
    return _date_key(value)[:7]


def _date_key(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(sorted_values: Sequence[float], quantile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _top_share(values: Sequence[float], positive_total: float, count: int) -> float | None:
    if positive_total <= 0:
        return None
    top = sorted((value for value in values if value > 0.0), reverse=True)[:count]
    return sum(top) / positive_total


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        value = _mapping(value).get(key)
    return value


def _date_range(mapping: Mapping[str, Any] | None) -> str:
    values = _mapping(mapping)
    start = values.get("start_date")
    end = values.get("end_date")
    if start and end:
        return f"{start}..{end}"
    return ""


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    text_rows = [[str(cell) if cell is not None else "" for cell in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in text_rows)) if text_rows else len(header)
        for index, header in enumerate(headers)
    ]
    header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    sep = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        for row in text_rows
    ]
    return "\n".join([header_line, sep, *body])


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return _fmt_bool(value)
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return str(value)


def _fmt_bool(value: Any) -> str:
    if value is None:
        return ""
    return "true" if bool(value) else "false"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, help="Report JSON path; repeat or comma-separate.")
    parser.add_argument("--label", action="append", default=[], help="Optional label; repeatable.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", help="Optional output path.")
    parser.add_argument("--require-final", action="store_true", help="Exit nonzero unless every report is final and integrity-passed.")
    parser.add_argument("--schema", help="Optional scratch schema for DB-backed monthly/daily analysis.")
    parser.add_argument("--database-url", help="Database URL; defaults to DATABASE_URL.")
    parser.add_argument("--start-date", help="Explicit DB-backed analysis start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Explicit DB-backed analysis end date, YYYY-MM-DD.")
    parser.add_argument(
        "--allow-unbounded-schema",
        action="store_true",
        help="Allow diagnostic DB-backed analysis without date bounds.",
    )
    parser.add_argument("--tradeable-rate-threshold", type=float, default=DEFAULT_TRADEABLE_RATE_THRESHOLD)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        analysis = analyze_report_paths(
            args.report,
            labels=args.label,
            schema=args.schema,
            database_url=args.database_url,
            start_date=args.start_date,
            end_date=args.end_date,
            allow_unbounded_schema=args.allow_unbounded_schema,
            require_final=args.require_final,
            tradeable_rate_threshold=args.tradeable_rate_threshold,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    payload = (
        json.dumps(analysis, indent=2, sort_keys=True, default=str)
        if args.format == "json"
        else render_text_analysis(analysis)
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload + "\n")
    else:
        print(payload)
    if analysis["require_final_failures"]:
        return 2
    if any(report.get("error") for report in analysis["reports"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
