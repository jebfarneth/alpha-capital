#!/usr/bin/env python3
"""Compare two PIT-clean I12 decision-time snapshot schemas.

This tool is read-only. It compares active PIT candidate attempts and their
quote/cost replay evidence across two named scratch schemas for a bounded date
window and path mode.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence


EXIT_ROLES = ("same_day_exit", "next_open_exit")
QUOTE_ROLES = ("entry", "same_day_exit", "next_open_exit")
FORBIDDEN_SCHEMAS = {
    "",
    "canonical",
    "default",
    "information_schema",
    "main",
    "pg_catalog",
    "prod",
    "production",
    "public",
}
SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SAMPLE_LIMIT = 20


def compare_snapshots(
    *,
    left_schema: str,
    left_label: str,
    right_schema: str,
    right_label: str,
    start_date: str | date,
    end_date: str | date,
    decision_time_left: str,
    decision_time_right: str,
    minute_path_mode: str,
    left_report: str | Path | None = None,
    right_report: str | Path | None = None,
    database_url: str | None = None,
    db_session: Session | None = None,
    require_final: bool = False,
) -> dict[str, Any]:
    left_schema = _validate_scratch_schema(left_schema, side="left")
    right_schema = _validate_scratch_schema(right_schema, side="right")
    start = _parse_date_required(start_date, "start_date")
    end = _parse_date_required(end_date, "end_date")
    if end < start:
        raise ValueError("end_date must be >= start_date")
    if minute_path_mode not in {"strict_contiguous", "sparse_zero_fill"}:
        raise ValueError("minute_path_mode must be strict_contiguous or sparse_zero_fill")
    _validate_decision_time(decision_time_left)
    _validate_decision_time(decision_time_right)
    finality = _load_and_validate_final_reports(
        require_final=require_final,
        left_report=left_report,
        right_report=right_report,
        left_schema=left_schema,
        right_schema=right_schema,
        start_date=start,
        end_date=end,
        decision_time_left=decision_time_left,
        decision_time_right=decision_time_right,
        minute_path_mode=minute_path_mode,
    )

    owns_session = False
    session = db_session
    if session is None:
        url = database_url or os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL is required unless db_session is provided")
        session = _create_session(url)
        owns_session = True
    assert session is not None
    dialect = getattr(getattr(session.get_bind(), "dialect", None), "name", "")
    params = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "left_decision": decision_time_left,
        "right_decision": decision_time_right,
        "path_mode": minute_path_mode,
    }
    try:
        left_integrity = _side_integrity(
            session,
            dialect=dialect,
            schema=left_schema,
            label=left_label,
            decision_time=decision_time_left,
            params=params,
        )
        right_integrity = _side_integrity(
            session,
            dialect=dialect,
            schema=right_schema,
            label=right_label,
            decision_time=decision_time_right,
            params=params,
        )
        _fail_on_duplicate_active_candidates(left_integrity, side="left")
        _fail_on_duplicate_active_candidates(right_integrity, side="right")
        _fail_on_duplicate_child_evidence(left_integrity, side="left")
        _fail_on_duplicate_child_evidence(right_integrity, side="right")
        left_dates = set(left_integrity.get("observed_dates") or [])
        right_dates = set(right_integrity.get("observed_dates") or [])
        left_integrity["missing_days"] = sorted(right_dates - left_dates)
        right_integrity["missing_days"] = sorted(left_dates - right_dates)
        overlap = _overlap(session, left_schema, right_schema, params)
        transitions = _transitions(session, left_schema, right_schema, params)
        economics = _economics(session, left_schema, right_schema, params)
        liquidity = _liquidity_deltas(
            session,
            dialect=dialect,
            left_schema=left_schema,
            right_schema=right_schema,
            params=params,
        )
        edge_timing = _edge_timing(economics, liquidity)
        samples = _samples(
            session,
            dialect=dialect,
            left_schema=left_schema,
            right_schema=right_schema,
            params=params,
        )
        warnings = _warnings(
            left_integrity=left_integrity,
            right_integrity=right_integrity,
            overlap=overlap,
            liquidity=liquidity,
            require_final=require_final,
        )
        return {
            "inputs": {
                "left_schema": left_schema,
                "left_label": left_label,
                "right_schema": right_schema,
                "right_label": right_label,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "decision_time_left": decision_time_left,
                "decision_time_right": decision_time_right,
                "minute_path_mode": minute_path_mode,
                "require_final": require_final,
            },
            "finality": finality,
            "integrity": {
                "left": left_integrity,
                "right": right_integrity,
            },
            "overlap": overlap,
            "transitions": transitions,
            "economics": economics,
            "economics_basis": {
                "headline": "diagnostic_displayed_size_cost_replay_only",
                "volume_participation": (
                    "use build_i12_pit_event_tape.py for volume-participation "
                    "economics from persisted candidate/quote/cost evidence"
                ),
            },
            "liquidity_deltas": liquidity,
            "edge_timing": edge_timing,
            "samples": samples,
            "warnings": warnings,
        }
    finally:
        if owns_session:
            session.rollback()
            session.close()


def render_text(analysis: Mapping[str, Any]) -> str:
    inputs = _mapping(analysis.get("inputs"))
    lines: list[str] = []
    lines.append("I12 PIT Snapshot Comparison")
    lines.append(
        f"{inputs.get('left_label')} ({inputs.get('decision_time_left')}) vs "
        f"{inputs.get('right_label')} ({inputs.get('decision_time_right')})"
    )
    lines.append(
        f"range={inputs.get('start_date')}..{inputs.get('end_date')} "
        f"path_mode={inputs.get('minute_path_mode')}"
    )
    lines.append("")
    lines.append("Corpus / Integrity")
    rows = []
    for side in ("left", "right"):
        item = _mapping(_nested(analysis, "integrity", side))
        rows.append(
            [
                side,
                item.get("label"),
                item.get("schema"),
                _fmt(item.get("active_candidate_row_count")),
                _fmt(item.get("passed_count")),
                _fmt(item.get("date_count")),
                item.get("min_date") or "-",
                item.get("max_date") or "-",
                _fmt(item.get("duplicate_active_ticker_date_count")),
                _fmt(item.get("missing_quote_role_count")),
                _fmt(item.get("duplicate_quote_role_count")),
                _fmt(item.get("missing_cost_role_count")),
                _fmt(item.get("duplicate_cost_role_count")),
            ]
        )
    lines.append(
        _format_table(
            [
                "side",
                "label",
                "schema",
                "active",
                "passed",
                "dates",
                "min",
                "max",
                "dups",
                "q_miss",
                "q_dup",
                "c_miss",
                "c_dup",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append("Overlap")
    overlap = _mapping(analysis.get("overlap"))
    lines.append(
        _format_table(
            [
                "passed_left",
                "passed_right",
                "both",
                "left_only",
                "right_only",
                "jaccard",
            ],
            [
                [
                    _fmt(overlap.get("passed_left_count")),
                    _fmt(overlap.get("passed_right_count")),
                    _fmt(overlap.get("passed_both_count")),
                    _fmt(overlap.get("left_only_count")),
                    _fmt(overlap.get("right_only_count")),
                    _fmt_pct(overlap.get("jaccard_overlap")),
                ]
            ],
        )
    )
    lines.append("")
    lines.append("Transition Matrix")
    trans = _mapping(analysis.get("transitions"))
    lines.append(
        _format_table(
            ["transition", "count"],
            [[key, _fmt(value)] for key, value in sorted(_mapping(trans.get("matrix")).items())],
        )
    )
    lines.append("")
    lines.append("Economics")
    basis = _mapping(analysis.get("economics_basis"))
    if basis:
        lines.append(f"basis={basis.get('headline')}")
    econ_rows = []
    for role, role_metrics in _mapping(analysis.get("economics")).items():
        for group, metrics in _mapping(role_metrics).items():
            metrics = _mapping(metrics)
            econ_rows.append(
                [
                    role,
                    group,
                    _fmt(metrics.get("candidates")),
                    _fmt(metrics.get("tradeable_count")),
                    _fmt(metrics.get("skipped_cash_count")),
                    _fmt_pct(metrics.get("mean_modeled_return_skips_as_cash")),
                    _fmt_pct(metrics.get("win_rate_skips_as_cash")),
                    json.dumps(metrics.get("skipped_cash_by_reason") or {}, sort_keys=True),
                ]
            )
    lines.append(
        _format_table(
            ["exit", "group", "cand", "trade", "skip", "mean_cash", "win", "skip_reasons"],
            econ_rows,
        )
    )
    lines.append("")
    lines.append("Liquidity Deltas (right - left, both passed)")
    liq = _mapping(analysis.get("liquidity_deltas"))
    delta_rows = [
        [name, _fmt_summary(_mapping(liq.get(name)))]
        for name in (
            "entry_spread_bps_delta",
            "entry_executable_notional_delta",
            "observed_cumulative_volume_delta",
            "observed_minute_count_delta",
            "path_coverage_ratio_delta",
        )
    ]
    lines.append(_format_table(["metric", "summary"], delta_rows))
    lines.append("")
    lines.append("Edge Timing")
    lines.append(json.dumps(analysis.get("edge_timing") or {}, sort_keys=True))
    warnings = list(analysis.get("warnings") or [])
    if warnings:
        lines.append("")
        lines.append("Warnings")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def _side_integrity(
    session: Session,
    *,
    dialect: str,
    schema: str,
    label: str,
    decision_time: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = _q(schema, "i12_pit_candidates")
    quotes = _q(schema, "i12_pit_quote_replays")
    costs = _q(schema, "i12_pit_cost_replays")
    scope_params = dict(params)
    scope_params["decision_time"] = decision_time
    row = _one(
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
        scope_params,
    )
    duplicate_groups = _scalar(
        session,
        f"""
        SELECT COUNT(*) FROM (
          SELECT ticker, decision_date, COUNT(*) AS row_count
          FROM {candidates}
          WHERE is_active IS TRUE
            AND decision_date BETWEEN :start_date AND :end_date
            AND decision_time_label = :decision_time
            AND path_mode = :path_mode
          GROUP BY ticker, decision_date
          HAVING COUNT(*) > 1
        ) d
        """,
        scope_params,
    )
    available_times = _list_values(
        session,
        f"""
        SELECT DISTINCT decision_time_label
        FROM {candidates}
        WHERE is_active IS TRUE
          AND decision_date BETWEEN :start_date AND :end_date
        ORDER BY decision_time_label
        """,
        scope_params,
    )
    available_modes = _list_values(
        session,
        f"""
        SELECT DISTINCT path_mode
        FROM {candidates}
        WHERE is_active IS TRUE
          AND decision_date BETWEEN :start_date AND :end_date
        ORDER BY path_mode
        """,
        scope_params,
    )
    observed_dates = _list_values(
        session,
        f"""
        SELECT decision_date
        FROM {candidates}
        WHERE is_active IS TRUE
          AND decision_date BETWEEN :start_date AND :end_date
          AND decision_time_label = :decision_time
          AND path_mode = :path_mode
        GROUP BY decision_date
        ORDER BY decision_date
        """,
        scope_params,
    )
    quote_completeness = _quote_completeness(session, candidates, quotes, scope_params)
    cost_completeness = _cost_completeness(session, candidates, costs, scope_params)
    feature_summary = _feature_summary(session, dialect, candidates, scope_params)
    return {
        "schema": schema,
        "label": label,
        "decision_time_label": decision_time,
        "active_candidate_row_count": int(row.get("active_candidate_row_count") or 0),
        "passed_count": int(row.get("passed_count") or 0),
        "date_count": int(row.get("date_count") or 0),
        "min_date": _str_or_none(row.get("min_date")),
        "max_date": _str_or_none(row.get("max_date")),
        "observed_dates": observed_dates,
        "missing_days": [],
        "duplicate_active_ticker_date_count": int(duplicate_groups or 0),
        "available_decision_time_labels": available_times,
        "available_path_modes": available_modes,
        **quote_completeness,
        **cost_completeness,
        "feature_summary": feature_summary,
    }


def _overlap(
    session: Session,
    left_schema: str,
    right_schema: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    rows = _all(
        session,
        _paired_sql(left_schema, right_schema)
        + """
        SELECT
          COUNT(*) AS identity_count,
          SUM(CASE WHEN lstatus = 'passed' THEN 1 ELSE 0 END) AS passed_left_count,
          SUM(CASE WHEN rstatus = 'passed' THEN 1 ELSE 0 END) AS passed_right_count,
          SUM(CASE WHEN lstatus = 'passed' AND rstatus = 'passed' THEN 1 ELSE 0 END) AS passed_both_count,
          SUM(CASE WHEN lstatus = 'passed' AND COALESCE(rstatus, '') != 'passed' THEN 1 ELSE 0 END) AS left_only_count,
          SUM(CASE WHEN rstatus = 'passed' AND COALESCE(lstatus, '') != 'passed' THEN 1 ELSE 0 END) AS right_only_count
        FROM paired
        """,
        params,
    )
    row = rows[0] if rows else {}
    both = int(row.get("passed_both_count") or 0)
    left_only = int(row.get("left_only_count") or 0)
    right_only = int(row.get("right_only_count") or 0)
    union = both + left_only + right_only
    daily = _all(
        session,
        _paired_sql(left_schema, right_schema)
        + """
        SELECT decision_date,
          SUM(CASE WHEN lstatus = 'passed' AND rstatus = 'passed' THEN 1 ELSE 0 END) AS both_count,
          SUM(CASE WHEN lstatus = 'passed' AND COALESCE(rstatus, '') != 'passed' THEN 1 ELSE 0 END) AS left_only_count,
          SUM(CASE WHEN rstatus = 'passed' AND COALESCE(lstatus, '') != 'passed' THEN 1 ELSE 0 END) AS right_only_count
        FROM paired
        GROUP BY decision_date
        ORDER BY decision_date
        """,
        params,
    )
    return {
        "identity_count": int(row.get("identity_count") or 0),
        "passed_left_count": int(row.get("passed_left_count") or 0),
        "passed_right_count": int(row.get("passed_right_count") or 0),
        "passed_both_count": both,
        "left_only_count": left_only,
        "right_only_count": right_only,
        "jaccard_overlap": both / union if union else None,
        "per_day_overlap_distribution": {
            "both_count": _distribution([item.get("both_count") for item in daily]),
            "left_only_count": _distribution([item.get("left_only_count") for item in daily]),
            "right_only_count": _distribution([item.get("right_only_count") for item in daily]),
        },
    }


def _transitions(
    session: Session,
    left_schema: str,
    right_schema: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    rows = _all(
        session,
        _paired_sql(left_schema, right_schema)
        + """
        SELECT
          CASE WHEN lstatus = 'passed' THEN 'left_passed' ELSE 'left_failed' END
          || ' -> ' ||
          CASE WHEN rstatus = 'passed' THEN 'right_passed' ELSE 'right_failed' END
          AS transition,
          COUNT(*) AS row_count
        FROM paired
        GROUP BY transition
        """,
        params,
    )
    fail_rows = _all(
        session,
        _paired_sql(left_schema, right_schema)
        + """
        SELECT
          COALESCE(lcoverage, 'missing_snapshot_row') AS left_coverage_status,
          COALESCE(lfail, 'none') AS left_fail_reason,
          COALESCE(rcoverage, 'missing_snapshot_row') AS right_coverage_status,
          COALESCE(rfail, 'none') AS right_fail_reason,
          COUNT(*) AS row_count
        FROM paired
        GROUP BY left_coverage_status, left_fail_reason, right_coverage_status, right_fail_reason
        ORDER BY row_count DESC
        LIMIT 25
        """,
        params,
    )
    return {
        "matrix": {row["transition"]: int(row["row_count"] or 0) for row in rows},
        "coverage_fail_reason_top": fail_rows,
    }


def _economics(
    session: Session,
    left_schema: str,
    right_schema: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for role in EXIT_ROLES:
        out[role] = {}
        for group, side in (
            ("left_passed", "left"),
            ("right_passed", "right"),
            ("both_left", "left"),
            ("both_right", "right"),
            ("left_only", "left"),
            ("right_only", "right"),
            ("left_passed_right_failed", "left"),
            ("left_failed_right_passed", "right"),
        ):
            out[role][group] = _cost_metrics_for_group(
                session,
                left_schema,
                right_schema,
                params,
                role=role,
                group=group,
                side=side,
            )
    return out


def _cost_metrics_for_group(
    session: Session,
    left_schema: str,
    right_schema: str,
    params: Mapping[str, Any],
    *,
    role: str,
    group: str,
    side: str,
) -> dict[str, Any]:
    cost_schema = left_schema if side == "left" else right_schema
    costs = _q(cost_schema, "i12_pit_cost_replays")
    id_col = "lid" if side == "left" else "rid"
    condition = _group_condition(group)
    sql_params = dict(params)
    sql_params["exit_role"] = role
    summary = _one(
        session,
        _paired_sql(left_schema, right_schema)
        + f"""
        , selected AS (
          SELECT {id_col} AS candidate_id
          FROM paired
          WHERE {condition}
        )
        SELECT
          COUNT(*) AS candidates,
          COUNT(c.i12_pit_cost_replay_id) AS cost_row_count,
          SUM(CASE WHEN c.i12_pit_cost_replay_id IS NULL THEN 1 ELSE 0 END) AS missing_cost_count,
          SUM(CASE WHEN c.tradeability_status = 'tradeable' THEN 1 ELSE 0 END) AS tradeable_count,
          SUM(CASE WHEN c.tradeability_status = 'skipped_cash' THEN 1 ELSE 0 END) AS skipped_cash_count,
          AVG(COALESCE(c.modeled_return, 0.0)) AS mean_model_return,
          AVG(CASE WHEN COALESCE(c.modeled_return, 0.0) > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
          AVG(CASE WHEN c.tradeability_status = 'tradeable' THEN c.modeled_return ELSE NULL END) AS mean_tradeable_return
        FROM selected s
        LEFT JOIN {costs} c
          ON c.i12_pit_candidate_id = s.candidate_id
         AND c.exit_role = :exit_role
         AND c.is_active IS TRUE
        """,
        sql_params,
    )
    skip_rows = _all(
        session,
        _paired_sql(left_schema, right_schema)
        + f"""
        , selected AS (
          SELECT {id_col} AS candidate_id
          FROM paired
          WHERE {condition}
        )
        SELECT COALESCE(c.skipped_reason, 'missing_cost') AS reason, COUNT(*) AS row_count
        FROM selected s
        LEFT JOIN {costs} c
          ON c.i12_pit_candidate_id = s.candidate_id
         AND c.exit_role = :exit_role
         AND c.is_active IS TRUE
        WHERE c.i12_pit_cost_replay_id IS NULL
           OR c.tradeability_status = 'skipped_cash'
        GROUP BY reason
        ORDER BY row_count DESC
        """,
        sql_params,
    )
    return {
        "candidates": int(summary.get("candidates") or 0),
        "cost_row_count": int(summary.get("cost_row_count") or 0),
        "missing_cost_count": int(summary.get("missing_cost_count") or 0),
        "tradeable_count": int(summary.get("tradeable_count") or 0),
        "skipped_cash_count": int(summary.get("skipped_cash_count") or 0),
        "skipped_cash_by_reason": {
            str(row["reason"]): int(row["row_count"] or 0)
            for row in skip_rows
        },
        "mean_modeled_return_skips_as_cash": _float_or_none(summary.get("mean_model_return")),
        "win_rate_skips_as_cash": _float_or_none(summary.get("win_rate")),
        "mean_tradeable_return": _float_or_none(summary.get("mean_tradeable_return")),
    }


def _liquidity_deltas(
    session: Session,
    *,
    dialect: str,
    left_schema: str,
    right_schema: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    lq = _q(left_schema, "i12_pit_quote_replays")
    rq = _q(right_schema, "i12_pit_quote_replays")
    row = _one(
        session,
        _paired_sql(left_schema, right_schema)
        + f"""
        , base AS (
          SELECT
            lqe.coverage_status AS left_entry_coverage_status,
            rqe.coverage_status AS right_entry_coverage_status,
            lqe.spread_bps AS left_spread_bps,
            rqe.spread_bps AS right_spread_bps,
            lqe.quote_age_seconds AS left_quote_age_seconds,
            rqe.quote_age_seconds AS right_quote_age_seconds,
            lqe.executable_notional AS left_executable_notional,
            rqe.executable_notional AS right_executable_notional,
            {_json_number(dialect, 'l', 'observed_cumulative_volume_before_decision')} AS left_observed_volume,
            {_json_number(dialect, 'r', 'observed_cumulative_volume_before_decision')} AS right_observed_volume,
            {_json_number(dialect, 'l', 'observed_minute_count_before_decision')} AS left_observed_minutes,
            {_json_number(dialect, 'r', 'observed_minute_count_before_decision')} AS right_observed_minutes,
            {_json_number(dialect, 'l', 'path_coverage_ratio')} AS left_path_coverage_ratio,
            {_json_number(dialect, 'r', 'path_coverage_ratio')} AS right_path_coverage_ratio
          FROM paired p
          JOIN {_q(left_schema, 'i12_pit_candidates')} l ON l.i12_pit_candidate_id = p.lid
          JOIN {_q(right_schema, 'i12_pit_candidates')} r ON r.i12_pit_candidate_id = p.rid
          LEFT JOIN {lq} lqe
            ON lqe.i12_pit_candidate_id = p.lid
           AND lqe.quote_role = 'entry'
           AND lqe.is_active IS TRUE
          LEFT JOIN {rq} rqe
            ON rqe.i12_pit_candidate_id = p.rid
           AND rqe.quote_role = 'entry'
           AND rqe.is_active IS TRUE
          WHERE p.lstatus = 'passed'
            AND p.rstatus = 'passed'
        )
        SELECT
          COUNT(*) AS paired_passed_count,
          AVG(CASE WHEN COALESCE(left_entry_coverage_status, '') != 'ok' THEN 1.0 ELSE 0.0 END) AS left_entry_quote_non_ok_rate,
          AVG(CASE WHEN COALESCE(right_entry_coverage_status, '') != 'ok' THEN 1.0 ELSE 0.0 END) AS right_entry_quote_non_ok_rate,
          COUNT(CASE WHEN right_spread_bps IS NOT NULL AND left_spread_bps IS NOT NULL THEN 1 END) AS entry_spread_bps_delta_count,
          AVG(CASE WHEN right_spread_bps IS NOT NULL AND left_spread_bps IS NOT NULL THEN right_spread_bps - left_spread_bps END) AS entry_spread_bps_delta_mean,
          COUNT(CASE WHEN right_executable_notional IS NOT NULL AND left_executable_notional IS NOT NULL THEN 1 END) AS entry_executable_notional_delta_count,
          AVG(CASE WHEN right_executable_notional IS NOT NULL AND left_executable_notional IS NOT NULL THEN right_executable_notional - left_executable_notional END) AS entry_executable_notional_delta_mean,
          COUNT(CASE WHEN right_quote_age_seconds IS NOT NULL AND left_quote_age_seconds IS NOT NULL THEN 1 END) AS entry_quote_age_seconds_delta_count,
          AVG(CASE WHEN right_quote_age_seconds IS NOT NULL AND left_quote_age_seconds IS NOT NULL THEN right_quote_age_seconds - left_quote_age_seconds END) AS entry_quote_age_seconds_delta_mean,
          COUNT(CASE WHEN right_observed_volume IS NOT NULL AND left_observed_volume IS NOT NULL THEN 1 END) AS observed_cumulative_volume_delta_count,
          AVG(CASE WHEN right_observed_volume IS NOT NULL AND left_observed_volume IS NOT NULL THEN right_observed_volume - left_observed_volume END) AS observed_cumulative_volume_delta_mean,
          COUNT(CASE WHEN right_observed_minutes IS NOT NULL AND left_observed_minutes IS NOT NULL THEN 1 END) AS observed_minute_count_delta_count,
          AVG(CASE WHEN right_observed_minutes IS NOT NULL AND left_observed_minutes IS NOT NULL THEN right_observed_minutes - left_observed_minutes END) AS observed_minute_count_delta_mean,
          COUNT(CASE WHEN right_path_coverage_ratio IS NOT NULL AND left_path_coverage_ratio IS NOT NULL THEN 1 END) AS path_coverage_ratio_delta_count,
          AVG(CASE WHEN right_path_coverage_ratio IS NOT NULL AND left_path_coverage_ratio IS NOT NULL THEN right_path_coverage_ratio - left_path_coverage_ratio END) AS path_coverage_ratio_delta_mean
        FROM base
        """,
        params,
    )
    return {
        "paired_passed_count": int(row.get("paired_passed_count") or 0),
        "left_entry_quote_non_ok_rate": _float_or_none(row.get("left_entry_quote_non_ok_rate")),
        "right_entry_quote_non_ok_rate": _float_or_none(row.get("right_entry_quote_non_ok_rate")),
        "entry_spread_bps_delta": _sql_summary(row, "entry_spread_bps_delta"),
        "entry_executable_notional_delta": _sql_summary(row, "entry_executable_notional_delta"),
        "entry_quote_age_seconds_delta": _sql_summary(row, "entry_quote_age_seconds_delta"),
        "observed_cumulative_volume_delta": _sql_summary(row, "observed_cumulative_volume_delta"),
        "observed_minute_count_delta": _sql_summary(row, "observed_minute_count_delta"),
        "path_coverage_ratio_delta": _sql_summary(row, "path_coverage_ratio_delta"),
    }


def _edge_timing(economics: Mapping[str, Any], liquidity: Mapping[str, Any]) -> dict[str, Any]:
    same_day = _mapping(economics.get("same_day_exit"))
    return {
        "description": "Descriptive only; not causal.",
        "same_day_left_only_mean": _nested(same_day, "left_only", "mean_modeled_return_skips_as_cash"),
        "same_day_right_only_mean": _nested(same_day, "right_only", "mean_modeled_return_skips_as_cash"),
        "same_day_both_left_mean": _nested(same_day, "both_left", "mean_modeled_return_skips_as_cash"),
        "same_day_both_right_mean": _nested(same_day, "both_right", "mean_modeled_return_skips_as_cash"),
        "same_day_left_passed_right_failed_mean": _nested(
            same_day,
            "left_passed_right_failed",
            "mean_modeled_return_skips_as_cash",
        ),
        "same_day_left_failed_right_passed_mean": _nested(
            same_day,
            "left_failed_right_passed",
            "mean_modeled_return_skips_as_cash",
        ),
        "both_passed_entry_spread_delta_mean": _nested(
            liquidity,
            "entry_spread_bps_delta",
            "mean",
        ),
        "both_passed_entry_notional_delta_mean": _nested(
            liquidity,
            "entry_executable_notional_delta",
            "mean",
        ),
    }


def _samples(
    session: Session,
    *,
    dialect: str,
    left_schema: str,
    right_schema: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "both_passed_top": _sample_rows(session, dialect, left_schema, right_schema, params, "both", "desc"),
        "both_passed_worst": _sample_rows(session, dialect, left_schema, right_schema, params, "both", "asc"),
        "left_only_top": _sample_rows(session, dialect, left_schema, right_schema, params, "left_only", "desc"),
        "left_only_worst": _sample_rows(session, dialect, left_schema, right_schema, params, "left_only", "asc"),
        "right_only_top": _sample_rows(session, dialect, left_schema, right_schema, params, "right_only", "desc"),
        "right_only_worst": _sample_rows(session, dialect, left_schema, right_schema, params, "right_only", "asc"),
    }


def _sample_rows(
    session: Session,
    dialect: str,
    left_schema: str,
    right_schema: str,
    params: Mapping[str, Any],
    group: str,
    direction: str,
) -> list[dict[str, Any]]:
    side = "left" if group in {"left_only", "both"} else "right"
    schema = left_schema if side == "left" else right_schema
    costs = _q(schema, "i12_pit_cost_replays")
    quotes = _q(schema, "i12_pit_quote_replays")
    id_col = "lid" if side == "left" else "rid"
    condition = _group_condition("both_left" if group == "both" else group)
    order = "ASC" if direction == "asc" else "DESC"
    rows = _all(
        session,
        _paired_sql(left_schema, right_schema)
        + f"""
        , selected AS (
          SELECT p.*, {id_col} AS candidate_id
          FROM paired p
          WHERE {condition}
        )
        SELECT
          s.ticker,
          s.decision_date,
          s.lstatus AS left_candidate_status,
          s.rstatus AS right_candidate_status,
          q.spread_bps AS entry_spread_bps,
          q.executable_notional AS entry_executable_notional,
          {_json_number(dialect, 'c', 'observed_cumulative_volume_before_decision')} AS observed_cumulative_volume_before_decision,
          cost.modeled_return AS modeled_return,
          cost.tradeability_status AS tradeability_status,
          cost.skipped_reason AS skipped_reason
        FROM selected s
        JOIN {_q(schema, 'i12_pit_candidates')} c ON c.i12_pit_candidate_id = s.candidate_id
        LEFT JOIN {quotes} q
          ON q.i12_pit_candidate_id = s.candidate_id
         AND q.quote_role = 'entry'
         AND q.is_active IS TRUE
        LEFT JOIN {costs} cost
          ON cost.i12_pit_candidate_id = s.candidate_id
         AND cost.exit_role = 'same_day_exit'
         AND cost.is_active IS TRUE
        ORDER BY COALESCE(cost.modeled_return, 0.0) {order}, s.ticker ASC
        LIMIT {SAMPLE_LIMIT}
        """,
        params,
    )
    return rows


def _quote_completeness(
    session: Session,
    candidates: str,
    quotes: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    by_role: dict[str, dict[str, int]] = {}
    missing_total = 0
    duplicate_total = 0
    for role in QUOTE_ROLES:
        role_params = dict(params)
        role_params["quote_role"] = role
        row = _one(
            session,
            f"""
            WITH passed AS (
              SELECT i12_pit_candidate_id
              FROM {candidates}
              WHERE is_active IS TRUE
                AND candidate_status = 'passed'
                AND decision_date BETWEEN :start_date AND :end_date
                AND decision_time_label = :decision_time
                AND path_mode = :path_mode
            ), quote_counts AS (
              SELECT p.i12_pit_candidate_id, COUNT(q.i12_pit_quote_replay_id) AS row_count
              FROM passed p
              LEFT JOIN {quotes} q
                ON q.i12_pit_candidate_id = p.i12_pit_candidate_id
               AND q.quote_role = :quote_role
               AND q.is_active IS TRUE
              GROUP BY p.i12_pit_candidate_id
            )
            SELECT
              SUM(CASE WHEN row_count = 0 THEN 1 ELSE 0 END) AS missing_count,
              SUM(CASE WHEN row_count > 1 THEN 1 ELSE 0 END) AS duplicate_count
            FROM quote_counts
            """,
            role_params,
        )
        missing = int(row.get("missing_count") or 0)
        duplicate = int(row.get("duplicate_count") or 0)
        by_role[role] = {
            "missing": missing,
            "duplicate": duplicate,
        }
        missing_total += missing
        duplicate_total += duplicate
    return {
        "quote_completeness_by_role": by_role,
        "missing_quote_role_count": missing_total,
        "duplicate_quote_role_count": duplicate_total,
    }


def _cost_completeness(
    session: Session,
    candidates: str,
    costs: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    by_role: dict[str, dict[str, int]] = {}
    missing_total = 0
    duplicate_total = 0
    for role in EXIT_ROLES:
        role_params = dict(params)
        role_params["exit_role"] = role
        row = _one(
            session,
            f"""
            WITH passed AS (
              SELECT i12_pit_candidate_id
              FROM {candidates}
              WHERE is_active IS TRUE
                AND candidate_status = 'passed'
                AND decision_date BETWEEN :start_date AND :end_date
                AND decision_time_label = :decision_time
                AND path_mode = :path_mode
            ), cost_counts AS (
              SELECT p.i12_pit_candidate_id, COUNT(c.i12_pit_cost_replay_id) AS row_count
              FROM passed p
              LEFT JOIN {costs} c
                ON c.i12_pit_candidate_id = p.i12_pit_candidate_id
               AND c.exit_role = :exit_role
               AND c.is_active IS TRUE
              GROUP BY p.i12_pit_candidate_id
            )
            SELECT
              SUM(CASE WHEN row_count = 0 THEN 1 ELSE 0 END) AS missing_count,
              SUM(CASE WHEN row_count > 1 THEN 1 ELSE 0 END) AS duplicate_count
            FROM cost_counts
            """,
            role_params,
        )
        missing = int(row.get("missing_count") or 0)
        duplicate = int(row.get("duplicate_count") or 0)
        by_role[role] = {
            "missing": missing,
            "duplicate": duplicate,
        }
        missing_total += missing
        duplicate_total += duplicate
    return {
        "cost_completeness_by_role": by_role,
        "missing_cost_role_count": missing_total,
        "duplicate_cost_role_count": duplicate_total,
    }


def _feature_summary(
    session: Session,
    dialect: str,
    candidates: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    return _one(
        session,
        f"""
        SELECT
          AVG({_json_number(dialect, 'c', 'observed_cumulative_volume_before_decision')}) AS observed_cumulative_volume_mean,
          AVG({_json_number(dialect, 'c', 'observed_minute_count_before_decision')}) AS observed_minute_count_mean,
          AVG({_json_number(dialect, 'c', 'path_coverage_ratio')}) AS path_coverage_ratio_mean,
          AVG({_json_number(dialect, 'c', 'zero_fill_imputed_minute_count')}) AS zero_fill_imputed_minute_count_mean,
          AVG({_json_number(dialect, 'c', 'zero_fill_imputed_minute_ratio')}) AS zero_fill_imputed_minute_ratio_mean,
          AVG({_json_number(dialect, 'c', 'early_return')}) AS early_return_mean
        FROM {candidates} c
        WHERE c.is_active IS TRUE
          AND c.candidate_status = 'passed'
          AND c.decision_date BETWEEN :start_date AND :end_date
          AND c.decision_time_label = :decision_time
          AND c.path_mode = :path_mode
        """,
        params,
    )


def _paired_sql(left_schema: str, right_schema: str) -> str:
    left = _q(left_schema, "i12_pit_candidates")
    right = _q(right_schema, "i12_pit_candidates")
    return f"""
    WITH l AS (
      SELECT
        ticker,
        decision_date,
        path_mode,
        i12_pit_candidate_id AS lid,
        candidate_status AS lstatus,
        coverage_status AS lcoverage,
        fail_reason AS lfail
      FROM {left}
      WHERE is_active IS TRUE
        AND decision_date BETWEEN :start_date AND :end_date
        AND decision_time_label = :left_decision
        AND path_mode = :path_mode
    ), r AS (
      SELECT
        ticker,
        decision_date,
        path_mode,
        i12_pit_candidate_id AS rid,
        candidate_status AS rstatus,
        coverage_status AS rcoverage,
        fail_reason AS rfail
      FROM {right}
      WHERE is_active IS TRUE
        AND decision_date BETWEEN :start_date AND :end_date
        AND decision_time_label = :right_decision
        AND path_mode = :path_mode
    ), keys AS (
      SELECT ticker, decision_date, path_mode FROM l
      UNION
      SELECT ticker, decision_date, path_mode FROM r
    ), paired AS (
      SELECT
        k.ticker,
        k.decision_date,
        k.path_mode,
        l.lid,
        l.lstatus,
        l.lcoverage,
        l.lfail,
        r.rid,
        r.rstatus,
        r.rcoverage,
        r.rfail
      FROM keys k
      LEFT JOIN l
        ON l.ticker = k.ticker
       AND l.decision_date = k.decision_date
       AND l.path_mode = k.path_mode
      LEFT JOIN r
        ON r.ticker = k.ticker
       AND r.decision_date = k.decision_date
       AND r.path_mode = k.path_mode
    )
    """


def _group_condition(group: str) -> str:
    if group == "left_passed":
        return "lstatus = 'passed' AND lid IS NOT NULL"
    if group == "right_passed":
        return "rstatus = 'passed' AND rid IS NOT NULL"
    if group in {"both", "both_left", "both_right"}:
        return "lstatus = 'passed' AND rstatus = 'passed' AND lid IS NOT NULL AND rid IS NOT NULL"
    if group == "left_only":
        return "lstatus = 'passed' AND lid IS NOT NULL AND COALESCE(rstatus, '') != 'passed'"
    if group == "right_only":
        return "rstatus = 'passed' AND rid IS NOT NULL AND COALESCE(lstatus, '') != 'passed'"
    if group == "left_passed_right_failed":
        return "lstatus = 'passed' AND lid IS NOT NULL AND COALESCE(rstatus, '') != 'passed'"
    if group == "left_failed_right_passed":
        return "rstatus = 'passed' AND rid IS NOT NULL AND COALESCE(lstatus, '') != 'passed'"
    raise ValueError(f"unknown group {group!r}")


def _warnings(
    *,
    left_integrity: Mapping[str, Any],
    right_integrity: Mapping[str, Any],
    overlap: Mapping[str, Any],
    liquidity: Mapping[str, Any],
    require_final: bool,
) -> list[str]:
    warnings: list[str] = [
        "snapshot_compare_economics_displayed_size_diagnostic_only",
        "use_event_tape_for_volume_participation_economics",
    ]
    for side, integrity in (("left", left_integrity), ("right", right_integrity)):
        if integrity.get("missing_days"):
            warnings.append(f"{side}_missing_calendar_days")
        if int(integrity.get("duplicate_active_ticker_date_count") or 0) > 0:
            warnings.append(f"{side}_duplicate_active_ticker_date_rows")
        if int(integrity.get("missing_quote_role_count") or 0) > 0:
            warnings.append(f"{side}_missing_quote_evidence")
        if int(integrity.get("duplicate_quote_role_count") or 0) > 0:
            warnings.append(f"{side}_duplicate_quote_evidence")
        if int(integrity.get("missing_cost_role_count") or 0) > 0:
            warnings.append(f"{side}_missing_cost_evidence")
        if int(integrity.get("duplicate_cost_role_count") or 0) > 0:
            warnings.append(f"{side}_duplicate_cost_evidence")
        if len(integrity.get("available_decision_time_labels") or []) > 1:
            warnings.append(f"{side}_mixed_decision_time_labels_available")
        if len(integrity.get("available_path_modes") or []) > 1:
            warnings.append(f"{side}_mixed_path_modes_available")
    left_rows = int(left_integrity.get("active_candidate_row_count") or 0)
    right_rows = int(right_integrity.get("active_candidate_row_count") or 0)
    if max(left_rows, right_rows) and abs(left_rows - right_rows) / max(left_rows, right_rows) > 0.02:
        warnings.append("source_identity_denominators_differ_materially")
    if (overlap.get("jaccard_overlap") is not None) and float(overlap["jaccard_overlap"]) < 0.20:
        warnings.append("snapshot_passed_overlap_extremely_low")
    spread_delta_mean = _nested(liquidity, "entry_spread_bps_delta", "mean")
    if spread_delta_mean is not None and spread_delta_mean > 0:
        warnings.append("right_snapshot_entry_spread_worse")
    left_non_ok = liquidity.get("left_entry_quote_non_ok_rate")
    right_non_ok = liquidity.get("right_entry_quote_non_ok_rate")
    if left_non_ok is not None and right_non_ok is not None and right_non_ok > left_non_ok:
        warnings.append("right_snapshot_entry_quote_staleness_or_missing_worse")
    return warnings


def _fail_on_duplicate_active_candidates(integrity: Mapping[str, Any], *, side: str) -> None:
    duplicate_count = int(integrity.get("duplicate_active_ticker_date_count") or 0)
    if duplicate_count:
        raise RuntimeError(
            f"{side} snapshot has {duplicate_count} duplicate active ticker/date rows "
            "for the requested decision time and path mode; refusing to compare "
            "because downstream joins would multiply metrics"
        )


def _fail_on_duplicate_child_evidence(integrity: Mapping[str, Any], *, side: str) -> None:
    duplicate_quote_count = int(integrity.get("duplicate_quote_role_count") or 0)
    duplicate_cost_count = int(integrity.get("duplicate_cost_role_count") or 0)
    blockers: list[str] = []
    if duplicate_quote_count:
        blockers.append(f"duplicate_quote_role_count={duplicate_quote_count}")
    if duplicate_cost_count:
        blockers.append(f"duplicate_cost_role_count={duplicate_cost_count}")
    if blockers:
        raise RuntimeError(
            f"{side} snapshot has duplicate active child evidence "
            f"({', '.join(blockers)}); refusing to compare because quote/cost joins "
            "would multiply metrics"
        )


def _load_and_validate_final_reports(
    *,
    require_final: bool,
    left_report: str | Path | None,
    right_report: str | Path | None,
    left_schema: str,
    right_schema: str,
    start_date: date,
    end_date: date,
    decision_time_left: str,
    decision_time_right: str,
    minute_path_mode: str,
) -> dict[str, Any]:
    if not require_final:
        return {"required": False, "checked": False}
    if not left_report or not right_report:
        raise RuntimeError("--require-final requires --left-report and --right-report")
    left = _load_report_json(left_report, side="left")
    right = _load_report_json(right_report, side="right")
    _validate_final_report(
        left,
        side="left",
        expected_schema=left_schema,
        expected_start_date=start_date,
        expected_end_date=end_date,
        expected_decision_time=decision_time_left,
        expected_path_mode=minute_path_mode,
    )
    _validate_final_report(
        right,
        side="right",
        expected_schema=right_schema,
        expected_start_date=start_date,
        expected_end_date=end_date,
        expected_decision_time=decision_time_right,
        expected_path_mode=minute_path_mode,
    )
    left_source = _first_present(left, "source_hur_schema")
    right_source = _first_present(right, "source_hur_schema")
    if left_source != right_source:
        raise RuntimeError(
            "--require-final source_hur_schema mismatch: "
            f"left={left_source!r} right={right_source!r}"
        )
    return {
        "required": True,
        "checked": True,
        "left_report": str(left_report),
        "right_report": str(right_report),
    }


def _load_report_json(path: str | Path, *, side: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except Exception as exc:
        raise RuntimeError(f"{side} final report could not be loaded from {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{side} final report must be a JSON object")
    return payload


def _validate_final_report(
    report: Mapping[str, Any],
    *,
    side: str,
    expected_schema: str,
    expected_start_date: date,
    expected_end_date: date,
    expected_decision_time: str,
    expected_path_mode: str,
) -> None:
    missing_metadata: list[str] = []
    blockers: list[str] = []
    if report.get("conclusions_final") is not True:
        blockers.append("conclusions_final is not true")
    if report.get("data_integrity_passed") is not True:
        blockers.append("data_integrity_passed is not true")
    for key in ("source_replay_complete", "quote_replay_complete", "cost_replay_complete"):
        if key not in report:
            missing_metadata.append(key)
        elif report.get(key) is not True:
            blockers.append(f"{key} is not true")
    for key in (
        "missing_source_attempt_count",
        "extra_source_attempt_count",
        "missing_source_attempt_identity_count",
        "extra_source_attempt_identity_count",
    ):
        if key not in report:
            missing_metadata.append(key)
        elif int(report.get(key) or 0) != 0:
            blockers.append(f"{key}={report.get(key)}")

    schema = _required_report_field(
        report,
        "schema",
        "report_schema",
        "output_schema",
        "target_schema",
        missing=missing_metadata,
    )
    if schema is not None and schema != expected_schema:
        blockers.append(f"schema {schema!r} does not match {expected_schema!r}")

    report_start = _required_report_field(
        report,
        "start_date",
        "report_start_date",
        "progress_source_start_date",
        missing=missing_metadata,
    )
    report_start_date = _parse_report_date(report_start, field="start_date", blockers=blockers)
    if report_start_date is not None and report_start_date > expected_start_date:
        blockers.append(
            f"start_date {report_start!r} does not cover requested {expected_start_date.isoformat()!r}"
        )
    report_end = _required_report_field(
        report,
        "end_date",
        "report_end_date",
        "progress_source_end_date",
        missing=missing_metadata,
    )
    report_end_date = _parse_report_date(report_end, field="end_date", blockers=blockers)
    if report_end_date is not None and report_end_date < expected_end_date:
        blockers.append(
            f"end_date {report_end!r} does not cover requested {expected_end_date.isoformat()!r}"
        )

    path_mode = _required_report_field(
        report,
        "report_path_mode",
        "minute_path_mode",
        missing=missing_metadata,
    )
    if path_mode is not None and path_mode != expected_path_mode:
        blockers.append(f"path_mode {path_mode!r} does not match {expected_path_mode!r}")

    source_hur_schema = _required_report_field(
        report,
        "source_hur_schema",
        missing=missing_metadata,
    )
    if source_hur_schema is not None and not str(source_hur_schema).strip():
        blockers.append("source_hur_schema is empty")

    labels = _required_report_field(
        report,
        "report_decision_time_labels",
        "decision_time_labels",
        missing=missing_metadata,
    )
    if labels is not None:
        if isinstance(labels, str):
            label_set = {labels}
        else:
            label_set = {str(item) for item in labels}
        if expected_decision_time not in label_set:
            blockers.append(
                f"decision_time {expected_decision_time!r} not present in report labels {sorted(label_set)!r}"
            )

    if missing_metadata:
        blockers.append(
            "missing required report metadata "
            f"{sorted(set(missing_metadata))}; regenerate report with current code"
        )

    if blockers:
        raise RuntimeError(f"{side} final report failed --require-final checks: {'; '.join(blockers)}")


def _required_report_field(
    report: Mapping[str, Any],
    *keys: str,
    missing: list[str],
) -> Any:
    value = _first_present(report, *keys)
    if value is None:
        missing.append("/".join(keys))
    return value


def _parse_report_date(value: Any, *, field: str, blockers: list[str]) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        blockers.append(f"{field} {value!r} is not parseable")
        return None


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return None


def _validate_scratch_schema(value: str, *, side: str = "schema") -> str:
    schema = str(value or "").strip()
    if not SCHEMA_RE.fullmatch(schema):
        raise ValueError(f"{side} schema must be a valid SQL identifier")
    if schema in FORBIDDEN_SCHEMAS or not schema.startswith("scratch_"):
        raise ValueError(f"{side} schema must be a named scratch schema")
    return schema


def _validate_decision_time(value: str) -> str:
    if not re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", str(value or "")):
        raise ValueError(f"decision time must be HH:MM, got {value!r}")
    return str(value)


def _parse_date_required(value: str | date, name: str) -> date:
    if value is None:
        raise ValueError(f"{name} is required")
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _q(schema: str, table: str) -> str:
    return f'"{schema}"."{table}"'


def _json_number(dialect: str, alias: str, key: str) -> str:
    if dialect == "postgresql":
        return f"CAST(NULLIF({alias}.feature_json::jsonb ->> '{key}', '') AS DOUBLE PRECISION)"
    return f"CAST(json_extract({alias}.feature_json, '$.{key}') AS FLOAT)"


def _one(session: Session, sql: str, params: Mapping[str, Any]) -> dict[str, Any]:
    rows = _all(session, sql, params)
    return rows[0] if rows else {}


def _all(session: Session, sql: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = session.execute(_sql_text(sql), dict(params))
    return [dict(row._mapping) for row in result]


def _scalar(session: Session, sql: str, params: Mapping[str, Any]) -> Any:
    return session.execute(_sql_text(sql), dict(params)).scalar()


def _list_values(session: Session, sql: str, params: Mapping[str, Any]) -> list[str]:
    return [str(row[0]) for row in session.execute(_sql_text(sql), dict(params)).all() if row[0] is not None]


def _create_session(url: str):
    create_engine, _, session_cls = _sqlalchemy()
    return session_cls(bind=create_engine(url))


def _sql_text(sql: str):
    _, text, _ = _sqlalchemy()
    return text(sql)


def _sqlalchemy():
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import Session
    except ModuleNotFoundError as exc:  # pragma: no cover - DB execution dependency.
        if exc.name == "sqlalchemy":
            venv_python = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
            if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
                os.execv(
                    str(venv_python),
                    [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
                )
        raise
    return create_engine, text, Session


def _distribution(values: Sequence[Any]) -> dict[str, Any]:
    numbers = sorted(_float_or_none(value) for value in values)
    numbers = [value for value in numbers if value is not None]
    if not numbers:
        return {"count": 0, "mean": None, "p50": None, "p90": None}
    return {
        "count": len(numbers),
        "mean": mean(numbers),
        "p50": median(numbers),
        "p90": _percentile(numbers, 0.90),
    }


def _sql_summary(row: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "count": int(row.get(f"{prefix}_count") or 0),
        "mean": _float_or_none(row.get(f"{prefix}_mean")),
        "p50": None,
        "p90": None,
    }


def _delta_summary(rows: Sequence[Mapping[str, Any]], right_key: str, left_key: str) -> dict[str, Any]:
    deltas: list[float] = []
    for row in rows:
        right = _float_or_none(row.get(right_key))
        left = _float_or_none(row.get(left_key))
        if right is not None and left is not None:
            deltas.append(right - left)
    return _distribution(deltas)


def _rate(values: Sequence[bool] | Any) -> float | None:
    observed = list(values)
    if not observed:
        return None
    return sum(1 for value in observed if value) / len(observed)


def _percentile(sorted_numbers: Sequence[float], pct: float) -> float:
    if not sorted_numbers:
        raise ValueError("percentile requires values")
    if len(sorted_numbers) == 1:
        return sorted_numbers[0]
    index = (len(sorted_numbers) - 1) * pct
    lo = int(index)
    hi = min(lo + 1, len(sorted_numbers) - 1)
    weight = index - lo
    return sorted_numbers[lo] * (1 - weight) + sorted_numbers[hi] * weight


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _str_or_none(value: Any) -> str | None:
    return None if value is None else str(value)[:10]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_pct(value: Any) -> str:
    number = _float_or_none(value)
    return "-" if number is None else f"{number * 100:.2f}%"


def _fmt_summary(summary: Mapping[str, Any]) -> str:
    if not summary or not summary.get("count"):
        return "-"
    return (
        f"n={summary.get('count')} mean={_fmt(summary.get('mean'))} "
        f"p50={_fmt(summary.get('p50'))} p90={_fmt(summary.get('p90'))}"
    )


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in str_rows))
        if str_rows
        else len(str(header))
        for index, header in enumerate(headers)
    ]
    lines = [" | ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append("-+-".join("-" * width for width in widths))
    for row in str_rows:
        lines.append(" | ".join(row[index].ljust(widths[index]) for index in range(len(headers))))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-schema", required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-schema", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--decision-time-left", required=True)
    parser.add_argument("--decision-time-right", required=True)
    parser.add_argument(
        "--minute-path-mode",
        required=True,
        choices=["strict_contiguous", "sparse_zero_fill"],
    )
    parser.add_argument("--left-report", help="Final PIT report JSON for the left snapshot; required with --require-final.")
    parser.add_argument("--right-report", help="Final PIT report JSON for the right snapshot; required with --require-final.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--output")
    parser.add_argument("--require-final", action="store_true")
    parser.add_argument("--database-url", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        analysis = compare_snapshots(
            left_schema=args.left_schema,
            left_label=args.left_label,
            right_schema=args.right_schema,
            right_label=args.right_label,
            start_date=args.start_date,
            end_date=args.end_date,
            decision_time_left=args.decision_time_left,
            decision_time_right=args.decision_time_right,
            minute_path_mode=args.minute_path_mode,
            left_report=args.left_report,
            right_report=args.right_report,
            database_url=args.database_url,
            require_final=args.require_final,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    output = (
        json.dumps(analysis, indent=2, sort_keys=True, default=str)
        if args.format == "json"
        else render_text(analysis)
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output + "\n")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
