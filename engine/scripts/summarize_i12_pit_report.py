#!/usr/bin/env python3
"""Summarize PIT-clean I12 report artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXIT_ROLES = ("same_day_exit", "next_open_exit")
SUMMARY_KEYS = ("p50", "p75", "p90")


def summarize_report_paths(
    paths: Sequence[Path | str],
    *,
    labels: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    labels = labels or ()
    for index, path_value in enumerate(paths):
        path = Path(path_value)
        label = labels[index] if index < len(labels) else path.stem
        try:
            report = json.loads(path.read_text())
        except FileNotFoundError:
            summaries.append(_error_summary(label, path, "missing_report_file"))
            continue
        except json.JSONDecodeError as exc:
            summaries.append(_error_summary(label, path, f"invalid_json: {exc}"))
            continue
        if not isinstance(report, dict):
            summaries.append(_error_summary(label, path, "report_root_not_object"))
            continue
        summaries.append(summarize_report(report, label=label, path=path))
    return summaries


def summarize_report(
    report: Mapping[str, Any],
    *,
    label: str,
    path: Path | None = None,
) -> dict[str, Any]:
    exit_metrics = _mapping(report.get("exit_metrics"))
    path_mode_metrics = _mapping(report.get("path_mode_metrics"))
    return {
        "label": label,
        "path": str(path) if path is not None else None,
        "error": None,
        "report_path_mode": report.get("report_path_mode"),
        "report_decision_time_labels": list(report.get("report_decision_time_labels") or []),
        "start_date": report.get("start_date") or report.get("progress_source_start_date"),
        "end_date": report.get("end_date") or report.get("progress_source_end_date"),
        "conclusions_final": report.get("conclusions_final"),
        "training_status": report.get("training_status"),
        "pit_candidate_count": report.get("pit_candidate_count"),
        "active_candidate_row_count": (
            report.get("active_candidate_row_count")
            if report.get("active_candidate_row_count") is not None
            else report.get("actual_candidate_row_count")
        ),
        "actual_candidate_row_count": report.get("actual_candidate_row_count"),
        "quote_replay_complete": report.get("quote_replay_complete"),
        "cost_replay_complete": report.get("cost_replay_complete"),
        "quote_coverage_rate": report.get("quote_coverage_rate"),
        "source_replay": {
            "expected_candidate_attempts": report.get("expected_candidate_attempts"),
            "missing_source_attempt_count": report.get("missing_source_attempt_count"),
            "extra_source_attempt_count": report.get("extra_source_attempt_count"),
            "source_identity_denominator_known": report.get(
                "source_identity_denominator_known"
            ),
            "missing_source_attempt_identity_count": report.get(
                "missing_source_attempt_identity_count"
            ),
            "extra_source_attempt_identity_count": report.get(
                "extra_source_attempt_identity_count"
            ),
        },
        "exits": {
            role: _summarize_exit(exit_metrics.get(role))
            for role in EXIT_ROLES
        },
        "path_mode_metrics": {
            mode: _summarize_path_mode_metrics(metrics)
            for mode, metrics in sorted(path_mode_metrics.items())
            if isinstance(metrics, Mapping)
        },
    }


def render_text_table(summaries: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    report_rows = []
    for summary in summaries:
        report_rows.append([
            summary.get("label"),
            summary.get("report_path_mode"),
            ",".join(summary.get("report_decision_time_labels") or []),
            _fmt(summary.get("conclusions_final")),
            summary.get("training_status"),
            _fmt(summary.get("pit_candidate_count")),
            _fmt(summary.get("active_candidate_row_count")),
            _fmt(summary.get("quote_replay_complete")),
            _fmt(summary.get("cost_replay_complete")),
            _fmt_pct(summary.get("quote_coverage_rate")),
            _fmt(_nested(summary, "source_replay", "missing_source_attempt_count")),
            _fmt(_nested(summary, "source_replay", "extra_source_attempt_count")),
            _fmt(_nested(summary, "source_replay", "missing_source_attempt_identity_count")),
            _fmt(_nested(summary, "source_replay", "extra_source_attempt_identity_count")),
            summary.get("error") or "",
        ])
    lines.append("Reports")
    lines.append(_format_table([
        "label",
        "mode",
        "decision",
        "final",
        "status",
        "pit",
        "rows",
        "quotes",
        "costs",
        "quote_cov",
        "miss",
        "extra",
        "id_miss",
        "id_extra",
        "error",
    ], report_rows))

    exit_rows = []
    for summary in summaries:
        for role in EXIT_ROLES:
            metrics = _mapping(_nested(summary, "exits", role))
            exit_rows.append([
                summary.get("label"),
                role,
                _fmt(metrics.get("candidates")),
                _fmt(metrics.get("tradeable_count")),
                _fmt_pct(metrics.get("tradeable_rate")),
                _fmt(metrics.get("skipped_cash_count")),
                json.dumps(metrics.get("skipped_cash_by_reason") or {}, sort_keys=True),
                _fmt_pct(metrics.get("mean_modeled_return_skips_as_cash")),
                _fmt_pct(metrics.get("win_rate_skips_as_cash")),
                _fmt(metrics.get("spread_bps_p50")),
                _fmt(metrics.get("spread_bps_p75")),
                _fmt(metrics.get("spread_bps_p90")),
                _fmt(metrics.get("executable_notional_p50")),
                _fmt(metrics.get("executable_notional_p75")),
                _fmt(metrics.get("executable_notional_p90")),
            ])
    lines.append("")
    lines.append("Exits")
    lines.append(_format_table([
        "label",
        "exit",
        "cand",
        "trade",
        "trade_rate",
        "skipped",
        "skip_reasons",
        "mean_ret",
        "win",
        "spr_p50",
        "spr_p75",
        "spr_p90",
        "not_p50",
        "not_p75",
        "not_p90",
    ], exit_rows))

    path_rows = []
    for summary in summaries:
        for mode, metrics in _mapping(summary.get("path_mode_metrics")).items():
            path_rows.append([
                summary.get("label"),
                mode,
                metrics.get("training_status"),
                _fmt(metrics.get("conclusions_final")),
                _fmt(metrics.get("candidate_count")),
                _fmt(metrics.get("passed_candidate_count")),
                _fmt(metrics.get("missing_source_attempt_count")),
                _fmt(metrics.get("extra_source_attempt_count")),
                _fmt(metrics.get("missing_source_attempt_identity_count")),
                _fmt(metrics.get("extra_source_attempt_identity_count")),
            ])
    if path_rows:
        lines.append("")
        lines.append("Path Modes")
        lines.append(_format_table([
            "label",
            "mode",
            "status",
            "final",
            "cand",
            "passed",
            "miss",
            "extra",
            "id_miss",
            "id_extra",
        ], path_rows))
    return "\n".join(lines)


def _summarize_exit(metrics_value: Any) -> dict[str, Any]:
    metrics = _mapping(metrics_value)
    spread = _mapping(metrics.get("spread_bps"))
    notional = _mapping(metrics.get("executable_notional"))
    out = {
        "candidates": metrics.get("candidates"),
        "tradeable_count": metrics.get("tradeable_count"),
        "tradeable_rate": metrics.get("tradeable_rate"),
        "skipped_cash_count": metrics.get("skipped_cash_count"),
        "skipped_cash_by_reason": metrics.get("skipped_cash_by_reason") or {},
        "mean_modeled_return_skips_as_cash": metrics.get(
            "mean_modeled_return_skips_as_cash"
        ),
        "win_rate_skips_as_cash": metrics.get("win_rate_skips_as_cash"),
    }
    for key in SUMMARY_KEYS:
        out[f"spread_bps_{key}"] = spread.get(key)
        out[f"executable_notional_{key}"] = notional.get(key)
    return out


def _summarize_path_mode_metrics(metrics_value: Any) -> dict[str, Any]:
    metrics = _mapping(metrics_value)
    return {
        "training_status": metrics.get("training_status"),
        "conclusions_final": metrics.get("conclusions_final"),
        "candidate_count": metrics.get("candidate_count"),
        "passed_candidate_count": metrics.get("passed_candidate_count"),
        "missing_source_attempt_count": metrics.get("missing_source_attempt_count"),
        "extra_source_attempt_count": metrics.get("extra_source_attempt_count"),
        "missing_source_attempt_identity_count": metrics.get(
            "missing_source_attempt_identity_count"
        ),
        "extra_source_attempt_identity_count": metrics.get(
            "extra_source_attempt_identity_count"
        ),
        "exit_metrics": {
            role: _summarize_exit(_mapping(metrics.get("exit_metrics")).get(role))
            for role in EXIT_ROLES
        },
    }


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
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return str(value)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        value = _mapping(value).get(key)
    return value


def _error_summary(label: str, path: Path, error: str) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "error": error,
        "report_path_mode": None,
        "report_decision_time_labels": [],
        "start_date": None,
        "end_date": None,
        "conclusions_final": None,
        "training_status": None,
        "pit_candidate_count": None,
        "active_candidate_row_count": None,
        "actual_candidate_row_count": None,
        "quote_replay_complete": None,
        "cost_replay_complete": None,
        "quote_coverage_rate": None,
        "source_replay": {},
        "exits": {role: _summarize_exit({}) for role in EXIT_ROLES},
        "path_mode_metrics": {},
    }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, help="Report JSON path.")
    parser.add_argument("--label", action="append", default=[], help="Optional label; repeatable.")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    summaries = summarize_report_paths(args.report, labels=args.label)
    if args.format == "json":
        print(json.dumps(summaries, indent=2, sort_keys=True))
    else:
        print(render_text_table(summaries))
    return 1 if any(summary.get("error") for summary in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
