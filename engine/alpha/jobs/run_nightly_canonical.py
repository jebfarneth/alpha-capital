#!/usr/bin/env python3
"""Canonical nightly signal-accumulation orchestrator.

This entrypoint drives the production accumulation path: build the operating
universe, run the per-pattern daily evidence/signal jobs (with frozen
signal_context), then emit a structured health report. It NEVER runs forward
returns and never re-implements firing, detector, or price logic -- it
orchestrates the same proven universe -> daily-assembly flow used by the
scratch rehearsal runner across every wired pattern (M4, M1, ...).

Three modes:

* ``--live``    Accumulate into the canonical PostgreSQL target. Requires
                ``--confirm-canonical-write``. Refuses SQLite, refuses a set
                ``ALPHA_DB_SCHEMA`` (canonical writes go to the default
                search_path, not a scratch schema), verifies the database is at
                the Alembic head, and refuses a future decision date.
* ``--scratch`` Accumulate into an isolated PostgreSQL scratch schema. Requires
                ``--schema`` and creates the schema/tables from metadata.
* ``--dry-run`` Resolve and print the run plan only. Performs zero database
                writes and zero provider API calls.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
import re
import statistics
import subprocess
import sys
import urllib.parse
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from alpha.db.engine import reset_globals, schema_connect_args
from alpha.db.models import (
    CanonicalUniverseScan,
    DataLineage,
    EvidenceJob,
    EvidenceJobRun,
    FeatureSnapshot,
    ForwardReturnObservation,
    SignalRegistry,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.jobs import run_forward_context, run_m1_daily, run_m4_daily, run_universe
from alpha.market_calendar import (
    is_us_equity_session,
    resolve_us_equity_session,
    us_equity_session_close_timestamp,
)
from alpha.runtime_env import load_runtime_env

REQUIRED_ENV = ("DATABASE_URL", "FMP_API_KEY", "POLYGON_API_KEY", "BENZINGA_API_KEY")
SECRET_ENV_KEYS = ("FMP_API_KEY", "POLYGON_API_KEY", "BENZINGA_API_KEY")
SUCCESS_RUN_STATUSES = {"finished"}
SECRET_KEY_NAMES = (
    "apikey",
    "api_key",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "bearer",
    "secret",
    "password",
    "client_secret",
    "private_key",
)
SECRET_VALUE_PATTERNS = (
    re.compile(r"(api_?key|access_token|token|authorization)\s*=\s*\S+", re.IGNORECASE),
    re.compile(r"Authorization\s*:\s*Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}\b", re.IGNORECASE),
)
HARD_FIRED_SOURCE_STATUSES = (
    "provider_error",
    "parse_error",
    "validation_error",
    "unavailable",
)
SOURCE_STATUS_SPLIT_KEYS = (
    "provider_error",
    "parse_error",
    "validation_error",
    "unavailable",
    "pit_excluded",
    "no_data",
)
FORWARD_CONTEXT_REQUIRED_PROVIDERS = ("polygon", "benzinga")
FORWARD_CONTEXT_USABLE_PROVIDER_STATUSES = {"matched", "no_data", "pit_excluded"}
FORWARD_CONTEXT_HARD_PROVIDER_FAILURE_STATUSES = {
    "provider_error",
    "parse_error",
    "validation_error",
    "unavailable",
}


@dataclass(frozen=True)
class RunInvocation:
    exit_code: int
    run_id: Optional[str] = None
    run_status: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# environment / target helpers
# ---------------------------------------------------------------------------
def _url_metadata(url: str) -> Dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""
    if parsed.scheme.startswith("sqlite"):
        host_class = "sqlite"
    elif hostname.endswith(".supabase.co") and hostname.startswith("db."):
        host_class = "direct_supabase"
    elif "pooler" in hostname:
        host_class = "pooler"
    elif hostname in {"localhost", "127.0.0.1", "::1"}:
        host_class = "localhost"
    else:
        host_class = "postgres_other"
    return {
        "scheme": parsed.scheme,
        "hostname": hostname,
        "port": parsed.port,
        "database": parsed.path.lstrip("/") or None,
        "host_class": host_class,
    }


def _parse_timestamp(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("run_timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _app_commit_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip()
    return sha or None


# ---------------------------------------------------------------------------
# clock resolution + validation
# ---------------------------------------------------------------------------
class CanonicalRunError(ValueError):
    """Raised when a guard refuses to proceed with a canonical run."""


class NonTradingDayNoOp(Exception):
    """Signals a default-clock run resolved to a non-trading day; clean no-op exit."""


def resolve_canonical_clock(
    run_timestamp: datetime,
    decision_date_override: Optional[str],
) -> Dict[str, str]:
    """Resolve the canonical decision/evidence/execution sessions.

    When ``decision_date_override`` is supplied it must be a regular U.S.
    equity session that is not in the future relative to the run timestamp's
    own decision date. Evidence/execution sessions are then anchored on that
    session's close so downstream M4 resolution stays consistent.
    """
    base = resolve_us_equity_session(run_timestamp)
    if not decision_date_override:
        if not is_us_equity_session(date.fromisoformat(base.decision_date)):
            raise NonTradingDayNoOp(
                f"run_timestamp resolves to non-trading decision date "
                f"{base.decision_date}; next session is {base.next_execution_session}"
            )
        return {
            "decision_date": base.decision_date,
            "evidence_session_date": base.evidence_session_date,
            "next_execution_session": base.next_execution_session,
            "effective_run_timestamp": run_timestamp.isoformat(),
        }

    try:
        chosen = date.fromisoformat(decision_date_override)
    except ValueError as exc:
        raise CanonicalRunError(
            f"--decision-date {decision_date_override!r} is not a valid ISO date"
        ) from exc
    if not is_us_equity_session(chosen):
        raise CanonicalRunError(
            f"--decision-date {decision_date_override} is not a regular "
            "U.S. equity trading session (weekend or holiday)"
        )
    if chosen.isoformat() > base.decision_date:
        raise CanonicalRunError(
            f"--decision-date {decision_date_override} is in the future "
            f"relative to the run-timestamp decision date {base.decision_date}"
        )
    anchored_ts = us_equity_session_close_timestamp(chosen)
    anchored = resolve_us_equity_session(anchored_ts)
    return {
        "decision_date": anchored.decision_date,
        "evidence_session_date": anchored.evidence_session_date,
        "next_execution_session": anchored.next_execution_session,
        "effective_run_timestamp": anchored_ts.isoformat(),
    }


def require_canonical_target(url: str) -> None:
    """Guard the canonical (--live) write target before any DB/API work."""
    if not url:
        raise CanonicalRunError("DATABASE_URL is required for canonical writes")
    if not url.startswith("postgresql"):
        raise CanonicalRunError(
            "canonical writes require a PostgreSQL DATABASE_URL; SQLite is refused"
        )
    if os.environ.get("ALPHA_DB_SCHEMA"):
        raise CanonicalRunError(
            "ALPHA_DB_SCHEMA must not be set for canonical writes; canonical "
            "accumulation targets the default search_path, not a scratch schema"
        )


def require_scratch_schema(schema: Optional[str], url: str) -> None:
    if not schema:
        raise CanonicalRunError("--scratch requires --schema")
    if schema.strip().casefold() == "public":
        raise CanonicalRunError("--scratch refuses the public schema")
    if url and url.startswith("sqlite"):
        raise CanonicalRunError("--scratch requires a PostgreSQL DATABASE_URL")


def verify_alembic_head(url: str) -> Dict[str, Any]:
    """Confirm the canonical database is migrated to the Alembic head."""
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    config = Config("alembic.ini")
    script = ScriptDirectory.from_config(config)
    heads = set(script.get_heads())
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current = set(context.get_current_heads())
    finally:
        engine.dispose()
    at_head = current == heads and bool(heads)
    if not at_head:
        raise CanonicalRunError(
            f"database is not at the Alembic head (current={sorted(current)}, "
            f"heads={sorted(heads)}); run migrations before canonical accumulation"
        )
    return {"current": sorted(current), "heads": sorted(heads)}


# ---------------------------------------------------------------------------
# health report builder (independently testable against any session)
# ---------------------------------------------------------------------------
def _safe_json(value: Optional[str]) -> Dict[str, Any]:
    try:
        payload = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _date_bounds(day: str) -> tuple[datetime, datetime]:
    parsed = date.fromisoformat(day)
    start = datetime.combine(parsed, time.min, timezone.utc)
    return start, start + timedelta(days=1)


def _json_list(value: Optional[str]) -> List[str]:
    try:
        payload = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if item is not None]


def _nested_get(payload: Dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_value(payload: Dict[str, Any], paths: Iterable[tuple[str, ...]]) -> Any:
    for path in paths:
        value = _nested_get(payload, *path)
        if value is not None:
            return value
    return None


def _iter_source_attempts(context: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for category in context.values():
        if not isinstance(category, dict):
            continue
        for attempt in category.get("source_attempts", []) or []:
            if isinstance(attempt, dict):
                yield attempt


def _source_attempt_status_counts(contexts: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for context in contexts:
        for attempt in _iter_source_attempts(context):
            status = attempt.get("status")
            if status:
                counts[str(status)] += 1
    return dict(sorted(counts.items()))


def _attempt_status_summary(context: Dict[str, Any]) -> Dict[str, int]:
    return _source_attempt_status_counts([context])


def _normalize_secret_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


NORMALIZED_SECRET_KEYS = {
    _normalize_secret_key(value) for value in SECRET_KEY_NAMES
}


def _env_secret_values() -> List[str]:
    return [
        value
        for key in SECRET_ENV_KEYS
        for value in [os.environ.get(key)]
        if value and len(value) >= 16
    ]


def _nonnegative_int_metric(value: Any) -> int:
    """Coerce a telemetry count to a non-negative int without crashing.

    Defensive only: the M4 producer emits ints today. Guards report
    generation against a future serialization quirk that could put a
    numeric string or junk into context_persistence_mismatch_count.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float):
        return int(value) if value > 0 else 0
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except (TypeError, ValueError):
            return 0
        return parsed if parsed > 0 else 0
    return 0


def _forward_context_panel_capture_ok(metrics: Optional[Dict[str, Any]]) -> bool:
    if metrics is None:
        return True
    if metrics.get("no_op_reason"):
        return True
    eligible = _nonnegative_int_metric(metrics.get("eligible_signal_count"))
    if eligible == 0:
        return True
    if _forward_context_metrics_show_dead_required_provider(metrics):
        return False
    captured = (
        _nonnegative_int_metric(metrics.get("rows_inserted"))
        + _nonnegative_int_metric(metrics.get("rows_existing"))
    )
    return captured >= eligible


def _forward_context_metrics_show_dead_required_provider(
    metrics: Dict[str, Any],
) -> bool:
    if _nonnegative_int_metric(metrics.get("degraded_signal_count")) > 0:
        return True
    provider_counts = metrics.get("required_provider_status_counts")
    if isinstance(provider_counts, dict):
        for provider in FORWARD_CONTEXT_REQUIRED_PROVIDERS:
            counts = _status_count_map(provider_counts.get(provider))
            if _forward_context_provider_dead(counts):
                return True
        return False

    counts = _status_count_map(metrics.get("source_attempt_status_counts"))
    return _forward_context_provider_dead(counts)


def _status_count_map(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(status): _nonnegative_int_metric(count)
        for status, count in value.items()
    }


def _forward_context_provider_dead(counts: Dict[str, int]) -> bool:
    if not counts:
        return False
    usable = sum(
        counts.get(status, 0)
        for status in FORWARD_CONTEXT_USABLE_PROVIDER_STATUSES
    )
    if usable > 0:
        return False
    hard = sum(
        counts.get(status, 0)
        for status in FORWARD_CONTEXT_HARD_PROVIDER_FAILURE_STATUSES
    )
    total = sum(counts.values())
    return hard > 0 and hard == total


def _scan_json_for_secrets(value: Any, env_values: List[str]) -> Dict[str, bool]:
    hits = {
        "secret_key": False,
        "secret_value": False,
        "secret_url_fragment": False,
        "raw_payload": False,
        "api_key": False,
        "token": False,
    }

    def merge(child: Dict[str, bool]) -> None:
        for key, present in child.items():
            hits[key] = hits[key] or present

    if isinstance(value, dict):
        for key, item in value.items():
            normalized = _normalize_secret_key(str(key))
            if normalized == "rawpayload":
                # Load-bearing guard: feature_json may carry full diagnostics, so
                # raw provider payload objects must fail health wherever they appear.
                hits["raw_payload"] = True
            if normalized in NORMALIZED_SECRET_KEYS:
                hits["secret_key"] = True
            if normalized in {"apikey", "apiKey".lower()}:
                hits["api_key"] = True
            if normalized in {"token", "accesstoken", "refreshtoken"}:
                hits["token"] = True
            merge(_scan_json_for_secrets(item, env_values))
    elif isinstance(value, list):
        for item in value:
            merge(_scan_json_for_secrets(item, env_values))
    elif isinstance(value, str):
        if any(pattern.search(value) for pattern in SECRET_VALUE_PATTERNS):
            hits["secret_url_fragment"] = True
        if any(secret in value for secret in env_values):
            hits["secret_value"] = True
    return hits


def _feature_json_secret_scan(feature_jsons: List[str]) -> Dict[str, int]:
    env_values = _env_secret_values()
    counts: Counter[str] = Counter()
    for blob in feature_jsons:
        try:
            payload = json.loads(blob or "{}")
        except (TypeError, ValueError):
            payload = blob or ""
        hits = _scan_json_for_secrets(payload, env_values)
        if hits["raw_payload"]:
            counts["raw_payload"] += 1
        if hits["secret_key"]:
            counts["secret_key"] += 1
        if hits["secret_value"]:
            counts["secret_value"] += 1
        if hits["secret_url_fragment"]:
            counts["secret_url_fragment"] += 1
        if hits["api_key"]:
            counts["api_key"] += 1
        if hits["token"]:
            counts["token"] += 1
    return {
        "raw_payload": counts["raw_payload"],
        "secret_key": counts["secret_key"],
        "secret_value": counts["secret_value"],
        "secret_url_fragment": counts["secret_url_fragment"],
        "api_key": counts["api_key"],
        "token": counts["token"],
    }


def _run_matches_decision(row: EvidenceJobRun, decision_date: str) -> bool:
    metrics = _safe_json(row.metric_json)
    params = _safe_json(row.params_json)
    metric_decision_date = metrics.get("decision_date")
    if metric_decision_date is not None:
        return metric_decision_date == decision_date
    return params.get("trading_date") == decision_date


def _latest_job_run_for_decision(
    session,
    *,
    job_name: str,
    decision_date: str,
    success_only: bool = False,
) -> Optional[EvidenceJobRun]:
    rows = (
        session.query(EvidenceJobRun)
        .join(EvidenceJob, EvidenceJob.job_id == EvidenceJobRun.job_id)
        .filter(EvidenceJob.job_name == job_name)
        .order_by(EvidenceJobRun.ended_at.desc().nullslast(), EvidenceJobRun.started_at.desc())
        .all()
    )
    for row in rows:
        if success_only and row.run_status not in SUCCESS_RUN_STATUSES:
            continue
        if _run_matches_decision(row, decision_date):
            return row
    return None


def _latest_m4_run_metrics(session, decision_date: str) -> Dict[str, Any]:
    row = _latest_job_run_for_decision(
        session,
        job_name="m4_daily_feature_assembly",
        decision_date=decision_date,
        success_only=True,
    )
    return _safe_json(row.metric_json) if row is not None else {}


def _m4_run_ids_for_decision(session, decision_date: str) -> List[str]:
    rows = (
        session.query(EvidenceJobRun)
        .join(EvidenceJob, EvidenceJob.job_id == EvidenceJobRun.job_id)
        .filter(EvidenceJob.job_name == "m4_daily_feature_assembly")
        .order_by(EvidenceJobRun.started_at.asc())
        .all()
    )
    run_ids: List[str] = []
    for row in rows:
        if row.run_status not in SUCCESS_RUN_STATUSES:
            continue
        metrics = _safe_json(row.metric_json)
        if metrics.get("decision_date") == decision_date:
            run_ids.append(row.job_run_id)
    return run_ids


def _m4_run_diagnostics(
    session,
    *,
    decision_date: str,
    current_m4_run_ids: Iterable[str],
) -> Dict[str, Any]:
    current = set(current_m4_run_ids)
    rows = (
        session.query(EvidenceJobRun)
        .join(EvidenceJob, EvidenceJob.job_id == EvidenceJobRun.job_id)
        .filter(EvidenceJob.job_name == "m4_daily_feature_assembly")
        .order_by(EvidenceJobRun.ended_at.desc().nullslast(), EvidenceJobRun.started_at.desc())
        .all()
    )
    failed: List[EvidenceJobRun] = []
    for row in rows:
        if row.job_run_id in current:
            continue
        metrics = _safe_json(row.metric_json)
        if metrics.get("decision_date") == decision_date and row.run_status not in SUCCESS_RUN_STATUSES:
            failed.append(row)
    return {
        "failed_same_date_m4_run_count": len(failed),
        "latest_failed_same_date_m4_run_id": failed[0].job_run_id if failed else None,
        "failed_same_date_run_ids": [row.job_run_id for row in failed[:20]],
    }


def _universe_scan_metrics(scan: UniverseScan, source: str) -> Dict[str, Any]:
    metrics = _safe_json(scan.metric_json)
    if not metrics:
        metrics = {
            "raw_count": scan.raw_count,
            "deduped_count": scan.deduped_count,
            "included": scan.included_count,
            "excluded": scan.excluded_count,
            "duplicate_symbol_count": scan.duplicate_symbol_count,
        }
    metrics = dict(metrics)
    metrics["universe_metrics_source"] = source
    return metrics


def _latest_universe_metrics(session, decision_date: str) -> Dict[str, Any]:
    canonical = (
        session.query(CanonicalUniverseScan)
        .filter(CanonicalUniverseScan.trading_date == decision_date)
        .first()
    )
    if canonical is not None:
        scan = session.get(UniverseScan, canonical.scan_id)
        if scan is not None:
            return _universe_scan_metrics(scan, "canonical_pointer")

    scan = (
        session.query(UniverseScan)
        .filter(UniverseScan.trading_date == decision_date)
        .order_by(UniverseScan.asof_timestamp.desc())
        .first()
    )
    if scan is None:
        return {}
    return _universe_scan_metrics(scan, "latest_fallback")


def _current_feature_snapshots(
    session,
    *,
    decision_date: str,
    evidence_session_date: str,
    next_execution_session: str,
    current_m4_run_ids: List[str],
    fired_feature_ids: set[str],
    allow_window_fallback: bool = True,
) -> List[FeatureSnapshot]:
    features_by_id: Dict[str, FeatureSnapshot] = {}
    if current_m4_run_ids:
        # M4DailyAssemblyJob currently runs DetectorOrchestrationJob under the
        # same evidence ctx, so tests prove FeatureSnapshot.job_run_id equals the
        # M4 assembly EvidenceJobRun id today. If orchestration is ever wrapped
        # in its own run_job, this join must be updated or non-fired context
        # candidates will be silently dropped.
        for feature in (
            session.query(FeatureSnapshot)
            .filter(
                FeatureSnapshot.pattern_id == "M4",
                FeatureSnapshot.job_run_id.in_(current_m4_run_ids),
            )
            .all()
        ):
            features_by_id[feature.feature_snapshot_id] = feature
    elif allow_window_fallback:
        start, end = _date_bounds(evidence_session_date)
        for feature in (
            session.query(FeatureSnapshot)
            .filter(
                FeatureSnapshot.pattern_id == "M4",
                FeatureSnapshot.asof_timestamp >= start,
                FeatureSnapshot.asof_timestamp < end,
            )
            .all()
        ):
            payload = _safe_json(feature.feature_json)
            if (
                payload.get("trading_date") in (None, decision_date)
                or payload.get("next_execution_session") == next_execution_session
            ):
                features_by_id[feature.feature_snapshot_id] = feature
    if fired_feature_ids:
        for feature in (
            session.query(FeatureSnapshot)
            .filter(
                FeatureSnapshot.pattern_id == "M4",
                FeatureSnapshot.feature_snapshot_id.in_(fired_feature_ids),
            )
            .all()
        ):
            features_by_id[feature.feature_snapshot_id] = feature
    return list(features_by_id.values())


def _current_data_lineage_count(session, features: List[FeatureSnapshot], signals: List[SignalRegistry]) -> int:
    lineage_ids: set[str] = set()
    for feature in features:
        lineage_ids.update(_json_list(feature.data_lineage_ids))
    for signal in signals:
        lineage_ids.update(_json_list(signal.data_lineage_ids))
    if not lineage_ids:
        return 0
    return (
        session.query(DataLineage)
        .filter(DataLineage.data_lineage_id.in_(sorted(lineage_ids)))
        .count()
    )


def _duplicate_signal_count(signals: List[SignalRegistry]) -> int:
    counts: Counter[str] = Counter(
        signal.signal_identity_hash
        for signal in signals
        if signal.signal_identity_hash
    )
    return sum(1 for value in counts.values() if value > 1)


def _category_status(context: Dict[str, Any], key: str) -> Optional[str]:
    value = context.get(key)
    return value.get("status") if isinstance(value, dict) else None


def _calendar_subcategory_status(calendar: Dict[str, Any], key: str) -> Optional[str]:
    value = calendar.get(key)
    if not isinstance(value, dict):
        return None
    if value.get("status"):
        return value.get("status")
    if (value.get("row_count") or 0) > 0:
        return "matched"
    if (value.get("pit_excluded_count") or 0) > 0:
        return "pit_excluded"
    if value:
        return "no_data"
    return None


def _source_attempt_lineage_values(context: Dict[str, Any], key: str) -> set[str]:
    values: set[str] = set()
    for attempt in _iter_source_attempts(context):
        value = attempt.get(key)
        if value:
            values.add(str(value))
    return values


def _fired_context_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    context = payload.get("signal_context")
    if not isinstance(context, dict):
        return {}
    calendar = context.get("benzinga_calendar")
    if not isinstance(calendar, dict):
        calendar = {}
    return {
        "X_M4": payload.get("X_M4"),
        "price": payload.get("price") or payload.get("P_close"),
        "high_52w": payload.get("high_52w") or payload.get("H_52w"),
        "short_interest": {
            "status": _category_status(context, "polygon_short_interest"),
            "short_interest": _first_value(context, (
                ("polygon_short_interest", "short_interest"),
                ("polygon_short_interest", "latest", "short_interest"),
            )),
            "days_to_cover": _first_value(context, (
                ("polygon_short_interest", "days_to_cover"),
                ("polygon_short_interest", "latest", "days_to_cover"),
            )),
        },
        "short_volume": {
            "status": _category_status(context, "polygon_short_volume"),
            "short_volume_ratio": _first_value(context, (
                ("polygon_short_volume", "short_volume_ratio"),
                ("polygon_short_volume", "latest", "short_volume_ratio"),
            )),
        },
        "polygon_news": {
            "status": _category_status(context, "polygon_news"),
            "article_count_90d": _nested_get(context, "polygon_news", "article_count_90d"),
        },
        "benzinga_news": {
            "status": _category_status(context, "benzinga_news"),
            "article_count_7d": _nested_get(context, "benzinga_news", "article_count_7d"),
            "wiim_count_7d": _nested_get(context, "benzinga_news", "wiim_count_7d"),
        },
        "insider": {
            "status": _category_status(context, "benzinga_insider"),
            "net_discretionary_shares": _nested_get(
                context, "benzinga_insider", "net_discretionary_shares"
            ),
            "net_discretionary_value": _nested_get(
                context, "benzinga_insider", "net_discretionary_value"
            ),
        },
        "calendar_status": {
            "earnings": _calendar_subcategory_status(calendar, "earnings"),
            "ratings": _calendar_subcategory_status(calendar, "ratings"),
            "offerings": _calendar_subcategory_status(calendar, "offerings"),
            "dividends": _calendar_subcategory_status(calendar, "dividends"),
        },
        "benzinga_ma": {
            "status": _category_status(context, "benzinga_ma"),
        },
        "source_attempt_status_counts": _attempt_status_summary(context),
        "source_attempt_lineage_id_count": len(
            _source_attempt_lineage_values(context, "lineage_id")
        ),
        "source_attempt_lineage_hash_count": len(
            _source_attempt_lineage_values(context, "lineage_hash")
            | _source_attempt_lineage_values(context, "raw_payload_hash")
        ),
    }


def build_m4_health_report(
    session,
    *,
    mode: str,
    schema: Optional[str],
    host_class: str,
    app_commit_sha: Optional[str],
    decision_date: str,
    evidence_session_date: str,
    next_execution_session: str,
    run_timestamp: str,
    universe_metrics: Optional[Dict[str, Any]] = None,
    m4_metrics: Optional[Dict[str, Any]] = None,
    primary_m4_run_id: Optional[str] = None,
    rerun_m4_run_id: Optional[str] = None,
    rerun_m4_metrics: Optional[Dict[str, Any]] = None,
    universe_run_id: Optional[str] = None,
    m1_run_id: Optional[str] = None,
    m1_metrics: Optional[Dict[str, Any]] = None,
    forward_context_run_id: Optional[str] = None,
    forward_context_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the structured M4 accumulation health report (sections a-j)."""

    universe_metrics = dict(
        universe_metrics
        if universe_metrics is not None
        else _latest_universe_metrics(session, decision_date)
    )
    if universe_run_id and "universe_metrics_source" not in universe_metrics:
        universe_metrics["universe_metrics_source"] = "current_invocation"
    m4_metrics = (
        m4_metrics
        if m4_metrics is not None
        else _latest_m4_run_metrics(session, decision_date)
    )
    m1_metrics = dict(m1_metrics or {})
    explicit_m4_run_ids = [
        run_id for run_id in (primary_m4_run_id, rerun_m4_run_id) if run_id
    ]
    current_m4_run_ids = explicit_m4_run_ids or _m4_run_ids_for_decision(
        session, decision_date
    )
    m4_run_diagnostics = _m4_run_diagnostics(
        session,
        decision_date=decision_date,
        current_m4_run_ids=current_m4_run_ids,
    )
    signals = (
        session.query(SignalRegistry)
        .filter(
            SignalRegistry.pattern_id == "M4",
            SignalRegistry.trading_date == decision_date,
        )
        .all()
    )
    fired_feature_ids = {
        signal.feature_snapshot_id for signal in signals if signal.feature_snapshot_id
    }
    features = _current_feature_snapshots(
        session,
        decision_date=decision_date,
        evidence_session_date=evidence_session_date,
        next_execution_session=next_execution_session,
        current_m4_run_ids=current_m4_run_ids,
        fired_feature_ids=fired_feature_ids,
        # Normal CLI calls thread guarded child run ids into the report, so they
        # never rely on this fallback. It exists for direct/manual report builds
        # without run ids, where stale same-window features may be included with
        # lost run linkage.
        allow_window_fallback=(
            not current_m4_run_ids
            and m4_run_diagnostics["failed_same_date_m4_run_count"] == 0
        ),
    )
    feature_by_id = {f.feature_snapshot_id: f for f in features}
    feature_jsons = [f.feature_json or "" for f in features]
    payloads = {f.feature_snapshot_id: _safe_json(f.feature_json) for f in features}
    sizes = [len(blob) for blob in feature_jsons]
    context_payloads = [
        payload.get("signal_context")
        for payload in payloads.values()
        if isinstance(payload.get("signal_context"), dict)
    ]

    features_with_context = sum(
        1
        for payload in payloads.values()
        if isinstance(payload.get("signal_context"), dict)
    )

    fired_missing_context = 0
    next_execution_missing = 0
    fired_feature_run_id_mismatch_count = 0
    fired_feature_run_id_mismatches: List[str] = []
    fired_rows: List[Dict[str, Any]] = []
    for signal in signals:
        payload = payloads.get(signal.feature_snapshot_id, {})
        feature = feature_by_id.get(signal.feature_snapshot_id)
        context = payload.get("signal_context")
        has_context = isinstance(context, dict)
        if not has_context:
            fired_missing_context += 1
        if not signal.next_execution_session:
            next_execution_missing += 1
        if current_m4_run_ids and (
            feature is None or feature.job_run_id not in current_m4_run_ids
        ):
            fired_feature_run_id_mismatch_count += 1
            if signal.feature_snapshot_id:
                fired_feature_run_id_mismatches.append(signal.feature_snapshot_id)
        row = {
            "ticker": signal.ticker,
            "direction": signal.direction,
            "signal_identity_hash": (signal.signal_identity_hash or "")[:12],
            "trading_date": signal.trading_date,
            "next_execution_session": signal.next_execution_session,
            "raw_signal_strength": signal.raw_signal_strength,
            "has_signal_context": has_context,
            "fidelity_tier": signal.fidelity_tier,
        }
        row.update(_fired_context_summary(payload))
        feature_lineage_ids = set(_json_list(feature.data_lineage_ids)) if feature else set()
        source_lineage_ids = (
            _source_attempt_lineage_values(context, "lineage_id")
            if isinstance(context, dict)
            else set()
        )
        row["context_lineage_id_count"] = len(feature_lineage_ids | source_lineage_ids)
        row["context_lineage_hash_count"] = row.get("source_attempt_lineage_hash_count", 0)
        fired_rows.append(row)
    fired_rows.sort(key=lambda row: row["ticker"])

    secret_scan = _feature_json_secret_scan(feature_jsons)
    raw_payload_hits = secret_scan["raw_payload"]
    secret_key_hits = secret_scan["secret_key"]
    secret_value_hits = secret_scan["secret_value"]
    secret_url_fragment_hits = secret_scan["secret_url_fragment"]
    api_key_hits = secret_scan["api_key"]
    token_hits = secret_scan["token"]
    secret_total = secret_key_hits + secret_value_hits + secret_url_fragment_hits
    primary_mismatch = _nonnegative_int_metric(
        (m4_metrics.get("signal_context") or {}).get(
            "context_persistence_mismatch_count"
        )
    )
    rerun_mismatch = _nonnegative_int_metric(
        (rerun_m4_metrics or {}).get("signal_context", {}).get(
            "context_persistence_mismatch_count"
        )
    )
    source_attempt_status_counts = _source_attempt_status_counts(context_payloads)
    fired_context_payloads = [
        payloads[feature_id].get("signal_context")
        for feature_id in fired_feature_ids
        if isinstance(payloads.get(feature_id, {}).get("signal_context"), dict)
    ]
    non_fired_context_payloads = [
        payload.get("signal_context")
        for feature_id, payload in payloads.items()
        if feature_id not in fired_feature_ids
        and isinstance(payload.get("signal_context"), dict)
    ]
    fired_status_counts = _source_attempt_status_counts(fired_context_payloads)
    non_fired_status_counts = _source_attempt_status_counts(non_fired_context_payloads)
    provider_error_count = source_attempt_status_counts.get("provider_error", 0)
    fired_hard_error_count = sum(
        fired_status_counts.get(status, 0) for status in HARD_FIRED_SOURCE_STATUSES
    )
    # Source errors on non-fired candidates are warning-only, but raw_payload_hits
    # is a global leak guard and fails health for fired and non-fired features alike.
    source_status_split = {
        status: {
            "fired": fired_status_counts.get(status, 0),
            "non_fired": non_fired_status_counts.get(status, 0),
        }
        for status in SOURCE_STATUS_SPLIT_KEYS
    }

    # The canonical runner never computes forward returns. A freshly persisted
    # signal carries forward_return_status="pending" as its un-run baseline, so
    # that state is NOT evidence of a forward-return run. Only a populated
    # forward_return, a recorded attempt, a non-pending status, or an
    # observation row indicates forward returns were (wrongly) computed here.
    current_signal_ids = [signal.signal_id for signal in signals]
    if current_signal_ids:
        forward_observation_count = (
            session.query(ForwardReturnObservation)
            .filter(ForwardReturnObservation.signal_id.in_(current_signal_ids))
            .count()
        )
    else:
        forward_observation_count = 0
    signals_with_forward_return = sum(
        1
        for signal in signals
        if signal.forward_return is not None
        or (signal.forward_return_attempts or 0) > 0
        or (signal.forward_return_status not in (None, "pending"))
    )
    forward_rows_created = forward_observation_count + signals_with_forward_return

    signal_count = len(signals)
    duplicate_signal_count = _duplicate_signal_count(signals)
    has_successful_m4_run = bool(current_m4_run_ids)
    forward_context_capture_ok = _forward_context_panel_capture_ok(
        forward_context_metrics
    )

    verdict_checks = {
        "has_signals": signal_count > 0,
        "has_successful_m4_run": has_successful_m4_run,
        "no_fired_missing_signal_context": fired_missing_context == 0,
        "no_secret_leaks": secret_total == 0 and raw_payload_hits == 0,
        "no_missing_next_execution_session": next_execution_missing == 0,
        "no_forward_return_rows": forward_rows_created == 0,
        "no_fired_signal_context_errors": fired_hard_error_count == 0,
        "no_duplicate_signals": duplicate_signal_count == 0,
        "no_persistence_mismatch": (primary_mismatch + rerun_mismatch) == 0,
        "forward_context_panel_captured": forward_context_capture_ok,
    }
    failing = sorted(name for name, ok in verdict_checks.items() if not ok)

    report = {
        "schema_version": "m4_canonical_health.v1",
        "run_metadata": {
            "mode": mode,
            "schema": schema,
            "host_class": host_class,
            "app_commit_sha": app_commit_sha,
            "run_timestamp": run_timestamp,
            "asof_timestamp": us_equity_session_close_timestamp(
                date.fromisoformat(evidence_session_date)
            ).isoformat(),
            "decision_date": decision_date,
            "evidence_session_date": evidence_session_date,
            "next_execution_session": next_execution_session,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "universe": {
            "universe_scans": (
                session.query(UniverseScan)
                .filter(UniverseScan.trading_date == decision_date)
                .count()
            ),
            "canonical_universe_scans": (
                session.query(CanonicalUniverseScan)
                .filter(CanonicalUniverseScan.trading_date == decision_date)
                .count()
            ),
            "universe_snapshots": (
                session.query(UniverseSnapshot)
                .join(UniverseScan, UniverseSnapshot.scan_id == UniverseScan.scan_id)
                .filter(UniverseScan.trading_date == decision_date)
                .count()
            ),
            "data_lineage_rows": _current_data_lineage_count(session, features, signals),
            "universe_run_id": universe_run_id,
            "universe_metrics_source": universe_metrics.get("universe_metrics_source"),
            "raw_unique_count": universe_metrics.get("raw_count"),
            "deduped_count": universe_metrics.get("deduped_count"),
            "excluded_count": universe_metrics.get("excluded"),
            "identity_coverage": universe_metrics.get("identity_coverage_ratio")
            or universe_metrics.get("security_profile_coverage_ratio"),
            "included_universe_size": universe_metrics.get("included"),
        },
        "m4_assembly": {
            "primary_m4_run_id": primary_m4_run_id,
            "rerun_m4_run_id": rerun_m4_run_id,
            "included_universe_size": m4_metrics.get("included_universe_size"),
            "assembled_count": (m4_metrics.get("assembly") or {}).get("assembled_count"),
            "fetched_symbol_count": m4_metrics.get("fetched_symbol_count"),
            "fetched_bar_count": m4_metrics.get("fetched_bar_count"),
            "fetch_error_count": m4_metrics.get("fetch_error_count"),
            "m4_feature_snapshots": len(features),
        },
        "m1_assembly": {
            "m1_run_id": m1_run_id,
            "no_op_reason": m1_metrics.get("no_op_reason"),
            "included_universe_size": m1_metrics.get("included_universe_size"),
            "announcing_universe_event_count": m1_metrics.get(
                "announcing_universe_event_count"
            ),
            "foster_computed_count": m1_metrics.get("foster_computed_count"),
            "foster_insufficient_history_count": m1_metrics.get(
                "foster_insufficient_history_count"
            ),
            "foster_eligible_fraction": m1_metrics.get("foster_eligible_fraction"),
            "friction_computed_count": m1_metrics.get("friction_computed_count"),
            "friction_population_count": m1_metrics.get("friction_population_count"),
            "market_factor_symbol": m1_metrics.get("market_factor_symbol"),
            "assembled_count": (m1_metrics.get("assembly") or {}).get("assembled_count"),
            "m1_signals_persisted": (
                m1_metrics.get("orchestration") or {}
            ).get("total_signals_persisted"),
        },
        "m4_signals": {
            "signal_count": signal_count,
            "duplicate_signal_count": duplicate_signal_count,
            "features_with_signal_context": features_with_context,
            "fired_tickers": [row["ticker"] for row in fired_rows],
            "fired_feature_run_id_mismatch_count": fired_feature_run_id_mismatch_count,
            "fired_feature_run_id_mismatches": fired_feature_run_id_mismatches[:20],
        },
        "data_quality": {
            "feature_json_max_bytes": max(sizes) if sizes else 0,
            "feature_json_median_bytes": int(statistics.median(sizes)) if sizes else 0,
            "feature_json_raw_payload_hits": raw_payload_hits,
            "feature_json_api_key_hits": api_key_hits,
            "feature_json_token_hits": token_hits,
            "feature_json_secret_key_hits": secret_key_hits,
            "feature_json_secret_value_hits": secret_value_hits,
            "feature_json_secret_url_fragment_hits": secret_url_fragment_hits,
            "secret_hit_total": secret_total,
        },
        "forward_return_guard": {
            "forward_return_observation_count": forward_observation_count,
            "signals_with_forward_return": signals_with_forward_return,
            "forward_return_rows_created": forward_rows_created,
            # Static runner-action flag plus contamination detector from persisted rows.
            "forward_returns_run": False,
            "forward_return_contamination_detected": forward_rows_created > 0,
        },
        "forward_context_panel": {
            "job_run_id": forward_context_run_id,
            "schema_version": (forward_context_metrics or {}).get("schema_version"),
            "forward_session_date": (forward_context_metrics or {}).get(
                "forward_session_date"
            ),
            "asof_timestamp": (forward_context_metrics or {}).get("asof_timestamp"),
            "active_signal_count": (forward_context_metrics or {}).get(
                "active_signal_count"
            ),
            "eligible_signal_count": (forward_context_metrics or {}).get(
                "eligible_signal_count"
            ),
            "pending_signal_count": (forward_context_metrics or {}).get(
                "pending_signal_count"
            ),
            "rows_inserted": (forward_context_metrics or {}).get("rows_inserted"),
            "rows_existing": (forward_context_metrics or {}).get("rows_existing"),
            "ticker_fetch_count": (forward_context_metrics or {}).get(
                "ticker_fetch_count"
            ),
            "source_attempt_status_counts": (forward_context_metrics or {}).get(
                "source_attempt_status_counts"
            ),
            "no_op_reason": (forward_context_metrics or {}).get("no_op_reason"),
            "capture_complete": forward_context_capture_ok,
        },
        "source_attempts": {
            "source_attempt_status_counts": source_attempt_status_counts,
            "fired_source_attempt_status_counts": fired_status_counts,
            "non_fired_source_attempt_status_counts": non_fired_status_counts,
            "source_attempt_status_split": source_status_split,
            "source_attempt_count": sum(source_attempt_status_counts.values()),
            "signal_context_provider_error_count": provider_error_count,
            "fired_signal_context_error_count": fired_hard_error_count,
            "context_attached_count": (m4_metrics.get("signal_context") or {}).get(
                "context_attached_count"
            ),
            "context_reused_count": (m4_metrics.get("signal_context") or {}).get(
                "context_reused_count"
            ),
        },
        "freeze_reuse": {
            "fired_missing_signal_context_count": fired_missing_context,
            "next_execution_session_missing_count": next_execution_missing,
            "context_reused_from_persistence_count": (
                m4_metrics.get("signal_context") or {}
            ).get("context_reused_from_persistence_count"),
            "context_reused_in_memory_count": (
                m4_metrics.get("signal_context") or {}
            ).get("context_reused_in_memory_count"),
            "context_enriched_count": (m4_metrics.get("signal_context") or {}).get(
                "context_enriched_count"
            ),
            "context_persistence_miss_count": (
                m4_metrics.get("signal_context") or {}
            ).get("context_persistence_miss_count"),
            "context_persistence_mismatch_count": primary_mismatch,
            "context_persistence_mismatch_reasons": (
                m4_metrics.get("signal_context") or {}
            ).get("context_persistence_mismatch_reasons"),
        },
        "idempotency_rerun": {
            "m4_run_id": rerun_m4_run_id,
            "persistence_rerun_checked": bool(rerun_m4_run_id),
            "context_reused_from_persistence_count": (
                rerun_m4_metrics or {}
            ).get("signal_context", {}).get("context_reused_from_persistence_count"),
            "context_reused_in_memory_count": (
                rerun_m4_metrics or {}
            ).get("signal_context", {}).get("context_reused_in_memory_count"),
            "context_enriched_count": (
                rerun_m4_metrics or {}
            ).get("signal_context", {}).get("context_enriched_count"),
            "context_persistence_miss_count": (
                rerun_m4_metrics or {}
            ).get("signal_context", {}).get("context_persistence_miss_count"),
            "context_persistence_mismatch_count": (
                rerun_m4_metrics or {}
            ).get("signal_context", {}).get("context_persistence_mismatch_count"),
            "context_persistence_mismatch_reasons": (
                rerun_m4_metrics or {}
            ).get("signal_context", {}).get("context_persistence_mismatch_reasons"),
        },
        "run_diagnostics": m4_run_diagnostics,
        "fired_signal_table": fired_rows,
        "health_verdict": {
            "checks": verdict_checks,
            "failing_checks": failing,
        },
        "health": not failing,
    }
    return report


def _print_report(report: Dict[str, Any]) -> None:
    print("\n== M4 canonical accumulation health report ==")
    for section in (
        "run_metadata",
        "universe",
        "m4_assembly",
        "m1_assembly",
        "m4_signals",
        "data_quality",
        "forward_return_guard",
        "forward_context_panel",
        "source_attempts",
        "freeze_reuse",
        "idempotency_rerun",
        "run_diagnostics",
    ):
        if section not in report:
            continue
        print(f"\n[{section}]")
        for key in sorted(report[section]):
            print(f"  {key}: {report[section][key]}")
    print("\n[fired_signal_table]")
    if not report["fired_signal_table"]:
        print("  (no fired signals)")
    for row in report["fired_signal_table"]:
        print(
            f"  {row['ticker']:<8} {row['direction']:<6} "
            f"id={row['signal_identity_hash']} "
            f"next={row['next_execution_session']} "
            f"ctx={row['has_signal_context']} "
            f"strength={row['raw_signal_strength']}"
        )
    print("\n[health_verdict]")
    print(f"  health: {report['health']}")
    if report["health_verdict"]["failing_checks"]:
        print(f"  failing_checks: {report['health_verdict']['failing_checks']}")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def _run_universe(
    *,
    live: bool,
    schema: Optional[str],
    decision_date: str,
    skip_identity_enrichment: bool,
    args: argparse.Namespace,
) -> RunInvocation:
    argv = ["--live", "--trading-date", decision_date]
    if schema:
        argv += ["--schema", schema, "--create-tables"]
    if skip_identity_enrichment:
        argv += ["--skip-identity-enrichment"]
    else:
        argv += ["--require-identity-enrichment"]
    argv += [
        "--retry-backoff-seconds", str(args.retry_backoff_seconds),
        "--profile-max-workers", str(args.profile_max_workers),
        "--profile-rate-limit-per-minute", str(args.profile_rate_limit_per_minute),
    ]
    print("\n== Universe build ==")
    exit_code = run_universe.main(argv)
    return _recover_invocation(
        url=os.environ.get("DATABASE_URL", ""),
        schema=schema,
        job_name="universe_builder",
        decision_date=decision_date,
        exit_code=exit_code,
    )


def _run_m4(
    *,
    schema: Optional[str],
    run_timestamp: str,
    decision_date: str,
    args: argparse.Namespace,
    rerun: bool,
) -> RunInvocation:
    argv = ["--live", "--run-timestamp", run_timestamp,
            "--signal-context-breakout-buffer", str(args.signal_context_breakout_buffer)]
    if schema:
        argv += ["--schema", schema, "--create-tables"]
    print("\n== M4 daily rerun ==" if rerun else "\n== M4 daily ==")
    exit_code = run_m4_daily.main(argv)
    return _recover_invocation(
        url=os.environ.get("DATABASE_URL", ""),
        schema=schema,
        job_name="m4_daily_feature_assembly",
        decision_date=decision_date,
        exit_code=exit_code,
    )


def _run_m1(
    *,
    schema: Optional[str],
    run_timestamp: str,
    decision_date: str,
    args: argparse.Namespace,
) -> RunInvocation:
    argv = [
        "--live",
        "--run-timestamp", run_timestamp,
        "--earnings-window-sessions", str(args.m1_earnings_window_sessions),
        "--next-earnings-calendar-days", str(args.m1_next_earnings_calendar_days),
        "--price-lookback-calendar-days", str(args.m1_price_lookback_calendar_days),
    ]
    if schema:
        argv += ["--schema", schema, "--create-tables"]
    print("\n== M1 daily ==")
    exit_code = run_m1_daily.main(argv)
    return _recover_invocation(
        url=os.environ.get("DATABASE_URL", ""),
        schema=schema,
        job_name="m1_daily_feature_assembly",
        decision_date=decision_date,
        exit_code=exit_code,
    )


def _run_forward_context(
    *,
    schema: Optional[str],
    run_timestamp: str,
    decision_date: str,
) -> RunInvocation:
    argv = ["--live", "--run-timestamp", run_timestamp, "--pattern-id", "M4"]
    if schema:
        argv += ["--schema", schema, "--create-tables"]
    print("\n== Forward context panel ==")
    exit_code = run_forward_context.main(argv)
    return _recover_invocation(
        url=os.environ.get("DATABASE_URL", ""),
        schema=schema,
        job_name="forward_context_panel_collector",
        decision_date=decision_date,
        exit_code=exit_code,
    )


def _coerce_invocation(value: Any) -> RunInvocation:
    if isinstance(value, RunInvocation):
        return value
    if isinstance(value, int):
        return RunInvocation(exit_code=value, metrics={})
    raise TypeError(f"unexpected runner return value: {type(value).__name__}")


def _recover_invocation(
    *,
    url: str,
    schema: Optional[str],
    job_name: str,
    decision_date: str,
    exit_code: int,
) -> RunInvocation:
    if not url:
        return RunInvocation(exit_code=exit_code, metrics={})
    engine, session = _report_session(url, schema)
    try:
        row = _latest_job_run_for_decision(
            session,
            job_name=job_name,
            decision_date=decision_date,
            success_only=False,
        )
        if row is None:
            return RunInvocation(exit_code=exit_code, metrics={})
        return RunInvocation(
            exit_code=exit_code,
            run_id=row.job_run_id,
            run_status=row.run_status,
            metrics=_safe_json(row.metric_json),
        )
    finally:
        session.close()
        engine.dispose()


def _report_session(url: str, schema: Optional[str]):
    engine = create_engine(url, **schema_connect_args(url, schema))
    session = sessionmaker(bind=engine)()
    return engine, session


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical daily M4 signal accumulation runner."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true",
                      help="Accumulate into the canonical PostgreSQL target.")
    mode.add_argument("--scratch", action="store_true",
                      help="Accumulate into an isolated PostgreSQL scratch schema.")
    mode.add_argument("--dry-run", action="store_true",
                      help="Resolve and print the run plan only; no writes, no API calls.")
    parser.add_argument("--confirm-canonical-write", action="store_true",
                        help="Required acknowledgement for --live canonical writes.")
    parser.add_argument("--schema", help="Scratch schema name (required with --scratch).")
    parser.add_argument("--database-url", help="Override DATABASE_URL for this run.")
    parser.add_argument("--decision-date",
                        help="Explicit decision date (YYYY-MM-DD). Must be a session, not future.")
    parser.add_argument("--run-timestamp", help="Timezone-aware run timestamp.")
    parser.add_argument("--json-output", help="Write the health report JSON to this path.")
    parser.add_argument("--skip-identity-enrichment", action="store_true",
                        help="Skip Polygon identity enrichment in the universe build.")
    parser.add_argument("--skip-rerun", action="store_true",
                        help="Skip the idempotency freeze/reuse rerun.")
    parser.add_argument("--enable-m1", action="store_true",
                        help="Enable the M1 PEAD signal-only assembly step.")
    parser.add_argument("--skip-m1", action="store_true",
                        help="Keep the M1 PEAD signal-only assembly step disabled.")
    parser.add_argument("--skip-forward-context", action="store_true",
                        help="Skip the post-M4 forward-context panel collector.")
    parser.add_argument("--signal-context-breakout-buffer", type=float, default=0.02)
    parser.add_argument("--retry-backoff-seconds", type=float, default=0.25)
    parser.add_argument("--profile-max-workers", type=int, default=20)
    parser.add_argument("--profile-rate-limit-per-minute", type=int, default=2000)
    parser.add_argument("--m1-earnings-window-sessions", type=int, default=15)
    parser.add_argument("--m1-next-earnings-calendar-days", type=int, default=140)
    parser.add_argument("--m1-price-lookback-calendar-days", type=int, default=430)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    previous_env = {
        "DATABASE_URL": os.environ.get("DATABASE_URL"),
        "ALPHA_DB_SCHEMA": os.environ.get("ALPHA_DB_SCHEMA"),
    }
    try:
        return _main_impl(argv)
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_globals()


def _main_impl(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    load_runtime_env()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
    url = os.environ.get("DATABASE_URL", "")

    if args.enable_m1 and args.skip_m1:
        print("ERROR: --enable-m1 and --skip-m1 cannot be combined")
        return 1

    try:
        run_ts = _parse_timestamp(args.run_timestamp)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        clock = resolve_canonical_clock(run_ts, args.decision_date)
    except NonTradingDayNoOp as exc:
        print(f"NO-OP: {exc}")
        return 0
    except CanonicalRunError as exc:
        print(f"ERROR: {exc}")
        return 1

    decision_date = clock["decision_date"]
    effective_run_timestamp = clock["effective_run_timestamp"]
    host_class = _url_metadata(url)["host_class"]
    app_commit_sha = _app_commit_sha()

    mode = "live" if args.live else "scratch" if args.scratch else "dry-run"
    schema = args.schema if args.scratch else None

    print("== M4 canonical accumulation ==")
    print(f"Mode:                   {mode}")
    print(f"DB host class:          {host_class}")
    print(f"Schema:                 {schema or '(default search_path)'}")
    print(f"App commit:             {app_commit_sha}")
    print(f"Run timestamp:          {effective_run_timestamp}")
    print(f"Decision date:          {decision_date}")
    print(f"Evidence session date:  {clock['evidence_session_date']}")
    print(f"Next execution session: {clock['next_execution_session']}")

    # --- mode gates (refuse before any DB/API work) ---
    try:
        if args.live:
            if not args.confirm_canonical_write:
                raise CanonicalRunError(
                    "--live requires --confirm-canonical-write to acknowledge a "
                    "canonical write"
                )
            require_canonical_target(url)
        elif args.scratch:
            require_scratch_schema(schema, url)
    except CanonicalRunError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.dry_run:
        print("\nDry run: no database writes and no provider API calls performed.")
        m1_step = "M1 daily" if args.enable_m1 and not args.skip_m1 else "M1 daily skipped"
        print(
            "Planned steps: universe build -> M4 daily -> "
            f"{m1_step} -> "
            "forward context panel -> health report."
        )
        return 0

    if args.live:
        try:
            head = verify_alembic_head(url)
            print(f"Alembic head verified:  {head['heads']}")
        except CanonicalRunError as exc:
            print(f"ERROR: {exc}")
            return 1
        except Exception as exc:  # pragma: no cover - environment specific
            print(f"ERROR: alembic head check failed: {exc.__class__.__name__}: {exc}")
            return 1

    if args.scratch:
        os.environ["ALPHA_DB_SCHEMA"] = schema
    reset_globals()

    universe_invocation = _coerce_invocation(_run_universe(
        live=args.live,
        schema=schema,
        decision_date=decision_date,
        skip_identity_enrichment=args.skip_identity_enrichment,
        args=args,
    ))
    if universe_invocation.exit_code != 0:
        print(f"ERROR: universe build failed with exit {universe_invocation.exit_code}")
        return universe_invocation.exit_code
    if (
        not universe_invocation.run_id
        or universe_invocation.run_status not in SUCCESS_RUN_STATUSES
    ):
        print("ERROR: universe build returned exit 0 without a finished job run")
        return 1

    reset_globals()
    primary_m4_invocation = _coerce_invocation(_run_m4(
        schema=schema,
        run_timestamp=effective_run_timestamp,
        decision_date=decision_date,
        args=args,
        rerun=False,
    ))
    if primary_m4_invocation.exit_code != 0:
        print(f"ERROR: M4 daily failed with exit {primary_m4_invocation.exit_code}")
        return primary_m4_invocation.exit_code
    if (
        not primary_m4_invocation.run_id
        or primary_m4_invocation.run_status not in SUCCESS_RUN_STATUSES
    ):
        print("ERROR: M4 daily returned exit 0 without a finished job run")
        return 1

    rerun_m4_invocation = RunInvocation(exit_code=0, metrics={})
    if not args.skip_rerun:
        reset_globals()
        rerun_m4_invocation = _coerce_invocation(_run_m4(
            schema=schema,
            run_timestamp=effective_run_timestamp,
            decision_date=decision_date,
            args=args,
            rerun=True,
        ))
        if rerun_m4_invocation.exit_code != 0:
            print(f"ERROR: M4 idempotency rerun failed with exit {rerun_m4_invocation.exit_code}")
            return rerun_m4_invocation.exit_code
        if (
            not rerun_m4_invocation.run_id
            or rerun_m4_invocation.run_status not in SUCCESS_RUN_STATUSES
        ):
            print("ERROR: M4 idempotency rerun returned exit 0 without a finished job run")
            return 1

    m1_invocation = RunInvocation(
        exit_code=0,
        run_status="finished",
        metrics={"no_op_reason": "skipped_default_off"},
    )
    if args.skip_m1:
        m1_invocation = RunInvocation(
            exit_code=0,
            run_status="finished",
            metrics={"no_op_reason": "skipped_by_cli"},
        )
        print("\nM1 daily skipped by --skip-m1.")
    elif not args.enable_m1:
        print("\nM1 daily skipped; pass --enable-m1 after re-audit to run it.")
    else:
        reset_globals()
        m1_invocation = _coerce_invocation(_run_m1(
            schema=schema,
            run_timestamp=effective_run_timestamp,
            decision_date=decision_date,
            args=args,
        ))
        if m1_invocation.exit_code != 0:
            print(f"ERROR: M1 daily failed with exit {m1_invocation.exit_code}")
            return m1_invocation.exit_code
        if (
            not m1_invocation.run_id
            or m1_invocation.run_status not in SUCCESS_RUN_STATUSES
        ):
            print("ERROR: M1 daily returned exit 0 without a finished job run")
            return 1

    forward_context_invocation = RunInvocation(
        exit_code=0,
        run_status="finished",
        metrics={},
    )
    if args.skip_forward_context:
        forward_context_invocation = RunInvocation(
            exit_code=0,
            run_status="finished",
            metrics={"no_op_reason": "skipped_by_cli"},
        )
        print("\nForward context panel skipped by --skip-forward-context.")
    elif args.decision_date:
        # A historical decision-date override may anchor effective_run_timestamp
        # to a past close. Forward context is capture-or-lose live evidence, so
        # do not pretend a historical rerun can recover that panel.
        forward_context_invocation = RunInvocation(
            exit_code=0,
            run_status="finished",
            metrics={"no_op_reason": "skipped_for_decision_date_override"},
        )
        print("\nForward context panel skipped for explicit decision-date rerun.")
    else:
        reset_globals()
        forward_context_invocation = _coerce_invocation(_run_forward_context(
            schema=schema,
            run_timestamp=effective_run_timestamp,
            decision_date=decision_date,
        ))
        if forward_context_invocation.exit_code != 0:
            print(
                "ERROR: forward context panel failed with exit "
                f"{forward_context_invocation.exit_code}"
            )
            return forward_context_invocation.exit_code
        if (
            not forward_context_invocation.run_id
            or forward_context_invocation.run_status not in SUCCESS_RUN_STATUSES
        ):
            print("ERROR: forward context panel returned exit 0 without a finished job run")
            return 1

    engine, session = _report_session(url, schema)
    try:
        report = build_m4_health_report(
            session,
            mode=mode,
            schema=schema,
            host_class=host_class,
            app_commit_sha=app_commit_sha,
            decision_date=decision_date,
            evidence_session_date=clock["evidence_session_date"],
            next_execution_session=clock["next_execution_session"],
            run_timestamp=effective_run_timestamp,
            universe_metrics=universe_invocation.metrics,
            universe_run_id=universe_invocation.run_id,
            m1_run_id=m1_invocation.run_id,
            m1_metrics=m1_invocation.metrics,
            m4_metrics=primary_m4_invocation.metrics,
            primary_m4_run_id=primary_m4_invocation.run_id,
            rerun_m4_run_id=rerun_m4_invocation.run_id,
            rerun_m4_metrics=rerun_m4_invocation.metrics,
            forward_context_run_id=forward_context_invocation.run_id,
            forward_context_metrics=forward_context_invocation.metrics,
        )
    finally:
        session.close()
        engine.dispose()

    _print_report(report)
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"\nHealth report JSON written to {args.json_output}")

    if not report["health"]:
        print("\nERROR: canonical accumulation health verdict failed")
        return 1
    print("\nCanonical accumulation completed; health verdict OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
