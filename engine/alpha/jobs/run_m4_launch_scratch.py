#!/usr/bin/env python3
"""Launch-like Supabase scratch rehearsal for universe -> M4 daily.

This entrypoint intentionally refuses SQLite and default schemas. It is a
repeatable live rehearsal command, not a production accumulation command.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import create_engine, text

from alpha.db.engine import (
    prepare_writable_schema_target,
    reset_globals,
    schema_connect_args,
)
from alpha.db.models import (
    CanonicalUniverseScan,
    DataLineage,
    FeatureSnapshot,
    SignalRegistry,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.jobs import run_m4_daily, run_universe
from alpha.market_calendar import resolve_us_equity_session
from alpha.runtime_env import load_runtime_env


REQUIRED_ENV = ("DATABASE_URL", "FMP_API_KEY", "POLYGON_API_KEY", "BENZINGA_API_KEY")


def _schema_name(now: Optional[datetime] = None) -> str:
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    return f"m4_live_scratch_{ts}"


def _parse_timestamp(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("run_timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _url_metadata(url: str) -> Dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""
    if hostname.endswith(".supabase.co") and hostname.startswith("db."):
        host_class = "direct_supabase"
    elif "pooler.supabase" in hostname or "pooler" in hostname:
        host_class = "pooler"
    elif hostname in {"localhost", "127.0.0.1", "::1"}:
        host_class = "localhost"
    elif parsed.scheme.startswith("sqlite"):
        host_class = "sqlite"
    else:
        host_class = "postgres_other"
    return {
        "scheme": parsed.scheme,
        "hostname": hostname,
        "port": parsed.port,
        "database": parsed.path.lstrip("/") or None,
        "sslmode": urllib.parse.parse_qs(parsed.query).get("sslmode", ["absent"])[0],
        "host_class": host_class,
    }


def _require_safe_database(url: str, schema: str) -> None:
    if not url:
        raise ValueError("DATABASE_URL is required")
    if not url.startswith("postgresql"):
        raise ValueError("launch scratch requires a PostgreSQL DATABASE_URL")
    if not schema or schema == "public":
        raise ValueError("launch scratch requires a non-public scratch schema")
    if not schema.startswith("m4_live_scratch_"):
        raise ValueError("scratch schema must start with m4_live_scratch_")


def _presence_report(keys: Iterable[str]) -> Dict[str, bool]:
    return {key: bool(os.environ.get(key)) for key in keys}


def _preflight_network(url: str) -> None:
    meta = _url_metadata(url)
    host = meta["hostname"]
    port = meta["port"] or 5432
    if not host:
        raise ValueError("DATABASE_URL host is missing")
    print(f"DB host class:          {meta['host_class']}")
    print(f"DB scheme:              {meta['scheme']}")
    print(f"DB host:                {host}")
    print(f"DB port:                {port}")
    print(f"DB database:            {meta['database']}")
    print(f"DB sslmode:             {meta['sslmode']}")
    infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    families = sorted({
        "IPv6" if item[0] == socket.AF_INET6 else "IPv4"
        if item[0] == socket.AF_INET else str(item[0])
        for item in infos
    })
    print(f"DNS resolves:           yes ({','.join(families)})")
    start = time.monotonic()
    conn = socket.create_connection((host, port), timeout=10)
    conn.close()
    print(f"TCP connect:            yes ({int((time.monotonic() - start) * 1000)} ms)")


def _preflight_database(url: str, schema: str) -> None:
    prepare_writable_schema_target(
        url=url,
        schema=schema,
        create_tables=True,
    )
    engine = create_engine(url, **schema_connect_args(url, schema))
    try:
        with engine.begin() as conn:
            conn.execute(text(f'SET search_path TO "{schema}"'))
            current_schema = conn.execute(text("SELECT current_schema()")).scalar()
            database = conn.execute(text("SELECT current_database()")).scalar()
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS _connectivity_check "
                "(id integer primary key, created_at timestamptz default now())"
            ))
            conn.execute(text(
                "INSERT INTO _connectivity_check(id) VALUES (1) "
                "ON CONFLICT (id) DO NOTHING"
            ))
            count = conn.execute(text("SELECT count(*) FROM _connectivity_check")).scalar()
            conn.execute(text("DROP TABLE _connectivity_check"))
        print(f"DB connection:          yes")
        print(f"DB current_database:    {database}")
        print(f"DB current_schema:      {current_schema}")
        print(f"Scratch write/read:     yes ({count})")
    finally:
        engine.dispose()


def _run_universe(schema: str, decision_date: str, args: argparse.Namespace) -> int:
    argv = [
        "--live",
        "--schema", schema,
        "--create-tables",
        "--trading-date", decision_date,
        "--require-identity-enrichment",
        "--retry-backoff-seconds", str(args.retry_backoff_seconds),
        "--profile-max-workers", str(args.profile_max_workers),
        "--profile-rate-limit-per-minute", str(args.profile_rate_limit_per_minute),
    ]
    print("\n== Universe build ==")
    return run_universe.main(argv)


def _run_m4(schema: str, run_timestamp: str, args: argparse.Namespace, *, rerun: bool) -> int:
    argv = [
        "--live",
        "--schema", schema,
        "--create-tables",
        "--run-timestamp", run_timestamp,
        "--signal-context-breakout-buffer", str(args.signal_context_breakout_buffer),
    ]
    print("\n== M4 daily rerun ==" if rerun else "\n== M4 daily ==")
    return run_m4_daily.main(argv)


def _query_summary(url: str, schema: str) -> Dict[str, Any]:
    engine = create_engine(url, **schema_connect_args(url, schema))
    try:
        from sqlalchemy.orm import sessionmaker

        session = sessionmaker(bind=engine)()
        try:
            features = (
                session.query(FeatureSnapshot)
                .filter(FeatureSnapshot.pattern_id == "M4")
                .all()
            )
            signals = (
                session.query(SignalRegistry)
                .filter(SignalRegistry.pattern_id == "M4")
                .all()
            )
            feature_payloads = [_safe_json(feature.feature_json) for feature in features]
            with_context = [
                payload for payload in feature_payloads
                if isinstance(payload.get("signal_context"), dict)
            ]
            sizes = [len(feature.feature_json or "") for feature in features]
            signal_feature_ids = {signal.feature_snapshot_id for signal in signals}
            feature_by_id = {feature.feature_snapshot_id: feature for feature in features}
            fired_missing_context = 0
            for feature_id in signal_feature_ids:
                payload = _safe_json(getattr(feature_by_id.get(feature_id), "feature_json", "{}"))
                if not isinstance(payload.get("signal_context"), dict):
                    fired_missing_context += 1
            return {
                "universe_scans": session.query(UniverseScan).count(),
                "canonical_universe_scans": session.query(CanonicalUniverseScan).count(),
                "universe_snapshots": session.query(UniverseSnapshot).count(),
                "data_lineage": session.query(DataLineage).count(),
                "m4_feature_snapshots": len(features),
                "m4_signal_registry": len(signals),
                "m4_features_with_signal_context": len(with_context),
                "m4_fired_missing_signal_context": fired_missing_context,
                "feature_json_max_bytes": max(sizes) if sizes else 0,
                "feature_json_median_bytes": int(statistics.median(sizes)) if sizes else 0,
                "feature_json_raw_payload_hits": _substring_hits(features, "raw_payload"),
                "feature_json_api_key_hits": _substring_hits(features, "apiKey"),
                "feature_json_token_hits": _substring_hits(features, "token"),
            }
        finally:
            session.close()
    finally:
        engine.dispose()


def _safe_json(value: str) -> Dict[str, Any]:
    try:
        payload = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _substring_hits(features: Iterable[FeatureSnapshot], needle: str) -> int:
    return sum(1 for feature in features if needle in (feature.feature_json or ""))


def _print_summary(title: str, summary: Dict[str, Any]) -> None:
    print(f"\n== {title} ==")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a guarded live Supabase scratch rehearsal for M4."
    )
    parser.add_argument("--database-url", help="Override DATABASE_URL for this run")
    parser.add_argument("--schema", help="Scratch schema name. Defaults to generated.")
    parser.add_argument("--run-timestamp", help="Timezone-aware run timestamp.")
    parser.add_argument(
        "--signal-context-breakout-buffer",
        type=float,
        default=0.02,
        help="M4 signal_context prefilter buffer.",
    )
    parser.add_argument("--skip-rerun", action="store_true", help="Skip M4 freeze/reuse rerun.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=0.25)
    parser.add_argument("--profile-max-workers", type=int, default=20)
    parser.add_argument("--profile-rate-limit-per-minute", type=int, default=2000)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    load_runtime_env()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
    url = os.environ.get("DATABASE_URL", "")
    schema = args.schema or _schema_name()
    run_ts = _parse_timestamp(args.run_timestamp)
    decision_date = resolve_us_equity_session(run_ts).decision_date

    try:
        _require_safe_database(url, schema)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    os.environ["ALPHA_DB_SCHEMA"] = schema
    reset_globals()

    print("== Preflight ==")
    print(f"Scratch schema:         {schema}")
    print(f"Run timestamp:          {run_ts.isoformat()}")
    print(f"Decision date:          {decision_date}")
    for key, present in _presence_report(REQUIRED_ENV).items():
        print(f"{key} present:          {present}")

    try:
        _preflight_network(url)
        _preflight_database(url, schema)
    except Exception as exc:
        print(f"ERROR: preflight failed: {exc.__class__.__name__}: {exc}")
        return 1

    universe_status = _run_universe(schema, decision_date, args)
    if universe_status != 0:
        print(f"ERROR: universe build failed with exit {universe_status}")
        return universe_status

    reset_globals()
    first_status = _run_m4(schema, run_ts.isoformat(), args, rerun=False)
    first_summary = _query_summary(url, schema)
    _print_summary("Scratch summary after first M4", first_summary)
    if first_status != 0:
        print(f"ERROR: M4 daily failed with exit {first_status}")
        return first_status
    if first_summary["m4_fired_missing_signal_context"]:
        print("ERROR: at least one fired M4 signal is missing signal_context")
        return 1

    if not args.skip_rerun:
        reset_globals()
        rerun_status = _run_m4(schema, run_ts.isoformat(), args, rerun=True)
        rerun_summary = _query_summary(url, schema)
        _print_summary("Scratch summary after rerun", rerun_summary)
        if rerun_status != 0:
            print(f"ERROR: M4 rerun failed with exit {rerun_status}")
            return rerun_status

    print("\nLaunch scratch rehearsal completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
