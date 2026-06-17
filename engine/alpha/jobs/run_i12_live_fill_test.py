#!/usr/bin/env python3
"""Run the read-only I12 Stage-0 live fill-test machine."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from alpha.data.alpaca import AlpacaAdapter
from alpha.data.config import AlpacaConfig, ConfigError
from alpha.data.contracts import stable_hash
from alpha.db.engine import (
    SchemaTargetError,
    open_writable_session,
    prepare_writable_schema_target,
    reset_globals,
    schema_connect_args,
)
from alpha.db.models import FeatureSnapshot, MLModelRegistry, SignalRegistry, SignalMLScore
from alpha.jobs.i12_live_fill_test import (
    DEFAULT_INTENDED_ORDER_USD,
    DEFAULT_MAX_SPREAD_BPS,
    DEFAULT_MAX_QUOTE_AGE_SECONDS,
    DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
    DEFAULT_MAX_SNAPSHOT_ERROR_OR_MISSING_RATE,
    DEFAULT_MIN_CONTEXT_COUNT,
    DEFAULT_MIN_EXIT_QUOTE_OK_RATE,
    DEFAULT_MIN_GATE0_DISTINCT_TRADING_DAYS,
    DEFAULT_MIN_GATE0_INTENDED_COUNT,
    DEFAULT_MIN_GATE0_TRADEABLE_RATE,
    DEFAULT_MIN_INTENDED_COUNT,
    DEFAULT_MIN_QUOTE_OK_RATE,
    DEFAULT_MIN_SCORE_MODEL_OK_RATE,
    DEFAULT_MIN_SNAPSHOT_OK_RATE,
    DEFAULT_TOP_K,
    EXPECTED_I12_LIVE_FEATURES,
    I12LiveFillConfig,
    I12LiveFillTestJob,
    capture_i12_exit_quotes,
    i12_gate0_report,
    select_i12_model,
)
from alpha.jobs.paper_execution import load_premarket_context_artifact
from alpha.jobs.runner import run_job
from alpha.ml.inference import _artifact_identity_mismatch, _load_artifact, score_signal_shadow
from alpha.ml.model_features import audit_feature_schema_no_leakage, feature_schema_hash
from alpha.runtime_env import load_runtime_env


EASTERN = ZoneInfo("America/New_York")
I12_FILL_TEST_REQUIRED_TABLES = (
    "evidence_jobs",
    "evidence_job_runs",
    "feature_snapshots",
    "signal_registry",
    "signal_ml_scores",
    "ml_model_registry",
    "i12_fill_log",
)


def _run(args: argparse.Namespace) -> int:
    load_runtime_env()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        reset_globals()
    if args.schema:
        os.environ["ALPHA_DB_SCHEMA"] = args.schema
        reset_globals()

    try:
        alpaca = AlpacaAdapter(AlpacaConfig.from_env())
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.alpaca_probe_symbol:
        return _run_quote_probe(alpaca, args)

    try:
        schema = require_stage0_scratch_schema(args.schema or os.environ.get("ALPHA_DB_SCHEMA"))
    except SchemaTargetError as exc:
        print(f"ERROR: {exc}")
        return 1
    try:
        prepare_writable_schema_target(
            schema=schema,
            create_tables=args.create_tables,
            required_tables=I12_FILL_TEST_REQUIRED_TABLES,
        )
        session = open_writable_session(schema=schema)
    except (SchemaTargetError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        if args.exit_quotes:
            result = capture_i12_exit_quotes(
                session,
                alpaca,
                feed=args.feed,
                asof=_parse_datetime(args.asof) if args.asof else None,
                max_quote_age_seconds=args.max_quote_age_seconds,
            )
            print(json.dumps(result, sort_keys=True))
            return 0 if "error" not in result else 1

        if args.gate0_report:
            report = i12_gate0_report(
                session,
                decision_date=_parse_date(args.trading_date) if args.trading_date else None,
                asof=_parse_datetime(args.asof) if args.asof else None,
                max_spread_bps=args.max_spread_bps,
                intended_order_usd=args.intended_order_usd,
                min_context_count=args.min_context_count,
                min_intended_count=args.min_intended_count,
                min_snapshot_ok_rate=args.min_snapshot_ok_rate,
                max_snapshot_error_or_missing_rate=(
                    args.max_snapshot_error_or_missing_rate
                ),
                min_score_model_ok_rate=args.min_score_model_ok_rate,
                min_quote_ok_rate=args.min_quote_ok_rate,
                min_exit_quote_ok_rate=args.min_exit_quote_ok_rate,
                min_gate0_intended_count=args.min_gate0_intended_count,
                min_gate0_distinct_trading_days=(
                    args.min_gate0_distinct_trading_days
                ),
                min_gate0_tradeable_rate=args.min_gate0_tradeable_rate,
            )
            print(json.dumps(report, sort_keys=True))
            if args.fail_on_gate0_fail and not report["passed"]:
                return 2
            return 0

        trading_date = _parse_date(args.trading_date) if args.trading_date else date.today()
        contexts = load_premarket_context_artifact(
            args.context_artifact,
            expected_date=trading_date,
        )
        context_artifact_hash = _context_artifact_hash(args.context_artifact)
        ensure_model_registry_row_in_scratch(
            database_url=args.database_url or os.environ.get("DATABASE_URL"),
            scratch_session=session,
            model_id=args.model_id,
            allow_latest_model=args.allow_latest_model,
            feed=args.feed,
        )
        config = I12LiveFillConfig(
            model_id=args.model_id,
            allow_latest_model=args.allow_latest_model,
            top_k=args.top_k,
            intended_order_usd=args.intended_order_usd,
            max_spread_bps=args.max_spread_bps,
            feed=args.feed,
            max_quote_age_seconds=args.max_quote_age_seconds,
            max_snapshot_age_seconds=args.max_snapshot_age_seconds,
            min_context_count=args.min_context_count,
            min_intended_count=args.min_intended_count,
            min_snapshot_ok_rate=args.min_snapshot_ok_rate,
            max_snapshot_error_or_missing_rate=args.max_snapshot_error_or_missing_rate,
            min_score_model_ok_rate=args.min_score_model_ok_rate,
            min_quote_ok_rate=args.min_quote_ok_rate,
            min_exit_quote_ok_rate=args.min_exit_quote_ok_rate,
            context_artifact_hash=context_artifact_hash,
            require_market_open=not args.allow_market_closed,
        )
        while True:
            loop_asof = _parse_datetime(args.asof) if args.asof else datetime.now(timezone.utc)
            if not args.once and _local_trading_date(loop_asof) != trading_date:
                print(
                    json.dumps(
                        {
                            "status": "finished",
                            "reason": "trading_date_elapsed",
                            "configured_trading_date": trading_date.isoformat(),
                            "asof": loop_asof.isoformat(),
                            "read_only": True,
                        },
                        sort_keys=True,
                    )
                )
                return 0
            job = I12LiveFillTestJob(
                session=session,
                alpaca_adapter=alpaca,
                contexts=contexts,
                config=config,
                asof=loop_asof,
            )
            result = run_job(
                session,
                job,
                params={
                    "trading_date": trading_date.isoformat(),
                    "top_k": args.top_k,
                    "intended_order_usd": args.intended_order_usd,
                    "max_spread_bps": args.max_spread_bps,
                    "max_quote_age_seconds": args.max_quote_age_seconds,
                    "max_snapshot_age_seconds": args.max_snapshot_age_seconds,
                    "min_context_count": args.min_context_count,
                    "min_intended_count": args.min_intended_count,
                    "min_snapshot_ok_rate": args.min_snapshot_ok_rate,
                    "max_snapshot_error_or_missing_rate": (
                        args.max_snapshot_error_or_missing_rate
                    ),
                    "min_score_model_ok_rate": args.min_score_model_ok_rate,
                    "min_quote_ok_rate": args.min_quote_ok_rate,
                    "min_exit_quote_ok_rate": args.min_exit_quote_ok_rate,
                    "context_artifact_hash": context_artifact_hash,
                    "feed": args.feed,
                    "read_only": True,
                    "schema": schema,
                },
            )
            print(json.dumps(_job_result_payload(result), sort_keys=True))
            if args.once:
                return 0 if result.status == "finished" else 1
            time.sleep(args.poll_interval_seconds)
    finally:
        session.close()


def _run_quote_probe(alpaca: AlpacaAdapter, args: argparse.Namespace) -> int:
    response = alpaca.get_latest_quote(args.alpaca_probe_symbol, feed=args.feed)
    if not response.ok or response.data is None:
        print(json.dumps({"ok": False, "error": str(response.error)}, sort_keys=True))
        return 1
    quote = response.data
    print(
        json.dumps(
            {
                "ok": True,
                "symbol": quote.symbol,
                "bid": quote.bid_price,
                "ask": quote.ask_price,
                "bid_size": quote.bid_size,
                "ask_size": quote.ask_size,
                "timestamp": quote.timestamp,
                "read_only": True,
            },
            sort_keys=True,
        )
    )
    return 0


def require_stage0_scratch_schema(schema: str | None) -> str:
    if not schema:
        raise SchemaTargetError("Stage-0 fill-test persistence requires --schema SCRATCH_SCHEMA")
    if schema.strip().casefold() == "public":
        raise SchemaTargetError("Stage-0 fill-test persistence refuses public schema")
    return schema


def ensure_model_registry_row_in_scratch(
    *,
    database_url: str | None,
    scratch_session: Any,
    model_id: str | None,
    allow_latest_model: bool,
    feed: str,
) -> str:
    url = database_url or os.environ.get("DATABASE_URL", "sqlite:///alpha_capital.db")
    canonical = _open_canonical_session(url)
    try:
        contract = select_i12_model(
            canonical,
            model_id=model_id,
            allow_latest_model=allow_latest_model,
            feed=feed,
        )
        source = contract.model
        validate_i12_stage0_artifact_preflight(source)
        existing = scratch_session.get(MLModelRegistry, source.model_id)
        if existing is None:
            scratch_model = _copy_model_registry_row(source)
            scratch_session.add(scratch_model)
        else:
            _copy_model_registry_values(source, existing)
            scratch_model = existing
        scratch_session.flush()
        validate_i12_stage0_artifact_preflight(
            scratch_model,
            session=scratch_session,
        )
        scratch_session.commit()
        return source.model_id
    finally:
        canonical.close()


def validate_i12_stage0_artifact_preflight(
    model: MLModelRegistry,
    *,
    session: Any | None = None,
) -> None:
    artifact = _load_artifact(model.artifact_uri)
    mismatches = _artifact_identity_mismatch(artifact, model)
    if mismatches:
        raise RuntimeError(
            "I12 Stage-0 model artifact identity mismatch: "
            f"{json.dumps(mismatches, sort_keys=True, default=str)}"
        )
    feature_names = tuple(artifact.get("feature_names") or ())
    if feature_names != EXPECTED_I12_LIVE_FEATURES:
        raise RuntimeError(
            "I12 Stage-0 model artifact feature_names do not match live contract: "
            f"{feature_names!r}"
        )
    feature_schema = artifact.get("feature_schema")
    if not isinstance(feature_schema, dict):
        raise RuntimeError("I12 Stage-0 model artifact missing feature_schema object")
    audit_feature_schema_no_leakage(feature_schema)
    if feature_schema_hash(feature_schema) != model.feature_schema_hash:
        raise RuntimeError("I12 Stage-0 model artifact feature_schema_hash mismatch")
    ranges = artifact.get("training_feature_ranges")
    if not isinstance(ranges, list) or len(ranges) != len(feature_names):
        raise RuntimeError(
            "I12 Stage-0 model artifact training_feature_ranges missing or "
            "wrong length"
        )
    model_obj = artifact.get("model")
    if model_obj is None or not hasattr(model_obj, "predict"):
        raise RuntimeError("I12 Stage-0 model artifact does not contain a scoring model")
    if session is None:
        return
    _score_stage0_artifact_smoke(session, model, feature_names, ranges)


def _score_stage0_artifact_smoke(
    session: Any,
    model: MLModelRegistry,
    feature_names: tuple[str, ...],
    training_feature_ranges: list[Any],
) -> None:
    suffix = stable_hash(
        {
            "model_id": model.model_id,
            "artifact_uri": model.artifact_uri,
            "stage": "i12_stage0_preflight",
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    )[:24]
    feature_snapshot_id = f"i12-stage0-preflight-fs-{suffix}"
    signal_id = f"i12-stage0-preflight-sig-{suffix}"
    ticker = f"PF{suffix[:8]}".upper()
    asof_ts = datetime.now(timezone.utc)
    feature_payload = _feature_payload_from_training_ranges(
        feature_names,
        training_feature_ranges,
    )
    feature = FeatureSnapshot(
        feature_snapshot_id=feature_snapshot_id,
        pattern_id="I12",
        ticker=ticker,
        asof_timestamp=asof_ts,
        feature_manifest_version="i12_live_stage0_preflight",
        feature_json=json.dumps(feature_payload, sort_keys=True),
        feature_hash=stable_hash(feature_payload),
        data_lineage_ids="[]",
        fidelity_tier="stage0_preflight",
        point_in_time_passed=True,
        lookahead_guard_passed=True,
        input_hashes="{}",
        output_hash=suffix,
    )
    signal = SignalRegistry(
        signal_id=signal_id,
        pattern_id="I12",
        ticker=ticker,
        direction="long",
        signal_timestamp=asof_ts,
        raw_signal_strength=0.0,
        raw_expected_edge=0.0,
        signal_horizon="1d",
        thesis_category="stage0_preflight",
        route_class="stage0_read_only",
        fidelity_tier="stage0_preflight",
        data_confidence=1.0,
        feature_snapshot_id=feature_snapshot_id,
        signal_status="active",
        trading_date=asof_ts.date().isoformat(),
        next_execution_session=asof_ts.date().isoformat(),
        detector_version="i12_live_stage0_preflight",
        point_in_time_passed=True,
        lookahead_guard_passed=True,
        data_lineage_ids="[]",
        signal_identity_hash=suffix,
        intended_entry_price=0.0,
    )
    try:
        session.add(feature)
        session.add(signal)
        session.flush()
        score = score_signal_shadow(
            session,
            signal_id=signal_id,
            model_id=model.model_id,
            score_status="stage0_preflight",
        )
        if score.score_source != "model_shadow":
            raise RuntimeError(
                "I12 Stage-0 model artifact smoke score fell back: "
                f"{score.fallback_reason}"
            )
    finally:
        for row in (
            session.query(SignalMLScore)
            .filter(SignalMLScore.signal_id == signal_id)
            .all()
        ):
            session.delete(row)
        if signal in session:
            session.delete(signal)
        if feature in session:
            session.delete(feature)
        session.flush()


def _feature_payload_from_training_ranges(
    feature_names: tuple[str, ...],
    training_feature_ranges: list[Any],
) -> dict[str, float]:
    payload: dict[str, float] = {}
    for name, bounds in zip(feature_names, training_feature_ranges):
        if not isinstance(bounds, dict):
            raise RuntimeError(
                "I12 Stage-0 model artifact training_feature_ranges entries "
                "must be objects"
            )
        lower = _finite_optional_bound(bounds.get("min"), name, "min")
        upper = _finite_optional_bound(bounds.get("max"), name, "max")
        if lower is not None and upper is not None:
            if lower > upper:
                raise RuntimeError(
                    "I12 Stage-0 model artifact training_feature_ranges min "
                    f"exceeds max for {name!r}"
                )
            value = (lower + upper) / 2.0
        elif lower is not None:
            value = lower
        elif upper is not None:
            value = upper
        else:
            value = 0.0
        if not math.isfinite(value):
            raise RuntimeError(
                "I12 Stage-0 model artifact smoke feature value is non-finite "
                f"for {name!r}"
            )
        payload[name] = float(value)
    return payload


def _finite_optional_bound(value: Any, feature_name: str, bound_name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(
            "I12 Stage-0 model artifact training_feature_ranges "
            f"{bound_name} for {feature_name!r} must be finite or null"
        )
    return float(value)


def _open_canonical_session(url: str) -> Any:
    engine = create_engine(
        url,
        echo=False,
        pool_pre_ping=True,
        **schema_connect_args(url, None),
    )
    Session = sessionmaker(bind=engine)
    session = Session()
    session._stage0_engine = engine  # type: ignore[attr-defined]

    original_close = session.close

    def close_with_engine_dispose() -> None:
        try:
            original_close()
        finally:
            engine.dispose()

    session.close = close_with_engine_dispose  # type: ignore[method-assign]
    return session


def _copy_model_registry_row(source: MLModelRegistry) -> MLModelRegistry:
    return MLModelRegistry(**_model_registry_values(source))


def _copy_model_registry_values(source: MLModelRegistry, target: MLModelRegistry) -> None:
    for key, value in _model_registry_values(source).items():
        setattr(target, key, value)


def _model_registry_values(source: MLModelRegistry) -> dict[str, Any]:
    values = {
        column.name: getattr(source, column.name)
        for column in MLModelRegistry.__table__.columns
    }
    values["job_run_id"] = None
    return values


def _context_artifact_hash(path: str) -> str:
    return stable_hash(Path(path).read_text())


def _job_result_payload(result: Any) -> dict[str, Any]:
    return {
        "status": result.status,
        "metrics": result.metrics or {},
        "errors": result.errors or [],
    }


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_datetime(value: str) -> datetime:
    raw = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("--asof must be timezone-aware")
    return parsed


def _local_trading_date(value: datetime) -> date:
    return value.astimezone(EASTERN).date()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the scratch-bound, read-only I12 Stage-0 fill test."
    )
    parser.add_argument("--database-url")
    parser.add_argument("--schema", help="Required scratch schema for any DB write.")
    parser.add_argument("--create-tables", action="store_true")
    parser.add_argument("--context-artifact")
    parser.add_argument("--trading-date")
    parser.add_argument("--model-id")
    parser.add_argument("--allow-latest-model", action="store_true")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--intended-order-usd", type=float, default=DEFAULT_INTENDED_ORDER_USD)
    parser.add_argument("--max-spread-bps", type=float, default=DEFAULT_MAX_SPREAD_BPS)
    parser.add_argument("--feed", default="sip", choices=("iex", "sip", "otc"))
    parser.add_argument("--max-quote-age-seconds", type=float, default=DEFAULT_MAX_QUOTE_AGE_SECONDS)
    parser.add_argument("--max-snapshot-age-seconds", type=float, default=DEFAULT_MAX_SNAPSHOT_AGE_SECONDS)
    parser.add_argument("--min-context-count", type=int, default=DEFAULT_MIN_CONTEXT_COUNT)
    parser.add_argument("--min-intended-count", type=int, default=DEFAULT_MIN_INTENDED_COUNT)
    parser.add_argument("--min-snapshot-ok-rate", type=float, default=DEFAULT_MIN_SNAPSHOT_OK_RATE)
    parser.add_argument(
        "--max-snapshot-error-or-missing-rate",
        type=float,
        default=DEFAULT_MAX_SNAPSHOT_ERROR_OR_MISSING_RATE,
    )
    parser.add_argument("--min-score-model-ok-rate", type=float, default=DEFAULT_MIN_SCORE_MODEL_OK_RATE)
    parser.add_argument("--min-quote-ok-rate", type=float, default=DEFAULT_MIN_QUOTE_OK_RATE)
    parser.add_argument("--min-exit-quote-ok-rate", type=float, default=DEFAULT_MIN_EXIT_QUOTE_OK_RATE)
    parser.add_argument(
        "--min-gate0-intended-count",
        type=int,
        default=DEFAULT_MIN_GATE0_INTENDED_COUNT,
    )
    parser.add_argument(
        "--min-gate0-distinct-trading-days",
        type=int,
        default=DEFAULT_MIN_GATE0_DISTINCT_TRADING_DAYS,
    )
    parser.add_argument(
        "--min-gate0-tradeable-rate",
        type=float,
        default=DEFAULT_MIN_GATE0_TRADEABLE_RATE,
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=float, default=60.0)
    parser.add_argument("--allow-market-closed", action="store_true")
    parser.add_argument("--asof")
    parser.add_argument("--exit-quotes", action="store_true")
    parser.add_argument("--gate0-report", action="store_true")
    parser.add_argument("--fail-on-gate0-fail", action="store_true")
    parser.add_argument(
        "--alpaca-probe-symbol",
        help="Read-only quote probe; does not require --schema or write to DB.",
    )
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.intended_order_usd <= 0:
        parser.error("--intended-order-usd must be positive")
    if args.max_spread_bps <= 0:
        parser.error("--max-spread-bps must be positive")
    if args.max_quote_age_seconds <= 0:
        parser.error("--max-quote-age-seconds must be positive")
    if args.max_snapshot_age_seconds <= 0:
        parser.error("--max-snapshot-age-seconds must be positive")
    if args.min_context_count < 0:
        parser.error("--min-context-count must be >= 0")
    if args.min_intended_count < 0:
        parser.error("--min-intended-count must be >= 0")
    if args.min_gate0_intended_count < 0:
        parser.error("--min-gate0-intended-count must be >= 0")
    if args.min_gate0_distinct_trading_days < 0:
        parser.error("--min-gate0-distinct-trading-days must be >= 0")
    for rate_name in (
        "min_snapshot_ok_rate",
        "max_snapshot_error_or_missing_rate",
        "min_score_model_ok_rate",
        "min_quote_ok_rate",
        "min_exit_quote_ok_rate",
        "min_gate0_tradeable_rate",
    ):
        if getattr(args, rate_name) < 0 or getattr(args, rate_name) > 1:
            parser.error(f"--{rate_name.replace('_', '-')} must be between 0 and 1")
    if args.poll_interval_seconds <= 0:
        parser.error("--poll-interval-seconds must be positive")
    if args.alpaca_probe_symbol:
        return args
    if not args.exit_quotes and not args.gate0_report and not args.context_artifact:
        parser.error("--context-artifact is required unless --exit-quotes or --gate0-report")
    if not args.exit_quotes and not args.gate0_report:
        if not args.model_id and not args.allow_latest_model:
            parser.error("--model-id is required unless --allow-latest-model is set")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    return _run(_parse_args(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
