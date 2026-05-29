"""Guard and health-report tests for the canonical M4 accumulation runner."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from alpha.db.models import (
    CanonicalUniverseScan,
    EvidenceJob,
    EvidenceJobRun,
    FeatureSnapshot,
    ForwardReturnObservation,
    SignalRegistry,
    UniverseScan,
)
from alpha.jobs import run_m4_canonical
from alpha.jobs.m4_daily import M4DailyAssemblyJob
from alpha.jobs.run_m4_canonical import (
    CanonicalRunError,
    build_m4_health_report,
    require_canonical_target,
    require_scratch_schema,
    resolve_canonical_clock,
)
from alpha.jobs.runner import run_job

from tests.test_m4_daily_job import (
    FakeBenzingaContextAdapter,
    FakeHistoricalAdapter,
    FakePolygonContextAdapter,
    _adapter_response,
    _m4_breakout_bars,
    _setup_canonical_universe,
)


def _utc(*parts) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


_DEFAULT_JOB_RUN = object()


# ---------------------------------------------------------------------------
# (a) clock determinism
# ---------------------------------------------------------------------------
def test_clock_is_deterministic_for_fixed_run_timestamp():
    run_ts = _utc(2026, 5, 26, 8, 0)
    first = resolve_canonical_clock(run_ts, None)
    second = resolve_canonical_clock(run_ts, None)
    assert first == second
    assert first["decision_date"] == "2026-05-26"
    assert first["evidence_session_date"] == "2026-05-22"


def test_clock_anchors_evidence_on_explicit_decision_date():
    run_ts = _utc(2026, 5, 26, 8, 0)
    clock = resolve_canonical_clock(run_ts, "2026-05-22")
    assert clock["decision_date"] == "2026-05-22"
    assert clock["evidence_session_date"] == "2026-05-22"
    # effective run timestamp anchored to that session's close (20:00 UTC = 16:00 ET)
    assert clock["effective_run_timestamp"].startswith("2026-05-22T20:00:00")


def test_health_report_exposes_pinned_evidence_asof_timestamp(db_session):
    _add_feature(db_session, "feature-asof", "ASOF")
    _add_signal(db_session, "signal-asof", "feature-asof", "ASOF")

    report = build_m4_health_report(
        db_session,
        mode="scratch",
        schema="m4_live_scratch_test",
        host_class="postgres_other",
        app_commit_sha="deadbeef",
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
        next_execution_session="2026-05-26",
        run_timestamp="2026-05-22T20:00:00+00:00",
    )

    assert report["run_metadata"]["asof_timestamp"] == "2026-05-22T20:00:00+00:00"


def test_explicit_decision_date_rerun_keeps_identical_asof_timestamp(db_session):
    run_ts = _utc(2026, 5, 29, 20, 0)
    first = resolve_canonical_clock(run_ts, "2026-05-22")
    second = resolve_canonical_clock(_utc(2026, 6, 1, 20, 0), "2026-05-22")

    assert first["effective_run_timestamp"] == second["effective_run_timestamp"]
    assert first["evidence_session_date"] == second["evidence_session_date"] == "2026-05-22"


# ---------------------------------------------------------------------------
# (b) future decision date rejection
# ---------------------------------------------------------------------------
def test_future_decision_date_is_refused():
    run_ts = _utc(2026, 5, 26, 8, 0)
    with pytest.raises(CanonicalRunError, match="future"):
        resolve_canonical_clock(run_ts, "2026-05-29")


# ---------------------------------------------------------------------------
# (c) weekend / holiday decision date rejection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_day", ["2026-05-24", "2026-05-25"])  # Sunday, Memorial Day
def test_weekend_or_holiday_decision_date_is_refused(bad_day):
    run_ts = _utc(2026, 5, 27, 8, 0)
    with pytest.raises(CanonicalRunError, match="trading session"):
        resolve_canonical_clock(run_ts, bad_day)


def test_invalid_decision_date_string_is_refused():
    run_ts = _utc(2026, 5, 26, 8, 0)
    with pytest.raises(CanonicalRunError, match="valid ISO date"):
        resolve_canonical_clock(run_ts, "not-a-date")


# ---------------------------------------------------------------------------
# (d) --live requires --confirm-canonical-write
# ---------------------------------------------------------------------------
def test_live_without_confirm_is_refused(monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db.example.supabase.co/postgres")
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    rc = run_m4_canonical.main(["--live", "--run-timestamp", "2026-05-26T08:00:00+00:00"])
    assert rc == 1
    assert "--confirm-canonical-write" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# (e) --live refuses SQLite
# ---------------------------------------------------------------------------
def test_live_refuses_sqlite(monkeypatch):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    with pytest.raises(CanonicalRunError, match="PostgreSQL"):
        require_canonical_target("sqlite:///alpha_capital.db")


def test_live_cli_refuses_sqlite(monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///alpha_capital.db")
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    rc = run_m4_canonical.main([
        "--live",
        "--confirm-canonical-write",
        "--run-timestamp", "2026-05-26T08:00:00+00:00",
    ])
    assert rc == 1
    assert "SQLite is refused" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# (f) --live refuses a set ALPHA_DB_SCHEMA
# ---------------------------------------------------------------------------
def test_live_refuses_alpha_db_schema_env(monkeypatch):
    monkeypatch.setenv("ALPHA_DB_SCHEMA", "m4_live_scratch_x")
    with pytest.raises(CanonicalRunError, match="ALPHA_DB_SCHEMA"):
        require_canonical_target("postgresql+psycopg://u:p@host/db")


# ---------------------------------------------------------------------------
# (g) --scratch requires --schema
# ---------------------------------------------------------------------------
def test_scratch_requires_schema():
    with pytest.raises(CanonicalRunError, match="--schema"):
        require_scratch_schema(None, "postgresql+psycopg://u:p@host/db")


def test_scratch_refuses_public_schema():
    with pytest.raises(CanonicalRunError, match="public"):
        require_scratch_schema("public", "postgresql+psycopg://u:p@host/db")


def test_scratch_cli_without_schema_is_refused(monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@host/db")
    rc = run_m4_canonical.main(["--scratch", "--run-timestamp", "2026-05-26T08:00:00+00:00"])
    assert rc == 1
    assert "--schema" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# (h) --dry-run performs zero writes and zero API calls
# ---------------------------------------------------------------------------
def test_dry_run_makes_no_writes_or_calls(monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@host/db")
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)

    def _explode(*_a, **_k):  # pragma: no cover - must never be called
        raise AssertionError("dry-run must not orchestrate universe/M4")

    monkeypatch.setattr(run_m4_canonical, "_run_universe", _explode)
    monkeypatch.setattr(run_m4_canonical, "_run_m4", _explode)
    monkeypatch.setattr(run_m4_canonical, "_report_session", _explode)

    rc = run_m4_canonical.main(["--dry-run", "--run-timestamp", "2026-05-26T08:00:00+00:00"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no database writes and no provider API calls" in out


# ---------------------------------------------------------------------------
# helpers to populate a session via the real M4 daily job + mocked providers
# ---------------------------------------------------------------------------
class CleanPolygonContextAdapter(FakePolygonContextAdapter):
    """Polygon stub with no provider errors (clean news instead of HTTP 500)."""

    def get_news(self, **kwargs):
        self.calls.append(("get_news", kwargs))
        return _adapter_response(
            provider="Polygon",
            endpoint="/v2/reference/news",
            asof=kwargs["asof"],
            data=[],
        )


class CleanBenzingaContextAdapter(FakeBenzingaContextAdapter):
    """Benzinga stub with no provider/validation errors."""

    def get_ratings(self, **kwargs):
        self.calls.append(("get_ratings", kwargs))
        return _adapter_response(
            provider="Benzinga",
            endpoint="/api/v2.1/calendar/ratings",
            asof=kwargs["asof"],
            data=[],
        )


def _run_m4_daily_into(db_session, *, polygon=None, benzinga=None):
    _setup_canonical_universe(db_session)
    adapter = FakeHistoricalAdapter(_m4_breakout_bars(date(2026, 5, 22)))
    job = M4DailyAssemblyJob(
        db_session,
        adapter=adapter,
        polygon_adapter=polygon,
        benzinga_adapter=benzinga,
        run_timestamp=_utc(2026, 5, 26, 8, 0),
    )
    result = run_job(db_session, job, params={})
    assert result.ok, result.errors
    return result


def _build_report(db_session, *, m4_metrics=None):
    return build_m4_health_report(
        db_session,
        mode="scratch",
        schema="m4_live_scratch_test",
        host_class="postgres_other",
        app_commit_sha="deadbeef",
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
        next_execution_session="2026-05-26",
        run_timestamp="2026-05-26T08:00:00+00:00",
        m4_metrics=m4_metrics,
    )


def _minimal_context(*, status: str = "matched"):
    return {
        "schema_version": "m4-signal-context-v1",
        "asof_timestamp": "2026-05-22T20:00:00+00:00",
        "identity": {
            "status": status,
            "source_attempts": [{"source": "Polygon identity", "status": status}],
        },
    }


def _ensure_test_m4_run(
    db_session,
    *,
    run_id: str = "test-current-m4-run",
    run_status: str = "finished",
    decision_date: str = "2026-05-26",
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    metrics: dict | None = None,
):
    job = (
        db_session.query(EvidenceJob)
        .filter(EvidenceJob.job_name == "m4_daily_feature_assembly")
        .first()
    )
    if job is None:
        job = EvidenceJob(
            job_id="test-m4-job",
            job_name="m4_daily_feature_assembly",
            job_type="feature_assembly",
            owner_component="test",
        )
        db_session.add(job)
        db_session.flush()
    run = db_session.get(EvidenceJobRun, run_id)
    if run is None:
        metric_json = metrics or {
            "decision_date": decision_date,
            "included_universe_size": 1,
            "fetched_symbol_count": 1,
            "fetched_bar_count": 61,
            "fetch_error_count": 0,
            "assembly": {"assembled_count": 1},
            "signal_context": {
                "context_attached_count": 1,
                "context_reused_count": 0,
                "context_reused_from_persistence_count": 0,
                "context_reused_in_memory_count": 0,
                "context_enriched_count": 1,
                "context_persistence_miss_count": 1,
                "context_persistence_mismatch_count": 0,
                "context_persistence_mismatch_reasons": {},
            },
        }
        run = EvidenceJobRun(
            job_run_id=run_id,
            job_id=job.job_id,
            run_status=run_status,
            started_at=started_at or _utc(2026, 5, 26, 20, 1),
            ended_at=ended_at or _utc(2026, 5, 26, 20, 2),
            metric_json=json.dumps(metric_json, sort_keys=True),
            params_json=json.dumps({"run_timestamp": "2026-05-26T20:00:00+00:00"}),
        )
        db_session.add(run)
        db_session.flush()
    return run.job_run_id


def _add_feature(
    db_session,
    feature_id: str,
    ticker: str,
    *,
    asof: datetime | None = None,
    payload: dict | None = None,
    job_run_id: str | None | object = _DEFAULT_JOB_RUN,
):
    payload = payload or {"signal_context": _minimal_context()}
    if job_run_id is _DEFAULT_JOB_RUN:
        job_run_id = _ensure_test_m4_run(db_session)
    feature = FeatureSnapshot(
        feature_snapshot_id=feature_id,
        job_run_id=job_run_id,
        pattern_id="M4",
        ticker=ticker,
        asof_timestamp=asof or _utc(2026, 5, 22, 20, 0),
        feature_manifest_version="test",
        feature_json=json.dumps(payload, sort_keys=True),
        feature_hash=f"hash-{feature_id}",
        data_lineage_ids=json.dumps([f"lineage-{feature_id}"]),
        fidelity_tier="test",
        point_in_time_passed=True,
        lookahead_guard_passed=True,
    )
    db_session.add(feature)
    db_session.flush()
    return feature


def _add_signal(
    db_session,
    signal_id: str,
    feature_id: str,
    ticker: str,
    *,
    trading_date: str = "2026-05-26",
    identity_hash: str | None = None,
    forward_return_status: str = "pending",
):
    signal = SignalRegistry(
        signal_id=signal_id,
        pattern_id="M4",
        ticker=ticker,
        direction="long",
        signal_timestamp=_utc(2026, 5, 22, 20, 0),
        raw_signal_strength=1.0,
        raw_expected_edge=0.1,
        signal_horizon="15d",
        thesis_category="M4",
        route_class="base",
        fidelity_tier="test",
        data_confidence=1.0,
        feature_snapshot_id=feature_id,
        signal_status="active",
        trading_date=trading_date,
        next_execution_session="2026-05-26",
        detector_version="test",
        point_in_time_passed=True,
        lookahead_guard_passed=True,
        data_lineage_ids=json.dumps([f"lineage-{feature_id}"]),
        signal_identity_hash=identity_hash or f"identity-{signal_id}",
        forward_return_status=forward_return_status,
    )
    db_session.add(signal)
    db_session.flush()
    return signal


def _add_forward_observation(db_session, observation_id: str, signal: SignalRegistry):
    obs = ForwardReturnObservation(
        forward_return_observation_id=observation_id,
        signal_id=signal.signal_id,
        pattern_id="M4",
        ticker=signal.ticker,
        direction=signal.direction,
        signal_timestamp=signal.signal_timestamp,
        signal_horizon=signal.signal_horizon,
        next_execution_session=signal.next_execution_session,
        entry_session_date=signal.next_execution_session,
        entry_price=1.0,
        entry_price_source="test",
        status="computed",
        input_hash=f"input-{observation_id}",
        outcome_hash=f"outcome-{observation_id}",
    )
    db_session.add(obs)
    db_session.flush()
    return obs


# ---------------------------------------------------------------------------
# (i) health report contains every required section
# ---------------------------------------------------------------------------
def test_health_report_has_all_sections(db_session):
    _run_m4_daily_into(
        db_session,
        polygon=CleanPolygonContextAdapter(),
        benzinga=CleanBenzingaContextAdapter(),
    )
    report = _build_report(db_session)
    for section in (
        "run_metadata",
        "universe",
        "m4_assembly",
        "m4_signals",
        "data_quality",
        "forward_return_guard",
        "source_attempts",
        "freeze_reuse",
        "idempotency_rerun",
        "run_diagnostics",
        "fired_signal_table",
        "health_verdict",
    ):
        assert section in report, section
    assert report["run_metadata"]["app_commit_sha"] == "deadbeef"
    assert report["m4_signals"]["signal_count"] == 1
    assert report["fired_signal_table"][0]["ticker"] == "LCUT"
    assert report["fired_signal_table"][0]["has_signal_context"] is True


# ---------------------------------------------------------------------------
# (j) clean run is healthy
# ---------------------------------------------------------------------------
def test_clean_run_is_healthy(db_session):
    result = _run_m4_daily_into(
        db_session,
        polygon=CleanPolygonContextAdapter(),
        benzinga=CleanBenzingaContextAdapter(),
    )
    report = _build_report(db_session, m4_metrics=result.metrics)
    assert report["health"] is True
    assert report["health_verdict"]["failing_checks"] == []
    assert report["freeze_reuse"]["fired_missing_signal_context_count"] == 0


# ---------------------------------------------------------------------------
# (k) leaks / errors / missing context flip the verdict unhealthy
# ---------------------------------------------------------------------------
def test_no_signals_is_unhealthy(db_session):
    report = _build_report(db_session)
    assert report["health"] is False
    assert "has_signals" in report["health_verdict"]["failing_checks"]


def test_provider_error_is_unhealthy(db_session):
    # default FakePolygonContextAdapter.get_news returns a provider_error
    _run_m4_daily_into(
        db_session,
        polygon=FakePolygonContextAdapter(),
        benzinga=CleanBenzingaContextAdapter(),
    )
    report = _build_report(db_session)
    assert report["source_attempts"]["signal_context_provider_error_count"] >= 1
    assert report["health"] is False
    assert "no_fired_signal_context_errors" in report["health_verdict"]["failing_checks"]


def test_secret_leak_is_unhealthy(db_session):
    _run_m4_daily_into(
        db_session,
        polygon=CleanPolygonContextAdapter(),
        benzinga=CleanBenzingaContextAdapter(),
    )
    from alpha.db.models import FeatureSnapshot

    feature = db_session.query(FeatureSnapshot).filter_by(ticker="LCUT").one()
    payload = json.loads(feature.feature_json)
    payload["leak"] = {"apiKey": "should-not-be-here"}
    feature.feature_json = json.dumps(payload, sort_keys=True)
    db_session.flush()

    report = _build_report(db_session)
    assert report["data_quality"]["feature_json_api_key_hits"] == 1
    assert report["health"] is False
    assert "no_secret_leaks" in report["health_verdict"]["failing_checks"]


def test_env_secret_value_in_feature_json_is_detected(db_session, monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "TOPSECRETVALUE123")
    _run_m4_daily_into(
        db_session,
        polygon=CleanPolygonContextAdapter(),
        benzinga=CleanBenzingaContextAdapter(),
    )
    from alpha.db.models import FeatureSnapshot

    feature = db_session.query(FeatureSnapshot).filter_by(ticker="LCUT").one()
    payload = json.loads(feature.feature_json)
    payload["leaked"] = "embedding TOPSECRETVALUE123 here"
    feature.feature_json = json.dumps(payload, sort_keys=True)
    db_session.flush()

    report = _build_report(db_session)
    assert report["data_quality"]["feature_json_secret_value_hits"] == 1
    assert report["health"] is False


@pytest.mark.parametrize(
    "title",
    [
        "Company Secret Sauce",
        "Bearer Capital raises outlook",
        "tokenized_context update",
        "authorization platform",
    ],
)
def test_legitimate_secret_words_do_not_fail_health(db_session, title):
    payload = {
        "signal_context": {
            "benzinga_news": {
                "status": "matched",
                "latest_title": title,
                "source_attempts": [{"source": "Benzinga news", "status": "matched"}],
            }
        }
    }
    _add_feature(db_session, "feature-title", "TITLE", payload=payload)
    _add_signal(db_session, "signal-title", "feature-title", "TITLE")

    report = _build_report(db_session)

    assert report["data_quality"]["secret_hit_total"] == 0
    assert report["health"] is True


@pytest.mark.parametrize(
    "payload,field",
    [
        ({"signal_context": {"apiKey": "redacted-marker"}}, "feature_json_secret_key_hits"),
        (
            {"signal_context": {"latest_url": "https://x.test/p?apiKey=redacted-marker"}},
            "feature_json_secret_url_fragment_hits",
        ),
        (
            {"signal_context": {"header": "Authorization: Bearer SECRETTOKENVALUE123456"}},
            "feature_json_secret_url_fragment_hits",
        ),
    ],
)
def test_structured_secret_leaks_fail_health_without_printing_values(db_session, capsys, payload, field):
    _add_feature(db_session, "feature-secret", "SEC", payload=payload)
    _add_signal(db_session, "signal-secret", "feature-secret", "SEC")

    report = _build_report(db_session)
    run_m4_canonical._print_report(report)
    rendered = json.dumps(report, sort_keys=True)
    out = capsys.readouterr().out

    assert report["data_quality"][field] == 1
    assert report["health"] is False
    assert "no_secret_leaks" in report["health_verdict"]["failing_checks"]
    assert "SECRETTOKENVALUE123456" not in rendered
    assert "SECRETTOKENVALUE123456" not in out


def test_exact_env_secret_value_fails_health_without_printing_value(db_session, monkeypatch, capsys):
    monkeypatch.setenv("FMP_API_KEY", "FAKESECRET123456")
    payload = {"signal_context": {"benzinga_news": {"latest_title": "FAKESECRET123456"}}}
    _add_feature(db_session, "feature-env-secret", "ENV", payload=payload)
    _add_signal(db_session, "signal-env-secret", "feature-env-secret", "ENV")

    report = _build_report(db_session)
    run_m4_canonical._print_report(report)
    rendered = json.dumps(report, sort_keys=True)
    out = capsys.readouterr().out

    assert report["data_quality"]["feature_json_secret_value_hits"] == 1
    assert report["health"] is False
    assert "FAKESECRET123456" not in rendered
    assert "FAKESECRET123456" not in out


def test_short_env_secret_value_does_not_fail_health(db_session, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "SHORTKEY")
    payload = {
        "signal_context": {
            "benzinga_news": {
                "status": "matched",
                "latest_title": "SHORTKEY",
                "source_attempts": [{"source": "Benzinga news", "status": "matched"}],
            }
        }
    }
    _add_feature(db_session, "feature-short-env-secret", "SENV", payload=payload)
    _add_signal(db_session, "signal-short-env-secret", "feature-short-env-secret", "SENV")

    report = _build_report(db_session)

    assert report["data_quality"]["feature_json_secret_value_hits"] == 0
    assert report["health"] is True


def test_long_env_secret_value_still_fails_health(db_session, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "LONGSECRET123456")
    payload = {
        "signal_context": {
            "benzinga_news": {
                "status": "matched",
                "latest_title": "LONGSECRET123456",
                "source_attempts": [{"source": "Benzinga news", "status": "matched"}],
            }
        }
    }
    _add_feature(db_session, "feature-long-env-secret", "LENV", payload=payload)
    _add_signal(db_session, "signal-long-env-secret", "feature-long-env-secret", "LENV")

    report = _build_report(db_session)

    assert report["data_quality"]["feature_json_secret_value_hits"] == 1
    assert report["health"] is False
    assert "no_secret_leaks" in report["health_verdict"]["failing_checks"]


def test_raw_payload_text_value_does_not_fail_health(db_session):
    payload = {
        "signal_context": {
            "benzinga_news": {
                "status": "matched",
                "latest_title": "article says raw_payload is an engineering term",
                "source_attempts": [{"source": "Benzinga news", "status": "matched"}],
            }
        }
    }
    _add_feature(db_session, "feature-raw-text", "RAWT", payload=payload)
    _add_signal(db_session, "signal-raw-text", "feature-raw-text", "RAWT")

    report = _build_report(db_session)

    assert report["data_quality"]["feature_json_raw_payload_hits"] == 0
    assert report["data_quality"]["secret_hit_total"] == 0
    assert report["health"] is True


def test_raw_payload_key_still_fails_health(db_session):
    payload = {"signal_context": {"raw_payload": {"x": 1}}}
    _add_feature(db_session, "feature-raw-key", "RAWK", payload=payload)
    _add_signal(db_session, "signal-raw-key", "feature-raw-key", "RAWK")

    report = _build_report(db_session)

    assert report["data_quality"]["feature_json_raw_payload_hits"] == 1
    assert report["health"] is False
    assert "no_secret_leaks" in report["health_verdict"]["failing_checks"]


def test_non_fired_raw_payload_object_fails_health_globally(db_session):
    _add_feature(db_session, "feature-clean-fired", "CLEAN", payload={"signal_context": _minimal_context()})
    _add_signal(db_session, "signal-clean-fired", "feature-clean-fired", "CLEAN")
    _add_feature(
        db_session,
        "feature-non-fired-raw-payload",
        "RAWNF",
        payload={"signal_context": {"raw_payload": {"provider": "debug"}}},
    )

    report = _build_report(db_session)

    assert report["data_quality"]["feature_json_raw_payload_hits"] == 1
    assert report["health"] is False
    assert "no_secret_leaks" in report["health_verdict"]["failing_checks"]


# ---------------------------------------------------------------------------
# (l) forward-return guard: report always shows zero rows created
# ---------------------------------------------------------------------------
def test_forward_return_guard_reports_zero(db_session):
    _run_m4_daily_into(
        db_session,
        polygon=CleanPolygonContextAdapter(),
        benzinga=CleanBenzingaContextAdapter(),
    )
    report = _build_report(db_session)
    guard = report["forward_return_guard"]
    assert guard["forward_return_rows_created"] == 0
    assert guard["forward_return_observation_count"] == 0
    assert guard["forward_returns_run"] is False
    assert guard["forward_return_contamination_detected"] is False


def test_forward_return_contamination_flag_tracks_rows(db_session):
    _add_feature(db_session, "feature-current-forward-flag", "FRF")
    signal = _add_signal(db_session, "signal-current-forward-flag", "feature-current-forward-flag", "FRF")

    clean = _build_report(db_session)
    assert clean["forward_return_guard"]["forward_return_contamination_detected"] is False

    _add_forward_observation(db_session, "current-forward-flag", signal)
    contaminated = _build_report(db_session)

    assert contaminated["forward_return_guard"]["forward_return_rows_created"] == 1
    assert contaminated["forward_return_guard"]["forward_returns_run"] is False
    assert contaminated["forward_return_guard"]["forward_return_contamination_detected"] is True
    assert contaminated["health"] is False
    assert "no_forward_return_rows" in contaminated["health_verdict"]["failing_checks"]


# ---------------------------------------------------------------------------
# (m) idempotent rerun produces no new signals and reuses frozen context
# ---------------------------------------------------------------------------
def test_idempotent_rerun_adds_no_signals(db_session):
    _setup_canonical_universe(db_session)
    adapter = FakeHistoricalAdapter(_m4_breakout_bars(date(2026, 5, 22)))

    first = M4DailyAssemblyJob(
        db_session,
        adapter=adapter,
        polygon_adapter=CleanPolygonContextAdapter(),
        benzinga_adapter=CleanBenzingaContextAdapter(),
        run_timestamp=_utc(2026, 5, 26, 8, 0),
    )
    assert run_job(db_session, first, params={}).ok
    report_one = _build_report(db_session)

    second_polygon = CleanPolygonContextAdapter()
    second_benzinga = CleanBenzingaContextAdapter()
    second = M4DailyAssemblyJob(
        db_session,
        adapter=adapter,
        polygon_adapter=second_polygon,
        benzinga_adapter=second_benzinga,
        run_timestamp=_utc(2026, 5, 26, 8, 0),
    )
    second_result = run_job(db_session, second, params={})
    assert second_result.ok
    report_two = _build_report(db_session, m4_metrics=second_result.metrics)

    # no new signals, frozen context reused without re-calling providers
    assert report_two["m4_signals"]["signal_count"] == report_one["m4_signals"]["signal_count"] == 1
    assert second_polygon.calls == []
    assert second_benzinga.calls == []
    assert report_two["health"] is True


def test_primary_persistence_mismatch_fails_health(db_session):
    run_id = _ensure_test_m4_run(db_session, run_id="primary-mismatch-run")
    _add_feature(db_session, "feature-primary-mismatch", "PMIS", job_run_id=run_id)
    _add_signal(db_session, "signal-primary-mismatch", "feature-primary-mismatch", "PMIS")

    report = build_m4_health_report(
        db_session,
        mode="scratch",
        schema="m4_live_scratch_test",
        host_class="postgres_other",
        app_commit_sha="deadbeef",
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
        next_execution_session="2026-05-26",
        run_timestamp="2026-05-26T08:00:00+00:00",
        primary_m4_run_id=run_id,
        m4_metrics={
            "decision_date": "2026-05-26",
            "assembly": {"assembled_count": 1},
            "signal_context": {"context_persistence_mismatch_count": 1},
        },
    )

    assert report["freeze_reuse"]["context_persistence_mismatch_count"] == 1
    assert report["health"] is False
    assert "no_persistence_mismatch" in report["health_verdict"]["failing_checks"]


def test_rerun_persistence_mismatch_fails_health(db_session):
    run_id = _ensure_test_m4_run(db_session, run_id="rerun-mismatch-run")
    _add_feature(db_session, "feature-rerun-mismatch", "RMIS", job_run_id=run_id)
    _add_signal(db_session, "signal-rerun-mismatch", "feature-rerun-mismatch", "RMIS")

    report = build_m4_health_report(
        db_session,
        mode="scratch",
        schema="m4_live_scratch_test",
        host_class="postgres_other",
        app_commit_sha="deadbeef",
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
        next_execution_session="2026-05-26",
        run_timestamp="2026-05-26T08:00:00+00:00",
        primary_m4_run_id=run_id,
        rerun_m4_run_id="rerun-mismatch-id",
        m4_metrics={"decision_date": "2026-05-26", "assembly": {"assembled_count": 1}},
        rerun_m4_metrics={
            "decision_date": "2026-05-26",
            "signal_context": {"context_persistence_mismatch_count": 2},
        },
    )

    assert report["idempotency_rerun"]["context_persistence_mismatch_count"] == 2
    assert report["health"] is False
    assert "no_persistence_mismatch" in report["health_verdict"]["failing_checks"]


def test_string_persistence_mismatch_count_fails_health(db_session):
    run_id = _ensure_test_m4_run(db_session, run_id="string-mismatch-run")
    _add_feature(db_session, "feature-string-mismatch", "SMIS", job_run_id=run_id)
    _add_signal(db_session, "signal-string-mismatch", "feature-string-mismatch", "SMIS")

    report = build_m4_health_report(
        db_session,
        mode="scratch",
        schema="m4_live_scratch_test",
        host_class="postgres_other",
        app_commit_sha="deadbeef",
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
        next_execution_session="2026-05-26",
        run_timestamp="2026-05-26T08:00:00+00:00",
        primary_m4_run_id=run_id,
        m4_metrics={
            "decision_date": "2026-05-26",
            "assembly": {"assembled_count": 1},
            "signal_context": {"context_persistence_mismatch_count": "2"},
        },
    )

    assert report["freeze_reuse"]["context_persistence_mismatch_count"] == 2
    assert report["health"] is False
    assert "no_persistence_mismatch" in report["health_verdict"]["failing_checks"]


def test_malformed_persistence_mismatch_count_does_not_crash(db_session):
    run_id = _ensure_test_m4_run(db_session, run_id="malformed-mismatch-run")
    _add_feature(db_session, "feature-malformed-mismatch", "BMIS", job_run_id=run_id)
    _add_signal(db_session, "signal-malformed-mismatch", "feature-malformed-mismatch", "BMIS")

    for value in ("bad", [], {}, None, False):
        report = build_m4_health_report(
            db_session,
            mode="scratch",
            schema="m4_live_scratch_test",
            host_class="postgres_other",
            app_commit_sha="deadbeef",
            decision_date="2026-05-26",
            evidence_session_date="2026-05-22",
            next_execution_session="2026-05-26",
            run_timestamp="2026-05-26T08:00:00+00:00",
            primary_m4_run_id=run_id,
            m4_metrics={
                "decision_date": "2026-05-26",
                "assembly": {"assembled_count": 1},
                "signal_context": {"context_persistence_mismatch_count": value},
            },
        )

        assert report["freeze_reuse"]["context_persistence_mismatch_count"] == 0
        assert report["health"] is True
        assert "no_persistence_mismatch" not in report["health_verdict"]["failing_checks"]

    missing = build_m4_health_report(
        db_session,
        mode="scratch",
        schema="m4_live_scratch_test",
        host_class="postgres_other",
        app_commit_sha="deadbeef",
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
        next_execution_session="2026-05-26",
        run_timestamp="2026-05-26T08:00:00+00:00",
        primary_m4_run_id=run_id,
        m4_metrics={
            "decision_date": "2026-05-26",
            "assembly": {"assembled_count": 1},
            "signal_context": {},
        },
    )

    assert missing["freeze_reuse"]["context_persistence_mismatch_count"] == 0
    assert missing["health"] is True
    assert "no_persistence_mismatch" not in missing["health_verdict"]["failing_checks"]


def test_zero_persistence_mismatch_passes_health(db_session):
    run_id = _ensure_test_m4_run(db_session, run_id="zero-mismatch-run")
    _add_feature(db_session, "feature-zero-mismatch", "ZMIS", job_run_id=run_id)
    _add_signal(db_session, "signal-zero-mismatch", "feature-zero-mismatch", "ZMIS")

    report = build_m4_health_report(
        db_session,
        mode="scratch",
        schema="m4_live_scratch_test",
        host_class="postgres_other",
        app_commit_sha="deadbeef",
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
        next_execution_session="2026-05-26",
        run_timestamp="2026-05-26T08:00:00+00:00",
        primary_m4_run_id=run_id,
        m4_metrics={
            "decision_date": "2026-05-26",
            "assembly": {"assembled_count": 1},
            "signal_context": {"context_persistence_mismatch_count": 0},
        },
        rerun_m4_metrics={"signal_context": {}},
    )

    assert report["freeze_reuse"]["context_persistence_mismatch_count"] == 0
    assert report["health"] is True
    assert "no_persistence_mismatch" not in report["health_verdict"]["failing_checks"]


def test_skip_rerun_reports_persistence_rerun_not_checked(db_session):
    run_id = _ensure_test_m4_run(db_session, run_id="skip-rerun-run")
    _add_feature(db_session, "feature-skip-rerun", "SKIP", job_run_id=run_id)
    _add_signal(db_session, "signal-skip-rerun", "feature-skip-rerun", "SKIP")

    skipped = build_m4_health_report(
        db_session,
        mode="scratch",
        schema="m4_live_scratch_test",
        host_class="postgres_other",
        app_commit_sha="deadbeef",
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
        next_execution_session="2026-05-26",
        run_timestamp="2026-05-26T08:00:00+00:00",
        primary_m4_run_id=run_id,
        m4_metrics={
            "decision_date": "2026-05-26",
            "assembly": {"assembled_count": 1},
            "signal_context": {"context_persistence_mismatch_count": 0},
        },
    )
    checked = build_m4_health_report(
        db_session,
        mode="scratch",
        schema="m4_live_scratch_test",
        host_class="postgres_other",
        app_commit_sha="deadbeef",
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
        next_execution_session="2026-05-26",
        run_timestamp="2026-05-26T08:00:00+00:00",
        primary_m4_run_id=run_id,
        rerun_m4_run_id="skip-rerun-rerun-id",
        m4_metrics={
            "decision_date": "2026-05-26",
            "assembly": {"assembled_count": 1},
            "signal_context": {"context_persistence_mismatch_count": 0},
        },
        rerun_m4_metrics={"signal_context": {"context_persistence_mismatch_count": 0}},
    )

    assert skipped["idempotency_rerun"]["persistence_rerun_checked"] is False
    assert skipped["health"] is True
    assert checked["idempotency_rerun"]["persistence_rerun_checked"] is True
    assert checked["health"] is True


def test_health_report_scoped_to_decision_date(db_session):
    _add_feature(
        db_session,
        "feature-old",
        "OLD",
        asof=_utc(2026, 5, 21, 20, 0),
        payload={"signal_context": _minimal_context()},
        job_run_id=None,
    )
    _add_signal(
        db_session,
        "signal-old",
        "feature-old",
        "OLD",
        trading_date="2026-05-21",
        identity_hash="old-identity",
    )
    _add_feature(
        db_session,
        "feature-current",
        "CUR",
        payload={"signal_context": _minimal_context()},
    )
    _add_signal(
        db_session,
        "signal-current",
        "feature-current",
        "CUR",
        identity_hash="current-identity",
    )

    report = _build_report(db_session)

    assert report["m4_signals"]["signal_count"] == 1
    assert report["m4_signals"]["fired_tickers"] == ["CUR"]
    assert report["m4_assembly"]["m4_feature_snapshots"] == 1
    assert report["fired_signal_table"][0]["ticker"] == "CUR"


def test_failed_same_date_run_features_are_diagnostics_only(db_session):
    failed_run = _ensure_test_m4_run(
        db_session,
        run_id="failed-same-date-run",
        run_status="failed",
        started_at=_utc(2026, 5, 26, 19, 0),
        ended_at=_utc(2026, 5, 26, 19, 5),
        metrics={
            "decision_date": "2026-05-26",
            "assembly": {"assembled_count": 1},
            "signal_context": {"provider_error_count": 1},
        },
    )
    clean_run = _ensure_test_m4_run(
        db_session,
        run_id="clean-current-run",
        run_status="finished",
        started_at=_utc(2026, 5, 26, 20, 0),
        ended_at=_utc(2026, 5, 26, 20, 5),
    )
    _add_feature(
        db_session,
        "feature-failed-run",
        "FAIL",
        job_run_id=failed_run,
        payload={
            "signal_context": {
                "polygon_news": {
                    "status": "provider_error",
                    "source_attempts": [{"source": "Polygon news", "status": "provider_error"}],
                }
            }
        },
    )
    _add_feature(db_session, "feature-clean-run", "CUR", job_run_id=clean_run)
    _add_signal(db_session, "signal-clean-run", "feature-clean-run", "CUR")

    report = build_m4_health_report(
        db_session,
        mode="scratch",
        schema="m4_live_scratch_test",
        host_class="postgres_other",
        app_commit_sha="deadbeef",
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
        next_execution_session="2026-05-26",
        run_timestamp="2026-05-26T08:00:00+00:00",
        primary_m4_run_id=clean_run,
    )

    assert report["health"] is True
    assert report["m4_assembly"]["m4_feature_snapshots"] == 1
    assert report["source_attempts"]["source_attempt_status_counts"] == {"matched": 1}
    assert report["run_diagnostics"]["failed_same_date_m4_run_count"] == 1
    assert report["run_diagnostics"]["failed_same_date_run_ids"] == ["failed-same-date-run"]


def test_current_fired_feature_included_with_run_id_mismatch_warning(db_session):
    current_run = _ensure_test_m4_run(db_session, run_id="current-run")
    _add_feature(db_session, "feature-mismatch", "MM", job_run_id=None)
    _add_signal(db_session, "signal-mismatch", "feature-mismatch", "MM")

    report = build_m4_health_report(
        db_session,
        mode="scratch",
        schema="m4_live_scratch_test",
        host_class="postgres_other",
        app_commit_sha="deadbeef",
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
        next_execution_session="2026-05-26",
        run_timestamp="2026-05-26T08:00:00+00:00",
        primary_m4_run_id=current_run,
    )

    assert report["m4_signals"]["signal_count"] == 1
    assert report["m4_assembly"]["m4_feature_snapshots"] == 1
    assert report["m4_signals"]["fired_feature_run_id_mismatch_count"] == 1
    assert report["m4_signals"]["fired_feature_run_id_mismatches"] == ["feature-mismatch"]
    assert report["health"] is True


def test_no_successful_m4_run_for_date_is_unhealthy(db_session):
    _add_feature(db_session, "feature-orphan", "ORPHAN", job_run_id=None)
    _add_signal(db_session, "signal-orphan", "feature-orphan", "ORPHAN")

    report = _build_report(db_session)

    assert report["health"] is False
    assert "has_successful_m4_run" in report["health_verdict"]["failing_checks"]


def test_old_forward_return_does_not_poison_current_health(db_session):
    _add_feature(
        db_session,
        "feature-old",
        "OLD",
        asof=_utc(2026, 5, 21, 20, 0),
        payload={"signal_context": _minimal_context()},
        job_run_id=None,
    )
    old_signal = _add_signal(
        db_session,
        "signal-old",
        "feature-old",
        "OLD",
        trading_date="2026-05-21",
        identity_hash="old-identity",
    )
    _add_forward_observation(db_session, "old-forward-return", old_signal)
    _add_feature(
        db_session,
        "feature-current",
        "CUR",
        payload={"signal_context": _minimal_context()},
    )
    _add_signal(
        db_session,
        "signal-current",
        "feature-current",
        "CUR",
        identity_hash="current-identity",
    )

    report = _build_report(db_session)

    assert report["forward_return_guard"]["forward_return_observation_count"] == 0
    assert report["forward_return_guard"]["forward_return_rows_created"] == 0
    assert report["health"] is True


def test_current_forward_return_fails_health(db_session):
    _add_feature(
        db_session,
        "feature-current",
        "CUR",
        payload={"signal_context": _minimal_context()},
    )
    signal = _add_signal(
        db_session,
        "signal-current",
        "feature-current",
        "CUR",
        identity_hash="current-identity",
    )
    _add_forward_observation(db_session, "current-forward-return", signal)

    report = _build_report(db_session)

    assert report["forward_return_guard"]["forward_return_observation_count"] == 1
    assert report["forward_return_guard"]["forward_return_rows_created"] == 1
    assert report["health"] is False
    assert "no_forward_return_rows" in report["health_verdict"]["failing_checks"]


def test_non_fired_context_provider_error_visible(db_session):
    _add_feature(
        db_session,
        "feature-fired",
        "CUR",
        payload={"signal_context": _minimal_context()},
    )
    _add_signal(
        db_session,
        "signal-current",
        "feature-fired",
        "CUR",
        identity_hash="current-identity",
    )
    _add_feature(
        db_session,
        "feature-non-fired-error",
        "ERR",
        payload={
            "signal_context": {
                "polygon_news": {
                    "status": "provider_error",
                    "source_attempts": [
                        {"source": "Polygon news", "status": "provider_error"}
                    ],
                }
            }
        },
    )

    report = _build_report(db_session)

    assert report["source_attempts"]["signal_context_provider_error_count"] == 1
    assert report["source_attempts"]["source_attempt_status_counts"]["provider_error"] == 1
    assert report["source_attempts"]["non_fired_source_attempt_status_counts"]["provider_error"] == 1
    assert report["source_attempts"]["fired_source_attempt_status_counts"].get("provider_error", 0) == 0
    assert report["health"] is True


@pytest.mark.parametrize("status", ["provider_error", "parse_error", "validation_error", "unavailable"])
def test_fired_context_hard_error_statuses_fail_health(db_session, status):
    payload = {
        "signal_context": {
            "polygon_news": {
                "status": status,
                "source_attempts": [{"source": "Polygon news", "status": status}],
            }
        }
    }
    _add_feature(db_session, "feature-fired-hard-error", "ERR", payload=payload)
    _add_signal(db_session, "signal-fired-hard-error", "feature-fired-hard-error", "ERR")

    report = _build_report(db_session)

    assert report["source_attempts"]["fired_source_attempt_status_counts"][status] == 1
    assert report["source_attempts"]["source_attempt_status_split"][status]["fired"] == 1
    assert report["health"] is False
    assert "no_fired_signal_context_errors" in report["health_verdict"]["failing_checks"]


@pytest.mark.parametrize("status", ["pit_excluded", "no_data"])
def test_pit_excluded_and_no_data_do_not_fail_health(db_session, status):
    payload = {
        "signal_context": {
            "polygon_news": {
                "status": status,
                "source_attempts": [{"source": "Polygon news", "status": status}],
            }
        }
    }
    _add_feature(db_session, "feature-benign-status", "BENIGN", payload=payload)
    _add_signal(db_session, "signal-benign-status", "feature-benign-status", "BENIGN")

    report = _build_report(db_session)

    assert report["source_attempts"]["fired_source_attempt_status_counts"][status] == 1
    assert report["health"] is True


def test_fired_table_includes_context_summaries(db_session):
    payload = {
        "X_M4": 1.23,
        "price": 12.5,
        "high_52w": 12.0,
        "signal_context": {
            "polygon_short_interest": {
                "status": "matched",
                "short_interest": 123456,
                "days_to_cover": 2.5,
                "source_attempts": [{"source": "Polygon short interest", "status": "matched"}],
            },
            "polygon_short_volume": {
                "status": "matched",
                "short_volume_ratio": 42.0,
                "source_attempts": [{"source": "Polygon short volume", "status": "matched"}],
            },
            "polygon_news": {"status": "matched", "article_count_90d": 3},
            "benzinga_news": {
                "status": "matched",
                "article_count_7d": 2,
                "wiim_count_7d": 1,
            },
            "benzinga_insider": {
                "status": "matched",
                "net_discretionary_shares": "100",
            },
            "benzinga_calendar": {
                "earnings": {"status": "matched"},
                "ratings": {"status": "no_data"},
                "offerings": {"status": "matched"},
                "dividends": {"status": "no_data"},
            },
        },
    }
    _add_feature(db_session, "feature-rich", "RICH", payload=payload)
    _add_signal(db_session, "signal-rich", "feature-rich", "RICH")

    report = _build_report(db_session)
    row = report["fired_signal_table"][0]

    assert row["X_M4"] == 1.23
    assert row["price"] == 12.5
    assert row["high_52w"] == 12.0
    assert row["short_interest"] == {
        "status": "matched",
        "short_interest": 123456,
        "days_to_cover": 2.5,
    }
    assert row["short_volume"]["short_volume_ratio"] == 42.0
    assert row["polygon_news"]["article_count_90d"] == 3
    assert row["benzinga_news"]["article_count_7d"] == 2
    assert row["benzinga_news"]["wiim_count_7d"] == 1
    assert row["insider"]["net_discretionary_shares"] == "100"
    assert row["calendar_status"] == {
        "earnings": "matched",
        "ratings": "no_data",
        "offerings": "matched",
        "dividends": "no_data",
    }
    assert row["source_attempt_status_counts"] == {"matched": 2}


def test_fired_table_matches_real_m4_daily_signal_context_payload(db_session):
    result = _run_m4_daily_into(
        db_session,
        polygon=CleanPolygonContextAdapter(),
        benzinga=CleanBenzingaContextAdapter(),
    )
    feature = db_session.query(FeatureSnapshot).filter_by(ticker="LCUT").one()
    m4_run = (
        db_session.query(EvidenceJobRun)
        .join(EvidenceJob, EvidenceJob.job_id == EvidenceJobRun.job_id)
        .filter(EvidenceJob.job_name == "m4_daily_feature_assembly")
        .one()
    )
    assert feature.job_run_id == m4_run.job_run_id

    report = _build_report(db_session, m4_metrics=result.metrics)
    row = report["fired_signal_table"][0]

    assert row["ticker"] == "LCUT"
    assert row["X_M4"] == 1.1
    assert row["price"] == 11.0
    assert row["high_52w"] == 10.0
    assert row["short_interest"] == {
        "status": "matched",
        "short_interest": 1200,
        "days_to_cover": "2.5",
    }
    assert row["short_volume"] == {
        "status": "no_data",
        "short_volume_ratio": None,
    }
    assert row["polygon_news"] == {
        "status": "no_data",
        "article_count_90d": 0,
    }
    assert row["benzinga_news"]["article_count_7d"] == 1
    assert row["benzinga_news"]["wiim_count_7d"] == 0
    assert row["insider"]["status"] == "matched"
    assert row["insider"]["net_discretionary_shares"] == "75"
    assert row["calendar_status"] == {
        "earnings": "matched",
        "ratings": "no_data",
        "offerings": "no_data",
        "dividends": "no_data",
    }
    assert row["benzinga_ma"]["status"] == "no_data"
    assert row["context_lineage_id_count"] > 0
    assert row["context_lineage_hash_count"] == 0
    assert row["source_attempt_status_counts"]["matched"] >= 1


def test_duplicate_signal_detection(db_session):
    _add_feature(db_session, "feature-one", "ONE", payload={"signal_context": _minimal_context()})
    _add_feature(db_session, "feature-two", "TWO", payload={"signal_context": _minimal_context()})
    _add_signal(db_session, "signal-one", "feature-one", "ONE", identity_hash="duplicate-identity")
    _add_signal(db_session, "signal-two", "feature-two", "TWO", identity_hash="duplicate-identity")

    report = _build_report(db_session)

    assert report["m4_signals"]["duplicate_signal_count"] == 1
    assert report["health"] is False
    assert "no_duplicate_signals" in report["health_verdict"]["failing_checks"]


@pytest.mark.parametrize(
    "run_ts",
    [
        _utc(2026, 5, 30, 12, 0),  # Saturday
        _utc(2026, 5, 31, 12, 0),  # Sunday
    ],
)
def test_weekend_timestamp_does_not_produce_weekend_decision(run_ts):
    with pytest.raises(CanonicalRunError, match="non-trading decision date"):
        resolve_canonical_clock(run_ts, None)


def test_report_metrics_populated_after_orchestration(db_session):
    result = _run_m4_daily_into(
        db_session,
        polygon=CleanPolygonContextAdapter(),
        benzinga=CleanBenzingaContextAdapter(),
    )

    report = _build_report(db_session)

    assert report["m4_assembly"]["assembled_count"] == result.metrics["assembly"]["assembled_count"]
    assert report["m4_assembly"]["included_universe_size"] == result.metrics["included_universe_size"]
    assert report["source_attempts"]["context_reused_count"] is not None
    assert report["freeze_reuse"]["context_enriched_count"] is not None


def test_explicit_empty_m4_metrics_do_not_fallback_to_db_metrics(db_session):
    run_id = _ensure_test_m4_run(
        db_session,
        run_id="same-date-m4-with-metrics",
        metrics={
            "decision_date": "2026-05-26",
            "assembly": {"assembled_count": 44},
            "fetched_symbol_count": 44,
            "fetched_bar_count": 44,
            "fetch_error_count": 44,
        },
    )
    _add_feature(db_session, "feature-empty-m4-metrics", "EM4", job_run_id=run_id)
    _add_signal(db_session, "signal-empty-m4-metrics", "feature-empty-m4-metrics", "EM4")

    report = build_m4_health_report(
        db_session,
        mode="scratch",
        schema="m4_live_scratch_test",
        host_class="postgres_other",
        app_commit_sha="deadbeef",
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
        next_execution_session="2026-05-26",
        run_timestamp="2026-05-26T08:00:00+00:00",
        primary_m4_run_id=run_id,
        m4_metrics={},
    )

    assert report["m4_assembly"]["assembled_count"] is None
    assert report["m4_assembly"]["fetched_symbol_count"] is None
    assert report["m4_assembly"]["fetched_bar_count"] is None
    assert report["m4_assembly"]["fetch_error_count"] is None


def test_explicit_empty_universe_metrics_do_not_fallback_to_canonical_pointer(db_session):
    run_id = _ensure_test_m4_run(db_session, run_id="empty-universe-m4-run")
    _add_feature(db_session, "feature-empty-universe-metrics", "EUV", job_run_id=run_id)
    _add_signal(db_session, "signal-empty-universe-metrics", "feature-empty-universe-metrics", "EUV")
    db_session.add(UniverseScan(
        scan_id="canonical-scan-with-metrics",
        trading_date="2026-05-26",
        asof_timestamp=_utc(2026, 5, 26, 14, 0),
        raw_count=44,
        deduped_count=44,
        included_count=44,
        excluded_count=0,
        run_status="finished",
        metric_json=json.dumps({"raw_count": 44, "deduped_count": 44, "included": 44, "excluded": 0}),
    ))
    db_session.flush()
    db_session.add(CanonicalUniverseScan(
        trading_date="2026-05-26",
        scan_id="canonical-scan-with-metrics",
        selection_reason="test",
    ))

    report = build_m4_health_report(
        db_session,
        mode="scratch",
        schema="m4_live_scratch_test",
        host_class="postgres_other",
        app_commit_sha="deadbeef",
        decision_date="2026-05-26",
        evidence_session_date="2026-05-22",
        next_execution_session="2026-05-26",
        run_timestamp="2026-05-26T08:00:00+00:00",
        universe_metrics={},
        universe_run_id="current-universe-run",
        primary_m4_run_id=run_id,
        m4_metrics={"decision_date": "2026-05-26", "assembly": {"assembled_count": 1}},
    )

    assert report["universe"]["included_universe_size"] is None
    assert report["universe"]["universe_metrics_source"] == "current_invocation"


def test_main_threads_current_invocation_run_ids_and_writes_scoped_json(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@host/db")
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)
    calls = []

    def fake_universe(**kwargs):
        calls.append(("universe", kwargs))
        return run_m4_canonical.RunInvocation(
            exit_code=0,
            run_id="universe-run",
            run_status="finished",
            metrics={"included": 3, "universe_metrics_source": "current_invocation"},
        )

    def fake_m4(**kwargs):
        calls.append(("m4", kwargs))
        return run_m4_canonical.RunInvocation(
            exit_code=0,
            run_id="m4-rerun" if kwargs["rerun"] else "m4-primary",
            run_status="finished",
            metrics={
                "decision_date": "2026-05-26",
                "assembly": {"assembled_count": 1},
                "signal_context": {
                    "context_enriched_count": 0 if kwargs["rerun"] else 1,
                    "context_reused_from_persistence_count": 1 if kwargs["rerun"] else 0,
                },
            },
        )

    class FakeSession:
        def close(self):
            calls.append(("session_close", {}))

    class FakeEngine:
        def dispose(self):
            calls.append(("engine_dispose", {}))

    def fake_report_session(url, schema):
        calls.append(("report_session", {"schema": schema}))
        return FakeEngine(), FakeSession()

    def fake_build_report(session, **kwargs):
        calls.append(("build_report", kwargs))
        return {
            "health": True,
            "run_metadata": {
                "asof_timestamp": "2026-05-26T20:00:00+00:00",
            },
            "universe": {},
            "m4_assembly": {},
            "m4_signals": {},
            "data_quality": {},
            "forward_return_guard": {},
            "source_attempts": {},
            "freeze_reuse": {},
            "idempotency_rerun": {},
            "run_diagnostics": {},
            "fired_signal_table": [],
            "health_verdict": {"failing_checks": []},
        }

    monkeypatch.setattr(run_m4_canonical, "_run_universe", fake_universe)
    monkeypatch.setattr(run_m4_canonical, "_run_m4", fake_m4)
    monkeypatch.setattr(run_m4_canonical, "_report_session", fake_report_session)
    monkeypatch.setattr(run_m4_canonical, "build_m4_health_report", fake_build_report)
    monkeypatch.setattr(run_m4_canonical, "_app_commit_sha", lambda: "deadbeef")

    output = tmp_path / "report.json"
    rc = run_m4_canonical.main([
        "--scratch",
        "--schema", "scratch_schema",
        "--run-timestamp", "2026-05-29T20:00:00-04:00",
        "--decision-date", "2026-05-26",
        "--json-output", str(output),
    ])

    assert rc == 0
    build_kwargs = [payload for name, payload in calls if name == "build_report"][0]
    assert build_kwargs["universe_run_id"] == "universe-run"
    assert build_kwargs["primary_m4_run_id"] == "m4-primary"
    assert build_kwargs["rerun_m4_run_id"] == "m4-rerun"
    assert build_kwargs["m4_metrics"]["signal_context"]["context_enriched_count"] == 1
    assert build_kwargs["rerun_m4_metrics"]["signal_context"]["context_reused_from_persistence_count"] == 1
    assert output.exists()
    assert json.loads(output.read_text())["run_metadata"]["asof_timestamp"] == "2026-05-26T20:00:00+00:00"


def test_main_aborts_on_zero_exit_with_failed_run_status(monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@host/db")
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)

    def fake_universe(**kwargs):
        return run_m4_canonical.RunInvocation(
            exit_code=0,
            run_id="universe-run",
            run_status="finished",
            metrics={"included": 1},
        )

    def fake_m4(**kwargs):
        return run_m4_canonical.RunInvocation(
            exit_code=0,
            run_id=None,
            run_status="failed",
            metrics={},
        )

    def fail_report(*_args, **_kwargs):  # pragma: no cover - must not be called
        raise AssertionError("health report must not build after invalid child run")

    monkeypatch.setattr(run_m4_canonical, "_run_universe", fake_universe)
    monkeypatch.setattr(run_m4_canonical, "_run_m4", fake_m4)
    monkeypatch.setattr(run_m4_canonical, "build_m4_health_report", fail_report)

    rc = run_m4_canonical.main([
        "--scratch",
        "--schema", "scratch_schema",
        "--run-timestamp", "2026-05-29T20:00:00-04:00",
    ])

    assert rc == 1
    assert "without a finished job run" in capsys.readouterr().out


def test_run_match_prefers_metric_decision_date_over_conflicting_params(db_session):
    job = EvidenceJob(
        job_id="decision-match-job",
        job_name="m4_daily_feature_assembly",
        job_type="feature_assembly",
        owner_component="test",
    )
    db_session.add(job)
    db_session.flush()
    db_session.add(EvidenceJobRun(
        job_run_id="decision-conflict-run",
        job_id=job.job_id,
        run_status="finished",
        started_at=_utc(2026, 5, 26, 20, 0),
        ended_at=_utc(2026, 5, 26, 20, 5),
        metric_json=json.dumps({"decision_date": "2026-05-26"}),
        params_json=json.dumps({"trading_date": "2026-05-27"}),
    ))
    db_session.flush()

    matched_metric_date = run_m4_canonical._latest_job_run_for_decision(
        db_session,
        job_name="m4_daily_feature_assembly",
        decision_date="2026-05-26",
        success_only=True,
    )
    matched_param_date = run_m4_canonical._latest_job_run_for_decision(
        db_session,
        job_name="m4_daily_feature_assembly",
        decision_date="2026-05-27",
        success_only=True,
    )

    assert matched_metric_date.job_run_id == "decision-conflict-run"
    assert matched_param_date is None


def test_failed_latest_m4_run_does_not_override_successful_metrics(db_session):
    successful = _ensure_test_m4_run(
        db_session,
        run_id="older-success",
        run_status="finished",
        started_at=_utc(2026, 5, 26, 18, 0),
        ended_at=_utc(2026, 5, 26, 18, 5),
        metrics={
            "decision_date": "2026-05-26",
            "included_universe_size": 3,
            "fetched_symbol_count": 3,
            "fetched_bar_count": 183,
            "fetch_error_count": 0,
            "assembly": {"assembled_count": 3},
            "signal_context": {
                "context_attached_count": 2,
                "context_reused_from_persistence_count": 0,
                "context_reused_in_memory_count": 0,
                "context_enriched_count": 2,
                "context_persistence_miss_count": 2,
                "context_persistence_mismatch_count": 0,
                "context_persistence_mismatch_reasons": {},
            },
        },
    )
    _ensure_test_m4_run(
        db_session,
        run_id="newer-failed",
        run_status="failed",
        started_at=_utc(2026, 5, 26, 20, 0),
        ended_at=_utc(2026, 5, 26, 20, 5),
        metrics={
            "decision_date": "2026-05-26",
            "included_universe_size": 999,
            "assembly": {"assembled_count": 999},
            "signal_context": {"context_enriched_count": 999},
        },
    )
    _add_feature(db_session, "feature-success-metric", "MET", job_run_id=successful)
    _add_signal(db_session, "signal-success-metric", "feature-success-metric", "MET")

    report = _build_report(db_session)

    assert report["m4_assembly"]["included_universe_size"] == 3
    assert report["m4_assembly"]["assembled_count"] == 3
    assert report["m4_assembly"]["fetched_symbol_count"] == 3
    assert report["m4_assembly"]["fetched_bar_count"] == 183
    assert report["m4_assembly"]["fetch_error_count"] == 0
    assert report["source_attempts"]["context_attached_count"] == 2
    assert report["freeze_reuse"]["context_enriched_count"] == 2
    assert report["run_diagnostics"]["failed_same_date_m4_run_count"] == 1


def test_universe_metrics_prefer_canonical_pointer_over_later_scan(db_session):
    canonical = UniverseScan(
        scan_id="canonical-scan",
        trading_date="2026-05-26",
        asof_timestamp=_utc(2026, 5, 26, 14, 0),
        raw_count=3,
        deduped_count=3,
        included_count=3,
        excluded_count=0,
        run_status="finished",
        metric_json=json.dumps({"raw_count": 3, "deduped_count": 3, "included": 3, "excluded": 0}),
    )
    later = UniverseScan(
        scan_id="later-failed-scan",
        trading_date="2026-05-26",
        asof_timestamp=_utc(2026, 5, 26, 15, 0),
        raw_count=999,
        deduped_count=999,
        included_count=999,
        excluded_count=0,
        run_status="failed",
        metric_json=json.dumps({"raw_count": 999, "deduped_count": 999, "included": 999, "excluded": 0}),
    )
    db_session.add_all([canonical, later])
    db_session.flush()
    db_session.add(CanonicalUniverseScan(
        trading_date="2026-05-26",
        scan_id="canonical-scan",
        selection_reason="test",
    ))
    _add_feature(db_session, "feature-universe", "UNI")
    _add_signal(db_session, "signal-universe", "feature-universe", "UNI")

    report = _build_report(db_session)

    assert report["universe"]["included_universe_size"] == 3
    assert report["universe"]["universe_metrics_source"] == "canonical_pointer"


def test_universe_metrics_latest_fallback_is_flagged(db_session):
    db_session.add(UniverseScan(
        scan_id="fallback-scan",
        trading_date="2026-05-26",
        asof_timestamp=_utc(2026, 5, 26, 15, 0),
        raw_count=4,
        deduped_count=4,
        included_count=4,
        excluded_count=0,
        run_status="finished",
        metric_json=json.dumps({"raw_count": 4, "deduped_count": 4, "included": 4, "excluded": 0}),
    ))
    _add_feature(db_session, "feature-universe-fallback", "UNIF")
    _add_signal(db_session, "signal-universe-fallback", "feature-universe-fallback", "UNIF")

    report = _build_report(db_session)

    assert report["universe"]["included_universe_size"] == 4
    assert report["universe"]["universe_metrics_source"] == "latest_fallback"
