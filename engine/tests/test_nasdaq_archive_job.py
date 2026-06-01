from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
import requests

from alpha.data.contracts import AdapterResponse, LineageMeta, ProviderError
from alpha.data.nasdaq import (
    ADDS_DELETES,
    HALT_RSS,
    NASDAQ_LISTED,
    OTHER_LISTED,
    NasdaqArchiveCaptureResult,
    NasdaqListingStatus,
    NasdaqTraderListingAdapter,
)
from alpha.db.models import EvidenceJob, EvidenceJobRun
from alpha.jobs import run_nasdaq_archive
from alpha.jobs.contracts import JobResult
from alpha.jobs.run_nasdaq_archive import NasdaqArchiveJob
from alpha.jobs.runner import run_job
from alpha.market_calendar import us_equity_session_close_timestamp


ET = ZoneInfo("America/New_York")
ASOF = us_equity_session_close_timestamp(date(2026, 6, 1))


def _lineage(endpoint="nasdaq_archive_current_snapshot"):
    return LineageMeta(
        provider="nasdaq_trader",
        endpoint=endpoint,
        request_timestamp=datetime(2026, 6, 1, 22, 15, tzinfo=timezone.utc),
        asof_timestamp=ASOF,
        raw_payload_hash=f"{endpoint}-hash",
        source_authority="NASDAQ_TRADER",
    )


def _capture_response(
    *,
    captured_sources=(NASDAQ_LISTED, OTHER_LISTED, ADDS_DELETES, HALT_RSS),
    failed_sources=(),
    inserted_snapshots=4,
    existing_snapshots=0,
    inserted_rows=12721,
):
    return AdapterResponse(
        data=NasdaqArchiveCaptureResult(
            captured_sources=tuple(captured_sources),
            inserted_snapshots=inserted_snapshots,
            existing_snapshots=existing_snapshots,
            inserted_rows=inserted_rows,
            raw_payload_hashes={source: f"{source}-hash" for source in captured_sources},
            failed_sources=tuple(failed_sources),
        ),
        lineage=_lineage(),
    )


def _provider_failure(endpoint=HALT_RSS):
    return AdapterResponse(
        data=None,
        lineage=_lineage(endpoint),
        error=ProviderError(
            provider="nasdaq_trader",
            endpoint=endpoint,
            status_code=None,
            error_type="timeout",
            message=f"{endpoint} timeout",
            retryable=True,
        ),
    )


class FakeArchiveAdapter:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def archive_current_snapshot(self, session, *, asof):
        self.calls.append({"session": session, "asof": asof})
        if len(self.calls) <= len(self.responses):
            return self.responses[len(self.calls) - 1]
        return self.responses[-1]


def _mock_response(text: str, status_code: int = 200):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    return resp


def _nasdaq_directory():
    rows = ["AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N"]
    rows.extend(f"N{idx:04d}|N filler {idx}|Q|N|N|100|N|N" for idx in range(1000))
    return "\n".join([
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares",
        *rows,
        "File Creation Time: 0601202618:15|||||||",
    ])


def _other_directory():
    rows = [
        f"O{idx:04d}|O filler {idx}|N|O{idx:04d}|N|100|N|O{idx:04d}"
        for idx in range(1001)
    ]
    return "\n".join([
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol",
        *rows,
        "File Creation Time: 0601202618:15||||||",
    ])


def _adds_deletes():
    return "\n".join([
        "Symbol|Company Name|NASDAQ Action|BX Action|PSX Action|Effective Date|Primary Listing Market",
        "File Creation Time: 0601202618:15|||||",
    ])


def _halt_rss():
    return """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:ndaq="http://www.nasdaqtrader.com/">
  <channel>
    <title>NASDAQTrader.com</title>
    <pubDate>Mon, 01 Jun 2026 22:15:00 GMT</pubDate>
    <ndaq:numItems>0</ndaq:numItems>
  </channel>
</rss>"""


def _job(session, adapter, *, max_attempts=3, sleeper=None):
    return NasdaqArchiveJob(
        session=session,
        adapter=adapter,
        asof_timestamp=ASOF,
        max_attempts=max_attempts,
        retry_sleep_seconds=0,
        sleeper=sleeper or (lambda _seconds: None),
    )


def _latest_metric_json(db_session):
    run = (
        db_session.query(EvidenceJobRun)
        .join(EvidenceJob, EvidenceJob.job_id == EvidenceJobRun.job_id)
        .filter(EvidenceJob.job_name == "nasdaq_listing_archive")
        .order_by(EvidenceJobRun.started_at.desc())
        .first()
    )
    assert run is not None
    return json.loads(run.metric_json)


def test_nasdaq_archive_job_success_path_persists_metrics(db_session):
    adapter = FakeArchiveAdapter(_capture_response())

    result = run_job(db_session, _job(db_session, adapter), params={"source": "test"})

    assert result.ok
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["asof"] == ASOF
    metrics = _latest_metric_json(db_session)
    assert metrics["captured_source_count"] == 4
    assert metrics["failed_source_count"] == 0
    assert metrics["inserted_snapshots"] == 4
    assert metrics["inserted_rows"] == 12721
    assert metrics["raw_payload_hashes"][HALT_RSS] == "trade_halt_rss-hash"


def test_nasdaq_archive_capture_completeness_requires_all_sources():
    incomplete = _capture_response(
        captured_sources=(NASDAQ_LISTED, OTHER_LISTED, ADDS_DELETES),
        failed_sources=(),
        inserted_snapshots=3,
    )
    complete = _capture_response()

    assert not run_nasdaq_archive._capture_response_is_complete(incomplete)
    assert run_nasdaq_archive._capture_response_is_complete(complete)


def test_nasdaq_archive_job_retries_halt_failure_and_finishes(db_session):
    sleeps = []
    adapter = FakeArchiveAdapter(
        _capture_response(
            captured_sources=(NASDAQ_LISTED, OTHER_LISTED, ADDS_DELETES),
            failed_sources=(HALT_RSS,),
            inserted_snapshots=3,
            inserted_rows=12708,
        ),
        _capture_response(
            inserted_snapshots=1,
            existing_snapshots=3,
            inserted_rows=13,
        ),
    )
    job = NasdaqArchiveJob(
        session=db_session,
        adapter=adapter,
        asof_timestamp=ASOF,
        max_attempts=3,
        retry_sleep_seconds=0.1,
        sleeper=sleeps.append,
    )

    result = run_job(db_session, job, params={})

    assert result.ok
    assert len(adapter.calls) == 2
    assert sleeps == [0.1]
    metrics = result.metrics
    assert metrics["attempt_count"] == 2
    assert metrics["attempts"][0]["failed_sources"] == [HALT_RSS]
    assert metrics["attempts"][0]["will_retry"] is True
    assert metrics["failed_source_count"] == 0
    assert metrics["total_inserted_snapshots"] == 4
    assert metrics["total_existing_snapshots"] == 3


def test_nasdaq_archive_job_retries_directory_abort(db_session):
    adapter = FakeArchiveAdapter(
        _provider_failure(NASDAQ_LISTED),
        _capture_response(),
    )

    result = run_job(db_session, _job(db_session, adapter), params={})

    assert result.ok
    assert len(adapter.calls) == 2
    assert result.metrics["attempts"][0]["error"]["endpoint"] == NASDAQ_LISTED
    assert result.metrics["attempts"][1]["captured_sources"] == [
        NASDAQ_LISTED,
        OTHER_LISTED,
        ADDS_DELETES,
        HALT_RSS,
    ]


def test_nasdaq_archive_job_all_fail_surfaces_failed(db_session):
    adapter = FakeArchiveAdapter(
        _provider_failure(HALT_RSS),
        _provider_failure(HALT_RSS),
        _provider_failure(HALT_RSS),
    )

    result = run_job(db_session, _job(db_session, adapter), params={})

    assert not result.ok
    assert result.status == "failed"
    assert len(adapter.calls) == 3
    assert result.metrics["attempt_count"] == 3
    assert result.errors[0]["provider_error"]["endpoint"] == HALT_RSS
    run = db_session.query(EvidenceJobRun).one()
    assert run.run_status == "failed"


def test_nasdaq_archive_job_real_adapter_idempotent_rerun(db_session):
    session = MagicMock(spec=requests.Session)
    source_payloads = [
        _nasdaq_directory(),
        _other_directory(),
        _adds_deletes(),
        _halt_rss(),
    ]
    session.get.side_effect = [
        *(_mock_response(payload) for payload in source_payloads),
        *(_mock_response(payload) for payload in source_payloads),
    ]
    adapter = NasdaqTraderListingAdapter(session=session)

    first = run_job(db_session, _job(db_session, adapter, max_attempts=1), params={})
    second = run_job(db_session, _job(db_session, adapter, max_attempts=1), params={})

    assert first.ok
    assert first.metrics["inserted_snapshots"] == 4
    assert first.metrics["existing_snapshots"] == 0
    assert second.ok
    assert second.metrics["inserted_snapshots"] == 0
    assert second.metrics["existing_snapshots"] == 4
    assert second.metrics["inserted_rows"] == 0


def test_run_nasdaq_archive_default_non_trading_day_noops_before_db(monkeypatch, capsys):
    monkeypatch.setattr(run_nasdaq_archive, "load_runtime_env", lambda: None)
    monkeypatch.setattr(
        run_nasdaq_archive,
        "_utcnow",
        lambda: datetime(2026, 5, 30, 18, 15, tzinfo=ET),
    )
    monkeypatch.setattr(
        run_nasdaq_archive,
        "get_session",
        lambda: (_ for _ in ()).throw(AssertionError("database should not open")),
    )

    rc = run_nasdaq_archive.main(["--live"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "No-op reason:" in out
    assert "non_trading_day" in out


def test_run_nasdaq_archive_explicit_non_trading_day_fails_before_db(monkeypatch, capsys):
    monkeypatch.setattr(run_nasdaq_archive, "load_runtime_env", lambda: None)
    monkeypatch.setattr(
        run_nasdaq_archive,
        "get_session",
        lambda: (_ for _ in ()).throw(AssertionError("database should not open")),
    )

    rc = run_nasdaq_archive.main([
        "--live",
        "--run-timestamp",
        "2026-05-30T18:15:00-04:00",
    ])

    assert rc == 1
    assert "ERROR:" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("run_timestamp", "session_day"),
    [
        ("2026-06-01T16:00:00-04:00", date(2026, 6, 1)),
        ("2026-06-01T18:15:00-04:00", date(2026, 6, 1)),
        ("2026-11-27T13:00:00-05:00", date(2026, 11, 27)),
    ],
)
def test_run_nasdaq_archive_post_close_captures_with_session_close_asof(
    monkeypatch,
    capsys,
    run_timestamp,
    session_day,
):
    monkeypatch.setattr(run_nasdaq_archive, "load_runtime_env", lambda: None)

    class FakeSession:
        def close(self):
            pass

    class FakeAdapter:
        pass

    captured = {}

    def fake_run_job(session, job, params):
        captured["session"] = session
        captured["job"] = job
        captured["params"] = params
        return JobResult(
            status="finished",
            metrics={
                "archive_session_date": "2026-06-01",
                "asof_timestamp": job._asof_timestamp.isoformat(),
                "attempt_count": 1,
                "max_attempts": 3,
                "captured_sources": [],
                "failed_sources": [],
                "inserted_snapshots": 0,
                "existing_snapshots": 0,
                "inserted_rows": 0,
                "raw_payload_hashes": {},
            },
        )

    monkeypatch.setattr(run_nasdaq_archive, "get_session", lambda: FakeSession())
    monkeypatch.setattr(run_nasdaq_archive, "NasdaqTraderListingAdapter", FakeAdapter)
    monkeypatch.setattr(run_nasdaq_archive, "run_job", fake_run_job)

    rc = run_nasdaq_archive.main([
        "--live",
        "--run-timestamp",
        run_timestamp,
    ])

    expected_asof = us_equity_session_close_timestamp(session_day)
    assert rc == 0
    assert captured["params"]["asof_timestamp"] == expected_asof.isoformat()
    assert captured["job"]._asof_timestamp == expected_asof
    assert "As-of timestamp:" in capsys.readouterr().out


@pytest.mark.parametrize(
    "run_timestamp",
    [
        "2026-06-01T09:45:00-04:00",
        "2026-06-01T15:59:00-04:00",
        "2026-11-27T12:59:00-05:00",
    ],
)
def test_run_nasdaq_archive_pre_close_noops_before_db_and_job(
    monkeypatch,
    capsys,
    run_timestamp,
):
    monkeypatch.setattr(run_nasdaq_archive, "load_runtime_env", lambda: None)
    calls = {"run_job": 0}
    monkeypatch.setattr(
        run_nasdaq_archive,
        "get_session",
        lambda: (_ for _ in ()).throw(AssertionError("database should not open")),
    )

    def fake_run_job(*args, **kwargs):
        calls["run_job"] += 1
        raise AssertionError("run_job should not be called")

    monkeypatch.setattr(run_nasdaq_archive, "run_job", fake_run_job)

    rc = run_nasdaq_archive.main([
        "--live",
        "--run-timestamp",
        run_timestamp,
    ])

    out = capsys.readouterr().out
    assert rc == 0
    assert calls["run_job"] == 0
    assert "pre_session_close_skip" in out
    assert "Session close:" in out


def test_run_nasdaq_archive_default_pre_close_noops_before_db_and_job(monkeypatch, capsys):
    monkeypatch.setattr(run_nasdaq_archive, "load_runtime_env", lambda: None)
    monkeypatch.setattr(
        run_nasdaq_archive,
        "_utcnow",
        lambda: datetime(2026, 6, 1, 9, 45, tzinfo=ET),
    )
    calls = {"run_job": 0}
    monkeypatch.setattr(
        run_nasdaq_archive,
        "get_session",
        lambda: (_ for _ in ()).throw(AssertionError("database should not open")),
    )

    def fake_run_job(*args, **kwargs):
        calls["run_job"] += 1
        raise AssertionError("run_job should not be called")

    monkeypatch.setattr(run_nasdaq_archive, "run_job", fake_run_job)

    rc = run_nasdaq_archive.main(["--live"])

    out = capsys.readouterr().out
    assert rc == 0
    assert calls["run_job"] == 0
    assert "pre_session_close_skip" in out


def test_run_nasdaq_archive_pre_close_persists_nothing_for_replay(
    monkeypatch,
    capsys,
    db_session,
):
    monkeypatch.setattr(run_nasdaq_archive, "load_runtime_env", lambda: None)
    monkeypatch.setattr(
        run_nasdaq_archive,
        "get_session",
        lambda: (_ for _ in ()).throw(AssertionError("database should not open")),
    )

    rc = run_nasdaq_archive.main([
        "--live",
        "--run-timestamp",
        "2026-06-01T09:45:00-04:00",
    ])
    replay = NasdaqTraderListingAdapter().get_listing_status(
        "AAPL",
        asof=ASOF,
        archive_session=db_session,
        use_live=False,
    )

    assert rc == 0
    assert "pre_session_close_skip" in capsys.readouterr().out
    assert replay.ok
    assert replay.data.status is not NasdaqListingStatus.LISTED_ACTIVE


def test_run_nasdaq_archive_naive_timestamp_fails_before_db(monkeypatch, capsys):
    monkeypatch.setattr(run_nasdaq_archive, "load_runtime_env", lambda: None)
    monkeypatch.setattr(
        run_nasdaq_archive,
        "get_session",
        lambda: (_ for _ in ()).throw(AssertionError("database should not open")),
    )

    rc = run_nasdaq_archive.main([
        "--live",
        "--run-timestamp",
        "2026-06-01T09:45:00",
    ])

    assert rc == 1
    assert "run_timestamp must be timezone-aware" in capsys.readouterr().out


def test_nasdaq_archive_build_does_not_modify_cleared_adapter_helpers():
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        pytest.skip("git metadata unavailable")
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", "engine/alpha/data/nasdaq.py"],
        cwd=root,
        check=False,
    )
    assert result.returncode == 0


@pytest.mark.skipif(
    not os.environ.get("ALPHA_LIVE_NASDAQ_ARCHIVE_SMOKE"),
    reason="set ALPHA_LIVE_NASDAQ_ARCHIVE_SMOKE=1 for live Nasdaq archive smoke",
)
def test_live_nasdaq_archive_smoke(db_session):
    adapter = NasdaqTraderListingAdapter()
    result = run_job(
        db_session,
        _job(db_session, adapter, max_attempts=3),
        params={"source": "live_smoke"},
    )

    assert result.ok
    assert result.metrics["failed_source_count"] == 0
    assert set(result.metrics["captured_sources"]) == {
        NASDAQ_LISTED,
        OTHER_LISTED,
        ADDS_DELETES,
        HALT_RSS,
    }
