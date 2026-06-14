from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone

import pytest
from sqlalchemy.exc import OperationalError

from alpha.data.contracts import AdapterResponse, LineageMeta, stable_hash
from alpha.data.fmp import FmpBar
from alpha.data.polygon import PolygonBar
from alpha.db.models import (
    FeatureSnapshot,
    HistoricalUniverseReconstruction,
    IntradayEventDetail,
    SignalRegistry,
)
from alpha.jobs.i11_historical_corpus import (
    CANDIDATE_SCREEN_STAMP,
    I11HistoricalCorpusJob,
    OUTCOME_FAILED_TEST,
    OUTCOME_NEVER_CONFIRMED,
)
from alpha.jobs.i12_historical_corpus import OUTCOME_CONFIRMED
from alpha.jobs.paper_execution import EASTERN, I11_PATTERN_ID
from alpha.jobs.run_i11_historical_corpus import (
    I11_CORPUS_REQUIRED_TABLES,
    _load_catalyst_tags_artifact,
    _parse_args,
    _validate_write_target,
)
from alpha.jobs.run_i12_historical_corpus import I12_CORPUS_REQUIRED_TABLES
from alpha.jobs.runner import run_job
from alpha.market_calendar import next_us_equity_session, previous_us_equity_session
from alpha.ml.security_type_exclusions import ExclusionArtifactError, SecurityTypeClassification


DAY = date(2026, 6, 3)
LEAKY_FEATURE_KEY_RE = re.compile(
    r"(full_day|leaky|ret_|mae|mfe|next_open|next_close|sessions_to_delist)"
)


class FakeFmp:
    def __init__(self, bars_by_ticker: dict[str, list[FmpBar]]) -> None:
        self.bars_by_ticker = {ticker.upper(): bars for ticker, bars in bars_by_ticker.items()}

    def get_historical_price(self, ticker, from_date=None, to_date=None, **kwargs):
        bars = self.bars_by_ticker[ticker.upper()]
        if from_date is not None:
            bars = [bar for bar in bars if date.fromisoformat(bar.date) >= from_date]
        if to_date is not None:
            bars = [bar for bar in bars if date.fromisoformat(bar.date) <= to_date]
        return AdapterResponse(data=bars, lineage=_lineage("FMP", ticker, bars))


class FakePolygon:
    def __init__(self, bars_by_ticker_date: dict[tuple[str, date], list[PolygonBar]]) -> None:
        self.bars_by_ticker_date = {
            (ticker.upper(), trading_date): bars
            for (ticker, trading_date), bars in bars_by_ticker_date.items()
        }
        self.calls: Counter[tuple[str, date]] = Counter()

    def get_minute_aggs(self, ticker, from_date, to_date, **kwargs):
        trading_date = date.fromisoformat(from_date)
        self.calls[(ticker.upper(), trading_date)] += 1
        bars = self.bars_by_ticker_date[(ticker.upper(), trading_date)]
        return AdapterResponse(data=bars, lineage=_lineage("Polygon", ticker, bars))


def _lineage(provider: str, ticker: str, bars: list) -> LineageMeta:
    return LineageMeta(
        provider=provider,
        endpoint="fixture",
        request_timestamp=datetime(2026, 6, 12, tzinfo=timezone.utc),
        asof_timestamp=datetime(2026, 6, 12, tzinfo=timezone.utc),
        raw_payload_hash=stable_hash({"ticker": ticker, "bars": [bar.__dict__ for bar in bars]}),
    )


def _classification(security_type: str = "common_stock") -> SecurityTypeClassification:
    return SecurityTypeClassification(
        ticker="TEST",
        security_type=security_type,
        reason="fixture",
        signals=1,
    )


def _seed_hur(db_session, ticker: str, trading_date: date = DAY) -> None:
    existing = (
        db_session.query(HistoricalUniverseReconstruction)
        .filter(
            HistoricalUniverseReconstruction.replay_date == trading_date,
            HistoricalUniverseReconstruction.normalized_symbol == ticker.upper(),
        )
        .one_or_none()
    )
    if existing is not None:
        return
    db_session.add(
        HistoricalUniverseReconstruction(
            historical_universe_reconstruction_id=f"i11-hur-{ticker}-{trading_date}",
            replay_date=trading_date,
            ticker=ticker,
            normalized_symbol=ticker.upper(),
            exchange="NASDAQ",
            company_name=f"{ticker} Inc.",
            ipo_date=date(2020, 1, 1),
            delisted_date=None,
            inclusion_status="included",
            rejection_reason=None,
            source="fixture",
            source_provenance_json="{}",
            reconstructed=True,
            reconstruction_method="fixture_hur",
            pit_filter_status_json="{}",
            input_hash=stable_hash({"ticker": ticker, "date": trading_date.isoformat()}),
            output_hash=stable_hash({"included": ticker, "date": trading_date.isoformat()}),
        )
    )
    db_session.flush()


def _daily_bars(
    *,
    day_open: float = 9.8,
    day_high: float = 10.9,
    day_low: float = 9.6,
    day_close: float = 10.7,
    prior_close: float = 9.8,
    max_close: float = 10.0,
    lookback_sessions: int = 252,
    include_next_day: bool = True,
) -> list[FmpBar]:
    days: list[date] = []
    cursor = DAY
    for _ in range(lookback_sessions):
        cursor = previous_us_equity_session(cursor)
        days.append(cursor)
    days.reverse()
    bars: list[FmpBar] = []
    for idx, day in enumerate(days):
        close = max_close if idx == 0 else prior_close
        bars.append(_daily_bar(day, close, high=close, low=close, close=close, volume=100_000))
    bars.append(_daily_bar(DAY, day_open, high=day_high, low=day_low, close=day_close, volume=600_000))
    if include_next_day:
        next_day = next_us_equity_session(DAY + timedelta(days=1))
        bars.append(_daily_bar(next_day, 11.2, high=11.7, low=11.0, close=11.4, volume=500_000))
    return sorted(bars, key=lambda bar: bar.date)


def _daily_bar(day: date, open_: float, *, high: float, low: float, close: float, volume: int) -> FmpBar:
    return FmpBar(
        date=day.isoformat(),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        split_adjusted_close=close,
        adj_close=close,
    )


def _minute_bars_confirmed() -> list[PolygonBar]:
    bars = [
        _minute_bar(0, 9.80, 9.90, 9.75, 9.85, 500),
        _minute_bar(1, 9.86, 9.93, 9.82, 9.88, 500),
        _minute_bar(2, 9.88, 9.96, 9.84, 9.90, 500),
        _minute_bar(3, 9.90, 9.98, 9.86, 9.92, 500),
        _minute_bar(4, 9.92, 9.99, 9.88, 9.94, 500),
        _minute_bar(5, 10.05, 10.25, 10.00, 10.15, 8_000),
        _minute_bar(6, 10.30, 10.50, 10.20, 10.40, 3_000),
        _minute_bar(385, 10.70, 10.90, 10.60, 10.80, 2_000),
    ]
    return bars


def _minute_bars_never_confirmed() -> list[PolygonBar]:
    bars = [
        _minute_bar(0, 9.80, 9.90, 9.75, 9.85, 100),
        _minute_bar(1, 10.02, 10.20, 10.00, 10.08, 100),
        _minute_bar(2, 10.06, 10.18, 10.02, 10.09, 100),
        _minute_bar(5, 10.04, 10.12, 9.98, 10.06, 100),
        _minute_bar(385, 10.20, 10.35, 10.10, 10.25, 200),
    ]
    return bars


def _minute_bars_failed_test() -> list[PolygonBar]:
    bars = [
        _minute_bar(0, 9.80, 9.90, 9.75, 9.85, 100),
        _minute_bar(1, 9.95, 10.12, 9.90, 9.98, 100),
        _minute_bar(2, 9.96, 10.10, 9.88, 9.97, 100),
        _minute_bar(5, 9.94, 10.08, 9.86, 9.96, 100),
        _minute_bar(385, 9.90, 10.05, 9.70, 9.82, 200),
    ]
    return bars


def _minute_bars_daily_high_unconfirmed() -> list[PolygonBar]:
    return [
        _minute_bar(0, 9.80, 9.90, 9.75, 9.85, 100),
        _minute_bar(1, 9.86, 9.95, 9.82, 9.88, 100),
        _minute_bar(5, 9.88, 9.98, 9.80, 9.90, 100),
        _minute_bar(385, 9.90, 9.95, 9.70, 9.82, 200),
    ]


def _minute_bars_late_confirmed() -> list[PolygonBar]:
    return [
        _minute_bar(385, 9.80, 9.90, 9.75, 9.92, 500),
        _minute_bar(387, 10.05, 10.40, 10.00, 10.30, 500_000),
        _minute_bar(388, 10.35, 10.50, 10.20, 10.40, 1_000),
    ]


def _minute_bars_late_never_confirmed() -> list[PolygonBar]:
    return [
        _minute_bar(385, 9.80, 9.90, 9.75, 9.92, 100),
        _minute_bar(387, 10.02, 10.20, 10.00, 10.08, 100),
        _minute_bar(388, 10.12, 10.25, 10.05, 10.16, 100),
    ]


def _minute_bar(
    minute_index: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> PolygonBar:
    ts = datetime.combine(DAY, time(9, 30), EASTERN) + timedelta(minutes=minute_index)
    return PolygonBar(
        timestamp=int(ts.astimezone(timezone.utc).timestamp() * 1000),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _run_i11(
    db_session,
    *,
    ticker: str = "TEST",
    daily=None,
    minutes=None,
    classifications=None,
    polygon=None,
    minute_cache_dir=None,
    skip_existing=False,
    max_db_retries=3,
    db_retry_backoff_seconds=5.0,
    catalyst_tags_by_ticker_date=None,
):
    _seed_hur(db_session, ticker)
    if classifications is None:
        classifications = {ticker: _classification()}
    if polygon is None:
        polygon = FakePolygon({(ticker, DAY): minutes or _minute_bars_confirmed()})
    job = I11HistoricalCorpusJob(
        session=db_session,
        fmp_adapter=FakeFmp({ticker: daily or _daily_bars()}),
        polygon_adapter=polygon,
        start_date=DAY,
        end_date=DAY,
        classification_records=classifications,
        minute_cache_dir=minute_cache_dir,
        skip_existing=skip_existing,
        max_db_retries=max_db_retries,
        db_retry_backoff_seconds=db_retry_backoff_seconds,
        catalyst_tags_by_ticker_date=catalyst_tags_by_ticker_date,
    )
    return run_job(db_session, job), polygon


def _assert_feature_json_pit_pure(payload, path=()):
    if path and path[0] in {"candidate_screen", "pit_caveats", "research_only_leaky"}:
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            if not path and key in {"candidate_screen", "pit_caveats", "research_only_leaky"}:
                continue
            assert not LEAKY_FEATURE_KEY_RE.search(key), ".".join(path + (key,))
            _assert_feature_json_pit_pure(value, path + (key,))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _assert_feature_json_pit_pure(value, path + (str(index),))


def test_i11_corpus_persists_confirmed_entry_with_next_open_primary_label(db_session):
    result, polygon = _run_i11(
        db_session,
        catalyst_tags_by_ticker_date={("TEST", DAY): ["offering"]},
    )

    assert result.status == "finished"
    assert result.metrics["confirmed"] == 1
    assert polygon.calls[("TEST", DAY)] == 1
    signal = db_session.query(SignalRegistry).filter_by(pattern_id=I11_PATTERN_ID, ticker="TEST").one()
    assert signal.signal_horizon == "1d"
    assert signal.forward_return_status == "computed"
    assert signal.raw_expected_edge == 0.0
    assert signal.point_in_time_passed is False
    detail = db_session.query(IntradayEventDetail).filter_by(pattern_id=I11_PATTERN_ID).one()
    assert detail.signal_id == signal.signal_id
    assert detail.outcome == OUTCOME_CONFIRMED
    assert detail.entry_price == pytest.approx(10.30)
    assert detail.ret_next_open == pytest.approx((11.2 / 10.30) - 1.0)
    assert signal.forward_return == pytest.approx(detail.ret_next_open)
    feature_snapshot = db_session.get(FeatureSnapshot, signal.feature_snapshot_id)
    assert feature_snapshot.point_in_time_passed is False
    feature_json = json.loads(detail.feature_json)
    assert feature_json["candidate_screen"] == CANDIDATE_SCREEN_STAMP
    assert feature_json["catalyst_tags"] == ["offering"]
    assert feature_json["mom20"] is not None
    assert feature_json["off_low252"] is not None
    assert "avg20_volume" not in feature_json
    assert "price" not in feature_json
    assert set(feature_json["research_only_leaky"]) == {
        "avg20_volume",
        "price",
    }
    assert "cross_minute" not in feature_json["gate_values"]
    assert "day_high" not in feature_json["gate_values"]
    _assert_feature_json_pit_pure(feature_json)
    labels = json.loads(detail.label_json)
    assert labels["ret_next_open_primary"] is True
    assert labels["ret_next_close"] == pytest.approx((11.4 / 10.30) - 1.0)
    assert labels["overnight_gap_ret"] == pytest.approx((11.2 / 10.7) - 1.0)
    assert labels["chase_over_hb_pct"] == pytest.approx((10.30 / 10.0) - 1.0)
    assert labels["minute_volume_up_to_confirmation"] == pytest.approx(10_500)
    assert labels["premarket_volume_status"] == "not_available_polygon_regular_session_cache"


def test_i11_confirmed_missing_next_open_is_not_computed_null(db_session):
    result, _polygon = _run_i11(
        db_session,
        daily=_daily_bars(include_next_day=False),
    )

    assert result.status == "finished"
    assert result.metrics["confirmed"] == 1
    assert result.metrics["primary_label_unavailable"] == 1
    signal = db_session.query(SignalRegistry).filter_by(pattern_id=I11_PATTERN_ID).one()
    assert signal.forward_return_status == "outcome_unavailable"
    assert signal.outcome_unavailable_reason == "missing_next_open_price"
    assert signal.forward_return is None
    detail = db_session.query(IntradayEventDetail).filter_by(pattern_id=I11_PATTERN_ID).one()
    assert detail.ret_next_open is None
    flags = json.loads(detail.artifact_flags_json)
    assert flags["primary_label_unavailable"] is True
    assert flags["primary_label_unavailable_reason"] == "missing_next_open_price"


def test_i11_late_confirmation_does_not_time_travel_ret_conf(db_session):
    result, _polygon = _run_i11(db_session, minutes=_minute_bars_late_confirmed())

    assert result.status == "finished"
    detail = db_session.query(IntradayEventDetail).filter_by(pattern_id=I11_PATTERN_ID).one()
    assert detail.outcome == OUTCOME_CONFIRMED
    assert detail.entry_minute == 388
    assert detail.exit_timestamp < detail.entry_timestamp
    assert detail.ret_conf is None
    labels = json.loads(detail.label_json)
    flags = json.loads(detail.artifact_flags_json)
    assert labels["entry_after_session_exit_reference"] is True
    assert flags["entry_after_session_exit_reference"] is True


def test_i11_late_control_reference_does_not_time_travel_ret_conf(db_session):
    result, _polygon = _run_i11(db_session, minutes=_minute_bars_late_never_confirmed())

    assert result.status == "finished"
    detail = db_session.query(IntradayEventDetail).filter_by(pattern_id=I11_PATTERN_ID).one()
    assert detail.outcome == OUTCOME_NEVER_CONFIRMED
    assert detail.ret_conf is None
    labels = json.loads(detail.label_json)
    flags = json.loads(detail.artifact_flags_json)
    assert labels["control_reference_timestamp"] is not None
    assert labels["entry_after_session_exit_reference"] is True
    assert flags["entry_after_session_exit_reference"] is True


def test_i11_corpus_is_idempotent_on_rerun(db_session):
    first, _polygon = _run_i11(db_session)
    second, _polygon = _run_i11(db_session)

    assert first.status == "finished"
    assert second.status == "finished"
    assert second.metrics["inserted_details"] == 0
    assert second.metrics["reused_details"] == 1
    assert db_session.query(IntradayEventDetail).filter_by(pattern_id=I11_PATTERN_ID).count() == 1
    assert db_session.query(SignalRegistry).filter_by(pattern_id=I11_PATTERN_ID).count() == 1


def test_i11_corpus_uses_disk_cached_polygon_minutes(db_session, tmp_path):
    first_polygon = FakePolygon({("TEST", DAY): _minute_bars_confirmed()})
    first, _polygon = _run_i11(
        db_session,
        polygon=first_polygon,
        minute_cache_dir=tmp_path,
    )

    assert first.status == "finished"
    assert first.metrics["minute_cache_misses"] == 1
    assert first_polygon.calls[("TEST", DAY)] == 1

    second_polygon = FakePolygon({})
    second, _polygon = _run_i11(
        db_session,
        polygon=second_polygon,
        minute_cache_dir=tmp_path,
    )

    assert second.status == "finished"
    assert second.metrics["minute_cache_hits"] == 1
    assert second.metrics["minute_cache_misses"] == 0
    assert second_polygon.calls == Counter()


def test_i11_corpus_skip_existing_avoids_refetching_processed_ticker_day(db_session):
    first_polygon = FakePolygon({("TEST", DAY): _minute_bars_confirmed()})
    first, _polygon = _run_i11(db_session, polygon=first_polygon)

    assert first.status == "finished"
    assert first.metrics["inserted_details"] == 1

    second_polygon = FakePolygon({})
    second, _polygon = _run_i11(
        db_session,
        polygon=second_polygon,
        skip_existing=True,
    )

    assert second.status == "finished"
    assert second.metrics["skipped_existing"] == 1
    assert second.metrics["candidates"] == 0
    assert second.metrics["inserted_details"] == 0
    assert second.metrics["reused_details"] == 0
    assert second.metrics["minute_cache_hits"] == 0
    assert second.metrics["minute_cache_misses"] == 0
    assert second_polygon.calls == Counter()
    assert db_session.query(IntradayEventDetail).filter_by(pattern_id=I11_PATTERN_ID, ticker="TEST").count() == 1
    assert db_session.query(SignalRegistry).filter_by(pattern_id=I11_PATTERN_ID, ticker="TEST").count() == 1


def test_i11_corpus_retries_transient_db_disconnect_once(db_session):
    _seed_hur(db_session, "TEST")
    job = I11HistoricalCorpusJob(
        session=db_session,
        fmp_adapter=FakeFmp({"TEST": _daily_bars()}),
        polygon_adapter=FakePolygon({("TEST", DAY): _minute_bars_confirmed()}),
        start_date=DAY,
        end_date=DAY,
        classification_records={"TEST": _classification()},
        max_db_retries=1,
        db_retry_backoff_seconds=0,
    )
    original_load_hur_rows = job._load_hur_rows
    calls = {"count": 0}

    def _flaky_load_hur_rows(trading_dates):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OperationalError(
                "SELECT historical_universe_reconstructions",
                {},
                Exception("terminating connection due to administrator command"),
            )
        return original_load_hur_rows(trading_dates)

    job._load_hur_rows = _flaky_load_hur_rows

    result = run_job(db_session, job)

    assert result.status == "finished"
    assert calls["count"] == 2
    assert result.metrics["db_reconnect_retries"] == 1
    assert result.metrics["inserted_details"] == 1
    assert result.metrics["inserted_signals"] == 1
    assert db_session.query(IntradayEventDetail).filter_by(pattern_id=I11_PATTERN_ID, ticker="TEST").count() == 1
    assert db_session.query(SignalRegistry).filter_by(pattern_id=I11_PATTERN_ID, ticker="TEST").count() == 1


def test_i11_never_confirmed_control_keeps_cross_anatomy_without_signal(db_session):
    result, _polygon = _run_i11(db_session, minutes=_minute_bars_never_confirmed())

    assert result.status == "finished"
    assert result.metrics["never_confirmed"] == 1
    assert db_session.query(SignalRegistry).filter_by(pattern_id=I11_PATTERN_ID).count() == 0
    detail = db_session.query(IntradayEventDetail).filter_by(pattern_id=I11_PATTERN_ID).one()
    assert detail.outcome == OUTCOME_NEVER_CONFIRMED
    assert detail.ret_next_open is not None
    assert detail.entry_price is None
    labels = json.loads(detail.label_json)
    gate_values = json.loads(detail.gate_values_json)
    assert labels["cross_minute"] == 1
    assert labels["cross_price"] == pytest.approx(10.0)
    assert labels["cross_hold_minute"] == 1
    assert labels["control_reference_basis"] == "next_minute_open_after_held_cross"
    assert labels["minute_volume_up_to_cross"] == pytest.approx(200)
    assert gate_values["premarket_volume_status"] == "not_available_polygon_regular_session_cache"


def test_i11_failed_test_control_records_touch_without_held_cross(db_session):
    result, _polygon = _run_i11(db_session, minutes=_minute_bars_failed_test())

    assert result.status == "finished"
    assert result.metrics["failed_test"] == 1
    assert db_session.query(SignalRegistry).filter_by(pattern_id=I11_PATTERN_ID).count() == 0
    detail = db_session.query(IntradayEventDetail).filter_by(pattern_id=I11_PATTERN_ID).one()
    assert detail.outcome == OUTCOME_FAILED_TEST
    labels = json.loads(detail.label_json)
    assert labels["cross_minute"] == 1
    assert labels["cross_hold_minute"] is None
    assert labels["control_reference_basis"] == "next_minute_open_after_intraday_touch"
    assert labels["ret_next_open"] is not None


def test_i11_failed_test_daily_high_unconfirmed_by_minutes_has_distinct_basis(db_session):
    result, _polygon = _run_i11(db_session, minutes=_minute_bars_daily_high_unconfirmed())

    assert result.status == "finished"
    assert result.metrics["failed_test"] == 1
    detail = db_session.query(IntradayEventDetail).filter_by(pattern_id=I11_PATTERN_ID).one()
    labels = json.loads(detail.label_json)
    assert labels["cross_minute"] is None
    assert labels["control_reference_basis"] == "daily_high_unconfirmed_by_minute_bars"


def test_i11_daily_screen_out_does_not_fetch_minutes_or_write_detail(db_session):
    polygon = FakePolygon({})
    result, polygon = _run_i11(
        db_session,
        daily=_daily_bars(day_high=9.95),
        polygon=polygon,
    )

    assert result.status == "finished"
    assert result.metrics["candidates"] == 0
    assert result.metrics["candidates_screened_out"] == 1
    assert result.metrics["candidate_screen_fail_reasons"] == {"day_high_cross_screen": 1}
    assert polygon.calls == Counter()
    assert db_session.query(IntradayEventDetail).filter_by(pattern_id=I11_PATTERN_ID).count() == 0


def test_i11_missing_security_type_artifact_fails_closed(db_session):
    result, _polygon = _run_i11(db_session, classifications={})

    assert result.status == "failed"
    assert "not covered by the I12 exclusion artifact" in result.errors[0]["exception"]
    assert db_session.query(IntradayEventDetail).filter_by(pattern_id=I11_PATTERN_ID).count() == 0


def test_i11_validate_write_target_refuses_public_default_without_confirmation():
    with pytest.raises(ValueError, match="I11 corpus"):
        _validate_write_target(schema=None, confirm_live_write=False)
    with pytest.raises(ValueError, match="sequencing gates"):
        _validate_write_target(schema=None, confirm_live_write=True)
    with pytest.raises(ValueError, match="I11 corpus"):
        _validate_write_target(schema="public", confirm_live_write=False)
    with pytest.raises(ValueError, match="sequencing gates"):
        _validate_write_target(schema="public", confirm_live_write=True)
    _validate_write_target(schema="i11_pilot_20260612", confirm_live_write=False)


def test_i11_runner_skip_existing_and_retry_args():
    args = _parse_args([
        "--live",
        "--schema",
        "i11_pilot_20260612",
        "--start-date",
        "2024-01-02",
        "--end-date",
        "2024-01-03",
        "--polygon-cache-dir",
        "/var/tmp/i11_polygon_cache",
        "--skip-existing",
        "--max-db-retries",
        "5",
        "--db-retry-backoff-seconds",
        "0.25",
    ])

    assert args.polygon_cache_dir == "/var/tmp/i11_polygon_cache"
    assert args.skip_existing is True
    assert args.max_db_retries == 5
    assert args.db_retry_backoff_seconds == 0.25


def test_i11_and_i12_required_tables_include_delisted_source():
    assert "fmp_delisted_companies" in I11_CORPUS_REQUIRED_TABLES
    assert "fmp_delisted_companies" in I12_CORPUS_REQUIRED_TABLES


def test_i11_runner_loads_catalyst_tag_artifact(tmp_path):
    path = tmp_path / "tags.json"
    path.write_text(json.dumps([
        {"ticker": "test", "trading_date": DAY.isoformat(), "tags": ["offering", "NT_late_filer"]},
    ]))

    tags = _load_catalyst_tags_artifact(str(path))

    assert tags[("TEST", DAY)] == ["NT_late_filer", "offering"]
