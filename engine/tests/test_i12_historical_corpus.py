from __future__ import annotations

import json
import re
import threading
import time as time_module
from concurrent.futures import TimeoutError as FuturesTimeoutError
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone

import pytest
from sqlalchemy.exc import OperationalError

from alpha.data.contracts import AdapterResponse, LineageMeta, stable_hash
from alpha.data.fmp import FmpBar
from alpha.data.polygon import PolygonBar
from alpha.db.models import (
    DataLineage,
    EvidenceJobRun,
    FeatureSnapshot,
    FmpDelistedCompanyRecord,
    ForwardReturnObservation,
    HistoricalUniverseReconstruction,
    IntradayEventDetail,
    SignalRegistry,
)
from alpha.jobs.i12_historical_corpus import (
    CANDIDATE_SCREEN_STAMP,
    DEFAULT_FETCH_DEADLINE_SECONDS,
    I12HistoricalCorpusJob,
    OUTCOME_CONFIRMED,
    OUTCOME_NEVER_CONFIRMED,
    OUTCOME_POISON_PREMARKET,
)
from alpha.jobs.watchdog import (
    ProviderOutageCircuitBreaker,
    WatchdogState,
    call_with_daemon_deadline,
)
from alpha.jobs.run_i12_historical_corpus import _parse_args, _validate_write_target
from alpha.jobs.runner import run_job
from alpha.jobs.paper_execution import EASTERN
from alpha.market_calendar import next_us_equity_session, previous_us_equity_session
from alpha.evidence.writer import record_feature_snapshot, record_signal
from alpha.ml import security_type_exclusions as ste
from alpha.ml.security_type_exclusions import ExclusionArtifactError, SecurityTypeClassification


DAY = date(2026, 6, 3)
LEAKY_FEATURE_KEY_RE = re.compile(
    r"(full_day|leaky|ret_|mae|mfe|next_open|sessions_to_delist)"
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


class SleepyFmp(FakeFmp):
    def __init__(
        self,
        bars_by_ticker: dict[str, list[FmpBar]],
        *,
        delays: dict[str, float],
    ) -> None:
        super().__init__(bars_by_ticker)
        self.delays = {ticker.upper(): delay for ticker, delay in delays.items()}
        self.calls: Counter[str] = Counter()

    def get_historical_price(self, ticker, from_date=None, to_date=None, **kwargs):
        ticker = ticker.upper()
        self.calls[ticker] += 1
        delay = self.delays.get(ticker, 0.0)
        if delay > 0:
            time_module.sleep(delay)
        return super().get_historical_price(
            ticker,
            from_date=from_date,
            to_date=to_date,
            **kwargs,
        )


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


class InstrumentedPolygon(FakePolygon):
    def __init__(
        self,
        bars_by_ticker_date: dict[tuple[str, date], list[PolygonBar]],
        *,
        delays: dict[tuple[str, date], float] | None = None,
        observed_events: list[tuple[str, dict]] | None = None,
    ) -> None:
        super().__init__(bars_by_ticker_date)
        self.delays = {
            (ticker.upper(), trading_date): delay
            for (ticker, trading_date), delay in (delays or {}).items()
        }
        self.observed_events = observed_events
        self.events_seen_at_fetch: list[list[tuple[str, dict]]] = []

    def get_minute_aggs(self, ticker, from_date, to_date, **kwargs):
        trading_date = date.fromisoformat(from_date)
        key = (ticker.upper(), trading_date)
        self.calls[key] += 1
        if self.observed_events is not None:
            self.events_seen_at_fetch.append(list(self.observed_events))
        delay = self.delays.get(key, 0)
        if delay > 0:
            time_module.sleep(delay)
        bars = self.bars_by_ticker_date[key]
        return AdapterResponse(data=bars, lineage=_lineage("Polygon", ticker, bars))


def _lineage(provider: str, ticker: str, bars: list) -> LineageMeta:
    return LineageMeta(
        provider=provider,
        endpoint="fixture",
        request_timestamp=datetime(2026, 6, 12, tzinfo=timezone.utc),
        asof_timestamp=datetime(2026, 6, 12, tzinfo=timezone.utc),
        raw_payload_hash=stable_hash({"ticker": ticker, "bars": [bar.__dict__ for bar in bars]}),
    )


def _classification(
    security_type: str = "common_stock",
    *,
    ticker: str = "TEST",
) -> SecurityTypeClassification:
    return SecurityTypeClassification(
        ticker=ticker,
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
            historical_universe_reconstruction_id=f"hur-{ticker}-{trading_date}",
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
    ticker_open: float = 10.0,
    ticker_close: float = 11.0,
    prior_close: float = 10.0,
    max_close: float = 20.4,
    volume: int = 700_000,
    include_weekend_duplicate: bool = False,
    lookback_sessions: int = 252,
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
        bars.append(_daily_bar(day, close, close=close, volume=100_000))
    if include_weekend_duplicate:
        bars.append(_daily_bar(date(2026, 2, 22), prior_close, close=prior_close, volume=100_000))
    bars.append(_daily_bar(DAY, ticker_open, close=ticker_close, volume=volume))
    next_day = next_us_equity_session(DAY + timedelta(days=1))
    bars.append(_daily_bar(next_day, ticker_close + 0.2, close=ticker_close + 0.4, volume=volume))
    return sorted(bars, key=lambda bar: bar.date)


def _daily_bar(day: date, open_: float, *, close: float, volume: int) -> FmpBar:
    return FmpBar(
        date=day.isoformat(),
        open=open_,
        high=max(open_, close),
        low=min(open_, close),
        close=close,
        volume=volume,
        split_adjusted_close=close,
        adj_close=close,
    )


def _daily_bar_adjusted(
    day: date,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    split_adjusted_close: float,
    volume: int,
) -> FmpBar:
    return FmpBar(
        date=day.isoformat(),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        split_adjusted_close=split_adjusted_close,
        adj_close=split_adjusted_close,
    )


def _minute_bars(
    *,
    open_: float = 10.0,
    confirmation_volume: float = 1_500.0,
    low_volume: bool = False,
) -> list[PolygonBar]:
    bars: list[PolygonBar] = []
    for minute_index in range(0, 7):
        volume = 100.0 if low_volume else confirmation_volume
        price = open_ + minute_index * 0.03
        bars.append(_minute_bar(minute_index, price, price + 0.05, price - 0.05, price + 0.02, volume))
    bars.append(_minute_bar(385, 10.8, 11.1, 10.7, 11.0, 2_000.0))
    return bars


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


def _run_i12(
    db_session,
    *,
    ticker: str = "TEST",
    daily=None,
    minutes=None,
    classifications=None,
    run_timestamp=None,
    progress_callback=None,
    fetch_deadline_seconds=DEFAULT_FETCH_DEADLINE_SECONDS,
):
    _seed_hur(db_session, ticker)
    if classifications is None:
        classifications = {ticker: _classification()}
    job = I12HistoricalCorpusJob(
        session=db_session,
        fmp_adapter=FakeFmp({ticker: daily or _daily_bars()}),
        polygon_adapter=FakePolygon({(ticker, DAY): minutes or _minute_bars()}),
        start_date=DAY,
        end_date=DAY,
        classification_records=classifications,
        run_timestamp=run_timestamp,
        fetch_deadline_seconds=fetch_deadline_seconds,
        progress_callback=progress_callback,
    )
    return run_job(db_session, job)


def _run_i12_with_polygon(
    db_session,
    *,
    polygon,
    ticker: str = "TEST",
    daily=None,
    classifications=None,
    minute_cache_dir=None,
    skip_existing=False,
    progress_callback=None,
    fetch_deadline_seconds=DEFAULT_FETCH_DEADLINE_SECONDS,
):
    _seed_hur(db_session, ticker)
    if classifications is None:
        classifications = {ticker: _classification()}
    job = I12HistoricalCorpusJob(
        session=db_session,
        fmp_adapter=FakeFmp({ticker: daily or _daily_bars()}),
        polygon_adapter=polygon,
        start_date=DAY,
        end_date=DAY,
        classification_records=classifications,
        minute_cache_dir=minute_cache_dir,
        skip_existing=skip_existing,
        fetch_deadline_seconds=fetch_deadline_seconds,
        progress_callback=progress_callback,
    )
    return run_job(db_session, job)


def _seed_delisted(db_session, ticker: str, delisted_date: date, *, suffix: str = "") -> None:
    db_session.add(
        FmpDelistedCompanyRecord(
            fmp_delisted_company_id=f"delisted-{ticker}-{delisted_date}{suffix}",
            symbol=ticker,
            normalized_symbol=ticker.upper(),
            company_name=f"{ticker} Inc.",
            exchange="NASDAQ",
            exchange_key="NASDAQ",
            ipo_date=date(2020, 1, 1),
            delisted_date=delisted_date,
            delisted_date_key=delisted_date.isoformat(),
            source="fixture",
            source_endpoint="/stable/delisted-companies",
            page_number=0,
            page_limit=0,
            page_row_index=0,
            row_status="active",
            exchange_relevance_status="us_listed_relevant",
            raw_payload_hash=stable_hash({
                "ticker": ticker,
                "delisted_date": delisted_date.isoformat(),
                "suffix": suffix,
            }),
        )
    )
    db_session.flush()


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


def test_i12_corpus_persists_confirmed_entry_and_is_idempotent(db_session):
    result = _run_i12(db_session)

    assert result.status == "finished"
    assert result.metrics["confirmed"] == 1
    assert result.metrics["forward_return_observations_inserted"] == 1
    signal = db_session.query(SignalRegistry).filter_by(pattern_id="I12", ticker="TEST").one()
    assert signal.signal_horizon == "1d"
    assert signal.forward_return_status == "computed"
    assert signal.raw_expected_edge == 0.0
    assert signal.point_in_time_passed is False
    assert signal.lookahead_guard_passed is True
    feature_snapshot = db_session.get(FeatureSnapshot, signal.feature_snapshot_id)
    assert feature_snapshot.point_in_time_passed is False
    assert feature_snapshot.lookahead_guard_passed is True
    detail = db_session.query(IntradayEventDetail).filter_by(ticker="TEST").one()
    assert detail.signal_id == signal.signal_id
    assert detail.outcome == OUTCOME_CONFIRMED
    assert signal.forward_return == pytest.approx(detail.ret_next_open)
    assert detail.ret_conf == pytest.approx((11.0 / detail.entry_price) - 1.0)
    feature_json = json.loads(detail.feature_json)
    assert "ret_conf" not in feature_json
    assert "outcome" not in feature_json
    assert "sessions_to_delist" not in feature_json
    assert "avg20_volume" not in feature_json
    assert "dollar_volume" not in feature_json
    assert "price" not in feature_json
    assert set(feature_json["research_only_leaky"]) == {
        "avg20_volume",
        "dollar_volume",
        "price",
    }
    assert feature_json["candidate_screen"] == CANDIDATE_SCREEN_STAMP
    _assert_feature_json_pit_pure(feature_json)
    labels = json.loads(detail.label_json)
    gate_values = json.loads(detail.gate_values_json)
    assert labels["full_day_volume_ratio_leaky_research_only"] is True
    assert gate_values["full_day_volume_ratio_leaky_research_only"] is True
    observation = db_session.query(ForwardReturnObservation).filter_by(
        pattern_id="I12",
        signal_id=signal.signal_id,
    ).one()
    assert observation.signal_horizon == "1d"
    assert observation.status == "computed"
    assert observation.forward_return == pytest.approx(detail.ret_next_open)
    assert observation.entry_price == pytest.approx(detail.entry_price)
    assert observation.exit_price == pytest.approx(detail.next_open_price)
    assert observation.entry_session_date == DAY.isoformat()
    assert observation.exit_session_date == next_us_equity_session(DAY + timedelta(days=1)).isoformat()

    second = _run_i12(db_session)

    assert second.metrics["inserted_details"] == 0
    assert second.metrics["reused_details"] == 1
    assert second.metrics["forward_return_observations_inserted"] == 0
    assert second.metrics["forward_return_observations_reused"] == 1
    assert db_session.query(SignalRegistry).filter_by(pattern_id="I12", ticker="TEST").count() == 1
    assert db_session.query(IntradayEventDetail).filter_by(pattern_id="I12", ticker="TEST").count() == 1
    assert db_session.query(ForwardReturnObservation).filter_by(pattern_id="I12").count() == 1


def test_i12_corpus_ret_next_open_uses_next_session_open(db_session):
    result = _run_i12(db_session)

    assert result.status == "finished"
    detail = db_session.query(IntradayEventDetail).one()
    next_session_open = 11.2
    assert detail.next_open_price == pytest.approx(next_session_open)
    assert detail.ret_next_open == pytest.approx((next_session_open / detail.entry_price) - 1.0)
    assert detail.ret_next_open != pytest.approx(0.0)


def test_i12_corpus_final_session_range_fetches_next_open(db_session):
    result = _run_i12(db_session)

    assert result.status == "finished"
    detail = db_session.query(IntradayEventDetail).one()
    assert detail.next_open_price == pytest.approx(11.2)
    assert detail.ret_next_open is not None


def test_i12_corpus_sessions_to_delist_counts_future_weekend_delist(db_session):
    _seed_delisted(db_session, "TEST", date(2026, 6, 6))

    result = _run_i12(db_session)

    assert result.status == "finished"
    detail = db_session.query(IntradayEventDetail).one()
    assert detail.sessions_to_delist == 2
    labels = json.loads(detail.label_json)
    assert labels["sessions_to_delist"] == 2


def test_i12_corpus_sessions_to_delist_uses_future_reused_ticker_row(db_session):
    _seed_delisted(db_session, "TEST", date(2024, 1, 5), suffix="-old")
    _seed_delisted(db_session, "TEST", date(2026, 6, 8), suffix="-future")

    result = _run_i12(db_session)

    assert result.status == "finished"
    detail = db_session.query(IntradayEventDetail).one()
    assert detail.sessions_to_delist == 3


def test_i12_corpus_lineage_uses_run_timestamp(db_session):
    run_timestamp = datetime(2026, 6, 12, 8, 0, tzinfo=timezone.utc)

    result = _run_i12(db_session, run_timestamp=run_timestamp)

    assert result.status == "finished"
    rows = db_session.query(DataLineage).all()
    assert rows
    for row in rows:
        request_timestamp = row.request_timestamp
        asof_timestamp = row.asof_timestamp
        if request_timestamp.tzinfo is None:
            request_timestamp = request_timestamp.replace(tzinfo=timezone.utc)
        if asof_timestamp.tzinfo is None:
            asof_timestamp = asof_timestamp.replace(tzinfo=timezone.utc)
        assert request_timestamp == run_timestamp
        assert asof_timestamp == run_timestamp


def test_i12_corpus_failed_run_records_partial_metrics_and_heartbeat(db_session):
    events = []
    result = _run_i12(
        db_session,
        classifications={},
        progress_callback=lambda event, payload: events.append((event, dict(payload))),
    )

    assert result.status == "failed"
    assert "not covered by the I12 exclusion artifact" in result.errors[0]["exception"]
    assert any(event == "ticker_day_progress" for event, _payload in events)
    run = (
        db_session.query(EvidenceJobRun)
        .order_by(EvidenceJobRun.started_at.desc())
        .first()
    )
    metrics = json.loads(run.metric_json)
    assert metrics["ticker_days_scanned"] == 1
    assert metrics["trading_date_count"] == 1


def test_i12_corpus_non_candidate_does_not_fetch_minutes_or_write_detail(db_session):
    polygon = FakePolygon({})
    result = _run_i12_with_polygon(
        db_session,
        polygon=polygon,
        daily=_daily_bars(prior_close=50.0, max_close=51.0, ticker_open=50.0, ticker_close=51.0),
    )

    assert result.status == "finished"
    assert result.metrics["ticker_days_scanned"] == 1
    assert result.metrics["candidates"] == 0
    assert result.metrics["candidates_screened_out"] == 1
    assert result.metrics["candidate_screen_fail_reasons"] == {"drawdown_screen": 1}
    assert polygon.calls == Counter()
    assert db_session.query(IntradayEventDetail).count() == 0
    assert db_session.query(SignalRegistry).filter_by(pattern_id="I12").count() == 0


def test_i12_corpus_digest_only_bar_lineage(db_session):
    result = _run_i12(db_session)

    assert result.status == "finished"
    rows = db_session.query(DataLineage).order_by(DataLineage.provider).all()
    assert rows
    for row in rows:
        assert row.raw_payload_json is None
        flags = json.loads(row.data_quality_flags)
        assert flags["bar_count"] > 0
        assert flags["bars_digest"] == row.raw_payload_hash


def test_i12_corpus_rejects_content_revision_for_existing_identity(db_session):
    result = _run_i12(db_session)
    assert result.status == "finished"

    revised = _run_i12(db_session, daily=_daily_bars(ticker_close=11.2))

    assert revised.status == "failed"
    assert "event content changed for existing identity" in revised.errors[0]["exception"]
    assert db_session.query(IntradayEventDetail).filter_by(pattern_id="I12", ticker="TEST").count() == 1


def test_i12_corpus_split_basis_mismatch_records_detail_without_signal(db_session):
    result = _run_i12(db_session, minutes=_minute_bars(open_=10.3))

    assert result.status == "finished"
    assert result.metrics["artifact_excluded"] == 1
    assert result.metrics["inserted_signals"] == 0
    assert result.metrics["forward_return_observations_inserted"] == 0
    assert db_session.query(SignalRegistry).filter_by(pattern_id="I12").count() == 0
    assert db_session.query(ForwardReturnObservation).filter_by(pattern_id="I12").count() == 0
    detail = db_session.query(IntradayEventDetail).one()
    assert detail.outcome == OUTCOME_CONFIRMED
    assert detail.split_basis_mismatch is True
    assert detail.is_ml_excluded is True
    assert detail.ml_exclusion_reason == "split_basis_mismatch"
    assert detail.signal_id is None


def test_i12_corpus_primary_label_unavailable_records_detail_without_signal(db_session):
    daily = _daily_bars()[:-1]
    result = _run_i12(db_session, daily=daily)

    assert result.status == "finished"
    assert result.metrics["confirmed"] == 1
    assert result.metrics["inserted_signals"] == 0
    assert result.metrics["forward_return_observations_inserted"] == 0
    assert db_session.query(SignalRegistry).filter_by(pattern_id="I12").count() == 0
    assert db_session.query(ForwardReturnObservation).filter_by(pattern_id="I12").count() == 0
    detail = db_session.query(IntradayEventDetail).one()
    assert detail.outcome == OUTCOME_CONFIRMED
    assert detail.signal_id is None
    assert detail.ret_next_open is None
    assert detail.next_open_price is None
    assert detail.is_ml_excluded is True
    assert detail.ml_exclusion_reason == "primary_label_unavailable"
    feature_json = json.loads(detail.feature_json)
    assert feature_json["is_ml_excluded"] is True
    assert feature_json["ml_exclusion_reason"] == "primary_label_unavailable"


def test_i12_corpus_uses_disk_cached_polygon_minutes(db_session, tmp_path):
    first_polygon = FakePolygon({("TEST", DAY): _minute_bars()})
    first = _run_i12_with_polygon(
        db_session,
        polygon=first_polygon,
        minute_cache_dir=tmp_path,
    )

    assert first.status == "finished"
    assert first.metrics["minute_cache_misses"] == 1
    assert first_polygon.calls[("TEST", DAY)] == 1

    second_polygon = FakePolygon({})
    second = _run_i12_with_polygon(
        db_session,
        polygon=second_polygon,
        minute_cache_dir=tmp_path,
    )

    assert second.status == "finished"
    assert second.metrics["minute_cache_hits"] == 1
    assert second.metrics["minute_cache_misses"] == 0
    assert second.metrics["reused_details"] == 1
    assert second_polygon.calls == Counter()


def test_i12_corpus_skip_existing_avoids_refetching_processed_ticker_day(db_session):
    first_polygon = FakePolygon({("TEST", DAY): _minute_bars()})
    first = _run_i12_with_polygon(
        db_session,
        polygon=first_polygon,
    )

    assert first.status == "finished"
    assert first.metrics["inserted_details"] == 1

    second_polygon = FakePolygon({})
    second = _run_i12_with_polygon(
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
    assert db_session.query(IntradayEventDetail).filter_by(pattern_id="I12", ticker="TEST").count() == 1
    assert db_session.query(SignalRegistry).filter_by(pattern_id="I12", ticker="TEST").count() == 1


def test_i12_corpus_retries_transient_db_disconnect_once(db_session):
    _seed_hur(db_session, "TEST")
    job = I12HistoricalCorpusJob(
        session=db_session,
        fmp_adapter=FakeFmp({"TEST": _daily_bars()}),
        polygon_adapter=FakePolygon({("TEST", DAY): _minute_bars()}),
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
    assert result.metrics["forward_return_observations_inserted"] == 1
    assert db_session.query(IntradayEventDetail).filter_by(pattern_id="I12", ticker="TEST").count() == 1
    assert db_session.query(SignalRegistry).filter_by(pattern_id="I12", ticker="TEST").count() == 1
    assert db_session.query(ForwardReturnObservation).filter_by(pattern_id="I12", ticker="TEST").count() == 1


def test_i12_corpus_never_confirmed_control_has_no_signal(db_session):
    result = _run_i12(db_session, minutes=_minute_bars(low_volume=True))

    assert result.status == "finished"
    assert result.metrics["never_confirmed"] == 1
    assert db_session.query(SignalRegistry).filter_by(pattern_id="I12").count() == 0
    detail = db_session.query(IntradayEventDetail).one()
    assert detail.outcome == OUTCOME_NEVER_CONFIRMED
    assert detail.signal_id is None


def test_i12_corpus_evaluates_thirty_prior_sessions_and_stamps_insufficient_history(db_session):
    result = _run_i12(db_session, daily=_daily_bars(lookback_sessions=30))

    assert result.status == "finished"
    assert result.metrics["confirmed"] == 1
    detail = db_session.query(IntradayEventDetail).one()
    feature_json = json.loads(detail.feature_json)
    assert feature_json["insufficient_history"] == {"prior252": True}


def test_i12_corpus_off_low252_uses_split_consistent_low(db_session):
    days: list[date] = []
    cursor = DAY
    for _ in range(252):
        cursor = previous_us_equity_session(cursor)
        days.append(cursor)
    days.reverse()
    bars: list[FmpBar] = []
    for idx, day in enumerate(days):
        if idx == 0:
            bars.append(_daily_bar(day, 20.4, close=20.4, volume=100_000))
        elif idx == 1:
            bars.append(
                _daily_bar_adjusted(
                    day,
                    open_=100.0,
                    high=110.0,
                    low=50.0,
                    close=100.0,
                    split_adjusted_close=10.0,
                    volume=100_000,
                )
            )
        else:
            bars.append(_daily_bar(day, 10.0, close=10.0, volume=100_000))
    bars.append(_daily_bar(DAY, 10.0, close=11.0, volume=700_000))
    next_day = next_us_equity_session(DAY)
    bars.append(_daily_bar(next_day, 11.2, close=11.4, volume=700_000))

    result = _run_i12(db_session, daily=bars)

    assert result.status == "finished"
    detail = db_session.query(IntradayEventDetail).one()
    feature_json = json.loads(detail.feature_json)
    assert feature_json["off_low252"] == pytest.approx(1.0)


def test_i12_corpus_poison_gap_writes_daily_only_control_without_minute_fetch(db_session):
    polygon = FakePolygon({})
    result = _run_i12_with_polygon(
        db_session,
        polygon=polygon,
        daily=_daily_bars(ticker_open=9.3, ticker_close=9.0),
    )

    assert result.status == "finished"
    assert result.metrics["candidates"] == 0
    assert result.metrics["candidates_screened_out"] == 0
    assert result.metrics["poison_premarket"] == 1
    assert result.metrics["outcome_counts"] == {OUTCOME_POISON_PREMARKET: 1}
    assert polygon.calls == Counter()
    assert db_session.query(SignalRegistry).filter_by(pattern_id="I12").count() == 0
    assert db_session.query(FeatureSnapshot).filter_by(pattern_id="I12").count() == 0
    detail = db_session.query(IntradayEventDetail).one()
    assert detail.outcome == OUTCOME_POISON_PREMARKET
    assert detail.signal_id is None
    assert detail.ret_conf is None
    assert detail.ret_open_close == pytest.approx((9.0 / 9.3) - 1.0)
    assert detail.ret_next_open is not None
    feature_json = json.loads(detail.feature_json)
    assert "avg20_volume" not in feature_json
    assert "dollar_volume" not in feature_json
    assert "price" not in feature_json
    assert set(feature_json["research_only_leaky"]) == {
        "avg20_volume",
        "dollar_volume",
        "price",
    }
    _assert_feature_json_pit_pure(feature_json)
    assert feature_json.get("minute_price_basis") is None
    labels = json.loads(detail.label_json)
    gate_values = json.loads(detail.gate_values_json)
    assert labels["full_day_volume_ratio_leaky_research_only"] is True
    assert gate_values["full_day_volume_ratio_leaky_research_only"] is True
    daily_lineage = db_session.query(DataLineage).filter_by(provider="FMP").one()
    assert json.loads(detail.data_lineage_ids_json) == [daily_lineage.data_lineage_id]
    assert db_session.query(DataLineage).filter_by(provider="Polygon").count() == 0

    second_polygon = FakePolygon({})
    second = _run_i12_with_polygon(
        db_session,
        polygon=second_polygon,
        daily=_daily_bars(ticker_open=9.3, ticker_close=9.0),
    )

    assert second.status == "finished"
    assert second.metrics["inserted_details"] == 0
    assert second.metrics["reused_details"] == 1
    assert second_polygon.calls == Counter()
    assert db_session.query(IntradayEventDetail).count() == 1


def test_i12_corpus_gap_up_i1_exclusion_still_writes_no_row(db_session):
    polygon = FakePolygon({})
    result = _run_i12_with_polygon(
        db_session,
        polygon=polygon,
        daily=_daily_bars(ticker_open=10.5, ticker_close=10.8),
    )

    assert result.status == "finished"
    assert result.metrics["candidates"] == 0
    assert result.metrics["candidates_screened_out"] == 1
    assert result.metrics["candidate_screen_fail_reasons"] == {"gap_screen_i1_exclusion": 1}
    assert polygon.calls == Counter()
    assert db_session.query(SignalRegistry).filter_by(pattern_id="I12").count() == 0
    assert db_session.query(IntradayEventDetail).count() == 0


def test_i12_corpus_gap_exactly_minus_five_percent_remains_minute_candidate(db_session):
    polygon = FakePolygon({("TEST", DAY): _minute_bars(open_=9.5)})
    result = _run_i12_with_polygon(
        db_session,
        polygon=polygon,
        daily=_daily_bars(ticker_open=9.5, ticker_close=10.0),
    )

    assert result.status == "finished"
    assert result.metrics["candidates"] == 1
    assert result.metrics["confirmed"] == 1
    assert polygon.calls[("TEST", DAY)] == 1
    assert db_session.query(IntradayEventDetail).one().outcome == OUTCOME_CONFIRMED


def test_i12_minute_fetch_watchdog_quarantines_and_continues(db_session):
    slow_ticker = "ASLOW"
    fast_ticker = "BFAST"
    _seed_hur(db_session, slow_ticker)
    _seed_hur(db_session, fast_ticker)
    polygon = InstrumentedPolygon(
        {
            (slow_ticker, DAY): _minute_bars(),
            (fast_ticker, DAY): _minute_bars(),
        },
        delays={(slow_ticker, DAY): 0.8},
    )
    job = I12HistoricalCorpusJob(
        session=db_session,
        fmp_adapter=FakeFmp({
            slow_ticker: _daily_bars(),
            fast_ticker: _daily_bars(),
        }),
        polygon_adapter=polygon,
        start_date=DAY,
        end_date=DAY,
        classification_records={
            slow_ticker: _classification(ticker=slow_ticker),
            fast_ticker: _classification(ticker=fast_ticker),
        },
        fetch_deadline_seconds=0.05,
    )

    result = run_job(db_session, job)

    assert result.status == "finished"
    assert result.metrics["watchdog_timeouts"] == 1
    assert result.metrics["fetch_errors"] == 1
    assert result.metrics["quarantined"] == 1
    assert result.metrics["confirmed"] == 1
    assert result.errors[0] | {
        "ticker": slow_ticker,
        "trading_date": DAY.isoformat(),
        "error": "fetch_watchdog_timeout",
        "deadline_seconds": 0.05,
    } == result.errors[0]
    assert polygon.calls[(slow_ticker, DAY)] == 1
    assert polygon.calls[(fast_ticker, DAY)] == 1
    assert db_session.query(IntradayEventDetail).filter_by(ticker=fast_ticker).count() == 1
    assert db_session.query(IntradayEventDetail).filter_by(ticker=slow_ticker).count() == 0


def test_i12_daily_fmp_fetch_watchdog_quarantines_and_continues(db_session):
    slow_ticker = "ASLOW"
    fast_ticker = "BFAST"
    _seed_hur(db_session, slow_ticker)
    _seed_hur(db_session, fast_ticker)
    fmp = SleepyFmp(
        {
            slow_ticker: _daily_bars(),
            fast_ticker: _daily_bars(),
        },
        delays={slow_ticker: 0.8},
    )
    polygon = FakePolygon({
        (slow_ticker, DAY): _minute_bars(),
        (fast_ticker, DAY): _minute_bars(),
    })
    job = I12HistoricalCorpusJob(
        session=db_session,
        fmp_adapter=fmp,
        polygon_adapter=polygon,
        start_date=DAY,
        end_date=DAY,
        classification_records={
            slow_ticker: _classification(ticker=slow_ticker),
            fast_ticker: _classification(ticker=fast_ticker),
        },
        fetch_deadline_seconds=0.05,
    )

    result = run_job(db_session, job)

    assert result.status == "finished"
    assert result.metrics["watchdog_timeouts"] == 1
    assert result.metrics["fetch_errors"] == 1
    assert result.metrics["quarantined"] == 1
    assert result.metrics["confirmed"] == 1
    assert result.errors[0] | {
        "ticker": slow_ticker,
        "trading_date": DAY.isoformat(),
        "error": "daily_fetch_watchdog_timeout",
        "deadline_seconds": 0.05,
    } == result.errors[0]
    assert fmp.calls[slow_ticker] == 1
    assert fmp.calls[fast_ticker] == 1
    assert polygon.calls[(slow_ticker, DAY)] == 0
    assert polygon.calls[(fast_ticker, DAY)] == 1
    assert db_session.query(IntradayEventDetail).filter_by(ticker=fast_ticker).count() == 1
    assert db_session.query(IntradayEventDetail).filter_by(ticker=slow_ticker).count() == 0


def test_i12_fetch_watchdog_consecutive_timeout_streak_trips_breaker(db_session):
    first_ticker = "ASLOW"
    second_ticker = "BSLOW"
    for ticker in (first_ticker, second_ticker):
        _seed_hur(db_session, ticker)
    fmp = SleepyFmp(
        {
            first_ticker: _daily_bars(),
            second_ticker: _daily_bars(),
        },
        delays={
            first_ticker: 0.06,
            second_ticker: 0.06,
        },
    )
    polygon = FakePolygon({
        (first_ticker, DAY): _minute_bars(),
        (second_ticker, DAY): _minute_bars(),
    })
    job = I12HistoricalCorpusJob(
        session=db_session,
        fmp_adapter=fmp,
        polygon_adapter=polygon,
        start_date=DAY,
        end_date=DAY,
        classification_records={
            first_ticker: _classification(ticker=first_ticker),
            second_ticker: _classification(ticker=second_ticker),
        },
        fetch_deadline_seconds=0.01,
        max_outstanding_fetch_timeouts=10,
        max_consecutive_fetch_timeouts=2,
    )

    result = run_job(db_session, job)

    assert result.status == "partial_failed"
    assert result.metrics["watchdog_timeouts"] == 2
    breaker = next(error for error in result.errors if error["error"] == "provider_outage_circuit_breaker")
    assert breaker["circuit_reason"] == "watchdog_timeout:max_consecutive_timeouts"
    assert breaker["max_consecutive_fetch_timeouts"] == 2
    assert breaker["max_outstanding_fetch_timeouts"] == 10
    assert fmp.calls[first_ticker] == 1
    assert fmp.calls[second_ticker] == 1
    assert polygon.calls[(first_ticker, DAY)] == 0
    assert polygon.calls[(second_ticker, DAY)] == 0

    deadline = time_module.monotonic() + 1.0
    while job._fetch_watchdog.outstanding_timeouts and time_module.monotonic() < deadline:
        time_module.sleep(0.01)
    assert job._fetch_watchdog.outstanding_timeouts == 0


def test_i12_daily_cache_hit_does_not_reset_provider_timeout_streak(db_session):
    cache_ticker = "BCACHE"
    first_timeout_ticker = "ASLOW"
    second_timeout_ticker = "CSLOW"
    next_day = next_us_equity_session(DAY + timedelta(days=1))
    _seed_hur(db_session, cache_ticker, trading_date=DAY)
    for ticker in (first_timeout_ticker, cache_ticker, second_timeout_ticker):
        _seed_hur(db_session, ticker, trading_date=next_day)
    fmp = SleepyFmp(
        {
            cache_ticker: _daily_bars(volume=100),
            first_timeout_ticker: _daily_bars(),
            second_timeout_ticker: _daily_bars(),
        },
        delays={
            first_timeout_ticker: 0.06,
            second_timeout_ticker: 0.06,
        },
    )
    polygon = FakePolygon({
        (first_timeout_ticker, next_day): _minute_bars(),
        (second_timeout_ticker, next_day): _minute_bars(),
    })
    job = I12HistoricalCorpusJob(
        session=db_session,
        fmp_adapter=fmp,
        polygon_adapter=polygon,
        start_date=DAY,
        end_date=next_day,
        classification_records={
            first_timeout_ticker: _classification(ticker=first_timeout_ticker),
            cache_ticker: _classification(ticker=cache_ticker),
            second_timeout_ticker: _classification(ticker=second_timeout_ticker),
        },
        fetch_deadline_seconds=0.01,
        max_outstanding_fetch_timeouts=10,
        max_consecutive_fetch_timeouts=2,
    )

    result = run_job(db_session, job)

    assert result.status == "partial_failed"
    assert result.metrics["watchdog_timeouts"] == 2
    assert fmp.calls[cache_ticker] == 1
    assert fmp.calls[first_timeout_ticker] == 1
    assert fmp.calls[second_timeout_ticker] == 1
    breaker = next(error for error in result.errors if error["error"] == "provider_outage_circuit_breaker")
    assert breaker["circuit_reason"] == "watchdog_timeout:max_consecutive_timeouts"
    assert breaker["consecutive_watchdog_timeouts"] == 2
    assert breaker["max_consecutive_fetch_timeouts"] == 2

    deadline = time_module.monotonic() + 1.0
    while job._fetch_watchdog.outstanding_timeouts and time_module.monotonic() < deadline:
        time_module.sleep(0.01)
    assert job._fetch_watchdog.outstanding_timeouts == 0


def test_daemon_deadline_unit_bounds_sleeping_callable():
    state = WatchdogState(max_outstanding_timeouts=5)

    started = time_module.monotonic()
    with pytest.raises(FuturesTimeoutError):
        call_with_daemon_deadline(
            lambda: time_module.sleep(0.5),
            timeout_seconds=0.02,
            thread_name="unit-sleeping-watchdog",
            state=state,
        )
    elapsed = time_module.monotonic() - started

    assert elapsed < 0.2
    assert state.total_timeouts == 1
    assert state.outstanding_timeouts == 1


def test_daemon_deadline_circuit_breaker_bounds_persistent_hangs():
    state = WatchdogState(max_outstanding_timeouts=3)
    before = threading.active_count()

    for _idx in range(2):
        with pytest.raises(FuturesTimeoutError):
            call_with_daemon_deadline(
                lambda: time_module.sleep(60),
                timeout_seconds=0.01,
                thread_name="unit-persistent-watchdog",
                state=state,
            )
    with pytest.raises(ProviderOutageCircuitBreaker) as excinfo:
        call_with_daemon_deadline(
            lambda: time_module.sleep(60),
            timeout_seconds=0.01,
            thread_name="unit-persistent-watchdog",
            state=state,
        )

    assert excinfo.value.payload["error"] == "provider_outage_circuit_breaker"
    assert state.circuit_open is True
    assert state.outstanding_timeouts == 3
    assert threading.active_count() - before <= 3


def test_daemon_deadline_thread_start_failure_trips_breaker(monkeypatch):
    state = WatchdogState(max_outstanding_timeouts=10)

    def fail_start(self):  # noqa: ANN001
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", fail_start)

    with pytest.raises(ProviderOutageCircuitBreaker) as excinfo:
        call_with_daemon_deadline(
            lambda: None,
            timeout_seconds=0.01,
            thread_name="unit-start-failure",
            state=state,
        )

    assert excinfo.value.payload["error"] == "provider_outage_circuit_breaker"
    assert state.thread_start_failures == 1
    assert state.circuit_open is True


def test_daemon_deadline_late_worker_decrements_outstanding_timeout():
    state = WatchdogState(max_outstanding_timeouts=2)
    finished = threading.Event()

    def slow_finish():
        time_module.sleep(0.04)
        finished.set()
        return "late"

    with pytest.raises(FuturesTimeoutError):
        call_with_daemon_deadline(
            slow_finish,
            timeout_seconds=0.01,
            thread_name="unit-late-finish-watchdog",
            state=state,
        )

    assert finished.wait(0.5)
    assert state.total_timeouts == 1
    assert state.outstanding_timeouts == 0
    assert state.circuit_open is False


def test_daemon_deadline_cache_hit_resets_recoverable_timeout_streak():
    state = WatchdogState(max_outstanding_timeouts=2, max_consecutive_timeouts=2)

    for _idx in range(3):
        with pytest.raises(FuturesTimeoutError):
            call_with_daemon_deadline(
                lambda: time_module.sleep(0.03),
                timeout_seconds=0.01,
                thread_name="unit-recoverable-watchdog",
                state=state,
            )
        time_module.sleep(0.05)
        assert state.outstanding_timeouts == 0
        state.record_success()

    assert state.total_timeouts == 3
    assert state.consecutive_timeouts == 0
    assert state.circuit_open is False


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_daemon_deadline_process_control_start_exceptions_propagate(
    monkeypatch,
    exc_type,
):
    state = WatchdogState(max_outstanding_timeouts=10)

    def interrupt_start(self):  # noqa: ANN001
        raise exc_type()

    monkeypatch.setattr(threading.Thread, "start", interrupt_start)

    with pytest.raises(exc_type):
        call_with_daemon_deadline(
            lambda: None,
            timeout_seconds=0.01,
            thread_name="unit-process-control",
            state=state,
        )

    assert state.thread_start_failures == 0
    assert state.circuit_open is False


def test_i12_progress_records_uncached_fetch_heartbeat_before_fetch(db_session):
    events: list[tuple[str, dict]] = []
    polygon = InstrumentedPolygon(
        {("TEST", DAY): _minute_bars()},
        observed_events=events,
    )

    result = _run_i12_with_polygon(
        db_session,
        polygon=polygon,
        progress_callback=lambda event, payload: events.append((event, dict(payload))),
    )

    assert result.status == "finished"
    fetch_payload = next(payload for event, payload in events if event == "minute_fetch_start")
    assert fetch_payload["ticker"] == "TEST"
    assert fetch_payload["trading_date"] == DAY.isoformat()
    assert fetch_payload["deadline_seconds"] == DEFAULT_FETCH_DEADLINE_SECONDS
    assert fetch_payload["cache_status"] == "miss"
    assert fetch_payload["wall_clock_utc"].endswith("Z")
    datetime.fromisoformat(fetch_payload["wall_clock_utc"].replace("Z", "+00:00"))
    assert any(event == "minute_fetch_start" for event, _payload in polygon.events_seen_at_fetch[0])
    batch_finish_payload = next(payload for event, payload in events if event == "batch_finish")
    assert batch_finish_payload["wall_clock_utc"].endswith("Z")


def test_i12_corpus_poison_gap_low_full_day_volume_writes_no_row(db_session):
    polygon = FakePolygon({})
    result = _run_i12_with_polygon(
        db_session,
        polygon=polygon,
        daily=_daily_bars(ticker_open=9.4, ticker_close=9.0, volume=100_000),
    )

    assert result.status == "finished"
    assert result.metrics["candidates"] == 0
    assert result.metrics["candidates_screened_out"] == 1
    assert result.metrics["candidate_screen_fail_reasons"] == {"full_day_volume_screen": 1}
    assert polygon.calls == Counter()
    assert db_session.query(IntradayEventDetail).count() == 0


def test_i12_corpus_real_exclusion_loader_fails_closed_for_absent_ticker(db_session):
    ticker = "ZZZZZZZZ"
    _seed_hur(db_session, ticker)
    ste.load_artifact_metadata.cache_clear()
    ste.load_classifications.cache_clear()
    ste.non_common_tickers.cache_clear()
    job = I12HistoricalCorpusJob(
        session=db_session,
        fmp_adapter=FakeFmp({ticker: _daily_bars()}),
        polygon_adapter=FakePolygon({}),
        start_date=DAY,
        end_date=DAY,
        classification_records=None,
    )

    result = run_job(db_session, job)

    assert result.status == "failed"
    assert "not covered by the I12 exclusion artifact" in result.errors[0]["exception"]
    assert db_session.query(IntradayEventDetail).count() == 0


def test_i12_corpus_fails_closed_when_type_artifact_misses_ticker(db_session):
    result = _run_i12(db_session, classifications={})

    assert result.status == "failed"
    assert "not covered by the I12 exclusion artifact" in result.errors[0]["exception"]
    assert db_session.query(IntradayEventDetail).count() == 0


def test_i12_corpus_stamps_ml_excluded_rows(db_session):
    result = _run_i12(db_session, classifications={"TEST": _classification("etf")})

    assert result.status == "finished"
    assert result.metrics["excluded_by_type"] == 1
    assert result.metrics["inserted_signals"] == 0
    assert result.metrics["forward_return_observations_inserted"] == 0
    assert db_session.query(SignalRegistry).filter_by(pattern_id="I12").count() == 0
    assert db_session.query(ForwardReturnObservation).filter_by(pattern_id="I12").count() == 0
    detail = db_session.query(IntradayEventDetail).one()
    assert detail.signal_id is None
    assert detail.is_ml_excluded is True
    assert detail.security_type == "etf"
    assert detail.ml_exclusion_reason == "fixture"
    feature_json = json.loads(detail.feature_json)
    assert feature_json["is_ml_excluded"] is True
    assert feature_json["security_type"] == "etf"


def test_i12_corpus_signal_identity_conflict_is_controlled_error(db_session):
    lineage = db_session.query(DataLineage).first()
    if lineage is None:
        lineage = DataLineage(
            provider="fixture",
            endpoint="fixture",
            request_timestamp=datetime(2026, 6, 12, tzinfo=timezone.utc),
            asof_timestamp=datetime(2026, 6, 12, tzinfo=timezone.utc),
            raw_payload_hash="fixture",
            data_quality_flags="{}",
        )
        db_session.add(lineage)
        db_session.flush()
    feature = record_feature_snapshot(
        db_session,
        pattern_id="I12",
        ticker="TEST",
        asof_timestamp=datetime(2026, 6, 3, 13, 35, tzinfo=timezone.utc),
        features={"fixture": True},
        data_lineage_ids=[lineage.data_lineage_id],
        job_run_id=None,
        feature_manifest_version="fixture",
        fidelity_tier="fixture",
        point_in_time_passed=True,
        lookahead_guard_passed=True,
        input_hashes={"fixture": "fixture"},
    )
    record_signal(
        db_session,
        pattern_id="I12",
        ticker="TEST",
        direction="long",
        signal_timestamp=datetime(2026, 6, 3, 13, 35, tzinfo=timezone.utc),
        raw_signal_strength=1.0,
        raw_expected_edge=0.0,
        feature_snapshot_id=feature.feature_snapshot_id,
        job_run_id=None,
        signal_horizon="0d",
        thesis_category="fixture",
        route_class="fixture",
        fidelity_tier="fixture",
        data_confidence=1.0,
        data_lineage_ids=[lineage.data_lineage_id],
        trading_date=DAY.isoformat(),
        next_execution_session=DAY.isoformat(),
        detector_version="fixture",
        point_in_time_passed=True,
        lookahead_guard_passed=True,
        signal_event_sequence=1,
        signal_identity_hash="different-existing-hash",
    )
    db_session.commit()

    result = _run_i12(db_session)

    assert result.status == "failed"
    assert "i12_signal_identity_conflict" in result.errors[0]["exception"]
    assert "different-existing-hash" in result.errors[0]["exception"]


def test_i12_corpus_skips_non_session_daily_bars_with_telemetry(db_session):
    result = _run_i12(db_session, daily=_daily_bars(include_weekend_duplicate=True))

    assert result.status == "finished"
    assert result.metrics["non_session_bars_skipped"] == 1
    assert result.metrics["non_session_bar_skip_sample"] == [
        {"ticker": "TEST", "date": "2026-02-22"}
    ]


def test_i12_corpus_runner_refuses_public_without_confirmation():
    with pytest.raises(ValueError):
        _validate_write_target(schema=None, confirm_live_write=False)
    with pytest.raises(ValueError):
        _validate_write_target(schema="public", confirm_live_write=False)

    _validate_write_target(schema="scratch_i12", confirm_live_write=False)


def test_i12_runner_minute_cache_alias_and_skip_existing_args():
    args = _parse_args([
        "--live",
        "--confirm-live-write",
        "--start-date",
        "2024-01-02",
        "--end-date",
        "2024-01-03",
        "--minute-cache-dir",
        "/var/tmp/i12_minutes",
        "--skip-existing",
        "--max-db-retries",
        "5",
        "--fetch-deadline-seconds",
        "7.5",
        "--max-outstanding-fetch-timeouts",
        "4",
        "--max-consecutive-fetch-timeouts",
        "6",
    ])

    assert args.minute_cache_dir == "/var/tmp/i12_minutes"
    assert args.skip_existing is True
    assert args.max_db_retries == 5
    assert args.fetch_deadline_seconds == 7.5
    assert args.max_outstanding_fetch_timeouts == 4
    assert args.max_consecutive_fetch_timeouts == 6

    alias = _parse_args([
        "--live",
        "--confirm-live-write",
        "--start-date",
        "2024-01-02",
        "--end-date",
        "2024-01-03",
        "--polygon-cache-dir",
        "/var/tmp/i12_polygon_alias",
    ])
    assert alias.minute_cache_dir == "/var/tmp/i12_polygon_alias"
