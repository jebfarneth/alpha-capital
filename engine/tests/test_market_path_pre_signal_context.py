from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from alpha.data.contracts import AdapterResponse, LineageMeta, stable_hash
from alpha.data.fmp import FmpBar, HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.db.models import (
    DataLineage,
    FeatureSnapshot,
    HistoricalUniverseReconstruction,
    MarketPathFeature,
    MarketPathPreSignalContext,
    MarketPathPreSignalLink,
)
from alpha.evidence.writer import (
    record_data_lineage,
    record_feature_snapshot,
    record_signal,
)
from alpha.jobs.market_path_features import FEATURE_VERSION as FORWARD_FEATURE_VERSION
from alpha.jobs.market_path_features import MarketPathFeatureJob
from alpha.jobs.historical_m4_signal_selector import (
    HISTORICAL_M4_REPLAY_RECONSTRUCTION_METHOD,
)
from alpha.jobs import run_market_path_pre_signal_context as pre_signal_runner
from alpha.jobs.market_path_pre_signal_context import (
    FEATURE_VERSION,
    ROW_STATUS_COMPUTED,
    ROW_STATUS_OUTSIDE_UNIVERSE,
    MarketPathPreSignalContextJob,
)
from alpha.jobs.run_market_path_pre_signal_context import _parse_args, _validate_write_target
from alpha.jobs.runner import run_job
from alpha.market_calendar import is_us_equity_session, previous_us_equity_session


RUN_TS = datetime(2026, 6, 13, 16, 0, tzinfo=timezone.utc)


class FakeFmpAdapter:
    def __init__(self, bars_by_ticker: dict[str, list[FmpBar]]) -> None:
        self.bars_by_ticker = {
            ticker.upper(): bars for ticker, bars in bars_by_ticker.items()
        }
        self.calls: list[dict] = []
        self.cache_misses = 0

    def get_historical_price(self, ticker, from_date=None, to_date=None, asof=None, **kwargs):
        ticker = ticker.upper()
        self.cache_misses += 1
        self.calls.append({
            "ticker": ticker,
            "from_date": from_date,
            "to_date": to_date,
            "asof": asof,
            "kwargs": kwargs,
        })
        bars = self.bars_by_ticker.get(ticker, [])
        if from_date is not None:
            bars = [bar for bar in bars if date.fromisoformat(bar.date) >= from_date]
        if to_date is not None:
            bars = [bar for bar in bars if date.fromisoformat(bar.date) <= to_date]
        return AdapterResponse(
            data=bars,
            lineage=LineageMeta(
                provider="FMP",
                endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
                request_timestamp=RUN_TS,
                asof_timestamp=asof or RUN_TS,
                raw_payload_hash=stable_hash([bar.__dict__ for bar in bars]),
            ),
        )


def _add_signal(
    db_session,
    *,
    ticker: str = "SETU",
    signal_day: date = date(2026, 6, 5),
):
    lineage = record_data_lineage(
        db_session,
        provider="fixture",
        endpoint="/fixture/signal",
        asof_timestamp=datetime.combine(signal_day, datetime.min.time(), timezone.utc),
        raw_payload={"ticker": ticker, "signal_day": signal_day.isoformat()},
    )
    feature = record_feature_snapshot(
        db_session,
        pattern_id="M4",
        ticker=ticker,
        asof_timestamp=datetime.combine(signal_day, datetime.min.time(), timezone.utc),
        features={"ticker": ticker, "detector_signal_identity_hash": f"{ticker}-{signal_day}"},
        data_lineage_ids=[lineage.data_lineage_id],
        point_in_time_passed=True,
        lookahead_guard_passed=True,
    )
    return record_signal(
        db_session,
        pattern_id="M4",
        ticker=ticker,
        direction="long",
        signal_timestamp=datetime.combine(signal_day, datetime.min.time(), timezone.utc),
        raw_signal_strength=1.0,
        raw_expected_edge=0.01,
        feature_snapshot_id=feature.feature_snapshot_id,
        signal_horizon="15d",
        trading_date=signal_day.isoformat(),
        next_execution_session=signal_day.isoformat(),
        point_in_time_passed=True,
        lookahead_guard_passed=True,
        signal_identity_hash=f"{ticker}-{signal_day}",
        data_lineage_ids=[lineage.data_lineage_id],
    )


def _stamp_signal_historical_m4_replay(db_session, signal) -> None:
    feature = db_session.get(FeatureSnapshot, signal.feature_snapshot_id)
    payload = json.loads(feature.feature_json)
    payload["reconstruction_method"] = HISTORICAL_M4_REPLAY_RECONSTRUCTION_METHOD
    payload["historical_replay"] = {
        "reconstruction_method": HISTORICAL_M4_REPLAY_RECONSTRUCTION_METHOD,
    }
    feature.feature_json = json.dumps(payload, sort_keys=True)
    db_session.flush()


def _add_replay_signal(db_session, **kwargs):
    signal = _add_signal(db_session, **kwargs)
    _stamp_signal_historical_m4_replay(db_session, signal)
    return signal


def _seed_hur(db_session, ticker: str, days: list[date]) -> None:
    for day in days:
        db_session.add(
            HistoricalUniverseReconstruction(
                historical_universe_reconstruction_id=f"hur-{ticker}-{day}",
                replay_date=day,
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
                reconstruction_method="fixture",
                pit_filter_status_json="{}",
                input_hash=stable_hash({"ticker": ticker, "date": day.isoformat()}),
                output_hash=stable_hash({"ticker": ticker, "included": True, "date": day.isoformat()}),
            )
        )
    db_session.flush()


def _pre_signal_dates(signal_day: date, window: int) -> list[date]:
    out: list[date] = []
    cursor = signal_day
    for _ in range(window):
        cursor = previous_us_equity_session(cursor)
        out.append(cursor)
    return out


def _bar_dates_through(bars: list[FmpBar], through: date) -> list[date]:
    return [
        date.fromisoformat(bar.date)
        for bar in bars
        if date.fromisoformat(bar.date) <= through
    ]


def _session_bars(
    *,
    ticker: str = "SETU",
    through: date = date(2026, 6, 5),
    sessions: int = 90,
    future_spike: bool = False,
) -> list[FmpBar]:
    days: list[date] = []
    cursor = through
    while len(days) < sessions:
        if is_us_equity_session(cursor):
            days.append(cursor)
        cursor -= timedelta(days=1)
    bars: list[FmpBar] = []
    for index, day in enumerate(reversed(days)):
        close = 10.0 + index * 0.05
        if future_spike and day == through:
            close = 999.0
        bars.append(
            FmpBar(
                date=day.isoformat(),
                open=close - 0.05,
                high=close + 0.25,
                low=close - 0.25,
                close=close,
                volume=100_000 + index * 1_000,
                split_adjusted_close=close,
                adj_close=close,
                vwap=close,
            )
        )
    return bars


def test_pre_signal_context_default_source_processes_only_historical_replay(db_session):
    signal_day = date(2026, 6, 5)
    replay_signal = _add_replay_signal(db_session, ticker="RPLY", signal_day=signal_day)
    live_signal = _add_signal(db_session, ticker="LIVE", signal_day=signal_day)
    replay_dates = _pre_signal_dates(signal_day, 1)
    _seed_hur(db_session, "RPLY", replay_dates)
    _seed_hur(db_session, "LIVE", replay_dates)
    adapter = FakeFmpAdapter({
        "RPLY": _session_bars(ticker="RPLY", through=signal_day),
        "LIVE": _session_bars(ticker="LIVE", through=signal_day),
    })

    result = run_job(
        db_session,
        MarketPathPreSignalContextJob(
            session=db_session,
            fmp_adapter=adapter,
            signal_start_date=signal_day,
            signal_end_date=signal_day,
            run_timestamp=RUN_TS,
            pre_signal_window=1,
        ),
    )

    assert result.status == "finished"
    assert result.metrics["signals_scanned"] == 1
    assert {call["ticker"] for call in adapter.calls} == {"RPLY"}
    assert db_session.get(
        MarketPathPreSignalContext,
        ("RPLY", replay_dates[0], "pre_signal_context", FEATURE_VERSION),
    ) is not None
    assert db_session.get(
        MarketPathPreSignalContext,
        ("LIVE", replay_dates[0], "pre_signal_context", FEATURE_VERSION),
    ) is None
    assert (
        db_session.query(MarketPathPreSignalLink)
        .filter(MarketPathPreSignalLink.signal_id == replay_signal.signal_id)
        .count()
    ) == 1
    assert (
        db_session.query(MarketPathPreSignalLink)
        .filter(MarketPathPreSignalLink.signal_id == live_signal.signal_id)
        .count()
    ) == 0


def test_pre_signal_context_uses_only_bars_through_feature_date(db_session):
    signal_day = date(2026, 6, 5)
    signal = _add_replay_signal(db_session, signal_day=signal_day)
    feature_dates = _pre_signal_dates(signal_day, 2)
    _seed_hur(db_session, "SETU", feature_dates)
    adapter = FakeFmpAdapter({"SETU": _session_bars(through=signal_day, future_spike=True)})

    result = run_job(
        db_session,
        MarketPathPreSignalContextJob(
            session=db_session,
            fmp_adapter=adapter,
            signal_start_date=signal_day,
            signal_end_date=signal_day,
            run_timestamp=RUN_TS,
            pre_signal_window=2,
            batch_days=1,
        ),
    )

    assert result.status == "finished"
    assert adapter.calls[0]["to_date"] == date(2026, 6, 4)
    row = db_session.get(
        MarketPathPreSignalContext,
        ("SETU", date(2026, 6, 4), "pre_signal_context", FEATURE_VERSION),
    )
    assert row is not None
    assert row.row_status == ROW_STATUS_COMPUTED
    assert row.close_price != 999.0
    payload = json.loads(row.feature_json)
    assert payload["max_input_date"] == "2026-06-04"
    assert payload["strict_no_lookahead"]["uses_only_bars_lte_feature_session_date"] is True
    forbidden = json.dumps(payload)
    assert "return_from_entry" not in forbidden
    assert "forward_return" not in forbidden
    assert "ret_next" not in forbidden
    assert (
        db_session.query(MarketPathPreSignalLink)
        .filter(
            MarketPathPreSignalLink.signal_id == signal.signal_id,
            MarketPathPreSignalLink.relative_session_index == -1,
        )
        .one()
        .feature_session_date
        == date(2026, 6, 4)
    )


def test_pre_signal_context_lineage_payload_uses_pre_signal_reconstruction_method(db_session):
    signal_day = date(2026, 6, 5)
    _add_replay_signal(db_session, ticker="LINE", signal_day=signal_day)
    feature_dates = _pre_signal_dates(signal_day, 1)
    _seed_hur(db_session, "LINE", feature_dates)

    result = run_job(
        db_session,
        MarketPathPreSignalContextJob(
            session=db_session,
            fmp_adapter=FakeFmpAdapter({"LINE": _session_bars(ticker="LINE", through=signal_day)}),
            signal_start_date=signal_day,
            signal_end_date=signal_day,
            run_timestamp=RUN_TS,
            pre_signal_window=1,
        ),
    )

    assert result.status == "finished"
    row = db_session.get(
        MarketPathPreSignalContext,
        ("LINE", feature_dates[0], "pre_signal_context", FEATURE_VERSION),
    )
    lineage = db_session.get(DataLineage, row.data_lineage_id)
    payload = json.loads(lineage.raw_payload_json)
    assert payload["reconstruction_method"] == "m4_pre_signal_context_fmp_eod_v1"


def test_pre_signal_context_writes_outside_hur_status_without_fetching_wrong_era(db_session):
    signal_day = date(2026, 6, 5)
    _add_replay_signal(db_session, ticker="REUS", signal_day=signal_day)
    feature_dates = _pre_signal_dates(signal_day, 2)
    _seed_hur(db_session, "REUS", [feature_dates[0]])
    adapter = FakeFmpAdapter({"REUS": _session_bars(ticker="REUS", through=signal_day)})

    result = run_job(
        db_session,
        MarketPathPreSignalContextJob(
            session=db_session,
            fmp_adapter=adapter,
            signal_start_date=signal_day,
            signal_end_date=signal_day,
            run_timestamp=RUN_TS,
            pre_signal_window=2,
            batch_days=1,
        ),
    )

    assert result.status == "finished"
    outside = db_session.get(
        MarketPathPreSignalContext,
        ("REUS", feature_dates[1], "pre_signal_context", FEATURE_VERSION),
    )
    assert outside is not None
    assert outside.row_status == ROW_STATUS_OUTSIDE_UNIVERSE
    assert outside.close_price is None
    assert json.loads(outside.feature_json)["row_input_window_end"] is None
    assert result.metrics["outside_universe_coverage_rows"] == 1


def test_pre_signal_context_stamps_rank_and_split_caveats(db_session):
    signal_day = date(2026, 6, 5)
    _add_replay_signal(db_session, ticker="PENY", signal_day=signal_day)
    feature_dates = _pre_signal_dates(signal_day, 1)
    _seed_hur(db_session, "PENY", feature_dates)
    bars = _session_bars(ticker="PENY", through=signal_day)
    bars = [
        FmpBar(
            date=bar.date,
            open=0.9 if date.fromisoformat(bar.date) == feature_dates[0] else bar.open,
            high=1.0 if date.fromisoformat(bar.date) == feature_dates[0] else bar.high,
            low=0.8 if date.fromisoformat(bar.date) == feature_dates[0] else bar.low,
            close=0.95 if date.fromisoformat(bar.date) == feature_dates[0] else bar.close,
            volume=bar.volume,
            split_adjusted_close=0.95 if date.fromisoformat(bar.date) == feature_dates[0] else bar.split_adjusted_close,
            adj_close=0.95 if date.fromisoformat(bar.date) == feature_dates[0] else bar.adj_close,
            vwap=bar.vwap,
        )
        for bar in bars
    ]

    run_job(
        db_session,
        MarketPathPreSignalContextJob(
            session=db_session,
            fmp_adapter=FakeFmpAdapter({"PENY": bars}),
            signal_start_date=signal_day,
            signal_end_date=signal_day,
            run_timestamp=RUN_TS,
            pre_signal_window=1,
        ),
    )

    row = db_session.get(
        MarketPathPreSignalContext,
        ("PENY", feature_dates[0], "pre_signal_context", FEATURE_VERSION),
    )
    assert row.rank_status == "not_applicable_predictor_row"
    assert row.retroactive_adjustment_caveat is True
    assert row.sub_dollar is True
    payload = json.loads(row.feature_json)
    assert payload["rank_status"]["reason"] == "fired_cohort_pre_signal_ranks_are_circular"
    assert "dollar_volume" in payload["split_adjustment_caveats"]["affected_price_level_fields"]


def test_pre_signal_context_happy_path_populates_within_window_features(db_session):
    signal_day = date(2026, 6, 5)
    _add_replay_signal(db_session, ticker="FULL", signal_day=signal_day)
    feature_dates = _pre_signal_dates(signal_day, 1)
    bars = _session_bars(ticker="FULL", through=signal_day, sessions=90)
    _seed_hur(db_session, "FULL", _bar_dates_through(bars, feature_dates[0]))

    result = run_job(
        db_session,
        MarketPathPreSignalContextJob(
            session=db_session,
            fmp_adapter=FakeFmpAdapter({"FULL": bars}),
            signal_start_date=signal_day,
            signal_end_date=signal_day,
            run_timestamp=RUN_TS,
            pre_signal_window=1,
        ),
    )

    assert result.status == "finished"
    row = db_session.get(
        MarketPathPreSignalContext,
        ("FULL", feature_dates[0], "pre_signal_context", FEATURE_VERSION),
    )
    assert row.row_status == ROW_STATUS_COMPUTED
    assert row.previous_close is not None
    assert row.median_volume_20d is not None
    assert row.median_dollar_volume_20d is not None
    assert row.volume_expansion_20d is not None
    assert row.return_1d is not None
    assert row.return_5d is not None
    assert row.return_20d is not None
    assert row.sigma_20d is not None
    status = json.loads(row.status_json)
    assert "window_identity_boundary" not in status
    assert status["insufficient_history"]["prior20"] is False


def test_pre_signal_context_nulls_within_window_features_across_hur_identity_boundary(db_session):
    signal_day = date(2026, 6, 5)
    _add_replay_signal(db_session, ticker="BNDY", signal_day=signal_day)
    feature_dates = _pre_signal_dates(signal_day, 1)
    bars = _session_bars(ticker="BNDY", through=signal_day, sessions=90)
    bar_dates = _bar_dates_through(bars, feature_dates[0])
    excluded_day = bar_dates[-2]
    _seed_hur(db_session, "BNDY", [day for day in bar_dates if day != excluded_day])

    result = run_job(
        db_session,
        MarketPathPreSignalContextJob(
            session=db_session,
            fmp_adapter=FakeFmpAdapter({"BNDY": bars}),
            signal_start_date=signal_day,
            signal_end_date=signal_day,
            run_timestamp=RUN_TS,
            pre_signal_window=1,
        ),
    )

    assert result.status == "finished"
    row = db_session.get(
        MarketPathPreSignalContext,
        ("BNDY", feature_dates[0], "pre_signal_context", FEATURE_VERSION),
    )
    assert row.row_status == ROW_STATUS_COMPUTED
    assert row.open_price is not None
    assert row.close_price is not None
    assert row.previous_close is None
    assert row.median_volume_20d is None
    assert row.median_dollar_volume_20d is None
    assert row.volume_expansion_20d is None
    assert row.return_1d is None
    assert row.return_5d is None
    assert row.return_20d is None
    assert row.sigma_20d is None
    status = json.loads(row.status_json)
    assert status["insufficient_history"]["prior20"] is False
    boundary = status["window_identity_boundary"]
    assert boundary["first_excluded_date"] == excluded_day.isoformat()
    assert boundary["excluded_count"] == 1
    assert boundary["excluded_dates"] == [excluded_day.isoformat()]
    assert set(boundary["fields"]) == {
        "previous_close",
        "return_1d",
        "return_5d",
        "return_20d",
        "sigma_20d",
        "median_volume_20d",
        "median_dollar_volume_20d",
        "volume_expansion_20d",
    }


def test_pre_signal_context_rerun_is_content_idempotent(db_session):
    signal_day = date(2026, 6, 5)
    _add_replay_signal(db_session, ticker="IDEM", signal_day=signal_day)
    feature_dates = _pre_signal_dates(signal_day, 2)
    _seed_hur(db_session, "IDEM", feature_dates)
    adapter = FakeFmpAdapter({"IDEM": _session_bars(ticker="IDEM", through=signal_day)})
    kwargs = dict(
        session=db_session,
        fmp_adapter=adapter,
        signal_start_date=signal_day,
        signal_end_date=signal_day,
        run_timestamp=RUN_TS,
        pre_signal_window=2,
    )

    first = run_job(db_session, MarketPathPreSignalContextJob(**kwargs))
    hashes = {
        (row.ticker, row.feature_session_date): row.output_hash
        for row in db_session.query(MarketPathPreSignalContext).all()
    }
    second = run_job(db_session, MarketPathPreSignalContextJob(**kwargs))

    assert first.metrics["context_rows_inserted"] == 2
    assert second.metrics["context_rows_material_updates"] == 0
    assert second.metrics["context_rows_unchanged"] == 2
    assert db_session.query(MarketPathPreSignalContext).count() == 2
    assert db_session.query(MarketPathPreSignalLink).count() == 2
    assert {
        (row.ticker, row.feature_session_date): row.output_hash
        for row in db_session.query(MarketPathPreSignalContext).all()
    } == hashes


def test_pre_signal_context_rows_do_not_enter_forward_rank_pass(db_session):
    signal_day = date(2026, 6, 5)
    signal_a = _add_replay_signal(db_session, ticker="FRDA", signal_day=date(2026, 6, 3))
    signal_b = _add_replay_signal(db_session, ticker="FRDB", signal_day=date(2026, 6, 3))
    _add_replay_signal(db_session, ticker="SETU", signal_day=signal_day)
    feature_date = date(2026, 6, 4)
    _seed_hur(db_session, "SETU", [feature_date])
    lineage = record_data_lineage(
        db_session,
        provider="fixture",
        endpoint="/fixture/forward",
        asof_timestamp=RUN_TS,
        raw_payload={"rows": 2},
    )
    for signal, ticker, dollar_volume in (
        (signal_a, "FRDA", 200_000.0),
        (signal_b, "FRDB", 100_000.0),
    ):
        db_session.add(
            MarketPathFeature(
                market_path_feature_id=f"mp-{ticker}",
                signal_id=signal.signal_id,
                pattern_id="M4",
                ticker=ticker,
                signal_horizon="15d",
                signal_date="2026-06-03",
                entry_session_date="2026-06-04",
                feature_session_date="2026-06-04",
                path_sequence=1,
                feature_role="forward_path_day",
                feature_version=FORWARD_FEATURE_VERSION,
                asof_timestamp=RUN_TS,
                reconstruction_method="fixture",
                dollar_volume=dollar_volume,
                feature_json="{}",
                source_provider="fixture",
                source_endpoint="/fixture",
                data_lineage_id=lineage.data_lineage_id,
                input_hash=stable_hash({"ticker": ticker, "input": True}),
                output_hash=stable_hash({"ticker": ticker, "ranked": False}),
            )
        )
    rank_job = MarketPathFeatureJob(
        session=db_session,
        fmp_adapter=FakeFmpAdapter({}),
        run_timestamp=RUN_TS,
        pattern_ids=("M4",),
        signal_start_date=feature_date,
        signal_end_date=feature_date,
        through_date=feature_date,
    )
    assert rank_job._populate_cross_sectional_ranks(
        start_date=feature_date,
        through_date=feature_date,
    ) == 2
    before = {
        row.market_path_feature_id: (row.output_hash, row.dollar_volume_rank)
        for row in db_session.query(MarketPathFeature).all()
    }

    run_job(
        db_session,
        MarketPathPreSignalContextJob(
            session=db_session,
            fmp_adapter=FakeFmpAdapter({"SETU": _session_bars(through=signal_day)}),
            signal_start_date=signal_day,
            signal_end_date=signal_day,
            run_timestamp=RUN_TS,
            pre_signal_window=1,
        ),
    )
    assert db_session.query(MarketPathPreSignalContext).count() == 1
    assert rank_job._populate_cross_sectional_ranks(
        start_date=feature_date,
        through_date=feature_date,
    ) == 0
    assert {
        row.market_path_feature_id: (row.output_hash, row.dollar_volume_rank)
        for row in db_session.query(MarketPathFeature).all()
    } == before


def test_pre_signal_runner_hard_refuses_public_default_writes():
    with pytest.raises(ValueError, match="Refusing public/default"):
        _validate_write_target(schema=None, confirm_live_write=True)
    with pytest.raises(ValueError, match="Refusing public/default"):
        _validate_write_target(schema="public", confirm_live_write=True)
    with pytest.raises(ValueError, match="Refusing public/default"):
        _validate_write_target(schema="", confirm_live_write=True)
    with pytest.raises(ValueError, match="Refusing public/default"):
        _validate_write_target(schema="   ", confirm_live_write=True)
    _validate_write_target(schema="m4_pre_signal_scratch", confirm_live_write=False)


def test_pre_signal_runner_requires_explicit_signal_source():
    with pytest.raises(SystemExit):
        _parse_args([
            "--live",
            "--schema",
            "m4_pre_signal_scratch",
            "--signal-start-date",
            "2026-06-01",
            "--signal-end-date",
            "2026-06-05",
        ])


def test_pre_signal_runner_preserves_job_progress_artifact(db_session, tmp_path, monkeypatch):
    signal_day = date(2026, 6, 5)
    _add_replay_signal(db_session, ticker="ARTI", signal_day=signal_day)
    feature_dates = _pre_signal_dates(signal_day, 1)
    _seed_hur(db_session, "ARTI", feature_dates)
    fake_adapter = FakeFmpAdapter({"ARTI": _session_bars(ticker="ARTI", through=signal_day)})
    progress_artifact = tmp_path / "pre_signal_progress.json"

    monkeypatch.setattr(pre_signal_runner, "load_runtime_env", lambda: None)
    monkeypatch.setattr(pre_signal_runner, "reset_globals", lambda: None)
    monkeypatch.setattr(
        pre_signal_runner,
        "prepare_writable_schema_target",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(pre_signal_runner, "get_session", lambda: db_session)
    monkeypatch.setattr(pre_signal_runner.FmpConfig, "from_env", staticmethod(lambda: object()))
    monkeypatch.setattr(pre_signal_runner, "FmpAdapter", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        pre_signal_runner,
        "CachedHistoricalPriceFmpAdapter",
        lambda adapter: fake_adapter,
    )
    monkeypatch.setattr(
        pre_signal_runner,
        "RetryingHistoricalPriceFmpAdapter",
        lambda adapter, **kwargs: adapter,
    )

    rc = pre_signal_runner.main([
        "--live",
        "--schema",
        "m4_pre_signal_scratch",
        "--signal-start-date",
        signal_day.isoformat(),
        "--signal-end-date",
        signal_day.isoformat(),
        "--signal-source",
        "historical-m4-replay",
        "--pre-signal-window",
        "1",
        "--progress-artifact",
        str(progress_artifact),
    ])

    assert rc == 0
    job_payload = json.loads(progress_artifact.read_text())
    assert "batches" in job_payload
    assert job_payload["batches"][0]["status"] == "finished"
    assert job_payload["summary"]["context_rows_inserted"] == 1
    assert "events" not in job_payload
    runner_artifact = pre_signal_runner._runner_artifact_path(progress_artifact)
    runner_payload = json.loads(runner_artifact.read_text())
    assert runner_payload["job_progress_artifact"] == str(progress_artifact)
    assert runner_payload["events"]
    assert runner_payload["result"]["status"] == "finished"
