from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from sqlalchemy import event, func

from alpha.data.contracts import AdapterResponse, LineageMeta, stable_hash
from alpha.data.fmp import FmpBar, HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.data.polygon import PolygonBar
from alpha.db.models import (
    CanonicalUniverseScan,
    FeatureSnapshot,
    HistoricalUniverseReconstruction,
    SecurityProfile,
    SignalRegistry,
    UniverseScan,
    UniverseSnapshot,
)
from alpha.jobs.run_historical_m4_range_replay import (
    _BulkDetectionRecord,
    _existing_feature_snapshots,
    main as range_cli_main,
    run_historical_m4_range_replay,
)
from alpha.market_calendar import previous_us_equity_session, us_equity_session_close_timestamp


START_DAY = date(2024, 6, 3)
END_DAY = date(2024, 6, 4)


class _FakeFmpAdapter:
    def __init__(self, bars_by_ticker: dict[str, list[FmpBar]]):
        self.bars_by_ticker = {
            ticker.upper(): list(bars)
            for ticker, bars in bars_by_ticker.items()
        }
        self.calls: list[dict] = []
        self.cache_hits = 0
        self.cache_misses = 0

    def get_historical_price(
        self,
        ticker,
        from_date=None,
        to_date=None,
        asof=None,
        *,
        adjusted=False,
        require_split_adjusted_close=True,
        **_,
    ):
        self.cache_misses += 1
        self.calls.append(
            {
                "ticker": ticker,
                "from_date": from_date,
                "to_date": to_date,
                "asof": asof,
                "adjusted": adjusted,
                "require_split_adjusted_close": require_split_adjusted_close,
            }
        )
        bars = [
            bar
            for bar in self.bars_by_ticker.get(ticker.upper(), [])
            if (from_date is None or date.fromisoformat(bar.date) >= from_date)
            and (to_date is None or date.fromisoformat(bar.date) <= to_date)
        ]
        lineage = LineageMeta(
            provider="FMP",
            endpoint=HISTORICAL_PRICE_FULL_ENDPOINT,
            request_timestamp=_ts(),
            asof_timestamp=asof or _ts(),
            raw_payload_hash=stable_hash(
                {
                    "ticker": ticker.upper(),
                    "bars": [bar.__dict__ for bar in bars],
                }
            ),
            source_authority="FMP",
            data_quality_flags={"test": True},
        )
        return AdapterResponse(data=bars, lineage=lineage)


class _FakePolygonAdapter:
    def __init__(self, bars_by_ticker: dict[str, list[PolygonBar]]):
        self.bars_by_ticker = {
            ticker.upper(): list(bars)
            for ticker, bars in bars_by_ticker.items()
        }
        self.calls: list[dict] = []

    def get_daily_bars(
        self,
        ticker,
        from_date,
        to_date,
        limit=5000,
        *,
        adjusted=None,
    ):
        self.calls.append(
            {
                "ticker": ticker,
                "from_date": from_date,
                "to_date": to_date,
                "limit": limit,
                "adjusted": adjusted,
            }
        )
        from_day = date.fromisoformat(str(from_date))
        to_day = date.fromisoformat(str(to_date))
        bars = []
        if adjusted is True:
            bars = [
                bar
                for bar in self.bars_by_ticker.get(ticker.upper(), [])
                if from_day <= _polygon_bar_date(bar) <= to_day
            ]
        endpoint = f"/v2/aggs/ticker/{ticker.upper()}/range/1/day/{from_date}/{to_date}"
        lineage = LineageMeta(
            provider="Polygon",
            endpoint=endpoint,
            request_timestamp=_ts(),
            asof_timestamp=_ts(),
            raw_payload_hash=stable_hash(
                {
                    "ticker": ticker.upper(),
                    "bars": [bar.__dict__ for bar in bars],
                }
            ),
            source_authority="Polygon",
            data_quality_flags={
                "test": True,
                "adjusted": adjusted is True,
                "requested_adjusted": adjusted,
                "price_basis": "polygon_daily_close_split_adjusted"
                if adjusted is True
                else None,
                "adjustment_basis": "split_adjusted"
                if adjusted is True
                else "unknown",
            },
        )
        return AdapterResponse(data=bars, lineage=lineage)


def _ts() -> datetime:
    return datetime(2024, 6, 4, 21, 0, tzinfo=timezone.utc)


def _seed_active_universe(
    db_session,
    tickers: list[str],
    *,
    scan_trading_date: str = "2026-06-06",
) -> None:
    scan_id = "range-active-current"
    db_session.add(
        UniverseScan(
            scan_id=scan_id,
            trading_date=scan_trading_date,
            asof_timestamp=_ts(),
            provider="FMP",
            raw_count=len(tickers),
            deduped_count=len(tickers),
            included_count=len(tickers),
            excluded_count=0,
        )
    )
    db_session.flush()
    db_session.add(
        CanonicalUniverseScan(
            trading_date=scan_trading_date,
            scan_id=scan_id,
            selected_at=_ts(),
            selection_reason="test_current_active_source",
        )
    )
    for ticker in tickers:
        db_session.add(
            UniverseSnapshot(
                universe_snapshot_id=f"range-active-{ticker}",
                scan_id=scan_id,
                ticker=ticker,
                asof_timestamp=_ts(),
                source_provider="FMP",
                market_cap=100_000_000,
                price=10.0,
                security_type="common_stock",
                primary_exchange="NASDAQ",
                operating_universe_inclusion=True,
                source_lineage_hash=f"lineage-{ticker}",
            )
        )
        db_session.merge(
            SecurityProfile(
                symbol=ticker.upper(),
                security_type="common_stock",
                source_provider="FMP",
                profile_payload_hash=f"profile-{ticker}",
                raw_profile_json=json.dumps(
                    {
                        "symbol": ticker.upper(),
                        "companyName": f"{ticker} Inc.",
                        "exchange": "NASDAQ",
                        "ipoDate": "2020-01-01",
                    },
                    sort_keys=True,
                ),
            )
        )
    db_session.flush()


def _bars_for_range(
    *,
    evidence_closes: dict[date, float],
    prior_close: float = 10.0,
) -> list[FmpBar]:
    prior_days: list[date] = []
    cursor = START_DAY
    for _ in range(252):
        cursor = previous_us_equity_session(cursor)
        prior_days.append(cursor)
    prior_days.reverse()
    bars = [
        _bar(day, prior_close)
        for day in prior_days
    ]
    cursor = START_DAY
    while cursor <= END_DAY:
        if cursor in evidence_closes:
            bars.append(_bar(cursor, evidence_closes[cursor]))
        cursor += timedelta(days=1)
    return bars


def _polygon_bars_for_range(
    *,
    evidence_closes: dict[date, float],
    prior_close: float = 10.0,
) -> list[PolygonBar]:
    prior_days: list[date] = []
    cursor = START_DAY
    for _ in range(252):
        cursor = previous_us_equity_session(cursor)
        prior_days.append(cursor)
    prior_days.reverse()
    bars = [_polygon_bar(day, prior_close) for day in prior_days]
    cursor = START_DAY
    while cursor <= END_DAY:
        if cursor in evidence_closes:
            bars.append(_polygon_bar(cursor, evidence_closes[cursor]))
        cursor += timedelta(days=1)
    return bars


def _bar(day: date, close: float) -> FmpBar:
    return FmpBar(
        date=day.isoformat(),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=100_000,
        split_adjusted_close=close,
        adj_close=999.0,
    )


def _polygon_bar(day: date, close: float) -> PolygonBar:
    timestamp = int(
        datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
        * 1000
    )
    return PolygonBar(
        timestamp=timestamp,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=100_000,
    )


def _polygon_bar_date(bar: PolygonBar) -> date:
    return datetime.fromtimestamp(
        bar.timestamp / 1000,
        tz=timezone.utc,
    ).date()


def _feature_for(db_session, ticker: str, replay_day: date) -> dict:
    row = (
        db_session.query(FeatureSnapshot)
        .filter(
            FeatureSnapshot.pattern_id == "M4",
            FeatureSnapshot.ticker == ticker,
            FeatureSnapshot.asof_timestamp == us_equity_session_close_timestamp(replay_day),
        )
        .one()
    )
    return json.loads(row.feature_json)


def _seed_unrelated_feature_snapshots(
    db_session,
    replay_day: date,
    *,
    count: int,
) -> None:
    asof = us_equity_session_close_timestamp(replay_day)
    mappings = [
        {
            "feature_snapshot_id": f"noise-feature-{replay_day.isoformat()}-{index}",
            "pattern_id": "M4",
            "ticker": f"ZZ{index:04d}",
            "asof_timestamp": asof,
            "feature_manifest_version": "noise",
            "feature_json": json.dumps(
                {
                    "reconstruction_method": "historical_m4_replay_fmp_eod",
                    "replay_date": replay_day.isoformat(),
                    "lookback_end": previous_us_equity_session(
                        replay_day,
                    ).isoformat(),
                },
                sort_keys=True,
            ),
            "feature_hash": f"noise-hash-{replay_day.isoformat()}-{index}",
            "data_lineage_ids": json.dumps([f"noise-lineage-{index}"]),
            "input_hashes": json.dumps({"noise": index}),
            "output_hash": f"noise-output-{replay_day.isoformat()}-{index}",
        }
        for index in range(count)
    ]
    db_session.execute(FeatureSnapshot.__table__.insert(), mappings)
    db_session.flush()


def test_range_replay_fetches_each_ticker_once_and_slices_bars_per_date(db_session):
    _seed_active_universe(db_session, ["RNGF", "RNGN"])
    adapter = _FakeFmpAdapter(
        {
            "RNGF": _bars_for_range(
                evidence_closes={
                    START_DAY: 10.1,
                    END_DAY: 99.0,
                }
            ),
            "RNGN": _bars_for_range(
                evidence_closes={
                    START_DAY: 9.9,
                    END_DAY: 9.8,
                }
            ),
        }
    )
    events = []

    result = run_historical_m4_range_replay(
        session=db_session,
        fmp_adapter=adapter,
        start_date=START_DAY,
        end_date=END_DAY,
        run_timestamp=_ts(),
        progress_callback=lambda event, payload: events.append((event, payload)),
        progress_every=1,
    )

    assert result.status == "finished"
    assert result.metrics["unique_ticker_count"] == 2
    assert result.metrics["date_ticker_equivalent_fetch_count"] == 4
    assert len(adapter.calls) == 2
    assert {call["ticker"] for call in adapter.calls} == {"RNGF", "RNGN"}
    assert all(call["to_date"] == END_DAY for call in adapter.calls)
    first_features = _feature_for(db_session, "RNGF", START_DAY)
    second_features = _feature_for(db_session, "RNGF", END_DAY)
    assert first_features["H_52w"] == 10.0
    assert first_features["P_close"] == 10.1
    assert second_features["P_close"] == 99.0
    assert first_features["lookback_end"] < START_DAY.isoformat()
    assert first_features["reconstruction_method"] == "historical_m4_replay_fmp_eod"
    no_fire_features = _feature_for(db_session, "RNGN", START_DAY)
    assert no_fire_features["signal_generated"] is False
    assert no_fire_features["reconstruction_method"] == "historical_m4_replay_fmp_eod"
    assert no_fire_features["bar_lineage_id"]
    event_names = [event for event, _payload in events]
    assert "range_universe_candidate_load_finish" in event_names
    assert "range_universe_persistence_progress" in event_names
    assert "range_ticker_fetch_finish" in event_names
    assert "range_date_lineage_stage_finish" in event_names
    assert "range_date_feature_snapshot_stage_finish" in event_names
    assert "range_date_signal_stage_finish" in event_names
    assert "range_date_link_stage_finish" in event_names
    assert "range_date_detector_finish" in event_names


def test_range_replay_fmp_missing_uses_polygon_once_and_slices_per_date(db_session):
    _seed_active_universe(db_session, ["PFALL"])
    fmp = _FakeFmpAdapter({"PFALL": []})
    polygon = _FakePolygonAdapter(
        {
            "PFALL": _polygon_bars_for_range(
                evidence_closes={
                    START_DAY: 10.1,
                    END_DAY: 10.2,
                }
            )
        }
    )

    result = run_historical_m4_range_replay(
        session=db_session,
        fmp_adapter=fmp,
        polygon_adapter=polygon,
        start_date=START_DAY,
        end_date=END_DAY,
        run_timestamp=_ts(),
        progress_every=1,
    )

    assert result.status == "finished"
    assert len(fmp.calls) == 1
    assert len(polygon.calls) == 1
    assert polygon.calls[0]["to_date"] == END_DAY.isoformat()
    assert polygon.calls[0]["adjusted"] is True
    assert result.metrics["fmp_fetch"]["polygon_fallback_count"] == 1
    assert result.metrics["fmp_fetch"]["missing_price_evidence_count"] == 0
    first_features = _feature_for(db_session, "PFALL", START_DAY)
    second_features = _feature_for(db_session, "PFALL", END_DAY)
    assert first_features["P_close"] == 10.1
    assert second_features["P_close"] == 10.2
    for features in (first_features, second_features):
        assert features["bar_provider"] == "Polygon"
        assert features["fallback_used"] is True
        assert features["price_basis"] == "polygon_daily_close_split_adjusted"
        assert features["bar_requested_adjusted"] is True
        assert features["bar_adjusted"] is True
        replay = features["historical_replay"]
        assert replay["price_basis"] == "polygon_daily_close_split_adjusted"
        assert replay["requested_adjusted"] is True
        assert replay["adjusted"] is True
        assert replay["source_attempt_count"] == 2
        assert [attempt["status"] for attempt in replay["source_attempts"]] == [
            "no_usable_bars",
            "usable",
        ]


def test_range_replay_uses_polygon_only_for_dates_missing_fmp_evidence_bar(
    db_session,
):
    _seed_active_universe(db_session, ["MIXPX"])
    fmp = _FakeFmpAdapter(
        {
            "MIXPX": _bars_for_range(
                evidence_closes={
                    END_DAY: 10.3,
                }
            )
        }
    )
    polygon = _FakePolygonAdapter(
        {
            "MIXPX": _polygon_bars_for_range(
                evidence_closes={
                    START_DAY: 10.1,
                    END_DAY: 99.0,
                }
            )
        }
    )

    result = run_historical_m4_range_replay(
        session=db_session,
        fmp_adapter=fmp,
        polygon_adapter=polygon,
        start_date=START_DAY,
        end_date=END_DAY,
        run_timestamp=_ts(),
        progress_every=1,
    )

    assert result.status == "finished"
    assert len(fmp.calls) == 1
    assert len(polygon.calls) == 1
    assert result.metrics["fmp_fetch"]["polygon_fallback_count"] == 1
    assert result.metrics["fmp_fetch"]["missing_price_evidence_count"] == 0
    first_date, second_date = result.metrics["date_results"]
    assert first_date["missing_price_evidence_count"] == 0
    assert first_date["polygon_fallback_count"] == 1
    assert first_date["rejected_or_no_fire_count"] == 0
    assert second_date["missing_price_evidence_count"] == 0
    assert second_date["polygon_fallback_count"] == 0
    first_features = _feature_for(db_session, "MIXPX", START_DAY)
    second_features = _feature_for(db_session, "MIXPX", END_DAY)
    assert first_features["bar_provider"] == "Polygon"
    assert first_features["fallback_used"] is True
    assert first_features["P_close"] == 10.1
    assert second_features["bar_provider"] == "FMP"
    assert second_features["fallback_used"] is False
    assert second_features["P_close"] == 10.3


def test_range_replay_quarantines_missing_price_without_no_fire_label(db_session):
    _seed_active_universe(db_session, ["RGOOD", "RMISS"])
    fmp = _FakeFmpAdapter(
        {
            "RGOOD": _bars_for_range(
                evidence_closes={
                    START_DAY: 10.1,
                    END_DAY: 10.2,
                }
            ),
            "RMISS": [],
        }
    )
    polygon = _FakePolygonAdapter({"RMISS": []})

    result = run_historical_m4_range_replay(
        session=db_session,
        fmp_adapter=fmp,
        polygon_adapter=polygon,
        start_date=START_DAY,
        end_date=END_DAY,
        run_timestamp=_ts(),
        progress_every=1,
    )

    assert result.status == "finished"
    assert result.metrics["completion_classification"] == (
        "completed_with_non_evaluable_price_evidence"
    )
    assert result.metrics["coverage_status"] == "partial_price_evidence"
    assert result.metrics["total_historical_universe_included_count"] == 4
    assert result.metrics["total_m4_evaluable_count"] == 2
    assert result.metrics["total_m4_non_evaluable_count"] == 2
    assert result.metrics["total_missing_price_evidence_count"] == 2
    assert result.metrics["total_fired_m4_signal_count"] == 2
    assert result.metrics["total_rejected_or_no_fire_count"] == 0
    assert result.errors == []
    assert db_session.query(SignalRegistry).filter(SignalRegistry.ticker == "RGOOD").count() == 2
    assert db_session.query(FeatureSnapshot).filter(FeatureSnapshot.ticker == "RGOOD").count() == 2
    assert db_session.query(SignalRegistry).filter(SignalRegistry.ticker == "RMISS").count() == 0
    assert db_session.query(FeatureSnapshot).filter(FeatureSnapshot.ticker == "RMISS").count() == 0

    first_date, second_date = result.metrics["date_results"]
    for date_metrics in (first_date, second_date):
        assert date_metrics["completion_classification"] == (
            "completed_with_non_evaluable_price_evidence"
        )
        assert date_metrics["coverage_status"] == "partial_price_evidence"
        assert date_metrics["historical_universe_included_count"] == 2
        assert date_metrics["m4_evaluable_count"] == 1
        assert date_metrics["m4_non_evaluable_count"] == 1
        assert date_metrics["missing_price_evidence_count"] == 1
        assert date_metrics["rejected_or_no_fire_count"] == 0
        sample = date_metrics["non_evaluable_price_evidence_samples"][0]
        assert sample["ticker"] == "RMISS"
        assert sample["source"] == "current_active_universe"
        assert sample["exchange"] == "NASDAQ"
        assert sample["security_type"] == "common_stock"
        assert date_metrics["replay_date"] in sample["missing_evidence_dates"]
        assert [attempt["status"] for attempt in sample["provider_attempt_statuses"]] == [
            "no_usable_bars",
            "no_usable_bars",
        ]
    summary_sample = result.metrics["non_evaluable_price_evidence_samples"][0]
    assert summary_sample["ticker"] == "RMISS"
    assert summary_sample["missing_evidence_dates"] == [
        START_DAY.isoformat(),
        END_DAY.isoformat(),
    ]


def test_range_replay_missing_after_fmp_and_fallback_is_non_evaluable(db_session):
    _seed_active_universe(db_session, ["MISSPX"])
    fmp = _FakeFmpAdapter({"MISSPX": []})
    polygon = _FakePolygonAdapter({"MISSPX": []})

    result = run_historical_m4_range_replay(
        session=db_session,
        fmp_adapter=fmp,
        polygon_adapter=polygon,
        start_date=START_DAY,
        end_date=END_DAY,
        run_timestamp=_ts(),
        progress_every=1,
    )

    assert result.status == "failed"
    assert len(fmp.calls) == 1
    assert len(polygon.calls) == 1
    assert result.metrics["fmp_fetch"]["tickers_missing_bars"] == 1
    assert result.metrics["fmp_fetch"]["non_evaluable_ticker_count"] == 1
    assert result.metrics["fmp_fetch"]["missing_price_evidence_count"] == 1
    assert result.metrics["completion_classification"] == (
        "failed_no_evaluable_price_evidence"
    )
    assert result.metrics["coverage_status"] == "no_evaluable_price_evidence"
    assert result.metrics["total_m4_evaluable_count"] == 0
    assert result.metrics["total_m4_non_evaluable_count"] == 2
    assert result.metrics["total_missing_price_evidence_count"] == 2
    assert result.metrics["total_fired_m4_signal_count"] == 0
    assert db_session.query(SignalRegistry).count() == 0
    assert db_session.query(FeatureSnapshot).count() == 0
    first_date = result.metrics["date_results"][0]
    assert first_date["missing_price_evidence_count"] == 1
    assert first_date["m4_evaluable_count"] == 0
    assert first_date["rejected_or_no_fire_count"] == 0
    assert first_date["fetch_errors"][0]["error_type"] == "missing_price_evidence"


def test_range_replay_idempotent_rerun_reuses_signals_without_duplicates(db_session):
    _seed_active_universe(db_session, ["RIDM"])
    adapter = _FakeFmpAdapter(
        {
            "RIDM": _bars_for_range(
                evidence_closes={
                    START_DAY: 10.1,
                    END_DAY: 10.2,
                }
            )
        }
    )

    first = run_historical_m4_range_replay(
        session=db_session,
        fmp_adapter=adapter,
        start_date=START_DAY,
        end_date=END_DAY,
        run_timestamp=_ts(),
        progress_every=1,
    )
    _seed_unrelated_feature_snapshots(db_session, START_DAY, count=1200)
    existing_feature = (
        db_session.query(FeatureSnapshot)
        .filter(
            FeatureSnapshot.pattern_id == "M4",
            FeatureSnapshot.ticker == "RIDM",
            FeatureSnapshot.asof_timestamp == us_equity_session_close_timestamp(START_DAY),
        )
        .one()
    )
    lookup_record = _BulkDetectionRecord(
        result=SimpleNamespace(
            pattern_id="M4",
            asof_timestamp=existing_feature.asof_timestamp,
        ),
        ticker="RIDM",
        feature_payload={},
        feature_hash=existing_feature.feature_hash,
        feature_json="{}",
        output_hash="lookup-test",
        data_lineage_ids=[],
        universe_snapshot_id=None,
        next_execution_session=None,
    )
    captured_sql: list[str] = []

    def capture_sql(_conn, _cursor, statement, _parameters, _context, _executemany):
        captured_sql.append(statement)

    event.listen(db_session.bind, "before_cursor_execute", capture_sql)
    try:
        lookup = _existing_feature_snapshots(db_session, [lookup_record])
    finally:
        event.remove(db_session.bind, "before_cursor_execute", capture_sql)

    assert lookup.row_count >= 1201
    assert existing_feature.feature_snapshot_id in lookup.rows.values()
    lookup_sql = "\n".join(captured_sql).casefold()
    assert "feature_json" not in lookup_sql
    assert "data_lineage_ids" not in lookup_sql
    assert "input_hashes" not in lookup_sql
    assert "output_hash" not in lookup_sql
    assert "feature_snapshot_id" in lookup_sql
    assert "feature_hash" in lookup_sql

    second = run_historical_m4_range_replay(
        session=db_session,
        fmp_adapter=adapter,
        start_date=START_DAY,
        end_date=END_DAY,
        run_timestamp=_ts(),
        progress_every=1,
    )

    assert first.metrics["total_rows_inserted"] == 2
    assert second.metrics["total_rows_inserted"] == 0
    assert second.metrics["total_rows_reused"] == 2
    first_date_orchestration = second.metrics["date_results"][0]["orchestration"]
    second_date_orchestration = second.metrics["date_results"][1]["orchestration"]
    assert first_date_orchestration["features_inserted"] == 0
    assert first_date_orchestration["features_reused"] == 1
    assert first_date_orchestration["existing_feature_lookup_record_count"] == 1
    assert first_date_orchestration["existing_feature_lookup_row_count"] >= 1201
    assert first_date_orchestration["existing_feature_lookup_seconds"] >= 0
    assert first_date_orchestration["existing_signal_lookup_row_count"] == 1
    assert first_date_orchestration["existing_signal_lookup_seconds"] >= 0
    assert second_date_orchestration["features_inserted"] == 0
    assert second_date_orchestration["features_reused"] == 1
    assert second_date_orchestration["existing_feature_lookup_record_count"] == 1
    assert second_date_orchestration["existing_feature_lookup_row_count"] == 1
    event_names = [event["event"] for event in second.metrics["progress_events"]]
    assert "range_date_existing_feature_lookup_finish" in event_names
    assert "range_date_existing_signal_lookup_finish" in event_names
    assert second.metrics["validation"]["duplicate_m4_signal_identity_groups"] == 0
    assert second.metrics["validation"]["duplicate_m4_feature_snapshot_groups"] == 0
    assert second.metrics["validation"]["duplicate_historical_universe_groups"] == 0
    assert second.metrics["validation"]["missing_historical_replay_stamp_count"] == 0
    duplicate_groups = (
        db_session.query(
            SignalRegistry.pattern_id,
            SignalRegistry.trading_date,
            SignalRegistry.ticker,
            SignalRegistry.signal_identity_hash,
            func.count().label("row_count"),
        )
        .group_by(
            SignalRegistry.pattern_id,
            SignalRegistry.trading_date,
            SignalRegistry.ticker,
            SignalRegistry.signal_identity_hash,
        )
        .having(func.count() > 1)
        .count()
    )
    assert duplicate_groups == 0
    duplicate_feature_groups = (
        db_session.query(
            FeatureSnapshot.pattern_id,
            FeatureSnapshot.ticker,
            FeatureSnapshot.asof_timestamp,
            func.count().label("row_count"),
        )
        .group_by(
            FeatureSnapshot.pattern_id,
            FeatureSnapshot.ticker,
            FeatureSnapshot.asof_timestamp,
        )
        .having(func.count() > 1)
        .count()
    )
    assert duplicate_feature_groups == 0


def test_range_replay_persists_included_hur_rows_only(db_session):
    _seed_active_universe(db_session, ["KEEP", "FUTR"])
    profile = db_session.get(SecurityProfile, "FUTR")
    raw = json.loads(profile.raw_profile_json)
    raw["ipoDate"] = "2026-01-01"
    profile.raw_profile_json = json.dumps(raw, sort_keys=True)
    db_session.flush()
    adapter = _FakeFmpAdapter(
        {
            "KEEP": _bars_for_range(
                evidence_closes={
                    START_DAY: 10.1,
                    END_DAY: 10.2,
                }
            )
        }
    )

    result = run_historical_m4_range_replay(
        session=db_session,
        fmp_adapter=adapter,
        start_date=START_DAY,
        end_date=END_DAY,
        run_timestamp=_ts(),
        progress_every=1,
    )

    assert result.status == "finished"
    assert [m["included_count"] for m in result.metrics["universe"]["date_metrics"]] == [1, 1]
    assert [m["excluded_count"] for m in result.metrics["universe"]["date_metrics"]] == [1, 1]
    assert [m["suppressed_excluded_count"] for m in result.metrics["universe"]["date_metrics"]] == [1, 1]
    assert (
        db_session.query(HistoricalUniverseReconstruction)
        .filter(HistoricalUniverseReconstruction.normalized_symbol == "FUTR")
        .count()
    ) == 0
    assert (
        db_session.query(HistoricalUniverseReconstruction)
        .filter(HistoricalUniverseReconstruction.inclusion_status != "included")
        .count()
    ) == 0


def test_range_replay_rerun_ignores_replay_created_canonical_scans(db_session):
    _seed_active_universe(
        db_session,
        ["KEEP", "FUTR"],
        scan_trading_date=(START_DAY - timedelta(days=2)).isoformat(),
    )
    profile = db_session.get(SecurityProfile, "FUTR")
    raw = json.loads(profile.raw_profile_json)
    raw["ipoDate"] = "2026-01-01"
    profile.raw_profile_json = json.dumps(raw, sort_keys=True)
    db_session.flush()
    adapter = _FakeFmpAdapter(
        {
            "KEEP": _bars_for_range(
                evidence_closes={
                    START_DAY: 10.1,
                    END_DAY: 10.2,
                }
            )
        }
    )

    first = run_historical_m4_range_replay(
        session=db_session,
        fmp_adapter=adapter,
        start_date=START_DAY,
        end_date=END_DAY,
        run_timestamp=_ts(),
        progress_every=1,
    )
    second = run_historical_m4_range_replay(
        session=db_session,
        fmp_adapter=adapter,
        start_date=START_DAY,
        end_date=END_DAY,
        run_timestamp=_ts(),
        progress_every=1,
    )

    assert first.status == "finished"
    assert second.status == "finished"
    for result in (first, second):
        universe = result.metrics["universe"]
        assert universe["active_current_rows_seen"] == 2
        assert universe["candidate_count"] == 2
        assert [m["included_count"] for m in universe["date_metrics"]] == [1, 1]
        assert [m["excluded_count"] for m in universe["date_metrics"]] == [1, 1]
        assert [m["suppressed_excluded_count"] for m in universe["date_metrics"]] == [1, 1]

    assert second.metrics["universe"]["persistence"]["rows_inserted"] == 0
    assert second.metrics["validation"]["duplicate_historical_universe_groups"] == 0
    assert second.metrics["validation"]["duplicate_m4_feature_snapshot_groups"] == 0
    assert (
        db_session.query(HistoricalUniverseReconstruction)
        .filter(HistoricalUniverseReconstruction.normalized_symbol == "FUTR")
        .count()
    ) == 0


def test_range_cli_refuses_public_or_missing_schema(monkeypatch):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)

    assert range_cli_main(
        ["--live", "--start-date", START_DAY.isoformat(), "--end-date", END_DAY.isoformat()]
    ) == 1
    assert range_cli_main(
        [
            "--live",
            "--schema",
            "public",
            "--start-date",
            START_DAY.isoformat(),
            "--end-date",
            END_DAY.isoformat(),
        ]
    ) == 1


def test_range_cli_rejects_unsupported_m1(monkeypatch):
    monkeypatch.delenv("ALPHA_DB_SCHEMA", raising=False)

    assert range_cli_main(
        [
            "--live",
            "--schema",
            "scratch_range",
            "--pattern-id",
            "M1",
            "--start-date",
            START_DAY.isoformat(),
            "--end-date",
            END_DAY.isoformat(),
        ]
    ) == 1
