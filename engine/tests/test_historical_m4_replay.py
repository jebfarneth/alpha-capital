from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import func

from alpha.data.contracts import AdapterResponse, LineageMeta, stable_hash
from alpha.data.fmp import FmpBar, HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.data.polygon import PolygonBar
from alpha.db.models import (
    DataLineage,
    FeatureSnapshot,
    HistoricalUniverseReconstruction,
    SignalRegistry,
)
from alpha.jobs.historical_m4_replay import HistoricalM4ReplayJob
from alpha.jobs.runner import run_job
from alpha.jobs.run_historical_m4_replay import main as replay_cli_main
from alpha.market_calendar import previous_us_equity_session


REPLAY_DAY = date(2024, 6, 3)


class _FakeFmpAdapter:
    def __init__(self, bars_by_ticker: dict[str, list[FmpBar]]):
        self.bars_by_ticker = {
            ticker.upper(): list(bars)
            for ticker, bars in bars_by_ticker.items()
        }
        self.calls: list[dict] = []

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
        bars = self.bars_by_ticker.get(ticker.upper(), [])
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
    def __init__(
        self,
        bars_by_ticker: dict[str, list[PolygonBar]],
        *,
        unadjusted_bars_by_ticker: dict[str, list[PolygonBar]] | None = None,
    ):
        self.bars_by_ticker = {
            ticker.upper(): list(bars)
            for ticker, bars in bars_by_ticker.items()
        }
        self.unadjusted_bars_by_ticker = {
            ticker.upper(): list(bars)
            for ticker, bars in (unadjusted_bars_by_ticker or {}).items()
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
        bars_by_ticker = (
            self.bars_by_ticker
            if adjusted is True
            else self.unadjusted_bars_by_ticker
        )
        bars = [
            bar
            for bar in bars_by_ticker.get(ticker.upper(), [])
            if from_day <= _polygon_bar_date(bar) <= to_day
        ]
        endpoint = f"/v2/aggs/ticker/{ticker.upper()}/range/1/day/{from_date}/{to_date}"
        price_basis = (
            "polygon_daily_close_split_adjusted"
            if adjusted is True
            else "polygon_daily_close_unadjusted"
            if adjusted is False
            else None
        )
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
                "price_basis": price_basis,
                "adjustment_basis": "split_adjusted"
                if adjusted is True
                else "unadjusted"
                if adjusted is False
                else "unknown",
            },
        )
        return AdapterResponse(data=bars, lineage=lineage)


def _ts() -> datetime:
    return datetime(2024, 6, 3, 21, 0, tzinfo=timezone.utc)


def _bars(
    *,
    evidence_close: float,
    prior_close: float = 10.0,
    evidence_raw_high: float | None = None,
    prior_raw_high: float | None = None,
    prior_adj_close: float | None = None,
) -> list[FmpBar]:
    days: list[date] = []
    cursor = REPLAY_DAY
    for _ in range(252):
        cursor = previous_us_equity_session(cursor)
        days.append(cursor)
    days.reverse()
    bars = [
        FmpBar(
            date=day.isoformat(),
            open=prior_close,
            high=prior_raw_high if prior_raw_high is not None else prior_close,
            low=prior_close,
            close=prior_close,
            volume=100_000,
            split_adjusted_close=prior_close,
            adj_close=prior_adj_close,
        )
        for day in days
    ]
    bars.append(
        FmpBar(
            date=REPLAY_DAY.isoformat(),
            open=evidence_close,
            high=evidence_raw_high if evidence_raw_high is not None else evidence_close,
            low=evidence_close,
            close=evidence_close,
            volume=100_000,
            split_adjusted_close=evidence_close,
            adj_close=999.0,
        )
    )
    return bars


def _polygon_bars(
    *,
    evidence_close: float,
    prior_close: float = 10.0,
) -> list[PolygonBar]:
    days: list[date] = []
    cursor = REPLAY_DAY
    for _ in range(252):
        cursor = previous_us_equity_session(cursor)
        days.append(cursor)
    days.reverse()
    bars = [_polygon_bar(day, prior_close) for day in days]
    bars.append(_polygon_bar(REPLAY_DAY, evidence_close))
    return bars


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


def _seed_reconstruction(
    db_session,
    ticker: str,
    *,
    replay_day: date = REPLAY_DAY,
    partial: bool = False,
) -> None:
    provenance = {
        "delisted_source_complete": not partial,
        "delisted_source_partial_reason": "max_pages_reached" if partial else None,
        "source_intervals": [
            {
                "source": "test",
                "symbol": ticker,
                "inclusion_status": "included",
            }
        ],
    }
    db_session.add(
        HistoricalUniverseReconstruction(
            replay_date=replay_day,
            ticker=ticker,
            normalized_symbol=ticker.upper(),
            exchange="NASDAQ",
            company_name=f"{ticker} Inc.",
            ipo_date=date(2020, 1, 1),
            delisted_date=None,
            inclusion_status="included",
            rejection_reason=None,
            source="test",
            source_provenance_json=json.dumps(provenance, sort_keys=True),
            reconstructed=True,
            reconstruction_method="test_reconstruction",
            pit_filter_status_json=json.dumps({"test": True}, sort_keys=True),
            input_hash=stable_hash({"ticker": ticker, "input": True}),
            output_hash=stable_hash({"ticker": ticker, "output": True}),
        )
    )


def _run_replay(
    db_session,
    adapter,
    *,
    polygon_adapter=None,
    allow_partial_universe=False,
    progress_callback=None,
):
    job = HistoricalM4ReplayJob(
        session=db_session,
        fmp_adapter=adapter,
        polygon_adapter=polygon_adapter,
        replay_dates=[REPLAY_DAY],
        run_timestamp=_ts(),
        allow_partial_universe=allow_partial_universe,
        progress_callback=progress_callback,
        progress_every=1,
    )
    return run_job(
        db_session,
        job,
        params={
            "source": "test_historical_m4_replay",
            "allow_partial_universe": allow_partial_universe,
        },
    )


def _feature(db_session, ticker: str) -> dict:
    row = (
        db_session.query(FeatureSnapshot)
        .filter(FeatureSnapshot.pattern_id == "M4", FeatureSnapshot.ticker == ticker)
        .one()
    )
    return json.loads(row.feature_json)


def _assert_replay_stamped(features: dict) -> None:
    assert features["reconstructed"] is True
    assert features["reconstruction_method"] == "historical_m4_replay_fmp_eod"
    assert features["replay_date"] == REPLAY_DAY.isoformat()
    assert features["evidence_session_date"] == REPLAY_DAY.isoformat()
    assert features["source_universe_method"] == "active_current_plus_fmp_delisted_v1"
    assert features["bar_provider"] == "FMP"
    assert features["bar_provider_policy"] == "fmp_primary_polygon_fallback"
    assert features["fallback_used"] is False
    assert features["bar_lineage_id"]
    assert features["bar_lineage_hash"]
    assert features["price_basis"] == "fmp_full_close_as_split_adjusted_close"
    assert features["historical_replay"]["bar_provider"] == "FMP"
    assert features["historical_replay"]["fallback_used"] is False
    assert features["historical_replay"]["price_basis"] == (
        "fmp_full_close_as_split_adjusted_close"
    )


def test_replay_uses_split_adjusted_close_not_raw_high_or_adjclose(db_session):
    _seed_reconstruction(db_session, "BASIS")
    adapter = _FakeFmpAdapter(
        {
            "BASIS": _bars(
                evidence_close=10.5,
                prior_close=10.0,
                prior_raw_high=99.0,
                prior_adj_close=88.0,
            )
        }
    )

    result = _run_replay(db_session, adapter)

    assert result.status == "finished"
    assert result.metrics["total_fired_m4_signal_count"] == 1
    features = _feature(db_session, "BASIS")
    assert features["H_52w"] == 10.0
    assert features["high_52w_basis"] == "split_adjusted_close_prior_252_sessions"
    assert features["P_close"] == 10.5
    call = adapter.calls[0]
    assert call["adjusted"] is False
    assert call["require_split_adjusted_close"] is True


def test_replay_excludes_evidence_session_from_h52w_lookback(db_session):
    _seed_reconstruction(db_session, "LCUT")
    adapter = _FakeFmpAdapter({"LCUT": _bars(evidence_close=12.0, prior_close=10.0)})

    result = _run_replay(db_session, adapter)

    assert result.status == "finished"
    features = _feature(db_session, "LCUT")
    assert features["H_52w"] == 10.0
    assert features["evidence_split_adjusted_close"] == 12.0
    assert features["lookback_end"] < REPLAY_DAY.isoformat()


def test_below_high_row_does_not_fire(db_session):
    _seed_reconstruction(db_session, "BELOW")
    adapter = _FakeFmpAdapter({"BELOW": _bars(evidence_close=9.99, prior_close=10.0)})

    result = _run_replay(db_session, adapter)

    assert result.status == "finished"
    assert result.metrics["total_fired_m4_signal_count"] == 0
    assert db_session.query(SignalRegistry).count() == 0


def test_exact_high_close_fires(db_session):
    _seed_reconstruction(db_session, "EXACT")
    adapter = _FakeFmpAdapter({"EXACT": _bars(evidence_close=10.0, prior_close=10.0)})

    result = _run_replay(db_session, adapter)

    assert result.status == "finished"
    assert result.metrics["total_fired_m4_signal_count"] == 1
    signal = db_session.query(SignalRegistry).one()
    assert signal.pattern_id == "M4"
    assert signal.trading_date == REPLAY_DAY.isoformat()


def test_replay_provenance_is_stamped(db_session):
    _seed_reconstruction(db_session, "STAMP")
    adapter = _FakeFmpAdapter({"STAMP": _bars(evidence_close=10.1, prior_close=10.0)})

    _run_replay(db_session, adapter)

    features = _feature(db_session, "STAMP")
    _assert_replay_stamped(features)


def test_replay_fmp_missing_bars_uses_polygon_fallback(db_session):
    _seed_reconstruction(db_session, "PFALL")
    fmp = _FakeFmpAdapter({"PFALL": []})
    polygon = _FakePolygonAdapter(
        {"PFALL": _polygon_bars(evidence_close=10.1, prior_close=10.0)}
    )

    result = _run_replay(db_session, fmp, polygon_adapter=polygon)

    assert result.status == "finished"
    assert result.metrics["total_tickers_with_bars"] == 1
    assert result.metrics["total_tickers_missing_bars"] == 0
    assert result.metrics["total_polygon_fallback_count"] == 1
    assert result.metrics["total_missing_price_evidence_count"] == 0
    assert len(fmp.calls) == 1
    assert len(polygon.calls) == 1
    assert polygon.calls[0]["adjusted"] is True
    attempt_providers = {
        row.provider
        for row in db_session.query(DataLineage.provider)
        .filter(DataLineage.provider.in_(["FMP", "Polygon"]))
        .all()
    }
    assert attempt_providers == {"FMP", "Polygon"}
    features = _feature(db_session, "PFALL")
    assert features["P_close"] == 10.1
    assert features["bar_provider"] == "Polygon"
    assert features["fallback_used"] is True
    assert features["bar_provider_policy"] == "fmp_primary_polygon_fallback"
    assert features["price_basis"] == "polygon_daily_close_split_adjusted"
    assert features["bar_requested_adjusted"] is True
    assert features["bar_adjusted"] is True
    replay = features["historical_replay"]
    assert replay["bar_provider"] == "Polygon"
    assert replay["fallback_used"] is True
    assert replay["price_basis"] == "polygon_daily_close_split_adjusted"
    assert replay["requested_adjusted"] is True
    assert replay["adjusted"] is True
    assert replay["source_attempt_count"] == 2
    assert [attempt["status"] for attempt in replay["source_attempts"]] == [
        "no_usable_bars",
        "usable",
    ]
    assert replay["source_attempts"][1]["price_basis"] == (
        "polygon_daily_close_split_adjusted"
    )
    assert replay["source_attempts"][1]["requested_adjusted"] is True
    assert replay["source_attempts"][1]["adjusted"] is True


def test_replay_fmp_lookback_without_evidence_bar_uses_polygon_fallback(db_session):
    _seed_reconstruction(db_session, "FMPNOEVD")
    fmp = _FakeFmpAdapter(
        {"FMPNOEVD": _bars(evidence_close=99.0, prior_close=10.0)[:-1]}
    )
    polygon = _FakePolygonAdapter(
        {"FMPNOEVD": _polygon_bars(evidence_close=10.1, prior_close=10.0)}
    )

    result = _run_replay(db_session, fmp, polygon_adapter=polygon)

    assert result.status == "finished"
    assert result.metrics["total_tickers_with_bars"] == 1
    assert result.metrics["total_tickers_missing_bars"] == 0
    assert result.metrics["total_polygon_fallback_count"] == 1
    assert result.metrics["total_missing_price_evidence_count"] == 0
    assert result.metrics["total_fired_m4_signal_count"] == 1
    features = _feature(db_session, "FMPNOEVD")
    assert features["bar_provider"] == "Polygon"
    assert features["fallback_used"] is True
    assert features["P_close"] == 10.1
    replay = features["historical_replay"]
    assert [attempt["status"] for attempt in replay["source_attempts"]] == [
        "missing_evidence_session_bar",
        "usable",
    ]
    assert replay["source_attempts"][0]["evidence_session_bar_present"] is False
    assert replay["source_attempts"][1]["evidence_session_bar_present"] is True


def test_replay_prior_bars_without_evidence_from_all_providers_is_non_evaluable(
    db_session,
):
    _seed_reconstruction(db_session, "NOEVD")
    fmp = _FakeFmpAdapter({"NOEVD": _bars(evidence_close=99.0, prior_close=10.0)[:-1]})
    polygon = _FakePolygonAdapter(
        {"NOEVD": _polygon_bars(evidence_close=99.0, prior_close=10.0)[:-1]}
    )

    result = _run_replay(db_session, fmp, polygon_adapter=polygon)

    assert result.status == "failed"
    assert result.metrics["total_tickers_with_bars"] == 0
    assert result.metrics["total_tickers_missing_bars"] == 1
    assert result.metrics["total_non_evaluable_ticker_count"] == 1
    assert result.metrics["total_missing_price_evidence_count"] == 1
    assert result.metrics["total_polygon_fallback_count"] == 0
    assert result.metrics["total_rejected_or_no_fire_count"] == 0
    assert db_session.query(SignalRegistry).count() == 0
    assert db_session.query(FeatureSnapshot).count() == 0
    date_metrics = result.metrics["date_results"][0]
    assert date_metrics["missing_price_evidence_count"] == 1
    assert date_metrics["rejected_or_no_fire_count"] == 0
    assert [
        attempt["status"]
        for attempt in date_metrics["fetch_errors"][0]["source_attempts"]
    ] == [
        "missing_evidence_session_bar",
        "missing_evidence_session_bar",
    ]


def test_replay_polygon_fallback_uses_split_adjusted_basis_for_m4_decision(db_session):
    _seed_reconstruction(db_session, "PSPLT")
    fmp = _FakeFmpAdapter({"PSPLT": []})
    polygon = _FakePolygonAdapter(
        {"PSPLT": _polygon_bars(evidence_close=10.5, prior_close=10.0)},
        unadjusted_bars_by_ticker={
            "PSPLT": _polygon_bars(evidence_close=10.5, prior_close=20.0)
        },
    )

    result = _run_replay(db_session, fmp, polygon_adapter=polygon)

    assert result.status == "finished"
    assert result.metrics["total_polygon_fallback_count"] == 1
    assert result.metrics["total_fired_m4_signal_count"] == 1
    assert polygon.calls[0]["adjusted"] is True
    features = _feature(db_session, "PSPLT")
    assert features["bar_provider"] == "Polygon"
    assert features["fallback_used"] is True
    assert features["H_52w"] == 10.0
    assert features["P_close"] == 10.5
    assert features["signal_generated"] is True
    assert features["price_basis"] == "polygon_daily_close_split_adjusted"
    assert features["historical_replay"]["price_basis"] == (
        "polygon_daily_close_split_adjusted"
    )
    assert features["historical_replay"]["requested_adjusted"] is True
    assert features["historical_replay"]["adjusted"] is True
    assert "as_reported_no_split_adjusted" not in json.dumps(features)


def test_replay_fmp_success_does_not_call_polygon_fallback(db_session):
    _seed_reconstruction(db_session, "FMPONLY")
    fmp = _FakeFmpAdapter(
        {"FMPONLY": _bars(evidence_close=10.1, prior_close=10.0)}
    )
    polygon = _FakePolygonAdapter(
        {"FMPONLY": _polygon_bars(evidence_close=99.0, prior_close=99.0)}
    )

    result = _run_replay(db_session, fmp, polygon_adapter=polygon)

    assert result.status == "finished"
    assert result.metrics["total_polygon_fallback_count"] == 0
    assert polygon.calls == []
    features = _feature(db_session, "FMPONLY")
    assert features["bar_provider"] == "FMP"
    assert features["fallback_used"] is False
    assert features["P_close"] == 10.1


def test_replay_missing_after_fmp_and_fallback_is_non_evaluable(db_session):
    _seed_reconstruction(db_session, "MISSPX")
    fmp = _FakeFmpAdapter({"MISSPX": []})
    polygon = _FakePolygonAdapter({"MISSPX": []})

    result = _run_replay(db_session, fmp, polygon_adapter=polygon)

    assert result.status == "failed"
    assert result.metrics["total_tickers_with_bars"] == 0
    assert result.metrics["total_tickers_missing_bars"] == 1
    assert result.metrics["total_non_evaluable_ticker_count"] == 1
    assert result.metrics["total_missing_price_evidence_count"] == 1
    assert result.metrics["total_polygon_fallback_count"] == 0
    assert result.metrics["total_fired_m4_signal_count"] == 0
    assert db_session.query(SignalRegistry).count() == 0
    assert db_session.query(FeatureSnapshot).count() == 0
    date_metrics = result.metrics["date_results"][0]
    assert date_metrics["non_evaluable_ticker_count"] == 1
    assert date_metrics["missing_price_evidence_count"] == 1
    assert date_metrics["fetch_errors"][0]["error_type"] == "missing_price_evidence"
    assert [
        attempt["status"]
        for attempt in date_metrics["fetch_errors"][0]["source_attempts"]
    ] == [
        "no_usable_bars",
        "no_usable_bars",
    ]


def test_replay_progress_and_compact_bar_lineage_payload(db_session):
    _seed_reconstruction(db_session, "PROG")
    events = []
    adapter = _FakeFmpAdapter({"PROG": _bars(evidence_close=10.1, prior_close=10.0)})

    result = _run_replay(
        db_session,
        adapter,
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    assert result.status == "finished"
    date_metrics = result.metrics["date_results"][0]
    event_names = [event for event, _payload in events]
    assert "included_universe_load_start" in event_names
    assert "ticker_fetch_progress" in event_names
    assert "ticker_fetch_finish" in event_names
    assert "assembly_finish" in event_names
    assert "detector_finish" in event_names
    assert "persistence_metadata_finish" in event_names
    assert date_metrics["stage_timing_seconds"]["ticker_fetch_seconds"] >= 0
    assert date_metrics["progress_events"]

    lineage = (
        db_session.query(DataLineage)
        .filter(DataLineage.provider == "FMP")
        .filter(DataLineage.endpoint == HISTORICAL_PRICE_FULL_ENDPOINT)
        .one()
    )
    payload = json.loads(lineage.raw_payload_json)
    assert payload["payload_policy"] == "compact_bar_digest"
    assert payload["bar_count"] == 253
    assert payload["bars_digest"]
    assert "bars" not in payload


def test_replay_provenance_stamps_fired_and_no_fire_feature_snapshots(db_session):
    _seed_reconstruction(db_session, "FIRED")
    _seed_reconstruction(db_session, "NOFIRE")
    adapter = _FakeFmpAdapter(
        {
            "FIRED": _bars(evidence_close=10.1, prior_close=10.0),
            "NOFIRE": _bars(evidence_close=9.99, prior_close=10.0),
        }
    )

    first = _run_replay(db_session, adapter)
    second = _run_replay(db_session, adapter)

    assert first.status == "finished"
    assert first.metrics["total_fired_m4_signal_count"] == 1
    assert first.metrics["total_rows_inserted"] == 1
    assert first.metrics["date_results"][0]["stamped_fired_feature_count"] == 1
    assert first.metrics["date_results"][0]["stamped_no_fire_feature_count"] == 1
    assert second.status == "finished"
    assert second.metrics["total_fired_m4_signal_count"] == 0
    assert second.metrics["total_rows_inserted"] == 0
    assert second.metrics["total_rows_reused"] == 1
    assert second.metrics["date_results"][0]["stamped_fired_feature_count"] == 1
    assert second.metrics["date_results"][0]["stamped_no_fire_feature_count"] == 1

    fired_features = _feature(db_session, "FIRED")
    no_fire_features = _feature(db_session, "NOFIRE")
    _assert_replay_stamped(fired_features)
    _assert_replay_stamped(no_fire_features)
    assert fired_features["signal_generated"] is True
    assert no_fire_features["signal_generated"] is False
    assert no_fire_features["rejection_reason"] == "below_high"

    duplicate_signal_groups = (
        db_session.query(
            SignalRegistry.pattern_id,
            SignalRegistry.ticker,
            SignalRegistry.signal_identity_hash,
            func.count().label("row_count"),
        )
        .group_by(
            SignalRegistry.pattern_id,
            SignalRegistry.ticker,
            SignalRegistry.signal_identity_hash,
        )
        .having(func.count() > 1)
        .count()
    )
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
    assert duplicate_signal_groups == 0
    assert duplicate_feature_groups == 0


def test_idempotent_rerun_prevents_duplicate_signals(db_session):
    _seed_reconstruction(db_session, "IDEMP")
    adapter = _FakeFmpAdapter({"IDEMP": _bars(evidence_close=10.1, prior_close=10.0)})

    first = _run_replay(db_session, adapter)
    second = _run_replay(db_session, adapter)

    assert first.metrics["total_fired_m4_signal_count"] == 1
    assert second.metrics["total_fired_m4_signal_count"] == 0
    assert second.metrics["total_rows_reused"] == 1
    duplicate_groups = (
        db_session.query(
            SignalRegistry.pattern_id,
            SignalRegistry.ticker,
            SignalRegistry.signal_identity_hash,
            func.count().label("row_count"),
        )
        .group_by(
            SignalRegistry.pattern_id,
            SignalRegistry.ticker,
            SignalRegistry.signal_identity_hash,
        )
        .having(func.count() > 1)
        .count()
    )
    assert duplicate_groups == 0


def test_partial_universe_fails_without_explicit_allow(db_session):
    _seed_reconstruction(db_session, "PART", partial=True)
    adapter = _FakeFmpAdapter({"PART": _bars(evidence_close=10.1, prior_close=10.0)})

    blocked = _run_replay(db_session, adapter)
    allowed = _run_replay(db_session, adapter, allow_partial_universe=True)

    assert blocked.status == "failed"
    assert blocked.errors[0]["error_type"] == "partial_historical_universe"
    assert allowed.status == "finished"
    features = _feature(db_session, "PART")
    assert features["historical_replay"]["partial_universe_reason"] == "max_pages_reached"


def test_cli_refuses_public_schema():
    assert replay_cli_main(
        ["--live", "--schema", "public", "--replay-date", REPLAY_DAY.isoformat()]
    ) == 1


def test_cli_requires_schema():
    with pytest.raises(SystemExit):
        replay_cli_main(["--live", "--replay-date", REPLAY_DAY.isoformat()])
