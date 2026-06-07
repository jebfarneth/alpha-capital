from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import func

from alpha.data.contracts import AdapterResponse, LineageMeta, stable_hash
from alpha.data.fmp import FmpBar, HISTORICAL_PRICE_FULL_ENDPOINT
from alpha.db.models import (
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


def _run_replay(db_session, adapter, *, allow_partial_universe=False):
    job = HistoricalM4ReplayJob(
        session=db_session,
        fmp_adapter=adapter,
        replay_dates=[REPLAY_DAY],
        run_timestamp=_ts(),
        allow_partial_universe=allow_partial_universe,
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
    assert features["bar_lineage_id"]
    assert features["bar_lineage_hash"]
    assert features["price_basis"] == "fmp_full_close_as_split_adjusted_close"
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
